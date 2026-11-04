"""End to end on CPU with tiny random versions of the real architectures (no download):
apply_lora -> train -> audit -> check, for Gemma-EX, Gemma-AL and Qwen, plus checkpoint resume."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from medera.config import load_yaml  # noqa: E402
from medera.train.audit_adapter import audit, check  # noqa: E402
from medera.train.lora_cpt import apply_lora, arm_config, fast_path_status, train  # noqa: E402

TRAIN = load_yaml("configs/train.yaml")
MODELS = load_yaml("configs/models.yaml")
VOCAB = 256


def _tiny_gemma4():
    cfg_cls = getattr(transformers, "Gemma4Config", None)
    if cfg_cls is None:
        pytest.skip("transformers has no Gemma4")
    cfg = cfg_cls(
        text_config=dict(num_hidden_layers=2, hidden_size=64, intermediate_size=64, vocab_size=VOCAB,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=32,
                         num_global_key_value_heads=1, enable_moe_block=True, num_experts=4, top_k_experts=2,
                         moe_intermediate_size=32, vocab_size_per_layer_input=VOCAB, hidden_size_per_layer_input=0,
                         layer_types=["sliding_attention", "full_attention"], sliding_window=16),
        vision_config=dict(num_hidden_layers=1, hidden_size=32, intermediate_size=32, num_attention_heads=2,
                           num_key_value_heads=2, head_dim=16, global_head_dim=16))
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(cfg)


def _tiny_qwen35():
    cfg_cls = getattr(transformers, "Qwen3_5TextConfig", None)
    if cfg_cls is None:
        pytest.skip("transformers has no Qwen3.5")
    cfg = cfg_cls(num_hidden_layers=2, hidden_size=64, intermediate_size=64, vocab_size=VOCAB,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                  layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
                  linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16)
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(cfg)


def _cfg(arm, **over):
    cfg = arm_config(TRAIN, arm)
    cfg.update(grad_accum=1, micro_batch=2, log_every=1, **over)
    return cfg


def _blocks(n=12, length=16):
    return np.random.default_rng(0).integers(0, VOCAB, size=(n, length), dtype=np.uint32)


def test_gemma_ex_adapts_experts_with_nonzero_grads():
    cfg = _cfg("gemma_ex")
    model = apply_lora(_tiny_gemma4(), cfg, MODELS["gemma4_26b_a4b"])
    res = train(model, _blocks(), cfg, max_steps=3, log=lambda m: None)
    report = audit(model, res["grad_norms"], "gemma_ex", True)
    assert res["step"] == 3 and all(np.isfinite(res["losses"]))
    assert report["groups"]["experts"]["lora_tensors"] == 8        # 2 layers x 2 fused params x (A, B)
    assert report["groups"]["experts"]["lora_nonzero_grad"] == 8
    assert report["groups"]["vision"]["lora_tensors"] == 0
    assert check(report) == []


def test_gemma_al_has_no_expert_or_vision_adapters():
    cfg = _cfg("gemma_al")
    model = apply_lora(_tiny_gemma4(), cfg, MODELS["gemma4_26b_a4b"])
    res = train(model, _blocks(), cfg, max_steps=2, log=lambda m: None)
    report = audit(model, res["grad_norms"], "gemma_al", False)
    assert report["groups"]["experts"]["lora_tensors"] == 0
    assert report["groups"]["vision"]["lora_tensors"] == 0
    assert report["groups"]["router"]["lora_tensors"] == 4          # router.proj is an nn.Linear (all-linear)
    assert check(report) == []


def test_qwen_dense_adapts_linear_attention():
    cfg = _cfg("qwen")
    model = apply_lora(_tiny_qwen35(), cfg, MODELS["qwen35_27b"])
    res = train(model, _blocks(), cfg, max_steps=2, log=lambda m: None)
    report = audit(model, res["grad_norms"], "qwen", False)
    assert report["groups"]["attention"]["lora_tensors"] > 0
    assert fast_path_status(model) is False                          # CPU: torch fallback, which the smoke rejects
    assert check(report) == []


def test_train_checkpoint_resume(tmp_path):
    cfg = _cfg("gemma_ex", save_every_steps=1)
    first = apply_lora(_tiny_gemma4(), cfg, MODELS["gemma4_26b_a4b"])
    train(first, _blocks(), cfg, max_steps=2, ckpt_dir=tmp_path, log=lambda m: None)
    assert (tmp_path / "state.pt").exists()
    logs = []
    second = apply_lora(_tiny_gemma4(), cfg, MODELS["gemma4_26b_a4b"])
    res = train(second, _blocks(), cfg, max_steps=4, ckpt_dir=tmp_path, log=logs.append)
    assert any("resumed at step" in line for line in logs)
    assert res["step"] == 4 and len(res["losses"]) == 4


def test_train_wall_guard_stops_and_saves(tmp_path):
    cfg = _cfg("gemma_al")
    model = apply_lora(_tiny_gemma4(), cfg, MODELS["gemma4_26b_a4b"])
    res = train(model, _blocks(), cfg, ckpt_dir=tmp_path, deadline=0.0, log=lambda m: None)
    assert res["stopped"] == "wall" and res["step"] == 1 and (tmp_path / "state.pt").exists()


def test_gemma_ex_adapter_save_and_reload(tmp_path):
    from peft import PeftModel

    cfg = _cfg("gemma_ex")
    model = apply_lora(_tiny_gemma4(), cfg, MODELS["gemma4_26b_a4b"])
    train(model, _blocks(), cfg, max_steps=2, log=lambda m: None)
    model.save_pretrained(tmp_path / "adapter")
    x = torch.from_numpy(_blocks(1).astype(np.int64))
    model.eval()
    with torch.no_grad():
        want = model(input_ids=x).logits
        reloaded = PeftModel.from_pretrained(_tiny_gemma4(), tmp_path / "adapter").eval()
        got = reloaded(input_ids=x).logits
    assert torch.allclose(want, got, atol=1e-5)
