import math
from pathlib import Path

import pytest

from medera.config import load_yaml
from medera.train.lora_cpt import arm_config, fast_path_status, lr_at

ROOT = Path(__file__).resolve().parents[1]
TRAIN = load_yaml(ROOT / "configs" / "train.yaml")


def test_arm_config_merges_and_overrides():
    cfg = arm_config(TRAIN, "gemma_ex")
    assert cfg["arm"] == "gemma_ex" and cfg["experts"] is True and cfg["precision"] == "bf16"
    assert cfg["lora"] == {"r": 16, "alpha": 32, "dropout": 0.0}
    assert arm_config(TRAIN, "gemma_ex", {"precision": "qlora4"})["precision"] == "qlora4"
    assert arm_config(TRAIN, "gemma_ex", {"precision": None})["precision"] == "bf16"
    assert arm_config(TRAIN, "gemma_ex_q4")["precision"] == "qlora4"


def test_arm_config_rejects_bad_precision_and_unknown_arm():
    with pytest.raises(ValueError):
        arm_config(TRAIN, "qwen", {"precision": "fp16"})
    with pytest.raises(KeyError):
        arm_config(TRAIN, "llama")


def test_core_arms_share_settings_and_are_bf16():
    core = ["qwen", "gemma_al", "gemma_ex"]
    for arm in core:
        assert TRAIN["arms"][arm]["precision"] == "bf16"          # explicit, not inherited
        assert not TRAIN["arms"][arm].get("stretch")
    cfgs = [arm_config(TRAIN, a) for a in core]
    for key in ("lora", "lr", "block_len", "micro_batch", "grad_accum", "seed", "weight_decay", "warmup_ratio"):
        assert len({repr(c[key]) for c in cfgs}) == 1, key
    al, ex = arm_config(TRAIN, "gemma_al"), arm_config(TRAIN, "gemma_ex")
    assert {k for k in al if al[k] != ex[k]} == {"arm", "experts"}
    assert all(TRAIN["arms"][a]["stretch"] for a in TRAIN["arms"] if a.endswith("_q4"))


def test_lr_schedule_warmup_then_cosine():
    total, warmup = 100, 3
    assert lr_at(0, total, warmup) == pytest.approx(1 / 3)
    assert lr_at(2, total, warmup) == pytest.approx(1.0)
    assert lr_at(3, total, warmup) == pytest.approx(1.0)
    mid = warmup + (total - warmup) // 2
    assert lr_at(mid, total, warmup) == pytest.approx(0.5 * (1 + math.cos(math.pi * (mid - warmup) / (total - warmup))))
    assert lr_at(total, total, warmup) == pytest.approx(0.0, abs=1e-12)


def test_fast_path_detector():
    torch = pytest.importorskip("torch")

    def layer(module_name):
        m = torch.nn.Module()
        fn = lambda: None  # noqa: E731
        fn.__module__ = module_name
        m.chunk_gated_delta_rule = fn
        return m

    fast, slow = torch.nn.Module(), torch.nn.Module()
    fast.a, fast.b = layer("fla.ops.gated_delta_rule.chunk"), layer("fla.ops.gated_delta_rule.chunk")
    slow.a, slow.b = layer("fla.ops.gated_delta_rule.chunk"), layer("transformers.models.qwen3_5.modeling_qwen3_5")
    assert fast_path_status(fast) is True
    assert fast_path_status(slow) is False
    assert fast_path_status(torch.nn.Linear(2, 2)) is None
