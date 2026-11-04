"""Audit logic on a tiny fake MoE model with a vision tower, adapted with real PEFT."""
import json

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
nn = torch.nn

from medera.train import audit_adapter  # noqa: E402
from medera.train.audit_adapter import audit, check  # noqa: E402
from medera.train.targets import expert_targets, linear_targets  # noqa: E402


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(4, 16, 8) * 0.1)   # (E, out, in) like Gemma-4
        self.down_proj = nn.Parameter(torch.randn(4, 8, 8) * 0.1)

    def forward(self, x):
        h = torch.einsum("ni,eoi->neo", x, self.gate_up_proj)
        h = torch.nn.functional.silu(h[..., :8]) * h[..., 8:]
        return torch.einsum("nei,eoi->no", h, self.down_proj) / 4


class _Fake(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(8, 8)
        layer.router = nn.Module()
        layer.router.proj = nn.Linear(8, 4)
        layer.experts = _Experts()
        self.language_model.layers = nn.ModuleList([layer])
        self.vision_tower = nn.Module()
        self.vision_tower.q_proj = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 10)

    def forward(self, x):
        layer = self.language_model.layers[0]
        h = layer.self_attn.q_proj(x)
        h = h + layer.experts(h) + 0.01 * layer.router.proj(h).sum(-1, keepdim=True)
        h = h + 0.01 * self.vision_tower.q_proj(x)
        return self.lm_head(h)


def _adapted(experts: bool, vision: bool = False):
    torch.manual_seed(0)
    model = _Fake()
    targets = linear_targets(model) + (["vision_tower.q_proj"] if vision else [])
    params = expert_targets(model, ["experts.gate_up_proj", "experts.down_proj"]) if experts else None
    cfg = peft.LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=targets, target_parameters=params)
    return peft.inject_adapter_in_model(cfg, model)


def _train(model, steps: int) -> dict[str, float]:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    norms: dict[str, float] = {}
    for _ in range(steps):
        model(torch.randn(5, 8)).pow(2).mean().backward()
        for n, p in model.named_parameters():
            if p.requires_grad:
                g = 0.0 if p.grad is None else p.grad.norm().item()
                norms[n] = max(norms.get(n, 0.0), g)
        opt.step()
        opt.zero_grad()
    return norms


def test_audit_counts_groups():
    model = _adapted(experts=True)
    groups = audit(model, _train(model, 2), "gemma_ex", True)["groups"]
    assert groups["attention"]["lora_tensors"] == 2       # q_proj A, B
    assert groups["router"]["lora_tensors"] == 2          # router.proj A, B
    assert groups["experts"]["lora_tensors"] == 4         # gate_up_proj and down_proj, A and B
    assert groups["vision"]["lora_tensors"] == 0
    assert groups["experts"]["base_params"] == 4 * 16 * 8 + 4 * 8 * 8
    assert groups["experts"]["base_dtypes"] == {"float32": 4 * 16 * 8 + 4 * 8 * 8}


def test_check_passes_expert_arm_after_two_steps():
    model = _adapted(experts=True)
    report = audit(model, _train(model, 2), "gemma_ex", True)
    assert report["groups"]["experts"]["lora_nonzero_grad"] == 4   # B moved after step 1, so A gets grads at step 2
    assert check(report) == []


def test_check_fails_expert_arm_without_expert_grads():
    model = _adapted(experts=True)
    assert "expert LoRA tensors never received a nonzero gradient" in check(audit(model, {}, "gemma_ex", True))
    plain = _adapted(experts=False)
    assert "expert arm adapts 0 expert tensors" in check(audit(plain, _train(plain, 2), "gemma_ex", True))


def test_check_fails_on_vision_lora():
    model = _adapted(experts=False, vision=True)
    assert "2 vision tensors adapted" in check(audit(model, _train(model, 2), "gemma_al", False))


def test_check_fails_non_expert_arm_with_experts():
    model = _adapted(experts=True)
    assert "non-expert arm adapts 4 expert tensors" in check(audit(model, _train(model, 2), "gemma_al", False))


def test_cli_rechecks_saved_reports(tmp_path, capsys):
    model = _adapted(experts=True)
    good = tmp_path / "good.json"
    good.write_text(json.dumps(audit(model, _train(model, 2), "gemma_ex", True)))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(audit(model, {}, "gemma_ex", True)))
    assert audit_adapter.main([str(good)]) == 0
    assert audit_adapter.main([str(good), str(bad)]) == 1
    assert "FAIL" in capsys.readouterr().out
