"""Which tensors get LoRA adapters, and which audit group each parameter belongs to.

Both base checkpoints are multimodal. Only text weights are adapted: the vision (and any audio)
tower is excluded by name, and so is lm_head. Gemma-4's experts are fused 3-D nn.Parameters,
not nn.Linear, so `all-linear` never reaches them; expert arms list them via PEFT
`target_parameters` instead.
"""
import torch.nn as nn

GROUPS = ("attention", "dense_mlp", "router", "experts", "vision", "other")
_NON_TEXT = ("vision", "visual", "audio", "multi_modal")
_ATTENTION = {"self_attn", "linear_attn", "attn", "attention", "cross_attn"}


def param_group(name: str) -> str:
    """Audit group for a module or parameter name, by dotted segment, in precedence order."""
    segs = name.split(".")
    if any(key in seg for seg in segs for key in _NON_TEXT):
        return "vision"
    if "experts" in segs:
        return "experts"
    if "router" in segs or "gate" in segs:
        return "router"
    if any(seg in _ATTENTION for seg in segs):
        return "attention"
    if "mlp" in segs:
        return "dense_mlp"
    return "other"


def linear_targets(model: nn.Module) -> list[str]:
    """Full names of every text-model Linear layer (bnb Linear4bit subclasses nn.Linear), minus lm_head."""
    return [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and param_group(name) != "vision" and name.split(".")[-1] != "lm_head"
    ]


def expert_targets(model: nn.Module, suffixes: list[str]) -> list[str]:
    """Full names of the fused expert parameters ending in one of `suffixes` (e.g. experts.gate_up_proj)."""
    return [
        name for name, _ in model.named_parameters()
        if any(name == s or name.endswith("." + s) for s in suffixes) and param_group(name) != "vision"
    ]
