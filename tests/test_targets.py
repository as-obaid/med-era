import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from medera.train.targets import expert_targets, linear_targets, param_group  # noqa: E402


@pytest.mark.parametrize("name,group", [
    # Qwen3.5 (text-only causal LM)
    ("model.layers.0.linear_attn.in_proj_qkv", "attention"),
    ("model.layers.3.self_attn.q_proj", "attention"),
    ("model.layers.0.mlp.down_proj", "dense_mlp"),
    ("model.embed_tokens.weight", "other"),
    ("lm_head", "other"),
    # Gemma-4 (multimodal)
    ("model.language_model.layers.0.experts.gate_up_proj", "experts"),
    ("model.language_model.layers.0.router.proj", "router"),
    ("model.language_model.layers.0.mlp.gate_proj", "dense_mlp"),
    ("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear", "vision"),
    ("model.embed_vision.embedding_projection", "vision"),
    # PEFT-wrapped names
    ("base_model.model.model.language_model.layers.0.experts.base_layer.lora_A.default.weight", "experts"),
    ("base_model.model.model.language_model.layers.1.self_attn.o_proj.lora_B.default.weight", "attention"),
    ("base_model.model.model.visual.blocks.0.attn.qkv.lora_A.default.weight", "vision"),
])
def test_param_group_real_names(name, group):
    assert param_group(name) == group


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(4, 16, 8))
        self.down_proj = nn.Parameter(torch.zeros(4, 8, 8))


def _fake_multimodal():
    m = nn.Module()
    m.model = nn.Module()
    m.model.language_model = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(8, 8)
    layer.mlp = nn.Module()
    layer.mlp.up_proj = nn.Linear(8, 8)
    layer.router = nn.Module()
    layer.router.proj = nn.Linear(8, 4)
    layer.experts = _Experts()
    m.model.language_model.layers = nn.ModuleList([layer])
    m.model.vision_tower = nn.Module()
    m.model.vision_tower.q_proj = nn.Module()
    m.model.vision_tower.q_proj.linear = nn.Linear(8, 8)
    m.model.vision_tower.experts = _Experts()  # must never be picked up
    m.model.embed_vision = nn.Module()
    m.model.embed_vision.embedding_projection = nn.Linear(8, 8)
    m.lm_head = nn.Linear(8, 10)
    return m


def test_linear_targets_exclude_vision_and_lm_head():
    assert linear_targets(_fake_multimodal()) == [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.mlp.up_proj",
        "model.language_model.layers.0.router.proj",
    ]


def test_expert_targets_finds_fused_params():
    assert expert_targets(_fake_multimodal(), ["experts.gate_up_proj", "experts.down_proj"]) == [
        "model.language_model.layers.0.experts.gate_up_proj",
        "model.language_model.layers.0.experts.down_proj",
    ]
    assert expert_targets(_fake_multimodal(), []) == []
