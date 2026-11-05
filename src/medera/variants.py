"""The 11 core model variants that perplexity and probing run on: 2 base checkpoints + 9 trained adapters.

    qwen_base, gemma_base                  pinned base checkpoints (configs/models.yaml), no adapter
    <arm>_<window>                         base + the LoRA adapter in runs/adapters/<arm>_<window>/ (DONE marker)

Gemma-AL and Gemma-EX share gemma_base. The order here is the evaluation order (bases first, then each window's arms
interleaved, like training) and must match VARIANTS_ORDER in hpc/_dispatch_lib.sh (a test checks it).
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

WINDOWS = ("w1", "w2", "w3")
BASE_NAMES = {"qwen35_27b": "qwen_base", "gemma4_26b_a4b": "gemma_base"}


@dataclass(frozen=True)
class Variant:
    name: str
    model: str                 # key in configs/models.yaml
    arm: str | None = None     # None for a base checkpoint
    window: str | None = None

    @property
    def is_base(self) -> bool:
        return self.arm is None

    @property
    def adapter(self) -> str | None:
        return None if self.is_base else f"{self.arm}_{self.window}"


def core_variants(train_cfg: dict) -> list[Variant]:
    arms = {a: c for a, c in train_cfg["arms"].items() if not c.get("stretch")}
    models = list(dict.fromkeys(c["model"] for c in arms.values()))
    out = [Variant(BASE_NAMES[m], m) for m in models]
    out += [Variant(f"{a}_{w}", c["model"], a, w) for w in WINDOWS for a, c in arms.items()]
    return out


def get_variant(train_cfg: dict, name: str) -> Variant:
    for v in core_variants(train_cfg):
        if v.name == name:
            return v
    raise KeyError(f"unknown variant {name!r}; choose from {[v.name for v in core_variants(train_cfg)]}")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def adapter_info(runs: Path, v: Variant) -> dict:
    """Provenance of a trained adapter (its DONE marker and weights hash); {} for a base. Raises if not trained."""
    if v.is_base:
        return {}
    d = runs / "adapters" / v.adapter
    done, weights = d / "DONE", d / "adapter_model.safetensors"
    if not (done.exists() and (d / "adapter_config.json").exists() and weights.exists()):
        raise FileNotFoundError(f"adapter {d} is not finished (needs DONE, adapter_config.json, "
                                "adapter_model.safetensors)")
    return {"adapter": v.adapter, "adapter_sha256": sha256_file(weights), "adapter_done": json.loads(done.read_text())}


def load_variant(v: Variant, model_cfg: dict, runs: Path, base_loader=None):
    """The base checkpoint in bf16 on GPU 0 (lora_cpt.load_model: pinned revision, fails on missing weights), with the
    variant's adapter attached unmerged, in eval mode, so the function is exactly the one that was trained.
    `base_loader(model_cfg)` replaces the real loader in tests."""
    if base_loader is None:
        from medera.train.lora_cpt import load_model

        def base_loader(mc):
            return load_model(mc, "bf16")
    model = base_loader(model_cfg)
    if not v.is_base:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, runs / "adapters" / v.adapter, is_trainable=False)
    model.eval()
    return model
