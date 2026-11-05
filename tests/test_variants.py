"""The 11 evaluation variants: names, order (shared with the bash dispatcher) and adapter provenance."""
import json
import re
from pathlib import Path

import pytest

from medera.config import load_yaml
from medera.variants import adapter_info, core_variants, get_variant

REPO = Path(__file__).resolve().parents[1]
TRAIN = load_yaml(REPO / "configs" / "train.yaml")


def test_eleven_core_variants_bases_first_then_windows_interleaved():
    names = [v.name for v in core_variants(TRAIN)]
    assert names == ["qwen_base", "gemma_base", "qwen_w1", "gemma_al_w1", "gemma_ex_w1", "qwen_w2", "gemma_al_w2",
                     "gemma_ex_w2", "qwen_w3", "gemma_al_w3", "gemma_ex_w3"]
    assert {v.model for v in core_variants(TRAIN) if v.arm in ("gemma_al", "gemma_ex")} == {"gemma4_26b_a4b"}
    assert get_variant(TRAIN, "gemma_base").is_base and get_variant(TRAIN, "gemma_ex_w2").adapter == "gemma_ex_w2"
    assert not any("q4" in n for n in names)                               # stretch arms are not core variants


def test_bash_dispatcher_uses_the_same_order():
    lib = (REPO / "hpc" / "_dispatch_lib.sh").read_text()
    order = re.search(r'^VARIANTS_ORDER="([^"]+)"', lib, re.M).group(1).split()
    assert order == [v.name for v in core_variants(TRAIN)]


def test_unknown_variant_is_an_error():
    with pytest.raises(KeyError, match="unknown variant"):
        get_variant(TRAIN, "qwen_q4_w1")


def test_adapter_info_needs_a_finished_adapter(tmp_path):
    v = get_variant(TRAIN, "qwen_w1")
    d = tmp_path / "adapters" / "qwen_w1"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"weights")
    with pytest.raises(FileNotFoundError, match="not finished"):
        adapter_info(tmp_path, v)
    (d / "DONE").write_text(json.dumps({"steps": 196, "git_sha": "abc"}))
    info = adapter_info(tmp_path, v)
    assert info["adapter"] == "qwen_w1" and len(info["adapter_sha256"]) == 64 and info["adapter_done"]["steps"] == 196
    assert adapter_info(tmp_path, get_variant(TRAIN, "qwen_base")) == {}
