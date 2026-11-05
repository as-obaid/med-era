"""Held-out perplexity (T3) on CPU with tiny random Qwen3.5/Gemma-4 models: the NLL, adapters really applied,
resumable cells, the deadline stop, the CLI wiring and the gate table."""
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from test_tiny_models import MODELS, TRAIN, VOCAB, _tiny_gemma4, _tiny_qwen35  # noqa: E402

from medera.jobs import Stopper  # noqa: E402
from medera.measure import perplexity as ppl  # noqa: E402
from medera.train.lora_cpt import apply_lora, arm_config  # noqa: E402
from medera.variants import WINDOWS, core_variants, get_variant, load_variant  # noqa: E402


def _blocks(n=3, length=16, seed=0):
    return np.random.default_rng(seed).integers(3, VOCAB, size=(n, length), dtype=np.uint32)


class CountingStopper:
    """fits() is true for the first n units of work, then false (a job hitting its deadline)."""

    def __init__(self, n):
        self.n = n

    def fits(self, est_s):
        self.n -= 1
        return self.n >= 0

    def minutes_left(self):
        return 0.0


def _save_adapter(runs, arm, window, builder):
    """A 'trained' adapter: LoRA as training applies it, with random lora_B so it changes the function."""
    cfg = arm_config(TRAIN, arm)
    model = apply_lora(builder(), cfg, MODELS[cfg["model"]])
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
    d = runs / "adapters" / f"{arm}_{window}"
    model.save_pretrained(d)
    (d / "DONE").write_text(json.dumps({"steps": 1}) + "\n")
    return model.eval()


def test_block_nll_is_the_training_loss_summed():
    model = _tiny_qwen35().eval()
    x = torch.from_numpy(_blocks(1).astype(np.int64))
    with torch.no_grad():
        want = model(input_ids=x, labels=x).loss.item() * (x.shape[1] - 1)
    assert ppl.block_nll(model, x) == pytest.approx(want, rel=1e-5)


@pytest.mark.parametrize("arm,builder,base", [("gemma_ex", _tiny_gemma4, "gemma_base"),
                                              ("gemma_al", _tiny_gemma4, "gemma_base"),
                                              ("qwen", _tiny_qwen35, "qwen_base")])
def test_loaded_adapter_is_the_trained_function_and_differs_from_base(tmp_path, arm, builder, base):
    trained = _save_adapter(tmp_path, arm, "w1", builder)
    mc = MODELS[arm_config(TRAIN, arm)["model"]]
    loaded = load_variant(get_variant(TRAIN, f"{arm}_w1"), mc, tmp_path, base_loader=lambda _: builder())
    plain = load_variant(get_variant(TRAIN, base), mc, tmp_path, base_loader=lambda _: builder())
    x = torch.from_numpy(_blocks(1).astype(np.int64))
    assert not loaded.training
    assert ppl.block_nll(loaded, x) == pytest.approx(ppl.block_nll(trained, x), rel=1e-6)
    assert abs(ppl.block_nll(loaded, x) - ppl.block_nll(plain, x)) > 1e-3


def _quiet(msg):
    pass


def _heldout():
    return {e: (_blocks(2 + i, seed=i), f"sha-{e}") for i, e in enumerate(WINDOWS)}


def test_cells_resume_after_a_stop_and_skip_when_current(tmp_path):
    v, loads = get_variant(TRAIN, "qwen_base"), []

    def loader():
        loads.append(1)
        return _tiny_qwen35().eval()

    meta, prov = {"adapter_sha256": None, "model_revision": "r"}, {"git_sha": "abc", "job_id": "1"}
    held = _heldout()
    prog = tmp_path / "runs" / "progress.json"
    assert ppl.run_variant(v, loader, held, tmp_path, CountingStopper(1), prov, meta, prog, log=_quiet) == "wall"
    cells = sorted(p.name for p in (tmp_path / "scores" / "perplexity" / "qwen_base").iterdir())
    assert cells == ["w1.json"] and json.loads(prog.read_text())["step"] == 1
    assert ppl.run_variant(v, loader, held, tmp_path, CountingStopper(9), prov, meta, prog, log=_quiet) == "done"
    d = tmp_path / "scores" / "perplexity" / "qwen_base"
    assert sorted(p.name for p in d.iterdir()) == ["DONE", "w1.json", "w2.json", "w3.json"] and len(loads) == 2
    c = json.loads((d / "w3.json").read_text())
    assert c["n_blocks"] == 4 and c["n_tokens"] == 4 * 15 and len(c["block_nll"]) == 4
    assert c["ppl"] == pytest.approx(math.exp(math.fsum(c["block_nll"]) / c["n_tokens"]))
    assert c["heldout_sha256"] == "sha-w3" and c["provenance"]["git_sha"] == "abc" and c["model_revision"] == "r"
    # current cells are never recomputed (no model load); a changed held-out file is
    assert ppl.run_variant(v, loader, held, tmp_path, CountingStopper(9), prov, meta, None, log=_quiet) == "done"
    assert len(loads) == 2
    held["w2"] = (held["w2"][0], "sha-new")
    ppl.run_variant(v, loader, held, tmp_path, CountingStopper(9), prov, meta, None, log=lambda m: None)
    assert len(loads) == 3 and json.loads((d / "w2.json").read_text())["heldout_sha256"] == "sha-new"


def test_stopper_uses_the_slurm_end_time_and_sigterm():
    t0 = 1000.0
    s = Stopper(t0, {"wall_guard_hours": 1.0, "wall_margin_s": 300}, env={"MEDERA_JOB_END_EPOCH": "2000"},
                clock=lambda: t0)
    assert s.deadline == 1700.0 and s.fits(600) and not s.fits(701)
    s.signal = 15
    assert not s.fits(0)


def test_main_variant_writes_three_cells(tmp_path, monkeypatch):
    hd = tmp_path / "data" / "processed" / "hrt" / "qwen35_27b"
    hd.mkdir(parents=True)
    for i, e in enumerate(WINDOWS):
        np.save(hd / f"{e}_heldout.npy", _blocks(2, seed=i))
    monkeypatch.setattr(ppl, "load_variant", lambda v, mc, runs: _tiny_qwen35().eval())
    args = ["--data-root", str(tmp_path / "data"), "--runs-dir", str(tmp_path / "runs"),
            "--results-dir", str(tmp_path / "results")]
    assert ppl.main(["--variant", "qwen_base", *args]) == 0
    d = tmp_path / "results" / "scores" / "perplexity" / "qwen_base"
    assert (d / "DONE").exists() and json.loads((d / "w1.json").read_text())["model_hf_id"] == "Qwen/Qwen3.5-27B"
    with pytest.raises(FileNotFoundError):                    # an adapter needs runs/adapters/<a>/DONE
        ppl.main(["--variant", "qwen_w1", *args])


def _cell(results, variant, era, value):
    p = ppl.cell_path(results, variant, era)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"variant": variant, "era": era, "n_blocks": 1, "n_tokens": 10,
                             "nll_sum": 10 * math.log(value), "mean_nll": math.log(value), "ppl": value}))


def test_table_matrix_and_gate(tmp_path):
    for v in core_variants(TRAIN):
        for e in WINDOWS:
            base = 10.0 if v.model == "qwen35_27b" else 20.0
            val = base if v.is_base else base * (0.9 if v.window == e else 0.97)
            _cell(tmp_path, v.name, e, val)
    _cell(tmp_path, "gemma_ex_w3", "w3", 21.0)            # one adapter worse than its base on its own era
    out = ppl.write_table(tmp_path, TRAIN)
    gates = {g["adapter"]: g for g in out["gates"]}
    assert out["cells"] == 33 and len(gates) == 9
    assert gates["qwen_w1"]["gate"] == "PASS" and gates["qwen_w1"]["delta_pct"] == pytest.approx(-10.0)
    assert gates["qwen_w1"]["lowest_of_arm_on_own_era"] is True
    assert gates["gemma_ex_w3"]["gate"] == "FAIL" and gates["gemma_ex_w3"]["lowest_of_arm_on_own_era"] is False
    md = (tmp_path / "tables" / "T3_perplexity.md").read_text()
    assert "| qwen_w2 | qwen35_27b | 9.700 | 9.000 | 9.700 |" in md and "| gemma_ex_w3 | gemma_base |" in md
    assert len((tmp_path / "tables" / "T3_perplexity.csv").read_text().splitlines()) == 34
    assert ppl.main(["--table", "--results-dir", str(tmp_path)]) == 0


def test_table_with_missing_cells_marks_gate_pending(tmp_path):
    _cell(tmp_path, "qwen_w1", "w1", 9.0)
    gates = {g["adapter"]: g for g in ppl.write_table(tmp_path, TRAIN)["gates"]}
    assert gates["qwen_w1"]["gate"] == "pending" and gates["qwen_w1"]["lowest_of_arm_on_own_era"] is None
