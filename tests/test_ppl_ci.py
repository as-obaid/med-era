"""T3 on CPU from synthetic per-block cells: gate CIs, all 27 cells, era specificity, held-out sha checks, the figure.
No torch needed (the GPU path is tested in test_perplexity.py)."""
import csv
import json
import math

import numpy as np
import pytest

from medera.config import load_yaml
from medera.measure import perplexity as ppl
from medera.variants import WINDOWS, core_variants

N_BOOT = 400
TOK = 15                                       # predicted tokens per block (block_len 16)


@pytest.fixture
def train_cfg(repo_root):
    return load_yaml(repo_root / "configs" / "train.yaml")


def _write(results, variant, era, nll, sha):
    nll = [float(x) for x in nll]
    n_tok = len(nll) * TOK
    c = {"variant": variant, "era": era, "n_blocks": len(nll), "block_len": TOK + 1, "n_tokens": n_tok,
         "nll_sum": math.fsum(nll), "mean_nll": math.fsum(nll) / n_tok, "ppl": math.exp(math.fsum(nll) / n_tok),
         "heldout_sha256": sha, "block_nll": nll}
    p = ppl.cell_path(results, variant, era)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(c))
    return c


def _shift(v, era):
    """Per-token NLL shift vs base: own era -0.03, other eras -0.01; gemma_ex_w1 on w3 is worse than base (+0.005)."""
    if v.is_base:
        return 0.0
    if v.name == "gemma_ex_w1" and era == "w3":
        return 0.005
    return -0.03 if v.window == era else -0.01


def _all_cells(results, train_cfg, n_blocks=12):
    rng = np.random.default_rng(0)
    level = {(m, e): rng.uniform(1.2, 2.0, n_blocks) for m in ("qwen35_27b", "gemma4_26b_a4b") for e in WINDOWS}
    for i, v in enumerate(core_variants(train_cfg)):
        for j, e in enumerate(WINDOWS):
            noise = np.random.default_rng(100 + 3 * i + j).normal(0, 1e-4, n_blocks)
            _write(results, v.name, e, (level[(v.model, e)] + _shift(v, e) + noise) * TOK, f"{v.model}-{e}")


def _csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_tables_with_cis_cells_and_specificity(tmp_path, train_cfg):
    _all_cells(tmp_path, train_cfg)
    out = ppl.write_table(tmp_path, train_cfg, n_boot=N_BOOT)
    gates = {g["adapter"]: g for g in out["gates"]}
    assert len(gates) == 9 and all(g["gate"] == "PASS" for g in gates.values())
    g = gates["qwen_w1"]
    assert g["delta_pct"] == pytest.approx(100 * math.expm1(-0.03), abs=0.05)
    assert g["ci_lo_pct"] < g["delta_pct"] < g["ci_hi_pct"] < 0
    t = tmp_path / "tables"
    gate_csv = _csv(t / "T3_perplexity_gate.csv")
    assert list(gate_csv[0])[:9] == ["adapter", "arm", "window", "base", "ppl_base", "ppl_adapter", "delta_pct",
                                     "ci_lo_pct", "ci_hi_pct"] and gate_csv[0]["gate"] == "PASS"
    cells = _csv(t / "T3_perplexity_cells.csv")
    assert len(cells) == 27 and sum(r["own_era"] == "True" for r in cells) == 9
    bad = next(r for r in cells if r["adapter"] == "gemma_ex_w1" and r["era"] == "w3")
    assert float(bad["delta_pct"]) > 0 and float(bad["ci_lo_pct"]) > 0
    spec = _csv(t / "T3_perplexity_specificity.csv")
    assert len(spec) == 18 and all(r["ci_vs_0"] == "below 0" for r in spec)
    one = next(r for r in spec if r["own_adapter"] == "qwen_w2" and r["other_adapter"] == "qwen_w3")
    assert float(one["diff_pct"]) == pytest.approx(100 * math.expm1(-0.02), abs=0.05)
    md = (t / "T3_perplexity.md").read_text()
    for s in ("| 95% CI |", "## Era specificity", "9 of 9 arm x era cases", "descriptive, not pre-registered",
              "w1 has only 16 held-out articles", "blocks from the same article are correlated"):
        assert s in md


def test_gate_semantics_unchanged_and_ci_on_a_fail(tmp_path, train_cfg):
    _all_cells(tmp_path, train_cfg)
    base = json.loads(ppl.cell_path(tmp_path, "gemma_base", "w3").read_text())
    _write(tmp_path, "gemma_ex_w3", "w3", np.asarray(base["block_nll"]) + 0.01 * TOK, base["heldout_sha256"])
    gates = {g["adapter"]: g for g in ppl.write_table(tmp_path, train_cfg, n_boot=N_BOOT)["gates"]}
    g = gates["gemma_ex_w3"]
    assert g["gate"] == "FAIL" and g["delta_pct"] > 0 and g["ci_lo_pct"] > 0
    assert g["lowest_of_arm_on_own_era"] is False


def test_identical_adapter_and_base_give_zero_with_zero_width_ci(tmp_path, train_cfg):
    _all_cells(tmp_path, train_cfg)
    base = json.loads(ppl.cell_path(tmp_path, "qwen_base", "w2").read_text())
    _write(tmp_path, "qwen_w2", "w2", base["block_nll"], base["heldout_sha256"])
    g = {g["adapter"]: g for g in ppl.write_table(tmp_path, train_cfg, n_boot=N_BOOT)["gates"]}["qwen_w2"]
    assert g["delta_pct"] == 0.0 and g["ci_lo_pct"] == 0.0 and g["ci_hi_pct"] == 0.0 and g["gate"] == "FAIL"


def test_different_heldout_files_are_refused(tmp_path, train_cfg):
    _all_cells(tmp_path, train_cfg)
    c = json.loads(ppl.cell_path(tmp_path, "qwen_w1", "w1").read_text())
    _write(tmp_path, "qwen_w1", "w1", c["block_nll"], "another-file")
    with pytest.raises(ValueError, match="different held-out files"):
        ppl.write_table(tmp_path, train_cfg, n_boot=N_BOOT)


def test_block_nlls_must_reproduce_the_saved_perplexity(tmp_path, train_cfg):
    _all_cells(tmp_path, train_cfg)
    p = ppl.cell_path(tmp_path, "gemma_al_w2", "w2")
    c = json.loads(p.read_text())
    c["ppl"] *= 1.001
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="per-block NLLs give perplexity"):
        ppl.write_table(tmp_path, train_cfg, n_boot=N_BOOT)


def test_cells_without_block_nll_still_tabulate_without_cis(tmp_path, train_cfg):
    for v in core_variants(train_cfg):
        for e in WINDOWS:
            p = ppl.cell_path(tmp_path, v.name, e)
            p.parent.mkdir(parents=True, exist_ok=True)
            val = 10.0 if v.is_base else 9.0
            p.write_text(json.dumps({"variant": v.name, "era": e, "n_blocks": 1, "n_tokens": 10,
                                     "nll_sum": 10 * math.log(val), "mean_nll": math.log(val), "ppl": val}))
    out = ppl.write_table(tmp_path, train_cfg, n_boot=N_BOOT)
    assert all(g["gate"] == "PASS" and g["ci_lo_pct"] is None for g in out["gates"])
    assert all(r["ci_vs_0"] is None for r in out["specificity"])


def test_figure_and_cli_write_png_and_svg(tmp_path, train_cfg):
    pytest.importorskip("matplotlib")
    _all_cells(tmp_path, train_cfg)
    paths = ppl.write_figure(tmp_path, train_cfg, n_boot=N_BOOT)
    assert [p.name for p in paths] == ["T3_perplexity_change.png", "T3_perplexity_change.svg"]
    assert all(p.stat().st_size > 1000 for p in paths)
    svg = paths[1].read_text()
    assert "Muhammad Obaidullah" in svg and "<dc:date>" not in svg
    (tmp_path / "figures" / "T3_perplexity_change.png").unlink()
    assert ppl.main(["--table", "--figure", "--results-dir", str(tmp_path)]) == 0
    assert (tmp_path / "figures" / "T3_perplexity_change.png").exists()
    assert (tmp_path / "tables" / "T3_perplexity_specificity.csv").exists()


def test_cli_needs_exactly_one_mode():
    with pytest.raises(SystemExit):
        ppl.main([])
    with pytest.raises(SystemExit):
        ppl.main(["--variant", "qwen_base", "--table"])
