"""Manipulation check (T3): held-out perplexity of every variant on every era's held-out set.

    python -m medera.measure.perplexity --variant gemma_ex_w2   # GPU: the 3 era cells of one variant (resumable)
    python -m medera.measure.perplexity --table                 # CPU: cells -> results/tables/T3_perplexity*.{csv,md}
    python -m medera.measure.perplexity --figure                # CPU: results/figures/T3_perplexity_change.{png,svg}

A cell (variant, era) is the summed token NLL over that era's held-out blocks for the variant's model
(data/processed/hrt/<model>/<era>_heldout.npy: the 2048-token packing used for training, with Gemma's BOS on every
block), one block at a time in eval mode, logits upcast to float32, sums in float64. Perplexity = exp(NLL / tokens),
where tokens = blocks x 2047 predicted positions. Qwen and Gemma use different tokenizers, so perplexities compare
only within a model. Each cell is written atomically to results/scores/perplexity/<variant>/<era>.json with the
per-block NLLs (for CIs later) and full provenance; a rerun skips cells that are already there and match the
held-out file's sha256, and writes DONE once all three exist. Exit codes as in medera.jobs (0 done, 3 stopped
before the deadline: resubmit to resume).

The table gives the era x era matrix, before/after per arm (base vs adapter on the adapter's own era) and the
pre-registered gate: every adapter must have a lower held-out perplexity than its base on its own window (SCOPE
§6). Perplexity may run before the pre-registration tag; probing may not (medera.probe.generate).

Descriptive 95% intervals come from a paired block bootstrap (medera.stats.bootstrap) on the saved per-block NLLs:
the gate's own-era change, the change vs base in all 27 adapter x era cells (T3_perplexity_cells.csv), and era
specificity, the own-era adapter vs each other adapter of its arm on that era (T3_perplexity_specificity.csv).
Within a model every variant is scored on the same blocks (same held-out sha256, checked), so each comparison
resamples block indices once for both sides. The gate itself stays the point comparison above.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from medera.config import load_yaml
from medera.jobs import (EXIT_FAIL, EXIT_OK, EXIT_SMALL_GPU, EXIT_WALL, Stopper, env_versions, provenance,
                         write_json_atomic, write_progress, write_text_atomic)
from medera.stats.bootstrap import N_BOOT, block_indices, ppl_change_ci
from medera.stats.bootstrap import SEED as BOOT_SEED
from medera.variants import WINDOWS, Variant, adapter_info, core_variants, get_variant, load_variant, sha256_file

DEFAULT_S_PER_BLOCK = 1.5   # first-cell estimate before any block is timed (H100 forward of 2048 tokens is ~0.3-0.5 s)
LOAD_EST_S = 600            # loading a 52-54 GB bf16 base took 2-6 min on the H100s; 10 min with margin


def block_nll(model, x) -> float:
    """Summed NLL (nats) of x[0, 1:] given its prefix, for one block x [1, L] (long, on the model's device)."""
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        logits = model(input_ids=x, use_cache=False).logits
        return float(F.cross_entropy(logits[0, :-1].float(), x[0, 1:], reduction="sum").double())


def eval_cell(model, blocks, log=print) -> dict:
    import torch

    device = next(model.parameters()).device
    nll, t0 = [], time.time()
    for i in range(len(blocks)):
        x = torch.from_numpy(np.asarray(blocks[i : i + 1], dtype=np.int64)).to(device)
        nll.append(block_nll(model, x))
    n_tokens = len(blocks) * (blocks.shape[1] - 1)
    total = math.fsum(nll)
    secs = time.time() - t0
    log(f"[ppl]   {len(blocks)} blocks in {secs:.0f}s ({secs / max(1, len(blocks)):.2f} s/block)")
    return {"n_blocks": len(blocks), "block_len": int(blocks.shape[1]), "n_tokens": n_tokens, "nll_sum": total,
            "mean_nll": total / n_tokens, "ppl": math.exp(total / n_tokens), "block_nll": nll, "seconds": secs}


def cell_path(results: Path, variant: str, era: str) -> Path:
    return results / "scores" / "perplexity" / variant / f"{era}.json"


def cell_is_current(path: Path, v: Variant, era: str, heldout_sha: str, adapter_sha: str | None) -> bool:
    if not path.exists():
        return False
    c = json.loads(path.read_text())
    return (c.get("variant") == v.name and c.get("era") == era and c.get("heldout_sha256") == heldout_sha
            and c.get("adapter_sha256") == adapter_sha)


def run_variant(v: Variant, model_loader, heldout: dict, results: Path, stopper: Stopper, prov: dict,
                meta: dict, progress: Path | None = None, log=print) -> str:
    """Evaluate the missing era cells of one variant. heldout: era -> (blocks [n, L], sha256 of the .npy).
    model_loader() is called only if some cell is missing. Returns "done" or "wall"."""
    adapter_sha = meta.get("adapter_sha256")
    todo = [e for e in WINDOWS if not cell_is_current(cell_path(results, v.name, e), v, e, heldout[e][1], adapter_sha)]
    done_n = len(WINDOWS) - len(todo)
    log(f"[ppl] {v.name}: {done_n}/{len(WINDOWS)} cells already done; to do: {todo or 'none'}")
    model, measured = None, 0.0
    for era in todo:
        blocks, sha = heldout[era]
        est = len(blocks) * (measured or DEFAULT_S_PER_BLOCK) + (LOAD_EST_S if model is None else 0)
        if not stopper.fits(est):
            log(f"[ppl] stopping before {v.name}/{era}: ~{est / 60:.1f} min needed, "
                f"{stopper.minutes_left():.1f} min left (or SIGTERM)")
            return "wall"
        if model is None:
            model = model_loader()
        log(f"[ppl] {v.name} on {era}: {len(blocks)} held-out blocks")
        res = eval_cell(model, blocks, log=log)
        measured = max(measured, res["seconds"] / max(1, len(blocks)))
        write_json_atomic(cell_path(results, v.name, era),
                          {"variant": v.name, "model": v.model, "arm": v.arm, "window": v.window, "era": era,
                           **{k: res[k] for k in ("n_blocks", "block_len", "n_tokens", "nll_sum", "mean_nll", "ppl")},
                           "heldout_sha256": sha, **meta, "provenance": prov, "seconds": res["seconds"],
                           "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "block_nll": res["block_nll"]})
        log(f"[ppl] {v.name} on {era}: ppl {res['ppl']:.4f} (mean NLL {res['mean_nll']:.5f}, "
            f"{res['n_tokens']:,} tokens)")
        done_n += 1
        write_progress(progress, {"step": done_n}, len(WINDOWS), prov)
    write_json_atomic(results / "scores" / "perplexity" / v.name / "DONE",
                      {"variant": v.name, "cells": list(WINDOWS), "git_sha": prov.get("git_sha"),
                       "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=None)
    return "done"


# ----------------------------------------------------------------------------------------------- T3 table (CPU)

HELDOUT_ARTICLES_W1 = 16    # ~5% of w1's 315 articles held out (SCOPE §5); the smallest held-out set
NONE_CI = {"pct": None, "lo": None, "hi": None}


def load_cells(results: Path) -> dict:
    out = {}
    for p in sorted((results / "scores" / "perplexity").glob("*/w*.json")):
        c = json.loads(p.read_text())
        out[(c["variant"], c["era"])] = c
    return out


def block_arrays(cell: dict):
    """(summed NLL per block, predicted tokens per block) of a cell, or None if it has no per-block NLLs. Per-block
    token counts are not stored; every block has block_len - 1 predicted tokens, so tokens per block is
    n_tokens / n_blocks, and the per-block NLLs must reproduce the cell's saved perplexity."""
    nll = cell.get("block_nll")
    if not nll:
        return None
    name, n, n_tok = f"{cell['variant']}/{cell['era']}", cell["n_blocks"], cell["n_tokens"]
    if len(nll) != n or n_tok % n:
        raise ValueError(f"{name}: {len(nll)} block NLLs for {n} blocks and {n_tok} tokens")
    ppl = math.exp(math.fsum(nll) / n_tok)
    if not math.isclose(ppl, cell["ppl"], rel_tol=1e-9):
        raise ValueError(f"{name}: per-block NLLs give perplexity {ppl!r}, the cell says {cell['ppl']!r}")
    return np.asarray(nll, dtype=np.float64), n_tok // n


def compare(cell_a: dict, cell_b: dict, n_boot: int = N_BOOT, idx_cache: dict | None = None) -> dict:
    """% change in perplexity of A vs B (same model, same era) and its paired bootstrap 95% CI. Both cells must come
    from the same held-out file (sha256); lo/hi are None when a cell has no per-block NLLs."""
    sha_a, sha_b = cell_a.get("heldout_sha256"), cell_b.get("heldout_sha256")
    if sha_a != sha_b:
        raise ValueError(f"{cell_a['variant']} and {cell_b['variant']} on {cell_a['era']} were scored on different "
                         f"held-out files ({sha_a} vs {sha_b})")
    out = {"pct": 100 * (cell_a["ppl"] / cell_b["ppl"] - 1), "lo": None, "hi": None}
    arr_a, arr_b = block_arrays(cell_a), block_arrays(cell_b)
    if arr_a is None or arr_b is None:
        return out
    if sha_a is None:
        raise ValueError(f"{cell_a['variant']}/{cell_a['era']}: per-block NLLs without a held-out sha256; cannot "
                         "check that both sides were scored on the same blocks")
    (a, tok), (b, tok_b) = arr_a, arr_b
    if a.shape != b.shape or tok != tok_b:
        raise ValueError(f"{cell_a['variant']} and {cell_b['variant']} on {cell_a['era']}: same held-out sha256 "
                         "but different blocks")
    cache = {} if idx_cache is None else idx_cache
    key = (a.size, n_boot)
    if key not in cache:
        cache[key] = block_indices(a.size, n_boot, BOOT_SEED)
    r = ppl_change_ci(a, b, tok, idx=cache[key])
    out.update(lo=r["lo"], hi=r["hi"])
    return out


def _base_of(variants: list[Variant]) -> dict:
    return {v.model: v.name for v in variants if v.is_base}


def cell_rows(variants: list[Variant], cells: dict, n_boot: int = N_BOOT, idx_cache: dict | None = None) -> list[dict]:
    """All adapter x era cells: % change in perplexity vs the model's base, with its 95% CI."""
    base_of, cache = _base_of(variants), {} if idx_cache is None else idx_cache
    rows = []
    for v in sorted((v for v in variants if not v.is_base), key=lambda v: (_arms(variants).index(v.arm), v.window)):
        for e in WINDOWS:
            own, base = cells.get((v.name, e)), cells.get((base_of[v.model], e))
            c = compare(own, base, n_boot, cache) if own and base else NONE_CI
            rows.append({"adapter": v.name, "arm": v.arm, "window": v.window, "era": e, "own_era": v.window == e,
                         "base": base_of[v.model], "n_blocks": own["n_blocks"] if own else None,
                         "ppl_base": base["ppl"] if base else None, "ppl_adapter": own["ppl"] if own else None,
                         "delta_pct": c["pct"], "ci_lo_pct": c["lo"], "ci_hi_pct": c["hi"]})
    return rows


def gate_rows(variants: list[Variant], cells: dict, cell_ci: dict | None = None) -> list[dict]:
    """Per adapter: base vs adapter perplexity on the adapter's own era (pre-registered gate: adapter < base), the
    descriptive 95% CI of the change (cell_ci: (adapter, era) -> compare() result), and whether it is the lowest of
    its arm's three adapters on that era (descriptive era specificity)."""
    base_of, cell_ci = _base_of(variants), cell_ci or {}
    rows = []
    for v in variants:
        if v.is_base:
            continue
        own = cells.get((v.name, v.window))
        base = cells.get((base_of[v.model], v.window))
        siblings = [cells.get((f"{v.arm}_{w}", v.window)) for w in WINDOWS]
        row = {"adapter": v.name, "arm": v.arm, "window": v.window, "base": base_of[v.model],
               "ppl_base": base["ppl"] if base else None, "ppl_adapter": own["ppl"] if own else None}
        ci = cell_ci.get((v.name, v.window), NONE_CI)
        if own and base:
            row["delta_pct"] = 100 * (own["ppl"] / base["ppl"] - 1)
            row.update(ci_lo_pct=ci["lo"], ci_hi_pct=ci["hi"])
            row["gate"] = "PASS" if own["ppl"] < base["ppl"] else "FAIL"
        else:
            row.update(delta_pct=None, ci_lo_pct=None, ci_hi_pct=None, gate="pending")
        if all(siblings):
            row["lowest_of_arm_on_own_era"] = min(siblings, key=lambda c: c["ppl"])["variant"] == v.name
        else:
            row["lowest_of_arm_on_own_era"] = None
        rows.append(row)
    return rows


def _arms(variants: list[Variant]) -> list[str]:
    return list(dict.fromkeys(v.arm for v in variants if not v.is_base))


def _ci_side(lo, hi) -> str | None:
    if lo is None:
        return None
    return "below 0" if hi < 0 else "above 0" if lo > 0 else "spans 0"


def specificity_rows(variants: list[Variant], cells: dict, n_boot: int = N_BOOT,
                     idx_cache: dict | None = None) -> list[dict]:
    """On each era, the arm's adapter trained on that era vs each of the arm's other adapters: % difference in
    perplexity (negative: the own-era adapter fits that era's text better) with its 95% CI."""
    names, cache, rows = {v.name for v in variants}, {} if idx_cache is None else idx_cache, []
    for arm in _arms(variants):
        for e in WINDOWS:
            own_name = f"{arm}_{e}"
            for w in WINDOWS:
                other_name = f"{arm}_{w}"
                if w == e or own_name not in names or other_name not in names:
                    continue
                own, other = cells.get((own_name, e)), cells.get((other_name, e))
                c = compare(own, other, n_boot, cache) if own and other else NONE_CI
                rows.append({"arm": arm, "era": e, "own_adapter": own_name, "other_adapter": other_name,
                             "ppl_own": own["ppl"] if own else None, "ppl_other": other["ppl"] if other else None,
                             "diff_pct": c["pct"], "ci_lo_pct": c["lo"], "ci_hi_pct": c["hi"],
                             "ci_vs_0": _ci_side(c["lo"], c["hi"])})
    return rows


def _fmt(x, nd=3):
    return "–" if x is None else (f"{x:+.2f}%" if nd == "pct" else f"{x:.{nd}f}")


def _fmt_ci(lo, hi) -> str:
    return "–" if lo is None else f"[{lo:+.2f}, {hi:+.2f}]"


def _write_csv(path: Path, rows: list[dict], fallback: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else fallback)
        w.writeheader()
        w.writerows(rows)


def _ci_md(cells: dict, cell_list: list[dict], spec: list[dict], variants: list[Variant]) -> list[str]:
    """The markdown sections for the all-cells matrix, era specificity and the caveats on the intervals."""
    md = ["", "## Change vs base in every adapter x era cell (95% CI)", "",
          "% change in held-out perplexity vs the model's base (negative: lower perplexity). Rows: adapter; columns: "
          "held-out era; **bold**: the adapter's own era. Also in `T3_perplexity_cells.csv`.", "",
          "| adapter | " + " | ".join(f"held-out {e}" for e in WINDOWS) + " |", "|---|" + "---:|" * len(WINDOWS)]
    by_adapter: dict = {}
    for r in cell_list:
        by_adapter.setdefault(r["adapter"], {})[r["era"]] = r
    for name, row in by_adapter.items():
        parts = []
        for e in WINDOWS:
            r = row[e]
            txt = f"{_fmt(r['delta_pct'], 'pct')} {_fmt_ci(r['ci_lo_pct'], r['ci_hi_pct'])}"
            parts.append(f"**{txt}**" if r["own_era"] else txt)
        md.append(f"| {name} | " + " | ".join(parts) + " |")

    md += ["", "## Era specificity: own-era adapter vs the arm's other adapters", "",
           "On each era's held-out text, the % perplexity difference between the adapter trained on that era and each "
           "other adapter of the same arm (negative: the own-era adapter fits that era's text better). Also in "
           "`T3_perplexity_specificity.csv`.", ""]
    with_ci = [r for r in spec if r["ci_vs_0"] is not None]
    if with_ci:
        groups: dict = {}
        for r in with_ci:
            groups.setdefault((r["arm"], r["era"]), []).append(r["ci_vs_0"])
        clear = sum(all(s == "below 0" for s in sides) for sides in groups.values())
        md.append(f"The own-era adapter is lower than both other adapters of its arm (CI below 0) in {clear} of "
                  f"{len(groups)} arm x era cases.")
        rest = [r for r in with_ci if r["ci_vs_0"] != "below 0"]
        if rest:
            md.append("Comparisons where the interval does not lie below 0: " + "; ".join(
                f"{r['own_adapter']} vs {r['other_adapter']} on {r['era']} {_fmt(r['diff_pct'], 'pct')} "
                f"{_fmt_ci(r['ci_lo_pct'], r['ci_hi_pct'])} ({r['ci_vs_0']})" for r in rest) + ".")
        md.append("")
    md += ["| arm | held-out era | own-era adapter | vs | difference | 95% CI | CI vs 0 |",
           "|---|---|---|---|---:|---:|---|"]
    for r in spec:
        md.append(f"| {r['arm']} | {r['era']} | {r['own_adapter']} | {r['other_adapter']} | "
                  f"{_fmt(r['diff_pct'], 'pct')} | {_fmt_ci(r['ci_lo_pct'], r['ci_hi_pct'])} | {r['ci_vs_0'] or '–'} |")

    blocks = {}
    for v in variants:
        if v.is_base:
            blocks[v.name] = " / ".join(f"{e} {cells[(v.name, e)]['n_blocks']}" for e in WINDOWS if (v.name, e) in cells)
    md += ["", "## How the intervals are computed, and their limits", "",
           f"- Paired block bootstrap on the saved per-block NLLs: {N_BOOT:,} resamples of the held-out blocks with "
           f"replacement (seed {BOOT_SEED}). Both sides of a comparison use the same resampled blocks, and every "
           "comparison on one model's held-out set uses the same draws. Statistic: 100 x (exp(mean token NLL of A "
           "- mean token NLL of B) - 1), token-weighted over the resampled blocks; interval: 2.5th to 97.5th "
           "percentile.",
           "- Within a model every variant is scored on identical blocks (same held-out sha256, checked before each "
           "comparison). Held-out blocks per era: " + "; ".join(f"{k} {s}" for k, s in blocks.items()) + ".",
           "- The intervals are descriptive, not pre-registered tests. The pre-registered gate is the point "
           "comparison in the gate table (adapter < base on its own era) and does not use them.",
           "- They are optimistic (too narrow). Blocks are consecutive 2048-token pieces of packed articles, so "
           "blocks from the same article are correlated, while the bootstrap resamples blocks as if independent. "
           f"w1 has only {HELDOUT_ARTICLES_W1} held-out articles, so its intervals are the least reliable.", ""]
    return md


def write_table(results: Path, train_cfg: dict, n_boot: int = N_BOOT) -> dict:
    variants = core_variants(train_cfg)
    cells = load_cells(results)
    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    long_rows = [{"variant": v.name, "model": v.model, "arm": v.arm or "", "window": v.window or "", "era": e,
                  **{k: cells[(v.name, e)][k] for k in ("n_blocks", "n_tokens", "nll_sum", "mean_nll", "ppl")}}
                 for v in variants for e in WINDOWS if (v.name, e) in cells]
    _write_csv(tables / "T3_perplexity.csv", long_rows,
               ["variant", "model", "arm", "window", "era", "n_blocks", "n_tokens", "nll_sum", "mean_nll", "ppl"])
    idx_cache: dict = {}
    cell_list = cell_rows(variants, cells, n_boot, idx_cache)
    cell_ci = {(r["adapter"], r["era"]): {"pct": r["delta_pct"], "lo": r["ci_lo_pct"], "hi": r["ci_hi_pct"]}
               for r in cell_list}
    gates = gate_rows(variants, cells, cell_ci)
    spec = specificity_rows(variants, cells, n_boot, idx_cache)
    _write_csv(tables / "T3_perplexity_gate.csv", gates, ["adapter"])
    _write_csv(tables / "T3_perplexity_cells.csv", cell_list, ["adapter"])
    _write_csv(tables / "T3_perplexity_specificity.csv", spec, ["arm"])
    md = ["# T3 Manipulation check: held-out perplexity", "",
          f"{len(cells)} of {len(variants) * len(WINDOWS)} cells. Rows: variant; columns: held-out era. Compare "
          "within a model only (Qwen and Gemma tokenize differently).", "",
          "| variant | model | " + " | ".join(f"held-out {e}" for e in WINDOWS) + " |",
          "|---|---|" + "---:|" * len(WINDOWS)]
    for v in variants:
        md.append(f"| {v.name} | {v.model} | "
                  + " | ".join(_fmt(cells[(v.name, e)]["ppl"]) if (v.name, e) in cells else "–" for e in WINDOWS)
                  + " |")
    md += ["", "## Gate: adapter vs base on the adapter's own era (pre-registered: adapter < base)", "",
           "| adapter | base | ppl base | ppl adapter | change | 95% CI | gate | lowest of its arm on this era |",
           "|---|---|---:|---:|---:|---:|---|---|"]
    for g in gates:
        low = {True: "yes", False: "no", None: "–"}[g["lowest_of_arm_on_own_era"]]
        md.append(f"| {g['adapter']} | {g['base']} | {_fmt(g['ppl_base'])} | {_fmt(g['ppl_adapter'])} | "
                  f"{_fmt(g['delta_pct'], 'pct')} | {_fmt_ci(g['ci_lo_pct'], g['ci_hi_pct'])} | {g['gate']} | {low} |")
    md += ["", "A FAIL is checked for a bug and rerun once with the same settings; if it fails again the adapter is "
           "probed anyway and the failure reported (SCOPE §6, pre-registration). The 95% CI is descriptive (see the "
           "last section); the gate uses the point comparison only."]
    md += _ci_md(cells, cell_list, spec, variants)
    write_text_atomic(tables / "T3_perplexity.md", "\n".join(md))
    return {"cells": len(cells), "gates": gates, "cell_rows": cell_list, "specificity": spec}


# ----------------------------------------------------------------------------------------------- T3 figure (CPU)

def _text_color(rgba) -> str:
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3])
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.30 else "#0b0b0b"


def write_figure(results: Path, train_cfg: dict, n_boot: int = N_BOOT) -> list[Path]:
    """3 panels (one per arm), each the 3 x 3 grid of % perplexity change vs base: rows = adapter era, columns =
    held-out era, own era outlined, one colour scale centred at 0 shared by all panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    from medera import figures as F

    variants = core_variants(train_cfg)
    rows = cell_rows(variants, load_cells(results), n_boot)
    vals = {(r["arm"], r["window"], r["era"]): r for r in rows}
    found = [abs(r["delta_pct"]) for r in rows if r["delta_pct"] is not None]
    if not found:
        raise FileNotFoundError(f"no perplexity cells under {results / 'scores' / 'perplexity'}")
    lim = max(0.5, math.ceil(max(found) * 2) / 2)
    norm, cmap = Normalize(-lim, lim), F.diverging_cmap()
    arms = _arms(variants)
    gap = 0.04
    labels = [f"{e}\n{F.ERA_YEARS[e]}" for e in WINDOWS]
    with plt.rc_context(F.RC):
        fig, axes = plt.subplots(1, len(arms), figsize=(4.0 * len(arms), 5.0))
        fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.30, wspace=0.18)
        for k, (ax, arm) in enumerate(zip(axes, arms)):
            for i, aw in enumerate(WINDOWS):
                for j, e in enumerate(WINDOWS):
                    r = vals.get((arm, aw, e))
                    pct = r["delta_pct"] if r else None
                    face = cmap(norm(pct)) if pct is not None else "#f6f6f4"
                    ax.add_patch(Rectangle((j + gap, i + gap), 1 - 2 * gap, 1 - 2 * gap, facecolor=face,
                                           edgecolor="none"))
                    if pct is None:
                        ax.text(j + 0.5, i + 0.5, "–", ha="center", va="center", color=F.MUTED)
                        continue
                    ink = _text_color(face)
                    ax.text(j + 0.5, i + 0.44, f"{pct:+.2f}%", ha="center", va="center", fontsize=12,
                            fontweight="semibold", color=ink)
                    if r["ci_lo_pct"] is not None:
                        ax.text(j + 0.5, i + 0.68, _fmt_ci(r["ci_lo_pct"], r["ci_hi_pct"]), ha="center",
                                va="center", fontsize=7.5, color=ink)
            for d in range(len(WINDOWS)):   # the adapter's own era
                ax.add_patch(Rectangle((d + gap, d + gap), 1 - 2 * gap, 1 - 2 * gap, fill=False, edgecolor=F.INK,
                                       linewidth=2.2))
            ax.set_xlim(0, len(WINDOWS))
            ax.set_ylim(len(WINDOWS), 0)
            ax.set_aspect("equal")
            ax.set_xticks([j + 0.5 for j in range(len(WINDOWS))], labels)
            ax.set_yticks([i + 0.5 for i in range(len(WINDOWS))], labels)
            ax.tick_params(length=0, pad=4)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xlabel("held-out era")
            if k == 0:
                ax.set_ylabel("adapter trained on")
            ax.set_title(F.ARM_LABELS.get(arm, arm), loc="left", fontsize=12, fontweight="semibold", pad=22)
            ax.text(0, 1.025, F.ARM_SUBTITLES.get(arm, ""), transform=ax.transAxes, fontsize=8.5, color=F.INK_2,
                    va="bottom")
        fig.suptitle("Held-out perplexity change vs the model's base", x=0.08, y=0.965, ha="left", fontsize=13,
                     fontweight="semibold")
        cax = fig.add_axes([0.30, 0.12, 0.44, 0.03])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        step = 1.0 if lim > 1.5 else 0.5
        ticks = np.arange(-lim, lim + 1e-9, step)
        cb.set_ticks(ticks, labels=["0%" if abs(t) < 1e-9 else f"{t:+.{0 if step == 1.0 else 1}f}%" for t in ticks])
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=8.5)
        cb.set_label("% change in perplexity vs base:  blue = lower (better),  red = higher (worse)", fontsize=9,
                     color=F.INK_2)
        fig.text(0.08, 0.025, "Cell: % change; brackets: paired block-bootstrap 95% CI (descriptive, optimistic). "
                 "Outlined: the adapter's own era. Compare within a model only.", fontsize=8, color=F.MUTED)
        paths = F.save(fig, results / "figures", "T3_perplexity_change",
                       "T3 held-out perplexity change vs base, adapter era x held-out era")
        plt.close(fig)
    return paths


# ----------------------------------------------------------------------------------------------- GPU entry point

def main(argv=None) -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", help="GPU: one of the 11 core variants, e.g. qwen_base, gemma_ex_w2")
    ap.add_argument("--table", action="store_true", help="CPU: aggregate the cells into results/tables/T3_*")
    ap.add_argument("--figure", action="store_true", help="CPU: results/figures/T3_perplexity_change.{png,svg}")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--models-config", default="configs/models.yaml")
    args = ap.parse_args(argv)
    if bool(args.variant) == (args.table or args.figure):
        ap.error("give --variant (GPU), or --table and/or --figure (CPU)")
    train_cfg = load_yaml(args.train_config)
    results, runs = Path(args.results_dir), Path(args.runs_dir)
    if args.table or args.figure:
        if args.table:
            out = write_table(results, train_cfg)
            for gr in out["gates"]:
                print(f"[ppl] {gr['adapter']:12s} gate {gr['gate']:7s} {_fmt(gr['delta_pct'], 'pct')} "
                      f"{_fmt_ci(gr['ci_lo_pct'], gr['ci_hi_pct'])}")
            print(f"[ppl] {out['cells']} cells -> {results / 'tables' / 'T3_perplexity.md'}")
        if args.figure:
            for path in write_figure(results, train_cfg):
                print(f"[ppl] figure -> {path}")
        return EXIT_OK

    import torch

    from medera.train.lora_cpt import MIN_GPU_BYTES, gpu_big_enough

    v = get_variant(train_cfg, args.variant)
    model_cfg = load_yaml(args.models_config)[v.model]
    meta = {"model_hf_id": model_cfg["hf_id"], "model_revision": model_cfg["revision"], "precision": "bf16",
            "adapter_sha256": None, **adapter_info(runs, v)}
    hdir = Path(args.data_root) / "processed" / "hrt" / v.model
    heldout = {e: (np.load(hdir / f"{e}_heldout.npy", mmap_mode="r"), sha256_file(hdir / f"{e}_heldout.npy"))
               for e in WINDOWS}
    prov = provenance(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    prov["versions"] = env_versions()
    print(f"[ppl] variant={v.name} model={model_cfg['hf_id']}@{model_cfg['revision'][:8]} "
          f"adapter={meta.get('adapter') or '-'} attempt={prov['attempt']} gpu={prov['gpu']} node={prov['node']} "
          f"commit={prov['git_sha'][:7]}", flush=True)
    if torch.cuda.is_available() and not gpu_big_enough(torch.cuda.get_device_properties(0).total_memory):
        print(f"[ppl] ERROR: {prov['gpu']} is under {MIN_GPU_BYTES / 1e9:.0f} GB. Exit {EXIT_SMALL_GPU}.", flush=True)
        return EXIT_SMALL_GPU
    torch.use_deterministic_algorithms(True, warn_only=True)
    stopper = Stopper(t_start, train_cfg["defaults"]).install()
    print(f"[ppl] stop deadline in {stopper.minutes_left():.0f} min", flush=True)

    def loader():
        t0 = time.time()
        m = load_variant(v, model_cfg, runs)
        print(f"[ppl] loaded {v.name} in {time.time() - t0:.0f}s", flush=True)
        return m

    status = run_variant(v, loader, heldout, results, stopper, prov, meta,
                         progress=runs / "eval" / "ppl" / v.name / "progress.json", log=lambda m: print(m, flush=True))
    return EXIT_OK if status == "done" else EXIT_WALL if status == "wall" else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
