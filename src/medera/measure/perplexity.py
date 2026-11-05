"""Manipulation check (T3): held-out perplexity of every variant on every era's held-out set.

    python -m medera.measure.perplexity --variant gemma_ex_w2   # GPU: the 3 era cells of one variant (resumable)
    python -m medera.measure.perplexity --table                 # CPU: cells -> results/tables/T3_perplexity.{csv,md}

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
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

from medera.config import load_yaml
from medera.jobs import (EXIT_FAIL, EXIT_OK, EXIT_SMALL_GPU, EXIT_WALL, Stopper, env_versions, provenance,
                         write_json_atomic, write_progress, write_text_atomic)
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
    import numpy as np
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

def load_cells(results: Path) -> dict:
    out = {}
    for p in sorted((results / "scores" / "perplexity").glob("*/w*.json")):
        c = json.loads(p.read_text())
        out[(c["variant"], c["era"])] = c
    return out


def gate_rows(variants: list[Variant], cells: dict) -> list[dict]:
    """Per adapter: base vs adapter perplexity on the adapter's own era (pre-registered gate: adapter < base), and
    whether it is the lowest of its arm's three adapters on that era (descriptive era specificity)."""
    base_of = {v.model: v.name for v in variants if v.is_base}
    rows = []
    for v in variants:
        if v.is_base:
            continue
        own = cells.get((v.name, v.window))
        base = cells.get((base_of[v.model], v.window))
        siblings = [cells.get((f"{v.arm}_{w}", v.window)) for w in WINDOWS]
        row = {"adapter": v.name, "arm": v.arm, "window": v.window, "base": base_of[v.model],
               "ppl_base": base["ppl"] if base else None, "ppl_adapter": own["ppl"] if own else None}
        if own and base:
            row["delta_pct"] = 100 * (own["ppl"] / base["ppl"] - 1)
            row["gate"] = "PASS" if own["ppl"] < base["ppl"] else "FAIL"
        else:
            row["delta_pct"], row["gate"] = None, "pending"
        if all(siblings):
            row["lowest_of_arm_on_own_era"] = min(siblings, key=lambda c: c["ppl"])["variant"] == v.name
        else:
            row["lowest_of_arm_on_own_era"] = None
        rows.append(row)
    return rows


def _fmt(x, nd=3):
    return "–" if x is None else (f"{x:+.2f}%" if nd == "pct" else f"{x:.{nd}f}")


def write_table(results: Path, train_cfg: dict) -> dict:
    variants = core_variants(train_cfg)
    cells = load_cells(results)
    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    long_rows = [{"variant": v.name, "model": v.model, "arm": v.arm or "", "window": v.window or "", "era": e,
                  **{k: cells[(v.name, e)][k] for k in ("n_blocks", "n_tokens", "nll_sum", "mean_nll", "ppl")}}
                 for v in variants for e in WINDOWS if (v.name, e) in cells]
    with open(tables / "T3_perplexity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "model", "arm", "window", "era", "n_blocks", "n_tokens",
                                          "nll_sum", "mean_nll", "ppl"])
        w.writeheader()
        w.writerows(long_rows)
    gates = gate_rows(variants, cells)
    with open(tables / "T3_perplexity_gate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gates[0]) if gates else ["adapter"])
        w.writeheader()
        w.writerows(gates)
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
           "| adapter | base | ppl base | ppl adapter | change | gate | lowest of its arm on this era |",
           "|---|---|---:|---:|---:|---|---|"]
    for g in gates:
        low = {True: "yes", False: "no", None: "–"}[g["lowest_of_arm_on_own_era"]]
        md.append(f"| {g['adapter']} | {g['base']} | {_fmt(g['ppl_base'])} | {_fmt(g['ppl_adapter'])} | "
                  f"{_fmt(g['delta_pct'], 'pct')} | {g['gate']} | {low} |")
    md += ["", "A FAIL is checked for a bug and rerun once with the same settings; if it fails again the adapter is "
           "probed anyway and the failure reported (SCOPE §6, pre-registration).", ""]
    write_text_atomic(tables / "T3_perplexity.md", "\n".join(md))
    return {"cells": len(cells), "gates": gates}


# ----------------------------------------------------------------------------------------------- GPU entry point

def main(argv=None) -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--variant", help="one of the 11 core variants, e.g. qwen_base, gemma_ex_w2")
    g.add_argument("--table", action="store_true", help="CPU: aggregate the cells into results/tables/T3_*")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--models-config", default="configs/models.yaml")
    args = ap.parse_args(argv)
    train_cfg = load_yaml(args.train_config)
    results, runs = Path(args.results_dir), Path(args.runs_dir)
    if args.table:
        out = write_table(results, train_cfg)
        for gr in out["gates"]:
            print(f"[ppl] {gr['adapter']:12s} gate {gr['gate']:7s} {_fmt(gr['delta_pct'], 'pct')}")
        print(f"[ppl] {out['cells']} cells -> {results / 'tables' / 'T3_perplexity.md'}")
        return EXIT_OK

    import numpy as np
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
