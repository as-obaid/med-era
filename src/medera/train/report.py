"""Training reports from saved results (CPU): T2 adapter audit, the training summary and the loss curves.

    python -m medera.train.report                  # all three
    python -m medera.train.report --only audit     # or: summary, figure

Inputs are the gitignored mirrors of the cluster outputs: runs/results/adapter_audit/<adapter>.json,
runs/results/train/<adapter>.json, the job logs runs/logs/medera_<adapter>_<job>.out (load time of attempts before
the last) and the T3 cells results/scores/perplexity/<adapter>/<era>.json (held-out NLL). Outputs:

    results/tables/T2_adapter_audit.{md,csv}       adapted tensors and LoRA parameters by group, nonzero grads, gate
    results/adapter_audit/<adapter>.json           sanitized copies of the audit reports (medera.publish)
    results/tables/training_summary.{md,csv}       steps, losses, attempts, GPU type, memory, GPU-hours, commit
    results/figures/training_loss.{png,svg}        loss per optimizer step, one panel per arm, one line per era

Job IDs and node names are read only to find an attempt's log; nothing written names them.

GPU-hours per adapter = the wall time of every optimizer step (step_times, all attempts; the first step of each
attempt includes warm-up) + the model load of every attempt (process start to base model loaded: `load_s` of the last
attempt from the run summary, earlier attempts from the "[train] loaded in Ns" line of their job log, or the last
attempt's load time when that log is missing, flagged). Not counted: interpreter start-up before the timer, LoRA
setup, checkpoint saves, the time between the stop deadline and job exit, queueing, and any job that crashed
before training began.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

from medera.config import load_yaml
from medera.jobs import write_text_atomic
from medera.publish import sanitized_copy
from medera.variants import WINDOWS, core_variants

SMOOTH = 9          # centred moving average for the loss figure (steps)
_LOADED = re.compile(r"\[train\] loaded in (\d+(?:\.\d+)?)s")
_GPU = re.compile(r"\b([A-Z]\d{2,3})\b")


def adapter_names(train_cfg: dict) -> list[str]:
    """The 9 core adapters, arm-major (qwen w1-w3, gemma_al w1-w3, gemma_ex w1-w3)."""
    adapters = [v for v in core_variants(train_cfg) if not v.is_base]
    arms = list(dict.fromkeys(v.arm for v in adapters))
    return [v.name for v in sorted(adapters, key=lambda v: (arms.index(v.arm), v.window))]


def _load_dir(d: Path, names: list[str]) -> dict:
    return {n: json.loads((d / f"{n}.json").read_text()) for n in names if (d / f"{n}.json").exists()}


def _short(n: int | float) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K" if n >= 1e3 else f"{n:.0f}"


# ------------------------------------------------------------------------------------------------- T2 audit

def audit_rows(audits: dict) -> list[dict]:
    """Per adapter: adapted tensors, LoRA parameters and nonzero-gradient tensors per audit group, totals and the
    audit gate (medera.train.audit_adapter.check)."""
    from medera.train.audit_adapter import check

    rows = []
    for name, a in audits.items():
        fails = check(a)
        row = {"adapter": name, "arm": a["arm"], "window": a.get("window"), "model": a.get("model"),
               "precision": a.get("precision"), "expect_experts": a["expect_experts"]}
        for g, s in a["groups"].items():
            row.update({f"{g}_tensors": s["lora_tensors"], f"{g}_params": s["lora_params"],
                        f"{g}_nonzero_grad": s["lora_nonzero_grad"]})
        row.update(lora_tensors=a["lora_tensors"], lora_params=a["lora_params"],
                   lora_nonzero_grad=a["lora_nonzero_grad"], base_params=a["base_params"],
                   trainable_pct=100 * a["lora_params"] / a["base_params"],
                   audit="PASS" if not fails else "FAIL", failures="; ".join(fails))
        rows.append(row)
    return rows


def write_audit(runs: Path, results: Path, names: list[str]) -> list[dict]:
    audits = _load_dir(runs / "results" / "adapter_audit", names)
    if not audits:
        raise FileNotFoundError(f"no adapter audits in {runs / 'results' / 'adapter_audit'}")
    rows = audit_rows(audits)
    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "T2_adapter_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    groups = list(next(iter(audits.values()))["groups"])
    md = ["# T2 Adapter audit: adapted tensors and LoRA parameters by group", "",
          "Per adapter: LoRA tensors / trainable LoRA parameters in each audit group, all tensors with a nonzero "
          "gradient during training, and the audit gate. The gate fails if an expert arm (Gemma-EX) adapts no expert "
          "tensor or none of its expert tensors received a nonzero gradient, if a non-expert arm adapts experts, or "
          "if any vision tensor or anything outside the text-model groups is adapted (SCOPE §6). From "
          "`runs/results/adapter_audit/` (copies in `results/adapter_audit/`); all columns in "
          "`T2_adapter_audit.csv`.", "",
          "| adapter | " + " | ".join(g.replace("_", " ") for g in groups)
          + " | LoRA tensors | LoRA params | % of base | nonzero-grad tensors | audit |",
          "|---|" + "---:|" * len(groups) + "---:|---:|---:|---:|---|"]
    for r in rows:
        cells = [f"{r[f'{g}_tensors']} / {_short(r[f'{g}_params'])}" if r[f"{g}_tensors"] else "0" for g in groups]
        md.append(f"| {r['adapter']} | " + " | ".join(cells) + f" | {r['lora_tensors']:,} | {r['lora_params']:,} | "
                  f"{r['trainable_pct']:.2f}% | {r['lora_nonzero_grad']:,} of {r['lora_tensors']:,} | "
                  f"{r['audit']}{(': ' + r['failures']) if r['failures'] else ''} |")
    md += ["", "Cells: LoRA tensors / LoRA parameters (M = million). `% of base` is LoRA parameters over the base "
           "model's parameters (vision tower included). The two Gemma arms share a base and differ only in the "
           "expert group: Gemma-EX also adapts the fused expert weights.", ""]
    write_text_atomic(tables / "T2_adapter_audit.md", "\n".join(md))
    out_dir = results / "adapter_audit"
    for name in audits:
        sanitized_copy(runs / "results" / "adapter_audit" / f"{name}.json", out_dir / f"{name}.json")
    return rows


# ------------------------------------------------------------------------------------------ training summary

def gpu_type(name: str | None) -> str:
    """'NVIDIA H100 80GB HBM3' -> 'H100'."""
    if not name:
        return "unknown"
    m = _GPU.search(name)
    return m.group(1) if m else name.replace("NVIDIA ", "")


def log_load_seconds(logs: Path, adapter: str, job_id) -> float | None:
    """The model load time an attempt's job log reports, or None if the log is not there."""
    if not job_id:
        return None
    p = logs / f"medera_{adapter}_{job_id}.out"
    if not p.exists():
        return None
    m = _LOADED.search(p.read_text(errors="replace"))
    return float(m.group(1)) if m else None


def epoch1_steps(cfg: dict) -> int:
    """Optimizer steps that lie entirely within epoch 1 (epoch 2 starts at micro-batch n_blocks // micro_batch)."""
    return (cfg["n_blocks"] // cfg["micro_batch"]) // cfg["grad_accum"]


def epoch2_start(cfg: dict) -> float:
    """Where epoch 2 begins, in optimizer steps (fractional when a step straddles the boundary)."""
    return (cfg["n_blocks"] // cfg["micro_batch"]) / cfg["grad_accum"]


def _mean(xs) -> float | None:
    return math.fsum(xs) / len(xs) if xs else None


def summary_row(name: str, t: dict, logs: Path, heldout: dict | None) -> dict:
    cfg, attempts, losses = t["config"], t.get("attempts") or [], t["losses"]
    loads, estimated = [], False
    for i, a in enumerate(attempts):
        if i == len(attempts) - 1:
            loads.append(t["load_s"])
            continue
        s = log_load_seconds(logs, name, a.get("job_id"))
        if s is None:
            s, estimated = t["load_s"], True
        loads.append(s)
    if not attempts:
        loads = [t["load_s"]]
    spans = [f"{a['start_step']}-{a['end_step']}" for a in attempts]
    contiguous = all(attempts[i]["end_step"] == attempts[i + 1]["start_step"] for i in range(len(attempts) - 1))
    e1 = epoch1_steps(cfg)
    step_s = math.fsum(t["step_times"])
    gpus = sorted({gpu_type(a.get("gpu")) for a in attempts} or {gpu_type(t.get("gpu"))})
    return {"adapter": name, "arm": t["arm"], "window": t["window"], "steps": t["steps"],
            "total_steps": t["total_steps"], "epochs": cfg["epochs"], "epoch1_steps": e1,
            "loss_first10": _mean(losses[:10]), "loss_epoch1_last10": _mean(losses[max(0, e1 - 10):e1]),
            "loss_last10": _mean(losses[-10:]),
            "heldout_nll_own_era": heldout["mean_nll"] if heldout else None,
            "attempts": len(attempts), "step_spans": " ".join(spans), "spans_contiguous": contiguous,
            "gpu_types": "/".join(gpus), "peak_mem_gb": t["peak_mem_gb"],
            "step_hours": step_s / 3600, "load_hours": math.fsum(loads) / 3600,
            "gpu_hours": (step_s + math.fsum(loads)) / 3600, "load_estimated": estimated,
            "git_sha": t["git_sha"]}


def write_summary(runs: Path, results: Path, names: list[str]) -> list[dict]:
    train = _load_dir(runs / "results" / "train", names)
    if not train:
        raise FileNotFoundError(f"no training summaries in {runs / 'results' / 'train'}")
    rows = []
    for name, t in train.items():
        cell = results / "scores" / "perplexity" / name / f"{t['window']}.json"
        rows.append(summary_row(name, t, runs / "logs", json.loads(cell.read_text()) if cell.exists() else None))
    total = {k: None for k in rows[0]}
    total.update(adapter="total", steps=sum(r["steps"] for r in rows), total_steps=sum(r["total_steps"] for r in rows),
                 attempts=sum(r["attempts"] for r in rows),
                 gpu_types="/".join(sorted({g for r in rows for g in r["gpu_types"].split("/")})),
                 peak_mem_gb=max(r["peak_mem_gb"] for r in rows),
                 step_hours=math.fsum(r["step_hours"] for r in rows), load_hours=math.fsum(r["load_hours"] for r in rows),
                 gpu_hours=math.fsum(r["gpu_hours"] for r in rows),
                 load_estimated=any(r["load_estimated"] for r in rows),
                 git_sha="/".join(sorted({r["git_sha"] for r in rows})))
    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "training_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows + [total])

    def f3(x):
        return "–" if x is None else f"{x:.3f}"

    md = ["# Training summary", "",
          "Per adapter: optimizer steps, training loss (mean NLL per token of the batch) at the start, at the end of "
          "epoch 1 and at the end, the adapter's held-out NLL on its own era (T3), attempts with the steps each "
          "covered, GPU type, peak GPU memory, GPU-hours and the training commit. All columns in "
          "`training_summary.csv`.", "",
          "| adapter | steps | epochs | loss, first 10 steps | loss, last 10 steps of epoch 1 | loss, last 10 steps "
          "| held-out NLL, own era | attempts (steps) | GPU | peak GB | GPU-h | commit |",
          "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|"]
    for r in rows:
        att = f"{r['attempts']} ({', '.join(r['step_spans'].split())})"
        md.append(f"| {r['adapter']} | {r['steps']} | {r['epochs']} | {f3(r['loss_first10'])} | "
                  f"{f3(r['loss_epoch1_last10'])} | {f3(r['loss_last10'])} | {f3(r['heldout_nll_own_era'])} | {att} | "
                  f"{r['gpu_types']} | {r['peak_mem_gb']:.1f} | {r['gpu_hours']:.2f}"
                  f"{'*' if r['load_estimated'] else ''} | {r['git_sha'][:7]} |")
    md.append(f"| **total** | {total['steps']:,} | | | | | | {total['attempts']} | {total['gpu_types']} | "
              f"{total['peak_mem_gb']:.1f} (max) | **{total['gpu_hours']:.2f}** | |")
    e1 = list(dict.fromkeys((r["arm"], r["epoch1_steps"], r["total_steps"]) for r in rows))
    contiguous = all(r["spans_contiguous"] for r in rows)
    md += ["", "- **GPU-hours** = the wall time of every optimizer step across all attempts "
           f"({total['step_hours']:.2f} h) + the model load of every attempt ({total['load_hours']:.2f} h): process "
           "start to base model loaded, from the run summary for the last attempt and from each earlier attempt's job "
           "log. Not counted: interpreter start-up, LoRA setup, checkpoint saves, the time between the stop deadline "
           "and job exit, and queueing. The first step of each attempt includes warm-up and is counted."
           + (" \\* marks a load time estimated from the last attempt (log missing)." if total["load_estimated"]
              else ""),
           "- **Epoch 1** ends inside a step when the blocks do not split evenly into steps of "
           "micro-batch x grad-accum; 'last 10 steps of epoch 1' are the last 10 steps entirely within it ("
           + ", ".join(f"{arm}: steps {s - 9}-{s} of {n}" for arm, s, n in e1) + ").",
           "- **Attempts**: a job that reaches its time limit saves a checkpoint and the next attempt resumes from that "
           "step; " + ("the step spans are contiguous, so no step is repeated or missing." if contiguous else
                       "WARNING: some step spans are not contiguous (see `spans_contiguous` in the CSV).")
           + " **Peak GB** is the last attempt's peak allocated GPU memory. Training loss and held-out NLL compare "
           "within a model only (Qwen and Gemma tokenize differently).", ""]
    write_text_atomic(tables / "training_summary.md", "\n".join(md))
    return rows + [total]


# ---------------------------------------------------------------------------------------------- loss figure

def smooth(xs, k: int = SMOOTH):
    """Centred moving average over k steps (shorter windows at the ends)."""
    import numpy as np

    x = np.asarray(xs, dtype=float)
    h = k // 2
    c = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.clip(np.arange(len(x)) - h, 0, len(x))
    hi = np.clip(np.arange(len(x)) + h + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def write_loss_figure(runs: Path, results: Path, names: list[str]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from medera import figures as F

    train = _load_dir(runs / "results" / "train", names)
    if not train:
        raise FileNotFoundError(f"no training summaries in {runs / 'results' / 'train'}")
    arms = list(dict.fromkeys(t["arm"] for t in train.values()))
    with plt.rc_context(F.RC):
        fig, axes = plt.subplots(1, len(arms), figsize=(4.0 * len(arms), 4.3), sharey=True, squeeze=False)
        axes = axes[0]
        fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.18, wspace=0.08)
        for k, (ax, arm) in enumerate(zip(axes, arms)):
            runs_of_arm = {t["window"]: t for t in train.values() if t["arm"] == arm}
            ax.grid(axis="y", color=F.GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            n_max = 0
            for w in WINDOWS:
                t = runs_of_arm.get(w)
                if t is None:
                    continue
                y = t["losses"]
                x = range(1, len(y) + 1)
                n_max = max(n_max, len(y))
                ax.plot(x, y, color=F.ERA_COLORS[w], linewidth=0.8, alpha=0.28, solid_capstyle="round")
                ax.plot(x, smooth(y), color=F.ERA_COLORS[w], linewidth=2.0, solid_capstyle="round",
                        solid_joinstyle="round")
            cfg = next(iter(runs_of_arm.values()))["config"]
            b = epoch2_start(cfg)
            ax.axvline(b, color=F.MUTED, linewidth=1.0, zorder=1)
            ax.text(b + 2, 1.0, "epoch 2", transform=ax.get_xaxis_transform(), fontsize=8.5, color=F.INK_2,
                    va="top", ha="left")
            ax.set_xlim(0, n_max + 1)
            ax.set_xlabel("optimizer step")
            if k == 0:
                ax.set_ylabel("training loss (mean NLL per token)")
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.set_title(F.ARM_LABELS.get(arm, arm), loc="left", fontsize=12, fontweight="semibold", pad=22)
            ax.text(0, 1.03, f"{F.ARM_SUBTITLES.get(arm, '')}, {n_max} steps", transform=ax.transAxes, fontsize=8.5,
                    color=F.INK_2, va="bottom")
        handles = [Line2D([], [], color=F.ERA_COLORS[w], linewidth=2.0, label=f"{w} ({F.ERA_YEARS[w]})")
                   for w in WINDOWS]
        handles.append(Line2D([], [], color=F.MUTED, linewidth=0.8, alpha=0.6, label="raw per step"))
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.985), ncol=4, fontsize=9,
                   handlelength=1.8, columnspacing=1.4)
        fig.suptitle("Training loss by adapter era", x=0.07, y=0.965, ha="left", fontsize=13, fontweight="semibold")
        fig.text(0.07, 0.025, f"Solid: {SMOOTH}-step centred moving average; faint: raw loss per optimizer step "
                 "(8 x 2048-token blocks). Qwen and Gemma tokenize differently: compare within a model.",
                 fontsize=8, color=F.MUTED)
        paths = F.save(fig, results / "figures", "training_loss", "Training loss per optimizer step by adapter era")
        plt.close(fig)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("audit", "summary", "figure"), help="build one output (default: all)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--train-config", default="configs/train.yaml")
    args = ap.parse_args(argv)
    runs, results = Path(args.runs_dir), Path(args.results_dir)
    names = adapter_names(load_yaml(args.train_config))
    if args.only in (None, "audit"):
        rows = write_audit(runs, results, names)
        bad = [r["adapter"] for r in rows if r["audit"] != "PASS"]
        print(f"[report] T2: {len(rows)} adapters, audit {'FAIL: ' + ', '.join(bad) if bad else 'PASS'} -> "
              f"{results / 'tables' / 'T2_adapter_audit.md'}")
    if args.only in (None, "summary"):
        rows = write_summary(runs, results, names)
        print(f"[report] training summary: {len(rows) - 1} adapters, {rows[-1]['gpu_hours']:.2f} GPU-h -> "
              f"{results / 'tables' / 'training_summary.md'}")
    if args.only in (None, "figure"):
        for p in write_loss_figure(runs, results, names):
            print(f"[report] figure -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
