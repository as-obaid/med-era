"""Adapter audit: adapted tensors and parameter counts by group, and the pass/fail gate (SCOPE §6).

The run fails unless an expert arm (Gemma-EX) adapts more than zero expert tensors that received a
nonzero gradient during training, and no arm adapts vision tensors. Non-expert arms must not adapt
experts (so Gemma-AL vs Gemma-EX differ in exactly one thing), and nothing outside the known groups
(embeddings, lm_head) may be adapted.

    python -m medera.train.audit_adapter runs/results/adapter_audit/*.json   # table + exit 1 on failure

4-bit base tensors are stored packed (uint8), so their `base_params` are storage elements.
"""
import argparse
import json
import sys

from medera.train.targets import GROUPS, param_group


def is_lora(name: str) -> bool:
    return ".lora_" in name


def audit(model, grad_norms: dict[str, float], arm: str, expect_experts: bool) -> dict:
    groups = {g: {"lora_tensors": 0, "lora_params": 0, "lora_nonzero_grad": 0,
                  "base_params": 0, "base_dtypes": {}} for g in GROUPS}
    for name, p in model.named_parameters():
        g = groups[param_group(name)]
        if is_lora(name):
            g["lora_tensors"] += 1
            g["lora_params"] += p.numel()
            g["lora_nonzero_grad"] += int(grad_norms.get(name, 0.0) > 0)
        else:
            g["base_params"] += p.numel()
            dtype = str(p.dtype).replace("torch.", "")
            g["base_dtypes"][dtype] = g["base_dtypes"].get(dtype, 0) + p.numel()
    total = lambda key: sum(g[key] for g in groups.values())  # noqa: E731
    return {"arm": arm, "expect_experts": expect_experts, "groups": groups,
            "lora_tensors": total("lora_tensors"), "lora_params": total("lora_params"),
            "lora_nonzero_grad": total("lora_nonzero_grad"), "base_params": total("base_params")}


def check(report: dict) -> list[str]:
    """Failure messages; an empty list means the adapter passes the audit."""
    g, fails = report["groups"], []
    if g["vision"]["lora_tensors"]:
        fails.append(f"{g['vision']['lora_tensors']} vision tensors adapted")
    if g["other"]["lora_tensors"]:
        fails.append(f"{g['other']['lora_tensors']} tensors adapted outside the text-model groups")
    if report["expect_experts"]:
        if g["experts"]["lora_tensors"] == 0:
            fails.append("expert arm adapts 0 expert tensors")
        elif g["experts"]["lora_nonzero_grad"] == 0:
            fails.append("expert LoRA tensors never received a nonzero gradient")
    elif g["experts"]["lora_tensors"]:
        fails.append(f"non-expert arm adapts {g['experts']['lora_tensors']} expert tensors")
    if report["lora_tensors"] == 0:
        fails.append("no LoRA tensors")
    elif report["lora_nonzero_grad"] == 0:
        fails.append("no LoRA tensor received a nonzero gradient")
    return fails


def format_table(reports: list[dict]) -> str:
    lines = [f"{'arm':14s} {'group':10s} {'lora_tensors':>12s} {'nonzero_grad':>12s} {'lora_params':>14s} {'base_params':>15s}"]
    for r in reports:
        for name in GROUPS:
            g = r["groups"][name]
            lines.append(f"{r['arm']:14s} {name:10s} {g['lora_tensors']:12d} {g['lora_nonzero_grad']:12d} "
                         f"{g['lora_params']:14,d} {g['base_params']:15,d}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Print and re-check saved adapter audit reports.")
    ap.add_argument("reports", nargs="+")
    args = ap.parse_args(argv)
    reports = [json.load(open(p)) for p in args.reports]
    print(format_table(reports))
    failed = False
    for path, r in zip(args.reports, reports):
        fails = check(r)
        failed |= bool(fails)
        print(f"[audit] {'FAIL' if fails else 'PASS'} {path}" + "".join(f"\n  - {f}" for f in fails))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
