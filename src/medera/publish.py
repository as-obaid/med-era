"""Publish safety for result JSON: strip cluster provenance before a file sits in results/ (which is committed).

Job outputs record where they ran (Slurm job ID, partition, node) for debugging. Those belong in the gitignored
runs/ mirror, not in the public repo. The perplexity cells that hpc/sync_back.sh pulls into results/scores/ carry
them in `provenance`, so after every sync run

    python -m medera.publish            # results/scores/**: raw copies -> runs/results/scores_raw/, results/ sanitized

A file that names any private key is first copied byte for byte to runs/results/scores_raw/<same path> (nothing is
deleted), then rewritten in place without those keys. Files that are already clean are left alone, so the command
is idempotent. Every written file is re-checked for cluster paths and node names, and the run fails if one remains.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from medera.jobs import write_text_atomic

PRIVATE_KEYS = frozenset({"job_id", "node", "partition", "host", "hostname", "user", "username"})
_ABS_PATH = re.compile(r"(^|[\s'\"=:(])/(scratch|home|Users|work|projects|private|tmp|var)/")
_NODE_NAME = re.compile(r"^[a-z]\d{4}$")    # cluster node names: one letter and four digits


def scrub(obj):
    """A copy of obj with every PRIVATE_KEYS entry removed, at any depth."""
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items() if k not in PRIVATE_KEYS}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def leaks(obj, where: str = "") -> list[str]:
    """Locations of private keys, absolute cluster or laptop paths, and node-like names left in obj."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            loc = f"{where}.{k}" if where else str(k)
            if k in PRIVATE_KEYS:
                found.append(loc)
            found += leaks(v, loc)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += leaks(v, f"{where}[{i}]")
    elif isinstance(obj, str) and (_ABS_PATH.search(obj) or _NODE_NAME.match(obj)):
        found.append(f"{where}={obj!r}")
    return found


def dump(obj, like: str) -> str:
    """JSON text in the layout of the original file (indented if it was)."""
    return json.dumps(obj, indent=2 if "\n" in like.strip() else None) + "\n"


def sanitized_copy(src: Path, dst: Path) -> bool:
    """Write a sanitized copy of the JSON file src to dst. Returns True if anything was removed."""
    text = src.read_text()
    obj = json.loads(text)
    clean = scrub(obj)
    bad = leaks(clean)
    if bad:
        raise ValueError(f"{src}: private values remain after scrubbing: {bad[:5]}")
    write_text_atomic(dst, dump(clean, text) if clean != obj else text)
    return clean != obj


def sanitize_in_place(root: Path, backup: Path) -> dict:
    """Sanitize every JSON file under root (DONE markers included). A file that changes is first copied to
    backup/<path relative to root>. Returns {"checked", "sanitized", "backed_up": [paths]}."""
    out = {"checked": 0, "sanitized": 0, "backed_up": []}
    for p in sorted(q for q in root.rglob("*") if q.is_file() and not q.name.endswith(".tmp")):
        text = p.read_text()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        out["checked"] += 1
        clean = scrub(obj)
        bad = leaks(clean)
        if bad:
            raise ValueError(f"{p}: private values that scrubbing cannot remove: {bad[:5]}")
        if clean == obj:
            continue
        keep = backup / p.relative_to(root)
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, keep)
        if keep.read_bytes() != text.encode():
            raise OSError(f"backup of {p} does not match the original; nothing rewritten")
        write_text_atomic(p, dump(clean, text))
        out["sanitized"] += 1
        out["backed_up"].append(keep)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args(argv)
    root = Path(args.results_dir) / "scores"
    backup = Path(args.runs_dir) / "results" / "scores_raw"
    if not root.exists():
        print(f"[publish] nothing to do: {root} does not exist")
        return 0
    out = sanitize_in_place(root, backup)
    print(f"[publish] {out['checked']} JSON files under {root}: {out['sanitized']} sanitized "
          f"(raw copies in {backup}), {out['checked'] - out['sanitized']} already clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
