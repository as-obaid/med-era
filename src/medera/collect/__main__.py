"""CLI: python -m medera.collect count | harvest | merge"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from medera.collect.europepmc import EuropePMC
from medera.collect.harvest import harvest
from medera.collect.manifest import DROPPED_COLUMNS, merge, read_manifest, write_manifest
from medera.collect.screen import agreement_summary, export_window, import_labels, write_import
from medera.collect.query import COUNT_COLUMNS, PART_COLUMNS, count_rows, or_guard, part_count_rows
from medera.config import load_yaml
from medera.preprocess.filters import is_usable


def make_client(cfg: dict, pause_s: float | None = None) -> EuropePMC:
    p = cfg["polite"]
    return EuropePMC(cfg["api_base"], cfg["user_agent"], max_retries=p["max_retries"], backoff_s=p["backoff_s"],
                     timeout_s=p["timeout_s"], pause_s=p["pause_s"] if pause_s is None else pause_s)


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, columns)
        w.writeheader()
        w.writerows(rows)


def cmd_count(cfg, root: Path, args) -> None:
    client = make_client(cfg, pause_s=cfg["polite"].get("count_pause_s", 1.0))
    rows = count_rows(client, cfg)
    parts = part_count_rows(client, cfg)
    _write_csv(root / "manifests" / "counts.csv", COUNT_COLUMNS, rows)
    _write_csv(root / "manifests" / "counts_parts.csv", PART_COLUMNS, parts)
    for r in rows:
        print(f"[count] {r['query_id']:5s} {r['window']} {r['access']:11s} {r['hit_count']:>7}")
    for r in parts:
        print(f"[count]   part {r['part']:22s} {r['window']} {r['access']:11s} {r['hit_count']:>7}")
    print(f"[count] wrote {root / 'manifests' / 'counts.csv'} and counts_parts.csv")
    fails = or_guard(rows, parts)
    for f in fails:
        print(f"[count] GUARD FAILED: {f}", flush=True)
    if fails:
        raise SystemExit(1)
    print("[count] OR guard passed: every query returns at least its largest part")


def cmd_harvest(cfg, root: Path, args) -> None:
    harvest(make_client(cfg), cfg, args.window, args.query, root, limit=args.limit, resume=args.resume,
            rank_by=args.rank_by, top_up_to=args.top_up_to)


def cmd_merge(cfg, root: Path, args) -> None:
    corpus = cfg["corpus"]
    rows_by_window = {}
    for w in cfg["windows"]:
        path = root / "manifests" / f"{corpus}_{w}.csv"
        if not path.exists():
            print(f"[merge] WARNING: {path} is missing; window {w} is skipped", flush=True)
            continue
        rows_by_window[w] = read_manifest(path)
    if not rows_by_window:
        raise SystemExit(f"[merge] no window manifests under {root / 'manifests'}")
    windows = {w: tuple(v) for w, v in cfg["windows"].items()}
    dropped = []
    out = merge(rows_by_window, windows, dropped)
    path = root / "manifests" / f"{corpus}_all.csv"
    write_manifest(path, out)
    dropped_path = root / "manifests" / "merge_dropped.csv"
    with open(dropped_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, DROPPED_COLUMNS)
        wr.writeheader()
        wr.writerows(dropped)
    for d in dropped:
        print(f"[merge] dropped {d['pmcid']} ({d['window']}) {d['reason']}, kept {d['kept_pmcid']}")
    cpt = cfg["chars_per_token"]
    for w, rows in rows_by_window.items():
        kept = [r for r in out if r["window"] == w]
        xml_paths = [root / "raw" / corpus / w / f"{r['pmcid']}.xml" for r in kept]
        missing = sum(not p.exists() for p in xml_paths)
        usable = [r for r, p in zip(kept, xml_paths) if p.exists() and is_usable(p.read_bytes(), cfg["filters"])]
        est = sum(int(r["n_chars"] or 0) for r in kept) / cpt
        usable_est = sum(int(r["n_chars"] or 0) for r in usable) / cpt
        print(f"[merge] {w}: in={len(rows)} kept={len(kept)} usable(filters)={len(usable)} est_tokens={est:,.0f} "
              f"usable_est_tokens={usable_est:,.0f}")
        if missing:
            print(f"[merge] WARNING: {missing} {w} XML files are missing, so they are not counted as usable")
    print(f"[merge] wrote {path} ({len(out)} rows) and {dropped_path} ({len(dropped)} dropped)")


def cmd_screen_export(cfg, root: Path, args) -> None:
    batches = export_window(cfg, root, args.window, args.batch_size or cfg["screen"]["batch_size"], args.overwrite)
    for name, n in batches:
        print(f"[screen] {name}.jsonl {n} items")
    print(f"[screen] {args.window}: {sum(n for b, n in batches if '_b' in b)} items in "
          f"{sum('_b' in b for b, _ in batches)} batches + {sum(n for b, n in batches if '_d' in b)} double-screen; "
          f"key -> {root / cfg['screen']['dir'] / 'key.csv'}")


def cmd_screen_import(cfg, root: Path, args) -> None:
    labels, agreement, errors, warnings = import_labels(cfg, root, args.window, args.screener)
    for w in warnings:
        print(f"[screen] warning: {w}")
    if errors:
        for e in errors:
            print(f"[screen] ERROR: {e}")
        raise SystemExit(1)
    lp, ap = write_import(cfg, root, labels, agreement)
    for w in sorted({r["window"] for r in labels}):
        c = Counter(r["label"] for r in labels if r["window"] == w)
        print(f"[screen] {w}: " + " ".join(f"{k}={c[k]}" for k in ("relevant", "borderline", "off_topic")))
    for name, n, rate, kappa in agreement_summary(agreement):
        print(f"[screen] double-screen {name}: n={n} agreement={rate:.3f} kappa={kappa:.3f}")
    print(f"[screen] wrote {lp} ({len(labels)} labels) and {ap} ({len(agreement)} pairs)")


def main(argv=None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/corpus.yaml")
    common.add_argument("--data-root", default="data")
    ap = argparse.ArgumentParser(prog="python -m medera.collect", description="Europe PMC corpus collector")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("count", parents=[common], help="hitCount per query x window x access -> counts.csv")
    h = sub.add_parser("harvest", parents=[common], help="download one window's full-text XML + manifest")
    h.add_argument("--window", required=True)
    h.add_argument("--query", required=True, choices=["core", "broad"])
    h.add_argument("--limit", type=int, default=None,
                   help="max NEW manifest rows this run (downloaded, or rebuilt from local XML with --resume)")
    h.add_argument("--resume", action="store_true",
                   help="keep the manifest, skip PMCIDs already in it, and rebuild rows from XML already on disk")
    h.add_argument("--rank-by", choices=["hrt-density"], default=None)
    h.add_argument("--top-up-to", type=int, default=None, help="stop at N usable rows (implies --resume)")
    sub.add_parser("merge", parents=[common], help="dedupe window manifests -> hrt_all.csv")
    se = sub.add_parser("screen-export", parents=[common], help="blinded relevance-screen batches for one window")
    se.add_argument("--window", required=True)
    se.add_argument("--batch-size", type=int, default=None, help="items per batch (default: screen.batch_size)")
    se.add_argument("--overwrite", action="store_true", help="re-export a window that was already exported")
    si = sub.add_parser("screen-import", parents=[common], help="validate + un-blind labels -> relevance_labels.csv")
    si.add_argument("--window", action="append", default=None, help="limit to these windows (repeatable)")
    si.add_argument("--screener", default="claude-code-subagent", help="screener id when a label line has none")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    if args.cmd in ("harvest", "screen-export") and args.window not in cfg["windows"]:
        ap.error(f"--window must be one of {list(cfg['windows'])}")
    commands = {"count": cmd_count, "harvest": cmd_harvest, "merge": cmd_merge,
                "screen-export": cmd_screen_export, "screen-import": cmd_screen_import}
    commands[args.cmd](cfg, Path(args.data_root), args)


if __name__ == "__main__":
    main()
