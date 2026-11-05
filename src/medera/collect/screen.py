"""Blinded relevance screen (docs/relevance_screen.md).

export: every harvested article of a window becomes one blinded item, {item_id, title, abstract, keywords},
taken from the local XML. No PMCID, DOI, journal, date or license. item_id = sha256(salt + pmcid)[:12]. Items
are shuffled with a fixed seed and written in batches; a reproducible 10% subset is exported again as
double-screen batches. The salt and the pmcid <-> item_id map live in <data-root>/raw/screen/key.csv.

import: validates one label per exported item and batch, un-blinds with key.csv, and writes
manifests/relevance_labels.csv plus manifests/relevance_agreement.csv (primary vs double screen).
"""
import csv
import hashlib
import json
import random
import re
import secrets
from collections import Counter
from datetime import date
from pathlib import Path

from medera.collect.manifest import read_manifest
from medera.preprocess.jats import screen_fields

LABELS = ("relevant", "borderline", "off_topic")
OFF_TOPIC_CLASSES = ("non_menopausal_hrt", "cancer_endocrine", "contraception", "passing_mention", "other")
KEY_COLUMNS = ["item_id", "pmcid", "window", "batch", "salt"]
LABEL_COLUMNS = ["pmcid", "window", "label", "off_topic_class", "screener", "screened_on"]
AGREEMENT_COLUMNS = ["pmcid", "window", "primary_label", "double_label", "agree", "primary_class", "double_class"]
BATCH = re.compile(r"^(?P<window>.+)_(?P<kind>[bd])(?P<n>\d{2,})$")
MAX_NOTE_WORDS = 10


def item_id(salt: str, pmcid: str) -> str:
    return hashlib.sha256((salt + pmcid).encode()).hexdigest()[:12]


def _screen_dir(cfg: dict, root: Path) -> Path:
    return root / cfg["screen"]["dir"]


def _read_key(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, columns)
        w.writeheader()
        w.writerows(rows)


def _chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def export_window(cfg: dict, root: Path, window: str, batch_size: int, overwrite: bool = False) -> list[tuple[str, int]]:
    """Write <window>_bNN.jsonl (all items) and <window>_dNN.jsonl (double-screen subset); return (batch, n)."""
    sc, corpus = cfg["screen"], cfg["corpus"]
    sdir = _screen_dir(cfg, root)
    batches_dir, key_path = sdir / "batches", sdir / "key.csv"
    key = _read_key(key_path)
    if any(r["window"] == window for r in key) and not overwrite:
        raise SystemExit(f"[screen] {window} is already exported (key.csv); pass --overwrite to re-export it")
    salt = key[0]["salt"] if key else secrets.token_hex(16)
    key = [r for r in key if r["window"] != window]
    for old in batches_dir.glob(f"{window}_[bd][0-9]*.jsonl"):  # only this window's own generated batches
        old.unlink()

    pmcids = sorted({r["pmcid"] for r in read_manifest(root / "manifests" / f"{corpus}_{window}.csv")})
    xml_dir = root / "raw" / corpus / window
    missing = [p for p in pmcids if not (xml_dir / f"{p}.xml").exists()]
    if missing:
        raise SystemExit(f"[screen] {len(missing)} {window} XML files missing (first: {missing[:5]})")
    order = list(pmcids)
    random.Random(sc["shuffle_seed"]).shuffle(order)
    double = random.Random(sc["double_seed"]).sample(pmcids, round(sc["double_frac"] * len(pmcids)))

    batches_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for kind, items in (("b", order), ("d", double)):
        for n, chunk in enumerate(_chunks(items, batch_size), start=1):
            name = f"{window}_{kind}{n:02d}"
            with open(batches_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for pmcid in chunk:
                    rec = {"item_id": item_id(salt, pmcid), **screen_fields((xml_dir / f"{pmcid}.xml").read_bytes())}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    key.append({"item_id": rec["item_id"], "pmcid": pmcid, "window": window, "batch": name, "salt": salt})
            written.append((name, len(chunk)))
    _write_csv(key_path, KEY_COLUMNS, key)
    return written


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _check_record(rec: dict) -> str | None:
    label, cls = rec.get("label"), rec.get("off_topic_class") or ""
    if label not in LABELS:
        return f"illegal label {label!r} (use one of {', '.join(LABELS)})"
    if label == "off_topic" and cls not in OFF_TOPIC_CLASSES:
        return f"off_topic needs off_topic_class in {', '.join(OFF_TOPIC_CLASSES)}, got {cls!r}"
    if label != "off_topic" and cls:
        return f"off_topic_class {cls!r} is only allowed with label off_topic"
    return None


def import_labels(cfg: dict, root: Path, windows: list[str] | None = None, screener: str = "claude-code-subagent"):
    """Returns (label_rows, agreement_rows, errors, warnings). Nothing is written when errors is non-empty."""
    sdir = _screen_dir(cfg, root)
    key = [r for r in _read_key(sdir / "key.csv") if windows is None or r["window"] in windows]
    if not key:
        return [], [], [f"no exported items in {sdir / 'key.csv'} for windows {windows or 'any'}"], []
    by_batch: dict[str, dict[str, dict]] = {}
    for r in key:
        by_batch.setdefault(r["batch"], {})[r["item_id"]] = r
    errors, warnings, primary, double = [], [], {}, {}
    for batch in sorted(by_batch):
        items, path = by_batch[batch], sdir / "labels" / f"{batch}.jsonl"
        if not path.exists():
            errors.append(f"{batch}: label file missing ({len(items)} items unlabelled)")
            continue
        file_date = date.fromtimestamp(path.stat().st_mtime).isoformat()
        seen = set()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"{batch}:{lineno}: not valid JSON")
                continue
            iid = rec.get("item_id")
            if iid not in items:
                errors.append(f"{batch}:{lineno}: unknown item_id {iid!r} (not exported in this batch)")
                continue
            if iid in seen:
                errors.append(f"{batch}:{lineno}: duplicate item_id {iid}")
                continue
            seen.add(iid)
            problem = _check_record(rec)
            if problem:
                errors.append(f"{batch}:{lineno}: {problem}")
                continue
            if len((rec.get("note") or "").split()) > MAX_NOTE_WORDS:
                warnings.append(f"{batch}:{lineno}: note longer than {MAX_NOTE_WORDS} words")
            k = items[iid]
            row = {"pmcid": k["pmcid"], "window": k["window"], "label": rec["label"],
                   "off_topic_class": rec.get("off_topic_class") or "", "screener": rec.get("screener") or screener,
                   "screened_on": rec.get("screened_on") or file_date}
            (double if BATCH.match(batch).group("kind") == "d" else primary)[iid] = row
        unlabelled = sorted(set(items) - seen)
        if unlabelled:
            errors.append(f"{batch}: {len(unlabelled)} exported items have no label (e.g. {unlabelled[:3]})")
    labels = sorted(primary.values(), key=lambda r: (r["window"], r["pmcid"]))
    agreement = [{"pmcid": p["pmcid"], "window": p["window"], "primary_label": p["label"],
                  "double_label": double[i]["label"], "agree": int(p["label"] == double[i]["label"]),
                  "primary_class": p["off_topic_class"], "double_class": double[i]["off_topic_class"]}
                 for i, p in primary.items() if i in double]
    agreement.sort(key=lambda r: (r["window"], r["pmcid"]))
    return labels, agreement, errors, warnings


def agreement_summary(agreement: list[dict]) -> list[tuple[str, int, float, float]]:
    """(window or 'all', n, agreement rate, Cohen's kappa) on the three-way label."""
    out = []
    groups = {w: [r for r in agreement if r["window"] == w] for w in sorted({r["window"] for r in agreement})}
    for name, rows in list(groups.items()) + [("all", agreement)]:
        a, b = [r["primary_label"] for r in rows], [r["double_label"] for r in rows]
        out.append((name, len(rows), sum(r["agree"] for r in rows) / len(rows) if rows else float("nan"),
                    cohen_kappa(a, b)))
    return out


def write_import(cfg: dict, root: Path, labels: list[dict], agreement: list[dict]) -> tuple[Path, Path]:
    lp, ap = root / "manifests" / "relevance_labels.csv", root / "manifests" / "relevance_agreement.csv"
    _write_csv(lp, LABEL_COLUMNS, labels)
    _write_csv(ap, AGREEMENT_COLUMNS, agreement)
    return lp, ap
