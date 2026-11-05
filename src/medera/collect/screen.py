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
# b = primary, d = its double-screen subset; t/td = the same for incremental (top-up) exports.
BATCH = re.compile(r"^(?P<window>.+)_(?P<kind>td|[bdt])(?P<n>\d{2,})$")
DOUBLE_KINDS = {"d", "td"}
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


def _write_batches(batches_dir: Path, xml_dir: Path, window: str, salt: str, groups) -> tuple[list, list]:
    """groups: [(kind, first_number, pmcids, batch_size)]; never overwrites an existing batch file."""
    written, key_rows = [], []
    batches_dir.mkdir(parents=True, exist_ok=True)
    for kind, first, items, batch_size in groups:
        for n, chunk in enumerate(_chunks(items, batch_size), start=first):
            name = f"{window}_{kind}{n:02d}"
            path = batches_dir / f"{name}.jsonl"
            if path.exists():
                raise SystemExit(f"[screen] {path} already exists; refusing to overwrite it")
            with open(path, "w", encoding="utf-8") as f:
                for pmcid in chunk:
                    rec = {"item_id": item_id(salt, pmcid), **screen_fields((xml_dir / f"{pmcid}.xml").read_bytes())}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    key_rows.append({"item_id": rec["item_id"], "pmcid": pmcid, "window": window, "batch": name,
                                     "salt": salt})
            written.append((name, len(chunk)))
    return written, key_rows


def _labelled_pmcids(cfg: dict, root: Path) -> set[str]:
    path = root / cfg["filters"]["relevance_labels"]
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["pmcid"] for r in csv.DictReader(f)}


def export_incremental(cfg: dict, root: Path, window: str, batch_size: int) -> list[tuple[str, int]]:
    """Export only the window's manifest rows that have no relevance label and were never exported, into new
    <window>_tNN batches plus a 10% <window>_tdNN double-screen subset. Same salt and blinding; key.csv rows
    are appended; existing batches and key rows stay byte-identical."""
    sc, corpus = cfg["screen"], cfg["corpus"]
    sdir = _screen_dir(cfg, root)
    batches_dir, key_path = sdir / "batches", sdir / "key.csv"
    key = _read_key(key_path)
    if not key:
        raise SystemExit(f"[screen] {key_path} is missing: run a full screen-export first")
    salt = key[0]["salt"]
    exported = {r["pmcid"] for r in key if r["window"] == window}
    labelled = _labelled_pmcids(cfg, root)
    pmcids = sorted({r["pmcid"] for r in read_manifest(root / "manifests" / f"{corpus}_{window}.csv")}
                    - labelled - exported)
    if not pmcids:
        return []
    xml_dir = root / "raw" / corpus / window
    missing = [p for p in pmcids if not (xml_dir / f"{p}.xml").exists()]
    if missing:
        raise SystemExit(f"[screen] {len(missing)} {window} XML files missing (first: {missing[:5]})")
    order = list(pmcids)
    random.Random(sc["shuffle_seed"]).shuffle(order)
    double = random.Random(sc["double_seed"]).sample(pmcids, round(sc["double_frac"] * len(pmcids)))

    def next_number(kind):
        used = [int(m.group("n")) for b in {r["batch"] for r in key if r["window"] == window}
                if (m := BATCH.match(b)) and m.group("kind") == kind]
        return max(used, default=0) + 1

    written, rows = _write_batches(batches_dir, xml_dir, window, salt,
                                   [("t", next_number("t"), order, batch_size),
                                    ("td", next_number("td"), double, batch_size)])
    with open(key_path, "a", newline="", encoding="utf-8") as f:   # append only: old rows stay byte-identical
        csv.DictWriter(f, KEY_COLUMNS).writerows(rows)
    return written


def export_window(cfg: dict, root: Path, window: str, batch_size: int, overwrite: bool = False) -> list[tuple[str, int]]:
    """Write <window>_bNN.jsonl (all items) and <window>_dNN.jsonl (double-screen subset); return (batch, n)."""
    sc, corpus = cfg["screen"], cfg["corpus"]
    sdir = _screen_dir(cfg, root)
    batches_dir, key_path = sdir / "batches", sdir / "key.csv"
    key = _read_key(key_path)
    re_export = any(r["window"] == window for r in key)
    if re_export and not overwrite:
        raise SystemExit(f"[screen] {window} is already exported (key.csv); pass --overwrite to re-export it")
    salt = key[0]["salt"] if key else secrets.token_hex(16)
    key = [r for r in key if r["window"] != window]
    for old in batches_dir.glob(f"{window}_*.jsonl"):  # only this window's own generated batches
        if BATCH.match(old.stem):
            old.unlink()

    pmcids = sorted({r["pmcid"] for r in read_manifest(root / "manifests" / f"{corpus}_{window}.csv")})
    xml_dir = root / "raw" / corpus / window
    missing = [p for p in pmcids if not (xml_dir / f"{p}.xml").exists()]
    if missing:
        raise SystemExit(f"[screen] {len(missing)} {window} XML files missing (first: {missing[:5]})")
    order = list(pmcids)
    random.Random(sc["shuffle_seed"]).shuffle(order)
    double = random.Random(sc["double_seed"]).sample(pmcids, round(sc["double_frac"] * len(pmcids)))

    written, rows = _write_batches(batches_dir, xml_dir, window, salt,
                                   [("b", 1, order, batch_size), ("d", 1, double, batch_size)])
    if re_export or not key_path.exists():
        _write_csv(key_path, KEY_COLUMNS, key + rows)
    else:  # a new window: append, so every existing key row stays byte-identical
        with open(key_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, KEY_COLUMNS).writerows(rows)
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
    # Provenance of labels already imported: an unchanged label keeps its screener and date on re-import.
    previous = {}
    lp = root / "manifests" / "relevance_labels.csv"
    if lp.exists():
        with open(lp, newline="", encoding="utf-8") as f:
            previous = {r["pmcid"]: r for r in csv.DictReader(f)}
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
            old = previous.get(k["pmcid"])
            is_primary = BATCH.match(batch).group("kind") not in DOUBLE_KINDS
            if (is_primary and old and not rec.get("screener") and not rec.get("screened_on")
                    and (old["label"], old["off_topic_class"]) == (row["label"], row["off_topic_class"])):
                row.update(screener=old["screener"], screened_on=old["screened_on"])
            (double if BATCH.match(batch).group("kind") in DOUBLE_KINDS else primary)[iid] = row
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


def screen_summary(cfg: dict, root: Path, windows: list[str]) -> list[dict]:
    """Per window: manifest rows, labelled/unlabelled, label counts, kept (label in keep_labels AND passes the
    Task 5 filters on the XML) and kept est. tokens (n_chars / chars_per_token), plus the pre-registered top-up
    rule (configs/corpus.yaml: topup_rule) for its window."""
    from medera.preprocess.filters import is_usable

    path = root / cfg["filters"]["relevance_labels"]
    labels = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            labels = {r["pmcid"]: r["label"] for r in csv.DictReader(f)}
    keep, rule, corpus = set(cfg["filters"]["keep_labels"]), cfg["topup_rule"], cfg["corpus"]
    out = []
    for w in windows:
        rows = read_manifest(root / "manifests" / f"{corpus}_{w}.csv")
        xml_dir = root / "raw" / corpus / w
        kept = [r for r in rows if labels.get(r["pmcid"]) in keep
                and is_usable((xml_dir / f"{r['pmcid']}.xml").read_bytes(), cfg["filters"])]
        tokens = sum(int(r["n_chars"] or 0) for r in kept) / cfg["chars_per_token"]
        counts = Counter(labels[r["pmcid"]] for r in rows if r["pmcid"] in labels)
        rule_pass = (len(kept) >= rule["min_kept"] and tokens >= rule["min_est_tokens"]) if w == rule["window"] else None
        out.append({"window": w, "manifest_rows": len(rows), "labelled": sum(counts.values()),
                    "unlabelled": len(rows) - sum(counts.values()), "relevant": counts["relevant"],
                    "borderline": counts["borderline"], "off_topic": counts["off_topic"], "kept": len(kept),
                    "kept_est_tokens": tokens, "rule_pass": rule_pass})
    return out
