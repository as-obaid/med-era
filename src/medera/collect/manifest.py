"""Manifest CSV schema, IO and the cross-window merge."""
import csv
import os
import re
from pathlib import Path

COLUMNS = ["pmcid", "pmid", "doi", "title", "journal", "first_pub_date", "pub_year", "window", "license",
           "matched_terms", "query_id", "n_chars", "sha256", "retrieved_on"]
DROPPED_COLUMNS = ["pmcid", "window", "reason", "kept_pmcid"]


def pmcid_key(pmcid: str):
    """Sort PMCIDs by number (PMC9 < PMC10), not as strings."""
    digits = pmcid[3:]
    return (int(digits), pmcid) if digits.isdigit() else (float("inf"), pmcid)


def norm_title(title: str) -> str:
    return re.sub(r"[\W_]+", " ", (title or "").lower()).strip()


def read_manifest(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def write_manifest(path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _inside(row, windows, w) -> bool:
    lo, hi = windows[w]
    return lo <= row["first_pub_date"] <= hi


def merge(rows_by_window: dict[str, list[dict]], windows: dict[str, tuple[str, str]],
          dropped: list | None = None) -> list[dict]:
    """Dedupe by PMCID (within and across windows, resolved by date), then by DOI (keep the first), then by
    normalized title (keep the earliest first_pub_date, ties to the lowest PMCID). Every row stays in its window.
    Each dropped row is appended to `dropped` as {pmcid, window, reason, kept_pmcid}."""
    def drop(r, w, reason, kept_pmcid):
        if dropped is not None:
            dropped.append({"pmcid": r["pmcid"], "window": w, "reason": reason, "kept_pmcid": kept_pmcid})

    per_window = {}
    for w, rows in rows_by_window.items():
        seen, kept = set(), []
        for r in rows:
            if r["pmcid"] in seen:
                drop(r, w, "duplicate_pmcid", r["pmcid"])
                continue
            seen.add(r["pmcid"])
            kept.append(r)
        per_window[w] = kept

    homes = {}
    for w, rows in per_window.items():
        for r in rows:
            homes.setdefault(r["pmcid"], []).append(w)
    owner = {}
    for pmcid, ws in homes.items():
        if len(ws) == 1:
            continue
        inside = [w for w in ws if _inside(next(r for r in per_window[w] if r["pmcid"] == pmcid), windows, w)]
        if len(inside) != 1:
            raise ValueError(f"{pmcid} is in windows {ws} and its date does not resolve it (inside: {inside})")
        owner[pmcid] = inside[0]

    survivors, dois = [], {}
    for w, rows in per_window.items():
        for r in rows:
            if owner.get(r["pmcid"], w) != w:
                drop(r, w, "cross_window_date", r["pmcid"])
                continue
            if not _inside(r, windows, w):
                raise ValueError(f"{r['pmcid']} first_pub_date {r['first_pub_date']} is outside {w} {windows[w]}")
            doi = (r.get("doi") or "").strip().lower()
            if doi:
                if doi in dois:
                    drop(r, w, "duplicate_doi", dois[doi])
                    continue
                dois[doi] = r["pmcid"]
            survivors.append((w, r))

    by_title = {}
    for _, r in survivors:
        if norm_title(r.get("title")):
            by_title.setdefault(norm_title(r["title"]), []).append(r)
    winner = {}
    for rows in by_title.values():
        best = min(rows, key=lambda r: (r["first_pub_date"], pmcid_key(r["pmcid"])))
        winner.update({id(r): best["pmcid"] for r in rows if r is not best})
    out = []
    for w, r in survivors:
        if id(r) in winner:
            drop(r, w, "duplicate_title", winner[id(r)])
        else:
            out.append(r)

    pmcids = [r["pmcid"] for r in out]
    if len(pmcids) != len(set(pmcids)):
        raise ValueError("a PMCID survived merge in two windows")
    return out
