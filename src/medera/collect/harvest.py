"""Harvest one window: search, download full-text XML, write the window manifest."""
import hashlib
import html
import os
import re
from datetime import date
from pathlib import Path

from medera.collect.manifest import COLUMNS, append_row, pmcid_key, read_manifest, write_manifest
from medera.collect.query import window_query
from medera.preprocess.filters import is_usable
from medera.preprocess.jats import article_license, jats_to_text

TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>")  # HTML tags only, so "p < 0.05" survives
WORD = re.compile(r"\w+")


def _plain(s) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG.sub(" ", s or ""))).strip()


def _title_abstract(result) -> str:
    return _plain(result.get("title")) + " " + _plain(result.get("abstractText"))


def _phrase_re(phrases):
    alts = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    return re.compile(rf"\b(?:{alts})\b", re.IGNORECASE)


def matched_terms(result: dict, cfg: dict) -> list[str]:
    keywords = (result.get("keywordList") or {}).get("keyword") or []
    text = _title_abstract(result) + " " + " ".join(_plain(k) for k in keywords)
    found = {p for p in cfg["hrt_phrases"] if re.search(rf"\b{re.escape(p)}\b", text, re.IGNORECASE)}
    mesh = {(m.get("descriptorName") or "") for m in (result.get("meshHeadingList") or {}).get("meshHeading") or []}
    found |= {f"MeSH:{h}" for h in cfg.get("hrt_mesh", []) if h in mesh}
    return sorted(found)


def hrt_density(result: dict, cfg: dict) -> float:
    """HRT phrase + abbreviation hits per 1000 words of title + abstract."""
    text = _title_abstract(result)
    words = len(WORD.findall(text))
    if not words:
        return 0.0
    hits = len(_phrase_re(cfg["hrt_phrases"]).findall(text))
    abbr = "|".join(re.escape(a) for a in cfg["hrt_abbreviations"])
    hits += len(re.findall(rf"\b(?:{abbr})\b", text))
    return hits * 1000 / words


def manifest_row(result: dict, window: str, query_id: str, xml: bytes, retrieved_on: str, cfg: dict) -> dict:
    """Raises ValueError when the XML does not parse."""
    journal = ((result.get("journalInfo") or {}).get("journal") or {}).get("title")
    row = {
        "pmcid": result.get("pmcid") or "",
        "pmid": result.get("pmid") or "",
        "doi": result.get("doi") or "",
        "title": _plain(result.get("title")),
        "journal": _plain(journal),
        "first_pub_date": result.get("firstPublicationDate") or "",
        "pub_year": result.get("pubYear") or "",
        "window": window,
        "license": result.get("license") or article_license(xml),
        "matched_terms": "; ".join(matched_terms(result, cfg)),
        "query_id": query_id,
        "n_chars": len(jats_to_text(xml)),
        "sha256": hashlib.sha256(xml).hexdigest(),
        "retrieved_on": retrieved_on,
    }
    return {c: row[c] for c in COLUMNS}


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _file_date(path: Path) -> str:
    return date.fromtimestamp(path.stat().st_mtime).isoformat()


def _skip_no_pmcid(result, stats) -> None:
    stats["no_pmcid"] += 1
    print(f"[harvest] skipping a record without PMCID: id={result.get('id')} source={result.get('source')} "
          f"title={_plain(result.get('title'))[:70]!r}", flush=True)


def harvest(client, cfg, window, query_id, data_root, limit=None, resume=False, rank_by=None, top_up_to=None,
            today=None, top_up_add=None) -> dict:
    """Top-up modes (both imply --resume; PMCIDs already in the manifest are never re-added):
    - top_up_to=N: stop once the window manifest holds N usable rows in total, across all query_ids.
    - top_up_add=N: append up to N NEW usable rows, ranked by title+abstract HRT density (implies
      rank_by="hrt-density"; zero-density candidates are never added).
    "Usable" = passes the Task 5 filters (filters.is_usable); the relevance screen comes later."""
    data_root = Path(data_root)
    corpus = cfg["corpus"]
    xml_dir = data_root / "raw" / corpus / window
    xml_dir.mkdir(parents=True, exist_ok=True)
    man_path = data_root / "manifests" / f"{corpus}_{window}.csv"
    filters = cfg["filters"]
    today = today or date.today().isoformat()
    lo, hi = cfg["windows"][window]
    if top_up_to is not None and top_up_add is not None:
        raise ValueError("use either top_up_to or top_up_add, not both")
    if top_up_add is not None:
        rank_by = rank_by or "hrt-density"
    if top_up_to is not None or top_up_add is not None:
        resume = True

    existing = read_manifest(man_path) if resume and man_path.exists() else []
    kept = [r for r in existing if (xml_dir / f"{r['pmcid']}.xml").exists()]
    if not resume or len(kept) != len(existing):  # fresh run, or drop rows whose XML vanished
        write_manifest(man_path, kept)
    done = {r["pmcid"] for r in kept}
    # Usable = passes the same filters as `preprocess text`; existing rows are judged from the XML on disk.
    usable_by_pmcid = {r["pmcid"]: is_usable((xml_dir / f"{r['pmcid']}.xml").read_bytes(), filters) for r in kept}
    usable = sum(usable_by_pmcid.values())
    if top_up_add is not None:
        top_up_to = usable + top_up_add
    # written = downloaded this run; reused = rebuilt from XML already on disk (resume only, never refetched)
    stats = {"written": 0, "reused": 0, "not_found": 0, "fetch_errors": 0, "parse_failures": 0, "no_pmcid": 0}

    q = window_query(cfg, query_id, window, cfg["harvest_access"])
    page_size = cfg["polite"]["page_size"]
    if limit and not resume and not rank_by:
        page_size = min(page_size, limit)
    candidates = client.search(q, page_size=page_size)
    if rank_by == "hrt-density":
        pool = {}
        for r in candidates:  # fetch every search result before downloading anything
            if not r.get("pmcid"):
                _skip_no_pmcid(r, stats)
            elif r["pmcid"] not in done:
                pool.setdefault(r["pmcid"], r)
        ranked = sorted(pool.values(), key=lambda r: (-hrt_density(r, cfg), pmcid_key(r["pmcid"])))
        candidates = [r for r in ranked if hrt_density(r, cfg) > 0]  # never add a zero-density document
        stats["zero_density_skipped"] = len(ranked) - len(candidates)
        print(f"[harvest] {window}/{query_id}: {len(candidates)} ranked candidates not yet in the manifest "
              f"({stats['zero_density_skipped']} with zero title+abstract HRT density are never added)", flush=True)
    elif rank_by is not None:
        raise ValueError(f"unknown rank_by {rank_by!r}")

    for result in candidates:
        if limit is not None and stats["written"] + stats["reused"] >= limit:  # --limit = max NEW manifest rows
            break
        if top_up_to is not None and usable >= top_up_to:
            break
        pmcid = result.get("pmcid")
        if not pmcid:
            _skip_no_pmcid(result, stats)
            continue
        if pmcid in done:
            continue
        local = xml_dir / f"{pmcid}.xml"
        if resume and local.exists():
            xml = local.read_bytes()
            try:
                row = manifest_row(result, window, query_id, xml, _file_date(local), cfg)
            except ValueError:
                row = None  # unparseable local copy: fetch it again below
            if row is not None:
                append_row(man_path, row)
                done.add(pmcid)
                stats["reused"] += 1
                usable_by_pmcid[pmcid] = is_usable(xml, filters)
                usable += usable_by_pmcid[pmcid]
                continue
        try:
            xml = client.fulltext_xml(pmcid)
        except RuntimeError as e:  # retries exhausted; a later --resume run tries it again
            stats["fetch_errors"] += 1
            print(f"[harvest] {pmcid}: fetch failed: {e}", flush=True)
            continue
        if xml is None:
            stats["not_found"] += 1
            print(f"[harvest] {pmcid}: full text not found (404/500)", flush=True)
            continue
        try:
            row = manifest_row(result, window, query_id, xml, today, cfg)
        except ValueError:
            stats["parse_failures"] += 1
            continue
        _write_atomic(xml_dir / f"{pmcid}.xml", xml)
        append_row(man_path, row)
        done.add(pmcid)
        stats["written"] += 1
        usable_by_pmcid[pmcid] = is_usable(xml, filters)
        usable += usable_by_pmcid[pmcid]
        if stats["written"] % 50 == 0:
            print(f"[harvest] {window}/{query_id}: {stats['written']} written", flush=True)

    rows = read_manifest(man_path)
    usable_rows = [r for r in rows if usable_by_pmcid.get(r["pmcid"])]
    stats.update(
        manifest_rows=len(rows),
        usable=len(usable_rows),
        est_tokens=sum(int(r["n_chars"] or 0) for r in rows) / cfg["chars_per_token"],
        usable_est_tokens=sum(int(r["n_chars"] or 0) for r in usable_rows) / cfg["chars_per_token"],
        missing_license=sum(not r["license"] for r in rows),
        outside_window=sum(not (lo <= r["first_pub_date"] <= hi) for r in rows),
    )
    print(f"[harvest] {window}/{query_id} this run: written={stats['written']} reused_local={stats['reused']} "
          f"not_found(404/500)={stats['not_found']} fetch_errors={stats['fetch_errors']} "
          f"xml_parse_failures={stats['parse_failures']} no_pmcid={stats['no_pmcid']}", flush=True)
    print(f"[harvest] {man_path}: rows={stats['manifest_rows']} usable(filters)={stats['usable']} "
          f"est_tokens={stats['est_tokens']:,.0f} usable_est_tokens={stats['usable_est_tokens']:,.0f} "
          f"missing_license={stats['missing_license']} dates_outside_{window}={stats['outside_window']}", flush=True)
    return stats
