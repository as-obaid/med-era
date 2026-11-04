import csv

import pytest

from medera.collect.__main__ import main
from medera.collect.manifest import COLUMNS, append_row, merge, read_manifest, write_manifest

WINDOWS = {"w1": ("2011-01-01", "2015-12-31"), "w2": ("2016-01-01", "2020-12-31"),
           "w3": ("2021-01-01", "2026-12-31")}


def row(pmcid, window, date, doi="", n_chars=1000, title=""):
    r = {c: "" for c in COLUMNS}
    r.update(pmcid=pmcid, window=window, first_pub_date=date, doi=doi, n_chars=str(n_chars), title=title)
    return r


def ids(rows):
    return [(r["pmcid"], r["window"]) for r in rows]


def test_columns_exact():
    assert COLUMNS == ["pmcid", "pmid", "doi", "title", "journal", "first_pub_date", "pub_year", "window",
                       "license", "matched_terms", "query_id", "n_chars", "sha256", "retrieved_on"]


def test_append_row_writes_header_once(tmp_path):
    path = tmp_path / "m" / "hrt_w1.csv"
    append_row(path, row("PMC1", "w1", "2012-01-01", doi='10.1/"x", y'))
    append_row(path, row("PMC2", "w1", "2013-01-01"))
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(COLUMNS) and len(lines) == 3
    rows = read_manifest(path)
    assert [r["pmcid"] for r in rows] == ["PMC1", "PMC2"]
    assert rows[0]["doi"] == '10.1/"x", y' and list(rows[0]) == COLUMNS


def test_write_manifest_roundtrip(tmp_path):
    path = tmp_path / "hrt_all.csv"
    write_manifest(path, [row("PMC1", "w1", "2012-01-01")])
    write_manifest(path, [row("PMC2", "w2", "2017-01-01")])
    assert ids(read_manifest(path)) == [("PMC2", "w2")]
    write_manifest(path, [])
    assert path.read_text().strip() == ",".join(COLUMNS)


def test_merge_dedupes_pmcid_within_window():
    out = merge({"w1": [row("PMC1", "w1", "2012-01-01"), row("PMC1", "w1", "2012-01-01"),
                        row("PMC2", "w1", "2013-01-01")]}, WINDOWS)
    assert ids(out) == [("PMC1", "w1"), ("PMC2", "w1")]


def test_merge_dedupes_doi_fallback():
    out = merge({"w1": [row("PMC1", "w1", "2012-01-01", doi="10.1/ABC"),
                        row("PMC2", "w1", "2012-01-01", doi="10.1/abc"),
                        row("PMC3", "w1", "2012-01-01"), row("PMC4", "w1", "2012-01-01")],
                 "w2": [row("PMC5", "w2", "2017-01-01", doi="10.1/Abc")]}, WINDOWS)
    assert ids(out) == [("PMC1", "w1"), ("PMC3", "w1"), ("PMC4", "w1")]


def test_merge_resolves_cross_window_by_date():
    out = merge({"w1": [row("PMC1", "w1", "2016-03-01"), row("PMC2", "w1", "2014-01-01")],
                 "w2": [row("PMC1", "w2", "2016-03-01")]}, WINDOWS)
    assert ids(out) == [("PMC2", "w1"), ("PMC1", "w2")]


def test_merge_raises_on_unresolvable_cross_window():
    with pytest.raises(ValueError, match="PMC1"):
        merge({"w1": [row("PMC1", "w1", "2030-01-01")], "w2": [row("PMC1", "w2", "2030-01-01")]}, WINDOWS)


def test_merge_raises_on_date_outside_window():
    with pytest.raises(ValueError, match="PMC7"):
        merge({"w1": [row("PMC7", "w1", "2017-01-01")]}, WINDOWS)


def test_merge_dedupes_title_keeps_earliest_then_lowest_pmcid():
    dropped = []
    out = merge({"w1": [row("PMC50", "w1", "2013-02-01", title="Hormone Therapy: A Review."),
                        row("PMC40", "w1", "2013-02-01", title="hormone therapy -- a review"),
                        row("PMC8", "w1", "2014-01-01", title="Hormone therapy, a review"),
                        row("PMC1", "w1", "2012-01-01"), row("PMC2", "w1", "2012-01-01")],
                 "w2": [row("PMC3", "w2", "2017-01-01", title="HORMONE THERAPY A REVIEW")]}, WINDOWS, dropped)
    assert ids(out) == [("PMC40", "w1"), ("PMC1", "w1"), ("PMC2", "w1")]  # empty titles are never deduped
    assert [(d["pmcid"], d["window"], d["reason"], d["kept_pmcid"]) for d in dropped] == [
        ("PMC50", "w1", "duplicate_title", "PMC40"), ("PMC8", "w1", "duplicate_title", "PMC40"),
        ("PMC3", "w2", "duplicate_title", "PMC40")]


def test_merge_reports_dropped_rows_with_reasons():
    dropped = []
    merge({"w1": [row("PMC1", "w1", "2012-01-01", doi="10.1/X"), row("PMC1", "w1", "2012-01-01"),
                  row("PMC2", "w1", "2016-03-01")],
           "w2": [row("PMC2", "w2", "2016-03-01"), row("PMC3", "w2", "2017-01-01", doi="10.1/x")]}, WINDOWS, dropped)
    assert [(d["pmcid"], d["window"], d["reason"], d["kept_pmcid"]) for d in dropped] == [
        ("PMC1", "w1", "duplicate_pmcid", "PMC1"), ("PMC2", "w1", "cross_window_date", "PMC2"),
        ("PMC3", "w2", "duplicate_doi", "PMC1")]


def test_merge_output_has_no_overlap():
    out = merge({"w1": [row("PMC1", "w1", "2012-01-01"), row("PMC2", "w1", "2016-06-01")],
                 "w2": [row("PMC2", "w2", "2016-06-01"), row("PMC3", "w2", "2018-01-01")],
                 "w3": [row("PMC3", "w3", "2018-01-01"), row("PMC4", "w3", "2022-01-01")]}, WINDOWS)
    pmcids = [r["pmcid"] for r in out]
    assert len(pmcids) == len(set(pmcids)) == 4
    for r in out:
        lo, hi = WINDOWS[r["window"]]
        assert lo <= r["first_pub_date"] <= hi


def test_merge_cli_warns_on_missing_window(tmp_path, capsys, repo_root, make_jats):
    cfg_path = str(repo_root / "configs" / "corpus.yaml")
    with pytest.raises(SystemExit):
        main(["merge", "--config", cfg_path, "--data-root", str(tmp_path)])
    write_manifest(tmp_path / "manifests" / "hrt_w1.csv", [
        row("PMC1", "w1", "2012-01-01", n_chars=4000, title="A"), row("PMC2", "w1", "2013-01-01", n_chars=4000, title="a"),
        row("PMC3", "w1", "2013-01-01", n_chars=4000, title="C")])
    raw = tmp_path / "raw" / "hrt" / "w1"
    raw.mkdir(parents=True)
    (raw / "PMC1.xml").write_bytes(make_jats(body="<p>" + "Estradiol lowered hot flushes. " * 40 + "</p>"))
    (raw / "PMC3.xml").write_bytes(make_jats(body="<p>" + "Estradiol lowered hot flushes. " * 40 + "</p>",
                                             article_type="correction"))
    main(["merge", "--config", cfg_path, "--data-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "WARNING" in out and "hrt_w2.csv" in out and "hrt_w3.csv" in out
    assert ids(read_manifest(tmp_path / "manifests" / "hrt_all.csv")) == [("PMC1", "w1"), ("PMC3", "w1")]
    assert "est_tokens=2,000" in out  # 8000 chars / chars_per_token 4.0
    assert "usable(filters)=1" in out  # PMC3 is a correction, so the filters reject it
    assert "PMC2" in out and "duplicate_title" in out
    with open(tmp_path / "manifests" / "merge_dropped.csv", newline="") as f:
        assert list(csv.DictReader(f)) == [
            {"pmcid": "PMC2", "window": "w1", "reason": "duplicate_title", "kept_pmcid": "PMC1"}]
