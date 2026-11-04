import hashlib
import subprocess
import sys

import pytest

from medera.collect.harvest import harvest, hrt_density, manifest_row, matched_terms
from medera.collect.manifest import COLUMNS, read_manifest, write_manifest
from medera.collect.query import window_query
from medera.preprocess.jats import jats_to_text

LONG = "<p>" + "Estradiol was given to postmenopausal women in the trial. " * 20 + "</p>"
SHORT = "<p>Too short.</p>"


def res(pmcid, title="Hormone replacement therapy outcomes", abstract="Women took HRT daily.",
        date="2013-05-01", license="cc by", **extra):
    r = {"pmcid": pmcid, "pmid": pmcid[3:], "doi": f"10.1/{pmcid}", "title": title,
         "journalInfo": {"journal": {"title": "Menopause Journal"}}, "firstPublicationDate": date,
         "pubYear": date[:4], "license": license, "abstractText": abstract}
    r.update(extra)
    return r


class FakeClient:
    def __init__(self, results, xml):
        self.results, self.xml = results, xml
        self.events, self.queries = [], []

    def search(self, q, page_size=1000):
        self.queries.append(q)
        for r in self.results:
            self.events.append(("search", r.get("pmcid")))
            yield r

    def fulltext_xml(self, pmcid):
        self.events.append(("xml", pmcid))
        v = self.xml[pmcid]
        if isinstance(v, Exception):
            raise v
        return v

    def fetched(self):
        return [p for kind, p in self.events if kind == "xml"]


def manifest_path(root, w="w1"):
    return root / "manifests" / f"hrt_{w}.csv"


def test_matched_terms(cfg):
    r = res("PMC1", title="Hormone replacement therapy and <i>bone</i>", abstract="",
            keywordList={"keyword": ["Estrogen Therapy"]},
            meshHeadingList={"meshHeading": [{"descriptorName": "Humans"},
                                             {"descriptorName": "Hormone Replacement Therapy"}]})
    assert matched_terms(r, cfg) == ["MeSH:Hormone Replacement Therapy", "estrogen therapy",
                                     "hormone replacement therapy"]
    assert matched_terms({"pmcid": "PMC2"}, cfg) == []


def test_hrt_density_counts_phrases_and_abbreviations(cfg):
    r = res("PMC1", title="Menopausal hormone therapy and HRT use",
            abstract="<b>Hormone therapy</b> and hrt in MHT users.")
    # 13 words; phrases: "Menopausal hormone therapy" (longest first, no overlap) + "Hormone therapy";
    # abbreviations: HRT, MHT (case-sensitive, so "hrt" does not count).
    assert hrt_density(r, cfg) == pytest.approx(4 * 1000 / 13)
    assert hrt_density({"pmcid": "PMC2"}, cfg) == 0.0


def test_manifest_row_fields(cfg, make_jats):
    xml = make_jats(body=LONG)
    r = res("PMC9", title="Hormone <i>therapy</i>  and menopause", license="")
    row = manifest_row(r, "w1", "core", xml, "2026-11-04", cfg)
    assert list(row) == COLUMNS
    assert row["title"] == "Hormone therapy and menopause"
    assert (row["pmcid"], row["pmid"], row["doi"], row["journal"]) == ("PMC9", "9", "10.1/PMC9", "Menopause Journal")
    assert (row["first_pub_date"], row["pub_year"], row["window"], row["query_id"]) == ("2013-05-01", "2013", "w1", "core")
    assert row["license"] == "" and row["retrieved_on"] == "2026-11-04"
    assert row["matched_terms"] == "hormone therapy"


def test_n_chars_is_clean_text_length(cfg, make_jats):
    xml = make_jats(body=LONG, back="<ref-list><ref><mixed-citation>Ref</mixed-citation></ref></ref-list>")
    row = manifest_row(res("PMC1"), "w1", "core", xml, "2026-11-04", cfg)
    assert row["n_chars"] == len(jats_to_text(xml))
    assert row["n_chars"] != len(xml)
    assert row["sha256"] == hashlib.sha256(xml).hexdigest()


def test_harvest_writes_xml_and_manifest(cfg, make_jats, tmp_path):
    xml = make_jats(body=LONG)
    client = FakeClient([res("PMC1")], {"PMC1": xml})
    summary = harvest(client, cfg, "w1", "core", tmp_path, today="2026-11-04")
    assert client.queries == [window_query(cfg, "core", "w1", "oa_fulltext")]
    assert (tmp_path / "raw" / "hrt" / "w1" / "PMC1.xml").read_bytes() == xml
    assert sorted(p.name for p in (tmp_path / "raw" / "hrt" / "w1").iterdir()) == ["PMC1.xml"]
    header = manifest_path(tmp_path).read_text().splitlines()[0]
    assert header.split(",") == COLUMNS
    rows = read_manifest(manifest_path(tmp_path))
    assert [(r["pmcid"], r["window"], r["query_id"]) for r in rows] == [("PMC1", "w1", "core")]
    assert summary["written"] == 1 and summary["usable"] == 1


def test_harvest_summary_self_checks(cfg, make_jats, tmp_path):
    good = make_jats(body=LONG)
    results = [{"id": "12345", "source": "MED", "title": "No PMCID here"}, res("PMC1"), res("PMC2"), res("PMC3"),
               res("PMC4"), res("PMC5", license="", date="2017-01-01"), res("PMC6")]
    xml = {"PMC1": good, "PMC2": None, "PMC3": b"<article><body>", "PMC4": RuntimeError("500 x7"),
           "PMC5": good, "PMC6": make_jats(body=SHORT)}
    s = harvest(FakeClient(results, xml), cfg, "w1", "core", tmp_path, today="2026-11-04")
    assert (s["written"], s["not_found"], s["parse_failures"], s["fetch_errors"], s["no_pmcid"]) == (3, 1, 1, 1, 1)
    assert (s["missing_license"], s["outside_window"]) == (1, 1)
    assert s["manifest_rows"] == 3 and s["usable"] == 2
    n = len(jats_to_text(good))
    assert s["est_tokens"] == pytest.approx((2 * n + len(jats_to_text(make_jats(body=SHORT)))) / cfg["chars_per_token"])
    assert s["usable_est_tokens"] == pytest.approx(2 * n / cfg["chars_per_token"])
    assert not (tmp_path / "raw" / "hrt" / "w1" / "PMC3.xml").exists()
    assert [r["pmcid"] for r in read_manifest(manifest_path(tmp_path))] == ["PMC1", "PMC5", "PMC6"]


def test_harvest_limit_and_resume_skips_done(cfg, make_jats, tmp_path):
    xml = make_jats(body=LONG)
    results = [res("PMC1"), res("PMC2"), res("PMC3")]
    xmls = {p: xml for p in ("PMC1", "PMC2", "PMC3")}

    s = harvest(FakeClient(results, xmls), cfg, "w1", "core", tmp_path, limit=1)
    assert s["written"] == 1
    assert [r["pmcid"] for r in read_manifest(manifest_path(tmp_path))] == ["PMC1"]

    client = FakeClient(results, xmls)
    harvest(client, cfg, "w1", "core", tmp_path, resume=True)
    assert client.fetched() == ["PMC2", "PMC3"]
    pmcids = [r["pmcid"] for r in read_manifest(manifest_path(tmp_path))]
    assert pmcids == ["PMC1", "PMC2", "PMC3"]

    # A manifest row whose XML vanished is dropped and re-fetched, never duplicated.
    (tmp_path / "raw" / "hrt" / "w1" / "PMC2.xml").unlink()
    client = FakeClient(results, xmls)
    harvest(client, cfg, "w1", "core", tmp_path, resume=True)
    assert client.fetched() == ["PMC2"]
    assert sorted(r["pmcid"] for r in read_manifest(manifest_path(tmp_path))) == ["PMC1", "PMC2", "PMC3"]

    # Without resume the manifest starts fresh.
    harvest(FakeClient(results, xmls), cfg, "w1", "core", tmp_path, limit=1)
    assert [r["pmcid"] for r in read_manifest(manifest_path(tmp_path))] == ["PMC1"]


def test_topup_ranks_by_density_with_pmcid_tiebreak(cfg, make_jats, tmp_path):
    long_xml, short_xml = make_jats(body=LONG), make_jats(body=SHORT)
    corr_xml = make_jats(body=LONG, article_type="correction")  # long enough, but excluded by the filters
    hi = dict(title="HRT and MHT", abstract="HRT HRT")              # 4 hits / 5 words
    mid = dict(title="Menopause and HRT", abstract="Some words here.")  # 1 hit / 6 words
    lo = dict(title="Menopause symptoms", abstract="Hot flushes in many women and HRT was rare overall.")  # 1 / 12
    results = [res("PMC1", **lo), res("PMC4", **mid), res("PMC3", **hi), res("PMC2", **hi),
               res("PMC5", **hi), res("PMC6", title="Menopause", abstract="Sleep."),
               {"id": "999", "source": "MED", "title": "HRT HRT HRT"}]
    xml = {"PMC1": long_xml, "PMC2": short_xml, "PMC3": corr_xml, "PMC4": long_xml,
           "PMC5": long_xml, "PMC6": long_xml, "PMC7": corr_xml}
    # PMC5 (usable) and PMC7 (a correction, so not usable: judged from the XML on disk) come from the core harvest.
    existing = FakeClient([res("PMC5"), res("PMC7")], xml)
    assert harvest(existing, cfg, "w1", "core", tmp_path)["usable"] == 1

    client = FakeClient(results, xml)
    s = harvest(client, cfg, "w1", "broad", tmp_path, rank_by="hrt-density", top_up_to=3)
    # All search results are fetched before any download.
    kinds = [k for k, _ in client.events]
    assert kinds == ["search"] * 7 + ["xml"] * 4
    assert client.queries == [window_query(cfg, "broad", "w1", "oa_fulltext")]
    # PMC5 skipped; PMC2 and PMC3 tie (PMC2 first). PMC2 is short and PMC3 is a correction, so neither is
    # usable; PMC4 and PMC1 bring usable rows to 3, and PMC6 is never fetched.
    assert client.fetched() == ["PMC2", "PMC3", "PMC4", "PMC1"]
    rows = read_manifest(manifest_path(tmp_path))
    assert [(r["pmcid"], r["query_id"]) for r in rows] == [
        ("PMC5", "core"), ("PMC7", "core"), ("PMC2", "broad"), ("PMC3", "broad"), ("PMC4", "broad"),
        ("PMC1", "broad")]
    assert s["usable"] == 3 and s["no_pmcid"] == 1


def test_topup_tiebreak_is_numeric_pmcid(cfg, make_jats, tmp_path):
    xml = make_jats(body=LONG)
    results = [res("PMC10"), res("PMC9"), res("PMC100")]
    client = FakeClient(results, {r["pmcid"]: xml for r in results})
    harvest(client, cfg, "w1", "broad", tmp_path, rank_by="hrt-density", top_up_to=3)
    assert client.fetched() == ["PMC9", "PMC10", "PMC100"]


def test_topup_respects_limit(cfg, make_jats, tmp_path):
    xml = make_jats(body=LONG)
    results = [res(f"PMC{i}") for i in range(1, 6)]
    write_manifest(manifest_path(tmp_path), [])
    client = FakeClient(results, {r["pmcid"]: xml for r in results})
    harvest(client, cfg, "w1", "broad", tmp_path, rank_by="hrt-density", top_up_to=10, limit=2)
    assert client.fetched() == ["PMC1", "PMC2"]


def test_cli_help_lists_subcommands(repo_root):
    out = subprocess.run([sys.executable, "-m", "medera.collect", "--help"], capture_output=True, text=True,
                         cwd=repo_root, env={"PYTHONPATH": str(repo_root / "src")})
    assert out.returncode == 0
    for sub in ("count", "harvest", "merge"):
        assert sub in out.stdout
