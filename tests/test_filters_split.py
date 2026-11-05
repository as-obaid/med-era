import csv
import json
import random

from medera.collect.manifest import COLUMNS, write_manifest
from medera.preprocess.__main__ import main
from medera.preprocess.filters import exclusion_reason, is_usable
from medera.preprocess.near_dup import near_duplicates
from medera.preprocess.split import heldout_split

META = {"article_type": "research-article", "lang": "en", "title": "Hormone therapy and bone"}


def test_exclusion_reasons(cfg):
    f = cfg["filters"]
    ok = "x" * 800
    assert exclusion_reason(ok, META, f) is None
    assert exclusion_reason(ok, {**META, "article_type": "correction"}, f) == "article_type"
    assert exclusion_reason(ok, {**META, "title": "Erratum: Hormone therapy"}, f) == "title"
    assert exclusion_reason(ok, {**META, "title": "  Retraction notice"}, f) == "title"
    assert exclusion_reason(ok, {**META, "title": "Corrections to dose in HRT"}, f) is None
    assert exclusion_reason(ok, {**META, "lang": "de"}, f) == "language"
    assert exclusion_reason(ok, {**META, "lang": "EN"}, f) is None
    assert exclusion_reason(ok, {**META, "lang": ""}, f) is None
    assert exclusion_reason("x" * 799, META, f) == "short"


def test_meeting_abstracts_exclusion(cfg):
    f = cfg["filters"]
    ok = "x" * 800
    for t in ("Scientific Abstracts", "Abstracts of the 2014 NAMS meeting", "Proceedings of the IMS",
              "Poster session II", "Oral presentations", "Oral Abstracts", "ESHRE 2015 Congress Abstracts",
              "Annual Meeting Abstracts"):
        assert exclusion_reason(ok, {**META, "title": t}, f) == "meeting_abstracts", t
    for at in ("abstract", "meeting-report", "collection", "Abstract"):
        assert exclusion_reason(ok, {**META, "article_type": at}, f) == "meeting_abstracts", at
    assert exclusion_reason(ok, {**META, "title": "How abstracts misreport hormone therapy trials"}, f) is None
    assert exclusion_reason(ok, {**META, "title": "Abstraction of HRT records"}, f) is None
    # Order: article_type -> title -> language -> meeting_abstracts -> short.
    assert exclusion_reason(ok, {**META, "article_type": "correction", "title": "Scientific Abstracts"}, f) == "article_type"
    assert exclusion_reason(ok, {**META, "lang": "de", "title": "Scientific Abstracts"}, f) == "language"
    assert exclusion_reason("x", {**META, "title": "Scientific Abstracts"}, f) == "meeting_abstracts"


def test_is_usable(cfg, make_jats):
    body = "<p>" + "Estradiol lowered hot flushes in the trial. " * 30 + "</p>"
    assert is_usable(make_jats(body=body), cfg["filters"])
    assert not is_usable(make_jats(body="<p>tiny</p>"), cfg["filters"])
    assert not is_usable(make_jats(body=body, article_type="correction"), cfg["filters"])
    assert not is_usable(make_jats(body=body, title="Scientific Abstracts"), cfg["filters"])
    assert not is_usable(b"<article>", cfg["filters"])


def words(seed, n=1000):
    rng = random.Random(seed)
    return [f"w{rng.randrange(5000)}" for _ in range(n)]


def test_near_duplicates_identical_and_edited_docs():
    base = words(1)
    tail_edit = base[:900] + words(99, 100)  # last 10% of the words rewritten
    scattered = [w if i % 100 else "changed" for i, w in enumerate(base)]  # 1 word in 100 changed
    docs = {"PMC1": ("2013-01-01", " ".join(base)), "PMC2": ("2017-05-01", " ".join(base)),
            "PMC3": ("2014-01-01", " ".join(tail_edit)), "PMC4": ("2012-06-01", " ".join(words(2))),
            "PMC5": ("2019-01-01", " ".join(scattered)), "PMC6": ("2020-01-01", " ".join(words(3)))}
    pairs = near_duplicates(docs)
    assert [(k, d) for k, d, _ in pairs] == [("PMC1", "PMC2"), ("PMC1", "PMC3"), ("PMC1", "PMC5")]
    jac = {d: j for _, d, j in pairs}
    assert jac["PMC2"] == 1.0 and 0.8 <= jac["PMC3"] < 0.85 and 0.85 < jac["PMC5"] < 0.95


def test_near_duplicates_tie_drops_higher_pmcid_and_is_deterministic():
    text = " ".join(words(5))
    docs = {"PMC10": ("2015-01-01", text), "PMC9": ("2015-01-01", text), "PMC77": ("2015-02-01", " ".join(words(6)))}
    assert near_duplicates(docs) == [("PMC9", "PMC10", 1.0)]
    assert near_duplicates(dict(reversed(list(docs.items())))) == near_duplicates(docs)
    assert near_duplicates({"PMC1": ("2015-01-01", " ".join(words(7))), "PMC2": ("2015-01-01", " ".join(words(8)))}) == []


def test_heldout_split_deterministic():
    ids = [f"PMC{i}" for i in range(100)]
    train, held = heldout_split(ids, 0.05, 42)
    assert len(held) == 5 and len(train) == 95
    assert not set(train) & set(held) and set(train) | set(held) == set(ids)
    assert heldout_split(list(reversed(ids)), 0.05, 42) == (train, held)
    assert heldout_split(ids, 0.05, 42) == (train, held)
    assert heldout_split(ids, 0.05, 7)[1] != held
    assert train == sorted(train) and held == sorted(held)


def test_heldout_split_small():
    assert heldout_split(["PMC1", "PMC2"], 0.05, 42)[1] != []
    assert len(heldout_split(["PMC1", "PMC2"], 0.05, 42)[1]) == 1
    assert heldout_split(["PMC1"], 0.05, 42) == (["PMC1"], [])
    assert heldout_split([], 0.05, 42) == ([], [])


def _row(pmcid, window, date):
    r = {c: "" for c in COLUMNS}
    r.update(pmcid=pmcid, window=window, first_pub_date=date)
    return r


def _body(seed):
    return "<p>" + " ".join(words(seed, 300)) + "</p>"


def test_text_cli_writes_jsonl_and_excluded(tmp_path, repo_root, make_jats):
    docs = {"PMC1": make_jats(body=_body(1)), "PMC2": make_jats(body=_body(2)),
            "PMC3": make_jats(body=_body(3), article_type="correction"),
            "PMC4": make_jats(body="<p>tiny</p>"), "PMC5": b"<article>", "PMC6": make_jats(body=_body(6)),
            "PMC7": make_jats(body=_body(1)), "PMC8": make_jats(body="<p>tiny</p>")}
    windows = {"PMC1": "w1", "PMC2": "w1", "PMC3": "w1", "PMC4": "w1", "PMC5": "w1", "PMC6": "w2", "PMC7": "w2",
               "PMC8": "w3"}
    dates = {"PMC1": "2013-01-01", "PMC7": "2017-01-01", "PMC6": "2018-01-01", "PMC8": "2022-01-01"}
    for pmcid, xml in docs.items():
        d = tmp_path / "raw" / "hrt" / windows[pmcid]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{pmcid}.xml").write_bytes(xml)
    write_manifest(tmp_path / "manifests" / "hrt_all.csv",
                   [_row(p, windows[p], dates.get(p, "2014-01-01"))
                    for p in ("PMC2", "PMC1", "PMC3", "PMC4", "PMC5", "PMC6", "PMC7", "PMC8")])
    with open(tmp_path / "manifests" / "relevance_labels.csv", "w", newline="") as f:  # every article screened
        f.write("pmcid,window,label,off_topic_class,screener,screened_on\n")
        f.writelines(f"{p},{windows[p]},relevant,,s,2026-11-04\n" for p in docs)
    # Smoke path: exclusions re-derived from the XML (the real corpus uses hrt_excluded.csv instead).
    main(["text", "--config", str(repo_root / "configs" / "corpus.yaml"), "--data-root", str(tmp_path),
          "--no-relevance-screen"])
    out = tmp_path / "processed" / "hrt"
    w1 = [json.loads(line) for line in (out / "w1.jsonl").read_text().splitlines()]
    assert [d["pmcid"] for d in w1] == ["PMC1", "PMC2"]
    assert sorted(d["split"] for d in w1) == ["heldout", "train"]
    assert w1[0]["text"].startswith("Hormone therapy in women\n\n")
    w2 = [json.loads(line) for line in (out / "w2.jsonl").read_text().splitlines()]
    assert [(d["pmcid"], d["split"]) for d in w2] == [("PMC6", "train")]
    assert (out / "w3.jsonl").read_text() == ""
    with open(out / "excluded.csv", newline="") as f:
        excluded = list(csv.DictReader(f))
    # PMC7 (w2) repeats PMC1 (w1) word for word: the later window loses it. PMC4 and PMC8 are both "tiny"
    # but already excluded as short, so they are not near-duplicate candidates.
    assert [(r["pmcid"], r["window"], r["reason"]) for r in excluded] == [
        ("PMC3", "w1", "article_type"), ("PMC4", "w1", "short"), ("PMC5", "w1", "parse_error"),
        ("PMC7", "w2", "near_duplicate"), ("PMC8", "w3", "short")]
    with open(out / "near_duplicates.csv", newline="") as f:
        nd = list(csv.DictReader(f))
    assert [(r["kept_pmcid"], r["kept_window"], r["dropped_pmcid"], r["dropped_window"]) for r in nd] == [
        ("PMC1", "w1", "PMC7", "w2")]
    assert float(nd[0]["jaccard"]) == 1.0
