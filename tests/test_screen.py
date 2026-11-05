"""Blinded relevance screen: export, import validation, agreement, and the preprocessing gate."""
import csv
import hashlib
import json

import pytest

from medera.collect import __main__ as cli
from medera.collect.manifest import write_manifest
from medera.collect.screen import cohen_kappa, item_id
from medera.preprocess.__main__ import main as preprocess_main

ABSTRACT = "<p>Postmenopausal women used estradiol for hot flushes.</p><p>© 2013 Elsevier Ireland Ltd.</p>"


def _row(pmcid, window="w1", date="2013-05-01"):
    return {"pmcid": pmcid, "pmid": "111" + pmcid[3:], "doi": f"10.9999/{pmcid}", "title": f"Title {pmcid}",
            "journal": "Maturitas Secret Journal", "first_pub_date": date, "pub_year": date[:4], "window": window,
            "license": "cc by-nc-nd", "matched_terms": "", "query_id": "core", "n_chars": "5000", "sha256": "x",
            "retrieved_on": "2026-11-04"}


def _xml(make_jats, pmcid):
    xml = make_jats(body="<p>" + "Estradiol and the endometrium in women. " * 40 + "</p>",
                    title=f"Hormone therapy study {pmcid[3:]}", abstract=ABSTRACT).decode()
    kwd = "<kwd-group><kwd>menopause</kwd><kwd>estrogen</kwd></kwd-group>"
    return xml.replace("</article-meta>", kwd + "</article-meta>").replace("PMC1<", pmcid + "<").encode()


@pytest.fixture
def harvested(tmp_path, make_jats):
    """25 w1 and 12 w2 articles with manifests and XML on disk, like a finished harvest."""
    for w, n, year in (("w1", 25, "2013"), ("w2", 12, "2018")):
        rows = [_row(f"PMC{w[1]}{i:03d}", w, f"{year}-03-0{1 + i % 9}") for i in range(n)]
        write_manifest(tmp_path / "manifests" / f"hrt_{w}.csv", rows)
        d = tmp_path / "raw" / "hrt" / w
        d.mkdir(parents=True)
        for r in rows:
            (d / f"{r['pmcid']}.xml").write_bytes(_xml(make_jats, r["pmcid"]))
    return tmp_path


def _cli(root, repo_root, *args):
    cli.main([args[0], "--config", str(repo_root / "configs" / "corpus.yaml"), "--data-root", str(root), *args[1:]])


def _key(root):
    with open(root / "raw" / "screen" / "key.csv", newline="") as f:
        return list(csv.DictReader(f))


def _batches(root):
    return sorted((root / "raw" / "screen" / "batches").glob("*.jsonl"))


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_export_is_blinded(harvested, repo_root):
    _cli(harvested, repo_root, "screen-export", "--window", "w1", "--batch-size", "10")
    names = [p.name for p in _batches(harvested)]
    assert names == ["w1_b01.jsonl", "w1_b02.jsonl", "w1_b03.jsonl", "w1_d01.jsonl"]
    key = _key(harvested)
    salt = key[0]["salt"]
    assert len(salt) == 32 and {r["salt"] for r in key} == {salt}
    for path in _batches(harvested):
        raw = path.read_text()
        for leak in ("PMC1", "10.9999", "Maturitas Secret Journal", "2013", "cc by", "Elsevier", "111"):
            assert leak not in raw, (path.name, leak)
        for rec in _lines(path):
            assert set(rec) == {"item_id", "title", "abstract", "keywords"}
            assert rec["abstract"] == "Postmenopausal women used estradiol for hot flushes."
            assert rec["keywords"] == ["menopause", "estrogen"]
    for r in key:
        assert r["item_id"] == hashlib.sha256((salt + r["pmcid"]).encode()).hexdigest()[:12] == item_id(salt, r["pmcid"])


def test_export_counts_shuffle_and_double_subset(harvested, repo_root):
    _cli(harvested, repo_root, "screen-export", "--window", "w1", "--batch-size", "10")
    primary = [rec["item_id"] for p in _batches(harvested) if "_b" in p.name for rec in _lines(p)]
    double = [rec["item_id"] for p in _batches(harvested) if "_d" in p.name for rec in _lines(p)]
    key = {r["item_id"]: r["pmcid"] for r in _key(harvested)}
    assert len(primary) == len(set(primary)) == 25
    assert len(double) == 2 and set(double) <= set(primary)                 # round(10% of 25)
    assert [key[i] for i in primary] != sorted(key[i] for i in primary)    # shuffled, not PMCID order
    first = [p.read_text() for p in _batches(harvested)]
    _cli(harvested, repo_root, "screen-export", "--window", "w1", "--batch-size", "10", "--overwrite")
    assert [p.read_text() for p in _batches(harvested)] == first           # same salt, seed and order
    with pytest.raises(SystemExit):                                         # refuses to clobber silently
        _cli(harvested, repo_root, "screen-export", "--window", "w1")
    _cli(harvested, repo_root, "screen-export", "--window", "w2")
    assert {r["window"] for r in _key(harvested)} == {"w1", "w2"} and len({r["salt"] for r in _key(harvested)}) == 1


def _label_all(root, label="relevant", cls="", skip=None, extra=None, batches=None):
    labels = root / "raw" / "screen" / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    for path in _batches(root):
        if batches and path.stem not in batches:
            continue
        out = []
        for rec in _lines(path):
            if rec["item_id"] == skip:
                continue
            out.append({"item_id": rec["item_id"], "label": label, "off_topic_class": cls, "note": "HRT is the topic"})
        out += extra or []
        (labels / path.name).write_text("".join(json.dumps(o) + "\n" for o in out))


def test_import_unblinds_and_reports_agreement(harvested, repo_root, capsys):
    _cli(harvested, repo_root, "screen-export", "--window", "w1", "--batch-size", "10")
    _label_all(harvested)
    first_item = _lines(_batches(harvested)[0])[0]["item_id"]
    path = harvested / "raw" / "screen" / "labels" / "w1_b01.jsonl"
    lines = path.read_text().splitlines()
    lines[0] = json.dumps({"item_id": first_item, "label": "off_topic", "off_topic_class": "non_menopausal_hrt",
                           "note": "testosterone in men", "screener": "claude-opus-test"})
    path.write_text("\n".join(lines) + "\n")
    _cli(harvested, repo_root, "screen-import")
    with open(harvested / "manifests" / "relevance_labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["pmcid", "window", "label", "off_topic_class", "screener", "screened_on"]
    assert len(rows) == 25 and {r["window"] for r in rows} == {"w1"}
    key = {r["item_id"]: r["pmcid"] for r in _key(harvested)}
    off = [r for r in rows if r["label"] == "off_topic"]
    assert [(r["pmcid"], r["off_topic_class"], r["screener"]) for r in off] == [
        (key[first_item], "non_menopausal_hrt", "claude-opus-test")]
    assert (harvested / "manifests" / "relevance_agreement.csv").exists()
    assert "kappa" in capsys.readouterr().out


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unknown", "illegal_label", "missing_class"])
def test_import_rejects_bad_labels(harvested, repo_root, problem):
    _cli(harvested, repo_root, "screen-export", "--window", "w1", "--batch-size", "10")
    first = _lines(_batches(harvested)[0])[0]["item_id"]
    if problem == "missing":
        _label_all(harvested, skip=first)
    elif problem == "duplicate":
        _label_all(harvested, extra=[{"item_id": first, "label": "relevant", "off_topic_class": "", "note": ""}],
                   batches={"w1_b01"})
        _label_all(harvested, batches={"w1_b02", "w1_b03", "w1_d01"})
    elif problem == "unknown":
        _label_all(harvested, extra=[{"item_id": "deadbeef0000", "label": "relevant", "off_topic_class": "", "note": ""}])
    elif problem == "illegal_label":
        _label_all(harvested, label="maybe")
    else:
        _label_all(harvested, label="off_topic", cls="")
    with pytest.raises(SystemExit) as e:
        _cli(harvested, repo_root, "screen-import")
    assert e.value.code == 1
    assert not (harvested / "manifests" / "relevance_labels.csv").exists()


def test_cohen_kappa():
    a = ["relevant", "relevant", "borderline", "off_topic"]
    b = ["relevant", "borderline", "borderline", "off_topic"]
    assert cohen_kappa(a, b) == pytest.approx((0.75 - 0.3125) / (1 - 0.3125))
    assert cohen_kappa(a, a) == pytest.approx(1.0)
    assert cohen_kappa(["relevant"] * 3, ["relevant"] * 3) == 1.0


def _preprocess_setup(harvested, labels):
    rows = [_row(f"PMC1{i:03d}", "w1") for i in range(4)]
    write_manifest(harvested / "manifests" / "hrt_all.csv", rows)
    with open(harvested / "manifests" / "relevance_labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["pmcid", "window", "label", "off_topic_class", "screener", "screened_on"])
        w.writeheader()
        for pmcid, (label, cls) in labels.items():
            w.writerow({"pmcid": pmcid, "window": "w1", "label": label, "off_topic_class": cls,
                        "screener": "s", "screened_on": "2026-11-04"})


def _excluded(root):
    with open(root / "processed" / "hrt" / "excluded.csv", newline="") as f:
        return {r["pmcid"]: r["reason"] for r in csv.DictReader(f)}


def test_preprocess_excludes_off_topic_and_keeps_borderline(harvested, repo_root):
    _preprocess_setup(harvested, {"PMC1000": ("relevant", ""), "PMC1001": ("borderline", ""),
                                  "PMC1002": ("off_topic", "cancer_endocrine"), "PMC1003": ("relevant", "")})
    preprocess_main(["text", "--config", str(repo_root / "configs" / "corpus.yaml"), "--data-root", str(harvested)])
    excluded = _excluded(harvested)
    assert excluded.get("PMC1002") == "off_topic_screen"
    assert "PMC1001" not in excluded and "PMC1000" not in excluded


def test_preprocess_unlabelled_article_is_a_hard_error(harvested, repo_root):
    _preprocess_setup(harvested, {"PMC1000": ("relevant", ""), "PMC1001": ("borderline", ""),
                                  "PMC1002": ("relevant", "")})               # PMC1003 never screened
    with pytest.raises(SystemExit) as e:
        preprocess_main(["text", "--config", str(repo_root / "configs" / "corpus.yaml"), "--data-root", str(harvested)])
    assert "PMC1003" in str(e.value)
    assert not (harvested / "processed" / "hrt" / "excluded.csv").exists()


def test_preprocess_missing_label_file_is_a_hard_error_unless_opted_out(harvested, repo_root, capsys):
    rows = [_row(f"PMC1{i:03d}", "w1") for i in range(4)]
    write_manifest(harvested / "manifests" / "hrt_all.csv", rows)
    args = ["text", "--config", str(repo_root / "configs" / "corpus.yaml"), "--data-root", str(harvested)]
    with pytest.raises(SystemExit):
        preprocess_main(args)
    preprocess_main(args + ["--no-relevance-screen"])                        # smoke data only, and loud about it
    assert "relevance screen NOT applied" in capsys.readouterr().out
