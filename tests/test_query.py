import csv

from medera.collect.query import COUNT_COLUMNS, count_rows, window_query


def _counts(repo_root):
    with open(repo_root / "data" / "manifests" / "counts.csv", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def test_window_query_matches_counts_csv(cfg, repo_root):
    _, rows = _counts(repo_root)
    assert len(rows) == 12
    for row in rows:
        assert window_query(cfg, row["query_id"], row["window"], row["access"]) == row["query"], row


class _CountClient:
    def __init__(self):
        self.queries = []

    def hit_count(self, q):
        self.queries.append(q)
        return 7


def test_count_rows_match_counts_csv_layout(cfg, repo_root):
    header, committed = _counts(repo_root)
    client = _CountClient()
    rows = count_rows(client, cfg, today="2026-11-04")
    assert COUNT_COLUMNS == header
    assert [list(r) for r in rows] == [header] * len(committed)
    key = ("query_id", "window", "date_from", "date_to", "access", "query")
    assert [tuple(r[k] for k in key) for r in rows] == [tuple(r[k] for k in key) for r in committed]
    assert all(r["hit_count"] == 7 and r["retrieved_on"] == "2026-11-04" for r in rows)
    assert client.queries == [r["query"] for r in committed]


# --- Structural fix for the 2026-11-04 core-query bug -------------------------------------------
# The old core ended in a bare `... OR MESH:"Hormone Replacement Therapy" OR MESH:"Estrogen Replacement
# Therapy")`. Europe PMC mis-parsed it: w1 OA returned 580 (identical to a plain full-text phrase search),
# while core without the MESH pair returned 657. The tested fix groups the MESH pair.
FIXED_CORE = (
    '((TITLE_ABS:"hormone replacement therapy" OR TITLE_ABS:"menopausal hormone therapy" OR '
    'TITLE_ABS:"estrogen replacement therapy" OR TITLE_ABS:"oestrogen replacement therapy") OR '
    '((TITLE_ABS:"hormone therapy" OR TITLE_ABS:"estrogen therapy" OR TITLE_ABS:"oestrogen therapy") AND '
    'TITLE_ABS:(menopause OR menopausal OR postmenopausal OR perimenopausal OR climacteric)) OR '
    '(KW:"hormone replacement therapy" OR KW:"menopausal hormone therapy" OR KW:"estrogen replacement therapy" OR '
    'KW:"oestrogen replacement therapy") OR '
    '(MESH:"Hormone Replacement Therapy" OR MESH:"Estrogen Replacement Therapy"))'
)
BUGGY_TAIL = ' OR MESH:"Hormone Replacement Therapy" OR MESH:"Estrogen Replacement Therapy")'


def test_core_is_the_tested_fix_exactly(cfg):
    from medera.collect.query import base_query

    assert base_query(cfg, "core") == FIXED_CORE
    assert not base_query(cfg, "core").endswith(BUGGY_TAIL)


def test_every_core_part_is_parenthesized_and_mesh_grouped(cfg):
    from medera.collect.query import base_query, query_parts

    built = base_query(cfg, "core")
    parts = query_parts(cfg, "core")
    assert [p["name"] for p in parts] == ["title_abs_phrases", "generic_and_menopause", "kw_phrases", "mesh"]
    assert built == "(" + " OR ".join(f"({p['query']})" for p in parts) + ")"
    for p in parts:
        assert f"({p['query']})" in built
    assert '(MESH:"Hormone Replacement Therapy" OR MESH:"Estrogen Replacement Therapy")' in built
    assert query_parts(cfg, "broad") == []                      # plain string queries have no parts
    assert base_query(cfg, "broad") == cfg["queries"]["broad"]


def test_part_count_rows_cover_every_part_window_access(cfg):
    from medera.collect.query import PART_COLUMNS, part_count_rows

    client = _CountClient()
    rows = part_count_rows(client, cfg, today="2026-11-04")
    assert len(rows) == 4 * 3 * 2 and list(rows[0]) == PART_COLUMNS
    first = rows[0]
    assert first["query_id"] == "core" and first["part"] == "title_abs_phrases" and first["window"] == "w1"
    assert first["query"].startswith('(TITLE_ABS:"hormone replacement therapy"')
    assert first["query"].endswith(" AND FIRST_PDATE:[2011-01-01 TO 2015-12-31] AND LANG:eng")


def test_or_guard_fails_when_core_returns_fewer_than_a_part():
    from medera.collect.query import or_guard

    totals = [{"query_id": "core", "window": "w1", "access": "oa_fulltext", "hit_count": 580}]
    parts = [{"query_id": "core", "part": "title_abs_phrases", "window": "w1", "access": "oa_fulltext", "hit_count": 657},
             {"query_id": "core", "part": "mesh", "window": "w1", "access": "oa_fulltext", "hit_count": 24}]
    fails = or_guard(totals, parts)
    assert len(fails) == 1 and "580" in fails[0] and "657" in fails[0] and "title_abs_phrases" in fails[0]
    totals[0]["hit_count"] = 676
    assert or_guard(totals, parts) == []


def test_count_cli_writes_parts_and_exits_nonzero_on_guard(monkeypatch, tmp_path, repo_root):
    import pytest

    from medera.collect import __main__ as cli

    class Client:
        def hit_count(self, q):
            return 5 if q.startswith("((") else 9   # every part beats the full core query: parser bug

    monkeypatch.setattr(cli, "make_client", lambda cfg, **kw: Client())
    with pytest.raises(SystemExit) as e:
        cli.main(["count", "--config", str(repo_root / "configs" / "corpus.yaml"), "--data-root", str(tmp_path)])
    assert e.value.code == 1
    assert (tmp_path / "manifests" / "counts.csv").exists()
    assert (tmp_path / "manifests" / "counts_parts.csv").exists()
