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
