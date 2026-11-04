"""Europe PMC query strings per window, byte-for-byte those in data/manifests/counts.csv."""
from datetime import date

COUNT_COLUMNS = ["query_id", "window", "date_from", "date_to", "access", "hit_count", "query", "retrieved_on"]


def window_query(cfg: dict, query_id: str, window: str, access: str) -> str:
    date_from, date_to = cfg["windows"][window]
    suffix = cfg["query_suffix"].format(date_from=date_from, date_to=date_to)
    return cfg["queries"][query_id] + suffix + (cfg["access"][access] or "")


def count_rows(client, cfg: dict, today: str | None = None) -> list[dict]:
    """One hitCount per (query, window, access), in the row order of counts.csv."""
    today = today or date.today().isoformat()
    rows = []
    for query_id in cfg["queries"]:
        for window, (date_from, date_to) in cfg["windows"].items():
            for access in cfg["access"]:
                q = window_query(cfg, query_id, window, access)
                rows.append({"query_id": query_id, "window": window, "date_from": date_from, "date_to": date_to,
                             "access": access, "hit_count": client.hit_count(q), "query": q,
                             "retrieved_on": today})
    return rows
