"""Europe PMC query strings per window, and the OR guard used by `count`.

A query in corpus.yaml is either a plain string (broad) or a list of named parts (core). Parts are
always wrapped in their own parentheses and OR-ed, "((p1) OR (p2) OR ...)", because a bare
`A OR B` inside an outer OR makes Europe PMC mis-parse the query (the 2026-11-04 core bug: the
ungrouped MESH pair cut w1 open-access hits from 676 to 580).
"""
from datetime import date

COUNT_COLUMNS = ["query_id", "window", "date_from", "date_to", "access", "hit_count", "query", "retrieved_on"]
PART_COLUMNS = ["query_id", "part", "window", "date_from", "date_to", "access", "hit_count", "query", "retrieved_on"]


def query_parts(cfg: dict, query_id: str) -> list[dict]:
    q = cfg["queries"][query_id]
    return list(q["parts"]) if isinstance(q, dict) else []


def base_query(cfg: dict, query_id: str) -> str:
    parts = query_parts(cfg, query_id)
    if not parts:
        return cfg["queries"][query_id]
    return "(" + " OR ".join(f"({p['query']})" for p in parts) + ")"


def _suffix(cfg: dict, window: str, access: str) -> str:
    date_from, date_to = cfg["windows"][window]
    return cfg["query_suffix"].format(date_from=date_from, date_to=date_to) + (cfg["access"][access] or "")


def window_query(cfg: dict, query_id: str, window: str, access: str) -> str:
    return base_query(cfg, query_id) + _suffix(cfg, window, access)


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


def part_count_rows(client, cfg: dict, today: str | None = None) -> list[dict]:
    """hitCount of every part of every multi-part query, per (window, access)."""
    today = today or date.today().isoformat()
    rows = []
    for query_id in cfg["queries"]:
        for part in query_parts(cfg, query_id):
            for window, (date_from, date_to) in cfg["windows"].items():
                for access in cfg["access"]:
                    q = f"({part['query']})" + _suffix(cfg, window, access)
                    rows.append({"query_id": query_id, "part": part["name"], "window": window,
                                 "date_from": date_from, "date_to": date_to, "access": access,
                                 "hit_count": client.hit_count(q), "query": q, "retrieved_on": today})
    return rows


def or_guard(totals: list[dict], parts: list[dict]) -> list[str]:
    """An OR must return at least as many hits as its largest part; anything else is a parser bug."""
    fails = []
    for t in totals:
        key = (t["query_id"], t["window"], t["access"])
        same = [p for p in parts if (p["query_id"], p["window"], p["access"]) == key]
        if not same:
            continue
        top = max(same, key=lambda p: int(p["hit_count"]))
        if int(t["hit_count"]) < int(top["hit_count"]):
            fails.append(f"{t['query_id']} {t['window']} {t['access']}: {t['hit_count']} hits < part "
                         f"{top['part']} {top['hit_count']} (an OR cannot return fewer than its largest part)")
    return fails
