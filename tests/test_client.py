import pytest
import requests

from medera.collect.europepmc import EuropePMC

BASE = "https://example.org/rest"


def _client(session, sleeps, **kw):
    kw.setdefault("pause_s", 0)
    return EuropePMC(BASE, "MedEra-collector/0.1", session=session, sleep=sleeps.append, **kw)


def _page(results, next_cursor):
    data = {"hitCount": 3, "resultList": {"result": results}}
    if next_cursor is not None:
        data["nextCursorMark"] = next_cursor
    return data


def test_search_follows_cursor_mark(fake_http):
    Session, Resp = fake_http
    session = Session([
        Resp(json_data=_page([{"pmcid": "PMC1"}], "A")),
        Resp(json_data=_page([{"pmcid": "PMC2"}], "B")),
        Resp(json_data=_page([{"pmcid": "PMC3"}], "B")),
    ])
    out = list(_client(session, []).search("q", page_size=1))
    assert [r["pmcid"] for r in out] == ["PMC1", "PMC2", "PMC3"]
    assert [c["params"]["cursorMark"] for c in session.calls] == ["*", "A", "B"]
    p = session.calls[0]["params"]
    assert session.calls[0]["url"] == f"{BASE}/search"
    assert (p["query"], p["format"], p["resultType"], p["pageSize"]) == ("q", "json", "core", 1)


def test_search_stops_on_empty_page_or_missing_cursor(fake_http):
    Session, Resp = fake_http
    session = Session([Resp(json_data=_page([{"pmcid": "PMC1"}], None))])
    assert len(list(_client(session, []).search("q"))) == 1
    session = Session([Resp(json_data=_page([{"pmcid": "PMC1"}], "A")), Resp(json_data=_page([], "B"))])
    assert len(list(_client(session, []).search("q"))) == 1
    assert len(session.calls) == 2


def test_get_retries_429_and_5xx_with_backoff(fake_http):
    Session, Resp = fake_http
    session = Session([
        Resp(429, headers={"Retry-After": "3"}),
        Resp(503),
        Resp(200, content=b"<article/>"),
    ])
    sleeps = []
    assert _client(session, sleeps, backoff_s=1.0).fulltext_xml("PMC1") == b"<article/>"
    assert sleeps == [3, 2.0]
    assert session.calls[0]["url"] == f"{BASE}/PMC1/fullTextXML"


def test_get_retries_connection_errors_and_timeouts(fake_http):
    Session, Resp = fake_http
    session = Session([requests.ConnectionError("down"), requests.Timeout("slow"), Resp(200, content=b"x")])
    sleeps = []
    assert _client(session, sleeps, backoff_s=0.5).fulltext_xml("PMC1") == b"x"
    assert sleeps == [0.5, 1.0]


def test_get_gives_up_after_max_retries(fake_http):
    Session, Resp = fake_http
    session = Session([Resp(503), Resp(503), Resp(503)])
    sleeps = []
    with pytest.raises(RuntimeError):
        _client(session, sleeps, max_retries=2).fulltext_xml("PMC1")
    assert len(session.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_fulltext_404_returns_none(fake_http):
    Session, Resp = fake_http
    session = Session([Resp(404)])
    sleeps = []
    assert _client(session, sleeps).fulltext_xml("PMC1") is None
    assert sleeps == []


def test_fulltext_500_is_missing_after_one_retry(fake_http):
    # Europe PMC answers 500 (not 404) for an article without full text.
    Session, Resp = fake_http
    session = Session([Resp(500), Resp(500)])
    sleeps = []
    assert _client(session, sleeps).fulltext_xml("PMC1") is None
    assert len(session.calls) == 2 and sleeps == [1.0]


def test_fulltext_500_then_200_returns_body(fake_http):
    Session, Resp = fake_http
    session = Session([Resp(500), Resp(200, content=b"<article/>")])
    assert _client(session, []).fulltext_xml("PMC1") == b"<article/>"


def test_fulltext_502_503_504_keep_full_backoff(fake_http):
    Session, Resp = fake_http
    session = Session([Resp(502), Resp(503), Resp(504), Resp(200, content=b"x")])
    sleeps = []
    assert _client(session, sleeps).fulltext_xml("PMC1") == b"x"
    assert sleeps == [1.0, 2.0, 4.0]


def test_search_500_keeps_full_backoff(fake_http):
    Session, Resp = fake_http
    session = Session([Resp(500), Resp(500), Resp(json_data=_page([{"pmcid": "PMC1"}], None))])
    sleeps = []
    assert [r["pmcid"] for r in _client(session, sleeps).search("q")] == ["PMC1"]
    assert sleeps == [1.0, 2.0]


def test_user_agent_is_generic(fake_http, cfg):
    Session, Resp = fake_http
    session = Session([Resp(json_data={"hitCount": 42})])
    client = EuropePMC(BASE, cfg["user_agent"], session=session, sleep=lambda s: None, timeout_s=9)
    assert client.hit_count("q") == 42
    ua = session.calls[0]["headers"]["User-Agent"]
    assert ua == "MedEra-collector/0.1" and "@" not in ua
    assert session.calls[0]["timeout"] == 9


def test_pause_between_requests(fake_http):
    Session, Resp = fake_http
    session = Session([Resp(json_data={"hitCount": 1}), Resp(json_data={"hitCount": 2})])
    sleeps = []
    client = _client(session, sleeps, pause_s=0.2)
    client.hit_count("a")
    client.hit_count("b")
    assert sleeps == [0.2]


def test_search_idlist_pages_metadata_only(fake_http):
    Session, Resp = fake_http
    session = Session([Resp(json_data=_page([{"pmcid": "PMC1"}], "A")), Resp(json_data=_page([], "A"))])
    out = list(_client(session, []).search("q", page_size=200, result_type="idlist"))
    assert [r["pmcid"] for r in out] == ["PMC1"]
    assert {(c["params"]["resultType"], c["params"]["pageSize"]) for c in session.calls} == {("idlist", 200)}
