"""Minimal, polite Europe PMC REST client (search with cursorMark, full-text JATS XML)."""
import time
from collections.abc import Iterator

import requests

RETRY_ERRORS = (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError)


class EuropePMC:
    def __init__(self, base_url, user_agent, session=None, sleep=time.sleep, max_retries=6, backoff_s=1.0,
                 timeout_s=60, pause_s=0.2):
        self.base_url = base_url.rstrip("/")
        self.headers = {"User-Agent": user_agent}
        self.session = session if session is not None else requests.Session()
        self.sleep = sleep
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.timeout_s = timeout_s
        self.pause_s = pause_s
        self._n_requests = 0

    def _get(self, url, params=None, missing_500_after=None):
        """GET with retries on 429, 5xx, connection errors and timeouts. Returns 200 and 404 responses, and a
        500 once `missing_500_after` retries have been spent on it (Europe PMC's answer for missing full text)."""
        if self._n_requests and self.pause_s:
            self.sleep(self.pause_s)
        self._n_requests += 1
        last = None
        for attempt in range(self.max_retries + 1):
            retry_after = None
            try:
                resp = self.session.get(url, params=params, headers=self.headers, timeout=self.timeout_s)
            except RETRY_ERRORS as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code in (200, 404):
                    return resp
                if resp.status_code == 500 and missing_500_after is not None and attempt >= missing_500_after:
                    return resp
                if resp.status_code != 429 and resp.status_code < 500:
                    raise RuntimeError(f"HTTP {resp.status_code} for {url}")
                last = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("Retry-After")
            if attempt == self.max_retries:
                break
            try:
                wait = float(retry_after)
            except (TypeError, ValueError):
                wait = self.backoff_s * 2 ** attempt
            self.sleep(wait)
        raise RuntimeError(f"giving up on {url} after {self.max_retries} retries ({last})")

    def _search_page(self, q, page_size, cursor, result_type="core"):
        params = {"query": q, "format": "json", "resultType": result_type, "pageSize": page_size,
                  "cursorMark": cursor}
        resp = self._get(f"{self.base_url}/search", params)
        if resp.status_code != 200:
            raise RuntimeError(f"search returned HTTP {resp.status_code}")
        return resp.json()

    def hit_count(self, q: str) -> int:
        return int(self._search_page(q, 1, "*", result_type="lite")["hitCount"])

    def search(self, q: str, page_size: int = 1000) -> Iterator[dict]:
        cursor = "*"
        while True:
            data = self._search_page(q, page_size, cursor)
            results = (data.get("resultList") or {}).get("result") or []
            yield from results
            nxt = data.get("nextCursorMark")
            if not results or not nxt or nxt == cursor:
                return
            cursor = nxt

    def fulltext_xml(self, pmcid: str) -> bytes | None:
        resp = self._get(f"{self.base_url}/{pmcid}/fullTextXML", missing_500_after=1)
        return None if resp.status_code in (404, 500) else resp.content
