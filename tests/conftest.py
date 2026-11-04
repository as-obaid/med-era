"""Shared fixtures: the real corpus config, a JATS builder and a fake HTTP session."""
from pathlib import Path

import pytest

from medera.config import load_yaml

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root():
    return REPO


@pytest.fixture
def cfg():
    return load_yaml(REPO / "configs" / "corpus.yaml")


def _jats(body="<p>Body text.</p>", title="Hormone therapy in women", article_type="research-article",
          lang="en", abstract="<p>Abstract text.</p>", back=""):
    lang_attr = f' xml:lang="{lang}"' if lang else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<article xmlns:mml="http://www.w3.org/1998/Math/MathML" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink"{lang_attr} article-type="{article_type}">'
        "<front><journal-meta><journal-title-group><journal-title>Menopause Journal</journal-title>"
        "</journal-title-group></journal-meta><article-meta>"
        '<article-id pub-id-type="pmcid">PMC1</article-id>'
        f"<title-group><article-title>{title}</article-title></title-group>"
        f"<abstract>{abstract}</abstract></article-meta></front>"
        f"<body>{body}</body><back>{back}</back></article>"
    ).encode("utf-8")


@pytest.fixture
def make_jats():
    return _jats


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b"", headers=None):
        self.status_code = status_code
        self._json = json_data
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self._json


class FakeSession:
    """Serves queued responses (or raises queued exceptions) and records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {}),
                           "timeout": timeout})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def fake_http():
    return FakeSession, FakeResponse


@pytest.fixture(autouse=True)
def _no_tracking(monkeypatch):
    """Tests never talk to W&B: no key, tracking disabled."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "disabled")
