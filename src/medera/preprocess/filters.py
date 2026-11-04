"""Document-level exclusion rules (corrections, retractions, non-English, meeting abstracts, too short)."""
import re

from medera.preprocess.jats import article_meta, jats_to_text

MEETING_ARTICLE_TYPES = {"abstract", "meeting-report", "collection"}
MEETING_TITLE = re.compile(r"^(scientific )?abstracts\b|^proceedings\b|^poster|^oral (presentations|abstracts)"
                           r"|congress abstracts|meeting abstracts", re.IGNORECASE)


def exclusion_reason(text: str, meta: dict, filters: dict) -> str | None:
    article_type = (meta.get("article_type") or "").lower()
    title = meta.get("title") or ""
    if article_type in filters["exclude_article_types"]:
        return "article_type"
    if re.match(filters["exclude_title_regex"], title, re.IGNORECASE):
        return "title"
    lang = (meta.get("lang") or "").lower().split("-")[0]
    if lang and lang not in filters["languages"]:  # no xml:lang: trust the LANG:eng query
        return "language"
    if article_type in MEETING_ARTICLE_TYPES or MEETING_TITLE.search(title):
        return "meeting_abstracts"
    if len(text) < filters["min_chars"]:
        return "short"
    return None


def is_usable(xml: bytes, filters: dict) -> bool:
    """The training-set test, shared by the harvest top-up, the harvest/merge summaries and `preprocess text`."""
    try:
        return exclusion_reason(jats_to_text(xml), article_meta(xml), filters) is None
    except ValueError:
        return False
