"""Flatten Europe PMC JATS XML to training text: title, abstract and body paragraphs only."""
import re
import unicodedata
import xml.etree.ElementTree as ET

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
DROP_TAGS = {"ref-list", "back", "table-wrap", "table", "fig", "fig-group", "disp-formula", "tex-math",
             "supplementary-material"}
# Europe PMC's fullTextXML usually has no <back>: back matter is folded into <body> as sections.
BACK_SEC_TYPES = {"ack", "ref-list", "fn-group", "associated-data", "contrib-info", "supplementary-material",
                  "COI-statement", "data-availability"}
BACK_SEC_TITLE = re.compile(  # untyped body-level back-matter sections, matched on the whole title
    r"(?:[\dIVX]+[.)]?\s+)?(?:acknowledge?ments?|"
    r"(?:credit )?author(?:s|ship)?(?:['’]s?)? contributions?(?: statement)?|contributors|"
    r"(?:conflicts? of interests?|competing interests?)(?: statement)?|"
    r"declarations?(?: of (?:competing )?interests?)?|disclosures?(?:\s.*)?|"
    r"funding(?: statement| information| sources?)?|financial (?:support|disclosures?)|"
    r"data availability(?: statement)?|availability of data(?: and materials?)?|"
    r"ethic(?:s|al)(?: statement| approval| considerations)?(?: and consent to participate)?|"
    r"approval by ethics committee|consent(?: for publication| to participate)?|human and animal rights|"
    r"publisher['’]?s note|generative ai statement|provenance and peer review|"
    r"supplementary (?:materials?|data|information)|supporting information|footnotes|references|"
    r"associated data|contributor information|(?:list of )?abbreviations|glossary)[.:]?", re.IGNORECASE)
BLOCK_TAGS = {"title", "p"}
SPACED_TAGS = {"p", "title", "list-item", "break"}  # nested blocks get a space so words do not glue
# Leftovers once citation xrefs are gone: "[]", "[1-3]" -> "[-]", "( )", "(–)". Parentheses never lose an
# ASCII hyphen, so "ER(-)" survives.
EMPTY_BRACKETS = re.compile(r"\s*(?:\[[\s,;\-–—]*\]|\([\s,;–—]*\))")
SEPARATOR_SUP = re.compile(r"[\s,;–]*")  # e.g. <sup>,</sup> between two removed citation xrefs
CITE_SEPARATOR = re.compile(r"[\s,;\-–—]*")  # tail between two citation xrefs, e.g. "[3],[4]" or "[12]–[14]"
DANGLING_COMMA = re.compile(r",\s*(?=[;)])")  # "(Smith et al., )" once the year xref is gone
SPACE_BEFORE_PUNCT = re.compile(r"(?<=[\w)\]])\s+(?=[.,;:](?:\s|$))")  # "than men ." -> "than men."


def _parse(xml: bytes) -> ET.Element:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise ValueError(f"unparseable XML: {e}") from e
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.rsplit("}", 1)[1]
    return root


def _is_cite(el) -> bool:
    return el is not None and el.tag == "xref" and el.get("ref-type") == "bibr"


def _drop(root: ET.Element, pred) -> None:
    """Remove every element matching pred, keeping its tail text in place (except separators inside a
    citation run)."""
    for parent in list(root.iter()):
        for child in list(parent):
            if not pred(child):
                continue
            kids = list(parent)
            idx = kids.index(child)
            tail = child.tail or ""
            nxt = kids[idx + 1] if idx + 1 < len(kids) else None
            if _is_cite(child) and _is_cite(nxt) and CITE_SEPARATOR.fullmatch(tail):
                tail = ""
            if tail:
                if idx:
                    kids[idx - 1].tail = (kids[idx - 1].tail or "") + tail
                else:
                    parent.text = (parent.text or "") + tail
            parent.remove(child)


def _is_dropped(el) -> bool:
    return (el.tag in DROP_TAGS or el.tag.endswith("math") or _is_cite(el)
            or (el.tag == "sec" and el.get("sec-type") in BACK_SEC_TYPES))


def _is_back_title_sec(el) -> bool:
    title = el.find("title") if el.tag == "sec" else None
    return title is not None and BACK_SEC_TITLE.fullmatch(_clean("".join(title.itertext()))) is not None


def _text(el) -> str:
    parts = [el.text or ""]
    for child in el:
        sep = " " if child.tag in SPACED_TAGS else ""
        parts += [sep, _text(child), sep, child.tail or ""]
    return "".join(parts)


def _clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = DANGLING_COMMA.sub("", EMPTY_BRACKETS.sub("", s))
    s = re.sub(r"\s+", " ", s).strip()
    return SPACE_BEFORE_PUNCT.sub("", s)


def _blocks(el, out: list) -> None:
    for child in el:
        if child.tag in BLOCK_TAGS:  # do not descend: nested p are already inside this block's text
            out.append(_text(child))
        else:
            _blocks(child, out)


def _title(root) -> str:
    t = root.find("front/article-meta/title-group/article-title")
    return _clean(_text(t)) if t is not None else ""


def jats_to_text(xml: bytes) -> str:
    root = _parse(xml)
    for body in root.findall("body"):
        for sec in [c for c in body if _is_back_title_sec(c)]:
            body.remove(sec)
    _drop(root, _is_dropped)
    _drop(root, lambda el: el.tag == "sup" and SEPARATOR_SUP.fullmatch("".join(el.itertext())) is not None)
    raw = [_title(root)]
    for container in root.findall("front/article-meta/abstract") + root.findall("body"):
        _blocks(container, raw)
    return "\n\n".join(b for b in map(_clean, raw) if b)


def article_meta(xml: bytes) -> dict:
    root = _parse(xml)
    return {"article_type": root.get("article-type", ""), "lang": root.get(XML_LANG, ""), "title": _title(root)}
