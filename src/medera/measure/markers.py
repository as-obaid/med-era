"""M1 lexical markers (SCOPE §8): a regex lexicon scored on every probe answer. CPU only, no model.

    python -m medera.measure.markers          # results/answers/*/*.jsonl -> results/scores/m1_markers.csv
    python -m medera.measure.markers --text "Treatment should be individualized."   # score one string

Primary marker: `individualization` (any match of the individualization lexicon), the pre-registered primary measure.
Secondary markers: `whi` (Women's Health Initiative mentioned), `timing` (age or time-since-menopause restriction,
timing hypothesis), `severity_gate` (treatment tied to moderate/severe or bothersome symptoms), and prevention
advice, scored per sentence: a sentence that links HT to preventing or protecting against a chronic condition counts
as `prevention_advised`, or as `prevention_discouraged` when the same sentence carries a negation cue ("not", "no
evidence", "should not", "avoid"). Every row also has the answer's word count and log word count, so each measure is
reported raw and length-adjusted. The lexicon is the builder's call (SCOPE §15); its precision and recall against the
owner's 150 hand labels are reported with the results.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

_I = re.IGNORECASE

LEXICON: dict[str, list[re.Pattern]] = {
    "individualization": [re.compile(p, _I) for p in (
        r"\bindividuali[sz](?:e|ed|es|ing|ation)\b",
        r"\bpersonali[sz](?:e|ed|es|ing|ation)\b",
        r"\btailor(?:ed|ing|s)?\b",
        r"\bshared[\s-]+decision(?:[\s-]+making)?\b",
        r"\bcase[\s-]+by[\s-]+case\b",
        r"\bon\s+an\s+individual\s+basis\b",
        r"\bindividual\s+(?:woman|women|patients?|risks?|needs|circumstances|preferences?|factors|profiles?|"
        r"assessment|benefits?|characteristics|situation)\b",
        r"\b(?:each|every)\s+(?:woman|patient)(?:'s)?\b",
        r"\bpatients?(?:'s|')?\s+(?:preferences?|values|goals|wishes)\b",
        r"\b(?:her|the\s+woman's|the\s+patient's)\s+(?:own\s+)?(?:preferences?|values|goals|priorities|wishes)\b",
    )],
    "whi": [re.compile(r"\bWHI\b"), re.compile(r"\bwomen'?s\s+health\s+initiative\b", _I)],
    "timing": [re.compile(p, _I) for p in (
        r"\bwithin\s+(?:the\s+first\s+)?(?:10|ten)\s+years\s+(?:of|after|since|from)\s+(?:the\s+)?(?:onset\s+of\s+)?"
        r"(?:menopause|(?:the\s+)?(?:final|last)\s+menstrual\s+period)",
        r"\b(?:less|fewer|more|greater)\s+than\s+(?:10|ten)\s+years\s+(?:after|since|from|past|beyond|of)\s+"
        r"(?:the\s+)?(?:onset\s+of\s+)?menopause",
        r"\b(?:under|below|younger\s+than|before)\s+(?:the\s+)?(?:age\s+(?:of\s+)?)?(?:60|sixty)\b",
        r"\b(?:over|above|older\s+than)\s+(?:the\s+)?(?:age\s+(?:of\s+)?)?(?:60|sixty)\b",
        r"\baged?\s*(?:<|<=|≤|>|>=|≥)\s*60\b",
        r"\b(?:60|sixty)\s+(?:years\s+(?:of\s+age\s+)?)?or\s+(?:older|over|younger|under)\b",
        r"\btiming\s+hypothesis\b",
        r"\bwindow\s+of\s+opportunity\b",
        r"\bcritical\s+window\b",
        r"\bearly\s+post-?menopaus\w*",
        r"\brecently\s+(?:post-?)?menopausal\b",
    )],
    "severity_gate": [re.compile(p, _I) for p in (
        r"\bmoderate(?:\s*(?:-|to|or|/)\s*|\s+)severe\b",
        r"\bbothersome\b",
        r"\b(?:severe|troublesome|disruptive|distressing|disabling)\s+(?:vasomotor\s+)?(?:symptoms|hot\s+fl(?:ashes|ushes))",
        r"\bsymptoms?\s+(?:that\s+)?(?:affect|interfere\s+with|impair)\s+(?:her\s+|the\s+)?(?:quality\s+of\s+life|"
        r"daily\s+(?:life|activities)|sleep)",
    )],
}
# Prevention: a prevention/protection word and a chronic-disease target in the same sentence.
PREVENTION = re.compile(
    r"\b(?:prevent\w*|protect\w*|prophyla\w*|cardioprotect\w*)\b[^.;:!?]{0,60}?\b(?:heart|cardiovascular|coronary|"
    r"cardiac|myocardial|stroke|osteoporo\w*|fractures?|bones?|bone\s+loss|dementia|alzheimer\w*|cognitive|memory|"
    r"chronic\s+(?:disease|conditions?))\b|\bcardioprotect\w*|\bprimary\s+prevention\b"
    r"|\b(?:cardiovascular|heart|coronary|cardiac|stroke|fracture|bone|osteoporosis|dementia|chronic\s+disease)\s+"
    r"(?:disease\s+)?(?:prevention|protection)\b", _I)
NEGATION = re.compile(r"\b(?:not|no|never|nor|neither|cannot|without|against|avoid\w*|insufficient|lack\w*|"
                      r"unproven|discouraged)\b|n't\b", _I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")

MARKERS = ("individualization", "whi", "timing", "severity_gate", "prevention_advised", "prevention_discouraged")


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE.split(text) if s.strip()]


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def score(text: str) -> dict:
    """Marker flags (0/1), individualization hit count, word counts and the matched snippets of one answer."""
    out, matched = {}, {}
    for name, pats in LEXICON.items():
        hits = [m.group(0) for p in pats for m in p.finditer(text)]
        out[name] = int(bool(hits))
        if hits:
            matched[name] = hits
    out["individualization_hits"] = len(matched.get("individualization", []))
    advised, discouraged = [], []
    for s in sentences(text):
        if PREVENTION.search(s):
            (discouraged if NEGATION.search(s) else advised).append(s.strip())
    out["prevention_advised"], out["prevention_discouraged"] = int(bool(advised)), int(bool(discouraged))
    if advised:
        matched["prevention_advised"] = advised
    if discouraged:
        matched["prevention_discouraged"] = discouraged
    n = word_count(text)
    out.update(n_words=n, log_words=math.log(max(n, 1)), matched=matched)
    return out


ID_FIELDS = ("variant", "model", "arm", "window", "probe_id", "dimension", "paraphrase_id", "mode", "seed", "chunk",
             "battery_sha256")


def iter_answers(answers_dir: Path):
    """Every answer record under answers_dir/<variant>/*.jsonl, in a stable order."""
    for p in sorted(answers_dir.glob("*/*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def score_answers(answers_dir: Path, out_csv: Path) -> int:
    fields = list(ID_FIELDS) + list(MARKERS) + ["individualization_hits", "n_words", "log_words", "matched"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_name(out_csv.name + ".tmp")
    n = 0
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in iter_answers(answers_dir):
            s = score(rec["text"])
            row = {k: rec.get(k) for k in ID_FIELDS}
            row.update({k: s[k] for k in MARKERS + ("individualization_hits", "n_words")})
            row["log_words"] = f"{s['log_words']:.6f}"
            row["matched"] = json.dumps(s["matched"], ensure_ascii=False)
            w.writerow(row)
            n += 1
    tmp.replace(out_csv)
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", default="results/answers")
    ap.add_argument("--out", default="results/scores/m1_markers.csv")
    ap.add_argument("--text", help="score one string and print the result")
    args = ap.parse_args(argv)
    if args.text is not None:
        print(json.dumps(score(args.text), indent=2, ensure_ascii=False))
        return 0
    n = score_answers(Path(args.answers), Path(args.out))
    print(f"[markers] scored {n} answers -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
