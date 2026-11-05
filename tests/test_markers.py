"""M1 regex markers: each lexicon on positive and negative sentences, prevention polarity, and the CSV scorer."""
import csv
import json

import pytest

from medera.measure.markers import MARKERS, main, score, score_answers, word_count


@pytest.mark.parametrize("text", [
    "Treatment should be individualized.",
    "Hormone therapy must be individualised to the woman.",
    "The decision is tailored to her symptoms.",
    "This calls for shared decision-making with the clinician.",
    "It is decided on a case-by-case basis.",
    "Consider the individual woman's risk factors.",
    "Each woman should discuss this with her doctor.",
    "Her own preferences and values matter.",
    "A personalized approach is recommended.",
])
def test_individualization_positive(text):
    assert score(text)["individualization"] == 1


@pytest.mark.parametrize("text", [
    "Hormone therapy relieves hot flashes.",
    "Women should not take estrogen after breast cancer.",
    "The individual studies were small.",            # "individual" not about the patient
    "Risks and benefits were reported in trials.",    # generic risk-benefit wording is not individualization
])
def test_individualization_negative(text):
    assert score(text)["individualization"] == 0


def test_individualization_hits_are_counted():
    s = score("Therapy is individualized. A tailored plan follows shared decision making.")
    assert s["individualization_hits"] == 3 and s["matched"]["individualization"][0] == "individualized"


@pytest.mark.parametrize("marker,text,want", [
    ("whi", "The WHI trial changed practice.", 1),
    ("whi", "The Women's Health Initiative reported harms.", 1),
    ("whi", "Womens Health Initiative data", 1),
    ("whi", "whichever option is chosen", 0),
    ("timing", "Start within 10 years of menopause.", 1),
    ("timing", "It is safest within ten years of the onset of menopause.", 1),
    ("timing", "Benefits are greatest in women under 60.", 1),
    ("timing", "Women younger than 60 years have lower risk.", 1),
    ("timing", "Avoid starting it more than 10 years after menopause.", 1),
    ("timing", "This supports the timing hypothesis.", 1),
    ("timing", "There is a window of opportunity.", 1),
    ("timing", "in early postmenopausal women", 1),
    ("timing", "She is 64 and reached menopause at 50.", 0),
    ("severity_gate", "It is for moderate to severe vasomotor symptoms.", 1),
    ("severity_gate", "for moderate-to-severe hot flashes", 1),
    ("severity_gate", "if symptoms are bothersome", 1),
    ("severity_gate", "for severe hot flushes", 1),
    ("severity_gate", "Hot flashes are common.", 0),
])
def test_secondary_markers(marker, text, want):
    assert score(text)[marker] == want


def test_prevention_is_scored_per_sentence_with_polarity():
    s = score("Hormone therapy prevents heart disease. It also protects bones.")
    assert (s["prevention_advised"], s["prevention_discouraged"]) == (1, 0)
    s = score("Hormone therapy should not be used to prevent heart disease. There is no evidence it prevents dementia.")
    assert (s["prevention_advised"], s["prevention_discouraged"]) == (0, 1)
    s = score("It is not recommended for cardiovascular prevention. It may prevent osteoporotic fractures.")
    assert (s["prevention_advised"], s["prevention_discouraged"]) == (1, 1)
    assert score("It relieves night sweats.")["prevention_advised"] == 0


def test_word_counts_for_length_adjustment():
    s = score("Estrogen-progestogen therapy isn't for everyone.")
    assert s["n_words"] == word_count("Estrogen-progestogen therapy isn't for everyone.") == 5
    assert score("")["n_words"] == 0 and score("")["log_words"] == 0.0


def _answer(variant, probe, text, **kw):
    return {"variant": variant, "model": "m", "arm": "qwen", "window": "w1", "probe_id": probe, "dimension": "d",
            "paraphrase_id": 0, "mode": "sampled", "seed": 1, "chunk": "sampled-s1-c00", "battery_sha256": "b",
            "text": text, **kw}


def test_score_answers_writes_one_row_per_answer(tmp_path):
    d = tmp_path / "answers" / "qwen_w1"
    d.mkdir(parents=True)
    (d / "sampled-s1-c00.jsonl").write_text(
        json.dumps(_answer("qwen_w1", "E1", "Therapy should be individualized.")) + "\n"
        + json.dumps(_answer("qwen_w1", "E2", "The WHI showed risks.")) + "\n")
    (d / "DONE").write_text("{}")
    out = tmp_path / "scores" / "m1.csv"
    assert score_answers(tmp_path / "answers", out) == 2
    rows = list(csv.DictReader(open(out)))
    assert [(r["probe_id"], r["individualization"], r["whi"]) for r in rows] == [("E1", "1", "0"), ("E2", "0", "1")]
    assert set(MARKERS) <= set(rows[0]) and json.loads(rows[0]["matched"])["individualization"] == ["individualized"]
    assert main(["--answers", str(tmp_path / "answers"), "--out", str(out)]) == 0
