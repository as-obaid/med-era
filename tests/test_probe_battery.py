"""The probe battery in configs/probes.yaml: structure, wording rules, prompt format, hash and chunk plan."""
from collections import Counter
from pathlib import Path

from medera.config import load_yaml
from medera.measure.markers import score
from medera.probe.battery import (battery_sha256, build_prompt, gen_settings, plan_chunks, validate, wordings)

REPO = Path(__file__).resolve().parents[1]
CFG = load_yaml(REPO / "configs" / "probes.yaml")


def test_battery_is_valid_and_has_the_planned_size():
    assert validate(CFG) == []
    assert CFG["status"] in ("draft", "frozen") and CFG["prereg_tag"]
    assert len(CFG["probes"]) == 19 and len(wordings(CFG)) == 43
    per_dim = Counter(p["dimension"] for p in CFG["probes"])
    assert set(per_dim) == set(CFG["dimensions"]) and min(per_dim.values()) >= 3
    assert all(len(p["paraphrases"]) >= 1 for p in CFG["probes"])


def test_primer_and_probes_do_not_carry_the_markers_they_are_scored_on():
    primer = score(CFG["format_primer"])
    assert all(primer[m] == 0 for m in ("individualization", "whi", "timing", "severity_gate", "prevention_advised",
                                        "prevention_discouraged"))
    for w in wordings(CFG):
        s = score(w.question)
        assert (s["individualization"], s["whi"], s["timing"], s["severity_gate"]) == (0, 0, 0, 0), w


def test_prompt_is_completion_style_with_no_chat_template():
    p = build_prompt(CFG, "Is transdermal  estrogen safer?")
    assert p.startswith("The following questions and answers") and p.endswith("Question: Is transdermal estrogen "
                                                                               "safer?\nAnswer:")
    assert p.count("Question:") == 3 and "\n\n\n" not in p
    for marker in ("<|im_start|>", "<start_of_turn>", "<|turn>", "[INST]", "<bos>"):
        assert marker not in p


def test_plan_three_seeds_and_a_greedy_pass_in_fixed_chunks():
    plan = plan_chunks(CFG)
    sampled = [c for c in plan if c.mode == "sampled"]
    assert sum(len(c.wordings) for c in sampled) == 129 and sum(len(c.wordings) for c in plan) == 172
    assert {c.seed for c in sampled} == {1, 2, 3} and len({c.id for c in plan}) == len(plan) == 8
    assert len({c.torch_seed for c in sampled}) == len(sampled)
    assert max(len(c.wordings) for c in plan) <= CFG["decoding"]["batch_size"]
    # the same probe/seed/chunk gets the same random stream in every variant: seeds depend on the plan only
    assert [c.torch_seed for c in plan_chunks(CFG)] == [c.torch_seed for c in plan]


def test_decoding_is_fully_explicit():
    s, g = gen_settings(CFG, "sampled"), gen_settings(CFG, "greedy")
    assert s == {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 0, "repetition_penalty": 1.0,
                 "max_new_tokens": 400}
    assert g["do_sample"] is False and g["top_k"] == 0 and g["temperature"] == 1.0 and g["max_new_tokens"] == 400


def test_hash_ignores_status_but_not_wording():
    h = battery_sha256(CFG)
    assert battery_sha256({**CFG, "status": "frozen"}) == h
    changed = {**CFG, "probes": [{**CFG["probes"][0], "original": "Who may take hormone therapy?"}] + CFG["probes"][1:]}
    assert battery_sha256(changed) != h


def test_validate_reports_broken_batteries():
    bad = {**CFG, "status": "maybe", "probes": CFG["probes"][:1] * 2,
           "decoding": {**CFG["decoding"], "sampled": {**CFG["decoding"]["sampled"], "temperature": 0}}}
    problems = " ".join(validate(bad))
    assert "status" in problems and "duplicate probe id E1" in problems and "temperature" in problems
