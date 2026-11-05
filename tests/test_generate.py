"""Probing on CPU with tiny random models and a character tokenizer: answer splitting, per-row stop strings, seeded and
padding-independent generation, explicit decoding settings, resumable chunks, refusals, and the pre-registration gate
(in a scratch git repo)."""
import json
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from test_perplexity import CountingStopper  # noqa: E402
from test_tiny_models import MODELS, TRAIN, _tiny_gemma4, _tiny_qwen35  # noqa: E402

from medera.config import load_yaml  # noqa: E402
from medera.probe import gate  # noqa: E402
from medera.probe import generate as gen  # noqa: E402
from medera.probe.battery import Chunk, battery_sha256, plan_chunks, wordings  # noqa: E402
from medera.variants import get_variant  # noqa: E402

CFG = load_yaml("configs/probes.yaml")
PAD, EOS, SEP = 0, 1, 2


class CharTok:
    """id = 3 + ord(c) for ASCII; 0 pad, 1 eos, 2 the document separator. Left or right padding."""
    pad_token_id, eos_token_id = PAD, EOS

    def __init__(self):
        self.padding_side = "right"

    def convert_tokens_to_ids(self, t):
        return SEP

    def encode(self, s):
        return [3 + min(ord(c), 250) for c in s]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - 3) if i >= 3 else ("" if skip_special_tokens else f"<{i}>") for i in ids)

    def __call__(self, texts, return_tensors="pt", padding=True):
        enc = [self.encode(t) for t in texts]
        n = max(map(len, enc))
        ids, mask = [], []
        for e in enc:
            pad = [PAD] * (n - len(e))
            ids.append(pad + e if self.padding_side == "left" else e + pad)
            mask.append([0] * len(pad) + [1] * len(e) if self.padding_side == "left" else [1] * len(e) + [0] * len(pad))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def _small_cfg(**dec):
    """The real battery's primer and template, 3 probes (7 wordings), short answers."""
    d = {"sampled": {**CFG["decoding"]["sampled"], "max_new_tokens": 6, "seeds": [1, 2]},
         "greedy": {**CFG["decoding"]["greedy"], "max_new_tokens": 6}, "batch_size": 4}
    d.update(dec)
    return {**CFG, "format_primer": "Primer.", "probes": CFG["probes"][:3], "decoding": d}


def test_split_answer_cuts_at_stop_eos_and_padding():
    tok = CharTok()
    enc = tok.encode
    a = gen.split_answer(enc(" Yes.\nQuestion: next") + [PAD, PAD], tok, ["\nQuestion:"], {EOS, SEP}, PAD, 50)
    assert (a["text"], a["finish"]) == ("Yes.", "stop") and a["text_raw"] == " Yes.\nQuestion: next"
    a = gen.split_answer(enc(" Done") + [SEP, PAD, 9], tok, ["\nQuestion:"], {EOS, SEP}, PAD, 50)
    assert (a["text"], a["finish"], a["n_new_tokens"]) == ("Done", "eos", 5)
    a = gen.split_answer(enc("abcdef"), tok, ["\nQuestion:"], {EOS}, PAD, 6)
    assert a["finish"] == "length"


def test_stop_criteria_is_per_row_and_ignores_the_prompt():
    tok = CharTok()
    rows = [tok.encode("Q?") + tok.encode(" a\nQuestion:"), tok.encode("Q?") + tok.encode(" still going")]
    n = max(map(len, rows))
    x = torch.tensor([r + [PAD] * (n - len(r)) for r in rows])
    crit = gen.stop_criteria(tok, ["\nQuestion:"], prompt_len=2)
    assert crit(x, None).tolist() == [True, False]
    prompt_only = torch.tensor([tok.encode("\nQuestion:")])           # a stop string inside the prompt does not count
    assert gen.stop_criteria(tok, ["\nQuestion:"], prompt_len=10)(prompt_only, None).tolist() == [False]


def _chunk(cfg, mode="greedy", seed=None, n=3):
    return Chunk("c", mode, seed, 0, tuple(wordings(cfg)[:n]))


@pytest.mark.parametrize("builder", [_tiny_qwen35, _tiny_gemma4])
def test_greedy_batch_with_left_padding_equals_one_at_a_time(builder):
    """Left padding must not change an answer: a padded batch gives what each prompt gives alone."""
    cfg, tok, model = _small_cfg(), CharTok(), builder().eval()
    eos = [EOS, SEP]
    together = gen.generate_chunk(model, tok, cfg, _chunk(cfg), eos)
    alone = [gen.generate_chunk(model, tok, cfg, Chunk("c", "greedy", None, 0, (w,)), eos)[0]
             for w in _chunk(cfg).wordings]
    assert [a["text_raw"] for a in together] == [a["text_raw"] for a in alone]


def test_sampling_is_seeded_and_the_models_own_defaults_never_leak():
    cfg, tok, model = _small_cfg(), CharTok(), _tiny_qwen35().eval()
    model.generation_config.top_k = 1                    # a model default that must not survive
    c1 = Chunk("s", "sampled", 1, 0, _chunk(cfg).wordings)
    first = gen.generate_chunk(model, tok, cfg, c1, [EOS, SEP])
    assert model.generation_config.top_k == 0 and model.generation_config.temperature == 0.7
    assert gen.generate_chunk(model, tok, cfg, c1, [EOS, SEP]) == first
    other = gen.generate_chunk(model, tok, cfg, Chunk("s", "sampled", 2, 0, c1.wordings), [EOS, SEP])
    assert [a["text_raw"] for a in other] != [a["text_raw"] for a in first]


def test_eos_ids_union():
    tok, model = CharTok(), _tiny_qwen35()
    model.generation_config.eos_token_id = [7, EOS]
    assert gen.eos_ids(tok, model, MODELS["qwen35_27b"]) == [EOS, SEP, 7]


def _run(tmp_path, cfg, stopper, loads, meta=None, variant="qwen_w1"):
    v = get_variant(TRAIN, variant)

    def loader():
        loads.append(1)
        return _tiny_qwen35().eval(), CharTok()
    meta = meta or {"adapter": "qwen_w1", "adapter_sha256": "a" * 64, "model_revision": "rev",
                    "prereg": {"tag": "prereg-v1", "commit": "c0ffee"}}
    return gen.run_variant(v, cfg, loader, tmp_path / "answers" / v.name, stopper, {"git_sha": "abc", "gpu": "cpu"},
                           meta, MODELS["qwen35_27b"], progress=tmp_path / "progress.json", log=lambda m: None)


def test_chunks_are_written_once_resumed_and_complete(tmp_path):
    cfg, loads = _small_cfg(), []
    plan = plan_chunks(cfg)
    assert [c.id for c in plan] == ["sampled-s1-c00", "sampled-s1-c01", "sampled-s2-c00", "sampled-s2-c01",
                                    "greedy-c00", "greedy-c01"]
    assert _run(tmp_path, cfg, CountingStopper(2), loads) == "wall"
    out = tmp_path / "answers" / "qwen_w1"
    assert sorted(p.name for p in out.iterdir()) == ["sampled-s1-c00.jsonl", "sampled-s1-c01.jsonl"]
    assert json.loads((tmp_path / "progress.json").read_text())["step"] == 2
    assert _run(tmp_path, cfg, CountingStopper(99), loads) == "done" and len(loads) == 2
    recs = [r for c in plan for r in gen.read_chunk(out / f"{c.id}.jsonl")]
    assert len(recs) == 7 * 3 and (out / "DONE").exists()
    r = recs[0]
    assert (r["variant"], r["probe_id"], r["paraphrase_id"], r["mode"], r["seed"], r["chunk_seed"]) == \
        ("qwen_w1", "E1", 0, "sampled", 1, 1000)
    assert r["decoding"]["top_k"] == 0 and r["battery_sha256"] == battery_sha256(cfg)
    assert r["provenance"]["git_sha"] == "abc" and r["model_revision"] == "rev" and r["adapter_sha256"] == "a" * 64
    assert r["prereg"]["tag"] == "prereg-v1" and r["finish"] in ("stop", "eos", "length", "other")
    assert {(x["probe_id"], x["paraphrase_id"]) for x in recs if x["mode"] == "greedy"} == \
        {(x["probe_id"], x["paraphrase_id"]) for x in recs if x["seed"] == 2}
    assert _run(tmp_path, cfg, CountingStopper(99), loads) == "done" and len(loads) == 2   # idempotent: no model


def test_answers_with_unicode_line_separators_stay_one_record_per_line(tmp_path, monkeypatch):
    """A real answer can contain U+0085 or U+2028, which str.splitlines() splits on: records must survive."""
    cfg, loads = _small_cfg(), []
    monkeypatch.setattr(gen, "generate_chunk", lambda model, tok, cfg, c, eos: [
        {"text": "a\x85b\u2028c", "text_raw": "a\x85b\u2028c", "finish": "length", "n_new_tokens": 6}
        for _ in c.wordings])
    assert _run(tmp_path, cfg, CountingStopper(99), loads) == "done"
    out = tmp_path / "answers" / "qwen_w1"
    recs = gen.read_chunk(out / "sampled-s1-c00.jsonl")
    assert len(recs) == 4 and recs[0]["text"] == "a\x85b\u2028c"
    assert len((out / "sampled-s1-c00.jsonl").read_text().splitlines()) == 4          # ASCII JSON on disk
    assert _run(tmp_path, cfg, CountingStopper(99), loads) == "done" and len(loads) == 1


def test_answers_from_another_battery_or_adapter_are_refused(tmp_path):
    cfg, loads = _small_cfg(), []
    _run(tmp_path, cfg, CountingStopper(1), loads)
    changed = {**cfg, "format_primer": "Another primer."}
    with pytest.raises(gen.Refused, match="another battery"):
        _run(tmp_path, changed, CountingStopper(99), loads)
    with pytest.raises(gen.Refused):
        _run(tmp_path, cfg, CountingStopper(99), loads, meta={"adapter_sha256": "b" * 64})


# ------------------------------------------------------------------------------------------- pre-registration gate

def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org",
                    "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args], check=True, capture_output=True)


def _frozen_repo(tmp_path, status="frozen", approved=True):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "configs" / "probes.yaml").write_text(f"status: {status}\nprereg_tag: prereg-v1\n")
    (repo / "docs" / "preregistration.md").write_text("# Pre-registration\n\n**Status:** "
                                                      + ("approved\n" if approved else "DRAFT, not approved\n"))
    _git(repo, "init", "-q")
    _git(repo, "add", "configs/probes.yaml", "docs/preregistration.md")
    _git(repo, "commit", "-qm", "freeze")
    _git(repo, "tag", "prereg-v1")
    return repo, load_yaml(repo / "configs" / "probes.yaml")


def test_gate_opens_only_for_a_frozen_approved_tagged_battery(tmp_path):
    repo, cfg = _frozen_repo(tmp_path)
    ok, reasons, info = gate.check(repo, cfg)
    assert ok and reasons == [] and len(info["tag_commit"]) == 40
    (repo / "configs" / "probes.yaml").write_text("status: frozen\nprereg_tag: prereg-v1\n# edited\n")
    ok, reasons, _ = gate.check(repo, cfg)
    assert not ok and "configs/probes.yaml differs from tag" in reasons[0]


def test_gate_closed_without_tag_draft_or_approval(tmp_path):
    repo, cfg = _frozen_repo(tmp_path / "a", status="draft", approved=False)
    ok, reasons, _ = gate.check(repo, cfg)
    assert not ok and any("not 'frozen'" in r for r in reasons) and any("Status: approved" in r for r in reasons)
    _git(repo, "tag", "-d", "prereg-v1")
    ok, reasons, _ = gate.check(repo, {**cfg, "status": "frozen"})
    assert not ok and "does not exist" in reasons[-1]
    assert not gate.check(tmp_path / "not-a-repo", {"status": "frozen", "prereg_tag": "prereg-v1"})[0]


def test_gate_needs_head_to_contain_the_tag(tmp_path):
    repo, cfg = _frozen_repo(tmp_path)
    _git(repo, "checkout", "-q", "--orphan", "other")
    _git(repo, "commit", "-qm", "unrelated")
    ok, reasons, _ = gate.check(repo, cfg)
    assert not ok and any("HEAD does not contain" in r for r in reasons)


def test_cli_refuses_adapters_without_the_gate_and_previews_only_bases(tmp_path):
    repo, _ = _frozen_repo(tmp_path, status="draft", approved=False)
    common = ["--repo", str(repo), "--check-gate"]
    assert gen.main(["--variant", "qwen_w1", *common]) == gen.EXIT_REFUSED
    assert gen.main(["--variant", "gemma_ex_w3", "--preview", *common]) == gen.EXIT_REFUSED
    assert gen.main(["--variant", "qwen_base", *common]) == gen.EXIT_REFUSED
    assert gen.main(["--variant", "qwen_base", "--preview", *common]) == gen.EXIT_OK
    frozen, _ = _frozen_repo(tmp_path / "f")
    assert gen.main(["--variant", "qwen_w1", "--repo", str(frozen), "--probes-config",
                     str(frozen / "configs" / "probes.yaml"), "--check-gate"]) == gen.EXIT_REFUSED   # not a battery
    assert gen.main(["--plan"]) == gen.EXIT_OK


def test_cli_preview_runs_a_base_model_into_runs_not_results(tmp_path, monkeypatch):
    repo, _ = _frozen_repo(tmp_path, status="draft", approved=False)
    small = tmp_path / "probes.yaml"
    import yaml
    small.write_text(yaml.safe_dump(_small_cfg()))
    monkeypatch.setattr(gen, "load_tokenizer", lambda mc: CharTok())
    monkeypatch.setattr(gen, "load_variant", lambda v, mc, runs: _tiny_qwen35().eval())
    rc = gen.main(["--variant", "qwen_base", "--preview", "--repo", str(repo), "--probes-config", str(small),
                   "--runs-dir", str(tmp_path / "runs"), "--results-dir", str(tmp_path / "results")])
    assert rc == gen.EXIT_OK and (tmp_path / "runs" / "probe_preview" / "qwen_base" / "DONE").exists()
    assert not (tmp_path / "results").exists()
    preview = tmp_path / "runs" / "probe_preview" / "qwen_base"
    rec = json.loads((preview / "greedy-c00.jsonl").read_text().split("\n")[0])
    assert rec["preview"] is True and rec["prereg"] is None and rec["adapter"] is None
    assert np.isfinite(rec["n_new_tokens"])
