import json

import numpy as np
import pytest

from medera.preprocess import __main__ as cli
from medera.preprocess.budget import equal_budget, pack_blocks, take_budget


def docs(n_docs, start=0, length=10):
    return [list(range(start + i * length, start + (i + 1) * length)) for i in range(n_docs)]


def test_equal_budget_is_min():
    assert equal_budget({"w1": 500, "w2": 300, "w3": 900}) == 300


def test_take_budget_exact_and_seeded():
    d = docs(30)
    a = take_budget(d, 123, seed=42)
    assert len(a) == 123 and all(isinstance(t, int) for t in a)
    assert take_budget(d, 123, seed=42) == a
    assert take_budget(d, 123, seed=43) != a
    # Whole documents in shuffled order, then a cut: the stream starts at a document boundary.
    assert a[0] % 10 == 0 and a[:10] == list(range(a[0], a[0] + 10))


def test_take_budget_smallest_window_uses_all():
    d = docs(7, length=13)
    total = sum(len(x) for x in d)
    s = take_budget(d, total, seed=42)
    assert sorted(s) == sorted(t for x in d for t in x)


def test_take_budget_raises_when_short():
    with pytest.raises(ValueError):
        take_budget(docs(2), 21, seed=42)


def test_pack_blocks_shapes_no_bos():
    b = pack_blocks(list(range(5000)), 2048, None)
    assert b.shape == (2, 2048) and b.dtype == np.uint32
    assert b.ravel().tolist() == list(range(4096))


def test_pack_blocks_with_bos():
    b = pack_blocks(list(range(10, 5010)), 2048, 1)
    assert b.shape == (2, 2048) and b.dtype == np.uint32
    assert (b[:, 0] == 1).all()
    assert b[:, 1:].ravel().tolist() == list(range(10, 10 + 2 * 2047))


def test_pack_blocks_too_short_gives_empty():
    assert pack_blocks([1, 2, 3], 2048, None).shape == (0, 2048)


def test_equal_windows_get_equal_blocks():
    windows = {"w1": docs(40, length=101), "w2": docs(90, start=10**5, length=57), "w3": docs(300, start=10**6, length=33)}
    budget = equal_budget({w: sum(map(len, d)) for w, d in windows.items()})
    shapes = {w: pack_blocks(take_budget(d, budget, 42), 256, 0).shape for w, d in windows.items()}
    assert len(set(shapes.values())) == 1 and shapes["w1"][0] > 0


class FakeTok:
    eos_token, eos_token_id = "</s>", 0
    bos_token_id = 1
    unk_token_id = 99
    vocab = {"</s>": 0, "<s>": 1, "<sep>": 7}

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def __call__(self, text, add_special_tokens=True):
        ids = [2 + (ord(c) % 50) for c in text]
        return type("Enc", (), {"input_ids": ([self.bos_token_id] if add_special_tokens else []) + ids})()


SIZES = {"w1": 300, "w2": 500, "w3": 800}


def _pack(tmp_path, repo_root, monkeypatch, model_yaml, tok=None):
    out = tmp_path / "processed" / "hrt"
    out.mkdir(parents=True, exist_ok=True)
    for w, n in SIZES.items():
        rows = [{"pmcid": f"PMC{w}{i}", "split": "heldout" if i == 0 else "train", "text": "ab" * n}
                for i in range(4)]
        (out / f"{w}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    models = tmp_path / "models.yaml"
    models.write_text("tiny:\n  hf_id: fake/tiny\n  revision: abc123\n" + model_yaml)
    monkeypatch.setattr(cli, "load_tokenizer", lambda hf_id, revision: tok or FakeTok())
    cli.main(["pack", "--model", "tiny", "--models", str(models), "--config",
              str(repo_root / "configs" / "corpus.yaml"), "--data-root", str(tmp_path), "--block-len", "64"])
    return out, json.loads((out / "tiny" / "budget.json").read_text())


def test_pack_uses_doc_separator(tmp_path, repo_root, monkeypatch):
    out, meta = _pack(tmp_path, repo_root, monkeypatch, '  doc_separator: "<sep>"\n')
    assert (meta["doc_separator"], meta["separator_id"]) == ("<sep>", 7)
    tr = np.load(out / "tiny" / "w3_train.npy")
    assert (tr == 7).sum() > 0 and (tr == 0).sum() == 0  # separator, never the eos


class NoUnkTok(FakeTok):
    unk_token_id = None

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)


@pytest.mark.parametrize("tok", [FakeTok(), NoUnkTok()])
def test_pack_rejects_unknown_doc_separator(tmp_path, repo_root, monkeypatch, tok):
    with pytest.raises(SystemExit, match="<nope>"):
        _pack(tmp_path, repo_root, monkeypatch, '  doc_separator: "<nope>"\n', tok=tok)


def test_pack_cli_with_fake_tokenizer(tmp_path, repo_root, monkeypatch, capsys):
    sizes = SIZES
    out, meta = _pack(tmp_path, repo_root, monkeypatch, "")
    assert (meta["doc_separator"], meta["separator_id"]) == ("</s>", 0)  # no key: fall back to the eos
    # Each train doc is 2*n tokens + EOS; 3 train docs per window; the smallest window sets the budget.
    assert meta["budget_tokens"] == 3 * (600 + 1)
    assert (meta["model"], meta["hf_id"], meta["revision"], meta["block_len"], meta["bos_id"]) == (
        "tiny", "fake/tiny", "abc123", 64, 1)
    assert meta["below_min"] is True and meta["min_train_tokens"] == 2000000
    assert "WARNING" in capsys.readouterr().out
    n_blocks = meta["budget_tokens"] // 63
    for w, n in sizes.items():
        tr = np.load(out / "tiny" / f"{w}_train.npy")
        ho = np.load(out / "tiny" / f"{w}_heldout.npy")
        assert tr.shape == (n_blocks, 64) and tr.dtype == np.uint32 and (tr[:, 0] == 1).all()
        assert ho.shape == ((2 * n + 1) // 63, 64)
        info = meta["windows"][w]
        assert (info["train_docs"], info["train_tokens_available"], info["train_blocks"]) == (3, 3 * (2 * n + 1), n_blocks)
        assert (info["heldout_docs"], info["heldout_blocks"]) == (1, ho.shape[0])
