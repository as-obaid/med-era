"""Paired block bootstrap for perplexity ratios: exact nulls, known shifts, pairing and input checks."""
import math

import numpy as np
import pytest

from medera.stats.bootstrap import block_indices, ppl_change_ci


def _blocks(n=50, tokens=100, seed=0):
    return np.random.default_rng(seed).uniform(1.0, 3.0, n) * tokens   # summed NLL per block


def test_identical_inputs_give_zero_with_a_zero_width_interval():
    b = _blocks()
    r = ppl_change_ci(b, b.copy(), 100, n_boot=500)
    assert r == {"pct": 0.0, "lo": 0.0, "hi": 0.0}


def test_a_constant_per_token_shift_gives_the_known_percent_in_every_resample():
    tokens = 100
    b = _blocks(tokens=tokens)
    k = 0.02                                   # +0.02 nats per token on every block
    r = ppl_change_ci(b + k * tokens, b, tokens, n_boot=500)
    want = 100 * math.expm1(k)                 # ~ +2.02%
    assert r["pct"] == pytest.approx(want, abs=1e-12)
    assert r["lo"] == pytest.approx(want, abs=1e-9) and r["hi"] == pytest.approx(want, abs=1e-9)
    down = ppl_change_ci(b - k * tokens, b, tokens, n_boot=500)
    assert down["pct"] == pytest.approx(100 * math.expm1(-k), abs=1e-12) and down["hi"] < 0


def test_point_estimate_is_the_perplexity_ratio_token_weighted():
    rng = np.random.default_rng(1)
    tok = rng.integers(50, 150, 40).astype(float)        # unequal blocks: weights must be tokens, not blocks
    a, b = rng.uniform(1.4, 1.6, 40) * tok, rng.uniform(1.5, 1.7, 40) * tok
    ppl_a, ppl_b = math.exp(a.sum() / tok.sum()), math.exp(b.sum() / tok.sum())
    r = ppl_change_ci(a, b, tok, n_boot=2000)
    assert r["pct"] == pytest.approx(100 * (ppl_a / ppl_b - 1), rel=1e-10)
    assert r["lo"] < r["pct"] < r["hi"]


def test_pairing_removes_block_difficulty_from_the_interval():
    """Blocks differ a lot in difficulty but A is uniformly 1% better per token: the paired interval is tight."""
    rng = np.random.default_rng(2)
    tokens = 2047
    level = rng.uniform(1.0, 2.5, 60)                     # per-token NLL of each block (hard and easy blocks)
    b = level * tokens
    a = (level - 0.01 + rng.normal(0, 0.002, 60)) * tokens
    r = ppl_change_ci(a, b, tokens)
    assert r["hi"] < 0 and r["hi"] - r["lo"] < 0.2       # unpaired, block difficulty would swamp a 1% effect


def test_same_seed_same_interval_and_shared_indices():
    a, b = _blocks(seed=3), _blocks(seed=4)
    r1, r2 = ppl_change_ci(a, b, 100, n_boot=1000, seed=7), ppl_change_ci(a, b, 100, n_boot=1000, seed=7)
    assert r1 == r2
    idx = block_indices(len(a), 1000, 7)
    assert ppl_change_ci(a, b, 100, idx=idx) == r1
    assert idx.shape == (1000, len(a)) and idx.min() >= 0 and idx.max() < len(a)


@pytest.mark.parametrize("a,b,tok", [([1.0, 2.0], [1.0], 1), ([], [], 1), ([1.0], [1.0], 0)])
def test_bad_inputs_raise(a, b, tok):
    with pytest.raises(ValueError):
        ppl_change_ci(a, b, tok, n_boot=10)


def test_indices_must_match_the_number_of_blocks():
    with pytest.raises(ValueError):
        ppl_change_ci([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1, idx=block_indices(4, 10))
