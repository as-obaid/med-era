"""Paired block bootstrap for perplexity ratios (T3 confidence intervals).

Two variants of the same model are scored on the same held-out blocks, so a comparison resamples block indices once
and applies the same draw to both: the interval reflects block-to-block variation in the difference, not in the
level. A cell's perplexity is exp(summed NLL / predicted tokens), so the resampled statistic is

    100 * (exp((sum NLL_A - sum NLL_B) / sum tokens) - 1)      over the resampled blocks,

the token-weighted change in perplexity of A relative to B, in percent. The interval is the percentile interval of
that statistic. The draw depends only on (number of blocks, seed), so every comparison on one held-out set uses
the same resamples (common random numbers) and the intervals in one table are mutually consistent.

The intervals are descriptive. Blocks are consecutive pieces of packed articles, so blocks from one article are
correlated; resampling blocks as if independent makes the intervals too narrow when there are few articles.
"""
import math

import numpy as np

N_BOOT = 10_000
SEED = 6620
CI = 95.0


def block_indices(n_blocks: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """Resampled block indices, shape [n_boot, n_blocks], drawn with replacement."""
    if n_blocks < 1:
        raise ValueError("need at least one block")
    return np.random.default_rng(seed).integers(0, n_blocks, size=(n_boot, n_blocks))


def ppl_change_ci(nll_a, nll_b, tokens, idx: np.ndarray | None = None, n_boot: int = N_BOOT, seed: int = SEED,
                  ci: float = CI) -> dict:
    """Percent change in perplexity of A relative to B on the same blocks, with a paired percentile bootstrap CI.

    nll_a, nll_b: summed NLL per block (nats), same blocks in the same order.
    tokens: predicted tokens per block, a scalar (equal-length blocks) or one value per block.
    idx: resampled indices from block_indices(); drawn here from (n_boot, seed) if not given.
    Returns {"pct", "lo", "hi"}; "pct" equals 100 * (ppl_A / ppl_B - 1) up to rounding.
    """
    a, b = np.asarray(nll_a, dtype=np.float64), np.asarray(nll_b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError(f"need two equal-length 1-D arrays of per-block NLL, got {a.shape} and {b.shape}")
    t = np.broadcast_to(np.asarray(tokens, dtype=np.float64), a.shape)
    if np.any(t <= 0):
        raise ValueError("tokens per block must be positive")
    d = a - b
    point = 100 * math.expm1(math.fsum(d) / math.fsum(t))
    if idx is None:
        idx = block_indices(a.size, n_boot, seed)
    elif idx.ndim != 2 or idx.shape[1] != a.size:
        raise ValueError(f"idx has shape {idx.shape}, expected [n_boot, {a.size}]")
    boot = 100 * np.expm1(d[idx].sum(axis=1) / t[idx].sum(axis=1))
    tail = (100 - ci) / 2
    lo, hi = np.percentile(boot, [tail, 100 - tail])
    return {"pct": point, "lo": float(lo), "hi": float(hi)}
