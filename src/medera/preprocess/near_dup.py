"""Near-duplicate documents: word 5-gram shingles, MinHash + LSH banding for candidates, exact Jaccard."""
import hashlib
import re

import numpy as np

from medera.collect.manifest import pmcid_key

NGRAM = 5
NUM_PERM = 128
BANDS = 32  # 32 bands x 4 rows: a pair at Jaccard 0.8 becomes a candidate with probability ~1 - 4e-9
SEED = 42
PRIME = (1 << 31) - 1  # hashes are reduced mod this prime, so a * x + b stays inside uint64
WORD = re.compile(r"\w+")


def shingles(text: str) -> np.ndarray:
    """Sorted unique 64-bit blake2b hashes of the lowercase word 5-grams."""
    words = WORD.findall(text.lower())
    grams = {" ".join(words[i:i + NGRAM]) for i in range(max(len(words) - NGRAM + 1, 1))} if words else set()
    hashes = [int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little") for g in grams]
    return np.unique(np.array(hashes, dtype=np.uint64))


def _signature(hashes: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = hashes % np.uint64(PRIME)
    return ((x[:, None] * a[None, :] + b[None, :]) % np.uint64(PRIME)).min(axis=0)


def jaccard(x: np.ndarray, y: np.ndarray) -> float:
    inter = np.intersect1d(x, y, assume_unique=True).size
    return inter / (x.size + y.size - inter)


def near_duplicates(docs: dict[str, tuple[str, str]], threshold: float = 0.8) -> list[tuple[str, str, float]]:
    """docs maps pmcid -> (first_pub_date, text). Returns (kept_pmcid, dropped_pmcid, jaccard), sorted by the
    dropped PMCID. Docs are visited oldest first (ties: lower PMCID) and one is dropped when its Jaccard with an
    earlier KEPT doc is >= threshold, so the later doc of a pair always goes and every kept_pmcid is kept."""
    order = sorted(docs, key=lambda p: (docs[p][0], pmcid_key(p)))
    sh = {p: shingles(docs[p][1]) for p in order}
    order = [p for p in order if sh[p].size]
    rng = np.random.default_rng(SEED)
    a = rng.integers(1, PRIME, NUM_PERM, dtype=np.uint64)
    b = rng.integers(0, PRIME, NUM_PERM, dtype=np.uint64)
    rows = NUM_PERM // BANDS
    buckets = {}
    for i, p in enumerate(order):
        sig = _signature(sh[p], a, b)
        for band in range(BANDS):
            buckets.setdefault((band, sig[band * rows:(band + 1) * rows].tobytes()), []).append(i)
    earlier = {}  # doc index -> indices of earlier docs sharing at least one band
    for members in buckets.values():
        for k, i in enumerate(members):
            if k:
                earlier.setdefault(i, set()).update(members[:k])

    kept, out = set(), []
    for i, p in enumerate(order):
        for j in sorted(earlier.get(i, ())):
            if j in kept and (jac := jaccard(sh[order[j]], sh[p])) >= threshold:
                out.append((order[j], p, round(jac, 4)))
                break
        else:
            kept.add(i)
    return sorted(out, key=lambda t: pmcid_key(t[1]))
