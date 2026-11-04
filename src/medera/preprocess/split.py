"""Deterministic held-out split, independent of input order."""
import random


def heldout_split(ids: list[str], frac: float, seed: int) -> tuple[list[str], list[str]]:
    ids = sorted(ids)
    n_heldout = round(len(ids) * frac)
    if len(ids) >= 2:
        n_heldout = max(n_heldout, 1)
    held = set(random.Random(seed).sample(ids, n_heldout))
    return [i for i in ids if i not in held], [i for i in ids if i in held]
