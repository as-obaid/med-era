"""Equal-token budget across windows and packing into fixed-length blocks."""
import random

import numpy as np


def equal_budget(train_lengths: dict[str, int]) -> int:
    return min(train_lengths.values())


def take_budget(docs: list[list[int]], budget: int, seed: int) -> list[int]:
    """Shuffle whole documents, concatenate them, and cut the stream to exactly `budget` tokens."""
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    chosen, n = [], 0
    for i in order:
        if n >= budget:
            break
        chosen.append(np.asarray(docs[i], dtype=np.int64))
        n += len(docs[i])
    if n < budget:
        raise ValueError(f"only {n} tokens available for a budget of {budget}")
    if not chosen:
        return []
    return np.concatenate(chosen)[:budget].tolist()


def pack_blocks(stream: list[int], block_len: int, bos_id: int | None) -> np.ndarray:
    """uint32 [n, block_len]; with a BOS each row is [bos] + block_len-1 stream tokens. The remainder is dropped."""
    step = block_len if bos_id is None else block_len - 1
    n = len(stream) // step
    body = np.asarray(stream[: n * step], dtype=np.uint32).reshape(n, step)
    if bos_id is None:
        return body
    return np.concatenate([np.full((n, 1), bos_id, dtype=np.uint32), body], axis=1)
