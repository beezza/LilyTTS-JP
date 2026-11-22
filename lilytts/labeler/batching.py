"""Torch-free helpers for the recogniser: batching and error rates."""

from __future__ import annotations

import random


def duration_batches(durations: list[float], max_seconds: float, shuffle: bool = True,
                     seed: int = 0, bucket: int = 100) -> list[list[int]]:
    """Group indices into batches of at most ``max_seconds`` of padded audio.

    Items are sorted by length inside buckets of ``bucket * typical batch`` so
    padding stays small, then batches are shuffled. Padded cost is
    ``len(batch) * longest``, which is what the GPU actually processes.
    """
    order = list(range(len(durations)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(order)
    chunk = max(bucket * 16, 1)
    batches: list[list[int]] = []
    for c in range(0, len(order), chunk):
        part = sorted(order[c:c + chunk], key=lambda i: durations[i])
        cur: list[int] = []
        longest = 0.0
        for i in part:
            d = durations[i]
            if d > max_seconds:
                continue  # longer than a whole batch: skip
            if cur and max(longest, d) * (len(cur) + 1) > max_seconds:
                batches.append(cur)
                cur, longest = [], 0.0
            cur.append(i)
            longest = max(longest, d)
        if cur:
            batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(refs: list[str], hyps: list[str]) -> float:
    errors = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return errors / total if total else 0.0
