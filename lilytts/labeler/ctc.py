"""CTC scoring of candidate readings against recogniser output.

``log_probs`` is a ``(T, V)`` array of per-frame log-probabilities (log-softmax
output of the recogniser). All functions are plain numpy so they run on the
CPU workers next to the GPU recogniser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEG_INF = -np.inf


def ctc_log_likelihood(log_probs: np.ndarray, labels: list[int], blank: int = 0) -> float:
    """log P(labels | log_probs) under CTC (forward algorithm)."""
    T = log_probs.shape[0]
    if not labels:
        return float(log_probs[:, blank].sum())
    ext = [blank]
    for l in labels:
        ext += [l, blank]
    S = len(ext)
    ext_arr = np.asarray(ext)
    # A label can skip the blank before it unless it repeats the previous label.
    can_skip = np.zeros(S, dtype=bool)
    for s in range(2, S):
        can_skip[s] = ext[s] != blank and ext[s] != ext[s - 2]

    alpha = np.full(S, NEG_INF)
    alpha[0] = log_probs[0, blank]
    alpha[1] = log_probs[0, ext[1]]
    for t in range(1, T):
        prev = alpha
        stay = prev
        step = np.concatenate(([NEG_INF], prev[:-1]))
        skip = np.where(can_skip, np.concatenate(([NEG_INF, NEG_INF], prev[:-2])), NEG_INF)
        alpha = np.logaddexp(np.logaddexp(stay, step), skip) + log_probs[t, ext_arr]
    return float(np.logaddexp(alpha[-1], alpha[-2]))


def greedy_decode(log_probs: np.ndarray, blank: int = 0) -> list[int]:
    best = log_probs.argmax(axis=1)
    out: list[int] = []
    prev = -1
    for b in best:
        if b != prev and b != blank:
            out.append(int(b))
        prev = b
    return out


def best_path_score(log_probs: np.ndarray) -> float:
    """Log-probability of the single most likely frame path (a fit reference)."""
    return float(log_probs.max(axis=1).sum())


@dataclass(frozen=True)
class Ranking:
    order: list[int]  # candidate indices, best first
    scores: list[float]  # per candidate, same order as the input
    margin: float  # (best - second best) / frames; inf when only one candidate
    fit: float  # (best - best_path) / frames; near 0 = the audio fits the best candidate well


def rank_candidates(log_probs: np.ndarray, candidates: list[list[int]], blank: int = 0) -> Ranking:
    if not candidates:
        raise ValueError("no candidates")
    T = log_probs.shape[0]
    scores = [ctc_log_likelihood(log_probs, c, blank) for c in candidates]
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    best = scores[order[0]]
    margin = (best - scores[order[1]]) / T if len(order) > 1 else float("inf")
    return Ranking(order, scores, margin, (best - best_path_score(log_probs)) / T)
