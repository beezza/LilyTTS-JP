import itertools

import numpy as np

from lilytts.labeler.ctc import ctc_log_likelihood, greedy_decode, rank_candidates
from lilytts.labeler.vocab import KanaVocab, phonetic


def brute_force(log_probs, labels, blank=0):
    T, V = log_probs.shape
    total = -np.inf
    for path in itertools.product(range(V), repeat=T):
        collapsed = [p for i, p in enumerate(path) if p != blank and (i == 0 or p != path[i - 1])]
        if collapsed == list(labels):
            total = np.logaddexp(total, sum(log_probs[t, p] for t, p in enumerate(path)))
    return total


def random_log_probs(T, V, seed):
    x = np.random.default_rng(seed).normal(size=(T, V))
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def test_matches_brute_force():
    lp = random_log_probs(5, 3, 0)
    for labels in ([1], [1, 2], [1, 1], [2, 1, 2], []):
        assert np.isclose(ctc_log_likelihood(lp, labels), brute_force(lp, labels))


def test_impossible_sequence_is_minus_inf():
    lp = random_log_probs(2, 3, 1)
    assert ctc_log_likelihood(lp, [1, 1, 1]) == -np.inf


def _peaky(vocab, kana, frames_per_char=3):
    ids = vocab.encode(kana)
    T = len(ids) * frames_per_char
    lp = np.full((T, len(vocab)), np.log(1e-4))
    for t in range(T):
        tok = ids[t // frames_per_char] if t % frames_per_char == 0 else vocab.blank
        lp[t, tok] = np.log(0.99)
    return lp - np.log(np.exp(lp).sum(axis=1, keepdims=True))


def test_ranks_the_spoken_reading_first():
    v = KanaVocab()
    lp = _peaky(v, "コンニチワ")
    cands = [v.encode(k) for k in ("キョーワ", "コンニチワ", "コンニチハ")]
    r = rank_candidates(lp, cands)
    assert r.order[0] == 1 and r.margin > 0
    assert v.decode(greedy_decode(lp)) == "コンニチワ"


def test_vocab_phonetic_normalisation():
    v = KanaVocab()
    assert phonetic("をぢづ") == "オジズ"
    assert v.decode(v.encode("ケンキューヲ")) == "ケンキューオ"
    assert len(v) < 100
