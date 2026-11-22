"""Judge how a target span was read, from a full-sentence reading.

Works for any system that outputs the reading of the whole sentence: a text
front end (OpenJTalk, Yomi) or a recogniser run on synthesised speech
(method 1 in docs/LILYBENCH.md §4). Readings are compared with
:func:`lilytts.text.kana.reading_key`, so ``キョー`` and ``キョウ`` match.
"""

from __future__ import annotations

from lilytts.labeler.vocab import phonetic
from lilytts.text.kana import reading_key

CORRECT, WRONG, UNDETERMINED = "correct", "wrong", "undetermined"


def _contains(reading: str, kana: str) -> bool:
    """True if ``kana`` occurs in ``reading``.

    The exact (phonetic) form is tried first. The lossy key is only a fallback
    for long-vowel spelling differences (キョー / キョウ), because it can merge
    morae across word boundaries (デ + イッカイ → デーッカイ).
    """
    return phonetic(kana) in phonetic(reading) or reading_key(kana) in reading_key(reading)


def judge_span(sentence_reading: str, gold: list[str], confusers: list[str]) -> str:
    gold_hits = [g for g in gold if _contains(sentence_reading, g)]
    # A confuser counts only if it is not just part of a matched gold reading.
    conf_hits = [
        c for c in confusers
        if _contains(sentence_reading, c) and not any(_contains(g, c) for g in gold_hits)
    ]
    if gold_hits and not conf_hits:
        return CORRECT
    if conf_hits and not gold_hits:
        return WRONG
    return UNDETERMINED
