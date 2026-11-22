"""Recover how a line was actually read, using the recogniser as a judge.

1. OpenJTalk splits the text into words with a default pronunciation.
2. *Slots* are words whose reading may differ from that default: homographs
   from ``data/homographs.tsv`` and numeral + counter groups (``十三本``, ``何本``,
   ``３月``, ``一日``) whose readings follow from :mod:`lilytts.text.numerals`.
3. Every combination of slot readings is turned into a kana string and scored
   against the recogniser's CTC output; the best one is the reading.

The result keeps OpenJTalk's own reading, so lines where the audio disagrees
with OpenJTalk (``changed``) can be collected as dictionary-error examples.
Accent types are OpenJTalk's; the accent estimator replaces them later.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from importlib import resources
from typing import Protocol

import numpy as np

from ..text.kana import hira_to_kata
from ..text.numerals import UNITS, parse_numeral, readings_for
from ..text.openjtalk import OJTWord, words, words_to_sentence
from ..text.pk import to_pk
from .ctc import best_path_score, ctc_log_likelihood
from .vocab import KanaVocab


class Scorer(Protocol):
    frames: int

    def score(self, kana: str) -> float: ...

    def best_path(self) -> float: ...


class CTCScorer:
    """Scores kana strings against one utterance's recogniser output."""

    def __init__(self, log_probs: np.ndarray, vocab: KanaVocab | None = None) -> None:
        self.log_probs = log_probs
        self.vocab = vocab or KanaVocab()
        self.frames = int(log_probs.shape[0])
        self._cache: dict[str, float] = {}

    def score(self, kana: str) -> float:
        if kana not in self._cache:
            ids = self.vocab.encode(kana)
            self._cache[kana] = ctc_log_likelihood(self.log_probs, ids, self.vocab.blank)
        return self._cache[kana]

    def best_path(self) -> float:
        return best_path_score(self.log_probs)


def load_lexicon(path: str | None = None) -> dict[str, list[str]]:
    """Load a homograph lexicon (``surface<TAB>reading|reading``)."""
    if path is None:
        text = resources.files("lilytts.labeler").joinpath("data/homographs.tsv").read_text("utf-8")
    else:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    lex: dict[str, list[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        surface, readings = line.split("\t")
        lex[surface] = [hira_to_kata(r) for r in readings.split("|") if r]
    return lex


@dataclass
class Slot:
    first: int  # word index range [first, last)
    last: int
    kind: str  # "homograph" | "numeral"
    default: str  # OpenJTalk's pronunciation of the range
    options: list[str]  # candidate readings, default first
    accepted: list[str] = field(default_factory=list)  # rule-accepted readings (numerals)


_MERGED_NUMERAL = re.compile(r"^([0-9０-９〇一二三四五六七八九十百千]+|何)(本|匹|杯|分|回|個|冊|軒|階|足|歳|円|人|つ|月|日|時)$")


def _numeral_value(surface: str) -> tuple[bool, int | None]:
    if surface == "何":
        return True, None
    n = parse_numeral(surface)
    return (n is not None), n


def find_slots(ws: list[OJTWord], lexicon: dict[str, list[str]]) -> list[Slot]:
    slots: list[Slot] = []
    i = 0
    prev_surface = ""
    while i < len(ws):
        w = ws[i]
        # numeral + unit written as one word (５つ, 一日, ３月)
        m = _MERGED_NUMERAL.match(w.surface)
        if m:
            ok, n = _numeral_value(m.group(1))
            if ok and _supported(n, m.group(2)):
                slots.append(_numeral_slot(ws, i, i + 1, n, m.group(2), prev_surface))
                prev_surface = w.surface
                i += 1
                continue
        # numeral words (十, 三) or 何, followed by a unit word
        j = i
        while j < len(ws) and parse_numeral(ws[j].surface) is not None:
            j += 1
        if j == i and w.surface == "何":
            j = i + 1
        if j > i and j < len(ws) and ws[j].surface in UNITS:
            num_surface = "".join(x.surface for x in ws[i:j])
            ok, n = _numeral_value(num_surface)
            if ok and _supported(n, ws[j].surface):
                slots.append(_numeral_slot(ws, i, j + 1, n, ws[j].surface, prev_surface))
                prev_surface = ws[j].surface
                i = j + 1
                continue
        if w.surface in lexicon and not w.is_symbol:
            opts = [w.pron] + [r for r in lexicon[w.surface] if r != w.pron]
            slots.append(Slot(i, i + 1, "homograph", w.pron, opts))
        prev_surface = w.surface
        i += 1
    return slots


def _supported(n: int | None, unit: str) -> bool:
    try:
        readings_for(n, unit)
    except (ValueError, KeyError):
        return False
    return True


def _numeral_slot(ws, first, last, n, unit, prev_surface) -> Slot:
    accepted, confusers = readings_for(n, unit, after_month=prev_surface.endswith("月"))
    default = "".join(w.pron for w in ws[first:last])
    opts = [default] + [r for r in accepted + confusers if r != default]
    return Slot(first, last, "numeral", default, opts, accepted)


@dataclass
class SlotResult:
    start: int | None  # character span in the text
    end: int | None
    surface: str
    kind: str
    default: str
    chosen: str
    options: list[str]
    scores: list[float]
    margin: float  # (best - second best) / frames, other slots fixed
    accepted: list[str]


@dataclass
class ReadingResult:
    text: str
    kana: str
    pk: str
    openjtalk_kana: str
    openjtalk_pk: str
    slots: list[SlotResult]
    margin: float  # smallest slot margin (inf without slots)
    fit: float  # (score of chosen reading - best path) / frames
    changed: bool  # the audio disagrees with OpenJTalk somewhere


def _prons_with(ws: list[OJTWord], slots: list[Slot], choice: tuple[int, ...]) -> list[str]:
    prons = [w.pron for w in ws]
    for slot, k in zip(slots, choice):
        prons[slot.first] = slot.options[k]
        for idx in range(slot.first + 1, slot.last):
            prons[idx] = ""
    return prons


def _kana(ws: list[OJTWord], prons: list[str]) -> str:
    return "".join(p for w, p in zip(ws, prons) if not w.is_symbol)


def recover_reading(
    text: str,
    scorer: Scorer,
    lexicon: dict[str, list[str]] | None = None,
    max_combinations: int = 64,
) -> ReadingResult:
    ws = words(text)
    lex = load_lexicon() if lexicon is None else lexicon
    slots = find_slots(ws, lex)

    def score(choice: tuple[int, ...]) -> float:
        return scorer.score(_kana(ws, _prons_with(ws, slots, choice)))

    sizes = [len(s.options) for s in slots]
    total = int(np.prod(sizes)) if sizes else 1
    if total <= max_combinations:
        best = max(itertools.product(*[range(n) for n in sizes]), key=score)
    else:  # coordinate ascent: improve one slot at a time
        best = tuple(0 for _ in slots)
        for _ in range(2):
            for si, n in enumerate(sizes):
                best = max((best[:si] + (k,) + best[si + 1:] for k in range(n)), key=score)

    frames = max(scorer.frames, 1)
    slot_results: list[SlotResult] = []
    for si, slot in enumerate(slots):
        scores = [score(best[:si] + (k,) + best[si + 1:]) for k in range(len(slot.options))]
        ranked = sorted(scores, reverse=True)
        margin = (ranked[0] - ranked[1]) / frames if len(ranked) > 1 else float("inf")
        span_words = [w for w in ws[slot.first:slot.last]]
        start = span_words[0].start
        end = span_words[-1].end
        slot_results.append(
            SlotResult(start, end, "".join(w.surface for w in span_words), slot.kind, slot.default,
                       slot.options[best[si]], slot.options, scores, margin, slot.accepted)
        )

    prons = _prons_with(ws, slots, best)
    best_score = scorer.score(_kana(ws, prons))
    return ReadingResult(
        text=text,
        kana=_kana(ws, prons),
        pk=to_pk(words_to_sentence(ws, prons)),
        openjtalk_kana=_kana(ws, [w.pron for w in ws]),
        openjtalk_pk=to_pk(words_to_sentence(ws)),
        slots=slot_results,
        margin=min((s.margin for s in slot_results), default=float("inf")),
        fit=(best_score - scorer.best_path()) / frames,
        changed=any(s.chosen != s.default for s in slot_results),
    )
