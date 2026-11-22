"""OpenJTalk front end → words and PK.

Needs ``pyopenjtalk`` (``pip install pyopenjtalk-plus``). Imported lazily so
the rest of the package works without it.

OpenJTalk's NJD output (``pyopenjtalk.run_frontend``) is used after its own
accent-phrase and accent-type processing: ``chain_flag == 1`` means a word
continues the current accent phrase, and the ``acc`` of the first word of a
phrase is the accent of the whole phrase.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .kana import is_kana, split_morae
from .pk import AccentPhrase, PKSentence, to_pk

_PAUSE_MARKS = set("、，,・…‥;；:：")
_END_MARKS = set("。．.！!")
_QUESTION_MARKS = set("？?")
_DEVOICE = "’"


@dataclass(frozen=True)
class OJTWord:
    surface: str  # as returned by OpenJTalk (full-width normalised)
    pos: str
    pron: str  # katakana pronunciation, devoicing marks removed
    acc: int
    chain_flag: int
    start: int | None  # character span in the input text, when it can be found
    end: int | None

    @property
    def is_symbol(self) -> bool:
        return self.pos == "記号" or not self.pron


def _pyopenjtalk():
    try:
        import pyopenjtalk
    except ImportError as e:  # pragma: no cover - exercised only without the dependency
        raise ImportError("OpenJTalk features need pyopenjtalk: pip install pyopenjtalk-plus") from e
    return pyopenjtalk


def _clean_pron(pron: str) -> str:
    return "".join(c for c in pron.replace(_DEVOICE, "") if is_kana(c))


def words(text: str) -> list[OJTWord]:
    """Run OpenJTalk and return its words, with spans into ``text`` when possible.

    Spans are found by searching the NFKC form of each surface in the NFKC form
    of the text. OpenJTalk rewrites some input (``3`` becomes ``三``), so such
    words get ``start = end = None``.
    """
    njd = _pyopenjtalk().run_frontend(text)
    norm = unicodedata.normalize("NFKC", text)
    # NFKC can change lengths; only trust spans when it does not.
    spans_ok = len(norm) == len(text)
    out: list[OJTWord] = []
    cursor = 0
    for w in njd:
        surface = w["string"]
        start = end = None
        if spans_ok:
            key = unicodedata.normalize("NFKC", surface)
            found = norm.find(key, cursor)
            if found >= 0:
                start, end = found, found + len(key)
                cursor = end
        pron = "" if w["pos"] == "記号" else _clean_pron(w["pron"])
        out.append(OJTWord(surface, w["pos"], pron, int(w["acc"]), int(w["chain_flag"]), start, end))
    return out


def words_to_sentence(ws: list[OJTWord], prons: list[str] | None = None) -> PKSentence:
    """Group words into accent phrases.

    ``prons`` optionally replaces each word's pronunciation (same length as
    ``ws``), which is how alternative readings are turned into PK. The accent
    type is kept from OpenJTalk and clamped to the new phrase length.
    """
    if prons is not None and len(prons) != len(ws):
        raise ValueError("prons must have one entry per word")
    sentence = PKSentence()
    cur: list[str] = []
    cur_acc = 0

    def close(pause: bool) -> None:
        nonlocal cur, cur_acc
        if cur:
            sentence.phrases.append(AccentPhrase(cur, min(cur_acc, len(cur)), pause_after=pause))
        elif pause and sentence.phrases:
            sentence.phrases[-1].pause_after = True
        cur, cur_acc = [], 0

    for i, w in enumerate(ws):
        pron = w.pron if prons is None else prons[i]
        if w.is_symbol:
            s = unicodedata.normalize("NFKC", w.surface)
            if s and s[-1] in _QUESTION_MARKS and i == len(ws) - 1:
                sentence.question = True
                close(False)
            elif s and (s[-1] in _PAUSE_MARKS or s[-1] in _END_MARKS or s[-1] in _QUESTION_MARKS):
                close(True)
            continue
        morae = split_morae(pron)
        if not morae:
            continue
        if w.chain_flag != 1 or not cur:
            close(False)
            cur_acc = max(w.acc, 0)
        cur.extend(morae)
    close(False)
    if sentence.phrases:
        sentence.phrases[-1].pause_after = False
    return sentence


def text_to_pk(text: str) -> str:
    """OpenJTalk's reading and accent of ``text`` as canonical PK."""
    return to_pk(words_to_sentence(words(text)))


def label_phrases(text: str) -> list[tuple[int, int]]:
    """(mora count, accent) per accent phrase, read from full-context labels.

    Used to check :func:`words_to_sentence` against OpenJTalk's own labels.
    A phrase is identified by its breath group (``I:..@i3``) and its position
    in that group (``F:..@f5``).
    """
    import re

    pat = re.compile(r"/F:(\d+)_(\d+)#\d+_\d+@(\d+)_\d+.*/I:\d+-\d+@(\d+)")
    out: list[tuple[int, int]] = []
    seen: set[tuple[str, str]] = set()
    for lab in _pyopenjtalk().extract_fullcontext(text):
        m = pat.search(lab)
        if not m:
            continue  # silence / pause
        key = (m.group(4), m.group(3))
        if key not in seen:
            seen.add(key)
            out.append((int(m.group(1)), int(m.group(2))))
    return out
