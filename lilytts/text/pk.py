"""PK (Prosodic Kana): katakana morae plus Tokyo-dialect pitch-accent marks.

Canonical form has no spaces, e.g. ``^キョ]ーワ#ヨ]イ#テ]ンキデスネ$``.

==========  =====================================================
Symbol      Meaning
==========  =====================================================
``^``       start of sentence
``$``       end of a declarative sentence
``?``       end of a question (rising)
``#``       accent-phrase boundary without pause
``_``       accent-phrase boundary with pause
``[``       pitch rises (after the first mora of the phrase)
``]``       accent nucleus: pitch falls after the preceding mora
==========  =====================================================

An accent phrase is a list of morae and a nucleus position ``k``:
``k == 0`` is flat (heiban), otherwise the pitch falls after mora ``k``
(1-based). Pitch marks follow the standard Tokyo pattern: the first mora is
low unless ``k == 1``; ``[`` is written after mora 1 when ``k != 1`` and the
phrase has more than one mora; ``]`` is written after mora ``k``.
Whitespace is ignored when parsing, so ``^ キョ]ーワ # ヨ]イ $`` is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .kana import is_kana, split_morae

BOS, EOS, QUESTION = "^", "$", "?"
BOUNDARY, PAUSE = "#", "_"
RISE, FALL = "[", "]"
SYMBOLS = (BOS, EOS, QUESTION, BOUNDARY, PAUSE, RISE, FALL)


class PKError(ValueError):
    pass


@dataclass
class AccentPhrase:
    morae: list[str]
    nucleus: int = 0  # 0 = flat, k = fall after mora k (1-based)
    pause_after: bool = False  # boundary to the next phrase is a pause

    def __post_init__(self) -> None:
        if not self.morae:
            raise PKError("empty accent phrase")
        if not 0 <= self.nucleus <= len(self.morae):
            raise PKError(f"nucleus {self.nucleus} out of range for {len(self.morae)} morae")

    @property
    def kana(self) -> str:
        return "".join(self.morae)


@dataclass
class PKSentence:
    phrases: list[AccentPhrase] = field(default_factory=list)
    question: bool = False

    @property
    def kana(self) -> str:
        return "".join(p.kana for p in self.phrases)

    @property
    def morae(self) -> list[str]:
        return [m for p in self.phrases for m in p.morae]


def _phrase_to_pk(p: AccentPhrase) -> str:
    out: list[str] = []
    n = len(p.morae)
    for i, m in enumerate(p.morae, 1):
        out.append(m)
        if i == p.nucleus:
            out.append(FALL)
        elif i == 1 and n > 1 and p.nucleus != 1:
            out.append(RISE)
    return "".join(out)


def to_pk(sentence: PKSentence) -> str:
    """Serialise a sentence to canonical PK."""
    parts = [BOS]
    for i, p in enumerate(sentence.phrases):
        if i:
            parts.append(PAUSE if sentence.phrases[i - 1].pause_after else BOUNDARY)
        parts.append(_phrase_to_pk(p))
    parts.append(QUESTION if sentence.question else EOS)
    return "".join(parts)


def tokenize(pk: str) -> list[str]:
    """Split PK into tokens: symbols and morae."""
    tokens: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            tokens.extend(split_morae("".join(buf)))
            buf.clear()

    for c in pk:
        if c.isspace():
            continue
        if c in SYMBOLS:
            flush()
            tokens.append(c)
        elif is_kana(c):
            buf.append(c)
        else:
            raise PKError(f"invalid character {c!r} in PK")
    flush()
    return tokens


def parse_pk(pk: str) -> PKSentence:
    """Parse PK (canonical or spaced) into a :class:`PKSentence`."""
    tokens = tokenize(pk)
    if tokens and tokens[0] == BOS:
        tokens = tokens[1:]
    question = False
    if tokens and tokens[-1] in (EOS, QUESTION):
        question = tokens[-1] == QUESTION
        tokens = tokens[:-1]

    sentence = PKSentence(question=question)
    morae: list[str] = []
    nucleus = 0
    rose = False

    def close(pause: bool) -> None:
        nonlocal morae, nucleus, rose
        if not morae:
            raise PKError("empty accent phrase")
        if rose and nucleus == 1:
            raise PKError("rise after an initial-accent mora")
        sentence.phrases.append(AccentPhrase(morae, nucleus, pause_after=pause))
        morae, nucleus, rose = [], 0, False

    for t in tokens:
        if t in (BOUNDARY, PAUSE):
            close(t == PAUSE)
        elif t == RISE:
            if len(morae) != 1:
                raise PKError("rise must follow the first mora of a phrase")
            rose = True
        elif t == FALL:
            if not morae or nucleus:
                raise PKError("misplaced accent nucleus")
            nucleus = len(morae)
        elif t in (BOS, EOS, QUESTION):
            raise PKError(f"unexpected {t!r} inside sentence")
        else:
            morae.append(t)
    close(False)
    return sentence


def normalize_pk(pk: str) -> str:
    """Parse and re-serialise, e.g. to drop spaces or check validity."""
    return to_pk(parse_pk(pk))
