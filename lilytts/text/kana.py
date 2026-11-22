"""Kana helpers shared by the text front end and the data pipeline."""

from __future__ import annotations

_HIRA_START = 0x3041  # ぁ
_HIRA_END = 0x3096  # ゖ
_KATA_OFFSET = 0x60  # ァ - ぁ

# Characters that may appear in a reading besides kana proper.
_READING_EXTRA = set("ー・ヽヾゝゞ")


def hira_to_kata(s: str) -> str:
    """Convert hiragana to katakana. Other characters are left unchanged."""
    return "".join(
        chr(ord(c) + _KATA_OFFSET) if _HIRA_START <= ord(c) <= _HIRA_END else c
        for c in s
    )


def is_kana(c: str) -> bool:
    o = ord(c)
    return (
        _HIRA_START <= o <= _HIRA_END
        or 0x30A1 <= o <= 0x30FA  # ァ..ヺ
        or c in _READING_EXTRA
    )


def is_kana_string(s: str) -> bool:
    return bool(s) and all(is_kana(c) for c in s)


def is_kanji(c: str) -> bool:
    """True for CJK ideographs and the marks that behave like them in ruby bases."""
    o = ord(c)
    return (
        0x4E00 <= o <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= o <= 0x4DBF  # Extension A
        or 0x20000 <= o <= 0x2FA1F  # Extensions B.. and compatibility supplement
        or 0xF900 <= o <= 0xFAFF  # Compatibility Ideographs
        or c in "々〆ヶヵ〇"
    )


# --- morae -----------------------------------------------------------------

_SMALL = set("ァィゥェォャュョヮ")


def split_morae(kata: str) -> list[str]:
    """Split a katakana string into morae.

    Small kana (ャュョァィゥェォヮ) join the preceding kana; ッ, ン and ー are
    morae of their own. Hiragana is converted first.
    """
    morae: list[str] = []
    for c in hira_to_kata(kata):
        if c in _SMALL and morae and morae[-1][-1] not in "ッンー":
            morae[-1] += c
        else:
            morae.append(c)
    return morae


# Vowel of each katakana (for long-vowel normalisation).
_VOWEL_ROWS = {
    "ア": "アカサタナハマヤラワガザダバパァャヮ",
    "イ": "イキシチニヒミリギジヂビピィ",
    "ウ": "ウクスツヌフムユルグズヅブプヴゥュ",
    "エ": "エケセテネヘメレゲゼデベペェ",
    "オ": "オコソトノホモヨロヲゴゾドボポォョ",
}
_VOWEL_OF = {c: v for v, row in _VOWEL_ROWS.items() for c in row}


def mora_vowel(mora: str) -> str | None:
    """Vowel (ア/イ/ウ/エ/オ) of a mora, or None for ッ, ン, ー and symbols."""
    return _VOWEL_OF.get(mora[-1]) if mora else None


def reading_key(s: str) -> str:
    """Lossy key for comparing readings written in different conventions.

    ``キョー``, ``キョウ`` and ``きょう`` all map to the same key: ー becomes the
    vowel of the previous mora, and オ-row + ウ / エ-row + イ are treated as
    long vowels. Use it only to compare readings, never as output.
    """
    out: list[str] = []
    for m in split_morae(s):
        prev_v = mora_vowel(out[-1]) if out else None
        if m == "ー" and prev_v:
            m = prev_v
        if m == "ウ" and prev_v == "オ":
            m = "オ"
        elif m == "イ" and prev_v == "エ":
            m = "エ"
        out.append(m)
    return "".join(out)


def to_pron(kata: str) -> str:
    """Approximate pronunciation form: long vowels written as ``ー``.

    ``ジュウ`` → ``ジュー``, ``トオカ`` → ``トーカ``, ``セイ`` → ``セー``. This
    follows OpenJTalk's ``pron`` convention for regular cases and is meant for
    generated readings (numerals, lexicon entries); words whose ウ/イ is not a
    long vowel (``オモウ``) should come from OpenJTalk instead.
    """
    out: list[str] = []
    for m in split_morae(hira_to_kata(kata)):
        v = mora_vowel(out[-1]) if out and out[-1] != "ー" else None
        if v and (
            (m == "ウ" and v in "ウオ")
            or (m == "イ" and v in "イエ")
            or (m == "オ" and v == "オ")
            or (m == "ア" and v == "ア")
        ):
            m = "ー"
        out.append(m)
    return "".join(out)
