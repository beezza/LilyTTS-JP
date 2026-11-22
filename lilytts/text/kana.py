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
