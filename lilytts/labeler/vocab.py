"""Character vocabulary for the kana CTC recogniser.

Labels are OpenJTalk-style pronunciations (``キョーシツ``) normalised to what
is audible: ヲ→オ, ヂ→ジ, ヅ→ズ, ヮ→ァ. Each katakana character is one token
(small kana are separate tokens), which keeps the output layer small (~90).
"""

from __future__ import annotations

from ..text.kana import hira_to_kata

BLANK = "<b>"
UNK = "<unk>"
_PHONETIC = str.maketrans({"ヲ": "オ", "ヂ": "ジ", "ヅ": "ズ", "ヮ": "ァ"})
_CHARS = [chr(c) for c in range(0x30A1, 0x30F5) if chr(c) not in "ヲヂヅヮヰヱ"] + ["ー"]


def phonetic(kana: str) -> str:
    """Normalise a reading to the recogniser's label space."""
    return hira_to_kata(kana).translate(_PHONETIC)


class KanaVocab:
    def __init__(self) -> None:
        self.tokens = [BLANK, UNK] + _CHARS
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.blank = 0
        self.unk = 1

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, kana: str) -> list[int]:
        """Encode a reading; characters outside the vocabulary become <unk>."""
        return [self.index.get(c, self.unk) for c in phonetic(kana) if not c.isspace()]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.tokens[i] for i in ids if i > self.unk)

    def to_hf_vocab(self) -> dict[str, int]:
        """Vocabulary dict for ``transformers.Wav2Vec2CTCTokenizer``."""
        return {("<pad>" if t == BLANK else t): i for i, t in enumerate(self.tokens)}
