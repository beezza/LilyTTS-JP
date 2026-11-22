import pytest

from lilytts.text.kana import reading_key, split_morae
from lilytts.text.pk import AccentPhrase, PKError, PKSentence, normalize_pk, parse_pk, to_pk


def test_split_morae():
    assert split_morae("キョーシツ") == ["キョ", "ー", "シ", "ツ"]
    assert split_morae("にっぽんばし") == ["ニ", "ッ", "ポ", "ン", "バ", "シ"]
    assert split_morae("ティーパーティー") == ["ティ", "ー", "パ", "ー", "ティ", "ー"]


def test_reading_key_merges_long_vowel_spellings():
    assert reading_key("キョー") == reading_key("きょう") == reading_key("キョウ")
    assert reading_key("セーセキ") == reading_key("せいせき")
    assert reading_key("キョー") != reading_key("コンニチ")


def test_pk_roundtrip_patterns():
    s = PKSentence(
        [
            AccentPhrase(["キョ", "ー", "ワ"], 1),
            AccentPhrase(["ヨ", "イ"], 1),
            AccentPhrase(["テ", "ン", "キ", "デ", "ス", "ネ"], 1),
        ]
    )
    pk = to_pk(s)
    assert pk == "^キョ]ーワ#ヨ]イ#テ]ンキデスネ$"
    back = parse_pk(pk)
    assert [(p.kana, p.nucleus) for p in back.phrases] == [("キョーワ", 1), ("ヨイ", 1), ("テンキデスネ", 1)]


def test_pk_flat_middle_and_question():
    s = PKSentence(
        [AccentPhrase(["ニ", "ホ", "ン", "バ", "シ"], 0, pause_after=True), AccentPhrase(["ア", "ン", "ゼ", "ン"], 3)],
        question=True,
    )
    assert to_pk(s) == "^ニ[ホンバシ_ア[ンゼ]ン?"
    assert normalize_pk("^ ニ[ホンバシ _ ア[ンゼ]ン ?") == "^ニ[ホンバシ_ア[ンゼ]ン?"
    assert parse_pk("^ニ[ホンバシ_ア[ンゼ]ン?").phrases[0].pause_after


@pytest.mark.parametrize("bad", ["^キ]ョ]ー$", "^キョー[ワ$", "^キョ#$", "^キョーA$", "^ヨ[]イ$"])
def test_pk_rejects_invalid(bad):
    with pytest.raises(PKError):
        parse_pk(bad)
