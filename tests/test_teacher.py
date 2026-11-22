import pytest

pytest.importorskip("pyopenjtalk")

from lilytts.labeler.reading import load_lexicon  # noqa: E402
from lilytts.labeler.teacher import check, label  # noqa: E402

LEX = load_lexicon()


def rec(text, **kw):
    return {"text": text, "text_check": "asr_phone", "split": "train", **kw}


@pytest.mark.parametrize(
    "text,reason",
    [
        ("わかったよ、ありがとう。", None),
        ("安全教室ですか？", None),
        ("今日は晴れ。", "ambiguous_reading"),
        ("鉛筆を三本ください。", "ambiguous_reading"),
        ("明日香先輩、おはよう。", "proper_noun"),
        ("璃々愛ちゃん、おはよう。", {"unknown_word", "proper_noun"}),
        ("えへへ～♪", "hidden_sounds"),
        ("AIの話", "hidden_sounds"),
    ],
)
def test_reasons(text, reason):
    got = check(rec(text), LEX)[0]
    assert got in reason if isinstance(reason, set) else got == reason


def test_metadata_gates():
    assert check(rec("ありがとう", text_check="unknown"), LEX)[0] == "untrusted_text"
    assert check(rec("ありがとう", phrase_end_mismatch=True), LEX)[0] == "phrase_end_mismatch"
    assert check(rec("ありがとう", split="bench"), LEX)[0] == "not_train_split"


def test_label_is_phonetic_kana():
    reason, ws = check(rec("安全教室ですか？"), LEX)
    assert reason is None and label(ws) == "アンゼンキョーシツデスカ"
