import pytest

pytest.importorskip("pyopenjtalk")

from lilytts.text.openjtalk import label_phrases, text_to_pk, words, words_to_sentence  # noqa: E402

SENTENCES = [
    "今日は良い天気ですね。",
    "安全教室ですか？",
    "今日は日本橋で3本買った。",
    "えっと、明日は東京駅に行きます。それから大阪へ。",
    "ねえ、どうしたの？",
    "AIの研究をしています。",
    "あ、う、え。",
]


def _norm(phrases):
    # A nucleus on the last mora and a flat phrase have the same pitch inside the
    # phrase; OpenJTalk labels write flat phrases that way.
    return [(n, 0 if a == n else a) for n, a in phrases]


@pytest.mark.parametrize("text", SENTENCES)
def test_phrases_match_openjtalk_labels(text):
    s = words_to_sentence(words(text))
    assert _norm([(len(p.morae), p.nucleus) for p in s.phrases]) == _norm(label_phrases(text))


def test_known_outputs():
    assert text_to_pk("今日は良い天気ですね。") == "^キョ]ーワ#ヨ]イ#テ]ンキデスネ$"
    assert text_to_pk("安全教室ですか？") == "^ア[ンゼンキョ]ーシツデスカ?"


def test_word_spans_and_prons():
    ws = words("今日は日本橋で")
    assert [(w.surface, w.pron, w.start, w.end) for w in ws] == [
        ("今日", "キョー", 0, 2),
        ("は", "ワ", 2, 3),
        ("日本橋", "ニホンバシ", 3, 6),
        ("で", "デ", 6, 7),
    ]


def test_alternative_prons_keep_phrasing():
    ws = words("今日は日本橋で")
    prons = [w.pron for w in ws]
    prons[2] = "ニッポンバシ"
    s = words_to_sentence(ws, prons)
    assert s.kana == "キョーワニッポンバシデ"
