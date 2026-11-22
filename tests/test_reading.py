import numpy as np
import pytest

pytest.importorskip("pyopenjtalk")

from lilytts.labeler.reading import CTCScorer, find_slots, load_lexicon, recover_reading  # noqa: E402
from lilytts.labeler.vocab import KanaVocab  # noqa: E402
from lilytts.text.openjtalk import words  # noqa: E402
from lilytts.text.pk import parse_pk  # noqa: E402

VOCAB = KanaVocab()


def spoken(kana, frames_per_char=3, p=0.95):
    """Fake recogniser output that clearly says ``kana``."""
    ids = VOCAB.encode(kana)
    T = len(ids) * frames_per_char
    lp = np.full((T, len(VOCAB)), (1 - p) / (len(VOCAB) - 1))
    for t in range(T):
        tok = ids[t // frames_per_char] if t % frames_per_char == 0 else VOCAB.blank
        lp[t, tok] = p
    return CTCScorer(np.log(lp), VOCAB)


def test_homographs_follow_the_audio():
    r = recover_reading("今日は日本橋で", spoken("コンニチワニッポンバシデ"))
    assert r.kana == "コンニチワニッポンバシデ"
    assert r.openjtalk_kana == "キョーワニホンバシデ"
    assert r.changed and r.margin > 0
    assert [(s.surface, s.default, s.chosen) for s in r.slots] == [
        ("今日", "キョー", "コンニチ"),
        ("日本橋", "ニホンバシ", "ニッポンバシ"),
    ]
    assert parse_pk(r.pk).kana == "コンニチワニッポンバシデ"


def test_audio_agreeing_with_openjtalk_is_unchanged():
    r = recover_reading("今日は日本橋で", spoken("キョーワニホンバシデ"))
    assert not r.changed and r.kana == r.openjtalk_kana


def test_numeral_slot_follows_the_audio():
    # OpenJTalk's default for 何本 depends on whether its 何 predictor
    # (onnxruntime) is installed, so only the chosen reading is checked.
    r = recover_reading("何本ある？", spoken("ナンボンアル"))
    (slot,) = [s for s in r.slots if s.kind == "numeral"]
    assert slot.surface == "何本" and slot.chosen == "ナンボン" and slot.chosen in slot.accepted

    r = recover_reading("靴下を四足買った", spoken("クツシタオヨンソクカッタ"))
    (slot,) = [s for s in r.slots if s.kind == "numeral"]
    assert (slot.surface, slot.default, slot.chosen) == ("四足", "シソク", "ヨンソク")
    assert r.changed


def test_multi_word_numeral_and_dates():
    ws = words("鉛筆を13本買った")
    (slot,) = find_slots(ws, load_lexicon())
    assert slot.kind == "numeral" and "ジューサンボン" in slot.accepted

    ws = words("3月1日に会おう")
    slots = {"".join(w.surface for w in ws[s.first:s.last]): s for s in find_slots(ws, load_lexicon())}
    assert slots["一日"].accepted == ["ツイタチ"]  # a date after 月
    assert "サンガツ" in slots["３月"].accepted

    ws = words("1日中寝ていた")
    (slot,) = [s for s in find_slots(ws, load_lexicon()) if s.kind == "numeral"]
    assert {"ツイタチ", "イチニチ"} <= set(slot.accepted)  # date or duration


def test_poor_fit_when_audio_says_something_else():
    good = recover_reading("今日は", spoken("キョーワ"))
    bad = recover_reading("今日は", spoken("アシタワ"))
    assert bad.fit < good.fit
