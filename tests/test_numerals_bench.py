import pytest

from bench.build.numerals import build
from bench.score.span_judge import CORRECT, UNDETERMINED, WRONG, judge_span
from lilytts.text.numerals import counter_readings, day_readings, hour_readings, parse_numeral, readings_for


@pytest.mark.parametrize(
    "n,c,expected",
    [
        (1, "本", ["イッポン"]),
        (3, "本", ["サンボン"]),
        (6, "匹", ["ロッピキ"]),
        (8, "杯", ["ハッパイ", "ハチハイ"]),
        (11, "本", ["ジューイッポン"]),
        (20, "本", ["ニジュッポン", "ニジッポン"]),
        (100, "本", ["ヒャッポン"]),
        (None, "本", ["ナンボン"]),
        (1, "人", ["ヒトリ"]),
        (4, "人", ["ヨニン"]),
        (24, "人", ["ニジューヨニン"]),
        (20, "歳", ["ハタチ", "ニジュッサイ", "ニジッサイ"]),
        (4, "円", ["ヨエン"]),
        (3, "軒", ["サンゲン"]),
        (5, "つ", ["イツツ"]),
    ],
)
def test_counter_readings(n, c, expected):
    assert counter_readings(n, c) == expected


def test_days_hours_and_confusers():
    assert day_readings(1) == ["ツイタチ"] and day_readings(20) == ["ハツカ"] and day_readings(21) == ["ニジューイチニチ"]
    assert hour_readings(4) == ["ヨジ"] and "ジューヨジ" in hour_readings(14)
    gold, conf = readings_for(1, "日", after_month=False)
    assert {"ツイタチ", "イチニチ"} <= set(gold)
    gold, conf = readings_for(1, "日", after_month=True)
    assert gold == ["ツイタチ"] and "イチニチ" in conf
    gold, conf = readings_for(None, "本")
    assert "ナニボン" in conf and not set(gold) & set(conf)


def test_parse_numeral():
    assert [parse_numeral(s) for s in ("13", "１３", "十三", "二十", "百五", "二〇二六", "abc")] == [13, 13, 13, 20, 105, 2026, None]


def test_judge_span_handles_word_boundaries_and_long_vowels():
    # デ + イッカイ must not be merged into a long vowel
    assert judge_span("エレベーターデイッカイニイク", ["イッカイ"], ["イチカイ"]) == CORRECT
    assert judge_span("キョウワ", ["キョー"], ["コンニチ"]) == CORRECT
    assert judge_span("コンニチワ", ["キョー"], ["コンニチ"]) == WRONG
    assert judge_span("アシタワ", ["キョー"], ["コンニチ"]) == UNDETERMINED


def test_built_items_are_consistent():
    items = build(seed=0)
    assert len(items) > 200
    assert len({it["id"] for it in items}) == len(items)
    for it in items:
        for t in it["targets"]:
            s, e = t["span"]
            surface = it["text"][s:e]
            assert surface and surface[-1] in "本匹杯分回個冊軒階足歳円人つ月日時", (it["id"], surface)
            assert t["gold"] and not set(t["gold"]) & set(t["confusers"])
