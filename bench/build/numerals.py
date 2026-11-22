"""LilyBench-JP Yomi: sentences with numerals and counters (source R2).

Correct readings follow from the rules in :mod:`lilytts.text.numerals`, so no
human labelling is needed. Each item has one or more targets: a character span,
the accepted readings and plausible wrong readings (confusers).

    python -m bench.build.numerals -o bench/data/yomi_numerals.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from lilytts.text.numerals import COUNTERS, int_to_kanji, readings_for

TEMPLATES: dict[str, list[str]] = {
    "本": ["鉛筆を{x}買いました。", "傘が{x}置いてある。", "ペンを{x}ください。"],
    "匹": ["猫が{x}いる。", "金魚を{x}飼っている。"],
    "杯": ["コーヒーを{x}飲んだ。", "ご飯を{x}食べた。"],
    "分": ["あと{x}で着きます。", "{x}だけ待ってください。"],
    "回": ["もう{x}挑戦した。", "週に{x}練習する。"],
    "個": ["りんごを{x}買った。", "消しゴムが{x}ある。"],
    "冊": ["本を{x}読んだ。", "ノートを{x}用意して。"],
    "軒": ["家が{x}並んでいる。", "店を{x}回った。"],
    "階": ["エレベーターで{x}に行く。", "{x}の部屋です。"],
    "足": ["靴下を{x}買った。", "靴を{x}持っている。"],
    "歳": ["弟は{x}になった。", "彼女は{x}だそうだ。"],
    "円": ["これは{x}です。", "あと{x}足りない。"],
    "人": ["子どもが{x}いる。", "{x}で行こう。"],
    "つ": ["箱が{x}ある。", "{x}選んでください。"],
    "時": ["{x}に集合ね。", "今朝は{x}に起きた。"],
}
# Kanji numeral + unit spellings that are also ordinary words.
AMBIGUOUS_WORDS = {"十分", "一日", "一時", "一足", "一個", "一本", "一杯", "一人", "一回", "一軒"}

DATE_TEMPLATES = ["{m}月{d}日に会おう。", "誕生日は{m}月{d}日です。"]


def _write(n: int, style: str) -> str:
    return str(n) if style == "digit" else int_to_kanji(n)


def _numbers(unit: str, rng: random.Random) -> list[int | None]:
    c = COUNTERS.get(unit)
    hi = c.max_n if c else 24
    # 10 + つ is written 十 alone (とお), not 十つ, so つ stops at 9.
    ns: list[int | None] = list(range(1, min(hi, 9 if unit == "つ" else 10) + 1))
    if hi > 10:
        # tricky last digits (1, 3, 6, 8, 0) are oversampled
        pool = [n for n in range(11, hi + 1) if n % 10 in (0, 1, 3, 4, 6, 8)]
        ns += rng.sample(pool, min(6, len(pool)))
        ns += rng.sample([n for n in range(11, hi + 1) if n not in ns], 3)
    if c and c.hundred:
        ns.append(100)
    if unit == "時":
        ns.append(0)
    if (c and c.what) or unit == "時":
        ns.append(None)
    return ns


def build(seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    items: list[dict] = []
    for unit, templates in TEMPLATES.items():
        for n in _numbers(unit, rng):
            style = "what" if n is None else rng.choice(["digit", "digit", "kanji"])
            if style == "kanji" and int_to_kanji(n) + unit in AMBIGUOUS_WORDS:
                style = "digit"  # 十分 could be じゅうぶん
            numeral = "何" if n is None else _write(n, style)
            tmpl_i = rng.randrange(len(templates))
            tmpl = templates[tmpl_i]
            x = numeral + unit
            text = tmpl.format(x=x)
            start = tmpl.index("{x}")
            gold, conf = readings_for(n, unit)
            if not conf:
                continue  # nothing to discriminate (e.g. 3分: さんぷん / さんふん both fine)
            items.append({
                "id": f"num-{unit}-{'what' if n is None else n}-{style}-{tmpl_i}",
                "text": text,
                "targets": [{"span": [start, start + len(x)], "gold": gold, "confusers": conf}],
                "category": "numeral",
                "subcategory": unit,
                "source": "rule",
            })
    months = list(range(1, 13))
    days = list(range(1, 11)) + [14, 17, 19, 20, 24, 27, 29, 31] + rng.sample(range(11, 31), 4)
    for d in days:
        m = rng.choice(months)
        style = rng.choice(["digit", "digit", "kanji"])
        tmpl_i = rng.randrange(len(DATE_TEMPLATES))
        tmpl = DATE_TEMPLATES[tmpl_i]
        ms, ds = _write(m, style), _write(d, style)
        text = tmpl.format(m=ms, d=ds)
        m_start = tmpl.index("{m}")
        d_start = m_start + len(ms) + 1
        mg, mc = readings_for(m, "月")
        dg, dc = readings_for(d, "日", after_month=True)
        targets = [{"span": [d_start, d_start + len(ds) + 1], "gold": dg, "confusers": dc}]
        if mc:
            targets.insert(0, {"span": [m_start, m_start + len(ms) + 1], "gold": mg, "confusers": mc})
        items.append({
            "id": f"num-date-{m}-{d}-{style}-{tmpl_i}",
            "text": text,
            "targets": targets,
            "category": "numeral",
            "subcategory": "日付",
            "source": "rule",
        })
    return items


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    items = build(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} items -> {args.output}")


if __name__ == "__main__":
    main()
