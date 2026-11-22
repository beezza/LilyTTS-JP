"""Readings of numerals with counters (助数詞).

Used in two places:

* the labeler, to list the readings a speaker may have used for ``3本`` so the
  recogniser can pick the one actually spoken;
* LilyBench-JP, to generate sentences whose correct reading follows from rules
  (source R2 in docs/LILYBENCH.md).

Readings are written in hiragana in the tables and returned as katakana in
pronunciation form (long vowels as ``ー``, e.g. ``ジューイッポン``).
Where usage varies (``はっぽん`` / ``はちほん``, ``じゅっぽん`` / ``じっぽん``)
every common form is accepted.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from .kana import hira_to_kata, to_pron

# Plain numeral readings, used for compounds (tens prefixes) and confusers.
_TENS_PREFIX = {2: "に", 3: "さん", 4: "よん", 5: "ご", 6: "ろく", 7: "なな", 8: "はち", 9: "きゅう"}
_PLAIN = {1: ["いち"], 2: ["に"], 3: ["さん"], 4: ["よん", "し"], 5: ["ご"], 6: ["ろく"],
          7: ["なな", "しち"], 8: ["はち"], 9: ["きゅう", "く"], 10: ["じゅう"]}


@dataclass(frozen=True)
class Counter:
    """A counter whose readings compose regularly from 1 to 99."""

    surface: str
    base: list[str]  # plain readings of the counter, for confusers (ほん, ぼん, ぽん)
    units: dict[int, list[str]]  # 1..9: number+counter as the last digit
    ten: list[str]  # the counter after a tens digit: 10, 20 = に+ten, ...
    hundred: list[str] = field(default_factory=list)
    what: list[str] = field(default_factory=list)  # 何+counter
    alone: dict[int, list[str]] = field(default_factory=dict)  # overrides when not compounded
    max_n: int = 99


def _c(surface, base, units, ten, hundred=(), what=(), alone=None, max_n=99):
    u = {i + 1: (v if isinstance(v, list) else [v]) for i, v in enumerate(units)}
    return Counter(surface, list(base), u, list(ten), list(hundred), list(what),
                   {k: (v if isinstance(v, list) else [v]) for k, v in (alone or {}).items()}, max_n)


COUNTERS: dict[str, Counter] = {c.surface: c for c in [
    _c("本", ["ほん", "ぼん", "ぽん"],
       ["いっぽん", "にほん", "さんぼん", "よんほん", "ごほん", "ろっぽん", "ななほん", ["はっぽん", "はちほん"], "きゅうほん"],
       ["じゅっぽん", "じっぽん"], ["ひゃっぽん"], ["なんぼん"]),
    _c("匹", ["ひき", "びき", "ぴき"],
       ["いっぴき", "にひき", "さんびき", "よんひき", "ごひき", "ろっぴき", "ななひき", ["はっぴき", "はちひき"], "きゅうひき"],
       ["じゅっぴき", "じっぴき"], ["ひゃっぴき"], ["なんびき"]),
    _c("杯", ["はい", "ばい", "ぱい"],
       ["いっぱい", "にはい", "さんばい", "よんはい", "ごはい", "ろっぱい", "ななはい", ["はっぱい", "はちはい"], "きゅうはい"],
       ["じゅっぱい", "じっぱい"], ["ひゃっぱい"], ["なんばい"]),
    _c("分", ["ふん", "ぷん"],
       ["いっぷん", "にふん", ["さんぷん", "さんふん"], ["よんぷん", "よんふん"], "ごふん", "ろっぷん", "ななふん",
        ["はっぷん", "はちふん"], "きゅうふん"],
       ["じゅっぷん", "じっぷん"], ["ひゃっぷん"], ["なんぷん"], max_n=59),
    _c("回", ["かい"],
       ["いっかい", "にかい", "さんかい", "よんかい", "ごかい", "ろっかい", "ななかい", ["はっかい", "はちかい"], "きゅうかい"],
       ["じゅっかい", "じっかい"], ["ひゃっかい"], ["なんかい"]),
    _c("個", ["こ"],
       ["いっこ", "にこ", "さんこ", "よんこ", "ごこ", "ろっこ", "ななこ", ["はっこ", "はちこ"], "きゅうこ"],
       ["じゅっこ", "じっこ"], ["ひゃっこ"], ["なんこ"]),
    _c("冊", ["さつ"],
       ["いっさつ", "にさつ", "さんさつ", "よんさつ", "ごさつ", "ろくさつ", "ななさつ", ["はっさつ", "はちさつ"], "きゅうさつ"],
       ["じゅっさつ", "じっさつ"], ["ひゃくさつ", "ひゃっさつ"], ["なんさつ"]),
    _c("軒", ["けん", "げん"],
       ["いっけん", "にけん", "さんげん", "よんけん", "ごけん", "ろっけん", "ななけん", ["はっけん", "はちけん"], "きゅうけん"],
       ["じゅっけん", "じっけん"], ["ひゃっけん"], ["なんげん"]),
    _c("階", ["かい", "がい"],
       ["いっかい", "にかい", ["さんがい", "さんかい"], "よんかい", "ごかい", "ろっかい", "ななかい", ["はっかい", "はちかい"],
        "きゅうかい"],
       ["じゅっかい", "じっかい"], ["ひゃっかい"], ["なんがい", "なんかい"]),
    _c("足", ["そく", "ぞく"],
       ["いっそく", "にそく", "さんぞく", "よんそく", "ごそく", "ろくそく", "ななそく", "はっそく", "きゅうそく"],
       ["じゅっそく", "じっそく"], ["ひゃくそく"], ["なんぞく"]),
    _c("歳", ["さい"],
       ["いっさい", "にさい", "さんさい", "よんさい", "ごさい", "ろくさい", "ななさい", "はっさい", "きゅうさい"],
       ["じゅっさい", "じっさい"], ["ひゃくさい"], ["なんさい"],
       alone={20: ["はたち", "にじゅっさい", "にじっさい"]}),
    _c("円", ["えん"],
       ["いちえん", "にえん", "さんえん", "よえん", "ごえん", "ろくえん", "ななえん", "はちえん", "きゅうえん"],
       ["じゅうえん"], ["ひゃくえん"], ["なんえん"]),
    _c("人", ["にん"],
       ["いちにん", "ににん", "さんにん", "よにん", "ごにん", "ろくにん", ["ななにん", "しちにん"], "はちにん", "きゅうにん"],
       ["じゅうにん"], ["ひゃくにん"], ["なんにん"],
       alone={1: ["ひとり"], 2: ["ふたり"]}),
    _c("つ", ["つ"],
       ["ひとつ", "ふたつ", "みっつ", "よっつ", "いつつ", "むっつ", "ななつ", "やっつ", "ここのつ"],
       ["とお"], max_n=10),
]}

MONTHS = {1: ["いちがつ"], 2: ["にがつ"], 3: ["さんがつ"], 4: ["しがつ"], 5: ["ごがつ"], 6: ["ろくがつ"],
          7: ["しちがつ"], 8: ["はちがつ"], 9: ["くがつ"], 10: ["じゅうがつ"], 11: ["じゅういちがつ"],
          12: ["じゅうにがつ"]}
MONTH_CONFUSERS = {4: ["よんがつ", "よがつ"], 7: ["なながつ"], 9: ["きゅうがつ"]}

DAYS = {1: ["ついたち"], 2: ["ふつか"], 3: ["みっか"], 4: ["よっか"], 5: ["いつか"], 6: ["むいか"],
        7: ["なのか", "なぬか"], 8: ["ようか"], 9: ["ここのか"], 10: ["とおか"], 14: ["じゅうよっか"],
        17: ["じゅうしちにち", "じゅうななにち"], 19: ["じゅうくにち", "じゅうきゅうにち"], 20: ["はつか"],
        24: ["にじゅうよっか"], 27: ["にじゅうしちにち", "にじゅうななにち"],
        29: ["にじゅうくにち", "にじゅうきゅうにち"]}
HOURS = {1: ["いちじ"], 2: ["にじ"], 3: ["さんじ"], 4: ["よじ"], 5: ["ごじ"], 6: ["ろくじ"],
         7: ["しちじ", "ななじ"], 8: ["はちじ"], 9: ["くじ"], 10: ["じゅうじ"], 11: ["じゅういちじ"],
         12: ["じゅうにじ"]}
HOUR_CONFUSERS = {4: ["よんじ"], 9: ["きゅうじ"]}


def _tens(n: int) -> str:
    return "じゅう" if n == 1 else _TENS_PREFIX[n] + "じゅう"


def _plain(n: int) -> list[str]:
    """Plain readings of 1..99 (no counter)."""
    if n <= 10:
        return _PLAIN[n]
    t, u = divmod(n, 10)
    head = _tens(t)
    return [head] if u == 0 else [head + r for r in _PLAIN[u]]


def _pron(readings) -> list[str]:
    seen: list[str] = []
    for r in readings:
        p = to_pron(hira_to_kata(r))
        if p not in seen:
            seen.append(p)
    return seen


def day_readings(n: int) -> list[str]:
    """Readings of ``n日`` as a day of the month (1..31)."""
    if not 1 <= n <= 31:
        raise ValueError(n)
    if n in DAYS:
        return _pron(DAYS[n])
    return _pron(r + "にち" for r in _plain(n))


def counter_readings(n: int | None, counter: str) -> list[str]:
    """Accepted readings of ``n`` + ``counter``. ``n=None`` means 何."""
    c = COUNTERS[counter]
    if n is None:
        if not c.what:
            raise ValueError(f"no 何 form for {counter}")
        return _pron(c.what)
    if n in c.alone:
        return _pron(c.alone[n])
    if n == 100 and c.hundred:
        return _pron(c.hundred)
    if not 1 <= n <= c.max_n:
        raise ValueError(f"{n}{counter} is outside the supported range")
    if n < 10:
        return _pron(c.units[n])
    t, u = divmod(n, 10)
    if u == 0:
        prefix = "" if t == 1 else _TENS_PREFIX[t]
        return _pron(prefix + r for r in c.ten)
    return _pron(_tens(t) + r for r in c.units[u])


def counter_confusers(n: int | None, counter: str) -> list[str]:
    """Plausible wrong readings: plain numeral + each counter form, minus the accepted ones."""
    c = COUNTERS[counter]
    gold = set(counter_readings(n, counter))
    nums = ["なん"] if n is None else (_plain(n) if n <= 99 else ["ひゃく"])
    cands = _pron(a + b for a, b in itertools.product(nums, c.base))
    return [x for x in cands if x not in gold]


# --- writing numbers ----------------------------------------------------------

_KANJI_DIGITS = "〇一二三四五六七八九"
_KANJI_VALUE = {c: i for i, c in enumerate(_KANJI_DIGITS)} | {"零": 0, "十": 10, "百": 100, "千": 1000}


def int_to_kanji(n: int) -> str:
    """1..9999 in kanji numerals (二十三, 百五)."""
    if not 1 <= n <= 9999:
        raise ValueError(n)
    out = []
    for value, mark in ((1000, "千"), (100, "百"), (10, "十")):
        d, n = divmod(n, value)
        if d:
            out.append(("" if d == 1 else _KANJI_DIGITS[d]) + mark)
    if n:
        out.append(_KANJI_DIGITS[n])
    return "".join(out)


_NUMERAL_RE = re.compile(r"^[0-9０-９]+$|^[〇一二三四五六七八九十百千]+$")


def parse_numeral(s: str) -> int | None:
    """Parse Arabic (half or full width) or kanji numerals up to 9999."""
    if not s or not _NUMERAL_RE.match(s):
        return None
    if s[0] in "0123456789０１２３４５６７８９":
        return int(s.translate(str.maketrans("０１２３４５６７８９", "0123456789")))
    total, digit = 0, None
    positional = all(c in _KANJI_DIGITS for c in s)
    if positional:  # 二〇二六
        return int("".join(str(_KANJI_VALUE[c]) for c in s))
    for c in s:
        v = _KANJI_VALUE[c]
        if v >= 10:
            total += (1 if digit is None else digit) * v
            digit = None
        else:
            digit = v
    return total + (digit or 0)


# --- months, hours and a single entry point -------------------------------------


def month_readings(n: int) -> list[str]:
    if n not in MONTHS:
        raise ValueError(n)
    return _pron(MONTHS[n])


def hour_readings(n: int) -> list[str]:
    """Readings of ``n時`` for 0..24."""
    if n == 0:
        return _pron(["れいじ"])
    if n in HOURS:
        return _pron(HOURS[n])
    if not 13 <= n <= 24:
        raise ValueError(n)
    t, u = divmod(n, 10)
    if u == 0:
        return _pron([_tens(t) + "じ"])
    extra = {9: ["きゅうじ"]}.get(u, [])
    return _pron(_tens(t) + r for r in HOURS[u] + extra)


UNITS = set(COUNTERS) | {"月", "日", "時"}


def readings_for(n: int | None, unit: str, after_month: bool = False) -> tuple[list[str], list[str]]:
    """(accepted readings, confusers) for a numeral ``n`` (None = 何) and a unit.

    ``日`` after a month is a date (ついたち); otherwise both the date and the
    duration reading (いちにち) are accepted.
    """
    if unit in COUNTERS:
        return counter_readings(n, unit), counter_confusers(n, unit)
    if n is None:
        what = {"月": ["なんがつ"], "日": ["なんにち"], "時": ["なんじ"]}[unit]
        return _pron(what), _pron(["なに" + what[0][2:]])
    plain = _plain(n) if n <= 99 else []
    if unit == "月":
        gold = month_readings(n)
        conf = _pron(MONTH_CONFUSERS.get(n, []) + [p + "がつ" for p in plain])
    elif unit == "日":
        gold = day_readings(n)
        duration = _pron(p + "にち" for p in plain)
        if not after_month:
            gold = gold + [d for d in duration if d not in gold]
        conf = duration + _pron(p + "か" for p in plain)
    elif unit == "時":
        gold = hour_readings(n)
        conf = _pron(HOUR_CONFUSERS.get(n, []) + [p + "じ" for p in plain])
    else:
        raise KeyError(unit)
    return gold, [c for c in dict.fromkeys(conf) if c not in gold]
