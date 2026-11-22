"""Ruby (furigana) extraction.

Two entry points:

* :func:`parse_inline_ruby` removes ruby markup from a text and returns the
  plain text plus character spans. It understands the notations that show up
  in novel-game scripts and Aozora Bunko text:

  - ``｜日本橋《にっぽんばし》`` / ``|日本橋《にっぽんばし》`` (explicit base)
  - ``今日《こんにち》`` (implicit base: the run of kanji right before ``《``)
  - ``{今日|こんにち}`` (LilyTTS user markup)
  - ``<ruby>今日<rt>こんにち</rt></ruby>`` (HTML, ``<rp>``/``<rb>`` allowed)
  - ``[ruby text="きょう"]今`` (KiriKiri/KAG: applies to the next character)

* :func:`normalize_ruby_field` turns a ruby field that is stored next to the
  text (a list of dicts, a list of pairs, or a dict) into the same spans.

Readings are normalised to katakana. Spans whose reading is not kana are
dropped with a warning, and emphasis dots (傍点 written as ruby, e.g.
``《・・》``) are dropped silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .kana import hira_to_kata, is_kana_string, is_kanji

_EMPHASIS_CHARS = set("・﹅﹆゛ヽ●○")


@dataclass(frozen=True)
class RubySpan:
    start: int
    end: int  # exclusive
    reading: str  # katakana

    def to_json(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "reading": self.reading}


_INLINE = re.compile(
    r"<ruby>(?P<html>.*?)</ruby>"
    r"|[｜|](?P<xbase>[^｜|《》\n]+?)《(?P<xread>[^《》\n]+)》"
    r"|\{(?P<bbase>[^{}|\n]+)\|(?P<bread>[^{}|\n]+)\}"
    r"|\[ruby\s+text\s*=\s*\"?(?P<kread>[^\"\]\n]+?)\"?\s*\](?P<kbase>[^\[\n])"
    r"|《(?P<iread>[^《》\n]+)》",
    re.DOTALL,
)
_RT = re.compile(r"<rt>(.*?)</rt>", re.DOTALL)
_RP = re.compile(r"<rp>.*?</rp>", re.DOTALL)
_TAG = re.compile(r"</?rb>|</?rtc>")


def _clean_reading(raw: str) -> tuple[str | None, str | None]:
    """Return (katakana reading, warning). Both None means: drop silently."""
    r = re.sub(r"\s+", "", raw)
    if r and all(c in _EMPHASIS_CHARS for c in r):
        return None, None
    r = hira_to_kata(r)
    if not is_kana_string(r):
        return None, f"non-kana reading {raw!r}"
    return r, None


def parse_inline_ruby(text: str) -> tuple[str, list[RubySpan], list[str]]:
    """Strip inline ruby markup.

    Returns ``(plain_text, spans, warnings)``. Spans index into ``plain_text``.
    """
    out: list[str] = []
    spans: list[RubySpan] = []
    warnings: list[str] = []
    pos = 0
    plain_len = 0

    def emit(s: str) -> None:
        nonlocal plain_len
        out.append(s)
        plain_len += len(s)

    for m in _INLINE.finditer(text):
        emit(text[pos : m.start()])
        pos = m.end()

        if m.group("iread") is not None:
            # Implicit base: trailing kanji run of the plain text so far.
            plain_so_far = "".join(out)
            k = len(plain_so_far)
            floor = spans[-1].end if spans else 0  # never reach into the previous ruby
            while k > floor and is_kanji(plain_so_far[k - 1]):
                k -= 1
            if k == len(plain_so_far):
                warnings.append(f"《{m.group('iread')}》 has no kanji base; kept as text")
                emit(m.group(0))
                continue
            reading, warn = _clean_reading(m.group("iread"))
            if warn:
                warnings.append(warn)
            if reading:
                spans.append(RubySpan(k, plain_len, reading))
            continue

        if m.group("html") is not None:
            inner = _RP.sub("", m.group("html"))
            raw_read = "".join(_RT.findall(inner))
            base = _TAG.sub("", _RT.sub("", inner))
        elif m.group("xbase") is not None:
            base, raw_read = m.group("xbase"), m.group("xread")
        elif m.group("bbase") is not None:
            base, raw_read = m.group("bbase"), m.group("bread")
        else:
            base, raw_read = m.group("kbase"), m.group("kread")

        start = plain_len
        emit(base)
        if not base:
            warnings.append(f"empty ruby base in {m.group(0)!r}")
            continue
        reading, warn = _clean_reading(raw_read)
        if warn:
            warnings.append(warn)
        if reading:
            spans.append(RubySpan(start, plain_len, reading))

    emit(text[pos:])
    plain = "".join(out)
    spans, overlap_warnings = _drop_overlaps(spans)
    return plain, spans, warnings + overlap_warnings


_READING_KEYS = ("reading", "ruby", "kana", "yomi", "rt")
_BASE_KEYS = ("base", "surface", "text", "kanji", "rb")
_START_KEYS = ("start", "begin", "offset", "pos")
_END_KEYS = ("end", "stop")


def _first(d: dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def normalize_ruby_field(value: Any, text: str) -> tuple[list[RubySpan], list[str]]:
    """Convert a stored ruby field into spans over ``text``.

    Accepted shapes:

    * ``[{"start": 0, "end": 2, "reading": "こんにち"}, ...]``
    * ``[{"base": "今日", "reading": "こんにち"}, ...]`` (located left to right)
    * ``[["今日", "こんにち"], ...]``
    * ``{"今日": "こんにち", ...}`` (every occurrence, left to right)
    """
    if value is None or value == "" or value == [] or value == {}:
        return [], []
    warnings: list[str] = []
    items: list[tuple[int | None, int | None, str | None, str]] = []

    if isinstance(value, dict):
        for base, reading in value.items():
            start = text.find(base)
            if start < 0:
                warnings.append(f"ruby base {base!r} not found in text")
            while start >= 0:
                items.append((start, start + len(base), base, str(reading)))
                start = text.find(base, start + len(base))
    elif isinstance(value, list):
        for it in value:
            if isinstance(it, dict):
                reading = _first(it, _READING_KEYS)
                if reading is None:
                    warnings.append(f"ruby entry without reading: {it!r}")
                    continue
                start, end = _first(it, _START_KEYS), _first(it, _END_KEYS)
                base = _first(it, _BASE_KEYS)
                if end is None and start is not None and base is not None:
                    end = int(start) + len(base)
                items.append(
                    (
                        None if start is None else int(start),
                        None if end is None else int(end),
                        base,
                        str(reading),
                    )
                )
            elif isinstance(it, (list, tuple)) and len(it) == 2:
                items.append((None, None, str(it[0]), str(it[1])))
            else:
                warnings.append(f"unrecognised ruby entry: {it!r}")
    else:
        warnings.append(f"unrecognised ruby field type: {type(value).__name__}")
        return [], warnings

    spans: list[RubySpan] = []
    cursor = 0
    for start, end, base, raw_read in items:
        if start is None or end is None:
            if not base:
                warnings.append("ruby entry has neither offsets nor base")
                continue
            found = text.find(base, cursor)
            if found < 0:
                found = text.find(base)
            if found < 0:
                warnings.append(f"ruby base {base!r} not found in text")
                continue
            start, end = found, found + len(base)
        if not (0 <= start < end <= len(text)):
            warnings.append(f"ruby span {start}:{end} out of range for text of length {len(text)}")
            continue
        if base is not None and text[start:end] != base:
            warnings.append(f"ruby base {base!r} does not match text[{start}:{end}]={text[start:end]!r}")
            continue
        reading, warn = _clean_reading(raw_read)
        if warn:
            warnings.append(warn)
        if reading:
            spans.append(RubySpan(start, end, reading))
        cursor = end

    spans, overlap_warnings = _drop_overlaps(spans)
    return spans, warnings + overlap_warnings


def _drop_overlaps(spans: list[RubySpan]) -> tuple[list[RubySpan], list[str]]:
    spans = sorted(set(spans), key=lambda s: (s.start, s.end))
    kept: list[RubySpan] = []
    warnings: list[str] = []
    for s in spans:
        if kept and s.start < kept[-1].end:
            warnings.append(f"overlapping ruby span {s.start}:{s.end} dropped")
            continue
        kept.append(s)
    return kept, warnings
