"""Convert novel-game voice metadata into the LilyTTS common manifest.

Input: JSONL (one record per line), or JSON files holding one record or a list
of records. Expected source fields (others are ignored)::

    uid / catalog_id, audio_path, text, game_id, character_id,
    performer_id, cleaning_status, duration_sec, duration_is_estimated

Ruby is taken from, in order:

1. ``--ruby-text-field``: a field that holds the text *with* inline ruby
   markup (``｜漢字《かな》``, ``漢字《かな》``, ``{漢字|かな}``, ``<ruby>``,
   ``[ruby text=...]``).
2. ``--ruby-field`` or an auto-detected field (``ruby``, ``rubies``,
   ``furigana``, ``ruby_spans``, ``yomi``): spans, pairs or a dict.
3. Inline markup found in ``text`` itself.

Output record (one per line)::

    {"id", "audio", "text", "ruby": [{"start", "end", "reading"}],
     "speaker": "<game_id>/<character_id>", "title": "<game_id>",
     "voice_actor"?, "duration", "duration_estimated", "src_status"}

Fields from other TTS projects (``frames``, ``split``, ``catalog_id`` ...) are
dropped. In particular the source ``split`` is not used: LilyTTS re-splits by
title so that no game appears in both training and evaluation.

Example::

    python -m data_pipeline.novelgame.to_manifest meta/*.jsonl \\
        -o data/manifest/novelgame.jsonl \\
        --audio-root /mnt/data/dataset/voices \\
        --report data/manifest/novelgame.report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

from lilytts.text.ruby import normalize_ruby_field, parse_inline_ruby

AUTO_RUBY_FIELDS = ("ruby", "rubies", "furigana", "ruby_spans", "yomi")
MAX_WARNING_EXAMPLES = 20


def iter_records(paths: list[str]) -> Iterator[tuple[str, int, dict[str, Any]]]:
    for p in paths:
        with open(p, encoding="utf-8") as f:
            if p.endswith(".jsonl"):
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        yield p, i, json.loads(line)
            else:
                data = json.load(f)
                for i, rec in enumerate(data if isinstance(data, list) else [data], 1):
                    yield p, i, rec


_WARN_VARIABLE = re.compile(r"'[^']*'|\"[^\"]*\"|《[^》]*》|\d+(?::\d+)?")


def warning_kind(w: str) -> str:
    """Group warnings by message shape, ignoring quoted values and numbers."""
    return _WARN_VARIABLE.sub("…", w)


def make_id(rec: dict[str, Any]) -> str:
    if rec.get("uid"):
        return str(rec["uid"])
    if rec.get("catalog_id") is not None:
        return f"cat{rec['catalog_id']}"
    return hashlib.sha256(str(rec.get("audio_path", "")).encode()).hexdigest()


def relativize(path: str, root: str | None) -> str:
    if not root:
        return path
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return path


def extract_ruby(
    rec: dict[str, Any], ruby_field: str | None, ruby_text_field: str | None
) -> tuple[str, list[dict[str, Any]], list[str], str]:
    """Return (plain_text, spans_json, warnings, ruby_source)."""
    text = str(rec.get("text") or "").strip()

    if ruby_text_field and rec.get(ruby_text_field):
        plain, spans, warns = parse_inline_ruby(str(rec[ruby_text_field]).strip())
        if text and plain != text:
            warns.append(f"plain text from {ruby_text_field} differs from text")
        return plain, [s.to_json() for s in spans], warns, ruby_text_field

    field = ruby_field or next((k for k in AUTO_RUBY_FIELDS if rec.get(k)), None)
    plain, inline_spans, warns = parse_inline_ruby(text)

    if field and rec.get(field) is not None:
        if inline_spans:
            warns.append(f"text has inline ruby and field {field!r}; using inline")
            return plain, [s.to_json() for s in inline_spans], warns, "inline"
        spans, field_warns = normalize_ruby_field(rec[field], plain)
        return plain, [s.to_json() for s in spans], warns + field_warns, field

    return plain, [s.to_json() for s in inline_spans], warns, "inline" if inline_spans else "none"


def probe_durations(paths: list[str], audio_root: str | None, workers: int) -> list[float | None]:
    try:
        import soundfile as sf
    except ImportError:
        sys.exit("--probe-duration needs soundfile: pip install soundfile")

    def probe(p: str) -> float | None:
        full = p if os.path.isabs(p) or not audio_root else os.path.join(audio_root, p)
        try:
            return float(sf.info(full).duration)
        except Exception:
            return None

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(probe, paths))


def convert(args: argparse.Namespace) -> dict[str, Any]:
    keep_status = set(args.keep_status.split(",")) if args.keep_status else None
    stats: dict[str, Any] = {
        "records_in": 0,
        "records_out": 0,
        "rejected": Counter(),
        "status": Counter(),
        "ruby_source": Counter(),
        "records_with_ruby": 0,
        "ruby_spans": 0,
        "warnings": Counter(),
        "warning_examples": [],
        "source_keys": Counter(),
    }
    speaker_sec: dict[str, float] = defaultdict(float)
    title_sec: dict[str, float] = defaultdict(float)
    out_records: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []

    for src, line_no, rec in iter_records(args.inputs):
        stats["records_in"] += 1
        stats["source_keys"].update(rec.keys())
        status = rec.get("cleaning_status")
        stats["status"][str(status)] += 1

        reason = None
        if not rec.get("audio_path"):
            reason = "missing_audio_path"
        elif not str(rec.get("text") or "").strip() and not (
            args.ruby_text_field and rec.get(args.ruby_text_field)
        ):
            reason = "empty_text"
        elif not rec.get("game_id") or not rec.get("character_id"):
            reason = "missing_game_or_character"
        elif keep_status is not None and str(status) not in keep_status:
            reason = "status_filtered"
        if reason:
            stats["rejected"][reason] += 1
            rejects.append({"src": src, "line": line_no, "reason": reason, "id": make_id(rec)})
            continue

        plain, spans, warns, ruby_source = extract_ruby(rec, args.ruby_field, args.ruby_text_field)
        if not plain:
            stats["rejected"]["empty_text_after_ruby"] += 1
            rejects.append({"src": src, "line": line_no, "reason": "empty_text_after_ruby", "id": make_id(rec)})
            continue
        for w in warns:
            stats["warnings"][warning_kind(w)] += 1
            if len(stats["warning_examples"]) < MAX_WARNING_EXAMPLES:
                stats["warning_examples"].append({"id": make_id(rec), "warning": w})
        stats["ruby_source"][ruby_source] += 1
        if spans:
            stats["records_with_ruby"] += 1
            stats["ruby_spans"] += len(spans)

        game, char = str(rec["game_id"]), str(rec["character_id"])
        out = {
            "id": make_id(rec),
            "audio": relativize(str(rec["audio_path"]), args.audio_root),
            "text": plain,
            "ruby": spans,
            "speaker": f"{game}/{char}",
            "title": game,
        }
        if rec.get("performer_id"):
            out["voice_actor"] = str(rec["performer_id"])
        dur = rec.get("duration_sec")
        out["duration"] = None if dur is None else float(dur)
        out["duration_estimated"] = bool(rec.get("duration_is_estimated", dur is None))
        if status is not None:
            out["src_status"] = str(status)
        out_records.append(out)

    if args.probe_duration and out_records:
        durs = probe_durations([r["audio"] for r in out_records], args.audio_root, args.workers)
        for r, d in zip(out_records, durs):
            if d is None:
                stats["warnings"]["audio unreadable"] += 1
            else:
                r["duration"], r["duration_estimated"] = round(d, 4), False

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r["duration"] is not None:
                speaker_sec[r["speaker"]] += r["duration"]
                title_sec[r["title"]] += r["duration"]
    stats["records_out"] = len(out_records)

    if args.rejects:
        with open(args.rejects, "w", encoding="utf-8") as f:
            for r in rejects:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_sec = sum(speaker_sec.values())
    report = {
        **{k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()},
        "hours": round(total_sec / 3600, 2),
        "titles": len(title_sec),
        "speakers": len(speaker_sec),
        "top_speakers_hours": [
            [s, round(sec / 3600, 2)]
            for s, sec in sorted(speaker_sec.items(), key=lambda x: -x[1])[: args.top]
        ],
        "top_titles_hours": [
            [t, round(sec / 3600, 2)]
            for t, sec in sorted(title_sec.items(), key=lambda x: -x[1])[: args.top]
        ],
    }
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="source .jsonl / .json files")
    ap.add_argument("-o", "--output", required=True, help="output manifest .jsonl")
    ap.add_argument("--audio-root", help="store audio paths relative to this directory")
    ap.add_argument("--ruby-field", help="source field with ruby spans/pairs/dict (default: auto-detect)")
    ap.add_argument("--ruby-text-field", help="source field with text containing inline ruby markup")
    ap.add_argument("--keep-status", help="comma-separated cleaning_status values to keep (default: all)")
    ap.add_argument("--probe-duration", action="store_true", help="read real durations from audio files")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--report", help="write a JSON summary here (also printed)")
    ap.add_argument("--rejects", help="write rejected records here")
    ap.add_argument("--top", type=int, default=20, help="rows in top speaker/title lists")
    args = ap.parse_args(argv)

    report = convert(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
