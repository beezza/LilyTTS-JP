"""Split a LilyTTS manifest by title (game), so no title is in both train and eval.

Every line gets a ``split``:

* ``train``: titles not selected for evaluation.
* ``speaker_ft``: speakers with enough audio (default ≥ 2 h) from eval titles,
  for the Speaker-FT benchmark (their own lines are split again later into
  adaptation and test parts).
* ``zeroshot``: other speakers (default ≥ 60 s) from eval titles, for the
  Zero-shot benchmark.
* ``bench``: remaining lines from eval titles: material for Yomi (R4),
  Accent and Basic sets.

Eval titles are chosen deterministically from a hash of the title and the
seed until they cover ``--eval-frac`` of all audio. Titles that alone exceed
``--max-title-frac`` are never chosen, so one huge title cannot eat the
training set.

    python -m data_pipeline.split data/manifest/novelgame.jsonl \\
        -o data/manifest/novelgame.split.jsonl --summary data/manifest/split.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def _duration(rec: dict, default: float) -> tuple[float, bool]:
    d = rec.get("duration")
    return (float(d), True) if d is not None else (default, False)


def plan(
    manifest: str,
    eval_frac: float = 0.03,
    min_eval_titles: int = 10,
    max_title_frac: float = 0.02,
    zeroshot_speakers: int = 100,
    zeroshot_min_sec: float = 60.0,
    ft_speakers: int = 20,
    ft_min_hours: float = 2.0,
    seed: int = 0,
    default_duration: float = 3.0,
) -> dict:
    title_sec: dict[str, float] = defaultdict(float)
    speaker_sec: dict[str, float] = defaultdict(float)
    speaker_title: dict[str, str] = {}
    missing = 0
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            d, known = _duration(r, default_duration)
            missing += not known
            title_sec[r["title"]] += d
            speaker_sec[r["speaker"]] += d
            speaker_title[r["speaker"]] = r["title"]

    total = sum(title_sec.values())
    eval_titles: list[str] = []
    eval_sec = 0.0
    for t in sorted(title_sec, key=lambda t: _rank(seed, t)):
        if eval_sec >= eval_frac * total and len(eval_titles) >= min_eval_titles:
            break
        if title_sec[t] > max_title_frac * total:
            continue
        eval_titles.append(t)
        eval_sec += title_sec[t]
    eval_set = set(eval_titles)

    eval_speakers = sorted((s for s in speaker_sec if speaker_title[s] in eval_set), key=lambda s: _rank(seed, s))
    ft = [s for s in eval_speakers if speaker_sec[s] >= ft_min_hours * 3600][:ft_speakers]
    ft_set = set(ft)
    zs = [s for s in eval_speakers if s not in ft_set and speaker_sec[s] >= zeroshot_min_sec][:zeroshot_speakers]

    return {
        "seed": seed,
        "eval_titles": sorted(eval_titles),
        "speaker_ft": sorted(ft),
        "zeroshot": sorted(zs),
        "total_hours": round(total / 3600, 2),
        "eval_title_hours": round(eval_sec / 3600, 2),
        "lines_without_duration": missing,
        "default_duration": default_duration,
        "warnings": (["fewer speaker_ft speakers than requested"] if len(ft) < ft_speakers else [])
        + (["fewer zeroshot speakers than requested"] if len(zs) < zeroshot_speakers else []),
    }


def apply(manifest: str, output: str, p: dict) -> dict:
    eval_titles = set(p["eval_titles"])
    ft, zs = set(p["speaker_ft"]), set(p["zeroshot"])
    hours: dict[str, float] = defaultdict(float)
    lines: dict[str, int] = defaultdict(int)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, encoding="utf-8") as f, open(output, "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["title"] not in eval_titles:
                split = "train"
            elif r["speaker"] in ft:
                split = "speaker_ft"
            elif r["speaker"] in zs:
                split = "zeroshot"
            else:
                split = "bench"
            r["split"] = split
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            hours[split] += _duration(r, p["default_duration"])[0] / 3600
            lines[split] += 1
    return {k: {"lines": lines[k], "hours": round(hours[k], 2)} for k in sorted(lines)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--summary", help="write the plan and per-split totals here")
    ap.add_argument("--eval-frac", type=float, default=0.03)
    ap.add_argument("--min-eval-titles", type=int, default=10)
    ap.add_argument("--max-title-frac", type=float, default=0.02)
    ap.add_argument("--zeroshot-speakers", type=int, default=100)
    ap.add_argument("--zeroshot-min-sec", type=float, default=60.0)
    ap.add_argument("--ft-speakers", type=int, default=20)
    ap.add_argument("--ft-min-hours", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    p = plan(args.manifest, args.eval_frac, args.min_eval_titles, args.max_title_frac,
             args.zeroshot_speakers, args.zeroshot_min_sec, args.ft_speakers, args.ft_min_hours, args.seed)
    p["splits"] = apply(args.manifest, args.output, p)
    text = json.dumps(p, ensure_ascii=False, indent=2)
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(text + "\n", encoding="utf-8")
    brief = {k: v for k, v in p.items() if k not in ("eval_titles", "speaker_ft", "zeroshot")}
    brief["eval_titles"] = len(p["eval_titles"])
    brief["speaker_ft"] = len(p["speaker_ft"])
    brief["zeroshot"] = len(p["zeroshot"])
    print(json.dumps(brief, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
