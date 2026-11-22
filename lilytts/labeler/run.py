"""Label a manifest with readings recovered from audio.

GPU: the recogniser computes log-probabilities in batches. CPU: a process pool
runs OpenJTalk and candidate scoring (:func:`lilytts.labeler.reading.recover_reading`).

    python -m lilytts.labeler.run data/manifest/novelgame.split.jsonl \\
        --model runs/recognizer_r0/best --audio-root /mnt/data/dataset/voices \\
        -o data/labels/readings_r0.jsonl --disagreements data/labels/openjtalk_errors_r0.tsv

Output lines: ``{id, kana, pk, openjtalk_kana, changed, margin, fit, slots}``.
``--disagreements`` aggregates, per word, how often the audio disagreed with
OpenJTalk: the dictionary-error list used for Yomi and LilyBench-JP (R4).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .batching import duration_batches


def _recover(args: tuple[str, str, np.ndarray]) -> dict:
    from .reading import CTCScorer, recover_reading

    rid, text, lp = args
    r = recover_reading(text, CTCScorer(lp))
    d = asdict(r)
    d["id"] = rid
    for s in d["slots"]:
        s["scores"] = [round(x, 3) for x in s["scores"]]
    return d


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio-root")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--disagreements")
    ap.add_argument("--splits", default="", help="comma-separated splits to label (default: all)")
    ap.add_argument("--max-batch-seconds", type=float, default=200.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    from .recognizer import load_audio, log_probs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.model)

    keep = set(filter(None, args.splits.split(",")))
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if not keep or r.get("split") in keep]
    batches = duration_batches([r.get("duration") or 3.0 for r in rows], args.max_batch_seconds, shuffle=False)

    disagree: Counter = Counter()
    n_changed = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool, open(args.output, "w", encoding="utf-8") as out:
        for b in batches:
            items = [rows[i] for i in b]
            lps = log_probs(model, fe, [load_audio(r["audio"], args.audio_root) for r in items], device)
            for d in pool.map(_recover, [(r["id"], r["text"], lp) for r, lp in zip(items, lps)]):
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_changed += d["changed"]
                for s in d["slots"]:
                    if s["chosen"] != s["default"]:
                        disagree[(s["surface"], s["default"], s["chosen"])] += 1
    if args.disagreements:
        with open(args.disagreements, "w", encoding="utf-8") as f:
            f.write("surface\topenjtalk\taudio\tcount\n")
            for (surface, default, chosen), c in disagree.most_common():
                f.write(f"{surface}\t{default}\t{chosen}\t{c}\n")
    print(json.dumps({"lines": len(rows), "changed": n_changed, "distinct_disagreements": len(disagree)}))


if __name__ == "__main__":
    main()
