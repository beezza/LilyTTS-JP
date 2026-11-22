"""Kana CTC recogniser: fine-tuning and inference.

Initialised from ``reazon-research/japanese-wav2vec2-base-rs35kh`` (character
CTC on ReazonSpeech 35k h); its output layer is replaced by the kana
vocabulary in :mod:`lilytts.labeler.vocab`.

Needs the ``train`` extra (torch, torchaudio, transformers, soundfile).

Train (RTX 5070 Ti, bf16)::

    python -m lilytts.labeler.recognizer train \\
        --teacher data/labels/teacher_round0.jsonl --audio-root /mnt/data/dataset/voices \\
        --out runs/recognizer_r0 --max-batch-seconds 160 --steps 30000

Inference returns per-frame log-probabilities for
:class:`lilytts.labeler.reading.CTCScorer`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np

from .batching import cer, duration_batches
from .vocab import KanaVocab

DEFAULT_INIT = "reazon-research/japanese-wav2vec2-base-rs35kh"
SAMPLE_RATE = 16000


def _torch():
    import torch

    return torch


def load_audio(path: str, audio_root: str | None = None) -> np.ndarray:
    """Mono float32 at 16 kHz."""
    import soundfile as sf

    full = path if os.path.isabs(path) or not audio_root else os.path.join(audio_root, path)
    wav, sr = sf.read(full, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        import torchaudio.functional as AF

        torch = _torch()
        wav = AF.resample(torch.from_numpy(wav), sr, SAMPLE_RATE).numpy()
    return wav


def build(init: str = DEFAULT_INIT, vocab: KanaVocab | None = None):
    """Model and feature extractor with a kana output layer."""
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    vocab = vocab or KanaVocab()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(init)
    model = Wav2Vec2ForCTC.from_pretrained(
        init,
        vocab_size=len(vocab),
        pad_token_id=vocab.blank,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        ignore_mismatched_sizes=True,  # new lm_head for the kana vocabulary
    )
    model.freeze_feature_encoder()
    return model, fe


def _collate(fe, wavs: list[np.ndarray], labels: list[list[int]] | None, device):
    torch = _torch()
    feats = fe(wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True, return_attention_mask=True)
    batch = {"input_values": feats.input_values.to(device), "attention_mask": feats.attention_mask.to(device)}
    if labels is not None:
        L = max(len(l) for l in labels)
        lab = torch.full((len(labels), L), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            lab[i, : len(l)] = torch.tensor(l)
        batch["labels"] = lab.to(device)
    return batch


def log_probs(model, fe, wavs: list[np.ndarray], device=None) -> list[np.ndarray]:
    """Per-utterance (T, V) log-probabilities, padding frames removed."""
    torch = _torch()
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        batch = _collate(fe, wavs, None, device)
        logits = model(**batch).logits.float()
    lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
    lengths = model._get_feat_extract_output_lengths(batch["attention_mask"].sum(-1)).cpu().numpy()
    return [lp[i, : int(n)] for i, n in enumerate(lengths)]


def _greedy(lp: np.ndarray, vocab: KanaVocab) -> str:
    from .ctc import greedy_decode

    return vocab.decode(greedy_decode(lp, vocab.blank))


def train(args: argparse.Namespace) -> None:
    torch = _torch()
    vocab = KanaVocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [json.loads(l) for l in open(args.teacher, encoding="utf-8") if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_dev = min(args.dev_lines, len(rows) // 20 or 1)
    dev, rows = rows[:n_dev], rows[n_dev:]
    durations = [r.get("duration") or 3.0 for r in rows]

    model, fe = build(args.init, vocab)
    model.to(device)
    head = [p for n, p in model.named_parameters() if n.startswith("lm_head") and p.requires_grad]
    body = [p for n, p in model.named_parameters() if not n.startswith("lm_head") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / args.steps)))
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps(vocab.to_hf_vocab(), ensure_ascii=False))

    step, best = 0, float("inf")
    epoch = 0
    while step < args.steps:
        for batch_idx in duration_batches(durations, args.max_batch_seconds, seed=args.seed + epoch):
            items = [rows[i] for i in batch_idx]
            wavs = [load_audio(r["audio"], args.audio_root) for r in items]
            labels = [vocab.encode(r["kana"]) for r in items]
            model.train()
            batch = _collate(fe, wavs, labels, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                print(json.dumps({"step": step, "loss": round(loss.item(), 4), "lr": sched.get_last_lr()[0]}))
            if step % args.eval_every == 0 or step == args.steps:
                score = evaluate(model, fe, dev, vocab, args.audio_root)
                print(json.dumps({"step": step, "dev_kana_cer": round(score, 4)}))
                if score < best:
                    best = score
                    model.save_pretrained(out / "best")
                    fe.save_pretrained(out / "best")
                model.save_pretrained(out / "last")
                fe.save_pretrained(out / "last")
            if step >= args.steps:
                break
        epoch += 1


def evaluate(model, fe, dev: list[dict], vocab: KanaVocab, audio_root: str | None, batch: int = 16) -> float:
    refs, hyps = [], []
    for i in range(0, len(dev), batch):
        items = dev[i:i + batch]
        lps = log_probs(model, fe, [load_audio(r["audio"], audio_root) for r in items])
        refs += [vocab.decode(vocab.encode(r["kana"])) for r in items]
        hyps += [_greedy(lp, vocab) for lp in lps]
    return cer(refs, hyps)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--teacher", required=True)
    t.add_argument("--audio-root")
    t.add_argument("--out", required=True)
    t.add_argument("--init", default=DEFAULT_INIT)
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--max-batch-seconds", type=float, default=160.0)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--head-lr", type=float, default=1e-3)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--dev-lines", type=int, default=2000)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "train":
        train(args)


if __name__ == "__main__":
    main()
