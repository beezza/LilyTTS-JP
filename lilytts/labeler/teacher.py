"""Select lines whose OpenJTalk reading can be trusted as a recogniser teacher.

The novel-game text was checked for *words* but not for *readings* (see
data_pipeline/novelgame/status_policy.py), so the first recogniser is trained
only on lines where the reading is not in doubt:

* the text was confirmed (``text_check`` is human / ctc / asr_phone) and the
  audio does not stretch or cut the phrase end (no ``phrase_end_mismatch``);
* no homograph or numeral slot (:func:`lilytts.labeler.reading.find_slots`);
* no proper noun (character names often have creative readings) and no
  unknown word that OpenJTalk broke into single kanji;
* no Latin letters, digits or symbols that hide sounds (～, ♪, ー runs, …).

Later rounds add lines labelled by the recogniser itself (self-training).

    python -m lilytts.labeler.teacher data/manifest/novelgame.split.jsonl \\
        -o data/labels/teacher_round0.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from ..text.kana import is_kana, is_kanji
from ..text.openjtalk import OJTWord, words
from .reading import find_slots, load_lexicon
from .vocab import phonetic

TRUSTED_TEXT = {"human", "ctc", "asr_phone"}
_BAD_CHARS = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ0-9０-９～〜♪♡♥☆★※＊*＃#＠@＆&／/＼|｜（）()「」『』【】<>＜＞]")
_OK_SYMBOLS = set("、。，．！？!?…‥・ 　")


def check(rec: dict, lexicon: dict) -> tuple[str | None, list[OJTWord]]:
    """(reason the line cannot be a teacher or None, OpenJTalk words)."""
    if rec.get("text_check") not in TRUSTED_TEXT:
        return "untrusted_text", []
    if rec.get("phrase_end_mismatch"):
        return "phrase_end_mismatch", []
    if rec.get("split") not in (None, "train"):
        return "not_train_split", []
    text = rec["text"]
    if _BAD_CHARS.search(text):
        return "hidden_sounds", []
    if any(not (is_kana(c) or is_kanji(c) or c in _OK_SYMBOLS) for c in text):
        return "other_chars", []
    ws = words(text)
    content = [w for w in ws if not w.is_symbol]
    if not content:
        return "no_speech", ws
    if any(w.pos_group1 == "固有名詞" for w in content):
        return "proper_noun", ws
    # Numerals and homographs first: 三 + 本 would otherwise look like an unknown word.
    if find_slots(ws, lexicon):
        return "ambiguous_reading", ws
    for a, b in zip(content, content[1:]):
        # consecutive single-kanji nouns = an unknown word split into characters
        if all(len(w.surface) == 1 and is_kanji(w.surface) and w.pos == "名詞" for w in (a, b)):
            return "unknown_word", ws
    return None, ws


def label(ws: list[OJTWord]) -> str:
    return phonetic("".join(w.pron for w in ws if not w.is_symbol))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--max-lines", type=int, default=0, help="stop after this many teachers (0 = all)")
    args = ap.parse_args(argv)

    lex = load_lexicon()
    reasons: Counter = Counter()
    kept = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            reason, ws = check(rec, lex)
            if reason:
                reasons[reason] += 1
                continue
            out.write(json.dumps({"id": rec["id"], "audio": rec["audio"], "text": rec["text"],
                                  "kana": label(ws), "duration": rec.get("duration")},
                                 ensure_ascii=False) + "\n")
            kept += 1
            if args.max_lines and kept >= args.max_lines:
                break
    print(json.dumps({"kept": kept, "rejected": dict(reasons)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
