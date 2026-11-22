"""Score a text front end (text → reading) on LilyBench-JP Yomi items.

    python -m bench.score.text_reading bench/data/yomi_numerals.jsonl --system openjtalk

Prints accuracy per subcategory. An item is correct when every target is.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from .span_judge import CORRECT, UNDETERMINED, WRONG, judge_span


def openjtalk_reading(text: str) -> str:
    from lilytts.text.openjtalk import words

    return "".join(w.pron for w in words(text) if not w.is_symbol)


SYSTEMS = {"openjtalk": openjtalk_reading}


def score(items: list[dict], system) -> dict:
    by_sub: dict[str, Counter] = defaultdict(Counter)
    errors: list[dict] = []
    for it in items:
        reading = system(it["text"])
        verdicts = [judge_span(reading, t["gold"], t["confusers"]) for t in it["targets"]]
        if all(v == CORRECT for v in verdicts):
            v = CORRECT
        elif WRONG in verdicts:
            v = WRONG
        else:
            v = UNDETERMINED
        by_sub[it.get("subcategory", "-")][v] += 1
        by_sub["(all)"][v] += 1
        if v != CORRECT:
            errors.append({"id": it["id"], "text": it["text"], "reading": reading, "verdict": v,
                           "gold": [t["gold"] for t in it["targets"]]})
    table = {}
    for sub, c in sorted(by_sub.items()):
        judged = c[CORRECT] + c[WRONG]
        table[sub] = {
            "n": sum(c.values()),
            "accuracy": round(c[CORRECT] / judged, 4) if judged else None,
            "undetermined": c[UNDETERMINED],
        }
    return {"by_subcategory": table, "errors": errors}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("items")
    ap.add_argument("--system", choices=sorted(SYSTEMS), default="openjtalk")
    ap.add_argument("--errors", type=int, default=20, help="how many errors to print")
    args = ap.parse_args(argv)
    with open(args.items, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    result = score(items, SYSTEMS[args.system])
    for sub, row in result["by_subcategory"].items():
        print(f"{sub:8s} n={row['n']:4d} acc={row['accuracy']} undetermined={row['undetermined']}")
    for e in result["errors"][: args.errors]:
        print(json.dumps(e, ensure_ascii=False))


if __name__ == "__main__":
    main()
