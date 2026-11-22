import json
import random

from data_pipeline.split import main


def make_manifest(path, n_titles=40, seed=0):
    rng = random.Random(seed)
    rows = []
    for t in range(n_titles):
        title = f"g{t:03d}"
        for c in range(rng.randint(3, 8)):
            # one long-running heroine per title, the rest smaller
            n_lines = 2500 if c == 0 else rng.randint(5, 300)
            for i in range(n_lines):
                rows.append({"id": f"{title}-{c}-{i}", "title": title, "speaker": f"{title}/c{c}",
                             "text": "あ", "duration": 3.0})
    # one huge title that must never be picked for eval
    rows += [{"id": f"huge-{i}", "title": "huge", "speaker": "huge/c0", "text": "あ", "duration": 3.0}
             for i in range(20000)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def run(tmp_path, *extra):
    src = tmp_path / "m.jsonl"
    make_manifest(src)
    out, summ = tmp_path / "o.jsonl", tmp_path / "s.json"
    main([str(src), "-o", str(out), "--summary", str(summ), *extra])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    return rows, json.loads(summ.read_text())


def test_titles_never_cross_splits(tmp_path):
    rows, s = run(tmp_path, "--zeroshot-speakers", "30", "--ft-speakers", "5")
    split_of_title = {}
    for r in rows:
        group = "train" if r["split"] == "train" else "eval"
        assert split_of_title.setdefault(r["title"], group) == group
    assert "huge" not in s["eval_titles"]
    assert len(s["eval_titles"]) >= 10
    assert s["eval_title_hours"] >= 0.03 * s["total_hours"]


def test_speaker_sets(tmp_path):
    rows, s = run(tmp_path, "--zeroshot-speakers", "30", "--ft-speakers", "5")
    assert len(s["speaker_ft"]) == 5 and len(s["zeroshot"]) == 30
    assert not set(s["speaker_ft"]) & set(s["zeroshot"])
    by_speaker = {}
    for r in rows:
        by_speaker.setdefault(r["speaker"], set()).add(r["split"])
    for sp in s["speaker_ft"]:
        assert by_speaker[sp] == {"speaker_ft"}
    for sp in s["zeroshot"]:
        assert by_speaker[sp] == {"zeroshot"}


def test_deterministic(tmp_path):
    _, a = run(tmp_path, "--seed", "7")
    _, b = run(tmp_path, "--seed", "7")
    _, c = run(tmp_path, "--seed", "8")
    assert a["eval_titles"] == b["eval_titles"] != c["eval_titles"]
