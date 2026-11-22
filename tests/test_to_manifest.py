import json

from data_pipeline.novelgame.to_manifest import main

SAMPLE = {
    "catalog_id": 1070127,
    "uid": "2a046e12070043b4f48b5fcff52d0a41a32ede5f030188a62527a3fe2227aef8",
    "audio_path": "/mnt/data/dataset/voices/GalGame/sprite_Ao no Kanata no Four Rhythm/明日香/asuka_00956.ogg",
    "text": "安全教室ですか",
    "game_id": "sprite_Ao no Kanata no Four Rhythm",
    "character_id": "明日香",
    "performer_id": None,
    "split": "validation",
    "cleaning_status": "acoustic_and_asr_phone_match",
    "frames": 54,
    "duration_sec": 2.16,
    "duration_is_estimated": True,
}


def run(tmp_path, records, *extra):
    src = tmp_path / "in.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    out = tmp_path / "out.jsonl"
    rep = tmp_path / "rep.json"
    main([str(src), "-o", str(out), "--report", str(rep), *extra])
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    return rows, json.loads(rep.read_text(encoding="utf-8"))


def test_sample_record(tmp_path):
    rows, rep = run(tmp_path, [SAMPLE], "--audio-root", "/mnt/data/dataset/voices")
    assert rows == [
        {
            "id": SAMPLE["uid"],
            "audio": "GalGame/sprite_Ao no Kanata no Four Rhythm/明日香/asuka_00956.ogg",
            "text": "安全教室ですか",
            "ruby": [],
            "speaker": "sprite_Ao no Kanata no Four Rhythm/明日香",
            "title": "sprite_Ao no Kanata no Four Rhythm",
            "duration": 2.16,
            "duration_estimated": True,
            "src_status": "acoustic_and_asr_phone_match",
        }
    ]
    assert rep["records_out"] == 1 and rep["source_keys"]["frames"] == 1
    assert rep["ruby_source"] == {"none": 1}


def test_ruby_field_and_inline(tmp_path):
    a = {**SAMPLE, "uid": "a", "ruby": [{"base": "安全", "reading": "あんぜん"}]}
    b = {**SAMPLE, "uid": "b", "text": "｜明日香《あすか》、行くよ"}
    rows, rep = run(tmp_path, [a, b])
    assert rows[0]["ruby"] == [{"start": 0, "end": 2, "reading": "アンゼン"}]
    assert rows[1]["text"] == "明日香、行くよ"
    assert rows[1]["ruby"] == [{"start": 0, "end": 3, "reading": "アスカ"}]
    assert rep["records_with_ruby"] == 2


def test_ruby_text_field_and_status_filter(tmp_path):
    a = {**SAMPLE, "uid": "a", "text_ruby": "｜安全《あんぜん》教室ですか"}
    b = {**SAMPLE, "uid": "b", "cleaning_status": "rejected"}
    rows, rep = run(tmp_path, [a, b], "--ruby-text-field", "text_ruby",
                    "--keep-status", "acoustic_and_asr_phone_match")
    assert [r["id"] for r in rows] == ["a"]
    assert rows[0]["ruby"] == [{"start": 0, "end": 2, "reading": "アンゼン"}]
    assert rep["rejected"] == {"status_filtered": 1}
    assert rep["warnings"] == {}
