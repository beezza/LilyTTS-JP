# LilyTTS-JP

日本語の「読み」と「アクセント」で SOTA を狙う、個人利用のゼロショット音声合成（TTS）プロジェクト。

- 設計書（RTX 5070 Ti + RTX 3050 6GB 版）: [docs/DESIGN.md](docs/DESIGN.md)
- 評価セット LilyBench-JP の自動構築仕様: [docs/LILYBENCH.md](docs/LILYBENCH.md)
- 参考: フルスケール版の設計: [docs/DESIGN_fullscale.md](docs/DESIGN_fullscale.md)

主な特徴（計画）:
- **Yomi**: テキストを「カナ＋アクセント記号列（PK）」に変換する推定器（日本語LLM + LoRA）。ゲーム音声から復元した読みで学習する。読みとアクセントはユーザーが上書きできる
- **Lily**: Qwen3-TTS-12Hz-1.7B-Base を LoRA で微調整した音響モデル。主データはノベルゲーム音声 5〜10k時間
- **LilyBench-JP**: 難読文とアクセント付き文の評価セットを、人手なしで構築・採点する

## 実装状況

| 部品 | 場所 | 状態 |
|---|---|---|
| マニフェスト変換 | `data_pipeline/novelgame/to_manifest.py` | 実装・テスト済み |
| 作品単位の分割（学習 / bench / zeroshot / speaker_ft） | `data_pipeline/split.py` | 実装・テスト済み |
| PK の仕様（カナ＋アクセント記号） | `lilytts/text/pk.py` | 実装・テスト済み |
| OpenJTalk → 語・PK | `lilytts/text/openjtalk.py` | 実装・テスト済み（OpenJTalk のラベルと照合） |
| 数詞・助数詞・日付・時刻の読み | `lilytts/text/numerals.py` | 実装・テスト済み |
| 音声からの読みの復元（同形異音語・数詞の候補を CTC で選ぶ） | `lilytts/labeler/reading.py` | 実装・テスト済み（模擬の認識器出力で） |
| 認識器の初期教師の選別 | `lilytts/labeler/teacher.py` | 実装・テスト済み |
| カナ音素認識器の学習・推論 | `lilytts/labeler/recognizer.py` | 実装済み・**未実行**（GPU と torch が必要） |
| マニフェスト全体の読みラベル付け | `lilytts/labeler/run.py` | 実装済み・**未実行** |
| LilyBench-JP 数詞セット（規則で生成） | `bench/build/numerals.py` → `bench/data/yomi_numerals.jsonl` | 実装・テスト済み |
| テキスト前処理系の採点 | `bench/score/text_reading.py` | 実装・テスト済み |
| アクセント推定器、Yomi、Lily | — | 未着手 |

## 使い方（手元の環境で）

```bash
pip install -e ".[text,train,dev]"
```

1. 元データをマニフェストに変換する。

```bash
python -m data_pipeline.novelgame.to_manifest /path/to/meta/*.jsonl -o data/manifest/novelgame.jsonl --audio-root /mnt/data/dataset/voices --report data/manifest/novelgame.report.json
```

2. 作品単位で分割する。

```bash
python -m data_pipeline.split data/manifest/novelgame.jsonl -o data/manifest/novelgame.split.jsonl --summary data/manifest/split.json
```

3. 認識器の初期教師を選ぶ（読みが一意な台詞だけ）。

```bash
python -m lilytts.labeler.teacher data/manifest/novelgame.split.jsonl -o data/labels/teacher_round0.jsonl
```

4. カナ音素認識器を学習する（RTX 5070 Ti）。

```bash
python -m lilytts.labeler.recognizer train --teacher data/labels/teacher_round0.jsonl --audio-root /mnt/data/dataset/voices --out runs/recognizer_r0
```

5. 音声から読みを復元し、OpenJTalk との食い違いを集計する。

```bash
python -m lilytts.labeler.run data/manifest/novelgame.split.jsonl --model runs/recognizer_r0/best --audio-root /mnt/data/dataset/voices -o data/labels/readings_r0.jsonl --disagreements data/labels/openjtalk_errors_r0.tsv
```

6. 数詞セットでテキスト前処理系を採点する（例: OpenJTalk）。

```bash
python -m bench.score.text_reading bench/data/yomi_numerals.jsonl --system openjtalk
```

テスト:

```bash
python -m pytest -q
```
