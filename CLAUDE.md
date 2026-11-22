# LilyTTS-JP — Claude Code 向けメモ

日本語の「読み」と「アクセント」で SOTA を狙う個人用 TTS。重みは公開しない。
設計は `docs/DESIGN.md`（現行）、評価の自動構築は `docs/LILYBENCH.md`。`docs/DESIGN_fullscale.md` は参考で、計画ではない。

## 前提
- GPU: RTX 5070 Ti 16GB（学習・重い推論）、RTX 3050 **6GB**（前処理・評価・Yomi 推論）。2枚で一緒に学習はしない。5070 Ti は sm_120 なので PyTorch は CUDA 12.8 以降のビルドを使う。
- 主データ: ノベルゲーム音声 5〜10k時間。元のメタデータは他の TTS と共用していたもので、不要なフィールドが混ざっている（`to_manifest.py` が捨てる）。
  - **読み（ルビ）の情報は無い**。元データの音素一致は「OpenJTalk(元台詞)」と「Whisper の書き起こしを正規化した音素」の比較で、`ctc_confirms_original` の CTC（reazon-research/japanese-wav2vec2-base-rs35kh）も文字を出す ASR。どれも**語の正しさしか保証しない。読みは必ず音声から取る**（`lilytts/labeler/`）。
  - 成人向けの台詞はフィルタ済み。
- 話者: ゼロショットの本体 ＋ 話者ごとの LoRA。話者IDの埋め込みは使わない。

## コマンド
- 依存: `pip install -e ".[text,train,dev]"`（`text` は `pyopenjtalk-plus[onnxruntime]`）
- テスト: `python -m pytest -q`（pyopenjtalk が無いと OpenJTalk 系のテストは skip）
- パイプラインの順番は README の「使い方」1〜6。

## 決まりごと
- ドキュメントとユーザー向けの文章は日本語。コードとコメントは英語。
- コミット前にテストの**終了コード**を確認する。`pytest | tail` のようにパイプすると失敗が後続に伝わらない（実際に一度やらかした）。
- 正解ラベルは評価対象のモデル（Yomi / Lily）と独立した情報源から取る。学習の報酬に使うモデルと評価の採点に使うモデルは別にする（認識器: 報酬・ラベル用は Reazon wav2vec2、採点用は rinna HuBERT）。
- 学習と評価は**作品（title）単位**で分ける（`data_pipeline/split.py`）。元データの `split` は使わない。

## ハマりどころ
- **pyopenjtalk-plus は onnxruntime が無いと「何」の読み予測が無効**になり、何本→ナニボン などになる。onnxruntime 有無で OpenJTalk の既定の読みが変わるので、テストで既定の読みを固定しない。
- OpenJTalk のラベルは平板型を「最終モーラに核」と書く。`words_to_sentence` は 0 と書く。比較するときは同一視する（`tests/test_openjtalk.py` の `_norm`）。
- OpenJTalk は数字を漢数字に書き換える（3→三）。その語の文字位置は `None` になる。
- `reading_key()` は長音の表記ゆれを吸収するが、語の境界をまたいでモーラを結合してしまう（デ+イッカイ→デーッカイ）。比較では、まずそのままのカナで照合し、`reading_key` は代替手段として使う（`bench/score/span_judge.py`）。
- PK の「キョ」は1モーラ。正しくは `^キョ]ーワ#…`。
- 未知の名前（璃々愛）は OpenJTalk が1文字ずつ推測して読む。固有名詞タグや単漢字名詞の連続で検出して、認識器の教師から除外している。

## 状態（2026-11-22 時点）
実装・テスト済み: マニフェスト変換、作品単位の分割、PK、OpenJTalk→PK、数詞の規則、音声からの読み復元（模擬の認識器出力でテスト）、教師の選別、数詞の評価セット（238文。OpenJTalk の正解率 99.6%）。
**実装済みで未実行**: `lilytts/labeler/recognizer.py`（学習）と `lilytts/labeler/run.py`（全データへの読み付け）。クラウド環境に torch と GPU が無かったため。最初に実際のデータで動かして直すこと。

## 次にやること（優先順）
1. README の 1〜5 を実データで実行する。`to_manifest` のレポート（`source_keys` / `status`）、`split.json`、教師選別の除外理由の内訳、認識器の `dev_kana_cer`、`openjtalk_errors_r0.tsv` の上位を確認する。
2. 認識器を自己学習で回す（`docs/DESIGN.md` §5.4: 高信頼の復元結果を教師に追加して2〜3回）。
3. 採点用の別系統の認識器（rinna/japanese-hubert-base + カナ CTC）を作る。
4. LilyBench-JP の残り（`docs/LILYBENCH.md`）: 青空文庫のルビ（R1）、Wikipedia の読み・駅名・郵便番号データ（R3）、OpenJTalk の食い違いから作る難例（R4）、JSUT のアクセントラベル（A1）、難しさによる選別とランダムセット。
5. アクセント推定器（強制アライメント → RMVPE で F0 → 分類器。教師は JSUT の人手ラベル）。
6. Yomi（テキスト→PK、日本語 LLM + LoRA）、Lily（Qwen3-TTS-12Hz-1.7B-Base + LoRA）、DPO、話者 LoRA。
