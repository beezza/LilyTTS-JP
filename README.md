# LilyTTS-JP

日本語でSOTAを狙うゼロショット音声合成（TTS）プロジェクト。

- 設計書: [docs/DESIGN.md](docs/DESIGN.md)

主な特徴（計画）:
- **Reading-CoT**: LMが「カナ＋アクセント記号列」を先に生成してから音声を生成する。読みとアクセントはユーザーが上書きできる
- 音声から取った読み・アクセントラベルによる学習
- AR LM（日本語LLM初期化）＋ チャンク因果 Flow Matching による 48kHz ストリーミング合成
- 日本語特化の評価ベンチ **LilyBench-JP**（Yomi-Hard / Accent / Zero-shot / Expressive / Long-form）
