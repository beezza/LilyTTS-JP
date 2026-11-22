# LilyTTS-JP

日本語でSOTAを狙うゼロショット音声合成（TTS）プロジェクト。

- 設計書（RTX 5070 Ti + RTX 3050 版）: [docs/DESIGN.md](docs/DESIGN.md)
- 参考: フルスケール版の設計: [docs/DESIGN_fullscale.md](docs/DESIGN_fullscale.md)

主な特徴（計画）:
- **Yomi**: テキストを「カナ＋アクセント記号列（PK）」に変換する推定器（日本語LLM + LoRA、テキストだけで学習）。読みとアクセントはユーザーが上書きできる
- **Lily**: Qwen3-TTS-12Hz-1.7B-Base を LoRA で微調整し、PK に忠実に読ませる音響モデル
- 音声から取った読み・アクセントラベルによる学習
- 日本語特化の評価ベンチ **LilyBench-JP**（Yomi-Hard / Accent / Zero-shot / Expressive / Long-form）
