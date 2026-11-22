# LilyTTS-JP

日本語の「読み」と「アクセント」で SOTA を狙う、個人利用のゼロショット音声合成（TTS）プロジェクト。

- 設計書（RTX 5070 Ti + RTX 3050 6GB 版）: [docs/DESIGN.md](docs/DESIGN.md)
- 評価セット LilyBench-JP の自動構築仕様: [docs/LILYBENCH.md](docs/LILYBENCH.md)
- 参考: フルスケール版の設計: [docs/DESIGN_fullscale.md](docs/DESIGN_fullscale.md)

主な特徴（計画）:
- **Yomi**: テキストを「カナ＋アクセント記号列（PK）」に変換する推定器（日本語LLM + LoRA）。ゲームスクリプトのルビも学習に使う。読みとアクセントはユーザーが上書きできる
- **Lily**: Qwen3-TTS-12Hz-1.7B-Base を LoRA で微調整した音響モデル。主データはノベルゲーム音声 5〜10k時間
- **LilyBench-JP**: 難読文とアクセント付き文の評価セットを、人手なしで構築・採点する
