# TASK-007 実戦評価計画

## 目的

完成したCLIと8モデルを実際に接続し、ゲームが安定して完走するか、モデルごとの応答品質・速度・戦略傾向に差があるかを確認する。

## 固定条件

- モデル数：8
- 役職構成：人狼2、狂人1、占い師1、村人4
- 温度：`0.7`
- コンテキスト長：`4096`
- タイムアウト：`180`秒
- 実行モード：`ollama`
- 乱数シード：ゲームごとに記録する
- まず1ゲームで疎通確認し、その後最低3ゲームを実行する

## 実行手順

Ollamaを起動する。

```powershell
ollama serve
```

別のPowerShellで、モデルとAPIの状態を確認する。

```powershell
ollama list
Invoke-RestMethod -Uri http://localhost:11434/api/tags
```

まず1ゲームを実行する。

```powershell
python src/werewolf/cli.py `
  --mode ollama `
  --models-file config/players.json `
  --seed 1001 `
  --timeout 180 `
  --temperature 0.7 `
  --num-ctx 4096 `
  --log-dir evaluation-results/games
```

疎通確認後、異なるシードで最低3ゲームを実行する。

```powershell
python src/werewolf/cli.py --mode ollama --models-file config/players.json --seed 1001 --timeout 180 --temperature 0.7 --num-ctx 4096 --log-dir evaluation-results/games
python src/werewolf/cli.py --mode ollama --models-file config/players.json --seed 1002 --timeout 180 --temperature 0.7 --num-ctx 4096 --log-dir evaluation-results/games
python src/werewolf/cli.py --mode ollama --models-file config/players.json --seed 1003 --timeout 180 --temperature 0.7 --num-ctx 4096 --log-dir evaluation-results/games
```

## 記録する指標

- ゲーム完走・中断・エラー
- 勝者、ゲーム日数、生存者
- モデル別の役職別応答時間
- 形式遵守率、パース成功率
- 通信リトライ、意味的再プロンプト、フォールバック回数
- 自己投票・死亡者への行動などの不正行動
- 人狼相談、占い、投票の行動傾向

## 分析ルール

- 1ゲームだけの勝敗でモデルの強さを断定しない。
- 勝敗と応答品質を分けて評価する。
- モデル名、Ollamaタグ、設定値、シードを必ずレポートへ含める。
- 秘密ログと公開ログを混同しない。
- エラーがあった場合もログを削除せず、再現手順を残す。

## 成果物

- `evaluation-results/games/game_*.jsonl`
- `evaluation-results/analysis/summary.md`
- 必要に応じて集計用JSONまたはCSV
- 次に実施する改善タスクの一覧
