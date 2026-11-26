# Public conversion tool — 公開用ログ変換ツール

ゲームログ `logs/game_<ID>.jsonl` から、**プレイヤー名をモデル名に置換した
公開用Markdown** を生成するツールです。

## 使い方

```powershell
# ゲームID指定 (既定で ../logs を検索 / 出力は本フォルダの public_<ID>.md)
python "Public conversion tool/log_to_public_md.py" 425755ce9a83

# jsonlパス直接指定・出力先変更
python "Public conversion tool/log_to_public_md.py" logs\game_xxxx.jsonl -o out.md

# ログディレクトリの変更
python "Public conversion tool/log_to_public_md.py" <ID> --logs-dir D:\other\logs
```

外部ライブラリ不要 (Python標準ライブラリのみのため `python` があれば動きます)。

## 公開Markdownに含まれる情報 (ホワイトリスト方式)

- 参加モデル一覧 (モデル名・人物設定・最期) と結果
- 各Day: 初夜ルール注記 / 朝の死亡者 / 議論発言 (モデル名に置換)
- 公開CO記録 (役職CO・占い結果申告。重複は統合)
- 投票結果 (投票者→投票先・処刑者)
- 最終結果 (勝敗と生存者)

上記以外のイベント (LLMリクエスト設定・トークン統計・内部解析) は全て無視されます。
`private: true` のイベントも一切出力しません。**真の役職・未公開の夜行動は
ログ上にも存在しないため記載できません** (公開されたCOのみ)。

## セーフティ機能

- 発言文中の `Player N` / `PlayerN` / `プレイヤーN` (全角数字含む) をすべてモデル名に置換
- 変換後に未置換のプレイヤー表記が残った場合は `[WARN]` を stderr に出力

## 生成物

- `public_<ゲームID>.md` — 公開用Markdown
- 既存: `public_425755ce9a83.md` (サンプル生成済み)
