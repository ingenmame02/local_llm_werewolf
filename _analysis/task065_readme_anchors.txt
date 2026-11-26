102: | `--num-ctx INT` | 未設定 | コンテキスト長。省略時は API リクエストに含めず、サーバー／モデル既定値が使われます |
108: | `--thinking {auto,on,off}` | `auto` | 思考生成。`auto` は行動別既定（議論・人狼相談 on／襲撃・占い・投票・CO補正 off） |
121: | `--language {English,Japanese}` | 設定／日本語 | 応答言語 |
219: - 日をまたぐ論理状態の正規化（既定 `normalized` / `--continuity off` で比較・TASK-062）: `continuity_state.py` 
220: - 役職能力の境界整理（既定・TASK-064）: `role_action_provenance.py` が役職情報を4つへ分けます。`engine_action`（エンジンがそ
221: - 引用の時点分類と未解決参照の原因分解（観測専用・本文は書き換えない）: `TurnSnapshot.citation_status()` が引用相手の状態を「同一ターンの先行発
222: - 発言時点の時系列明示（既定 on / `--day-timeline off` で比較）: `[DAY_TIMELINE]` に「現在時点・自分の発言順・このターンで既知の先行
TOTAL 359
