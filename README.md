# ローカルLLM人狼（Local LLM Werewolf）

Ollama でローカルLLM 8体 を人狼ゲームのプレイヤーとして実行する、完全ターン制の CLI アプリケーションです。
ゲームロジック（エンジン）と LLM 呼び出しを分離しており、Ollama がなくても **モックモード** で全フローを動作確認できます。

## 主な特徴

- 役職配布、昼議論、投票、夜行動（人狼相談・襲撃・占い）、勝敗判定を決定的なゲームエンジンが管理
- 各 LLM 呼び出しを JSONL に記録（観戦用の公開ログと管理者用ログを分離）
- 議論応答を CO（役職名乗り）と本文に構造化し、公開主張・占い結果を台帳で冪等に管理
- 秘匿情報（真役職・人狼相談・占い結果）を他プレイヤーへ渡さない情報分離と漏えい検出
- 行動別（議論・占い・投票など）の生成上限 `num_predict` と思考生成 `think` の制御
- コンテキスト戦略・コンテキスト長・サンプリングのモデル別設定と比較ベンチマーク

## ゲームルール

| 項目 | 内容 |
|---|---|
| 役職 | 人狼 2 / 狂人 1 / 占い師 1 / 村人 4（計 8 名） |
| 村人陣営の勝利 | 生存している人狼が 0 名 |
| 人狼陣営の勝利 | 生存人狼数 ≥ 生存（占い師＋村人）数（狂人は人数計算に含めない） |
| 初夜（Day 1） | 占い師の占いのみ（襲撃・人狼相談なし、死亡者発表なし） |
| Day 2 以降の夜 | 人狼相談（人狼 2 名時、最大 2 ターン）→ 襲撃 → 占い |
| 昼 | 議論 3 ターン → 投票・処刑 → 勝敗判定 |
| 占い結果 | 狂人を占うと「人間」が出る。結果は占い師本人のみが参照できる |

詳細な仕様は `SPECIFICATION.md` を参照してください。

## インストール

```powershell
pip install -e .              # 本体
pip install -e ".[test]"      # 開発（pytest）
pip install -e ".[benchmark]" # コンテキストベンチマーク（psutil）
```

Python 3.10 以上が必要です。モックモードは追加の依存なく動作します。実プレイには Ollama（既定 `http://localhost:11434`）と 8 体のモデルが必要です。

## 使い方

### モックモード（Ollama 不要・開発・テスト用）

```powershell
python src/werewolf/cli.py --mode mock --seed 42
```

### Ollama モード（1 ゲーム）

```powershell
python src/werewolf/cli.py --mode ollama `
  --models-file config/players.json `
  --seed 20261106 `
  --timeout 180 `
  --temperature 0.7 `
  --language Japanese `
  --log-dir logs
```

秘密情報（人狼相談・占い結果）を端末に表示し、管理者ログも残す場合:

```powershell
python src/werewolf/cli.py --mode ollama --models-file config/players.json --seed 20261106 --timeout 180 --private-log --show-private
```

### 推奨の Ollama サーバー設定

モデルの並列数・コンテキスト長は **Ollama サーバープロセスの環境変数** で設定します（クライアント CLI からは設定できません）。ゲームは直列進行なので `OLLAMA_NUM_PARALLEL=1` を推奨します。

```powershell
$env:CUDA_VISIBLE_DEVICES = "0,1"   # GPU を複数使う場合
$env:OLLAMA_SCHED_SPREAD = "1"
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_CONTEXT_LENGTH = "8192"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_KEEP_ALIVE = "5m"

ollama serve
```

既定の `config/players.json` は 8 プレイヤーそれぞれに別モデルを割り当てます（モデル名は環境に合わせて変更してください）。

| プレイヤー | 既定モデル |
|---|---|
| Player 1 | `werewolf-calm3-22b-8k` |
| Player 2 | `werewolf-muse-glimmer-8k` |
| Player 3 | `werewolf-mistral-small-8k` |
| Player 4 | `werewolf-glm-4.7-flash-8k` |
| Player 5 | `werewolf-qwen3.8-8k` |
| Player 6 | `werewolf-deepseek-r1-8k` |
| Player 7 | `werewolf-gpt-oss-20b-8k` |
| Player 8 | `werewolf-gemma4-8k` |

## CLI オプション

| オプション | 既定値 | 説明 |
|---|---|---|
| `--mode {mock,ollama}` | `mock` | 実行モード |
| `--models-file PATH` | `config/players.json` | プレイヤー設定ファイル |
| `--seed INT` | `42` | 役職配布・同票処理を再現可能にする乱数シード |
| `--timeout SEC` | `120` | Ollama リクエストのタイムアウト |
| `--temperature FLOAT` | `0.7` | 生成温度（全体設定） |
| `--num-ctx INT` | 未設定 | コンテキスト長。省略時は API リクエストに含めず、サーバー／モデル既定値が使われます |
| `--override-player-num-ctx` | off | `--num-ctx` をプレイヤー別 `num_ctx` より優先させる（`--num-ctx` と併用） |
| `--no-dynamic-context` | off | 動的コンテキストを停止（固定コンテキストの比較用） |
| `--num-predict INT` | 未設定 | 生成上限（全体設定）。省略時は行動別既定値 |
| `--num-predict-by-action ACTION=N` | — | 行動別生成上限の上書き（複数指定可） |
| `--thinking {auto,on,off}` | `auto` | 思考生成。`auto` は行動別既定（議論・人狼相談 on／襲撃・占い・投票・CO補正 off） |
| `--thinking-by-action ACTION=true\|false` | — | 行動別思考の上書き（複数指定可） |
| `--discussion-claim-mode {legacy,candidates,all}` | `legacy` | 議論 CO の生成方式（旧来併走／候補のみ二段階／全員二段階） |
| `--context-strategy {...}` | `hybrid_retrieval` | 議論に渡す履歴の構成（後述） |
| `--question-hint-mode {strict,soft,none}` | `soft` | 直前の質問への回答指示の強さ。`soft` は観測のみ／`strict` は旧来（1文目ルール＋1回再生成）／`none` は質問欄も指示も出しません |
| `--language {English,Japanese}` | 設定／日本語 | 応答言語 |
| `--personality {…profiles…,free}` | 設定そのまま | 全プレイヤーの個性プロファイルを上書き。`free` は自由発言モード（話し方の preset なし） |
| `--log-dir DIR` | `logs` | JSONL ログの出力先 |
| `--base-url URL` | `http://localhost:11434` | Ollama API のベース URL |
| `--show-private` | off | 秘密情報を端末に表示し、管理者ログも生成 |
| `--private-log` | off | 管理者ログのみ生成（端末には秘密情報を表示しません） |

行動キー: `discussion` / `wolf_discussion`（別名 `wolf_consult`）/ `attack` / `seer` / `vote`（別名 `target`）/ `co_correction`

## プレイヤー設定（`config/players.json`）

各プレイヤーにモデルと生成条件を指定します。**プレイヤー設定は常に CLI の全体設定より優先されます。**

```json
{
  "players": [
    {
      "name": "Player 1",
      "model": "werewolf-calm3-22b-8k",
      "num_ctx": 12288,
      "num_predict": 1536,
      "num_predict_by_action": { "discussion": 2560, "vote": 768 },
      "thinking_by_action": { "discussion": false },
      "personality_profile": "analytical",
      "response_language": "日本語"
    }
  ]
}
```

| フィールド | 説明 |
|---|---|
| `name` / `model` | 表示名 / Ollama モデル名 |
| `num_ctx` | API リクエストに送るコンテキスト長（`--num-ctx` より優先） |
| `max_num_ctx` / `recommended_num_ctx` | 動的コンテキストの拡張上限 / 推奨値 |
| `timeout` | このプレイヤーだけのリクエスト時間上限（秒） |
| `num_predict` / `num_predict_by_action` | 生成上限（全体 / 行動別） |
| `thinking` / `thinking_by_action` | 思考生成（全体 / 行動別） |
| `personality_profile` | 議論の話し方の傾向（`balanced` `analytical` `questioning` `assertive` `cautious` `consensus` `intuitive`）。`null` / `"free"` / `"none"` は自由発言モード（preset 適用なし） |
| `context_strategy` | このプレイヤーだけのコンテキスト戦略 |
| `claim_coaching` | CO の先頭形式案内を強調（既定 `false`。議論の自由度・CO 保留・パスは制限しません） |
| `temperature` / `top_p` / `top_k` / `repeat_penalty` | プレイヤー別サンプリング（許容範囲 0–2 / 0–1 / 0–400 / 0–2） |
| `response_language` | 応答言語（既定 日本語） |

## 議論応答形式（CO / 本文）

昼の議論は 1 回の生成で公開主張（CO）と本文を分離します。日本語は `【CO】` / `【本文】`、英語は `[CO]` / `[BODY]` を使います。CO は公開主張の構造化に使い、会話履歴には本文だけを追加します（旧来形式の文章 CO も互換のため受け付けます）。

```text
【CO】なし
【本文】Player 2 への疑いは弱めます。Player 3 の発言も確認したいです。
```

- CO欄は `なし` と `保留` を区別します。CO 行だけを公開主張として解析し、本文だけを会話履歴へ登録します。
- 占い結果を公開しない表記（`【結果】None`・空・`なし`・`未公開`）は「結果なし」として正規化し、形式失敗や再試行の原因にしません。誤って結果行が付いた場合も公開しません。
- 公開 CO と占い結果は台帳で冪等に管理され、同一 CO の再送は本文から除去します。CO 判定は毎ターン行わず、初回・未公開の新しい占い結果・明示的な訂正要求時に限定します。
- 修復できる違反（死亡者への行動提案など）は再生成せず `ledger_repair` として記録します。候補外の襲撃・占い・投票は短い再回答後に合法候補へ補正します。
- 真占い師が未公開の判定を口にした応答は CO 行を決定的に補完して受理し、再試行とフォールバックを避けます（修復文は応答言語で書き、本文を保持します）。
- 昼議論プロンプトには確定ルール・ゲーム台帳とは別に「人狼セオリーカード」を含めます（対抗 CO のない初日単独 CO の評価、狂人を占うと人間が出る＝人間判定は村人の確定ではない、等）。
- 個性（`personality_profile`）は昼の議論本文だけに影響し、CO の判定や対象選択には影響しません。議論の質・事実整合性・質問回答・形式指示より優先度が低く、追加の説明や質問を強制しません。
- 自由発言モード（`personality_profile` を `null` / `"free"` / `"none"` にするか `--personality free`）は個性 preset を一切適用しません。口調・構成・文量・注目点をモデル自身の判断に任せ、`[Expression Freedom]` ブロックで「preset がない」ことを明示します。昼議論の `[BODY]` も文量・構成の固定指示を外します。役職の目的・勝敗条件・行動制約・確定事実・出力形式（CO/結果行）・秘匿情報の禁は preset 適用時と同一です。 preset を適用しない設定ファイルは `config/players_free.json`（`--models-file config/players_free.json`）です。
- `--discussion-claim-mode candidates`（占い師・CO 公開済みプレイヤーのみ）/ `all`（全員）は二段階 CO の比較用です。CO 専用リクエストが増えるため呼び出し回数・所要時間が増える可能性があります。同一シードで各モードを実行し、ログの `discussion_claim_mode` と `ledger_metrics`（CO 呼び出し、実送信・再試行、重複 CO、台帳修復、本文保持、質問未回答、処理時間）を比較してください。
## 推理メモ（【推理メモ】 / [REASONING]）

昼議論の応答に任意の私用推理メモ欄を追加します。モデルは毎ターン自分の推理を構造化して記録し、次のターンで自身の分析を発展させることができます。

```text
【CO】なし
【推理メモ】Player 3が怪しい。Player 1と同じ発言パターン。Player 4がPlayer 1をかばっている。次はPlayer 3に投票意向を問う。
【本文】Player 3の発言パターンが気になります。投票意向を教えてください。
```

- `【推理メモ】`（日本語） / `[REASONING]`（英語）は**任意**です。書かない場合も従来通り動作します。
- 推理メモは本人だけの情報です。公開本文（`【本文】` / `[BODY]`）には含めず、公開会話履歴にも登録しません。
- 内容の推奨: 誰を疑うかと根拠、他者の気になる発言、誰が誰をかばっているか、今後の方針（質問・投票先）。
- 次のターンのプロンプトに `[YOUR_PREVIOUS_REASONING_MEMO]` / `【あなたの以前の推理メモ】` として直近5ターン分の自分のメモが返され、一貫性の維持と分析の発展に使えます。
- システム由来の `CONTINUITY_MEMO`（公開事実・投票・CO台帳由来）とモデルが自分で書いた推理メモは別々に保持されます。
- ベンチマークの `--reasoning-memory on/off/both` はシステム由来メモの有無を比較します。モデルが書く推理メモは `reasoning_memory_enabled` が `true` の場合に有効になります。



## コンテキスト戦略

| 戦略 | 内容 |
|---|---|
| `full_transcript` | 公開会話の全文 |
| `structured_ledger` | 台帳（確定事実・CO・占い・投票・死亡履歴）＋現在の質問 |
| `hybrid_retrieval`（既定） | 直近発言＋現在の質問に関連する過去発言・投票理由・CO・占い結果 |
| `round_transcript_memory` | 現在ラウンドの全文＋プレイヤー別「継続推理メモ」＋確定台帳 |

- 動的コンテキスト（既定 on）: 実測したプロンプトサイズ＋行動別生成上限に合わせて `num_ctx` を調整し、必要時は `max_num_ctx` まで拡張します。`--no-dynamic-context` で固定できます。
- `round_transcript_memory` は過去ゲーム全体の全文を毎回投入する方式ではないため、モデル別の実効コンテキスト上限を超えにくく、秘匿情報をプレイヤー間で混ぜません。再試行でも現在ラウンド全文と継続メモを維持します。
- 推理メモはゲームエンジンが公開台帳・投票・発言 ID から決定的に更新します（LLM は差分の整理だけを行い、確定事実・役職・秘匿情報を生成させません）。狼相談や占い師本人の結果は本人専用領域に分離し、他プレイヤーへ共有しません。メモ更新に失敗した場合は直前の有効メモと確定台帳でゲームを継続します。
- メタ漏洩・異常発言の警告も、内容を漏らさない安全な要約として次のプレイヤーへ渡します。
- 再試行では失敗理由に応じた修正指示と、直近の公開発言・公開 CO など必要な情報だけを再送します。占い師本人の非公開結果は、その占い師の再試行にのみ含めます。
- 全履歴系のプロンプト監査では公開発言の SHA-256、通し発言番号、選択理由、除外範囲を記録し、会話本文や非公開の狼相談・占い結果は診断イベントに含めません。

## 生成上限（num_predict）と思考生成（think）

- 優先順位は **player 行動別 → CLI 行動別 → 全体設定 → player 全体 → CLI 全体 → 行動別既定値** です。行動別既定値は 議論 1536 / 人狼相談 768 / 襲撃・占い・投票 256 / CO補正 256。
- 最終回答が 1 行で済む行動（襲撃・占い・投票・CO補正）は上限キャップ（襲撃・占い・投票 768、CO補正 384）へ降ろし、思考系モデルの無駄な生成による長時間応答を抑制します。議論・人狼相談は根拠を説明できる文量を確保するため全体設定をそのまま使います（昼議論の文量目安は 5〜8 文）。
- 生成起因の失敗（空応答・思考のみ・`done_reason=length`）では `num_predict` を倍にして再送します。議論の 3 回目再要求は生成起因の失敗だけに限定し、意味検証の失敗は従来の 2 回までです。
- `num_predict` は思考トークンも消費します。思考の既定は 議論・人狼相談 = on / 襲撃・占い・投票・CO補正 = off で、リクエスト毎の `think` として Ollama に送ります。再要求（CO補正・軽い再要求・対象選択の再要求）は思考 off に切り替え、プロンプトにも「思考生成（think）はオフ」と実際に送信する生成上限を明記して、指示とリクエストの値を一致させます。
- 未指定の思考 on 行動では思考残余を確保するため下限へ引き上げます（議論 3072、人狼相談 1536）。player・CLI の明示設定は優先順位上より高いため、下限で上書きしません。
- 実機で思考が上限を使い切って `done_reason=length` ＋空 content になったモデルは `thinking_budget_exhausted` として記録し、以後そのモデルの思考行動を off に落とします。
- `think` を受け付けないモデルでは Ollama が HTTP 400 を返します。同じ要求を再送せず、ただちに `think=false` で 1 回だけ再送し、以後の全行動で思考 off にします。
- 各行動の要求は `num_predict_source`（`player_by_action` / `cli_by_action` / `global_capped` / `player_global` / `cli_global` / `default_by_action` / `thinking_allowance`）と `thinking_source` を記録し、CLI は「行動」「優先元」「think=on|off」を表示します。

## ログ

| ファイル | 内容 |
|---|---|
| `logs/game_<id>.jsonl` | 観戦用の公開ログ（公開発言、CO、投票、死亡、襲撃、占い履歴の公開分、勝敗と判定に用いた生存人数） |
| `logs/game_<id>.private.jsonl` | 管理者用ログ（人狼相談、占い師本人の占い結果、候補集合、プロンプト／生応答の診断メタ、失敗理由、台帳スナップショット、`context_strategy`、`integrity_alerts`、推理メモ、入力トークン量） |

- `--show-private`: 管理者ログを生成し、端末にも秘密情報を表示
- `--private-log`: 管理者ログだけを生成（端末には表示しません）
- 公開ログには秘密情報・診断メタデータ・失敗詳細を含めません。管理者ログを共有する場合は取り扱いに注意してください。

## 評価・ベンチマーク

### モデル評価（固定シナリオ）

8 モデル一括ではなく、まず 1〜2 モデルだけ先行実行できます。

```powershell
python scripts/evaluate_models.py --models muse-glimmer --output evaluation-results/pilot
python scripts/evaluate_models.py --models glm-4.7 --models calm3 --output evaluation-results/pilot
python scripts/evaluate_models.py   # 絞り込みなし = 全モデル
```

- `--models` はモデル名・プレイヤー名の部分一致（大文字小文字無視）で評価対象を絞り込みます。プロンプトは常に 8 人分の名前を使うため、絞り込みしてもゲーム状況は同じです。
- 主なオプション: `--models-file` / `--output` / `--personality` / `--rounds` / `--timeout` / `--temperature` / `--context-length` / `--max-context-length` / `--no-dynamic-context` / `--num-predict` / `--num-predict-by-action` / `--thinking` / `--thinking-by-action` / `--base-url`
- `--personality free` は個性 preset を廃した自由発言モードです。同じモデル・同じシード・同じラウンド数で preset 適用時（`--personality analytical` 等）と別出力先へ比較し、`summary.json` の `personality_override` / `personality_profiles` で条件を区別します。
- 文量の変更効果は出力文字数ではなく、`done_reason` 別（空応答・思考のみ・`done_reason=length`）、生成トークン数（`eval_count`）、生成上限の実解決と出所（`action_budget_key` / `num_predict_source` / `num_predict_by_attempt`）、解析成功率、具体性（`specific_rate`）、実送信回数（`request_attempts`）、処理時間、フォールバック率で比較します。
- `summary.json` に `models_filter`・`selected_models`・モデル別の `by_action`・思考 on/off 送信数・`thinking_affordable_num_predict` を記録します。`generation_truncated` が多いモデルは上限増の候補、タイムアウトが多いモデルは上限据え置き候補として分けて判定します。
- Seer CO 比較では CO 成立率・結果公開率・ゲーム内 CO 公開率・フォールバック率を出力します。

### コンテキスト比較ベンチマーク

Ollama サーバーはゲームと同様に直列実行します。`OLLAMA_NUM_PARALLEL=1` を**サーバープロセスの環境**へ設定して再起動してから実行してください。

```powershell
pip install -e ".[benchmark]"
python scripts/benchmark_context.py `
  --models werewolf-calm3-22b-8k `
  --context-lengths 8192 12288 16384 `
  --context-strategies full_transcript structured_ledger hybrid_retrieval `
  --reasoning-memory both `
  --rounds 10 `
  --actions discussion `
  --output evaluation-results/context-benchmark.jsonl
```

- 同一の固定シナリオ・生成設定でコンテキスト長・戦略・推理メモ on/off を組み合わせ、質問回答率、新しい根拠カテゴリの有無、投票意向の具体性、再試行・フォールバック、入出力トークン、処理時間、CPU/RAM/GPU/VRAM を JSONL に追記します。各試行には要求・実効 `num_ctx`、トークン数、終了理由、応答時間、メモリサンプルが記録されます。
- 条件を一度だけ固定して実行する場合は `--reasoning-memory off` または `--reasoning-memory on` を指定します（既定は `both` で on/off を別条件として比較します）。
- GPU 使用率と VRAM は `nvidia-smi` 対応 GPU がある場合のみ記録されます。`ollama ps` の PROCESSOR は CPU/GPU へのロード割当であり使用率そのものではありません。
- 大きいコンテキスト（例 32k）の採用基準は事先決めします: 最低 10 件の議論で質問回答率・新根拠カテゴリ率・投票意向具体率のいずれかが 10 ポイント以上改善、解析成功率が悪化せず、平均応答時間の増加が 25% 以内。`thinking_present` は記録しますが、Ollama 応答から思考トークンだけを分離して正確に数える値は取得できません。
- 単一ゲームで条件比較するときは `--seed` とモデル設定を固定し、`--context-strategy` と `--num-ctx` だけを切り替えます。`config/players.json` のプレイヤー別 `num_ctx` を指定値で上書きするには `--override-player-num-ctx` を併用してください。

### Ollama コンテキスト診断

Modelfile の `PARAMETER num_ctx`、Python プロセスの `OLLAMA_CONTEXT_LENGTH`、API リクエスト値、ロード済みモデルの `/api/ps` の CONTEXT/PROCESSOR はそれぞれ別の値です。モデル別の診断結果を JSON に保存できます（`--num-ctx` を省略すると API 要求値は未指定として記録されます）。

```powershell
python scripts/diagnose_ollama_context.py
python scripts/diagnose_ollama_context.py --num-ctx 4096 --output config/ollama_context-4096.json
```

- コンテキストの変更は Ollama サーバーの環境変数または API 要求で設定し、モデルが既にロード済みならアンロード／再ロードして実際の CONTEXT と PROCESSOR を確認してください。
- 大きいコンテキストを評価する場合は、JSONL の `loaded_context` / `effective_num_ctx` が実際にその値になった環境でのみ評価します。`loaded_context` が `null`（診断時にアンロード済み）や要求より小さい値に丸められた場合は実測とは見なさず、Ollama サーバー設定と利用可能メモリを確認します。サーバーに `OLLAMA_KEEP_ALIVE` と `OLLAMA_NUM_PARALLEL=1` を設定して再起動し、再実行してください。大きいコンテキストはモデル・VRAM で実行可能な場合のみ指定してください。

## 派生モデル（Modelfile）

コンテキスト長を変えたモデルは元モデルを**上書きせず**、派生モデル名で作成します（`modelfiles/` に 8k 派生と 32k 派生の例があります）。同じ重みを使い、既定コンテキストだけを変更します。

```text
FROM existing-model:latest
PARAMETER num_ctx 4096
```

```powershell
ollama create werewolf-existing-model-4k -f Modelfile
ollama show werewolf-existing-model-4k --modelfile
```

量子化済み GGUF モデルを使う場合は「モデル名-量子化方式-コンテキスト長」の形式で登録し、元モデルの派生とは分けて評価します（`PARAMETER num_ctx` の変更は量子化ではなく、Ollama のコンテキスト長設定です）。

## リポジトリ構成

```text
src/werewolf/                ゲームエンジン、プロンプト生成、Ollama クライアント、CO/本文解析、漏えいガード、CLI
scripts/                     evaluate_models.py（モデル評価） / benchmark_context.py（コンテキスト比較） / diagnose_ollama_context.py（コンテキスト診断）
config/players.json          プレイヤー 8 体の設定
modelfiles/                  Ollama 派生モデル定義
tests/                       pytest（ルール、対象解析、秘匿性、CO、再試行、動的コンテキストの検証）
logs/                        ゲームログ（公開 / 管理者用）
evaluation-results/          評価・ベンチマーク結果（JSONL・summary.json）
SPECIFICATION.md             詳細仕様
```

## 開発

```powershell
pip install -e ".[test]"
python -m pytest -q
```

Ollama 不要で全フロー検証できるため、CI やローカル負荷なしのテストにもモックモードが使えます。

## 安全・再現性の設計

- LLM の出力はゲーム状態へ直接反映せず、必ず「ラベル解析 → 生存者・候補・行動者自身の検証 → 再要求 → 合法候補からのシード付き乱数フォールバック」を通ります。通信エラーは初回を含め最大 3 回（再試行 2 回）まで試行し、モデルが応答しなくてもゲーム全体は停止しません。
- 投票・占い・襲撃の対象行がない応答も、別称（`プレイヤー3` / `P3`）や最初の行の単一候補名から解決するため、ランダムフォールバックが減ります。
- プロンプトの案内文の例（プレースホルダ）をそのまま模倣した応答は致命的扱いではなく、台帳修復で該当文・該当行だけ除去し、本文と正当な CO は保持します（幻覚対象を含む結果行・CO 行は公開せず記録します）。
- 他プレイヤーの役職、占い結果、人狼相談が誤って漏えいしないことをテストで確認しています。
- 同一乱数シード＋同一モック応答で同一結果が再現できます（実モデルの応答は再現しません）。

## 関連ドキュメント

- `SPECIFICATION.md` — ゲーム仕様、プロンプト、Ollama 接続、ログと受け入れ条件
- `PROMPT_TEMPLATES.md` — プロンプトテンプレート
- `TASKS.md` / `PROJECT_PLAN.md` / `EVALUATION_PLAN.md` — 内部の開発・変更履歴（公開用説明は本 README に集約しています）



