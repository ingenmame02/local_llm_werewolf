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
$env:OLLAMA_CONTEXT_LENGTH = "16384"
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
| `--no-body-reservation` | off | モデル×アクション別の本文予算予約（初回リクエストでthinking余白＋本文最低予算を分離して確保）を無効化し、従来動作との比較を可能にする |
| `--num-predict INT` | 未設定 | 生成上限（全体設定）。省略時は行動別既定値 |
| `--num-predict-by-action ACTION=N` | — | 行動別生成上限の上書き（複数指定可） |
| `--thinking {auto,on,off}` | `auto` | 思考生成。`auto` は行動別既定（議論・人狼相談 on／襲撃・占い・投票・CO補正 off） |
| `--thinking-by-action ACTION=true\|false` | — | 行動別思考の上書き（複数指定可） |
| `--discussion-claim-mode {legacy,candidates,all}` | `legacy` | 議論 CO の生成方式（旧来併走／候補のみ二段階／全員二段階） |
| `--context-strategy {...}` | `hybrid_retrieval` | 議論に渡す履歴の構成（後述） |
| `--question-hint-mode {strict,soft,none}` | `soft` | 直前の質問への回答指示の強さ。`soft` は観測のみ／`strict` は旧来（1文目ルール＋1回再生成）／`none` は質問欄も指示も出しません |
| `--novelty-policy {publish,append,legacy_repair}` | `publish` | 低新規性（既出論点の連呼）時の処理。`publish` は本文をそのまま公開（再生成・文の除去なし）／`append` は本文を保持し未論点への短い任意の追記要求のみ／`legacy_repair` は旧挙動（重複文の除去＋1回の意味的再要求・A/B比較用） |
| `--leak-guard-policy {tiered,legacy_strict}` | `tiered` | 秘密漏洩ガードのゲート分類。`tiered` は未公開の真の情報（未公開の占い判定・狼仲間関係・自分の真陣営・プロンプトや生成設定）だけ遮断し、公開済みCOや判定の再言及・他者の役職推測は本文をそのまま公開（推理メモは観測のみ）。漏洩した文だけの中立化を優先して全文再生成を避けます。`legacy_strict` は旧挙動（推理メモを含む全文照合で1件でも検出すれば全文再生成→削句。誤検出由来の再要求負荷とのA/B比較用） |
| `--bluff-role-claim {allow,deny}` | `allow` | 実際にその役職を持たないプレイヤーの役職CO（ブラフCO）と、持っていない占い結果の申告を許すか。`allow` はCO・対抗CO・判定申告をそのまま公開し、台帳では「本人の申告（`public_claim_line`）」として記録するだけで、エンジンが発行した実際の占い結果（`RoleEvidence`）とは分離します（どちらが真の占い師かは公開ログ・プロンプトに出ません）。`deny` は実保有者だけがCO・判定を公開でき、非保有者のCO行と結果行は公開前に除去しますが、それ以外の本文は一字一句そのまま残し再生成も要求しません。不受理数は中立カウンタ `role_claims_disallowed` だけで計上し、試みた役職は公開イベントへ出しません |
| `--day-timeline {explicit,off}` | `explicit` | 議論プロンプトに発言時点の時系列欄 `[DAY_TIMELINE]` を出すか。`explicit` は現在時点（Day / Turn / #順序）・自分の発言順（`[speech:N]`）・このターンで既知の先行発言・**まだ起きていないこと**（自分より後の発言者、今日の投票、今夜の夜行動）を5行／460字以内で示します。口調・論点・結論や推理の順番は決めず、公開本文は書き換えません（未来参照の検出は観測専用で `body_modified=false`）。`off` は時系列欄と検査を行わない旧挙動（A/B比較用）。発言ヘッダの短縮と台帳の時点スタンプは両設定で効きます |
| `--prompt-contract {simplified,legacy}` | `simplified` | プロンプト契約の整理。`simplified` は強い制約（非公開情報の秘匿・創作禁止・一人称視点・メタ話禁止・`[REASONING]` を `[BODY]` に混ぜない・ブロック順・応答言語）を `[BEHAVIOR_CONTRACT]` へ1箇所に集め、同じ命令文の再掲を減らします。発言本文は `[CONVERSATION_EVIDENCE]` の1箇所だけに出し、質問引用・自分の発言欄・行動記録の発言行・推測行は `[ref:sN]` 参照にします（履歴から削られた発言の空参照は作りません）。割り当てが足りないときだけ optional 節（論点要約・推理誘導・仮説差分・推測行・期限切れガイダンス・整合アラート・派生メモ）を段階的に落とし、確定事実台帳・現在の質問・自分の行動記録・自分の推理メモ・発言本文・時系列欄は落としません。口調・文量・構成・論点・結論は自由のままです。`legacy` は旧来の形（命令文の重複と本文再掲）でA/B比較用 |
| `--core-compression {layered,off}` | `layered` | core予算の段階圧縮。`layered` は証拠欄の中で、既に台帳へ正規化済みの過去の低優先発言の本文だけを時点・話者・`[speech:N]` を残す約70字の行へ畳みます（`[body withheld ...]`＝削除ではなくプロンプトでの省略で、公開ログの本文はそのまま）。質問対象・反証・投票・役職主張と占い判定の根拠、直近の発言は本文のまま温存し、単純な末尾切り捨ては行いません。ルール・秘匿境界・現在の質問・行動形式・自分の行動記録・エンジン発行情報（`anchor`層）はどの設定でも無傷です。`off` はTASK-057まで旧挙動（予算超過時はレコード削除のみ）でA/B比較用。※節順そのものは `--prompt-layout` で選びます
| `--prompt-layout {v5,v4,v3,v2,auto}` | `v5` | 議論プロンプトに載せる情報の**位置**を決める設定（言い方・論点・結論・推理順には干渉せず、口調・文量・構成の自由は変わりません）。`v5`（TASK-064の既定）は `v4` の連鎖へ `[ROLE_ACTION_BOUNDARY]`（エンジンが実際に実行した能力と本人の公開主張の境界）を挟み、現在質問の直後・自分の発言の直前に置きます（代表字数26,340での貪欲探索が最良と出した位置で凍結済み）。中身は `[DONE]`（エンジンがそのプレイヤー分として実行した占い・襲撃の日数だけで、判定本文は本人限定節のまま）と `[CLAIM]`（本人の公開CO `[claim:N]` は未検証である事実）の2行相当のみ。`v4` は境界節なしのTASK-062配置でA/B比較の基準側です。`v3` は ルール→応答言語→自分の行動記録→盤面の確定事実→本人限定情報、中部（セオリーカード・仮説差分・既出論点・派生メモ・期限付き案内といった落とせる派生情報のみ）、終端へ 公開主張の索引→時点→発言証拠→現在質問→**自分の発言と自分の推理メモ**→中立な議論の案内→秘匿境界→強い契約→出力形式→最終チェックリスト。人狼セオリーカード（約2,300字）は事実台帳の内側（先頭帯）から独立節へ出し、割り当て不足時に落ちる optional にしました。出力形式仕様の後ろに案内が載って形式指定が遠のく配置も解消済みです。`v2` はTASK-058の節順、`auto` は `--prompt-contract` に追随（legacy ではTASK-057までの字节順）。A/Bは同じseedで `ledger_metrics` の `avg_prompt_zone_{primacy,middle,recency}_chars` / `avg_prompt_instructions_to_end_chars` / `avg_gap_question_to_evidence` / `avg_gap_timeline_to_evidence` / `avg_gap_own_speeches_to_question` / `avg_gap_state_to_{evidence,question}` / `avg_gap_role_boundary_to_{question,own_speeches}` / `avg_claim_boundary_chars` / `duplicate_constraint_total` / `dedup_saved_chars_total` と、照応率・立場の一貫性・parse失敗率を突き合わせます |
| `--continuity {normalized,off}` | `normalized` | 日をまたぐ論理状態の正規化。エンジンが発行する短い状態差分 `[CONTINUITY_CHECKPOINT]`（確定事実 `[CONFIRMED]`／公開申告 `[SAID-unverified]`／自分の未検証仮説 `[YOUR-UNVERIFIED]`／反証済み `[NO-LONGER-STANDS]` を権威タグで分け、`observed DdTt` と証拠ID付きの1行だけ）を次の発言へ渡します。表示は変わった項目・反証・未解決の問いのみで、変わっていない項目は「個数＋チェックポイントID」の1行に畳み、正規化した日の平常発言本文は ID 行へ縮約（`evidence_continuity_folded_count`）して、プロンプトが日に比例して増えるのを止めます。反証された読みは履歴として出し、古い行が新しい事実として読まれるのを防ぎます。「結論を維持せよ」等の指示はなく、立場を自由に変えられます。`off` はチェックポイントを出さない旧挙動（A/B比較用） |
| `--context-integrity {off,warn,cap}` | `warn` | 実効コンテキストの整合チェック（TASK-065）。`warn` は設定値そのまま送りを続け、`configured`（プレイヤー/CLI設定）・`server_observed`（プロセスenv・Modelfile）・`loaded`（/api/ps実測）・`requested`（送信要求）を1つの値へ潰さず分離記録し、不一致は**起動時に一度だけ**警告して出します（timeoutや通信失敗の後で初めて知る状態をやめる）。`cap` はサーバー証拠が確認できた安全値へ送信前に縮退します（採用値は解決結果に反映され、実際の送信num_ctx・動的予算・再要求fitまで同じ枠になる。`num_ctx`はリクエストでModelfileを上書きできるため、縮退は明示選択時のみ）。`off` はTASK-065以前の旧挙動でA/B比較用。議論プロンプトへ制約文は増えず、出力は数値とモデル名だけです（秘密情報なし） |
| `--timeline-intent {separated,literal}` | `separated` | 未発言者に関する未来参照の意図分離。`separated` は「その人が過去に述べた」と断定する実質的な未来参照（`past_attribution_to_unspoken`）だけを `future_event_reference` に数え、これから説明・回答を求める質問行為（`future_question_to_unspoken`）とこれからの投票・夜行動への推測（`future_action_speculation`）は別カウンタへ逃がして違反にしません（game_425755ce9a83 Day2 T1 Player 1の「問いただすことを提案します」型）。`literal` はTASK-059までの平坦算入（誤検出率のA/B基準）。どちらの設定でも公開本文はちょう一词一句そのままです |
| `--player-name-resolution {normalized,literal}` | `normalized` | プレイヤー名表記ゆれの内部解決。`Player 6`（NARROW NO-BREAK SPACE）、`player6`、全角 `Ｐｌａｙｅｒ６`、`プレイヤー7`、`P 7` を台帳登録・証拠ID解決・質問回答判定の入口だけで正規名へ解決し、`unknown_player_reference` と ledger repair を減らします（公開ログの表記は一切書き換えない／実在しない番号は正規化せず未解決として観測）。`literal` は完全一致の旧来判定でA/B比較用。内部判定でのみcanonical nameを使い、発言そのものの語彙・口調には干渉しません |
| `--output-stability-scope {action,model}` | `action` | thinking・timeout・生成上限の処置スコープ。`action` はモデル×行動×直近失敗窓（4送信）に限定し、投票で thinking が枯れても議論・占い・襲撃の思考は自由のまま残します。同じ種の失敗が2つ以上の行動で確認できたときだけ証拠の合算としてモデル側へ広げ、本文付き成功が3続けば縮小・offを解除します（`stabilization_relaxations_total`）。`model` は恒久適用の旧挙動（A/B比較用）。推理手順・結論・口調・話題は指定しません |
| `--retry-context-policy {tiered,legacy_early_compression}` | `tiered` | 再生成コンテキストの段階化（TASK-066）。`tiered` は再生成の1回目で**初回生成と同じ公開証拠をそのまま**使い、指摘されたエラーだけの短い補足と前回書いた発言（秘匿違反の疑いが無い場合のみ引用扱い）を足して「その問題だけ」直します。古い発言本文を畳む圧縮は、実効コンテキスト割り当て不足・サーバー報告のコンテキスト超過・timeout 再送・**同一失敗クラスが続いたとき**だけ進めます（2回目=`light_compressed`＝既出論点・推理誘導・仮説差分・推測行・セオリーカード・派生メモ・整合アラート・期限付き案内・重複引用・ラウンド台本といった optional 派生節のみ除去、3回目=`last_resort_compressed`＝core_ledger のみ）。形式エラーや質問未回答で即座に履歴を捨てる従来挙動をやめ、議論の切れ目を減らします。`legacy_early_compression` は TASK-066以前（再生成種別を問わず最初の再生成から core_ledger 短縮）で A/B 基準側。どちらの設定でも口調・文量・論点・結論・推理順は指定せず、公開ログの本文は一字一句変更しません（圧縮はプロンプト上の表現だけ）。診断はイベント `retry_context_stage` とサマリ `retry_full_preserved_rate` / `retry_compression_rate` / `avg_retry_prompt_saved_chars` / `retry_prompt_ratio` / `avg_dropped_speech_count_when_folded` / `avg_dropped_evidence_ids_when_folded` / `avg_preserved_evidence_ids` / `retry_body_shortened_rate`。timeout・コンテキスト起因の即時圧縮は「圧縮は最終手段」に対する**意図的な例外**として `retry_timeout_stage_events` / `retry_timeout_stage_share` で別建てにし、削除した証拠IDは載っていた節で分類して core節・現在質問由来の想定外欠落を `retry_dropped_ids_from_core_sections_total` / `retry_dropped_ids_question_linked_total` / `retry_unexpected_id_loss_events` に数える（light_compressed の削除は optional派生節由来のみ＝設計どおり、発言本文は保持されることをローカル模擬で実測）。詳細は「コンテキスト戦略」節 |

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
- 再生成コンテキストの段階化（既定 `tiered` / `--retry-context-policy legacy_early_compression` で比較・TASK-066）: `retry_context_policy.py` が失敗クラス（`output_format` / `required_format_or_claim` 形式系、`question_answer`、`semantic_light`、`secret_leak`、`empty_or_thinking_only`、`generation_cap`、`timeout`、`context_overflow`）ごとに**圧縮を進める条件を分けます**。`full_preserved` は初回生成と同一のプロンプト＋対象エラーだけの補足で、`[speech:N]`・`[claim:N]`・`[CONTINUITY_CHECKPOINT]`・発言本文まで無傷のままです（実測: 初回13,825字 → 再生成14,454字。節約0でも証拠欠落0）。同じ失敗が 2 度続いたときの `light_compressed` は optional 派生節（既出論点・推理誘導・仮説差分・推測行・セオリーカード・継続メモ・整合アラート・期限付き案内・重複引用・ラウンド台本）だけを除去し、確定事実台帳・現在の質問・役職主張・投票／死亡履歴・直近発言・反証・checkpoint・自分の行動記録と推理メモ・秘匿境界・出力形式は残します（実測 10,385字／−3,440字、除去6節）。`last_resort_compressed`（core_ledger 短縮）は、コンテキスト超過・実効割り当て不足・timeout再送・同一失敗の3回目だけで発火します（A/Bでは legacy が最初の再生成から全証拠を落としていたのに対し、tiered の1回目は ID 欠落 0、light では 1 件のみ）。再生成プロンプトの**節順**は通常生成の `PROMPT_LAYOUT_V5_ORDER` を変えず、再生成専用のブロック語彙（`RETRY_PROMPT_BLOCKS`／`RETRY_PROMPT_ORDER_V1`）として凍結しました。同じ内容のまま配置だけを比べるコストモデル（LL不使用の貪欲探索）で legacy 配置 −20.0 → 凍結配置 **47.56**、隣接ペナルティ 0.0（現在質問→直近発言→修正指示→出力形式を終端帯に隣接させ、未検証のセオリー行を先頭帯から退避）。崩れは `validate_retry_order` が静的検査し（身分・確定事実を3ブロック目以内に置く／形式仕様を末尾に置く／判定語を修正指示より後ろへ置かない）、ハード違反はスコアから差し引きます。圧縮時の節約実測は `retry_context_stage` イベント（`retry_stage` / `compression_trigger` / `retry_prompt_style` / `preserved_sections` / `dropped_sections` / `preserved_evidence_ids` / `dropped_evidence_ids` / `dropped_speech_count` / `dropped_speech_chars` / 畳んだ発言の最古・最新時点）、集計はモデル別 `retry_full_preserved_rate`（早期圧縮を避けられた度合い）、`retry_compression_rate`、`avg_retry_prompt_saved_chars`、`retry_prompt_ratio`、`avg_dropped_speech_count_when_folded`、`avg_dropped_evidence_ids_when_folded`、`avg_preserved_evidence_ids`、`retry_body_shortened_rate`。発言イベントには `body_source_stage` / `regenerated_body` を添え、再生成文の根拠接続率を通常生成と分けて見ます。削除した証拠IDは**載っていた節で分類**する（`dropped_ids_protected_count`=core節由来＝圧縮の想定外、`dropped_ids_question_linked_count`=現在質問が引用していたID、`dropped_ids_optional_only_count`=optional派生節由来のみ＝設計どおり、`dropped_speech_bodies_still_present` / `dropped_speech_bodies_lost`=ID行が消えても発言本文がプロンプトに残るか。実測で light_compressed の削除1件は optional由のみ・core由0・質問由0・本文喪失0、テンプレ案内 `[speech:N]` 等はIDとして数えない）。想定外の欠落はサマリ `retry_dropped_ids_from_core_sections_total` / `retry_dropped_ids_question_linked_total` / `retry_unexpected_id_loss_events`（発言履歴を畳まない段階でcore節IDが消えた事件数）に積む。timeout とコンテキスト起因の即時圧縮（`last_resort_compressed` 直行）は「圧縮は最終手段」という原則に対する**意図的な例外**として `retry_timeout_caused` / `retry_immediate_compression_exception` をイベントに残し、別建て評価用へ `retry_timeout_stage_events` / `retry_timeout_stage_share` に分離する（待ち時間と超過解決が目的で、長文再送は逆効果というTASK-035/046の実測に基づく。原則側の改善率と混ぜないこと）。口調・文量・論点・結論・推理順を指示する文言は新設しておらず（テストで禁止語を照合）、本文の自動破棄も意味的書き換えもしません。
- 全履歴系のプロンプト監査では公開発言の SHA-256、通し発言番号、選択理由、除外範囲を記録し、会話本文や非公開の狼相談・占い結果は診断イベントに含めません。
- core予算の段階圧縮（既定 `layered` / `--core-compression off` で比較）: TASK-057でも8k系モデルは optional を stage4まで削った後に core が残って割り当てを超えていました。`core_compression.py` が core を `anchor`（常に全文）/ `recent`（本文保持を優先する直近情報）/ `historical`（台帳済みの古い低優先発言）へ分け、`[CONVERSATION_EVIDENCE]` の中だけを stub 行（時点・ID・話者を残す約70字）へ畳みます。同じ群では古い順に畳み、質問対象・反証・投票・役職主張と占い判定（`looks_like_claim`）には触らないため、単純な末尾切り捨てで最新の議論や根拠が消えることはありません。実測は `core_anchor_chars`（layered/off で一致するはず）、`core_recent_chars`、`core_historical_chars`、`evidence_withheld_body_total`、`evidence_withheld_saved_chars_total` で行い、公開ログの本文は不変です。
- プロンプトの節順（attention最適化・layout v2）: `simplified` では先頭（primacy）にルール→確定事実台帳→自分の一貫性アンカー（行動記録・今日の発言・エンジン発行情報・自分の推理メモ）、中部（注意が弱い帯）に落としやすい派生節（仮説差分・既出論点・期限切れ案内・派生メモ・整合アラート）、その直後に発言証拠と現在の質問、終端（recency）に秘匿境界→強い契約→出力形式→チェックリストを置きます。`PROMPT_LAYOUT_V2_ORDER` を `validate_prompt_layout` で検査し、崩れは `prompt_layout_violations` として記録します（口調・論点・結論や推理順を決める文言を追加する節は存在しません）。この節順は TASK-060 で `v3`（セオリーカードを独立optional節へ、索引→時点→証拠→質問→自分の記録の連鎖を終端帯へ）、TASK-062 で `v4`（同じ連鎖へ `[CONTINUITY_CHECKPOINT]` を発言証拠の直後・現在質問の直前に追加）へ更新され、`PROMPT_LAYOUT_V4_ORDER` を同じ検査器で検証します。代表字数での貪欲探索が最良として再現する順を凍結しており、A/Bは `adopted_layout_score()` / `v3_layout_score()` と `avg_prompt_zone_*`、`avg_gap_*`、照応率・立場の一貫性・parse失敗率で行います。`legacy` は旧来の字节順のままなので、同一seedで `--prompt-contract legacy --core-compression off` を回せば基準側が壊れていないことを確認できます。
- 日をまたぐ論理状態の正規化（既定 `normalized` / `--continuity off` で比較・TASK-062）: `continuity_state.py` が、エンジン確定の事実（襲撃死・投票・処刑・生存者）、公開されたCOと占い申告、本人の未検証仮説、反証されて成り立たなくなった読みを、それぞれ権威レベルを明記したID付き短行（1行280字上限）としてプレイヤー別に保持します。プロンプトには「前回から変わった項目」「反証・差し替え」「未解決の問い」だけを出し（changed 4／overturned 3／unresolved 3行）、変わっていない項目は本文もIDも再掲せず個数とチェックポイントIDの1行に畳みます。エンジン日誌では `continuity_state_chars`（実測 約1.0k字／リクエスト）、`avg_continuity_items_changed`、`avg_continuity_items_unchanged`、`evidence_continuity_folded_total`、`continuity_fold_saved_chars_total`、ゲーム終了時に `continuity_chars_by_day` と `continuity_max_day_delta`（過去の発言を積み増さないため日跨ぎで線形増加しないことを見る）を記録します。状態節には口調・論点・結論・推理順を決める命令文を置いていません（テストで禁止語を照合）。
- 運用安定化の4整理（TASK-065・いずれも観測と数値制御だけで本文の自由を縛らない）:
  (1) **実効コンテキスト整合** `context_integrity.py`＋`--context-integrity`。ゲーム開始前にモデル別へ一度だけ診断し（ログイベント `context_integrity_preflight`）、`configured / server_observed / loaded / requested` と `max_num_ctx` を分離保持します。設定がサーバー観測値を超えていれば kinds（`configured_above_server_observed`／`loaded_context_unavailable` 等）として記録し、端末へ「送信前に確認してください」を出します（`game_425755ce9a83` の Player 1型：設定16384／実観測4096を timeout後でなく起動時に把握できる）。`cap` のときだけ安全値へ縮退し、採用値は**解決結果そのもの**（`EffectiveContextResolution.limit`）に反映されるため、送信オプション・動的コンテキスト逆算・再要求fit・リクエストログまで同じ枠で揃います（`num_ctx_source=context_integrity_cap`、設定値は `integrity_requested_num_ctx` に残す）。起動時はモデル未ロードで `/api/ps` が空になりやすいので、その解決には `provisional` マークを付け、**送信が完了した時点で実ロード値を取り直す**（`context_resolution_stage=post_load_refresh`／試行上限3回。取得後は確定キャッシュなので毎アクションの診断は増えない）。集計は `context_integrity_checks` / `context_integrity_mismatches` / `context_integrity_caps_applied_events` / `context_resolutions_provisional` / `context_resolutions_refreshed_after_load` / `context_integrity_checked_models` / `context_integrity_mismatch_models` と、ゲームサマリ `context_integrity_evidence`（由来別の生の証拠）。
  (2) **未来参照の意図分離** `day_timeline.py`＋`--timeline-intent`。未発言者について、質問行為・これからの行動推測を違反から外し、過去の発言as根拠の断定だけ `future_event_reference` に残します（TASK-056/059の検出強度は下げない）。イベントは `timeline_intent_counts` / `past_attributions_observed` / `future_question_acts_observed` / `future_action_speculations_observed` / `timeline_intent_policy_label`、サマリは `timeline_unspoken_intent_counts` / `timeline_past_attributions_per_body` / `timeline_question_acts_per_body` / `timeline_action_speculations_per_body`。finding には `intent` を添え、`body_modified=false` は従来どおりです。
  (3) **プレイヤー名の内部解決** `name_resolution.py`＋`--player-name-resolution`。空白類（U+00A0/U+202F/U+3000等）・全角英数・大小文字・`プレイヤーN`/`P N`を NFKC 折畳み索引で実在名へ引きます（公開本文は不変・存在しない番号は吸収しない）。llm_actions は `unknown_player_reference` 判定・台帳登録入口・質問回答判定（`shares_mentioned_players`）でこの解決を使い、prompts の「自分への質問」認識（`_is_directed_to_player`）と self_consistency の事実突合も表記ゆれを追跡します。イベント `name_reference_resolution`（`name_variants_absorbed` / `unresolved_name_references` / `legacy_unknown_flags_avoided` / `mention_labels`）をモデル別に積まれ、サマリは `name_variants_absorbed_total` / `legacy_unknown_name_flags_avoided_total` / `unresolved_name_references_total`。
  (4) **thinking・timeout処置のスコープ限定** `output_stability.py`＋`--output-stability-scope`。失敗カテゴリを `thinking_budget / generation_cap / timeout / transport / output_format / prompt_logic` に分類してモデル×行動の窓へ積み、縮小（`timeout_fit`）と thinking off を「その行動」だけに適用します（投票で枯れても議論の思考は自由）。同じ kind が2行動で確認できたときだけ `model_wide_evidence` としてモデル側へ広げ、本文付き成功が3続けば `relax()` で解除します（`output_stabilization_relaxed`）。ログイベント `model_timeout_downgrade`（`stabilization_scope` / `scoped_to_action` / `action_budget_key` / `actions_with_downgrade` / `recent_categories` / `body_modified=false`）、`thinking_stabilization`、サマリ `output_stabilization_scope_events` / `stabilization_relaxations_total` / モデル別 `stabilized_actions`。本文が得られた場合はそのまま保持し、予算安定化を理由に口調・論点・結論を変えません。

- 役職能力の境界整理（既定・TASK-064）: `role_action_provenance.py` が役職情報を4つへ分けます。`engine_action`（エンジンがそのプレイヤー分として実際に実行した占い・襲撃の日付だけ。判定内容そのものは本人限定節から出しません）、`public_role_claim`（本人が名乗った公開COと `[claim:N]`。ブラフもこの分類で受理し本文は書き換えません）、`unsupported_role_action`（実行記録がないのに能力を行使したかのように断定した発言）、`inference`（他者の能力についての推測）です。検出は**観測専用**で、疑問形・仮定形・引用・否定は行使として数えず、本文の無害化・再生成には使いません（CO可否は従来どおり `--bluff-role-claim`、既定は許可）。議論プロンプトには `[ROLE_ACTION_BOUNDARY]` として「エンジン実行＝ `[DONE]`（何夜分やったか）／本人申告＝ `[CLAIM]`（未検証）」の2行相当だけを現在質問の直後・自分の発言の直前に置き、エンジン結果・公開主張・推測の本文は重複掲載しません（代表字数プローブで最良と出た位置を `PROMPT_LAYOUT_V5_ORDER` として凍結し、同じ検査器で節順の崩れを `prompt_layout_violations` に記録）。実測は境界節 平均194字（プロンプト増加は約+180字）、`avg_gap_role_boundary_to_question` 0.0076・`avg_gap_role_boundary_to_own_speeches` 0.0。どのCOが本物か・エンジン結果との一致／不一致は管理者ログ（`serialize_admin()`）だけが持ち、公開ログ・checkpoint・プロンプトへは中立な合計値だけを出します（`role_boundary_sections` / `avg_claim_boundary_chars` / `role_ability_assertions_observed` / `hedged_ability_mentions_observed` / `unsupported_role_action_count` / `claim_vs_engine_conflict_count` / `retry_added_ability_expressions` / `avg_claim_to_ability_turn_delta`）。夜の実行（占い・襲撃決定）は engine 側で [DONE] 実行記録へ積み、公開発言から能力断定を観測したときは CO初出→能力断定（またはその逆）のターン距離を数値だけのイベントとして積むので、「自分が実際にやった行動」と「自分が言った主張」の混同を抑えつつ、騙り・疑い・立場変更の自由はそのままです（命令文・誘導文言は増やさず、テストで禁止語を照合）。
- 引用の時点分類と未解決参照の原因分解（観測専用・本文は書き換えない）: `TurnSnapshot.citation_status()` が引用相手の状態を「同一ターンの先行発言／別ターン・別日の既発言／死亡後の履歴参照／まだ公開発言なし」へ分け、実際の未来参照候補は最後だけに絞ります。婉曲での仮定引用（`hedged_pending_citations_observed`）や未発言者名の一般言及（`mentions_without_attribution_observed`）は違反に数えず、自由な議論を萎ませません。本文中の `[speech:N]` / `[ref:sN]` は `evidence_reference_counts` で「時点で解決済み／まだ存在しないID＝実際の未来参照／時点情報がない旧ログ互換／帰属なのにIDなし」へ分解し、空本文のリクエストは `timeline_checks_skipped_empty_body` として検査対象に水増ししません。公開台帳の `serialize()["metrics"]` の形はTASK-056のまま据え置き、内訳はイベント診断と `observation_metrics()`（非公開面）だけで見ます。
- 発言時点の時系列明示（既定 on / `--day-timeline off` で比較）: `[DAY_TIMELINE]` に「現在時点・自分の発言順・このターンで既知の先行発言・まだ起きていないこと（後発プレイヤー／今日の投票／今夜の夜行動）」を5行460字以内で出します。エンジン側は発言ごとに `day / discussion_turn / speech_id / timeline_index` をスタンプし、日の開始・投票・処刑・襲撃死も登順で台帳（`public_timeline`）へ登録するため、「自分が先に言ったのか、相手が後に言ったのか」を本文を引かずに追えます。口調・論点・結論や推理の順番は指示しません。
- 同じ機能で節約も行う: 発言履歴ヘッダを従来形 `[Day 1 / Speech 3 / Player 2 / In-game speech]`（45字）から時点つき短縮形 `[D1 T2 s14 Player 2]`（約21字）へ縮約したため、ローカル模擬の実測では新しい欄の実費（約256字／リクエスト）より大きい削減（ヘッダ合計 約557字／同一履歴）となり、プロンプト总量は旧構成より約300字減少しました（`avg_day_timeline_chars`、`avg_evidence_char_count`、`evidence_header_style` でA/B検証可）。
- プロンプト契約の簡素化（既定 `simplified` / `--prompt-contract legacy` で比較）: `[BEHAVIOR_CONTRACT]` が強い制約を1度だけ書き、口調・文量・構成・論点・結論と推理の順番は自由だと明記します（質問へ答える義務なども「直接問われたときだけ」と条件つきで、自由行と矛盾させません）。節には `purpose / source / authority / expires_at` を持つ台帳（`prompt_contract.py`）を当て、コンテキスト不足時は optional 節だけを段階的に落とすため、確定事実・現在の質問・自分の行動記録・自分の推理メモ・発言本文は無傷のままです。
- 発言本文の二重送信をやめる: 同じ発言は `[CONVERSATION_EVIDENCE]` だけに載せ、`[Current Question]` の引用・`[Direct Answer to Previous Question]`・`[Your Speeches Today]`・`[SELF_ACTION_MEMORY]` の発言行・`Player Speech & Inferences` 行は `[ref:sN]`（=`[speech:N]` と同じ番号）で参照します。ローカル模擬の実測では `avg_prompt_user_chars` 17,484 → 14,114（−19.2%）、`max_prompt_user_chars` 21,659 → 16,509、同じ制約文の重複カウント 436 → 184、ID参照 573箇所、証拠外での本文再掲 6.80箇所／406字 → 0箇所／0字（同一履歴A/B）でした。`duplicate_constraint_total`、`dedup_saved_chars_total`、`contract_stage{1..4}_removals`、`avg_prompt_build_calls` でA/B検証できます。

## 生成上限（num_predict）と思考生成（think）

- 優先順位は **player 行動別 → CLI 行動別 → 全体設定 → player 全体 → CLI 全体 → 行動別既定値** です。行動別既定値は 議論 1536 / 人狼相談 768 / 襲撃・占い・投票 256 / CO補正 256。
- 最終回答が 1 行で済む行動（襲撃・占い・投票・CO補正）は上限キャップ（襲撃・占い・投票 768、CO補正 384）へ降ろし、思考系モデルの無駄な生成による長時間応答を抑制します。議論・人狼相談は根拠を説明できる文量を確保するため全体設定をそのまま使います（昼議論の文量目安は 5〜8 文）。
- 生成起因の失敗（空応答・思考のみ・`done_reason=length`）では `num_predict` を倍にして再送します。議論の 3 回目再要求は生成起因の失敗だけに限定し、意味検証の失敗は従来の 2 回までです。
- `num_predict` は思考トークンも消費します。思考の既定は 議論・人狼相談 = on / 襲撃・占い・投票・CO補正 = off で、リクエスト毎の `think` として Ollama に送ります。再要求（CO補正・軽い再要求・対象選択の再要求）は思考 off に切り替え、プロンプトにも「思考生成（think）はオフ」と実際に送信する生成上限を明記して、指示とリクエストの値を一致させます。
- 未指定の思考 on 行動では思考残余を確保するため下限へ引き上げます（議論 3072、人狼相談 1536）。player・CLI の明示設定は優先順位上より高いため、下限で上書きしません。
- 実機で思考が上限を使い切って `done_reason=length` ＋空 content になったモデルは `thinking_budget_exhausted` として記録し、以後そのモデルの思考行動を off に落とします。
- `think` を受け付けないモデルでは Ollama が HTTP 400 を返します。同じ要求を再送せず、ただちに `think=false` で 1 回だけ再送し、以後の全行動で思考 off にします。
- 失敗原因の分離（`output_budget_trace.py`・TASK-063）: モデル×アクション別に各試行を記録し、失敗を **thinking予算**（`thinking_budget`）／**生成上限**（`generation_cap`）／**timeout**／**通信**（`transport`）／**出力形式**（`output_format`）／**プロンプト論理**（`prompt_logic`）の6カテゴリへ分けます。`budget_bound_failures`（出力予算が原因）と `logic_bound_failures`（議論の論理が原因）を別々に数えるため、thinkingを止める前に「本当はどちらで壊れているか」を実測で判定できます。実測キーは `output_budget_categories`、`budget_bound_rate` / `logic_bound_rate`、`avg_prompt_char_count` / `max_prompt_char_count`、`content_preserved_count` / `content_lost_fallback_count`（フォールバック時に本文が保持されたか）です。
- `thinking-fixed` 系モデルの行動別予算（同）: モデル名に thinking / fixed を含むモデルは、本文の最低量（`FIXED_MODEL_BODY_MIN_TOKENS_BY_ACTION`）を確保したうえで thinking 上限（`FIXED_MODEL_THINKING_CAP_TOKENS_BY_ACTION`）を行動別に当てます。議論の文量を削らず thinking だけ絞るため、発言の自由（口調・論点・構成）には触れません。
- thinking-only の再要求は短縮版（同）: 思考だけを吐いて本文が無い場合、同じ長いプロンプトを再送せず、エンジン発行の `[CONTINUITY_CHECKPOINT]`（TASK-062の状態差分）＋現在質問＋出力形式だけを載せた短縮プロンプトへ切り替えます（`short_retry_requests` / `short_retry_rate`）。入力実字数が減るので強化モデルの再要求コストが下がり、状態は失われないため一貫性も保ちます。
- 各行動の要求は `num_predict_source`（`player_by_action` / `cli_by_action` / `global_capped` / `player_global` / `cli_global` / `default_by_action` / `thinking_allowance`）と `thinking_source` を記録し、CLI は「行動」「優先元」「think=on|off」を表示します。

## ログ

| ファイル | 内容 |
|---|---|
| `logs/game_<id>.jsonl` | 観戦用の公開ログ（公開発言、CO、投票、死亡、襲撃、占い履歴の公開分、勝敗と判定に用いた生存人数） |
| `logs/game_<id>.private.jsonl` | 管理者用ログ（人狼相談、占い師本人の占い結果、候補集合、プロンプト／生応答の診断メタ、失敗理由、台帳スナップショット、`context_strategy`、`integrity_alerts`、推理メモ、入力トークン量） |

- `--show-private`: 管理者ログを生成し、端末にも秘密情報を表示
- `--private-log`: 管理者ログだけを生成（端末には表示しません）
- 公開ログには秘密情報・診断メタデータ・失敗詳細を含めません。管理者ログを共有する場合は取り扱いに注意してください。
- `game_result` の `public_timeline` は公開イベントの時系列（時点 `available_at`、`speech_id`、`[speech:N]` の証拠ID、46字以内の要約のみ。発言全文・占い結果・役職・襲撃対象は載せません）。併せて `day_timeline_policy`、カテゴリ別の未来参照検出数（`vote_outcome_before_vote` / `death_before_death` / `speech_from_unspoken_player` / `night_action_before_night`）、発言時点の根拠追跡率 `evidence_grounding_rate_at_speech_time`、`avg_day_timeline_chars` / `avg_evidence_char_count` を記録します。検出は観測専用で、`discussion_message.body_modified` は常に `false`（本文は一字一句そのまま公開）です。

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

コンテキスト長を変えたモデルは元モデルを**上書きせず**、派生モデル名で作成します（`modelfiles/` に 8k・16k・32k 派生の例があります）。同じ重みを使い、既定コンテキストだけを変更します。現在の通常設定（`config/players.json` と `config/players_free.json`）は、8k系8モデルの16k派生（`werewolf-*-16k`）を使用します。8kモデルへ戻す場合は、各設定のモデル名と `num_ctx` を旧値へ戻してください。

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
src/werewolf/                ゲームエンジン、プロンプト生成、Ollama クライアント、CO/本文解析、漏えいガード、役職出所台帳、Day内時系列台帳（day_timeline.py）、CLI
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



