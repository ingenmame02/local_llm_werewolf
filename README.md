python src/werewolf/cli.py --mode ollama --num-predict 2048 --timeout 180

python src/werewolf/cli.py `
  --mode ollama `
  --models-file config/players.json `
  --seed 20261106 `
  --timeout 180 `
  --temperature 0.7 `
  --log-dir logs `
  --private-log `
  --show-private


$env:CUDA_VISIBLE_DEVICES="0,1"
$env:OLLAMA_SCHED_SPREAD="1"
$env:OLLAMA_FLASH_ATTENTION="1"
$env:OLLAMA_CONTEXT_LENGTH="8192"


$env:OLLAMA_NUM_PARALLEL="1"
$env:OLLAMA_KEEP_ALIVE = "5m"
ollama serve

`--num-ctx`は省略時、Ollama APIリクエストには含まれず、サーバー／モデル側の既定値が使われます。現在の`config/players.json`では全プレイヤーの`num_ctx`を12288に設定しているため、プレイヤーごとのAPI要求へ明示されます。プレイヤー別`num_ctx`はCLIの`--num-ctx`より優先されます。

--show-privateで管理者用ログを生成し、端末にも表示
--private-logで管理者用ログだけを生成
--private-logでは端末に秘密情報を表示しない

## Ollamaコンテキスト診断

Modelfileの`PARAMETER num_ctx`、Pythonプロセスの`OLLAMA_CONTEXT_LENGTH`、API要求値、ロード済みモデルの`/api/ps` CONTEXT/PROCESSORは別の値です。次のコマンドでモデル別の診断を`config/ollama_context.json`へ保存できます。`--num-ctx`を省略するとAPI要求値は未指定として記録されます。

```powershell
python scripts/diagnose_ollama_context.py
python scripts/diagnose_ollama_context.py --num-ctx 4096 --output config/ollama_context-4096.json
```

コンテキスト変更はOllamaサーバーの環境変数またはAPI要求へ設定し、モデルが既にロード済みならアンロード／再ロードしてください。Modelfileを変更する場合は元モデルを上書きせず、派生モデル名を使います。

```text
FROM existing-model:latest
PARAMETER num_ctx 4096
```

```powershell
ollama create werewolf-existing-model-4k -f Modelfile
```

## コンテキスト比較ベンチマーク

Ollamaサーバーはゲームと同様に直列実行し、`OLLAMA_NUM_PARALLEL=1`を設定してから再起動してください（この環境変数はクライアントCLIからではなく、Ollamaサーバープロセスの環境へ設定します）。4096と8192の二層形式率・形式適合率・解析成功率・応答時間・再試行・CPU使用率・Ollamaプロセスメモリ、利用可能ならNVIDIA GPU使用率・VRAMを比較し、結果をJSONLへ追記します。

```powershell
pip install -e ".[benchmark]"
python scripts/benchmark_context.py --context-lengths 4096 8192 --rounds 2
```

結果は`evaluation-results/context-benchmark.jsonl`に保存されます。GPU使用率とVRAMは`nvidia-smi`対応GPUがある場合に記録されます。`ollama ps`のPROCESSORはCPU/GPUへのロード割当であり、使用率そのものではありません。Modelfile設定やサーバー環境変更後はモデルを再ロードし、実際のCONTEXTとPROCESSORを確認してください。

TASK-030では会話コンテキストを`full_transcript`（公開会話全文）、`structured_ledger`（台帳と現在の質問）、`hybrid_retrieval`（直近発言と関連発言。既定）から選べます。比較ベンチマークは同一の固定シナリオ・生成設定でコンテキスト長と戦略を組み合わせ、質問回答率、新しい根拠カテゴリの有無、投票意向の具体性、再試行・フォールバック、入出力トークン、処理時間、CPU/RAM/GPU/VRAMをJSONLに記録します。コンテキスト拡大による改善・残存・新規失敗も比較します。32k以上を採用する基準は事前に固定し、最低10件の議論で質問回答率・新根拠カテゴリ率・投票意向具体率のいずれかが10ポイント以上改善、解析成功率が悪化せず、平均応答時間の増加が25%以内の場合に限ります。`thinking_present`は記録しますが、Ollama応答から思考トークンだけを分離して正確に数える値は取得できません。

TASK-031では、プレイヤー設定の`num_ctx`（要求値）、`max_num_ctx`、`recommended_num_ctx`、`context_strategy`でモデル別の運用値を指定できます。実効値は`/api/ps`から取得できた場合のみ記録し、取得できない場合は未計測のままです。議論プロンプトにはゲームエンジンが公開台帳・投票・発言IDから決定的に更新したプレイヤー別推理メモを付けます。狼相談や占い師本人の結果は本人専用のメモ領域に分離し、他プレイヤーへ共有しません。

コンテキストベンチマークは既定で推理メモoff/onを別条件として比較します。既存の比較コマンドを一度だけ実行する場合は`--reasoning-memory off`または`--reasoning-memory on`を指定してください。出力JSONLの各試行には要求・実効`num_ctx`、トークン数、終了理由、応答時間、メモリサンプルが記録されます。

例としてCALM3のハイブリッド履歴を、8k／12k／16k要求と推理メモoff/onで各10件比較します。32k条件を追加する場合は、JSONLの`effective_num_ctx`が実際に32768となる環境でのみ評価してください。

```powershell
python scripts\benchmark_context.py --models werewolf-calm3-22b-8k --context-lengths 8192 12288 16384 --context-strategies hybrid_retrieval --reasoning-memory both --rounds 10 --actions discussion --output evaluation-results\task031-calm3-memory-r10.jsonl
```

```powershell
ollama create werewolf-calm3-22b-32k -f modelfiles\calm3-22b-32k.Modelfile
ollama show werewolf-calm3-22b-32k --modelfile
```

32k派生モデルは8kモデルと同じ重みを使い、既定コンテキストだけを32kに設定します。専用設定`config/players_task030_32k.json`はPlayer 1だけを派生モデルへ切り替え、残りのプレイヤーと議論条件はそのままにします。まず1条件だけ試し、JSONLの`loaded_context`が32768になったことを確認してください。`loaded_context`が`null`の場合はモデルが診断時にアンロード済みです。Ollamaサーバープロセスに`OLLAMA_KEEP_ALIVE=5m`と`OLLAMA_NUM_PARALLEL=1`を設定して再起動し、パイロットを再実行してください。16k以下のままなら32k実測とは見なさず、Ollamaサーバー設定や利用可能メモリを確認します。32kはモデル・VRAMで実行可能な場合のみ指定してください。

```powershell
python scripts\benchmark_context.py --models-file config\players_task030_32k.json --models werewolf-calm3-22b-32k --context-lengths 32768 --context-strategies hybrid_retrieval --rounds 1 --actions discussion --output evaluation-results\task030-calm3-32k-pilot.jsonl
```

`loaded_context`が32768と確認できたら、各条件10ラウンドの本比較を実行します。8k側も同じラウンド数で取り直す場合は、既定設定ファイルを使い、モデル名を`werewolf-calm3-22b-8k`、出力先を別ファイルにしてください。単一ゲームを同じ条件で比較するときは`--seed`とモデル設定を固定し、`--context-strategy`と`--num-ctx`だけを切り替えます。`config/players.json`のプレイヤー別`num_ctx`を指定値で上書きするには`--override-player-num-ctx`も付けてください。全履歴モードのプロンプト監査には公開発言のSHA-256、通し発言番号、選択理由、除外範囲が記録され、会話本文や非公開の狼相談・占い結果は診断イベントに含めません。

```powershell
python scripts\benchmark_context.py --models-file config\players_task030_32k.json --models werewolf-calm3-22b-32k --context-lengths 8192 12288 16384 32768 --context-strategies full_transcript structured_ledger hybrid_retrieval --rounds 10 --actions discussion --output evaluation-results\task030-calm3-32k-r10.jsonl
```

比較対照の8kモデルも10ラウンドで実行する場合は、32k結果とは別ファイルへ保存します。

```powershell
python scripts\benchmark_context.py --models-file config\players.json --models werewolf-calm3-22b-8k --context-lengths 8192 12288 16384 --context-strategies full_transcript structured_ledger hybrid_retrieval --rounds 10 --actions discussion --output evaluation-results\task030-calm3-8k-r10.jsonl
```

## 議論応答形式

新形式は1回の生成でCOと議論本文を分離します。COは公開主張の構造化に使い、会話履歴には本文だけを追加します。旧形式の明示的な文章COも互換のため受け付けます。

二段階COを全プレイヤーに適用して1ゲーム確認する場合は、次のように実行します。全員にCO専用リクエストが追加されるため、従来方式より呼び出し回数・所要時間が増える可能性があります。

```powershell
python src/werewolf/cli.py --mode ollama --models-file config/players.json --seed 20261106 --timeout 180 --discussion-claim-mode all --log-dir evaluation-results/task027-task028-game --private-log
```

`--discussion-claim-mode candidates`は占い師・CO公開済みプレイヤーを対象とする比較用です。`all`は全員に二段階COを適用します。ログには非公開情報が含まれるため、共有時は取り扱いに注意してください。

公開COと占い結果は`ClaimLedger`で冪等に管理され、同一COの再送は本文から取り除かれます。`all`でもCO判定は同じプレイヤーに毎ターン行わず、初回判定・未公開の新しい占い結果・明示的な訂正要求時に限定します。候補外の襲撃・占い・投票は短い再回答後に合法候補へ補正し、死亡者への回答要求など台帳で修復できる違反は再生成せず`ledger_repair`として記録します。

ゲーム結果JSONLには公開FactLedgerと比較用の`ledger_metrics`（CO呼び出し、実送信・再試行、重複CO、台帳修復、本文保持、質問未回答、処理時間）が含まれます。占い師本人だけの結果、候補集合、生成前後の台帳スナップショットは管理者用の`--private-log`にのみ記録されます。同一シードでモードを比較する場合は、各モードを同じ`--seed`で実行し、JSONLの`discussion_claim_mode`と`ledger_metrics`を比較してください。

CO専用応答では、構造化されたCO／結果行だけを採用し、余分な説明文は議論本文へ混入しないよう破棄します。COなし・保留に誤った結果行が付いた場合も、結果を公開せずCOなし・保留として扱います。COプロンプトは会話履歴を重ねず、既に記録された公開COと占い師本人の結果だけを渡します。

```text
【CO】なし または 占い師CO：対象=Player 2、判定=人間、理由=占い結果、投票提案=Player 3へ投票
【本文】Player 2への疑いは弱めます。Player 3の発言も確認したいです。
```

昼議論プロンプトには、確定ルール・ゲーム台帳とは分けて「人狼セオリーカード」を含めます。単独占い師COは1巡目後も対抗がいない場合に限って真寄りと評価し、真確定にはしません。未公開の人狼判定は早期公開を促し、人間判定だけの場合はCO・結果を保留する選択肢も示します。

`config/players.json`ではプレイヤーごとに弱い議論スタイル（分析、質問、慎重、合意形成、直感、主張など）を指定します。個性は昼の議論本文だけに適用し、CO判定や対象選択には適用しません。議論の質・事実整合性・質問回答・形式指示より優先度を下げ、追加の説明や質問を強制しない設定です。

再試行では失敗理由に応じた修正指示と、直近の公開発言・公開COなど必要な情報だけを再送します。占い師本人の非公開結果は、その占い師の再試行にのみ含めます。モデルごとの生成上限は`config/players.json`のプレイヤー別`num_predict`で調整できます。

## 行動別の生成上限（num_predict）

議論・人狼相談・襲撃・占い・投票・CO補正で生成上限を分けます。優先順位はplayer行動別 → CLI行動別 → 全体設定 → player全体 → CLI全体 → 行動別既定値です。`--num-predict`を省略すると行動別既定値を使い、`--num-predict-by-action`（複数指定可）で上書きできます。

```powershell
python src/werewolf/cli.py --mode ollama --num-predict-by-action discussion=1024 --num-predict-by-action vote=256
```

player設定では`players.json`の`num_predict_by_action`に行動キーで書けます。キーは`discussion`、`wolf_discussion`（別名`wolf_consult`）、`attack`、`seer`、`vote`（別名`target`）、`co_correction`です。既定値は議論(1536)、人狼相談(768)、襲撃(256)、占い(256)、投票(256)、CO補正(256)です。襲撃・占い・投票・CO補正は最終回答が1行（またはCO最小3行）で済む行動なので、全体`num_predict`が大きい思考系モデルでは上限キャップ（襲撃/占い/投票=768、CO補正=384）へ降ろし、ムダな思考生成による長時間応答を抑制します。議論・人狼相談は根拠を説明できる文量を確保するため全体設定をそのまま使います（昼議論の文量目安は5〜8文）。生成起因の再試行（空応答・思考のみ・`done_reason=length`）では`num_predict`を倍へ増やして再送します。各行動の要求は`llm_request_configuration`へ`action_budget_key`と`num_predict_source`（`player_by_action`、`cli_by_action`、`global_capped`、`player_global`、`cli_global`、`default_by_action`）を記録し、CLI表示にも「行動」「優先元」を出します。

## 思考生成（think）の行動別設定

Ollamaの`num_predict`は思考トークンも消費します。思考系モデルでは、1行で済む行動（襲撃・占い・投票・CO補正）の思考が上限を食い、`done_reason=length`＋空content→再試行、というリトライが大量に発生していました。そこで思考生成を行動別に制御します。

既定方針は議論(思考on)、人狼相談(思考on)、襲撃(思考off)、占い(思考off)、投票(思考off)、CO補正(思考off)です。Ollamaへはリクエスト毎の`think`として送り、アプリのPython環境や他のリクエストには影響しません。優先順位はplayer行動別（`players.json`の`thinking_by_action`）→ CLI行動別（`--thinking-by-action`）→ player全体（`thinking`）→ CLI全体（`--thinking`）→ 行動別既定値です。

設定未指定（行動別既定値を使った場合）の思考on行動では、思考トークンの残余を確保するため下限へ引き上げます（議論3072、人狼相談1536。`num_predict_source=thinking_allowance`）。player・CLIの明示設定は優先順位上より高いため、下限で上書きしません。実機で思考onが上限を思考だけで使い切って`done_reason=length`＋空contentになったモデルは `thinking_budget_exhausted` として記録し、以後そのモデルの思考行動はoffへ落とします（`thinking_source=thinking_budget_exhausted`）。

```powershell
python src/werewolf/cli.py --mode ollama --thinking-by-action discussion=true --thinking-by-action vote=false
```

再要求（CO補正・軽い再要求・対象選択の再要求）は、短い必須項目だけを求めるため思考をoffに切り替えます（`thinking_source=retry_thinking_off`）。このときプロンプトにも「思考生成（think）はオフ」と、実際に送信する生成上限（倍へ増やした値）を明記し、指示とリクエストの値を一致させます。`llm_request_configuration`と評価JSONLには`thinking_enabled`・`thinking_source`・試行別の`thinking_by_attempt`を記録し、CLIは`think=on|off (優先元=...)`を表示します。集計側は`thinking_off_requests`・`thinking_retry_off_requests`・`thinking_only_count`・`truncated_count`でモデル別の削減効果を比較できます。

`think`を受け付けないモデルではOllamaがHTTP 400を返します。同じ要求を再送せず、ただちに`think=false`で1回だけ再送し、`thinking_unavailable`イベントと`GameState.model_capabilities`へ記録して、以後の全行動でも思考をoffにします（`thinking_source=model_capability`）。CLIは「思考非対応のためthink=offへ切替」を表示します。

## 評価ハーネスのモデル絞り込みと文量比較

評価は8モデル一括ではなく、まず1〜2モデルだけ先行実行できます。`--models`はモデル名・プレイヤー名の部分一致（大文字小文字無視）で評価対象を絞り、プロンプトは常に8人分の名前を使います。

```powershell
python scripts/evaluate_models.py --models muse-glimmer --output evaluation-results/task024-disc
python scripts/evaluate_models.py --models glm-4.7 --models calm3 --output evaluation-results/task024-pilot
python scripts/evaluate_models.py  # 絞り込みなし＝全8モデル
```

文量の変更効果は出力文字数ではなく、`done_reason`別（空応答・思考のみ・`done_reason=length`）、生成トークン数（`eval_count`）、生成上限の実解決と出所（`action_budget_key`・`num_predict_source`・`num_predict_by_attempt`）、解析成功率、具体性（`specific_rate`）、実送信回数（`request_attempts`）、処理時間、フォールバック率で比較します。`summary.json`には`models_filter`・`selected_models`・モデル別の`by_action`・思考on/off送信数・`thinking_affordable_num_predict`が記録され、量子化評価と同じ指標でJSONLへ保存されます。`generation_truncated`が多いモデルは上限増の候補、タイムアウトが多いモデルは上限据え置き候補として分けて判定します。
