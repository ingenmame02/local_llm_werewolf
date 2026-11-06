python src/werewolf/cli.py --mode ollama --num-predict 2048 --timeout 180

python src/werewolf/cli.py `
  --mode ollama `
  --models-file config/players.json `
  --timeout 180 `
  --num-predict 2048 `
  --log-dir logs


$env:CUDA_VISIBLE_DEVICES="0,1"
$env:OLLAMA_SCHED_SPREAD="1"
$env:OLLAMA_FLASH_ATTENTION="1"
$env:OLLAMA_CONTEXT_LENGTH="8192"
ollama serve

`--num-ctx`は省略時、Ollama APIリクエストには含まれず、サーバー／モデル側の既定値が使われます。必要な場合だけCLIの`--num-ctx`または`config/players.json`のプレイヤー別`num_ctx`で指定してください。

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

## 議論応答形式

新形式は1回の生成でCOと議論本文を分離します。COは公開主張の構造化に使い、会話履歴には本文だけを追加します。旧形式の明示的な文章COも互換のため受け付けます。

```text
【CO】なし または 占い師CO：対象=Player 2、判定=人間、理由=占い結果、投票提案=Player 3へ投票
【本文】Player 2への疑いは弱めます。Player 3の発言も確認したいです。
```

昼議論プロンプトには、確定ルール・ゲーム台帳とは分けて「人狼セオリーカード」を含めます。単独占い師COは1巡目後も対抗がいない場合に限って真寄りと評価し、真確定にはしません。未公開の人狼判定は早期公開を促し、人間判定だけの場合はCO・結果を保留する選択肢も示します。

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
