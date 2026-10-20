python src/werewolf/cli.py --mode ollama --num-predict 2048 --timeout 180

python src/werewolf/cli.py `
  --mode ollama `
  --models-file config/players.json `
  --timeout 180 `
  --num-predict 2048 `
  --log-dir logs$env:CUDA_VISIBLE_DEVICES="0,1"


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
