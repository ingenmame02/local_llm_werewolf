"""P0効果の確認用（LLMを呼ばない）。ログJSONLから発言履歴保持・再要求率・ctx使用率を出力する。

使用例:
  python _p0_compare.py logs/game_xxxx.private.jsonl
  python _p0_compare.py logs/game_22754212a062.private.jsonl   # 基準（修正前のゲーム）
"""
import json, sys, collections

path = sys.argv[1] if len(sys.argv) > 1 else r'c:\Users\yukic\local_llm_werewolf\logs\game_22754212a062.private.jsonl'
rows = [json.loads(l) for l in open(path, encoding='utf-8')]
sel = [r for r in rows if r.get('type') == 'discussion_context_selected']
comp = [r for r in rows if r.get('type') == 'llm_action_completed']
metrics = next((r.get('ledger_metrics') for r in rows if r.get('type') == 'game_result'), {})

print(f'log: {path}')
print('evidence budget distribution:', collections.Counter(r.get('evidence_char_budget') for r in sel))
kept = sum(len(r.get('selected_event_indices') or []) for r in sel)
total = sum(r.get('transcript_event_count') or 0 for r in sel)
chars = sum(r.get('evidence_char_count') or 0 for r in sel)
protected = sum(
    sum(1 for v in (r.get('selected_reasons') or {}).values() if 'recent_protection' in v) for r in sel
)
print(f'evidence: kept {kept}/{total} speeches, {chars} chars total (avg {round(chars / max(1, len(sel)))}), protected kept={protected}')
print(f'retry_rate={metrics.get("retry_rate")} fallback_rate={metrics.get("fallback_rate")} '
      f'body_preservation={metrics.get("discussion_body_preservation_rate")} '
      f'duplicate_claim_rate={metrics.get("duplicate_claim_rate")} secret_leaks={metrics.get("secret_leaks_detected")}')
winner = next((r.get('winner') for r in rows if r.get('type') == 'game_result'), None)
print(f'winner={winner} days={next((r.get("day") for r in rows if r.get("type") == "game_result"), None)}')

sat = 0
for c in comp:
    inp, out, eff = c.get('input_token_count'), c.get('output_token_count'), c.get('effective_num_ctx')
    if inp and eff:
        usage = (inp + (out or 0)) / eff
        if usage >= 0.92:
            sat += 1
print(f'requests using >=92% of assigned context: {sat}/{len(comp)}')
print('failure classes:', collections.Counter(c.get('failure_reason') for c in comp if c.get('failure_reason')))
