import json
import re
from collections import Counter, defaultdict
from pathlib import Path

LOG = Path(r'c:\Users\yukic\local_llm_werewolf\logs\game_e2991441d6f0.jsonl')
OUT = Path(r'c:\Users\yukic\local_llm_werewolf\_analysis\log_scan_out.txt')
events = [json.loads(line) for line in LOG.open(encoding='utf-8')]
lines: list[str] = []

# 入力トーク数の実測（プレイヤー・行動別）
toks = defaultdict(list)
for e in events:
    if e.get('type') == 'llm_action_completed' and isinstance(e.get('input_token_count'), int):
        toks[(e.get('player'), e.get('action'))].append(e['input_token_count'])
lines.append('=== input_token_count (real) by player/action ===')
for key, values in sorted(toks.items()):
    lines.append(f'{key[0]} {key[1]}: n={len(values)} max={max(values)} avg={sum(values)/len(values):.0f}')
all_req = [e.get('input_token_count') for e in events
           if e.get('type') == 'llm_action_completed' and isinstance(e.get('input_token_count'), int)]
lines.append(f'ALL: n={len(all_req)} max={max(all_req)} avg={sum(all_req)/len(all_req):.0f} over4000={sum(1 for t in all_req if t > 4000)}')

lines.append('\n=== llm_action_parse_failed ===')
for e in events:
    if e.get('type') == 'llm_action_parse_failed':
        lines.append(f"{e.get('player')} {e.get('action')} day{e.get('day')} reason={e.get('reason')} "
                     f"class={e.get('failure_class')} will_retry={e.get('will_retry')} target={e.get('parsed_target')}")

lines.append('\n=== fallback / repair / leak events ===')
for e in events:
    t = e.get('type')
    if t in {'ledger_repair', 'discussion_integrity_repaired', 'secret_role_leak_detected',
             'seer_result_claim_missing', 'duplicate_claim_suppressed', 'thinking_budget_exhausted',
             'claim_tool_unavailable'}:
        lines.append(f'{t}: {e.get("player")} {e.get("action")} day{e.get("day")} ' +
                     json.dumps({k: v for k, v in e.items() if k in
                                 {'reason', 'fatal_safety_fallback', 'fallback_used', 'body_preserved',
                                  'regeneration_avoided', 'content_preserved', 'repair_applied', 'raw'}},
                                 ensure_ascii=False)[:300])

lines.append('\n=== completed actions with fallback ===')
for e in events:
    if e.get('type') == 'llm_action_completed' and (e.get('fallback_used') or e.get('fatal_safety_fallback')):
        lines.append(f"{e.get('player')} {e.get('action')} day{e.get('day')} fallback_used={e.get('fallback_used')} "
                     f"fatal={e.get('fatal_safety_fallback')} retries={e.get('total_retries')} "
                     f"severity={e.get('validation_severity')} classes={e.get('failure_classes')}")

lines.append('\n=== Player 3 responses (all, truncated) ===')
ph = re.compile(r'Player\s+X|プレイヤー\s*X|人狼\s*/\s*人間|werewolf\s*/\s*human', re.IGNORECASE)
for e in events:
    if e.get('type') == 'discussion_message' and e.get('speaker') == 'Player 3':
        body = e.get('content') or ''
        lines.append(f"day{e.get('day')} turn{e.get('turn')} placeholder={bool(ph.search(body))}: {body[:260]}")

OUT.write_text('\n'.join(lines), encoding='utf-8')
print('written', len(lines), 'lines')
