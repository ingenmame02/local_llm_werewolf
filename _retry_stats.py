import json, statistics

LOG = r'c:\Users\yukic\local_llm_werewolf\logs\game_22754212a062.private.jsonl'
rows = [json.loads(l) for l in open(LOG, encoding='utf-8')]
speeches = [r for r in rows if r.get('type') == 'discussion_message']
comps = [r for r in rows if r.get('type') == 'llm_action_completed']
metrics = [r for r in rows if r.get('type') == 'game_result'][0]['ledger_metrics']
print('ledger_metrics:', json.dumps(metrics, ensure_ascii=False))

short = 0
retry_sourced = 0
for s in speeches:
    body = s.get('content') or ''
    if len(body) < 120:
        short += 1
    src = [c for c in comps if c.get('player') == s.get('speaker') and c.get('day') == s.get('day') and c.get('sequence', 0) < s.get('sequence', 0)]
    if src:
        last = src[-1]
        if last.get('action') == 'discussion_safety_retry' or (last.get('total_retries') or 0) > 0:
            retry_sourced += 1
print(f'published discussion messages={len(speeches)} | bodies <120 chars={short} | sourced from retry/safety_retry={retry_sourced}')

print('\nshort bodies:')
for s in speeches:
    body = s.get('content') or ''
    if len(body) < 140:
        print(f"  d{s.get('day')} {s.get('speaker')} ({len(body)} chars): {body[:120]}")
