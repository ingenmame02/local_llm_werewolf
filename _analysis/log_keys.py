import json
from collections import Counter
from pathlib import Path

LOG = Path(r'c:\Users\yukic\local_llm_werewolf\logs\game_e2991441d6f0.jsonl')
events = [json.loads(line) for line in LOG.open(encoding='utf-8')]
out = Path(r'c:\Users\yukic\local_llm_werewolf\_analysis\log_scan_out.txt')
lines = [f'total events {len(events)}']
for kind, n in Counter(e.get('type') or e.get('event_type') or e.get('kind') for e in events).most_common():
    lines.append(f'{kind} {n}')
lines.append('\nkeys of first 3 events:')
for e in events[:3]:
    lines.append(json.dumps(sorted(e.keys()), ensure_ascii=False))
lines.append('\nsample event:')
lines.append(json.dumps(events[12], ensure_ascii=False)[:1500])
out.write_text('\n'.join(lines), encoding='utf-8')
print('written')
