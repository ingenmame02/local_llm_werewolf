import json
import re
from collections import Counter
from pathlib import Path

LOG = Path(r'c:\Users\yukic\local_llm_werewolf\logs\game_e2991441d6f0.jsonl')
events = [json.loads(line) for line in LOG.open(encoding='utf-8')]
print('total events', len(events))
for kind, n in Counter(e.get('event') for e in events).most_common():
    print(kind, n)

print('\n=== failure / repair / fallback events ===')
for e in events:
    ev = e.get('event') or ''
    if 'fail' in ev or 'fallback' in ev or 'repair' in ev or 'retry' in ev:
        print(ev, '|', e.get('action'), e.get('actor'), 'day', e.get('day'),
              '| reason', e.get('reason'), '| fallback', e.get('fallback'),
              '| target', e.get('chosen_target') or e.get('target') or e.get('parsed_target'),
              '| will_retry', e.get('will_retry'))

print('\n=== raw responses containing template/placeholder text ===')
ph = re.compile(r'Player\s+X|プレイヤー\s*X|人狼\s*/\s*人間|werewolf\s*/\s*human|対象名|プレースホルダ', re.IGNORECASE)
for e in events:
    for key, raw in e.items():
        if isinstance(raw, str) and raw and ph.search(raw):
            print('---', e.get('event'), e.get('action'), e.get('actor'), 'day', e.get('day'), key)
            print(raw[:300])
            break
