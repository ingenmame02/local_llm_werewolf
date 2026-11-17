import json

LOG = r'c:\Users\yukic\local_llm_werewolf\logs\game_22754212a062.jsonl'
rows = [json.loads(l) for l in open(LOG, encoding='utf-8')]
msgs = [r for r in rows if r.get('type') == 'discussion_message']
for m in msgs:
    body = (m.get('content') or '').replace('\n', ' ')
    print(f"d{m.get('day')} t{m.get('discussion_turn')} {m.get('speaker')}: {body[:160]}")
