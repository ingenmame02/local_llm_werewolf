import json

p = r'c:\Users\yukic\local_llm_werewolf\logs\game_e2991441d6f0.jsonl'
events = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
for e in events:
    t = e['type']
    if t == 'discussion_message' and e.get('speaker') == 'Player 3':
        print(f"--- P3 day {e.get('day')} turn {e.get('discussion_turn')} seq {e.get('sequence')}")
        print((e.get('content') or '')[:500])
    elif t in ('claim_role_published', 'seer_claim', 'seer_result_claim', 'seer_result_claim_missing', 'duplicate_claim_suppressed'):
        print(f"*** {t} day {e.get('day')} seq {e.get('sequence')}", json.dumps({k: v for k, v in e.items() if k not in ('type', 'timestamp', 'game_id', 'sequence')}, ensure_ascii=False)[:300])
