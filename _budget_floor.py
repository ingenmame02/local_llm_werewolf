import json, collections

LOG = r'c:\Users\yukic\local_llm_werewolf\logs\game_22754212a062.private.jsonl'
rows = [json.loads(l) for l in open(LOG, encoding='utf-8')]
sel = [r for r in rows if r.get('type') == 'discussion_context_selected']
print('evidence_char_budget applied:', collections.Counter(r.get('evidence_char_budget') for r in sel))
print('transcript events -> kept events:')
for r in sel:
    print(f"  d{r['day']} {r.get('player')}: total={r['transcript_event_count']} kept={len(r.get('selected_event_indices') or [])} dropped={len(r.get('evidence_dropped_indices') or [])} ev_chars={r.get('evidence_char_count')}")
