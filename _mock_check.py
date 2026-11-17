import json, collections, glob

path = sorted(glob.glob(r'c:\Users\yukic\local_llm_werewolf\_analysis\mock-p0\*.private.jsonl'))[-1]
rows = [json.loads(l) for l in open(path, encoding='utf-8')]
sel = [r for r in rows if r.get('type') == 'discussion_context_selected']
print('file:', path, 'selections:', len(sel))
print('budget distribution:', collections.Counter(r.get('evidence_char_budget') for r in sel))
for r in sel:
    reasons = r.get('selected_reasons') or {}
    protected = sum(1 for v in reasons.values() if 'recent_protection' in v)
    print(f"  d{r['day']} {r.get('player')}: events={r['transcript_event_count']} kept={len(r.get('selected_event_indices') or [])}"
          f" protected_kept={protected} ev_chars={r.get('evidence_char_count')} budget={r.get('evidence_char_budget')}"
          f" dropped={len(r.get('evidence_dropped_indices') or [])}")
reqs = [r for r in rows if r.get('type') == 'llm_request_configuration']
print('num_ctx sources:', collections.Counter(r.get('num_ctx_source') for r in reqs))
print('raw chars avg:', round(sum(r.get('raw_prompt_char_count', 0) for r in reqs) / max(1, len(reqs))))
