import json

LOG = r'c:\Users\yukic\local_llm_werewolf\logs\game_22754212a062.private.jsonl'
rows = [json.loads(l) for l in open(LOG, encoding='utf-8')]
gr = [r for r in rows if r.get('type') == 'game_result'][0]
memos = gr['reasoning_memos']
print('per-player memo: (history_len, history_chars, latest_chars, json_chars)')
for name, m in memos.items():
    hist = m.get('model_memo_history', [])
    payload = json.dumps({
        'confirmed_public_facts': m.get('confirmed_facts', []),
        'unverified_public_claims': m.get('public_claims', []),
        'suspicion_ranking_from_public_votes': m.get('suspicion_ranking', []),
        'unanswered_questions': m.get('unresolved_questions', []),
        'next_targets_to_check': m.get('next_targets', []),
        'updates_since_previous_turn': m.get('updates', []),
        'current_round_evidence': m.get('current_round_evidence', []),
    }, ensure_ascii=False)
    alerts = json.dumps(m.get('integrity_alerts', []), ensure_ascii=False)
    priv = json.dumps(m.get('private_role_facts', []), ensure_ascii=False)
    print(f'  {name}: hist={len(hist)} hist_chars={sum(len(x) for x in hist)} latest_chars={len(m.get("model_written_memo",""))} payload_chars={len(payload)} alerts_chars={len(alerts)} priv_chars={len(priv)}')
