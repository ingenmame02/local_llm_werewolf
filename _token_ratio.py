import json, math

LOG = r'c:\Users\yukic\local_llm_werewolf\logs\game_22754212a062.private.jsonl'
rows = [json.loads(l) for l in open(LOG, encoding='utf-8')]
cfgs = [r for r in rows if r.get('type') == 'llm_request_configuration']
comp = [r for r in rows if r.get('type') == 'llm_action_completed']

ratios = []
for c in comp:
    inp = c.get('input_token_count')
    if not inp:
        continue
    cand = [
        x for x in cfgs
        if x.get('player') == c.get('player')
        and x.get('day') == c.get('day')
        and x.get('action') == c.get('action')
        and x.get('sequence', 0) <= c.get('sequence', 0) + 1
    ]
    if not cand:
        continue
    raw = cand[-1].get('raw_prompt_char_count')
    if raw:
        ratios.append((c.get('player'), c.get('action'), raw, inp, inp / raw))

vals = [r[4] for r in ratios]
print(f'measured prompt_tok/raw_chars: min={min(vals):.2f} max={max(vals):.2f} avg={sum(vals)/len(vals):.2f} n={len(vals)}')
print('estimate factor used in code: 0.8 tokens/char')
for r in ratios[:12]:
    print(f'{r[0]} {r[1]} raw={r[2]} prompt_tok={r[3]} ratio={r[4]:.2f}')
