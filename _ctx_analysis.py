import json, collections, sys

path = sys.argv[1] if len(sys.argv) > 1 else r'c:\Users\yukic\local_llm_werewolf\logs\game_22754212a062.private.jsonl'
rows = [json.loads(l) for l in open(path, encoding='utf-8')]
cfg = [r for r in rows if r.get('type') == 'llm_request_configuration']
comp = [r for r in rows if r.get('type') == 'llm_action_completed']

byact = collections.defaultdict(list)
for c in cfg:
    t = c.get('context_needed_tokens') or 0
    cx = c.get('requested_num_ctx') or 0
    byact[c.get('action')].append((t, cx))
print('=== request config ===')
for a, v in sorted(byact.items()):
    n = len(v)
    ov = sum(1 for t, cx in v if t > cx)
    print(f'{a}: n={n} overflow={ov} avg_needed={round(sum(t for t, _ in v)/n)} req_ctx={sorted({cx for _, cx in v})}')
print('req_reason:', collections.Counter(c.get('request_reason') for c in cfg))
print('num_ctx_source:', collections.Counter(c.get('num_ctx_source') for c in cfg))
print('thinking_enabled:', collections.Counter(c.get('thinking_enabled') for c in cfg))
print('num_predict:', collections.Counter(c.get('num_predict') for c in cfg))

print('=== completed ===')
for c in comp:
    print(f"{c.get('player')} {c.get('action')} d{c.get('day')} in={c.get('input_token_count')} out={c.get('output_token_count')} eff={c.get('effective_num_ctx')} think={c.get('thinking_enabled')} retries={c.get('total_retries')} fb={c.get('fallback_used')} done={c.get('done_reason')} fail={c.get('failure_reason')}")
