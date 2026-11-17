import json, collections

LOG = r'c:\Users\yukic\local_llm_werewolf\logs\game_22754212a062.private.jsonl'
rows = [json.loads(l) for l in open(LOG, encoding='utf-8')]
cfg = [r for r in rows if r.get('type') == 'llm_request_configuration']
per_day = collections.defaultdict(list)
for c in cfg:
    per_day[c.get('day')].append((c.get('action'), c.get('raw_prompt_char_count'), c.get('user_prompt_char_count'), c.get('system_prompt_char_count')))
for day, vals in sorted(per_day.items()):
    raws = [v[1] for v in vals]
    users = [v[2] for v in vals]
    sys_ = [v[3] for v in vals]
    print(f'Day {day}: requests={len(vals)} raw avg={round(sum(raws)/len(raws))} max={max(raws)} | user avg={round(sum(users)/len(users))} max={max(users)} | system avg={round(sum(sys_)/len(sys_))}')
    print('   ', collections.Counter(v[0] for v in vals))
