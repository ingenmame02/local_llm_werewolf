import json, math, sys, collections

sys.path.insert(0, r'c:\Users\yukic\local_llm_werewolf\src')
from werewolf.models import GameState, Player, Role, Message, PublicRoleClaim, PublicSeerClaim, LedgerFact, FactLedger, ReasoningMemo, PlayerActionMemory
from werewolf.prompts import PromptBuilder

LOG = r'c:\Users\yukic\local_llm_werewolf\logs\game_22754212a062.private.jsonl'
CFG = r'c:\Users\yukic\local_llm_werewolf\config\players_free.json'
rows = [json.loads(l) for l in open(LOG, encoding='utf-8')]
gr = [r for r in rows if r.get('type') == 'game_result'][0]
cfg_players = {p['name']: p for p in json.load(open(CFG, encoding='utf-8'))['players']}

players = []
for p in gr['players']:
    c = cfg_players.get(p['name'], {})
    players.append(Player(
        name=p['name'], model=p['model'], role=Role(p['role']), is_alive=bool(p['is_alive']),
        partner=p.get('partner'), num_ctx=c.get('num_ctx', 12288), max_num_ctx=c.get('max_num_ctx'),
        num_predict=c.get('num_predict'), num_predict_by_action=c.get('num_predict_by_action'),
        personality_profile='free', response_language='日本語',
        temperature=c.get('temperature'), top_p=c.get('top_p'), top_k=c.get('top_k'),
    ))

MAX_DAY = 2


def by_day(entries, prefix_pattern=None):
    """day <= MAX_DAY の項目だけ残す（[fact:<day>:...] 形式のメモを絞る）。"""
    out = []
    for e in entries:
        if isinstance(e, str) and e.startswith('[fact:'):
            parts = e.split(':')
            try:
                if int(parts[1]) > MAX_DAY:
                    continue
            except ValueError:
                pass
        out.append(e)
    return out


state = GameState(
    players=players,
    day=gr['day'],
    conversation_log=[Message(**m) for m in gr['conversation_log']],
    wolf_chat_log=[Message(**m) for m in gr['wolf_chat_log']],
    seer_results=dict(gr['seer_results']),
    fact_ledger=FactLedger(
        confirmed_facts=[LedgerFact(**f) for f in gr['fact_ledger']['confirmed_facts'] if f['day'] <= MAX_DAY],
        public_claims=[PublicRoleClaim(**c) for c in gr['fact_ledger']['public_claims'] if c['day'] <= MAX_DAY],
        model_inferences=[LedgerFact(**f) for f in gr['fact_ledger']['model_inferences'] if f['day'] <= MAX_DAY],
    ),
    reasoning_memos={
        name: ReasoningMemo(
            confirmed_facts=by_day(m.get('confirmed_facts', [])), public_claims=by_day(m.get('public_claims', [])),
            suspicion_ranking=m.get('suspicion_ranking', []), unresolved_questions=by_day(m.get('unresolved_questions', [])),
            next_targets=m.get('next_targets', []), private_role_facts=by_day(m.get('private_role_facts', [])),
            current_round_evidence=by_day(m.get('current_round_evidence', [])), integrity_alerts=[a for a in m.get('integrity_alerts', [])],
            updates=by_day(m.get('updates', [])), model_written_memo=m.get('model_written_memo', ''),
            model_memo_history=m.get('model_memo_history', []),
        ) for name, m in gr['reasoning_memos'].items()
    },
    public_role_claims=[PublicRoleClaim(**c) for c in gr['public_role_claims'] if c['day'] <= MAX_DAY],
    seer_claims=[PublicSeerClaim(**c) for c in gr['seer_claims'] if c['day'] <= MAX_DAY],
    player_action_memories={
        name: [PlayerActionMemory(**{k: v for k, v in m.items() if k in PlayerActionMemory.__dataclass_fields__})
               for m in msgs if m.get('day', 1) <= MAX_DAY]
        for name, msgs in gr['player_action_memories'].items()
    },
    question_hint_mode='none',
)

actor = next(p for p in state.players if p.name == 'Player 2')
history = tuple(m for m in state.conversation_log if m.speaker != 'system')[:23]
ctx = PromptBuilder.build_discussion_prompt(
    actor, tuple(p.name for p in state.players if p.is_alive), history, day=2, state=state,
    include_claim=True, context_strategy='hybrid_retrieval', evidence_char_budget=600,
    question_hint_mode='none',
)
full = ctx.system_prompt + '\n' + ctx.user_prompt
open(r'c:\Users\yukic\local_llm_werewolf\_prompt_dump.txt', 'w', encoding='utf-8').write(full)
print(f'system={len(ctx.system_prompt)} user={len(ctx.user_prompt)} total={len(full)} chars')

lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
counts = collections.Counter(lines)
dupes = [(n, line) for line, n in counts.items() if n > 1 and len(line) > 40]
dupes.sort(reverse=True)
print(f'\nduplicated lines (>1 occurrence, len>40): {len(dupes)} lines, wasted chars={sum((n-1)*len(l) for n,l in dupes)}')
for n, line in dupes[:12]:
    print(f'  x{n}  {line[:130]}')

phrases = [
    'first person',
    'third person',
    'never mention',
    'do not fabricate',
    'Not in record',
    'unverified',
    'never disclose',
    'Madman is divined',
    '狂人',
    'past games',
    'system settings',
    'never quote',
    'Do NOT',
    'CO/RESULT',
    'CO ブロック',
    '本文に写さない',
]
print('\nphrase occurrences across whole prompt:')
for ph in phrases:
    print(f'  {ph!r}: {full.count(ph)}')
