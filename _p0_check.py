import json, math, sys

sys.path.insert(0, r'c:\Users\yukic\local_llm_werewolf\src')
from werewolf.models import (
    GameState, Player, Role, Message, PublicRoleClaim, PublicSeerClaim,
    LedgerFact, FactLedger, ReasoningMemo, PlayerActionMemory,
)
from werewolf.prompts import PromptBuilder, RECENT_PROTECTED_SPEECHES
from werewolf.llm_actions import (
    CONTEXT_TOKENS_PER_CHAR, CONTEXT_MARGIN_TOKENS, CONTEXT_EVIDENCE_MIN_CHARS,
    CONTEXT_EVIDENCE_MAX_CHARS, fit_dynamic_context,
)

LOG = r'c:\Users\yukic\local_llm_werewolf\logs\game_22754212a062.private.jsonl'
CFG = r'c:\Users\yukic\local_llm_werewolf\config\players_free.json'
rows = [json.loads(l) for l in open(LOG, encoding='utf-8')]
gr = [r for r in rows if r.get('type') == 'game_result'][0]
cfg_players = {p['name']: p for p in json.load(open(CFG, encoding='utf-8'))['players']}
MAX_DAY = gr['day']


def by_day(entries):
    out = []
    for e in entries:
        if isinstance(e, str) and e.startswith('[fact:'):
            try:
                if int(e.split(':')[1]) > MAX_DAY:
                    continue
            except ValueError:
                pass
        out.append(e)
    return out


players = [
    Player(
        name=p['name'], model=p['model'], role=Role(p['role']), is_alive=bool(p['is_alive']),
        partner=p.get('partner'), num_ctx=cfg_players.get(p['name'], {}).get('num_ctx', 12288),
        max_num_ctx=cfg_players.get(p['name'], {}).get('max_num_ctx'),
        num_predict=cfg_players.get(p['name'], {}).get('num_predict'),
        num_predict_by_action=cfg_players.get(p['name'], {}).get('num_predict_by_action'),
        personality_profile='free', response_language='日本語',
    ) for p in gr['players']
]

state = GameState(
    players=players,
    day=MAX_DAY,
    conversation_log=[Message(**m) for m in gr['conversation_log']],
    wolf_chat_log=[Message(**m) for m in gr['wolf_chat_log']],
    seer_results=dict(gr['seer_results']),
    fact_ledger=FactLedger(
        confirmed_facts=[LedgerFact(**f) for f in gr['fact_ledger']['confirmed_facts'] if f['day'] <= MAX_DAY],
        public_claims=[PublicRoleClaim(**c) for c in gr['fact_ledger']['public_claims']],
        model_inferences=[LedgerFact(**f) for f in gr['fact_ledger']['model_inferences']],
    ),
    public_role_claims=[PublicRoleClaim(**c) for c in gr['public_role_claims']],
    seer_claims=[PublicSeerClaim(**c) for c in gr['seer_claims']],
    reasoning_memos={
        name: ReasoningMemo(
            confirmed_facts=by_day(m.get('confirmed_facts', [])), public_claims=by_day(m.get('public_claims', [])),
            suspicion_ranking=m.get('suspicion_ranking', []), unresolved_questions=by_day(m.get('unresolved_questions', [])),
            next_targets=m.get('next_targets', []), private_role_facts=by_day(m.get('private_role_facts', [])),
            current_round_evidence=by_day(m.get('current_round_evidence', [])), integrity_alerts=m.get('integrity_alerts', []),
            updates=by_day(m.get('updates', [])), model_written_memo=m.get('model_written_memo', ''),
            model_memo_history=m.get('model_memo_history', []),
        ) for name, m in gr['reasoning_memos'].items()
    },
    player_action_memories={
        name: [PlayerActionMemory(**{k: v for k, v in mm.items() if k in PlayerActionMemory.__dataclass_fields__}) for mm in msgs]
        for name, msgs in gr['player_action_memories'].items()
    },
    question_hint_mode='none',
)
deaths = {}
for r in rows:
    if r.get('type') == 'morning_casualty':
        name = r.get('player') or r.get('victim') or r.get('target')
        if name:
            deaths[name] = r.get('day')
    elif r.get('type') == 'voting_result':
        name = r.get('target') or r.get('exiled')
        if name and name != 'None':
            deaths[name] = r.get('day')
print('deaths by day:', deaths)


def state_for_day(day):
    """その日の時点相当の台帳・メモ・生存状態で GameState を作り直す（実測サイズに近げる）。"""
    alive_players = [
        Player(
            name=p.name, model=p.model, role=p.role, partner=p.partner,
            is_alive=deaths.get(p.name, 10_000) > day,
            num_ctx=p.num_ctx, max_num_ctx=p.max_num_ctx, num_predict=p.num_predict,
            num_predict_by_action=p.num_predict_by_action,
            personality_profile='free', response_language='日本語',
        ) for p in players
    ]
    return GameState(
        players=alive_players,
        day=day,
        conversation_log=[Message(**m) for m in gr['conversation_log']],
        wolf_chat_log=[Message(**m) for m in gr['wolf_chat_log']],
        seer_results=dict(gr['seer_results']),
        fact_ledger=FactLedger(
            confirmed_facts=[LedgerFact(**f) for f in gr['fact_ledger']['confirmed_facts'] if f['day'] <= day],
            public_claims=[PublicRoleClaim(**c) for c in gr['fact_ledger']['public_claims']],
            model_inferences=[LedgerFact(**f) for f in gr['fact_ledger']['model_inferences']],
        ),
        public_role_claims=[PublicRoleClaim(**c) for c in gr['public_role_claims'] if c['day'] <= day],
        seer_claims=[PublicSeerClaim(**c) for c in gr['seer_claims'] if c['day'] <= day],
        reasoning_memos={
            name: ReasoningMemo(
                confirmed_facts=[e for e in by_day(m.get('confirmed_facts', []))],
                public_claims=[e for e in by_day(m.get('public_claims', []))],
                suspicion_ranking=m.get('suspicion_ranking', []),
                unresolved_questions=by_day(m.get('unresolved_questions', [])),
                next_targets=m.get('next_targets', []),
                private_role_facts=by_day(m.get('private_role_facts', [])),
                current_round_evidence=by_day(m.get('current_round_evidence', [])),
                integrity_alerts=m.get('integrity_alerts', []),
                updates=by_day(m.get('updates', [])),
            ) for name, m in gr['reasoning_memos'].items()
        },
        player_action_memories={
            name: [PlayerActionMemory(**{k: v for k, v in mm.items() if k in PlayerActionMemory.__dataclass_fields__})
                   for mm in msgs if mm.get('day', 1) <= day]
            for name, msgs in gr['player_action_memories'].items()
        },
        question_hint_mode='none',
    )


sel = [r for r in rows if r.get('type') == 'discussion_context_selected']
print(f'constants: coef={CONTEXT_TOKENS_PER_CHAR} floor={CONTEXT_EVIDENCE_MIN_CHARS} max={CONTEXT_EVIDENCE_MAX_CHARS} protected={RECENT_PROTECTED_SPEECHES}')
violations = 0
for ev in sel:
    day = ev['day']
    st = state_for_day(day)
    actor = next(p for p in st.players if p.name == ev['player'])
    history = tuple(m for m in st.conversation_log if m.speaker != 'system')[:ev['transcript_event_count']]
    alive = tuple(p.name for p in st.players if p.is_alive)
    gen = (actor.num_predict_by_action or {}).get('discussion', 2560)
    alloc = actor.num_ctx or 12288
    base = PromptBuilder.build_discussion_prompt(
        actor, alive, history, day=day, state=st, include_claim=True,
        context_strategy='hybrid_retrieval', question_hint_mode='none',
    )
    raw = len(base.system_prompt) + len(base.user_prompt)
    ev_full = int(base.diagnostics.get('evidence_char_count') or 0)
    fixed = math.ceil(max(0, raw - ev_full) * CONTEXT_TOKENS_PER_CHAR)
    budget = int((alloc - gen - CONTEXT_MARGIN_TOKENS - fixed) / CONTEXT_TOKENS_PER_CHAR)
    budget = min(CONTEXT_EVIDENCE_MAX_CHARS, max(CONTEXT_EVIDENCE_MIN_CHARS, budget))
    total = ev['transcript_event_count']
    protected = set(range(max(1, total - RECENT_PROTECTED_SPEECHES + 1), total + 1))
    if ev_full > budget:
        rebuilt = PromptBuilder.build_discussion_prompt(
            actor, alive, history, day=day, state=st, include_claim=True,
            context_strategy='hybrid_retrieval', evidence_char_budget=budget, question_hint_mode='none',
        )
        kept = set(rebuilt.diagnostics['selected_event_indices'])
        lost = protected - kept
        violations += 1 if lost else 0
        print(f"  d{day} {ev['player']}: raw={raw} full={ev_full} budget={budget} kept={len(kept)}/{total}"
              f" after={int(rebuilt.diagnostics['evidence_char_count'])} (old_applied={ev['evidence_char_count']})"
              f" lost_protected={sorted(lost)}")
    else:
        print(f"  d{day} {ev['player']}: raw={raw} full={ev_full} budget={budget} -> no trimming (all kept)")
print(f'\nprotected-window violations: {violations}')

print('\nnum_ctx fit sanity (coef=0.5 vs measured real ratio max 0.39):')
worst_headroom = 10**9
for ev in sel[-10:]:
    st = state_for_day(ev['day'])
    actor = next(p for p in st.players if p.name == ev['player'])
    if not actor.is_alive:
        continue
    history = tuple(m for m in st.conversation_log if m.speaker != 'system')[:ev['transcript_event_count']]
    ctx = PromptBuilder.build_discussion_prompt(
        actor, tuple(p.name for p in st.players if p.is_alive), history, day=ev['day'], state=st,
        include_claim=True, context_strategy='hybrid_retrieval', question_hint_mode='none',
    )
    raw = len(ctx.system_prompt) + len(ctx.user_prompt)
    gen = (actor.num_predict_by_action or {}).get('discussion', 2560)
    options = {'num_ctx': actor.num_ctx or 12288, 'num_predict': gen}
    fitted, needed, changed = fit_dynamic_context(
        options, [{'content': ctx.system_prompt}, {'content': ctx.user_prompt}], ceiling=actor.max_num_ctx
    )
    real_max = math.ceil(raw * 0.39)
    headroom = fitted['num_ctx'] - real_max - gen
    worst_headroom = min(worst_headroom, headroom)
    print(f"  {ev['player']} d{ev['day']}: raw={raw} est_needed={needed} ctx={fitted['num_ctx']}"
          f" (fit_changed={changed}) real_tok_upper={real_max} headroom_after_generation={headroom}")
print(f'worst headroom (negative = truncation risk): {worst_headroom}')

