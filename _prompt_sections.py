import json, math, sys

sys.path.insert(0, r'c:\Users\yukic\local_llm_werewolf\src')

from werewolf.models import (
    GameState,
    Player,
    Role,
    Message,
    PublicRoleClaim,
    PublicSeerClaim,
    LedgerFact,
    FactLedger,
    ReasoningMemo,
    PlayerActionMemory,
)
from werewolf.prompts import PromptBuilder
from werewolf.llm_actions import CONTEXT_TOKENS_PER_CHAR, CONTEXT_MARGIN_TOKENS

LOG = r'c:\Users\yukic\local_llm_werewolf\logs\game_22754212a062.private.jsonl'
CFG = r'c:\Users\yukic\local_llm_werewolf\config\players_free.json'

rows = [json.loads(l) for l in open(LOG, encoding='utf-8')]
gr = [r for r in rows if r.get('type') == 'game_result'][0]
cfg_players = json.load(open(CFG, encoding='utf-8'))['players']
cfg_by_name = {p['name']: p for p in cfg_players}

players = []
for p in gr['players']:
    c = cfg_by_name.get(p['name'], {})
    players.append(
        Player(
            name=p['name'],
            model=p['model'],
            role=Role(p['role']),
            is_alive=bool(p['is_alive']),
            partner=p.get('partner'),
            num_ctx=c.get('num_ctx', 12288),
            max_num_ctx=c.get('max_num_ctx'),
            num_predict=c.get('num_predict'),
            num_predict_by_action=c.get('num_predict_by_action'),
            personality_profile='free',
            response_language='日本語',
            claim_coaching=bool(c.get('claim_coaching', False)),
            temperature=c.get('temperature'),
            top_p=c.get('top_p'),
            top_k=c.get('top_k'),
        )
    )

fact_ledger = FactLedger(
    confirmed_facts=[LedgerFact(**f) for f in gr['fact_ledger']['confirmed_facts']],
    public_claims=[PublicRoleClaim(**c) for c in gr['fact_ledger']['public_claims']],
    model_inferences=[LedgerFact(**f) for f in gr['fact_ledger']['model_inferences']],
)

memos = {
    name: ReasoningMemo(**{k: v for k, v in m.items() if k in ReasoningMemo.__dataclass_fields__})
    for name, m in gr['reasoning_memos'].items()
}
memos = {name: ReasoningMemo(
    confirmed_facts=m.get('confirmed_facts', []),
    public_claims=m.get('public_claims', []),
    suspicion_ranking=m.get('suspicion_ranking', []),
    unresolved_questions=m.get('unresolved_questions', []),
    next_targets=m.get('next_targets', []),
    private_role_facts=m.get('private_role_facts', []),
    current_round_evidence=m.get('current_round_evidence', []),
    integrity_alerts=m.get('integrity_alerts', []),
    updates=m.get('updates', []),
    model_written_memo=m.get('model_written_memo', ''),
    model_memo_history=m.get('model_memo_history', []),
) for name, m in gr['reasoning_memos'].items()}

action_memories = {
    name: [PlayerActionMemory(**{k: v for k, v in m.items() if k in PlayerActionMemory.__dataclass_fields__})
           for m in msgs]
    for name, msgs in gr['player_action_memories'].items()
}

state = GameState(
    players=players,
    day=gr['day'],
    discussion_turn=0,
    conversation_log=[Message(**m) for m in gr['conversation_log']],
    wolf_chat_log=[Message(**m) for m in gr['wolf_chat_log']],
    seer_results=dict(gr['seer_results']),
    vote_history=[],
    public_role_claims=[PublicRoleClaim(**c) for c in gr['public_role_claims']],
    seer_claims=[PublicSeerClaim(**c) for c in gr['seer_claims']],
    fact_ledger=fact_ledger,
    reasoning_memos=memos,
    reasoning_memory_enabled=True,
    player_action_memories=action_memories,
    question_hint_mode='none',
)

print('=== reconstructed state ===')
print('conversation messages:', len(state.conversation_log))
print('fact_ledger confirmed:', len(state.fact_ledger.confirmed_facts),
      'claims:', len(state.fact_ledger.public_claims),
      'inferences:', len(state.fact_ledger.model_inferences))
print('memos:', len(state.reasoning_memos), 'action memories:', sum(len(v) for v in state.player_action_memories.values()))

MARKERS = [
    "[SYSTEM_RULES]",
    "[OUTPUT_LANGUAGE]",
    "[FACT_LEDGER]",
    "[CLAIM_LEDGER]",
    "[SELF_ACTION_MEMORY",
    "[PRIVATE_SEER_RESULT",
    "[CONTINUITY_MEMO:",
    "[YOUR_PREVIOUS_REASONING_MEMO:",
    "[Your Speeches Today",
    "[Full Current Round Dialogue]",
    "[CONVERSATION_EVIDENCE:",
    "[CURRENT_TASK]",
    "[秘匿情報の扱い",
    "[Instructions]",
    "[REASONING_MEMO_GUIDANCE]",
    "[Discussion Guidelines",
    "[Final Output Checklist]",
]


def split_sections(user_prompt: str) -> list[tuple[str, int]]:
    positions = []
    for marker in MARKERS:
        idx = user_prompt.find(marker)
        if idx >= 0:
            positions.append((marker, idx))
    positions.sort(key=lambda x: x[1])
    sections = []
    if positions and positions[0][1] > 0:
        sections.append(("(head)", positions[0][1]))
    for i, (marker, idx) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(user_prompt)
        sections.append((marker, end - idx))
    return sections


def measure(day: int, turn: int, actor_name: str, history_len: int, budget: int | None) -> dict:
    actor = next(p for p in state.players if p.name == actor_name)
    history = tuple(m for m in state.conversation_log if m.speaker != "system")[:history_len]
    state.day = day
    state.discussion_turn = turn
    alive = tuple(p.name for p in state.players if p.is_alive)
    ctx = PromptBuilder.build_discussion_prompt(
        actor,
        alive,
        history,
        day=day,
        state=state,
        include_claim=True,
        context_strategy="hybrid_retrieval",
        evidence_char_budget=budget,
        question_hint_mode="none",
    )
    raw = len(ctx.system_prompt) + len(ctx.user_prompt)
    ev_chars = int(ctx.diagnostics.get("evidence_char_count") or 0)
    return {
        "day": day,
        "turn": turn,
        "actor": actor_name,
        "system_chars": len(ctx.system_prompt),
        "user_chars": len(ctx.user_prompt),
        "raw_chars": raw,
        "evidence_chars": ev_chars,
        "selected_events": len(ctx.diagnostics.get("selected_event_indices") or []),
        "transcript_events": ctx.diagnostics.get("transcript_event_count"),
        "budget": budget,
        "sections": split_sections(ctx.user_prompt),
    }


def report(m: dict) -> None:
    print(f"\n--- Day {m['day']} turn {m['turn']} {m['actor']} ---")
    print(
        f"system={m['system_chars']} user={m['user_chars']} raw={m['raw_chars']} chars"
        f" | est_tokens={math.ceil(m['raw_chars'] * CONTEXT_TOKENS_PER_CHAR)}"
        f" | evidence={m['evidence_chars']} chars"
        f" ({m['selected_events']}/{m['transcript_events']} events kept)"
    )
    for name, size in m["sections"]:
        print(f"  {size:6d} chars {size / m['user_chars'] * 100:5.1f}%  {name}")


sel = [r for r in rows if r.get("type") == "discussion_context_selected"]
targets = [ev for ev in sel if ev["day"] >= 2 and ev["player"] in ("Player 2", "Player 5")]
for ev in targets:
    speeches_before = [
        r
        for r in rows
        if r.get("type") == "discussion_message"
        and r.get("day") == ev["day"]
        and r.get("sequence", 0) < ev["sequence"]
    ]
    report(measure(ev["day"], len(speeches_before), ev["player"], ev["transcript_event_count"], ev["evidence_char_budget"]))


# ---- log-based quantification: where the context budget went ----
cfg = [r for r in rows if r.get("type") == "llm_request_configuration"]
comp = [r for r in rows if r.get("type") == "llm_action_completed"]
comp_by_key = {}
for c in comp:
    comp_by_key[(c.get("player"), c.get("day"), c.get("action"), c.get("sequence"))] = c

print("\n=== context budget breakdown per request (log) ===")
neg = 0
total = 0
for c in cfg:
    raw = c.get("raw_prompt_char_count") or 0
    ev = c.get("evidence_char_count")
    if ev is None:
        sel_ev = [
            r
            for r in sel
            if r.get("player") == c.get("player") and r.get("day") == c.get("day") and r.get("sequence", 0) < c.get("sequence", 0)
        ]
        ev = sel_ev[-1].get("evidence_char_count") if sel_ev else 0
    gen = c.get("num_predict") or 512
    ctx = c.get("requested_num_ctx") or 12288
    fixed_tokens = math.ceil(max(0, raw - (ev or 0)) * CONTEXT_TOKENS_PER_CHAR)
    available = int(ctx) - int(gen) - CONTEXT_MARGIN_TOKENS
    budget_raw = int((available - fixed_tokens) / CONTEXT_TOKENS_PER_CHAR)
    if budget_raw < 600:
        neg += 1
    total += 1
    if c.get("action") in {"discussion", "vote", "wolf_discussion"} and total % 1 == 0:
        print(
            f"{c.get('player')} d{c.get('day')} {c.get('action')} raw={raw} ev={ev} gen={gen} "
            f"num_ctx={ctx} fixed_tok_est={fixed_tokens} available_input_tok={available} "
            f"budget_raw={budget_raw} -> applied={ev}"
        )
print(f"requests where evidence budget would go below the 600-char floor: {neg}/{total}")

print("\n=== measured prompt tokens vs assigned num_ctx ===")
ratios = []
saturated = 0
for c in comp:
    inp = c.get("input_token_count")
    out = c.get("output_token_count")
    eff = c.get("effective_num_ctx")
    if inp and eff:
        if out and inp + out >= eff * 0.92:
            saturated += 1
        print(f"{c.get('player')} d{c.get('day')} {c.get('action')} prompt_tok={inp} out_tok={out} ctx={eff} usage={(inp + (out or 0)) / eff:.2f}")
print(f"requests using >=92% of assigned context: {saturated}")

