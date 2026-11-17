import sys
from pathlib import Path
from collections import Counter

SRC = Path(r'c:\Users\yukic\local_llm_werewolf\src')
sys.path.insert(0, str(SRC))

from werewolf.engine import GameEngine, UNVERIFIED_HYPOTHESIS_AS_FACT
from werewolf.llm_actions import OllamaActions
from werewolf.setup import create_game

CONFIG = r'c:\Users\yukic\local_llm_werewolf\config\players.json'

sys.path.insert(0, r'c:\Users\yukic\local_llm_werewolf\_analysis')
_spec_src = Path(r'c:\Users\yukic\local_llm_werewolf\_analysis\sim_game.py').read_text(encoding='utf-8')
# SimulatedClient だけ再定義（実機Ollamaへは一切送信しない・末尾実行は避ける）
_class_src = _spec_src.split('state = create_game')[0]
_ns = {'__file__': str(_spec_src)}
exec(_class_src, _ns)
SimulatedClient = _ns['SimulatedClient']

state = create_game(seed=20261115, config_path=CONFIG, response_language='日本語')
client = SimulatedClient(state)
events: list[dict[str, object]] = []
actions = OllamaActions(
    state,
    client,
    seed=20261115,
    options={'question_hint_mode': 'none', 'temperature': 0.7},
    discussion_claim_mode='legacy',
    event_callback=events.append,
)
engine = GameEngine(state, actions, seed=20261115, event_callback=events.append)
result = engine.run()

print('winner:', result.winner, '| day:', result.day, '| requests:', client.requests)
print('hypothesis metrics:', state.hypothesis_ledger.metrics())
print(
    'public hypotheses:',
    len(state.hypothesis_ledger.serialize_public()['records']),
    '| private-only:',
    len(state.hypothesis_ledger.serialize()['records'])
    - len(state.hypothesis_ledger.serialize_public()['records']),
)
print(
    'contradictions:',
    Counter(str(pair.kind) for pair in state.hypothesis_ledger.contradictions),
)
print(
    'integrity alerts:',
    Counter(str(alert.category) for alert in state.integrity_alerts),
)
print('unverified_hypothesis_as_fact alerts:', sum(
    1 for alert in state.integrity_alerts if alert.category == UNVERIFIED_HYPOTHESIS_AS_FACT
))
sel = [e for e in events if e.get('type') == 'discussion_context_selected']
print('context_selected events:', len(sel))
print('hypothesis ledger chars avg:', round(
    sum(int(e.get('hypothesis_ledger_chars') or 0) for e in sel) / max(1, len(sel)), 1
))
print('continuity memo chars avg:', round(
    sum(int(e.get('continuity_memo_chars') or 0) for e in sel) / max(1, len(sel)), 1
))
print('claim ledger chars avg:', round(
    sum(int(e.get('claim_ledger_chars') or 0) for e in sel) / max(1, len(sel)), 1
))
print('inference ledger chars avg:', round(
    sum(int(e.get('inference_ledger_chars') or 0) for e in sel) / max(1, len(sel)), 1
))
print('required evidence retention all 1.0:', sel and all(
    e.get('required_evidence_retention') == 1.0 for e in sel
))
print('memo reuses ids only:', all(
    entry.startswith('[') for memo in state.reasoning_memos.values()
    for entry in memo.confirmed_facts + memo.public_claims
))
print('dead-target hypotheses:', state.hypothesis_ledger.metrics()['hypothesis_status_counts']['historical'])
print('unresolved questions referencing dead players:', sum(
    1
    for memo in state.reasoning_memos.values()
    for entry in memo.unresolved_questions
    if any(name in entry for name in (p.name for p in state.players if not p.is_alive))
))
print('annotated as unanswerable:', sum(
    1
    for memo in state.reasoning_memos.values()
    for entry in memo.unresolved_questions
    if 'dead players cannot answer' in entry
))

# コンテキスト量の比較（TASK-047以前の描画 = 全文再掲 vs 現在のID参照+仮説要約）
import json
from werewolf.prompts import PromptBuilder

actor = next(p for p in state.players if p.is_alive)
ctx = PromptBuilder.build_discussion_prompt(
    actor,
    tuple(state.alive_players),
    tuple(state.conversation_log),
    day=state.day,
    state=state,
    include_claim=True,
    context_strategy='hybrid_retrieval',
    question_hint_mode='soft',
)
diag = ctx.diagnostics
new_total = (
    int(diag['claim_ledger_chars'])
    + int(diag['inference_ledger_chars'])
    + int(diag['hypothesis_ledger_chars'])
    + int(diag['continuity_memo_chars'])
)
# 旧描画の概算: CLAIM_LEDGER（死亡・投票の重複行 + COの判定重複）＋推測全文＋memoの全文列挙
# （旧memoは確定事実・投票・死亡・主張を本文付きで再掲していた）
legacy_claim = sum(
    len(f"Day {fact.day}: {fact.description}")
    for fact in state.fact_ledger.confirmed_facts
    if fact.visibility == 'public'
) + sum(
    len(f"Day {day} vote: " + " / ".join(f'{v}->{t}' for v, t in votes.items()))
    for day, _r, votes in state.vote_history
) + sum(
    len(f"Day {day}: {name} killed in attack") for day, name in state.attack_deaths
) + sum(
    len(f"Day {day}: {name} executed") for day, name in state.executions
) + sum(
    len(f"Day {c.day}: {c.claimant} claims {c.claimed_role}"
        + (f"; {c.target}={c.result}" if c.target and c.result else ''))
    for c in state.public_role_claims
) + sum(
    len(f"Day {c.day}: {c.claimant} claimed {c.target}={c.result}")
    for c in state.seer_claims if c.target and c.result
)
legacy_inference = sum(len(f.description) for f in state.fact_ledger.model_inferences[-6:])
legacy_memo = sum(
    len(f"[fact:{f.day}:{i}] {f.description}")
    for i, f in enumerate(state.fact_ledger.confirmed_facts, 1) if f.visibility == 'public'
) + sum(
    len(f"[vote:{day}:{'runoff' if r else 'initial'}:{v}] {v} voted for {t}")
    for day, r, votes in state.vote_history for v, t in votes.items()
) + sum(
    len(f"[death:attack:{day}:{name}] {name} died in an attack")
    for day, name in state.attack_deaths
) + sum(
    len(f"[death:execution:{day}:{name}] {name} was executed")
    for day, name in state.executions
) + sum(
    len(f"[claim:{i}] Day {c.day}: {c.claimant} publicly claimed {c.claimed_role}")
    for i, c in enumerate(state.public_role_claims, 1)
) + sum(
    len(f"[seer_claim:{i}] Day {c.day}: {c.claimant} publicly claimed {c.target}={c.result}")
    for i, c in enumerate(state.seer_claims, 1) if c.target and c.result
)
print(f"ledger+inference+hypotheses+memo chars now: {new_total}")
print(f"legacy claim-ledger+inference+memo chars: {legacy_claim + legacy_inference + legacy_memo}")
print(f"user_prompt chars now: {len(ctx.user_prompt)}")

# 厳密な差分: memo本文列挙→ID化 / CLAIM_LEDGER重複統合 / 推測全文→仮説要約 / 仮説欄の追加分
memo = state.reasoning_memos[actor.name]
new_memo_ids = sum(len(entry) for entry in memo.confirmed_facts + memo.public_claims)
delta_memo = legacy_memo - new_memo_ids
delta_claim = legacy_claim - int(diag['claim_ledger_chars'])
delta_inference = legacy_inference - int(diag['inference_ledger_chars'])
added_hypotheses = int(diag['hypothesis_ledger_chars'])
print(f"memo ids-only saving: {delta_memo} chars")
print(f"claim-ledger dedup saving: {delta_claim} chars")
print(f"inference summary saving: {delta_inference} chars")
print(f"hypothesis ledger added: {added_hypotheses} chars")
print(f"net change: {delta_memo + delta_claim + delta_inference - added_hypotheses} chars saved")
