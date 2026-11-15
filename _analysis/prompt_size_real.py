import sys
from pathlib import Path

SRC = Path(r'c:\Users\yukic\local_llm_werewolf\src')
sys.path.insert(0, str(SRC))

import importlib

from werewolf.engine import GameEngine
from werewolf.llm_actions import OllamaActions
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game

sys.path.insert(0, r'c:\Users\yukic\local_llm_werewolf\_analysis')
from sim_game import SimulatedClient  # reuse the scripted responder

CONFIG = r'c:\Users\yukic\local_llm_werewolf\config\players.json'
state = create_game(seed=20261115, config_path=CONFIG, response_language="日本語")
client = SimulatedClient(state)
actions = OllamaActions(
    state, client, seed=20261115,
    options={"question_hint_mode": "none", "temperature": 0.7},
    discussion_claim_mode="legacy",
)
engine = GameEngine(state, actions, seed=20261115)

# Day 3 の議論局面で計測するため、生存しているプレイヤーを使う
actor = next(p for p in state.players if p.is_alive)
alive = tuple(p.name for p in state.players if p.is_alive)
history = tuple(state.conversation_log)
day = state.day if state.day >= 2 else 3

ctx = PromptBuilder.build_discussion_prompt(
    actor=actor, alive_players=alive, conversation_history=history, day=3,
    state=state, include_claim=True, context_strategy="hybrid_retrieval",
    question_hint_mode="none",
)
header_with = PromptBuilder._fact_header(actor, alive, 3, "Day (Discussion)", state,
                                         conversation_history=history, include_structured_ledger=True)
header_without = PromptBuilder._fact_header(actor, alive, 3, "Day (Discussion)", state,
                                            conversation_history=history, include_structured_ledger=False)
removed = len(header_with) - len(header_without)
total = len(ctx.system_prompt) + len(ctx.user_prompt)
print(f'realistic day-3 discussion: sys={len(ctx.system_prompt)} user={len(ctx.user_prompt)} total={total}')
print(f'standard: facts/claims duplication removed from discussion header: {removed} chars')
print(f'public claims in ledger: {len(state.public_role_claims)}, confirmed facts: {len(state.fact_ledger.confirmed_facts)}, inferences: {len(state.fact_ledger.model_inferences)}')
print(f'est. tokens (0.45/char, log-measured ratio): {int(total*0.45)}')
print(f'would-be before total (with duplication): {total + removed} -> est {int((total+removed)*0.45)}')
