import json
import sys
from pathlib import Path

SRC = Path(r'c:\Users\yukic\local_llm_werewolf\src')
sys.path.insert(0, str(SRC))

from werewolf.models import Message, Role
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game

CONFIG = r'c:\Users\yukic\local_llm_werewolf\config\players.json'
MARKERS = [
    '[SYSTEM_RULES]', '\n[OUTPUT_LANGUAGE]', '[FACT_LEDGER]', '\n[CLAIM_LEDGER]',
    '[SELF_ACTION_MEMORY', '[PRIVATE_SEER_RESULT', '[CONTINUITY_MEMO',
    '[Your Speeches Today', '[Day 1 First Turn', '[Full Current Round Dialogue]',
    '\n\n[CONVERSATION_EVIDENCE', '\n\n[CURRENT_TASK]', '[Discussion Guidelines',
]


def report(label, ctx):
    print(f'== {label}: sys={len(ctx.system_prompt)} user={len(ctx.user_prompt)} '
          f'total={len(ctx.system_prompt) + len(ctx.user_prompt)}')
    text = ctx.user_prompt
    positions = [(text.find(m), m) for m in MARKERS]
    positions = [(pos, m) for pos, m in positions if pos >= 0]
    positions.sort()
    for i, (pos, marker) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        print(f'   {marker:28s} {end - pos:6d}')


state = create_game(seed=7, config_path=CONFIG)
alive = tuple(p.name for p in state.players if p.is_alive)
actor = next(p for p in state.players if p.name == 'Player 3')
history: list[Message] = [Message(speaker='system', content='Day 1 begins.')]
speech = ('Player 1 CO占い師、Player 1は人間と判定。Player 2は投票でPlayer 1を疑う。'
          'Player 4はCOなし、投票はPlayer 2へ。議論は vote の流れとCOの整合を重視する。')
for day in (1, 2):
    history.append(Message(speaker='system', content=f'Day {day} begins.'))
    for turn in range(1, 4):
        for name in alive:
            history.append(Message(speaker=name, content=f'{speech} ({day}-{turn}-{name})'))
state.discussion_turn = 2
ctx = PromptBuilder.build_discussion_prompt(
    actor=actor, alive_players=alive, conversation_history=tuple(history), day=2,
    state=state, include_claim=True, context_strategy='hybrid_retrieval',
    question_hint_mode='none',
)
report('discussion day2 hybrid', ctx)
vote = PromptBuilder.build_vote_prompt(
    actor=actor, alive_players=alive,
    valid_candidates=tuple(n for n in alive if n != actor.name), day=2, state=state,
)
report('vote day2', vote)
seer_actor = next(p for p in state.players if p.role is Role.SEER)
seer = PromptBuilder.build_seer_prompt(
    actor=seer_actor, alive_players=tuple(state.alive_players), seer_results={}, day=2, state=state
)
report('seer night2', seer)
