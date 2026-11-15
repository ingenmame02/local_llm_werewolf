import sys
from pathlib import Path

SRC = Path(r'c:\Users\yukic\local_llm_werewolf\src')
sys.path.insert(0, str(SRC))

from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game
from werewolf.models import Message
from werewolf.engine import extract_public_role_claim, extract_public_seer_result
from werewolf.discussion_format import parse_discussion_output

state = create_game(seed=1001, config_path=r'c:\Users\yukic\local_llm_werewolf\config\players.json')
print('roles:', [(p.name, p.role.value) for p in state.players])
alive = tuple(p.name for p in state.players if p.is_alive)
history = tuple(state.conversation_log) + (Message(speaker='system', content='Day 1 begins.'),)

MARKERS = [
    '[SYSTEM_RULES]', '\n[OUTPUT_LANGUAGE]', '[FACT_LEDGER]', '\n[CLAIM_LEDGER]',
    '[SELF_ACTION_MEMORY', '[PRIVATE_SEER_RESULT', '[CONTINUITY_MEMO',
    '[Your Speeches Today', '[Day 1 First Turn', '[Full Current Round Dialogue]',
    '\n\n[CONVERSATION_EVIDENCE', '\n\n[CURRENT_TASK]', '[秘匿情報の扱い',
    '[Instructions]', '[Discussion Guidelines', '[最終確認', '- 最終確認',
]


def sections(prompt: str) -> list[tuple[str, int]]:
    positions = []
    for marker in MARKERS:
        idx = prompt.find(marker)
        if idx >= 0:
            positions.append((idx, marker))
    positions.sort()
    out = []
    head = positions[0][0] if positions else 0
    out.append(('(head/system)', head))
    for i, (idx, marker) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(prompt)
        out.append((marker.strip(), end - idx))
    return out


for day, turn, extra in ((1, 1, []), (1, 2, ['p3']), (3, 1, ['day2'])):
    actor = next(p for p in state.players if p.name == 'Player 3')
    ctx = PromptBuilder.build_discussion_prompt(
        actor=actor, alive_players=alive, conversation_history=tuple(history),
        day=day, state=state, include_claim=True,
        context_strategy='hybrid_retrieval', question_hint_mode='none',
    )
    total = len(ctx.system_prompt) + len(ctx.user_prompt)
    print(f'--- day {day} Player 3 discussion chars sys={len(ctx.system_prompt)} user={len(ctx.user_prompt)} total={total} est_tokens={int(total*0.8)}')
    for name, size in sections(ctx.user_prompt):
        print(f'    {size:6d}  {name}')

vote_actor = next(p for p in state.players if p.name == 'Player 3')
vctx = PromptBuilder.build_vote_prompt(actor=vote_actor, alive_players=alive,
    valid_candidates=tuple(n for n in alive if n != 'Player 3'), today_summary='x', day=1, state=state)
print('vote chars:', len(vctx.system_prompt), len(vctx.user_prompt), 'est_tokens', int((len(vctx.system_prompt)+len(vctx.user_prompt))*0.8))

# hallucinated Player-3 day-1 output reproduction
halluc = "[CO] 占い師\n\n[RESULT] ターゲット=プレイヤーX 結果=人間\n\n[BODY] これは、プレイヤーXが人間であることを示唆する新しい証拠です。"
parsed = parse_discussion_output(halluc)
print('parsed:', parsed)
names = tuple(p.name for p in state.players)
print('claim recognized:', extract_public_role_claim(parsed.claim if parsed else '', 'Player 3'))
print('result extracted:', extract_public_seer_result(parsed.result if parsed else '', names))
print('result extracted full:', extract_public_seer_result(halluc, names))
