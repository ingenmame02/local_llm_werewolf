import sys
from pathlib import Path

SRC = Path(r'c:\Users\yukic\local_llm_werewolf\src')
sys.path.insert(0, str(SRC))

from werewolf.models import Message, PublicRoleClaim, Role
from werewolf.prompts import (
    ROLE_DESCRIPTIONS_EN,
    RULE_CARD,
    WEREWOLF_THEORY_CARD,
    PromptBuilder,
)
from werewolf.setup import create_game

state = create_game(seed=1001, config_path=r'c:\Users\yukic\local_llm_werewolf\config\players.json')
alive = tuple(p.name for p in state.players if p.is_alive)
player = next(p for p in state.players if p.name == 'Player 2')
seer = next(p for p in state.players if p.role is Role.SEER)
wolf = next(p for p in state.players if p.role is Role.WOLF)
partner = next(p.name for p in state.players if p.role is Role.WOLF and p.name != wolf.name)

checks = []


def check(label, haystack, needles):
    for needle in needles:
        ok = needle in haystack
        checks.append((ok, label, needle))


check('RULE_CARD', RULE_CARD, ['a Madman is divined as Human', '狂人の占い結果は「人間」になる'])
check('THEORY', WEREWOLF_THEORY_CARD, [
    'Day 1', '真置き', 'the longer that lone-CO state persists',
    'contradicting verdicts on the same target', 'definitive proof',
    'Villager OR a Madman', 'never reports a Madman as Werewolf',
])
check('ROLE_WOLF', ROLE_DESCRIPTIONS_EN[Role.WOLF], ['Perspective Split', '村人陣営', 'Madman'])
check('ROLE_MADMAN', ROLE_DESCRIPTIONS_EN[Role.MADMAN], ["always 'Human'", 'Madman'])
check('ROLE_SEER', ROLE_DESCRIPTIONS_EN[Role.SEER], ["divined as 'Human'", '真置き', 'Madman'])
check('ROLE_VILLAGER', ROLE_DESCRIPTIONS_EN[Role.VILLAGER], ['Madman', "'Human'"])

# 単独COのガイダンス（Day2 / Day1巡2 / Day1巡1）は別stateで検証する
for day, turn, needles in (
    (2, 1, ['真置き', 'stood alone since Day 1']),
    (1, 2, ['first discussion round finished']),
    (1, 1, ['little to doubt']),
):
    h_state = create_game(seed=1001, config_path=r'c:\Users\yukic\local_llm_werewolf\config\players.json')
    h_alive = tuple(p.name for p in h_state.players if p.is_alive)
    h_state.public_role_claims = [PublicRoleClaim(day=1, claimant='Player 4', claimed_role='seer')]
    h_state.seer_claims = []
    h_state.discussion_turn = turn
    h_player = next(p for p in h_state.players if p.name == 'Player 2')
    check(f'header_day{day}_turn{turn}',
          PromptBuilder._fact_header(h_player, h_alive, day, 'Day (Discussion)', h_state), needles)

wolf_ctx = PromptBuilder.build_wolf_discuss_prompt(
    actor=wolf, partner_name=partner, alive_players=alive, wolf_chat_history=(), day=2, state=state
)
check('wolf_consult', wolf_ctx.user_prompt,
      ['[Consult Perspective (internal to your camp)]', '村人陣営', 'village order', '村の秩序'])

state.seer_results = {'Player 1': 'human'}
co_ctx = PromptBuilder.build_discussion_claim_prompt(actor=seer, alive_players=alive, day=1, state=state)
check('co_prompt', co_ctx.user_prompt, ['CO now beats silence', '真置き'])
rival = next(name for name in alive if name != seer.name)
state.public_role_claims = [PublicRoleClaim(day=1, claimant=rival, claimed_role='seer')]
rival_ctx = PromptBuilder.build_discussion_claim_prompt(actor=seer, alive_players=alive, day=2, state=state)
if 'CO now beats silence' in rival_ctx.user_prompt:
    checks.append((False, 'co_prompt_rival', 'must not contain CO now beats silence'))
state.public_role_claims = []

vote_ctx = PromptBuilder.build_vote_prompt(
    actor=player, alive_players=alive,
    valid_candidates=tuple(n for n in alive if n != player.name), day=1, state=state
)
check('vote_prompt', vote_ctx.user_prompt, ['Required format (write this line first)', 'FIRST line'])
seer_ctx = PromptBuilder.build_seer_prompt(
    actor=seer, alive_players=alive, seer_results={}, day=1, state=state
)
check('seer_prompt', seer_ctx.user_prompt, ['Required format (write this line first)'])

state.seer_claims.append(
    __import__('werewolf.models', fromlist=['PublicSeerClaim']).PublicSeerClaim(
        day=1, claimant=seer.name, target='Player 1', result='human'
    )
)
ledger = PromptBuilder._discussion_claim_ledger(state)
check('claim_ledger', ledger, ['Interpretation', '狂人の占い結果は「人間」になる'])
private_block = PromptBuilder._private_seer_block(
    actor=seer, current_seer_results=['Player 1: Human'], state=state, day=1,
    pending_wolf_results=[], human_results_can_be_held=True, include_claim=True,
)
check('private_seer', private_block, ['狂人は占い結果が「人間」になる', 'Madman'])

# TASK-043: 案内文にプレースホルダ（"Player X"）を出さない / 投票候補に自分を含めない
full_prompt = (
    wolf_ctx.system_prompt + wolf_ctx.user_prompt + co_ctx.user_prompt
    + vote_ctx.user_prompt + seer_ctx.user_prompt
)
checks.append(('Player X' not in full_prompt, 'placeholder_free', 'no literal "Player X" in prompts'))
checks.append((player.name not in vote_ctx.user_prompt.split('Eligible Vote Candidates')[1][:200],
               'vote_self_excluded', 'vote candidates exclude the voter'))

failed = [c for c in checks if not c[0]]
for ok, label, needle in checks:
    print(f'{"OK " if ok else "FAIL"} {label}: {needle[:60]!r}')
print(f'-- {len(checks) - len(failed)}/{len(checks)} passed')

# プロンプト量の計測
history = tuple(state.conversation_log) + (Message(speaker='system', content='Day 1 begins.'),)
for day in (1, 3):
    ctx = PromptBuilder.build_discussion_prompt(
        actor=player, alive_players=alive, conversation_history=history, day=day,
        state=state, include_claim=True, context_strategy='hybrid_retrieval',
        question_hint_mode='none',
    )
    total = len(ctx.system_prompt) + len(ctx.user_prompt)
    print(f'discussion day{day}: sys={len(ctx.system_prompt)} user={len(ctx.user_prompt)} total={total} est_tok={int(total*0.46)}')
vt = len(vote_ctx.system_prompt) + len(vote_ctx.user_prompt)
print(f'vote: total={vt} est_tok={int(vt*0.46)}')
sys.exit(1 if failed else 0)
