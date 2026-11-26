# TASK-064: test_task057 の盤面（seed1001/turn2）で simplified を v4 / v5 / legacy で比較
import sys

sys.path.insert(0, "src")

from werewolf.models import Message, Phase  # noqa: E402
from werewolf.prompt_contract import find_sections  # noqa: E402
from werewolf.prompts import PromptBuilder  # noqa: E402
from werewolf.setup import create_game  # noqa: E402

LONG_SPEECH = (
    "{name}の{index}番目の発言です。{rival}の占い師COと投票の動きを比べると、"
    "昨夜の襲撃先との整合が怪しく見えるので、今日の投票ではその点を確認したいと考えています。"
)


def _state(seed=1001):
    return create_game(
        seed=seed, config_path="config/players.json", response_language="日本語"
    )


state = _state()
names = [p.name for p in state.players]
state.public_timeline.record_day_start(state.day)
state.conversation_log.append(Message("system", f"Day {state.day} begins."))
state.phase = Phase.DAY_DISCUSSION
state.discussion_turn = 2
state.begin_discussion_turn()
for index, name in enumerate(names[:4]):
    state.append_discussion_message(
        Message(
            name, LONG_SPEECH.format(name=name, index=index + 1, rival=names[(index + 1) % 8])
        )
    )
actor_name = names[4]
asker = names[3]
state.append_discussion_message(
    Message(
        asker,
        f"{actor_name}さんに聞きます。{asker}が先に述べた投票案と比べて、"
        f"{actor_name}はどう考えますか？根拠を具体的に教えてください。",
    )
)

actor = next(p for p in state.players if p.name == actor_name)


def build(layout, contract, compression):
    return PromptBuilder.build_discussion_prompt(
        actor,
        tuple(state.alive_players),
        tuple(state.conversation_log),
        day=state.day,
        state=state,
        include_claim=True,
        context_strategy="hybrid_retrieval",
        question_hint_mode="strict",
        prompt_contract_policy=contract,
        core_compression_policy=compression,
        prompt_layout_policy=layout,
    )


combos = [
    ("v5/layered", "v5", "simplified", "layered"),
    ("v4/layered", "v4", "simplified", "layered"),
    ("auto(off)", "auto", "simplified", "off"),
    ("legacy", "auto", "legacy", "layered"),
]
for label, layout, contract, compression in combos:
    ctx = build(layout, contract, compression)
    spans = find_sections(ctx.user_prompt)
    boundary = spans.get("role_action_boundary")
    print(
        f"{label}: chars={len(ctx.user_prompt)} boundary={(boundary[1] - boundary[0]) if boundary else 0}"
    )
