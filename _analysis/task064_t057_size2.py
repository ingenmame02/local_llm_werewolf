# test_task057 の実シナリオ（full_transcript / include_claim=False / hint=none）でサイズ比較
import sys

sys.path.insert(0, "src")

from werewolf.models import Message, Phase  # noqa: E402
from werewolf.setup import create_game  # noqa: E402

LONG_SPEECH = (
    "{name}の{index}番目の発言です。{rival}の占い師COと投票の動きを比べると、"
    "昨夜の襲撃先との整合が怪しく見えるので、今日の投票ではその点を確認したいと考えています。"
)

state = create_game(
    seed=1001, config_path="config/players.json", response_language="日本語"
)
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


def build(layout, contract):
    from werewolf.prompts import PromptBuilder

    return PromptBuilder.build_discussion_prompt(
        actor,
        tuple(state.alive_players),
        tuple(state.conversation_log),
        day=1,
        state=state,
        include_claim=False,
        context_strategy="full_transcript",
        question_hint_mode="none",
        prompt_contract_policy=contract,
        prompt_layout_policy=layout,
    )


for label, layout, contract in (
    ("simplified v5", "v5", "simplified"),
    ("simplified v4", "v4", "simplified"),
    ("simplified auto", "auto", "simplified"),
    ("legacy auto", "auto", "legacy"),
):
    ctx = build(layout, contract)
    print(f"{label}: {len(ctx.user_prompt)}")
