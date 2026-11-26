# TASK-064 境界節のコスト実測（空状態／実行記録＋本人CO状態）
import sys

sys.path.insert(0, "src")

from werewolf.models import (  # noqa: E402
    Message,
    Phase,
    PublicRoleClaim,
    RoleActionProvenance,
)
from werewolf.prompt_contract import find_sections  # noqa: E402
from werewolf.prompts import PromptBuilder  # noqa: E402
from werewolf.setup import create_game  # noqa: E402


def _state(seed=2064):
    return create_game(
        seed=seed, config_path="config/players.json", response_language="日本語"
    )


LONG = (
    "{name}の{index}番目の発言です。{rival}の占い師COと投票の動きを比べると、"
    "昨夜の襲撃先との整合が怪しく見えるので、今日の投票ではその点を確認したいと考えています。"
)


def build(state, actor_name, layout, contract="simplified"):
    actor = next(p for p in state.players if p.name == actor_name)
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
        core_compression_policy="layered",
        prompt_layout_policy=layout,
    )


def _day1(state):
    state.public_timeline.record_day_start(state.day)
    state.conversation_log.append(Message("system", f"Day {state.day} begins."))
    state.phase = Phase.DAY_DISCUSSION
    state.discussion_turn = 3
    state.begin_discussion_turn()
    names = [p.name for p in state.players]
    for index, name in enumerate(names[:4]):
        state.append_discussion_message(
            Message(
                name,
                LONG.format(name=name, index=index + 1, rival=names[(index + 1) % 8]),
            )
        )
    asker, actor_name = names[3], names[4]
    state.append_discussion_message(
        Message(
            asker,
            f"{actor_name}さんに聞きます。{asker}が先に述べた投票案と比べて、"
            f"{actor_name}はどう考えますか？根拠を具体的に教えてください。",
        )
    )
    return actor_name, names


for seed in (1060, 2064, 3064):
    state = _state(seed)
    actor_name, names = _day1(state)
    v5 = build(state, actor_name, "v5")
    v4 = build(state, actor_name, "v4")
    legacy = build(state, actor_name, "auto", contract="legacy")
    spans = find_sections(v5.user_prompt)
    start, end = spans.get("role_action_boundary", (0, 0))
    print(
        f"[empty] seed={seed} v5={len(v5.user_prompt)} v4={len(v4.user_prompt)} "
        f"legacy={len(legacy.user_prompt)} boundary={end - start}"
    )

state = _state(4064)
actor_name, names = _day1(state)
tracker = RoleActionProvenance()
tracker.record_engine_action(actor_name, "divination", 1)
tracker.record_engine_action(actor_name, "divination", 3)
tracker.record_engine_action(names[0], "attack", 2)
state.role_action_provenance = tracker
state.public_role_claims.append(
    PublicRoleClaim(day=1, claimant=actor_name, claimed_role="seer")
)
state.public_role_claims.append(
    PublicRoleClaim(day=2, claimant=actor_name, claimed_role="villager")
)
v5_full = build(state, actor_name, "v5")
spans = find_sections(v5_full.user_prompt)
start, end = spans.get("role_action_boundary", (0, 0))
print(f"[full] boundary={end - start}")
print(v5_full.user_prompt[start:end].strip())
