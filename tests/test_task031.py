import json

import pytest

from werewolf import (
    GameState,
    Message,
    OllamaActions,
    Player,
    Role,
    load_player_configs,
)
from werewolf.prompts import PromptBuilder
from scripts.evaluate_models import generate_scenarios


def _players() -> list[Player]:
    roles = (
        Role.SEER,
        Role.VILLAGER,
        Role.WOLF,
        Role.MADMAN,
        Role.VILLAGER,
        Role.VILLAGER,
        Role.VILLAGER,
        Role.VILLAGER,
    )
    return [
        Player(f"Player {index}", f"model-{index}", role)
        for index, role in enumerate(roles, 1)
    ]


def test_model_specific_context_settings_load_and_apply(tmp_path) -> None:
    config_path = tmp_path / "players.json"
    players = [
        {"name": f"Player {index}", "model": f"model-{index}"}
        for index in range(1, 9)
    ]
    players[0].update(
        {
            "num_ctx": 12288,
            "max_num_ctx": 16384,
            "recommended_num_ctx": 12288,
            "context_strategy": "hybrid_retrieval",
        }
    )
    config_path.write_text(json.dumps({"players": players}), encoding="utf-8")

    config = load_player_configs(config_path)[0]
    assert config.num_ctx == 12288
    assert config.max_num_ctx == 16384
    assert config.recommended_num_ctx == 12288
    assert config.context_strategy == "hybrid_retrieval"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_num_ctx", 0),
        ("recommended_num_ctx", True),
        ("context_strategy", "unknown"),
    ],
)
def test_model_specific_context_settings_reject_invalid_values(
    tmp_path, key: str, value: object
) -> None:
    config_path = tmp_path / "players.json"
    players = [
        {"name": f"Player {index}", "model": f"model-{index}"}
        for index in range(1, 9)
    ]
    players[0][key] = value
    config_path.write_text(json.dumps({"players": players}), encoding="utf-8")

    with pytest.raises(ValueError, match=key):
        load_player_configs(config_path)


def test_context_request_uses_recommendation_and_respects_model_maximum() -> None:
    players = _players()
    players[0].recommended_num_ctx = 12288
    players[0].max_num_ctx = 16384
    state = GameState(players=players)
    actions = OllamaActions(state, object())

    recommended, *_ = actions._options_for(players[0])
    assert recommended["num_ctx"] == 12288

    players[0].num_ctx = 32768
    capped, *_ = actions._options_for(players[0])
    assert capped["num_ctx"] == 16384

    players[0].num_ctx = None
    players[0].recommended_num_ctx = None
    bounded, *_ = actions._options_for(players[0])
    assert bounded["num_ctx"] == 16384


def test_reasoning_memo_uses_cited_public_events_and_keeps_private_info_per_actor() -> None:
    players = _players()
    state = GameState(players=players)
    state.seer_results["Player 2"] = "wolf"
    state.vote_history.append((1, False, {"Player 4": "Player 2"}))
    state.conversation_log.extend(
        [
            Message("Player 4", "Player 1さん、Player 2への投票理由は何ですか？"),
            Message("Player 5", "Player 2の発言を確認したいです。"),
        ]
    )

    state.refresh_reasoning_memos()
    seer_memo = state.reasoning_memos["Player 1"]
    villager_memo = state.reasoning_memos["Player 2"]

    assert "vote:1:initial:Player 4" in seer_memo.suspicion_ranking[0]
    assert "[speech:1]" in seer_memo.unresolved_questions[0]
    assert seer_memo.private_role_facts == ["[private_seer:1] Player 2=wolf"]
    assert villager_memo.private_role_facts == []
    assert "Player 2=wolf" not in json.dumps(villager_memo.__dict__, ensure_ascii=False)


def test_discussion_prompt_only_receives_the_actor_private_memo() -> None:
    players = _players()
    state = GameState(players=players)
    state.seer_results["Player 2"] = "wolf"
    state.refresh_reasoning_memos()

    seer_prompt = PromptBuilder.build_discussion_prompt(
        actor=players[0],
        alive_players=tuple(state.alive_players),
        conversation_history=(Message("Player 3", "Player 2は怪しい？"),),
        day=1,
        state=state,
        context_strategy="structured_ledger",
    ).user_prompt
    villager_prompt = PromptBuilder.build_discussion_prompt(
        actor=players[1],
        alive_players=tuple(state.alive_players),
        conversation_history=(Message("Player 3", "Player 2は怪しい？"),),
        day=1,
        state=state,
        context_strategy="structured_ledger",
    ).user_prompt

    assert "[private_seer:1] Player 2=wolf" in seer_prompt
    assert "[private_seer:1] Player 2=wolf" not in villager_prompt
    assert "ACTOR_PRIVATE_MEMO" in villager_prompt


def test_benchmark_generates_memory_on_and_off_as_independent_conditions() -> None:
    actor = Player("Player 1", "model-1", Role.VILLAGER)
    names = tuple(f"Player {index}" for index in range(1, 9))

    memory_on = next(
        scenario
        for scenario in generate_scenarios(
            actor, names, reasoning_memory=True
        )
        if scenario.action_type == "discussion"
    )
    memory_off = next(
        scenario
        for scenario in generate_scenarios(
            actor, names, reasoning_memory=False
        )
        if scenario.action_type == "discussion"
    )

    assert "【CONTINUITY_MEMO" in memory_on.prompt_ctx.user_prompt
    assert "【CONTINUITY_MEMO" not in memory_off.prompt_ctx.user_prompt
    assert memory_on.prompt_ctx.diagnostics["reasoning_memory_enabled"] is True
    assert memory_off.prompt_ctx.diagnostics["reasoning_memory_enabled"] is False
