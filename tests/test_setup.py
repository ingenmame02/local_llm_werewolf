import json
from collections import Counter

import pytest

from werewolf import (
    GameState,
    Phase,
    Player,
    Role,
    create_game,
    load_player_configs,
    parse_action_num_predict_pairs,
    parse_thinking_by_action_pairs,
)
from werewolf.config import PlayerConfig


def test_loads_all_eight_models_from_configuration() -> None:
    configs = load_player_configs()

    assert len(configs) == 8
    assert [config.model for config in configs] == [
        "werewolf-calm3-22b-8k",
        "werewolf-muse-glimmer-8k",
        "werewolf-mistral-small-8k",
        "werewolf-glm-4.7-flash-8k",
        "werewolf-qwen3.8-8k",
        "werewolf-deepseek-r1-8k",
        "werewolf-gpt-oss-20b-8k",
        "werewolf-gemma4-8k",
    ]
    assert [config.num_predict for config in configs] == [
        1024,
        4096,
        512,
        768,
        768,
        1536,
        1024,
        4096,
    ]
    assert configs[0].personality_profile == configs[4].personality_profile == "analytical"
    assert len({config.personality_profile for config in configs}) >= 2


def test_create_game_assigns_expected_roles_and_wolf_partners() -> None:
    state = create_game(seed=123)

    assert Counter(player.role for player in state.players) == {
        Role.WOLF: 2,
        Role.MADMAN: 1,
        Role.SEER: 1,
        Role.VILLAGER: 4,
    }
    wolves = [player for player in state.players if player.role is Role.WOLF]
    assert wolves[0].partner == wolves[1].name
    assert wolves[1].partner == wolves[0].name
    assert all(
        player.partner is None for player in state.players if player.role is not Role.WOLF
    )


def test_seed_reproduces_role_assignment() -> None:
    first = create_game(seed=987)
    second = create_game(seed=987)

    assert [(p.name, p.role) for p in first.players] == [
        (p.name, p.role) for p in second.players
    ]
    assert first.seed == second.seed == 987


def test_game_state_tracks_alive_players_and_centralizes_death() -> None:
    state = create_game(seed=3)
    victim = state.players[0]

    state.mark_dead(victim.name)

    assert victim.name not in state.alive_players
    assert len(state.alive_players) == 7
    with pytest.raises(ValueError, match="already dead"):
        state.mark_dead(victim.name)
    with pytest.raises(KeyError, match="Unknown player"):
        state.mark_dead("not a player")


def test_game_state_rejects_duplicate_player_names() -> None:
    players = [
        Player(name="same", model="model-a", role=Role.VILLAGER),
        Player(name="same", model="model-b", role=Role.SEER),
    ]

    with pytest.raises(ValueError, match="unique"):
        GameState(players=players)


def test_game_initializes_in_night_phase() -> None:
    state = create_game(seed=7)

    assert state.phase is Phase.NIGHT_WOLF
    assert state.day == 1


def test_player_configuration_rejects_duplicate_names(tmp_path) -> None:
    config_file = tmp_path / "players.json"
    configs = [PlayerConfig(f"Player {index}", f"model-{index}") for index in range(1, 9)]
    configs[1] = PlayerConfig(configs[0].name, configs[1].model)
    config_file.write_text(
        '{"players": ['
        + ", ".join(
            f'{{"name": "{config.name}", "model": "{config.model}"}}'
            for config in configs
        )
        + "]}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_player_configs(config_file)


def test_player_configuration_applies_per_model_context_length(tmp_path) -> None:
    config_file = tmp_path / "players.json"
    players = [
        {"name": f"Player {index}", "model": f"model-{index}"}
        for index in range(1, 9)
    ]
    players[5]["num_ctx"] = 16384
    config_file.write_text(
        json.dumps({"players": players}),
        encoding="utf-8",
    )

    configs = load_player_configs(config_file)
    state = create_game(seed=1, config_path=config_file)

    assert configs[5].num_ctx == 16384
    assert configs[4].num_ctx is None
    assert state.players[5].num_ctx == 16384
    assert state.players[4].num_ctx is None


def test_player_configuration_applies_model_budgets_and_response_language(tmp_path) -> None:
    config_file = tmp_path / "players.json"
    players = [
        {"name": f"Player {index}", "model": f"model-{index}"}
        for index in range(1, 9)
    ]
    players[2].update(
        {"timeout": 18.5, "num_ctx": 8192, "num_predict": 256, "response_language": "English"}
    )
    config_file.write_text(json.dumps({"players": players}), encoding="utf-8")

    config = load_player_configs(config_file)[2]
    player = create_game(seed=1, config_path=config_file).players[2]

    assert config.timeout == player.timeout == 18.5
    assert config.num_ctx == player.num_ctx == 8192
    assert config.num_predict == player.num_predict == 256
    assert config.response_language == player.response_language == "English"
    assert load_player_configs(config_file)[0].response_language == "日本語"


def test_player_personality_profile_loads_into_game_and_rejects_unknown_values(tmp_path) -> None:
    config_file = tmp_path / "players.json"
    players = [
        {"name": f"Player {index}", "model": f"model-{index}"}
        for index in range(1, 9)
    ]
    players[0]["personality_profile"] = "questioning"
    config_file.write_text(json.dumps({"players": players}), encoding="utf-8")

    assert load_player_configs(config_file)[0].personality_profile == "questioning"
    assert create_game(seed=1, config_path=config_file).players[0].personality_profile == "questioning"

    players[0]["personality_profile"] = "win-at-all-costs"
    config_file.write_text(json.dumps({"players": players}), encoding="utf-8")
    with pytest.raises(ValueError, match="personality_profile"):
        load_player_configs(config_file)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("timeout", 0),
        ("timeout", True),
        ("num_predict", 0),
        ("num_predict", 1.5),
        ("response_language", "  "),
    ],
)
def test_player_configuration_rejects_invalid_model_settings(tmp_path, key, value) -> None:
    config_file = tmp_path / "players.json"
    players = [
        {"name": f"Player {index}", "model": f"model-{index}"}
        for index in range(1, 9)
    ]
    players[0][key] = value
    config_file.write_text(json.dumps({"players": players}), encoding="utf-8")

    with pytest.raises(ValueError, match=key):
        load_player_configs(config_file)


@pytest.mark.parametrize("invalid_num_ctx", [0, -1, True, "4096"])
def test_player_configuration_rejects_invalid_context_length(
    tmp_path, invalid_num_ctx
) -> None:
    config_file = tmp_path / "players.json"
    players = [
        {"name": f"Player {index}", "model": f"model-{index}"}
        for index in range(1, 9)
    ]
    players[0]["num_ctx"] = invalid_num_ctx
    config_file.write_text(
        json.dumps({"players": players}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="num_ctx"):
        load_player_configs(config_file)


def test_player_configuration_applies_per_action_generation_budgets(tmp_path) -> None:
    config_file = tmp_path / "players.json"
    players = [
        {"name": f"Player {index}", "model": f"model-{index}"}
        for index in range(1, 9)
    ]
    players[3]["num_predict_by_action"] = {"discussion": 1024, "vote": 256}
    config_file.write_text(json.dumps({"players": players}), encoding="utf-8")

    config = load_player_configs(config_file)[3]
    player = create_game(seed=1, config_path=config_file).players[3]

    assert config.num_predict_by_action == {"discussion": 1024, "vote": 256}
    assert player.num_predict_by_action == {"discussion": 1024, "vote": 256}
    assert load_player_configs(config_file)[0].num_predict_by_action is None


@pytest.mark.parametrize(
    "invalid_budgets",
    [
        {"attack": 0},
        {"attack": -8},
        {"attack": True},
        {"attack": "256"},
        {"unknown_action": 256},
        "256",
    ],
)
def test_player_configuration_rejects_invalid_per_action_budgets(
    tmp_path, invalid_budgets
) -> None:
    config_file = tmp_path / "players.json"
    players = [
        {"name": f"Player {index}", "model": f"model-{index}"}
        for index in range(1, 9)
    ]
    players[0]["num_predict_by_action"] = invalid_budgets
    config_file.write_text(json.dumps({"players": players}), encoding="utf-8")

    with pytest.raises(ValueError, match="num_predict_by_action"):
        load_player_configs(config_file)


def test_cli_action_num_predict_pairs_are_parsed_and_validated() -> None:
    assert parse_action_num_predict_pairs([]) == {}
    assert parse_action_num_predict_pairs(
        ["discussion=1024", "wolf_discussion=640"]
    ) == {"discussion": 1024, "wolf_discussion": 640}

    for invalid in ["discussion", "discussion=", "=128", "discussion=0", "discussion=-5",
                    "discussion=abc", "attack=1.5", "unknown_action=10"]:
        with pytest.raises(ValueError):
            parse_action_num_predict_pairs([invalid])


def test_player_configuration_applies_thinking_settings(tmp_path) -> None:
    config_file = tmp_path / "players.json"
    players = [
        {"name": f"Player {index}", "model": f"model-{index}"}
        for index in range(1, 9)
    ]
    players[2]["thinking"] = False
    players[3]["thinking_by_action"] = {"discussion": True, "vote": False}
    players[4]["thinking_by_action"] = {"discussion": "off"}
    config_file.write_text(json.dumps({"players": players}), encoding="utf-8")

    configs = load_player_configs(config_file)
    state = create_game(seed=1, config_path=config_file)

    assert configs[2].thinking is False
    assert state.players[2].thinking is False
    assert configs[3].thinking_by_action == {"discussion": True, "vote": False}
    assert state.players[3].thinking_by_action == {"discussion": True, "vote": False}
    # 文字列のboolean風指定は正規化する
    assert configs[4].thinking_by_action == {"discussion": False}
    assert configs[0].thinking is None
    assert configs[0].thinking_by_action is None


@pytest.mark.parametrize("invalid_thinking", [1, 0, "maybe", 1.5, [True]])
def test_player_configuration_rejects_invalid_thinking(
    tmp_path, invalid_thinking
) -> None:
    config_file = tmp_path / "players.json"
    players = [
        {"name": f"Player {index}", "model": f"model-{index}"}
        for index in range(1, 9)
    ]
    players[0]["thinking"] = invalid_thinking
    config_file.write_text(json.dumps({"players": players}), encoding="utf-8")

    with pytest.raises(ValueError, match="thinking"):
        load_player_configs(config_file)


@pytest.mark.parametrize(
    "invalid_flags",
    [{"vote": "maybe"}, {"vote": 1}, {"vote": None}, {"unknown_action": True}, "true"],
)
def test_player_configuration_rejects_invalid_per_action_thinking(
    tmp_path, invalid_flags
) -> None:
    config_file = tmp_path / "players.json"
    players = [
        {"name": f"Player {index}", "model": f"model-{index}"}
        for index in range(1, 9)
    ]
    players[0]["thinking_by_action"] = invalid_flags
    config_file.write_text(json.dumps({"players": players}), encoding="utf-8")

    with pytest.raises(ValueError, match="thinking_by_action"):
        load_player_configs(config_file)


def test_cli_thinking_pairs_are_parsed_and_validated() -> None:
    assert parse_thinking_by_action_pairs([]) == {}
    assert parse_thinking_by_action_pairs(
        ["discussion=true", "vote=off", "co_correction=false"]
    ) == {"discussion": True, "vote": False, "co_correction": False}

    for invalid in [
        "discussion",
        "discussion=",
        "=true",
        "discussion=maybe",
        "unknown_action=true",
    ]:
        with pytest.raises(ValueError):
            parse_thinking_by_action_pairs([invalid])
