from collections import Counter

import pytest

from werewolf import GameState, Phase, Player, Role, create_game, load_player_configs
from werewolf.config import PlayerConfig


def test_loads_all_eight_models_from_configuration() -> None:
    configs = load_player_configs()

    assert len(configs) == 8
    assert [config.model for config in configs] == [
        "CALM3-22B",
        "muse-glimmer",
        "mistral-small",
        "glm-4.7-flash",
        "qwen3.8:27b",
        "deepseek-r1:32b",
        "gpt-oss-20b",
        "gemma4:26b",
    ]


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

    assert state.phase is Phase.NIGHT
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
