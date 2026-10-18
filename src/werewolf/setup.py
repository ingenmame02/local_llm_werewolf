import random
from pathlib import Path

from .config import load_player_configs
from .models import GameState, Player, Role


INITIAL_ROLES = (
    Role.WOLF,
    Role.WOLF,
    Role.MADMAN,
    Role.SEER,
    Role.VILLAGER,
    Role.VILLAGER,
    Role.VILLAGER,
    Role.VILLAGER,
)


def create_game(
    seed: int | None = None,
    config_path: str | Path | None = None,
) -> GameState:
    player_configs = load_player_configs(config_path)
    roles = list(INITIAL_ROLES)
    random.Random(seed).shuffle(roles)

    players = [
        Player(name=config.name, model=config.model, role=role)
        for config, role in zip(player_configs, roles, strict=True)
    ]
    wolves = [player for player in players if player.role is Role.WOLF]
    for player in wolves:
        player.partner = next(wolf.name for wolf in wolves if wolf is not player)

    return GameState(players=players, seed=seed)
