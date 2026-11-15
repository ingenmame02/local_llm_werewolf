import random
from pathlib import Path

from .config import load_player_configs, PlayerConfig
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
    response_language: str | None = None,
) -> GameState:
    player_configs = load_player_configs(config_path)
    # TASK-038: CLI --language引数でresponse_languageを上書き
    # PlayerConfigはfrozen dataclassであるため、新しいオブジェクトを作成して上書きする
    if response_language is not None:
        player_configs = tuple(
            PlayerConfig(
                name=config.name,
                model=config.model,
                num_ctx=config.num_ctx,
                max_num_ctx=config.max_num_ctx,
                recommended_num_ctx=config.recommended_num_ctx,
                context_strategy=config.context_strategy,
                timeout=config.timeout,
                num_predict=config.num_predict,
                response_language=response_language,
                num_predict_by_action=config.num_predict_by_action,
                thinking=config.thinking,
                thinking_by_action=config.thinking_by_action,
                personality_profile=config.personality_profile,
                claim_coaching=config.claim_coaching,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repeat_penalty,
            )
            for config in player_configs
        )
    roles = list(INITIAL_ROLES)
    random.Random(seed).shuffle(roles)

    players = [
        Player(
            name=config.name,
            model=config.model,
            role=role,
            num_ctx=config.num_ctx,
            max_num_ctx=config.max_num_ctx,
            recommended_num_ctx=config.recommended_num_ctx,
            context_strategy=config.context_strategy,
            timeout=config.timeout,
            num_predict=config.num_predict,
            response_language=config.response_language,
            num_predict_by_action=config.num_predict_by_action,
            thinking=config.thinking,
            thinking_by_action=config.thinking_by_action,
            personality_profile=config.personality_profile,
            claim_coaching=config.claim_coaching,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
        )
        for config, role in zip(player_configs, roles, strict=True)
    ]
    wolves = [player for player in players if player.role is Role.WOLF]
    for player in wolves:
        player.partner = next(wolf.name for wolf in wolves if wolf is not player)

    state = GameState(players=players, seed=seed)
    for player in wolves:
        if player.name in state.player_self_memories:
            state.player_self_memories[player.name].partner = player.partner
    return state
