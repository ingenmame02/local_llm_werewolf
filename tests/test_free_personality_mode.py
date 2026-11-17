"""TASK-044の回帰テスト（LL不使用）: 自由発言モード（personality_profile="free"）。

自由発言モードは話し方の preset（個性プロファイル）を廃し、口調・構成・文量・
注目点をモデル自身の判断に任せる。役職の目的・行動制約・確定事実・出力形式・
秘匿情報の禁は preset 適用時と完全に同じでなければならない。
"""
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest

from werewolf.cli import build_parser
from werewolf.config import (
    FREE_PERSONALITY_PROFILE,
    PERSONALITY_CHOICE_KEYS,
    PERSONALITY_PROFILES,
    PlayerConfig,
    is_free_personality,
    load_player_configs,
    normalize_personality_profile,
)
from werewolf.models import Message, Role
from werewolf.prompts import FREE_PERSONALITY_SYSTEM_BLOCK, PromptBuilder
from werewolf.setup import create_game

CONFIG_PATH = "config/players.json"
FREE_CONFIG_PATH = "config/players_free.json"


def _state(personality: str | None = None, response_language: str = "日本語"):
    return create_game(
        seed=1001,
        config_path=CONFIG_PATH,
        response_language=response_language,
        personality_profile=personality,
    )


def _player(state, name: str):
    return next(player for player in state.players if player.name == name)


def _alive(state) -> tuple[str, ...]:
    return tuple(player.name for player in state.players if player.is_alive)


def _preset_block(profile: str) -> str:
    """preset 適用時のシステムプロンプト末尾（個性プロファイルブロック）。"""
    return (
        "\n\n[Personality Profile (Subtle Speaking Style Tendency)]\n"
        f"{PERSONALITY_PROFILES[profile]}\n"
        "Reflect your model's reasoning ability and naturally develop discussion based on this style. "
        "This has lower priority than discussion quality, fact consistency, answering questions, and format instructions. "
        "Do not force extra questions or explanations; apply only where natural. "
        "Do not alter role goals, win conditions, action constraints, confirmed facts, or output format."
    )


def _discussion_ctx(state, name: str):
    actor = _player(state, name)
    history = tuple(state.conversation_log) + (
        Message(speaker="Player 2", content="Player 3 の投票理由を確認したい。"),
    )
    return PromptBuilder.build_discussion_prompt(
        actor=actor,
        alive_players=_alive(state),
        conversation_history=history,
        day=1,
        state=state,
        include_claim=True,
    )


def test_normalize_accepts_profiles_aliases_and_none() -> None:
    assert normalize_personality_profile("analytical", "profile") == "analytical"
    assert normalize_personality_profile("Analytical", "profile") == "analytical"
    assert normalize_personality_profile(FREE_PERSONALITY_PROFILE, "profile") == "free"
    for alias in ("none", "null", "off", "natural", "custom", ""):
        assert normalize_personality_profile(alias, "profile") == "free"
    assert normalize_personality_profile(None, "profile") == "free"
    assert is_free_personality("free") is True
    assert is_free_personality("analytical") is False
    assert is_free_personality(None) is False
    assert PERSONALITY_CHOICE_KEYS[-1] == "free"
    assert set(PERSONALITY_PROFILES) <= set(PERSONALITY_CHOICE_KEYS)


def test_normalize_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="personality_profile"):
        normalize_personality_profile("aggressive", "Player personality_profile")
    with pytest.raises(ValueError, match="free"):
        normalize_personality_profile(123, "Player personality_profile")


def test_player_config_validates_personality_profile() -> None:
    with pytest.raises(ValueError, match="personality_profile"):
        PlayerConfig(name="Player 1", model="m", personality_profile="unknown")
    assert (
        PlayerConfig(
            name="Player 1", model="m", personality_profile=FREE_PERSONALITY_PROFILE
        ).personality_profile
        == "free"
    )


def test_load_player_configs_accepts_free_and_null(tmp_path: Path) -> None:
    entries = [
        {
            "name": f"Player {index}",
            "model": f"model-{index}",
            "personality_profile": profile,
        }
        for index, profile in enumerate(
            [None, "free", "none", "analytical", "questioning", "cautious", "consensus", "intuitive"],
            start=1
        )
    ]
    path = tmp_path / "players.json"
    path.write_text(json.dumps({"players": entries}), encoding="utf-8")
    configs = load_player_configs(path)
    assert [config.personality_profile for config in configs[:3]] == [
        "free",
        "free",
        "free",
    ]

    bad_entries = [
        {
            "name": f"Player {index}",
            "model": f"model-{index}",
            "personality_profile": "hostile",
        }
        for index in range(1, 9)
    ]
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"players": bad_entries}), encoding="utf-8")
    with pytest.raises(ValueError, match="personality_profile at index 0"):
        load_player_configs(bad_path)


def test_player_normalizes_personality_aliases_to_free() -> None:
    state = _state(personality="none")
    assert all(player.personality_profile == "free" for player in state.players)


def test_free_system_prompt_has_no_preset_block() -> None:
    free_actor = _player(_state(personality="free"), "Player 1")
    preset_state = _state()
    preset_actor = _player(preset_state, "Player 1")
    assert preset_actor.personality_profile == "analytical"

    free_prompt = PromptBuilder.build_system_prompt(free_actor, include_personality=True)
    preset_prompt = PromptBuilder.build_system_prompt(preset_actor, include_personality=True)
    assert "[Personality Profile" not in free_prompt
    assert "[Expression Freedom" in free_prompt
    assert FREE_PERSONALITY_SYSTEM_BLOCK in free_prompt
    assert "[Personality Profile" in preset_prompt
    assert "[Expression Freedom" not in preset_prompt
    # preset 部分以外は同一（役職・情報・形式指示は変わらない）
    assert (
        free_prompt.replace(FREE_PERSONALITY_SYSTEM_BLOCK, "")
        == preset_prompt.replace(_preset_block("analytical"), "")
    )


def test_free_discussion_prompt_relaxes_body_structure_only() -> None:
    free_ctx = _discussion_ctx(_state(personality="free"), "Player 3")
    preset_ctx = _discussion_ctx(_state(), "Player 3")
    assert "3-5 sentences" not in free_ctx.user_prompt
    assert "3-5 sentences" in preset_ctx.user_prompt
    assert (
        "[BODY] length, structure, and wording are yours to choose"
        in free_ctx.user_prompt
    )
    assert "[Expression Freedom" in free_ctx.system_prompt
    # 出力形式・CO形式・秘匿情報の禁は preset 適用時と同じ（応答言語が日本語なので日本語ブロック）
    for marker in ("[CO]", "[RESULT]", "[BODY]", "Never leave [BODY] empty"):
        assert marker in free_ctx.user_prompt and marker in preset_ctx.user_prompt
    for marker in (
        "Never disclose private information",
        "[秘匿情報の扱い - 公開発言に写さない]",
        "未公開の判定",
    ):
        assert marker in free_ctx.user_prompt and marker in preset_ctx.user_prompt


def test_free_mode_keeps_role_rules_and_private_boundaries() -> None:
    free_state = _state(personality="free")
    preset_state = _state()
    wolf_name = next(p.name for p in free_state.players if p.role is Role.WOLF)
    seer_name = next(p.name for p in free_state.players if p.role is Role.SEER)
    for name in (wolf_name, seer_name):
        free_ctx = _discussion_ctx(free_state, name)
        preset_ctx = _discussion_ctx(preset_state, name)
        assert "[Expression Freedom" in free_ctx.system_prompt
        assert "[Personality Profile" not in free_ctx.system_prompt
        # 役職の説明・視点分離・台帳・形式指示は preset 適用時と同一
        preset_profile = _player(preset_state, name).personality_profile
        assert (
            free_ctx.system_prompt.replace(FREE_PERSONALITY_SYSTEM_BLOCK, "")
            == preset_ctx.system_prompt.replace(_preset_block(preset_profile), "")
        )
        # user_prompt の差異は [BODY] の文量・構成指示の1行だけ（台帳・秘匿情報・形式は同一）
        free_lines = free_ctx.user_prompt.splitlines()
        preset_lines = preset_ctx.user_prompt.splitlines()
        assert len(free_lines) == len(preset_lines)
        differing = [
            (free_line, preset_line)
            for free_line, preset_line in zip(free_lines, preset_lines, strict=True)
            if free_line != preset_line
        ]
        assert len(differing) == 1
        assert differing[0][0].startswith("- [BODY] length, structure")
        assert differing[0][1].startswith("- [BODY] should be 3-5 sentences")

    wolf = _player(free_state, wolf_name)
    partner = wolf.partner
    assert partner is not None
    consult = PromptBuilder.build_wolf_discuss_prompt(
        actor=wolf,
        partner_name=partner,
        alive_players=_alive(free_state),
        wolf_chat_history=(),
        day=2,
        state=free_state,
    )
    assert f"Werewolf Partner: {partner}" in consult.user_prompt
    # 公開議論のシステムプロンプトには相方名を出さない
    assert partner not in _discussion_ctx(free_state, wolf_name).system_prompt


def test_personality_override_preserves_other_player_settings() -> None:
    base_state = _state()
    free_state = _state(personality="free")
    for base_player, free_player in zip(base_state.players, free_state.players):
        assert base_player.personality_profile != "free"
        assert free_player.personality_profile == "free"
        assert free_player.model == base_player.model
        assert free_player.num_ctx == base_player.num_ctx
        assert free_player.num_predict == base_player.num_predict
        assert free_player.num_predict_by_action == base_player.num_predict_by_action
        assert free_player.thinking_by_action == base_player.thinking_by_action
        assert free_player.role is base_player.role


def test_free_config_preset_loads_all_players_free() -> None:
    configs = load_player_configs(FREE_CONFIG_PATH)
    assert len(configs) == 8
    assert all(config.personality_profile == "free" for config in configs)
    default_configs = load_player_configs(CONFIG_PATH)
    assert all(config.personality_profile != "free" for config in default_configs)


def test_cli_parser_accepts_free_personality() -> None:
    parser = build_parser()
    assert parser.parse_args(["--mode", "mock"]).personality is None
    assert (
        parser.parse_args(["--mode", "mock", "--personality", "free"]).personality
        == "free"
    )
    assert (
        parser.parse_args(["--mode", "mock", "--personality", "analytical"]).personality
        == "analytical"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "mock", "--personality", "hostile"])
