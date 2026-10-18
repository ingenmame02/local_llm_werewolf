import pytest

from werewolf import (
    GameEngine,
    GameState,
    Message,
    OllamaActions,
    Player,
    PromptBuilder,
    PromptContext,
    ROLE_DESCRIPTIONS_JA,
    ROLE_NAMES_JA,
    Role,
    create_game,
)


class RecordingPromptClient:
    """Mock OllamaClient that records every prompt context received."""

    def __init__(self, canned_response: str = "投票：Player 2\n襲撃対象：Player 2\n占い対象：Player 2") -> None:
        self.canned_response = canned_response
        self.recorded_calls: list[dict[str, object]] = []

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        self.recorded_calls.append({"model": model, "messages": messages})
        return self.canned_response


def test_system_prompt_common_structure_and_role_description() -> None:
    for role in Role:
        player = Player(name="Alice", model="dummy-model", role=role)
        sys_prompt = PromptBuilder.build_system_prompt(player)

        # 全LLMで共通システムプロンプトの骨格が含まれること
        assert "あなたは人狼ゲームに参加しているプレイヤーです。" in sys_prompt
        assert f"あなたの役職は「{ROLE_NAMES_JA[role]}」です。" in sys_prompt
        assert "【ゲームの基本ルール】" in sys_prompt
        assert "【あなたの役職について】" in sys_prompt
        assert ROLE_DESCRIPTIONS_JA[role] in sys_prompt


def test_confidentiality_other_players_role_never_leaks_in_discussion_and_vote() -> None:
    state = create_game(seed=42)
    # roles check
    wolves = [p for p in state.players if p.role is Role.WOLF]
    seer = next(p for p in state.players if p.role is Role.SEER)
    madman = next(p for p in state.players if p.role is Role.MADMAN)
    villagers = [p for p in state.players if p.role is Role.VILLAGER]

    alive_players = tuple(state.alive_players)
    conv_history = (
        Message("Player 1", "おはようございます。怪しい人はいませんか？"),
        Message("Player 2", "まだ手がかりがありません。"),
    )

    # 村人・狂人・占い師の昼発言プロンプトを検査
    for player in villagers + [madman, seer]:
        ctx = PromptBuilder.build_discussion_prompt(
            actor=player,
            alive_players=alive_players,
            conversation_history=conv_history,
        )
        # 他プレイヤーの役職名が含まれていないか
        # 狼のプレイヤー名が「人狼」としてプロンプトに出てこないか
        for wolf in wolves:
            assert f"{wolf.name}：人狼" not in ctx.user_prompt
            assert f"{wolf.name}（人狼）" not in ctx.user_prompt
            assert f"人狼: {wolf.name}" not in ctx.user_prompt

        # システムプロンプトには自役職のみ
        assert f"あなたの役職は「{ROLE_NAMES_JA[player.role]}」です。" in ctx.system_prompt
        for other_role in Role:
            if other_role != player.role:
                assert f"あなたの役職は「{ROLE_NAMES_JA[other_role]}」です。" not in ctx.system_prompt


def test_wolf_consultation_log_isolated_to_wolves_only() -> None:
    state = create_game(seed=42)
    wolves = [p for p in state.players if p.role is Role.WOLF]
    non_wolves = [p for p in state.players if p.role is not Role.WOLF]

    wolf_chat = (
        Message(wolves[0].name, "今夜はPlayer 1を襲撃しましょう。"),
        Message(wolves[1].name, "賛成です。襲撃対象：Player 1"),
    )
    secret_text = "今夜はPlayer 1を襲撃しましょう。"

    # 非人狼プレイヤーのプロンプト生成
    for player in non_wolves:
        disc_ctx = PromptBuilder.build_discussion_prompt(
            actor=player,
            alive_players=tuple(state.alive_players),
            conversation_history=tuple(state.conversation_log),
        )
        vote_ctx = PromptBuilder.build_vote_prompt(
            actor=player,
            alive_players=tuple(state.alive_players),
            valid_candidates=tuple(p for p in state.alive_players if p != player.name),
        )
        assert secret_text not in disc_ctx.user_prompt
        assert secret_text not in disc_ctx.system_prompt
        assert secret_text not in vote_ctx.user_prompt
        assert secret_text not in vote_ctx.system_prompt

        # 非人狼がwolf_discussを呼ぼうとすると拒否される
        with pytest.raises(PermissionError, match="is not a wolf"):
            PromptBuilder.build_wolf_discuss_prompt(
                actor=player,
                partner_name="Nobody",
                alive_players=tuple(state.alive_players),
                wolf_chat_history=wolf_chat,
            )

    # 人狼には相方名と相談ログが渡る
    wolf_ctx = PromptBuilder.build_wolf_discuss_prompt(
        actor=wolves[0],
        partner_name=wolves[1].name,
        alive_players=tuple(state.alive_players),
        wolf_chat_history=wolf_chat,
    )
    assert wolves[1].name in wolf_ctx.user_prompt
    assert secret_text in wolf_ctx.user_prompt


def test_seer_results_isolated_to_seer_only() -> None:
    state = create_game(seed=42)
    seer = next(p for p in state.players if p.role is Role.SEER)
    others = [p for p in state.players if p is not seer]

    seer_secret = "Player 4: wolf"
    seer_results = {"Player 4": "wolf"}

    # 占い師本人のプロンプトには占い結果が含まれる
    seer_ctx = PromptBuilder.build_seer_prompt(
        actor=seer,
        alive_players=tuple(state.alive_players),
        seer_results=seer_results,
    )
    assert seer_secret in seer_ctx.user_prompt

    # 他プレイヤーがbuild_seer_promptを呼ぶと例外
    for other in others:
        with pytest.raises(PermissionError, match="is not a seer"):
            PromptBuilder.build_seer_prompt(
                actor=other,
                alive_players=tuple(state.alive_players),
                seer_results=seer_results,
            )

        # 昼発言や投票にも占い師の秘密結果は混入しない
        disc_ctx = PromptBuilder.build_discussion_prompt(
            actor=other,
            alive_players=tuple(state.alive_players),
            conversation_history=tuple(state.conversation_log),
        )
        assert seer_secret not in disc_ctx.user_prompt
        assert seer_secret not in disc_ctx.system_prompt


def test_discussion_prompts_include_identity_facts_and_role_scoped_seer_results() -> None:
    state = create_game(seed=42)
    state.day = 4
    dead_player = next(player for player in state.players if player.role is Role.VILLAGER)
    state.mark_dead(dead_player.name)
    state.attack_deaths.append((3, dead_player.name))
    state.executions.append((2, dead_player.name))
    state.vote_history.append((2, False, {"Player 1": "Player 3"}))
    state.seer_results["Player 4"] = "wolf"
    state.seer_result_history.append((3, "Player 4", "wolf"))
    secret_result = "Day 3: Player 4 = wolf"
    secret_wolf_chat = "WOLF_CHAT_SECRET: attack Player 6"
    state.wolf_chat_log.append(Message("Player 1", secret_wolf_chat))

    client = RecordingPromptClient()
    actions = OllamaActions(state, client)
    for player in state.players:
        if player.is_alive:
            actions.discuss(player.name, state.day, tuple(state.alive_players), ())

    prompts_by_player = {
        call["messages"][-1]["content"].split("- あなたのプレイヤー名：", 1)[1]
        .splitlines()[0]: call["messages"][-1]["content"]
        for call in client.recorded_calls
    }
    for player in state.players:
        if not player.is_alive:
            continue
        prompt = prompts_by_player[player.name]
        assert f"- あなたの役職：{ROLE_NAMES_JA[player.role]}" in prompt
        assert "- 現在の日数：4日目" in prompt
        assert "- 現在のフェーズ：昼（議論）" in prompt
        assert f"- 現在の生存者：{', '.join(state.alive_players)}" in prompt
        assert f"- 既知の死亡者：{dead_player.name}" in prompt
        assert "Day 3: " + dead_player.name in prompt
        assert "Day 2: " + dead_player.name in prompt
        assert "Player 1 → Player 3" in prompt
        assert "確定事実として扱わないでください" in prompt
        assert secret_wolf_chat not in prompt
        if player.role is Role.SEER:
            assert secret_result in prompt
        else:
            assert secret_result not in prompt


def test_dead_players_cannot_receive_prompts() -> None:
    state = create_game(seed=42)
    dead_player = state.players[0]
    state.mark_dead(dead_player.name)
    assert not dead_player.is_alive

    with pytest.raises(ValueError, match="cannot act"):
        PromptBuilder.build_discussion_prompt(
            actor=dead_player,
            alive_players=tuple(state.alive_players),
            conversation_history=(),
        )

    with pytest.raises(ValueError, match="cannot act"):
        PromptBuilder.build_vote_prompt(
            actor=dead_player,
            alive_players=tuple(state.alive_players),
            valid_candidates=("Player 2",),
        )

    if dead_player.role is Role.WOLF:
        with pytest.raises(ValueError, match="cannot act"):
            PromptBuilder.build_attack_prompt(
                actor=dead_player,
                wolves=(dead_player.name,),
                valid_candidates=("Player 2",),
            )
    elif dead_player.role is Role.SEER:
        with pytest.raises(ValueError, match="cannot act"):
            PromptBuilder.build_seer_prompt(
                actor=dead_player,
                alive_players=tuple(state.alive_players),
                seer_results={},
            )


def test_eight_players_prompt_snapshot_inspection() -> None:
    """8プレイヤー全員について、実際の呼び出しメッセージをスナップショット検査する。"""
    state = create_game(seed=42)
    client = RecordingPromptClient(canned_response="投票：Player 2\n襲撃対象：Player 2\n占い対象：Player 2")
    actions = OllamaActions(state, client)

    # 1. 夜の相談
    wolves = [p for p in state.players if p.role is Role.WOLF]
    actions.wolf_discuss(wolves[0].name, wolves[1].name, ())

    # 2. 襲撃
    attack_candidates = tuple(p.name for p in state.players if p.role is not Role.WOLF)
    actions.choose_attack(tuple(w.name for w in wolves), attack_candidates)

    # 3. 占い
    seer = next(p for p in state.players if p.role is Role.SEER)
    seer_candidates = tuple(state.alive_players)
    actions.choose_seer_target(seer.name, seer_candidates, ())

    # 4. 昼の議論 (8人全員)
    for p in state.players:
        actions.discuss(p.name, 1, tuple(state.alive_players), ())

    # 5. 投票 (8人全員)
    for p in state.players:
        actions.vote(p.name, tuple(state.alive_players), 1, runoff=False)

    # 全てのLLM呼び出しを検査
    calls = client.recorded_calls
    assert len(calls) >= 1 + 1 + 1 + 8 + 8  # 19 calls minimum

    for call in calls:
        messages = call["messages"]
        # 全呼び出しに system と user が付与されていること
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

        sys_msg = next(m["content"] for m in messages if m["role"] == "system")
        assert "あなたは人狼ゲームに参加しているプレイヤーです。" in sys_msg
        assert "【ゲームの基本ルール】" in sys_msg
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "【ゲームシステムが確定した事実】" in user_msg
        player_name = user_msg.split("- あなたのプレイヤー名：", 1)[1].splitlines()[0]
        assert player_name in state.alive_players
        assert any(
            f"- あなたの役職：{role_name}" in user_msg
            for role_name in ROLE_NAMES_JA.values()
        )
        assert "- 現在の日数：1日目" in user_msg
        assert "- 現在のフェーズ：" in user_msg
        assert f"- 現在の生存者：{', '.join(state.alive_players)}" in user_msg
        assert "- 既知の死亡者：なし" in user_msg
        assert "- 過去の投票結果：" in user_msg
        assert "確定事実として扱わないでください" in user_msg
