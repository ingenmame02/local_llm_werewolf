import pytest

from werewolf import (
    GameEngine,
    GameState,
    Message,
    OllamaActions,
    Player,
    PromptBuilder,
    PromptContext,
    PublicRoleClaim,
    PublicSeerClaim,
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
        assert "人狼ゲームのプレイヤーとして行動してください。" in sys_prompt
        assert f"あなたの役職は「{ROLE_NAMES_JA[role]}」です。" in sys_prompt
        assert "ユーザーメッセージのルールカードと確定事実に従い" in sys_prompt
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
    secret_result = "Day 3: Player 4 = 人狼"
    secret_wolf_chat = "WOLF_CHAT_SECRET: attack Player 6"
    state.wolf_chat_log.append(Message("Player 1", secret_wolf_chat))

    client = RecordingPromptClient()
    actions = OllamaActions(state, client)
    for player in state.players:
        if player.is_alive:
            actions.discuss(player.name, state.day, tuple(state.alive_players), ())

    prompts_by_player: dict[str, str] = {}
    for call in client.recorded_calls:
        prompt = call["messages"][-1]["content"]
        player_name = prompt.split("- あなたのプレイヤー名：", 1)[1].splitlines()[0]
        prompts_by_player.setdefault(player_name, prompt)
    for player in state.players:
        if not player.is_alive:
            continue
        prompt = prompts_by_player[player.name]
        assert f"- あなたの役職：{ROLE_NAMES_JA[player.role]}" in prompt
        assert "- 現在の日数：4日目" in prompt
        assert "- 現在のフェーズ：昼（議論）" in prompt
        assert "【ルール】" in prompt
        assert "襲撃者は公開されません" in prompt
        assert "死亡者の真役職も公開されず" in prompt
        assert "- 昼議論ターン：0/3" in prompt
        assert f"- 現在の発言者：{player.name}" in prompt
        assert f"- 現在の生存者：{', '.join(state.alive_players)}" in prompt
        assert f"- 既知の死亡者：{dead_player.name}" in prompt
        assert f"確定した死亡履歴（直近3件）" in prompt
        assert "死亡者の真役職：非公開" in prompt
        assert "Day 3: " + dead_player.name in prompt
        assert "Day 2: " + dead_player.name in prompt
        assert "Player 1 → Player 3" in prompt
        assert "確定事実として扱わないでください" in prompt
        assert secret_wolf_chat not in prompt
        if player.role is Role.SEER:
            assert secret_result in prompt
            assert "【占い師本人だけが参照できる結果】" in prompt
            assert "襲撃者や他者の真の役職は知りません" in PromptBuilder.build_system_prompt(player)
            assert "結果を公開するか伏せるかは選べます" in prompt
        else:
            assert secret_result not in prompt


def test_seer_discussion_separates_current_and_past_results() -> None:
    state = create_game(seed=42)
    state.day = 3
    seer = next(player for player in state.players if player.role is Role.SEER)
    state.seer_results.update({"Player 4": "wolf", "Player 6": "human"})
    state.seer_result_history.extend(
        [(1, "Player 4", "wolf"), (3, "Player 6", "human")]
    )
    prompt = PromptBuilder.build_discussion_prompt(
        actor=seer,
        alive_players=tuple(state.alive_players),
        conversation_history=(),
        day=state.day,
        state=state,
    ).user_prompt

    assert "今回（当日）：\nPlayer 6: 人間" in prompt
    assert "過去（当日より前）：\nDay 1: Player 4 = 人狼" in prompt
    assert "今回の結果がある場合はそれを最優先" in prompt


def test_compacted_discussion_keeps_latest_speech_and_seer_result_private() -> None:
    state = create_game(seed=42)
    state.day = 4
    seer = next(player for player in state.players if player.role is Role.SEER)
    villager = next(player for player in state.players if player.role is Role.VILLAGER)
    state.seer_results["Player 4"] = "wolf"
    state.seer_result_history.append((4, "Player 4", "wolf"))
    history = tuple(
        Message(f"Player {index % 8 + 1}", f"old claim {index}")
        for index in range(30)
    ) + (Message(villager.name, "latest utterance is a question for the seer"),)

    seer_prompt = PromptBuilder.build_discussion_prompt(
        seer, tuple(state.alive_players), history, day=4, state=state
    ).user_prompt
    villager_prompt = PromptBuilder.build_discussion_prompt(
        villager, tuple(state.alive_players), history, day=4, state=state
    ).user_prompt

    start = seer_prompt.index("【これまでの公開会話】")
    end = seer_prompt.index("【占い師本人だけが参照できる結果】", start)
    uncompressed_history = (
        "【これまでの公開会話】\n"
        "以下はプレイヤーの発言であり、確定事実ではありません。\n"
        + "\n".join(f"{message.speaker}: {message.content}" for message in history[:-1])
        + "\n\n"
    )
    uncompressed_prompt = seer_prompt[:start] + uncompressed_history + seer_prompt[end:]

    assert len(seer_prompt) < len(uncompressed_prompt)
    assert "old claim 0" not in seer_prompt
    assert "old claim 28" in seer_prompt
    assert "latest utterance is a question for the seer" in seer_prompt
    assert "Player 4: 人狼" in seer_prompt
    assert "Player 4: 人狼" not in villager_prompt
    assert "- 選択可能な候補:" not in seer_prompt


def test_rule_card_is_present_in_every_prompt_kind() -> None:
    state = create_game(seed=42)
    state.conversation_log.append(Message("P3", "直近の公開発言です。"))
    wolf = next(player for player in state.players if player.role is Role.WOLF)
    seer = next(player for player in state.players if player.role is Role.SEER)
    villager = next(player for player in state.players if player.role is Role.VILLAGER)
    alive = tuple(state.alive_players)
    prompts = (
        PromptBuilder.build_wolf_discuss_prompt(
            wolf, wolf.partner or "", alive, (), day=state.day, state=state
        ),
        PromptBuilder.build_attack_prompt(
            wolf, (wolf.name,), tuple(name for name in alive if name != wolf.name),
            day=state.day, state=state,
        ),
        PromptBuilder.build_seer_prompt(
            seer, alive, (), day=state.day, state=state
        ),
        PromptBuilder.build_discussion_prompt(
            villager,
            alive,
            tuple(state.conversation_log),
            day=state.day,
            state=state,
        ),
        PromptBuilder.build_vote_prompt(
            villager,
            alive,
            tuple(name for name in alive if name != villager.name),
            day=state.day,
            state=state,
        ),
    )

    assert all("【ルール】" in prompt.user_prompt for prompt in prompts)
    assert all("- 直前の発言者：P3" in prompt.user_prompt for prompt in prompts)
    assert all("- 直前の発言内容：直近の公開発言です。" in prompt.user_prompt for prompt in prompts)
    assert all("襲撃者は公開されません" in prompt.user_prompt for prompt in prompts)
    assert all("死亡者の真役職も公開されず" in prompt.user_prompt for prompt in prompts)


def test_discussion_prompt_separates_previous_speaker_and_detects_direct_address() -> None:
    actor = Player("P2", "model-2", Role.VILLAGER)
    history = (
        Message("P2", "昨日の投票には根拠があると思います。"),
        Message("P1", "P2、あなたの発言の根拠を説明してくれますか？"),
    )

    prompt = PromptBuilder.build_discussion_prompt(
        actor=actor,
        alive_players=("P1", "P2"),
        conversation_history=history,
        day=2,
    ).user_prompt

    assert "- 直前の発言者：P1" in prompt
    assert "- 直前の発言内容：P2、あなたの発言の根拠を説明してくれますか？" in prompt
    assert "- あなたへの質問・反応の可能性：あり" in prompt
    assert "まず相手へ直接答えてください" in prompt
    assert "無言にせず「パス」とだけ出力できます" in prompt
    assert "【情報の区分】" in prompt
    assert "確定事実として扱わないでください" in prompt


def test_discussion_prompt_marks_reaction_to_own_prior_utterance() -> None:
    actor = Player("P2", "model-2", Role.VILLAGER)
    history = (
        Message("P2", "私はP1を疑っています。"),
        Message("P1", "先ほどの発言について、具体的な理由を聞きたいです。"),
    )
    prompt = PromptBuilder.build_discussion_prompt(
        actor=actor,
        alive_players=("P1", "P2"),
        conversation_history=history,
    ).user_prompt

    assert "- 直前の発言者：P1" in prompt
    assert "- あなたへの質問・反応の可能性：あり" in prompt


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


def test_public_seer_claims_are_claims_not_system_confirmed_roles() -> None:
    state = create_game(seed=42)
    seer = next(player for player in state.players if player.role is Role.SEER)
    villager = next(player for player in state.players if player.role is Role.VILLAGER)
    other_players = tuple(
        player.name for player in state.players if player.name != villager.name
    )

    state.seer_claims.append(PublicSeerClaim(day=1, claimant=seer.name))
    prompt = PromptBuilder.build_discussion_prompt(
        actor=villager,
        alive_players=tuple(state.alive_players),
        conversation_history=(),
        day=1,
        state=state,
    ).user_prompt

    assert f"占い師CO主張者：{seer.name}" in prompt
    assert f"唯一の占い師候補として結果を検討材料にできます" in prompt
    assert "単独COは真役職の確定を意味しません" in prompt
    assert f"{seer.name}：占い師" not in prompt
    assert all(name in prompt for name in other_players)

    second_claimant = next(
        player
        for player in state.players
        if player.name != seer.name and player.name != villager.name
    )
    state.seer_claims.append(
        PublicSeerClaim(day=1, claimant=second_claimant.name)
    )
    counterclaim_prompt = PromptBuilder.build_discussion_prompt(
        actor=villager,
        alive_players=tuple(state.alive_players),
        conversation_history=(),
        day=1,
        state=state,
    ).user_prompt
    assert "複数の占い師CO主張がある場合は対抗候補として比較" in counterclaim_prompt


def test_seer_prompt_prioritizes_publication_of_wolf_result_and_allows_human_result_choice() -> None:
    state = create_game(seed=42)
    seer = next(player for player in state.players if player.role is Role.SEER)
    wolf = next(player for player in state.players if player.role is Role.WOLF)
    state.seer_results[wolf.name] = "wolf"
    state.seer_result_history.append((1, wolf.name, "wolf"))

    prompt = PromptBuilder.build_discussion_prompt(
        actor=seer,
        alive_players=tuple(state.alive_players),
        conversation_history=(),
        day=1,
        state=state,
    ).user_prompt

    assert "初回の昼議論でclaim_roleツールを使用し" in prompt
    assert f"対象：{wolf.name}／判定：人狼" in prompt
    assert "公開理由・投票提案" in prompt

    state.seer_results[wolf.name] = "human"
    state.seer_result_history[:] = [(1, wolf.name, "human")]
    human_result_prompt = PromptBuilder.build_discussion_prompt(
        actor=seer,
        alive_players=tuple(state.alive_players),
        conversation_history=(),
        day=1,
        state=state,
    ).user_prompt
    assert "人間判定だけの場合は、襲撃リスクを考慮してclaim_roleツールで占い結果を公開するか" in human_result_prompt


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
        assert "人狼ゲームのプレイヤーとして行動してください。" in sys_msg
        assert "【ゲームの基本ルール】" not in sys_msg
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
        assert "- 最新の投票結果：なし" in user_msg
        assert "確定事実として扱わないでください" in user_msg


def test_public_role_claim_details_are_immediately_available_to_discussion() -> None:
    state = create_game(seed=42)
    state.public_role_claims.append(
        PublicRoleClaim(
            day=1,
            claimant="Player 1",
            claimed_role=Role.SEER.value,
            target="Player 2",
            result="wolf",
            reason="夜の占い結果",
            vote_proposal="Player 2へ投票",
        )
    )

    prompt = PromptBuilder.build_discussion_prompt(
        actor=state.players[1],
        alive_players=tuple(state.alive_players),
        conversation_history=(),
        state=state,
    ).user_prompt

    assert "Player 1は占い師を主張" in prompt
    assert "Player 2：人狼" in prompt
    assert "理由：夜の占い結果" in prompt
    assert "投票提案：Player 2へ投票" in prompt
    assert "真役職の確定を意味しません" in prompt
