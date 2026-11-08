from werewolf import GameEngine, GameState, Message, MockActions, Phase, Player, Role
from werewolf.prompts import PromptBuilder


def _state() -> GameState:
    state = GameState(
        players=[
            Player("Player 1", "model-1", Role.VILLAGER),
            Player("Player 2", "model-2", Role.VILLAGER),
            Player("Player 3", "model-3", Role.VILLAGER),
        ],
        phase=Phase.DAY_DISCUSSION,
    )
    state.discussion_turn = 2
    return state


def test_round_transcript_memory_includes_all_current_round_messages() -> None:
    state = _state()
    state.current_round_transcript = [
        Message("Player 1", "Player 2さんの投票理由を確認します。"),
        Message("Player 2", "Player 1さん、私はPlayer 3を疑います。"),
    ]
    prompt = PromptBuilder.build_discussion_prompt(
        actor=state.players[2],
        alive_players=tuple(state.alive_players),
        conversation_history=tuple(state.conversation_log),
        day=1,
        state=state,
        context_strategy="round_transcript_memory",
    )

    assert "Player 2さんの投票理由を確認します。" in prompt.user_prompt
    assert "Player 1さん、私はPlayer 3を疑います。" in prompt.user_prompt
    assert prompt.diagnostics["round_transcript"] is True
    assert prompt.diagnostics["transcript_event_count"] == 2


def test_integrity_alert_is_kept_without_exposing_leaked_content() -> None:
    state = _state()
    state.add_integrity_alert(
        speaker="Player 2",
        category="strategy_leak",
        speech_id=4,
        summary="公開前に秘匿性ガードが検出したため内容は非公開",
    )
    state.refresh_reasoning_memos()
    prompt = PromptBuilder.build_discussion_prompt(
        actor=state.players[0],
        alive_players=tuple(state.alive_players),
        conversation_history=tuple(state.conversation_log),
        day=1,
        state=state,
        context_strategy="round_transcript_memory",
    )

    assert "strategy_leak" in prompt.user_prompt
    assert "内容は非公開" in prompt.user_prompt
    assert "INTEGRITY_ALERTS" in prompt.user_prompt


def test_discussion_round_is_mirrored_separately_from_full_history() -> None:
    state = _state()
    GameEngine(state, MockActions()).run_discussion()

    assert len(state.conversation_log) == 9
    assert len(state.current_round_transcript) == 3
    assert all(message.speaker != "system" for message in state.current_round_transcript)
    assert state.current_round_transcript == state.conversation_log[-3:]


def test_discussion_messages_are_emitted_when_appended() -> None:
    state = _state()
    events: list[dict[str, object]] = []

    GameEngine(state, MockActions(), event_callback=events.append).run_discussion()

    messages = [event for event in events if event["type"] == "discussion_message"]
    assert len(messages) == 9
    assert all(event["private"] is False for event in messages)
    assert [event["content"] for event in messages] == [
        message.content for message in state.conversation_log
    ]


def test_secret_leak_alert_does_not_store_the_secret_text() -> None:
    state = _state()

    class LeakingActions(MockActions):
        def discuss(self, player, day, alive, history):
            if player == "Player 1":
                return "私が占い師を騙っている設定上、Player 2を疑います。"
            return "公開情報だけで判断します。"

        def retry_after_secret_leak(
            self, player, day, alive, history, leak_categories
        ):
            return "公開情報だけで判断します。"

    GameEngine(state, LeakingActions()).run_discussion()

    assert state.integrity_alerts
    assert any(alert.category == "strategy_leak" for alert in state.integrity_alerts)
    assert all(
        "騙っている設定上" not in alert.summary
        for alert in state.integrity_alerts
    )
    assert all(
        "私が占い師を騙っている設定上" not in message.content
        for message in state.conversation_log
    )
