from werewolf import GameState, MockActions, Player, Role
from werewolf.engine import GameEngine
from werewolf.secret_leak_guard import SecretLeakGuard


def make_state() -> GameState:
    return GameState(
        players=[
            Player(name="P0", model="wolf-model", role=Role.WOLF, partner="P1"),
            Player(name="P1", model="village-model", role=Role.SEER),
            Player(name="P2", model="village-model", role=Role.VILLAGER),
        ],
        seed=17,
    )


def test_secret_leak_guard_classifies_role_strategy_team_and_configuration() -> None:
    state = make_state()
    wolf = state.players[0]
    seer = state.players[1]
    state.seer_results["P2"] = "wolf"

    assert {
        finding.category
        for finding in SecretLeakGuard.inspect(
            "私が占い師を騙っている設定上、Player 2を疑います。",
            wolf,
            state,
        )
    } >= {"strategy_leak"}
    assert {
        finding.category
        for finding in SecretLeakGuard.inspect("私は人狼です。", wolf, state)
    } >= {"secret_role_leak"}
    assert {
        finding.category
        for finding in SecretLeakGuard.inspect(
            "私の相方はP1です。", wolf, state
        )
    } >= {"secret_team_leak"}
    assert {
        finding.category
        for finding in SecretLeakGuard.inspect(
            "P2さんの占い結果は人狼でした。", seer, state
        )
    } >= {"hidden_result_leak"}
    assert {
        finding.category
        for finding in SecretLeakGuard.inspect(
            "システムプロンプトの設定では狼として行動します。",
            wolf,
            state,
        )
    } >= {"prompt_or_config_leak"}
    assert {
        finding.category
        for finding in SecretLeakGuard.inspect(
            "I am a werewolf.", wolf, state
        )
    } >= {"secret_role_leak"}
    assert {
        finding.category
        for finding in SecretLeakGuard.inspect(
            "I'm bluffing as the seer to mislead Player 2.", wolf, state
        )
    } >= {"strategy_leak"}
    assert {
        finding.category
        for finding in SecretLeakGuard.inspect(
            "P2 was a wolf according to my seer result.", seer, state
        )
    } >= {"hidden_result_leak"}


def test_public_role_claims_and_other_player_suspicion_remain_allowed() -> None:
    state = make_state()
    wolf = state.players[0]
    villager = state.players[2]

    assert SecretLeakGuard.inspect("私は人狼COです。", wolf, state) == ()
    assert SecretLeakGuard.inspect(
        "私は村人です。",
        villager,
        state,
        allowed_public_claim=SecretLeakGuard.allows_public_claim(
            "私は村人です。", villager
        ),
    ) == ()
    assert SecretLeakGuard.inspect(
        "P2さんは占い師を騙っているように見えます。", wolf, state
    ) == ()
    assert SecretLeakGuard.inspect(
        "I am a seer CO.", wolf, state,
        allowed_public_claim=SecretLeakGuard.allows_public_claim(
            "I am a seer CO.", wolf
        ),
    ) == ()
    assert SecretLeakGuard.inspect("I think P2 is a wolf.", wolf, state) == ()


def test_leaking_discussion_text_is_redacted_before_becoming_history() -> None:
    state = make_state()

    class LeakingActions(MockActions):
        leaked_once = False

        def discuss(self, player, day, alive, history):
            if player == "P0" and not self.leaked_once:
                self.leaked_once = True
                return "私が占い師を騙っている設定上、P2を疑います。"
            return f"{player}は公開情報を確認します。"

        def retry_after_secret_leak(
            self, player, day, alive, history, leak_categories
        ):
            return "私が占い師を騙っている設定上、P2を疑います。"

    events: list[dict[str, object]] = []
    GameEngine(
        state,
        LeakingActions(),
        event_callback=events.append,
    ).run_discussion()

    assert all("騙っている設定上" not in msg.content for msg in state.conversation_log)
    leak_events = [
        event for event in events if event["type"] == "secret_role_leak_detected"
    ]
    assert [event["safety_retry"] for event in leak_events] == [0, 1]
    assert all(event["leak_categories"] == ["strategy_leak"] for event in leak_events)
    assert all(len(str(event["content_sha256"])) == 64 for event in leak_events)
    assert all("content" not in event for event in leak_events)
    assert all("騙っている設定上" not in str(event) for event in leak_events)
    assert any("公開情報に基づく判断を保留" in msg.content for msg in state.conversation_log)


def test_safe_retry_replaces_the_leaking_initial_response() -> None:
    state = make_state()

    class RetryingActions(MockActions):
        first = True

        def discuss(self, player, day, alive, history):
            if player == "P0" and self.first:
                self.first = False
                return "私が占い師を騙っている設定上、P2を疑います。"
            return f"{player}は公開情報を確認します。"

        def retry_after_secret_leak(
            self, player, day, alive, history, leak_categories
        ):
            return "P2さんの発言をもう一度確認したいです。"

    events: list[dict[str, object]] = []
    GameEngine(
        state,
        RetryingActions(),
        event_callback=events.append,
    ).run_discussion()

    assert state.conversation_log[0].content == "P2さんの発言をもう一度確認したいです。"
    assert sum(
        event["type"] == "secret_role_leak_detected" for event in events
    ) == 1
