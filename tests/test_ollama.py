import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

from werewolf import (
    ActionKind,
    ActionParser,
    GameState,
    OllamaActions,
    OllamaClient,
    OllamaError,
    Player,
    Role,
)


@pytest.mark.parametrize(
    ("response", "action", "expected"),
    [
        ("襲撃対象：Player 2", ActionKind.ATTACK, "Player 2"),
        ("理由です。\n占い対象 : 「Player 3」。", ActionKind.SEER, "Player 3"),
        ("投票: Player 1！", ActionKind.VOTE, "Player 1"),
        ("投票：\n", ActionKind.VOTE, None),
        ("Player 1に投票します。", ActionKind.VOTE, None),
        ("**襲撃対象：Player 4**", ActionKind.ATTACK, "Player 4"),
        ("さっさと終わらせよう。\n\n**襲撃対象：Player 4**", ActionKind.ATTACK, "Player 4"),
        ("**投票**：**Player 1**", ActionKind.VOTE, "Player 1"),
        ("*占い対象*：*Player 3*", ActionKind.SEER, "Player 3"),
    ],
)
def test_parser_extracts_only_labeled_targets(
    response: str, action: ActionKind, expected: str | None
) -> None:
    assert ActionParser.parse(response, action) == expected


def test_parser_validates_candidates_and_can_reject_self_vote() -> None:
    assert ActionParser.validate_target("P1", ("P1", "P2"))
    assert not ActionParser.validate_target("P3", ("P1", "P2"))
    assert not ActionParser.validate_target("P1", ("P1", "P2"), actor="P1")


def test_client_sends_non_streaming_chat_request_and_returns_content() -> None:
    received: dict[str, object] = {}

    def transport(request, timeout):
        received["url"] = request.full_url
        received["timeout"] = timeout
        received["body"] = json.loads(request.data)
        return b'{"message": {"content": "ok"}}'

    client = OllamaClient(base_url="http://ollama:11434/", timeout=17, transport=transport)

    assert client.chat("model-x", [{"role": "user", "content": "hello"}]) == "ok"
    assert received["url"] == "http://ollama:11434/api/chat"
    assert received["timeout"] == 17
    assert received["body"] == {
        "model": "model-x",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }


def test_client_retries_http_errors_twice_and_returns_structured_error() -> None:
    calls = 0

    def transport(request, timeout):
        nonlocal calls
        calls += 1
        raise HTTPError(
            request.full_url,
            503,
            "unavailable",
            {},
            BytesIO(b"model is loading"),
        )

    client = OllamaClient(transport=transport)

    with pytest.raises(OllamaError) as raised:
        client.chat("model-x", [{"role": "user", "content": "hello"}])

    assert calls == 3
    assert raised.value.kind == "http"
    assert raised.value.status_code == 503
    assert raised.value.attempts == 3
    assert raised.value.to_dict()["message"] == "model is loading"


def test_client_retries_timeout_and_accepts_empty_model_content() -> None:
    calls = 0

    def transport(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("slow")
        return b'{"message": {"content": ""}}'

    client = OllamaClient(max_retries=1, transport=transport)

    assert client.chat("model-x", []) == ""
    assert calls == 2


def test_client_chat_with_metadata_tracks_attempts() -> None:
    calls = 0

    def transport(request, timeout):
        nonlocal calls
        calls += 1
        if calls < 2:
            raise TimeoutError("slow")
        return b'{"message": {"content": "success after retry"}}'

    client = OllamaClient(max_retries=2, transport=transport)
    resp = client.chat_with_metadata("model-x", [])

    assert resp.content == "success after retry"
    assert resp.attempts == 2
    assert calls == 2



def make_state(seed: int = 23) -> GameState:
    return GameState(
        players=[
            Player("P0", "model-0", Role.VILLAGER),
            Player("P1", "model-1", Role.WOLF),
            Player("P2", "model-2", Role.SEER),
        ],
        seed=seed,
    )


class FakeClient:
    def __init__(self, *responses: str | OllamaError) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        self.prompts.append(messages[-1]["content"])
        response = self.responses.pop(0)
        if isinstance(response, OllamaError):
            raise response
        return response


def test_invalid_response_is_reprompted_then_validated_before_return() -> None:
    client = FakeClient("投票：Unknown", "投票：P1")
    actions = OllamaActions(make_state(), client)

    target = actions.vote("P0", ("P0", "P1", "P2"), day=1, runoff=False)

    assert target == "P1"
    assert len(client.prompts) == 2
    assert "P0" not in client.prompts[0].split("選択可能な候補: ", 1)[1].splitlines()[0]
    assert "不正" in client.prompts[1]
    assert actions.traces[-1].fallback_used is False
    assert actions.traces[-1].raw_responses == ("投票：Unknown", "投票：P1")


def test_invalid_response_after_reprompt_uses_seeded_valid_fallback() -> None:
    first_client = FakeClient("投票：Unknown", "")
    second_client = FakeClient("投票：Unknown", "")
    first = OllamaActions(make_state(seed=71), first_client)
    second = OllamaActions(make_state(seed=71), second_client)

    first_target = first.vote("P0", ("P0", "P1", "P2"), day=1, runoff=False)
    second_target = second.vote("P0", ("P0", "P1", "P2"), day=1, runoff=False)

    assert first_target == second_target
    assert first_target in {"P1", "P2"}
    assert first.traces[-1].fallback_used
    assert len(first.traces[-1].raw_responses) == 2


def test_ollama_failure_falls_back_and_preserves_structured_error() -> None:
    client = FakeClient(
        OllamaError("timeout", "request timed out", attempts=3)
    )
    actions = OllamaActions(make_state(seed=5), client)

    target = actions.choose_attack(("P1",), ("P0", "P2"))

    assert target in {"P0", "P2"}
    assert len(client.prompts) == 1
    assert actions.traces[-1].fallback_used
    assert actions.traces[-1].error == {
        "kind": "timeout",
        "message": "request timed out",
        "attempts": 3,
        "status_code": None,
    }


def test_action_validation_excludes_dead_players_and_wolf_attackers() -> None:
    state = make_state()
    state.mark_dead("P2")
    client = FakeClient("襲撃対象：P1", "襲撃対象：P2")
    actions = OllamaActions(state, client)

    target = actions.choose_attack(("P1",), ("P0", "P1", "P2"))

    assert target == "P0"
    assert len(client.prompts) == 2
    assert "P1, P0" not in client.prompts[0]
    assert actions.traces[-1].fallback_used


def test_discussion_failure_uses_explicit_fallback_and_trace() -> None:
    client = FakeClient(OllamaError("connection", "offline", attempts=3))
    actions = OllamaActions(make_state(), client)

    text = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert text == "P0 は発言できませんでした。"
    assert actions.traces[-1].action == "discussion"
    assert actions.traces[-1].fallback_used


def test_empty_discussion_is_reprompted_once_then_accepted_if_nonempty() -> None:
    client = FakeClient(" \n\t ", "話し合いを続けましょう。")
    actions = OllamaActions(make_state(), client)

    text = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert text == "話し合いを続けましょう。"
    assert len(client.prompts) == 2
    assert "空白のみ" in client.prompts[1]
    assert actions.traces[-1].raw_responses == (" \n\t ", "話し合いを続けましょう。")
    assert actions.traces[-1].fallback_used is False


def test_empty_discussion_after_reprompt_uses_explicit_fallback() -> None:
    client = FakeClient("", " \t\n ")
    actions = OllamaActions(make_state(), client)

    text = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert text == "P0 は発言できませんでした。"
    assert len(client.prompts) == 2
    assert actions.traces[-1].raw_responses == ("", " \t\n ")
    assert actions.traces[-1].fallback_used is True


def test_action_lifecycle_events_include_reprompt_fallback_and_timing() -> None:
    client = FakeClient("投票：unknown", "投票：unknown")
    events: list[dict[str, object]] = []
    actions = OllamaActions(make_state(), client, event_callback=events.append)

    actions.vote("P0", ("P0", "P1", "P2"), day=1, runoff=False)

    assert [event["type"] for event in events] == [
        "llm_action_started",
        "llm_action_parse_failed",
        "llm_action_parse_failed",
        "llm_action_completed",
    ]
    assert events[0]["player"] == "P0"
    assert events[0]["action"] == "vote"
    assert events[0]["private"] is False
    assert events[1]["will_retry"] is True
    assert events[2]["will_retry"] is False
    assert events[3]["semantic_retries"] == 1
    assert events[3]["fallback_used"] is True
    assert isinstance(events[3]["elapsed_seconds"], float)
