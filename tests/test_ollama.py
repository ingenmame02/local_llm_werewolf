import json
from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from werewolf import (
    ActionKind,
    ActionParser,
    ChatResponse,
    GameEngine,
    GameState,
    Message,
    OllamaActions,
    OllamaClient,
    OllamaError,
    Player,
    PublicSeerClaim,
    Role,
    ToolCall,
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

    assert client.chat(
        "model-x",
        [{"role": "user", "content": "hello"}],
        options={"temperature": 0.4, "num_ctx": 8192},
    ) == "ok"
    assert received["url"] == "http://ollama:11434/api/chat"
    assert received["timeout"] == 17
    assert received["body"] == {
        "model": "model-x",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "options": {"temperature": 0.4, "num_ctx": 8192},
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


def test_client_chat_with_metadata_tracks_completion_and_thinking_fields() -> None:
    def transport(request, timeout):
        return json.dumps(
            {
                "message": {"content": "", "thinking": "internal chain"},
                "done_reason": "length",
            }
        ).encode("utf-8")

    response = OllamaClient(transport=transport).chat_with_metadata("model-x", [])

    assert response.content == ""
    assert response.done_reason == "length"
    assert response.content_length == 0
    assert response.thinking_present is True
    assert response.to_metadata() == {
        "done_reason": "length",
        "content_length": 0,
        "thinking_present": True,
    }


def test_client_sends_claim_tool_and_parses_tool_calls() -> None:
    received: dict[str, object] = {}

    def transport(request, timeout):
        received["body"] = json.loads(request.data)
        return json.dumps(
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "claim_role",
                                "arguments": {
                                    "role": "seer",
                                    "target": "P1",
                                    "result": "wolf",
                                },
                            },
                        }
                    ],
                },
                "done_reason": "stop",
            }
        ).encode("utf-8")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "claim_role",
                "parameters": {"type": "object"},
            },
        }
    ]
    response = OllamaClient(transport=transport).chat_with_metadata(
        "model-x",
        [{"role": "user", "content": "公開CO"}],
        tools=tools,
    )

    assert received["body"]["tools"] == tools
    assert response.tool_calls == (
        ToolCall(
            name="claim_role",
            arguments={"role": "seer", "target": "P1", "result": "wolf"},
            call_id="call-1",
        ),
    )


def test_context_diagnostics_parse_ollama_ps_and_report_process_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "12288")
    def transport(request, timeout):
        assert request.full_url.endswith("/api/ps")
        return json.dumps(
            {
                "models": [
                    {
                        "name": "model-x:latest",
                        "context_length": 8192,
                        "processor": "100% GPU",
                    }
                ]
            }
        ).encode("utf-8")

    monkeypatch.setattr(
        "werewolf.ollama.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "NAME              ID              SIZE      PROCESSOR    CONTEXT    UNTIL\n"
                "model-x:latest    abc123          4.7 GB    100% GPU     8192       4 minutes from now\n"
            ),
            stderr="",
        ),
    )

    diagnostics = OllamaClient(transport=transport).context_diagnostics("model-x")

    assert diagnostics["process_ollama_context_length_env"] == "12288"
    assert diagnostics["ollama_ps_context"] == 8192
    assert diagnostics["ollama_ps_processor"] == "100% GPU"
    assert diagnostics["ollama_ps_source"] == "api"
    assert diagnostics["ollama_ps_error"] is None


def test_api_ps_parser_returns_none_for_model_not_loaded() -> None:
    response = {"models": [{"name": "other:latest", "context_length": 4096}]}

    assert OllamaClient.parse_api_ps_model(response, "model-x") is None


def test_ollama_context_parsers_read_modelfile_default_and_loaded_processor() -> None:
    assert OllamaClient.parse_modelfile_num_ctx(
        "FROM model-x\nPARAMETER temperature 0.7\nPARAMETER num_ctx 4096\n"
    ) == 4096
    ps_output = (
        "NAME              ID              SIZE      PROCESSOR    CONTEXT    UNTIL\n"
        "model-x:latest    abc123          4.7 GB    70%/30% CPU/GPU    8192    4 minutes from now\n"
    )
    assert (
        OllamaClient.parse_ollama_ps_column(ps_output, "model-x", "PROCESSOR")
        == "70%/30% CPU/GPU"
    )




def make_state(seed: int = 23) -> GameState:
    return GameState(
        players=[
            Player("P0", "model-0", Role.VILLAGER),
            Player("P1", "model-1", Role.WOLF),
            Player("P2", "model-2", Role.SEER),
        ],
        seed=seed,
    )


def test_text_claims_have_identical_public_state_with_tool_and_non_tool_clients() -> None:
    seer_report = (
        "占い師CO：対象=P2、判定=人狼、理由=夜の占い結果、"
        "投票提案=P2へ投票します。"
    )
    counterclaim = (
        "対抗占い師CO：対象=P2、判定=人間、理由=独自の占い結果、"
        "投票提案=P2への投票は保留します。"
    )

    class ScriptedClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.prompt_by_actor: dict[str, str] = {}
            self.counts: dict[str, int] = {}

        def response_for(self, messages):
            prompt = messages[-1]["content"]
            actor = prompt.split("- あなたのプレイヤー名：", 1)[1].splitlines()[0]
            self.prompt_by_actor.setdefault(actor, prompt)
            count = self.counts.get(actor, 0)
            self.counts[actor] = count + 1
            if actor == "P0" and count == 0:
                return seer_report
            if actor == "P1" and count == 0:
                return counterclaim
            return "公開情報を踏まえて投票先を考えます。"

        def chat(self, model, messages, options=None):
            return self.response_for(messages)

    # The metadata client returns the same ordinary text as the non-tool client.
    class ToolCapableClient(ScriptedClient):
        def __init__(self) -> None:
            super().__init__()
            self.tool_arguments: list[object] = []

        def chat_with_metadata(self, model, messages, options=None, tools=None):
            self.tool_arguments.append(tools)
            content = self.response_for(messages)
            return ChatResponse(
                content=content,
                attempts=1,
                content_length=len(content),
            )

    def run_game(client):
        state = make_state()
        state.players[0].role = Role.SEER
        state.players[2].role = Role.VILLAGER
        state.seer_results["P2"] = "wolf"
        state.seer_result_history.append((1, "P2", "wolf"))
        GameEngine(state, OllamaActions(state, client)).run_discussion()
        return state, client

    no_tool_state, no_tool_client = run_game(ScriptedClient())
    tool_state, tool_client = run_game(ToolCapableClient())

    assert no_tool_state.public_role_claims == tool_state.public_role_claims
    assert no_tool_state.seer_claims == tool_state.seer_claims
    assert [
        (claim.claimant, claim.claimed_role, claim.target, claim.result)
        for claim in no_tool_state.public_role_claims
    ] == [
        ("P0", "seer", None, None),
        ("P0", "seer", "P2", "wolf"),
        ("P1", "seer", None, None),
        ("P1", "seer", "P2", "human"),
    ]
    assert no_tool_state.public_role_claims[1].reason == "夜の占い結果"
    assert no_tool_state.public_role_claims[1].vote_proposal == "P2へ投票します"
    assert no_tool_state.public_role_claims[3].reason == "独自の占い結果"
    assert no_tool_client.prompt_by_actor["P1"] == tool_client.prompt_by_actor["P1"]
    assert no_tool_client.prompt_by_actor["P2"] == tool_client.prompt_by_actor["P2"]
    assert "P0は占い師を主張（P2：人狼）" in no_tool_client.prompt_by_actor["P1"]
    assert "P0は占い師を主張（P2：人狼）" in no_tool_client.prompt_by_actor["P2"]
    assert "P1は占い師を主張（P2：人間）" in no_tool_client.prompt_by_actor["P2"]
    assert not callable(getattr(no_tool_client, "chat_with_metadata", None))
    assert any(tools is not None for tools in tool_client.tool_arguments)


def test_model_rejecting_tool_calls_retries_the_same_request_as_plain_text() -> None:
    class ToolUnsupportedClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.plain_requests = 0

        def chat_with_metadata(self, model, messages, options=None, tools=None):
            assert tools is not None
            raise OllamaError(
                "http",
                "model does not support tools",
                attempts=1,
                status_code=400,
            )

        def chat(self, model, messages, options=None):
            self.plain_requests += 1
            prompt = messages[-1]["content"]
            actor = prompt.split("- あなたのプレイヤー名：", 1)[1].splitlines()[0]
            if actor == "P0" and self.plain_requests == 1:
                return (
                    "【CO】占い師CO：対象=P1、判定=人狼、理由=占い結果、"
                    "投票提案=P1へ投票します。\n【本文】P1への投票を提案します。"
                )
            return "【CO】なし\n【本文】公開情報を確認して投票先を考えます。"

    state = make_state()
    state.players[0].role = Role.SEER
    events: list[dict[str, object]] = []
    client = ToolUnsupportedClient()
    actions = OllamaActions(state, client, event_callback=events.append)

    GameEngine(state, actions, event_callback=events.append).run_discussion()

    assert any(
        claim.claimant == "P0"
        and claim.claimed_role == "seer"
        and claim.target == "P1"
        and claim.result == "wolf"
        for claim in state.public_role_claims
    )
    assert client.plain_requests > 0
    assert any(event["type"] == "claim_tool_unavailable" for event in events)


class FakeClient:
    def __init__(self, *responses: str | OllamaError) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.options: list[dict[str, object] | None] = []
        self.timeouts: list[float | None] = []

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, object] | None = None,
    ) -> str:
        self.prompts.append(messages[-1]["content"])
        self.options.append(options)
        self.timeouts.append(getattr(self, "timeout", None))
        response = self.responses.pop(0)
        if isinstance(response, OllamaError):
            raise response
        return response


@pytest.mark.parametrize("role", list(Role))
def test_all_roles_can_publish_claims_without_revealing_true_role(role) -> None:
    class ToolClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.responses = [
                ChatResponse(
                    content="",
                    attempts=1,
                    tool_calls=(
                        ToolCall(
                            "claim_role",
                            {"role": "villager", "reason": "公開情報を整理"},
                        ),
                    ),
                ),
                ChatResponse(
                    content="今日は発言を確認して投票先を考えます。",
                    attempts=1,
                ),
            ]
            self.requests: list[tuple[list[dict[str, object]], object]] = []

        def chat_with_metadata(self, model, messages, options=None, tools=None):
            self.requests.append((list(messages), tools))
            self.prompts.append(messages[-1]["content"])
            return self.responses.pop(0)

    state = make_state()
    state.players[0].role = role
    client = ToolClient()
    events: list[dict[str, object]] = []
    actions = OllamaActions(state, client, event_callback=events.append)

    assert actions.discuss("P0", 1, tuple(state.alive_players), ()) == (
        "今日は発言を確認して投票先を考えます。"
    )
    assert state.public_role_claims[0].claimant == "P0"
    assert state.public_role_claims[0].claimed_role == "villager"
    assert client.requests[0][1] is not None
    assert client.requests[1][1] is not None
    tool_result = json.loads(client.requests[1][0][-1]["content"])
    assert tool_result == {
        "status": "published",
        "message": "公開主張を記録しました。",
    }
    assert "role" not in tool_result
    assert [event["type"] for event in events if "claim_role" in event["type"]] == [
        "claim_role_published"
    ]
    assert all(
        "role" not in event
        for event in events
        if event["type"] == "claim_role_published"
    )


def test_claim_tool_rejects_conflicting_and_duplicate_public_claims() -> None:
    class ToolClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.responses = [
                ChatResponse(
                    content="",
                    attempts=1,
                    tool_calls=(
                        ToolCall(
                            "claim_role",
                            {
                                "role": "seer",
                                "target": "P1",
                                "result": "wolf",
                                "reason": "占い結果",
                                "vote_proposal": "P1に投票",
                            },
                        ),
                    ),
                ),
                ChatResponse(content="考えを整理します。", attempts=1),
                ChatResponse(
                    content="",
                    attempts=1,
                    tool_calls=(
                        ToolCall(
                            "claim_role",
                            {
                                "role": "seer",
                                "target": "P1",
                                "result": "human",
                            },
                        ),
                    ),
                ),
                ChatResponse(content="今日は投票先を考えます。", attempts=1),
                ChatResponse(
                    content="",
                    attempts=1,
                    tool_calls=(
                        ToolCall(
                            "claim_role",
                            {
                                "role": "seer",
                                "target": "P1",
                                "result": "wolf",
                            },
                        ),
                    ),
                ),
                ChatResponse(content="もう一度状況を確認します。", attempts=1),
            ]

        def chat_with_metadata(self, model, messages, options=None, tools=None):
            self.prompts.append(messages[-1]["content"])
            return self.responses.pop(0)

    state = make_state()
    events: list[dict[str, object]] = []
    actions = OllamaActions(state, ToolClient(), event_callback=events.append)

    actions.discuss("P0", 1, tuple(state.alive_players), ())
    assert len(state.public_role_claims) == 1
    assert state.public_role_claims[0].reason == "占い結果"
    assert state.public_role_claims[0].vote_proposal == "P1に投票"
    actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert len(state.public_role_claims) == 1
    assert any(
        event["type"] == "claim_role_rejected"
        and event["rejection_reason"] == "inconsistent_seer_result"
        for event in events
    )
    actions.discuss("P0", 1, tuple(state.alive_players), ())
    assert sum(
        event.get("rejection_reason") == "duplicate_claim" for event in events
    ) == 1


def test_dead_player_cannot_publish_a_role_claim() -> None:
    state = make_state()
    state.mark_dead("P0")
    events: list[dict[str, object]] = []
    actions = OllamaActions(state, FakeClient(), event_callback=events.append)

    result = actions._handle_public_tool_call(
        state.players[0],
        ToolCall("claim_role", {"role": "seer"}),
        "discussion",
        0.0,
    )

    assert result["status"] == "rejected"
    assert state.public_role_claims == []
    assert events[0]["rejection_reason"] == "dead_player"


def test_invalid_response_is_reprompted_then_validated_before_return() -> None:
    client = FakeClient("投票：Unknown", "投票：P1")
    actions = OllamaActions(make_state(), client)

    target = actions.vote("P0", ("P0", "P1", "P2"), day=1, runoff=False)

    assert target == "P1"
    assert len(client.prompts) == 2
    assert "P0" not in client.prompts[0].split("選択可能な候補: ", 1)[1].splitlines()[0]
    assert "候補外" in client.prompts[1]
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


@pytest.mark.parametrize(
    ("action", "first_response", "second_response"),
    [
        ("vote", "投票：P1", "投票：P2"),
        ("attack", "襲撃対象：P0", "襲撃対象：P0"),
        ("seer", "占い対象：P0", "占い対象：P0"),
    ],
)
def test_truncated_target_response_retries_with_increased_generation_limit(
    action, first_response, second_response
) -> None:
    from werewolf import ChatResponse

    class MetadataClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.chat_responses = [
                ChatResponse(
                    content=first_response,
                    attempts=1,
                    done_reason="length",
                    content_length=len(first_response),
                ),
                ChatResponse(
                    content=second_response,
                    attempts=1,
                    done_reason="stop",
                    content_length=len(second_response),
                ),
            ]

        def chat(self, model, messages, options=None):
            self.prompts.append(messages[-1]["content"])
            self.options.append(options)
            response = self.chat_responses.pop(0)
            self._last_chat_response = response
            return response.content

    state = make_state()
    actor_name = {"vote": "P0", "attack": "P1", "seer": "P2"}[action]
    next(player for player in state.players if player.name == actor_name).num_predict = 256
    client = MetadataClient()
    actions = OllamaActions(state, client)

    if action == "vote":
        target = actions.vote("P0", ("P0", "P1", "P2"), day=1, runoff=False)
    elif action == "attack":
        target = actions.choose_attack(("P1",), ("P0", "P2"))
    else:
        target = actions.choose_seer_target("P2", ("P0", "P1"), ())

    expected_target = {"vote": "P2", "attack": "P0", "seer": "P0"}[action]
    assert target == expected_target
    assert client.options == [
        {"num_predict": 256},
        {"num_predict": 512},
    ]
    assert len(client.prompts) == 2
    assert actions.traces[-1].failure_reasons == ("generation_truncated",)
    assert actions.traces[-1].fallback_used is False


def test_thinking_only_target_responses_retry_then_use_legal_fallback() -> None:
    from werewolf import ChatResponse

    class MetadataClient(FakeClient):
        def chat(self, model, messages, options=None):
            self.prompts.append(messages[-1]["content"])
            self.options.append(options)
            response = ChatResponse(
                content="<think>選択を検討中</think>",
                attempts=1,
                done_reason="stop",
                content_length=len("<think>選択を検討中</think>"),
                thinking_present=True,
            )
            self._last_chat_response = response
            return response.content

    state = make_state(seed=71)
    state.players[0].num_predict = 256
    client = MetadataClient()
    actions = OllamaActions(state, client)
    events: list[dict[str, object]] = []
    actions.event_callback = events.append

    target = actions.vote("P0", ("P0", "P1", "P2"), day=1, runoff=False)

    assert target in {"P1", "P2"}
    assert client.options == [
        {"num_predict": 256},
        {"num_predict": 512},
    ]
    assert len(client.prompts) == 2
    assert actions.traces[-1].failure_reasons == (
        "thinking_only_response",
        "thinking_only_response",
    )
    assert actions.traces[-1].fallback_used is True
    parse_failures = [
        event for event in events if event["type"] == "llm_action_parse_failed"
    ]
    assert [event["reason"] for event in parse_failures] == [
        "thinking_only_response",
        "thinking_only_response",
    ]
    assert [event["will_retry"] for event in parse_failures] == [True, False]


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

    assert text == "パス"
    assert actions.traces[-1].action == "discussion"
    assert actions.traces[-1].fallback_used
    assert actions.traces[-1].failure_reasons == ("communication_failure",)


def test_empty_discussion_is_reprompted_once_then_accepted_if_nonempty() -> None:
    client = FakeClient(" \n\t ", "話し合いを続けましょう。")
    actions = OllamaActions(make_state(), client)

    text = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert text == "話し合いを続けましょう。"
    assert len(client.prompts) == 2
    assert "前回は本文が空でした" in client.prompts[1]
    assert actions.traces[-1].raw_responses == (" \n\t ", "話し合いを続けましょう。")
    assert actions.traces[-1].fallback_used is False


def test_empty_discussion_after_reprompt_uses_explicit_fallback() -> None:
    client = FakeClient("", " \t\n ")
    actions = OllamaActions(make_state(), client)

    text = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert text == "パス"
    assert len(client.prompts) == 2
    assert actions.traces[-1].raw_responses == ("", " \t\n ")
    assert actions.traces[-1].fallback_used is True


def test_discussion_pass_is_explicit_valid_response_without_retry() -> None:
    client = FakeClient("  パス。  ")
    actions = OllamaActions(make_state(), client)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert response == "パス"
    assert len(client.prompts) == 1
    assert actions.traces[-1].fallback_used is False
    assert actions.traces[-1].raw_responses == ("  パス。  ",)


def test_thinking_only_response_is_retried_and_thought_text_is_not_returned() -> None:
    client = FakeClient("<think>推理中</think>", "<think>内部発言</think>結論です。")
    actions = OllamaActions(make_state(), client)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert response == "結論です。"
    assert len(client.prompts) == 2
    assert actions.traces[-1].fallback_used is False


def test_vote_target_inside_think_tag_is_ignored() -> None:
    client = FakeClient("<think>投票：P1</think>\n投票：P2")
    actions = OllamaActions(make_state(), client)

    target = actions.vote("P0", ("P0", "P1", "P2"), day=1, runoff=False)

    assert target == "P2"
    assert len(client.prompts) == 1


def test_pass_after_think_tag_is_valid_and_only_final_text_is_recorded() -> None:
    client = FakeClient("<think>考えています</think> パス")
    actions = OllamaActions(make_state(), client)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert response == "パス"
    assert actions.traces[-1].fallback_used is False


def test_two_block_discussion_accepts_co_none_and_hold_without_retry() -> None:
    for co_value in ("なし", "保留"):
        expected = f"【CO】{co_value}\n【本文】意見を整理して投票先を考えます。"
        client = FakeClient(expected)
        actions = OllamaActions(make_state(), client)

        response = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

        assert response == expected
        assert len(client.prompts) == 1
        assert actions.traces[-1].fallback_used is False


def test_seer_can_hold_only_human_results_without_publishing_them() -> None:
    state = make_state()
    state.players[0].role = Role.SEER
    state.seer_results["P1"] = "human"
    response = "【CO】保留\n【本文】まだ情報を整理してから判断します。"
    client = FakeClient(response)
    actions = OllamaActions(state, client)

    assert actions.discuss("P0", 1, ("P0", "P1", "P2"), ()) == response
    assert len(client.prompts) == 1
    assert actions.traces[-1].fallback_used is False
    assert state.public_role_claims == []


def test_discussion_retry_uses_short_context_and_seer_only_private_results() -> None:
    state = make_state()
    state.players[0].role = Role.SEER
    state.seer_results["P2"] = "human"
    state.conversation_log.extend(
        Message("P1", f"古い公開発言 {index}") for index in range(6)
    )
    state.conversation_log.append(Message("P2", "直近の公開発言です。"))
    client = FakeClient(
        "【CO】壊れた形式\n本文マーカーなし",
        "【CO】保留\n【本文】結果は伏せ、発言をもう少し確認します。",
    )
    actions = OllamaActions(state, client)

    response = actions.discuss(
        "P0", 1, ("P0", "P1", "P2"), tuple(state.conversation_log)
    )

    assert response.startswith("【CO】保留")
    assert len(client.prompts) == 2
    assert "古い公開発言 0" not in client.prompts[1]
    assert "直近の公開発言です。" in client.prompts[1]
    assert "P2: 人間" in client.prompts[1]
    assert "discussion_output_format" not in client.prompts[1]
    assert actions.traces[-1].fallback_used is False


def test_broken_two_block_discussion_is_retried_once() -> None:
    client = FakeClient(
        "【CO】なし\n本文にマーカーがありません。",
        "【CO】なし\n【本文】意見を整理して投票先を考えます。",
    )
    actions = OllamaActions(make_state(), client)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert response.startswith("【CO】なし\n【本文】")
    assert len(client.prompts) == 2
    assert "前回は二層形式が不正でした" in client.prompts[1]
    assert actions.traces[-1].fallback_used is False


def test_repeated_two_block_markers_fall_back_to_safe_pass() -> None:
    malformed = (
        "【CO】なし\n【本文】発言します。\n【本文】もう一度発言します。"
    )
    client = FakeClient(malformed, malformed)
    actions = OllamaActions(make_state(), client)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert response == "パス"
    assert len(client.prompts) == 2
    assert actions.traces[-1].fallback_used is True


def test_two_block_seer_result_without_co_uses_task19_retry() -> None:
    state = make_state()
    state.players[0].role = Role.SEER
    state.seer_results["P2"] = "human"
    client = FakeClient(
        "【CO】なし\n【本文】P2は人間と判定されました。",
        "【CO】占い師CO：対象=P2、判定=人間、理由=夜の占い、"
        "投票提案=別の発言を確認\n【本文】P2への疑いを弱めます。",
    )
    actions = OllamaActions(state, client)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert response.startswith("【CO】占い師CO")
    assert len(client.prompts) == 2
    assert "同じ発言内で明示的に占い師COしてください" in client.prompts[1]


def test_wolf_can_pass_consultation_but_attack_selection_remains_required() -> None:
    client = FakeClient("パス", "襲撃対象：P0")
    actions = OllamaActions(make_state(), client)

    consultation = actions.wolf_discuss("P1", "P1", ())
    target = actions.choose_attack(("P1",), ("P0", "P2"))

    assert consultation == "パス"
    assert target == "P0"
    assert len(client.prompts) == 2


def test_seer_must_use_result_before_pass_and_can_pass_after_contribution() -> None:
    state = make_state()
    state.seer_results["P1"] = "wolf"
    client = FakeClient(
        "パス",
        "占い師CO。対象：P1／判定：人狼／根拠：占い結果です。"
        "次の提案：P1に投票したいです。",
        "パス",
    )
    actions = OllamaActions(state, client)

    first_response = actions.discuss("P2", 1, ("P0", "P1", "P2"), ())
    state.seer_claims.extend(
        [
            PublicSeerClaim(day=1, claimant="P2"),
            PublicSeerClaim(day=1, claimant="P2", target="P1", result="wolf"),
        ]
    )
    second_response = actions.discuss(
        "P2",
        1,
        ("P0", "P1", "P2"),
        (Message("P2", first_response),),
    )

    assert first_response.startswith("占い師CO。対象：P1／判定：人狼")
    assert second_response == "パス"
    assert len(client.prompts) == 3
    assert "前回の応答は空、パス" in client.prompts[1]
    assert "P1: 人狼" in client.prompts[0]
    assert "今回（当日）：\nP1: 人狼" in client.prompts[0]
    assert actions.traces[0].fallback_used is False
    assert actions.traces[1].fallback_used is False


def test_seer_forced_result_fallback_when_retry_does_not_use_result() -> None:
    state = make_state()
    state.seer_results["P1"] = "human"
    client = FakeClient("今日はパス", "まだ意見はありません")
    actions = OllamaActions(state, client)

    response = actions.discuss("P2", 1, ("P0", "P1", "P2"), ())

    assert response.startswith("【CO】占い師CO：対象=P1、判定=人間")
    assert "理由=実際の占い結果" in response
    assert "投票提案=" in response
    assert "新情報か具体的な投票方針を必ず加えてください" in client.prompts[0]
    assert len(client.prompts) == 2
    assert actions.traces[-1].fallback_used is True


def test_seer_wolf_result_fallback_is_a_structured_public_report() -> None:
    state = make_state()
    state.seer_results["P1"] = "wolf"
    client = FakeClient("パス", "まだ意見はありません")
    actions = OllamaActions(state, client)

    response = actions.discuss("P2", 1, ("P0", "P1", "P2"), ())

    assert response.startswith("【CO】占い師CO：対象=P1、判定=人狼")
    assert "理由=" in response
    assert "投票提案=" in response
    assert actions.traces[-1].fallback_used is True


@pytest.mark.parametrize(
    ("result", "spoken_result"),
    [("human", "人間"), ("wolf", "人狼")],
)
def test_seer_result_without_claim_is_reprompted_and_explicit_claim_is_accepted(
    result: str, spoken_result: str
) -> None:
    state = make_state()
    state.players[0].role = Role.SEER
    state.seer_results["P2"] = result
    state.seer_result_history.append((1, "P2", result))
    client = FakeClient(
        f"P2は{spoken_result}と判定された。",
        f"占い師CO：対象=P2、判定={spoken_result}、"
        "理由=夜の占い結果、投票提案=P2への投票を提案します。",
    )
    events: list[dict[str, object]] = []
    actions = OllamaActions(state, client, event_callback=events.append)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert response.startswith("占い師CO：対象=P2")
    assert len(client.prompts) == 2
    assert "同じ発言内で明示的に占い師COしてください" in client.prompts[1]
    assert actions.traces[-1].fallback_used is False
    assert [
        event["will_retry"]
        for event in events
        if event["type"] == "seer_result_claim_missing"
    ] == [True]


def test_missing_seer_claim_falls_back_and_is_published_to_next_player() -> None:
    state = make_state()
    state.players[0].role = Role.SEER
    state.seer_results["P2"] = "human"
    state.seer_result_history.append((1, "P2", "human"))
    client = FakeClient(
        "P2は人間と判定された。",
        "P2は人間と判定された。",
        "状況を確認します。",
        "発言を見て投票先を考えます。",
        "P2への占い結果を踏まえ、別の発言も確認したいです。",
        "引き続き状況を確認します。",
        "投票先を考えます。",
        "ほかの発言も見たいです。",
        "今日は全体を見て判断します。",
        "ほかの意見も確認します。",
        "投票先を改めて考えます。",
        "議論を続けましょう。",
    )
    events: list[dict[str, object]] = []
    actions = OllamaActions(state, client, event_callback=events.append)
    GameEngine(state, actions, event_callback=events.append).run_discussion()

    seer_claim = next(
        claim
        for claim in state.public_role_claims
        if claim.claimant == "P0" and claim.target == "P2"
    )
    assert (seer_claim.claimed_role, seer_claim.result) == ("seer", "human")
    assert "P0は占い師を主張（P2：人間）" in client.prompts[2]
    assert [
        event["will_retry"]
        for event in events
        if event["type"] == "seer_result_claim_missing"
    ] == [True, False]
    assert any(event["type"] == "seer_result_claim_fallback" for event in events)
    assert actions.traces[0].fallback_used is True


def test_seer_can_use_result_without_publicly_disclosing_it() -> None:
    state = make_state()
    state.players[0].role = Role.SEER
    state.seer_results["P2"] = "human"
    client = FakeClient("P2への疑いは弱め、別の発言を確認したいです。")
    actions = OllamaActions(state, client)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert response == "P2への疑いは弱め、別の発言を確認したいです。"
    assert len(client.prompts) == 1
    assert actions.traces[-1].fallback_used is False


def test_seer_retries_until_unpublished_wolf_result_is_publicly_reported() -> None:
    state = make_state()
    state.seer_results["P1"] = "wolf"
    state.seer_result_history.append((1, "P1", "wolf"))
    client = FakeClient(
        "P1は怪しいので投票したいです。",
        "占い師CO。対象：P1／判定：人狼／根拠：占い結果です。"
        "次の提案：P1に投票したいです。",
    )
    actions = OllamaActions(state, client)

    response = actions.discuss("P2", 1, ("P0", "P1", "P2"), ())

    assert response.startswith("占い師CO。対象：P1／判定：人狼")
    assert len(client.prompts) == 2
    assert "未公開の人狼判定" in client.prompts[1]
    assert "wolf_result_not_disclosed" in actions.traces[-1].failure_reasons


def test_claimant_cannot_change_its_published_seer_result() -> None:
    state = make_state()
    state.seer_claims.extend(
        [
            PublicSeerClaim(day=1, claimant="P1"),
            PublicSeerClaim(day=1, claimant="P1", target="P2", result="human"),
        ]
    )
    client = FakeClient(
        "占い師CO。対象：P2／判定：人狼／根拠：占い結果です。",
        "今日はP0の発言をもう少し見たいです。",
    )
    actions = OllamaActions(state, client)

    response = actions.discuss("P1", 1, ("P0", "P1", "P2"), ())

    assert response == "今日はP0の発言をもう少し見たいです。"
    assert len(client.prompts) == 2
    assert "過去の主張と一貫" in client.prompts[1]
    assert "inconsistent_seer_result" in actions.traces[-1].failure_reasons


def test_context_overflow_compacts_old_discussion_and_retries() -> None:
    state = make_state()
    state.discussion_turn = 2
    history = tuple(
        Message(f"P{index % 3}", f"old statement {index}")
        for index in range(12)
    ) + (Message("P1", "latest direct question"),)
    client = FakeClient(
        OllamaError("http", "prompt is too long for context window", attempts=1),
        "回答します。",
    )
    actions = OllamaActions(state, client, options={"temperature": 0.3, "num_ctx": 4096})

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), history)

    assert response == "回答します。"
    assert len(client.prompts) == 2
    assert "old statement 0" not in client.prompts[0]
    assert "old statement 10" in client.prompts[0]
    assert "old statement 0" not in client.prompts[1]
    assert "old statement 10" in client.prompts[1]
    assert "old statement 4" not in client.prompts[1]
    assert "古い会話" in client.prompts[1]
    assert "latest direct question" in client.prompts[1]
    assert "- 昼議論ターン：2/3" in client.prompts[1]
    assert actions.traces[-1].fallback_used is False


def test_model_specific_num_ctx_overrides_global_context_option() -> None:
    state = make_state()
    state.players[0].num_ctx = 16384
    state.players[0].num_predict = 192
    client = FakeClient("議論を続けましょう。")
    actions = OllamaActions(
        state,
        client,
        options={"temperature": 0.4, "num_ctx": 4096},
    )

    actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert client.options == [
        {"temperature": 0.4, "num_ctx": 16384, "num_predict": 192}
    ]


def test_request_configuration_records_effective_options_and_timeout() -> None:
    class DiagnosticClient(FakeClient):
        timeout = 30.0

        def context_diagnostics(self, model):
            return {
                "process_ollama_context_length_env": "12288",
                "ollama_ps_context": 8192,
                "ollama_ps_error": None,
            }

    state = make_state()
    state.players[0].num_ctx = 16384
    state.players[0].num_predict = 192
    state.players[0].timeout = 7.5
    client = DiagnosticClient("議論を続けましょう。")
    events: list[dict[str, object]] = []
    actions = OllamaActions(
        state,
        client,
        options={"temperature": 0.4, "num_ctx": 4096, "num_predict": 512},
        event_callback=events.append,
    )

    actions.discuss("P0", 1, tuple(state.alive_players), ())

    configuration = next(
        event for event in events if event["type"] == "llm_request_configuration"
    )
    assert configuration["request_options"] == {
        "temperature": 0.4,
        "num_ctx": 16384,
        "num_predict": 192,
    }
    assert configuration["request_timeout_seconds"] == 7.5
    assert configuration["num_ctx_source"] == "player_setting"
    assert configuration["context_diagnostics"]["ollama_ps_context"] == 8192
    assert configuration["context_diagnostics"][
        "process_ollama_context_length_env"
    ] == "12288"
    assert client.options == [configuration["request_options"]]
    assert client.timeouts == [7.5]


def test_unset_context_is_not_sent_even_when_process_environment_is_set() -> None:
    class DiagnosticClient(FakeClient):
        def context_diagnostics(self, model):
            return {
                "process_ollama_context_length_env": "12288",
                "ollama_ps_context": 8192,
                "ollama_ps_error": None,
            }

    events: list[dict[str, object]] = []
    client = DiagnosticClient("発言します。")
    actions = OllamaActions(
        make_state(),
        client,
        event_callback=events.append,
    )

    actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    configuration = next(
        event for event in events if event["type"] == "llm_request_configuration"
    )
    assert "num_ctx" not in configuration["request_options"]
    assert configuration["num_ctx_source"] == "ollama_service_default_or_environment"
    assert configuration["context_diagnostics"][
        "process_ollama_context_length_env"
    ] == "12288"


def test_global_context_option_is_reported_as_cli_setting() -> None:
    class DiagnosticClient(FakeClient):
        def context_diagnostics(self, model):
            return {
                "process_ollama_context_length_env": None,
                "ollama_ps_context": 4096,
                "ollama_ps_error": None,
            }

    events: list[dict[str, object]] = []
    actions = OllamaActions(
        make_state(),
        DiagnosticClient("議論を続けます。"),
        options={"num_ctx": 4096, "num_predict": 128},
        event_callback=events.append,
    )

    actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    configuration = next(
        event for event in events if event["type"] == "llm_request_configuration"
    )
    assert configuration["request_options"]["num_ctx"] == 4096
    assert configuration["num_ctx_source"] == "cli_setting"


def test_english_discussion_is_reprompted_and_japanese_answer_is_kept() -> None:
    client = FakeClient("I think P1 is suspicious.", "P1の投票先について確認したいです。")
    actions = OllamaActions(make_state(), client)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert response == "P1の投票先について確認したいです。"
    assert len(client.prompts) == 2
    assert "指定された応答言語" in client.prompts[1]
    assert actions.traces[-1].fallback_used is False


def test_self_reference_and_repeated_discussion_are_retried() -> None:
    state = make_state()
    state.conversation_log.append(
        Message("P0", "P0は村人で、今日はPlayer 1に投票するべきだと思います。")
    )
    client = FakeClient(
        "P0は村人なので、P1への投票を主張します。",
        "P1への投票は早計です。発言内容をもう一度確認したいです。",
    )
    actions = OllamaActions(state, client)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), tuple(state.conversation_log))

    assert response == "P1への投票は早計です。発言内容をもう一度確認したいです。"
    assert len(client.prompts) == 2
    assert "自分自身を第三者" in client.prompts[1]
    assert actions.traces[-1].fallback_used is False


def test_question_target_must_be_named_in_the_first_sentence() -> None:
    state = make_state()
    question = Message("P1", "P0、投票の根拠を説明してください。")
    state.conversation_log.append(question)
    client = FakeClient(
        "投票理由は発言が怪しかったからです。",
        "P1さん、発言の矛盾を根拠にしました。",
    )
    actions = OllamaActions(state, client)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), (question,))

    assert response.startswith("P1さん")
    assert len(client.prompts) == 2
    assert "回答対象：P1" in client.prompts[0]
    assert "最初の一文でP1への回答" in client.prompts[1]


def test_unrecorded_death_event_is_reprompted() -> None:
    client = FakeClient(
        "P1は昨夜襲撃されました。",
        "昨夜の死亡情報は台帳にありません。P1の投票理由を確認します。",
    )
    actions = OllamaActions(make_state(), client)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert response.startswith("昨夜の死亡情報")
    assert len(client.prompts) == 2
    assert "台帳にない襲撃・処刑・死亡" in client.prompts[1]


def test_villager_cannot_claim_to_know_the_attacker() -> None:
    client = FakeClient(
        "P1が昨夜P2を襲撃しました。",
        "P1を疑っていますが、襲撃者は未確認なので発言の根拠を確認します。",
    )
    actions = OllamaActions(make_state(), client)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert response.startswith("P1を疑っています")
    assert len(client.prompts) == 2
    assert "襲撃者は公開されていません" in client.prompts[1]


def test_fabricated_quote_and_repeated_speech_are_reprompted() -> None:
    state = make_state()
    state.conversation_log.extend(
        [
            Message("P1", "昨日は様子を見ます。"),
            Message("P0", "Player 1は「P2に投票すると言った」と発言しました。"),
        ]
    )
    client = FakeClient(
        "P0は「同じ主張をもう一度伝えます」と説明しました。",
        "P1の発言は投票先を明示していません。今日は発言の変化を確認します。",
    )
    actions = OllamaActions(state, client)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), tuple(state.conversation_log))

    assert response.startswith("P1の発言")
    assert len(client.prompts) == 2
    assert "履歴にない発言" in client.prompts[1]


def test_timeout_does_not_trigger_an_additional_semantic_retry() -> None:
    state = make_state()
    state.players[0].timeout = 7.0
    client = FakeClient(
        OllamaError("timeout", "prompt is too long for context window", attempts=3)
    )
    client.timeout = 30.0
    actions = OllamaActions(state, client)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert response == "パス"
    assert len(client.prompts) == 1
    assert client.timeouts == [7.0]
    assert client.timeout == 30.0
    assert actions.traces[-1].fallback_used is True


@pytest.mark.parametrize(
    ("content", "thinking", "done_reason", "reason"),
    [
        ("", False, "stop", "empty_content"),
        ("<think>分析中</think>", False, "stop", "thinking_only_response"),
        ("途中の回答", False, "length", "generation_truncated"),
    ],
)
def test_empty_generation_is_classified_and_retried_with_more_tokens(
    content, thinking, done_reason, reason
) -> None:
    from werewolf import ChatResponse

    class MetadataClient(FakeClient):
        def chat(self, model, messages, options=None):
            self.prompts.append(messages[-1]["content"])
            self.options.append(options)
            if len(self.prompts) == 1:
                self._last_chat_response = ChatResponse(
                    content=content,
                    attempts=1,
                    done_reason=done_reason,
                    content_length=len(content),
                    thinking_present=thinking,
                )
                return content
            return "議論ではP1の投票先を確認します。"

    state = make_state()
    state.players[0].num_predict = 256
    client = MetadataClient(content)
    actions = OllamaActions(state, client)

    response = actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    assert response == "議論ではP1の投票先を確認します。"
    assert len(client.prompts) == 2
    assert client.options[0] == {"num_predict": 256}
    assert client.options[1] == {"num_predict": 512}
    assert actions.traces[-1].failure_reasons == (reason,)
    assert actions.traces[-1].response_metadata[0] == {
        "done_reason": done_reason,
        "content_length": len(content),
        "thinking_present": thinking,
    }


def test_seer_skips_previously_scanned_living_candidates_and_dead_players() -> None:
    state = make_state()
    state.players[0].role = Role.SEER
    state.mark_dead("P1")
    state.seer_results["P0"] = "human"
    client = FakeClient("占い対象：P2")
    actions = OllamaActions(state, client)

    target = actions.choose_seer_target(
        "P0", ("P0", "P1", "P2"), (("P0", "human"),)
    )

    assert target == "P2"
    assert "選択可能な占い対象：P2" in client.prompts[0]


@pytest.mark.parametrize(
    ("response", "expected_reason"),
    [
        ("P1は人狼でした。", "unverified_dead_role"),
        ("P1に投票すべきです。", "dead_player_action_target"),
        ("今夜は占います。", "out_of_phase_action_claim"),
    ],
)
def test_game_rule_inconsistent_discussion_is_reprompted(
    response, expected_reason
) -> None:
    state = make_state()
    state.mark_dead("P1")
    client = FakeClient(response, "生存者のP2について投票根拠を確認します。")
    actions = OllamaActions(state, client)

    answer = actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert answer == "生存者のP2について投票根拠を確認します。"
    assert len(client.prompts) == 2
    assert expected_reason in actions.traces[-1].failure_reasons


def test_discussion_can_refer_to_a_dead_player_without_proposing_an_action() -> None:
    state = make_state()
    state.mark_dead("P1")
    response = "P1への投票は昨日のことで、今の対象は生存者から考えます。"
    client = FakeClient(response)
    actions = OllamaActions(state, client)

    assert actions.discuss("P0", 1, tuple(state.alive_players), ()) == response
    assert len(client.prompts) == 1


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
