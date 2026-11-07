import json
from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from werewolf.discussion_format import parse_discussion_claim
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
    THINKING_CAPABILITY_SOURCE,
    THINKING_NUM_PREDICT_SOURCE,
    THINKING_RESOLUTION_PRIORITY,
    apply_thinking_floor,
    normalize_action_budget_key,
    resolve_action_num_predict,
    resolve_thinking,
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
        "eval_count": None,
        "eval_duration_ms": None,
        "prompt_eval_count": None,
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
    # 1行回答の行動では思考生成をoff（Ollamaの think=false）で送信する
    assert client.options == [
        {"num_predict": 256, "thinking": False},
        {"num_predict": 512, "thinking": False},
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
        {"num_predict": 256, "thinking": False},
        {"num_predict": 512, "thinking": False},
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


def test_staged_claim_prompt_is_short_and_thinking_is_disabled() -> None:
    state = make_state()
    state.players[0].role = Role.SEER
    state.seer_results["P1"] = "wolf"
    client = FakeClient("【CO】占い師CO\n【結果】対象=P1 判定=人狼")
    actions = OllamaActions(state, client, discussion_claim_mode="all")
    history = (Message("P2", "CO判断に不要な古い会話内容です。"),)

    claim = actions.choose_discussion_claim(
        "P0", 1, tuple(state.alive_players), history
    )

    assert claim == "【CO】占い師CO\n【結果】対象=P1 判定=人狼"
    assert len(client.prompts[0]) < 800
    assert "CO判断に不要な古い会話内容" not in client.prompts[0]
    assert "理由・本文・前置きは不要" in client.prompts[0]
    assert "【結果】対象=P1 判定=人狼" in client.prompts[0]
    assert "個性プロファイル" not in client.prompts[0]
    assert "【本文】" not in client.prompts[0]
    assert client.options[0]["thinking"] is False
    assert actions.traces[-1].fallback_used is False


def test_staged_claim_discards_body_and_stray_result_for_no_claim() -> None:
    parsed = parse_discussion_claim(
        "【CO】COなし\n【結果】なし\n説明を誤って追加"
    )
    assert parsed is not None
    assert parsed.claim == "なし"
    assert parsed.result == ""
    assert parsed.ignored_trailing_text is True

    events: list[dict[str, object]] = []
    client = FakeClient(
        "【CO】COなし\n【結果】なし\n"
        "Player 1とPlayer 3の主張を確認したいです。"
    )
    actions = OllamaActions(
        make_state(),
        client,
        discussion_claim_mode="all",
        event_callback=events.append,
    )

    claim = actions.choose_discussion_claim(
        "P0", 1, ("P0", "P1", "P2"), ()
    )

    assert claim == "【CO】なし"
    assert len(client.prompts) == 1
    assert actions.traces[-1].fallback_used is False
    assert any(
        event["type"] == "discussion_claim_extra_text_discarded"
        for event in events
    )
    completion = next(
        event
        for event in events
        if event["type"] == "llm_action_completed"
    )
    assert completion["content"] == "【CO】なし"


def test_staged_seer_claim_keeps_valid_result_and_discards_explanation() -> None:
    state = make_state()
    state.players[0].role = Role.SEER
    state.seer_results["P1"] = "wolf"
    client = FakeClient(
        "【CO】占い師CO\n【結果】対象=P1 判定=人狼\n"
        "この結果を見て投票先を決めました。"
    )
    actions = OllamaActions(state, client, discussion_claim_mode="all")

    claim = actions.choose_discussion_claim(
        "P0", 1, tuple(state.alive_players), ()
    )

    assert claim == "【CO】占い師CO\n【結果】対象=P1 判定=人狼"
    assert len(client.prompts) == 1
    assert actions.traces[-1].fallback_used is False


def test_invalid_staged_claim_retries_once_then_deterministically_publishes_wolf_result() -> None:
    state = make_state()
    state.players[0].role = Role.SEER
    state.seer_results["P1"] = "wolf"
    client = FakeClient("占い師だと思う", "【CO】なし")
    actions = OllamaActions(state, client, discussion_claim_mode="all")

    claim = actions.choose_discussion_claim(
        "P0", 1, tuple(state.alive_players), ()
    )

    assert claim == "【CO】占い師CO\n【結果】対象=P1 判定=人狼"
    assert len(client.prompts) == 2
    assert "COだけを再回答してください" in client.prompts[1]
    assert actions.traces[-1].fallback_used is True
    assert "wolf_result_not_disclosed" in actions.traces[-1].failure_reasons


def test_discussion_claim_modes_stage_all_or_only_seer_and_existing_claimants() -> None:
    state = make_state()
    legacy = OllamaActions(state, FakeClient(), discussion_claim_mode="legacy")
    candidates = OllamaActions(
        state, FakeClient(), discussion_claim_mode="candidates"
    )
    all_players = OllamaActions(state, FakeClient(), discussion_claim_mode="all")

    assert not legacy.should_stage_discussion_claim("P0")
    assert not candidates.should_stage_discussion_claim("P0")
    assert all_players.should_stage_discussion_claim("P0")

    state.players[0].role = Role.SEER
    assert candidates.should_stage_discussion_claim("P0")


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
        {"temperature": 0.4, "num_ctx": 16384, "num_predict": 192, "thinking": True}
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
        "thinking": True,
    }
    # 議論は思考生成をonのまま送る
    assert configuration["thinking_enabled"] is True
    assert configuration["thinking_source"] == "default_by_action"
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
    # 初回議論は思考on、生成起因の再要求は思考offで再送する
    assert client.options[0] == {"num_predict": 256, "thinking": True}
    assert client.options[1] == {"num_predict": 512, "thinking": False}
    assert actions.traces[-1].failure_reasons == (reason,)
    assert actions.traces[-1].response_metadata[0] == {
        "done_reason": done_reason,
        "content_length": len(content),
        "thinking_present": thinking,
        "eval_count": None,
        "eval_duration_ms": None,
        "prompt_eval_count": None,
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


def test_discussion_dead_player_question_is_retried_to_a_living_player() -> None:
    state = make_state()
    state.mark_dead("P1")
    client = FakeClient(
        "P1さんに初夜の占い理由を説明してもらいたいです。",
        "P1さんに初夜の占い理由を説明してもらいたいです。"
        "P2さんに投票先の根拠を説明してください。",
    )
    events: list[dict[str, object]] = []
    actions = OllamaActions(state, client, event_callback=events.append)

    answer = actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert "P1さんは死亡しており回答できないため" in answer
    assert "P2さんに記録から分かる経緯を確認します。" in answer
    assert "P2さんに投票先の根拠を説明してください。" in answer
    assert actions.traces[-1].failure_reasons == ("dead_player_question",)
    assert any(
        event["type"] == "discussion_integrity_repaired"
        and event["reason"] == "dead_player_question"
        for event in events
    )
    assert "死亡者は回答できません" in client.prompts[1]


@pytest.mark.parametrize(
    ("speech", "expected_issue"),
    [
        ("P1は以前のゲームで人狼だったそうです。", "external_game_memory"),
        ("P1は昨日の投票でP2に投票したそうです。", "unverified_past_vote_reference"),
    ],
)
def test_unavailable_external_memory_and_past_vote_are_retried(
    speech: str, expected_issue: str
) -> None:
    state = make_state()
    client = FakeClient(
        speech,
        "その情報は現在のゲーム記録にないため、記録にないため判断できません。"
        "現在の発言と投票を確認します。",
    )
    events: list[dict[str, object]] = []
    actions = OllamaActions(state, client, event_callback=events.append)

    answer = actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert "記録にないため判断できない" in client.prompts[1]
    assert answer.startswith("その情報は現在のゲーム記録にないため")
    assert expected_issue in actions.traces[-1].failure_reasons
    assert any(
        event["type"] == "llm_action_parse_failed"
        and event["reason"] == expected_issue
        for event in events
    )


def test_external_memory_is_partially_repaired_after_a_second_invalid_response() -> None:
    state = make_state()
    client = FakeClient(
        "P1は以前のゲームで人狼だったそうです。",
        "P1は以前のゲームで人狼だったそうです。P2の投票理由を確認します。",
    )
    events: list[dict[str, object]] = []
    actions = OllamaActions(state, client, event_callback=events.append)

    answer = actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert answer.startswith("その情報は現在のゲーム記録にないため")
    assert "P2の投票理由を確認します。" in answer
    assert "以前のゲームで人狼だった" not in answer
    assert actions.traces[-1].failure_reasons == ("external_game_memory",)
    assert any(
        event["type"] == "discussion_integrity_repaired"
        and event["reason"] == "external_game_memory"
        for event in events
    )


def test_logged_past_vote_and_death_are_preserved_as_current_game_facts() -> None:
    state = make_state()
    state.day = 2
    state.mark_dead("P1")
    state.vote_history.append((1, False, {"P0": "P1", "P2": "P1"}))
    state.executions.append((1, "P1"))
    response = "昨日の投票記録ではP1に票が集まり、P1が処刑されました。"
    client = FakeClient(response)
    actions = OllamaActions(state, client)

    answer = actions.discuss("P0", 2, tuple(state.alive_players), ())

    assert answer == response
    assert len(client.prompts) == 1


def test_yesterdays_vote_must_match_that_day_not_an_older_vote() -> None:
    state = make_state()
    state.day = 3
    state.vote_history.extend(
        [
            (1, False, {"P2": "P1"}),
            (2, False, {"P2": "P0"}),
        ]
    )
    client = FakeClient(
        "昨日の投票ではP2がP1に投票しました。",
        "昨日の投票結果は現在のログを確認してから判断します。",
    )
    actions = OllamaActions(state, client)

    answer = actions.discuss("P0", 3, tuple(state.alive_players), ())

    assert answer == "昨日の投票結果は現在のログを確認してから判断します。"
    assert actions.traces[-1].failure_reasons == (
        "unverified_past_vote_reference",
    )


def test_dead_player_mention_and_hedged_role_guess_are_allowed() -> None:
    state = make_state()
    state.mark_dead("P1")
    responses = (
        "P1の過去の発言を参考に、P2へ投票理由を確認します。",
        "P1は人狼だったかもしれませんが、真の役職は未確認です。",
    )
    for response in responses:
        client = FakeClient(response)
        actions = OllamaActions(state, client)
        assert actions.discuss("P0", 1, tuple(state.alive_players), ()) == response
        assert len(client.prompts) == 1


def test_repeated_question_is_detected_across_the_whole_current_day() -> None:
    state = make_state()
    history = (
        Message("P2", "P1さんに、前の発言と違う理由を説明してください。"),
        Message("P1", "この点はあとで考えます。"),
        Message("P2", "別の参加者には投票理由を聞きました。"),
        Message("P1", "追加の根拠を探してみます。"),
        Message("P2", "今日は投票先を慎重に考えましょう。"),
        Message("P1", "まだ別の意見も確認したいです。"),
    )
    client = FakeClient(
        "P1さんに、前の発言と違う理由を説明してください。",
        "P2の投票理由を新しい根拠として確認します。",
    )
    actions = OllamaActions(state, client)

    answer = actions.discuss("P0", 1, tuple(state.alive_players), history)

    assert answer == "P2の投票理由を新しい根拠として確認します。"
    assert actions.traces[-1].failure_reasons == ("repeated_discussion_topic",)
    assert "当日の議論ですでに出た質問・結論を繰り返さず" in client.prompts[1]


def test_same_topic_with_a_new_vote_target_or_evidence_is_not_a_repeat() -> None:
    state = make_state()
    history = (
        Message("P2", "P1さんの投票先と理由を説明してください。"),
    )
    actions = OllamaActions(state, FakeClient())

    assert not actions._discussion_topic_repeated(
        "P1さんの投票先について、新しい根拠からP2へ投票します。",
        history,
    )
    assert not actions._discussion_topic_repeated(
        "P1さんの投票先と理由を確認します。反証は発言記録にあります。",
        history,
    )


def test_soft_discussion_issue_keeps_valid_body_after_retry() -> None:
    state = make_state()
    history = (
        Message("P2", "P0さんはどこを疑っていますか？"),
    )
    client = FakeClient(
        "P1を疑います。投票記録の矛盾が根拠です。",
        "【CO】壊れた\n【本文】P1を疑います。投票記録の矛盾が根拠です。",
    )
    events: list[dict[str, object]] = []
    actions = OllamaActions(state, client, event_callback=events.append)

    response = actions.discuss("P0", 1, tuple(state.alive_players), history)

    assert "P1を疑います。" in response
    assert response != "パス"
    assert actions.traces[-1].fallback_used is False
    assert any(event["type"] == "llm_discussion_soft_issue_kept" for event in events)


def test_repetition_check_does_not_cross_a_new_day_boundary() -> None:
    state = make_state()
    state.day = 2
    repeated_question = "P1さんに、前の発言と違う理由を説明してください。"
    state.conversation_log.append(Message("P2", repeated_question))
    GameEngine(state, OllamaActions(state, FakeClient())).resolve_morning()
    state.conversation_log.append(
        Message("P2", "新しい日の議論を始めましょう。")
    )
    history = tuple(state.conversation_log)
    response = "P1さんに、前の発言と違う理由を説明してください。"
    client = FakeClient(response)
    actions = OllamaActions(state, client)

    answer = actions.discuss("P0", 1, tuple(state.alive_players), history)

    assert answer == response
    assert len(client.prompts) == 1


def test_action_lifecycle_events_include_reprompt_fallback_and_timing() -> None:
    client = FakeClient("投票：unknown", "投票：unknown")
    events: list[dict[str, object]] = []
    actions = OllamaActions(make_state(), client, event_callback=events.append)

    actions.vote("P0", ("P0", "P1", "P2"), day=1, runoff=False)

    lifecycle_events = [
        event
        for event in events
        if event["type"]
        in {
            "llm_action_started",
            "llm_action_parse_failed",
            "llm_action_completed",
        }
    ]
    request_events = [
        event for event in events if event["type"] == "llm_request_configuration"
    ]
    assert [event["type"] for event in lifecycle_events] == [
        "llm_action_started",
        "llm_action_parse_failed",
        "llm_action_parse_failed",
        "llm_action_completed",
    ]
    assert lifecycle_events[0]["player"] == "P0"
    assert lifecycle_events[0]["action"] == "vote"
    assert lifecycle_events[0]["private"] is False
    assert lifecycle_events[1]["failure_class"] == "invalid_target"
    assert lifecycle_events[1]["will_retry"] is True
    assert lifecycle_events[2]["will_retry"] is False
    assert lifecycle_events[3]["semantic_retries"] == 1
    assert lifecycle_events[3]["fallback_used"] is True
    assert isinstance(lifecycle_events[3]["elapsed_seconds"], float)
    assert len(request_events) == 2
    assert request_events[0]["raw_prompt_char_count"] > 0
    assert request_events[0]["requested_num_ctx"] is None


# ---------------------------------------------------------------------------
# TASK-022 占い師COの確実な公開と非対応モデル向け短縮経路
# ---------------------------------------------------------------------------


def make_seer_state(result: str = "human", seed: int = 31) -> GameState:
    state = make_state(seed)
    state.players[0].role = Role.SEER
    state.players[2].role = Role.VILLAGER
    state.seer_results["P2"] = result
    state.seer_result_history.append((1, "P2", result))
    return state


class EndlessClient(FakeClient):
    """用意した応答を使い切った後は、安全な既定応答を返すクライアント。"""

    default_response = "公開情報を確認して投票先を考えます。"

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, object] | None = None,
    ) -> str:
        self.prompts.append(messages[-1]["content"])
        self.options.append(options)
        self.timeouts.append(getattr(self, "timeout", None))
        if not self.responses:
            return self.default_response
        response = self.responses.pop(0)
        if isinstance(response, OllamaError):
            raise response
        return response


def test_minimal_co_and_result_blocks_are_accepted_without_retry() -> None:
    state = make_seer_state("human")
    minimal = "【CO】占い師\n【結果】対象=P2 判定=人間"
    client = FakeClient(minimal)
    actions = OllamaActions(state, client)

    response = actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert response == minimal
    assert len(client.prompts) == 1
    assert actions.traces[-1].fallback_used is False
    assert actions.traces[-1].failure_reasons == ()


def test_minimal_co_and_result_blocks_with_body_are_accepted() -> None:
    state = make_seer_state("wolf")
    minimal = (
        "【CO】占い師\n"
        "【結果】対象=P2 判定=人狼\n"
        "【本文】P2への投票を提案します。"
    )
    client = FakeClient(minimal)
    actions = OllamaActions(state, client)

    response = actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert response == minimal
    assert len(client.prompts) == 1
    assert actions.traces[-1].fallback_used is False


def test_result_only_response_is_corrected_by_a_short_co_prompt() -> None:
    state = make_seer_state("human")
    corrected = "【CO】占い師\n【結果】対象=P2 判定=人間"
    client = FakeClient("【CO】なし\n【結果】対象=P2 判定=人間", corrected)
    events: list[dict[str, object]] = []
    actions = OllamaActions(state, client, event_callback=events.append)

    response = actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert response == corrected
    # CO補正は最大1回だけ送る
    assert len(client.prompts) == 2
    correction = client.prompts[1]
    assert correction.startswith("【CO補正】")
    assert "同じ発言内で明示的に占い師COしてください" in correction
    assert "【CO】占い師" in correction
    assert "【結果】対象=Player X 判定=人狼/人間" in correction
    # 長い再要求ではなく、短い補正プロンプトを使う
    assert len(correction) < len(client.prompts[0])
    # 占い師本人の結果だけを補正プロンプトへ含める
    assert "対象=P2、判定=人間" in correction
    assert actions.traces[-1].fallback_used is False
    assert [
        event["will_retry"]
        for event in events
        if event["type"] == "seer_result_claim_missing"
    ] == [True]


def test_result_only_response_without_markers_is_corrected_by_short_prompt() -> None:
    state = make_seer_state("wolf")
    corrected = (
        "占い師CO：対象=P2、判定=人狼、理由=夜の占い結果、投票提案=P2へ投票します。"
    )
    client = FakeClient("P2は人狼と判定された。", corrected)
    actions = OllamaActions(state, client)

    response = actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert response == corrected
    assert len(client.prompts) == 2
    assert client.prompts[1].startswith("【CO補正】")
    assert "同じ発言内で明示的に占い師COしてください" in client.prompts[1]
    assert actions.traces[-1].fallback_used is False


def test_failed_co_correction_falls_back_to_structured_co() -> None:
    state = make_seer_state("wolf")
    client = FakeClient("P2は人狼と判定された。", "P2は人狼と判定された。")
    events: list[dict[str, object]] = []
    actions = OllamaActions(state, client, event_callback=events.append)

    response = actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert response.startswith("【CO】占い師CO：対象=P2、判定=人狼")
    assert "【本文】" in response
    assert len(client.prompts) == 2
    assert actions.traces[-1].fallback_used is True
    assert [
        event["will_retry"]
        for event in events
        if event["type"] == "seer_result_claim_missing"
    ] == [True, False]
    assert any(event["type"] == "seer_result_claim_fallback" for event in events)
    assert any(
        event["type"] == "llm_action_parse_failed"
        and event["reason"] == "seer_result_without_claim"
        and event["will_retry"] is False
        for event in events
    )


def test_natural_language_seer_co_is_accepted_without_correction() -> None:
    state = make_seer_state("wolf")
    natural = "私は占い師です。対象=P2、判定=人狼。"
    client = FakeClient(natural)
    actions = OllamaActions(state, client)

    response = actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert response == natural
    assert len(client.prompts) == 1
    assert actions.traces[-1].fallback_used is False


def test_result_block_without_co_is_never_returned_for_non_seer() -> None:
    state = make_state()
    client = FakeClient(
        "【CO】なし\n【結果】対象=P2 判定=人狼",
        "【CO】なし\n【結果】対象=P2 判定=人狼",
    )
    actions = OllamaActions(state, client)

    response = actions.discuss("P0", 1, tuple(state.alive_players), ())

    # COが成立しない結果は、他プレイヤーへ公開せずに安全側へ落とす
    assert response == "パス"
    assert len(client.prompts) == 2
    assert client.prompts[1].startswith("【CO補正】")
    assert actions.traces[-1].fallback_used is True
    assert state.public_role_claims == []
    assert state.seer_claims == []


def test_seer_co_correction_result_is_published_to_the_next_player() -> None:
    state = make_seer_state("human")
    client = EndlessClient(
        "【CO】なし\n【結果】対象=P2 判定=人間",
        "【CO】なし\n【結果】対象=P2 判定=人間",
        "状況を確認して投票先を考えます。",
        "他の発言も確認します。",
        "投票先をもう少し考えます。",
        "P0の占い師COを確認しました。",
        "公開情報を整理します。",
        "今日は全体を見て判断します。",
        "ほかの意見も確認します。",
    )
    events: list[dict[str, object]] = []
    actions = OllamaActions(state, client, event_callback=events.append)

    GameEngine(state, actions, event_callback=events.append).run_discussion()

    claims = [
        (claim.claimant, claim.claimed_role, claim.target, claim.result)
        for claim in state.public_role_claims
    ]
    # 補正に失敗した後の構造化COが、公開状態へ即時に登録される
    assert claims == [
        ("P0", "seer", None, None),
        ("P0", "seer", "P2", "human"),
    ]
    # 次のプレイヤーのプロンプトへCOと結果が反映される
    assert "P0は占い師を主張（P2：人間）" in client.prompts[2]
    # 会話ログへは本文だけが入り、CO・結果ブロックや未成の結果は残らない
    assert all(
        "【CO】" not in message.content
        and "【結果】" not in message.content
        and "対象=P2" not in message.content
        for message in state.conversation_log
        if message.speaker != "system"
    )
    assert any(event["type"] == "seer_result_claim_fallback" for event in events)


def test_wolf_seer_result_is_published_as_one_public_event() -> None:
    state = make_seer_state("wolf")
    client = EndlessClient(
        "【CO】占い師\n"
        "【結果】対象=P2 判定=人狼\n"
        "【本文】P2への投票を提案します。",
        "公開情報を確認して投票先を考えます。",
        "P0の占い師COを確認しました。",
        "他の発言も確認します。",
        "投票先をもう少し考えます。",
        "状況を整理します。",
        "今日は全体を見て判断します。",
        "ほかの意見も確認します。",
        "議論を続けましょう。",
    )
    events: list[dict[str, object]] = []
    actions = OllamaActions(state, client, event_callback=events.append)

    GameEngine(state, actions, event_callback=events.append).run_discussion()

    # COと人狼結果が、連続した同一の公開イベント列として記録される
    assert [
        event["type"]
        for event in events
        if event["type"]
        in {"claim_role_published", "seer_claim", "seer_result_claim"}
    ] == ["claim_role_published", "seer_claim", "seer_result_claim"]
    result_event = next(
        event for event in events if event["type"] == "seer_result_claim"
    )
    assert (result_event["target"], result_event["claimed_result"]) == (
        "P2",
        "wolf",
    )
    # 村人側のプロンプトへCOと結果が確実に反映される
    assert "P0は占い師を主張（P2：人狼）" in client.prompts[1]
    assert any(
        claim.claimant == "P0"
        and claim.claimed_role == "seer"
        and claim.target == "P2"
        and claim.result == "wolf"
        for claim in state.public_role_claims
    )
    assert state.conversation_log[0].content == "P2への投票を提案します。"


def test_tool_unsupported_model_is_recorded_once_and_stops_tool_trials() -> None:
    class ToolUnsupportedClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.tool_requests = 0
            self.plain_requests = 0

        def chat_with_metadata(self, model, messages, options=None, tools=None):
            assert tools is not None
            self.tool_requests += 1
            raise OllamaError(
                "http",
                "model does not support tools",
                attempts=1,
                status_code=400,
            )

        def chat(self, model, messages, options=None):
            self.plain_requests += 1
            return "【CO】なし\n【本文】公開情報を確認して投票先を考えます。"

    state = make_state()
    events: list[dict[str, object]] = []
    client = ToolUnsupportedClient()
    actions = OllamaActions(state, client, event_callback=events.append)

    for _ in range(3):
        actions.discuss("P0", 1, tuple(state.alive_players), ())

    # ツール非対応の事実をゲーム中の能力情報へ記録する
    assert state.model_capabilities == {"model-0": {"claim_role_tool": False}}
    # 判明した同一モデルへ、以後ツールを再送しない
    assert client.tool_requests == 1
    assert client.plain_requests == 3
    assert [event["type"] for event in events].count("claim_tool_unavailable") == 1
    # 文章CO経路だけで通常どおり処理される
    assert actions.traces[-1].fallback_used is False
    assert state.public_role_claims == []

def test_action_num_predict_prefers_player_per_action_over_global_and_cli() -> None:
    player = Player(
        name="P0",
        model="model-0",
        role=Role.VILLAGER,
        num_predict=1024,
        num_predict_by_action={"discussion": 512, "attack": 128},
    )
    cli_options = {
        "num_predict": 256,
        "num_predict_by_action:discussion": 4096,
        "num_predict_by_action:seer": 320,
    }

    assert resolve_action_num_predict(player, cli_options, "discussion") == (
        512,
        "player_by_action",
    )
    assert resolve_action_num_predict(player, cli_options, "attack") == (
        128,
        "player_by_action",
    )
    # 行動別設定が無い対象選択行動では、CLIの行動別上書きが全体設定より優先される
    assert resolve_action_num_predict(player, cli_options, "seer") == (
        320,
        "cli_by_action",
    )


def test_target_actions_cap_large_global_and_discussion_keeps_it() -> None:
    """思考系モデルの大きな全体設定は、対象選択・CO補正だけキャップする。"""
    player = Player(name="P0", model="model-0", role=Role.VILLAGER, num_predict=4096)

    assert resolve_action_num_predict(player, {}, "vote") == (768, "global_capped")
    assert resolve_action_num_predict(player, {}, "attack") == (768, "global_capped")
    assert resolve_action_num_predict(player, {}, "seer") == (768, "global_capped")
    assert resolve_action_num_predict(player, {}, "co_correction") == (384, "global_capped")
    # 議論・人狼相談は根拠を説明できる文量を確保するため全体設定をそのまま使う
    assert resolve_action_num_predict(player, {}, "discussion") == (4096, "player_global")
    assert resolve_action_num_predict(player, {}, "wolf_discussion") == (4096, "player_global")

    # 全体設定がキャップ以下ならそのまま（明示的な短くしを尊重する）
    small = Player(name="P1", model="model-1", role=Role.VILLAGER, num_predict=256)
    assert resolve_action_num_predict(small, {}, "vote") == (256, "player_global")


def test_action_num_predict_falls_back_to_cli_and_action_defaults() -> None:
    player = Player(name="P0", model="model-0", role=Role.VILLAGER)
    cli_options = {"num_predict": 256, "num_predict_by_action:discussion": 4096}

    assert resolve_action_num_predict(player, cli_options, "discussion") == (
        4096,
        "cli_by_action",
    )
    assert resolve_action_num_predict(player, cli_options, "vote") == (
        256,
        "cli_global",
    )
    assert resolve_action_num_predict(player, {}, "wolf_discussion") == (
        768,
        "default_by_action",
    )
    assert resolve_action_num_predict(None, {}, "vote") == (256, "default_by_action")


def test_action_budget_key_aliases_are_normalized() -> None:
    assert normalize_action_budget_key("wolf_consult") == "wolf_discussion"
    assert normalize_action_budget_key("target") == "vote"
    assert normalize_action_budget_key("discussion_seer_co") == "co_correction"
    assert normalize_action_budget_key("WOLF_CONSULT") == "wolf_discussion"
    assert normalize_action_budget_key("unknown_key") == "discussion"


def test_request_options_use_player_budgets_and_drop_internal_action_keys() -> None:
    state = make_state()
    state.players[0].num_ctx = 8192
    state.players[0].num_predict = None
    state.players[0].num_predict_by_action = {"discussion": 512}
    client = FakeClient("議論を続けましょう。")
    actions = OllamaActions(
        state,
        client,
        options={
            "temperature": 0.4,
            "num_predict": 256,
            "num_predict_by_action:discussion": 4096,
        },
    )

    actions.discuss("P0", 1, ("P0", "P1", "P2"), ())

    # 内部的な行動別キーはLLM送信オプションへ混ぜない
    assert client.options[0] == {
        "temperature": 0.4,
        "num_ctx": 8192,
        "num_predict": 512,
        "thinking": True,
    }
    assert not any(
        key.startswith("thinking_by_action:") for key in client.options[0] or {}
    )


def test_target_selection_uses_action_default_without_player_settings() -> None:
    state = make_state()
    client = FakeClient("投票：P1")
    actions = OllamaActions(state, client, options={"temperature": 0.4})

    target = actions.vote("P0", ("P0", "P1", "P2"), day=1, runoff=False)

    assert target == "P1"
    # 投票は思考オフで送り、1行回答の生成だけを行う
    assert client.options[0] == {"temperature": 0.4, "num_predict": 256, "thinking": False}


def test_target_selection_caps_large_global_and_escalates_retry_budget() -> None:
    """TASK-024: 大きな全体設定は対象選択だけキャップし、生成起因の再試行では上限を増やす。"""
    from werewolf import ChatResponse

    class MetadataClient(FakeClient):
        def chat(self, model, messages, options=None):
            self.prompts.append(messages[-1]["content"])
            self.options.append(options)
            if len(self.prompts) == 1:
                self._last_chat_response = ChatResponse(
                    content="",
                    attempts=1,
                    done_reason="length",
                    content_length=0,
                    thinking_present=True,
                )
                return ""
            return "投票：P1"

    state = make_state()
    state.players[0].num_predict = 4096
    client = MetadataClient()
    actions = OllamaActions(state, client, options={"temperature": 0.4})

    target = actions.vote("P0", ("P0", "P1", "P2"), day=1, runoff=False)

    assert target == "P1"
    assert client.options[0] == {"temperature": 0.4, "num_predict": 768, "thinking": False}
    assert client.options[1] == {"temperature": 0.4, "num_predict": 1536, "thinking": False}
    assert actions.traces[-1].failure_reasons == ("generation_truncated",)
    assert actions.traces[-1].fallback_used is False


def test_co_correction_request_uses_its_own_action_budget() -> None:
    """TASK-024: CO補正は議論より小さい行動別上限を使い、本文の長文生成を抑制する。"""
    state = make_seer_state("wolf")
    state.players[0].num_predict_by_action = {"discussion": 1024, "co_correction": 256}
    corrected = (
        "【CO】占い師\n"
        "【結果】対象=P2 判定=人狼\n"
        "【本文】初夜の占い結果です。P2への投票を提案します。"
    )
    client = FakeClient("P2は人狼と判定された。", corrected)
    actions = OllamaActions(state, client, options={"temperature": 0.4})

    response = actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert response == corrected
    assert len(client.prompts) == 2
    assert client.options[0] == {"temperature": 0.4, "num_predict": 1024, "thinking": True}
    assert client.options[1] == {"temperature": 0.4, "num_predict": 256, "thinking": False}
    # 補正プロンプトには行動別上限のガイダンスが反映される
    assert "num_predict=256" in client.prompts[1]
    # 思考オフでは「思考を書かない」指示をプロンプトへ明記する
    assert "思考生成（think）はオフ" in client.prompts[1]


def test_default_action_budgets_keep_co_correction_above_one_line() -> None:
    """CO補正の既定上限は最小3行（CO・結果・本文）を生成できる量にする。"""
    assert resolve_action_num_predict(None, {}, "co_correction") == (
        256,
        "default_by_action",
    )
    # TASK-024: 議論は文量緩和、人狼相談は議論より短いが提案・理由を書ける量
    assert resolve_action_num_predict(None, {}, "discussion") == (
        1536,
        "default_by_action",
    )
    assert resolve_action_num_predict(None, {}, "wolf_discussion") == (
        768,
        "default_by_action",
    )
    assert resolve_action_num_predict(None, {}, "vote") == (256, "default_by_action")


def test_thinking_floor_only_applies_to_unconfigured_action_defaults() -> None:
    """TASK-024: 思考onの生成上限下限は設定未指定（行動別既定値）のときだけ適用する。"""
    assert apply_thinking_floor(1536, "default_by_action", True, "discussion") == (
        3072,
        THINKING_NUM_PREDICT_SOURCE,
    )
    assert apply_thinking_floor(768, "default_by_action", True, "wolf_discussion") == (
        1536,
        THINKING_NUM_PREDICT_SOURCE,
    )
    # 別名キーも正規化して下限を適用する
    assert apply_thinking_floor(768, "default_by_action", True, "wolf_consult") == (
        1536,
        THINKING_NUM_PREDICT_SOURCE,
    )
    # 明示設定（player行動別・CLI行動別・全体・キャップ）は下限で上書きしない
    assert apply_thinking_floor(512, "player_by_action", True, "discussion") == (
        512,
        "player_by_action",
    )
    assert apply_thinking_floor(1024, "cli_by_action", True, "discussion") == (
        1024,
        "cli_by_action",
    )
    assert apply_thinking_floor(256, "player_global", True, "discussion") == (
        256,
        "player_global",
    )
    assert apply_thinking_floor(768, "global_capped", True, "vote") == (768, "global_capped")
    # 思考off・下限のない行動・上限未解決は変更しない
    assert apply_thinking_floor(1536, "default_by_action", False, "discussion") == (
        1536,
        "default_by_action",
    )
    assert apply_thinking_floor(256, "default_by_action", True, "vote") == (
        256,
        "default_by_action",
    )
    assert apply_thinking_floor(None, "default_by_action", True, "discussion") == (
        None,
        "default_by_action",
    )
    # 下限以上の既定値はそのまま
    assert apply_thinking_floor(4096, "default_by_action", True, "discussion") == (
        4096,
        "default_by_action",
    )


def test_thinking_floor_reaches_discussion_request_and_prompt_guidance() -> None:
    """TASK-024: 設定未指定の議論は下限3072を実際に送信し、プロンプト表記も一致する。"""
    state = make_state()
    client = FakeClient("【CO】なし\n【本文】P1を疑います。根拠は初動の薄さです。")
    actions = OllamaActions(state, client, options={"temperature": 0.4})

    actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert client.options[0] == {"temperature": 0.4, "num_predict": 3072, "thinking": True}
    assert "num_predict=3072" in client.prompts[0]


def test_explicit_discussion_budget_is_not_raised_by_thinking_floor() -> None:
    """TASK-024: player行動別設定の議論上限は思考onでも変更しない。"""
    state = make_state()
    state.players[0].num_predict_by_action = {"discussion": 640}
    client = FakeClient("【CO】なし\n【本文】P2の投票理由を確認します。")
    actions = OllamaActions(state, client, options={"temperature": 0.4})

    actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert client.options[0] == {"temperature": 0.4, "num_predict": 640, "thinking": True}
    assert "num_predict=640" in client.prompts[0]


def test_thinking_defaults_keep_reasoning_only_for_discussion_actions() -> None:
    """TASK-024: 思考生成は議論・人狼相談だけon、1行回答の行動はoff。"""
    assert resolve_thinking(None, {}, "discussion") == (True, "default_by_action")
    assert resolve_thinking(None, {}, "wolf_discussion") == (True, "default_by_action")
    for budget_key in ("attack", "seer", "vote", "co_correction"):
        assert resolve_thinking(None, {}, budget_key) == (False, "default_by_action")
    # 別名キー（target）も投票扱いとしてoffに寄せる
    assert resolve_thinking(None, {}, "target") == (False, "default_by_action")


def test_thinking_resolution_priority_matches_documented_order() -> None:
    assert THINKING_RESOLUTION_PRIORITY == (
        "player_by_action",
        "cli_by_action",
        "player_global",
        "cli_global",
        "default_by_action",
    )

    player = Player(
        name="P0",
        model="m",
        role=Role.VILLAGER,
        thinking=True,
        thinking_by_action={"discussion": False},
    )
    cli_options = {"thinking": False, "thinking_by_action:vote": True}

    assert resolve_thinking(player, cli_options, "discussion") == (
        False,
        "player_by_action",
    )
    assert resolve_thinking(player, cli_options, "vote") == (True, "cli_by_action")
    assert resolve_thinking(player, cli_options, "seer") == (True, "player_global")
    assert resolve_thinking(None, {"thinking": False}, "attack") == (False, "cli_global")
    assert resolve_thinking(None, None, "discussion") == (True, "default_by_action")


def test_thinking_option_is_sent_as_ollama_request_think_field() -> None:
    """アプリの`thinking`設定はOllamaリクエストの`think`だけへ反映する。"""
    payloads: list[dict[str, object]] = []

    def transport(request, timeout):
        payloads.append(json.loads(request.data.decode("utf-8")))
        return json.dumps(
            {"message": {"content": "投票：P1"}, "done_reason": "stop"}
        ).encode("utf-8")

    client = OllamaClient(timeout=5.0, transport=transport)
    client.chat(
        "model-x",
        [{"role": "user", "content": "投票してください"}],
        options={"num_predict": 256, "thinking": False},
    )
    assert payloads[0]["think"] is False
    assert payloads[0]["options"] == {"num_predict": 256}
    assert "thinking" not in payloads[0]["options"]

    client.chat(
        "model-x",
        [{"role": "user", "content": "発言してください"}],
        options={"num_predict": 1024, "thinking": True},
    )
    assert payloads[1]["think"] is True

    # 未設定時は`think`を送らず、Ollama側既定に委ねる
    client.chat(
        "model-x",
        [{"role": "user", "content": "投票してください"}],
        options={"num_predict": 256},
    )
    assert "think" not in payloads[2]


def test_player_thinking_override_reaches_target_selection_request() -> None:
    state = make_state()
    state.players[0].thinking_by_action = {"vote": True}
    client = FakeClient("投票：P1")
    actions = OllamaActions(
        state, client, options={"temperature": 0.4, "thinking": False}
    )

    target = actions.vote("P0", ("P0", "P1", "P2"), day=1, runoff=False)

    assert target == "P1"
    # player行動別設定がCLI全体の思考offより優先される
    assert client.options[0] == {
        "temperature": 0.4,
        "num_predict": 256,
        "thinking": True,
    }


def test_discussion_retry_forces_thinking_off_and_short_prompt() -> None:
    """TASK-024: 空応答の再要求は思考off＋短い必須項目だけで再送する。"""
    state = make_state()
    corrected = (
        "【CO】なし\n"
        "【本文】P1を疑います。根拠は初日の反応の薄さです。P2に質問します。"
    )

    class DiagnosticClient(FakeClient):
        def context_diagnostics(self, model):
            return {"ollama_ps_context": 8192}

    client = DiagnosticClient("", corrected)
    events: list[dict[str, object]] = []
    actions = OllamaActions(
        state, client, options={"temperature": 0.4}, event_callback=events.append
    )

    response = actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert response == corrected
    configurations = [
        event for event in events if event["type"] == "llm_request_configuration"
    ]
    assert len(configurations) == 2
    assert configurations[0]["thinking_enabled"] is True
    assert configurations[0]["thinking_source"] == "default_by_action"
    assert configurations[1]["thinking_enabled"] is False
    assert configurations[1]["thinking_source"] == "retry_thinking_off"
    assert configurations[1]["request_reason"] == "軽い再要求"
    # 思考オフでは「思考を書かない」指示をプロンプトへ明記する
    assert "思考生成（think）はオフ" in client.prompts[1]
    assert client.options[1]["thinking"] is False
    # 初回議論は思考onの下限3072を送るため、プロンプトの上限表記も3072に一致する
    assert "num_predict=3072" in client.prompts[0]
    assert configurations[0]["num_predict"] == 3072
    assert configurations[0]["num_predict_source"] == "thinking_allowance"
    assert "num_predict=3072" in client.prompts[1]
    assert client.options[1]["num_predict"] == 3072


def test_target_selection_retry_records_thinking_off_source() -> None:
    class DiagnosticClient(FakeClient):
        def context_diagnostics(self, model):
            return {"ollama_ps_context": 8192}

    state = make_state()
    client = DiagnosticClient("", "投票：P1")
    events: list[dict[str, object]] = []
    actions = OllamaActions(
        state, client, options={"temperature": 0.4}, event_callback=events.append
    )

    target = actions.vote("P0", ("P0", "P1", "P2"), day=1, runoff=False)

    assert target == "P1"
    configurations = [
        event for event in events if event["type"] == "llm_request_configuration"
    ]
    assert [event["thinking_enabled"] for event in configurations] == [False, False]
    assert [event["thinking_source"] for event in configurations] == [
        "default_by_action",
        "retry_thinking_off",
    ]
    # 生成起因の再試行では上限を倍へ増やす
    assert [event["num_predict"] for event in configurations] == [256, 512]
    # 再要求は長い原文を再送せず、必須項目だけの短いプロンプトにする
    assert "選択可能な候補:" not in client.prompts[1]


def test_retry_prompt_states_the_limit_actually_sent() -> None:
    """TASK-024: 再要求プロンプトの上限・思考表記は実際の送信値と一致する。"""
    state = make_state()
    client = FakeClient("", "投票：P1")
    actions = OllamaActions(state, client, options={"temperature": 0.4})

    target = actions.vote("P0", ("P0", "P1", "P2"), day=1, runoff=False)

    assert target == "P1"
    # 初回は行動別既定256、再要求は倍の512を送る
    assert "num_predict=256" in client.prompts[0]
    assert "num_predict=512" in client.prompts[1]
    assert "思考生成（think）はオフ" in client.prompts[1]
    assert client.options[0]["num_predict"] == 256
    assert client.options[1]["num_predict"] == 512


def test_model_rejecting_thinking_resends_once_with_think_off_and_records_capability() -> None:
    """TASK-024: think非対応モデルの400は同じ要求を再送せず、think=offで1回だけ再送する。"""

    class DiagnosticClient(FakeClient):
        def context_diagnostics(self, model):
            return {"ollama_ps_context": 8192}

    state = make_state()
    rejected = OllamaError(
        "http",
        "model does not support thinking",
        attempts=1,
        status_code=400,
    )
    speech = "【CO】なし\n【本文】P1の発言が曖昧なので疑います。"
    client = DiagnosticClient(rejected, speech, speech)
    events: list[dict[str, object]] = []
    actions = OllamaActions(
        state, client, options={"temperature": 0.4}, event_callback=events.append
    )

    response = actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert response == speech
    # 400→off再送のみ。意味的再試行・フォールバックとして扱わない
    assert len(client.options) == 2
    assert client.options[0]["thinking"] is True
    assert client.options[1]["thinking"] is False
    assert actions.traces[-1].raw_responses == (speech,)
    assert actions.traces[-1].fallback_used is False
    assert [event["type"] for event in events].count("llm_action_parse_failed") == 0
    # 能力情報へ記録し、イベントは初回のみ
    assert state.model_capabilities == {"model-0": {"thinking": False}}
    assert [event["type"] for event in events].count("thinking_unavailable") == 1
    configurations = [
        event for event in events if event["type"] == "llm_request_configuration"
    ]
    assert [event["thinking_enabled"] for event in configurations] == [True, False]
    assert configurations[1]["thinking_source"] == THINKING_CAPABILITY_SOURCE

    # 以後の行動も400を避けoffで送る（再度の400イベントは出ない）
    actions.discuss("P0", 1, tuple(state.alive_players), ())
    assert len(client.options) == 3
    assert client.options[-1]["thinking"] is False
    assert configurations[-1]["thinking_source"] == THINKING_CAPABILITY_SOURCE
    assert [event["type"] for event in events].count("thinking_unavailable") == 1


def test_thinking_unsupported_sends_only_two_http_requests() -> None:
    """TASK-024: 400はトランスポート再送対象ではないため、送信は初回＋off再送の2回だけ。"""
    calls: list[dict[str, object]] = []
    speech = "【CO】なし\n【本文】P1の発言が曖昧なので、P1へ質問します。"

    def transport(request, timeout):
        body = json.loads(request.data)
        calls.append(body)
        if body.get("think") is True:
            raise HTTPError(
                request.full_url,
                400,
                "Bad Request",
                {},
                BytesIO(b"model does not support thinking"),
            )
        return json.dumps({"message": {"content": speech}}).encode("utf-8")

    state = make_state()
    client = OllamaClient(transport=transport)
    actions = OllamaActions(state, client, options={"temperature": 0.4})

    response = actions.discuss("P0", 1, tuple(state.alive_players), ())

    assert response == speech
    assert len(calls) == 2
    assert calls[0]["think"] is True
    assert calls[1]["think"] is False
