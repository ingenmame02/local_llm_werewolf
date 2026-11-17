"""TASK-046: 実効コンテキスト予算の統一・予算構成分解・証拠圧縮の回帰テスト（LLM不使用）。

game_02c5a776cdb4 では requested_num_ctx=12288 に対し env=4096 / Modelfile=8192 が
記録され、設定値と実効値の境界が曖昧だった。単一値解決（effective_context_limit +
context_budget_source）、ceiling超過のnum_ctxを送らないこと、入力/出力/thinking/余白の
予算分離、必須証拠（CO・判定・投票・直近・質問・反証）の保持率100%を固定する。
モデルの実機テストは行わず、モッククライアントで検証する。
"""
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.context_budget import (
    parse_positive_int,
    process_env_context,
    resolve_effective_context_limit,
)
from werewolf.llm_actions import (
    THINKING_MARGIN_TOKENS,
    OllamaActions,
    context_budget_breakdown,
    fit_dynamic_context,
    request_token_estimate,
)
from werewolf.models import Message, Role
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game


class FakeContextClient:
    """context_diagnostics を固定値で返すモック（実Ollamaへは一切送信しない）。"""

    def __init__(self, diagnostics: dict[str, object], responses: list[str] | None = None) -> None:
        self.timeout = 120.0
        self.max_retries = 2
        self.timeout_retries = 2
        self._diagnostics = diagnostics
        self._responses = responses or ["[CO] None\n[BODY] 確認します。"]
        self._index = 0

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return self._diagnostics

    def chat(self, model: str, messages: list[dict[str, object]], options: dict | None = None) -> str:
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response


def _state_and_actor(seed: int = 1001):
    state = create_game(seed=seed, config_path="config/players.json", response_language="日本語")
    actor = next(player for player in state.players if player.role is Role.VILLAGER)
    return state, actor


def test_parse_positive_int_rejects_invalid_values() -> None:
    assert parse_positive_int("4096") == 4096
    assert parse_positive_int(0) is None
    assert parse_positive_int(True) is None
    assert parse_positive_int("abc") is None
    assert parse_positive_int(None) is None


def test_process_env_context_prefers_legacy_num_ctx(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "8192")
    value, key = process_env_context()
    assert value == 4096
    assert key == "OLLAMA_NUM_CTX"


def test_resolve_effective_context_limit_single_value() -> None:
    # 設定値が優先され、サーバー観測値との不一致は conflict で可視化される。
    resolved = resolve_effective_context_limit(
        {
            "player_setting": 12288,
            "cli_setting": 16384,
            "api_ps_loaded": 4096,
            "process_env": 4096,
            "modelfile": 8192,
        }
    )
    assert resolved.limit == 12288
    assert resolved.source == "player_setting"
    assert resolved.conflict is True
    assert resolved.candidates["modelfile"] == 8192
    # 由来別名称（レビュー指摘）: 設定由来とサーバー観測値が別値として保持される。
    assert resolved.configured == 12288
    assert resolved.configured_source == "player_setting"
    assert resolved.server_observed == 4096
    assert resolved.server_observed_source == "api_ps_loaded"

    # 設定値なしでは、観測済み実効値（/api/ps）→ env → Modelfile の順で単一値を採用。
    resolved = resolve_effective_context_limit(
        {"player_setting": None, "api_ps_loaded": 4096, "process_env": 8192, "modelfile": 12288}
    )
    assert resolved.limit == 4096
    assert resolved.source == "api_ps_loaded"
    assert resolved.conflict is False
    assert resolved.configured is None
    assert resolved.server_observed == 4096
    assert resolved.server_observed_source == "api_ps_loaded"

    resolved = resolve_effective_context_limit({"modelfile": 8192})
    assert resolved.limit == 8192
    assert resolved.source == "modelfile"

    resolved = resolve_effective_context_limit({})
    assert resolved.limit is None
    assert resolved.source is None


def test_budget_breakdown_separates_components() -> None:
    messages = [{"content": "あ" * 1000}, {"content": "b" * 1000}]
    breakdown = context_budget_breakdown(messages, 256)
    assert breakdown["input_tokens_estimate"] == 1000
    assert breakdown["generation_limit_tokens"] == 256
    assert breakdown["thinking_margin_tokens"] == 0
    assert breakdown["safety_margin_tokens"] == 512
    assert breakdown["total_needed_tokens"] == request_token_estimate(messages, 256)

    thinking = context_budget_breakdown(messages, 256, thinking_enabled=True)
    assert thinking["thinking_margin_tokens"] == THINKING_MARGIN_TOKENS
    assert thinking["total_needed_tokens"] == breakdown["total_needed_tokens"] + THINKING_MARGIN_TOKENS

    limited = context_budget_breakdown(messages, 256, effective_limit=4096)
    assert limited["effective_context_limit"] == 4096
    assert limited["headroom_tokens"] == 4096 - int(breakdown["total_needed_tokens"])
    assert limited["overflow_tokens"] == 0
    overflow = context_budget_breakdown(messages, 256, effective_limit=1000)
    assert overflow["overflow_tokens"] == int(breakdown["total_needed_tokens"]) - 1000


def test_fit_dynamic_context_never_exceeds_ceiling() -> None:
    options = {"num_ctx": 12288, "num_predict": 1024}
    messages = [{"content": "あ" * 20000}]
    fitted, needed, applied = fit_dynamic_context(options, messages, ceiling=4096)
    assert applied is True
    assert fitted["num_ctx"] == 4096
    assert fitted["num_ctx"] <= 4096
    assert needed > 4096
    # ceiling以内で必要量がnum_ctxと2倍の間に収まる設定値はそのまま尊重される（無修正）。
    small = [{"content": "あ" * 100}]
    untouched, _needed, applied = fit_dynamic_context(
        {"num_ctx": 3000, "num_predict": 1024}, small, ceiling=16384
    )
    assert untouched["num_ctx"] == 3000
    assert applied is False


def test_effective_context_resolution_event_recorded_once(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    state, actor = _state_and_actor()
    events: list[dict[str, object]] = []
    client = FakeContextClient({"modelfile_num_ctx": 8192, "ollama_ps_context": 4096})
    actions = OllamaActions(state, client, seed=1001, event_callback=events.append)
    resolved = actions._resolve_context_budget(actor, "discussion", time.perf_counter())
    assert resolved.limit == actor.num_ctx  # プレイヤー設定が単一値として採用される
    assert resolved.source == "player_setting"
    assert resolved.conflict is True  # env=4096 / ps=4096 / modelfile=8192 < 設定12288
    resolution_events = [e for e in events if e["type"] == "effective_context_resolution"]
    assert len(resolution_events) == 1
    assert resolution_events[0]["context_budget_candidates"]["process_env"] == 4096
    assert resolution_events[0]["configured_context_limit"] == actor.num_ctx
    assert resolution_events[0]["server_observed_context_limit"] == 4096
    # 2回目（同モデル）は再解決・再イベントなし（起動時の1回だけ）。
    actions._resolve_context_budget(actor, "discussion", time.perf_counter())
    assert len([e for e in events if e["type"] == "effective_context_resolution"]) == 1


def test_request_configuration_logs_budget_and_conflict(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    state, actor = _state_and_actor()
    state.conversation_log.append(Message(speaker="system", content="Day 1 begins."))
    state.conversation_log.append(Message(speaker="Player 3", content="Player 2に質問があります。"))
    events: list[dict[str, object]] = []
    client = FakeContextClient({"modelfile_num_ctx": 8192, "ollama_ps_context": None})
    actions = OllamaActions(state, client, seed=1001, event_callback=events.append)
    alive = tuple(state.alive_players)
    actions.discuss_body(actor.name, 1, alive, tuple(state.conversation_log))
    configs = [e for e in events if e["type"] == "llm_request_configuration"]
    assert configs, "リクエスト設定イベントが記録される"
    config = configs[0]
    assert config["effective_context_limit"] == actor.num_ctx
    assert config["context_budget_source"] == "player_setting"
    assert config["context_limit_conflict"] is True
    # 由来別名称と実効リクエスト値（TASK-046レビュー指摘）を個別に検証する。
    assert config["configured_context_limit"] == actor.num_ctx
    assert config["server_observed_context_limit"] == 4096
    assert config["effective_requested_num_ctx"] == config["requested_num_ctx"]
    breakdown = config["context_budget_breakdown"]
    assert isinstance(breakdown, dict)
    assert breakdown["input_tokens_estimate"] > 0
    assert breakdown["generation_limit_tokens"] > 0
    assert breakdown["safety_margin_tokens"] == 512
    assert "total_needed_tokens" in breakdown


def test_required_evidence_retention_is_full_under_tiny_budget() -> None:
    state, actor = _state_and_actor()
    names = [player.name for player in state.players]
    history = [Message(speaker="system", content="Day 1 begins.")]
    for index in range(24):
        history.append(
            Message(
                speaker=names[index % len(names)],
                content=(
                    f"議論 {index + 1}: Player {names[(index + 1) % len(names)]} の"
                    "投票先と占い結果を確認したい。"
                ),
            )
        )
    context = PromptBuilder.build_discussion_prompt(
        actor,
        tuple(state.alive_players),
        tuple(history),
        day=1,
        state=state,
        include_claim=True,
        context_strategy="hybrid_retrieval",
        evidence_char_budget=600,
        question_hint_mode="none",
    )
    diagnostics = context.diagnostics
    assert diagnostics["required_evidence_retention"] == 1.0, "必須証拠の欠落は許されない"
    assert diagnostics["required_events_kept"] == diagnostics["required_event_indices"]
    assert diagnostics["compression_stage"] == "evidence_budget_compressed"
    assert diagnostics["dropped_events_detail"], "圧縮で何を落としたか検証可能である"
    dropped = diagnostics["dropped_events_detail"][0]
    assert {"index", "day", "turn", "speaker", "reasons"} <= set(dropped)
    # 必須証拠は dropped_events_detail に現れない。
    assert all(
        entry["index"] not in diagnostics["required_event_indices"]
        for entry in diagnostics["dropped_events_detail"]
    )


def test_core_ledger_retry_keeps_unanswered_question_and_counters() -> None:
    state, actor = _state_and_actor()
    assert actor.name == "Player 2"
    state.conversation_log.append(Message(speaker="system", content="Day 1 begins."))
    state.conversation_log.append(
        Message(speaker="Player 3", content="Player 2の昨日の投票理由を教えてください。")
    )
    state.conversation_log.append(
        Message(speaker="Player 4", content="Player 2の主張は矛盾していると感じる。")
    )
    actions = OllamaActions(state, FakeContextClient({}), seed=1001)
    prompt = actions._short_discussion_retry_prompt(actor, None)
    assert "[Unanswered Question To You] Player 3:" in prompt
    assert "[Recent Counter-arguments (Unverified)]" in prompt
    assert "[Public Claims & Results]" in prompt
    assert "Vote History" in prompt


def test_secret_leak_retry_emits_compression_stage(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    state, actor = _state_and_actor()
    assert actor.name == "Player 2"
    state.conversation_log.append(Message(speaker="system", content="Day 1 begins."))
    state.conversation_log.append(Message(speaker="Player 2", content="私は村人です。"))
    events: list[dict[str, object]] = []
    actions = OllamaActions(
        state,
        FakeContextClient({}, responses=["[CO] None\n[BODY] 投票先を確認したい。"]),
        seed=1001,
        event_callback=events.append,
    )
    result = actions.retry_after_secret_leak(
        actor.name,
        1,
        tuple(state.alive_players),
        tuple(state.conversation_log),
        ("secret_role_leak",),
    )
    assert result is not None
    compression = [e for e in events if e["type"] == "context_compression_stage"]
    assert compression, "core-ledger圧縮の段階が記録される"
    assert compression[0]["compression_stage"] == "core_ledger"
    assert compression[0]["compression_reason"] == "secret_leak_safety_retry"
    assert "unanswered_question_to_actor" in compression[0]["protected_evidence"]


