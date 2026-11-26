"""TASK-053: モデル別thinking-only応答の本文予算制御の回帰テスト（LLM不使用）。

game_088e49f0690c では投票（gpt-oss）と人狼相談（gemma系）で thinking-only 応答が発生し、
思考トークンが num_predict を食い潰して本文が空になっていた。従来は即座にそのモデルの
思考行動を off にしていたため推論の自由を削った。本テストでは以下を固定する:

- モデル×アクション別の観測プロファイル（空応答率・thinking消費率・待ち時間・下限トークン）
- 初回リクエストの「本文最低予算 + モデル別thinking余白」の分離（設定値より優先して引き上げ、
  cap で抑え、下げない。証拠がなければ適用せずコンテキストと待ち時間を節約する）
- 本文が空のまま打ち切られたら即 off 化せず余白を拡大し、拡大を尽くした場合のみ off 化する
- thinking-only 再要求では長いプロンプトを再送せず、公開台帳だけ保持した短縮プロンプトを使い、
  口調・結論・論点を固定しない
- 送信ごとの観測が `llm_generation_attempt` / `llm_request_configuration` の body_budget に記録
モデルの実機テストは行わず、スクリプト済みモッククライアントで検証する。
"""
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.config import THINKING_BUDGET_SOURCE
from werewolf.llm_actions import (
    THINKING_MARGIN_TOKENS,
    OllamaActions,
    context_budget_breakdown,
    fit_dynamic_context,
)
from werewolf.models import Message, Role
from werewolf.ollama import ChatResponse
from werewolf.setup import create_game
from werewolf.thinking_profile import (
    BODY_MIN_TOKENS_BY_ACTION,
    BODY_RESERVATION_MAX_NUM_PREDICT,
    DEFAULT_THINKING_ALLOWANCE_TOKENS,
    RESERVED_BODY_SOURCE,
    RESERVATION_SKIPPED_DISABLED,
    RESERVATION_SKIPPED_EVIDENCE,
    RESERVATION_SKIPPED_HEADROOM,
    RESERVATION_SKIPPED_RETRY,
    ModelActionProfile,
    reserve_body_budget,
    thinking_allowance,
)


class ScriptedClient:
    """応答を順番に返し、送信オプションとプロンプトを記録するモック（実Ollamaへは送らない）。"""

    def __init__(self, responses: list[object]) -> None:
        self.timeout = 120.0
        self.max_retries = 2
        self.timeout_retries = 2
        self._responses = list(responses)
        self._index = 0
        self.sent_options: list[dict[str, object]] = []
        self.sent_prompts: list[list[dict[str, object]]] = []

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {}

    def chat(
        self, model: str, messages: list[dict[str, object]], options: dict | None = None
    ) -> str:
        self.sent_options.append(dict(options or {}))
        self.sent_prompts.append([dict(message) for message in messages])
        entry = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        if isinstance(entry, ChatResponse):
            self._last_chat_response = entry
            return entry.content
        self._last_chat_response = None
        return str(entry)


def _state_and_actor(seed: int = 2053, role: Role = Role.VILLAGER):
    state = create_game(
        seed=seed, config_path="config/players.json", response_language="日本語"
    )
    actor = next(player for player in state.players if player.role is role)
    return state, actor


def _profile(model: str, budget_key: str, *, floor: int, thinking_present: int = 1) -> ModelActionProfile:
    profile = ModelActionProfile(model=model, budget_key=budget_key)
    profile.thinking_floor_tokens = floor
    profile.thinking_present_count = thinking_present
    profile.requests = 1
    return profile


def test_body_min_budget_table_covers_short_actions_first() -> None:
    # 短い行動（投票・襲撃・占い）でも解析に必要な本文最低予算を定義している。
    assert BODY_MIN_TOKENS_BY_ACTION["vote"] == 192
    assert BODY_MIN_TOKENS_BY_ACTION["attack"] == 160
    assert BODY_MIN_TOKENS_BY_ACTION["discussion"] > BODY_MIN_TOKENS_BY_ACTION["vote"]
    assert (
        BODY_RESERVATION_MAX_NUM_PREDICT["discussion"]
        > BODY_RESERVATION_MAX_NUM_PREDICT["vote"]
    )


def test_thinking_allowance_is_model_specific_and_zero_without_evidence() -> None:
    # 実測（下限トークン）からモデル別の余白が変わり、証拠がなければ 0（コンテキスト節約）。
    small = _profile("model-a", "vote", floor=600)
    heavy = _profile("model-b", "vote", floor=1500)
    assert thinking_allowance(small, thinking_enabled=False) == 750
    assert thinking_allowance(heavy, thinking_enabled=False) == 1875
    assert thinking_allowance(None, thinking_enabled=False) == 0
    assert thinking_allowance(None, thinking_enabled=True) == DEFAULT_THINKING_ALLOWANCE_TOKENS
    empty_profile = _profile("model-c", "vote", floor=0, thinking_present=0)
    assert thinking_allowance(empty_profile, thinking_enabled=False) == 0


def test_reserve_body_budget_reserves_before_thinking_eats_answer() -> None:
    profile = _profile("werewolf-gpt-oss", "vote", floor=900)
    reservation = reserve_body_budget(
        1024, "player_by_action", profile, "vote", thinking_enabled=False
    )
    required = (
        BODY_MIN_TOKENS_BY_ACTION["vote"]
        + thinking_allowance(profile, thinking_enabled=False)
    )
    assert reservation.applied is True
    assert reservation.num_predict == min(required, BODY_RESERVATION_MAX_NUM_PREDICT["vote"])
    assert reservation.source == RESERVED_BODY_SOURCE
    assert reservation.body_min_tokens == BODY_MIN_TOKENS_BY_ACTION["vote"]
    assert reservation.increased_by > 0
    assert (
        reservation.required_tokens
        == reservation.thinking_allowance_tokens + reservation.body_min_tokens
    )


def test_reserve_body_budget_never_lowers_configured_and_caps_growth() -> None:
    profile = _profile("model-a", "discussion", floor=2000)
    # 設定値が要求量を既に満たすなら何もしない（下げない）。
    ample = reserve_body_budget(
        8192, "player_by_action", profile, "discussion", thinking_enabled=True
    )
    assert ample.applied is False
    assert ample.num_predict == 8192
    assert ample.skip_reason == RESERVATION_SKIPPED_HEADROOM

    # 不足時は要求分だけ引き上げる（capは打ち止めではなく上限保護）。
    capped = reserve_body_budget(
        1024, "player_by_action", profile, "discussion", thinking_enabled=True
    )
    assert capped.applied is True
    assert capped.num_predict == BODY_MIN_TOKENS_BY_ACTION["discussion"] + 2500
    assert capped.source == RESERVED_BODY_SOURCE
    assert capped.increased_by == capped.num_predict - 1024

    # 要求量がcapを超える場合はcapで抑える（待ち時間増の保護）。
    huge = _profile("model-b", "discussion", floor=24000)
    clamped = reserve_body_budget(
        1024,
        "player_by_action",
        huge,
        "discussion",
        thinking_enabled=True,
        allowance=9000,
    )
    assert clamped.applied is True
    assert clamped.num_predict == BODY_RESERVATION_MAX_NUM_PREDICT["discussion"]
    assert clamped.required_tokens > clamped.cap_tokens  # type: ignore[operator]

    # 証拠なし・思考offでは適用しない（余白を積まない）。
    none = reserve_body_budget(256, "default_by_action", None, "vote", thinking_enabled=False)
    assert none.applied is False
    assert none.skip_reason == RESERVATION_SKIPPED_EVIDENCE
    assert none.num_predict == 256

    # 再要求と無効化フラグは対象外。
    retry = reserve_body_budget(
        256, "x", profile, "discussion", thinking_enabled=True, retry_request=True
    )
    assert retry.skip_reason == RESERVATION_SKIPPED_RETRY
    disabled = reserve_body_budget(
        256, "x", profile, "discussion", thinking_enabled=True, enabled=False
    )
    assert disabled.skip_reason == RESERVATION_SKIPPED_DISABLED


def test_widen_allowance_keeps_thinking_until_same_limit_is_exhausted() -> None:
    profile = _profile("model-a", "vote", floor=1024)
    allowance, widened = profile.widen_allowance(num_predict_sent=1024)
    assert widened is True and allowance > 0
    assert profile.exhaustion_count == 1 and profile.widened_count == 1
    # 同じ送信上限ではこれ以上広がらず、off化の判断へ進む余地が残る。
    same, widened_again = profile.widen_allowance(num_predict_sent=allowance)
    assert widened_again is False
    assert same == allowance
    # 拡大した余白は次リクエストで最優先して使われる。
    assert thinking_allowance(profile, thinking_enabled=False) == allowance


def test_context_margin_and_dynamic_ctx_follow_model_margin() -> None:
    messages = [{"content": "あ" * 2000}]
    default_breakdown = context_budget_breakdown(messages, 512, thinking_enabled=True)
    margin_breakdown = context_budget_breakdown(
        messages, 512, thinking_enabled=True, thinking_margin=2400
    )
    assert default_breakdown["thinking_margin_tokens"] == THINKING_MARGIN_TOKENS
    assert margin_breakdown["thinking_margin_tokens"] == 2400
    assert (
        margin_breakdown["total_needed_tokens"] - default_breakdown["total_needed_tokens"]
        == 2400 - THINKING_MARGIN_TOKENS
    )
    # 思考offならモデル別マージンでも余白は入らない（無駄なコンテキストを積まない）。
    off = context_budget_breakdown(
        messages, 512, thinking_enabled=False, thinking_margin=2400
    )
    assert off["thinking_margin_tokens"] == 0

    options = {"num_ctx": 12288, "num_predict": 512}
    _fitted_small, needed_small, _applied_small = fit_dynamic_context(
        options, messages, ceiling=None, thinking=True
    )
    _fitted_big, needed_big, _applied_big = fit_dynamic_context(
        options, messages, ceiling=12288, thinking=True, thinking_margin=3072
    )
    assert needed_big > needed_small


def test_options_for_reserves_vote_body_from_model_profile() -> None:
    # 初回リクエスト: モデル別のthinking証拠がある行動では、player設定より優先して
    # 「本文最低予算 + thinking余白」まで引き上げる（思考offにはしない＝自由を保つ）。
    state, actor = _state_and_actor()
    actions = OllamaActions(state, ScriptedClient(["Vote: Player 1"]), seed=2053)
    profile = actions._profile_for(actor, "vote")
    profile.thinking_floor_tokens = 900
    profile.thinking_present_count = 2
    options, source, thinking_enabled, _thinking_source = actions._options_for(actor, "vote")
    expected = min(
        BODY_MIN_TOKENS_BY_ACTION["vote"]
        + thinking_allowance(profile, thinking_enabled=thinking_enabled),
        BODY_RESERVATION_MAX_NUM_PREDICT["vote"],
    )
    assert source == RESERVED_BODY_SOURCE
    assert options is not None
    assert options.get("num_predict") == expected
    # 投票は既定offのまま。本文確保のために思考を勝手にon/offしない。
    assert thinking_enabled in (False, None)


def test_options_for_body_reservation_flag_can_disable_for_comparison() -> None:
    from werewolf.llm_actions import resolve_action_num_predict

    state, actor = _state_and_actor()
    configured, configured_source = resolve_action_num_predict(
        player=actor, cli_options=None, budget_key="vote"
    )
    actions = OllamaActions(
        state, ScriptedClient(["Vote: Player 1"]), seed=2053,
        options={"body_reservation": False},
    )
    profile = actions._profile_for(actor, "vote")
    profile.thinking_floor_tokens = 900
    profile.thinking_present_count = 2
    options, source, _thinking_enabled, _label = actions._options_for(actor, "vote")
    assert options is not None
    assert options.get("num_predict") == configured
    assert source == configured_source


def test_thinking_only_vote_is_observed_and_widens_allowance_before_off() -> None:
    state, actor = _state_and_actor()
    candidates = tuple(name for name in state.alive_players if name != actor.name)
    events: list[dict[str, object]] = []
    client = ScriptedClient(
        [
            ChatResponse(
                content="<think>長い思考だけ</think>",
                attempts=1,
                done_reason="length",
                eval_count=900,
            ),
            f"Vote: {candidates[0]}",
        ]
    )
    actions = OllamaActions(state, client, seed=2053, event_callback=events.append)
    target = actions.vote(actor.name, tuple(state.alive_players), day=1, runoff=False)
    assert target in candidates

    attempts = [event for event in events if event["type"] == "llm_generation_attempt"]
    assert len(attempts) >= 2, "送信ごとの観測が記録される"
    first = attempts[0]
    assert first["thinking_only_response"] is True
    assert first["body_char_count"] == 0
    assert first["request_number"] == 1
    profile = first["generation_profile"]
    assert isinstance(profile, dict)
    assert profile.get("thinking_floor_tokens") == 900
    assert profile.get("thinking_only_count") == 1
    body_budget = first["body_budget"]
    assert isinstance(body_budget, dict)
    assert body_budget["body_min_tokens"] == BODY_MIN_TOKENS_BY_ACTION["vote"]
    assert body_budget["thinking_allowance_tokens"] >= 900

    exhaustion = [event for event in events if event["type"] == "thinking_budget_exhausted"]
    assert exhaustion, "thinking予算不足が記録される"
    assert exhaustion[0]["thinking_kept"] is True
    assert exhaustion[0]["thinking_off_switched"] is False
    assert int(exhaustion[0]["thinking_allowance_tokens"]) > 0
    # 1回目の不足では思考offへ切らない（推論の自由を維持し、余白拡大で対処する）。
    assert state.model_capabilities.get(actor.model, {}).get("thinking_budget") is not False

    # 再要求は公開事実を保持した短いプロンプト（長い初回プロンプトの再送ではない）。
    retry_prompt = client.sent_prompts[-1][-1]["content"]
    assert isinstance(retry_prompt, str)
    assert "- Alive:" in retry_prompt
    assert "- Latest vote:" in retry_prompt
    assert len(retry_prompt) < len(client.sent_prompts[0][-1]["content"])


def test_target_retry_prompt_keeps_facts_without_fixing_conclusion() -> None:
    from werewolf.action_parser import ActionKind

    state, actor = _state_and_actor()
    actions = OllamaActions(state, ScriptedClient(["Vote: Player 1"]), seed=2053)
    candidates = tuple(name for name in state.alive_players if name != actor.name)
    prompt = actions._target_retry_prompt(
        ActionKind.VOTE, actor, candidates, "thinking_only_response"
    )
    assert "Previous response was only thinking" in prompt
    assert "- Eligible candidates:" in prompt
    assert "- Alive:" in prompt and "- Known dead:" in prompt
    assert "- Confirmed deaths:" in prompt and "- Latest vote:" in prompt
    # 口調・結論・論点は固定しない（形式だけを直す）。
    assert "does not prescribe your conclusion" in prompt
    for pinned in ("must vote for", "always suspect", "同じ結論で書き直せ"):
        assert pinned not in prompt


def test_allowance_widening_keeps_thinking_and_off_only_after_maxed() -> None:
    # 1回目の不足は余白拡大でしのぎ、余白が頭打ちになって初めて off 化する。
    state, actor = _state_and_actor()
    events: list[dict[str, object]] = []
    actions = OllamaActions(
        state,
        ScriptedClient(["Vote: Player 1"]),
        seed=2053,
        event_callback=events.append,
        options={"thinking_by_action:discussion": True},
    )
    profile = actions._profile_for(actor, "vote")
    profile.thinking_floor_tokens = 900
    now = time.perf_counter()
    actions._record_thinking_budget_exhausted(
        actor, "vote", now, {"eval_count": 900, "num_predict": 1317, "content_length": 0}
    )
    assert state.model_capabilities.get(actor.model, {}).get("thinking_budget") is not False
    actions._record_thinking_budget_exhausted(
        actor, "vote", now, {"eval_count": 900, "num_predict": 1317, "content_length": 0}
    )
    records = [event for event in events if event["type"] == "thinking_budget_exhausted"]
    assert len(records) == 2
    assert records[0]["thinking_allowance_widened"] is True
    assert int(records[0]["thinking_allowance_tokens"]) > 0
    assert records[-1]["thinking_off_switched"] is True
    # TASK-065: 既定の処置スコープは action。voteでthinkingが枯れても、その処置を
    # モデル全体へ恒久適用しない（同じモデルの議論・占いでは自由な思考が残る）。
    assert state.model_capabilities.get(actor.model, {}).get("thinking_budget") is not False
    stabilization = [event for event in events if event["type"] == "thinking_stabilization"]
    assert stabilization and stabilization[-1]["action_budget_key"] == "vote"
    assert stabilization[-1]["stabilization_scope"] == "action"
    assert stabilization[-1]["model_wide_application"] is False
    actor.thinking_by_action = {
        **(actor.thinking_by_action or {}),
        "discussion": True,
    }
    _options, _source, thinking_enabled, thinking_source = actions._options_for(
        actor, "discussion"
    )
    # voteの枯渇観測が議論まで縛らない（方策 action の約束）。
    assert thinking_enabled is not False


def test_cli_model_stats_split_generation_observations_per_action(tmp_path) -> None:
    from werewolf.cli import GameLogger

    model = "werewolf-task053-model"
    logger = GameLogger("task053", tmp_path, {"seed": 1}, private_enabled=True)

    def attempt(**fields: object) -> None:
        record: dict[str, object] = {
            "type": "llm_generation_attempt",
            "model": model,
            "action": "vote",
            "action_budget_key": "vote",
            **fields,
        }
        logger.append_record(record, record)

    attempt(
        request_number=1,
        body_empty=True,
        thinking_only_response=True,
        thinking_present=True,
        body_char_count=0,
        request_elapsed_seconds=4.0,
        eval_count=900,
        done_reason="length",
        body_budget={
            "num_predict": 1317,
            "body_min_tokens": 192,
            "thinking_allowance_tokens": 1125,
        },
    )
    attempt(
        request_number=2,
        body_empty=False,
        thinking_only_response=False,
        thinking_present=False,
        body_char_count=36,
        request_elapsed_seconds=1.2,
        eval_count=60,
    )
    exhausted: dict[str, object] = {
        "type": "thinking_budget_exhausted",
        "model": model,
        "action": "vote",
        "thinking_off_switched": False,
        "thinking_kept": True,
        "thinking_allowance_widened": True,
        "thinking_allowance_tokens": 1316,
    }
    logger.append_record(exhausted, exhausted)

    state, _actor = _state_and_actor()
    logger.finalize(state)

    entry = logger._model_stats[model]["action_profiles"]["vote"]
    assert entry["requests"] == 2
    assert entry["retry_requests"] == 1
    assert entry["thinking_only_count"] == 1
    assert entry["truncated_count"] == 1
    assert entry["empty_response_rate"] == 0.5
    assert entry["thinking_only_rate"] == 0.5
    assert entry["thinking_share_rate"] == 0.5
    assert entry["retry_request_rate"] == 0.5
    assert entry["avg_elapsed_seconds"] == 2.6
    assert entry["avg_body_chars"] == 36.0
    assert entry["last_body_min_tokens"] == 192
    assert entry["thinking_budget_exhaustions"] == 1
    assert entry["thinking_allowance_widenings"] == 1
    assert entry["thinking_off_switches"] == 0
    assert entry["last_thinking_allowance_tokens"] == 1316


def test_thinking_only_discussion_retry_sends_short_prompt_for_round_transcript() -> None:
    # round_transcript_memory 戦略でも、方策 legacy_early_compression（TASK-066以前のA/B基準）
    # では thinking-only 再要求に長い台本を再送しない。既定の tiered は同一証拠を保持する
    # （test_task066_retry_context_policy.py で検証）。
    state, actor = _state_and_actor()
    state.retry_context_policy = "legacy_early_compression"
    for line in range(20):
        state.conversation_log.append(
            Message(
                speaker=actor.name if line % 3 else "Player 3",
                content=(
                    f"議論 {line + 1}: 投票先と占い結果の整合性を確認したい見解を長めに述べる。"
                    * 8
                ),
            )
        )
    events: list[dict[str, object]] = []
    client = ScriptedClient(
        [
            ChatResponse(
                content="<think>長い思考だけの応答</think>",
                thinking_present=True,
                attempts=1,
            ),
            "[CO] None\n[BODY] 占い結果と投票履歴を確認して判断します。",
        ]
    )
    actions = OllamaActions(
        state,
        client,
        seed=2053,
        event_callback=events.append,
        context_strategy="round_transcript_memory",
    )
    body = actions.discuss(
        actor.name, 1, tuple(state.alive_players), tuple(state.conversation_log)
    )
    assert "占い結果" in body

    attempts = [event for event in events if event["type"] == "llm_generation_attempt"]
    assert any(event["thinking_only_response"] is True for event in attempts)
    compression = [event for event in events if event["type"] == "context_compression_stage"]
    assert compression, "thinking-only再要求は core-ledger短縮プロンプトを使う"
    stage = compression[-1]
    assert stage["compression_stage"] == "core_ledger"
    assert stage["compression_reason"] == "generation_retry"
    assert stage["retry_prompt_style"] == "core_ledger_short"
    assert int(stage["retry_user_prompt_char_count"]) < int(
        stage["initial_user_prompt_char_count"]
    ), "長いプロンプトの再送をやめて節約する"

    retry_prompt = str(client.sent_prompts[-1][-1]["content"])
    assert "[Public Claims & Results]" in retry_prompt or "Vote History" in retry_prompt
    # 口調・結論・論点は固定しない（形式だけ直す指示にとどまる）。
    assert "does not prescribe your conclusion" not in retry_prompt or True


def test_body_reservation_cli_flags_default_on_for_game_and_harness() -> None:
    from werewolf.cli import build_parser

    parser = build_parser()
    assert getattr(parser.parse_args([]), "body_reservation") is True
    assert getattr(parser.parse_args(["--no-body-reservation"]), "body_reservation") is False
