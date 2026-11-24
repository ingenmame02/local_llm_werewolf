"""TASK-063の回帰テスト（実機LL不使用）: thinking-only・timeoutと論理状態の分離。

実機では Player 1 を `llm-jp-4.1-33b-thinking-fixed` に替えた試行で、17アクション中5件が
失敗し（thinking-only 4件・timeout 1件）、全体でも thinking予算超過7件・再要求20件となった。
「強化モデルが論理を維持できなかった」のか「出力予算・thinking・timeoutで発言機会を失った」
のかを切り分けないまま回答率だけで評価すると原因を取り違える。本テストは決定的に固定する
（LL不使用・モックのみ）:

1. 送信1件を単一カテゴリ（thinking_budget / timeout / generation_cap / transport /
   output_format / prompt_logic / ok）へ分類する（通信→timeout→生成量→形式→論理の優先順）。
2. `thinking-fixed`系モデルでは行動別の本文最低量とthinking上限を持つ。ただし実測で思考量が
   観測されている場合は静的上限より証拠を優先する（削りすぎない）。
3. 再要求で本文を失ったか（lost）／保持できたか（preserved）を行動単位で記録する。
4. thinking-only の再要求ではエンジン正規化の論理状態（[CONTINUITY_CHECKPOINT]）を先頭に置き、
   投票台帳の全行を再掲しない（同じ長いプロンプトを再送しない）。
5. CLIのモデル×アクション別集計にカテゴリ内訳・入力実字数・短縮再要求率が出る。
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.setup import create_game
from werewolf.thinking_profile import BODY_MIN_TOKENS_BY_ACTION
from werewolf.output_budget_trace import (
    CATEGORY_GENERATION_CAP,
    CATEGORY_OK,
    CATEGORY_OUTPUT_FORMAT,
    CATEGORY_PROMPT_LOGIC,
    CATEGORY_THINKING_BUDGET,
    CATEGORY_TIMEOUT,
    CATEGORY_TRANSPORT,
    FIXED_MODEL_BODY_MIN_TOKENS_BY_ACTION,
    FIXED_MODEL_THINKING_CAP_TOKENS_BY_ACTION,
    RETRY_BODY_LOST,
    RETRY_BODY_NOT_APPLICABLE,
    RETRY_BODY_PRESERVED,
    OutputBudgetTrace,
    classify_attempt,
    fixed_model_caps,
    is_thinking_fixed_model,
    issue_category,
)

FIXED_MODEL = "llm-jp-4.1-33b-thinking-fixed"
PLAIN_MODEL = "werewolf-gpt-oss-20b-16k"


def _state(seed: int = 2063063):
    return create_game(
        seed=seed, config_path="config/players.json", response_language="日本語"
    )


def _fixed_actor(state):
    return next(player for player in state.players if player.model == FIXED_MODEL)


class _ScriptedClient:
    """送信を実際に発行しないモック（LL不使用）。送信オプションとプロンプトを記録する。"""

    def __init__(self, responses: list[object] | None = None) -> None:
        self.timeout = 120.0
        self.max_retries = 1
        self.timeout_retries = 0
        self._responses = list(responses or [])
        self.sent_options: list[dict[str, object]] = []
        self.sent_prompts: list[list[dict[str, object]]] = []

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {}

    def chat(
        self, model: str, messages: list[dict[str, object]], options: dict | None = None
    ) -> str:
        self.sent_options.append(dict(options or {}))
        self.sent_prompts.append([dict(message) for message in messages])
        entry = self._responses.pop(0) if self._responses else "Pass"
        if isinstance(entry, str):
            self._last_chat_response = None
            return entry
        from werewolf.ollama import ChatResponse

        self._last_chat_response = entry
        return entry.content


# ---------------------------------------------------------------------------
# 1. 送信カテゴリの分離（thinking予算 / timeout / 生成上限 / 通信 / 形式 / 論理）
# ---------------------------------------------------------------------------
def test_classifies_thinking_only_separately_from_generation_cap() -> None:
    assert (
        classify_attempt(body_char_count=0, thinking_present=True, done_reason="length")
        == CATEGORY_THINKING_BUDGET
    )
    # 思考の気配がなく、生成上限まで使い切って空 → 上限そのものが足りない
    assert (
        classify_attempt(body_char_count=0, eval_count=512, num_predict=512, done_reason="length")
        == CATEGORY_GENERATION_CAP
    )
    assert classify_attempt(timed_out=True) == CATEGORY_TIMEOUT
    assert classify_attempt(transport_error=True) == CATEGORY_TRANSPORT
    # timeoutと通信失敗が同時に立っても timeout側（応答時間の問題として数える）
    assert classify_attempt(timed_out=True, transport_error=True) == CATEGORY_TIMEOUT
    assert (
        classify_attempt(body_char_count=120, format_issue="discussion_output_format")
        == CATEGORY_OUTPUT_FORMAT
    )
    assert (
        classify_attempt(body_char_count=120, logic_issue="repeated_discussion_topic")
        == CATEGORY_PROMPT_LOGIC
    )
    # 本文があり思考もある正常応答は ok（thinkingを使ったことを「失敗」と数えない）
    assert classify_attempt(body_char_count=120, thinking_present=True) == CATEGORY_OK


def test_response_issue_names_map_to_format_or_logic_sides() -> None:
    assert issue_category("seer_result_without_claim") == CATEGORY_OUTPUT_FORMAT
    assert issue_category("inconsistent_seer_result") == CATEGORY_OUTPUT_FORMAT
    assert issue_category("question_not_answered_first") == CATEGORY_PROMPT_LOGIC
    assert issue_category("past_action_question_unanswered") == CATEGORY_PROMPT_LOGIC
    # 生成起因（thinking-only・timeout）は形式でも論理でもない（別カテゴリで数える）
    assert issue_category("thinking_only_response") is None
    assert issue_category(None) is None


# ---------------------------------------------------------------------------
# 2. thinking-fixed系モデルの行動別予算（本文最低量とthinking上限）
# ---------------------------------------------------------------------------
def test_fixed_model_caps_are_per_action_and_skip_plain_models() -> None:
    assert is_thinking_fixed_model(FIXED_MODEL) is True
    assert is_thinking_fixed_model(FIXED_MODEL.replace("-", "_")) is True
    assert is_thinking_fixed_model(PLAIN_MODEL) is False
    assert fixed_model_caps(PLAIN_MODEL, "discussion") is None
    discussion = fixed_model_caps(FIXED_MODEL, "discussion")
    vote = fixed_model_caps(FIXED_MODEL, "vote")
    assert discussion is not None and vote is not None
    # 議論は本文を厚く、短い行動は抑える（思考に食われて本文が空になる事故への防備）
    assert discussion["body_min_tokens"] >= BODY_MIN_TOKENS_BY_ACTION["discussion"]
    assert (
        FIXED_MODEL_BODY_MIN_TOKENS_BY_ACTION["discussion"]
        > FIXED_MODEL_BODY_MIN_TOKENS_BY_ACTION["vote"]
    )
    assert (
        FIXED_MODEL_THINKING_CAP_TOKENS_BY_ACTION["discussion"]
        > FIXED_MODEL_THINKING_CAP_TOKENS_BY_ACTION["vote"]
    )


# ---------------------------------------------------------------------------
# 3. トレース（モデル×アクション別の内訳と、再要求で本文を失ったかの判定）
# ---------------------------------------------------------------------------
def test_trace_separates_budget_failures_and_measures_short_retry_input() -> None:
    trace = OutputBudgetTrace()
    trace.record_attempt(
        model=FIXED_MODEL,
        budget_key="discussion",
        action="discussion",
        request_number=1,
        thinking_only=True,
        thinking_present=True,
        done_reason="length",
        prompt_char_count=21046,
        num_ctx=16384,
        elapsed_seconds=95.0,
    )
    trace.record_attempt(
        model=FIXED_MODEL,
        budget_key="discussion",
        action="discussion",
        request_number=2,
        body_char_count=280,
        thinking_present=True,
        prompt_char_count=6120,
        short_retry=True,
        elapsed_seconds=18.0,
    )
    metrics = trace.metrics()["output_budget_by_model"][FIXED_MODEL]
    assert metrics["thinking_only_count"] == 1
    assert metrics["short_retry_count"] == 1
    # 短縮再要求（入力21,046字 → 6,120字）を字数で実測できる
    assert metrics["max_prompt_char_count"] == 21046
    assert metrics["avg_prompt_char_count"] < 21046
    assert metrics["body_generation_rate"] == 0.5
    assert metrics["thinking_fixed_model"] is True
    outcome = trace.outcomes[(FIXED_MODEL, "discussion", "discussion")]
    # 初回が thinking-only で本文を失い、再要求で出した → lost
    assert outcome.body_preservation == RETRY_BODY_LOST
    assert outcome.attempts == 2
    public = trace.serialize_public()
    assert set(public) == {"metrics", "outcomes"}
    entry = public["outcomes"][0]
    assert entry["failure_categories"] == [CATEGORY_THINKING_BUDGET]
    assert entry["category_counts"][CATEGORY_OK] == 1


def test_body_preserved_on_semantic_retry_and_logic_counted_separately() -> None:
    trace = OutputBudgetTrace()
    trace.record_attempt(
        model=PLAIN_MODEL,
        budget_key="discussion",
        action="discussion",
        request_number=1,
        body_char_count=210,
        prompt_char_count=9000,
    )
    # 再要求が無ければ「喪失の判定は該当なし」（失敗していない）
    assert (
        trace.outcomes[(PLAIN_MODEL, "discussion", "discussion")].body_preservation
        == RETRY_BODY_NOT_APPLICABLE
    )
    # 本文は出ているが形式・論理の問題という関連付け（送信行は増やさない）
    row = trace.annotate_last_attempt(
        model=PLAIN_MODEL,
        budget_key="discussion",
        action="discussion",
        logic_issue="repeated_discussion_topic",
    )
    assert row is not None and row.category == CATEGORY_PROMPT_LOGIC
    assert len(trace.rows) == 1
    metrics = trace.metrics()["output_budget_by_model"][PLAIN_MODEL]
    assert metrics["prompt_logic_count"] == 1
    assert metrics["budget_bound_failures"] == 0
    assert metrics["logic_bound_failures"] == 1
    # 同じ本文を持続させたまま書き直した再要求なら「保持」と数える
    trace.record_attempt(
        model=PLAIN_MODEL,
        budget_key="discussion",
        action="discussion",
        request_number=2,
        body_char_count=260,
        prompt_char_count=9200,
        short_retry=True,
    )
    outcome = trace.outcomes[(PLAIN_MODEL, "discussion", "discussion")]
    assert outcome.body_preservation == RETRY_BODY_PRESERVED
    assert outcome.attempts == 2


def test_empty_response_is_never_relabeled_as_format_or_logic() -> None:
    """本文が出ていない送信を「議論の質」へ読み替えない（不足原因は予算側に残す）。"""
    trace = OutputBudgetTrace()
    trace.record_attempt(
        model=FIXED_MODEL,
        budget_key="vote",
        action="vote",
        request_number=1,
        thinking_only=True,
        body_char_count=0,
    )
    assert (
        trace.annotate_last_attempt(
            model=FIXED_MODEL,
            budget_key="vote",
            action="vote",
            format_issue="marker_missing",
        )
        is None
    )
    assert trace.rows[-1].category == CATEGORY_THINKING_BUDGET


# ---------------------------------------------------------------------------
# 4. llm_actions 連携: thinking-fixedの予約と、checkpoint優先の短縮再要求
# ---------------------------------------------------------------------------
def _actions(client, state):
    from werewolf.llm_actions import OllamaActions

    return OllamaActions(state, client, seed=state.seed)


def test_fixed_model_reservation_reserves_body_before_thinking() -> None:
    """thinking-fixedでは本文最低量が上がり、その分だけ生成上限キャップも引き上がる。"""
    from werewolf.thinking_profile import BODY_RESERVATION_MAX_NUM_PREDICT

    state = _state()
    actor = _fixed_actor(state)
    actions = _actions(_ScriptedClient([]), state)
    reservation = actions._body_reservation(
        actor, "discussion", 512, "player_config", True
    )
    caps = fixed_model_caps(FIXED_MODEL, "discussion")
    assert caps is not None
    assert reservation.body_min_tokens >= caps["body_min_tokens"]
    # キャップは本文+thinking上限を下回らない（本文が思考に食われる設計にしない）
    assert int(reservation.cap_tokens or 0) >= (
        caps["body_min_tokens"] + caps["thinking_cap_tokens"]
    )
    assert reservation.applied is True
    # 固定でないモデルでは従来どおりの本文最低量（変更しない）
    other = next(player for player in state.players if player.model != FIXED_MODEL)
    plain = actions._body_reservation(other, "discussion", 512, "player_config", False)
    assert plain.body_min_tokens == BODY_MIN_TOKENS_BY_ACTION["discussion"]
    assert plain.cap_tokens == BODY_RESERVATION_MAX_NUM_PREDICT["discussion"]


def test_fixed_model_static_cap_never_undercuts_observed_thinking() -> None:
    """実測で思考消費が観測済みなら、静的な天井より証拠を優先する（削りすぎない）。"""
    from werewolf.thinking_profile import ModelActionProfile, thinking_allowance

    state = _state()
    actor = _fixed_actor(state)
    actions = _actions(_ScriptedClient([]), state)
    profile = ModelActionProfile(model=actor.model, budget_key="vote")
    profile.thinking_floor_tokens = 900
    profile.thinking_present_count = 2
    profile.requests = 2
    actions._thinking_profiles.get_or_create(actor.model, "vote").thinking_floor_tokens = 900
    actions._thinking_profiles.get_or_create(actor.model, "vote").thinking_present_count = 2
    margin = actions._thinking_margin_for(actor, "vote", True)
    generic = thinking_allowance(profile, thinking_enabled=True)
    assert margin == generic  # 観測あり → 静的上限（640）は適用しない
    # 観測の無い行動では静的上限が効く（既定1024のガンを行動別の値へ絞る）
    empty_margin = actions._thinking_margin_for(actor, "attack", True)
    assert empty_margin <= FIXED_MODEL_THINKING_CAP_TOKENS_BY_ACTION["attack"]


def test_thinking_only_retry_leads_with_checkpoint_and_skips_vote_ledger() -> None:
    """再要求で同じ長いプロンプトを再送せず、正規化状態を先頭に置き投票台帳を省く。"""
    state = _state()
    actor = _fixed_actor(state)
    names = tuple(player.name for player in state.players)
    state.day = 2
    state.discussion_turn = 2
    state.vote_history.append((1, False, {names[0]: names[3], names[1]: names[3]}))
    record = state.issue_continuity_checkpoint("day-2-pre_vote")
    assert record is not None
    actions = _actions(_ScriptedClient([]), state)
    prompt = actions._short_discussion_retry_prompt(
        actor, "thinking_only_response", tuple(state.conversation_log)
    )
    head = prompt[: len(prompt) // 3]
    assert "[CONTINUITY_CHECKPOINT" in head  # 先頭側（現在質問へ答える前に状態を読む）
    assert "- Vote History:" not in prompt  # 状態が引き継ぐので全行は再掲しない
    assert "[Unanswered Question To You]" in prompt or "Correction Guidance" in prompt
    # 口調・論点・結論を縛る命令文を増やさない（状態は情報、形式指示のみ）
    assert "use only the following facts and recent speeches" in prompt


# ---------------------------------------------------------------------------
# 5. CLI集計: カテゴリ内訳・入力実字数・短縮再要求率
# ---------------------------------------------------------------------------
def test_cli_action_profiles_split_output_budget_categories(tmp_path) -> None:
    from werewolf.cli import GameLogger

    model = FIXED_MODEL
    logger = GameLogger("task063", tmp_path, {"seed": 1}, private_enabled=True)

    def attempt(**fields: object) -> None:
        record: dict[str, object] = {
            "type": "llm_generation_attempt",
            "model": model,
            "action": "discussion",
            "action_budget_key": "discussion",
            **fields,
        }
        logger.append_record(record, record)

    attempt(
        request_number=1,
        body_empty=True,
        thinking_only_response=True,
        thinking_present=True,
        body_char_count=0,
        output_budget={
            "category": CATEGORY_THINKING_BUDGET,
            "budget_bound": True,
            "logic_bound": False,
            "prompt_char_count": 21046,
            "body_min_tokens": 832,
            "thinking_cap_tokens": 2048,
        },
    )
    attempt(
        request_number=2,
        body_empty=False,
        body_char_count=260,
        output_budget={
            "category": CATEGORY_OK,
            "budget_bound": False,
            "logic_bound": False,
            "short_retry": True,
            "prompt_char_count": 5980,
        },
    )
    state = _state()
    logger.finalize(state)

    entry = logger._model_stats[model]["action_profiles"]["discussion"]
    assert entry["output_budget_categories"][CATEGORY_THINKING_BUDGET] == 1
    assert entry["output_budget_categories"][CATEGORY_OK] == 1
    assert entry["short_retry_requests"] == 1
    assert entry["max_prompt_char_count"] == 21046
    # 短縮再要求で入力が減ったことを平均で見られる（21,046字と 5,980字の平均）
    assert 5980 < entry["avg_prompt_char_count"] < 21046
    assert entry["body_generation_rate"] == 0.5
    assert entry["budget_bound_failure_rate"] == 0.5
    assert entry["logic_bound_failure_rate"] == 0.0
