"""TASK-065の回帰テスト（実機LL不使用）: 実効コンテキスト整合・thinking安定化・意図分離・名前正規化。

実機ログ `game_425755ce9a83` で確認された4つの運用課題を決定的に固定する:
1. Player 1 の設定 `num_ctx=16384` に対し Modelfile / プロセスenv は 4096 だった。採用値を
   送り続けて通信失敗・約360秒timeout・思考枯渇が起きた後で初めて判明する状態をやめる
   （`configured / server_observed / loaded / requested` を分離記録し、cap のときだけ縮退）。
2. Player 8 の議論 thinking-only、Player 1 の投票 thinking-only など失敗は行動別だったのに、
   timeout / thinking 処置をモデル単位の恒久フラグへ保存していた（投票で詰まっただけで
   議論の思考まで落とさない。成功が続けば解除する）。
3. Day2 T1 Player 1 の「次に、プレイヤー6に…問いただすことを提案します。」はこれから質問する
   行為なのに `speech_from_unspoken_player` へ分類された。実質的な過去帰属だけ findings に
   数え、質問行為とこれからの行動推測は別カウンタへ移す（本文は変更しない）。
4. `Player\u202f6`（NARROW NBSP）・`player6`・全角表記が `unknown_player_reference` の ledger
   repair を誘発した。内部照合の入口だけで実在名へ解決し、公開本文の表記は一切変えない。

LLを呼ばない純粋な判定テストで、既定（warn / separated / normalized / action）とA/B用の
旧挙動（cap相当の縮退・literal）を分けて確認する。議論本文・CO・ブラフ・疑問・仮定・
立場変更の自由を狭める挙動がないことも見る。
"""
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.context_integrity import (
    CONTEXT_INTEGRITY_POLICY_CAP,
    CONTEXT_INTEGRITY_POLICY_DEFAULT,
    CONTEXT_INTEGRITY_POLICY_WARN,
    KIND_ALIGNED,
    KIND_CONFIGURED_ABOVE_SERVER,
    KIND_LOADED_UNAVAILABLE,
    assess_context_integrity,
    context_integrity_policy_label,
    normalize_context_integrity_policy,
)
from werewolf.day_timeline import (
    FUTURE_SPEECH_FROM_UNSPOKEN,
    INTENT_FUTURE_ACTION_SPECULATION,
    INTENT_FUTURE_QUESTION_TO_UNSPOKEN,
    INTENT_PAST_ATTRIBUTION,
    SUBTYPE_ATTRIBUTION_NEVER_SPOKEN,
    TIMELINE_INTENT_POLICY_SEPARATED,
    build_turn_snapshot,
    inspect_public_body,
    normalize_timeline_intent_policy,
    timeline_intent_policy_label,
)
from werewolf.name_resolution import (
    NAME_RESOLUTION_POLICY_LITERAL,
    NAME_RESOLUTION_POLICY_NORMALIZED,
    build_name_lookup,
    has_unresolved_reference,
    mention_labels,
    name_resolution_policy_label,
    names_in_text,
    normalize_name_resolution_policy,
    resolve_known_names,
    shares_mentioned_players,
    unresolved_references,
    variant_mention_count,
)
from werewolf.models import Message, Phase
from werewolf.output_stability import (
    CATEGORY_GENERATION_CAP,
    CATEGORY_OK,
    CATEGORY_OUTPUT_FORMAT,
    CATEGORY_PROMPT_LOGIC,
    CATEGORY_THINKING_BUDGET,
    CATEGORY_TIMEOUT,
    DOWNGRADE_THINKING_OFF,
    DOWNGRADE_TIMEOUT_FIT,
    OUTPUT_STABILIZATION_SCOPE_ACTION,
    OUTPUT_STABILIZATION_SCOPE_MODEL,
    OutputStabilityTracker,
    normalize_output_stabilization_scope,
    output_stabilization_scope_label,
)
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game

NARROW_NBSP = "\u202f"
PLAYER_NAMES = tuple(f"Player {index}" for index in range(1, 9))
MODEL_NAME = "llm-jp-4.1-33b-thinking-fixed"


def _state(seed: int = 2065065):
    return create_game(seed=seed, config_path="config/players.json")


def _start_day(state) -> None:
    state.public_timeline.record_day_start(state.day)
    state.conversation_log.append(Message("system", f"Day {state.day} begins."))
    state.phase = Phase.DAY_DISCUSSION
    state.discussion_turn = 1
    state.begin_discussion_turn()

# ---------------------------------------------------------------------------
# 1. プレイヤー名の内部解決（公開本文は変更しない・入口だけ）
# ---------------------------------------------------------------------------
def test_normalized_policy_absorbs_whitespace_kana_and_fullwidth_variants() -> None:
    narrow = f"次に、Player{NARROW_NBSP}6へ投票理由を尋ねます。"
    assert resolve_known_names(narrow, PLAYER_NAMES) == ("Player 6",)
    assert not has_unresolved_reference(narrow, PLAYER_NAMES)
    assert variant_mention_count(narrow, PLAYER_NAMES) == 1
    katakana = "プレイヤー7の投票先を確認したい。"
    assert resolve_known_names(katakana, PLAYER_NAMES) == ("Player 7",)
    assert mention_labels(katakana, PLAYER_NAMES).get("katakana") == 1
    fullwidth = "Ｐｌａｙｅｒ５に質問します。"
    assert resolve_known_names(fullwidth, PLAYER_NAMES) == ("Player 5",)


def test_literal_policy_reproduces_the_legacy_string_mismatch() -> None:
    narrow = f"Player{NARROW_NBSP}6に説明を求めます。"
    # 旧来判定（`Player\s+\d+` の完全一致）では表記ゆれが未知名前として残る
    assert unresolved_references(narrow, PLAYER_NAMES, policy="literal")
    assert not has_unresolved_reference(narrow, PLAYER_NAMES)
    assert names_in_text(narrow, PLAYER_NAMES) == ("Player 6",)


def test_unknown_numbers_are_never_forced_into_existing_names() -> None:
    for text in ("Player 9", "P 12へ投票", "プレイヤー１２"):
        assert has_unresolved_reference(text, PLAYER_NAMES), text
    assert not has_unresolved_reference("Player 3", PLAYER_NAMES)
    lookup = build_name_lookup(("Alice",))
    assert lookup.get("alice") == "Alice"
    assert "player1" not in lookup  # 番号なし固有名詞に別ラベルを作らない


def test_larger_numbers_do_not_resolve_into_smaller_player_numbers() -> None:
    # `player 100` が `player 1` の副串として Player 1 にならない（TASK-065の境界ガード）
    for text in ("player 100を参照", "Player 67に投票", "プレイヤー45が言った"):
        assert has_unresolved_reference(text, PLAYER_NAMES), text
        assert resolve_known_names(text, PLAYER_NAMES) == ()
    # 桁の後ろに非数字が続く通常の言及はこれまでどおり解決する
    assert resolve_known_names("Player 1番に投票", PLAYER_NAMES) == ("Player 1",)
    assert resolve_known_names("Player 6、お願いします", PLAYER_NAMES) == ("Player 6",)


def test_name_resolution_never_rewrites_the_public_body() -> None:
    original = f"Player{NARROW_NBSP}6は疑わしい。player7も同じ投票をした。"
    resolved = set(resolve_known_names(original, PLAYER_NAMES))
    assert resolved == {"Player 6", "Player 7"}
    assert NARROW_NBSP in original  # 公開ログに残る表記はこのまま
    # 解決名は本文側に存在しない（＝書き換えではなく内部照合だけである証拠）
    assert all(name not in original for name in resolved)


def test_question_answer_attribution_uses_resolved_names() -> None:
    asked = f"Player{NARROW_NBSP}6、初日の投票理由を教えてください。"
    answered = "Player 6さんへの質問には答えませんが、投票理由は観察に基づきます。"
    assert shares_mentioned_players(asked, answered, PLAYER_NAMES)
    assert not shares_mentioned_players(
        asked, answered, PLAYER_NAMES, policy=NAME_RESOLUTION_POLICY_LITERAL
    )
    assert not shares_mentioned_players(asked, "Player 8についての話です。", PLAYER_NAMES)


def test_name_resolution_policy_helpers() -> None:
    assert normalize_name_resolution_policy("LITERAL") == "literal"
    assert normalize_name_resolution_policy(None) == NAME_RESOLUTION_POLICY_NORMALIZED
    assert normalize_name_resolution_policy("bogus") == NAME_RESOLUTION_POLICY_NORMALIZED
    assert (
        name_resolution_policy_label(NAME_RESOLUTION_POLICY_LITERAL)
        == "exact_string_reference_lookup"
    )
    assert name_resolution_policy_label(None) == "canonical_reference_lookup"

# ---------------------------------------------------------------------------
# 2. 時系列検出の意図分離（質問行為を未来参照に数えない）
# ---------------------------------------------------------------------------
BASELINE_QUESTION_SENTENCE = (
    "次に、プレイヤー6に具体的な発言内容や投票理由を問いただすことを提案します。"
)


def _snapshot_for_late_speaker(seed: int, actor: str = "Player 2"):
    state = _state(seed=seed)
    _start_day(state)
    state.append_discussion_message(Message("Player 1", "私はまだ情報を集めているところです。"))
    return build_turn_snapshot(state, actor), tuple(
        player.name for player in state.players
    )


def test_question_act_toward_unspoken_player_is_not_future_reference() -> None:
    snapshot, names = _snapshot_for_late_speaker(2065065)
    separated = inspect_public_body(
        BASELINE_QUESTION_SENTENCE,
        snapshot,
        names,
        intent_policy="separated",
    )
    # これから質問する行為 → findings には数えない（TASK-059までの誤検出の解消）
    assert separated.findings == ()
    assert separated.future_question_acts == 1
    assert separated.intent_counts.get(INTENT_FUTURE_QUESTION_TO_UNSPOKEN) == 1
    assert separated.past_attributions_observed == 0
    fields = separated.to_log_fields()
    assert fields["future_event_reference_count"] == 0
    assert fields["timeline_intent_counts"][INTENT_FUTURE_QUESTION_TO_UNSPOKEN] == 1
    assert fields["timeline_intent_policy_label"] == "intent_separated_detection"
    # 本文は検出結果にかかわらず変更されない（観測専用であることの証跡）
    assert fields["body_modified"] is False


def test_literal_policy_keeps_the_legacy_flat_detection() -> None:
    snapshot, names = _snapshot_for_late_speaker(2065066)
    literal = inspect_public_body(
        BASELINE_QUESTION_SENTENCE,
        snapshot,
        names,
        intent_policy="literal",
    )
    assert len(literal.findings) == 1
    finding = literal.findings[0]
    assert finding.category == FUTURE_SPEECH_FROM_UNSPOKEN
    assert finding.subtype == SUBTYPE_ATTRIBUTION_NEVER_SPOKEN
    assert literal.intent_counts == {}
    assert literal.future_question_acts == 0
    assert timeline_intent_policy_label("literal") == "legacy_flat_detection"


def test_past_attribution_to_unspoken_player_stays_detected() -> None:
    # 「その人が過去に述べた」という実質的な未来参照は separated でも必ず検出する
    snapshot, names = _snapshot_for_late_speaker(2065067)
    body = "Player 6が事前に人狼だと主張したと聞いているので、投票を集中させます。"
    separated = inspect_public_body(body, snapshot, names, intent_policy="separated")
    assert len(separated.findings) == 1
    assert separated.findings[0].category == FUTURE_SPEECH_FROM_UNSPOKEN
    assert separated.findings[0].intent == INTENT_PAST_ATTRIBUTION
    assert separated.intent_counts.get(INTENT_PAST_ATTRIBUTION) == 1
    assert separated.past_attributions_observed == 1
    assert separated.future_question_acts == 0
    assert separated.findings[0].to_dict().get("intent") == INTENT_PAST_ATTRIBUTION


def test_future_action_speculation_is_counted_separately() -> None:
    # 未発言者の「これからの行動」への推測は違反ではなく意図として数える
    snapshot, names = _snapshot_for_late_speaker(2065068)
    body = "Player 6は次の投票で私に票を入れるはずだ。だから先に牽制しておく。"
    report = inspect_public_body(body, snapshot, names)
    assert report.intent_policy == TIMELINE_INTENT_POLICY_SEPARATED
    assert report.intent_counts.get(INTENT_FUTURE_ACTION_SPECULATION) == 1
    assert report.future_action_speculations == 1
    assert report.findings == ()  # 引用語がないため未来参照にはしない（従来条件を壊さない）


def test_intent_policy_default_and_helpers() -> None:
    assert normalize_timeline_intent_policy("LITERAL") == "literal"
    assert normalize_timeline_intent_policy(None) == TIMELINE_INTENT_POLICY_SEPARATED
    assert normalize_timeline_intent_policy("junk") == TIMELINE_INTENT_POLICY_SEPARATED
    assert timeline_intent_policy_label("separated") == "intent_separated_detection"

# ---------------------------------------------------------------------------
# 3. 実効コンテキスト整合（configured / server_observed / loaded / requested）
# ---------------------------------------------------------------------------
PLAYER1_CASE = {
    "model": MODEL_NAME,
    "configured": 16384,
    "configured_source": "player_setting",
    "ps_loaded_context": None,
    "modelfile_num_ctx": 4096,
    "process_env_num_ctx": 4096,
}


def test_warn_records_mismatch_without_changing_behavior() -> None:
    check = assess_context_integrity(**PLAYER1_CASE, policy="warn")
    assert check.mismatch is True
    assert KIND_CONFIGURED_ABOVE_SERVER in check.kinds
    assert KIND_LOADED_UNAVAILABLE in check.kinds
    assert check.requested_num_ctx == 16384
    assert check.applied is False
    assert check.adopted_num_ctx == 16384  # warn は挙動を変えない（記録だけの分離）
    assert check.safe_value == 4096
    warnings = "\n".join(check.warning_lines())
    assert "4096" in warnings and "超えています" in warnings
    fields = check.to_fields()
    assert fields["context_integrity_applied"] is False
    assert fields["requested_num_ctx"] == 16384
    assert fields["safe_context_value"] == 4096


def test_cap_policy_adopts_the_safe_value_before_send() -> None:
    check = assess_context_integrity(**PLAYER1_CASE, policy=CONTEXT_INTEGRITY_POLICY_CAP)
    assert check.applied is True
    assert check.adopted_num_ctx == 4096
    assert check.safe_value == 4096
    assert any("縮退" in line for line in check.warning_lines())
    off_check = assess_context_integrity(**PLAYER1_CASE, policy="off")
    assert off_check.policy == "off"
    assert off_check.applied is False  # off では縮退しない（旧挙動・A/B比較用）


def test_aligned_evidence_produces_no_warnings_and_hides_no_secrets() -> None:
    aligned = assess_context_integrity(
        "gpt-oss-20b",  # 役職語を含まないモデル名で検証する（実在名の副串は数えない）
        configured=8192,
        configured_source="player_setting",
        ps_loaded_context=8192,
        modelfile_num_ctx=4096,
        process_env_num_ctx=None,
        policy="warn",
    )
    assert aligned.kinds == (KIND_ALIGNED,)
    assert aligned.mismatch is False
    assert aligned.warning_lines() == ()
    blob = str(aligned.to_fields())
    for secret_term in ("人狼", "占い師", "襲撃", "占い結果"):
        assert secret_term not in blob  # 数値とモデル名だけの出力（公開情報の外側）
    assert normalize_context_integrity_policy("CAP") == CONTEXT_INTEGRITY_POLICY_CAP
    assert normalize_context_integrity_policy(None) == CONTEXT_INTEGRITY_POLICY_WARN
    assert context_integrity_policy_label("off") == "integrity_check_disabled"
    assert context_integrity_policy_label("cap") == "safe_value_adoption"


# ---------------------------------------------------------------------------
# 4. モデル×行動別の安定化（恒久適用にしない・成功で解除）
# ---------------------------------------------------------------------------
def test_downgrade_is_scoped_to_the_model_action_pair() -> None:
    tracker = OutputStabilityTracker()
    assert tracker.scope == "action"
    payload = tracker.apply_downgrade(MODEL_NAME, "vote", DOWNGRADE_TIMEOUT_FIT)
    assert payload["scope"] == OUTPUT_STABILIZATION_SCOPE_ACTION
    assert tracker.has_downgrade(MODEL_NAME, "vote", DOWNGRADE_TIMEOUT_FIT)
    # 同じモデルの別行動（議論）は自由なまま（投票で詰まっただけで議論を絞らない）
    assert not tracker.has_downgrade(MODEL_NAME, "discussion", DOWNGRADE_TIMEOUT_FIT)
    assert not tracker.model_wide_evidence(MODEL_NAME, DOWNGRADE_TIMEOUT_FIT)
    assert payload["first_application"] is True


def test_model_wide_evidence_needs_two_actions() -> None:
    tracker = OutputStabilityTracker()
    tracker.apply_downgrade(MODEL_NAME, "vote", DOWNGRADE_THINKING_OFF)
    assert not tracker.model_wide_evidence(MODEL_NAME, DOWNGRADE_THINKING_OFF)
    tracker.apply_downgrade(MODEL_NAME, "attack", DOWNGRADE_THINKING_OFF)
    assert tracker.model_wide_evidence(MODEL_NAME, DOWNGRADE_THINKING_OFF)
    assert tracker.actions_with(MODEL_NAME, DOWNGRADE_THINKING_OFF) == {"vote", "attack"}
    model_scope = OutputStabilityTracker("model")
    assert model_scope.scope == OUTPUT_STABILIZATION_SCOPE_MODEL
    assert output_stabilization_scope_label("model") == "whole_model_downgrade"
    assert normalize_output_stabilization_scope("junk") == "action"


def test_relax_after_success_streak_returns_thinking_freedom() -> None:
    tracker = OutputStabilityTracker()
    tracker.record(MODEL_NAME, "discussion", CATEGORY_THINKING_BUDGET)
    tracker.apply_downgrade(MODEL_NAME, "discussion", DOWNGRADE_THINKING_OFF)
    assert tracker.relax(MODEL_NAME, "discussion") is None  # まだ成功が無い
    for _ in range(3):
        tracker.record(MODEL_NAME, "discussion", CATEGORY_OK)
    relaxed = tracker.relax(MODEL_NAME, "discussion")
    assert relaxed is not None
    assert relaxed["cleared_downgrades"] == [DOWNGRADE_THINKING_OFF]
    assert not tracker.has_downgrade(MODEL_NAME, "discussion", DOWNGRADE_THINKING_OFF)


def test_logic_and_format_failures_are_not_budget_pressure() -> None:
    tracker = OutputStabilityTracker()
    tracker.record(MODEL_NAME, "vote", CATEGORY_OUTPUT_FORMAT)
    tracker.record(MODEL_NAME, "vote", CATEGORY_PROMPT_LOGIC)
    _, streak = tracker.streaks(MODEL_NAME, "vote")
    assert streak == 0  # 議論の論理側・形式側を縮小の材料にしない
    tracker.record(MODEL_NAME, "vote", CATEGORY_TIMEOUT)
    tracker.record(MODEL_NAME, "vote", CATEGORY_GENERATION_CAP)
    _, streak = tracker.streaks(MODEL_NAME, "vote")
    assert streak == 2
    snapshot = tracker.serialize()
    assert snapshot["scope"] == "action"
    row = snapshot["models"][MODEL_NAME]["vote"]
    assert row["budget_bound_in_window"] >= 2 and row["budget_bound_streak"] == 2

# ---------------------------------------------------------------------------
# 5. state・起動時preflight・質問指名の統合配線（LL不使用）
# ---------------------------------------------------------------------------
class _StubClient:
    """/api/ps相当の診断だけ返すスタブ（実機Ollamaは使わない）。"""

    def __init__(self, ps_context=None, modelfile=None) -> None:
        self._ps = ps_context
        self._modelfile = modelfile

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {"ollama_ps_context": self._ps, "modelfile_num_ctx": self._modelfile}


class _StubLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def append_record(self, public: dict[str, object], private=None) -> None:
        self.records.append(dict(public))


def test_state_defaults_and_policy_normalization() -> None:
    state = _state(seed=2065070)
    # 既定はすべて「節約・一貫性を保つ側」。旧挙動はA/B用の逃げ道として残す
    assert state.timeline_intent_policy == "separated"
    assert state.player_name_resolution_policy == "normalized"
    assert state.context_integrity_policy == CONTEXT_INTEGRITY_POLICY_DEFAULT
    assert state.output_stabilization_scope == "action"
    assert isinstance(state.context_integrity_checks, dict)
    # 未知の設定値はそれぞれ既定へ、A/B用の別名（MODEL等）は正規化して受け取る
    state.timeline_intent_policy = "bogus"
    state.player_name_resolution_policy = "bogus"
    state.context_integrity_policy = "bogus"
    state.output_stabilization_scope = "MODEL"
    state.__post_init__()
    assert state.timeline_intent_policy == "separated"
    assert state.player_name_resolution_policy == "normalized"
    assert state.context_integrity_policy == "warn"
    assert state.output_stabilization_scope == "model"


def test_preflight_records_evidence_and_warns_before_first_send() -> None:
    from werewolf.cli import run_context_integrity_preflight

    state = _state(seed=2065071)
    logger = _StubLogger()
    client = _StubClient(ps_context=None, modelfile=4096)
    run_context_integrity_preflight(client, state, logger)
    unique_models = {player.model for player in state.players if player.model}
    assert len(logger.records) == len(unique_models)
    assert all(r["type"] == "context_integrity_preflight" for r in logger.records)
    assert set(state.context_integrity_checks.keys()) == unique_models
    sample = next(iter(state.context_integrity_checks.values()))
    assert "configured_context_limit" in sample and "safe_context_value" in sample
    # off 設定では診断ごと行わない（旧挙動・A/B比較用）
    state.context_integrity_policy = "off"
    logger_off = _StubLogger()
    assert run_context_integrity_preflight(client, state, logger_off) == []
    assert logger_off.records == []


def test_directed_question_recognition_absorbs_name_variants() -> None:
    state = _state(seed=2065072)
    actor = next(player for player in state.players if player.name == "Player 6")
    history = (
        Message("system", "Day 1 begins."),
        Message("Player 2", "今日は情報集めから始めたい。"),
    )
    canonical = Message("Player 2", "Player 6、初日の投票理由を教えてください。")
    variant = Message("Player 2", f"Player{NARROW_NBSP}6、初日の投票理由を教えてください。")
    unrelated = Message("Player 2", "今日は夜が落ち着きそうだ。")
    assert PromptBuilder._is_directed_to_player(actor, canonical, history)
    # 表記ゆれでも「自分への質問」を認識する（公開本文の表記はそのまま）
    assert PromptBuilder._is_directed_to_player(actor, variant, history)
    assert not PromptBuilder._is_directed_to_player(actor, unrelated, history)


# ---------------------------------------------------------------------------
# 6. cap方策が実際の送信値へ効くか（レビュー対応の統合確認・LL不使用）
# ---------------------------------------------------------------------------
class _SendCaptureClient:
    """送信オプションを記録するモック（実Ollamaには一切送らない）。

    ps は「起動時（未ロード）=None → 送信後=実ロード値」と段階的に変わる構成にでき、
    実機での流れ（ロードは送信時に起きる）を再現する。
    """

    def __init__(self, *, modelfile=4096, ps_before=None, ps_after=None) -> None:
        self.timeout = 120.0
        self.max_retries = 1
        self.timeout_retries = 0
        self._modelfile = modelfile
        self._ps_before = ps_before
        self._ps_after = ps_after
        self.sent_options: list[dict[str, object]] = []
        self.sent_prompts: list[list[dict[str, object]]] = []
        self._sends = 0

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {
            "ollama_ps_context": (self._ps_after if self._sends > 0 else self._ps_before),
            "modelfile_num_ctx": self._modelfile,
        }

    def chat(
        self, model: str, messages: list[dict[str, object]], options: dict | None = None
    ) -> str:
        self._sends += 1
        self.sent_options.append(dict(options or {}))
        self.sent_prompts.append([dict(message) for message in messages])
        return "[CO] None\n[BODY] 了解しました。情報集めを続けます。"


def _player_with_large_ctx(state, num_ctx: int):
    """Player 1型（設定は大きい / サーバー観測は小さい）をテスト用に作る。"""
    actor = state.players[0]
    actor.num_ctx = num_ctx
    actor.recommended_num_ctx = None
    actor.max_num_ctx = None
    return actor


def test_cap_policy_sends_the_adopted_num_ctx_not_the_setting(monkeypatch) -> None:
    """cap方策では解決結果の採用値が実際の送信num_ctxへ反映される（レビュー指摘の修正）。

    実機ログの実例（設定16384 / Modelfile・env 4096）を再現する。修正前は
    `context_integrity_applied=true adopted_num_ctx=4096` でも送信側は旧 limit を
    使い続けていたため、送信オプションとリクエスト設定ログの両方を見る。
    """
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    monkeypatch.delenv("OLLAMA_CONTEXT_LENGTH", raising=False)
    from werewolf.llm_actions import OllamaActions

    state = _state(seed=2065080)
    state.context_integrity_policy = CONTEXT_INTEGRITY_POLICY_CAP
    actor = _player_with_large_ctx(state, 16384)
    events: list[dict[str, object]] = []
    client = _SendCaptureClient(modelfile=4096, ps_before=None, ps_after=None)
    actions = OllamaActions(state, client, seed=2065080, event_callback=events.append)

    resolved = actions._resolve_context_budget(actor, "discussion", time.perf_counter())
    assert resolved.limit == 4096
    assert resolved.source == "context_integrity_cap"
    assert resolved.integrity_requested_limit == 16384
    assert resolved.integrity_policy == CONTEXT_INTEGRITY_POLICY_CAP
    assert resolved.provisional is True  # /api/ps が空なので再解決可能マーク

    body = actions.discuss_body(
        actor.name, 1, tuple(state.alive_players), tuple(state.conversation_log)
    )
    assert isinstance(body, str) and body.strip()
    sent = client.sent_options[0]
    assert isinstance(sent.get("num_ctx"), int)
    assert int(sent["num_ctx"]) <= 4096, f"送信値が採用値を超えている: {sent}"
    configs = [e for e in events if e["type"] == "llm_request_configuration"]
    assert configs
    assert int(configs[0]["requested_num_ctx"]) <= 4096
    assert configs[0]["context_budget_source_after_integrity"] == "context_integrity_cap"
    assert configs[0]["integrity_requested_num_ctx"] == 16384
    resolution_events = [e for e in events if e["type"] == "effective_context_resolution"]
    assert resolution_events[0]["adopted_num_ctx"] == 4096
    assert resolution_events[0]["effective_context_limit"] == 4096


def test_warn_policy_keeps_the_setting_value_in_send(monkeypatch) -> None:
    """warn（既定）は記録だけで挙動不変。capと違い送信値は設定値のまま残る。"""
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    monkeypatch.delenv("OLLAMA_CONTEXT_LENGTH", raising=False)
    from werewolf.llm_actions import OllamaActions

    state = _state(seed=2065081)
    actor = _player_with_large_ctx(state, 16384)
    client = _SendCaptureClient(modelfile=4096, ps_before=None, ps_after=None)
    actions = OllamaActions(state, client, seed=2065081)
    resolved = actions._resolve_context_budget(actor, "discussion", time.perf_counter())
    assert resolved.limit == 16384
    assert resolved.source != "context_integrity_cap"
    assert resolved.provisional is True
    actions.discuss_body(
        actor.name, 1, tuple(state.alive_players), tuple(state.conversation_log)
    )
    sent = client.sent_options[0]
    assert int(sent["num_ctx"]) == 16384


def test_loaded_context_is_reobserved_after_first_send(monkeypatch) -> None:
    """ロード後の実測値がキャッシュされず、次解決で `/api/ps` を取り直す（レビュー指摘）。

    実機では起動時にモデル未ロードなので ps=None。以後も旧キャッシュを返し続けると
    実ロード値を二度と参照できない。送信が完了したら1回だけ取り直すことを確認する。
    """
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    monkeypatch.delenv("OLLAMA_CONTEXT_LENGTH", raising=False)
    from werewolf.llm_actions import OllamaActions

    state = _state(seed=2065082)
    actor = _player_with_large_ctx(state, 12288)
    events: list[dict[str, object]] = []
    client = _SendCaptureClient(modelfile=8192, ps_before=None, ps_after=6144)
    actions = OllamaActions(state, client, seed=2065082, event_callback=events.append)

    first = actions._resolve_context_budget(actor, "discussion", time.perf_counter())
    assert first.loaded_context_from_ps is None
    assert first.provisional is True
    # 送信前なら再解決しない（同じキャッシュを返す＝診断を毎回は走らせない）
    again = actions._resolve_context_budget(actor, "discussion", time.perf_counter())
    assert again is first

    actions.discuss_body(
        actor.name, 1, tuple(state.alive_players), tuple(state.conversation_log)
    )
    refreshed = actions._resolve_context_budget(actor, "discussion", time.perf_counter())
    assert refreshed.loaded_context_from_ps == 6144
    assert refreshed.provisional is False  # ロード値が取れたので以後は確定キャッシュ
    stages = [
        e.get("context_resolution_stage")
        for e in events
        if e["type"] == "effective_context_resolution"
    ]
    assert "initial" in stages and "post_load_refresh" in stages
    # 3回目は確定キャッシュ（これ以上イベントを増やさない）
    final = actions._resolve_context_budget(actor, "discussion", time.perf_counter())
    assert final is refreshed
    assert len([e for e in events if e["type"] == "effective_context_resolution"]) == len(stages)



def test_cap_adopts_loaded_value_when_smaller_than_modelfile(monkeypatch) -> None:
    """capは「実際にロードされた値」を安全側の証拠として採用する（Modelfileより優先）。

    Modelfile 8192 / ロード実測 4096・設定 12288 の状況で、初回（未ロード）は env/Modelfile
    側の安全値 8192、送信後は実ロード値 4096 まで縮退し以後その枠で送ることを確認する。
    """
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    monkeypatch.delenv("OLLAMA_CONTEXT_LENGTH", raising=False)
    from werewolf.llm_actions import OllamaActions

    state = _state(seed=2065083)
    state.context_integrity_policy = CONTEXT_INTEGRITY_POLICY_CAP
    actor = _player_with_large_ctx(state, 12288)
    client = _SendCaptureClient(modelfile=8192, ps_before=None, ps_after=4096)
    actions = OllamaActions(state, client, seed=2065083)

    first = actions._resolve_context_budget(actor, "discussion", time.perf_counter())
    assert first.limit == 8192  # ロード値未知 → env / Modelfile の安全値
    first_body = actions.discuss_body(
        actor.name, 1, tuple(state.alive_players), tuple(state.conversation_log)
    )
    assert first_body.strip()
    refreshed = actions._resolve_context_budget(actor, "discussion", time.perf_counter())
    assert refreshed.loaded_context_from_ps == 4096
    assert refreshed.limit == 4096
    assert refreshed.source == "context_integrity_cap"
    assert refreshed.integrity_requested_limit == 12288
    second_body = actions.discuss_body(
        actor.name, 1, tuple(state.alive_players), tuple(state.conversation_log)
    )
    assert second_body.strip()
    sent_ctx = [o.get("num_ctx") for o in client.sent_options[1:]]
    assert all(isinstance(value, int) and int(value) <= 4096 for value in sent_ctx), sent_ctx

