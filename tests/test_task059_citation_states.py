"""TASK-059の回帰テスト（LL不使用）: 引用の時点分類と未解決参照の原因分解。

実機ログ game_106015c79ec9 では議論54件のうち時系列検査53件、未来参照8件
（death_before_death=1 / speech_from_unspoken_player=6 / night_action_before_night=1）、
未解決参照6件で `future_event_reference_rate=0.1509` だった。TASK-056の方針どおり本文は
直していないが、「誰の・どの時点の発言」を引きたかったのかを区別できず、婉曲な仮定や単なる
名指しまで違反に数えている可能性が残る。本テストは以下を固定する（実機LLMは使わない）。

1. 引用された相手の時点状態を4分類する（同一ターンの先行発言／別ターン・別日の既発言／
   死亡後の履歴参照／まだ公開発言なし）。実際の未来参照候補は最後の状態だけ。
2. 判定（future_event_reference）は従来条件のみ。婉曲での仮定引用と未発言者名の一般言及は
   観察カウンタへ逃がし、自由な議論を萎ませない。
3. 本文の `[speech:N]` / `[ref:sN]` を時点照合し原因を分ける（解決済み／まだ存在しないID＝
   実際の未来参照／時点情報がない旧ログ互換／帰属なのにIDなし）。
4. `day_timeline=off` の互換動作と公開台帳の字数を維持する（台帳情報を水増ししない）。
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.day_timeline import (
    CAUSE_NO_PUBLIC_SPEECH_AT_POINT,
    CITATION_DEAD_HISTORY,
    CITATION_EARLIER_TURN,
    CITATION_NEVER_SPOKEN_YET,
    CITATION_SAME_TURN,
    CITATION_STATUSES,
    DAY_TIMELINE_POLICY_OFF,
    FUTURE_SPEECH_FROM_UNSPOKEN,
    REFERENCE_CAUSE_ID_NOT_SPOKEN_YET,
    REFERENCE_CAUSE_LEGACY_HISTORY,
    REFERENCE_CAUSE_MISSING_ID,
    REFERENCE_RESOLVED_AT_POINT,
    SUBTYPE_ATTRIBUTION_NEVER_SPOKEN,
    TurnSnapshot,
    build_turn_snapshot,
    classify_reference_tokens,
    inspect_public_body,
)
from werewolf.engine import GameEngine
from werewolf.models import Message, Phase
from werewolf.setup import create_game


def _state():
    return create_game(seed=2059, config_path="config/players.json")


def _names(state) -> tuple[str, ...]:
    return tuple(player.name for player in state.players)


def _start_day1(state, turn: int = 1) -> None:
    state.public_timeline.record_day_start(state.day)
    state.conversation_log.append(Message("system", f"Day {state.day} begins."))
    state.phase = Phase.DAY_DISCUSSION
    state.discussion_turn = turn
    state.begin_discussion_turn()


def _say(state, name: str, content: str) -> Message:
    return state.append_discussion_message(Message(name, content))


# ---------------------------------------------------------------------------
# 1. 引用できる時点の分類（実装案2）
# ---------------------------------------------------------------------------
def test_citation_status_splits_same_turn_earlier_turn_and_never_spoken() -> None:
    state = _state()
    _start_day1(state, turn=1)
    _say(state, "Player 1", "T1で先に発言しておく。")
    state.discussion_turn = 2
    snapshot = build_turn_snapshot(state, "Player 2")
    # T2時点では Player 1 は「別ターンの既発言」、Player 3 は未発言扱い
    assert snapshot.citation_status("Player 1") == CITATION_EARLIER_TURN
    assert snapshot.citation_status("Player 3") == CITATION_NEVER_SPOKEN_YET
    _say(state, "Player 2", "T2で発言する。")
    same_turn = build_turn_snapshot(state, "Player 3")
    assert same_turn.citation_status("Player 2") == CITATION_SAME_TURN
    assert same_turn.citation_status("Player 1") == CITATION_EARLIER_TURN
    assert set(CITATION_STATUSES) == {
        CITATION_SAME_TURN,
        CITATION_EARLIER_TURN,
        CITATION_DEAD_HISTORY,
        CITATION_NEVER_SPOKEN_YET,
    }


def test_citation_status_of_a_dead_player_is_public_history() -> None:
    state = _state()
    _start_day1(state, turn=1)
    _say(state, "Player 1", "生存中に一度発言した。")
    state.public_timeline.record_death(1, name="Player 1", executed=False)
    state.mark_dead("Player 1")
    state.attack_deaths.append((1, "Player 1"))
    snapshot = build_turn_snapshot(state, "Player 2")
    assert snapshot.citation_status("Player 1") == CITATION_DEAD_HISTORY
    # 死亡したが一度も発言していない人は履歴に引ける発言が無い
    state.public_timeline.record_death(1, name="Player 4", executed=False)
    state.mark_dead("Player 4")
    state.attack_deaths.append((1, "Player 4"))
    later = build_turn_snapshot(state, "Player 2")
    assert later.citation_status("Player 4") == CITATION_NEVER_SPOKEN_YET


# ---------------------------------------------------------------------------
# 2. 判定は変えない（婉曲と一般名指しは観察専用）
# ---------------------------------------------------------------------------
def test_hedged_citation_of_a_pending_speaker_is_observed_not_flagged() -> None:
    state = _state()
    _start_day1(state, turn=3)
    _say(state, "Player 1", "T3で先に発言する。")
    snapshot = build_turn_snapshot(state, "Player 2")
    hedged = "もし Player 6 が先に発言していたなら、その言い方を参考にできたと思う。"
    report = inspect_public_body(hedged, snapshot, _names(state))
    # 断定ではないので future_event_reference には数えない（TASK-056と同じ判定）
    assert report.findings == ()
    assert report.unresolvable_references == 0
    # 「未発言者を仮定で引いた」ことだけを観察値として残す（誤検出率の監視用）
    assert report.hedged_pending_citations == 1
    assert report.citation_status_counts[CITATION_NEVER_SPOKEN_YET] >= 1
    fields = report.to_log_fields()
    assert fields["body_modified"] is False
    assert int(fields["hedged_pending_citations_observed"]) == 1


def test_general_mention_without_attribution_is_not_an_error() -> None:
    state = _state()
    _start_day1(state, turn=1)
    snapshot = build_turn_snapshot(state, "Player 2")
    plain = "Player 7 の立場も考えて、村としては慎重に議論を進めたい。"
    report = inspect_public_body(plain, snapshot, _names(state))
    # 名指ししただけ（発言の帰属ではない）なので違反にはしない
    assert report.findings == ()
    assert report.attributed_references == 0
    assert report.mentions_without_attribution == 1
    assert int(report.to_log_fields()["mentions_without_attribution_observed"]) == 1


def test_true_misattribution_is_still_flagged_with_subtype_and_cause() -> None:
    state = _state()
    _start_day1(state, turn=3)
    _say(state, "Player 1", "T3で先に発言する。")
    snapshot = build_turn_snapshot(state, "Player 2")
    body = "Player 6 の発言を聞いたが、あの人は村人だろう。"
    report = inspect_public_body(body, snapshot, _names(state))
    assert [finding.category for finding in report.findings] == [
        FUTURE_SPEECH_FROM_UNSPOKEN
    ]
    finding = report.findings[0]
    # 内訳タグ（観測専用）。実際の未来参照かID欠落かを後から分解できる
    assert finding.subtype == SUBTYPE_ATTRIBUTION_NEVER_SPOKEN
    assert finding.cause == CAUSE_NO_PUBLIC_SPEECH_AT_POINT
    payload = finding.to_dict()
    assert payload["subtype"] == SUBTYPE_ATTRIBUTION_NEVER_SPOKEN
    assert payload["cause"] == CAUSE_NO_PUBLIC_SPEECH_AT_POINT
    assert report.to_log_fields()["body_modified"] is False


def test_citation_of_an_earlier_turn_speech_is_not_unresolved() -> None:
    state = _state()
    _start_day1(state, turn=1)
    first = _say(state, "Player 1", "T1の発言内容を確認したい。")
    state.discussion_turn = 2
    snapshot = build_turn_snapshot(state, "Player 2")
    body = "Player 1 の発言を受けて、私は投票先を変えたい。"
    report = inspect_public_body(body, snapshot, _names(state))
    assert report.findings == ()
    assert report.attributed_references >= 1
    assert report.unresolvable_references == 0
    # 引用側は「別ターンの既発言」として計上される（＝時点は追えている）
    assert report.citation_status_counts[CITATION_EARLIER_TURN] >= 1
    assert first.speech_id == 1


# ---------------------------------------------------------------------------
# 3. `[speech:N]` / `[ref:sN]` 参照の原因分類（実装案3）
# ---------------------------------------------------------------------------
def test_reference_tokens_are_resolved_against_the_speech_point_ledger() -> None:
    state = _state()
    _start_day1(state, turn=2)
    _say(state, "Player 1", "先に発言した内容です。")
    snapshot = build_turn_snapshot(state, "Player 2")
    known = int(snapshot.all_speech_points[0][3])
    counts = classify_reference_tokens(f"[speech:{known}] を見ると状況が掴める。", snapshot)
    assert counts == {REFERENCE_RESOLVED_AT_POINT: 1}
    future = classify_reference_tokens("[ref:s99] を根拠にしたい。", snapshot)
    assert future == {REFERENCE_CAUSE_ID_NOT_SPOKEN_YET: 1}


def test_reference_tokens_in_legacy_history_are_unverifiable() -> None:
    state = _state()
    _start_day1(state, turn=1)
    snapshot = TurnSnapshot(
        day=1,
        turn=1,
        actor="Player 2",
        sequence=1,
        prior_this_turn=(),
        own_speeches_today=(),
        prior_today_events=0,
        speakers_pending=_names(state),
        spoken_players=(),
        known_dead=(),
        alive_players=tuple(state.alive_players),
        today_vote_done=False,
        night_pending=True,
        next_speech_id=1,
        all_speech_points=(),  # 旧ログ・手組み履歴（時点情報なし）
    )
    counts = classify_reference_tokens("[speech:4] の通り進めたい。", snapshot)
    assert counts == {REFERENCE_CAUSE_LEGACY_HISTORY: 1}


def test_citation_without_an_evidence_id_is_counted_as_missing_id() -> None:
    state = _state()
    _start_day1(state, turn=1)
    _say(state, "Player 1", "T1で先に発言した内容だ。")
    snapshot = build_turn_snapshot(state, "Player 2")
    body = "Player 1 が言ったとおりの進め方だと思う。"
    report = inspect_public_body(body, snapshot, _names(state))
    # 帰属しているのに証拠IDが無い（追跡の糸口が本文しかない）ケースを分けて数える
    assert report.reference_token_counts.get(REFERENCE_CAUSE_MISSING_ID) == 1
    assert REFERENCE_RESOLVED_AT_POINT not in report.reference_token_counts
    assert report.findings == ()
    assert report.to_log_fields()["evidence_reference_counts"][
        REFERENCE_CAUSE_MISSING_ID
    ] == 1


# ---------------------------------------------------------------------------
# 4. エンジン統合（本文を変えずに分類だけ残す）
# ---------------------------------------------------------------------------
class _NoActions:
    def wolf_discuss(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("このテストでは夜行動を実行しない")


def _engine_with_events(state):
    events = []
    engine = GameEngine(
        state, _NoActions(), seed=state.seed, event_callback=events.append
    )
    return engine, events


def test_engine_reports_citation_states_without_touching_the_body() -> None:
    state = _state()
    _start_day1(state, turn=1)
    engine, events = _engine_with_events(state)
    body = "Player 5 が言っていた内容を踏まえて、私は慎重に投票したい。"
    engine._append_discussion_message(Message("Player 6", body))
    event = next(e for e in events if e["type"] == "discussion_message")
    # 公開本文は一字一句そのまま（分類のために語彙・口調・結論を変えない）
    assert event["content"] == body
    assert event["body_modified"] is False
    # 引用した相手の時点状態と観察値がイベントに乗る（cli側でモデル別に分解できる）
    assert "citation_status_counts" in event
    assert "evidence_reference_counts" in event
    assert "hedged_pending_citations_observed" in event
    assert event["timeline_checks_skipped_empty_body"] is False
    timeline = state.public_timeline
    assert timeline.bodies_checked == 1
    assert timeline.bodies_kept_unmodified == 1


def test_empty_public_body_is_reported_as_a_skipped_check() -> None:
    state = _state()
    _start_day1(state, turn=1)
    engine, events = _engine_with_events(state)
    engine._append_discussion_message(Message("Player 6", ""))
    event = next(e for e in events if e["type"] == "discussion_message")
    # 「数えなかった理由」を残して検査対象数の水増しを防ぐ（TASK-059完了条件）
    assert event["timeline_checks_skipped_empty_body"] is True


# ---------------------------------------------------------------------------
# 5. 互換動作: off では検査しない／公開台帳を水増ししない
# ---------------------------------------------------------------------------
def test_off_policy_still_skips_inspection_and_citation_fields() -> None:
    state = _state()
    _start_day1(state, turn=1)
    state.timeline_policy = DAY_TIMELINE_POLICY_OFF
    state.public_timeline.policy = DAY_TIMELINE_POLICY_OFF
    engine, events = _engine_with_events(state)
    engine._append_discussion_message(
        Message("Player 6", "Player 5 が言っていたと思う。")
    )
    event = next(e for e in events if e["type"] == "discussion_message")
    assert event["timeline_observed"] is False
    assert "citation_status_counts" not in event


def test_public_ledger_metrics_keep_the_old_shape() -> None:
    # TASK-056で固定した「台帳の公開面は中立カウンタのみ」を壊さない（字数を増やさない）
    state = _state()
    _start_day1(state, turn=1)
    _say(state, "Player 1", "とても長い発言 " * 40)
    metrics = state.public_timeline.serialize()["metrics"]
    for key in (
        "citation_status_counts",
        "hedged_pending_citations_observed",
        "mentions_without_attribution_observed",
        "evidence_reference_cause_counts",
    ):
        assert key not in metrics


