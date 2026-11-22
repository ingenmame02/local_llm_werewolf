"""TASK-056の回帰テスト（LL不使用）: Day内イベントの時系列明示と未来参照の観測。

実ログ game_15b1bdf65773 では、Day 1 の24発言が `[Day 1 / Speech n / Player k /
In-game speech]`（＝その日の連番だけ）で並び、議論ターン T1/T2/T3 と推順、
台帳が使う `[speech:N]` との対応が失われていた。圧縮や要約を経ると「自分が先に
言ったのか、相手が後に言ったのか」の境界が薄れ、未来の投票結果を根拠のように
扱う余地が残る。本テストは以下を固定する（実機LLMは使わない）。

1. 発言ヘッダは時点つき（`[D1 T2 s14 Player 2]`）で従来形より短い（コンテキスト節約）。
2. 時系列欄 `[DAY_TIMELINE]` は5行・460字以内で、時点・ID・未発生イベントだけを出す。
   口調・論点・結論を決める文言は含まない（議論の自由を縛らない）。
3. 発言直前の知識境界（snapshot）は自分より後の発言者・今日の投票・今夜の夜行動を
   「まだ起きていないこと」として分け、既知の先行発言と区別する。
4. 台帳は登順・時点つきで、同じ時点と同一内容の重複登録をしない。公開情報だけを載せ、
   エンジン発行の非公開情報（占い結果・役職・襲撃対象）は載せない。
5. 未来参照の検出は観測専用：本文は一字一句そのまま（`body_modified=false`）、婉曲表現と
   疑問文は数えない。根拠追跡率 `evidence_grounding_rate_at_speech_time` を算出する。
6. 仮説台帳へ観測時点（observed DdTt#N）を刻み、古い要約を新しい事実として扱わない。
7. `timeline_policy="off"` では時系列欄だけ消える（A/B比較用の旧挙動）。ヘッダ短縮は残る。
"""
import json
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.day_timeline import (
    DAY_TIMELINE_MAX_CHARS,
    DAY_TIMELINE_MAX_LINES,
    DAY_TIMELINE_POLICY_DEFAULT,
    DAY_TIMELINE_POLICY_EXPLICIT,
    DAY_TIMELINE_POLICY_OFF,
    EVIDENCE_HEADER_LEGEND,
    FUTURE_DEATH_BEFORE_DEATH,
    FUTURE_NIGHT_ACTION,
    FUTURE_REFERENCE_CATEGORIES,
    FUTURE_SPEECH_FROM_UNSPOKEN,
    FUTURE_VOTE_OUTCOME,
    KIND_ATTACK_DEATH,
    KIND_DAY_START,
    KIND_EXECUTION,
    KIND_SPEECH,
    KIND_VOTE_PHASE,
    PublicTimeline,
    TimelineEvent,
    TurnSnapshot,
    build_turn_snapshot,
    day_timeline_policy_label,
    inspect_public_body,
    normalize_day_timeline_policy,
    render_timeline_section,
    timeline_evidence_header,
)
from werewolf.engine import GameEngine
from werewolf.hypothesis_ledger import STANCE_SUSPECT
from werewolf.models import GameState, Message, Phase
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game

LEGACY_HEADER_SHAPE = "[Day 1 / Speech 3 / Player 2 / In-game speech]"


def _state(response_language: str = "日本語") -> GameState:
    return create_game(
        seed=1001, config_path="config/players.json", response_language=response_language
    )


def _player(state: GameState, name: str):
    return next(player for player in state.players if player.name == name)


def _names(state: GameState) -> tuple[str, ...]:
    return tuple(player.name for player in state.players)


def _start_day1(state: GameState, turn: int = 1) -> None:
    """日の開始と議論ターンを再現する（engine の run_discussion と同じ形にする）。"""
    state.public_timeline.record_day_start(state.day)
    state.conversation_log.append(Message("system", f"Day {state.day} begins."))
    state.phase = Phase.DAY_DISCUSSION
    state.discussion_turn = turn
    state.begin_discussion_turn()


def _say(state: GameState, name: str, content: str) -> Message:
    return state.append_discussion_message(Message(name, content))


def _snapshot(state: GameState, actor: str) -> TurnSnapshot:
    return build_turn_snapshot(state, actor)


# ---------------------------------------------------------------------------
# 1. 発言ヘッダの短縮（新しい欄の増加を相殺する側の節約）
# ---------------------------------------------------------------------------
def test_stamped_header_is_shorter_than_legacy_and_keeps_the_same_ids() -> None:
    compact = timeline_evidence_header(1, 2, 14, "Player 2")
    legacy = timeline_evidence_header(1, None, None, "Player 2", legacy_index=3)
    assert compact == "[D1 T2 s14 Player 2]"
    assert legacy == LEGACY_HEADER_SHAPE
    # 1行あたり20字以上の削減（Day 1 の24行で約570字、新規欄の上限相当を相殺できる）
    assert len(legacy) - len(compact) >= 20
    # s14 は仮説台帳・論点台帳が使う [speech:14] と同じ番号（本文再掲なしで参照できる）
    assert "s14" in compact
    # 情報が欠けた旧ログでは従来形へ戻す（誤った時点を作らない）
    unstamped = timeline_evidence_header(1, 0, None, "Player 2")
    assert unstamped == "[Day 1 / Player 2 / In-game speech]"
    assert unstamped != compact


def test_prompt_uses_stamped_headers_and_reports_the_style() -> None:
    state = _state()
    _start_day1(state, turn=2)
    for index, name in enumerate(_names(state)[:6]):
        _say(state, name, f"{name}の{index + 1}番目の発言です。投票先を確認したい。")
    actor = _player(state, "Player 8")
    ctx = PromptBuilder.build_discussion_prompt(
        actor,
        tuple(state.alive_players),
        tuple(state.conversation_log),
        day=1,
        state=state,
        include_claim=False,
        context_strategy="full_transcript",
        question_hint_mode="none",
    )
    diagnostics = ctx.diagnostics
    assert diagnostics["evidence_header_style"] == "stamped_compact"
    assert int(diagnostics["evidence_stamped_headers"]) == 6
    assert EVIDENCE_HEADER_LEGEND in ctx.user_prompt
    # 時点つきヘッダが6行すべてに出る（従来形の断片は混ざらない）
    assert ctx.user_prompt.count("[D1 T2 s") >= 6
    assert "/ In-game speech]" not in ctx.user_prompt


def test_legacy_history_falls_back_to_the_old_header_without_ids() -> None:
    state = _state()
    history = (
        Message("system", "Day 1 begins."),
        Message("Player 2", "投票先を確認したい。"),
        Message("Player 3", "私はまだ情報を集めている。"),
    )
    actor = _player(state, "Player 4")
    ctx = PromptBuilder.build_discussion_prompt(
        actor,
        tuple(state.alive_players),
        history,
        day=1,
        state=state,
        include_claim=False,
        context_strategy="full_transcript",
        question_hint_mode="none",
    )
    assert ctx.diagnostics["evidence_header_style"] == "legacy"
    assert int(ctx.diagnostics["evidence_stamped_headers"]) == 0
    assert "[D1 T1 s" not in ctx.user_prompt
    assert "/ In-game speech]" in ctx.user_prompt


# ---------------------------------------------------------------------------
# 2. 発言の時点スタンプと台帳登録（エンジン側の一貫性）
# ---------------------------------------------------------------------------
def test_public_speech_gets_a_time_stamp_and_is_registered_in_order() -> None:
    state = _state()
    _start_day1(state)
    first = _say(state, "Player 1", "まずは誰を調べたいか聞きたい。")
    second = _say(state, "Player 2", "Player 1の視点を確認したい。")
    assert (first.day, first.turn, first.speech_id) == (1, 1, 1)
    assert (second.day, second.turn, second.speech_id) == (1, 1, 2)
    # 台帳の登順（日開始 -> 発言1 -> 発言2）と available_at（情報が届いた時点）
    kinds = [event.kind for event in state.public_timeline.events]
    assert kinds == [KIND_DAY_START, KIND_SPEECH, KIND_SPEECH]
    points = [event.available_at for event in state.public_timeline.events]
    assert points == ["D1T0#1", "D1T1#2", "D1T1#3"]
    assert state.public_timeline.events[1].speech_id == 1
    assert state.public_timeline.events[1].evidence_id == "[speech:1]"
    # ターン横断でもグローバルIDは連続する（日毎の連番で失われていた対応関係）
    state.discussion_turn = 2
    third = _say(state, "Player 3", "昨夜の出来事を整理したい。")
    assert (third.turn, third.speech_id) == (2, 3)
    assert third.timeline_index == 4


def test_speech_summary_is_short_and_bodies_are_not_copied_into_the_ledger() -> None:
    state = _state()
    _start_day1(state)
    body = "とても長い発言 " * 40
    _say(state, "Player 1", body)
    event = next(e for e in state.public_timeline.events if e.kind == KIND_SPEECH)
    assert len(event.summary) <= 46
    serialized_text = json.dumps(
        state.public_timeline.serialize(), ensure_ascii=False
    )
    assert serialized_text.count(body) == 0
    assert len(serialized_text) < len(body) + 900
    assert state.public_timeline.serialize()["metrics"]["timeline_speeches_recorded"] == 1


def test_ledger_skips_duplicates_and_rejects_unknown_kinds() -> None:
    ledger = PublicTimeline()
    first = ledger.record(day=2, turn=0, kind=KIND_VOTE_PHASE, summary="Vote: 5 ballots")
    assert first is not None and first.sequence == 1
    # 同じ時点・同一内容の再登録は登順をずらさない（投票ログの再生成でも安全）
    assert (
        ledger.record(day=2, turn=0, kind=KIND_VOTE_PHASE, summary="Vote: 5 ballots")
        is None
    )
    assert ledger.next_sequence == 2
    try:
        ledger.record(day=2, turn=0, kind="night_attack_target", summary="target")
    except ValueError as error:
        assert "Unknown timeline event kind" in str(error)
    else:  # pragma: no cover - 常に例外になる
        raise AssertionError("未知のイベント種別は拒否する")


def test_deaths_and_vote_are_recorded_as_public_time_points() -> None:
    state = _state()
    _start_day1(state)
    _say(state, "Player 1", "今日は Player 2 を確認したい。")
    state.public_timeline.record_vote_phase(
        1, {"Player 1": "Player 2"}, ballot_id="vote-1-initial"
    )
    state.public_timeline.record_death(1, name="Player 2", executed=True)
    kinds = [event.kind for event in state.public_timeline.events]
    assert kinds[-2:] == [KIND_VOTE_PHASE, KIND_EXECUTION]
    assert state.public_timeline.events[-2].evidence_id == "vote-1-initial"
    # 台帳は登順を壊さない（投票より後の出来事を前に並べない）
    sequences = [event.sequence for event in state.public_timeline.events]
    assert sequences == sorted(sequences)


def test_ledger_serialization_carries_only_public_information() -> None:
    state = _state()
    _start_day1(state)
    _say(state, "Player 1", "誰を疑うか話したい。")
    # エンジン発行の非公開情報（占い結果・襲撃対象）を台帳へ入れない
    state.seer_results["Player 3"] = "wolf"
    serialized_text = json.dumps(state.public_timeline.serialize(), ensure_ascii=False)
    assert "wolf" not in serialized_text
    assert "attack_target" not in serialized_text
    kinds = {event["kind"] for event in state.public_timeline.serialize()["events"]}
    assert kinds <= {KIND_DAY_START, KIND_SPEECH}


# ---------------------------------------------------------------------------
# 3. 発言直前の知識境界（既知の先行発言と未発生イベントの分離）
# ---------------------------------------------------------------------------
def test_snapshot_splits_known_prior_events_from_events_that_have_not_happened() -> None:
    state = _state()
    _start_day1(state, turn=2)
    _say(state, "Player 1", "T2の最初の発言。")
    _say(state, "Player 2", "T2の二番目の発言。")
    snapshot = _snapshot(state, "Player 4")
    # 自分のターン内の先行発言（自分が語る前に既知だったものだけ）
    assert [name for name, _ in snapshot.prior_this_turn] == ["Player 1", "Player 2"]
    assert snapshot.point() == f"D1T2#{snapshot.sequence}"
    # 自分より後に発言するプレイヤーは未知（この時点で引用できない）
    assert "Player 4" not in snapshot.speakers_pending
    assert "Player 5" in snapshot.speakers_pending
    assert snapshot.spoken_before("Player 1") is True
    assert snapshot.spoken_before("Player 5") is False
    # 今日の投票と今夜の夜行動は、この時点では存在しない
    assert snapshot.today_vote_done is False
    assert snapshot.night_pending is True
    # 自分の本日発言（ターンと [speech:N]）も区別して持つ（一貫性の基準点）
    earlier_own = _say(state, "Player 4", "T1で私は情報を集めていた。")
    after = _snapshot(state, "Player 4")
    assert (earlier_own.turn, earlier_own.speech_id) in after.own_speeches_today
    log_fields = snapshot.to_log_fields()
    assert log_fields["timeline_point"] == snapshot.point()
    assert log_fields["pending_speakers"] == list(snapshot.speakers_pending)
    # ログへ本文・非公開情報を出さない
    assert "content" not in log_fields
    assert "wolf" not in json.dumps(log_fields, ensure_ascii=False, default=str)


def test_snapshot_marks_the_vote_and_deaths_as_known_once_they_happen() -> None:
    state = _state()
    _start_day1(state)
    _say(state, "Player 1", "今日は Player 2 を見たい。")
    # まだ起きていない状態（投票前・死亡前）
    before = _snapshot(state, "Player 3")
    assert before.today_vote_done is False
    assert before.was_alive("Player 2") is True
    # 投票が成立し、処刑と襲撃死が公開された後の時点
    state.voting_completed_day = state.day
    state.public_timeline.record_vote_phase(1, {"Player 1": "Player 2"})
    state.public_timeline.record_death(1, name="Player 2", executed=True)
    state.mark_dead("Player 2")
    state.executions.append((1, "Player 2"))
    state.attack_deaths.append((2, "Player 5"))
    after = _snapshot(state, "Player 3")
    assert after.today_vote_done is True
    assert after.was_alive("Player 2") is False
    assert ("Player 2", 1) in after.known_dead
    assert ("Player 5", 2) in after.known_dead
    # 死亡者はこのターンで発言予定に入らない（未来の発言者として扱わない）
    assert "Player 2" not in after.speakers_pending
    assert "Player 5" in after.alive_players


def test_snapshot_restores_prior_speeches_from_stamped_history_when_ledger_is_empty() -> None:
    """台帳が未構築でも、Message のスタンプだけでその日の境界を作れる。"""
    state = GameState(players=_state().players)
    state.phase = Phase.DAY_DISCUSSION
    state.day = 1
    state.discussion_turn = 2
    state.conversation_log.extend(
        [
            Message("system", "Day 1 begins."),
            Message("Player 1", "T1の発言。", day=1, turn=1, speech_id=1),
            Message("Player 2", "T2の先行発言。", day=1, turn=2, speech_id=2),
        ]
    )
    snapshot = build_turn_snapshot(state, "Player 4")
    assert [name for name, _ in snapshot.prior_this_turn] == ["Player 2"]
    assert snapshot.next_speech_id == 3
    assert snapshot.point().startswith("D1T2#")
    # 別日の発言（この日の既知情報ではない）は先行発言として数えない
    assert ("Player 3", 3) not in snapshot.prior_this_turn


# ---------------------------------------------------------------------------
# 4. 時系列欄（コンテキスト節約と中立性の両立）
# ---------------------------------------------------------------------------
def test_timeline_section_is_bounded_and_sets_no_topic_or_tone() -> None:
    state = _state()
    _start_day1(state, turn=2)
    for name in _names(state)[:5]:
        _say(state, name, f"{name}が先に発言した。投票先は保留したい。")
    block, diagnostics = render_timeline_section(
        _snapshot(state, "Player 7"), language="日本語"
    )
    lines = [line for line in block.splitlines() if line.strip()]
    assert lines[0] == "[DAY_TIMELINE]"
    assert len(lines) <= DAY_TIMELINE_MAX_LINES
    assert len(block) <= DAY_TIMELINE_MAX_CHARS + 2
    # 論点・口調・結論・推理順を決めないことを明記する（議論の自由を縛らない）
    assert "論点・口調・結論や推理の順番は決めません" in block
    # 時点と ID だけを出す（発言本文は再掲しない）
    assert "投票先は保留したい" not in block
    assert "[s" in block and "[speech:" in block
    # 未発生イベント（後発プレイヤー・今日の投票・今夜の夜行動）を明示する
    assert "まだ起きていないこと" in block
    assert "今日の投票" in block and "今夜の夜行動" in block
    assert int(diagnostics["day_timeline_chars"]) == len(block)
    assert int(diagnostics["day_timeline_lines"]) == len(lines)
    assert int(diagnostics["timeline_prior_events_this_turn"]) == 5


def test_timeline_section_english_and_budget_trim_keep_the_point_and_ids() -> None:
    state = _state(response_language="English")
    _start_day1(state, turn=3)
    for name in _names(state)[:7]:
        _say(state, name, "I would like to compare our reading of the votes later.")
    snapshot = _snapshot(state, "Player 8")
    block, diagnostics = render_timeline_section(snapshot, language="English")
    assert "[DAY_TIMELINE]" in block
    assert "sets no topic, tone, conclusion, or reasoning order" in block
    assert "Not yet occurred" in block
    assert len(block) <= DAY_TIMELINE_MAX_CHARS + 2
    assert (
        int(diagnostics["day_timeline_lines"]) <= DAY_TIMELINE_MAX_LINES
    ), "予算超過時は説明を短くするだけで、時点と未発生イベントは落とさない"
    # 節約のため説明文を短縮した版が選ばれても、時点・ID・先行発言数は残る
    assert int(diagnostics["timeline_render_variant"]) >= 1
    assert snapshot.point() in block
    assert "[speech:" in block
    # 一覧は上限件数＋超過件数で打ち切られる（全員分を書いて予算を膨らませない）
    assert f"(+{len(snapshot.prior_this_turn) - 4})" in block
    # 表示される先行発言は4名まで（[speech:N] の自分枠とは別カウント）
    displayed = re.findall(r"Player \d+ \[s\d+\]", block)
    assert len(displayed) <= 4
    assert len(displayed) >= 1



def test_timeline_section_survives_a_turn_with_no_prior_speech() -> None:
    state = _state()
    _start_day1(state, turn=1)
    snapshot = _snapshot(state, "Player 1")
    block, diagnostics = render_timeline_section(snapshot, language="日本語")
    assert "このターンの先行発言: なし" in block
    assert diagnostics["timeline_next_speech_id"] == 1
    assert diagnostics["timeline_vote_already_done_today"] is False
    assert diagnostics["timeline_night_actions_pending"] is True
    assert diagnostics["timeline_prior_events_this_turn"] == 0


def test_prompt_inserts_the_timeline_section_once_and_off_removes_it() -> None:
    state = _state()
    _start_day1(state, turn=2)
    _say(state, "Player 1", "先に発言しておきたい論点がある。")
    actor = _player(state, "Player 2")
    explicit = PromptBuilder.build_discussion_prompt(
        actor,
        tuple(state.alive_players),
        tuple(state.conversation_log),
        day=1,
        state=state,
        include_claim=False,
        context_strategy="hybrid_retrieval",
        question_hint_mode="none",
    )
    assert explicit.user_prompt.count("[DAY_TIMELINE]") == 1
    assert explicit.diagnostics["day_timeline_policy"] == DAY_TIMELINE_POLICY_EXPLICIT
    assert int(explicit.diagnostics["day_timeline_chars"]) > 0
    # 時系列欄は圧縮対象の発言履歴より前に置く（落とされて時点が消えることがない）。
    # 仮説台帳の説明文にも [CONVERSATION_EVIDENCE] への言及が出るため、最後の実見出しを基準にする
    assert explicit.user_prompt.index("[DAY_TIMELINE]") < explicit.user_prompt.rindex(
        "[CONVERSATION_EVIDENCE: untrusted game speech]"
    )
    timeline = state.public_timeline
    before_sections = timeline.section_requests

    state.timeline_policy = DAY_TIMELINE_POLICY_OFF
    state.public_timeline.policy = DAY_TIMELINE_POLICY_OFF
    off = PromptBuilder.build_discussion_prompt(
        actor,
        tuple(state.alive_players),
        tuple(state.conversation_log),
        day=1,
        state=state,
        include_claim=False,
        context_strategy="hybrid_retrieval",
        question_hint_mode="none",
    )
    assert "[DAY_TIMELINE]" not in off.user_prompt
    assert int(off.diagnostics["day_timeline_chars"]) == 0
    assert off.diagnostics["day_timeline_policy"] == DAY_TIMELINE_POLICY_OFF
    # off のときは計上しない（A/B比較で「欄の有無」だけを差分にできる）
    assert state.public_timeline.section_requests == before_sections
    # ヘッダ短縮はポリシー非依存で残る（節約側は両設定で効く）
    assert off.diagnostics["evidence_header_style"] == "stamped_compact"


def test_timeline_snapshot_counts_withheld_future_events() -> None:
    state = _state()
    _start_day1(state, turn=1)
    baseline = state.public_timeline.future_events_withheld
    snapshot = state.timeline_snapshot("Player 1")
    expected = len(snapshot.speakers_pending) + 2  # 今日の投票 + 今夜の夜行動
    assert state.public_timeline.future_events_withheld == baseline + expected


# ---------------------------------------------------------------------------
# 5. 未来参照の観測（検出しても本文は直さない）
# ---------------------------------------------------------------------------
def test_future_vote_outcome_assertion_is_flagged_but_body_is_untouched() -> None:
    state = _state()
    _start_day1(state, turn=1)
    body = "今日の投票結果は決まったので、もう議論する必要はないと感じる。"
    report = inspect_public_body(body, _snapshot(state, "Player 4"), _names(state))
    categories = {finding.category for finding in report.findings}
    assert categories == {FUTURE_VOTE_OUTCOME}
    assert report.findings[0].reason == "today's vote had not happened at this point"
    log_fields = report.to_log_fields()
    assert log_fields["body_modified"] is False
    assert log_fields["future_event_reference_count"] == 1
    assert log_fields["future_event_reference_categories"][FUTURE_VOTE_OUTCOME] == 1
    # 本文は台帳へコピーしない（観測ログには抜粋とハッシュだけ残す）
    finding = log_fields["future_event_reference_findings"][0]
    assert "sentence_sha256" in finding and len(finding["sentence_excerpt"]) <= 80


def test_questions_and_hedges_are_not_counted_as_future_references() -> None:
    state = _state()
    _start_day1(state, turn=1)
    snapshot = _snapshot(state, "Player 4")
    hedged = (
        "今日の投票結果はまだわからないので、もし決まるならそれを踏まえたい。"
        "今夜誰かが襲われるかもしれないし、Player 2が死亡した可能性も残る。"
    )
    report = inspect_public_body(hedged, snapshot, _names(state))
    assert report.findings == ()
    question = "今日の投票結果はどうなるのだろう？ Player 2はもう死亡したのだろうか？"
    assert inspect_public_body(question, snapshot, _names(state)).findings == ()
    assert report.grounding_rate == 1.0


def test_alive_player_death_assertion_is_flagged_until_the_death_happens() -> None:
    state = _state()
    _start_day1(state, turn=1)
    body = "Player 7はすでに死亡した。だから議論は打ち切っていい。"
    flagged = inspect_public_body(body, _snapshot(state, "Player 4"), _names(state))
    categories = {finding.category for finding in flagged.findings}
    assert categories == {FUTURE_DEATH_BEFORE_DEATH}
    assert any("Player 7" in str(finding.names) for finding in flagged.findings)
    # 死亡が公開された後なら同じ文は数えない（時点が変われば判定も変わる）
    state.public_timeline.record_death(2, name="Player 7", executed=False)
    state.mark_dead("Player 7")
    state.attack_deaths.append((2, "Player 7"))
    after = inspect_public_body(body, _snapshot(state, "Player 4"), _names(state))
    assert {finding.category for finding in after.findings} == set()


def test_attributed_speech_from_a_player_who_has_not_spoken_yet_is_flagged() -> None:
    state = _state()
    _start_day1(state, turn=3)
    _say(state, "Player 1", "T3で先に発言する。")
    snapshot = build_turn_snapshot(state, "Player 2")
    cited = "Player 1の発言を受けて、私は Player 5 を疑っている。"
    ok_report = inspect_public_body(cited, snapshot, _names(state))
    assert ok_report.findings == ()
    assert ok_report.attributed_references >= 1
    assert ok_report.grounding_rate == 1.0
    # Player 6 はまだ一度も発言していない（この時点で引用できる証拠が無い）
    unspoken = "Player 6の発言を聞いたが、あの人は村人だろう。"
    report = inspect_public_body(unspoken, snapshot, _names(state))
    assert [f.category for f in report.findings] == [FUTURE_SPEECH_FROM_UNSPOKEN]
    assert report.unresolvable_references == 1
    assert report.attributed_references == 1
    # その1件だけを追跡できなかった場合の根拠追跡率は 0（本文はそのまま）
    assert report.grounding_rate == 0.0
    assert report.to_log_fields()["body_modified"] is False
    # 追跡できる参照とできない参照が混ざれば中間の率になる（観測値としての割合）
    mixed = "Player 1の発言を聞いた。Player 6の発言も村人の根拠になるはずだ。"
    mixed_report = inspect_public_body(mixed, snapshot, _names(state))
    assert mixed_report.attributed_references == 2
    assert mixed_report.unresolvable_references == 1
    assert mixed_report.grounding_rate == 0.5


def test_night_action_assertion_before_the_night_is_flagged() -> None:
    state = _state()
    _start_day1(state, turn=1)
    body = "今夜、Player 3が襲われたことが確定している。"
    report = inspect_public_body(body, _snapshot(state, "Player 4"), _names(state))
    categories = {f.category for f in report.findings}
    assert categories == {FUTURE_NIGHT_ACTION}
    # 夜が明けて死亡が公開された時点なら同じ文を疑わない
    state.public_timeline.record_death(2, name="Player 3", executed=False)
    state.mark_dead("Player 3")
    state.attack_deaths.append((2, "Player 3"))
    state.day = 2
    later = inspect_public_body(body, build_turn_snapshot(state, "Player 4"), _names(state))
    assert FUTURE_NIGHT_ACTION not in {f.category for f in later.findings}


def test_empty_bodies_produce_a_neutral_report() -> None:
    state = _state()
    _start_day1(state)
    snapshot = _snapshot(state, "Player 4")
    for body in (None, "", "   \n  "):
        report = inspect_public_body(body, snapshot, _names(state))
        assert report.findings == ()
        assert report.attributed_references == 0
        assert report.grounding_rate == 1.0
        assert report.to_log_fields()["body_modified"] is False


def test_grounding_rate_counts_only_citations_and_never_drops_below_zero() -> None:
    ledger = PublicTimeline()
    ledger.attributed_references = 8
    ledger.unresolvable_references = 2
    metrics = ledger.metrics()
    assert metrics["evidence_grounding_rate_at_speech_time"] == 0.75
    assert metrics["future_event_reference_rate"] == 0.0
    empty = PublicTimeline().metrics()
    assert empty["evidence_grounding_rate_at_speech_time"] == 1.0
    assert empty["timeline_event_count"] == 0


# ---------------------------------------------------------------------------
# 6. エンジン統合（観測のための検査で本文と発言の自由を変えない）
# ---------------------------------------------------------------------------
class _NoActions:
    def wolf_discuss(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("このテストでは夜行動を実行しない")


def _engine_with_events(state: GameState) -> tuple[GameEngine, list[dict[str, object]]]:
    events: list[dict[str, object]] = []
    engine = GameEngine(
        state, _NoActions(), seed=state.seed, event_callback=events.append
    )
    return engine, events


def test_engine_publishes_the_body_unchanged_and_reports_the_observation() -> None:
    state = _state()
    _start_day1(state, turn=1)
    engine, events = _engine_with_events(state)
    body = "今日の投票結果は決まった。だから Player 2 を疑いたい。"
    engine._append_discussion_message(Message("Player 4", body))
    speech_events = [event for event in events if event["type"] == "discussion_message"]
    assert len(speech_events) == 1
    event = speech_events[0]
    # 公開された本文は一字一句そのまま（未来参照を検出しても語彙・口調・結論を変えない）
    assert event["content"] == body
    assert state.conversation_log[-1].content == body
    assert event["body_modified"] is False
    assert int(event["future_event_reference_count"]) >= 1
    assert event["timeline_observed"] is True
    assert event["speech_id"] == 1
    assert str(event["timeline_point"]).startswith("D1T1#")
    assert event["discussion_turn"] == 1
    # 検査による再生成要求や本文の自動修正イベントは起きていない
    assert not [event for event in events if "retry" in str(event.get("type"))]
    timeline = state.public_timeline
    assert timeline.bodies_checked == 1
    assert timeline.bodies_kept_unmodified == 1
    assert timeline.reference_checks == 1


def test_engine_off_policy_skips_the_inspection_but_still_stamps() -> None:
    state = _state()
    _start_day1(state, turn=1)
    state.timeline_policy = DAY_TIMELINE_POLICY_OFF
    state.public_timeline.policy = DAY_TIMELINE_POLICY_OFF
    engine, events = _engine_with_events(state)
    engine._append_discussion_message(Message("Player 4", "今日の投票結果は決まった。"))
    event = next(e for e in events if e["type"] == "discussion_message")
    assert event["timeline_observed"] is False
    assert "future_event_reference_count" not in event
    # スタンプと登順はポリシー非依存（旧挙動との差分は欄の有無だけにする）
    assert event["speech_id"] == 1
    assert state.public_timeline.events[-1].kind == KIND_SPEECH
    assert state.public_timeline.reference_checks == 0


def test_system_messages_are_not_inspected_or_registered() -> None:
    state = _state()
    _start_day1(state)
    engine, events = _engine_with_events(state)
    engine._append_discussion_message(Message("system", "Player 9 was voted out."))
    event = next(e for e in events if e["type"] == "discussion_message")
    assert "timeline_observed" not in event
    assert "speech_id" not in event
    assert state.conversation_log[-1].speaker == "system"
    assert state.public_timeline.reference_checks == 0


# ---------------------------------------------------------------------------
# 7. 仮説台帳の観測時点（古い要約を新しい事実として扱わないための補助）
# ---------------------------------------------------------------------------
def test_hypotheses_carry_the_point_when_they_were_raised() -> None:
    state = _state()
    _start_day1(state, turn=2)
    _say(state, "Player 1", "T2の先行発言。")
    speech = "Player 3さんは人狼だと思います。"
    created = state.record_hypotheses_from_speech(
        "Player 2", speech, day=state.day, turn=state.discussion_turn
    )
    assert created == 1
    record = state.hypothesis_ledger.records["H-Player 2-Player 3-suspect"]
    assert record.stance == STANCE_SUSPECT
    assert record.observed_at == f"D{state.day}T2#{state.public_timeline.next_sequence}"
    rendered, _ = state.render_hypotheses_for("Player 2")
    assert "observed " in rendered
    assert "older line is never a new fact" in rendered
    assert (
        state.hypothesis_ledger.metrics()["hypothesis_records_with_observed_time"] == 1
    )
    serialized = state.hypothesis_ledger.serialize_public()
    assert serialized["records"][0]["observed_at"] == record.observed_at


def test_hypothesis_record_falls_back_to_day_and_turn_without_explicit_point() -> None:
    ledger = _state().hypothesis_ledger
    ledger.record(
        subject="Player 4",
        actor="Player 2",
        day=2,
        turn=3,
        stance=STANCE_SUSPECT,
        statement="Player 4の発言が揺れている",
        evidence_ids=["[speech:9]"],
    )
    record = ledger.records["H-Player 2-Player 4-suspect"]
    assert record.observed_at == "D2T3"


# ---------------------------------------------------------------------------
# 8. ルール設定の正規化とA/B比較（既定は明示）
# ---------------------------------------------------------------------------
def test_policy_normalization_and_labels() -> None:
    assert DAY_TIMELINE_POLICY_DEFAULT == DAY_TIMELINE_POLICY_EXPLICIT
    assert normalize_day_timeline_policy("off") == DAY_TIMELINE_POLICY_OFF
    assert normalize_day_timeline_policy(" EXPLICIT ") == DAY_TIMELINE_POLICY_EXPLICIT
    assert normalize_day_timeline_policy(True) == DAY_TIMELINE_POLICY_EXPLICIT
    assert normalize_day_timeline_policy(False) == DAY_TIMELINE_POLICY_OFF
    assert normalize_day_timeline_policy("legacy") == DAY_TIMELINE_POLICY_OFF
    assert normalize_day_timeline_policy(None) == DAY_TIMELINE_POLICY_DEFAULT
    assert normalize_day_timeline_policy(object()) == DAY_TIMELINE_POLICY_DEFAULT
    assert day_timeline_policy_label("off") == "off"


def test_game_state_keeps_ledger_policy_in_sync_with_its_own_setting() -> None:
    state = GameState(players=_state().players, timeline_policy="off")
    assert state.timeline_policy == DAY_TIMELINE_POLICY_OFF
    assert state.public_timeline.policy == DAY_TIMELINE_POLICY_OFF
    default_state = GameState(players=_state().players)
    assert default_state.timeline_policy == DAY_TIMELINE_POLICY_EXPLICIT
    assert default_state.public_timeline.policy == DAY_TIMELINE_POLICY_EXPLICIT


def test_message_stamps_have_neutral_defaults_for_old_logs() -> None:
    plain = Message("Player 1", "旧的なログの発言")
    assert (plain.day, plain.turn, plain.speech_id, plain.timeline_index) == (
        0,
        0,
        None,
        None,
    )
    # キーワード指定で旧ログから復元しても壊れない（Message(**row) 互換）
    restored = Message(speaker="Player 2", content="x", day=2, turn=3, speech_id=9)
    assert restored.speech_id == 9 and restored.turn == 3


def test_future_reference_categories_are_stable_for_log_consumers() -> None:
    assert FUTURE_REFERENCE_CATEGORIES == (
        FUTURE_VOTE_OUTCOME,
        FUTURE_DEATH_BEFORE_DEATH,
        FUTURE_SPEECH_FROM_UNSPOKEN,
        FUTURE_NIGHT_ACTION,
    )
    event = TimelineEvent(
        sequence=3, day=2, turn=1, kind=KIND_SPEECH, actor="Player 2", speech_id=7
    )
    assert event.available_at == "D2T1#3"
    assert event.short_point() == "Player 2 [s7]"
    payload = event.to_dict()
    assert payload["available_at"] == "D2T1#3"
    assert payload["timeline_sequence"] == 3
    vote = TimelineEvent(sequence=9, day=2, turn=0, kind=KIND_VOTE_PHASE)
    assert vote.short_point() == "vote"
    death = TimelineEvent(
        sequence=10, day=2, turn=0, kind=KIND_ATTACK_DEATH, actor="Player 5"
    )
    assert death.short_point() == "Player 5"






