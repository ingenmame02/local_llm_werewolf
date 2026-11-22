"""TASK-056: Day内イベントの時系列明示と未来参照の防止（観測のみ）。

現行のプロンプトは日・ターン・発言者が各所へ分散し、発言履歴には「その日何番目の
発言か」という連番しか残らない。そのため「この発言時点で既に知っていた情報」と
「同じ日の後続ターンで公開された情報」の境界が弱く、圧縮や要約を挟むと順序が落ちて
未来の投票結果を過去の根拠のように扱う可能性がある。

ここでは公開イベントだけを時系列台帳へ並べ、発言直前の知識境界（snapshot）と
まだ発生していないイベントを短い1ブロックだけ添える。

設計方針（本タスクの思想そのまま）:
- コンテキスト節約: 本文は再掲せず時点・ID・短い要約のみ。あわせて発言ヘッダを
  `[D1 T2 s14 Player 2]` 形へ縮約し、新しい節の増加分を既存の重複で相殺する。
- モデルの一貫性: ターン開始より後のイベントは候補にも欄にも出さない。仮説・論点
  台帳には観測時点を残し、古い要約が新しい事実として読み替えられないようにする。
- 合理的な議論: 順序と時点（因果の基準点）を与える。推理の順番は指示しない。
- 発言の自由: この欄は口調・論点・結論を決めない。未来参照の検出は観測だけで、
  公開本文の語彙・口調・結論は一切書き換えない（`body_modified=false` で記録）。

注意: この台帳は公開情報専用。エンジンが発行した非公開情報（占い結果・襲撃対象・
役職）は snapshot にも serialize() にも一切載せない。
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - 型注釈のみ（models との循環importを避ける）
    from .models import GameState, Message

# ルール設定（A/B比較用に明示的時系列を消せる）。既定は明示。
DAY_TIMELINE_POLICY_EXPLICIT = "explicit"
DAY_TIMELINE_POLICY_OFF = "off"
DAY_TIMELINE_POLICIES: tuple[str, ...] = (
    DAY_TIMELINE_POLICY_EXPLICIT,
    DAY_TIMELINE_POLICY_OFF,
)
DAY_TIMELINE_POLICY_DEFAULT = DAY_TIMELINE_POLICY_EXPLICIT

# コンテキスト予算（TASK-056完了条件「コンテキストを増やさず」の上限設計値）。
DAY_TIMELINE_MAX_LINES = 5
DAY_TIMELINE_MAX_CHARS = 460
EVENT_SUMMARY_MAX_CHARS = 46
PRIOR_EVENT_DISPLAY_MAX = 4

# イベント種別（公開されたものだけ。夜行動の結果など非公開情報は登録しない）
KIND_DAY_START = "day_start"
KIND_SPEECH = "speech"
KIND_VOTE_PHASE = "vote_phase"
KIND_EXECUTION = "execution"
KIND_ATTACK_DEATH = "attack_death"
TIMELINE_EVENT_KINDS: tuple[str, ...] = (
    KIND_DAY_START,
    KIND_SPEECH,
    KIND_VOTE_PHASE,
    KIND_EXECUTION,
    KIND_ATTACK_DEATH,
)

# 未来参照の検出カテゴリ（観測のみ・本文は修正しない）
FUTURE_VOTE_OUTCOME = "vote_outcome_before_vote"
FUTURE_DEATH_BEFORE_DEATH = "death_before_death"
FUTURE_SPEECH_FROM_UNSPOKEN = "speech_from_unspoken_player"
FUTURE_NIGHT_ACTION = "night_action_before_night"
FUTURE_REFERENCE_CATEGORIES: tuple[str, ...] = (
    FUTURE_VOTE_OUTCOME,
    FUTURE_DEATH_BEFORE_DEATH,
    FUTURE_SPEECH_FROM_UNSPOKEN,
    FUTURE_NIGHT_ACTION,
)


def normalize_day_timeline_policy(value: object) -> str:
    """設定値を \"explicit\" / \"off\" へ正規化する（不明値は既定の明示）。"""
    if isinstance(value, bool):
        return DAY_TIMELINE_POLICY_EXPLICIT if value else DAY_TIMELINE_POLICY_OFF
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in DAY_TIMELINE_POLICIES:
            return candidate
        if candidate in {"on", "true", "timeline"}:
            return DAY_TIMELINE_POLICY_EXPLICIT
        if candidate in {"none", "legacy", "false"}:
            return DAY_TIMELINE_POLICY_OFF
    return DAY_TIMELINE_POLICY_DEFAULT


def day_timeline_policy_label(policy: object) -> str:
    """ログ・CLI表示用のラベル（正規化済みの文字列を返す）。"""
    return normalize_day_timeline_policy(policy)


def timeline_evidence_header(
    day: int,
    turn: int | None,
    speech_id: int | None,
    speaker: str,
    legacy_index: int | None = None,
) -> str:
    """発言履歴1行の接頭辞を作る（TASK-056: 時点とグローバルIDを短く示す）。

    `turn` と `speech_id` が揃っているときだけ新形式（約21字）を使い、旧ログなど
    情報が欠けるときは従来形（`[Day 1 / Speech 3 / Player 2 / In-game speech]`）へ
    戻す。`sN` は推理メモ・論点台帳・仮説台帳が使う `[speech:N]` と同じ番号なので、
    本文を再掲せずに同じ証拠を指し示せる。
    """
    if turn and turn > 0 and speech_id is not None:
        return f"[D{day} T{turn} s{speech_id} {speaker}]"
    index_label = f" / Speech {legacy_index}" if legacy_index else ""
    return f"[Day {day}{index_label} / {speaker} / In-game speech]"


EVIDENCE_HEADER_LEGEND = (
    "Header D=day, T=discussion turn, sN=speech id (same numbering as [speech:N])."
)


@dataclass(frozen=True)
class TimelineEvent:
    """公開イベント1件。時点・ID・短い要約だけを持ち、発言本文は保持しない。"""

    sequence: int
    day: int
    turn: int
    kind: str
    actor: str | None = None
    speech_id: int | None = None
    summary: str = ""
    evidence_id: str | None = None

    @property
    def available_at(self) -> str:
        """この情報が全員へ届いた時点（D日Tターン#順序）。"""
        return f"D{self.day}T{self.turn}#{self.sequence}"

    def short_point(self) -> str:
        if self.speech_id is not None:
            return f"{self.actor or '?'} [s{self.speech_id}]"
        label = {
            KIND_VOTE_PHASE: "vote",
            KIND_EXECUTION: "execution",
            KIND_ATTACK_DEATH: "attack_death",
            KIND_DAY_START: "day start",
        }.get(self.kind, self.kind)
        return f"{self.actor or label}"

    def to_dict(self) -> dict[str, object]:
        return {
            "timeline_sequence": self.sequence,
            "available_at": self.available_at,
            "day": self.day,
            "discussion_turn": self.turn,
            "kind": self.kind,
            "actor": self.actor,
            "speech_id": self.speech_id,
            "summary": self.summary,
            "evidence_id": self.evidence_id,
        }


@dataclass(frozen=True)
class TurnSnapshot:
    """発言直前に「既に知っていたこと／まだ起きていないこと」を分けた境界点。"""

    day: int
    turn: int
    actor: str
    sequence: int
    prior_this_turn: tuple[tuple[str, int], ...]
    own_speeches_today: tuple[tuple[int, int], ...]
    prior_today_events: int
    speakers_pending: tuple[str, ...]
    spoken_players: tuple[str, ...]
    known_dead: tuple[tuple[str, int], ...]
    alive_players: tuple[str, ...]
    today_vote_done: bool
    night_pending: bool
    # この時点で次回発言に割り当てられる [speech:N] の番号（発言順の基準点）
    next_speech_id: int = 0

    def point(self) -> str:
        return f"D{self.day}T{self.turn}#{self.sequence}"

    def spoken_before(self, name: str) -> bool:
        return name in self.spoken_players

    def was_alive(self, name: str) -> bool:
        return name in self.alive_players

    def to_log_fields(self) -> dict[str, object]:
        """ログへ載せる最小集合（本文・非公開情報は含まない）。"""
        return {
            "timeline_point": self.point(),
            "day": self.day,
            "discussion_turn": self.turn,
            "prior_events_this_turn": len(self.prior_this_turn),
            "prior_events_today": self.prior_today_events,
            "pending_speakers": list(self.speakers_pending),
            "today_vote_done": self.today_vote_done,
            "night_actions_pending": self.night_pending,
        }


@dataclass
class PublicTimeline:
    """公開イベントの時系列台帳（発言本文・非公開情報は保持しない）。"""

    events: list[TimelineEvent] = field(default_factory=list)
    policy: str = DAY_TIMELINE_POLICY_DEFAULT
    # 観測カウンタ（中立。未来参照を検出しても本文は直さない）
    reference_checks: int = 0
    bodies_checked: int = 0
    bodies_kept_unmodified: int = 0
    future_reference_findings: int = 0
    attributed_references: int = 0
    unresolvable_references: int = 0
    future_events_withheld: int = 0
    section_chars_total: int = 0
    section_lines_total: int = 0
    section_requests: int = 0
    _seen_keys: set[tuple[object, ...]] = field(
        default_factory=set, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.policy = normalize_day_timeline_policy(self.policy)

    @property
    def next_sequence(self) -> int:
        return len(self.events) + 1

    def record(
        self,
        *,
        day: int,
        turn: int,
        kind: str,
        actor: str | None = None,
        summary: str = "",
        speech_id: int | None = None,
        evidence_id: str | None = None,
    ) -> TimelineEvent | None:
        """イベントを1件登録する。同じ時点・同一内容の重複登録はスキップする。"""
        if kind not in TIMELINE_EVENT_KINDS:
            raise ValueError(f"Unknown timeline event kind: {kind!r}")
        text = " ".join(str(summary or "").split())[:EVENT_SUMMARY_MAX_CHARS]
        key = (day, turn, kind, actor, text, speech_id, evidence_id)
        if key in self._seen_keys:
            return None
        self._seen_keys.add(key)
        event = TimelineEvent(
            sequence=self.next_sequence,
            day=day,
            turn=turn,
            kind=kind,
            actor=actor,
            speech_id=speech_id,
            summary=text,
            evidence_id=evidence_id,
        )
        self.events.append(event)
        return event

    def record_speech(
        self,
        *,
        day: int,
        turn: int,
        speaker: str,
        speech_id: int,
        content: str,
    ) -> TimelineEvent | None:
        """公開発言を登録する（本文は保持せず要約とIDだけ残す）。"""
        return self.record(
            day=day,
            turn=turn,
            kind=KIND_SPEECH,
            actor=speaker,
            summary=content,
            speech_id=speech_id,
            evidence_id=f"[speech:{speech_id}]",
        )

    def record_day_start(self, day: int) -> TimelineEvent | None:
        return self.record(day=day, turn=0, kind=KIND_DAY_START, summary="Day begins")

    def record_vote_phase(
        self,
        day: int,
        votes: dict[str, str],
        runoff: bool = False,
        ballot_id: str | None = None,
    ) -> TimelineEvent | None:
        label = "Runoff vote" if runoff else "Vote"
        return self.record(
            day=day,
            turn=0,
            kind=KIND_VOTE_PHASE,
            summary=f"{label}: {len(votes)} ballots cast",
            evidence_id=ballot_id,
        )

    def record_death(self, day: int, name: str, *, executed: bool) -> TimelineEvent | None:
        kind = KIND_EXECUTION if executed else KIND_ATTACK_DEATH
        verb = "voted out" if executed else "attacked at night"
        return self.record(day=day, turn=0, kind=kind, actor=name, summary=f"{name} {verb}")

    def on_day(self, day: int) -> tuple[TimelineEvent, ...]:
        return tuple(event for event in self.events if event.day == day)

    def speeches(self, day: int | None = None) -> tuple[TimelineEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.kind == KIND_SPEECH and (day is None or event.day == day)
        )

    def latest(self) -> TimelineEvent | None:
        return self.events[-1] if self.events else None

    def metrics(self) -> dict[str, object]:
        """公開ログ用のメトリクス（中立カウンタのみ・真偽や非公開情報なし）。"""
        kind_counts = {
            kind: sum(1 for event in self.events if event.kind == kind)
            for kind in TIMELINE_EVENT_KINDS
        }
        checks = max(1, self.reference_checks)
        attributed = max(1, self.attributed_references)
        return {
            "timeline_policy": self.policy,
            "timeline_event_count": len(self.events),
            "timeline_speeches_recorded": len(self.speeches()),
            "timeline_event_kind_counts": kind_counts,
            "timeline_reference_checks": self.reference_checks,
            "timeline_bodies_checked": self.bodies_checked,
            "timeline_bodies_kept_unmodified": self.bodies_kept_unmodified,
            "future_event_reference_findings": self.future_reference_findings,
            "future_event_reference_rate": round(
                self.future_reference_findings / checks, 4
            ),
            "attributed_references_observed": self.attributed_references,
            "unresolvable_references": self.unresolvable_references,
            # 発言時点の根拠追跡率（参照した証拠が時点で追跡できる割合）
            "evidence_grounding_rate_at_speech_time": round(
                1.0 - (self.unresolvable_references / attributed), 4
            ),
            "future_events_withheld": self.future_events_withheld,
            "day_timeline_chars_avg": round(
                self.section_chars_total / max(1, self.section_requests), 1
            ),
            "day_timeline_lines_avg": round(
                self.section_lines_total / max(1, self.section_requests), 2
            ),
        }

    def serialize(self) -> dict[str, object]:
        """ログ用の逐列表現（時点・ID・要約のみ。本文と非公開情報は含まれない）。"""
        return {
            "policy": self.policy,
            "events": [event.to_dict() for event in self.events],
            "metrics": self.metrics(),
        }


_JAPANESE_LANGUAGE_TOKENS = {"日本語", "japanese", "ja"}


def _is_japanese(language: object) -> bool:
    """応答言語が日本語かを判定する（prompts と同じ規則。循環import回避のため局所）。"""
    text = str(language or "").strip()
    return text == "日本語" or text.casefold() in _JAPANESE_LANGUAGE_TOKENS


def _death_day_map(state: object) -> dict[str, int]:
    """公開された死亡者の発生日（襲撃死→処刑の順で登録）を作る。"""
    days: dict[str, int] = {}
    for entry in getattr(state, "attack_deaths", ()) or ():
        try:
            event_day, name = entry
        except (TypeError, ValueError):
            continue
        days.setdefault(str(name), int(event_day))
    for entry in getattr(state, "executions", ()) or ():
        try:
            event_day, name = entry
        except (TypeError, ValueError):
            continue
        days.setdefault(str(name), int(event_day))
    return days


def _next_speech_id(state: object) -> int:
    """次回発言に割り当てられる [speech:N] の番号（非システム発言の数+1）。"""
    log = getattr(state, "conversation_log", ()) or ()
    return sum(1 for message in log if getattr(message, "speaker", "") != "system") + 1


def build_turn_snapshot(state: object, actor_name: str) -> TurnSnapshot:
    """発言直前の知識境界を作る（公開情報のみ・エンジン発行の非公開情報は載せない）。

    台帳が空の場合（LL不使用テストや旧ログのリプレイ）に備え、その日の発言だけを
    conversation_log のスタンプから復元して境界点に使う。
    """
    timeline = getattr(state, "public_timeline", None)
    if not isinstance(timeline, PublicTimeline):
        timeline = PublicTimeline()
    day = max(1, int(getattr(state, "day", 1)))
    turn = int(getattr(state, "discussion_turn", 0))
    players = tuple(getattr(state, "players", ()) or ())
    alive = tuple(str(player.name) for player in players if bool(player.is_alive))
    known_dead = tuple(
        sorted(_death_day_map(state).items(), key=lambda item: (item[1], item[0]))
    )
    phase_value = str(getattr(getattr(state, "phase", None), "value", "") or "")
    winner = getattr(state, "winner", None)
    today_events = tuple(event for event in timeline.events if event.day == day)
    today_speeches = tuple(event for event in today_events if event.kind == KIND_SPEECH)
    if today_speeches:
        all_speeches = timeline.speeches()
    else:
        # 台帳が未構築のときだけ conversation_log のスタンプから復元する
        restored: list[TimelineEvent] = []
        index = 0
        for message in getattr(state, "conversation_log", ()) or ():
            if getattr(message, "speaker", "system") == "system":
                continue
            index += 1
            if int(getattr(message, "day", 0) or 0) != day:
                continue
            restored.append(
                TimelineEvent(
                    sequence=index,
                    day=day,
                    turn=int(getattr(message, "turn", 0) or 0),
                    kind=KIND_SPEECH,
                    actor=str(getattr(message, "speaker", "")),
                    speech_id=int(getattr(message, "speech_id", 0) or 0) or index,
                )
            )
        today_speeches = tuple(restored)
        all_speeches = tuple(restored)
    prior_this_turn = tuple(
        (str(event.actor), int(event.speech_id or 0))
        for event in today_speeches
        if event.turn == turn and event.actor and event.actor != actor_name
    )
    own_speeches_today = tuple(
        (event.turn, int(event.speech_id or 0))
        for event in today_speeches
        if event.actor == actor_name
    )
    spoken_players: list[str] = []
    for event in all_speeches:
        if event.actor and event.actor not in spoken_players:
            spoken_players.append(event.actor)
    pending_speakers = tuple(
        name
        for name in alive
        if name != actor_name
        and not any(
            event.turn == turn and event.actor == name for event in today_speeches
        )
    )
    return TurnSnapshot(
        day=day,
        turn=turn,
        actor=actor_name,
        sequence=timeline.next_sequence,
        prior_this_turn=prior_this_turn,
        own_speeches_today=own_speeches_today,
        prior_today_events=len(today_events),
        speakers_pending=pending_speakers,
        spoken_players=tuple(spoken_players),
        known_dead=known_dead,
        alive_players=alive,
        today_vote_done=getattr(state, "voting_completed_day", None) == day,
        night_pending=winner is None and phase_value in {"day_discussion", "voting"},
        next_speech_id=_next_speech_id(state),
    )


def _event_short(event: TimelineEvent) -> str:
    if event.speech_id is not None:
        return f"{event.actor or '?'} [s{event.speech_id}]"
    return f"{event.actor or event.kind} ({event.kind})"


def render_timeline_section(
    snapshot: TurnSnapshot,
    *,
    language: str = "English",
) -> tuple[str, dict[str, object]]:
    """発言直前の時系列欄を作る（時点・IDのみ。推理順や結論は指示しない）。

    コンテキスト節約のため、説明文を段階的に短くするだけで「時点・先行発言・
    まだ起きていないこと」の3要素は必ず残す（行を落として情報が消える設計にしない）。
    """
    japanese = _is_japanese(language)
    prior_names = [
        f"{name} [s{speech_id}]" for name, speech_id in snapshot.prior_this_turn
    ]
    none_prior = "なし" if japanese else "none"
    own_parts = [
        f"T{turn} [s{speech_id}]" for turn, speech_id in snapshot.own_speeches_today
    ]
    # 一覧は上限件数＋「ほかN件」で打ち切る（全員分を書いて予算を膨らませない）
    display_prior = prior_names[:PRIOR_EVENT_DISPLAY_MAX]
    overflow_prior = len(prior_names) - len(display_prior)
    prior_join = ", ".join(display_prior) or none_prior
    if overflow_prior > 0:
        prior_join += (
            f" ほか{overflow_prior}件" if japanese else f" (+{overflow_prior})"
        )
    pending: list[str] = []
    if snapshot.speakers_pending:
        pending.append(
            f"他{len(snapshot.speakers_pending)}人が自分の後"
            if japanese
            else f"+{len(snapshot.speakers_pending)} speak after you"
        )
    if not snapshot.today_vote_done:
        pending.append("今日の投票" if japanese else "today's vote")
    if snapshot.night_pending:
        pending.append("今夜の夜行動" if japanese else "tonight's night actions")
    point = f"D{snapshot.day}T{snapshot.turn}#{snapshot.sequence}"
    next_speech = f"[speech:{snapshot.next_speech_id}]"
    if japanese:
        full_lines = [
            "[DAY_TIMELINE]",
            "- 時点とIDだけで「この時点で知れていた範囲」を示す参考欄です。論点・口調・結論や推理の順番は決めません。",
            (
                f"- 現在 {point}、あなたはこれから {next_speech} として発言。"
                f"このターンの先行発言: {prior_join}"
                f"（今日の公開イベント {snapshot.prior_today_events} 件）。"
            ),
        ]
        compact_lines = [
            "[DAY_TIMELINE]",
            "- 時点とIDのみ。論点・口調・結論や推理の順番は決めません。",
            (
                f"- {point}: 次の発言は {next_speech}。"
                f"先行発言 {prior_join}（本日 {snapshot.prior_today_events + 1} 件目）。"
            ),
        ]
        if own_parts:
            full_lines.append(
                f"- 今日の自分の発言: {', '.join(own_parts[-3:])}。"
                "この履歴と揃えるか、立場を変えるなら変えて構いません。"
            )
            compact_lines.append(f"- 自分の本日発言: {', '.join(own_parts[-2:])}。")
        if pending:
            full_lines.append(
                f"- まだ起きていないこと: {'、'.join(pending)}。"
                "これらの結果はまだ存在しないので、現時点では根拠にできません"
                "（推測すること自体は自由です）。"
            )
            compact_lines.append(
                f"- まだ起きていないこと: {'、'.join(pending)}（まだ根拠にできません）。"
            )
    else:
        full_lines = [
            "[DAY_TIMELINE]",
            (
                "- Order and IDs only, marking what was knowable at this point. "
                "This section sets no topic, tone, conclusion, or reasoning order."
            ),
            (
                f"- Now {point}: you speak next as {next_speech}. "
                f"Earlier this turn: {prior_join}"
                f" ({snapshot.prior_today_events} public event(s) known today)."
            ),
        ]
        compact_lines = [
            "[DAY_TIMELINE]",
            "- Order and IDs only; sets no topic, tone, conclusion, or reasoning order.",
            (
                f"- {point}: you speak next as {next_speech}; earlier this turn: "
                f"{prior_join} (event {snapshot.prior_today_events + 1} today)."
            ),
        ]
        if own_parts:
            full_lines.append(
                f"- Your own speeches today: {', '.join(own_parts[-3:])}. "
                "Stay aligned with them or change position openly - both are valid."
            )
            compact_lines.append(f"- Your speeches today: {', '.join(own_parts[-2:])}.")
        if pending:
            full_lines.append(
                f"- Not yet occurred: {', '.join(pending)}. Those outcomes do not exist "
                "at this point, so they cannot be cited now; speculating about them "
                "remains your choice."
            )
            compact_lines.append(
                f"- Not yet occurred: {', '.join(pending)} (not citable now)."
            )
    # コンテキスト予算（TASK-056「コンテキストを増やさず」）: 説明文の長い版から順に試し、
    # 収まる形を使う。最小形でも収まらないときだけ自分の本日発言行を落とす（時点・先行
    # 発言・まだ起きていないことは常に残す）。
    candidates = [full_lines, compact_lines]
    lines: list[str] = []
    variant_used = 0
    for index, candidate in enumerate(candidates):
        shaped = [line for line in candidate if line]
        if sum(len(line) for line in shaped) <= DAY_TIMELINE_MAX_CHARS:
            lines = shaped
            variant_used = index + 1
            break
    if not lines:
        lines = [
            line
            for line in compact_lines
            if line and not line.startswith(("- 今日の自分の発言", "- Your speeches today"))
        ]
        variant_used = len(candidates) + 1
    block = "\n".join(lines) + "\n\n"
    diagnostics: dict[str, object] = {
        "day_timeline_chars": len(block),
        "day_timeline_lines": len(lines),
        "timeline_point": snapshot.point(),
        "timeline_prior_events_this_turn": len(snapshot.prior_this_turn),
        "timeline_prior_events_today": snapshot.prior_today_events,
        "timeline_pending_speakers": len(snapshot.speakers_pending),
        "timeline_vote_already_done_today": snapshot.today_vote_done,
        "timeline_night_actions_pending": snapshot.night_pending,
        "timeline_next_speech_id": snapshot.next_speech_id,
        "timeline_own_speeches_today": len(snapshot.own_speeches_today),
        # 説明の長い版（1）/短い版（2）/最小形（3）のどれを使ったか（節約度の観測）
        "timeline_render_variant": variant_used,
    }
    return block, diagnostics


# --- 未来参照の検出（観測専用: 公開本文も発言の自由も書き換えない） --------------

# 婉曲・疑問・推測の表現は数えない（誤検出で議論の自由度を削らないための設計）。
_HEDGE_TERMS: tuple[str, ...] = (
    "かもしれない",
    "かも",
    "と思う",
    "思われる",
    "思います",
    "思いますが",
    "可能性がある",
    "可能性",
    "もし",
    "仮に",
    "推測",
    "未知",
    "未確認",
    "不明",
    "わからない",
    "分からない",
    "確実では",
    "確定では",
    "断定",
    "おそらく",
    "確信が持て",
    "検証したい",
    "確認したい",
    "unknown",
    "unconfirmed",
    "unverified",
    "unresolved",
    "notsure",
    "unsure",
    "uncertain",
    "maybe",
    "might",
    "possibly",
    "perhaps",
    "could",
    "guess",
    "suspect",
    "think",
    "believe",
    "suppose",
    "assume",
    "unlikely",
    "likely",
    "probably",
    "presumably",
    "whether",
    "wonder",
    "hypoth",
    "speculat",
    "seems",
    "appears",
    "soundslike",
)

# 他者の発言・公表を指す語（根拠参照の分子。ここに含まれる文だけを追跡対象にする）
_ATTRIBUTION_TERMS: tuple[str, ...] = (
    "言った",
    "言い",
    "発言",
    "述べ",
    "主張",
    "公表",
    "明らかに",
    "明言",
    "暴露",
    "答えた",
    "回答",
    "指摘",
    "名乗",
    "coした",
    "co済み",
    "宣言",
    "said",
    "says",
    "told",
    "stated",
    "claimed",
    "revealed",
    "announced",
    "pointedout",
    "answered",
    "responded",
    "argued",
    "mentioned",
    "declared",
)

_VOTE_ASSERTION_TERMS: tuple[str, ...] = (
    "今日の投票",
    "本日の投票",
    "今日この投票",
    "投票結果",
    "todaysvote",
    "thevote",
    "wevoted",
    "votedout",
)
_VOTE_OUTCOME_TERMS: tuple[str, ...] = (
    "決まった",
    "確定",
    "決着",
    "可決",
    "過半数",
    "全会一致",
    "passed",
    "decided",
    "settled",
    "unanimous",
    "majority",
    "carried",
)

_DEATH_ASSERTION_TERMS: tuple[str, ...] = (
    "襲撃死",
    "襲われた",
    "襲撃された",
    "やられた",
    "殺された",
    "死んだ",
    "死亡した",
    "死亡確認",
    "処刑された",
    "処刑で",
    "追放された",
    "killed",
    "dead",
    "died",
    "death",
    "attacked",
    "exiled",
    "executed",
    "murdered",
)
_TONIGHT_TERMS: tuple[str, ...] = ("今夜", "今晩", "tonight", "thisnight")


def _normalize_text(text: str) -> str:
    """比較用の正規化（NFKCで全角英数字・空白除去・小文字化）。"""
    folded = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", "", folded).casefold()


def mentions_players(sentence: str, player_names: tuple[str, ...]) -> tuple[str, ...]:
    """文からプレイヤー名を拾う（`Player 2` / `プレイヤー2` などの別表記にも対応）。"""
    norm = _normalize_text(sentence)
    hits: list[str] = []
    for name in player_names:
        key = _normalize_text(name)
        if re.search(rf"{re.escape(key)}(?!\d)", norm):
            hits.append(name)
            continue
        tail = key.replace("player", "")
        if tail and re.search(
            rf"(?:プレイヤー|Person|民){re.escape(tail)}(?![0-9])", norm
        ):
            hits.append(name)
    return tuple(dict.fromkeys(hits))


def _is_question(sentence: str) -> bool:
    return "?" in sentence or "？" in sentence


def _is_hedged(norm: str) -> bool:
    return any(term in norm for term in _HEDGE_TERMS)


@dataclass(frozen=True)
class FutureReferenceFinding:
    """発言時点で未発生だった事象を断定していた箇所（観測のみ・本文は触らない）。"""

    category: str
    sentence: str
    names: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        digest = hashlib.sha256(self.sentence.encode("utf-8")).hexdigest()[:16]
        return {
            "category": self.category,
            "names": list(self.names),
            "reason": self.reason,
            "sentence_excerpt": self.sentence[:80],
            "sentence_sha256": digest,
        }


@dataclass(frozen=True)
class TimelineReferenceReport:
    """公開本文を発言時点と突き合わせた結果（根拠追跡率と指摘の一覧）。"""

    findings: tuple[FutureReferenceFinding, ...]
    attributed_references: int
    unresolvable_references: int
    body_char_count: int

    @property
    def grounding_rate(self) -> float:
        """発言時点の根拠追跡率（参照した証拠が時点で確認できる割合）。"""
        if self.attributed_references <= 0:
            return 1.0
        return round(
            1.0 - (self.unresolvable_references / self.attributed_references), 4
        )

    def to_log_fields(self) -> dict[str, object]:
        categories = {category: 0 for category in FUTURE_REFERENCE_CATEGORIES}
        for finding in self.findings:
            categories[finding.category] = categories.get(finding.category, 0) + 1
        return {
            "future_event_reference": bool(self.findings),
            "future_event_reference_count": len(self.findings),
            "future_event_reference_categories": categories,
            "future_event_reference_findings": [
                finding.to_dict() for finding in self.findings[:8]
            ],
            "attributed_references_observed": self.attributed_references,
            "unresolvable_references": self.unresolvable_references,
            "evidence_grounding_rate_at_speech_time": self.grounding_rate,
            # 観測専用であることの証跡（自動修正・再生成要求は行わない）
            "body_modified": False,
        }


def inspect_public_body(
    body: str | None,
    snapshot: TurnSnapshot,
    player_names: tuple[str, ...],
) -> TimelineReferenceReport:
    """公開予定の本文を発言時点と突き合わせ、未来参照を観測する（書き換えない）。

    断定表現だけを対象にし、疑問文・推測・婉曲は数えない。検出しても本文の語彙・
    口調・結論には一切干渉しない（観測専用）。
    """
    if not isinstance(body, str) or not body.strip():
        return TimelineReferenceReport((), 0, 0, len(body or ""))
    from .discussion_format import split_sentences

    sentences = split_sentences(body) or [body]
    findings: list[FutureReferenceFinding] = []
    attributed = 0
    unresolvable = 0
    dead_known = dict(snapshot.known_dead)
    for sentence in sentences:
        norm = _normalize_text(sentence)
        if not norm or _is_question(sentence):
            continue
        names = mentions_players(sentence, tuple(player_names))
        others = tuple(name for name in names if name != snapshot.actor)
        hedged = _is_hedged(norm)
        cites_speech = any(term in norm for term in _ATTRIBUTION_TERMS)
        if cites_speech and names:
            attributed += 1
        category: str | None = None
        reason = ""
        # 1) まだ投票していない日の投票結果を断定していないか
        if (
            not snapshot.today_vote_done
            and not hedged
            and any(term in norm for term in _VOTE_ASSERTION_TERMS)
            and any(term in norm for term in _VOTE_OUTCOME_TERMS)
        ):
            category = FUTURE_VOTE_OUTCOME
            reason = "today's vote had not happened at this point"
        # 2) 今夜の夜行動の結果を発生前に断定していないか
        if (
            category is None
            and snapshot.night_pending
            and not hedged
            and any(term in norm for term in _TONIGHT_TERMS)
            and any(term in norm for term in _DEATH_ASSERTION_TERMS)
        ):
            # この時点で死亡が公開済みの人だけを挙げる文は「過ぎた夜」の話なので疑わない
            # （誰を指さない一般論や、生存者を対象にした断定だけが対象）
            named_past_death = bool(names) and all(
                name in dead_known or not snapshot.was_alive(name) for name in names
            )
            if not named_past_death:
                category = FUTURE_NIGHT_ACTION
                reason = "tonight's night actions had not resolved at this point"
        # 3) この時点で生存しているプレイヤーの死亡を断定していないか
        if category is None and not hedged:
            for name in names:
                if name in dead_known:
                    continue
                if any(term in norm for term in _DEATH_ASSERTION_TERMS):
                    category = FUTURE_DEATH_BEFORE_DEATH
                    reason = f"{name} was still alive at this point"
                    break
        # 4) まだ一度も公開発言していない相手の発言・COを引用していないか
        if (
            category is None
            and cites_speech
            and not hedged
            and others
            and all(not snapshot.spoken_before(name) for name in others)
        ):
            category = FUTURE_SPEECH_FROM_UNSPOKEN
            reason = "no public speech by them existed at this point"
        if category is not None:
            findings.append(
                FutureReferenceFinding(
                    category=category,
                    sentence=sentence.strip()[:200],
                    names=names,
                    reason=reason,
                )
            )
            # 根拠追跡率の分母は「他者の発言を引用した参照」だけ。投票・死亡・夜行動の
            # 断定は別枠の未来参照として数える（引用追跡率とゴチャ混ぜにしない）。
            if category == FUTURE_SPEECH_FROM_UNSPOKEN:
                unresolvable += 1
    return TimelineReferenceReport(
        findings=tuple(findings),
        attributed_references=attributed,
        unresolvable_references=unresolvable,
        body_char_count=len(body),
    )





