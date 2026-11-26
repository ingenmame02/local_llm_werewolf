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

# TASK-059: 引用された人物の「その時点での発言履歴」状態。誤帰属の内訳を分けることで、
# 実際の未来参照と、歴史参照・別ターン既発言の混同を区別し、婉曲や一般名指しを
# 違反として数えないための分類軸にする（本文の書き換えは一切しない）。
CITATION_SAME_TURN = "already_spoken_same_turn"
CITATION_EARLIER_TURN = "spoken_in_earlier_turn_or_day"
CITATION_DEAD_HISTORY = "dead_player_public_history"
CITATION_NEVER_SPOKEN_YET = "no_public_speech_at_point"
CITATION_STATUSES: tuple[str, ...] = (
    CITATION_SAME_TURN,
    CITATION_EARLIER_TURN,
    CITATION_DEAD_HISTORY,
    CITATION_NEVER_SPOKEN_YET,
)

# speech_from_unspoken_player の副型（TASK-059実装案2）。観測専用のタグで、本文は触らない。
SUBTYPE_ATTRIBUTION_NEVER_SPOKEN = "attribution_to_never_spoken_player"
SUBTYPE_GENERAL_MENTION_UNSPOKEN = "general_mention_without_attribution_observed"
SUBTYPE_HEDGED_CITATION_PENDING = "hedged_citation_of_pending_speech_observed"
CAUSE_NO_PUBLIC_SPEECH_AT_POINT = "actor_had_no_public_speech_at_this_point"

# TASK-065（実装案3）: 未発言者に関する文の「意図」を分ける。従来は未発言者の名前が
# 出て断定しているだけで `speech_from_unspoken_player`（未来参照）に数えたため、
# 「次にプレイヤー6へ投票理由を問いただすことを提案します」という**これから質問する行為**
# が未来参照として検出された（game_425755ce9a83 Day2 T1 Player 1）。
# 実質的な未来参照＝その人物が過去に述べたと断定するものだけを集計し、
# 質問行為とこれからの行動推測は別メトリクスへ移す（本文は変更しない）。
INTENT_PAST_ATTRIBUTION = "past_attribution_to_unspoken"
INTENT_FUTURE_QUESTION_TO_UNSPOKEN = "future_question_to_unspoken"
INTENT_FUTURE_ACTION_SPECULATION = "future_action_speculation"
INTENT_NONE = "none"
UNSPOKEN_INTENTS: tuple[str, ...] = (
    INTENT_PAST_ATTRIBUTION,
    INTENT_FUTURE_QUESTION_TO_UNSPOKEN,
    INTENT_FUTURE_ACTION_SPECULATION,
)

# 方策（A/B比較用）。separated が既定。literal は TASK-059 までの旧算入を再現する。
TIMELINE_INTENT_POLICY_SEPARATED = "separated"
TIMELINE_INTENT_POLICY_LITERAL = "literal"
TIMELINE_INTENT_POLICIES: tuple[str, ...] = (
    TIMELINE_INTENT_POLICY_SEPARATED,
    TIMELINE_INTENT_POLICY_LITERAL,
)
TIMELINE_INTENT_POLICY_DEFAULT = TIMELINE_INTENT_POLICY_SEPARATED


def normalize_timeline_intent_policy(value: object) -> str:
    """設定値を意図分離の方策へ正規化する（未知値は既定の separated）。"""
    if isinstance(value, str) and value.strip():
        candidate = value.strip().lower()
        if candidate in TIMELINE_INTENT_POLICIES:
            return candidate
    return TIMELINE_INTENT_POLICY_DEFAULT


def timeline_intent_policy_label(policy: object) -> str:
    """ログ用ラベル（intent_separated_detection / legacy_flat_detection）。"""
    if normalize_timeline_intent_policy(policy) == TIMELINE_INTENT_POLICY_LITERAL:
        return "legacy_flat_detection"
    return "intent_separated_detection"


# TASK-059（実装案3）: 本文に現れる証拠ID参照 `[speech:N]` / `[ref:sN]` を数え、未解決の
# 原因を分類する（ID欠落・旧ログ互換＝時点情報なし履歴・まだ存在しないID＝実際の未来
# 参照）。誤検出を増やさないため、この分類は観測だけで future_event_reference は増やさない。
REFERENCE_TOKEN_RE = re.compile(r"\[(?:speech|ref:s)\s*:?\s*(\d+)\]", re.IGNORECASE)
REFERENCE_RESOLVED_AT_POINT = "resolved_at_point"
REFERENCE_CAUSE_MISSING_ID = "citation_without_evidence_id"
REFERENCE_CAUSE_LEGACY_HISTORY = "unverifiable_history_without_stamps"
REFERENCE_CAUSE_ID_NOT_SPOKEN_YET = "id_not_spoken_at_this_point"
REFERENCE_TOKEN_CAUSES: tuple[str, ...] = (
    REFERENCE_RESOLVED_AT_POINT,
    REFERENCE_CAUSE_MISSING_ID,
    REFERENCE_CAUSE_LEGACY_HISTORY,
    REFERENCE_CAUSE_ID_NOT_SPOKEN_YET,
)


def classify_reference_tokens(
    sentence: str,
    snapshot: "TurnSnapshot",
) -> dict[str, int]:
    """1文の中の `[speech:N]` / `[ref:sN]` 参照を時点照合して分類する（観測専用）。

    - `resolved_at_point`: そのIDはこの時点で既に公開済み（発言順として追える）
    - `id_not_spoken_at_this_point`: まだ登られていないID（実際の未来参照）
    - `unverifiable_history_without_stamps`: 履歴に時点情報がなく検証不能（旧ログ互換）
    """
    counts: dict[str, int] = {}
    tokens = [int(value) for value in REFERENCE_TOKEN_RE.findall(sentence)]
    if not tokens:
        return counts
    known_ids = {point[3] for point in snapshot.all_speech_points if point[3]}
    legacy_history = not snapshot.all_speech_points
    for token in tokens:
        if token in known_ids:
            key = REFERENCE_RESOLVED_AT_POINT
        elif legacy_history:
            key = REFERENCE_CAUSE_LEGACY_HISTORY
        else:
            key = REFERENCE_CAUSE_ID_NOT_SPOKEN_YET
        counts[key] = counts.get(key, 0) + 1
    return counts



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
    # TASK-059: これまでの全公開発言のポイント (話者, day, turn, speech_id)。引用が
    # 「同一ターンの先行発言」「別ターン/別日の既発言」「死亡者の履歴」のどれかを
    # 判定するために使う（プロンプト本文や時系列欄の見た目はこの変更で変わらない）。
    all_speech_points: tuple[tuple[str, int, int, int], ...] = ()

    def point(self) -> str:
        return f"D{self.day}T{self.turn}#{self.sequence}"

    def spoken_before(self, name: str) -> bool:
        return name in self.spoken_players

    def was_alive(self, name: str) -> bool:
        return name in self.alive_players

    def citation_status(self, name: str) -> str:
        """TASK-059: この時点での引用可能状態を返す（観測専用の分類）。

        - 死亡が公開済みの人は、発言の有無に関わらず「履歴としての参照」（死後参照）へ寄せる
          （同一ターンの先行発言でも、その人物はもう発言できない立場だから）
        - 別ターン・別日に既に発言済みか
        - まだ公開発言が無いのか（実際の未来参照候補はこの状態だけ）
        """
        points = [point for point in self.all_speech_points if point[0] == name]
        dead = name in dict(self.known_dead)
        if dead:
            return CITATION_DEAD_HISTORY if points else CITATION_NEVER_SPOKEN_YET
        if any(point[1] == self.day and point[2] == self.turn for point in points):
            return CITATION_SAME_TURN
        if points:
            return CITATION_EARLIER_TURN
        return CITATION_NEVER_SPOKEN_YET

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
    # TASK-059: 引用された相手の時点別内訳（違反の指摘ではなく「どの時点の証拠を
    # 引いたか」の観測）。婉曲での仮定引用・未発言者名の一般言及は自由な議論なので
    # future_event_referenceとは別枠に数える。本文は書き換えない。
    citations_same_turn: int = 0
    citations_earlier_turn: int = 0
    citations_dead_history: int = 0
    citations_never_spoken: int = 0
    hedged_pending_citations_observed: int = 0
    mentions_without_attribution_observed: int = 0
    checks_skipped_empty_body: int = 0
    reference_cause_counts: dict[str, int] = field(default_factory=dict)
    # TASK-065: 未発言者に関する意図の内訳（観測専用）。findings（実質的な未来参照）とは
    # 別枠で、質問行為とこれからの行動推測を数える。本文の書き換えは一切しない。
    past_attribution_unspoken: int = 0
    future_question_unspoken: int = 0
    future_action_speculation_observed: int = 0
    intent_counts: dict[str, int] = field(default_factory=dict)
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
            # TASK-059の観測値（引用時点の内訳・観察カウンタ）は here では公開しない。
            # 台帳の公開面はTASK-056で固定した中立カウンタの形を保ち、内訳はイベント側
            # （report.to_log_fields）と observation_metrics() だけで見る。
            "day_timeline_chars_avg": round(
                self.section_chars_total / max(1, self.section_requests), 1
            ),
            "day_timeline_lines_avg": round(
                self.section_lines_total / max(1, self.section_requests), 2
            ),
        }

    def observation_metrics(self) -> dict[str, object]:
        """TASK-059の観測専用メトリクス（公開台帳の serialize には載せない内訳）。

        未来参照の指摘件数とは別枠で、「誰のどの時点の発言を根拠にしたかったのか」の
        分解と、違反に数えなかった観察（婉曲での仮定引用・未発言者名の一般言及）を出す。
        cli の診断ログ側（非公開面）でのみ使い、実機の議論の自由度へは干渉しない。
        """
        return {
            "citation_status_counts": {
                CITATION_SAME_TURN: self.citations_same_turn,
                CITATION_EARLIER_TURN: self.citations_earlier_turn,
                CITATION_DEAD_HISTORY: self.citations_dead_history,
                CITATION_NEVER_SPOKEN_YET: self.citations_never_spoken,
            },
            "hedged_pending_citations_observed": int(
                self.hedged_pending_citations_observed
            ),
            "mentions_without_attribution_observed": int(
                self.mentions_without_attribution_observed
            ),
            "timeline_checks_skipped_empty_body": int(self.checks_skipped_empty_body),
            "evidence_reference_cause_counts": dict(self.reference_cause_counts),
            # TASK-065: 未発言者に関する意図の内訳（実質的未来参照・質問行為・行動推測）。
            # future_event_reference_findings と違い、質問したことを違反として数えないための欄。
            "timeline_unspoken_intent_counts": {
                INTENT_PAST_ATTRIBUTION: self.past_attribution_unspoken,
                INTENT_FUTURE_QUESTION_TO_UNSPOKEN: self.future_question_unspoken,
                INTENT_FUTURE_ACTION_SPECULATION: self.future_action_speculation_observed,
            },
            "future_question_acts_observed": int(self.future_question_unspoken),
            "future_action_speculations_observed": int(
                self.future_action_speculation_observed
            ),
        }

    def record_observation(self, report: "TimelineReferenceReport") -> None:
        """TASK-059: 1検査分の観測値を台帳へ積む（本文・発言内容の修正は一切しない）。

        未来参照の指摘（findings）とは別に、引用された相手の時点別内訳と
        「違反に数えなかった観察」（婉曲での仮定引用・未発言者名の一般言及）を積む。
        """
        status_map = {
            CITATION_SAME_TURN: "citations_same_turn",
            CITATION_EARLIER_TURN: "citations_earlier_turn",
            CITATION_DEAD_HISTORY: "citations_dead_history",
            CITATION_NEVER_SPOKEN_YET: "citations_never_spoken",
        }
        for status, count in dict(report.citation_status_counts or {}).items():
            attribute = status_map.get(str(status))
            if attribute is not None:
                setattr(self, attribute, int(getattr(self, attribute)) + int(count))
        self.hedged_pending_citations_observed += int(report.hedged_pending_citations)
        self.mentions_without_attribution_observed += int(
            report.mentions_without_attribution
        )
        for cause, count in dict(report.reference_token_counts or {}).items():
            key = str(cause)
            self.reference_cause_counts[key] = (
                int(self.reference_cause_counts.get(key, 0)) + int(count)
            )
        # TASK-065: 意図の内訳を積む（実質的未来参照・質問行為・これからの行動推測）。
        # findings の合計とは別枠で、検出しても本文は直していない。
        self.past_attribution_unspoken += int(getattr(report, "past_attributions_observed", 0))
        self.future_question_unspoken += int(getattr(report, "future_question_acts", 0))
        self.future_action_speculation_observed += int(
            getattr(report, "future_action_speculations", 0)
        )
        for intent, count in dict(getattr(report, "intent_counts", {}) or {}).items():
            key = str(intent)
            self.intent_counts[key] = int(self.intent_counts.get(key, 0)) + int(count)

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
        # TASK-059: 全公開発言のポイント（話者, day, turn, speech_id）。引用の時点判定に使う。
        all_speech_points=tuple(
            (str(event.actor or ""), int(event.day), int(event.turn), int(event.speech_id or 0))
            for event in all_speeches
            if event.actor
        ),
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


# --- TASK-065: 未発言者に関する文の意図分類（観測専用・本文は書き換えない） ---------

# 「その人が過去に述べた」という帰属の形（実質的な未来参照として残す数え手）。
# 引用の形を広く取りすぎると「これから聞く」依頼を巻き込むため、過去の出来事として
# 耳にした・読んだと述べる形と完了形の帰属に絞る（TASK-056の検出強度は下げない）。
_PAST_ATTRIBUTION_TERMS: tuple[str, ...] = (
    "と言った",
    "が言った",
    "いい残",
    "言い残",
    "述べた",
    "語った",
    "答えた",
    "返した",
    "名乗った",
    "公言した",
    "公表した",
    "明かした",
    "暴露した",
    "指摘した",
    "主張した",
    "表明した",
    "発言を聞い",
    "発言を耳にし",
    "発言を見た",
    "先の発言",
    "発言内容から",
    "過去の発言",
    "先ほどの発言",
    "前回の発言",
    "その発言は",
    "said",
    "stated",
    "claimed",
    "told",
    "argued",
    "declared",
)

# これから聞き出す側（質問行為）。動詞と誘い文句の両方を見て判断する。
_REQUEST_VERB_TERMS: tuple[str, ...] = (
    "質問",
    "問い",
    "尋ね",
    "たずね",
    # TASK-065: 「問いただす／問い質す／問い正す／問い詰める」は未来の回答を求める
    # 行為（game_425755ce9a83 Day2 T1 Player 1の誤検出文）。旧語彙だと別名の未発言者へ
    # の質問が「過去の発言の引用」として future_event_reference に数えられていた。
    "問いただ",
    "問い質",
    "問い正",
    "問い詰め",
    "聞きたい",
    "聞いて",
    "聞く",
    "聞かせて",
    "ききたい",
    "教え",
    "答えを",
    "回答",
    "説明",
    "共有",
    "述べて",
    "報告",
    "開示",
    "comment",
    "ask",
    "question",
    "explain",
    "clarify",
    "tellme",
)
_SOLICITATION_TERMS: tuple[str, ...] = (
    "ください",
    "下さい",
    "ほしい",
    "欲しい",
    "もらいたい",
    "願う",
    "依頼",
    "求め",
    "提案",
    "検討",
    "予定",
    "次に",
    "これから",
    "今後",
    "次回",
    "まずは",
    "順番",
    "待ち",
    "ask",
    "wantto",
    "wouldlike",
    "please",
    "next",
)
# これから起こるその人物自身の行動への推測（投票・夜行動・COなど）。
_ACTION_SPECULATION_VERBS: tuple[str, ...] = (
    "する",
    "します",
    "つもり",
    "はず",
    "だろう",
    "ことになる",
    "予定",
    "投票",
    "襲",
    "攻撃",
    "占",
    "名乗",
    "co",
    "発言",
    "protect",
    "vote",
    "attack",
)
# 名の直後に目的語マーカーが来たら「その人への提案」であり本人の行動ではない。
_OBJECT_PARTICLES: tuple[str, ...] = ("に", "を", "へ", "から", "で", "against", "to")


def classify_unspoken_intent(
    sentence: str,
    norm: str,
    snapshot: "TurnSnapshot",
    mentions: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    """未発言者に関する文の意図を分ける（観測専用・判定の緩和強化はここでは行わない）。

    戻り値は (意図, 対象にした未発言者)。未発言者を挙げていなければ INTENT_NONE。
    順序は「これから聞く」→「過去の発言として扱う」→「これからの行動推測」。
    疑問文・依頼文を無条件に無害扱いせず、過去の発言を根拠にしているか、
    未来の回答を求めているかを見て分ける（TASK-065実装案3）。
    """
    targets = tuple(
        name
        for name in mentions
        if name != snapshot.actor and snapshot.citation_status(name) == CITATION_NEVER_SPOKEN_YET
    )
    if not targets:
        return INTENT_NONE, ()
    question = _is_question(sentence)
    request_verb = any(term in norm for term in _REQUEST_VERB_TERMS)
    solicitation = any(term in norm for term in _SOLICITATION_TERMS)
    past_terms = [term for term in _PAST_ATTRIBUTION_TERMS if term in norm]
    action_gaps = _action_speculation_gaps(norm, targets)
    if question or (request_verb and solicitation):
        return INTENT_FUTURE_QUESTION_TO_UNSPOKEN, targets
    if past_terms:
        # 未発言者の「過去の発言」を根拠にしている＝実質的な未来参照
        return INTENT_PAST_ATTRIBUTION, targets
    if action_gaps:
        return INTENT_FUTURE_ACTION_SPECULATION, targets
    return INTENT_NONE, targets


def _action_speculation_gaps(norm: str, targets: tuple[str, ...]) -> tuple[str, ...]:
    """言及名の直後にある行動表現（本人が主語のものだけ）。目的語化している物は除く。"""
    gaps: list[str] = []
    for name in targets:
        key = _normalize_text(name)
        if not key:
            continue
        for match in re.finditer(re.escape(key), norm):
            tail = norm[match.end() : match.end() + 14]
            if not tail:
                continue
            if tail[0] in _OBJECT_PARTICLES:
                # 「Player 6に投票すべきだ」型は本人の行動ではなく他者からの提案
                continue
            if any(verb in tail for verb in _ACTION_SPECULATION_VERBS):
                gaps.append(tail[:20])
    return tuple(gaps)




@dataclass(frozen=True)
class FutureReferenceFinding:
    """発言時点で未発生だった事象を断定していた箇所（観測のみ・本文は触らない）。"""

    category: str
    sentence: str
    names: tuple[str, ...]
    reason: str
    # TASK-059: speech_from_unspoken_player の内訳分類（観測専用タグ）。判定は変えない。
    subtype: str = ""
    cause: str = ""
    # TASK-065: 未発言者に関する文の意図（past_attribution_to_unspoken など）。
    # 質問行為・これからの行動推測は findings へ数えないための分離キー。
    intent: str = ""

    def to_dict(self) -> dict[str, object]:
        digest = hashlib.sha256(self.sentence.encode("utf-8")).hexdigest()[:16]
        payload: dict[str, object] = {
            "category": self.category,
            "names": list(self.names),
            "reason": self.reason,
            "sentence_excerpt": self.sentence[:80],
            "sentence_sha256": digest,
        }
        if self.subtype:
            payload["subtype"] = self.subtype
        if self.cause:
            payload["cause"] = self.cause
        if self.intent:
            payload["intent"] = self.intent
        return payload


@dataclass(frozen=True)
class TimelineReferenceReport:
    """公開本文を発言時点と突き合わせた結果（根拠追跡率と指摘の一覧）。"""

    findings: tuple[FutureReferenceFinding, ...]
    attributed_references: int
    unresolvable_references: int
    body_char_count: int
    # TASK-059: 引用された相手の時点別内訳と、違反には数えない観察カウンタ。
    # （婉曲での仮定引用・未発言者名の一般言及は自由な議論なので区別して数える）
    citation_status_counts: dict = field(default_factory=dict)
    hedged_pending_citations: int = 0
    mentions_without_attribution: int = 0
    # TASK-059: 本文に現れる証拠ID参照 `[speech:N]` / `[ref:sN]` を時点照合した内訳（解決済み・
    # まだ存在しないID・旧ログ互換で検証不能・IDなしの帰属）
    reference_token_counts: dict = field(default_factory=dict)
    # TASK-065: 未発言者に関する文の意図内訳。実質的な未来参照（past_attribution_to_unspoken）
    # だけを findings へ数え、質問行為とこれからの行動推測は別カウンタへ寄せる。
    intent_counts: dict = field(default_factory=dict)
    past_attributions_observed: int = 0
    future_question_acts: int = 0
    future_action_speculations: int = 0
    intent_policy: str = TIMELINE_INTENT_POLICY_DEFAULT

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
            # TASK-059: 引用の時点状態と観察専用カウンタ（誤検出率の監視に使う。
            # future_event_reference_categories と違い、これは違反の指摘ではない）。
            "citation_status_counts": dict(self.citation_status_counts),
            "hedged_pending_citations_observed": int(self.hedged_pending_citations),
            "mentions_without_attribution_observed": int(
                self.mentions_without_attribution
            ),
            # TASK-059: 証拠ID参照の時点照合結果（原因別の内訳）。実在しないIDを指して
            # いないかを実機ログで分解するために使う（本文は変えない）。
            "evidence_reference_counts": dict(self.reference_token_counts),
            # TASK-065: 未発言者に関する意図の内訳。違反として数えるのは実質的な未来参照
            # （past_attribution_to_unspoken）だけで、質問行為とこれからの行動推測は
            # 観測専用へ分ける（「質問したから未来参照」をなくすための分離キー）。
            "timeline_intent_counts": dict(self.intent_counts),
            "past_attributions_observed": int(self.past_attributions_observed),
            "future_question_acts_observed": int(self.future_question_acts),
            "future_action_speculations_observed": int(self.future_action_speculations),
            "timeline_intent_policy": self.intent_policy,
            "timeline_intent_policy_label": timeline_intent_policy_label(self.intent_policy),
            # 観測専用であることの証跡（自動修正・再生成要求は行わない）
            "body_modified": False,
        }


def inspect_public_body(
    body: str | None,
    snapshot: TurnSnapshot,
    player_names: tuple[str, ...],
    *,
    intent_policy: object = TIMELINE_INTENT_POLICY_DEFAULT,
) -> TimelineReferenceReport:
    """公開予定の本文を発言時点と突き合わせ、未来参照を観測する（書き換えない）。

    断定表現だけを対象にし、疑問文・推測・婉曲は数えない。検出しても本文の語彙・
    口調・結論には一切干渉しない（観測専用）。

    TASK-065: `separated`（既定）では未発言者に関する文の意図を分ける。これから回答を
    求める質問行為（`future_question_to_unspoken`）は future_event_reference に数えず
    別メトリクスへ移す。これからの行動推測（`future_action_speculation`）も意図として
    分けて数えるが、発言引用を伴う断定なら従来どおり検出する（TASK-056/059の強度を
    下げない）。実質的な未来参照は `past_attribution_to_unspoken`（その人が過去に
    述べたとする断定）ですくう。`literal` は TASK-059 までと同じ算入（A/B比較用の旧挙動）。
    """
    normalized_intent_policy = normalize_timeline_intent_policy(intent_policy)
    if not isinstance(body, str) or not body.strip():
        return TimelineReferenceReport(
            (),
            0,
            0,
            len(body or ""),
            intent_policy=normalized_intent_policy,
        )
    from .discussion_format import split_sentences

    sentences = split_sentences(body) or [body]
    findings: list[FutureReferenceFinding] = []
    attributed = 0
    unresolvable = 0
    # TASK-059: 時点別の引用状態カウントと、違反扱いしない観察値（婉曲・一般名指し）
    citation_counts: dict[str, int] = {}
    hedged_pending = 0
    mentions_no_attribution = 0
    reference_counts: dict[str, int] = {}
    dead_known = dict(snapshot.known_dead)
    # TASK-065: 意図の内訳をまず全文から拾う（疑問文の質問行為もここなら数えられる）。
    # 以降の違反判定ループは従来どおりに動かし、観測だけを追加する（本文は触らない）。
    intent_counts: dict[str, int] = {}
    past_attributions = 0
    question_acts = 0
    action_speculations = 0
    if normalized_intent_policy == TIMELINE_INTENT_POLICY_SEPARATED:
        for sentence in sentences:
            folded_intent = _normalize_text(sentence)
            if not folded_intent:
                continue
            names_intent = mentions_players(sentence, tuple(player_names))
            if not names_intent:
                continue
            intent, targets = classify_unspoken_intent(
                sentence, folded_intent, snapshot, names_intent
            )
            if intent == INTENT_NONE or not targets:
                continue
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            if intent == INTENT_FUTURE_QUESTION_TO_UNSPOKEN:
                question_acts += 1
            elif intent == INTENT_FUTURE_ACTION_SPECULATION:
                action_speculations += 1
            else:
                past_attributions += 1
    for sentence in sentences:
        norm = _normalize_text(sentence)
        if not norm or _is_question(sentence):
            continue
        names = mentions_players(sentence, tuple(player_names))
        others = tuple(name for name in names if name != snapshot.actor)
        hedged = _is_hedged(norm)
        cites_speech = any(term in norm for term in _ATTRIBUTION_TERMS)
        # TASK-059: 本文に書かれた証拠ID参照（`[speech:N]` / `[ref:sN]`）を時点照合する。
        # ここは分類だけで future_event_reference の件数も本文も変えない（誤検出を増やさない）。
        sentence_reference = classify_reference_tokens(sentence, snapshot)
        for reference_key, reference_count in sentence_reference.items():
            reference_counts[reference_key] = (
                reference_counts.get(reference_key, 0) + reference_count
            )
        if cites_speech and names:
            attributed += 1
            if others and not sentence_reference:
                # 発言の帰属なのに証拠IDが付いていない（追跡の糸口が本文しかない）ケース
                reference_counts[REFERENCE_CAUSE_MISSING_ID] = (
                    reference_counts.get(REFERENCE_CAUSE_MISSING_ID, 0) + 1
                )
        # TASK-059: 引用された相手それぞれの時点での発言履歴を分類して記録する。
        # 判定（future_event_reference）はこの追記では変えない — 数えるのは従来どおり
        # 「未発言者を断定した場合」だけで、婉曲と一般名指しは観察専用にする。
        if others:
            statuses = {name: snapshot.citation_status(name) for name in others}
            for status in statuses.values():
                citation_counts[status] = citation_counts.get(status, 0) + 1
            all_without_speech = all(
                status == CITATION_NEVER_SPOKEN_YET for status in statuses.values()
            )
            if cites_speech and hedged and all_without_speech:
                # 「もし Player X が言っていたなら」式の仮定は自由な議論なので指摘しない
                hedged_pending += 1
            elif not cites_speech and all_without_speech:
                # 未発言者の名前を一般論として挙げただけ（誤帰属ではない）を観察する
                mentions_no_attribution += 1
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
        # 4) まだ一度も公開発言していない相手の「過去の発言」を断定していないか
        #    （TASK-065: これから回答を求める質問行為だけ違反から外す。これからの行動
        #     推測は別メトリクスへ分けるが、発言引用を伴う断定であれば従来どおり検出し、
        #     TASK-056 / TASK-059 の検出強度を落とさない）
        sentence_intent = INTENT_NONE
        if (
            category is None
            and cites_speech
            and not hedged
            and others
            and all(not snapshot.spoken_before(name) for name in others)
        ):
            sentence_intent, _ = classify_unspoken_intent(sentence, norm, snapshot, names)
            asked_future_answer = (
                normalized_intent_policy == TIMELINE_INTENT_POLICY_SEPARATED
                and sentence_intent == INTENT_FUTURE_QUESTION_TO_UNSPOKEN
            )
            if not asked_future_answer:
                category = FUTURE_SPEECH_FROM_UNSPOKEN
                reason = "no public speech by them existed at this point"
        if category is not None:
            subtype = ""
            cause = ""
            if category == FUTURE_SPEECH_FROM_UNSPOKEN:
                # TASK-059: 実際に未発言者への断定だった場合の内訳タグ（観測専用で、
                # 判定の緩和・強化はこのフィールドでは行わない）。ID参照の実在まで見る。
                subtype = SUBTYPE_ATTRIBUTION_NEVER_SPOKEN
                if REFERENCE_CAUSE_ID_NOT_SPOKEN_YET in sentence_reference:
                    # 本文がまだ登られていないIDを引いていた（実際の未来参照）
                    cause = REFERENCE_CAUSE_ID_NOT_SPOKEN_YET
                elif REFERENCE_CAUSE_LEGACY_HISTORY in sentence_reference:
                    # 時点情報がない履歴（旧ログ互換）では検証不能だった、と分ける
                    cause = REFERENCE_CAUSE_LEGACY_HISTORY
                else:
                    # IDなしに言及しただけ／そもそもその人物の公開発言が存在しない
                    cause = CAUSE_NO_PUBLIC_SPEECH_AT_POINT
            findings.append(
                FutureReferenceFinding(
                    category=category,
                    sentence=sentence.strip()[:200],
                    names=names,
                    reason=reason,
                    subtype=subtype,
                    cause=cause,
                    # TASK-065: 未発言者に関する断定だった場合だけ意図を添える
                    # （past_attribution_to_unspoken。質問・推測はここで違反にならない）
                    intent=sentence_intent if category == FUTURE_SPEECH_FROM_UNSPOKEN else "",
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
        citation_status_counts=dict(citation_counts),
        hedged_pending_citations=int(hedged_pending),
        mentions_without_attribution=int(mentions_no_attribution),
        reference_token_counts=dict(reference_counts),
        # TASK-065: 意図の内訳。findings（future_event_reference）へ数えたのは
        # past_attribution の断定だけ。質問行為とこれからの行動推測は観測専用で分ける。
        intent_counts=dict(intent_counts),
        past_attributions_observed=int(past_attributions),
        future_question_acts=int(question_acts),
        future_action_speculations=int(action_speculations),
        intent_policy=normalized_intent_policy,
    )





