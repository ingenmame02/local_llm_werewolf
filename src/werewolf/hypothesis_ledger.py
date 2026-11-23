"""TASK-047: 確定事実台帳（FactLedger）と未確定解釈台帳（HypothesisLedger）の分離。

`FactLedger` はシステムが確定した事実（死亡・投票・役職COの公開・占い結果の公開・
実行済み行動）だけを保持する。`HypothesisLedger` はモデルの未確定解釈（誰が真らしいか、
矛盾、反証、確信度）を `evidence_ids`・生成者・対象日・反証状態付きで保持し、確定事実とは
別の欄として短く提示する。

プロンプト生成前に「死亡者への質問」と「未公開情報の参照」を解決する（死亡者を対象とする
仮説は historical 化して質問対象から除き、未公開情報を含む仮説は本人だけが見る）。
モデルへは全文ではなく「確定事実」「未解決の矛盾」「前回から変わった仮説」だけを短く渡す。
確信度の数値は参考情報であり、結論・口調・論点の強制は一切行わない。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 循環import回避。実行時は duck typing で GameState を扱う。
    from .models import GameState

STANCE_SUSPECT = "suspect"
STANCE_TRUST = "trust"

REFUTATION_OPEN = "open"
REFUTATION_CONTESTED = "contested"
REFUTATION_HISTORICAL = "historical"
REFUTATION_SUPERSEDED = "superseded"

VISIBILITY_PUBLIC = "public"
VISIBILITY_PRIVATE = "private"

# 確信度の数値は参考情報（スコアで結論や口調を縛らない）。
CONFIDENCE_SCORES = {"low": 0.2, "medium": 0.5, "high": 0.8}

STATEMENT_MAX_CHARS = 80
EVIDENCE_MAX_PER_LINE = 3
# 提示量の上限（モデルの論点・口調を縛らず、同じ状態の再掲だけを絞る）
RENDER_OWN_MAX = 4
RENDER_CHANGED_MAX = 3
RENDER_CONTRADICTION_MAX = 3
RENDER_HISTORICAL_MAX = 2

# 未公開情報の参照（本人限定の証拠）。他プレイヤーへは提示しない。
_PRIVATE_EVIDENCE_TOKENS = ("[private_", "private_", "[memo:", "memo:")

# 発言から未確定解釈の立場を読む（確定事実の語とは重ねない）。
_SUSPECT_CUES = re.compile(
    r"(?:疑|怪し|危|人狼|狼|黒|投票(?:先|対象|理由)?|処刑|追放|排除)"
    r"|(?:suspect|suspicious|werewolf|wolf|vote for|execute|exile|eliminate)",
    re.IGNORECASE,
)
_TRUST_CUES = re.compile(
    r"(?:信頼|信じ|安心|クリア|人間|信用)"
    r"|(?:trust|believe|credible|clean|human|not a wolf)",
    re.IGNORECASE,
)
_HIGH_CONFIDENCE = re.compile(
    r"(?:確信|間違いなく|確実に)|(?:confident|definitely|certainly|surely)",
    re.IGNORECASE,
)
_LOW_CONFIDENCE = re.compile(
    r"(?:可能性|かもしれ|まだ|保留|不確か)"
    r"|(?:maybe|might|possibly|still|hold|uncertain|not sure)",
    re.IGNORECASE,
)


def confidence_from_text(text: str | None) -> str:
    """発言の断定度から確信度ラベルを返す（数値ではなく分類のみ）。"""
    if not isinstance(text, str) or not text:
        return "medium"
    if _HIGH_CONFIDENCE.search(text):
        return "high"
    if _LOW_CONFIDENCE.search(text):
        return "low"
    return "medium"


def stance_of_sentence(sentence: str) -> str | None:
    """文から仮説の立場（疑う／信頼する）を読む。両方の語を含む文は判断しない。"""
    suspect = _SUSPECT_CUES.search(sentence) is not None
    trust = _TRUST_CUES.search(sentence) is not None
    if suspect and not trust:
        return STANCE_SUSPECT
    if trust and not suspect:
        return STANCE_TRUST
    return None
@dataclass(frozen=True)
class ContradictionPair:
    """確定事実・公開主張と仮説の間に検出した矛盾（警告として渡す・自動修正しない）。"""

    hypothesis_id: str
    kind: str
    summary: str
    evidence_ids: tuple[str, ...]


@dataclass
class HypothesisRecord:
    """未確定の解釈1件。確定事実（FactLedger）とは混ぜて保持しない。"""

    hypothesis_id: str
    subject: str
    stance: str
    statement: str
    generated_by: str
    day: int
    turn: int
    evidence_ids: list[str] = field(default_factory=list)
    confidence: str = "medium"
    confidence_score: float | None = None
    visibility: str = VISIBILITY_PUBLIC
    refutation_status: str = REFUTATION_OPEN
    counter_evidence_ids: list[str] = field(default_factory=list)
    first_day: int = 1
    last_updated_day: int = 1
    actionable: bool = True
    # TASK-056: この解釈が「いつ」上がってきたか（D日Tターン#時系列順）。同じ論点の
    # 要約が新しい事実として読み替えられないための発生時点だが、本文の追記はしない。
    observed_at: str = ""

    def signature(self) -> tuple[object, ...]:
        """前回提示からの差分判定用（立場・本文・確信度・反証状態・証拠集合）。"""
        return (
            self.stance,
            self.statement[:STATEMENT_MAX_CHARS],
            self.confidence,
            self.refutation_status,
            tuple(sorted(self.evidence_ids)),
        )


@dataclass
class HypothesisLedger:
    """未確定解釈台帳。同じ根拠の重複登録を抑制し、反証・死亡者参照を解決する。"""

    records: dict[str, HypothesisRecord] = field(default_factory=dict)
    contradictions: list[ContradictionPair] = field(default_factory=list)
    # 各プレイヤーへ最後に提示した仮説の署名（「前回から変わった仮説」の基準）
    presented: dict[str, dict[str, tuple[object, ...]]] = field(default_factory=dict)
    duplicates_suppressed: int = 0
    stance_changes: int = 0
    dead_references_suppressed: int = 0
    # 未公開情報の根拠を持つ公開仮説を他人へ提示したレコードID（重複カウントしない）
    _private_suppressed_ids: set[str] = field(default_factory=set, repr=False)

    @staticmethod
    def record_key(actor: str, subject: str, stance: str) -> str:
        return f"H-{actor}-{subject}-{stance}"

    def record(
        self,
        *,
        subject: str,
        actor: str,
        day: int,
        turn: int,
        stance: str,
        statement: str,
        evidence_ids: tuple[str, ...] | list[str],
        confidence: str = "medium",
        visibility: str = VISIBILITY_PUBLIC,
        observed_at: str = "",
    ) -> bool:
        """仮説を登録する。同一生成者・対象・立場・本文の再掲は証拠IDの統合のみ行う。"""
        key = self.record_key(actor, subject, stance)
        existing = self.records.get(key)
        incoming = [str(eid) for eid in evidence_ids if eid]
        statement_text = (statement or "").strip()[:STATEMENT_MAX_CHARS]
        if existing is not None and existing.visibility == visibility:
            # 同じ証拠の重複提示を抑制し、根拠IDだけ統合する
            for eid in incoming:
                if eid not in existing.evidence_ids:
                    existing.evidence_ids.append(eid)
            existing.last_updated_day = day
            existing.turn = turn
            if existing.statement[:60] != statement_text[:60]:
                existing.statement = statement_text
            self.duplicates_suppressed += 1
            return False
        # 立場の変更は旧仮説を superseded にし、新仮説を別レコードとして立てる
        for other in self.records.values():
            if (
                other.generated_by == actor
                and other.subject == subject
                and other.visibility == visibility
                and other.stance != stance
                and other.refutation_status != REFUTATION_SUPERSEDED
            ):
                other.refutation_status = REFUTATION_SUPERSEDED
                self.stance_changes += 1
        self.records[key] = HypothesisRecord(
            hypothesis_id=key,
            subject=subject,
            stance=stance,
            statement=statement_text,
            generated_by=actor,
            day=day,
            turn=turn,
            evidence_ids=incoming,
            confidence=confidence,
            confidence_score=CONFIDENCE_SCORES.get(confidence),
            visibility=visibility,
            first_day=day,
            last_updated_day=day,
            observed_at=observed_at or f"D{day}T{turn}",
        )
        return True

    def resolve(self, state: "GameState") -> None:
        """死亡者参照・公開占い結果との矛盾・重複COを台帳側で解決する。"""
        dead_names = frozenset(
            player.name for player in getattr(state, "players", ()) if not player.is_alive
        )
        verdicts: dict[str, list[tuple[str, str, str]]] = {}
        for index, claim in enumerate(getattr(state, "seer_claims", ()), 1):
            if claim.target is not None and claim.result is not None:
                verdicts.setdefault(claim.target, []).append(
                    (claim.claimant, claim.result, f"[seer_claim:{index}]")
                )
        role_claimants: dict[str, list[tuple[str, str]]] = {}
        for index, claim in enumerate(getattr(state, "public_role_claims", ()), 1):
            role_claimants.setdefault(claim.claimed_role, []).append(
                (claim.claimant, f"[claim:{index}]")
            )
        contradictions: list[ContradictionPair] = []
        for record in self.records.values():
            if record.refutation_status == REFUTATION_SUPERSEDED:
                continue
            if record.subject in dead_names:
                # 死亡者への質問・現在のアクションは成立しないので過去記録扱いにする
                if record.refutation_status != REFUTATION_HISTORICAL:
                    self.dead_references_suppressed += 1
                record.refutation_status = REFUTATION_HISTORICAL
                record.actionable = False
                record.counter_evidence_ids = []
                continue
            record.actionable = True
            counter_ids = sorted(
                {
                    evidence_id
                    for claimant, result, evidence_id in verdicts.get(record.subject, ())
                    if claimant != record.generated_by
                    and (
                        (record.stance == STANCE_SUSPECT and result == "human")
                        or (record.stance == STANCE_TRUST and result == "wolf")
                    )
                }
            )
            if counter_ids:
                record.refutation_status = REFUTATION_CONTESTED
                record.counter_evidence_ids = counter_ids
                contradictions.append(
                    ContradictionPair(
                        hypothesis_id=record.hypothesis_id,
                        kind="verdict_conflict",
                        summary=(
                            f"{record.subject}: stance '{record.stance}' by {record.generated_by} "
                            "cannot coexist with the published verdicts listed as counter-evidence"
                        ),
                        evidence_ids=tuple(counter_ids),
                    )
                )
            else:
                record.refutation_status = REFUTATION_OPEN
                record.counter_evidence_ids = []
            for role, claimants in role_claimants.items():
                if not any(claimant == record.subject for claimant, _ in claimants):
                    continue
                if len(claimants) <= 1:
                    continue
                ids = tuple(sorted({evidence_id for _, evidence_id in claimants}))
                contradictions.append(
                    ContradictionPair(
                        hypothesis_id=record.hypothesis_id,
                        kind="duplicate_role_claim",
                        summary=(
                            f"{record.subject} publicly claimed {role}, and "
                            f"{len(claimants) - 1} other player(s) claimed the same role; "
                            "at most one of them can be genuine"
                        ),
                        evidence_ids=ids,
                    )
                )
        self.contradictions = contradictions

    @staticmethod
    def _has_private_evidence(record: HypothesisRecord) -> bool:
        return any(
            token in evidence_id
            for evidence_id in record.evidence_ids
            for token in _PRIVATE_EVIDENCE_TOKENS
        )

    def visible_for(self, actor_name: str) -> list[HypothesisRecord]:
        """本人に見える仮説（公開分＋自分の私的仮説）。他人の私的仮説は渡さない。"""
        visible: list[HypothesisRecord] = []
        for record in self.records.values():
            if record.refutation_status == REFUTATION_SUPERSEDED:
                continue
            if record.visibility == VISIBILITY_PRIVATE and record.generated_by != actor_name:
                continue
            if (
                record.visibility == VISIBILITY_PUBLIC
                and record.generated_by != actor_name
                and self._has_private_evidence(record)
            ):
                self._private_suppressed_ids.add(record.hypothesis_id)
                continue
            visible.append(record)
        return visible

    def own_for(self, actor_name: str) -> list[HypothesisRecord]:
        return [
            record
            for record in self.visible_for(actor_name)
            if record.generated_by == actor_name
        ]

    def changed_for(self, actor_name: str) -> list[HypothesisRecord]:
        """前回このプレイヤーへ提示した状態から変わった仮説だけ返す。"""
        previous = self.presented.get(actor_name, {})
        return [
            record
            for record in self.visible_for(actor_name)
            if previous.get(record.hypothesis_id) != record.signature()
        ]

    def summary_for(self, actor_name: str, max_items: int = 6) -> str:
        """本人の仮説を1行の要約にする（発言全文の再掲の代わりに使う）。"""
        records = self.own_for(actor_name)[-max_items:]
        if not records:
            return ""
        return " / ".join(
            f"{record.subject} [{record.stance}] {record.statement[:60]} "
            f"({record.refutation_status}; evidence {', '.join(record.evidence_ids[:2])})"
            for record in records
        )

    @staticmethod
    def _render_line(record: HypothesisRecord) -> str:
        evidence_ids = record.evidence_ids[:EVIDENCE_MAX_PER_LINE]
        omitted = len(record.evidence_ids) - len(evidence_ids)
        score = CONFIDENCE_SCORES.get(record.confidence)
        parts = [
            f"{record.subject} [{record.stance}] {record.statement[:STATEMENT_MAX_CHARS]}"
            f" (confidence={record.confidence}"
            + (f", reference={score:.2f}" if score is not None else "")
            + f", {record.refutation_status})"
        ]
        if evidence_ids:
            parts.append(
                "evidence " + ", ".join(evidence_ids)
                + (f" (+{omitted} more)" if omitted > 0 else "")
            )
        if record.counter_evidence_ids:
            parts.append(
                "counter-evidence "
                + ", ".join(record.counter_evidence_ids[:EVIDENCE_MAX_PER_LINE])
            )
        if record.observed_at:
            # TASK-056: 観測時点を示す（古い解釈を新しい事実として扱わないための補助）
            parts.append(f"observed {record.observed_at}")
        if record.refutation_status == REFUTATION_HISTORICAL:
            parts.append("[historical_only]")
        return "- " + "; ".join(parts)

    def render_for(self, actor_name: str) -> tuple[str, dict[str, object]]:
        """本人向けの仮説台帳を短く描画する（確定事実の欄には含めない）。"""
        visible = self.visible_for(actor_name)
        own = [record for record in visible if record.generated_by == actor_name]
        changed = self.changed_for(actor_name)
        historical = [
            record
            for record in visible
            if record.refutation_status == REFUTATION_HISTORICAL
        ]
        own_ids = {record.hypothesis_id for record in own}
        # 他プレイヤーの仮説は公開発言（CONVERSATION_EVIDENCE）に既に存在するため再掲しない。
        # 確定事実・公開主張と矛盾する分だけをID付きの警告として渡す。
        contradictions = [
            pair
            for pair in self.contradictions
            if pair.hypothesis_id in {record.hypothesis_id for record in visible}
            and (pair.hypothesis_id in own_ids or pair.kind == "duplicate_role_claim")
        ]
        lines = [
            "[HYPOTHESIS_LEDGER: unverified interpretations, never confirmed facts]",
            "Confidence is a reference hint only - your conclusion, tone and reasoning stay yours. "
            "IDs point to [CLAIM_LEDGER] / the speech section below. An `observed DdTt` "
            "stamp marks when the item was raised, so an older line is never a new fact.",
        ]
        # 初回のみ自分の仮説一覧を渡す。以降は「前回から変わった仮説」の差分だけ示す
        # （同じ状態の再掲を省く。モデルの論点・口調には干渉しない）。
        if own and actor_name not in self.presented:
            lines.append("- Your own hypotheses:")
            lines.extend(self._render_line(record) for record in own[-RENDER_OWN_MAX:])
        if changed:
            lines.append("- Changed since your previous turn:")
            lines.extend(self._render_line(record) for record in changed[-RENDER_CHANGED_MAX:])
        if contradictions:
            lines.append("- Unresolved contradictions (warnings, not corrections):")
            lines.extend(
                f"- {pair.summary} [{pair.kind}] "
                + ", ".join(pair.evidence_ids[:EVIDENCE_MAX_PER_LINE])
                for pair in contradictions[-RENDER_CONTRADICTION_MAX:]
            )
        if historical:
            lines.append(
                "- Past record of dead subjects (questions or actions to them are impossible):"
            )
            lines.extend(self._render_line(record) for record in historical[-RENDER_HISTORICAL_MAX:])
        if not visible:
            lines.append("(No unverified hypotheses recorded yet)")
        block = "\n".join(lines)
        diagnostics: dict[str, object] = {
            "hypothesis_records_total": len(self.records),
            "hypothesis_records_rendered": len(visible),
            "hypothesis_records_own": len(own),
            "hypothesis_records_changed_since_last_turn": len(changed),
            "hypothesis_contradictions": len(contradictions),
            "hypothesis_historical_dead_subjects": len(historical),
            "hypothesis_duplicates_suppressed": self.duplicates_suppressed,
            "hypothesis_private_references_suppressed": len(self._private_suppressed_ids),
            "hypothesis_section_chars": len(block),
        }
        return block, diagnostics

    def snapshot_presentation(self, actor_name: str) -> None:
        """このプレイヤーへ提示済みの仮説状態を記録する（次回ターンの差分基準）。"""
        self.presented[actor_name] = {
            record.hypothesis_id: record.signature()
            for record in self.visible_for(actor_name)
        }

    def metrics(self) -> dict[str, object]:
        statuses = [record.refutation_status for record in self.records.values()]
        return {
            "hypothesis_count": len(self.records),
            "hypothesis_duplicates_suppressed": self.duplicates_suppressed,
            "hypothesis_stance_changes": self.stance_changes,
            "hypothesis_dead_references_suppressed": self.dead_references_suppressed,
            "hypothesis_private_references_suppressed": len(self._private_suppressed_ids),
            "hypothesis_contradictions": len(self.contradictions),
            # TASK-056: 発生時点つきのレコード数（古い要約を新事実として扱わないための観測）
            "hypothesis_records_with_observed_time": sum(
                1 for record in self.records.values() if record.observed_at
            ),
            "hypothesis_status_counts": {
                status: statuses.count(status)
                for status in (
                    REFUTATION_OPEN,
                    REFUTATION_CONTESTED,
                    REFUTATION_HISTORICAL,
                    REFUTATION_SUPERSEDED,
                )
            },
        }

    def serialize_public(self) -> dict[str, object]:
        """公開仮説のみシリアライズする（私的仮説・推理メモ由来はログに出さない）。"""
        serialized = self.serialize()
        return {
            "records": [
                record
                for record in serialized["records"]
                if record["visibility"] == VISIBILITY_PUBLIC
            ],
            "contradictions": serialized["contradictions"],
            "metrics": self.metrics(),
        }

    def serialize(self) -> dict[str, object]:
        return {
            "records": [
                {
                    "hypothesis_id": record.hypothesis_id,
                    "subject": record.subject,
                    "stance": record.stance,
                    "statement": record.statement,
                    "generated_by": record.generated_by,
                    "day": record.day,
                    "turn": record.turn,
                    "first_day": record.first_day,
                    "last_updated_day": record.last_updated_day,
                    "observed_at": record.observed_at,
                    "evidence_ids": list(record.evidence_ids),
                    "confidence": record.confidence,
                    "confidence_score": record.confidence_score,
                    "visibility": record.visibility,
                    "refutation_status": record.refutation_status,
                    "counter_evidence_ids": list(record.counter_evidence_ids),
                    "actionable": record.actionable,
                }
                for record in self.records.values()
            ],
            "contradictions": [
                {
                    "hypothesis_id": pair.hypothesis_id,
                    "kind": pair.kind,
                    "summary": pair.summary,
                    "evidence_ids": list(pair.evidence_ids),
                }
                for pair in self.contradictions
            ],
            "metrics": self.metrics(),
        }

