"""TASK-054: 秘密情報漏洩ガードのゲート分類・公開済み事実ID索引・モデル別ガード強度。

設計思想:
1. コンテキスト節約: 検査対象を「実際に公開される面」へ絞り、公開されない推理メモは
   観測（memo-only）へ逃がす。誤検出による全文再生成（数十秒・数千トークン）を削る。
2. モデルの一貫性: FactLedger / ClaimLedger で裏付けられる役職・判定の再言及は漏洩に
   しない。同じ証拠ID（``[claim:N]`` / ``[seer_claim:N]``）を検査結果へ残す。
3. 合理的な議論: 他者の役職推測は常に合法（gate=observe）。未公開の真の情報だけを
   hard gate にする。
4. 自由の担保: hardゲートは「非公開の真情報の開示」に限定し、口調・論点・結論を縛らない。
   ガード強度はモデル別の実績（真漏洩 vs 誤検出）で変え、一律にしない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:  # pragma: no cover - 型注釈のみ（循環import回避）
    from .models import GameState

# ---------------------------------------------------------------------------
# ゲート（遮断強度）の分類
# ---------------------------------------------------------------------------
GATE_HARD = "hard"
GATE_SOFT = "soft"
GATE_OBSERVE = "observe"
ALL_GATES: frozenset[str] = frozenset({GATE_HARD, GATE_SOFT, GATE_OBSERVE})

# ---------------------------------------------------------------------------
# ガーデニングポリシー（A/B比較用）
# ---------------------------------------------------------------------------
LEAK_POLICY_TIERED = "tiered"
LEAK_POLICY_LEGACY_STRICT = "legacy_strict"
DEFAULT_LEAK_GUARD_POLICY = LEAK_POLICY_TIERED
LEAK_GUARD_POLICIES: frozenset[str] = frozenset(
    {LEAK_POLICY_TIERED, LEAK_POLICY_LEGACY_STRICT}
)


def normalize_leak_guard_policy(value: object) -> str:
    """CLI・設定値をゲートポリシーへ正規化する（不明値は tiered）。"""
    if isinstance(value, str) and value.strip():
        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"tiered", "gate", "gated"}:
            return LEAK_POLICY_TIERED
        if normalized in {"legacy_strict", "legacy", "strict_legacy", "legacy_all"}:
            return LEAK_POLICY_LEGACY_STRICT
    return DEFAULT_LEAK_GUARD_POLICY


# ---------------------------------------------------------------------------
# モデル別ガード強度（一律にしない）
# ---------------------------------------------------------------------------
STRENGTH_BALANCED = "balanced"
STRENGTH_STRICT = "strict"
ALL_GUARD_STRENGTHS: frozenset[str] = frozenset({STRENGTH_BALANCED, STRENGTH_STRICT})

# 真漏洩の実績が1件以上あり、かつ誤検出実績を上回るモデルは strict へ寄せる。
STRICT_HARD_DETECTIONS = 1
# 誤検出（soft）が2件以上で真漏洩ゼロのモデルは balanced を維持する。
BALANCED_SOFT_NOTES = 2


# ---------------------------------------------------------------------------
# ルールID（TASK-054実装案1 の分類分離）
# ---------------------------------------------------------------------------
# hard: 非公開の真の情報だけを遮断する
RULE_SELF_ROLE_TRUE_CAMP = "self_role_true_camp_disclosure"
RULE_THIRD_PERSON_SELF_TRUE_CAMP = "third_person_self_true_camp_disclosure"
RULE_PARTNER_IDENTITY = "wolf_partner_identity"
RULE_PRIVATE_TEAM_REFERENCE = "private_team_reference"
RULE_TEAM_ACTION_VIEW = "wolf_team_action_view"
RULE_UNPUBLISHED_SEER_RESULT = "unpublished_seer_result"
RULE_SELF_ROLE_META = "self_role_meta_disclosure"
RULE_THIRD_PERSON_SELF_META = "third_person_self_role_meta"
RULE_INTERNAL_CONFIG = "internal_configuration_reference"
RULE_MODEL_IDENTIFIER = "model_identifier"
RULE_LEGACY_SELF_ROLE_ASSERTION = "self_role_assertion"
RULE_LEGACY_THIRD_PERSON_SELF = "third_person_self_role_assertion"

HARD_RULES: frozenset[str] = frozenset(
    {
        RULE_SELF_ROLE_TRUE_CAMP,
        RULE_THIRD_PERSON_SELF_TRUE_CAMP,
        RULE_PARTNER_IDENTITY,
        RULE_PRIVATE_TEAM_REFERENCE,
        RULE_TEAM_ACTION_VIEW,
        RULE_UNPUBLISHED_SEER_RESULT,
        RULE_SELF_ROLE_META,
        RULE_THIRD_PERSON_SELF_META,
        RULE_INTERNAL_CONFIG,
        RULE_MODEL_IDENTIFIER,
        RULE_LEGACY_SELF_ROLE_ASSERTION,
        RULE_LEGACY_THIRD_PERSON_SELF,
    }
)

# soft: 警告のみ。本文はそのまま公開する（再生成・置換しない）
RULE_SELF_ROLE_PUBLIC_CLAIM = "self_role_public_claim"
RULE_THIRD_PERSON_SELF_STYLE = "third_person_self_reference_note"
RULE_PARTNER_CO_MENTION = "wolf_partner_co_mention"

SOFT_RULES: frozenset[str] = frozenset(
    {
        RULE_SELF_ROLE_PUBLIC_CLAIM,
        RULE_THIRD_PERSON_SELF_STYLE,
        RULE_PARTNER_CO_MENTION,
    }
)

# observe: 観測のみ（公開済み事実の再言及・他者への推測・非公開領域）
RULE_PUBLIC_CLAIM_RESTATEMENT = "public_claim_restatement"
RULE_PUBLISHED_VERDICT_RESTATEMENT = "published_seer_verdict_restatement"
RULE_THIRD_PARTY_SPECULATION = "third_party_role_speculation"
RULE_MEMO_ONLY = "reasoning_memo_region"

OBSERVE_RULES: frozenset[str] = frozenset(
    {
        RULE_PUBLIC_CLAIM_RESTATEMENT,
        RULE_PUBLISHED_VERDICT_RESTATEMENT,
        RULE_THIRD_PARTY_SPECULATION,
        RULE_MEMO_ONLY,
    }
)

ALL_LEAK_RULES: frozenset[str] = HARD_RULES | SOFT_RULES | OBSERVE_RULES


def gate_of(rule_id: str) -> str:
    """ルールIDから既定ゲートを引く（未知のルールは hard 側で安全寄りにする）。"""
    if rule_id in SOFT_RULES or rule_id in OBSERVE_RULES:
        return GATE_SOFT if rule_id in SOFT_RULES else GATE_OBSERVE
    return GATE_HARD


# strict モデルでは soft 指摘も本文中立化の対象にする（誤検出モデルは本文そのまま）。
STRICT_REPAIRED_SOFT_RULES: frozenset[str] = frozenset(
    {RULE_SELF_ROLE_PUBLIC_CLAIM, RULE_THIRD_PERSON_SELF_STYLE}
)


# ---------------------------------------------------------------------------
# 公開済み事実の索引（TASK-054実装案2）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicRoleFact:
    """公開COの裏付け。``[claim:N]`` IDで台帳と対応する。"""

    claimant: str
    claimed_role: str
    fact_id: str


@dataclass(frozen=True)
class PublicVerdictFact:
    """公開占い判定の裏付け。``[seer_claim:N]`` IDで台帳と対応する。"""

    target: str
    result: str
    claimant: str
    fact_id: str


@dataclass(frozen=True)
class PublicFactIndex:
    """役職・判定が「既に公開済み」かを引き、漏洩ではなく再言及として扱う索引。"""

    role_facts: tuple[PublicRoleFact, ...] = ()
    verdict_facts: tuple[PublicVerdictFact, ...] = ()

    @classmethod
    def build(cls, state: "GameState") -> "PublicFactIndex":
        role_facts: list[PublicRoleFact] = []
        verdict_facts: list[PublicVerdictFact] = []
        for index, claim in enumerate(getattr(state, "public_role_claims", ()) or (), 1):
            claimant = getattr(claim, "claimant", None)
            role = getattr(claim, "claimed_role", None)
            if claimant and role:
                role_facts.append(
                    PublicRoleFact(
                        claimant=str(claimant),
                        claimed_role=str(role),
                        fact_id=f"[claim:{index}]",
                    )
                )
            target = getattr(claim, "target", None)
            result = getattr(claim, "result", None)
            if claimant and target and result:
                verdict_facts.append(
                    PublicVerdictFact(
                        target=str(target),
                        result=str(result),
                        claimant=str(claimant),
                        fact_id=f"[seer_claim:{index}]",
                    )
                )
        for index, claim in enumerate(getattr(state, "seer_claims", ()) or (), 1):
            claimant = getattr(claim, "claimant", None)
            target = getattr(claim, "target", None)
            result = getattr(claim, "result", None)
            if claimant and target and result:
                verdict_facts.append(
                    PublicVerdictFact(
                        target=str(target),
                        result=str(result),
                        claimant=str(claimant),
                        fact_id=f"[seer_claim:{index}]",
                    )
                )
        role_index = {(fact.claimant, fact.claimed_role) for fact in role_facts}
        verdict_index = {(fact.target, fact.result) for fact in verdict_facts}
        ledger = getattr(state, "claim_ledger", None)
        if ledger is not None:
            for claimant, record in (getattr(ledger, "public_claims", {}) or {}).items():
                role = getattr(record, "claimed_role", None)
                key = (str(claimant), str(role)) if role else None
                if key is not None and key not in role_index:
                    role_facts.append(
                        PublicRoleFact(
                            claimant=str(claimant),
                            claimed_role=str(role),
                            fact_id=f"[claim_ledger:{claimant}]",
                        )
                    )
                    role_index.add(key)
            for (claimant, target), result in (
                getattr(ledger, "public_seer_results", {}) or {}
            ).items():
                if not (claimant and target and result):
                    continue
                key = (str(target), str(result))
                if key not in verdict_index:
                    verdict_facts.append(
                        PublicVerdictFact(
                            target=str(target),
                            result=str(result),
                            claimant=str(claimant),
                            fact_id=f"[seer_claim_ledger:{claimant}:{target}]",
                        )
                    )
                    verdict_index.add(key)
        return cls(role_facts=tuple(role_facts), verdict_facts=tuple(verdict_facts))

    def role_fact_ids(self, claimant: str, role: str | None) -> tuple[str, ...]:
        """自分の公開CO（claimant + role）の裏付けIDを返す。"""
        if not claimant or not role:
            return ()
        return tuple(
            fact.fact_id
            for fact in self.role_facts
            if fact.claimant == claimant and fact.claimed_role == role
        )

    def own_claim_fact_ids(self, claimant: str) -> tuple[str, ...]:
        return tuple(fact.fact_id for fact in self.role_facts if fact.claimant == claimant)

    def verdict_fact_ids(self, target: str, result: str) -> tuple[str, ...]:
        return tuple(
            fact.fact_id
            for fact in self.verdict_facts
            if fact.target == target and fact.result == result
        )

    @property
    def public_verdict_pairs(self) -> frozenset[tuple[str, str]]:
        return frozenset((fact.target, fact.result) for fact in self.verdict_facts)

    @property
    def all_fact_ids(self) -> tuple[str, ...]:
        return tuple(fact.fact_id for fact in self.role_facts) + tuple(
            fact.fact_id for fact in self.verdict_facts
        )

# ---------------------------------------------------------------------------
# モデル別のガード実績プロファイル（TASK-054実装案4: 強度を一律にしない）
# ---------------------------------------------------------------------------


@dataclass
class LeakModelProfile:
    """モデル別に漏洩・誤検出・補正・再要求負荷を積上げる（強度判断の入力）。"""

    model: str
    hard_detections: int = 0
    soft_notes: int = 0
    observe_notes: int = 0
    sentence_repairs: int = 0
    regenerations: int = 0
    redactions: int = 0
    retry_seconds: float = 0.0
    body_char_total: int = 0
    body_char_kept: int = 0
    rule_counts: dict[str, int] = field(default_factory=dict)

    @property
    def false_note_rate(self) -> float:
        total = self.hard_detections + self.soft_notes
        return round(self.soft_notes / total, 4) if total else 0.0

    @property
    def body_preservation_rate(self) -> float:
        if not self.body_char_total:
            return 1.0
        return round(min(1.0, self.body_char_kept / self.body_char_total), 4)

    @property
    def guard_strength(self) -> str:
        if (
            self.hard_detections >= STRICT_HARD_DETECTIONS
            and self.hard_detections >= self.soft_notes
        ):
            return STRENGTH_STRICT
        return STRENGTH_BALANCED

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "guard_strength": self.guard_strength,
            "hard_detections": self.hard_detections,
            "soft_notes": self.soft_notes,
            "observe_notes": self.observe_notes,
            "sentence_repairs": self.sentence_repairs,
            "regenerations": self.regenerations,
            "redactions": self.redactions,
            "retry_seconds": round(self.retry_seconds, 3),
            "body_char_total": self.body_char_total,
            "body_char_kept": self.body_char_kept,
            "rule_counts": dict(sorted(self.rule_counts.items())),
            "false_note_rate": self.false_note_rate,
            "body_preservation_rate": self.body_preservation_rate,
        }


@dataclass
class LeakGuardProfiles:
    """モデル別実績からガード強度を決める（安全側のhardゲート自体は緩めない）。"""

    profiles: dict[str, LeakModelProfile] = field(default_factory=dict)
    policy: str = DEFAULT_LEAK_GUARD_POLICY

    def entry(self, model: str | None) -> LeakModelProfile:
        key = str(model or "unknown-model")
        profile = self.profiles.get(key)
        if profile is None:
            profile = LeakModelProfile(model=key)
            self.profiles[key] = profile
        return profile

    def observe(
        self,
        model: str | None,
        *,
        gate: str,
        rule_ids: tuple[str, ...] = (),
        repaired: bool = False,
        regenerated: bool = False,
        redacted: bool = False,
        retry_seconds: float = 0.0,
        body_char_count: int = 0,
        kept_char_count: int = 0,
    ) -> LeakModelProfile:
        profile = self.entry(model)
        if gate == GATE_HARD:
            profile.hard_detections += 1
        elif gate == GATE_SOFT:
            profile.soft_notes += 1
        else:
            profile.observe_notes += 1
        for rule_id in rule_ids:
            profile.rule_counts[rule_id] = profile.rule_counts.get(rule_id, 0) + 1
        if repaired:
            profile.sentence_repairs += 1
        if regenerated:
            profile.regenerations += 1
            profile.retry_seconds += max(0.0, retry_seconds)
        if redacted:
            profile.redactions += 1
        profile.body_char_total += max(0, body_char_count)
        profile.body_char_kept += max(0, min(body_char_count, kept_char_count))
        return profile

    def strength_for(self, model: str | None) -> str:
        """実績からガード強度を返す（真漏洩中心のモデルだけ strict に寄せる）。"""
        profile = self.profiles.get(str(model or "unknown-model"))
        if profile is None:
            return STRENGTH_BALANCED
        if (
            profile.hard_detections >= STRICT_HARD_DETECTIONS
            and profile.hard_detections >= profile.soft_notes
        ):
            return STRENGTH_STRICT
        return STRENGTH_BALANCED

    def summary(self) -> dict[str, object]:
        return {key: profile.to_dict() for key, profile in sorted(self.profiles.items())}

    def as_mapping(self) -> Mapping[str, object]:
        return self.summary()

