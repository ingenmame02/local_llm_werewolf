"""TASK-055: 役職主張（CO）と役職由来情報のプロヴェナンス分離。

3つの概念を分ける。

- ``RoleEvidence``: ゲームエンジンが発行した情報だけを保持する（夜の占い結果など）。
  保有者（owner）が確定しており、``source="private_action"`` ですぐに特定できる。
- ``PublicInterpretation``: 公開されたCO行・占い結果行を「誰がどう申告したか」として
  記録し、エンジン発行の情報と照合できる読み方にしておく。
- ``PublicRoleClaim``（既存のRoleClaim台帳行）: 発言から登録された各行に
  ``source`` と ``evidence_ids`` を持たせる。

設計上の重要な制約（推論ゲームを壊さないための割り切り）:
    エンジン照合の結果（本物の占い結果と一致するか否か）は管理者ログ専用にする。
    公開ログや他プレイヤーのプロンプトへ engine_supported / unsupported を出すと、
    その瞬間に「真占い師かブラフか」が判ってしまう。そのため公開面ではすべての
    CO行・結果行を例外なく同じ中立ラベル ``public_claim_line`` で揃え、判定の根拠は
    「本人の申告」であることを明示する。真偽の追跡は private ログの台帳で行う。

ブラフ自体は自由（既定は許可）。COするか、どの役職を名乗り、誰を疑うかはモデルの判断に
委ね、本文の削除・再生成・口調の強制は新たには行わない。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# ---------------------------------------------------------------------------
# 出所（source）語彙（TASK-055実装案1）
# ---------------------------------------------------------------------------
SOURCE_PRIVATE_ACTION = "private_action"
SOURCE_PUBLIC_CLAIM = "public_claim"
SOURCE_MODEL_INFERENCE = "model_inference"
SOURCE_COMPATIBILITY_TEXT = "compatibility_text"

EVIDENCE_SOURCES: frozenset[str] = frozenset({SOURCE_PRIVATE_ACTION})
CLAIM_SOURCES: frozenset[str] = frozenset(
    {
        SOURCE_PUBLIC_CLAIM,
        SOURCE_COMPATIBILITY_TEXT,
        SOURCE_MODEL_INFERENCE,
    }
)
ALL_PROVENANCE_SOURCES: frozenset[str] = EVIDENCE_SOURCES | CLAIM_SOURCES


def normalize_provenance_source(
    value: object, default: str = SOURCE_COMPATIBILITY_TEXT
) -> str:
    """CLI・台帳・イベント値を出所語彙へ正規化する（未知値は保守的に default）。"""
    if isinstance(value, str):
        candidate = value.strip().casefold()
        # ツール呼び出しで公開されたCOも「本人の公開主張」なので public_claim に寄せる
        if candidate == "tool":
            return SOURCE_PUBLIC_CLAIM
        if candidate in ALL_PROVENANCE_SOURCES:
            return candidate
    return default


# ---------------------------------------------------------------------------
# ルール設定 allow_bluff_role_claim（TASK-055実装案2）
# ---------------------------------------------------------------------------
ALLOW_BLUFF_ROLE_CLAIM_DEFAULT = True

_ALLOW_ALIASES = {"allow", "allowed", "true", "yes", "on", "1", "bluff"}
_DENY_ALIASES = {"deny", "denied", "reject", "rejected", "false", "no", "off", "0", "strict"}

BLUFF_ROLE_CLAIM_ALLOW = "allow"
BLUFF_ROLE_CLAIM_DENY = "deny"


def normalize_allow_bluff_role_claim(value: object) -> bool:
    """boolだけでなく "allow"/"deny" 等の設定文字列も受け取る（不明値は既定の許可）。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        candidate = value.strip().casefold()
        if candidate in _ALLOW_ALIASES:
            return True
        if candidate in _DENY_ALIASES:
            return False
    return ALLOW_BLUFF_ROLE_CLAIM_DEFAULT


def bluff_role_claim_policy_label(allowed: bool) -> str:
    return BLUFF_ROLE_CLAIM_ALLOW if allowed else BLUFF_ROLE_CLAIM_DENY


# ---------------------------------------------------------------------------
# 分類カテゴリ（公開面へ出さない内部カテゴリ）
# ---------------------------------------------------------------------------
# エンジン発行の占い結果と一致した判定行（管理者のみが読める）
CATEGORY_ENGINE_ISSUED = "engine_issued_evidence"
# 本人が申告しただけの行（役職CO単体など、照合対象が存在しない）
CATEGORY_SELF_REPORTED = "self_reported_unverified"
# 本人が実際に持っていない役職由来情報の申告（捏造された占い結果など）
CATEGORY_UNSUPPORTED = "unsupported_role_evidence"
# allow_bluff_role_claim=False の設定で受理しなかった行
CATEGORY_DISALLOWED = "bluff_role_claim_disallowed"
CATEGORIES: frozenset[str] = frozenset(
    {
        CATEGORY_ENGINE_ISSUED,
        CATEGORY_SELF_REPORTED,
        CATEGORY_UNSUPPORTED,
        CATEGORY_DISALLOWED,
    }
)

# 公開ログ・プロンプトで全CO行に付ける中立ラベル（真偽を示さない）
PUBLIC_LINE_LABEL = "public_claim_line"

DIVINATION_KIND = "divination"


def divination_evidence_id(day: int, owner: str, target: str) -> str:
    """エンジンが発行した占い結果の証拠ID（保有者名を含む＝管理者専用ID）。"""
    return f"[engine_seer:{day}:{owner}->{target}]"


@dataclass(frozen=True)
class RoleEvidence:
    """ゲームエンジンが発行した非公開情報（保有者にだけ見える確定的な情報）。"""

    evidence_id: str
    day: int
    owner: str
    kind: str = DIVINATION_KIND
    target: str | None = None
    result: str | None = None
    source: str = SOURCE_PRIVATE_ACTION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PublicInterpretation:
    """公開されたCO行・結果行を「誰の申告か」として読むための記録。"""

    claim_ref: str
    day: int
    claimant: str
    claimed_role: str
    target: str | None = None
    result: str | None = None
    source: str = SOURCE_COMPATIBILITY_TEXT
    category: str = CATEGORY_SELF_REPORTED
    engine_supported: bool = False
    engine_evidence_ids: tuple[str, ...] = ()
    discussion_turn: int = 0

    def public_view(self) -> dict[str, object]:
        """公開ログ・他プレイヤーへ出してよい形（真偽を示す情報は含まない）。"""
        return {
            "claim_ref": self.claim_ref,
            "day": self.day,
            "discussion_turn": self.discussion_turn,
            "claimant": self.claimant,
            "claimed_role": self.claimed_role,
            "target": self.target,
            "claimed_result": self.result,
            "source": normalize_provenance_source(self.source),
            "interpretation": PUBLIC_LINE_LABEL,
        }

    def admin_view(self) -> dict[str, object]:
        """管理者ログ用。エンジン発行情報との照合結果を含める。"""
        view = self.public_view()
        view["category"] = self.category
        view["engine_supported"] = self.engine_supported
        view["engine_evidence_ids"] = list(self.engine_evidence_ids)
        return view


class RoleProvenanceLedger:
    """役職由来情報（エンジン発行）と公開申告を分けて保持する台帳。"""

    def __init__(self) -> None:
        self.evidence: list[RoleEvidence] = []
        self.interpretations: list[PublicInterpretation] = []
        # 中立カウンタ（公開メトリクスへ出してよい数だけ持つ）
        self.role_claims_recorded = 0
        self.verdict_lines_recorded = 0
        self.role_claims_disallowed = 0
        # 管理者専用カウンタ（真偽の内訳）
        self.engine_supported_verdict_lines = 0
        self.unsupported_role_evidence_lines = 0

    # ------------------------------------------------------------------
    # エンジン発行情報の登録
    # ------------------------------------------------------------------
    def record_engine_divination(
        self,
        owner: str,
        day: int,
        target: str,
        result: str,
    ) -> str:
        """エンジンが決定した占い結果を登録し、証拠IDを返す（同一再実行は統合）。"""
        evidence_id = divination_evidence_id(day, owner, target)
        existing = next(
            (item for item in self.evidence if item.evidence_id == evidence_id),
            None,
        )
        if existing is not None:
            if existing.result != result:
                # 同じ夜の対象で判定が変わった記録は上書きせず別行に残す
                self.evidence.append(
                    RoleEvidence(
                        evidence_id=f"{evidence_id}#2",
                        day=day,
                        owner=owner,
                        target=target,
                        result=result,
                    )
                )
            return evidence_id
        self.evidence.append(
            RoleEvidence(
                evidence_id=evidence_id,
                day=day,
                owner=owner,
                target=target,
                result=result,
            )
        )
        return evidence_id

    def engine_divination_match(
        self,
        claimant: str,
        actor_role: str | None,
        target: str | None,
        result: str | None,
    ) -> tuple[bool, tuple[str, ...]]:
        """申告された占い結果が本人のエンジン発行情報と一致するかを返す。

        本人が実際に占い師で、同一 (target, result) のエンジン発行記録がある場合だけ
        True になる。他者の結果を引用した場合や推測は一致扱いにしない。
        """
        if target is None or result is None or actor_role != "seer":
            return False, ()
        matches = tuple(
            item.evidence_id
            for item in self.evidence
            if item.owner == claimant and item.target == target and item.result == result
        )
        return bool(matches), matches

    # ------------------------------------------------------------------
    # 公開申告の登録（CO行・結果行）
    # ------------------------------------------------------------------
    def record_claim(
        self,
        *,
        claim_ref: str,
        day: int,
        claimant: str,
        claimed_role: str,
        source: str,
        actor_role: str | None = None,
        target: str | None = None,
        result: str | None = None,
        discussion_turn: int = 0,
    ) -> PublicInterpretation:
        """公開されたCO行・結果行を出所付きで登録する（本文・口調には干渉しない）。"""
        supported, evidence_ids = self.engine_divination_match(
            claimant=claimant,
            actor_role=actor_role,
            target=target,
            result=result,
        )
        if target is not None and result is not None:
            category = CATEGORY_ENGINE_ISSUED if supported else CATEGORY_UNSUPPORTED
            self.verdict_lines_recorded += 1
            if supported:
                self.engine_supported_verdict_lines += 1
            else:
                self.unsupported_role_evidence_lines += 1
        else:
            category = CATEGORY_SELF_REPORTED
        self.role_claims_recorded += 1
        interpretation = PublicInterpretation(
            claim_ref=claim_ref,
            day=day,
            claimant=claimant,
            claimed_role=claimed_role,
            target=target,
            result=result,
            source=normalize_provenance_source(source),
            category=category,
            engine_supported=supported,
            engine_evidence_ids=evidence_ids,
            discussion_turn=discussion_turn,
        )
        self.interpretations.append(interpretation)
        return interpretation

    def record_disallowed_claim(
        self,
        *,
        day: int,
        claimant: str,
        claimed_role: str,
        source: str,
        target: str | None = None,
        result: str | None = None,
        discussion_turn: int = 0,
    ) -> PublicInterpretation:
        """allow_bluff_role_claim=False で受理しなかった行を記録する（管理者専用）。"""
        self.role_claims_disallowed += 1
        interpretation = PublicInterpretation(
            claim_ref=f"[disallowed:{day}:{claimant}]",
            day=day,
            claimant=claimant,
            claimed_role=claimed_role,
            target=target,
            result=result,
            source=normalize_provenance_source(source),
            category=CATEGORY_DISALLOWED,
            engine_supported=False,
            engine_evidence_ids=(),
            discussion_turn=discussion_turn,
        )
        self.interpretations.append(interpretation)
        return interpretation

    # ------------------------------------------------------------------
    # 参照
    # ------------------------------------------------------------------
    def claim_by_ref(self, claim_ref: str) -> PublicInterpretation | None:
        for item in self.interpretations:
            if item.claim_ref == claim_ref:
                return item
        return None

    def prior_verdict_by(self, claimant: str, target: str | None = None) -> str | None:
        """そのプレイヤーが既に申告した判定（設定で不受理にした分の補完参照用）。"""
        for item in reversed(self.interpretations):
            if item.claimant != claimant or item.result is None:
                continue
            if target is not None and item.target != target:
                continue
            return item.result
        return None

    def own_claimed_roles(self, player_name: str) -> list[str]:
        """そのプレイヤーが公開した役職COを一覧化する（本人向けの一貫性情報）。"""
        return [
            item.claimed_role
            for item in self.interpretations
            if item.claimant == player_name
        ]

    # ------------------------------------------------------------------
    # メトリクスとシリアライズ
    # ------------------------------------------------------------------
    def metrics(self) -> dict[str, int]:
        """公開ログへ出してよい中立カウンタだけ返す。"""
        return {
            "role_claims_recorded": self.role_claims_recorded,
            "verdict_lines_self_reported": self.verdict_lines_recorded,
            "role_claims_disallowed": self.role_claims_disallowed,
        }

    def admin_metrics(self) -> dict[str, object]:
        """管理者用。エンジン発行情報と照合できた行・できなかった行の内訳を含める。"""
        admin: dict[str, object] = dict(self.metrics())
        admin.update(
            {
                "engine_supported_verdict_lines": self.engine_supported_verdict_lines,
                "unsupported_role_evidence_lines": self.unsupported_role_evidence_lines,
                "evidence_records": len(self.evidence),
                "claim_records": len(self.interpretations),
            }
        )
        per_claimant: dict[str, dict[str, int]] = {}
        for item in self.interpretations:
            entry = per_claimant.setdefault(
                item.claimant,
                {"claims": 0, "verdict_lines": 0, "unsupported_role_evidence": 0},
            )
            entry["claims"] += 1
            if item.target is not None and item.result is not None:
                entry["verdict_lines"] += 1
            if item.category == CATEGORY_UNSUPPORTED:
                entry["unsupported_role_evidence"] += 1
        admin["per_claimant"] = per_claimant
        return admin

    def serialize_public(self) -> dict[str, object]:
        """公開ログ用。真偽を示すキーは含まない（テストで検証する）。

        allow_bluff_role_claim=False で受理しなかった行は載せない。受理可否を役職名つきで
        公開すると「そのプレイヤーが本当にその役職でないこと」が外から判ってしまうため、
        中立カウンタ（role_claims_disallowed）だけを立てて中身は管理者ビュー専用にする。
        """
        published = [
            item
            for item in self.interpretations
            if item.category != CATEGORY_DISALLOWED
        ]
        return {
            "interpretation_label": PUBLIC_LINE_LABEL,
            "metrics": self.metrics(),
            "claims": [item.public_view() for item in published],
        }

    def serialize_admin(self) -> dict[str, object]:
        """管理者ログ用。エンジン発行情報の保有者と照合結果を含める。"""
        return {
            "metrics": self.admin_metrics(),
            "evidence": [item.to_dict() for item in self.evidence],
            "claims": [item.admin_view() for item in self.interpretations],
        }




