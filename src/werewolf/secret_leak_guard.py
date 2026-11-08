import hashlib
import re
from dataclasses import dataclass

from .models import GameState, Player, Role


@dataclass(frozen=True)
class LeakFinding:
    category: str
    rule_id: str


class SecretLeakGuard:
    _SELF = r"(?:私|僕|ぼく|俺|自分|わたし|わたくし|I|I'm|I am|my)"
    _ROLE = r"(?:人狼|狼|狂人|占い師|村人|werewolf|wolf|seer|villager|madman)"
    _META_ROLE = re.compile(
        rf"{_SELF}[^。\n!?！？]{{0,45}}"
        r"(?:設定上|設定では|本当(?:は|の)|実際(?:は|の)|真の役職|"
        r"騙(?:り|って|っている)|偽(?:って|の)|ロールプレイ|as an AI)|"
        r"\b(?:I['’]m|I am)\b[^.!?\n]{0,45}"
        r"\b(?:pretending|bluffing|faking|role-playing|playing as)\b",
        re.IGNORECASE,
    )
    _ROLE_ASSERTION = re.compile(
        rf"{_SELF}[^。\n!?！？]{{0,20}}{_ROLE}"
        r"(?:陣営|です|だ|である|として|陣営の一員)",
        re.IGNORECASE,
    )
    _ENGLISH_ROLE_ASSERTION = re.compile(
        r"\b(?:I['’]m|I am|my role is)\s+(?:a\s+)?"
        r"(?:werewolf|wolf|seer|villager|madman)\b",
        re.IGNORECASE,
    )
    _TEAM_LEAK = re.compile(
        rf"{_SELF}[^。\n!?！？]{{0,35}}"
        r"(?:相方|仲間の人狼|狼相談|人狼チャット|狼陣営|"
        r"my (?:wolf )?partner|wolf chat)",
        re.IGNORECASE,
    )
    _HIDDEN_RESULT = re.compile(
        r"(?:占い|判定|結果)[^。\n]{0,20}(?:人狼|狼|人間|白|黒)|"
        r"(?:人狼|狼|人間|白|黒)[^。\n]{0,20}(?:占い|判定|結果)|"
        r"\b(?:seer|scan|divination|result)\b[^.!?\n]{0,35}"
        r"\b(?:werewolf|wolf|human|villager)\b|"
        r"\b(?:werewolf|wolf|human|villager)\b[^.!?\n]{0,35}"
        r"\b(?:seer|scan|divination|result)\b",
        re.IGNORECASE,
    )
    _PROMPT_CONFIG = re.compile(
        r"(?:system prompt|システムプロンプト|内部プロンプト|"
        r"モデル設定|生成設定|評価シナリオ|num_ctx|num_predict|"
        r"temperature|tool call|ツール呼び出し|プロンプトに従い|"
        r"プロンプトで指示|モデルとして)",
        re.IGNORECASE,
    )
    _INTERNAL_CUE = re.compile(
        r"(?:内部|秘匿|秘密|設定|生成|評価|実装|シミュレーション|"
        r"プロンプト|モデル|システム|ツール|本当|実際|騙り|騙って|"
        r"狼相談|相方|my partner|prompt|model|simulation|as an AI)",
        re.IGNORECASE,
    )
    _SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])\s*|\n+")
    _SAFE_FALLBACK = "現時点では公開情報に基づく判断を保留します。"

    @classmethod
    def allows_public_claim(cls, content: str, actor: Player) -> bool:
        from .engine import extract_public_role_claim

        claim_text = re.sub(r"^\s*【(?:CO|本文)】\s*", "", content)
        claim = extract_public_role_claim(claim_text)
        if claim is None:
            english_claim = re.match(
                r"\s*(?:I['’]m|I am|my role is)\s+(?:a\s+)?"
                r"(?P<role>werewolf|wolf|seer|villager|madman)"
                r"(?:\s+(?P<explicit>CO|claim))?(?=\s|[.!?]|$)",
                claim_text,
                re.IGNORECASE,
            )
            if english_claim is not None:
                role = english_claim.group("role").casefold()
                explicit_co = english_claim.group("explicit") is not None
                if role in {"wolf", "werewolf"} and actor.role is Role.WOLF:
                    return explicit_co
                return True
        if claim is None:
            return False
        if (
            actor.role is Role.WOLF
            and claim == Role.WOLF.value
            and not re.search(r"(?:人狼|狼)\s*CO", content, re.IGNORECASE)
        ):
            return False
        return True

    @classmethod
    def inspect(
        cls,
        content: str,
        actor: Player,
        state: GameState,
        *,
        allowed_public_claim: bool = False,
        allowed_public_results: frozenset[tuple[str, str]] = frozenset(),
    ) -> tuple[LeakFinding, ...]:
        findings: list[LeakFinding] = []
        if cls._META_ROLE.search(content):
            findings.append(LeakFinding("strategy_leak", "self_role_meta_disclosure"))
        elif not allowed_public_claim and (
            cls._ROLE_ASSERTION.search(content)
            or cls._ENGLISH_ROLE_ASSERTION.search(content)
        ):
            findings.append(LeakFinding("secret_role_leak", "self_role_assertion"))

        if cls._TEAM_LEAK.search(content):
            findings.append(LeakFinding("secret_team_leak", "private_team_reference"))
        elif (
            actor.role is Role.WOLF
            and actor.partner
            and actor.partner in content
            and re.search(r"(?:相方|仲間|狼|wolf|partner)", content, re.IGNORECASE)
            and re.search(r"(?:私|僕|俺|自分|my|秘密|内部)", content, re.IGNORECASE)
        ):
            findings.append(LeakFinding("secret_team_leak", "wolf_partner_identity"))

        if actor.role is Role.SEER:
            public_results = {
                (claim.target, claim.result)
                for claim in state.seer_claims
                if claim.claimant == actor.name
            }
            public_results.update(
                (claim.target, claim.result)
                for claim in state.public_role_claims
                if claim.claimant == actor.name
            )
            public_results.update(allowed_public_results)
            for target, result in state.seer_results.items():
                if (target, result) in public_results:
                    continue
                result_terms = (
                    ("人狼", "狼", "黒", "黒判定", "werewolf", "wolf")
                    if result == "wolf"
                    else ("人間", "白", "白判定", "human", "villager")
                )
                if target in content and cls._HIDDEN_RESULT.search(content) and any(
                    term in content for term in result_terms
                ):
                    findings.append(
                        LeakFinding("hidden_result_leak", "unpublished_seer_result")
                    )
                    break

        if cls._PROMPT_CONFIG.search(content) and cls._INTERNAL_CUE.search(content):
            findings.append(
                LeakFinding("prompt_or_config_leak", "internal_configuration_reference")
            )
        elif actor.model and actor.model.casefold() in content.casefold():
            findings.append(LeakFinding("prompt_or_config_leak", "model_identifier"))

        return tuple(findings)

    @classmethod
    def redact(
        cls,
        content: str,
        actor: Player,
        state: GameState,
        *,
        allowed_public_results: frozenset[tuple[str, str]] = frozenset(),
    ) -> tuple[str, tuple[LeakFinding, ...]]:
        kept: list[str] = []
        all_findings: list[LeakFinding] = []
        for segment in cls._SENTENCE_SPLIT.split(content):
            if not segment.strip():
                continue
            allowed_claim = cls.allows_public_claim(segment, actor)
            findings = cls.inspect(
                segment,
                actor,
                state,
                allowed_public_claim=allowed_claim,
                allowed_public_results=allowed_public_results,
            )
            if findings:
                all_findings.extend(findings)
            else:
                kept.append(segment.strip())
        redacted = "\n".join(kept).strip()
        return redacted or cls._SAFE_FALLBACK, tuple(
            dict.fromkeys(all_findings)
        )

    @staticmethod
    def content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
