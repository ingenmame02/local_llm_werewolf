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
    # 複数列一人称（我々・私たち等）は人狼陣営の内部視点を示す。単数一人称の
    # `_SELF`へ混入すると、村人が漏洩発言を引用した文まで誤検出するため分離する（TASK-035）。
    _TEAM_SELF = r"(?:我々|われわれ|私たち|わたしたち|我らの|僕ら|ぼくら|俺ら|we|our|ours|us)"
    # 陣営視点の一人称＋人狼語＋行動語（「我々人狼としてどう行動すべきか」等）。
    # 行動語を直後付近のみ要求し、引用・懸念の文は検出しない。
    _TEAM_ACTION_VIEW = re.compile(
        rf"{_TEAM_SELF}[^。\n!?！？]{{0,15}}(?:人狼|狼|werewolf|wolf)\s*(?:陣営)?(?:として|の)?"
        r"[^。\n!?！？]{0,20}"
        r"(?:行動|戦略|作戦|方針|立ち回|相談|投票|狙|排除|勝|should|act|vote|kill|exile)",
        re.IGNORECASE,
    )
    # 一人称を伴わなくても、真役職の陣営視点で行動を語る発言
    # （「人狼としての行動戦略を示すため」等）を人狼・狂人の公開発言で検出する。
    _ROLE_ACTION_VIEW = re.compile(
        r"(?:人狼|狼|werewolf|wolf)\s*(?:陣営)?(?:として(?:の)?|の)\s*"
        r"(?:行動|戦略|作戦|方針|立ち回|相談|狙|排除)",
        re.IGNORECASE,
    )
    # 英語の陣営視点発言（"as wolves we should vote..."等）を人狼・狂人で検出する。
    _ENGLISH_TEAM_ACTION = re.compile(
        r"\bas\s+(?:a\s+|an\s+|the\s+)?(?:werewolves|werewolf|wolves|wolf team|wolf camp)\b"
        r"[^.!?\n]{0,25}\b(?:act|vote|attack|kill|exile|plan|strategy|should)\b|"
        r"\b(?:act|vote|attack|kill|exile|plan|strategy|should|action|strategy)\b"
        r"[^.!?\n]{0,25}\bas\s+(?:a\s+|an\s+|the\s+)?(?:werewolves|werewolf|wolves|wolf)\b|"
        r"\bas\s+(?:a\s+|an\s+|the\s+)?(?:werewolves|werewolf|wolves|wolf)\b"
        r"[^.!?\n]{0,25}\b(?:action|strategy|plan)\b|"
        rf"{_TEAM_SELF}\b[^.!?\n]{{0,25}}\b(?:werewolves|werewolf|wolves|wolf)\b"
        r"[^.!?\n]{0,25}\b(?:should|vote|act|attack|kill|exile)\b",
        re.IGNORECASE,
    )
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
    # 未公開の占い結果への言及（「P1は人間と占われました」型の自然文も含む）を
    # 検出する。判定語の直後の「占われました/占われた」も漏洩として拾う（TASK-039）。
    _HIDDEN_RESULT = re.compile(
        r"(?:占い|占わ|占っ|判定|結果)[^。\n]{0,20}(?:人狼|狼|人間|白|黒)|"
        r"(?:人狼|狼|人間|白|黒)[^。\n]{0,20}(?:占い|占わ|占っ|判定|結果)|"
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
    _SAFE_FALLBACK = "I will hold my judgment based on public information for now."

    @classmethod
    def allows_public_claim(cls, content: str | None, actor: Player) -> bool:
        from .engine import extract_public_role_claim

        if content is None or not isinstance(content, str):
            return False
        claim_text = re.sub(r"^\s*【(?:CO|本文)】\s*", "", content)
        claim = extract_public_role_claim(claim_text, actor.name)
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
        content: str | None,
        actor: Player,
        state: GameState,
        *,
        allowed_public_claim: bool = False,
        allowed_public_results: frozenset[tuple[str, str]] = frozenset(),
    ) -> tuple[LeakFinding, ...]:
        findings: list[LeakFinding] = []
        if content is None or not isinstance(content, str):
            return ()
        if cls._META_ROLE.search(content):
            findings.append(LeakFinding("strategy_leak", "self_role_meta_disclosure"))
        elif not allowed_public_claim and (
            cls._ROLE_ASSERTION.search(content)
            or cls._ENGLISH_ROLE_ASSERTION.search(content)
        ):
            findings.append(LeakFinding("secret_role_leak", "self_role_assertion"))

        # 自分のプレイヤー名で自分を役職付きの三人称として論じる発言（「Player 1が
        # 偽の占い師か」等）は、正当なCO形式（`allows_public_claim`）でない限り漏洩・
        # メタ発言として扱う（TASK-039）。他人の名前や引用は対象外。
        if not allowed_public_claim and actor.name:
            name_self = rf"{re.escape(actor.name)}(?:さん)?"
            if re.search(
                rf"{name_self}[^。\n!?！？]{{0,20}}{cls._ROLE}"
                rf"(?:陣営|です|だ|である|として|陣営の一員)"
                rf"|{name_self}[^.\n!?!?]{{0,20}}(?i:\b(?:is|was|seems|must be)\b\s*(?:a\s+|the\s+)?(?:fake\s+)?(?:werewolves|werewolf|wolf|seer|villager|madman)\b)",
                content,
                re.IGNORECASE,
            ):
                findings.append(
                    LeakFinding("secret_role_leak", "third_person_self_role_assertion")
                )
            if re.search(
                rf"{name_self}[^。\n!?！？]{{0,45}}"
                rf"(?:設定上|設定では|本当(?:は|の)|実際(?:は|の)|真の役職|"
                rf"騙(?:り|って|っている)|偽(?:って|の)|ロールプレイ)"
                rf"|{name_self}[^.\n!?!?]{{0,45}}(?i:\b(?:pretending|bluffing|faking|role[- ]playing|playing as|actual role|true role|system prompt)\b)",
                content,
                re.IGNORECASE,
            ):
                findings.append(
                    LeakFinding("strategy_leak", "third_person_self_role_meta")
                )

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

        # 人狼陣営に所属する役職（人狼・狂人）の公開発言では、陣営視点で行動を述べる
        # 表現を漏洩として扱う。村人の「人狼を疑う」等の推測は対象外（TASK-035）。
        if (
            not allowed_public_claim
            and actor.role in {Role.WOLF, Role.MADMAN}
            and (
                cls._TEAM_ACTION_VIEW.search(content)
                or cls._ROLE_ACTION_VIEW.search(content)
                or cls._ENGLISH_TEAM_ACTION.search(content)
            )
        ):
            findings.append(LeakFinding("secret_team_leak", "wolf_team_action_view"))

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
