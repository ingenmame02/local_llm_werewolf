import hashlib
import re
from dataclasses import dataclass, field

from .leak_gate_policy import (
    DEFAULT_LEAK_GUARD_POLICY,
    GATE_HARD,
    GATE_OBSERVE,
    GATE_SOFT,
    LEAK_POLICY_LEGACY_STRICT,
    RULE_INTERNAL_CONFIG,
    RULE_LEGACY_SELF_ROLE_ASSERTION,
    RULE_LEGACY_THIRD_PERSON_SELF,
    RULE_MEMO_ONLY,
    RULE_MODEL_IDENTIFIER,
    RULE_PARTNER_CO_MENTION,
    RULE_PARTNER_IDENTITY,
    RULE_PRIVATE_TEAM_REFERENCE,
    RULE_PUBLIC_CLAIM_RESTATEMENT,
    RULE_PUBLISHED_VERDICT_RESTATEMENT,
    RULE_SELF_ROLE_META,
    RULE_SELF_ROLE_PUBLIC_CLAIM,
    RULE_SELF_ROLE_TRUE_CAMP,
    RULE_TEAM_ACTION_VIEW,
    RULE_THIRD_PARTY_SPECULATION,
    RULE_THIRD_PERSON_SELF_META,
    RULE_THIRD_PERSON_SELF_STYLE,
    RULE_THIRD_PERSON_SELF_TRUE_CAMP,
    RULE_UNPUBLISHED_SEER_RESULT,
    STRICT_REPAIRED_SOFT_RULES,
    STRENGTH_BALANCED,
    STRENGTH_STRICT,
    LeakGuardProfiles,
    PublicFactIndex,
    gate_of,
    normalize_leak_guard_policy,
)
from .models import GameState, Player, Role

# 本文保持の下限（この比率を切る場合は全文再生成・赤黒フォールバックへ回す）。
MIN_BODY_PRESERVATION_RATIO = 0.6
# メタ発言・システム情報への言及は中立化よりも除去が安全なルール。
DROP_ONLY_RULES: frozenset[str] = frozenset(
    {
        RULE_INTERNAL_CONFIG,
        RULE_MODEL_IDENTIFIER,
        RULE_SELF_ROLE_META,
        RULE_THIRD_PERSON_SELF_META,
        RULE_LEGACY_SELF_ROLE_ASSERTION,
        RULE_LEGACY_THIRD_PERSON_SELF,
    }
)


@dataclass(frozen=True)
class LeakFinding:
    """漏洩検出結果。TASK-054でゲート・検出根拠・公開済み事実IDを持つようになった。

    `evidence` / `fact_ids` / 位置情報は比較対象から外しているため、既存の
    `dict.fromkeys(findings)` による重複排除（category + rule_id）は従来どおり動く。
    """

    category: str
    rule_id: str
    gate: str = GATE_HARD
    evidence: str = field(default="", compare=False, repr=False)
    fact_ids: tuple[str, ...] = field(default=(), compare=False)
    region: str = field(default="public", compare=False, repr=False)
    char_start: int = field(default=-1, compare=False, repr=False)
    char_end: int = field(default=-1, compare=False, repr=False)

    @property
    def is_hard(self) -> bool:
        return self.gate == GATE_HARD


@dataclass(frozen=True)
class LeakRegion:
    """発言内容の領域。公開面（CO・結果・本文）と非公開メモを分離する。"""

    kind: str
    is_public: bool
    start: int
    end: int


@dataclass(frozen=True)
class LeakClassification:
    """TASK-054の分類結果。hardは遮断、softは警告のみ、observeは観測だけ行う。"""

    policy: str
    strength: str
    hard: tuple[LeakFinding, ...] = ()
    soft: tuple[LeakFinding, ...] = ()
    observe: tuple[LeakFinding, ...] = ()
    memo_rules: tuple[str, ...] = ()
    legacy_hard_rules: tuple[str, ...] = ()
    fact_ids: tuple[str, ...] = ()
    public_regions: int = 0

    @property
    def has_hard(self) -> bool:
        return bool(self.hard)

    @property
    def hard_rules(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(finding.rule_id for finding in self.hard))

    @property
    def soft_rules(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(finding.rule_id for finding in self.soft))

    @property
    def observe_rules(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(finding.rule_id for finding in self.observe))

    @property
    def false_positives_avoided(self) -> bool:
        """legacy_strictなら遮断されたはずの指摘を、現ポリシーで保持できたか。"""
        return bool(self.legacy_hard_rules) and not bool(self.hard)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(
            sorted({finding.category for finding in (*self.hard, *self.soft, *self.observe)})
        )


@dataclass(frozen=True)
class LeakRepair:
    """漏洩文だけを差し替えた結果（全文再生成を避け本文保持率を上げる）。"""

    text: str
    replaced_rules: tuple[str, ...] = ()
    dropped_rules: tuple[str, ...] = ()
    edit_count: int = 0
    original_char_count: int = 0
    kept_char_count: int = 0
    leaked_char_count: int = 0
    preserved_ratio: float = 1.0
    body_preservable: bool = True
    edits: tuple[tuple[str, str], ...] = ()

    @property
    def applied(self) -> bool:
        return self.edit_count > 0


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

    # ------------------------------------------------------------------
    # TASK-054: 公開面の領域分割と文単位スパン
    # ------------------------------------------------------------------
    _SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("co", re.compile(r"^\s*(?:【CO】|\[CO\])", re.IGNORECASE)),
        ("result", re.compile(r"^\s*(?:【結果】|\[RESULT\])", re.IGNORECASE)),
        ("memo", re.compile(r"^\s*(?:【推理メモ】|\[REASONING\])", re.IGNORECASE)),
        ("body", re.compile(r"^\s*(?:【本文】|\[BODY\])", re.IGNORECASE)),
    )
    _SENTENCE_CHUNK = re.compile(r"[^。.!?!?\n]+[。.!?!?\n]*")
    # 自分の役職を否定する形（「私は騙りではありません」）は開示ではないため対象外にする。
    _META_NEGATION = re.compile(
        r"(?:騙(?:り|りが)?|偽(?:り|い)?|pretending|bluffing|faking)"
        r"[^。.!?!?\n]{0,10}"
        r"(?:ではありません|じゃない|ではない|ではありません|でなく|"
        r"not\b|isn'?t\b|don'?t\b|never\b)",
        re.IGNORECASE,
    )
    # 他者役職の推測語（合法な議論）。誤検出を減らし、観測だけ残す。
    _SPECULATION = re.compile(
        r"(?:かもしれない|かも|らしい|っぽ|可能性|懸念|疑わ|注視|警戒|では["
        r"ない]|と見る|思う|想定|仮に|候補|"
        r"\bmaybe\b|\bprobably\b|\blikey?\b|\blikely\b|\bcould be\b|\bseems?\b|"
        r"\bsuspect\w*\b|\bsuspicion\b|\bmight\b)",
        re.IGNORECASE,
    )
    # 仲間関係を明示する語（同じ文内で名前と結合した場合のみ hard gate）。
    _ALLY_BINDING = re.compile(
        r"(?:相方|仲間の人狼|狼の仲間|狼と仲間|味方|組ん(?:で|んでいる)|"
        r"内通|協力関係|my (?:wolf )?partner|wolf partner|my ally|our team)",
        re.IGNORECASE,
    )
    _CAMP_HINT = re.compile(r"(?:人狼|狼|werewolf|wolf|陣営|夜|襲撃)", re.IGNORECASE)

    @classmethod
    def regions(cls, content: str | None) -> tuple[LeakRegion, ...]:
        """応答を公開面（CO・結果・本文）と非公開メモ（推理メモ）の領域へ分割する。

        `[REASONING]` / `【推理メモ】` は公開メッセージへ載らないため、ここでの検出は
        漏洩ゲートにしない（TASK-054の誤検出7件中3件がメモ内だった）。
        """
        if not isinstance(content, str) or not content:
            return ()
        regions: list[LeakRegion] = []
        kind = "plain"
        start = 0
        offset = 0
        for line in content.splitlines(keepends=True):
            section_kind = None
            stripped = line.strip()
            if stripped:
                for candidate, pattern in cls._SECTION_PATTERNS:
                    if pattern.match(line):
                        section_kind = candidate
                        break
            if section_kind is not None and section_kind != kind:
                if offset > start:
                    regions.append(
                        LeakRegion(kind, kind != "memo", start, offset)
                    )
                kind = section_kind
                start = offset
            offset += len(line)
        if offset > start:
            regions.append(LeakRegion(kind, kind != "memo", start, offset))
        elif not regions:
            regions.append(LeakRegion(kind, kind != "memo", start, offset))
        return tuple(regions)

    @staticmethod
    def _sentence_spans(text: str, base: int) -> tuple[tuple[int, int, str], ...]:
        """領域内から（絶対開始, 絶対終了, 文）の組を抽出する（改行・余白は保持）。"""
        spans: list[tuple[int, int, str]] = []
        for match in SecretLeakGuard._SENTENCE_CHUNK.finditer(text):
            raw = match.group(0)
            stripped = raw.strip()
            if not stripped:
                continue
            lead = len(raw) - len(raw.lstrip())
            tail = len(raw) - len(raw.rstrip())
            start = match.start() + lead
            end = match.end() - tail
            spans.append(
                (
                    base + match.start() + lead,
                    base + match.end() - tail,
                    raw[lead : len(raw) - tail],
                )
            )
        return tuple(spans)

    @classmethod
    def _asserted_roles(cls, sentence: str) -> tuple[str, ...]:
        """文中の役職語を正規化して並べる（他者CO言でも出るため名前の照合と併用する）。"""
        found: list[str] = []
        lowered = sentence.casefold()
        for token, role in (
            ("人狼", Role.WOLF.value),
            ("狼", Role.WOLF.value),
            ("werewolf", Role.WOLF.value),
            ("wolf", Role.WOLF.value),
            ("狂人", Role.MADMAN.value),
            ("madman", Role.MADMAN.value),
            ("占い師", Role.SEER.value),
            ("seer", Role.SEER.value),
            ("村人", Role.VILLAGER.value),
            ("villager", Role.VILLAGER.value),
        ):
            if token in lowered and role not in found:
                found.append(role)
        return tuple(found)

    @staticmethod
    def _name_mentioned(sentence: str, name: str | None) -> bool:
        """プレイヤー名（別称含む）が文中に出るか。`プレイヤー2` / `P2` も同一視する。"""
        if not name or not isinstance(sentence, str):
            return False
        from .engine import _player_name_pattern  # 循環import回避のため遅延読み込み

        return bool(
            re.search(_player_name_pattern(name), sentence, re.IGNORECASE)
        )

    @classmethod
    def _role_gate(
        cls,
        actor: Player,
        asserted_roles: tuple[str, ...],
        facts: PublicFactIndex,
    ) -> tuple[str, str, tuple[str, ...]]:
        """役職言及のゲートを「真情報の開示 / 合法な自己申告 / 公開済み再言及」へ分離する。"""
        hidden_camp = {Role.WOLF.value, Role.MADMAN.value}
        for role in asserted_roles:
            if actor.role in hidden_camp and role == actor.role.value:
                return GATE_HARD, RULE_SELF_ROLE_TRUE_CAMP, ()
        for role in asserted_roles:
            fact_ids = facts.role_fact_ids(actor.name, role)
            if fact_ids:
                return GATE_OBSERVE, RULE_PUBLIC_CLAIM_RESTATEMENT, fact_ids
        return GATE_SOFT, RULE_SELF_ROLE_PUBLIC_CLAIM, ()

    # ------------------------------------------------------------------
    # レガシー（legacy_strict）: TASK-054以前の全文照合をそのまま再現する
    # ------------------------------------------------------------------
    @classmethod
    def _legacy_findings(
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

    # ------------------------------------------------------------------
    # TASK-054: ゲート付き分類（公開面のみ・文単位）
    # ------------------------------------------------------------------
    @classmethod
    def classify(
        cls,
        content: str | None,
        actor: Player,
        state: GameState,
        *,
        allowed_public_claim: bool = False,
        allowed_public_results: frozenset[tuple[str, str]] = frozenset(),
        guard_policy: object = None,
        guard_strength: str = STRENGTH_BALANCED,
    ) -> LeakClassification:
        """発言を hard / soft / observe へ分離する（TASK-054実装案1・2）。

        - 非公開の推理メモはゲートにせず観測だけ残す。
        - FactLedger / ClaimLedger で裏付く役職・判定の再言及は observe（事実IDを記録）。
        - 他者への役職推測は常に合法（observe）として扱い、議論の自由を削らない。
        """
        policy = normalize_leak_guard_policy(
            guard_policy
            if guard_policy is not None
            else getattr(state, "leak_guard_policy", DEFAULT_LEAK_GUARD_POLICY)
        )
        strength = (
            STRENGTH_STRICT
            if guard_strength == STRENGTH_STRICT
            else STRENGTH_BALANCED
        )
        if content is None or not isinstance(content, str) or not content.strip():
            return LeakClassification(policy=policy, strength=strength)
        facts = PublicFactIndex.build(state)
        legacy_findings = cls._legacy_findings(
            content,
            actor,
            state,
            allowed_public_claim=allowed_public_claim,
            allowed_public_results=allowed_public_results,
        )
        legacy_rules = tuple(dict.fromkeys(finding.rule_id for finding in legacy_findings))
        if policy == LEAK_POLICY_LEGACY_STRICT:
            return LeakClassification(
                policy=policy,
                strength=strength,
                hard=tuple(
                    LeakFinding(
                        finding.category,
                        finding.rule_id,
                        GATE_HARD,
                        evidence=content[:180],
                        fact_ids=facts.all_fact_ids,
                        region="legacy",
                    )
                    for finding in legacy_findings
                ),
                legacy_hard_rules=legacy_rules,
                fact_ids=facts.all_fact_ids,
                public_regions=1,
            )

        other_names = tuple(
            player.name
            for player in state.players
            if player.name != actor.name
        )
        hard: list[LeakFinding] = []
        soft: list[LeakFinding] = []
        observe: list[LeakFinding] = []
        memo_rules: list[str] = []
        public_regions = 0
        for region in cls.regions(content):
            text = content[region.start : region.end]
            if not text.strip():
                continue
            if not region.is_public:
                # 公開されないメモでの検出は観測だけ行う（次ターン以降の再生成要因にしない）。
                memo_findings = cls._legacy_findings(
                    text,
                    actor,
                    state,
                    allowed_public_claim=allowed_public_claim,
                    allowed_public_results=allowed_public_results,
                )
                for finding in memo_findings:
                    if RULE_MEMO_ONLY not in memo_rules:
                        memo_rules.append(RULE_MEMO_ONLY)
                    if finding.rule_id not in memo_rules:
                        memo_rules.append(finding.rule_id)
                continue
            public_regions += 1
            for start, end, sentence in cls._sentence_spans(text, region.start):
                findings = cls._tiered_sentence_findings(
                    sentence,
                    actor,
                    state,
                    facts,
                    other_names=other_names,
                    region=region.kind,
                    char_start=start,
                    char_end=end,
                    allowed_public_claim=allowed_public_claim,
                    allowed_public_results=allowed_public_results,
                )
                for finding in findings:
                    if finding.gate == GATE_HARD:
                        hard.append(finding)
                    elif finding.gate == GATE_SOFT:
                        soft.append(finding)
                    else:
                        observe.append(finding)
        fact_ids = tuple(
            dict.fromkeys(
                [*facts.all_fact_ids, *(fid for f in observe for fid in f.fact_ids)]
            )
        )
        return LeakClassification(
            policy=policy,
            strength=strength,
            hard=tuple(hard),
            soft=tuple(soft),
            observe=tuple(observe),
            memo_rules=tuple(memo_rules),
            legacy_hard_rules=legacy_rules,
            fact_ids=fact_ids,
            public_regions=public_regions,
        )


    @classmethod
    def _tiered_sentence_findings(
        cls,
        sentence: str,
        actor: Player,
        state: GameState,
        facts: PublicFactIndex,
        *,
        other_names: tuple[str, ...],
        region: str,
        char_start: int,
        char_end: int,
        allowed_public_claim: bool,
        allowed_public_results: frozenset[tuple[str, str]],
    ) -> tuple[LeakFinding, ...]:
        """公開面の1文だけを照合し、ゲート別に所見を返す（TASK-054実装案1）。"""
        text = sentence.strip()
        if not text:
            return ()
        findings: list[LeakFinding] = []

        def add(
            category: str,
            rule_id: str,
            gate: str,
            fact_ids: tuple[str, ...] = (),
        ) -> None:
            findings.append(
                LeakFinding(
                    category,
                    rule_id,
                    gate,
                    evidence=text[:200],
                    fact_ids=fact_ids,
                    region=region,
                    char_start=char_start,
                    char_end=char_end,
                )
            )

        claim_ok = allowed_public_claim or cls.allows_public_claim(text, actor)

        # システム・生成条件への言及は依然として致命的（同じ文内で両方の語を要求する）。
        if cls._PROMPT_CONFIG.search(text) and cls._INTERNAL_CUE.search(text):
            add("prompt_or_config_leak", RULE_INTERNAL_CONFIG, GATE_HARD)
        elif actor.model and actor.model.casefold() in text.casefold():
            add("prompt_or_config_leak", RULE_MODEL_IDENTIFIER, GATE_HARD)

        negated_meta = cls._META_NEGATION.search(text) is not None
        if not claim_ok and not negated_meta and cls._META_ROLE.search(text):
            add("strategy_leak", RULE_SELF_ROLE_META, GATE_HARD)

        if not claim_ok and actor.name:
            name_self = rf"{re.escape(actor.name)}(?:さん)?"
            third_person_role = re.search(
                rf"{name_self}[^。\n!?！？]{{0,20}}{cls._ROLE}"
                rf"(?:陣営|です|だ|である|として|陣営の一員)"
                rf"|{name_self}[^.\n!?!?]{{0,20}}"
                rf"(?i:\b(?:is|was|seems|must be)\b\s*(?:a\s+|the\s+)?"
                rf"(?:fake\s+)?(?:werewolves|werewolf|wolf|seer|villager|madman)\b)",
                text,
                re.IGNORECASE,
            )
            if third_person_role is not None:
                gate, rule_id, fact_ids = cls._third_person_gate(
                    actor, facts, cls._asserted_roles(text)
                )
                category = {
                    GATE_HARD: "secret_role_leak",
                    GATE_OBSERVE: "public_claim_note",
                }.get(gate, "style_note")
                add(category, rule_id, gate, fact_ids)
            if (
                not claim_ok
                and not negated_meta
                and re.search(
                    rf"{name_self}[^。\n!?！？]{{0,45}}"
                    rf"(?:設定上|設定では|本当(?:は|の)|実際(?:は|の)|真の役職|"
                    rf"騙(?:り|って|っている)|偽(?:って|の)|ロールプレイ)"
                    rf"|{name_self}[^.\n!?!?]{{0,45}}"
                    rf"(?i:\b(?:pretending|bluffing|faking|role[- ]playing|playing as"
                    rf"|actual role|true role|system prompt)\b)",
                    text,
                    re.IGNORECASE,
                )
            ):
                add("strategy_leak", RULE_THIRD_PERSON_SELF_META, GATE_HARD)


        # 一人称の役職宣言: 狼陣営の真役職の開示だけが hard、村側やブラフは soft / observe。
        if not claim_ok and (
            cls._ROLE_ASSERTION.search(text) or cls._ENGLISH_ROLE_ASSERTION.search(text)
        ):
            gate, rule_id, fact_ids = cls._role_gate(
                actor, cls._asserted_roles(text), facts
            )
            category = {
                GATE_HARD: "secret_role_leak",
                GATE_OBSERVE: "public_claim_note",
            }.get(gate, "in_game_role_claim")
            add(category, rule_id, gate, fact_ids)

        # 仲間関係は同じ文内で関係を明示した場合だけ遮断する。単なる共起（実機ログの
        # 「Player 2さんや私を疑う」型）は soft とし、別領域の語と結合して拾わない。
        if cls._TEAM_LEAK.search(text):
            add("secret_team_leak", RULE_PRIVATE_TEAM_REFERENCE, GATE_HARD)
        elif (
            actor.role is Role.WOLF
            and actor.partner
            and cls._name_mentioned(text, actor.partner)
        ):
            binding = cls._ALLY_BINDING.search(text) is not None
            first_person = bool(re.search(r"(?:私|僕|ぼく|俺|自分|わたし|I\b|my)", text))
            if binding and first_person:
                add("secret_team_leak", RULE_PARTNER_IDENTITY, GATE_HARD)
            elif first_person or cls._CAMP_HINT.search(text):
                add(
                    "secret_team_leak",
                    RULE_PARTNER_CO_MENTION,
                    GATE_SOFT,
                    facts.own_claim_fact_ids(actor.name),
                )

        if (
            not claim_ok
            and actor.role in {Role.WOLF, Role.MADMAN}
            and (
                cls._TEAM_ACTION_VIEW.search(text)
                or cls._ROLE_ACTION_VIEW.search(text)
                or cls._ENGLISH_TEAM_ACTION.search(text)
            )
        ):
            add("secret_team_leak", RULE_TEAM_ACTION_VIEW, GATE_HARD)

        # 占い判定は公開済みペアなら fact_id付きの再言及、未公開ペアだけ遮断する。
        if actor.role is Role.SEER:
            for target, result in state.seer_results.items():
                result_terms = (
                    ("人狼", "狼", "黒", "werewolf", "wolf")
                    if result == "wolf"
                    else ("人間", "白", "human", "villager")
                )
                if not cls._HIDDEN_RESULT.search(text):
                    continue
                if not cls._name_mentioned(text, target):
                    continue
                if not any(term in text for term in result_terms):
                    continue
                fact_ids = facts.verdict_fact_ids(target, result)
                if fact_ids or (target, result) in allowed_public_results:
                    add(
                        "published_fact_note",
                        RULE_PUBLISHED_VERDICT_RESTATEMENT,
                        GATE_OBSERVE,
                        fact_ids or (f"[allowed:{target}={result}]",),
                    )
                else:
                    add("hidden_result_leak", RULE_UNPUBLISHED_SEER_RESULT, GATE_HARD)
                break

        # 他者への役職推測は常に合法な議論として観測だけ残す（実装案1の分離）。
        if not findings and other_names and cls._SPECULATION.search(text):
            roles = cls._asserted_roles(text)
            if roles and any(cls._name_mentioned(text, name) for name in other_names):
                add("role_speculation_note", RULE_THIRD_PARTY_SPECULATION, GATE_OBSERVE)

        return tuple(findings)

    @classmethod
    def _third_person_gate(
        cls,
        actor: Player,
        facts: PublicFactIndex,
        asserted_roles: tuple[str, ...],
    ) -> tuple[str, str, tuple[str, ...]]:
        """三人称の自己参照を「真陣営の開示 / 公開済みCOの再言及 / 言い回し」へ分離する。"""
        hidden_camp = {Role.WOLF.value, Role.MADMAN.value}
        for role in asserted_roles:
            if actor.role in hidden_camp and role == actor.role.value:
                return GATE_HARD, RULE_THIRD_PERSON_SELF_TRUE_CAMP, ()
        fact_ids = facts.own_claim_fact_ids(actor.name)
        if fact_ids:
            return GATE_OBSERVE, RULE_PUBLIC_CLAIM_RESTATEMENT, fact_ids
        return GATE_SOFT, RULE_THIRD_PERSON_SELF_STYLE, ()

    @classmethod
    def inspect(
        cls,
        content: str | None,
        actor: Player,
        state: GameState,
        *,
        allowed_public_claim: bool = False,
        allowed_public_results: frozenset[tuple[str, str]] = frozenset(),
        guard_policy: object = None,
        guard_strength: str = STRENGTH_BALANCED,
    ) -> tuple[LeakFinding, ...]:
        """致命的（hard）所見だけを返す。soft/observe は公開を止めない。

        `guard_policy=\"legacy_strict\"` は TASK-054以前の全文照合をそのまま再現するので、
        誤検出による再生成負荷の A/B 比較に使える。
        """
        return cls.classify(
            content,
            actor,
            state,
            allowed_public_claim=allowed_public_claim,
            allowed_public_results=allowed_public_results,
            guard_policy=guard_policy,
            guard_strength=guard_strength,
        ).hard


    # ------------------------------------------------------------------
    # TASK-054実装案3: 漏洩文だけを選ぶ補正（全文再生成・定型文置換を避ける）
    # ------------------------------------------------------------------
    _NEUTRAL_REPLACEMENTS: dict[str, tuple[str, str]] = {
        RULE_SELF_ROLE_TRUE_CAMP: (
            "I will speak about my position through future in-game claims.",
            "私の立場は、今後のゲーム内COと行動で明らかにします。",
        ),
        RULE_THIRD_PERSON_SELF_TRUE_CAMP: (
            "I will state my own position about myself in later claims.",
            "自分の立場は、今後のCOの形で述べます。",
        ),
        RULE_PARTNER_IDENTITY: (
            "I have no publishable information about alliances between players.",
            "プレイヤー間の関係について、公開できる情報は持っていません。",
        ),
        RULE_PRIVATE_TEAM_REFERENCE: (
            "I will not discuss relationships that are not recorded in public.",
            "公開記録にないプレイヤー同士の関係には触れません。",
        ),
        RULE_TEAM_ACTION_VIEW: (
            "I will judge and vote using public information only.",
            "判断と投票は、公表情報だけで行います。",
        ),
        RULE_UNPUBLISHED_SEER_RESULT: (
            "I have not published that divination verdict yet.",
            "その占い判定はまだ公開していません。",
        ),
        RULE_SELF_ROLE_PUBLIC_CLAIM: (
            "I will reveal my role later as an in-game claim.",
            "役職は、今後のCOとしてゲーム内で明かします。",
        ),
        RULE_THIRD_PERSON_SELF_STYLE: (
            "I speak about myself here in the first person.",
            "自分のことは一人称で述べることにします。",
        ),
    }

    @classmethod
    def _replacement_text(cls, rule_id: str, actor: Player) -> str | None:
        """補正文を応答言語で返す（drop対象ルールは None を返す）。"""
        if rule_id in DROP_ONLY_RULES:
            return None
        pair = cls._NEUTRAL_REPLACEMENTS.get(rule_id)
        if pair is None:
            return None
        from .prompts import is_japanese_language  # promptsはguardをimportしない

        return pair[1] if is_japanese_language(actor.response_language) else pair[0]

    @classmethod
    def repair_public_text(
        cls,
        content: str,
        classification: LeakClassification,
        actor: Player,
        *,
        repair_soft: bool = False,
    ) -> LeakRepair:
        """漏洩を含む文だけを差し替え・除去し、他の文は一字一句そのまま保持する。

        `repair_soft=True`（strict強度のモデル）では曖昧な役職宣言の文も中立化するが、
        balanced では本文へ手を入れず観測だけ行う。全文の書き直しは行わない。
        """
        if not isinstance(content, str) or not content.strip():
            return LeakRepair(text=content or "")
        candidates = [
            finding
            for finding in classification.hard
            if finding.char_start >= 0 and finding.region != "legacy"
        ]
        if repair_soft:
            candidates.extend(
                finding
                for finding in classification.soft
                if finding.char_start >= 0
                and finding.rule_id in STRICT_REPAIRED_SOFT_RULES
            )
        spans = cls._sentence_spans(content, 0)
        public_total = sum(len(sentence.strip()) for _, _, sentence in spans)
        unlocalizable = [finding for finding in classification.hard if finding.char_start < 0]
        if not candidates and not unlocalizable:
            return LeakRepair(
                text=content,
                original_char_count=public_total,
                kept_char_count=public_total,
                preserved_ratio=1.0,
                body_preservable=True,
            )
        replacements: list[tuple[int, int, str]] = []
        edits: list[tuple[str, str]] = []
        replaced_rules: list[str] = []
        dropped_rules: list[str] = []
        leaked_chars = 0
        targeted_spans: set[tuple[int, int]] = set()
        for finding in candidates:
            start, end = finding.char_start, finding.char_end
            replacement = cls._replacement_text(finding.rule_id, actor)
            if (start, end) in targeted_spans:
                continue
            targeted_spans.add((start, end))
            before = content[start:end].strip()
            leaked_chars += len(before)
            if replacement is None:
                replacement = ""
                dropped_rules.append(finding.rule_id)
            else:
                replaced_rules.append(finding.rule_id)
            edits.append((before, replacement))
            replacements.append((start, end, replacement))
        repaired = content
        for start, end, replacement in sorted(replacements, key=lambda item: -item[0]):
            repaired = repaired[:start] + replacement + repaired[end:]
        preserved_ratio = (
            round(1.0 - (leaked_chars / public_total), 4) if public_total else 1.0
        )
        body_preservable = (
            not unlocalizable and preserved_ratio >= MIN_BODY_PRESERVATION_RATIO
        )
        return LeakRepair(
            text=repaired,
            replaced_rules=tuple(dict.fromkeys(replaced_rules)),
            dropped_rules=tuple(dict.fromkeys(dropped_rules)),
            edit_count=len(replacements),
            original_char_count=public_total,
            kept_char_count=max(0, public_total - leaked_chars),
            leaked_char_count=leaked_chars,
            preserved_ratio=preserved_ratio,
            body_preservable=body_preservable,
            edits=tuple(edits),
        )


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
