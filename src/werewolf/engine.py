import random
import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .models import (
    ActionCandidateSet,
    ClaimRecord,
    GameState,
    LedgerFact,
    Message,
    Phase,
    Player,
    PublicRoleClaim,
    PublicSeerClaim,
    Role,
    PlayerActionMemory,
    SelfMemory,
)
from .discussion_format import (
    DEFAULT_QUESTION_HINT_MODE,
    QUESTION_HINT_NONE,
    QUESTION_HINT_STRICT,
    SEVERITY_FATAL,
    SEVERITY_OK,
    SEVERITY_REPAIRABLE,
    SEVERITY_WARNING,
    DiscussionBodyDiagnosis,
    answers_previous_question,
    extract_discussion_body,
    extract_reasoning_memo,
    find_repeated_topic,
    is_pass_body,
    normalize_question_hint_mode,
    parse_discussion_claim,
    parse_discussion_output,
    split_sentences,
)
from .leak_gate_policy import (
    DEFAULT_LEAK_GUARD_POLICY,
    GATE_HARD,
    GATE_OBSERVE,
    GATE_SOFT,
    RULE_MEMO_ONLY,
    STRENGTH_STRICT,
    LeakGuardProfiles,
    normalize_leak_guard_policy,
)
from .secret_leak_guard import (
    LeakClassification,
    LeakFinding,
    LeakRepair,
    SecretLeakGuard,
)
from .self_consistency import SelfConsistencyGuard
from .dead_player_resolution import analyze_dead_player_references
# TASK-055: 役職由来情報（エンジン発行）と公開申告のプロヴェナンスを分離する
from .role_provenance import (
    PUBLIC_LINE_LABEL,
    SOURCE_COMPATIBILITY_TEXT,
)
# TASK-056: Day内イベントの時系列台帳と未来参照の観測（検出は観測専用で本文は書き換えない）
from .day_timeline import (
    DAY_TIMELINE_POLICY_DEFAULT,
    DAY_TIMELINE_POLICY_OFF,
    TurnSnapshot,
    inspect_public_body,
    normalize_day_timeline_policy,
)

_CLAIMED_ROLE_NAMES = {
    "占い師": "seer",
    "人狼": "wolf",
    "狂人": "madman",
    "村人": "villager",
    "seer": "seer",
    "wolf": "wolf",
    "werewolf": "wolf",
    "madman": "madman",
    "villager": "villager",
}


@dataclass(frozen=True)
class _ClaimPublication:
    role_published: bool = False
    result_published: bool = False
    duplicate_suppressed: bool = False
    # TASK-055: allow_bluff_role_claim=False の設定で受理しなかった行があるか
    role_claim_disallowed: bool = False


# 役職COの自己参照（一人称語）prefix。自分のプレイヤー名でのCOも正当な公開主張
# として認識するため、`self_name`を指定するとprefixへ加える（TASK-039）。
_CLAIM_SELF_PREFIX = r"(?:私|僕|俺|自分|わたし|わたくし|ぼく|I|I'm|I am|my)"
# 役職名の直後に認められる区切り・説明語の照合。他者COの言及を自分の主張として
# 扱わないためのlookahead（TASK-039）。
_CLAIM_CONTINUATION = r"(?=\s*[:：、,，／/。.]|(?=$)|\s+(?:対象|判定|結果|理由|投票|target|result|reason|vote|because|since|divin)|CO)"
# 文頭マーカー（`【CO】`等）を剥がしてから文単位のCO認識をする。
_CLAIM_MARKER_PREFIX = re.compile(r"^\s*(?:【(?:CO|結果|本文)】|\[(?:CO|RESULT|BODY)\])\s*", re.IGNORECASE)


def extract_public_role_claim(
    content: str, self_name: str | None = None
) -> str | None:
    """公開された役職CO主張を認識する。

    旧来は全文の先頭形式だけを見ていたが、実機では本文の途中や別文にCOが現れる
    （`game_20da533bc4a6` Player 1）ため、文単位の先頭形式と自分のプレイヤー名
    形式も認識する（TASK-039）。役職名の直後に区切り・CO・対象/判定が続く場合だけ
    主張とし、「Player 3の占い師COを確認した」のような他者COの言及は主張にしない。
    """
    self_terms = [_CLAIM_SELF_PREFIX]
    if self_name:
        self_terms.append(f"{re.escape(self_name)}(?:さん)?")
    anchor = re.compile(
        rf"^\s*(?:(?:counter(?:-?claim)?)\s*)?"
        rf"(?:(?:{'|'.join(self_terms)})"
        rf"(?:\s+(?:am|claim|is|state|declare|playing|pretend to be|claim to be)\s*"
        rf"(?:a|an|the)?\s*)?)?\s*"
        rf"(?P<role>{'|'.join(_CLAIMED_ROLE_NAMES)})",
        re.IGNORECASE,
    )
    explicit_claim = re.compile(
        r"\s*(?:CO|claim(?:s|ing|ing to be)?|am|as|to be|は|が)"
        + _CLAIM_CONTINUATION,
        re.IGNORECASE,
    )
    claim_separator = re.compile(r"\s*(?:[:：、,，／/。.]|$)")
    for sentence in split_sentences(content):
        stripped = _CLAIM_MARKER_PREFIX.sub("", sentence.strip())
        match = anchor.match(stripped)
        if match is None:
            continue
        remainder = stripped[match.end() :]
        if (
            explicit_claim.match(remainder) is None
            and claim_separator.match(remainder) is None
        ):
            continue
        return _CLAIMED_ROLE_NAMES[match.group("role").lower()]
    return None


def is_public_seer_claim(content: str, self_name: str | None = None) -> bool:
    return extract_public_role_claim(content, self_name) == Role.SEER.value


_PLAYER_NUMBER_ALIAS = re.compile(r"Player\s*(\d+)", re.IGNORECASE)


def _player_name_pattern(name: str) -> str:
    """プレイヤー名の別称（プレイヤー5 / Player5 / P5）を同一プレイヤーとして認識する（TASK-041）。

    弱いモデルは `Player 5` を「プレイヤー5」や「P5」で書くため、占い結果の抽出だけ
    失敗して `seer_result_without_claim` → フォールバックになる経路を消す。
    """
    alternatives = [re.escape(name)]
    match = _PLAYER_NUMBER_ALIAS.fullmatch(name.strip())
    if match is not None:
        number = match.group(1)
        alternatives.append(rf"Player\s*{number}")
        alternatives.append(rf"プレイヤー\s*{number}")
        alternatives.append(rf"(?<![\w])P\s*{number}")
    return "|".join(alternatives)


def extract_public_seer_result(
    content: str | None, player_names: tuple[str, ...]
) -> tuple[str, str] | None:
    if content is None or not isinstance(content, str) or not content:
        return None
    names = sorted(player_names, key=len, reverse=True)
    target = next(
        (
            name
            for name in names
            if re.search(
                rf"(?:占い)?(?:対象名|対象者|対象|ターゲット|target)"
                rf"\s*[：:=＝]\s*(?:{_player_name_pattern(name)})"
                rf"(?:さん)?(?:\s|[／/、。,，]|$)",
                content,
                re.IGNORECASE,
            )
        ),
        None,
    )
    result = re.search(
        r"(?:判定|占い結果|結果|result)\s*[：:=＝]\s*(人狼|人間|werewolf|wolf|human)",
        content,
        re.IGNORECASE,
    )
    if target is not None and result is not None:
        val = result.group(1).lower()
        return target, "wolf" if val in ("人狼", "werewolf", "wolf") else "human"

    for name in names:
        natural_result = re.search(
            rf"(?:{_player_name_pattern(name)})(?:さん)?\s*(?:(?:は|が)\s*"
            rf"(?:(?:占い結果|判定)で\s*)?(人狼|人間)"
            rf"(?:と(?:判定(?:された|です|でした)?)?)?|"
            rf"\s+is\s+(?:a\s+)?(werewolf|wolf|human))",
            content,
            re.IGNORECASE,
        )
        if natural_result is not None:
            val = (natural_result.group(1) or natural_result.group(2)).lower()
            return name, "wolf" if val in ("人狼", "werewolf", "wolf") else "human"
    return None


# 議論本文の検証を重大度別に扱うための分類。
# 致命的：秘匿役職・未公開結果・狼仲間・システム情報の漏洩のみ。
# 修復可能：質問順・死亡者参照・未知参照・重複・冗長さなど。
_FATAL_REASONS = frozenset(
    {
        "secret_role_leak",
        "secret_team_leak",
        "hidden_result_leak",
        "prompt_or_config_leak",
        "seer_result_without_claim",
        "empty_body",
    }
)
_REPAIRABLE_REASONS = frozenset(
    {
        "dead_player_question",
        "dead_player_action_target",
        "unverified_dead_role",
        "unverified_past_vote_reference",
        "external_game_memory",
        "invalid_target",
        # TASK-048: 論点の再掲は修復対象から外した（警告＋新規性計測のみ・本文はそのまま公開）
        # 旧挙動との比較は novelty_policy="legacy_repair" で行える。
        # "repeated_discussion_topic",
        # TASK-043: プロンプトの案内文・プレースホルダの模倣は致命的ではなく、
        # 該当文の除去で修復できる（幻覚の占い結果を公開しない）。
        "template_placeholder_echo",
        # 該当文の除去で修復できる（幻覚の占い結果を公開しない）。
        "template_placeholder_echo",
    }
)


def _dead_player_names(state: GameState) -> frozenset[str]:
    return frozenset(player.name for player in state.players if not player.is_alive)


def _asks_question(text: str | None) -> bool:
    if not isinstance(text, str) or not text:
        return False
    return bool(
        re.search(
            r"[？?]|(?:教|聞|なぜ|どうして|何|誰|どこ|いつ)"
            r"|(?:\bwho\b|\bwhat\b|\bwhy\b|\bhow\b|\bwhen\b|\bwhere\b"
            r"|tell me|explain|do you know|can you)",
            text,
            re.IGNORECASE,
        )
    )


def _references_dead_player(text: str | None, dead_names: frozenset[str]) -> bool:
    if not isinstance(text, str) or not text:
        return False
    return any(re.search(rf"{re.escape(name)}(?:さん)?", text) for name in dead_names)


def _asserts_dead_role_as_fact(text: str | None, dead_names: frozenset[str]) -> bool:
    if not isinstance(text, str) or not text:
        return False
    for name in dead_names:
        if re.search(
            rf"{re.escape(name)}(?:さん)?\s*(?:は|が|is|was|seems)\s*"
            r"(?:人狼|狼|人間|占い師|村人|狂人"
            r"|werewolf|wolf|human|seer|villager|madman)"
            r"(?:です|だ|である|と思|だと|らしい|確定|本物|真の"
            r"|for sure|definitely|certainly|confirmed)",
            text,
            re.IGNORECASE,
        ):
            return True
    return False


# TASK-047: 未確定の仮説を「確定事実」として断定した発言の検出。修復や再生成は行わず
# 警告のみとする（論点・口調・結論はモデルの選択のまま、評価ログで再掲率を測る）。
UNVERIFIED_HYPOTHESIS_AS_FACT = "unverified_hypothesis_as_fact"

_CERTAINTY_CUE = re.compile(
    r"(?:確定|確実に|間違いなく|断定|断定する)"
    r"|(?:\bdefinitely\b|\bcertainly\b|\bfor sure\b|\bconfirmed\b|\bundeniably\b|\bproven\b)",
    re.IGNORECASE,
)
_ROLE_CUE = re.compile(
    r"(?:人狼|狼|占い師|村人|狂人)|(?:werewolf|wolf|seer|villager|madman)",
    re.IGNORECASE,
)


def _asserts_unverified_role_as_fact(
    text: str | None, state: GameState
) -> tuple[str, ...]:
    """生存者の役職を台帳の裏付けなしに断定している文を検出する（未確定仮説の確定化）。"""
    if not isinstance(text, str) or not text:
        return ()
    alive = frozenset(state.alive_players)
    confirmed_text = " ".join(
        fact.description
        for fact in state.fact_ledger.confirmed_facts
        if fact.visibility == "public"
    )
    hits: list[str] = []
    for sentence in split_sentences(text):
        if _CERTAINTY_CUE.search(sentence) is None or _ROLE_CUE.search(sentence) is None:
            continue
        mentioned = [name for name in alive if name in sentence]
        if not mentioned:
            continue
        if any(name in confirmed_text for name in mentioned):
            # 台帳の確定事実が裏付けになっている場合は指摘しない
            continue
        hits.append(sentence)
    return tuple(hits)


def _references_external_game_memory(text: str | None) -> bool:
    if not isinstance(text, str) or not text:
        return False
    return bool(
        re.search(
            r"(?:過去のゲーム|前のゲーム|前回のゲーム|前にやった|別のゲーム|"
            r"前世|前回の役職|学習|トレーニングデータ|研修|データセット)"
            r"|(?:past|previous|last|other|another|other games?|earlier|in a previous)"
            r"[^.!?]{0,24}(?:werewolf game|game|match|role|village)"
            r"|(?:training data|training|learned|real-world|real person|in real life)",
            text,
            re.IGNORECASE,
        )
    )


# TASK-043: 案内文・プレースホルダの模倣（`Player X` / `プレイヤーX` / 判定の二者選択等）。
# 実在しない対象を占った幻覚は、致命的扱いではなく修復対象として本文から除去する。
_TEMPLATE_PLACEHOLDER = re.compile(
    r"(?:\bplayer\s+[xX]\b"
    r"|プレイヤー\s*[xXｘ]"
    r"|(?:対象名|対象|ターゲット|target)\s*[:=＝]\s*[xXｘ]\b"
    r"|werewolf\s*/\s*human"
    r"|人狼\s*/\s*人間"
    r"|\bplayer\s+name\b"
    r"|\bplaceholder\b|プレースホルダ)",
    re.IGNORECASE,
)


def _references_template_placeholder(text: str | None) -> bool:
    """プロンプトの案内文・プレースホルダの模倣を含むか（TASK-043）。"""
    return isinstance(text, str) and bool(_TEMPLATE_PLACEHOLDER.search(text))


def _directly_addressed(text: str, previous_message: Message, actor_name: str) -> bool:
    if previous_message is None or not previous_message.content:
        return False
    content = previous_message.content
    return (
        actor_name in content
        or re.search(r"(?:あなた|君|お前|\byou\b|\byour\b)", content, re.IGNORECASE) is not None
        or _asks_question(content)
    )


def _answers_previous_question(text: str, previous_content: str) -> bool:
    """直前の質問に対する応答があるかを軽量に判定する（共通実装へ委譲）。"""
    return answers_previous_question(text, previous_content)


def _is_redundant_body(body: str, transcript: tuple[Message, ...]) -> bool:
    return find_repeated_topic(body, transcript)


def _has_published_seer_result(
    state: GameState, claimant: str, target: str, result: str
) -> bool:
    return any(
        claim.claimant == claimant
        and claim.target == target
        and claim.result == result
        for claim in state.seer_claims
    ) or (claimant, target) in state.claim_ledger.public_seer_results


def diagnose_discussion_body(
    body: str | None,
    actor: Player,
    state: GameState,
    previous_message: Message | None = None,
    round_transcript: tuple[Message, ...] = (),
    question_hint_mode: str | None = None,
) -> DiscussionBodyDiagnosis:
    """議論本文の違反を重大度別に診断する。

    致命的違反（秘匿情報漏洩・未公開の占い結果CO欠落・空本文）は
    安全フォールバックの対象とし、それ以外は修復可能または品質警告とする。

    `question_hint_mode`（省略時は `state.question_hint_mode`）で質問回答の強制度を決める。
    softでは品質警告のみ（再生成しない）、noneでは判定しない。
    """
    reasons: list[str] = []
    fatal = repairable = warning = False
    hint_mode = normalize_question_hint_mode(
        question_hint_mode
        if question_hint_mode is not None
        else getattr(state, "question_hint_mode", DEFAULT_QUESTION_HINT_MODE)
    )

    if body is None or not body.strip():
        return DiscussionBodyDiagnosis(
            SEVERITY_FATAL, ("empty_body",), True, False, False, False
        )
    stripped = body.strip()
    if is_pass_body(stripped):
        return DiscussionBodyDiagnosis(
            SEVERITY_OK, (), False, False, False, True
        )

    dead_names = _dead_player_names(state)

    # 致命的：秘匿情報の漏洩（役職・陣営・非公開結果・システム情報）
    findings = SecretLeakGuard.inspect(stripped, actor, state)
    if findings:
        reasons.extend(finding.category for finding in findings)
        fatal = True

    # 非公開の占い結果をCOなしで発言（真占い師の判定結果に限る）
    if not fatal and actor.role is Role.SEER:
        public_result = extract_public_seer_result(
            stripped, tuple(player.name for player in state.players)
        )
        if (
            public_result is not None
            and not is_public_seer_claim(stripped, actor.name)
            and state.seer_results.get(public_result[0]) == public_result[1]
            and not _has_published_seer_result(
                state, actor.name, public_result[0], public_result[1]
            )
        ):
            reasons.append("seer_result_without_claim")
            fatal = True

    if not fatal:
        # TASK-047: 未確定仮説の確定化は警告のみ（本文はそのまま公開し、修復・再生成しない）
        if _asserts_unverified_role_as_fact(stripped, state):
            warning = True
            reasons.append(UNVERIFIED_HYPOTHESIS_AS_FACT)
        if _references_dead_player(stripped, dead_names):
            living_names = frozenset(
                player.name for player in state.players if player.is_alive
            )
            classifications = analyze_dead_player_references(
                stripped, dead_names, living_names
            )
            has_actionable = any(c.is_actionable_target for c in classifications)
            has_question = any(c.is_dead_player_question for c in classifications)
            if has_actionable:
                reasons.append("dead_player_action_target")
                repairable = True
            elif has_question:
                # 死亡者への質問・回答要求は修復対象
                reasons.append("dead_player_question")
                repairable = True
            else:
                warning = True
                reasons.append("dead_player_mention")
        elif _asserts_dead_role_as_fact(stripped, dead_names):
            reasons.append("unverified_dead_role")
            repairable = True
        elif _references_external_game_memory(stripped):
            reasons.append("external_game_memory")
            repairable = True
        elif _references_template_placeholder(stripped):
            # TASK-043: 案内文・プレースホルダの模倣は本文の該当文除去で修復する
            # （再生成や安全フォールバックを作らない）。
            reasons.append("template_placeholder_echo")
            repairable = True
        elif (
            hint_mode != QUESTION_HINT_NONE
            and previous_message is not None
            and _directly_addressed(stripped, previous_message, actor.name)
            and not _answers_previous_question(stripped, previous_message.content)
        ):
            # TASK-038: strict（1文目ルール）のみ再生成対象。softは内容回答を確認して警告のみ。
            reasons.append("question_not_answered_first")
            repairable = hint_mode == QUESTION_HINT_STRICT
            warning = warning or not repairable
        elif find_repeated_topic(stripped, round_transcript):
            # TASK-048: 論点の再掲は致命的でも修復対象でもなく「品質警告＋新規性の計測」。
            # 本文はそのまま公開し、文の除去・再生成・定型文の挿入を行わない。
            # （同一論点の連呼と、新根拠を伴う正当な繰り返しは llm_actions 側で区別する）
            reasons.append("repeated_discussion_topic")
            warning = True

    if not fatal and not repairable and _is_redundant_body(stripped, round_transcript):
        warning = True
        reasons.append("discussion_redundancy")

    severity = (
        SEVERITY_FATAL
        if fatal
        else SEVERITY_REPAIRABLE
        if repairable
        else SEVERITY_WARNING
        if warning
        else SEVERITY_OK
    )
    return DiscussionBodyDiagnosis(
        severity,
        tuple(reasons),
        fatal,
        repairable,
        warning,
        publishable=not fatal,
    )


class ActionPolicy(Protocol):
    def wolf_discuss(
        self, wolf: str, partner: str, history: tuple[Message, ...]
    ) -> str: ...

    def choose_attack(self, wolves: tuple[str, ...], candidates: tuple[str, ...]) -> str: ...

    def choose_seer_target(
        self, seer: str, candidates: tuple[str, ...], results: tuple[tuple[str, str], ...]
    ) -> str: ...

    def discuss(
        self, player: str, day: int, alive: tuple[str, ...], history: tuple[Message, ...]
    ) -> str: ...

    def vote(
        self, voter: str, candidates: tuple[str, ...], day: int, runoff: bool
    ) -> str: ...


class MockActions:
    """Deterministic policy for exercising the game without an LLM."""

    def wolf_discuss(
        self, wolf: str, partner: str, history: tuple[Message, ...]
    ) -> str:
        return f"{wolf} confers with {partner}."

    def choose_attack(self, wolves: tuple[str, ...], candidates: tuple[str, ...]) -> str:
        return candidates[0]

    def choose_seer_target(
        self, seer: str, candidates: tuple[str, ...], results: tuple[tuple[str, str], ...]
    ) -> str:
        return candidates[0]

    def discuss(
        self, player: str, day: int, alive: tuple[str, ...], history: tuple[Message, ...]
    ) -> str:
        return f"Day {day}: {player} speaks."

    def vote(
        self, voter: str, candidates: tuple[str, ...], day: int, runoff: bool
    ) -> str:
        eligible = tuple(candidate for candidate in candidates if candidate != voter)
        if not eligible:
            raise ValueError(f"No eligible vote targets for {voter}.")
        return eligible[0]


class GameEngine:
    def __init__(
        self,
        state: GameState,
        actions: ActionPolicy,
        seed: int | None = None,
        event_callback: Callable[[dict[str, object]], None] | None = None,
    ):
        self.state = state
        self.actions = actions
        self._rng = random.Random(state.seed if seed is None else seed)
        self.event_callback = event_callback

    def _record_secret_leak(
        self,
        player: Player,
        action: str,
        content: str,
        findings: tuple[LeakFinding, ...],
        attempt: int,
        classification: LeakClassification | None = None,
    ) -> None:
        speech_id = sum(
            message.speaker != "system"
            for message in self.state.conversation_log
        ) + 1
        for category in sorted({finding.category for finding in findings}):
            self.state.add_integrity_alert(
                speaker=player.name,
                category=category,
                speech_id=speech_id,
                summary="Content withheld: the secrecy guard detected hidden information before publishing",
            )
        if self.event_callback is None:
            return
        phase_labels = {
            Phase.DAY_DISCUSSION: "昼・議論",
        }
        event: dict[str, object] = {
            "type": "secret_role_leak_detected",
            "day": self.state.day,
            "phase": self.state.phase.value,
            "phase_label": phase_labels.get(
                self.state.phase, self.state.phase.value
            ),
            "player": player.name,
            "model": player.model,
            "action": action,
            "speech_id": speech_id,
            "leak_categories": sorted({finding.category for finding in findings}),
            "detection_rules": sorted({finding.rule_id for finding in findings}),
            "content_sha256": SecretLeakGuard.content_hash(content),
            "safety_retry": attempt,
            "private": False,
            "elapsed_seconds": 0.0,
        }
        if classification is not None:
            # 秘匿内容は書かず、ゲート・ルール・公開事実ID・位置だけを残す（公開ログsafe）。
            event.update(
                {
                    "guard_policy": classification.policy,
                    "guard_strength": classification.strength,
                    "leak_gates": [finding.gate for finding in findings],
                    "leak_regions": [finding.region for finding in findings],
                    "detection_spans": [
                        [finding.char_start, finding.char_end] for finding in findings
                    ],
                    "sentence_sha256": [
                        SecretLeakGuard.content_hash(finding.evidence)
                        for finding in findings
                    ],
                    "published_fact_ids": list(classification.fact_ids)[:12],
                    "soft_note_rules": sorted(set(classification.soft_rules)),
                    "observe_rules": sorted(set(classification.observe_rules)),
                    "memo_region_rules": list(classification.memo_rules),
                    "legacy_hard_rules": list(classification.legacy_hard_rules),
                    "false_positive_avoided": classification.false_positives_avoided,
                }
            )
        self.event_callback(event)

    def _leak_guard_context(self) -> tuple[str, LeakGuardProfiles]:
        """TASK-054: ガードポリシーとモデル別プロファイル（強度の算出元）を返す。"""
        policy = normalize_leak_guard_policy(
            getattr(self.state, "leak_guard_policy", DEFAULT_LEAK_GUARD_POLICY)
        )
        profiles = getattr(self.state, "leak_guard_profiles", None)
        if not isinstance(profiles, LeakGuardProfiles):
            profiles = LeakGuardProfiles(policy=policy)
            self.state.leak_guard_profiles = profiles
        profiles.policy = policy
        return policy, profiles

    def _emit_leak_observation(
        self,
        player: Player,
        action: str,
        classification: LeakClassification,
        *,
        body_char_count: int,
        kept_char_count: int,
        regeneration_avoided: bool = True,
    ) -> None:
        """soft / observe 指摘を「本文を公開したまま観測」する（遮断も削句もしない）。

        legacy_strict なら遮断されていたはずの指摘を保持できた件数は、ログ比較で
        そのまま誤検出自動回避として計測できる（TASK-054の完了条件）。
        """
        if self.event_callback is None:
            return
        notes = [*classification.soft, *classification.observe]
        phase_labels = {Phase.DAY_DISCUSSION: "昼・議論"}
        self.event_callback(
            {
                "type": "secret_leak_note",
                "day": self.state.day,
                "phase": self.state.phase.value,
                "phase_label": phase_labels.get(self.state.phase, self.state.phase.value),
                "player": player.name,
                "model": player.model,
                "action": action,
                "guard_policy": classification.policy,
                "guard_strength": classification.strength,
                "note_rules": sorted({finding.rule_id for finding in notes}),
                "note_gates": [finding.gate for finding in notes],
                "note_categories": sorted({finding.category for finding in notes}),
                "published_fact_ids": list(
                    dict.fromkeys(fid for finding in notes for fid in finding.fact_ids)
                )[:12],
                "memo_region_rules": list(classification.memo_rules),
                "legacy_hard_rules": list(classification.legacy_hard_rules),
                "false_positive_avoided": classification.false_positives_avoided,
                "body_char_count": body_char_count,
                "kept_char_count": kept_char_count,
                "regeneration_avoided": regeneration_avoided,
                "private": False,
                "elapsed_seconds": 0.0,
            }
        )

    def _emit_leak_repair(
        self,
        player: Player,
        action: str,
        classification: LeakClassification,
        repair: LeakRepair,
        *,
        retry_seconds: float = 0.0,
        regeneration_avoided: bool = True,
    ) -> None:
        """TASK-054実装案3: 漏洩文だけを選んだ補正記録（全文再生成を避けられたか残す）。"""
        if self.event_callback is None:
            return
        phase_labels = {Phase.DAY_DISCUSSION: "昼・議論"}
        self.event_callback(
            {
                "type": "secret_leak_sentence_repair",
                "day": self.state.day,
                "phase": self.state.phase.value,
                "phase_label": phase_labels.get(self.state.phase, self.state.phase.value),
                "player": player.name,
                "model": player.model,
                "action": action,
                "guard_policy": classification.policy,
                "guard_strength": classification.strength,
                "replaced_rules": list(repair.replaced_rules),
                "dropped_rules": list(repair.dropped_rules),
                "edit_count": repair.edit_count,
                "body_char_count": repair.original_char_count,
                "kept_char_count": repair.kept_char_count,
                "body_preservation_ratio": repair.preserved_ratio,
                "regeneration_avoided": regeneration_avoided,
                "retry_seconds": round(retry_seconds, 3),
                # 秘匿内容は公開しない。置換後の本文ハッシュと位置だけを残す
                # （私的ログの元応答contentから対応を取れる）。
                "repaired_content_sha256": SecretLeakGuard.content_hash(repair.text),
                "edit_spans": [
                    [finding.char_start, finding.char_end]
                    for finding in (*classification.hard, *classification.soft)
                    if finding.char_start >= 0
                ],
                "published_fact_ids": list(classification.fact_ids)[:12],
                "private": False,
                "elapsed_seconds": 0.0,
            }
        )


    def _authorized_public_results(
        self, player_name: str, claim: str, result: str
    ) -> frozenset[tuple[str, str]]:
        player = next(player for player in self.state.players if player.name == player_name)
        if (
            player.role is not Role.SEER
            or extract_public_role_claim(claim, player_name) != Role.SEER.value
        ):
            return frozenset()
        parsed_result = extract_public_seer_result(
            f"{claim}\n{result}",
            tuple(player.name for player in self.state.players),
        )
        if (
            parsed_result is None
            or self.state.seer_results.get(parsed_result[0]) != parsed_result[1]
        ):
            return frozenset()
        return frozenset({parsed_result})

    def _complete_seer_claim_for_publish(
        self, player: Player, content: str, findings: tuple[LeakFinding, ...]
    ) -> tuple[str, frozenset[tuple[str, str]]] | None:
        """未公開の占い結果に言及した真占い師の発言へCO行を決定的に補完する（TASK-039）。

        判定は台帳の実際の占い結果と一致する場合だけ公開し、人狼判定・人間判定の
        どちらでもCO形式を先頭へ追加する。メタ・陣営漏洩は補完しない。
        """
        if player.role is not Role.SEER or not findings:
            return None
        if any(
            finding.category not in {"secret_role_leak", "hidden_result_leak"}
            for finding in findings
        ):
            return None
        spoken = extract_public_seer_result(
            content, tuple(item.name for item in self.state.players)
        )
        if spoken is None:
            return None
        target, result = spoken
        if self.state.seer_results.get(target) != result:
            return None
        if _has_published_seer_result(self.state, player.name, target, result):
            return None
        label = "Werewolf" if result == "wolf" else "Human"
        proposal = (
            f"Vote for {target}."
            if result == "wolf"
            else f"Suspicion toward {target} is weak; I want to consider other possibilities carefully."
        )
        claim_line = (
            f"I am Seer CO: Target={target}, Result={label}, "
            f"Reason=divination result from Night 1, Vote proposal={proposal}"
        )
        return f"{claim_line}\n{content.strip()}", frozenset({(target, result)})

    def _guard_public_text(
        self,
        player_name: str,
        action: str,
        content: str,
        history: tuple[Message, ...],
        *,
        allow_retry: bool = True,
        allow_public_claim: bool = False,
        allowed_public_results: frozenset[tuple[str, str]] = frozenset(),
    ) -> str:
        player = next(player for player in self.state.players if player.name == player_name)
        policy, profiles = self._leak_guard_context()
        strength = profiles.strength_for(player.model)
        body_char_count = len(content.strip())

        def classify(text: str) -> LeakClassification:
            return SecretLeakGuard.classify(
                text,
                player,
                self.state,
                allowed_public_claim=(
                    allow_public_claim
                    or SecretLeakGuard.allows_public_claim(text, player)
                ),
                allowed_public_results=allowed_public_results,
                guard_policy=policy,
                guard_strength=strength,
            )

        classification = classify(content)
        hard = classification.hard
        if not hard:
            # 警告・観測のみのときは本文をそのまま公開する（旧来はここで全文再生成していた）。
            if classification.soft or classification.observe or classification.memo_rules:
                self._emit_leak_observation(
                    player,
                    action,
                    classification,
                    body_char_count=body_char_count,
                    kept_char_count=body_char_count,
                )
                profiles.observe(
                    player.model,
                    gate=GATE_SOFT if classification.soft else GATE_OBSERVE,
                    rule_ids=tuple(
                        dict.fromkeys(
                            (
                                *classification.soft_rules,
                                *classification.observe_rules,
                                *classification.memo_rules,
                            )
                        )
                    ),
                    body_char_count=body_char_count,
                    kept_char_count=body_char_count,
                )
            return content
        # 真占い師が未公開の占い結果に言及した場合は、安全再生成ではなく決定的な
        # CO補完で直して情報を保持する（TASK-039）。再生成だと本文が消失し、三人称の
        # 自己参照だけの発言が公開されていた（`game_20da533bc4a6` Player 1）。
        completion = self._complete_seer_claim_for_publish(player, content, hard)
        if completion is not None:
            completed, authorized = completion
            if not classify(completed).hard:
                target, result = next(iter(authorized))
                if self.event_callback is not None:
                    self.event_callback(
                        {
                            "type": "seer_claim_deterministically_completed",
                            "day": self.state.day,
                            "phase": self.state.phase.value,
                            "player": player.name,
                            "model": player.model,
                            "action": action,
                            "target": target,
                            "result": result,
                            "guard_policy": classification.policy,
                            "guard_strength": classification.strength,
                            "private": False,
                            "elapsed_seconds": 0.0,
                        }
                    )
                return completed

        self._record_secret_leak(
            player, action, content, hard, 0, classification=classification
        )

        # TASK-054実装案3: 本文全体を定型文へ置換せず、漏洩を含む文だけを選んで中立化する。
        # 本文保持率が閾値以上なら再生成不要（モデルの論旨・口調はそのまま残る）。
        repair = SecretLeakGuard.repair_public_text(
            content,
            classification,
            player,
            repair_soft=(strength == STRENGTH_STRICT),
        )
        if repair.applied and repair.body_preservable and repair.text.strip():
            if not classify(repair.text).hard:
                self._emit_leak_repair(player, action, classification, repair)
                profiles.observe(
                    player.model,
                    gate=GATE_HARD,
                    rule_ids=tuple((*repair.replaced_rules, *repair.dropped_rules)),
                    repaired=True,
                    regenerated=False,
                    body_char_count=repair.original_char_count,
                    kept_char_count=repair.kept_char_count,
                )
                return repair.text

        retry = getattr(self.actions, "retry_after_secret_leak", None)
        if allow_retry and callable(retry):
            retained_body = repair.text if (repair.applied and repair.text.strip()) else None
            leak_categories = tuple(sorted({finding.category for finding in hard}))
            rule_ids = tuple(dict.fromkeys(finding.rule_id for finding in hard))
            started_at = time.perf_counter()
            try:
                candidate = retry(
                    player_name,
                    self.state.day,
                    tuple(self.state.alive_players),
                    history,
                    leak_categories,
                    rule_ids=rule_ids,
                    retained_body=retained_body,
                )
            except TypeError:  # 旧シグネチャの実装（モック・後方互換）
                candidate = retry(
                    player_name,
                    self.state.day,
                    tuple(self.state.alive_players),
                    history,
                    leak_categories,
                )
            retry_seconds = round(time.perf_counter() - started_at, 3)
            if isinstance(candidate, str) and candidate.strip():
                retry_classification = classify(candidate)
                if not retry_classification.hard:
                    self._emit_leak_regeneration(
                        player,
                        action,
                        classification,
                        retry_seconds=retry_seconds,
                        body_char_count=body_char_count,
                        kept_char_count=len(candidate.strip()),
                        outcome="regenerated",
                    )
                    profiles.observe(
                        player.model,
                        gate=GATE_HARD,
                        rule_ids=rule_ids,
                        regenerated=True,
                        retry_seconds=retry_seconds,
                        body_char_count=body_char_count,
                        kept_char_count=len(candidate.strip()),
                    )
                    return candidate.strip()
                self._record_secret_leak(
                    player,
                    action,
                    candidate,
                    retry_classification.hard,
                    1,
                    classification=retry_classification,
                )
                sanitized, _ = SecretLeakGuard.redact(
                    candidate,
                    player,
                    self.state,
                    allowed_public_results=allowed_public_results,
                )
                self._emit_leak_regeneration(
                    player,
                    action,
                    classification,
                    retry_seconds=retry_seconds,
                    body_char_count=body_char_count,
                    kept_char_count=len(sanitized.strip()),
                    outcome="redacted_after_retry",
                )
                profiles.observe(
                    player.model,
                    gate=GATE_HARD,
                    rule_ids=rule_ids,
                    regenerated=True,
                    redacted=True,
                    retry_seconds=retry_seconds,
                    body_char_count=body_char_count,
                    kept_char_count=len(sanitized.strip()),
                )
                return sanitized

        sanitized, _ = SecretLeakGuard.redact(
            content,
            player,
            self.state,
            allowed_public_results=allowed_public_results,
        )
        self._emit_leak_regeneration(
            player,
            action,
            classification,
            retry_seconds=0.0,
            body_char_count=body_char_count,
            kept_char_count=len(sanitized.strip()),
            outcome="redacted_without_retry",
        )
        profiles.observe(
            player.model,
            gate=GATE_HARD,
            rule_ids=tuple(dict.fromkeys(finding.rule_id for finding in hard)),
            redacted=True,
            regenerated=False,
            body_char_count=body_char_count,
            kept_char_count=len(sanitized.strip()),
        )
        return sanitized

    def _emit_leak_regeneration(
        self,
        player: Player,
        action: str,
        classification: LeakClassification,
        *,
        retry_seconds: float,
        body_char_count: int,
        kept_char_count: int,
        outcome: str,
    ) -> None:
        """TASK-054: 実際に再要求・削句へ回った場合の負荷をモデル別に記録する。"""
        if self.event_callback is None:
            return
        phase_labels = {Phase.DAY_DISCUSSION: "昼・議論"}
        self.event_callback(
            {
                "type": "secret_leak_regeneration",
                "day": self.state.day,
                "phase": self.state.phase.value,
                "phase_label": phase_labels.get(self.state.phase, self.state.phase.value),
                "player": player.name,
                "model": player.model,
                "action": action,
                "guard_policy": classification.policy,
                "guard_strength": classification.strength,
                "hard_rules": sorted(set(classification.hard_rules)),
                "outcome": outcome,
                "retry_seconds": round(retry_seconds, 3),
                "body_char_count": body_char_count,
                "kept_char_count": kept_char_count,
                "body_preservation_ratio": (
                    round(kept_char_count / body_char_count, 4) if body_char_count else 1.0
                ),
                "regeneration_avoided": False,
                "private": False,
                "elapsed_seconds": 0.0,
            }
        )

    def run(self) -> GameState:
        self.check_winner()
        while self.state.winner is None:
            self.run_round()
        return self.state

    def run_round(self) -> None:
        if self.state.winner is not None or self.state.phase is Phase.ENDED:
            raise RuntimeError("Cannot run a round after the game has ended.")
        self.run_wolf_phase()
        if self.state.winner is not None:
            return
        self.run_seer_phase()
        self.resolve_morning()
        if self.state.winner is not None:
            return
        self.run_discussion()
        self.run_voting()

    def run_wolf_phase(self) -> None:
        self.state.phase = Phase.NIGHT_WOLF
        wolves = self._living_by_role(Role.WOLF)
        if not wolves:
            self.check_winner()
            return
        if self.state.day == 1:
            self.state.pending_attack = None
            return

        if len(wolves) == 2:
            for wolf in wolves:
                partner = next(player for player in wolves if player is not wolf)
                content = self.actions.wolf_discuss(
                    wolf.name, partner.name, tuple(self.state.wolf_chat_log)
                )
                self.state.wolf_chat_log.append(Message(wolf.name, content))
                self._record_action_memory(
                    player=wolf,
                    action_type="wolf_discussion",
                    day=self.state.day,
                    phase=self.state.phase.value,
                    summary=content,
                    reasoning_summary=self._reason_from_text(content),
                )

        candidates = tuple(
            player.name
            for player in self.state.players
            if player.is_alive and player.role is not Role.WOLF
        )
        if candidates:
            candidate_set = ActionCandidateSet(
                action="attack",
                actor=wolves[0].name,
                day=self.state.day,
                candidates=candidates,
                fallback=candidates[0],
            )
            self.state.action_candidate_sets["attack"] = candidate_set
            self.state.snapshot_ledgers("before", "attack", wolves[0].name)
            target = self.actions.choose_attack(
                tuple(wolf.name for wolf in wolves), candidate_set.candidates
            )
            self.state.snapshot_ledgers("after", "attack", wolves[0].name)
            self.state.pending_attack = (
                target if target in candidate_set.candidates else candidate_set.fallback
            )
            if self.state.pending_attack is not None:
                self._record_action_memory(
                    player=wolves[0],
                    action_type="attack",
                    day=self.state.day,
                    phase=self.state.phase.value,
                    target=self.state.pending_attack,
                    reasoning_summary=self._latest_action_reason(
                        wolves[0].name, {"wolf_discussion"}
                    ),
                    evidence=[f"attack-candidates-day-{self.state.day}"],
                )
        else:
            self.state.pending_attack = None

    def run_seer_phase(self) -> None:
        self.state.phase = Phase.NIGHT_SEER
        seers = self._living_by_role(Role.SEER)
        if not seers:
            return

        seer = seers[0]
        candidates = tuple(
            player.name
            for player in self.state.players
            if player.is_alive
            and player.name != seer.name
            and player.name not in self.state.seer_results
        )
        candidate_set = ActionCandidateSet(
            action="seer",
            actor=seer.name,
            day=self.state.day,
            candidates=candidates,
            fallback=candidates[0] if candidates else None,
        )
        self.state.action_candidate_sets["seer"] = candidate_set
        if not candidates:
            return
        self.state.snapshot_ledgers("before", "seer", seer.name)
        target = self.actions.choose_seer_target(
            seer.name,
            candidate_set.candidates,
            tuple(self.state.seer_results.items()),
        )
        self.state.snapshot_ledgers("after", "seer", seer.name)
        if target not in candidate_set.candidates:
            target = candidate_set.fallback
        if target is None:
            return
        target_player = self._player(target)
        result = "wolf" if target_player.role is Role.WOLF else "human"
        # TASK-055: エンジンが発行した占い結果だけを RoleEvidence として登録する。
        # 昼に公開された申告行とは別台帳であり、このIDを持つ情報が本物の判定である。
        evidence_id = self.state.role_provenance.record_engine_divination(
            owner=seer.name,
            day=self.state.day,
            target=target,
            result=result,
        )
        self.state.seer_results[target] = result
        self.state.seer_result_history.append((self.state.day, target, result))
        self.state.fact_ledger.confirmed_facts.append(
            LedgerFact(
                day=self.state.day,
                kind="seer_result",
                description=f"{target}: {result}",
                visibility="seer",
                actor=seer.name,
                evidence_id=evidence_id,
            )
        )
        self._record_action_memory(
            player=seer,
            action_type="seer",
            day=self.state.day,
            phase=self.state.phase.value,
            target=target,
            reasoning_summary=self._latest_action_reason(
                seer.name, {"discussion", "answer"}
            ),
            evidence=[f"seer-result-day-{self.state.day}-{target}"],
        )

    def resolve_morning(self) -> None:
        self.state.phase = Phase.MORNING
        victim = self.state.pending_attack
        self.state.pending_attack = None
        if victim is not None and self._player(victim).is_alive:
            self.state.mark_dead(victim)
            self.state.attack_deaths.append((self.state.day, victim))
            self.state.fact_ledger.confirmed_facts.append(
                LedgerFact(
                    day=self.state.day,
                    kind="attack_death",
                    description=f"{victim} was attacked during the night.",
                )
            )
            self.state.conversation_log.append(
                Message("system", f"{victim} was attacked during the night.")
            )
            # TASK-056: 公開された死亡を時系列台帳へ登録する（議論ターン外なので turn=0）
            self.state.public_timeline.record_death(
                day=self.state.day, name=victim, executed=False
            )
        self.check_winner()
        if self.state.winner is None:
            self.state.conversation_log.append(
                Message("system", f"Day {self.state.day} begins.")
            )
            # TASK-056: 日の開始を時系列台帳へ登録する（この時点から議論イベントが始まる）
            self.state.public_timeline.record_day_start(self.state.day)
            self.state.phase = Phase.DAY_DISCUSSION

    def run_discussion(self) -> None:
        self.state.phase = Phase.DAY_DISCUSSION
        for turn in range(1, 4):
            self.state.discussion_turn = turn
            self.state.begin_discussion_turn()
            # TASK-047: ターン開始時に仮説台帳の死亡者参照・反証・重複CO矛盾を解決する
            self.state.resolve_hypothesis_ledger()
            for player in self.state.players:
                if not player.is_alive:
                    continue
                question_candidates = tuple(
                    name for name in self.state.alive_players if name != player.name
                )
                self.state.action_candidate_sets[f"question:{player.name}"] = (
                    ActionCandidateSet(
                        action="question",
                        actor=player.name,
                        day=self.state.day,
                        candidates=question_candidates,
                        fallback=question_candidates[0] if question_candidates else None,
                    )
                )
                stage_claim = getattr(
                    self.actions, "should_stage_discussion_claim", None
                )
                choose_claim = getattr(
                    self.actions, "choose_discussion_claim", None
                )
                discuss_body = getattr(self.actions, "discuss_body", None)
                has_staged_interface = (
                    callable(stage_claim)
                    and callable(choose_claim)
                    and callable(discuss_body)
                    and getattr(self.actions, "discussion_claim_mode", "legacy")
                    != "legacy"
                )
                if has_staged_interface:
                    if stage_claim(player.name):
                        self.state.snapshot_ledgers(
                            "before", "discussion_claim", player.name
                        )
                        claim_response = choose_claim(
                            player.name,
                            self.state.day,
                            tuple(self.state.alive_players),
                            tuple(self.state.conversation_log),
                        )
                        decision = parse_discussion_claim(claim_response)
                        authorized_results = (
                            self._authorized_public_results(
                                player.name, decision.claim, decision.result
                            )
                            if decision is not None
                            else frozenset()
                        )
                        claim_response = self._guard_public_text(
                            player.name,
                            "discussion_claim",
                            claim_response,
                            tuple(self.state.conversation_log),
                            allow_public_claim=(
                                decision is not None
                                and extract_public_role_claim(decision.claim, player.name)
                                is not None
                            ),
                            allowed_public_results=authorized_results,
                        )
                        decision = parse_discussion_claim(claim_response)
                        deterministic_claim = getattr(
                            self.actions, "deterministic_discussion_claim", None
                        )
                        if decision is None and callable(deterministic_claim):
                            claim_response = deterministic_claim(player.name)
                            fallback_decision = parse_discussion_claim(claim_response)
                            authorized_results = (
                                self._authorized_public_results(
                                    player.name,
                                    fallback_decision.claim,
                                    fallback_decision.result,
                                )
                                if fallback_decision is not None
                                else frozenset()
                            )
                            claim_response = self._guard_public_text(
                                player.name,
                                "discussion_claim_fallback",
                                claim_response,
                                tuple(self.state.conversation_log),
                                allow_retry=False,
                                allow_public_claim=(
                                    fallback_decision is not None
                                    and extract_public_role_claim(
                                        fallback_decision.claim, player.name
                                    )
                                    is not None
                                ),
                                allowed_public_results=authorized_results,
                            )
                            decision = parse_discussion_claim(claim_response)
                        self.state.snapshot_ledgers(
                            "after", "discussion_claim", player.name
                        )
                        if decision is not None and decision.claim not in {"none", "hold"}:
                            public_claim = "\n".join(
                                part
                                for part in (decision.claim, decision.result)
                                if part
                            )
                            publication = self._record_public_role_claim(
                                player.name,
                                public_claim,
                                claim_text=decision.claim,
                            )
                            # TASK-055: 設定で不受理にしたCOを自分の行動履歴へは残さない
                            if not publication.role_claim_disallowed:
                                self._record_action_memory(
                                    player=player,
                                    action_type="claim",
                                    day=self.state.day,
                                    phase=self.state.phase.value,
                                    summary=public_claim,
                                    reasoning_summary=self._reason_from_text(
                                        public_claim
                                    ),
                                    evidence=[
                                        f"public-claim-day-{self.state.day}-"
                                        f"{player.name}"
                                    ],
                                )
                            public_lines = []
                            if publication.role_published:
                                public_lines.append(decision.claim)
                            if publication.result_published and decision.result:
                                public_lines.append(f"[RESULT] {decision.result}")
                            if (
                                publication.duplicate_suppressed
                                and not publication.result_published
                            ):
                                public_lines = []
                            if public_lines:
                                self._append_discussion_message(
                                    Message(player.name, "\n".join(public_lines))
                                )
                    self.state.snapshot_ledgers(
                        "before", "discussion_body", player.name
                    )
                    body = discuss_body(
                        player.name,
                        self.state.day,
                        tuple(self.state.alive_players),
                        tuple(self.state.conversation_log),
                    )
                    body = self._guard_public_text(
                        player.name,
                        "discussion_body",
                        body,
                        tuple(self.state.conversation_log),
                    )
                    self.state.snapshot_ledgers(
                        "after", "discussion_body", player.name
                    )
                    # TASK-034: 検証を重大度別に扱い、本文が存在する場合はパスへ置換せず情報を保持する。
                    # 致命的違反（秘匿情報漏洩・未公開結果CO欠落）または空本文のみ安全フォールバックする。
                    round_transcript = tuple(self.state.current_round_transcript)
                    previous_message = next(
                        (
                            message
                            for message in reversed(round_transcript)
                            if message.speaker not in {"system", player.name}
                        ),
                        None,
                    )
                    consistency_findings = SelfConsistencyGuard.inspect(
                        body, player, self.state, previous_message
                    )
                    repair_findings = [
                        finding
                        for finding in consistency_findings
                        if finding.severity == "repair"
                    ]
                    if repair_findings:
                        finding = repair_findings[0]
                        self.state.add_integrity_alert(
                            speaker=player.name,
                            category=finding.category,
                            speech_id=len(self.state.conversation_log) + 1,
                            summary=finding.summary,
                        )
                        body = (
                            "I cannot verify a statement consistent with my past action records,"
                            "so I will hold this judgment."
                        )
                    else:
                        for finding in consistency_findings:
                            self.state.add_integrity_alert(
                                speaker=player.name,
                                category=finding.category,
                                speech_id=len(self.state.conversation_log) + 1,
                                summary=finding.summary,
                            )
                    diagnosis = diagnose_discussion_body(
                        body,
                        player,
                        self.state,
                        previous_message=previous_message,
                        round_transcript=round_transcript,
                    )
                    if diagnosis.fatal:
                        self.state.add_integrity_alert(
                            speaker=player.name,
                            category=(
                                diagnosis.reasons[0] if diagnosis.reasons else "discussion_invalid"
                            ),
                            speech_id=len(self.state.conversation_log) + 1,
                            summary=(
                                "Empty body or hidden info detected; publishing withheld, replaced with Pass"
                                if not body.strip()
                                else "Replaced with Pass after detecting hidden info or an unpublished divination result"
                            ),
                        )
                        body = "Pass"
                    elif diagnosis.repairable:
                        self.state.add_integrity_alert(
                            speaker=player.name,
                            category=diagnosis.reasons[0] if diagnosis.reasons else "discussion_repair",
                            speech_id=len(self.state.conversation_log) + 1,
                            summary=(
                                "Discussion body kept as repairable: "
                                + ",".join(diagnosis.reasons)
                            ),
                        )
                    elif diagnosis.warning:
                        self.state.add_integrity_alert(
                            speaker=player.name,
                            category=diagnosis.reasons[0] if diagnosis.reasons else "discussion_quality",
                            speech_id=len(self.state.conversation_log) + 1,
                            summary=(
                                "Discussion body quality warning (kept public): "
                                + ",".join(diagnosis.reasons)
                            ),
                        )
                    if body:
                        self._append_discussion_message(Message(player.name, body))
                        self._record_model_inference(player.name, body)
                        self._record_public_speech_memory(player, body)
                        # TASK-047: 公開発言から未確定解釈だけを仮説台帳へ登録する
                        self.state.record_hypotheses_from_speech(
                            player.name,
                            extract_discussion_body(body) or body,
                            self.state.day,
                            turn,
                        )
                        self.state.snapshot_hypothesis_presentation(player.name)
                        # TASK-048: 既出論点台帳へ論点（主張・質問・回答・投票提案）を登録し、
                        # 次ターンの要約に使う（本文の再掲はしない）。
                        self.state.record_topics_from_speech(
                            player.name, extract_discussion_body(body) or body, self.state.day, turn
                        )
                    # モデル推理メモの抽出・保存（discussion_bodyパス）
                    raw_body_for_reasoning = body
                    reasoning_memo = extract_reasoning_memo(raw_body_for_reasoning)
                    if reasoning_memo:
                        self.state.store_model_reasoning_memo(player.name, reasoning_memo)
                        # 本人限定の私的仮説として台帳へ登録する（他プレイヤーには渡さない）
                        self.state.record_hypotheses_from_reasoning_memo(
                            player.name, reasoning_memo, self.state.day, turn
                        )
                else:
                    self.state.snapshot_ledgers("before", "discussion", player.name)
                    content = self.actions.discuss(
                        player.name,
                        self.state.day,
                        tuple(self.state.alive_players),
                        tuple(self.state.conversation_log),
                    )
                    structured_output = parse_discussion_output(content)
                    claim_text = (
                        structured_output.claim
                        if structured_output is not None
                        else content
                    )
                    result_text = (
                        structured_output.result
                        if structured_output is not None
                        else content
                    )
                    allow_public_claim = (
                        extract_public_role_claim(claim_text, player.name) is not None
                        or SecretLeakGuard.allows_public_claim(content, player)
                    )
                    authorized_results = self._authorized_public_results(
                        player.name, claim_text, result_text
                    )
                    content = self._guard_public_text(
                        player.name,
                        "discussion",
                        content,
                        tuple(self.state.conversation_log),
                        allow_public_claim=allow_public_claim,
                        allowed_public_results=authorized_results,
                    )
                    self.state.snapshot_ledgers("after", "discussion", player.name)
                    structured_output = parse_discussion_output(content)
                    if structured_output is None:
                        publication = self._record_public_role_claim(
                            player.name, content
                        )
                        if (
                            extract_public_role_claim(content, player.name) is not None
                            and not publication.role_claim_disallowed
                        ):
                            self._record_action_memory(
                                player=player,
                                action_type="claim",
                                day=self.state.day,
                                phase=self.state.phase.value,
                                summary=content,
                                reasoning_summary=self._reason_from_text(content),
                                evidence=[
                                    f"public-claim-day-{self.state.day}-"
                                    f"{player.name}"
                                ],
                            )
                        # TASK-055: ブラフCOを許可しない設定では、CO・結果行だけを除去して
                        # 公開する（それ以外の本文は一字一句そのまま残し、再生成も要求しない）。
                        body = (
                            self._strip_role_claim_text(content, player.name)
                            if publication.role_claim_disallowed
                            else self._remove_duplicate_claim(content, player.name)
                            if publication.duplicate_suppressed
                            else content
                        )
                        if body:
                            self._append_discussion_message(
                                Message(player.name, body)
                            )
                            self._record_model_inference(player.name, body)
                            # TASK-047: マーカーなしの自由本文（legacy応答）からも未確定解釈を
                            # 台帳へ登録する。確定事実の登録は行わない。
                            self.state.record_hypotheses_from_speech(
                                player.name,
                                extract_discussion_body(body) or body,
                                self.state.day,
                                turn,
                            )
                            self.state.snapshot_hypothesis_presentation(player.name)
                            # TASK-048: legacy応答の自由本文からも既出論点を登録する。
                            self.state.record_topics_from_speech(
                                player.name,
                                extract_discussion_body(body) or body,
                                self.state.day,
                                turn,
                            )
                    else:
                        publication = _ClaimPublication()
                        if structured_output.claim not in {"none", "hold"}:
                            claim_parts = [
                                part
                                for part in (
                                    structured_output.claim,
                                    structured_output.result,
                                    structured_output.body,
                                )
                                if part
                            ]
                            publication = self._record_public_role_claim(
                                player.name,
                                "\n".join(claim_parts),
                                claim_text=structured_output.claim,
                            )
                            # TASK-055: 設定で不受理にしたCOを自分の行動履歴へは残さない
                            if not publication.role_claim_disallowed:
                                self._record_action_memory(
                                    player=player,
                                    action_type="claim",
                                    day=self.state.day,
                                    phase=self.state.phase.value,
                                    summary="\n".join(claim_parts),
                                    reasoning_summary=self._reason_from_text(
                                        "\n".join(claim_parts)
                                    ),
                                    evidence=[
                                        f"public-claim-day-{self.state.day}-"
                                        f"{player.name}"
                                    ],
                                )
                        body = structured_output.body
                        if publication.duplicate_suppressed:
                            body = self._remove_duplicate_claim(body, player.name)
                        if body:
                            previous_message = next(
                                (
                                    message
                                    for message in reversed(
                                        self.state.current_round_transcript
                                    )
                                    if message.speaker
                                    not in {"system", player.name}
                                ),
                                None,
                            )
                            consistency_findings = SelfConsistencyGuard.inspect(
                                body, player, self.state, previous_message
                            )
                            repair_findings = [
                                finding
                                for finding in consistency_findings
                                if finding.severity == "repair"
                            ]
                            if repair_findings:
                                finding = repair_findings[0]
                                self.state.add_integrity_alert(
                                    speaker=player.name,
                                    category=finding.category,
                                    speech_id=len(self.state.conversation_log) + 1,
                                    summary=finding.summary,
                                )
                                body = (
                                    "I cannot verify a statement consistent with my past action records,"
                                    "so I will hold this judgment."
                                )
                            else:
                                for finding in consistency_findings:
                                    self.state.add_integrity_alert(
                                        speaker=player.name,
                                        category=finding.category,
                                        speech_id=len(self.state.conversation_log) + 1,
                                        summary=finding.summary,
                                    )
                            self._append_discussion_message(
                                Message(player.name, body)
                            )
                            self._record_model_inference(player.name, body)
                            self._record_public_speech_memory(player, body)
                            # TASK-047: 公開発言から未確定解釈だけを仮説台帳へ登録する
                            self.state.record_hypotheses_from_speech(
                                player.name, body, self.state.day, turn
                            )
                            self.state.snapshot_hypothesis_presentation(player.name)
                            # TASK-048: 既出論点台帳へ登録（次ターンの要約・連呼計測に使う）
                            self.state.record_topics_from_speech(
                                player.name, body, self.state.day, turn
                            )
                    # モデル推理メモの抽出・保存（structured outputパス）
                    reasoning_memo = None
                    if structured_output is not None and structured_output.reasoning:
                        reasoning_memo = structured_output.reasoning
                    else:
                        reasoning_memo = extract_reasoning_memo(content)
                    if reasoning_memo:
                        self.state.store_model_reasoning_memo(player.name, reasoning_memo)
                        # 本人限定の私的仮説として台帳へ登録する（他プレイヤーには渡さない）
                        self.state.record_hypotheses_from_reasoning_memo(
                            player.name, reasoning_memo, self.state.day, turn
                        )
                if self.state.reasoning_memory_enabled:
                    self.state.refresh_reasoning_memos()
        self.state.phase = Phase.VOTING

    def run_voting(self) -> None:
        if (
            self.state.phase is not Phase.VOTING
            or self.state.voting_completed_day == self.state.day
        ):
            return

        alive = tuple(self.state.alive_players)
        if not alive:
            self.check_winner()
            return

        day = self.state.day
        initial_phase_id = f"day-{day}-initial"
        runoff_phase_id = f"day-{day}-runoff"
        initial_already_completed = (
            initial_phase_id in self.state.completed_vote_phases
        )
        first_votes = self._collect_votes(
            alive, alive, runoff=False, phase_id=initial_phase_id, attempt=1
        )
        tied = self._top_voted(first_votes)
        final_votes = first_votes
        if not initial_already_completed:
            self.state.vote_history.append((day, False, dict(first_votes)))
        if len(tied) > 1:
            runoff_already_completed = (
                runoff_phase_id in self.state.completed_vote_phases
            )
            final_votes = self._collect_votes(
                alive,
                tuple(tied),
                runoff=True,
                phase_id=runoff_phase_id,
                attempt=2,
            )
            if not runoff_already_completed:
                self.state.vote_history.append((day, True, dict(final_votes)))
            tied = self._top_voted(final_votes)
        executed = tied[0] if len(tied) == 1 else self._rng.choice(tied)

        self.state.votes = final_votes
        self.state.voting_completed_day = day
        self.state.mark_dead(executed)
        self.state.executions.append((day, executed))
        self.state.fact_ledger.confirmed_facts.append(
            LedgerFact(
                day=day,
                kind="execution",
                description=f"{executed} was voted out.",
            )
        )
        self.state.conversation_log.append(
            Message("system", f"{executed} was voted out.")
        )
        # TASK-056: 処刑を時系列台帳へ登録する（この時点以降の発言だけが依拠できる）
        self.state.public_timeline.record_death(day=day, name=executed, executed=True)
        self.check_winner()
        if self.state.winner is None:
            self.state.day += 1
            self.state.phase = Phase.NIGHT_WOLF
        self._emit_voting_event(
            "voting_phase_ended",
            voting_phase_id=f"day-{day}",
            day=day,
            executed=executed,
            next_phase=self.state.phase.value,
        )

    def check_winner(self) -> str | None:
        living_wolves = sum(
            player.is_alive and player.role is Role.WOLF for player in self.state.players
        )
        living_madmen = sum(
            player.is_alive and player.role is Role.MADMAN
            for player in self.state.players
        )
        living_village = sum(
            player.is_alive and player.role in (Role.SEER, Role.VILLAGER)
            for player in self.state.players
        )
        self.state.winner_counts = {
            "living_wolves": living_wolves,
            "living_madmen": living_madmen,
            "living_village": living_village,
        }
        previous_winner = self.state.winner
        if living_wolves == 0:
            self.state.winner = "village"
            self.state.winner_reason = "no_living_wolves"
        else:
            if living_wolves >= living_village:
                self.state.winner = "werewolf"
                self.state.winner_reason = "living_wolves_reached_village_parity"

        if self.state.winner is not None:
            self.state.phase = Phase.ENDED
            if previous_winner is None and self.event_callback is not None:
                self.event_callback(
                    {
                        "type": "winner_decided",
                        "winner": self.state.winner,
                        "reason": self.state.winner_reason,
                        "counts": dict(self.state.winner_counts),
                    }
                )
        return self.state.winner

    def _record_public_role_claim(
        self, speaker: str, content: str | None, claim_text: str | None = None
    ) -> _ClaimPublication:
        if not isinstance(content, str):
            content = ""
        claimed_role = extract_public_role_claim(
            claim_text if claim_text is not None else content, speaker
        )
        if claimed_role is None:
            return _ClaimPublication()

        # TASK-055実装案2: allow_bluff_role_claim=False の設定では、保有していない役職のCOを
        # 台帳へ登録しない。本文や他の発言はそのまま残し、公開イベントにも試みた役職を
        # 載せない（真偽が他プレイヤーへ漏れないようにするため）。
        actor = next(
            (player for player in self.state.players if player.name == speaker),
            None,
        )
        if (
            actor is not None
            and not getattr(self.state, "allow_bluff_role_claim", True)
            and claimed_role != actor.role.value
        ):
            self.state.role_provenance.record_disallowed_claim(
                day=self.state.day,
                claimant=speaker,
                claimed_role=claimed_role,
                source=SOURCE_COMPATIBILITY_TEXT,
                discussion_turn=self.state.discussion_turn,
            )
            self._emit_public_claim(
                "role_claim_disallowed",
                day=self.state.day,
                discussion_turn=self.state.discussion_turn,
                claimant=speaker,
                body_preserved=True,
                role_line_dropped=True,
                interpretation=PUBLIC_LINE_LABEL,
            )
            return _ClaimPublication(role_claim_disallowed=True)

        names = tuple(player.name for player in self.state.players)
        ledger_claim = self.state.claim_ledger.public_claims.get(speaker)
        prior_role = ledger_claim.claimed_role if ledger_claim is not None else next(
            (
                claim.claimed_role
                for claim in self.state.public_role_claims
                if claim.claimant == speaker
            ),
            (
                Role.SEER.value
                if any(claim.claimant == speaker for claim in self.state.seer_claims)
                else None
            ),
        )
        correction = bool(
            re.search(
                r"(?:訂正|訂正します|誤りでした|撤回します"
                r"|correct|corrected|correction|withdraw|retract|was wrong|my mistake)",
                content,
                re.IGNORECASE,
            )
        )
        role_published = ledger_claim is None
        duplicate_suppressed = ledger_claim is not None and prior_role == claimed_role
        if prior_role is not None and prior_role != claimed_role:
            if not correction or (
                ledger_claim is not None and not ledger_claim.updateable
            ):
                self._emit_public_claim(
                    "claim_role_conflict",
                    day=self.state.day,
                    claimant=speaker,
                    claimed_role=claimed_role,
                    previous_role=prior_role,
                )
                return _ClaimPublication()
            role_published = True
            if ledger_claim is not None:
                ledger_claim.claimed_role = claimed_role
                ledger_claim.latest_published_day = self.state.day
                ledger_claim.latest_published_turn = self.state.discussion_turn
        elif ledger_claim is None:
            ledger_claim = ClaimRecord(
                claimant=speaker,
                claimed_role=claimed_role,
                first_published_day=self.state.day,
                first_published_turn=self.state.discussion_turn,
                latest_published_day=self.state.day,
                latest_published_turn=self.state.discussion_turn,
            )
            self.state.claim_ledger.public_claims[speaker] = ledger_claim
        if duplicate_suppressed:
            self._emit_public_claim(
                "duplicate_claim_suppressed",
                day=self.state.day,
                discussion_turn=self.state.discussion_turn,
                claimant=speaker,
                claimed_role=claimed_role,
            )
        reason_match = re.search(
            r"(?:理由|根拠|reason|basis)\s*[：:=＝]\s*([^、,，。\n]+)",
            content,
            re.IGNORECASE,
        )
        vote_match = re.search(
            r"(?:投票提案|次の提案|vote proposal|vote)\s*[：:=＝]\s*([^。\n]+)",
            content,
            re.IGNORECASE,
        )
        if role_published:
            # TASK-055: この行は「そのプレイヤーが公に語った主張」であることを明示する。
            # エンジンが発行した占い結果のIDは付けない（公開すると真偽が漏れるため、
            # 対応関係は role_provenance 台帳（管理者ログ）だけが持つ）。
            claim_ref = f"[claim:{len(self.state.public_role_claims) + 1}]"
            self.state.public_role_claims.append(
                PublicRoleClaim(
                    day=self.state.day,
                    claimant=speaker,
                    claimed_role=claimed_role,
                    reason=(
                        reason_match.group(1).strip()
                        if reason_match
                        else "correction"
                        if correction
                        else None
                    ),
                    vote_proposal=vote_match.group(1).strip() if vote_match else None,
                    source=SOURCE_COMPATIBILITY_TEXT,
                    evidence_ids=(claim_ref,),
                )
            )
            self.state.fact_ledger.public_claims.append(
                self.state.public_role_claims[-1]
            )
            self.state.role_provenance.record_claim(
                claim_ref=claim_ref,
                day=self.state.day,
                claimant=speaker,
                claimed_role=claimed_role,
                source=SOURCE_COMPATIBILITY_TEXT,
                actor_role=actor.role.value if actor is not None else None,
                discussion_turn=self.state.discussion_turn,
            )
            self._emit_public_claim(
                "claim_role_corrected"
                if correction and prior_role is not None
                else "claim_role_published",
                day=self.state.day,
                claimant=speaker,
                claimed_role=claimed_role,
                source="text_compatibility",
                claim_ref=claim_ref,
                interpretation=PUBLIC_LINE_LABEL,
            )

        if claimed_role != Role.SEER.value:
            return _ClaimPublication(
                role_published=role_published,
                duplicate_suppressed=duplicate_suppressed,
            )

        has_claimed = any(
            claim.claimant == speaker for claim in self.state.seer_claims
        )
        if not has_claimed:
            self.state.seer_claims.append(
                PublicSeerClaim(day=self.state.day, claimant=speaker)
            )
            self._emit_public_claim(
                "seer_claim",
                day=self.state.day,
                claimant=speaker,
            )

        result = extract_public_seer_result(content, names)
        if result is None:
            return _ClaimPublication(
                role_published=role_published,
                duplicate_suppressed=duplicate_suppressed,
            )
        target, claimed_result = result
        if target == speaker:
            self._emit_public_claim(
                "illegal_seer_claim_target",
                day=self.state.day,
                claimant=speaker,
                target=target,
                reason="self_target",
            )
            return _ClaimPublication(
                role_published=role_published,
                duplicate_suppressed=duplicate_suppressed,
            )
        result_key = (speaker, target)
        previous_result = self.state.claim_ledger.public_seer_results.get(result_key)
        result_published = previous_result is None or (
            previous_result != claimed_result and correction
        )
        if previous_result is not None and previous_result != claimed_result and not correction:
            self._emit_public_claim(
                "claim_role_conflict",
                day=self.state.day,
                claimant=speaker,
                claimed_role=claimed_role,
                target=target,
                claimed_result=claimed_result,
                previous_result=previous_result,
            )
            return _ClaimPublication(
                role_published=role_published,
                duplicate_suppressed=duplicate_suppressed,
            )
        if not result_published:
            return _ClaimPublication(
                role_published=role_published,
                duplicate_suppressed=duplicate_suppressed,
            )
        self.state.claim_ledger.public_seer_results[result_key] = claimed_result
        if speaker in self.state.player_self_memories:
            self.state.player_self_memories[speaker].unrevealed_info = None
        if ledger_claim is not None:
            ledger_claim.latest_published_day = self.state.day
            ledger_claim.latest_published_turn = self.state.discussion_turn
        # TASK-055実装案3: 結果行は「そのプレイヤーの申告」としてだけ登録する。CO自体は
        # 削除せず、エンジンが発行した判定（RoleEvidence）との一致可否は管理者台帳にのみ残す。
        # ここで state.seer_results（本物の占い結果）・確定FactLedger・他プレイヤーの
        # 非公開情報へはこの申告を書き込まない。
        seer_ref = f"[seer_claim:{len(self.state.seer_claims) + 1}]"
        result_claim_ref = f"[claim:{len(self.state.public_role_claims) + 1}]"
        self.state.seer_claims.append(
            PublicSeerClaim(
                day=self.state.day,
                claimant=speaker,
                target=target,
                result=claimed_result,
            )
        )
        self.state.public_role_claims.append(
            PublicRoleClaim(
                day=self.state.day,
                claimant=speaker,
                claimed_role=Role.SEER.value,
                target=target,
                result=claimed_result,
                reason=(
                    reason_match.group(1).strip()
                    if reason_match
                    else "correction"
                    if correction
                    else None
                ),
                vote_proposal=vote_match.group(1).strip() if vote_match else None,
                source=SOURCE_COMPATIBILITY_TEXT,
                evidence_ids=(result_claim_ref, seer_ref),
            )
        )
        self.state.fact_ledger.public_claims.append(
            self.state.public_role_claims[-1]
        )
        self.state.role_provenance.record_claim(
            claim_ref=result_claim_ref,
            day=self.state.day,
            claimant=speaker,
            claimed_role=Role.SEER.value,
            source=SOURCE_COMPATIBILITY_TEXT,
            actor_role=actor.role.value if actor is not None else None,
            target=target,
            result=claimed_result,
            discussion_turn=self.state.discussion_turn,
        )
        self._emit_public_claim(
            "seer_result_corrected"
            if correction and previous_result is not None
            else "seer_result_claim",
            day=self.state.day,
            claimant=speaker,
            target=target,
            claimed_result=claimed_result,
            claim_ref=result_claim_ref,
            interpretation=PUBLIC_LINE_LABEL,
        )
        return _ClaimPublication(
            role_published=role_published,
            result_published=True,
            duplicate_suppressed=duplicate_suppressed,
        )

    def _remove_duplicate_claim(self, content: str | None, speaker: str | None = None) -> str:
        if not isinstance(content, str) or not content:
            return ""
        role = extract_public_role_claim(content, speaker)
        if role is None:
            return content.strip()
        lines = content.splitlines()
        if len(lines) > 1 and extract_public_role_claim(lines[0], speaker) == role:
            return "\n".join(lines[1:]).strip()
        sentence = re.search(r"[。！？!?]", content)
        if sentence is not None and extract_public_role_claim(
            content[: sentence.end()], speaker
        ) == role:
            return content[sentence.end() :].strip()
        return ""

    def _strip_role_claim_text(self, content: str | None, speaker: str | None = None) -> str:
        """TASK-055: ブラフCOを許可しない設定で、役職CO行と占い結果行だけを除去する。

        本文の他の部分・口調・結論には一切触れない（発言の自由はそのまま残す）。
        マーカー形式（[CO] / [RESULT]）と自由文中のCO文の両方に対応し、本文全体を
        再生成したり空にしたりはしない。
        """
        if not isinstance(content, str) or not content:
            return ""
        result_line_marker = re.compile(
            r"^\s*(?:【結果】|\[RESULT\])",
            re.IGNORECASE,
        )
        kept: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue
            marker = _CLAIM_MARKER_PREFIX.match(stripped) is not None
            if marker and (
                extract_public_role_claim(stripped, speaker) is not None
                or result_line_marker.match(stripped) is not None
            ):
                continue
            sentence_end = re.search(r"[。！？!?]", stripped)
            head = stripped[: sentence_end.end()] if sentence_end else stripped
            after_sentence = stripped[sentence_end.end() :] if sentence_end else ""
            detected = extract_public_role_claim(head, speaker)
            if detected is None:
                kept.append(line)
                continue
            # CO述及句だけを最小範囲で落とす（同じ文の後半や後文まで消さない）。
            # 「I am a Villager, and I propose voting for Player 5.」のように主張句の
            # 後へ投票提案・理由が続く場合でも、それらは本文として残す。
            fragments: list[str] = []
            for clause_end in re.finditer(r"[、,，;；：:]", head):
                candidate_head = head[: clause_end.start()]
                if extract_public_role_claim(candidate_head, speaker) is not None:
                    # 区切り以降（投票提案・理由など）は本文として残す
                    fragments.append(head[clause_end.end() :].strip())
                    break
            # 区切りが無い場合はこの文全体がCOの述及なので、同じ行の後文だけを残す
            if after_sentence.strip():
                fragments.append(after_sentence.strip())
            for fragment in fragments:
                if fragment:
                    kept.append(fragment)
        return "\n".join(kept).strip()

    def _record_model_inference(self, speaker: str, content: str | None) -> None:
        if isinstance(content, str) and content.strip() and not is_pass_body(content.strip()):
            self.state.fact_ledger.model_inferences.append(
                LedgerFact(
                    day=self.state.day,
                    kind="unverified_model_statement",
                    description=content.strip(),
                    actor=speaker,
                )
            )

    def _record_action_memory(
        self,
        player: Player,
        action_type: str,
        day: int,
        phase: str,
        target: str = "",
        reason: str = "",
        summary: str = "",
        suspicion: list[str] | None = None,
        confidence: str = "low",
        unanswered: list[str] | None = None,
        evidence: list[str] | None = None,
        previous_position: str = "",
        position_change_reason: str = "",
        reasoning_summary: str | None = None,
    ) -> None:
        """Store the committed outcome, not a model's unverified action claim."""
        memory_list = self.state.player_action_memories.setdefault(player.name, [])
        previous_vote = next(
            (
                memory.vote_target
                for memory in reversed(memory_list)
                if memory.action_type == "vote"
            ),
            "",
        )
        if not previous_position:
            previous_position = previous_vote
        if (
            action_type == "vote"
            and previous_position
            and previous_position != target
            and not position_change_reason
        ):
            position_change_reason = self._latest_action_reason(
                player.name, {"discussion", "answer"}
            )
        reason = reasoning_summary if reasoning_summary is not None else reason
        if not reason and action_type in {"vote", "seer", "attack"}:
            reason = "Original reason was not recorded"
        memory = PlayerActionMemory(
            action_id=(
                f"{player.name}-{day}-{phase}-{action_type}-"
                f"{len(memory_list) + 1}"
            ),
            day=day,
            phase=phase,
            action_type=action_type,
            selected_target=target if action_type in ("seer", "attack") else "",
            vote_target=target if action_type == "vote" else "",
            public_statement_summary=summary,
            suspicion_targets=suspicion if suspicion else [],
            reasoning_summary=reason,
            confidence=confidence,
            unanswered_questions=unanswered if unanswered else [],
            evidence_ids=evidence if evidence else [],
            previous_position=previous_position,
            position_change_reason=position_change_reason,
        )
        memory_list.append(memory)
        player.action_memory = memory

# Update SelfMemory
        # Only update for alive players
        if not player.is_alive:
            return
        self_mem = self.state.player_self_memories.get(player.name)
        if self_mem is not None:
            if not self_mem.role:
                self_mem.role = {
                    Role.WOLF: "Werewolf",
                    Role.MADMAN: "Madman",
                    Role.SEER: "Seer",
                    Role.VILLAGER: "Villager",
                }[player.role]
            # Add current action to past actions (if applicable)
            # This is called after actions like discussion, voting, night actions
            # The action text should be provided by the caller
        elif player.self_memory is None:  # 8 spaces indent (the fix!)
            # Initialize new SelfMemory
            role_name = {
                Role.WOLF: "Werewolf",
                Role.MADMAN: "Madman",
                Role.SEER: "Seer",
                Role.VILLAGER: "Villager",
            }[player.role]
            memory = SelfMemory(
                role=role_name,
                partner=player.partner,
            )
            self.state.player_self_memories[player.name] = memory
            player.self_memory = memory
        
            self_mem.role = {
                    Role.WOLF: "Werewolf",
                    Role.MADMAN: "Madman",
                    Role.SEER: "Seer",
                    Role.VILLAGER: "Villager",
                }.get(player.role, str(player.role.value))
            if not self_mem.partner and player.partner:
                self_mem.partner = player.partner

            # Action formatting
            if action_type == "vote":
                self_mem.add_action(f"Day {day} Vote: {target}")
            elif action_type == "seer":
                seer_verdict = "Werewolf" if self.state.seer_results.get(target) == "wolf" else "Human"
                self_mem.add_action(f"Night {day} Divine: {target} -> {seer_verdict}")
                # Set unrevealed info
                self_mem.unrevealed_info = f"Divined {target} as {seer_verdict}"
            elif action_type == "attack":
                self_mem.add_action(f"Night {day} Attack: {target}")
            elif action_type in {"discussion", "answer"}:
                short = summary.replace("\n", " ").strip()
                if len(short) > 60:
                    short = short[:57] + "..."
                self_mem.add_action(f"Day {day} Discussion: {short}")
                if suspicion:
                    self_mem.current_stance = f"Suspecting {', '.join(suspicion)}"
                elif "pass" in short.lower() or short == "Pass":
                    self_mem.current_stance = "Passed / Low-profile"
                else:
                    self_mem.current_stance = f"Observing ({short[:30]})"

    def _record_public_speech_memory(self, player: Player, content: str | None) -> None:
        content = content if isinstance(content, str) else ""
        previous_message = next(
            (
                message
                for message in reversed(self.state.current_round_transcript)
                if message.speaker not in {"system", player.name}
            ),
            None,
        )
        previous_content = (
            previous_message.content
            if previous_message is not None and isinstance(previous_message.content, str)
            else ""
        )
        previous_is_question = bool(
            previous_message is not None
            and previous_message.speaker != player.name
            and (
                _asks_question(previous_content)
                or player.name in previous_content
                or re.search(
                    r"(?:あなた|君|\byou\b|\byour\b)", previous_content, re.IGNORECASE
                )
            )
        )
        action_type = (
            "pass"
            if is_pass_body(content)
            else "answer"
            if previous_is_question
            else "discussion"
        )
        unanswered = (
            [previous_content]
            if previous_is_question
            and previous_message is not None
            and not _answers_previous_question(content, previous_content)
            else []
        )
        suspicion = self._suspicion_targets(content)
        reason = self._reason_from_text(content)
        self._record_action_memory(
            player=player,
            action_type=action_type,
            day=self.state.day,
            phase=self.state.phase.value,
            summary=content,
            suspicion=suspicion,
            confidence=self._confidence_from_text(content),
            unanswered=unanswered,
            reasoning_summary=reason,
            evidence=[
                f"public-message-{len(self.state.conversation_log)}"
            ],
        )
        if action_type != "pass" and _asks_question(content):
            self._record_action_memory(
                player=player,
                action_type="question",
                day=self.state.day,
                phase=self.state.phase.value,
                summary=content,
                suspicion=suspicion,
                confidence=self._confidence_from_text(content),
                reasoning_summary=reason,
                evidence=[
                    f"public-message-{len(self.state.conversation_log)}"
                ],
            )

    def _latest_action_reason(
        self, player_name: str, action_types: set[str]
    ) -> str:
        for memory in reversed(
            self.state.player_action_memories.get(player_name, [])
        ):
            if memory.action_type in action_types:
                if memory.reasoning_summary:
                    return memory.reasoning_summary
                if memory.public_statement_summary:
                    return (
                        "What the player stated before voting (no explicit reason): "
                        + memory.public_statement_summary[:240]
                    )
        return "The reason at that time was not recorded"

    def _reason_from_text(self, text: str | None) -> str:
        if not isinstance(text, str) or not text:
            return ""
        match = re.search(
            r"(?:理由|根拠|変更理由|reason|basis|change reason)\s*[：:=＝]\s*([^。\n]+)",
            text,
            re.IGNORECASE,
        )
        return match.group(1).strip()[:240] if match else ""

    def _suspicion_targets(self, text: str | None) -> list[str]:
        if not isinstance(text, str) or not text:
            return []
        if not re.search(
            r"(?:疑|怪し|投票|処刑|人狼)|(?:suspect|suspicious|vote|execute|exile|werewolf)",
            text,
            re.IGNORECASE,
        ):
            return []
        return [
            player.name
            for player in self.state.players
            if player.name in text and player.name != "system"
        ]

    @staticmethod
    def _confidence_from_text(text: str | None) -> str:
        if not isinstance(text, str) or not text:
            return "medium"
        if re.search(
            r"(?:確信|間違いなく|確実に)|(?:confident|definitely|certainly|surely)",
            text,
            re.IGNORECASE,
        ):
            return "high"
        if re.search(
            r"(?:可能性|かもしれ|まだ|保留|不確か)|(?:maybe|might|possibly|still|hold|uncertain|not sure)",
            text,
            re.IGNORECASE,
        ):
            return "low"
        return "medium"

    def _emit_public_claim(self, event_type: str, **details: object) -> None:
        if self.event_callback is not None:
            self.event_callback({"type": event_type, **details})

    def _append_discussion_message(self, message: Message) -> None:
        """Append a public speech and emit it immediately for live UIs/loggers.

        TASK-056: 公開「前」の知識境界で本文を観測し、まだ起きていないイベントを根拠に
        していないかをログに残す。本文は一字一句そのまま公開する - この検査は観測専用で、
        語彙・口調・結論の自動修正や再生成要求は一切行わない（`body_modified=false`）。
        `timeline_policy="off"` では知識境界そのものを作らない（旧挙動とのA/B比較用）。
        """
        policy = normalize_day_timeline_policy(
            getattr(self.state, "timeline_policy", DAY_TIMELINE_POLICY_DEFAULT)
        )
        inspectable = message.speaker != "system" and policy != DAY_TIMELINE_POLICY_OFF
        snapshot = self.state.timeline_snapshot(message.speaker) if inspectable else None
        appended = self.state.append_discussion_message(message)
        if self.event_callback is None:
            return
        event: dict[str, object] = {
            "type": "discussion_message",
            "day": self.state.day,
            "discussion_turn": int(appended.turn or self.state.discussion_turn),
            "phase": self.state.phase.value,
            "phase_label": "Day discussion",
            "speaker": appended.speaker,
            "player": appended.speaker,
            "action": "discussion",
            "private": False,
            "content": appended.content,
        }
        if appended.speech_id is not None:
            event["speech_id"] = int(appended.speech_id)
        event.update(
            self._timeline_reference_fields(appended, snapshot, policy)
        )
        self.event_callback(event)

    def _timeline_reference_fields(
        self,
        message: Message,
        snapshot: TurnSnapshot | None,
        policy: str,
    ) -> dict[str, object]:
        """公開本文を発言時点と突き合わせる（台帳とログ用の観測値だけを返す）。"""
        if snapshot is None:
            if message.speaker == "system":
                return {}
            # off 設定: 検査自体を行わない（観測カウンタも増やさない）
            return {"timeline_observed": False, "timeline_policy": policy}
        report = inspect_public_body(
            message.content,
            snapshot,
            tuple(player.name for player in self.state.players),
        )
        timeline = self.state.public_timeline
        timeline.reference_checks += 1
        timeline.bodies_checked += 1
        # この検査は本文を触っていない（公開文字列＝モデル応答から抽出した本文そのもの）
        timeline.bodies_kept_unmodified += 1
        timeline.future_reference_findings += len(report.findings)
        timeline.attributed_references += report.attributed_references
        timeline.unresolvable_references += report.unresolvable_references
        fields: dict[str, object] = {"timeline_observed": True, "timeline_policy": policy}
        fields.update(snapshot.to_log_fields())
        fields.update(report.to_log_fields())
        return fields

    def _collect_votes(
        self,
        voters: tuple[str, ...],
        candidates: tuple[str, ...],
        runoff: bool,
        phase_id: str | None = None,
        attempt: int | None = None,
    ) -> dict[str, str]:
        day = self.state.day
        ballot_id = phase_id or f"day-{day}-{'runoff' if runoff else 'initial'}"
        vote_attempt = attempt or (2 if runoff else 1)
        votes = self.state.vote_phase_results.setdefault(ballot_id, {})
        if ballot_id in self.state.completed_vote_phases:
            return dict(votes)
        self._emit_voting_event(
            "voting_phase_started",
            voting_phase_id=ballot_id,
            day=day,
            attempt=vote_attempt,
            voters=list(voters),
            candidates=list(candidates),
        )
        living = set(self.state.alive_players)
        requested_voters = set(voters)
        if len(requested_voters) != len(voters):
            raise ValueError(f"Duplicate voters supplied for {ballot_id}.")
        if set(votes) - requested_voters:
            raise ValueError(f"Unexpected recorded voters for {ballot_id}.")
        for voter in voters:
            if voter in votes:
                continue
            eligible = tuple(
                candidate
                for candidate in candidates
                if candidate != voter and candidate in living
            )
            if not eligible:
                raise ValueError(f"No eligible living vote targets for {voter}.")
            candidate_set = ActionCandidateSet(
                action=f"vote:{ballot_id}",
                actor=voter,
                day=day,
                candidates=eligible,
                fallback=eligible[0],
            )
            self.state.action_candidate_sets[f"vote:{ballot_id}:{voter}"] = (
                candidate_set
            )
            self._emit_voting_event(
                "vote_started",
                voting_phase_id=ballot_id,
                day=day,
                voter=voter,
                attempt=vote_attempt,
            )
            self.state.snapshot_ledgers("before", candidate_set.action, voter)
            choice = self.actions.vote(
                voter, candidate_set.candidates, self.state.day, runoff
            )
            self.state.snapshot_ledgers("after", candidate_set.action, voter)
            selected = (
                choice
                if choice in candidate_set.candidates
                else candidate_set.fallback
            )
            if selected is None:
                raise RuntimeError(f"No deterministic vote fallback for {voter}.")
            votes[voter] = selected
            self._record_action_memory(
                player=self._player(voter),
                action_type="vote",
                day=day,
                phase=Phase.VOTING.value,
                target=selected,
                summary=f"{voter} voted for {selected}.",
                reasoning_summary=self._latest_action_reason(
                    voter, {"discussion", "answer"}
                ),
                evidence=[f"vote-{ballot_id}-{voter}"],
            )
            self._emit_voting_event(
                "vote_completed",
                voting_phase_id=ballot_id,
                day=day,
                voter=voter,
                attempt=vote_attempt,
                target=selected,
                fallback_used=choice not in eligible,
            )
        self.state.completed_vote_phases.add(ballot_id)
        for voter, target in votes.items():
            self.state.fact_ledger.confirmed_facts.append(
                LedgerFact(
                    day=day,
                    kind="vote",
                    description=f"{voter} voted for {target}.",
                )
            )
        self._emit_voting_event(
            "voting_ballot_completed",
            voting_phase_id=ballot_id,
            day=day,
            attempt=vote_attempt,
            votes=dict(votes),
        )
        # TASK-056: 投票の成立を時系列台帳へ登録する（以降の発言はこれを根拠にできる）
        self.state.public_timeline.record_vote_phase(
            day, votes, runoff=runoff, ballot_id=ballot_id
        )
        return dict(votes)

    def _emit_voting_event(self, event_type: str, **details: object) -> None:
        if self.event_callback is None:
            return
        self.event_callback(
            {"type": event_type, "phase": Phase.VOTING.value, **details}
        )

    @staticmethod
    def _top_voted(votes: dict[str, str]) -> list[str]:
        counts: dict[str, int] = {}
        for target in votes.values():
            counts[target] = counts.get(target, 0) + 1
        maximum = max(counts.values())
        return [target for target, count in counts.items() if count == maximum]

    def _living_by_role(self, role: Role) -> list[Player]:
        return [
            player
            for player in self.state.players
            if player.is_alive and player.role is role
        ]

    def _player(self, name: str) -> Player:
        return next(player for player in self.state.players if player.name == name)
