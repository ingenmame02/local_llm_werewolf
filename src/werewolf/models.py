import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum

from .config import (
    ACTION_BUDGET_KEYS,
    GENERATION_SAMPLING_BOUNDS,
    normalize_personality_profile,
)
from .discussion_novelty import DiscussionTopicLedger
from .leak_gate_policy import DEFAULT_LEAK_GUARD_POLICY, LeakGuardProfiles
from .hypothesis_ledger import (
    REFUTATION_SUPERSEDED,
    STATEMENT_MAX_CHARS,
    VISIBILITY_PRIVATE,
    HypothesisLedger,
    confidence_from_text,
    stance_of_sentence,
)
# TASK-055: 役職主張と役職由来情報のプロヴェナンス分離（このモジュールは models をimportしない）
from .role_provenance import (
    ALLOW_BLUFF_ROLE_CLAIM_DEFAULT,
    SOURCE_COMPATIBILITY_TEXT,
    RoleProvenanceLedger,
)
# TASK-056: Day内イベントの時系列台帳（day / discussion_turn / speech_id / sequence を残す。
# このモジュールは models をimportしないため、循環参照にはならない）
from .day_timeline import (
    DAY_TIMELINE_POLICY_DEFAULT,
    PublicTimeline,
    TimelineEvent,
    TurnSnapshot,
    build_turn_snapshot,
    normalize_day_timeline_policy,
)
# TASK-057: プロンプト契約のポリシー（simplified / legacy）。このモジュールは models を
# import しないため循環参照にならない。節の台帳・証拠ID参照・段階除去を prompts 側で使う。
from .prompt_contract import (
    PROMPT_CONTRACT_POLICY_DEFAULT,
    normalize_prompt_contract_policy,
)


class Role(str, Enum):
    WOLF = "wolf"
    MADMAN = "madman"
    SEER = "seer"
    VILLAGER = "villager"


class Phase(str, Enum):
    NIGHT_WOLF = "night_wolf"
    NIGHT_SEER = "night_seer"
    MORNING = "morning"
    DAY_DISCUSSION = "day_discussion"
    VOTING = "voting"
    ENDED = "ended"


@dataclass
class Player:
    name: str
    model: str
    role: Role
    is_alive: bool = True
    partner: str | None = None
    num_ctx: int | None = None
    timeout: float | None = None
    num_predict: int | None = None
    response_language: str = "English"
    num_predict_by_action: Mapping[str, int] | None = None
    thinking: bool | None = None
    thinking_by_action: Mapping[str, bool] | None = None
    personality_profile: str = "balanced"
    max_num_ctx: int | None = None
    recommended_num_ctx: int | None = None
    context_strategy: str | None = None
    # 役職COの先頭形式支援（TASK-039）。CO形式の案内を強調するだけで、
    # 議論の自由度・CO保留・パスの選択は制限しない。
    claim_coaching: bool = False
    # TASK-043: プレイヤー別の生成サンプリング。幻覚を起こすモデルは temperature を
    # 下げて安定させる（未指定なら CLI / モデル既定値がそのまま使える）。
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    action_memory: "PlayerActionMemory | None" = None
    self_memory: "SelfMemory | None" = None

    def __post_init__(self) -> None:
        if self.num_ctx is not None and (
            not isinstance(self.num_ctx, int)
            or isinstance(self.num_ctx, bool)
            or self.num_ctx <= 0
        ):
            raise ValueError("Player num_ctx must be a positive integer.")
        for key, context_value in (
            ("max_num_ctx", self.max_num_ctx),
            ("recommended_num_ctx", self.recommended_num_ctx),
        ):
            if context_value is not None and (
                not isinstance(context_value, int)
                or isinstance(context_value, bool)
                or context_value <= 0
            ):
                raise ValueError(f"Player {key} must be a positive integer.")
        if (
            self.max_num_ctx is not None
            and self.recommended_num_ctx is not None
            and self.recommended_num_ctx > self.max_num_ctx
        ):
            raise ValueError(
                "Player recommended_num_ctx cannot exceed max_num_ctx."
            )
        if self.context_strategy is not None and self.context_strategy not in {
            "full_transcript",
            "structured_ledger",
            "hybrid_retrieval",
            "round_transcript_memory",
        }:
            raise ValueError("Player context_strategy is not supported.")
        if self.timeout is not None and (
            not isinstance(self.timeout, (int, float))
            or isinstance(self.timeout, bool)
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("Player timeout must be a positive number.")
        if self.num_predict is not None and (
            not isinstance(self.num_predict, int)
            or isinstance(self.num_predict, bool)
            or self.num_predict <= 0
        ):
            raise ValueError("Player num_predict must be a positive integer.")
        if not isinstance(self.response_language, str) or not self.response_language.strip():
            raise ValueError("Player response_language cannot be empty.")
        # TASK-044: 個性プロファイルは free（自由発言モード）も受け、正規化したキーへ直す。
        # 別名（none/off/null相当）や None は free に解決される。
        self.personality_profile = normalize_personality_profile(
            self.personality_profile, "Player personality_profile"
        )
        if not isinstance(self.claim_coaching, bool):
            raise ValueError("Player claim_coaching must be a boolean.")
        if self.num_predict_by_action is not None:
            if not isinstance(self.num_predict_by_action, Mapping):
                raise ValueError(
                    "Player num_predict_by_action must be a mapping of action keys "
                    "to positive integers."
                )
            for key, num_predict in self.num_predict_by_action.items():
                if key not in ACTION_BUDGET_KEYS:
                    raise ValueError(
                        f"Player num_predict_by_action contains unsupported action key {key!r}."
                    )
                if isinstance(num_predict, bool) or not isinstance(num_predict, int) or num_predict <= 0:
                    raise ValueError(
                        f"Player num_predict_by_action value for {key} must be a positive integer."
                    )
        if self.thinking is not None and not isinstance(self.thinking, bool):
            raise ValueError("Player thinking must be a boolean.")
        if self.thinking_by_action is not None:
            if not isinstance(self.thinking_by_action, Mapping):
                raise ValueError(
                    "Player thinking_by_action must be a mapping of action keys "
                    "to booleans."
                )
            for key, thinking in self.thinking_by_action.items():
                if key not in ACTION_BUDGET_KEYS:
                    raise ValueError(
                        f"Player thinking_by_action contains unsupported action key {key!r}."
                    )
                if not isinstance(thinking, bool):
                    raise ValueError(
                        f"Player thinking_by_action value for {key} must be a boolean."
                    )
        # TASK-043: 生成サンプリングの範囲検証（config.GENERATION_SAMPLING_BOUNDS と共通）。
        for key, value in (
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("top_k", self.top_k),
            ("repeat_penalty", self.repeat_penalty),
        ):
            if value is None:
                continue
            low, high, requires_integer = GENERATION_SAMPLING_BOUNDS[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Player {key} must be a number.")
            if not math.isfinite(value):
                raise ValueError(f"Player {key} must be a finite number.")
            if requires_integer and int(value) != value:
                raise ValueError(f"Player {key} must be an integer.")
            if not low <= value <= high:
                raise ValueError(
                    f"Player {key} must be between {low} and {high} (got {value})."
                )


@dataclass(frozen=True)
class Message:
    speaker: str
    content: str
    # TASK-056: 公開イベントの発生時点（エンジンが公開時にスタンプする）。
    # 旧ログや手組みテストでは未設定のままでも動くよう、すべて既定値を持つ。
    day: int = 0
    turn: int = 0
    speech_id: int | None = None
    timeline_index: int | None = None


@dataclass(frozen=True)
class PublicSeerClaim:
    day: int
    claimant: str
    target: str | None = None
    result: str | None = None


@dataclass(frozen=True)
class PublicRoleClaim:
    """公開COの台帳行（TASK-055で出所と証拠IDを持つ RoleClaim）。"""

    day: int
    claimant: str
    claimed_role: str
    target: str | None = None
    result: str | None = None
    reason: str | None = None
    vote_proposal: str | None = None
    # 出所: public_claim（ツール等の明示申告）/ compatibility_text（本文から認識した行）。
    # どちらも「そのプレイヤーが公に語った」情報であり、エンジン発行情報ではない。
    source: str = SOURCE_COMPATIBILITY_TEXT
    # 公開面に立てる証拠ID（`[claim:N]` / `[seer_claim:N]`）。エンジン発行のIDはここに
    # 置かない（公開されると真偽が漏れるため、対応関係は role_provenance 側だけが知る）。
    evidence_ids: tuple[str, ...] = ()


@dataclass
class ClaimRecord:
    claimant: str
    claimed_role: str
    first_published_day: int
    first_published_turn: int
    latest_published_day: int
    latest_published_turn: int
    updateable: bool = True


@dataclass
class ClaimLedger:
    public_claims: dict[str, ClaimRecord] = field(default_factory=dict)
    public_seer_results: dict[tuple[str, str], str] = field(default_factory=dict)


@dataclass(frozen=True)
class LedgerFact:
    day: int
    kind: str
    description: str
    visibility: str = "public"
    actor: str | None = None
    # TASK-055: エンジンが発行した事実には証拠IDを付け、後からCO行と照合できるようにする
    evidence_id: str | None = None


@dataclass
class FactLedger:
    confirmed_facts: list[LedgerFact] = field(default_factory=list)
    public_claims: list[PublicRoleClaim] = field(default_factory=list)
    model_inferences: list[LedgerFact] = field(default_factory=list)


@dataclass
class ReasoningMemo:
    confirmed_facts: list[str] = field(default_factory=list)
    public_claims: list[str] = field(default_factory=list)
    suspicion_ranking: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    next_targets: list[str] = field(default_factory=list)
    private_role_facts: list[str] = field(default_factory=list)
    current_round_evidence: list[str] = field(default_factory=list)
    integrity_alerts: list[str] = field(default_factory=list)
    updates: list[str] = field(default_factory=list)
    # モデルが毎ターン書く推理メモ（誰を疑うか、気になる発言、防御行動、今後の方針）
    model_written_memo: str = ""
    # 過去ターン分のモデル推理メモ履歴（直近N分だけ保持）
    model_memo_history: list[str] = field(default_factory=list)
    MODEL_MEMO_HISTORY_LIMIT: int = 5


@dataclass
class SelfMemory:
    """Player's self memory for prompt consistency."""
    
    role: str
    partner: str | None = None
    past_actions: list[str] = field(default_factory=list, metadata={"max_size": 5})
    current_stance: str = "Neutral observation"
    unrevealed_info: str | None = None
    
    MAX_PAST_ACTIONS: int = 5
    
    def add_action(self, action: str) -> None:
        """Add a past action, maintaining max size and skipping duplicates."""
        # Skip if the same as the most recent action
        if self.past_actions and self.past_actions[-1] == action:
            return
        # Add and trim to max size
        self.past_actions.append(action)
        if len(self.past_actions) > self.MAX_PAST_ACTIONS:
            self.past_actions = self.past_actions[-self.MAX_PAST_ACTIONS:]
    
    def update_stance(self, stance: str) -> None:
        """Update current stance."""
        self.current_stance = stance
    
    def clear_unrevealed_info(self) -> None:
        """Clear unrevealed info (e.g., after seer reveals result)."""
        self.unrevealed_info = None
    
    def format_text(self) -> str:
        """Render self memory to string."""
        role_name = self.role
        partner_str = f"partner: {self.partner}" if self.partner else "None"
        
        actions_lines = ""
        if self.past_actions:
            actions_lines = "  - " + "\n  - ".join(self.past_actions)
        else:
            actions_lines = "  - None"
        
        return (
            f"- Role: {role_name} ({partner_str})\\n"
            f"- My past actions:\\n"
            f"{actions_lines}\\n"
            f"- Current stance: {self.current_stance}\\n"
            f"- Unrevealed info: {self.unrevealed_info or 'None'}"
        )

    role: str = ""
    partner: str | None = None
    past_actions: list[str] = field(default_factory=list)
    current_stance: str = "Neutral observation"
    unrevealed_info: str | None = None

    def add_action(self, action_summary: str, max_count: int = 5) -> None:
        """Add an action summary, keeping at most max_count entries."""
        if not action_summary:
            return
        if self.past_actions and self.past_actions[-1] == action_summary:
            return
        self.past_actions.append(action_summary)
        if len(self.past_actions) > max_count:
            self.past_actions = self.past_actions[-max_count:]

    def format_text(self) -> str:
        """Render self memory to string."""
        partner_str = f"partner: {self.partner}" if self.partner else "None"
        if self.role.lower() == "werewolf":
            role_line = f"- Role: Werewolf ({partner_str})"
        else:
            role_line = f"- Role: {self.role} ({partner_str})"

        actions_lines = [f"  - {act}" for act in self.past_actions]
        if not actions_lines:
            actions_str = "  - None"
        else:
            actions_str = "\n".join(actions_lines)

        unrevealed_str = self.unrevealed_info if self.unrevealed_info else "None"

        return (
            f"{role_line}\n"
            "- My past actions:\n"
            f"{actions_str}\n"
            f"- Current stance: {self.current_stance}\n"
            f"- Unrevealed info: {unrevealed_str}"
        )


@dataclass
class PlayerActionMemory:
    """プレイヤーごとの行動・判断履歴。

    過去の投票先、発言内容、疑い対象、判断理由・信頼度、
    過去からの変更理由、未回答質問などを構造化して保持する。
    次の発言で「なぜその選択になったか」を説明するために使用する。
    """

    action_id: str = ""
    day: int = 1
    phase: str = "night_wolf"
    action_type: str = ""  # "vote", "seer", "attack", "discussion", "claim", "pass", "question"
    selected_target: str = ""
    vote_target: str = ""
    public_statement_summary: str = ""
    suspicion_targets: list[str] = field(default_factory=list)
    reasoning_summary: str = ""
    confidence: str = "low"  # low | medium | high
    unanswered_questions: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    previous_position: str = ""  # 以前の投票先や疑い対象
    position_change_reason: str = ""  # 変更した場合の理由


@dataclass(frozen=True)
class IntegrityAlert:
    """A public-safety or consistency alert without exposing secret content."""

    day: int
    discussion_turn: int
    speaker: str
    category: str
    speech_id: int
    summary: str


@dataclass(frozen=True)
class ActionCandidateSet:
    action: str
    actor: str
    day: int
    candidates: tuple[str, ...]
    fallback: str | None = None

    def __post_init__(self) -> None:
        if len(self.candidates) != len(set(self.candidates)):
            raise ValueError("Action candidates must be unique.")
        if self.fallback is not None and self.fallback not in self.candidates:
            raise ValueError("Action fallback must be one of the candidates.")


@dataclass(frozen=True)
class LedgerSnapshot:
    boundary: str
    action: str
    actor: str
    day: int
    public_claims: tuple[tuple[str, str, int, int], ...]
    public_seer_results: tuple[tuple[str, str, str], ...]
    seer_results: tuple[tuple[str, str], ...]
    dead_players: tuple[str, ...]
    votes: tuple[tuple[str, str], ...]


@dataclass
class GameState:
    players: list[Player]
    day: int = 1
    phase: Phase = Phase.NIGHT_WOLF
    conversation_log: list[Message] = field(default_factory=list)
    wolf_chat_log: list[Message] = field(default_factory=list)
    seer_results: dict[str, str] = field(default_factory=dict)
    seer_result_history: list[tuple[int, str, str]] = field(default_factory=list)
    votes: dict[str, str] = field(default_factory=dict)
    vote_history: list[tuple[int, bool, dict[str, str]]] = field(default_factory=list)
    attack_deaths: list[tuple[int, str]] = field(default_factory=list)
    executions: list[tuple[int, str]] = field(default_factory=list)
    winner: str | None = None
    seed: int | None = None
    pending_attack: str | None = None
    discussion_turn: int = 0
    vote_phase_results: dict[str, dict[str, str]] = field(default_factory=dict)
    completed_vote_phases: set[str] = field(default_factory=set)
    voting_completed_day: int | None = None
    seer_claims: list[PublicSeerClaim] = field(default_factory=list)
    public_role_claims: list[PublicRoleClaim] = field(default_factory=list)
    winner_reason: str | None = None
    winner_counts: dict[str, int] = field(default_factory=dict)
    claim_ledger: ClaimLedger = field(default_factory=ClaimLedger)
    fact_ledger: FactLedger = field(default_factory=FactLedger)
    # TASK-047: 未確定解釈（仮説・矛盾・反証）は確定事実台帳とは別台帳で保持する
    hypothesis_ledger: HypothesisLedger = field(default_factory=HypothesisLedger)
    # TASK-048: 既出論点（主張・質問・回答・投票提案・立場）の台帳。次ターンの
    # 「既出論点の要約」に使い、連呼と正当な繰り返しを区別する（観測のみ・強制しない）。
    topic_ledger: DiscussionTopicLedger = field(default_factory=DiscussionTopicLedger)
    # TASK-048: 低新規性時の処理ポリシー（"publish" / "append" / "legacy_repair"）。既定 publish。
    novelty_policy: str = "publish"
    # TASK-054: 漏洩ガードのゲートポリシー（"tiered" / "legacy_strict"）。既定 tiered。
    # tiered は非公開の真の情報だけ遮断し、公開済みCO・判定の再言及や他者役職の推測は
    # 本文そのまま公開（警告・観測のみ）。legacy_strict は旧来動作でA/B比較に使う。
    leak_guard_policy: str = DEFAULT_LEAK_GUARD_POLICY
    # TASK-054: モデル別の漏洩実績（hard/soft・補正・再要求秒・本文保持率）と強度。
    leak_guard_profiles: LeakGuardProfiles = field(default_factory=LeakGuardProfiles)
    action_candidate_sets: dict[str, ActionCandidateSet] = field(default_factory=dict)
    discussion_claim_checks: set[tuple[int, str]] = field(default_factory=set)
    ledger_snapshots: list[LedgerSnapshot] = field(default_factory=list)
    # ゲーム中に判明したモデル能力（例：{"model-x": {"claim_role_tool": False}}）。
    model_capabilities: dict[str, dict[str, bool]] = field(default_factory=dict)
    reasoning_memos: dict[str, ReasoningMemo] = field(default_factory=dict)
    reasoning_memory_enabled: bool = True
    current_round_transcript: list[Message] = field(default_factory=list)
    integrity_alerts: list[IntegrityAlert] = field(default_factory=list)
    player_action_memories: dict[str, list[PlayerActionMemory]] = field(default_factory=dict)
    player_self_memories: dict[str, SelfMemory] = field(default_factory=dict)
    # TASK-038: 質問への回答の強制度（"strict" / "soft" / "none"）。既定はsoft。
    question_hint_mode: str = "soft"
    # TASK-055: ブラフとしての役職COを許すか（既定は許可）。許可時は村人・人狼・狂人のCOも
    # 公開主張として受理するが、実際の占い結果として登録されるのはエンジンが発行した
    # 情報だけである。不許可時は対象行を公開前に除去する（本文はそのまま残す）。
    allow_bluff_role_claim: bool = ALLOW_BLUFF_ROLE_CLAIM_DEFAULT
    # TASK-055: エンジン発行情報（RoleEvidence）と公開申告（PublicInterpretation）の台帳。
    # 照合結果は管理者ログ専用で、公開面では全CO行を同じ中立ラベルに揃える。
    role_provenance: RoleProvenanceLedger = field(default_factory=RoleProvenanceLedger)
    # TASK-056: 公開イベントの時系列台帳（時点・ID・短い要約のみ。発言本文は保持しない）。
    # 発言・投票・処刑・襲撃死を登順で並べ、「この時点で既知」と「まだ発生していない」
    # の境界をプロンプトとログの両方で追跡できるようにする。
    public_timeline: PublicTimeline = field(default_factory=PublicTimeline)
    # TASK-056: 時系列明示のルール設定（"explicit" / "off"）。offはA/B比較用の旧挙動。
    timeline_policy: str = DAY_TIMELINE_POLICY_DEFAULT
    # TASK-057: プロンプト契約の簡素化（"simplified" / "legacy"）。simplified では同じ制約・
    # 発言本文の再掲を1箇所に集め、コンテキスト不足時に optional 節だけを段階的に落とす。
    # legacy はA/B比較用の旧挙動（重複した命令文のまま）で、発言の口調・論点には干渉しない。
    prompt_contract_policy: str = PROMPT_CONTRACT_POLICY_DEFAULT

    def __post_init__(self) -> None:
        if self.day < 0:
            raise ValueError("Day cannot be negative.")
        # TASK-056: 時系列明示のポリシーを正規化し、台帳と同じ値に揃える
        self.timeline_policy = normalize_day_timeline_policy(self.timeline_policy)
        self.public_timeline.policy = self.timeline_policy
        # TASK-057: プロンプト契約のポリシーも正規化する（未知の設定値は既定の簡素化）
        self.prompt_contract_policy = normalize_prompt_contract_policy(
            self.prompt_contract_policy
        )
        if not 0 <= self.discussion_turn <= 3:
            raise ValueError("Discussion turn must be between zero and three.")
        names = [player.name for player in self.players]
        if len(names) != len(set(names)):
            raise ValueError("Player names must be unique.")
        for claim in self.public_role_claims:
            self.claim_ledger.public_claims.setdefault(
                claim.claimant,
                ClaimRecord(
                    claimant=claim.claimant,
                    claimed_role=claim.claimed_role,
                    first_published_day=claim.day,
                    first_published_turn=0,
                    latest_published_day=claim.day,
                    latest_published_turn=0,
                ),
            )
            if claim.target is not None and claim.result is not None:
                self.claim_ledger.public_seer_results[
                    (claim.claimant, claim.target)
                ] = claim.result
        for claim in self.seer_claims:
            if claim.target is not None and claim.result is not None:
                self.claim_ledger.public_seer_results[
                    (claim.claimant, claim.target)
                ] = claim.result
        for player in self.players:
            if player.name not in self.player_self_memories:
                role_name = {
                    Role.WOLF: "Werewolf",
                    Role.MADMAN: "Madman",
                    Role.SEER: "Seer",
                    Role.VILLAGER: "Villager",
                }.get(player.role, str(player.role.value))
                memory = SelfMemory(
                    role=role_name,
                    partner=player.partner,
                )
                self.player_self_memories[player.name] = memory
                if player.self_memory is None:
                    player.self_memory = memory

    @property
    def alive_players(self) -> list[str]:
        return [player.name for player in self.players if player.is_alive]

    def mark_dead(self, player_name: str) -> None:
        player = next((p for p in self.players if p.name == player_name), None)
        if player is None:
            raise KeyError(f"Unknown player: {player_name}")
        if not player.is_alive:
            raise ValueError(f"Player is already dead: {player_name}")
        player.is_alive = False

    def append_discussion_message(self, message: Message) -> Message:
        """Append a public discussion message and mirror it into the live round.

        TASK-056: 公開発言には発生時点（day / discussion turn / [speech:N] / 時系列順）を
        スタンプし、同じ時点で公開イベント台帳へ登録する。台帳へは短い要約とIDだけ残し、
        本文はコピーしない（コンテキスト増加を防ぐため）。
        """
        stamped = message
        if message.speaker != "system":
            speech_id = int(
                message.speech_id
                if message.speech_id is not None
                else sum(
                    1 for row in self.conversation_log if row.speaker != "system"
                )
                + 1
            )
            if (
                message.day != self.day
                or message.turn != self.discussion_turn
                or message.timeline_index is None
            ):
                stamped = replace(
                    message,
                    day=self.day,
                    turn=self.discussion_turn,
                    speech_id=speech_id,
                    timeline_index=self.public_timeline.next_sequence,
                )
            if self.phase is Phase.DAY_DISCUSSION:
                self.public_timeline.record_speech(
                    day=self.day,
                    turn=self.discussion_turn,
                    speaker=stamped.speaker,
                    speech_id=speech_id,
                    content=stamped.content,
                )
        self.conversation_log.append(stamped)
        if stamped.speaker != "system" and self.phase is Phase.DAY_DISCUSSION:
            self.current_round_transcript.append(stamped)
        return stamped

    def record_public_event(
        self,
        kind: str,
        *,
        actor: str | None = None,
        summary: str = "",
        day: int | None = None,
        turn: int | None = None,
    ) -> TimelineEvent | None:
        """公開イベント（投票・処刑・襲撃死・日の開始）を時系列台帳へ登録する。"""
        return self.public_timeline.record(
            day=self.day if day is None else int(day),
            turn=self.discussion_turn if turn is None else int(turn),
            kind=kind,
            actor=actor,
            summary=summary,
        )

    def timeline_snapshot(self, actor_name: str) -> TurnSnapshot:
        """発言直前の知識境界（既知の先行発言と、まだ起きていないイベント）を返す。"""
        snapshot = build_turn_snapshot(self, actor_name)
        # プロンプトへ出さなかった後発イベント数を観測する（未来参照の検出基準にも使う）
        withheld = len(snapshot.speakers_pending)
        if not snapshot.today_vote_done:
            withheld += 1
        if snapshot.night_pending:
            withheld += 1
        self.public_timeline.future_events_withheld += withheld
        return snapshot

    def begin_discussion_turn(self) -> None:
        self.current_round_transcript = []

    def add_integrity_alert(
        self,
        *,
        speaker: str,
        category: str,
        speech_id: int,
        summary: str,
    ) -> None:
        alert = IntegrityAlert(
            day=self.day,
            discussion_turn=self.discussion_turn,
            speaker=speaker,
            category=category,
            speech_id=speech_id,
            summary=summary,
        )
        if alert not in self.integrity_alerts:
            self.integrity_alerts.append(alert)

    def refresh_reasoning_memos(self) -> None:
        """Refresh actor-specific notes using public state and actor-owned secrets only."""
        # TASK-047: 確定事実・公開主張は本文を再掲せずIDだけ渡す。同じID付きの本文は
        # [CLAIM_LEDGER] に1回だけ存在するため、同じ証拠の重複提示が省ける。
        public_facts = [
            f"[fact:{fact.day}:{index}]"
            for index, fact in enumerate(self.fact_ledger.confirmed_facts, 1)
            if fact.visibility == "public"
            # TASK-047: 死亡・投票は [death:*] / [vote:*] のIDで1回だけ提示する
            and fact.kind not in {"attack_death", "execution", "vote"}
        ]
        public_facts.extend(
            f"[vote:{day}:{'runoff' if runoff else 'initial'}:{voter}]"
            for day, runoff, votes in self.vote_history
            for voter, target in votes.items()
        )
        public_facts.extend(
            f"[death:attack:{day}:{player}]" for day, player in self.attack_deaths
        )
        public_facts.extend(
            f"[death:execution:{day}:{player}]" for day, player in self.executions
        )
        public_claims = [
            f"[claim:{index}]" for index, claim in enumerate(self.public_role_claims, 1)
        ]
        public_claims.extend(
            f"[seer_claim:{index}]"
            for index, claim in enumerate(self.seer_claims, 1)
            if claim.target is not None and claim.result is not None
        )
        votes_by_target: dict[str, list[str]] = {}
        for day, runoff, votes in self.vote_history:
            ballot = "runoff" if runoff else "initial"
            for voter, target in votes.items():
                votes_by_target.setdefault(target, []).append(
                    f"[vote:{day}:{ballot}:{voter}]"
                )
        suspicion_ranking = [
            f"{target}: {len(sources)} recorded vote(s); evidence {', '.join(sources)}"
            for target, sources in sorted(
                votes_by_target.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        ]
        speech_records: list[tuple[int, Message]] = []
        for message in self.conversation_log:
            if message.speaker != "system":
                speech_records.append((len(speech_records) + 1, message))
        # TASK-047: 死亡者への質問は回答不能なので、未回答項目には注記して渡す
        dead_names_for_notes = frozenset(
            player.name for player in self.players if not player.is_alive
        )

        for player in self.players:
            unresolved: list[str] = []
            # TASK-047: 死亡者宛の質問は未回答のまま永久に提示しない（回答不能な未解決項目は
            # プロンプト生成前に解決済みとし、死亡者への質問誘導を作らない）。
            if player.is_alive:
                for position, (speech_id, message) in enumerate(speech_records):
                    if message.speaker == player.name:
                        continue
                    is_question = (
                        "?" in message.content
                        or "？" in message.content
                        or any(
                            cue in message.content
                            for cue in (
                                "質問",
                                "聞きたい",
                                "確認したい",
                                "教えて",
                                "question",
                                "ask",
                                "tell me",
                                "check",
                            )
                        )
                    )
                    addresses_player = (
                        player.name in message.content
                    )
                    later_reply = any(
                        later.speaker == player.name
                        for _, later in speech_records[position + 1 :]
                    )
                    if is_question and addresses_player and not later_reply:
                        dead_reference = next(
                            (name for name in dead_names_for_notes if name in message.content),
                            None,
                        )
                        note = (
                            " (dead players cannot answer; ask living players about the record instead)"
                            if dead_reference
                            else ""
                        )
                        unresolved.append(
                            f"[speech:{speech_id}] Unanswered question from {message.speaker}{note}"
                        )

            next_targets = [
                target
                for target in (
                    item.split(":", 1)[0] for item in suspicion_ranking
                )
                if target != player.name and target in self.alive_players
            ][:3]
            if not next_targets:
                next_targets = [
                    name for name in self.alive_players if name != player.name
                ][:3]

            private_facts: list[str] = []
            if player.role is Role.SEER:
                private_facts = [
                    f"[private_seer:{index}] {target}={result}"
                    for index, (target, result) in enumerate(
                        self.seer_results.items(), 1
                    )
                ]
            elif player.role is Role.WOLF and player.partner is not None:
                private_facts = [f"[private_wolf_partner] {player.partner}"]

            current_round_objects = {
                id(message) for message in self.current_round_transcript
            }
            current_round_evidence = [
                f"[speech:{speech_id}] {message.speaker}: {message.content}"
                for speech_id, message in speech_records
                if id(message) in current_round_objects
                and message.speaker != player.name
            ][-12:]
            integrity_alerts = [
                f"[speech:{alert.speech_id}] {alert.speaker}: {alert.category} - {alert.summary}"
                for alert in self.integrity_alerts
                if alert.speaker != player.name
            ][-8:]

            previous = self.reasoning_memos.get(player.name)
            current_parts = {
                "confirmed_facts": public_facts,
                "public_claims": public_claims,
                "suspicion_ranking": suspicion_ranking,
                "unresolved_questions": unresolved,
                "next_targets": next_targets,
                "private_role_facts": private_facts,
                "current_round_evidence": current_round_evidence,
                "integrity_alerts": integrity_alerts,
            }
            updates = (
                ["Reasoning memo initialized"]
                if previous is None
                else [
                    f"Updated {key}"
                    for key, value in current_parts.items()
                    if getattr(previous, key) != value
                ]
            )
            self.reasoning_memos[player.name] = ReasoningMemo(
                **current_parts,
                updates=updates or ["No change since previous turn"],
            )

    def store_model_reasoning_memo(self, player_name: str, reasoning_text: str) -> None:
        """モデルが書いた推理メモを保存し、履歴へ追加する（本人限定・公開本文には含めない）。"""
        if not isinstance(reasoning_text, str) or not reasoning_text.strip():
            return
        memo = self.reasoning_memos.get(player_name)
        if memo is None:
            memo = ReasoningMemo()
            self.reasoning_memos[player_name] = memo
        memo.model_written_memo = reasoning_text.strip()
        memo.model_memo_history.append(reasoning_text.strip())
        if len(memo.model_memo_history) > memo.MODEL_MEMO_HISTORY_LIMIT:
            memo.model_memo_history = memo.model_memo_history[-memo.MODEL_MEMO_HISTORY_LIMIT:]

    def _speech_id_for(self, speaker: str, content: str) -> str:
        """発言の証拠ID（reasoning memo の `[speech:N]` と同じ番号）を返す。"""
        index = 0
        for message in self.conversation_log:
            if message.speaker == "system":
                continue
            index += 1
            if message.speaker == speaker and message.content == content:
                return f"[speech:{index}]"
        return f"[speech:{index}]"

    def record_hypotheses_from_speech(
        self, speaker: str, content: str | None, day: int, turn: int
    ) -> int:
        """公開発言から未確定解釈（疑い・信頼）だけを仮説台帳へ登録する。

        確定事実（死亡・投票・CO・占い結果の公開）は FactLedger 側が持つため、
        ここでは発言の解釈だけを `evidence_ids` 付きで登録する。同じ立場・本文の
        再掲は台帳側で統合され、新しいレコードは作らない。
        """
        if not isinstance(content, str) or not content.strip():
            return 0
        evidence_id = self._speech_id_for(speaker, content)
        # TASK-056: 発生時点（D日Tターン#時系列順）を覚えさせて、同じ論点の再掲を
        # 新しい事実として読み替えないための手がかりにする（本文は追加しない）
        observed_point = f"D{day}T{turn}#{self.public_timeline.next_sequence}"
        confidence = confidence_from_text(content)
        created = 0
        for sentence in re.split(r"(?<=[。！？!?])\s*|\n+", content):
            if created >= 4:
                break
            stance = stance_of_sentence(sentence)
            if stance is None:
                continue
            for player in self.players:
                if player.name == speaker or player.name not in sentence:
                    continue
                if created >= 4:
                    break
                created += int(
                    self.hypothesis_ledger.record(
                        subject=player.name,
                        actor=speaker,
                        observed_at=observed_point,
                        day=day,
                        turn=turn,
                        stance=stance,
                        statement=sentence.strip()[:STATEMENT_MAX_CHARS],
                        evidence_ids=[evidence_id],
                        confidence=confidence,
                    )
                )
        return created

    def record_hypotheses_from_reasoning_memo(
        self, actor_name: str, memo_text: str | None, day: int, turn: int
    ) -> int:
        """モデル自身の推理メモから仮説を登録する（本人だけの私的仮説として扱う）。"""
        if not isinstance(memo_text, str) or not memo_text.strip():
            return 0
        evidence_id = f"[memo:{actor_name}-day-{day}-turn-{turn}]"
        # TASK-056: 推理メモ由来の仮説にも観測時点を持たせる（本人専用）
        observed_point = f"D{day}T{turn}#{self.public_timeline.next_sequence}"
        confidence = confidence_from_text(memo_text)
        created = 0
        for sentence in re.split(r"(?<=[。！？!?])\s*|\n+", memo_text):
            if created >= 4:
                break
            stance = stance_of_sentence(sentence)
            if stance is None:
                continue
            for player in self.players:
                if player.name == actor_name or player.name not in sentence:
                    continue
                if created >= 4:
                    break
                created += int(
                    self.hypothesis_ledger.record(
                        subject=player.name,
                        actor=actor_name,
                        observed_at=observed_point,
                        day=day,
                        turn=turn,
                        stance=stance,
                        statement=sentence.strip()[:STATEMENT_MAX_CHARS],
                        evidence_ids=[evidence_id],
                        confidence=confidence,
                        visibility=VISIBILITY_PRIVATE,
                    )
                )
        return created

    def resolve_hypothesis_ledger(self) -> None:
        """プロンプト生成前に死亡者参照・反証・重複COの矛盾を台帳へ反映する。"""
        self.hypothesis_ledger.resolve(self)

    def render_hypotheses_for(self, actor_name: str) -> tuple[str, dict[str, object]]:
        """本人向けの仮説台帳（短縮版）と診断情報を返す。"""
        return self.hypothesis_ledger.render_for(actor_name)

    def snapshot_hypothesis_presentation(self, actor_name: str) -> None:
        """このターンで本人へ提示した仮説状態を記録する（次ターンの差分基準）。"""
        self.hypothesis_ledger.snapshot_presentation(actor_name)

    def record_topics_from_speech(
        self, speaker: str, content: str | None, day: int, turn: int
    ) -> int:
        """公開発言から既出論点（主張・質問・回答・投票提案・立場）を台帳へ登録する。

        本文の再掲はせず、短縮本文＋`[speech:N]` IDで保持する。同じ論点の再掲は
        seen を増やして統合し、新根拠を伴う再確認は正当な繰り返しとして数える。
        """
        if not isinstance(content, str) or not content.strip():
            return 0
        return self.topic_ledger.record_speech(
            speaker=speaker,
            content=content,
            day=day,
            turn=turn,
            speech_id=self._speech_id_for(speaker, content),
            player_names=tuple(player.name for player in self.players),
        )

    def topic_summary_for(self, actor_name: str, day: int) -> tuple[str, dict[str, object]]:
        """次ターンへ渡す既出論点の要約（観測のみ・1ブロックだけ）を返す。"""
        dead_names = frozenset(player.name for player in self.players if not player.is_alive)
        return self.topic_ledger.summarize_for(actor_name, day, dead_names=dead_names)

    def own_stances_for(self, actor_name: str) -> dict[str, str]:
        """本人がこれまでに取った立場（対象→立場）を返す（立場変更の計測基準）。"""
        stances: dict[str, str] = {}
        for record in self.hypothesis_ledger.records.values():
            if (
                record.generated_by != actor_name
                or record.refutation_status == REFUTATION_SUPERSEDED
            ):
                continue
            stances[record.subject] = record.stance
        return stances

    def snapshot_ledgers(
        self, boundary: str, action: str, actor: str
    ) -> LedgerSnapshot:
        snapshot = LedgerSnapshot(
            boundary=boundary,
            action=action,
            actor=actor,
            day=self.day,
            public_claims=tuple(
                sorted(
                    (
                        claim.claimant,
                        claim.claimed_role,
                        claim.first_published_day,
                        claim.first_published_turn,
                    )
                    for claim in self.claim_ledger.public_claims.values()
                )
            ),
            public_seer_results=tuple(
                sorted(
                    (claimant, target, result)
                    for (claimant, target), result
                    in self.claim_ledger.public_seer_results.items()
                )
            ),
            seer_results=tuple(sorted(self.seer_results.items())),
            dead_players=tuple(
                player.name for player in self.players if not player.is_alive
            ),
            votes=tuple(sorted(self.votes.items())),
        )
        self.ledger_snapshots.append(snapshot)
        return snapshot
