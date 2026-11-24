"""TASK-060: 議論プロンプトの節順を「実測字数 × attentionモデル」で選び直す。

TASK-058 の layout v2 は primacy（ルール→台帳→自分のアンカー）／中部（派生要約）／
pre-recency（証拠→現在質問）／recency（秘匿境界→契約→出力形式→チェックリスト）を
宣言した。ただし実コードを見ると、次の4点が設計思想に反していた。

1. `[Werewolf Theory Card]`（約2.7k字）が `_fact_header()` の末尾に貼り付いて先頭帯に
   居た。`[FACT_LEDGER]` のスパンに吸収されて `anchor` 字数に計上され、optional の
   段階除去も効かない。しかも推論誘導（単独COの真置きなど）を先頭に置いている。
2. `[Discussion Guidelines & Logical Reasoning]` が `[Instructions]`（出力形式仕様）より
   後ろにあり、出力形式の指定が終端から押し下げられていた。
3. 同じ制約（一人称・記録外を使わない・死者の扱い・応答言語）がヘッダと契約と
   ガイドラインに再掲されていた。
4. 自分の一貫性アンカー（今日の発言・自分の推理メモ）が現在の質問と発言証拠から
   遠く、時系列欄も中部に沈んでいた。

本モジュールのコストモデルは「内容」ではなく**配置だけの構造的な代理指標**を測る。

    score = Σ (節の実測字数) × role_weight × attention_prior(中心位置) − penalties

attention_prior は lost-in-the-middle の定性的知見（先頭と末尾が見えやすく中部が最も
見えにくい）をU字カーブで近似したもので、実機LLMで校正した値ではない。あくまで
「同じ内容のまま配置だけを比べる」ための透明なコスト関数であり、`--prompt-layout` で
実機A/Bして最終決定する前提で使う。パラメータはこのファイルに定数として書く。

設計思想との対応:
- コンテキスト節約: optional 節（理論カード含む）を中部へ寄せ、割り当てが足りないときに
  落ちる順を明確にする。同じ制約の再掲は v3 のヘッダ側だけで削る。
- モデルの一貫性: 自分の一貫性アンカーを現在質問の直前へ群やし、距離で測る。
  anchor の本文はどの設定でも削除しない（core は drop 不可）。
- 合理的な議論: 中立な案内は残すが、出力形式仕様の後ろに回さない。
- 自由の担保: 口調・文量・論点・結論・推理順を決める文言は新規に追加しない。
  誘導の強い節（理論カード）は先頭から外す方向にしか動かさない。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .core_compression import (
    PROMPT_LAYOUT_LEGACY,
    PROMPT_LAYOUT_V2,
    PROMPT_LAYOUT_V2_ORDER,
)
from .prompt_contract import find_sections

# ---------------------------------------------------------------------------
# ポリシー（既定 v3。v2 はTASK-058の配置、legacy はTASK-057までの字节順）
# ---------------------------------------------------------------------------

PROMPT_LAYOUT_POLICY_AUTO = "auto"
PROMPT_LAYOUT_POLICY_V2 = "v2"
PROMPT_LAYOUT_POLICY_V3 = "v3"
# TASK-062: v4 は `[CONTINUITY_CHECKPOINT]`（日をまたぐ論理状態）を含む配置で既定になる。
PROMPT_LAYOUT_POLICY_V4 = "v4"
PROMPT_LAYOUT_POLICIES: tuple[str, ...] = (
    PROMPT_LAYOUT_POLICY_V4,
    PROMPT_LAYOUT_POLICY_V3,
    PROMPT_LAYOUT_POLICY_V2,
    PROMPT_LAYOUT_POLICY_AUTO,
)
PROMPT_LAYOUT_POLICY_DEFAULT = PROMPT_LAYOUT_POLICY_V4

# レイアウトラベル（ログに載る値）。v2/legacy は core_compression の既存ラベルを再利用する。
PROMPT_LAYOUT_V3 = "v3_grounding"
PROMPT_LAYOUT_V4 = "v4_continuity"


def normalize_prompt_layout_policy(value: object) -> str:
    """設定値を \"v3\" / \"v2\" / \"auto\" へ正規化する（不明値は既定の v3）。

    `auto` は契約ポリシーに追随する指定で、legacy では旧字节順、simplified では v3 を選ぶ。
    bool は True=v3 / False=v2 として扱う（A/B切り替えの簡易指定）。
    """
    if isinstance(value, bool):
        return PROMPT_LAYOUT_POLICY_V4 if value else PROMPT_LAYOUT_POLICY_V2
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in PROMPT_LAYOUT_POLICIES:
            return candidate
        if candidate in {PROMPT_LAYOUT_V4, "continuity", "task062", "on", "true", "best"}:
            return PROMPT_LAYOUT_POLICY_V4
        if candidate in {PROMPT_LAYOUT_V3, "grounding", "task060"}:
            return PROMPT_LAYOUT_POLICY_V3
        if candidate in {PROMPT_LAYOUT_V2, "attention", "task058"}:
            return PROMPT_LAYOUT_POLICY_V2
        if candidate in {PROMPT_LAYOUT_LEGACY, "legacy", "legacy_v1", "off", "false", "old"}:
            return PROMPT_LAYOUT_POLICY_AUTO
    return PROMPT_LAYOUT_POLICY_DEFAULT


def resolve_prompt_layout(policy: object, contract_policy: object) -> str:
    """契約ポリシーとあわせて、実際に使うレイアウトラベルを決める。

    `prompt_contract_policy=legacy` ではTASK-057までの字节順を優先する（A/B比較の基準側を
    壊さないため、ここではLayout指定より契約ポリシーが勝つ）。simplified では
    auto / v3 が `v3_grounding`、v2 が `v2_attention` を選ぶ。
    """
    resolved = normalize_prompt_layout_policy(policy)
    contract_is_legacy = str(contract_policy or "").strip().lower() in {
        "legacy",
        "legacy_v1",
        "off",
        "before",
        "old",
        "false",
    }
    if contract_is_legacy:
        return PROMPT_LAYOUT_LEGACY
    if resolved == PROMPT_LAYOUT_POLICY_V2:
        return PROMPT_LAYOUT_V2
    if resolved == PROMPT_LAYOUT_POLICY_V3:
        return PROMPT_LAYOUT_V3
    return PROMPT_LAYOUT_V4


def prompt_layout_policy_label(policy: object) -> str:
    """CLI・ログ用の正規化ラベル（\"v3\" / \"v2\" / \"auto\"）。"""
    return normalize_prompt_layout_policy(policy)


# ---------------------------------------------------------------------------
# attention prior（先頭と末尾が見えやすく、中部が最も見えにくいというU字近似）
# ---------------------------------------------------------------------------

# 先頭この割合を超えると primacy の利得が消える
ATTENTION_PRIMACY_EDGE = 0.10
# 末尾この割合から recency の利得が立ち上がる
ATTENTION_RECENCY_EDGE = 0.35
# 中部の相対的な見えやすさ（ここが lost-in-the-middle の底）
ATTENTION_MIDDLE_BASE = 0.60
# 先頭の最大利得 / 末尾の最大利得（末尾は「いま読む命令」なのでやや大きく取る）
ATTENTION_PRIMACY_GAIN = 0.60
ATTENTION_RECENCY_GAIN = 0.65


def attention_prior(fraction_center: float) -> float:
    """節の中心位置（プロンプト全体の割合 0..1）からの相対的な見えやすさ。"""
    center = min(1.0, max(0.0, float(fraction_center)))
    primacy_gain = max(0.0, 1.0 - center / ATTENTION_PRIMACY_EDGE) * ATTENTION_PRIMACY_GAIN
    tail_start = 1.0 - ATTENTION_RECENCY_EDGE
    recency_gain = (
        max(0.0, (center - tail_start) / ATTENTION_RECENCY_EDGE) * ATTENTION_RECENCY_GAIN
    )
    return round(ATTENTION_MIDDLE_BASE + max(primacy_gain, recency_gain), 4)


def attention_zone_for_center(center_fraction: float) -> str:
    """中心位置から attention 帯（primacy / weak_middle / recency）を割り当てる。"""
    if center_fraction <= ATTENTION_PRIMACY_EDGE:
        return "primacy"
    if center_fraction >= 1.0 - ATTENTION_RECENCY_EDGE:
        return "recency"
    return "weak_middle"


ATTENTION_ZONES: tuple[str, ...] = ("primacy", "weak_middle", "recency")

# ---------------------------------------------------------------------------
# 節の役割（何を「今」使うか）。weight が高いほど配置の恩恵が大きい。
# ---------------------------------------------------------------------------

ROLE_RULES = "game_rules"  # ルール・勝敗条件（常に効く。先頭が最適）
ROLE_LANGUAGE = "output_language"  # 応答言語の指定
ROLE_ENGINE_FACT = "engine_fact"  # エンジンが発行した確定事実
ROLE_PUBLIC_INDEX = "claim_index"  # 公開主張の索引（ID付き台帳）
ROLE_SELF_HISTORY = "self_history"  # 自分の行動記録・今日の発言・自分の推理メモ
ROLE_ENGINE_PRIVATE = "engine_private"  # 本人限定のエンジン情報（後帯に置かない）
ROLE_TEMPORAL = "temporal_frame"  # 時点とまだ起きていないこと
ROLE_EVIDENCE = "evidence_bodies"  # 発言本文そのもの
ROLE_TASK = "current_task"  # いま行う行動と現在質問
ROLE_FORMAT = "output_format"  # 出力形式と最終チェックリスト
ROLE_SECRECY = "secrecy_boundary"  # 秘匿境界・強い契約
ROLE_ADVISORY = "advisory"  # 議論を促す中立案内
ROLE_HEURISTIC = "role_heuristics"  # 人狼セオリーカード（推論の目安・誘導が強い）
ROLE_DERIVED = "derived_unverified"  # 未検証の派生情報（仮説差分・要約・観測）
ROLE_TEMPORARY_GUIDANCE = "temporary_guidance"  # 期限付きの暫定案内
# TASK-062: エンジンが発行する「日をまたぐ論理状態」（確定事実・反証・未解決の問い）。
# いま有効な状態なので中部に沈ませたくないが、読み枠を先頭で決める誘導節でもない。
ROLE_CONTINUITY = "continuity_state"

SECTION_ROLES: dict[str, str] = {
    "system_rules": ROLE_RULES,
    "output_language": ROLE_LANGUAGE,
    "fact_ledger": ROLE_ENGINE_FACT,
    "claim_ledger": ROLE_PUBLIC_INDEX,
    "self_action_memory": ROLE_SELF_HISTORY,
    "own_speeches": ROLE_SELF_HISTORY,
    "reasoning_memo_own": ROLE_SELF_HISTORY,
    "actor_private_memo": ROLE_ENGINE_PRIVATE,
    "private_seer_result": ROLE_ENGINE_PRIVATE,
    "info_boundaries": ROLE_ENGINE_PRIVATE,
    "day_timeline": ROLE_TEMPORAL,
    "conversation_evidence": ROLE_EVIDENCE,
    "current_task": ROLE_TASK,
    "question_block": ROLE_TASK,
    "direct_answer": ROLE_TASK,
    "leak_boundary": ROLE_SECRECY,
    "behavior_contract": ROLE_SECRECY,
    "instructions": ROLE_FORMAT,
    "output_checklist": ROLE_FORMAT,
    "discussion_guidelines": ROLE_ADVISORY,
    "reasoning_guidance": ROLE_ADVISORY,
    "werewolf_theory_card": ROLE_HEURISTIC,
    "hypothesis_ledger": ROLE_DERIVED,
    "inference_line": ROLE_DERIVED,
    "previous_topics": ROLE_DERIVED,
    "continuity_memo": ROLE_DERIVED,
    # TASK-062: 正規化された論理状態（エンジン発行）。自分の記録と同じくらい「いま」使う
    "continuity_checkpoint": ROLE_CONTINUITY,
    "integrity_alerts": ROLE_DERIVED,
    "round_transcript": ROLE_DERIVED,
    "day1_first_turn_guidance": ROLE_TEMPORARY_GUIDANCE,
    "first_night_guidance": ROLE_TEMPORARY_GUIDANCE,
}

# 役割別の配置利得。中部に置かれても困らない節（derived / advisory / heuristics）は低い。
ROLE_WEIGHTS: dict[str, float] = {
    ROLE_RULES: 1.15,
    ROLE_LANGUAGE: 1.05,
    ROLE_ENGINE_FACT: 1.05,
    ROLE_PUBLIC_INDEX: 1.00,
    ROLE_SELF_HISTORY: 1.30,
    ROLE_ENGINE_PRIVATE: 1.10,
    ROLE_TEMPORAL: 1.20,
    ROLE_EVIDENCE: 1.25,
    ROLE_TASK: 1.40,
    ROLE_FORMAT: 1.35,
    ROLE_SECRECY: 1.30,
    ROLE_ADVISORY: 0.70,
    ROLE_HEURISTIC: 0.60,
    ROLE_DERIVED: 0.55,
    ROLE_TEMPORARY_GUIDANCE: 0.55,
    # TASK-062: 状態そのものなので、議論の案内や派生要約より高く置く（中部に沈ませない）
    ROLE_CONTINUITY: 1.20,
}

# 「いま使う」役割の節が中部帯に沈むことを減点する役割集合
MUST_USE_ROLES: frozenset[str] = frozenset(
    {
        ROLE_TASK,
        ROLE_EVIDENCE,
        ROLE_FORMAT,
        ROLE_SECRECY,
        ROLE_SELF_HISTORY,
        ROLE_TEMPORAL,
        # TASK-062: 日をまたぐ論理状態も「いま使う」情報（中部に沈めると記憶減衰が出る）
        ROLE_CONTINUITY,
    }
)

# 先頭帯を占めると「読み枠を先に決める」効果が出る役割集合（中部へ追いやりたい側）。
NON_ANCHOR_ROLES: frozenset[str] = frozenset(
    {
        ROLE_ADVISORY,
        ROLE_HEURISTIC,
        ROLE_DERIVED,
        ROLE_TEMPORARY_GUIDANCE,
    }
)



def role_of(name: str) -> str:
    """節名の役割（未登録の名前は derived_unverified として保守的に扱う）。"""
    return SECTION_ROLES.get(str(name), ROLE_DERIVED)


# ---------------------------------------------------------------------------
# 隣接させたいペア（距離がひらくほど減点する）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdjacencyTarget:
    """隣接配置を測る2節とペナルティ係数（距離/プロンプト長 × weight × 基本係数）。"""

    left: str
    right: str
    weight: float
    reason: str


ADJACENCY_TARGETS: tuple[AdjacencyTarget, ...] = (
    AdjacencyTarget(
        "conversation_evidence",
        "question_block",
        1.30,
        "現在の質問は証拠を読んだ直後に出す（引用の所在を辿らせる）",
    ),
    AdjacencyTarget(
        "own_speeches",
        "reasoning_memo_own",
        1.20,
        "自分の発言と自分の推理メモを離すと立場の更新が途切れる",
    ),
    AdjacencyTarget(
        "question_block",
        "own_speeches",
        1.10,
        "質問への回答は自分の過去の発言と接続させる（TASK-035 / TASK-039系）",
    ),
    AdjacencyTarget(
        "day_timeline",
        "conversation_evidence",
        1.25,
        "時系列欄と発言証拠を離すと、TASK-059で実測された誤帰属が出る",
    ),
    AdjacencyTarget(
        "claim_ledger",
        "conversation_evidence",
        0.85,
        "主張台帳のID索引のすぐ下に本文があると追跡しやすい",
    ),
    AdjacencyTarget(
        "behavior_contract",
        "instructions",
        1.15,
        "強い契約と出力形式をまとめて終端に置く",
    ),
    AdjacencyTarget(
        "own_speeches",
        "reasoning_guidance",
        0.75,
        "推理メモの指針は、自分の前の推理メモの隣で読ませる（更新しやすくする）",
    ),
    # TASK-062: 正規化された論理状態は「発言証拠のすぐそば」「現在質問のすぐそば」に置く。
    # 離れると、確定事実・反証済みの仮説・未解決の問いを現在の議論に接続できない。
    AdjacencyTarget(
        "continuity_checkpoint",
        "conversation_evidence",
        1.05,
        "状態→発言本文の順で読むと、IDと実際の発言の対応が辿れる（記憶減衰対策）",
    ),
    AdjacencyTarget(
        "continuity_checkpoint",
        "question_block",
        0.95,
        "現在質問に答える直前に論理状態があると、自分の立場を更新したまま答えられる",
    ),
)



# ---------------------------------------------------------------------------
# ハード制約（スコアに関係なく必ず守る配置）
# ---------------------------------------------------------------------------

# 終端に固める4節（この相対順を保ったままプロンプト末尾に置く）
TAIL_TIGHT_SECTIONS: tuple[str, ...] = (
    "leak_boundary",
    "behavior_contract",
    "instructions",
    "output_checklist",
)
# 本人限定のエンジン情報は終端帯へ下げない（TASK-058で「判定語は前列＋終端契約で抑える」
# 方針を踏襲した判断。秘匿ガードの再要求を増やさないための安全側の線引き）
ENGINE_PRIVATE_SECTIONS: tuple[str, ...] = (
    "private_seer_result",
    "actor_private_memo",
)
# この2つは必ず隣接させる（自分の発言と自分の推理メモ）
ADJACENT_PAIR_REQUIRED: tuple[tuple[str, str], ...] = (
    ("own_speeches", "reasoning_memo_own"),
)
# 誘導の強い節は事実台帳より前に出さない（＝先頭帯で推理の枠を決めない）
NO_PRIMACY_SECTIONS: tuple[str, ...] = (
    "werewolf_theory_card",
    "continuity_memo",
    # TASK-062: 正規化状態も「読み枠を先に決める」位置には置かない（末尾帯で使う）
    "continuity_checkpoint",
)
# TASK-062: 論理状態は主張索引（CLAIM_LEDGER）のIDを参照するので、索引より前に来ない
CONTINUITY_STATE_SECTION = "continuity_checkpoint"


def validate_declared_order(order: Sequence[str]) -> list[dict[str, object]]:
    """宣言済みの節順がハード制約を満たすかを静的検査する（字数不要）。"""
    names = [str(name) for name in order]
    problems: list[dict[str, object]] = []
    if not names or names[0] != "system_rules":
        problems.append({"kind": "rules_not_first", "name": names[0] if names else ""})
    if "output_language" in names and names.index("output_language") > 2:
        problems.append({"kind": "language_too_late", "name": "output_language"})
    present_tail = [name for name in TAIL_TIGHT_SECTIONS if name in names]
    if len(present_tail) >= 2:
        tail_start = names.index(present_tail[0])
        for name in present_tail[1:]:
            if names.index(name) <= tail_start:
                problems.append({"kind": "tail_band_broken", "name": name})
        if names[-1] != present_tail[-1]:
            problems.append({"kind": "checklist_not_last", "name": present_tail[-1]})
    midpoint = max(1, len(names) // 2)
    for name in ENGINE_PRIVATE_SECTIONS:
        if name in names and names.index(name) >= midpoint:
            problems.append({"kind": "private_in_recency", "name": name})
    for left, right in ADJACENT_PAIR_REQUIRED:
        if left in names and right in names:
            if abs(names.index(left) - names.index(right)) != 1:
                problems.append({"kind": "anchor_split", "name": f"{left}|{right}"})
    for name in NO_PRIMACY_SECTIONS:
        if name in names and "fact_ledger" in names:
            if names.index(name) < names.index("fact_ledger"):
                problems.append({"kind": "heuristics_before_facts", "name": name})
    # 盤面と自分（生存者・死亡・当日・自分の名前）の基礎は早く読ませる
    if "fact_ledger" in names and names.index("fact_ledger") > 4:
        problems.append({"kind": "facts_too_deep", "name": "fact_ledger"})
    # 議論の案内は「証拠を読んだあとに促す」。前に置いて読み枠を先に決めない
    if "discussion_guidelines" in names and "conversation_evidence" in names:
        if names.index("discussion_guidelines") < names.index("conversation_evidence"):
            problems.append({"kind": "guidelines_before_evidence", "name": "discussion_guidelines"})
    if "discussion_guidelines" in names and "instructions" in names:
        if names.index("discussion_guidelines") > names.index("instructions"):
            problems.append({"kind": "guidelines_after_format", "name": "discussion_guidelines"})
    # TASK-062: 論理状態は主張索引（CLAIM_LEDGER）のIDを参照する。索引より前に出すと、
    # 存在しないIDを辿らせることになり、追跡性が壊れる。
    if CONTINUITY_STATE_SECTION in names and "claim_ledger" in names:
        if names.index(CONTINUITY_STATE_SECTION) < names.index("claim_ledger"):
            problems.append(
                {
                    "kind": "state_before_index",
                    "name": CONTINUITY_STATE_SECTION,
                    "after": "claim_ledger",
                }
            )
    return problems


# ---------------------------------------------------------------------------
# コストモデル本体
# ---------------------------------------------------------------------------

# 隣接距離ペナルティの基礎係数（距離がプロンプト長の100%で weight×この値ぶん減点）
ADJACENCY_PENALTY_PER_FRACTION = 90.0
# 出力形式仕様の後ろに非形式節が載る場合の減点率（字数割合 × この値）
POST_FORMAT_DILUTION_WEIGHT = 140.0
# 「いま使う」節が中部帯に沈む場合の減点率（字数割合 × この値）
WEAK_MIDDLE_MUST_USE_WEIGHT = 120.0
# 誘導系の節が先頭帯を占める場合の減点率（字数割合 × この値＝読み枠を先に決める効果）
PRIMACY_CLUTTER_WEIGHT = 90.0
# スコアの目盛り（attention利得の平均 × これ）
SCORE_SCALE = 100.0
# ハード制約違反1件あたりの減点（候補から確実に落ちるように大きく取る）
HARD_VIOLATION_PENALTY = 35.0


def _center_fraction(start: int, end: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return (start + max(0, end - start) / 2.0) / float(total)


# 長い節は内側で見え方が変わる（先頭はよく見え、中部は沈む）。スパン内で prior を
# サンプル平均して、証拠のように長い節の評価を fair にする。
ATTENTION_SPAN_SAMPLES = 12


def attention_prior_span(start: int, end: int, total: int) -> float:
    """スパン（開始, 終了, 全体字数）から、節全体の実効的な見えやすさを求める。"""
    if total <= 0:
        return ATTENTION_MIDDLE_BASE
    lo = min(1.0, max(0.0, start / float(total)))
    hi = min(1.0, max(lo, end / float(total)))
    if hi - lo < 1e-9:
        return attention_prior(lo)
    steps = max(2, int(ATTENTION_SPAN_SAMPLES))
    sampled = [attention_prior(lo + (hi - lo) * i / float(steps - 1)) for i in range(steps)]
    return round(sum(sampled) / float(len(sampled)), 4)


def middle_band_bounds(total: int) -> tuple[float, float]:
    """中部（注意が最も弱い）帯の文字数上の範囲 (開始, 終了)。"""
    size = max(1, int(total))
    return ATTENTION_PRIMACY_EDGE * size, (1.0 - ATTENTION_RECENCY_EDGE) * size


def _middle_overlap_chars(start: int, end: int, total: int) -> int:
    """スパンが中部帯と重なっている字数（長い節の中部沈みを正しく測るため）。"""
    lo, hi = middle_band_bounds(total)
    return max(0, int(min(end, hi) - max(start, lo)))




def _span_distance(spans: Mapping[str, tuple[int, int]], left: str, right: str) -> int:
    """2スパンの隙間（文字数）。重なり・逆順は 0 として扱う。"""
    left_span = spans.get(left)
    right_span = spans.get(right)
    if left_span is None or right_span is None:
        return 0
    first, second = sorted((left_span, right_span), key=lambda span: int(span[0]))
    return max(0, int(second[0]) - int(first[1]))


@dataclass
class OrderScore:
    """1つの節順がコストモデルでどれだけ良いかを示す構造的スコア（内容評価ではない）。"""

    order: tuple[str, ...]
    score: float
    attention_efficiency: float = 0.0
    adjacency_penalty: float = 0.0
    dilution_penalty: float = 0.0
    weak_middle_penalty: float = 0.0
    primacy_clutter_penalty: float = 0.0
    violations: tuple[dict[str, object], ...] = ()
    zone_chars: dict[str, int] = field(default_factory=dict)
    gaps: dict[str, float] = field(default_factory=dict)
    section_gain: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "layout_order": list(self.order),
            "layout_score": round(float(self.score), 2),
            "attention_efficiency": round(float(self.attention_efficiency), 4),
            "adjacency_penalty": round(float(self.adjacency_penalty), 2),
            "post_format_dilution_penalty": round(float(self.dilution_penalty), 2),
            "weak_middle_must_use_penalty": round(float(self.weak_middle_penalty), 2),
            "primacy_clutter_penalty": round(float(self.primacy_clutter_penalty), 2),
            "hard_violations": [dict(item) for item in self.violations],
            "simulated_zone_chars": dict(self.zone_chars),
            "simulated_adjacency_gap_fraction": dict(self.gaps),
        }


def spans_from_order(order: Sequence[str], section_chars: Mapping[str, int]) -> dict[str, tuple[int, int]]:
    """宣言順と節別字数から、その順序で並んだときのスパン（開始, 終了）を作る。"""
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name in order:
        key = str(name)
        size = max(0, int(section_chars.get(key, 0) or 0))
        if size <= 0:
            continue
        spans[key] = (cursor, cursor + size)
        cursor += size
    return spans


def score_order(
    order: Sequence[str],
    section_chars: Mapping[str, int],
) -> OrderScore:
    """節順と節別字数を受け取り、構造的スコアを返す（実プロンプト不要のシミュレーション）。

    各節は累積字数で中心位置を得て、`attention_prior × role_weight` の利得を稼ぐ。
    減点は (1) 隣接させたい節の距離、(2) 出力形式仕様の後ろに載る非形式節、
    (3) 「いま使う」節が中部帯へ沈んだ分量、(4) ハード制約違反。
    """
    names = [str(name) for name in order]
    sizes = {str(name): max(0, int(section_chars.get(str(name), 0) or 0)) for name in names}
    violations = validate_declared_order(names)
    spans = spans_from_order(names, sizes)
    total = sum(spans[name][1] - spans[name][0] for name in spans)
    if total <= 0:
        return OrderScore(order=tuple(names), score=0.0, violations=tuple(violations))

    gain = 0.0
    zone_chars: dict[str, int] = {zone: 0 for zone in ATTENTION_ZONES}
    weak_middle_must_use = 0
    primacy_clutter = 0
    section_gain: dict[str, float] = {}
    for name, (start, end) in spans.items():
        center = _center_fraction(start, end, total)
        role = role_of(name)
        weight = ROLE_WEIGHTS.get(role, 1.0)
        contribution = (end - start) * weight * attention_prior_span(start, end, total)
        gain += contribution
        section_gain[name] = round(float(contribution), 2)
        zone = attention_zone_for_center(center)
        zone_chars[zone] += end - start
        if role in MUST_USE_ROLES:
            weak_middle_must_use += _middle_overlap_chars(start, end, total)
        elif role in NON_ANCHOR_ROLES and zone == "primacy":
            primacy_clutter += end - start
    efficiency = gain / float(total)

    adjacency_penalty = 0.0
    gaps: dict[str, float] = {}
    for target in ADJACENCY_TARGETS:
        distance = _span_distance(spans, target.left, target.right)
        fraction = distance / float(total)
        gaps[f"{target.left}->{target.right}"] = round(fraction, 4)
        adjacency_penalty += fraction * ADJACENCY_PENALTY_PER_FRACTION * target.weight

    dilution_penalty = 0.0
    format_span = spans.get("instructions")
    if format_span is not None:
        after_chars = sum(
            end - start
            for name, (start, end) in spans.items()
            if start >= format_span[1] and role_of(name) != ROLE_FORMAT
        )
        dilution_penalty = (after_chars / float(total)) * POST_FORMAT_DILUTION_WEIGHT

    weak_middle_penalty = (weak_middle_must_use / float(total)) * WEAK_MIDDLE_MUST_USE_WEIGHT
    clutter_penalty = (primacy_clutter / float(total)) * PRIMACY_CLUTTER_WEIGHT

    score = (
        efficiency * SCORE_SCALE
        - adjacency_penalty
        - dilution_penalty
        - weak_middle_penalty
        - clutter_penalty
    )
    score -= HARD_VIOLATION_PENALTY * len(violations)
    return OrderScore(
        order=tuple(names),
        score=round(float(score), 2),
        attention_efficiency=round(float(efficiency), 4),
        adjacency_penalty=round(float(adjacency_penalty), 2),
        dilution_penalty=round(float(dilution_penalty), 2),
        weak_middle_penalty=round(float(weak_middle_penalty), 2),
        primacy_clutter_penalty=round(float(clutter_penalty), 2),
        violations=tuple(violations),
        zone_chars=dict(zone_chars),
        gaps=dict(gaps),
        section_gain=dict(section_gain),
    )


def score_built_prompt(text: str, order: Sequence[str]) -> OrderScore:
    """実プロンプトの実測スパンで、その節順とコストモデルを評価する（診断用）。"""
    spans = find_sections(text)
    chars = {name: max(0, end - start) for name, (start, end) in spans.items()}
    present_order = tuple(name for name in order if name in chars)
    result = score_order(present_order, chars)
    return result


# ---------------------------------------------------------------------------
# 実プロンプトの帯（zone）実測と配置品質の診断
# ---------------------------------------------------------------------------


def measure_zones(text: str) -> dict[str, int]:
    """実プロンプトの節を位置から3帯に分けて字数を測る（未登録部分は unregistered）。"""
    spans = find_sections(text)
    total = max(1, len(text))
    totals: dict[str, int] = {"primacy": 0, "weak_middle": 0, "recency": 0, "unregistered": 0}
    for name, (start, end) in spans.items():
        zone = attention_zone_for_center(_center_fraction(start, end, total))
        totals[zone] += max(0, end - start)
    accounted = sum(totals.values())
    totals["unregistered"] = max(0, len(text) - accounted)
    return totals


def adjacency_gap(text: str, left: str, right: str) -> float:
    """2節の実測距離をプロンプト長の割合で返す（どちらかが無ければ 0.0）。"""
    spans = find_sections(text)
    distance = _span_distance(spans, left, right)
    return round(distance / float(max(1, len(text))), 4)


def post_format_dilution(text: str) -> int:
    """`[Instructions]` の終端より後ろに載っている非形式節の字数（出力形式の希釈）。"""
    spans = find_sections(text)
    format_span = spans.get("instructions")
    if format_span is None:
        return 0
    chars = 0
    for name, (start, end) in spans.items():
        if start >= format_span[1] and role_of(name) != ROLE_FORMAT:
            chars += max(0, end - start)
    return int(chars)


def weak_middle_must_use_chars(text: str) -> int:
    """「いま使う」役割の節が中部帯に重なっている字数（スパンの実測ベース）。

    TASK-060で中心位置だけで判定する方式をやめた。発言証拠のように長くまたがる節は、
    中心が中部というだけで全長を罰すると配置改善が見えなくなるため、中部帯と重なった
    部分だけを数える（コストモデル `score_order` の減点と同じ基準に揃える）。
    """
    spans = find_sections(text)
    total = max(1, len(text))
    chars = 0
    for name, (start, end) in spans.items():
        if role_of(name) in MUST_USE_ROLES:
            chars += _middle_overlap_chars(start, end, total)
    return int(chars)


# ---------------------------------------------------------------------------
# 節順の探索（ブロック単位。内容は変えずに配置だけ比べる）
# ---------------------------------------------------------------------------

# 先頭2つは固定（ルールと応答言語）。どの設定でも動かさない。
PROMPT_SEARCH_HEAD: tuple[str, ...] = ("system_rules", "output_language")
# 終端4つは固定（秘匿境界→契約→出力形式→最終チェックリスト）
PROMPT_SEARCH_TAIL: tuple[str, ...] = TAIL_TIGHT_SECTIONS

# 動き回るブロック。隣へ置くべき節は同じブロックにして相対順序を保つ。
PROMPT_SEARCH_BLOCKS: tuple[tuple[str, ...], ...] = (
    ("fact_ledger",),
    ("werewolf_theory_card",),
    ("private_seer_result", "actor_private_memo", "info_boundaries"),
    ("self_action_memory",),
    ("claim_ledger",),
    ("day_timeline",),
    # TASK-062: エンジン発行の論理状態（チェックポイント）。主張索引のIDを参照する位置に置く
    ("continuity_checkpoint",),
    ("inference_line", "hypothesis_ledger", "previous_topics"),
    ("continuity_memo", "integrity_alerts"),
    ("day1_first_turn_guidance", "first_night_guidance"),
    ("round_transcript",),
    ("reasoning_guidance",),
    ("conversation_evidence",),
    ("own_speeches", "reasoning_memo_own"),
    ("current_task", "question_block", "direct_answer"),
    ("discussion_guidelines",),
)


def flatten_blocks(blocks: Sequence[Sequence[str]]) -> tuple[str, ...]:
    """ブロック列を1つの節順へ平坦化する（重複を除き、出現順を保つ）。"""
    flat: list[str] = []
    for block in blocks:
        for name in block:
            if str(name) not in flat:
                flat.append(str(name))
    return tuple(flat)


def _legal(blocks: Sequence[Sequence[str]]) -> bool:
    """ブロック列がハード制約を満たすか（探索の枝刈り）。"""
    return not validate_declared_order(flatten_blocks(blocks))


# ブロックの略称（候補順を短く読めるようにする）
BLOCK_ALIASES: dict[str, tuple[str, ...]] = {
    "facts": ("fact_ledger",),
    "theory": ("werewolf_theory_card",),
    "private": ("private_seer_result", "actor_private_memo", "info_boundaries"),
    "action": ("self_action_memory",),
    "claims": ("claim_ledger",),
    "timeline": ("day_timeline",),
    "state": ("continuity_checkpoint",),
    "derived": ("inference_line", "hypothesis_ledger", "previous_topics"),
    "memo": ("continuity_memo", "integrity_alerts"),
    "temp": ("day1_first_turn_guidance", "first_night_guidance"),
    "round": ("round_transcript",),
    "rguid": ("reasoning_guidance",),
    "evid": ("conversation_evidence",),
    "selfpair": ("own_speeches", "reasoning_memo_own"),
    "task": ("current_task", "question_block", "direct_answer"),
    "guid": ("discussion_guidelines",),
}


def blocks_from_keys(keys: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """略称の列をブロック列へ展開する（未知の略称は ValueError）。"""
    blocks: list[tuple[str, ...]] = []
    for key in keys:
        block = BLOCK_ALIASES.get(str(key))
        if block is None:
            raise ValueError(f"unknown layout block alias: {key!r}")
        blocks.append(block)
    return tuple(blocks)


# 探索の初期候補（movable の並びのみ。先頭と終端は自動で付く）。
# TASK-062で `state`（エンジン発行の論理状態）が加わった。配置候補を4パターン含める。
SEED_KEYS: tuple[tuple[str, ...], ...] = (
    # seed A: TASK-058 の v2 配置を、今のブロック制約へ写した近似（state は証拠の直前）
    ("facts", "theory", "claims", "action", "selfpair", "private", "timeline", "state",
     "derived", "memo", "temp", "round", "rguid", "evid", "task", "guid"),
    # seed B: 自分の記録を先に置き、中部の後に証拠→自分→質問（state は索引の直後）
    ("facts", "private", "action", "theory", "derived", "memo", "temp", "round",
     "rguid", "claims", "state", "timeline", "evid", "selfpair", "task", "guid"),
    # seed C: 議論案内を証拠の直前へ、質問だけは最後まで後ろ（state は質問の直後）
    ("facts", "private", "action", "theory", "derived", "memo", "temp", "round",
     "claims", "guid", "rguid", "timeline", "evid", "state", "selfpair", "task"),
    # seed D: 議論案内を自分の連続性の前（証拠より前）に置く（state は時点の隣）
    ("facts", "private", "theory", "action", "claims", "state", "derived", "memo", "temp",
     "round", "rguid", "selfpair", "timeline", "evid", "guid", "task"),
    # seed E (TASK-062): v3 の終端連鎖（索引→時点→証拠→質問）に状態を挟む候補から始める
    ("action", "facts", "private", "theory", "derived", "memo", "temp", "round",
     "claims", "timeline", "state", "evid", "task", "selfpair", "rguid", "guid"),
    # seed F (TASK-062): 状態を質問の直後に置き、自分の記録へ橋渡しする候補
    ("action", "facts", "private", "theory", "derived", "memo", "temp", "round",
     "claims", "timeline", "evid", "task", "state", "selfpair", "rguid", "guid"),
)


def candidate_orders() -> list[tuple[str, ...]]:
    """候補順の初期集合（先頭2つ＋movable＋終端4つの平坦列）。"""
    orders: list[tuple[str, ...]] = []
    for keys in SEED_KEYS:
        blocks = (PROMPT_SEARCH_HEAD, *blocks_from_keys(keys), PROMPT_SEARCH_TAIL)
        orders.append(flatten_blocks(blocks))
    return orders


def _improved(
    blocks: Sequence[Sequence[str]],
    section_chars: Mapping[str, int],
    *,
    max_passes: int = 6,
) -> tuple[tuple[tuple[str, ...], ...], OrderScore]:
    """1つの候補を、単一ブロックの移動と2ブロックの入替で貪欲に改善する（決定的）。"""
    current = [tuple(block) for block in blocks]
    best = score_order(flatten_blocks(current), section_chars)
    if len(current) < 3:
        return tuple(current), best
    head, tail = current[0], current[-1]
    movable = current[1:-1]
    for _ in range(max(1, int(max_passes))):
        improved = False
        # (a) 1つのブロックを別の場所へ移す
        for source in range(len(movable)):
            for target in range(len(movable)):
                if source == target:
                    continue
                trial = list(movable)
                moved = trial.pop(source)
                trial.insert(target, moved)
                candidate = [head, *trial, tail]
                if not _legal(candidate):
                    continue
                scored = score_order(flatten_blocks(candidate), section_chars)
                if scored.score > best.score + 1e-9:
                    best = scored
                    movable = trial
                    improved = True
        # (b) 2つのブロックを入替
        for left in range(len(movable)):
            for right in range(left + 1, len(movable)):
                trial = list(movable)
                trial[left], trial[right] = trial[right], trial[left]
                candidate = [head, *trial, tail]
                if not _legal(candidate):
                    continue
                scored = score_order(flatten_blocks(candidate), section_chars)
                if scored.score > best.score + 1e-9:
                    best = scored
                    movable = trial
                    improved = True
        if not improved:
            break
    return tuple([head, *movable, tail]), best


def search_prompt_orders(
    section_chars: Mapping[str, int],
    *,
    top_k: int = 6,
    max_passes: int = 6,
) -> list[dict[str, object]]:
    """代表字数で候補順を貪欲改善し、上位をスコア付きで返す（決定的・LL不使用）。"""
    found: dict[tuple[tuple[str, ...], ...], OrderScore] = {}
    for keys in SEED_KEYS:
        blocks = (PROMPT_SEARCH_HEAD, *blocks_from_keys(keys), PROMPT_SEARCH_TAIL)
        refined, scored = _improved(blocks, section_chars, max_passes=max_passes)
        previous = found.get(refined)
        if previous is None or scored.score > previous.score:
            found[refined] = scored
    ranked = sorted(found.values(), key=lambda item: (-item.score, tuple(item.order)))
    return [result.as_dict() for result in ranked[: max(1, int(top_k))]]


# ---------------------------------------------------------------------------
# 採用する節順（layout v3）
# ---------------------------------------------------------------------------

# `_analysis/task060_layout_study.py` が代表プロンプトの実測字数で探索した結果を凍結したもの。
# 読み方: ルールと応答言語 → 自分の行動記録 → 盤面の確定事実 → 本人限定情報
#        → （中部: 誘導の強い理論カード・未検証の派生情報・期限付き案内・ラウンド見出し）
#        → 公開主張の索引 → 時点 → 発言証拠 → 現在質問 → 自分の発言と推理メモ
#        → 推理メモの指針 → 議論の案内（中立）→ 秘匿境界 → 強い契約 → 出力形式 → 最終チェックリスト
PROMPT_LAYOUT_V3_ORDER: tuple[str, ...] = flatten_blocks(
    (
        PROMPT_SEARCH_HEAD,
        ("self_action_memory",),
        ("fact_ledger",),
        ("private_seer_result", "actor_private_memo", "info_boundaries"),
        ("werewolf_theory_card",),
        ("inference_line", "hypothesis_ledger", "previous_topics"),
        ("continuity_memo", "integrity_alerts"),
        ("day1_first_turn_guidance", "first_night_guidance"),
        ("round_transcript",),
        ("claim_ledger",),
        ("day_timeline",),
        ("conversation_evidence",),
        ("current_task", "question_block", "direct_answer"),
        ("own_speeches", "reasoning_memo_own"),
        ("reasoning_guidance",),
        ("discussion_guidelines",),
        PROMPT_SEARCH_TAIL,
    )
)

# TASK-062: layout v4（既定）。`_analysis/task062_order_probe.py` が代表字数で探索した結果を
# 凍結したもの。v3 と同じ連鎖を保ったまま、エンジン発行の論理状態を
# 「発言証拠の直後・現在質問の直前」へ置く（スコア46.93／違反0、v3末尾_append案は38.98）。
# 読み方: ルールと応答言語 → 自分の行動記録 → 盤面の確定事実 → 本人限定情報
#        → （中部: 理論カード・未検証の派生情報・期限付き案内・ラウンド見出し）
#        → 公開主張の索引 → 時点 → 発言証拠 → [正規化された論理状態] → 現在質問
#        → 自分の発言と推理メモ → 推理メモの指針 → 議論の案内（中立）→ 秘匿境界
#        → 強い契約 → 出力形式 → 最終チェックリスト
PROMPT_LAYOUT_V4_ORDER: tuple[str, ...] = flatten_blocks(
    (
        PROMPT_SEARCH_HEAD,
        ("self_action_memory",),
        ("fact_ledger",),
        ("private_seer_result", "actor_private_memo", "info_boundaries"),
        ("werewolf_theory_card",),
        ("inference_line", "hypothesis_ledger", "previous_topics"),
        ("continuity_memo", "integrity_alerts"),
        ("day1_first_turn_guidance", "first_night_guidance"),
        ("round_transcript",),
        ("claim_ledger",),
        ("day_timeline",),
        ("conversation_evidence",),
        ("continuity_checkpoint",),
        ("current_task", "question_block", "direct_answer"),
        ("own_speeches", "reasoning_memo_own"),
        ("reasoning_guidance",),
        ("discussion_guidelines",),
        PROMPT_SEARCH_TAIL,
    )
)

# v3 から v4 へ変わった点（節の新設と位置だけ。文言と発言の自由は変えない）
V4_CHANGED_FROM_V3: tuple[str, ...] = (
    "continuity_checkpoint: 発言証拠の直後・現在質問の直前へ（証拠との距離0、質問との距離0.0034）",
    "自分の一貫性アンカー（自分の発言→推理メモ）は質問の直後に残し、状態節がその前段になった",
    "誘導系（理論カード・派生情報・期限付き案内）は中部帯のまま、状態節だけは終端帯で読む",
)

# v2 の宣言順から v3 へ変わった点（ログで説明できるようにしておく）
V3_CHANGED_FROM_V2: tuple[str, ...] = (
    "werewolf_theory_card: FACT_LEDGER の内側（先頭帯）から中部帯へ降格し、optional 化",
    "claim_ledger / day_timeline: 発言証拠の直前へ寄せてID追跡の距離を縮めた",
    "discussion_guidelines: 出力形式仕様の後ろから、証拠と質問の後（契約の直前）へ移した",
    "own_speeches + reasoning_memo_own: 現在質問の直後・契約の直前へ群らした（一貫性アンカーの隣接）",
    "reasoning_guidance: 自分の推理メモの隣へ寄せ、更新として書ける位置にした",
    "actor_private_memo / private_seer_result: 前半帯に固定（秘匿ガードの再要求を増やさない）",
)

# 探索を再現するための代表字数（`_analysis/task060_layout_study.py` の実測平均を丸めた値）。
# 実際の節字数は進行・モデル・予算で変わるため、ここでの値は探索を再現するためのサンプルにすぎない。
REPRESENTATIVE_SECTION_CHARS: dict[str, int] = {
    "system_rules": 940,
    "output_language": 210,
    "fact_ledger": 1580,
    "werewolf_theory_card": 2680,
    "private_seer_result": 640,
    "actor_private_memo": 520,
    "info_boundaries": 380,
    "self_action_memory": 760,
    "inference_line": 260,
    "hypothesis_ledger": 540,
    "previous_topics": 320,
    "continuity_memo": 620,
    "integrity_alerts": 180,
    # TASK-062: `[CONTINUITY_CHECKPOINT]` の代表字数。見出し+権威説明が約620字、
    # 差分・反証・未解決の問いの行で最大約900字（Day 2以降の差分提示の実測平均を想定）。
    "continuity_checkpoint": 900,
    "day1_first_turn_guidance": 1450,
    "round_transcript": 120,
    "reasoning_guidance": 620,
    "claim_ledger": 900,
    "day_timeline": 460,
    "conversation_evidence": 6200,
    "own_speeches": 260,
    "reasoning_memo_own": 420,
    "current_task": 90,
    "question_block": 340,
    "direct_answer": 260,
    "discussion_guidelines": 1750,
    "leak_boundary": 520,
    "behavior_contract": 1180,
    "instructions": 780,
    "output_checklist": 260,
}


def adopted_layout_score(
    section_chars: Mapping[str, int] | None = None,
    order: Sequence[str] | None = None,
) -> OrderScore:
    """採用中の節順（TASK-062以降は v4）をコストモデルで評価する（既定は代表字数）。"""
    chars = REPRESENTATIVE_SECTION_CHARS if section_chars is None else section_chars
    chosen = PROMPT_LAYOUT_V4_ORDER if order is None else tuple(order)
    return score_order(chosen, chars)


def v3_layout_score(
    section_chars: Mapping[str, int] | None = None,
) -> OrderScore:
    """A/B比較の基準側（TASK-060の v3 節順）を評価する。"""
    chars = REPRESENTATIVE_SECTION_CHARS if section_chars is None else section_chars
    return score_order(PROMPT_LAYOUT_V3_ORDER, chars)


__all__ = [
    "ADJACENCY_TARGETS",
    "ADJACENCY_PENALTY_PER_FRACTION",
    "ADJACENT_PAIR_REQUIRED",
    "ATTENTION_MIDDLE_BASE",
    "ATTENTION_PRIMACY_EDGE",
    "ATTENTION_PRIMACY_GAIN",
    "ATTENTION_RECENCY_EDGE",
    "ATTENTION_RECENCY_GAIN",
    "ATTENTION_SPAN_SAMPLES",
    "ATTENTION_ZONES",
    "BLOCK_ALIASES",
    "CONTINUITY_STATE_SECTION",
    "ENGINE_PRIVATE_SECTIONS",
    "HARD_VIOLATION_PENALTY",
    "MUST_USE_ROLES",
    "NON_ANCHOR_ROLES",
    "NO_PRIMACY_SECTIONS",
    "OrderScore",
    "POST_FORMAT_DILUTION_WEIGHT",
    "PRIMACY_CLUTTER_WEIGHT",
    "PROMPT_LAYOUT_LEGACY",
    "PROMPT_LAYOUT_POLICIES",
    "PROMPT_LAYOUT_POLICY_AUTO",
    "PROMPT_LAYOUT_POLICY_DEFAULT",
    "PROMPT_LAYOUT_POLICY_V2",
    "PROMPT_LAYOUT_POLICY_V3",
    "PROMPT_LAYOUT_POLICY_V4",
    "PROMPT_LAYOUT_V2",
    "PROMPT_LAYOUT_V2_ORDER",
    "PROMPT_LAYOUT_V3",
    "PROMPT_LAYOUT_V3_ORDER",
    "PROMPT_LAYOUT_V4",
    "PROMPT_LAYOUT_V4_ORDER",
    "PROMPT_SEARCH_BLOCKS",
    "PROMPT_SEARCH_HEAD",
    "PROMPT_SEARCH_TAIL",
    "REPRESENTATIVE_SECTION_CHARS",
    "ROLE_ADVISORY",
    "ROLE_DERIVED",
    "ROLE_EVIDENCE",
    "ROLE_ENGINE_FACT",
    "ROLE_ENGINE_PRIVATE",
    "ROLE_FORMAT",
    "ROLE_HEURISTIC",
    "ROLE_CONTINUITY",
    "ROLE_LANGUAGE",
    "ROLE_PUBLIC_INDEX",
    "ROLE_RULES",
    "ROLE_SECRECY",
    "ROLE_SELF_HISTORY",
    "ROLE_TASK",
    "ROLE_TEMPORAL",
    "ROLE_TEMPORARY_GUIDANCE",
    "ROLE_WEIGHTS",
    "SCORE_SCALE",
    "SECTION_ROLES",
    "SEED_KEYS",
    "TAIL_TIGHT_SECTIONS",
    "V3_CHANGED_FROM_V2",
    "V4_CHANGED_FROM_V3",
    "WEAK_MIDDLE_MUST_USE_WEIGHT",
    "adjacency_gap",
    "adopted_layout_score",
    "v3_layout_score",
    "attention_prior",
    "attention_prior_span",
    "attention_zone_for_center",
    "blocks_from_keys",
    "candidate_orders",
    "flatten_blocks",
    "measure_zones",
    "middle_band_bounds",
    "normalize_prompt_layout_policy",
    "post_format_dilution",
    "prompt_layout_policy_label",
    "resolve_prompt_layout",
    "role_of",
    "score_built_prompt",
    "score_order",
    "search_prompt_orders",
    "spans_from_order",
    "validate_declared_order",
    "weak_middle_must_use_chars",
]









