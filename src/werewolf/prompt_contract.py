"""TASK-057: プロンプト契約の簡素化と情報ブロック間の重複削減。

目的は「プロンプトを短くする」ことだけではない。同じ事実・制約・発言が別の表現で
複数のブロックに現れると、モデルは優先順位を誤ったり、古い要約を現在の命令として
扱ったりする。ここでは情報の **役割（purpose）・出所（source）・権威（authority）・
期限（expires_at）** を台帳化し、同じ証拠は本文1箇所・他はID参照に置き換える。

設計思想（このモジュールが守る4つの線引き）:
1. コンテキスト節約: 本文の再掲をやめ、参照と短い見出しだけに置き換える。
2. モデルの一貫性: core（確定事実・自分の行動記録・自分の推理メモ）は絶対に削らない。
   削る順は「派生した要約 → 仮説差分 → 期限切れの暫定案内」で、常に core を残す。
3. 合理的な議論: 議論を促す案内（guidelines）は残し、重複した禁止命令だけを1箇所に寄せる。
4. 自由の担保: 口調・文量・構成・論点・結論を指定する命令は増やさない。契約節には
   「強い契約だけを書く」と明記し、秘匿と形式といった最小限だけ強くする。

実機LLMの挙動評価とは切り離して、ログ上の数値（セクション別字数・重複制約数・
除去段階・推定トークン）を比較できるようにする。発言本文の書き換えは一切行わない。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# ポリシー（既定は簡素化。legacy はA/B比較用の旧挙動）
# ---------------------------------------------------------------------------

PROMPT_CONTRACT_POLICY_LEGACY = "legacy"
PROMPT_CONTRACT_POLICY_SIMPLIFIED = "simplified"
PROMPT_CONTRACT_POLICIES: tuple[str, ...] = (
    PROMPT_CONTRACT_POLICY_SIMPLIFIED,
    PROMPT_CONTRACT_POLICY_LEGACY,
)
PROMPT_CONTRACT_POLICY_DEFAULT = PROMPT_CONTRACT_POLICY_SIMPLIFIED


def normalize_prompt_contract_policy(value: object) -> str:
    """設定値を \"simplified\" / \"legacy\" へ正規化する（不明値は既定の簡素化）。"""
    if isinstance(value, bool):
        return (
            PROMPT_CONTRACT_POLICY_SIMPLIFIED
            if value
            else PROMPT_CONTRACT_POLICY_LEGACY
        )
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in PROMPT_CONTRACT_POLICIES:
            return candidate
        if candidate in {"on", "true", "simple", "contract"}:
            return PROMPT_CONTRACT_POLICY_SIMPLIFIED
        if candidate in {"off", "old", "false", "before"}:
            return PROMPT_CONTRACT_POLICY_LEGACY
    return PROMPT_CONTRACT_POLICY_DEFAULT


def prompt_contract_policy_label(policy: object) -> str:
    """ログ・CLI表示用のラベル（正規化済みの文字列を返す）。"""
    return normalize_prompt_contract_policy(policy)


# ---------------------------------------------------------------------------
# セクション台帳（purpose / source / authority / expires_at）
# ---------------------------------------------------------------------------

TIER_CORE = "core"
TIER_OPTIONAL = "optional"

# 権威の種類。`authority` を曖昧にしないための列挙（TASK-057実装案1）。
AUTHORITY_RULES = "game_rules"  # エンジンが定めるルール（変更不可）
AUTHORITY_ENGINE = "engine_issued"  # エンジンが発行した事実・非公開情報
AUTHORITY_PUBLIC_RECORD = "public_record"  # 公開発言・投票などの記録
AUTHORITY_UNVERIFIED = "unverified_derivation"  # モデル発言から導いた未検証の派生情報
AUTHORITY_SELF = "actor_self_output"  # モデル自身が前回書いたもの
AUTHORITY_TEMPORARY = "temporary_guidance"  # 期限付きの案内（事実ではない）
AUTHORITY_FORMAT = "output_format"  # 出力形式の契約


@dataclass(frozen=True)
class PromptSectionMeta:
    """プロンプト1節の情報的役割。本文の優先順位を決めるためのメタデータ。

    - `markers`: その節の先頭に現れるマーカー（言語違いの実ヘッダを並べる）
    - `tier`: core は絶対に削らない、optional は段階的に削ってよい
    - `purpose`: その節が何のために存在するか（ログで解釈できるようにする）
    - `source`: 情報の出所（エンジン発行情報か、モデル発言からの導出か）
    - `authority`: 権威（確定事実か未検証か、形式契約か暫定案内か）
    - `expires_at`: 期限。`D<day>T<turn>` / `N<day>` 形。None は期限なし（恒久の役割）
    """

    name: str
    markers: tuple[str, ...]
    tier: str
    purpose: str
    source: str
    authority: str
    expires_at: str | None = None


# プロンプト組み立て順に沿った台帳。入れ子見出し（[INTEGRITY_ALERTS] など）も別名にし、
# 除去範囲が隣の core 節を巻き込まないようにする。
PROMPT_SECTIONS: tuple[PromptSectionMeta, ...] = (
    PromptSectionMeta(
        name="system_rules",
        markers=("[SYSTEM_RULES]", "[Rules]"),
        tier=TIER_CORE,
        purpose="win conditions and game rules",
        source=AUTHORITY_RULES,
        authority=AUTHORITY_RULES,
    ),
    PromptSectionMeta(
        name="output_language",
        markers=("[OUTPUT_LANGUAGE]",),
        tier=TIER_CORE,
        purpose="the language the answer must be written in",
        source=AUTHORITY_FORMAT,
        authority=AUTHORITY_FORMAT,
    ),
    PromptSectionMeta(
        name="fact_ledger",
        markers=("[FACT_LEDGER]",),
        tier=TIER_CORE,
        purpose="alive players and engine-confirmed facts",
        source=AUTHORITY_ENGINE,
        authority=AUTHORITY_ENGINE,
    ),
    # TASK-060: 人狼セオリーカードは `_fact_header()` の末尾に貼り付いていたが、台帳へ
    # 登録しないと [FACT_LEDGER] のスパンに吸われて anchor 字数に計上され、段階除去も
    # 効かなかった。役割（推論の目安＝確定事実ではない）を明示して optional にする。
    PromptSectionMeta(
        name="werewolf_theory_card",
        markers=("[Werewolf Theory Card",),
        tier=TIER_OPTIONAL,
        purpose="role heuristics for CO and verdict interpretation (advisory, not facts)",
        source=AUTHORITY_TEMPORARY,
        authority=AUTHORITY_TEMPORARY,
        expires_at=None,
    ),
    PromptSectionMeta(
        name="claim_ledger",
        markers=("[CLAIM_LEDGER]",),
        tier=TIER_CORE,
        purpose="public COs, votes and deaths recorded with evidence ids",
        source=AUTHORITY_PUBLIC_RECORD,
        authority=AUTHORITY_ENGINE,
    ),
    PromptSectionMeta(
        name="inference_line",
        markers=("- Player Speech & Inferences (Unverified)",),
        tier=TIER_OPTIONAL,
        purpose="one-line digest of model-derived inferences",
        source=AUTHORITY_UNVERIFIED,
        authority=AUTHORITY_UNVERIFIED,
    ),
    PromptSectionMeta(
        name="hypothesis_ledger",
        markers=("[HYPOTHESIS_LEDGER]",),
        tier=TIER_OPTIONAL,
        purpose="the actor's own unverified hypotheses (summary + evidence ids)",
        source=AUTHORITY_UNVERIFIED,
        authority=AUTHORITY_UNVERIFIED,
    ),
    PromptSectionMeta(
        name="previous_topics",
        markers=("[PREVIOUS_TOPICS]",),
        tier=TIER_OPTIONAL,
        purpose="topics already raised, shown as state, not as an order to follow",
        source=AUTHORITY_UNVERIFIED,
        authority=AUTHORITY_UNVERIFIED,
    ),
    PromptSectionMeta(
        name="self_action_memory",
        markers=("[SELF_ACTION_MEMORY",),
        tier=TIER_CORE,
        purpose="the actor's own confirmed action history (consistency anchor)",
        source=AUTHORITY_ENGINE,
        authority=AUTHORITY_ENGINE,
    ),
    PromptSectionMeta(
        name="private_seer_result",
        markers=("[PRIVATE_SEER_RESULT",),
        tier=TIER_CORE,
        purpose="engine-issued verdicts for this actor only",
        source=AUTHORITY_ENGINE,
        authority=AUTHORITY_ENGINE,
    ),
    PromptSectionMeta(
        name="info_boundaries",
        markers=("[INFO_BOUNDARIES", "[情報の境界"),
        tier=TIER_CORE,
        purpose="what the actor actually knows vs. what they have claimed",
        source=AUTHORITY_ENGINE,
        authority=AUTHORITY_ENGINE,
    ),
    PromptSectionMeta(
        name="continuity_memo",
        markers=("[CONTINUITY_MEMO",),
        tier=TIER_OPTIONAL,
        purpose="derived reasoning state as evidence ids (text lives in [CLAIM_LEDGER])",
        source=AUTHORITY_UNVERIFIED,
        authority=AUTHORITY_UNVERIFIED,
    ),
    # TASK-062: エンジンが発行する論理状態のチェックポイント。確定事実・公開申告・反証・
    # 未解決の問いを権威レベル付きの短いID行で保持し、past本文の再掲の代わりにする。
    # core（絶対に落とさない）だが行数上限が決まっているため日跨ぎでも字数が増えない。
    PromptSectionMeta(
        name="continuity_checkpoint",
        markers=("[CONTINUITY_CHECKPOINT",),
        tier=TIER_CORE,
        purpose=(
            "engine-normalized logical state carried across days and turns "
            "(confirmed facts, public statements, refuted hypotheses, open questions with ids)"
        ),
        source=AUTHORITY_ENGINE,
        authority=AUTHORITY_ENGINE,
    ),
    # TASK-064: 役職能力の境界節。エンジンが実際に実行した行動と、本人が公開した主張を
    # 短い固定行で分ける（[DONE]=エンジン実行、[CLAIM]=本人申告）。エンジン結果・公開
    # 主張・推測の本文は重複掲載せずID参照のみ。口調・論点・結論には干渉しない。
    PromptSectionMeta(
        name="role_action_boundary",
        markers=("[ROLE_ACTION_BOUNDARY",),
        tier=TIER_CORE,
        purpose=(
            "what abilities the engine actually executed for this actor vs. what they "
            "publicly claimed (short symbol lines; no result bodies, ids refer to their ledgers)"
        ),
        source=AUTHORITY_ENGINE,
        authority=AUTHORITY_ENGINE,
    ),
    PromptSectionMeta(
        name="integrity_alerts",
        markers=("[INTEGRITY_ALERTS]",),
        tier=TIER_OPTIONAL,
        purpose="observe-only alerts on unusual public speech",
        source=AUTHORITY_UNVERIFIED,
        authority=AUTHORITY_UNVERIFIED,
    ),
    PromptSectionMeta(
        name="actor_private_memo",
        markers=("[ACTOR_PRIVATE_MEMO",),
        tier=TIER_CORE,
        purpose="engine-issued information allowed only to this actor",
        source=AUTHORITY_ENGINE,
        authority=AUTHORITY_ENGINE,
    ),
    PromptSectionMeta(
        name="reasoning_memo_own",
        markers=("[YOUR_PREVIOUS_REASONING_MEMO",),
        tier=TIER_CORE,
        purpose="the actor's own private reasoning from prior turns (consistency anchor)",
        source=AUTHORITY_SELF,
        authority=AUTHORITY_SELF,
    ),
    PromptSectionMeta(
        name="own_speeches",
        markers=("[Your Speeches Today",),
        tier=TIER_CORE,
        purpose="what this actor said today, by id when the body is already in evidence",
        source=AUTHORITY_PUBLIC_RECORD,
        authority=AUTHORITY_UNVERIFIED,
    ),
    PromptSectionMeta(
        name="day1_first_turn_guidance",
        markers=("[Day 1 First Turn",),
        tier=TIER_OPTIONAL,
        purpose="temporary guidance for Day 1 turn 1 only",
        source=AUTHORITY_TEMPORARY,
        authority=AUTHORITY_TEMPORARY,
        expires_at="D1T1",
    ),
    PromptSectionMeta(
        name="first_night_guidance",
        markers=("[First Night Guidance",),
        tier=TIER_OPTIONAL,
        purpose="temporary guidance for Night 1 only",
        source=AUTHORITY_TEMPORARY,
        authority=AUTHORITY_TEMPORARY,
        expires_at="N1",
    ),
    PromptSectionMeta(
        name="day_timeline",
        markers=("[DAY_TIMELINE]",),
        tier=TIER_CORE,
        purpose="what happened by now and which events have not happened yet",
        source=AUTHORITY_ENGINE,
        authority=AUTHORITY_ENGINE,
    ),
    PromptSectionMeta(
        name="round_transcript",
        markers=("[Full Current Round Dialogue]",),
        tier=TIER_OPTIONAL,
        purpose="header of the unshortened round transcript",
        source=AUTHORITY_PUBLIC_RECORD,
        authority=AUTHORITY_UNVERIFIED,
    ),
    PromptSectionMeta(
        name="conversation_evidence",
        markers=("[CONVERSATION_EVIDENCE", "[Previous Public Dialogue]"),
        tier=TIER_CORE,
        purpose="the single place where speech bodies are placed",
        source=AUTHORITY_PUBLIC_RECORD,
        authority=AUTHORITY_UNVERIFIED,
    ),
    PromptSectionMeta(
        name="current_task",
        markers=("[CURRENT_TASK]",),
        tier=TIER_CORE,
        purpose="the action to take on this turn",
        source=AUTHORITY_FORMAT,
        authority=AUTHORITY_FORMAT,
    ),
    PromptSectionMeta(
        name="question_block",
        markers=("[Current Question]",),
        tier=TIER_CORE,
        purpose="the question currently directed at this actor",
        source=AUTHORITY_PUBLIC_RECORD,
        authority=AUTHORITY_UNVERIFIED,
    ),
    PromptSectionMeta(
        name="direct_answer",
        markers=("[Direct Answer to Previous Question]",),
        tier=TIER_OPTIONAL,
        purpose="a restatement of the same question (droppable; body stays in evidence)",
        source=AUTHORITY_PUBLIC_RECORD,
        authority=AUTHORITY_UNVERIFIED,
    ),
    PromptSectionMeta(
        name="leak_boundary",
        markers=("[Leak Boundary", "[秘匿情報の扱い"),
        tier=TIER_CORE,
        purpose="role-specific secrecy boundaries",
        source=AUTHORITY_RULES,
        authority=AUTHORITY_RULES,
    ),
    PromptSectionMeta(
        name="behavior_contract",
        markers=("[BEHAVIOR_CONTRACT",),
        tier=TIER_CORE,
        purpose="the single consolidated contract block for behavior constraints",
        source=AUTHORITY_RULES,
        authority=AUTHORITY_FORMAT,
    ),
    PromptSectionMeta(
        name="instructions",
        markers=("[Instructions]",),
        tier=TIER_CORE,
        purpose="the required output format for this request",
        source=AUTHORITY_FORMAT,
        authority=AUTHORITY_FORMAT,
    ),
    PromptSectionMeta(
        name="reasoning_guidance",
        markers=("[REASONING_MEMO_GUIDANCE]",),
        tier=TIER_OPTIONAL,
        purpose="advice on what a private reasoning memo may contain",
        source=AUTHORITY_TEMPORARY,
        authority=AUTHORITY_TEMPORARY,
        expires_at=None,
    ),
    PromptSectionMeta(
        name="discussion_guidelines",
        markers=("[Discussion Guidelines & Logical Reasoning]",),
        tier=TIER_CORE,
        purpose="how to argue rationally (kept; duplicated bans move to the contract)",
        source=AUTHORITY_RULES,
        authority=AUTHORITY_RULES,
    ),
    PromptSectionMeta(
        name="output_checklist",
        markers=("[Final Output Checklist]", "[最終確認 - 日本語で出力]"),
        tier=TIER_CORE,
        purpose="a short final checklist in the response language (recency boost)",
        source=AUTHORITY_FORMAT,
        authority=AUTHORITY_FORMAT,
    ),
)

SECTION_META_BY_NAME: dict[str, PromptSectionMeta] = {
    section.name: section for section in PROMPT_SECTIONS
}


def section_meta(name: str) -> PromptSectionMeta:
    """名前でセクションメタを探す（未知の名前は ValueError）。"""
    try:
        return SECTION_META_BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown prompt section name: {name!r}") from None


def tier_of(name: str) -> str:
    return section_meta(name).tier


# ---------------------------------------------------------------------------
# optional の段階的除去（core は絶対に削らない）
# ---------------------------------------------------------------------------

# stage1 で論点要約と推理誘導を落とし、stage2 で仮説差分、stage3 で期限付きの暫定
# 案内と重複引用欄、stage4 で観測専用アラートと派生メモJSONを落とす。自分の推理メモ・
# エンジン発行情報・発言本文・現在の質問はどの段階でも削らない（一貫性优先）。
OPTIONAL_REMOVAL_STAGES: tuple[tuple[str, ...], ...] = (
    ("previous_topics", "reasoning_guidance"),
    # TASK-060: セオリーカード（推論の目安）は確定事実でも証拠でもないため、未検証の派生
    # 情報と同じ第2段階で落とす。ここを削ると約2.7k字が浮き、発言証拠と質問の領域が守れる。
    ("hypothesis_ledger", "inference_line", "werewolf_theory_card"),
    ("day1_first_turn_guidance", "first_night_guidance", "direct_answer"),
    ("integrity_alerts", "continuity_memo"),
)

OPTIONAL_REMOVAL_MAX_STAGE = len(OPTIONAL_REMOVAL_STAGES)


def optional_names_dropped_up_to(stage: int) -> tuple[str, ...]:
    """段階番号までの除去対象名を累計で返す（0 は除去なし）。"""
    if stage <= 0:
        return ()
    limited = min(int(stage), OPTIONAL_REMOVAL_MAX_STAGE)
    names: list[str] = []
    for stage_names in OPTIONAL_REMOVAL_STAGES[:limited]:
        for name in stage_names:
            if name not in names:
                names.append(name)
    return tuple(names)


CORE_NEVER_DROPPED = tuple(
    section.name for section in PROMPT_SECTIONS if section.tier == TIER_CORE
)


def drop_sections(text: str, names: Iterable[str]) -> tuple[str, tuple[str, ...], int]:
    """指定した optional 節だけを本文から除く（core の文字列には触れない）。

    節の区切りは台帳の全マーカーの出现位置で決めるため、入れ子見出しをまたいで
    除去することがない。返す値は (新しい本文, 実際に除いた名前, 減った文字数)。
    core を指定した場合と未知の名前を指定した場合は ValueError とする（設計上の禁止）。
    """
    wanted = [str(name) for name in names]
    unknown = sorted({name for name in wanted if name not in SECTION_META_BY_NAME})
    if unknown:
        raise ValueError(f"unknown prompt section names: {', '.join(unknown)}")
    forbidden = sorted(name for name in wanted if name in CORE_NEVER_DROPPED)
    if forbidden:
        # 確定事実・現在の質問・自分の行動記録が消えると一貫性が守れない。
        raise ValueError("core sections are never dropped: " + ", ".join(forbidden))
    if not text or not wanted:
        return text, (), 0
    spans = find_sections(text)
    cuts: list[tuple[int, int]] = []
    removed_names: list[str] = []
    for name in wanted:
        span = spans.get(name)
        if span is None:
            continue
        cuts.append(span)
        removed_names.append(name)
    if not cuts:
        return text, (), 0
    cuts.sort(reverse=True)
    pruned = text
    for start, end in cuts:
        pruned = pruned[:start] + pruned[end:]
    saved = len(text) - len(pruned)
    return pruned, tuple(sorted(removed_names)), max(0, saved)


def _first_marker_hit(text: str, marker: str) -> int:
    """マーカーの「見出しとしての出現位置」を求める（行頭 or プロンプト先頭を優先する）。

    TASK-060で必要になった修正。派生節の説明文が他の見出しトークンを本文中に再掲する
    ことがある（例: `[CONTINUITY_MEMO]` の案内が `[CLAIM_LEDGER]` に言及する）。素朴な
    `str.find()` だとその言及を節の開始と見誤り、スパン・節順検査・`drop_sections` が
    ずれる。ここでは行頭（または文頭）の出現を優先し、そうした出現が無い場合だけ従来どおり
    最初の出現へ戻る（既存プロンプトの後方互換）。
    """
    pattern = re.compile(re.escape(marker))
    for match in pattern.finditer(text):
        index = match.start()
        if index == 0 or text[index - 1] == "\n":
            return index
    return text.find(marker)


def _marker_positions(text: str) -> list[tuple[int, str]]:
    """台帳の全マーカー出现位置を (位置, 節名) の先頭順で返す（重複マーカーは除外）。"""
    found: list[tuple[int, str]] = []
    for section in PROMPT_SECTIONS:
        best_index = -1
        for marker in section.markers:
            index = _first_marker_hit(text, marker)
            if index >= 0 and (best_index < 0 or index < best_index):
                best_index = index
        if best_index >= 0:
            found.append((best_index, section.name, 0))
    found.sort()
    seen_indexes: set[int] = set()
    deduped: list[tuple[int, str]] = []
    for index, name, _tiebreak in found:
        if index in seen_indexes:
            continue
        seen_indexes.add(index)
        deduped.append((index, name))
    return deduped


def find_sections(text: str) -> dict[str, tuple[int, int]]:
    """現れる全節について (開始位置, 終了位置) を返す（終了は次のマーカー or 文末）。"""
    positions = _marker_positions(text)
    spans: dict[str, tuple[int, int]] = {}
    for order, (index, name) in enumerate(positions):
        end = positions[order + 1][0] if order + 1 < len(positions) else len(text)
        spans[name] = (index, end)
    return spans


def measure_section_chars(text: str) -> dict[str, int]:
    """セクション別の文字数を測る（TASK-057実装案4の `prompt_section_chars`）。"""
    spans = find_sections(text)
    return {name: max(0, end - start) for name, (start, end) in spans.items()}


def tier_char_totals(text: str) -> dict[str, int]:
    """core と optional の合計文字数（除去前後の比較に使う）。"""
    totals = {TIER_CORE: 0, TIER_OPTIONAL: 0}
    for name, chars in measure_section_chars(text).items():
        tier = SECTION_META_BY_NAME[name].tier
        totals[tier] = totals.get(tier, 0) + chars
    return totals


# ---------------------------------------------------------------------------
# 期限（expires_at）と暫定案内
# ---------------------------------------------------------------------------


def expires_label(day: int, turn: int, prefix: str = "D") -> str:
    """期限ラベルを作る（`D1T1` 形。夜ガイダンスは `N1`）。"""
    if prefix == "N":
        return f"N{int(day)}"
    return f"D{int(day)}T{int(turn)}"


def _parse_point(label: object) -> tuple[int, int] | None:
    """`D12T3` / `N1` を (day, turn) へ。読めなければ None（夜は turn=0）。"""
    if not isinstance(label, str):
        return None
    candidate = label.strip()
    night = re.fullmatch(r"[Nn](\d+)", candidate)
    if night:
        return int(night.group(1)), 0
    day_turn = re.fullmatch(r"[Dd](\d+)[Tt](\d+)", candidate)
    if day_turn:
        return int(day_turn.group(1)), int(day_turn.group(2))
    return None


def guidance_expired(expires_at: object, day: int, turn: int) -> bool:
    """暫定案内の期限 (`expires_at`) を過ぎたかを判定する（読めない・None は False）。"""
    limit = _parse_point(expires_at)
    if limit is None:
        return False
    current = (int(day), int(turn))
    if current[0] > limit[0]:
        return True
    return current[0] == limit[0] and current[1] > limit[1]


def expired_sections_present(text: str, day: int, turn: int) -> tuple[str, ...]:
    """本文に含まれる期限切れの暫定案内名を返す（観測専用・自動削除はしない）。"""
    stale = [
        name
        for name in find_sections(text)
        if SECTION_META_BY_NAME[name].authority == AUTHORITY_TEMPORARY
        and guidance_expired(SECTION_META_BY_NAME[name].expires_at, day, turn)
    ]
    return tuple(sorted(stale))


# ---------------------------------------------------------------------------
# 証拠IDによる参照（同じ発言本文を2度送らないための置き換え）
# ---------------------------------------------------------------------------

REFERENCE_PREFIX = "[ref:s"


def evidence_reference(speech_id: int) -> str:
    """発言本文の再掲の代わりに使う短い参照（`[ref:sN]`＝[CONVERSATION_EVIDENCE] の該当行）。"""
    return f"{REFERENCE_PREFIX}{int(speech_id)}]"


def quote_or_reference(
    content: object,
    speech_id: object,
    evidence_speech_ids: frozenset[int],
    *,
    note: str = "(same speech, quoted below)",
) -> tuple[str, bool, int]:
    """発言引用を「参照」へ差し替えられるかどうかを決める。

    - `speech_id` が無く、その発言が履歴に載っていると確実なときだけ参照化する
      （履歴から落ちた本文を指す空参照を作らないため、ID台帳と照合する）。
    - 返す値は (差し替えた文字列, 参照化できたか, 節約できた文字数)。
      節約が0以下のときは本文を残す（長いIDより短い引用のほうが合理的だから）。
    """
    body = str(content)
    if not isinstance(speech_id, int) or isinstance(speech_id, bool):
        return body, False, 0
    if int(speech_id) not in evidence_speech_ids:
        return body, False, 0
    reference = f"{evidence_reference(int(speech_id))} {note}"
    saved = len(body) - len(reference)
    if saved <= 0:
        return body, False, 0
    return reference, True, saved


def speech_body_ids(history: Iterable[object]) -> frozenset[int]:
    """発言リストから `speech_id` の集合を作る（None・bool は含めない）。"""
    ids: set[int] = set()
    for message in history:
        value = getattr(message, "speech_id", None)
        if isinstance(value, int) and not isinstance(value, bool):
            ids.add(int(value))
    return frozenset(ids)


# `[speech:12] Player 3: ...` 形の派生メモ行（推理メモの current_round_evidence など）。
_SPEECH_ENTRY_PATTERN = re.compile(r"^\[speech:(\d+)\]\s*([^:]{1,60}):\s*(.+)$", re.DOTALL)


def evidence_entry_reference(
    entry: object,
    evidence_speech_ids: frozenset[int],
) -> tuple[str, bool, int]:
    """派生メモ内の発言行を `[speech:N] 名前 (参照)` へ縮める。

    [CONVERSATION_EVIDENCE] に同じ本文が載っているIDだけを対象にする（履歴から落ちた
    発言を指す空参照を作らない）。変換できない行はそのまま返し、変更フラグと節約字数は
    診断へ渡す。名前とIDは残すので、議論の追跡性（どの発言を根拠にしたか）は保たれる。
    """
    text = str(entry)
    matched = _SPEECH_ENTRY_PATTERN.match(text.strip())
    if matched is None:
        return text, False, 0
    speech_id = int(matched.group(1))
    if speech_id not in evidence_speech_ids:
        return text, False, 0
    speaker = matched.group(2).strip()
    replacement = f"[speech:{speech_id}] {speaker} {evidence_reference(speech_id)}"
    saved = len(text) - len(replacement)
    if saved <= 0:
        return text, False, 0
    return replacement, True, saved


def evidence_entries_to_references(
    entries: Iterable[object],
    evidence_speech_ids: frozenset[int],
) -> tuple[list[str], dict[str, int]]:
    """派生メモの発言リストを一括で参照化し、節約量を診断にまとめる。"""
    converted: list[str] = []
    saved_chars = 0
    references = 0
    for entry in entries:
        replacement, changed, saved = evidence_entry_reference(entry, evidence_speech_ids)
        converted.append(replacement)
        if changed:
            references += 1
            saved_chars += saved
    diagnostics = {}
    if references:
        diagnostics["dedup_saved_chars"] = saved_chars
        diagnostics["dedup_references"] = references
    return converted, diagnostics


# ---------------------------------------------------------------------------
# behavior_contract（散在する命令を1つの契約へ集約し、重複・矛盾を静的検査する）
# ---------------------------------------------------------------------------

STRENGTH_STRONG = "strong"  # ゲートに相当する強い契約（秘匿・形式など）
STRENGTH_ADVISORY = "advisory"  # 議論を促す案内（守らなくても発話の自由を奪わない）
STRENGTH_FREEDOM = "freedom"  # モデルに返す自由の記述

# 同じ次元で競合しうる命令の次元名。自由と義務が同居するときの検査に使う。
DIMENSION_SECRECY = "secrecy"
DIMENSION_SOURCE_DISCIPLINE = "source_discipline"
DIMENSION_SELF_REFERENCE = "self_reference"
DIMENSION_FORMAT = "format"
DIMENSION_PASS = "pass"
DIMENSION_ANSWER_PRIORITY = "answer_priority"
DIMENSION_STRUCTURE = "structure"
DIMENSION_EXPRESSION = "expression"
DIMENSION_CONSISTENCY = "consistency"


@dataclass(frozen=True)
class BehaviorContractLine:
    """契約節の1行。`rule_id` が同じ行が2つあれば重複、次元次第では矛盾として数える。"""

    rule_id: str
    text: str
    strength: str
    dimension: str
    condition: str | None = None


# 正準となる制約ID（ログで規則を追跡できるようにする）。
CONTRACT_CORE_RULE_IDS: tuple[str, ...] = (
    "no_private_leak",
    "verdict_blocks_only",
    "speak_from_record",
    "no_third_person_self",
    "keep_required_format",
    "answer_in_language",
    "reasoning_not_in_body",
    "no_meta_talk",
    "pass_literal",
    "body_not_empty",
    "expression_free",
    "no_order_imposed",
    "update_not_contradict",
)

# rule_id -> (strength, dimension, condition)。ここで強度と発動条件を固定し、
# 「無条件の義務」と「無条件の自由」が同じ次元に並ぶ設計ミスを静的検査できるようにする。
CONTRACT_LINE_SPECS: dict[str, tuple[str, str, str | None]] = {
    "no_private_leak": (STRENGTH_STRONG, DIMENSION_SECRECY, None),
    "verdict_blocks_only": (STRENGTH_STRONG, DIMENSION_SECRECY, None),
    "speak_from_record": (STRENGTH_STRONG, DIMENSION_SOURCE_DISCIPLINE, None),
    "no_third_person_self": (STRENGTH_STRONG, DIMENSION_SELF_REFERENCE, None),
    "keep_required_format": (STRENGTH_STRONG, DIMENSION_FORMAT, None),
    "answer_in_language": (STRENGTH_STRONG, DIMENSION_FORMAT, None),
    "reasoning_not_in_body": (STRENGTH_STRONG, DIMENSION_FORMAT, None),
    "no_meta_talk": (STRENGTH_STRONG, DIMENSION_SOURCE_DISCIPLINE, None),
    "pass_literal": (STRENGTH_STRONG, DIMENSION_PASS, None),
    "body_not_empty": (STRENGTH_STRONG, DIMENSION_FORMAT, None),
    "expression_free": (STRENGTH_FREEDOM, DIMENSION_EXPRESSION, None),
    "no_order_imposed": (STRENGTH_FREEDOM, DIMENSION_STRUCTURE, None),
    "update_not_contradict": (STRENGTH_ADVISORY, DIMENSION_CONSISTENCY, None),
    # strictモードで直接問われたときだけ発動する義務。条件を明記するため次元は
    # answer_priority（自由側の structure / expression とは別次元にして誤検出を防ぐ）。
    "answer_when_asked": (
        STRENGTH_STRONG,
        DIMENSION_ANSWER_PRIORITY,
        "directly_addressed_and_strict_mode",
    ),
    "no_preset_style": (STRENGTH_FREEDOM, DIMENSION_EXPRESSION, None),
    "claim_separate_from_body": (STRENGTH_STRONG, DIMENSION_FORMAT, None),
}

# 正準テキスト。同じ制約を別の箇所に書く代わりに、この1行だけを契約節へ置く。
CONTRACT_LINE_TEXTS: dict[str, dict[str, str]] = {
    "no_private_leak": {
        "en": "- Strong: your true role or camp, unpublished verdicts, night actions and "
        "your ally never enter public speech.",
        "ja": "- 強い契約: 自分の真の役職・陣営、未公開の判定、夜の行動、仲間は公開の本文に書かない。",
    },
    "verdict_blocks_only": {
        "en": "- Verdict words belong only inside the CO and RESULT blocks.",
        "ja": "- 判定語を書くのは CO ブロックと結果ブロックの中だけ。",
    },
    "speak_from_record": {
        "en": "- Strong: speak only from the ledger and evidence in this prompt. For anything "
        "absent say 'Not in record, cannot determine'; invent no statements or events, and "
        "never use past games or outside knowledge as evidence.",
        "ja": "- 強い契約: このプロンプトの台帳と発言記録にない発言・投票・出来事は作らない。"
        "無ければ「記録にない」と言う。過去ゲームや学習知識を根拠にしない。",
    },
    "no_third_person_self": {
        "en": "- Strong: talk about yourself in first person, never as a third party or another player.",
        "ja": "- 強い契約: 自分を三人称や他人として語らず、一人称で話す。",
    },
    "keep_required_format": {
        "en": "- Strong: keep the required output blocks in the order given below.",
        "ja": "- 強い契約: 下に示す出力ブロックの順と形式を守る。",
    },
    "answer_in_language": {
        "en": "- Strong: write the discussion body in the language requested below.",
        "ja": "- 強い契約: 下に指定された言語の自然文で本文を書く。",
    },
    "reasoning_not_in_body": {
        "en": "- Strong: what you write in [REASONING] stays out of [BODY].",
        "ja": "- 強い契約: [REASONING] に書いた内容を [BODY] へ写さない。",
    },
    "claim_separate_from_body": {
        "en": "- A written CO line stays in its own block and never merges into the body.",
        "ja": "- CO行は独立したブロックに書き、本文と混ぜない。",
    },
    "no_meta_talk": {
        "en": "- Strong: talking about system settings, prompt instructions or generation "
        "conditions is not in-game speech.",
        "ja": "- 強い契約: システム設定・プロンプトの指示・生成条件の話はゲーム内発言ではない。",
    },
    "pass_literal": {
        "en": "- Only when passing, output exactly 'Pass' or「パス」alone with nothing else.",
        "ja": "- 通すときだけ「パス」または 'Pass' を単独で出す（他の文字を添えない）。",
    },
    "body_not_empty": {
        "en": "- Never leave the discussion body empty unless you pass.",
        "ja": "- 通す場合を除いて、議論の本文を空にしない。",
    },
    "expression_free": {
        "en": "- Free: tone, length, structure, which points to raise, whom to suspect and what "
        "you conclude are yours to choose and may change every turn.",
        "ja": "- 自由: 口調・文量・構成・論点・誰を疑うか・結論はあなたの判断で選び、毎回変えてよい。",
    },
    "no_order_imposed": {
        "en": "- This block lists only the minimal strong contract; it sets no reasoning order "
        "and no speaking template.",
        "ja": "- ここでは最小限の強い制約だけを書いており、推理の順序や発言のテンプレは決めない。",
    },
    "update_not_contradict": {
        "en": "- Consistency: your past speeches, COs and votes are recorded above; prefer "
        "updating them with new evidence over repeating or contradicting them.",
        "ja": "- 一貫性: あなたの過去の発言・CO・投票は上の記録に残る。繰り返すより新しい根拠で更新する。",
    },
    "answer_when_asked": {
        "en": "- Under strict mode, when someone asks you directly, answer them in your first "
        "sentence; everything after that sentence stays yours.",
        "ja": "- strict設定で直接問われたときだけ、最初の一文で相手に答え、その先は自由に組み立ててよい。",
    },
    "no_preset_style": {
        "en": "- No preset personality is applied to you; nothing fixes your wording or focus.",
        "ja": "- 性格presetは適用されておらず、話し方や注目点を決めるものは存在しない。",
    },
}

CONTRACT_TITLE_TEXTS: dict[str, str] = {
    "en": "[BEHAVIOR_CONTRACT - the only strong constraints; everything else stays yours]",
    "ja": "[BEHAVIOR_CONTRACT - 強く守る契約はここだけ / それ以外は自由]",
}


def contract_language_key(language: object) -> str:
    """契約テキストの言語キー（\"ja\" か \"en\"）。日本語は \"ja\" に寄せる。"""
    if isinstance(language, str):
        candidate = language.strip().lower()
        if candidate in {"日本語", "japanese", "ja"}:
            return "ja"
    return "en"


def behavior_contract_lines(
    *,
    language: object = "English",
    hint_mode: str = "soft",
    include_claim: bool = True,
    free_personality: bool = False,
) -> list[BehaviorContractLine]:
    """契約節の行を組み立てる（強いのは秘匿と形式だけ、他は自由の明記）。"""
    key = contract_language_key(language)
    order = list(CONTRACT_CORE_RULE_IDS)
    if include_claim and "claim_separate_from_body" not in order:
        # COを書く権利がある依頼では「CO行と本文を混ぜない」を強い契約として1行だけ足す。
        order.insert(order.index("body_not_empty"), "claim_separate_from_body")
    if str(hint_mode).strip().lower() == "strict":
        order.append("answer_when_asked")
    # TASK-057: 自由発言モードでも契約行は増やさない。preset との差分はシステムプロンプトの
    # [Expression Freedom] ブロックと [BODY] 案内1行だけに留める設計（TASK-044）を保つため、
    # 「preset なし」の説明は契約節では繰り返さない（自由行の文面は共通で十分）。
    lines: list[BehaviorContractLine] = []
    for rule_id in order:
        spec = CONTRACT_LINE_SPECS.get(rule_id)
        texts = CONTRACT_LINE_TEXTS.get(rule_id)
        if spec is None or texts is None:
            continue
        text = texts.get(key) or texts["en"]
        strength, dimension, condition = spec
        lines.append(
            BehaviorContractLine(
                rule_id=rule_id,
                text=text,
                strength=strength,
                dimension=dimension,
                condition=condition,
            )
        )
    return lines


def render_behavior_contract(
    *,
    language: object = "English",
    hint_mode: str = "soft",
    include_claim: bool = True,
    free_personality: bool = False,
) -> tuple[str, dict[str, object]]:
    """契約節を1ブロックとして描画する（`[BEHAVIOR_CONTRACT]`）。

    返す診断には静的検査の結果を入れる。契約行の重複や矛盾があっても本文は書き換えず、
    数値として記録だけ行う（設計上のミスをログで発見できるようにするため）。
    """
    lines = behavior_contract_lines(
        language=language,
        hint_mode=hint_mode,
        include_claim=include_claim,
        free_personality=free_personality,
    )
    key = contract_language_key(language)
    title = CONTRACT_TITLE_TEXTS[key]
    block = title + "\n" + "\n".join(line.text for line in lines) + "\n\n"
    findings = inspect_behavior_contract(lines)
    diagnostics: dict[str, object] = {
        "behavior_contract_chars": len(block),
        "behavior_contract_lines": len(lines),
        "behavior_contract_duplicate_rules": sum(
            1 for finding in findings if finding["kind"] == "duplicate_rule"
        ),
        "behavior_contract_conflict_count": sum(
            1 for finding in findings if finding["kind"] == "obligation_vs_freedom"
        ),
        "behavior_contract_conditional_pairs": sum(
            1 for finding in findings if finding["kind"] == "conditional_pair"
        ),
        "behavior_contract_findings": [
            f"{finding['kind']}:{','.join(finding['rule_ids']) or '-'}"
            for finding in findings
        ],
    }
    return block, diagnostics


CONTRACT_TITLE_MARKERS: tuple[str, ...] = ("[BEHAVIOR_CONTRACT",)


def inspect_behavior_contract(
    lines: Iterable[BehaviorContractLine],
) -> list[dict[str, object]]:
    """契約行の重複と矛盾を静的検査する（本文の書き換えではなく設計の検証）。

    - `duplicate_rule`: 同じ rule_id が2行以上ある（同じ制約の再掲＝優先順位の混乱のもと）
    - `obligation_vs_freedom`: 無条件の義務と無条件の自由が同一次元でぶつかる
    - `conditional_pair`: 発動条件つき義務と自由行が同じ次元にある（矛盾ではない）
    - `freedom_without_anchor`: 強い制約だけでは書けるが自由度の宣言がない
    """
    items = [line for line in lines if isinstance(line, BehaviorContractLine)]
    findings: list[dict[str, object]] = []
    seen: dict[str, int] = {}
    for item in items:
        seen[item.rule_id] = seen.get(item.rule_id, 0) + 1
    duplicated = sorted(rule_id for rule_id, count in seen.items() if count > 1)
    if duplicated:
        findings.append(
            {
                "kind": "duplicate_rule",
                "rule_ids": duplicated,
                "note": "the same constraint appears more than once in the contract",
            }
        )
    unconditional_obligations = [
        item
        for item in items
        if item.strength == STRENGTH_STRONG and item.condition is None
    ]
    conditioned_obligations = [
        item
        for item in items
        if item.strength == STRENGTH_STRONG and item.condition is not None
    ]
    freedom_items = [
        item for item in items if item.strength == STRENGTH_FREEDOM and item.condition is None
    ]
    conflicted_dimensions: set[str] = set()
    for obligation in unconditional_obligations:
        for freedom in freedom_items:
            if freedom.dimension != obligation.dimension:
                continue
            conflicted_dimensions.add(obligation.dimension)
            findings.append(
                {
                    "kind": "obligation_vs_freedom",
                    "rule_ids": sorted({obligation.rule_id, freedom.rule_id}),
                    "dimension": obligation.dimension,
                    "note": (
                        "an unconditional obligation and an unconditional freedom claim "
                        "cover the same dimension"
                    ),
                }
            )
    for obligation in conditioned_obligations:
        for freedom in freedom_items:
            if freedom.dimension != obligation.dimension:
                continue
            if obligation.dimension in conflicted_dimensions:
                continue
            findings.append(
                {
                    "kind": "conditional_pair",
                    "rule_ids": sorted({obligation.rule_id, freedom.rule_id}),
                    "dimension": obligation.dimension,
                    "note": f"the obligation applies only under {obligation.condition}",
                }
            )
    has_strong = any(item.strength == STRENGTH_STRONG for item in items)
    if has_strong and not freedom_items:
        findings.append(
            {
                "kind": "freedom_without_anchor",
                "rule_ids": [],
                "note": "constraints are stated without declaring what stays free",
            }
        )
    return findings


# ---------------------------------------------------------------------------
# プロンプト全体の制約重複スキャン（legacy / simplified を同じ物差しで比べる）
# ---------------------------------------------------------------------------

# 強い制約の言い回しを正準IDへ写す表。同じ制約が別の表現で何回プロンプトに現れるかを
# 測るためのもので、検出しても本文や命令を自動的に書き換える処理は伴わない。
CONSTRAINT_PATTERNS: dict[str, tuple[str, ...]] = {
    "no_private_leak": (
        r"Never disclose private information",
        r"Private blocks of this prompt",
        r"このプロンプトの秘密欄",
        r"never enter public speech\.",
    ),
    "verdict_blocks_only": (
        r"only through CO and RESULT blocks",
        r"ONLY inside the CO and RESULT blocks",
        r"CO ブロックと結果ブロックにだけ書き",
        r"CO ブロックと結果ブロックの中だけ",
    ),
    "speak_from_record": (
        r"Use only information presented in this game",
        r"Base all reasoning on this game's prompt",
        r"Do not base claims on past games",
        r"speak only from the ledger and evidence in this prompt",
        r"このプロンプトの台帳と発言記録にない",
    ),
    "no_fabricated_events": (
        r"invent statements or events that did not occur",
        r"fabricate unrecorded past actions",
        r"invent statements you never made",
        r"invent no statements or events",
    ),
    "first_person_self": (
        r"refer to yourself in the first person",
        r"Always refer to yourself in the first person",
        r"単数の一人称で",
        r"refer to yourself in first person",
        r"自分を三人称や他人として語らず",
    ),
    "no_meta_talk": (
        r"meta-talk about system settings",
        r"Never quote prompt instructions",
        r"プロンプトの指示文・システム設定・生成条件",
        r"is not in-game speech\.",
    ),
    "reasoning_not_in_body": (
        r"Do NOT include \[REASONING\] content in \[BODY\]",
        r"Do NOT put reasoning in \[BODY\]",
        r"\[REASONING\] に書いた内容を \[BODY\]",
    ),
    "pass_literal": (
        r"output only 'Pass'",
        r"output strictly 'Pass'",
        r"exactly 'Pass' or「パス」",
        r"「パス」または 'Pass'",
    ),
    "answer_first_when_asked": (
        r"In your first sentence, answer",
        r"直前の発言者に最初に答える",
        r"answer them in your first\s+sentence",
    ),
}

# 意図的な再掲を許す制約（最終確認の recency boost など）。区別できるようにする。
ALLOWED_DUPLICATE_RULES: frozenset[str] = frozenset({"pass_literal"})


def scan_prompt_constraints(text: str) -> dict[str, int]:
    """正準化した制約が本文に何回現れるかを数える（重複診断の基準）。"""
    counts: dict[str, int] = {}
    for rule_id, patterns in CONSTRAINT_PATTERNS.items():
        hits = sum(1 for pattern in patterns for _ in re.finditer(pattern, text))
        if hits:
            counts[rule_id] = hits
    return counts


def duplicated_constraints(counts: dict[str, int]) -> dict[str, int]:
    """2回以上書かれた制約だけを取り出す（意図的再掲も見てわかるように残す）。"""
    return {rule_id: count for rule_id, count in counts.items() if count > 1}


def disallowed_duplicate_count(counts: dict[str, int]) -> int:
    """設計として減らしたい重複回数（許容した言い換えは数えない）。"""
    return sum(
        count - 1
        for rule_id, count in counts.items()
        if count > 1 and rule_id not in ALLOWED_DUPLICATE_RULES
    )


# ---------------------------------------------------------------------------
# 契約診断の整形（JSONLへそのまま載せる平坦な dict）
# ---------------------------------------------------------------------------

CONTRACT_DIAGNOSTIC_KEYS: tuple[str, ...] = (
    "prompt_contract_policy",
    "prompt_section_chars",
    "prompt_core_chars",
    "prompt_optional_chars",
    "prompt_user_char_count",
    "prompt_duplicate_constraint_count",
    "prompt_duplicate_constraint_rules",
    "dedup_saved_chars",
    "dedup_reference_count",
    "contract_removal_stage",
    "dropped_optional_sections",
)


@dataclass
class ContractDiagnostics:
    """1リクエスト分の契約診断（ログキー名のまま保持する）。"""

    policy: str
    stage: int = 0
    dropped: tuple[str, ...] = ()
    dedup_saved_chars: int = 0
    dedup_references: int = 0
    contract_findings: dict[str, object] = field(default_factory=dict)

    def apply_to(self, user_prompt: str) -> dict[str, object]:
        """組み上がり後の user_prompt から契約診断をまとめて作る（JSON化される前提）。"""
        totals = tier_char_totals(user_prompt)
        counts = scan_prompt_constraints(user_prompt)
        duplicated = duplicated_constraints(counts)
        return {
            **self.contract_findings,
            "prompt_contract_policy": normalize_prompt_contract_policy(self.policy),
            "prompt_section_chars": measure_section_chars(user_prompt),
            "prompt_core_chars": totals[TIER_CORE],
            "prompt_optional_chars": totals[TIER_OPTIONAL],
            "prompt_user_char_count": len(user_prompt),
            "prompt_duplicate_constraint_count": disallowed_duplicate_count(counts),
            "prompt_duplicate_constraint_rules": sorted(duplicated),
            "dedup_saved_chars": max(0, int(self.dedup_saved_chars)),
            "dedup_reference_count": max(0, int(self.dedup_references)),
            "contract_removal_stage": max(0, int(self.stage)),
            "dropped_optional_sections": [str(name) for name in self.dropped],
        }


__all__ = [
    "ALLOWED_DUPLICATE_RULES",
    "AUTHORITY_ENGINE",
    "AUTHORITY_FORMAT",
    "AUTHORITY_PUBLIC_RECORD",
    "AUTHORITY_RULES",
    "AUTHORITY_SELF",
    "AUTHORITY_TEMPORARY",
    "AUTHORITY_UNVERIFIED",
    "CONTRACT_DIAGNOSTIC_KEYS",
    "CONTRACT_CORE_RULE_IDS",
    "CONTRACT_LINE_SPECS",
    "CONTRACT_LINE_TEXTS",
    "CONTRACT_TITLE_MARKERS",
    "CONTRACT_TITLE_TEXTS",
    "CONSTRAINT_PATTERNS",
    "CORE_NEVER_DROPPED",
    "DIMENSION_ANSWER_PRIORITY",
    "DIMENSION_CONSISTENCY",
    "DIMENSION_EXPRESSION",
    "DIMENSION_FORMAT",
    "DIMENSION_PASS",
    "DIMENSION_SECRECY",
    "DIMENSION_SELF_REFERENCE",
    "DIMENSION_SOURCE_DISCIPLINE",
    "DIMENSION_STRUCTURE",
    "OPTIONAL_REMOVAL_MAX_STAGE",
    "OPTIONAL_REMOVAL_STAGES",
    "PROMPT_CONTRACT_POLICIES",
    "PROMPT_CONTRACT_POLICY_DEFAULT",
    "PROMPT_CONTRACT_POLICY_LEGACY",
    "PROMPT_CONTRACT_POLICY_SIMPLIFIED",
    "PROMPT_SECTIONS",
    "REFERENCE_PREFIX",
    "SECTION_META_BY_NAME",
    "STRENGTH_ADVISORY",
    "STRENGTH_FREEDOM",
    "STRENGTH_STRONG",
    "TIER_CORE",
    "TIER_OPTIONAL",
    "BehaviorContractLine",
    "ContractDiagnostics",
    "PromptSectionMeta",
    "behavior_contract_lines",
    "contract_language_key",
    "disallowed_duplicate_count",
    "drop_sections",
    "duplicated_constraints",
    "evidence_entry_reference",
    "evidence_entries_to_references",
    "evidence_reference",
    "expired_sections_present",
    "expires_label",
    "find_sections",
    "guidance_expired",
    "inspect_behavior_contract",
    "measure_section_chars",
    "normalize_prompt_contract_policy",
    "optional_names_dropped_up_to",
    "prompt_contract_policy_label",
    "quote_or_reference",
    "render_behavior_contract",
    "scan_prompt_constraints",
    "section_meta",
    "speech_body_ids",
    "tier_char_totals",
    "tier_of",
]
