"""TASK-058: core予算の段階圧縮（削除ではなく「役割を保った畳み込み」）。

TASK-057 の simplified でも、8k系モデルは optional を stage4まで落とした後に
core だけで割り当てを超える（game_106015c79ec9: 平均core 14,257.5字）。そこで
core を3つの保持層に分け、層ごとに違う扱いを許可する。

- anchor   : ルール・秘匿境界・現在質問・現在の行動形式・自分の行動記録・
             自分の推理メモ・エンジン発行情報・DAY_TIMELINE。常に全文保持。
             ここが減ると立場の一貫性と現在の質問への回答が壊れる。
- recent   : 現在ターンの先行発言・直近の主張更新・現在の主張台帳。本文保持を
             優先する（議論の接続点だから）。
- historical: 既に台帳へ正規化済みの過去イベント。本文ではなく時点・actor・
             evidence IDだけの短い行へ畳む（同じ事実の二重列挙を防ぐ）。

設計思想（TASK-058 / ユーザー指示）:
- コンテキスト節約: 過去の低優先発言は stub 行（約40字）へ畳み、本文ぶんを削る。
  ただし単純な末尾切り捨てはしない。現在の質問の対象、直近の反証、役職主張と
  その根拠、投票に関係する発言は優先して本文を残す。
- モデルの一貫性: anchor はどのリクエストでも無傷。stub化したIDは
  `[ref:sN]` 参照候補から外す（本文が消えた場所を指す空参照を作らない）。
- 合理的な議論: stub行にも時点・ID・話者は残るので「何が起きていたか」の追跡は
  保たれる。推理の順番は指示しない。
- 自由の担保: 発言本文そのものの書き換え・再生成は行わない。圧縮はプロンプト
  上だけで、公開ログには本文がそのまま残る（`[body withheld ...]` は
  「プロンプトで載せない」ことを示す追跡マーカー）。

レイアウト（attention最適化）もここで宣言する。順序は挙動を変えてしまうため、
`legacy` ではTASK-057までの並びをそのまま使い、既定（simplified）だけ v2 へ移す。
"""

from __future__ import annotations

import re

from .prompt_contract import find_sections

# ---------------------------------------------------------------------------
# ポリシー（既定は layered。off は A/B 比較用の旧挙動＝予算超過時は従来削除のみ）
# ---------------------------------------------------------------------------

CORE_COMPRESSION_POLICY_OFF = "off"
CORE_COMPRESSION_POLICY_LAYERED = "layered"
CORE_COMPRESSION_POLICIES: tuple[str, ...] = (
    CORE_COMPRESSION_POLICY_LAYERED,
    CORE_COMPRESSION_POLICY_OFF,
)
CORE_COMPRESSION_POLICY_DEFAULT = CORE_COMPRESSION_POLICY_LAYERED


def normalize_core_compression_policy(value: object) -> str:
    """設定値を "layered" / "off" へ正規化する（不明値は既定の layered）。"""
    if isinstance(value, bool):
        return CORE_COMPRESSION_POLICY_LAYERED if value else CORE_COMPRESSION_POLICY_OFF
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in CORE_COMPRESSION_POLICIES:
            return candidate
        if candidate in {"on", "true", "core", "layers"}:
            return CORE_COMPRESSION_POLICY_LAYERED
        if candidate in {"none", "legacy", "false", "plain"}:
            return CORE_COMPRESSION_POLICY_OFF
    return CORE_COMPRESSION_POLICY_DEFAULT


def core_compression_policy_label(policy: object) -> str:
    """ログ・CLI表示用のラベル（正規化済みの文字列を返す）。"""
    return normalize_core_compression_policy(policy)


# ---------------------------------------------------------------------------
# core の保持層（anchor / recent / historical）
# ---------------------------------------------------------------------------

LAYER_ANCHOR = "anchor"
LAYER_RECENT = "recent"
LAYER_HISTORICAL = "historical"
CORE_RETENTION_LAYERS: tuple[str, ...] = (LAYER_ANCHOR, LAYER_RECENT, LAYER_HISTORICAL)

# 節名 -> 保持層。core節のうち「常に全文」（anchor）と「本文保持を優先する直近情報」
# （recent）をここで宣言する。historical は節そのものではなく、証拠欄の中で
# stub化された行（HISTORICAL_STUB_MARK を含む行）として現れる。
CORE_SECTION_LAYERS: dict[str, str] = {
    "system_rules": LAYER_ANCHOR,
    "output_language": LAYER_ANCHOR,
    "fact_ledger": LAYER_ANCHOR,
    "claim_ledger": LAYER_RECENT,
    "self_action_memory": LAYER_ANCHOR,
    "own_speeches": LAYER_RECENT,
    "private_seer_result": LAYER_ANCHOR,
    "info_boundaries": LAYER_ANCHOR,
    "actor_private_memo": LAYER_ANCHOR,
    "reasoning_memo_own": LAYER_ANCHOR,
    "day_timeline": LAYER_ANCHOR,
    "conversation_evidence": LAYER_RECENT,
    "current_task": LAYER_ANCHOR,
    # TASK-062: エンジン発行の論理状態チェックポイント。確定事実・反証・未解決の問いを
    # 載せる「いま使う」欄なので anchor（常に全文保持）。行数上限があるため日跨ぎで増えない。
    "continuity_checkpoint": LAYER_ANCHOR,
    # TASK-064: エンジン発行の役職能力の境界。実行と申告の区別という「いま使う」情報で、
    # 行数が本人COぶんの固定行しかないため anchor（常に全文保持・本文の再掲なし）。
    "role_action_boundary": LAYER_ANCHOR,
    "question_block": LAYER_ANCHOR,
    "leak_boundary": LAYER_ANCHOR,
    "behavior_contract": LAYER_ANCHOR,
    "instructions": LAYER_ANCHOR,
    "output_checklist": LAYER_ANCHOR,
}

# 証拠欄で本文の代わりに残す行の末尾マーカー。この文字列を含む行が historical の
# 実体であり、その字数を core_historical_chars として測る。
HISTORICAL_STUB_MARK = "[body withheld"

# stub行の注意書き（英語1形式だけにして行数解釈を安定させる。時点・IDはヘッダ側が持つ）
HISTORICAL_STUB_NOTE = "[body withheld in prompt - full text stays in the public log]"


def historical_stub_line(header: str, speech_id: object) -> str:
    """過去発言の本文の代わりに置く1行を作る（`[D1 T2 s7 Player 3] [body withheld ...]`）。"""
    if isinstance(speech_id, int) and not isinstance(speech_id, bool):
        return f"{header} [body withheld in prompt - full text stays in the public log as [speech:{speech_id}]]"
    return f"{header} {HISTORICAL_STUB_NOTE}"


def core_layer_char_totals(text: str) -> dict[str, int]:
    """core節を保持層ごとに測る。historical は stub行の実字数（証拠欄内）。"""
    totals = {layer: 0 for layer in CORE_RETENTION_LAYERS}
    spans = find_sections(text)
    for name, (start, end) in spans.items():
        layer = CORE_SECTION_LAYERS.get(name)
        if layer is None or layer == LAYER_HISTORICAL:
            continue
        totals[layer] += max(0, end - start)
    # 証拠欄の中で stub化した行ぶんは recent の本文字数から差し引き、historical へ移す。
    stub_chars = sum(len(line) + 1 for line in text.splitlines() if HISTORICAL_STUB_MARK in line)
    if stub_chars:
        totals[LAYER_RECENT] = max(0, totals[LAYER_RECENT] - stub_chars)
        totals[LAYER_HISTORICAL] += stub_chars
    return totals



# ---------------------------------------------------------------------------
# 証拠レコードの優先度（TASK-058実装案2「単純な末尾切り捨てを禁止」）
# ---------------------------------------------------------------------------

# 高いほど温存。0は必須（mandatory_reasons側で保護されている）。
PRIORITY_KEEP_NONE = 0
PRIORITY_CLAIM = 4  # 役職主張・占い判定とその根拠（常にstub化候補から外す）
PRIORITY_QUESTION = 3  # 現在の質問の対象になった発言
PRIORITY_COUNTER = 2  # 反証・投票関連（直近の論戦を作る材料）
PRIORITY_PLAIN = 1  # 上記に当たらない通常発言（古い順に畳む）

# 直近この本数だけは plain でも畳まない（現在ターン前後の議論の接続点だから）。
RECENT_KEEP_COUNT = 5

# 反証・投票関連の語（hybrid_retrieval側の選択理由と軽く、ここでは別枠にする）
COUNTER_VOTE_CUES: tuple[str, ...] = (
    "反証",
    "反論",
    "矛盾",
    "投票",
    "vote",
    "voting",
)


# 役職主張・占い判定を本文から拾う語（選択理由が付かない発言でも本文を残すための判定）。
# prompts.py が claim_or_seer_result を選ぶ正規表現と同じ対象を温存する。
CLAIM_CONTENT_PATTERN = re.compile(
    r"(?:\bCO\b|\bco\b|\bseer\b|\bmadman\b|\bprophet\b|divin|\bresult\b|占い|判定|結果|名乗)",
    re.IGNORECASE,
)


def looks_like_claim(content: str) -> bool:
    """本文に役職主張・占い判定の気配があるか（選択理由が無くても本文は畳まない）。"""
    return bool(CLAIM_CONTENT_PATTERN.search(str(content or "")))


def evidence_record_priority(
    reasons: object,
    content: str,
) -> int:
    """証拠1件を stub化する優先度を決める（高いほど本文を残す）。

    mandatoryな理由は呼び出し側で先に除外済み。ここでは claim > question >
    counter/vote > plain の順で、同じ優先度内では「古いものから」畳む。
    """
    reason_set = {str(reason) for reason in (reasons or ())}
    if "claim_or_seer_result" in reason_set:
        return PRIORITY_CLAIM
    if any(str(reason).startswith("question_related") for reason in reason_set):
        return PRIORITY_QUESTION
    text = str(content or "")
    if looks_like_claim(text):
        # 選択理由が無くても、役職主張・占い判定の本文は畳まない（TASK-058完了条件）
        return PRIORITY_CLAIM
    # hybrid_retrieval の vote_reason_or_question は「投票や質問の文脈で選ばれた」ことを
    # 示す広い理由で、古い雑談にも付く。これを反証として温存すると畳むものが無くなるため、
    # 反証・投票の気配は本文側の語で判定する（質問対象は question_related* で守られる）。
    lowered = text.lower()
    if any(cue in text or cue in lowered for cue in COUNTER_VOTE_CUES):
        return PRIORITY_COUNTER
    return PRIORITY_PLAIN


def sort_records_for_withholding(records: list) -> list:
    """stub化の順序（最初に落とすものが先頭）を返す。

    群は plain < counter/vote < question の順で畳み、claim群には一切触れない。
    同じ群の中では古い index を優先する（新しい直近は残す）。
    """
    return sorted(
        records,
        key=lambda record: (
            int(record.get("priority", PRIORITY_PLAIN)),
            int(record["index"]),
        ),
    )


# ---------------------------------------------------------------------------
# prompt layout v2（attention最適化の組み立て順）
# ---------------------------------------------------------------------------

PROMPT_LAYOUT_LEGACY = "legacy_v1"
PROMPT_LAYOUT_V2 = "v2_attention"

# v2 の節順（無い節は読み飛ばす）。動機:
# - primacy: ルールと言語 → 確定事実台帳 → 自分の一貫性アンカー（行動記録・
#   自分の発言・自分の推理メモ・エンジン発行情報）。ここは絶対に中身が減らない。
# - middle: プロンプト中部は注意が弱い。落としやすい派生要約（仮説差分・論点・
#   期限切れ案内・派生メモ）だけをこの帯に集め、段階除去と物理的に一致させる。
# - pre-recency: 発言証拠と現在質問を終端寄りに置く（直前の議論と質問への回答が
#   最優先で出るべきだから）。
# - recency: 秘匿境界と強い契約、出力形式と最終チェックリストで締める。
# 口調・論点・結論や推理の順番を決める節はどこにも置かない（中立文言は各節の内側）。
PROMPT_LAYOUT_V2_ORDER: tuple[str, ...] = (
    "system_rules",
    "output_language",
    "fact_ledger",
    "claim_ledger",
    "self_action_memory",
    "own_speeches",
    "actor_private_memo",
    "reasoning_memo_own",
    "private_seer_result",
    "info_boundaries",
    "day_timeline",
    "inference_line",
    "hypothesis_ledger",
    "previous_topics",
    "continuity_memo",
    "integrity_alerts",
    "day1_first_turn_guidance",
    "first_night_guidance",
    "round_transcript",
    "reasoning_guidance",
    "conversation_evidence",
    "current_task",
    "question_block",
    "direct_answer",
    "leak_boundary",
    "behavior_contract",
    "instructions",
    "discussion_guidelines",
    "output_checklist",
)


def validate_prompt_layout(text: str, order: tuple[str, ...]) -> list[dict[str, object]]:
    """実プロンプトの節が指定順に並んでいるかを検証する（編集ミスの静的検査）。

    欠落は許す（節が存在しないターンがある）。テキスト出現順に歩き、より後に来るはずの
    節が先に現れていた場所を「そこ以降が崩れている」として報告する。報告対象は実際に
    順番を守れていない側（`name`）で、先に現れた要因の節名を `after` に残す。
    """
    spans = find_sections(text)
    ranks = {name: position for position, name in enumerate(order)}
    violations: list[dict[str, object]] = []
    best_rank = -1
    best_name: str | None = None
    for name, (start, _end) in sorted(spans.items(), key=lambda item: int(item[1][0])):
        rank = ranks.get(name)
        if rank is None:
            continue  # レイアウト外（legacy専用の節など）は照合しない
        if best_name is not None and rank < best_rank:
            violations.append(
                {
                    "kind": "out_of_order",
                    "name": name,
                    "after": best_name,
                    "start": int(start),
                    "previous_start": int(spans[best_name][0]),
                }
            )
            continue
        if rank >= best_rank:
            best_rank = rank
            best_name = name
    return violations


__all__ = [
    "CORE_COMPRESSION_POLICIES",
    "CORE_COMPRESSION_POLICY_DEFAULT",
    "CORE_COMPRESSION_POLICY_LAYERED",
    "CORE_COMPRESSION_POLICY_OFF",
    "CORE_RETENTION_LAYERS",
    "CORE_SECTION_LAYERS",
    "COUNTER_VOTE_CUES",
    "CLAIM_CONTENT_PATTERN",
    "HISTORICAL_STUB_MARK",
    "HISTORICAL_STUB_NOTE",
    "LAYER_ANCHOR",
    "LAYER_HISTORICAL",
    "LAYER_RECENT",
    "PRIORITY_CLAIM",
    "PRIORITY_COUNTER",
    "PRIORITY_KEEP_NONE",
    "PRIORITY_PLAIN",
    "PRIORITY_QUESTION",
    "PROMPT_LAYOUT_LEGACY",
    "PROMPT_LAYOUT_V2",
    "PROMPT_LAYOUT_V2_ORDER",
    "RECENT_KEEP_COUNT",
    "core_compression_policy_label",
    "core_layer_char_totals",
    "evidence_record_priority",
    "historical_stub_line",
    "looks_like_claim",
    "normalize_core_compression_policy",
    "sort_records_for_withholding",
    "validate_prompt_layout",
]

