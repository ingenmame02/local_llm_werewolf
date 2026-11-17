
import re
from dataclasses import dataclass
from typing import Literal

from .models import Message

CO_MARKER = "【CO】"
RESULT_MARKER = "【結果】"
BODY_MARKER = "【本文】"
REASONING_MARKER = "【推理メモ】"
CO_MARKER_EN = "[CO]"
RESULT_MARKER_EN = "[RESULT]"
BODY_MARKER_EN = "[BODY]"
REASONING_MARKER_EN = "[REASONING]"

ValidationSeverity = Literal["fatal", "repairable", "warning", "ok"]

SEVERITY_FATAL: ValidationSeverity = "fatal"
SEVERITY_REPAIRABLE: ValidationSeverity = "repairable"
SEVERITY_WARNING: ValidationSeverity = "warning"
SEVERITY_OK: ValidationSeverity = "ok"

# TASK-038: 「質問に答えてください」のお節介をモード化して、形式合わせのコストを減らす。
# - strict: 1文目ルール（旧来動作・意味的再試行あり）
# - soft:   観測のみ（質問者・引用を提示し、回答は強制・再試行しない）
# - none:   質問欄も指示も出さない（完全中立・検証も行う）
QUESTION_HINT_STRICT = "strict"
QUESTION_HINT_SOFT = "soft"
QUESTION_HINT_NONE = "none"
QUESTION_HINT_MODES: frozenset[str] = frozenset(
    {QUESTION_HINT_STRICT, QUESTION_HINT_SOFT, QUESTION_HINT_NONE}
)
DEFAULT_QUESTION_HINT_MODE = QUESTION_HINT_SOFT


def normalize_question_hint_mode(value: object) -> str:
    """CLI・プレイヤー設定の値を質問ヒントモードへ正規化する（不明値はsoft）。"""
    if isinstance(value, str) and value.strip():
        normalized = value.strip().lower()
        if normalized in QUESTION_HINT_MODES:
            return normalized
    return DEFAULT_QUESTION_HINT_MODE


def answers_previous_question(text: str, previous_content: str) -> bool:
    """直前の発言への回答が含まれるかを軽量に判定する（名前言及または語彙の重複）。

    1文目に名前があるかではなく、本文のどこかに回答内容が含まれるかを見る。
    文を句読点・助詞で分割した語彙（2文字以上）が本文に現れれば回答とみなす。
    """
    if not isinstance(text, str) or not isinstance(previous_content, str):
        return False
    if not text or not previous_content:
        return False
    important: list[str] = []
    for sentence in split_sentences(previous_content):
        important.extend(re.findall(r"Player\s+\d+|P\d+|[\u3040-\u9fffA-Za-z0-9]+", sentence))
    tokens = [
        token
        for token in (
            part.strip()
            for part in re.split(r"[\sのをにへはがでとやも、,.！？?・:：]+", " ".join(important))
        )
        if len(token) >= 2
    ]
    mentioned = re.findall(r"Player\s+\d+|P\d+", previous_content)
    if any(name in text for name in mentioned):
        # 直前の発言者（または話題のプレイヤー）へ言及していれば回答とみなす
        return True
    return any(token in text for token in tokens[:8])

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])\s*|\n+")


@dataclass(frozen=True)
class DiscussionOutput:
    claim: str
    body: str
    result: str = ""
    reasoning: str = ""


@dataclass(frozen=True)
class DiscussionBodyDiagnosis:
    """議論本文の違反診断結果。秘匿情報は含まず、診断メタ情報のみを持つ。"""

    severity: ValidationSeverity
    reasons: tuple[str, ...] = ()
    fatal: bool = False
    repairable: bool = False
    warning: bool = False
    publishable: bool = False


# 弱いモデルが「まだ結果を公開しない」意思を [RESULT] 行に書く場合の空表記。
# 「判定未公開」の表記を空へ正規化し、`[RESULT] None` を公開判定として誤検出する
# `seer_result_without_claim` と形式失敗（再試行→フォールバック）を防ぐ（TASK-041）。
_NO_RESULT_TOKENS: frozenset[str] = frozenset(
    {
        "",
        "none",
        "null",
        "n/a",
        "na",
        "no",
        "no result",
        "no_result",
        "not revealed",
        "unpublished",
        "なし",
        "無し",
        "未公開",
        "未発表",
        "結果なし",
        "-",
        "ー",
        "－",
    }
)


def normalize_result_field(value: object) -> str:
    """[RESULT]行の値を正規化する（未公開を表す表記は空文字、判定語はそのまま）。"""
    if value is None or not isinstance(value, str):
        return ""
    text = value.strip()
    if text.lower() in _NO_RESULT_TOKENS or text in _NO_RESULT_TOKENS:
        return ""
    return text


# `[CO]`行の値を正規化する。弱いモデルは English 指示を日本語で返すため
# `[CO] なし`・`[CO] 保留` 等の表記も none/hold に寄せる（TASK-041）。
_CLAIM_NORMALIZED: dict[str, str] = {
    "coなし": "none",
    "co無し": "none",
    "coしない": "none",
    "なし": "none",
    "無し": "none",
    "しない": "none",
    "none": "none",
    "no": "none",
    "no co": "none",
    "co none": "none",
    "co保留": "hold",
    "保留": "hold",
    "hold": "hold",
    "co hold": "hold",
}


def normalize_claim_field(value: object) -> str:
    """[CO]行の値を正規化する（none/holdの日本語・英語表記を統一し、役職主張はそのまま）。"""
    if value is None or not isinstance(value, str):
        return ""
    text = value.strip()
    return _CLAIM_NORMALIZED.get(text.lower(), text)


def extract_discussion_body(content: str | None) -> str | None:
    """議論本文を自由形式で抽出する。

    - 思考タグ（`<think>`…`</think>`）を除去する
    - `【CO】` / `[CO]` / `【結果】` / `[RESULT]` 行は役職主張として分離し、本文から除外する
    - `【推理メモ】` / `[REASONING]` 行は本人限定の推理メモとして本文から除外する
    - `【本文】` / `[BODY]` マーカーがあればその中身を優先する
    - JSON・見出し・箇条書きがなくても自然文をそのまま返す
    - 空本文・Noneは None を返す（「パス」は有効な発言としてそのまま返す）
    """
    if content is None or not isinstance(content, str) or not content:
        return None
    text = _THINK_BLOCK.sub("", content)
    text = _UNCLOSED_THINK_BLOCK.sub("", text)
    body_marker_match = re.search(r"(?:【本文】|\[BODY\])\s*", text, re.IGNORECASE)
    if body_marker_match is not None:
        text = text[body_marker_match.end():]
    kept_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not re.match(
            r"^(?:【(?:CO|結果|推理メモ)】|\[(?:CO|RESULT|REASONING)\])",
            line.strip(),
            re.IGNORECASE,
        )
    ]
    body = " ".join(kept_lines).strip()
    if not body:
        return None
    return body
def extract_reasoning_memo(content: str | None) -> str | None:
    """モデル出力から推理メモを抽出する（本人限定・公開本文には含めない）。

    `【推理メモ】` / `[REASONING]` マーカーの直後から次マーカーまたは文末までを
    取り出す。マーカーが存在しない場合は None を返す。
    """
    if content is None or not isinstance(content, str) or not content:
        return None
    text = _THINK_BLOCK.sub("", content)
    text = _UNCLOSED_THINK_BLOCK.sub("", text)
    marker_match = re.search(
        r"(?:【推理メモ】|\[REASONING\])\s*",
        text,
        re.IGNORECASE,
    )
    if marker_match is None:
        return None
    remaining = text[marker_match.end():]
    # 次のマーカー行（CO, RESULT, BODY）までを推理メモとして取り出す
    next_marker = re.search(
        r"^(?:【(?:CO|結果|本文)】|\[(?:CO|RESULT|BODY)\])",
        remaining,
        re.IGNORECASE | re.MULTILINE,
    )
    if next_marker is not None:
        memo_text = remaining[:next_marker.start()]
    else:
        memo_text = remaining
    memo = memo_text.strip()
    if not memo:
        return None
    return memo




def is_pass_body(body: str | None) -> bool:
    if body is None:
        return False
    stripped = body.strip().strip(".!。！").lower()
    return stripped in ("パス", "pass")


def split_sentences(text: str | None) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def find_repeated_topic(
    body: str | None, transcript: tuple[Message, ...] = ()
) -> bool:
    """現在ラウンドの発言と本文が実質的に重複するかを軽量に判定する。"""
    if not isinstance(body, str) or not body or not transcript:
        return False
    body_sentences = split_sentences(body)
    if len(body_sentences) < 2:
        return False
    history = "\n".join(
        message.content
        for message in transcript
        if isinstance(message.content, str)
        and message.content
        and not is_pass_body(message.content)
    )
    if not history:
        return False
    matched = sum(1 for sentence in body_sentences if sentence in history)
    return matched >= max(1, len(body_sentences) // 2)


@dataclass(frozen=True)
class DiscussionClaim:
    claim: str
    result: str = ""
    ignored_trailing_text: bool = False


def has_discussion_output_markers(content: str | None) -> bool:
    if content is None or not isinstance(content, str) or not content:
        return False
    return bool(
        re.search(r"【(?:CO|結果|本文|推理メモ)】|\[(?:CO|RESULT|BODY|REASONING)\]", content, re.IGNORECASE)
    )


def parse_discussion_output(content: str | None) -> DiscussionOutput | None:
    """`【CO】` / `[CO]`（+任意の`【結果】` / `[RESULT]` + 任意の`【推理メモ】` / `[REASONING]`）+`【本文】` / `[BODY]`の構造化応答を解析する。"""
    if content is None or not isinstance(content, str) or not content:
        return None
    if not has_discussion_output_markers(content):
        return None
    co_count = len(re.findall(r"【CO】|\[CO\]", content, re.IGNORECASE))
    body_count = len(re.findall(r"【本文】|\[BODY\]", content, re.IGNORECASE))
    result_count = len(re.findall(r"【結果】|\[RESULT\]", content, re.IGNORECASE))
    reasoning_count = len(re.findall(r"【推理メモ】|\[REASONING\]", content, re.IGNORECASE))
    if co_count != 1 or body_count > 1 or result_count > 1 or reasoning_count > 1:
        return None
    has_result = result_count > 0
    has_reasoning = reasoning_count > 0
    has_body = body_count > 0
    if not has_result and not has_body:
        return None
    # 行頭マーカー直後の空白は水平空白に限定する。空の `[RESULT]` 行（未公開の表記）で
    # 改行を消費すると `[BODY]` 行が匹配不能になり、形式失敗→再試行になっていた（TASK-041）。
    match = re.fullmatch(
        r"\s*(?:【CO】|\[CO\])[ \t]*(?P<claim>.*?)[ \t]*"
        r"(?:\r?\n[ \t]*(?:【結果】|\[RESULT\])[ \t]*(?P<result>.*?)[ \t]*)?"
        r"(?:\r?\n[ \t]*(?:【推理メモ】|\[REASONING\])[ \t]*(?P<reasoning>.*?)[ \t]*)?"
        r"(?:\r?\n[ \t]*(?:【本文】|\[BODY\])[ \t]*(?P<body>.*?)[ \t]*)?",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return None
    claim = normalize_claim_field(match.group("claim"))
    result = (match.group("result") or "").strip()
    reasoning = (match.group("reasoning") or "").strip()
    body = (match.group("body") or "").strip()
    if not claim:
        return None
    if has_body and not body:
        return None
    # 未公開を表す [RESULT] 値は「結果なし」へ正規化する（公開判定の語はそのまま）。
    return DiscussionOutput(
        claim=claim, body=body, result=normalize_result_field(result), reasoning=reasoning
    )


def parse_discussion_claim(content: str | None) -> DiscussionClaim | None:
    """Parse a CO decision and safely ignore prose outside its structured lines."""
    if content is None or not isinstance(content, str) or not content:
        return None
    co_count = len(re.findall(r"【CO】|\[CO\]", content, re.IGNORECASE))
    result_count = len(re.findall(r"【結果】|\[RESULT\]", content, re.IGNORECASE))
    if co_count != 1 or result_count > 1:
        return None
    lines = content.strip().splitlines()
    if not lines:
        return None
    claim_match = re.fullmatch(
        r"(?:【CO】|\[CO\])\s*(?P<claim>.+?)\s*", lines[0].strip(), re.IGNORECASE
    )
    if claim_match is None:
        return None
    # 共通の正規化（`[CO] なし`・`[CO] 保留` 等の日本語表記も none/hold に寄せる）。
    claim = normalize_claim_field(claim_match.group("claim"))
    if not claim:
        return None
    remaining = lines[1:]
    result = ""
    if remaining and re.match(r"^(?:【結果】|\[RESULT\])", remaining[0].strip(), re.IGNORECASE):
        result_line = remaining.pop(0).strip()
        result = re.sub(r"^(?:【結果】|\[RESULT\])\s*", "", result_line, flags=re.IGNORECASE).strip()
        # 未公開を表す値は「結果なし」へ正規化する（COなし/保留との組合せも受理する）。
        result = normalize_result_field(result)
    ignored_trailing_text = any(line.strip() for line in remaining)
    if claim in {"none", "hold"} and result:
        result = ""
        ignored_trailing_text = True
    return DiscussionClaim(
        claim=claim,
        result=result,
        ignored_trailing_text=ignored_trailing_text,
    )
