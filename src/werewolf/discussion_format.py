import re
from dataclasses import dataclass
from typing import Literal

from .models import Message

CO_MARKER = "【CO】"
RESULT_MARKER = "【結果】"
BODY_MARKER = "【本文】"

ValidationSeverity = Literal["fatal", "repairable", "warning", "ok"]

SEVERITY_FATAL: ValidationSeverity = "fatal"
SEVERITY_REPAIRABLE: ValidationSeverity = "repairable"
SEVERITY_WARNING: ValidationSeverity = "warning"
SEVERITY_OK: ValidationSeverity = "ok"

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])\s*|\n+")


@dataclass(frozen=True)
class DiscussionOutput:
    claim: str
    body: str
    result: str = ""


@dataclass(frozen=True)
class DiscussionBodyDiagnosis:
    """議論本文の違反診断結果。秘匿情報は含まず、診断メタ情報のみを持つ。"""

    severity: ValidationSeverity
    reasons: tuple[str, ...] = ()
    fatal: bool = False
    repairable: bool = False
    warning: bool = False
    publishable: bool = False


def extract_discussion_body(content: str) -> str | None:
    """議論本文を自由形式で抽出する。

    - 思考タグ（`<think>`…`</think>`）を除去する
    - `【CO】` / `【結果】` 行は役職主張として分離し、本文から除外する
    - `【本文】` マーカーがあればその中身を優先する
    - JSON・見出し・箇条書きがなくても自然文をそのまま返す
    - 空本文は None を返す（「パス」は有効な発言としてそのまま返す）
    """
    if not content:
        return None
    text = _THINK_BLOCK.sub("", content)
    text = _UNCLOSED_THINK_BLOCK.sub("", text)
    body_marker_index = text.find(BODY_MARKER)
    if body_marker_index >= 0:
        text = text[body_marker_index + len(BODY_MARKER):]
    kept_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not line.strip().startswith((CO_MARKER, RESULT_MARKER))
    ]
    body = " ".join(kept_lines).strip()
    if not body:
        return None
    return body


def is_pass_body(body: str | None) -> bool:
    return body is not None and body.strip() == "パス"


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def find_repeated_topic(
    body: str, transcript: tuple[Message, ...] = ()
) -> bool:
    """現在ラウンドの発言と本文が実質的に重複するかを軽量に判定する。"""
    if not body or not transcript:
        return False
    body_sentences = split_sentences(body)
    if len(body_sentences) < 2:
        return False
    history = "\n".join(
        message.content
        for message in transcript
        if message.content and not is_pass_body(message.content)
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


def has_discussion_output_markers(content: str) -> bool:
    return CO_MARKER in content or RESULT_MARKER in content or BODY_MARKER in content


def parse_discussion_output(content: str) -> DiscussionOutput | None:
    """`【CO】`（+任意の`【結果】`）+`【本文】`の構造化応答を解析する。

    - `【CO】claim` + `【本文】body`（従来の2ブロック）
    - `【CO】claim` + `【結果】対象/判定` + `【本文】body`（結果付き3ブロック）
    - `【CO】claim` + `【結果】対象/判定`（最小形式。本文は任意）
    """
    if not has_discussion_output_markers(content):
        return None
    if (
        content.count(CO_MARKER) != 1
        or content.count(BODY_MARKER) > 1
        or content.count(RESULT_MARKER) > 1
    ):
        return None
    has_result = RESULT_MARKER in content
    has_body = BODY_MARKER in content
    if not has_result and not has_body:
        return None
    match = re.fullmatch(
        r"\s*【CO】\s*(?P<claim>.*?)\s*"
        r"(?:\r?\n\s*【結果】\s*(?P<result>.*?)\s*)?"
        r"(?:\r?\n\s*【本文】\s*(?P<body>.*?)\s*)?",
        content,
        flags=re.DOTALL,
    )
    if match is None:
        return None
    claim = (match.group("claim") or "").strip()
    result = (match.group("result") or "").strip()
    body = (match.group("body") or "").strip()
    if not claim:
        return None
    if has_result and not result:
        return None
    if has_body and not body:
        return None
    return DiscussionOutput(claim=claim, body=body, result=result)


def parse_discussion_claim(content: str) -> DiscussionClaim | None:
    """Parse a CO decision and safely ignore prose outside its structured lines."""
    if content.count(CO_MARKER) != 1 or content.count(RESULT_MARKER) > 1:
        return None
    lines = content.strip().splitlines()
    if not lines:
        return None
    claim_match = re.fullmatch(r"【CO】\s*(?P<claim>.+?)\s*", lines[0].strip())
    if claim_match is None:
        return None
    claim = claim_match.group("claim").strip()
    claim = {
        "COなし": "なし",
        "CO無し": "なし",
        "COしない": "なし",
        "CO保留": "保留",
    }.get(claim, claim)
    remaining = lines[1:]
    result = ""
    if remaining and remaining[0].strip().startswith(RESULT_MARKER):
        result_line = remaining.pop(0).strip()
        result = result_line[len(RESULT_MARKER) :].strip()
        if not result:
            return None
    ignored_trailing_text = any(line.strip() for line in remaining)
    if claim in {"なし", "保留"} and result:
        result = ""
        ignored_trailing_text = True
    return DiscussionClaim(
        claim=claim,
        result=result,
        ignored_trailing_text=ignored_trailing_text,
    )
