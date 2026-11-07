import re
from dataclasses import dataclass

CO_MARKER = "【CO】"
RESULT_MARKER = "【結果】"
BODY_MARKER = "【本文】"


@dataclass(frozen=True)
class DiscussionOutput:
    claim: str
    body: str
    result: str = ""


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
