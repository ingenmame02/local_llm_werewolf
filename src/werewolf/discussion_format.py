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
