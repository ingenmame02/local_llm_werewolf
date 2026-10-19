import re
from dataclasses import dataclass

CO_MARKER = "【CO】"
BODY_MARKER = "【本文】"


@dataclass(frozen=True)
class DiscussionOutput:
    claim: str
    body: str


def has_discussion_output_markers(content: str) -> bool:
    return CO_MARKER in content or BODY_MARKER in content


def parse_discussion_output(content: str) -> DiscussionOutput | None:
    if not has_discussion_output_markers(content):
        return None
    if content.count(CO_MARKER) != 1 or content.count(BODY_MARKER) != 1:
        return None
    match = re.fullmatch(
        r"\s*【CO】\s*(.*?)\s*\r?\n\s*【本文】\s*(.*?)\s*",
        content,
        flags=re.DOTALL,
    )
    if match is None:
        return None
    claim, body = (part.strip() for part in match.groups())
    if not claim or not body:
        return None
    return DiscussionOutput(claim=claim, body=body)
