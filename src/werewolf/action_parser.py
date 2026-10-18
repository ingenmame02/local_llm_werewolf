import re
from enum import Enum


class ActionKind(str, Enum):
    ATTACK = "attack"
    SEER = "seer"
    VOTE = "vote"


_LABELS = {
    ActionKind.ATTACK: "襲撃対象",
    ActionKind.SEER: "占い対象",
    ActionKind.VOTE: "投票",
}
_TRAILING_PUNCTUATION = re.compile(r"[\s。．.!！?？,，;；:：]+$")
_OUTER_QUOTES = " \t\r\n「」『』\"'`【】[]（）()"


class ActionParser:
    @staticmethod
    def parse(response: str, action: ActionKind) -> str | None:
        label = re.escape(_LABELS[action])
        match = re.search(
            rf"(?m)^\s*(?:\*{{1,3}}|_{{1,3}})?\s*{label}\s*(?:\*{{1,3}}|_{{1,3}})?\s*[:：]\s*(.*?)\s*$",
            response,
        )
        if match is None:
            return None
        target = _TRAILING_PUNCTUATION.sub("", match.group(1)).strip(_OUTER_QUOTES + "*_")
        return target or None

    @staticmethod
    def validate_target(
        target: str | None,
        candidates: tuple[str, ...],
        actor: str | None = None,
    ) -> bool:
        return target in candidates and (actor is None or target != actor)
