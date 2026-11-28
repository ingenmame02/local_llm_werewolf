import json
import re
from enum import Enum


class ActionKind(str, Enum):
    ATTACK = "attack"
    SEER = "seer"
    MEDIUM = "medium"
    VOTE = "vote"


_LABELS = {
    ActionKind.ATTACK: ("Attack target", "襲撃対象", "Attack Target", "Attack"),
    ActionKind.SEER: ("Seer target", "占い対象", "Seer Target", "Divine target", "Divination target"),
    ActionKind.MEDIUM: ("Medium target", "霊媒対象", "Medium Target"),
    ActionKind.VOTE: ("Vote", "投票", "Vote target", "Vote Target"),
}
_TRAILING_PUNCTUATION = re.compile(r"[\s。．.!！?？,，;；:：]+$")
_OUTER_QUOTES = " \t\r\n「」『』\"'`【】[]（）()"

# TASK-043: 弱いモデルは `Player 1` を「プレイヤー1」「P1」等で書くため、候補名へ別名解決する。
# 別名解決がないと invalid_target → 再試行 → ランダムフォールバックの経路が残る。
_PLAYER_ALIAS = re.compile(r"(?:player\s*|プレイヤー\s*|p\s*)(\d+)", re.IGNORECASE)


class ActionParser:
    @staticmethod
    def _player_mentions(text: str, candidates: tuple[str, ...]) -> list[str]:
        """応答文本から候補名（別名含む）を出現順に列挙する。"""
        if not text or not candidates:
            return []
        found: list[tuple[int, str]] = []
        for name in sorted(candidates, key=len, reverse=True):
            for match in re.finditer(rf"{re.escape(name)}(?:さん)?", text):
                found.append((match.start(), name))
        for match in _PLAYER_ALIAS.finditer(text):
            number = match.group(1)
            for name in candidates:
                alias = _PLAYER_ALIAS.fullmatch(name.strip())
                if alias is not None and alias.group(1) == number:
                    found.append((match.start(), name))
        found.sort()
        ordered: list[str] = []
        for _, name in found:
            if name not in ordered:
                ordered.append(name)
        return ordered

    @staticmethod
    def resolve_target(raw: str | None, candidates: tuple[str, ...]) -> str | None:
        """対象表記を候補名へ解決する（`プレイヤー3` / `P3` / `Player 3さん` 等を含む）。"""
        if raw is None or not isinstance(raw, str) or not candidates:
            return None
        text = _TRAILING_PUNCTUATION.sub("", raw).strip(_OUTER_QUOTES + "*_")
        if not text:
            return None
        if text in candidates:
            return text
        mentions = ActionParser._player_mentions(text, candidates)
        return mentions[0] if mentions else None

    @staticmethod
    def parse(
        response: str | None,
        action: ActionKind,
        candidates: tuple[str, ...] | None = None,
    ) -> str | None:
        """対象行を解析する。`candidates`を指定すると、別名解決と自然文からの抽出も行う。"""
        if response is None or not isinstance(response, str) or not response:
            return None
        stripped_json = response.strip()
        if stripped_json.startswith("{") and stripped_json.endswith("}"):
            try:
                parsed_json = json.loads(stripped_json)
            except ValueError:
                parsed_json = None
            if isinstance(parsed_json, dict) and isinstance(parsed_json.get("target"), str):
                raw_target = parsed_json["target"].strip()
                if candidates:
                    resolved = ActionParser.resolve_target(raw_target, candidates)
                    if resolved is not None:
                        return resolved
                elif raw_target:
                    return raw_target
        labels = _LABELS[action]
        pattern = "|".join(re.escape(label) for label in labels)
        match = re.search(
            rf"(?m)^\s*(?:\*{{1,3}}|_{{1,3}})?\s*(?:{pattern})\s*(?:\*{{1,3}}|_{{1,3}})?\s*[:：]\s*(.*?)\s*$",
            response,
            re.IGNORECASE,
        )
        if match is None and candidates:
            # TASK-043: 「投票: Player 1」形式の行がない応答も、最初の行の唯一の候補名や
            # 応答全体の唯一の候補名から対象を読み取り、再試行・フォールバックを減らす。
            stripped = response.strip()
            first_line = stripped.splitlines()[0] if stripped else ""
            first_line_mentions = ActionParser._player_mentions(first_line, candidates)
            if len(first_line_mentions) == 1:
                return first_line_mentions[0]
            whole_mentions = ActionParser._player_mentions(stripped, candidates)
            if len(whole_mentions) == 1:
                return whole_mentions[0]
            return None
        if match is None:
            return None
        target = _TRAILING_PUNCTUATION.sub("", match.group(1)).strip(_OUTER_QUOTES + "*_")
        if candidates:
            return ActionParser.resolve_target(target, candidates)
        return target or None

    @staticmethod
    def validate_target(
        target: str | None,
        candidates: tuple[str, ...],
        actor: str | None = None,
    ) -> bool:
        return target in candidates and (actor is None or target != actor)
