"""TASK-064: 役職騙りと能力行使の境界整理（RoleActionProvenance）。

役職に関する情報を4つの出所へ分離する。

- ``engine_action``: エンジンが実際に実行した能力行動（占い・襲撃決定）。保有者にだけ
  見える非公開結果そのものはここに載せず、公開台帳・公開プロンプトへは絶対に流さない。
- ``public_role_claim``: 本人が「そう名乗った」「そう申告した」と公開した行。
  ブラフもこの分類で受理し、本文の言い換え・削除はしない。
- ``unsupported_role_action``: 実行記録がないのに能力を行使したかのように発言した表現。
  **観測・分類専用**であり、これを使って本文を再生成したり無害化したりはしない
  （設計制約「役職騙り・疑問形・仮定形・引用を自動修正しない」を守るための境界）。
- ``inference``: 他者の能力や役職についての推測（ Player X が占ったはず、等の一人称ではない言及）。

TASK-055の割り引き（エンジン照合の真偽は管理者ログ専用。公開面では全CO行を同じ中立
ラベル ``public_claim_line`` に揃える）をそのまま引き継ぐ。このモジュールが公開面へ
出すのは個数と時点だけで、どのプレイヤーの申告が本物かは管理者ビュー専用のままとする。

4つの設計思想との対応:
1. コンテキスト節約: 本文を保存せず、時点・ID・カテゴリの短い観測行だけを持つ。
   プロンプト側の境界節（prompts.py）はエンジン結果・公開主張・推測の本文を再掲しない。
2. モデルの一貫性: COの初出と能力表現の初出をプレイヤー別に保持し、距離（ターン差）を
   記録する。「自分が実際にやった行動」と「自分が言った主張」を混同させないための台帳。
3. 合理的な議論: 記号 ``[DONE]`` / ``[CLAIM]`` で事実性と主張を1行で分ける。
4. 議論の自由: 禁止・誘導の命令文は一切持たない。検出しても発言内容もモデルの判断も
   変えない（数えるだけ）。ブラフ可否は既存の ``allow_bluff_role_claim`` のまま。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .role_provenance import PUBLIC_LINE_LABEL

# ---------------------------------------------------------------------------
# 出所カテゴリ（TASK-064実装案1）。公開ログにはカテゴリ個数だけを出し、
# プレイヤー別の真偽内訳は管理者ビュー専用に残す。
# ---------------------------------------------------------------------------
CATEGORY_ENGINE_ACTION = "engine_action"
CATEGORY_PUBLIC_ROLE_CLAIM = "public_role_claim"
CATEGORY_UNSUPPORTED_ROLE_ACTION = "unsupported_role_action"
CATEGORY_INFERENCE = "inference"

ROLE_ACTION_CATEGORIES: tuple[str, ...] = (
    CATEGORY_ENGINE_ACTION,
    CATEGORY_PUBLIC_ROLE_CLAIM,
    CATEGORY_UNSUPPORTED_ROLE_ACTION,
    CATEGORY_INFERENCE,
)

# 能力の種類（表現検出の単位）
KIND_DIVINATION = "divination"
KIND_ATTACK = "attack"

# プロンプト・境界節で使う短い表示記号（権威の区別。本文の言い回しは変えない）
DONE_MARKER = "[DONE]"
CLAIM_MARKER = "[CLAIM]"


# ---------------------------------------------------------------------------
# 能力行使表現の検出（観測専用・本文は変更しない）
#
# 誤検出を増やさないための保守的な線引き（TASK-059系と同じ思想）:
# - 疑問文（「あなたが占った結果は？」）や仮定形（「もし私が占い師なら」）、引用
#   （「『私は占った』という言葉だけが残った」）は断定的な能力行使として数えない。
# - 否定（「私は占っていない」）も断定的な行使として数えない。
# - 三人称（Player X が占った）は推測カテゴリへ寄せる。
# ---------------------------------------------------------------------------

_JP_SELF_MARKERS = r"(?:私|僕|俺|ボク|オレ)"
_JP_NEGATION = r"(?:占(?:って)?(?:い)?ない|襲(?:っ)?ていない|鑑定していない)"
_JP_HEDGE = r"(?:もし|かも|かもしれない|はず|と思う|仮に|たとえ)"
_JP_DIVINATION_ASSERT = (
    rf"(?:{_JP_SELF_MARKERS}[^。！？\n]{{0,28}}?(?:占(?:った|てやり|あげた)))"
    rf"|{_JP_SELF_MARKERS}(?:の)?占い(?:結果|鑑定)"
    r"|(?:占い結果|占った結果)[^。！？\n]{0,20}(?:人狼|狼|人間|役職)"
    rf"|{_JP_SELF_MARKERS}が占(?:い|った)"
)
_JP_ATTACK_ASSERT = (
    rf"(?:{_JP_SELF_MARKERS}|私たち|我々|俺たち)[^。！？\n]{{0,20}}?襲"
    r"(?:撃)?(?:った|て決めた)"
    r"|襲撃対象[^。！？\n]{0,14}(?:決めた|確定)"
)
_EN_NEGATION = r"(?i)(?:did\s*not|didn't|never|no\s+divination|have\s*not\s*divined)"
_EN_HEDGE = (
    r"(?i)(?:if\s+i|might|maybe|possibly|would\s*have|hypothetically|presumably|assume)"
)
_EN_DIVINATION_ASSERT = (
    r"(?i)\b(?:i\s*(?:divined|checked|tested)|my\s*(?:divination|check)\b)"
)
_EN_ATTACK_ASSERT = (
    r"(?i)\b(?:i|we)\s*(?:attacked|killed)\b|\bour\s*attack\s*target\b"
)
# 三人称の能力言及（推測カテゴリ）。名前語は呼び出し側で差し込む。
_THIRD_PERSON_TEMPLATE = (
    r"(?:{name}\s*(?:が|は|とは)?[^。！？\n]{{0,20}}?(?:占(?:った|てる|いて)|鑑定した)|"
    r"{name}\s+(?:divined|checked))"
)
_QUESTION_MARK = ("?", "？", "ですか", "Did you", "Are you")
_QUOTE_BRACKETS = (("\u300c", "\u300d"), ("\u300e", "\u300f"))


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def third_person_matches(sentence: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(sentence) for pattern in patterns)


def first_matched_name(
    sentence: str, names: tuple[str, ...], exclude: str | None
) -> str | None:
    for name in names:
        if name and name != exclude and name in sentence:
            return name
    return None


def detect_ability_expressions(
    text: str,
    *,
    actor_name: str | None = None,
    known_player_names: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    """発言文から役職能力の行使表現だけを列挙する（本文は一切変更しない）。

    各観測は ``{kind, first_person, attributed_to, hedged}`` を返す。断定的な一人称
    行使だけが発生候補になり、疑問・仮定・引用・否定は ``hedged=True`` で区別する
    （TASK-064実装案2「観測専用」、TASK-059系「婉曲表現の誤検出を増やさない」）。
    """
    found: list[dict[str, object]] = []
    if not isinstance(text, str) or not text.strip():
        return found
    sentences = [
        piece.strip()
        for piece in re.split(r"(?<=[。！？!?])|[\r\n]+", text)
        if piece.strip()
    ]
    third_person_patterns = tuple(
        re.compile(_THIRD_PERSON_TEMPLATE.format(name=re.escape(name)))
        for name in known_player_names
        if name and name != actor_name
    )
    for sentence in sentences:
        negated = bool(re.search(_JP_NEGATION, sentence)) or bool(
            re.search(_EN_NEGATION, sentence)
        )
        quoted = any(
            open_bracket in sentence and close_bracket in sentence
            for open_bracket, close_bracket in _QUOTE_BRACKETS
        )
        hedged = (
            negated
            or quoted
            or _contains_any(sentence, _QUESTION_MARK)
            or bool(re.search(_JP_HEDGE, sentence))
            or bool(re.search(_EN_HEDGE, sentence))
        )
        kind: str | None = None
        first_person = False
        attributed_to: str | None = None
        third_person = third_person_matches(sentence, third_person_patterns)
        if re.search(_JP_ATTACK_ASSERT, sentence) or re.search(
            _EN_ATTACK_ASSERT, sentence
        ):
            kind = KIND_ATTACK
            first_person = not third_person
        elif re.search(_JP_DIVINATION_ASSERT, sentence) or re.search(
            _EN_DIVINATION_ASSERT, sentence
        ):
            kind = KIND_DIVINATION
            first_person = not third_person
        elif third_person:
            kind = KIND_DIVINATION
            attributed_to = first_matched_name(
                sentence, known_player_names, actor_name
            )
        if kind is None:
            continue
        found.append(
            {
                "kind": kind,
                "first_person": bool(first_person),
                "attributed_to": attributed_to,
                "hedged": bool(hedged),
            }
        )
    return found


# ---------------------------------------------------------------------------
# データ行
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineAbilityAction:
    """エンジンが実際にそのプレイヤーのために実行した能力行動（管理者ビュー専用詳細）。"""

    player: str
    kind: str  # divination / attack
    day: int

    def to_dict(self) -> dict[str, object]:
        return {"player": self.player, "kind": self.kind, "day": self.day}


@dataclass(frozen=True)
class AbilityObservation:
    """公開発言で観測した能力表現の分類結果。カテゴリは管理者ビュー専用。"""

    player: str
    day: int
    discussion_turn: int
    kind: str
    assertion: bool  # 断定された行使表現か（疑問・仮定・引用・否定はFalse）
    supported: bool  # エンジン実行記録と一致したか（管理者のみが読む）
    category: str
    conflict: bool = False  # 本人の実際のエンジン結果と矛盾する申告だったか

    def public_view(self) -> dict[str, object]:
        """公開イベントへ出してよい形（真偽・カテゴリを示さない中立表記）。"""
        return {
            "player": self.player,
            "day": self.day,
            "discussion_turn": self.discussion_turn,
            "kind": self.kind,
            "assertion": self.assertion,
            "interpretation": PUBLIC_LINE_LABEL,
        }

    def admin_view(self) -> dict[str, object]:
        view = self.public_view()
        view["supported"] = self.supported
        view["category"] = self.category
        view["conflict"] = self.conflict
        return view


_TURNS_PER_DAY = 3  # 昼議論は3ターン（ルールカードと同値。日跨ぎの距離を1軸に畳むため使う）


def turn_position(day: int, discussion_turn: int) -> int:
    return max(0, int(day)) * _TURNS_PER_DAY + max(0, int(discussion_turn))


def turn_delta(earlier: tuple[int, int], later: tuple[int, int]) -> int:
    """2時点（day, turn）のターン距離。日跨ぎでも単調な整数差で比較できる。"""
    return max(0, turn_position(*later) - turn_position(*earlier))


@dataclass
class RoleActionProvenance:
    """役職能力の実行記録と公開主張の境界を持つ台帳（TASK-064実装案1・2・6）。"""

    engine_actions: list[EngineAbilityAction] = field(default_factory=list)
    observations: list[AbilityObservation] = field(default_factory=list)
    # プレイヤー別のCO初出・能力断定表現の初出（距離測定用）
    first_claim_turn: dict[str, tuple[int, int]] = field(default_factory=dict)
    first_ability_turn: dict[str, tuple[int, int]] = field(default_factory=dict)
    # カテゴリ別カウンタ（合計値だけ公開。プレイヤー別の内訳は管理者ビュー専用）
    category_counts: dict[str, int] = field(
        default_factory=lambda: {category: 0 for category in ROLE_ACTION_CATEGORIES}
    )
    unsupported_role_action_count: int = 0
    claim_vs_engine_conflict_count: int = 0
    hedged_mentions_observed: int = 0
    # CO補正などの再要求で新たに増えた能力断定ぶん（観測専用・管理者カウンタ）
    retry_added_ability_expressions: int = 0
    ability_assertions_observed: int = 0
    # 直近の観測で測った「CO初出→能力断定」またはその逆のターン差（None可）
    last_claim_to_ability_delta: int | None = None
    allow_bluff_role_claim: bool = True

    # ------------------------------------------------------------------
    # エンジン実行記録の登録（夜フェーズから呼ぶ）
    # ------------------------------------------------------------------
    def record_engine_action(self, player: str, kind: str, day: int) -> None:
        """エンジンが実際にそのプレイヤー分として実行した行動を登録する（重複統合）。"""
        for item in self.engine_actions:
            if (
                item.player == player
                and item.kind == kind
                and item.day == int(day)
            ):
                return
        self.engine_actions.append(
            EngineAbilityAction(player=player, kind=kind, day=int(day))
        )

    def own_engine_days(self, player: str, kind: str) -> tuple[int, ...]:
        """そのプレイヤーのエンジン実行記録のある日（境界節は個数と日だけ参照する）。"""
        return tuple(
            sorted(
                item.day
                for item in self.engine_actions
                if item.player == player and item.kind == kind
            )
        )

    # ------------------------------------------------------------------
    # 公開CO・能力断定表現の初出記録（距離測定の軸）
    # ------------------------------------------------------------------
    def mark_public_claim(self, player: str, day: int, discussion_turn: int) -> int | None:
        """公開COの初出を記録し、先に能力表現が出ていた場合はターン差を返す（観測のみ）。"""
        if player in self.first_claim_turn:
            return None
        self.first_claim_turn[player] = (int(day), int(discussion_turn))
        self.category_counts[CATEGORY_PUBLIC_ROLE_CLAIM] += 1
        ability = self.first_ability_turn.get(player)
        if ability is not None:
            self.last_claim_to_ability_delta = turn_delta(
                ability, self.first_claim_turn[player]
            )
        return self.last_claim_to_ability_delta

    def mark_ability_assertion(self, player: str, day: int, discussion_turn: int) -> int | None:
        """能力断定表現の初出を記録し、先にCOが出ていた場合はターン差を返す（観測のみ）。"""
        if player in self.first_ability_turn:
            return None
        self.first_ability_turn[player] = (int(day), int(discussion_turn))
        claim = self.first_claim_turn.get(player)
        if claim is not None:
            self.last_claim_to_ability_delta = turn_delta(
                claim, self.first_ability_turn[player]
            )
        return self.last_claim_to_ability_delta

    # ------------------------------------------------------------------
    # 公開発言の観測（本文は変更しない）
    # ------------------------------------------------------------------
    def observe_public_speech(
        self,
        *,
        player: str,
        day: int,
        discussion_turn: int,
        text: str,
        own_engine_results: dict[str, str] | None = None,
        allow_bluff_role_claim: bool = True,
        known_player_names: tuple[str, ...] = (),
    ) -> list[AbilityObservation]:
        """公開された発言文から能力行使表現を観測して分類する。

        ``own_engine_results`` はそのプレイヤーがエンジンから実際に受け取った占い結果
        （本人分のみ。他者分は絶対に渡さない）。一致すれば ``engine_action``、記録が
        なければ ``unsupported_role_action``。どちらも公開面では同じ中立ラベルで、
        判定は管理者ビュー専用に残すだけ（本文にも他の節にも出さない）。
        """
        self.allow_bluff_role_claim = bool(allow_bluff_role_claim)
        observations: list[AbilityObservation] = []
        results = dict(own_engine_results or {})
        for expression in detect_ability_expressions(
            text, actor_name=player, known_player_names=tuple(known_player_names)
        ):
            kind = str(expression.get("kind") or KIND_DIVINATION)
            hedged = bool(expression.get("hedged"))
            first_person = bool(expression.get("first_person"))
            if not first_person:
                # 他者の能力についての言及は推測（inference）。違反には数えない。
                self.category_counts[CATEGORY_INFERENCE] += 1
                continue
            if hedged:
                # 疑問・仮定・引用・否定は断定的な行使として数えない（誤検出防止）。
                self.hedged_mentions_observed += 1
                continue
            self.ability_assertions_observed += 1
            supported = False
            conflict = False
            if kind == KIND_DIVINATION:
                spoken = extract_spoken_verdict(text)
                if spoken is None:
                    supported = bool(results) or bool(
                        self.own_engine_days(player, KIND_DIVINATION)
                    )
                else:
                    target, verdict = spoken
                    actual = results.get(target)
                    if actual is not None and actual == verdict:
                        supported = True
                    elif actual is not None and actual != verdict:
                        conflict = True
                        self.claim_vs_engine_conflict_count += 1
            elif kind == KIND_ATTACK:
                # 襲撃表現は「そのプレイヤーに襲撃の実行記録があるか」だけを見る。
                # 狂人は襲撃担当にならないため、狂人の襲撃行使表現は実行記録なしになる。
                supported = bool(self.own_engine_days(player, KIND_ATTACK))
            category = (
                CATEGORY_ENGINE_ACTION if supported else CATEGORY_UNSUPPORTED_ROLE_ACTION
            )
            self.category_counts[category] += 1
            if not supported:
                self.unsupported_role_action_count += 1
            delta = self.mark_ability_assertion(player, day, discussion_turn)
            observation = AbilityObservation(
                player=player,
                day=int(day),
                discussion_turn=int(discussion_turn),
                kind=kind,
                assertion=True,
                supported=supported,
                category=category,
                conflict=conflict,
            )
            observations.append(observation)
            self.observations.append(observation)
        return observations

    def note_retry_delta(self, before_assertions: int, after_assertions: int) -> int:
        """CO補正などの再要求で能力断定表現がどれだけ増えたかを観測する（管理者カウンタ）。"""
        added = max(0, int(after_assertions) - int(before_assertions))
        self.retry_added_ability_expressions += added
        return added

    # ------------------------------------------------------------------
    # メトリクスとシリアライズ（公開面は中立合計のみ／内訳は管理者ビュー専用）
    # ------------------------------------------------------------------
    def metrics(self) -> dict[str, object]:
        """公開ログへ出してよい中立な合計値だけ返す（プレイヤー別の内訳は含まない）。"""
        return {
            "role_action_provenance_counts": dict(self.category_counts),
            "unsupported_role_action_count": int(self.unsupported_role_action_count),
            "claim_vs_engine_conflict_count": int(self.claim_vs_engine_conflict_count),
            "ability_assertions_observed": int(self.ability_assertions_observed),
            "hedged_mentions_observed": int(self.hedged_mentions_observed),
            "retry_added_ability_expressions": int(self.retry_added_ability_expressions),
            "engine_actions_recorded": len(self.engine_actions),
            "players_with_public_claims": len(self.first_claim_turn),
            "players_with_ability_assertions": len(self.first_ability_turn),
        }

    def admin_metrics(self) -> dict[str, object]:
        """管理者ビュー。カテゴリの内訳・本人エンジン結果との矛盾を保有者に載せる。"""
        admin: dict[str, object] = dict(self.metrics())
        per_player: dict[str, dict[str, object]] = {}
        for player in sorted(set(self.first_claim_turn) | set(self.first_ability_turn)):
            entry: dict[str, object] = {
                "first_claim_turn": list(self.first_claim_turn[player])
                if player in self.first_claim_turn
                else None,
                "first_ability_turn": list(self.first_ability_turn[player])
                if player in self.first_ability_turn
                else None,
            }
            claim = self.first_claim_turn.get(player)
            ability = self.first_ability_turn.get(player)
            if claim is not None and ability is not None:
                entry["claim_to_ability_turn_delta"] = turn_delta(claim, ability)
            per_player[player] = entry
        for observation in self.observations:
            entry = per_player.setdefault(observation.player, {})
            counts = entry.setdefault("category_counts", {})
            assert isinstance(counts, dict)
            counts[observation.category] = int(counts.get(observation.category, 0)) + 1
            if observation.conflict:
                entry["conflicts"] = int(entry.get("conflicts", 0)) + 1
        admin["per_player"] = per_player
        return admin

    def serialize_public(self) -> dict[str, object]:
        """公開ログ用。カテゴリ合計と中立ラベルだけで、保有者・真偽の内訳を含まない。"""
        return {
            "interpretation_label": PUBLIC_LINE_LABEL,
            "done_marker": DONE_MARKER,
            "claim_marker": CLAIM_MARKER,
            "metrics": self.metrics(),
        }

    def serialize_admin(self) -> dict[str, object]:
        """管理者ログ用。エンジン実行記録と観測分類（真偽内訳）を含む。"""
        return {
            "metrics": self.admin_metrics(),
            "engine_actions": [item.to_dict() for item in self.engine_actions],
            "observations": [item.admin_view() for item in self.observations],
        }


# ---------------------------------------------------------------------------
# 申告判定の取り出し（照合用の内部補助。公開文面は作らない）
# ---------------------------------------------------------------------------

_VERDICT_PATTERNS = (
    re.compile(
        r"(?:対象|target)\s*[=:＝]\s*(Player\s?\d+|プレイヤー\d+)"
        r"[^。！？\n]{0,24}?(?:判定|result)\s*[=:＝]\s*"
        r"(人狼|狼|Werewolf|人間|Human)",
    ),
    re.compile(
        r"(Player\s?\d+|プレイヤー\d+)\s*(?:は|が|を)?\s*"
        r"(人狼|狼|Werewolf)"
    ),
    re.compile(
        r"(Player\s?\d+|プレイヤー\d+)\s*(?:は|が|を)?\s*"
        r"(人間|Human)"
    ),
)


def normalize_verdict(value: object) -> str | None:
    lowered = str(value).strip().casefold()
    if lowered in {"人狼", "狼", "werewolf", "wolf"}:
        return "wolf"
    if lowered in {"人間", "human"}:
        return "human"
    return None


def normalize_player_name(value: object) -> str:
    candidate = str(value).strip()
    match = re.search(r"(?:player|プレイヤー)\s*(\d+)", candidate, re.IGNORECASE)
    if match is None:
        return candidate
    return f"Player {int(match.group(1))}"


def extract_spoken_verdict(text: str) -> tuple[str, str] | None:
    """能力表現から「対象=Player X、判定=人狼/人間」を取り出す（照合用の内部補助）。"""
    for pattern in _VERDICT_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        groups = match.groups()
        target = normalize_player_name(groups[0])
        verdict = normalize_verdict(groups[-1])
        if verdict is not None:
            return target, verdict
    return None


__all__ = [
    "AbilityObservation",
    "CATEGORY_ENGINE_ACTION",
    "CATEGORY_INFERENCE",
    "CATEGORY_PUBLIC_ROLE_CLAIM",
    "CATEGORY_UNSUPPORTED_ROLE_ACTION",
    "CLAIM_MARKER",
    "DONE_MARKER",
    "EngineAbilityAction",
    "KIND_ATTACK",
    "KIND_DIVINATION",
    "RoleActionProvenance",
    "ROLE_ACTION_CATEGORIES",
    "detect_ability_expressions",
    "extract_spoken_verdict",
    "normalize_player_name",
    "normalize_verdict",
    "turn_delta",
    "turn_position",
]




