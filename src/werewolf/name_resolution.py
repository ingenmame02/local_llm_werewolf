"""TASK-065: プレイヤー名の表記ゆれを内部照合の入口で吸収する（公開本文は触らない）。

実機ログ `game_425755ce9a83` では、モデルが `Player\u202f6`（NARROW NO-BREAK SPACE）や
`player6`、全角の `Ｐｌａｙｅｒ６` のような表記で他人を指した。既存の未知名前判定は
`re.findall(r"Player\\s+\\d+")` を実在名集合と**文字列一致**で突き合わせるため、
空白類が一文字違えば「存在しないプレイヤーを呼んだ」と誤判定し、ledger repair が
発生した（台帳・時系列診断・質問回答判定がノイズを被る。本文自体は無傷）。

本モジュールはLLMを呼ばない純粋な正規化だけを持つ（TASK-065実装案4そのまま）:

- 台帳登録・証拠ID解決・質問回答判定の入口だけで正規名へ解決する
- 正規化前の本文は公開ログへそのまま残す（表記の書き換え・再生成は一切しない）
- 存在しない番号は正規化せず、従来どおり未解決参照として観測する

方策（A/B比較用）:
- `normalized`（既定）: 空白類・全角・大小文字・ハイフンに加え `プレイヤーN` / `P N`
  といった別ラベルも実在名へ解決する
- `literal`: 旧来動作（`Player\\s+\\d+` の完全一致のみ）。同じseedで誤検出削減量を測る逃げ道
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 方策
# ---------------------------------------------------------------------------

NAME_RESOLUTION_POLICY_NORMALIZED = "normalized"
NAME_RESOLUTION_POLICY_LITERAL = "literal"
NAME_RESOLUTION_POLICIES: tuple[str, ...] = (
    NAME_RESOLUTION_POLICY_NORMALIZED,
    NAME_RESOLUTION_POLICY_LITERAL,
)
NAME_RESOLUTION_POLICY_DEFAULT = NAME_RESOLUTION_POLICY_NORMALIZED


def normalize_name_resolution_policy(value: object) -> str:
    """設定値を名前正規化ポリシーへ正規化する（未知値は既定の normalized）。"""
    if isinstance(value, str) and value.strip():
        candidate = value.strip().lower()
        if candidate in NAME_RESOLUTION_POLICIES:
            return candidate
    return NAME_RESOLUTION_POLICY_DEFAULT


def name_resolution_policy_label(policy: object) -> str:
    """ログ用ラベル（canonical_reference_lookup / exact_string_reference_lookup）。"""
    if normalize_name_resolution_policy(policy) == NAME_RESOLUTION_POLICY_LITERAL:
        return "exact_string_reference_lookup"
    return "canonical_reference_lookup"


# ---------------------------------------------------------------------------
# 正規化の基本形
# ---------------------------------------------------------------------------

# \s は Unicode の空白類（U+00A0 / U+202F / U+205F / U+3000 等）を拾うが、
# BOM・ゼロ幅文字は入らないため明示する。
_WHITESPACE_CHARS = "\u00a0\u00ad\u200b\u200c\u200d\u2007\u2008\u2009\u200a\u202f\u205f\u3000\ufeff"
_WHITESPACE_RE = re.compile(r"[\s" + _WHITESPACE_CHARS + r"]+")
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def fold_layer_text(text: object) -> str:
    """内部照合専用の折畳み（NFKC → 空白類を半角スペース1つへ → 小文字化）。

    公開本文は変更しない。返り値はあくまで照合用の文字列で、
    ログに残す本文や発言表示には使わない。None・空文字は空へ寄せる。
    """
    if text is None:
        return ""
    folded = unicodedata.normalize("NFKC", str(text)).translate(_FULLWIDTH_DIGITS)
    folded = _WHITESPACE_RE.sub(" ", folded)
    return folded.strip().casefold()


# ---------------------------------------------------------------------------
# 言及スキャン（別ラベル対応）
# ---------------------------------------------------------------------------

# `Player 6` / `player6` / `player-6` / `P 6` / `プレイヤー６` を拾う。
# TASK-065: 番号の直後にさらに数字が続く表記（`player 100`・`P 12`）は別物として扱う。
# 桁数ガードがないと `player 100` が Player 1 の部分文字列として誤解決してしまう。
_REFERENCE_RE = re.compile(
    r"(?:player\s*[-_.]?\s*(\d+)(?![0-9]))"
    r"|(?:プレイヤー\s*[-_.]?\s*(\d+)(?![0-9]))"
    r"|(?<![0-9a-z])p\s*[-_.]?\s*(\d+)(?![0-9])",
    re.IGNORECASE,
)
# 旧来判定が使った形（literal ポリシー用の再現）
_LEGACY_REFERENCE_RE = re.compile(r"Player\s+\d+")


def _name_number(name: object) -> int | None:
    """名前から読み取れる番号1つ（無ければ None。番号なし固有名詞の運用に備える）。"""
    digits = re.findall(r"\d+", fold_layer_text(name))
    return int(digits[0]) if len(digits) == 1 else None


@dataclass(frozen=True)
class NameReference:
    """本文の中の名前言及1件（内部照合用。公開テキストそのものは変換しない）。"""

    token: str  # 折畳み後のマッチ文字列（可読な識別子）
    number: int | None  # 読み取れた番号（読め無ければ None）
    canonical: str | None  # 解決先の正式名（未解決なら None）
    resolved: bool
    exact: bool  # 原文に正式名がそのまま含まれていたか（＝表記ゆれではない）
    label: str  # canonical / whitespace_or_label_variant / katakana / p_short / unknown_number

    def to_dict(self) -> dict[str, object]:
        return {
            "token": self.token,
            "number": self.number,
            "canonical": self.canonical,
            "resolved": self.resolved,
            "exact": self.exact,
            "label": self.label,
        }


def build_name_lookup(player_names: tuple[str, ...]) -> dict[str, str]:
    """折畳みキー（正式名・略称・カタカナ）から正式名へ引く索引を作る。"""
    lookup: dict[str, str] = {}
    for name in player_names:
        if not isinstance(name, str) or not name.strip():
            continue
        folded = fold_layer_text(name)
        lookup[folded] = name
        number = _name_number(name)
        if number is None:
            continue
        for alias in (
            f"player{number}",
            f"プレイヤー{number}",
            f"p{number}",
            f"player_{number}",
            f"player-{number}",
        ):
            lookup.setdefault(alias, name)
    return lookup


def _classify_label(raw: str, canonical: str | None) -> str:
    if canonical is None:
        return "unknown_number"
    compact = raw.replace(" ", "")
    if raw == fold_layer_text(canonical):
        return "canonical"
    if "プレイヤー" in compact:
        return "katakana"
    if re.fullmatch(r"p\d+", compact):
        return "p_short"
    return "whitespace_or_label_variant"


def scan_references(
    text: object,
    player_names: tuple[str, ...],
    *,
    policy: str = NAME_RESOLUTION_POLICY_DEFAULT,
) -> tuple[NameReference, ...]:
    """本文の名前言及を走査し、内部解決結果を並べる（観測専用の純関数）。

    `literal` では旧来判定と同じ `Player\\s+\\d+` の完全一致だけを対象にする（A/B基準）。
    `normalized` では空白類・全角・別ラベルも解決する。存在しない番号は正規化せず、
    ここでは決して正式名へ書き換えない。
    """
    if not isinstance(text, str) or not text:
        return ()
    names = tuple(name for name in player_names if isinstance(name, str) and name.strip())
    if normalize_name_resolution_policy(policy) == NAME_RESOLUTION_POLICY_LITERAL:
        return _scan_literal(text, names)
    lookup = build_name_lookup(names)
    folded = fold_layer_text(text)
    references: list[NameReference] = []
    seen: set[tuple[str, str]] = set()
    for match in _REFERENCE_RE.finditer(folded):
        raw = match.group(0)
        digits = next((group for group in match.groups() if group), "")
        number = int(digits) if digits.isdigit() else None
        canonical = None
        if number is not None:
            canonical = (
                lookup.get(f"player{number}")
                or lookup.get(f"プレイヤー{number}")
                or lookup.get(f"p{number}")
            )
        canonical = canonical or lookup.get(fold_layer_text(raw))
        label = _classify_label(raw, canonical)
        exact = bool(canonical) and canonical in text
        key = (raw, canonical or "")
        if key in seen:
            continue
        seen.add(key)
        references.append(
            NameReference(
                token=raw[:40],
                number=number,
                canonical=canonical,
                resolved=canonical is not None,
                exact=exact,
                label=label,
            )
        )
    # 別ラベル表に無い運用名（固有名詞など）は折畳み一致で拾う。
    # TASK-065: `player 1` が `player 100` の副串として実在名へ化けないよう、
    # 前後が英数字の境界であることを見る（存在しない番号は未解決のまま残す）。
    for name in names:
        folded_name = fold_layer_text(name)
        if not folded_name:
            continue
        if not re.search(
            rf"(?<![0-9a-z]){re.escape(folded_name)}(?![0-9a-z])", folded
        ):
            continue
        key = (folded_name, name)
        if key in seen:
            continue
        seen.add(key)
        references.append(
            NameReference(
                token=folded_name[:40],
                number=_name_number(name),
                canonical=name,
                resolved=True,
                exact=name in text,
                label="canonical" if name in text else "whitespace_or_label_variant",
            )
        )
    return tuple(references)


def _scan_literal(text: str, names: tuple[str, ...]) -> tuple[NameReference, ...]:
    """旧来動作の再現（`Player\\s+\\d+` を実在名集合と完全一致させる）。"""
    known = {str(name) for name in names}
    references: list[NameReference] = []
    seen: set[str] = set()
    for match in _LEGACY_REFERENCE_RE.finditer(text):
        token = match.group(0)
        digits = re.findall(r"\d+", token)
        number = int(digits[0]) if digits else None
        canonical = token if token in known else None
        if token in seen:
            continue
        seen.add(token)
        references.append(
            NameReference(
                token=token[:40],
                number=number,
                canonical=canonical,
                resolved=canonical is not None,
                exact=canonical is not None,
                label="canonical" if canonical is not None else "unknown_number",
            )
        )
    return tuple(references)


# ---------------------------------------------------------------------------
# 公開API（内部照合だけ。本文は返さない）
# ---------------------------------------------------------------------------


def resolve_known_names(
    text: object,
    player_names: tuple[str, ...],
    *,
    policy: str = NAME_RESOLUTION_POLICY_DEFAULT,
) -> tuple[str, ...]:
    """本文から解決できた正式名を並べる（台帳登録・質問回答判定の入口で使う）。"""
    return tuple(
        dict.fromkeys(
            ref.canonical
            for ref in scan_references(text, player_names, policy=policy)
            if ref.resolved and ref.canonical
        )
    )


def unresolved_references(
    text: object,
    player_names: tuple[str, ...],
    *,
    policy: str = NAME_RESOLUTION_POLICY_DEFAULT,
) -> tuple[NameReference, ...]:
    """実在名へ解決できなかった言及だけ（未知名前検出の本体）。"""
    return tuple(
        ref for ref in scan_references(text, player_names, policy=policy) if not ref.resolved
    )


def has_unresolved_reference(
    text: object,
    player_names: tuple[str, ...],
    *,
    policy: str = NAME_RESOLUTION_POLICY_DEFAULT,
) -> bool:
    """存在しないプレイヤーを指しているか（旧 `name not in known_players` の置換先）。"""
    return bool(unresolved_references(text, player_names, policy=policy))


def variant_mention_count(
    text: object,
    player_names: tuple[str, ...],
    *,
    policy: str = NAME_RESOLUTION_POLICY_DEFAULT,
) -> int:
    """正式名の完全一致では取れないが解決できた言及数（吸収した表記ゆれの実測）。"""
    return sum(
        1
        for ref in scan_references(text, player_names, policy=policy)
        if ref.resolved and not ref.exact
    )


def mention_labels(
    text: object,
    player_names: tuple[str, ...],
    *,
    policy: str = NAME_RESOLUTION_POLICY_DEFAULT,
) -> dict[str, int]:
    """言及ラベル別の内訳（canonical / whitespace_or_label_variant / katakana など）。"""
    counts: dict[str, int] = {}
    for ref in scan_references(text, player_names, policy=policy):
        counts[ref.label] = counts.get(ref.label, 0) + 1
    return counts


def names_in_text(
    text: object,
    player_names: tuple[str, ...],
    *,
    policy: str = NAME_RESOLUTION_POLICY_DEFAULT,
) -> tuple[str, ...]:
    """本文から解決できた正式名（旧 `re.findall(r"Player\\s+\\d+")` の後継）。

    表記ゆれで崩れた照合を吸収する。未解決の番号は含まない（実在しないプレイヤーを
    ここで無理に実名へ変換しない）。返り値は内部照合専用で、公開本文は変わらない。
    """
    return resolve_known_names(text, player_names, policy=policy)


def shares_mentioned_players(
    left: object,
    right: object,
    player_names: tuple[str, ...],
    *,
    policy: str = NAME_RESOLUTION_POLICY_DEFAULT,
) -> bool:
    """両方の本文で同じ人物を指しているか（質問者名と回答中の言及の突合）。

    旧判定は文字列一致だったため `Player\\u202f6` と `Player 6` を別物として扱った。
    ここでは内部解決した正式名だけで比べる。公開本文の表記は一切変更しない。
    """
    shared = set(names_in_text(left, player_names, policy=policy)) & set(
        names_in_text(right, player_names, policy=policy)
    )
    return bool(shared)
