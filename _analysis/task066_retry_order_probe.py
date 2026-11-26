"""TASK-066: 再生成（短縮）プロンプトの節順を、実測字数で最適化するプローブ。

`_short_discussion_retry_prompt` が実際に作るブロック本文を捕捉して字数を出し、その実測値で
再生成順序を貪欲探索する（LL不使用・決定的）。狙いは「内容は同じまま、載せ順だけ」を変えること。
いま使う情報（現在質問・直近発言・修正指示・出力形式）が遠くに離れない順を探し、誘導系
（未検証のセオリー行）が先頭帯を占めないことも制約として見る。

実行方法: python _analysis/task066_retry_order_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from werewolf import llm_actions as L  # noqa: E402
from werewolf.llm_actions import OllamaActions  # noqa: E402
from werewolf.models import Message, PublicRoleClaim, Role  # noqa: E402
from werewolf.retry_context_policy import (  # noqa: E402
    RETRY_PROMPT_BLOCKS,
    RETRY_PROMPT_ORDER_LEGACY,
    RETRY_PROMPT_ORDER_V1,
    legacy_retry_order_score,
    retry_adopted_order_score,
    search_retry_orders,
    validate_retry_order,
)
from werewolf.setup import create_game  # noqa: E402

SENTENCE = (
    "占い結果と投票の履歴を突き合わせると、初日に出た主張の整合性が怪しく見えるので、"
    "まずは本人に理由を聞いてから投票先を決めたい。"
)


class _StubClient:
    """送信しないモック（プローブはプロンプト組み立てだけを行う）。"""

    timeout = 120.0
    max_retries = 0
    timeout_retries = 0

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {}


def _scenario(*, day: int, turn: int, checkpoint: bool, claims: bool, votes: bool):
    state = create_game(
        seed=7066, config_path="config/players.json", response_language="日本語"
    )
    actor = next(player for player in state.players if player.role is Role.SEER)
    state.day = day
    state.discussion_turn = turn
    names = [player.name for player in state.players if player.is_alive]
    others = [name for name in names if name != actor.name]
    for index in range(4 + day * 2):
        speaker = others[index % len(others)]
        state.conversation_log.append(
            Message(
                speaker=speaker,
                content=f"{SENTENCE}（論点{index + 1}）",
                day=day,
                turn=(index % 3) + 1,
            )
        )
    if checkpoint:
        text = (
            "[CONTINUITY_CHECKPOINT]\n"
            "- Confirmed: a Seer CO exists on Day 1 [claim:1]; result revealed [speech:9]\n"
            "- Open question to you: why that vote? [speech:12]\n"
            "- Ledger: vote Day1 A->B; death Day2 C"
        )
        state.continuity_checkpoint_text = lambda: text  # type: ignore[assignment]
    # 本人への未回答質問と直近反証は、実ログで必ず再生成の鍵になる（probeで落ちないようにする）
    if others:
        state.conversation_log.append(
            Message(
                speaker=others[0],
                content=(
                    f"{actor.name}へ質問です。なぜその投票先を選んだのですか？"
                    "理由と根拠を教えてほしいです。"
                ),
                day=day,
                turn=turn,
            )
        )
        state.conversation_log.append(
            Message(
                speaker=others[1 % len(others)],
                content=(
                    "それは違うと思います。反論として、初日の占いCOは死亡時刻と矛盾しています。"
                ),
                day=day,
                turn=turn,
            )
        )
    if claims and len(others) > 1:
        state.public_role_claims = [
            PublicRoleClaim(
                day=1,
                claimant=others[0],
                claimed_role=Role.SEER.value,
                target=others[1],
                result="wolf",
            ),
            PublicRoleClaim(
                day=2, claimant=others[1], claimed_role=Role.VILLAGER.value
            ),
        ]
    if votes and len(names) > 2 and len(others) > 3:
        state.vote_history = [
            (1, False, {names[0]: others[0], names[1]: others[0], names[2]: others[1]}),
            (2, True, {names[0]: others[2], names[1]: others[2], names[2]: others[1]}),
        ]
        state.attack_deaths = [(2, others[3])]
        state.executions = [(1, others[0])]
    # 占い師本人限定の結果（再生成プロンプトの非公開判定語）も実測へ含める
    if actor.role is Role.SEER and len(others) > 2:
        state.seer_results = {others[2]: "wolf"}
        state.seer_result_history = [(day, others[2], "wolf")]
    actions = OllamaActions(state, _StubClient(), seed=7066)
    return state, actor, actions


def capture_blocks(actions, actor, *, body_only: bool) -> dict[str, str]:
    """`assemble_retry_prompt` を差し替え、実際に渡されたブロック本文を捕まえる。"""
    original = L.assemble_retry_prompt
    captured: dict[str, dict[str, str]] = {}

    def spy(blocks, order=None):
        captured["blocks"] = {str(key): str(value) for key, value in blocks.items()}
        return original(blocks, order)

    L.assemble_retry_prompt = spy  # type: ignore[assignment]
    try:
        text = actions._short_discussion_retry_prompt(
            actor,
            "discussion_output_format",
            tuple(actions.state.conversation_log),
            body_only=body_only,
        )
    finally:
        L.assemble_retry_prompt = original  # type: ignore[assignment]
    blocks = captured.get("blocks", {})
    assert blocks, "短縮プロンプトのブロックが捕捉できない"
    assert isinstance(text, str) and len(text) > 200, "短縮プロンプトが短すぎる"
    return blocks


def measure() -> dict[str, int]:
    """3シナリオ（発言のみ／checkpointあり／台帳あり）で実測字数の平均を取る。"""
    totals: dict[str, list[int]] = {name: [] for name in RETRY_PROMPT_BLOCKS}
    for kwargs in (
        {"day": 1, "turn": 1, "checkpoint": False, "claims": False, "votes": False},
        {"day": 2, "turn": 2, "checkpoint": True, "claims": True, "votes": False},
        {"day": 3, "turn": 3, "checkpoint": True, "claims": True, "votes": True},
    ):
        _state, actor, actions = _scenario(**kwargs)
        blocks = capture_blocks(actions, actor, body_only=False)
        for name in RETRY_PROMPT_BLOCKS:
            totals[name].append(len(blocks.get(name, "").strip()))
    return {
        name: int(round(sum(values) / float(len(values))))
        for name, values in totals.items()
    }


# 口調・論点・結論・推理順を newly 指示していないかの検査（既存の形式案内は許容）。
BANNED_RETRY_PHRASES = (
    "conclude that",
    "you must believe",
    "argue that",
    "in this reasoning order",
    "polite tone",
    "be concise and agreeable",
)


def freedom_guard(blocks: dict[str, str]) -> list[str]:
    lowered = "\n".join(blocks.values()).lower()
    return [phrase for phrase in BANNED_RETRY_PHRASES if phrase in lowered]


def main() -> None:
    chars = measure()
    print("measured block chars (3-scenario average):")
    for name in RETRY_PROMPT_BLOCKS:
        print(f"  {name}={chars.get(name, 0)}")

    legacy_score = legacy_retry_order_score(chars)
    adopted_score = retry_adopted_order_score(chars)
    print("\nlegacy vs adopted V1:")
    print(f"  legacy score={legacy_score.score} order={list(legacy_score.order)}")
    print(f"  adopted score={adopted_score.score} order={list(adopted_score.order)}")

    ranked = search_retry_orders(chars, start=RETRY_PROMPT_ORDER_LEGACY, top_k=5)
    print("\ngreedy search (single-move + swap, LL-free):")
    for index, item in enumerate(ranked, start=1):
        print(f"  #{index} score={item['retry_layout_score']}")
        print(f"      order={item['retry_layout_order']}")
        print(
            "      adjacency="
            f"{item['retry_adjacency_penalty']} weak_middle="
            f"{item['retry_weak_middle_must_use_penalty']} primacy_clutter="
            f"{item['retry_primacy_clutter_penalty']} post_format="
            f"{item['retry_post_format_dilution_penalty']} efficiency="
            f"{item['retry_attention_efficiency']}"
        )

    best_order = tuple(ranked[0]["retry_layout_order"]) if ranked else ()
    print("\nhard-constraint check:")
    for label, order in (
        ("legacy", RETRY_PROMPT_ORDER_LEGACY),
        ("adopted V1", RETRY_PROMPT_ORDER_V1),
        ("search best", best_order),
    ):
        print(f"  {label}: violations={validate_retry_order(order)}")

    _state, actor, actions = _scenario(
        day=2, turn=2, checkpoint=True, claims=True, votes=True
    )
    problems = freedom_guard(capture_blocks(actions, actor, body_only=False))
    print(f"\nfreedom guard (口調・論点・結論の強制文言): {problems or 'OK'}")
    assert not validate_retry_order(RETRY_PROMPT_ORDER_V1)
    assert not validate_retry_order(RETRY_PROMPT_ORDER_LEGACY)


if __name__ == "__main__":
    main()
