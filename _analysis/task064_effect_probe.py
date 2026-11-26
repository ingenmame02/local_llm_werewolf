"""TASK-064 の効果をLLMなしで測るプローブ（実機テストの代わりに設定・検出の数値確認）。

出すもの:
1. v5(既定) と v4(A/B基準) の送信プロンプト字数差・境界節の字数・隣接距離（複数シード）
2. game_bcd7c97e7e48 型の発言文での能力行使検出（断定/婉曲/三人称の内訳と期待値照合）
3. CO補正プロンプトの短縮倍率（同じ盤面の通常議論プロンプト比）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf.models import Message, Phase, PublicRoleClaim, Role  # noqa: E402
from werewolf.prompt_contract import find_sections  # noqa: E402
from werewolf.prompt_layout import PROMPT_LAYOUT_V5, ROLE_BOUNDARY_SECTION  # noqa: E402
from werewolf.prompts import PromptBuilder  # noqa: E402
from werewolf.role_action_provenance import (  # noqa: E402
    KIND_DIVINATION,
    detect_ability_expressions,
)
from werewolf.setup import create_game  # noqa: E402

LONG_SPEECH = (
    "{name}の{index}番目の発言です。{rival}の占い師COと投票の動きを比べると、"
    "昨夜の襲撃先との整合が怪しく見えるので、今日の投票ではその点を確認したいと考えています。"
)

# game_bcd7c97e7e48 で実際に公開された言い回しの再現（断定=Trueとして数えるべき文）
CORPUS = (
    ("私が占った結果、Player 1は狼だった", True),
    ("対象=Player 1 Result=Werewolf です（私が占った）", True),
    ("私が先に占ったのは Player 4 で、人間でした", True),
    ("あなたが占った結果を教えてもらえますか？", False),
    ("Player 3が「私が占った」と言っていましたね", False),
    ("もし私が占い師なら Player 2 を占ったはず", False),
    ("私は占っていませんし、COも考えていません", False),
    ("Player 5が占ったらしいという噂が流れている", False),
    ("I divined Player 2 and the result was werewolf", True),
    ("Did you divine anyone last night?", False),
)


def _state(seed: int):
    state = create_game(
        seed=seed, config_path="config/players.json", response_language="日本語"
    )
    names = [p.name for p in state.players]
    state.public_timeline.record_day_start(state.day)
    state.conversation_log.append(Message("system", f"Day {state.day} begins."))
    state.phase = Phase.DAY_DISCUSSION
    state.discussion_turn = 3
    state.begin_discussion_turn()
    for index, name in enumerate(names[:4]):
        state.append_discussion_message(
            Message(
                name,
                LONG_SPEECH.format(
                    name=name, index=index + 1, rival=names[(index + 1) % 8]
                ),
            )
        )
    actor = names[4]
    asker = names[3]
    state.append_discussion_message(
        Message(
            asker,
            f"{actor}さんに聞きます。{asker}が先に述べた投票案と比べて、"
            f"{actor}はどう考えますか？根拠を具体的に教えてください。",
        )
    )
    return state, actor, names


def build(state, actor, layout):
    player = next(p for p in state.players if p.name == actor)
    return PromptBuilder.build_discussion_prompt(
        player,
        tuple(state.alive_players),
        tuple(state.conversation_log),
        day=state.day,
        state=state,
        include_claim=True,
        context_strategy="hybrid_retrieval",
        question_hint_mode="strict",
        prompt_contract_policy="simplified",
        core_compression_policy="layered",
        prompt_layout_policy=layout,
    )


def main() -> int:
    print("=== 1. 配置コスト（v5 / v4・同じ盤面） ===")
    totals = {"v5": 0, "v4": 0}
    boundary_sizes: list[int] = []
    gaps_q: list[float] = []
    gaps_own: list[float] = []
    violations = 0
    for seed in (1060, 2064, 3064, 4064, 5064):
        state, actor, _names = _state(seed)
        v5 = build(state, actor, "v5")
        v4 = build(state, actor, "v4")
        spans = find_sections(v5.user_prompt)
        start, end = spans[ROLE_BOUNDARY_SECTION]
        boundary_sizes.append(end - start)
        diag = v5.diagnostics or {}
        assert diag["prompt_layout"] == PROMPT_LAYOUT_V5, diag["prompt_layout"]
        violations += len(diag["prompt_layout_violations"])
        gaps_q.append(float(diag["prompt_gap_role_boundary_to_question"]))
        gaps_own.append(float(diag["prompt_gap_role_boundary_to_own_speeches"]))
        totals["v5"] += len(v5.user_prompt)
        totals["v4"] += len(v4.user_prompt)

        # 占い実行＋本人COがあるケース（最大行ぶん）でも増え方を測る
        seer = next(p for p in state.players if p.name == actor)
        seer.role = Role.SEER
        state.role_action_provenance.record_engine_action(actor, KIND_DIVINATION, 1)
        state.public_role_claims.append(
            PublicRoleClaim(day=1, claimant=actor, claimed_role="seer")
        )
        full_spans = find_sections(build(state, actor, "v5").user_prompt)[
            ROLE_BOUNDARY_SECTION
        ]
        boundary_sizes.append(full_spans[1] - full_spans[0])

    seeds = 5
    print(
        f"平均プロンプト字数 v5={totals['v5'] / seeds:.0f} v4={totals['v4'] / seeds:.0f} "
        f"差={(totals['v5'] - totals['v4']) / seeds:+.0f}字 / 違反={violations}件"
    )
    print(
        f"境界節の字数: min={min(boundary_sizes)} max={max(boundary_sizes)} "
        f"平均={sum(boundary_sizes) / len(boundary_sizes):.0f}"
    )
    print(
        f"境界→現在質問 の距離率 平均={sum(gaps_q) / len(gaps_q):.4f} / "
        f"境界→自分の発言 平均={sum(gaps_own) / len(gaps_own):.4f}"
    )

    print()
    print("=== 2. 能力行使の検出内訳（game_bcd7c97e7e48 型の言い回し） ===")
    names = tuple(f"Player {i}" for i in range(1, 9))
    assertions = hedged = third = misses = 0
    for text, expect_assertion in CORPUS:
        found = detect_ability_expressions(
            text, actor_name="Player 4", known_player_names=names
        )
        strong = [item for item in found if item["first_person"] and not item["hedged"]]
        if strong:
            assertions += 1
        if any(item["hedged"] for item in found):
            hedged += 1
        if any(not item["first_person"] for item in found):
            third += 1
        ok = bool(strong) == expect_assertion
        if not ok:
            misses += 1
        print(
            f"  [{'assert' if strong else ('third' if found else 'none ')}] "
            f"{'OK ' if ok else 'NG!'} {text[:56]}"
        )

    print()
    print("=== 3. CO補正プロンプトの短縮倍率（旧プロンプト再送との比較） ===")
    from werewolf.llm_actions import OllamaActions

    class _NoClient:
        def chat(self, *args, **kwargs):
            raise AssertionError("補正検証でLLMを呼ばない")

    ratios = []
    for seed in (1060, 2064, 3064):
        state, actor, _names = _state(seed)
        player = next(p for p in state.players if p.name == actor)
        player.role = Role.MADMAN
        events: list[dict[str, object]] = []
        actions = OllamaActions(
            state, _NoClient(), seed=state.seed, event_callback=events.append
        )
        correction = actions._seer_co_correction_prompt(player, None)
        normal = build(state, actor, "v5")
        ratios.append(len(correction) / len(normal.user_prompt))
        print(
            f"  seed={seed}: 補正={len(correction)}字 / 通常={len(normal.user_prompt)}字"
            f" = {ratios[-1]:.3f}"
        )
    print(f"平均倍率={sum(ratios) / len(ratios):.3f}（1.0未満＝長い旧プロンプトより短い）")

    print()
    expected_asserts = sum(1 for _, expect in CORPUS if expect)
    print(
        f"検出内訳: 断定={assertions}件（期待 {expected_asserts}件）/ 誤検出={misses}件 / "
        f"婉曲を含む文={hedged}件 / 三人称を拾った文={third}件"
    )
    return 1 if misses else 0


if __name__ == "__main__":
    raise SystemExit(main())

