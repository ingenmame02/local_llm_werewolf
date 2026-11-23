"""TASK-058/060のレイアウト実測（LL不使用）: 組み立て順の崩れを静的に検査する。

simplified の layout v2（TASK-058）は primacy→中部（落としやすい派生節のみ）→pre-recency
（発言証拠・現在質問）→recency（秘匿境界・契約・出力形式）を、v3（TASK-060）はそれに加えて
「主張索引→時点→発言証拠→現在質問→自分の発言と推理メモ→中立的な案内→契約→出力形式」を
意図している。編集で節順がずれていないかを `validate_prompt_layout()` が全構成で拾い、
v3 では出力形式仕様の後ろに非形式節が載らないか（dilution＝形式指定の希釈）と自分の記録と質問の距離も出す。
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.core_compression import PROMPT_LAYOUT_V2_ORDER, validate_prompt_layout
# TASK-060: v3の評価に必要な配置品質の実測（出力形式の希釈と隣接距離）
from werewolf.prompt_layout import adjacency_gap, post_format_dilution
from werewolf.day_timeline import KIND_SPEECH  # noqa: F401  (台帳の形が変わっていないことの確認用)
from werewolf.models import Message, Phase
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game


def _state(seed: int = 7058):
    return create_game(seed=seed, config_path="config/players.json")


def _start_day(state, day: int, turn: int) -> None:
    state.day = day
    state.discussion_turn = turn
    state.phase = Phase.DAY_DISCUSSION
    state.public_timeline.record_day_start(day)
    state.conversation_log.append(Message("system", f"Day {day} begins."))
    state.begin_discussion_turn()


def _say(state, name: str, content: str) -> Message:
    return state.append_discussion_message(Message(name, content))


def _build(state, actor_name: str, **kwargs):
    actor = next(player for player in state.players if player.name == actor_name)
    common = dict(
        alive_players=tuple(state.alive_players),
        conversation_history=tuple(state.conversation_log),
        day=state.day,
        state=state,
        include_claim=True,
        context_strategy="hybrid_retrieval",
    )
    common.update(kwargs)
    return PromptBuilder.build_discussion_prompt(actor, **common)


CONFIGS = (
    # (day, turn, speeches per player, hint mode, evidence budget)
    (1, 1, 0, "strict", None),
    (1, 2, 5, "strict", 4200),
    (2, 1, 6, "soft", 2600),
    (3, 3, 7, "none", 1800),
)


def main() -> None:
    failures = 0
    for day, turn, per_player, hint, budget in CONFIGS:
        state = _state()
        _start_day(state, day, turn)
        for round_index in range(per_player):
            for player in tuple(player.name for player in state.players)[: len(state.players)]:
                _say(
                    state,
                    player,
                    f"{player}の{round_index + 1}回目の発言です。"
                    + ("私は村人占いの結果を報告します。" if round_index == 1 else ""),
                )
            if turn > 1 and round_index + 1 < turn:
                state.discussion_turn = round_index + 2
        ctx = _build(
            state,
            "Player 8",
            question_hint_mode=hint,
            evidence_char_budget=budget,
            prompt_layout_policy="v2",  # TASK-060で既定がv3になったため、この検証はv2に固定
        )
        layout = ctx.diagnostics.get("prompt_layout")
        violations = ctx.diagnostics.get("prompt_layout_violations") or []
        print(f"day{day}T{turn} {hint} budget={budget} layout={layout}")
        if layout != "v2_attention":
            print("  !! unexpected layout")
            failures += 1
        if violations:
            failures += 1
            for item in violations[:6]:
                print(f"  !! {item['name']} after {item['after']} at {item['start']}")
        # anchor層（ルール・現在質問・行動形式・契約）が生きているか
        missing = [
            marker
            for marker in ("[SYSTEM_RULES]", "[CURRENT_TASK]", "[BEHAVIOR_CONTRACT", "[Instructions]")
            if marker not in ctx.user_prompt
        ]
        if missing:
            print(f"  !! missing anchors: {missing}")
            failures += 1
        print(
            f"  chars={len(ctx.user_prompt)} anchor={ctx.diagnostics.get('core_anchor_chars')}"
            f" recent={ctx.diagnostics.get('core_recent_chars')}"
            f" historical={ctx.diagnostics.get('core_historical_chars')}"
            f" withheld={ctx.diagnostics.get('evidence_withheld_body_count')}"
            f" stage={ctx.diagnostics.get('compression_stage')}"
        )
        # legacy側はv1順（validateしない）
        ctx_legacy = _build(
            state, "Player 8", question_hint_mode=hint, evidence_char_budget=budget,
            prompt_contract_policy="legacy", core_compression_policy="off",
        )
        if ctx_legacy.diagnostics.get("prompt_layout") != "legacy_v1":
            print("  !! legacy layout mislabeled")
            failures += 1
        # TASK-060: 既定の v3（節順・帯・隣接距離・出力形式の終端固定）も同じ盤面で検証する
        ctx_v3 = _build(state, "Player 8", question_hint_mode=hint, evidence_char_budget=budget)
        label_v3 = ctx_v3.diagnostics.get("prompt_layout")
        violations_v3 = ctx_v3.diagnostics.get("prompt_layout_violations") or []
        dilution_v2 = post_format_dilution(ctx.user_prompt)
        dilution_v3 = post_format_dilution(ctx_v3.user_prompt)
        gap_own_v3 = adjacency_gap(ctx_v3.user_prompt, "question_block", "own_speeches")
        gap_own_v2 = adjacency_gap(ctx.user_prompt, "question_block", "own_speeches")
        print(
            f"  v3 layout={label_v3} chars={len(ctx_v3.user_prompt)}"
            f" violations={len(violations_v3)} dilution={dilution_v2}->{dilution_v3}"
            f" own_question_gap={gap_own_v2:.4f}->{gap_own_v3:.4f}"
            f" dup={ctx.diagnostics.get('prompt_duplicate_constraint_count')}"
            f"->{ctx_v3.diagnostics.get('prompt_duplicate_constraint_count')}"
        )
        if label_v3 != "v3_grounding" or violations_v3:
            failures += 1
            for item in violations_v3[:6]:
                print(f"  !! v3 {item}")
    print("[DONE]", "LAYOUT-OK" if failures == 0 else f"LAYOUT-FAIL({failures})")


if __name__ == "__main__":
    main()
