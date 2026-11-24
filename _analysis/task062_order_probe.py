"""TASK-062: `[CONTINUITY_CHECKPOINT]` を含めた節順をコストモデルで探索するプローブ。

実機LLMは使わない。代表字数（`REPRESENTATIVE_SECTION_CHARS`）に対して、単一ブロック移動と
2ブロック入替の貪欲適用（決定的）で最良候補を求める。凍結済みの v4 節順がこの結果と一致する
ことを `tests/test_task062_continuity_state.py` で固定する。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf.prompt_layout import (  # noqa: E402
    PROMPT_LAYOUT_V3_ORDER,
    REPRESENTATIVE_SECTION_CHARS,
    flatten_blocks,
    score_order,
    search_prompt_orders,
)


def main() -> int:
    chars = dict(REPRESENTATIVE_SECTION_CHARS)
    print("representative total chars:", sum(chars.values()))
    ranked = search_prompt_orders(chars, top_k=8)
    for position, candidate in enumerate(ranked, 1):
        order = tuple(candidate["layout_order"])
        print(
            f"{position}. score={candidate['layout_score']:.2f} "
            f"violations={len(candidate['hard_violations'])} "
            f"checkpoint_at={order.index('continuity_checkpoint') if 'continuity_checkpoint' in order else -1}"
            f"/{len(order)} zones={candidate['simulated_zone_chars']}"
        )
        print("   ", " ".join(order))
        print(
            "    gaps:",
            {
                key.replace("continuity_checkpoint", "STATE"): value
                for key, value in candidate["simulated_adjacency_gap_fraction"].items()
            },
        )
    # v3（チェックポイント無し）と、v3末尾へ機械的に足した場合との比較
    v3_only = score_order(PROMPT_LAYOUT_V3_ORDER, {k: v for k, v in chars.items() if k != "continuity_checkpoint"})
    print("v3 without checkpoint:", v3_only.score)
    appended = list(PROMPT_LAYOUT_V3_ORDER)
    appended.insert(appended.index("discussion_guidelines"), "continuity_checkpoint")
    print("v3 + checkpoint before guidelines:", score_order(tuple(appended), chars).score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
