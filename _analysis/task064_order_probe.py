"""TASK-064: `[ROLE_ACTION_BOUNDARY]` を含めた節順をコストモデルで探索するプローブ。

実機LLMは使わない。代表字数（`REPRESENTATIVE_SECTION_CHARS`）に対して、単一ブロック移動と
2ブロック入替の貪欲適用（決定的）で最良候補を求める。凍結済みの v5 節順がこの結果と一致する
ことを `tests/test_task064_role_action_boundary.py` で固定する。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf.prompt_layout import (  # noqa: E402
    PROMPT_LAYOUT_V4_ORDER,
    REPRESENTATIVE_SECTION_CHARS,
    ROLE_BOUNDARY_SECTION,
    score_order,
    search_prompt_orders_v5,
    v3_layout_score,
    v4_layout_score,
    v5_layout_score,
)


def main() -> int:
    chars = dict(REPRESENTATIVE_SECTION_CHARS)
    print("representative total chars:", sum(chars.values()))
    ranked = search_prompt_orders_v5(chars, top_k=8)
    for position, candidate in enumerate(ranked, 1):
        order = tuple(candidate["layout_order"])
        print(
            f"{position}. score={candidate['layout_score']:.2f} "
            f"violations={len(candidate['hard_violations'])} "
            f"rbound_at={order.index(ROLE_BOUNDARY_SECTION) if ROLE_BOUNDARY_SECTION in order else -1}"
            f"/{len(order)} zones={candidate['simulated_zone_chars']}"
        )
        print("   ", " ".join(order))
        print(
            "    gaps:",
            {
                key.replace("continuity_checkpoint", "STATE").replace(
                    "role_action_boundary", "RBOUND"
                ): value
                for key, value in candidate["simulated_adjacency_gap_fraction"].items()
            },
        )
    # 基準（v3: 状態なし／v4: 境界なし）と、v4末尾へ機械的に足した場合との比較
    v3_score = v3_layout_score({k: v for k, v in chars.items() if k not in {"continuity_checkpoint", ROLE_BOUNDARY_SECTION}})
    print("v3 (no state/rbound):", v3_score.score)
    print("v4 baseline (no rbound):", v4_layout_score().score)
    print("v5 frozen:", v5_layout_score().score)
    appended = list(PROMPT_LAYOUT_V4_ORDER)
    appended.insert(appended.index("discussion_guidelines"), ROLE_BOUNDARY_SECTION)
    print("v4 + rbound before guidelines:", score_order(tuple(appended), chars).score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
