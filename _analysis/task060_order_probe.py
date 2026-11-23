"""TASK-060: コストモデルでの節順探索を、代表字数で確認する簡易プローブ（LL不使用）。"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.prompt_layout import (  # noqa: E402
    PROMPT_LAYOUT_V2_ORDER,
    REPRESENTATIVE_SECTION_CHARS,
    adopted_layout_score,
    score_order,
    search_prompt_orders,
)


def main() -> None:
    results = search_prompt_orders(REPRESENTATIVE_SECTION_CHARS, top_k=8)
    for index, item in enumerate(results):
        print(
            f"[{index}] score={item['layout_score']} eff={item['attention_efficiency']} "
            f"adj={item['adjacency_penalty']} dil={item['post_format_dilution_penalty']} "
            f"wm={item['weak_middle_must_use_penalty']} clutter={item['primacy_clutter_penalty']}"
        )
        print("     violations:", item["hard_violations"])
        print("     zones:", item["simulated_zone_chars"])
        print("     order:", ", ".join(item["layout_order"]))
    frozen = adopted_layout_score()
    v2_like = score_order(PROMPT_LAYOUT_V2_ORDER, REPRESENTATIVE_SECTION_CHARS)
    print("frozen v3 :", frozen.score, frozen.violations, frozen.zone_chars)
    print("v2 order  :", v2_like.score, v2_like.violations, v2_like.zone_chars)


if __name__ == "__main__":
    main()
