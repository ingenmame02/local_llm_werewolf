"""TASK-060の効果実測（LL不使用）: 同じ盤面・同じ発言者で v3 / v2 / legacy の配置を比べる。

prompts.py の実プロンプトを作り、節の位置・帯字数・隣接距離・重複制約数・総字数を表にする。
実機LLMは使わず、配置の違いだけを数値で確認するための補助スクリプト。
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.models import Message, Phase  # noqa: E402
from werewolf.prompt_contract import find_sections  # noqa: E402
from werewolf.prompt_layout import measure_zones, post_format_dilution  # noqa: E402
from werewolf.prompts import PromptBuilder  # noqa: E402
from werewolf.setup import create_game  # noqa: E402

LONG_SPEECH = (
    "{name}の{index}番目の発言です。{rival}の占い師COと投票の動きを比べると、"
    "昨夜の襲撃先との整合が怪しく見えるので、今日の投票ではその点を確認したいと考えています。"
)

MARKERS = (
    "[SYSTEM_RULES]",
    "[OUTPUT_LANGUAGE]",
    "[SELF_ACTION_MEMORY",
    "[FACT_LEDGER]",
    "[Werewolf Theory Card",
    "[ACTOR_PRIVATE_MEMO",
    "[CLAIM_LEDGER]",
    "[DAY_TIMELINE]",
    "[CONVERSATION_EVIDENCE",
    "[CURRENT_TASK]",
    "[Current Question]",
    "[Your Speeches Today",
    "[YOUR_PREVIOUS_REASONING_MEMO",
    "[REASONING_MEMO_GUIDANCE]",
    "[Discussion Guidelines & Logical Reasoning]",
    "[Leak Boundary",
    "[BEHAVIOR_CONTRACT",
    "[Instructions]",
    "[Final Output Checklist]",
)


def _fixture():
    state = create_game(seed=1060, config_path="config/players.json", response_language="日本語")
    names = tuple(player.name for player in state.players)
    actor_name = names[2]
    state.public_timeline.record_day_start(state.day)
    state.conversation_log.append(Message("system", f"Day {state.day} begins."))
    state.phase = Phase.DAY_DISCUSSION
    state.discussion_turn = 4
    state.begin_discussion_turn()
    order = (actor_name, names[0], names[1])
    for index, name in enumerate(order):
        rival = names[(index + 3) % len(names)]
        state.append_discussion_message(
            Message(name, LONG_SPEECH.format(name=name, index=index + 1, rival=rival))
        )
    asker = names[3] if names[3] != actor_name else names[5]
    state.append_discussion_message(
        Message(
            asker,
            f"{actor_name}さんに聞きます。{asker}が先に述べた投票案と比べて、"
            f"{actor_name}はどう考えますか？根拠を具体的に教えてください。",
        )
    )
    state.store_model_reasoning_memo(
        actor_name,
        "占い師COを出した2名が昨夜の襲撃先と矛盾していないか、今日の発言順を確認したい。",
    )
    return state, next(p for p in state.players if p.name == actor_name), asker


def main() -> None:
    for layout in ("v3", "v2", "legacy"):
        contract = "legacy" if layout == "legacy" else "simplified"
        state, actor, asker = _fixture()
        ctx = PromptBuilder.build_discussion_prompt(
            actor,
            tuple(state.alive_players),
            tuple(state.conversation_log),
            day=state.day,
            state=state,
            include_claim=True,
            context_strategy="hybrid_retrieval",
            question_hint_mode="strict",
            prompt_contract_policy=contract,
            core_compression_policy="layered" if contract != "legacy" else "off",
            prompt_layout_policy=layout,
        )
        prompt = ctx.user_prompt
        diag = ctx.diagnostics or {}
        print("=" * 74)
        print(
            f"layout={layout} label={diag.get('prompt_layout')} chars={len(prompt)} "
            f"duplicate_constraints={diag.get('prompt_duplicate_constraint_count')} "
            f"dedup_saved={diag.get('dedup_saved_chars')}"
        )
        zones = measure_zones(prompt)
        print("zones:", zones, "| instructions_to_end:", post_format_dilution(prompt))
        print("gaps q->e:", diag.get("prompt_gap_question_to_evidence"),
              "t->e:", diag.get("prompt_gap_timeline_to_evidence"),
              "own->q:", diag.get("prompt_gap_own_speeches_to_question"))
        print("weak_middle_must_use:", diag.get("prompt_weak_middle_must_use_chars"),
              "| theory_card_chars:", diag.get("prompt_theory_card_chars"))
        print("violations:", diag.get("prompt_layout_violations"))
        positions = []
        for marker in MARKERS:
            index = prompt.find(marker)
            if index >= 0:
                positions.append((round(1000 * index / len(prompt), 1), marker))
        for permille, marker in sorted(positions):
            print(f"   {permille:>5.1f}‰  {marker}")


if __name__ == "__main__":
    main()
