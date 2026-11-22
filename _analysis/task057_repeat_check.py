"""TASK-057: simplified で自分の行動記録などに本文が残る箇所を特定する（実機不使用）。"""
import re
import sys
from pathlib import Path

REPO = Path(r"c:\Users\yukic\local_llm_werewolf")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "_analysis"))

from task057_prompt_ab import SimClient, build, unique_bodies  # noqa: E402
from werewolf.engine import GameEngine  # noqa: E402
from werewolf.llm_actions import OllamaActions  # noqa: E402
from werewolf.prompt_contract import (  # noqa: E402
    PROMPT_CONTRACT_POLICY_SIMPLIFIED,
    find_sections,
)
from werewolf.setup import create_game  # noqa: E402

CONFIG = str(REPO / "config" / "players.json")
SEED = 20261121


def run_state():
    state = create_game(seed=SEED, config_path=CONFIG, response_language="日本語")
    client = SimClient(state)
    actions = OllamaActions(
        state,
        client,
        seed=SEED,
        options={"question_hint_mode": "strict"},
        discussion_claim_mode="legacy",
    )
    engine = GameEngine(state, actions, seed=SEED)
    engine.run()
    return state


def main() -> None:
    state = run_state()
    history = tuple(
        message
        for message in state.conversation_log
        if message.speaker != "system" or "begins" in (message.content or "")
    )
    bodies = unique_bodies(state)
    verbose = "--detail" in sys.argv
    actors = [player for player in state.players if player.is_alive]
    totals: dict[str, list[int]] = {}
    for actor in actors:
        ctx = build(actor, state, history, PROMPT_CONTRACT_POLICY_SIMPLIFIED)
        text = ctx.user_prompt
        spans = find_sections(text)
        ordered = sorted(spans.items(), key=lambda kv: kv[1][0])

        def where(pos: int, ordered=ordered) -> str:
            current = "(prefix)"
            for name, (start, end) in ordered:
                if start <= pos:
                    current = name
                if start <= pos < end:
                    return current
            return current

        repeated = [body for body in bodies if text.count(body) > 1]
        for body in repeated:
            for match in re.finditer(re.escape(body), text):
                bucket = totals.setdefault(where(match.start()), [0, 0])
                bucket[0] += 1
                bucket[1] += len(body)
        if verbose and actor.name == "Player 2":
            print(f"--- detail actor={actor.name} repeated={len(repeated)}", flush=True)
            for body in repeated:
                print("  " + "=" * 6, f"{len(body)}字 {body[:40]!r}", flush=True)
                for match in re.finditer(re.escape(body), text):
                    line_start = text.rfind("\n", 0, match.start()) + 1
                    line_end = text.find("\n", match.start())
                    line = text[line_start : (line_end if line_end > 0 else len(text))]
                    print(
                        f"     [{where(match.start())}] {line[:120]!r}",
                        flush=True,
                    )
    print("== simplified で同じ本文を2度以上載せている場所（全生存者合計） ==")
    for section, (count, chars) in sorted(totals.items(), key=lambda kv: -kv[1][1]):
        tag = "（証拠＝本来の1箇所）" if section == "conversation_evidence" else ""
        print(f"   {section}: {count}箇所 / {chars}字 {tag}")


if __name__ == "__main__":
    main()
