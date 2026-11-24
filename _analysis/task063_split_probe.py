"""TASK-063の効果測定（実機LL不使用・スクリプト済みモック）。

強化モデル（thinking-fixed系）で起きる thinking-only・timeoutを「プロンプト論理の失敗」と
分離できるか、また短縮再要求で入力がどれだけ減るか、本文を失ったかを数える。
送信は ScriptedClient が順番に返すだけで、Ollamaへはアクセスしない。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf.llm_actions import OllamaActions  # noqa: E402
from werewolf.models import Message  # noqa: E402
from werewolf.ollama import ChatResponse  # noqa: E402
from werewolf.output_budget_trace import CATEGORY_OK  # noqa: E402
from werewolf.setup import create_game  # noqa: E402

FIXED_MODEL = "llm-jp-4.1-33b-thinking-fixed"


class ScriptedClient:
    def __init__(self, responses: list[object]) -> None:
        self.timeout = 120.0
        self.max_retries = 1
        self.timeout_retries = 0
        self._responses = list(responses)
        self.sent_prompts: list[list[dict[str, object]]] = []

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {}

    def chat(self, model, messages, options=None):
        self.sent_prompts.append([dict(m) for m in messages])
        entry = self._responses.pop(0) if self._responses else "[CO] None\n[BODY] 一旦様子を見ます。"
        if isinstance(entry, ChatResponse):
            self._last_chat_response = entry
            return entry.content
        self._last_chat_response = None
        return str(entry)


def main() -> int:
    state = create_game(
        seed=2063063, config_path="config/players.json", response_language="日本語"
    )
    actor = next(p for p in state.players if p.model == FIXED_MODEL)
    names = tuple(player.name for player in state.players)
    # Day 2 の実形（CO・占い申告・投票）を作ってチェックポイントを出しておく
    state.day = 1
    state.conversation_log.append(Message("system", "Day 1 begins."))
    state.public_timeline.record_day_start(1)
    for index, name in enumerate(names[:4]):
        body = (
            f"{name}は昨夜の動きを確認したいと考えています。"
            f"{names[(index + 3) % len(names)]}の後続の説明も、この場で聞かせてください。"
        )
        state.append_discussion_message(Message(name, body))
        state.store_model_reasoning_memo(name, f"{names[(index + 1) % len(names)]}の続報を確認したい。")
    checkpoint = state.issue_continuity_checkpoint("day-1-pre_vote")
    state.vote_history.append((1, False, {names[0]: names[3], names[1]: names[3]}))

    client = ScriptedClient(
        [
            # 1送信目: thinkingのみで本文が空（生成上限を思考が食い潰した）
            ChatResponse(
                content="",
                attempts=1,
                content_length=0,
                thinking_present=True,
                done_reason="length",
                eval_count=1024,
            ),
            # 2送信目: 短縮再要求で本文を出す
            "[CO] None\n[BODY] Player 5の確認要求には答えられていないので、続報を尋ねます。",
        ]
    )
    actions = OllamaActions(state, client, seed=state.seed)
    alive = tuple(state.alive_players)
    history = tuple(state.conversation_log)
    actions.state.reasoning_memory_enabled = True
    body = actions.discuss(actor.name, 2, alive, history)

    metrics = actions.output_budget_snapshot()
    per_model = metrics["metrics"]["output_budget_by_model"].get(FIXED_MODEL, {})
    print("checkpoint chars:", int(checkpoint["char_count"]) if checkpoint else 0)
    print("final body:", repr(body[:60]), "chars=", len(body))
    print("categories:", per_model.get("category_counts"))
    print("body_generation_rate:", per_model.get("body_generation_rate"))
    print(
        "budget_bound:", per_model.get("budget_bound_failures"),
        "logic_bound:", per_model.get("logic_bound_failures"),
    )
    print("short_retry_count:", per_model.get("short_retry_count"))
    sizes = [sum(len(str(m.get("content", ""))) for m in msgs) for msgs in client.sent_prompts]
    print("sent prompt char sizes:", sizes)
    if len(sizes) >= 2:
        print(f"retry saved: {sizes[0] - sizes[1]} chars ({round((1 - sizes[1] / sizes[0]) * 100)}%)")
    outcomes = metrics["outcomes"]
    for outcome in outcomes:
        if outcome["model"] == FIXED_MODEL:
            print(
                "outcome:", outcome["action"], outcome["attempt_categories"],
                "body_preservation=", outcome["body_preservation"],
            )
    assert per_model.get("thinking_only_count") == 1, per_model
    assert CATEGORY_OK in (per_model.get("category_counts") or {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
