"""TASK-062の効果測定（実機LL不使用・モック盤面）。

Day 1〜3へ発言を積み上げたとき、`[CONTINUITY_CHECKPOINT]`が論理状態を引き継ぐことで
①発言本文の再掲（recent層）が減るか、②状態節自身の字数が日に比例して増えないかを数える。
Ollamaへは送信しない（エンジンとプロンプト構築のみ）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf.engine import GameEngine  # noqa: E402
from werewolf.models import GameState, Message, Phase  # noqa: E402
from werewolf.prompts import PromptBuilder  # noqa: E402
from werewolf.setup import create_game  # noqa: E402

# hybrid_retrieval が選びつつ core の優先度は plain（＝畳める）になる文章にする。
# 「確認したい」系で選択され、反証・投票の語を含まないので stub化の対象になる。
LONG_SPEECH = (
    "{name}の{index}番目の発言です。{rival}の話していた順番と昨夜の出来事を並べてみると、"
    "もう少し詳しく確認したくなってきました。私としては、{rival}が最初に触れた件と、"
    "その後に続けた説明の続きを、この場で改めて聞きたい気持ちです。"
    "{name}として別の角度から見ると、まだ判断しかねる部分が残っています。"
)


def _build(seed: int, continuity: str, days: int = 4):
    state = create_game(
        seed=seed, config_path="config/players.json", response_language="日本語"
    )
    state.continuity_policy = continuity
    GameState.__post_init__(state)
    names = tuple(player.name for player in state.players)
    engine = GameEngine(state, None, seed=seed)
    for day in range(1, days + 1):
        state.day = day
        state.conversation_log.append(Message("system", f"Day {day} begins."))
        state.public_timeline.record_day_start(day)
        state.phase = Phase.DAY_DISCUSSION
        for turn in range(1, 4):
            state.discussion_turn = turn
            state.begin_discussion_turn()
            for offset in range(3):
                speaker = names[(day * 3 + turn + offset) % len(names)]
                rival = names[(day * 3 + turn + offset + 3) % len(names)]
                body = LONG_SPEECH.format(name=speaker, index=offset, rival=rival)
                state.append_discussion_message(Message(speaker, body))
                state.record_topics_from_speech(speaker, body, day, turn)
                if offset == 2:
                    state.store_model_reasoning_memo(speaker, f"{rival}の根拠を確認したい。")
            engine._issue_continuity_checkpoint(f"day-{day}-post_turn{turn}")
    return state, engine


def _measure(state, actor_name: str, layout: str) -> dict[str, object]:
    actor = next(p for p in state.players if p.name == actor_name)
    ctx = PromptBuilder.build_discussion_prompt(
        actor,
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
    diag = dict(ctx.diagnostics or {})
    return {
        "user_chars": len(ctx.user_prompt),
        "recent_chars": int(diag.get("core_recent_chars") or 0),
        "historical_chars": int(diag.get("core_historical_chars") or 0),
        "anchor_chars": int(diag.get("core_anchor_chars") or 0),
        "state_chars": int(diag.get("continuity_state_chars") or 0),
        "folded": int(diag.get("evidence_continuity_folded_count") or 0),
        "fold_saved": int(diag.get("evidence_continuity_fold_saved_chars") or 0),
        "items_total": int(diag.get("continuity_items_total") or 0),
        "items_unchanged": int(diag.get("continuity_items_unchanged") or 0),
        "items_changed": int(diag.get("continuity_items_changed") or 0),
        "open_questions": int(diag.get("continuity_open_questions") or 0),
        "layout": diag.get("prompt_layout"),
        "violations": len(list(diag.get("prompt_layout_violations") or [])),
    }


def main() -> int:
    seed = 2062061
    results: dict[str, dict[int, dict[str, object]]] = {}
    actor_name: str | None = None
    for continuity, layout in (("off", "v3"), ("normalized", "v4")):
        state, _engine = _build(seed, continuity)
        actor_name = actor_name or state.alive_players[2]
        per_day: dict[int, dict[str, object]] = {}
        for day in (1, 2, 3, 4):
            state.day = day
            per_day[day] = _measure(state, actor_name, layout)
        results[continuity] = per_day
    print(f"actor={actor_name} seed={seed}")
    for name, per_day in results.items():
        print(f"[continuity={name}]")
        for day, row in sorted(per_day.items()):
            print(
                f"  Day {day}: user={row['user_chars']} recent={row['recent_chars']} "
                f"hist={row['historical_chars']} anchor={row['anchor_chars']} "
                f"state={row['state_chars']} folded={row['folded']} "
                f"saved={row['fold_saved']} layout={row['layout']} viol={row['violations']}"
            )
        first, last = per_day[1], per_day[4]
        print(
            "  Day1->Day4 delta: user="
            f"{int(last['user_chars']) - int(first['user_chars'])}, recent="
            f"{int(last['recent_chars']) - int(first['recent_chars'])}, state="
            f"{int(last['state_chars']) - int(first['state_chars'])}"
        )
    off_last, on_last = results["off"][4], results["normalized"][4]
    print(
        "Day4 off vs normalized: user diff="
        f"{int(on_last['user_chars']) - int(off_last['user_chars'])}, recent diff="
        f"{int(on_last['recent_chars']) - int(off_last['recent_chars'])}, checkpoint chars="
        f"{on_last['state_chars']}, folded bodies={on_last['folded']} "
        f"(saved {on_last['fold_saved']} chars), unchanged(count only)="
        f"{on_last['items_unchanged']}, questions={on_last['open_questions']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
