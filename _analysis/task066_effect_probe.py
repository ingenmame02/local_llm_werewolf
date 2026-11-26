"""TASK-066: 再生成コンテキスト段階化の効果を、実プロンプトでモック計測するプローブ。

実機LLは使わない（ローカル環境の都合上、実機Ollamaでの比較は行っていない）。`_ScriptedClient` で
「空応答が3回続く」＝同一条件の連続失敗を再現し、段階が
`full_preserved → light_compressed → last_resort_compressed` と進むことを実プロンプトで確認する。
同じ盤面のまま方策だけを `legacy_early_compression`（TASK-066以前の早期圧縮）へ替えてA/Bを取り、
再生成プロンプトの実字数・保持節／削除節・証拠ID・畳まれた発言数を突き合わせる。

注目する数値:
- 送信プロンプト実字数（初回生成に対する比率と節約字数）と送信合計（節約の総量）
- 保持節／削除節（light_compressed で何が落ち、何を残したままにできるか）
- 発言履歴を畳んだときに落ちた発言数・文字数・最古/最新時点・失われた証拠ID
- 本文が出たか・前回本文との長さ差（圧縮で不自然な短文化が起きていないか）
- thinking off / 生成上限引き上げがいつ効いたか（出力予算側と圧縮側の分離）

実行方法: python _analysis/task066_effect_probe.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from werewolf.llm_actions import OllamaActions  # noqa: E402
from werewolf.models import Message  # noqa: E402
from werewolf.retry_context_policy import (  # noqa: E402
    RETRY_STAGE_FULL_PRESERVED,
    RETRY_STAGE_LAST_RESORT_COMPRESSED,
    RETRY_STAGE_LIGHT_COMPRESSED,
)
from werewolf.setup import create_game  # noqa: E402

# 空応答の次に出る「形は正しく中身は素朴な」発言。圧縮度合いで直前の議論へどれだけ接続できるか
# 比較したいので、文面はこちらで固定する（モデルの自由を測るプローブではない）。
VALID_BODY = (
    "[CO] None\n"
    "[BODY] 初日なので断定はできませんが、まず各役職の主張を出し合わせるのが早いと思います。"
    "私はまだ情報を未取得なので、占い結果が出ている方がいれば先行して共有してほしいです。"
)


class _ScriptedClient:
    """送信を実際に発行しないモック（LL不使用）。送信プロンプトとオプションを記録する。"""

    def __init__(self, responses: list[object]) -> None:
        self.timeout = 120.0
        self.max_retries = 1
        self.timeout_retries = 0
        self._responses = list(responses)
        self.sent_options: list[dict[str, object]] = []
        self.sent_prompts: list[list[dict[str, object]]] = []

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {}

    def chat(
        self, model: str, messages: list[dict[str, object]], options: dict | None = None
    ) -> str:
        self.sent_options.append(dict(options or {}))
        self.sent_prompts.append([dict(message) for message in messages])
        entry = self._responses.pop(0) if self._responses else VALID_BODY
        if isinstance(entry, BaseException):
            # タイムアウト・コンテキスト超過の再現（転送失敗は`OllamaError`で届く）
            raise entry
        assert isinstance(entry, str), "このプローブは文字列応答か例外だけを使う"
        return entry


def _scenario(*, retry_context_policy: str, empties: int, timeout_first: bool = False) -> dict[str, object]:
    """同一seed・同一盤面で方策だけ差し替えた実行結果（再生成3回まで）を返す。"""
    state = create_game(
        seed=7066, config_path="config/players.json", response_language="日本語"
    )
    state.retry_context_policy = retry_context_policy
    actor = state.players[0]
    names = tuple(player.name for player in state.players)
    others = tuple(name for name in names if name != actor.name)
    # 直前の議論＝再生成で読むべき証拠。段階によってプロンプトに残るかどうかが要点。
    state.day = 2
    state.discussion_turn = 2
    for index, speaker in enumerate(others[:4]):
        state.conversation_log.append(
            Message(
                speaker=speaker,
                content=(
                    f"{actor.name}に聞きますが、なぜその人物に投票しようとしているのですか？"
                    f"根拠は論点{index + 1}（{names[(index + 1) % len(names)]}の主張）です。"
                ),
                day=2,
                turn=1,
                # エンジンが発行する発言IDを再現する（再生成で `[speech:N]` がどう残るかを見るため）
                speech_id=index + 1,
                timeline_index=index + 1,
            )
        )
    responses: list[object] = []
    if timeout_first:
        # TASK-035の実測どおり、タイムアウトの再送に長いプロンプトを戻さない挙動を見る
        from werewolf.ollama import OllamaError

        responses.append(OllamaError("timeout", "mock timeout", 1))
    responses.extend([""] * empties)
    client = _ScriptedClient(responses)
    events: list[dict[str, object]] = []
    actions = OllamaActions(
        state, client, seed=state.seed, event_callback=events.append
    )
    prompt_ctx = actions._build_discussion_prompt_with_budget(
        actor,
        tuple(state.alive_players),
        tuple(state.conversation_log),
        2,
        include_claim=True,
        budget_key="discussion",
    )
    text = actions._free_text_with_context(
        "discussion",
        actor,
        prompt_ctx,
        "議論を続けます。",
        time.monotonic(),
        recent_messages=tuple(state.conversation_log),
        budget_key="discussion",
    )
    stages = [event for event in events if event.get("type") == "retry_context_stage"]
    return {
        "policy": retry_context_policy,
        "client": client,
        "stages": stages,
        "final_body": text,
        "initial_prompt_chars": len(prompt_ctx.user_prompt),
    }


def _user_chars(messages: list[dict[str, object]]) -> int:
    for message in messages:
        if str(message.get("role")) == "user":
            return len(str(message.get("content", "")))
    return 0


def _stage_names(stages: list[dict[str, object]]) -> list[str]:
    return [str(event.get("retry_stage") or "") for event in stages]


def summarize(result: dict[str, object]) -> None:
    print(f"\n=== policy={result['policy']} ===")
    print(f"初回生成プロンプト実字数: {result['initial_prompt_chars']}")
    client = result["client"]
    prompts = [_user_chars(messages) for messages in client.sent_prompts]
    print(f"送信プロンプト実字数（送信順）: {prompts}")
    if len(prompts) >= 2 and prompts[0]:
        print(
            "  1回目再生成/初回: "
            f"ratio={round(prompts[1] / float(prompts[0]), 4)} "
            f"saved={prompts[0] - prompts[1]}字"
        )
    for index, option in enumerate(client.sent_options):
        print(
            f"  send#{index + 1}: num_predict={option.get('num_predict')} "
            f"thinking={option.get('thinking')} num_ctx={option.get('num_ctx')}"
        )
    stages = list(result["stages"])
    print(f"段階イベント数: {len(stages)}")
    for event in stages:
        print(
            f"  #{event.get('request_number')} stage={event.get('retry_stage')} "
            f"trigger={event.get('compression_trigger')} "
            f"class={event.get('retry_failure_class')} "
            f"compresses={event.get('retry_compresses_prompt')} "
            f"folds={event.get('retry_folds_speech_history')} "
            f"thinking_off={event.get('retry_thinking_off')} "
            f"escalate={event.get('retry_escalate_generation')} "
            f"budget_fit={event.get('retry_budget_fit')} outcome={event.get('retry_outcome')}"
        )
        print(
            "      prompt chars: initial="
            f"{event.get('initial_user_prompt_char_count')} retry="
            f"{event.get('retry_user_prompt_char_count')} saved="
            f"{event.get('retry_prompt_saved_chars')} ratio={event.get('retry_prompt_ratio')}"
        )
        print(
            f"      style={event.get('retry_prompt_style')} "
            f"light_dropped={event.get('retry_light_dropped_sections')}"
        )
        print(f"      dropped_sections={event.get('dropped_sections')}")
        preserved = list(event.get("preserved_sections") or [])
        print(f"      preserved_sections({len(preserved)})={preserved}")
        print(
            "      evidence ids: preserved="
            f"{len(event.get('preserved_evidence_ids') or [])} dropped="
            f"{len(event.get('dropped_evidence_ids') or [])} "
            f"{list(event.get('dropped_evidence_ids') or [])[:8]}"
        )
        # TASK-066（レビュー対応）: 削除IDを「載っていた節」で分ける。core節・現在質問由来が
        # 0 なら、落としたのは optional 派生節に載っていたIDだけ（圧縮の想定内）。
        print(
            "      id safety: total="
            f"{event.get('dropped_evidence_id_total', 0)} core節由来="
            f"{event.get('dropped_ids_protected_count', 0)} 質問由来="
            f"{event.get('dropped_ids_question_linked_count', 0)} optionalのみ="
            f"{event.get('dropped_ids_optional_only_count', 0)} clean="
            f"{event.get('dropped_ids_safety_clean')}"
            f" 詳細={list(event.get('dropped_ids_protected_detail') or [])[:4]}"
        )
        print(
            "      IDが消えた発言の本文: 保持="
            f"{event.get('dropped_speech_bodies_still_present')} 喪失="
            f"{event.get('dropped_speech_bodies_lost')}"
        )
        print(
            f"      timeout起因={event.get('retry_timeout_caused')} "
            f"即時圧縮（意図的な例外）={event.get('retry_immediate_compression_exception')}"
        )
        if event.get("dropped_speech_count"):
            print(
                "      folded speech: count="
                f"{event.get('dropped_speech_count')} chars="
                f"{event.get('dropped_speech_chars')} oldest="
                f"{event.get('dropped_speech_oldest_point')} newest="
                f"{event.get('dropped_speech_newest_point')}"
            )
        print(
            f"      body_preserved={event.get('body_preserved')} delta="
            f"{event.get('retry_body_char_delta')} shortened="
            f"{event.get('retry_body_shortened')}"
        )
    print(f"最終本文（先頭80字）: {str(result['final_body'])[:80]!r}")


EVIDENCE_MARK = "なぜその人物に投票しようとしているのですか"
# 最も古い発言（core-ledger短縮プロンプトでは「直近3発言」しかないため落ちる）
OLD_SPEECH_MARK = "根拠は論点1"
# light_compressed で落としてよい派生節のみ（core・現在質問・証拠台帳は落ちたら失敗）
ALLOWED_LIGHT_DROPS = frozenset(
    {
        "continuity_memo",
        "hypothesis_ledger",
        "inference_line",
        "integrity_alerts",
        "reasoning_guidance",
        "werewolf_theory_card",
        "previous_topics",
        "direct_answer",
        "day1_first_turn_guidance",
        "first_night_guidance",
        "round_transcript",
    }
)
MUST_KEEP = (
    "conversation_evidence",
    "question_block",
    "fact_ledger",
    "claim_ledger",
    "continuity_checkpoint",
)


def main() -> None:
    tiered = _scenario(retry_context_policy="tiered", empties=3)
    summarize(tiered)

    legacy = _scenario(retry_context_policy="legacy_early_compression", empties=3)
    summarize(legacy)

    slow = _scenario(retry_context_policy="tiered", empties=2, timeout_first=True)
    summarize(slow)

    tiered_prompts = [_user_chars(msgs) for msgs in tiered["client"].sent_prompts]
    legacy_prompts = [_user_chars(msgs) for msgs in legacy["client"].sent_prompts]
    slow_prompts = [_user_chars(msgs) for msgs in slow["client"].sent_prompts]
    tiered_stages = _stage_names(list(tiered["stages"]))
    legacy_stages = _stage_names(list(legacy["stages"]))
    slow_stages = _stage_names(list(slow["stages"]))
    print("\n=== A/B（同一盤面・送信プロンプト合計字数）===")
    print(
        f"  tiered total={sum(tiered_prompts)} legacy total={sum(legacy_prompts)} "
        f"delta={sum(tiered_prompts) - sum(legacy_prompts)}"
    )
    print(f"  tiered stages={tiered_stages}")
    print(f"  legacy stages={legacy_stages}")
    print(f"  timeout(tiered) stages={slow_stages} prompts={slow_prompts}")

    # tiered: 再生成1回目は同一証拠をそのまま保持（初回生成と同じ議論が読める）
    assert tiered_stages[:2] == [
        RETRY_STAGE_FULL_PRESERVED,
        RETRY_STAGE_LIGHT_COMPRESSED,
    ], tiered_stages
    first_retry = str(tiered["client"].sent_prompts[1][-1]["content"])
    light_retry = str(tiered["client"].sent_prompts[2][-1]["content"])
    assert EVIDENCE_MARK in first_retry, "full_preservedでは直前の議論が残る"
    assert OLD_SPEECH_MARK in first_retry, "full_preservedでは古い発言まで同一証拠のまま残る"
    assert EVIDENCE_MARK in light_retry, "light_compressedでも直前の議論は残す"
    assert OLD_SPEECH_MARK in light_retry, "light_compressedでも発言本文は畳まない"
    tiers = list(tiered["stages"])
    assert tiers[0].get("retry_compresses_prompt") is False
    assert not tiers[0].get("dropped_sections"), "初回再生成で節を落とさない"
    light_drops = set(tiers[1].get("dropped_sections") or [])
    assert light_drops and light_drops <= ALLOWED_LIGHT_DROPS, light_drops
    kept_after_light = set(tiers[1].get("preserved_sections") or [])
    assert set(MUST_KEEP) <= kept_after_light, sorted(set(MUST_KEEP) - kept_after_light)

    # legacy（TASK-066以前）: 再生成1回目から発言履歴を畳み、証拠IDも失っていた
    assert set(legacy_stages) == {RETRY_STAGE_LAST_RESORT_COMPRESSED}, legacy_stages
    legacy_first = str(legacy["client"].sent_prompts[1][-1]["content"])
    assert EVIDENCE_MARK in legacy_first, "旧方策でも直近3発言までは読む"
    assert OLD_SPEECH_MARK not in legacy_first, "旧方策では再生成1回目から古い発言本文が消える"
    legacy_ids_lost = max(
        len(event.get("dropped_evidence_ids") or []) for event in legacy["stages"]
    )
    print(f"\n  legacy で失われた証拠ID（最大）: {legacy_ids_lost}")
    print(f"  tiered 1回目の節約字数: {tiers[0].get('retry_prompt_saved_chars')}")
    print(f"  tiered light の節約字数: {tiers[1].get('retry_prompt_saved_chars')}")

    # タイムアウトの再送は長いプロンプトを戻さない（待ち時間の実測が主因）
    assert slow_stages and slow_stages[0] == RETRY_STAGE_LAST_RESORT_COMPRESSED, slow_stages
    slow_events = list(slow["stages"])
    assert slow_events[0].get("compression_trigger") == "timeout_repeat_short_prompt", (
        slow_events[0].get("compression_trigger")
    )
    assert slow_prompts[1] < slow_prompts[0], "タイムアウト再送は短縮プロンプト"
    slow_retry = str(slow["client"].sent_prompts[1][-1]["content"])
    assert "[Unanswered Question To You]" in slow_retry or "[CONTINUITY_CHECKPOINT" in slow_retry

    # TASK-066（レビュー対応）: light_compressed でIDが落ちても、それが現在質問・core節由来
    # ではなく、発言本文は残っていることを診断で確かめる（圧縮の安全性を数値で判断できるようにする）
    light_event = tiers[1]
    print(
        f"\n  light_compressed の削除ID: {list(light_event.get('dropped_evidence_ids') or [])}"
    )
    print(
        "  id safety: core節由来="
        f"{light_event.get('dropped_ids_protected_count')} 質問由来="
        f"{light_event.get('dropped_ids_question_linked_count')} optionalのみ="
        f"{light_event.get('dropped_ids_optional_only_count')} clean="
        f"{light_event.get('dropped_ids_safety_clean')}"
    )
    print(
        "  IDが消えた発言の本文: 保持="
        f"{light_event.get('dropped_speech_bodies_still_present')} 喪失="
        f"{light_event.get('dropped_speech_bodies_lost')}"
    )
    assert int(light_event.get("dropped_ids_protected_count") or 0) == 0, light_event.get(
        "dropped_ids_protected_detail"
    )
    assert int(light_event.get("dropped_ids_question_linked_count") or 0) == 0
    assert (
        int(light_event.get("dropped_evidence_id_total") or 0)
        == int(light_event.get("dropped_ids_optional_only_count") or 0)
    ), "light_compressed で消えるIDは optional 派生節由来だけ"
    assert int(light_event.get("dropped_speech_bodies_lost") or 0) == 0, (
        "light_compressed は発言本文を失わない（ID行だけがoptional節から消える）"
    )

    # legacy（A/B基準側）は発言履歴を畳むため core 節のIDも失う。ここが新方策との差
    legacy_core_losses = max(
        int(event.get("dropped_ids_protected_count") or 0) for event in legacy["stages"]
    )
    print(f"  legacy（旧方策）で失われたcore節由来ID（最大）: {legacy_core_losses}")

    # timeout起因は「圧縮は最終手段」に対する意図的な例外として診断へ明記される
    assert slow_events[0].get("retry_timeout_caused") is True, slow_events[0]
    assert slow_events[0].get("retry_immediate_compression_exception") is True
    non_timeout_events = [event for event in tiers if not event.get("retry_timeout_caused")]
    assert len(non_timeout_events) == len(tiers), "形式・空応答再生成にtimeout例外は混ざらない"
    print("\nOK: tiered は同一証拠保持→派生節のみ、legacy は初回再生成から履歴を畳む")
    print("OK: タイムアウト再送は意図的な例外（即時圧縮）として診断へ明記される")
    print("OK: light_compressed の削除IDは optional 派生節由来のみで発言本文は保持")


if __name__ == "__main__":
    main()

