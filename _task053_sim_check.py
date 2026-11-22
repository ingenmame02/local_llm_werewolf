"""TASK-053の動作確認（実機なし・模擬クライアントでゲーム1本を回す）。

game_088e49f0690c で観測された「思考がnum_predictを食い潰して本文が空（done_reason=length）」
を再現し、以下を検証する:

- 送信num_predictが小さい行動では thinking-only 打ち切り → 本文予算予約
  (`model_action_body_reservation`) が利くと本文が出る
- 常に思考だけで終わるモデルは、余白拡大を尽くした後にのみ thinking off（能力判定）化する
- thinking-only 再要求では長いプロンプトを再送しない（core-ledger短縮）
- 送信ごとの観測がモデル×アクション別に集計され、game_result へ残る
"""
import json
import sys
from collections import Counter
from pathlib import Path

SRC = Path(r'c:\Users\yukic\local_llm_werewolf\src')
sys.path.insert(0, str(SRC))

from werewolf.cli import GameLogger
from werewolf.engine import GameEngine
from werewolf.llm_actions import OllamaActions
from werewolf.models import Role
from werewolf.ollama import ChatResponse
from werewolf.setup import create_game

CONFIG = r'c:\Users\yukic\local_llm_werewolf\config\players.json'
LOG_DIR = Path(r'c:\Users\yukic\local_llm_werewolf\_scratch_task053')
# 実機ログ game_088e49f0690c で「思考が本文枠を食い潰した」送信の実測生成上限。
# この値以下では thinking-only + done_reason=length を返して失敗を再現する。
STARVE_NUM_PREDICT_BY_ACTION: dict[str, int] = {"seer": 1024, "attack": 1024, "vote": 768}


class ThinkingStarvingClient:
    """送信num_predict・thinking設定に応じて実機の失敗形態を再現する模擬クライアント。"""

    timeout = 60.0
    max_retries = 1
    timeout_retries = 1

    def __init__(
        self,
        state,
        *,
        short_action_models: tuple[str, ...] = (),
        hopeless_wolf_model: str | None = None,
        first_request_thinking_only_models: frozenset[str] = frozenset(),
    ) -> None:
        self.state = state
        self.names = [p.name for p in state.players]
        self.requests = 0
        # 短い行動（投票・占い・襲撃）で生成上限が小さい間のみ思考-only で空にするモデル。
        self.short_action_models = set(short_action_models)
        # 人狼相談では thinking off 判定が出るまで一貫して思考-only を返し続けるモデル。
        self.hopeless_wolf_model = hopeless_wolf_model
        # 各行動の初回送信で一度だけ思考-only を返し、短縮再要求経路を検証するモデル。
        self.first_request_models = set(first_request_thinking_only_models)
        self._first_request_used: set[tuple[str, str]] = set()
        self.sent_budgets: list[tuple[str, str, int, bool | None]] = []

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {"ollama_ps_context": None}

    def _others(self, speaker: str) -> list[str]:
        return [n for n in self.names if n != speaker and n in self.state.alive_players]

    def chat(self, model, messages, options=None) -> str:
        response = self.chat_with_metadata(model, messages, options)
        # 実Ollamaクライアントと同様に直近応答のメタ情報（done_reason・eval_count）を残す。
        # `_chat` は送信前にこの属性を消してから読み取るため、每次ここで設定し直す。
        self._last_chat_response = response
        return response.content

    @staticmethod
    def _thinking_only(sent: int) -> ChatResponse:
        return ChatResponse(
            content="<think>長い思考。ここで全文を消費して打ち切られる。</think>",
            attempts=1,
            done_reason="length",
            eval_count=sent,
        )

    def chat_with_metadata(self, model, messages, options=None, tools=None) -> ChatResponse:
        self.requests += 1
        opts = dict(options or {})
        num_predict = opts.get("num_predict")
        sent = int(num_predict) if isinstance(num_predict, int) else 512
        thinking = opts.get("thinking") if isinstance(opts.get("thinking"), bool) else None
        system = str(messages[0]["content"]) if messages else ""
        user = str(messages[1]["content"]) if len(messages) > 1 else ""
        speaker = ""
        for name in self.names:
            if f'You are "{name}"' in system:
                speaker = name
                break
        others = self._others(speaker)
        other = others[0] if others else speaker

        if "Consult with your partner" in user:
            action = "wolf_discussion"
        elif "Living wolves choose 1 non-wolf" in user:
            action = "attack"
        elif "Divine 1 undivined living player" in user:
            action = "seer"
        elif "Vote for 1 other living player" in user:
            action = "vote"
        elif "[CO] None, Hold, or your public role claim" in user:
            action = "discussion_body"
        else:
            action = "other"
        self.sent_budgets.append((model, action, sent, thinking))

        # (a) 人狼相談で常に思考だけ（実機ログのgemma系相当）。off判定まで本文を出さない。
        if (
            action == "wolf_discussion"
            and model == self.hopeless_wolf_model
            and thinking is not False
        ):
            return self._thinking_only(sent)
        # (b) 指定モデルの短い行動は、生成上限が小さい間は思考が本文枠を削る（gpt-oss相当）。
        limit = STARVE_NUM_PREDICT_BY_ACTION.get(action)
        if limit is not None and model in self.short_action_models and sent <= limit:
            return self._thinking_only(sent)
        # (c) 指定モデルの初回送信だけ思考-only にし、短縮再要求（core-ledger）を検証する。
        first_key = (model, action)
        if (
            action in {"discussion", "discussion_body"}
            and model in self.first_request_models
            and first_key not in self._first_request_used
        ):
            self._first_request_used.add(first_key)
            return self._thinking_only(sent)

        if action == "seer":
            return ChatResponse(content=f"Seer target: {other}", attempts=1)
        if action == "vote":
            return ChatResponse(content=f"Vote: {other}", attempts=1)
        if action == "attack":
            return ChatResponse(content=f"Attack target: {other}", attempts=1)
        if action == "wolf_discussion":
            return ChatResponse(
                content=f"今夜は{other}を攻撃したい。\nAttack target: {other}", attempts=1
            )
        if action == "discussion_body":
            return ChatResponse(
                content=(
                    f"[CO] None\n[BODY] {other}さんの投票理由とCOの整合性を確認したい。"
                    "私は公開情報から疑いを整理し、次の質問先を決めたい。"
                ),
                attempts=1,
            )
        return ChatResponse(content=f"[CO] None\n[BODY] {other}さんのCOを確認したい。", attempts=1)


def _print_attempt_breakdown(attempts: list[dict[str, object]]) -> None:
    grouped = Counter(
        (str(a.get("model")), str(a.get("action_budget_key"))) for a in attempts
    )
    print("\n[送信の内訳（モデル×アクション）]")
    for (model, key), total in sorted(grouped.items()):
        rows = [
            a
            for a in attempts
            if str(a.get("model")) == model and str(a.get("action_budget_key")) == key
        ]
        starved = sum(1 for a in rows if a.get("thinking_only_response"))
        empty = sum(1 for a in rows if a.get("body_empty"))
        budgets = sorted(
            {int(a["num_predict"]) for a in rows if isinstance(a.get("num_predict"), int)}
        )
        last_profile = rows[-1].get("generation_profile") or {}
        print(f"  {model} × {key}: requests={total} thinking_only={starved} empty={empty} num_predict={budgets}")
        print(
            f"    floor={last_profile.get('thinking_floor_tokens')}"
            f" allowance={last_profile.get('allowance_tokens')}"
            f" exhaustions={last_profile.get('exhaustion_count')}"
            f" widenings={last_profile.get('widened_count')}"
            f" covered_thinking_only={last_profile.get('covered_thinking_only_count')}"
            f" thinking_share={last_profile.get('thinking_share_rate')}"
            f" avg_body_chars={last_profile.get('avg_body_chars')}"
        )


def main() -> None:
    game_id = "task053sim"
    state = create_game(seed=20261119, config_path=CONFIG, response_language="日本語")
    seer_model = next(
        (player.model for player in state.players if player.role is Role.SEER), None
    )
    wolf_model = next(
        (player.model for player in state.players if player.role is Role.WOLF), None
    )
    # 占い役のモデルは毎晩占うため、思考枯渇→余白拡大→次回以降は本文予約、という流れを
    # 観測できる。人狼役のモデルは人狼相談で思考を使い切る失敗（実機のgemma系相当）を出す。
    client = ThinkingStarvingClient(
        state,
        short_action_models=tuple(model for model in (seer_model,) if model),
        hopeless_wolf_model=wolf_model,
        first_request_thinking_only_models=frozenset(
            model for model in (seer_model, wolf_model) if model
        ),
    )
    events: list[dict[str, object]] = []
    logger = GameLogger(
        game_id, LOG_DIR, {"seed": 20261119, "task": "TASK-053"}, private_enabled=True
    )

    def on_event(event: dict[str, object]) -> None:
        events.append(event)
        public_event = {
            key: value
            for key, value in event.items()
            if key
            not in {
                "content",
                "selected_target",
                "parsed_target",
                "error",
                "response_metadata",
            }
        }
        logger.append_record(public_event, event)

    actions = OllamaActions(
        state,
        client,
        seed=20261119,
        options={"question_hint_mode": "none", "temperature": 0.7},
        discussion_claim_mode="legacy",
        event_callback=on_event,
    )
    engine = GameEngine(state, actions, seed=20261119, event_callback=on_event)
    result = engine.run()
    logger.finalize(state, traces=None, error=None)

    print("winner:", getattr(result, "winner", result), "| requests:", client.requests)
    counts = Counter(str(event.get("type")) for event in events)
    print("generation_attempt:", counts.get("llm_generation_attempt"))

    attempts = [e for e in events if e.get("type") == "llm_generation_attempt"]
    _print_attempt_breakdown(attempts)

    configs = [e for e in events if e.get("type") == "llm_request_configuration"]
    sources = Counter(str(e.get("num_predict_source")) for e in configs)
    print("\n[num_predict解決元]", dict(sources))
    reserved = [
        e for e in configs if e.get("num_predict_source") == "model_action_body_reservation"
    ]
    print("予約適用リクエスト:", len(reserved))
    for e in reserved[:10]:
        budget = e.get("body_budget") or {}
        print(
            f"  {e.get('model')} × {e.get('action_budget_key')}: sent={e.get('num_predict')}"
            f" body_min={budget.get('body_min_tokens')}"
            f" allowance={budget.get('thinking_allowance_tokens')}"
            f" required={budget.get('required_generation_tokens')}"
            f" covered={budget.get('body_covered')}"
            f" margin={e.get('thinking_margin_tokens')}"
        )

    traces = list(actions.traces)
    all_names = {player.name for player in state.players}
    fallback_traces = sum(1 for trace in traces if getattr(trace, "fallback_used", False))
    fatal_traces = sum(
        1 for trace in traces if getattr(trace, "fatal_safety_fallback", False)
    )
    unknown_targets = sum(
        1
        for trace in traces
        if getattr(trace, "selected_target", None)
        and str(getattr(trace, "selected_target")) not in all_names
    )
    print(
        f"\n[行動トレース] 件数={len(traces)} フォールバック={fallback_traces}"
        f" 致命的安全フォールバック={fatal_traces} 存在しない名前への指定={unknown_targets}"
    )

    profiles = actions.thinking_profiles_snapshot()
    print("\n[モデル×アクション別の観測プロファイル（空だった行動だけ）]")
    for model in sorted(profiles):
        for key in sorted(profiles[model]):
            prof = profiles[model][key]
            if not (
                prof.get("thinking_only_count")
                or prof.get("empty_content_count")
                or prof.get("covered_thinking_only_count")
                or prof.get("allowance_tokens")
            ):
                continue
            print(
                f"  {model} × {key}: thinking_only={prof['thinking_only_count']}"
                f" empty={prof['empty_content_count']}"
                f" floor={prof['thinking_floor_tokens']} allowance={prof['allowance_tokens']}"
                f" exhaustions={prof['exhaustion_count']} widenings={prof['widened_count']}"
                f" covered_thinking_only={prof['covered_thinking_only_count']}"
                f" empty_rate={prof['empty_response_rate']}"
                f" thinking_share={prof['thinking_share_rate']}"
                f" avg_body_chars={prof['avg_body_chars']}"
            )

    exhaustions = [e for e in events if e.get("type") == "thinking_budget_exhausted"]
    print("\n[thinking予算不足への対処（まず余白拡大、offは最終手段）]")
    for e in exhaustions:
        print(
            f"  {e.get('model')} × {e.get('action')}: widened={e.get('thinking_allowance_widened')}"
            f" kept={e.get('thinking_kept')} off_switched={e.get('thinking_off_switched')}"
            f" allowance={e.get('thinking_allowance_tokens')} count={e.get('thinking_exhaustion_count')}"
        )
    print(
        "  能力判定でthinking off化:",
        sorted(
            model
            for model, caps in state.model_capabilities.items()
            if caps.get("thinking_budget") is False
        ),
    )

    compressions = [e for e in events if e.get("type") == "context_compression_stage"]
    print("\n[再要求プロンプトの短縮（長いプロンプトを再送しない）]")
    for e in compressions[:8]:
        initial, retry = (
            e.get("initial_user_prompt_char_count"),
            e.get("retry_user_prompt_char_count"),
        )
        saving = (
            round((initial - retry) / initial, 3)
            if isinstance(initial, int) and isinstance(retry, int) and initial
            else None
        )
        print(
            f"  {e.get('model')} × {e.get('action')}: style={e.get('retry_prompt_style')}"
            f" reason={e.get('compression_reason')} initial={initial} retry={retry} saving={saving}"
        )

    private_path = Path(logger.private_log_file or (LOG_DIR / f"game_{game_id}.private.jsonl"))
    lines = private_path.read_text(encoding="utf-8").splitlines()
    result_line = next(json.loads(line) for line in reversed(lines) if '"game_result"' in line)
    model_stats = result_line.get("model_stats", {})
    print("\n[game_result.model_stats.action_profiles]")
    for model in sorted(model_stats):
        stats = model_stats[model]
        profiles = stats.get("action_profiles", {})
        print(f"  {model}: thinking_budget_fallbacks={stats.get('thinking_budget_fallbacks')}")
        for key in sorted(profiles):
            entry = profiles[key]
            print(
                f"    {key}: requests={entry['requests']} empty_rate={entry['empty_response_rate']}"
                f" thinking_only_rate={entry['thinking_only_rate']}"
                f" widenings={entry['thinking_allowance_widenings']}"
                f" off_switches={entry['thinking_off_switches']}"
                f" avg_elapsed={entry['avg_elapsed_seconds']}"
                f" avg_body_chars={entry['avg_body_chars']}"
            )


if __name__ == "__main__":
    main()

