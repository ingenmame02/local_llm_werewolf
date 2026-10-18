#!/usr/bin/env python3
"""TASK-006: CLI runner for Local LLM Werewolf."""

import argparse
import json
import os
import signal
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from werewolf import (
    ActionPolicy,
    GameEngine,
    GameState,
    Message,
    MockActions,
    OllamaActions,
    OllamaClient,
    OllamaError,
    Phase,
    Player,
    Role,
    create_game,
    load_player_configs,
)

# フェーズ表示用ラベル
PHASE_NAMES_JA = {
    Phase.NIGHT_WOLF: "夜（人狼襲撃）",
    Phase.NIGHT_SEER: "夜（占い）",
    Phase.MORNING: "朝（襲撃結果発表）",
    Phase.DAY_DISCUSSION: "昼（議論）",
    Phase.VOTING: "投票（処刑）",
    Phase.ENDED: "ゲーム終了",
}

ROLE_NAMES_JA = {
    Role.WOLF: "人狼",
    Role.MADMAN: "狂人",
    Role.SEER: "占い師",
    Role.VILLAGER: "村人",
}


class GameLogger:
    def __init__(self, game_id: str, log_dir: Path, metadata: dict[str, Any]) -> None:
        self.game_id = game_id
        self.log_dir = log_dir
        self.log_file = log_dir / f"game_{game_id}.jsonl"
        self.metadata = metadata
        self._written = False

        self.log_dir.mkdir(parents=True, exist_ok=True)
        # 1行目にゲーム開始メタデータを書き込み
        init_record = {
            "type": "game_init",
            "game_id": self.game_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        self.append_record(init_record)

    def append_record(self, record: dict[str, Any]) -> None:
        if "timestamp" not in record:
            record["timestamp"] = datetime.now(timezone.utc).isoformat()
        record["game_id"] = self.game_id
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finalize(self, state: GameState, traces: list[Any] | None = None, error: str | None = None) -> None:
        if self._written:
            return
        self._written = True

        players_data = [
            {
                "name": p.name,
                "model": p.model,
                "role": p.role.value,
                "is_alive": p.is_alive,
                "partner": p.partner,
            }
            for p in state.players
        ]
        conversation_data = [{"speaker": m.speaker, "content": m.content} for m in state.conversation_log]
        wolf_chat_data = [{"speaker": m.speaker, "content": m.content} for m in state.wolf_chat_log]

        traces_data = []
        if traces:
            for t in traces:
                traces_data.append(
                    {
                        "action": getattr(t, "action", str(t)),
                        "player": getattr(t, "player", ""),
                        "raw_responses": getattr(t, "raw_responses", ()),
                        "parsed_targets": getattr(t, "parsed_targets", ()),
                        "selected_target": getattr(t, "selected_target", None),
                        "fallback_used": getattr(t, "fallback_used", False),
                        "error": getattr(t, "error", None),
                    }
                )

        summary_record = {
            "type": "game_result",
            "game_id": self.game_id,
            "day": state.day,
            "phase": state.phase.value,
            "winner": state.winner,
            "error": error,
            "players": players_data,
            "alive_players": state.alive_players,
            "conversation_log": conversation_data,
            "wolf_chat_log": wolf_chat_data,
            "seer_results": state.seer_results,
            "votes": state.votes,
            "traces": traces_data,
        }
        self.append_record(summary_record)


def print_banner(text: str) -> None:
    print(f"\n{'=' * 60}\n{text}\n{'=' * 60}")


def print_round_header(day: int, phase: Phase, alive: list[str]) -> None:
    phase_str = PHASE_NAMES_JA.get(phase, phase.value)
    print(f"\n--- Day {day} : {phase_str} ---")
    print(f"生存者 ({len(alive)}名): {', '.join(alive)}")


def run_cli_game(args: argparse.Namespace) -> int:
    game_id = uuid.uuid4().hex[:12]
    log_dir = Path(args.log_dir)

    print_banner(f"人狼ゲーム CLI (Game ID: {game_id})")
    print(f"モード: {args.mode}")
    print(f"シード値: {args.seed}")
    print(f"プレイヤー設定: {args.models_file}")
    print(f"ログ保存先: {log_dir / f'game_{game_id}.jsonl'}")

    options = {
        "temperature": args.temperature,
        "num_ctx": args.num_ctx,
    }

    # ゲーム初期化
    state = create_game(seed=args.seed, config_path=Path(args.models_file))

    meta = {
        "mode": args.mode,
        "seed": args.seed,
        "options": options,
        "timeout": args.timeout,
        "base_url": args.base_url,
    }
    logger = GameLogger(game_id, log_dir, meta)

    # クライアントとポリシーの設定
    client: OllamaClient | None = None
    actions: ActionPolicy
    if args.mode == "ollama":
        client = OllamaClient(base_url=args.base_url, timeout=args.timeout, max_retries=2)
        if not client.ping():
            err_msg = f"Ollamaサーバー（{args.base_url}）に接続できません。Ollamaが起動しているか確認してください。"
            print_banner(f"[エラー] {err_msg}")
            logger.finalize(state, traces=None, error=err_msg)
            print(f"ログを保存しました: {logger.log_file}")
            return 1
        actions = OllamaActions(state=state, client=client, seed=args.seed, options=options)
    else:
        actions = MockActions()

    engine = GameEngine(state, actions, seed=args.seed)

    def cleanup_and_exit(signum=None, frame=None):
        print("\n\n[中断] シグナルを受信しました。ログを保存して終了します...")
        traces = getattr(actions, "traces", None)
        logger.finalize(state, traces=traces, error="interrupted")
        print(f"ログ保存完了: {logger.log_file}")
        sys.exit(130)

    # Ctrl+C ハンドラ
    try:
        signal.signal(signal.SIGINT, cleanup_and_exit)
    except (ValueError, AttributeError):
        pass  # Windowsや特定スレッドでsignal設定が制限される場合を考慮

    try:
        # 開始前の勝敗判定
        engine.check_winner()

        while state.winner is None:
            # 1. 夜フェーズ（人狼襲撃）
            print_round_header(state.day, Phase.NIGHT_WOLF, state.alive_players)
            engine.run_wolf_phase()
            if state.pending_attack:
                logger.append_record({
                    "type": "action_pending_attack",
                    "day": state.day,
                    "target": state.pending_attack,
                })
            if state.winner is not None:
                break

            # 2. 夜フェーズ（占い師）
            engine.run_seer_phase()
            logger.append_record({
                "type": "seer_results",
                "day": state.day,
                "results": dict(state.seer_results),
            })

            # 3. 朝フェーズ（犠牲者発表）
            prev_alive = set(state.alive_players)
            engine.resolve_morning()
            dead_in_night = list(prev_alive - set(state.alive_players))
            print_round_header(state.day, Phase.MORNING, state.alive_players)
            if dead_in_night:
                print(f"昨夜の犠牲者: {', '.join(dead_in_night)}")
            else:
                print("昨夜の犠牲者はいませんでした。")
            logger.append_record({
                "type": "morning_casualty",
                "day": state.day,
                "dead": dead_in_night,
                "alive": state.alive_players,
            })
            if state.winner is not None:
                break

            # 4. 昼フェーズ（議論）
            print_round_header(state.day, Phase.DAY_DISCUSSION, state.alive_players)
            prev_log_len = len(state.conversation_log)
            engine.run_discussion()
            # 新規会話の表示
            new_msgs = state.conversation_log[prev_log_len:]
            for msg in new_msgs:
                print(f"  [{msg.speaker}]: {msg.content}")
                logger.append_record({
                    "type": "discussion_message",
                    "day": state.day,
                    "speaker": msg.speaker,
                    "content": msg.content,
                })

            # 5. 投票フェーズ（処刑）
            prev_alive = set(state.alive_players)
            vote_day = state.day
            engine.run_voting()
            dead_in_vote = list(prev_alive - set(state.alive_players))
            print(f"\n--- 投票結果 (Day {vote_day}) ---")
            for voter, target in state.votes.items():
                print(f"  {voter} -> {target}")
            if dead_in_vote:
                print(f"処刑されたプレイヤー: {', '.join(dead_in_vote)}")
            print(f"生存者 ({len(state.alive_players)}名): {', '.join(state.alive_players)}")
            logger.append_record({
                "type": "voting_result",
                "day": vote_day,
                "votes": state.votes,
                "executed": dead_in_vote,
                "alive": state.alive_players,
            })

    except KeyboardInterrupt:
        cleanup_and_exit()
    except OllamaError as err:
        print_banner(f"[エラー] Ollama通信エラーが発生しました: {err}")
        traces = getattr(actions, "traces", None)
        logger.finalize(state, traces=traces, error=str(err))
        print(f"ログを保存しました: {logger.log_file}")
        return 1
    except Exception as exc:
        print_banner(f"[エラー] 予期せぬエラーが発生しました: {exc}")
        traces = getattr(actions, "traces", None)
        logger.finalize(state, traces=traces, error=str(exc))
        print(f"ログを保存しました: {logger.log_file}")
        return 1

    # ゲーム終了表示
    winner_str = "村人陣営 (village)" if state.winner == "village" else "人狼陣営 (werewolf)"
    print_banner(f"ゲーム終了！ 勝者: {winner_str}")

    print("\n【最終プレイヤー一覧】")
    for p in state.players:
        status_str = "生存" if p.is_alive else "死亡"
        role_str = ROLE_NAMES_JA.get(p.role, p.role.value)
        print(f"  - {p.name.ljust(10)}: {role_str.ljust(6)} ({p.model}) [{status_str}]")

    traces = getattr(actions, "traces", None)
    logger.finalize(state, traces=traces)
    print(f"\nゲームID: {game_id}")
    print(f"JSONLログ保存先: {logger.log_file}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Werewolf Game CLI")
    parser.add_argument(
        "--mode",
        choices=["mock", "ollama"],
        default="mock",
        help="Game execution mode ('mock' or 'ollama')",
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        default=REPO_ROOT / "config" / "players.json",
        help="Path to players.json configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable role assignment and tie-breaking",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout for Ollama (seconds)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for LLM generation",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=4096,
        help="Context window length for Ollama (num_ctx)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=REPO_ROOT / "logs",
        help="Directory to save game JSONL logs",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API base URL",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = run_cli_game(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
