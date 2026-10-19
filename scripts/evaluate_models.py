#!/usr/bin/env python3
"""TASK-005: Model evaluation harness script for Local LLM Werewolf."""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add src to sys.path if not present
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf import (
    ActionKind,
    ActionParser,
    ChatResponse,
    Message,
    OllamaClient,
    OllamaError,
    Player,
    PromptBuilder,
    PromptContext,
    Role,
    load_player_configs,
)
from werewolf.discussion_format import (
    has_discussion_output_markers,
    parse_discussion_output,
)


@dataclass
class PromptTestScenario:
    action_type: str  # "discussion", "vote", "attack", "seer"
    prompt_ctx: PromptContext
    expected_action_kind: ActionKind | None
    candidates: tuple[str, ...]
    actor_name: str


def generate_scenarios(actor: Player, all_player_names: tuple[str, ...]) -> list[PromptTestScenario]:
    """Generates standard benchmark scenarios for 4 actions: discussion, vote, attack, seer."""
    scenarios: list[PromptTestScenario] = []

    # 1. 昼の議論 (discussion)
    dummy_history = (
        Message("Player 1", "おはようございます。みなさん怪しい人はいませんか？"),
        Message("Player 2", "まだ初日なので手がかりが少ないですね。"),
    )
    disc_ctx = PromptBuilder.build_discussion_prompt(
        actor=actor,
        alive_players=all_player_names,
        conversation_history=dummy_history,
    )
    scenarios.append(
        PromptTestScenario(
            action_type="discussion",
            prompt_ctx=disc_ctx,
            expected_action_kind=None,
            candidates=(),
            actor_name=actor.name,
        )
    )

    # 2. 投票 (vote)
    vote_candidates = tuple(p for p in all_player_names if p != actor.name)
    vote_ctx = PromptBuilder.build_vote_prompt(
        actor=actor,
        alive_players=all_player_names,
        valid_candidates=vote_candidates,
        today_summary="Player 1とPlayer 2の間で意見の食い違いがありました。",
        runoff=False,
    )
    scenarios.append(
        PromptTestScenario(
            action_type="vote",
            prompt_ctx=vote_ctx,
            expected_action_kind=ActionKind.VOTE,
            candidates=vote_candidates,
            actor_name=actor.name,
        )
    )

    # 3. 襲撃対象 (attack) - 人狼役職としてテスト
    wolf_actor = Player(name=actor.name, model=actor.model, role=Role.WOLF, is_alive=True)
    partner = next(p for p in all_player_names if p != actor.name)
    wolves = (actor.name, partner)
    attack_candidates = tuple(p for p in all_player_names if p not in wolves)
    attack_ctx = PromptBuilder.build_attack_prompt(
        actor=wolf_actor,
        wolves=wolves,
        valid_candidates=attack_candidates,
    )
    scenarios.append(
        PromptTestScenario(
            action_type="attack",
            prompt_ctx=attack_ctx,
            expected_action_kind=ActionKind.ATTACK,
            candidates=attack_candidates,
            actor_name=actor.name,
        )
    )

    # 4. 占い対象 (seer) - 占い師役職としてテスト
    seer_actor = Player(name=actor.name, model=actor.model, role=Role.SEER, is_alive=True)
    seer_results = {all_player_names[0]: "human"}
    seer_candidates = all_player_names
    seer_ctx = PromptBuilder.build_seer_prompt(
        actor=seer_actor,
        alive_players=seer_candidates,
        seer_results=seer_results,
    )
    scenarios.append(
        PromptTestScenario(
            action_type="seer",
            prompt_ctx=seer_ctx,
            expected_action_kind=ActionKind.SEER,
            candidates=seer_candidates,
            actor_name=actor.name,
        )
    )

    return scenarios


def _label_for_action(action: ActionKind) -> str:
    return {
        ActionKind.ATTACK: "襲撃対象",
        ActionKind.SEER: "占い対象",
        ActionKind.VOTE: "投票",
    }[action]


def evaluate_scenario(
    client: OllamaClient,
    model: str,
    scenario: PromptTestScenario,
    options: dict[str, Any],
) -> dict[str, Any]:
    system_prompt = scenario.prompt_ctx.system_prompt
    user_prompt = scenario.prompt_ctx.user_prompt

    start_time = time.perf_counter()

    raw_responses: list[str] = []
    error_info: dict[str, Any] | None = None
    transport_retries = 0
    semantic_retries = 0

    parsed_target: str | None = None
    is_format_compliant = False
    is_parse_valid = False
    result_format = "invalid"

    current_user_prompt = user_prompt

    for attempt_idx in range(2):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": current_user_prompt},
        ]
        content: str | None = None
        attempts = 1
        try:
            resp = client.chat(model=model, messages=messages, options=options)
            if isinstance(resp, ChatResponse):
                content = resp.content
                attempts = resp.attempts
            elif isinstance(resp, str):
                content = resp
                attempts = getattr(client, "_last_attempts", 1)
                if not isinstance(attempts, int):
                    attempts = 1
            else:
                # MagicMock や未指定の戻り値
                content = str(resp) if resp is not None else ""
                attempts = getattr(client, "_last_attempts", 1)
                if not isinstance(attempts, int):
                    attempts = 1
            transport_retries += max(0, attempts - 1)
        except OllamaError as err:
            error_info = err.to_dict()
            transport_retries += max(0, err.attempts - 1)
            break
        except Exception as exc:
            error_info = {"kind": "unexpected", "message": str(exc), "attempts": 1, "status_code": None}
            break

        raw_responses.append(content)

        if scenario.expected_action_kind is not None:
            parsed_target = ActionParser.parse(content, scenario.expected_action_kind)
            is_format_compliant = parsed_target is not None
            is_parse_valid = ActionParser.validate_target(
                parsed_target,
                scenario.candidates,
                actor=scenario.actor_name if scenario.expected_action_kind == ActionKind.VOTE else None,
            )
        else:
            # discussion
            structured_output = parse_discussion_output(content)
            if has_discussion_output_markers(content):
                is_format_compliant = structured_output is not None
                is_parse_valid = (
                    structured_output is not None
                    and bool(structured_output.body.strip())
                )
                discussion_format = (
                    "two_block" if is_format_compliant else "invalid"
                )
            else:
                is_format_compliant = bool(content.strip())
                is_parse_valid = is_format_compliant
                discussion_format = "legacy" if is_parse_valid else "invalid"
            result_format = discussion_format

        if is_parse_valid:
            break

        # 形式不遵守またはパース失敗の場合に1回だけ再プロンプト
        if attempt_idx == 0:
            semantic_retries += 1
            if scenario.expected_action_kind is not None:
                current_user_prompt = (
                    f"{user_prompt}\n前回の応答は形式または候補が不正でした。"
                    f"候補から1名を選び、最後に必ず「{_label_for_action(scenario.expected_action_kind)}：名前」とだけ明示してください。"
                )
            else:
                current_user_prompt = (
                    f"{user_prompt}\n前回の応答は空または不正でした。"
                    "「【CO】なし または CO内容」の次行に"
                    "「【本文】議論本文」を出力してください。"
                )

    elapsed_seconds = round(time.perf_counter() - start_time, 4)
    total_retries = transport_retries + semantic_retries
    fallback_used = not is_parse_valid
    status = "OK" if is_parse_valid else "FAILED"

    # raw_response は後方互換性のため最終応答（またはNone）を設定
    last_raw_response = raw_responses[-1] if raw_responses else None

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "player_name": scenario.actor_name,
        "action_type": scenario.action_type,
        "discussion_format": (
            result_format if scenario.action_type == "discussion" else None
        ),
        "options": options,
        "elapsed_seconds": elapsed_seconds,
        "status": status,
        "retry_count": total_retries,
        "transport_retries": transport_retries,
        "semantic_retries": semantic_retries,
        "total_retries": total_retries,
        "fallback_used": fallback_used,
        "error": error_info,
        "raw_response": last_raw_response,
        "raw_responses": raw_responses,
        "parsed_target": parsed_target,
        "is_format_compliant": is_format_compliant,
        "is_parse_valid": is_parse_valid,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local LLM werewolf capabilities")
    parser.add_argument(
        "--models-file",
        type=Path,
        default=REPO_ROOT / "config" / "players.json",
        help="Path to players.json config file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "evaluation-results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of evaluation rounds",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds for each Ollama request",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for Ollama generation",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
        help="Context window length (num_ctx) for Ollama",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama base API URL",
    )

    args = parser.parse_args()

    # 出力先ディレクトリの確保
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    player_configs = load_player_configs(args.models_file)
    all_player_names = tuple(cfg.name for cfg in player_configs)

    client = OllamaClient(base_url=args.base_url, timeout=args.timeout, max_retries=2)
    options = {
        "temperature": args.temperature,
        "num_ctx": args.context_length,
    }

    records_by_model: dict[str, list[dict[str, Any]]] = {
        cfg.model: [] for cfg in player_configs
    }

    print(f"=== Starting Model Evaluation ===")
    print(f"Models file: {args.models_file}")
    print(f"Output directory: {output_dir}")
    print(f"Rounds: {args.rounds}, Timeout: {args.timeout}s, Temp: {args.temperature}")
    print(f"Evaluating {len(player_configs)} models...")

    for round_idx in range(1, args.rounds + 1):
        print(f"\n--- Round {round_idx} / {args.rounds} ---")
        for cfg in player_configs:
            model = cfg.model
            player_actor = Player(name=cfg.name, model=model, role=Role.VILLAGER, is_alive=True)
            scenarios = generate_scenarios(player_actor, all_player_names)

            print(f"[{cfg.name}] Testing model '{model}' ({len(scenarios)} actions)...")
            for scenario in scenarios:
                try:
                    result = evaluate_scenario(client, model, scenario, options)
                    result["round"] = round_idx
                    records_by_model[model].append(result)

                    status = "OK" if result["is_parse_valid"] else "FAILED"
                    if result["error"]:
                        status = f"ERROR: {result['error'].get('kind')}"
                    print(
                        f"  - {scenario.action_type.ljust(11)}: {status} "
                        f"({result['elapsed_seconds']}s, retries: {result['retry_count']})"
                    )
                except Exception as ex:
                    # 途中で1モデルが予期せぬ失敗をしても全体を止めない
                    print(f"  - {scenario.action_type.ljust(11)}: UNCAUGHT EXCEPTION: {ex}")
                    err_result = {
                        "round": round_idx,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": model,
                        "player_name": cfg.name,
                        "action_type": scenario.action_type,
                        "options": options,
                        "elapsed_seconds": 0.0,
                        "retry_count": 0,
                        "error": {"kind": "uncaught", "message": str(ex)},
                        "raw_response": None,
                        "parsed_target": None,
                        "is_format_compliant": False,
                        "is_parse_valid": False,
                        "system_prompt": scenario.prompt_ctx.system_prompt,
                        "user_prompt": scenario.prompt_ctx.user_prompt,
                    }
                    records_by_model[model].append(err_result)

    # 1. 各モデルごとの JSONL ファイル出力
    summary_by_model: dict[str, Any] = {}

    for model, records in records_by_model.items():
        safe_model_filename = model.replace(":", "_").replace("/", "_") + ".jsonl"
        jsonl_path = output_dir / safe_model_filename

        with jsonl_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # モデルごとの集計
        total_actions = len(records)
        format_compliant_count = sum(1 for r in records if r["is_format_compliant"])
        parse_valid_count = sum(1 for r in records if r["is_parse_valid"])
        error_count = sum(1 for r in records if r["error"] is not None)
        total_time = sum(r["elapsed_seconds"] for r in records)
        avg_time = round(total_time / total_actions, 3) if total_actions > 0 else 0.0
        transport_retries = sum(r.get("transport_retries", 0) for r in records)
        semantic_retries = sum(r.get("semantic_retries", 0) for r in records)
        total_retries = sum(r.get("total_retries", r.get("retry_count", 0)) for r in records)
        fallback_count = sum(1 for r in records if r.get("fallback_used", not r["is_parse_valid"]))

        # アクション種別ごとの内訳
        by_action: dict[str, Any] = {}
        for action_name in ("discussion", "vote", "attack", "seer"):
            act_recs = [r for r in records if r["action_type"] == action_name]
            act_total = len(act_recs)
            act_valid = sum(1 for r in act_recs if r["is_parse_valid"])
            act_compliant = sum(1 for r in act_recs if r["is_format_compliant"])
            act_transport_retries = sum(r.get("transport_retries", 0) for r in act_recs)
            act_semantic_retries = sum(r.get("semantic_retries", 0) for r in act_recs)
            act_total_retries = sum(r.get("total_retries", r.get("retry_count", 0)) for r in act_recs)
            act_fallback = sum(1 for r in act_recs if r.get("fallback_used", not r["is_parse_valid"]))
            by_action[action_name] = {
                "total": act_total,
                "format_compliant": act_compliant,
                "parse_valid": act_valid,
                "transport_retries": act_transport_retries,
                "semantic_retries": act_semantic_retries,
                "total_retries": act_total_retries,
                "fallback_count": act_fallback,
                "format_compliance_rate": round(act_compliant / act_total, 3) if act_total else 0.0,
                "parse_success_rate": round(act_valid / act_total, 3) if act_total else 0.0,
            }

        summary_by_model[model] = {
            "total_actions": total_actions,
            "format_compliance_rate": round(format_compliant_count / total_actions, 3) if total_actions else 0.0,
            "parse_success_rate": round(parse_valid_count / total_actions, 3) if total_actions else 0.0,
            "parse_failure_rate": round((total_actions - parse_valid_count) / total_actions, 3) if total_actions else 0.0,
            "error_count": error_count,
            "transport_retries": transport_retries,
            "semantic_retries": semantic_retries,
            "total_retries": total_retries,
            "fallback_count": fallback_count,
            "average_response_seconds": avg_time,
            "options": options,
            "by_action": by_action,
        }

    # 全体サマリ計算
    total_transport_retries = sum(m["transport_retries"] for m in summary_by_model.values())
    total_semantic_retries = sum(m["semantic_retries"] for m in summary_by_model.values())
    overall_total_retries = sum(m["total_retries"] for m in summary_by_model.values())
    total_fallback_count = sum(m["fallback_count"] for m in summary_by_model.values())

    # 2. summary.json 出力
    overall_summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models_file": str(args.models_file),
        "rounds": args.rounds,
        "timeout": args.timeout,
        "transport_retries": total_transport_retries,
        "semantic_retries": total_semantic_retries,
        "total_retries": overall_total_retries,
        "fallback_count": total_fallback_count,
        "options": options,
        "models": summary_by_model,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    print("\n=== Evaluation Complete ===")
    print(f"Summary saved to: {summary_path}")
    print(f"Per-model logs saved to: {output_dir}/*.jsonl")


if __name__ == "__main__":
    main()
