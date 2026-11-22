#!/usr/bin/env python3
"""TASK-005: Model evaluation harness script for Local LLM Werewolf."""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add src to sys.path if not present
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf import (
    ACTION_BUDGET_DEFAULT_LABELS,
    ACTION_BUDGET_KEYS,
    ACTION_BUDGET_RESOLUTION_PRIORITY,
    DEFAULT_ACTION_NUM_PREDICT,
    DEFAULT_THINKING_BY_ACTION,
    MINIMAL_ACTION_MAX_NUM_PREDICT,
    THINKING_AFFORDABLE_NUM_PREDICT,
    THINKING_BUDGET_SOURCE,
    THINKING_CAPABILITY_SOURCE,
    THINKING_DEFAULT_LABELS,
    THINKING_RESOLUTION_PRIORITY,
    ActionKind,
    ActionParser,
    ChatResponse,
    GameState,
    LedgerFact,
    Message,
    OllamaClient,
    OllamaError,
    PERSONALITY_CHOICE_KEYS,
    Player,
    PromptBuilder,
    PromptContext,
    PublicRoleClaim,
    Role,
    apply_thinking_floor,
    create_game,
    load_player_configs,
    parse_action_num_predict_pairs,
    parse_thinking_by_action_pairs,
    resolve_action_num_predict,
    resolve_thinking,
)
from werewolf.discussion_format import (
    has_discussion_output_markers,
    parse_discussion_output,
)
from werewolf.engine import extract_public_seer_result, is_public_seer_claim
from werewolf.llm_actions import fit_dynamic_context
from werewolf.thinking_profile import (
    ThinkingProfiles,
    reserve_body_budget,
)


@dataclass
class PromptTestScenario:
    action_type: str  # "discussion", "vote", "attack", "seer", "discussion_seer_co"
    prompt_ctx: PromptContext
    expected_action_kind: ActionKind | None
    candidates: tuple[str, ...]
    actor_name: str
    player_names: tuple[str, ...] = ()
    # 占い師COシナリオで、そのモデルが実際に持っている占い結果（本番のフォールバック評価用）
    expected_seer_result: tuple[str, str] | None = None
    scenario_role: str | None = None
    personality_profile: str = "balanced"
    question_expected: bool = False
    questioner_name: str | None = None
    prior_suspicion_reasons: tuple[str, ...] = ()


# 評価シナリオの action_type と行動別生成上限キーの対応
ACTION_TYPE_TO_BUDGET_KEY: dict[str, str] = {
    "discussion": "discussion",
    "vote": "vote",
    "attack": "attack",
    "seer": "seer",
    "discussion_seer_co": "co_correction",
}

# 生成起因の失敗（本番では再試行時に num_predict を増やす）
GENERATION_ISSUE_REASONS: frozenset[str] = frozenset(
    {"empty_content", "thinking_only_response", "generation_truncated"}
)

# 失敗理由別の短い再要求文言（本番の再要求方針と同じ趣旨）
RETRY_HINTS: dict[str, str] = {
    "empty_content": (
        "Your previous final answer was empty. Output no thinking; produce only a short final answer."
    ),
    "thinking_only_response": (
        "Your previous response was only thinking with no final answer. Output no thinking; write only the final answer."
    ),
    "generation_truncated": (
        "Your previous response was cut off by the generation limit. Output only the conclusion, short and complete."
    ),
    "discussion_output_format": "Your previous format was invalid.",
    "seer_result_without_claim": (
        "Your previous response had no explicit Seer CO. Claim Seer CO explicitly in the same speech."
    ),
    "invalid_target": (
        "Your previous target was outside the candidates. Choose only from eligible candidates."
    ),
    "invalid_action_response": "Your previous target could not be parsed.",
    "empty_body": "Your previous body was empty. Make a substantive statement.",
}

# Discussion retries keep the volume and require 5-8 sentences covering 4 elements (TASK-024).
# Target-selection and CO-correction retries stay in the short RETRY_HINTS form.
DISCUSSION_RETRY_HINTS: dict[str, str] = {
    "empty_content": (
        "Your previous body was empty. Output no thinking; write 5-8 sentences in [BODY]."
    ),
    "thinking_only_response": (
        "Your previous response had thinking but no body. Output no thinking; "
        "write 5-8 sentences in [BODY]."
    ),
    "generation_truncated": (
        "Your previous response was cut off by the generation limit. Keep the structured format and "
        "write [BODY] in 5-8 sentences without stopping, to the end."
    ),
    "discussion_output_format": "Your previous format was invalid.",
    "empty_body": "Your previous body was empty. Make a substantive statement in 5-8 sentences.",
}


def _retry_hint(scenario: PromptTestScenario, issue: str) -> str:
    """Short form for target selection / CO correction, 5-8 sentences for day discussion."""
    if scenario.expected_action_kind is None and scenario.action_type == "discussion":
        discussion_hint = DISCUSSION_RETRY_HINTS.get(issue)
        if discussion_hint is not None:
            return discussion_hint
    return RETRY_HINTS.get(issue, "Please correct your previous response.")


def escalate_num_predict(current: int | None, cap: int = 8192) -> int:
    """生成起因の再試行で使う生成上限（倍へ増やす、上限8192）。"""
    if isinstance(current, bool) or not isinstance(current, int) or current <= 0:
        return min(1024, cap)
    return min(max(current * 2, 512), cap)


def classify_attempt_issue(
    content: str,
    metadata: dict[str, object],
    scenario: PromptTestScenario,
    result_format: str,
    parsed_target: str | None,
    co_established: bool | None,
    seer_result_published: bool | None,
) -> str:
    """失敗した応答の理由を本番と同じ分類名へマップする。"""
    if metadata.get("done_reason") == "length":
        return "generation_truncated"
    if not content.strip():
        return (
            "thinking_only_response"
            if metadata.get("thinking_present")
            else "empty_content"
        )
    if scenario.expected_action_kind is not None:
        return "invalid_target" if parsed_target is not None else "invalid_action_response"
    if scenario.action_type == "discussion_seer_co":
        if seer_result_published and not co_established:
            return "seer_result_without_claim"
        if result_format == "invalid":
            return "discussion_output_format"
        return "co_missing"
    if result_format == "invalid":
        return "discussion_output_format"
    return "empty_body"


def _average_metric(values: list[Any]) -> float:
    """整数メトリクスの平均（None・負値を除外）。"""
    numeric = [
        value
        for value in values
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    ]
    return round(sum(numeric) / len(numeric), 2) if numeric else 0.0


def _metadata_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """応答メタデータから空応答・思考のみ・生成切り詰めを数える。"""
    empty = thinking_only = truncated = 0
    for rec in records:
        metas = rec.get("response_metadata")
        if not isinstance(metas, list):
            metas = []
        dicts = [m for m in metas if isinstance(m, dict)]
        if any(not m.get("content_length") for m in dicts):
            empty += 1
        if any(m.get("thinking_present") for m in dicts):
            thinking_only += 1
        if any(m.get("done_reason") == "length" for m in dicts):
            truncated += 1
    return {
        "empty_response_count": empty,
        "thinking_only_count": thinking_only,
        "truncated_count": truncated,
    }


def record_harness_attempts(
    profiles: ThinkingProfiles,
    model: str,
    budget_key: str,
    result: dict[str, Any],
) -> dict[str, object]:
    """1シナリオの評価結果からモデル×アクション別の生成観測を加算する（TASK-053）。

    本番の `llm_actions._record_generation_response` と同じ形に揃え、本文が空のまま
    打ち切られた送信を thinking 消費量の下限証拠として残す。空が続いた場合は本番同様
    モデル別余白を拡大し、次のシナリオ・ラウンドの初回リクエストへ反映する。
    """

    metas = [meta for meta in (result.get("response_metadata") or []) if isinstance(meta, dict)]
    attempts = metas or [{}]
    elapsed_total = float(result.get("elapsed_seconds") or 0.0)
    per_attempt_seconds = elapsed_total / max(1, len(attempts))
    error_info = result.get("error") if isinstance(result.get("error"), dict) else {}
    error_kind = error_info.get("kind")
    profile = profiles.get_or_create(model, budget_key)
    sent_predict = result.get("num_predict")
    sent_value = (
        int(sent_predict)
        if isinstance(sent_predict, int) and not isinstance(sent_predict, bool)
        else None
    )
    for index, meta in enumerate(attempts, start=1):
        content_length = meta.get("content_length")
        body_chars = (
            int(content_length)
            if isinstance(content_length, int) and not isinstance(content_length, bool)
            else 0
        )
        thinking_present = bool(meta.get("thinking_present"))
        empty = body_chars == 0
        is_last = index == len(attempts)
        profiles.record(
            model,
            budget_key,
            request_number=index,
            body_char_count=body_chars,
            thinking_present=thinking_present,
            done_reason=meta.get("done_reason"),
            eval_count=meta.get("eval_count"),
            elapsed_seconds=per_attempt_seconds,
            empty=empty and not thinking_present,
            thinking_only=empty and thinking_present,
            timed_out=bool(is_last and error_kind == "timeout"),
            transport_error=bool(is_last and error_info and error_kind != "timeout"),
        )
        if empty and thinking_present and meta.get("done_reason") == "length":
            # 思考が生成上限を食い潰した → まずモデル別余白を拡大（本番と同じ段階）。
            profile.widen_allowance(num_predict_sent=sent_value)
    return profile.serialize()


def _failure_reason_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """失敗理由別の発生回数（モデル別の改善対象特定用）。"""
    counts: dict[str, int] = {}
    for rec in records:
        reasons = rec.get("failure_reasons")
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            if isinstance(reason, str):
                counts[reason] = counts.get(reason, 0) + 1
    return counts


def _request_attempts(records: list[dict[str, Any]]) -> int:
    """実際にOllamaへ送信した回数（再要求を含む）。"""
    total = 0
    for rec in records:
        attempts = rec.get("request_attempts")
        if isinstance(attempts, int) and not isinstance(attempts, bool):
            total += attempts
        else:
            total += len(rec.get("raw_responses") or [])
    return total


def _dynamic_context_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """動的コンテキストの調整回数と実際に送ったnum_ctxの分布（TASK-036）。"""
    sent = [
        rec["num_ctx_sent"]
        for rec in records
        if isinstance(rec.get("num_ctx_sent"), int)
        and not isinstance(rec.get("num_ctx_sent"), bool)
    ]
    return {
        "dynamic_context_fit_requests": sum(
            1 for rec in records if rec.get("dynamic_context_fit")
        ),
        "avg_num_ctx_sent": round(sum(sent) / len(sent), 1) if sent else None,
        "min_num_ctx_sent": min(sent) if sent else None,
        "max_num_ctx_sent": max(sent) if sent else None,
    }


def evaluate_specificity(
    content: str,
    scenario: PromptTestScenario,
    parsed_target: str | None,
    co_established: bool | None,
) -> bool:
    """応答が具体的根拠（CO・占い結果・対象名など）を含んでいるかを判定する。"""
    if scenario.expected_action_kind is not None:
        return parsed_target is not None
    if scenario.action_type == "discussion_seer_co":
        return bool(co_established)
    names = scenario.player_names
    if has_discussion_output_markers(content):
        structured = parse_discussion_output(content)
        if structured is None:
            return False
        if is_public_seer_claim(structured.claim):
            return True
        return any(name in structured.body for name in names)
    return bool(content.strip()) and any(name in content for name in names)


def extract_discussion_signals(
    content: str, player_names: tuple[str, ...]
) -> dict[str, list[str]]:
    """議論本文から質問・疑い・投票提案の対象を抽出する（語彙ではなく対象選択を見る）。"""
    structured = parse_discussion_output(content)
    body = structured.body if structured is not None else content
    signals: dict[str, set[str]] = {
        "question_targets": set(),
        "suspicion_targets": set(),
        "suspicion_reasons": set(),
        "vote_proposals": set(),
    }
    for line in body.splitlines():
        mentioned = {name for name in player_names if name in line}
        if (
            "?" in line
            or "？" in line
            or any(
                cue in line
                for cue in ("質問", "聞きたい", "尋ね", "確認したい")
            )
            or any(
                cue in line.lower()
                for cue in ("ask", "question", "tell me", "check", "explain", "?")
            )
        ):
            signals["question_targets"].update(mentioned)
        suspicion_cues = (
            "疑",
            "怪しい",
            "人狼",
            "黒だ",
            "黒と思",
            "suspect",
            "suspicious",
            "werewolf",
            "wolf",
        )
        if any(cue in line.lower() for cue in suspicion_cues):
            signals["suspicion_targets"].update(mentioned)
            reason_categories = {
                "inconsistent_statements": (
                    "矛盾",
                    "変わ",
                    "一貫",
                    "前と違",
                    "inconsisten",
                    "contradict",
                    "changed",
                    "consistent",
                ),
                "voting_behavior": ("投票", "票", "vote", "ballot"),
                "role_claim_evidence": (
                    "CO",
                    "占い",
                    "結果",
                    "判定",
                    "seer",
                    "divination",
                    "result",
                    "claim",
                ),
                "evasive_response": (
                    "回答",
                    "質問",
                    "避け",
                    "答え",
                    "answer",
                    "avoid",
                    "reply",
                ),
                "discussion_behavior": (
                    "発言",
                    "態度",
                    "様子",
                    "反応",
                    "speech",
                    "reaction",
                    "attitude",
                    "statement",
                ),
            }
            matched_reasons = {
                reason
                for reason, cues in reason_categories.items()
                if any(cue in line.lower() for cue in cues)
            }
            signals["suspicion_reasons"].update(matched_reasons or {"other"})
        if any(
            cue in line.lower()
            for cue in (
                "投票を提案",
                "投票する",
                "投票します",
                "投票すべき",
                "投票先",
                "処刑",
                "吊る",
                "vote for",
                "propose voting",
                "vote target",
                "should vote",
                "execute",
                "exile",
                "hang",
            )
        ):
            signals["vote_proposals"].update(mentioned)
    return {key: sorted(targets) for key, targets in signals.items()}


def summarize_discussion_signals(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """対象選択の分布と多様性をモデル・個性別に集計する。"""
    distributions: dict[str, dict[str, int]] = {
        "question_targets": {},
        "suspicion_targets": {},
        "suspicion_reasons": {},
        "vote_proposals": {},
    }
    for record in records:
        signals = record.get("discussion_signals")
        if not isinstance(signals, dict):
            continue
        for signal_name, counts in distributions.items():
            targets = signals.get(signal_name)
            if not isinstance(targets, list):
                continue
            for target in targets:
                if isinstance(target, str):
                    counts[target] = counts.get(target, 0) + 1
    summary: dict[str, dict[str, Any]] = {}
    for signal_name, counts in distributions.items():
        count_key, diversity_key = (
            ("reason_counts", "distinct_reasons")
            if signal_name == "suspicion_reasons"
            else ("target_counts", "distinct_targets")
        )
        summary[signal_name] = {
            count_key: dict(sorted(counts.items())),
            diversity_key: len(counts),
        }
    return summary


def evaluate_seer_co_response(
    content: str, player_names: tuple[str, ...]
) -> tuple[bool, bool]:
    """応答から「明示的な占い師CO」と「占い結果の公開」の成立を判定する。"""
    structured = parse_discussion_output(content)
    if structured is not None:
        claim_text = structured.claim
        result_text = "\n".join(
            part
            for part in (structured.claim, structured.result, structured.body)
            if part
        )
    else:
        claim_text = content
        result_text = content
    co_established = is_public_seer_claim(claim_text)
    result_published = extract_public_seer_result(result_text, player_names) is not None
    return co_established, result_published


def generate_scenarios(
    actor: Player,
    all_player_names: tuple[str, ...],
    context_strategy: str = "hybrid_retrieval",
    reasoning_memory: bool = False,
) -> list[PromptTestScenario]:
    """Generates standard benchmark scenarios for 4 actions: discussion, vote, attack, seer."""
    scenarios: list[PromptTestScenario] = []

    # 1. 昼の議論 (discussion)
    questioner = next(
        (name for name in all_player_names if name != actor.name), actor.name
    )
    vote_target = next(
        (
            name
            for name in all_player_names
            if name not in {actor.name, questioner}
        ),
        questioner,
    )
    dummy_history = (
        Message("system", "Day 1 begins."),
        Message(
            questioner,
            f"Based on the reason for voting {vote_target}, I want to reconsider my vote target.",
        ),
        *tuple(
            Message(
                all_player_names[index % len(all_player_names)],
                f"Day 1 discussion record {index}: no new information in particular.",
            )
            for index in range(20)
        ),
        Message("system", "Day 2 begins."),
        Message(
            questioner,
            f"{actor.name}, how do you think about your vote reason for {vote_target} yesterday?",
        ),
    )
    discussion_state = create_game(seed=42)
    discussion_state.reasoning_memory_enabled = reasoning_memory
    state_actor = next(
        (player for player in discussion_state.players if player.name == actor.name),
        None,
    )
    if state_actor is not None:
        state_actor.role = actor.role
        state_actor.partner = actor.partner
    discussion_state.day = 2
    discussion_state.vote_history.append((1, False, {questioner: vote_target}))
    dead_names = tuple(
        name
        for name in reversed(all_player_names)
        if name not in {actor.name, questioner, vote_target}
    )[:2]
    for player in discussion_state.players:
        if player.name in dead_names:
            player.is_alive = False
    alive_names = tuple(
        name for name in all_player_names if name not in dead_names
    )
    if dead_names:
        discussion_state.attack_deaths.append((2, dead_names[0]))
    if len(dead_names) > 1:
        discussion_state.executions.append((1, dead_names[1]))
    discussion_state.public_role_claims.append(
        PublicRoleClaim(
            day=1,
            claimant=questioner,
            claimed_role=Role.SEER.value,
            target=vote_target,
            result="human",
        )
    )
    discussion_state.fact_ledger.confirmed_facts.append(
        LedgerFact(
            day=1,
            kind="vote",
            description=f"{questioner} voted for {vote_target}.",
        )
    )
    if reasoning_memory:
        discussion_state.refresh_reasoning_memos()
    prior_signals = extract_discussion_signals(
        "\n".join(
            message.content
            for message in dummy_history
            if message.speaker != "system"
        ),
        all_player_names,
    )
    disc_ctx = PromptBuilder.build_discussion_prompt(
        actor=actor,
        alive_players=alive_names,
        conversation_history=dummy_history,
        day=2,
        state=discussion_state,
        context_strategy=context_strategy,
    )
    scenarios.append(
        PromptTestScenario(
            action_type="discussion",
            prompt_ctx=disc_ctx,
            expected_action_kind=None,
            candidates=(),
            actor_name=actor.name,
            player_names=all_player_names,
            scenario_role=actor.role.value,
            personality_profile=actor.personality_profile,
            question_expected=True,
            questioner_name=questioner,
            prior_suspicion_reasons=tuple(prior_signals["suspicion_reasons"]),
        )
    )

    # 2. 投票 (vote)
    vote_candidates = tuple(p for p in all_player_names if p != actor.name)
    vote_ctx = PromptBuilder.build_vote_prompt(
        actor=actor,
        alive_players=all_player_names,
        valid_candidates=vote_candidates,
        today_summary="Player 1 and Player 2 disagreed about the target.",
        runoff=False,
    )
    scenarios.append(
        PromptTestScenario(
            action_type="vote",
            prompt_ctx=vote_ctx,
            expected_action_kind=ActionKind.VOTE,
            candidates=vote_candidates,
            actor_name=actor.name,
            scenario_role=actor.role.value,
            personality_profile=actor.personality_profile,
        )
    )

    # 3. 襲撃対象 (attack) - 人狼役職としてテスト
    wolf_actor = Player(
        name=actor.name,
        model=actor.model,
        role=Role.WOLF,
        is_alive=True,
        personality_profile=actor.personality_profile,
    )
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
            scenario_role=wolf_actor.role.value,
            personality_profile=actor.personality_profile,
        )
    )

    # 4. 占い対象 (seer) - 占い師役職としてテスト
    seer_actor = Player(
        name=actor.name,
        model=actor.model,
        role=Role.SEER,
        is_alive=True,
        personality_profile=actor.personality_profile,
    )
    scanned_target = next(name for name in all_player_names if name != actor.name)
    seer_results = {scanned_target: "human"}
    seer_state_players = [
        Player(
            name=name,
            model=actor.model,
            role=Role.SEER if name == actor.name else Role.VILLAGER,
            personality_profile=actor.personality_profile,
        )
        for name in all_player_names
    ]
    seer_state = GameState(players=seer_state_players, day=2)
    seer_state.seer_results.update(seer_results)
    seer_state.seer_result_history.append((1, scanned_target, "human"))
    seer_candidates = tuple(name for name in all_player_names if name != scanned_target)
    seer_ctx = PromptBuilder.build_seer_prompt(
        actor=seer_actor,
        alive_players=all_player_names,
        seer_results=seer_results,
        day=2,
        state=seer_state,
    )
    scenarios.append(
        PromptTestScenario(
            action_type="seer",
            prompt_ctx=seer_ctx,
            expected_action_kind=ActionKind.SEER,
            candidates=seer_candidates,
            actor_name=actor.name,
            scenario_role=seer_actor.role.value,
            personality_profile=actor.personality_profile,
        )
    )

    # 5. 未公開の人狼判定を持つ占い師の昼議論（TASK-022 CO評価）
    seer_state = GameState(
        players=[
            Player(
                name=name,
                model=actor.model,
                role=Role.VILLAGER,
                is_alive=True,
            )
            for name in all_player_names
        ]
    )
    seer_speaker = next(
        player for player in seer_state.players if player.name == actor.name
    )
    seer_speaker.role = Role.SEER
    seer_speaker.personality_profile = actor.personality_profile
    co_target = next(name for name in all_player_names if name != actor.name)
    seer_state.seer_results[co_target] = "wolf"
    seer_state.seer_result_history.append((1, co_target, "wolf"))
    seer_state.discussion_turn = 1
    co_ctx = PromptBuilder.build_discussion_prompt(
        actor=seer_speaker,
        alive_players=all_player_names,
        conversation_history=dummy_history,
        day=1,
        state=seer_state,
        context_strategy=context_strategy,
    )
    scenarios.append(
        PromptTestScenario(
            action_type="discussion_seer_co",
            prompt_ctx=co_ctx,
            expected_action_kind=None,
            candidates=(),
            actor_name=actor.name,
            player_names=all_player_names,
            expected_seer_result=(co_target, "wolf"),
            scenario_role=seer_speaker.role.value,
            personality_profile=actor.personality_profile,
        )
    )

    return scenarios


def _label_for_action(action: ActionKind) -> str:
    return {
        ActionKind.ATTACK: "Attack target",
        ActionKind.SEER: "Seer target",
        ActionKind.VOTE: "Vote",
    }[action]


def _compact_retry_prompt(scenario: PromptTestScenario, issue: str) -> str:
    """Do not resend the long prompt; build a short retry with only the failure reason and required items (TASK-024)."""
    hint = _retry_hint(scenario, issue)
    lines = [
        f"[Retry] {hint}",
        f"- Action: {scenario.action_type}",
        f"- You: {scenario.actor_name}",
    ]
    if scenario.expected_action_kind is not None:
        lines.append(f"- Eligible candidates: {', '.join(scenario.candidates)}")
        lines.append(
            "Choose exactly 1 candidate and output only one line in the format "
            f"'{_label_for_action(scenario.expected_action_kind)}: Player Name'."
        )
    elif scenario.action_type == "discussion_seer_co":
        if scenario.expected_seer_result is not None:
            target, result = scenario.expected_seer_result
            label = "Werewolf" if result == "wolf" else "Human"
            lines.append(f"- Your actual divination target: {target} / Result: {label}")
        lines.append(
            "Output only 2 lines: '[CO] Seer' then "
            "'[RESULT] Target=Player X Result=Werewolf/Human'."
        )
    else:
        lines.append(
            "Include all 4 elements: the answer to the previous speech, verifiable facts, "
            "the target you suspect with its basis, and the next point to check."
        )
        lines.append(
            "Use the 2-line structure: '[CO] None or CO content' then "
            "'[BODY] discussion body'; write [BODY] in 5-8 sentences to the end."
        )
    if scenario.expected_action_kind is None and scenario.action_type == "discussion":
        lines.append(
            "Output strictly in English, with no thinking or preamble: only the final answer "
            "containing the answer to the question, facts, suspicion basis, and next checks."
        )
    else:
        lines.append(
            "Output strictly in English, short, with no thinking or preamble; only the final answer."
        )
    return "\n".join(lines)


def _is_thinking_unsupported(error: OllamaError) -> bool:
    """Ollamaが思考生成（think）非対応を理由に400を返したかどうか（本番と同じ判定）。"""
    message = error.message.casefold()
    return any(
        marker in message
        for marker in (
            "does not support thinking",
            "doesn't support thinking",
            "thinking is not supported",
            "thinking not supported",
            "does not support reasoning",
        )
    )


def evaluate_scenario(
    client: OllamaClient,
    model: str,
    scenario: PromptTestScenario,
    options: dict[str, Any],
    thinking_supported: dict[str, bool] | None = None,
    thinking_budget_ok: dict[str, bool] | None = None,
    dynamic_context: bool = True,
    max_context_length: int | None = None,
) -> dict[str, Any]:
    system_prompt = scenario.prompt_ctx.system_prompt
    user_prompt = scenario.prompt_ctx.user_prompt

    start_time = time.perf_counter()

    raw_responses: list[str] = []
    error_info: dict[str, Any] | None = None
    transport_retries = 0
    semantic_retries = 0

    parsed_target: str | None = None
    invalid_target = False
    is_format_compliant = False
    is_parse_valid = False
    result_format = "invalid"
    co_established: bool | None = None
    seer_result_published: bool | None = None
    is_specific = False
    eval_count: int | None = None
    eval_duration_ms: int | None = None
    prompt_eval_count: int | None = None

    current_user_prompt = user_prompt
    attempt_options = dict(options)
    num_predict_by_attempt: list[int | None] = []
    thinking_by_attempt: list[bool | None] = []
    thinking_disabled_on_retry = False
    response_metadata: list[dict[str, str | int | bool | None]] = []
    failure_reasons: list[str] = []
    generation_escalated = False
    retry_prompt: str | None = None
    thinking_capability_fallback = False
    thinking_budget_adaptive = False
    thinking_budget_exhausted = False
    # TASK-036: 動的コンテキスト（実測プロンプトサイズ＋生成上限からnum_ctxを合わせる）
    num_ctx_sent: int | None = options.get("num_ctx")
    context_needed_tokens: int | None = None
    dynamic_context_fit = False

    for attempt_idx in range(2):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": current_user_prompt},
        ]
        if dynamic_context:
            attempt_options, context_needed_tokens, fit_applied = fit_dynamic_context(
                attempt_options, messages, ceiling=max_context_length
            )
            dynamic_context_fit = dynamic_context_fit or bool(fit_applied)
        num_ctx_sent = (
            attempt_options.get("num_ctx")
            if isinstance(attempt_options.get("num_ctx"), int)
            else num_ctx_sent
        )
        content: str = ""
        attempts = 1
        metadata: dict[str, str | int | bool | None] = {}
        if (
            attempt_options.get("thinking") is True
            and thinking_supported is not None
            and thinking_supported.get(model) is False
        ):
            # 以前に思考生成非対応と判明したモデルは、400を避け最初からoffで送る
            attempt_options = dict(attempt_options)
            attempt_options["thinking"] = False
            thinking_capability_fallback = True
        elif (
            attempt_options.get("thinking") is True
            and thinking_budget_ok is not None
            and thinking_budget_ok.get(model) is False
        ):
            # 以前に思考が生成上限をすべて消費して空応答化したモデルは、
            # 同じonでは同じ切り詰めになるためoffで送る（offでは本文が生成できる）。
            attempt_options = dict(attempt_options)
            attempt_options["thinking"] = False
            thinking_budget_adaptive = True
        num_predict_by_attempt.append(attempt_options.get("num_predict"))
        thinking_by_attempt.append(attempt_options.get("thinking"))
        try:
            try:
                resp = client.chat(model=model, messages=messages, options=attempt_options)
            except OllamaError as thinking_error:
                if (
                    thinking_supported is None
                    or not _is_thinking_unsupported(thinking_error)
                    or attempt_options.get("thinking") is not True
                ):
                    raise
                # 思考生成非対応と判明したため、think=offで1回だけ再送する。
                # 意味的再試行ではなく能力情報に基づく同一要求の再送として扱う。
                thinking_supported[model] = False
                attempt_options = dict(attempt_options)
                attempt_options["thinking"] = False
                thinking_capability_fallback = True
                thinking_by_attempt.append(False)
                resp = client.chat(model=model, messages=messages, options=attempt_options)
            if isinstance(resp, ChatResponse):
                content = resp.content
                attempts = resp.attempts
                metadata = resp.to_metadata()
            else:
                content = resp if isinstance(resp, str) else (str(resp) if resp is not None else "")
                attempts = getattr(client, "_last_attempts", 1)
                if not isinstance(attempts, int):
                    attempts = 1
                # 実クライアントは chat() の応答メタデータを保持する（思考のみ・切り詰め判定用）
                last_response = getattr(client, "_last_chat_response", None)
                if isinstance(last_response, ChatResponse):
                    metadata = last_response.to_metadata()
                    attempts = max(attempts, last_response.attempts)
            eval_count = metadata.get("eval_count") if isinstance(metadata.get("eval_count"), int) else None
            eval_duration_ms = (
                metadata.get("eval_duration_ms")
                if isinstance(metadata.get("eval_duration_ms"), int)
                else None
            )
            prompt_eval_count = (
                metadata.get("prompt_eval_count")
                if isinstance(metadata.get("prompt_eval_count"), int)
                else None
            )
            transport_retries += max(0, attempts - 1)
            if (
                metadata.get("done_reason") == "length"
                and metadata.get("thinking_present")
                and not content.strip()
                and attempt_options.get("thinking") is True
            ):
                # 思考だけが生成上限を消費し、本文が生成されなかった。
                # 以後このモデルの思考行動はoffで送り、同じ切り詰めを繰り返さない。
                if thinking_budget_ok is not None:
                    thinking_budget_ok[model] = False
                thinking_budget_exhausted = True
        except OllamaError as err:
            error_info = err.to_dict()
            transport_retries += max(0, err.attempts - 1)
            num_predict_by_attempt.pop()
            break
        except Exception as exc:
            error_info = {"kind": "unexpected", "message": str(exc), "attempts": 1, "status_code": None}
            num_predict_by_attempt.pop()
            break

        raw_responses.append(content)
        response_metadata.append(metadata)

        if scenario.expected_action_kind is not None:
            parsed_target = ActionParser.parse(content, scenario.expected_action_kind)
            is_format_compliant = parsed_target is not None
            is_parse_valid = ActionParser.validate_target(
                parsed_target,
                scenario.candidates,
                actor=scenario.actor_name if scenario.expected_action_kind == ActionKind.VOTE else None,
            )
            invalid_target = invalid_target or (
                parsed_target is not None and not is_parse_valid
            )
        else:
            # discussion / discussion_seer_co
            structured_output = parse_discussion_output(content)
            if has_discussion_output_markers(content):
                is_format_compliant = structured_output is not None
                discussion_format = (
                    "two_block" if is_format_compliant else "invalid"
                )
            else:
                is_format_compliant = bool(content.strip())
                discussion_format = "legacy" if is_format_compliant else "invalid"
            result_format = discussion_format
            if scenario.action_type == "discussion_seer_co":
                # 占い師CO評価：明示的なCOと占い結果公開の成立を見る
                co_established, seer_result_published = evaluate_seer_co_response(
                    content, scenario.player_names
                )
                is_parse_valid = bool(co_established and seer_result_published)
            else:
                is_parse_valid = is_format_compliant and bool(
                    structured_output.body.strip()
                    if structured_output is not None
                    else content.strip()
                )

        is_specific = is_specific or evaluate_specificity(
            content, scenario, parsed_target, co_established
        )

        if is_parse_valid:
            break

        # 失敗理由を本番と同じ分類で記録する（空・思考のみ・切り詰め・形式・対象不正）
        issue = classify_attempt_issue(
            content=content,
            metadata=metadata,
            scenario=scenario,
            result_format=result_format,
            parsed_target=parsed_target,
            co_established=co_established,
            seer_result_published=seer_result_published,
        )
        failure_reasons.append(issue)

        # 形式不遵守またはパース失敗の場合に1回だけ再プロンプト（短い必須項目のみ）
        if attempt_idx == 0:
            semantic_retries += 1
            if issue in GENERATION_ISSUE_REASONS:
                # 本番と同じく、生成起因の再試行では生成上限を倍へ増やす
                attempt_options["num_predict"] = escalate_num_predict(
                    attempt_options.get("num_predict")
                )
                generation_escalated = True
            # 本番と同じく、再要求では思考をoffにして思考トークンが
            # num_predict を消費しないようにする
            if attempt_options.get("thinking") is not False:
                attempt_options["thinking"] = False
                thinking_disabled_on_retry = True
            current_user_prompt = _compact_retry_prompt(scenario, issue)
            retry_prompt = current_user_prompt

    elapsed_seconds = round(time.perf_counter() - start_time, 4)
    total_retries = transport_retries + semantic_retries
    fallback_used = not is_parse_valid
    status = "OK" if is_parse_valid else "FAILED"
    num_predict = (
        num_predict_by_attempt[-1]
        if num_predict_by_attempt
        else options.get("num_predict")
    )
    budget_key = ACTION_TYPE_TO_BUDGET_KEY.get(scenario.action_type)
    budget_exhausted = (
        any(metadata.get("done_reason") == "length" for metadata in response_metadata)
        or (
            isinstance(eval_count, int)
            and isinstance(num_predict, int)
            and eval_count >= num_predict
        )
    )
    last_metadata = response_metadata[-1] if response_metadata else {}

    # 本番ゲームでは、真占い師の未公開人狼判定は構造化COフォールバックで公開される。
    # モデル単体のCO成立（co_established）とゲーム上の公開成立を分けて記録する。
    co_published_in_game: bool | None = None
    if scenario.action_type == "discussion_seer_co":
        co_published_in_game = bool(co_established) or scenario.expected_seer_result is not None

    # raw_response は後方互換性のため最終応答（またはNone）を設定
    last_raw_response = raw_responses[-1] if raw_responses else None

    discussion_signals = (
        extract_discussion_signals(last_raw_response, scenario.player_names)
        if scenario.action_type in {"discussion", "discussion_seer_co"}
        and last_raw_response is not None
        else {}
    )
    question_answered: bool | None = None
    if scenario.question_expected:
        structured = parse_discussion_output(last_raw_response or "")
        body = (
            structured.body
            if structured is not None
            else last_raw_response or ""
        )
        first_sentence = re.split(r"[。！？\n]", body, maxsplit=1)[0]
        question_answered = bool(
            scenario.questioner_name
            and scenario.questioner_name in first_sentence
        )
    new_evidence_categories = sorted(
        set(discussion_signals.get("suspicion_reasons", ()))
        - set(scenario.prior_suspicion_reasons)
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "player_name": scenario.actor_name,
        "scenario_role": scenario.scenario_role,
        "personality_profile": scenario.personality_profile,
        "action_type": scenario.action_type,
        "discussion_format": (
            result_format
            if scenario.action_type in {"discussion", "discussion_seer_co"}
            else None
        ),
        "co_established": co_established,
        "seer_result_published": seer_result_published,
        "co_published_in_game": co_published_in_game,
        "options": options,
        "elapsed_seconds": elapsed_seconds,
        "status": status,
        "retry_count": total_retries,
        "transport_retries": transport_retries,
        "semantic_retries": semantic_retries,
        "total_retries": total_retries,
        "request_attempts": len(raw_responses),
        "fallback_used": fallback_used,
        "error": error_info,
        "raw_response": last_raw_response,
        "raw_responses": raw_responses,
        "parsed_target": parsed_target,
        "invalid_target": invalid_target,
        "discussion_signals": discussion_signals,
        "question_answered": question_answered,
        "new_evidence_categories": new_evidence_categories,
        "is_format_compliant": is_format_compliant,
        "is_parse_valid": is_parse_valid,
        "is_specific": is_specific,
        "is_specific_valid": bool(is_specific and is_parse_valid),
        "eval_count": eval_count,
        "eval_duration_ms": eval_duration_ms,
        "prompt_eval_count": prompt_eval_count,
        "num_predict": num_predict,
        "num_predict_by_attempt": num_predict_by_attempt,
        # TASK-036: 実際に送ったnum_ctxと、実測必要量からの動的調整が発生したか
        "num_ctx_sent": num_ctx_sent,
        "context_needed_tokens": context_needed_tokens,
        "dynamic_context_fit": dynamic_context_fit,
        "generation_escalated": generation_escalated,
        "action_budget_key": budget_key,
        "budget_exhausted": budget_exhausted,
        "thinking_enabled": options.get("thinking"),
        "thinking_by_attempt": thinking_by_attempt,
        "thinking_disabled_on_retry": thinking_disabled_on_retry,
        # 思考生成非対応と判明して think=off へ切り替えた（400回避／同一要求の再送）か
        "thinking_capability_fallback": thinking_capability_fallback,
        # 思考が生成上限を消費して切り詰めた判定（exhausted）と、offへ切り替えて送信したか
        "thinking_budget_adaptive": thinking_budget_adaptive,
        "thinking_budget_exhausted": thinking_budget_exhausted,
        "failure_reasons": failure_reasons,
        "response_metadata": response_metadata,
        "done_reason": last_metadata.get("done_reason"),
        "thinking_present": bool(last_metadata.get("thinking_present")),
        "is_empty_response": any(
            not metadata.get("content_length") for metadata in response_metadata
        ),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "retry_prompt": retry_prompt,
    }


def select_player_configs(
    configs: list[Any], model_filters: list[str]
) -> list[Any]:
    """`--models` で指定したモデル・プレイヤーだけ評価対象へ絞る（1〜2モデルの先行比較用）。

    大文字小文字を無視した部分一致で一致させる。指定なし（空白のみを含む）は全モデルを返す。
    1つも一致しない場合は、設定されているモデル名を列挙してエラーにする。
    """
    if not model_filters:
        return list(configs)
    wanted = [filter_value.strip().lower() for filter_value in model_filters if filter_value.strip()]
    if not wanted:
        # 空白だけの指定は「絞り込みなし」として全モデルを評価する
        return list(configs)
    selected = [
        cfg
        for cfg in configs
        if any(
            filter_value in cfg.model.lower() or filter_value in cfg.name.lower()
            for filter_value in wanted
        )
    ]
    if not selected:
        available = ", ".join(cfg.model for cfg in configs)
        raise ValueError(
            f"--models {', '.join(wanted)} matched no configured model. Available: {available}"
        )
    return selected


def build_parser() -> argparse.ArgumentParser:
    """評価ハーネスの引数（モデル別生成上限・思考設定・1〜2モデルへの絞り込み）。"""
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
        "--models",
        action="append",
        default=[],
        metavar="MODEL_OR_NAME",
        help=(
            "Evaluate only models or player names containing this substring (repeatable)."
            " Use for a 1-2 model pilot before the full 8-model run."
        ),
    )
    parser.add_argument(
        "--personality",
        choices=list(PERSONALITY_CHOICE_KEYS),
        default=None,
        help=(
            "Override configured personality for selected models. Use the same model "
            "filters and multiple rounds to compare personalities. 'free' disables the "
            "preset speaking style entirely (free-expression mode)."
        ),
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
        "--no-dynamic-context",
        dest="dynamic_context",
        action="store_false",
        default=True,
        help=(
            "Disable dynamic num_ctx. By default the measured prompt size plus the "
            "generation budget decides num_ctx (shrinking over-allocations and "
            "expanding up to --max-context-length when the prompt needs more)."
        ),
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=8192,
        help="Upper bound for dynamic num_ctx expansion (used with dynamic context)",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help=(
            "Global generation limit (num_predict). When omitted, per-action values from "
            "players.json or built-in defaults are used: "
            + " / ".join(ACTION_BUDGET_DEFAULT_LABELS.values())
        ),
    )
    parser.add_argument(
        "--num-predict-by-action",
        action="append",
        default=[],
        metavar="ACTION=NUM_PREDICT",
        help=(
            "Per-action generation limit override (repeatable). Actions: "
            + ", ".join(ACTION_BUDGET_KEYS)
            + ". Example: --num-predict-by-action discussion=1024 "
            "--num-predict-by-action vote=256"
        ),
    )
    parser.add_argument(
        "--thinking",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Ollama thinking generation for all actions ('auto' uses per-action "
            "defaults: discussion/wolf_discussion on, attack/seer/vote/co_correction off)"
        ),
    )
    parser.add_argument(
        "--thinking-by-action",
        action="append",
        default=[],
        metavar="ACTION=true|false",
        help=(
            "Per-action thinking override (repeatable). Actions: "
            + ", ".join(ACTION_BUDGET_KEYS)
            + ". Example: --thinking-by-action vote=false "
            "--thinking-by-action discussion=true"
        ),
    )
    parser.add_argument(
        "--no-body-reservation",
        dest="body_reservation",
        action="store_false",
        default=True,
        help=(
            "Disable TASK-053 body-budget reservation for the first request of each model x "
            "action. By default a separate thinking allowance plus a minimum body budget is "
            "reserved inside num_predict based on measured thinking-only/empty responses "
            "(capped per action), mirroring production behavior."
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama base API URL",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    # 出力先ディレクトリの確保
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    all_configs = load_player_configs(args.models_file)
    if args.personality is not None:
        all_configs = tuple(
            replace(config, personality_profile=args.personality)
            for config in all_configs
        )
    # 議論プロンプトは8人全員の名前を使うが、評価対象は --models で1〜2モデルへ絞れる
    all_player_names = tuple(cfg.name for cfg in all_configs)
    player_configs = select_player_configs(all_configs, args.models)

    client = OllamaClient(base_url=args.base_url, timeout=args.timeout, max_retries=2)
    cli_action_budgets = parse_action_num_predict_pairs(args.num_predict_by_action)
    cli_thinking_by_action = parse_thinking_by_action_pairs(args.thinking_by_action)
    base_options: dict[str, Any] = {
        "temperature": args.temperature,
        "num_ctx": args.context_length,
    }
    if args.num_predict is not None:
        base_options["num_predict"] = args.num_predict
    if args.thinking in {"on", "off"}:
        base_options["thinking"] = args.thinking == "on"
    cli_options: dict[str, Any] = dict(base_options)
    for key, value in cli_action_budgets.items():
        cli_options[f"num_predict_by_action:{key}"] = value
    for key, value in cli_thinking_by_action.items():
        cli_options[f"thinking_by_action:{key}"] = value

    records_by_model: dict[str, list[dict[str, Any]]] = {
        cfg.model: [] for cfg in player_configs
    }
    # TASK-053: モデル×アクション別の生成観測プロファイル（本文予約とthinking余白の算出元）
    thinking_profiles = ThinkingProfiles()
    # 実際にAPIへ送信した行動別生成上限（モデル別の設定解決結果）を記録する
    applied_budgets_by_model: dict[str, dict[str, dict[str, Any]]] = {
        cfg.model: {} for cfg in player_configs
    }
    cfg_num_predict_by_model: dict[str, int | None] = {
        cfg.model: cfg.num_predict for cfg in player_configs
    }
    cfg_action_budgets_by_model: dict[str, dict[str, int] | None] = {
        cfg.model: dict(cfg.num_predict_by_action)
        if cfg.num_predict_by_action
        else None
        for cfg in player_configs
    }
    # 実際にAPIへ送信した思考生成（Ollamaの think）の解決結果を記録する
    applied_thinking_by_model: dict[str, dict[str, dict[str, Any]]] = {
        cfg.model: {} for cfg in player_configs
    }
    # 思考生成が実際に使えるか。400「does not support thinking」を受けたモデルは
    # 以後の行動で think=false だけ送り、同じ400を繰り返さない。
    thinking_supported_by_model: dict[str, bool] = {
        cfg.model: True for cfg in player_configs
    }
    # 思考onで思考トークンが生成上限をすべて消費したモデルはFalseになり、
    # 以後の思考行動（議論・人狼相談）はoffで送る。offなら本文が生成できるため、
    # 文量・具体性はむしろ改善する。
    thinking_budget_ok_by_model: dict[str, bool] = {
        cfg.model: True for cfg in player_configs
    }
    cfg_thinking_by_model: dict[str, bool | None] = {
        cfg.model: cfg.thinking for cfg in player_configs
    }
    cfg_thinking_by_action_by_model: dict[str, dict[str, bool] | None] = {
        cfg.model: dict(cfg.thinking_by_action)
        if cfg.thinking_by_action
        else None
        for cfg in player_configs
    }
    personality_by_model: dict[str, str] = {
        cfg.model: cfg.personality_profile for cfg in player_configs
    }

    print(f"=== Starting Model Evaluation ===")
    print(f"Models file: {args.models_file}")
    print(f"Output directory: {output_dir}")
    print(f"Rounds: {args.rounds}, Timeout: {args.timeout}s, Temp: {args.temperature}")
    print(
        "行動別生成上限（規定）: "
        + " / ".join(ACTION_BUDGET_DEFAULT_LABELS.values())
    )
    print(
        "対象選択・CO補正の上限キャップ（全体設定が大きいモデル）: "
        + " / ".join(
            f"{key}={value}" for key, value in sorted(MINIMAL_ACTION_MAX_NUM_PREDICT.items())
        )
    )
    print(
        f"CLI全体num_predict: "
        f"{args.num_predict if args.num_predict is not None else '未設定（行動別既定を使う）'}"
    )
    print(
        "CLI行動別num_predict: "
        + (
            " / ".join(f"{key}={value}" for key, value in cli_action_budgets.items())
            if cli_action_budgets
            else "未設定"
        )
    )
    print(f"行動別思考生成（規定）: {' / '.join(THINKING_DEFAULT_LABELS.values())}")
    print(
        "CLI全体思考設定: "
        + (
            f"{'on' if args.thinking == 'on' else 'off'}"
            if args.thinking in {"on", "off"}
            else "未設定（行動別既定を使う）"
        )
    )
    print(
        "CLI行動別思考設定: "
        + (
            " / ".join(
                f"{key}={'on' if value else 'off'}"
                for key, value in cli_thinking_by_action.items()
            )
            if cli_thinking_by_action
            else "未設定"
        )
    )
    print(
        "本文予算予約（TASK-053）: "
        + (
            "on（初回リクエストでモデル別のthinking余白と本文最低予算を分離）"
            if bool(getattr(args, "body_reservation", True))
            else "off（num_predictは設定値のまま。比較ラン用）"
        )
    )
    if args.models:
        print(f"--models 指定: {', '.join(args.models)}")
    print(f"Evaluating {len(player_configs)} models...")

    for round_idx in range(1, args.rounds + 1):
        print(f"\n--- Round {round_idx} / {args.rounds} ---")
        for cfg in player_configs:
            model = cfg.model
            player_actor = Player(
                name=cfg.name,
                model=model,
                role=Role.VILLAGER,
                is_alive=True,
                num_ctx=cfg.num_ctx,
                timeout=cfg.timeout,
                num_predict=cfg.num_predict,
                num_predict_by_action=cfg.num_predict_by_action,
                thinking=cfg.thinking,
                thinking_by_action=cfg.thinking_by_action,
                personality_profile=cfg.personality_profile,
            )
            scenarios = generate_scenarios(player_actor, all_player_names)

            print(f"[{cfg.name}] Testing model '{model}' ({len(scenarios)} actions)...")
            for scenario in scenarios:
                scenario_options = dict(base_options)
                try:
                    budget_key = ACTION_TYPE_TO_BUDGET_KEY[scenario.action_type]
                    num_predict, num_predict_source = resolve_action_num_predict(
                        player_actor, cli_options, budget_key
                    )
                    thinking_enabled, thinking_source = resolve_thinking(
                        player_actor, cli_options, budget_key
                    )
                    # 本番のOllamaActionsと同じ下限適用（設定未指定時のみ）で、
                    # 議論・人狼相談の思考トークン残余を確保する。
                    num_predict, num_predict_source = apply_thinking_floor(
                        num_predict, num_predict_source, thinking_enabled, budget_key
                    )
                    # TASK-053: 本番と同じ本文予算予約を適用する。モデル×アクション別の
                    # 観測から「thinking余白 + 本文最低予算」を分離し、思考が上限を食い潰して
                    # 空応答になるのを防ぐ（cap超えは抑え、待ち時間増を限定）。
                    body_reservation = reserve_body_budget(
                        num_predict,
                        num_predict_source,
                        thinking_profiles.get(model, budget_key),
                        budget_key,
                        thinking_enabled=thinking_enabled,
                        enabled=bool(getattr(args, "body_reservation", True)),
                    )
                    if body_reservation.applied and body_reservation.num_predict is not None:
                        num_predict = body_reservation.num_predict
                        num_predict_source = body_reservation.source
                    if num_predict is not None:
                        scenario_options["num_predict"] = num_predict
                    if thinking_enabled is not None:
                        scenario_options["thinking"] = thinking_enabled
                    result = evaluate_scenario(
                        client,
                        model,
                        scenario,
                        scenario_options,
                        thinking_supported_by_model,
                        thinking_budget_ok_by_model,
                        dynamic_context=bool(getattr(args, "dynamic_context", True)),
                        max_context_length=args.max_context_length,
                    )
                    result["round"] = round_idx
                    result["action_budget_key"] = budget_key
                    result["num_predict_source"] = num_predict_source
                    if result.get("thinking_capability_fallback"):
                        # 能力情報により実際にoffで送信した（設定のon/既定と異なる）
                        thinking_source = THINKING_CAPABILITY_SOURCE
                        result["thinking_source"] = thinking_source
                    elif result.get("thinking_budget_adaptive"):
                        # 思考予算不足の判定により実際にoffで送信した
                        thinking_source = THINKING_BUDGET_SOURCE
                        result["thinking_source"] = thinking_source
                    else:
                        result["thinking_source"] = thinking_source
                    applied_budgets_by_model[model][budget_key] = {
                        "num_predict": num_predict,
                        "source": num_predict_source,
                    }
                    applied_thinking_by_model[model][budget_key] = {
                        "thinking": (
                            result["thinking_by_attempt"][-1]
                            if result["thinking_by_attempt"]
                            else thinking_enabled
                        ),
                        "source": thinking_source,
                    }
                    # TASK-053: 本文予約の適用結果と、この結果から更新したモデル×アクション別
                    # プロファイルを評価JSONLへ残す（実機での比較・能力プロファイル更新の検証用）。
                    result["body_reservation"] = body_reservation.serialize()
                    result["generation_profile"] = record_harness_attempts(
                        thinking_profiles, model, budget_key, result
                    )
                    records_by_model[model].append(result)

                    status = "OK" if result["is_parse_valid"] else "FAILED"
                    if result["error"]:
                        status = f"ERROR: {result['error'].get('kind')}"
                    dynamic_note = (
                        f", 動的コンテキスト調整(必要量概算={result['context_needed_tokens']})"
                        if result.get("dynamic_context_fit")
                        else ""
                    )
                    print(
                        f"  - {scenario.action_type.ljust(11)}: {status} "
                        f"(elapsed: {result['elapsed_seconds']}s, sends: {result['request_attempts']}, "
                        f"retries: {result['retry_count']}, "
                        f"num_predict: {result['num_predict']} [{num_predict_source}] "
                        f"attempts={result['num_predict_by_attempt']}"
                        f"{', 生成上限を倍へ増加' if result['generation_escalated'] else ''}, "
                        f"think: {result['thinking_enabled']} [{thinking_source}] "
                        f"think_attempts={result['thinking_by_attempt']}"
                        f"{', 再要求で思考off' if result['thinking_disabled_on_retry'] else ''}"
                        f"{', 思考非対応→off' if result['thinking_capability_fallback'] else ''}"
                        f"{', 思考予算不足→off' if result['thinking_budget_adaptive'] else ''}, "
                        f"eval_count: {result['eval_count']}, "
                        f"eval_ms: {result['eval_duration_ms']}, "
                        f"num_ctx_sent: {result['num_ctx_sent']}{dynamic_note}, "
                        f"done_reason: {result['done_reason']}, "
                        f"thinking: {result['thinking_present']}, "
                        f"specific: {result['is_specific']}, "
                        f"budget_exhausted: {result['budget_exhausted']}, "
                        f"reasons: {result['failure_reasons']})"
                    )
                except Exception as ex:
                    # 途中で1モデルが予期せぬ失敗をしても全体を止めない
                    print(f"  - {scenario.action_type.ljust(11)}: UNCAUGHT EXCEPTION: {ex}")
                    err_result = {
                        "round": round_idx,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": model,
                        "player_name": cfg.name,
                        "scenario_role": scenario.scenario_role,
                        "personality_profile": cfg.personality_profile,
                        "action_type": scenario.action_type,
                        "options": dict(scenario_options),
                        "elapsed_seconds": 0.0,
                        "retry_count": 0,
                        "transport_retries": 0,
                        "semantic_retries": 0,
                        "total_retries": 0,
                        "request_attempts": 0,
                        "error": {"kind": "uncaught", "message": str(ex)},
                        "raw_response": None,
                        "raw_responses": [],
                        "parsed_target": None,
                        "invalid_target": False,
                        "discussion_signals": {},
                        "is_format_compliant": False,
                        "is_parse_valid": False,
                        "fallback_used": True,
                        "is_specific": False,
                        "is_specific_valid": False,
                        "eval_count": None,
                        "eval_duration_ms": None,
                        "prompt_eval_count": None,
                        "num_predict": scenario_options.get("num_predict"),
                        "num_predict_by_attempt": [],
                        "num_ctx_sent": scenario_options.get("num_ctx"),
                        "context_needed_tokens": None,
                        "dynamic_context_fit": False,
                        "generation_escalated": False,
                        "action_budget_key": ACTION_TYPE_TO_BUDGET_KEY[scenario.action_type],
                        "budget_exhausted": False,
                        "thinking_enabled": scenario_options.get("thinking"),
                        "thinking_source": scenario_options.get(
                            "thinking_source", "unresolved"
                        ),
                        "thinking_by_attempt": [],
                        "thinking_disabled_on_retry": False,
                        "thinking_capability_fallback": False,
                        "thinking_budget_adaptive": False,
                        "thinking_budget_exhausted": False,
                        "failure_reasons": ["uncaught_exception"],
                        "response_metadata": [],
                        "done_reason": None,
                        "thinking_present": False,
                        "is_empty_response": False,
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
        specific_count = sum(1 for r in records if r.get("is_specific"))
        specific_valid_count = sum(1 for r in records if r.get("is_specific_valid"))
        budget_exhausted_count = sum(1 for r in records if r.get("budget_exhausted"))
        error_count = sum(1 for r in records if r["error"] is not None)
        total_time = sum(r["elapsed_seconds"] for r in records)
        avg_time = round(total_time / total_actions, 3) if total_actions > 0 else 0.0
        avg_eval_count = _average_metric([r.get("eval_count") for r in records])
        avg_eval_duration_ms = _average_metric([r.get("eval_duration_ms") for r in records])
        transport_retries = sum(r.get("transport_retries", 0) for r in records)
        semantic_retries = sum(r.get("semantic_retries", 0) for r in records)
        total_retries = sum(r.get("total_retries", r.get("retry_count", 0)) for r in records)
        fallback_count = sum(1 for r in records if r.get("fallback_used", not r["is_parse_valid"]))

        # アクション種別ごとの内訳
        by_action: dict[str, Any] = {}
        for action_name in (
            "discussion",
            "discussion_seer_co",
            "vote",
            "attack",
            "seer",
        ):
            act_recs = [r for r in records if r["action_type"] == action_name]
            action_budget_key = ACTION_TYPE_TO_BUDGET_KEY[action_name]
            # TASK-053: このモデル×アクションの観測プロファイル（率・平均待ち時間を含む）
            action_profile = thinking_profiles.get(model, action_budget_key)
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
                "invalid_target_count": sum(
                    1 for r in act_recs if r.get("invalid_target")
                ),
                "invalid_target_rate": (
                    round(
                        sum(1 for r in act_recs if r.get("invalid_target")) / act_total,
                        3,
                    )
                    if act_total
                    else 0.0
                ),
                "format_compliance_rate": round(act_compliant / act_total, 3) if act_total else 0.0,
                "parse_success_rate": round(act_valid / act_total, 3) if act_total else 0.0,
                "specific_count": sum(1 for r in act_recs if r.get("is_specific")),
                "specific_rate": (
                    round(sum(1 for r in act_recs if r.get("is_specific")) / act_total, 3)
                    if act_total
                    else 0.0
                ),
                "specific_valid_rate": (
                    round(sum(1 for r in act_recs if r.get("is_specific_valid")) / act_total, 3)
                    if act_total
                    else 0.0
                ),
                "avg_eval_count": _average_metric([r.get("eval_count") for r in act_recs]),
                "avg_eval_duration_ms": _average_metric(
                    [r.get("eval_duration_ms") for r in act_recs]
                ),
                "budget_exhausted_count": sum(1 for r in act_recs if r.get("budget_exhausted")),
                "request_attempts": _request_attempts(act_recs),
                "thinking_off_requests": sum(
                    1 for r in act_recs if r.get("thinking_enabled") is False
                ),
                "thinking_retry_off_requests": sum(
                    1 for r in act_recs if r.get("thinking_disabled_on_retry")
                ),
                "thinking_capability_fallback_requests": sum(
                    1 for r in act_recs if r.get("thinking_capability_fallback")
                ),
                "thinking_budget_adaptive_requests": sum(
                    1 for r in act_recs if r.get("thinking_budget_adaptive")
                ),
                "thinking_budget_exhausted_requests": sum(
                    1 for r in act_recs if r.get("thinking_budget_exhausted")
                ),
                **_metadata_counts(act_recs),
                "failure_reasons": _failure_reason_counts(act_recs),
                "action_budget_key": action_budget_key,
                # TASK-053: モデル×アクション別の空応答率・thinking消費率・平均待ち時間
                # （本文予算制御の効果比較に使う実測プロファイル）。
                "generation_profile": (
                    action_profile.serialize() if action_profile is not None else None
                ),
            }

        co_recs = [
            r for r in records if r.get("action_type") == "discussion_seer_co"
        ]
        co_total = len(co_recs)
        co_established_count = sum(1 for r in co_recs if r.get("co_established"))
        co_published_count = sum(1 for r in co_recs if r.get("seer_result_published"))
        co_in_game_count = sum(1 for r in co_recs if r.get("co_published_in_game"))

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
            "invalid_target_count": sum(1 for r in records if r.get("invalid_target")),
            "invalid_target_rate": (
                round(
                    sum(1 for r in records if r.get("invalid_target")) / total_actions,
                    3,
                )
                if total_actions
                else 0.0
            ),
            "seer_target_invalid_count": sum(
                1
                for r in records
                if r.get("action_type") == "seer" and r.get("invalid_target")
            ),
            "seer_target_invalid_rate": (
                round(
                    sum(
                        1
                        for r in records
                        if r.get("action_type") == "seer" and r.get("invalid_target")
                    )
                    / sum(1 for r in records if r.get("action_type") == "seer"),
                    3,
                )
                if any(r.get("action_type") == "seer" for r in records)
                else 0.0
            ),
            "fallback_rate": round(fallback_count / total_actions, 3) if total_actions else 0.0,
            "seer_co_total": co_total,
            "seer_co_established": co_established_count,
            "seer_co_established_rate": (
                round(co_established_count / co_total, 3) if co_total else 0.0
            ),
            "seer_co_result_published_rate": (
                round(co_published_count / co_total, 3) if co_total else 0.0
            ),
            # モデル単体のCO成立と、ゲーム内で構造化COフォールバックにより公開された割合
            "seer_co_published_in_game": co_in_game_count,
            "seer_co_published_in_game_rate": (
                round(co_in_game_count / co_total, 3) if co_total else 0.0
            ),
            "average_response_seconds": avg_time,
            "avg_eval_count": avg_eval_count,
            "avg_eval_duration_ms": avg_eval_duration_ms,
            "specific_count": specific_count,
            "specific_rate": round(specific_count / total_actions, 3) if total_actions else 0.0,
            "specific_valid_rate": (
                round(specific_valid_count / total_actions, 3) if total_actions else 0.0
            ),
            "personality_profile": personality_by_model.get(model, "balanced"),
            "discussion_signal_distribution": summarize_discussion_signals(records),
            "budget_exhausted_count": budget_exhausted_count,
            "request_attempts": _request_attempts(records),
            **_metadata_counts(records),
            **_dynamic_context_stats(records),
            "failure_reasons": _failure_reason_counts(records),
            "options": base_options,
            "cli_num_predict_by_action": cli_action_budgets,
            "default_action_num_predict": dict(DEFAULT_ACTION_NUM_PREDICT),
            "player_num_predict": cfg_num_predict_by_model.get(model),
            "player_num_predict_by_action": cfg_action_budgets_by_model.get(model),
            "applied_action_num_predict": applied_budgets_by_model.get(model, {}),
            "player_thinking": cfg_thinking_by_model.get(model),
            "player_thinking_by_action": cfg_thinking_by_action_by_model.get(model),
            "applied_thinking": applied_thinking_by_model.get(model, {}),
            "thinking_off_requests": sum(
                1 for r in records if r.get("thinking_enabled") is False
            ),
            "thinking_retry_off_requests": sum(
                1 for r in records if r.get("thinking_disabled_on_retry")
            ),
            "thinking_capability_fallback_requests": sum(
                1 for r in records if r.get("thinking_capability_fallback")
            ),
            "thinking_budget_adaptive_requests": sum(
                1 for r in records if r.get("thinking_budget_adaptive")
            ),
            "thinking_budget_exhausted_requests": sum(
                1 for r in records if r.get("thinking_budget_exhausted")
            ),
            # 思考生成が実際に使えるモデルか（400を受けたモデルはFalse）
            "thinking_supported": thinking_supported_by_model.get(model, True),
            # 思考onが生成上限を思考だけで消費しなかったか（不足なら以後offで送信）
            "thinking_budget_ok": thinking_budget_ok_by_model.get(model, True),
            # TASK-053: モデル別×アクション別の生成プロファイル（本文予約の成否を含む）
            "generation_profiles": thinking_profiles.snapshot().get(model, {}),
            "reserved_body_actions": sorted(
                key
                for key, applied in applied_budgets_by_model.get(model, {}).items()
                if isinstance(applied, dict)
                and applied.get("source") == "model_action_body_reservation"
            ),
            "by_action": by_action,
        }

    # 全体サマリ計算
    total_transport_retries = sum(m["transport_retries"] for m in summary_by_model.values())
    total_semantic_retries = sum(m["semantic_retries"] for m in summary_by_model.values())
    overall_total_retries = sum(m["total_retries"] for m in summary_by_model.values())
    total_fallback_count = sum(m["fallback_count"] for m in summary_by_model.values())
    all_records = [rec for recs in records_by_model.values() for rec in recs]

    # 2. summary.json 出力
    overall_summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models_file": str(args.models_file),
        "models_filter": list(args.models),
        "personality_override": args.personality,
        "personality_profiles": personality_by_model,
        "selected_models": [cfg.model for cfg in player_configs],
        "rounds": args.rounds,
        "timeout": args.timeout,
        # TASK-053: 本文予算予約の設定（A/B比較の判別用）とモデル×アクション別プロファイル
        "body_reservation": bool(getattr(args, "body_reservation", True)),
        "generation_profiles": thinking_profiles.snapshot(),
        "transport_retries": total_transport_retries,
        "semantic_retries": total_semantic_retries,
        "total_retries": overall_total_retries,
        "fallback_count": total_fallback_count,
        "avg_eval_count": _average_metric([r.get("eval_count") for r in all_records]),
        "avg_eval_duration_ms": _average_metric([r.get("eval_duration_ms") for r in all_records]),
        "specific_rate": round(
            sum(1 for r in all_records if r.get("is_specific")) / len(all_records), 3
        )
        if all_records
        else 0.0,
        "specific_valid_rate": round(
            sum(1 for r in all_records if r.get("is_specific_valid")) / len(all_records), 3
        )
        if all_records
        else 0.0,
        "budget_exhausted_count": sum(1 for r in all_records if r.get("budget_exhausted")),
        "request_attempts": _request_attempts(all_records),
        **_metadata_counts(all_records),
        **_dynamic_context_stats(all_records),
        "failure_reasons": _failure_reason_counts(all_records),
        "cli_num_predict_by_action": cli_action_budgets,
        "default_action_num_predict": dict(DEFAULT_ACTION_NUM_PREDICT),
        "minimal_action_max_num_predict": dict(MINIMAL_ACTION_MAX_NUM_PREDICT),
        "budget_resolution_priority": list(ACTION_BUDGET_RESOLUTION_PRIORITY),
        "applied_action_num_predict": applied_budgets_by_model,
        "cli_thinking": args.thinking,
        "cli_thinking_by_action": cli_thinking_by_action,
        "default_thinking_by_action": dict(DEFAULT_THINKING_BY_ACTION),
        "thinking_resolution_priority": list(THINKING_RESOLUTION_PRIORITY),
        "applied_thinking": applied_thinking_by_model,
        "thinking_off_requests": sum(
            1 for r in all_records if r.get("thinking_enabled") is False
        ),
        "thinking_retry_off_requests": sum(
            1 for r in all_records if r.get("thinking_disabled_on_retry")
        ),
        "thinking_capability_fallback_requests": sum(
            1 for r in all_records if r.get("thinking_capability_fallback")
        ),
        "thinking_budget_adaptive_requests": sum(
            1 for r in all_records if r.get("thinking_budget_adaptive")
        ),
        "thinking_budget_exhausted_requests": sum(
            1 for r in all_records if r.get("thinking_budget_exhausted")
        ),
        "thinking_supported_models": {
            model: supported for model, supported in sorted(thinking_supported_by_model.items())
        },
        "thinking_budget_ok_models": {
            model: ok for model, ok in sorted(thinking_budget_ok_by_model.items())
        },
        "thinking_affordable_num_predict": dict(THINKING_AFFORDABLE_NUM_PREDICT),
        "dynamic_context": bool(getattr(args, "dynamic_context", True)),
        "max_context_length": args.max_context_length,
        "options": base_options,
        "models": summary_by_model,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    print("\n=== Evaluation Complete ===")
    print(f"Summary saved to: {summary_path}")
    print(f"Per-model logs saved to: {output_dir}/*.jsonl")

    print("\n=== 生成トークン・特異性メトリクス ===")
    for model, stat in summary_by_model.items():
        print(f"  {model}:")
        applied = stat.get("applied_action_num_predict") or {}
        if applied:
            print(
                "    適用num_predict: "
                + " / ".join(
                    f"{key}={value['num_predict']}[{value['source']}]"
                    for key, value in applied.items()
                )
            )
        applied_thinking = stat.get("applied_thinking") or {}
        if applied_thinking:
            print(
                "    適用思考(think): "
                + " / ".join(
                    f"{key}={'on' if value['thinking'] else 'off'}[{value['source']}]"
                    if value["thinking"] is not None
                    else f"{key}=未指定[{value['source']}]"
                    for key, value in applied_thinking.items()
                )
            )
        print(
            f"    思考off送信={stat.get('thinking_off_requests')} "
            f"再要求で思考off={stat.get('thinking_retry_off_requests')} "
            f"思考予算不足→off={stat.get('thinking_budget_adaptive_requests')} "
            f"思考予算不足判定={stat.get('thinking_budget_exhausted_requests')} "
            f"thinking_supported={stat.get('thinking_supported')} "
            f"thinking_budget_ok={stat.get('thinking_budget_ok')}"
        )
        print(
            f"    動的コンテキスト: 調整={stat.get('dynamic_context_fit_requests')} "
            f"送信num_ctx 平均={stat.get('avg_num_ctx_sent')} "
            f"最小={stat.get('min_num_ctx_sent')} 最大={stat.get('max_num_ctx_sent')}"
        )
        for action_name, act in stat["by_action"].items():
            print(
                f"    {action_name.ljust(11)}: avg_eval_count={act['avg_eval_count']} "
                f"avg_eval_ms={act['avg_eval_duration_ms']} "
                f"specific={act['specific_rate']} specific+valid={act['specific_valid_rate']} "
                f"budget_exhausted={act['budget_exhausted_count']} "
                f"sends={act['request_attempts']} empty={act['empty_response_count']} "
                f"thinking_only={act['thinking_only_count']} truncated={act['truncated_count']}"
            )
        print(
            f"    合計: avg_eval_count={stat['avg_eval_count']} "
            f"avg_eval_ms={stat['avg_eval_duration_ms']} "
            f"specific={stat['specific_rate']} specific+valid={stat['specific_valid_rate']} "
            f"budget_exhausted={stat['budget_exhausted_count']} "
            f"sends={stat['request_attempts']} empty={stat['empty_response_count']} "
            f"thinking_only={stat['thinking_only_count']} truncated={stat['truncated_count']}"
        )
        if stat["failure_reasons"]:
            print(
                "    失敗理由: "
                + " / ".join(
                    f"{reason}={count}"
                    for reason, count in sorted(stat["failure_reasons"].items())
                )
            )

    print("\n=== Seer CO comparison (CO成立率 / 結果公開率 / ゲーム内公開 / フォールバック率) ===")
    for model, stat in summary_by_model.items():
        print(
            f"  {model}: CO成立率={stat['seer_co_established_rate']} "
            f"({stat['seer_co_established']}/{stat['seer_co_total']}), "
            f"結果公開率={stat['seer_co_result_published_rate']}, "
            f"ゲーム内CO公開率={stat['seer_co_published_in_game_rate']}, "
            f"フォールバック率={stat['fallback_rate']}"
        )


if __name__ == "__main__":
    main()
