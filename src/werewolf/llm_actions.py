import random
import re
import time
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable

from .action_parser import ActionKind, ActionParser
from .engine import (
    ActionPolicy,
    extract_public_role_claim,
    extract_public_seer_result,
    is_public_seer_claim,
)
from .discussion_format import (
    has_discussion_output_markers,
    parse_discussion_output,
)
from .models import GameState, Message, Player, PublicRoleClaim, PublicSeerClaim, Role
from .ollama import CLAIM_ROLE_TOOL, ChatResponse, OllamaClient, OllamaError, ToolCall
from .prompts import PromptBuilder, PromptContext, ROLE_NAMES_JA

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ActionTrace:
    action: ActionKind | str
    player: str
    raw_responses: tuple[str, ...]
    parsed_targets: tuple[str | None, ...]
    selected_target: str | None
    fallback_used: bool
    error: dict[str, str | int | None] | None = None
    failure_reasons: tuple[str, ...] = ()
    response_metadata: tuple[dict[str, str | int | bool | None], ...] = ()


class OllamaActions(ActionPolicy):
    def __init__(
        self,
        state: GameState,
        client: OllamaClient,
        seed: int | None = None,
        options: dict[str, object] | None = None,
        event_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.state = state
        self.client = client
        self.options = options
        self.event_callback = event_callback
        self._rng = random.Random(state.seed if seed is None else seed)
        self.traces: list[ActionTrace] = []
        self._seer_info_used_day: dict[str, int] = {}

    def wolf_discuss(
        self, wolf: str, partner: str, history: tuple[Message, ...]
    ) -> str:
        started_at = self._start_action("wolf_discussion", self._player(wolf))
        player = self._player(wolf)
        prompt_ctx = PromptBuilder.build_wolf_discuss_prompt(
            actor=player,
            partner_name=partner,
            alive_players=tuple(self.state.alive_players),
            wolf_chat_history=history,
            day=self.state.day,
            state=self.state,
        )
        return self._free_text_with_context(
            "wolf_discussion",
            player,
            prompt_ctx,
            f"{wolf} は相談に応答できませんでした。",
            started_at,
        )

    def choose_attack(self, wolves: tuple[str, ...], candidates: tuple[str, ...]) -> str:
        started_at = self._start_action("attack", self._player(wolves[0]))
        actor = self._player(wolves[0])
        valid_candidates = self._valid_candidates(
            actor, candidates, forbidden=wolves
        )
        prompt_ctx = PromptBuilder.build_attack_prompt(
            actor=actor,
            wolves=wolves,
            valid_candidates=valid_candidates,
            day=self.state.day,
            state=self.state,
        )
        return self._select_target(
            ActionKind.ATTACK,
            actor,
            valid_candidates,
            prompt_ctx,
            started_at,
            forbidden=wolves,
        )

    def choose_seer_target(
        self, seer: str, candidates: tuple[str, ...], results: tuple[tuple[str, str], ...]
    ) -> str:
        started_at = self._start_action("seer", self._player(seer))
        actor = self._player(seer)
        living_candidates = self._valid_candidates(actor, candidates)
        unscanned_candidates = tuple(
            candidate
            for candidate in living_candidates
            if candidate not in self.state.seer_results
        )
        eligible_candidates = unscanned_candidates or living_candidates
        prompt_ctx = PromptBuilder.build_seer_prompt(
            actor=actor,
            alive_players=eligible_candidates,
            seer_results=results,
            day=self.state.day,
            state=self.state,
        )
        return self._select_target(
            ActionKind.SEER, actor, eligible_candidates, prompt_ctx, started_at
        )

    def discuss(
        self, player: str, day: int, alive: tuple[str, ...], history: tuple[Message, ...]
    ) -> str:
        started_at = self._start_action("discussion", self._player(player))
        actor = self._player(player)
        prompt_ctx = PromptBuilder.build_discussion_prompt(
            actor=actor,
            alive_players=alive,
            conversation_history=history,
            day=day,
            state=self.state,
        )
        return self._free_text_with_context(
            "discussion",
            actor,
            prompt_ctx,
            f"{player} は発言できませんでした。",
            started_at,
            recent_messages=history,
        )

    def vote(
        self, voter: str, candidates: tuple[str, ...], day: int, runoff: bool
    ) -> str:
        started_at = self._start_action("vote", self._player(voter))
        actor = self._player(voter)
        valid_candidates = self._valid_candidates(
            actor, candidates, forbid_self=True
        )
        recent_speech = [
            f"{message.speaker}: {message.content}"
            for message in self.state.conversation_log
            if message.speaker != "system"
        ][-8:]
        prompt_ctx = PromptBuilder.build_vote_prompt(
            actor=actor,
            alive_players=tuple(self.state.alive_players),
            valid_candidates=valid_candidates,
            today_summary="\n".join(recent_speech),
            runoff=runoff,
            day=day,
            state=self.state,
        )
        return self._select_target(
            ActionKind.VOTE,
            actor,
            candidates,
            prompt_ctx,
            started_at,
            forbid_self=True,
        )

    def _select_target(
        self,
        action: ActionKind,
        actor: Player,
        candidates: tuple[str, ...],
        prompt_ctx: PromptContext,
        started_at: float,
        forbid_self: bool = False,
        forbidden: tuple[str, ...] = (),
    ) -> str:
        valid_candidates = self._valid_candidates(
            actor, candidates, forbid_self=forbid_self, forbidden=forbidden
        )
        if not valid_candidates:
            raise ValueError(f"No valid candidates for {action.value} by {actor.name}.")

        user_prompts = [prompt_ctx.user_prompt]
        raw_responses: list[str] = []
        parsed_targets: list[str | None] = []
        last_error: OllamaError | None = None
        transport_attempts = 0
        semantic_retries = 0
        last_response_issue: str | None = None
        failure_reasons: list[str] = []
        response_metadata_history: list[dict[str, str | int | bool | None]] = []

        for request_number in range(2):
            messages = [
                {"role": "system", "content": prompt_ctx.system_prompt},
                {"role": "user", "content": user_prompts[-1]},
            ]
            try:
                chat_response = self._chat(
                    actor,
                    messages,
                    increase_generation=(
                        request_number == 1
                        and last_response_issue
                        in {
                            "empty_content",
                            "thinking_only_response",
                            "generation_truncated",
                        }
                    ),
                    request_number=request_number + 1,
                    request_reason=(
                        "初回"
                        if request_number == 0
                        else "意味的再要求"
                    ),
                    tool_based=False,
                )
            except OllamaError as error:
                last_error = error
                context_overflow = (
                    error.kind != "timeout" and self._is_context_overflow(error)
                )
                failure_reason = (
                    "context_overflow_compacted"
                    if context_overflow
                    else self._transport_failure_reason(error)
                )
                failure_reasons.append(failure_reason)
                transport_attempts += error.attempts
                if (
                    request_number == 0
                    and context_overflow
                ):
                    self._emit(
                        "llm_action_parse_failed",
                        action,
                        actor,
                        started_at,
                        reason="context_overflow_compacted",
                        will_retry=True,
                    )
                    user_prompts.append(
                        self._target_retry_prompt(
                            action,
                            actor,
                            valid_candidates,
                            "context_overflow_compacted",
                        )
                    )
                    semantic_retries += 1
                    continue
                self._emit(
                    "llm_action_parse_failed",
                    action,
                    actor,
                    started_at,
                    reason=failure_reason,
                    will_retry=False,
                )
                break

            transport_attempts += chat_response.attempts
            response_metadata = chat_response.to_metadata()
            response_metadata_history.append(response_metadata)
            raw_responses.append(chat_response.content)
            final_response = self._final_response(chat_response.content)
            if response_metadata.get("done_reason") == "length":
                response_issue = "generation_truncated"
            elif not final_response:
                response_issue = (
                    "thinking_only_response"
                    if response_metadata.get("thinking_present")
                    or self._has_thinking(chat_response.content)
                    else "empty_content"
                )
            else:
                response_issue = None
            last_response_issue = response_issue
            target = (
                None
                if response_issue is not None
                else ActionParser.parse(final_response, action)
            )
            parsed_targets.append(target)
            if ActionParser.validate_target(
                target, valid_candidates, actor.name if forbid_self else None
            ):
                trace = ActionTrace(
                    action, actor.name, tuple(raw_responses),
                    tuple(parsed_targets), target, False,
                    failure_reasons=tuple(failure_reasons),
                    response_metadata=tuple(response_metadata_history),
                )
                self.traces.append(trace)
                self._complete_action(
                    action,
                    actor,
                    started_at,
                    semantic_retries,
                    transport_attempts,
                    False,
                    selected_target=target,
                    failure_reasons=tuple(failure_reasons),
                    response_metadata=response_metadata,
                )
                return target

            if response_issue is not None:
                failure_reason = response_issue
                self._emit(
                    "llm_action_parse_failed",
                    action,
                    actor,
                    started_at,
                    transport_retries=max(0, transport_attempts - len(raw_responses)),
                    parsed_target=None,
                    reason=failure_reason,
                    response_metadata=response_metadata,
                    will_retry=request_number == 0,
                )
                failure_reasons.append(failure_reason)
                if request_number == 0:
                    semantic_retries += 1
                    user_prompts.append(
                        self._target_retry_prompt(
                            action,
                            actor,
                            valid_candidates,
                            failure_reason,
                        )
                    )
                continue

            self._emit(
                "llm_action_parse_failed",
                action,
                actor,
                started_at,
                transport_retries=max(0, transport_attempts - len(raw_responses)),
                parsed_target=target,
                reason=(
                    "invalid_target"
                    if target is not None
                    else "invalid_action_response"
                ),
                response_metadata=response_metadata,
                will_retry=request_number == 0,
            )
            failure_reasons.append(
                "invalid_target" if target is not None else "invalid_action_response"
            )
            if request_number == 0:
                semantic_retries += 1
                user_prompts.append(
                    self._target_retry_prompt(
                        action,
                        actor,
                        valid_candidates,
                        "invalid_target" if target is not None else "invalid_action_response",
                    )
                )

        fallback = self._rng.choice(valid_candidates)
        self.traces.append(
            ActionTrace(
                action,
                actor.name,
                tuple(raw_responses),
                tuple(parsed_targets),
                fallback,
                True,
                last_error.to_dict() if last_error is not None else None,
                tuple(failure_reasons),
                tuple(response_metadata_history),
            )
        )
        self._complete_action(
            action,
            actor,
            started_at,
            semantic_retries,
            transport_attempts,
            True,
            selected_target=fallback,
            error=last_error,
            failure_reasons=tuple(failure_reasons),
            failure_reason=(
                failure_reasons[-1] if failure_reasons else "invalid_action_response"
            ),
            response_metadata=(
                response_metadata_history[-1]
                if response_metadata_history
                else None
            ),
        )
        return fallback

    def _valid_candidates(
        self,
        actor: Player,
        candidates: tuple[str, ...],
        forbid_self: bool = False,
        forbidden: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        alive = set(self.state.alive_players)
        return tuple(
            candidate
            for candidate in candidates
            if candidate in alive
            and candidate not in forbidden
            and (not forbid_self or candidate != actor.name)
        )

    def _target_retry_prompt(
        self,
        action: ActionKind,
        actor: Player,
        valid_candidates: tuple[str, ...],
        failure_reason: str,
    ) -> str:
        corrections = {
            "empty_content": "前回は空でした。",
            "thinking_only_response": "前回は思考だけでした。回答だけを出してください。",
            "generation_truncated": "前回は生成上限で途切れました。説明せず形式だけを出してください。",
            "invalid_target": "前回の対象は候補外でした。有効候補だけから選んでください。",
            "invalid_action_response": "前回は対象を解析できませんでした。",
            "context_overflow_compacted": "前回は入力が長すぎました。候補だけを見て簡潔に選んでください。",
        }
        return (
            f"【再回答】{corrections.get(failure_reason, '前回の回答を修正してください。')}\n"
            f"- 行動：{self._label(action)}\n"
            f"- あなた：{actor.name}\n"
            f"- 有効候補：{', '.join(valid_candidates)}\n"
            f"候補から1名だけ選び、「{self._label(action)}：名前」の1行だけを返してください。"
        )

    def _free_text_with_context(
        self,
        action: str,
        actor: Player,
        prompt_ctx: PromptContext,
        fallback: str,
        started_at: float,
        recent_messages: tuple[Message, ...] | None = None,
    ) -> str:
        raw_responses: list[str] = []
        last_error: OllamaError | None = None
        transport_attempts = 0
        last_response_issue: str | None = None
        failure_reasons: list[str] = []
        response_metadata_history: list[dict[str, str | int | bool | None]] = []
        last_spoken_result: tuple[str, str] | None = None
        for attempt in range(2):
            user_prompt = (
                self._seer_co_correction_prompt(actor, last_spoken_result)
                if attempt
                and action == "discussion"
                and last_response_issue == "seer_result_without_claim"
                else self._short_discussion_retry_prompt(
                    actor, last_response_issue, recent_messages
                )
                if attempt and action == "discussion"
                else prompt_ctx.user_prompt
                + (
                    self._discussion_retry_instruction(
                        action, actor, last_response_issue
                    )
                    if attempt
                    else ""
                )
            )
            messages = [
                {"role": "system", "content": prompt_ctx.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                chat_response = self._chat_with_claim_tools(
                    actor,
                    messages,
                    action,
                    increase_generation=(
                        attempt == 1
                        and last_response_issue
                        in {"empty_content", "thinking_only_response", "generation_truncated"}
                    ),
                    started_at=started_at,
                    request_number=attempt + 1,
                    request_reason=(
                        "初回"
                        if attempt == 0
                        else (
                            "CO補正"
                            if last_response_issue == "seer_result_without_claim"
                            else "意味的再要求"
                        )
                    ),
                    tool_based=(action == "discussion"),
                )
            except OllamaError as error:
                last_error = error
                context_overflow = (
                    error.kind != "timeout" and self._is_context_overflow(error)
                )
                last_response_issue = (
                    "context_overflow_compacted"
                    if context_overflow
                    else self._transport_failure_reason(error)
                )
                failure_reasons.append(last_response_issue)
                transport_attempts += error.attempts
                if (
                    attempt == 0
                    and context_overflow
                ):
                    self._emit(
                        "llm_action_parse_failed",
                        action,
                        actor,
                        started_at,
                        reason="context_overflow_compacted",
                        will_retry=True,
                    )
                    continue
                self._emit(
                    "llm_action_parse_failed",
                    action,
                    actor,
                    started_at,
                    reason=last_response_issue,
                    will_retry=False,
                )
                break

            transport_attempts += chat_response.attempts
            metadata = chat_response.to_metadata()
            response_metadata_history.append(metadata)
            raw_responses.append(chat_response.content)
            response_text = self._final_response(chat_response.content).strip()
            is_pass = self._is_pass(response_text)
            must_use_seer_info = self._must_use_seer_info(action, actor)
            uses_seer_info = self._references_seer_result(actor, response_text)
            holds_human_seer_result = self._holds_human_seer_result(
                action, actor, response_text
            )
            response_issue = (
                self._discussion_response_issue(actor, response_text)
                if action == "discussion"
                and response_text
                and (
                    not is_pass or has_discussion_output_markers(response_text)
                )
                else None
            )
            if metadata.get("done_reason") == "length" and attempt == 0:
                response_issue = "generation_truncated"
            elif not response_text and response_issue is None:
                response_issue = (
                    "thinking_only_response"
                    if metadata.get("thinking_present")
                    or self._has_thinking(chat_response.content)
                    else "empty_content"
                )
            last_response_issue = response_issue
            if response_issue == "seer_result_without_claim":
                last_spoken_result = extract_public_seer_result(
                    response_text,
                    tuple(player.name for player in self.state.players),
                )
                self._emit(
                    "seer_result_claim_missing",
                    action,
                    actor,
                    started_at,
                    reason=response_issue,
                    will_retry=attempt == 0,
                )
            if response_text and response_issue is None and (
                not must_use_seer_info
                or (
                    not is_pass
                    and (uses_seer_info or holds_human_seer_result)
                )
            ):
                if is_pass:
                    response_text = "パス"
                elif must_use_seer_info and (
                    uses_seer_info or holds_human_seer_result
                ):
                    self._seer_info_used_day[actor.name] = self.state.day
                self.traces.append(
                    ActionTrace(
                        action,
                        actor.name,
                        tuple(raw_responses),
                        (),
                        None,
                        False,
                        failure_reasons=tuple(failure_reasons),
                        response_metadata=tuple(response_metadata_history),
                    )
                )
                self._complete_action(
                    action,
                    actor,
                    started_at,
                    attempt,
                    transport_attempts,
                    False,
                    content=response_text,
                    failure_reasons=tuple(failure_reasons),
                    response_metadata=metadata,
                )
                return response_text

            reason = (
                response_issue
                if response_issue is not None
                else "seer_result_required"
                if must_use_seer_info
                else "invalid_discussion_response"
            )
            failure_reasons.append(reason)
            self._emit(
                "llm_action_parse_failed",
                action,
                actor,
                started_at,
                transport_retries=max(0, transport_attempts - len(raw_responses)),
                reason=reason,
                response_metadata=metadata,
                will_retry=attempt == 0,
            )

        forced_seer_message = (
            self._seer_info_fallback(actor)
            if self._must_use_seer_info(action, actor)
            or (
                action == "discussion"
                and "seer_result_without_claim" in failure_reasons
            )
            else None
        )
        if forced_seer_message is not None:
            self._seer_info_used_day[actor.name] = self.state.day
            if "seer_result_without_claim" in failure_reasons:
                self._emit(
                    "seer_result_claim_fallback",
                    action,
                    actor,
                    started_at,
                    reason="seer_result_without_claim",
                )
        fallback_response = forced_seer_message or "パス"
        self.traces.append(
            ActionTrace(
                action,
                actor.name,
                tuple(raw_responses),
                (),
                None,
                True,
                last_error.to_dict() if last_error is not None else None,
                tuple(failure_reasons),
                tuple(response_metadata_history),
            )
        )
        self._complete_action(
            action,
            actor,
            started_at,
            max(0, len(raw_responses) - 1),
            transport_attempts,
            True,
            content=fallback_response,
            error=last_error,
            failure_reasons=tuple(failure_reasons),
            failure_reason=(
                failure_reasons[-1] if failure_reasons else "unavailable_response"
            ),
            response_metadata=(
                response_metadata_history[-1]
                if response_metadata_history
                else None
            ),
        )
        return fallback_response

    @staticmethod
    def _final_response(response: str) -> str:
        final = _THINK_BLOCK.sub("", response)
        final = _UNCLOSED_THINK_BLOCK.sub("", final)
        return final.replace("</think>", "").strip()

    @staticmethod
    def _is_pass(response: str) -> bool:
        normalized = response.strip().strip("。.!！?？「」『』\"' ")
        return normalized.casefold() in {"パス", "pass"}

    def _must_use_seer_info(self, action: str, actor: Player) -> bool:
        return (
            action == "discussion"
            and actor.role.value == "seer"
            and bool(self.state.seer_results)
            and self._seer_info_used_day.get(actor.name) != self.state.day
        )

    def _holds_human_seer_result(
        self, action: str, actor: Player, response: str
    ) -> bool:
        structured = parse_discussion_output(response)
        return (
            action == "discussion"
            and actor.role is Role.SEER
            and structured is not None
            and structured.claim == "保留"
            and bool(self.state.seer_results)
            and all(result == "human" for result in self.state.seer_results.values())
        )

    def _references_seer_result(self, actor: Player, response: str) -> bool:
        if (
            actor.role.value != "seer"
            or not response
            or not self.state.seer_results
        ):
            return False
        latest_target, _ = self._latest_seer_result()
        if any(
            claim.claimant == actor.name
            and claim.claimed_role == Role.SEER.value
            and claim.target == latest_target
            for claim in self.state.public_role_claims
        ):
            return True
        mentions_result_target = latest_target in response
        usage_terms = (
            "占い",
            "結果",
            "情報",
            "疑い",
            "警戒",
            "信頼",
            "擁護",
            "推理",
            "seer",
            "result",
            "information",
            "suspect",
        )
        return mentions_result_target and any(
            term.casefold() in response.casefold() for term in usage_terms
        )

    def _seer_info_fallback(self, actor: Player) -> str | None:
        if actor.role.value != "seer" or not self.state.seer_results:
            return None
        target, result = self._latest_seer_result()
        if result == "wolf":
            return (
                "【CO】"
                f"占い師CO：対象={target}、判定=人狼、"
                f"理由=実際の占い結果です、投票提案={target}に投票します。\n"
                f"【本文】{target}への投票を提案します。"
            )
        return (
            "【CO】"
            f"占い師CO：対象={target}、判定=人間、"
            f"理由=実際の占い結果を踏まえ、その人への疑いを弱めます、"
            "投票提案=別の可能性を慎重に考えます。\n"
            f"【本文】{target}への疑いは弱め、別の可能性を慎重に考えたいです。"
        )

    def _discussion_retry_instruction(
        self, action: str, actor: Player, response_issue: str | None = None
    ) -> str:
        retry_guidance = {
            "empty_content": "前回は本文が空でした。短い最終回答だけを生成してください。",
            "thinking_only_response": "前回は思考だけで最終回答がありませんでした。思考を出さず、最終回答だけを生成してください。",
            "generation_truncated": "前回は生成上限で途切れました。結論だけを短く最後まで出力してください。",
            "discussion_output_format": "前回は二層形式が不正でした。必ず「【CO】なし または CO内容」の次行に「【本文】議論本文」を出力してください。CO保留は「【CO】保留」とし、本文は空にしないでください。",
            "wrong_response_language": "指定された応答言語で短く発言してください。",
            "third_person_self_reference": "自分自身を第三者のように名前で説明せず、一人称で発言してください。",
            "unknown_player_reference": "実在しないプレイヤー名を使わず、現在のプレイヤーだけに言及してください。",
            "unverified_game_event": "ゲーム台帳にない襲撃・処刑・死亡を事実として述べないでください。",
            "unknown_attacker_claim": "襲撃者は公開されていません。推測を確定した事実として断言しないでください。",
            "unverified_dead_role": "死亡者の真役職は非公開です。役職を確定事項として扱わないでください。",
            "dead_player_action_target": "死亡者は現在の投票・処刑対象にできません。生存者だけに言及してください。",
            "out_of_phase_action_claim": "昼議論中に夜の能力行動を実行済み・実行予定として断言しないでください。",
            "unverified_attributed_speech": "履歴にない発言を他人の発言として引用しないでください。",
            "repeated_speech": "過去の自分の発言を繰り返さず、新しい根拠か提案を加えてください。",
            "question_not_answered_first": "最初の一文で直前の発言者へ直接回答してください。",
            "wolf_result_not_disclosed": "真占い師として占い師COを行い、未公開の人狼判定を対象・判定・理由・投票提案とともに公開してください。",
            "seer_result_without_claim": "占い結果の対象や判定を発言したため、同じ発言内で明示的に占い師COしてください。定型文「占い師CO：対象=対象名、判定=人狼/人間、理由=理由、投票提案=提案」を使ってください。結果を公開しない場合は対象名と判定を述べず、結果を伏せた発言にしてください。",
            "inconsistent_seer_result": "あなた自身が以前に主張した占い対象の判定を変更せず、過去の主張と一貫させてください。",
            "inconsistent_role_claim": "以前の役職主張を別の役職へ変更せず、一貫させてください。",
            "vague_wait_and_see": "「様子見」や「情報不足」だけの繰り返しは避け、具体的な対象・根拠・判断を述べてください。",
        }
        if response_issue in retry_guidance:
            return f"\n前回の応答を採用できませんでした。{retry_guidance[response_issue]}"
        if self._must_use_seer_info(action, actor):
            target, _ = self._latest_seer_result()
            return (
                "\n前回の応答は空、パス、または占い結果を活用していませんでした。"
                f"今回は必ず{target}の占い結果を踏まえた判断を具体的に話してください。"
                "結果を公開せずに活用しても構いません。"
            )
        return "\n前回の応答は空または空白のみでした。内容のある発言をしてください。"

    def _short_discussion_retry_prompt(
        self,
        actor: Player,
        response_issue: str | None,
        recent_messages: tuple[Message, ...] | None = None,
    ) -> str:
        public_claims = list(self.state.public_role_claims)
        recorded_seer_claims = {
            (claim.day, claim.claimant, claim.target, claim.result)
            for claim in public_claims
            if claim.claimed_role == Role.SEER.value
        }
        for claim in self.state.seer_claims:
            key = (claim.day, claim.claimant, claim.target, claim.result)
            if key not in recorded_seer_claims:
                public_claims.append(
                    PublicRoleClaim(
                        day=claim.day,
                        claimant=claim.claimant,
                        claimed_role=Role.SEER.value,
                        target=claim.target,
                        result=claim.result,
                    )
                )
        public_claims = [
            f"{claim.claimant}は"
            f"{ROLE_NAMES_JA.get(Role(claim.claimed_role), claim.claimed_role)}を主張"
            + (
                f"（{claim.target}={'人狼' if claim.result == 'wolf' else '人間'}）"
                if claim.target is not None and claim.result is not None
                else ""
            )
            for claim in public_claims[-4:]
        ]
        death_events = sorted(
            [
                (day, "襲撃", name)
                for day, name in self.state.attack_deaths
            ]
            + [(day, "処刑", name) for day, name in self.state.executions]
        )[-3:]
        death_summary = "、".join(
            f"{day}日目{name}死亡（{cause}）" for day, cause, name in death_events
        ) or "なし"
        dead_players = tuple(
            player.name for player in self.state.players if not player.is_alive
        )
        latest_vote = (
            "、".join(
                f"{voter}→{target}"
                for voter, target in self.state.vote_history[-1][2].items()
            )
            if self.state.vote_history
            else "なし"
        )
        conversation = (
            recent_messages
            if recent_messages is not None
            else tuple(self.state.conversation_log)
        )
        recent_speech = [
            f"{message.speaker}: {message.content[:240]}"
            for message in conversation[-3:]
            if message.speaker != "system"
        ]
        private_seer_results = (
            [
                f"{target}: {'人狼' if result == 'wolf' else '人間'}"
                for _, target, result in self.state.seer_result_history[-3:]
            ]
            if actor.role is Role.SEER
            else []
        )
        if (
            actor.role is Role.SEER
            and not private_seer_results
            and self.state.seer_results
        ):
            private_seer_results = [
                f"{target}: {'人狼' if result == 'wolf' else '人間'}"
                for target, result in list(self.state.seer_results.items())[-3:]
            ]
        guidance = self._discussion_retry_instruction(
            "discussion", actor, response_issue
        ).strip()
        previous_message = next(
            (message for message in reversed(conversation) if message.speaker != "system"),
            None,
        )
        if PromptBuilder._is_directed_to_player(
            actor,
            previous_message,
            tuple(message for message in conversation if message.speaker != "system"),
        ):
            guidance += (
                f" 最初の一文で{previous_message.speaker}への回答をしてください。"
            )
        return (
            "短い再回答です。省略された古い会話は参照せず、次の事実と直近発言だけを使ってください。\n"
            "【ゲームシステムが確定した事実】\n"
            f"- あなたのプレイヤー名：{actor.name}\n"
            f"- あなたの役職：{ROLE_NAMES_JA[actor.role]}\n"
            f"- 現在の日数：{self.state.day}日目\n"
            "- 現在のフェーズ：昼議論\n"
            f"- 昼議論ターン：{self.state.discussion_turn}/3\n"
            f"- 現在の生存者：{', '.join(self.state.alive_players)}\n"
            f"- 既知の死亡者：{', '.join(dead_players) or 'なし'}\n"
            f"- 確定した死亡履歴（直近3件）：{death_summary}\n"
            f"- 最新の投票結果：{latest_vote}\n"
            f"- 応答言語：{actor.response_language}\n"
            "【セオリー（推論・確定事実ではない）】未公開の人狼判定は早期CO・公開を優先。"
            "人間判定だけならCO・結果の保留も可。単独COは1巡後も対抗なしなら真寄りだが真確定ではない。"
            "人狼・狂人の騙りや対抗COは選択肢であり必須ではない。\n"
            f"【公開CO・結果】{'; '.join(public_claims) or 'なし'}。\n"
            f"【直近の公開発言（未確認）】{'; '.join(recent_speech) or 'なし'}。\n"
            "【情報区分】公開発言・CO主張は未確認情報で、確定事実として扱わないでください。\n"
            + (
                f"【占い師本人だけの非公開情報】{'; '.join(private_seer_results)}。\n"
                if private_seer_results
                else ""
            )
            + f"【修正指示】{guidance}\n"
            "回答は必ず「【CO】なし/保留/CO内容」の次行に「【本文】本文」の2行構造にしてください。"
            "占い結果を公開する場合のみ「【CO】占い師」＋「【結果】対象=Player X 判定=人狼/人間」の行を追加できます。"
            "本文は短く1～2文にし、直前の質問があれば本文の最初に回答してください。"
        )

    @staticmethod
    def _result_label(result: str) -> str:
        return "人狼" if result == "wolf" else "人間"

    def _seer_co_correction_prompt(
        self, actor: Player, spoken_result: tuple[str, str] | None
    ) -> str:
        """CO欠落を補正する、COと結果だけの短い再要求プロンプト。"""
        lines = [
            "【CO補正】占い結果を述べる発言に占い師COがありません。"
            "同じ発言内で明示的に占い師COしてください。",
            f"- あなた：{actor.name}",
        ]
        if actor.role is Role.SEER and self.state.seer_results:
            target, result = self._latest_seer_result()
            lines.append(
                f"- あなたの占い結果：対象={target}、判定={self._result_label(result)}"
            )
            lines.append("この結果を占い師COとセットで公開してください。")
        elif spoken_result is not None:
            lines.append(
                f"- 前回述べた結果：対象={spoken_result[0]}、"
                f"判定={self._result_label(spoken_result[1])}"
            )
            lines.append("この結果を占い師COとセットで出力してください。")
        lines.extend(
            [
                "COと結果だけを次の2行で出力してください。他の内容は不要です。",
                "【CO】占い師",
                "【結果】対象=Player X 判定=人狼/人間",
                "理由・投票提案は任意で、付ける場合は3行目に【本文】を付けてください。",
            ]
        )
        return "\n".join(lines)

    def _discussion_response_issue(self, actor: Player, response: str) -> str | None:
        structured_output = parse_discussion_output(response)
        if has_discussion_output_markers(response) and structured_output is None:
            return "discussion_output_format"
        body = structured_output.body if structured_output is not None else response
        claim_content = (
            structured_output.claim
            if structured_output is not None
            and structured_output.claim not in {"なし", "保留"}
            else ""
            if structured_output is not None
            else response
        )
        result_block = (
            structured_output.result if structured_output is not None else ""
        )
        result_content = (
            "\n".join(part for part in (claim_content, result_block, body) if part)
            if structured_output is not None
            else response
        )
        known_players = {player.name for player in self.state.players}
        explicit_seer_co = is_public_seer_claim(claim_content)
        public_result = extract_public_seer_result(result_content, tuple(known_players))

        # 占い結果を公開する応答は、本文の内容を検証する前に明示的な占い師COを要求する。
        if result_block and not explicit_seer_co:
            return "seer_result_without_claim"
        if public_result is not None and not explicit_seer_co:
            target, result = public_result
            if (
                actor.role is Role.SEER
                and self.state.seer_results.get(target) == result
                and not self._has_published_seer_result(actor.name, target, result)
            ):
                return "seer_result_without_claim"

        if body and actor.response_language == "日本語" and not re.search(
            r"[\u3040-\u30ff\u3400-\u9fff]", body
        ):
            return "wrong_response_language"

        history = tuple(
            message
            for message in self.state.conversation_log
            if message.speaker != "system"
        )
        previous_message = history[-1] if history else None
        if body and PromptBuilder._is_directed_to_player(
            actor, previous_message, history
        ):
            first_sentence = re.split(r"[。！？\n]", body, maxsplit=1)[0]
            if previous_message is not None and previous_message.speaker not in first_sentence:
                return "question_not_answered_first"

        own_name = re.escape(actor.name)
        if re.search(
            rf"(?m)^\s*{own_name}(?:さん)?(?:は|が|も)(?=.{{0,32}}(?:人狼|村人|占い師|狂人|発言|意見|怪しい|投票))",
            body,
        ):
            return "third_person_self_reference"

        mentioned_players = re.findall(r"Player\s+\d+", response)
        if any(name not in known_players for name in mentioned_players):
            return "unknown_player_reference"
        claimed_role = extract_public_role_claim(claim_content)
        public_claims = [
            claim
            for claim in self.state.public_role_claims
            if claim.claimant == actor.name
        ]
        if claimed_role is not None and any(
            claim.claimed_role != claimed_role for claim in public_claims
        ):
            return "inconsistent_role_claim"
        if public_result is not None:
            target, result = public_result
            if any(
                claim.claimant == actor.name
                and claim.target == target
                and claim.result is not None
                and claim.result != result
                for claim in self.state.seer_claims
            ) or any(
                claim.claimant == actor.name
                and claim.claimed_role == Role.SEER.value
                and claim.target == target
                and claim.result is not None
                and claim.result != result
                for claim in self.state.public_role_claims
            ):
                return "inconsistent_seer_result"
        if actor.role is Role.SEER:
            unpublished_wolf_results = [
                target
                for target, result in self.state.seer_results.items()
                if result == "wolf"
                and not any(
                    claim.claimant == actor.name
                    and claim.target == target
                    and claim.result == "wolf"
                    for claim in self.state.seer_claims
                )
            ]
            if unpublished_wolf_results and (
                not is_public_seer_claim(claim_content)
                or public_result
                not in {(target, "wolf") for target in unpublished_wolf_results}
            ):
                return "wolf_result_not_disclosed"
        dead_players = {
            player.name for player in self.state.players if not player.is_alive
        }
        for dead_name in dead_players:
            name = re.escape(dead_name)
            if re.search(
                rf"{name}(?:さん)?(?:は|が)?[^。！？\n]{{0,16}}"
                rf"(?:人狼|狂人|占い師|村人)(?:だった|でした|です|だ|と判明|と確定)",
                body,
            ):
                return "unverified_dead_role"
            if re.search(
                rf"{name}(?:さん)?(?:に|へ)(?:は)?"
                rf"(?:投票(?:する|したい|すべき|しよう)?|"
                rf"票を入れる(?:べき|べきです|べきだ|たい)?|"
                rf"処刑(?:する|したい|すべき)?|吊(?:る|りたい|るべき))",
                body,
            ) or re.search(
                rf"(?:投票先|処刑対象)(?:は|として|に)\s*{name}(?:さん)?",
                body,
            ):
                return "dead_player_action_target"

        if re.search(
            r"(?:今夜|夜に)(?:は)?(?:私は|僕は|私が|僕が)?"
            r"(?:占う|占います|襲撃する|襲撃します)",
            body,
        ):
            return "out_of_phase_action_claim"

        for player in self.state.players:
            name = re.escape(player.name)
            if actor.role is not Role.WOLF and (
                re.search(
                    rf"{name}(?:さん)?(?:が|は)[^。！？\n]{{0,20}}(?:襲撃した|襲撃しました|襲った|襲いました|襲撃者(?:です|だ))",
                    body,
                )
                or re.search(
                    rf"襲撃者は{name}(?:さん)?(?:です|だ)",
                    body,
                )
            ):
                return "unknown_attacker_claim"
            if re.search(
                rf"{name}(?:さん)?(?:は|が)?[^。！？\n]{{0,20}}襲撃され(?:た|ました)",
                body,
            ) and not any(victim == player.name for _, victim in self.state.attack_deaths):
                return "unverified_game_event"
            if re.search(
                rf"{name}(?:さん)?(?:は|が)?[^。！？\n]{{0,20}}処刑され(?:た|ました)",
                body,
            ) and not any(executed == player.name for _, executed in self.state.executions):
                return "unverified_game_event"
            if re.search(
                rf"{name}(?:さん)?(?:は|が)?[^。！？\n]{{0,20}}死亡し(?:た|ました)",
                body,
            ) and player.is_alive:
                return "unverified_game_event"

        known_speech = history
        for player in self.state.players:
            attributed_quote = re.compile(
                rf"{re.escape(player.name)}(?:さん)?(?:は|が)[^。！？\n]{{0,24}}[「『\"]([^」』\"]+)[」』\"]"
            )
            for match in attributed_quote.finditer(body):
                quote = match.group(1).strip()
                if quote and not any(
                    message.speaker == player.name and quote in message.content
                    for message in known_speech
                ):
                    return "unverified_attributed_speech"

        previous_speeches = [
            message.content
            for message in known_speech
            if message.speaker == actor.name
        ]
        normalized = re.sub(r"[\W_]+", "", body, flags=re.UNICODE)
        for previous in previous_speeches:
            prior_normalized = re.sub(r"[\W_]+", "", previous, flags=re.UNICODE)
            if len(normalized) >= 20 and normalized == prior_normalized:
                return "repeated_speech"
            if min(len(normalized), len(prior_normalized)) >= 40 and (
                SequenceMatcher(None, normalized, prior_normalized).ratio() >= 0.92
            ):
                return "repeated_speech"
        # 様子見・情報不足のみの反復を検出（パスは除外）
        if not self._is_pass(response):
            vague_patterns = [
                r"様子見",
                r"様子を",
                r"情報不足",
                r"まだわからない",
                r"判断できない",
                r"何とも言えない",
                r"様子を伺",
            ]
            import re as _re
            if any(_re.search(p, body, _re.IGNORECASE) for p in vague_patterns):
                # 単に様子見と言っているだけで新しい情報がない場合
                if len(body.strip()) < 60:
                    return "vague_wait_and_see"
        return None

    def _latest_seer_result(self) -> tuple[str, str]:
        if self.state.seer_result_history:
            _, target, result = self.state.seer_result_history[-1]
            return target, result
        return next(reversed(self.state.seer_results.items()))

    def _has_published_seer_result(
        self, claimant: str, target: str, result: str
    ) -> bool:
        return any(
            claim.claimant == claimant
            and claim.target == target
            and claim.result == result
            for claim in self.state.seer_claims
        ) or any(
            claim.claimant == claimant
            and claim.claimed_role == Role.SEER.value
            and claim.target == target
            and claim.result == result
            for claim in self.state.public_role_claims
        )

    def _options_for(self, actor: Player) -> dict[str, object] | None:
        options = (
            {
                key: value
                for key, value in self.options.items()
                if value is not None
            }
            if self.options is not None
            else {}
        )
        if actor.num_ctx is not None:
            options["num_ctx"] = actor.num_ctx
        if actor.num_predict is not None:
            options["num_predict"] = actor.num_predict
        return options or None

    def _chat_with_claim_tools(
        self,
        actor: Player,
        messages: list[dict[str, object]],
        action: str,
        *,
        increase_generation: bool,
        started_at: float,
        request_number: int = 1,
        request_reason: str = "初回",
        tool_based: bool = False,
    ) -> ChatResponse:
        # ツール非対応と判明したモデルは、以後ツールを送らず文章CO経路だけを使う。
        capabilities = self.state.model_capabilities.get(actor.model, {})
        tools = (
            [CLAIM_ROLE_TOOL]
            if action == "discussion"
            and capabilities.get("claim_role_tool", True)
            and callable(getattr(self.client, "chat_with_metadata", None))
            else None
        )
        try:
            response = self._chat(
                actor,
                messages,
                increase_generation=increase_generation,
                tools=tools,
                action=action,
                started_at=started_at,
                request_number=request_number,
                request_reason=request_reason,
                tool_based=tool_based,
            )
        except OllamaError as error:
            if tools is None or not self._is_tool_call_unsupported(error):
                raise
            self._record_tool_unsupported(actor, action, started_at)
            tools = None
            response = self._chat(
                actor,
                messages,
                increase_generation=increase_generation,
                action=action,
                started_at=started_at,
                request_number=request_number,
                request_reason=request_reason,
                tool_based=False,
            )
        for _ in range(2):
            if not response.tool_calls:
                return response
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            }
                        }
                        for call in response.tool_calls
                    ],
                }
            )
            for call in response.tool_calls:
                result = self._handle_public_tool_call(
                    actor, call, action, started_at
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": call.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            response = self._chat(
                actor,
                messages,
                tools=tools,
                action=action,
                started_at=started_at,
                request_number=request_number,
                request_reason=request_reason,
                tool_based=tool_based,
            )
        if response.tool_calls:
            return ChatResponse(
                content="",
                attempts=response.attempts,
                done_reason=response.done_reason,
                content_length=0,
                thinking_present=response.thinking_present,
            )
        return response

    @staticmethod
    def _is_tool_call_unsupported(error: OllamaError) -> bool:
        message = error.message.casefold()
        return any(
            marker in message
            for marker in (
                "does not support tools",
                "doesn't support tools",
                "tool calling is not supported",
                "tool calls are not supported",
                "tools are not supported",
                "ツールに対応していません",
                "ツール呼び出しに対応していません",
            )
        )

    def _record_tool_unsupported(
        self, actor: Player, action: str, started_at: float
    ) -> None:
        capabilities = self.state.model_capabilities.setdefault(actor.model, {})
        first_detection = capabilities.get("claim_role_tool") is not False
        capabilities["claim_role_tool"] = False
        if first_detection:
            self._emit(
                "claim_tool_unavailable",
                action,
                actor,
                started_at,
                reason="model_does_not_support_tools",
            )

    def _handle_public_tool_call(
        self,
        actor: Player,
        call: ToolCall,
        action: str,
        started_at: float,
    ) -> dict[str, str]:
        if call.name != "claim_role":
            return {
                "status": "rejected",
                "message": "この公開ツールは利用できません。",
            }
        if action != "discussion" or not actor.is_alive:
            self._emit(
                "claim_role_rejected",
                action,
                actor,
                started_at,
                rejection_reason=(
                    "dead_player" if not actor.is_alive else "invalid_phase"
                ),
            )
            return {
                "status": "rejected",
                "message": "現在のフェーズでは主張を公開できません。",
            }

        def reject(message: str, reason: str) -> dict[str, str]:
            self._emit(
                "claim_role_rejected",
                action,
                actor,
                started_at,
                rejection_reason=reason,
            )
            return {"status": "rejected", "message": message}

        args = call.arguments
        role = args.get("role")
        if not isinstance(role, str) or role not in {item.value for item in Role}:
            return reject("roleの指定が不正です。", "invalid_role")
        target = args.get("target")
        result = args.get("result")
        reason = args.get("reason")
        vote_proposal = args.get("vote_proposal")
        for field_name, value, max_length in (
            ("target", target, 128),
            ("reason", reason, 500),
            ("vote_proposal", vote_proposal, 300),
        ):
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > max_length
            ):
                return reject(
                    f"{field_name}の指定が不正です。",
                    f"invalid_{field_name}",
                )
        if result is not None and (
            not isinstance(result, str) or result not in {"wolf", "human"}
        ):
            return reject("resultの指定が不正です。", "invalid_result")
        if role == Role.SEER.value:
            if (target is None) != (result is None):
                return reject(
                    "占い結果にはtargetとresultの両方が必要です。",
                    "incomplete_seer_result",
                )
            if target is not None and target not in {
                player.name for player in self.state.players
            }:
                return reject(
                    "対象プレイヤーが存在しません。",
                    "unknown_target",
                )
        elif target is not None or result is not None:
            return reject(
                "占い結果は占い師主張でのみ公開できます。",
                "non_seer_result",
            )

        existing_claims = [
            claim
            for claim in self.state.public_role_claims
            if claim.claimant == actor.name
        ]
        previous_seer_claims = [
            claim
            for claim in self.state.seer_claims
            if claim.claimant == actor.name
        ]
        prior_role = next(
            (
                claim.claimed_role
                for claim in existing_claims
            ),
            Role.SEER.value if previous_seer_claims else None,
        )
        if prior_role is not None and prior_role != role:
            return reject(
                "以前の役職主張と矛盾しています。",
                "inconsistent_role_claim",
            )
        previous_result = next(
            (
                claim.result
                for claim in [*existing_claims, *previous_seer_claims]
                if role == Role.SEER.value
                and claim.target == target
                and claim.result is not None
            ),
            None,
        )
        if previous_result is not None and previous_result != result:
            return reject(
                "以前の占い結果主張と矛盾しています。",
                "inconsistent_seer_result",
            )
        duplicate = next(
            (
                claim
                for claim in existing_claims
                if claim.claimed_role == role
                and claim.target == target
                and claim.result == result
            ),
            None,
        )
        if duplicate is not None or (
            role == Role.SEER.value
            and target is not None
            and any(
                claim.target == target and claim.result == result
                for claim in previous_seer_claims
            )
        ):
            self._emit(
                "claim_role_rejected",
                action,
                actor,
                started_at,
                rejection_reason="duplicate_claim",
            )
            return {
                "status": "already_published",
                "message": "同一の公開主張はすでに記録されています。",
            }

        self.state.public_role_claims.append(
            PublicRoleClaim(
                day=self.state.day,
                claimant=actor.name,
                claimed_role=role,
                target=target,
                result=result,
                reason=reason.strip() if isinstance(reason, str) else None,
                vote_proposal=(
                    vote_proposal.strip()
                    if isinstance(vote_proposal, str)
                    else None
                ),
            )
        )
        if role == Role.SEER.value:
            if not previous_seer_claims:
                self.state.seer_claims.append(
                    PublicSeerClaim(day=self.state.day, claimant=actor.name)
                )
            if target is not None and result is not None:
                self.state.seer_claims.append(
                    PublicSeerClaim(
                        day=self.state.day,
                        claimant=actor.name,
                        target=target,
                        result=result,
                    )
                )
        self._emit(
            "claim_role_published",
            action,
            actor,
            started_at,
            claimed_role=role,
            target=target,
            claimed_result=result,
            reason=reason,
            vote_proposal=vote_proposal,
            source="tool",
        )
        return {
            "status": "published",
            "message": "公開主張を記録しました。",
        }

    def _chat(
        self,
        actor: Player,
        messages: list[dict[str, object]],
        increase_generation: bool = False,
        tools: list[dict[str, object]] | None = None,
        action: str | None = None,
        started_at: float | None = None,
        request_number: int = 1,
        request_reason: str = "初回",
        tool_based: bool = False,
    ) -> ChatResponse:
        options = self._options_for(actor)
        if increase_generation:
            current_limit = options.get("num_predict", 512) if options else 512
            if isinstance(current_limit, int) and not isinstance(current_limit, bool):
                options = dict(options or {})
                options["num_predict"] = max(current_limit, min(current_limit * 2, 8192))
        original_timeout = getattr(self.client, "timeout", None)
        effective_timeout = (
            actor.timeout
            if actor.timeout is not None
            else original_timeout
        )
        if original_timeout is not None and actor.timeout is not None:
            self.client.timeout = actor.timeout
        if (
            action is not None
            and started_at is not None
            and callable(getattr(self.client, "context_diagnostics", None))
        ):
            context_diagnostics = self.client.context_diagnostics(actor.model)
            requested_num_ctx = options.get("num_ctx") if options else None
            if actor.num_ctx is not None:
                num_ctx_source = "player_setting"
            elif self.options is not None and self.options.get("num_ctx") is not None:
                num_ctx_source = "cli_setting"
            elif requested_num_ctx is not None:
                num_ctx_source = "api_request"
            else:
                num_ctx_source = "ollama_service_default_or_environment"
            self._emit(
                "llm_request_configuration",
                action,
                actor,
                started_at,
                request_number=request_number,
                request_reason=request_reason,
                tool_based=tool_based,
                request_options=options or {},
                request_timeout_seconds=effective_timeout,
                num_ctx_source=num_ctx_source,
                context_diagnostics=context_diagnostics,
            )
        try:
            if hasattr(self.client, "_last_chat_response"):
                delattr(self.client, "_last_chat_response")
            if tools is not None and callable(
                getattr(self.client, "chat_with_metadata", None)
            ):
                return self.client.chat_with_metadata(
                    actor.model,
                    messages,
                    options=options,
                    tools=tools,
                )
            if options is not None:
                content = self.client.chat(actor.model, messages, options=options)
            else:
                content = self.client.chat(actor.model, messages)
            metadata = getattr(self.client, "_last_chat_response", None)
            if isinstance(metadata, ChatResponse):
                return metadata
            return ChatResponse(
                content=content,
                attempts=max(1, getattr(self.client, "_last_attempts", 1)),
                content_length=len(content),
                thinking_present=self._has_thinking(content),
            )
        finally:
            if original_timeout is not None:
                self.client.timeout = original_timeout

    @staticmethod
    def _has_thinking(response: str) -> bool:
        return bool(_THINK_BLOCK.search(response) or _UNCLOSED_THINK_BLOCK.search(response))

    @staticmethod
    def _is_context_overflow(error: OllamaError) -> bool:
        message = error.message.casefold()
        return any(
            marker in message
            for marker in (
                "context length",
                "context window",
                "maximum context",
                "prompt is too long",
                "input length",
                "num_ctx",
                "コンテキスト長",
                "コンテキストが長",
            )
        )

    @staticmethod
    def _transport_failure_reason(error: OllamaError) -> str:
        return {
            "timeout": "timeout",
            "connection": "communication_failure",
            "http": "communication_failure",
            "invalid_response": "invalid_ollama_response",
        }.get(error.kind, "communication_failure")

    def _start_action(self, action: str, actor: Player) -> float:
        started_at = time.perf_counter()
        self._emit(
            "llm_action_started",
            action,
            actor,
            started_at,
        )
        return started_at

    def _complete_action(
        self,
        action: ActionKind | str,
        actor: Player,
        started_at: float,
        semantic_retries: int,
        transport_attempts: int,
        fallback_used: bool,
        selected_target: str | None = None,
        content: str | None = None,
        error: OllamaError | None = None,
        failure_reason: str | None = None,
        failure_reasons: tuple[str, ...] = (),
        response_metadata: dict[str, str | int | bool | None] | None = None,
    ) -> None:
        self._emit(
            "llm_action_completed",
            action,
            actor,
            started_at,
            elapsed_seconds=round(time.perf_counter() - started_at, 3),
            semantic_retries=semantic_retries,
            transport_retries=max(0, transport_attempts - (semantic_retries + 1)),
            total_retries=semantic_retries
            + max(0, transport_attempts - (semantic_retries + 1)),
            fallback_used=fallback_used,
            selected_target=selected_target,
            content=content,
            error=error.to_dict() if error is not None else None,
            failure_reason=failure_reason,
            failure_reasons=failure_reasons,
            response_metadata=response_metadata,
        )

    def _emit(
        self,
        event_type: str,
        action: ActionKind | str,
        actor: Player,
        started_at: float,
        **details: object,
    ) -> None:
        if self.event_callback is None:
            return
        action_name = action.value if isinstance(action, ActionKind) else action
        phase_labels = {
            "night_wolf": "夜・人狼相談",
            "night_seer": "夜・占い",
            "day_discussion": "昼・議論",
            "voting": "投票",
        }
        self.event_callback(
            {
                "type": event_type,
                "day": self.state.day,
                "phase": self.state.phase.value,
                "phase_label": phase_labels.get(
                    self.state.phase.value, self.state.phase.value
                ),
                "player": actor.name,
                "model": actor.model,
                "action": action_name,
                "private": action_name
                in {"wolf_discussion", ActionKind.ATTACK.value, ActionKind.SEER.value},
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                **details,
            }
        )

    @staticmethod
    def _label(action: ActionKind) -> str:
        return {
            ActionKind.ATTACK: "襲撃対象",
            ActionKind.SEER: "占い対象",
            ActionKind.VOTE: "投票",
        }[action]

    def _player(self, name: str) -> Player:
        return next(player for player in self.state.players if player.name == name)
