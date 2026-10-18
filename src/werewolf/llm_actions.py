import random
import re
import time
from dataclasses import dataclass
from typing import Callable

from .action_parser import ActionKind, ActionParser
from .engine import ActionPolicy
from .models import GameState, Message, Player
from .ollama import OllamaClient, OllamaError
from .prompts import PromptBuilder, PromptContext

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
        prompt_ctx = PromptBuilder.build_seer_prompt(
            actor=actor,
            alive_players=candidates,
            seer_results=results,
            day=self.state.day,
            state=self.state,
        )
        return self._select_target(
            ActionKind.SEER, actor, candidates, prompt_ctx, started_at
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
        options = self._options_for(actor)
        raw_responses: list[str] = []
        parsed_targets: list[str | None] = []
        last_error: OllamaError | None = None
        transport_attempts = 0
        semantic_retries = 0

        for request_number in range(2):
            messages = [
                {"role": "system", "content": prompt_ctx.system_prompt},
                {"role": "user", "content": user_prompts[-1]},
            ]
            try:
                if options is not None:
                    response = self.client.chat(
                        actor.model,
                        messages,
                        options=options,
                    )
                else:
                    response = self.client.chat(
                        actor.model,
                        messages,
                    )
            except OllamaError as error:
                last_error = error
                transport_attempts += error.attempts
                compacted = prompt_ctx.compact(recent_lines=2)
                if (
                    request_number == 0
                    and self._is_context_overflow(error)
                    and compacted.user_prompt != prompt_ctx.user_prompt
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
                        compacted.user_prompt
                        + "\n古い会話を圧縮しました。確定事実と直前の発言を優先してください。"
                    )
                    semantic_retries += 1
                    continue
                break

            transport_attempts += max(1, getattr(self.client, "_last_attempts", 1))
            raw_responses.append(response)
            target = ActionParser.parse(self._final_response(response), action)
            parsed_targets.append(target)
            if ActionParser.validate_target(
                target, valid_candidates, actor.name if forbid_self else None
            ):
                trace = ActionTrace(
                    action, actor.name, tuple(raw_responses),
                    tuple(parsed_targets), target, False,
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
                )
                return target

            self._emit(
                "llm_action_parse_failed",
                action,
                actor,
                started_at,
                transport_retries=max(0, transport_attempts - len(raw_responses)),
                parsed_target=target,
                will_retry=request_number == 0,
            )
            if request_number == 0:
                semantic_retries += 1
                user_prompts.append(
                    f"{prompt_ctx.user_prompt}\n前回の応答は形式または候補が不正でした。"
                    f"候補から1名を選び、最後に必ず「{self._label(action)}：名前」とだけ明示してください。"
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

    def _free_text_with_context(
        self,
        action: str,
        actor: Player,
        prompt_ctx: PromptContext,
        fallback: str,
        started_at: float,
    ) -> str:
        raw_responses: list[str] = []
        last_error: OllamaError | None = None
        transport_attempts = 0
        for attempt in range(2):
            user_prompt = prompt_ctx.user_prompt
            if attempt:
                user_prompt += self._discussion_retry_instruction(action, actor)
            messages = [
                {"role": "system", "content": prompt_ctx.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                options = self._options_for(actor)
                if options is not None:
                    response = self.client.chat(
                        actor.model, messages, options=options
                    )
                else:
                    response = self.client.chat(actor.model, messages)
            except OllamaError as error:
                last_error = error
                transport_attempts += error.attempts
                compacted = prompt_ctx.compact(recent_lines=2)
                if (
                    attempt == 0
                    and self._is_context_overflow(error)
                    and compacted.user_prompt != prompt_ctx.user_prompt
                ):
                    prompt_ctx = compacted
                    self._emit(
                        "llm_action_parse_failed",
                        action,
                        actor,
                        started_at,
                        reason="context_overflow_compacted",
                        will_retry=True,
                    )
                    continue
                break

            transport_attempts += max(1, getattr(self.client, "_last_attempts", 1))
            raw_responses.append(response)
            response_text = self._final_response(response).strip()
            is_pass = self._is_pass(response_text)
            must_use_seer_info = self._must_use_seer_info(action, actor)
            uses_seer_info = self._references_seer_result(actor, response_text)
            if response_text and (
                not must_use_seer_info or (not is_pass and uses_seer_info)
            ):
                if is_pass:
                    response_text = "パス"
                elif must_use_seer_info:
                    self._seer_info_used_day[actor.name] = self.state.day
                self.traces.append(
                    ActionTrace(
                        action,
                        actor.name,
                        tuple(raw_responses),
                        (),
                        None,
                        False,
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
                )
                return response_text

            if attempt == 0:
                reason = (
                    "empty_response"
                    if not response_text
                    else "seer_result_required"
                    if must_use_seer_info
                    else "invalid_discussion_response"
                )
                self._emit(
                    "llm_action_parse_failed",
                    action,
                    actor,
                    started_at,
                    transport_retries=max(0, transport_attempts - len(raw_responses)),
                    reason=reason,
                    will_retry=True,
                )

        forced_seer_message = (
            self._seer_info_fallback(actor)
            if self._must_use_seer_info(action, actor)
            else None
        )
        if forced_seer_message is not None:
            self._seer_info_used_day[actor.name] = self.state.day
        fallback_response = forced_seer_message or fallback
        self.traces.append(
            ActionTrace(
                action,
                actor.name,
                tuple(raw_responses),
                (),
                None,
                True,
                last_error.to_dict() if last_error is not None else None,
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

    def _references_seer_result(self, actor: Player, response: str) -> bool:
        if (
            actor.role.value != "seer"
            or not response
            or not self.state.seer_results
        ):
            return False
        latest_target, _ = self._latest_seer_result()
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
                f"占い結果を踏まえると、{target}さんは人狼でした。"
                "この情報を共有し、今日は特に警戒して議論したいです。"
            )
        return (
            f"占い結果を踏まえると、{target}さんは人間でした。"
            "現時点ではその人への疑いを弱め、別の可能性を慎重に考えたいです。"
        )

    def _discussion_retry_instruction(self, action: str, actor: Player) -> str:
        if self._must_use_seer_info(action, actor):
            target, _ = self._latest_seer_result()
            return (
                "\n前回の応答は空、パス、または占い結果を活用していませんでした。"
                f"今回は必ず{target}の占い結果を踏まえた判断を具体的に話してください。"
                "結果を公開せずに活用しても構いません。"
            )
        return "\n前回の応答は空または空白のみでした。内容のある発言をしてください。"

    def _latest_seer_result(self) -> tuple[str, str]:
        if self.state.seer_result_history:
            _, target, result = self.state.seer_result_history[-1]
            return target, result
        return next(reversed(self.state.seer_results.items()))

    def _options_for(self, actor: Player) -> dict[str, object] | None:
        options = dict(self.options) if self.options is not None else {}
        if actor.num_ctx is not None:
            options["num_ctx"] = actor.num_ctx
        return options or None

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
