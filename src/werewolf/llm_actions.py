import random
from dataclasses import dataclass

from .action_parser import ActionKind, ActionParser
from .engine import ActionPolicy
from .models import GameState, Message, Player
from .ollama import OllamaClient, OllamaError
from .prompts import PromptBuilder, PromptContext


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
    ) -> None:
        self.state = state
        self.client = client
        self.options = options
        self._rng = random.Random(state.seed if seed is None else seed)
        self.traces: list[ActionTrace] = []

    def wolf_discuss(
        self, wolf: str, partner: str, history: tuple[Message, ...]
    ) -> str:
        player = self._player(wolf)
        prompt_ctx = PromptBuilder.build_wolf_discuss_prompt(
            actor=player,
            partner_name=partner,
            alive_players=tuple(self.state.alive_players),
            wolf_chat_history=history,
        )
        return self._free_text_with_context(
            "wolf_discussion", player, prompt_ctx, f"{wolf} は相談に応答できませんでした。"
        )

    def choose_attack(self, wolves: tuple[str, ...], candidates: tuple[str, ...]) -> str:
        actor = self._player(wolves[0])
        valid_candidates = self._valid_candidates(
            actor, candidates, forbidden=wolves
        )
        prompt_ctx = PromptBuilder.build_attack_prompt(
            actor=actor,
            wolves=wolves,
            valid_candidates=valid_candidates,
        )
        return self._select_target(
            ActionKind.ATTACK, actor, valid_candidates, prompt_ctx, forbidden=wolves
        )

    def choose_seer_target(
        self, seer: str, candidates: tuple[str, ...], results: tuple[tuple[str, str], ...]
    ) -> str:
        actor = self._player(seer)
        prompt_ctx = PromptBuilder.build_seer_prompt(
            actor=actor,
            alive_players=candidates,
            seer_results=results,
        )
        return self._select_target(ActionKind.SEER, actor, candidates, prompt_ctx)

    def discuss(
        self, player: str, day: int, alive: tuple[str, ...], history: tuple[Message, ...]
    ) -> str:
        actor = self._player(player)
        prompt_ctx = PromptBuilder.build_discussion_prompt(
            actor=actor,
            alive_players=alive,
            conversation_history=history,
        )
        return self._free_text_with_context(
            "discussion", actor, prompt_ctx, f"{player} は発言できませんでした。"
        )

    def vote(
        self, voter: str, candidates: tuple[str, ...], day: int, runoff: bool
    ) -> str:
        actor = self._player(voter)
        valid_candidates = self._valid_candidates(
            actor, candidates, forbid_self=True
        )
        prompt_ctx = PromptBuilder.build_vote_prompt(
            actor=actor,
            alive_players=tuple(self.state.alive_players),
            valid_candidates=valid_candidates,
            runoff=runoff,
        )
        return self._select_target(
            ActionKind.VOTE, actor, candidates, prompt_ctx, forbid_self=True
        )

    def _select_target(
        self,
        action: ActionKind,
        actor: Player,
        candidates: tuple[str, ...],
        prompt_ctx: PromptContext,
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

        for request_number in range(2):
            messages = [
                {"role": "system", "content": prompt_ctx.system_prompt},
                {"role": "user", "content": user_prompts[-1]},
            ]
            try:
                if self.options is not None:
                    response = self.client.chat(
                        actor.model,
                        messages,
                        options=self.options,
                    )
                else:
                    response = self.client.chat(
                        actor.model,
                        messages,
                    )
            except OllamaError as error:
                last_error = error
                break

            raw_responses.append(response)
            target = ActionParser.parse(response, action)
            parsed_targets.append(target)
            if ActionParser.validate_target(
                target, valid_candidates, actor.name if forbid_self else None
            ):
                self.traces.append(
                    ActionTrace(
                        action, actor.name, tuple(raw_responses),
                        tuple(parsed_targets), target, False,
                    )
                )
                return target

            if request_number == 0:
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
        self, action: str, actor: Player, prompt_ctx: PromptContext, fallback: str
    ) -> str:
        messages = prompt_ctx.to_messages()
        try:
            if self.options is not None:
                response = self.client.chat(actor.model, messages, options=self.options)
            else:
                response = self.client.chat(actor.model, messages)
        except OllamaError as error:
            self.traces.append(
                ActionTrace(
                    action,
                    actor.name,
                    (),
                    (),
                    None,
                    True,
                    error.to_dict(),
                )
            )
            return fallback
        self.traces.append(
            ActionTrace(action, actor.name, (response,), (), None, False)
        )
        return response

    @staticmethod
    def _label(action: ActionKind) -> str:
        return {
            ActionKind.ATTACK: "襲撃対象",
            ActionKind.SEER: "占い対象",
            ActionKind.VOTE: "投票",
        }[action]

    def _player(self, name: str) -> Player:
        return next(player for player in self.state.players if player.name == name)
