import random
from dataclasses import dataclass

from .action_parser import ActionKind, ActionParser
from .engine import ActionPolicy
from .models import GameState, Message, Player
from .ollama import OllamaClient, OllamaError


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
    ) -> None:
        self.state = state
        self.client = client
        self._rng = random.Random(state.seed if seed is None else seed)
        self.traces: list[ActionTrace] = []

    def wolf_discuss(
        self, wolf: str, partner: str, history: tuple[Message, ...]
    ) -> str:
        player = self._player(wolf)
        prompt = (
            f"夜の人狼相談です。あなたは人狼で、相方は{partner}です。\n"
            f"生存者: {', '.join(self.state.alive_players)}\n"
            f"これまでの相談: {self._format_history(history)}\n"
            "襲撃対象について相談し、最後に必ず「襲撃対象：プレイヤー名」と出力してください。"
        )
        return self._free_text(
            "wolf_discussion", player, prompt, f"{wolf} は相談に応答できませんでした。"
        )

    def choose_attack(self, wolves: tuple[str, ...], candidates: tuple[str, ...]) -> str:
        actor = self._player(wolves[0])
        valid_candidates = self._valid_candidates(
            actor, candidates, forbidden=wolves
        )
        prompt = self._target_prompt(
            "襲撃対象", actor, valid_candidates,
            f"人狼は {', '.join(wolves)} です。今夜襲撃する対象を選んでください。",
        )
        return self._select_target(
            ActionKind.ATTACK, actor, valid_candidates, prompt, forbidden=wolves
        )

    def choose_seer_target(
        self, seer: str, candidates: tuple[str, ...], results: tuple[tuple[str, str], ...]
    ) -> str:
        actor = self._player(seer)
        past_results = "\n".join(f"{name}: {result}" for name, result in results) or "なし"
        prompt = self._target_prompt(
            "占い対象", actor, candidates,
            f"あなたは占い師です。これまでの占い結果:\n{past_results}\n今夜占う対象を選んでください。",
        )
        return self._select_target(ActionKind.SEER, actor, candidates, prompt)

    def discuss(
        self, player: str, day: int, alive: tuple[str, ...], history: tuple[Message, ...]
    ) -> str:
        actor = self._player(player)
        prompt = (
            f"昼の議論です。あなたの役職は{actor.role.value}、日数は{day}日目です。\n"
            f"生存者: {', '.join(alive)}\n"
            f"公開会話:\n{self._format_history(history)}\n"
            "議論内容だけを発言してください。"
        )
        return self._free_text(
            "discussion", actor, prompt, f"{player} は発言できませんでした。"
        )

    def vote(
        self, voter: str, candidates: tuple[str, ...], day: int, runoff: bool
    ) -> str:
        actor = self._player(voter)
        prompt = self._target_prompt(
            "投票", actor, candidates,
            f"{day}日目の投票です。再投票: {'はい' if runoff else 'いいえ'}。処刑対象を選んでください。",
            forbid_self=True,
        )
        return self._select_target(
            ActionKind.VOTE, actor, candidates, prompt, forbid_self=True
        )

    def _select_target(
        self,
        action: ActionKind,
        actor: Player,
        candidates: tuple[str, ...],
        prompt: str,
        forbid_self: bool = False,
        forbidden: tuple[str, ...] = (),
    ) -> str:
        valid_candidates = self._valid_candidates(
            actor, candidates, forbid_self=forbid_self, forbidden=forbidden
        )
        if not valid_candidates:
            raise ValueError(f"No valid candidates for {action.value} by {actor.name}.")

        prompts = [prompt]
        raw_responses: list[str] = []
        parsed_targets: list[str | None] = []
        last_error: OllamaError | None = None

        for request_number in range(2):
            try:
                response = self.client.chat(
                    actor.model,
                    [{"role": "user", "content": prompts[-1]}],
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
                prompts.append(
                    f"{prompt}\n前回の応答は形式または候補が不正でした。"
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

    def _free_text(
        self, action: str, actor: Player, prompt: str, fallback: str
    ) -> str:
        try:
            response = self.client.chat(actor.model, [{"role": "user", "content": prompt}])
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

    def _target_prompt(
        self,
        label: str,
        actor: Player,
        candidates: tuple[str, ...],
        instruction: str,
        forbid_self: bool = False,
    ) -> str:
        valid_candidates = self._valid_candidates(
            actor, candidates, forbid_self=forbid_self
        )
        return (
            f"{instruction}\nあなたの役職: {actor.role.value}\n"
            f"選択可能な候補: {', '.join(valid_candidates)}\n"
            f"必ず最後に「{label}：プレイヤー名」と出力してください。"
        )

    @staticmethod
    def _label(action: ActionKind) -> str:
        return {
            ActionKind.ATTACK: "襲撃対象",
            ActionKind.SEER: "占い対象",
            ActionKind.VOTE: "投票",
        }[action]

    @staticmethod
    def _format_history(history: tuple[Message, ...]) -> str:
        return "\n".join(f"{message.speaker}: {message.content}" for message in history) or "なし"

    def _player(self, name: str) -> Player:
        return next(player for player in self.state.players if player.name == name)
