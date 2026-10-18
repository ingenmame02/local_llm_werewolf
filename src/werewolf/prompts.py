from dataclasses import dataclass
from typing import Mapping

from .models import GameState, Message, Player, Role

ROLE_NAMES_JA: dict[Role, str] = {
    Role.WOLF: "人狼",
    Role.MADMAN: "狂人",
    Role.SEER: "占い師",
    Role.VILLAGER: "村人",
}

ROLE_DESCRIPTIONS_JA: dict[Role, str] = {
    Role.WOLF: (
        "あなたは人狼です。夜に村人を襲撃して減らしていくのが目的です。\n"
        "相方の人狼と協力し、昼は村人のふりをして疑われないように立ち回ってください。\n"
        "最終的に人狼陣営の数が村人陣営以上になれば勝利です。"
    ),
    Role.MADMAN: (
        "あなたは狂人です。人狼陣営の仲間ですが、特殊能力はありません。\n"
        "昼は村人のふりをして、人狼が有利になるように村を混乱させたり、誤った方向に議論を誘導したりしてください。\n"
        "自分が処刑されても、人狼陣営が勝てばあなたの勝ちです。"
    ),
    Role.SEER: (
        "あなたは占い師です。夜に1人を占って「人狼」か「人間」かを知ることができます。\n"
        "得た情報をうまく使って人狼を見つけ、村を勝利に導くのが目的です。\n"
        "ただし、自分が占い師だとバレすぎると襲撃されるリスクもあるので、情報の出し方には注意してください。"
    ),
    Role.VILLAGER: (
        "あなたは村人です。特別な能力はありません。\n"
        "昼の議論と投票で人狼を見つけ出し、処刑していくのが目的です。\n"
        "怪しい言動をしている人を注意深く観察してください。"
    ),
}

COMMON_SYSTEM_TEMPLATE = """人狼ゲームのプレイヤーとして行動してください。
あなたの役職は「{role}」です。
{role_description}
ユーザーメッセージのルールカードと確定事実に従い、指定された形式だけを簡潔に返してください。"""

RULE_CARD = """【ルール】
8人：人狼2・狂人1・占い師1・村人4。夜は相談/襲撃/占い、朝は死亡発表、昼議論3巡後に投票。
狂人は襲撃不可。占い結果は人狼/人間。自分への投票不可。同票は再投票1回後に抽選。
昼の「パス」は有効な発言スキップで、空応答とは異なる。

"""


@dataclass(frozen=True)
class PromptContext:
    system_prompt: str
    user_prompt: str

    def to_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]

    def compact(self, recent_lines: int = 4) -> PromptContext:
        start_marker = (
            "【これまでの公開会話】"
            if "【これまでの公開会話】" in self.user_prompt
            else "【公開会話（発言は未確認情報）】"
            if "【公開会話（発言は未確認情報）】" in self.user_prompt
            else "【これまでの相談内容（提案・意見であり、確定事実ではありません）】"
        )
        return PromptContext(
            system_prompt=self.system_prompt,
            user_prompt=_compact_history_section(
                self.user_prompt,
                start_marker,
                recent_lines,
            ),
        )


def _compact_history_section(
    prompt: str, start_marker: str, recent_lines: int
) -> str:
    start = prompt.find(start_marker)
    end_marker = (
        "【占い師本人だけが参照できる結果】"
        if "【占い師本人だけが参照できる結果】" in prompt
        else "【指示】"
    )
    end = prompt.find(end_marker, start + len(start_marker))
    if start < 0 or end < 0:
        return prompt
    section = prompt[start:end]
    lines = [line for line in section.splitlines() if line.strip()]
    content_start = 2 if len(lines) > 1 and lines[1].startswith("以下は") else 1
    history = lines[content_start:]
    if len(history) <= recent_lines:
        return prompt
    older_history = history[:-recent_lines]
    recent_history = history[-recent_lines:]
    summary_lines = [
        line if len(line) <= 120 else line[:117] + "..."
        for line in older_history[-3:]
        if line.strip()
    ]
    omitted = len(older_history) - len(summary_lines)
    compressed_section = "\n".join(
        [
            *lines[:content_start],
            f"（古い会話を圧縮。{omitted} 行省略、以下は短縮した未確認発言）",
            *summary_lines,
            "（直近の発言）",
            *recent_history,
            "",
        ]
    )
    return prompt[:start] + compressed_section + prompt[end:]


class PromptBuilder:
    @staticmethod
    def _fact_header(
        actor: Player,
        alive_players: tuple[str, ...],
        day: int,
        phase: str,
        state: GameState | None = None,
        conversation_history: tuple[Message, ...] | None = None,
    ) -> str:
        dead_players = (
            tuple(player.name for player in state.players if not player.is_alive)
            if state is not None
            else ()
        )
        latest_attack = (
            f"Day {state.attack_deaths[-1][0]}: {state.attack_deaths[-1][1]}"
            if state is not None and state.attack_deaths
            else "なし"
        )
        latest_execution = (
            f"Day {state.executions[-1][0]}: {state.executions[-1][1]}"
            if state is not None and state.executions
            else "なし"
        )
        latest_vote = (
            f"Day {state.vote_history[-1][0]}"
            f"{'（決選再投票）' if state.vote_history[-1][1] else '（通常投票）'}: "
            + "、".join(
                f"{voter} → {target}"
                for voter, target in state.vote_history[-1][2].items()
            )
            if state is not None and state.vote_history
            else "なし"
        )
        current_turn = (
            f"{state.discussion_turn}/3"
            if state is not None and phase.startswith("昼")
            else "該当なし"
        )
        source_history = (
            conversation_history
            if conversation_history is not None
            else tuple(state.conversation_log) if state is not None else ()
        )
        previous_message = next(
            (message for message in reversed(source_history) if message.speaker != "system"),
            None,
        )
        return (
            RULE_CARD
            + "【ゲームシステムが確定した事実】\n"
            f"- あなたのプレイヤー名：{actor.name}\n"
            f"- あなたの役職：{ROLE_NAMES_JA[actor.role]}\n"
            f"- 現在の日数：{day}日目\n"
            f"- 現在のフェーズ：{phase}\n"
            f"- 昼議論ターン：{current_turn}\n"
            f"- 現在の発言者：{actor.name}\n"
            f"- 直前の発言者：{previous_message.speaker if previous_message else 'なし'}\n"
            f"- 直前の発言内容：{previous_message.content if previous_message else 'なし'}\n"
            f"- 現在の生存者：{', '.join(alive_players) or 'なし'}\n"
            f"- 既知の死亡者：{', '.join(dead_players) or 'なし'}\n"
            f"- 最新の襲撃死亡者：{latest_attack}\n"
            f"- 最新の処刑者：{latest_execution}\n"
            f"- 最新の投票結果：{latest_vote}\n"
            "\n"
            "【情報の区分】\n"
            "この欄のゲームシステム記録だけが確定事実です。プレイヤーの発言・主張・推測は未確認情報であり、"
            "確定事実として扱わないでください。\n\n"
        )

    @staticmethod
    def build_system_prompt(player: Player) -> str:
        role_ja = ROLE_NAMES_JA[player.role]
        role_desc = ROLE_DESCRIPTIONS_JA[player.role]
        return COMMON_SYSTEM_TEMPLATE.format(
            role=role_ja,
            role_description=role_desc,
        )

    @classmethod
    def build_wolf_discuss_prompt(
        cls,
        actor: Player,
        partner_name: str,
        alive_players: tuple[str, ...],
        wolf_chat_history: tuple[Message, ...],
        day: int = 1,
        state: GameState | None = None,
    ) -> PromptContext:
        if actor.role is not Role.WOLF:
            raise PermissionError(f"Player {actor.name} ({actor.role}) is not a wolf.")
        if not actor.is_alive:
            raise ValueError(f"Dead player {actor.name} cannot act.")

        formatted_history = (
            "\n".join(f"{m.speaker}: {m.content}" for m in wolf_chat_history)
            if wolf_chat_history
            else "なし"
        )
        user_prompt = (
            cls._fact_header(actor, alive_players, day, "夜（人狼同士の相談）", state)
            + f"相方の人狼：{partner_name}\n\n"
            "【これまでの相談内容（提案・意見であり、確定事実ではありません）】\n"
            f"{formatted_history}\n\n"
            "【指示】\n"
            "相方と相談して、今夜襲撃する相手を決めたいです。\n"
            "- 誰を襲撃したいか、その理由を話してください\n"
            "- 相方の意見も踏まえて調整して構いません\n\n"
            "最後に必ず以下の形式で襲撃対象を明示してください：\n"
            "襲撃対象：プレイヤー名"
        )
        return PromptContext(
            system_prompt=cls.build_system_prompt(actor),
            user_prompt=user_prompt,
        )

    @classmethod
    def build_attack_prompt(
        cls,
        actor: Player,
        wolves: tuple[str, ...],
        valid_candidates: tuple[str, ...],
        day: int = 1,
        state: GameState | None = None,
    ) -> PromptContext:
        if actor.role is not Role.WOLF:
            raise PermissionError(f"Player {actor.name} ({actor.role}) is not a wolf.")
        if not actor.is_alive:
            raise ValueError(f"Dead player {actor.name} cannot act.")

        user_prompt = (
            cls._fact_header(
                actor,
                tuple(state.alive_players) if state is not None else (*wolves, *valid_candidates),
                day,
                "夜（襲撃）",
                state,
            )
            + "【現在の状況】\n"
            f"- 人狼一覧：{', '.join(wolves)}\n"
            f"- 選択可能な襲撃対象：{', '.join(valid_candidates)}\n\n"
            "【指示】\n"
            "今夜襲撃する対象を1人選んでください。\n"
            "最後に必ず以下の形式で襲撃対象を明示してください：\n"
            "襲撃対象：プレイヤー名"
        )
        return PromptContext(
            system_prompt=cls.build_system_prompt(actor),
            user_prompt=user_prompt,
        )

    @classmethod
    def build_seer_prompt(
        cls,
        actor: Player,
        alive_players: tuple[str, ...],
        seer_results: Mapping[str, str] | tuple[tuple[str, str], ...],
        day: int = 1,
        state: GameState | None = None,
    ) -> PromptContext:
        if actor.role is not Role.SEER:
            raise PermissionError(f"Player {actor.name} ({actor.role}) is not a seer.")
        if not actor.is_alive:
            raise ValueError(f"Dead player {actor.name} cannot act.")

        if isinstance(seer_results, Mapping):
            results_list = [f"{k}: {v}" for k, v in seer_results.items()]
        else:
            results_list = [f"{k}: {v}" for k, v in seer_results]
        today_results = (
            [
                f"{target}: {'人狼' if result == 'wolf' else '人間'}"
                for result_day, target, result in state.seer_result_history
                if result_day == day
            ]
            if state is not None
            else []
        )
        if state is not None and not state.seer_result_history and state.seer_results:
            today_results = [
                f"{target}: {'人狼' if result == 'wolf' else '人間'}"
                for target, result in state.seer_results.items()
            ]
        past_results = (
            [
                f"Day {result_day}: {target} = "
                f"{'人狼' if result == 'wolf' else '人間'}"
                for result_day, target, result in state.seer_result_history
                if result_day < day
            ]
            if state is not None and state.seer_result_history
            else results_list
        )
        today_result_text = "\n".join(today_results) or "なし"
        past_result_text = "\n".join(past_results) or "なし"

        user_prompt = (
            cls._fact_header(actor, alive_players, day, "夜（占い）", state)
            + "【占い履歴】\n"
            f"- 今回（当日）の占い結果：\n{today_result_text}\n"
            f"- 過去（当日より前）の占い結果：\n{past_result_text}\n\n"
            f"選択可能な占い対象：{', '.join(alive_players)}\n"
            "【指示】\n"
            "今夜占う相手を1人選んでください。\n"
            "- なぜその人を占うのか、簡単に理由を述べても構いません\n\n"
            "最後に必ず以下の形式で占い対象を明示してください：\n"
            "占い対象：プレイヤー名"
        )
        return PromptContext(
            system_prompt=cls.build_system_prompt(actor),
            user_prompt=user_prompt,
        )

    @classmethod
    def build_discussion_prompt(
        cls,
        actor: Player,
        alive_players: tuple[str, ...],
        conversation_history: tuple[Message, ...],
        day: int = 1,
        state: GameState | None = None,
    ) -> PromptContext:
        if not actor.is_alive:
            raise ValueError(f"Dead player {actor.name} cannot act.")

        player_history = tuple(
            message for message in conversation_history if message.speaker != "system"
        )
        formatted_history = (
            "\n".join(f"{m.speaker}: {m.content}" for m in player_history[:-1])
            if len(player_history) > 1
            else "なし"
        )
        previous_message = player_history[-1] if player_history else None
        directly_addressed = cls._is_directed_to_player(
            actor, previous_message, player_history
        )
        own_seer_results = (
            state.seer_results
            if actor.role is Role.SEER and state is not None
            else {}
        )
        current_seer_results = (
            [
                f"{target}: {'人狼' if result == 'wolf' else '人間'}"
                for result_day, target, result in state.seer_result_history
                if result_day == day
            ]
            if state is not None
            else []
        )
        if state is not None and not state.seer_result_history and own_seer_results:
            current_seer_results = [
                f"{target}: {'人狼' if result == 'wolf' else '人間'}"
                for target, result in own_seer_results.items()
            ]
        role_ja = ROLE_NAMES_JA[actor.role]
        user_prompt = (
            cls._fact_header(
                actor,
                alive_players,
                day,
                "昼（議論）",
                state,
                conversation_history=player_history,
            )
            + f"- あなたへの質問・反応の可能性：{'あり' if directly_addressed else '明確な手がかりなし'}\n\n"
            + "【これまでの公開会話】\n"
            "以下はプレイヤーの発言であり、確定事実ではありません。\n"
            f"{formatted_history}\n\n"
            + (
                "【占い師本人だけが参照できる結果】\n"
                + "今回（当日）：\n"
                + (
                    "\n".join(current_seer_results) or "なし"
                )
                + "\n過去（当日より前）：\n"
                + (
                    "\n".join(
                        f"Day {event_day}: {target} = "
                        f"{'人狼' if result == 'wolf' else '人間'}"
                        for event_day, target, result in state.seer_result_history
                        if event_day < day
                    )
                    or "なし"
                )
                + "\n\n"
                if actor.role is Role.SEER and state is not None
                else ""
            )
            + "【指示】\n"
            "上記を踏まえて、今の状況についてあなたの考えを述べてください。\n"
            "- 直前の発言があなたの名前や「あなた」「君」に言及している、質問している、またはあなたの発言に明らかに反応している場合、まず相手へ直接答えてください\n"
            "- 答えた後に推理・反論・質問を続けても構いません\n"
            "- 答えられない場合は、無言にせず「パス」とだけ出力できます\n"
            "- 疑っている人や、怪しいと思う点があれば話してください\n"
            "- 自分の意見をはっきり言う必要はありません（様子を見る発言でも構いません）\n"
            "- 普段のあなたらしい話し方でお願いします\n"
            "- パスする場合は、前後に説明や句読点を付けず「パス」とだけ出力してください\n"
            + (
                "- 占い結果を持つ占い師は、今日まだ結果を活用した発言をしていない場合、今回の発言で必ず結果を推理に反映してください\n"
                "- 結果を公開するか伏せるかは選べますが、対象への疑い・信頼・警戒など、結果を踏まえた判断を具体的に述べてください\n"
                "- 今日の最初の情報活用発言をするまでは、パスだけを出力してはいけません\n"
                "- 今回の結果がある場合はそれを最優先し、今回の結果がない場合は最新の過去結果を使ってください\n"
                if actor.role is Role.SEER and own_seer_results
                else ""
            )
            + "\n発言または「パス」のみを出力してください。"
        )
        return PromptContext(
            system_prompt=cls.build_system_prompt(actor),
            user_prompt=user_prompt,
        ).compact(recent_lines=6)

    @staticmethod
    def _is_directed_to_player(
        actor: Player,
        previous_message: Message | None,
        player_history: tuple[Message, ...],
    ) -> bool:
        if previous_message is None or previous_message.speaker == actor.name:
            return False

        content = previous_message.content
        address_cues = (
            actor.name,
            "あなた",
            "貴方",
            "君",
            "きみ",
            "お前",
        )
        reaction_cues = (
            "さっきの発言",
            "先ほどの発言",
            "あなたの発言",
            "君の発言",
            "言っていた",
            "言ってた",
            "主張",
            "意見",
        )
        addressed = any(cue in content for cue in address_cues)
        has_own_prior_statement = any(
            message.speaker == actor.name for message in player_history[:-1]
        )
        reacts_to_own_statement = has_own_prior_statement and any(
            cue in content for cue in reaction_cues
        )
        return addressed or reacts_to_own_statement

    @classmethod
    def build_vote_prompt(
        cls,
        actor: Player,
        alive_players: tuple[str, ...],
        valid_candidates: tuple[str, ...],
        today_summary: str = "",
        runoff: bool = False,
        day: int = 1,
        state: GameState | None = None,
    ) -> PromptContext:
        if not actor.is_alive:
            raise ValueError(f"Dead player {actor.name} cannot act.")


        summary_text = today_summary if today_summary else "（議論ログを参照してください）"
        user_prompt = (
            cls._fact_header(
                actor,
                alive_players,
                day,
                f"投票{'（決戦再投票）' if runoff else ''}",
                state,
            )
            + f"- 選択可能な候補: {', '.join(valid_candidates)}\n"
            "【本日の議論まとめ（未確認の主張・推測を含みます）】\n"
            f"{summary_text}\n\n"
            "【指示】\n"
            "処刑する相手を1人選んでください。\n"
            "- 自分自身には投票できません\n"
            "- 簡単な理由を述べても構いません\n\n"
            "最後に必ず以下の形式で投票先を明示してください：\n"
            "投票：プレイヤー名"
        )
        return PromptContext(
            system_prompt=cls.build_system_prompt(actor),
            user_prompt=user_prompt,
        )
