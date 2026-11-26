import re
from dataclasses import dataclass

from .models import GameState, Message, Player, PlayerActionMemory
# TASK-065: `Player\u202f6` / `プレイヤー6` などの表記ゆれを内部判定の入口だけで吸収する。
# 公開発言の表記・本文はそのまま。方策はstateの値（既定 normalized / A/B用 literal）に追随し、
# literal では旧来と同じ実在名の完全一致で動く（存在しない番号は未解決のまま観測される）。
from .name_resolution import resolve_known_names


def _mentioned_player_names(state: GameState, content: str) -> set[str]:
    """本文の名前言及を実在名集合へ解決する（内部照合専用。公開面は変更しない）。"""
    names = tuple(player.name for player in state.players)
    policy = getattr(state, "player_name_resolution_policy", None)
    return set(resolve_known_names(content, names, policy=policy or "normalized"))


@dataclass(frozen=True)
class ConsistencyFinding:
    category: str
    severity: str
    summary: str


class SelfConsistencyGuard:
    _HISTORY_DENIAL = re.compile(
        r"(?:記録|台帳|履歴)(?:に|では|が|は|されて)?"
        r"(?:ない|ありません|いない|いません|残っていない|見当たらない)"
        r"|(?i:no record|no records|not recorded|is not recorded|nothing recorded"
        r"|no ledger|not in the ledger|not in the record|not in my record"
        r"|there is no record|there are no records)"
    )
    _ACTION_HISTORY_DENIAL = re.compile(
        r"(?:投票|占い|襲撃|行動|判断)"
        r"(?:(?!理由)[^。！？!?]){0,24}"
        r"(?:記録|台帳|履歴)(?:に|では|が|は|されて)?"
        r"(?:ない|ありません|いない|いません|残っていない|見当たらない)"
        r"|(?i:(?:vote|voted|voting|divination|divined|attack|attacked|action|judgment|decision)"
        r"[^!?]{0,24}(?:no record|not recorded|is not recorded|no ledger"
        r"|not in the ledger|not in the record|there is no record))|"
        r"(?i:(?:no record|not recorded|is not recorded|no ledger|not in the ledger"
        r"|not in the record|there is no record|there are no records)"
        r"[^!?]{0,24}(?:vote|voted|voting|ballot|divination|divined|attack|attacked|action|judgment|decision))"
    )
    _FACT_HISTORY_DENIAL = re.compile(
        r"(?:投票|票|占い|結果|襲撃|死亡|処刑|CO|発言)"
        r"(?:(?!理由)[^。！？!?]){0,24}"
        r"(?:記録|台帳|履歴)(?:に|では|が|は|されて)?"
        r"(?:ない|ありません|いない|いません|残っていない|見当たらない)"
        r"|(?i:(?:vote|votes|voting|ballot|divination|divined|attack|attacked|death|died|execution|exile|claim|speech|result)"
        r"[^!?]{0,24}(?:no record|not recorded|is not recorded|no ledger"
        r"|not in the ledger|not in the record|there is no record))|"
        r"(?i:(?:no record|not recorded|is not recorded|no ledger|not in the ledger"
        r"|not in the record|there is no record|there are no records)"
        r"[^!?]{0,24}(?:vote|votes|voting|ballot|divination|divined|attack|attacked|death|died|execution|exile|claim|speech|result))"
    )
    _CURRENT_ACTION = re.compile(
        r"(?:質問(?:を)?(?:する|します|したい)|聞(?:く|きます|きたい)|"
        r"投票(?:する|します|したい)|占(?:う|います|いたい)|"
        r"襲撃(?:する|します|したい)|"
        r"(?:質問|投票|占い|襲撃)?対象に(?:する|します|したい))"
        r"|(?i:(?:ask|question|vote|divine|attack|target(?:ing)?|choose|exile))"
    )
    _PAST_REFERENCE = re.compile(
        r"(?:昨日|前回|以前|過去|Day\s*\d+|\d+日目|"
        r"投票した|票を入れた|占った|占いました|襲撃した|襲撃しました)"
        r"|(?i:yesterday|previous(?:ly)?|earlier|past day|last (?:day|turn|round|vote)"
        r"|on day\s*\d+|day\s*\d+|voted|cast my vote|divined|attacked)"
    )
    _ACTION_TARGET = re.compile(
        r"(?:投票先|投票対象|処刑対象|占い対象|襲撃対象)"
        r"(?:は|が|を|として)?\s*[：:=：]?\s*"
        r"(?P<target>Player\s+\d+|P\d+)"
        r"|(?i:(?:vote target|vote for|execution target|exile target|divination target"
        r"|attack target|my vote)\s*(?:is|as|:|=)?\s*(?P<target_en>Player\s+\d+|P\d+))"
    )
    _VOTED_FOR = re.compile(
        r"(?P<target>Player\s+\d+|P\d+)(?:さん)?に"
        r"(?:投票(?:した|しました|していた)|票を入れ(?:た|ました))"
        r"|(?i:(?:voted for|cast my vote for|my vote was for)\s*(?P<target_en>Player\s+\d+|P\d+))"
    )
    _PAST_ROLE_ACTION_TARGET = re.compile(
        r"(?:(?P<seer_label>占い対象)(?:は|が|を)?\s*[：:=：]?\s*"
        r"(?P<seer_label_target>Player\s+\d+|P\d+))|"
        r"(?:(?P<attack_label>襲撃対象)(?:は|が|を)?\s*[：:=：]?\s*"
        r"(?P<attack_label_target>Player\s+\d+|P\d+))|"
        r"(?P<seer_narrative_target>Player\s+\d+|P\d+)(?:さん)?"
        r"(?:を|に)占い(?:ました|した|していた)|"
        r"(?P<attack_narrative_target>Player\s+\d+|P\d+)(?:さん)?"
        r"(?:を|に)襲撃(?:した|しました|していた)"
        r"|(?i:(?:divination target|divin(?:ed|ing|e))\s*(?P<seer_label_target_en>Player\s+\d+|P\d+))|"
        r"(?i:(?:attack target|attack(?:ed|ing)?|attacked)\s*(?P<attack_label_target_en>Player\s+\d+|P\d+))"
    )
    _REASON_QUESTION = re.compile(
        r"(?:なぜ|どうして|理由|根拠|変更したのは)"
        r"[^。！？!?]{0,40}(?:投票|占い|襲撃|判断|疑い|変更)|"
        r"(?:投票|占い|襲撃|判断|疑い|変更)"
        r"[^。！？!?]{0,40}(?:なぜ|どうして|理由|根拠)|"
        r"(?i:(?:why|what changed|reason|basis)[^!?]{0,40}(?:vote|divin|attack|judg|suspect|chang))|"
        r"(?i:(?:vote|divin|attack|judg|suspect|chang)[^!?]{0,40}(?:why|reason|basis))"
    )
    _REASON_RESPONSE = re.compile(
        r"(?:理由|根拠|当時|記録されていない|記録がない|分からない|"
        r"わからない|覚えていない|変更した|考え直した)"
        r"|(?i:reason|basis|at that time|not recorded|no record|do not know|don't know"
        r"|changed|reconsidered|initially)"
    )

    _REASON_TERM = re.compile(r"(?:理由|根拠)|(?i:\breason\b|\bbasis\b)")
    _SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])\s*|\n+")

    @classmethod
    def _denial_match(cls, pattern: re.Pattern[str], content: str) -> re.Match | None:
        """理由説明の文（「理由が記録されていない」）は履歴の否定として扱わない。"""
        for sentence in cls._SENTENCE_SPLIT.split(content):
            if not sentence.strip() or cls._REASON_TERM.search(sentence):
                continue
            match = pattern.search(sentence)
            if match is not None:
                return match
        return None

    @classmethod
    def inspect(
        cls,
        content: str | None,
        actor: Player,
        state: GameState,
        previous_message: Message | None = None,
    ) -> tuple[ConsistencyFinding, ...]:
        findings: list[ConsistencyFinding] = []
        if content is None or not isinstance(content, str) or not content.strip():
            return ()
        memories = state.player_action_memories.get(actor.name, [])

        dead_names = {
            player.name for player in state.players if not player.is_alive
        }
        for name in dead_names:
            target_reference = re.search(
                rf"{re.escape(name)}(?:さん)?(?:へ|に)"
                r"(?P<action>[^。！？!?]{0,32})|"
                rf"(?P<action_en>[^.!?]{{0,32}}){re.escape(name)}(?!\d)[^.!?]{{0,32}}",
                content,
            )
            span = (target_reference.group("action") if target_reference else "") or ""
            if (
                target_reference is not None
                and cls._CURRENT_ACTION.search(span)
                and not cls._HISTORY_DENIAL.search(span)
                and not re.search(
                    r"(?i:\bvoted\b|\battacked\b|\bdivined\b|\bwas\b|\bwere\b|\bhad\b)", span
                )
            ):
                findings.append(
                    ConsistencyFinding(
                        "dead_player_action_target",
                        "repair",
                        "You designated a dead player as a current action target.",
                    )
                )
                break

        prior_votes = [
            memory for memory in memories if memory.action_type == "vote"
        ]
        if prior_votes and cls._PAST_REFERENCE.search(content):
            latest_vote = prior_votes[-1]
            for pattern in (cls._ACTION_TARGET, cls._VOTED_FOR):
                match = pattern.search(content)
                if match is None:
                    continue
                mentioned_target = match.group("target") or match.group("target_en")
                if (
                    latest_vote.vote_target
                    and mentioned_target != latest_vote.vote_target
                    and (
                        "投票" in match.group(0)
                        or "票を入れた" in match.group(0)
                        or "vote" in match.group(0).lower()
                        or "ballot" in match.group(0).lower()
                    )
                ):
                    findings.append(
                        ConsistencyFinding(
                            "contradictory_past_vote",
                            "repair",
                            "Your past vote target differs from the recorded action history.",
                        )
                    )
                    break

        if cls._PAST_REFERENCE.search(content):
            for match in cls._PAST_ROLE_ACTION_TARGET.finditer(content):
                action_type = (
                    "seer"
                    if match.group("seer_label")
                    or match.group("seer_narrative_target")
                    or match.group("seer_label_target_en")
                    else "attack"
                )
                target = (
                    match.group("seer_label_target")
                    or match.group("attack_label_target")
                    or match.group("seer_narrative_target")
                    or match.group("attack_narrative_target")
                    or match.group("seer_label_target_en")
                    or match.group("attack_label_target_en")
                )
                previous_action = next(
                    (
                        memory
                        for memory in reversed(memories)
                        if memory.action_type == action_type
                        and memory.selected_target
                    ),
                    None,
                )
                if (
                    previous_action is not None
                    and target != previous_action.selected_target
                ):
                    findings.append(
                        ConsistencyFinding(
                            "contradictory_past_action",
                            "repair",
                            "Your past role action differs from the recorded action history.",
                        )
                    )
                    break

        if cls._HISTORY_DENIAL.search(content) and cls._denial_match(
            cls._ACTION_HISTORY_DENIAL, content
        ):
            memory_targets = {
                memory.vote_target
                for memory in memories
                if memory.action_type == "vote" and memory.vote_target
            }
            refers_to_another_player = bool(
                _mentioned_player_names(state, content) - {actor.name}
            )
            if memory_targets and re.search(
                r"(?:投票|票|自分の行動|自分の判断)"
                r"|(?i:\bmy\b[^.!?\n]{0,16}\b(vote|voted|action|judgment|decision)\b"
                r"|\bi\s+(?:voted|voted for)\b)",
                content,
            ) and not refers_to_another_player:
                findings.append(
                    ConsistencyFinding(
                        "recorded_action_denied",
                        "repair",
                        "You denied your own action that is recorded in the action ledger.",
                    )
                )

        if cls._HISTORY_DENIAL.search(content) and cls._denial_match(
            cls._FACT_HISTORY_DENIAL, content
        ):
            mentioned_players = _mentioned_player_names(state, content)
            fact_topics = {
                "vote": re.search(
                    r"(?:投票|票)|(?i:\bvote\b|votes|voting|ballot)", content
                )
                is not None,
                "seer_result": re.search(
                    r"(?:占い|結果)|(?:divinat|divination result)", content, re.IGNORECASE
                )
                is not None,
                "attack_death": re.search(
                    r"(?:襲撃|死亡)|(?:attack|killed|died|death)", content, re.IGNORECASE
                )
                is not None,
                "execution": re.search(
                    r"(?:処刑|追放)|(?:executed|exiled|voted out)", content, re.IGNORECASE
                )
                is not None,
            }
            fact_terms = {
                "vote": "voted",
                "seer_result": None,
                "attack_death": "attacked",
                "execution": "voted out",
            }
            contradicts_fact = any(
                fact.visibility == "public"
                and fact_topics.get(fact.kind, False)
                and (
                    not mentioned_players
                    or mentioned_players.issubset(
                        {
                            player.name
                            for player in state.players
                            if player.name in fact.description
                        }
                    )
                )
                and (
                    fact_terms.get(fact.kind) is None
                    or fact_terms[fact.kind] in fact.description.lower()
                )
                for fact in state.fact_ledger.confirmed_facts
            )
            if contradicts_fact:
                findings.append(
                    ConsistencyFinding(
                        "public_fact_denied",
                        "repair",
                        "You claimed a fact present in the public FactLedger is unrecorded.",
                    )
                )

        if (
            previous_message is not None
            and previous_message.speaker != actor.name
            and cls._REASON_QUESTION.search(previous_message.content)
            and not cls._REASON_RESPONSE.search(content)
        ):
            findings.append(
                ConsistencyFinding(
                    "past_action_question_unanswered",
                    "warning",
                    "You did not directly answer the question asking for your past action reason.",
                )
            )

        return tuple(findings)

    @staticmethod
    def serialize(memory: PlayerActionMemory) -> dict[str, object]:
        return {
            "action_id": memory.action_id,
            "day": memory.day,
            "phase": memory.phase,
            "action_type": memory.action_type,
            "selected_target": memory.selected_target,
            "vote_target": memory.vote_target,
            "public_statement_summary": memory.public_statement_summary,
            "suspicion_targets": list(memory.suspicion_targets),
            "reasoning_summary": memory.reasoning_summary,
            "confidence": memory.confidence,
            "unanswered_questions": list(memory.unanswered_questions),
            "evidence_ids": list(memory.evidence_ids),
            "previous_position": memory.previous_position,
            "position_change_reason": memory.position_change_reason,
        }
