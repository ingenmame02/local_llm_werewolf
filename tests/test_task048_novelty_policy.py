"""TASK-048の回帰テスト（LL不使用）: 議論の新規性を検索・計測で扱う。

実機ログでは `ledger_repair` 15件の多くが `repeated_discussion_topic` で、本文生成後の
修復（重複文の除去→意味的再要求）が1ターン最大180秒級の遅延と本文損失を作っていた。
本テストは以下を固定する（実機Ollamaへは一切送信しない）。

1. 既出論点台帳は同一論点の連呼と、新根拠を伴う正当な繰り返し（投票先の再確認など）を区別する。
2. 出力後の本文は 新規証拠 / 反証 / 立場変更 / 質問への応答（＋新論点）で計測される。
3. 低新規性でも本文をそのまま公開し、再生成・文の除去・ledger_repairを作らない（publish）。
4. append は既存本文を保持し、短い追記だけ追加する（全文再生成しない）。
5. legacy_repair は旧挙動（重複文の除去→台帳修復）を保持する（A/B比較用）。
6. 既出論点の要約は観測情報として1回だけ渡す（必ず・禁止などの強制指示を出さない）。
"""
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.discussion_format import extract_discussion_body
from werewolf.discussion_novelty import (
    NOVELTY_NEW_EVIDENCE,
    NOVELTY_NEW_TOPIC,
    NOVELTY_NO_SIGNAL,
    NOVELTY_POLICY_APPEND,
    NOVELTY_POLICY_LEGACY_REPAIR,
    NOVELTY_POLICY_PUBLISH,
    NOVELTY_POSITION_CHANGE,
    NOVELTY_QUESTION_ANSWER,
    NOVELTY_REFUTATION,
    NOVELTY_RESTATEMENT,
    DiscussionTopicLedger,
    classify_novelty,
    normalize_novelty_policy,
)
from werewolf.hypothesis_ledger import STANCE_SUSPECT
from werewolf.llm_actions import OllamaActions
from werewolf.models import Message
from werewolf.ollama import ChatResponse
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game

PLAYER_NAMES = tuple(f"Player {index}" for index in range(1, 6))


class ScriptedClient:
    """脚本化された応答を返すOllamaクライアントの代わり（リクエスト数を数える・LL不使用）。"""

    timeout = 60.0
    max_retries = 1
    timeout_retries = 1

    def __init__(self, responses: list[ChatResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.requests = 0

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {"ollama_ps_context": None}

    def _next(self) -> ChatResponse:
        self.requests += 1
        if self.responses:
            return self.responses.pop(0)
        return ChatResponse(content="", attempts=1)

    def chat(self, model, messages, options=None) -> str:
        return self._next().content

    def chat_with_metadata(self, model, messages, options=None, tools=None) -> ChatResponse:
        return self._next()


def _state(response_language: str = "日本語"):
    return create_game(
        seed=1001, config_path="config/players.json", response_language=response_language
    )


def _player(state, name: str):
    return next(player for player in state.players if player.name == name)


def _alive(state) -> tuple[str, ...]:
    return tuple(player.name for player in state.players if player.is_alive)


def _speaker_without_seer(state, exclude: str) -> str:
    from werewolf.models import Role

    return next(
        player.name
        for player in state.players
        if player.is_alive and player.name != exclude and player.role is not Role.SEER
    )


REPEATED_BODY = (
    "[CO] None\n[BODY] Player 3さんの今日の投票の理由を確認したい。"
    "私は公開情報から疑いを整理し、次の質問先を決めたい。"
)



def _day1_with_prior_speech(state) -> tuple[Message, ...]:
    """Day 1 の議論で同一本文が既出の状態を作る（台帳にも論点を登録する）。"""
    state.conversation_log.append(Message(speaker="system", content="Day 1 begins."))
    state.conversation_log.append(Message(speaker="Player 2", content=REPEATED_BODY))
    state.record_topics_from_speech(
        "Player 2", extract_discussion_body(REPEATED_BODY) or REPEATED_BODY, 1, 1
    )
    return tuple(state.conversation_log)



def test_topic_ledger_separates_restatement_from_legitimate_repetition() -> None:
    ledger = DiscussionTopicLedger()
    ledger.record_speech(
        speaker="Player 2",
        content="Player 3さんへの投票を提案する。",
        day=1,
        turn=1,
        speech_id="[speech:1]",
        player_names=PLAYER_NAMES,
    )
    # 新シグナルのない再掲は連呼（restatement）として数え、レコードは増やさない
    ledger.record_speech(
        speaker="Player 4",
        content="Player 3さんへの投票を提案する。",
        day=1,
        turn=2,
        speech_id="[speech:2]",
        player_names=PLAYER_NAMES,
    )
    assert ledger.restatements == 1
    assert ledger.legitimate_repetitions == 0
    assert len(ledger.records) == 1
    record = next(iter(ledger.records.values()))
    assert record.seen == 2
    assert record.speakers == ["Player 2", "Player 4"]
    # 新根拠を伴う再確認（投票先の再確認）は正当な繰り返しとして区別する
    ledger.record_speech(
        speaker="Player 4",
        content="私はPlayer 3さんへの投票を提案する。理由はDay 1の投票パターンです。",
        day=1,
        turn=3,
        speech_id="[speech:3]",
        player_names=PLAYER_NAMES,
    )
    assert ledger.legitimate_repetitions == 1
    assert ledger.restatements == 1
    # 投票先（Player 3）の論点は1件に統合される（新根拠の文は別の論点として立つ）
    vote = ledger.records["T-vote_proposal-Player 3"]
    assert vote.seen == 3
    assert len(ledger.records) == 2


def test_classify_novelty_measures_the_four_classes() -> None:
    ledger = DiscussionTopicLedger()
    ledger.record_speech(
        speaker="Player 2",
        content="Player 3さんへの投票を提案する。",
        day=1,
        turn=1,
        speech_id="[speech:1]",
        player_names=PLAYER_NAMES,
    )
    prior = ledger.topics_for_day(1)
    # 同一論点の連呼（新シグナルなし）は restatement（新規性なし）
    restated = classify_novelty(
        "Player 3さんへの投票を提案する。",
        actor_name="Player 4",
        player_names=PLAYER_NAMES,
        prior_topics=prior,
    )
    assert restated.primary == NOVELTY_RESTATEMENT
    assert restated.has_novelty is False
    assert restated.novelty_score == 0.0
    assert restated.restated_topic_ids
    # 新根拠を伴う再確認は 新規証拠 ＋ 正当な繰り返し
    evidenced = classify_novelty(
        "Player 3さんへの投票を提案する。理由はDay 1の投票パターンです。",
        actor_name="Player 4",
        player_names=PLAYER_NAMES,
        prior_topics=prior,
    )
    assert NOVELTY_NEW_EVIDENCE in evidenced.classes
    assert evidenced.legitimate_repetition_ids
    # 反証
    refuted = classify_novelty(
        "Player 3さんのCOは二重なので矛盾だ。私は別の可能性を確認したい。",
        actor_name="Player 4",
        player_names=PLAYER_NAMES,
        prior_topics=prior,
    )
    assert NOVELTY_REFUTATION in refuted.classes
    # 立場変更（本人の以前の立場と逆の立場）
    changed = classify_novelty(
        "Player 3さんは信頼できる。",
        actor_name="Player 4",
        player_names=PLAYER_NAMES,
        prior_topics=prior,
        own_prior_stances={"Player 3": STANCE_SUSPECT},
    )
    assert changed.primary == NOVELTY_POSITION_CHANGE
    # 質問への応答
    answered = classify_novelty(
        "回答: Player 3さんは占い師のCOを出しました。",
        actor_name="Player 4",
        player_names=PLAYER_NAMES,
        prior_topics=prior,
        question_answered=True,
    )
    assert answered.primary == NOVELTY_QUESTION_ANSWER
    # 論点を持たない本文は no_signal（誘導ではなく計測だけ）
    silent = classify_novelty(
        "まだ情報が足りません。",
        actor_name="Player 4",
        player_names=PLAYER_NAMES,
        prior_topics=prior,
    )
    assert silent.primary == NOVELTY_NO_SIGNAL
    assert silent.topic_sentence_count == 0


def test_normalize_novelty_policy_aliases() -> None:
    assert normalize_novelty_policy(None) == NOVELTY_POLICY_PUBLISH
    assert normalize_novelty_policy("APPEND") == NOVELTY_POLICY_APPEND
    assert normalize_novelty_policy("legacy") == NOVELTY_POLICY_LEGACY_REPAIR
    assert normalize_novelty_policy("unexpected") == NOVELTY_POLICY_PUBLISH


def test_low_novelty_body_is_published_without_regeneration() -> None:
    state = _state()
    history = _day1_with_prior_speech(state)
    actor_name = _speaker_without_seer(state, "Player 2")
    client = ScriptedClient([ChatResponse(content=REPEATED_BODY, attempts=1)])
    events: list[dict[str, object]] = []
    actions = OllamaActions(
        state,
        client,
        seed=1001,
        options={"question_hint_mode": "none"},
        event_callback=events.append,
    )
    assert actions.novelty_policy == NOVELTY_POLICY_PUBLISH
    body = actions.discuss(actor_name, 1, _alive(state), history)
    # 本文はそのまま公開される（文の除去も全文再生成も作らない）
    assert body == REPEATED_BODY
    assert client.requests == 1
    assert not [event for event in events if event["type"] == "ledger_repair"]
    kept = [event for event in events if event["type"] == "llm_discussion_low_novelty_kept"]
    assert kept and kept[0]["regeneration_avoided"] is True
    assert kept[0]["reason"] == "repeated_discussion_topic"
    measured = [event for event in events if event["type"] == "discussion_novelty_measured"]
    assert measured and measured[0]["novelty_primary"] == NOVELTY_RESTATEMENT
    assert measured[0]["novelty_restated_topics"] >= 1
    assert actions.traces[-1].fallback_used is False
    assert actions.traces[-1].content_preserved is True


def test_append_policy_keeps_body_and_adds_only_a_short_addition() -> None:
    state = _state()
    history = _day1_with_prior_speech(state)
    actor_name = _speaker_without_seer(state, "Player 2")
    addition = "[BODY] Player 5さんの今日の投票先を新たに質問したい。"
    client = ScriptedClient(
        [
            ChatResponse(content=REPEATED_BODY, attempts=1),
            ChatResponse(content=addition, attempts=1),
        ]
    )
    events: list[dict[str, object]] = []
    actions = OllamaActions(
        state,
        client,
        seed=1001,
        options={"question_hint_mode": "none", "novelty_policy": "append"},
        event_callback=events.append,
    )
    assert actions.novelty_policy == NOVELTY_POLICY_APPEND
    body = actions.discuss(actor_name, 1, _alive(state), history)
    # 既存本文を保持し、未論点への追記だけを足す（全文再生成ではない）
    assert "Player 3さんの今日の投票の理由" in body
    assert "Player 5さんの今日の投票先" in body
    assert client.requests == 2
    appended = [event for event in events if event["type"] == "discussion_novelty_appended"]
    assert appended and appended[0]["body_preserved"] is True
    assert not [event for event in events if event["type"] == "ledger_repair"]


def test_legacy_repair_policy_still_removes_repeated_sentences() -> None:
    state = _state()
    history = _day1_with_prior_speech(state)
    actor_name = _speaker_without_seer(state, "Player 2")
    mixed = (
        "[CO] None\n[BODY] Player 3さんの今日の投票の理由を確認したい。"
        "Player 5さんの今日の投票先を質問したい。"
    )
    client = ScriptedClient([ChatResponse(content=mixed, attempts=1)])
    events: list[dict[str, object]] = []
    actions = OllamaActions(
        state,
        client,
        seed=1001,
        options={"question_hint_mode": "none", "novelty_policy": "legacy_repair"},
        event_callback=events.append,
    )
    assert actions.novelty_policy == NOVELTY_POLICY_LEGACY_REPAIR
    body = actions.discuss(actor_name, 1, _alive(state), history)
    repairs = [event for event in events if event["type"] == "ledger_repair"]
    assert repairs and repairs[0]["reason"] == "repeated_discussion_topic"
    assert "Player 3さんの今日の投票の理由" not in body
    assert "Player 5さんの今日の投票先" in body


def test_previous_topics_section_is_observation_only_and_once() -> None:
    state = _state()
    history = _day1_with_prior_speech(state)
    state.conversation_log.append(Message(speaker="Player 4", content=REPEATED_BODY))
    state.record_topics_from_speech(
        "Player 4", extract_discussion_body(REPEATED_BODY) or REPEATED_BODY, 1, 2
    )
    block, diagnostics = state.topic_summary_for("Player 5", 1)
    # 連呼は1行に統合される（本文の再掲・コンテキストの重複増を避ける）
    assert "raised 2x" in block
    assert "[speech:" in block
    assert int(diagnostics["topic_summary_lines"]) <= 6
    # 観測のみ：論点・口調・結論の強制指示を出さない
    for forbidden in ("必ず", "must", "禁止", "do not repeat", "回答しなければならない"):
        assert forbidden not in block
    actor = _player(state, "Player 5")
    alive = _alive(state)
    ctx = PromptBuilder.build_discussion_prompt(
        actor,
        alive,
        tuple(state.conversation_log),
        day=1,
        state=state,
        include_claim=True,
        context_strategy="hybrid_retrieval",
        question_hint_mode="soft",
    )
    assert ctx.user_prompt.count("[PREVIOUS_TOPICS]") == 1
    assert int(ctx.diagnostics["topic_summary_chars"]) > 0
    assert int(ctx.diagnostics["topic_records_today"]) == 2
    off_ctx = PromptBuilder.build_discussion_prompt(
        actor,
        alive,
        tuple(state.conversation_log),
        day=1,
        state=state,
        include_claim=True,
        context_strategy="hybrid_retrieval",
        question_hint_mode="soft",
        include_topic_summary=False,
    )
    assert "[PREVIOUS_TOPICS]" not in off_ctx.user_prompt


def test_game_logger_counts_novelty_and_repair_metrics(tmp_path) -> None:
    """評価ログの比較指標（修復回数・再生成・本文保持・新規性・待ち時間）を検証する。"""
    from werewolf.cli import GameLogger

    state = _state()
    logger = GameLogger(
        game_id="task048",
        log_dir=tmp_path,
        metadata={"mode": "mock", "novelty_policy": NOVELTY_POLICY_PUBLISH},
        private_enabled=False,
    )
    logger.append_record(
        {
            "type": "llm_action_completed",
            "action": "discussion",
            "player": "Player 2",
            "content_preserved": True,
            "fallback_used": False,
            "elapsed_seconds": 1.5,
        }
    )
    logger.append_record(
        {
            "type": "ledger_repair",
            "action": "discussion",
            "player": "Player 2",
            "reason": "repeated_discussion_topic",
        }
    )
    logger.append_record(
        {
            "type": "llm_action_parse_failed",
            "action": "discussion_body",
            "player": "Player 3",
            "will_retry": True,
            "reason": "repeated_discussion_topic",
        }
    )
    logger.append_record(
        {
            "type": "discussion_novelty_measured",
            "action": "discussion",
            "player": "Player 2",
            "model": "mock-model",
            "novelty_primary": NOVELTY_RESTATEMENT,
            "has_novelty": False,
            "novelty_restated_topics": 2,
            "novelty_legitimate_repetitions": 1,
        }
    )
    logger.append_record(
        {
            "type": "llm_discussion_low_novelty_kept",
            "action": "discussion",
            "player": "Player 2",
            "reason": "repeated_discussion_topic",
            "regeneration_avoided": True,
        }
    )
    logger.append_record(
        {
            "type": "discussion_novelty_appended",
            "action": "discussion",
            "player": "Player 3",
            "body_preserved": True,
        }
    )
    state.record_topics_from_speech("Player 2", REPEATED_BODY, 1, 1)
    logger.finalize(state, traces=[])

    records = [
        json.loads(line)
        for line in (tmp_path / "game_task048.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records[0]["type"] == "game_init"
    assert records[0]["novelty_policy"] == NOVELTY_POLICY_PUBLISH
    result = records[-1]
    metrics = result["ledger_metrics"]
    assert metrics["discussion_bodies"] == 1
    assert metrics["discussion_bodies_preserved"] == 1
    assert metrics["ledger_repairs"] == 1
    assert metrics["discussion_regenerations"] == 1
    assert metrics["novelty_measurements"] == 1
    assert metrics["novelty_present"] == 0
    assert metrics["novelty_restated_topics"] == 2
    assert metrics["novelty_legitimate_repetitions"] == 1
    assert metrics["low_novelty_kept"] == 1
    assert metrics["novelty_appends"] == 1
    assert metrics["novelty_class_counts"] == {NOVELTY_RESTATEMENT: 1}
    # 修復・再生成は「本文1件あたり」の率として比較できる（既定publishでは両方0になる）
    assert metrics["ledger_repair_rate"] == 1.0
    assert metrics["regeneration_rate"] == 1.0
    assert metrics["avg_discussion_seconds"] == 1.5
    assert metrics["topic_records_total"] >= 1
    # 論点台帳（連呼・正当な繰り返し・本文の再掲なし）が評価ログに残る
    assert result["topic_ledger"]["metrics"]["topic_records_total"] >= 1
    assert result["topic_ledger"]["records"]



