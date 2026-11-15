"""重複判定のシグナル抽出が None group で落ちる回帰テスト（LLM不使用）。

`_adds_new_discussion_signal` の cue は分岐を含む正規表現で、末尾の capture group が
英語分岐側だけで参加していたため、日本語の cue が一致すると group(1) が None になり
`re.sub` が TypeError を投げていた。
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, SRC_DIR)

from werewolf.llm_actions import OllamaActions
from werewolf.models import Message, Role
from werewolf.setup import create_game


class NoneContentClient:
    timeout = 60.0
    max_retries = 1
    timeout_retries = 1

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {"ollama_ps_context": None}

    def chat(self, model, messages, options=None):
        return None

    def chat_with_metadata(self, model, messages, options=None, tools=None):
        return None


def test_japanese_cue_signal_no_type_error() -> None:
    # 日本語の cue（理由/根拠等）が第一分岐で一致するケース。
    assert OllamaActions._adds_new_discussion_signal(
        " Player 3への投票が理由です。",
        " Player 4への投票が理由です。",
    ) is True
    assert OllamaActions._adds_new_discussion_signal(
        " Player 3への投票が理由です。",
        " Player 3への投票が理由です。",
    ) is False
    # cue の直後が句読点で空になる文も安全に処理する。
    assert OllamaActions._adds_new_discussion_signal("理由は。", "理由は。") is False


def test_english_vote_target_signal() -> None:
    assert OllamaActions._adds_new_discussion_signal(
        "Vote for Player 5.",
        "Vote for Player 4.",
    ) is True
    assert OllamaActions._adds_new_discussion_signal(
        "Vote for Player 5.",
        "Vote for Player 5.",
    ) is False


def test_signal_helpers_tolerate_none() -> None:
    assert OllamaActions._adds_new_discussion_signal(None, "理由があります。") is False
    assert OllamaActions._adds_new_discussion_signal("理由があります。", None) is True
    assert OllamaActions._adds_new_discussion_signal("", "") is False


def test_topic_repeated_handles_japanese_cues() -> None:
    state = create_game(seed=1001, config_path="config/players.json", response_language="日本語")
    actions = OllamaActions(state, NoneContentClient(), seed=1001)
    prior = (Message(speaker="Player 2", content=" Player 4への投票が理由です。"),)
    # 完全重複（新シグナルなし）は重複扱い。
    assert actions._discussion_topic_repeated(" Player 4への投票が理由です。", prior) is True
    # 投票先が新しい文は新シグナルとして重複扱いしない（旧実装では TypeError）。
    assert actions._discussion_topic_repeated(" Player 3への投票が理由です。", prior) is False
    # 英語 cue も同じ経路で安全に処理する。
    assert actions._discussion_topic_repeated(
        " The reason is the voting pattern of Player 4.", prior
    ) is False
