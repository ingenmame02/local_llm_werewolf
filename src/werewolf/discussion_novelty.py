"""TASK-048: 議論の新規性を「誘導」ではなく検索・計測で改善する。

`DiscussionTopicLedger` は直近の議論から 主張・質問・回答・投票提案・立場 を軽量に抽出し、
次ターンのコンテキストへ「既出論点の要約」として1回だけ添える（全文の再掲はしない）。
論点・口調・結論の強制は一切行わず、観測だけを渡す。

`classify_novelty` は出力後の本文を 新規証拠 / 反証 / 立場変更 / 質問への応答（＋新論点）
として計測する。同じ論点の再確認でも新しい根拠・反証を伴うものは正当な繰り返しとして
区別し、新シグナルのない文だけを連呼（restatement）として数える。再生成・文の除去・
修復定型文の挿入はここでは行わない（本文の自由を削らない）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Mapping, Sequence

from .hypothesis_ledger import stance_of_sentence

TOPIC_CLAIM = "claim"
TOPIC_QUESTION = "question"
TOPIC_ANSWER = "answer"
TOPIC_VOTE = "vote_proposal"
TOPIC_STANCE = "stance"
TOPIC_KINDS: frozenset[str] = frozenset(
    {TOPIC_CLAIM, TOPIC_QUESTION, TOPIC_ANSWER, TOPIC_VOTE, TOPIC_STANCE}
)

# 出力後の計測クラス（プロンプトで「必ず」要求するものではない）
NOVELTY_QUESTION_ANSWER = "question_answer"
NOVELTY_POSITION_CHANGE = "position_change"
NOVELTY_REFUTATION = "refutation"
NOVELTY_NEW_EVIDENCE = "new_evidence"
NOVELTY_NEW_TOPIC = "new_topic"
NOVELTY_RESTATEMENT = "restatement"
NOVELTY_NO_SIGNAL = "no_signal"

MEASURED_NOVELTY_CLASSES: tuple[str, ...] = (
    NOVELTY_QUESTION_ANSWER,
    NOVELTY_POSITION_CHANGE,
    NOVELTY_REFUTATION,
    NOVELTY_NEW_EVIDENCE,
    NOVELTY_NEW_TOPIC,
)
NOVELTY_CLASSES: tuple[str, ...] = MEASURED_NOVELTY_CLASSES + (
    NOVELTY_RESTATEMENT,
    NOVELTY_NO_SIGNAL,
)

# 低新規性のときの処理ポリシー（既定はそのまま公開＝再生成も文の除去もしない）
NOVELTY_POLICY_PUBLISH = "publish"
NOVELTY_POLICY_APPEND = "append"
NOVELTY_POLICY_LEGACY_REPAIR = "legacy_repair"
NOVELTY_POLICIES: frozenset[str] = frozenset(
    {NOVELTY_POLICY_PUBLISH, NOVELTY_POLICY_APPEND, NOVELTY_POLICY_LEGACY_REPAIR}
)
DEFAULT_NOVELTY_POLICY = NOVELTY_POLICY_PUBLISH

# 低新規性として計測する検証理由名（llm_actions の応答検証理由と同一の語を使う）。
# publish / append ではこれらを失敗として扱わず、本文をそのまま公開する。
NOVELTY_MEASURED_ISSUES: frozenset[str] = frozenset(
    {"repeated_discussion_topic", "repeated_speech", "vague_wait_and_see"}
)

_NOVELTY_POLICY_ALIASES: dict[str, str] = {
    "measure": NOVELTY_POLICY_PUBLISH,
    "keep": NOVELTY_POLICY_PUBLISH,
    "no_repair": NOVELTY_POLICY_PUBLISH,
    "publish_only": NOVELTY_POLICY_PUBLISH,
    "append_request": NOVELTY_POLICY_APPEND,
    "short_append": NOVELTY_POLICY_APPEND,
    "repair": NOVELTY_POLICY_LEGACY_REPAIR,
    "legacy": NOVELTY_POLICY_LEGACY_REPAIR,
}


def normalize_novelty_policy(value: object) -> str:
    """低新規性ポリシーの正規化（未知の値は既定 publish に落とす）。"""
    if value is None:
        return DEFAULT_NOVELTY_POLICY
    if isinstance(value, bool):
        return NOVELTY_POLICY_APPEND if value else NOVELTY_POLICY_PUBLISH
    if not isinstance(value, str):
        return DEFAULT_NOVELTY_POLICY
    normalized = value.strip().lower()
    if normalized in NOVELTY_POLICIES:
        return normalized
    return _NOVELTY_POLICY_ALIASES.get(normalized, DEFAULT_NOVELTY_POLICY)


# --- 論点の抽出（論点・口調の縛りではなく「何を述べたか」の識別のみ） ---

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])\s*|\n+")

_QUESTION_CUES: tuple[str, ...] = (
    "?",
    "？",
    "ですか",
    "ますか",
    "でしょうか",
    "聞きたい",
    "聞かせて",
    "教えて",
    "説明して",
    "確認したい",
    "質問",
    "なぜ",
    "どうして",
    "who",
    "what",
    "why",
    "how",
    "ask",
    "question",
    "explain",
    "check",
)
_CLAIM_CUE = re.compile(
    r"(?:占い師|人狼|狂人|村人)?CO|主張|判定\s*=|(?i:\bCO\b|result\s*=|\bclaim\b)",
    re.IGNORECASE,
)
_ANSWER_CUE = re.compile(
    r"^(?:To\s+|回答|答えます|答え(?:ます|たい)|(?:さんの)?質問に答)|(?i:^(?:answer|respond|reply)\b)",
    re.IGNORECASE,
)
_VOTE_TARGET = re.compile(
    r"(?:投票先(?:は|を)?\s*)?(Player\s+\d+|P\d+)(?:さん)?(?:へ|に)(?:の)?(?:投票|票)"
    r"|(?:vote for|voting for|my vote is for|vote target[:=]?)\s*(Player\s+\d+|P\d+)",
    re.IGNORECASE,
)
_ROLE_WORD_CUE = re.compile(
    r"(?:人狼|狼|狂人|占い師|村人)|(?i:\b(?:werewolf|wolf|madman|seer|villager)\b)",
    re.IGNORECASE,
)
# 新シグナルの抽出（投票先・根拠・反証）
_SIGNAL_CUES: tuple[tuple[str, str], ...] = (
    (
        "evidence",
        r"(?:根拠|理由|具体的には|記録では|発言では)|(?i:\b(?:basis|reason|evidence|record|speech)\b)",
    ),
    (
        "refutation",
        r"(?:反証|反論|矛盾|ただし|しかし|一方で|訂正)|(?i:\b(?:refut|contradict|however|but|correct|revis)\w*)",
    ),
)
_REFUTATION_CUE = _SIGNAL_CUES[1][1]

TOPIC_TEXT_MAX_CHARS = 70
SUMMARY_TEXT_MAX_CHARS = 60
TOPIC_MIN_CHARS = 10
TOPIC_SIMILARITY_THRESHOLD = 0.82
SUMMARY_MAX_LINES = 6
MAX_TOPICS_PER_SPEECH = 6


def split_sentences(text: str | None) -> tuple[str, ...]:
    """文分割（None・非文字列は空_tuple。cue抽出のTypeErrorを防止する）。"""
    if text is None or not isinstance(text, str):
        return ()
    return tuple(
        sentence.strip()
        for sentence in _SENTENCE_SPLIT.split(text)
        if sentence.strip()
    )


def normalize_text(text: str | None) -> str:
    if text is None or not isinstance(text, str):
        return ""
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def extract_signals(sentence: str | None) -> set[tuple[str, str]]:
    """文から新シグナル（投票先・根拠・反証）を抽出する。

    cue は分岐を含むため `(?:cue)` で包む。素直に連結すると末尾の capture group が
    英語分岐側だけになり、日本語一致で group(1) が None になる（TASK-042の修正と同じ）。
    """
    if not sentence or not isinstance(sentence, str):
        return set()
    found: set[tuple[str, str]] = set()
    vote_target = _VOTE_TARGET.search(sentence)
    if vote_target is not None:
        target = vote_target.group(1) or vote_target.group(2)
        if target:
            found.add(("vote_target", normalize_text(target)))
    for kind, cue in _SIGNAL_CUES:
        match = re.search(r"(?:" + cue + r")([^。！？!?]{1,36})", sentence)
        if match is not None:
            value = normalize_text(match.group(1) or "")
            if value:
                found.add((kind, value))
    return found


def adds_new_signal(current: str | None, previous: str | None) -> bool:
    """同一論点の繰り返しでも、新しい根拠・反証・投票先があれば新シグナルとみなす。"""
    return bool(extract_signals(current) - extract_signals(previous))


def topic_kind(sentence: str) -> str | None:
    """文の論点種別（質問 > 主張 > 投票提案 > 回答 > 立場）。None は論点を持たない文。"""
    if not isinstance(sentence, str) or not sentence.strip():
        return None
    lowered = sentence.lower()
    if any(cue in sentence or cue in lowered for cue in _QUESTION_CUES):
        return TOPIC_QUESTION
    if _CLAIM_CUE.search(sentence) is not None:
        return TOPIC_CLAIM
    if _VOTE_TARGET.search(sentence) is not None:
        return TOPIC_VOTE
    if _ANSWER_CUE.search(sentence) is not None:
        return TOPIC_ANSWER
    if stance_of_sentence(sentence) is not None:
        return TOPIC_STANCE
    if _ROLE_WORD_CUE.search(sentence) is not None:
        return TOPIC_CLAIM
    return None


def topic_subject(
    sentence: str, player_names: Iterable[str], speaker: str | None = None
) -> str | None:
    """文が対象とするプレイヤー（発言者自身を除く）を文中の出現順で返す。"""
    if not isinstance(sentence, str):
        return None
    best: tuple[int, str] | None = None
    for name in player_names:
        if not isinstance(name, str) or name == speaker:
            continue
        index = sentence.find(name)
        if index < 0:
            continue
        if best is None or index < best[0]:
            best = (index, name)
    return best[1] if best is not None else None


def topic_signature(kind: str, subject: str | None, sentence: str) -> str:
    """論点のID。対象を持つ論点は（種別, 対象）で安定する（文言が変わっても同一論点）。

    対象を持たない論点（自身の立場の説明など）は短縮本文で区別する。安定IDにより、
    新根拠を添えた再確認でレコードが増えず、要約も1行に統合される。
    """
    if subject is not None:
        return f"T-{kind}-{subject}"
    normalized = normalize_text(sentence)[:TOPIC_TEXT_MAX_CHARS]
    return f"T-{kind}-general-{normalized}"


def _similar(a: str, b: str) -> bool:
    if len(a) < TOPIC_MIN_CHARS or len(b) < TOPIC_MIN_CHARS:
        return False
    return SequenceMatcher(None, a, b).ratio() >= TOPIC_SIMILARITY_THRESHOLD


@dataclass
class TopicRecord:
    """既出論点1件。本文は短縮して保持し、`speech_id` で全文を参照する。"""

    topic_id: str
    kind: str
    subject: str | None
    speaker: str
    text: str
    day: int
    turn: int
    speech_id: str
    seen: int = 1
    last_day: int = 1
    last_turn: int = 0
    speakers: list[str] = field(default_factory=list)
    signals: set[tuple[str, str]] = field(default_factory=set)


@dataclass
class DiscussionTopicLedger:
    """既出論点台帳。連呼の統合（新レコードを作らない）と正当な繰り返しを区別する。"""

    records: dict[str, TopicRecord] = field(default_factory=dict)
    restatements: int = 0
    legitimate_repetitions: int = 0
    created_topics: int = 0

    def match_topic(
        self, sentence: str, kind: str, subject: str | None
    ) -> TopicRecord | None:
        """同じ論点種別・対象で近似一致する既存記録を探す（対象を持たない論点の統合用）。"""
        normalized = normalize_text(sentence)
        for record in self.records.values():
            if record.kind != kind or record.subject != subject:
                continue
            if _similar(normalized, normalize_text(record.text)):
                return record
        return None

    def find_topic(
        self, sentence: str, kind: str, subject: str | None
    ) -> TopicRecord | None:
        """既出論点として同じ論点種別・対象の記録を探す。

        対象を持つ論点は（種別, 対象）でグループ化する（投票先の再確認など、文言が変わっても
        同じ論点として扱える）。対象を持たない論点は文中の近似一致で統合する。
        """
        if subject is not None:
            for record in self.records.values():
                if record.kind == kind and record.subject == subject:
                    return record
            return None
        return self.match_topic(sentence, kind, subject)

    def record_speech(
        self,
        *,
        speaker: str,
        content: str | None,
        day: int,
        turn: int,
        speech_id: str,
        player_names: Iterable[str],
    ) -> int:
        """公開発言から論点を登録する。同じ論点の再掲は seen を増やして統合する。"""
        if not isinstance(content, str) or not content.strip():
            return 0
        names = tuple(player_names)
        sentences = split_sentences(content)
        # 発言全体のシグナル（新根拠・新投票先・反証）を先に集める。同一論点の再確認でも
        # 別の文に新根拠があれば正当な繰り返しとして扱う（文単位の照合では取りこぼすため）。
        speech_signals: set[tuple[str, str]] = set()
        for sentence in sentences:
            speech_signals |= extract_signals(sentence)
        created = 0
        for sentence in sentences:
            if created >= MAX_TOPICS_PER_SPEECH:
                break
            kind = topic_kind(sentence)
            if kind is None:
                continue
            subject = topic_subject(sentence, names, speaker)
            existing = self.find_topic(sentence, kind, subject)
            if existing is not None:
                if adds_new_signal(sentence, existing.text) or bool(
                    speech_signals - existing.signals
                ):
                    # 正当な繰り返し（投票先の再確認＋新根拠など）は連呼として数えない
                    self.legitimate_repetitions += 1
                    existing.text = sentence.strip()[:TOPIC_TEXT_MAX_CHARS]
                else:
                    self.restatements += 1
                existing.seen += 1
                existing.last_day = day
                existing.last_turn = turn
                if speaker not in existing.speakers:
                    existing.speakers.append(speaker)
                existing.signals |= extract_signals(sentence)
                continue
            signature = topic_signature(kind, subject, sentence)
            if signature in self.records:
                record = self.records[signature]
                record.seen += 1
                record.last_day = day
                record.last_turn = turn
                if speaker not in record.speakers:
                    record.speakers.append(speaker)
                record.signals |= extract_signals(sentence)
                self.restatements += 1
                continue
            self.records[signature] = TopicRecord(
                topic_id=signature,
                kind=kind,
                subject=subject,
                speaker=speaker,
                text=sentence.strip()[:TOPIC_TEXT_MAX_CHARS],
                day=day,
                turn=turn,
                speech_id=speech_id,
                last_day=day,
                last_turn=turn,
                speakers=[speaker],
                signals=extract_signals(sentence),
            )
            self.created_topics += 1
            created += 1
        return created

    def topics_for_day(self, day: int) -> list[TopicRecord]:
        return [record for record in self.records.values() if record.last_day == day]

    def prior_signals(self, day: int) -> set[tuple[str, str]]:
        return {
            signal
            for record in self.topics_for_day(day)
            for signal in record.signals
        }

    def summarize_for(
        self, actor_name: str, day: int, max_lines: int = SUMMARY_MAX_LINES
    ) -> tuple[str, dict[str, object]]:
        """既出論点の要約（観測のみ・論点や口調の強制しない）を1ブロックだけ作る。"""
        topics = self.topics_for_day(day)
        diagnostics: dict[str, object] = {
            "topic_records_total": len(self.records),
            "topic_records_today": len(topics),
            "topic_restatements_observed": self.restatements,
            "topic_legitimate_repetitions_observed": self.legitimate_repetitions,
            "topic_summary_lines": 0,
            "topic_summary_chars": 0,
        }
        if not topics:
            return "", diagnostics
        lines = [
            "- Topics already raised today (observation only, not a rule):",
            "Nothing here forces your topic, conclusion or tone. You may restate a point "
            "with a new reason, change position, ask something else, or pass.",
        ]
        recent = topics[-max_lines:]
        for record in reversed(recent):
            other_speakers = [
                name for name in record.speakers if name != record.speaker
            ][:2]
            parts = [
                f"Day {record.last_day} Turn {record.last_turn} {record.speaker} "
                f"[{record.kind}]"
                + (f" about {record.subject}" if record.subject else "")
                + f" {record.speech_id}: {record.text[:SUMMARY_TEXT_MAX_CHARS]}"
            ]
            if record.seen > 1:
                parts.append(f"raised {record.seen}x")
            if other_speakers:
                parts.append("also " + ", ".join(other_speakers))
            lines.append("- " + "; ".join(parts))
        omitted = len(topics) - len(recent)
        if omitted > 0:
            lines.append(f"- ({omitted} older topic(s) not listed)")
        block = "\n".join(lines)
        diagnostics["topic_summary_lines"] = len(recent)
        diagnostics["topic_summary_chars"] = len(block)
        return block, diagnostics

    def metrics(self) -> dict[str, object]:
        return {
            "topic_records_total": len(self.records),
            "topic_records_created": self.created_topics,
            "topic_restatements_observed": self.restatements,
            "topic_legitimate_repetitions_observed": self.legitimate_repetitions,
            "topic_kind_counts": {
                kind: sum(1 for record in self.records.values() if record.kind == kind)
                for kind in TOPIC_KINDS
            },
        }

    def serialize(self) -> dict[str, object]:
        return {
            "records": [
                {
                    "topic_id": record.topic_id,
                    "kind": record.kind,
                    "subject": record.subject,
                    "speaker": record.speaker,
                    "text": record.text,
                    "day": record.day,
                    "turn": record.turn,
                    "speech_id": record.speech_id,
                    "seen": record.seen,
                    "last_day": record.last_day,
                    "last_turn": record.last_turn,
                    "speakers": list(record.speakers),
                    "signals": sorted(
                        f"{kind}:{value}" for kind, value in record.signals
                    ),
                }
                for record in self.records.values()
            ],
            "metrics": self.metrics(),
        }


@dataclass(frozen=True)
class NoveltyResult:
    """出力後本文の新規性計測結果（修復・再生成の指示ではない）。"""

    primary: str
    classes: tuple[str, ...]
    restated_topic_ids: tuple[str, ...]
    legitimate_repetition_ids: tuple[str, ...]
    new_topic_ids: tuple[str, ...]
    novelty_score: float
    topic_sentence_count: int

    @property
    def has_novelty(self) -> bool:
        return bool(self.classes)


def find_prior_topic(
    sentence: str, kind: str, subject: str | None, prior_topics: Sequence[TopicRecord]
) -> TopicRecord | None:
    """既出論点台帳から同じ論点（種別・対象）を探す（対象なしは近似一致で統合）。"""
    if subject is not None:
        for record in prior_topics:
            if record.kind == kind and record.subject == subject:
                return record
        return None
    normalized = normalize_text(sentence)
    for record in prior_topics:
        if record.kind == kind and record.subject is None and _similar(
            normalized, normalize_text(record.text)
        ):
            return record
    return None


def classify_novelty(
    body: str | None,
    *,
    actor_name: str,
    player_names: Iterable[str],
    prior_topics: Sequence[TopicRecord],
    question_answered: bool = False,
    own_prior_stances: Mapping[str, str] | None = None,
) -> NoveltyResult:
    """本文を 新規証拠・反証・立場変更・質問への応答（＋新論点）で計測する。

    同じ論点の再確認でも新根拠・新投票先を伴うものは legitimate repetition に分け、
    新シグナルのない文だけを restatement として数える。判定は観測であり、
    本文の書き換え・再生成要求には使わない（論点・口調・結論はモデルの選択のまま）。
    """
    classes: list[str] = []
    restated: list[str] = []
    legitimate: list[str] = []
    new_topics: list[str] = []
    prior_signals = {
        signal for record in prior_topics for signal in record.signals
    }
    stances = own_prior_stances or {}
    sentences = split_sentences(body)
    # 本文全体のシグナル（新根拠・新投票先・反証）。同一論点の再確認でも本文のどこかに
    # 新シグナルがあれば連呼ではなく正当な繰り返しとして区別する。
    body_signals: set[tuple[str, str]] = set()
    for sentence in sentences:
        body_signals |= extract_signals(sentence)
    new_body_signals = body_signals - prior_signals
    topic_count = 0
    for sentence in sentences:
        kind = topic_kind(sentence)
        if kind is None:
            continue
        topic_count += 1
        subject = topic_subject(sentence, player_names, actor_name)
        signature = topic_signature(kind, subject, sentence)
        matched = find_prior_topic(sentence, kind, subject, prior_topics)
        signals = extract_signals(sentence)
        new_signals = signals - prior_signals
        if matched is None:
            if signature not in new_topics:
                new_topics.append(signature)
        elif adds_new_signal(sentence, matched.text) or bool(new_body_signals):
            if signature not in legitimate:
                legitimate.append(signature)
            if any(kind_name == "vote_target" for kind_name, _ in new_signals):
                if signature not in new_topics:
                    new_topics.append(signature)
        elif signature not in restated:
            restated.append(signature)
        if any(kind_name == "refutation" for kind_name, _ in new_signals) or (
            re.search(_REFUTATION_CUE, sentence) is not None
        ):
            if NOVELTY_REFUTATION not in classes:
                classes.append(NOVELTY_REFUTATION)
        if any(kind_name == "evidence" for kind_name, _ in new_signals):
            if NOVELTY_NEW_EVIDENCE not in classes:
                classes.append(NOVELTY_NEW_EVIDENCE)
        stance = stance_of_sentence(sentence)
        if stance is not None and subject is not None:
            prior_stance = stances.get(subject)
            if prior_stance is not None and prior_stance != stance:
                if NOVELTY_POSITION_CHANGE not in classes:
                    classes.append(NOVELTY_POSITION_CHANGE)
    if question_answered and NOVELTY_QUESTION_ANSWER not in classes:
        classes.append(NOVELTY_QUESTION_ANSWER)
    if new_topics and NOVELTY_NEW_TOPIC not in classes:
        classes.append(NOVELTY_NEW_TOPIC)
    ordered = tuple(
        clazz for clazz in MEASURED_NOVELTY_CLASSES if clazz in classes
    )
    if ordered:
        primary = ordered[0]
        score = round(len(ordered) / max(1, len(ordered) + len(restated)), 4)
    elif restated:
        primary = NOVELTY_RESTATEMENT
        score = 0.0
    else:
        primary = NOVELTY_NO_SIGNAL
        score = 0.0
    return NoveltyResult(
        primary=primary,
        classes=ordered,
        restated_topic_ids=tuple(restated),
        legitimate_repetition_ids=tuple(legitimate),
        new_topic_ids=tuple(new_topics),
        novelty_score=score,
        topic_sentence_count=topic_count,
    )




    if not isinstance(value, str):
        return DEFAULT_NOVELTY_POLICY
    normalized = value.strip().lower()
    if normalized in NOVELTY_POLICIES:
        return normalized
    return _NOVELTY_POLICY_ALIASES.get(normalized, DEFAULT_NOVELTY_POLICY)
