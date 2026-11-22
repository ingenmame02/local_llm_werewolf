"""TASK-052の回帰テスト（LL不使用）: 死亡者参照の事前解決と軽量補正。

設計思想の検証:
1. コンテキストを節約して性能を上げる:
   - プロンプト生成時（PREVIOUS_TOPICS, HYPOTHESIS_LEDGER, SELF_ACTION_MEMORY）に
     死亡者言及へ軽量属性（[historical_only], [cannot_answer]）を付与し、重複文や冗長な禁止命令文を増やさない。
   - 誤検出による80秒超の再生成や数千トークンの再要求を回避し、対象文のみの軽量補正で処理する。
2. モデルの一貫性を保つ:
   - 過去の投票理由や死亡者の役職・襲撃状況に関する論理的な考察をそのまま保持する（本文保持率100%）。
   - 「私の現在の判断対象は〜」のような文脈破壊的な定型文の誤挿入を防止する。
3. 合理的な議論をさせる:
   - 「過去の記録・推論への言及（Historical Reference）」と「現在の行動要求・質問（Actionable Demand）」
     を厳密に分離し、過去の投票や結果を根拠にした合理的な推理を妨げない。
4. 自由な論点、口調、考え方を縛らない:
   - 口調・結論・引用・疑念表明を制約せず、合法な言及はwarningのみでそのまま公開する。
"""
from __future__ import annotations

import pytest

from werewolf.dead_player_resolution import (
    REF_ACTIONABLE,
    REF_HISTORICAL,
    REF_QUESTION,
    analyze_dead_player_references,
    classify_dead_player_sentence,
)
from werewolf.discussion_novelty import DiscussionTopicLedger
from werewolf.engine import (
    SEVERITY_OK,
    SEVERITY_REPAIRABLE,
    SEVERITY_WARNING,
    diagnose_discussion_body,
)
from werewolf.hypothesis_ledger import HypothesisRecord, REFUTATION_HISTORICAL, STANCE_SUSPECT
from werewolf.llm_actions import OllamaActions
from werewolf.models import GameState, Message, Player, PlayerActionMemory, Role
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game


def _player(state: GameState, name: str) -> Player:
    return next(player for player in state.players if player.name == name)


def _setup_game_with_dead() -> GameState:
    state = create_game(
        seed=1001, config_path="config/players.json", response_language="日本語"
    )
    # Day 1 -> Day 2: Player 4 executed, Player 1 attacked
    state.mark_dead("Player 4")
    state.executions.append((1, "Player 4"))
    state.mark_dead("Player 1")
    state.attack_deaths.append((1, "Player 1"))
    state.day = 2
    return state


def test_classify_dead_player_sentence_historical_vs_actionable() -> None:
    dead_names = frozenset(["Player 1", "Player 4"])
    living_names = frozenset(["Player 2", "Player 3", "Player 5", "Player 6", "Player 7", "Player 8"])

    # 1. 過去の投票行動・投票理由の言及（合法的推理）
    c1 = classify_dead_player_sentence("先ほどの投票でPlayer 1に投票しました。", dead_names, living_names)
    assert c1.category == REF_HISTORICAL
    assert c1.is_historical_reference is True

    c2 = classify_dead_player_sentence(
        "私はPlayer 1に投票した理由は、Player 4の「Player 1＝狼」という主張を疑ったためです。",
        dead_names,
        living_names,
    )
    assert c2.category == REF_HISTORICAL

    c3 = classify_dead_player_sentence(
        "それよりも、今議論すべきは、初日に多数派に逆らってPlayer 1さんに投票したPlayer 7さんの動機ではないでしょうか。",
        dead_names,
        living_names,
    )
    assert c3.category == REF_HISTORICAL

    # 2. 生存者宛ての質問文中で死亡者に言及（背景情報としての言及）
    c4 = classify_dead_player_sentence(
        "Player 7さん、初日にPlayer 1さんを投票先にした具体的な理由と、Player 4さんのCOを疑わなかった根拠を一つだけ明確に説明していただけますか。",
        dead_names,
        living_names,
    )
    assert c4.category == REF_HISTORICAL
    assert c4.addressed_player == "Player 7"

    # 3. 他者の過去発言引用・処刑済みの指摘（メタ批判）
    c5 = classify_dead_player_sentence(
        "Player 7さんは本日「今日の投票ではPlayer 4に票を入れることを検討しています」と発言していますが（[speech:11]）、Player 4さんは既に処刑済みであり、投票対象にはなり得ません。",
        dead_names,
        living_names,
    )
    assert c5.category == REF_HISTORICAL

    # 4. 死亡・襲撃事実についての客観的整理
    c6 = classify_dead_player_sentence(
        "Player 1は夜襲で死亡し、Seerの結果も疑わしく、投票傾向から疑いが残ります。",
        dead_names,
        living_names,
    )
    assert c6.category == REF_HISTORICAL

    # 5. 死亡者への現在の直接質問（違反）
    c7 = classify_dead_player_sentence(
        "Player 2さんの疑問に答えるために、Player 1の基準を聞いてみたいです。",
        dead_names,
        living_names,
    )
    assert c7.category == REF_QUESTION
    assert c7.dead_player == "Player 1"
    assert c7.is_dead_player_question is True

    c8 = classify_dead_player_sentence(
        "Player 4さん、なぜPlayer 3に投票したのですか？理由を教えてください。",
        dead_names,
        living_names,
    )
    assert c8.category == REF_QUESTION
    assert c8.dead_player == "Player 4"

    # 6. 死亡者への現在の行動要求（違反）
    c9 = classify_dead_player_sentence(
        "今日の投票先はPlayer 4さんにします。",
        dead_names,
        living_names,
    )
    assert c9.category == REF_ACTIONABLE
    assert c9.dead_player == "Player 4"
    assert c9.is_actionable_target is True

    c10 = classify_dead_player_sentence(
        "Player 1を処刑すべきです。",
        dead_names,
        living_names,
    )
    assert c10.category == REF_ACTIONABLE
    assert c10.dead_player == "Player 1"


def test_diagnose_discussion_body_permits_historical_references() -> None:
    state = _setup_game_with_dead()
    actor = _player(state, "Player 5")

    # 過去投票への言及・他者への問いかけは warning のみ（修復・再生成不要）
    historical_body = (
        "Player 3さんの整理に同意します。"
        "ここで一つ重要な事実を整理すると、Player 1さんは夜襲で死亡しており、狼が自陣を襲わない以上、Player 1さんが狼だったとは考えにくい。"
        "初日投票を見れば、多数がPlayer 4を処刑したのに対し、Player 1とPlayer 7だけがPlayer 1に票を投じ、事実上Player 4を守っていました。"
        "Player 4さんは既に処刑済みなので、残る1匹の狼としてPlayer 7さんを重点的に追及すべきだと考えます。"
        "Player 7さん、初日にPlayer 1さんを投票先にした具体的な理由と、Player 4さんのCOを疑わなかった根拠を一つだけ明確に説明していただけますか。"
    )
    diag = diagnose_discussion_body(historical_body, actor, state, previous_message=None, round_transcript=())
    assert diag.publishable is True
    assert diag.repairable is False
    assert diag.fatal is False
    assert "dead_player_mention" in diag.reasons
    assert "dead_player_question" not in diag.reasons
    assert "dead_player_action_target" not in diag.reasons

    # 死亡者への直接質問は repairable
    question_body = "Player 1さん、あなたの占い基準を教えてください。"
    diag_q = diagnose_discussion_body(question_body, actor, state, previous_message=None, round_transcript=())
    assert diag_q.repairable is True
    assert "dead_player_question" in diag_q.reasons

    # 死亡者への投票要求は repairable
    target_body = "今日の投票先はPlayer 4にします。"
    diag_t = diagnose_discussion_body(target_body, actor, state, previous_message=None, round_transcript=())
    assert diag_t.repairable is True
    assert "dead_player_action_target" in diag_t.reasons


def test_llm_actions_preserves_historical_body_without_regeneration() -> None:
    state = _setup_game_with_dead()
    actions = OllamaActions(state, None)  # LL不使用
    actor = _player(state, "Player 7")

    # Player 7 の過去投票説明文
    raw_response = (
        "【CO】none\n"
        "【本文】先ほどの投票でPlayer 1に投票しました。"
        "Player 1は夜襲で死亡し、Seerの結果も疑わしく、投票傾向から疑いが残ります。"
        "Player 5がPlayer 4を疑っている点は理解できますが、現在の情報だけではPlayer 4のCOが真偽を判断できません。"
        "私はPlayer 2とPlayer 3の発言を観察し、投票を決める予定です。"
    )
    issue = actions._discussion_response_issue(actor, raw_response, ())
    assert issue is None  # 誤検出なく一発で受理

    # 修復関数に渡しても、過去記録文は置換されない（保持率100%）
    body = (
        "先ほどの投票でPlayer 1に投票しました。"
        "Player 1は夜襲で死亡し、Seerの結果も疑わしく、投票傾向から疑いが残ります。"
    )
    assert actions._dead_player_action_targets(body) == ()
    assert actions._dead_player_question_targets(body) == ()


def test_llm_actions_lightweight_sentence_repair_avoids_regeneration() -> None:
    state = _setup_game_with_dead()
    emitted_events = []
    actions = OllamaActions(
        state,
        None,
        event_callback=lambda event: emitted_events.append(event),
    )
    actor = _player(state, "Player 3")

    # Player 3 が末尾で死亡者へ質問してしまったケース
    raw_response = (
        "【CO】none\n"
        "【本文】Player 2さんの指摘に同意します。"
        "Player 1の占いが真なら、真占い師が狙われる展開は整合性があります。"
        "Player 2さんの疑問に答えるために、Player 1の基準を聞いてみたいです。"
    )
    issue = actions._discussion_response_issue(actor, raw_response, ())
    assert issue == "dead_player_question"

    repaired = actions._repair_discussion_response(actor, raw_response, issue, ())
    assert repaired is not None
    # 前半の合法的推論はそのまま保持されている
    assert "Player 2さんの指摘に同意します。" in repaired
    assert "Player 1の占いが真なら、真占い師が狙われる展開は整合性があります。" in repaired
    # 末尾の違反文のみが置換されている
    assert "Player 1の基準を聞いてみたいです。" not in repaired
    assert "Player 1さんは死亡しているため答えられません" in repaired


def test_prompt_pre_resolution_annotations() -> None:
    state = _setup_game_with_dead()
    actor = _player(state, "Player 5")

    # 1. PREVIOUS_TOPICS: 死亡者関連のトピックに [historical_only] タグが付与される
    topic_ledger = state.topic_ledger
    topic_ledger.record_speech(
        speaker="Player 7",
        content="Player 1に投票した理由を説明します。",
        day=2,
        turn=1,
        speech_id=1,
        player_names=tuple(p.name for p in state.players),
    )
    topic_ledger.record_speech(
        speaker="Player 2",
        content="Player 4さん、あなたのCO理由は何ですか？",
        day=1,
        turn=1,
        speech_id=2,
        player_names=tuple(p.name for p in state.players),
    )
    block, diag = state.topic_summary_for("Player 5", day=2)
    assert "[historical_only" in block

    # 2. HYPOTHESIS_LEDGER: 死亡者対象の仮説に [historical_only] が付与される
    state.record_hypotheses_from_speech(
        "Player 5",
        "Player 4さんは狼だったと考えます。",
        day=2,
        turn=1,
    )
    state.resolve_hypothesis_ledger()
    hyp_block, hyp_diag = state.render_hypotheses_for("Player 5")
    assert "[historical_only]" in hyp_block

    # 3. SELF_ACTION_MEMORY: 過去の投票対象が死亡している場合 [dead: historical record] が付与される
    state.player_action_memories.setdefault("Player 5", []).append(
        PlayerActionMemory(
            day=1,
            phase="voting",
            action_type="vote",
            vote_target="Player 4",
            reasoning_summary="Player 4 suspect",
            unanswered_questions=["Why vote for Player 4?"],
        )
    )
    mem_text = PromptBuilder._self_action_memory_text(actor, state, public_phase=True)
    assert "[dead: historical record]" in mem_text
    assert "[cannot_answer: dead player]" in mem_text


def test_game_088e49f0690c_all_false_positives_eliminated_and_body_preserved() -> None:
    """実機ログ game_088e49f0690c で発生した7件の死亡者参照を再現検証。

    - 誤検出されていた6件（過去投票の言及・他者への問いかけ・引用）が一発で受理され、
      本文保持率100%かつ再生成・ledger_repairなしで通過すること。
    - 唯一の真の死亡者質問（Player 3の末尾文）のみが最小限に補正され、全文再生成が回避されること。
    """
    state = _setup_game_with_dead()
    actions = OllamaActions(state, None)

    # 1. Player 5 (Discussion 4): 生存者Player 7への質問中でPlayer 1, 4に言及
    p5_speech_turn4 = (
        "【CO】none\n"
        "【本文】Player 3さんの整理に同意します。ここで一つ重要な事実を整理すると、"
        "Player 1さんは夜襲で死亡しており、狼が自陣を襲わない以上、Player 1さんが狼だったとは考えにくい。"
        "するとPlayer 4さんの「Player 1＝狼」判定は偽り、Player 4さんが偽CO（狼）だった可能性が高いと考えます。"
        "初日投票を見れば、多数がPlayer 4を処刑したのに対し、Player 1とPlayer 7だけがPlayer 1に票を投じ、事実上Player 4を守っていました。"
        "Player 4さんは既に処刑済みなので、残る1匹の狼としてPlayer 7さんを重点的に追及すべきだと考えます。"
        "Player 7さん、初日にPlayer 1さんを投票先にした具体的な理由と、Player 4さんのCOを疑わなかった根拠を一つだけ明確に説明していただけますか。"
    )
    p5 = _player(state, "Player 5")
    assert actions._discussion_response_issue(p5, p5_speech_turn4, ()) is None
    diag_p5_4 = diagnose_discussion_body(p5_speech_turn4, p5, state, previous_message=None, round_transcript=())
    assert diag_p5_4.publishable is True
    assert diag_p5_4.repairable is False

    # 2. Player 7 (Discussion 4): 自身の初日投票先の言及
    p7_speech_turn4 = (
        "【CO】none\n"
        "【本文】先ほどの投票でPlayer 1に投票しました。"
        "Player 1は夜襲で死亡し、Seerの結果も疑わしく、投票傾向から疑いが残ります。"
        "Player 5がPlayer 4を疑っている点は理解できますが、現在の情報だけではPlayer 4のCOが真偽を判断できません。"
        "私はPlayer 2とPlayer 3の発言を観察し、投票を決める予定です。"
    )
    p7 = _player(state, "Player 7")
    assert actions._discussion_response_issue(p7, p7_speech_turn4, ()) is None
    diag_p7_4 = diagnose_discussion_body(p7_speech_turn4, p7, state, previous_message=None, round_transcript=())
    assert diag_p7_4.publishable is True
    assert diag_p7_4.repairable is False

    # 3. Player 5 (Discussion 5): 初日投票の分析とPlayer 7への追及
    p5_speech_turn5 = (
        "【CO】none\n"
        "【本文】Player 3さんの整理に同意します。ここで確認すべき核心は、"
        "Player 1さんが夜襲で死亡した以上狼ではありえない、つまりPlayer 4さんの「Player 1＝狼」判定は偽であり、"
        "Player 4さんが狼だったという点です。初日投票でPlayer 4に6票が集中した中、"
        "Player 1とPlayer 7だけがPlayer 1に票を投じ、事実上Player 4を守った形になっています。"
        "Player 7さん、初日にPlayer 1に投票した具体的な理由を今この場で説明していただけますか。"
    )
    assert actions._discussion_response_issue(p5, p5_speech_turn5, ()) is None
    diag_p5_5 = diagnose_discussion_body(p5_speech_turn5, p5, state, previous_message=None, round_transcript=())
    assert diag_p5_5.publishable is True
    assert diag_p5_5.repairable is False

    # 4. Player 8 (Discussion 5): 議論対象の焦点化（初日投票の動機）
    p8_speech_turn5 = (
        "【CO】none\n"
        "【本文】Player 7さんが仰る通り、Player 4さんの主張には矛盾がありましたが、"
        "Player 4さんは既に処刑されているため、これ以上彼から根拠を引き出すことは不可能です。"
        "それよりも、今議論すべきは、初日に多数派に逆らってPlayer 1さんに投票したPlayer 7さんの動機ではないでしょうか。"
        "Player 5さんやPlayer 6さんが指摘するように、Player 7さんの行動は非常に不自然です。"
    )
    p8 = _player(state, "Player 8")
    assert actions._discussion_response_issue(p8, p8_speech_turn5, ()) is None
    diag_p8_5 = diagnose_discussion_body(p8_speech_turn5, p8, state, previous_message=None, round_transcript=())
    assert diag_p8_5.publishable is True
    assert diag_p8_5.repairable is False

    # 5. Player 5 (Discussion 6): 他者引用と処刑済み事実の指摘（88s再生成の元凶）
    p7_msg = Message(speaker="Player 7", content="今日の投票ではPlayer 4に票を入れることを検討しています")
    p5_speech_turn6 = (
        "【CO】none\n"
        "【本文】Player 2さん、Player 3さんの整理に同意します。ここで決定的な事実を一つ追加します。"
        "Player 7さんは本日「今日の投票ではPlayer 4に票を入れることを検討しています」と発言していますが（[speech:11]）、"
        "Player 4さんは既に処刑済みであり、投票対象にはなり得ません。"
        "初日に多数派と逆の投票でPlayer 4を守り、その説明を今も示せず、さらに死亡済みプレイヤーへの投票意向まで述べている"
        "Player 7さんの行動は、他の誰よりも不自然です。私は今日、投票先をPlayer 7さんに確定します。"
    )
    assert actions._discussion_response_issue(p5, p5_speech_turn6, (p7_msg,)) is None
    diag_p5_6 = diagnose_discussion_body(p5_speech_turn6, p5, state, previous_message=None, round_transcript=())
    assert diag_p5_6.publishable is True
    assert diag_p5_6.repairable is False  # 再生成ゼロ！

    # 6. Player 7 (Discussion 6): 初日投票の理由説明
    p7_speech_turn6 = (
        "【CO】none\n"
        "【本文】私はPlayer 1に投票した理由は、Player 4の「Player 1＝狼」という主張を疑ったためです。"
        "Player 4は既に処刑され、彼の主張が真ならPlayer 1が狼であることになるため、私はPlayer 1に注意を向けました。"
    )
    assert actions._discussion_response_issue(p7, p7_speech_turn6, ()) is None
    diag_p7_6 = diagnose_discussion_body(p7_speech_turn6, p7, state, previous_message=None, round_transcript=())
    assert diag_p7_6.publishable is True
    assert diag_p7_6.repairable is False

    # 7. Player 3 (Discussion 4): 末尾で死亡者へ質問してしまった真の違反
    p3_speech_turn4 = (
        "【CO】none\n"
        "【本文】Player 2さんの指摘に同意します。Player 4の主張が真なら、Player 1は狼であり、狼が自陣を襲う理由が説明できないという点は重要です。"
        "Player 1の占いが真なら、真占い師が狙われる展開は整合性があります。私はPlayer 1から人間判定を受けていますが、まだ具体的な疑念を持っているプレイヤーが誰かを確認する必要があります。"
        "Player 2さんの疑問に答えるために、Player 1の基準を聞いてみたいです。"
    )
    p3 = _player(state, "Player 3")
    issue_p3 = actions._discussion_response_issue(p3, p3_speech_turn4, ())
    assert issue_p3 == "dead_player_question"
    repaired_p3 = actions._repair_discussion_response(p3, p3_speech_turn4, issue_p3, ())
    assert repaired_p3 is not None
    # 前半の長文推論は100%保持されている
    assert "Player 2さんの指摘に同意します。" in repaired_p3
    assert "Player 1の占いが真なら、真占い師が狙われる展開は整合性があります。" in repaired_p3
    # 末尾の質問のみが安全に補正されている
    assert "Player 1の基準を聞いてみたいです。" not in repaired_p3
    assert "Player 1さんは死亡しているため答えられません" in repaired_p3

