"""TASK-052: 死亡者参照の事前解決と軽量補正。

設計思想:
1. コンテキストを節約して性能を上げる:
   - プロンプト生成時に死亡者への過去の質問・行動要求へ軽量属性（`[historical_only]`, `[cannot_answer]`）
     を付与し、重複文や冗長な禁止命令文を増やさない。
   - 誤検出を排除し、真の違反時も全文再生成ではなく単一文の軽量置換とすることで、
     長時間の再生成遅延とトークン浪費を根絶する。
2. モデルの一貫性を保つ:
   - 過去の投票理由や死亡者の役職・襲撃状況に関する考察を正常な推論としてそのまま保持する。
   - 文脈と乖離した定型文の安易な挿入を停止し、文脈の一貫性を保護する。
3. 合理的な議論をさせる:
   - 「過去の記録・推論への言及（Historical Reference）」と「現在の行動要求・質問（Actionable Demand）」
     を厳密に分離する。
   - 過去の処刑・襲撃や他者の過去投票行動を根拠とした論理的な推理・反証を阻害しない。
4. 自由な論点、口調、考え方を縛らない:
   - モデルの口調・結論・引用・疑念表明を制限する命令を追加しない。
   - 検出は警告（dead_player_mention）を基本とし、修復は現在実行不能な要求文のみの最小限の置換に留める。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

REF_HISTORICAL = "historical_reference"
REF_ACTIONABLE = "actionable_target"
REF_QUESTION = "dead_player_question"
REF_NONE = "none"

_PAST_CUES = (
    r"投票した",
    r"票を入れた",
    r"票を投じた",
    r"処刑された",
    r"処刑済み",
    r"処刑した",
    r"襲撃された",
    r"襲撃で死亡",
    r"死亡した",
    r"死んだ",
    r"落ちた",
    r"吊られた",
    r"voted for",
    r"was executed",
    r"executed",
    r"was attacked",
    r"died",
    r"eliminated",
)

_REASON_OR_MODIFIER_CUES = (
    r"への投票理由",
    r"に投票した理由",
    r"に票を入れた動機",
    r"を守った形",
    r"への投票行動",
    r"を処刑したのに対し",
    r"への投票先",
    r"を投票先にした",
    r"への投票",
    r"の発言",
    r"の主張",
    r"のCO",
    r"の占い",
    r"の結果",
    r"の死",
    r"が真なら",
    r"が偽",
    r"と矛盾",
    r"を疑ったため",
    r"を疑った理由",
    r"を疑う理由",
    r"を庇",
    r"をかば",
    r"reason for voting",
    r"motive",
    r"defended",
    r"was eliminated",
    r"already dead",
)

_QUOTE_OR_META_CUES = (
    r"「.*?」",
    r"『.*?』",
    r"と発言して",
    r"と言って",
    r"と主張して",
    r"について",
    r"に関しては",
    r"処刑済みであり",
    r"投票対象にはなり得ません",
    r"投票対象になり得ない",
    r"既に死亡",
    r"もう死んで",
    r"cannot be voted",
    r"already executed",
)

_PAST_PATTERN = re.compile("|".join(_PAST_CUES))
_REASON_PATTERN = re.compile("|".join(_REASON_OR_MODIFIER_CUES))
_QUOTE_PATTERN = re.compile("|".join(_QUOTE_OR_META_CUES))


def split_sentences(text: str) -> list[str]:
    """テキストを自然な文単位に分割する。"""
    if not text:
        return []
    raw_parts = re.split(r"(?<=[。！？!?\n])", text)
    sentences = [p.strip() for p in raw_parts if p.strip()]
    return sentences if sentences else [text.strip()]


@dataclass(frozen=True)
class DeadReferenceClassification:
    category: str  # REF_HISTORICAL, REF_ACTIONABLE, REF_QUESTION, or REF_NONE
    dead_player: str | None
    sentence: str
    addressed_player: str | None = None
    is_historical_reference: bool = False
    is_actionable_target: bool = False
    is_dead_player_question: bool = False


def classify_dead_player_sentence(
    sentence: str,
    dead_names: frozenset[str],
    living_names: frozenset[str] = frozenset(),
) -> DeadReferenceClassification:
    """単一文における死亡者への言及種別を分類する。"""
    matching_dead = [name for name in dead_names if name in sentence]
    if not matching_dead:
        return DeadReferenceClassification(
            category=REF_NONE,
            dead_player=None,
            sentence=sentence,
        )

    # 1. 文頭の呼びかけの判定 (例: "Player 4さん、", "Player 7さん、", "Player Xへ")
    initial_address_match = re.match(
        r"^[\s,、]*(Player\s+\d+)(?:さん)?(?:[、,]|に|へ)",
        sentence,
    )
    if initial_address_match:
        initial_name = initial_address_match.group(1)
        if initial_name in dead_names:
            # 死亡者への直接呼びかけ
            # 質問や説明要求が含まれていれば dead_player_question
            if re.search(
                r"[？?]|なぜ|どうして|理由|教えて|説明して|答えて|回答して|話して|聞きたい|聞いてみたい|どう思う|見解"
                r"|(?:\b(?:why|how|what|tell|explain|answer|clarify|please)\b)",
                sentence,
                re.IGNORECASE,
            ):
                return DeadReferenceClassification(
                    category=REF_QUESTION,
                    dead_player=initial_name,
                    sentence=sentence,
                    is_dead_player_question=True,
                )
        elif initial_name in living_names:
            # 生存者への直接呼びかけ（文中の死亡者名は背景の過去記録言及）
            return DeadReferenceClassification(
                category=REF_HISTORICAL,
                dead_player=matching_dead[0],
                sentence=sentence,
                addressed_player=initial_name,
                is_historical_reference=True,
            )

    # 文中の呼びかけ（例: "質問ですが、Player 7さんに聞きたいです。Player 1に投票した理由は..."）
    addressed_living: str | None = None
    for living in living_names:
        escaped_living = re.escape(living)
        if re.search(
            rf"{escaped_living}(?:さん)?(?:に|へ)[^。！？!?]{{0,32}}(?:質問|聞きたい|教えて|説明して|答えて)",
            sentence,
        ) or re.search(
            rf"^[\s,、]*{escaped_living}\b[^.!?\n]{{0,32}}(?:ask|tell|explain|what|why|how|please)",
            sentence,
            re.IGNORECASE,
        ):
            # ただし "Player 3に投票した理由を聞きたい" のように "に投票した" を挟む場合は質問相手ではない
            if not re.search(rf"{escaped_living}(?:さん)?(?:に|へ)[^。！？!?]{{0,16}}(?:投票|票を|処刑|襲撃)", sentence):
                addressed_living = living
                break

    # 生存者宛ての質問・要求文の中で死亡者名が出ている場合、死亡者は質問対象ではなく言及対象（過去記録）
    if addressed_living is not None:
        return DeadReferenceClassification(
            category=REF_HISTORICAL,
            dead_player=matching_dead[0],
            sentence=sentence,
            addressed_player=addressed_living,
            is_historical_reference=True,
        )

    # 2. 過去形・理由節・引用・メタ否定の検出
    has_past = bool(_PAST_PATTERN.search(sentence))
    has_reason = bool(_REASON_PATTERN.search(sentence))
    has_quote = bool(_QUOTE_PATTERN.search(sentence))

    # 3. 死亡者への現在の質問（DEAD_PLAYER_QUESTION）の判定
    # 死亡者への直接の呼びかけ＋要求動詞がある場合は最優先で検出
    # 例: "Player 1さん、教えてください", "Player 1さん、昨夜の占い理由を教えてください"
    for dead in matching_dead:
        escaped_dead = re.escape(dead)
        direct_address_and_ask = re.search(
            rf"{escaped_dead}(?:さん)?(?:[、,]|に|へ)[^。！？!?]{{0,48}}"
            rf"(?:教えて(?:ください|ほしい)?|説明して(?:ください|ほしい)?|答えて(?:ください|ほしい)?|回答して(?:ください|ほしい)?|話して(?:ください|ほしい)?|質問(?:したい|します|させて)|聞(?:きたい|いてみたい|きます))",
            sentence,
        ) or re.search(
            rf"{escaped_dead}(?:さん)?(?:の)?(?:基準|理由|意図|見解)(?:を)?(?:聞いてみたい|聞きたい|教えて|説明して)",
            sentence,
        ) or re.search(
            rf"^[\s,、]*{escaped_dead}\b[^.!?\n]{{0,48}}(?:please\s+)?(?:tell|explain|answer|what|why|how|clarify)",
            sentence,
            re.IGNORECASE,
        ) or re.search(
            rf"(?:ask|asking)\s+{escaped_dead}\b[^.!?\n]{{0,32}}(?:for|about|to)",
            sentence,
            re.IGNORECASE,
        )
        if direct_address_and_ask and not (has_quote and ("発言" in sentence or "言って" in sentence)):
            # "〜と発言している" などの引用文中でなければ直接質問
            return DeadReferenceClassification(
                category=REF_QUESTION,
                dead_player=dead,
                sentence=sentence,
                is_dead_player_question=True,
            )

    # 4. 現在の行動要求（ACTIONABLE_TARGET）の判定
    # 例: "Player 4に投票します", "Player 1を処刑すべきです", "今日の投票先はPlayer 4さんにします"
    actionable_found: str | None = None
    for dead in matching_dead:
        escaped_dead = re.escape(dead)
        # 過去形や理由節で修飾されている場合は現在要求ではない
        # 例: "Player 1に投票した理由" "Player 1に投票しました"
        past_action_match = re.search(
            rf"{escaped_dead}(?:さん)?(?:に|へ|を)(?:[^。！？!?]{{0,12}})(?:投票した|票を入れた|処刑した|処刑された|襲撃された|守った)",
            sentence,
        ) or re.search(
            rf"(?:voted\s+for|executed|attacked)\s+{escaped_dead}\b",
            sentence,
            re.IGNORECASE,
        )
        if past_action_match:
            continue

        # 引用文中の投票言及（例: 「Player 4に票を入れる」と発言していますが）
        if has_quote and (
            "発言" in sentence or "言って" in sentence or "検討して" in sentence
        ):
            continue

        # 現在の投票・処刑・襲撃・占いの表明
        current_action = re.search(
            rf"{escaped_dead}(?:さん)?(?:に|へ|を)(?:は)?\s*"
            rf"(?:投票(?:する|したい|します|すべき|しよう|を一票)|票を入れる(?:予定|つもり|べき|たい)?|処刑(?:する|したい|すべき|しよう)|吊(?:る|りたい|るべき))",
            sentence,
        ) or re.search(
            rf"(?:今日の)?(?:投票先|処刑対象|襲撃対象|占い対象)(?:は|として|に)\s*{escaped_dead}(?:さん)?",
            sentence,
        ) or re.search(
            rf"\bVote:\s*{escaped_dead}\b",
            sentence,
            re.IGNORECASE,
        ) or re.search(
            rf"(?:vote\s+for|execute|divine|attack)\s+{escaped_dead}\b",
            sentence,
            re.IGNORECASE,
        )
        if current_action:
            actionable_found = dead
            break

    if actionable_found:
        return DeadReferenceClassification(
            category=REF_ACTIONABLE,
            dead_player=actionable_found,
            sentence=sentence,
            is_actionable_target=True,
        )

    # 上記の現在要求・質問に該当しないものは全て過去言及（合法な考察・背景・事実言及）
    return DeadReferenceClassification(
        category=REF_HISTORICAL,
        dead_player=matching_dead[0],
        sentence=sentence,
        is_historical_reference=True,
    )


def analyze_dead_player_references(
    text: str,
    dead_names: frozenset[str],
    living_names: frozenset[str] = frozenset(),
) -> list[DeadReferenceClassification]:
    """テキスト全体の各文を走査し、死亡者言及の分類結果一覧を返す。"""
    sentences = split_sentences(text)
    classifications: list[DeadReferenceClassification] = []
    for s in sentences:
        c = classify_dead_player_sentence(s, dead_names, living_names)
        if c.dead_player is not None:
            classifications.append(c)
    return classifications
