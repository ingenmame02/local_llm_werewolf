# TASK-064 簡易確認（LL不使用）: 検出とカテゴリ、公開シリアライズの中立性
import sys

sys.path.insert(0, "src")

from werewolf.role_action_provenance import (  # noqa: E402
    CATEGORY_INFERENCE,
    CATEGORY_UNSUPPORTED_ROLE_ACTION,
    RoleActionProvenance,
    detect_ability_expressions,
    extract_spoken_verdict,
)

print(extract_spoken_verdict("私が占った結果、Player 1は狼だった。"))
print(extract_spoken_verdict("対象=Player 8 判定=人狼"))

tracker = RoleActionProvenance()
obs = tracker.observe_public_speech(
    player="Player 4",
    day=1,
    discussion_turn=3,
    text="【本文】私が占った結果、Player 1は狼だった。慎重に検討したい。",
    known_player_names=("Player 1", "Player 4"),
)
print("obs:", [(o.kind, o.supported, o.category) for o in obs])
print("metrics:", tracker.metrics())

real = RoleActionProvenance()
real.record_engine_action("Player 1", "divination", 1)
obs2 = real.observe_public_speech(
    player="Player 1",
    day=2,
    discussion_turn=1,
    text="私が占った結果、Player 8は狼だった。投票を提案する。",
    own_engine_results={"Player 8": "wolf"},
    known_player_names=("Player 1", "Player 8"),
)
print("real obs:", [(o.kind, o.supported, o.category) for o in obs2])

inf = detect_ability_expressions(
    "Player 2が占ったはずだと皆が言っている。",
    actor_name="Player 4",
    known_player_names=("Player 1", "Player 2", "Player 4"),
)
print("third:", inf)
public_blob = str(real.serialize_public())
print("public has admin keys:", any(k in public_blob for k in ("engine_supported", "category", "supported")))
print("inference counter:", tracker.category_counts[CATEGORY_INFERENCE])
print("unsupported:", tracker.category_counts[CATEGORY_UNSUPPORTED_ROLE_ACTION])
