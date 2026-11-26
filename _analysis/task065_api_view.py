"""TASK-065テスト作成前のAPI確認（読み取り専用・LL不使用）。UTF-8で結果を書く。"""
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "werewolf"
OUT = ROOT / "_analysis" / "task065_api_view.txt"

CHECKS = [
    ("cli.py", "--context-integrity"),
    ("cli.py", "--timeline-intent"),
    ("cli.py", "--player-name-resolution"),
    ("cli.py", "--output-stabilization-scope"),
    ("cli.py", "preflight"),
    ("models.py", "context_integrity_checks"),
    ("models.py", "timeline_intent_policy"),
    ("models.py", "player_name_resolution_policy"),
    ("models.py", "output_stabilization_scope"),
    ("llm_actions.py", "def _stability_record"),
    ("llm_actions.py", "def run_context_preflight"),
    ("llm_actions.py", "relax("),
    ("llm_actions.py", "apply_downgrade("),
    ("llm_actions.py", "model_wide_evidence("),
    ("engine.py", "name_resolution"),
    ("prompts.py", "name_resolution"),
    ("continuity_state.py", "name_resolution"),
]


def main() -> None:
    chunks: list[str] = []
    for name, needle in CHECKS:
        lines = io.open(SRC / name, encoding="utf-8").read().splitlines()
        hits = [
            f"{i + 1}: {line.strip()[:150]}"
            for i, line in enumerate(lines)
            if needle in line
        ]
        chunks.append(f"--- {name} :: {needle} hits={len(hits)}")
        chunks.extend(hits[:14])
    OUT.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
