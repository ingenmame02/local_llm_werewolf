"""TASK-065残作業の切り分け: 名前正規化と安定化スコープの消費箇所を洗い出す（読み取り専用）。"""
import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "werewolf"
OUT = ROOT / "_analysis" / "task065_consumers.txt"

PATTERNS = {
    "consumers_of_name_resolution": r"resolve_known_names|names_in_text|scan_references|shares_mentioned_players|has_unresolved_reference|variant_mention_count|mention_labels",
    "stabilization_flags_in_cli": r"stabilization|stability",
    "unknown_name_checks_in_llm": r"unknown_player_reference|_sentence_has_unresolved|mentioned_players|def _question_answer",
    "ledger_entry_points_engine": r"register_topic|topic_ledger\.|hypothesis_ledger\.|claim_ledger\.|fact_ledger\.",
    "directed_to_actor_prompts": r"_is_directed_to_player|directed_to_actor|pending_speaker",
}


def main() -> None:
    chunks: list[str] = []
    for label, pattern in PATTERNS.items():
        rx = re.compile(pattern)
        chunks.append(f"===== {label} =====")
        for path in sorted(SRC.glob("*.py")):
            lines = io.open(path, encoding="utf-8").read().splitlines()
            hits = [
                f"  {path.name}:{i + 1}: {line.strip()[:130]}"
                for i, line in enumerate(lines)
                if rx.search(line)
            ]
            if hits:
                chunks.extend(hits[:24])
    OUT.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
