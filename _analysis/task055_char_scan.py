"""TASK-055で触れたファイルに簡体字混入が無いか機械的に検査する（一時確認用）。"""
import sys
from pathlib import Path

REPO = Path(r"c:\Users\yukic\local_llm_werewolf")
sys.path.insert(0, str(REPO / "src"))

SIMPLIFIED = set("实分离模块调剂抢们结论语发对让进展关线标准产认识记录许达边临众")
FILES = [
    "src/werewolf/role_provenance.py",
    "src/werewolf/models.py",
    "src/werewolf/engine.py",
    "src/werewolf/prompts.py",
    "src/werewolf/cli.py",
    "src/werewolf/llm_actions.py",
    "tests/test_task055_role_claim_provenance.py",
    "README.md",
    "TASKS.md",
]

total = 0
for rel in FILES:
    text = (REPO / rel).read_text(encoding="utf-8")
    hits = [
        (number, [char for char in SIMPLIFIED if char in line])
        for number, line in enumerate(text.splitlines(), 1)
        if any(char in SIMPLIFIED for char in line)
    ]
    # 「分離」等は日本語でも使うので混同字は除外判定（実/模块等の明らかな誤字のみ数える）
    real = [(n, [c for c in chars if c not in {"分", "离", "发", "让", "关", "线", "许", "边", "众"}]) for n, chars in hits]
    real = [(n, chars) for n, chars in real if chars]
    total += len(real)
    print(rel, "->", real[:3] if real else "clean")
print("TOTAL", total)
