#!/usr/bin/env python3
import re

with open('c:\\Users\\yukic\\local_llm_werewolf\\src\\werewolf\\config.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Insert claim_coaching line after players.append(
old = '        players.append(\\n            PlayerConfig('
new = '        players.append(\\n        claim_coaching = entry.get(\"claim_coaching\", False)\\n            PlayerConfig('
content = content.replace(old, new)

with open('c:\\Users\\yukic\\local_llm_werewolf\\src\\werewolf\\config.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')