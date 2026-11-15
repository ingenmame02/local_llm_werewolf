with open('c:\\Users\\yukic\\local_llm_werewolf\\src\\werewolf\\config.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = '        players.append(\n            PlayerConfig('
new = '        players.append(\n        claim_coaching = entry.get("claim_coaching", False)\n            PlayerConfig('

if old in content:
    content = content.replace(old, new)
    with open('c:\\Users\\yukic\\local_llm_werewolf\\src\\werewolf\\config.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Success!')
else:
    print('old not found')
    idx = content.find('players.append')
    if idx >= 0:
        print('Found at index', idx)
        print('Context:', repr(content[idx:idx+50]))