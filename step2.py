with open('c:\\Users\\yukic\\local_llm_werewolf\\src\\werewolf\\config.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = '            )\n        players.append('
new = '            )\n        claim_coaching = entry.get("claim_coaching", False)\n        players.append('

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
        print('Context:', repr(content[idx-5:idx+50]))