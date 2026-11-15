with open('c:\\Users\\yukic\\local_llm_werewolf\\src\\werewolf\\config.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the position after 'players.append(' and before 'PlayerConfig('
# and insert the claim_coaching line
old_text = '''        players.append(
            PlayerConfig(
                name=name,'''

new_text = '''        players.append(
        claim_coaching = entry.get("claim_coaching", False)
            PlayerConfig(
                name=name,'''

if old_text in content:
    content = content.replace(old_text, new_text)
    with open('c:\\Users\\yukic\\local_llm_werewolf\\src\\werewolf\\config.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Success!')
else:
    print('old_text not found')
    # Let's debug - show what's around players.append
    idx = content.find('players.append(')
    if idx >= 0:
        print('Found players.append at', idx)
        print('Context:', repr(content[idx:idx+100]))