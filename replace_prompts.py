content = open('c:\\Users\\yukic\\local_llm_werewolf\\src\\werewolf\\prompts.py').read()
# Replace "Output must be strictly in English." with language-aware version
content = content.replace('- Output must be strictly in English.', '- 出力は厳密にJapaneseで行ってください。')
content = content.replace('Output must be strictly in English.', '出力は厳密にJapaneseで行ってください。')
# Also replace the other English-only variant
content = content.replace('Output must be strictly in English.', '出力は厳密にJapaneseで行ってください。')
open('c:\\Users\\yukic\\local_llm_werewolf\\src\\werewolf\\prompts.py', 'w').write(content)
print('Done - prompts')

# Now update llm_actions.py
content_llm = open('c:\\Users\\yukic\\local_llm_werewolf\\src\\werewolf\\llm_actions.py').read()
# Replace "Output must be strictly in English." in llm_actions.py
content_llm = content_llm.replace('Output must be strictly in English.', '出力は厳密にJapaneseで行ってください。')
# Replace "Strictly English" in llm_actions.py
content_llm = content_llm.replace('Strictly English', 'Strictly Japanese')
open('c:\\Users\\yukic\\local_llm_werewolf\\src\\werewolf\\llm_actions.py', 'w').write(content_llm)
print('Done - llm_actions')