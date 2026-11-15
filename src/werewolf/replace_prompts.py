content = open('src/werewolf/prompts.py').read()
# Replace 'Output must be strictly in English' with dynamic version
content = content.replace('Output must be strictly in English', 'Output must be strictly in {response_language.lower()}')
# Replace 'Do not use Japanese' with dynamic version  
content = content.replace('Do not use Japanese', 'Do not use {response_language.lower()}')
# Replace 'strictly in English.' with dynamic version
content = content.replace('strictly in English.', 'strictly in {response_language.lower()}.')
# Replace 'strictly in English' with dynamic version (when followed by other text)
content = content.replace('strictly in English', 'strictly in {response_language.lower()}')
open('src/werewolf/prompts.py', 'w').write(content)
print('Done')