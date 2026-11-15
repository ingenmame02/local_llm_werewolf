content = open('c:\\Users\\yukic\\local_llm_werewolf\\src\\werewolf\\cli.py').read()
# Add --language argument after --thinking-by-action
content = content.replace(
    'parser.add_argument(\n        \"--log-dir\",',
    'parser.add_argument(\n        \"--language\",\n        choices=[\"English\", \"Japanese\"],\n        default=None,\n        help=\"Language for player responses (English or Japanese)\",\n    ),\n    parser.add_argument(\n        \"--log-dir\",'
)
open('c:\\Users\\yukic\\local_llm_werewolf\\src\\werewolf\\cli.py', 'w').write(content)
print('Done - cli.py')