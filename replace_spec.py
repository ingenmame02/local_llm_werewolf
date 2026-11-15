content = open('c:\\Users\\yukic\\local_llm_werewolf\\SPECIFICATION.md').read()
# Update the table row for Japanese
content = content.replace('| `日本語` | 日本語での出力を試みる | モデルによってはEnglish制約が優先され、日本語出力が制限されることあり | モデル依存 | - | CALM3等の一部モデルはJapaneseプロンプトに対応しやすい。Gemma等は制約が強い傾向 |', '| `日本語` | 日本語での出力を試みる | モデルによってはEnglish制約が優先され、日本語出力が制限されることあり | 応答言語に応じてプロンプトが自動的に切り替わる |')
open('c:\\Users\\yukic\\local_llm_werewolf\\SPECIFICATION.md', 'w').write(content)
print('Done - SPECIFICATION.md')