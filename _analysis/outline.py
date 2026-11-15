import re, sys
for path in sys.argv[1:]:
    src = open(path, encoding='utf-8').read()
    lines = src.splitlines()
    print(f'=== {path} lines={len(lines)} chars={len(src)}')
    for i, l in enumerate(lines, 1):
        if re.match(r'^(def |class )', l) or re.match(r'^[A-Z][A-Z0-9_]* *=', l):
            print(f'{i}: {l[:90]}')
