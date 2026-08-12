#!/usr/bin/env python3
"""Chapter prose metrics, per chapter_makeover.md §3 -- with its two documented
over-counting bugs fixed:
  1. enumerate/itemize are stripped, so a lead-in plus its list items is not
     counted as one sentence.
  2. `.)` is normalised to `).` before splitting, so a sentence ending in a
     parenthetical does not swallow the next one.
Semicolons use (?<!\\); so LaTeX math spacing \; is not counted as prose.
"""
import re, sys

PATH = 'blueprint/src/content.tex'
HEAD = sys.argv[1] if len(sys.argv) > 1 else r'\chapter{EfficientNet}'

t = open(PATH).read()
parts = [x for x in re.split(r'\n(?=\\chapter)', t) if x.startswith(HEAD)]
if not parts:
    sys.exit(f'chapter not found: {HEAD}')
p = parts[0]

# Out of scope: formal bodies and verbatim/figure environments.
# ▶ Replaced with a full stop, NOT deleted: a stripped block otherwise welds the
# sentence before it to the sentence after it, which is what inflates max-sentence.
b = re.sub(r'\\begin\{(theorem|proof|definition|lemma|verbatim|tabular'
           r'|tikzpicture|axis|align\*?)\}.*?\\end\{\1\}', ' . ', p, flags=re.S)

em   = b.count('---')
semi = len(re.findall(r'(?<!\\);', b))
print(f'{HEAD}\n  lines {len(p.splitlines())}  em-dash {em}  prose-semicolon {semi}')

if em or semi:
    print('  --- sites ---')
    for i, l in enumerate(b.split('\n')):
        if '---' in l or re.search(r'(?<!\\);', l):
            print(f'   {l.strip()[:110]}')

# Max sentence, with the §3 corrections applied.
s = re.sub(r'\\begin\{(enumerate|itemize|center)\}.*?\\end\{\1\}', ' . ', b, flags=re.S)
s = re.sub(r'\\(sub)*section\*?\{[^}]*\}', '', s)
s = re.sub(r'\\prosesection\{[^}]*\}', '', s)
s = s.replace('.)', ').')
# Same class of false-join as `.)`: a sentence ending in a closing quote or
# brace does not fire the split regex either, so it swallows the next one.
s = s.replace(".''", "''.").replace('.}', '}.')
s = re.sub(r'\s+', ' ', s)
best = (0, '')
for sent in re.split(r'(?<=[.!?])\s+', s):
    n = len(sent.split())
    if n > best[0]:
        best = (n, sent)
print(f'  max sentence {best[0]} w')
if best[0] > 60:
    print(f'   > {best[1][:400]}')
