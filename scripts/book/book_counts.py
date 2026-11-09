#!/usr/bin/env python3
"""The counts the book states by hand, checked against what they count.

    python3 scripts/book/book_counts.py          # exit 1 on a count the tree contradicts

Each check is one sentence pattern in blueprint/src/content.tex and the thing it counts:

  * bestiary files            "Forty-six further architectures", "bestiary's 46 architecture families"
                              against the number of Bestiary/*.lean files;
  * demo groups               "twelve of the groups carry a demo" against the `--- demo:` sections;
  * comparator                "re-runs ... over N theorems", "The N theorems split across three files", "The N are not the whole
                              proof suite", "What the N add" against the three tests/comparator/config*.json
                              name lists; "The last N are ChallengeTier.lean" and "the N-theorem tie"
                              against config-tier.json;
  * theorem budget            the three column sums of tab:theorem_budget against its totals row.

A count that moves (a new bestiary entry, a new tier theorem) now fails here instead of waiting for a
reader; fix the sentence, not the check. Word numerals up to twenty are accepted.
"""
import glob, json, re, sys

TEX = 'blueprint/src/content.tex'
WORDS = {w: i for i, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve '
         'thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
for t, v in [('thirty', 30), ('forty', 40), ('fifty', 50), ('sixty', 60), ('seventy', 70), ('eighty', 80), ('ninety', 90)]:
    WORDS[t] = v
    for w, i in list(WORDS.items())[1:10]: WORDS[f'{t}-{w}'] = v + i

def num(tok):
    tok = tok.replace('{,}', '').replace(',', '').lower()
    return int(tok) if tok.isdigit() else WORDS.get(tok)

def main():
    tex = open(TEX).read()
    bad = []
    def expect(pattern, actual, what):
        hits = re.findall(pattern, tex)
        if not hits:
            bad.append(f'{what}: no sentence matches /{pattern}/'); return
        for h in hits:
            v = num(h)
            if v != actual:
                bad.append(f'{what}: the book says {h!r}, the tree has {actual}')

    n_bestiary = len(glob.glob('Bestiary/*.lean'))
    expect(r'\\textbf\{Part 2: The bestiary \(Chapters[^)]*\)\.\} ([A-Za-z-]+) further\s+architectures', n_bestiary, 'bestiary count (Part 2 blurb)')
    expect(r"bestiary's (\d+) architecture families", n_bestiary, 'bestiary count (appendix)')

    n_demo = len(re.findall(r'^\\section\{[^\n]*--- demo:', tex, re.M))
    expect(r'and ([a-z]+) of the\s+groups carry a demo', n_demo, 'demo groups')

    cfg = {c: len(json.load(open(f'tests/comparator/{c}.json'))['theorem_names']) for c in ('config', 'config-arch', 'config-tier')}
    total = sum(cfg.values()); tier = cfg['config-tier']
    expect(r'over\s+(\d+) theorems, sandboxed', total, 'comparator total (Level 3)')
    expect(r'The (\d+) theorems split across three files', total, 'comparator total (appendix)')
    expect(r'The (\d+) are not the whole proof suite', total, 'comparator total (appendix, "are not the whole")')
    expect(r'What the (\d+) add is a second', total, 'comparator total (appendix, "what the N add")')
    expect(r'The last (\d+) are \\texttt\{ChallengeTier\.lean\}', tier, 'comparator tier')

    m = re.search(r'\\label\{tab:theorem_budget\}(.*?)\\end\{tabular\}', tex, re.S)
    if not m:
        bad.append('theorem budget: table not found')
    else:
        rows = [r for r in m.group(1).split('\\\\') if '&' in r]
        body = [r for r in rows if 'Chapter &' not in r]
        data = [[c.strip() for c in r.replace('\\hline', '').split('&')] for r in body]
        chapters = [d for d in data if d[0] and re.match(r'\d', d[0])]
        totals = [d for d in data if not d[0] and '\\textbf' in d[1]]
        def cell(c):
            c = re.sub(r'\\textbf\{([^}]*)\}', r'\1', c).strip()
            return 0 if c == '---' else int(c)
        sums = [sum(cell(d[i]) for d in chapters) for i in (1, 2, 3)]
        if not totals:
            bad.append('theorem budget: totals row not found')
        else:
            tot = [cell(totals[0][i]) for i in (1, 2, 3)]
            if sums != tot:
                bad.append(f'theorem budget: columns sum to {sums}, totals row says {tot}')
            grand = re.search(r'total \\textbf\{(\d+)\}', totals[0][4])
            if grand and int(grand.group(1)) != sum(tot):
                bad.append(f'theorem budget: grand total {grand.group(1)} but the three totals sum to {sum(tot)}')
    for pat, what in [(r'\\textbf\{(\d+) contracts\}', 'contracts'), (r'\\textbf\{(\d+) witnesses\}', 'witnesses'),
                      (r'\\textbf\{(\d+) step certificates\}', 'step certificates')]:
        h = re.search(pat, tex)
        if h and m and totals:
            idx = {'contracts': 1, 'witnesses': 2, 'step certificates': 3}[what]
            if int(h.group(1)) != cell(totals[0][idx]):
                bad.append(f'theorem budget prose: {h.group(1)} {what} vs table total {cell(totals[0][idx])}')

    if bad:
        print('\n'.join(bad)); print(f'\n{len(bad)} count(s) disagree with the tree'); sys.exit(1)
    print(f'counts agree: {n_bestiary} bestiary files, {n_demo} demo groups, comparator {cfg} = {total}, budget table sums')

if __name__ == '__main__':
    main()
