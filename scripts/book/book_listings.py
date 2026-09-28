#!/usr/bin/env python3
"""Every code listing in the book, checked against the file it quotes.

    python3 scripts/book/book_listings.py            # one block per listing: its file, and every quoted unit the file no longer has
    python3 scripts/book/book_listings.py --summary  # one line per listing

The book quotes specs, configs, drivers, trainer bodies and renders in `verbatim` blocks and
nothing checks them against the tree (the MNIST drivers said `xVerified.train` for a month after
the code became `xVerified.toNet.train`). A listing is compared unit by unit: a unit is a line
plus its continuation lines (more indented, not starting a field, a `def`, an `import` or a
`|` arm, and not the first statement of a block a `do` / `then` / `else` / `where` opened; a
line ending in `\\` continues regardless), with whitespace removed and ASCII arrows read as the Unicode ones, so the book's re-alignment,
wrapping of `layers` lists and `->` for `→` do not count. `--` comments, blank lines, and `...` elisions are not compared. The file
is the one under LeanMlir/, apps/, jax/, Bestiary/, scripts/, tests/ or the lakefile whose collapsed text
contains the most units; a block whose best file holds at least half its units is a quotation
of that file, and each unit the file lacks is printed with the file's closest unit, which is
usually what it drifted from; a Lean block no file holds half of (a listing assembled from a spec
file and a driver, marked `*`) is checked unit by unit against every file. MLIR blocks are matched line-exact against verified_mlir/.
Blocks matching nothing (transcripts, shell sessions, pseudocode) are only counted.
"""
import bisect, difflib, os, re, sys
from collections import Counter

TEX = 'blueprint/src/content.tex'
CODE_ROOTS = ['LeanMlir', 'apps', 'jax', 'Bestiary', 'scripts', 'tests', 'lakefile.lean']
CODE_EXT = {'.lean', '.py', '.sh', '.yml', '.yaml', '.c', '.h', '.conf', '.toml'}
HEAD = re.compile(r'\\(chapter\*?|section\*?|subsection\*?|prosesection|prosesubsection)\{(.*)\}')
NEW_UNIT = re.compile(r'^\s*(def |theorem |lemma |structure |instance |import |namespace |end |open |#|\||[A-Za-z_][A-Za-z0-9_.\']*\s*:=|\.\.\.)')

ARROWS = {'->': '→', '<-': '←', '=>': '⇒'}
def squash(s):
    """the comparable form: ASCII arrows as the book prints them are the Unicode ones, whitespace does not count"""
    for a, u in ARROWS.items(): s = s.replace(a, u)
    s = re.sub(r'^\s*noncomputable\s+', '', s)
    return re.sub(r'\s+', '', s)

def units(lines):
    """merge continuation lines into their unit; drop comments, docstrings, blanks and elisions"""
    out = []; doc = False
    for l in lines:
        s = l.strip()
        if doc:
            if '-/' in s: doc = False
            continue
        if s.startswith('/-'):
            doc = '-/' not in s; continue
        if not s or s.startswith('--'): continue
        elided = '...' in s
        s = re.sub(r'\s--\s.*$', '', l.rstrip())          # trailing comment
        indent = len(l) - len(l.lstrip())
        prev_open = bool(out) and out[-1][3]                          # the previous line did not finish
        cont = out and not NEW_UNIT.match(l) and (prev_open or (indent > out[-1][0] and not out[-1][4]))
        block_start = bool(re.search(r'\b(do|then|else|where)\s*$', s))
        if cont:
            out[-1] = (out[-1][0], out[-1][1] + ' ' + s.strip(), out[-1][2] or elided, s.endswith('\\'), block_start)
        else:
            out.append((indent, s.strip(), elided, s.endswith('\\'), block_start))
    return [squash(u) for _, u, e, _, _ in out if not e and len(u) >= 6]

def walk(roots, exts):
    for root in roots:
        paths = [root] if os.path.isfile(root) else [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
        for p in paths:
            if os.path.splitext(p)[1] in exts and '/.lake/' not in p and '__pycache__' not in p:
                yield p

def load_code():
    files = {}
    for p in walk(CODE_ROOTS, CODE_EXT):
        try: src = open(p, encoding='utf-8', errors='replace').read()
        except OSError: continue
        lines = src.split('\n')
        files[p] = (' ' + ' '.join(units(lines)) + ' ', src)
    return files

def load_mlir():
    idx = {}
    for p in walk(['verified_mlir'], {'.mlir'}):
        for l in open(p, encoding='utf-8', errors='replace'):
            s = squash(l)
            if len(s) >= 12: idx.setdefault(s, set()).add(p)
    return idx

def blocks(tex_lines):
    out = []; i = 0
    while i < len(tex_lines):
        if tex_lines[i].strip() == '\\begin{verbatim}':
            j = i + 1
            while j < len(tex_lines) and tex_lines[j].strip() != '\\end{verbatim}': j += 1
            out.append((i + 1, tex_lines[i + 1:j])); i = j
        i += 1
    return out

def kind(body):
    t = '\n'.join(body)
    if re.search(r'\[pjrt_ffi\]|^\s*epoch \d|test_acc|val_acc|Epoch \d+/', t, re.M): return 'transcript'
    if 'stablehlo.' in t or 'func.func' in t: return 'mlir'
    if re.search(r'^\s*(\$ |lake |\./|LEAN_MLIR|python3 |pip |git |cd )', t, re.M): return 'shell'
    if re.search(r'^\s*(def|theorem|structure|import|noncomputable|namespace|#eval|#guard|\|)\b', t, re.M): return 'lean'
    return 'other'

def candidates(us, files):
    """files declaring a name the block declares, else every file"""
    names = {m.group(1) for u in us for m in [re.match(r'(?:noncomputable )?def ([A-Za-z_][\w.\']*)', u)] if m}
    if names:
        c = [p for p, (txt, src) in files.items() if any(f'def {n} ' in src or f'def {n}\n' in src or f'def {n}:' in src for n in names)]
        if c: return c
    return list(files)

def main():
    summary = '--summary' in sys.argv
    tex_lines = open(TEX).read().split('\n')
    heads = [(i + 1, m.group(2)) for i, l in enumerate(tex_lines) for m in [HEAD.match(l)] if m]
    head_lines = [h[0] for h in heads]
    heading = lambda n: heads[bisect.bisect_right(head_lines, n) - 1][1] if bisect.bisect_right(head_lines, n) else '(front)'
    files = load_code(); mlir = load_mlir()
    quoted = drifted = 0; kinds = Counter()
    for start, body in blocks(tex_lines):
        k = kind(body)
        if k == 'mlir':
            ls = [squash(l) for l in body if len(squash(l)) >= 12]
            votes = Counter(p for l in ls for p in mlir.get(l, ()))
            best, n = votes.most_common(1)[0] if votes else (None, 0)
            missing = [l for l in ls if best not in mlir.get(l, ())] if best else ls
            total = len(ls)
        else:
            us = units(body); total = len(us)
            best, n, missing = None, 0, us
            if us:
                for p in candidates(us, files):
                    hit = sum(1 for u in us if f' {u} ' in files[p][0])
                    if hit > n: best, n = p, hit
                if best: missing = [u for u in us if f' {u} ' not in files[best][0]]
        if (not best or n < max(1, total / 2)) and not (k == 'lean' and best):
            kinds[k] += 1
            if summary: print(f'  --  L{start:<6d} {k:10s} {total:3d} units   {heading(start)[:50]}')
            continue
        composite = n < max(1, total / 2)
        quoted += 1
        if k != 'mlir':
            missing_anywhere = [u for u in missing if not any(f' {u} ' in txt for txt, _ in files.values())]
        else:
            missing_anywhere = [u for u in missing if u not in mlir]
        drifted += bool(missing_anywhere)
        tag = f'{n}/{total}' + ('*' if composite else '')
        if summary:
            print(f'{"DRIFT" if missing_anywhere else "  ok "} L{start:<6d} {tag:>7s}  {best}   {heading(start)[:40]}'); continue
        print(f'\n=== L{start}  {heading(start)}  ->  {best}  ({tag} units found)')
        if k == 'mlir':
            pool = list(mlir)
        else:
            pool = units(files[best][1].split('\n'))
        for u in missing:
            elsewhere = [p for p, (txt, _) in files.items() if f' {u} ' in txt] if k != 'mlir' else sorted(mlir.get(u, ()))
            print(f'  book: {u[:150]}')
            if elsewhere:
                print(f'  from: {elsewhere[0]}' + (f' (+{len(elsewhere)-1})' if len(elsewhere) > 1 else '')); continue
            near = difflib.get_close_matches(u, pool, n=1, cutoff=0.6)
            print(f'  file: {(near[0][:150] if near else "(no unit like it)")}')
    print(f'\n{quoted} listings quote a file, {drifted} with units the file no longer has; '
          + ', '.join(f'{v} {k}' for k, v in kinds.most_common()) + ' matched nothing')

if __name__ == '__main__':
    main()
