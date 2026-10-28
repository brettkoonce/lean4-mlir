#!/usr/bin/env python3
r"""Every code listing in the book, checked against the file it quotes.

    python3 scripts/book/book_listings.py            # one block per listing: its file, and every quoted unit the file no longer has
    python3 scripts/book/book_listings.py --summary  # one line per listing
    python3 scripts/book/book_listings.py --check    # CI: exit 1 on a drifted listing not in the historical list,
                                                     #     or on a listing line over --max-cols=88 characters

A listing the book deliberately quotes at a past commit ("its recipe as the run trained it") is
named in scripts/gates/book_listings_historical.txt as `<section heading> | <file>`; `--check`
reports it but does not fail on it. Everything else that drifts fails the gate.

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
file and a driver, marked `*`) is checked unit by unit against every file. An MLIR block is an
excerpt of one committed verified_mlir/ artifact, compared op by op: an op is its logical line
(an SSA definition, `return`, a region line, with the continuation lines the artifact or the book
wraps it over), whitespace removed, the default `precision = [DEFAULT, DEFAULT]` dropped and
`//` comments skipped, and every op the book prints must be in the artifact the block quotes,
which is the `verified\_mlir/<name>.mlir` the prose names between the previous heading and the
block (else the artifact holding most of its ops).
An MLIR block matching no artifact fails the gate: the book has no illustrative StableHLO.
Other blocks matching nothing (transcripts, shell sessions, pseudocode) are only counted.
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

MLIR_NEW = re.compile(r'^\s*(%\w+\s*=|return\b|stablehlo\.return\b|\^|\}|func\.func|module|//)')
def mlir_ops(lines):
    """the comparable ops of an MLIR text: logical lines joined, `//` comments and `...` dropped, precision default dropped, squashed"""
    out = []
    for l in lines:
        s = l.rstrip()
        if not s.strip(): continue
        if MLIR_NEW.match(s) or not out: out.append(s.strip())
        else: out[-1] += ' ' + s.strip()
    ops = []
    for o in out:
        if o.startswith('//') or o.strip() == '...': continue
        o = re.sub(r'\s//.*$', '', o).replace(', precision = [DEFAULT, DEFAULT]', '')
        q = squash(o)
        if len(q) >= 12: ops.append(q)
    return ops

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
        text = s.strip(); backslash = text.endswith('\\')
        if backslash: text = text[:-1].rstrip()          # a `\`-continued string: the break itself does not count
        if cont:
            out[-1] = (out[-1][0], out[-1][1] + ' ' + text, out[-1][2] or elided, backslash, block_start)
        else:
            out.append((indent, text, elided, backslash, block_start))
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
        for op in mlir_ops(open(p, encoding='utf-8', errors='replace').read().split('\n')):
            idx.setdefault(op, set()).add(p)
    return idx

NAMED = re.compile(r'verified\\_mlir/([A-Za-z0-9_\\]+)\.mlir')
def named_artifact(tex_lines, start):
    r"""the `verified\_mlir/<name>.mlir` the prose names between the previous heading and the block, if one"""
    for i in range(start - 1, max(-1, start - 40), -1):
        if HEAD.match(tex_lines[i]): break
        m = NAMED.findall(tex_lines[i])
        if m: return 'verified_mlir/' + m[-1].replace('\\_', '_') + '.mlir'
    return None

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

HISTORICAL = 'scripts/gates/book_listings_historical.txt'

def historical():
    if not os.path.exists(HISTORICAL): return set()
    out = set()
    for l in open(HISTORICAL):
        l = l.strip()
        if not l or l.startswith('#'): continue
        h, _, f = l.partition('|')
        out.add((h.strip(), f.strip()))
    return out

def main():
    summary = '--summary' in sys.argv
    check = '--check' in sys.argv
    allowed = historical(); failures = []
    tex_lines = open(TEX).read().split('\n')
    heads = [(i + 1, m.group(2)) for i, l in enumerate(tex_lines) for m in [HEAD.match(l)] if m]
    head_lines = [h[0] for h in heads]
    heading = lambda n: heads[bisect.bisect_right(head_lines, n) - 1][1] if bisect.bisect_right(head_lines, n) else '(front)'
    files = load_code(); mlir = load_mlir()
    quoted = drifted = 0; kinds = Counter()
    for start, body in blocks(tex_lines):
        k = kind(body)
        if k == 'mlir':
            ls = mlir_ops(body)
            votes = Counter(p for l in ls for p in mlir.get(l, ()))
            best, n = votes.most_common(1)[0] if votes else (None, 0)
            named = named_artifact(tex_lines, start)          # the file the introducing sentence names wins
            if named in votes: best, n = named, votes[named]
            elif named: best, n = named, 0
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
            if k == 'mlir':
                failures.append((start, heading(start), f'no verified_mlir/ artifact holds half of its {total} ops'))
                if check or summary: print(f'DRIFT L{start:<6d} {n}/{total:<4d}  (no artifact)   {heading(start)[:40]}')
            if summary: print(f'  --  L{start:<6d} {k:10s} {total:3d} units   {heading(start)[:50]}')
            continue
        composite = n < max(1, total / 2)
        quoted += 1
        if k != 'mlir':
            missing_anywhere = [u for u in missing if not any(f' {u} ' in txt for txt, _ in files.values())]
        else:
            missing_anywhere = missing          # an excerpt is of one artifact: every op must be in it
        drifted += bool(missing_anywhere)
        if missing_anywhere and (heading(start), best) not in allowed:
            failures.append((start, heading(start), best))
        tag = f'{n}/{total}' + ('*' if composite else '')
        if check:
            if missing_anywhere:
                print(f'{"hist." if (heading(start), best) in allowed else "DRIFT"} L{start:<6d} {tag:>7s}  {best}   {heading(start)[:40]}')
            continue
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
    # the column limit: a code line in the printed page holds about 88 characters, and a longer one
    # runs into the margin (the config listings grow with every recipe pass, so this is a gate).
    # Reported after the drift report, so a run with both problems shows both.
    max_cols = next((int(a.split('=', 1)[1]) for a in sys.argv if a.startswith('--max-cols=')), 88)
    long_lines = [(start + 1 + k, len(l)) for start, body in blocks(tex_lines)
                  for k, l in enumerate(body) if len(l) > max_cols]
    if check:
        if failures:
            print(f'{len(failures)} listing(s) drifted from the tree and are not listed in {HISTORICAL}:')
            for start, h, best in failures: print(f'  L{start}  {h}  ({best})')
        else:
            print('every listing matches its file (or is listed as historical)')
    if long_lines:
        print(f'{len(long_lines)} listing line(s) over {max_cols} characters:')
        for n, w in long_lines: print(f'  L{n}  {w} chars')
    else:
        print(f'no listing line over {max_cols} characters')
    if check and (failures or long_lines): sys.exit(1)

if __name__ == '__main__':
    main()
