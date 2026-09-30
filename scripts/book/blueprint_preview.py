#!/usr/bin/env python3
"""Review a book change as a rendered diff: the committed blueprint against the working tree.

    python3 scripts/book/blueprint_preview.py                  # build both, write /tmp/blueprint_preview
    python3 scripts/book/blueprint_preview.py --serve 8765     # ... and serve it (Ctrl-C stops)
    python3 scripts/book/blueprint_preview.py --no-build       # re-diff builds already there

`current/` is blueprint/src at --base (HEAD, or any git ref), `proposed/` is the working tree. Each
is built as web (plastex: the interactive dep graph works, TikZ figures are print-only) and as PDF
(latexmk + xelatex), the four builds in parallel, about four minutes. Then diff.html: the two PDFs'
text sentence by sentence (a unit is cut at a sentence end or before a heading, caption, listing
or bullet, so its boundaries do not move with the page break) with word-level <del>/<ins>, in page
order, every block linking the page in both PDFs. Filtered out as not-a-change: theorem /
definition / section renumbering ("Theorem 12" -> "Theorem 13", "10.3.2", "§10.3.5"), page numbers
and pgfplots tick labels, text that only moved (a section that changed chapters prints once, as the
edits inside it, not as a wall of green), hyphenation and ligature reflow. index.html links the
diff, both webs, both dep graphs and both PDFs.

Needs plastex + leanblueprint (pip), latexmk / xelatex, pdftotext (poppler) and git. Nothing under
blueprint/ is written: both trees are copied out first (a copy must keep print.tex AND
macros/print.tex; the PDF build needs both).
"""
import argparse, difflib, html, os, re, shutil, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path

# ── the two builds ────────────────────────────────────────────────────────────

SKIP = shutil.ignore_patterns('*.aux', '*.fdb_latexmk', '*.fls', '*.log', '*.out', '*.pdf', '*.toc', '*.xdv', '*.paux')

def export(base, dst):
    """blueprint/src at git ref `base` -> dst (a fresh copy)."""
    if dst.exists(): shutil.rmtree(dst)
    with tempfile.TemporaryDirectory() as tmp:
        tar = subprocess.run(['git', 'archive', base, 'blueprint/src'], capture_output=True, check=True).stdout
        subprocess.run(['tar', '-x', '-C', tmp], input=tar, check=True)
        shutil.copytree(Path(tmp) / 'blueprint' / 'src', dst, ignore=SKIP)

def copy_worktree(dst):
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree('blueprint/src', dst, ignore=SKIP)

def start_build(src, out):
    """web + PDF of one source tree, both in the background; returns [(name, Popen)]."""
    procs = []
    for name, cmd in (('web', ['plastex', '-c', 'plastex.cfg', '--imager=none', '--vector-imager=none',
                               '--dir=%s' % (out / 'web'), 'web.tex']),
                      ('pdf', ['latexmk', '-xelatex', '-interaction=nonstopmode',
                               '-output-directory=%s' % (out / 'pdf'), 'print.tex'])):
        (out / name).mkdir(parents=True, exist_ok=True)
        log = open(out / ('%s.log' % name), 'w')
        procs.append((name, subprocess.Popen(cmd, cwd=src, stdout=log, stderr=subprocess.STDOUT)))
    return procs

# ── the rendered diff ─────────────────────────────────────────────────────────

LIG = str.maketrans({'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl', '’': "'", '“': '"', '”': '"'})
KW = {'Theorem', 'Theorems', 'Definition', 'Definitions', 'Lemma', 'Lemmas', 'Figure', 'Figures',
      'Table', 'Tables', 'Chapter', 'Section', 'Appendix', 'Corollary'}
ALNUM = r'[^0-9A-Za-zͰ-Ͽ∀-⋿\U0001d400-\U0001d7ff]'   # keeps Greek, operators, math italics

def key(tok):
    """comparison key of one token: hyphenation, soft hyphens and ligatures do not count"""
    return re.sub(r'[-­]', '', tok.translate(LIG))

def keys(ws):
    """comparison keys of a token list; a numeral inside a "Theorem 12" / "Theorems 3 and 4" /
    "(§ 125)" run is masked, because renumbering is not an edit"""
    out, run = [], False
    for n, t in enumerate(ws):
        bare = re.sub(r'[^\w]', '', re.sub(r"[’']s$", '', t))
        if (n == 0 and re.fullmatch(r'\d+(\.\d+)+', t)) or re.fullmatch(r'\d+\.\d+\.\d+', t):
            out.append('#'); continue                                          # a heading's number
        if '§' in t: out.append(key(re.sub(r'\d+(\.\d+)*', '#', t))); run = True; continue   # "§10.3.5"
        if re.sub(r'[^A-Za-z§]', '', t) in KW or '§' in t:
            run = True; out.append(key(t)); continue
        if run and (re.fullmatch(r'\d+', bare) or bare in ('and', 'to', '')):
            out.append('#' if bare and bare[0].isdigit() else key(t)); continue
        run = False; out.append(key(t))
    return out

def norm(s): return re.sub(ALNUM, '', key(s))

# a unit ends at a sentence end, or where a heading, caption, listing or bullet begins
CUT = re.compile(r'(?<=[.!?])\s+|\s+(?=\d+\.\d+(?:\.\d+)?\s+[A-Z]|(?:Chapter|Appendix)\s+[A-Z0-9]+\s|'
                 r'(?:Figure|Table)\s+\d+\.\d+:|def\s+\w+\s+:\s+NetSpec|•)')

def paras(pdf):
    """[(page, unit)] of a PDF: every page's text (bare page numbers dropped) joined, then cut at
    sentence ends, so a unit's boundaries do not move with a page break or a figure's gap the way
    pdftotext's paragraphs do; dot leaders collapse to one ellipsis"""
    txt = subprocess.run(['pdftotext', str(pdf), '-'], capture_output=True, text=True, check=True).stdout
    pages = []
    for pno, page in enumerate(txt.split('\f'), 1):
        page = re.sub(r'(^|\n)\s*\d+\s*(\n|$)', r'\1\2', page)     # page-number lines
        page = re.sub(r'\n\d+\s*$', '', page.rstrip())                # a page number glued to the last line
        page = re.sub(r'(\s*\.){3,}', ' … ', page)
        page = re.sub(r'\s+', ' ', page).strip()
        if page: pages.append((pno, page))
    out, carry, cpage = [], '', 1
    for pno, page in pages:
        text = (carry + ' ' + page).strip() if carry else page
        units = re.split(CUT, text)
        carry = units.pop()                                            # the last unit may run on to the next page
        out.extend((cpage if n == 0 and carry != text else pno, u) for n, u in enumerate(units) if len(u) > 1)
        if not units: cpage = cpage if carry else pno
        else: cpage = pno
    if len(carry) > 1: out.append((cpage, carry))
    return out

def wdiff(a, b):
    """word diff of two paragraphs -> (html, changed-token count, deleted tokens, inserted tokens)"""
    aw, bw = a.split(' '), b.split(' ')
    s = difflib.SequenceMatcher(None, keys(aw), keys(bw), autojunk=False)
    out, changed, D, I = [], 0, [], []
    for op, i1, i2, j1, j2 in s.get_opcodes():
        if op == 'equal': out.append(html.escape(' '.join(aw[i1:i2]))); continue
        changed += (i2 - i1) + (j2 - j1); D += aw[i1:i2]; I += bw[j1:j2]
        if op in ('delete', 'replace'): out.append('<del>%s</del>' % html.escape(' '.join(aw[i1:i2])))
        if op in ('insert', 'replace'): out.append('<ins>%s</ins>' % html.escape(' '.join(bw[j1:j2])))
    dn, inn = norm(' '.join(D)), norm(' '.join(I))
    numeric = lambda x: re.fullmatch(r'[\d.,]*', x) is not None
    if dn == inn or (numeric(dn) and numeric(inn) and (not dn or not inn)): changed = 0      # reflow; a tick label
    if all(re.fullmatch(r'[\d.,()–-]*', t) for t in D + I) and len(D) >= 3 and len(I) >= 3: changed = 0   # a column of numbers
    return ' '.join(out), changed, D, I

def diff_html(cur_pdf, pro_pdf, out, title):
    A, B = paras(cur_pdf), paras(pro_pdf)
    ka = [' '.join(keys(p.split(' '))) for _, p in A]; kb = [' '.join(keys(p.split(' '))) for _, p in B]
    ops = [o for o in difflib.SequenceMatcher(None, ka, kb, autojunk=False).get_opcodes() if o[0] != 'equal']
    # a paragraph that only moved -- out of one run of opcodes and verbatim into another, as every
    # paragraph of a section that changed chapters does -- is not an edit: drop it from both sides
    # before pairing what is left, else the matcher anchors on the moved block and prints the text
    # it jumped over as one green wall (and a run it could not pair at all was never printed)
    da = Counter(norm(ka[i]) for _, i1, i2, _, _ in ops for i in range(i1, i2))
    db = Counter(norm(kb[j]) for _, _, _, j1, j2 in ops for j in range(j1, j2))
    moved = Counter({k: min(da[k], db[k]) for k in da if k in db})
    seenA, seenB = Counter(), Counter()
    def live(K, lo, hi, seen):
        """the indices in [lo, hi) that did not merely move"""
        kept = []
        for i in range(lo, hi):
            k = norm(K[i])
            if seen[k] < moved[k]: seen[k] += 1
            else: kept.append(i)
        return kept
    blocks, skipped = [], sum(moved.values())
    numlist = lambda x: re.fullmatch(r'[\d\s.,–-]*', x) is not None and len(re.findall(r'\d+', x)) >= 3
    def emit(op, ia, jb, pa, pb):
        """one block from the live paragraphs A[ia] / B[jb]; pa, pb the pages when a side is empty"""
        nonlocal skipped
        a = ' '.join(A[i][1] for i in ia); b = ' '.join(B[j][1] for j in jb)
        if ia: pa = A[ia[0]][0]
        if jb: pb = B[jb[0]][0]
        if op == 'replace' and norm(a) == norm(b): skipped += 1; return       # reflow / hyphenation only
        if re.fullmatch(r'[\d\s.,]*', a) and re.fullmatch(r'[\d\s.,]*', b): skipped += 1; return
        if numlist(a) and numlist(b): skipped += 1; return
        if op == 'replace':
            body, changed, _, _ = wdiff(a, b)
            if changed == 0: skipped += 1; return
        elif op == 'delete': body = '<del>%s</del>' % html.escape(a)
        else: body = '<ins>%s</ins>' % html.escape(b)
        blocks.append((pb, pa, body))
    runs = [(op, i1, i2, j1, j2, live(ka, i1, i2, seenA), live(kb, j1, j2, seenB)) for op, i1, i2, j1, j2 in ops]
    LA = [i for *_, la, _ in runs for i in la]; LB = [j for *_, _, lb in runs for j in lb]
    dropA, dropB = set(), set()
    def pairs(ia, jb):
        """a replace run of several paragraphs, paired one to one in order by similarity; the rest
        are inserts and deletes (one word diff over the joined runs interleaves unrelated text)"""
        out, y0 = [], 0
        for i in ia:
            best = None
            for y in range(y0, len(jb)):
                r = difflib.SequenceMatcher(None, A[i][1].split(), B[jb[y]][1].split(), autojunk=False).ratio()
                if r >= 0.5 and (best is None or r > best[1]): best = (y, r)
            if best is None: out.append(('delete', [i], [])); continue
            if best[0] > y0: out.append(('insert', [], jb[y0:best[0]]))
            out.append(('replace', [i], [jb[best[0]]])); y0 = best[0] + 1
        if y0 < len(jb): out.append(('insert', [], jb[y0:]))
        return out
    for op, i1, i2, j1, j2, la, lb in runs:
        la = [i for i in la if i not in dropA]; lb = [j for j in lb if j not in dropB]
        pa = A[i1][0] if i1 < len(A) else A[-1][0]; pb = B[j1][0] if j1 < len(B) else B[-1][0]
        sub = difflib.SequenceMatcher(None, [ka[i] for i in la], [kb[j] for j in lb], autojunk=False)
        for sop, x1, x2, y1, y2 in sub.get_opcodes():
            if sop == 'equal': continue
            na = sum(len(A[i][1]) for i in la[x1:x2]); nb = sum(len(B[j][1]) for j in lb[y1:y2])
            if sop == 'replace' and max(na, nb) > 1.5 * min(na, nb) + 200:
                for pop, ia, jb in pairs(la[x1:x2], lb[y1:y2]): emit(pop, ia, jb, pa, pb)
            else: emit(sop, la[x1:x2], lb[y1:y2], pa, pb)
    blocks.sort(key=lambda t: t[:2])
    css = ("body{font:15px/1.55 Georgia,serif;max-width:60em;margin:2em auto;padding:0 1em;color:#222}"
           "del{background:#fdd;color:#900;text-decoration:line-through}ins{background:#dfd;color:#060;text-decoration:none}"
           ".b{margin:1.4em 0;padding:.6em 1em;border-left:4px solid #ccc}.p{font:13px system-ui;color:#666;margin-bottom:.3em}"
           "h1{font:22px system-ui}a{color:#06c}")
    page = ['<!doctype html><meta charset="utf-8"><title>%s — rendered diff</title><style>%s</style>' % (html.escape(title), css),
            '<h1>Rendered text diff — %s</h1>' % html.escape(title),
            '<p>%d changed paragraphs in page order (%d renumbering / reflow / moved blocks filtered). '
            '<del>removed</del> <ins>added</ins>. Page links open the proposed PDF, “cur” the current one. '
            '<a href="index.html">index</a></p>' % (len(blocks), skipped)]
    for pb, pa, body in blocks:
        page.append('<div class="b"><div class="p"><a href="proposed/pdf/print.pdf#page=%d">p.%d</a> '
                    '(<a href="current/pdf/print.pdf#page=%d">cur p.%d</a>)</div>%s</div>' % (pb, pb, pa, pa, body))
    (out / 'diff.html').write_text('\n'.join(page))
    return len(blocks), skipped

def index_html(out, title):
    row = lambda name, d: ('<tr><td><b>%s</b></td><td><a href="%s/web/index.html">web</a></td>'
                           '<td><a href="%s/web/dep_graph_document.html">dep graph</a></td>'
                           '<td><a href="%s/pdf/print.pdf">print.pdf</a></td></tr>' % (name, d, d, d))
    (out / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>%s</title>'
        '<style>body{font:16px/1.5 system-ui;max-width:52em;margin:3em auto;padding:0 1em}'
        'td,th{padding:.4em 1em;border-bottom:1px solid #ccc;text-align:left}</style>'
        '<h1>%s</h1><p style="font-size:1.2em"><b><a href="diff.html">▶ rendered diff, old → new, in page order</a></b></p>'
        '<table><tr><th></th><th>web</th><th>dep graph</th><th>PDF</th></tr>%s%s</table>'
        % (html.escape(title), html.escape(title), row('current', 'current'), row('proposed', 'proposed')))

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--base', default='HEAD', help='git ref for current/ (default HEAD)')
    ap.add_argument('--out', default='/tmp/blueprint_preview', help='site directory (default /tmp/blueprint_preview)')
    ap.add_argument('--no-build', action='store_true', help='reuse the builds under --out, only rewrite the diff')
    ap.add_argument('--serve', type=int, metavar='PORT', help='serve --out on this port when done (Ctrl-C stops)')
    args = ap.parse_args()
    out = Path(args.out).resolve()
    sha = subprocess.run(['git', 'rev-parse', '--short', args.base], capture_output=True, text=True, check=True).stdout.strip()
    title = 'blueprint: %s (%s) → working tree' % (args.base, sha)
    if not args.no_build:
        export(args.base, out / 'src' / 'current'); copy_worktree(out / 'src' / 'proposed')
        for d in ('current', 'proposed'):
            for sub in ('web', 'pdf'): shutil.rmtree(out / d / sub, ignore_errors=True)
        procs = [(d, n, p) for d in ('current', 'proposed') for n, p in start_build(out / 'src' / d, out / d)]
        print('building current (%s) and proposed (working tree), web + PDF each, in parallel ...' % sha, flush=True)
        failed = [(d, n) for d, n, p in procs if p.wait() != 0]
        if failed:
            for d, n in failed: print('FAILED: %s %s — see %s' % (d, n, out / d / ('%s.log' % n)), file=sys.stderr)
            sys.exit(1)
    for d in ('current', 'proposed'):
        if not (out / d / 'pdf' / 'print.pdf').exists(): sys.exit('no %s/pdf/print.pdf under %s (build first)' % (d, out))
    n, skipped = diff_html(out / 'current' / 'pdf' / 'print.pdf', out / 'proposed' / 'pdf' / 'print.pdf', out, title)
    index_html(out, title)
    print('%s: %d changed paragraphs (%d filtered) -> %s' % (title, n, skipped, out / 'diff.html'))
    if args.serve:
        ip = subprocess.run(['sh', '-c', "ip -4 addr show tailscale0 2>/dev/null | grep -oE 'inet [0-9.]+' | cut -d' ' -f2"],
                            capture_output=True, text=True).stdout.strip() or 'localhost'
        print('serving http://%s:%d/  (Ctrl-C stops)' % (ip, args.serve), flush=True)
        os.chdir(out)
        subprocess.run([sys.executable, '-m', 'http.server', str(args.serve), '--bind', '0.0.0.0'])

if __name__ == '__main__':
    main()
