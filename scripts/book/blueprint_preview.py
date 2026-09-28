#!/usr/bin/env python3
"""Review a book change as a rendered diff: the committed blueprint against the working tree.

    python3 scripts/book/blueprint_preview.py                  # build both, write /tmp/blueprint_preview
    python3 scripts/book/blueprint_preview.py --serve 8765     # ... and serve it (Ctrl-C stops)
    python3 scripts/book/blueprint_preview.py --no-build       # re-diff builds already there

`current/` is blueprint/src at --base (HEAD, or any git ref), `proposed/` is the working tree. Each
is built as web (plastex: the interactive dep graph works, TikZ figures are print-only) and as PDF
(latexmk + xelatex), the four builds in parallel, about four minutes. Then diff.html: the two PDFs'
text paragraph by paragraph with word-level <del>/<ins>, in page order, every block linking the
page in both PDFs. Filtered out as not-a-change: theorem / definition / section renumbering
("Theorem 12" -> "Theorem 13"), page numbers and pgfplots tick labels, a paragraph that only
moved across a page break, hyphenation and ligature reflow. index.html links the diff, both webs,
both dep graphs and both PDFs.

Needs plastex + leanblueprint (pip), latexmk / xelatex, pdftotext (poppler) and git. Nothing under
blueprint/ is written: both trees are copied out first (a copy must keep print.tex AND
macros/print.tex; the PDF build needs both).
"""
import argparse, difflib, html, os, re, shutil, subprocess, sys, tempfile
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
    for t in ws:
        bare = re.sub(r'[^\w]', '', re.sub(r"[’']s$", '', t))
        if re.sub(r'[^A-Za-z§]', '', t) in KW or '§' in t:
            run = True; out.append(key(t)); continue
        if run and (re.fullmatch(r'\d+', bare) or bare in ('and', 'to', '')):
            out.append('#' if bare and bare[0].isdigit() else key(t)); continue
        run = False; out.append(key(t))
    return out

def norm(s): return re.sub(ALNUM, '', key(s))

def paras(pdf):
    """[(page, paragraph)] of a PDF, bare page numbers dropped"""
    txt = subprocess.run(['pdftotext', str(pdf), '-'], capture_output=True, text=True, check=True).stdout
    out = []
    for pno, page in enumerate(txt.split('\f'), 1):
        page = re.sub(r'(^|\n)\s*\d+\s*(\n|$)', r'\1\2', page)     # page-number lines
        page = re.sub(r'\n\d+\s*$', '', page.rstrip())                # a page number glued to the last line
        for p in re.split(r'\n\s*\n', page):
            p = re.sub(r'\s+', ' ', p).strip()
            if len(p) > 1: out.append((pno, p))
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
    raw = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ka, kb, autojunk=False).get_opcodes():
        if op == 'equal': continue
        pa = A[i1][0] if i1 < len(A) else A[-1][0]; pb = B[j1][0] if j1 < len(B) else B[-1][0]
        raw.append((op, pa, pb, ' '.join(p for _, p in A[i1:i2]), ' '.join(p for _, p in B[j1:j2])))
    dels = {}
    for r in raw:
        if r[0] == 'delete': dels.setdefault(norm(r[3]), []).append(r)
    blocks, skipped = [], 0
    numlist = lambda x: re.fullmatch(r'[\d\s.,–-]*', x) is not None and len(re.findall(r'\d+', x)) >= 3
    for op, pa, pb, a, b in raw:
        if op == 'delete' and norm(a) in dels: continue                       # paired below, or dropped as moved
        if op == 'insert' and dels.get(norm(b)): dels[norm(b)].pop(); skipped += 1; continue
        if op == 'replace' and norm(a) == norm(b): skipped += 1; continue      # reflow / hyphenation only
        if re.fullmatch(r'[\d\s.,]*', a) and re.fullmatch(r'[\d\s.,]*', b): skipped += 1; continue
        if numlist(a) and numlist(b): skipped += 1; continue
        if op == 'replace':
            body, changed, D, I = wdiff(a, b)
            if changed == 0: skipped += 1; continue
        elif op == 'delete': body, D, I = '<del>%s</del>' % html.escape(a), a.split(' '), []
        else: body, D, I = '<ins>%s</ins>' % html.escape(b), [], b.split(' ')
        blocks.append((pb, pa, body, norm(' '.join(D)), norm(' '.join(I))))
    # a paragraph that only moved: its whole deleted text is another block's whole inserted text
    Dn = {d for _, _, _, d, i in blocks if d and not i}; In = {i for _, _, _, d, i in blocks if i and not d}
    keep = [b for b in blocks if not ((b[3] and not b[4] and b[3] in In) or (b[4] and not b[3] and b[4] in Dn))]
    skipped += len(blocks) - len(keep)
    keep.sort()
    css = ("body{font:15px/1.55 Georgia,serif;max-width:60em;margin:2em auto;padding:0 1em;color:#222}"
           "del{background:#fdd;color:#900;text-decoration:line-through}ins{background:#dfd;color:#060;text-decoration:none}"
           ".b{margin:1.4em 0;padding:.6em 1em;border-left:4px solid #ccc}.p{font:13px system-ui;color:#666;margin-bottom:.3em}"
           "h1{font:22px system-ui}a{color:#06c}")
    page = ['<!doctype html><meta charset="utf-8"><title>%s — rendered diff</title><style>%s</style>' % (html.escape(title), css),
            '<h1>Rendered text diff — %s</h1>' % html.escape(title),
            '<p>%d changed paragraphs in page order (%d renumbering / reflow / moved blocks filtered). '
            '<del>removed</del> <ins>added</ins>. Page links open the proposed PDF, “cur” the current one. '
            '<a href="index.html">index</a></p>' % (len(keep), skipped)]
    for pb, pa, body, _, _ in keep:
        page.append('<div class="b"><div class="p"><a href="proposed/pdf/print.pdf#page=%d">p.%d</a> '
                    '(<a href="current/pdf/print.pdf#page=%d">cur p.%d</a>)</div>%s</div>' % (pb, pb, pa, pa, body))
    (out / 'diff.html').write_text('\n'.join(page))
    return len(keep), skipped

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
