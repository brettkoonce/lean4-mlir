"""Dependency graphs as TikZ at the head of every theorems section, from
blueprint/src/content.tex:

    python3 scripts/blueprint_depgraph_tikz.py        # writes blueprint/src/figures/depgraph/*.tex

Re-run after `scripts/blueprint_uses.py --fix` changes an edge or a statement moves;
the files are committed (the layout needs graphviz via pygraphviz, which CI does not
have). One figure per chapter (per section when a chapter's
statements span several; a section may be cut once more at a named node), from
blueprint/src/content.tex — the print twin of templates/dep_graph.html. Within a
figure: its statements and the edges between them; a dependency outside the figure
is a dashed portal naming where it lives ("ch 4", "above", "below"); a statement used
by four or more chapters (pdiv, hasvjp) is drawn once, in chapter 1; one that feeds four or more nodes of
a figure, and the imports of a node that has four or more, are left to the
statement's own \\uses line rather than drawn (the web graph has them all). Graphviz
lays each figure out (pygraphviz); positions and splines are copied into TikZ with
the node sizes dot was given, so labels are set in the book's font and every node
is a \\hyperref to its statement."""
import re, sys
from collections import defaultdict
import pygraphviz as pgv

AMBIENT = 4
ENVS = ("theorem", "lemma", "definition", "axiom")
# A section too deep for one page is cut at these nodes: the node and everything
# downstream of it (within the section) become the second figure.
CUTS = {"Attention proofs": ["ax:mhsa_has_vjp_mat"]}
BASE = 7.0          # label font size at scale 1, points (portals 5.5)

def parse(tex):
    chapters, nodes, edges = [], {}, []
    ch, sec, num, nums = -1, '', 0, {}
    tok = re.compile(r'\\chapter(\*?)\{([^}]*)\}|\\section\*?\{([^}]*)\}|\\begin\{(%s)\}(.*?)\\end\{\4\}|\\appendix' % '|'.join(ENVS), re.S)
    for m in tok.finditer(tex):
        if m.group(0) == '\\appendix': num = -1000; continue
        if m.group(2) is not None:
            chapters.append(m.group(2)); ch = len(chapters) - 1; sec = ''
            if not m.group(1): num += 1; nums[ch] = num if num > 0 else None
            continue
        if m.group(3) is not None: sec = m.group(3); continue
        body = m.group(5)
        lab = re.search(r'\\label\{([^}]*)\}', body); ln = re.search(r'\\lean\{([^}]*)\}', body)
        if not lab or not ln or ln.group(1).strip().startswith('Layer.'): continue
        nodes[lab.group(1)] = dict(ch=ch, sec=sec, env=m.group(4))
        for u in re.findall(r'\\uses\{([^}]*)\}', body, re.S):
            for d in u.replace('\n', ' ').split(','):
                d = d.strip()
                if d: edges.append((d, lab.group(1)))
    return chapters, nums, nodes, [e for e in edges if e[0] in nodes and e[1] in nodes]

def units(nodes, edges):
    """Figure units in document order: (chapter, section-or-'', part, [node labels])."""
    secs = defaultdict(list)
    for l in nodes:
        if nodes[l]['sec'] not in secs[nodes[l]['ch']]: secs[nodes[l]['ch']].append(nodes[l]['sec'])
    out = []
    for ch in sorted(secs):
        for s in secs[ch]:
            sec = s if len(secs[ch]) > 1 else ''
            mine = [l for l in nodes if nodes[l]['ch'] == ch and nodes[l]['sec'] == s]
            if s in CUTS:
                down = set(CUTS[s]); grew = True
                while grew:
                    grew = False
                    for a, b in edges:
                        if a in down and b in mine and b not in down: down.add(b); grew = True
                out.append((ch, sec, 1, [l for l in mine if l not in down]))
                out.append((ch, sec, 2, [l for l in mine if l in down]))
            else: out.append((ch, sec, 0, mine))
    return out

def esc(s): return s.replace('_', '\\_').replace('&', '\\&')

def wrap(label, limit=20):
    """A long name becomes two lines, split at the underscore nearest its middle."""
    if len(label) <= limit or '_' not in label: return [label]
    cut = min((i for i, c in enumerate(label) if c == '_'), key=lambda i: abs(i - len(label) // 2))
    return [label[:cut + 1], label[cut + 1:]]

def figure(chapters, nums, nodes, edges, unit, allunits, rankdir='TB', textwidth_pt=460.0, textheight_pt=640.0):
    ch, sec, part, mine = unit
    mineset = set(mine)
    order = [u[:3] for u in allunits]
    unit_of = {l: u[:3] for u in allunits for l in u[3]}
    here = (ch, sec, part)
    reach = defaultdict(set)
    for a, b in edges:
        if nodes[a]['ch'] != nodes[b]['ch']: reach[a].add(nodes[b]['ch'])
    ambient = {a for a in reach if len(reach[a]) >= AMBIENT}
    def where(a):
        if nodes[a]['ch'] != ch: return 'ch %s' % nums.get(nodes[a]['ch'], '?')
        return 'above' if order.index(unit_of[a]) < order.index(here) else 'below'
    kept, uses, imports, hidden = [], defaultdict(set), defaultdict(set), defaultdict(set)
    for a, b in edges:
        if b in mineset and a in mineset: kept.append((a, b))
        elif b in mineset and a not in ambient: uses[a].add(b); imports[b].add(a)
        elif b in mineset: hidden[b].add(a)
    # Listed under the figure instead of drawn: a source feeding AMBIENT or more nodes here
    # (it would be a hub), and the imports of a node that has AMBIENT or more of them (they
    # would be a cloud around one node). Everything else is a portal.
    listed = sorted(a for a in uses if len(uses[a]) >= AMBIENT)
    heavy = {b: sorted(a for a in imports[b] if a not in listed) for b in imports if len(imports[b]) >= AMBIENT}
    portal_edges = [(a, b) for a, b in edges if b in mineset and a in uses and a not in listed and b not in heavy]
    portals = {a for a, b in portal_edges}
    # A statement whose every source is hidden by the rules above (ambient, listed or heavy)
    # would float. It hangs from the figure's one shared box instead, which names the chapters
    # its sources live in: the whole-net certificates of a chapter each stand on the same layer
    # VJPs from earlier chapters, and that is the fact to draw, not nine islands.
    for b in heavy: hidden[b] |= imports[b]
    for a in listed:
        for b in uses[a]: hidden[b].add(a)
    drawn_in = {b for a, b in kept} | {b for a, b in portal_edges}
    islands = [b for b in mine if b not in drawn_in and hidden[b] and not any(a == b for a, _ in kept)]
    shared = sorted(set().union(*(hidden[b] for b in islands))) if islands else []
    shared_where = sorted({where(a) for a in shared}, key=lambda w: (w[:2] != 'ch', w))
    shared_label = ', '.join(w for w in shared_where).replace(', ch ', ', ')
    G = pgv.AGraph(directed=True, strict=True, rankdir=rankdir, ranksep=0.35, nodesep=0.15, splines='true')
    G.node_attr.update(fontsize=1)   # sizes are fixed below; dot's own label metrics are unused
    def box(label, fs, lines=1):   # what TikZ will draw (rectangles; a rounded one costs nothing extra)
        return dict(fixedsize='true', width=(len(label) * 0.6 * fs + 6) / 72.0, height=(fs * 1.25 * lines + 4) / 72.0)
    for l in mine:
        ls = wrap(l.split(':', 1)[1])
        G.add_node(l, shape='box', **box(max(ls, key=len), BASE, lines=len(ls)))
    for a in sorted(portals):
        G.add_node('portal|' + a, shape='box', **box(max(a.split(':', 1)[1], where(a), key=len), 5.5, lines=2))
    if shared: G.add_node('shared', shape='box', **box(max('%d statements of' % len(shared), shared_label, key=len), 5.5, lines=2))
    outdeg = defaultdict(int)
    for a, b in kept: outdeg[a] += 1
    fan = defaultdict(int)
    for a, b in kept:   # unflatten: a source with many children spreads them over up to four ranks,
        if outdeg[a] >= 5:   # so a top-down figure is a few nodes wide instead of one rank of fourteen
            depth = min(4, (outdeg[a] + 3) // 4)
            G.add_edge(a, b, minlen=1 + fan[a] % depth); fan[a] += 1
        else: G.add_edge(a, b)
    for a, b in portal_edges: G.add_edge('portal|' + a, b, style='dashed')
    for b in islands: G.add_edge('shared', b, style='dashed')
    G.layout('dot')
    pts = []
    for n in G.nodes():
        x, y = map(float, n.attr['pos'].split(',')); w, h = float(n.attr['width']) * 72, float(n.attr['height']) * 72
        pts += [(x - w / 2, y - h / 2), (x + w / 2, y + h / 2)]
    for e in G.edges():
        for p in e.attr['pos'].split():
            x, y = map(float, p.split(',')[-2:]); pts.append((x, y))
    x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts); x1, y1 = max(p[0] for p in pts), max(p[1] for p in pts)
    s_up = min(1.0, textwidth_pt / (x1 - x0), textheight_pt / (y1 - y0))
    s_rot = min(1.0, textheight_pt / (x1 - x0), textwidth_pt / (y1 - y0))
    rotate = s_rot > 1.5 * s_up
    s = s_rot if rotate else s_up
    P = lambda p: '(%.1fpt,%.1fpt)' % ((float(p.split(',')[-2]) - x0) * s, (float(p.split(',')[-1]) - y0) * s)
    fs = lambda base: '\\fontsize{%.2f}{%.2f}\\selectfont' % (base * s, base * 1.2 * s)
    out = ['\\begin{tikzpicture}[x=1pt,y=1pt, every node/.style={inner sep=%.2fpt, font=%s\\ttfamily}]' % (1.5 * s, fs(BASE))]
    for n in G.nodes():
        name = n.get_name(); w, h = float(n.attr['width']) * 72 * s, float(n.attr['height']) * 72 * s
        if name == 'shared':
            out.append('  \\node[draw=gray!70, dashed, rounded corners=2pt, minimum width=%.1fpt, minimum height=%.1fpt, text=gray!60!black, align=center, font=%s\\ttfamily] at %s {%d statements of\\\\%s};'
                       % (w, h, fs(5.5), P(n.attr['pos']), len(shared), esc(shared_label)))
            continue
        if name.startswith('portal|'):
            src = name[7:]
            out.append('  \\node[draw=gray!70, dashed, rounded corners=2pt, minimum width=%.1fpt, minimum height=%.1fpt, text=gray!60!black, align=center, font=%s\\ttfamily] at %s {\\hyperref[%s]{%s}\\\\(%s)};'
                       % (w, h, fs(5.5), P(n.attr['pos']), src, esc(src.split(':', 1)[1]), esc(where(src))))
        else:   # definitions square-cornered, theorems rounded — the web graph's box/ellipse split
            shape = 'rectangle' if nodes[name]['env'] == 'definition' else 'rectangle, rounded corners=%.1fpt' % (4 * s)
            # one \hyperref per line: a line break inside the link text breaks TikZ's align
            label = '\\\\'.join('\\hyperref[%s]{%s}' % (name, esc(x)) for x in wrap(name.split(':', 1)[1]))
            out.append('  \\node[draw=green!50!black, fill=green!15, %s, align=center, minimum width=%.1fpt, minimum height=%.1fpt%s] at %s {%s};'
                       % (shape, w, h, ', double' if name in ambient else '', P(n.attr['pos']), label))
    for e in G.edges():
        p = e.attr['pos'].split(); end = p.pop(0)[2:] if p[0].startswith('e,') else None
        segs = [P(p[0])] + ['.. controls %s and %s .. %s' % (P(p[i]), P(p[i + 1]), P(p[i + 2])) for i in range(1, len(p) - 2, 3)]
        style = 'dashed, gray!60' if e.attr.get('style') == 'dashed' else 'black!70'
        out.append('  \\draw[->, >=stealth, %s, line width=%.2fpt] %s%s;' % (style, 0.4 * max(s, 0.6), ' '.join(segs), (' -- ' + P(end)) if end else ''))
    out.append('\\end{tikzpicture}')
    body = '\n'.join(out)
    if rotate: body = '\\rotatebox{-90}{%\n' + body + '}'
    W, H = ((y1 - y0) * s, (x1 - x0) * s) if rotate else ((x1 - x0) * s, (y1 - y0) * s)
    return body, W, H, (listed, heavy, islands, shared), BASE * s

if __name__ == '__main__':
    chapters, nums, nodes, edges = parse(open('blueprint/src/content.tex').read())
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'blueprint/src/figures/depgraph'
    import os; os.makedirs(outdir, exist_ok=True)
    k = 0; alls = units(nodes, edges)
    slug = lambda s: re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')
    for u in alls:
        # Top-down (sources upper left, the capstone lower right) whenever it reads at 5.5 pt or
        # better; left-right only for a figure that would otherwise be too small.
        cands = {rd: figure(chapters, nums, nodes, edges, u, alls, rankdir=rd) + (rd,) for rd in ('TB', 'LR')}
        tikz, w, h, (listed, heavy, islands, shared), font, rd = cands['TB'] if cands['TB'][4] >= 5.5 else max(cands.values(), key=lambda r: r[4])
        k += 1
        fname = 'ch%s' % nums.get(u[0]) + ('_' + slug(u[1]) if u[1] else '') + ('_%d' % u[2] if u[2] else '')
        short = lambda x: '\\hyperref[%s]{\\texttt{%s}}' % (x, esc(x.split(':', 1)[1]))
        open('%s/%s.tex' % (outdir, fname), 'w').write('\\begin{center}\n' + tikz + '\n\\end{center}\n')
        print('%-26s ch %2s %-24s %-18s part %d %s %5.0f x %5.0f pt  font %.1f%s  listed: %s  heavy: %s  shared: %d -> %s' % (fname, nums.get(u[0]), chapters[u[0]][:24], u[1][:18], u[2], rd, w, h, font, '  (rotated)' if tikz.startswith('\\rotatebox') else '', ', '.join(x.split(':', 1)[1] for x in listed) or '-', ', '.join('%s(%d)' % (b.split(':',1)[1], len(v)) for b, v in heavy.items()) or '-', len(shared), ', '.join(b.split(':', 1)[1] for b in islands) or '-'))
