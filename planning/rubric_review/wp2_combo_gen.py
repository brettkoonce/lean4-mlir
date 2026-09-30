"""WP2 P-API-2 generator (planning/rubric_review.md §WP2): builds `<net>_net_tied_lossGrad`, the
combined tie ∧ loss-gradient theorem at ONE cotangent chain, from the tie theorem and the
`*NetLossTiedB` def. Used for r34/r50/mnv2/mnv4; B0/ConvNeXt/ViT are parked (see the plan)."""
import re, sys

def grab(path, head, end):
    s = open(path).read()
    i = s.index(head)
    j = s.index(end, i)
    return s[i:j]

def split_stmt(text, colon_re):
    m = re.search(colon_re, text)
    binders = text[:m.start()]
    lines = text[m.end():].split('\n')
    lets = []; i = 0; base = None
    while i < len(lines):
        ln = lines[i]; st = ln.strip(); ind = len(ln) - len(ln.lstrip())
        if st == '': i += 1; continue
        if st.startswith('let '):
            if base is None: base = ind
            lets.append([ln]); i += 1; continue
        if st.startswith('--'):
            lets.append([ln]); i += 1; continue
        if base is not None and ind > base and lets:
            lets[-1].append(ln); i += 1; continue
        break
    rest = '\n'.join(lines[i:])
    parts = re.split(r'\n\s*∧ ', rest)
    conj = [p.strip() for p in parts if p.strip()]
    return binders, lets, conj

def lets_names(lets):
    names = []
    for l in lets:
        m = re.match(r'\s*let (\w+)', l[0])
        if m: names.append(m.group(1))
    return names

def wrap(prefix, items, suffix, ind='    '):
    lines = []; cur = prefix
    for k, it in enumerate(items):
        piece = it + (', ' if k < len(items) - 1 else '')
        if len(cur) + len(piece) > 100:
            lines.append(cur.rstrip()); cur = ind + piece
        else:
            cur += piece
    lines.append(cur + suffix)
    return '\n'.join(lines)

def gen(tie_path, tie_name, loss_path, loss_def, name, extra_binders, loss_let, tie_call, loss_call,
        rw_loss='', doc='', pre_re=None, groups=None, pre_simp=''):
    tie = grab(tie_path, f'theorem {tie_name} ', ':= by')
    tb, tl, tc = split_stmt(tie, r'\s:\n')
    ld = grab(loss_path, f'def {loss_def} ', '\n\n')
    lb, ll, lc = split_stmt(ld, r'\s: Prop :=\n')
    if groups is None:
        assert len(tc) == len(lc), (name, len(tc), len(lc))
        groups = [([k], [k]) for k in range(len(tc))]
    if pre_re:
        for ti, li in groups:
            act = tc[ti[0]].split()[-2]
            for j in li:
                lc[j] = re.sub(pre_re, act, lc[j])
    binders = tb.replace(f'theorem {tie_name} ', f'theorem {name} ', 1).rstrip() + '\n    ' + extra_binders
    out = []
    out.append(doc)
    out.append(binders + ' :')
    for l in tl:
        out.extend(l)
    base = min(len(l[0]) - len(l[0].lstrip()) for l in tl if l[0].strip().startswith('let '))
    if loss_let:
        out.append(' ' * base + loss_let)
    for k, (ti, li) in enumerate(groups):
        items = [re.sub(r'\n\s+', '\n        ', tc[j]) for j in ti] + \
                [re.sub(r'\n\s+', '\n        ', lc[j]) for j in li]
        pre = ' ' * base if k == 0 else '  ∧ '
        out.append(f'{pre}(' + '\n      ∧ '.join(items) + ')')
    out[-1] += ' := by'
    names = lets_names(tl) + (['L'] if loss_let else [])
    out.append('  intro ' + ' '.join(names))
    n = len(tc)
    out.append(wrap('  obtain ⟨', [f't{i}' for i in range(n)], '⟩ :=') + '\n    ' + tie_call)
    nl = len(lc)
    out.append(f'  have hl :=\n    {loss_call}')
    if rw_loss: out.append(f'  {rw_loss}')
    if pre_simp: out.append(f'  {pre_simp}')
    out.append(wrap('  obtain ⟨', [f'l{i}' for i in range(nl)], '⟩ := hl'))
    out.append(wrap('  exact ⟨', ['⟨' + ', '.join([f't{i}' for i in ti] + [f'l{j}' for j in li]) + '⟩'
                                    for ti, li in groups], '⟩'))
    return '\n'.join(out) + '\n'
