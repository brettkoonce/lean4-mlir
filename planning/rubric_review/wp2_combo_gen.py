"""WP2 P-API-2 generator (planning/rubric_review.md §WP2): builds `<net>_net_tied_lossGrad`, the
combined tie ∧ loss-gradient theorem at ONE cotangent chain, from the tie theorem and the
`*NetLossTiedB` def. Used for r34/r50/mnv2/mnv4 (drivers not kept) and B0/ConvNeXt/ViT (drivers below)."""
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

def wrap(prefix, items, suffix, ind='    ', sep=', '):
    lines = []; cur = prefix
    for k, it in enumerate(items):
        piece = it + (sep if k < len(items) - 1 else '')
        if len(cur) + len(piece) > 100:
            lines.append(cur.rstrip()); cur = ind + piece
        else:
            cur += piece
    lines.append(cur + suffix)
    return '\n'.join(lines)

def gen(tie_path, tie_name, loss_path, loss_def, name, extra_binders, loss_let, tie_call, loss_call,
        rw_loss='', doc='', pre_re=None, groups=None, pre_simp='', bridges=None, extra_haves=(),
        unfold_hl=None, extra_rw=(), extract=False, remerge=()):
    """`bridges`: `(pre_call, let_name, apply_lemma_call)` in forward order, each `pre_call = let_name`
    proved by `rw [apply_lemma_call, <previous bridge>]`; with `unfold_hl` they are rewritten into the
    loss hypothesis (after `extra_rw`, whose equations `extra_haves` prove), so the final `exact`
    compares the tie's let names on both sides instead of unfolding the `*Pre*` functions.
    `extract`: `extract_lets` the tie and the loss hypothesis, merging their lets into the goal's.
    `remerge`: the goal's chain lets (top-down) whose loss-side copies do not merge because the loss
    def abstracted a proof argument (`*NetLossTied*._proof_*`); each is equated with its copy one
    step at a time, since matching the whole chain at once exceeds `maxRecDepth`."""
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
    base = min(len(l[0]) - len(l[0].lstrip()) for l in tl if l[0].strip().startswith('let '))
    # the loss let goes after the last tie let, before any trailing comment
    last = max(k for k, l in enumerate(tl) if l[0].strip().startswith('let '))
    for k, l in enumerate(tl):
        out.extend(l)
        if k == last and loss_let:
            out.append(' ' * base + loss_let)
    for k, (ti, li) in enumerate(groups):
        items = [re.sub(r'\n\s+', '\n        ', tc[j]) for j in ti] + \
                [re.sub(r'\n\s+', '\n        ', lc[j]) for j in li]
        pre = ' ' * base if k == 0 else '  ∧ '
        out.append(f'{pre}(' + '\n      ∧ '.join(items) + ')')
    out[-1] += ' := by'
    names = lets_names(tl) + (['L'] if loss_let else [])
    out.append(wrap('  intro ', names, '', sep=' '))
    n = len(tc)
    if extract:
        out.append(f'  have htie :=\n    {tie_call}')
        out.append('  extract_lets at htie')
        out.append(wrap('  obtain ⟨', [f't{i}' for i in range(n)], '⟩ := htie'))
    else:
        out.append(wrap('  obtain ⟨', [f't{i}' for i in range(n)], '⟩ :=') + '\n    ' + tie_call)
    nl = len(lc)
    out.append(f'  have hl :=\n    {loss_call}')
    if rw_loss: out.append(f'  {rw_loss}')
    if bridges:
        out.append("  -- the loss side's activations and logits, in the tie's spelling")
        for k, (pre, nm, ap) in enumerate(bridges):
            rws = ap + (f', e{k - 1}' if k else '')
            out.append(f'  have e{k} : {pre} = {nm} := by rw [{rws}]')
    for h in extra_haves:
        out.append(f'  {h}')
    if unfold_hl:
        out.append(f'  unfold {unfold_hl} at hl')
    if bridges or extra_rw:
        eqs = list(extra_rw) + [f'e{k}' for k in reversed(range(len(bridges or [])))]
        out.append(wrap('  rw [', eqs, '] at hl'))
    if extract:
        out.append('  extract_lets at hl')
    if remerge:
        ds = [f'd_{v}' for v in remerge]
        out.append("  -- the loss def's abstracted proof arguments keep these from merging; match them stepwise")
        out.append(wrap('  rename_i ', ds, '', sep=' '))
        out.append(f'  have q_{remerge[0]} : d_{remerge[0]} = {remerge[0]} := rfl')
        for up, v in zip(remerge, remerge[1:]):
            out.append(f'  have q_{v} : d_{v} = {v} := by simp only [d_{v}, q_{up}]; rfl')
        out.append(wrap('  rw [', [f'q_{v}' for v in reversed(remerge)], '] at hl'))
    if pre_simp: out.append(f'  {pre_simp}')
    out.append(wrap('  obtain ⟨', [f'l{i}' for i in range(nl)], '⟩ := hl'))
    out.append(wrap('  exact ⟨', ['⟨' + ', '.join([f't{i}' for i in ti] + [f'l{j}' for j in li]) + '⟩'
                                    for ti, li in groups], '⟩'))
    return '\n'.join(out) + '\n'


# ── Drivers for the nets whose tie spells activations as its own lets ──────────────────────────
# `python3 wp2_combo_gen.py cnx` prints the theorem to paste at the end of the ParamGrad file.

CNX_DIR = 'LeanMlir/Proofs/Nets/ConvNeXt/'
CNX_PRE = ['S', 'B1', 'B2', 'B3', 'D0', 'B4', 'B5', 'B6', 'D1', 'B7', 'B8', 'B9', 'B10', 'B11',
           'B12', 'B13', 'B14', 'B15', 'D2', 'B16', 'B17', 'B18']
CNX_LETS = ['ib1', 'ib2', 'ib3', 'ibD0', 'ib4', 'ib5', 'ib6', 'ibD1', 'ib7', 'ib8', 'ib9', 'ib10',
            'ib11', 'ib12', 'ib13', 'ib14', 'ib15', 'ibD2', 'ib16', 'ib17', 'ib18', 'xhead']
CNX_DOC = '''/-- **The emitted ConvNeXt-T step's gradient nodes ARE the loss's gradient, at one chain.** For
    each of the 182 parameter slots, at ONE cotangent chain (the tie's own, from the emitted
    smoothed-loss cotangent `g`): the node denotes its layer's Jacobian against the chain cotangent
    (`cnx_net_tiedGB`), and the batched smoothed loss of `cnxNetB` with that one slot varied is
    differentiable there with the node as its gradient (`cnx_net_lossGrad_smoothedCE`). The tie
    spells each block input as its own let; the proof rewrites the loss side's `cnxPre*` into those
    lets (`cnxPreS_apply`, …) and the loss side's logits into the tie's (`cnx_logitsB_eq`). -/'''


def cnx():
    # every prefix past the stem runs through a block, so it takes the GELU's form `gf`
    bridges = [(f'cnxPre{p} {"" if p == "S" else "gf "}N ε w x', nm, f'cnxPre{p}_apply N ε w x')
               for p, nm in zip(CNX_PRE, CNX_LETS)]
    eh = len(bridges) - 1
    return gen(CNX_DIR + 'ConvNeXtStepTieGB.lean', 'cnx_net_tiedGB',
               CNX_DIR + 'ConvNeXtParamGrad.lean', 'CnxNetLossTiedGB', 'cnx_net_tied_lossGrad',
               '(hK : 0 < nC) (hε : 0 < ε) (ht : ∀ n, ∑ k : Fin nC, batchSlice N nC t n k = 1)',
               'let L := smoothedBatchLossDiv N nC α B t',
               'cnx_net_tiedGB N xN epsStr cotN dN aStr negAK bStr logN ohN ε α B w x t',
               'cnx_net_lossGrad_smoothedCE (gf := gf) xN epsStr cotN dN aStr negAK bStr logN ohN N hK ε α B hε w x t ht',
               doc=CNX_DOC, pre_re=r'\(cnxPre\w+ (?:gf )?N ε w x\)', bridges=bridges,
               extra_haves=['have eg : den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN\n'
                            f'      (cnxNetB gf N ε w x) t) = g := by rw [← cnx_logitsB_eq, e{eh}]'],
               unfold_hl='CnxNetLossTiedGB', extra_rw=['eg'])


ENET_DIR = 'LeanMlir/Proofs/Nets/EfficientNet/'
ENET_DOC = '''/-- **The emitted EfficientNet-B0 step's gradient nodes ARE the loss's gradient, at one chain.**
    For each of the 262 parameter slots, at ONE cotangent chain (the tie's own, from the emitted
    smoothed-loss cotangent `g`): the node denotes its layer's Jacobian against the chain cotangent
    (`efficientnet_net_tiedG`), and the batched smoothed loss of `efficientnetForwardBFull` with that
    one slot varied is differentiable there with the node as its gradient
    (`enet_net_lossGrad_smoothedCE`). The tie spells each block input as its own let; the proof
    rewrites the loss side's `enetPre*` into those lets (`enetPreB0_apply`, …) and the loss side's
    logits into the tie's (`enet_forward_eq_head`). -/'''


def enet():
    bridges = [(f'enetPreB{k} N w x', f'a{k}', f'enetPreB{k}_apply N w x') for k in range(17)]
    return gen(ENET_DIR + 'EfficientNetStepTieG.lean', 'efficientnet_net_tiedG',
               ENET_DIR + 'EfficientNetParamGrad.lean', 'EnetNetLossTiedG', 'enet_net_tied_lossGrad',
               '(hK : 0 < nCls) (ht : ∀ n, ∑ k : Fin nCls, targetRow N nCls t n k = 1)',
               'let L := smoothedBatchLoss N nCls α B t',
               'efficientnet_net_tiedG xN vN epsStr cotN dN N w hεw aStr negAK bStr logN ohN α B x t',
               'enet_net_lossGrad_smoothedCE xN vN epsStr cotN dN aStr negAK bStr logN ohN N hK α B w hεw x t ht',
               doc=ENET_DOC, pre_re=r'\(enetPreB\d+ N w x\)', bridges=bridges,
               extra_haves=['have eg : unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN\n'
                            '      (rowB N nCls (efficientnetForwardBFull N w x)) t)) = g := by\n'
                            '    rw [enet_forward_eq_head, e16]'],
               unfold_hl='EnetNetLossTiedG', extra_rw=['eg'], extract=True,
               remerge=[f'dy{k}' for k in range(15, -1, -1)])


VIT_DIR = 'LeanMlir/Proofs/Nets/ViT/'
VIT_DOC = '''/-- **The emitted ViT-Tiny step's gradient nodes ARE the loss's gradient, at one chain.** For each
    of the 200 parameter slots, at ONE cotangent chain (the tie's own, from the emitted
    smoothed-loss cotangent `g`): the node denotes its layer's Jacobian against the chain cotangent
    (`vit_net_tiedGB`), and the batched smoothed loss of `vitNetB` with that one slot varied is
    differentiable there with the node as its gradient (`vit_net_lossGrad_smoothedCE`). The tie's
    final-LN and classifier conjuncts pair with the one head conjunct of the loss side. The tie
    spells each block input as its own let; the proof rewrites the loss side's `vitPre*` into those
    lets (`vitPreE_apply`, …) and the loss side's logits into the tie's (`vit_logitsB_eq`). -/'''


def vit():
    pres = ['vitPreE N w img'] + [f'vitPreB{k} gf N ε w img' for k in range(1, 13)]
    aps = ['vitPreE_apply N w img'] + [f'vitPreB{k}_apply N ε w img' for k in range(1, 13)]
    lets = [f'ib{k}' for k in range(1, 13)] + ['b12out']
    bridges = list(zip(pres, lets, aps))
    groups = [([k], [k]) for k in range(12)] + [([12, 13], [12]), ([14], [13])]
    return gen(VIT_DIR + 'ViTStepTieGB.lean', 'vit_net_tiedGB',
               VIT_DIR + 'ViTParamGrad.lean', 'ViTNetLossTiedGB', 'vit_net_tied_lossGrad',
               '(hK : 0 < nC) (hε : 0 < ε) (ht : ∀ n, ∑ k : Fin nC, batchSlice N nC t n k = 1)',
               'let L := smoothedBatchLossDiv N nC α B t',
               'vit_net_tiedGB (gf := gf) N xN aN epsStr cotN aStr negAK bStr logN ohN ε α B w img t',
               'vit_net_lossGrad_smoothedCE (gf := gf) xN aN epsStr cotN aStr negAK bStr logN ohN N hK ε α B hε w img t ht',
               doc=VIT_DOC, pre_re=r'\(vitPre\w+ (?:gf )?N (?:ε )?w img\)', groups=groups, bridges=bridges,
               extra_haves=['have eg : den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN\n'
                            f'      (vitNetB gf N ε w img) t) = g := by rw [← vit_logitsB_eq, e{len(bridges) - 1}]'],
               unfold_hl='ViTNetLossTiedGB', extra_rw=['eg'])


if __name__ == '__main__':
    print({'cnx': cnx, 'enet': enet, 'vit': vit}[sys.argv[1]]())
