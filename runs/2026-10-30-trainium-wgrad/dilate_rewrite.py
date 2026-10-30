"""Replace materialized dilation in a verified render with the conv's own dilation attributes.
  dilate_rewrite.py IN.mlir OUT.mlir [NEWFUNC]        rewrite (and rename the entry function)
  dilate_rewrite.py check NAME                        XLA:CPU: verified_mlir/NAME.mlir vs its rewrite
Two patterns, both exact (the same nonzero products, zeros dropped):
  W  du = pad(dy, interior s-1, high u) ; dt = transpose(du) ; conv(x, dt) rhs_dilate 1, pad [lo,hi]
     →  dt = transpose(dy) ; conv(x, dt) rhs_dilate s, pad [lo, hi-u]           (weight gradients)
  I  up = pad(dy, interior s-1, high u) ; conv(up, w) lhs_dilate 1, pad [lo,hi]
     →  conv(dy, w) lhs_dilate s, pad [lo, hi+u]                                 (input gradients)
Pads whose consumer is anything else (the ResNets' nine-pad max-pool backward) are left alone."""
import re, sys

PAD = re.compile(r'^(\s*)(%[\w.]+) = stablehlo\.pad (%[\w.]+), (%[\w.]+), low = \[([\d, ]+)\], high = \[([\d, ]+)\], interior = \[([\d, ]+)\] : \(tensor<([\dx]+)xf32>, tensor<f32>\) -> tensor<([\dx]+)xf32>\s*$')
ints = lambda s: [int(x) for x in s.split(',')]
dims = lambda s: [int(x) for x in s.split('x')]
ty = lambda d: 'tensor<' + 'x'.join(map(str, d)) + 'xf32>'

def find_uses(lines, name, start, limit=60):
    pat = re.compile(re.escape(name) + r'(?![\w.])')
    return [j for j in range(start, min(start + limit, len(lines))) if pat.search(lines[j]) and not lines[j].lstrip().startswith(name + ' =')]

def window(lines, k):
    """the convolution at line k: returns (k, kw) with kw the index of its window line."""
    assert 'stablehlo.convolution(' in lines[k]
    kw = k + 2; assert 'window = {' in lines[kw], lines[kw]
    return kw

def rewrite(txt):
    lines = txt.split('\n'); n_w = n_i = 0; drop = set()
    for i, l in enumerate(lines):
        m = PAD.match(l)
        if not m: continue
        ind, res, src, _z, low, high, inter, tin, tout = m.groups()
        low, high, inter, tin, tout = ints(low), ints(high), ints(inter), dims(tin), dims(tout)
        if not any(inter): continue
        if any(low): continue                      # the max-pool backward's nine shifted pads: not ours
        assert inter[0] == inter[1] == 0 and high[0] == high[1] == 0 and len(tin) == 4, l
        sH, sW = inter[2] + 1, inter[3] + 1; uH, uW = high[2], high[3]
        uses = find_uses(lines, res, i + 1)
        if len(uses) != 1: continue
        j = uses[0]; u = lines[j]
        if 'stablehlo.transpose' in u:                                         # pattern W
            mt = re.match(r'^(\s*)(%[\w.]+) = stablehlo\.transpose (%[\w.]+), dims = \[1, 0, 2, 3\] : \(tensor<([\dx]+)xf32>\) -> tensor<([\dx]+)xf32>\s*$', u)
            assert mt and mt.group(3) == res, u
            tr = mt.group(2); uses2 = find_uses(lines, tr, j + 1)
            assert len(uses2) == 1, (res, uses2)
            k = uses2[0]; mc = re.match(r'^(\s*)(%[\w.]+) = stablehlo\.convolution\((%[\w.]+), (%[\w.]+)\)\s*$', lines[k])
            assert mc and mc.group(4) == tr, lines[k]
            kw = window(lines, k)
            if 'batch_group_count = 1 ' not in lines[kw + 1]:
                # a grouped (depthwise) weight gradient: neuronx-cc 2.28 fails rhs_dilate with
                # batch_group_count > 1 (NCC_IBIR243) at 3 of B0's 4 shapes; keep the pad there.
                continue
            mw = re.match(r'^(\s*)window = \{stride = \[1, 1\], pad = \[\[(\d+), (\d+)\], \[(\d+), (\d+)\]\], lhs_dilate = \[1, 1\], rhs_dilate = \[1, 1\]\}\s*$', lines[kw])
            assert mw, lines[kw]
            loH, hiH, loW, hiW = (int(x) for x in mw.groups()[1:])
            tnew = [tin[1], tin[0], tin[2], tin[3]]
            lines[j] = f'{mt.group(1)}{tr} = stablehlo.transpose {src}, dims = [1, 0, 2, 3] : ({ty(tin)}) -> {ty(tnew)}'
            told = [tout[1], tout[0], tout[2], tout[3]]
            kt = kw + 1; assert ty(told) in lines[kt], (lines[kt], told)
            lines[kt] = lines[kt].replace(f', {ty(told)})', f', {ty(tnew)})', 1)
            hH, hW = hiH - uH, hiW - uW
            if hH < 0 or hW < 0:
                # the materialized kernel's trailing zero rows read input rows the dilated kernel
                # no longer covers; dropping them is a slice of the (transposed) input, not a
                # negative pad — keeps the conv's padding non-negative for every backend.
                cH, cW = max(0, -hH), max(0, -hW); hH, hW = max(0, hH), max(0, hW)
                lhs = mc.group(3); ml = re.search(r': \(tensor<([\dx]+)xf32>, ', lines[kt]); tl = dims(ml.group(1))
                tls = [tl[0], tl[1], tl[2] - cH, tl[3] - cW]; sl = f'%dil{n_w}_s'
                lines[k] = f'{mc.group(1)}{sl} = stablehlo.slice {lhs} [0:{tl[0]}, 0:{tl[1]}, 0:{tls[2]}, 0:{tls[3]}] : ({ty(tl)}) -> {ty(tls)}\n' + \
                           f'{mc.group(1)}{mc.group(2)} = stablehlo.convolution({sl}, {tr})'
                lines[kt] = lines[kt].replace(f': ({ty(tl)}, ', f': ({ty(tls)}, ', 1)
            lines[kw] = f'{mw.group(1)}window = {{stride = [1, 1], pad = [[{loH}, {hH}], [{loW}, {hW}]], lhs_dilate = [1, 1], rhs_dilate = [{sH}, {sW}]}}'
            drop.add(i); n_w += 1
        elif 'stablehlo.convolution(' in u:                                     # pattern I
            mc = re.match(r'^(\s*)(%[\w.]+) = stablehlo\.convolution\((%[\w.]+), (%[\w.]+)\)\s*$', u)
            assert mc and mc.group(3) == res, u
            kw = window(lines, j)
            mw = re.match(r'^(\s*)window = \{stride = \[1, 1\], pad = \[\[(\d+), (\d+)\], \[(\d+), (\d+)\]\], lhs_dilate = \[1, 1\], rhs_dilate = \[1, 1\]\}\s*$', lines[kw])
            assert mw, lines[kw]
            loH, hiH, loW, hiW = (int(x) for x in mw.groups()[1:])
            lines[j] = f'{mc.group(1)}{mc.group(2)} = stablehlo.convolution({src}, {mc.group(4)})'
            lines[kw] = f'{mw.group(1)}window = {{stride = [1, 1], pad = [[{loH}, {hiH + uH}], [{loW}, {hiW + uW}]], lhs_dilate = [{sH}, {sW}], rhs_dilate = [1, 1]}}'
            kt = kw + 1; assert f'({ty(tout)}, ' in lines[kt], (lines[kt], tout)
            lines[kt] = lines[kt].replace(f'({ty(tout)}, ', f'({ty(tin)}, ', 1)
            drop.add(i); n_i += 1
    out = '\n'.join(l for i, l in enumerate(lines) if i not in drop)
    return out, n_w, n_i

if __name__ == '__main__':
    if sys.argv[1] == 'check':
        import numpy as np
        sys.path.insert(0, '/home/skoonce/lean/lean4-mlir/scripts/platform'); import tier2
        name = sys.argv[2]; path = tier2.ROOT / 'verified_mlir' / f'{name}.mlir'
        new, nw, ni = rewrite(path.read_text())
        print(f'{name}: {nw} weight-grad and {ni} input-grad dilations rewritten; interior pads left: {len(re.findall("interior = .0, 0, [1-9]", new))}')
        fn, ins, outs = tier2.signature(name); inputs = tier2.make_inputs(name, ins)
        tmp = tier2.ROOT / f'.dil_{name}.mlir'; tmp.write_text(new)
        try: a = tier2.run_cpu(fn, path, inputs); b = tier2.run_cpu(fn, tmp, inputs)
        finally: tmp.unlink()
        rows = []
        for i, (x, y) in enumerate(zip(a, b)):
            d = float(np.max(np.abs(x - y))); s = float(np.max(np.abs(x))) or 1e-30
            rows.append((d / s, d, i, x.shape, int(np.sum(x != y))))
        rows.sort(reverse=True)
        print(f'  outputs {len(a)}; bit-identical: {sum(1 for r in rows if r[4] == 0)}/{len(rows)}; worst:')
        for rel, d, i, sh, ne in rows[:5]: print(f'    out {i} {sh}: max abs {d:.2e} rel {rel:.1e} differing {ne}')
    else:
        src, dst = sys.argv[1], sys.argv[2]
        new, nw, ni = rewrite(open(src).read())
        if len(sys.argv) > 3:
            old = re.search(r'func\.func @(\w+)\(', new).group(1); new = new.replace(f'func.func @{old}(', f'func.func @{sys.argv[3]}(', 1)
        open(dst, 'w').write(new); print(f'{src}: W={nw} I={ni} -> {dst}')
