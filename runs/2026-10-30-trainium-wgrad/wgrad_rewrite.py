"""Rewrite the ViT patch-embedding weight gradient in a verified render.
  wgrad_rewrite.py {B|C} IN.mlir OUT.mlir        write the rewritten render
  wgrad_rewrite.py check {B|C} NAME              XLA:CPU, original vs rewritten (tier2 inputs)
Form B: drop the interior-15 pad and give the existing weight-grad conv rhs_dilate = [16,16] (the same
        sum with the zero terms removed; XLA's own gradient of a strided conv).
Form C: the same gradient as one matmul over patches — reshape x to [B·14·14, 3·16·16] and dy to
        [B·14·14, 192], contract the patch axis (what an im2col patch-embed does).
The pattern is the three-op tail: pad(interior) → transpose(dy) → convolution(xT, dyT)."""
import re, sys

PAT = re.compile(
    r'^(?P<ind> *)(?P<pad>%[\w.]+) = stablehlo\.pad (?P<dy>%[\w.]+), (?P<z>%[\w.]+), low = \[0, 0, 0, 0\], high = \[0, 0, 0, 0\], interior = \[0, 0, (?P<s1>\d+), (?P<s1b>\d+)\] : \(tensor<(?P<B>\d+)x(?P<O>\d+)x(?P<h>\d+)x(?P<w>\d+)xf32>, tensor<f32>\) -> tensor<(?P<B2>\d+)x(?P<O2>\d+)x(?P<H>\d+)x(?P<W>\d+)xf32>\n'
    r' *(?P<xt>%[\w.]+) = stablehlo\.transpose (?P<x>%[\w.]+), dims = \[1, 0, 2, 3\] : \(tensor<\d+x(?P<C>\d+)x(?P<X>\d+)x(?P<Xw>\d+)xf32>\) -> tensor<[\dx]+xf32>\n'
    r' *(?P<pt>%[\w.]+) = stablehlo\.transpose (?P=pad), dims = \[1, 0, 2, 3\] : \(tensor<[\dx]+xf32>\) -> tensor<[\dx]+xf32>\n'
    r' *(?P<c>%[\w.]+) = stablehlo\.convolution\((?P=xt), (?P=pt)\)\n'
    r' *dim_numbers = \[b, f, 0, 1\]x\[o, i, 0, 1\]->\[b, f, 0, 1\],\n'
    r' *window = \{stride = \[1, 1\], pad = \[\[0, 0\], \[0, 0\]\], lhs_dilate = \[1, 1\], rhs_dilate = \[1, 1\]\}\n'
    r' *\{batch_group_count = 1 : i64, feature_group_count = 1 : i64\} : \(tensor<[\dx]+xf32>, tensor<[\dx]+xf32>\) -> tensor<(?P<C3>\d+)x(?P<O3>\d+)x(?P<P>\d+)x(?P<Pw>\d+)xf32>\n'
    r' *(?P<r>%[\w.]+) = stablehlo\.transpose (?P=c), dims = \[1, 0, 2, 3\] : \(tensor<[\dx]+xf32>\) -> tensor<(?P<T>[\dx]+)xf32>\n',
    re.M)

def rewrite(txt, form):
    k = 0
    def one(m):
        nonlocal k; k += 1
        B, O, h, w, C, X, P = (int(m[g]) for g in 'BOhwCXP'); s = int(m['s1']) + 1
        assert s == P and h * P == X and m['T'] == f'{O}x{C}x{P}x{P}', (s, P, h, X, m['T'])
        ind = m['ind']; x, dy, r = m['x'], m['dy'], m['r']
        n = lambda t: f'%nwg{k}_{t}'
        if form == 'B':
            o = [f'{n("xt")} = stablehlo.transpose {x}, dims = [1, 0, 2, 3] : (tensor<{B}x{C}x{X}x{X}xf32>) -> tensor<{C}x{B}x{X}x{X}xf32>',
                 f'{n("dt")} = stablehlo.transpose {dy}, dims = [1, 0, 2, 3] : (tensor<{B}x{O}x{h}x{w}xf32>) -> tensor<{O}x{B}x{h}x{w}xf32>',
                 f'{n("c")} = stablehlo.convolution({n("xt")}, {n("dt")})',
                 f'  dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],',
                 f'  window = {{stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [{s}, {s}]}}',
                 f'  {{batch_group_count = 1 : i64, feature_group_count = 1 : i64}} : (tensor<{C}x{B}x{X}x{X}xf32>, tensor<{O}x{B}x{h}x{w}xf32>) -> tensor<{C}x{O}x{P}x{P}xf32>',
                 f'{r} = stablehlo.transpose {n("c")}, dims = [1, 0, 2, 3] : (tensor<{C}x{O}x{P}x{P}xf32>) -> tensor<{O}x{C}x{P}x{P}xf32>']
        else:
            N = B * h * w; K = C * P * P
            o = [f'{n("x6")} = stablehlo.reshape {x} : (tensor<{B}x{C}x{X}x{X}xf32>) -> tensor<{B}x{C}x{h}x{P}x{w}x{P}xf32>',
                 f'{n("xt")} = stablehlo.transpose {n("x6")}, dims = [0, 2, 4, 1, 3, 5] : (tensor<{B}x{C}x{h}x{P}x{w}x{P}xf32>) -> tensor<{B}x{h}x{w}x{C}x{P}x{P}xf32>',
                 f'{n("xp")} = stablehlo.reshape {n("xt")} : (tensor<{B}x{h}x{w}x{C}x{P}x{P}xf32>) -> tensor<{N}x{K}xf32>',
                 f'{n("dt")} = stablehlo.transpose {dy}, dims = [0, 2, 3, 1] : (tensor<{B}x{O}x{h}x{w}xf32>) -> tensor<{B}x{h}x{w}x{O}xf32>',
                 f'{n("dp")} = stablehlo.reshape {n("dt")} : (tensor<{B}x{h}x{w}x{O}xf32>) -> tensor<{N}x{O}xf32>',
                 f'{n("g")} = stablehlo.dot_general {n("dp")}, {n("xp")}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<{N}x{O}xf32>, tensor<{N}x{K}xf32>) -> tensor<{O}x{K}xf32>',
                 f'{r} = stablehlo.reshape {n("g")} : (tensor<{O}x{K}xf32>) -> tensor<{O}x{C}x{P}x{P}xf32>']
        return ''.join(ind + l + '\n' for l in o)
    assert '%nwg' not in txt
    return PAT.sub(one, txt), k

if __name__ == '__main__':
    if sys.argv[1] == 'check':
        import numpy as np, pathlib
        sys.path.insert(0, '/home/skoonce/lean/lean4-mlir/scripts/platform'); import tier2
        form, name = sys.argv[2], sys.argv[3]
        path = tier2.ROOT / 'verified_mlir' / f'{name}.mlir'
        new, k = rewrite(path.read_text(), form)
        print(f'{name} form {form}: rewrote {k} weight-grad conv(s); 209x209 left: {new.count("209x209")}')
        fn, ins, outs = tier2.signature(name)
        inputs = tier2.make_inputs(name, ins)
        tmp = tier2.ROOT / f'.wgrad_{form}_tmp.mlir'; tmp.write_text(new)
        try:
            a = tier2.run_cpu(fn, path, inputs); b = tier2.run_cpu(fn, tmp, inputs)
        finally:
            tmp.unlink()
        worst = []
        for i, (x, y) in enumerate(zip(a, b)):
            if x.shape != y.shape: print('SHAPE MISMATCH', i, x.shape, y.shape); continue
            d = float(np.max(np.abs(x - y))); s = float(np.max(np.abs(x))) or 1e-30
            worst.append((d / s, d, i, x.shape, int(np.sum(x != y))))
        worst.sort(reverse=True)
        print(f'  outputs {len(a)}; bit-identical outputs: {sum(1 for w in worst if w[4] == 0)}/{len(worst)}')
        for rel, d, i, sh, ne in worst[:4]: print(f'  out {i} shape {sh}: max abs diff {d:.3e} rel {rel:.1e} differing elems {ne}')
    else:
        form, src, dst = sys.argv[1:4]
        new, k = rewrite(open(src).read(), form)
        open(dst, 'w').write(new); print(f'rewrote {k} -> {dst}')
