"""Prototype of the Neuron-only rewrite: each 2x2/s2 GE/add select_and_scatter becomes four
strided slices, a row-major first-max mask, four interior pads summed (maxPool3s2BackText's
recipe at 2x2). Usage: sas_rewrite.py check NAME   (XLA:CPU, original vs rewritten)."""
import re, sys

SAS = re.compile(
    r'^(?P<ind> *)(?P<res>%[\w.]+) = "stablehlo\.select_and_scatter"\((?P<x>%[\w.]+), (?P<dy>%[\w.]+), (?P<init>%[\w.]+)\) \(\{\n'
    r'.*?\}\) \{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>\}'
    r' : \(tensor<(?P<B>\d+)x(?P<C>\d+)x(?P<H>\d+)x(?P<W>\d+)xf32>, tensor<\d+x\d+x\d+x\d+xf32>, tensor<f32>\) -> tensor<[\dx]+xf32>\n',
    re.S | re.M)


def rewrite(txt):
    k = 0
    def one(m):
        nonlocal k
        k += 1
        B, C, H, W = (int(m[g]) for g in 'BCHW'); h, w = H // 2, W // 2
        T = f'tensor<{B}x{C}x{H}x{W}xf32>'; t = f'tensor<{B}x{C}x{h}x{w}xf32>'; ti = f'tensor<{B}x{C}x{h}x{w}xi1>'
        n = lambda s: f'%nsas{k}_{s}'
        o = [f'{n("z")} = stablehlo.constant dense<0.0> : tensor<f32>',
             f'{n("zs")} = stablehlo.constant dense<0.0> : {t}']
        offs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i, j in offs:
            o.append(f'{n(f"s{i}{j}")} = stablehlo.slice {m["x"]} [0:{B}, 0:{C}, {i}:{H}:2, {j}:{W}:2] : ({T}) -> {t}')
        o.append(f'{n("m1")} = stablehlo.maximum {n("s00")}, {n("s01")} : {t}')
        o.append(f'{n("m2")} = stablehlo.maximum {n("m1")}, {n("s10")} : {t}')
        o.append(f'{n("m")} = stablehlo.maximum {n("m2")}, {n("s11")} : {t}')
        # Back to full resolution through matmuls with 0/1 matrices, not pads: Neuron miscompiles an
        # interleave (pads summed, or a broadcast upsample) that feeds a transposed conv.
        o += [f'{n("ih")} = stablehlo.iota dim = 0 : tensor<{h}x{H}xi32>',
              f'{n("iH")} = stablehlo.iota dim = 1 : tensor<{h}x{H}xi32>',
              f'{n("one")} = stablehlo.constant dense<1> : tensor<{h}x{H}xi32>',
              f'{n("iH2")} = stablehlo.shift_right_logical {n("iH")}, {n("one")} : tensor<{h}x{H}xi32>',
              f'{n("Ub")} = stablehlo.compare EQ, {n("ih")}, {n("iH2")} : (tensor<{h}x{H}xi32>, tensor<{h}x{H}xi32>) -> tensor<{h}x{H}xi1>',
              f'{n("o1")} = stablehlo.constant dense<1.0> : tensor<{h}x{H}xf32>',
              f'{n("o0")} = stablehlo.constant dense<0.0> : tensor<{h}x{H}xf32>',
              f'{n("U")} = stablehlo.select {n("Ub")}, {n("o1")}, {n("o0")} : tensor<{h}x{H}xi1>, tensor<{h}x{H}xf32>',
              f'{n("pr")} = stablehlo.iota dim = 2 : tensor<{B}x{C}x{H}x{W}xi32>',
              f'{n("pc")} = stablehlo.iota dim = 3 : tensor<{B}x{C}x{H}x{W}xi32>',
              f'{n("one4")} = stablehlo.constant dense<1> : tensor<{B}x{C}x{H}x{W}xi32>',
              f'{n("prb")} = stablehlo.and {n("pr")}, {n("one4")} : tensor<{B}x{C}x{H}x{W}xi32>',
              f'{n("pcb")} = stablehlo.and {n("pc")}, {n("one4")} : tensor<{B}x{C}x{H}x{W}xi32>',
              f'{n("zero4")} = stablehlo.constant dense<0> : tensor<{B}x{C}x{H}x{W}xi32>',
              f'{n("r0")} = stablehlo.compare EQ, {n("prb")}, {n("zero4")} : (tensor<{B}x{C}x{H}x{W}xi32>, tensor<{B}x{C}x{H}x{W}xi32>) -> tensor<{B}x{C}x{H}x{W}xi1>',
              f'{n("c0")} = stablehlo.compare EQ, {n("pcb")}, {n("zero4")} : (tensor<{B}x{C}x{H}x{W}xi32>, tensor<{B}x{C}x{H}x{W}xi32>) -> tensor<{B}x{C}x{H}x{W}xi1>',
              f'{n("r1")} = stablehlo.not {n("r0")} : tensor<{B}x{C}x{H}x{W}xi1>',
              f'{n("c1")} = stablehlo.not {n("c0")} : tensor<{B}x{C}x{H}x{W}xi1>',
              f'{n("ZF")} = stablehlo.constant dense<0.0> : {T}']
        assert H == W
        claimed = None; acc = None
        for i, j in offs:
            e = n(f'e{i}{j}')
            o.append(f'{e} = stablehlo.compare EQ, {n(f"s{i}{j}")}, {n("m")} : ({t}, {t}) -> {ti}')
            mask = e
            if claimed:
                o += [f'{n(f"nc{i}{j}")} = stablehlo.not {claimed} : {ti}',
                      f'{n(f"mk{i}{j}")} = stablehlo.and {e}, {n(f"nc{i}{j}")} : {ti}',
                      f'{n(f"cl{i}{j}")} = stablehlo.or {claimed}, {e} : {ti}']
                mask = n(f'mk{i}{j}'); claimed = n(f'cl{i}{j}')
            else:
                claimed = e
            tw = f'tensor<{B}x{C}x{w}x{H}xf32>'
            o += [f'{n(f"c{i}{j}")} = stablehlo.select {mask}, {m["dy"]}, {n("zs")} : {ti}, {t}',
                  f'{n(f"d{i}{j}")} = stablehlo.dot_general {n(f"c{i}{j}")}, {n("U")}, contracting_dims = [2] x [0] : ({t}, tensor<{h}x{H}xf32>) -> {tw}',
                  f'{n(f"u{i}{j}")} = stablehlo.dot_general {n(f"d{i}{j}")}, {n("U")}, contracting_dims = [2] x [0] : ({tw}, tensor<{w}x{W}xf32>) -> {T}',
                  f'{n(f"q{i}{j}")} = stablehlo.and {n("r" + str(i))}, {n("c" + str(j))} : tensor<{B}x{C}x{H}x{W}xi1>',
                  f'{n(f"p{i}{j}")} = stablehlo.select {n(f"q{i}{j}")}, {n(f"u{i}{j}")}, {n("ZF")} : tensor<{B}x{C}x{H}x{W}xi1>, {T}']
            if acc:
                o.append(f'{n(f"a{i}{j}")} = stablehlo.add {acc}, {n(f"p{i}{j}")} : {T}'); acc = n(f'a{i}{j}')
            else:
                acc = n(f'p{i}{j}')
        # the init value is added once to every cell, as the scatter's add region does
        o.append(f'{n("ib")} = stablehlo.broadcast_in_dim {m["init"]}, dims = [] : (tensor<f32>) -> {T}')
        o.append(f'{m["res"]} = stablehlo.add {acc}, {n("ib")} : {T}')
        return ''.join(m['ind'] + l + '\n' for l in o)
    assert '%nsas' not in txt
    out = SAS.sub(one, txt)
    return out, k


if __name__ == '__main__':
    import numpy as np
    sys.path.insert(0, '/home/skoonce/lean/lean4-mlir/scripts/platform'); import tier2
    name = sys.argv[2]
    path = tier2.ROOT / 'verified_mlir' / f'{name}.mlir'
    txt = path.read_text()
    new, k = rewrite(txt)
    left = new.count('select_and_scatter')
    print(f'{name}: rewrote {k}, {left} left')
    fn, ins, outs = tier2.signature(name)
    inputs = tier2.make_inputs(name, ins)
    tmp = tier2.ROOT / '.sas_tmp.mlir'; tmp.write_text(new)
    try:
        a = tier2.run_cpu(fn, path, inputs); b = tier2.run_cpu(fn, tmp, inputs)
    finally:
        tmp.unlink()
    e = max(float(np.max(np.abs(x - y)) / max(np.max(np.abs(x)), 1e-30)) for x, y in zip(a, b))
    print(f'  max rel diff over {len(a)} outputs: {e:.1e}')
