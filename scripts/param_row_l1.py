#!/usr/bin/env python3
"""Per-fan-in ROW-L1 profile of a checkpoint — the conv/dense face `layerBudget` throws away.

`layerAct m w beta A = m*w*A + beta` charges the fan-in times the UNIFORM max weight. What that
expression is an upper bound FOR is the output row's L1 norm:

    |(Wx)_o| = |sum_i W[o,i]*x_i| <= (sum_i |W[o,i]|) * A     and    m*w' >= max_o sum_i |W[o,i]|

so replacing `m*w'` by the measured `max_o ||W_o||_1` is a strictly tighter bound of the SAME
kind — a measured property of the committed checkpoint, exactly as `|w| <= 21/10` already is. No
new hypothesis, no modelling change. `scripts/adjoint_chain_probe.py`'s header names the loose
form as the PROVEN tier's whole weakness (`H_i = prod m*max|W_j|`, "EXACTLY the old
FloatClose.comp interval fold"); the row-L1 is the static, provable half of what its MEASURED
tier gets from the on-trajectory Jacobian.

⚠ The ROUNDING half of `layerBudget` — `(1+u)^(m+2) - 1` — keeps the fan-in `m` and does NOT
change. Only the magnitude face moves.

⚠ Measured on ResNet-34 only so far. §3.9 finding 5's rule holds for the other five nets: which
kind is the outlier is not predictable, so measure rather than assume.

Run: python3 scripts/param_row_l1.py [<loader.py> <checkpoint.bin> <w_prime>]
"""
import re
import sys

import numpy as np

DEFAULT = ("ResNet-34", "jax/generated/generated_resnet34_imagenet_short.py",
           "/home/skoonce/resnet/r34_imagenet_bf16_e79.bin", 21 / 10)

SLOT = re.compile(
    r"^\s*(\w+) = jnp\.array\(buf\[idx:idx\+(\d+)\](?:\.reshape\(([\d, ]+)\))?(\.T)?\)", re.M)


def row_l1(loader_path, ckpt_path):
    """fan-in -> max over every stage of that shape of `max_o sum_i |W[o,i]|`.

    Keyed by fan-in and maximised rather than kept per stage, so the result is independent of the
    order a fold walks the net in and is an upper bound at every stage of that shape.
    ⚠ `.T` matters: the classifier is stored `(512, 1000).T`, so its ROWS are the 1000 outputs."""
    body = (open(loader_path).read().split("def init_params_from_file", 1)[1]
            .split("\ndef ", 1)[0])
    buf = np.fromfile(ckpt_path, dtype=np.float32)
    out, i = {}, 0
    for m in SLOT.finditer(body):
        name, n, shp, transposed = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        seg = buf[i:i + n]
        i += n
        if name != "W" or not shp:
            continue
        W = seg.reshape([int(d) for d in shp.split(",") if d.strip()])
        if transposed:
            W = W.T
        oc, fanin = W.shape[0], int(np.prod(W.shape[1:]))
        out[fanin] = max(out.get(fanin, 0.0),
                         float(np.abs(W.reshape(oc, fanin)).sum(1).max()))
    return out, i, len(buf)


def r34_fold(l1):
    """Both r34 forwards folded with the conv face at the measured row-L1 — the whole point.

    Monkeypatches `conv` rather than adding a flag, because the flag would have to reach every
    net's chain and this is a scoping measurement, not a committed number."""
    sys.path.insert(0, "scripts")
    from fractions import Fraction as F
    import float_budget_envelope as fb
    orig, missing = fb.conv, []

    def conv_l1(st, m, w, b, u=fb.U32):
        if m not in l1:
            missing.append(m)
            return orig(st, m, w, b, u)
        return orig(st, m, F(l1[m]).limit_denominator(10 ** 6) / m, b, u)

    fb.conv = conv_l1
    try:
        train = dict(fb.r34_train_chain())["dense"]
        infer = fb.r34_eval_chain()[-1][1]
    finally:
        fb.conv = orig
    assert not missing, f"fan-ins with no measured row-L1: {sorted(set(missing))}"
    return train, infer


def sci(x):
    from fractions import Fraction as F
    import float_budget_envelope as fb
    e = fb.ilog10(x)
    return f"{float(x / F(10) ** e):.4g}e{e}"


if __name__ == "__main__":
    tag, loader, ckpt, wp = (sys.argv[1:4] + [float(sys.argv[4])] if len(sys.argv) == 5
                             else DEFAULT)
    l1, used, total = row_l1(loader, ckpt)
    print(f"{tag}: {total:,} f32 ({used:,} read), {len(l1)} distinct fan-ins")
    print(f"  {'fan-in':>8} {'row-L1':>10} {'m*w_prime':>12} {'loose by':>10}")
    prod = 1.0
    for k in sorted(l1):
        prod *= (k * wp) / l1[k]
        print(f"  {k:>8} {l1[k]:>10.3f} {k * wp:>12.1f} {k * wp / l1[k]:>9.1f}x")
    print(f"  per-SHAPE product: {prod:.3e}  (the fold walks 36 stages, so more)")
    if tag == "ResNet-34":
        (tA, tE), (iA, iE) = r34_fold(l1)
        print("\n  r34 @ TRAINING BN   committed 8.748e80 / 1.752e81")
        print(f"                      row-L1    {sci(tA)} / {sci(tE)}")
        print("  r34 @ INFERENCE BN  committed 3.152e211 / 1.548e209")
        print(f"                      row-L1    {sci(iA)} / {sci(iE)}")
