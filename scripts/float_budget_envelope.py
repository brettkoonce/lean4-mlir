#!/usr/bin/env python3
"""Exact-rational fold of the whole-net float ENVELOPE, in the Lean lemmas' semantics.

Provenance for the numerals in `LeanMlir/Proofs/Float/*FloatBudget.lean`: this script
folds `(output window, error budget)` through a net with EXACTLY the arithmetic the
`FloatBridgesTo.Maps` leaves prove (`layerAct`/`layerBudget`, `bnNormBudget`/`bnReluBudget`
via `mulErr`, `gamma_num`'s `k·u/(1−k·u)`, and the residual/two-branch combinators), rounds
every stage UP to four significant figures, re-asserts each rounded inequality exactly, and
emits the Lean `have` chain from the same numbers.

The kernel is the check; this is only the search for the numerals.
Run: python3 scripts/float_budget_envelope.py
"""
from fractions import Fraction as F
from math import floor, log10

U32 = F(1, 2 ** 24)


def gamma_q(k: int, u=U32) -> F:
    """`FloatModel.gamma_num`'s rational bound on (1+u)^k − 1: k·u/(1 − k·u)."""
    return (k * u) / (1 - k * u)


def ilog10(x: F) -> int:
    """floor(log10 x) for a positive rational, in exact integer arithmetic (the whole-net
    windows overflow binary floating point long before the fold ends)."""
    e = len(str(x.numerator)) - len(str(x.denominator))
    while F(10) ** (e + 1) <= x:
        e += 1
    while F(10) ** e > x:
        e -= 1
    return e


def r4(x: F) -> F:
    """Round a nonnegative rational UP to 4 significant figures."""
    if x == 0:
        return F(0)
    step = F(10) ** (ilog10(x) - 3)
    n = -((-x) // step)          # ceil division on Fractions
    return F(n) * step


def mulErr(u, A, C, ea, ec):
    return u * ((A + ea) * (C + ec)) + (A * ec + ea * C + ea * ec)


def bnNormBudget(u, D, S, G, Bb, em, ei):
    inner = mulErr(u, D, S, u * (D + em) + em, ei)
    outer = mulErr(u, G, D * S, F(0), inner)
    return u * (G * (D * S) + outer + Bb) + outer


# ── the leaf steps: (window, error) -> (window, error), Lean-exact ───────────
def conv(st, m, w, b, u=U32):
    """One convolution. `g` is the ROUNDED-UP γ bound — the same rational the Lean chain
    passes to `FloatModel.gamma_num`, not the exact `(1+u)^k − 1`. Folding with the exact
    value instead makes every emitted stage numeral marginally too small, which the kernel
    then rejects: the fold must assert exactly what the proof asserts."""
    A, E = st
    g = r4(gamma_q(m + 2, u))
    return ((1 + g) * (m * w * A + b),
            g * (m * w * (A + E) + b) + m * w * E)


def dense(st, m, w, b, u=U32):
    return conv(st, m, w, b, u)


def ident(st):
    return st


def gap(st, hw, q=U32, u=U32):
    A, E = st
    g = r4(gamma_q(hw + 1, u))
    return (A * ((1 + g) * (1 + q)), A * (q * (1 + g) + g) + E)


def bn(st, S, G, Bb, mrel, iabs, Sq, Tq, q=U32):
    A, E = st
    nb = bnNormBudget(q, 2 * A, S, G, Bb, mrel * A, iabs)
    return (G * (2 * A * S) + Bb + nb,
            nb + G * ((E + E) * Sq + 2 * A * (8 * A * E * Tq)))


def residual(inp, body, q=U32):
    A, E = inp
    Bd, Ed = body
    return (Bd + A + q * (Bd + A), q * (Bd + Ed + A + E) + (Ed + E))


def bipath(proj, body, q=U32):
    Pd, Ep = proj
    Bd, Ed = body
    return (Pd + Bd + q * (Pd + Bd), q * (Pd + Ep + Bd + Ed) + (Ep + Ed))


# ── CIFAR-8 regression case (must reproduce Cifar8FloatBudget.lean) ──────────
def cifar8():
    w, b = F(2, 5), F(1, 100)
    st = (F(1), F(0))
    fanins = [3 * 9, 16 * 9, 16 * 9, 16 * 9, 16 * 9, 32 * 9, 32 * 9, 32 * 9]
    out = []
    for i, m in enumerate(fanins):
        st = conv(st, m, w, b)
        st = (r4(st[0]), r4(st[1]))
        out.append(("conv%d" % (i + 1), st))
    for m in (128, 64, 64):
        st = dense(st, m, w, b)
        st = (r4(st[0]), r4(st[1]))
        out.append(("dense%d" % m, st))
    return out


if __name__ == "__main__":
    print("── CIFAR-8 regression (Cifar8FloatBudget.lean) ──")
    for name, (A, E) in cifar8():
        print(f"  {name:8s} window {float(A):.4e}   budget {float(E):.4e}")


# ════════════════════════════════════════════════════════════════════════════
# ResNet-34 @224², INFERENCE BatchNorm (the deployed `@resnet34_fwd_eval` forward)
# ════════════════════════════════════════════════════════════════════════════
#
# Profile (measured on /home/skoonce/resnet/r34_imagenet_bf16_e79.bin, the 79-epoch
# ImageNet run): every stored parameter has |·| ≤ 21/10 (global max 2.0741; the
# 99.99th percentile of the conv weights is 0.43, so the uniform bound is loose and
# the fold's size is not sensitive to it). ε ≥ 10⁻⁵ ⇒ 1/√ε ≤ 317. `es` is the
# SUPPLIED device-rsqrt accuracy (no IEEE spec, so modelled, as `esig`/`eexp` are).

R34_W = F(21, 10)     # conv / dense weights
R34_B = F(21, 10)     # conv / dense biases
R34_G = F(21, 10)     # BN γ
R34_BB = F(21, 10)    # BN β
R34_MB = F(21, 10)    # BN frozen running mean
R34_S = F(317)        # 1/√ε at ε ≥ 1e-5
R34_ES = F(1, 100)    # device rsqrt accuracy


def bn_eval(st, S=R34_S, G=R34_G, Bb=R34_BB, Mb=R34_MB, es=R34_ES, q=U32):
    A, E = st
    nb = bnNormBudget(q, A + Mb, S, G, Bb, F(0), es)
    return (G * ((A + Mb) * S) + Bb + nb, nb + G * S * E)


def r34_eval_chain():
    """Every stage of the deployed r34 eval forward, at block granularity.

    Yields (tag, (window, budget)) after each numeric step, in exactly the order the
    Lean `Maps` chain composes them (`relu` and the 3×3/s2 pool carry the envelope
    through unchanged and produce no entry)."""
    def R(st):
        return (r4(st[0]), r4(st[1]))
    out = []
    st = (F(1), F(0))
    st = R(conv(st, 3 * 7 * 7, R34_W, R34_B)); out.append(("stem.conv", st))
    st = R(bn_eval(st));                       out.append(("stem.bn", st))
    plan = [("id", 64, 64, "a0"), ("id", 64, 64, "a1"), ("id", 64, 64, "a2"),
            ("down", 64, 128, "d2"),
            ("id", 128, 128, "b0"), ("id", 128, 128, "b1"), ("id", 128, 128, "b2"),
            ("down", 128, 256, "d3"),
            ("id", 256, 256, "c0"), ("id", 256, 256, "c1"), ("id", 256, 256, "c2"),
            ("id", 256, 256, "c3"), ("id", 256, 256, "c4"),
            ("down", 256, 512, "d4"),
            ("id", 512, 512, "e0"), ("id", 512, 512, "e1")]
    for kind, ic, oc, tag in plan:
        blkin = st
        if kind == "id":
            s = R(conv(blkin, oc * 9, R34_W, R34_B)); out.append((f"{tag}.conv1", s))
            s = R(bn_eval(s));                        out.append((f"{tag}.bn1", s))
            s = R(conv(s, oc * 9, R34_W, R34_B));     out.append((f"{tag}.conv2", s))
            s = R(bn_eval(s));                        out.append((f"{tag}.bn2", s))
            st = R(residual(blkin, s));               out.append((f"{tag}.out", st))
        else:
            p = R(conv(blkin, ic, R34_W, R34_B));     out.append((f"{tag}.projconv", p))
            p = R(bn_eval(p));                        out.append((f"{tag}.projbn", p))
            s = R(conv(blkin, ic * 9, R34_W, R34_B)); out.append((f"{tag}.conv1", s))
            s = R(bn_eval(s));                        out.append((f"{tag}.bn1", s))
            s = R(conv(s, oc * 9, R34_W, R34_B));     out.append((f"{tag}.conv2", s))
            s = R(bn_eval(s));                        out.append((f"{tag}.bn2", s))
            st = R(bipath(p, s));                     out.append((f"{tag}.out", st))
    st = R(gap(st, 49));                              out.append(("gap", st))
    st = R(dense(st, 512, R34_W, R34_B));             out.append(("dense", st))
    return out


def lean_num(x: F) -> str:
    """A 4-significant-figure rational as a Lean literal."""
    if x == 0:
        return "0"
    if x.denominator == 1 and x < 10000:
        return str(x.numerator)          # the relu6 clamp's window is exactly `6`
    e = ilog10(x) - 3
    m = x / (F(10) ** e)
    assert m.denominator == 1 and 1000 <= m <= 9999, (m, e)
    if e >= 0:
        return f"{m.numerator} * 10 ^ {e}"
    return f"{m.numerator} / 10 ^ {-e}"


def verify_r34(rows) -> int:
    """Re-assert EVERY rounded inequality the Lean chain closes, exactly.

    The fold rounds each stage up, so each stage's emitted numeral must still dominate the
    exact value computed from the PREVIOUS stage's emitted numerals — that is the property
    the kernel checks, and the only one worth asserting here. Returns the count checked;
    raises on the first failure."""
    W, B, G, Bb = R34_W, R34_B, R34_G, R34_BB
    Mb, S, es, q = R34_MB, R34_S, R34_ES, U32
    r = dict(rows)
    n = 0

    def ck(tag, lhs, rhs):
        nonlocal n
        assert lhs <= rhs, f"{tag}: {float(lhs)} > {float(rhs)}"
        n += 1

    def conv_ck(tag, m, src, dst):
        A, E = src
        g = r4(gamma_q(m + 2))
        ck(tag + ".A", (1 + g) * (m * W * A + B), dst[0])
        ck(tag + ".E", g * (m * W * (A + E) + B) + m * W * E, dst[1])

    def bn_ck(tag, src, dst):
        A, E = src
        nb = bnNormBudget(q, A + Mb, S, G, Bb, F(0), es)
        ck(tag + ".A", G * ((A + Mb) * S) + Bb + nb, dst[0])
        ck(tag + ".E", nb + G * S * E, dst[1])

    st = (F(1), F(0))
    conv_ck("stem.conv", 3 * 7 * 7, st, r["stem.conv"])
    bn_ck("stem.bn", r["stem.conv"], r["stem.bn"])
    st = r["stem.bn"]
    plan = [("id", 64, 64, "a0"), ("id", 64, 64, "a1"), ("id", 64, 64, "a2"),
            ("down", 64, 128, "d2"),
            ("id", 128, 128, "b0"), ("id", 128, 128, "b1"), ("id", 128, 128, "b2"),
            ("down", 128, 256, "d3"),
            ("id", 256, 256, "c0"), ("id", 256, 256, "c1"), ("id", 256, 256, "c2"),
            ("id", 256, 256, "c3"), ("id", 256, 256, "c4"),
            ("down", 256, 512, "d4"),
            ("id", 512, 512, "e0"), ("id", 512, 512, "e1")]
    for kind, ic, oc, t in plan:
        blkin = st
        if kind == "id":
            conv_ck(f"{t}.conv1", oc * 9, blkin, r[f"{t}.conv1"])
            bn_ck(f"{t}.bn1", r[f"{t}.conv1"], r[f"{t}.bn1"])
            conv_ck(f"{t}.conv2", oc * 9, r[f"{t}.bn1"], r[f"{t}.conv2"])
            bn_ck(f"{t}.bn2", r[f"{t}.conv2"], r[f"{t}.bn2"])
            A4, E4 = r[f"{t}.bn2"]
            ck(f"{t}.resA", A4 + blkin[0] + q * (A4 + blkin[0]), r[f"{t}.out"][0])
            ck(f"{t}.resE", q * (A4 + E4 + blkin[0] + blkin[1]) + (E4 + blkin[1]),
               r[f"{t}.out"][1])
        else:
            conv_ck(f"{t}.projconv", ic, blkin, r[f"{t}.projconv"])
            bn_ck(f"{t}.projbn", r[f"{t}.projconv"], r[f"{t}.projbn"])
            conv_ck(f"{t}.conv1", ic * 9, blkin, r[f"{t}.conv1"])
            bn_ck(f"{t}.bn1", r[f"{t}.conv1"], r[f"{t}.bn1"])
            conv_ck(f"{t}.conv2", oc * 9, r[f"{t}.bn1"], r[f"{t}.conv2"])
            bn_ck(f"{t}.bn2", r[f"{t}.conv2"], r[f"{t}.bn2"])
            P2, Q2 = r[f"{t}.projbn"]
            A4, E4 = r[f"{t}.bn2"]
            ck(f"{t}.sumA", P2 + A4 + q * (P2 + A4), r[f"{t}.out"][0])
            ck(f"{t}.sumE", q * (P2 + Q2 + A4 + E4) + (Q2 + E4), r[f"{t}.out"][1])
        st = r[f"{t}.out"]
    g50 = r4(gamma_q(49 + 1))
    ck("gap.A", st[0] * ((1 + g50) * (1 + q)), r["gap"][0])
    ck("gap.E", st[0] * (q * (1 + g50) + g50) + st[1], r["gap"][1])
    conv_ck("dense", 512, r["gap"], r["dense"])   # dense: same shape at fan-in 512
    return n


# ════════════════════════════════════════════════════════════════════════════
# MobileNetV2 @224², INFERENCE BatchNorm (the deployed eval forward)
# ════════════════════════════════════════════════════════════════════════════
#
# Profile measured on /home/skoonce/mnv2_350ep/mobilenet_v2_imagenet.bin (3.5M f32):
# global max |·| = 2.7157, 99.99th percentile 1.53, two entries above 2 — so a uniform
# |·| ≤ 28/10 covers every stored parameter with room, the r34 pattern. ε ≥ 10⁻⁵ ⇒
# 1/√ε ≤ 317; `es` is the SUPPLIED device-rsqrt accuracy (no IEEE spec, so modelled).
#
# ⭐ The one structural difference from r34: `relu6 x i = min (max (x i) 0) 6` is bounded
# by 6 WHATEVER its input, so `floatClose_relu6`'s `FloatClose A (min A 6)` RESETS the
# window at every expand/depthwise site. Every BN in this net outputs a window far above
# 6, so every relu6 stage emits exactly `6` and the fold is periodic in the window. The
# BUDGET is unmoved by that — it is driven by the per-site error gain `G·S`, i.e. by the
# ε-floor `S = 1/√ε`. Window and budget are separate levers; only `S` moves the budget.

MNV2_W = F(28, 10)      # every stored parameter, uniformly
MNV2_S = F(317)         # 1/√ε at ε ≥ 1e-5
MNV2_ES = F(1, 100)     # device rsqrt accuracy

# (tag, kind, expand fan-in = ic, project fan-in = mid); depthwise fan-in is kH·kW = 9.
# Read off `mobilenetv2Forward_full_pc` (MobileNetV2RenderPC.lean): b1/b3/b5/b6 are
# `invresBodyStridedPC` (no skip — channels and/or spatial change), b2/b4 are
# `residual (invresBodyPC …)`.
MNV2_PLAN = [("b1", "strided", 16, 64), ("b2", "skip", 24, 96),
             ("b3", "strided", 24, 96), ("b4", "skip", 32, 128),
             ("b5", "strided", 32, 128), ("b6", "strided", 64, 256)]


def bn_eval_mnv2(st, S=MNV2_S, w=MNV2_W, es=MNV2_ES, q=U32):
    G = Bb = Mb = w
    A, E = st
    nb = bnNormBudget(q, A + Mb, S, G, Bb, F(0), es)
    return (G * ((A + Mb) * S) + Bb + nb, nb + G * S * E)


def relu6(st):
    """`Maps.relu6`: window `min Ā 6` (the clamp), error unchanged (relu6 is exact in
    float and 1-Lipschitz). The half of the clamp `Maps.relu` does not have."""
    return (min(st[0], F(6)), st[1])


def mnv2_eval_chain(S=MNV2_S, w=MNV2_W, es=MNV2_ES, q=U32, relu6_clamp=True):
    """Every stage of the deployed MobileNetV2 eval forward, at block granularity.

    Yields (tag, (window, budget)) after each numeric step, in exactly the order the Lean
    `Maps` chain composes them. `relu6_clamp=False` reverts to the pre-2026-09-03 leaf
    (`FloatClose A A`, the clamp thrown away) and is kept only to reproduce the "window
    compounds to 10¹⁰⁰ instead of 10³" comparison in planning/float_budget_numbers.md."""
    def R(st):
        return (r4(st[0]), r4(st[1]))

    def r6(st):
        return R(relu6(st)) if relu6_clamp else st

    def bnE(st):
        return R(bn_eval_mnv2(st, S, w, es, q))

    def cv(st, m):
        return R(conv(st, m, w, w))

    out = []
    st = (F(1), F(0))
    st = cv(st, 3 * 3 * 3);  out.append(("stem.conv", st))    # 3×3/s2 stem
    st = bnE(st);            out.append(("stem.bn", st))
    st = r6(st);             out.append(("stem.r6", st))
    for tag, kind, fe, fp in MNV2_PLAN:
        blkin = st
        s = cv(blkin, fe);   out.append((f"{tag}.econv", s))  # expand 1×1
        s = bnE(s);          out.append((f"{tag}.ebn", s))
        s = r6(s);           out.append((f"{tag}.er6", s))
        s = cv(s, 9);        out.append((f"{tag}.dw", s))     # depthwise 3×3 (fan-in 9)
        s = bnE(s);          out.append((f"{tag}.dbn", s))
        s = r6(s);           out.append((f"{tag}.dr6", s))
        s = cv(s, fp);       out.append((f"{tag}.pconv", s))  # project 1×1 (no relu6)
        s = bnE(s);          out.append((f"{tag}.pbn", s))
        if kind == "skip":
            s = R(residual(blkin, s, q)); out.append((f"{tag}.out", s))
        st = s
    st = cv(st, 64);         out.append(("head.conv", st))    # head 1×1, 64 → 128
    st = bnE(st);            out.append(("head.bn", st))
    st = r6(st);             out.append(("head.r6", st))
    st = R(gap(st, 49, q));  out.append(("gap", st))
    st = R(dense(st, 128, w, w)); out.append(("dense", st))
    return out


def verify_mnv2(rows, S=MNV2_S, w=MNV2_W, es=MNV2_ES, q=U32) -> int:
    """Re-assert EVERY rounded inequality the Lean chain closes, exactly.

    The peer of `verify_r34`, and the pass §3.2 step 6 called for: each stage's emitted
    numeral must still dominate the exact value computed from the PREVIOUS stage's emitted
    numerals, which is exactly what the kernel checks. On r34 this pass is what caught the
    rounded-γ bug. Returns the count checked; raises on the first failure."""
    G = Bb = Mb = w
    r = dict(rows)
    n = 0

    def ck(tag, lhs, rhs):
        nonlocal n
        assert lhs <= rhs, f"{tag}: {float(lhs)} > {float(rhs)}"
        n += 1

    def conv_ck(tag, m, src, dst):
        A, E = src
        g = r4(gamma_q(m + 2))
        ck(tag + ".A", (1 + g) * (m * w * A + w), dst[0])
        ck(tag + ".E", g * (m * w * (A + E) + w) + m * w * E, dst[1])

    def bn_ck(tag, src, dst):
        A, E = src
        nb = bnNormBudget(q, A + Mb, S, G, Bb, F(0), es)
        ck(tag + ".A", G * ((A + Mb) * S) + Bb + nb, dst[0])
        ck(tag + ".E", nb + G * S * E, dst[1])

    def r6_ck(tag, src, dst):
        ck(tag + ".A", min(src[0], F(6)), dst[0])
        ck(tag + ".E", src[1], dst[1])

    conv_ck("stem.conv", 3 * 3 * 3, (F(1), F(0)), r["stem.conv"])
    bn_ck("stem.bn", r["stem.conv"], r["stem.bn"])
    r6_ck("stem.r6", r["stem.bn"], r["stem.r6"])
    st = r["stem.r6"]
    for tag, kind, fe, fp in MNV2_PLAN:
        blkin = st
        conv_ck(f"{tag}.econv", fe, blkin, r[f"{tag}.econv"])
        bn_ck(f"{tag}.ebn", r[f"{tag}.econv"], r[f"{tag}.ebn"])
        r6_ck(f"{tag}.er6", r[f"{tag}.ebn"], r[f"{tag}.er6"])
        conv_ck(f"{tag}.dw", 9, r[f"{tag}.er6"], r[f"{tag}.dw"])
        bn_ck(f"{tag}.dbn", r[f"{tag}.dw"], r[f"{tag}.dbn"])
        r6_ck(f"{tag}.dr6", r[f"{tag}.dbn"], r[f"{tag}.dr6"])
        conv_ck(f"{tag}.pconv", fp, r[f"{tag}.dr6"], r[f"{tag}.pconv"])
        bn_ck(f"{tag}.pbn", r[f"{tag}.pconv"], r[f"{tag}.pbn"])
        st = r[f"{tag}.pbn"]
        if kind == "skip":
            A4, E4 = st
            ck(f"{tag}.resA", A4 + blkin[0] + q * (A4 + blkin[0]), r[f"{tag}.out"][0])
            ck(f"{tag}.resE", q * (A4 + E4 + blkin[0] + blkin[1]) + (E4 + blkin[1]),
               r[f"{tag}.out"][1])
            st = r[f"{tag}.out"]
    conv_ck("head.conv", 64, st, r["head.conv"])
    bn_ck("head.bn", r["head.conv"], r["head.bn"])
    r6_ck("head.r6", r["head.bn"], r["head.r6"])
    g50 = r4(gamma_q(49 + 1))
    A, E = r["head.r6"]
    ck("gap.A", A * ((1 + g50) * (1 + q)), r["gap"][0])
    ck("gap.E", A * (q * (1 + g50) + g50) + E, r["gap"][1])
    conv_ck("dense", 128, r["gap"], r["dense"])
    return n


# ════════════════════════════════════════════════════════════════════════════
# EfficientNet-B0 (the representative B0), INFERENCE BatchNorm
# ════════════════════════════════════════════════════════════════════════════
#
# Profile measured on /home/skoonce/enet_b0_350_4gpu/efficientnet_b0_imagenet.bin
# (5,288,548 f32): global max |·| = 4.0545, 99.99th percentile 1.3949, 100 entries
# above 2 and 2 above 4 — so a uniform |·| ≤ 41/10 covers every stored parameter.
# ε ≥ 10⁻⁵ ⇒ 1/√ε ≤ 317. `es` is the supplied device-rsqrt accuracy and `esig` the
# supplied sigmoid accuracy — both modelled, no IEEE spec for either.
#
# ⭐ Two of this net's leaves had to be TIGHTENED before the fold was statable at all
# (2026-09-03, `EnetFloatBridge.lean`); the numbers below are at the tightened ones.
#   * `floatClose_swish`'s modulus was `mulErr + (1 + A/4)·e` — the inherited error
#     multiplied by the WINDOW at every swish site. It is now the `min` of that and
#     `mulErr + (A + e)` (`swishScalar_lipschitz_abs'`, bounding |σa − σb| by the gate's
#     own range instead of by ¼|a−b|). Multiplicative-only: budget 1e1737.
#   * `floatClose_seScale`'s window was derived as |float − real| + |real|, charging the
#     GATE'S ERROR (A · Lg 0) to the magnitude. `FloatClose`'s magnitude clause already
#     bounds the float gate, so the window is A·Bg·(1+u). Untightened: window 1e417.
# Both fixes are the relu6 pattern — a bound proved one lemma down and discarded by the
# generic combinator. `norm_num` refuses numerals past ~1e300, so each was a hard block.

B0_W = F(41, 10)        # every stored parameter, uniformly
B0_S = F(317)           # 1/√ε at ε ≥ 1e-5
B0_ES = F(1, 100)       # device rsqrt accuracy
B0_ESIG = F(1, 100)     # deployed sigmoid accuracy

# (tag, kind, expand fan-in, depthwise fan-in kH·kW, SE channels c, SE reduction r,
#  SE spatial h·w, project fan-in), read off `efficientnetForwardB`
# (EfficientNetRenderPC.lean). `expand = None` is the MBConv1 no-expand block.
B0_PLAN = [("b1", "noexp",   None, 9,  32,  8, 112 * 112, 32),
           ("b2", "strided", 16,   9,  96,  4, 56 * 56,   96),
           ("b3", "resid",   24,   25, 144, 6, 56 * 56,   144)]


def b0_eval_chain(w=B0_W, S=B0_S, es=B0_ES, esig=B0_ESIG, q=U32,
                  swish_min=True, se_tight=True):
    """Every stage of the deployed EfficientNet-B0 eval forward, at block granularity.

    Yields (tag, (window, budget)) in exactly the order the Lean `Maps` chain composes
    them. `swish_min=False` / `se_tight=False` revert to the pre-2026-09-03 leaves and are
    kept only to reproduce the "not statable" comparison in
    planning/float_budget_numbers.md §3.4 — do not emit numerals from them."""
    G = Bb = Mb = w

    def R(st):
        return (r4(st[0]), r4(st[1]))

    def bnE(st):
        A, E = st
        nb = bnNormBudget(q, A + Mb, S, G, Bb, F(0), es)
        return R((G * ((A + Mb) * S) + Bb + nb, nb + G * S * E))

    def cv(st, m):
        return R(conv(st, m, w, w))

    def swish(st):
        A, E = st
        me = mulErr(q, A, F(1), F(0), esig)
        tail = min((1 + A / 4) * E, A + E) if swish_min else (1 + A / 4) * E
        return R((A + me, me + tail))

    def sigmoid(st):
        return R((1 + esig, esig + st[1] / 4))

    out = []
    st = (F(1), F(0))
    st = cv(st, 3 * 3 * 3);  out.append(("stem.conv", st))     # 3×3/s2 stem
    st = bnE(st);            out.append(("stem.bn", st))
    st = swish(st);          out.append(("stem.swish", st))
    for tag, kind, fe, fd, c, r, hw, fp in B0_PLAN:
        blkin = st
        if fe is not None:                                     # expand 1×1 (b2/b3)
            st = cv(st, fe);   out.append((f"{tag}.econv", st))
            st = bnE(st);      out.append((f"{tag}.ebn", st))
            st = swish(st);    out.append((f"{tag}.eswish", st))
        st = cv(st, fd);       out.append((f"{tag}.dw", st))    # depthwise (3×3 or 5×5)
        st = bnE(st);          out.append((f"{tag}.dbn", st))
        st = swish(st);        out.append((f"{tag}.dswish", st))
        # squeeze-excite: gate = broadcast ∘ sigmoid ∘ dense(r) ∘ swish ∘ dense(c) ∘ gap,
        # then the rescale x ⊙ gate(x). `broadcast` carries the envelope through unchanged.
        A, E = st
        g = R(gap(st, hw, q)); out.append((f"{tag}.sq", g))
        g = cv(g, c);          out.append((f"{tag}.sd1", g))
        g = swish(g);          out.append((f"{tag}.ssw", g))
        g = cv(g, r);          out.append((f"{tag}.sd2", g))
        g = sigmoid(g);        out.append((f"{tag}.ssig", g))
        Cg, Eg = g
        st = R((A * Cg + q * (A * Cg), mulErr(q, A, Cg, E, Eg)) if se_tight
               else (A * Cg + mulErr(q, A, Cg, F(0), Eg), mulErr(q, A, Cg, E, Eg)))
        out.append((f"{tag}.se", st))
        st = cv(st, fp);       out.append((f"{tag}.pconv", st))  # project 1×1, no swish
        st = bnE(st);          out.append((f"{tag}.pbn", st))
        if kind == "resid":
            st = R(residual(blkin, st, q)); out.append((f"{tag}.out", st))
    st = cv(st, 24);           out.append(("head.conv", st))     # head 1×1, 24 → 1280
    st = bnE(st);              out.append(("head.bn", st))
    st = swish(st);            out.append(("head.swish", st))
    st = R(gap(st, 56 * 56, q)); out.append(("gap", st))
    st = R(dense(st, 1280, w, w)); out.append(("dense", st))
    return out


def verify_b0(rows, w=B0_W, S=B0_S, es=B0_ES, esig=B0_ESIG, q=U32) -> int:
    """Re-assert EVERY rounded inequality the Lean chain closes, exactly.

    The peer of `verify_r34` / `verify_mnv2`. Returns the count checked; raises on the
    first failure."""
    G = Bb = Mb = w
    r_ = dict(rows)
    n = 0

    def ck(tag, lhs, rhs):
        nonlocal n
        assert lhs <= rhs, f"{tag}: {float(lhs)} > {float(rhs)}"
        n += 1

    def conv_ck(tag, m, src, dst):
        A, E = src
        g = r4(gamma_q(m + 2))
        ck(tag + ".A", (1 + g) * (m * w * A + w), dst[0])
        ck(tag + ".E", g * (m * w * (A + E) + w) + m * w * E, dst[1])

    def bn_ck(tag, src, dst):
        A, E = src
        nb = bnNormBudget(q, A + Mb, S, G, Bb, F(0), es)
        ck(tag + ".A", G * ((A + Mb) * S) + Bb + nb, dst[0])
        ck(tag + ".E", nb + G * S * E, dst[1])

    def swish_ck(tag, src, dst):
        A, E = src
        me = mulErr(q, A, F(1), F(0), esig)
        ck(tag + ".A", A + me, dst[0])
        ck(tag + ".E", me + min((1 + A / 4) * E, A + E), dst[1])

    def gap_ck(tag, hw, src, dst):
        A, E = src
        g = r4(gamma_q(hw + 1))
        ck(tag + ".A", A * ((1 + g) * (1 + q)), dst[0])
        ck(tag + ".E", A * (q * (1 + g) + g) + E, dst[1])

    conv_ck("stem.conv", 3 * 3 * 3, (F(1), F(0)), r_["stem.conv"])
    bn_ck("stem.bn", r_["stem.conv"], r_["stem.bn"])
    swish_ck("stem.swish", r_["stem.bn"], r_["stem.swish"])
    st = r_["stem.swish"]
    for tag, kind, fe, fd, c, r, hw, fp in B0_PLAN:
        blkin = st
        if fe is not None:
            conv_ck(f"{tag}.econv", fe, st, r_[f"{tag}.econv"])
            bn_ck(f"{tag}.ebn", r_[f"{tag}.econv"], r_[f"{tag}.ebn"])
            swish_ck(f"{tag}.eswish", r_[f"{tag}.ebn"], r_[f"{tag}.eswish"])
            st = r_[f"{tag}.eswish"]
        conv_ck(f"{tag}.dw", fd, st, r_[f"{tag}.dw"])
        bn_ck(f"{tag}.dbn", r_[f"{tag}.dw"], r_[f"{tag}.dbn"])
        swish_ck(f"{tag}.dswish", r_[f"{tag}.dbn"], r_[f"{tag}.dswish"])
        A, E = r_[f"{tag}.dswish"]
        gap_ck(f"{tag}.sq", hw, (A, E), r_[f"{tag}.sq"])
        conv_ck(f"{tag}.sd1", c, r_[f"{tag}.sq"], r_[f"{tag}.sd1"])
        swish_ck(f"{tag}.ssw", r_[f"{tag}.sd1"], r_[f"{tag}.ssw"])
        conv_ck(f"{tag}.sd2", r, r_[f"{tag}.ssw"], r_[f"{tag}.sd2"])
        Ain, Ein = r_[f"{tag}.sd2"]
        ck(f"{tag}.ssig.A", 1 + esig, r_[f"{tag}.ssig"][0])
        ck(f"{tag}.ssig.E", esig + Ein / 4, r_[f"{tag}.ssig"][1])
        Cg, Eg = r_[f"{tag}.ssig"]
        ck(f"{tag}.se.A", A * Cg + q * (A * Cg), r_[f"{tag}.se"][0])
        ck(f"{tag}.se.E", mulErr(q, A, Cg, E, Eg), r_[f"{tag}.se"][1])
        conv_ck(f"{tag}.pconv", fp, r_[f"{tag}.se"], r_[f"{tag}.pconv"])
        bn_ck(f"{tag}.pbn", r_[f"{tag}.pconv"], r_[f"{tag}.pbn"])
        st = r_[f"{tag}.pbn"]
        if kind == "resid":
            Bd, Ed = st
            ck(f"{tag}.resA", Bd + blkin[0] + q * (Bd + blkin[0]), r_[f"{tag}.out"][0])
            ck(f"{tag}.resE", q * (Bd + Ed + blkin[0] + blkin[1]) + (Ed + blkin[1]),
               r_[f"{tag}.out"][1])
            st = r_[f"{tag}.out"]
    conv_ck("head.conv", 24, st, r_["head.conv"])
    bn_ck("head.bn", r_["head.conv"], r_["head.bn"])
    swish_ck("head.swish", r_["head.bn"], r_["head.swish"])
    gap_ck("gap", 56 * 56, r_["head.swish"], r_["gap"])
    conv_ck("dense", 1280, r_["gap"], r_["dense"])
    return n




# ════════════════════════════════════════════════════════════════════════════
# ConvNeXt-T @224², channel LayerNorm — the CAPPED fold (`ConvNeXtFloatBudget.lean`)
# ════════════════════════════════════════════════════════════════════════════
#
# ⛔ READ THIS BEFORE QUOTING THE NUMBER. Every LayerNorm site here goes through
# `FloatBridgesTo.capped`: its modulus is `min(fold, 2·window)` and the right branch is what
# closes. So the ConvNeXt budget is the TRIANGLE INEQUALITY — "the float and real forwards both
# land in the certified window" — and not the interval fold that r34 / MobileNetV2 /
# EfficientNet-B0's numbers are. The tell is `budget / window = 2.00`. It has to be that way:
# LayerNorm reduces its statistics out of its own input, so its modulus is quadratic in the
# window, and unlike BatchNorm it has no frozen-statistics variant to switch to
# (planning/float_budget_numbers.md §0.1). Uncapped, this fold is 10^11631.
#
# Profile MEASURED on /home/skoonce/convnext/convnext_t300_4gpu/convnext_tiny_imagenet.bin (the
# finished 300-epoch 4-GPU run, 28,587,592 f32) — and it does NOT split uniformly:
#
#     conv/dense kernels  28,524,000   max 0.5962      →  w' = 6/10
#     biases + LN β           49,576   max 2.9499      →  bb = 3
#     LN γ                     7,392   max 4.7700      →  gl = 48/10
#     layer scale              6,624   max 8.3766      →  sl = 84/10
#
# ⭐ A single uniform bound is therefore 8.4, which is 14× loose on exactly the entries the conv
# fan-in multiplies, and the fold then lands at 10^301 — past what `norm_num` will evaluate.
# Split by kind it is 10^227. The four-bound `CnxBlockChBounded` is not tidiness; it is what
# makes the theorem exist.
#
# ⚠ That checkpoint predates the 2026-08-30 head-LayerNorm restoration (it is short by exactly
# 1,536 = 2×768, which is how the missing layer was found), so the head LN's γ/β are not in the
# measurement. They initialise at γ=1, β=0 and the bounds above cover them with room.

CNX_W = F(6, 10)        # conv / dense kernels
CNX_BB = F(3)           # conv / dense biases, and every LayerNorm β
CNX_GL = F(48, 10)      # LayerNorm γ
CNX_SL = F(84, 10)      # layer scale
CNX_S = F(317)          # 1/√ε at ε ≥ 1e-5
CNX_EMR = F(1, 100)     # deployed LN mean accuracy, RELATIVE to the window
CNX_EI = F(1, 100)      # deployed LN inverse-stddev accuracy, absolute
CNX_EGELU = F(1, 100)   # deployed GELU accuracy

# (channels, expand, blocks, spatial) — the [3,3,9,3] ch-variant schedule of `CnxTWeightsCh`.
CNX_STAGES = [(96, 384, 3, 56), (192, 768, 3, 28), (384, 1536, 9, 14), (768, 3072, 3, 7)]


def cnx_ln_leaf(A, S=CNX_S, emr=CNX_EMR, ei=CNX_EI, q=U32):
    """`Maps.bnCapped` at the pure-normalise LayerNorm (γ=1, β=0): the window is the honest
    `2Ā·S + bnNormBudget`, the modulus is `2·Ā'` — the cap, not the fold."""
    return 2 * A * S + bnNormBudget(q, 2 * A, S, F(1), F(0), emr * A, ei)


def cnx_diag(st, Sd, q=U32):
    """`Maps.diagBack` at `es = 0` (a stored weight, no transcendental): the LN γ multiply and
    the layer scale."""
    A, E = st
    me = mulErr(q, Sd, A, F(0), F(0))
    return (Sd * A + me, me + Sd * E)


def cnx_bias(st, Bb, q=U32):
    """`Maps.biasAdd`: the LN β shift, one rounded add per coordinate."""
    A, E = st
    return (A + Bb + q * (A + Bb), q * (A + Bb) + E)


def cnx_gelu(st, egelu=CNX_EGELU, sat=True):
    """`Maps.gelu`, through the `3/2` branch of `floatClose_gelu`'s `min` — the global
    saturation constant (`Architectures/GeluSaturation.lean`). `sat=False` reverts to the
    magnitude polynomial, which is CUBIC in the window and reaches ~400 here."""
    A, E = st
    L = F(3, 2) * E if sat else (1 + F(4, 5) / 2 * A * (1 + 3 * F(44715, 10 ** 6) * A ** 2)) * E
    return (A + egelu, egelu + L)


def cnx_eval_chain(w=CNX_W, bb=CNX_BB, gl=CNX_GL, sl=CNX_SL, S=CNX_S,
                   emr=CNX_EMR, ei=CNX_EI, egelu=CNX_EGELU, q=U32,
                   ln_cap=True, gelu_sat=True, head_ln=True):
    """Every stage of the ConvNeXt-T forward, at exactly the granularity the Lean `Maps` chain
    composes them (the four layout permutations and the per-row lift are envelope-preserving and
    produce no entry).

    The three flags reproduce planning/float_budget_numbers.md §3.3's ablation table:
    `ln_cap=False` is the LayerNorm leaf as `Maps.bn` would state it (quadratic in the window),
    `gelu_sat=False` is `floatClose_gelu` before the saturation branch was wired in, and
    `head_ln=False` is the net the whole-net bridge described until 2026-09-03, when its head
    slot still held `id`."""
    def R(st):
        return (r4(st[0]), r4(st[1]))

    out = []

    def lnsite(st, tag):
        A, E = st
        mag = r4(cnx_ln_leaf(A, S, emr, ei, q))
        if ln_cap:
            err = r4(2 * mag)
        else:                                   # the uncapped `bnLeafMod`, for the ablation only
            nb = bnNormBudget(q, 2 * A, S, F(1), F(0), emr * A, ei)
            err = r4(nb + ((E + E) * S + 2 * A * (8 * A * E * F(158500000))))
        st = (mag, err); out.append((tag + '.ln', st))
        st = R(cnx_diag(st, gl, q)); out.append((tag + '.lng', st))
        st = R(cnx_bias(st, bb, q)); out.append((tag + '.lnb', st))
        return st

    st = (F(1), F(0))
    st = R(conv(st, 3 * 4 * 4, w, bb)); out.append(('stem.conv', st))   # 4×4/s4 patchify
    st = lnsite(st, 'stem')
    for si, (c, ce, nblk, _hw) in enumerate(CNX_STAGES):
        if si > 0:
            cin = CNX_STAGES[si - 1][0]
            st = lnsite(st, f'd{si}')
            st = R(conv(st, cin * 2 * 2, w, bb)); out.append((f'd{si}.conv', st))
        for b in range(nblk):
            t = f's{si + 1}b{b}'
            blkin = st
            s = R(conv(st, 7 * 7, w, bb)); out.append((t + '.dw', s))
            s = lnsite(s, t)
            s = R(conv(s, c, w, bb)); out.append((t + '.ex', s))
            s = R(cnx_gelu(s, egelu, gelu_sat)); out.append((t + '.ge', s))
            s = R(conv(s, ce, w, bb)); out.append((t + '.pr', s))
            s = R(cnx_diag(s, sl, q)); out.append((t + '.ls', s))
            st = R(residual(blkin, s, q)); out.append((t + '.out', st))
    st = R(gap(st, 49, q)); out.append(('gap', st))
    if head_ln:
        st = lnsite(st, 'head')
    st = R(dense(st, 768, w, bb)); out.append(('dense', st))
    return out


def verify_cnx(rows, w=CNX_W, bb=CNX_BB, gl=CNX_GL, sl=CNX_SL, S=CNX_S,
               emr=CNX_EMR, ei=CNX_EI, egelu=CNX_EGELU, q=U32) -> int:
    """Re-assert EVERY rounded inequality `ConvNeXtFloatBudget.lean` closes, exactly — each
    stage's emitted numeral must still dominate the exact value computed from the PREVIOUS
    stage's emitted numerals. Returns the count checked; raises on the first failure."""
    r = dict(rows)
    n = 0

    def ck(tag, lhs, rhs):
        nonlocal n
        assert lhs <= rhs, f"{tag}: {tag} left > right"
        n += 1

    def conv_ck(tag, m, src, dst):
        A, E = src
        g = r4(gamma_q(m + 2))
        ck(tag + '.A', (1 + g) * (m * w * A + bb), dst[0])
        ck(tag + '.E', g * (m * w * (A + E) + bb) + m * w * E, dst[1])

    def ln_ck(tag, src):
        A, E = src
        ck(tag + '.ln.A', cnx_ln_leaf(A, S, emr, ei, q), r[tag + '.ln'][0])
        ck(tag + '.ln.E', 2 * r[tag + '.ln'][0], r[tag + '.ln'][1])
        A1, E1 = r[tag + '.ln']
        me = mulErr(q, gl, A1, F(0), F(0))
        ck(tag + '.lng.A', gl * A1 + me, r[tag + '.lng'][0])
        ck(tag + '.lng.E', me + gl * E1, r[tag + '.lng'][1])
        A2, E2 = r[tag + '.lng']
        ck(tag + '.lnb.A', A2 + bb + q * (A2 + bb), r[tag + '.lnb'][0])
        ck(tag + '.lnb.E', q * (A2 + bb) + E2, r[tag + '.lnb'][1])
        return r[tag + '.lnb']

    st = (F(1), F(0))
    conv_ck('stem.conv', 3 * 4 * 4, st, r['stem.conv'])
    st = ln_ck('stem', r['stem.conv'])
    for si, (c, ce, nblk, _hw) in enumerate(CNX_STAGES):
        if si > 0:
            cin = CNX_STAGES[si - 1][0]
            st = ln_ck(f'd{si}', st)
            conv_ck(f'd{si}.conv', cin * 2 * 2, st, r[f'd{si}.conv'])
            st = r[f'd{si}.conv']
        for b in range(nblk):
            t = f's{si + 1}b{b}'
            blkin = st
            conv_ck(t + '.dw', 7 * 7, blkin, r[t + '.dw'])
            s = ln_ck(t, r[t + '.dw'])
            conv_ck(t + '.ex', c, s, r[t + '.ex'])
            A, E = r[t + '.ex']
            ck(t + '.ge.A', A + egelu, r[t + '.ge'][0])
            ck(t + '.ge.E', egelu + F(3, 2) * E, r[t + '.ge'][1])
            conv_ck(t + '.pr', ce, r[t + '.ge'], r[t + '.pr'])
            A, E = r[t + '.pr']
            me = mulErr(q, sl, A, F(0), F(0))
            ck(t + '.ls.A', sl * A + me, r[t + '.ls'][0])
            ck(t + '.ls.E', me + sl * E, r[t + '.ls'][1])
            A8, E8 = r[t + '.ls']
            ck(t + '.res.A', A8 + blkin[0] + q * (A8 + blkin[0]), r[t + '.out'][0])
            ck(t + '.res.E', q * (A8 + E8 + blkin[0] + blkin[1]) + (E8 + blkin[1]),
               r[t + '.out'][1])
            st = r[t + '.out']
    g50 = r4(gamma_q(49 + 1))
    ck('gap.A', st[0] * ((1 + g50) * (1 + q)), r['gap'][0])
    ck('gap.E', st[0] * (q * (1 + g50) + g50) + st[1], r['gap'][1])
    st = ln_ck('head', r['gap'])
    conv_ck('dense', 768, st, r['dense'])
    return n
