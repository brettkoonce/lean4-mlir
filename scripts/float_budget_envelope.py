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
    windows overflow binary floating point long before the fold ends).

    ⚠ The seed is a BIT-LENGTH estimate, not `len(str(·))`: CPython caps int→str at 4300
    digits (3.10.7+), and the uncapped-LayerNorm ablations run to 10¹¹⁶³¹ (ConvNeXt) and
    beyond — so the string form raised `ValueError` on exactly the folds §0.1 is about.
    `bit_length` is exact and unguarded; 30103/100000 is log10(2), and the two correction
    loops below make the seed's ±1 error irrelevant."""
    e = ((x.numerator.bit_length() - x.denominator.bit_length()) * 30103) // 100000
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


# ════════════════════════════════════════════════════════════════════════════
# ViT-Tiny @224², the depth-12 distinct-per-block net (`Proofs.vitForwardKV`)
# ════════════════════════════════════════════════════════════════════════════
#
# ⛔ The net is `vitForwardKV` (`Architectures/ViTDepthK.lean`), NOT `vit_full`.
# `vit_full` shares ONE parameter tuple across all 12 blocks and carries SCALAR LayerNorm
# affines; the trained checkpoint has per-block weights and vector-[D] affines. They are
# different functions, and only the first is what `vitFwdGraphKMHV_faithful` denotes.
#
# Profile, measured per parameter KIND on /home/skoonce/vit/vit_tiny_imagenet_bf16.bin
# (5,717,416 f32, the layout of `vitImagenetVerified.toSpecs`):
#
#     kind                count      max|·|     bound
#     patch conv kernel   147,456    0.2522     3/10
#     attention kernels   1,769,472  0.6594     7/10      Wq/Wk/Wv/Wo
#     MLP kernels         3,538,944  0.7960     8/10      Wfc1/Wfc2
#     head kernel         192,000    0.3408     4/10
#     biases              21,928     0.8624     9/10
#     LayerNorm γ         4,800      1.6645     17/10
#     LayerNorm β         4,800      0.5609     6/10
#     CLS token           192        0.5454     6/10
#     positional embed    37,824     0.7229     8/10
#
# ⭐ Unlike ConvNeXt-T (whose kinds are 14× apart, and whose uniform bound is UNSTATABLE at
# 10³⁰¹), ViT-Tiny's spread is only 2.5× and the single outlier — the FINAL LayerNorm γ at
# 1.6645 — sits after everything, multiplying only the head. So splitting the profile buys
# ~18 orders here and is NOT load-bearing; see `vit_chain(uniform=True)`.

D_VIT, MLP_VIT, HEADS_VIT, DH_VIT, K_VIT = 192, 768, 3, 64, 12
NTOK_VIT = 197                    # 196 patches + CLS
PATCH_FANIN = 3 * 16 * 16         # 768, the 16×16/s16 patchify

VIT_WA = F(7, 10)       # attention kernels  Wq/Wk/Wv/Wo
VIT_WM = F(8, 10)       # MLP kernels        Wfc1/Wfc2
VIT_WP = F(3, 10)       # patch-embed conv kernel
VIT_WH = F(4, 10)       # classifier head kernel
VIT_BB = F(9, 10)       # every bias
VIT_GL = F(17, 10)      # LayerNorm γ
VIT_BL = F(6, 10)       # LayerNorm β
VIT_CLS = F(6, 10)      # CLS token                                   measured 0.5454
VIT_POS = F(8, 10)      # positional embedding                         measured 0.7229
VIT_PB = F(9, 10)       # ⭐ the patch embed's SINGLE bound, covering pos_embed, cls_token AND
                        # b_conv — `floatClose_patchEmbed` takes one `pb` for all three, so it
                        # is their max (0.7229 / 0.5454 / 0.8624)
VIT_UNI = F(17, 10)     # the single uniform bound, for the ablation

VIT_S = F(317)          # 1/√ε at ε ≥ 1e-5 (ViTRender.lean: ε=1e-5)
VIT_EMR = F(1, 100)     # deployed LN mean accuracy, RELATIVE to the window
VIT_EI = F(1, 100)      # deployed LN inverse-stddev accuracy, absolute
VIT_EG = F(1, 100)      # deployed GELU accuracy
VIT_EEXP = F(1, 100)    # deployed exp accuracy (softmax), RELATIVE


def sm_rho(u, eexp, n):
    """`FloatBridge.lean`'s `smRho`: ((1+u)^(n+1) − 1)(1 + eexp) + eexp. The softmax leaf's
    side condition is `smRho < 1`, which the whole-net statement must carry and disclose."""
    return gamma_q(n + 1, u) * (1 + eexp) + eexp


def sm_kappa(u, eexp, n):
    """`FloatBridge.lean`'s `smKappa`: (eexp + smRho)/(1 − smRho)."""
    rho = sm_rho(u, eexp, n)
    return (eexp + rho) / (1 - rho)


def sm_cap(u, eexp, n):
    """`Proofs.smCap` — `u(1+kappa) + kappa`, the float softmax row's absolute distance from the
    real one at the SAME logits.  This is `smErr` with its `Real.exp (2*delta) - 1` term absent,
    and keeping it as its own name is what keeps the exponential out of the numerals."""
    kap = sm_kappa(u, eexp, n)
    return u * (1 + kap) + kap


def vit_softmax_window(n=NTOK_VIT, eexp=VIT_EEXP, q=U32):
    """The capped row softmax's window, `1 + smCap` (`Proofs.smCap`).  ⭐ CONSTANT in the input
    window: `softmax_abs_le_one` puts the real row in [0,1] and `softmaxF_close` puts the float
    row within `smCap = u(1+kappa) + kappa` of it.  Rational — no `Real.exp` — which is the
    whole point, since `smErr`'s modulus is exponential in the inherited error."""
    return 1 + sm_cap(q, eexp, n)


def vit_attn(st, m=D_VIT, n=NTOK_VIT, w=VIT_WA, b=VIT_BB, eexp=VIT_EEXP, q=U32,
             mode='cap'):
    """**Multi-head projected attention, as ONE stage** — `Maps.mhProjAttnFullCap`.

    ⛔ It is one stage and not four because `FloatBridgesTo` composes single-input maps, and
    attention FANS OUT (X -> Q,K,V) before it rejoins.  The repo's `mhProjAttnFullFlat` already
    bundles it for that reason; what is new here is the WINDOW.

    `mode='cap'` (shipped) is `mhpBCap`: the float output is a rounded dot of float softmax
    weights (<= 1 + smCap) against float V (<= the dense window), giving
    `(1+g2)*n*(1+smCap)*(1+g1)*(m*w*A + b)` — exp-free.
    `mode='convex'` additionally uses the convex-combination bound (drop the `n`), which needs
    a float-side peer of `sdpa_abs_le` that is NOT proved.
    `mode='mhpB'` is the ORIGINAL window `vA_F + attnOutErr`, derived as |real| + |float-real|;
    it carries `smErr` and therefore `Real.exp`, so it returns an exp-taint marker.  ⛔ Capping
    does NOT rescue it: `capped` replaces the modulus, never the window."""
    A, E = st
    g1 = r4(gamma_q(m + 2, q))
    g2 = r4(gamma_q(n + 1, q))
    sc = r4(sm_cap(q, eexp, n))
    vAF = (1 + g1) * (m * w * A + b)                 # the float Q/K/V projection window
    if mode == 'mhpB':
        mag = r4(vAF)
        return (mag, r4(2 * mag), True)              # unwritable: the Real.exp is in the window
    fan = F(1) if mode == 'convex' else F(n)
    # ⚠ round the WINDOW first, then double it. `Maps.capped` closes `2·Ā' ≤ Ē'` against the
    # EMITTED window, and rounding `mag` and `2·mag` up independently can break exactly that
    # (`2·r4(x)` can exceed `r4(2·x)`). `verify_vit` caught this on its first run.
    mag = r4((1 + g2) * (fan * ((1 + sc) * vAF)))
    return (mag, r4(2 * mag), False)


def vit_ln_leaf(A, S=VIT_S, emr=VIT_EMR, ei=VIT_EI, q=U32):
    """`Maps.bnCapped` at the pure-normalise LayerNorm (γ=1, β=0) — identical to ConvNeXt's
    `cnx_ln_leaf`. LayerNorm is LayerNorm, and §0.1 applies unchanged: no eval mode exists,
    so the cap is the only statement available."""
    return 2 * A * S + bnNormBudget(q, 2 * A, S, F(1), F(0), emr * A, ei)


def redErr(u, n, Mr, ef):
    """`redErr` (`PatchEmbedBackFloatBridge.lean`): one rounded-reduction level,
    `gamma_{n+1}*(n*(Mr+ef)) + n*ef`.  ⚠ EXACT `(1+u)^(n+1) - 1` here, not `gamma_q` — the
    patch-embed lemmas are stated on the exact power and the reductions are small (n = 3, 16),
    so `norm_num` evaluates them directly."""
    return ((1 + u) ** (n + 1) - 1) * (F(n) * (Mr + ef)) + F(n) * ef


def pe_convMag(ic, P, wc, A):
    """`patchEmbedConvMag`: the real conv-dot magnitude `ic*P^2*wc*A`."""
    return F(ic) * (F(P) * (F(P) * (wc * A)))


def pe_mag(ic, P, wc, pb, A):
    """`patchEmbedMag`: `pos_embed + (b_conv | cls_token) + conv-dot`."""
    return pb + (pb + pe_convMag(ic, P, wc, A))


def pe_tripleErr(u, ic, P, wc, A):
    """`patchEmbedTripleErr`: the c/kh/kw nest, three `redErr`s over one `mulErr` leaf."""
    return redErr(u, ic, F(P) * (F(P) * (wc * A)),
             redErr(u, P, F(P) * (wc * A),
               redErr(u, P, wc * A, mulErr(u, wc, A, F(0), F(0)))))


def pe_roundErr(u, ic, P, wc, pb, A):
    """`patchEmbedRoundErr`: the triple-sum error plus the two constant adds."""
    t = pe_tripleErr(u, ic, P, wc, A)
    b = u * (pb + pe_convMag(ic, P, wc, A) + t) + t
    return u * (pb + (pb + pe_convMag(ic, P, wc, A)) + b) + b


def vit_patch_embed(st, ic=3, P=16, wc=VIT_WP, pb=VIT_BB, q=U32):
    """**The patch embed, as ONE stage** — `Maps.patchEmbed`.

    ⛔ It is one stage and not three because `patchEmbed_flat` is a single definition with an
    `if n.val = 0` branch selecting the CLS token, NOT a composition `concatCls . convStride16`.
    An earlier fold here modelled it as conv -> CLS-concat -> pos-add and was describing a
    function the repo does not contain; the same granularity trap attention set (see `vit_attn`).
    ⭐ `pb` is ONE bound covering pos_embed, cls_token AND b_conv, because the lemma takes one —
    so it is the max of the three measured kinds (0.7229, 0.5454, 0.8624 -> 9/10).
    ⛔ Not capped: the patch embed does not reduce, its modulus is linear in the inherited error,
    and at the net's input that error is 0. It is the one honest fold in ViT's chain."""
    A, E = st
    return (pe_mag(ic, P, wc, pb, A) + pe_roundErr(q, ic, P, wc, pb, A),
            pe_roundErr(q, ic, P, wc, pb, A) + pe_convMag(ic, P, wc, E))


def vit_chain(wa=VIT_WA, wm=VIT_WM, wp=VIT_WP, wh=VIT_WH, bb=VIT_BB,
              gl=VIT_GL, bl=VIT_BL, pb=VIT_PB,
              S=VIT_S, emr=VIT_EMR, ei=VIT_EI, eg=VIT_EG, eexp=VIT_EEXP, q=U32,
              k=K_VIT, ln_cap=True, gelu_sat=True, attn_mode='cap',
              uniform=False):
    """Every stage of `vitForwardKV` at ViT-Tiny's shapes, at the granularity a Lean `Maps`
    chain composes them.  Returns (rows, exp_tainted_tags) — the second is the list of stage
    numerals that would contain a `Real.exp` and therefore CANNOT BE WRITTEN, which is the
    softmax cap's actual justification (not the magnitude of the result).

    `uniform=True` reproduces ConvNeXt's mistake — one bound for every parameter kind."""
    if uniform:
        wa = wm = wp = wh = bb = gl = bl = pb = VIT_UNI

    def R(st):
        return (r4(st[0]), r4(st[1]))

    out, tainted = [], []
    taint = False        # once a Real.exp enters, every later numeral carries it …

    def emit(tag, st):
        out.append((tag, st))
        if taint:
            tainted.append(tag)

    def lnsite(st, tag, g_b, b_b):
        """… until a capped LN site, whose modulus is `2·mag` and depends only on the input
        WINDOW, resets it. That is why the taint is per-segment and not terminal."""
        nonlocal taint
        A, E = st
        mag = r4(vit_ln_leaf(A, S, emr, ei, q))
        if ln_cap:
            err = r4(2 * mag)
            taint = False                     # the cap discards the inherited error entirely
        else:
            nb = bnNormBudget(q, 2 * A, S, F(1), F(0), emr * A, ei)
            err = r4(nb + ((E + E) * S + 2 * A * (8 * A * E * F(158500000))))
        st = (mag, err); emit(tag + '.ln', st)
        st = R(cnx_diag(st, g_b, q)); emit(tag + '.lng', st)
        st = R(cnx_bias(st, b_b, q)); emit(tag + '.lnb', st)
        return st

    # ── patch embed: ONE leaf (conv + CLS branch + pos add, as the definition spells it) ──
    st = (F(1), F(0))
    st = R(vit_patch_embed(st, 3, 16, wp, pb, q)); emit('pe', st)

    # ── k transformer blocks, pre-norm, two residual sublayers each ──
    for i in range(k):
        t = f'b{i}'
        # attention sublayer:  X + Wo·(capped multi-head attention over LN(X))
        skip = st
        s = lnsite(st, t + '.a', gl, bl)
        amag, amod, a_taint = vit_attn(s, D_VIT, NTOK_VIT, wa, bb, eexp, q, attn_mode)
        if a_taint:
            taint = True
        s = (amag, amod);                 emit(t + '.attn', s)   # already rounded, see vit_attn
        s = R(dense(s, D_VIT, wa, bb));   emit(t + '.o', s)
        st = R(residual(skip, s, q));     emit(t + '.ares', st)
        # mlp sublayer:  X + fc2(gelu(fc1(LN(X))))
        skip = st
        s = lnsite(st, t + '.m', gl, bl)
        s = R(dense(s, D_VIT, wm, bb));   emit(t + '.fc1', s)
        s = R(cnx_gelu(s, eg, gelu_sat)); emit(t + '.ge', s)
        s = R(dense(s, MLP_VIT, wm, bb)); emit(t + '.fc2', s)
        st = R(residual(skip, s, q));     emit(t + '.mres', st)

    # ── final LayerNorm, CLS slice (exact, envelope-preserving), classifier ──
    st = lnsite(st, 'headln', gl, bl)
    emit('cls', st)
    st = R(dense(st, D_VIT, wh, bb)); emit('logits', st)
    return out, tainted


def verify_vit(rows, wa=VIT_WA, wm=VIT_WM, wp=VIT_WP, wh=VIT_WH, bb=VIT_BB,
               gl=VIT_GL, bl=VIT_BL, pb=VIT_PB,
               S=VIT_S, emr=VIT_EMR, ei=VIT_EI, eg=VIT_EG, eexp=VIT_EEXP, q=U32,
               k=K_VIT) -> int:
    """Re-assert EVERY rounded inequality a `ViTFloatBudget.lean` would close, exactly: each
    stage's emitted numeral must still dominate the exact value computed from the PREVIOUS
    stage's emitted numerals.  Returns the count checked; raises on the first failure.

    ⚠ This is the pass that catches folding with the exact `(1+u)^k − 1` instead of the ROUNDED
    `gamma_num` the Lean chain passes (§0).  Run it before emitting a single numeral."""
    r = dict(rows)
    n = 0

    def ck(tag, lhs, rhs):
        nonlocal n
        assert lhs <= rhs, f"{tag}: left > right"
        n += 1

    def conv_ck(tag, m, w, src, dst):
        A, E = src
        g = r4(gamma_q(m + 2, q))
        ck(tag + '.A', (1 + g) * (m * w * A + bb), dst[0])
        ck(tag + '.E', g * (m * w * (A + E) + bb) + m * w * E, dst[1])

    def ln_ck(tag, src):
        A, E = src
        ck(tag + '.ln.A', vit_ln_leaf(A, S, emr, ei, q), r[tag + '.ln'][0])
        ck(tag + '.ln.E', 2 * r[tag + '.ln'][0], r[tag + '.ln'][1])
        A1, E1 = r[tag + '.ln']
        me = mulErr(q, gl, A1, F(0), F(0))
        ck(tag + '.lng.A', gl * A1 + me, r[tag + '.lng'][0])
        ck(tag + '.lng.E', me + gl * E1, r[tag + '.lng'][1])
        A2, E2 = r[tag + '.lng']
        ck(tag + '.lnb.A', A2 + bl + q * (A2 + bl), r[tag + '.lnb'][0])
        ck(tag + '.lnb.E', q * (A2 + bl) + E2, r[tag + '.lnb'][1])
        return r[tag + '.lnb']

    # patch embed — ONE leaf (`Maps.patchEmbed`), stated on the exact `redErr` nest
    ck('pe.A', pe_mag(3, 16, wp, pb, F(1)) + pe_roundErr(q, 3, 16, wp, pb, F(1)), r['pe'][0])
    ck('pe.E', pe_roundErr(q, 3, 16, wp, pb, F(1)) + pe_convMag(3, 16, wp, F(0)), r['pe'][1])
    st = r['pe']

    for i in range(k):
        t = f'b{i}'
        # attention sublayer
        skip = st
        s = ln_ck(t + '.a', st)
        A, E = s
        g1 = r4(gamma_q(D_VIT + 2, q))
        g2 = r4(gamma_q(NTOK_VIT + 1, q))
        sc = r4(sm_cap(q, eexp, NTOK_VIT))
        vAF = (1 + g1) * (D_VIT * wa * A + bb)
        ck(t + '.attn.A', (1 + g2) * (F(NTOK_VIT) * ((1 + sc) * vAF)), r[t + '.attn'][0])
        ck(t + '.attn.E', 2 * r[t + '.attn'][0], r[t + '.attn'][1])   # `Maps.capped`
        conv_ck(t + '.o', D_VIT, wa, r[t + '.attn'], r[t + '.o'])
        A8, E8 = r[t + '.o']
        ck(t + '.ares.A', A8 + skip[0] + q * (A8 + skip[0]), r[t + '.ares'][0])
        ck(t + '.ares.E', q * (A8 + E8 + skip[0] + skip[1]) + (E8 + skip[1]),
           r[t + '.ares'][1])
        st = r[t + '.ares']
        # mlp sublayer
        skip = st
        s = ln_ck(t + '.m', st)
        conv_ck(t + '.fc1', D_VIT, wm, s, r[t + '.fc1'])
        A, E = r[t + '.fc1']
        ck(t + '.ge.A', A + eg, r[t + '.ge'][0])
        ck(t + '.ge.E', eg + F(3, 2) * E, r[t + '.ge'][1])            # the saturation branch
        conv_ck(t + '.fc2', MLP_VIT, wm, r[t + '.ge'], r[t + '.fc2'])
        A8, E8 = r[t + '.fc2']
        ck(t + '.mres.A', A8 + skip[0] + q * (A8 + skip[0]), r[t + '.mres'][0])
        ck(t + '.mres.E', q * (A8 + E8 + skip[0] + skip[1]) + (E8 + skip[1]),
           r[t + '.mres'][1])
        st = r[t + '.mres']

    st = ln_ck('headln', st)
    ck('cls.A', st[0], r['cls'][0])          # the CLS gather is exact
    ck('cls.E', st[1], r['cls'][1])
    conv_ck('logits', D_VIT, wh, r['cls'], r['logits'])
    return n


# ════════════════════════════════════════════════════════════════════════════
# ResNet-34 @224², the whole-net INPUT-GRADIENT VJP (`Proofs.r34InputGrad`)
# ════════════════════════════════════════════════════════════════════════════
#
# ⭐⭐ Phase 2, and the finding that reverses the plan's §3.7: **the backward is a FOLD, at
# TRAINING-mode BatchNorm — the mode the forward has no number for at all (1e7417).** As a map
# on the COTANGENT the BN backward is LINEAR: `floatClose_bnBack`'s modulus is
# `bnGradInputBudget(A) + bnGradInputReMag(e)` and `bnGradInputReMag` is linear in `e`. §0.1's
# quadratic came from the statistics MOVING with the input; a VJP's statistics are read off the
# saved activations, which the cotangent does not perturb. So no cap is needed and
# `budget / window` is 0.048, not 2.00.
#
# ⛔ THE WALL DOES NOT VANISH — IT RELOCATES. The saved float activations enter as SUPPLIED
# accuracies (`es` on the inverse-stddev, `exh` on the normalised activation), taken at 1e-2 like
# every other device-kernel accuracy. Unlike `DeviceRsqrt`'s, those are quantities the repo's
# forward fold does speak about, and at training-mode BN it says 1e7417, not 1e-2. So this number
# is an honest fold GIVEN a forward-accuracy hypothesis its own forward cannot discharge, and it
# must be said that way (planning/float_budget_numbers.md §3.7, §9).
#
# ⭐⭐ The load-bearing lemma is `bnXhat_sq_le` — `|x̂| ≤ √n`, the standardisation bound — and it
# was ALREADY IN THE REPO, proved for the "realistic seal" work in `Foundation/ResNet34.lean` and
# named after neither the float tier nor the backward. With it the fold is 1e288; deriving `Xh`
# from the FORWARD's certified window instead (`2·A·S`) gives 1e7271, unstatable. That is
# §3.3.0(b)'s lesson for the fourth time: before writing a bound, grep the whole repo for it.
#
# Profile, measured per parameter KIND on /home/skoonce/resnet/r34_imagenet_bf16_e79.bin
# (21,797,672 f32, the layout `init_params_from_file` spells):
#
#     kind                 count       max|·|    bound
#     conv/dense kernels   21,779,648  1.1007    12/10
#     BN γ                 8,512       2.0741    21/10
#     BN β                 8,512       0.8118    —      (the backward has no bias anywhere)
#     dense bias           1,000       0.0475    —
#
# ⚠ The forward's uniform 21/10 is a BN γ, and it is 1.9× loose on the 21.78 M entries every
# conv fan-in multiplies. Splitting buys 8 orders here (1e296 -> 1e288). Unlike ConvNeXt's split
# it is not the difference between statable and not — it is the difference between 4 orders of
# headroom under `norm_num`'s ceiling and 12.

R34_WK = F(12, 10)      # conv / dense kernels          measured 1.1007
R34_GL = F(21, 10)      # BN γ                          measured 2.0741
R34_ESB = F(1, 100)     # supplied float inverse-stddev accuracy
R34_EXH = F(1, 100)     # supplied float normalised-activation accuracy   ⛔ see the note above

# The r34 backward's shape plan, cotangent-first (`r34InputGradF` reads right to left).
R34_BACK_PLAN = [
    ("id", 512, 512, 49, "e1"), ("id", 512, 512, 49, "e0"), ("down", 256, 512, 49, "d4"),
    ("id", 256, 256, 196, "c4"), ("id", 256, 256, 196, "c3"), ("id", 256, 256, 196, "c2"),
    ("id", 256, 256, 196, "c1"), ("id", 256, 256, 196, "c0"), ("down", 128, 256, 196, "d3"),
    ("id", 128, 128, 784, "b2"), ("id", 128, 128, 784, "b1"), ("id", 128, 128, 784, "b0"),
    ("down", 64, 128, 784, "d2"),
    ("id", 64, 64, 3136, "a2"), ("id", 64, 64, 3136, "a1"), ("id", 64, 64, 3136, "a0"),
]


def isqrt_exact(n: int) -> int:
    """`√n` for the perfect squares the plan uses (h·w with h = w)."""
    import math
    r = math.isqrt(n)
    assert r * r == n, n
    return r


def bnGradInputReMag(n, G, Cdy, S, Xh):
    """`bnGradInputReMag` (`Codegen/BnBackComposeBridge.lean`) — the REAL BN input-gradient's
    magnitude at cotangent bound `Cdy`. ⭐ LINEAR in `Cdy`, which is the whole reason a backward
    fold exists where the forward's does not: it is the Lipschitz constant of a linear map."""
    N = F(n)
    return F(1) / N * S * (N * (G * Cdy) + N * (G * Cdy) + Xh * (N * (Xh * (G * Cdy))))


def bnGradInputBudgetQ(n, G, Cdy, S, Xh, es, exh, q=U32):
    """`FloatModel.bnGradInputBudget` with the exact `(1+u)^(n+1) − 1` replaced by the ROUNDED
    `gamma_num` bound the Lean chain passes (§0's ⚠). Reductions here run to n = 12544, so the
    Lean leaf will need a `peRoundErrQ`-style rational restatement — this is that restatement."""
    u = q
    gn = r4(gamma_q(n + 1, q))
    N = F(n)
    MD = G * Cdy
    eD = mulErr(u, G, Cdy, F(0), F(0))
    eSD = gn * (N * (MD + eD)) + N * eD
    MSD = N * MD + eSD
    MXD = Xh * MD
    eXD = mulErr(u, Xh, MD, exh, eD)
    eSXD = gn * (N * (MXD + eXD)) + N * eXD
    enD = mulErr(u, N, MD, F(0), eD)
    MnD = N * MD + enD
    e1 = u * (MnD + MSD) + (enD + eSD)
    M1 = (1 + u) * (MnD + MSD)
    eXS = mulErr(u, Xh, N * MXD, exh, eSXD)
    MXSf = Xh * (N * MXD) + eXS
    e2 = u * (M1 + MXSf) + (e1 + eXS)
    MTr = N * MD + N * MD + Xh * (N * MXD)
    eP = mulErr(u, F(1) / N, S, F(0), es)
    return mulErr(u, (F(1) / N) * S, MTr, eP, e2)


def bn_back_gain(n, Xh, G=R34_GL, S=R34_S, es=R34_ESB, exh=R34_EXH, q=U32):
    """⭐⭐ The site's two PER-UNIT constants, rounded up: the real map's Lipschitz constant and
    the rounding budget, both at cotangent window 1 (`Maps.bnPerChannelBackGain`'s `Kr`/`Kb`).

    Both `bnGradInputReMag` and `bnGradInputBudgetQ` are HOMOGENEOUS OF DEGREE 1 in the cotangent
    bound — the same linearity that makes a backward fold exist at all — so a site's whole
    envelope is `A ↦ A·(Kr+Kb)` and `(A,E) ↦ A·Kb + E·Kr`. ⭐ That is what makes the numerals
    checkable: stated directly, each site is a forty-node tree at the chain's full magnitude and
    `norm_num` needs seconds per site; factored, the expensive evaluation happens once per
    feature-map SIZE — five times for a ResNet-34, not sixty-eight."""
    return (r4(bnGradInputReMag(n, G, F(1), S, Xh)),
            r4(bnGradInputBudgetQ(n, G, F(1), S, Xh, es, exh, q)))


def bn_back(st, n, Xh, G=R34_GL, S=R34_S, es=R34_ESB, exh=R34_EXH, q=U32):
    """One per-channel BatchNorm BACKWARD site (`floatClose_bnBack`), over the cotangent.
    ⚠ Folded at the ROUNDED per-unit gains, because that is what the Lean chain passes (§0)."""
    A, E = st
    Kr, Kb = bn_back_gain(n, Xh, G, S, es, exh, q)
    return (A * (Kr + Kb), A * Kb + E * Kr)


def maxpool3s2_back(st, n, q=U32):
    """⛔ `floatClose_maxPool3s2Back` — He et al.'s 3×3/s2 stem pool's backward, the ACCUMULATING
    scatter. `maxPool2`'s windows tile so its backward is an exact lookup and its envelope is the
    identity; 3×3/s2 windows OVERLAP, so an input can be the argmax of up to FOUR outputs and the
    backward is a rounded reduction — window `4A(1+γ)`, modulus `γ·4A + 4E`, at `γ` over the
    `n = c·h·w` terms the kernel reduces over.

    ⛔ `r34InputGrad` used the 2×2 peer until 2026-09-03, so this stage was missing entirely and
    the whole-net backward was the reverse of a net the repo does not train (§3.10)."""
    A, E = st
    g = r4(gamma_q(n + 1, q))
    return (4 * A + g * (4 * A), g * (4 * A) + 4 * E)


def conv_back(st, m, w=R34_WK, q=U32):
    """`convFlatBack W = flatConv (reverseSwap W) 0` — the forward conv leaf at zero bias,
    fan-in `m` = (the cotangent's channel count)·kH·kW."""
    return conv(st, m, w, F(0), q)


def gap_back(st, hw, q=U32):
    """`gapBack c h w` — broadcast ÷ (h·w). One rounded multiply; magnitude-NONincreasing."""
    A, E = st
    inv = F(1) / F(hw)
    me = mulErr(q, inv, A, F(0), F(0))
    return (inv * A + me, me + inv * E)


def r34_back_chain(wk=R34_WK, G=R34_GL, S=R34_S, es=R34_ESB, exh=R34_EXH, q=U32,
                   xhat='sqrt'):
    """Every stage of `r34InputGrad` at r34's shapes, at the granularity a Lean `Maps` chain
    composes them, folded over the LOSS COTANGENT (`|p − y| ≤ 1` for softmax cross-entropy).

    `reluMaskBack`, `maxPoolFlatBack` and `decimateBack` are structural selects/scatters, exact
    in float, and produce no entry — the envelope passes through them unchanged.

    `xhat='sqrt'`   : `|x̂| ≤ √(h·w)`, the standardisation bound (`bnXhat_sq_le`).
    `xhat='window'` : `|x̂| ≤ 2·A·S` from the FORWARD's certified window — the ablation that
                      shows which of the two is load-bearing."""
    fwd = dict(r34_eval_chain())

    def Xh(hw, fwd_tag):
        return F(isqrt_exact(hw)) if xhat == 'sqrt' else 2 * fwd[fwd_tag][0] * S

    def R(st):
        return (r4(st[0]), r4(st[1]))

    out = []
    st = (F(1), F(0))
    st = R(conv_back(st, 10, wk, q));          out.append(("linBack", st))
    st = R(gap_back(st, 49, q));               out.append(("gapBack", st))
    for kind, ic, oc, hw, tag in R34_BACK_PLAN:
        blkin = st                              # reluMaskBack m_out: exact
        if kind == "id":
            s = R(bn_back(blkin, hw, Xh(hw, tag + ".bn2"), G, S, es, exh, q))
            out.append((tag + ".bnB2", s))
            s = R(conv_back(s, oc * 9, wk, q)); out.append((tag + ".cB2", s))
            s = R(bn_back(s, hw, Xh(hw, tag + ".bn1"), G, S, es, exh, q))
            out.append((tag + ".bnB1", s))
            s = R(conv_back(s, oc * 9, wk, q)); out.append((tag + ".cB1", s))
            st = R(residual(blkin, s, q));      out.append((tag + ".out", st))
        else:
            p = R(bn_back(blkin, hw, Xh(hw, tag + ".projbn"), G, S, es, exh, q))
            out.append((tag + ".bnBp", p))
            p = R(conv_back(p, oc, wk, q));     out.append((tag + ".cBp", p))
            s = R(bn_back(blkin, hw, Xh(hw, tag + ".bn2"), G, S, es, exh, q))
            out.append((tag + ".bnB2", s))
            s = R(conv_back(s, oc * 9, wk, q)); out.append((tag + ".cB2", s))
            s = R(bn_back(s, hw, Xh(hw, tag + ".bn1"), G, S, es, exh, q))
            out.append((tag + ".bnB1", s))
            s = R(conv_back(s, oc * 9, wk, q)); out.append((tag + ".cB1", s))
            st = R(bipath(p, s, q));            out.append((tag + ".out", st))
    # ⛔ the 3×3/s2 stem pool's backward ACCUMULATES (×4); then the stem:
    # reluMaskBack (exact) -> bnB -> flatConvStride2Back
    st = R(maxpool3s2_back(st, 64 * 56 * 56, q));  out.append(("mp3s2B", st))
    st = R(bn_back(st, 12544, Xh(12544, "stem.bn"), G, S, es, exh, q))
    out.append(("stem.bnB", st))
    st = R(conv_back(st, 64 * 49, wk, q));      out.append(("stem.cB", st))
    return out


def verify_r34_back(rows, wk=R34_WK, G=R34_GL, S=R34_S, es=R34_ESB, exh=R34_EXH,
                    q=U32) -> int:
    """Re-assert EVERY rounded inequality a `Resnet34BackFloatBudget.lean` would close, exactly:
    each stage's emitted numeral must still dominate the exact value computed from the PREVIOUS
    stage's emitted numerals. Returns the count checked; raises on the first failure."""
    r = dict(rows)
    n = 0

    def ck(tag, lhs, rhs):
        nonlocal n
        assert lhs <= rhs, f"{tag}: left > right"
        n += 1

    def conv_ck(tag, m, src, dst):
        A, E = src
        g = r4(gamma_q(m + 2, q))
        ck(tag + '.A', (1 + g) * (m * wk * A + 0), dst[0])
        ck(tag + '.E', g * (m * wk * (A + E) + 0) + m * wk * E, dst[1])

    def bn_ck(tag, hw, src, dst):
        A, E = src
        Xh = F(isqrt_exact(hw))
        Kr, Kb = bn_back_gain(hw, Xh, G, S, es, exh, q)
        # the two side conditions the gain form carries, at cotangent window 1
        ck(tag + '.Kr', bnGradInputReMag(hw, G, F(1), S, Xh), Kr)
        ck(tag + '.Kb', bnGradInputBudgetQ(hw, G, F(1), S, Xh, es, exh, q), Kb)
        ck(tag + '.A', A * (Kr + Kb), dst[0])
        ck(tag + '.E', A * Kb + E * Kr, dst[1])

    conv_ck('linBack', 10, (F(1), F(0)), r['linBack'])
    A, E = r['linBack']
    inv = F(1) / F(49)
    me = mulErr(q, inv, A, F(0), F(0))
    ck('gapBack.A', inv * A + me, r['gapBack'][0])
    ck('gapBack.E', me + inv * E, r['gapBack'][1])
    st = r['gapBack']
    for kind, ic, oc, hw, tag in R34_BACK_PLAN:
        skip = st
        if kind == "id":
            bn_ck(tag + '.bnB2', hw, skip, r[tag + '.bnB2'])
            conv_ck(tag + '.cB2', oc * 9, r[tag + '.bnB2'], r[tag + '.cB2'])
            bn_ck(tag + '.bnB1', hw, r[tag + '.cB2'], r[tag + '.bnB1'])
            conv_ck(tag + '.cB1', oc * 9, r[tag + '.bnB1'], r[tag + '.cB1'])
            Bd, Ed = r[tag + '.cB1']
            ck(tag + '.out.A', Bd + skip[0] + q * (Bd + skip[0]), r[tag + '.out'][0])
            ck(tag + '.out.E', q * (Bd + Ed + skip[0] + skip[1]) + (Ed + skip[1]),
               r[tag + '.out'][1])
        else:
            bn_ck(tag + '.bnBp', hw, skip, r[tag + '.bnBp'])
            conv_ck(tag + '.cBp', oc, r[tag + '.bnBp'], r[tag + '.cBp'])
            bn_ck(tag + '.bnB2', hw, skip, r[tag + '.bnB2'])
            conv_ck(tag + '.cB2', oc * 9, r[tag + '.bnB2'], r[tag + '.cB2'])
            bn_ck(tag + '.bnB1', hw, r[tag + '.cB2'], r[tag + '.bnB1'])
            conv_ck(tag + '.cB1', oc * 9, r[tag + '.bnB1'], r[tag + '.cB1'])
            Pd, Ep = r[tag + '.cBp']
            Bd, Ed = r[tag + '.cB1']
            ck(tag + '.out.A', Pd + Bd + q * (Pd + Bd), r[tag + '.out'][0])
            ck(tag + '.out.E', q * (Pd + Ep + Bd + Ed) + (Ep + Ed), r[tag + '.out'][1])
        st = r[tag + '.out']
    A, E = st
    gmp = r4(gamma_q(64 * 56 * 56 + 1, q))
    ck('mp3s2B.A', 4 * A + gmp * (4 * A), r['mp3s2B'][0])
    ck('mp3s2B.E', gmp * (4 * A) + 4 * E, r['mp3s2B'][1])
    bn_ck('stem.bnB', 12544, r['mp3s2B'], r['stem.bnB'])
    conv_ck('stem.cB', 64 * 49, r['stem.bnB'], r['stem.cB'])
    return n


# ════════════════════════════════════════════════════════════════════════════
# MobileNetV2 BACKWARD — the input-gradient VJP over the loss cotangent
# ════════════════════════════════════════════════════════════════════════════
#
# `mnv2InputGrad` (`MobileNetV2BackFloatBridge.lean`): the exact reverse of
# `mobilenetv2Forward_full_pc`. The r34 backward's story transfers WITHOUT CHANGE — every
# leaf is linear in the cotangent, the BN backs read their statistics off saved activations,
# and `reluMaskBack` (the relu6 kink at the smooth point `0 < preact < 6`) is a structural
# select: exact in float, envelope-preserving. ⭐ Note what that costs: relu6's CLAMP, which
# is the whole reason mnv2's forward window is 2154 instead of 1e100 (§3.2), buys the
# backward NOTHING — a 0/1 mask cannot reset a cotangent window the way `min A 6` resets an
# activation window. Window and budget are separate levers on the forward; on the backward
# the clamp is not a lever at all.
#
# Profile, measured per parameter KIND on /home/skoonce/mnv2_350ep/mobilenet_v2_imagenet.bin
# (3,504,872 f32, the layout `init_params_from_file` spells):
#
#     kind                 count       max|·|    bound
#     conv/dense kernels   3,469,760   2.7157    28/10
#     BN γ                 17,056      1.6869    17/10
#     BN β                 17,056      1.6406    —      (the backward has no bias anywhere)
#     dense bias           1,000       0.1029    —
#
# ⚠ The split runs the OPPOSITE way from ResNet-34's: there the uniform bound was a BN γ and
# the kernels were 1.9× tighter, here the maximum IS a kernel and it is γ that is 1.6× loose.
# So the split buys the BN gain and not the conv fan-in, and it is worth ~4 orders, not 8.

MNV2_WK = F(28, 10)     # conv / dense kernels         measured 2.7157
MNV2_GLB = F(17, 10)    # BN γ                         measured 1.6869
MNV2_SB = F(16)         # |istd| at the operating point (σ ≥ 1/16) — §3.7's escape 2
MNV2_ESB = F(1, 100)    # supplied float inverse-stddev accuracy
MNV2_EXH = F(1, 100)    # supplied float normalised-activation accuracy

# (tag, kind, ic, mid, oc, h, w), cotangent-first. `h`/`w` are the block's OUTPUT spatial dims;
# a "strided" block's expand stage runs at 2h × 2w (`invresBodyStridedBackPC`).
MNV2_BACK_PLAN = [
    ("b6", "strided", 64, 256, 64, 7, 7),
    ("b5", "strided", 32, 128, 64, 14, 14),
    ("b4", "skip", 32, 128, 32, 28, 28),
    ("b3", "strided", 24, 96, 32, 28, 28),
    ("b2", "skip", 24, 96, 24, 56, 56),
    ("b1", "strided", 16, 64, 24, 56, 56),
]


def mnv2_back_chain(wk=MNV2_WK, G=MNV2_GLB, S=MNV2_SB, es=MNV2_ESB, exh=MNV2_EXH, q=U32,
                    xhat='sqrt'):
    """Every stage of `mnv2InputGrad` at MobileNetV2's shapes, folded over the LOSS COTANGENT
    (`|p − y| ≤ 1`), at the granularity a Lean `Maps` chain composes them.

    `reluMaskBack` and `decimateBack` are exact structural selects/scatters and produce no
    entry. `xhat='sqrt'` is `bnXhat_sq_le`'s `|x̂| ≤ √(h·w)`; `xhat='window'` derives it from
    the forward's certified window (`2·A·S`) instead — the r34 ablation, repeated."""
    fwd = dict(mnv2_eval_chain())

    def Xh(hw, fwd_tag):
        return F(isqrt_exact(hw)) if xhat == 'sqrt' else 2 * fwd[fwd_tag][0] * S

    def R(st):
        return (r4(st[0]), r4(st[1]))

    out = []
    st = (F(1), F(0))
    st = R(conv_back(st, 10, wk, q));           out.append(("linBack", st))
    st = R(gap_back(st, 49, q));                out.append(("gapBack", st))
    # head: reluMaskBack (exact) -> bnBh (128ch @ 7×7) -> convFlatBack Wh (1×1, fan-in 128)
    st = R(bn_back(st, 49, Xh(49, "head.bn"), G, S, es, exh, q))
    out.append(("head.bnB", st))
    st = R(conv_back(st, 128, wk, q));          out.append(("head.cB", st))
    for tag, kind, ic, mid, oc, h, w in MNV2_BACK_PLAN:
        blkin = st
        hw = h * w
        he = (2 * h) * (2 * w) if kind == "strided" else hw
        # project back: bnBp (oc @ h×w) then convFlatBack Wp (1×1, fan-in oc)
        s = R(bn_back(blkin, hw, Xh(hw, tag + ".pbn"), G, S, es, exh, q))
        out.append((tag + ".bnBp", s))
        s = R(conv_back(s, oc, wk, q));         out.append((tag + ".cBp", s))
        # depthwise back: reluMaskBack (exact) -> bnBd (mid @ h×w) -> depthwiseFlatBack (fan-in 9)
        s = R(bn_back(s, hw, Xh(hw, tag + ".dbn"), G, S, es, exh, q))
        out.append((tag + ".bnBd", s))
        s = R(conv_back(s, 9, wk, q));          out.append((tag + ".dwB", s))
        # expand back: reluMaskBack (exact) -> bnBe (mid @ he) -> convFlatBack We (fan-in mid)
        s = R(bn_back(s, he, Xh(he, tag + ".ebn"), G, S, es, exh, q))
        out.append((tag + ".bnBe", s))
        s = R(conv_back(s, mid, wk, q));        out.append((tag + ".cBe", s))
        st = R(residual(blkin, s, q)) if kind == "skip" else s
        out.append((tag + ".out", st))
    # stem: reluMaskBack (exact) -> bnBs (16ch @ 112×112) -> flatConvStride2Back Ws (fan-in 16·9)
    st = R(bn_back(st, 12544, Xh(12544, "stem.bn"), G, S, es, exh, q))
    out.append(("stem.bnB", st))
    st = R(conv_back(st, 16 * 9, wk, q));       out.append(("stem.cB", st))
    return out


def verify_mnv2_back(rows, wk=MNV2_WK, G=MNV2_GLB, S=MNV2_SB, es=MNV2_ESB, exh=MNV2_EXH,
                     q=U32) -> int:
    """Re-assert EVERY rounded inequality a `MobileNetV2BackFloatBudget.lean` would close,
    exactly — the peer of `verify_r34_back`. Returns the count checked; raises on the first
    failure."""
    r = dict(rows)
    n = 0

    def ck(tag, lhs, rhs):
        nonlocal n
        assert lhs <= rhs, f"{tag}: left > right"
        n += 1

    def conv_ck(tag, m, src, dst):
        A, E = src
        g = r4(gamma_q(m + 2, q))
        ck(tag + '.A', (1 + g) * (m * wk * A + 0), dst[0])
        ck(tag + '.E', g * (m * wk * (A + E) + 0) + m * wk * E, dst[1])

    def bn_ck(tag, hw, src, dst):
        A, E = src
        Xh = F(isqrt_exact(hw))
        Kr, Kb = bn_back_gain(hw, Xh, G, S, es, exh, q)
        ck(tag + '.Kr', bnGradInputReMag(hw, G, F(1), S, Xh), Kr)
        ck(tag + '.Kb', bnGradInputBudgetQ(hw, G, F(1), S, Xh, es, exh, q), Kb)
        ck(tag + '.A', A * (Kr + Kb), dst[0])
        ck(tag + '.E', A * Kb + E * Kr, dst[1])

    conv_ck('linBack', 10, (F(1), F(0)), r['linBack'])
    A, E = r['linBack']
    inv = F(1) / F(49)
    me = mulErr(q, inv, A, F(0), F(0))
    ck('gapBack.A', inv * A + me, r['gapBack'][0])
    ck('gapBack.E', me + inv * E, r['gapBack'][1])
    bn_ck('head.bnB', 49, r['gapBack'], r['head.bnB'])
    conv_ck('head.cB', 128, r['head.bnB'], r['head.cB'])
    st = r['head.cB']
    for tag, kind, ic, mid, oc, h, w in MNV2_BACK_PLAN:
        blkin = st
        hw = h * w
        he = (2 * h) * (2 * w) if kind == "strided" else hw
        bn_ck(tag + '.bnBp', hw, blkin, r[tag + '.bnBp'])
        conv_ck(tag + '.cBp', oc, r[tag + '.bnBp'], r[tag + '.cBp'])
        bn_ck(tag + '.bnBd', hw, r[tag + '.cBp'], r[tag + '.bnBd'])
        conv_ck(tag + '.dwB', 9, r[tag + '.bnBd'], r[tag + '.dwB'])
        bn_ck(tag + '.bnBe', he, r[tag + '.dwB'], r[tag + '.bnBe'])
        conv_ck(tag + '.cBe', mid, r[tag + '.bnBe'], r[tag + '.cBe'])
        if kind == "skip":
            Bd, Ed = r[tag + '.cBe']
            ck(tag + '.out.A', Bd + blkin[0] + q * (Bd + blkin[0]), r[tag + '.out'][0])
            ck(tag + '.out.E', q * (Bd + Ed + blkin[0] + blkin[1]) + (Ed + blkin[1]),
               r[tag + '.out'][1])
        else:
            ck(tag + '.out.A', r[tag + '.cBe'][0], r[tag + '.out'][0])
            ck(tag + '.out.E', r[tag + '.cBe'][1], r[tag + '.out'][1])
        st = r[tag + '.out']
    bn_ck('stem.bnB', 12544, st, r['stem.bnB'])
    conv_ck('stem.cB', 16 * 9, r['stem.bnB'], r['stem.cB'])
    return n


# ════════════════════════════════════════════════════════════════════════════
# EfficientNet-B0 BACKWARD — the squeeze-excite question (§3.8 item 1)
# ════════════════════════════════════════════════════════════════════════════
#
# ⭐⭐ THE STRUCTURAL ANSWER, and it is the one the r34 result predicts: SE's backward is
# LINEAR in the cotangent, so a backward really is always a fold. `seInputGrad`
# (`SEBackFloatBridge.lean`) is
#
#     seBack(dy) = diagBack g dy  +  gateBack (diagBack xinp dy)
#
# — a `biPathSum` of two LINEAR maps, because the gate `g = gate(x)` and the input `x` are
# SAVED CONSTANTS. §0.1's forward quadratic came from the gate being grown out of the same
# input the rescale multiplies; on the backward that input is not the cotangent. Nothing in
# the SE backward is quadratic, and `budget/window` stays a fold ratio.
#
# ⛔⛔ AND IT IS NOT STATABLE ANYWAY — for a reason r34's backward does not have. Both of the
# SE backward's saved constants, AND the swish backward's saved derivative, are magnitudes the
# forward window has to supply:
#
#   * `diagBack xinp` scales by `Sx = |x|`, the SE block's own input activation. There is no
#     `bnXhat_sq_le` here: `x` is a post-swish activation, not a normalised one, so the only
#     available bound is the forward's certified window (7.6e9 / 5.5e24 / 4.9e40 at b1/b2/b3).
#   * `swBd`/`swBe`/`swBs`/`swBh` scale by `Ssw = |swish'(preact)|`, and the repo's only bound
#     is `swishScalar_lipschitz_abs`'s WINDOW-DEPENDENT `1 + A/4` — 1.2e51 at the head.
#     ⭐ The global constant `|swish'| ≤ 1.1` is NOT proved (§3.4 says so explicitly); it is
#     GELU's `geluScalarDeriv_abs_le` for the other smooth activation, and proving it is the
#     single highest-value lemma this probe found.
#
# So the question §3.8 asked ("is a backward always a fold?") answers YES, and the question it
# did not ask ("is a backward always STATABLE?") answers NO — and the two failure modes are
# §0.1's two, a third time: this one is MAGNITUDE, and it is imported from the forward.
#
# Profile, per KIND on /home/skoonce/enet_b0_350_4gpu/efficientnet_b0_imagenet.bin
# (5,288,548 f32):
#
#     kind                 count       max|·|    bound
#     conv/dense kernels   5,236,192   3.6857    37/10
#     BN γ                 21,008      4.0545    41/10
#     BN β                 21,008      2.5103    —      (no bias anywhere in the backward)
#     SE dense + fc bias   10,340      2.5185    —
#
# ⚠ Here the split runs r34's way — the uniform 41/10 is a BN γ, 1.1× loose on the 5.24 M
# entries every conv fan-in multiplies — but at 1.1× it is worth under an order.
#
# ⭐ `batchMap` never enters a numeral (`Maps.batchMap` is the identity on the envelope), so
# this fold is per-example and holds at any batch size `N`, exactly like the forward's.

B0_WK = F(37, 10)       # conv / dense kernels         measured 3.6857
B0_GLB = F(41, 10)      # BN γ                         measured 4.0545
B0_SB = F(16)           # |istd| at the operating point
B0_ESB = F(1, 100)      # supplied float inverse-stddev accuracy
B0_EXH = F(1, 100)      # supplied float normalised-activation accuracy
B0_ESAV = F(1, 100)     # supplied accuracy on EVERY saved vector the backward scales by
                        # (the gate `g`, the SE input `x`, `σ'(saved)`, `swish'(saved)`)
B0_SSIG = F(1, 4)       # |σ'| = |σ(1−σ)| ≤ 1/4
B0_SSW_TRUE = F(11, 10) # the TRUE global |swish'| — ⛔ NOT proved in the repo (see above)

# (tag, kind, cin, cmid, cout, h, w, kd, se_c, se_r), cotangent-first. `h`/`w` are the block's
# OUTPUT spatial dims; a "strided" block's expand stage runs at 2h × 2w. `se_c` is the SE's
# channel count (= cmid, or cin for the no-expand block) and `se_r` its reduced width.
B0_BACK_PLAN = [
    ("b3", "resid", 24, 144, 24, 56, 56, 25, 144, 6),
    ("b2", "strided", 16, 96, 24, 56, 56, 9, 96, 4),
    ("b1", "noexp", 32, 32, 16, 112, 112, 9, 32, 8),
]


def diag_back(st, Sd, esav, q=U32):
    """`floatClose_diagBack` — the saved-vector pointwise scale. Covers the swish/sigmoid
    backward (`s = act'(saved preact)`) and both of `seInputGrad`'s two saved multipliers."""
    A, E = st
    me = mulErr(q, Sd, A, esav, F(0))
    return (Sd * A + me, me + Sd * E)


def broadcast_back(st, N, nnz=None, q=U32):
    """`floatClose_broadcastBack` — the SE gate's spatial reduce (`Vec (c·h·w) → Vec c`, each
    channel summing its own `h·w` cells).

    ⚠ `nnz` is the ablation this leaf deserves. As PROVED the bound charges all `c·h·w` terms
    (`hsumabs` bounds every masked entry by `A`, including the `(c−1)·h·w` that are identically
    ZERO), so the window carries a spurious factor of `c`. `nnz = h·w` is the honest count."""
    A, E = st
    if nnz is None:
        nnz = N
    g = r4(gamma_q(N + 1, q))
    NN = F(nnz)
    return (NN * A + g * (NN * A), g * (NN * (A + E)) + NN * E)


def se_back(st, out, tag, se_c, hw, se_r, Sx, Sg, Ssw, w=B0_WK, esav=B0_ESAV,
            Ssig=B0_SSIG, q=U32, nnz=None):
    """`seInputGrad g xinp gateBack = biPathSum (diagBack g) (gateBack ∘ diagBack xinp)`, with
    `gateBack = gapBack ∘ linBack W₁ ∘ diagBack ssw ∘ linBack W₂ ∘ diagBack ssig ∘ broadcastBack`
    (`floatBridges_seGateBack`, the exact reverse of the gate's six forward stages).

    ⭐ BOTH branches are linear in the cotangent — the gate and the input are saved constants —
    which is the whole answer to §3.8's question. What the gate path costs is not nonlinearity
    but MAGNITUDE: `Sx` is the block's saved input activation, and `broadcastBack` multiplies by
    the reduce's fan-in before `gapBack` divides it back out."""
    def R(s):
        return (r4(s[0]), r4(s[1]))

    main = R(diag_back(st, Sg, esav, q));           out.append((tag + ".se.main", main))
    p = R(diag_back(st, Sx, esav, q));              out.append((tag + ".se.pre", p))
    p = R(broadcast_back(p, se_c * hw, nnz, q));    out.append((tag + ".se.bc", p))
    p = R(diag_back(p, Ssig, esav, q));             out.append((tag + ".se.sig", p))
    p = R(conv(p, se_c, w, F(0), q));               out.append((tag + ".se.d2", p))
    p = R(diag_back(p, Ssw, esav, q));              out.append((tag + ".se.sw", p))
    p = R(conv(p, se_r, w, F(0), q));               out.append((tag + ".se.d1", p))
    p = R(gap_back(p, hw, q));                      out.append((tag + ".se.gap", p))
    st = R(bipath(main, p, q));                     out.append((tag + ".se.out", st))
    return st


def b0_back_chain(wk=B0_WK, G=B0_GLB, S=B0_SB, es=B0_ESB, exh=B0_EXH, esav=B0_ESAV,
                  q=U32, xhat='sqrt', ssw='window', sx='window', se_nnz=False):
    """Every stage of `efficientnetInputGradB` at B0's shapes, folded over the LOSS COTANGENT.

    `ssw` / `sx` are the two forward-window imports this probe exists to measure:
      `ssw='window'` : `|swish'| ≤ 1 + A/4` at the forward's pre-swish window — what the repo
                       proves today (`swishScalar_lipschitz_abs`).
      `ssw=<rat>`    : a global constant (11/10 is the true one; ⛔ NOT proved).
      `sx='window'`  : the SE's saved input bounded by the forward's certified window.
      `sx=<rat>`     : an operating-point bound, §3.7's `|istd| ≤ 16` one op over.
    `se_nnz=True` tightens `broadcastBack` to its `h·w` nonzero terms."""
    fwd = dict(b0_eval_chain())

    def Xh(hw, fwd_tag):
        return F(isqrt_exact(hw)) if xhat == 'sqrt' else 2 * fwd[fwd_tag][0] * S

    def Ssw(fwd_tag):
        return (1 + fwd[fwd_tag][0] / 4) if ssw == 'window' else ssw

    def Sx(fwd_tag):
        return fwd[fwd_tag][0] if sx == 'window' else sx

    def R(st):
        return (r4(st[0]), r4(st[1]))

    out = []
    st = (F(1), F(0))
    st = R(conv_back(st, 10, wk, q));            out.append(("linBack", st))
    st = R(gap_back(st, 56 * 56, q));            out.append(("gapBack", st))
    # head: swBh -> bnBh (1280ch @ 56×56) -> convFlatBack Wh (1×1, fan-in 1280)
    st = R(diag_back(st, Ssw("head.bn"), esav, q));  out.append(("head.swB", st))
    st = R(bn_back(st, 3136, Xh(3136, "head.bn"), G, S, es, exh, q))
    out.append(("head.bnB", st))
    st = R(conv_back(st, 1280, wk, q));          out.append(("head.cB", st))
    for tag, kind, cin, cmid, cout, h, w, kd, se_c, se_r in B0_BACK_PLAN:
        blkin = st
        hw = h * w
        he = (2 * h) * (2 * w) if kind == "strided" else hw
        # project back: bnBp (cout @ h×w) then convFlatBack Wp (1×1, fan-in cout)
        s = R(bn_back(blkin, hw, Xh(hw, tag + ".pbn"), G, S, es, exh, q))
        out.append((tag + ".bnBp", s))
        s = R(conv_back(s, cout, wk, q));        out.append((tag + ".cBp", s))
        # the squeeze-excite product-rule backward
        s = se_back(s, out, tag, se_c, hw, se_r, Sx(tag + ".dswish"), 1 + B0_ESIG,
                    Ssw(tag + ".sd1"), wk, esav, B0_SSIG, q,
                    nnz=(hw if se_nnz else None))
        # depthwise back: swBd -> bnBd (cmid @ h×w) -> depthwiseFlatBack (fan-in kd)
        s = R(diag_back(s, Ssw(tag + ".dbn"), esav, q));  out.append((tag + ".swBd", s))
        s = R(bn_back(s, hw, Xh(hw, tag + ".dbn"), G, S, es, exh, q))
        out.append((tag + ".bnBd", s))
        s = R(conv_back(s, kd, wk, q));          out.append((tag + ".dwB", s))
        if kind != "noexp":
            # expand back: swBe -> bnBe (cmid @ he) -> convFlatBack We (fan-in cmid)
            s = R(diag_back(s, Ssw(tag + ".ebn"), esav, q));  out.append((tag + ".swBe", s))
            s = R(bn_back(s, he, Xh(he, tag + ".ebn"), G, S, es, exh, q))
            out.append((tag + ".bnBe", s))
            s = R(conv_back(s, cmid, wk, q));    out.append((tag + ".cBe", s))
        st = R(residual(blkin, s, q)) if kind == "resid" else s
        out.append((tag + ".out", st))
    # stem: swBs -> bnBs (32ch @ 112×112) -> flatConvStride2Back Ws (fan-in 32·9)
    st = R(diag_back(st, Ssw("stem.bn"), esav, q));  out.append(("stem.swB", st))
    st = R(bn_back(st, 12544, Xh(12544, "stem.bn"), G, S, es, exh, q))
    out.append(("stem.bnB", st))
    st = R(conv_back(st, 32 * 9, wk, q));        out.append(("stem.cB", st))
    return out


def verify_b0_back(rows, wk=B0_WK, G=B0_GLB, S=B0_SB, es=B0_ESB, exh=B0_EXH, esav=B0_ESAV,
                   q=U32, ssw='window', sx='window', se_nnz=False) -> int:
    """Re-assert EVERY rounded inequality an `EfficientNetBackFloatBudget.lean` would close,
    exactly — the peer of `verify_r34_back`. Returns the count checked; raises on the first
    failure."""
    fwd = dict(b0_eval_chain())
    r = dict(rows)
    n = 0

    def ck(tag, lhs, rhs):
        nonlocal n
        assert lhs <= rhs, f"{tag}: left > right"
        n += 1

    def Ssw(t):
        return (1 + fwd[t][0] / 4) if ssw == 'window' else ssw

    def Sx(t):
        return fwd[t][0] if sx == 'window' else sx

    def conv_ck(tag, m, src, dst):
        A, E = src
        g = r4(gamma_q(m + 2, q))
        ck(tag + '.A', (1 + g) * (m * wk * A + 0), dst[0])
        ck(tag + '.E', g * (m * wk * (A + E) + 0) + m * wk * E, dst[1])

    def bn_ck(tag, hw, src, dst):
        A, E = src
        Xh = F(isqrt_exact(hw))
        Kr, Kb = bn_back_gain(hw, Xh, G, S, es, exh, q)
        ck(tag + '.Kr', bnGradInputReMag(hw, G, F(1), S, Xh), Kr)
        ck(tag + '.Kb', bnGradInputBudgetQ(hw, G, F(1), S, Xh, es, exh, q), Kb)
        ck(tag + '.A', A * (Kr + Kb), dst[0])
        ck(tag + '.E', A * Kb + E * Kr, dst[1])

    def diag_ck(tag, Sd, src, dst):
        A, E = src
        me = mulErr(q, Sd, A, esav, F(0))
        ck(tag + '.A', Sd * A + me, dst[0])
        ck(tag + '.E', me + Sd * E, dst[1])

    def gap_ck(tag, hw, src, dst):
        A, E = src
        inv = F(1) / F(hw)
        me = mulErr(q, inv, A, F(0), F(0))
        ck(tag + '.A', inv * A + me, dst[0])
        ck(tag + '.E', me + inv * E, dst[1])

    def bipath_ck(tag, p, b, dst):
        Pd, Ep = p
        Bd, Ed = b
        ck(tag + '.A', Pd + Bd + q * (Pd + Bd), dst[0])
        ck(tag + '.E', q * (Pd + Ep + Bd + Ed) + (Ep + Ed), dst[1])

    conv_ck('linBack', 10, (F(1), F(0)), r['linBack'])
    gap_ck('gapBack', 56 * 56, r['linBack'], r['gapBack'])
    diag_ck('head.swB', Ssw("head.bn"), r['gapBack'], r['head.swB'])
    bn_ck('head.bnB', 3136, r['head.swB'], r['head.bnB'])
    conv_ck('head.cB', 1280, r['head.bnB'], r['head.cB'])
    st = r['head.cB']
    for tag, kind, cin, cmid, cout, h, w, kd, se_c, se_r in B0_BACK_PLAN:
        blkin = st
        hw = h * w
        he = (2 * h) * (2 * w) if kind == "strided" else hw
        bn_ck(tag + '.bnBp', hw, blkin, r[tag + '.bnBp'])
        conv_ck(tag + '.cBp', cout, r[tag + '.bnBp'], r[tag + '.cBp'])
        sein = r[tag + '.cBp']
        diag_ck(tag + '.se.main', 1 + B0_ESIG, sein, r[tag + '.se.main'])
        diag_ck(tag + '.se.pre', Sx(tag + ".dswish"), sein, r[tag + '.se.pre'])
        A, E = r[tag + '.se.pre']
        N = se_c * hw
        g = r4(gamma_q(N + 1, q))
        NN = F(hw if se_nnz else N)
        ck(tag + '.se.bc.A', NN * A + g * (NN * A), r[tag + '.se.bc'][0])
        ck(tag + '.se.bc.E', g * (NN * (A + E)) + NN * E, r[tag + '.se.bc'][1])
        diag_ck(tag + '.se.sig', B0_SSIG, r[tag + '.se.bc'], r[tag + '.se.sig'])
        conv_ck(tag + '.se.d2', se_c, r[tag + '.se.sig'], r[tag + '.se.d2'])
        diag_ck(tag + '.se.sw', Ssw(tag + ".sd1"), r[tag + '.se.d2'], r[tag + '.se.sw'])
        conv_ck(tag + '.se.d1', se_r, r[tag + '.se.sw'], r[tag + '.se.d1'])
        gap_ck(tag + '.se.gap', hw, r[tag + '.se.d1'], r[tag + '.se.gap'])
        bipath_ck(tag + '.se.out', r[tag + '.se.main'], r[tag + '.se.gap'], r[tag + '.se.out'])
        diag_ck(tag + '.swBd', Ssw(tag + ".dbn"), r[tag + '.se.out'], r[tag + '.swBd'])
        bn_ck(tag + '.bnBd', hw, r[tag + '.swBd'], r[tag + '.bnBd'])
        conv_ck(tag + '.dwB', kd, r[tag + '.bnBd'], r[tag + '.dwB'])
        last = r[tag + '.dwB']
        if kind != "noexp":
            diag_ck(tag + '.swBe', Ssw(tag + ".ebn"), last, r[tag + '.swBe'])
            bn_ck(tag + '.bnBe', he, r[tag + '.swBe'], r[tag + '.bnBe'])
            conv_ck(tag + '.cBe', cmid, r[tag + '.bnBe'], r[tag + '.cBe'])
            last = r[tag + '.cBe']
        if kind == "resid":
            Bd, Ed = last
            ck(tag + '.out.A', Bd + blkin[0] + q * (Bd + blkin[0]), r[tag + '.out'][0])
            ck(tag + '.out.E', q * (Bd + Ed + blkin[0] + blkin[1]) + (Ed + blkin[1]),
               r[tag + '.out'][1])
        else:
            ck(tag + '.out.A', last[0], r[tag + '.out'][0])
            ck(tag + '.out.E', last[1], r[tag + '.out'][1])
        st = r[tag + '.out']
    diag_ck('stem.swB', Ssw("stem.bn"), st, r['stem.swB'])
    bn_ck('stem.bnB', 12544, r['stem.swB'], r['stem.bnB'])
    conv_ck('stem.cB', 32 * 9, r['stem.bnB'], r['stem.cB'])
    return n


def sci(x: F) -> str:
    """4-significant-figure scientific form for a rational the size of a whole-net window."""
    if x == 0:
        return "0"
    e = ilog10(x)
    return f"{float(x / F(10) ** e):.3f}e{e}"


if __name__ == "__main__":
    print("\n── ViT-Tiny sizing probe (planning/float_budget_numbers.md §3.5) ──")
    kap = sm_kappa(U32, VIT_EEXP, NTOK_VIT)
    rho = sm_rho(U32, VIT_EEXP, NTOK_VIT)
    print(f"  softmax side condition smRho = {float(rho):.6f} < 1  ✓  (n = {NTOK_VIT} tokens)")
    print(f"  smKappa = {float(kap):.6f} → capped softmax window {float(1 + U32*(1+kap) + kap):.6f}, "
          f"modulus {float(2*(1 + U32*(1+kap) + kap)):.6f} — both RATIONAL, no Real.exp")

    rows, tainted = vit_chain()
    A, E = rows[-1][1]
    print(f"\n  SHIPPED SHAPE: {len(rows)} stages")
    print(f"    window        {sci(A)}")
    print(f"    budget        {sci(E)}")
    print(f"    budget/window {float(E / A):.3f}   ⛔ the CAP, not the fold (§9)")
    print(f"    re-assertions {verify_vit(rows)} — every rounded inequality re-checked exactly")
    print(f"    statable      {'YES' if max(ilog10(A), ilog10(E)) < 300 else 'NO'}"
          f"   (norm_num ceiling ≈ 1e300)")

    print(f"\n  {'variant':<42} {'window':>12} {'budget':>12} {'unwritable':>11}  statable")
    print("  " + "-" * 84)
    for name, kw in [
        ("shipped shape (Maps.mhProjAttnFullCap)", {}),
        ("uniform param bound (ConvNeXt's mistake)", dict(uniform=True)),
        ("+ convex attn·V bound (lemma NOT proved)", dict(attn_mode='convex')),
        ("mhpB window (|real| + |float-real|)", dict(attn_mode='mhpB')),
        ("cubic GELU (no saturation constant)", dict(gelu_sat=False)),
        ("LN UNCAPPED (the §0.1 quadratic)", dict(ln_cap=False)),
        ("depth 2 (the vitForward2V shape)", dict(k=2)),
        ("depth 6", dict(k=6)),
    ]:
        try:
            r, tt = vit_chain(**kw)
            a, e = r[-1][1]
            ok = 'yes' if max(ilog10(a), ilog10(e)) < 300 and not tt else 'NO'
            print(f"  {name:<42} {sci(a):>12} {sci(e):>12} {len(tt):>11}  {ok}")
        except Exception as ex:
            print(f"  {name:<42} {'—':>12} {'—':>12} {'—':>11}  raised {type(ex).__name__}")

    print("\n  ⭐ 'unwritable' counts stage numerals carrying a `Real.exp` at an argument with no")
    print("     rational bound — the softmax cap's real justification. The uncapped column is not")
    print("     'a bigger number', it is NO NUMBER: those stages cannot be written down at all.")

    print("\n── ResNet-34 BACKWARD sizing probe (planning/float_budget_numbers.md §3.7) ──")
    brows = r34_back_chain(S=F(16))
    bA, bE = brows[-1][1]
    print(f"  SHIPPED SHAPE: {len(brows)} stages, TRAINING-mode BatchNorm, |istd| <= 16")
    print(f"    window        {sci(bA)}")
    print(f"    budget        {sci(bE)}")
    print(f"    budget/window {float(bE / bA):.4f}   ⭐ a FOLD — the backward is LINEAR in the "
          f"cotangent")
    print(f"    re-assertions {verify_r34_back(brows, S=F(16))} — every rounded inequality re-checked "
          f"exactly")
    print(f"    statable      {'YES' if max(ilog10(bA), ilog10(bE)) < 300 else 'NO'}")
    print(f"\n  {'variant':<44} {'window':>12} {'budget':>12}  statable")
    print("  " + "-" * 76)
    for name, kw in [
        ("⭐ shipped: |istd| ≤ 16, per-kind profile, x̂ ≤ √n", dict(S=F(16))),
        ("the same at the unconditional ε-floor S = 317", {}),
        ("uniform 21/10 (the FORWARD's profile)", dict(wk=F(21, 10))),
        ("x̂ from the forward WINDOW (2·A·S)", dict(xhat='window')),
        ("variance floor 1e-3 (S = 32)", dict(S=F(32))),
        ("variance floor 1e-1 (S = 4)", dict(S=F(4))),
        ("σ² ≈ 1 (S = 1)", dict(S=F(1))),
    ]:
        rr = r34_back_chain(**kw)
        a, e = rr[-1][1]
        ok = 'yes' if max(ilog10(a), ilog10(e)) < 300 else 'NO'
        print(f"  {name:<44} {sci(a):>12} {sci(e):>12}  {ok}")
    print("\n  ⛔ `x̂ from the forward WINDOW` is the row that matters: `bnXhat_sq_le`'s |x̂| ≤ √n")
    print("     is what makes this number exist, and it was already in the repo.")
    print("  ⛔ And the fold is conditional on `es`/`exh` = 1e-2 — SUPPLIED float-activation")
    print("     accuracies the forward's own training-mode fold does not discharge (§3.7).")

    print("\n  window growth through block 0 (the per-block multiplier that sets the answer):")
    for tag, (a, e) in rows[3:25]:
        print(f"    {tag:<12} {sci(a):>12} {sci(e):>12}")

    print("\n── MobileNetV2 BACKWARD sizing probe (planning/float_budget_numbers.md §3.8) ──")
    mrows = mnv2_back_chain()
    mA, mE = mrows[-1][1]
    print(f"  SHIPPED SHAPE: {len(mrows)} stages, TRAINING-mode BatchNorm")
    print(f"    window        {sci(mA)}")
    print(f"    budget        {sci(mE)}")
    print(f"    budget/window {float(mE / mA):.4f}   ⭐ a FOLD — every leaf is linear in the "
          f"cotangent")
    print(f"    re-assertions {verify_mnv2_back(mrows)} — every rounded inequality re-checked "
          f"exactly")
    print(f"\n  {'variant':<44} {'window':>12} {'budget':>12}  statable")
    print("  " + "-" * 76)
    for name, kw in [
        ("shipped (per-kind profile, x̂ ≤ √n)", {}),
        ("uniform 28/10 (the FORWARD's profile)", dict(G=F(28, 10))),
        ("x̂ from the forward WINDOW (2·A·S)", dict(xhat='window')),
        ("⭐ NO operating point: S = 317 (the ε-floor)", dict(S=F(317))),
        ("variance floor 1e-3 (S = 32)", dict(S=F(32))),
        ("variance floor 1e-1 (S = 4)", dict(S=F(4))),
        ("σ² ≈ 1 (S = 1)", dict(S=F(1))),
    ]:
        rr = mnv2_back_chain(**kw)
        a, e = rr[-1][1]
        ok = 'yes' if max(ilog10(a), ilog10(e)) < 253 else 'NO'
        print(f"  {name:<44} {sci(a):>12} {sci(e):>12}  {ok}")
    print("\n  ⭐⭐ The ε-floor row is the result: MobileNetV2's backward is statable with NO")
    print("     operating-point hypothesis at all, where ResNet-34's needs |istd| ≤ 16 (§3.7(b)).")
    print("     20 BN sites against r34's 33, and 1×1 fan-ins (24–256) against r34's 512·9.")
    print("  ⛔ relu6's CLAMP — the whole reason the forward window is 2154 — buys the backward")
    print("     NOTHING: `reluMaskBack` is a 0/1 select, envelope-preserving, and a cotangent")
    print("     window has nothing to be clamped to.")

    print("\n── EfficientNet-B0 BACKWARD sizing probe: the SQUEEZE-EXCITE question ──")
    erows = b0_back_chain()
    eA, eE = erows[-1][1]
    print(f"  SHIPPED LEAVES: {len(erows)} stages, TRAINING-mode BatchNorm, any batch size N")
    print(f"    window        {sci(eA)}")
    print(f"    budget        {sci(eE)}")
    print(f"    budget/window {float(eE / eA):.4f}   ⭐ STILL A FOLD — SE does not break "
          f"linearity")
    print(f"    re-assertions {verify_b0_back(erows)} — every rounded inequality re-checked "
          f"exactly")
    print(f"    statable      NO — 1e{ilog10(eA)}, past `norm_num`'s ~1e253 ceiling (§3.7(a))")
    print(f"\n  {'variant':<48} {'window':>12} {'budget':>12}  statable")
    print("  " + "-" * 80)
    for name, kw in [
        ("shipped leaves (swish 1+A/4, Sx = fwd window)", {}),
        ("⭐ global |swish′| ≤ 11/10 ONLY (NOT proved)", dict(ssw=F(11, 10))),
        ("operating-point Sx ≤ 16 ONLY", dict(sx=F(16))),
        ("both", dict(ssw=F(11, 10), sx=F(16))),
        ("both + broadcastBack tightened to its h·w nonzeros", dict(ssw=F(11, 10), sx=F(16),
                                                                   se_nnz=True)),
        ("both + uniform 41/10 param bound", dict(ssw=F(11, 10), sx=F(16), wk=F(41, 10))),
        ("both, x̂ from the forward WINDOW", dict(ssw=F(11, 10), sx=F(16), xhat='window')),
        ("both, S = 317 (the ε-floor)", dict(ssw=F(11, 10), sx=F(16), S=F(317))),
    ]:
        rr = b0_back_chain(**kw)
        a, e = rr[-1][1]
        ok = 'yes' if max(ilog10(a), ilog10(e)) < 253 else 'NO'
        print(f"  {name:<48} {sci(a):>12} {sci(e):>12}  {ok}")
    print("\n  ⭐⭐ ONE UNPROVED SCALAR LEMMA IS THE WHOLE DIFFERENCE. A *global* bound on")
    print("     |swish′| — any constant, the window must simply not appear — moves B0's backward")
    print("     from 1e431 to 1e167 on its own; the operating-point Sx does NOT (1e359, still no).")
    print("     And the constant need not be sharp:")
    for nm, c in [("11/10 (the true sup)", F(11, 10)), ("137/100 (1 + 1/e)", F(137, 100)),
                  ("2 (the crudest global)", F(2))]:
        a, e = b0_back_chain(ssw=c, sx=F(16))[-1][1]
        print(f"       Ssw = {nm:<24} {sci(a):>12} {sci(e):>12}")
    print("  ⛔ §3.4 says that constant needs \"the decay of σ′, i.e. calculus\". It does NOT:")
    print("     σ′(x) = σ(x)(1−σ(x)) = e^{−|x|}/(1+e^{−|x|})² ≤ e^{−|x|}, and |x|·e^{−|x|} ≤ 1")
    print("     straight from e^t ≥ 1+t ≥ t — so |swish′| = |σ + x·σ′| ≤ 2 with")
    print("     `Real.add_one_le_exp` and no MVT, no sup, no derivative analysis. (The sharp")
    print("     1.0998 does need the sup; the table above says nobody needs it.)")
