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
