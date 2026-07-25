"""IBP L-infinity scorecard for a CONVOLUTIONAL net (IntervalBoundConv.lean).

The existing IBP tier (`LipschitzCertScorecardIBP*.lean`) certifies a 784->16->10
dense MLP: `ibp2_certified_at_eps` is hard-wired to `dense . relu . dense`, so
conv/pool layers and depth > 2 had no certificate at all. `IntervalBoundConv.lean`
removes that (compositional `BoxSound3`/`BoxSound3V`, conv + max-pool + dense +
relu, arbitrary depth); this script instantiates it on a real trained CNN.

Net:   conv(1->C, 3x3 SAME) -> relu -> maxpool2 -> denseT(C*4*4 -> 10)
Input: MNIST zero-padded 28x28 -> 32x32, then 4x4 average-pooled to 8x8
       (the repo's established pooled reduction, on a padded canvas so the
       max-pool dimension is even). Pixel values are exact k/4080 rationals.

Perturbation model: pixel L-infinity, `forall a b d, |delta| <= eps`, at
eps in {1,2,4,8}/255 — the same grid as the dense IBP scorecard, so the two are
directly comparable. Because a pooled coordinate is an AVERAGE of 16 raw pixels,
an L-infinity ball of radius eps on the raw 28x28 image maps INTO the radius-eps
ball on the pooled image, so these radii carry their literature meaning.

Everything below eps-propagation is exact `Fraction` arithmetic; only the
training is float. Weights are rationalized to k/256 and the QUANTIZED net is
what gets evaluated, certified, and emitted — so the certified network is the
deployed one, not a nearby real-valued idealization.

Emits LeanMlir/Proofs/Certificates/IbpConvScorecard.lean.
"""
import struct
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "LeanMlir" / "Proofs" / "Certificates" / "IbpConvScorecard.lean"

C = 4                      # conv output channels
S = 8                      # spatial size after pooling
P = S // 2                 # after max-pool
DEN_W = 256                # weight denominator
DEN_X = 4080               # pooled-pixel denominator (16 * 255)
# The scorecard's two sizes, deliberately separated (see the module docstring):
#   N_MEASURE — the fixed test subset the COUNTS are measured over (exact rational
#               interval propagation, no Lean; this is a measurement, not a theorem);
#   N_EMIT    — how many of those carry a per-image `CertifiedAtLinf3` THEOREM.
# Soundness lives in the engine, so kernel-checking the 57th image buys nothing the
# 56th didn't; the emitted set only has to show the engine bites on real weights.
N_MEASURE = int(sys.argv[1]) if len(sys.argv) > 1 else 100
N_EMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 8
EPSN = [1, 2, 4, 8]        # eps = n/255
SEED = 0


# ────────────────────────────── data ──────────────────────────────

def load(split):
    img_f = "train-images-idx3-ubyte" if split == "train" else "t10k-images-idx3-ubyte"
    lab_f = "train-labels-idx1-ubyte" if split == "train" else "t10k-labels-idx1-ubyte"
    with open(DATA / img_f, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        x = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)
    with open(DATA / lab_f, "rb") as f:
        struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8)
    return x, y


def pool_int(x):
    """28x28 uint8 -> 8x8 integer block sums (zero-pad 28->32, 4x4 blocks)."""
    n = x.shape[0]
    padded = np.zeros((n, 32, 32), dtype=np.int64)
    padded[:, 2:30, 2:30] = x
    return padded.reshape(n, 8, 4, 8, 4).sum(axis=(2, 4))     # ints in [0, 4080]


# ─────────────────────────── conv helpers ───────────────────────────

def patches(x):
    """(N,8,8) -> (N,8,8,9) SAME-padded 3x3 patches (matches Proofs.conv2d)."""
    n = x.shape[0]
    pad = np.zeros((n, S + 2, S + 2), dtype=x.dtype)
    pad[:, 1:S + 1, 1:S + 1] = x
    out = np.empty((n, S, S, 9), dtype=x.dtype)
    for kh in range(3):
        for kw in range(3):
            out[:, :, :, kh * 3 + kw] = pad[:, kh:kh + S, kw:kw + S]
    return out


ONES_PATCH = patches(np.ones((1, S, S)))[0]                   # (8,8,9) in/out-of-bounds mask


def pool_max(z):
    """(N,C,8,8) -> (N,C,4,4) 2x2 max-pool."""
    n, c = z.shape[0], z.shape[1]
    return z.reshape(n, c, P, 2, P, 2).max(axis=(3, 5))


def forward(x, Wc9, bc, Wd, bd):
    pt = patches(x)                                            # (N,8,8,9)
    z1 = np.einsum("nhwk,ck->nchw", pt, Wc9) + bc[None, :, None, None]
    a1 = np.maximum(z1, 0.0)
    p1 = pool_max(a1)
    flat = p1.reshape(x.shape[0], -1)
    return z1, a1, p1, flat, flat @ Wd + bd


def train_ibp(xtr, ytr, epochs=30, bs=128, lr=0.08, wd=1e-4,
              eps_train=10.0 / 255, kappa_max=0.6, warm=0.4):
    """Plain CE warm-up, then a ramped IBP loss (Gowal et al. 2018 style).

    The certified counts are what this script reports, so the net is trained for
    the thing being certified: the loss is cross-entropy on the WORST-CASE logit
    vector of the propagated box (`hi` for every wrong class, `lo` for the true
    one). Boxes are carried in the same sign-split form the Lean engine uses, so
    training and certification agree by construction.
    """
    rng = np.random.default_rng(SEED)
    Wc9 = rng.normal(0, 0.35, (C, 9))
    bc = np.zeros(C)
    Wd = rng.normal(0, 0.25, (C * P * P, 10))
    bd = np.zeros(10)
    n = xtr.shape[0]
    steps = epochs * ((n + bs - 1) // bs)
    step = 0
    for ep in range(epochs):
        idx = rng.permutation(n)
        cur = lr * (0.5 ** (ep // 10))
        for s in range(0, n, bs):
            step += 1
            t = min(1.0, max(0.0, (step / steps - warm) / max(1e-9, 1 - warm)))
            et, kap = eps_train * t, kappa_max * t
            b = idx[s:s + bs]
            xb, yb = xtr[b], ytr[b]
            m = len(b)
            pt = patches(xb)
            z1 = np.einsum("nhwk,ck->nchw", pt, Wc9) + bc[None, :, None, None]
            a1 = np.maximum(z1, 0.0)
            p1 = pool_max(a1)
            flat = p1.reshape(m, -1)
            logits = flat @ Wd + bd

            # ── clean CE ──
            lg = logits - logits.max(axis=1, keepdims=True)
            e = np.exp(lg); prob = e / e.sum(axis=1, keepdims=True)
            g = prob.copy(); g[np.arange(m), yb] -= 1.0; g *= (1 - kap) / m

            gWd = flat.T @ g
            gbd = g.sum(axis=0)
            dp1 = (g @ Wd.T).reshape(m, C, P, P)
            a1r = a1.reshape(m, C, P, 2, P, 2)
            mk = (a1r == p1[:, :, :, None, :, None])
            mk = mk / np.maximum(mk.sum(axis=(3, 5), keepdims=True), 1)
            dz1 = (mk * dp1[:, :, :, None, :, None]).reshape(m, C, S, S) * (z1 > 0)
            gWc9 = np.einsum("nchw,nhwk->ck", dz1, pt)
            gbc = dz1.sum(axis=(0, 2, 3))

            if kap > 0:
                # ── IBP box forward ──
                r1 = et * np.einsum("hwk,ck->chw", ONES_PATCH, np.abs(Wc9))
                lo1 = z1 - r1[None]; hi1 = z1 + r1[None]
                lo2 = np.maximum(lo1, 0.0); hi2 = np.maximum(hi1, 0.0)
                lo3 = pool_max(lo2); hi3 = pool_max(hi2)
                flo = lo3.reshape(m, -1); fhi = hi3.reshape(m, -1)
                Wp = Wd * (Wd >= 0); Wn = Wd * (Wd < 0)
                zlo = flo @ Wp + fhi @ Wn + bd
                zhi = fhi @ Wp + flo @ Wn + bd
                zt = zhi.copy(); zt[np.arange(m), yb] = zlo[np.arange(m), yb]

                lgt = zt - zt.max(axis=1, keepdims=True)
                et_ = np.exp(lgt); pr = et_ / et_.sum(axis=1, keepdims=True)
                gi = pr.copy(); gi[np.arange(m), yb] -= 1.0; gi *= kap / m
                Ghi = gi.copy(); Ghi[np.arange(m), yb] = 0.0
                Glo = np.zeros_like(gi); Glo[np.arange(m), yb] = gi[np.arange(m), yb]

                gWd += (fhi.T @ Ghi + flo.T @ Glo) * (Wd >= 0) \
                     + (flo.T @ Ghi + fhi.T @ Glo) * (Wd < 0)
                gbd += Ghi.sum(axis=0) + Glo.sum(axis=0)
                dflo = Ghi @ Wn.T + Glo @ Wp.T
                dfhi = Ghi @ Wp.T + Glo @ Wn.T

                def pool_back(d, pre, post):
                    pr_ = pre.reshape(m, C, P, 2, P, 2)
                    mm = (pr_ == post[:, :, :, None, :, None])
                    mm = mm / np.maximum(mm.sum(axis=(3, 5), keepdims=True), 1)
                    return (mm * d.reshape(m, C, P, P)[:, :, :, None, :, None]).reshape(m, C, S, S)

                dlo1 = pool_back(dflo, lo2, lo3) * (lo1 > 0)
                dhi1 = pool_back(dfhi, hi2, hi3) * (hi1 > 0)
                gWc9 += np.einsum("nchw,nhwk->ck", dlo1 + dhi1, pt)
                gbc += (dlo1 + dhi1).sum(axis=(0, 2, 3))
                dr1 = (dhi1 - dlo1).sum(axis=0)                      # (C,8,8)
                gWc9 += et * np.sign(Wc9) * np.einsum("chw,hwk->ck", dr1, ONES_PATCH)

            Wc9 -= cur * (gWc9 + wd * Wc9)
            bc -= cur * gbc
            Wd -= cur * (gWd + wd * Wd)
            bd -= cur * gbd
    return Wc9, bc, Wd, bd


def train(xtr, ytr, epochs=14, bs=128, lr=0.08, wd=2e-4):
    rng = np.random.default_rng(SEED)
    Wc9 = rng.normal(0, 0.35, (C, 9))
    bc = np.zeros(C)
    Wd = rng.normal(0, 0.25, (C * P * P, 10))
    bd = np.zeros(10)
    n = xtr.shape[0]
    for ep in range(epochs):
        idx = rng.permutation(n)
        cur = lr * (0.5 ** (ep // 5))
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            xb, yb = xtr[b], ytr[b]
            m = len(b)
            pt = patches(xb)
            z1 = np.einsum("nhwk,ck->nchw", pt, Wc9) + bc[None, :, None, None]
            a1 = np.maximum(z1, 0.0)
            p1 = pool_max(a1)
            flat = p1.reshape(m, -1)
            logits = flat @ Wd + bd
            logits -= logits.max(axis=1, keepdims=True)
            e = np.exp(logits)
            prob = e / e.sum(axis=1, keepdims=True)
            dz = prob.copy()
            dz[np.arange(m), yb] -= 1.0
            dz /= m
            gWd = flat.T @ dz + wd * Wd
            gbd = dz.sum(axis=0)
            dflat = dz @ Wd.T
            dp1 = dflat.reshape(m, C, P, P)
            # route through the pool argmax
            a1r = a1.reshape(m, C, P, 2, P, 2)
            mask = (a1r == p1[:, :, :, None, :, None])
            mask = mask / np.maximum(mask.sum(axis=(3, 5), keepdims=True), 1)
            da1 = (mask * dp1[:, :, :, None, :, None]).reshape(m, C, S, S)
            dz1 = da1 * (z1 > 0)
            gWc9 = np.einsum("nchw,nhwk->ck", dz1, pt) + wd * Wc9
            gbc = dz1.sum(axis=(0, 2, 3))
            Wc9 -= cur * gWc9
            bc -= cur * gbc
            Wd -= cur * gWd
            bd -= cur * gbd
    return Wc9, bc, Wd, bd


# ──────────────────────── exact rational IBP ────────────────────────

def quantize(a, den):
    return np.vectorize(lambda v: Fraction(int(round(v * den)), den))(a)


def ibp_box(xF, Wc9F, bcF, WdF, bdF, epsF, ACF):
    """Exact interval propagation for ONE image. xF: (8,8) Fractions."""
    ptF = patches(xF[None])[0]                                  # (8,8,9) Fractions
    z1 = [[[bcF[c] + sum(Wc9F[c][k] * ptF[i][j][k] for k in range(9))
            for j in range(S)] for i in range(S)] for c in range(C)]
    lo1 = [[[z1[c][i][j] - epsF * ACF[c][i][j] for j in range(S)] for i in range(S)]
           for c in range(C)]
    hi1 = [[[z1[c][i][j] + epsF * ACF[c][i][j] for j in range(S)] for i in range(S)]
           for c in range(C)]
    rl = [[[max(lo1[c][i][j], Fraction(0)) for j in range(S)] for i in range(S)] for c in range(C)]
    rh = [[[max(hi1[c][i][j], Fraction(0)) for j in range(S)] for i in range(S)] for c in range(C)]
    plo = [[[max(rl[c][2 * i][2 * j], rl[c][2 * i + 1][2 * j],
                 rl[c][2 * i][2 * j + 1], rl[c][2 * i + 1][2 * j + 1])
             for j in range(P)] for i in range(P)] for c in range(C)]
    phi = [[[max(rh[c][2 * i][2 * j], rh[c][2 * i + 1][2 * j],
                 rh[c][2 * i][2 * j + 1], rh[c][2 * i + 1][2 * j + 1])
             for j in range(P)] for i in range(P)] for c in range(C)]
    zlo, zhi = [], []
    for cls in range(10):
        lo = bdF[cls]
        hi = bdF[cls]
        for c in range(C):
            for i in range(P):
                for j in range(P):
                    wv = WdF[c][i][j][cls]
                    if wv >= 0:
                        lo += plo[c][i][j] * wv
                        hi += phi[c][i][j] * wv
                    else:
                        lo += phi[c][i][j] * wv
                        hi += plo[c][i][j] * wv
        zlo.append(lo)
        zhi.append(hi)
    return z1, lo1, hi1, plo, phi, zlo, zhi


# ───────────────────────────── emission ─────────────────────────────

def r(v):
    v = Fraction(v)
    return f"(({v.numerator}:ℝ)/{v.denominator})" if v.denominator != 1 else f"({v.numerator}:ℝ)"


def vec(vs):
    return "![" + ", ".join(r(v) for v in vs) + "]"


def t3(T, c, h, w):
    return "![" + ",\n    ".join("![" + ",\n     ".join(vec(T[o][i]) for i in range(h)) + "]"
                                 for o in range(c)) + "]"


def t4(W, c, h, w):
    """Fin c → Fin h → Fin w → Fin 10 → ℝ literal."""
    return "![" + ",\n    ".join(
        "![" + ",\n     ".join(
            "![" + ", ".join(vec(W[o][i][j]) for j in range(w)) + "]"
            for i in range(h)) + "]"
        for o in range(c)) + "]"


def main():
    xtr_raw, ytr = load("train")
    xte_raw, yte = load("test")
    xtr = pool_int(xtr_raw) / DEN_X
    xte = pool_int(xte_raw) / DEN_X
    print(f"[data] train {xtr.shape}  test {xte.shape}")

    Wc9, bc, Wd, bd = train_ibp(xtr, ytr, epochs=40,
                                eps_train=6.0 / 255, kappa_max=0.5)
    Wc9F = quantize(Wc9, DEN_W); bcF = quantize(bc, DEN_W)
    WdF_flat = quantize(Wd, DEN_W); bdF = quantize(bd, DEN_W)

    # float mirror of the QUANTIZED net, for accuracy reporting
    Wc9q = np.vectorize(float)(Wc9F); bcq = np.vectorize(float)(bcF)
    Wdq = np.vectorize(float)(WdF_flat); bdq = np.vectorize(float)(bdF)
    *_, lg = forward(xte, Wc9q, bcq, Wdq, bdq)
    acc = (lg.argmax(1) == yte).mean()
    print(f"[net]  quantized test accuracy {acc:.4f}")

    # AC = conv2d(|Wc|, 0, ones): per-net constant (the l1 weight per position)
    ACF = [[[sum(abs(Wc9F[c][k]) * Fraction(int(ONES_PATCH[i][j][k])) for k in range(9))
             for j in range(S)] for i in range(S)] for c in range(C)]
    # dense weights reshaped to (c,i,j,cls)
    WdF = [[[[WdF_flat[(c * P + i) * P + j][cls] for cls in range(10)]
             for j in range(P)] for i in range(P)] for c in range(C)]

    xteF = np.vectorize(lambda v: Fraction(int(v), DEN_X))(pool_int(xte_raw[:N_MEASURE]))

    certified = {n: [] for n in EPSN}
    chosen = []
    for k in range(N_MEASURE):             # the FIXED first-N measurement subset
        xF = xteF[k]
        pred = int(lg[k].argmax())
        if pred != int(yte[k]):
            continue                                   # only certify correct predictions
        rec = {"idx": k, "y": pred, "xF": xF, "eps": {}, "ok": []}
        best = None
        for n in EPSN:
            epsF = Fraction(n, 255)
            z1, lo1, hi1, plo, phi, zlo, zhi = ibp_box(xF, Wc9F, bcF, WdF, bdF, epsF, ACF)
            if all(zhi[j] < zlo[pred] for j in range(10) if j != pred):
                certified[n].append(k)
                rec["ok"].append(n)
                best = (n, lo1, hi1, plo, phi)
            rec["z1"] = z1
        if best is not None:
            # box data only for the LARGEST certifying radius; the rest are `.mono`
            rec["nmax"] = best[0]
            rec["eps"][best[0]] = best[1:]
            chosen.append(rec)
    measured = {n: len(certified[n]) for n in EPSN}
    for n in EPSN:
        print(f"[measure] eps={n}/255: {measured[n]} of the first {N_MEASURE} test images")
    # emit theorems for the FIRST N_EMIT certifying images (test-set order — an
    # unbiased, reproducible rule, not a cherry-pick of the easy ones)
    chosen = chosen[:N_EMIT]
    for n in EPSN:
        print(f"[emit]    eps={n}/255: {len([r for r in chosen if n in r['ok']])} "
              f"of the {len(chosen)} images carrying Lean certificates")

    # ───── Lean ─────
    L = []
    A = L.append
    A("import LeanMlir.Proofs.Foundation.IntervalBoundConv\n")
    n_emit = len(chosen)
    A(f'''/-! # IBP `L∞` scorecard for a CONVOLUTIONAL net — generated instance

The first certificate in this repo that covers a convolution, a max-pool, and
more than two layers. Generated by `scripts/ibp_conv_scorecard.py`; the engine
is `Foundation/IntervalBoundConv.lean` (`BoxSound3`/`BoxSound3V`,
`ibp3_certified_of_boxSound`).

Net (trained, then rationalized to `k/{DEN_W}` — the QUANTIZED net is what is
certified, so the certified network is the deployed one):

  `conv2d({C} out-channels, 3×3 SAME) → reluT → maxPool2 → denseT({C}·{P}·{P} → 10)`

Input: MNIST zero-padded 28×28 → 32×32, 4×4 average-pooled to {S}×{S}, exact
`k/{DEN_X}` pixel rationals. Quantized test accuracy **{acc:.4f}**.

Perturbation model: pixel `L∞`, `∀ a b d, |δ a b d| ≤ ε`, at ε = 1/255, 2/255,
4/255, 8/255 — the same grid as the dense `LipschitzCertScorecardIBP.lean`, so
the two tiers are directly comparable. A pooled coordinate is an AVERAGE of 16
raw pixels, so a radius-ε `L∞` ball on the raw image maps into the radius-ε ball
on the pooled image: these radii keep their literature meaning.

**Theorem vs. measurement — read this before quoting a number.** Soundness lives
in the ENGINE, which is proved once; kernel-checking the 57th image buys nothing
the 56th didn't. So this file separates the two:

* **measured** (exact rational interval propagation, no Lean in the loop) over
  the fixed first {N_MEASURE} test images: {measured[1]}/{N_MEASURE}, {measured[2]}/{N_MEASURE},
  {measured[4]}/{N_MEASURE}, {measured[8]}/{N_MEASURE} at ε = 1, 2, 4, 8/255;
* **proved** — the first {n_emit} of those certifying images (test-set order, an
  unbiased rule) each carry a `CertifiedAtLinf3 net ε x y` THEOREM, i.e.
  `∀ δ, (∀ a b d, |δ| ≤ ε) → ∀ j ≠ y, net (x+δ) j < net (x+δ) y`.

`scorecard_ibp_conv` below states ONLY the proved counts; the measured row is a
measurement and is labelled as one. Both are LOWER bounds — a loose box cannot
prove an image UNcertifiable. All `propext / Classical.choice / Quot.sound`. -/\n''')
    A("namespace Proofs\nnamespace IBP\nnamespace ConvNet\n")
    A("set_option maxRecDepth 100000")
    A("set_option maxHeartbeats 3200000\n")

    A("/-- Trained conv kernel (`k/256`). -/")
    A(f"noncomputable def Wc : Kernel4 {C} 1 3 3 :=\n  ![" + ",\n    ".join(
        "![![" + ",\n       ".join(vec([Wc9F[c][kh * 3 + kw] for kw in range(3)])
                                   for kh in range(3)) + "]]" for c in range(C)) + "]\n")
    A(f"noncomputable def bc : Vec {C} := {vec([bcF[c] for c in range(C)])}\n")
    A("/-- Trained dense head, indexed in place on the pooled activation. -/")
    A(f"noncomputable def Wd : Fin {C} → Fin {P} → Fin {P} → Fin 10 → ℝ :=\n"
      f"  {t4(WdF, C, P, P)}\n")
    A(f"noncomputable def bd : Vec 10 := {vec([bdF[cls] for cls in range(10)])}\n")

    A("""/-- The certified network: conv → relu → max-pool → dense head. Four layers,
    a genuine convolution and a genuine max-pool — the shapes `ibp2_certified_at_eps`
    could not reach. -/""")
    A(f"noncomputable def net : Tensor3 1 {S} {S} → Vec 10 :=\n"
      f"  denseT Wd bd ∘ maxPool2 (c := {C}) (h := {P}) (w := {P}) ∘ reluT ∘ conv2d Wc bc\n")

    mp = f"maxPool2 (c := {C}) (h := {P}) (w := {P})"
    A("/-- The propagated box, assembled by `BoxSound3.comp` — depth is composition. -/")
    A(f"noncomputable def netLo (lo hi : Tensor3 1 {S} {S}) : Vec 10 :=")
    A(f"  denseTLo Wd bd ({mp} (reluT (convLo Wc bc lo hi)))")
    A(f"                 ({mp} (reluT (convHi Wc bc lo hi)))\n")
    A(f"noncomputable def netHi (lo hi : Tensor3 1 {S} {S}) : Vec 10 :=")
    A(f"  denseTHi Wd bd ({mp} (reluT (convLo Wc bc lo hi)))")
    A(f"                 ({mp} (reluT (convHi Wc bc lo hi)))\n")
    A("theorem net_boxSound : BoxSound3V net netLo netHi :=")
    A("  (denseT_boxSound3V Wd bd).comp3")
    A(f"    ((maxPool2_boxSound3 (c := {C}) (h := {P}) (w := {P})).comp")
    A("      (reluT_boxSound3.comp (conv2d_boxSound3 Wc bc)))\n")

    A("""/-- `AC = |Wc| ⊛ 𝟙` — the per-position `ℓ1` weight of the kernel, a per-NET
    constant (SAME padding drops taps at the border, so it varies by position).
    `convLo_uniform`/`convHi_uniform` turn the first layer's box into
    `conv2d Wc bc x ∓ ε·AC`, so each image costs one forward pass, not two. -/""")
    A(f"noncomputable def AC : Tensor3 {C} {S} {S} :=\n  {t3(ACF, C, S, S)}\n")
    A(f"theorem AC_eval : ∀ (o : Fin {C}) (i j : Fin {S}),")
    A(f"    conv2d (absK Wc) (fun _ => 0) (onesT 1 {S} {S}) o i j = AC o i j := by")
    A("  intro o i j")
    A("  fin_cases o <;> fin_cases i <;> fin_cases j <;>")
    A("    · simp [conv2d, absK, onesT, Wc, AC, Fin.sum_univ_succ]")
    A("      try norm_num\n")

    for rec in chosen:
        k, y, xF = rec["idx"], rec["y"], rec["xF"]
        A(f"-- ═══════════════ MNIST test image #{k} (label {y}) ═══════════════")
        A(f"noncomputable def img{k} : Tensor3 1 {S} {S} :=\n  {t3([xF], 1, S, S)}\n")
        A(f"noncomputable def cx{k} : Tensor3 {C} {S} {S} :=\n  {t3(rec['z1'], C, S, S)}\n")
        A(f"theorem cx{k}_eval : ∀ (o : Fin {C}) (i j : Fin {S}),")
        A(f"    conv2d Wc bc img{k} o i j = cx{k} o i j := by")
        A("  intro o i j")
        A("  fin_cases o <;> fin_cases i <;> fin_cases j <;>")
        A(f"    · simp [conv2d, Wc, bc, img{k}, cx{k}, Fin.sum_univ_succ]")
        A("      try norm_num\n")
        for n, (lo1, hi1, plo, phi) in rec["eps"].items():
            tag = f"{k}e{n}"
            e = f"(({n}:ℝ)/255)"
            A(f"noncomputable def lo{tag} : Tensor3 {C} {S} {S} :=\n  {t3(lo1, C, S, S)}\n")
            A(f"noncomputable def hi{tag} : Tensor3 {C} {S} {S} :=\n  {t3(hi1, C, S, S)}\n")
            A("/-- The layer-1 box, via the uniform collapse `conv2d Wc bc x ∓ ε·AC`. -/")
            A(f"theorem box{tag} :")
            A(f"    convLo Wc bc (fun a b d => img{k} a b d - {e})")
            A(f"      (fun a b d => img{k} a b d + {e}) = lo{tag} ∧")
            A(f"    convHi Wc bc (fun a b d => img{k} a b d - {e})")
            A(f"      (fun a b d => img{k} a b d + {e}) = hi{tag} := by")
            A("  constructor <;> funext o i j")
            A(f"  · rw [convLo_uniform, cx{k}_eval, AC_eval]")
            A(f"    fin_cases o <;> fin_cases i <;> fin_cases j <;>")
            A(f"      · simp [cx{k}, AC, lo{tag}]")
            A("        try norm_num")
            A(f"  · rw [convHi_uniform, cx{k}_eval, AC_eval]")
            A(f"    fin_cases o <;> fin_cases i <;> fin_cases j <;>")
            A(f"      · simp [cx{k}, AC, hi{tag}]")
            A("        try norm_num\n")
            A(f"noncomputable def plo{tag} : Tensor3 {C} {P} {P} :=\n  {t3(plo, C, P, P)}\n")
            A(f"noncomputable def phi{tag} : Tensor3 {C} {P} {P} :=\n  {t3(phi, C, P, P)}\n")
            A("/-- The box after ReLU and max-pool (both monotone: pool the endpoints). -/")
            A(f"theorem pool{tag} :")
            A(f"    {mp} (reluT lo{tag}) = plo{tag} ∧ {mp} (reluT hi{tag}) = phi{tag} := by")
            A("  constructor <;> funext o i j <;>")
            A("    fin_cases o <;> fin_cases i <;> fin_cases j <;>")
            A(f"      · simp [maxPool2, reluT, lo{tag}, hi{tag}, plo{tag}, phi{tag}]")
            A("        try norm_num\n")
            A(f"""/-- **Certified at ε = {n}/255** (MNIST test image #{k}, class {y}): EVERY
    pixel-`L∞` perturbation of size ≤ {n}/255 leaves class {y} the strict argmax of
    the conv net — conv, max-pool and a dense head, four layers. -/""")
            A(f"theorem cert{tag} : CertifiedAtLinf3 net {e} img{k} {y} := by")
            A("  refine ibp3_certified_of_boxSound net_boxSound ?_")
            A("  intro j hj")
            A(f"  simp only [netLo, netHi, (box{tag}).1, (box{tag}).2,")
            A(f"    (pool{tag}).1, (pool{tag}).2]")
            A("  fin_cases j <;>")
            A("    first")
            A("    | exact absurd rfl hj")
            A("    | · simp only [denseTLo, denseTHi]")
            A(f"        simp [Wd, bd, plo{tag}, phi{tag}, Fin.sum_univ_succ]")
            A("        try norm_num\n")
        for n2 in rec["ok"]:
            if n2 == rec["nmax"]:
                continue
            A(f"""/-- Certified at ε = {n2}/255 (image #{k}) — the radius-{rec['nmax']}/255
    certificate restricted; no second box. -/""")
            A(f"theorem cert{k}e{n2} : CertifiedAtLinf3 net (({n2}:ℝ)/255) img{k} {y} :=")
            A(f"  cert{k}e{rec['nmax']}.mono (by norm_num)\n")

    A("-- ═══════════════════════ the scorecard ═══════════════════════\n")
    for n in EPSN:
        sel = [rec for rec in chosen if n in rec["ok"]]
        A(f"/-- Certificate witnesses at ε = {n}/255: `(test index, image, class)`. -/")
        A(f"noncomputable def certsE{n} : List (ℕ × Tensor3 1 {S} {S} × Fin 10) :=")
        A("  [" + ", ".join(f"({rec['idx']}, img{rec['idx']}, {rec['y']})" for rec in sel) + "]\n")
        A(f"theorem certsE{n}_certified :")
        A(f"    ∀ p ∈ certsE{n}, CertifiedAtLinf3 net (({n}:ℝ)/255) p.2.1 p.2.2 :=")
        if len(sel) > 1:
            A("  List.forall_iff_forall_mem.mp")
            A("    ⟨" + ", ".join(f"cert{rec['idx']}e{n}" for rec in sel) + "⟩\n")
        elif len(sel) == 1:
            A(f"  List.forall_iff_forall_mem.mp cert{sel[0]['idx']}e{n}\n")
        else:
            A(f"  by intro p hp; simp [certsE{n}] at hp\n")

    A(f"""/-- **The proved core of the scorecard.** Each count is tied to its per-image
    `CertifiedAtLinf3` proofs, not to a list length. These are the {n_emit} images
    that carry Lean certificates — NOT the measured dataset counts (see the header:
    {measured[1]}/{N_MEASURE}, {measured[2]}/{N_MEASURE}, {measured[4]}/{N_MEASURE}, {measured[8]}/{N_MEASURE} at
    ε = 1, 2, 4, 8/255), which are exact-rational measurements rather than theorems.
    The engine is what makes them sound; the images only witness non-vacuity. -/""")
    A("theorem scorecard_ibp_conv :")
    A("\n  ∧ ".join(
        f"(certsE{n}.length = {len([r for r in chosen if n in r['ok']])} ∧\n"
        f"      ∀ p ∈ certsE{n}, CertifiedAtLinf3 net (({n}:ℝ)/255) p.2.1 p.2.2)"
        for n in EPSN) + " :=")
    A("  ⟨" + ", ".join(f"⟨rfl, certsE{n}_certified⟩" for n in EPSN) + "⟩\n")
    A("end ConvNet\nend IBP\nend Proofs")

    OUT.write_text("\n".join(L) + "\n")
    print(f"[emit] {OUT}  ({len(open(OUT).readlines())} lines)")


if __name__ == "__main__":
    main()
