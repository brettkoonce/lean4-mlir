"""Trained-weight CNN descent, all four conv rungs: concrete satisfiable instances on real MNIST.

Emits two files, every hypothesis discharged by exact in-kernel rational arithmetic at TRAINED,
/128-rationalized weights (biases /16384) and a REAL MNIST test image:

- LeanMlir/Proofs/Training/Trained/CnnDescent.lean, net A (conv1 WITHOUT bias):
  `cnn_conv2_exact_sgd_descends` and `cnn_conv2_bias_exact_sgd_descends` (one exact-gradient SGD
  step on the conv2 kernel / bias decreases the cross-entropy loss). The point is the pool
  hypothesis `MaxPool2MarginQUpTo` with twins `ConvPatchEq`: the chosen image has a live 2x2
  window whose conv2 outputs are EQUAL (they read identical all-zero input patches), the case the
  old `MaxPool2MarginQ` could not state and real MNIST forces
  (scripts/probes/mnist_pool_twin_probe.py). conv1 is bias-free so that a zero input patch gives
  x1 = 0, the same value as the zero padding: then the cells of a blank corner read identical
  patches, and their conv2 outputs tie at b2.
- LeanMlir/Proofs/Training/Trained/CnnDescentConv1.lean, net B (conv1 WITH a trained bias):
  `cnn_conv1_exact_sgd_descends` and `cnn_conv1_bias_exact_sgd_descends`. Net A cannot serve the
  conv1 rungs: their relu1 margin needs every conv1 pre-activation nonzero, and a bias-free conv1
  is exactly 0 on a blank patch. The twins are two-layer (`ConvPatchEq2`): two cells whose 5x5
  two-conv receptive fields lie in a blank region, with every outer read in bounds.

No pool-tie regularizer is used: the ties are the data's. Net (reduced so every table is exact in
the kernel): 24x24-center-cropped MNIST, 2x2 block sums -> 12x12 (exact pixel sums /1020), conv1
1->2 3x3 SAME, relu, conv2 2->2 3x3 SAME + bias, relu, maxpool 2x2 -> 2x6x6, dense 72->8, relu,
dense 8->8, relu, dense 8->10; both nets trained by the same seed-0 numpy SGD.

Each step is the exact gradient (eta = 0). The kernel rungs take learning rate 2^-K, the largest
power of two for which every hypothesis holds at the explicit radius lr * cnnConv{1,2}GradBound;
each bias rung takes the largest 2^-K whose radius sits inside its kernel rung's, so the relu and
pool margins carry over by monotonicity.

conv2d semantics mirror LeanMlir/Proofs/Architectures/CNN.lean (cross-correlation, SAME zero
padding). Not in CI: it needs MNIST in data/. Regenerate by hand and confirm the committed Lean
comes back byte-identical.
"""
import os
import sys
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
from _mnist_io import mnist  # noqa: E402

OUT = os.path.join(ROOT, "LeanMlir/Proofs/Training/Trained/CnnDescent.lean")
DEN_W = 128          # weight rationalization grid
DEN_B = 16384        # bias grid: a trained conv2 bias sits below 1/256, and at /128 it rounds to 0
DEN_X = 1020         # 4 * 255: exact 2x2 pixel-sum denominator
S = 12               # input side
C = 2                # conv channels
D1 = 8               # dense width
NC = 10
H = S // 2           # pooled side

Xtr_raw, ytr = mnist("train")
Xte_raw, yte = mnist("test")


def pool12_sums(X):
    """center-crop 28->24 (rows/cols 2..26), 2x2 block sums -> (N,12,12) ints 0..1020."""
    Xc = X[:, 2:26, 2:26].astype(np.int64)
    return Xc.reshape(-1, S, 2, S, 2).sum(axis=(2, 4))


Str = pool12_sums(Xtr_raw)
Ste = pool12_sums(Xte_raw)
Xtr4 = (Str / float(DEN_X))[:, None]
Xte4 = (Ste / float(DEN_X))[:, None]

# ---------------------------------------------------------------- training (numpy, seed 0)


def im2col(x):  # (N, c, S, S) -> (N, S*S, c*9), cross-correlation order (c, kh, kw)
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    N, c = x.shape[0], x.shape[1]
    cols = np.empty((N, S, S, c, 3, 3), dtype=x.dtype)
    for kh in range(3):
        for kw in range(3):
            cols[:, :, :, :, kh, kw] = xp[:, :, kh:kh + S, kw:kw + S].transpose(0, 2, 3, 1)
    return cols.reshape(N, S * S, c * 9)


def conv_fwd(x, Wm, b):
    P = im2col(x)
    y = P @ Wm.T + b
    return y.transpose(0, 2, 1).reshape(-1, Wm.shape[0], S, S), P


def conv_bwd(dy, P, Wm, cin):
    n = dy.shape[0]
    dyf = dy.reshape(n, dy.shape[1], S * S).transpose(0, 2, 1)
    dW = np.einsum('npo,npk->ok', dyf, P) / n
    db = dyf.sum(axis=(0, 1)) / n
    dP = (dyf @ Wm).reshape(n, S, S, cin, 3, 3)
    dxp = np.zeros((n, cin, S + 2, S + 2))
    for kh in range(3):
        for kw in range(3):
            dxp[:, :, kh:kh + S, kw:kw + S] += dP[:, :, :, :, kh, kw].transpose(0, 3, 1, 2)
    return dW, db, dxp[:, :, 1:S + 1, 1:S + 1]


def maxpool_fwd(x):
    xw = x.reshape(-1, C, H, 2, H, 2).transpose(0, 1, 2, 4, 3, 5).reshape(-1, C, H, H, 4)
    am = xw.argmax(axis=-1)
    return xw.max(axis=-1), am


def maxpool_bwd(dy, am):
    n = dy.shape[0]
    dxw = np.zeros((n, C, H, H, 4))
    np.put_along_axis(dxw, am[..., None], dy[..., None], axis=-1)
    return dxw.reshape(n, C, H, H, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, C, S, S)


NAMES = ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5")


def forward(net, x4):
    z1, P1 = conv_fwd(x4, net["W1"], net["b1"]); a1 = np.maximum(z1, 0)
    z2, P2 = conv_fwd(a1, net["W2"], net["b2"]); a2 = np.maximum(z2, 0)
    p, am = maxpool_fwd(a2)
    f = p.reshape(len(x4), -1)
    z3 = f @ net["W3"] + net["b3"]; a3 = np.maximum(z3, 0)
    z4 = a3 @ net["W4"] + net["b4"]; a4 = np.maximum(z4, 0)
    z5 = a4 @ net["W5"] + net["b5"]
    return z1, P1, a1, z2, P2, a2, am, f, z3, a3, z4, a4, z5


def train(conv1_bias, tag):
    """Seed-0 numpy SGD, 12 epochs, batch 128. conv1_bias=False: conv1 has no bias (net A, the
    conv2 rungs); True: conv1 has a trained bias initialised at 0.1 (net B, the conv1 rungs)."""
    rng = np.random.default_rng(0)
    he = lambda *s: rng.normal(0, np.sqrt(2.0 / np.prod(s[1:])), s)
    net = {}
    net["W1"] = he(C, 9)
    net["W2"] = he(C, C * 9); net["b2"] = np.full(C, 0.1)
    net["W3"] = he(C * H * H, D1) * 0.5; net["b3"] = np.full(D1, 0.1)
    net["W4"] = he(D1, D1) * 0.5; net["b4"] = np.full(D1, 0.1)
    net["W5"] = he(D1, NC) * 0.5; net["b5"] = np.zeros(NC)
    net["b1"] = np.full(C, 0.1) if conv1_bias else np.zeros(C)
    lr0, bs = 0.05, 128
    for ep in range(12):
        lr = lr0 * (0.3 if ep >= 8 else 1.0)
        idx = rng.permutation(len(Xtr4))
        for bi in range(0, len(Xtr4), bs):
            sel = idx[bi:bi + bs]
            xb, yb = Xtr4[sel], ytr[sel]
            z1, P1, a1, z2, P2, a2, am, f, z3, a3, z4, a4, z5 = forward(net, xb)
            z = z5 - z5.max(1, keepdims=True)
            pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
            g5 = pr.copy(); g5[np.arange(len(yb)), yb] -= 1
            n = len(yb)
            dW5 = a4.T @ g5 / n; db5 = g5.mean(0)
            dz4 = (g5 @ net["W5"].T) * (z4 > 0)
            dW4 = a3.T @ dz4 / n; db4 = dz4.mean(0)
            dz3 = (dz4 @ net["W4"].T) * (z3 > 0)
            dW3 = f.T @ dz3 / n; db3 = dz3.mean(0)
            dp = (dz3 @ net["W3"].T).reshape(n, C, H, H)
            dz2 = maxpool_bwd(dp, am) * (z2 > 0)
            dW2, db2, da1 = conv_bwd(dz2, P2, net["W2"], C)
            dW1, db1, _ = conv_bwd(da1 * (z1 > 0), P1, net["W1"], 1)
            upd = [("W1", dW1), ("W2", dW2), ("W3", dW3), ("W4", dW4), ("W5", dW5),
                   ("b2", db2), ("b3", db3), ("b4", db4), ("b5", db5)]
            if conv1_bias:
                upd.append(("b1", db1))
            for k_, g_ in upd:
                net[k_] -= lr * g_
        acc = (forward(net, Xte4)[-1].argmax(1) == yte).mean()
        print(f"[{tag}] ep {ep}: test acc {acc:.4f}", flush=True)
    return net


def ratz(a, den=DEN_W):
    return np.vectorize(lambda v: Fraction(int(round(v * den)), den))(a)


def quantize(net, tag):
    """/128 weights, /16384 biases; prints the float and the quantized test accuracy."""
    q = {k: ratz(net[k], DEN_B if k.startswith("b") else DEN_W) for k in NAMES}
    fl = {k: q[k].astype(float) for k in NAMES}
    acc_float = (forward(net, Xte4)[-1].argmax(1) == yte).mean()
    acc_q = (forward(fl, Xte4)[-1].argmax(1) == yte).mean()
    print(f"[{tag}] float acc {acc_float:.4f}  quantized acc {acc_q:.4f}", flush=True)
    return q


QA = quantize(train(False, "A"), "A")
W1r, W2r, b2r, W3r, b3r = QA["W1"], QA["W2"], QA["b2"], QA["W3"], QA["b3"]
W4r, b4r, W5r, b5r = QA["W4"], QA["b4"], QA["W5"], QA["b5"]

# ---------------------------------------------------------------- exact forward
ZF = Fraction(0)


def conv_exact(x, Wr, br):
    """x: (cin,S,S) Fractions; Wr: (oc, cin*9). Mirrors CNN.lean conv2d (b o + sum W * pad)."""
    cin = x.shape[0]; oc = Wr.shape[0]
    out = np.empty((oc, S, S), dtype=object)
    for o in range(oc):
        for hi in range(S):
            for wi in range(S):
                s = br[o]
                for c_ in range(cin):
                    for kh in range(3):
                        for kw in range(3):
                            r, cc = hi + kh - 1, wi + kw - 1
                            if 0 <= r < S and 0 <= cc < S:
                                s += Wr[o, c_ * 9 + kh * 3 + kw] * x[c_, r, cc]
                out[o, hi, wi] = s
    return out


def relu_exact(t):
    return np.vectorize(lambda v: v if v > 0 else ZF)(t)


def patch(x, hi, wi):
    """zero-padded 3x3 x C patch of x at (hi, wi), the `convPad` reads."""
    return tuple(x[c_, hi + kh - 1, wi + kw - 1] if 0 <= hi + kh - 1 < S and 0 <= wi + kw - 1 < S
                 else ZF for c_ in range(x.shape[0]) for kh in range(3) for kw in range(3))


CELLS = [(0, 0), (0, 1), (1, 0), (1, 1)]   # fin_cases order on Fin 2 × Fin 2


def analyse(i):
    """Exact tables + per-window certificates for test image i, or None."""
    x0 = np.vectorize(lambda s: Fraction(int(s), DEN_X))(Ste[i])[None, :, :]
    c1 = conv_exact(x0, W1r, [ZF] * C)
    x1 = relu_exact(c1)
    c2 = conv_exact(x1, W2r, b2r)
    if any(v == 0 for v in c2.flat):
        return None
    certs, live_twin = {}, 0
    gaps = []
    for ci in range(C):
        for ho in range(H):
            for wo in range(H):
                pos = [(2 * ho + a, 2 * wo + b) for a, b in CELLS]
                vals = [c2[ci, r, s] for r, s in pos]
                if all(v <= 0 for v in vals):
                    certs[ci, ho, wo] = ("dead",)
                    continue
                m = max(range(4), key=lambda k: vals[k])
                kinds = []
                for k in range(4):
                    if k == m:
                        kinds.append("self")
                    elif patch(x1, *pos[m]) == patch(x1, *pos[k]):
                        if all(v == 0 for v in patch(x1, *pos[k])):
                            kinds.append("twin")
                        else:
                            return None          # only all-zero-patch twins are emitted
                    elif vals[k] < vals[m]:
                        kinds.append("gap"); gaps.append(vals[m] - vals[k])
                    else:
                        return None
                if "twin" in kinds:
                    live_twin += 1
                certs[ci, ho, wo] = ("live", m, kinds)
    r2 = relu_exact(c2)
    pool = np.empty((C, H, H), dtype=object)
    for ci in range(C):
        for ho in range(H):
            for wo in range(H):
                pool[ci, ho, wo] = max(r2[ci, 2 * ho + a, 2 * wo + b] for a, b in CELLS)
    f = pool.reshape(-1)
    d3 = np.array([sum(f[j] * W3r[j, k] for j in range(C * H * H)) + b3r[k] for k in range(D1)],
                  dtype=object)
    if any(v == 0 for v in d3):
        return None
    r3 = np.array([v if v > 0 else ZF for v in d3], dtype=object)
    d4 = np.array([sum(r3[j] * W4r[j, k] for j in range(D1)) + b4r[k] for k in range(D1)],
                  dtype=object)
    if any(v == 0 for v in d4):
        return None
    r4 = np.array([v if v > 0 else ZF for v in d4], dtype=object)
    z5 = np.array([sum(r4[j] * W5r[j, k] for j in range(D1)) + b5r[k] for k in range(NC)],
                  dtype=object)
    return dict(x0=x0, c1=c1, x1=x1, c2=c2, r2=r2, f=f, d3=d3, r3=r3, d4=d4, r4=r4, z5=z5,
                certs=certs, live_twin=live_twin, mingap=min(gaps) if gaps else None)


def frac_up(v, den=8):
    """smallest k/den >= v."""
    return Fraction(-((-v.numerator * den) // v.denominator), den)


chosen = None
for i in range(len(Ste)):
    t = analyse(i)
    if t is None or t["live_twin"] == 0:
        continue
    pred = int(np.argmax([float(v) for v in t["z5"]]))
    if pred != int(yte[i]):
        continue
    chosen = (i, t, pred)
    break
if chosen is None:
    sys.exit("NO INSTANCE FOUND")
IDX, T, PRED = chosen
print(f"instance: test #{IDX} label {yte[IDX]} pred {PRED}, live twin windows {T['live_twin']}",
      flush=True)

# ---------------------------------------------------------------- constants and the learning rate
A = frac_up(max(T["x1"].flat))
W3B = frac_up(max(abs(v) for v in W3r.flat), DEN_W)
W4B = frac_up(max(abs(v) for v in W4r.flat), DEN_W)
W5B = frac_up(max(abs(v) for v in W5r.flat), DEN_W)
P = C * C * 9
K2 = (2 * H) * (2 * H)
G = P * (K2 * A * (D1 * (W3B * (D1 * (W4B * (NC * (W5B * 1)))))))
minz2 = min(abs(v) for v in T["c2"].flat)
minz3 = min(abs(v) for v in T["d3"])
minz4 = min(abs(v) for v in T["d4"])


def ok(lr):
    R = lr * G
    aR = A * R
    if not aR < minz2:
        return False
    if not 2 * aR < T["mingap"]:
        return False
    if not W3B * (K2 * aR) < minz3:
        return False
    if not W4B * (D1 * (W3B * (K2 * aR))) < minz4:
        return False
    dl = W5B * (D1 * (W4B * (D1 * (W3B * (K2 * aR)))))
    if not 2 * dl < 1:
        return False
    num = 2 * NC * K2 ** 2 * D1 ** 2 * D1 ** 2 * W3B ** 2 * W4B ** 2 * W5B ** 2 * A ** 2
    return num / (1 - 2 * dl) * lr * P <= Fraction(1, 4)


KEXP = next(k for k in range(1, 200) if ok(Fraction(1, 2 ** k)))
LR = Fraction(1, 2 ** KEXP)
print(f"a = {A}, w3 = {W3B}, w4 = {W4B}, w5 = {W5B}, lr = 2^-{KEXP}", flush=True)

# The conv2-bias rung (rho = 1) on the same net and image: the largest 2^-k whose radius lr*G_b
# stays inside the kernel rung's a*lr*G, so its relu2 and pool margins are the kernel rung's by
# monotonicity; the head margins and the two smallness conditions are rechecked at lr*G_b.
BHEAD = D1 * (W3B * (D1 * (W4B * (NC * (W5B * 1)))))
GB2 = C * (K2 * 1 * BHEAD)


def ok_bias2(lr):
    RB = lr * GB2
    if not RB <= A * (LR * G):
        return False
    if not W3B * (K2 * RB) < minz3:
        return False
    if not W4B * (D1 * (W3B * (K2 * RB))) < minz4:
        return False
    dl = W5B * (D1 * (W4B * (D1 * (W3B * (K2 * RB)))))
    if not 2 * dl < 1:
        return False
    num = 2 * NC * K2 ** 2 * D1 ** 2 * D1 ** 2 * W3B ** 2 * W4B ** 2 * W5B ** 2
    return num / (1 - 2 * dl) * lr * C <= Fraction(1, 4)


KEXPB = next(k for k in range(1, 200) if ok_bias2(Fraction(1, 2 ** k)))
RB2 = Fraction(1, 2 ** KEXPB) * GB2
print(f"conv2 bias rung: lr = 2^-{KEXPB}", flush=True)

# ---------------------------------------------------------------- Lean emission
R = LR * G
AR = A * R            # the pool / relu₂ margin radius a·(lr·G), exact
SR3 = W3B * (K2 * AR)
SR4 = W4B * (D1 * (W3B * (K2 * AR)))
LBL = int(yte[IDX])


def lit(fr):
    fr = Fraction(fr)
    if fr.denominator == 1:
        return f"({fr.numerator} : ℝ)"
    return f"(({fr.numerator} : ℝ)/{fr.denominator})"


def vec_lit(vals):
    return "![" + ", ".join(lit(v) for v in vals) + "]"


def t3_lit(t):
    return "![" + ",\n    ".join(
        "![" + ",\n      ".join(vec_lit(t[ci, hi, :]) for hi in range(t.shape[1])) + "]"
        for ci in range(t.shape[0])) + "]"


def k4_lit(Wr, oc, cin):
    chunks = []
    for o in range(oc):
        rows = []
        for c_ in range(cin):
            rows.append("![" + ", ".join(vec_lit([Wr[o, c_ * 9 + kh * 3 + kw] for kw in range(3)])
                                         for kh in range(3)) + "]")
        chunks.append("![" + ",\n     ".join(rows) + "]")
    return "![" + ",\n   ".join(chunks) + "]"


def mat_lit(Wr):
    return "![" + ",\n    ".join(vec_lit(Wr[i, :]) for i in range(Wr.shape[0])) + "]"


def fm(k, n):
    return f"(⟨{k}, by norm_num⟩ : Fin {n})"


def conv_rows(name, Wn, bn, xn, tab, oc):
    """per-row conv lemmas + aggregator for conv2d Wn bn xn = tab (S×S)."""
    out = []
    for o in range(oc):
        for hi in range(S):
            out.append(f"""theorem {name}_r{o}_{hi} : ∀ wi : Fin {S},
    conv2d {Wn} {bn} {xn} {fm(o, oc)} {fm(hi, S)} wi = {tab} {fm(o, oc)} {fm(hi, S)} wi := by
  intro wi
  fin_cases wi <;> (simp [conv2d, {Wn}, {bn}, {xn}, {tab}, Fin.sum_univ_succ]; try norm_num)
""")
    bullets = "\n".join(f"  · exact {name}_r{o}_{hi} wi" for o in range(oc) for hi in range(S))
    out.append(f"""theorem {name} : ∀ (o : Fin {oc}) (hi wi : Fin {S}),
    conv2d {Wn} {bn} {xn} o hi wi = {tab} o hi wi := by
  intro o hi wi
  fin_cases o <;> fin_cases hi
{bullets}
""")
    return "\n".join(out)


def table_rows(name, stmt, tactic, oc):
    """per-(o, hi) lemmas `∀ wi, stmt o hi wi` + an aggregator over all cells."""
    out = []
    for o in range(oc):
        for hi in range(S):
            out.append(f"""theorem {name}_r{o}_{hi} : ∀ wi : Fin {S}, {stmt(fm(o, oc), fm(hi, S), 'wi')} := by
  intro wi
  fin_cases wi <;> {tactic}
""")
    bullets = "\n".join(f"  · exact {name}_r{o}_{hi} wi" for o in range(oc) for hi in range(S))
    seq = ";" if oc == 1 else " <;>"     # one channel: `<;>` trips the unnecessarySeqFocus linter
    out.append(f"""theorem {name} : ∀ (o : Fin {oc}) (hi wi : Fin {S}), {stmt('o', 'hi', 'wi')} := by
  intro o hi wi
  fin_cases o{seq} fin_cases hi
{bullets}
""")
    return "\n".join(out)


x1T, c1T, c2T, r2T = T["x1"], T["c1"], T["c2"], T["r2"]
fF, d3F, r3F, d4F = T["f"], T["d3"], T["r3"], T["d4"]

# cells with an all-zero conv2 input patch (the twins), and the window certificates
zero_cells = sorted({(2 * ho + CELLS[k][0], 2 * wo + CELLS[k][1])
                     for (ci, ho, wo), cert in T["certs"].items() if cert[0] == "live"
                     for k in range(4) if cert[2][k] in ("twin",) or
                     (cert[2][k] == "self" and "twin" in cert[2])})
zp_lemmas = "\n".join(f"""theorem zp_{r}_{s} : ∀ cc kh kw, convPad 3 3 x1V cc kh kw {fm(r, S)} {fm(s, S)} = 0 := by
  intro cc kh kw
  fin_cases cc <;> fin_cases kh <;> fin_cases kw <;> simp [convPad, x1V]
""" for r, s in zero_cells)


def cert_bullet(certs, ci, ho, wo, twin_term):
    """one window's certificate: dead, or a designated cell `m` with every other cell `m` itself,
    a twin of `m` (`twin_term(mr, ms, r, s)` proves it) or below it by the margin."""
    cert = certs[ci, ho, wo]
    if cert[0] == "dead":
        return f"""  · left
    intro cd
    fin_cases cd <;> norm_num [c2V, winRowInv, winColInv]"""
    _, m, kinds = cert
    ma, mb = CELLS[m]
    lines = [f"""  · right
    refine ⟨({fm(ma, 2)}, {fm(mb, 2)}), ?_⟩
    intro cd
    fin_cases cd"""]
    for k in range(4):
        r, s = 2 * ho + CELLS[k][0], 2 * wo + CELLS[k][1]
        mr, ms = 2 * ho + ma, 2 * wo + mb
        if kinds[k] == "self":
            lines.append("    · exact Or.inl rfl")
        elif kinds[k] == "twin":
            lines.append(f"    · exact Or.inr (Or.inl ({twin_term(mr, ms, r, s)}))")
        else:
            lines.append("    · exact Or.inr (Or.inr (by norm_num [c2V, winRowInv, winColInv]))")
    return "\n".join(lines)


def cert_lemmas_text(certs, rel, radius, twin_term):
    """`cert_{ci}_{ho}`: the certificates of one row of windows, twins `rel`, margin `radius`."""
    out = []
    for ci in range(C):
        for ho in range(H):
            bullets = "\n".join(cert_bullet(certs, ci, ho, wo, twin_term) for wo in range(H))
            out.append(f"""theorem cert_{ci}_{ho} : ∀ wo : Fin {H},
    (∀ cd : Fin 2 × Fin 2, c2V {fm(ci, C)} (winRowInv {fm(ho, H)} cd.1) (winColInv wo cd.2) ≤ 0) ∨
    ∃ m : Fin 2 × Fin 2, ∀ cd : Fin 2 × Fin 2,
      (winRowInv {fm(ho, H)} m.1, winColInv wo m.2) = (winRowInv {fm(ho, H)} cd.1, winColInv wo cd.2) ∨
      {rel} (winRowInv {fm(ho, H)} m.1, winColInv wo m.2)
        (winRowInv {fm(ho, H)} cd.1, winColInv wo cd.2) ∨
      c2V {fm(ci, C)} (winRowInv {fm(ho, H)} cd.1) (winColInv wo cd.2) + 2 * {lit(radius)} <
        c2V {fm(ci, C)} (winRowInv {fm(ho, H)} m.1) (winColInv wo m.2) := by
  intro wo
  fin_cases wo
{bullets}
""")
    return out


cert_lemmas = cert_lemmas_text(T["certs"], "ConvPatchEq 3 3 x1V", AR,
                               lambda mr, ms, r, s: f"ConvPatchEq.of_zero zp_{mr}_{ms} zp_{r}_{s}")
cert_bullets = "\n".join(f"  · exact cert_{ci}_{ho} wo" for ci in range(C) for ho in range(H))


def pool_bullets_text(fF):
    return "\n".join(
        f"""  · show maxPool2 (c := 2) (h := 6) (w := 6) r2V {fm(k // 36, 2)} {fm((k % 36) // 6, 6)} {fm(k % 6, 6)}
        = {lit(fF[k])}
    simp [maxPool2, r2V, max_def]
    try norm_num"""
        for k in range(C * H * H))


pool_bullets = pool_bullets_text(fF)

def weight_bound(name, mat, bound):
    """`|W i j| ≤ bound` for every entry, peeling the `![…]` literal row by row
    (`Fin.forall_fin_succ`): indexing it at a numeral far down the vector times out."""
    return f"""theorem {name} : ∀ i j, |{mat} i j| ≤ {lit(bound)} := by
  simp only [{mat}, Fin.forall_fin_succ, Matrix.cons_val_zero, Matrix.cons_val_succ,
    IsEmpty.forall_iff, and_true, abs_le]
  norm_num
"""


weight_bounds = "\n".join([weight_bound("hW3", "W3", W3B), weight_bound("hW4", "W4", W4B),
                           weight_bound("hW5", "W5", W5B)])


live_twin_windows = [(ci, ho, wo) for (ci, ho, wo), c in sorted(T["certs"].items())
                     if c[0] == "live" and "twin" in c[2]]
n_dead = sum(1 for c in T["certs"].values() if c[0] == "dead")

body = f'''import LeanMlir.Proofs.Training.SgdDescent.Cnn

/-! # Descent at TRAINED weights — the CNN conv2 rungs, through tied pool windows

**REDUCED CERTIFICATE MODEL** — this file's net is a 12×12-input, 2-channel MNIST CNN with an
8-wide dense head, NOT the canonical 28×28, 32-channel `cnnVerified`; chosen so every table is
exact rational arithmetic in the kernel. The canonical net's pool condition is measured on all
10000 test images by scripts/probes/mnist_pool_twin_probe.py.

`cnn_conv2_exact_sgd_descends` — one exact-gradient SGD step on the conv2 kernel decreases the
cross-entropy loss — instantiated at TRAINED, /128-rationalized weights (biases /16384) and REAL
MNIST test image #{IDX} (label {LBL}, classified correctly), every hypothesis discharged by exact
arithmetic: `trained_cnn_conv2_sgd_descends_concrete`, at learning rate `2⁻{KEXP}`. The conv2-bias
rung `cnn_conv2_bias_exact_sgd_descends` on the same net and image is
`trained_cnn_conv2_bias_sgd_descends_concrete`, at `2⁻{KEXPB}`: its radius sits inside the kernel
rung's, so the relu₂ and pool margins carry over by monotonicity. The conv1 rungs need a net with
a conv1 bias (`Trained.CnnDescentConv1`).

What the instance shows is the pool hypothesis. The image's blank corners make conv1's output
zero there (conv1 is bias-free, so a zero patch gives the padding's value), the cells of a
corner window then read identical all-zero patches, and their conv2 outputs are EQUAL — the
conv2 bias, positive in one channel. {len(live_twin_windows)} live windows tie that way and
{n_dead} are dead. The old `MaxPool2MarginQ` fails at every such window for every margin;
`MaxPool2MarginQUpTo` takes the twins `ConvPatchEq 3 3 x1V` (`ConvPatchEq.of_zero` per tied cell,
`zp_*`), and every other cell clears the margin (`cert_*`, through `windowMarginUpTo_of_cert`).
No pool-tie regularizer was used in training; the ties are the data's.

Net: 24×24-center-cropped MNIST, 2×2 block sums to 12×12 (exact pixel sums /1020), conv 1→2
3×3 SAME without bias → relu → conv 2→2 3×3 SAME → relu → maxpool 2×2 → dense 72→8 → relu →
dense 8→8 → relu → dense 8→10. The rungs move the conv2 kernel or bias; `x1V` = relu(conv1 image)
is their frozen input (`x1_eq`). The generator prints the net's test accuracy.

The step is the exact gradient, so the conclusion bounds the true loss, but the decrease
`lr·‖∇L‖₂²/2` is not shown positive: that needs a lower bound on the gradient, which the
softmax makes transcendental (`Trained.LinearDescent` gets one from a misclassified example).
Generated by `scripts/certs/trained_cnn_descent.py`; weights and input are DATA. -/

namespace Proofs
namespace TrainedCnnDescent

-- ════════════════════════════════════════════════════════════════
-- § Trained weights and the input (data)
-- ════════════════════════════════════════════════════════════════

/-- Test image #{IDX}, center-cropped and 2×2-summed to 12×12, exact pixel sums /1020. -/
noncomputable def T0 : Tensor3 1 12 12 :=
  {t3_lit(T["x0"])}

/-- conv1 kernel (1→2, 3×3), entries k/128; conv1 has no bias. -/
noncomputable def W1 : Kernel4 2 1 3 3 :=
  {k4_lit(W1r, C, 1)}

noncomputable def b1 : Vec 2 := ![(0 : ℝ), (0 : ℝ)]

/-- conv2 kernel (2→2, 3×3), entries k/128. -/
noncomputable def W2 : Kernel4 2 2 3 3 :=
  {k4_lit(W2r, C, C)}

noncomputable def b2 : Vec 2 := {vec_lit(b2r)}

/-- dense3 (72→8, input×output). -/
noncomputable def W3 : Mat (2*6*6) 8 :=
  {mat_lit(W3r)}

noncomputable def b3 : Vec 8 := {vec_lit(b3r)}

noncomputable def W4 : Mat 8 8 :=
  {mat_lit(W4r)}

noncomputable def b4 : Vec 8 := {vec_lit(b4r)}

noncomputable def W5 : Mat 8 10 :=
  {mat_lit(W5r)}

noncomputable def b5 : Vec 10 := {vec_lit(b5r)}

/-- The label of test image #{IDX}. -/
def lbl : Fin 10 := {LBL}

-- ════════════════════════════════════════════════════════════════
-- § conv1: the rung's frozen input is relu(conv1 image)
-- ════════════════════════════════════════════════════════════════

/-- conv1 pre-activations at the image, exact. -/
noncomputable def c1V : Tensor3 2 12 12 :=
  {t3_lit(c1T)}

{conv_rows("conv1_eq", "W1", "b1", "T0", "c1V", C)}
/-- relu(conv1) at the image: the conv2 rung's input, exact. -/
noncomputable def x1V : Tensor3 2 12 12 :=
  {t3_lit(x1T)}

{table_rows("x1_cell", lambda o, hi, wi: f"x1V {o} {hi} {wi} = if c1V {o} {hi} {wi} > 0 then c1V {o} {hi} {wi} else 0", "(simp [c1V, x1V]; try norm_num)", C)}
/-- **The rung's input is the real image's conv1 activation.** -/
theorem x1_eq : x1V = fun o hi wi =>
    if conv2d W1 b1 T0 o hi wi > 0 then conv2d W1 b1 T0 o hi wi else 0 := by
  funext o hi wi
  rw [conv1_eq, x1_cell]

{table_rows("x1_bound", lambda o, hi, wi: f"|x1V {o} {hi} {wi}| ≤ {lit(A)}", "(rw [abs_le]; constructor <;> norm_num [x1V])", C)}
-- ════════════════════════════════════════════════════════════════
-- § conv2: exact table, the relu₂ margin, the pooled vector
-- ════════════════════════════════════════════════════════════════

/-- conv2 pre-activations at the image, exact. -/
noncomputable def c2V : Tensor3 2 12 12 :=
  {t3_lit(c2T)}

{conv_rows("conv2_eq", "W2", "b2", "x1V", "c2V", C)}
theorem conv2_fun : conv2d W2 b2 x1V = c2V := by
  funext o hi wi
  exact conv2_eq o hi wi

{table_rows("c2_margin", lambda o, hi, wi: f"{lit(AR)} < |c2V {o} {hi} {wi}|", "(rw [lt_abs]; norm_num [c2V])", C)}
/-- relu(conv2) at the image (the max-pool input), exact. -/
noncomputable def r2V : Tensor3 2 12 12 :=
  {t3_lit(r2T)}

{table_rows("r2_cell", lambda o, hi, wi: f"r2V {o} {hi} {wi} = if c2V {o} {hi} {wi} > 0 then c2V {o} {hi} {wi} else 0", "(simp [c2V, r2V]; try norm_num)", C)}
theorem relu_c2 : relu (2 * (2*6) * (2*6)) (Tensor3.flatten c2V) = Tensor3.flatten r2V := by
  funext k
  obtain ⟨o, hi, wi, rfl⟩ := t3Idx_surj k
  rw [relu, flatten_t3Idx, flatten_t3Idx, r2_cell]

/-- The pooled feature vector (flattened maxpool output), exact. -/
noncomputable def p2f : Vec (2 * 6 * 6) := {vec_lit(fF)}

-- Decoding a flat index near the end of `Fin 72` back to `(ci, ho, wo)` recurses deeper than
-- the default; `max_def` is needed only where a window's maximum is not its first cell.
set_option maxRecDepth 8192 in
set_option linter.unusedSimpArgs false in
theorem pooled_eq : maxPoolFlat 2 6 6 (Tensor3.flatten r2V) = p2f := by
  show Tensor3.flatten (maxPool2 (Tensor3.unflatten (Tensor3.flatten r2V))) = p2f
  rw [Tensor3.unflatten_flatten]
  funext i
  fin_cases i
{pool_bullets}

-- ════════════════════════════════════════════════════════════════
-- § The pool margin up to twins: tied windows read all-zero patches
-- ════════════════════════════════════════════════════════════════

{zp_lemmas}
{"".join(cert_lemmas)}
/-- **The pool margin up to twins holds at the trained weights**, ties and all. -/
theorem pool_margin :
    MaxPool2MarginQUpTo (c := 2) (h := 6) (w := 6) {lit(AR)} (ConvPatchEq 3 3 x1V) c2V := by
  refine windowMarginUpTo_of_cert winRowInv winColInv (by norm_num) _
    (fun _ _ h => h.symm) (fun _ _ _ h₁ h₂ => h₁.trans h₂) ?_
  intro ci ho wo
  fin_cases ci <;> fin_cases ho
{cert_bullets}

-- ════════════════════════════════════════════════════════════════
-- § The dense head: the relu₃ and relu₄ margins
-- ════════════════════════════════════════════════════════════════

/-- dense3 pre-activations at the image, exact. -/
noncomputable def d3V : Fin 8 → ℝ := {vec_lit(d3F)}

theorem d3_eq : ∀ k, dense W3 b3 p2f k = d3V k := by
  intro k
  fin_cases k <;> (simp [dense, W3, b3, p2f, d3V, Fin.sum_univ_succ]; try norm_num)

/-- relu(dense3) at the image, exact. -/
noncomputable def r3V : Vec 8 := {vec_lit(r3F)}

theorem r3_eq : relu 8 (dense W3 b3 p2f) = r3V := by
  funext k
  show (if dense W3 b3 p2f k > 0 then dense W3 b3 p2f k else 0) = r3V k
  rw [d3_eq]
  fin_cases k <;> (simp [d3V, r3V]; try norm_num)

/-- dense4 pre-activations at the image, exact. -/
noncomputable def d4V : Fin 8 → ℝ := {vec_lit(d4F)}

theorem d4_eq : ∀ k, dense W4 b4 r3V k = d4V k := by
  intro k
  fin_cases k <;> (simp [dense, W4, b4, r3V, d4V, Fin.sum_univ_succ]; try norm_num)

theorem head_feat : maxPoolFlat 2 6 6 (relu (2 * (2*6) * (2*6))
    (Tensor3.flatten (conv2d W2 b2 x1V))) = p2f := by
  rw [conv2_fun, relu_c2, pooled_eq]

-- ════════════════════════════════════════════════════════════════
-- § Weight bounds
-- ════════════════════════════════════════════════════════════════

{weight_bounds}
-- ════════════════════════════════════════════════════════════════
-- § The descent instance
-- ════════════════════════════════════════════════════════════════

/-- The margin radius `a·(lr·G)` of the instance, exact. -/
theorem radius_eq : ({lit(A)} : ℝ) * ((1 / 2 ^ {KEXP} : ℝ) *
    cnnConv2GradBound 2 6 6 8 8 10 3 3 {lit(A)} {lit(W3B)} {lit(W4B)} {lit(W5B)}) = {lit(AR)} := by
  norm_num [cnnConv2GradBound]

/-- **One exact-gradient SGD step on the conv2 kernel of a trained MNIST CNN, at a real test image
    with tied pool windows, decreases the cross-entropy loss** by at least `lr·‖∇L‖₂²/2`, at
    `lr = 2⁻{KEXP}`. Every hypothesis of `cnn_conv2_exact_sgd_descends` is discharged above. -/
theorem trained_cnn_conv2_sgd_descends_concrete :
    (cnnConv2KernelLoss (h := 6) (w := 6) b2 x1V W3 b3 W4 b4 W5 b5 lbl) (Kernel4.flatten W2 -
        (1 / 2 ^ {KEXP} : ℝ) • gradAt (cnnConv2KernelLoss (h := 6) (w := 6) b2 x1V W3 b3 W4 b4 W5 b5 lbl)
          (Kernel4.flatten W2)) ≤
      (cnnConv2KernelLoss (h := 6) (w := 6) b2 x1V W3 b3 W4 b4 W5 b5 lbl) (Kernel4.flatten W2) -
        (1 / 2 ^ {KEXP} : ℝ) * (∑ idx, gradAt (cnnConv2KernelLoss (h := 6) (w := 6) b2 x1V W3 b3 W4 b4
          W5 b5 lbl) (Kernel4.flatten W2) idx ^ 2) / 2 := by
  refine cnn_conv2_exact_sgd_descends (h := 6) (w := 6) W2 b2 x1V W3 b3 W4 b4 W5 b5 lbl
    (ConvPatchEq 3 3 x1V) (by norm_num) x1_bound (fun _ _ h => h) (by norm_num) hW3
    (by norm_num) hW4 (by norm_num) hW5 (by norm_num) ?_ ?_ ?_ ?_ ?_ ?_
  · intro k
    obtain ⟨o, hi, wi, rfl⟩ := t3Idx_surj k
    rw [radius_eq, flatten_t3Idx, conv2_fun]
    exact c2_margin o hi wi
  · rw [radius_eq, conv2_fun]
    exact pool_margin
  · intro l
    rw [radius_eq, head_feat, d3_eq]
    fin_cases l <;> (rw [lt_abs]; norm_num [d3V])
  · intro q
    rw [radius_eq, head_feat, r3_eq, d4_eq]
    fin_cases q <;> (rw [lt_abs]; norm_num [d4V])
  · rw [radius_eq]; norm_num
  · rw [radius_eq]; norm_num

-- ════════════════════════════════════════════════════════════════
-- § The conv2-bias rung on the same net and image
-- ════════════════════════════════════════════════════════════════

/-- The conv2-bias rung's radius `lr·G_b`, exact; it sits inside the kernel rung's. -/
theorem radius_bias_eq : (1 / 2 ^ {KEXPB} : ℝ) *
    cnnConv2BiasGradBound 2 6 6 8 8 10 {lit(W3B)} {lit(W4B)} {lit(W5B)} = {lit(RB2)} := by
  norm_num [cnnConv2BiasGradBound]

/-- **One exact-gradient SGD step on the conv2 BIAS of the same trained MNIST CNN, at the same
    test image, decreases the cross-entropy loss** by at least `lr·‖∇L‖₂²/2`, at `lr = 2⁻{KEXPB}`.
    Every hypothesis of `cnn_conv2_bias_exact_sgd_descends` is discharged above; the relu₂ and
    pool margins are the kernel rung's (`c2_margin`, `pool_margin`) at a smaller radius. -/
theorem trained_cnn_conv2_bias_sgd_descends_concrete :
    (cnnConv2BiasLoss (h := 6) (w := 6) W2 x1V W3 b3 W4 b4 W5 b5 lbl) (b2 -
        (1 / 2 ^ {KEXPB} : ℝ) • gradAt (cnnConv2BiasLoss (h := 6) (w := 6) W2 x1V W3 b3 W4 b4 W5
          b5 lbl) b2) ≤
      (cnnConv2BiasLoss (h := 6) (w := 6) W2 x1V W3 b3 W4 b4 W5 b5 lbl) b2 -
        (1 / 2 ^ {KEXPB} : ℝ) * (∑ o, gradAt (cnnConv2BiasLoss (h := 6) (w := 6) W2 x1V W3 b3 W4
          b4 W5 b5 lbl) b2 o ^ 2) / 2 := by
  refine cnn_conv2_bias_exact_sgd_descends (h := 6) (w := 6) W2 b2 x1V W3 b3 W4 b4 W5 b5 lbl
    (ConvPatchEq 3 3 x1V) (fun _ _ h => h) (by norm_num) hW3 (by norm_num) hW4 (by norm_num) hW5
    (by norm_num) ?_ ?_ ?_ ?_ ?_ ?_
  · intro k
    obtain ⟨o, hi, wi, rfl⟩ := t3Idx_surj k
    rw [radius_bias_eq, flatten_t3Idx, conv2_fun]
    exact lt_of_le_of_lt (by norm_num) (c2_margin o hi wi)
  · rw [radius_bias_eq, conv2_fun]
    exact WindowMarginUpTo.mono winRowInv winColInv (by norm_num) pool_margin
  · intro l
    rw [radius_bias_eq, head_feat, d3_eq]
    fin_cases l <;> (rw [lt_abs]; norm_num [d3V])
  · intro q
    rw [radius_bias_eq, head_feat, r3_eq, d4_eq]
    fin_cases q <;> (rw [lt_abs]; norm_num [d4V])
  · rw [radius_bias_eq]; norm_num
  · rw [radius_bias_eq]; norm_num

end TrainedCnnDescent
end Proofs
'''

if os.environ.get("CNN_DESCENT_NO_WRITE") != "1":
    with open(OUT, "w") as fh:
        fh.write(body)
    print(f"wrote {OUT} ({len(body.splitlines())} lines)", flush=True)

# ================================================================ net B: the conv1 rungs
OUT1 = os.path.join(ROOT, "LeanMlir/Proofs/Training/Trained/CnnDescentConv1.lean")
QB = quantize(train(True, "B"), "B")


def blank_field(x0, r, s):
    """conv2-output cell (r, s) has a blank two-conv receptive field with every outer read in
    bounds: (r, s) and its 8 neighbours are image cells and each one's 3x3 patch of x0 is all zero.
    Two such cells are `ConvPatchEq2 3 3 x0` twins (`tw_*`)."""
    if not (1 <= r <= S - 2 and 1 <= s <= S - 2):
        return False
    return all(x0[0, a, b] == 0 for a in range(max(0, r - 2), min(S, r + 3))
               for b in range(max(0, s - 2), min(S, s + 3)))


def analyse_b(i):
    """Exact tables + per-window certificates of net B at test image i, or None."""
    x0 = np.vectorize(lambda s: Fraction(int(s), DEN_X))(Ste[i])[None, :, :]
    c1 = conv_exact(x0, QB["W1"], QB["b1"])
    if any(v == 0 for v in c1.flat):
        return None
    x1 = relu_exact(c1)
    c2 = conv_exact(x1, QB["W2"], QB["b2"])
    if any(v == 0 for v in c2.flat):
        return None
    certs, live_twin, gaps = {}, 0, []
    for ci in range(C):
        for ho in range(H):
            for wo in range(H):
                pos = [(2 * ho + a, 2 * wo + b) for a, b in CELLS]
                vals = [c2[ci, r, s] for r, s in pos]
                if all(v <= 0 for v in vals):
                    certs[ci, ho, wo] = ("dead",)
                    continue
                m = max(range(4), key=lambda k: vals[k])
                kinds = []
                for k in range(4):
                    if k == m:
                        kinds.append("self")
                    elif blank_field(x0, *pos[m]) and blank_field(x0, *pos[k]):
                        kinds.append("twin")
                    elif vals[k] < vals[m]:
                        kinds.append("gap"); gaps.append(vals[m] - vals[k])
                    else:
                        return None
                if "twin" in kinds:
                    live_twin += 1
                certs[ci, ho, wo] = ("live", m, kinds)
    r2 = relu_exact(c2)
    pool = np.empty((C, H, H), dtype=object)
    for ci in range(C):
        for ho in range(H):
            for wo in range(H):
                pool[ci, ho, wo] = max(r2[ci, 2 * ho + a, 2 * wo + b] for a, b in CELLS)
    f = pool.reshape(-1)
    d3 = np.array([sum(f[j] * QB["W3"][j, k] for j in range(C * H * H)) + QB["b3"][k]
                   for k in range(D1)], dtype=object)
    if any(v == 0 for v in d3):
        return None
    r3 = np.array([v if v > 0 else ZF for v in d3], dtype=object)
    d4 = np.array([sum(r3[j] * QB["W4"][j, k] for j in range(D1)) + QB["b4"][k]
                   for k in range(D1)], dtype=object)
    if any(v == 0 for v in d4):
        return None
    r4 = np.array([v if v > 0 else ZF for v in d4], dtype=object)
    z5 = np.array([sum(r4[j] * QB["W5"][j, k] for j in range(D1)) + QB["b5"][k]
                   for k in range(NC)], dtype=object)
    return dict(x0=x0, c1=c1, x1=x1, c2=c2, r2=r2, f=f, d3=d3, r3=r3, d4=d4, z5=z5,
                certs=certs, live_twin=live_twin, mingap=min(gaps) if gaps else None)


chosen_b = None
for i in range(len(Ste)):
    t = analyse_b(i)
    if t is None or t["live_twin"] == 0:
        continue
    if int(np.argmax([float(v) for v in t["z5"]])) != int(yte[i]):
        continue
    chosen_b = (i, t)
    break
if chosen_b is None:
    sys.exit("NO CONV1 INSTANCE FOUND")
IDXB, TB = chosen_b
print(f"conv1 instance: test #{IDXB} label {yte[IDXB]}, live two-layer twin windows "
      f"{TB['live_twin']}", flush=True)

A1 = frac_up(max(abs(v) for v in TB["x0"].flat))
W2Bb = frac_up(max(abs(v) for v in QB["W2"].flat), DEN_W)
W3Bb = frac_up(max(abs(v) for v in QB["W3"].flat), DEN_W)
W4Bb = frac_up(max(abs(v) for v in QB["W4"].flat), DEN_W)
W5Bb = frac_up(max(abs(v) for v in QB["W5"].flat), DEN_W)
M = C * 9                      # c·kH·kW, conv2's locality factor
P1 = C * 1 * 9
BHB = D1 * (W3Bb * (D1 * (W4Bb * (NC * (W5Bb * 1)))))
G1 = P1 * (K2 * (M * (W2Bb * A1)) * BHB)
G1B = C * (K2 * (M * (W2Bb * 1)) * BHB)
minb1 = min(abs(v) for v in TB["c1"].flat)
minb2 = min(abs(v) for v in TB["c2"].flat)
minb3 = min(abs(v) for v in TB["d3"])
minb4 = min(abs(v) for v in TB["d4"])


def ok_conv1(lr, rho_r, num, n):
    """every hypothesis of the conv1 rungs at the conv1-preactivation radius `rho_r`."""
    z2r = M * (W2Bb * rho_r)
    if not (rho_r < minb1 and z2r < minb2 and 2 * z2r < TB["mingap"]):
        return False
    s3 = M * (W2Bb * (K2 * rho_r))
    if not (W3Bb * s3 < minb3 and W4Bb * (D1 * (W3Bb * s3)) < minb4):
        return False
    dl = W5Bb * (D1 * (W4Bb * (D1 * (W3Bb * s3))))
    return 2 * dl < 1 and num / (1 - 2 * dl) * lr * n <= Fraction(1, 4)


NUM1B = 2 * NC * K2 ** 2 * M ** 2 * D1 ** 2 * D1 ** 2 * W2Bb ** 2 * W3Bb ** 2 * W4Bb ** 2 * W5Bb ** 2
KEXP1 = next(k for k in range(1, 300)
             if ok_conv1(Fraction(1, 2 ** k), A1 * (Fraction(1, 2 ** k) * G1), NUM1B * A1 ** 2, P1))
R1A = A1 * (Fraction(1, 2 ** KEXP1) * G1)        # the relu₁ margin radius a·(lr·G), exact
R1Z2 = M * (W2Bb * R1A)                            # the relu₂ / pool margin radius
KEXP1B = next(k for k in range(1, 300)
              if Fraction(1, 2 ** k) * G1B <= R1A and
              ok_conv1(Fraction(1, 2 ** k), Fraction(1, 2 ** k) * G1B, NUM1B, C))
RB1 = Fraction(1, 2 ** KEXP1B) * G1B
print(f"conv1 rungs: a = {A1}, w2 = {W2Bb}, kernel lr = 2^-{KEXP1}, bias lr = 2^-{KEXP1B}",
      flush=True)
LBLB = int(yte[IDXB])

# two-layer twin certificates: one `tw_*` per ordered (designated cell, twin) pair
tw_pairs = sorted({(2 * ho + CELLS[cert[1]][0], 2 * wo + CELLS[cert[1]][1],
                    2 * ho + CELLS[k][0], 2 * wo + CELLS[k][1])
                   for (ci, ho, wo), cert in TB["certs"].items() if cert[0] == "live"
                   for k in range(4) if cert[2][k] == "twin"})
tw_lemmas = "\n".join(f"""theorem tw_{mr}_{ms}_{r}_{s} : ConvPatchEq2 3 3 T0 ({fm(mr, S)}, {fm(ms, S)})
    ({fm(r, S)}, {fm(s, S)}) := by
  intro kh kw
  fin_cases kh <;> fin_cases kw <;>
    refine ⟨by simp, fun hp hq => ConvPatchEq.of_zero ?_ ?_⟩ <;>
    intro cc kh' kw' <;> fin_cases cc <;> fin_cases kh' <;> fin_cases kw' <;> simp [convPad, T0]
""" for mr, ms, r, s in tw_pairs)
cert_lemmas_b = cert_lemmas_text(TB["certs"], "ConvPatchEq2 3 3 T0", R1Z2,
                                 lambda mr, ms, r, s: f"tw_{mr}_{ms}_{r}_{s}")
live_twin_b = sum(1 for c in TB["certs"].values() if c[0] == "live" and "twin" in c[2])
n_dead_b = sum(1 for c in TB["certs"].values() if c[0] == "dead")


def kernel_bound(name, ker, bound):
    """`|W o c kh kw| ≤ bound` for every entry of a `Kernel4` literal, peeled as `weight_bound`."""
    return f"""theorem {name} : ∀ o cc kh kw, |{ker} o cc kh kw| ≤ {lit(bound)} := by
  simp only [{ker}, Fin.forall_fin_succ, Matrix.cons_val_zero, Matrix.cons_val_succ,
    IsEmpty.forall_iff, and_true, abs_le]
  norm_num
"""


DIMS = "(h := 6) (w := 6)"
L1 = f"cnnConv1KernelLoss {DIMS} b1 T0 W2 b2 W3 b3 W4 b4 W5 b5 lbl"
L1B = f"cnnConv1BiasLoss {DIMS} W1 T0 W2 b2 W3 b3 W4 b4 W5 b5 lbl"

body_b = f'''import LeanMlir.Proofs.Training.SgdDescent.Cnn

/-! # Descent at TRAINED weights — the CNN conv1 rungs, through two-layer twins

**REDUCED CERTIFICATE MODEL** — the same 12×12-input, 2-channel MNIST CNN shape as
`Trained.CnnDescent`, NOT the canonical 28×28, 32-channel `cnnVerified`, chosen so every table is
exact rational arithmetic in the kernel.

`cnn_conv1_exact_sgd_descends` and `cnn_conv1_bias_exact_sgd_descends` — one exact-gradient SGD
step on the FIRST conv's kernel, or its bias, decreases the cross-entropy loss — instantiated at
TRAINED, /128-rationalized weights (biases /16384) and REAL MNIST test image #{IDXB} (label
{LBLB}, classified correctly), every hypothesis discharged by exact arithmetic:
`trained_cnn_conv1_sgd_descends_concrete` at learning rate `2⁻{KEXP1}` and
`trained_cnn_conv1_bias_sgd_descends_concrete` at `2⁻{KEXP1B}` (its radius sits inside the kernel
rung's, so every relu and pool margin carries over by monotonicity).

This net has a trained conv1 bias. `Trained.CnnDescent`'s bias-free conv1 cannot serve these
rungs: their relu₁ margin needs every conv1 pre-activation nonzero, and a bias-free conv1 is
exactly zero on a blank patch.

The pool hypothesis takes two-layer twins `ConvPatchEq2 3 3 T0`: conv2-output cells whose 5×5
receptive fields through both convs lie in a blank image region, every outer read in bounds
(`tw_*`), so their conv2 outputs are equal for every conv1 kernel and bias. {live_twin_b} live
windows tie that way and {n_dead_b} are dead; every other cell clears the margin (`cert_*`, through
`windowMarginUpTo_of_cert`). No pool-tie regularizer was used in training.

Net: 24×24-center-cropped MNIST, 2×2 block sums to 12×12 (exact pixel sums /1020), conv 1→2
3×3 SAME + bias → relu → conv 2→2 3×3 SAME → relu → maxpool 2×2 → dense 72→8 → relu →
dense 8→8 → relu → dense 8→10. The generator prints the net's test accuracy.

As in `Trained.CnnDescent`, the step is the exact gradient and the decrease `lr·‖∇L‖₂²/2` is not
shown positive. Generated by `scripts/certs/trained_cnn_descent.py`; weights and input are DATA. -/

namespace Proofs
namespace TrainedCnnDescentConv1

-- ════════════════════════════════════════════════════════════════
-- § Trained weights and the input (data)
-- ════════════════════════════════════════════════════════════════

/-- Test image #{IDXB}, center-cropped and 2×2-summed to 12×12, exact pixel sums /1020. -/
noncomputable def T0 : Tensor3 1 12 12 :=
  {t3_lit(TB["x0"])}

/-- conv1 kernel (1→2, 3×3), entries k/128. -/
noncomputable def W1 : Kernel4 2 1 3 3 :=
  {k4_lit(QB["W1"], C, 1)}

noncomputable def b1 : Vec 2 := {vec_lit(QB["b1"])}

/-- conv2 kernel (2→2, 3×3), entries k/128. -/
noncomputable def W2 : Kernel4 2 2 3 3 :=
  {k4_lit(QB["W2"], C, C)}

noncomputable def b2 : Vec 2 := {vec_lit(QB["b2"])}

/-- dense3 (72→8, input×output). -/
noncomputable def W3 : Mat (2*6*6) 8 :=
  {mat_lit(QB["W3"])}

noncomputable def b3 : Vec 8 := {vec_lit(QB["b3"])}

noncomputable def W4 : Mat 8 8 :=
  {mat_lit(QB["W4"])}

noncomputable def b4 : Vec 8 := {vec_lit(QB["b4"])}

noncomputable def W5 : Mat 8 10 :=
  {mat_lit(QB["W5"])}

noncomputable def b5 : Vec 10 := {vec_lit(QB["b5"])}

/-- The label of test image #{IDXB}. -/
def lbl : Fin 10 := {LBLB}

{table_rows("x0_bound", lambda o, hi, wi: f"|T0 {o} {hi} {wi}| ≤ {lit(A1)}", "(rw [abs_le]; constructor <;> norm_num [T0])", 1)}
-- ════════════════════════════════════════════════════════════════
-- § conv1: exact table, the relu₁ margin, the conv2 input
-- ════════════════════════════════════════════════════════════════

/-- conv1 pre-activations at the image, exact. -/
noncomputable def c1V : Tensor3 2 12 12 :=
  {t3_lit(TB["c1"])}

{conv_rows("conv1_eq", "W1", "b1", "T0", "c1V", C)}
theorem conv1_fun : conv2d W1 b1 T0 = c1V := by
  funext o hi wi
  exact conv1_eq o hi wi

{table_rows("c1_margin", lambda o, hi, wi: f"{lit(R1A)} < |c1V {o} {hi} {wi}|", "(rw [lt_abs]; norm_num [c1V])", C)}
/-- relu(conv1) at the image (conv2's input), exact. -/
noncomputable def x1V : Tensor3 2 12 12 :=
  {t3_lit(TB["x1"])}

{table_rows("x1_cell", lambda o, hi, wi: f"x1V {o} {hi} {wi} = if c1V {o} {hi} {wi} > 0 then c1V {o} {hi} {wi} else 0", "(simp [c1V, x1V]; try norm_num)", C)}
theorem relu_c1 : relu (2 * (2*6) * (2*6)) (Tensor3.flatten c1V) = Tensor3.flatten x1V := by
  funext k
  obtain ⟨o, hi, wi, rfl⟩ := t3Idx_surj k
  rw [relu, flatten_t3Idx, flatten_t3Idx, x1_cell]

/-- **conv2's input is the real image's conv1 activation.** -/
theorem x1_fun : Tensor3.unflatten (relu (2 * (2*6) * (2*6)) (Tensor3.flatten (conv2d W1 b1 T0))) =
    x1V := by
  rw [conv1_fun, relu_c1, Tensor3.unflatten_flatten]

-- ════════════════════════════════════════════════════════════════
-- § conv2: exact table, the relu₂ margin, the pooled vector
-- ════════════════════════════════════════════════════════════════

/-- conv2 pre-activations at the image, exact. -/
noncomputable def c2V : Tensor3 2 12 12 :=
  {t3_lit(TB["c2"])}

{conv_rows("conv2_eq", "W2", "b2", "x1V", "c2V", C)}
theorem conv2_fun : conv2d W2 b2 x1V = c2V := by
  funext o hi wi
  exact conv2_eq o hi wi

{table_rows("c2_margin", lambda o, hi, wi: f"{lit(R1Z2)} < |c2V {o} {hi} {wi}|", "(rw [lt_abs]; norm_num [c2V])", C)}
/-- relu(conv2) at the image (the max-pool input), exact. -/
noncomputable def r2V : Tensor3 2 12 12 :=
  {t3_lit(TB["r2"])}

{table_rows("r2_cell", lambda o, hi, wi: f"r2V {o} {hi} {wi} = if c2V {o} {hi} {wi} > 0 then c2V {o} {hi} {wi} else 0", "(simp [c2V, r2V]; try norm_num)", C)}
theorem relu_c2 : relu (2 * (2*6) * (2*6)) (Tensor3.flatten c2V) = Tensor3.flatten r2V := by
  funext k
  obtain ⟨o, hi, wi, rfl⟩ := t3Idx_surj k
  rw [relu, flatten_t3Idx, flatten_t3Idx, r2_cell]

/-- The pooled feature vector (flattened maxpool output), exact. -/
noncomputable def p2f : Vec (2 * 6 * 6) := {vec_lit(TB["f"])}

-- Decoding a flat index near the end of `Fin 72` back to `(ci, ho, wo)` recurses deeper than
-- the default; `max_def` is needed only where a window's maximum is not its first cell.
set_option maxRecDepth 8192 in
set_option linter.unusedSimpArgs false in
theorem pooled_eq : maxPoolFlat 2 6 6 (Tensor3.flatten r2V) = p2f := by
  show Tensor3.flatten (maxPool2 (Tensor3.unflatten (Tensor3.flatten r2V))) = p2f
  rw [Tensor3.unflatten_flatten]
  funext i
  fin_cases i
{pool_bullets_text(TB["f"])}

-- ════════════════════════════════════════════════════════════════
-- § The pool margin up to two-layer twins
-- ════════════════════════════════════════════════════════════════

{tw_lemmas}
{"".join(cert_lemmas_b)}
/-- **The pool margin up to two-layer twins holds at the trained weights**, ties and all. -/
theorem pool_margin :
    MaxPool2MarginQUpTo (c := 2) (h := 6) (w := 6) {lit(R1Z2)} (ConvPatchEq2 3 3 T0) c2V := by
  refine windowMarginUpTo_of_cert winRowInv winColInv (by norm_num) _
    (fun _ _ h => h.symm) (fun _ _ _ h₁ h₂ => h₁.trans h₂) ?_
  intro ci ho wo
  fin_cases ci <;> fin_cases ho
{cert_bullets}

-- ════════════════════════════════════════════════════════════════
-- § The dense head: the relu₃ and relu₄ margins
-- ════════════════════════════════════════════════════════════════

/-- dense3 pre-activations at the image, exact. -/
noncomputable def d3V : Fin 8 → ℝ := {vec_lit(TB["d3"])}

theorem d3_eq : ∀ k, dense W3 b3 p2f k = d3V k := by
  intro k
  fin_cases k <;> (simp [dense, W3, b3, p2f, d3V, Fin.sum_univ_succ]; try norm_num)

/-- relu(dense3) at the image, exact. -/
noncomputable def r3V : Vec 8 := {vec_lit(TB["r3"])}

theorem r3_eq : relu 8 (dense W3 b3 p2f) = r3V := by
  funext k
  show (if dense W3 b3 p2f k > 0 then dense W3 b3 p2f k else 0) = r3V k
  rw [d3_eq]
  fin_cases k <;> (simp [d3V, r3V]; try norm_num)

/-- dense4 pre-activations at the image, exact. -/
noncomputable def d4V : Fin 8 → ℝ := {vec_lit(TB["d4"])}

theorem d4_eq : ∀ k, dense W4 b4 r3V k = d4V k := by
  intro k
  fin_cases k <;> (simp [dense, W4, b4, r3V, d4V, Fin.sum_univ_succ]; try norm_num)

theorem head_feat : maxPoolFlat 2 6 6 (relu (2 * (2*6) * (2*6))
    (Tensor3.flatten (conv2d W2 b2 x1V))) = p2f := by
  rw [conv2_fun, relu_c2, pooled_eq]

-- ════════════════════════════════════════════════════════════════
-- § Weight bounds
-- ════════════════════════════════════════════════════════════════

{kernel_bound("hW2", "W2", W2Bb)}
{weight_bound("hW3", "W3", W3Bb)}
{weight_bound("hW4", "W4", W4Bb)}
{weight_bound("hW5", "W5", W5Bb)}
-- ════════════════════════════════════════════════════════════════
-- § The descent instances
-- ════════════════════════════════════════════════════════════════

/-- The relu₁ margin radius `a·(lr·G)` of the kernel instance, exact. -/
theorem radius_eq : ({lit(A1)} : ℝ) * ((1 / 2 ^ {KEXP1} : ℝ) *
    cnnConv1GradBound 1 2 6 6 8 8 10 3 3 {lit(A1)} {lit(W2Bb)} {lit(W3Bb)} {lit(W4Bb)} {lit(W5Bb)}) =
      {lit(R1A)} := by
  norm_num [cnnConv1GradBound]

/-- **One exact-gradient SGD step on the conv1 kernel of a trained MNIST CNN, at a real test image
    with two-layer-twin pool ties, decreases the cross-entropy loss** by at least
    `lr·‖∇L‖₂²/2`, at `lr = 2⁻{KEXP1}`. Every hypothesis of `cnn_conv1_exact_sgd_descends` is
    discharged above. -/
theorem trained_cnn_conv1_sgd_descends_concrete :
    ({L1}) (Kernel4.flatten W1 -
        (1 / 2 ^ {KEXP1} : ℝ) • gradAt ({L1}) (Kernel4.flatten W1)) ≤
      ({L1}) (Kernel4.flatten W1) -
        (1 / 2 ^ {KEXP1} : ℝ) * (∑ idx, gradAt ({L1}) (Kernel4.flatten W1) idx ^ 2) / 2 := by
  refine cnn_conv1_exact_sgd_descends {DIMS} W1 b1 T0 W2 b2 W3 b3 W4 b4 W5 b5 lbl
    (ConvPatchEq2 3 3 T0) (by norm_num) x0_bound (fun _ _ h => h) (by norm_num) hW2
    (by norm_num) hW3 (by norm_num) hW4 (by norm_num) hW5 (by norm_num) ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · intro k
    obtain ⟨o, hi, wi, rfl⟩ := t3Idx_surj k
    rw [radius_eq, flatten_t3Idx, conv1_fun]
    exact c1_margin o hi wi
  · intro k
    obtain ⟨o, hi, wi, rfl⟩ := t3Idx_surj k
    rw [radius_eq, x1_fun, flatten_t3Idx, conv2_fun]
    exact lt_of_le_of_lt (by norm_num) (c2_margin o hi wi)
  · rw [radius_eq, x1_fun, conv2_fun]
    exact WindowMarginUpTo.mono winRowInv winColInv (by norm_num) pool_margin
  · intro l
    rw [radius_eq, x1_fun, head_feat, d3_eq]
    fin_cases l <;> (rw [lt_abs]; norm_num [d3V])
  · intro q
    rw [radius_eq, x1_fun, head_feat, r3_eq, d4_eq]
    fin_cases q <;> (rw [lt_abs]; norm_num [d4V])
  · rw [radius_eq]; norm_num
  · rw [radius_eq]; norm_num

/-- The bias instance's radius `lr·G_b`, exact; it sits inside the kernel instance's `a·lr·G`. -/
theorem radius_bias_eq : (1 / 2 ^ {KEXP1B} : ℝ) *
    cnnConv1BiasGradBound 2 6 6 8 8 10 3 3 {lit(W2Bb)} {lit(W3Bb)} {lit(W4Bb)} {lit(W5Bb)} =
      {lit(RB1)} := by
  norm_num [cnnConv1BiasGradBound]

/-- **One exact-gradient SGD step on the conv1 BIAS of the same trained MNIST CNN, at the same
    test image, decreases the cross-entropy loss** by at least `lr·‖∇L‖₂²/2`, at
    `lr = 2⁻{KEXP1B}`. Every hypothesis of `cnn_conv1_bias_exact_sgd_descends` is discharged above;
    the relu₁, relu₂ and pool margins are the kernel instance's (`c1_margin`, `c2_margin`,
    `pool_margin`) at a smaller radius. -/
theorem trained_cnn_conv1_bias_sgd_descends_concrete :
    ({L1B}) (b1 - (1 / 2 ^ {KEXP1B} : ℝ) • gradAt ({L1B}) b1) ≤
      ({L1B}) b1 - (1 / 2 ^ {KEXP1B} : ℝ) * (∑ o, gradAt ({L1B}) b1 o ^ 2) / 2 := by
  refine cnn_conv1_bias_exact_sgd_descends {DIMS} W1 b1 T0 W2 b2 W3 b3 W4 b4 W5 b5 lbl
    (ConvPatchEq2 3 3 T0) (fun _ _ h => h) (by norm_num) hW2 (by norm_num) hW3 (by norm_num) hW4
    (by norm_num) hW5 (by norm_num) ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · intro k
    obtain ⟨o, hi, wi, rfl⟩ := t3Idx_surj k
    rw [radius_bias_eq, flatten_t3Idx, conv1_fun]
    exact lt_of_le_of_lt (by norm_num) (c1_margin o hi wi)
  · intro k
    obtain ⟨o, hi, wi, rfl⟩ := t3Idx_surj k
    rw [radius_bias_eq, x1_fun, flatten_t3Idx, conv2_fun]
    exact lt_of_le_of_lt (by norm_num) (c2_margin o hi wi)
  · rw [radius_bias_eq, x1_fun, conv2_fun]
    exact WindowMarginUpTo.mono winRowInv winColInv (by norm_num) pool_margin
  · intro l
    rw [radius_bias_eq, x1_fun, head_feat, d3_eq]
    fin_cases l <;> (rw [lt_abs]; norm_num [d3V])
  · intro q
    rw [radius_bias_eq, x1_fun, head_feat, r3_eq, d4_eq]
    fin_cases q <;> (rw [lt_abs]; norm_num [d4V])
  · rw [radius_bias_eq]; norm_num
  · rw [radius_bias_eq]; norm_num

end TrainedCnnDescentConv1
end Proofs
'''

if os.environ.get("CNN_DESCENT_NO_WRITE") != "1":
    with open(OUT1, "w") as fh:
        fh.write(body_b)
    print(f"wrote {OUT1} ({len(body_b.splitlines())} lines)", flush=True)
