"""Trained-weight whole-net VJP witness, CNN rung (post-audit gap #3).

Emits LeanMlir/Proofs/TrainedCnnWitness.lean: the Chapter-3 mnistCnnNoBn
conditional whole-network VJP (`mnistCnnNoBn_has_vjp_at`) instantiated at
TRAINED, /128-rationalized weights and a REAL test input, with every
smoothness hypothesis discharged from exact rational arithmetic:

  h1  : all 72 conv1 pre-activations != 0
  h2  : all 72 conv2 pre-activations != 0
  h_mp: every 2x2 maxpool window of relu(conv2) has 4 pairwise-distinct values
  h3  : all 8 dense3 pre-activations != 0
  h4  : all 8 dense4 pre-activations != 0

Architecture (mnistCnnNoBnForward at ic=1, c=2, h=w=3, d1=8, nClasses=10,
kH=kW=3): 24x24-center-cropped MNIST, 4x4-average-pooled to 6x6 (exact
pixel-sum rationals s/4080), conv1 1->2 3x3 SAME, relu, conv2 2->2 3x3 SAME,
relu, maxpool 2x2 -> 2x3x3, flatten (C-order = Tensor3.flatten), dense 18->8,
relu, dense 8->8, relu, dense 8->10.

conv2d semantics mirrored from LeanMlir/Proofs/Architectures/CNN.lean:130 (cross-correlation,
SAME zero padding pH=pW=1). The witness search enforces the h_mp condition
(relu zeros collide, so each window may contain at most one negative conv2
pre-activation, and the positive values must be pairwise distinct exactly).
"""
import numpy as np, struct, sys
from fractions import Fraction

D = "/home/skoonce/lean/klawd_max_power/lean4-jax/data/"
OUT = "/home/skoonce/lean/klawd_max_power/lean4-jax/LeanMlir/Proofs/TrainedCnnWitness.lean"
DEN_W = 128          # weight rationalization grid
DEN_X = 4080         # 16 * 255 exact pooled-pixel denominator
C = 2                # conv channels
D1 = 8               # dense width
NC = 10

def load_images(fn):
    with open(fn, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)

def load_labels(fn):
    with open(fn, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

Xtr_raw = load_images(D + "train-images-idx3-ubyte"); ytr = load_labels(D + "train-labels-idx1-ubyte")
Xte_raw = load_images(D + "t10k-images-idx3-ubyte"); yte = load_labels(D + "t10k-labels-idx1-ubyte")

def pool6_sums(X):
    """center-crop 28->24 (rows/cols 2..26), 4x4 block sums -> (N,6,6) ints 0..4080."""
    Xc = X[:, 2:26, 2:26].astype(np.int64)
    return Xc.reshape(-1, 6, 4, 6, 4).sum(axis=(2, 4))

Str = pool6_sums(Xtr_raw); Ste = pool6_sums(Xte_raw)
Xtr = Str / float(DEN_X); Xte = Ste / float(DEN_X)

# ---------------------------------------------------------------- training
rng = np.random.default_rng(0)

def pad1(x):  # x: (N, c, 6, 6) -> (N, c, 8, 8)
    return np.pad(x, ((0,0),(0,0),(1,1),(1,1)))

def im2col(x):  # x: (N, c, 6, 6) -> (N, 36, c*9) patches, cross-correlation order
    xp = pad1(x)
    N, c = x.shape[0], x.shape[1]
    cols = np.empty((N, 6, 6, c, 3, 3), dtype=x.dtype)
    for kh in range(3):
        for kw in range(3):
            cols[:, :, :, :, kh, kw] = xp[:, :, kh:kh+6, kw:kw+6].transpose(0,2,3,1)
    return cols.reshape(N, 36, c*9)

def conv_fwd(x, Wm, b):  # Wm: (oc, cin*9), b: (oc,)  -> (N, oc, 6, 6), plus patches
    P = im2col(x)
    y = P @ Wm.T + b            # (N, 36, oc)
    return y.transpose(0, 2, 1).reshape(-1, Wm.shape[0], 6, 6), P

def conv_bwd(dy, P, Wm, cin):   # dy: (N, oc, 6, 6)
    dyf = dy.reshape(dy.shape[0], dy.shape[1], 36).transpose(0, 2, 1)  # (N,36,oc)
    dW = np.einsum('npo,npk->ok', dyf, P) / dy.shape[0]
    db = dyf.sum(axis=(0, 1)) / dy.shape[0]
    dP = dyf @ Wm                                                       # (N,36,cin*9)
    # fold patches back (col2im)
    N = dy.shape[0]
    dxp = np.zeros((N, cin, 8, 8))
    dPr = dP.reshape(N, 6, 6, cin, 3, 3)
    for kh in range(3):
        for kw in range(3):
            dxp[:, :, kh:kh+6, kw:kw+6] += dPr[:, :, :, :, kh, kw].transpose(0,3,1,2)
    return dW, db, dxp[:, :, 1:7, 1:7]

def maxpool_fwd(x):  # (N, c, 6, 6) -> (N, c, 3, 3), argmax indices
    xw = x.reshape(-1, x.shape[1], 3, 2, 3, 2).transpose(0,1,2,4,3,5).reshape(-1, x.shape[1], 3, 3, 4)
    am = xw.argmax(axis=-1)
    return xw.max(axis=-1), am

def maxpool_bwd(dy, am, c):
    N = dy.shape[0]
    dxw = np.zeros((N, c, 3, 3, 4))
    np.put_along_axis(dxw, am[..., None], dy[..., None], axis=-1)
    return dxw.reshape(N, c, 3, 3, 2, 2).transpose(0,1,2,4,3,5).reshape(N, c, 6, 6)

# init
he = lambda *s: rng.normal(0, np.sqrt(2.0 / np.prod(s[1:])), s)
W1m = he(C, 1*9);  b1v = np.full(C, 0.25)
W2m = he(C, C*9);  b2v = np.full(C, 0.25)
W3m = he(C*9, D1) * 0.5; b3v = np.full(D1, 0.25)
W4m = he(D1, D1) * 0.5;  b4v = np.full(D1, 0.25)
W5m = he(D1, NC) * 0.5;  b5v = np.zeros(NC)

Xtr4 = Xtr[:, None, :, :]; Xte4 = Xte[:, None, :, :]

def forward(x4):
    z1, P1 = conv_fwd(x4, W1m, b1v); a1 = np.maximum(z1, 0)
    z2, P2 = conv_fwd(a1, W2m, b2v); a2 = np.maximum(z2, 0)
    p, am = maxpool_fwd(a2)
    f = p.reshape(len(x4), -1)
    z3 = f @ W3m + b3v; a3 = np.maximum(z3, 0)
    z4 = a3 @ W4m + b4v; a4 = np.maximum(z4, 0)
    z5 = a4 @ W5m + b5v
    return z1, P1, a1, z2, P2, a2, am, f, z3, a3, z4, a4, z5

def test_acc():
    zs = forward(Xte4)[-1]
    return (zs.argmax(1) == yte).mean()

def tie_penalty_grad(z2):
    """Pool-tie margin regularizer: for each 2x2 window of conv2 pre-acts,
    penalize the SECOND-largest deficit (MARGIN - z)+ — i.e. push toward
    '≤ 1 sub-margin pre-activation per window', which is exactly what the
    MaxPool2Smooth hypothesis needs after ReLU. Returns dz2 (N,C,6,6)."""
    N = z2.shape[0]
    z2w = z2.reshape(N, C, 3, 2, 3, 2).transpose(0,1,2,4,3,5).reshape(N, C, 3, 3, 4)
    v = np.maximum(TIE_MARGIN - z2w, 0)                    # deficits
    order = np.argsort(v, axis=-1)                          # ascending
    sec_idx = order[..., -2:-1]                             # 2nd-largest deficit
    sec_val = np.take_along_axis(v, sec_idx, axis=-1)
    g = np.zeros_like(z2w)
    np.put_along_axis(g, sec_idx, -(sec_val > 0).astype(float), axis=-1)  # d(v)/dz = -1
    g = g.reshape(N, C, 3, 3, 2, 2).transpose(0,1,2,4,3,5).reshape(N, C, 6, 6)
    return LAM_TIE * g / N

TIE_MARGIN, LAM_TIE = 0.05, 4.0
lr0, bs = 0.08, 128
for ep in range(18):
    lr = lr0 * (0.3 if ep >= 12 else 1.0)
    idx = rng.permutation(len(Xtr4))
    for bi in range(0, len(Xtr4), bs):
        sel = idx[bi:bi+bs]
        xb, yb = Xtr4[sel], ytr[sel]
        z1, P1, a1, z2, P2, a2, am, f, z3, a3, z4, a4, z5 = forward(xb)
        z = z5 - z5.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        g5 = p.copy(); g5[np.arange(len(yb)), yb] -= 1
        n = len(yb)
        dW5 = a4.T @ g5 / n; db5 = g5.mean(0)
        da4 = g5 @ W5m.T; dz4 = da4 * (z4 > 0)
        dW4 = a3.T @ dz4 / n; db4 = dz4.mean(0)
        da3 = dz4 @ W4m.T; dz3 = da3 * (z3 > 0)
        dW3 = f.T @ dz3 / n; db3 = dz3.mean(0)
        df = dz3 @ W3m.T
        dp = df.reshape(n, C, 3, 3)
        da2 = maxpool_bwd(dp, am, C); dz2 = da2 * (z2 > 0)
        dz2 = dz2 + tie_penalty_grad(z2) * n   # conv_bwd divides by n
        dW2, db2, da1 = conv_bwd(dz2, P2, W2m, C)
        dz1 = da1 * (z1 > 0)
        dW1, db1, _ = conv_bwd(dz1, P1, W1m, 1)
        for Wp_, g_ in ((W1m,dW1),(W2m,dW2),(W3m,dW3),(W4m,dW4),(W5m,dW5),
                        (b1v,db1),(b2v,db2),(b3v,db3),(b4v,db4),(b5v,db5)):
            Wp_ -= lr * g_
    print(f"ep {ep}: test acc {test_acc():.4f}", flush=True)

acc_float = test_acc()

# ---------------------------------------------------------------- rationalize
def ratz(a):
    return np.vectorize(lambda v: Fraction(int(round(v * DEN_W)), DEN_W))(a)

W1r, b1r = ratz(W1m), ratz(b1v)
W2r, b2r = ratz(W2m), ratz(b2v)
W3r, b3r = ratz(W3m), ratz(b3v)
W4r, b4r = ratz(W4m), ratz(b4v)
W5r, b5r = ratz(W5m), ratz(b5v)

# float twins of the rationalized weights for quantized accuracy
W1q, b1q = W1r.astype(float), b1r.astype(float)
W2q, b2q = W2r.astype(float), b2r.astype(float)
W3q, b3q = W3r.astype(float), b3r.astype(float)
W4q, b4q = W4r.astype(float), b4r.astype(float)
W5q, b5q = W5r.astype(float), b5r.astype(float)

def forward_q(x4):
    z1, _ = conv_fwd(x4, W1q, b1q); a1 = np.maximum(z1, 0)
    z2, _ = conv_fwd(a1, W2q, b2q); a2 = np.maximum(z2, 0)
    p, _ = maxpool_fwd(a2)
    f = p.reshape(len(x4), -1)
    a3 = np.maximum(f @ W3q + b3q, 0)
    a4 = np.maximum(a3 @ W4q + b4q, 0)
    return a4 @ W5q + b5q

acc_q = (forward_q(Xte4).argmax(1) == yte).mean()
print(f"float acc {acc_float:.4f}  quantized acc {acc_q:.4f}", flush=True)

# ---------------------------------------------------------------- exact forward + witness search
def conv_exact(x, Wr, br):
    """x: (cin,6,6) Fractions; Wr: (oc, cin*9); mirrors CNN.lean conv2d."""
    cin = x.shape[0]; oc = Wr.shape[0]
    out = np.empty((oc, 6, 6), dtype=object)
    for o in range(oc):
        for hi in range(6):
            for wi in range(6):
                s = br[o]
                for c_ in range(cin):
                    for kh in range(3):
                        for kw in range(3):
                            r, cc = hi + kh - 1, wi + kw - 1
                            if 0 <= r < 6 and 0 <= cc < 6:
                                s += Wr[o, c_*9 + kh*3 + kw] * x[c_, r, cc]
                out[o, hi, wi] = s
    return out

def relu_exact(t):
    return np.vectorize(lambda v: v if v > 0 else Fraction(0))(t)

def check_witness(sums):
    """sums: (6,6) ints. Returns (ok, detail, tables) with exact Fractions."""
    x = np.vectorize(lambda s: Fraction(int(s), DEN_X))(sums)[None, :, :]
    c1 = conv_exact(x, W1r, b1r)
    if any(v == 0 for v in c1.flat): return False, "conv1 zero", None
    z1 = relu_exact(c1)
    c2 = conv_exact(z1, W2r, b2r)
    if any(v == 0 for v in c2.flat): return False, "conv2 zero", None
    r2 = relu_exact(c2)
    for ci in range(C):
        for ho in range(3):
            for wo in range(3):
                w4 = [r2[ci, 2*ho+a, 2*wo+b] for a in range(2) for b in range(2)]
                if len(set(w4)) != 4: return False, f"pool tie ch{ci} ({ho},{wo})", None
    pool = np.empty((C, 3, 3), dtype=object)
    for ci in range(C):
        for ho in range(3):
            for wo in range(3):
                pool[ci, ho, wo] = max(r2[ci, 2*ho+a, 2*wo+b] for a in range(2) for b in range(2))
    f = pool.reshape(-1)  # C-order flatten, matches Tensor3.flatten
    d3 = np.array([sum(f[i] * W3r[i, k] for i in range(C*9)) + b3r[k] for k in range(D1)], dtype=object)
    if any(v == 0 for v in d3): return False, "dense3 zero", None
    r3 = np.array([v if v > 0 else Fraction(0) for v in d3], dtype=object)
    d4 = np.array([sum(r3[i] * W4r[i, k] for i in range(D1)) + b4r[k] for k in range(D1)], dtype=object)
    if any(v == 0 for v in d4): return False, "dense4 zero", None
    r4 = np.array([v if v > 0 else Fraction(0) for v in d4], dtype=object)
    z5 = np.array([sum(r4[i] * W5r[i, k] for i in range(D1)) + b5r[k] for k in range(NC)], dtype=object)
    return True, "", (x, c1, z1, c2, r2, pool, f, d3, r3, d4, z5)

def float_prefilter(i):
    """cheap float screen: returns (ok, reason, n_bad_windows)."""
    x4 = Xte4[i:i+1]
    z1, _ = conv_fwd(x4, W1q, b1q)
    if np.abs(z1).min() < 1e-9: return False, "z1~0", -1
    a1 = np.maximum(z1, 0)
    z2, _ = conv_fwd(a1, W2q, b2q)
    if np.abs(z2).min() < 1e-9: return False, "z2~0", -1
    a2 = np.maximum(z2, 0)[0]
    nbad = 0
    for ci in range(C):
        for ho in range(3):
            for wo in range(3):
                w4 = sorted(a2[ci, 2*ho:2*ho+2, 2*wo:2*wo+2].reshape(-1))
                if min(b - a for a, b in zip(w4, w4[1:])) < 1e-12: nbad += 1
    if nbad: return False, "pooltie", nbad
    p, _ = maxpool_fwd(np.maximum(z2, 0))
    f = p.reshape(1, -1)
    z3 = f @ W3q + b3q
    if np.abs(z3).min() < 1e-9: return False, "z3~0", -1
    z4 = np.maximum(z3, 0) @ W4q + b4q
    if np.abs(z4).min() < 1e-9: return False, "z4~0", -1
    return True, "", 0

witness = None
fails = {}
badw = []
for i in range(len(Ste)):
    ok0, why0, nb = float_prefilter(i)
    if not ok0:
        fails[why0] = fails.get(why0, 0) + 1
        if why0 == "pooltie": badw.append(nb)
        continue
    ok, why, tabs = check_witness(Ste[i])
    if not ok:
        fails[why.split(" ch")[0].split(" (")[0]] = fails.get(why.split(" ch")[0].split(" (")[0], 0) + 1
        continue
    pred = int(np.argmax([float(v) for v in tabs[-1]]))
    if pred == int(yte[i]):
        witness = (i, tabs, pred)
        break
    if witness is None:
        witness = (i, tabs, pred)  # fallback: any valid witness
print("fail histogram (first pass):", fails, flush=True)
if badw:
    import collections
    print("bad-window count histogram:", dict(sorted(collections.Counter(badw).items())), flush=True)
if witness is None:
    sys.exit("NO WITNESS FOUND")
wi_idx, (xF, c1F, z1F, c2F, r2F, poolF, fF, d3F, r3F, d4F, z5F), pred = witness
correct = pred == int(yte[wi_idx])
print(f"witness: test #{wi_idx} label {yte[wi_idx]} pred {pred} correct={correct}", flush=True)

# ---------------------------------------------------------------- Lean emission
def lit(fr: Fraction) -> str:
    fr = Fraction(fr)
    if fr.denominator == 1:
        return f"({fr.numerator} : ℝ)" if fr.numerator < 0 else f"({fr.numerator} : ℝ)"
    return f"(({fr.numerator} : ℝ)/{fr.denominator})"

def vec_lit(vals):
    return "![" + ", ".join(lit(v) for v in vals) + "]"

def t3_lit(t):  # (c, h, w) object array -> nested ![]
    return "![" + ",\n    ".join(
        "![" + ",\n      ".join(vec_lit(t[ci, hi, :]) for hi in range(t.shape[1])) + "]"
        for ci in range(t.shape[0])) + "]"

def k4_lit(Wr, oc, cin):  # (oc, cin*9) -> Kernel4 oc cin 3 3 nested ![]
    chunks = []
    for o in range(oc):
        rows = []
        for c_ in range(cin):
            k = "![" + ", ".join(vec_lit([Wr[o, c_*9 + kh*3 + kw] for kw in range(3)]) for kh in range(3)) + "]"
            rows.append(k)
        chunks.append("![" + ",\n     ".join(rows) + "]")
    return "![" + ",\n   ".join(chunks) + "]"

def mat_lit(Wr):  # (m, n) -> ![row0, ...] rows of n
    return "![" + ",\n    ".join(vec_lit(Wr[i, :]) for i in range(Wr.shape[0])) + "]"

z1T = z1F; r2T = r2F
sums = Ste[wi_idx]

# ---------------------------------------------------------------- Lean emission
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
            k = "![" + ", ".join(vec_lit([Wr[o, c_*9 + kh*3 + kw] for kw in range(3)]) for kh in range(3)) + "]"
            rows.append(k)
        chunks.append("![" + ",\n     ".join(rows) + "]")
    return "![" + ",\n   ".join(chunks) + "]"

def mat_lit(Wr):
    return "![" + ",\n    ".join(vec_lit(Wr[i, :]) for i in range(Wr.shape[0])) + "]"

def fm(k, n):  # Fin literal
    return f"(⟨{k}, by norm_num⟩ : Fin {n})"

def conv_rows(name, Wn, bn, xn, tab, hb):
    """per-row conv lemmas + aggregator for conv2d Wn bn xn = tab."""
    out = []
    for o in range(2):
        for hi in range(6):
            out.append(f"""set_option maxRecDepth 16384 in
set_option maxHeartbeats {hb} in
theorem {name}_r{o}{hi} : ∀ wi : Fin 6,
    conv2d {Wn} {bn} {xn} {fm(o,2)} {fm(hi,6)} wi
      = {tab} {fm(o,2)} {fm(hi,6)} wi := by
  intro wi
  fin_cases wi <;> (simp [conv2d, {Wn}, {bn}, {xn}, {tab}, Fin.sum_univ_succ]; try norm_num)
""")
    bullets = "\n".join(f"  · exact {name}_r{o}{hi} wi" for o in range(2) for hi in range(6))
    out.append(f"""theorem {name} : ∀ (o : Fin 2) (hi wi : Fin 6),
    conv2d {Wn} {bn} {xn} o hi wi = {tab} o hi wi := by
  intro o hi wi
  fin_cases o <;> fin_cases hi
{bullets}
""")
    return "\n".join(out)

def smooth_rows(hb):
    out = []
    for ci in range(2):
        for ho in range(3):
            out.append(f"""set_option maxRecDepth 16384 in
set_option maxHeartbeats {hb} in
theorem r2sm_c{ci}h{ho} : ∀ (wo : Fin 3) (ab ab' : Fin 2 × Fin 2), ab ≠ ab' →
    r2V {fm(ci,2)} (winRowInv {fm(ho,3)} ab.1) (winColInv wo ab.2)
      ≠ r2V {fm(ci,2)} (winRowInv {fm(ho,3)} ab'.1) (winColInv wo ab'.2) := by
  rintro wo ⟨a, b⟩ ⟨a', b'⟩ hne
  fin_cases wo <;> fin_cases a <;> fin_cases b <;> fin_cases a' <;> fin_cases b' <;>
    first
      | exact absurd rfl hne
      | (simp [r2V, winRowInv, winColInv]; try norm_num)
""")
    bullets = "\n".join(f"  · exact r2sm_c{ci}h{ho} wo ab ab' hne" for ci in range(2) for ho in range(3))
    out.append(f"""/-- **Every 2×2 window of relu(conv2) has pairwise-distinct values** —
    the `MaxPool2Smooth` hypothesis, discharged from the trained tables. -/
theorem r2_smooth : MaxPool2Smooth (c := 2) (h := 3) (w := 3) r2V := by
  intro ci ho wo ab ab' hne
  fin_cases ci <;> fin_cases ho
{bullets}
""")
    return "\n".join(out)


pool_bullets = "\n".join(
    f"""  · show maxPool2 (c := 2) (h := 3) (w := 3) r2V {fm(k//9,2)} {fm((k%9)//3,3)} {fm(k%3,3)}
        = {lit(fF[k])}
    simp [maxPool2, r2V, max_def]
    try norm_num"""
    for k in range(18))

hdr = f'''import LeanMlir.Proofs.Architectures.MnistCNN

/-! # Trained-weight whole-network VJP witness — CNN rung

The `TrainedMlpWitness` program extended to a CONVOLUTIONAL net (the 2026-07
audit's gap #3): the Chapter-3 `mnistCnnNoBn` conditional whole-net VJP
(`mnistCnnNoBn_has_vjp_at`) instantiated at TRAINED, /128-rationalized
weights and a REAL test input, with every smoothness hypothesis discharged
by exact in-kernel rational arithmetic — inherited from training, not
engineered:

* h1/h2 — all 72+72 conv pre-activations are nonzero (`conv1_eq`/`conv2_eq`
  value tables + `c1_ne`/`c2_ne`);
* h_mp — every 2×2 max-pool window of relu(conv2) has four pairwise-distinct
  values (`r2_smooth`). ReLU zeros collide, so this needs ≤ 1 negative conv2
  pre-activation per window — trained in via a pool-tie margin regularizer
  (the h_mp analogue of the scorecard's spectral cap: the training method
  decides whether the hypotheses hold);
* h3/h4 — the dense pre-activations are nonzero (`d3_ne`/`d4_ne`).

Net: 24×24-center-cropped MNIST, 4×4-average-pooled to 6×6 (exact pixel sums
/4080), conv 1→2 3×3 SAME → relu → conv 2→2 3×3 SAME → relu → maxpool 2×2 →
dense 18→8 → relu → dense 8→8 → relu → dense 8→10. Float test acc
{acc_float:.3f}, /128-quantized {acc_q:.3f}. Witness: test digit #{wi_idx}
(label {yte[wi_idx]}, {"correctly classified" if correct else "misclassified"}).
Generated by `scripts/trained_cnn_witness.py`; weights/input are DATA here. -/

namespace Proofs
namespace TrainedCnn

-- ════════════════════════════════════════════════════════════════
-- § Trained weights and the witness input (data)
-- ════════════════════════════════════════════════════════════════

/-- Test image #{wi_idx}, 4×4-pooled 6×6, exact pixel sums /4080. -/
noncomputable def T0 : Tensor3 1 6 6 :=
  {t3_lit(np.vectorize(lambda s: Fraction(int(s), DEN_X))(sums)[None, :, :])}

/-- The flattened witness input. -/
noncomputable def X : Vec (1 * (2*3) * (2*3)) := Tensor3.flatten T0

/-- conv1 kernel (1→2, 3×3), entries k/128. -/
noncomputable def W1 : Kernel4 2 1 3 3 :=
  {k4_lit(W1r, C, 1)}

noncomputable def b1 : Vec 2 := {vec_lit(b1r)}

/-- conv2 kernel (2→2, 3×3), entries k/128. -/
noncomputable def W2 : Kernel4 2 2 3 3 :=
  {k4_lit(W2r, C, C)}

noncomputable def b2 : Vec 2 := {vec_lit(b2r)}

/-- dense3 (18→8, input×output). -/
noncomputable def W3 : Mat (2*3*3) 8 :=
  {mat_lit(W3r)}

noncomputable def b3 : Vec 8 := {vec_lit(b3r)}

noncomputable def W4 : Mat 8 8 :=
  {mat_lit(W4r)}

noncomputable def b4 : Vec 8 := {vec_lit(b4r)}

noncomputable def W5 : Mat 8 10 :=
  {mat_lit(W5r)}

noncomputable def b5 : Vec 10 := {vec_lit(b5r)}

-- ════════════════════════════════════════════════════════════════
-- § Generic plumbing
-- ════════════════════════════════════════════════════════════════

/-- ReLU commutes with `Tensor3.flatten` (both are pointwise). -/
theorem relu_flatten {{c h w : Nat}} (T : Tensor3 c h w) :
    relu (c * h * w) (Tensor3.flatten T) =
      Tensor3.flatten (fun ci hi wi => if T ci hi wi > 0 then T ci hi wi else 0) := rfl

/-- A nowhere-zero tensor flattens to a nowhere-zero vector. -/
theorem flatten_ne_zero {{c h w : Nat}} {{T : Tensor3 c h w}}
    (hT : ∀ ci hi wi, T ci hi wi ≠ 0) (k : Fin (c * h * w)) :
    Tensor3.flatten T k ≠ 0 := hT _ _ _

-- ════════════════════════════════════════════════════════════════
-- § conv1: exact value table, nonzero, relu fold
-- ════════════════════════════════════════════════════════════════

/-- conv1 pre-activations at the witness, exact. -/
noncomputable def c1V : Fin 2 → Fin 6 → Fin 6 → ℝ :=
  {t3_lit(c1F)}

'''

mid1 = conv_rows("conv1_eq", "W1", "b1", "T0", "c1V", 8000000)

mid2 = f'''
set_option maxRecDepth 16384 in
theorem c1_ne : ∀ (o : Fin 2) (hi wi : Fin 6), c1V o hi wi ≠ 0 := by
  intro o hi wi
  fin_cases o <;> fin_cases hi <;> fin_cases wi <;> norm_num [c1V]

/-- relu(conv1) at the witness, exact. -/
noncomputable def z1V : Tensor3 2 6 6 :=
  {t3_lit(z1T)}

set_option maxRecDepth 16384 in
set_option maxHeartbeats 8000000 in
theorem z1_eq :
    (fun o hi wi => if c1V o hi wi > 0 then c1V o hi wi else 0) = z1V := by
  funext o hi wi
  fin_cases o <;> fin_cases hi <;> fin_cases wi <;>
    (simp [c1V, z1V]; try norm_num)

/-- First conv→relu block at the witness = the exact table. -/
theorem block1_eq :
    (relu (2 * (2*3) * (2*3)) ∘ flatConv (h := 2*3) (w := 2*3) W1 b1) X
      = Tensor3.flatten z1V := by
  show relu _ (flatConv W1 b1 (Tensor3.flatten T0)) = _
  rw [flatConv, Tensor3.unflatten_flatten, relu_flatten]
  congr 1
  funext o hi wi
  rw [conv1_eq]
  exact congrFun (congrFun (congrFun z1_eq o) hi) wi

-- ════════════════════════════════════════════════════════════════
-- § conv2: exact value table, nonzero, relu fold
-- ════════════════════════════════════════════════════════════════

/-- conv2 pre-activations at the witness, exact. -/
noncomputable def c2V : Fin 2 → Fin 6 → Fin 6 → ℝ :=
  {t3_lit(c2F)}

'''

mid3 = conv_rows("conv2_eq", "W2", "b2", "z1V", "c2V", 16000000)

mid4 = f'''
set_option maxRecDepth 16384 in
theorem c2_ne : ∀ (o : Fin 2) (hi wi : Fin 6), c2V o hi wi ≠ 0 := by
  intro o hi wi
  fin_cases o <;> fin_cases hi <;> fin_cases wi <;> norm_num [c2V]

/-- relu(conv2) at the witness (the max-pool input), exact. -/
noncomputable def r2V : Tensor3 2 6 6 :=
  {t3_lit(r2T)}

set_option maxRecDepth 16384 in
set_option maxHeartbeats 8000000 in
theorem r2_eq :
    (fun o hi wi => if c2V o hi wi > 0 then c2V o hi wi else 0) = r2V := by
  funext o hi wi
  fin_cases o <;> fin_cases hi <;> fin_cases wi <;>
    (simp [c2V, r2V]; try norm_num)

/-- Both conv→relu blocks fold to the exact max-pool input table. -/
theorem blockZ_eq :
    ((relu (2 * (2*3) * (2*3)) ∘ flatConv (h := 2*3) (w := 2*3) W2 b2)
      ∘ (relu (2 * (2*3) * (2*3)) ∘ flatConv (h := 2*3) (w := 2*3) W1 b1)) X
      = Tensor3.flatten r2V := by
  rw [Function.comp_apply, block1_eq]
  show relu _ (flatConv W2 b2 (Tensor3.flatten z1V)) = _
  rw [flatConv, Tensor3.unflatten_flatten, relu_flatten]
  congr 1
  funext o hi wi
  rw [conv2_eq]
  exact congrFun (congrFun (congrFun r2_eq o) hi) wi

-- ════════════════════════════════════════════════════════════════
-- § MaxPool: no ties at the witness (trained, not engineered)
-- ════════════════════════════════════════════════════════════════

'''

mid5 = smooth_rows(8000000)

mid6 = f'''
-- ════════════════════════════════════════════════════════════════
-- § Pooled vector and the dense head
-- ════════════════════════════════════════════════════════════════

/-- The pooled feature vector (flattened maxpool output), exact. -/
noncomputable def p2f : Vec (2 * 3 * 3) := {vec_lit(fF)}

set_option maxRecDepth 16384 in
set_option maxHeartbeats 16000000 in
theorem pooled_eq :
    maxPoolFlat 2 3 3 (Tensor3.flatten r2V) = p2f := by
  show Tensor3.flatten (maxPool2 (Tensor3.unflatten (Tensor3.flatten r2V))) = p2f
  rw [Tensor3.unflatten_flatten]
  funext i
  fin_cases i
{pool_bullets}

/-- dense3 pre-activations at the witness, exact. -/
noncomputable def d3V : Fin 8 → ℝ := {vec_lit(d3F)}

set_option maxRecDepth 16384 in
set_option maxHeartbeats 16000000 in
theorem d3_eq : ∀ k, dense W3 b3 p2f k = d3V k := by
  intro k
  fin_cases k <;> (simp [dense, W3, b3, p2f, d3V, Fin.sum_univ_succ]; try norm_num)

set_option maxRecDepth 16384 in
theorem d3_ne : ∀ k, dense W3 b3 p2f k ≠ 0 := by
  intro k
  rw [d3_eq]
  fin_cases k <;> norm_num [d3V]

/-- relu(dense3) at the witness, exact. -/
noncomputable def r3V : Vec 8 := {vec_lit(r3F)}

set_option maxRecDepth 16384 in
set_option maxHeartbeats 8000000 in
theorem r3_eq : relu 8 (dense W3 b3 p2f) = r3V := by
  funext k
  show (if dense W3 b3 p2f k > 0 then dense W3 b3 p2f k else 0) = r3V k
  rw [d3_eq]
  fin_cases k <;> (simp [d3V, r3V]; try norm_num)

/-- dense4 pre-activations at the witness, exact. -/
noncomputable def d4V : Fin 8 → ℝ := {vec_lit(d4F)}

set_option maxRecDepth 16384 in
set_option maxHeartbeats 8000000 in
theorem d4_eq : ∀ k, dense W4 b4 r3V k = d4V k := by
  intro k
  fin_cases k <;> (simp [dense, W4, b4, r3V, d4V, Fin.sum_univ_succ]; try norm_num)

set_option maxRecDepth 16384 in
theorem d4_ne : ∀ k, dense W4 b4 r3V k ≠ 0 := by
  intro k
  rw [d4_eq]
  fin_cases k <;> norm_num [d4V]

-- ════════════════════════════════════════════════════════════════
-- § The capstone: trained-weight whole-net CNN VJP witness
-- ════════════════════════════════════════════════════════════════

/-- **Level 1: the trained-weight whole-net CNN VJP witness** —
    `HasVJPAt (mnistCnnNoBnForward …) X` with every one of the five
    smoothness hypotheses discharged at the trained weights and the real
    test input. The convolutional sibling of `trainedMlp_has_vjp_at`. -/
noncomputable def trainedCnn_has_vjp_at :
    HasVJPAt (mnistCnnNoBnForward W1 b1 W2 b2 W3 b3 W4 b4 W5 b5) X :=
  mnistCnnNoBn_has_vjp_at W1 b1 W2 b2 W3 b3 W4 b4 W5 b5
    (by norm_num) (by norm_num) (by norm_num) X
    -- h1: conv1 pre-activations nonzero
    (by intro k
        have he : flatConv (h := 2*3) (w := 2*3) W1 b1 X
            = Tensor3.flatten (conv2d W1 b1 T0) := by
          show flatConv W1 b1 (Tensor3.flatten T0) = _
          rw [flatConv, Tensor3.unflatten_flatten]
        rw [he]
        exact flatten_ne_zero
          (fun o hi wi => by rw [conv1_eq]; exact c1_ne o hi wi) k)
    -- h2: conv2 pre-activations nonzero
    (by intro k
        rw [block1_eq]
        have he : flatConv (h := 2*3) (w := 2*3) W2 b2 (Tensor3.flatten z1V)
            = Tensor3.flatten (conv2d W2 b2 z1V) := by
          rw [flatConv, Tensor3.unflatten_flatten]
        rw [he]
        exact flatten_ne_zero
          (fun o hi wi => by rw [conv2_eq]; exact c2_ne o hi wi) k)
    -- h_mp: no max-pool ties
    (by rw [blockZ_eq, Tensor3.unflatten_flatten]
        exact r2_smooth)
    -- h3: dense3 pre-activations nonzero
    (by intro k
        rw [blockZ_eq, pooled_eq]
        exact d3_ne k)
    -- h4: dense4 pre-activations nonzero
    (by intro k
        rw [blockZ_eq, pooled_eq, Function.comp_apply, r3_eq]
        exact d4_ne k)

/-- The witness's contract, exposed: the whole-net backward equals the
    `pdiv`-contracted Jacobian at the trained weights and real input. -/
theorem trainedCnn_has_vjp_correct (dy : Vec 10) (i : Fin (1 * (2*3) * (2*3))) :
    trainedCnn_has_vjp_at.backward dy i =
      ∑ j : Fin 10,
        pdiv (mnistCnnNoBnForward W1 b1 W2 b2 W3 b3 W4 b4 W5 b5) X i j * dy j :=
  trainedCnn_has_vjp_at.correct dy i

end TrainedCnn
end Proofs
'''

with open(OUT, "w") as f:
    f.write(hdr + mid1 + mid2 + mid3 + mid4 + mid5 + mid6)
print(f"wrote {OUT}", flush=True)
