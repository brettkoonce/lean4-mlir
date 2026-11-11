"""Trained-weight CNN descent, all four conv rungs: concrete satisfiable instances on real MNIST.

Emits two files, every hypothesis discharged by exact in-kernel integer arithmetic at TRAINED,
/128-rationalized weights (biases /16384) and a REAL 28x28 MNIST test image, all 784 pixels:

- LeanMlir/Proofs/Training/Trained/CnnDescent.lean, net A (conv1 WITHOUT bias):
  `cnn_conv2_exact_sgd_descends` and `cnn_conv2_bias_exact_sgd_descends` (one exact-gradient SGD
  step on the conv2 kernel / bias decreases the cross-entropy loss). The point is the pool
  hypothesis `MaxPool2MarginQUpTo` with twins `ConvPatchEq`: the image's blank border gives live
  2x2 windows whose conv2 outputs are EQUAL (they read identical all-zero input patches), the case
  the old `MaxPool2MarginQ` could not state and real MNIST forces
  (scripts/probes/mnist_pool_twin_probe.py). conv1 is bias-free so that a zero input patch gives
  x1 = 0, the same value as the zero padding: then the cells of a blank region read identical
  patches, and their conv2 outputs tie at b2.
- LeanMlir/Proofs/Training/Trained/CnnDescentConv1.lean, net B (conv1 WITH a trained bias):
  `cnn_conv1_exact_sgd_descends` and `cnn_conv1_bias_exact_sgd_descends`. Net A cannot serve the
  conv1 rungs: their relu1 margin needs every conv1 pre-activation nonzero, and a bias-free conv1
  is exactly 0 on a blank patch. The twins are two-layer (`ConvPatchEq2`): two cells whose 5x5
  two-conv receptive fields lie in a blank region and meet the zero padding the same way.

No pool-tie regularizer is used: the ties are the data's. The net is the CNN chapter's
`mnistCnnNoBn` architecture at 2 channels and an 8-wide head: the 1x28x28 image (pixels k/255),
conv1 1->2 3x3 SAME, relu, conv2 2->2 3x3 SAME + bias, relu, maxpool 2x2 -> 2x14x14, dense
392->8, relu, dense 8->8, relu, dense 8->10; both nets trained by the same seed-0 numpy SGD.

Data conventions (LeanMlir/Proofs/Architectures/ListConv.lean). Every table is nested `List ℤ`
numerators over one denominator, and each whole-table fact is ONE kernel evaluation
(`decide +kernel`) of a `Bool` this script evaluates first with the same integer formula:

    pixels            k/255            x0Z   [1][28][28]
    conv kernels      k/128            W1Z, W2Z   [out][in][3][3]
    conv1 preacts     k/D1             c1Z   [2][28][28];  D1 = 255*128 (net A), 255*16384 (net B)
    conv2 preacts     k/D2             c2Z   [2][28][28];  D2 = D1*128
    pooled features   k/D2             pZ    392, row-major
    dense3 columns    k/128            w3c{k}Z  392 each (`Mat` is input x output)

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
from _gencheck import modularize, module_header  # noqa: E402  (module-system header on what we write)

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
from _mnist_io import mnist  # noqa: E402
from _leanlit import zlist, zlit  # noqa: E402

OUT = os.path.join(ROOT, "LeanMlir/Proofs/Training/Trained/CnnDescent.lean")
DEN_W = 128          # weight rationalization grid
DEN_B = 16384        # bias grid: a trained conv2 bias sits below 1/256, and at /128 it rounds to 0
DEN_X = 255          # pixel denominator
S = 28               # input side
C = 2                # conv channels
D1 = 8               # dense width
NC = 10
H = S // 2           # pooled side
NF = C * H * H       # pooled features

Xtr_raw, ytr = mnist("train")
Xte_raw, yte = mnist("test")
Ste = Xte_raw.astype(np.int64)                 # (N, 28, 28) pixel integers 0..255
Xtr4 = (Xtr_raw / float(DEN_X))[:, None]
Xte4 = (Xte_raw / float(DEN_X))[:, None]

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


def test_acc(net):
    """Test accuracy, in chunks (one im2col of the whole test set is a few GB)."""
    hits = sum(int((forward(net, Xte4[b:b + 1000])[-1].argmax(1) == yte[b:b + 1000]).sum())
               for b in range(0, len(Xte4), 1000))
    return hits / len(Xte4)


def train(conv1_bias, tag):
    """Seed-0 numpy SGD, 12 epochs, batch 128. conv1_bias=False: conv1 has no bias (net A, the
    conv2 rungs); True: conv1 has a trained bias initialised at 0.1 (net B, the conv1 rungs)."""
    rng = np.random.default_rng(0)
    he = lambda *s: rng.normal(0, np.sqrt(2.0 / np.prod(s[1:])), s)
    net = {}
    net["W1"] = he(C, 9)
    net["W2"] = he(C, C * 9); net["b2"] = np.full(C, 0.1)
    net["W3"] = he(NF, D1) * 0.5; net["b3"] = np.full(D1, 0.1)
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
        print(f"[{tag}] ep {ep}: test acc {test_acc(net):.4f}", flush=True)
    return net


def ratz(a, den=DEN_W):
    return np.vectorize(lambda v: Fraction(int(round(v * den)), den))(a)


def quantize(net, tag):
    """/128 weights, /16384 biases; prints the float and the quantized test accuracy."""
    q = {k: ratz(net[k], DEN_B if k.startswith("b") else DEN_W) for k in NAMES}
    fl = {k: q[k].astype(float) for k in NAMES}
    print(f"[{tag}] float acc {test_acc(net):.4f}  quantized acc {test_acc(fl):.4f}", flush=True)
    return q


def numer(q, den):
    """integer numerators of an array of Fractions over `den`."""
    out = np.vectorize(lambda v: int(v * den))(q)
    assert all(Fraction(int(n), den) == v for n, v in zip(out.flat, q.flat))
    return out


# ---------------------------------------------------------------- exact forward
ZF = Fraction(0)
CELLS = [(0, 0), (0, 1), (1, 0), (1, 1)]   # (a, b) in the order `allN 2 (allN 2 ·)` reads them


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


def conv_int(xN, Wq, bN, mc):
    """`convCellZ` on every cell: xN (cin,S,S) int64 numerators, Wq (oc, cin*9) weight numerators,
    bN the bias numerators over the output denominator. Vectorized; the chosen image is rechecked
    against `conv_exact`."""
    xp = np.pad(xN, ((0, 0), (1, 1), (1, 1)))
    cin, oc = xN.shape[0], Wq.shape[0]
    out = np.empty((oc, S, S), dtype=np.int64)
    for o in range(oc):
        acc = np.zeros((S, S), dtype=np.int64)
        for c_ in range(cin):
            for kh in range(3):
                for kw in range(3):
                    acc += int(Wq[o, c_ * 9 + kh * 3 + kw]) * xp[c_, kh:kh + S, kw:kw + S]
        out[o] = int(bN[o]) + mc * acc
    return out


def zero_patch(xN):
    """`zeroPatchZ`: cell (r, s)'s zero-padded 3x3 patch is zero in every channel."""
    nz = np.pad((xN != 0).any(axis=0), 1)
    return np.array([[not nz[r:r + 3, s:s + 3].any() for s in range(S)] for r in range(S)])


def zero_twin(zp):
    """`zeroTwinZ`: both cells have all-zero patches. They are `ConvPatchEq` twins."""
    return lambda m, p: bool(zp[m] and zp[p])


def blank_twin(zp):
    """`blankTwinZ`: the two cells' two-conv receptive fields are blank and of the same shape. At
    every outer offset both reads fall in the padding, or both land on zero-patch cells. Interior
    cells of a blank region are such pairs, and so are two cells on the same edge of the image.
    They are `ConvPatchEq2` twins."""
    def tw(m, p):
        for kh in range(3):
            for kw in range(3):
                a, b = (m[0] + kh - 1, m[1] + kw - 1), (p[0] + kh - 1, p[1] + kw - 1)
                in_a = 0 <= a[0] < S and 0 <= a[1] < S
                in_b = 0 <= b[0] < S and 0 <= b[1] < S
                if in_a != in_b or (in_a and not (zp[a] and zp[b])):
                    return False
        return True
    return tw


def win_cert(x, tw, n, ho, wo):
    """`winCertZ` on one channel `x` (S×S ints): the window is dead, or some cell m has every
    cell at m's position, more than n below it, or paired with m by `tw`."""
    pos = [(2 * ho + a, 2 * wo + b) for a, b in CELLS]
    if all(x[p] <= 0 for p in pos):
        return True
    return any(all(p == m or x[p] + n < x[m] or tw(m, p) for p in pos) for m in pos)


def bias_over(bq, den):
    """bias numerators (over DEN_B) re-expressed over the table denominator `den`."""
    assert all((int(b) * den) % DEN_B == 0 for b in bq)
    return [int(b) * den // DEN_B for b in bq]


def ceil_frac(v):
    v = Fraction(v)
    return -((-v.numerator) // v.denominator)


def frac_up(v, den=8):
    """smallest k/den >= v."""
    return Fraction(ceil_frac(v * den), den)


def analyse(i, q, den1, two_layer):
    """Integer tables + window classification of net `q` at test image i, or None. `den1` is the
    conv1 output denominator; the twins are one-conv (`zero_twin` of relu(conv1)) or, with
    `two_layer`, two-conv (`blank_twin` of the image)."""
    x0 = Ste[i][None, :, :]
    mc1 = den1 // (DEN_X * DEN_W)
    den2 = den1 * DEN_W
    c1 = conv_int(x0, q["W1q"], bias_over(q["b1q"], den1), mc1)
    if two_layer and (c1 == 0).any():
        return None
    x1 = np.maximum(c1, 0)
    c2 = conv_int(x1, q["W2q"], bias_over(q["b2q"], den2), 1)
    if (c2 == 0).any():
        return None
    tw = blank_twin(zero_patch(x0)) if two_layer else zero_twin(zero_patch(x1))
    live_twin, n_dead, gaps = 0, 0, []
    for ci in range(C):
        for ho in range(H):
            for wo in range(H):
                pos = [(2 * ho + a, 2 * wo + b) for a, b in CELLS]
                vals = [int(c2[ci][p]) for p in pos]
                if all(v <= 0 for v in vals):
                    n_dead += 1
                    continue
                m = max(range(4), key=lambda k: vals[k])
                twin = False
                for k in range(4):
                    if k == m:
                        continue
                    if tw(pos[m], pos[k]):
                        twin = True
                    elif vals[k] < vals[m]:
                        gaps.append(vals[m] - vals[k])
                    else:
                        return None
                live_twin += twin
    p = np.maximum(c2, 0).reshape(C, H, 2, H, 2).max(axis=(2, 4)).reshape(-1)
    f = [Fraction(int(v), den2) for v in p]
    d3 = [sum(f[j] * q["W3"][j, k] for j in range(NF)) + q["b3"][k] for k in range(D1)]
    if any(v == 0 for v in d3):
        return None
    r3 = [v if v > 0 else ZF for v in d3]
    d4 = [sum(r3[j] * q["W4"][j, k] for j in range(D1)) + q["b4"][k] for k in range(D1)]
    if any(v == 0 for v in d4):
        return None
    r4 = [v if v > 0 else ZF for v in d4]
    z5 = [sum(r4[j] * q["W5"][j, k] for j in range(D1)) + q["b5"][k] for k in range(NC)]
    return dict(x0=x0, c1=c1, x1=x1, c2=c2, p=p, f=f, d3=d3, r3=r3, d4=d4, r4=r4, z5=z5, tw=tw,
                live_twin=live_twin, n_dead=n_dead, mingap=min(gaps) if gaps else None,
                den1=den1, den2=den2, mc1=mc1)


def prepare(q):
    """add the integer numerators the tables are built from."""
    q = dict(q)
    for k in ("W1", "W2", "W3"):
        q[k + "q"] = numer(q[k], DEN_W)
    for k in ("b1", "b2"):
        q[k + "q"] = numer(q[k], DEN_B)
    return q


def choose(q, den1, two_layer, what):
    for i in range(len(Ste)):
        t = analyse(i, q, den1, two_layer)
        if t is None or t["live_twin"] == 0 or t["mingap"] is None:
            continue
        if int(np.argmax([float(v) for v in t["z5"]])) != int(yte[i]):
            continue
        return i, t
    sys.exit(f"NO {what} INSTANCE FOUND")


def recheck(t, q):
    """The chosen image again, in exact `Fraction`s through `conv_exact`: the integer tables are
    the rational forward's numerators, and the pooled vector is the max-pool of relu(conv2)."""
    x0 = np.vectorize(lambda s: Fraction(int(s), DEN_X))(t["x0"])
    c1 = conv_exact(x0, q["W1"], q["b1"])
    assert all(Fraction(int(n), t["den1"]) == v for n, v in zip(t["c1"].flat, c1.flat))
    x1 = np.vectorize(lambda v: v if v > 0 else ZF)(c1)
    assert all(Fraction(int(n), t["den1"]) == v for n, v in zip(t["x1"].flat, x1.flat))
    c2 = conv_exact(x1, q["W2"], q["b2"])
    assert all(Fraction(int(n), t["den2"]) == v for n, v in zip(t["c2"].flat, c2.flat))
    for ci in range(C):
        for ho in range(H):
            for wo in range(H):
                best = max(max(c2[ci, 2 * ho + a, 2 * wo + b], ZF) for a, b in CELLS)
                assert t["f"][wo + H * (ho + H * ci)] == best
    return c1, x1, c2


QA = prepare(quantize(train(False, "A"), "A"))
assert all(v == 0 for v in QA["b1"])
IDX, T = choose(QA, DEN_X * DEN_W, False, "CONV2")
PRED = LBL = int(yte[IDX])
print(f"instance: test #{IDX} label {LBL}, live twin windows {T['live_twin']}, "
      f"dead {T['n_dead']}", flush=True)
c1F, x1F, c2F = recheck(T, QA)
W3r, b3r, W4r, b4r, W5r, b5r = QA["W3"], QA["b3"], QA["W4"], QA["b4"], QA["W5"], QA["b5"]
DA1, DA2 = T["den1"], T["den2"]

# ---------------------------------------------------------------- constants and the learning rate
A = frac_up(max(x1F.flat))
W3B = frac_up(max(abs(v) for v in W3r.flat), DEN_W)
W4B = frac_up(max(abs(v) for v in W4r.flat), DEN_W)
W5B = frac_up(max(abs(v) for v in W5r.flat), DEN_W)
P = C * C * 9
K2 = (2 * H) * (2 * H)
G = P * (K2 * A * (D1 * (W3B * (D1 * (W4B * (NC * (W5B * 1)))))))
minz2N = int(abs(T["c2"]).min())
minz3 = min(abs(v) for v in T["d3"])
minz4 = min(abs(v) for v in T["d4"])


def ok(lr):
    R = lr * G
    aR = A * R
    # the relu₂ and pool margins, as the integer checks the kernel evaluates
    if not ceil_frac(aR * DA2) < minz2N:
        return False
    if not ceil_frac(2 * aR * DA2) < T["mingap"]:
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


KEXP = next(k for k in range(1, 300) if ok(Fraction(1, 2 ** k)))
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


KEXPB = next(k for k in range(1, 300) if ok_bias2(Fraction(1, 2 ** k)))
RB2 = Fraction(1, 2 ** KEXPB) * GB2
print(f"conv2 bias rung: lr = 2^-{KEXPB}", flush=True)

# ---------------------------------------------------------------- Lean emission
R = LR * G
AR = A * R            # the pool / relu₂ margin radius a·(lr·G), exact


def lit(fr):
    fr = Fraction(fr)
    if fr.denominator == 1:
        return f"({fr.numerator} : ℝ)"
    return f"(({fr.numerator} : ℝ)/{fr.denominator})"


def vec_lit(vals):
    return "![" + ", ".join(lit(v) for v in vals) + "]"


def mat_lit(Wr):
    return "![" + ",\n    ".join(vec_lit(Wr[i, :]) for i in range(Wr.shape[0])) + "]"


def z3_lit(t):
    """a (c, h, w) integer table as nested `List ℤ`, one row per line."""
    return "[" + ",\n   ".join(
        "[" + ",\n    ".join(zlist(int(v) for v in t[ci][hi]) for hi in range(len(t[ci]))) + "]"
        for ci in range(len(t))) + "]"


def k4_lit(Wq, oc, cin):
    """a conv kernel's numerators as `[out][in][3][3]` nested `List ℤ`."""
    return "[" + ",\n   ".join(
        "[" + ", ".join(
            "[" + ", ".join(zlist(int(Wq[o, c_ * 9 + kh * 3 + kw]) for kw in range(3))
                            for kh in range(3)) + "]"
            for c_ in range(cin)) + "]"
        for o in range(oc)) + "]"


def weight_bound(name, mat, bound):
    """`|W i j| ≤ bound` for every entry, peeling the `![…]` literal row by row
    (`Fin.forall_fin_succ`): indexing it at a numeral far down the vector times out."""
    return f"""theorem {name} : ∀ i j, |{mat} i j| ≤ {lit(bound)} := by
  simp only [{mat}, Fin.forall_fin_succ, Matrix.cons_val_zero, Matrix.cons_val_succ,
    IsEmpty.forall_iff, and_true, abs_le]
  norm_num
"""


def bias_nums(name, nums, doc):
    return f"""/-- {doc} -/
def {name} : ℕ → ℤ := fun o => {zlist(nums)}.getD o 0
"""


def dense3_text(t, q, den2):
    """dense3 on the pooled list: the eight 392-term pre-activations, one kernel dot each."""
    w3 = [[int(q["W3q"][j, k]) for j in range(NF)] for k in range(D1)]
    pN = [int(v) for v in t["p"]]
    n3 = max(abs(v) for col in w3 for v in col)
    out = []
    for k in range(D1):
        out.append(f"/-- dense3's column {k}: unit {k}'s weights over the 392 pooled features, "
                   f"numerators over 128. -/\ndef w3c{k}Z : List ℤ := {zlist(w3[k])}\n")
    out.append("/-- The eight dense3 weight columns by unit. -/\ndef w3Z : Fin 8 → List ℤ\n"
               + "\n".join(f"  | {k} => w3c{k}Z" for k in range(D1)) + "\n")
    out.append("theorem w3Z_len : ∀ k, (w3Z k).length = 392 := by decide +kernel\n")
    out.append("/-- dense3 (392→8, input×output), entries k/128. -/\n"
               "noncomputable def W3 : Mat (2*14*14) 8 := fun i k => ((w3Z k).getD i 0 : ℝ) / 128\n")
    dots = []
    for k in range(D1):
        d = sum(a * b for a, b in zip(pN, w3[k], strict=True))
        assert Fraction(d, den2 * DEN_W) + q["b3"][k] == t["d3"][k]
        dots.append(f"theorem d3c{k} : dotZ pZ w3c{k}Z = {zlit(d)} := by decide +kernel")
    eqs = []
    for k in range(D1):
        eqs.append(f"""theorem d3_{k} : dense W3 b3 p2f {k} = {lit(t["d3"][k])} := by
  show (∑ i : Fin 392, (pZ.getD i 0 : ℝ) / {den2} * ((w3c{k}Z.getD i 0 : ℝ) / 128)) + b3 {k} = _
  rw [sum_getD_div (ys := w3c{k}Z) pZ_len (w3Z_len {k}) d3c{k} {den2} 128]
  norm_num [b3]
""")
    assert Fraction(n3, DEN_W) == frac_up(max(abs(v) for v in q["W3"].flat), DEN_W)
    bound = f"""theorem w3Z_range : ∀ k, ∀ z ∈ w3Z k, {zlit(-n3)} ≤ z ∧ z ≤ {n3} := by decide +kernel

theorem hW3 : ∀ i j, |W3 i j| ≤ {lit(Fraction(n3, DEN_W))} := fun i j =>
  abs_getD_div_le (w3Z j) {n3} (by norm_num) (w3Z_range j) (by norm_num) (by norm_num) i
"""
    return "\n".join(out), "\n".join(dots), "\n".join(eqs), bound


def head_text(t, q):
    """the dense head past dense3's dots: d3V, relu, d4V."""
    return f"""/-- dense3 pre-activations at the image, exact. -/
noncomputable def d3V : Fin 8 → ℝ := {vec_lit(t["d3"])}

theorem d3_eq : ∀ k, dense W3 b3 p2f k = d3V k := by
  intro k
  fin_cases k
  exacts [{", ".join(f"d3_{k}" for k in range(D1))}]

/-- relu(dense3) at the image, exact. -/
noncomputable def r3V : Vec 8 := {vec_lit(t["r3"])}

theorem r3_eq : relu 8 (dense W3 b3 p2f) = r3V := by
  funext k
  show (if dense W3 b3 p2f k > 0 then dense W3 b3 p2f k else 0) = r3V k
  rw [d3_eq]
  fin_cases k <;> (simp [d3V, r3V])

/-- dense4 pre-activations at the image, exact. -/
noncomputable def d4V : Fin 8 → ℝ := {vec_lit(t["d4"])}

theorem d4_eq : ∀ k, dense W4 b4 r3V k = d4V k := by
  intro k
  fin_cases k <;> (simp [dense, W4, b4, r3V, d4V, Fin.sum_univ_succ]; try norm_num)
"""


# net A's integer certificates, each evaluated here by the formula the kernel evaluates
b2NA = bias_over(QA["b2q"], DA2)
assert all(Fraction(n, DA2) == v for n, v in zip(b2NA, QA["b2"]))
NX1 = int(T["x1"].max())                       # |x1| ≤ NX1/D1 ≤ a
assert Fraction(NX1, DA1) <= A
NM2 = ceil_frac(AR * DA2)                      # a·lr·G ≤ NM2/D2 < |c2|
assert AR <= Fraction(NM2, DA2) and NM2 < minz2N
NPOOL = ceil_frac(2 * AR * DA2)                # 2·(a·lr·G) ≤ NPOOL/D2 < every gap
assert 2 * AR <= Fraction(NPOOL, DA2)
assert all(win_cert(T["c2"][ci], T["tw"], NPOOL, ho, wo)
           for ci in range(C) for ho in range(H) for wo in range(H))
D3_DEFS, D3_DOTS, D3_EQS, W3_BOUND = dense3_text(T, QA, DA2)

DESCENT_HEADER = module_header(["LeanMlir.Proofs.Training.SgdDescent.Cnn",
                                "LeanMlir.Proofs.Architectures.ListConv"],
                               ["Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp"])

body = f'''{DESCENT_HEADER}

/-! # Descent at TRAINED weights — the CNN conv2 rungs, through tied pool windows

The net is the CNN chapter's `mnistCnnNoBn` architecture at 2 channels and an 8-wide head, on the
28×28 image: `conv 1→2 3×3 SAME → relu → conv 2→2 3×3 SAME → relu → maxpool 2×2 → dense 392→8 →
relu → dense 8→8 → relu → dense 8→10`. Nothing is cropped or pooled away; only the widths differ
from the chapter's 32 channels and 512-wide head. The canonical net's pool condition is measured
on all 10000 test images by scripts/probes/mnist_pool_twin_probe.py.

`cnn_conv2_exact_sgd_descends` — one exact-gradient SGD step on the conv2 kernel decreases the
cross-entropy loss — instantiated at TRAINED, /128-rationalized weights (biases /16384) and REAL
MNIST test image #{IDX} (label {LBL}, classified correctly), all 784 pixels `k/255`, every
hypothesis discharged by exact arithmetic: `trained_cnn_conv2_sgd_descends_concrete`, at learning
rate `2⁻{KEXP}`. The conv2-bias rung `cnn_conv2_bias_exact_sgd_descends` on the same net and image
is `trained_cnn_conv2_bias_sgd_descends_concrete`, at `2⁻{KEXPB}`: its radius sits inside the
kernel rung's, so the relu₂ and pool margins carry over by monotonicity. The conv1 rungs need a
net with a conv1 bias (`Trained.CnnDescentConv1`).

What the instance shows is the pool hypothesis. The image's blank border makes conv1's output
zero there (conv1 is bias-free, so a zero patch gives the padding's value), the cells of a
border window then read identical all-zero patches, and their conv2 outputs are EQUAL — the
conv2 bias, positive in one channel. {T["live_twin"]} live windows tie that way and
{T["n_dead"]} are dead, of 392. The old `MaxPool2MarginQ` fails at every such window for every
margin; `MaxPool2MarginQUpTo` takes the twins `ConvPatchEq 3 3 x1V` (two cells whose patches are
all zero, `convPatchEq_of_zeroTwinZ`), and every other cell clears the margin (`cert_chk`, through
`maxPool2MarginQUpTo_of_poolCertZ`). No pool-tie regularizer was used in training; the ties are
the data's.

Every table is nested `List ℤ` numerators over one denominator, and each whole-table fact
(`conv1_chk`, `conv2_chk`, `pool_chk`, `cert_chk`, the margins and bounds) is one kernel
evaluation, transported to `ℝ` by `Architectures/ListConv`; the 392-term dense3 sums are
`Foundation/ListDot` dots. The rungs move the conv2 kernel or bias; `x1V` = relu(conv1 image) is
their frozen input (`x1_eq`). The generator prints the net's test accuracy.

The step is the exact gradient, so the conclusion bounds the true loss, but the decrease
`lr·‖∇L‖₂²/2` is not shown positive: that needs a lower bound on the gradient, which the
softmax makes transcendental (`Trained.LinearDescent` gets one from a misclassified example).
Generated by `scripts/certs/trained_cnn_descent.py`; weights and input are DATA. -/

namespace Proofs
namespace TrainedCnnDescent

-- ════════════════════════════════════════════════════════════════
-- § Trained weights and the input (data)
-- ════════════════════════════════════════════════════════════════

/-- MNIST test image #{IDX} (digit {LBL}), the 28×28 pixels as integers `0..255`. -/
def x0Z : List (List (List ℤ)) :=
  {z3_lit(T["x0"])}

/-- Test image #{IDX}, pixels `k/255`. -/
noncomputable def T0 : Tensor3 1 (2*14) (2*14) := tabR (g3 x0Z) 255

/-- conv1 kernel numerators (1→2, 3×3) over 128. -/
def W1Z : List (List (List (List ℤ))) :=
  {k4_lit(QA["W1q"], C, 1)}

/-- conv1 kernel (1→2, 3×3), entries k/128; conv1 has no bias. -/
noncomputable def W1 : Kernel4 2 1 3 3 := kerR (g4 W1Z) 128

noncomputable def b1 : Vec 2 := ![(0 : ℝ), (0 : ℝ)]

/-- conv2 kernel numerators (2→2, 3×3) over 128. -/
def W2Z : List (List (List (List ℤ))) :=
  {k4_lit(QA["W2q"], C, C)}

/-- conv2 kernel (2→2, 3×3), entries k/128. -/
noncomputable def W2 : Kernel4 2 2 3 3 := kerR (g4 W2Z) 128

noncomputable def b2 : Vec 2 := {vec_lit(QA["b2"])}

{D3_DEFS}
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

/-- conv1 pre-activation numerators at the image, over {DA1} (= 255·128). -/
def c1Z : List (List (List ℤ)) :=
  {z3_lit(T["c1"])}

/-- conv1 pre-activations at the image, exact. -/
noncomputable def c1V : Tensor3 2 (2*14) (2*14) := tabR (g3 c1Z) {DA1}

{bias_nums("b1N", [0, 0], f"conv1's bias numerators over {DA1}: none.")}
theorem conv1_chk : convTabZ 2 1 3 3 28 28 b1N 1 (g4 W1Z) (g3 x0Z) (g3 c1Z) = true := by
  decide +kernel

theorem conv1_fun : conv2d W1 b1 T0 = c1V :=
  conv2d_tab (fun o => by fin_cases o <;> simp [b1, b1N]) (by norm_num) conv1_chk

/-- relu(conv1) at the image: the conv2 rung's input, exact. -/
noncomputable def x1V : Tensor3 2 (2*14) (2*14) := tabR (reluZ (g3 c1Z)) {DA1}

/-- **The rung's input is the real image's conv1 activation.** -/
theorem x1_eq : x1V = fun o hi wi =>
    if conv2d W1 b1 T0 o hi wi > 0 then conv2d W1 b1 T0 o hi wi else 0 := by
  rw [conv1_fun]
  exact (relu_ite_tabR (g3 c1Z) (by norm_num)).symm

theorem x1_chk : absLeZ 2 28 28 (reluZ (g3 c1Z)) {NX1} = true := by decide +kernel

theorem x1_bound : ∀ cc i j, |x1V cc i j| ≤ {lit(A)} := fun cc i j =>
  tabR_abs_le x1_chk (by norm_num) (by norm_num) cc i j

-- ════════════════════════════════════════════════════════════════
-- § conv2: exact table, the relu₂ margin, the pooled vector
-- ════════════════════════════════════════════════════════════════

/-- conv2 pre-activation numerators at the image, over {DA2} (= 255·128²). -/
def c2Z : List (List (List ℤ)) :=
  {z3_lit(T["c2"])}

/-- conv2 pre-activations at the image, exact. -/
noncomputable def c2V : Tensor3 2 (2*14) (2*14) := tabR (g3 c2Z) {DA2}

{bias_nums("b2N", b2NA, f"conv2's bias numerators over {DA2}.")}
theorem conv2_chk : convTabZ 2 2 3 3 28 28 b2N 1 (g4 W2Z) (reluZ (g3 c1Z)) (g3 c2Z) = true := by
  decide +kernel

theorem conv2_fun : conv2d W2 b2 x1V = c2V :=
  conv2d_tab (fun o => by fin_cases o <;> norm_num [b2, b2N]) (by norm_num) conv2_chk

theorem c2_chk : ltAbsZ 2 28 28 (g3 c2Z) {NM2} = true := by decide +kernel

theorem c2_margin : ∀ o hi wi, {lit(AR)} < |c2V o hi wi| := fun o hi wi =>
  tabR_lt_abs c2_chk (by norm_num) (by norm_num) o hi wi

/-- The pooled features' numerators over {DA2}: the 2×2 max-pool of relu(conv2), row-major. -/
def pZ : List ℤ := {zlist(int(v) for v in T["p"])}

theorem pZ_len : pZ.length = 392 := by decide +kernel

/-- The pooled feature vector (flattened maxpool output), exact. -/
noncomputable def p2f : Vec (2 * 14 * 14) := fun i => (pZ.getD i 0 : ℝ) / {DA2}

theorem pool_chk : poolTabZ 2 14 14 (g3 c2Z) pZ = true := by decide +kernel

theorem head_feat : maxPoolFlat 2 14 14 (relu (2 * (2*14) * (2*14))
    (Tensor3.flatten (conv2d W2 b2 x1V))) = p2f := by
  rw [conv2_fun]
  exact maxPoolFlat_relu_tab (by norm_num) pool_chk

-- ════════════════════════════════════════════════════════════════
-- § The pool margin up to twins: tied windows read all-zero patches
-- ════════════════════════════════════════════════════════════════

theorem cert_chk : poolCertZ 2 14 14 (zeroTwinZ 2 3 3 28 28 (reluZ (g3 c1Z))) (g3 c2Z)
    {NPOOL} = true := by
  decide +kernel

/-- **The pool margin up to twins holds at the trained weights**, ties and all. -/
theorem pool_margin :
    MaxPool2MarginQUpTo (c := 2) (h := 14) (w := 14) {lit(AR)} (ConvPatchEq 3 3 x1V) c2V :=
  maxPool2MarginQUpTo_of_poolCertZ (by norm_num) (by norm_num) _
    (fun _ _ h => h.symm) (fun _ _ _ h₁ h₂ => h₁.trans h₂)
    (zeroTwinZ 2 3 3 28 28 (reluZ (g3 c1Z))) convPatchEq_of_zeroTwinZ (by norm_num) cert_chk

-- ════════════════════════════════════════════════════════════════
-- § The dense head: the relu₃ and relu₄ margins
-- ════════════════════════════════════════════════════════════════

{D3_DOTS}

{D3_EQS}
{head_text(T, QA)}
-- ════════════════════════════════════════════════════════════════
-- § Weight bounds
-- ════════════════════════════════════════════════════════════════

{W3_BOUND}
{weight_bound("hW4", "W4", W4B)}
{weight_bound("hW5", "W5", W5B)}
-- ════════════════════════════════════════════════════════════════
-- § The descent instance
-- ════════════════════════════════════════════════════════════════

/-- The margin radius `a·(lr·G)` of the instance, exact. -/
theorem radius_eq : ({lit(A)} : ℝ) * ((1 / 2 ^ {KEXP} : ℝ) *
    cnnConv2GradBound 2 14 14 8 8 10 3 3 {lit(A)} {lit(W3B)} {lit(W4B)} {lit(W5B)}) = {lit(AR)} := by
  norm_num [cnnConv2GradBound]

/-- **One exact-gradient SGD step on the conv2 kernel of a trained MNIST CNN, at a real 28×28
    test image with tied pool windows, decreases the cross-entropy loss** by at least
    `lr·‖∇L‖₂²/2`, at `lr = 2⁻{KEXP}`. Every hypothesis of `cnn_conv2_exact_sgd_descends` is
    discharged above. -/
theorem trained_cnn_conv2_sgd_descends_concrete :
    (cnnConv2KernelLoss (h := 14) (w := 14) b2 x1V W3 b3 W4 b4 W5 b5 lbl) (Kernel4.flatten W2 -
        (1 / 2 ^ {KEXP} : ℝ) • gradAt (cnnConv2KernelLoss (h := 14) (w := 14) b2 x1V W3 b3 W4 b4 W5 b5 lbl)
          (Kernel4.flatten W2)) ≤
      (cnnConv2KernelLoss (h := 14) (w := 14) b2 x1V W3 b3 W4 b4 W5 b5 lbl) (Kernel4.flatten W2) -
        (1 / 2 ^ {KEXP} : ℝ) * (∑ idx, gradAt (cnnConv2KernelLoss (h := 14) (w := 14) b2 x1V W3 b3 W4 b4
          W5 b5 lbl) (Kernel4.flatten W2) idx ^ 2) / 2 := by
  refine cnn_conv2_exact_sgd_descends (h := 14) (w := 14) W2 b2 x1V W3 b3 W4 b4 W5 b5 lbl
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
    cnnConv2BiasGradBound 2 14 14 8 8 10 {lit(W3B)} {lit(W4B)} {lit(W5B)} = {lit(RB2)} := by
  norm_num [cnnConv2BiasGradBound]

/-- **One exact-gradient SGD step on the conv2 BIAS of the same trained MNIST CNN, at the same
    test image, decreases the cross-entropy loss** by at least `lr·‖∇L‖₂²/2`, at `lr = 2⁻{KEXPB}`.
    Every hypothesis of `cnn_conv2_bias_exact_sgd_descends` is discharged above; the relu₂ and
    pool margins are the kernel rung's (`c2_margin`, `pool_margin`) at a smaller radius. -/
theorem trained_cnn_conv2_bias_sgd_descends_concrete :
    (cnnConv2BiasLoss (h := 14) (w := 14) W2 x1V W3 b3 W4 b4 W5 b5 lbl) (b2 -
        (1 / 2 ^ {KEXPB} : ℝ) • gradAt (cnnConv2BiasLoss (h := 14) (w := 14) W2 x1V W3 b3 W4 b4 W5
          b5 lbl) b2) ≤
      (cnnConv2BiasLoss (h := 14) (w := 14) W2 x1V W3 b3 W4 b4 W5 b5 lbl) b2 -
        (1 / 2 ^ {KEXPB} : ℝ) * (∑ o, gradAt (cnnConv2BiasLoss (h := 14) (w := 14) W2 x1V W3 b3 W4
          b4 W5 b5 lbl) b2 o ^ 2) / 2 := by
  refine cnn_conv2_bias_exact_sgd_descends (h := 14) (w := 14) W2 b2 x1V W3 b3 W4 b4 W5 b5 lbl
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
        fh.write(modularize(body))
    print(f"wrote {OUT} ({len(body.splitlines())} lines)", flush=True)

# ================================================================ net B: the conv1 rungs
OUT1 = os.path.join(ROOT, "LeanMlir/Proofs/Training/Trained/CnnDescentConv1.lean")
QB = prepare(quantize(train(True, "B"), "B"))
IDXB, TB = choose(QB, DEN_X * DEN_B, True, "CONV1")
LBLB = int(yte[IDXB])
print(f"conv1 instance: test #{IDXB} label {LBLB}, live two-layer twin windows "
      f"{TB['live_twin']}, dead {TB['n_dead']}", flush=True)
recheck(TB, QB)
DB1, DB2 = TB["den1"], TB["den2"]

A1 = frac_up(Fraction(int(TB["x0"].max()), DEN_X))
W2Bb = frac_up(max(abs(v) for v in QB["W2"].flat), DEN_W)
W3Bb = frac_up(max(abs(v) for v in QB["W3"].flat), DEN_W)
W4Bb = frac_up(max(abs(v) for v in QB["W4"].flat), DEN_W)
W5Bb = frac_up(max(abs(v) for v in QB["W5"].flat), DEN_W)
M = C * 9                      # c·kH·kW, conv2's locality factor
P1 = C * 1 * 9
BHB = D1 * (W3Bb * (D1 * (W4Bb * (NC * (W5Bb * 1)))))
G1 = P1 * (K2 * (M * (W2Bb * A1)) * BHB)
G1B = C * (K2 * (M * (W2Bb * 1)) * BHB)
minb1N = int(abs(TB["c1"]).min())
minb2N = int(abs(TB["c2"]).min())
minb3 = min(abs(v) for v in TB["d3"])
minb4 = min(abs(v) for v in TB["d4"])


def ok_conv1(lr, rho_r, num, n):
    """every hypothesis of the conv1 rungs at the conv1-preactivation radius `rho_r`; the relu₁,
    relu₂ and pool margins as the integer checks the kernel evaluates."""
    z2r = M * (W2Bb * rho_r)
    if not (ceil_frac(rho_r * DB1) < minb1N and ceil_frac(z2r * DB2) < minb2N and
            ceil_frac(2 * z2r * DB2) < TB["mingap"]):
        return False
    s3 = M * (W2Bb * (K2 * rho_r))
    if not (W3Bb * s3 < minb3 and W4Bb * (D1 * (W3Bb * s3)) < minb4):
        return False
    dl = W5Bb * (D1 * (W4Bb * (D1 * (W3Bb * s3))))
    return 2 * dl < 1 and num / (1 - 2 * dl) * lr * n <= Fraction(1, 4)


NUM1B = 2 * NC * K2 ** 2 * M ** 2 * D1 ** 2 * D1 ** 2 * W2Bb ** 2 * W3Bb ** 2 * W4Bb ** 2 * W5Bb ** 2
KEXP1 = next(k for k in range(1, 400)
             if ok_conv1(Fraction(1, 2 ** k), A1 * (Fraction(1, 2 ** k) * G1), NUM1B * A1 ** 2, P1))
R1A = A1 * (Fraction(1, 2 ** KEXP1) * G1)        # the relu₁ margin radius a·(lr·G), exact
R1Z2 = M * (W2Bb * R1A)                            # the relu₂ / pool margin radius
KEXP1B = next(k for k in range(1, 400)
              if Fraction(1, 2 ** k) * G1B <= R1A and
              ok_conv1(Fraction(1, 2 ** k), Fraction(1, 2 ** k) * G1B, NUM1B, C))
RB1 = Fraction(1, 2 ** KEXP1B) * G1B
print(f"conv1 rungs: a = {A1}, w2 = {W2Bb}, kernel lr = 2^-{KEXP1}, bias lr = 2^-{KEXP1B}",
      flush=True)

# net B's integer certificates
b1NB = bias_over(QB["b1q"], DB1)
b2NB = bias_over(QB["b2q"], DB2)
assert all(Fraction(n, DB1) == v for n, v in zip(b1NB, QB["b1"]))
assert all(Fraction(n, DB2) == v for n, v in zip(b2NB, QB["b2"]))
NX0 = int(TB["x0"].max())
assert Fraction(NX0, DEN_X) <= A1
NW2 = int(abs(QB["W2q"]).max())
assert Fraction(NW2, DEN_W) <= W2Bb
NM1B = ceil_frac(R1A * DB1)
assert R1A <= Fraction(NM1B, DB1) and NM1B < minb1N
NM2B = ceil_frac(R1Z2 * DB2)
assert R1Z2 <= Fraction(NM2B, DB2) and NM2B < minb2N
NPOOLB = ceil_frac(2 * R1Z2 * DB2)
assert 2 * R1Z2 <= Fraction(NPOOLB, DB2)
assert all(win_cert(TB["c2"][ci], TB["tw"], NPOOLB, ho, wo)
           for ci in range(C) for ho in range(H) for wo in range(H))
D3B_DEFS, D3B_DOTS, D3B_EQS, W3B_BOUND = dense3_text(TB, QB, DB2)

DIMS = "(h := 14) (w := 14)"
L1 = f"cnnConv1KernelLoss {DIMS} b1 T0 W2 b2 W3 b3 W4 b4 W5 b5 lbl"
L1B = f"cnnConv1BiasLoss {DIMS} W1 T0 W2 b2 W3 b3 W4 b4 W5 b5 lbl"

body_b = f'''{DESCENT_HEADER}

/-! # Descent at TRAINED weights — the CNN conv1 rungs, through two-layer twins

The net is the CNN chapter's `mnistCnnNoBn` architecture at 2 channels and an 8-wide head, on the
28×28 image, the same shape as `Trained.CnnDescent`: `conv 1→2 3×3 SAME + bias → relu → conv
2→2 3×3 SAME → relu → maxpool 2×2 → dense 392→8 → relu → dense 8→8 → relu → dense 8→10`. Nothing
is cropped or pooled away; only the widths differ from the chapter's 32 channels and 512-wide
head.

`cnn_conv1_exact_sgd_descends` and `cnn_conv1_bias_exact_sgd_descends` — one exact-gradient SGD
step on the FIRST conv's kernel, or its bias, decreases the cross-entropy loss — instantiated at
TRAINED, /128-rationalized weights (biases /16384) and REAL MNIST test image #{IDXB} (label
{LBLB}, classified correctly), all 784 pixels `k/255`, every hypothesis discharged by exact
arithmetic: `trained_cnn_conv1_sgd_descends_concrete` at learning rate `2⁻{KEXP1}` and
`trained_cnn_conv1_bias_sgd_descends_concrete` at `2⁻{KEXP1B}` (its radius sits inside the kernel
rung's, so every relu and pool margin carries over by monotonicity).

This net has a trained conv1 bias. `Trained.CnnDescent`'s bias-free conv1 cannot serve these
rungs: their relu₁ margin needs every conv1 pre-activation nonzero, and a bias-free conv1 is
exactly zero on a blank patch.

The pool hypothesis takes two-layer twins `ConvPatchEq2 3 3 T0`: conv2-output cells whose 5×5
receptive fields through both convs lie in a blank image region and meet the zero padding the
same way (`convPatchEq2_of_blankTwinZ`), so their conv2 outputs are equal for every conv1 kernel
and bias. {TB["live_twin"]} live windows tie that way and {TB["n_dead"]} are dead, of 392; every
other cell clears the margin (`cert_chk`, through `maxPool2MarginQUpTo_of_poolCertZ`). No pool-tie
regularizer was used in training.

Every table is nested `List ℤ` numerators over one denominator, and each whole-table fact
(`conv1_chk`, `conv2_chk`, `pool_chk`, `cert_chk`, the margins and bounds) is one kernel
evaluation, transported to `ℝ` by `Architectures/ListConv`; the 392-term dense3 sums are
`Foundation/ListDot` dots. The generator prints the net's test accuracy.

As in `Trained.CnnDescent`, the step is the exact gradient and the decrease `lr·‖∇L‖₂²/2` is not
shown positive. Generated by `scripts/certs/trained_cnn_descent.py`; weights and input are DATA. -/

namespace Proofs
namespace TrainedCnnDescentConv1

-- ════════════════════════════════════════════════════════════════
-- § Trained weights and the input (data)
-- ════════════════════════════════════════════════════════════════

/-- MNIST test image #{IDXB} (digit {LBLB}), the 28×28 pixels as integers `0..255`. -/
def x0Z : List (List (List ℤ)) :=
  {z3_lit(TB["x0"])}

/-- Test image #{IDXB}, pixels `k/255`. -/
noncomputable def T0 : Tensor3 1 (2*14) (2*14) := tabR (g3 x0Z) 255

/-- conv1 kernel numerators (1→2, 3×3) over 128. -/
def W1Z : List (List (List (List ℤ))) :=
  {k4_lit(QB["W1q"], C, 1)}

/-- conv1 kernel (1→2, 3×3), entries k/128. -/
noncomputable def W1 : Kernel4 2 1 3 3 := kerR (g4 W1Z) 128

noncomputable def b1 : Vec 2 := {vec_lit(QB["b1"])}

/-- conv2 kernel numerators (2→2, 3×3) over 128. -/
def W2Z : List (List (List (List ℤ))) :=
  {k4_lit(QB["W2q"], C, C)}

/-- conv2 kernel (2→2, 3×3), entries k/128. -/
noncomputable def W2 : Kernel4 2 2 3 3 := kerR (g4 W2Z) 128

noncomputable def b2 : Vec 2 := {vec_lit(QB["b2"])}

{D3B_DEFS}
noncomputable def b3 : Vec 8 := {vec_lit(QB["b3"])}

noncomputable def W4 : Mat 8 8 :=
  {mat_lit(QB["W4"])}

noncomputable def b4 : Vec 8 := {vec_lit(QB["b4"])}

noncomputable def W5 : Mat 8 10 :=
  {mat_lit(QB["W5"])}

noncomputable def b5 : Vec 10 := {vec_lit(QB["b5"])}

/-- The label of test image #{IDXB}. -/
def lbl : Fin 10 := {LBLB}

theorem x0_chk : absLeZ 1 28 28 (g3 x0Z) {NX0} = true := by decide +kernel

theorem x0_bound : ∀ cc i j, |T0 cc i j| ≤ {lit(A1)} := fun cc i j =>
  tabR_abs_le x0_chk (by norm_num) (by norm_num) cc i j

-- ════════════════════════════════════════════════════════════════
-- § conv1: exact table, the relu₁ margin, the conv2 input
-- ════════════════════════════════════════════════════════════════

/-- conv1 pre-activation numerators at the image, over {DB1} (= 255·128²). -/
def c1Z : List (List (List ℤ)) :=
  {z3_lit(TB["c1"])}

/-- conv1 pre-activations at the image, exact. -/
noncomputable def c1V : Tensor3 2 (2*14) (2*14) := tabR (g3 c1Z) {DB1}

{bias_nums("b1N", b1NB, f"conv1's bias numerators over {DB1}.")}
theorem conv1_chk : convTabZ 2 1 3 3 28 28 b1N {TB["mc1"]} (g4 W1Z) (g3 x0Z) (g3 c1Z) = true := by
  decide +kernel

theorem conv1_fun : conv2d W1 b1 T0 = c1V :=
  conv2d_tab (fun o => by fin_cases o <;> norm_num [b1, b1N]) (by norm_num) conv1_chk

theorem c1_chk : ltAbsZ 2 28 28 (g3 c1Z) {NM1B} = true := by decide +kernel

theorem c1_margin : ∀ o hi wi, {lit(R1A)} < |c1V o hi wi| := fun o hi wi =>
  tabR_lt_abs c1_chk (by norm_num) (by norm_num) o hi wi

/-- relu(conv1) at the image (conv2's input), exact. -/
noncomputable def x1V : Tensor3 2 (2*14) (2*14) := tabR (reluZ (g3 c1Z)) {DB1}

theorem relu_c1 : relu (2 * (2*14) * (2*14)) (Tensor3.flatten c1V) = Tensor3.flatten x1V :=
  relu_flatten_tabR (g3 c1Z) (by norm_num)

/-- **conv2's input is the real image's conv1 activation.** -/
theorem x1_fun : Tensor3.unflatten (relu (2 * (2*14) * (2*14)) (Tensor3.flatten (conv2d W1 b1 T0))) =
    x1V := by
  rw [conv1_fun, relu_c1, Tensor3.unflatten_flatten]

-- ════════════════════════════════════════════════════════════════
-- § conv2: exact table, the relu₂ margin, the pooled vector
-- ════════════════════════════════════════════════════════════════

/-- conv2 pre-activation numerators at the image, over {DB2} (= 255·128³). -/
def c2Z : List (List (List ℤ)) :=
  {z3_lit(TB["c2"])}

/-- conv2 pre-activations at the image, exact. -/
noncomputable def c2V : Tensor3 2 (2*14) (2*14) := tabR (g3 c2Z) {DB2}

{bias_nums("b2N", b2NB, f"conv2's bias numerators over {DB2}.")}
theorem conv2_chk : convTabZ 2 2 3 3 28 28 b2N 1 (g4 W2Z) (reluZ (g3 c1Z)) (g3 c2Z) = true := by
  decide +kernel

theorem conv2_fun : conv2d W2 b2 x1V = c2V :=
  conv2d_tab (fun o => by fin_cases o <;> norm_num [b2, b2N]) (by norm_num) conv2_chk

theorem c2_chk : ltAbsZ 2 28 28 (g3 c2Z) {NM2B} = true := by decide +kernel

theorem c2_margin : ∀ o hi wi, {lit(R1Z2)} < |c2V o hi wi| := fun o hi wi =>
  tabR_lt_abs c2_chk (by norm_num) (by norm_num) o hi wi

/-- The pooled features' numerators over {DB2}: the 2×2 max-pool of relu(conv2), row-major. -/
def pZ : List ℤ := {zlist(int(v) for v in TB["p"])}

theorem pZ_len : pZ.length = 392 := by decide +kernel

/-- The pooled feature vector (flattened maxpool output), exact. -/
noncomputable def p2f : Vec (2 * 14 * 14) := fun i => (pZ.getD i 0 : ℝ) / {DB2}

theorem pool_chk : poolTabZ 2 14 14 (g3 c2Z) pZ = true := by decide +kernel

theorem head_feat : maxPoolFlat 2 14 14 (relu (2 * (2*14) * (2*14))
    (Tensor3.flatten (conv2d W2 b2 x1V))) = p2f := by
  rw [conv2_fun]
  exact maxPoolFlat_relu_tab (by norm_num) pool_chk

-- ════════════════════════════════════════════════════════════════
-- § The pool margin up to two-layer twins
-- ════════════════════════════════════════════════════════════════

theorem cert_chk : poolCertZ 2 14 14 (blankTwinZ 1 3 3 28 28 (g3 x0Z)) (g3 c2Z)
    {NPOOLB} = true := by
  decide +kernel

/-- **The pool margin up to two-layer twins holds at the trained weights**, ties and all. -/
theorem pool_margin :
    MaxPool2MarginQUpTo (c := 2) (h := 14) (w := 14) {lit(R1Z2)} (ConvPatchEq2 3 3 T0) c2V :=
  maxPool2MarginQUpTo_of_poolCertZ (by norm_num) (by norm_num) _
    (fun _ _ h => h.symm) (fun _ _ _ h₁ h₂ => h₁.trans h₂)
    (blankTwinZ 1 3 3 28 28 (g3 x0Z)) convPatchEq2_of_blankTwinZ (by norm_num) cert_chk

-- ════════════════════════════════════════════════════════════════
-- § The dense head: the relu₃ and relu₄ margins
-- ════════════════════════════════════════════════════════════════

{D3B_DOTS}

{D3B_EQS}
{head_text(TB, QB)}
-- ════════════════════════════════════════════════════════════════
-- § Weight bounds
-- ════════════════════════════════════════════════════════════════

theorem W2_chk : kerAbsLeZ 2 2 3 3 (g4 W2Z) {NW2} = true := by decide +kernel

theorem hW2 : ∀ o cc kh kw, |W2 o cc kh kw| ≤ {lit(W2Bb)} := fun o cc kh kw =>
  kerR_abs_le W2_chk (by norm_num) (by norm_num) o cc kh kw

{W3B_BOUND}
{weight_bound("hW4", "W4", W4Bb)}
{weight_bound("hW5", "W5", W5Bb)}
-- ════════════════════════════════════════════════════════════════
-- § The descent instances
-- ════════════════════════════════════════════════════════════════

/-- The relu₁ margin radius `a·(lr·G)` of the kernel instance, exact. -/
theorem radius_eq : ({lit(A1)} : ℝ) * ((1 / 2 ^ {KEXP1} : ℝ) *
    cnnConv1GradBound 1 2 14 14 8 8 10 3 3 {lit(A1)} {lit(W2Bb)} {lit(W3Bb)} {lit(W4Bb)} {lit(W5Bb)}) =
      {lit(R1A)} := by
  norm_num [cnnConv1GradBound]

/-- **One exact-gradient SGD step on the conv1 kernel of a trained MNIST CNN, at a real 28×28
    test image with two-layer-twin pool ties, decreases the cross-entropy loss** by at least
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
    cnnConv1BiasGradBound 2 14 14 8 8 10 3 3 {lit(W2Bb)} {lit(W3Bb)} {lit(W4Bb)} {lit(W5Bb)} =
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
        fh.write(modularize(body_b))
    print(f"wrote {OUT1} ({len(body_b.splitlines())} lines)", flush=True)
