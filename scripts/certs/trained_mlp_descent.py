"""The MLP chapter's trained-weight descent instance.

Emits LeanMlir/Proofs/Training/Trained/MlpDescent.lean: one exact-gradient SGD step on the
input layer `W0` of the trained 784->8->8->10 MNIST MLP of `trained_mlp_witness.py` (same seed,
same recipe, same /128 weights -- asserted below against the lists committed in
`MlpWitness.lean`, which the emitted file imports), at one real 28x28 test image, decreases the
cross-entropy loss: `Proofs.mlp_input_sgd_descends` with every hypothesis discharged.

The gradient is `dL/dW0[i,k] = x_i * c_k`, `c = mask0 . W1 (mask1 . W2 (softmax - onehot))`, so
the two sums the window needs factor:

    S1 := sum |grad|  = (sum_i x_i)   * sum_k |c_k|
    S2 := sum grad^2  = (sum_i x_i^2) * sum_k c_k^2

Writing `G = mask0 W1 mask1 W2` (8x10, exact rationals over 128^2) and using `sum_j p_j = 1`,
`c_k = sum_{j != lbl} p_j (G[k,j] - G[k,lbl])`. Hence, with ZERO exponential evaluations,

    |c_k| <= U_k := max_{j != lbl} |G[k,j] - G[k,lbl]|                 so S1 <= Sx * sum_k U_k

and on a row `k*` whose nine differences share a sign with least magnitude `L`,
`|c_k*| >= L (1 - p_lbl)`. The image is MISCLASSIFIED, so `z_lbl <= z_pred` is an exact rational
inequality, `p_lbl <= 1/2` by exp-monotonicity alone, and

    S2 >= Sq * L^2 / 4.

The image is the first test image the rationalized net misclassifies that sits off every ReLU
kink and has such a sign-definite row; the row is the one with the largest `L`. The step size is
the largest power of two that clears both margins and the small-step condition and leaves the
smoothness condition `h2` a factor of two. Every fact is asserted here with exact integers /
`Fraction`s in EXACTLY the bound-forms the Lean discharges.

Not in CI: it needs MNIST in data/, which CI does not have. Regenerate by hand and confirm the
committed Lean comes back byte-identical.
"""
import numpy as np, os, re, sys
from fractions import Fraction
from _gencheck import modularize, module_header  # noqa: E402  (module-system header on what we write)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
from _mnist_io import mnist  # noqa: E402
from _leanlit import frac, zlist, zlit  # noqa: E402
OUT = os.path.join(ROOT, "LeanMlir/Proofs/Training/Trained/MlpDescent.lean")
WITNESS = os.path.join(ROOT, "LeanMlir/Proofs/Training/Trained/MlpWitness.lean")
DIM, H, K = 784, 8, 10
PX, WD = 255, 128                       # pixel and weight denominators
C0, C1 = PX * WD, PX * WD * WD          # layer-1 / layer-2 pre-activation denominators
C2 = C1 * WD                            # logit denominator

Xtr_raw, ytr = mnist("train", flat=True); Xte_raw, yte = mnist("test", flat=True)
Ste = Xte_raw.astype(np.int64)
Xtr = Xtr_raw / 255.0; Xte = Xte_raw / 255.0

# ── train: the recipe of trained_mlp_witness.py, verbatim (same seed, so the same net) ──
rng = np.random.default_rng(0)
W0 = rng.normal(0, np.sqrt(2.0 / DIM), (DIM, H))
W1 = rng.normal(0, np.sqrt(2.0 / H), (H, H))
W2 = rng.normal(0, np.sqrt(2.0 / H), (H, K))
lr_train, bs = 0.05, 64
for ep in range(12):
    idx = rng.permutation(len(Xtr))
    for b in range(0, len(Xtr), bs):
        xb = Xtr[idx[b:b+bs]]; yb = ytr[idx[b:b+bs]]
        h0 = xb @ W0; a0 = np.maximum(h0, 0)
        h1 = a0 @ W1; a1 = np.maximum(h1, 0)
        z = a1 @ W2; z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        g2 = p.copy(); g2[np.arange(len(yb)), yb] -= 1; g2 /= len(yb)
        g1 = (g2 @ W2.T) * (h1 > 0)
        g0 = (g1 @ W1.T) * (h0 > 0)
        W2 -= lr_train * a1.T @ g2; W1 -= lr_train * a0.T @ g1; W0 -= lr_train * xb.T @ g0
W0q = np.round(W0 * WD).astype(np.int64)
W1q = np.round(W1 * WD).astype(np.int64)
W2q = np.round(W2 * WD).astype(np.int64)
w0 = [[int(W0q[i, k]) for i in range(DIM)] for k in range(H)]      # columns, as the Lean stores them
w1 = [[int(W1q[k, m]) for k in range(H)] for m in range(H)]
w2 = [[int(W2q[m, c]) for m in range(H)] for c in range(K)]

# ── the emitted file imports MlpWitness for the weights: they must be THESE weights ──
wit = open(WITNESS).read()


def committed(name):
    m = re.search(rf"^def {name} : List ℤ := \[(.*)\]$", wit, re.M)
    assert m, name
    return [-int(t.split()[1]) - 1 if t.startswith("Int.negSucc") else int(t)
            for t in (s.strip() for s in m.group(1).split(","))]


for k in range(H):
    assert committed(f"w0c{k}Z") == w0[k], f"w0c{k}Z differs from MlpWitness.lean"
    assert committed(f"w1c{k}Z") == w1[k], f"w1c{k}Z differs from MlpWitness.lean"
for c in range(K):
    assert committed(f"w2c{c}Z") == w2[c], f"w2c{c}Z differs from MlpWitness.lean"

# exact integer forward on the whole test set
D0all = Ste @ W0q                                           # x (255*128)
D1all = np.maximum(D0all, 0) @ W1q                          # x (255*128^2)
D2all = np.maximum(D1all, 0) @ W2q                          # x (255*128^3)
assert abs(D2all).max() < 2 ** 62
accq = (D2all.argmax(1) == yte).mean()
W1B = int(abs(W1q).max()); W2B = int(abs(W2q).max())        # entrywise bounds, numerators
w1b = Fraction(W1B, WD); w2b = Fraction(W2B, WD)
print(f"mlp 784->8->8->10: q-acc {accq:.4f} |W1q|max {W1B} |W2q|max {W2B}")


def gmat(M0, M1):
    """G = mask0 . W1 . mask1 . W2, integer numerators over 128^2."""
    return [[M0[k] * sum(int(W1q[k, m]) * M1[m] * int(W2q[m, j]) for m in range(H))
             for j in range(K)] for k in range(H)]


def definite_rows(G, lbl):
    """Rows whose nine differences G[k,j] - G[k,lbl], j != lbl, share a strict sign: (k, sign, L)."""
    out = []
    for k in range(H):
        d = [G[k][j] - G[k][lbl] for j in range(K) if j != lbl]
        if all(v > 0 for v in d) or all(v < 0 for v in d):
            out.append((k, 1 if d[0] > 0 else -1, min(abs(v) for v in d)))
    return out


# ── the image: first misclassified test image off every ReLU kink with a sign-definite row ──
IDX = None
for i in np.nonzero((D2all.argmax(1) != yte) & (D0all != 0).all(1) & (D1all != 0).all(1))[0]:
    i = int(i)
    G_ = gmat([int(v > 0) for v in D0all[i]], [int(v > 0) for v in D1all[i]])
    if definite_rows(G_, int(yte[i])):
        IDX = i
        break
assert IDX is not None
s = [int(v) for v in Ste[IDX]]
LBL = int(yte[IDX])
dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))
D0 = [dot(s, w0[k]) for k in range(H)]
A0 = [max(v, 0) for v in D0]
D1 = [dot(A0, w1[m]) for m in range(H)]
A1 = [max(v, 0) for v in D1]
D2 = [dot(A1, w2[c]) for c in range(K)]
PRED = max(range(K), key=lambda c: D2[c])
assert all(v != 0 for v in D0) and all(v != 0 for v in D1)
assert PRED != LBL and D2[LBL] <= D2[PRED]
assert all(0 <= v <= 255 for v in s)
pre0 = [Fraction(v, C0) for v in D0]
pre1 = [Fraction(v, C1) for v in D1]
M0 = [1 if v > 0 else 0 for v in D0]
M1 = [1 if v > 0 else 0 for v in D1]
G = gmat(M0, M1)
Gq = [[Fraction(v, WD * WD) for v in row] for row in G]
rows = definite_rows(G, LBL)
KS, SGN, Lint = max(rows, key=lambda r: (r[2], -r[0]))
Lq = Fraction(Lint, WD * WD)
U = [max(abs(Gq[k][j] - Gq[k][LBL]) for j in range(K) if j != LBL) for k in range(H)]
SUMX = sum(s); SUMSQ = sum(v * v for v in s)
Sx = Fraction(SUMX, PX); Sq = Fraction(SUMSQ, PX * PX)
USUM = sum(U)
S1UB = Sx * USUM                                            # sum |grad| <= S1UB
S2LB = Sq * (Lq / 2) ** 2                                   # sum grad^2 >= S2LB
assert S2LB > 0
m0 = min(map(abs, pre0)); m1 = min(map(abs, pre1))
KC = 2 * K * H ** 2 * w1b ** 2 * w2b ** 2                   # 2·d₃·d₂²·w₁²·w₂²·a², a = 1


def window(LR):
    R = LR * S1UB                                           # the step's l1 radius, upper bound
    T = 1 - 2 * (w2b * (H * (w1b * R)))
    ok = R < m0 and w1b * R < m1 and T > 0 and KC * R ** 2 <= (LR * S2LB / 4) * T
    return R, T, ok, (float((LR * S2LB / 4) * T / (KC * R ** 2)) if T > 0 else 0.0)


EXP = next(e for e in range(1, 200) if window(Fraction(1, 2 ** e))[2] and window(Fraction(1, 2 ** e))[3] >= 2.0)
LR = Fraction(1, 2 ** EXP)
R, T, ok, h2_margin = window(LR)
assert ok and h2_margin >= 2.0
# the forms the Lean closes, one by one
assert all(1 * R < abs(v) for v in pre0)                    # hmargin0
assert all(w1b * (1 * R) < abs(v) for v in pre1)            # hmargin1
assert 2 * (w2b * (H * (w1b * (1 * R)))) < 1                # hsmall
assert KC * R ** 2 <= (LR * S2LB / 4) * T                   # h2, cross-multiplied
for k in range(H):                                          # |c_k| <= U_k, as linear forms in p
    assert all(abs(Gq[k][j] - Gq[k][LBL]) <= U[k] for j in range(K))
assert all(SGN * (Gq[KS][j] - Gq[KS][LBL]) >= Lq for j in range(K) if j != LBL)
DROP = LR * S2LB / 2
print(f"image #{IDX}: label {LBL}, predicted {PRED}; logits {[round(v / C2, 3) for v in D2]}")
print(f"  masks {M0} {M1}; min |preact| {float(m0):.4f} / {float(m1):.4f}")
print(f"  Sx={Sx}={float(Sx):.3f} Sq={Sq}={float(Sq):.3f}")
print(f"  U_k {[round(float(u), 3) for u in U]} sum {float(USUM):.3f}")
print(f"  sign-definite rows {[(k, sg, round(l / WD ** 2, 3)) for k, sg, l in rows]}; using row {KS}, L={Lq}={float(Lq):.4f}")
print(f"  S1ub={S1UB}={float(S1UB):.3f} S2lb={S2LB}={float(S2LB):.4f}")
print(f"  lr=2^-{EXP}: radius<={float(R):.3e}; h2 margin {h2_margin:.2f}x; guaranteed drop >= {float(DROP):.3e}")

FR_LR = f"(1 / 2 ^ {EXP} : ℝ)"
LOSS = "mlpInputLoss b8 W1V b8 W2V b10 xd lblD"
GRAD = f"gradAt ({LOSS}) (Mat.flatten W0V)"
P = lambda j: f"softmax 10 zd {j}"
n_on0, n_on1 = sum(M0), sum(M1)


def lin(coefs, var):
    """`c0 * var 0 + ... + c9 * var 9` with exact rational coefficients."""
    return " + ".join(f"{frac(c)} * {var(j)}" for j, c in enumerate(coefs))


L = []
A_ = L.append
A_(module_header(["LeanMlir.Proofs.Training.SgdDescent.Mlp", "LeanMlir.Proofs.Training.Trained.MlpWitness",
                  "LeanMlir.Proofs.Foundation.ListGetD"],
                 ["Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp"]))
A_("")
A_("/-! # Descent at TRAINED weights — the MLP's input layer")
A_("")
A_("The net is `MlpWitness`'s: Chapter 2's architecture at width 8 on the 28×28 image,")
A_("`dense 784→8 → relu → dense 8→8 → relu → dense 8→10`, bias-free, trained on MNIST and")
A_("rationalized to `k/128` (`TrainedMlp.W0V`, `W1V`, `W2V`, imported, not re-emitted). The")
A_(f"example is MNIST test image #{IDX}, all 784 pixels, `k/255`: a {LBL} the net reads as a {PRED}.")
A_("")
A_("**One SGD step with the exact gradient on the input weights `W₀`, at learning rate")
A_(f"`2⁻{EXP}`, decreases that example's cross-entropy loss by at least `lr·‖∇L‖₂²/2`**")
A_("(`trained_mlp_input_sgd_descends_concrete`): `mlp_input_sgd_descends` at `η = 0`, `a = 1`,")
A_(f"`w₁ = {W1B}/128`, `w₂ = {W2B}/128`, every hypothesis discharged. The step crosses both ReLU")
A_("layers, so the theorem needs both sign patterns frozen along it; the margins are read off")
A_(f"the exact pre-activations (`pre0_eq`: {n_on0} units on, {H - n_on0} off; `pre1_eq`: {n_on1} on, {H - n_on1} off),")
A_("each 784-term sum one kernel evaluation over `List ℤ` numerators (`Foundation/ListDot`).")
A_("")
A_("The window needs `Σ|∇|` from above and `Σ∇²` from below, and the gradient holds a softmax.")
A_("No `exp` is evaluated. By `mlp_input_loss_gradAt` the gradient is the outer product")
A_("`∇[i,k] = xᵢ·c_k`, `c = mask⁰ ⊙ W₁(mask¹ ⊙ W₂(p − onehot))` (`grad_key`), so the sums factor")
A_("as `(Σxᵢ)·Σ|c_k|` and `(Σxᵢ²)·Σc_k²`. Each `c_k` is a fixed rational combination of the ten")
A_("softmax entries (`backC_0` … `backC_7`), and `p` is a probability vector, so")
A_("`|c_k| ≤ max_j |G_kj − G_k,lbl|` (`cot_abs_le`), which gives `gradL1_le`. For the lower bound")
A_(f"the image is MISCLASSIFIED: `z_{LBL} ≤ z_{PRED}` is a rational inequality (`hz_lbl_le`), so")
A_(f"`p_{LBL} ≤ 1/2` by monotonicity of `exp` alone (`sm_lbl_le_half`); row {KS} of `G` has all nine")
A_(f"differences `G_{KS},j − G_{KS},{LBL}` of one sign, so `|c_{KS}| ≥ {Lq.numerator}/{Lq.denominator}·(1 − p_{LBL})` (`cot_lower`), which")
A_("gives `gradSq_lower`.")
A_("")
A_("The drop is therefore strictly positive (`trained_mlp_input_sgd_strictly_descends`, `<`),")
A_(f"by at least `lr·S2lb/2 ≈ {float(DROP):.2e}`. What is not shown: the gradient is exact (no float")
A_("model, unlike `LinearDescent`), the step moves `W₀` only, and it is one example, not a batch.")
A_("")
A_("Generated by `scripts/certs/trained_mlp_descent.py`; weights and input are DATA. -/")
A_("")
A_("namespace Proofs")
A_("namespace TrainedMlpDescent")
A_("")
A_("open scoped BigOperators")
A_("open TrainedMlp")
A_("")
A_("-- ── the data ──")
A_("")
A_(f"/-- MNIST test image #{IDX} (digit {LBL}), the 784 pixels as integers `0..255`. -/")
A_("def xdZ : List ℤ := " + zlist(s))
A_("")
A_("theorem xdZ_len : xdZ.length = 784 := by decide +kernel")
A_("")
A_("theorem xdZ_range : ∀ z ∈ xdZ, 0 ≤ z ∧ z ≤ 255 := by decide +kernel")
A_("")
A_(f"/-- MNIST test image #{IDX} (digit {LBL}), pixels `k/255`. -/")
A_("noncomputable def xd : Vec 784 := fun i => (xdZ.getD i 0 : ℝ) / 255")
A_("")
A_("/-- The true label. -/")
A_(f"def lblD : Fin 10 := {LBL}")
A_("")
A_("/-- The class the net predicts. -/")
A_(f"def cD : Fin 10 := {PRED}")
A_("")
A_("theorem xd_nonneg : ∀ i, 0 ≤ xd i := by")
A_("  intro i")
A_("  have h := getD_of_forall xdZ (fun z => 0 ≤ z ∧ z ≤ 255) (by norm_num)")
A_("    xdZ_range i")
A_("  have h0 : (0 : ℝ) ≤ (xdZ.getD i 0 : ℝ) := by exact_mod_cast h.1")
A_("  exact div_nonneg h0 (by norm_num)")
A_("")
A_("theorem xd_le_one : ∀ i, |xd i| ≤ 1 := by")
A_("  intro i")
A_("  have h := getD_of_forall xdZ (fun z => 0 ≤ z ∧ z ≤ 255) (by norm_num)")
A_("    xdZ_range i")
A_("  have h1 : (xdZ.getD i 0 : ℝ) ≤ 255 := by exact_mod_cast h.2")
A_("  rw [abs_of_nonneg (xd_nonneg i)]")
A_("  show (xdZ.getD i 0 : ℝ) / 255 ≤ 1")
A_("  rw [div_le_one (by norm_num)]")
A_("  exact h1")
A_("")
A_(f"theorem xdZ_sum : xdZ.sum = {SUMX} := by decide +kernel")
A_(f"theorem xdZ_sq : dotZ xdZ xdZ = {SUMSQ} := by decide +kernel")
A_("")
A_("/-- `Σᵢ xᵢ`, exactly. -/")
A_(f"theorem xd_sum : (∑ i, xd i) = {frac(Sx)} := by")
A_("  show (∑ i : Fin 784, (xdZ.getD i 0 : ℝ) / 255) = _")
A_("  rw [← Finset.sum_div, sum_getD 784 xdZ xdZ_len, xdZ_sum]")
A_("  norm_num")
A_("")
A_("/-- `Σᵢ xᵢ²`, exactly. -/")
A_(f"theorem xd_sq_sum : (∑ i, xd i ^ 2) = {frac(Sq)} := by")
A_("  have h := sum_getD_div xdZ_len xdZ_len xdZ_sq 255 255")
A_("  show (∑ i : Fin 784, ((xdZ.getD i 0 : ℝ) / 255) ^ 2) = _")
A_("  simp only [pow_two]")
A_("  rw [h]")
A_("  norm_num")
A_("")
A_("-- ── layer 1: the eight 784-term pre-activations, one kernel evaluation each ──")
A_("")
for k in range(H):
    A_(f"theorem dz0_{k} : dotZ xdZ w0c{k}Z = {zlit(D0[k])} := by decide +kernel")
A_("")
A_(f"/-- The exact first-layer pre-activations at the image ({n_on0} positive, {H - n_on0} negative). -/")
A_("noncomputable def hpre0d : Vec 8 := ![" + ", ".join(frac(v) for v in pre0) + "]")
A_("")
for k in range(H):
    A_(f"theorem pre0_{k} : dense W0V b8 xd {k} = {frac(pre0[k])} := by")
    A_(f"  show (∑ i : Fin 784, (xdZ.getD i 0 : ℝ) / 255 * ((w0c{k}Z.getD i 0 : ℝ) / 128)) + 0 = _")
    A_(f"  rw [sum_getD_div xdZ_len w0c{k}Z_len dz0_{k} 255 128]")
    A_("  norm_num")
    A_("")
A_("/-- The first-layer pre-activations at the image, exactly. -/")
A_("theorem pre0_eq (k : Fin 8) : dense W0V b8 xd k = hpre0d k := by")
A_("  fin_cases k")
A_("  exacts [" + ", ".join(f"pre0_{k}" for k in range(H)) + "]")
A_("")
A_("theorem pre0_ne : ∀ k, dense W0V b8 xd k ≠ 0 := by")
A_("  intro k")
A_("  rw [pre0_eq]")
A_("  fin_cases k <;> norm_num [hpre0d]")
A_("")
A_("-- ── layer 2: the eight 8-term pre-activations over the layer-1 activations ──")
A_("")
A_(f"/-- The first hidden layer's activations at the image, numerators over {C0} (= 255·128). -/")
A_("def a0dZ : List ℤ := " + zlist(A0))
A_("")
A_("theorem a0dZ_len : a0dZ.length = 8 := by decide +kernel")
A_("")
A_("/-- The first hidden layer's output at the image, exactly. -/")
A_("theorem relu_pre0_eq :")
A_(f"    relu 8 (dense W0V b8 xd) = fun k : Fin 8 => (a0dZ.getD k 0 : ℝ) / {C0} := by")
A_("  funext k")
A_("  show (if dense W0V b8 xd k > 0 then dense W0V b8 xd k else 0) = _")
A_("  rw [pre0_eq]")
A_("  fin_cases k <;> norm_num [hpre0d, a0dZ]")
A_("")
for m in range(H):
    A_(f"theorem dz1_{m} : dotZ a0dZ w1c{m}Z = {zlit(D1[m])} := by decide +kernel")
A_("")
A_(f"/-- The exact second-layer pre-activations at the image ({n_on1} positive, {H - n_on1} negative). -/")
A_("noncomputable def hpre1d : Vec 8 := ![" + ", ".join(frac(v) for v in pre1) + "]")
A_("")
for m in range(H):
    A_(f"theorem pre1_{m} : dense W1V b8 (relu 8 (dense W0V b8 xd)) {m} = {frac(pre1[m])} := by")
    A_("  rw [relu_pre0_eq]")
    A_(f"  show (∑ k : Fin 8, (a0dZ.getD k 0 : ℝ) / {C0} * ((w1c{m}Z.getD k 0 : ℝ) / 128)) + 0 = _")
    A_(f"  rw [sum_getD_div a0dZ_len w1c{m}Z_len dz1_{m} {C0} 128]")
    A_("  norm_num")
    A_("")
A_("/-- The second-layer pre-activations at the image, exactly. -/")
A_("theorem pre1_eq (m : Fin 8) :")
A_("    dense W1V b8 (relu 8 (dense W0V b8 xd)) m = hpre1d m := by")
A_("  fin_cases m")
A_("  exacts [" + ", ".join(f"pre1_{m}" for m in range(H)) + "]")
A_("")
A_("theorem pre1_ne : ∀ m, dense W1V b8 (relu 8 (dense W0V b8 xd)) m ≠ 0 := by")
A_("  intro m")
A_("  rw [pre1_eq]")
A_("  fin_cases m <;> norm_num [hpre1d]")
A_("")
A_("-- ── the logits: the image is misclassified ──")
A_("")
A_(f"/-- The second hidden layer's activations at the image, numerators over {C1} (= 255·128²). -/")
A_("def a1dZ : List ℤ := " + zlist(A1))
A_("")
A_("theorem a1dZ_len : a1dZ.length = 8 := by decide +kernel")
A_("")
A_("/-- The second hidden layer's output at the image, exactly. -/")
A_("theorem relu_pre1_eq :")
A_("    relu 8 (dense W1V b8 (relu 8 (dense W0V b8 xd))) =")
A_(f"      fun m : Fin 8 => (a1dZ.getD m 0 : ℝ) / {C1} := by")
A_("  funext m")
A_("  show (if dense W1V b8 (relu 8 (dense W0V b8 xd)) m > 0")
A_("    then dense W1V b8 (relu 8 (dense W0V b8 xd)) m else 0) = _")
A_("  rw [pre1_eq]")
A_("  fin_cases m <;> norm_num [hpre1d, a1dZ]")
A_("")
A_("/-- The trained net's logits at the image. -/")
A_("noncomputable def zd : Vec 10 :=")
A_("  dense W2V b10 (relu 8 (dense W1V b8 (relu 8 (dense W0V b8 xd))))")
A_("")
for c in (LBL, PRED):
    A_(f"theorem w2c{c}Z_len : w2c{c}Z.length = 8 := by decide +kernel")
    A_(f"theorem dz2_{c} : dotZ a1dZ w2c{c}Z = {zlit(D2[c])} := by decide +kernel")
A_("")
A_(f"/-- The image is misclassified: logit {LBL} ≤ logit {PRED}, exactly. -/")
A_("theorem hz_lbl_le : zd lblD ≤ zd cD := by")
A_("  unfold zd")
A_("  rw [relu_pre1_eq]")
A_(f"  show (∑ m : Fin 8, (a1dZ.getD m 0 : ℝ) / {C1} * ((w2c{LBL}Z.getD m 0 : ℝ) / 128)) + 0")
A_(f"    ≤ (∑ m : Fin 8, (a1dZ.getD m 0 : ℝ) / {C1} * ((w2c{PRED}Z.getD m 0 : ℝ) / 128)) + 0")
A_(f"  rw [sum_getD_div a1dZ_len w2c{LBL}Z_len dz2_{LBL} {C1} 128,")
A_(f"    sum_getD_div a1dZ_len w2c{PRED}Z_len dz2_{PRED} {C1} 128]")
A_("  norm_num")
A_("")
A_("-- ── softmax facts at the trained logits (no `exp` is ever evaluated) ──")
A_("")
A_("theorem sm_pos : ∀ j, 0 < softmax 10 zd j := fun j =>")
A_("  div_pos (Real.exp_pos _)")
A_("    (Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨j, Finset.mem_univ j⟩)")
A_("")
A_("theorem sm_sum : (∑ j, softmax 10 zd j) = 1 := by")
A_("  show (∑ j, Real.exp (zd j) / ∑ k, Real.exp (zd k)) = 1")
A_("  simp only [div_eq_mul_inv, ← Finset.sum_mul]")
A_("  exact mul_inv_cancel₀ (ne_of_gt (Finset.sum_pos")
A_("    (fun k _ => Real.exp_pos _) ⟨lblD, Finset.mem_univ _⟩))")
A_("")
A_("/-- The ten softmax entries sum to one, written out. -/")
A_("theorem sm_sum_ten : " + " + ".join(P(j) for j in range(K)) + " = 1 := by")
A_("  have h := sm_sum")
A_("  simp only [Fin.sum_univ_succ, Fin.sum_univ_zero, add_zero] at h")
A_("  rw [← h]")
A_("  simp only [add_assoc]")
A_("  rfl")
A_("")
A_("/-- **Softmax at the true label ≤ 1/2**, from misclassification alone:")
A_("    `exp z_lbl ≤ exp z_c` and both terms sit inside the positive sum. -/")
A_("theorem sm_lbl_le_half : softmax 10 zd lblD ≤ 1 / 2 := by")
A_("  have hee : Real.exp (zd lblD) ≤ Real.exp (zd cD) := Real.exp_le_exp.mpr hz_lbl_le")
A_("  have hSpos : 0 < ∑ k, Real.exp (zd k) :=")
A_("    Finset.sum_pos (fun k _ => Real.exp_pos _) ⟨lblD, Finset.mem_univ _⟩")
A_("  have hpair : Real.exp (zd lblD) + Real.exp (zd cD) ≤ ∑ k, Real.exp (zd k) := by")
A_("    have hsub := Finset.sum_le_sum_of_subset_of_nonneg")
A_("      (Finset.subset_univ ({lblD, cD} : Finset (Fin 10)))")
A_("      (fun k _ _ => (Real.exp_pos (zd k)).le)")
A_("    rwa [Finset.sum_pair (by decide : lblD ≠ cD)] at hsub")
A_("  show Real.exp (zd lblD) / (∑ k, Real.exp (zd k)) ≤ 1 / 2")
A_("  rw [div_le_iff₀ hSpos]")
A_("  nlinarith [Real.exp_pos (zd lblD)]")
A_("")
A_("-- ── the gradient: an outer product `xᵢ·c_k`, `c` a rational image of the cotangent ──")
A_("")
A_("/-- The loss cotangent at the logits, `softmax − onehot`. -/")
A_("noncomputable def qd : Vec 10 := fun j => softmax 10 zd j - oneHot 10 lblD j")
A_("")
A_("/-- The backward through both frozen masks: `mask⁰ ⊙ W₁(mask¹ ⊙ W₂ q)`. -/")
A_("noncomputable def backC (q : Vec 10) : Vec 8 := fun k =>")
A_("  (if hpre0d k > 0 then (1:ℝ) else 0) *")
A_("    ∑ l, W1V k l * ((if hpre1d l > 0 then (1:ℝ) else 0) * ∑ j, W2V l j * q j)")
A_("")
A_("/-- `∂L/∂W₀[i,k] = xᵢ·c_k` at the trained weights: `mlp_input_loss_gradAt`, the two masks")
A_("    read off the exact pre-activations. -/")
A_("theorem grad_key (i : Fin 784) (k : Fin 8) :")
A_(f"    {GRAD} (finProdFinEquiv (i, k)) =")
A_("      xd i * backC qd k := by")
A_("  unfold mlpInputLoss")
A_("  rw [mlp_input_loss_gradAt b8 W1V b8 W2V b10 xd lblD (Mat.flatten W0V)")
A_("    (by rw [Mat.unflatten_flatten]; exact pre0_ne)")
A_("    (by rw [Mat.unflatten_flatten]; exact pre1_ne) i k]")
A_("  simp only [Mat.unflatten_flatten, pre0_eq, pre1_eq]")
A_("  rfl")
A_("")
W1L = ", ".join(f"w1c{m}Z" for m in range(H)); W2L = ", ".join(f"w2c{c}Z" for c in range(K))
for k in range(H):
    if M0[k]:
        A_(f"theorem backC_{k} (q : Vec 10) : backC q {k} =")
        A_("    " + lin(Gq[k], lambda j: f"q {j}") + " := by")
        A_(f"  simp only [backC, hpre0d, hpre1d, W1V, W2V, w1Z, w2Z, {W1L},")
        A_(f"    {W2L}, Fin.sum_univ_succ, Fin.sum_univ_zero]")
        A_("  norm_num")
        A_("  ring")
    else:
        A_(f"theorem backC_{k} (q : Vec 10) : backC q {k} = 0 := by")
        A_("  simp only [backC, hpre0d]")
        A_("  norm_num")
    A_("")
for j in range(K):
    A_(f"theorem qd_{j} : qd {j} = softmax 10 zd {j} - {1 if j == LBL else 0} := by")
    A_("  simp [qd, oneHot, lblD]")
A_("")
QDS = ", ".join(f"qd_{j}" for j in range(K))
POS = ", ".join(f"sm_pos {j}" for j in range(K))
A_("/-- The eight per-unit bounds `U_k = max_j |G_kj − G_k,lbl|`. -/")
A_("noncomputable def cotUb : Vec 8 := ![" + ", ".join(frac(u) for u in U) + "]")
A_("")
for k in range(H):
    A_(f"theorem cot_abs_le_{k} : |backC qd {k}| ≤ {frac(U[k])} := by")
    if M0[k]:
        A_(f"  rw [backC_{k}, abs_le]")
        A_(f"  simp only [{QDS}]")
        A_(f"  constructor <;> linarith [sm_sum_ten, {POS}]")
    else:
        A_(f"  rw [backC_{k}]")
        A_("  norm_num")
    A_("")
A_("/-- `|c_k| ≤ U_k`: `c_k` is a convex combination of the differences `G_kj − G_k,lbl`. -/")
A_("theorem cot_abs_le (k : Fin 8) : |backC qd k| ≤ cotUb k := by")
A_("  fin_cases k")
A_("  exacts [" + ", ".join(f"cot_abs_le_{k}" for k in range(H)) + "]")
A_("")
A_(f"/-- Row {KS} is sign-definite: `{'c' if SGN > 0 else '−c'}_{KS} ≥ {Lq.numerator}/{Lq.denominator}·(1 − p_lbl) ≥ {(Lq / 2).numerator}/{(Lq / 2).denominator}`. -/")
A_(f"theorem cot_lower : {frac(Lq / 2)} ≤ {'' if SGN > 0 else '-'}backC qd {KS} := by")
A_(f"  rw [backC_{KS}]")
A_(f"  simp only [{QDS}]")
A_("  have hhalf : " + P(LBL) + " ≤ 1 / 2 := sm_lbl_le_half")
A_(f"  linarith [sm_sum_ten, {POS}]")
A_("")
A_(f"theorem cot_sq_lower : {frac((Lq / 2) ** 2)} ≤ ∑ k, backC qd k ^ 2 := by")
A_(f"  have hterm : {frac((Lq / 2) ** 2)} ≤ backC qd {KS} ^ 2 := by")
A_("    nlinarith [cot_lower]")
A_("  exact hterm.trans (Finset.single_le_sum (f := fun k => backC qd k ^ 2)")
A_(f"    (fun k _ => sq_nonneg _) (Finset.mem_univ ({KS} : Fin 8)))")
A_("")
A_("-- ── the gradient sums, bracketed by exact rationals ──")
A_("")
A_(f"/-- `Σ|∇L| ≤ (Σxᵢ)·ΣU_k`, exact. -/")
A_("theorem gradL1_le :")
A_(f"    (∑ idx, |{GRAD} idx|) ≤ {frac(S1UB)} := by")
A_(f"  rw [← finProdFinEquiv.sum_comp (fun idx => |{GRAD} idx|)]")
A_("  rw [Fintype.sum_prod_type]")
A_("  simp_rw [grad_key]")
A_("  calc (∑ i : Fin 784, ∑ k : Fin 8, |xd i * backC qd k|)")
A_("      = ∑ i : Fin 784, xd i * ∑ k : Fin 8, |backC qd k| := by")
A_("        refine Finset.sum_congr rfl fun i _ => ?_")
A_("        rw [Finset.mul_sum]")
A_("        refine Finset.sum_congr rfl fun k _ => ?_")
A_("        rw [abs_mul, abs_of_nonneg (xd_nonneg i)]")
A_("    _ ≤ ∑ i : Fin 784, xd i * ∑ k : Fin 8, cotUb k :=")
A_("        Finset.sum_le_sum fun i _ =>")
A_("          mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun k _ => cot_abs_le k) (xd_nonneg i)")
A_("    _ = (∑ i, xd i) * ∑ k : Fin 8, cotUb k := by")
A_("        rw [Finset.sum_mul]")
A_(f"    _ = {frac(S1UB)} := by")
A_("        rw [xd_sum]")
A_("        simp only [cotUb, Fin.sum_univ_succ, Fin.sum_univ_zero]")
A_("        norm_num")
A_("")
A_(f"/-- `Σ(∇L)² ≥ (Σxᵢ²)·({Lq.numerator}/{Lq.denominator})²/4`, exact. -/")
A_("theorem gradSq_lower :")
A_(f"    {frac(S2LB)} ≤ ∑ idx, {GRAD} idx ^ 2 := by")
A_(f"  rw [← finProdFinEquiv.sum_comp (fun idx => {GRAD} idx ^ 2)]")
A_("  rw [Fintype.sum_prod_type]")
A_("  simp_rw [grad_key]")
A_("  have hfac : (∑ i : Fin 784, ∑ k : Fin 8, (xd i * backC qd k) ^ 2) =")
A_("      (∑ i, xd i ^ 2) * ∑ k : Fin 8, backC qd k ^ 2 := by")
A_("    rw [Finset.sum_mul]")
A_("    refine Finset.sum_congr rfl fun i _ => ?_")
A_("    rw [Finset.mul_sum]")
A_("    refine Finset.sum_congr rfl fun k _ => ?_")
A_("    ring")
A_("  rw [hfac, xd_sq_sum]")
A_("  calc " + f"{frac(S2LB)} = {frac(Sq)} * {frac((Lq / 2) ** 2)} := by norm_num")
A_(f"    _ ≤ {frac(Sq)} * ∑ k : Fin 8, backC qd k ^ 2 :=")
A_("        mul_le_mul_of_nonneg_left cot_sq_lower (by norm_num)")
A_("")
A_("-- ── the hypotheses of `mlp_input_sgd_descends` ──")
A_("")
for nm, n, B, lists in (("W1V", H, W1B, [f"w1c{m}Z" for m in range(H)]),
                        ("W2V", K, W2B, [f"w2c{c}Z" for c in range(K)])):
    A_(f"theorem h{nm} : ∀ k m, |{nm} k m| ≤ {frac(Fraction(B, WD))} := by")
    A_("  intro k m")
    A_(f"  have key : ∀ xs : List ℤ, (∀ z ∈ xs, {zlit(-B)} ≤ z ∧ z ≤ {B}) →")
    A_(f"      |(xs.getD k 0 : ℝ) / 128| ≤ {frac(Fraction(B, WD))} := by")
    A_("    intro xs hxs")
    A_(f"    have h := getD_of_forall xs (fun z => {zlit(-B)} ≤ z ∧ z ≤ {B})")
    A_("      (by decide) hxs k")
    A_(f"    have h1 : (({zlit(-B)} : ℤ) : ℝ) ≤ (xs.getD k 0 : ℝ) := by exact_mod_cast h.1")
    A_(f"    have h2 : (xs.getD k 0 : ℝ) ≤ {B} := by exact_mod_cast h.2")
    A_("    rw [abs_le]")
    A_("    constructor")
    A_("    · rw [le_div_iff₀ (by norm_num)]")
    A_("      push_cast at h1")
    A_("      linarith")
    A_("    · rw [div_le_iff₀ (by norm_num)]")
    A_("      linarith")
    A_("  fin_cases m")
    for nm_l in lists:
        A_(f"  · exact key {nm_l} (by decide +kernel)")
    A_("")
RAD = f"stepRadius ({LOSS}) (Mat.flatten W0V) {FR_LR} 0"
A_(f"/-- The step's `ℓ1` radius `lr·Σ|∇L|` is at most `lr·S1ub`. -/")
A_("theorem radius_le :")
A_(f"    {RAD} ≤ {frac(R)} := by")
A_("  simp only [stepRadius, mul_zero, add_zero]")
A_("  have h := gradL1_le")
A_("  linarith")
A_("")
A_("theorem radius_nonneg :")
A_(f"    0 ≤ {RAD} := by")
A_("  simp only [stepRadius, mul_zero, add_zero]")
A_("  exact mul_nonneg (by norm_num) (Finset.sum_nonneg fun _ _ => abs_nonneg _)")
A_("")
A_("-- ── the capstones ──")
A_("")
A_("/-- **Descent at TRAINED weights, through both ReLU layers.** One SGD step with the exact")
A_(f"    gradient (lr = 2⁻{EXP}) on the input weights of the trained `784 → 8 → 8 → 10` MLP, at the")
A_("    misclassified test image, decreases the cross-entropy loss by at least `lr·‖∇L‖₂²/2`:")
A_("    every hypothesis of `mlp_input_sgd_descends` discharged at `η = 0`. -/")
A_("theorem trained_mlp_input_sgd_descends_concrete :")
A_(f"    ({LOSS}) (Mat.flatten W0V -")
A_(f"        {FR_LR} • {GRAD}) ≤")
A_(f"      ({LOSS}) (Mat.flatten W0V) -")
A_(f"        {FR_LR} * (∑ idx, {GRAD} idx ^ 2) / 2 := by")
A_("  have hR := radius_le")
A_("  have hR0 := radius_nonneg")
A_("  have hS2 := gradSq_lower")
A_(f"  refine mlp_input_sgd_descends W0V b8 W1V b8 W2V b10 xd lblD _ (lr := {FR_LR}) (η := 0)")
A_(f"    (a := 1) (w₁ := {frac(w1b)}) (w₂ := {frac(w2b)}) (by norm_num) xd_le_one (by norm_num) hW1V")
A_("    (by norm_num) hW2V (by norm_num) le_rfl (fun idx => by simp) ?_ ?_ ?_ ?_ ?_")
A_("  · -- hmargin0: the step cannot flip a first-layer ReLU")
A_("    intro j")
A_(f"    refine lt_of_le_of_lt (b := {frac(R)}) (by linarith) ?_")
A_("    rw [pre0_eq]")
A_("    fin_cases j <;> (rw [lt_abs]; norm_num [hpre0d])")
A_("  · -- hmargin1: nor a second-layer one")
A_("    intro l")
A_(f"    refine lt_of_le_of_lt (b := {frac(w1b * R)}) (by linarith) ?_")
A_("    rw [pre1_eq]")
A_("    fin_cases l <;> (rw [lt_abs]; norm_num [hpre1d])")
A_("  · -- hsmall")
A_("    push_cast")
A_("    linarith")
A_("  · -- h1: the oracle is exact")
A_("    have hS20 : (0:ℝ) ≤ " + f"∑ idx, {GRAD} idx ^ 2 :=")
A_("      Finset.sum_nonneg fun _ _ => sq_nonneg _")
A_("    rw [mul_zero, zero_mul]")
A_("    positivity")
A_("  · -- h2: the segment-Lipschitz term against the guaranteed drop")
A_(f"    set rad := {RAD} with hrad")
A_(f"    set S2 := ∑ idx, {GRAD} idx ^ 2 with hS2def")
A_("    push_cast")
A_(f"    have hT : (0:ℝ) < 1 - 2 * ({frac(w2b)} * (8 * ({frac(w1b)} * (1 * rad)))) := by linarith")
A_("    rw [div_mul_eq_mul_div, div_le_iff₀ hT]")
A_(f"    have hsq : rad ^ 2 ≤ {frac(R)} ^ 2 := pow_le_pow_left₀ hR0 hR 2")
A_(f"    have hprod : {FR_LR} * {frac(S2LB)} / 4 * (1 - 2 * ({frac(w2b)} * (8 * ({frac(w1b)} * (1 * {frac(R)})))))")
A_(f"        ≤ {FR_LR} * S2 / 4 * (1 - 2 * ({frac(w2b)} * (8 * ({frac(w1b)} * (1 * rad))))) :=")
A_("      mul_le_mul (by linarith) (by linarith) (by norm_num) (by linarith)")
A_("    nlinarith [hsq, hprod]")
A_("")
A_("/-- **The drop is strictly positive**: the image is misclassified and one hidden unit's")
A_("    backward coefficient is sign-definite, so `Σ∇² > 0` and the loss strictly decreases. -/")
A_("theorem trained_mlp_input_sgd_strictly_descends :")
A_(f"    ({LOSS}) (Mat.flatten W0V -")
A_(f"        {FR_LR} • {GRAD}) <")
A_(f"      ({LOSS}) (Mat.flatten W0V) := by")
A_("  have h := trained_mlp_input_sgd_descends_concrete")
A_("  have h2 := gradSq_lower")
A_("  linarith [h, h2]")
A_("")
A_("end TrainedMlpDescent")
A_("end Proofs")

with open(OUT, "w") as f:
    f.write(modularize("\n".join(L) + "\n"))
print(f"wrote {OUT}: {len(L)} lines")
