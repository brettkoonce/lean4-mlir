"""The MLP chapter's trained-weight VJP witness.

Emits LeanMlir/Proofs/Training/Trained/MlpWitness.lean: Chapter 2's own architecture,
dense 784->8 -> relu -> dense 8->8 -> relu -> dense 8->10 (`Proofs.mlpForward` at
d = 784, 8, 8, 10), bias-free, trained here with numpy and rationalized to /128, at a real
28x28 MNIST test image. The file instantiates the chapter's conditional VJP
`Proofs.mlpHasVJPAt` there: both ReLU smoothness hypotheses are discharged from the exact
rational pre-activations, and one Jacobian entry is evaluated in closed form through the two
ReLU masks, which seals the backward as not the zero map.

Data conventions (`Mat` is input x output, so a weight matrix is stored by output column):

    pixels          k/255             xZ      : List Z, 784 numerators
    weights         k/128             w0c{k}Z : 784 numerators, one list per hidden unit k
                                      w1c{m}Z, w2c{c}Z : 8 numerators each
    layer-1 preact  D0_k/(255*128)    D0_k = dotZ xZ w0c{k}Z, one kernel evaluation over 784 terms
    layer-2 preact  D1_m/(32640*128)  D1_m = dotZ a0Z w1c{m}Z, a0Z = max(D0, 0)

The witness image is the first test image the rationalized net classifies correctly at which
every one of the sixteen hidden pre-activations is nonzero. Every number written into the Lean
file is asserted here first with exact integers / `Fraction`s.

Not in CI: it needs MNIST in data/, which CI does not have. Regenerate by hand and confirm the
committed Lean comes back byte-identical.
"""
import numpy as np, os, sys
from fractions import Fraction
from _gencheck import modularize, module_header  # noqa: E402  (module-system header on what we write)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
from _mnist_io import mnist  # noqa: E402
from _leanlit import frac, zlist, zlit  # noqa: E402
OUT = os.path.join(ROOT, "LeanMlir/Proofs/Training/Trained/MlpWitness.lean")
DIM, H, K = 784, 8, 10
PX, WD = 255, 128                       # pixel and weight denominators
C0, C1 = PX * WD, PX * WD * WD          # layer-1 / layer-2 pre-activation denominators

Xtr_raw, ytr = mnist("train", flat=True); Xte_raw, yte = mnist("test", flat=True)
Ste = Xte_raw.astype(np.int64)
Xtr = Xtr_raw / 255.0; Xte = Xte_raw / 255.0

# ── train: bias-free 784 -> 8 -> 8 -> 10, He init, minibatch SGD on softmax cross-entropy ──
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


def fwd_float(X, A, B, C):
    return np.maximum(np.maximum(X @ A, 0) @ B, 0) @ C


acc = (np.argmax(fwd_float(Xte, W0, W1, W2), 1) == yte).mean()

# exact integer forward on the whole test set (object ints: no overflow question)
D0all = Ste @ W0q                                           # x (255*128)
A0all = np.maximum(D0all, 0)
D1all = A0all @ W1q                                         # x (255*128*128)
A1all = np.maximum(D1all, 0)
D2all = A1all @ W2q                                         # x (255*128^3)
assert abs(D2all).max() < 2 ** 62
accq = (D2all.argmax(1) == yte).mean()
print(f"mlp 784->8->8->10: acc {acc:.4f} q-acc {accq:.4f} "
      f"|W0q|max {abs(W0q).max()} |W1q|max {abs(W1q).max()} |W2q|max {abs(W2q).max()}")

# ── the witness image: first correctly classified test image off every ReLU kink ──
srt = np.sort(D2all, 1)
ok = (D2all.argmax(1) == yte) & (srt[:, -1] > srt[:, -2]) \
    & (D0all != 0).all(1) & (D1all != 0).all(1)
IDX = int(np.nonzero(ok)[0][0])
s = [int(v) for v in Ste[IDX]]
LBL = int(yte[IDX])
w0 = [[int(W0q[i, k]) for i in range(DIM)] for k in range(H)]      # columns
w1 = [[int(W1q[k, m]) for k in range(H)] for m in range(H)]
w2 = [[int(W2q[m, c]) for m in range(H)] for c in range(K)]
dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))
D0 = [dot(s, w0[k]) for k in range(H)]
A0 = [max(v, 0) for v in D0]
D1 = [dot(A0, w1[m]) for m in range(H)]
A1 = [max(v, 0) for v in D1]
D2 = [dot(A1, w2[c]) for c in range(K)]
assert all(v != 0 for v in D0) and all(v != 0 for v in D1)
assert max(range(K), key=lambda c: D2[c]) == LBL and sorted(D2)[-1] > sorted(D2)[-2]
pre0 = [Fraction(v, C0) for v in D0]
pre1 = [Fraction(v, C1) for v in D1]
M0 = [1 if v > 0 else 0 for v in D0]
M1 = [1 if v > 0 else 0 for v in D1]
print(f"image #{IDX}: label {LBL}; layer-1 preacts {[round(float(v), 3) for v in pre0]}")
print(f"  layer-2 preacts {[round(float(v), 3) for v in pre1]}")
print(f"  logits {[round(v / (C1 * WD), 3) for v in D2]}")
print(f"  masks {M0} {M1}; min |preact| {float(min(map(abs, pre0))):.4f} / {float(min(map(abs, pre1))):.4f}")


# ── the sealed Jacobian entry: the largest |d logit_LBL / d x_i| ──
def jac(i, c):
    return Fraction(sum(int(W0q[i, k]) * M0[k] * sum(int(W1q[k, m]) * M1[m] * int(W2q[m, c])
                                                     for m in range(H)) for k in range(H)), WD ** 3)


I0 = max(range(DIM), key=lambda i: (abs(jac(i, LBL)), -i))
PV = jac(I0, LBL)
assert PV != 0
ROW = [int(W0q[I0, k]) for k in range(H)]
print(f"pdiv entry: d logit_{LBL} / d x_{I0} = {PV} = {float(PV):.4f}")
n_on0, n_on1 = sum(M0), sum(M1)

L = []
A_ = L.append
A_(module_header(["LeanMlir.Proofs.Foundation.MLP", "LeanMlir.Proofs.Foundation.ListDot"],
                 ["LeanMlir.Proofs.Training.JacobianSeal"]))
A_("")
A_("/-! # Trained-weight whole-network VJP witness (MLP chapter)")
A_("")
A_("The net is Chapter 2's own architecture at width 8, on the 28×28 image:")
A_("`dense 784→8 → relu → dense 8→8 → relu → dense 8→10`, that is `mlpForward` at")
A_("`d = 784, 8, 8, 10`, bias-free, trained on MNIST and rationalized to `k/128` (the")
A_("generator prints its test accuracy). The input is MNIST test image")
A_(f"#{IDX} (a {LBL}, which the net classifies correctly), all 784 pixels, `k/255`. Nothing is")
A_("pooled or cropped; only the hidden width differs from the chapter's 512.")
A_("")
A_("The file instantiates the chapter's conditional VJP `mlpHasVJPAt` at that point. Its two")
A_("hypotheses, that no hidden pre-activation of either ReLU layer is zero, are discharged")
A_("from the exact rational pre-activations:")
A_("")
A_("* layer 1 (`preact_eq`, `preact_ne`): eight 784-term sums, each one kernel evaluation")
A_("  over `List ℤ` numerators (`Foundation/ListDot`);")
A_(f"  {n_on0} units strictly on, {H - n_on0} strictly off;")
A_("* layer 2 (`preact1_eq`, `preact1_ne`): eight 8-term sums over the layer-1 activations;")
A_(f"  {n_on1} strictly on, {H - n_on1} strictly off.")
A_("")
A_("Levels:")
A_("")
A_("* level 1 — `trainedMlpHasVJPAt` (+ `trainedMlpHasVJPAt_correct`): `mlpHasVJPAt` at the")
A_("  trained weights and the image; its backward is the `pdiv`-contracted Jacobian there;")
A_("* level 3 — `trainedMlp_backward_nontrivial`: the backward is not the zero map, via the")
A_("  Jacobian entry `pdiv_fwd_val`, read off the backward itself (`backward_eq`, `pdiv_fwd`):")
A_(f"  `∂ logit_{LBL} / ∂ x_{I0} = {PV.numerator}/{PV.denominator} ≈ {float(PV):.3f}`;")
A_("* `trainedMlp_jacobian_nonzero` / `trainedMlp_not_constant`: the `fderiv` forms.")
A_("")
A_("`Mat` is input×output, so each weight matrix is stored by output column. Generated by")
A_("`scripts/certs/trained_mlp_witness.py`; weights and input are DATA. -/")
A_("")
A_("namespace Proofs")
A_("namespace TrainedMlp")
A_("")
A_("open scoped BigOperators")
A_("")
A_("-- ── the data ──")
A_("")
A_(f"/-- MNIST test image #{IDX} (digit {LBL}), the 784 pixels as integers `0..255`. -/")
A_("def xZ : List ℤ := " + zlist(s))
A_("")
A_("theorem xZ_len : xZ.length = 784 := by decide +kernel")
A_("")
for k in range(H):
    A_(f"/-- Hidden unit {k}'s trained input weights, numerators over 128. -/")
    A_(f"def w0c{k}Z : List ℤ := " + zlist(w0[k]))
    A_("")
    A_(f"theorem w0c{k}Z_len : w0c{k}Z.length = 784 := by decide +kernel")
    A_("")
for m in range(H):
    A_(f"/-- Second-layer unit {m}'s trained weights over the first hidden layer, numerators over 128. -/")
    A_(f"def w1c{m}Z : List ℤ := " + zlist(w1[m]))
    A_("")
    A_(f"theorem w1c{m}Z_len : w1c{m}Z.length = 8 := by decide +kernel")
    A_("")
for c in range(K):
    A_(f"/-- Class {c}'s trained weights over the second hidden layer, numerators over 128. -/")
    A_(f"def w2c{c}Z : List ℤ := " + zlist(w2[c]))
    A_("")
for nm, n, doc in (("w0", H, "The eight first-layer weight columns by hidden unit."),
                   ("w1", H, "The eight second-layer weight columns by hidden unit."),
                   ("w2", K, "The ten output weight columns by class.")):
    A_(f"/-- {doc} -/")
    A_(f"def {nm}Z : Fin {n} → List ℤ")
    for k in range(n):
        A_(f"  | {k} => {nm}c{k}Z")
    A_("")
A_("/-- First-layer weights, `784×8`, entries `k/128`. -/")
A_("noncomputable def W0V : Mat 784 8 := fun i k => ((w0Z k).getD i 0 : ℝ) / 128")
A_("")
A_("/-- Second-layer weights, `8×8`, entries `k/128`. -/")
A_("noncomputable def W1V : Mat 8 8 := fun k m => ((w1Z m).getD k 0 : ℝ) / 128")
A_("")
A_("/-- Output weights, `8×10`, entries `k/128`. -/")
A_("noncomputable def W2V : Mat 8 10 := fun m c => ((w2Z c).getD m 0 : ℝ) / 128")
A_("")
A_("noncomputable def b8 : Vec 8 := fun _ => 0")
A_("noncomputable def b10 : Vec 10 := fun _ => 0")
A_("")
A_(f"/-- MNIST test image #{IDX} (digit {LBL}), pixels `k/255`. -/")
A_("noncomputable def xtV : Vec 784 := fun i => (xZ.getD i 0 : ℝ) / 255")
A_("")
A_("/-- The trained forward: the chapter's `mlpForward` at `784 → 8 → 8 → 10`. -/")
A_("noncomputable def fwd : Vec 784 → Vec 10 := mlpForward W0V b8 W1V b8 W2V b10")
A_("")
A_("-- ── layer 1: the eight 784-term pre-activations, one kernel evaluation each ──")
A_("")
for k in range(H):
    A_(f"theorem d0c{k} : dotZ xZ w0c{k}Z = {zlit(D0[k])} := by decide +kernel")
A_("")
A_(f"/-- The exact first-layer pre-activations at the image ({n_on0} positive, {H - n_on0} negative). -/")
A_("noncomputable def hpre0Vals : Vec 8 := ![" + ", ".join(frac(v) for v in pre0) + "]")
A_("")
for k in range(H):
    A_(f"theorem preact_{k} : dense W0V b8 xtV {k} = {frac(pre0[k])} := by")
    A_(f"  show (∑ i : Fin 784, (xZ.getD i 0 : ℝ) / 255 * ((w0c{k}Z.getD i 0 : ℝ) / 128)) + 0 = _")
    A_(f"  rw [sum_getD_div xZ_len w0c{k}Z_len d0c{k} 255 128]")
    A_("  norm_num")
    A_("")
A_("/-- The first-layer pre-activations at the image, exactly. -/")
A_("theorem preact_eq (k : Fin 8) : dense W0V b8 xtV k = hpre0Vals k := by")
A_("  fin_cases k")
A_("  exacts [" + ", ".join(f"preact_{k}" for k in range(H)) + "]")
A_("")
A_("/-- **Smoothness at the first ReLU**: no first-layer unit sits on the kink at the image. -/")
A_("theorem preact_ne : ∀ k, dense W0V b8 xtV k ≠ 0 := by")
A_("  intro k")
A_("  rw [preact_eq]")
A_("  fin_cases k <;> norm_num [hpre0Vals]")
A_("")
A_("-- ── layer 2: the eight 8-term pre-activations over the layer-1 activations ──")
A_("")
A_(f"/-- The first hidden layer's activations at the image, numerators over {C0} (= 255·128). -/")
A_("def a0Z : List ℤ := " + zlist(A0))
A_("")
A_("theorem a0Z_len : a0Z.length = 8 := by decide +kernel")
A_("")
A_("/-- The first hidden layer's output at the image, exactly. -/")
A_("theorem relu_preact_eq :")
A_(f"    relu 8 (dense W0V b8 xtV) = fun k : Fin 8 => (a0Z.getD k 0 : ℝ) / {C0} := by")
A_("  funext k")
A_("  show (if dense W0V b8 xtV k > 0 then dense W0V b8 xtV k else 0) = _")
A_("  rw [preact_eq]")
A_("  fin_cases k <;> norm_num [hpre0Vals, a0Z]")
A_("")
for m in range(H):
    A_(f"theorem d1c{m} : dotZ a0Z w1c{m}Z = {zlit(D1[m])} := by decide +kernel")
A_("")
A_(f"/-- The exact second-layer pre-activations at the image ({n_on1} positive, {H - n_on1} negative). -/")
A_("noncomputable def hpre1Vals : Vec 8 := ![" + ", ".join(frac(v) for v in pre1) + "]")
A_("")
for m in range(H):
    A_(f"theorem preact1_{m} : dense W1V b8 (relu 8 (dense W0V b8 xtV)) {m} = {frac(pre1[m])} := by")
    A_("  rw [relu_preact_eq]")
    A_(f"  show (∑ k : Fin 8, (a0Z.getD k 0 : ℝ) / {C0} * ((w1c{m}Z.getD k 0 : ℝ) / 128)) + 0 = _")
    A_(f"  rw [sum_getD_div a0Z_len w1c{m}Z_len d1c{m} {C0} 128]")
    A_("  norm_num")
    A_("")
A_("/-- The second-layer pre-activations at the image, exactly. -/")
A_("theorem preact1_eq (m : Fin 8) :")
A_("    dense W1V b8 (relu 8 (dense W0V b8 xtV)) m = hpre1Vals m := by")
A_("  fin_cases m")
A_("  exacts [" + ", ".join(f"preact1_{m}" for m in range(H)) + "]")
A_("")
A_("/-- **Smoothness at the second ReLU**: no second-layer unit sits on the kink at the image. -/")
A_("theorem preact1_ne : ∀ m, dense W1V b8 (relu 8 (dense W0V b8 xtV)) m ≠ 0 := by")
A_("  intro m")
A_("  rw [preact1_eq]")
A_("  fin_cases m <;> norm_num [hpre1Vals]")
A_("")
A_("-- ── level 1: the chapter's conditional VJP, instantiated ──")
A_("")
A_("/-- **Level 1: the trained-weight whole-net VJP witness** — the chapter's `mlpHasVJPAt` at")
A_("    the trained `784 → 8 → 8 → 10` weights and the real image, both ReLU smoothness")
A_("    hypotheses discharged (`preact_ne`, `preact1_ne`). -/")
A_("noncomputable def trainedMlpHasVJPAt : HasVJPAt fwd xtV :=")
A_("  mlpHasVJPAt W0V b8 W1V b8 W2V b10 xtV preact_ne preact1_ne")
A_("")
A_("/-- The backward of `trainedMlpHasVJPAt` is the `pdiv`-contracted Jacobian of the trained")
A_("    forward at the image: `mlpHasVJPAt_correct`, instantiated. -/")
A_("theorem trainedMlpHasVJPAt_correct (dy : Vec 10) (i : Fin 784) :")
A_("    trainedMlpHasVJPAt.backward dy i = ∑ j : Fin 10, pdiv fwd xtV i j * dy j :=")
A_("  mlpHasVJPAt_correct W0V b8 W1V b8 W2V b10 xtV preact_ne preact1_ne dy i")
A_("")
A_("/-- The backward, unfolded: `W₂`, the second ReLU's mask, `W₁`, the first ReLU's mask, `W₀`. -/")
A_("theorem backward_eq (dy : Vec 10) :")
A_("    trainedMlpHasVJPAt.backward dy =")
A_("      Mat.mulVec W0V (fun k => if dense W0V b8 xtV k > 0 then")
A_("        Mat.mulVec W1V (fun m => if dense W1V b8 (relu 8 (dense W0V b8 xtV)) m > 0 then")
A_("          Mat.mulVec W2V dy m else 0) k else 0) := rfl")
A_("")
A_("/-- The whole-net Jacobian in closed form at the image, read off the proven backward at a")
A_("    basis cotangent: `Σ_k W₀[i,k]·mask⁰_k·Σ_m W₁[k,m]·mask¹_m·W₂[m,c]`. -/")
A_("theorem pdiv_fwd (i : Fin 784) (c : Fin 10) :")
A_("    pdiv fwd xtV i c =")
A_("      ∑ k, W0V i k * ((if hpre0Vals k > 0 then (1:ℝ) else 0) *")
A_("        ∑ m, W1V k m * ((if hpre1Vals m > 0 then (1:ℝ) else 0) * W2V m c)) := by")
A_("  have h := trainedMlpHasVJPAt_correct (basisVec c) i")
A_("  rw [backward_eq] at h")
A_("  simp only [basisVec_apply, mul_ite, mul_one, mul_zero, Finset.sum_ite_eq',")
A_("    Finset.mem_univ, ite_true, Mat.mulVec, preact_eq, preact1_eq] at h")
A_("  rw [← h]")
A_("  refine Finset.sum_congr rfl fun k _ => ?_")
A_("  split_ifs")
A_("  · rw [one_mul]")
A_("    congr 1")
A_("    refine Finset.sum_congr rfl fun m _ => ?_")
A_("    split_ifs <;> simp")
A_("  · simp")
A_("")
A_(f"-- ── level 3: one Jacobian entry, ∂ logit_{LBL} / ∂ x_{I0}, exactly ──")
A_("")
for k in range(H):
    A_(f"theorem w0c{k}Z_at : w0c{k}Z.getD {I0} 0 = {zlit(ROW[k])} := by decide +kernel")
A_("")
A_(f"/-- Row {I0} of `W0V`: pixel {I0}'s eight outgoing weights. -/")
A_(f"theorem W0V_row (k : Fin 8) : W0V {I0} k = (![" + ", ".join(frac(Fraction(v, WD)) for v in ROW)
   + "] : Vec 8) k := by")
A_("  fin_cases k")
for k in range(H):
    A_(f"  · show ((w0c{k}Z.getD {I0} 0 : ℤ) : ℝ) / 128 = _")
    A_(f"    rw [w0c{k}Z_at]")
    A_("    norm_num")
A_("")
A_(f"/-- The Jacobian entry `(∂ logit_{LBL} / ∂ x_{I0})` at the image, exactly. -/")
A_(f"theorem pdiv_fwd_val : pdiv fwd xtV {I0} {LBL} = {frac(PV)} := by")
A_("  rw [pdiv_fwd]")
A_("  simp only [W0V_row]")
A_("  simp [W1V, W2V, w1Z, w2Z, " + ", ".join(f"w1c{m}Z" for m in range(H)) + f", w2c{LBL}Z,")
A_("    hpre0Vals, hpre1Vals, Fin.sum_univ_succ]")
A_("  norm_num")
A_("")
A_(f"theorem pdiv_fwd_ne : pdiv fwd xtV {I0} {LBL} ≠ 0 := by")
A_("  rw [pdiv_fwd_val]")
A_("  norm_num")
A_("")
A_("/-- **Level 3: the trained-weight backward is not the zero map.** -/")
A_("theorem trainedMlp_backward_nontrivial :")
A_(f"    trainedMlpHasVJPAt.backward (basisVec {LBL}) {I0} ≠ 0 :=")
A_("  trainedMlpHasVJPAt.backward_ne_zero_of_pdiv_ne pdiv_fwd_ne")
A_("")
A_("/-- The `fderiv` form: the whole-net Jacobian at the image is nonzero. -/")
A_("theorem trainedMlp_jacobian_nonzero : fderiv ℝ fwd xtV ≠ 0 := by")
A_("  intro h")
A_("  apply pdiv_fwd_ne")
A_("  unfold pdiv")
A_("  rw [h]")
A_("  rfl")
A_("")
A_("/-- The trained network is not a constant function. -/")
A_("theorem trainedMlp_not_constant : ¬ (∀ u v : Vec 784, fwd u = fwd v) := by")
A_("  intro h")
A_("  apply trainedMlp_jacobian_nonzero")
A_("  have hc : fwd = fun _ => fwd xtV := funext fun u => h u xtV")
A_("  rw [hc]")
A_("  exact (hasFDerivAt_const (fwd xtV) xtV).fderiv")
A_("")
A_("end TrainedMlp")
A_("end Proofs")

with open(OUT, "w") as f:
    f.write(modularize("\n".join(L) + "\n"))
print(f"wrote {OUT}: {len(L)} lines")
