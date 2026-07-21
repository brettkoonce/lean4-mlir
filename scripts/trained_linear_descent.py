"""Descent at TRAINED weights (planning/post_audit_roadmap.md §3).

Emits LeanMlir/Proofs/Training/TrainedLinearDescent.lean: one binary32 SGD step on a
TRAINED, /128-rationalized bias-free 49->10 pooled-MNIST linear classifier
provably decreases the real cross-entropy loss -- retiring the "the only
concrete descent instance is the degenerate W=0 net" caveat
(binary32_linear_sgd_descends_concrete).

The trick that makes the descent window rational-checkable with ZERO
exponential evaluations: pick a MISCLASSIFIED test sample. Then
z_lbl <= z_pred is an exact rational inequality, exp-monotonicity gives
softmax_lbl <= 1/2, and the two gradient sums the window needs are bracketed by

    S1 := sum |grad|  = (sum_i x_i) * 2*(1 - sm_lbl)      <= 2*sum_i x_i =: A
    S2 := sum grad^2 >= (sum_i x_i^2) * (1 - sm_lbl)^2   >= B/4,  B := sum x_i^2

both sides exact rationals. The float-forward drift hδ comes from the proven
dense_close_fresh budget (denseErr at b=0, e=0), evaluated exactly per class
column by norm_num (incl. the exact (1+2^-24)^51 power); the exp(2δ)-1 term
inside cotErr is bounded by the repo's γ-form exp_sub_one_le. Window margins
are asserted here with exact Fractions in EXACTLY the bound-forms the Lean
proof discharges.
"""
import numpy as np, struct
from fractions import Fraction

D = "/home/skoonce/lean/klawd_max_power/lean4-jax/data/"
OUT = "/home/skoonce/lean/klawd_max_power/lean4-jax/LeanMlir/Proofs/Training/TrainedLinearDescent.lean"
K, DIM = 10, 49
LR = Fraction(1, 8192)

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

def pool_sums(X):
    return X.reshape(-1, 7, 4, 7, 4).astype(np.int64).sum(axis=(2, 4)).reshape(-1, 49)

Str = pool_sums(Xtr_raw); Ste = pool_sums(Xte_raw)
Xtr = Str / 4080.0; Xte = Ste / 4080.0

rng = np.random.default_rng(0)
W = rng.normal(0, np.sqrt(1.0 / DIM), (K, DIM))
lr_train, bs = 0.15, 64
for ep in range(12):
    idx = rng.permutation(len(Xtr))
    for b in range(0, len(Xtr), bs):
        xb = Xtr[idx[b:b+bs]]; yb = ytr[idx[b:b+bs]]
        z = xb @ W.T; z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        g = p.copy(); g[np.arange(len(yb)), yb] -= 1; g /= len(yb)
        W -= lr_train * g.T @ xb
Wq = np.round(W * 128).astype(np.int64)
acc = (np.argmax(Xte @ W.T, 1) == yte).mean()
accq = (np.argmax(Xte @ (Wq / 128).T, 1) == yte).mean()
print(f"linear 49->10: acc {acc:.4f} q-acc {accq:.4f} |Wq|max {abs(Wq).max()}")

# ── the witness sample: first misclassified test image ──
IDX = 8
s = [int(v) for v in Ste[IDX]]
z = [Fraction(sum(int(Wq[j, d]) * s[d] for d in range(DIM)), 128 * 4080) for j in range(K)]
LBL = int(yte[IDX]); C = int(np.argmax([float(v) for v in z]))
assert LBL == 5 and C == 6 and z[LBL] < z[C], (LBL, C)
Sx = Fraction(sum(s), 4080)
A = 2 * Sx
B = Fraction(sum(v * v for v in s), 4080 * 4080)

# ── exact window constants ──
u = Fraction(1, 2 ** 24)
pow51 = (1 + u) ** 51 - 1
sig = [sum(abs(int(Wq[j, d])) * Fraction(s[d], 4080) for d in range(DIM)) / 128
       for j in range(K)]
d_raw = pow51 * max(sig)
DEL = Fraction(int(d_raw * 100000) + 1, 100000)          # δ0, nice upper grid
assert 2 * DEL < 1
E_raw = 2 * DEL / (1 - 2 * DEL)                          # ≥ exp(2δ)-1
EUB = Fraction(int(E_raw * 10 ** 7) + 1, 10 ** 7)
rho = (1 + u) ** 11 - 1
kap = rho / (1 - rho)
smE = u * (1 + kap) + kap + EUB
cot = u * (1 + smE) + smE
eta_raw = u * (1 + cot) + cot                            # mulErr u 1 1 0 cot
ETA = Fraction(int(eta_raw * 10 ** 6) + 1, 10 ** 6)

# window, in exactly the Lean bound-forms
Dub = LR * (A + 490 * ETA)
Tlb = 1 - 2 * Dub
hsmall_ok = 2 * Dub < 1
h1_ok = ETA * A <= (B / 4) / 4
h2_ok = Tlb > 0 and 2 * Dub ** 2 <= (LR * (B / 4) / 4) * Tlb
h2_margin = float((LR * (B / 4) / 4) * Tlb / (2 * Dub ** 2))
print(f"sample #{IDX}: lbl={LBL} pred={C} gap={float(z[C]-z[LBL]):.4f}")
print(f"A={A}={float(A):.4f} B={B}={float(B):.4f}")
print(f"δ0={DEL} E_ub={EUB} η_ub={ETA} lr={LR}")
print(f"hsmall={hsmall_ok} h1={h1_ok} h2={h2_ok} (h2 margin {h2_margin:.2f}x)")
assert hsmall_ok and h1_ok and h2_ok and h2_margin >= 2.0

def frac(q):
    q = Fraction(q)
    return f"(({q.numerator} : ℝ)/{q.denominator})" if q.denominator != 1 else f"({q.numerator} : ℝ)"

def row(vals, den):
    return "![" + ", ".join(f"(({int(v)} : ℝ)/{den})" for v in vals) + "]"

FR_LR, FR_DEL, FR_ETA, FR_A, FR_B = frac(LR), frac(DEL), frac(ETA), frac(A), frac(B)
GRAD = "gradAt (fun w => crossEntropy 10 (dense (Mat.unflatten w) bd xd) lblD)\n        (Mat.flatten Wd)"

L = []
A_ = L.append
A_("import LeanMlir.Proofs.Float.Binary32Instance")
A_("")
A_("/-! # Descent at TRAINED weights (post_audit_roadmap §3)")
A_("")
A_("`binary32_linear_sgd_descends_concrete` (the suite's only concrete descent")
A_("instance) holds at the degenerate `W = 0` net — a satisfiability witness.")
A_("This file retires that caveat: **one binary32 SGD step on a TRAINED,")
A_(f"/128-rationalized bias-free 49→10 pooled-MNIST linear classifier (test acc")
A_(f"{accq:.3f}) provably decreases the real cross-entropy loss**, at MNIST test")
A_(f"image #{IDX} with learning rate {LR}. The rounding model is the CONSTRUCTED")
A_("`rndP 23` grid (`binary32`), so the whole statement is axiom-free.")
A_("")
A_("The trick that closes the descent window with zero `exp` evaluations: the")
A_(f"witness sample is MISCLASSIFIED (true label {LBL}, predicted {C}), so")
A_(f"`z_{LBL} ≤ z_{C}` is an exact rational inequality (`hz_lbl_le`) and")
A_(f"exp-monotonicity alone gives `softmax_{LBL} ≤ 1/2` (`sm_lbl_le_half`) —")
A_("whence the two gradient sums the window needs are bracketed by rationals:")
A_("`Σ|∇| ≤ 2·Σxᵢ` (`gradL1_le`, softmax sums to 1) and `Σ∇² ≥ (Σxᵢ²)/4`")
A_("(`gradSq_lower`). The float-forward drift `hδ` is the proven")
A_("`dense_close_fresh` budget evaluated exactly per class column (norm_num")
A_("computes the exact `(1+2⁻²⁴)⁵¹`); the `exp(2δ)−1` term inside `cotErr` is")
A_("bounded by the γ-form `exp_sub_one_le`. Every hypothesis of")
A_("`linear_float_sgd_descends` is discharged — nothing assumed.")
A_("")
A_("Being misclassified also makes the guaranteed drop STRICTLY positive")
A_(f"(`trained_linear_sgd_strictly_descends`): the loss drops by ≥ lr·(Σxᵢ²)/8")
A_(f"≈ {float(LR * B / 8):.2e}. Generated by `scripts/trained_linear_descent.py`;")
A_("weights/input are DATA. -/")
A_("")
A_("namespace Proofs")
A_("namespace TrainedLinearDescent")
A_("")
A_(f"/-- Trained linear weights (input×class, entries `k/128`), test acc {accq:.3f}. -/")
A_("noncomputable def Wd : Mat 49 10 :=")
A_("  ![" + ",\n    ".join(row([Wq[j, i] for j in range(K)], 128) for i in range(DIM)) + "]")
A_("")
A_("noncomputable def bd : Vec 10 := fun _ => 0")
A_("")
A_(f"/-- MNIST test image #{IDX} (digit {LBL}), 4×4-pooled exact pixel sums. -/")
A_("noncomputable def xd : Vec 49 :=")
A_("  " + row(s, 4080))
A_("")
A_(f"def lblD : Fin 10 := {LBL}")
A_(f"def cD : Fin 10 := {C}")
A_("")
A_("theorem xd_nonneg : ∀ i, 0 ≤ xd i := by")
A_("  intro i")
A_("  fin_cases i <;> norm_num [xd]")
A_("")
A_("theorem xd_le_one : ∀ i, |xd i| ≤ 1 := by")
A_("  intro i")
A_("  fin_cases i <;> norm_num [xd]")
A_("")
A_(f"/-- `Σᵢ xᵢ` exactly (all coordinates nonneg). -/")
A_(f"theorem xd_sum : (∑ i, xd i) = {frac(Sx)} := by")
A_("  simp [xd, Fin.sum_univ_succ]")
A_("  norm_num")
A_("")
A_(f"theorem xd_sq_sum : (∑ i, xd i ^ 2) = {FR_B} := by")
A_("  simp [xd, Fin.sum_univ_succ]")
A_("  norm_num")
A_("")
A_(f"/-- The witness is misclassified: logit {LBL} ≤ logit {C}, exactly. -/")
A_("theorem hz_lbl_le : dense Wd bd xd lblD ≤ dense Wd bd xd cD := by")
A_("  show (∑ i, xd i * Wd i lblD) + bd lblD ≤ (∑ i, xd i * Wd i cD) + bd cD")
A_("  simp [Wd, xd, bd, lblD, cD, Fin.sum_univ_succ]")
A_("  norm_num")
A_("")
A_("-- ── softmax facts at the trained logits (no `exp` is ever evaluated) ──")
A_("")
A_("theorem sm_pos : ∀ j, 0 < softmax 10 (dense Wd bd xd) j := fun j =>")
A_("  div_pos (Real.exp_pos _)")
A_("    (Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨j, Finset.mem_univ j⟩)")
A_("")
A_("theorem sm_sum : (∑ j, softmax 10 (dense Wd bd xd) j) = 1 := by")
A_("  show (∑ j, Real.exp (dense Wd bd xd j) / ∑ k, Real.exp (dense Wd bd xd k)) = 1")
A_("  simp only [div_eq_mul_inv, ← Finset.sum_mul]")
A_("  exact mul_inv_cancel₀ (ne_of_gt (Finset.sum_pos")
A_("    (fun k _ => Real.exp_pos _) ⟨lblD, Finset.mem_univ _⟩))")
A_("")
A_("theorem sm_le_one : ∀ j, softmax 10 (dense Wd bd xd) j ≤ 1 := by")
A_("  intro j")
A_("  have h := Finset.single_le_sum")
A_("    (f := fun k => softmax 10 (dense Wd bd xd) k)")
A_("    (fun k _ => (sm_pos k).le) (Finset.mem_univ j)")
A_("  rw [sm_sum] at h")
A_("  exact h")
A_("")
A_("/-- **Softmax at the true label ≤ 1/2** — from misclassification alone:")
A_("    `exp z_lbl ≤ exp z_c` and both terms sit inside the positive sum. -/")
A_("theorem sm_lbl_le_half : softmax 10 (dense Wd bd xd) lblD ≤ 1 / 2 := by")
A_("  have hee : Real.exp (dense Wd bd xd lblD) ≤ Real.exp (dense Wd bd xd cD) :=")
A_("    Real.exp_le_exp.mpr hz_lbl_le")
A_("  have hSpos : 0 < ∑ k, Real.exp (dense Wd bd xd k) :=")
A_("    Finset.sum_pos (fun k _ => Real.exp_pos _) ⟨lblD, Finset.mem_univ _⟩")
A_("  have hpair : Real.exp (dense Wd bd xd lblD) + Real.exp (dense Wd bd xd cD) ≤")
A_("      ∑ k, Real.exp (dense Wd bd xd k) := by")
A_("    have hsub := Finset.sum_le_sum_of_subset_of_nonneg")
A_("      (Finset.subset_univ ({lblD, cD} : Finset (Fin 10)))")
A_("      (fun k _ _ => (Real.exp_pos (dense Wd bd xd k)).le)")
A_("    rwa [Finset.sum_pair (by decide : lblD ≠ cD)] at hsub")
A_("  show Real.exp (dense Wd bd xd lblD) / (∑ k, Real.exp (dense Wd bd xd k)) ≤ 1 / 2")
A_("  rw [div_le_iff₀ hSpos]")
A_("  nlinarith [Real.exp_pos (dense Wd bd xd lblD)]")
A_("")
A_("/-- `Σⱼ |softmaxⱼ − onehotⱼ| = 2(1 − softmax_lbl) ≤ 2`. -/")
A_("theorem sm_cot_l1 :")
A_("    (∑ j, |softmax 10 (dense Wd bd xd) j - oneHot 10 lblD j|) ≤ 2 := by")
A_("  have hsplit := Finset.add_sum_erase Finset.univ")
A_("    (fun j => |softmax 10 (dense Wd bd xd) j - oneHot 10 lblD j|)")
A_("    (Finset.mem_univ lblD)")
A_("  have hsplit1 := Finset.add_sum_erase Finset.univ")
A_("    (fun j => softmax 10 (dense Wd bd xd) j) (Finset.mem_univ lblD)")
A_("  rw [sm_sum] at hsplit1")
A_("  have herase : (∑ j ∈ Finset.univ.erase lblD,")
A_("      |softmax 10 (dense Wd bd xd) j - oneHot 10 lblD j|) =")
A_("      ∑ j ∈ Finset.univ.erase lblD, softmax 10 (dense Wd bd xd) j := by")
A_("    refine Finset.sum_congr rfl fun j hj => ?_")
A_("    rw [show oneHot 10 lblD j = 0 from if_neg (Finset.mem_erase.mp hj).1,")
A_("        sub_zero, abs_of_pos (sm_pos j)]")
A_("  have hlbl : |softmax 10 (dense Wd bd xd) lblD - oneHot 10 lblD lblD| =")
A_("      1 - softmax 10 (dense Wd bd xd) lblD := by")
A_("    rw [show oneHot 10 lblD lblD = 1 from if_pos rfl,")
A_("        abs_of_nonpos (by linarith [sm_le_one lblD]), neg_sub]")
A_("  have hs := sm_pos lblD")
A_("  linarith [hsplit, hsplit1, herase, hlbl]")
A_("")
A_("/-- `Σⱼ (softmaxⱼ − onehotⱼ)² ≥ 1/4` — the label term alone, via")
A_("    `softmax_lbl ≤ 1/2`. -/")
A_("theorem sm_cot_sq :")
A_("    (1 : ℝ) / 4 ≤ ∑ j, (softmax 10 (dense Wd bd xd) j - oneHot 10 lblD j) ^ 2 := by")
A_("  have hterm : (1 : ℝ) / 4 ≤")
A_("      (softmax 10 (dense Wd bd xd) lblD - oneHot 10 lblD lblD) ^ 2 := by")
A_("    rw [show oneHot 10 lblD lblD = 1 from if_pos rfl]")
A_("    nlinarith [sm_lbl_le_half, (sm_pos lblD).le]")
A_("  exact hterm.trans (Finset.single_le_sum")
A_("    (f := fun j => (softmax 10 (dense Wd bd xd) j - oneHot 10 lblD j) ^ 2)")
A_("    (fun j _ => sq_nonneg _) (Finset.mem_univ lblD))")
A_("")
A_("-- ── the gradient sums, bracketed by exact rationals ──")
A_("")
A_("theorem grad_key (i : Fin 49) (j : Fin 10) :")
A_(f"    {GRAD} (finProdFinEquiv (i, j)) =")
A_("      xd i * (softmax 10 (dense Wd bd xd) j - oneHot 10 lblD j) := by")
A_("  rw [linear_loss_gradAt, Mat.unflatten_flatten]")
A_("")
A_(f"/-- `Σ|∇L| ≤ {FR_A}` (= 2·Σxᵢ, exact). -/")
A_("theorem gradL1_le :")
A_(f"    (∑ idx, |{GRAD} idx|) ≤ {FR_A} := by")
A_(f"  rw [← finProdFinEquiv.sum_comp (fun idx => |{GRAD} idx|)]")
A_("  rw [Fintype.sum_prod_type]")
A_("  simp_rw [grad_key]")
A_("  calc (∑ i : Fin 49, ∑ j : Fin 10,")
A_("        |xd i * (softmax 10 (dense Wd bd xd) j - oneHot 10 lblD j)|)")
A_("      = ∑ i : Fin 49, xd i * ∑ j : Fin 10,")
A_("          |softmax 10 (dense Wd bd xd) j - oneHot 10 lblD j| := by")
A_("        refine Finset.sum_congr rfl fun i _ => ?_")
A_("        rw [Finset.mul_sum]")
A_("        refine Finset.sum_congr rfl fun j _ => ?_")
A_("        rw [abs_mul, abs_of_nonneg (xd_nonneg i)]")
A_("    _ ≤ ∑ i : Fin 49, xd i * 2 :=")
A_("        Finset.sum_le_sum fun i _ =>")
A_("          mul_le_mul_of_nonneg_left sm_cot_l1 (xd_nonneg i)")
A_("    _ = 2 * ∑ i, xd i := by")
A_("        rw [← Finset.sum_mul, mul_comm]")
A_(f"    _ = {FR_A} := by rw [xd_sum]; norm_num")
A_("")
A_(f"/-- `Σ(∇L)² ≥ {FR_B}/4` (= (Σxᵢ²)/4, exact). -/")
A_("theorem gradSq_lower :")
A_(f"    {FR_B} / 4 ≤ ∑ idx, {GRAD} idx ^ 2 := by")
A_(f"  rw [← finProdFinEquiv.sum_comp (fun idx => {GRAD} idx ^ 2)]")
A_("  rw [Fintype.sum_prod_type]")
A_("  simp_rw [grad_key]")
A_("  have hfac : (∑ i : Fin 49, ∑ j : Fin 10,")
A_("      (xd i * (softmax 10 (dense Wd bd xd) j - oneHot 10 lblD j)) ^ 2) =")
A_("      (∑ i, xd i ^ 2) * ∑ j : Fin 10,")
A_("        (softmax 10 (dense Wd bd xd) j - oneHot 10 lblD j) ^ 2 := by")
A_("    rw [Finset.sum_mul]")
A_("    refine Finset.sum_congr rfl fun i _ => ?_")
A_("    rw [Finset.mul_sum]")
A_("    refine Finset.sum_congr rfl fun j _ => ?_")
A_("    ring")
A_("  rw [hfac]")
A_("  have hB0 : (0:ℝ) ≤ ∑ i, xd i ^ 2 := Finset.sum_nonneg fun i _ => sq_nonneg _")
A_("  calc " + f"{FR_B} / 4 = {FR_B} * (1/4) := by norm_num")
A_(f"    _ = (∑ i, xd i ^ 2) * (1/4) := by rw [xd_sq_sum]")
A_("    _ ≤ (∑ i, xd i ^ 2) * ∑ j : Fin 10,")
A_("        (softmax 10 (dense Wd bd xd) j - oneHot 10 lblD j) ^ 2 :=")
A_("        mul_le_mul_of_nonneg_left sm_cot_sq hB0")
A_("")
A_("-- ── the float-side budgets, evaluated ──")
A_("")
A_(f"/-- Float-forward drift: `denseErr` at `b = 0`, `e = 0`, evaluated exactly per")
A_(f"    class column — `((1+2⁻²⁴)⁵¹ − 1)·Σᵢ|Wᵢₖ|·|xᵢ| ≤ {FR_DEL}`. -/")
A_("theorem hdelta : ∀ k, |binary32.dense Wd bd xd k - dense Wd bd xd k| ≤")
A_(f"    {FR_DEL} := by")
A_("  intro k")
A_("  refine (binary32.dense_close_fresh Wd bd xd k).trans ?_")
A_("  simp only [FloatModel.denseErr, binary32_u]")
A_("  fin_cases k <;>")
A_("    · simp [Wd, xd, bd, Fin.sum_univ_succ]")
A_("      norm_num [u32]")
A_("")
A_("theorem hrho : FloatModel.smRho binary32.u 0 10 < 1 := by")
A_("  simp only [FloatModel.smRho, binary32_u]")
A_("  norm_num [u32]")
A_("")
A_(f"/-- The gradient-oracle budget `η = mulErr u32 1 1 0 (cotErr u32 0 δ 10)`,")
A_(f"    bounded by `{FR_ETA}` — the `exp(2δ)−1` term via the γ-form. -/")
A_("theorem heta : FloatModel.mulErr binary32.u 1 1 0")
A_(f"    (FloatModel.cotErr binary32.u 0 {FR_DEL} 10) ≤ {FR_ETA} := by")
A_(f"  have hE : Real.exp (2 * {FR_DEL}) - 1 ≤ {frac(EUB)} :=")
A_("    (FloatModel.exp_sub_one_le (by norm_num)).trans (by norm_num)")
A_("  simp only [FloatModel.mulErr, FloatModel.cotErr, FloatModel.smErr,")
A_("    FloatModel.smKappa, FloatModel.smRho, binary32_u]")
A_("  norm_num [u32]")
A_("  nlinarith [hE]")
A_("")
A_("theorem heta0 : 0 ≤ FloatModel.mulErr binary32.u 1 1 0")
A_(f"    (FloatModel.cotErr binary32.u 0 {FR_DEL} 10) := by")
A_(f"  have hE0 : (0:ℝ) ≤ Real.exp (2 * {FR_DEL}) - 1 := by")
A_(f"    nlinarith [Real.add_one_le_exp (2 * {FR_DEL})]")
A_("  simp only [FloatModel.mulErr, FloatModel.cotErr, FloatModel.smErr,")
A_("    FloatModel.smKappa, FloatModel.smRho, binary32_u]")
A_("  norm_num [u32]")
A_("  nlinarith [hE0]")
A_("")
A_("-- ── the capstone ──")
A_("")
A_("set_option maxRecDepth 8192 in")
A_(f"/-- **Descent at TRAINED weights.** One binary32 SGD step (lr = {FR_LR}) on")
A_("    the trained linear classifier, at the misclassified witness, decreases")
A_("    the real cross-entropy by ≥ lr·‖∇L‖²/2 — every hypothesis of")
A_("    `linear_float_sgd_descends` discharged, the rounding model constructed")
A_("    (`rndP 23`), zero axioms. Retires the `W = 0` degeneracy caveat of")
A_("    `binary32_linear_sgd_descends_concrete`. -/")
A_("theorem trained_linear_sgd_descends_concrete :")
A_(f"    crossEntropy 10 (dense (Mat.unflatten (Mat.flatten Wd -")
A_(f"        {FR_LR} • binary32.linearFloatGrad Wd bd xd Real.exp lblD)) bd xd)")
A_("        lblD ≤")
A_("      crossEntropy 10 (dense (Mat.unflatten (Mat.flatten Wd)) bd xd) lblD -")
A_(f"        {FR_LR} * (∑ idx, {GRAD} idx ^ 2) / 2 := by")
A_("  have hS1 := gradL1_le")
A_("  have hS2 := gradSq_lower")
A_("  have hS10 : (0:ℝ) ≤ ∑ idx, " + f"|{GRAD} idx| :=")
A_("    Finset.sum_nonneg fun _ _ => abs_nonneg _")
A_("  have hη := heta")
A_("  have hη0 := heta0")
A_("  have hfexp : ∀ t, |Real.exp t - Real.exp t| ≤ (0:ℝ) * Real.exp t := by")
A_("    intro t; simp")
A_("  refine linear_float_sgd_descends binary32 Wd bd xd lblD Real.exp")
A_(f"    (lr := {FR_LR}) (a := 1) (eexp := 0) (δ := {FR_DEL})")
A_("    (by norm_num) xd_le_one (by norm_num) le_rfl (by norm_num) (by norm_num)")
A_("    hfexp hrho hdelta ?_ ?_ ?_")
A_("  · -- hsmall")
A_("    push_cast")
A_("    nlinarith [hS1, hS10, hη, hη0]")
A_("  · -- h1")
A_("    push_cast")
A_("    nlinarith [hS1, hS2, hS10, hη, hη0]")
A_("  · -- h2")
A_(f"    set S1 := ∑ idx, |{GRAD} idx| with hS1def")
A_(f"    set S2 := ∑ idx, {GRAD} idx ^ 2 with hS2def")
A_("    set η := FloatModel.mulErr binary32.u 1 1 0")
A_(f"      (FloatModel.cotErr binary32.u 0 {FR_DEL} 10) with hηdef")
A_(f"    have hD : {FR_LR} * (S1 + ((49 * 10 : ℕ) : ℝ) * η) ≤ {frac(Dub)} := by")
A_("      push_cast")
A_("      nlinarith [hS1, hη, hη0]")
A_(f"    have hD0 : (0:ℝ) ≤ {FR_LR} * (S1 + ((49 * 10 : ℕ) : ℝ) * η) := by")
A_("      push_cast")
A_("      nlinarith [hS10, hη0]")
A_("    have hTpos : (0:ℝ) < 1 - 2 * ((1:ℝ) * ({FR_LR} * (S1 + ((49 * 10 : ℕ) : ℝ) * η))) := by".replace("{FR_LR}", FR_LR))
A_("      nlinarith [hD, hD0]")
A_("    rw [div_mul_eq_mul_div, div_le_iff₀ hTpos]")
A_("    have hRHS0 : (0:ℝ) ≤ " + f"{FR_LR} * S2 / 4 := by")
A_("      have : (0:ℝ) ≤ S2 := Finset.sum_nonneg fun _ _ => sq_nonneg _")
A_("      positivity")
A_("    nlinarith [hD, hD0, hS2, hTpos, hRHS0,")
A_("      mul_le_mul_of_nonneg_left hD hD0]")
A_("")
A_("/-- **The drop is strictly positive**: the witness is misclassified, so")
A_(f"    `Σ∇² ≥ {FR_B}/4 > 0` and the loss strictly decreases. -/")
A_("theorem trained_linear_sgd_strictly_descends :")
A_(f"    crossEntropy 10 (dense (Mat.unflatten (Mat.flatten Wd -")
A_(f"        {FR_LR} • binary32.linearFloatGrad Wd bd xd Real.exp lblD)) bd xd)")
A_("        lblD <")
A_("      crossEntropy 10 (dense (Mat.unflatten (Mat.flatten Wd)) bd xd) lblD := by")
A_("  have h := trained_linear_sgd_descends_concrete")
A_("  have h2 := gradSq_lower")
A_("  linarith [h, h2]")
A_("")
A_("end TrainedLinearDescent")
A_("end Proofs")

with open(OUT, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"wrote {OUT}: {len(L)} lines")
