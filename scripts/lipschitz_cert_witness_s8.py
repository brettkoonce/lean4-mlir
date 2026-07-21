import numpy as np, struct
from fractions import Fraction
from math import ceil

rng = np.random.default_rng(0)
D = "/home/skoonce/lean/klawd_max_power/lean4-jax/data/"

def load_images(fn):
    with open(fn, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)

def load_labels(fn):
    with open(fn, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

Xtr_raw = load_images(D + "train-images-idx3-ubyte"); ytr = load_labels(D + "train-labels-idx1-ubyte")
Str = Xtr_raw.reshape(-1, 7, 4, 7, 4).astype(np.int64).sum(axis=(2, 4)).reshape(-1, 49)
Xtr = Str / 4080.0
H, K, DIM = 8, 10, 49
W1 = rng.normal(0, np.sqrt(2.0 / DIM), (H, DIM)); W2 = rng.normal(0, np.sqrt(2.0 / H), (K, H))
lr, bs = 0.15, 64
for ep in range(12):
    idx = rng.permutation(len(Xtr))
    for b in range(0, len(Xtr), bs):
        xb = Xtr[idx[b:b+bs]]; yb = ytr[idx[b:b+bs]]
        h = xb @ W1.T; hr = np.maximum(h, 0); z = hr @ W2.T
        z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        g = p.copy(); g[np.arange(len(yb)), yb] -= 1; g /= len(yb)
        W1 -= lr * ((g @ W2) * (h > 0)).T @ xb; W2 -= lr * g.T @ hr
W1q = np.round(W1 * 128).astype(np.int64); W2q = np.round(W2 * 128).astype(np.int64)

# preacts at test sample #1895 (same as committed hpreVals)
Xte_raw = load_images(D + "t10k-images-idx3-ubyte")
Ste = Xte_raw.reshape(-1, 7, 4, 7, 4).astype(np.int64).sum(axis=(2, 4)).reshape(-1, 49)
s = [int(v) for v in Ste[1895]]
pre_num = [sum(int(W1q[k, j]) * s[j] for j in range(DIM)) for k in range(H)]
mask = [1 if n > 0 else 0 for n in pre_num]
print("preact numerators:", pre_num)
print("mask:", mask)

# ── Part A: trained-weight VJP witness — pick (j0, c0) maximizing |pdiv| ──
# pdiv fwd xtV j c = Σ_k W1[k,j]/128 · mask_k · W2[c,k]/128
best = (None, None, Fraction(0))
for j in range(DIM):
    for c in range(K):
        v = Fraction(sum(int(W1q[k, j]) * mask[k] * int(W2q[c, k]) for k in range(H)), 128 * 128)
        if abs(v) > abs(best[2]):
            best = (j, c, v)
j0, c0, pv = best
print(f"pdiv argmax: j0={j0}, c0={c0}, value={pv} = {float(pv):.4f}")

# ── Part B: Schatten-8 ──
def gram(Wq):
    k = Wq.shape[0]
    return [[int(sum(int(Wq[a, j]) * int(Wq[b, j]) for j in range(Wq.shape[1]))) for b in range(k)] for a in range(k)]

G1 = gram(W1q); G2 = gram(W2q)
def gram2(G):  # H = GᵀG = G·G (symmetric), integer numerators over 16384²
    k = len(G)
    return [[int(sum(G[c][a] * G[c][b] for c in range(k))) for b in range(k)] for a in range(k)]

H1 = gram2(G1); H2 = gram2(G2)
DEN_H = (128 * 128) ** 2  # 268435456

def s8_B(Hm):
    S = Fraction(sum(x * x for row in Hm for x in row), DEN_H ** 2)  # ‖H‖_F² = Σσ⁸
    B = Fraction(ceil((float(S) ** 0.125) * 1000), 1000)
    while B ** 8 < S:
        B += Fraction(1, 1000)
    return S, B

S1, B1 = s8_B(H1); S2, B2 = s8_B(H2)
LL = B2 * B1
m = Fraction(6953, 500)
print(f"B1'={B1}={float(B1)}, B2'={B2}={float(B2)}, L={LL}={float(LL):.4f}")
print(f"s8 radius = {float(m)/(2**0.5*float(LL)):.5f}  (s4 was 0.11063, frobenius 0.04635)")
print(f"max |H1| numerator digits: {max(len(str(abs(x))) for r in H1 for x in r)}")

def frac(q):
    q = Fraction(q)
    return f"(({q.numerator} : ℝ)/{q.denominator})" if q.denominator != 1 else f"({q.numerator} : ℝ)"

def hrow(vals):
    return "![" + ", ".join(f"(({int(x)} : ℝ)/268435456)" for x in vals) + "]"

# ═══ emit Schatten-8 snippet (appended to LipschitzCertInstance) ═══
L = []
A = L.append
A("")
A(f"/-- `H1t = G1t²` (= `G1tᵀ·G1t`, 8×8, denominators 16384² = 268435456). -/")
A("noncomputable def H1t : Fin 8 → Fin 8 → ℝ :=")
A("  ![" + ",\n    ".join(hrow(H1[a]) for a in range(H)) + "]")
A("")
A("noncomputable def H2t : Fin 10 → Fin 10 → ℝ :=")
A("  ![" + ",\n    ".join(hrow(H2[a]) for a in range(K)) + "]")
A("")
A("set_option maxHeartbeats 1600000 in")
A("theorem H1t_eq : ∀ a b, H1t a b = ∑ c, G1t c a * G1t c b := by")
A("  intro a b")
A("  fin_cases a <;> fin_cases b <;>")
A("    · simp [H1t, G1t, Fin.sum_univ_succ]")
A("      norm_num")
A("")
A("set_option maxHeartbeats 1600000 in")
A("theorem H2t_eq : ∀ a b, H2t a b = ∑ c, G2t c a * G2t c b := by")
A("  intro a b")
A("  fin_cases a <;> fin_cases b <;>")
A("    · simp [H2t, G2t, Fin.sum_univ_succ]")
A("      norm_num")
A("")
A(f"/-- Schatten-8 bound: B₁' = {B1} ≈ (Σσ⁸)^(1/8) (true σ₁ ≈ 7.4525). -/")
A(f"theorem W1t_lip_gram2 : LipschitzL2 {frac(B1)} (denseE W1t) := by")
A("  refine denseE_lipschitzL2_gram2 W1t G1t H1t (by norm_num) G1t_eq H1t_eq ?_")
A("  simp [H1t, Fin.sum_univ_succ]")
A("  norm_num")
A("")
A(f"theorem W2t_lip_gram2 : LipschitzL2 {frac(B2)} (denseE W2t) := by")
A("  refine denseE_lipschitzL2_gram2 W2t G2t H2t (by norm_num) G2t_eq H2t_eq ?_")
A("  simp [H2t, Fin.sum_univ_succ]")
A("  norm_num")
A("")
A(f"/-- Schatten-8 product certificate: L = B₂'·(1·B₁') = {LL} ≈ {float(LL):.2f} — vs the")
A(f"    certified lower bounds ℓ₁·ℓ₂ = 57.38, provably within {float(LL)/ (7.452*7.7) - 1:.1%} of the")
A("    per-layer-optimal product. -/")
A(f"theorem mlpT_lip_gram2 : LipschitzL2 {frac(LL)} mlpT := by")
A("  have h := W2t_lip_gram2.comp (reluE_lipschitzL2.comp W1t_lip_gram2 (by norm_num)) (by norm_num)")
A(f"  have e : {frac(B2)} * (1 * {frac(B1)}) = {frac(LL)} := by norm_num")
A("  rw [e] at h; exact h")
A("")
A(f"theorem trained_radius_gram2_pos : 0 < {frac(m)} / (Real.sqrt 2 * {frac(LL)}) :=")
A("  div_pos (by norm_num)")
A("    (mul_pos (Real.sqrt_pos.mpr (by norm_num)) (by norm_num))")
A("")
A(f"/-- **Schatten-8 trained certificate**: radius ≈ {float(m)/(2**0.5*float(LL)):.4f} (3.3× Frobenius,")
A("    1.4× Schatten-4; the true-σ ceiling for the product method is 0.171). -/")
A("theorem trained_demo_certified_gram2 (δ : EuclideanSpace ℝ (Fin 49))")
A(f"    (hδ : ‖δ‖ < {frac(m)} / (Real.sqrt 2 * {frac(LL)})) :")
A("    ∀ j, j ≠ 2 → mlpT (xt + δ) j < mlpT (xt + δ) 2 :=")
A("  lipschitz_margin_certified_radius mlpT_lip_gram2 (by norm_num) xt_margin hδ")
open("/tmp/claude-1000/-home-skoonce-lean-klawd-max-power-lean4-jax/8f48005c-0a69-42db-9283-1fd4bbae3fe3/scratchpad/s8_snippet.lean", "w").write("\n".join(L) + "\n")

# ═══ emit trained-weight VJP witness file ═══
W = []
A = W.append
A("import LeanMlir.Proofs.Foundation.MLP")
A("import LeanMlir.Proofs.JacobianSeal")
A("import LeanMlir.Proofs.Certificates.LipschitzCertInstance")
A("")
A("/-! # Trained-weight whole-network VJP witness (MLP rung)")
A("")
A("The audit's \"live witnesses use synthetic weights\" gap, closed at the MLP rung:")
A("the SAME trained, /128-rationalized 49→8→10 pooled-MNIST network certified in")
A("`LipschitzCertInstance.lean` (test acc 89.8%) instantiates the conditional VJP")
A("framework at a REAL input — test digit #1895 — with every ReLU smoothness")
A("hypothesis discharged from the exact rational pre-activations (7 units strictly")
A("on, 1 strictly off; nothing sits on a kink), rather than engineered by synthetic")
A("β-shifts as in `ResNet34Live*`/`Mnv2Live`. Levels:")
A("")
A("* level 1 — `trainedMlp_has_vjp_at` (+ `.correct`): the whole-net backward exists")
A("  and equals the `fderiv`-contracted Jacobian at the witness;")
A("* level 3 — `trainedMlp_backward_nontrivial`: the backward is not the zero map")
A("  (via the explicit Jacobian entry `pdiv = " + f"{pv}" + " ≈ " + f"{float(pv):.2f}" + "`);")
A("* `trainedMlp_jacobian_nonzero` / `trainedMlp_not_constant`: the `fderiv` forms.")
A("")
A("Weights/input are imported from `LipschitzCertInstance` (generator:")
A("`scripts/lipschitz_cert_rationalize.py`); the dense convention is transposed")
A("(`Mat` is input×output) and biases are zero. -/")
A("")
A("namespace Proofs")
A("namespace TrainedMlp")
A("")
A("open LipschitzCertDemo")
A("")
A("/-- Hidden weights in the `Mat` (input×output) convention: `W1V i k = W1t k i`. -/")
A("noncomputable def W1V : Mat 49 8 := fun i k => W1t k i")
A("noncomputable def W2V : Mat 8 10 := fun k c => W2t c k")
A("noncomputable def b8 : Vec 8 := fun _ => 0")
A("noncomputable def b10 : Vec 10 := fun _ => 0")
A("")
A("/-- The pooled test digit as a `Vec` (coordinates of `xt`). -/")
A("noncomputable def xtV : Vec 49 := fun j => xt j")
A("")
A("/-- The trained forward in the VJP framework's vocabulary. -/")
A("noncomputable def fwd : Vec 49 → Vec 10 :=")
A("  dense W2V b10 ∘ relu 8 ∘ dense W1V b8")
A("")
A("/-- The hidden pre-activations transfer from the certificate file's exact")
A("    evaluation (`hpre_eval`): same sums, transposed convention. -/")
A("theorem preact_eq (k : Fin 8) : dense W1V b8 xtV k = hpreVals k := by")
A("  have h : dense W1V b8 xtV k = ∑ j, W1t k j * xt j := by")
A("    show (∑ j, xtV j * W1V j k) + b8 k = _")
A("    rw [show b8 k = (0:ℝ) from rfl, add_zero]")
A("    exact Finset.sum_congr rfl fun j _ => mul_comm _ _")
A("  rw [h]")
A("  exact hpre_eval k")
A("")
A("/-- **Smoothness at the trained witness**: no hidden unit sits on the ReLU kink —")
A("    the exact pre-activations are all nonzero (7 positive, 1 negative). The")
A("    condition the synthetic live witnesses had to engineer, here inherited from")
A("    training. -/")
A("theorem preact_ne : ∀ k, dense W1V b8 xtV k ≠ 0 := by")
A("  intro k")
A("  rw [preact_eq]")
A("  fin_cases k <;>")
A("    · simp [hpreVals]")
A("      norm_num")
A("")
A("/-- **Level 1: the trained-weight whole-net VJP witness** — `HasVJPAt fwd xtV`,")
A("    every hypothesis discharged (dense layers globally smooth, ReLU off-kink by")
A("    `preact_ne`). The 2-layer analogue of `mlp_has_vjp_at`, at trained weights")
A("    and a real input. -/")
A("noncomputable def trainedMlp_has_vjp_at : HasVJPAt fwd xtV := by")
A("  unfold fwd")
A("  have step1 : HasVJPAt (relu 8 ∘ dense W1V b8) xtV :=")
A("    vjp_comp_at (dense W1V b8) (relu 8) xtV")
A("      ((dense_differentiable W1V b8) xtV)")
A("      (relu_differentiableAt_of_smooth 8 _ preact_ne)")
A("      ((dense_has_vjp W1V b8).toHasVJPAt xtV)")
A("      (relu_has_vjp_at 8 _ preact_ne)")
A("  have step1_diff : DifferentiableAt ℝ (relu 8 ∘ dense W1V b8) xtV :=")
A("    (relu_differentiableAt_of_smooth 8 _ preact_ne).comp xtV")
A("      ((dense_differentiable W1V b8) xtV)")
A("  exact vjp_comp_at (relu 8 ∘ dense W1V b8) (dense W2V b10) xtV")
A("    step1_diff")
A("    ((dense_differentiable W2V b10) _)")
A("    step1")
A("    ((dense_has_vjp W2V b10).toHasVJPAt _)")
A("")
A("/-- The witness's contract, exposed: backward = the `pdiv`-contracted Jacobian. -/")
A("theorem trainedMlp_has_vjp_correct (dy : Vec 10) (i : Fin 49) :")
A("    trainedMlp_has_vjp_at.backward dy i = ∑ j, pdiv fwd xtV i j * dy j :=")
A("  trainedMlp_has_vjp_at.correct dy i")
A("")
A("/-- The whole-net Jacobian in closed form at the witness: dense → masked-ReLU →")
A("    dense collapses to `Σ_k W1[k,j]·mask_k·W2[c,k]` (chain rule through the two")
A("    proven layer Jacobians, both kinks avoided). -/")
A("theorem pdiv_fwd (j : Fin 49) (c : Fin 10) :")
A("    pdiv fwd xtV j c =")
A("      ∑ k, W1V j k * ((if hpreVals k > 0 then (1:ℝ) else 0) * W2V k c) := by")
A("  have hd1 : DifferentiableAt ℝ (dense W1V b8) xtV := (dense_differentiable W1V b8) xtV")
A("  have hrelu : DifferentiableAt ℝ (relu 8) (dense W1V b8 xtV) :=")
A("    relu_differentiableAt_of_smooth 8 _ preact_ne")
A("  have hinner_diff : DifferentiableAt ℝ (relu 8 ∘ dense W1V b8) xtV :=")
A("    hrelu.comp xtV hd1")
A("  have hd2 : DifferentiableAt ℝ (dense W2V b10) ((relu 8 ∘ dense W1V b8) xtV) :=")
A("    (dense_differentiable W2V b10) _")
A("  have houter : pdiv fwd xtV j c =")
A("      ∑ k, pdiv (relu 8 ∘ dense W1V b8) xtV j k *")
A("        pdiv (dense W2V b10) ((relu 8 ∘ dense W1V b8) xtV) k c := by")
A("    show pdiv (dense W2V b10 ∘ (relu 8 ∘ dense W1V b8)) xtV j c = _")
A("    exact pdiv_comp _ _ _ hinner_diff hd2 j c")
A("  have hinner : ∀ k : Fin 8, pdiv (relu 8 ∘ dense W1V b8) xtV j k =")
A("      W1V j k * (if hpreVals k > 0 then (1:ℝ) else 0) := by")
A("    intro k")
A("    have hcomp := pdiv_comp (dense W1V b8) (relu 8) xtV hd1 hrelu j k")
A("    rw [hcomp]")
A("    have hterm : ∀ m : Fin 8, pdiv (dense W1V b8) xtV j m *")
A("        pdiv (relu 8) (dense W1V b8 xtV) m k =")
A("        (if m = k then W1V j m * (if hpreVals m > 0 then (1:ℝ) else 0) else 0) := by")
A("      intro m")
A("      rw [pdiv_dense, pdiv_relu 8 _ preact_ne, preact_eq]")
A("      by_cases hmk : m = k")
A("      · rw [if_pos hmk, if_pos hmk]")
A("      · rw [if_neg hmk, if_neg hmk, mul_zero]")
A("    rw [Finset.sum_congr rfl fun m _ => hterm m,")
A("        Finset.sum_ite_eq' Finset.univ k")
A("          (fun m => W1V j m * (if hpreVals m > 0 then (1:ℝ) else 0))]")
A("    simp")
A("  rw [houter]")
A("  refine Finset.sum_congr rfl fun k _ => ?_")
A("  rw [hinner k, pdiv_dense]")
A("  ring")
A("")
A(f"/-- The Jacobian entry `(∂ logit_{c0} / ∂ x_{j0})` at the witness, exactly. -/")
A(f"theorem pdiv_fwd_val : pdiv fwd xtV {j0} {c0} = {frac(pv)} := by")
A(f"  rw [pdiv_fwd]")
A("  simp [W1V, W2V, W1t, W2t, hpreVals, Fin.sum_univ_succ]")
A("  norm_num")
A("")
A(f"theorem pdiv_fwd_ne : pdiv fwd xtV {j0} {c0} ≠ 0 := by")
A("  rw [pdiv_fwd_val]")
A("  norm_num")
A("")
A("/-- **Level 3: the trained-weight backward is not the zero map** — the seal the")
A("    synthetic witnesses carry, at trained weights. -/")
A("theorem trainedMlp_backward_nontrivial :")
A(f"    trainedMlp_has_vjp_at.backward (basisVec {c0}) {j0} ≠ 0 :=")
A("  trainedMlp_has_vjp_at.backward_ne_zero_of_pdiv_ne pdiv_fwd_ne")
A("")
A("/-- The `fderiv` form: the whole-net Jacobian at the trained witness is nonzero. -/")
A("theorem trainedMlp_jacobian_nonzero : fderiv ℝ fwd xtV ≠ 0 := by")
A("  intro h")
A("  apply pdiv_fwd_ne")
A("  unfold pdiv")
A("  rw [h]")
A("  rfl")
A("")
A("/-- The trained network is not a constant function. -/")
A("theorem trainedMlp_not_constant : ¬ (∀ u v : Vec 49, fwd u = fwd v) := by")
A("  intro h")
A("  apply trainedMlp_jacobian_nonzero")
A("  have hc : fwd = fun _ => fwd xtV := funext fun u => h u xtV")
A("  rw [hc]")
A("  exact (hasFDerivAt_const (fwd xtV) xtV).fderiv")
A("")
A("end TrainedMlp")
A("end Proofs")
open("/tmp/claude-1000/-home-skoonce-lean-klawd-max-power-lean4-jax/8f48005c-0a69-42db-9283-1fd4bbae3fe3/scratchpad/TrainedMlpWitness.lean", "w").write("\n".join(W) + "\n")
print("wrote both")
