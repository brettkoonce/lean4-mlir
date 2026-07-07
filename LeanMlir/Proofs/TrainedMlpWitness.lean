import LeanMlir.Proofs.MLP
import LeanMlir.Proofs.JacobianSeal
import LeanMlir.Proofs.LipschitzCertInstance

/-! # Trained-weight whole-network VJP witness (MLP rung)

**REDUCED CERTIFICATE MODEL** — this file's concrete net is the 4×4-pooled 49-dim
MNIST family (width-8 hidden, /128–/256 rational weights), NOT the canonical
784→512→512→10 `mlpVerified`; chosen so every margin/norm/SOS check is exact rational
arithmetic in-kernel. Canonical surface: `Proofs/MlpCanonical.lean`.

The audit's "live witnesses use synthetic weights" gap, closed at the MLP rung:
the SAME trained, /128-rationalized 49→8→10 pooled-MNIST network certified in
`LipschitzCertInstance.lean` (test acc 89.8%) instantiates the conditional VJP
framework at a REAL input — test digit #1895 — with every ReLU smoothness
hypothesis discharged from the exact rational pre-activations (7 units strictly
on, 1 strictly off; nothing sits on a kink), rather than engineered by synthetic
β-shifts as in `ResNet34Live*`/`Mnv2Live`. Levels:

* level 1 — `trainedMlp_has_vjp_at` (+ `.correct`): the whole-net backward exists
  and equals the `fderiv`-contracted Jacobian at the witness;
* level 3 — `trainedMlp_backward_nontrivial`: the backward is not the zero map
  (via the explicit Jacobian entry `pdiv = -85017/8192 ≈ -10.38`);
* `trainedMlp_jacobian_nonzero` / `trainedMlp_not_constant`: the `fderiv` forms.

Weights/input are imported from `LipschitzCertInstance` (generator:
`scripts/lipschitz_cert_rationalize.py`); the dense convention is transposed
(`Mat` is input×output) and biases are zero. -/

namespace Proofs
namespace TrainedMlp

open LipschitzCertDemo

/-- Hidden weights in the `Mat` (input×output) convention: `W1V i k = W1t k i`. -/
noncomputable def W1V : Mat 49 8 := fun i k => W1t k i
noncomputable def W2V : Mat 8 10 := fun k c => W2t c k
noncomputable def b8 : Vec 8 := fun _ => 0
noncomputable def b10 : Vec 10 := fun _ => 0

/-- The pooled test digit as a `Vec` (coordinates of `xt`). -/
noncomputable def xtV : Vec 49 := fun j => xt j

/-- The trained forward in the VJP framework's vocabulary. -/
noncomputable def fwd : Vec 49 → Vec 10 :=
  dense W2V b10 ∘ relu 8 ∘ dense W1V b8

/-- The hidden pre-activations transfer from the certificate file's exact
    evaluation (`hpre_eval`): same sums, transposed convention. -/
theorem preact_eq (k : Fin 8) : dense W1V b8 xtV k = hpreVals k := by
  have h : dense W1V b8 xtV k = ∑ j, W1t k j * xt j := by
    show (∑ j, xtV j * W1V j k) + b8 k = _
    rw [show b8 k = (0:ℝ) from rfl, add_zero]
    exact Finset.sum_congr rfl fun j _ => mul_comm _ _
  rw [h]
  exact hpre_eval k

/-- **Smoothness at the trained witness**: no hidden unit sits on the ReLU kink —
    the exact pre-activations are all nonzero (7 positive, 1 negative). The
    condition the synthetic live witnesses had to engineer, here inherited from
    training. -/
theorem preact_ne : ∀ k, dense W1V b8 xtV k ≠ 0 := by
  intro k
  rw [preact_eq]
  fin_cases k <;> norm_num [hpreVals]

/-- **Level 1: the trained-weight whole-net VJP witness** — `HasVJPAt fwd xtV`,
    every hypothesis discharged (dense layers globally smooth, ReLU off-kink by
    `preact_ne`). The 2-layer analogue of `mlp_has_vjp_at`, at trained weights
    and a real input. -/
noncomputable def trainedMlp_has_vjp_at : HasVJPAt fwd xtV := by
  unfold fwd
  have step1 : HasVJPAt (relu 8 ∘ dense W1V b8) xtV :=
    vjp_comp_at (dense W1V b8) (relu 8) xtV
      ((dense_differentiable W1V b8) xtV)
      (relu_differentiableAt_of_smooth 8 _ preact_ne)
      ((dense_has_vjp W1V b8).toHasVJPAt xtV)
      (relu_has_vjp_at 8 _ preact_ne)
  have step1_diff : DifferentiableAt ℝ (relu 8 ∘ dense W1V b8) xtV :=
    (relu_differentiableAt_of_smooth 8 _ preact_ne).comp xtV
      ((dense_differentiable W1V b8) xtV)
  exact vjp_comp_at (relu 8 ∘ dense W1V b8) (dense W2V b10) xtV
    step1_diff
    ((dense_differentiable W2V b10) _)
    step1
    ((dense_has_vjp W2V b10).toHasVJPAt _)

/-- The witness's contract, exposed: backward = the `pdiv`-contracted Jacobian. -/
theorem trainedMlp_has_vjp_correct (dy : Vec 10) (i : Fin 49) :
    trainedMlp_has_vjp_at.backward dy i = ∑ j, pdiv fwd xtV i j * dy j :=
  trainedMlp_has_vjp_at.correct dy i

/-- The whole-net Jacobian in closed form at the witness: dense → masked-ReLU →
    dense collapses to `Σ_k W1[k,j]·mask_k·W2[c,k]` (chain rule through the two
    proven layer Jacobians, both kinks avoided). -/
theorem pdiv_fwd (j : Fin 49) (c : Fin 10) :
    pdiv fwd xtV j c =
      ∑ k, W1V j k * ((if hpreVals k > 0 then (1:ℝ) else 0) * W2V k c) := by
  have hd1 : DifferentiableAt ℝ (dense W1V b8) xtV := (dense_differentiable W1V b8) xtV
  have hrelu : DifferentiableAt ℝ (relu 8) (dense W1V b8 xtV) :=
    relu_differentiableAt_of_smooth 8 _ preact_ne
  have hinner_diff : DifferentiableAt ℝ (relu 8 ∘ dense W1V b8) xtV :=
    hrelu.comp xtV hd1
  have hd2 : DifferentiableAt ℝ (dense W2V b10) ((relu 8 ∘ dense W1V b8) xtV) :=
    (dense_differentiable W2V b10) _
  have houter : pdiv fwd xtV j c =
      ∑ k, pdiv (relu 8 ∘ dense W1V b8) xtV j k *
        pdiv (dense W2V b10) ((relu 8 ∘ dense W1V b8) xtV) k c := by
    show pdiv (dense W2V b10 ∘ (relu 8 ∘ dense W1V b8)) xtV j c = _
    exact pdiv_comp _ _ _ hinner_diff hd2 j c
  have hinner : ∀ k : Fin 8, pdiv (relu 8 ∘ dense W1V b8) xtV j k =
      W1V j k * (if hpreVals k > 0 then (1:ℝ) else 0) := by
    intro k
    have hcomp := pdiv_comp (dense W1V b8) (relu 8) xtV hd1 hrelu j k
    rw [hcomp]
    have hterm : ∀ m : Fin 8, pdiv (dense W1V b8) xtV j m *
        pdiv (relu 8) (dense W1V b8 xtV) m k =
        (if m = k then W1V j m * (if hpreVals m > 0 then (1:ℝ) else 0) else 0) := by
      intro m
      rw [pdiv_dense, pdiv_relu 8 _ preact_ne, preact_eq]
      by_cases hmk : m = k
      · rw [if_pos hmk, if_pos hmk]
      · rw [if_neg hmk, if_neg hmk, mul_zero]
    rw [Finset.sum_congr rfl fun m _ => hterm m,
        Finset.sum_ite_eq' Finset.univ k
          (fun m => W1V j m * (if hpreVals m > 0 then (1:ℝ) else 0))]
    simp
  rw [houter]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [hinner k, pdiv_dense]
  ring

/-- The Jacobian entry `(∂ logit_1 / ∂ x_23)` at the witness, exactly. -/
theorem pdiv_fwd_val : pdiv fwd xtV 23 1 = ((-85017 : ℝ)/8192) := by
  rw [pdiv_fwd]
  simp [W1V, W2V, W1t, W2t, hpreVals, Fin.sum_univ_succ]
  norm_num

theorem pdiv_fwd_ne : pdiv fwd xtV 23 1 ≠ 0 := by
  rw [pdiv_fwd_val]
  norm_num

/-- **Level 3: the trained-weight backward is not the zero map** — the seal the
    synthetic witnesses carry, at trained weights. -/
theorem trainedMlp_backward_nontrivial :
    trainedMlp_has_vjp_at.backward (basisVec 1) 23 ≠ 0 :=
  trainedMlp_has_vjp_at.backward_ne_zero_of_pdiv_ne pdiv_fwd_ne

/-- The `fderiv` form: the whole-net Jacobian at the trained witness is nonzero. -/
theorem trainedMlp_jacobian_nonzero : fderiv ℝ fwd xtV ≠ 0 := by
  intro h
  apply pdiv_fwd_ne
  unfold pdiv
  rw [h]
  rfl

/-- The trained network is not a constant function. -/
theorem trainedMlp_not_constant : ¬ (∀ u v : Vec 49, fwd u = fwd v) := by
  intro h
  apply trainedMlp_jacobian_nonzero
  have hc : fwd = fun _ => fwd xtV := funext fun u => h u xtV
  rw [hc]
  exact (hasFDerivAt_const (fwd xtV) xtV).fderiv

end TrainedMlp
end Proofs
