import LeanMlir.Proofs.Float.FloatBridge
import LeanMlir.Proofs.SgdDescentLinear
import Mathlib.Data.Int.Log

/-! # The binary32 / E4M3 rounding models, CONSTRUCTED (zero axioms)

The whole `FloatBridge` suite is `∀ M : FloatModel, …` — abstract over any rounding
operator satisfying the standard relative-error model `|rnd x − x| ≤ u·|x|`. This file
provides the named hardware-precision inhabitants `binary32` / `fp8E4M3`.

Historically it did so via two explicit axioms (`ieeeRnd`/`ieeeRnd_err` — "a rounding
operator at unit roundoff `u` satisfying the standard model exists"), quarantined from
the zero-axiom `Proofs` suite. Those axioms are now DISCHARGED (post_audit_roadmap §2):
`rndP p` is round-to-nearest on the unbounded-exponent `p`-bit-significand grid —
exactly the idealization the old axiom's docstring said it modeled ("the `∀x` form
abstracts away overflow and the subnormal floor") — and `rndP_err` PROVES the standard
model `|rndP p x − x| ≤ 2⁻¹⁻ᵖ·|x|` (Higham §2.2) from Mathlib alone: scale into the
binade via `Int.zpow_log_le_self`, `|t − round t| ≤ 1/2`, unscale. `binary32` = the
`p = 23` grid at `u32 = 2⁻²⁴`; `fp8E4M3` = the `p = 3` grid at `u_e4m3 = 2⁻⁴`.

With the named models in hand, the 2026-06 audit's gaps 2 and 3 stay realized:
* **gap 2** — `binary32_e4m3_argmax_preserved`: the fp8 argmax-preservation theorem, an
  *unconditional* corollary about the named hardware models (no `∀ M ∀ L`).
* **gap 3** — `binary32_linear_sgd_descends_concrete`: one binary32 SGD step on a concrete
  linear classifier provably decreases the real loss, with the descent smallness conditions
  `hsmall`/`h1`/`h2` *discharged* (not assumed) for a concrete `(W, x, lr)`.

WHAT THE DISCHARGE DOES **NOT** BUY (kept honest): the kernel↔model boundary — FMA
contraction, reduction reassociation, "the GPU behaves like round-to-nearest on this
grid" — remains trusted exactly as before (`planning/floatbridge_certificate_gaps.md`);
true binary32 also has overflow and a subnormal floor `rndP` idealizes away
(`FloatSubnormalBridge` models the latter hypothesis-style). The trust moves from "an
operator with this bound exists" (mathematically mild — `id` satisfies it at any `u ≥ 0`)
to a CONCRETE, inspectable operator with the bound proved. The win is hygiene and
inspectability: the repo now contains zero `axiom` declarations anywhere, and this file
sits inside the ordinary `Proofs`/`AuditAxioms` 3-axiom closure like everything else.
-/

namespace Proofs
open Proofs

-- ════════════════════════════════════════════════════════════════
-- § Gap 1 — the IEEE standard-model rounding operator, constructed
-- ════════════════════════════════════════════════════════════════

/-- **Round-to-nearest on the unbounded-exponent `p`-bit-significand grid.**
    For `x ≠ 0` with binade exponent `e = Int.log 2 |x|` (i.e. `2^e ≤ |x| < 2^(e+1)`),
    round `x` to the nearest multiple of `2^(e−p)` — a significand of `p` fractional
    bits, every exponent available. This is IEEE round-to-nearest minus overflow and
    subnormals, the standard-model idealization. -/
noncomputable def rndP (p : ℕ) (x : ℝ) : ℝ :=
  if x = 0 then 0 else
    (round (x / (2 : ℝ) ^ (Int.log 2 |x| - (p : ℤ))) : ℝ) *
      (2 : ℝ) ^ (Int.log 2 |x| - (p : ℤ))

@[simp] theorem rndP_zero (p : ℕ) : rndP p 0 = 0 := by simp [rndP]

/-- **The standard model, PROVED** (formerly the `ieeeRnd_err` axiom):
    `|rndP p x − x| ≤ 2⁻¹⁻ᵖ·|x|`. The grid spacing at `x` is `2^(e−p)`, nearest-rounding
    contributes half a step `2^(e−p−1)`, and `2^e ≤ |x|` turns that into the relative
    bound. Mathlib-only: `Int.zpow_log_le_self` + `abs_sub_round`. -/
theorem rndP_err (p : ℕ) (x : ℝ) :
    |rndP p x - x| ≤ ((2 : ℝ) ^ (p + 1))⁻¹ * |x| := by
  rcases eq_or_ne x 0 with hx | hx
  · simp [hx]
  · have hax : (0 : ℝ) < |x| := abs_pos.mpr hx
    unfold rndP
    rw [if_neg hx]
    set e : ℤ := Int.log 2 |x| with he
    set s : ℝ := (2 : ℝ) ^ (e - (p : ℤ)) with hs
    have hs0 : (0 : ℝ) < s := zpow_pos (by norm_num) _
    have hkey : (round (x / s) : ℝ) * s - x = ((round (x / s) : ℝ) - x / s) * s := by
      rw [sub_mul, div_mul_cancel₀ x (ne_of_gt hs0)]
    rw [hkey, abs_mul, abs_of_pos hs0]
    have h1 : |(round (x / s) : ℝ) - x / s| ≤ 1 / 2 := by
      rw [abs_sub_comm]
      exact abs_sub_round (x / s)
    have h2 : (2 : ℝ) ^ e ≤ |x| := by
      exact_mod_cast Int.zpow_log_le_self (by norm_num) hax
    have hs_eq : (1 / 2 : ℝ) * s = ((2 : ℝ) ^ (p + 1))⁻¹ * (2 : ℝ) ^ e := by
      rw [hs, show (1 / 2 : ℝ) = (2 : ℝ) ^ (-1 : ℤ) by norm_num,
          ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0),
          ← zpow_natCast (2 : ℝ) (p + 1), ← zpow_neg,
          ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      congr 1
      push_cast
      ring
    calc |(round (x / s) : ℝ) - x / s| * s
        ≤ (1 / 2) * s := mul_le_mul_of_nonneg_right h1 (le_of_lt hs0)
      _ = ((2 : ℝ) ^ (p + 1))⁻¹ * (2 : ℝ) ^ e := hs_eq
      _ ≤ ((2 : ℝ) ^ (p + 1))⁻¹ * |x| := by
          exact mul_le_mul_of_nonneg_left h2 (by positivity)

/-- The `FloatModel` of the constructed `p`-bit grid, at any unit roundoff
    `u ≥ 2⁻¹⁻ᵖ` (formerly `ieeeModel`, whose `rnd` was the axiom). -/
noncomputable def gridModel (p : ℕ) (u : ℝ) (hu : ((2 : ℝ) ^ (p + 1))⁻¹ ≤ u) :
    FloatModel where
  rnd := rndP p
  u := u
  u_nonneg := le_trans (by positivity) hu
  err := fun x => (rndP_err p x).trans
    (mul_le_mul_of_nonneg_right hu (abs_nonneg x))

/-- **binary32** (IEEE-754 single, fp32 accumulate): the `p = 23` grid (24-bit
    significand) at unit roundoff `u32 = 2⁻²⁴` — the bound is tight for the grid. -/
noncomputable def binary32 : FloatModel := gridModel 23 u32 (by norm_num [u32])

/-- **fp8 E4M3** (the low-precision leaf): the `p = 3` grid (3 mantissa bits) at
    unit roundoff `u_e4m3 = 2⁻⁴`, normal-range model. -/
noncomputable def fp8E4M3 : FloatModel := gridModel 3 u_e4m3 (by norm_num [u_e4m3])

@[simp] theorem binary32_u : binary32.u = u32 := rfl
@[simp] theorem fp8E4M3_u : fp8E4M3.u = u_e4m3 := rfl

/-- `u32 ≤ u_e4m3`: fp32 is at least as precise as fp8 (`2⁻²⁴ ≤ 2⁻⁴`). -/
theorem u32_le_u_e4m3 : u32 ≤ u_e4m3 := by norm_num [u32, u_e4m3]

-- ════════════════════════════════════════════════════════════════
-- § Gap 2 — fp8 argmax preservation, now unconditional on the models
-- ════════════════════════════════════════════════════════════════

/-- **Gap 2: the fp8 guarantee for the named hardware models.** For the certified
    MNIST-linear classifier evaluated with an **fp32 accumulate (`binary32`) / fp8-E4M3 leaf
    (`fp8E4M3`)** mixed forward, with trained `|W| ≤ 3/5`, `|b| ≤ 1`, pixels `|x| ≤ 1`:
    whenever the exact-`ℝ` logit margin at the top class `k` exceeds `122`, the deployed
    fp8 forward keeps `k` as the strict argmax. This is `linear_e4m3_argmax_preserved`
    instantiated at the concrete models — no `∀ M ∀ L`, no axiom: the models are the
    constructed `rndP` grids (bare 3-axiom closure under `#print axioms`). -/
theorem binary32_e4m3_argmax_preserved {n : ℕ}
    {W : Mat 784 n} {b : Vec n} {x : Vec 784}
    (hW : ∀ i j, |W i j| ≤ 3 / 5) (hb : ∀ j, |b j| ≤ 1) (hx : ∀ i, |x i| ≤ 1)
    (k : Fin n)
    (hmargin : ∀ i, i ≠ k →
      (122 : ℝ) < Proofs.dense W b x k - Proofs.dense W b x i) :
    ∀ i, i ≠ k →
      binary32.denseMixed fp8E4M3 W b x i < binary32.denseMixed fp8E4M3 W b x k :=
  binary32.linear_e4m3_argmax_preserved fp8E4M3
    (by simp) (by simp) hW hb hx k hmargin

#print axioms binary32_e4m3_argmax_preserved

/-- **Gap 2, the "loose 122" is a fan-in artifact.** The worst-case fp8 logit budget at
    the MNIST input dimension (784) is `≤ 61`, forcing a `> 122` margin. That figure is
    `≈ 2·u_e4m3 · (m·w·a)` — *linear in the input dimension `m`*. At a small input
    (`m = 4`) the same worst-case budget is already `≤ 1/2`, so a margin of just `> 1`
    certifies the prediction. This makes precise why the deployed net (errors not aligned,
    activations far below the `m·w·a` ceiling) needs only the measured `0.38` drift, not
    `61`: the bound scales with realized fan-in, not the worst-case 784. -/
theorem binary32_e4m3_budget_small :
    FloatModel.denseMixedBudget binary32.u fp8E4M3.u 4 (3 / 5) 1 1 ≤ 1 / 2 := by
  simp only [binary32_u, fp8E4M3_u, FloatModel.denseMixedBudget]
  norm_num [u32, u_e4m3]

/-- The matching argmax corollary at the small input: a margin `> 1` (vs `> 122` at 784)
    suffices for the fp8 forward to preserve the prediction. -/
theorem binary32_e4m3_argmax_small {n : ℕ}
    {W : Mat 4 n} {b : Vec n} {x : Vec 4}
    (hW : ∀ i j, |W i j| ≤ 3 / 5) (hb : ∀ j, |b j| ≤ 1) (hx : ∀ i, |x i| ≤ 1)
    (k : Fin n)
    (hmargin : ∀ i, i ≠ k → (1 : ℝ) < Proofs.dense W b x k - Proofs.dense W b x i) :
    ∀ i, i ≠ k →
      binary32.denseMixed fp8E4M3 W b x i < binary32.denseMixed fp8E4M3 W b x k := by
  intro i hik
  -- per-logit perturbation is within the small budget
  have hB : ∀ j, |binary32.denseMixed fp8E4M3 W b x j - Proofs.dense W b x j| ≤ 1 / 2 :=
    fun j => le_trans
      (binary32.dense_close_mixed_uniform_budget fp8E4M3 (a := 1) (by norm_num) hW hb hx j)
      binary32_e4m3_budget_small
  have := FloatModel.argmax_preserved (z := Proofs.dense W b x)
    (z' := binary32.denseMixed fp8E4M3 W b x) (k := k) (B := 1 / 2) hB
    (fun i' hi' => by have := hmargin i' hi'; linarith)
  exact this i hik

#print axioms binary32_e4m3_argmax_small

-- ════════════════════════════════════════════════════════════════
-- § Gap 3 — one concrete binary32 SGD step, smallness conditions discharged
-- ════════════════════════════════════════════════════════════════

/-- The rounding of `0` is `0` (forced by the relative-error model at `x = 0`). -/
theorem FloatModel.rnd_zero (M : FloatModel) : M.rnd 0 = 0 := by
  have h := M.err 0
  rw [sub_zero, abs_zero, mul_zero] at h
  exact abs_nonpos_iff.mp h

/-- A rounded dot product against the all-zero vector is `0`. -/
theorem FloatModel.dot_right_zero (M : FloatModel) :
    ∀ {k : Nat} (x : Vec k), M.dot x (fun _ => (0 : ℝ)) = 0
  | 0, _ => rfl
  | _ + 1, x => by
      rw [FloatModel.dot_succ, M.dot_right_zero (fun i => x i.castSucc)]
      simp only [FloatModel.mul, mul_zero, FloatModel.add, add_zero, M.rnd_zero]

/-- A concrete minimal linear classifier: 2 inputs, 2 classes, all-zero weights,
    one-hot input, label 0. The all-zero weights make the exact and float forward both
    `0`, the softmax uniform `1/2`, and every gradient sum an exact rational — so the
    descent smallness conditions reduce to checkable arithmetic. (A *satisfiability*
    witness for the descent hypotheses, in the spirit of `CnnConcrete`. The
    NON-degenerate sibling is `TrainedLinearDescent.trained_linear_sgd_descends_concrete`
    — the same theorem at TRAINED weights and a real misclassified input.) -/
noncomputable def W0 : Mat 2 2 := fun _ _ => 0
noncomputable def b0 : Vec 2 := fun _ => 0
noncomputable def x0 : Vec 2 := fun i => if i = 0 then 1 else 0
def lbl : Fin 2 := 0

/-- **Gap 3: one binary32 SGD step provably decreases the real loss, with the descent
    smallness conditions discharged (not assumed).** The step uses the *actual*
    float-computed gradient `binary32.linearFloatGrad`; the conclusion bounds the *real*
    cross-entropy after the step by the real cross-entropy before minus `lr·‖∇‖²/2`. All
    of `hsmall`/`h1`/`h2` are proved for the concrete `(W0, x0, lr = 1/100)`. Axiom-free:
    `binary32` is the constructed `rndP 23` grid (bare triple surfaced below). -/
theorem binary32_linear_sgd_descends_concrete :
    crossEntropy 2 (dense (Mat.unflatten (Mat.flatten W0 -
        (1 / 100 : ℝ) • binary32.linearFloatGrad W0 b0 x0 Real.exp lbl)) b0 x0) lbl ≤
      crossEntropy 2 (dense (Mat.unflatten (Mat.flatten W0)) b0 x0) lbl -
        (1 / 100 : ℝ) * (∑ idx, gradAt
          (fun w => crossEntropy 2 (dense (Mat.unflatten w) b0 x0) lbl)
          (Mat.flatten W0) idx ^ 2) / 2 := by
  -- exact and float forward are both the zero vector
  have hdense : dense W0 b0 x0 = fun _ => (0 : ℝ) := by
    funext j
    show (∑ i, x0 i * W0 i j) + b0 j = 0
    simp [W0, b0]
  have hsm : ∀ j, softmax 2 (dense W0 b0 x0) j = 1 / 2 := by
    intro j
    rw [hdense]
    simp only [softmax, Real.exp_zero, Fin.sum_univ_two]
    norm_num
  have hfd : ∀ k', binary32.dense W0 b0 x0 k' = 0 := by
    intro k'
    simp only [FloatModel.dense]
    rw [show (fun i => W0 i k') = (fun _ => (0 : ℝ)) from rfl, binary32.dot_right_zero]
    simp only [FloatModel.add, show b0 k' = (0 : ℝ) from rfl, zero_add, binary32.rnd_zero]
  -- per-entry gradient (closed form): xᵢ · (½ − onehotⱼ)
  have key : ∀ (i j : Fin 2),
      gradAt (fun w => crossEntropy 2 (dense (Mat.unflatten w) b0 x0) lbl)
        (Mat.flatten W0) (finProdFinEquiv (i, j)) = x0 i * (1 / 2 - oneHot 2 lbl j) := by
    intro i j
    rw [linear_loss_gradAt, Mat.unflatten_flatten, hsm j]
  -- the two gradient sums, exactly
  have hSabs : (∑ idx, |gradAt
      (fun w => crossEntropy 2 (dense (Mat.unflatten w) b0 x0) lbl)
      (Mat.flatten W0) idx|) = 1 := by
    rw [← finProdFinEquiv.sum_comp (fun idx => |gradAt
      (fun w => crossEntropy 2 (dense (Mat.unflatten w) b0 x0) lbl) (Mat.flatten W0) idx|)]
    rw [Fintype.sum_prod_type]
    simp_rw [key]
    simp [Fin.sum_univ_two, x0, oneHot, lbl]
    norm_num
  have hSsq : (∑ idx, gradAt
      (fun w => crossEntropy 2 (dense (Mat.unflatten w) b0 x0) lbl)
      (Mat.flatten W0) idx ^ 2) = 1 / 2 := by
    rw [← finProdFinEquiv.sum_comp (fun idx => gradAt
      (fun w => crossEntropy 2 (dense (Mat.unflatten w) b0 x0) lbl) (Mat.flatten W0) idx ^ 2)]
    rw [Fintype.sum_prod_type]
    simp_rw [key]
    simp [Fin.sum_univ_two, x0, oneHot, lbl]
    norm_num
  -- the binary32 head budget η, bounded
  have hη : FloatModel.mulErr binary32.u 1 1 0 (FloatModel.cotErr binary32.u 0 0 2) ≤ 1 / 500 := by
    simp only [binary32_u, FloatModel.mulErr, FloatModel.cotErr, FloatModel.smErr,
      FloatModel.smKappa, FloatModel.smRho]
    norm_num [u32]
  have hη0 : 0 ≤ FloatModel.mulErr binary32.u 1 1 0 (FloatModel.cotErr binary32.u 0 0 2) := by
    simp only [binary32_u, FloatModel.mulErr, FloatModel.cotErr, FloatModel.smErr,
      FloatModel.smKappa, FloatModel.smRho]
    norm_num [u32]
  -- side hypotheses
  have hx : ∀ i, |x0 i| ≤ (1 : ℝ) := by
    intro i; simp only [x0]; split <;> norm_num
  have hfexp : ∀ t, |Real.exp t - Real.exp t| ≤ (0 : ℝ) * Real.exp t := by
    intro t; simp
  have hρ1 : FloatModel.smRho binary32.u 0 2 < 1 := by
    simp only [binary32_u, FloatModel.smRho]; norm_num [u32]
  have hδ : ∀ k', |binary32.dense W0 b0 x0 k' - dense W0 b0 x0 k'| ≤ (0 : ℝ) := by
    intro k'; rw [hfd, hdense]; simp
  -- abbreviation for the binary32 head budget (proved `≤ 1/500`, `≥ 0` above)
  set η := FloatModel.mulErr binary32.u 1 1 0 (FloatModel.cotErr binary32.u 0 0 2) with hηdef
  -- the three smallness/dominance conditions, now over concrete rationals
  have hsmall : 2 * ((1 : ℝ) * ((1 / 100) * ((∑ idx, |gradAt
      (fun w => crossEntropy 2 (dense (Mat.unflatten w) b0 x0) lbl)
      (Mat.flatten W0) idx|) + ((2 * 2 : ℕ) : ℝ) * η))) < 1 := by
    rw [hSabs]; push_cast; nlinarith [hη, hη0]
  have h1 : (1 / 100 : ℝ) * η *
      (∑ idx, |gradAt (fun w => crossEntropy 2 (dense (Mat.unflatten w) b0 x0) lbl)
        (Mat.flatten W0) idx|) ≤
      (1 / 100 : ℝ) * (∑ idx, gradAt
        (fun w => crossEntropy 2 (dense (Mat.unflatten w) b0 x0) lbl)
        (Mat.flatten W0) idx ^ 2) / 4 := by
    rw [hSabs, hSsq]; nlinarith [hη, hη0]
  have hTpos : (0 : ℝ) < 1 - 2 * (1 * ((1 / 100) * (1 + ((2 * 2 : ℕ) : ℝ) * η))) := by
    push_cast; nlinarith [hη, hη0]
  have h2 : (2 * (1 : ℝ) ^ 2 / (1 - 2 * (1 * ((1 / 100) * ((∑ idx, |gradAt
        (fun w => crossEntropy 2 (dense (Mat.unflatten w) b0 x0) lbl)
        (Mat.flatten W0) idx|) + ((2 * 2 : ℕ) : ℝ) * η))))) *
      ((1 / 100) * ((∑ idx, |gradAt
        (fun w => crossEntropy 2 (dense (Mat.unflatten w) b0 x0) lbl)
        (Mat.flatten W0) idx|) + ((2 * 2 : ℕ) : ℝ) * η)) ^ 2 ≤
      (1 / 100 : ℝ) * (∑ idx, gradAt
        (fun w => crossEntropy 2 (dense (Mat.unflatten w) b0 x0) lbl)
        (Mat.flatten W0) idx ^ 2) / 4 := by
    rw [hSabs, hSsq]
    rw [div_mul_eq_mul_div, div_le_iff₀ hTpos]
    push_cast; push_cast at hTpos; nlinarith [hη, hη0, hTpos, sq_nonneg η]
  -- discharge the descent lemma at the concrete instance
  exact linear_float_sgd_descends binary32 W0 b0 x0 lbl Real.exp
    (lr := 1 / 100) (a := 1) (eexp := 0) (δ := 0)
    (by norm_num) hx (by norm_num) le_rfl (by norm_num) le_rfl hfexp hρ1 hδ
    hsmall h1 h2

#print axioms binary32_linear_sgd_descends_concrete

end Proofs
