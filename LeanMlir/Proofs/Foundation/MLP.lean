import LeanMlir.Proofs.Foundation.Tensor
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# MLP VJP Proofs

Formal VJP correctness for the layers of a 3-layer MLP.
All definitions over `ℝ`, proofs use Mathlib's `Finset.sum`.
-/

open Finset BigOperators Classical

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Dense Layer:  y = xW + b
-- ════════════════════════════════════════════════════════════════

noncomputable def dense {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m) : Vec n :=
  fun j => (∑ i : Fin m, x i * W i j) + b j

/-- **Dense Jacobian** — `∂(W·x + b)_j/∂x_i = W_{ij}`. `dense W b` is the linear map
    `x ↦ (Σ_i x_i W_{ij})_j` plus the constant `b`, so `pdiv_of_affine` reads the entry off the
    basis vector and the Kronecker sum collapses. -/
theorem pdiv_dense {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (i : Fin m) (j : Fin n) :
    pdiv (dense W b) x i j = W i j := by
  rw [show dense W b = fun y => (fun k => ∑ i' : Fin m, y i' * W i' k) + b from rfl,
    pdiv_of_affine (fun y k => ∑ i' : Fin m, y i' * W i' k) b
      (fun _ _ => by funext; simp [add_mul, Finset.sum_add_distrib])
      (fun _ _ => by funext; simp [Finset.mul_sum, mul_assoc])]
  simp

/-- **Jacobian of dense wrt W** — `∂dense(W, b, x)_j/∂W_{i, j'} = x_i·δ(j, j')`. Over the
    flatten bijection `v ↦ dense (unflatten v) b x` is linear in `v` plus the constant `b`
    (`pdiv_of_affine`). Symmetric counterpart to `pdiv_dense`. -/
theorem pdiv_dense_W {m n : Nat} (b : Vec n) (x : Vec m) (W : Mat m n)
    (i : Fin m) (j' : Fin n) (j : Fin n) :
    pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
         (Mat.flatten W) (finProdFinEquiv (i, j')) j =
      if j = j' then x i else 0 := by
  rw [show (fun v : Vec (m * n) => dense (Mat.unflatten v) b x) =
      fun v => (fun k => ∑ i' : Fin m, x i' * v (finProdFinEquiv (i', k))) + b from rfl,
    pdiv_of_affine (fun v k => ∑ i' : Fin m, x i' * v (finProdFinEquiv (i', k))) b
      (fun _ _ => by funext; simp [mul_add, Finset.sum_add_distrib])
      (fun _ _ => by funext; simp [Finset.mul_sum, mul_left_comm])]
  rcases eq_or_ne j j' with h | h
  · subst h; simp [Prod.ext_iff]
  · simp [Prod.ext_iff, h]

/-- Dense VJP — proved. -/
noncomputable def dense_has_vjp {m n : Nat} (W : Mat m n) (b : Vec n) :
    HasVJP (dense W b) where
  backward := fun _x dy => Mat.mulVec W dy
  correct := by
    intro x dy i
    simp only [Mat.mulVec]
    congr 1; ext j; rw [pdiv_dense]

/-- The Chapter-1 demo model: a linear classifier is a single dense layer. -/
noncomputable def mnistLinear {m n : Nat} (W : Mat m n) (b : Vec n) : Vec m → Vec n :=
  dense W b

/-- Whole-model VJP contract for the linear classifier — the degenerate
simplest case of the per-architecture `*_has_vjp_correct` capstones, built
straight from the Chapter-1 kit. -/
theorem mnistLinear_has_vjp_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin m) :
    (dense_has_vjp W b).backward x dy i =
      ∑ j : Fin n, pdiv (mnistLinear W b) x i j * dy j :=
  (dense_has_vjp W b).correct x dy i

/-- **Dense is everywhere differentiable.** `dense W b` is affine in
    `x`, hence smooth; this is the underlying `Differentiable ℝ`
    statement that `vjp_comp_at` needs when composing through dense
    layers. -/
@[fun_prop]
theorem dense_differentiable {m n : Nat} (W : Mat m n) (b : Vec n) :
    Differentiable ℝ (dense W b) := by
  unfold dense; fun_prop

/-- **Dense weight gradient is the outer product** — theorem (Phase 7).

    `Mat.outer x dy` is the cotangent-contracted Jacobian of `dense(W, b, x)`
    with respect to `W`, at every index. This promotes the previous vacuous
    `rfl` about `Mat.outer` into a real theorem connecting the outer product
    to the actual weight gradient of `dense`.

    `(Mat.outer x dy) i j = ∑ k, pdiv (…) (Mat.flatten W) (fPF (i, j)) k · dy k` -/
theorem dense_weight_grad_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin m) (j : Fin n) :
    Mat.outer x dy i j =
      ∑ k : Fin n,
        pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
             (Mat.flatten W) (finProdFinEquiv (i, j)) k * dy k := by
  simp_rw [pdiv_dense_W]
  -- Σ k, (if k = j then x i else 0) * dy k  collapses to x i * dy j
  rw [Finset.sum_eq_single j
      (fun k _ hne => by rw [ite_eq_right hne]; ring)
      (fun h => absurd (Finset.mem_univ j) h)]
  simp [Mat.outer]

/-- **Dense bias gradient is identity** — theorem (Phase 7).

    `∂ dense(W, b, x)_j / ∂ b_{j'} = δ(j, j')`, so the bias backward is
    just `dy` itself. Derived from `pdiv_add` + `pdiv_const` + `pdiv_id`
    — no new axiom. -/
theorem pdiv_dense_b {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)
    (i j : Fin n) :
    pdiv (fun b' : Vec n => dense W b' x) b i j = if i = j then 1 else 0 := by
  -- Rewrite `fun b' => dense W b' x` as `(constant in b') + (identity on b')`.
  have hDec : (fun b' : Vec n => dense W b' x) =
              (fun b' k => (fun (_ : Vec n) (k' : Fin n) =>
                              ∑ i' : Fin m, x i' * W i' k') b' k +
                           (fun (y : Vec n) => y) b' k) := by
    funext b' k; rfl
  have h_const_diff : DifferentiableAt ℝ
      (fun (_ : Vec n) (k' : Fin n) => ∑ i' : Fin m, x i' * W i' k') b :=
    differentiableAt_const _
  have h_id_diff : DifferentiableAt ℝ (fun y : Vec n => y) b :=
    differentiableAt_id
  rw [hDec, pdiv_add _ _ _ h_const_diff h_id_diff, pdiv_const, pdiv_id]
  ring

theorem dense_bias_grad_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin n) :
    dy i =
      ∑ j : Fin n, pdiv (fun b' : Vec n => dense W b' x) b i j * dy j := by
  simp_rw [pdiv_dense_b W b x]
  rw [Finset.sum_eq_single i
      (fun j _ hne => by rw [ite_eq_right (Ne.symm hne)]; ring)
      (fun h => absurd (Finset.mem_univ i) h)]
  simp

/-- **Dense weight backward** — named accessor.
    `dW = x ⊗ dy` (outer product). -/
noncomputable def dense_weight_grad {m n : Nat}
    (x : Vec m) (dy : Vec n) : Mat m n :=
  Mat.outer x dy

/-- **Dense bias backward** — named accessor. `db = dy`. -/
def dense_bias_grad {n : Nat} (dy : Vec n) : Vec n := dy

-- ════════════════════════════════════════════════════════════════
-- § ReLU:  y = max(x, 0)
-- ════════════════════════════════════════════════════════════════

noncomputable def relu (n : Nat) (x : Vec n) : Vec n :=
  fun i => if x i > 0 then x i else 0

/-- **ReLU's local linear part at a smooth point** — the diagonal
    indicator CLM. At each coordinate `k`, projects to `y k` if
    `x k > 0`, otherwise zero. Two smooth points with the same sign
    pattern share this same CLM. -/
noncomputable def reluLinearPart (n : Nat) (x : Vec n) : Vec n →L[ℝ] Vec n :=
  ContinuousLinearMap.pi fun k =>
    if x k > 0 then ContinuousLinearMap.proj k else (0 : Vec n →L[ℝ] ℝ)

@[simp] theorem reluLinearPart_apply (n : Nat) (x y : Vec n) (k : Fin n) :
    reluLinearPart n x y k = if x k > 0 then y k else 0 := by
  show (ContinuousLinearMap.pi (fun k' =>
          if x k' > 0 then ContinuousLinearMap.proj k'
                      else (0 : Vec n →L[ℝ] ℝ))) y k = _
  rw [ContinuousLinearMap.pi_apply]
  by_cases hxk : x k > 0
  · rw [ite_eq_left hxk, ite_eq_left hxk]; rfl
  · rw [ite_eq_right hxk, ite_eq_right hxk]; rfl

/-- **ReLU is differentiable at smooth points.** Near `x` every coordinate keeps its sign
    (finitely many strict inequalities persist, `Filter.eventually_all`), so `relu n` agrees
    with `reluLinearPart n x` on a neighbourhood. `EventuallyEq` promotes the CLM's
    `HasFDerivAt` to ReLU's. -/
theorem relu_hasFDerivAt (n : Nat) (x : Vec n) (h_smooth : ∀ k, x k ≠ 0) :
    HasFDerivAt (relu n) (reluLinearPart n x) x := by
  refine (reluLinearPart n x).hasFDerivAt.congr_of_eventuallyEq ?_
  have hsign : ∀ k, ∀ᶠ y in nhds x, (y k > 0 ↔ x k > 0) := fun k => by
    have ht := (continuous_apply k).continuousAt.tendsto (x := x)
    rcases (h_smooth k).lt_or_gt with h | h
    · filter_upwards [ht.eventually (eventually_lt_nhds h)] with y hy
      exact iff_of_false hy.not_gt h.not_gt
    · filter_upwards [ht.eventually (eventually_gt_nhds h)] with y hy
      exact iff_of_true hy h
  filter_upwards [Filter.eventually_all.2 hsign] with y hy
  funext k; simp only [relu, reluLinearPart_apply, hy k]

/-- **ReLU is `DifferentiableAt` at smooth points.** Corollary of
    `relu_hasFDerivAt`; lets `vjp_comp_at` chain through ReLU. -/
@[fun_prop]
theorem relu_differentiableAt_of_smooth (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0) : DifferentiableAt ℝ (relu n) x :=
  (relu_hasFDerivAt n x h_smooth).differentiableAt

/-- **ReLU partial derivative** — proved via `relu_hasFDerivAt` and
    direct evaluation at `basisVec i`. -/
theorem pdiv_relu (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0)
    (i j : Fin n) :
    pdiv (relu n) x i j =
      if i = j then (if x i > 0 then 1 else 0) else 0 := by
  rcases Nat.eq_zero_or_pos n with hn0 | hn_pos
  · subst hn0; exact i.elim0
  unfold pdiv
  rw [(relu_hasFDerivAt n x h_smooth).fderiv, reluLinearPart_apply, basisVec_apply]
  by_cases hij : i = j
  · subst hij; rw [ite_eq_left rfl, ite_eq_left rfl]
  · rw [ite_eq_right (fun h : j = i => hij h.symm), ite_eq_right hij]
    by_cases hxj : x j > 0
    · rw [ite_eq_left hxj]
    · rw [ite_eq_right hxj]

/-- **ReLU bundled VJP — canonical (junk-at-kink) witness.**

    `HasVJP.correct` is satisfied by the canonical pdiv-derived backward:
    at smooth points it is the diagonal indicator (per `pdiv_relu`); at
    points where some coordinate is zero, `pdiv (relu n) x` agrees with
    `fderiv`'s junk default of `0`, so the canonical backward is `0`
    there too — and `correct` holds by `rfl`.

    The codegen (`MlirCodegen.lean`) emits the standard subgradient
    formula `if x > 0 then dy else 0` instead, which agrees with the
    canonical witness at smooth points and differs at the kinks (the
    convention `relu'(0) := 0` used by every ML framework). The
    smooth-point agreement is formal: see
    `relu_codegen_matches_canonical` below. The Lean-vs-codegen gap at
    the kinks is the codegen trust boundary — see
    `LeanMlir/Proofs/README.md`. -/
noncomputable def relu_has_vjp (n : Nat) : HasVJP (relu n) where
  backward x dy i := ∑ j : Fin n, pdiv (relu n) x i j * dy j
  correct _ _ _  := rfl

/-- **Bridge: `relu_has_vjp`'s canonical backward matches the codegen
    formula at smooth points.**

    At any point where no coordinate of `x` is zero, the canonical
    `pdiv`-derived backward `∑ j, pdiv (relu n) x i j * dy j` collapses
    to the framework subgradient `if x i > 0 then dy i else 0` that
    `MlirCodegen.lean` actually emits. Closes the smooth-point half of
    the codegen trust boundary — what's left is just the kink
    convention. -/
theorem relu_codegen_matches_canonical (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0) (dy : Vec n) (i : Fin n) :
    (relu_has_vjp n).backward x dy i = if x i > 0 then dy i else 0 := by
  show ∑ j : Fin n, pdiv (relu n) x i j * dy j = _
  simp_rw [pdiv_relu n x h_smooth i]
  rw [Finset.sum_eq_single i
      (fun j _ hne => by rw [ite_eq_right (Ne.symm hne)]; ring)
      (fun h => absurd (Finset.mem_univ i) h)]
  rw [ite_eq_left rfl]
  by_cases hx : x i > 0
  · rw [ite_eq_left hx, ite_eq_left hx]; ring
  · rw [ite_eq_right hx, ite_eq_right hx]; ring

/-- **Diagonal-indicator restatement of the smooth-point bridge.**
    `relu_has_vjp.backward x dy i = 1_{x i > 0} · dy i` at smooth
    points — same content as `relu_codegen_matches_canonical`,
    factored as ``(indicator) · dy i`` for downstream use. -/
theorem relu_canonical_diagonal (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0) (dy : Vec n) (i : Fin n) :
    (relu_has_vjp n).backward x dy i =
    (if x i > 0 then (1 : ℝ) else 0) * dy i := by
  rw [relu_codegen_matches_canonical n x h_smooth dy i]
  by_cases hx : x i > 0
  · rw [ite_eq_left hx, ite_eq_left hx]; ring
  · rw [ite_eq_right hx, ite_eq_right hx]; ring

/-- **ReLU pointwise VJP — no canonical-witness escape.**

    Constructs `HasVJPAt (relu n) x` at a smooth point. The backward
    is the codegen-shape `if x i > 0 then dy i else 0` directly; the
    `correct` field is a real proof via `pdiv_relu` (the smooth-point
    Jacobian) + sum-collapse, not `rfl`. -/
noncomputable def relu_has_vjp_at (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0) : HasVJPAt (relu n) x where
  backward dy i := if x i > 0 then dy i else 0
  correct := by
    intro dy i
    simp_rw [pdiv_relu n x h_smooth]
    rw [Finset.sum_eq_single i
        (fun j _ hne => by rw [ite_eq_right (Ne.symm hne)]; ring)
        (fun h => absurd (Finset.mem_univ i) h)]
    rw [ite_eq_left rfl]
    by_cases hxi : x i > 0
    · rw [ite_eq_left hxi, ite_eq_left hxi]; ring
    · rw [ite_eq_right hxi, ite_eq_right hxi]; ring

-- ════════════════════════════════════════════════════════════════
-- § Softmax Cross-Entropy Loss
-- ════════════════════════════════════════════════════════════════

noncomputable def softmax (c : Nat) (z : Vec c) : Vec c :=
  let e : Vec c := fun j => Real.exp (z j)
  let total := ∑ k : Fin c, e k
  fun j => e j / total

noncomputable def oneHot (c : Nat) (label : Fin c) : Vec c :=
  fun j => if j = label then 1 else 0

noncomputable def crossEntropy (c : Nat) (logits : Vec c) (label : Fin c) : ℝ :=
  -(Real.log (softmax c logits label))

-- `softmaxCE_grad` is proved in `Attention.lean` (after `pdiv_softmax` is
-- available). Its statement and proof live there; this file keeps only
-- `softmax`, `oneHot`, and `crossEntropy` definitions used downstream.

-- ════════════════════════════════════════════════════════════════
-- § MLP Composition
-- ════════════════════════════════════════════════════════════════

noncomputable def mlpForward {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) :
    Vec d₀ → Vec d₃ :=
  dense W₂ b₂ ∘ relu d₂ ∘ dense W₁ b₁ ∘ relu d₁ ∘ dense W₀ b₀

/-- **MLP composition VJP — canonical witness.**

    The MLP forward composes `dense W b` (everywhere `Differentiable`)
    with `relu` (non-`Differentiable` at the kinks). `vjp_comp` would
    require `Differentiable ℝ (relu n)`, which doesn't hold globally,
    so the chain-rule route is blocked. The canonical pdiv-derived
    backward inhabits `HasVJP.correct` directly via `rfl` — the
    codegen substitutes the subgradient formula at the kinks (see
    `LeanMlir/Proofs/README.md` for the trust-boundary discussion). -/
noncomputable def mlp_has_vjp {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) :
    HasVJP (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) where
  backward x dy i :=
    ∑ j : Fin d₃, pdiv (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x i j * dy j
  correct _ _ _  := rfl

/-- **MLP pointwise VJP — no canonical-witness escape.**

    Constructs `HasVJPAt (mlpForward …) x` by chaining `vjp_comp_at`
    through `dense → relu_at → dense → relu_at → dense`. Requires the
    intermediate pre-activations `dense W₀ b₀ x` and `dense W₁ b₁ z₀`
    to avoid zero (no coordinate ties the ReLU kink) — exactly the
    "smooth input" condition. Replaces the vacuous
    `mlp_has_vjp.correct := rfl` with a real chain-rule proof at
    smooth inputs. -/
noncomputable def mlp_has_vjp_at {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (x : Vec d₀)
    (h_smooth_0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (h_smooth_1 : ∀ k, dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) k ≠ 0) :
    HasVJPAt (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x := by
  unfold mlpForward
  -- relu d₁ ∘ dense W₀ b₀
  have step1 : HasVJPAt (relu d₁ ∘ dense W₀ b₀) x :=
    vjp_comp_at (dense W₀ b₀) (relu d₁) x
      ((dense_differentiable W₀ b₀) x)
      (relu_differentiableAt_of_smooth d₁ _ h_smooth_0)
      ((dense_has_vjp W₀ b₀).toHasVJPAt x)
      (relu_has_vjp_at d₁ _ h_smooth_0)
  have step1_diff : DifferentiableAt ℝ (relu d₁ ∘ dense W₀ b₀) x :=
    (relu_differentiableAt_of_smooth d₁ _ h_smooth_0).comp x
      ((dense_differentiable W₀ b₀) x)
  -- dense W₁ b₁ ∘ relu d₁ ∘ dense W₀ b₀
  have step2 : HasVJPAt (dense W₁ b₁ ∘ relu d₁ ∘ dense W₀ b₀) x :=
    vjp_comp_at (relu d₁ ∘ dense W₀ b₀) (dense W₁ b₁) x
      step1_diff
      ((dense_differentiable W₁ b₁) _)
      step1
      ((dense_has_vjp W₁ b₁).toHasVJPAt _)
  have step2_diff : DifferentiableAt ℝ (dense W₁ b₁ ∘ relu d₁ ∘ dense W₀ b₀) x :=
    ((dense_differentiable W₁ b₁) _).comp x step1_diff
  -- relu d₂ ∘ dense W₁ b₁ ∘ relu d₁ ∘ dense W₀ b₀
  have step3 : HasVJPAt (relu d₂ ∘ dense W₁ b₁ ∘ relu d₁ ∘ dense W₀ b₀) x :=
    vjp_comp_at (dense W₁ b₁ ∘ relu d₁ ∘ dense W₀ b₀) (relu d₂) x
      step2_diff
      (relu_differentiableAt_of_smooth d₂ _ h_smooth_1)
      step2
      (relu_has_vjp_at d₂ _ h_smooth_1)
  have step3_diff : DifferentiableAt ℝ
      (relu d₂ ∘ dense W₁ b₁ ∘ relu d₁ ∘ dense W₀ b₀) x :=
    (relu_differentiableAt_of_smooth d₂ _ h_smooth_1).comp x step2_diff
  -- dense W₂ b₂ ∘ (above)
  exact vjp_comp_at (relu d₂ ∘ dense W₁ b₁ ∘ relu d₁ ∘ dense W₀ b₀) (dense W₂ b₂) x
    step3_diff
    ((dense_differentiable W₂ b₂) _)
    step3
    ((dense_has_vjp W₂ b₂).toHasVJPAt _)

/-! ## Public correctness theorems for the canonical-witness defs

Each `_has_vjp` def above bundles a backward function with a `.correct`
field; these `_correct` theorems expose that field as a top-level
proposition so consumers (downstream code, `tests/comparator/`,
doc-gen4) can refer to the contract directly without reaching into
record internals. -/

/-- **Public correctness theorem for `relu_has_vjp`**: the canonical
witness's backward equals the `pdiv`-contracted Jacobian by definition. -/
theorem relu_has_vjp_correct (n : Nat) (x : Vec n) (dy : Vec n) (i : Fin n) :
    (relu_has_vjp n).backward x dy i =
    ∑ j : Fin n, pdiv (relu n) x i j * dy j :=
  (relu_has_vjp n).correct x dy i

/-- **Public correctness theorem for `mlp_has_vjp`**: same pattern as
`relu_has_vjp_correct`, lifted to the three-layer MLP forward. -/
theorem mlp_has_vjp_correct {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (x : Vec d₀) (dy : Vec d₃) (i : Fin d₀) :
    (mlp_has_vjp W₀ b₀ W₁ b₁ W₂ b₂).backward x dy i =
    ∑ j : Fin d₃, pdiv (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x i j * dy j :=
  (mlp_has_vjp W₀ b₀ W₁ b₁ W₂ b₂).correct x dy i

/-- **Public correctness theorem for `relu_has_vjp_at`** — the
pointwise (smooth-input) variant. Unlike `relu_has_vjp_correct`, this
wrapper's underlying `.correct` field is a real proof
(`pdiv_relu` + sum-collapse), not `rfl`; the wrapper exposes it as
a top-level proposition for `tests/comparator/` re-verification. -/
theorem relu_has_vjp_at_correct (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0) (dy : Vec n) (i : Fin n) :
    (relu_has_vjp_at n x h_smooth).backward dy i =
    ∑ j : Fin n, pdiv (relu n) x i j * dy j :=
  (relu_has_vjp_at n x h_smooth).correct dy i

/-- **Public correctness theorem for `mlp_has_vjp_at`** — the
pointwise variant composed via `vjp_comp_at` through
`dense → relu_at → dense → relu_at → dense`. The underlying
`.correct` field chains real chain-rule proofs (no `rfl` escape at
the ReLU kinks). -/
theorem mlp_has_vjp_at_correct {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (x : Vec d₀)
    (h_smooth_0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (h_smooth_1 : ∀ k, dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) k ≠ 0)
    (dy : Vec d₃) (i : Fin d₀) :
    (mlp_has_vjp_at W₀ b₀ W₁ b₁ W₂ b₂ x h_smooth_0 h_smooth_1).backward dy i =
    ∑ j : Fin d₃, pdiv (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x i j * dy j :=
  (mlp_has_vjp_at W₀ b₀ W₁ b₁ W₂ b₂ x h_smooth_0 h_smooth_1).correct dy i

end Proofs
