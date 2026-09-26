import LeanMlir.Proofs.Foundation.Tensor
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# MLP VJP Proofs

Formal VJP correctness for the layers of a 3-layer MLP.
All definitions over `ℝ`, proofs use Mathlib's `Finset.sum`.
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Dense Layer:  y = xW + b
-- ════════════════════════════════════════════════════════════════

/-- Affine layer `x ↦ xW + b` in the row-vector convention: `W : Mat m n` maps `Vec m → Vec n`,
    and `W i j` connects input `i` to output `j`. -/
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
noncomputable def denseHasVJP {m n : Nat} (W : Mat m n) (b : Vec n) :
    HasVJP (dense W b) where
  backward := fun _x dy => Mat.mulVec W dy
  correct := by
    intro x dy i
    simp only [Mat.mulVec]
    congr 1; ext j; rw [pdiv_dense]

/-- The Chapter-1 demo model: a linear classifier is a single dense layer. -/
noncomputable def mnistLinear {m n : Nat} (W : Mat m n) (b : Vec n) : Vec m → Vec n :=
  dense W b

/-- `denseHasVJP`'s `correct` field, restated for `mnistLinear` so it can be cited by
name. -/
theorem mnistLinearHasVJP_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin m) :
    (denseHasVJP W b).backward x dy i =
      ∑ j : Fin n, pdiv (mnistLinear W b) x i j * dy j :=
  (denseHasVJP W b).correct x dy i

/-- **Dense is everywhere differentiable.** `dense W b` is affine in
    `x`, hence smooth; this is the underlying `Differentiable ℝ`
    statement that `vjpCompAt` needs when composing through dense
    layers. -/
@[fun_prop]
theorem dense_differentiable {m n : Nat} (W : Mat m n) (b : Vec n) :
    Differentiable ℝ (dense W b) := by
  unfold dense; fun_prop

/-- **Dense weight gradient is the outer product.**

    `Mat.outer x dy` is the cotangent-contracted Jacobian of `dense(W, b, x)`
    with respect to `W`, at every index.

    `(Mat.outer x dy) i j = ∑ k, pdiv (…) (Mat.flatten W) (fPF (i, j)) k · dy k` -/
theorem denseWeightGrad_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin m) (j : Fin n) :
    Mat.outer x dy i j =
      ∑ k : Fin n,
        pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
             (Mat.flatten W) (finProdFinEquiv (i, j)) k * dy k := by
  simp [pdiv_dense_W, Mat.outer]

/-- **Dense bias gradient is identity.**

    `∂ dense(W, b, x)_j / ∂ b_{j'} = δ(j, j')`, so the bias backward is
    just `dy` itself: `b' ↦ dense W b' x` is the identity plus a constant (`pdiv_of_affine`). -/
theorem pdiv_dense_b {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)
    (i j : Fin n) :
    pdiv (fun b' : Vec n => dense W b' x) b i j = if i = j then 1 else 0 := by
  rw [show (fun b' : Vec n => dense W b' x) = fun v => (fun w => w) v + dense W 0 x by
        funext v k; simp [dense, add_comm],
      pdiv_of_affine (fun w => w) _ (fun _ _ => rfl) (fun _ _ => rfl)]
  simp [eq_comm]

theorem denseBiasGrad_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin n) :
    dy i =
      ∑ j : Fin n, pdiv (fun b' : Vec n => dense W b' x) b i j * dy j := by
  simp [pdiv_dense_b]

/-- **Dense weight backward** — named accessor.
    `dW = x ⊗ dy` (outer product). -/
noncomputable def denseWeightGrad {m n : Nat}
    (x : Vec m) (dy : Vec n) : Mat m n :=
  Mat.outer x dy

/-- **Dense bias backward** — named accessor. `db = dy`. -/
def denseBiasGrad {n : Nat} (dy : Vec n) : Vec n := dy

-- ════════════════════════════════════════════════════════════════
-- § ReLU:  y = max(x, 0)
-- ════════════════════════════════════════════════════════════════

/-- ReLU, coordinatewise: `x i` if `x i > 0`, else `0`. -/
noncomputable def relu (n : Nat) (x : Vec n) : Vec n :=
  fun i => if x i > 0 then x i else 0

/-- `relu` is `max · 0`, coordinatewise. -/
theorem relu_apply_eq_max {n : Nat} (x : Vec n) (i : Fin n) : relu n x i = max (x i) 0 :=
  (max_def_lt' (x i) 0).symm

/-- ReLU output is always nonnegative. -/
theorem relu_nonneg (n : Nat) (v : Vec n) (k : Fin n) : 0 ≤ relu n v k := by
  rw [relu_apply_eq_max]; exact le_max_right _ _

/-- ReLU is entrywise 1-Lipschitz — what lets a drift (or a rounding error) pass through a kinked
    layer unamplified. -/
theorem relu_entry_lipschitz (n : Nat) (u v : Vec n) (k : Fin n) :
    |relu n u k - relu n v k| ≤ |u k - v k| := by
  rw [relu_apply_eq_max, relu_apply_eq_max]; exact abs_max_sub_max_le_abs _ _ _

/-- **The Jacobian of a diagonal 0/1 mask.** If `f` has, at `x`, the derivative that keeps
    coordinate `k` when `p k` and zeroes it otherwise, its `pdiv` is the diagonal indicator of `p`.
    `relu` and `relu6` at a smooth point are both this. -/
theorem pdiv_of_hasFDerivAt_mask {n : Nat} (f : Vec n → Vec n) (x : Vec n) (p : Fin n → Prop)
    [DecidablePred p]
    (hf : HasFDerivAt f (ContinuousLinearMap.pi fun k =>
        if p k then ContinuousLinearMap.proj k else (0 : Vec n →L[ℝ] ℝ)) x) (i j : Fin n) :
    pdiv f x i j = if i = j then (if p i then 1 else 0) else 0 := by
  unfold pdiv
  rw [hf.fderiv, ContinuousLinearMap.pi_apply]
  by_cases hij : i = j
  · subst hij; split_ifs <;> simp_all [basisVec_apply]
  · split_ifs <;> simp [basisVec_apply, Ne.symm hij]

/-- **ReLU's local linear part at a smooth point** — the diagonal
    indicator CLM. At each coordinate `k`, projects to `y k` if
    `x k > 0`, otherwise zero. Two smooth points with the same sign
    pattern share this same CLM. -/
noncomputable def reluLinearPart (n : Nat) (x : Vec n) : Vec n →L[ℝ] Vec n :=
  ContinuousLinearMap.pi fun k =>
    if x k > 0 then ContinuousLinearMap.proj k else (0 : Vec n →L[ℝ] ℝ)

@[simp] theorem reluLinearPart_apply (n : Nat) (x y : Vec n) (k : Fin n) :
    reluLinearPart n x y k = if x k > 0 then y k else 0 := by
  rw [reluLinearPart, ContinuousLinearMap.pi_apply]; split_ifs <;> rfl

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
    `relu_hasFDerivAt`; lets `vjpCompAt` chain through ReLU. -/
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
      if i = j then (if x i > 0 then 1 else 0) else 0 :=
  pdiv_of_hasFDerivAt_mask _ x _ (relu_hasFDerivAt n x h_smooth) i j

/-- **ReLU bundled VJP — canonical (junk-at-kink) witness.**

    `HasVJP.correct` is satisfied by the canonical pdiv-derived backward:
    at smooth points it is the diagonal indicator (per `pdiv_relu`); at
    points where some coordinate is zero, `pdiv (relu n) x` agrees with
    `fderiv`'s junk default of `0`, so the canonical backward is `0`
    there too — the witness is `HasVJP.canonical`, whose `correct` holds by `rfl`.

    The codegen (`MlirCodegen.lean`) emits the standard subgradient
    formula `if x > 0 then dy else 0` instead, which agrees with the
    canonical witness at smooth points and differs at the kinks (the
    convention `relu'(0) := 0` used by every ML framework). The
    smooth-point agreement is formal: see
    `relu_codegen_matches_canonical` below. The Lean-vs-codegen gap at
    the kinks is the codegen trust boundary — see
    `LeanMlir/Proofs/README.md`. -/
noncomputable def reluHasVJP (n : Nat) : HasVJP (relu n) := HasVJP.canonical _

/-- **Bridge: `reluHasVJP`'s canonical backward matches the codegen
    formula at smooth points.**

    At any point where no coordinate of `x` is zero, the canonical
    `pdiv`-derived backward `∑ j, pdiv (relu n) x i j * dy j` collapses
    to the framework subgradient `if x i > 0 then dy i else 0` that
    `MlirCodegen.lean` actually emits. Closes the smooth-point half of
    the codegen trust boundary — what's left is just the kink
    convention. -/
theorem relu_codegen_matches_canonical (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0) (dy : Vec n) (i : Fin n) :
    (reluHasVJP n).backward x dy i = if x i > 0 then dy i else 0 := by
  show ∑ j : Fin n, pdiv (relu n) x i j * dy j = _
  simp_rw [pdiv_relu n x h_smooth i]; simp

/-- **Diagonal-indicator restatement of the smooth-point bridge.**
    `reluHasVJP.backward x dy i = 1_{x i > 0} · dy i` at smooth
    points — same content as `relu_codegen_matches_canonical`,
    factored as ``(indicator) · dy i`` for downstream use. -/
theorem relu_canonical_diagonal (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0) (dy : Vec n) (i : Fin n) :
    (reluHasVJP n).backward x dy i =
    (if x i > 0 then (1 : ℝ) else 0) * dy i := by
  rw [relu_codegen_matches_canonical n x h_smooth dy i, ite_mul, one_mul, zero_mul]

/-- **ReLU pointwise VJP — no canonical-witness escape.**

    Constructs `HasVJPAt (relu n) x` at a smooth point. The backward
    is the codegen-shape `if x i > 0 then dy i else 0` directly; the
    `correct` field is a real proof via `pdiv_relu` (the smooth-point
    Jacobian) + sum-collapse, not `rfl`. -/
noncomputable def reluHasVJPAt (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0) : HasVJPAt (relu n) x where
  backward dy i := if x i > 0 then dy i else 0
  correct dy i := by simp_rw [pdiv_relu n x h_smooth]; simp

-- ════════════════════════════════════════════════════════════════
-- § Softmax Cross-Entropy Loss
-- ════════════════════════════════════════════════════════════════

/-- Softmax over `c` classes: `exp (z j) / Σ_k exp (z k)`. -/
noncomputable def softmax (c : Nat) (z : Vec c) : Vec c :=
  let e : Vec c := fun j => Real.exp (z j)
  let total := ∑ k : Fin c, e k
  fun j => e j / total

/-- The one-hot vector of `label` over `c` classes. -/
noncomputable def oneHot (c : Nat) (label : Fin c) : Vec c :=
  fun j => if j = label then 1 else 0

/-- Cross-entropy at a hard label: `−log (softmax logits)_label`. -/
noncomputable def crossEntropy (c : Nat) (logits : Vec c) (label : Fin c) : ℝ :=
  -(Real.log (softmax c logits label))

theorem softmax_apply (c : Nat) (z : Vec c) (j : Fin c) :
    softmax c z j = Real.exp (z j) / ∑ k : Fin c, Real.exp (z k) := rfl

theorem oneHot_apply (c : Nat) (label j : Fin c) :
    oneHot c label j = if j = label then 1 else 0 := rfl

theorem crossEntropy_def (c : Nat) (logits : Vec c) (label : Fin c) :
    crossEntropy c logits label = -(Real.log (softmax c logits label)) := rfl

-- `softmaxCE_grad` (and `pdiv_softmax`, `softmaxHasVJP`) are in `Softmax.lean`, which
-- imports this file; this file keeps the `softmax`, `oneHot` and `crossEntropy` definitions.

-- ════════════════════════════════════════════════════════════════
-- § MLP Composition
-- ════════════════════════════════════════════════════════════════

/-- The three-layer MLP forward: `dense → relu → dense → relu → dense`. -/
noncomputable def mlpForward {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) :
    Vec d₀ → Vec d₃ :=
  dense W₂ b₂ ∘ relu d₂ ∘ dense W₁ b₁ ∘ relu d₁ ∘ dense W₀ b₀

/-- **MLP composition VJP — canonical witness.**

    The MLP forward composes `dense W b` (everywhere `Differentiable`)
    with `relu` (non-`Differentiable` at the kinks). `vjpComp` would
    require `Differentiable ℝ (relu n)`, which doesn't hold globally,
    so the chain-rule route is blocked. The canonical pdiv-derived
    backward inhabits `HasVJP.correct` directly via `rfl` — the
    codegen substitutes the subgradient formula at the kinks (see
    `LeanMlir/Proofs/README.md` for the trust-boundary discussion). -/
noncomputable def mlpHasVJP {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) :
    HasVJP (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) := HasVJP.canonical _

/-- **MLP pointwise VJP — no canonical-witness escape.**

    Constructs `HasVJPAt (mlpForward …) x` by chaining `vjpCompAt`
    through `dense → relu_at → dense → relu_at → dense`. Requires the
    intermediate pre-activations `dense W₀ b₀ x` and `dense W₁ b₁ z₀`
    to avoid zero (no coordinate ties the ReLU kink) — exactly the
    "smooth input" condition. Unlike the canonical `mlpHasVJP`, the
    backward here is built by the chain rule. -/
noncomputable def mlpHasVJPAt {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (x : Vec d₀)
    (h_smooth_0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (h_smooth_1 : ∀ k, dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) k ≠ 0) :
    HasVJPAt (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x :=
  let dn := fun {a c : Nat} (W : Mat a c) (b : Vec c) (y : Vec a) =>
    (⟨(denseHasVJP W b).toHasVJPAt y, dense_differentiable W b y⟩ :
      HasVJPDiffAt (dense W b) y)
  (vjpCompDiffAt _ _ x
    (vjpCompDiffAt _ _ x
      (vjpCompDiffAt _ _ x
        (vjpCompDiffAt _ _ x (dn W₀ b₀ x)
          ⟨reluHasVJPAt d₁ _ h_smooth_0, relu_differentiableAt_of_smooth d₁ _ h_smooth_0⟩)
        (dn W₁ b₁ _))
      ⟨reluHasVJPAt d₂ _ h_smooth_1, relu_differentiableAt_of_smooth d₂ _ h_smooth_1⟩)
    (dn W₂ b₂ _)).fst

/-! ## Public correctness theorems for the canonical-witness defs

Each `HasVJP` def above bundles a backward function with a `.correct`
field; these `_correct` theorems expose that field as a top-level
proposition so consumers (downstream code, `tests/comparator/`,
doc-gen4) can refer to the contract directly without reaching into
record internals. -/

/-- **Public correctness theorem for `reluHasVJP`**: the canonical
witness's backward equals the `pdiv`-contracted Jacobian by definition. -/
theorem reluHasVJP_correct (n : Nat) (x : Vec n) (dy : Vec n) (i : Fin n) :
    (reluHasVJP n).backward x dy i =
    ∑ j : Fin n, pdiv (relu n) x i j * dy j :=
  (reluHasVJP n).correct x dy i

/-- **Public correctness theorem for `mlpHasVJP`**: same pattern as
`reluHasVJP_correct`, lifted to the three-layer MLP forward. -/
theorem mlpHasVJP_correct {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (x : Vec d₀) (dy : Vec d₃) (i : Fin d₀) :
    (mlpHasVJP W₀ b₀ W₁ b₁ W₂ b₂).backward x dy i =
    ∑ j : Fin d₃, pdiv (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x i j * dy j :=
  (mlpHasVJP W₀ b₀ W₁ b₁ W₂ b₂).correct x dy i

/-- **Public correctness theorem for `reluHasVJPAt`** — the
pointwise (smooth-input) variant. Unlike `reluHasVJP_correct`, this
wrapper's underlying `.correct` field is a real proof
(`pdiv_relu` + sum-collapse), not `rfl`; the wrapper exposes it as
a top-level proposition for `tests/comparator/` re-verification. -/
theorem reluHasVJPAt_correct (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0) (dy : Vec n) (i : Fin n) :
    (reluHasVJPAt n x h_smooth).backward dy i =
    ∑ j : Fin n, pdiv (relu n) x i j * dy j :=
  (reluHasVJPAt n x h_smooth).correct dy i

/-- **Public correctness theorem for `mlpHasVJPAt`** — the
pointwise variant composed via `vjpCompAt` through
`dense → relu_at → dense → relu_at → dense`. The underlying
`.correct` field chains real chain-rule proofs (no `rfl` escape at
the ReLU kinks). -/
theorem mlpHasVJPAt_correct {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (x : Vec d₀)
    (h_smooth_0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (h_smooth_1 : ∀ k, dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) k ≠ 0)
    (dy : Vec d₃) (i : Fin d₀) :
    (mlpHasVJPAt W₀ b₀ W₁ b₁ W₂ b₂ x h_smooth_0 h_smooth_1).backward dy i =
    ∑ j : Fin d₃, pdiv (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x i j * dy j :=
  (mlpHasVJPAt W₀ b₀ W₁ b₁ W₂ b₂ x h_smooth_0 h_smooth_1).correct dy i

-- ════════════════════════════════════════════════════════════════
-- § ReLU6  y = min(max(x,0), 6)   (MobileNetV2 activation)
-- ════════════════════════════════════════════════════════════════

/-- ReLU6, coordinatewise: `min (max (x i) 0) 6` (MobileNetV2's activation). -/
noncomputable def relu6 (n : Nat) (x : Vec n) : Vec n :=
  fun i => min (max (x i) 0) 6

/-- ReLU6's local linear part at a smooth point: projects to `y k` when
    `0 < x k < 6`, otherwise zero. -/
noncomputable def relu6LinearPart (n : Nat) (x : Vec n) : Vec n →L[ℝ] Vec n :=
  ContinuousLinearMap.pi fun k =>
    if 0 < x k ∧ x k < 6 then ContinuousLinearMap.proj k else (0 : Vec n →L[ℝ] ℝ)

@[simp] theorem relu6LinearPart_apply (n : Nat) (x y : Vec n) (k : Fin n) :
    relu6LinearPart n x y k = if 0 < x k ∧ x k < 6 then y k else 0 := by
  rw [relu6LinearPart, ContinuousLinearMap.pi_apply]; split_ifs <;> rfl

theorem relu6_hasFDerivAt (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0 ∧ x k ≠ 6) :
    HasFDerivAt (relu6 n) (relu6LinearPart n x) x := by
  unfold relu6LinearPart
  rw [hasFDerivAt_pi]
  intro k
  -- Each coordinate is locally constant 0, the identity, or constant 6: `x k` sits strictly
  -- inside one of the three pieces and `y k` stays there for `y` near `x`.
  have ht := (continuous_apply k).continuousAt.tendsto (x := x)
  rcases (h_smooth k).1.lt_or_gt with h0 | h0
  · rw [ite_eq_right fun h => h0.not_gt h.1]
    refine (hasFDerivAt_const 0 x).congr_of_eventuallyEq ?_
    filter_upwards [ht.eventually (eventually_lt_nhds h0)] with y hy
    simp [relu6, hy.le]
  rcases (h_smooth k).2.lt_or_gt with h6 | h6
  · rw [ite_eq_left ⟨h0, h6⟩]
    refine (ContinuousLinearMap.proj k : Vec n →L[ℝ] ℝ).hasFDerivAt.congr_of_eventuallyEq ?_
    filter_upwards [ht.eventually (eventually_gt_nhds h0), ht.eventually (eventually_lt_nhds h6)]
      with y hy0 hy6
    simp [relu6, hy0.le, hy6.le]
  · rw [ite_eq_right fun h => h6.not_gt h.2]
    refine (hasFDerivAt_const 6 x).congr_of_eventuallyEq ?_
    filter_upwards [ht.eventually (eventually_gt_nhds h6)] with y hy
    simp [relu6, hy.le, ((show (0 : ℝ) < 6 by norm_num).trans hy).le]

@[fun_prop]
theorem relu6_differentiableAt_of_smooth (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0 ∧ x k ≠ 6) : DifferentiableAt ℝ (relu6 n) x :=
  (relu6_hasFDerivAt n x h_smooth).differentiableAt

theorem pdiv_relu6 (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0 ∧ x k ≠ 6) (i j : Fin n) :
    pdiv (relu6 n) x i j =
      if i = j then (if 0 < x i ∧ x i < 6 then 1 else 0) else 0 :=
  pdiv_of_hasFDerivAt_mask _ x _ (relu6_hasFDerivAt n x h_smooth) i j

noncomputable def relu6HasVJPAt (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0 ∧ x k ≠ 6) : HasVJPAt (relu6 n) x where
  backward dy i := if 0 < x i ∧ x i < 6 then dy i else 0
  correct := by
    intro dy i
    simp_rw [pdiv_relu6 n x h_smooth]; simp

/-- **ReLU6 is the identity inside its window.** Wherever every coordinate is strictly inside
    `(0,6)`, `min (max · 0) 6` does nothing — the step every structural witness takes to collapse a
    relu6 stage to its BatchNorm. Stated at the top level rather than inside a witness namespace:
    it is a fact about the op, and `BatchSeal`'s consumers outlive any one witness. -/
theorem relu6_id_window (n : Nat) (y : Vec n) (hy : ∀ k, 0 < y k ∧ y k < 6) :
    relu6 n y = y := by
  funext k
  simp only [relu6]
  obtain ⟨h0, h6⟩ := hy k
  rw [max_eq_left (le_of_lt h0), min_eq_left (le_of_lt h6)]

/-- **ReLU6 is continuous** — `min (max · 0) 6` coordinatewise. The peer of
    `relu_continuous` (below), for the ray argument of a relu6 net's seal. -/
@[fun_prop]
theorem relu6_continuous (n : Nat) : Continuous (relu6 n) := by
  refine continuous_pi (fun k => ?_)
  exact ((continuous_apply k).max continuous_const).min continuous_const

/-- `relu` is continuous everywhere — it is `max · 0`; only its *derivative* has a kink. -/
@[fun_prop]
theorem relu_continuous (n : Nat) : Continuous (relu n) :=
  continuous_pi fun k => by simp only [relu_apply_eq_max]; exact (continuous_apply k).max continuous_const

/-- ReLU is the identity on a strictly-positive vector. Discharges the
    ReLU-as-identity steps that fold the composition into a plain conv
    stack at a smooth (everywhere-positive) point. -/
theorem relu_id_of_pos {n : Nat} {v : Vec n} (hv : ∀ i, 0 < v i) : relu n v = v := by
  funext i; simp only [relu]; rw [ite_eq_left (hv i)]

end Proofs
