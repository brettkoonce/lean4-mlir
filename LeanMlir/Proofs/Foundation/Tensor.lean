import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.Ring
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.Calculus.FDeriv.Add
import Mathlib.Analysis.Calculus.FDeriv.Mul
import Mathlib.Analysis.Calculus.FDeriv.Comp
import Mathlib.Analysis.Calculus.FDeriv.Pi
import Mathlib.Analysis.Calculus.FDeriv.Linear
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Comp

/-!
# Tensor algebra and the VJP framework — the root of the proof suite

`Vec n := Fin n → ℝ`, `Mat m n`, `Tensor3 c h w`, and the backward-pass calculus over them,
grounded in Mathlib's Fréchet derivative:

  `pdiv f x i j := fderiv ℝ f x (basisVec i) j`

Every structural rule is a theorem against Mathlib's API; the bilinear ones carry
`Differentiable` hypotheses that propagate through every downstream chapter.

Contents, in file order:

| section | what |
|---|---|
| Types, Matrix Operations | `Vec`, `Mat`, `Mat.mulVec`/`mul`/`transpose`/`outer`, `basisVec` |
| Differentiation | `pdiv` and its rules (`pdiv_comp`, `pdiv_add`, `pdiv_mul`, `pdiv_reindex`, …); the linear-map kits `pdiv_of_affine` / `pdiv_of_linear`, `pdiv_elementwise`, `pdiv_finset_sum`, `pdiv_const_smul` |
| VJP Framework, Pointwise VJP | `HasVJP` (global) and `HasVJPAt` (at a point), `canonical`, `backward_unique`, `vjp_comp` / `vjp_comp_at` / `vjp_comp_diff_at` and their `_backward` peels, `biPath` |
| flattening | `Mat.flatten` / `unflatten` with `_apply` and `@[simp]` round-trips, `sum_finProdFinEquiv` |
| Matrix-level | `pdivMat`, `HasVJPMat`, `vjpMat_comp`, row-independence (`pdivMat_rowIndep*`, `rowwise_has_vjp_mat`), and the matmul / scale / transpose Jacobians and VJPs. The multi-head column-slab kit and `HasVJPMat3` live in `Attention.lean` |
| 3D tensors | `Tensor3`, `pdiv3`, `HasVJP3` / `HasVJPAt3` and `hasVJP3_to_hasVJP` — nets compose at `Vec` through it |

Zero project axioms: everything closes under `propext`, `Classical.choice`, `Quot.sound`.
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Types
-- ════════════════════════════════════════════════════════════════

abbrev Vec (n : Nat) := Fin n → ℝ
abbrev Mat (m n : Nat) := Fin m → Fin n → ℝ

-- ════════════════════════════════════════════════════════════════
-- § Matrix Operations
-- ════════════════════════════════════════════════════════════════

namespace Mat

noncomputable def mulVec (A : Mat m n) (v : Vec n) : Vec m :=
  fun i => ∑ j : Fin n, A i j * v j

def outer (u : Vec m) (v : Vec n) : Mat m n :=
  fun i j => u i * v j

noncomputable def mul (A : Mat m n) (B : Mat n p) : Mat m p :=
  fun i k => ∑ j : Fin n, A i j * B j k

/-- Matrix transpose: swap rows and columns. -/
def transpose (A : Mat m n) : Mat n m :=
  fun j i => A i j

end Mat

-- ════════════════════════════════════════════════════════════════
-- § Differentiation (Mathlib-grounded)
--
-- `pdiv f x i j` is the (i, j) entry of the Jacobian of
-- `f : Vec m → Vec n` at `x`, recovered by applying `fderiv ℝ f x` to
-- the i-th standard basis vector and reading off the j-th coordinate.
-- All six structural rules (id, const, reindex, sum, product, chain)
-- are theorems. The bilinear rules (sum/product/chain) carry
-- `DifferentiableAt` hypotheses, the form required to be consistent
-- with `fderiv`'s junk-default at non-smooth points.
-- ════════════════════════════════════════════════════════════════

/-- Standard basis vector `eᵢ` in `Vec m`: 1 at index i, 0 elsewhere.
    Avoids `Pi.single`'s dependent-type elaboration friction in
    contexts where the codomain family isn't immediately apparent. -/
@[reducible] def basisVec {m : Nat} (i : Fin m) : Vec m :=
  fun k => if k = i then (1 : ℝ) else 0

@[simp] theorem basisVec_apply {m : Nat} (i j : Fin m) :
    basisVec i j = if j = i then (1 : ℝ) else 0 := rfl

/-- The reindex map `y ↦ (k ↦ y (σ k))` packaged as a continuous linear
    map. Used to discharge `pdiv_reindex` and to provide
    `DifferentiableAt` evidence for reindex-shaped subexpressions. -/
noncomputable def reindexCLM {a b : Nat} (σ : Fin b → Fin a) :
    Vec a →L[ℝ] Vec b :=
  { toFun := fun y k => y (σ k)
    map_add' := by intros; rfl
    map_smul' := by intros; rfl
    cont := continuous_pi (fun k => continuous_apply (σ k)) }

@[simp] theorem reindexCLM_apply {a b : Nat} (σ : Fin b → Fin a) (y : Vec a) :
    reindexCLM σ y = fun k => y (σ k) := rfl

/-- **Partial derivative.** The (i, j) entry of the Jacobian of
    `f : Vec m → Vec n` at `x`. -/
noncomputable def pdiv {m n : Nat} (f : Vec m → Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) : ℝ :=
  fderiv ℝ f x (basisVec i) j

/-- **Identity Jacobian** — `δᵢⱼ`. -/
theorem pdiv_id {n : Nat} (x : Vec n) (i j : Fin n) :
    pdiv (fun y : Vec n => y) x i j = if i = j then 1 else 0 := by
  simp [pdiv, @eq_comm _ j i]

/-- **A `pdiv` entry is the derivative of one output coordinate** — `pdiv f x i j` is
    `fderiv f x (basisVec i) j`, and at a differentiable point that is the `i`-th directional
    derivative of `y ↦ f y j`. -/
theorem pdiv_eq_fderiv_coord {m n : Nat} {f : Vec m → Vec n} {x : Vec m}
    (hf : DifferentiableAt ℝ f x) (i : Fin m) (j : Fin n) :
    pdiv f x i j = fderiv ℝ (fun y => f y j) x (basisVec i) := by
  unfold pdiv; rw [fderiv_apply hf j]; rfl

/-- **Constant function Jacobian** — zero. -/
theorem pdiv_const {m n : Nat} (c : Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) :
    pdiv (fun _ : Vec m => c) x i j = 0 := by
  simp [pdiv]

/-- **Reindex Jacobian** — sparse, hits 1 only at i = σ(j). Subsumes
    `pdiv_id` (set a = b, σ = id). Covers transpose, flatten,
    unflatten, slicing, any permutation. -/
theorem pdiv_reindex {a b : Nat} (σ : Fin b → Fin a) (x : Vec a)
    (i : Fin a) (j : Fin b) :
    pdiv (fun y : Vec a => fun k : Fin b => y (σ k)) x i j =
    if i = σ j then 1 else 0 := by
  rw [pdiv, show (fun y : Vec a => fun k : Fin b => y (σ k)) =
      (reindexCLM σ : Vec a → Vec b) from rfl, ContinuousLinearMap.fderiv]
  simp [@eq_comm _ (σ j) i]

/-- **Product rule** for `pdiv`. `Vec n` is a normed algebra over ℝ
    via `Pi.normedAlgebra`, so `fderiv_mul` applies directly to the
    pointwise product `f * g`. Requires both factors to be
    `DifferentiableAt x`. -/
theorem pdiv_mul {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k * g y k) x i j
    = pdiv f x i j * g x j + f x j * pdiv g x i j := by
  unfold pdiv
  rw [show (fun y : Vec m => fun k => f y k * g y k) = (f * g) from rfl]
  rw [fderiv_mul hf hg]
  simp only [add_apply, smul_apply,
             smul_eq_mul, Pi.add_apply, Pi.mul_apply]
  ring

/-- **Sum rule** for `pdiv`. Requires both summands to be
    `DifferentiableAt x`. -/
theorem pdiv_add {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k + g y k) x i j
    = pdiv f x i j + pdiv g x i j := by
  unfold pdiv
  rw [show (fun y => fun k => f y k + g y k) = (f + g) from rfl]
  rw [fderiv_add hf hg]
  rfl

/-- **Chain rule** for `pdiv`. Requires `f` differentiable at `x` and
    `g` differentiable at `f x`. -/
theorem pdiv_comp {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (x : Vec m) (hf : DifferentiableAt ℝ f x)
    (hg : DifferentiableAt ℝ g (f x))
    (i : Fin m) (k : Fin p) :
    pdiv (g ∘ f) x i k =
    ∑ j : Fin n, pdiv f x i j * pdiv g (f x) j k := by
  rw [pdiv, fderiv_comp x hg hf, ContinuousLinearMap.comp_apply, ← ContinuousLinearMap.coe_coe,
    LinearMap.pi_apply_eq_sum_univ, Finset.sum_apply]
  exact Finset.sum_congr rfl fun j _ =>
    congrArg (fderiv ℝ f x (basisVec i) j * fderiv ℝ g (f x) · k)
      (funext fun _ => if_congr eq_comm rfl rfl)

/-- **Scalar multiple rule** for `pdiv` — `pdiv_mul` at a constant factor. -/
theorem pdiv_const_smul {m n : Nat} (c : ℝ) (f : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (i : Fin m) (j : Fin n) :
    pdiv (fun y k => c * f y k) x i j = c * pdiv f x i j := by
  have h := pdiv_mul (fun _ : Vec m => fun _ : Fin n => c) f x
    (differentiableAt_const _) hf i j
  rw [h, pdiv_const (fun _ : Fin n => c) x i j]
  ring

-- ════════════════════════════════════════════════════════════════
-- § Linear-map and elementwise kits, the finite-sum rule, the scalar-multiple rule
-- ════════════════════════════════════════════════════════════════

/-- **Elementwise rule** — a scalar function applied to every coordinate has a diagonal
    Jacobian, the scalar derivative on the diagonal: each coordinate is `φ ∘ proj k`
    (`HasDerivAt.comp_hasFDerivAt`), assembled by `hasFDerivAt_pi`. GELU, swish and sigmoid are
    this with `φ` their scalar function. -/
theorem pdiv_elementwise {n : Nat} (φ : ℝ → ℝ) (x : Vec n)
    (hφ : ∀ k, DifferentiableAt ℝ φ (x k)) (i j : Fin n) :
    pdiv (fun y : Vec n => fun k => φ (y k)) x i j = if i = j then deriv φ (x i) else 0 := by
  have h : HasFDerivAt (fun y : Vec n => fun k => φ (y k))
      (ContinuousLinearMap.pi fun k => deriv φ (x k) • ContinuousLinearMap.proj k) x :=
    hasFDerivAt_pi.2 fun k => by
      have := (hφ k).hasDerivAt.comp_hasFDerivAt x
        (ContinuousLinearMap.proj k : Vec n →L[ℝ] ℝ).hasFDerivAt
      exact this
  rw [pdiv, h.fderiv]
  rcases eq_or_ne i j with rfl | hij
  · simp
  · simp [hij, Ne.symm hij]

/-- **A scalar function of ONE coordinate, lifted to `Vec K → Vec 1`, and its `pdiv`.** The
    elementwise rule at one coordinate and a general `f`; the shape every summand of a per-class
    loss has. -/
theorem pdiv_coordFun {K : Nat} (f : ℝ → ℝ) (f' : ℝ) (k : Fin K) (z : Vec K)
    (hf : HasDerivAt f f' (z k)) (j : Fin K) :
    pdiv (fun z' : Vec K => fun _ : Fin 1 => f (z' k)) z j 0 = if j = k then f' else 0 := by
  have h : HasFDerivAt (fun z' : Vec K => fun _ : Fin 1 => f (z' k))
      (ContinuousLinearMap.pi fun _ => f' • ContinuousLinearMap.proj k) z :=
    hasFDerivAt_pi.2 fun _ => by
      have := hf.comp_hasFDerivAt z (ContinuousLinearMap.proj k : Vec K →L[ℝ] ℝ).hasFDerivAt
      exact this
  rw [pdiv, h.fderiv]
  simp [@eq_comm _ k j]

/-- **Finset-sum rule** — linearity of the derivative extended to
    arbitrary finite sums (`fderiv_fun_sum`). Requires each `f s` to be
    differentiable at `x`. -/
theorem pdiv_finset_sum {m n : Nat} {α : Type*} [DecidableEq α]
    (S : Finset α) (f : α → Vec m → Vec n) (x : Vec m)
    (hdiff : ∀ s ∈ S, DifferentiableAt ℝ (f s) x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => ∑ s ∈ S, f s y k) x i j =
    ∑ s ∈ S, pdiv (f s) x i j := by
  rw [pdiv, show (fun y k => ∑ s ∈ S, f s y k) = fun y => ∑ s ∈ S, f s y from by
    funext y k; simp [Finset.sum_apply], fderiv_fun_sum hdiff]
  simp [pdiv, Finset.sum_apply]

/-- `pdiv_finset_sum` at a scalar loss: the gradient of a sum of scalar losses (lifted to
    `Vec 1`) is the sum of their gradients. The form every summed-loss cotangent takes. -/
theorem pdiv_lift_sum {P : Nat} {α : Type*} [DecidableEq α] (S : Finset α)
    (f : α → Vec P → ℝ) (x : Vec P)
    (h : ∀ k ∈ S, DifferentiableAt ℝ (fun z : Vec P => fun _ : Fin 1 => f k z) x) (j : Fin P) :
    pdiv (fun z : Vec P => fun _ : Fin 1 => ∑ k ∈ S, f k z) x j 0
      = ∑ k ∈ S, pdiv (fun z : Vec P => fun _ : Fin 1 => f k z) x j 0 :=
  pdiv_finset_sum S (fun k z _ => f k z) x h j 0

/-- **Affine rule** — if `f` is additive and homogeneous it is a linear map on the
    finite-dimensional `Vec m`, hence continuous (`LinearMap.continuous_on_pi`), and the Jacobian
    of `v ↦ f v + c` at every point is `f` read on the basis vector. Every conv / depthwise /
    dense / patch-embed / pooling / BN-affine Jacobian in the suite is this with `c` the bias:
    two one-line linearity side goals instead of distributing `pdiv` through the sum. -/
theorem pdiv_of_affine {m n : Nat} (f : Vec m → Vec n) (c : Vec n)
    (hadd : ∀ u v, f (u + v) = f u + f v) (hsmul : ∀ (a : ℝ) v, f (a • v) = a • f v)
    (x : Vec m) (i : Fin m) (j : Fin n) :
    pdiv (fun v => f v + c) x i j = f (basisVec i) j := by
  let L : Vec m →ₗ[ℝ] Vec n := ⟨⟨f, hadd⟩, hsmul⟩
  have hf : (fun v => f v + c) = fun v => (⟨L, L.continuous_on_pi⟩ : Vec m →L[ℝ] Vec n) v + c :=
    rfl
  rw [pdiv, hf, fderiv_add_const, ContinuousLinearMap.fderiv]; rfl

/-- **Linear rule, unbundled** — `pdiv_of_affine` at `c = 0`. -/
theorem pdiv_of_linear {m n : Nat} (f : Vec m → Vec n)
    (hadd : ∀ u v, f (u + v) = f u + f v) (hsmul : ∀ (a : ℝ) v, f (a • v) = a • f v)
    (x : Vec m) (i : Fin m) (j : Fin n) :
    pdiv f x i j = f (basisVec i) j := by
  simpa using pdiv_of_affine f 0 hadd hsmul x i j

-- ════════════════════════════════════════════════════════════════
-- § VJP Framework
-- ════════════════════════════════════════════════════════════════

structure HasVJP {m n : Nat} (f : Vec m → Vec n) where
  backward : Vec m → Vec n → Vec m
  correct : ∀ (x : Vec m) (dy : Vec n) (i : Fin m),
    backward x dy i = ∑ j : Fin n, pdiv f x i j * dy j

/-- **The canonical witness** — the backward IS the `pdiv` contraction, so `correct` is `rfl`.
    Exists for every `f`; a hand-written backward is tied to it by `HasVJP.backward_unique`. -/
noncomputable def HasVJP.canonical {m n : Nat} (f : Vec m → Vec n) : HasVJP f where
  backward x dy i := ∑ j : Fin n, pdiv f x i j * dy j
  correct _ _ _ := rfl

/-- **Two VJP witnesses for EQUAL maps have the same backward.** Both `.correct` to the same
    `∑ pdiv f x i j * dy j`. Going through `.correct` rather than `hfg ▸ ·` avoids an
    `Eq.mpr`-blocked `backward` when the witnesses have different types (a respelling of `f`). -/
theorem HasVJP.backward_unique_of_eq {m n : Nat} {f g : Vec m → Vec n} (hfg : f = g)
    (h₁ : HasVJP f) (h₂ : HasVJP g) (x : Vec m) (dy : Vec n) :
    h₁.backward x dy = h₂.backward x dy := by
  subst hfg; funext i; rw [h₁.correct, h₂.correct]

/-- **Any two VJP witnesses for the same map have the same backward** — the backward is a
    property of `f`, not of how the witness was assembled. Lets a hand-written chain be tied to a
    tactic-built witness without unfolding it. -/
theorem HasVJP.backward_unique {m n : Nat} {f : Vec m → Vec n} (h₁ h₂ : HasVJP f)
    (x : Vec m) (dy : Vec n) : h₁.backward x dy = h₂.backward x dy :=
  HasVJP.backward_unique_of_eq rfl h₁ h₂ x dy

-- ════════════════════════════════════════════════════════════════
-- § The VJP of a coordinate reindex (every layout bridge, decimation and broadcast)
-- ════════════════════════════════════════════════════════════════

/-- **The VJP of a coordinate reindex** `y ↦ y ∘ σ` (a gather, `reindexCLM σ`): the backward
    scatters each output cotangent back to the input cell it was read from — `pdiv_reindex`'s
    indicator, contracted. The one witness behind every reindex-shaped layer: the BN layout
    bridges (`reassocFwd/Back`, `bnchwFwd/Back`), the `StridedConv` decimations, and
    `reindex_has_vjp`'s `correct`. (`broadcastFlat_has_vjp` keeps its own witness: its backward
    is spelled the way the IR's `broadcastBack` denotes, and the SE-gate graph ties match it by
    `rfl`.) -/
noncomputable def reindexVJP {a b : Nat} (σ : Fin b → Fin a) :
    HasVJP (fun y : Vec a => fun k : Fin b => y (σ k)) where
  backward := fun _v dy => fun idx => ∑ k : Fin b, (if idx = σ k then (1 : ℝ) else 0) * dy k
  correct _ _ _ := Finset.sum_congr rfl fun _ _ => by rw [pdiv_reindex]

/-- **Along a bijection the scatter is the inverse gather**: when `τ` inverts `σ`, exactly one
    delta survives, so `reindexVJP σ`'s backward is the reindex along `τ`. -/
theorem reindexVJP_backward_of_inv {a b : Nat} (σ : Fin b → Fin a) (τ : Fin a → Fin b)
    (hστ : ∀ i, σ (τ i) = i) (hτσ : ∀ k, τ (σ k) = k) (v : Vec a) (dy : Vec b) :
    (reindexVJP σ).backward v dy = fun i => dy (τ i) := by
  funext i
  show ∑ k, (if i = σ k then (1 : ℝ) else 0) * dy k = dy (τ i)
  have h : ∀ k, i = σ k ↔ τ i = k := fun k =>
    ⟨fun h => by rw [h, hτσ], fun h => by rw [← h, hστ]⟩
  simp only [h, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq, Finset.mem_univ, ite_true]

/-- The scatter written as a masked sum (`if i = σ k then dy k else 0`) — the spelling the IR's
    scatter ops denote. -/
theorem reindexVJP_backward {a b : Nat} (σ : Fin b → Fin a) (v : Vec a) (dy : Vec b) :
    (reindexVJP σ).backward v dy = fun i => ∑ k : Fin b, if i = σ k then dy k else 0 := by
  funext i
  show ∑ k, (if i = σ k then (1 : ℝ) else 0) * dy k = _
  simp only [ite_mul, one_mul, zero_mul]

/-- **Additive fan-in** — proved, no sorry. Requires `f` and `g` to be
    differentiable everywhere. -/
@[reducible] noncomputable def biPath {m n : Nat} (f g : Vec m → Vec n) : Vec m → Vec n :=
  fun x i => f x i + g x i

noncomputable def biPath_has_vjp {m n : Nat}
    (f g : Vec m → Vec n)
    (hf_diff : Differentiable ℝ f) (hg_diff : Differentiable ℝ g)
    (hf : HasVJP f) (hg : HasVJP g) :
    HasVJP (biPath f g) where
  backward := fun x dy i => hf.backward x dy i + hg.backward x dy i
  correct := by
    intro x dy i
    rw [hf.correct, hg.correct, ← Finset.sum_add_distrib]
    congr 1; ext j; rw [pdiv_add _ _ _ (hf_diff x) (hg_diff x)]; ring

/-- **Multiplicative fan-in** — proved, no sorry. Requires `f` and `g`
    to be differentiable everywhere. -/
@[reducible] noncomputable def elemwiseProduct {n : Nat}
    (f g : Vec n → Vec n) : Vec n → Vec n :=
  fun x i => f x i * g x i

noncomputable def elemwiseProduct_has_vjp {n : Nat}
    (f g : Vec n → Vec n)
    (hf_diff : Differentiable ℝ f) (hg_diff : Differentiable ℝ g)
    (hf : HasVJP f) (hg : HasVJP g) :
    HasVJP (elemwiseProduct f g) where
  backward := fun x dy i =>
    hf.backward x (fun j => g x j * dy j) i +
    hg.backward x (fun j => f x j * dy j) i
  correct := by
    intro x dy i
    rw [hf.correct, hg.correct, ← Finset.sum_add_distrib]
    congr 1; ext j
    rw [pdiv_mul _ _ _ (hf_diff x) (hg_diff x)]; ring

/-- **Identity VJP** — proved, no sorry. -/
def identity_has_vjp (n : Nat) : HasVJP (fun (x : Vec n) => x) where
  backward := fun _x dy => dy
  correct := by
    intro x dy i
    simp_rw [pdiv_id]
    simp [Finset.mem_univ]

-- ════════════════════════════════════════════════════════════════
-- § Pointwise VJP — same load-bearing structure, single input point
-- ════════════════════════════════════════════════════════════════

/-! **Why a separate `HasVJPAt`.** The global `HasVJP` framework
delivers a *single* backward function that's correct at *every* input.
For non-smooth operators (`relu`, `maxPool2`, …) the only honest
`correct` witness is the canonical `pdiv`-derived sum, which gives a
trivially-`rfl`-true contract that doesn't pin down behavior at the
kinks. `HasVJPAt f x` carries the same contract but only at a chosen
smooth point `x` — exactly enough to discharge the chain rule under
`DifferentiableAt` and to plug in real per-operator Jacobian formulas
(`pdiv_relu`, `pdiv3_maxPool2_smooth`, …) instead of `correct := rfl`.

Smooth operators (`dense`, `add`, `mul`, `softmax`, `batchNorm`, …)
keep their global `HasVJP` instances; we trivially lift to `HasVJPAt`
at any point via `HasVJP.toHasVJPAt` when composing. -/

structure HasVJPAt {m n : Nat} (f : Vec m → Vec n) (x : Vec m) where
  backward : Vec n → Vec m
  correct : ∀ (dy : Vec n) (i : Fin m),
    backward dy i = ∑ j : Fin n, pdiv f x i j * dy j

/-- Two `HasVJPAt` witnesses for EQUAL maps at one point have the same backward — the pointwise
    peer of `HasVJP.backward_unique_of_eq`, through `.correct` rather than a transport. -/
theorem HasVJPAt.backward_unique_of_eq {m n : Nat} {f g : Vec m → Vec n} {x : Vec m}
    (hfg : f = g) (h₁ : HasVJPAt f x) (h₂ : HasVJPAt g x) (dy : Vec n) :
    h₁.backward dy = h₂.backward dy := by
  subst hfg; funext i; rw [h₁.correct, h₂.correct]

/-- **Any two `HasVJPAt` witnesses for the same map at the same point have the same backward** —
    `HasVJP.backward_unique`'s pointwise peer. -/
theorem HasVJPAt.backward_unique {m n : Nat} {f : Vec m → Vec n} {x : Vec m}
    (h₁ h₂ : HasVJPAt f x) (dy : Vec n) : h₁.backward dy = h₂.backward dy :=
  HasVJPAt.backward_unique_of_eq rfl h₁ h₂ dy

/-- Trivial lift: a global `HasVJP` gives a `HasVJPAt` at any point. -/
def HasVJP.toHasVJPAt {m n : Nat} {f : Vec m → Vec n}
    (hf : HasVJP f) (x : Vec m) : HasVJPAt f x where
  backward dy := hf.backward x dy
  correct := hf.correct x

/-- **Identity pointwise VJP** — trivial. -/
def identity_has_vjp_at (n : Nat) (x : Vec n) :
    HasVJPAt (fun (y : Vec n) => y) x :=
  (identity_has_vjp n).toHasVJPAt x

/-- **Chain rule for pointwise VJPs.** Same shape as `vjp_comp`, but
    only requires `DifferentiableAt` at the relevant points (not
    everywhere). The pointwise analogue is what lets us compose
    through `relu` at smooth inputs. -/
noncomputable def vjp_comp_at {m n p : Nat}
    (f : Vec m → Vec n) (g : Vec n → Vec p) (x : Vec m)
    (hf_diff : DifferentiableAt ℝ f x)
    (hg_diff : DifferentiableAt ℝ g (f x))
    (hf : HasVJPAt f x) (hg : HasVJPAt g (f x)) :
    HasVJPAt (g ∘ f) x where
  backward dy := hf.backward (hg.backward dy)
  correct := by
    intro dy i
    rw [hf.correct]
    simp_rw [hg.correct]
    simp_rw [Finset.mul_sum]
    rw [Finset.sum_comm]
    congr 1; ext k
    rw [pdiv_comp _ _ _ hf_diff hg_diff]
    simp_rw [← mul_assoc]
    rw [← Finset.sum_mul]

/-- **Chain rule for VJPs** — `vjp_comp_at` at every point. Requires `f` and `g`
    to be differentiable everywhere. -/
noncomputable def vjp_comp {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (hf_diff : Differentiable ℝ f) (hg_diff : Differentiable ℝ g)
    (hf : HasVJP f) (hg : HasVJP g) :
    HasVJP (g ∘ f) where
  backward := fun x dy => hf.backward x (hg.backward (f x) dy)
  correct x :=
    (vjp_comp_at f g x (hf_diff x) (hg_diff (f x)) (hf.toHasVJPAt x) (hg.toHasVJPAt (f x))).correct

/-- `vjp_comp_at`'s backward: `g`'s, then `f`'s. Definitional; stated so a chain peels by `rw`.
    ⛔ Not by `simp only` in a whole-net tie: simp uses an `rfl` lemma as a `dsimp` step and the
    kernel re-derives the unfolding. -/
theorem vjp_comp_at_backward {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p) (x : Vec m)
    (hf_diff : DifferentiableAt ℝ f x) (hg_diff : DifferentiableAt ℝ g (f x))
    (hf : HasVJPAt f x) (hg : HasVJPAt g (f x)) (dy : Vec p) :
    (vjp_comp_at f g x hf_diff hg_diff hf hg).backward dy = hf.backward (hg.backward dy) := rfl

/-- `vjp_comp`'s backward: `g`'s at `f x`, then `f`'s. Definitional, for `rw`. -/
theorem vjp_comp_backward {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (hf_diff : Differentiable ℝ f) (hg_diff : Differentiable ℝ g)
    (hf : HasVJP f) (hg : HasVJP g) (x : Vec m) (dy : Vec p) :
    (vjp_comp f g hf_diff hg_diff hf hg).backward x dy = hf.backward x (hg.backward (f x) dy) :=
  rfl

/-- A VJP witness at `x` together with differentiability there — one stage of a whole-net
    chain. A `PProd`, so the `DifferentiableAt` `Prop` can ride along with the data. -/
abbrev HasVJPDiffAt {m n : Nat} (f : Vec m → Vec n) (x : Vec m) :=
  PProd (HasVJPAt f x) (DifferentiableAt ℝ f x)

/-- Compose two `HasVJPDiffAt` stages. The fold step of every whole-net apex. -/
noncomputable def vjp_comp_diff_at {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (x : Vec m) (hf : HasVJPDiffAt f x) (hg : HasVJPDiffAt g (f x)) :
    HasVJPDiffAt (g ∘ f) x :=
  ⟨vjp_comp_at f g x hf.snd hg.snd hf.fst hg.fst, hg.snd.comp x hf.snd⟩

/-- One `vjp_comp_diff_at` level's backward, unfolded: the composite runs `g`'s backward, then
    `f`'s. Definitional, stated so that a chain of levels peels by `rw` rather than by a
    `rfl` that has to find the same unfolding through concrete witnesses. ⛔ Not by
    `simp only`: simp would use it as a `dsimp` step and record nothing for the kernel to replay. -/
theorem vjp_comp_diff_at_fst_backward {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (x : Vec m) (hf : HasVJPDiffAt f x) (hg : HasVJPDiffAt g (f x)) (dy : Vec p) :
    (vjp_comp_diff_at f g x hf hg).fst.backward dy = hf.fst.backward (hg.fst.backward dy) := rfl

-- ════════════════════════════════════════════════════════════════
-- § Matrix ↔ Vector flattening (row-major)
-- ════════════════════════════════════════════════════════════════

/-! `Mat m n` and `Vec (m * n)` are in bijection by row-major flattening.
This bijection lets us **define** `pdivMat` in terms of `pdiv` rather
than introducing parallel axioms, and so **derive** the rank-2 chain,
sum, and identity rules as theorems. The 5 local Jacobian theorems
(matmul, scalarScale, transpose, rowIndep) are likewise derived from
foundation rules — they state genuine calculus facts about specific
operations, not structural framework. -/

/-- **Row-major reindexing of a flat sum** — a sum over `Fin (m * n)` is the double sum over
    `(i, j)` read through `finProdFinEquiv`. Mathlib has the `Fin m × Fin n` form
    (`Fintype.sum_prod_type`) but no `Fin (m * n)` one; every flatten/unflatten sum in the suite
    reduces to this split. -/
theorem sum_finProdFinEquiv {M : Type*} [AddCommMonoid M] {m n : Nat}
    (f : Fin (m * n) → M) :
    ∑ k, f k = ∑ i : Fin m, ∑ j : Fin n, f (finProdFinEquiv (i, j)) := by
  rw [← Equiv.sum_comp finProdFinEquiv f, Fintype.sum_prod_type]

/-- The rank-3 case: a `Fin (a·b·c)` sum is the row-major triple sum — the layout `Tensor3.flatten`
    and every `c·h·w` activation index use. -/
theorem sum_finProdFinEquiv₃ {M : Type*} [AddCommMonoid M] {a b c : Nat}
    (f : Fin (a * b * c) → M) :
    ∑ k, f k = ∑ i : Fin a, ∑ j : Fin b, ∑ l : Fin c,
      f (finProdFinEquiv (finProdFinEquiv (i, j), l)) := by
  rw [sum_finProdFinEquiv, sum_finProdFinEquiv]

namespace Mat

/-- Row-major flatten: `Mat m n → Vec (m * n)`. Uses Mathlib's
    `finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n)`. -/
noncomputable def flatten {m n : Nat} (A : Mat m n) : Vec (m * n) :=
  fun k => A (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2

/-- Row-major unflatten: `Vec (m * n) → Mat m n`. -/
noncomputable def unflatten {m n : Nat} (v : Vec (m * n)) : Mat m n :=
  fun i j => v (finProdFinEquiv (i, j))

theorem flatten_apply {m n : Nat} (A : Mat m n) (k : Fin (m * n)) :
    flatten A k = A (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2 := rfl

theorem unflatten_apply {m n : Nat} (v : Vec (m * n)) (i : Fin m) (j : Fin n) :
    unflatten v i j = v (finProdFinEquiv (i, j)) := rfl

/-- Unflatten is a left inverse of flatten. -/
@[simp] theorem unflatten_flatten {m n : Nat} (A : Mat m n) :
    unflatten (flatten A) = A := by
  funext i j; simp [unflatten, flatten]

/-- Flatten is a left inverse of unflatten. -/
@[simp] theorem flatten_unflatten {m n : Nat} (v : Vec (m * n)) :
    flatten (unflatten v) = v := by
  funext k; exact congrArg v (finProdFinEquiv.apply_symm_apply k)

end Mat

-- ════════════════════════════════════════════════════════════════
-- § Matrix-level differentiation (derived from `pdiv`)
-- ════════════════════════════════════════════════════════════════

/-- **Matrix partial derivative**, defined in terms of `pdiv` on the
    row-major flattened `Vec` form. No longer an axiom — the rank-2
    structural rules (chain/sum/id) now follow as theorems. -/
noncomputable def pdivMat {a b c d : Nat} (f : Mat a b → Mat c d) (A : Mat a b)
    (i : Fin a) (j : Fin b) (k : Fin c) (l : Fin d) : ℝ :=
  pdiv (fun v : Vec (a * b) => Mat.flatten (f (Mat.unflatten v)))
    (Mat.flatten A) (finProdFinEquiv (i, j)) (finProdFinEquiv (k, l))

/-- **Chain rule for `pdivMat`** — now a theorem, derived from `pdiv_comp`
    via the row-major flatten bijection. -/
theorem pdivMat_comp {a b c d e f : Nat}
    (F : Mat a b → Mat c d) (G : Mat c d → Mat e f)
    (A : Mat a b)
    (hF_diff : DifferentiableAt ℝ
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A))
    (hG_diff : DifferentiableAt ℝ
      (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u))) (Mat.flatten (F A)))
    (i : Fin a) (j : Fin b) (k : Fin e) (l : Fin f) :
    pdivMat (G ∘ F) A i j k l =
    ∑ p : Fin c, ∑ q : Fin d,
      pdivMat F A i j p q * pdivMat G (F A) p q k l := by
  unfold pdivMat
  have h_compose :
      (fun v : Vec (a * b) => Mat.flatten ((G ∘ F) (Mat.unflatten v))) =
      (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u))) ∘
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) := by
    funext v
    simp [Function.comp, Mat.unflatten_flatten]
  have h_mid :
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A)
      = Mat.flatten (F A) := by
    simp [Mat.unflatten_flatten]
  have hG_diff' : DifferentiableAt ℝ
      (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u)))
      ((fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A)) := by
    rw [h_mid]; exact hG_diff
  rw [h_compose, pdiv_comp _ _ _ hF_diff hG_diff']
  simp_rw [h_mid]
  rw [sum_finProdFinEquiv]

/-- **Sum rule for `pdivMat`** — theorem, via `pdiv_add`. Requires both
    flattened summands to be differentiable at `flatten A`. -/
theorem pdivMat_add {a b c d : Nat}
    (F G : Mat a b → Mat c d) (A : Mat a b)
    (hF_diff : DifferentiableAt ℝ
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A))
    (hG_diff : DifferentiableAt ℝ
      (fun v : Vec (a * b) => Mat.flatten (G (Mat.unflatten v))) (Mat.flatten A))
    (i : Fin a) (j : Fin b) (k : Fin c) (l : Fin d) :
    pdivMat (fun M r s => F M r s + G M r s) A i j k l
    = pdivMat F A i j k l + pdivMat G A i j k l := by
  unfold pdivMat
  have h_flat : (fun v : Vec (a * b) =>
                  Mat.flatten ((fun M r s => F M r s + G M r s) (Mat.unflatten v))) =
                (fun v k => (fun w => Mat.flatten (F (Mat.unflatten w))) v k +
                            (fun w => Mat.flatten (G (Mat.unflatten w))) v k) := by
    funext v k
    unfold Mat.flatten
    rfl
  rw [h_flat, pdiv_add _ _ _ hF_diff hG_diff]

/-- **Identity Jacobian for `pdivMat`** — theorem, via `pdiv_id`. -/
theorem pdivMat_id {a b : Nat} (A : Mat a b)
    (i : Fin a) (j : Fin b) (k : Fin a) (l : Fin b) :
    pdivMat (fun M : Mat a b => M) A i j k l =
    if i = k ∧ j = l then 1 else 0 := by
  simp [pdivMat, Mat.flatten_unflatten, pdiv_id]

-- ════════════════════════════════════════════════════════════════
-- § Matrix VJP Framework
-- ════════════════════════════════════════════════════════════════

/-- Matrix-level VJP: given a matrix-valued function of a matrix, a
    correct backward function contracts the `pdivMat` Jacobian against
    the output cotangent. Mirrors `HasVJP` for `Vec`. -/
structure HasVJPMat {a b c d : Nat} (f : Mat a b → Mat c d) where
  backward : Mat a b → Mat c d → Mat a b
  correct : ∀ (A : Mat a b) (dY : Mat c d) (i : Fin a) (j : Fin b),
    backward A dY i j = ∑ k : Fin c, ∑ l : Fin d,
      pdivMat f A i j k l * dY k l

/-- Two `HasVJPMat` witnesses for EQUAL maps have the same backward — the matrix peer of
    `HasVJP.backward_unique_of_eq`, through `.correct` rather than a transport. -/
theorem HasVJPMat.backward_unique_of_eq {a b c d : Nat} {f g : Mat a b → Mat c d}
    (hfg : f = g) (v : HasVJPMat f) (v' : HasVJPMat g) (A : Mat a b) (dY : Mat c d) :
    v.backward A dY = v'.backward A dY := by
  subst hfg; funext i j; rw [v.correct, v'.correct]

/-- **Chain rule for matrix VJPs** — proved, no sorry.
    Direct transcription of `vjp_comp` to rank-2 indices. -/
noncomputable def vjpMat_comp {a b c d e f : Nat}
    (F : Mat a b → Mat c d) (G : Mat c d → Mat e f)
    (hF_diff : Differentiable ℝ
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))))
    (hG_diff : Differentiable ℝ
      (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u))))
    (hF : HasVJPMat F) (hG : HasVJPMat G) :
    HasVJPMat (G ∘ F) where
  backward := fun A dY => hF.backward A (hG.backward (F A) dY)
  correct := by
    intro A dY i j
    rw [hF.correct]
    simp_rw [hG.correct]
    have hF_diff_at : DifferentiableAt ℝ
        (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A) :=
      hF_diff (Mat.flatten A)
    have hG_diff_at : DifferentiableAt ℝ
        (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u))) (Mat.flatten (F A)) :=
      hG_diff (Mat.flatten (F A))
    conv_rhs =>
      arg 2; ext k; arg 2; ext l
      rw [show pdivMat (G ∘ F) A i j k l * dY k l =
          (∑ p : Fin c, ∑ q : Fin d,
            pdivMat F A i j p q * pdivMat G (F A) p q k l) * dY k l
        from by rw [← pdivMat_comp _ _ _ hF_diff_at hG_diff_at]]
    simp_rw [Finset.sum_mul, mul_assoc, Finset.mul_sum]
    -- LHS: ∑p ∑q, pdivMat F · ∑k ∑l, pdivMat G · dY
    -- RHS: ∑k ∑l ∑p ∑q, pdivMat F · pdivMat G · dY
    -- Pack (p,q) and (k,l) into products, swap, unpack.
    calc _ = ∑ pq ∈ Finset.univ ×ˢ Finset.univ,
             ∑ kl ∈ Finset.univ ×ˢ Finset.univ,
               pdivMat F A i j pq.1 pq.2 *
                 (pdivMat G (F A) pq.1 pq.2 kl.1 kl.2 * dY kl.1 kl.2) := by
             simp_rw [Finset.sum_product]
         _ = ∑ kl ∈ Finset.univ ×ˢ Finset.univ,
             ∑ pq ∈ Finset.univ ×ˢ Finset.univ,
               pdivMat F A i j pq.1 pq.2 *
                 (pdivMat G (F A) pq.1 pq.2 kl.1 kl.2 * dY kl.1 kl.2) :=
             Finset.sum_comm
         _ = _ := by simp_rw [Finset.sum_product]

/-- **Additive fan-in for matrices** — proved, no sorry. -/
@[reducible] noncomputable def biPathMat {a b c d : Nat}
    (F G : Mat a b → Mat c d) : Mat a b → Mat c d :=
  fun M r s => F M r s + G M r s

noncomputable def biPathMat_has_vjp {a b c d : Nat}
    (F G : Mat a b → Mat c d)
    (hF_diff : Differentiable ℝ
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))))
    (hG_diff : Differentiable ℝ
      (fun v : Vec (a * b) => Mat.flatten (G (Mat.unflatten v))))
    (hF : HasVJPMat F) (hG : HasVJPMat G) :
    HasVJPMat (biPathMat F G) where
  backward := fun A dY i j => hF.backward A dY i j + hG.backward A dY i j
  correct := by
    intro A dY i j
    rw [hF.correct, hG.correct, ← Finset.sum_add_distrib]
    congr 1; ext k
    rw [← Finset.sum_add_distrib]
    congr 1; ext l
    rw [pdivMat_add _ _ _ (hF_diff (Mat.flatten A)) (hG_diff (Mat.flatten A))]; ring

/-- **Identity VJP for matrices** — proved, no sorry. -/
noncomputable def identityMat_has_vjp (a b : Nat) :
    HasVJPMat (fun (M : Mat a b) => M) where
  backward := fun _A dY => dY
  correct := by
    intro A dY i j
    simp_rw [pdivMat_id]
    simp [ite_and]

/-- **Bridge: `HasVJPMat` → `HasVJP` via the `Mat.flatten` bijection.**

    Given a matrix-level VJP for `f : Mat a b → Mat c d`, produce a
    vector-level VJP for the flattened version
    `fun v : Vec (a*b) => Mat.flatten (f (Mat.unflatten v))`. The backward
    reshapes the input/output flat vectors to matrices, applies the
    matrix backward, and flattens the result.

    Lets us compose `HasVJPMat` pieces (vit_body, transformer blocks)
    with rank-crossing pieces (patch embed, classifier head) that live
    natively as `Vec → Vec` by first bridging everything to `HasVJP`. -/
noncomputable def hasVJPMat_to_hasVJP {a b c d : Nat} {f : Mat a b → Mat c d}
    (hf : HasVJPMat f) :
    HasVJP (fun v : Vec (a * b) =>
              Mat.flatten (f (Mat.unflatten v))) where
  backward := fun v dy => fun idx =>
    let ij := finProdFinEquiv.symm idx
    hf.backward (Mat.unflatten v) (Mat.unflatten dy) ij.1 ij.2
  correct := by
    intro v dy idx
    obtain ⟨⟨r, s⟩, rfl⟩ := finProdFinEquiv.surjective idx
    simp only [Equiv.symm_apply_apply, hf.correct, pdivMat, Mat.flatten_unflatten,
      sum_finProdFinEquiv]
    rfl

-- ════════════════════════════════════════════════════════════════
-- § Matrix VJP Building Blocks (matmul, row-independent functions)
-- ════════════════════════════════════════════════════════════════

/-! The three theorems here are local Jacobians for the operations that
appear in scaled dot-product attention's backward pass:

1. **`pdivMat_matmul_left_const`** — right-factor varies, left factor fixed:
   `∂(C · B')_{kl} / ∂B'_{ij} = C_{ki} · [l = j]`.
2. **`pdivMat_matmul_right_const`** — left factor varies, right factor fixed:
   `∂(A' · D)_{kl} / ∂A'_{ij} = D_{jl} · [i = k]`.
3. **`pdivMat_rowIndep`** — functions that act row-wise have block-diagonal
   Jacobians, with the per-row block equal to the vector Jacobian of the
   row function `g`.

Each is a direct transcription of an elementary calculus fact. They are
numerically gradient-checked in `check_jacobians.py`. -/

/-- **Matmul Jacobian (left-const)** — `B' ↦ C·B'` is linear, so by `pdiv_of_linear` the entry
    is `C·E_{ij}` read at `(k, l)`: `C k i` when `l = j`. -/
theorem pdivMat_matmul_left_const {m p q : Nat} (C : Mat m p) (B : Mat p q)
    (i : Fin p) (j : Fin q) (k : Fin m) (l : Fin q) :
    pdivMat (fun B' : Mat p q => Mat.mul C B') B i j k l =
    if l = j then C k i else 0 := by
  rw [pdivMat, pdiv_of_linear (fun v => Mat.flatten (Mat.mul C (Mat.unflatten v)))
    (fun _ _ => by
      funext; simp [Mat.flatten, Mat.mul, Mat.unflatten, mul_add, Finset.sum_add_distrib])
    (fun _ _ => by
      funext; simp [Mat.flatten, Mat.mul, Mat.unflatten, Finset.mul_sum, mul_left_comm])]
  by_cases h : l = j <;> simp [Mat.flatten, Mat.mul, Mat.unflatten, h, Prod.ext_iff]

/-- **Matmul Jacobian (right-const)** — the left-const case with roles swapped: `A' ↦ A'·D` is
    linear, and `E_{ij}·D` read at `(k, l)` is `D j l` when `i = k`. -/
theorem pdivMat_matmul_right_const {m p q : Nat} (A : Mat m p) (D : Mat p q)
    (i : Fin m) (j : Fin p) (k : Fin m) (l : Fin q) :
    pdivMat (fun A' : Mat m p => Mat.mul A' D) A i j k l =
    if i = k then D j l else 0 := by
  rw [pdivMat, pdiv_of_linear (fun v => Mat.flatten (Mat.mul (Mat.unflatten v) D))
    (fun _ _ => by
      funext; simp [Mat.flatten, Mat.mul, Mat.unflatten, add_mul, Finset.sum_add_distrib])
    (fun _ _ => by funext; simp [Mat.flatten, Mat.mul, Mat.unflatten, Finset.mul_sum, mul_assoc])]
  by_cases h : i = k <;> simp [Mat.flatten, Mat.mul, Mat.unflatten, h, Prod.ext_iff, eq_comm]

/-- **Block-diagonal Jacobian of a per-row family, at a point.** Applying `g r` to each row `r`
    keeps the matrix Jacobian block-diagonal across rows: the `(i, j, k, l)` entry is
    `pdiv (g k) (A k) j l` when `i = k` and `0` otherwise. Each `g r` need only be differentiable
    at its own row `A r`: the flat map's coordinate `(r, l)` is `g r`'s coordinate `l` after the
    row projection, and `hasFDerivAt_pi` assembles the rows' derivatives. -/
theorem pdivMat_rowIndep_perRow_at {m n p : Nat} (g : Fin m → (Vec n → Vec p)) (A : Mat m n)
    (h_g_diff : ∀ r, DifferentiableAt ℝ (g r) (A r))
    (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin p) :
    pdivMat (fun M : Mat m n => fun r => g r (M r)) A i j k l =
    if i = k then pdiv (g k) (A k) j l else 0 := by
  let row : Fin m → (Vec (m * n) →L[ℝ] Vec n) := fun r =>
    reindexCLM fun j' => finProdFinEquiv (r, j')
  have hrow : ∀ r, row r (Mat.flatten A) = A r := fun r => by funext; simp [row, Mat.flatten]
  have h : HasFDerivAt (fun v : Vec (m * n) => Mat.flatten (fun r => g r (Mat.unflatten v r)))
      (ContinuousLinearMap.pi fun idx => (ContinuousLinearMap.proj (finProdFinEquiv.symm idx).2 :
        Vec p →L[ℝ] ℝ).comp ((fderiv ℝ (g (finProdFinEquiv.symm idx).1)
          (A (finProdFinEquiv.symm idx).1)).comp (row (finProdFinEquiv.symm idx).1)))
      (Mat.flatten A) :=
    hasFDerivAt_pi.2 fun idx => by
      have := (h_g_diff (finProdFinEquiv.symm idx).1).hasFDerivAt
      rw [← hrow] at this ⊢
      exact hasFDerivAt_pi'.1 (this.comp _ (row _).hasFDerivAt) _
  have hb : row k (basisVec (finProdFinEquiv (i, j))) = if i = k then basisVec j else 0 := by
    funext j'; rcases eq_or_ne i k with rfl | hik
    · simp [row, @eq_comm _ j' j]
    · simp [row, hik, hik.symm]
  rw [pdivMat, pdiv, h.fderiv]
  simp only [ContinuousLinearMap.pi_apply, ContinuousLinearMap.comp_apply,
    ContinuousLinearMap.proj_apply, Equiv.symm_apply_apply, hb]
  split_ifs <;> simp [pdiv]

/-- **Row-wise Jacobian decomposition** — proved (planning/archive/VJP.md follow-up D).

    For a row-independent function `M ↦ (r ↦ g (M r))`, the (i,j,k,l)
    Jacobian entry is `pdiv g (A i) j l` when `i = k` and `0` otherwise:
    `pdivMat_rowIndep_perRow_at` with one `g` for every row.

    Requires `Differentiable ℝ g`: without it, the flattened Pi-valued
    function may be non-differentiable at `Mat.flatten A` (per
    `differentiable_pi`'s coordinate-wise condition), making `fderiv = 0`
    junk and breaking the per-row decomposition. -/
theorem pdivMat_rowIndep {m n p : Nat} (g : Vec n → Vec p)
    (h_g_diff : Differentiable ℝ g)
    (A : Mat m n) (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin p) :
    pdivMat (fun M : Mat m n => fun r => g (M r)) A i j k l =
    if i = k then pdiv g (A i) j l else 0 := by
  rw [pdivMat_rowIndep_perRow_at (fun _ => g) A (fun _ => h_g_diff.differentiableAt)]
  split_ifs with h <;> simp [h]

/-- **Row-wise lifting of a `HasVJP`** (Phase 8, Tensor-level).

    Given any `g : Vec n → Vec p` with a proved `HasVJP`, applying `g`
    independently to each row of a matrix `A : Mat m n` gives a
    `HasVJPMat` on `Mat m n → Mat m p`. The backward is just `g.backward`
    applied per row. Generalizes `rowSoftmax_has_vjp_mat`: any per-token
    operation (LayerNorm, GELU, dense, activation) lifts to a per-sequence
    matrix operation via this one helper. -/
noncomputable def rowwise_has_vjp_mat {m n p : Nat} {g : Vec n → Vec p}
    (hg : HasVJP g) (hg_diff : Differentiable ℝ g) :
    HasVJPMat (fun A : Mat m n => fun r => g (A r)) where
  backward := fun A dY => fun r c => hg.backward (A r) (dY r) c
  correct := by
    intro A dY i j
    simp_rw [pdivMat_rowIndep g hg_diff]
    simp [hg.correct]

-- ════════════════════════════════════════════════════════════════
-- § Scale and transpose Jacobians, and the matmul / scale / transpose VJPs
-- ════════════════════════════════════════════════════════════════

/-- **Scalar-scale Jacobian** — theorem, derived from `pdiv_const_smul` +
    `pdiv_id` via the flatten bijection.
    `∂(s · A')_{kl} / ∂A'_{ij} = s · δ_{ik,jl}`. -/
theorem pdivMat_scalarScale {m n : Nat} (s : ℝ) (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin n) :
    pdivMat (fun M : Mat m n => fun r c => s * M r c) A i j k l =
    if i = k ∧ j = l then s else 0 := by
  rw [pdivMat, show (fun v : Vec (m * n) => Mat.flatten (fun r c => s * Mat.unflatten v r c)) =
      fun v k => s * Mat.flatten (Mat.unflatten v) k from rfl]
  simp only [Mat.flatten_unflatten]
  rw [pdiv_const_smul s (fun w => w) _ differentiableAt_id, pdiv_id]
  simp

/-- **Transpose Jacobian** — theorem, derived from `pdiv_reindex` via
    the flatten bijection.  `∂A^T_{kl} / ∂A_{ij} = δ_{l=i, k=j}`. -/
theorem pdivMat_transpose {m n : Nat} (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin n) (l : Fin m) :
    pdivMat (fun M : Mat m n => Mat.transpose M) A i j k l =
    if j = k ∧ i = l then 1 else 0 := by
  -- `flatten ∘ transpose ∘ unflatten` is the gather at the swapped index.
  exact (pdiv_reindex (fun idx => finProdFinEquiv
    ((finProdFinEquiv.symm idx).2, (finProdFinEquiv.symm idx).1)) _ _ _).trans (by simp [and_comm])

/-- **Matmul with right factor varying, left factor fixed** — proved.

    `f : Mat p q → Mat m q`,  `f B' = C · B'`.
    Backward: `dB' = C^T · dY`. -/
noncomputable def matmul_left_const_has_vjp {m p q : Nat} (C : Mat m p) :
    HasVJPMat (fun B' : Mat p q => Mat.mul C B') where
  backward := fun _B dY => fun i j => ∑ k : Fin m, C k i * dY k j
  correct := by
    intro B dY i j
    simp_rw [pdivMat_matmul_left_const]
    simp

/-- **Matmul with left factor varying, right factor fixed** — proved.

    `f : Mat m p → Mat m q`,  `f A' = A' · D`.
    Backward: `dA' = dY · D^T`. -/
noncomputable def matmul_right_const_has_vjp {m p q : Nat} (D : Mat p q) :
    HasVJPMat (fun A' : Mat m p => Mat.mul A' D) where
  backward := fun _A dY => fun i j => ∑ l : Fin q, dY i l * D j l
  correct := by
    intro A dY i j
    simp_rw [pdivMat_matmul_right_const]
    simp [mul_comm]

/-- **Scalar-scale VJP** — proved.  Backward: `dA = s · dY`. -/
noncomputable def scalarScale_has_vjp {m n : Nat} (s : ℝ) :
    HasVJPMat (fun M : Mat m n => fun r c => s * M r c) where
  backward := fun _A dY => fun i j => s * dY i j
  correct := by
    intro A dY i j
    simp_rw [pdivMat_scalarScale]
    simp [ite_and]

/-- **Transpose VJP** — proved.  Backward: `dA = (dY)^T`. -/
noncomputable def transpose_has_vjp {m n : Nat} :
    HasVJPMat (fun M : Mat m n => Mat.transpose M) where
  backward := fun _A dY => fun i j => dY j i
  correct := by
    intro A dY i j
    simp_rw [pdivMat_transpose]
    simp [ite_and]

-- ════════════════════════════════════════════════════════════════
-- § 3D Tensor VJP Framework (for CNN / Depthwise)
-- ════════════════════════════════════════════════════════════════

/-- A 3D feature map: channels × height × width (single sample). -/
abbrev Tensor3 (c h w : Nat) := Fin c → Fin h → Fin w → ℝ

namespace Tensor3

/-- Row-major flatten: `Tensor3 c h w → Vec (c * h * w)`. Two nested
    `finProdFinEquiv` calls: first bundle `(ci, hi)` into `Fin (c*h)`,
    then bundle with `wi` into `Fin ((c*h)*w) = Fin (c*h*w)`. -/
noncomputable def flatten {c h w : Nat} (T : Tensor3 c h w) : Vec (c * h * w) :=
  fun k => T (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1
    (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).2 (finProdFinEquiv.symm k).2

/-- Row-major unflatten: inverse of `flatten`. -/
noncomputable def unflatten {c h w : Nat} (v : Vec (c * h * w)) : Tensor3 c h w :=
  fun ci hi wi => v (finProdFinEquiv (finProdFinEquiv (ci, hi), wi))

theorem flatten_apply {c h w : Nat} (T : Tensor3 c h w) (k : Fin (c * h * w)) :
    flatten T k = T (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1
      (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).2 (finProdFinEquiv.symm k).2 := rfl

theorem unflatten_apply {c h w : Nat} (v : Vec (c * h * w)) (ci : Fin c) (hi : Fin h)
    (wi : Fin w) : unflatten v ci hi wi = v (finProdFinEquiv (finProdFinEquiv (ci, hi), wi)) :=
  rfl

@[simp] theorem unflatten_flatten {c h w : Nat} (T : Tensor3 c h w) :
    unflatten (flatten T) = T := by
  funext ci hi wi; simp [unflatten, flatten]

@[simp] theorem flatten_unflatten {c h w : Nat} (v : Vec (c * h * w)) :
    flatten (unflatten v) = v := by
  funext k; simp only [flatten, unflatten, Prod.mk.eta, Equiv.apply_symm_apply]

/-- **`Tensor3.flatten` is differentiable.** It is a coordinate
    reindexing: each output coordinate `flatten x k` is the single input
    coordinate `x (decode k)`, hence a projection. -/
@[fun_prop]
theorem flatten_differentiable {c h w : Nat} :
    Differentiable ℝ (Tensor3.flatten : Tensor3 c h w → Vec (c * h * w)) := by
  unfold flatten; fun_prop

/-- **`Tensor3.unflatten` is differentiable.** The inverse reindexing:
    each output coordinate `unflatten v ci hi wi` is the single input
    coordinate `v (encode (ci,hi,wi))`, hence a projection. -/
@[fun_prop]
theorem unflatten_differentiable {c h w : Nat} :
    Differentiable ℝ (Tensor3.unflatten : Vec (c * h * w) → Tensor3 c h w) := by
  unfold unflatten; fun_prop

end Tensor3

/-- **3D partial derivative** — `pdiv` of the flattened map, both indices read through the
    triple-nested flatten bijection. Operator-specific VJPs at rank 3 (`conv2d_has_vjp3`,
    `maxPool2_has_vjp3`, `depthwise_has_vjp3`) are bundled `HasVJP3` defs in their own files;
    nets compose them at `Vec`, through `hasVJP3_to_hasVJP` / `hasVJPAt3_to_hasVJPAt`. -/
noncomputable def pdiv3 {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    (f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂)
    (x : Tensor3 c₁ h₁ w₁)
    (ci : Fin c₁) (hi : Fin h₁) (wi : Fin w₁)
    (co : Fin c₂) (ho : Fin h₂) (wo : Fin w₂) : ℝ :=
  pdiv (fun v : Vec (c₁ * h₁ * w₁) =>
          Tensor3.flatten (f (Tensor3.unflatten v)))
    (Tensor3.flatten x)
    (finProdFinEquiv (finProdFinEquiv (ci, hi), wi))
    (finProdFinEquiv (finProdFinEquiv (co, ho), wo))

/-- VJP for 3D→3D functions. -/
structure HasVJP3 {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    (f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂) where
  backward : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂ → Tensor3 c₁ h₁ w₁
  correct : ∀ (x : Tensor3 c₁ h₁ w₁) (dy : Tensor3 c₂ h₂ w₂)
    (ci : Fin c₁) (hi : Fin h₁) (wi : Fin w₁),
    backward x dy ci hi wi =
    ∑ co : Fin c₂, ∑ ho : Fin h₂, ∑ wo : Fin w₂,
      pdiv3 f x ci hi wi co ho wo * dy co ho wo

-- ════════════════════════════════════════════════════════════════
-- § Pointwise VJP3 — Tensor3 analogue of HasVJPAt
-- ════════════════════════════════════════════════════════════════

/-- Tensor3 analogue of `HasVJPAt`: the same `pdiv3`-sum contract, but
    only required at the chosen smooth point `x`. The natural home for
    `maxPool2_has_vjp_at3` and any other kinked Tensor3 operator. -/
structure HasVJPAt3 {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    (f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂)
    (x : Tensor3 c₁ h₁ w₁) where
  backward : Tensor3 c₂ h₂ w₂ → Tensor3 c₁ h₁ w₁
  correct : ∀ (dy : Tensor3 c₂ h₂ w₂)
    (ci : Fin c₁) (hi : Fin h₁) (wi : Fin w₁),
    backward dy ci hi wi =
    ∑ co : Fin c₂, ∑ ho : Fin h₂, ∑ wo : Fin w₂,
      pdiv3 f x ci hi wi co ho wo * dy co ho wo

/-- **Bridge: `HasVJP3` → `HasVJP` via the `Tensor3.flatten` bijection.**

    Rank-3 analogue of `hasVJPMat_to_hasVJP`. Given a Tensor3-level VJP
    for `f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂`, produce a vector-level
    VJP for the flattened `fun v => Tensor3.flatten (f (Tensor3.unflatten v))`.
    The backward decodes the flat index in two `finProdFinEquiv.symm`
    levels (matching `pdiv3`'s row-major encode), applies the Tensor3
    backward, and the closing collapse folds the triple `co/ho/wo` sum
    back to the single flat sum via `sum_finProdFinEquiv` twice. -/
noncomputable def hasVJP3_to_hasVJP {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    {f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂}
    (hf : HasVJP3 f) :
    HasVJP (fun v : Vec (c₁ * h₁ * w₁) =>
              Tensor3.flatten (f (Tensor3.unflatten v))) where
  backward := fun v dy => fun idx =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.1
    hf.backward (Tensor3.unflatten v) (Tensor3.unflatten dy) q.1 q.2 p.2
  correct := by
    intro v dy idx
    obtain ⟨⟨ch, w⟩, rfl⟩ := finProdFinEquiv.surjective idx
    obtain ⟨⟨c, h⟩, rfl⟩ := finProdFinEquiv.surjective ch
    simp only [Equiv.symm_apply_apply, hf.correct, pdiv3, Tensor3.flatten_unflatten,
      sum_finProdFinEquiv (m := c₂ * h₂), sum_finProdFinEquiv (m := c₂)]
    rfl

/-- **Bridge: `HasVJPAt3` → `HasVJPAt` via the `Tensor3.flatten` bijection.**

    Smooth-point analogue of `hasVJP3_to_hasVJP`, with `x` fixed. Needed
    for kinked operators (e.g. `maxPool2`) that only carry `HasVJPAt3`.
    Same two-level index decode and triple→flat reindex collapse. -/
noncomputable def hasVJPAt3_to_hasVJPAt {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    {f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂}
    {x : Tensor3 c₁ h₁ w₁}
    (hf : HasVJPAt3 f x) :
    HasVJPAt (fun v : Vec (c₁ * h₁ * w₁) =>
              Tensor3.flatten (f (Tensor3.unflatten v)))
             (Tensor3.flatten x) where
  backward := fun dy => fun idx =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.1
    hf.backward (Tensor3.unflatten dy) q.1 q.2 p.2
  correct := by
    intro dy idx
    obtain ⟨⟨ch, w⟩, rfl⟩ := finProdFinEquiv.surjective idx
    obtain ⟨⟨c, h⟩, rfl⟩ := finProdFinEquiv.surjective ch
    simp only [Equiv.symm_apply_apply, hf.correct, pdiv3,
      sum_finProdFinEquiv (m := c₂ * h₂), sum_finProdFinEquiv (m := c₂)]
    rfl

end Proofs
