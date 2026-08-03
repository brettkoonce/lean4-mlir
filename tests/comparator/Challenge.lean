import Mathlib

open scoped Real

/-! # Challenge file for `leanprover/comparator` — the architecture-free layer

**This file imports Mathlib and nothing else.**

Each `chk_<name>` is a re-statement of a project theorem with `:= by sorry`.
`Solution.lean` discharges each via the corresponding proven theorem from
`LeanMlir.Proofs.*`. comparator confirms (a) the statements match
bit-identically, (b) only `propext, Quot.sound, Classical.choice` appear in
each Solution-side proof's transitive axiom closure, and (c) the Solution
typechecks against Lean's kernel independently of the elaborator.

The 13 theorems here are the calculus layer — chapter 1's `pdiv` rules and
chapter 9's matrix-level rules. **None of them mentions a neural network**,
which is exactly why this file can stand alone over Mathlib: the only
objects in scope are Mathlib's `fderiv` and a handful of type
abbreviations over it.

`ChallengeArch.lean` holds the other 39, the ones that state properties of
specific architectures. Those cannot be made Mathlib-only without copying
every network definition into the challenge file, which would put a second
writer on every architecture — a worse trust story than an import, not a
better one. That file says so in its own header.
-/

/-! ## The vocabulary, inlined

These are verbatim copies of the definitions in
`LeanMlir.Proofs.Foundation.Tensor`, reproduced here so that this file
imports nothing but Mathlib. They are one-liners on purpose: `pdiv` is
Mathlib's `fderiv` in a basis direction, and everything else is notation
over it. `chk_pdiv_is_fderiv` below pins that claim by `rfl`.

Copying them is safe in a way that copying an *architecture* would not be:
comparator compares the Challenge and Solution statements bit-identically,
so a copy that drifted from the original would fail the run rather than
pass a weaker theorem. -/

abbrev Vec (n : Nat) := Fin n → ℝ
abbrev Mat (m n : Nat) := Fin m → Fin n → ℝ

namespace Mat

noncomputable def mul {m n p : Nat} (A : Mat m n) (B : Mat n p) : Mat m p :=
  fun i k => ∑ j : Fin n, A i j * B j k

/-- Matrix transpose: swap rows and columns. -/
def transpose {m n : Nat} (A : Mat m n) : Mat n m :=
  fun j i => A i j

/-- Row-major flatten: `Mat m n → Vec (m * n)`. -/
noncomputable def flatten {m n : Nat} (A : Mat m n) : Vec (m * n) :=
  fun k => let p := finProdFinEquiv.symm k; A p.1 p.2

/-- Row-major unflatten: `Vec (m * n) → Mat m n`. -/
noncomputable def unflatten {m n : Nat} (v : Vec (m * n)) : Mat m n :=
  fun i j => v (finProdFinEquiv (i, j))

end Mat

/-- Standard basis vector `eᵢ` in `Vec m`: 1 at index i, 0 elsewhere. -/
@[reducible] def basisVec {m : Nat} (i : Fin m) : Vec m :=
  fun k => if k = i then (1 : ℝ) else 0

/-- **Partial derivative.** The (i, j) entry of the Jacobian of
    `f : Vec m → Vec n` at `x`. -/
noncomputable def pdiv {m n : Nat} (f : Vec m → Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) : ℝ :=
  fderiv ℝ f x (basisVec i) j

/-- **Matrix partial derivative**, defined in terms of `pdiv` on the
    row-major flattened `Vec` form. -/
noncomputable def pdivMat {a b c d : Nat} (f : Mat a b → Mat c d) (A : Mat a b)
    (i : Fin a) (j : Fin b) (k : Fin c) (l : Fin d) : ℝ :=
  pdiv (fun v : Vec (a * b) => Mat.flatten (f (Mat.unflatten v)))
    (Mat.flatten A) (finProdFinEquiv (i, j)) (finProdFinEquiv (k, l))

-- Foundation: structural calculus rules ────────────────────────────

/-- **`pdiv` is Mathlib's Fréchet derivative** (grounding):
$\mathrm{pdiv}\,f\,x\,i\,j = \big(\mathrm{fderiv}_{\mathbb{R}}\,f\,x\,(e_i)\big)_j$
— the directional Fréchet derivative along the i-th basis vector, j-th output
coord. Pins this file's calculus vocabulary to Mathlib: every other `chk_` is
stated via `pdiv`, and `pdivMat` is `pdiv` of the row-major flatten. -/
theorem chk_pdiv_is_fderiv {m n : Nat} (f : Vec m → Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) :
    pdiv f x i j = fderiv ℝ f x (basisVec i) j := by sorry

/-- **Chain rule** (foundation):
$\frac{\partial (g \circ f)_k}{\partial x_i} = \sum_j \frac{\partial f_j}{\partial x_i} \cdot \frac{\partial g_k}{\partial f_j}$

Used by every backward pass that composes two functions. The structural
glue that makes the whole proof tree compositional. -/
theorem chk_pdiv_comp {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (x : Vec m) (hf : DifferentiableAt ℝ f x)
    (hg : DifferentiableAt ℝ g (f x))
    (i : Fin m) (k : Fin p) :
    pdiv (g ∘ f) x i k =
    ∑ j : Fin n, pdiv f x i j * pdiv g (f x) j k := by sorry

/-- **Sum rule** / linearity (foundation):
$\frac{\partial (f+g)_j}{\partial x_i} = \frac{\partial f_j}{\partial x_i} + \frac{\partial g_j}{\partial x_i}$

Underpins additive fan-in (residual connections, multi-path attention). -/
theorem chk_pdiv_add {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k + g y k) x i j
    = pdiv f x i j + pdiv g x i j := by sorry

/-- **Product rule** / Leibniz (foundation):
$\frac{\partial (f \cdot g)_j}{\partial x_i} = \frac{\partial f_j}{\partial x_i} \cdot g_j + f_j \cdot \frac{\partial g_j}{\partial x_i}$

Underpins multiplicative fan-in (squeeze-and-excitation gating, attention
weights × values). -/
theorem chk_pdiv_mul {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k * g y k) x i j
    = pdiv f x i j * g x j + f x j * pdiv g x i j := by sorry

/-- **Identity Jacobian** (foundation):
$\frac{\partial \text{id}(x)_j}{\partial x_i} = \delta_{ij}$

Building block for skip connections and identity bridges in residuals. -/
theorem chk_pdiv_id {n : Nat} (x : Vec n) (i j : Fin n) :
    pdiv (fun y : Vec n => y) x i j = if i = j then 1 else 0 := by sorry

/-- **Constant has zero Jacobian** (foundation):
$\frac{\partial c}{\partial x_i} = 0$

Discharges bias terms when isolating weight gradients. -/
theorem chk_pdiv_const {m n : Nat} (c : Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) :
    pdiv (fun _ : Vec m => c) x i j = 0 := by sorry

/-- **Reindex / gather Jacobian** (foundation):
$\frac{\partial y_{\sigma(k)}}{\partial x_i} = \delta_{i, \sigma(k)}$

Backbone of every reshape/transpose/slice operation
(`Mat.flatten`, `Mat.unflatten`, attention head extraction). -/
theorem chk_pdiv_reindex {a b : Nat} (σ : Fin b → Fin a) (x : Vec a)
    (i : Fin a) (j : Fin b) :
    pdiv (fun y : Vec a => fun k : Fin b => y (σ k)) x i j =
    if i = σ j then 1 else 0 := by sorry

/-- **Linearity over finite sums** (foundation):
$\frac{\partial}{\partial x_i} \sum_{s \in S} f_s(y)_j = \sum_{s \in S} \frac{\partial f_s(y)_j}{\partial x_i}$

Extension of `pdiv_add` to arbitrary index sets. Load-bearing for
conv2d / depthwise weight gradients (their inner sums over kernel windows). -/
theorem chk_pdiv_finset_sum {m n : Nat} {α : Type*} [DecidableEq α]
    (S : Finset α) (f : α → Vec m → Vec n) (x : Vec m)
    (hdiff : ∀ s ∈ S, DifferentiableAt ℝ (f s) x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => ∑ s ∈ S, f s y k) x i j =
    ∑ s ∈ S, pdiv (f s) x i j := by sorry

-- Mat-level structural rules ───────────────────────────────────────

/-- **Row-independence for matrices** (foundation, was the last surviving
Mat-level axiom in earlier drafts; now proved):
A function that applies a vector-to-vector $g$ row-wise to a matrix has
a block-diagonal Jacobian: only entries within the same row see each other.

Lifts any proved $\text{Vec} \to \text{Vec}$ result to a row-wise matrix
operation (per-token `softmax`, per-token `layerNorm`, per-token `gelu`,
per-token dense). -/
theorem chk_pdivMat_rowIndep {m n p : Nat} (g : Vec n → Vec p)
    (h_g_diff : Differentiable ℝ g)
    (A : Mat m n) (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin p) :
    pdivMat (fun M : Mat m n => fun r => g (M r)) A i j k l =
    if i = k then pdiv g (A i) j l else 0 := by sorry

/-- **Matrix-level chain rule**: same form as the scalar `pdiv_comp`,
applied to functions $\text{Mat} \to \text{Mat}$ via the
`Mat.flatten`/`Mat.unflatten` bijection.

Glues the four pieces of attention's SDPA backward (matmul → scale →
rowSoftmax → matmul). -/
theorem chk_pdivMat_comp {a b c d e f : Nat}
    (F : Mat a b → Mat c d) (G : Mat c d → Mat e f)
    (A : Mat a b)
    (hF_diff : DifferentiableAt ℝ
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A))
    (hG_diff : DifferentiableAt ℝ
      (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u))) (Mat.flatten (F A)))
    (i : Fin a) (j : Fin b) (k : Fin e) (l : Fin f) :
    pdivMat (G ∘ F) A i j k l =
    ∑ p : Fin c, ∑ q : Fin d,
      pdivMat F A i j p q * pdivMat G (F A) p q k l := by sorry

/-- **Matmul Jacobian, left factor fixed**:
$\frac{\partial (C \cdot B)_{k,l}}{\partial B_{i,j}} = \delta_{l,j} \cdot C_{k,i}$

The "input gradient through a matmul" — building block for SDPA's
score computation $Q \cdot K^T$ and attention output $\text{weights} \cdot V$. -/
theorem chk_pdivMat_matmul_left_const {m p q : Nat} (C : Mat m p) (B : Mat p q)
    (i : Fin p) (j : Fin q) (k : Fin m) (l : Fin q) :
    pdivMat (fun B' : Mat p q => Mat.mul C B') B i j k l =
    if l = j then C k i else 0 := by sorry

/-- **Scalar-scale Jacobian**:
$\frac{\partial (s \cdot M)_{k,l}}{\partial M_{i,j}} = s \cdot \delta_{i,k} \delta_{j,l}$

Diagonal Jacobian for elementwise scaling. Used by attention's
$1/\sqrt{d_k}$ scale factor. -/
theorem chk_pdivMat_scalarScale {m n : Nat} (s : ℝ) (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin n) :
    pdivMat (fun M : Mat m n => fun r c => s * M r c) A i j k l =
    if i = k ∧ j = l then s else 0 := by sorry

/-- **Transpose Jacobian** (a permutation):
$\frac{\partial M^T_{k,l}}{\partial M_{i,j}} = \delta_{j,k} \delta_{i,l}$

Used by attention's $K^T$ in the score computation
$\text{scores} = Q \cdot K^T$. -/
theorem chk_pdivMat_transpose {m n : Nat} (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin n) (l : Fin m) :
    pdivMat (fun M : Mat m n => Mat.transpose M) A i j k l =
    if j = k ∧ i = l then 1 else 0 := by sorry

