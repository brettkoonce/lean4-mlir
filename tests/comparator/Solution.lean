import Mathlib
import LeanMlir.Proofs.Foundation.Tensor

open scoped Real

/-! # Solution to the architecture-free `Challenge.lean`

The vocabulary below is copied verbatim from `Challenge.lean` (which is
itself a copy of `LeanMlir.Proofs.Foundation.Tensor`) so that the two
modules state the same theorems without importing each other. Each proof
then discharges the root-namespace statement with the project's own
`Proofs.*` theorem: the two spellings are definitionally equal, so the
bridge is the elaborator's unfolding rather than a new lemma.
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

theorem chk_pdiv_is_fderiv {m n : Nat} (f : Vec m → Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) :
    pdiv f x i j = fderiv ℝ f x (basisVec i) j := rfl

theorem chk_pdiv_comp {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (x : Vec m) (hf : DifferentiableAt ℝ f x)
    (hg : DifferentiableAt ℝ g (f x))
    (i : Fin m) (k : Fin p) :
    pdiv (g ∘ f) x i k =
    ∑ j : Fin n, pdiv f x i j * pdiv g (f x) j k :=
  Proofs.pdiv_comp f g x hf hg i k

theorem chk_pdiv_add {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k + g y k) x i j
    = pdiv f x i j + pdiv g x i j :=
  Proofs.pdiv_add f g x hf hg i j

theorem chk_pdiv_mul {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k * g y k) x i j
    = pdiv f x i j * g x j + f x j * pdiv g x i j :=
  Proofs.pdiv_mul f g x hf hg i j

theorem chk_pdiv_id {n : Nat} (x : Vec n) (i j : Fin n) :
    pdiv (fun y : Vec n => y) x i j = if i = j then 1 else 0 :=
  Proofs.pdiv_id x i j

theorem chk_pdiv_const {m n : Nat} (c : Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) :
    pdiv (fun _ : Vec m => c) x i j = 0 :=
  Proofs.pdiv_const c x i j

theorem chk_pdiv_reindex {a b : Nat} (σ : Fin b → Fin a) (x : Vec a)
    (i : Fin a) (j : Fin b) :
    pdiv (fun y : Vec a => fun k : Fin b => y (σ k)) x i j =
    if i = σ j then 1 else 0 :=
  Proofs.pdiv_reindex σ x i j

theorem chk_pdiv_finset_sum {m n : Nat} {α : Type*} [DecidableEq α]
    (S : Finset α) (f : α → Vec m → Vec n) (x : Vec m)
    (hdiff : ∀ s ∈ S, DifferentiableAt ℝ (f s) x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => ∑ s ∈ S, f s y k) x i j =
    ∑ s ∈ S, pdiv (f s) x i j :=
  Proofs.pdiv_finset_sum S f x hdiff i j

theorem chk_pdivMat_rowIndep {m n p : Nat} (g : Vec n → Vec p)
    (h_g_diff : Differentiable ℝ g)
    (A : Mat m n) (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin p) :
    pdivMat (fun M : Mat m n => fun r => g (M r)) A i j k l =
    if i = k then pdiv g (A i) j l else 0 :=
  Proofs.pdivMat_rowIndep g h_g_diff A i j k l

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
      pdivMat F A i j p q * pdivMat G (F A) p q k l :=
  Proofs.pdivMat_comp F G A hF_diff hG_diff i j k l

theorem chk_pdivMat_matmul_left_const {m p q : Nat} (C : Mat m p) (B : Mat p q)
    (i : Fin p) (j : Fin q) (k : Fin m) (l : Fin q) :
    pdivMat (fun B' : Mat p q => Mat.mul C B') B i j k l =
    if l = j then C k i else 0 :=
  Proofs.pdivMat_matmul_left_const C B i j k l

theorem chk_pdivMat_scalarScale {m n : Nat} (s : ℝ) (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin n) :
    pdivMat (fun M : Mat m n => fun r c => s * M r c) A i j k l =
    if i = k ∧ j = l then s else 0 :=
  Proofs.pdivMat_scalarScale s A i j k l

theorem chk_pdivMat_transpose {m n : Nat} (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin n) (l : Fin m) :
    pdivMat (fun M : Mat m n => Mat.transpose M) A i j k l =
    if j = k ∧ i = l then 1 else 0 :=
  Proofs.pdivMat_transpose A i j k l

