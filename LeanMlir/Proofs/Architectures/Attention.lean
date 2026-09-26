import LeanMlir.Proofs.Foundation.Tensor
import LeanMlir.Proofs.Foundation.MLP
import LeanMlir.Proofs.Architectures.Softmax
import LeanMlir.Proofs.Architectures.CNN          -- needed for Kernel4 in patchEmbed
import LeanMlir.Proofs.Architectures.Residual
import LeanMlir.Proofs.Architectures.SE
import LeanMlir.Proofs.Architectures.LayerNorm
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.Complex.Trigonometric

/-!
# Attention — the Capstone

The fanciest architectural primitive in modern vision and language
models, formalized in one file. If you're reading the book straight
through, this is the chapter where everything you've learned clicks
together and you realize **there's nothing left to learn**.

## The cast of characters

Scaled dot-product attention:

    out = softmax((Q * K^T) / sqrt(d)) * V

where `Q = X Wq`, `K = X Wk`, `V = X Wv` — three dense projections of
the same input `X`. Every piece is something we already have:

| Piece                 | Chapter            | VJP move                 |
|-----------------------|--------------------|--------------------------|
| `Q = X Wq`            | `MLP.lean`         | dense backward           |
| `K = X Wk`            | `MLP.lean`         | dense backward           |
| `V = X Wv`            | `MLP.lean`         | dense backward           |
| `Q * K^T`             | (matmul = dense)   | chain rule               |
| `/ sqrt(d)`           | (scalar)           | chain rule + scale       |
| **`softmax(...)`**    | **`Softmax.lean`** | **closed-form collapse** |
| `... * V`             | (matmul = dense)   | chain rule               |
| three-way fan-in at X | `Residual.lean`    | `biPathHasVJP`         |

So the **only genuinely new ingredient in attention** is the standalone
softmax VJP (`softmaxHasVJP`, separate from the CE-loss gradient). Once
that's in hand, everything else is composition via tools we built in
earlier chapters.

## Structure of this file

0. **The multi-head matrix kit** — column-slab independence (`pdivMat_colIndep`,
   `colSlabwiseHasVJPMat`) and the ternary matrix VJP `HasVJPMat3`; then the
   differentiability helpers.
1. The standalone softmax VJP is in `Softmax.lean` (imported here).
2. **Scaled dot-product attention** — SDPA as a composition.
3. **Multi-head wrapper** — reshape/transpose boilerplate, no new math.
4. **Transformer block** — LN -> MHSA -> + -> LN -> MLP -> +, pure composition.
5. **Final commentary** — why the taxonomy is complete.
6. **Bridging ranks** — patch embedding, classifier head, and the weight-tied
   whole-ViT witness `vitFullHasVJP` (`vitFullHasVJP_correct`).
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Column-slab independence (vmap over a column-axis partition)
-- ════════════════════════════════════════════════════════════════

/-! ## Per-head / per-slab column independence

Multi-head attention applies the same per-head function to each of `heads`
column slabs of width `d_in` from a `Mat n (heads * d_in)` input. The
column-slab analog of `rowwiseHasVJPMat` factors that vmap-over-heads
structure: each head's output depends only on its own slab of the input,
so the matrix Jacobian is block-diagonal across the head axis. -/

/-- Apply `g : Mat n d_in → Mat n d_out` to each of the `heads` column
    slabs of width `d_in` in a `Mat n (heads * d_in)` input, producing
    a `Mat n (heads * d_out)` output. Output column `(h, j_out)` is
    column `j_out` of `g (slab h M)`, where `slab h M` extracts the
    `d_in`-wide column block at head index `h`. -/
noncomputable def colSlabApply {n heads d_in d_out : Nat}
    (g : Mat n d_in → Mat n d_out) : Mat n (heads * d_in) → Mat n (heads * d_out) :=
  fun M => fun r hj =>
    g (fun r' j_in => M r' (finProdFinEquiv ((finProdFinEquiv.symm hj).1, j_in)))
      r (finProdFinEquiv.symm hj).2

/-- **Column-slab independence Jacobian** — column-axis analog of
    `pdivMat_rowIndep`. For a slab-applied function `colSlabApply g`,
    the Jacobian is block-diagonal across the `heads` axis: zero unless
    the input slab `h_j` matches the output slab `h_l`, otherwise equal
    to `pdivMat g` on that slab.

    Requires `Differentiable ℝ (flat g)` for the same reason as
    `pdivMat_rowIndep`: the Pi-valued flat form must be differentiable
    everywhere so `fderiv` doesn't fall back to junk-default 0. -/
theorem pdivMat_colIndep {n heads d_in d_out : Nat} (g : Mat n d_in → Mat n d_out)
    (h_g_diff : Differentiable ℝ
                  (fun v : Vec (n * d_in) => Mat.flatten (g (Mat.unflatten v))))
    (A : Mat n (heads * d_in))
    (i : Fin n) (h_j : Fin heads) (j' : Fin d_in)
    (k : Fin n) (h_l : Fin heads) (j'' : Fin d_out) :
    pdivMat (colSlabApply g) A
            i (finProdFinEquiv (h_j, j'))
            k (finProdFinEquiv (h_l, j'')) =
    (if h_j = h_l then
      pdivMat g (fun r' j_in => A r' (finProdFinEquiv (h_l, j_in))) i j' k j''
     else 0) := by
  -- `slab h` reads head `h`'s columns out of the flat input; output coordinate `(r, (h, c))` is
  -- `g`'s flat coordinate `(r, c)` read after `slab h`, whose derivative is `D r h c`.
  let slab : Fin heads → (Vec (n * (heads * d_in)) →L[ℝ] Vec (n * d_in)) := fun h =>
    reindexCLM fun idx => finProdFinEquiv ((finProdFinEquiv.symm idx).1,
      finProdFinEquiv (h, (finProdFinEquiv.symm idx).2))
  let G := fun w : Vec (n * d_in) => Mat.flatten (g (Mat.unflatten w))
  let D : Fin n → Fin heads → Fin d_out → (Vec (n * (heads * d_in)) →L[ℝ] ℝ) := fun r h c =>
    (ContinuousLinearMap.proj (finProdFinEquiv (r, c)) : Vec (n * d_out) →L[ℝ] ℝ).comp
      ((fderiv ℝ G (slab h (Mat.flatten A))).comp (slab h))
  have hF : HasFDerivAt (fun v => Mat.flatten (colSlabApply g (Mat.unflatten v)))
      (ContinuousLinearMap.pi fun idx => D (finProdFinEquiv.symm idx).1
        (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).1
        (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).2) (Mat.flatten A) :=
    hasFDerivAt_pi.2 fun idx => by
      obtain ⟨⟨r, hc⟩, rfl⟩ := finProdFinEquiv.surjective idx
      obtain ⟨⟨h, c⟩, rfl⟩ := finProdFinEquiv.surjective hc
      rw [show (fun v : Vec (n * (heads * d_in)) => Mat.flatten (colSlabApply g (Mat.unflatten v))
          (finProdFinEquiv (r, finProdFinEquiv (h, c)))) = fun v => G (slab h v)
          (finProdFinEquiv (r, c)) by
        funext v; simp only [G, slab, reindexCLM_apply]
        unfold Mat.flatten Mat.unflatten colSlabApply; simp only [Equiv.symm_apply_apply]]
      simp only [Equiv.symm_apply_apply]
      exact hasFDerivAt_pi'.1 ((h_g_diff _).hasFDerivAt.comp _ (slab h).hasFDerivAt) _
  have hslab : slab h_l (Mat.flatten A) =
      Mat.flatten (fun r' j_in => A r' (finProdFinEquiv (h_l, j_in))) := by
    funext; simp only [slab, Mat.flatten, reindexCLM_apply, Equiv.symm_apply_apply]
  have hb : slab h_l (basisVec (finProdFinEquiv (i, finProdFinEquiv (h_j, j')))) =
      if h_j = h_l then basisVec (finProdFinEquiv (i, j')) else 0 := by
    funext idx; obtain ⟨⟨r, c⟩, rfl⟩ := finProdFinEquiv.surjective idx
    simp only [slab, reindexCLM_apply, Equiv.symm_apply_apply, basisVec_apply,
      EmbeddingLike.apply_eq_iff_eq, Prod.mk.injEq]
    rcases eq_or_ne h_j h_l with rfl | hne
    · simp
    · simp [hne, hne.symm]
  rw [pdivMat, pdiv, hF.fderiv]
  simp only [ContinuousLinearMap.pi_apply, Equiv.symm_apply_apply, D,
    ContinuousLinearMap.comp_apply, ContinuousLinearMap.proj_apply, hslab, hb]
  split_ifs <;> simp [G, pdivMat, pdiv]

/-- **Lift `HasVJPMat g` to column-slab vmap** — column-axis analog of
    `rowwiseHasVJPMat`. Given `g : Mat n d_in → Mat n d_out` with a
    matrix VJP, applying `g` independently to each of `heads`-many column
    slabs gives a `HasVJPMat` for `colSlabApply g`. The backward applies
    `g.backward` per slab. -/
noncomputable def colSlabwiseHasVJPMat {n heads d_in d_out : Nat}
    {g : Mat n d_in → Mat n d_out}
    (hg : HasVJPMat g)
    (hg_diff : Differentiable ℝ
                 (fun v : Vec (n * d_in) => Mat.flatten (g (Mat.unflatten v)))) :
    HasVJPMat (colSlabApply g (heads := heads)) where
  backward := fun M dY r hj =>
    hg.backward (fun r' j_in => M r' (finProdFinEquiv ((finProdFinEquiv.symm hj).1, j_in)))
                (fun r' j_out => dY r' (finProdFinEquiv ((finProdFinEquiv.symm hj).1, j_out)))
                r (finProdFinEquiv.symm hj).2
  correct := by
    intro M dY i jj
    obtain ⟨⟨h, j'⟩, rfl⟩ := finProdFinEquiv.surjective jj
    simp only [Equiv.symm_apply_apply, sum_finProdFinEquiv (m := heads),
      pdivMat_colIndep g hg_diff]
    simp [hg.correct]

-- ════════════════════════════════════════════════════════════════
-- § Ternary VJP for matrix functions (HasVJPMat3)
-- ════════════════════════════════════════════════════════════════

/-! ## Ternary matrix VJP

For ternary-input functions like SDPA `(Q, K, V) ↦ out`, package the
three per-input VJPs as a single structure analogous to `HasVJPMat`.
The backward returns the triple of per-input gradients; correctness
holds independently for each input (with the others fixed). -/

/-- VJP structure for `Mat × Mat × Mat → Mat` functions where all
    three inputs share the same shape `Mat n d_in` and the output is
    `Mat n d_out`. Backward returns the triple of per-input gradients;
    `correct_{1,2,3}` ensure each gradient matches the partial derivative
    treating the other two inputs as constants. -/
structure HasVJPMat3 {n d_in d_out : Nat}
    (F : Mat n d_in → Mat n d_in → Mat n d_in → Mat n d_out) where
  backward : Mat n d_in → Mat n d_in → Mat n d_in → Mat n d_out →
             (Mat n d_in × Mat n d_in × Mat n d_in)
  correct_1 : ∀ A B C dY i j,
    (backward A B C dY).1 i j =
    ∑ k : Fin n, ∑ l : Fin d_out,
      pdivMat (fun A' => F A' B C) A i j k l * dY k l
  correct_2 : ∀ A B C dY i j,
    (backward A B C dY).2.1 i j =
    ∑ k : Fin n, ∑ l : Fin d_out,
      pdivMat (fun B' => F A B' C) B i j k l * dY k l
  correct_3 : ∀ A B C dY i j,
    (backward A B C dY).2.2 i j =
    ∑ k : Fin n, ∑ l : Fin d_out,
      pdivMat (fun C' => F A B C') C i j k l * dY k l

-- ════════════════════════════════════════════════════════════════
-- § 0. Differentiable helpers for the matrix-VJP building blocks
--
-- After the foundation flip, every `vjpMatComp` and `biPathMatHasVJP`
-- call requires `Differentiable` evidence for the flattened versions of
-- the composed matrix functions. The four helpers below cover the linear
-- building blocks (matmul-by-const-left/right, scalar-scale, transpose);
-- non-linear ingredients (rowSoftmax, layerNorm, gelu) get dedicated
-- Diff theorems further down where they're introduced.
-- ════════════════════════════════════════════════════════════════

lemma matmul_right_const_flat_differentiable {m p q : Nat} (D : Mat p q) :
    Differentiable ℝ (fun v : Vec (m * p) =>
      Mat.flatten (Mat.mul (Mat.unflatten v) D)) := by
  unfold Mat.unflatten Mat.flatten Mat.mul; fun_prop

lemma matmul_left_const_flat_differentiable {m p q : Nat} (C : Mat m p) :
    Differentiable ℝ (fun v : Vec (p * q) =>
      Mat.flatten (Mat.mul C (Mat.unflatten v))) := by
  unfold Mat.unflatten Mat.flatten Mat.mul; fun_prop

lemma scalarScale_flat_differentiable {m n : Nat} (s : ℝ) :
    Differentiable ℝ (fun v : Vec (m * n) =>
      Mat.flatten (fun r c => s * (Mat.unflatten v) r c)) := by
  unfold Mat.unflatten Mat.flatten; fun_prop

lemma transpose_flat_differentiable {m n : Nat} :
    Differentiable ℝ (fun v : Vec (m * n) =>
      Mat.flatten (Mat.transpose (Mat.unflatten v) : Mat n m)) := by
  unfold Mat.unflatten Mat.flatten Mat.transpose; fun_prop

/-- Differentiability of the flattened per-token dense map.
    `fun X => fun n => dense W b (X n)` is linear in `X`, so the
    flattened version is `Differentiable` everywhere. -/
lemma dense_per_token_flat_differentiable {N inD outD : Nat}
    (W : Mat inD outD) (b : Vec outD) :
    Differentiable ℝ (fun v : Vec (N * inD) =>
      Mat.flatten ((fun X : Mat N inD => fun n => dense W b (X n))
                   (Mat.unflatten v))) := by
  unfold Mat.unflatten Mat.flatten dense; fun_prop

/-- Differentiability of the flattened per-token GELU map.
    `geluScalar = 0.5 · x · (1 + tanh(√(2/π)(x + 0.044715·x³)))`. With
    `differentiable_tanh` available to `fun_prop`, the proof
    discharges automatically. -/
theorem gelu_per_token_flat_differentiable (N D : Nat) :
    Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten ((fun X : Mat N D => fun n => gelu D (X n))
                   (Mat.unflatten v))) := by
  unfold Mat.unflatten Mat.flatten gelu geluScalar; fun_prop

/-- Differentiability of `layerNormForward D ε γ β` — it is `bnForward` (definitionally),
    differentiable when `ε > 0`. Tagged for `fun_prop`. -/
@[fun_prop]
lemma layerNorm_differentiable (D : Nat) (ε γ β : ℝ) (hε : 0 < ε) :
    Differentiable ℝ (layerNormForward D ε γ β) := by
  exact bnForward_differentiable D ε γ β hε

/-- Differentiability of the flattened per-token LayerNorm map: each output coordinate is a
    coordinate of `layerNormForward` (`layerNorm_differentiable`) applied to one row of the input. -/
theorem layerNorm_per_token_flat_differentiable (N D : Nat) (ε γ β : ℝ) (hε : 0 < ε) :
    Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten ((fun X : Mat N D => fun n => layerNormForward D ε γ β (X n))
                   (Mat.unflatten v))) := by
  unfold Mat.flatten Mat.unflatten; fun_prop (disch := assumption)

/-- Differentiability of the flattened identity matrix map.
    `Mat.flatten ∘ id ∘ Mat.unflatten = id` on `Vec (a*b)`. -/
lemma identity_mat_flat_differentiable (a b : Nat) :
    Differentiable ℝ (fun v : Vec (a * b) =>
      Mat.flatten ((fun X : Mat a b => X) (Mat.unflatten v))) := by
  simp only [Mat.flatten_unflatten]; exact differentiable_id

/-- Differentiability of a flattened composition `G ∘ F` from those of `F` and `G`: the
    `Mat.unflatten (Mat.flatten _)` round trip in the middle is the identity. -/
lemma flat_differentiable_comp {a b c d e f : Nat} {F : Mat a b → Mat c d} {G : Mat c d → Mat e f}
    (hF : Differentiable ℝ (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))))
    (hG : Differentiable ℝ (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u)))) :
    Differentiable ℝ (fun v : Vec (a * b) => Mat.flatten ((G ∘ F) (Mat.unflatten v))) := by
  simpa [Function.comp_def, Mat.unflatten_flatten] using hG.comp hF

-- ════════════════════════════════════════════════════════════════
-- § 2. Scaled Dot-Product Attention
-- ════════════════════════════════════════════════════════════════

/-! ## Attention as a composition

For a single sequence of `n` tokens, each with feature dim `d`, let
`X : Mat n d` be the input. Attention produces `out : Mat n d` via:

    Q = X * Wq        -- (n x d), dense projection
    K = X * Wk        -- (n x d)
    V = X * Wv        -- (n x d)
    scores = Q * K^T   -- (n x n)
    scaled = scores / sqrt(d)
    weights = softmax_row(scaled)   -- softmax applied per row
    out = weights * V               -- (n x d)

Because the input `X` is a matrix, we need matrix-level types. We work
with `Mat n d` throughout this section (already defined in `Tensor.lean`).

**Row-wise softmax** is just "apply the 1D softmax to each row
independently." Its VJP is just "apply the 1D softmax VJP to each row
independently." No new derivation; the fan-out structure is trivially
parallel.
-/

/-- Row-wise softmax of a matrix. -/
noncomputable def rowSoftmax {m n : Nat} (A : Mat m n) : Mat m n :=
  fun i => softmax n (A i)

/-- **Smoothness of `rowSoftmax`**: each output coordinate of the flattened map is a
    coordinate of `softmax n` (`softmax_differentiable`) applied to one row of the input. -/
theorem rowSoftmax_flat_differentiable (m n : Nat) :
    Differentiable ℝ (fun v : Vec (m * n) =>
      Mat.flatten (rowSoftmax (Mat.unflatten v) : Mat m n)) := by
  have := softmax_differentiable n
  unfold rowSoftmax Mat.flatten Mat.unflatten; fun_prop

/-- **Row-wise softmax VJP** — proved, no sorry.

    Rows are independent, so the Jacobian is block-diagonal with the
    standalone softmax Jacobian in each block. The backward just
    applies `softmaxHasVJP` per row; correctness is `rowwiseHasVJPMat`'s. -/
noncomputable def rowSoftmaxHasVJPMat {m n : Nat} :
    HasVJPMat (fun A : Mat m n => fun r => softmax n (A r)) where
  backward := fun A dY => fun r c => (softmaxHasVJP n).backward (A r) (dY r) c
  correct := (rowwiseHasVJPMat (softmaxHasVJP n) (softmax_differentiable n)).correct

/-- Alias so `rowSoftmaxHasVJPMat` types against the actual `rowSoftmax`
    definition (definitionally equal, but lets Lean unify on the name). -/
noncomputable def rowSoftmaxHasVJPMat' (m n : Nat) :
    HasVJPMat (@rowSoftmax m n) :=
  rowSoftmaxHasVJPMat

/-- **Scaled dot-product attention**, for a single sequence and a
    single head. `Q K V : Mat n d`.

    `sdpa Q K V = softmax_row(Q * K^T / sqrt(d)) * V`

    MLIR (`MlirCodegen.emitMHSAForward`):
      %mh_sc   = dot_general %mh_q, %mh_k, contracting_dims = [3] x [3]
      %mh_ss   = multiply %mh_sc, broadcast(1/sqrt(d))
      %mh_sm   = softmax(%mh_ss) -- via reduce max, shift, exp, reduce sum, divide
      %mh_av   = dot_general %mh_sm, %mh_v, contracting_dims = [3] x [2]
-/
noncomputable def sdpa (n d : Nat) (Q K V : Mat n d) : Mat n d :=
  let scores : Mat n n := Mat.mul Q (Mat.transpose K)
  let scale : ℝ := 1 / Real.sqrt (↑d)
  let scaled : Mat n n := fun i j => scale * scores i j
  let weights : Mat n n := rowSoftmax scaled
  Mat.mul weights V

/-! ## The backward pass through SDPA (by hand, then compositionally)

Working backward from `d_out : Mat n d`, four steps:

**Step 1.** Through the final matmul `out = weights * V`. By the dense
layer VJP generalized to matrices (same derivation as `denseHasVJP`,
just with a batch dimension):

    d_V       = weights^T * d_out     -- (n x d)
    d_weights = d_out * V^T           -- (n x n)

**Step 2.** Through the per-row softmax. Each row is independent, so
we apply `softmaxHasVJP` row-by-row:

    d_scaled_i = weights_i * (d_weights_i - <weights_i, d_weights_i> * 1)

**Step 3.** Through the scalar scale `scaled = scores / sqrt(d)`. Just
divide the incoming gradient by `sqrt(d)`:

    d_scores = d_scaled / sqrt(d)

**Step 4.** Through `scores = Q * K^T`. Same matrix-matmul VJP as
step 1, but now Q and K both flow back:

    d_Q = d_scores * K                       -- (n x d)
    d_K = d_scores^T * Q                     -- (n x d)

**Step 5.** Three parallel dense backwards from Q, K, V back to X.
Each uses `denseHasVJP`:

    d_X_via_Q = d_Q * Wq^T
    d_X_via_K = d_K * Wk^T
    d_X_via_V = d_V * Wv^T

**Step 6.** Fan-in at X — the three paths **add**:

    d_X = d_X_via_Q + d_X_via_K + d_X_via_V

This is `biPathHasVJP` from `Residual.lean`, applied twice (to
combine three paths). The three-way fan-in **is** the attention
backward pass at the input. Q, K, V are parallel branches reading
from `X`, so their gradients accumulate at `X`.

And the parameter gradients (for W_q, W_k, W_v, W_o) are collected
at each dense layer along the way — exactly as with any other dense
layer in the book.

**There is no novel structural move in attention.** It's three dense
layers, two matmuls, one row-softmax, one scale, and a three-way
fan-in. Every piece has been proved. The composition is mechanical.
-/

/-! ### The backward, concretely

This section gives:

1. **Concrete definitions** of `sdpaBackQ`, `sdpaBackK`, `sdpaBackV`
   transcribed from the step-by-step derivation above.
2. **Correctness theorems** `sdpaBackQ_correct`, `sdpaBackK_correct`,
   `sdpaBackV_correct`, stated in terms of `pdivMat` (the matrix-level
   partial derivative primitive from `Tensor.lean`), each proved
   compositionally as a `vjpMatComp` chain (four steps for Q).

The concrete formulas are also numerically gradient-checked in
`check_jacobians.py` (`test_sdpaBackQ/K/V`) for cross-validation.
-/

/-- `1 / sqrt(d)`, the SDPA scale factor. -/
noncomputable def sdpaScale (d : Nat) : ℝ := 1 / Real.sqrt (↑d)

/-- Softmax-weights under the SDPA scale, reused by all three backwards. -/
noncomputable def sdpaWeights (n d : Nat) (Q K : Mat n d) : Mat n n :=
  let scores : Mat n n := Mat.mul Q (Mat.transpose K)
  let scaled : Mat n n := fun i j => sdpaScale d * scores i j
  rowSoftmax scaled

/-- Gradient flowing into `weights` from the final matmul `out = weights · V`. -/
noncomputable def sdpaDWeights {n d : Nat} (V dOut : Mat n d) : Mat n n :=
  Mat.mul dOut (Mat.transpose V)

/-- Per-row softmax VJP: `p_i * (dw_i - <p_i, dw_i>)`. -/
noncomputable def sdpaDScaled (n d : Nat) (Q K V dOut : Mat n d) : Mat n n :=
  let p : Mat n n := sdpaWeights n d Q K
  let dw : Mat n n := sdpaDWeights V dOut
  fun i j =>
    let s : ℝ := ∑ k : Fin n, p i k * dw i k
    p i j * (dw i j - s)

/-- Gradient w.r.t. the pre-softmax scores, after undoing the `/ sqrt(d)` scale. -/
noncomputable def sdpaDScores (n d : Nat) (Q K V dOut : Mat n d) : Mat n n :=
  fun i j => sdpaScale d * sdpaDScaled n d Q K V dOut i j

/-- **Backward w.r.t. Q**: `dQ = dScores · K`. -/
noncomputable def sdpaBackQ (n d : Nat) (Q K V dOut : Mat n d) : Mat n d :=
  Mat.mul (sdpaDScores n d Q K V dOut) K

/-- **Backward w.r.t. K**: `dK = dScores^T · Q`. -/
noncomputable def sdpaBackK (n d : Nat) (Q K V dOut : Mat n d) : Mat n d :=
  Mat.mul (Mat.transpose (sdpaDScores n d Q K V dOut)) Q

/-- **Backward w.r.t. V**: `dV = weights^T · dOut`. (V does not appear on the
    RHS: `V`'s gradient flows only through the final matmul, not through
    `weights`.) -/
noncomputable def sdpaBackV (n d : Nat) (Q K _V dOut : Mat n d) : Mat n d :=
  Mat.mul (Mat.transpose (sdpaWeights n d Q K)) dOut

/-! ## Q and K correctness via compositional SDPA forward chain

For Q (with K, V fixed), `sdpa n d · K V` is the composition:

    Q ↦ Q · K^T   ↦   scale * _   ↦   rowSoftmax _   ↦   _ · V

Four steps, four already-proved `HasVJPMat` building blocks:

1. `matmulRightConstHasVJP (Mat.transpose K)` — ∂(Q · K^T)/∂Q
2. `scalarScaleHasVJP (sdpaScale d)` — ∂(scale · scores)/∂scores
3. `rowSoftmaxHasVJPMat` — ∂(rowSoftmax scaled)/∂scaled
4. `matmulRightConstHasVJP V` — ∂(weights · V)/∂weights

Chain them with `vjpMatComp` thrice → a `HasVJPMat` for the full
Q-path. Then show the chain's backward function equals `sdpaBackQ`
pointwise (trivial — the chain's backward literally computes the same
nested formula) and invoke its `.correct` to discharge the goal. -/

/-- Explicit 4-composition forward for SDPA, varying Q with K, V fixed. -/
noncomputable def sdpaQChain (n d : Nat) (K V : Mat n d) : Mat n d → Mat n d :=
  (fun w : Mat n n => Mat.mul w V) ∘
  (@rowSoftmax n n) ∘
  (fun s : Mat n n => fun r c => sdpaScale d * s r c) ∘
  (fun Q' : Mat n d => Mat.mul Q' (Mat.transpose K))

theorem sdpaQChain_eq (n d : Nat) (Q K V : Mat n d) :
    sdpaQChain n d K V Q = sdpa n d Q K V := by
  unfold sdpaQChain sdpa sdpaScale
  rfl

/-- `HasVJPMat` for the chain — built by nesting `vjpMatComp` thrice. -/
noncomputable def sdpaQChainHasVJP (n d : Nat) (K V : Mat n d) :
    HasVJPMat (sdpaQChain n d K V) :=
  -- Innermost (matmul Q' Kt → scalar scale):
  let innerHasVJP :=
    vjpMatComp _ (fun s : Mat n n => fun r c => sdpaScale d * s r c)
      (matmul_right_const_flat_differentiable (Mat.transpose K))
      (scalarScale_flat_differentiable (sdpaScale d))
      (matmulRightConstHasVJP (Mat.transpose K))
      (scalarScaleHasVJP (sdpaScale d))
  -- Diff of the innermost composition (scalar_scale ∘ matmul_right_const) — linear in v.
  have inner_diff : Differentiable ℝ
      (fun v : Vec (n * d) =>
        Mat.flatten ((fun s : Mat n n => fun r c => sdpaScale d * s r c)
          ((fun Q' : Mat n d => Mat.mul Q' (Mat.transpose K)) (Mat.unflatten v)))) := by
    unfold Mat.unflatten Mat.flatten Mat.mul; fun_prop
  -- Middle chain (… → rowSoftmax):
  let middleHasVJP :=
    vjpMatComp _ (@rowSoftmax n n)
      inner_diff (rowSoftmax_flat_differentiable n n)
      innerHasVJP (rowSoftmaxHasVJPMat' n n)
  -- Diff of the middle composition (rowSoftmax ∘ scaled-matmul) via composition.
  have middle_diff : Differentiable ℝ
      (fun v : Vec (n * d) =>
        Mat.flatten ((@rowSoftmax n n) (((fun s : Mat n n => fun r c => sdpaScale d * s r c) ∘
          (fun Q' : Mat n d => Mat.mul Q' (Mat.transpose K))) (Mat.unflatten v)))) :=
    flat_differentiable_comp inner_diff (rowSoftmax_flat_differentiable n n)
  -- Outermost (… → matmul w V):
  vjpMatComp _ (fun w : Mat n n => Mat.mul w V)
    middle_diff (matmul_right_const_flat_differentiable V)
    middleHasVJP
    (matmulRightConstHasVJP V)

/-- **Correctness of `sdpaBackQ`** — proved, no sorry.

    Two moves: (1) replace `fun Q' => sdpa n d Q' K V` by the chain via
    `sdpaQChain_eq`; (2) apply the chain's `.correct` and verify that
    the chain's backward reduces to `sdpaBackQ` (pure unfolding). -/
theorem sdpaBackQ_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpaBackQ n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun Q' => sdpa n d Q' K V) Q i j k l * dOut k l := by
  have hfwd : (fun Q' : Mat n d => sdpa n d Q' K V) = sdpaQChain n d K V := by
    funext Q'; exact (sdpaQChain_eq n d Q' K V).symm
  rw [hfwd]
  rw [← (sdpaQChainHasVJP n d K V).correct Q dOut i j]
  -- Goal: sdpaBackQ ... = (sdpaQChainHasVJP ...).backward Q dOut i j
  unfold sdpaBackQ sdpaDScores sdpaDScaled sdpaDWeights sdpaWeights
    sdpaQChainHasVJP
  rfl

/-! ## K case

K enters through a transpose before the first matmul. One extra step in
the chain: K ↦ K^T, then follow the Q chain (but with the matmul being
"left factor constant" this time because Q is fixed and K^T is on the
right). -/

noncomputable def sdpaKChain (n d : Nat) (Q V : Mat n d) : Mat n d → Mat n d :=
  (fun w : Mat n n => Mat.mul w V) ∘
  (@rowSoftmax n n) ∘
  (fun s : Mat n n => fun r c => sdpaScale d * s r c) ∘
  (fun Kt' : Mat d n => Mat.mul Q Kt') ∘
  (fun K' : Mat n d => Mat.transpose K')

theorem sdpaKChain_eq (n d : Nat) (Q K V : Mat n d) :
    sdpaKChain n d Q V K = sdpa n d Q K V := by
  unfold sdpaKChain sdpa sdpaScale
  rfl

noncomputable def sdpaKChainHasVJP (n d : Nat) (Q V : Mat n d) :
    HasVJPMat (sdpaKChain n d Q V) :=
  -- Innermost (transpose → matmul Q · Kt):
  let l1HasVJP :=
    vjpMatComp _ (fun Kt' : Mat d n => Mat.mul Q Kt')
      transpose_flat_differentiable
      (matmul_left_const_flat_differentiable Q)
      (@transposeHasVJP n d)
      (matmulLeftConstHasVJP Q)
  have l1_diff : Differentiable ℝ
      (fun v : Vec (n * d) =>
        Mat.flatten ((fun Kt' : Mat d n => Mat.mul Q Kt')
          (Mat.transpose (Mat.unflatten v : Mat n d) : Mat d n))) := by
    unfold Mat.unflatten Mat.flatten Mat.mul Mat.transpose; fun_prop
  -- Add scalar scale:
  let l2HasVJP :=
    vjpMatComp _ (fun s : Mat n n => fun r c => sdpaScale d * s r c)
      l1_diff
      (scalarScale_flat_differentiable (sdpaScale d))
      l1HasVJP
      (scalarScaleHasVJP (sdpaScale d))
  have l2_diff : Differentiable ℝ
      (fun v : Vec (n * d) =>
        Mat.flatten ((fun s : Mat n n => fun r c => sdpaScale d * s r c)
          ((fun Kt' : Mat d n => Mat.mul Q Kt')
            (Mat.transpose (Mat.unflatten v : Mat n d) : Mat d n)))) := by
    unfold Mat.unflatten Mat.flatten Mat.mul Mat.transpose; fun_prop
  -- Add rowSoftmax:
  let l3HasVJP :=
    vjpMatComp _ (@rowSoftmax n n)
      l2_diff (rowSoftmax_flat_differentiable n n)
      l2HasVJP (rowSoftmaxHasVJPMat' n n)
  have l3_diff : Differentiable ℝ
      (fun v : Vec (n * d) =>
        Mat.flatten ((@rowSoftmax n n) ((fun s : Mat n n => fun r c => sdpaScale d * s r c)
          ((fun Kt' : Mat d n => Mat.mul Q Kt')
            (Mat.transpose (Mat.unflatten v : Mat n d) : Mat d n))))) := by
    simpa [Function.comp_def, Mat.unflatten_flatten] using (rowSoftmax_flat_differentiable n n).comp l2_diff
  -- Outermost (… → matmul w V):
  vjpMatComp _ (fun w : Mat n n => Mat.mul w V)
    l3_diff (matmul_right_const_flat_differentiable V)
    l3HasVJP
    (matmulRightConstHasVJP V)

/-- **Correctness of `sdpaBackK`** — proved, no sorry.

    Same shape as Q, but the chain goes through a leading transpose
    step. The resulting backward computes `∑ k, Q k j * dScores k i`
    whereas `sdpaBackK` is `Mat.mul (Mat.transpose dScores) Q`, which
    expands to `∑ k, dScores k i * Q k j`. Equal by `mul_comm` at the
    summand level. -/
theorem sdpaBackK_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpaBackK n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun K' => sdpa n d Q K' V) K i j k l * dOut k l := by
  have hfwd : (fun K' : Mat n d => sdpa n d Q K' V) = sdpaKChain n d Q V := by
    funext K'; exact (sdpaKChain_eq n d Q K' V).symm
  rw [hfwd]
  rw [← (sdpaKChainHasVJP n d Q V).correct K dOut i j]
  unfold sdpaBackK sdpaDScores sdpaDScaled sdpaDWeights sdpaWeights
    sdpaKChainHasVJP vjpMatComp
    matmulRightConstHasVJP matmulLeftConstHasVJP transposeHasVJP
    scalarScaleHasVJP rowSoftmaxHasVJPMat' rowSoftmaxHasVJPMat
    softmaxHasVJP rowSoftmax
  -- Both sides now in sum-of-products form; differ only by mul_comm at the summand.
  simp only [Mat.mul, Mat.transpose, Function.comp]
  apply Finset.sum_congr rfl
  intro k _
  ring

/-- The final matmul in SDPA: for fixed Q, K, the function `V' ↦ sdpa Q K V'`
    is `V' ↦ W · V'` where `W = sdpaWeights Q K`. Pure rewrite; definitional. -/
theorem sdpa_eq_mul_weights (n d : Nat) (Q K V : Mat n d) :
    sdpa n d Q K V = Mat.mul (sdpaWeights n d Q K) V := by
  unfold sdpa sdpaWeights sdpaScale
  rfl

/-- **Correctness of `sdpaBackV`** — proved, no sorry.

    The V-path is the simplest case: `V'` only enters through the final
    matmul `out = weights · V'`. So `fun V' => sdpa n d Q K V'` is just
    `fun V' => Mat.mul W V'` where W is fixed (= `sdpaWeights n d Q K`),
    and the VJP comes directly from `matmulLeftConstHasVJP`. -/
theorem sdpaBackV_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpaBackV n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun V' => sdpa n d Q K V') V i j k l * dOut k l := by
  -- Replace `fun V' => sdpa n d Q K V'` by `fun V' => Mat.mul W V'`.
  have hfwd : (fun V' : Mat n d => sdpa n d Q K V') =
              (fun V' : Mat n d => Mat.mul (sdpaWeights n d Q K) V') := by
    funext V'; exact sdpa_eq_mul_weights n d Q K V'
  rw [hfwd]
  -- Apply the matmul VJP correctness backward (i.e., rewrite the RHS
  -- into the VJP's backward) and then match `sdpaBackV`.
  rw [← (matmulLeftConstHasVJP (sdpaWeights n d Q K)).correct V dOut i j]
  -- Goal: sdpaBackV n d Q K V dOut i j = Σ k, W k i * dOut k j
  unfold sdpaBackV Mat.mul Mat.transpose
  rfl

/-- **Bundled SDPA ternary VJP.** Packages `sdpaBackQ_correct`,
    `sdpaBackK_correct` and `sdpaBackV_correct` into a single `HasVJPMat3` instance. The backward triple
    `(sdpaBackQ, sdpaBackK, sdpaBackV)` gives per-input
    gradients; correctness is the three existing per-input theorems
    in one structure. -/
noncomputable def sdpaHasVJPMat3 (n d : Nat) :
    HasVJPMat3 (sdpa n d) where
  backward := fun Q K V dY =>
    (sdpaBackQ n d Q K V dY,
     sdpaBackK n d Q K V dY,
     sdpaBackV n d Q K V dY)
  correct_1 := sdpaBackQ_correct n d
  correct_2 := sdpaBackK_correct n d
  correct_3 := sdpaBackV_correct n d

-- ════════════════════════════════════════════════════════════════
-- § 3. Multi-Head wrapping (Phase 3 — proved via column-stacking)
-- ════════════════════════════════════════════════════════════════

/-! ## Multi-head: parallelism over a partition

Multi-head attention is:

  1. Project `X : Mat N D` three ways: `Q = X·Wq + bq`, `K = X·Wk + bk`, `V = X·Wv + bv`.
  2. Reshape each projection `(N, D) → (N, heads, d_head)` by slicing the feature axis.
  3. Run SDPA independently on each of the `heads` slices.
  4. Concatenate the head outputs back to `(N, D)`.
  5. Apply the output projection `Y = concat · Wo + bo`.

In the MLIR (`emitMHSAForward`):
    reshape (B, N, D) -> (B, N, H, D_h)
    transpose -> (B, H, N, D_h)
    [SDPA per head, using batching_dims = [0, 1]]
    transpose -> (B, N, H, D_h)
    reshape -> (B, N, D)
    dense projection (the "output projection" `Wo`)

This section proves `mhsaHasVJPMat` end-to-end: we *define*
`mhsaLayer` concretely in Lean (Q/K/V projections → per-head slice
→ sdpa-per-head → concat → Wo projection), then prove its `HasVJPMat`
via the `pdivMat_colIndep` + `colSlabwiseHasVJPMat` framework,
which lifts the per-head SDPA backward over the
head axis. Formula remains numerically gradient-checked in
`check_jacobians.py` for cross-validation. -/

/-- Multi-head SDPA on a single sequence: `Mat N (heads·d_head) → Mat N (heads·d_head)`.

    Concretely defined (not opaque):
    1. Q, K, V projections (each a per-token dense with its own Wq/Wk/Wv).
    2. For each head `h : Fin heads`, extract the `(N, d_head)` slice of Q/K/V
       by indexing `finProdFinEquiv (h, k)` in the combined axis.
    3. Run `sdpa` on each slice.
    4. Concatenate the head outputs back along the feature axis.
    5. Output projection Wo · concat + bo (per-token dense).

    The bundled VJP theorem below packages the correctness of this
    whole thing — composing dense Jacobians, the per-head SDPA
    jacobians (`sdpaBackQ_correct`, `sdpaBackK_correct`, `sdpaBackV_correct`), and
    the reshape/unreshape `pdiv_reindex` facts, with the per-head
    independence handled by the column-stacking framework
    (`pdivMat_colIndep` + `colSlabwiseHasVJPMat`). -/
noncomputable def mhsaLayer (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (X : Mat N (heads * d_head)) : Mat N (heads * d_head) :=
  let D := heads * d_head
  -- Q / K / V projections
  let Q : Mat N D := fun n j => (∑ k : Fin D, X n k * Wq k j) + bq j
  let K : Mat N D := fun n j => (∑ k : Fin D, X n k * Wk k j) + bk j
  let V : Mat N D := fun n j => (∑ k : Fin D, X n k * Wv k j) + bv j
  -- Per-head SDPA. `finProdFinEquiv (h, j) : Fin (heads * d_head)` picks out
  -- column `j` of head `h`. Extract, apply sdpa, note the result per head.
  let perHead : Fin heads → Mat N d_head := fun h =>
    let Qh : Mat N d_head := fun n j => Q n (finProdFinEquiv (h, j))
    let Kh : Mat N d_head := fun n j => K n (finProdFinEquiv (h, j))
    let Vh : Mat N d_head := fun n j => V n (finProdFinEquiv (h, j))
    sdpa N d_head Qh Kh Vh
  -- Concatenate heads: output[n, fPF(h, j)] = perHead h n j.
  let concat : Mat N D := fun n hj =>
    let hj' := finProdFinEquiv.symm hj
    perHead hj'.1 n hj'.2
  -- Output projection
  fun n j => (∑ k : Fin D, concat n k * Wo k j) + bo j

/-! ## Column-stacked SDPA — the bridge from `HasVJPMat3` to multi-head.

    Two facts are needed for multi-head attention:
    (1) joint differentiability of `(Q, K, V) ↦ sdpa Q K V`, which doesn't
    follow from the per-input `_flat_diff` lemmas; (2) the per-head
    "vmap" structure, which `colSlabwiseHasVJPMat` handles for
    *unary* per-slab functions but SDPA is naturally ternary.

    The fix: column-stack `(Q | K | V)` into a single `Mat n (3 * d_head)`
    "qkv slab", define `mhsaG : Mat n (3 * d_head) → Mat n d_head` as the
    unary view of SDPA on this slab, and lift via the existing framework.

    Both `mhsaG_flat_differentiable` and `mhsaGHasVJPMat` are then mechanical
    composition of existing pieces: the joint `_flat_diff` factors through
    `rowSoftmax_flat_differentiable` after stage-by-stage chaining, and the VJP comes
    from `sdpaHasVJPMat3` plus a "column-third projection" argument that
    matches the `(c : Fin 3)` index of the slab to the Q/K/V partial. -/

/-- Column-stacked SDPA: takes a slab `Mat n (3 * d_head)` whose columns
    encode `(c : Fin 3, j : Fin d_head)` via `finProdFinEquiv`, with `c = 0`
    being the Q-third, `c = 1` the K-third, `c = 2` the V-third. Returns
    `sdpa` applied to those three thirds. -/
noncomputable def mhsaG (n d : Nat) (slab : Mat n (3 * d)) : Mat n d :=
  sdpa n d
    (fun r j => slab r (finProdFinEquiv ((0 : Fin 3), j)))
    (fun r j => slab r (finProdFinEquiv ((1 : Fin 3), j)))
    (fun r j => slab r (finProdFinEquiv ((2 : Fin 3), j)))

/-- Pre-softmax matrix in `mhsaG`: `scale · Q · K^T` as a function of slab.
    Each entry is a polynomial in the slab's coords (linear projections
    times each other), so `fun_prop` discharges flat-diff after unfolding. -/
noncomputable def mhsaPreWeights (n d : Nat) (slab : Mat n (3 * d)) : Mat n n :=
  fun r c => sdpaScale d *
    Mat.mul
      (fun r' j => slab r' (finProdFinEquiv ((0 : Fin 3), j)))
      (Mat.transpose (fun r' j => slab r' (finProdFinEquiv ((1 : Fin 3), j))))
      r c

theorem mhsaPreWeights_flat_differentiable (n d : Nat) :
    Differentiable ℝ (fun v : Vec (n * (3 * d)) =>
      Mat.flatten ((mhsaPreWeights n d) (Mat.unflatten v))) := by
  unfold mhsaPreWeights Mat.flatten Mat.unflatten Mat.mul Mat.transpose
  fun_prop

/-- Post-softmax weights in `mhsaG`: `rowSoftmax(scale · Q · K^T)`. -/
noncomputable def mhsaWeights (n d : Nat) (slab : Mat n (3 * d)) : Mat n n :=
  rowSoftmax (mhsaPreWeights n d slab)

theorem mhsaWeights_flat_differentiable (n d : Nat) :
    Differentiable ℝ (fun v : Vec (n * (3 * d)) =>
      Mat.flatten ((mhsaWeights n d) (Mat.unflatten v))) :=
  flat_differentiable_comp (mhsaPreWeights_flat_differentiable n d) (rowSoftmax_flat_differentiable n n)

/-- **Joint flat-diff of column-stacked SDPA.**

    Joint diff in `(Q, K, V)` doesn't follow from the existing per-input `_flat_diff` lemmas. Here
    we prove it by treating the qkv-slab as the variable, factoring SDPA
    as `Mat.mul ∘ rowSoftmax ∘ scaled-matmul`: the weights are differentiable
    (`mhsaWeights_flat_differentiable`), and the final matmul with the V-third of the
    slab is polynomial in the weights and the slab coordinates. -/
theorem mhsaG_flat_differentiable (n d : Nat) :
    Differentiable ℝ (fun v : Vec (n * (3 * d)) =>
      Mat.flatten ((mhsaG n d) (Mat.unflatten v))) := by
  -- `mhsaG` is the softmax weights times the V-third of the slab; the weights enter through
  -- their flat-diff lemma, the rest is polynomial.
  have key : ∀ F : Vec (n * (3 * d)) → Vec (n * n), Differentiable ℝ F →
      Differentiable ℝ (fun v : Vec (n * (3 * d)) => Mat.flatten (Mat.mul (Mat.unflatten (F v))
        (fun r j => Mat.unflatten v r (finProdFinEquiv ((2 : Fin 3), j))))) := by
    intro F hF; unfold Mat.flatten Mat.mul Mat.unflatten; fun_prop
  have h := key _ (mhsaWeights_flat_differentiable n d)
  simp only [Mat.unflatten_flatten] at h
  exact h

/-! ### Column-stacked SDPA VJP

    `HasVJPMat (mhsaG n d)`: the backward column-stacks
    `(sdpaBackQ, sdpaBackK, sdpaBackV)` according to the c-third
    of the slab column index. Correctness reduces to `sdpaHasVJPMat3`
    after observing that perturbing the c-th third of the slab only
    perturbs the c-th input of SDPA. -/

/-- Column projection `slab ↦ slab^[c]` for a fixed `c : Fin 3`.
    Linear, so its flat form is a `reindexCLM`. -/
noncomputable def mhsaProjC {n d : Nat} (c : Fin 3) (slab : Mat n (3 * d)) : Mat n d :=
  fun r j => slab r (finProdFinEquiv (c, j))

/-- "Lift to slab third c": embeds `Vec (n * d)` into `Vec (n * (3 * d))` by
    placing `u` in the c-th column third and zero elsewhere. Linear, hence
    a CLM. Constructed from per-coord CLMs
    via `ContinuousLinearMap.pi`: each output coord is either a projection
    (if the index is in the c-third) or zero. -/
noncomputable def mhsaLiftCCLM (n d : Nat) (c : Fin 3) :
    Vec (n * d) →L[ℝ] Vec (n * (3 * d)) :=
  ContinuousLinearMap.pi (fun idx : Fin (n * (3 * d)) =>
    if (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).1 = c
    then ContinuousLinearMap.proj
      (finProdFinEquiv ((finProdFinEquiv.symm idx).1,
                        (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).2))
    else 0)

theorem mhsaLiftCCLM_apply (n d : Nat) (c : Fin 3) (u : Vec (n * d))
    (idx : Fin (n * (3 * d))) :
    mhsaLiftCCLM n d c u idx =
      (if (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).1 = c
       then u (finProdFinEquiv ((finProdFinEquiv.symm idx).1,
                                (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).2))
       else 0) := by
  simp only [mhsaLiftCCLM, ContinuousLinearMap.pi_apply]
  split_ifs <;> rfl

/-- "Embed Q' into slab at the c-th third, keep other thirds at slab's values."
    Affine function: `mhsaLiftCCLM c · u + (slab with c-th third zeroed)`. -/
noncomputable def mhsaEmbedC (n d : Nat) (c : Fin 3) (slab : Mat n (3 * d))
    (u : Vec (n * d)) : Vec (n * (3 * d)) :=
  fun idx =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.2
    if q.1 = c then u (finProdFinEquiv (p.1, q.2)) else Mat.flatten slab idx

theorem mhsaEmbedC_eq (n d : Nat) (c : Fin 3) (slab : Mat n (3 * d))
    (u : Vec (n * d)) :
    mhsaEmbedC n d c slab u = mhsaLiftCCLM n d c u +
      (fun idx =>
        if (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).1 = c
        then 0 else Mat.flatten slab idx) := by
  funext idx
  obtain ⟨⟨r, q⟩, rfl⟩ := finProdFinEquiv.surjective idx
  obtain ⟨⟨c', j⟩, rfl⟩ := finProdFinEquiv.surjective q
  by_cases hc : c' = c <;>
    simp only [mhsaEmbedC, mhsaLiftCCLM_apply, Pi.add_apply, Equiv.symm_apply_apply, hc,
      ite_true, ite_false, add_zero, zero_add]

theorem mhsaEmbedC_hasFDerivAt (n d : Nat) (c : Fin 3) (slab : Mat n (3 * d))
    (u₀ : Vec (n * d)) :
    HasFDerivAt (mhsaEmbedC n d c slab) (mhsaLiftCCLM n d c) u₀ := by
  rw [funext (mhsaEmbedC_eq n d c slab)]
  exact (mhsaLiftCCLM n d c).hasFDerivAt.add_const _

/-- The composition `mhsaG ∘ mhsaEmbedC c slab` equals "SDPA with the c-th
    argument variable, the other two fixed at `slab`'s projections". This is
    the freezing identity. -/
theorem mhsaG_comp_embed (n d : Nat) (c : Fin 3) (slab : Mat n (3 * d))
    (u : Vec (n * d)) :
    Mat.flatten ((mhsaG n d) (Mat.unflatten (mhsaEmbedC n d c slab u))) =
      (if c = (0 : Fin 3) then
         Mat.flatten (sdpa n d (Mat.unflatten u)
                        (mhsaProjC (1 : Fin 3) slab) (mhsaProjC (2 : Fin 3) slab))
       else if c = (1 : Fin 3) then
         Mat.flatten (sdpa n d (mhsaProjC (0 : Fin 3) slab)
                        (Mat.unflatten u) (mhsaProjC (2 : Fin 3) slab))
       else
         Mat.flatten (sdpa n d (mhsaProjC (0 : Fin 3) slab)
                        (mhsaProjC (1 : Fin 3) slab) (Mat.unflatten u))) := by
  -- The `c'`-th third of the embedded slab is `u` if `c' = c`, else `slab`'s.
  have h_proj_match : ∀ (c' : Fin 3),
      mhsaProjC c' (Mat.unflatten (mhsaEmbedC n d c slab u) : Mat n (3 * d)) =
      (if c' = c then (Mat.unflatten u : Mat n d) else mhsaProjC c' slab) := by
    intro c'; funext r j
    by_cases hc' : c' = c <;> simp [hc', mhsaProjC, mhsaEmbedC, Mat.unflatten, Mat.flatten]
  have hg : ∀ M, mhsaG n d M = sdpa n d (mhsaProjC 0 M) (mhsaProjC 1 M) (mhsaProjC 2 M) :=
    fun _ => rfl
  rw [hg, h_proj_match, h_proj_match, h_proj_match]
  fin_cases c <;> simp

theorem mhsaEmbedC_at_proj (n d : Nat) (c : Fin 3) (slab : Mat n (3 * d)) :
    mhsaEmbedC n d c slab (Mat.flatten (mhsaProjC c slab)) = Mat.flatten slab := by
  funext idx
  obtain ⟨⟨r, q⟩, rfl⟩ := finProdFinEquiv.surjective idx
  obtain ⟨⟨c', j⟩, rfl⟩ := finProdFinEquiv.surjective q
  by_cases hc : c' = c <;> simp [mhsaEmbedC, mhsaProjC, Mat.flatten, hc]

/-- **Helper for `pdivMat_mhsaG_split` (per-c chain rule).**
    For each `c : Fin 3`, the chain rule gives:
    `fderiv flat_g flat_slab ∘L mhsaLiftCCLM = fderiv flat_freeze_c flat_proj_c_slab`.
    Used in `pdivMat_mhsaG_split` after the basis-vector lift identity. -/
theorem pdivMat_mhsaG_split_chain (n d : Nat) (slab : Mat n (3 * d)) (c : Fin 3)
    (freeze_fn : Mat n d → Mat n d)
    (h_g_freeze_eq : ∀ u : Vec (n * d),
      Mat.flatten ((mhsaG n d) (Mat.unflatten (mhsaEmbedC n d c slab u))) =
      Mat.flatten (freeze_fn (Mat.unflatten u))) :
    (fderiv ℝ (fun v : Vec (n * (3 * d)) => Mat.flatten ((mhsaG n d) (Mat.unflatten v)))
              (Mat.flatten slab)).comp (mhsaLiftCCLM n d c) =
    fderiv ℝ (fun u : Vec (n * d) => Mat.flatten (freeze_fn (Mat.unflatten u)))
              (Mat.flatten (mhsaProjC c slab)) := by
  -- Chain rule for `flat_g ∘ embed_c` at the point `embed_c (proj_c slab) = slab`.
  have h := ((mhsaG_flat_differentiable n d) (Mat.flatten slab)).hasFDerivAt
  rw [← mhsaEmbedC_at_proj n d c slab] at h ⊢
  rw [← (h.comp _ (mhsaEmbedC_hasFDerivAt n d c slab _)).fderiv]
  exact congrArg (fderiv ℝ · _) (funext h_g_freeze_eq)

/-- **`pdivMat` of `mhsaG` splits per-c into the corresponding `pdivMat` of
    SDPA against its c-th argument.** The freezing lemma: changes in the
    c-th column third of the slab only perturb the c-th input of SDPA.
    Proved via the chain rule `mhsaG ∘ mhsaEmbedC = freeze_c`. -/
theorem pdivMat_mhsaG_split (n d : Nat) (slab : Mat n (3 * d))
    (i : Fin n) (c : Fin 3) (j : Fin d) (k : Fin n) (l : Fin d) :
    pdivMat (mhsaG n d) slab i (finProdFinEquiv (c, j)) k l =
    (if c = (0 : Fin 3) then
       pdivMat (fun Q' : Mat n d => sdpa n d Q' (mhsaProjC (1 : Fin 3) slab)
                                      (mhsaProjC (2 : Fin 3) slab))
               (mhsaProjC (0 : Fin 3) slab) i j k l
     else if c = (1 : Fin 3) then
       pdivMat (fun K' : Mat n d => sdpa n d (mhsaProjC (0 : Fin 3) slab) K'
                                      (mhsaProjC (2 : Fin 3) slab))
               (mhsaProjC (1 : Fin 3) slab) i j k l
     else
       pdivMat (fun V' : Mat n d => sdpa n d (mhsaProjC (0 : Fin 3) slab)
                                      (mhsaProjC (1 : Fin 3) slab) V')
               (mhsaProjC (2 : Fin 3) slab) i j k l) := by
  -- Compute mhsaLiftCCLM (basisVec (fPF(i, j))) = basisVec (fPF(i, fPF(c, j))).
  have h_lift_basis : mhsaLiftCCLM n d c (basisVec (finProdFinEquiv (i, j))) =
      basisVec (finProdFinEquiv (i, finProdFinEquiv (c, j))) := by
    funext idx
    obtain ⟨⟨r, q⟩, rfl⟩ := finProdFinEquiv.surjective idx
    obtain ⟨⟨c', j'⟩, rfl⟩ := finProdFinEquiv.surjective q
    by_cases hc : c' = c <;> simp [mhsaLiftCCLM_apply, hc]
  -- So the LHS is `(fderiv flat_g ∘L lift_c)` at a basis vector: per `c`, the chain rule
  -- (`pdivMat_mhsaG_split_chain`) with the freezing identity `mhsaG_comp_embed`.
  unfold pdivMat pdiv
  rw [← h_lift_basis, ← ContinuousLinearMap.comp_apply]
  by_cases hc0 : c = 0
  · subst hc0
    rw [ite_eq_left rfl, pdivMat_mhsaG_split_chain n d slab 0
      (fun Q' => sdpa n d Q' (mhsaProjC 1 slab) (mhsaProjC 2 slab))
      (fun u => by simpa using mhsaG_comp_embed n d 0 slab u)]
  by_cases hc1 : c = 1
  · subst hc1
    rw [ite_eq_right (by decide), ite_eq_left rfl, pdivMat_mhsaG_split_chain n d slab 1
      (fun K' => sdpa n d (mhsaProjC 0 slab) K' (mhsaProjC 2 slab))
      (fun u => by simpa using mhsaG_comp_embed n d 1 slab u)]
  obtain rfl : c = 2 := by fin_cases c <;> simp_all
  rw [ite_eq_right (by decide), ite_eq_right (by decide), pdivMat_mhsaG_split_chain n d slab 2
    (fun V' => sdpa n d (mhsaProjC 0 slab) (mhsaProjC 1 slab) V')
    (fun u => by simpa using mhsaG_comp_embed n d 2 slab u)]

/-- **HasVJPMat for column-stacked SDPA.** Backward column-stacks the three
    `sdpaBackQ`/`sdpaBackK`/`sdpaBackV` outputs by their `c : Fin 3` slot. Correctness comes from
    `pdivMat_mhsaG_split` (case-splits on c into the corresponding
    one-input SDPA pdivMat) and `sdpaHasVJPMat3.correct_*`. -/
noncomputable def mhsaGHasVJPMat (n d : Nat) :
    HasVJPMat (mhsaG n d) where
  backward := fun slab dY r kj =>
    let p := finProdFinEquiv.symm kj
    if p.1 = (0 : Fin 3) then
      sdpaBackQ n d (mhsaProjC (0 : Fin 3) slab) (mhsaProjC (1 : Fin 3) slab)
                      (mhsaProjC (2 : Fin 3) slab) dY r p.2
    else if p.1 = (1 : Fin 3) then
      sdpaBackK n d (mhsaProjC (0 : Fin 3) slab) (mhsaProjC (1 : Fin 3) slab)
                      (mhsaProjC (2 : Fin 3) slab) dY r p.2
    else
      sdpaBackV n d (mhsaProjC (0 : Fin 3) slab) (mhsaProjC (1 : Fin 3) slab)
                      (mhsaProjC (2 : Fin 3) slab) dY r p.2
  correct := by
    intro slab dY i kj
    obtain ⟨⟨c, j⟩, rfl⟩ := finProdFinEquiv.surjective kj
    simp only [pdivMat_mhsaG_split, Equiv.symm_apply_apply]
    by_cases hc0 : c = 0
    · subst hc0; simp only [ite_true]; exact sdpaBackQ_correct n d _ _ _ dY i j
    by_cases hc1 : c = 1
    · subst hc1; simp only [hc0, ite_true, ite_false]; exact sdpaBackK_correct n d _ _ _ dY i j
    simp only [hc0, hc1, ite_false]; exact sdpaBackV_correct n d _ _ _ dY i j

/-- Flat-diff for `colSlabApply g`: each output coord is `(g (slab h ·)) [n, j_out]`,
    factoring through the linear slab projection and `g` (flat-diff) by `flat_differentiable_comp`. -/
theorem colSlabApply_flat_differentiable {n heads d_in d_out : Nat}
    (g : Mat n d_in → Mat n d_out)
    (hg_diff : Differentiable ℝ
                 (fun v : Vec (n * d_in) => Mat.flatten (g (Mat.unflatten v)))) :
    Differentiable ℝ (fun v : Vec (n * (heads * d_in)) =>
      Mat.flatten (colSlabApply g (Mat.unflatten v) : Mat n (heads * d_out))) := by
  rw [differentiable_pi]; intro idx
  obtain ⟨⟨r, q⟩, rfl⟩ := finProdFinEquiv.surjective idx
  obtain ⟨⟨h, j⟩, rfl⟩ := finProdFinEquiv.surjective q
  -- Coordinate `(r, (h, j))` is coordinate `(r, j)` of `g` on the `h`-th column slab.
  have hh := flat_differentiable_comp (G := g) (F := fun M : Mat n (heads * d_in) =>
    fun r' j' => M r' (finProdFinEquiv (h, j'))) (by unfold Mat.flatten Mat.unflatten; fun_prop)
    hg_diff
  simpa [Mat.flatten, colSlabApply] using differentiable_pi.mp hh (finProdFinEquiv (r, j))

-- ════════════════════════════════════════════════════════════════
-- § 3.5 Multi-head composition: replace the two axioms with theorems.
-- ════════════════════════════════════════════════════════════════

/-- Combined Q/K/V weight matrix: stack `Wq | Wk | Wv` with the per-head
    interleave layout. Output column `(h, c, j) ↦ (Wq | Wk | Wv)[k, fPF(h, j)]`
    based on `c : Fin 3`. Used to express the three Q/K/V projections as a
    single per-token dense, enabling clean composition with `colSlabApply mhsaG`. -/
noncomputable def mhsaQkvW (heads d_head : Nat)
    (Wq Wk Wv : Mat (heads * d_head) (heads * d_head)) :
    Mat (heads * d_head) (heads * (3 * d_head)) :=
  fun k idx =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.2
    if q.1 = (0 : Fin 3) then Wq k (finProdFinEquiv (p.1, q.2))
    else if q.1 = (1 : Fin 3) then Wk k (finProdFinEquiv (p.1, q.2))
    else Wv k (finProdFinEquiv (p.1, q.2))

noncomputable def mhsaQkvB (heads d_head : Nat)
    (bq bk bv : Vec (heads * d_head)) :
    Vec (heads * (3 * d_head)) :=
  fun idx =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.2
    if q.1 = (0 : Fin 3) then bq (finProdFinEquiv (p.1, q.2))
    else if q.1 = (1 : Fin 3) then bk (finProdFinEquiv (p.1, q.2))
    else bv (finProdFinEquiv (p.1, q.2))

@[simp] theorem mhsaQkvW_eq0 (heads d_head : Nat)
    (Wq Wk Wv : Mat (heads * d_head) (heads * d_head))
    (k : Fin (heads * d_head)) (h : Fin heads) (j : Fin d_head) :
    mhsaQkvW heads d_head Wq Wk Wv k
      (finProdFinEquiv (h, finProdFinEquiv ((0 : Fin 3), j))) = Wq k (finProdFinEquiv (h, j)) := by
  unfold mhsaQkvW
  simp [Equiv.symm_apply_apply]

@[simp] theorem mhsaQkvW_eq1 (heads d_head : Nat)
    (Wq Wk Wv : Mat (heads * d_head) (heads * d_head))
    (k : Fin (heads * d_head)) (h : Fin heads) (j : Fin d_head) :
    mhsaQkvW heads d_head Wq Wk Wv k
      (finProdFinEquiv (h, finProdFinEquiv ((1 : Fin 3), j))) = Wk k (finProdFinEquiv (h, j)) := by
  unfold mhsaQkvW
  simp [Equiv.symm_apply_apply, show (1 : Fin 3) ≠ (0 : Fin 3) from by decide]

@[simp] theorem mhsaQkvW_eq2 (heads d_head : Nat)
    (Wq Wk Wv : Mat (heads * d_head) (heads * d_head))
    (k : Fin (heads * d_head)) (h : Fin heads) (j : Fin d_head) :
    mhsaQkvW heads d_head Wq Wk Wv k
      (finProdFinEquiv (h, finProdFinEquiv ((2 : Fin 3), j))) = Wv k (finProdFinEquiv (h, j)) := by
  unfold mhsaQkvW
  simp [Equiv.symm_apply_apply,
        show (2 : Fin 3) ≠ (0 : Fin 3) from by decide,
        show (2 : Fin 3) ≠ (1 : Fin 3) from by decide]

@[simp] theorem mhsaQkvB_eq0 (heads d_head : Nat)
    (bq bk bv : Vec (heads * d_head))
    (h : Fin heads) (j : Fin d_head) :
    mhsaQkvB heads d_head bq bk bv
      (finProdFinEquiv (h, finProdFinEquiv ((0 : Fin 3), j))) = bq (finProdFinEquiv (h, j)) := by
  unfold mhsaQkvB
  simp [Equiv.symm_apply_apply]

@[simp] theorem mhsaQkvB_eq1 (heads d_head : Nat)
    (bq bk bv : Vec (heads * d_head))
    (h : Fin heads) (j : Fin d_head) :
    mhsaQkvB heads d_head bq bk bv
      (finProdFinEquiv (h, finProdFinEquiv ((1 : Fin 3), j))) = bk (finProdFinEquiv (h, j)) := by
  unfold mhsaQkvB
  simp [Equiv.symm_apply_apply, show (1 : Fin 3) ≠ (0 : Fin 3) from by decide]

@[simp] theorem mhsaQkvB_eq2 (heads d_head : Nat)
    (bq bk bv : Vec (heads * d_head))
    (h : Fin heads) (j : Fin d_head) :
    mhsaQkvB heads d_head bq bk bv
      (finProdFinEquiv (h, finProdFinEquiv ((2 : Fin 3), j))) = bv (finProdFinEquiv (h, j)) := by
  unfold mhsaQkvB
  simp [Equiv.symm_apply_apply,
        show (2 : Fin 3) ≠ (0 : Fin 3) from by decide,
        show (2 : Fin 3) ≠ (1 : Fin 3) from by decide]

/-- The mhsaLayer factorization: it equals
    `output_dense ∘ colSlabApply mhsaG ∘ qkv_stack_dense`.

    All three pieces have HasVJPMat and flat-diff:
    - `qkv_stack_dense` uses `mhsaQkvW`, `mhsaQkvB` as a single per-token dense.
    - `colSlabApply mhsaG` lifts `mhsaGHasVJPMat` per-head.
    - `output_dense` is the standard per-token dense for Wo, bo. -/
theorem mhsaLayer_eq_compose (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (X : Mat N (heads * d_head)) :
    mhsaLayer N heads d_head Wq Wk Wv Wo bq bk bv bo X =
    (fun M : Mat N (heads * d_head) => fun n => dense Wo bo (M n))
      (colSlabApply (mhsaG N d_head) (heads := heads)
        ((fun X' : Mat N (heads * d_head) => fun n =>
           dense (mhsaQkvW heads d_head Wq Wk Wv) (mhsaQkvB heads d_head bq bk bv) (X' n))
         X)) := by
  funext n j
  simp only [mhsaLayer, dense]
  congr 1
  refine Finset.sum_congr rfl fun k _ => congrArg (· * Wo k j) ?_
  obtain ⟨⟨h, jo⟩, rfl⟩ := finProdFinEquiv.surjective k
  simp [colSlabApply, mhsaG, dense]

/-- **The composed MHSA VJP** — `Wo-dense ∘ colSlabApply mhsaG ∘ qkv-dense`,
    stated on the explicit composition (no `mhsaLayer_eq_compose` transport).
    This is the substantive witness; `mhsaHasVJPMat` below re-types it to
    `mhsaLayer` with the cast confined to the `correct` field, so
    `(mhsaHasVJPMat …).backward` reduces in one projection. -/
-- Why the split: defining `mhsaHasVJPMat := by rw [show mhsaLayer = …]; exact vjpMatComp …`
-- makes the constant's value an `Eq.mpr` cast around the structure. Any kernel defeq that
-- whnf's `(mhsaHasVJPMat …).backward` then replays the whole `mhsaLayer` rewrite, ~200 s of
-- kernel type-checking per downstream declaration that forces it (no cross-declaration whnf
-- cache); that was most of `ViTBackB0`'s and `ViTMhsaBackCertifiedTie`'s build time. With
-- `backward` a direct field, the projection whnfs in one hop.
noncomputable def mhsaComposedHasVJPMat (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    HasVJPMat
      ((fun M : Mat N (heads * d_head) => fun n => dense Wo bo (M n)) ∘
       (colSlabApply (mhsaG N d_head) (heads := heads)) ∘
       (fun X' : Mat N (heads * d_head) => fun n =>
          dense (mhsaQkvW heads d_head Wq Wk Wv)
                (mhsaQkvB heads d_head bq bk bv) (X' n))) := by
  -- VJPs and diffs for each piece (inline `densePerTokenHasVJPMat` since
  -- it's defined later in this file; use `rowwiseHasVJPMat` directly).
  have h_qkv_vjp : HasVJPMat (fun X' : Mat N (heads * d_head) => fun n =>
      dense (mhsaQkvW heads d_head Wq Wk Wv) (mhsaQkvB heads d_head bq bk bv) (X' n)) :=
    rowwiseHasVJPMat (denseHasVJP (mhsaQkvW heads d_head Wq Wk Wv)
                                        (mhsaQkvB heads d_head bq bk bv))
                        (dense_differentiable (mhsaQkvW heads d_head Wq Wk Wv)
                                    (mhsaQkvB heads d_head bq bk bv))
  have h_qkv_diff := dense_per_token_flat_differentiable
                      (N := N) (mhsaQkvW heads d_head Wq Wk Wv) (mhsaQkvB heads d_head bq bk bv)
  have h_g_diff := mhsaG_flat_differentiable N d_head
  have h_body_vjp : HasVJPMat (colSlabApply (mhsaG N d_head) (heads := heads)) :=
    colSlabwiseHasVJPMat (mhsaGHasVJPMat N d_head) h_g_diff
  have h_body_diff : Differentiable ℝ (fun v : Vec (N * (heads * (3 * d_head))) =>
      Mat.flatten ((colSlabApply (mhsaG N d_head) (heads := heads)) (Mat.unflatten v)
                   : Mat N (heads * d_head))) :=
    colSlabApply_flat_differentiable (mhsaG N d_head) h_g_diff
  have h_output_vjp : HasVJPMat (fun M : Mat N (heads * d_head) => fun n => dense Wo bo (M n)) :=
    rowwiseHasVJPMat (denseHasVJP Wo bo) (dense_differentiable Wo bo)
  have h_output_diff := dense_per_token_flat_differentiable (N := N) Wo bo
  -- Compose body ∘ qkv first.
  have h_body_qkv_vjp : HasVJPMat
      ((colSlabApply (mhsaG N d_head) (heads := heads)) ∘
       (fun X' : Mat N (heads * d_head) => fun n =>
          dense (mhsaQkvW heads d_head Wq Wk Wv) (mhsaQkvB heads d_head bq bk bv) (X' n))) :=
    vjpMatComp _ _ h_qkv_diff h_body_diff h_qkv_vjp h_body_vjp
  have h_body_qkv_diff : Differentiable ℝ
      (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten ((colSlabApply (mhsaG N d_head) (heads := heads) ∘
          (fun X' : Mat N (heads * d_head) => fun n =>
            dense (mhsaQkvW heads d_head Wq Wk Wv) (mhsaQkvB heads d_head bq bk bv) (X' n)))
          (Mat.unflatten v) : Mat N (heads * d_head))) :=
    flat_differentiable_comp h_qkv_diff h_body_diff
  -- Final compose with output.
  exact vjpMatComp _ _ h_body_qkv_diff h_output_diff h_body_qkv_vjp h_output_vjp

/-- **Multi-head SDPA VJP**, composed from `mhsaGHasVJPMat`, `colSlabwiseHasVJPMat`, and
    the per-token dense framework. `backward` is the composed witness's
    field DIRECTLY (kernel-cheap projection); the `mhsaLayer_eq_compose`
    transport lives only in `correct` — a `Prop` the kernel never reduces.
    See `mhsaComposedHasVJPMat`'s docstring for why. -/
noncomputable def mhsaHasVJPMat (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    HasVJPMat (mhsaLayer N heads d_head Wq Wk Wv Wo bq bk bv bo) where
  backward := (mhsaComposedHasVJPMat N heads d_head Wq Wk Wv Wo bq bk bv bo).backward
  correct := by
    have hfun : mhsaLayer N heads d_head Wq Wk Wv Wo bq bk bv bo =
        (fun M : Mat N (heads * d_head) => fun n => dense Wo bo (M n)) ∘
        (colSlabApply (mhsaG N d_head) (heads := heads)) ∘
        (fun X' : Mat N (heads * d_head) => fun n =>
           dense (mhsaQkvW heads d_head Wq Wk Wv)
                 (mhsaQkvB heads d_head bq bk bv) (X' n)) := by
      funext X
      exact mhsaLayer_eq_compose N heads d_head Wq Wk Wv Wo bq bk bv bo X
    intro A dY i j
    rw [hfun]
    exact (mhsaComposedHasVJPMat N heads d_head Wq Wk Wv Wo bq bk bv bo).correct A dY i j

/-- **Differentiability of the flattened multi-head SDPA layer** —
    composition of three `_flat_diff` lemmas. -/
theorem mhsaLayer_flat_differentiable (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (mhsaLayer N heads d_head Wq Wk Wv Wo bq bk bv bo
                     (Mat.unflatten v))) := by
  simpa [mhsaLayer_eq_compose, Function.comp_def, Mat.unflatten_flatten] using
    (dense_per_token_flat_differentiable (N := N) Wo bo).comp
      ((colSlabApply_flat_differentiable (mhsaG N d_head) (mhsaG_flat_differentiable N d_head)).comp
        (dense_per_token_flat_differentiable (N := N) (mhsaQkvW heads d_head Wq Wk Wv)
          (mhsaQkvB heads d_head bq bk bv)))

-- ════════════════════════════════════════════════════════════════
-- § 4. Transformer Block (Phase 8 — composition, no hand-waving)
-- ════════════════════════════════════════════════════════════════

/-! ## Per-token liftings (theorems)

Every per-token operation in a transformer (LN, dense, GELU) lifts from
`HasVJP` on `Vec D` to `HasVJPMat` on `Mat N D` via the single helper
`rowwiseHasVJPMat` (Tensor.lean). -/

/-- Per-token layer norm across a sequence. Applies `layerNormForward`
    to each row of the `(N, D)` input; the backward is block-diagonal. -/
noncomputable def layerNormPerTokenHasVJPMat (N D : Nat) (ε γ β : ℝ)
    (hε : 0 < ε) :
    HasVJPMat (fun X : Mat N D => fun n => layerNormForward D ε γ β (X n)) :=
  rowwiseHasVJPMat (layerNormHasVJP D ε γ β hε) (layerNorm_differentiable D ε γ β hε)

/-- Per-token dense projection across a sequence.
    `Q = X · W + b`, row-by-row dense with shared weights. -/
noncomputable def densePerTokenHasVJPMat (N inD outD : Nat)
    (W : Mat inD outD) (b : Vec outD) :
    HasVJPMat (fun X : Mat N inD => fun n => dense W b (X n)) :=
  rowwiseHasVJPMat (denseHasVJP W b) (dense_differentiable W b)

/-- Per-token GELU across a sequence. Elementwise activation,
    so diagonal Jacobian both across rows and within a row. -/
noncomputable def geluPerTokenHasVJPMat (N D : Nat) :
    HasVJPMat (fun X : Mat N D => fun n => gelu D (X n)) :=
  rowwiseHasVJPMat (geluHasVJP D) (gelu_differentiable D)

/-! ## A transformer encoder block

From `MlirCodegen.emitTransformerBlockForward`:

    block(x) = h1 + MLP(LN2(h1))       where h1 = x + MHSA(LN1(x))

Expanding:

    h1 = x + MHSA(LN1(x))       -- attention sub-layer with residual
    out = h1 + MLP(LN2(h1))     -- MLP sub-layer with residual

where `MLP(z) = dense(Wfc2, bfc2, gelu(dense(Wfc1, bfc1, z)))`.

Every piece is now a `HasVJPMat` on `Mat N D`:
- `LN1`, `LN2` — `layerNormPerTokenHasVJPMat` (theorem via `rowwiseHasVJPMat`)
- `MHSA`       — `mhsaHasVJPMat` (bundled `HasVJPMat` def)
- `MLP`        — two `densePerTokenHasVJPMat` + one `geluPerTokenHasVJPMat`, glued with `vjpMatComp`
- `+` residuals — `biPathMatHasVJP` (theorem, Tensor.lean) with identity

The transformer block theorem below glues these with `vjpMatComp` and
`biPathMatHasVJP`. -/

/-- MLP sublayer of a transformer block: `dense ∘ GELU ∘ dense` applied per-token.

    Concretely: `MLP(z) = Wfc2 · gelu(Wfc1 · z + bfc1) + bfc2`, applied row-wise. -/
noncomputable def transformerMlp (N D mlpDim : Nat)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim D) (bfc2 : Vec D) :
    Mat N D → Mat N D :=
  (fun Y : Mat N mlpDim => fun n => dense Wfc2 bfc2 (Y n)) ∘
  (fun Y : Mat N mlpDim => fun n => gelu mlpDim (Y n)) ∘
  (fun X : Mat N D      => fun n => dense Wfc1 bfc1 (X n))

/-- Differentiability of the flattened `transformerMlp` — `dense ∘ gelu ∘ dense` per token,
    all smooth (`differentiable_tanh` is tagged for `fun_prop`). -/
lemma transformerMlp_flat_differentiable (N D mlpDim : Nat)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim D) (bfc2 : Vec D) :
    Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten (transformerMlp N D mlpDim Wfc1 bfc1 Wfc2 bfc2
                     (Mat.unflatten v))) := by
  unfold transformerMlp Mat.unflatten Mat.flatten dense gelu geluScalar; fun_prop

/-- `HasVJPMat` for the MLP sublayer — chain of two `vjpMatComp`
    steps over per-token liftings (`dense ∘ gelu ∘ dense`). Every Diff
    hypothesis is discharged by the per-token-flat helpers above. -/
noncomputable def transformerMlpHasVJPMat (N D mlpDim : Nat)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim D) (bfc2 : Vec D) :
    HasVJPMat (transformerMlp N D mlpDim Wfc1 bfc1 Wfc2 bfc2) :=
  -- Inner composition: gelu ∘ dense₁
  let innerHasVJP :=
    vjpMatComp _ (fun Y : Mat N mlpDim => fun n => gelu mlpDim (Y n))
      (dense_per_token_flat_differentiable Wfc1 bfc1)
      (gelu_per_token_flat_differentiable N mlpDim)
      (densePerTokenHasVJPMat N D mlpDim Wfc1 bfc1)
      (geluPerTokenHasVJPMat N mlpDim)
  -- Diff of the inner composition (gelu ∘ dense₁), via Mat.flatten/unflatten
  -- round-trip + Differentiable.comp.
  have inner_diff : Differentiable ℝ
      (fun v : Vec (N * D) =>
        Mat.flatten (((fun Y : Mat N mlpDim => fun n => gelu mlpDim (Y n)) ∘
                      (fun X : Mat N D      => fun n => dense Wfc1 bfc1 (X n)))
                     (Mat.unflatten v))) :=
    flat_differentiable_comp (dense_per_token_flat_differentiable Wfc1 bfc1) (gelu_per_token_flat_differentiable N mlpDim)
  -- Outer composition: dense₂ ∘ (gelu ∘ dense₁)
  vjpMatComp _ (fun Y : Mat N mlpDim => fun n => dense Wfc2 bfc2 (Y n))
    inner_diff
    (dense_per_token_flat_differentiable Wfc2 bfc2)
    innerHasVJP
    (densePerTokenHasVJPMat N mlpDim D Wfc2 bfc2)

/-- VJP of a pre-norm residual sublayer `X ↦ X + F (L X)` (norm `L`, then body `F`):
    `biPathMatHasVJP` of the identity skip and the `vjpMatComp` chain `L`-back ∘
    `F`-back. Every transformer sublayer (scalar- and vector-LN, attention and MLP) is
    an instance. -/
noncomputable def preLNResHasVJPMat {N D : Nat} (L F : Mat N D → Mat N D)
    (hL : Differentiable ℝ (fun v : Vec (N * D) => Mat.flatten (L (Mat.unflatten v))))
    (hF : Differentiable ℝ (fun v : Vec (N * D) => Mat.flatten (F (Mat.unflatten v))))
    (vL : HasVJPMat L) (vF : HasVJPMat F) :
    HasVJPMat (biPathMat (fun X => X) (F ∘ L)) :=
  biPathMatHasVJP _ _ (identity_mat_flat_differentiable N D)
    (flat_differentiable_comp hL hF)
    (identityMatHasVJP N D) (vjpMatComp L F hL hF vL vF)

/-- Attention sublayer: `X ↦ X + MHSA(LN1(X))`. Top-level composition;
    the `biPathMat` skip-adds identity to the MHSA∘LN1 branch. -/
noncomputable def transformerAttnSublayer (N heads d_head : Nat) (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Mat N (heads * d_head) → Mat N (heads * d_head) :=
  biPathMat
    (fun X => X)
    ((mhsaLayer N heads d_head Wq Wk Wv Wo bq bk bv bo) ∘
     (fun X : Mat N (heads * d_head) => fun n =>
        layerNormForward (heads * d_head) ε γ1 β1 (X n)))

/-- MLP sublayer: `h ↦ h + MLP(LN2(h))`. Same biPathMat structure. -/
noncomputable def transformerMlpSublayer (N heads d_head mlpDim : Nat) (ε γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Mat N (heads * d_head) → Mat N (heads * d_head) :=
  biPathMat
    (fun X => X)
    ((transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2) ∘
     (fun X : Mat N (heads * d_head) => fun n =>
        layerNormForward (heads * d_head) ε γ2 β2 (X n)))

/-- **Transformer encoder block forward**: MLP-sublayer ∘ attention-sublayer.
    Signature matches the codegen: `Mat N (heads·d_head) → Mat N (heads·d_head)`. -/
noncomputable def transformerBlock (N heads d_head mlpDim : Nat) (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Mat N (heads * d_head) → Mat N (heads * d_head) :=
  (transformerMlpSublayer N heads d_head mlpDim ε γ2 β2 Wfc1 bfc1 Wfc2 bfc2) ∘
  (transformerAttnSublayer N heads d_head ε γ1 β1 Wq Wk Wv Wo bq bk bv bo)

/-- Differentiability of the flattened attention sublayer's non-trivial arm
    (`mhsa ∘ LN1`). Used by both the sublayer VJP proof and any downstream
    composition that needs Diff for the sublayer's arm. -/
lemma transformerAttnSublayer_inner_flat_differentiable
    (N heads d_head : Nat) (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten
        (((mhsaLayer N heads d_head Wq Wk Wv Wo bq bk bv bo) ∘
          (fun X : Mat N (heads * d_head) => fun n =>
            layerNormForward (heads * d_head) ε γ1 β1 (X n)))
         (Mat.unflatten v))) :=
  flat_differentiable_comp (layerNorm_per_token_flat_differentiable N (heads * d_head) ε γ1 β1 hε)
    (mhsaLayer_flat_differentiable N heads d_head Wq Wk Wv Wo bq bk bv bo)

/-- Differentiability of the flattened attention sublayer.
    `biPathMat (id) (mhsa ∘ LN1)` flattens to a sum, both arms Differentiable. -/
lemma transformerAttnSublayer_flat_differentiable
    (N heads d_head : Nat) (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (transformerAttnSublayer N heads d_head ε γ1 β1
                     Wq Wk Wv Wo bq bk bv bo (Mat.unflatten v))) := by
  exact (identity_mat_flat_differentiable N (heads * d_head)).add
    (transformerAttnSublayer_inner_flat_differentiable N heads d_head ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo)

/-- Attention sublayer VJP: `preLNResHasVJPMat` at `L = LN1`, `F = mhsa`. -/
noncomputable def transformerAttnSublayerHasVJPMat (N heads d_head : Nat)
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    HasVJPMat (transformerAttnSublayer N heads d_head ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo) :=
  preLNResHasVJPMat _ _ (layerNorm_per_token_flat_differentiable N (heads * d_head) ε γ1 β1 hε)
    (mhsaLayer_flat_differentiable N heads d_head Wq Wk Wv Wo bq bk bv bo)
    (layerNormPerTokenHasVJPMat N (heads * d_head) ε γ1 β1 hε)
    (mhsaHasVJPMat N heads d_head Wq Wk Wv Wo bq bk bv bo)

/-- Differentiability of the MLP sublayer's non-trivial arm
    (`transformerMlp ∘ LN2`). Composition of `transformerMlp_flat_differentiable`
    and `layerNorm_per_token_flat_differentiable`. -/
lemma transformerMlpSublayer_inner_flat_differentiable
    (N heads d_head mlpDim : Nat) (ε γ2 β2 : ℝ) (hε : 0 < ε)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten
        (((transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2) ∘
          (fun X : Mat N (heads * d_head) => fun n =>
            layerNormForward (heads * d_head) ε γ2 β2 (X n)))
         (Mat.unflatten v))) :=
  flat_differentiable_comp (layerNorm_per_token_flat_differentiable N (heads * d_head) ε γ2 β2 hε)
    (transformerMlp_flat_differentiable N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2)

/-- Differentiability of the flattened MLP sublayer.
    `biPathMat (id) (transformerMlp ∘ LN2)` flattens to a sum, both arms Differentiable. -/
lemma transformerMlpSublayer_flat_differentiable
    (N heads d_head mlpDim : Nat) (ε γ2 β2 : ℝ) (hε : 0 < ε)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (transformerMlpSublayer N heads d_head mlpDim ε γ2 β2
                     Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten v))) := by
  exact (identity_mat_flat_differentiable N (heads * d_head)).add
    (transformerMlpSublayer_inner_flat_differentiable N heads d_head mlpDim ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2)

/-- MLP sublayer VJP: `preLNResHasVJPMat` at `L = LN2`, `F = transformerMlp`. -/
noncomputable def transformerMlpSublayerHasVJPMat (N heads d_head mlpDim : Nat)
    (ε γ2 β2 : ℝ) (hε : 0 < ε)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    HasVJPMat (transformerMlpSublayer N heads d_head mlpDim ε γ2 β2
                 Wfc1 bfc1 Wfc2 bfc2) :=
  preLNResHasVJPMat _ _ (layerNorm_per_token_flat_differentiable N (heads * d_head) ε γ2 β2 hε)
    (transformerMlp_flat_differentiable N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2)
    (layerNormPerTokenHasVJPMat N (heads * d_head) ε γ2 β2 hε)
    (transformerMlpHasVJPMat N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2)

/-- Differentiability of the flattened transformer block.
    `MlpSublayer ∘ AttnSublayer`; both sublayers' flat Diff are theorems above. -/
lemma transformerBlock_flat_differentiable (N heads d_head mlpDim : Nat)
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (transformerBlock N heads d_head mlpDim ε γ1 β1
                     Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2
                   (Mat.unflatten v))) :=
  flat_differentiable_comp
    (transformerAttnSublayer_flat_differentiable N heads d_head ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo)
    (transformerMlpSublayer_flat_differentiable N heads d_head mlpDim ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2)

/-- **Transformer block VJP** — composition of attn + mlp sublayers:
    a single `vjpMatComp` of the two sublayer witnesses with their Diff
    helpers. Both LayerNorms are scalar-affine (`γ1 β1 γ2 β2 : ℝ`). -/
noncomputable def transformerBlockHasVJPMat (N heads d_head mlpDim : Nat)
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    HasVJPMat (transformerBlock N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo
                 γ2 β2 Wfc1 bfc1 Wfc2 bfc2) :=
  vjpMatComp _ (transformerMlpSublayer N heads d_head mlpDim ε γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
    (transformerAttnSublayer_flat_differentiable N heads d_head ε γ1 β1 hε
       Wq Wk Wv Wo bq bk bv bo)
    (transformerMlpSublayer_flat_differentiable N heads d_head mlpDim ε γ2 β2 hε
       Wfc1 bfc1 Wfc2 bfc2)
    (transformerAttnSublayerHasVJPMat N heads d_head ε γ1 β1 hε
       Wq Wk Wv Wo bq bk bv bo)
    (transformerMlpSublayerHasVJPMat N heads d_head mlpDim ε γ2 β2 hε
       Wfc1 bfc1 Wfc2 bfc2)

-- ════════════════════════════════════════════════════════════════
-- § 5. The ViT finale — k-block transformer tower
-- ════════════════════════════════════════════════════════════════

/-! ## Stacking transformer blocks

ViT-Tiny has 12 transformer blocks; ViT-Base has 12, ViT-Large has 24.
The stack is just k-fold composition of individual blocks. By
`vjpMatComp` and induction on k, if each block has a `HasVJPMat`
then so does the stack — for any depth.

The formal tower `transformerTower` uses a single shared parameter
tuple across all `k` blocks (`Nat.rec` over one tuple), with
scalar-affine LayerNorms; in practice every block has its own weights
and a per-feature LN. Per-block parameters are not formalized in this
file. -/

/-- k-fold iterated transformer block, sharing parameters across all
    k layers. Defined by `Nat.rec` so the `HasVJPMat` proof is a
    straightforward induction on k. -/
noncomputable def transformerTower (k N heads d_head mlpDim : Nat)
    (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Mat N (heads * d_head) → Mat N (heads * d_head) :=
  Nat.rec (motive := fun _ => Mat N (heads * d_head) → Mat N (heads * d_head))
    (fun X => X)
    (fun _ acc =>
      (transformerBlock N heads d_head mlpDim ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
         γ2 β2 Wfc1 bfc1 Wfc2 bfc2) ∘ acc)
    k

/-- Differentiability of the flattened k-fold transformer tower.
    Induction on `k`: zero case is identity, successor case is
    `block ∘ tower(k)` composed via `Differentiable.comp`. -/
lemma transformerTower_flat_differentiable (k N heads d_head mlpDim : Nat)
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (transformerTower k N heads d_head mlpDim ε γ1 β1
                     Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2
                   (Mat.unflatten v))) := by
  induction k with
  | zero => exact identity_mat_flat_differentiable N (heads * d_head)
  | succ k' ih =>
    -- `transformerTower (k'+1) = block ∘ transformerTower k'` by `Nat.rec`.
    exact flat_differentiable_comp ih (transformerBlock_flat_differentiable N heads d_head mlpDim ε γ1 β1 hε
      Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)

/-- **Transformer tower VJP** — k-fold composition of one shared block
    (`transformerTower`): induction on `k` via `vjpMatComp` and
    `transformerBlockHasVJPMat`. -/
noncomputable def transformerTowerHasVJPMat (k N heads d_head mlpDim : Nat)
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    HasVJPMat (transformerTower k N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo
                 γ2 β2 Wfc1 bfc1 Wfc2 bfc2) := by
  induction k with
  | zero =>
    show HasVJPMat (fun X : Mat N (heads * d_head) => X)
    exact identityMatHasVJP N (heads * d_head)
  | succ k' ih =>
    show HasVJPMat ((transformerBlock N heads d_head mlpDim ε γ1 β1
                       Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2) ∘
                    (transformerTower k' N heads d_head mlpDim ε γ1 β1
                       Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2))
    exact vjpMatComp _ _
      (transformerTower_flat_differentiable k' N heads d_head mlpDim ε γ1 β1 hε
         Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
      (transformerBlock_flat_differentiable N heads d_head mlpDim ε γ1 β1 hε
         Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
      ih
      (transformerBlockHasVJPMat N heads d_head mlpDim ε γ1 β1 hε
         Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)

/-! ## ViT body: tower + final LN

`vitBody` is the ViT backbone operating on a single `(N, D)` sequence,
*after* the patch embedding produced a `Mat N D` input and *before* the
classifier head slices the CLS token and runs dense+softmax CE.

    patch_embed(X : Tensor3 ic h w) : Mat N D   ← outside Mat-land
    vitBody(M : Mat N D) : Mat N D             ← the backbone (this file)
    classifier(M) = dense(W_cls, b_cls, M[0])   ← Mat → Vec, then softmax CE loss

The backbone is `finalLN ∘ transformerTower`. Both sides are `Mat N D`,
so `vjpMatComp` glues the two VJPs.

The patch-embedding and classifier-head steps exit `Mat`-land (they
change type to/from `Tensor3` and `Vec` respectively), so they don't fit
in the uniform `HasVJPMat` frame. They are bridged below (§ Bridging
ranks): the body is flattened with `HasVJPMat.toHasVJP` and composed with
`patchEmbedFlatHasVJP` and `classifierFlatHasVJP` in `vitFullHasVJP`. -/

/-- **ViT body** — transformer tower followed by final per-token LayerNorm.

    Composition is `finalLN ∘ transformerTower`; matches the codegen's
    `emitForwardBody` ordering for a `.transformerEncoder` followed by
    the implicit final LN block. -/
noncomputable def vitBody (k N heads d_head mlpDim : Nat) (ε : ℝ)
    (γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (γF βF : ℝ)  -- final LN params
    : Mat N (heads * d_head) → Mat N (heads * d_head) :=
  (fun X : Mat N (heads * d_head) => fun n =>
      layerNormForward (heads * d_head) ε γF βF (X n)) ∘
  (transformerTower k N heads d_head mlpDim ε γ1 β1
     Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)

/-- Differentiability of the flattened ViT body.
    `finalLN ∘ transformerTower` — both have flat Diff theorems above. -/
lemma vitBody_flat_differentiable (k N heads d_head mlpDim : Nat) (ε : ℝ) (hε : 0 < ε)
    (γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (γF βF : ℝ) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (vitBody k N heads d_head mlpDim ε γ1 β1
                     Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF
                   (Mat.unflatten v))) :=
  flat_differentiable_comp (transformerTower_flat_differentiable k N heads d_head mlpDim ε γ1 β1 hε
      Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
    (layerNorm_per_token_flat_differentiable N (heads * d_head) ε γF βF hε)

/-- **The ViT body VJP** — `finalLN ∘ transformerTower`: a single
    `vjpMatComp` of the tower + final LN with their Diff helpers.

    A depth-k weight-tied backbone (one parameter tuple for every block,
    scalar-affine LayerNorms) has a correct VJP, composed entirely from
    proved building blocks; `mhsaHasVJPMat` and
    `mhsaLayer_flat_differentiable` are proved through the column-stacking
    framework, so the chain uses no project axioms. -/
noncomputable def vitBodyHasVJPMat (k N heads d_head mlpDim : Nat) (ε : ℝ)
    (hε : 0 < ε)
    (γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (γF βF : ℝ) :
    HasVJPMat (vitBody k N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF) :=
  vjpMatComp _ (fun X : Mat N (heads * d_head) => fun n =>
                   layerNormForward (heads * d_head) ε γF βF (X n))
    (transformerTower_flat_differentiable k N heads d_head mlpDim ε γ1 β1 hε
       Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
    (layerNorm_per_token_flat_differentiable N (heads * d_head) ε γF βF hε)
    (transformerTowerHasVJPMat k N heads d_head mlpDim ε γ1 β1 hε
       Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
    (layerNormPerTokenHasVJPMat N (heads * d_head) ε γF βF hε)

-- ════════════════════════════════════════════════════════════════
-- § 6. The end of the road
-- ════════════════════════════════════════════════════════════════

/-! ## What we've proved (and what's left)

**Proved (zero sorry's, machine-checked):**
- Dense, ReLU (`MLP.lean`)
- Softmax cross-entropy loss gradient (`softmaxCE_grad`, `Softmax.lean`)
- Conv2d, MaxPool, Flatten (`CNN.lean`)
- BatchNorm closed-form backward (`BatchNorm.lean`)
- Residual / biPath fan-in (`Residual.lean`)
- Depthwise conv (`Depthwise.lean`)
- Squeeze-and-Excitation / elementwise product VJP (`SE.lean`)
- LayerNorm, GELU (`LayerNorm.lean`)
- Standalone softmax VJP (`softmaxHasVJP`, `Softmax.lean`)
- Scaled dot-product attention backwards `sdpaBackQ`/`sdpaBackK`/`sdpaBackV`
  (`sdpaBackQ_correct` via `vjpMatComp` composition of four matrix-level
  VJP building blocks: matmul, scalarScale, rowSoftmax, matmul). Formulas
  are also numerically gradient-checked in `check_jacobians.py`.
- Multi-head attention (`mhsaHasVJPMat`), the transformer block
  (`transformerBlockHasVJPMat`), the weight-tied tower and body
  (`transformerTowerHasVJPMat`, `vitBodyHasVJPMat`) and `vitFullHasVJP`
  (this file).

**Three calculus rules do all the structural work** (theorems
proved from Mathlib's `fderiv`):

    pdiv_comp   (chain rule — functions compose, derivatives compose)
    pdiv_add    (linearity — derivatives of sums are sums of derivatives)
    pdiv_mul    (product rule — derivatives of elementwise products)

**Five closed-form "Jacobian-structure tricks"** handle the layers
whose Jacobians are dense but exploitable:

1. **Diagonal** (activations) — collapse the sum_j to one term.
2. **Sparse toeplitz** (conv, depthwise) — reversed/transposed kernels.
3. **Binary selection** (max-pool) — route gradients to argmax cells.
4. **Rank-1 correction to diagonal** (softmax, BN, LN) — one
   extra scalar reduction, everything else is pointwise.
5. **Outer product + reductions** (dense, matmul) — rank-1 update
   accumulation.

**That is the complete taxonomy.** I've thought hard about this and
cannot find a sixth trick or a fourth calculus rule anywhere in the
modern architecture zoo. Every paper, every block, every optimization
is a rearrangement of these eight things.

## What this means for the reader

If you've read this far, you have a complete decoder for the
architecture-of-the-month. Pick any paper — Swin, ConvNeXt, CLIP,
Mamba, anything — and walk through the forward pass. For each
operation, ask:

  1. Is it **composition** of known ops? -> chain rule.
  2. Is it a **sum of branches**? -> fan-in add.
  3. Is it an **elementwise / scalar product of branches**? -> fan-in mul.
  4. Is it an **activation**? -> diagonal Jacobian template.
  5. Is it a **normalization**? -> closed-form three-term formula.
  6. Is it a **convolution or linear map**? -> the structured-matmul
     machinery.
  7. Is it an **attention or softmax-based selection**? -> the closed-form
     rank-1 collapse.

If the answer is "none of the above" — which it won't be — then you've
found the first genuinely new layer of the decade, and you get to
write the next chapter of this book.

Until then, welcome to the end of the road. -/

-- ════════════════════════════════════════════════════════════════
-- § 7. The ACTUAL grand finale — full ViT as a single `HasVJP`
-- ════════════════════════════════════════════════════════════════

/-! ## Bridging ranks: from Mat-land back to Vec-land

`vitBodyHasVJPMat` lives in `HasVJPMat` territory. The pieces at the
boundaries — patch embedding (image → tokens) and classifier head (tokens
→ logits) — change tensor rank. Rather than invent new mixed-rank VJP
frameworks, we flatten everything to `Vec` at the interfaces and compose
via plain `HasVJP`, glued by `vjpComp`.

Two ingredients needed:

- **`HasVJPMat.toHasVJP`** (Tensor.lean) — bridges any
  `HasVJPMat` to `HasVJP` on the flattened endpoints. One theorem, no
  new axioms.
- **`clsTokenFlatHasVJP`** — gathers row 0 of a flattened
  `Mat (N+1) D`. Derivable from `pdiv_reindex`. -/

/-- CLS token extraction, stated on the flattened matrix. Row 0 of a
    `Mat (N+1) D` is a `Vec D`; on the flattened `Vec ((N+1)*D)` this is
    the gather `v ↦ fun k => v (fPF (0, k))`. -/
noncomputable def clsTokenFlat (N D : Nat) :
    Vec ((N + 1) * D) → Vec D :=
  fun v k => v (finProdFinEquiv ((0 : Fin (N + 1)), k))

/-- **CLS slice VJP** — gather-style; backward scatters `dy` to row 0.
    Derived from `pdiv_reindex`. -/
noncomputable def clsTokenFlatHasVJP (N D : Nat) :
    HasVJP (clsTokenFlat N D) where
  backward := fun _v dy => fun idx =>
    let p := finProdFinEquiv.symm idx
    if p.1 = (0 : Fin (N + 1)) then dy p.2 else 0
  correct := by
    intro v dy idx
    obtain ⟨⟨n, d⟩, rfl⟩ := finProdFinEquiv.surjective idx
    unfold clsTokenFlat
    simp only [pdiv_reindex, Equiv.symm_apply_apply, EmbeddingLike.apply_eq_iff_eq,
      Prod.mk.injEq, ite_mul, one_mul, zero_mul, ite_and, Finset.sum_ite_irrel,
      Finset.sum_ite_eq, Finset.mem_univ, ite_true, Finset.sum_const_zero]

/-- **Classifier head**: flattened CLS slice + dense projection to `Vec nClasses`.

    `fun v : Vec ((N+1)*D) => dense W_cls b_cls (clsTokenFlat v)` -/
noncomputable def classifierFlat (N D nClasses : Nat)
    (Wcls : Mat D nClasses) (bcls : Vec nClasses) :
    Vec ((N + 1) * D) → Vec nClasses :=
  (dense Wcls bcls) ∘ (clsTokenFlat N D)

/-- Differentiability of `clsTokenFlat` — linear reindex. -/
lemma clsTokenFlat_differentiable (N D : Nat) :
    Differentiable ℝ (clsTokenFlat N D) := by
  unfold clsTokenFlat; fun_prop

/-- **Classifier head VJP** — composition via `vjpComp`. `clsTokenFlat`
    and `dense` are both linear, so their Diff hypotheses discharge by
    `fun_prop`. -/
noncomputable def classifierFlatHasVJP (N D nClasses : Nat)
    (Wcls : Mat D nClasses) (bcls : Vec nClasses) :
    HasVJP (classifierFlat N D nClasses Wcls bcls) :=
  vjpComp (clsTokenFlat N D) (dense Wcls bcls)
    (clsTokenFlat_differentiable N D)
    (dense_differentiable Wcls bcls)
    (clsTokenFlatHasVJP N D)
    (denseHasVJP Wcls bcls)

/-! ## Patch embedding

The patch embedding takes a flattened image `Vec (ic*H*W)` and produces
a flattened `Vec ((N+1)*D)` interpreted as `Mat (N+1) D`:

1. Conv projection with stride = patchSize: per-patch dense projection
   `W_conv : Kernel4 D ic patchSize patchSize` + bias `b_conv : Vec D`.
2. Reshape spatial `(D, H', W')` to tokens `(N, D)` — pure permutation.
3. Prepend learnable CLS token at row 0 → `(N+1, D)`.
4. Add learnable positional embedding matrix → `(N+1, D)`.

The forward `patchEmbedFlat` is a concrete `def`; its VJP
`patchEmbedFlatHasVJP` and `patchEmbedFlat_differentiable` are proved
from foundation rules.

The `N` parameter is independent of `(H, W, patchSize)` — out-of-range
patches contribute zero (via a `hpad` guard on the image read), so the
API does not require `N = (H/patchSize) * (W/patchSize)`. -/

/-- **Patch embedding forward** on flattened endpoints.

    Output at flat-index `idx_out = finProdFinEquiv (n, d)`:
    - `n = 0`: `cls_token d + pos_embed 0 d`.
    - `n > 0` (let `p := n - 1`, `h' := p / (W/patchSize)`,
      `w' := p % (W/patchSize)`):
      `b_conv d + Σ c kh kw, W_conv d c kh kw * img(c, h'*P+kh, w'*P+kw)
       + pos_embed n d`,
      where the image read is guarded by `h'*P+kh < H ∧ w'*P+kw < W`
      (returns 0 if out of range).

    This handles arbitrary `N` cleanly: for `n` whose decoded patch
    `(h', w')` falls outside the image grid, the inner sum is identically
    zero. -/
noncomputable def patchEmbedFlat
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv : Vec D)
    (cls_token : Vec D) (pos_embed : Mat (N + 1) D) :
    Vec (ic * H * W) → Vec ((N + 1) * D) :=
  fun img =>
    fun idx_out =>
      let n := (finProdFinEquiv.symm idx_out).1
      let d := (finProdFinEquiv.symm idx_out).2
      pos_embed n d +
        (if n.val = 0 then
          cls_token d
         else
          b_conv d +
          ∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
            W_conv d c kh kw *
              (let W' := W / patchSize
               let p := n.val - 1
               let h' := p / W'
               let w' := p % W'
               let hh := h' * patchSize + kh.val
               let ww := w' * patchSize + kw.val
               if hpad : hh < H ∧ ww < W then
                 img (finProdFinEquiv (finProdFinEquiv (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩))
               else 0))

/-- **Patch embedding differentiability** — proved from foundation rules.
    `patchEmbedFlat` is linear in `img` plus constants
    (`pos_embed`, `cls_token`, `b_conv`); the only non-trivial part is
    the dependent `if hpad : ... then img(σ hpad) else 0` pattern handled
    by `differentiableAt_pad_eval`. -/
lemma patchEmbedFlat_differentiable
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv : Vec D)
    (cls_token : Vec D) (pos_embed : Mat (N + 1) D) :
    Differentiable ℝ (patchEmbedFlat ic H W patchSize N D
                       W_conv b_conv cls_token pos_embed) := by
  unfold patchEmbedFlat
  intro img; rw [differentiableAt_pi]; intro idx_out
  by_cases hn : (finProdFinEquiv.symm idx_out).1.val = 0 <;>
    simp only [hn, ite_true, ite_false] <;> fun_prop

/-- **Closed-form input gradient for `patchEmbedFlat`** — direct formula,
    written as a sum over patches `p : Fin N` with reconstructed kernel
    offsets `(kh, kw)` matching the input position `(hh, ww)` decoded from
    `idx_in`. Equivalent (under the patch-row decomposition `h' := p/(W/P)`,
    `w' := p%(W/P)`) to the standard "deconvolution" formula for patchEmbed.

    The CLS row (n = 0) does not appear here — `idx_in` only flows through
    the conv-projection branch (n > 0), so the gradient sums over
    `p : Fin N` (corresponding to output rows `n = p+1`). -/
noncomputable def patchEmbedInputGradFormula
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize)
    (dy : Vec ((N + 1) * D)) : Vec (ic * H * W) :=
  fun idx_in =>
    let c  := (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1
    let hh := (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2
    let ww := (finProdFinEquiv.symm idx_in).2
    ∑ p : Fin N, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
      let W' := W / patchSize
      let h' := p.val / W'
      let w' := p.val % W'
      if _h_match : h' * patchSize + kh.val = hh.val ∧
                    w' * patchSize + kw.val = ww.val then
        ∑ d : Fin D, W_conv d c kh kw *
          dy (finProdFinEquiv (p.succ, d))
      else 0

/-- **Patch embedding VJP — proved from foundation rules.**

    The forward is affine in `img` (`pdiv_of_affine`): the pad-guarded conv read, identically
    zero on the CLS row, plus the constant `pos_embed + (cls_token | b_conv)`. Closing collapse
    mirrors `conv2dHasVJP3`, with one new wrinkle: split `Σ n : Fin (N+1)` into
    `n = 0` (CLS row, contributes 0 to img-grad) + `Σ p : Fin N` (n = p+1)
    via `Fin.sum_univ_succ`.

    Backward: `patchEmbedInputGradFormula W_conv dy`. -/
noncomputable def patchEmbedFlatHasVJP
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv : Vec D)
    (cls_token : Vec D) (pos_embed : Mat (N + 1) D) :
    HasVJP (patchEmbedFlat ic H W patchSize N D W_conv b_conv cls_token pos_embed) where
  backward := fun _img dy => patchEmbedInputGradFormula ic H W patchSize N D W_conv dy
  correct := by
    intro img dy idx_in
    -- Set abbreviations for idx_in's decoded components.
    set c_in : Fin ic :=
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1 with hc_in
    set hh_in : Fin H :=
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2 with hhh_in
    set ww_in : Fin W := (finProdFinEquiv.symm idx_in).2 with hww_in
    have hidx_in : idx_in = finProdFinEquiv (finProdFinEquiv (c_in, hh_in), ww_in) := by
      show idx_in = finProdFinEquiv (finProdFinEquiv
        ((finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1,
         (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2),
        (finProdFinEquiv.symm idx_in).2)
      rw [Prod.mk.eta, Equiv.apply_symm_apply, Prod.mk.eta, Equiv.apply_symm_apply]
    -- Step 1: per-(idx_in, idx_out) pdiv lemma. The forward is affine in the image
    -- (`pdiv_of_affine`): the pad-guarded conv read, zero on the CLS row, plus the
    -- constant `pos_embed + (cls_token | b_conv)`.
    have h_pdiv : ∀ idx_out : Fin ((N + 1) * D),
        pdiv (patchEmbedFlat ic H W patchSize N D
                W_conv b_conv cls_token pos_embed) img idx_in idx_out =
        if _hn0 : (finProdFinEquiv.symm idx_out).1.val = 0 then 0
        else
          ∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
            W_conv (finProdFinEquiv.symm idx_out).2 c kh kw *
              (let W' := W / patchSize
               let p := (finProdFinEquiv.symm idx_out).1.val - 1
               let h' := p / W'
               let w' := p % W'
               let hh := h' * patchSize + kh.val
               let ww := w' * patchSize + kw.val
               if hpad : hh < H ∧ ww < W then
                 (if idx_in = finProdFinEquiv (finProdFinEquiv
                     (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩) then (1 : ℝ) else 0)
               else 0) := by
      intro idx_out
      have hsplit : patchEmbedFlat ic H W patchSize N D W_conv b_conv cls_token pos_embed =
          fun v => (fun k : Fin ((N + 1) * D) =>
            if _hn0 : (finProdFinEquiv.symm k).1.val = 0 then 0
            else
              ∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
                W_conv (finProdFinEquiv.symm k).2 c kh kw *
                  (let W' := W / patchSize
                   let p := (finProdFinEquiv.symm k).1.val - 1
                   let h' := p / W'
                   let w' := p % W'
                   let hh := h' * patchSize + kh.val
                   let ww := w' * patchSize + kw.val
                   if hpad : hh < H ∧ ww < W then
                     v (finProdFinEquiv (finProdFinEquiv (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩))
                   else 0)) +
            fun k => pos_embed (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2 +
              if (finProdFinEquiv.symm k).1.val = 0 then cls_token (finProdFinEquiv.symm k).2
              else b_conv (finProdFinEquiv.symm k).2 := by
        funext v k; unfold patchEmbedFlat; dsimp only [Pi.add_apply]
        by_cases hn : (finProdFinEquiv.symm k).1.val = 0
        · rw [ite_eq_left hn, ite_eq_left hn, dite_eq_left hn, zero_add]
        · rw [ite_eq_right hn, ite_eq_right hn, dite_eq_right hn]; ring
      rw [hsplit, pdiv_of_affine]
      · simp only [basisVec_apply, @eq_comm _ _ idx_in]
      · intro u v; funext k
        simp only [Pi.add_apply, dite_add_dite, add_zero, ← Finset.sum_add_distrib, ← mul_add]
      · intro a v; funext k
        simp only [Pi.smul_apply, smul_eq_mul, mul_dite, mul_zero, Finset.mul_sum, mul_left_comm a]
    -- Step 2: closing collapse.
    show patchEmbedInputGradFormula ic H W patchSize N D W_conv dy idx_in =
         ∑ idx_out : Fin ((N + 1) * D),
           pdiv (patchEmbedFlat ic H W patchSize N D
                   W_conv b_conv cls_token pos_embed) img idx_in idx_out * dy idx_out
    unfold patchEmbedInputGradFormula
    -- Substitute h_pdiv on RHS.
    simp_rw [h_pdiv]
    -- Reindex Σ idx_out → Σ pair via finProdFinEquiv.symm.
    -- Reindex `Σ idx_out` as `Σ (n, d)`; the CLS row `n = 0` contributes nothing.
    rw [← Equiv.sum_comp finProdFinEquiv, Fintype.sum_prod_type, Fin.sum_univ_succ]
    simp only [Equiv.symm_apply_apply, Fin.val_zero, dite_true, zero_mul, Finset.sum_const_zero,
      zero_add]
    -- Per-p: simplify dite_eq_right at p.succ + match formula.
    apply Finset.sum_congr rfl; intro p _
    have h_p_ne : (Fin.succ p).val ≠ 0 := Nat.succ_ne_zero _
    simp_rw [dite_eq_right h_p_ne]
    -- The `(p.succ).val - 1 = p.val` simplification.
    have h_p_succ_sub : (Fin.succ p).val - 1 = p.val := by
      show p.val + 1 - 1 = p.val
      omega
    -- Per-p, prove the indicator helper: dite-form ⇔ conjunction-form.
    have h_indicator : ∀ c : Fin ic, ∀ kh kw : Fin patchSize,
        ((let W' := W / patchSize
          let p_val := (Fin.succ p).val - 1
          let h' := p_val / W'
          let w' := p_val % W'
          let hh := h' * patchSize + kh.val
          let ww := w' * patchSize + kw.val
          if hpad : hh < H ∧ ww < W then
            (if idx_in = finProdFinEquiv (finProdFinEquiv
                (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩) then (1 : ℝ) else 0)
          else 0) : ℝ) =
        (if c = c_in ∧
              p.val / (W / patchSize) * patchSize + kh.val = hh_in.val ∧
              p.val % (W / patchSize) * patchSize + kw.val = ww_in.val then
          (1 : ℝ) else 0) := by
      intro c kh kw
      simp only [h_p_succ_sub]
      by_cases hpad : p.val / (W / patchSize) * patchSize + kh.val < H ∧
                      p.val % (W / patchSize) * patchSize + kw.val < W
      · rw [dite_eq_left hpad, hidx_in]
        refine if_congr ?_ rfl rfl
        simp only [EmbeddingLike.apply_eq_iff_eq, Prod.mk.injEq, Fin.ext_iff, @eq_comm _ c_in,
          @eq_comm _ hh_in.val, @eq_comm _ ww_in.val, and_assoc]
      · rw [dite_eq_right hpad, ite_eq_right]
        rintro ⟨_, h_hh, h_ww⟩
        exact hpad ⟨h_hh ▸ hh_in.isLt, h_ww ▸ ww_in.isLt⟩
    -- Use h_indicator to rewrite dite to conjunction-indicator.
    simp_rw [h_indicator]
    -- Collapse `c = c_in` and float both sides to `∑ kh, [row match] ∑ kw, [col match] …`.
    simp only [ite_and, mul_ite, mul_one, mul_zero, Finset.sum_ite_irrel, Finset.sum_const_zero,
      Finset.sum_ite_eq', Finset.mem_univ, ite_true, Finset.sum_mul, ite_mul, zero_mul, dite_eq_ite]
    -- The RHS sums `d` outermost; move it inside both window sums.
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun kh _ => ?_
    rw [Finset.sum_ite_irrel, Finset.sum_const_zero]
    refine if_congr Iff.rfl ?_ rfl
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun kw _ => ?_
    rw [Finset.sum_ite_irrel, Finset.sum_const_zero]

/-! ## The full ViT theorem

Compose patch embed + ViT body (via the Mat→Vec bridge) + classifier.
All three are `HasVJP`s on `Vec`, so `vjpComp` chains them directly. -/

/-- **vitFull** — a weight-tied ViT forward from flattened image pixels to logits:
    all `kBlocks` transformer blocks share one parameter tuple, and every
    LayerNorm (`γ1 β1`, `γ2 β2`, final `γF βF`) is scalar-affine.

    `Vec (ic*H*W) → Vec nClasses`

    Composition: `patchEmbed → (flatten ∘ vitBody ∘ unflatten) → classifier`.
    Uses `D := heads * d_head` directly (no separate `D` parameter) so the
    type-level reinterpretation at the body is a no-op. -/
noncomputable def vitFull
    (ic H W patchSize N mlpDim heads d_head kBlocks nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (γF βF : ℝ)
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    Vec (ic * H * W) → Vec nClasses :=
  (classifierFlat N (heads * d_head) nClasses Wcls bcls) ∘
  (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten
      (vitBody kBlocks (N + 1) heads d_head mlpDim ε γ1 β1
         Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF
       (Mat.unflatten v))) ∘
  (patchEmbedFlat ic H W patchSize N (heads * d_head)
    W_conv b_conv cls_token pos_embed)

/-- Differentiability of the classifier head — composition of linear ops. -/
lemma classifierFlat_differentiable (N D nClasses : Nat)
    (Wcls : Mat D nClasses) (bcls : Vec nClasses) :
    Differentiable ℝ (classifierFlat N D nClasses Wcls bcls) := by
  unfold classifierFlat
  exact (dense_differentiable Wcls bcls).comp (clsTokenFlat_differentiable N D)

/-- **vitFull VJP** — the VJP of the weight-tied `vitFull`: three
    `vjpComp` steps glueing `patchEmbedFlatHasVJP`,
    `HasVJPMat.toHasVJP (vitBodyHasVJPMat ...)`, and
    `classifierFlatHasVJP`. Each `vjpComp`'s Diff hypotheses are
    discharged by the per-stage Diff theorems above. -/
noncomputable def vitFullHasVJP
    (ic H W patchSize N mlpDim heads d_head kBlocks nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (γF βF : ℝ)
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    HasVJP (vitFull ic H W patchSize N mlpDim heads d_head kBlocks nClasses
              W_conv b_conv cls_token pos_embed
              ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
              γ2 β2 Wfc1 bfc1 Wfc2 bfc2
              γF βF Wcls bcls) :=
  -- Inner: patchEmbed
  let body_bridge : HasVJP (fun v : Vec ((N + 1) * (heads * d_head)) =>
        Mat.flatten (vitBody kBlocks (N + 1) heads d_head mlpDim ε γ1 β1
                       Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF
                     (Mat.unflatten v))) :=
    HasVJPMat.toHasVJP (vitBodyHasVJPMat kBlocks (N + 1) heads d_head mlpDim
                          ε hε γ1 β1 Wq Wk Wv Wo bq bk bv bo
                          γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF)
  let body_bridge_diff := vitBody_flat_differentiable kBlocks (N + 1) heads d_head mlpDim
                            ε hε γ1 β1 Wq Wk Wv Wo bq bk bv bo
                            γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF
  let patch_diff := patchEmbedFlat_differentiable ic H W patchSize N (heads * d_head)
                      W_conv b_conv cls_token pos_embed
  let patchHasVJP := patchEmbedFlatHasVJP ic H W patchSize N (heads * d_head)
                        W_conv b_conv cls_token pos_embed
  -- Inner composition: body_bridge ∘ patchEmbed
  let innerHasVJP := vjpComp _ _ patch_diff body_bridge_diff
                        patchHasVJP body_bridge
  have inner_diff : Differentiable ℝ
      ((fun v : Vec ((N + 1) * (heads * d_head)) =>
          Mat.flatten (vitBody kBlocks (N + 1) heads d_head mlpDim ε γ1 β1
                         Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF
                       (Mat.unflatten v))) ∘
        (patchEmbedFlat ic H W patchSize N (heads * d_head)
           W_conv b_conv cls_token pos_embed)) :=
    body_bridge_diff.comp patch_diff
  -- Outer: classifierFlat ∘ (body_bridge ∘ patchEmbed)
  vjpComp _ _ inner_diff
    (classifierFlat_differentiable N (heads * d_head) nClasses Wcls bcls)
    innerHasVJP
    (classifierFlatHasVJP N (heads * d_head) nClasses Wcls bcls)

/-! ## Public correctness theorems for the attention defs

The `HasVJP` / `HasVJPMat` defs above bundle a backward function
with a `.correct` field; these `_correct` theorems expose that field
as a top-level proposition so consumers can refer to the contract
directly without reaching into record internals. -/

/-- **Public correctness theorem for `mhsaHasVJPMat`**: multi-head
SDPA's backward equals the `pdivMat`-contracted Jacobian. The
column-stacking proof (`mhsaGHasVJPMat`, `colSlabwiseHasVJPMat`) uses no
project axiom. -/
theorem mhsaHasVJPMat_correct (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (X : Mat N (heads * d_head)) (dY : Mat N (heads * d_head))
    (i : Fin N) (j : Fin (heads * d_head)) :
    (mhsaHasVJPMat N heads d_head Wq Wk Wv Wo bq bk bv bo).backward X dY i j =
    ∑ k : Fin N, ∑ l : Fin (heads * d_head),
      pdivMat (mhsaLayer N heads d_head Wq Wk Wv Wo bq bk bv bo)
              X i j k l * dY k l :=
  (mhsaHasVJPMat N heads d_head Wq Wk Wv Wo bq bk bv bo).correct X dY i j

/-- **Public correctness theorem for `transformerBlockHasVJPMat`**:
the full transformer block backward (attention sublayer + MLP sublayer
glued by `vjpMatComp`) equals the `pdivMat`-contracted Jacobian. -/
theorem transformerBlockHasVJPMat_correct
    (N heads d_head mlpDim : Nat)
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (X : Mat N (heads * d_head)) (dY : Mat N (heads * d_head))
    (i : Fin N) (j : Fin (heads * d_head)) :
    (transformerBlockHasVJPMat N heads d_head mlpDim ε γ1 β1 hε
        Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2).backward X dY i j =
    ∑ k : Fin N, ∑ l : Fin (heads * d_head),
      pdivMat (transformerBlock N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
              X i j k l * dY k l :=
  (transformerBlockHasVJPMat N heads d_head mlpDim ε γ1 β1 hε
     Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2).correct X dY i j

/-- **Public correctness theorem for `vitFullHasVJP`**: the backward of
    `vitFull` (weight-tied blocks, scalar-affine LayerNorms) equals the `pdiv`-contracted Jacobian (Jacobian-transpose applied to
    the cotangent). Exposes the witness's `.correct` field as a top-level
    proposition so consumers (and `#print axioms` audits) can cite the apex
    contract directly instead of reaching into the record. The long signature is
    `vitFull`'s parameter set; the proof is the witness field. -/
theorem vitFullHasVJP_correct
    (ic H W patchSize N mlpDim heads d_head kBlocks nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (γF βF : ℝ)
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) (dy : Vec nClasses) (i : Fin (ic * H * W)) :
    (vitFullHasVJP ic H W patchSize N mlpDim heads d_head kBlocks nClasses
        W_conv b_conv cls_token pos_embed ε γ1 β1 hε
        Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF Wcls bcls).backward x dy i =
      ∑ j : Fin nClasses,
        pdiv (vitFull ic H W patchSize N mlpDim heads d_head kBlocks nClasses
                W_conv b_conv cls_token pos_embed ε γ1 β1
                Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF Wcls bcls)
             x i j * dy j :=
  (vitFullHasVJP ic H W patchSize N mlpDim heads d_head kBlocks nClasses
      W_conv b_conv cls_token pos_embed ε γ1 β1 hε
      Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF Wcls bcls).correct x dy i

end Proofs
