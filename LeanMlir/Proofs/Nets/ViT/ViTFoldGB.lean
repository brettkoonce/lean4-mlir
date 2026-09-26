import LeanMlir.Proofs.Nets.ViT.ViTFoldG
import LeanMlir.Proofs.Foundation.GradNodesB

/-! # The gradient-node fold for ViT-Tiny at the batched index

`ViTFoldG.lean` holds the two per-example lemmas specific to this net (`posEmbedGrad_den`,
`clsGrad_den`); the other per-example peers are the fused `ViTPoC` lemmas. This file is the batched
peer, at `ViTRenderB.vitBackAllB`'s constructors, from which the committed ViT artifacts render.

Every batched form emits its per-example peer's text, and tests/TestBatchedEmitTie.lean pins each
one individually, so this file is not about different bytes; it is about the AST those bytes are
`pretty` of.

**The CLS token.**
The CLS token is ONE shared `[192]` vector, so its gradient is the sum of every example's CLS-row
cotangent. The per-example render emits `denseBiasGradB (N := 1)` — "sum one thing", correct there
because `pretty B` performed the batch lift OUTSIDE the AST — where the batched one emits
`denseBiasGradB (N := vbB)` and the sum is inside `den`. `clsGrad_denB` below is the
statement the per-example `clsGrad_den` could not make. Same emitted text either way, which is
why the byte tie cannot see it and `den_rowDenseBiasGradB_at_one` exists to argue the point.

## The op table of the committed f32 ViT train steps

The vector-LN and classifier-head lemmas, and the vector-LN tie clauses, are in `GradNodesB` under
this namespace, because ConvNeXt's batched tier uses them too.

| emitted node | lemma | per-example peer (fused `ViTPoC` op unless noted) |
|---|---|---|
| `veclnGammaGradB` (25 LN γ: LN1/LN2 × 12 + final) | `veclnGammaGradB_den` (`GradNodesB`) | `ViTPoC.veclnGammaSgd_den` |
| `rowDenseBiasGradB` (25 LN β) | `rowDenseBiasGradB_den_lnbeta` (`GradNodesB`) | `ViTPoC.rowDenseBiasSgd_den_lnbeta` |
| `rowDenseWeightGradB` (Wq/Wk/Wv/Wo/Wfc1/Wfc2 × 12) | `rowDenseWeightGradB_den` | `ViTPoC.rowDenseWeightSgd_den` |
| `rowDenseBiasGradB` (bq/bk/bv/bo/bfc1/bfc2 × 12) | `rowDenseBiasGradB_den` | `ViTPoC.rowDenseBiasSgd_den` |
| `patchEmbedWeightGradB` / `patchEmbedBiasGradB` | `patchEmbedWeightGradB_den` / `patchEmbedBiasGradB_den` | `ViTPoC.patchEmbedWeightSgd_den` / `patchEmbedBiasSgd_den` |
| `posEmbedGradB` | `posEmbedGradB_den` | `ViTPoCG.posEmbedGrad_den` (un-fused) |
| `denseBiasGradB` at `N = B` (the CLS token) | `clsGrad_denB` | `ViTPoCG.clsGrad_den` (un-fused), at `N = 1` |
| `weightGradB` / `biasGradB` (the classifier) | `headWGradB_den` / `headBGradB_den` (`GradNodesB`) | `ViTPoC.headW_den` / `headB_den` |

**No new mathematics: every proof is `Finset.sum_congr rfl` over the batch and then the
per-example bridge at `batchSlice n`.** That is `ResNet34PoCB.denseWGradB_den`'s shape, and
it is available because each batched `den` arm is literally `∑_batch` of the per-example one — the
constructors were written that way (`StableHLO.lean`'s own comment on `veclnGammaGradB`: *"TWO-LEVEL:
the outer `Σ_n` is the batch, the inner `Σ_r` the rows within one example"*).

**One lemma per f32 node kind.** Every lemma is `∀ cot`: the f32 AdamW, `wx`/`clip` and EMA
artifacts all emit these `*GradB` kinds, and the optimizer update that consumes the node is outside
these lemmas. The artifact whose accuracy the book quotes, `vitin_emadp128x4wxclipdropbf16`, is a
bf16 one, so its weight gradients go through the bf16 kinds listed below.

**The LayerNorm form is the vector one** (`γ β : Vec D`), which is what `vitForwardKV` runs.

## Scope
* **`biasGradB` is the identity on its operand** and the classifier bias's batch reduce is in the
  emitted text, outside the AST — the constructor says so (*"the channel sum happens in the emitted
  reduce"*) and it is the per-example `biasGrad` carve-out carried over unchanged, not a new one.
  So `headBGradB_den` is stated PER EXAMPLE, at `batchSlice n`, which is the whole of what the node
  denotes.
* Every lemma is `∀ cot`. The tie at these nodes, with the cotangents the emitted backward chain
  delivers, is `ViTTiePoCGB.vit_net_tiedGB`; the per-example `ViTStepTie.lean` stays at the
  SGD-inline `vit_train_step.mlir`.
* The `*bf16` artifacts emit `rowDenseWeightGradBBf16` / `patchEmbedWeightGradBBf16`, their own
  kinds; [`Foundation/Bf16GradNodes.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Foundation/Bf16GradNodes.lean) folds them (the row-dense one keeps its f32 result).
* `vitin_*dp128x4*` is four replicas: the all-reduce is its own `allReduceMeanF` node after each
  gradient node (`DataParallelNode.lean`), so these lemmas are about the per-replica gradient node
  it averages.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ViTPoCGB

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The per-token denses — Wq/Wk/Wv/Wo/Wfc1/Wfc2 and their biases
-- ════════════════════════════════════════════════════════════════

/-- **Batched per-token dense weight GRADIENT denotes the certified `Σ_n Σ_tokens x⊗dy`.** All six
    denses in each of the 12 blocks. Note: the emitted `dot_general` contracts batch AND token in
    ONE op; the two sums here are that contraction read apart. -/
theorem rowDenseWeightGradB_den {N tk a c : Nat} (xN cotN : String)
    (bb : Vec c) (x : Vec (N * (tk * a))) (W : Mat a c) (dy : Vec (N * (tk * c)))
    (i : Fin a) (j : Fin c) :
    den (SHlo.rowDenseWeightGradB (N := N) (tk := tk) (a := a) (c := c) xN x (.operand cotN dy))
        (finProdFinEquiv (i, j))
      = ∑ n : Fin N, ∑ o : Fin (tk * c),
          pdiv (fun v : Vec (a * c) =>
                  Mat.flatten (fun r =>
                    dense (Mat.unflatten v) bb (Mat.unflatten (batchSlice N (tk * a) x n) r)))
               (Mat.flatten W) (finProdFinEquiv (i, j)) o
            * batchSlice N (tk * c) dy n o := by
  simp only [denStep, denStepApp, Mat.flatten, Equiv.symm_apply_apply]
  apply Finset.sum_congr rfl
  intro n _
  exact vit_rowDenseW_grad_bridge bb
    (Mat.unflatten (batchSlice N (tk * a) x n)) W (batchSlice N (tk * c) dy n) i j

/-- **Batched per-token dense bias GRADIENT denotes the certified `Σ_n Σ_tokens dy`.**
    bq/bk/bv/bo/bfc1/bfc2. -/
theorem rowDenseBiasGradB_den {N tk a c : Nat} (cotN : String)
    (W : Mat a c) (X : Fin N → Mat tk a) (b : Vec c) (dy : Vec (N * (tk * c))) (i : Fin c) :
    den (SHlo.rowDenseBiasGradB (N := N) (R := tk) (c := c) (.operand cotN dy)) i
      = ∑ n : Fin N, ∑ o : Fin (tk * c),
          pdiv (fun b' : Vec c => Mat.flatten (fun r => dense W b' (X n r))) b i o
            * batchSlice N (tk * c) dy n o := by
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact vit_rowDenseb_grad_bridge W (X n) b (batchSlice N (tk * c) dy n) i

-- ════════════════════════════════════════════════════════════════
-- § The patch embedding — conv weight/bias, the CLS token, the positional table
-- ════════════════════════════════════════════════════════════════

/-- **Batched patch-embed conv weight GRADIENT denotes the certified `Σ_n` patchify weight
    gradient.** Note: the emitted `convolution` contracts the batch axis itself, so the outer sum
    here is inside one op rather than across `N` of them. -/
theorem patchEmbedWeightGradB_den {ic H W P tk D N : Nat} (xN cotN : String)
    (bc cls : Vec D) (pos : Mat (tk + 1) D) (img : Vec (N * (ic * H * W)))
    (Wp : Kernel4 D ic P P) (dy : Vec (N * ((tk + 1) * D)))
    (d : Fin D) (c : Fin ic) (kh kw : Fin P) :
    den (SHlo.patchEmbedWeightGradB (N := N) (ic := ic) (H := H) (W := W) (P := P) (tk := tk)
            (D := D) xN img (.operand cotN dy))
        (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw))
      = ∑ n : Fin N, ∑ o : Fin ((tk + 1) * D),
          pdiv (fun v : Vec (D * ic * P * P) =>
                  patchEmbedFlat ic H W P tk D (Kernel4.unflatten v) bc cls pos
                    (batchSlice N (ic * H * W) img n))
            (Kernel4.flatten Wp)
            (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw)) o
            * batchSlice N ((tk + 1) * D) dy n o := by
  simp only [denStep, denStepApp, patchEmbedWeightGradFlat, Kernel4.flatten, Equiv.symm_apply_apply]
  apply Finset.sum_congr rfl
  intro n _
  exact vit_patchW_grad_bridge Wp bc cls pos
    (batchSlice N (ic * H * W) img n) (batchSlice N ((tk + 1) * D) dy n) d c kh kw

/-- **Batched patch-embed conv bias GRADIENT denotes the certified `Σ_n` bias gradient**
    (`Σ_patches dy`, the CLS row excluded — the inner `p.succ`). -/
theorem patchEmbedBiasGradB_den {ic H W P tk D N : Nat} (cotN : String)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (tk + 1) D)
    (img : Vec (N * (ic * H * W))) (dy : Vec (N * ((tk + 1) * D))) (i : Fin D) :
    den (SHlo.patchEmbedBiasGradB (N := N) (tk := tk) (c := D) (.operand cotN dy)) i
      = ∑ n : Fin N, ∑ o : Fin ((tk + 1) * D),
          pdiv (fun b' : Vec D =>
                  patchEmbedFlat ic H W P tk D Wc b' cls pos
                    (batchSlice N (ic * H * W) img n)) bc i o
            * batchSlice N ((tk + 1) * D) dy n o := by
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact vit_patchb_grad_bridge Wc bc cls pos
    (batchSlice N (ic * H * W) img n) (batchSlice N ((tk + 1) * D) dy n) i

/-- **Batched positional-embed GRADIENT denotes the certified `Σ_n` gradient** — the summed
    cotangent, since the positional table is added to every token and its Jacobian is the
    identity. Note: this batch sum is invisible at
    `N = 1`: a render that dropped it type-checks and emits the same bytes. -/
theorem posEmbedGradB_den {ic H W P tk D N : Nat} (cotN : String)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (tk + 1) D)
    (img : Vec (N * (ic * H * W))) (dy : Vec (N * ((tk + 1) * D))) (i : Fin ((tk + 1) * D)) :
    den (SHlo.posEmbedGradB (N := N) (tk := tk) (D := D) (.operand cotN dy)) i
      = ∑ n : Fin N, ∑ o : Fin ((tk + 1) * D),
          pdiv (fun p : Vec ((tk + 1) * D) =>
                  patchEmbedFlat ic H W P tk D Wc bc cls (Mat.unflatten p)
                    (batchSlice N (ic * H * W) img n))
            (Mat.flatten pos) i o
            * batchSlice N ((tk + 1) * D) dy n o := by
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact Proofs.ViTPoCG.posEmbedGrad_den (ic := ic) (H := H) (W := W) (P := P) (N := tk) (D := D)
    cotN Wc bc cls pos (batchSlice N (ic * H * W) img n) (batchSlice N ((tk + 1) * D) dy n) i

/-- **The CLS-token GRADIENT, and this is the one statement the per-example fold could not make.**
    The render slices row 0 of every example's embed cotangent (`clsSlice`, lifted by `batchOp`)
    and reduces the result as an `[N, D]` batch, so the op is `denseBiasGradB` at `N = B` and its
    `den` sums over the batch — which is what a shared `[192]` parameter's gradient IS.

    `ViTPoCG.clsGrad_den` is the same theorem at `N = 1`, where the batch lift lived in
    `pretty B` outside the AST. The bytes are identical (`biasGrad`'s emitted reduce takes the `B`
    axis either way) and the functions are not; `den_rowDenseBiasGradB_at_one` is the general form
    of the trap.

    Note: stated at the committed ViT-Tiny dims rather than generically, for `ViTPoCG.clsGrad_den`'s
    reason: the CLS operand's type is `Vec (N * (1 * D))`, which reduces to `Vec (N * D)` only at a
    literal `D`. -/
theorem clsGrad_denB {N : Nat} (cotN : String)
    (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192) (pos : Mat 197 192)
    (img : Vec (N * (3 * 224 * 224))) (dyEmbed : Vec (N * (197 * 192))) (i : Fin 192) :
    den (SHlo.denseBiasGradB (N := N) (c := 192)
            (.operand cotN (batchMap N (clsSliceFlat 196 192) dyEmbed))) i
      = ∑ n : Fin N, ∑ j : Fin (197 * 192),
          pdiv (fun cl : Vec 192 =>
                  patchEmbedFlat 3 224 224 16 196 192 Wc bc cl pos
                    (batchSlice N (3 * 224 * 224) img n)) cls i j
            * batchSlice N (197 * 192) dyEmbed n j := by
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  have hslice : batchSlice N 192 (batchMap N (clsSliceFlat 196 192) dyEmbed) n
      = clsSliceFlat 196 192 (batchSlice N (197 * 192) dyEmbed n) := by
    funext k
    simp [batchSlice, batchMap, clsSliceFlat, Equiv.symm_apply_apply]
  rw [hslice]
  -- ⚠ The per-example lemma's LHS is the `N = 1` node, whose `den` is a one-term sum; unfolding it
  -- is what leaves the bare `clsSliceFlat` this goal is stated at.
  have h := Proofs.ViTPoCG.clsGrad_den cotN Wc bc cls pos
    (batchSlice N (3 * 224 * 224) img n) (batchSlice N (197 * 192) dyEmbed n) i
  simp only [denStepApp, Fin.sum_univ_one, batchSlice] at h
  exact h

-- ════════════════════════════════════════════════════════════════
-- § Tie clauses — one batched gradient node each (each its `_den` lemma's statement with the index
--   bound, over the flat input `x`; each `…_holds` below proves it)
-- ════════════════════════════════════════════════════════════════

/-- A batched per-token dense weight gradient node, tied (`rowDenseWeightGradB_den`). -/
def RowDenseWTiedB (N tk : Nat) {a c : Nat} (xN cotN : String) (bb : Vec c)
    (x : Vec (N * (tk * a))) (W : Mat a c) (dy : Vec (N * (tk * c))) : Prop :=
  ∀ (i : Fin a) (j : Fin c),
    den (SHlo.rowDenseWeightGradB (N := N) (tk := tk) (a := a) (c := c) xN x (.operand cotN dy))
        (finProdFinEquiv (i, j))
      = ∑ n : Fin N, ∑ o : Fin (tk * c),
          pdiv (fun v : Vec (a * c) =>
                  Mat.flatten (fun r =>
                    dense (Mat.unflatten v) bb (Mat.unflatten (batchSlice N (tk * a) x n) r)))
               (Mat.flatten W) (finProdFinEquiv (i, j)) o
            * batchSlice N (tk * c) dy n o

/-- A batched per-token dense bias gradient node, tied (`rowDenseBiasGradB_den`). -/
def RowDenseBTiedB (N tk : Nat) {a c : Nat} (cotN : String) (W : Mat a c)
    (x : Vec (N * (tk * a))) (b : Vec c) (dy : Vec (N * (tk * c))) : Prop :=
  ∀ i : Fin c,
    den (SHlo.rowDenseBiasGradB (N := N) (R := tk) (c := c) (.operand cotN dy)) i
      = ∑ n : Fin N, ∑ o : Fin (tk * c),
          pdiv (fun b' : Vec c =>
                  Mat.flatten (fun r => dense W b' (Mat.unflatten (batchSlice N (tk * a) x n) r)))
               b i o
            * batchSlice N (tk * c) dy n o

/-! Each clause holds, every argument implicit (read off the goal by a step tie's constructor). -/

theorem rowDenseWTiedB_holds {N tk a c : Nat} {xN cotN : String} {bb : Vec c}
    {x : Vec (N * (tk * a))} {W : Mat a c} {dy : Vec (N * (tk * c))} :
    RowDenseWTiedB N tk xN cotN bb x W dy := fun i j =>
  rowDenseWeightGradB_den xN cotN bb x W dy i j

theorem rowDenseBTiedB_holds {N tk a c : Nat} {cotN : String} {W : Mat a c}
    {x : Vec (N * (tk * a))} {b : Vec c} {dy : Vec (N * (tk * c))} :
    RowDenseBTiedB N tk cotN W x b dy := fun i =>
  rowDenseBiasGradB_den cotN W (fun n => Mat.unflatten (batchSlice N (tk * a) x n)) b dy i

end Proofs.ViTPoCGB
