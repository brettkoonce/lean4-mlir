import LeanMlir.Proofs.Architectures.ViTFaithfulPoCG

/-! # T3 §1 fold for ViT-Tiny at the BATCHED index — the op set 4c leg 4 renders

`ViTFaithfulPoCG.lean` folds the ten gradient nodes of the PER-EXAMPLE traversal
(`ViTRender.vitBackAll` at `adam := true`). This is its batched peer, at
`ViTRenderB.vitBackAllB`'s constructors, and it exists because 4c leg 4 moves every committed ViT
artifact onto that traversal.

⭐ **The bytes do not move and the denotation does.** Measured 2026-09-07: all nineteen drop-free
ViT artifacts — `vit_fwd`, `vitin_fwd` and the seventeen AdamW/EMA train steps — re-render
**byte-identically** off `vitBackAllB`, because every batched form was built to emit its
per-example peer's text and `tests/TestBatchedEmitTie.lean` pins each one individually. So this
file is not about different bytes; it is about the AST those bytes are `pretty` of.

⭐⭐ **And on one parameter the AST is genuinely better, which is the whole point of the leg.**
The CLS token is ONE shared `[192]` vector, so its gradient is the sum of every example's CLS-row
cotangent. The per-example render emits `denseBiasGradB (N := 1)` — "sum one thing", correct there
because `pretty B` performed the batch lift OUTSIDE the AST — where the batched one emits
`denseBiasGradB (N := vbB)` and the sum is inside `den`. `ViTRenderB.lean` flags that line as
*"THE ONE LINE WHERE `N := 1 → N := vbB` CHANGES THE FUNCTION"*, and `clsGrad_denB` below is the
statement the per-example `clsGrad_den` could not make. Same emitted text either way, which is
why the byte tie cannot see it and `den_rowDenseBiasGradB_at_one` exists to argue the point.

## The op table of every committed ViT train step, after leg 4

| emitted node | lemma | per-example peer it batches |
|---|---|---|
| `veclnGammaGradB` (25 LN γ: LN1/LN2 × 12 + final) | `veclnGammaGradB_den` | `ViTPoCG.veclnGammaGrad_den` |
| `rowDenseBiasGradB` (25 LN β) | `rowDenseBiasGradB_den_lnbeta` | `ViTPoCG.rowDenseBiasGrad_den_lnbeta` |
| `rowDenseWeightGradB` (Wq/Wk/Wv/Wo/Wfc1/Wfc2 × 12) | `rowDenseWeightGradB_den` | `ViTPoCG.rowDenseWeightGrad_den` |
| `rowDenseBiasGradB` (bq/bk/bv/bo/bfc1/bfc2 × 12) | `rowDenseBiasGradB_den` | `ViTPoCG.rowDenseBiasGrad_den` |
| `patchEmbedWeightGradB` / `patchEmbedBiasGradB` | `patchEmbedWeightGradB_den` / `patchEmbedBiasGradB_den` | the `*Grad_den` pair |
| `posEmbedGradB` | `posEmbedGradB_den` | `ViTPoCG.posEmbedGrad_den` |
| `denseBiasGradB` at `N = B` (the CLS token) | `clsGrad_denB` | `ViTPoCG.clsGrad_den`, at `N = 1` |
| `weightGradB` / `biasGradB` (the classifier) | `headWGradB_den` / `headBGradB_den` | `ViTPoCG.headWGrad_den` / `headBGrad_den` |

⭐ **No new mathematics: every proof is `Finset.sum_congr rfl` over the batch and then the
per-example bridge at `batchSlice n`.** That is `ResNet34FaithfulPoCB.denseWGradB_den`'s shape, and
it is available because each batched `den` arm is literally `∑_batch` of the per-example one — the
constructors were written that way (`StableHLO.lean`'s own comment on `veclnGammaGradB`: *"TWO-LEVEL:
the outer `Σ_n` is the batch, the inner `Σ_r` the rows within one example"*).

⭐ **One lemma per op kind certifies every optimizer tail at once** — AdamW, the `wx`/`clip`
variants, the EMA shadow and the 4× accumulation all consume the same `*GradB` node, and the
`vitin_adamdp128x4wxclipdrop` artifact whose accuracy the book quotes is one of them.

⭐ **The LayerNorm form is the VECTOR one** (`γ β : Vec D`), which is what the shipped
`vitForwardKV` runs; the scalar-affine spelling this cone was caught on three times is nowhere here.

## Honest residual
* ⚠ **`biasGradB` is the IDENTITY on its operand** and the classifier bias's batch reduce is in the
  emitted text, outside the AST — the constructor says so (*"the channel sum happens in the emitted
  reduce"*) and it is the per-example `biasGrad` carve-out carried over unchanged, not a new one.
  So `headBGradB_den` is stated PER EXAMPLE, at `batchSlice n`, which is the whole of what the node
  denotes.
* Every lemma is `∀ cot`. Pinning each to the emitted backward subgraph is the §1a tie:
  `ViTTiePoCGB.vit_net_tiedGB` at these nodes (4b.7); the per-example `ViTTiePoC.lean` stays at
  the SGD-inline `vit_train_step.mlir`.
* The `*bf16` artifacts emit `rowDenseWeightGradBBf16` / `patchEmbedWeightGradBBf16`, their own
  kinds; `Foundation/Bf16GradNodes.lean` folds them (the row-dense one keeps its f32 result).
* `vitin_adamdp128x4*` is four replicas: the all-reduce is emitted text outside the AST, so these
  lemmas are about the per-replica gradient node (4d).
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ViTPoCGB

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The vector LayerNorm — 25 γ sites and 25 β sites
-- ════════════════════════════════════════════════════════════════

/-- **Batched vector-LN γ GRADIENT denotes the certified `Σ_n` γ gradient.** Two levels: the outer
    sum is the batch, the inner one the tokens within one example. All 25 sites. -/
theorem veclnGammaGradB_den {N R D : Nat} (xN epsStr cotN : String)
    (ε : ℝ) (βv : Vec D) (x : Vec (N * (R * D))) (γ : Vec D) (dy : Vec (N * (R * D)))
    (k : Fin D) :
    den (SHlo.veclnGammaGradB (N := N) (R := R) (D := D) xN epsStr ε x (.operand cotN dy)) k
      = ∑ n : Fin N, ∑ o : Fin (R * D),
          pdiv (fun gv : Vec D =>
                  Mat.flatten (fun r =>
                    layerNormVec D ε gv βv (Mat.unflatten (batchSlice N (R * D) x n) r))) γ k o
            * batchSlice N (R * D) dy n o := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact vit_veclnGamma_grad_bridge ε βv γ
    (Mat.unflatten (batchSlice N (R * D) x n)) (batchSlice N (R * D) dy n) k

/-- **The SAME row-reduce op, certified against the vector-LN β forward.** An LN β gradient and a
    dense bias gradient are the identical two-level reduce, so this constructor appears twice in
    the table against two different certified Jacobians — as it does per example. All 25 β sites. -/
theorem rowDenseBiasGradB_den_lnbeta {N R D : Nat} (cotN : String)
    (ε : ℝ) (γv : Vec D) (X : Fin N → Mat R D) (β : Vec D) (dy : Vec (N * (R * D)))
    (i : Fin D) :
    den (SHlo.rowDenseBiasGradB (N := N) (R := R) (c := D) (.operand cotN dy)) i
      = ∑ n : Fin N, ∑ o : Fin (R * D),
          pdiv (fun bv : Vec D => Mat.flatten (fun r => layerNormVec D ε γv bv (X n r))) β i o
            * batchSlice N (R * D) dy n o := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact vit_veclnBeta_grad_bridge ε γv β (X n) (batchSlice N (R * D) dy n) i

-- ════════════════════════════════════════════════════════════════
-- § The per-token denses — Wq/Wk/Wv/Wo/Wfc1/Wfc2 and their biases
-- ════════════════════════════════════════════════════════════════

/-- **Batched per-token dense weight GRADIENT denotes the certified `Σ_n Σ_tokens x⊗dy`.** All six
    denses in each of the 12 blocks. ⚠ The emitted `dot_general` contracts batch AND token in ONE
    op; the two sums here are that contraction read apart. -/
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
  simp only [den, Mat.flatten, Equiv.symm_apply_apply]
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
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact vit_rowDenseb_grad_bridge W (X n) b (batchSlice N (tk * c) dy n) i

-- ════════════════════════════════════════════════════════════════
-- § The patch embedding — conv weight/bias, the CLS token, the positional table
-- ════════════════════════════════════════════════════════════════

/-- **Batched patch-embed conv weight GRADIENT denotes the certified `Σ_n` patchify weight
    gradient.** ⚠ The emitted `convolution` contracts the batch axis itself, so the outer sum here
    is inside one op rather than across `N` of them. -/
theorem patchEmbedWeightGradB_den {ic H W P tk D N : Nat} (xN cotN : String)
    (bc cls : Vec D) (pos : Mat (tk + 1) D) (img : Vec (N * (ic * H * W)))
    (Wp : Kernel4 D ic P P) (dy : Vec (N * ((tk + 1) * D)))
    (d : Fin D) (c : Fin ic) (kh kw : Fin P) :
    den (SHlo.patchEmbedWeightGradB (N := N) (ic := ic) (H := H) (W := W) (P := P) (tk := tk)
            (D := D) xN img (.operand cotN dy))
        (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw))
      = ∑ n : Fin N, ∑ o : Fin ((tk + 1) * D),
          pdiv (fun v : Vec (D * ic * P * P) =>
                  patchEmbed_flat ic H W P tk D (Kernel4.unflatten v) bc cls pos
                    (batchSlice N (ic * H * W) img n))
            (Kernel4.flatten Wp)
            (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw)) o
            * batchSlice N ((tk + 1) * D) dy n o := by
  simp only [den, patchEmbedWeightGradFlat, Kernel4.flatten, Equiv.symm_apply_apply]
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
                  patchEmbed_flat ic H W P tk D Wc b' cls pos
                    (batchSlice N (ic * H * W) img n)) bc i o
            * batchSlice N ((tk + 1) * D) dy n o := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact vit_patchb_grad_bridge Wc bc cls pos
    (batchSlice N (ic * H * W) img n) (batchSlice N ((tk + 1) * D) dy n) i

/-- **Batched positional-embed GRADIENT denotes the certified `Σ_n` gradient** — the summed
    cotangent, since the positional table is added to every token and its Jacobian is the
    identity. ⚠⚠ `den_patchEmbedBiasGradB`'s neighbour warns that this batch sum is INVISIBLE at
    `N = 1`: a render that dropped it type-checks and emits the same bytes. -/
theorem posEmbedGradB_den {ic H W P tk D N : Nat} (cotN : String)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (tk + 1) D)
    (img : Vec (N * (ic * H * W))) (dy : Vec (N * ((tk + 1) * D))) (i : Fin ((tk + 1) * D)) :
    den (SHlo.posEmbedGradB (N := N) (tk := tk) (D := D) (.operand cotN dy)) i
      = ∑ n : Fin N, ∑ o : Fin ((tk + 1) * D),
          pdiv (fun p : Vec ((tk + 1) * D) =>
                  patchEmbed_flat ic H W P tk D Wc bc cls (Mat.unflatten p)
                    (batchSlice N (ic * H * W) img n))
            (Mat.flatten pos) i o
            * batchSlice N ((tk + 1) * D) dy n o := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact Proofs.ViTPoCG.posEmbedGrad_den (ic := ic) (H := H) (W := W) (P := P) (N := tk) (D := D)
    cotN Wc bc cls pos (batchSlice N (ic * H * W) img n) (batchSlice N ((tk + 1) * D) dy n) i

/-- **The CLS-token GRADIENT, and this is the one statement the per-example fold could not make.**
    The render slices row 0 of every example's embed cotangent (`clsSlice`, lifted by `batchOp`)
    and reduces the result as an `[N, D]` batch, so the op is `denseBiasGradB` at `N = B` and its
    `den` sums over the batch — which is what a shared `[192]` parameter's gradient IS.

    ⭐ `ViTPoCG.clsGrad_den` is the same theorem at `N = 1`, where the batch lift lived in
    `pretty B` outside the AST. The bytes are identical (`biasGrad`'s emitted reduce takes the `B`
    axis either way) and the functions are not; `den_rowDenseBiasGradB_at_one` is the general form
    of the trap.

    ⚠ Stated at the committed ViT-Tiny dims rather than generically, for `ViTPoCG.clsGrad_den`'s
    reason: the CLS operand's type is `Vec (N * (1 * D))`, which reduces to `Vec (N * D)` only at a
    literal `D`. -/
theorem clsGrad_denB {N : Nat} (cotN : String)
    (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192) (pos : Mat 197 192)
    (img : Vec (N * (3 * 224 * 224))) (dyEmbed : Vec (N * (197 * 192))) (i : Fin 192) :
    den (SHlo.denseBiasGradB (N := N) (c := 192)
            (.operand cotN (batchMap N (clsSliceFlat 196 192) dyEmbed))) i
      = ∑ n : Fin N, ∑ j : Fin (197 * 192),
          pdiv (fun cl : Vec 192 =>
                  patchEmbed_flat 3 224 224 16 196 192 Wc bc cl pos
                    (batchSlice N (3 * 224 * 224) img n)) cls i j
            * batchSlice N (197 * 192) dyEmbed n j := by
  simp only [den]
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
  simp only [den, Fin.sum_univ_one, batchSlice] at h
  exact h

-- ════════════════════════════════════════════════════════════════
-- § The classifier head — one CLS vector per example
-- ════════════════════════════════════════════════════════════════

/-- **Batched classifier weight GRADIENT denotes the certified `Σ_n` outer product.** -/
theorem headWGradB_den {N D nC : Nat} (aN cotN : String)
    (a : Vec (N * D)) (Wc : Mat D nC) (bc : Vec nC) (cot : Vec (N * nC))
    (i : Fin D) (j : Fin nC) :
    den (SHlo.weightGradB (N := N) (m := D) (n := nC) aN a (.operand cotN cot))
        (finProdFinEquiv (i, j))
      = ∑ n : Fin N, ∑ k : Fin nC,
          pdiv (fun v : Vec (D * nC) => dense (Mat.unflatten v) bc (batchSlice N D a n))
               (Mat.flatten Wc) (finProdFinEquiv (i, j)) k * batchSlice N nC cot n k := by
  simp only [den, Mat.flatten, Equiv.symm_apply_apply]
  apply Finset.sum_congr rfl
  intro n _
  exact dense_weight_grad_correct Wc bc (batchSlice N D a n) (batchSlice N nC cot n) i j

/-- **Batched classifier bias GRADIENT denotes the certified cotangent, PER EXAMPLE.**

    ⚠ `biasGradB` is the identity on its operand — the reduce over the batch is in the emitted
    text, outside the AST — so the statement this node supports is the per-example one at every
    `batchSlice n`, and it is the per-example `biasGrad` carve-out carried over rather than a new
    one. `StableHLO.lean`'s constructor comment records the same thing on the emitter side. -/
theorem headBGradB_den {N D nC : Nat} (cotN : String)
    (Wc : Mat D nC) (a : Vec D) (bc : Vec nC) (cot : Vec (N * nC)) (n : Fin N) (i : Fin nC) :
    batchSlice N nC (den (SHlo.biasGradB (N := N) (n := nC) (.operand cotN cot))) n i
      = ∑ j : Fin nC, pdiv (fun b' : Vec nC => dense Wc b' a) bc i j * batchSlice N nC cot n j := by
  simp only [den]
  exact dense_bias_grad_correct Wc bc a (batchSlice N nC cot n) i

end Proofs.ViTPoCGB
