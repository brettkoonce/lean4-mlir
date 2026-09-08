import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFaithfulPoCG
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FaithfulPoCPaperG
import LeanMlir.Proofs.Nets.ViT.ViTFaithfulPoCGB

/-! # T3 §1 fold for ConvNeXt-T at the BATCHED index — the op set every ImageNet artifact renders

`ConvNeXtFaithfulPoCG.lean` folds the fourteen gradient nodes of the PER-EXAMPLE traversal
(`ConvNeXtRender.convNextBackAll` at `adam := true`). This is its batched peer, at
`ConvNeXtRenderB.convNextBackAllB`'s constructors — and unlike ViT's (`ViTFaithfulPoCGB.lean`), it
was owed BEFORE any renderer swap: every `convnextin_*` train step, every `*drop*` variant and the
ConvNeXt-S/B artifacts have rendered from the batched chain since they existed, so the artifact
behind this net's quoted ImageNet accuracy (`convnextin_adamdpwxclipdrop`) had a fold at the
per-example constructors that no committed byte of it is `pretty` of. 4b's "one lemma per op kind
certifies every optimizer tail" was, for ConvNeXt, a statement about the Imagenette pair only.

⭐ **The bytes are the same on the forward and differ on 78 backward lines.** The batched
`convBackBatched` emits the conv input-VJP's `transpose`/`reverse` in the other order from the
per-example `convBack` — commuting ops on disjoint axes, one kernel — and `tests/TestConvNeXtFwdBTie.lean`
allows exactly that pair and nothing else. 4c leg 3 moves the drop-free writers onto this chain;
this file lands first, per leg 1's ordering rule, so that no committed artifact is ever `pretty` of
an AST without a fold.

## The op table of `convnext_adam_train_step.mlir` and every `convnextin_*` train step

| emitted node | lemma | per-example peer it batches |
|---|---|---|
| `layerScaleChGammaGradB` (18 block γ) | `layerScaleChGammaGradB_den` | `CnxPoCG.layerScaleChGammaGrad_den` |
| `convWeightGradB` / `convBiasGradB` (18 expand + 18 project 1×1, + the stem bias) | `convWGradB_den` / `convBGradB_den` | `ResNet34PoCB`'s, verbatim |
| `depthwiseWeightGradB` / `depthwiseBiasGradB` (18 × 7×7) | `depthwiseWGradB_den` / `depthwiseBGradB_den` | `EnetPoCG` / `Mnv2PaperPoCG` |
| `convStridedWeightGradB` / `convStridedBiasGradB` (3 × 2×2/s2 downsample) | `convStridedWGradB_den` / `convStridedBGradB_den` | `ResNet34PoCB`'s, verbatim |
| `convStride4WeightGradB` (patchify stem) | `psWGradB_den` | `CnxPoCG.psWGrad_den` |
| `veclnGammaGradB` / `rowDenseBiasGradB` at `R = h·w` (22 spatial LN sites) | `chanLnGammaGradB_den` / `chanLnBetaGradB_den` | `CnxPoCG.chanLnGammaGrad_den` / `chanLnBetaGrad_den` |
| `veclnGammaGradB` / `rowDenseBiasGradB` at `R = 1` (the head LN, after GAP) | `headLnGammaGradB_den` / `headLnBetaGradB_den` | `ViTPoCGB`'s two-level LN lemmas |
| `weightGradB` / `biasGradB` (the classifier) | `headWGradB_den` / `headBGradB_den` | `ViTPoCGB.headWGradB_den` / `headBGradB_den` |
| `convWeightGradBBf16` / `depthwiseWeightGradBBf16` / `convStridedWeightGradBBf16` / `convStride4WeightGradBBf16` (the bf16 artifacts) | `Bf16PoC.convWGradBBf16_den` and its siblings, `Foundation/Bf16GradNodes.lean` | none — a bf16 node is its own op kind |

⭐ **No new mathematics.** Every proof is `Finset.sum_congr rfl` over the batch and then the
per-example bridge at `batchSlice n` — `ResNet34FaithfulPoCB.denseWGradB_den`'s shape — because
each batched `den` arm is literally the per-example one under a batch sum. The channel-LN sites
add one step: the batched render hands the LN ops `batchMap N (chanLNRows c h w)` of the saved
input and of the cotangent (the `[h·w, c]` transposed views, lifted per example), and
`batchSlice_batchMap` peels the lift so `ConvNeXtChannelLN`'s permutation argument applies at each
slice.

⭐ **The bf16 artifacts (`convnextin_adamwxclipdropbf16`, the S/B twins) emit `*GradBBf16`
constructors, not these nodes**: their `den` rounds the operands and the result once, outside the
batch sum. Those are their own op kinds, folded once for every net in
`Foundation/Bf16GradNodes.lean` (first stated in this file, 2026-09-07).

⭐ **One lemma per op kind certifies every optimizer tail at once** — AdamW, the `wx`/`clip`
variants, the EMA shadow, drop-path and the data-parallel twins all consume the same `*GradB`
node, and `convnextin_adamdpwxclipdrop`, whose accuracy the book quotes, is one of them.

## Honest residual
* ⚠ **`biasGradB` is the IDENTITY on its operand** and the classifier bias's batch reduce is in
  the emitted text, outside the AST — so `headBGradB_den` is stated PER EXAMPLE at `batchSlice n`,
  the per-example `biasGrad` carve-out carried over unchanged (as in `ViTFaithfulPoCGB`).
* Every lemma is `∀ cot`. Pinning each to the emitted backward subgraph is the §1a tie; ConvNeXt's
  capstone (`ConvNeXtTiePoC.lean`, 182 params) is at the per-example SGD-inline
  `convnext_train_step.mlir`, which stays on the per-example chain (the batched traversal has no
  fused-SGD arm). Re-pointing it at these nodes with `SmoothedLossCot` is 4b's ConvNeXt capstone,
  which this file is the prerequisite for.
* `convnextin_adamdp*` is four replicas: the all-reduce is emitted text outside the AST, so these
  lemmas are about the per-replica gradient node (4d).
* ⛔ SYMMETRIC padding at the three 2×2/s2 downsamples and the 4×4/s4 stem (`flatConvStride2`,
  `flatConvStride4`); ConvNeXt is PyTorch-origin and has no XLA-`SAME` site.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CnxPoCGB

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-channel layer scale — the one op kind unique to this net
-- ════════════════════════════════════════════════════════════════

/-- **Batched per-channel layer-scale γ GRADIENT denotes the certified `Σ_n` gradient.** The
    emitted reduce contracts batch and spatial in one op; `den` reads it as the batch sum of the
    per-example `dγ_c = Σ_{k : chanIdx k = c} x_k·dy_k`. All 18 blocks. -/
theorem layerScaleChGammaGradB_den {N c h w : Nat} (xN cotN : String)
    (x : Vec (N * (c * h * w))) (γ : Vec c) (dy : Vec (N * (c * h * w))) (cc : Fin c) :
    den (SHlo.layerScaleChGammaGradB (N := N) (c := c) (h := h) (w := w) xN x (.operand cotN dy)) cc
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c =>
                  layerScale (fun k => γ' (chanIdx c h w k)) (batchSlice N (c * h * w) x n))
               γ cc j * batchSlice N (c * h * w) dy n j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  have h := Proofs.CnxPoCG.layerScaleChGammaGrad_den (h := h) (w := w) xN cotN
    (batchSlice N (c * h * w) x n) γ (batchSlice N (c * h * w) dy n) cc
  simp only [den] at h
  exact h

-- ════════════════════════════════════════════════════════════════
-- § The 1×1 convolutions (expand / project) and the stem bias — ResNet-34's batched lemmas
-- ════════════════════════════════════════════════════════════════

/-- **Batched conv weight GRADIENT denotes the certified `Σ_n` weight gradient.** Kernel-generic:
    every 1×1 expand and project. `ResNet34PoCB.convWGradB_den` verbatim. -/
theorem convWGradB_den {N ic oc h w kH kW : Nat} (xN cotN : String)
    (b : Vec oc) (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                    (Tensor3.unflatten (batchSlice N (ic * h * w) x n))))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j :=
  Proofs.ResNet34PoCB.convWGradB_den xN cotN b x W cot idx

/-- **Batched conv bias GRADIENT denotes the certified `Σ_n` bias gradient.** The expand/project
    biases and the patchify stem's `psb` (at 4×4). -/
theorem convBGradB_den {N ic oc h w kH kW : Nat} (cotN : String)
    (W : Kernel4 oc ic kH kW) (x : Vec (N * (ic * h * w))) (b : Vec oc)
    (cot : Vec (N * (oc * h * w))) (o : Fin oc) :
    den (SHlo.convBiasGradB (h := h) (w := w) W x b (.operand cotN cot)) o
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc =>
                  Tensor3.flatten (conv2d W b'
                    (Tensor3.unflatten (batchSlice N (ic * h * w) x n))))
               b o j * batchSlice N (oc * h * w) cot n j :=
  Proofs.ResNet34PoCB.convBGradB_den cotN W x b cot o

-- ════════════════════════════════════════════════════════════════
-- § The 7×7 depthwise
-- ════════════════════════════════════════════════════════════════

/-- **Batched depthwise weight GRADIENT denotes the certified `Σ_n` weight gradient.** All 18
    blocks; the kernel size is a variable, so 7×7 is an instance. `EnetPoCG.depthwiseWGradB_den`. -/
theorem depthwiseWGradB_den {N c h w kH kW : Nat} (xN cotN : String)
    (b : Vec c) (x : Vec (N * (c * h * w))) (W : DepthwiseKernel c kH kW)
    (cot : Vec (N * (c * h * w))) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b
                    (Tensor3.unflatten (batchSlice N (c * h * w) x n))))
               (Tensor3.flatten W) idx j * batchSlice N (c * h * w) cot n j :=
  Proofs.EnetPoCG.depthwiseWGradB_den xN cotN b x W cot idx

/-- **Batched depthwise bias GRADIENT denotes the certified `Σ_n` bias gradient.** ConvNeXt's
    depthwises carry a bias (no BatchNorm follows them), which is `Mnv2PaperPoCG`'s lemma. -/
theorem depthwiseBGradB_den {N c h w kH kW : Nat} (cotN : String)
    (W : DepthwiseKernel c kH kW) (x : Vec (N * (c * h * w))) (b : Vec c)
    (cot : Vec (N * (c * h * w))) (o : Fin c) :
    den (SHlo.depthwiseBiasGradB W x b (.operand cotN cot)) o
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c =>
                  Tensor3.flatten (depthwiseConv2d W b'
                    (Tensor3.unflatten (batchSlice N (c * h * w) x n))))
               b o j * batchSlice N (c * h * w) cot n j :=
  Proofs.Mnv2PaperPoCG.depthwiseBGradB_den cotN W x b cot o

-- ════════════════════════════════════════════════════════════════
-- § The 2×2/s2 downsamples and the 4×4/s4 patchify stem
--   ⚠ SYMMETRIC padding — `flatConvStride2` / `flatConvStride4`, not the XLA-`SAME` twins.
-- ════════════════════════════════════════════════════════════════

/-- **Batched strided conv weight GRADIENT denotes the certified `Σ_n` weight gradient.** The
    three 2×2/s2 downsamples; kernel-generic, `ResNet34PoCB.convStridedWGradB_den` verbatim. -/
theorem convStridedWGradB_den {N ic oc h w kH kW : Nat} (xN cotN : String)
    (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStridedWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  flatConvStride2 (Kernel4.unflatten v') b
                    (batchSlice N (ic * (2 * h) * (2 * w)) x n))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j :=
  Proofs.ResNet34PoCB.convStridedWGradB_den xN cotN b x W cot idx

/-- **Batched strided conv bias GRADIENT denotes the certified `Σ_n` bias gradient.** -/
theorem convStridedBGradB_den {N ic oc h w kH kW : Nat} (cotN : String)
    (W : Kernel4 oc ic kH kW) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (b : Vec oc)
    (cot : Vec (N * (oc * h * w))) (o : Fin oc) :
    den (SHlo.convStridedBiasGradB (h := h) (w := w) W x b (.operand cotN cot)) o
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc =>
                  flatConvStride2 W b' (batchSlice N (ic * (2 * h) * (2 * w)) x n))
               b o j * batchSlice N (oc * h * w) cot n j :=
  Proofs.ResNet34PoCB.convStridedBGradB_den cotN W x b cot o

/-- **Batched patchify-stem weight GRADIENT denotes the certified `Σ_n` weight gradient.**
    ⚠ The emitted convolution contracts the batch axis itself (the transpose trick), so the outer
    sum is inside one op rather than across `N` of them — same as the strided ops. -/
theorem psWGradB_den {N ic oc h w kH kW : Nat} (xN cotN : String)
    (b : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStride4WeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  flatConvStride4 (Kernel4.unflatten v') b
                    (batchSlice N (ic * (2 * (2 * h)) * (2 * (2 * w))) x n))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride4_weight_grad_has_vjp b
    (batchSlice N (ic * (2 * (2 * h)) * (2 * (2 * w))) x n)).correct
    (Kernel4.flatten W) (batchSlice N (oc * h * w) cot n) idx

-- ════════════════════════════════════════════════════════════════
-- § The 22 spatial LayerNorm sites — the CHANNEL-LN form, batched
--   The render hands the two-level ops `batchMap N (chanLNRows c h w)` of the saved LN input and
--   of the cotangent — the per-example `[h·w, c]` view, lifted — and the certified Jacobian is
--   `chanLNTensor3`'s in the `c·h·w` layout at each `batchSlice n`.
-- ════════════════════════════════════════════════════════════════

/-- **Batched channel-LN γ GRADIENT denotes the certified `Σ_n` γ gradient.** All 22 spatial sites
    (1 stem + 18 block + 3 downsample). Two levels: the outer sum is the batch, the inner the
    `h·w` rows within one example. -/
theorem chanLnGammaGradB_den {N c h w : Nat} (xN epsStr cotN : String)
    (ε : ℝ) (β : Vec c) (x : Vec (N * (c * h * w))) (γ : Vec c) (cot : Vec (N * (c * h * w)))
    (k : Fin c) :
    den (SHlo.veclnGammaGradB (N := N) (R := h * w) (D := c) xN epsStr ε
          (batchMap N (chanLNRows c h w) x)
          (.operand cotN (batchMap N (chanLNRows c h w) cot))) k
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c => chanLNTensor3 c h w ε γ' β (batchSlice N (c * h * w) x n)) γ k j
            * batchSlice N (c * h * w) cot n j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  rw [batchSlice_batchMap, batchSlice_batchMap]
  have h := Proofs.CnxPoCG.chanLnGammaGrad_den xN epsStr cotN ε β
    (batchSlice N (c * h * w) x n) γ (batchSlice N (c * h * w) cot n) k
  simp only [den] at h
  exact h

/-- **Batched channel-LN β GRADIENT denotes the certified `Σ_n` β gradient.** The β gradient is
    the plain two-level row reduce, so the render uses the same `rowDenseBiasGradB` op ViT's LN β
    does. -/
theorem chanLnBetaGradB_den {N c h w : Nat} (cotN : String)
    (ε : ℝ) (γ : Vec c) (x : Vec (N * (c * h * w))) (β : Vec c) (cot : Vec (N * (c * h * w)))
    (k : Fin c) :
    den (SHlo.rowDenseBiasGradB (N := N) (R := h * w) (c := c)
          (.operand cotN (batchMap N (chanLNRows c h w) cot))) k
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun β' : Vec c => chanLNTensor3 c h w ε γ β' (batchSlice N (c * h * w) x n)) β k j
            * batchSlice N (c * h * w) cot n j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  rw [batchSlice_batchMap]
  have h := Proofs.CnxPoCG.chanLnBetaGrad_den cotN ε γ (batchSlice N (c * h * w) x n) β
    (batchSlice N (c * h * w) cot n) k
  simp only [den] at h
  exact h

-- ════════════════════════════════════════════════════════════════
-- § The head — the post-GAP LayerNorm at one row per example, and the classifier
-- ════════════════════════════════════════════════════════════════

/-- **Batched head-LN γ GRADIENT denotes the certified `Σ_n` γ gradient.** ViT's two-level vector
    LayerNorm lemma; the head LN runs after GAP, so the render instantiates it at `R = 1`. -/
theorem headLnGammaGradB_den {N R D : Nat} (xN epsStr cotN : String)
    (ε : ℝ) (βv : Vec D) (x : Vec (N * (R * D))) (γ : Vec D) (dy : Vec (N * (R * D)))
    (k : Fin D) :
    den (SHlo.veclnGammaGradB (N := N) (R := R) (D := D) xN epsStr ε x (.operand cotN dy)) k
      = ∑ n : Fin N, ∑ o : Fin (R * D),
          pdiv (fun gv : Vec D =>
                  Mat.flatten (fun r =>
                    layerNormVec D ε gv βv (Mat.unflatten (batchSlice N (R * D) x n) r))) γ k o
            * batchSlice N (R * D) dy n o :=
  Proofs.ViTPoCGB.veclnGammaGradB_den xN epsStr cotN ε βv x γ dy k

/-- **Batched head-LN β GRADIENT denotes the certified `Σ_n` β gradient.** -/
theorem headLnBetaGradB_den {N R D : Nat} (cotN : String)
    (ε : ℝ) (γv : Vec D) (X : Fin N → Mat R D) (β : Vec D) (dy : Vec (N * (R * D))) (i : Fin D) :
    den (SHlo.rowDenseBiasGradB (N := N) (R := R) (c := D) (.operand cotN dy)) i
      = ∑ n : Fin N, ∑ o : Fin (R * D),
          pdiv (fun bv : Vec D => Mat.flatten (fun r => layerNormVec D ε γv bv (X n r))) β i o
            * batchSlice N (R * D) dy n o :=
  Proofs.ViTPoCGB.rowDenseBiasGradB_den_lnbeta cotN ε γv X β dy i

/-- **Batched classifier weight GRADIENT denotes the certified `Σ_n` outer product.** -/
theorem headWGradB_den {N D nC : Nat} (aN cotN : String)
    (a : Vec (N * D)) (Wc : Mat D nC) (bc : Vec nC) (cot : Vec (N * nC))
    (i : Fin D) (j : Fin nC) :
    den (SHlo.weightGradB (N := N) (m := D) (n := nC) aN a (.operand cotN cot))
        (finProdFinEquiv (i, j))
      = ∑ n : Fin N, ∑ k : Fin nC,
          pdiv (fun v : Vec (D * nC) => dense (Mat.unflatten v) bc (batchSlice N D a n))
               (Mat.flatten Wc) (finProdFinEquiv (i, j)) k * batchSlice N nC cot n k :=
  Proofs.ViTPoCGB.headWGradB_den aN cotN a Wc bc cot i j

/-- **Batched classifier bias GRADIENT denotes the certified cotangent, PER EXAMPLE.** `biasGradB`
    is the identity on its operand — the batch reduce is emitted text outside the AST — so the
    statement is at every `batchSlice n`, the per-example carve-out carried over. -/
theorem headBGradB_den {N D nC : Nat} (cotN : String)
    (Wc : Mat D nC) (a : Vec D) (bc : Vec nC) (cot : Vec (N * nC)) (n : Fin N) (i : Fin nC) :
    batchSlice N nC (den (SHlo.biasGradB (N := N) (n := nC) (.operand cotN cot))) n i
      = ∑ j : Fin nC, pdiv (fun b' : Vec nC => dense Wc b' a) bc i j * batchSlice N nC cot n j :=
  Proofs.ViTPoCGB.headBGradB_den cotN Wc a bc cot n i

end Proofs.CnxPoCGB
