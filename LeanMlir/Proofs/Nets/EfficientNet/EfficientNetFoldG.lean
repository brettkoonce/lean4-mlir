import LeanMlir.Proofs.Nets.ResNet.ResNet34FoldB

/-! # T3 §1 fold for EfficientNet-B0 at the UN-FUSED gradient — the Adam artifact's op set

`EfficientNetFold.lean` makes every parameter output of the SGD-inline
`efficientnet_train_step.mlir` `den`-faithful: each `*SgdB` op denotes `θ − lr·(certified Jacobian ·
cotangent)`. Every OTHER train step this net renders — `efficientnet_adam_train_step`, the
`_rms_`/`_emarms_`/`_do_`/`_drop_`/`_dp_`/bf16 families, and the ImageNet
`efficientnetin_emarmsdp64dropdo_train_step` whose accuracy the book quotes — takes
`EfficientNetRender`'s `adam := true` branch, which emits the RAW gradient (`*GradB`) and hands it to
an optimizer tail. This file is the fold at those nodes. ⚠ The bf16 family's conv and depthwise
weight nodes are `*GradBBf16`, their own kind, folded in [`Foundation/Bf16GradNodes.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Foundation/Bf16GradNodes.lean).

⭐ **One lemma per op kind certifies every optimizer variant at once**, because RMSProp, AdamW,
plain SGD, EMA and the data-parallel peers all consume the same gradient node. The fusion itself is
`rfl` (`StableHLO.lean`'s `*SgdB_eq_grad` family), so nothing here is new mathematics — these are
`EfficientNetFold.lean`'s proofs with the `congr 1` / `congrArg (lr * ·)` wrapper peeling
dropped, exactly as `ResNet34FoldB.lean` did for r34.

## ⭐ Five of the eight op kinds were ALREADY proven, in r34's file

`ResNet34FoldB.lean` states its eight folds at op kinds, not at r34 — and five of them are
the same constructors this net emits, at the same generality: `convWeightGradB` (every 1×1 expand /
project / head conv), `bnGammaGradB` / `bnBetaGradB` (all 49 BatchNorm sites, and every conv bias,
which the render folds onto the following BN's β), and `denseWeightGradB` (the SE squeeze/excite
denses and the classifier). So B0's package is **three** genuinely new lemmas, not eight; the five
below are one-line delegations that say so out loud, so this file is still the complete op table for
the artifact. (`denseBiasGradB` is restated rather than delegated only to widen r34's square
`Mat c c` witness to the SE's rectangular `Mat a c` — the gradient is `W`-free either way.)

## The three that are new, and why

* **`convStridedXlaWeightGradB`** — the 3×3/s2 stem. ⚠ This is the XLA-`SAME` op
  (`flatConvStride2Xla`), where r34's peer is the symmetric `convStridedWeightGradB`
  (`flatConvStride2`). The two have identical types and emit identical shapes; only the certificate
  tells them apart, and B0 is the TF-origin net.
* **`depthwiseWeightGradB`** / **`depthwiseStridedWeightGradB`** — the MBConv depthwise kernels,
  3×3 and 5×5, which ResNet-34 has no instance of. ⚠ The strided one is the **symmetric** op, which
  is what `EfficientNetRender` emits on the forward side too (`.depthwiseStrided`, not
  `.depthwiseStridedXla`); the XLA phase is B0's stem only.

## Honest residual
* Every lemma is `∀ cot`, so it holds at the actual backward-chain cotangent without naming it.
  Pinning each to the emitted backward subgraph is the §1a tie (`EfficientNetStepTie.lean`), and
  re-pointing that capstone at these nodes needs the smoothed-target loss cotangent, which is
  scoped with r34's batched tie.
* The all-reduce in the `*dp*` artifacts is emitted text outside the AST, so these lemmas are about
  the per-replica gradient node.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.EnetPoCG

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Shared with ResNet-34 — same constructor, same generality, already proven
-- ════════════════════════════════════════════════════════════════

/-- **Batched 1×1-conv weight GRADIENT denotes the certified `Σ_n` weight gradient.** Every expand,
    project and head conv. Identical to r34's block convs as an op: `ResNet34PoCB.convWGradB_den`. -/
theorem convWGradB_den {N ic oc h w kH kW : Nat}
    (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic * h * w)))
    (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                    (Tensor3.unflatten (batchSlice N (ic * h * w) x n))))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j :=
  Proofs.ResNet34PoCB.convWGradB_den xN cotN b x W cot idx

/-- **Batched BN γ GRADIENT denotes the certified per-channel γ gradient** at the merged
    batch+spatial width `m = N·(h·w)`. All 49 sites. `ResNet34PoCB.bnGammaGradB_den`. -/
theorem bnGammaGradB_den {N oc h w : Nat}
    (vN epsStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (N * (oc * (h * w)))) (cot : Vec (N * (oc * (h * w)))) (c : Fin oc) :
    den (SHlo.bnGammaGradB vN epsStr ε v (.operand cotN cot)) c
      = ∑ j : Fin (oc * (N * (h * w))),
          pdiv (fun γ' : Vec oc =>
                  bnPerChannelFlat oc (N * (h * w)) ε γ' β (bnchwFwd N oc h w v))
               γ c j * bnchwFwd N oc h w cot j :=
  Proofs.ResNet34PoCB.bnGammaGradB_den vN epsStr cotN ε γ β v cot c

/-- **Batched BN β GRADIENT denotes the certified per-channel β gradient** `Σ_{batch,spatial} cot`.
    Used at all 49 BN βs AND at every conv bias, which the render folds onto the following BN.
    `ResNet34PoCB.bnBetaGradB_den`. -/
theorem bnBetaGradB_den {N oc h w : Nat}
    (cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (oc * (N * (h * w)))) (cot : Vec (N * (oc * (h * w)))) (c : Fin oc) :
    den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) (.operand cotN cot)) c
      = ∑ j : Fin (oc * (N * (h * w))),
          pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) ε γ β' v)
               β c j * bnchwFwd N oc h w cot j :=
  Proofs.ResNet34PoCB.bnBetaGradB_den cotN ε γ β v cot c

/-- **Batched dense weight GRADIENT denotes the certified `Σ_n` outer product.** The SE
    squeeze (`c → r`) and excite (`r → c`) denses and the classifier head.
    `ResNet34PoCB.denseWGradB_den`. -/
theorem denseWGradB_den {N a c : Nat}
    (xN cotN : String) (x : Vec (N * a)) (W : Mat a c) (b : Vec c) (cot : Vec (N * c))
    (i : Fin a) (j : Fin c) :
    den (SHlo.denseWeightGradB (c := c) xN x (.operand cotN cot)) (finProdFinEquiv (i, j))
      = ∑ n : Fin N, ∑ k : Fin c,
          pdiv (fun v : Vec (a * c) => dense (Mat.unflatten v) b (batchSlice N a x n))
               (Mat.flatten W) (finProdFinEquiv (i, j)) k * batchSlice N c cot n k :=
  Proofs.ResNet34PoCB.denseWGradB_den xN cotN x W b cot i j

/-- **Batched dense bias GRADIENT denotes the certified `Σ_n` cotangent sum.** r34's peer at a
    RECTANGULAR witness `Mat a c`, which the SE's `c → r` squeeze needs; the gradient is the
    channel sum and depends on neither `W` nor `x`, so the widening is free. -/
theorem denseBGradB_den {N a c : Nat}
    (cotN : String) (W : Mat a c) (x : Vec a) (b : Vec c) (cot : Vec (N * c)) (j : Fin c) :
    den (SHlo.denseBiasGradB (N := N) (.operand cotN cot)) j
      = ∑ n : Fin N, ∑ k : Fin c,
          pdiv (fun b' : Vec c => dense W b' x) b j k * batchSlice N c cot n k := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact dense_bias_grad_correct W b x (batchSlice N c cot n) j

-- ════════════════════════════════════════════════════════════════
-- § New: the XLA-`SAME` strided stem
--   ⚠ `flatConvStride2Xla`, NOT r34's symmetric `flatConvStride2`. Identical types.
-- ════════════════════════════════════════════════════════════════

/-- **Batched XLA-`SAME` strided conv weight GRADIENT denotes the certified `Σ_n` weight
    gradient.** B0's 3×3/s2 stem, the net's one XLA-phase site. `Σ_n` of
    `flatConvStride2Xla_weight_grad_has_vjp.correct` — the odd-phase weight VJP, so the certified
    gradient is the gradient of the net that ships. -/
theorem convStridedXlaWGradB_den {N ic oc h w kH kW : Nat}
    (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStridedXlaWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  flatConvStride2Xla (Kernel4.unflatten v') b
                    (batchSlice N (ic * (2 * h) * (2 * w)) x n))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride2Xla_weight_grad_has_vjp b
    (batchSlice N (ic * (2 * h) * (2 * w)) x n)).correct
    (Kernel4.flatten W) (batchSlice N (oc * h * w) cot n) idx

-- ════════════════════════════════════════════════════════════════
-- § New: the MBConv depthwise kernels (3×3 and 5×5), stride 1 and stride 2
--   ⚠ SYMMETRIC padding at the strided sites — the render's forward is `.depthwiseStrided` too.
-- ════════════════════════════════════════════════════════════════

/-- **Batched stride-1 depthwise weight GRADIENT denotes the certified `Σ_n` weight gradient.**
    `Σ_n` of the flattened `depthwise_weight_grad_has_vjp3.correct`. Generic in the kernel size, so
    the one lemma covers every 3×3 and every 5×5 depthwise. -/
theorem depthwiseWGradB_den {N c h w kH kW : Nat}
    (xN cotN : String) (b : Vec c) (x : Vec (N * (c * h * w)))
    (W : DepthwiseKernel c kH kW) (cot : Vec (N * (c * h * w))) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b
                    (Tensor3.unflatten (batchSlice N (c * h * w) x n))))
               (Tensor3.flatten W) idx j * batchSlice N (c * h * w) cot n j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  rw [← (hasVJP3_to_hasVJP (depthwise_weight_grad_has_vjp3 b
      (Tensor3.unflatten (batchSlice N (c * h * w) x n)))).correct
      (Tensor3.flatten W) (batchSlice N (c * h * w) cot n) idx]
  simp only [hasVJP3_to_hasVJP, Tensor3.flatten, Tensor3.unflatten_flatten]

/-- **Batched strided depthwise weight GRADIENT denotes the certified `Σ_n` weight gradient.** The
    strided VJP is already flat, so this is `Σ_n` of
    `depthwiseStride2_weight_grad_has_vjp.correct`. -/
theorem depthwiseStridedWGradB_den {N c h w kH kW : Nat}
    (xN cotN : String) (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w))))
    (W : DepthwiseKernel c kH kW) (cot : Vec (N * (c * h * w))) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseStridedWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  depthwiseStride2Flat (Tensor3.unflatten v') b
                    (batchSlice N (c * (2 * h) * (2 * w)) x n))
               (Tensor3.flatten W) idx j * batchSlice N (c * h * w) cot n j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact (depthwiseStride2_weight_grad_has_vjp b
    (batchSlice N (c * (2 * h) * (2 * w)) x n)).correct
    (Tensor3.flatten W) (batchSlice N (c * h * w) cot n) idx

end Proofs.EnetPoCG
