import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFold
import LeanMlir.Proofs.Nets.ResNet.ResNet34Fold

/-! # T3 §1 fold for ResNet-34 at TRUE BATCH-NORM — the UN-FUSED gradient ops

`ResNet34Fold.lean` makes every parameter output of the per-example SGD train step
`den`-faithful. This is its batched peer, and one thing about it is different in kind.

⛔ **r34's batched render emits `*GradB`, not `*SgdB`.** Every batched ResNet-34 train step —
`resnet34_sgd_train_step`, the Adam family, `resnet34in_mom256` and its data-parallel peers —
emits the RAW gradient and hands it to an optimizer tail (`adamMNextF`/`adamVNextF`, heavy-ball,
plain SGD). The fused `θ − lr·∂Loss/∂θ` op only appears in renders whose optimizer is SGD-inline,
which EfficientNet's is and r34's batched one is not. Every `den = certified` lemma in the repo
before this file is stated at the fused form, so none of them applies here.

⭐ **That makes this tier better, not worse.** A statement about the gradient covers every
optimizer variant at once: `sgd`, `mom`, `momdp64`, `adam` and `adamdp128` all consume the same
`*GradB` node, so one lemma per op kind certifies the whole family. ⚠ The bf16 twins do NOT: a
bf16 render emits `*GradBBf16`, its own kind, folded in [`Foundation/Bf16GradNodes.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Foundation/Bf16GradNodes.lean). It is also the
form ConvNeXt's `psW` carve-out already had to take for a different reason (a hand-written SGD
wrap).

⭐ **And no new mathematics: the `*SgdB` peers were already proven, and the fusion is `rfl`.**
`StableHLO.lean`'s `*SgdB_eq_grad` family (`convWeightSgdB_eq_grad`, …) says each fused op IS
`θ − lr·` applied to the un-fused one, all by `rfl`, and its own docstring says it exists to
"unblock a batched `resnet34_adam_train_step` rendered from `Proofs/` — the blocker was the fusion,
never Adam." So the eight lemmas below are `EfficientNetFold.lean`'s proofs with the
`congr 1` / `congrArg (lr * ·)` wrapper peeling dropped: the same per-example VJP bridge under the
same `Σ_n`.

⚠ **Symmetric padding, not XLA-`SAME`.** The strided lemmas here are about `convStridedWeightGradB`
/ `convStridedBiasGradB`, whose `den` is `flatConvStride2_*`; B0's peers are about the
`convStridedXla*` ops and `flatConvStride2Xla_*`. The two op families have identical types and
identical emitted shapes, so nothing but the certificate distinguishes them — and r34 is the
PyTorch-origin net, so symmetric is the shipped phase.

## Honest residual (the boundary every fold carries)
* The cotangents are free variables `cot` — each lemma is `∀ cot`, so it holds at the actual
  backward-chain cotangent without naming it. Pinning each to the emitted residual-backward
  subgraph is the §1a tie, and is `ResNet34StepTieB.lean`.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ResNet34PoCB

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Stride-1 conv weight / bias (block convs)
-- ════════════════════════════════════════════════════════════════

/-- **Batched stride-1 conv weight GRADIENT denotes the certified `Σ_n` weight gradient.** The
    un-fused peer of `EfficientNetPoC.convWB_den`: same `Σ_n` of `conv_weight_grad_bridge`, with no
    `θ − lr·` wrapper because the batched r34 render hands this node to an optimizer tail. -/
theorem convWGradB_den {N ic oc h w kH kW : Nat}
    (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic * h * w)))
    (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                    (Tensor3.unflatten (batchSlice N (ic * h * w) x n))))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact conv_weight_grad_bridge b (Tensor3.unflatten (batchSlice N (ic * h * w) x n))
    (Kernel4.flatten W) (batchSlice N (oc * h * w) cot n) idx

/-- **Batched stride-1 conv bias GRADIENT denotes the certified `Σ_n` bias gradient.** -/
theorem convBGradB_den {N ic oc h w kH kW : Nat}
    (cotN : String) (W : Kernel4 oc ic kH kW) (x : Vec (N * (ic * h * w))) (b : Vec oc)
    (cot : Vec (N * (oc * h * w))) (o : Fin oc) :
    den (SHlo.convBiasGradB (h := h) (w := w) W x b (.operand cotN cot)) o
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc =>
                  Tensor3.flatten (conv2d W b'
                    (Tensor3.unflatten (batchSlice N (ic * h * w) x n))))
               b o j * batchSlice N (oc * h * w) cot n j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact conv_bias_grad_bridge W (Tensor3.unflatten (batchSlice N (ic * h * w) x n)) b
    (batchSlice N (oc * h * w) cot n) o

-- ════════════════════════════════════════════════════════════════
-- § Strided conv weight / bias (7x7/s2 stem, downsample W1, 1x1/s2 projection)
--   ⚠ SYMMETRIC padding — `flatConvStride2`, not the XLA-`SAME` `flatConvStride2Xla` B0 uses.
-- ════════════════════════════════════════════════════════════════

/-- **Batched strided conv weight GRADIENT denotes the certified `Σ_n` weight gradient.** Generic
    in the kernel size, so the one lemma certifies the 7x7 stem AND every 3x3 downsample `W1` AND
    every 1x1 projection `Wp`. -/
theorem convStridedWGradB_den {N ic oc h w kH kW : Nat}
    (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStridedWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  flatConvStride2 (Kernel4.unflatten v') b
                    (batchSlice N (ic * (2 * h) * (2 * w)) x n))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride2_weight_grad_has_vjp b (batchSlice N (ic * (2 * h) * (2 * w)) x n)).correct
    (Kernel4.flatten W) (batchSlice N (oc * h * w) cot n) idx

/-- **Batched strided conv bias GRADIENT denotes the certified `Σ_n` bias gradient.** -/
theorem convStridedBGradB_den {N ic oc h w kH kW : Nat}
    (cotN : String) (W : Kernel4 oc ic kH kW) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (b : Vec oc) (cot : Vec (N * (oc * h * w))) (o : Fin oc) :
    den (SHlo.convStridedBiasGradB (h := h) (w := w) W x b (.operand cotN cot)) o
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc =>
                  flatConvStride2 W b' (batchSlice N (ic * (2 * h) * (2 * w)) x n))
               b o j * batchSlice N (oc * h * w) cot n j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride2_bias_grad_has_vjp W (batchSlice N (ic * (2 * h) * (2 * w)) x n)).correct
    b (batchSlice N (oc * h * w) cot n) o

-- ════════════════════════════════════════════════════════════════
-- § BatchNorm gamma / beta — the `den` folds N into the reduction `m = N*(h*w)`
-- ════════════════════════════════════════════════════════════════

/-- **Batched BN γ GRADIENT denotes the certified per-channel γ gradient over the merged
    batch+spatial axis `m = N·(h·w)`.** γ enters affinely, so there is no batch coupling in the
    PARAM gradient and this is `bnPerChannel_grad_gamma_correct` at that width, through the
    network→oc-major reindex `bnchwFwd`. Generic in the free `β`. -/
theorem bnGammaGradB_den {N oc h w : Nat}
    (vN epsStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (N * (oc * (h * w)))) (cot : Vec (N * (oc * (h * w)))) (c : Fin oc) :
    den (SHlo.bnGammaGradB vN epsStr ε v (.operand cotN cot)) c
      = ∑ j : Fin (oc * (N * (h * w))),
          pdiv (fun γ' : Vec oc =>
                  bnPerChannelFlat oc (N * (h * w)) ε γ' β (bnchwFwd N oc h w v))
               γ c j * bnchwFwd N oc h w cot j := by
  simp only [den]
  exact bnPerChannel_grad_gamma_correct oc (N * (h * w)) ε γ β
    (bnchwFwd N oc h w v) (bnchwFwd N oc h w cot) c

/-- **Batched BN β GRADIENT denotes the certified per-channel β gradient** `Σ_{batch,spatial} cot`
    at `m = N·(h·w)`. Carries a free `v`/`γ` — β's gradient is the channel sum and depends on
    neither. -/
theorem bnBetaGradB_den {N oc h w : Nat}
    (cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (oc * (N * (h * w)))) (cot : Vec (N * (oc * (h * w)))) (c : Fin oc) :
    den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) (.operand cotN cot)) c
      = ∑ j : Fin (oc * (N * (h * w))),
          pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) ε γ β' v)
               β c j * bnchwFwd N oc h w cot j := by
  simp only [den]
  exact bnPerChannel_grad_beta_correct oc (N * (h * w)) ε γ β v (bnchwFwd N oc h w cot) c

-- ════════════════════════════════════════════════════════════════
-- § Head dense weight / bias
-- ════════════════════════════════════════════════════════════════

/-- **Batched dense weight GRADIENT denotes the certified `Σ_n` outer product.** -/
theorem denseWGradB_den {N a c : Nat}
    (xN cotN : String) (x : Vec (N * a)) (W : Mat a c) (b : Vec c) (cot : Vec (N * c))
    (i : Fin a) (j : Fin c) :
    den (SHlo.denseWeightGradB (c := c) xN x (.operand cotN cot)) (finProdFinEquiv (i, j))
      = ∑ n : Fin N, ∑ k : Fin c,
          pdiv (fun v : Vec (a * c) => dense (Mat.unflatten v) b (batchSlice N a x n))
               (Mat.flatten W) (finProdFinEquiv (i, j)) k * batchSlice N c cot n k := by
  simp only [den, Mat.flatten, Equiv.symm_apply_apply]
  apply Finset.sum_congr rfl
  intro n _
  exact dense_weight_grad_correct W b (batchSlice N a x n) (batchSlice N c cot n) i j

/-- **Batched dense bias GRADIENT denotes the certified `Σ_n` cotangent sum.** -/
theorem denseBGradB_den {N c : Nat}
    (cotN : String) (W : Mat c c) (x : Vec c) (b : Vec c) (cot : Vec (N * c)) (j : Fin c) :
    den (SHlo.denseBiasGradB (N := N) (.operand cotN cot)) j
      = ∑ n : Fin N, ∑ k : Fin c,
          pdiv (fun b' : Vec c => dense W b' x) b j k * batchSlice N c cot n k := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact dense_bias_grad_correct W b x (batchSlice N c cot n) j

end Proofs.ResNet34PoCB
