import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Nets.Small.CnnTrainStep
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetClose

/-! # PoC: the full-16 (262-param) EfficientNet-B0 train step, proof-tied (the §1 fold, den)

The EfficientNet peer of `MobileNetV2FaithfulPoCPaper` (the mnv2 §1 fold), for the batched 262-param
train step `efficientnetTrainStepFaithfulV` (`EfficientNetRender.lean`). Every emitted param-SGD op
`den`otes the certified loss-descent step — `θ − lr·(certified Jacobian · cotangent)`.

**The batched wrinkle vs mnv2.** EfficientNet trains at the batched index `N·(c·h·w)` with the
fused-batch param-SGD ops (`convWeightSgdB`/`denseWeightSgdB`/`bn{Gamma,Beta}SgdB`/…), whose `den`
carries a **batch sum `Σ_n`** over the per-example gradients (the shared-weight batched gradient).
So unlike mnv2 (per-example, one `∑_j pdiv·cot`), each fold here is "the per-example bridge applied
inside `Σ_n`" — the **batch-sum bridge**. For the linear families (conv/dense weight, bias) this is a
`Finset.sum_congr` of the per-example `.correct`; for BN γ/β the `den` already folds `N` into the
per-channel reduction count `m = N·(h·w)`, so it is the cert's exact LHS (delegation).

`Σ_n` of the per-example VJP `.correct`: `conv_weight_grad_bridge` / `conv_bias_grad_bridge` (1×1
expand/project/head + the strided stem via `flatConvStride2Xla`), the dense outer-product
(`denseWeightSgdB`/`denseBiasSgdB`, SE squeeze/excite + head dense), and the depthwise grads
(`mnv2_render_depthwise*` / `enet_render_dw5*`). -/

open Proofs Proofs.StableHLO

namespace Proofs.EnetPoC

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § 1×1 conv weight / bias (expand / project / head) — Σ_n of the per-example conv VJP bridge
-- ════════════════════════════════════════════════════════════════

/-- **Batched 1×1-conv weight op denotes the certified Σ_n batched weight gradient.** Each emitted
    `convWeightSgdB` (expand/project/head) denotes `flatten W − lr·Σ_n (∂conv2d/∂W · cotₙ)` at the
    per-example slice `n`, via `Σ_n` of `conv_weight_grad_bridge`. Generic in dims + cotangent. -/
theorem convWB_den {N ic oc h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec oc) (x : Vec (N * (ic * h * w)))
    (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w))) (lr : ℝ) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convWeightSgdB xN wN lrStr b x W lr (.operand cotN cot)) idx
      = Kernel4.flatten W idx - lr * ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                    (Tensor3.unflatten (batchSlice N (ic * h * w) x n))))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j := by
  simp only [den]
  congr 1
  apply congrArg (lr * ·)
  apply Finset.sum_congr rfl
  intro n _
  exact conv_weight_grad_bridge b (Tensor3.unflatten (batchSlice N (ic * h * w) x n))
    (Kernel4.flatten W) (batchSlice N (oc * h * w) cot n) idx

/-- **Batched strided-stem 3×3 conv weight op denotes the certified Σ_n batched weight gradient.**
    `Σ_n` of `flatConvStride2Xla_weight_grad_has_vjp.correct`. The op is the XLA-`SAME`
    `convStridedXlaWeightSgdB` the render emits at the stem (`EfficientNetRender.lean`), whose
    weight-grad correlation pad is shifted one position; its `den` is the odd-phase weight VJP,
    so the certified gradient here is the gradient of the net that ships. -/
theorem convStridedWB_den {N ic oc h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec oc) (x : Vec (N * (ic * (2*h) * (2*w))))
    (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w))) (lr : ℝ) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStridedXlaWeightSgdB xN wN lrStr b x W lr (.operand cotN cot)) idx
      = Kernel4.flatten W idx - lr * ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  flatConvStride2Xla (Kernel4.unflatten v') b (batchSlice N (ic * (2*h) * (2*w)) x n))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j := by
  simp only [den]
  congr 1
  apply congrArg (lr * ·)
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride2Xla_weight_grad_has_vjp b (batchSlice N (ic * (2*h) * (2*w)) x n)).correct
    (Kernel4.flatten W) (batchSlice N (oc * h * w) cot n) idx

-- ════════════════════════════════════════════════════════════════
-- § Dense weight / bias (SE squeeze+excite, head classifier) — Σ_n of the per-example dense VJP
-- ════════════════════════════════════════════════════════════════

/-- **Batched dense weight op denotes the certified Σ_n batched weight gradient.** `Σ_n` of the
    dense outer-product `.correct` (`dense_weight_grad_correct`). Covers the SE squeeze/excite denses
    (`W₁ : c→r`, `W₂ : r→c`) and the head classifier. Generic in `b` (the grad is `b`-independent). -/
theorem denseWB_den {N a c : Nat}
    (xN wN lrStr cotN : String) (x : Vec (N * a)) (W : Mat a c) (b : Vec c) (cot : Vec (N * c))
    (lr : ℝ) (i : Fin a) (j : Fin c) :
    den (SHlo.denseWeightSgdB xN wN lrStr x W lr (.operand cotN cot)) (finProdFinEquiv (i, j))
      = W i j - lr * ∑ n : Fin N, ∑ k : Fin c,
          pdiv (fun v : Vec (a * c) => dense (Mat.unflatten v) b (batchSlice N a x n))
               (Mat.flatten W) (finProdFinEquiv (i, j)) k * batchSlice N c cot n k := by
  simp only [den, Mat.flatten, Equiv.symm_apply_apply]
  congr 1
  apply congrArg (lr * ·)
  apply Finset.sum_congr rfl
  intro n _
  exact dense_weight_grad_correct W b (batchSlice N a x n) (batchSlice N c cot n) i j

/-- **Batched dense bias op denotes the certified Σ_n batched bias gradient** (`Σ_{n} cotₙ` per
    output) — `Σ_n` of `dense_bias_grad_correct`. Covers the SE `b₁`/`b₂` and the head bias. -/
theorem denseBB_den {N c : Nat}
    (bN lrStr cotN : String) (W : Mat c c) (x : Vec c) (b : Vec c) (cot : Vec (N * c))
    (lr : ℝ) (j : Fin c) :
    den (SHlo.denseBiasSgdB bN lrStr b lr (.operand cotN cot)) j
      = b j - lr * ∑ n : Fin N, ∑ k : Fin c,
          pdiv (fun b' : Vec c => dense W b' x) b j k * batchSlice N c cot n k := by
  simp only [den]
  congr 1
  apply congrArg (lr * ·)
  apply Finset.sum_congr rfl
  intro n _
  exact dense_bias_grad_correct W b x (batchSlice N c cot n) j

-- ════════════════════════════════════════════════════════════════
-- § Batch-norm γ / β — the `den` folds N into the per-channel reduction `m = N·(h·w)` (delegation)
-- ════════════════════════════════════════════════════════════════

/-- **Batched BN γ op denotes the certified per-channel γ gradient over the merged batch+spatial
    axis `m = N·(h·w)`.** True batch-norm's γ grad is per-channel BN's γ grad at `m = N·h·w`
    (γ enters affinely — no batch coupling in the *param* grad), so this is a direct delegation to
    the generic `cifar_bn_render_gamma_certified` at `m = N·(h·w)` (the den's exact reduction count,
    via the network→oc-major reindex `bnchwFwd`). Generic in the free `β`. -/
theorem bnGammaB_den {N oc h w : Nat}
    (gN vN epsStr lrStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (N * (oc * (h * w)))) (cot : Vec (N * (oc * (h * w)))) (lr : ℝ) (idx : Fin oc) :
    den (SHlo.bnGammaSgdB gN vN epsStr lrStr ε γ v lr (.operand cotN cot)) idx
      = γ idx - lr * ∑ j : Fin (oc * (N * (h * w))),
          pdiv (fun γ' : Vec oc =>
                  bnPerChannelFlat oc (N * (h * w)) ε γ' β (bnchwFwd N oc h w v))
               γ idx j * bnchwFwd N oc h w cot j := by
  simp only [den]
  exact cifar_bn_render_gamma_certified oc (N * (h * w)) ε γ β
    (bnchwFwd N oc h w v) (bnchwFwd N oc h w cot) lr idx

/-- **Batched BN β op denotes the certified per-channel β gradient `Σ_{batch,spatial} cot`** at
    `m = N·(h·w)`. Used for every BN β AND (as the channel-sum) every conv/depthwise bias. Direct
    delegation to `cifar_bn_render_beta_certified`. The pdiv form carries a free `v`/`γ` (β's grad
    is the channel-sum, independent of them). -/
theorem bnBetaB_den {N oc h w : Nat}
    (bN lrStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (oc * (N * (h * w)))) (cot : Vec (N * (oc * (h * w)))) (lr : ℝ) (idx : Fin oc) :
    den (SHlo.bnBetaSgdB bN lrStr β lr (.operand cotN cot)) idx
      = β idx - lr * ∑ j : Fin (oc * (N * (h * w))),
          pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) ε γ β' v)
               β idx j * bnchwFwd N oc h w cot j := by
  simp only [den]
  exact cifar_bn_render_beta_certified oc (N * (h * w)) ε γ β v (bnchwFwd N oc h w cot) lr idx

-- ════════════════════════════════════════════════════════════════
-- § Depthwise weight (stride-1 / strided) — Σ_n of the per-example depthwise VJP (HasVJP3, flattened)
-- ════════════════════════════════════════════════════════════════

/-- **Batched stride-1 depthwise weight op denotes the certified Σ_n batched weight gradient.**
    `Σ_n` of the flattened `depthwise_weight_grad_has_vjp3.correct` (the per-slice grad bridge from
    `mnv2_render_depthwiseW_flat_certified`). Generic in the kernel size (3×3 and 5×5). -/
theorem depthwiseWB_den {N c h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec c) (x : Vec (N * (c * h * w)))
    (W : DepthwiseKernel c kH kW) (cot : Vec (N * (c * h * w))) (lr : ℝ) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseWeightSgdB xN wN lrStr b x W lr (.operand cotN cot)) idx
      = Tensor3.flatten W idx - lr * ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b
                    (Tensor3.unflatten (batchSlice N (c * h * w) x n))))
               (Tensor3.flatten W) idx j * batchSlice N (c * h * w) cot n j := by
  simp only [den]
  congr 1
  apply congrArg (lr * ·)
  apply Finset.sum_congr rfl
  intro n _
  rw [← (hasVJP3_to_hasVJP (depthwise_weight_grad_has_vjp3 b
      (Tensor3.unflatten (batchSlice N (c * h * w) x n)))).correct
      (Tensor3.flatten W) (batchSlice N (c * h * w) cot n) idx]
  simp only [hasVJP3_to_hasVJP, Tensor3.flatten, Tensor3.unflatten_flatten]

/-- **Batched strided depthwise weight op denotes the certified Σ_n batched weight gradient.** The
    strided VJP is already flat, so `Σ_n` of `depthwiseStride2_weight_grad_has_vjp.correct`. -/
theorem depthwiseStridedWB_den {N c h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec c) (x : Vec (N * (c * (2*h) * (2*w))))
    (W : DepthwiseKernel c kH kW) (cot : Vec (N * (c * h * w))) (lr : ℝ) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseStridedWeightSgdB xN wN lrStr b x W lr (.operand cotN cot)) idx
      = Tensor3.flatten W idx - lr * ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  depthwiseStride2Flat (Tensor3.unflatten v') b (batchSlice N (c * (2*h) * (2*w)) x n))
               (Tensor3.flatten W) idx j * batchSlice N (c * h * w) cot n j := by
  simp only [den]
  congr 1
  apply congrArg (lr * ·)
  apply Finset.sum_congr rfl
  intro n _
  exact (depthwiseStride2_weight_grad_has_vjp b (batchSlice N (c * (2*h) * (2*w)) x n)).correct
    (Tensor3.flatten W) (batchSlice N (c * h * w) cot n) idx

end Proofs.EnetPoC
