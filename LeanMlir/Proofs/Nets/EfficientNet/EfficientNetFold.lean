import LeanMlir.Proofs.Foundation.GradNodesB

/-! # The full-16 (262-param) EfficientNet-B0 train step, proof-tied (the fold, den)

The fold for the batched 262-param EfficientNet-B0
train step `efficientnetTrainStepFaithfulV` (`EfficientNetRender.Basic`). Every emitted param-SGD op
`den`otes the certified loss-descent step — `θ − lr·(certified Jacobian · cotangent)`.

**The batched wrinkle vs mnv2.** EfficientNet trains at the batched index `N·(c·h·w)` with the
fused-batch param-SGD ops (`convWeightSgdB`/`denseWeightSgdB`/`bn{Gamma,Beta}SgdB`/…), whose `den`
carries a **batch sum `Σ_n`** over the per-example gradients (the shared-weight batched gradient).
So unlike mnv2 (per-example, one `∑_j pdiv·cot`), each fold here is "the per-example bridge applied
inside `Σ_n`" — the **batch-sum bridge**. For the linear families (conv/dense weight, bias) this is a
`Finset.sum_congr` of the per-example `.correct`; for BN γ/β the `den` already folds `N` into the
per-channel reduction count `m = N·(h·w)`, so it is the cert's exact LHS (delegation).

Each fused op is `θ − lr·` its un-fused gradient node by `rfl` (`StableHLO.Basic`'s `*SgdB_eq_grad`),
so every lemma here is the un-fused fold — `GradNodeB`'s (conv, dense weight, BN, the XLA-`SAME`
stem, depthwise, the rectangular dense bias) — under that wrapper. -/

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
  rw [convWeightSgdB_eq_grad, GradNodeB.convWGradB_den]

/-- **Batched strided-stem 3×3 conv weight op denotes the certified Σ_n batched weight gradient.**
    `Σ_n` of `flatConvStride2XlaWeightGradHasVJP.correct`. The op is the XLA-`SAME`
    `convStridedXlaWeightSgdB` the render emits at the stem (`EfficientNetRender.Basic`), whose
    weight-grad correlation pad is shifted one position; its `den` is the odd-phase weight VJP,
    so the certified gradient here is the gradient of the net that ships. -/
theorem convStridedWB_den {N ic oc h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec oc) (x : Vec (N * (ic * (2*h) * (2*w))))
    (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w))) (lr : ℝ) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStridedXlaWeightSgdB xN wN lrStr b x W lr (.operand cotN cot)) idx
      = Kernel4.flatten W idx - lr * ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  flatConvStride2Xla (Kernel4.unflatten v') b (batchSlice N (ic * (2*h) * (2*w)) x n))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j :=
  congrArg (Kernel4.flatten W idx - lr * ·)
    (GradNodeB.convStridedXlaWGradB_den xN cotN b x W cot idx)

-- ════════════════════════════════════════════════════════════════
-- § Dense weight / bias (SE squeeze+excite, head classifier) — Σ_n of the per-example dense VJP
-- ════════════════════════════════════════════════════════════════

/-- **Batched dense weight op denotes the certified Σ_n batched weight gradient.** `Σ_n` of the
    dense outer-product `.correct` (`denseWeightGrad_correct`). Covers the SE squeeze/excite denses
    (`W₁ : c→r`, `W₂ : r→c`) and the head classifier. Generic in `b` (the grad is `b`-independent). -/
theorem denseWB_den {N a c : Nat}
    (xN wN lrStr cotN : String) (x : Vec (N * a)) (W : Mat a c) (b : Vec c) (cot : Vec (N * c))
    (lr : ℝ) (i : Fin a) (j : Fin c) :
    den (SHlo.denseWeightSgdB xN wN lrStr x W lr (.operand cotN cot)) (finProdFinEquiv (i, j))
      = W i j - lr * ∑ n : Fin N, ∑ k : Fin c,
          pdiv (fun v : Vec (a * c) => dense (Mat.unflatten v) b (batchSlice N a x n))
               (Mat.flatten W) (finProdFinEquiv (i, j)) k * batchSlice N c cot n k := by
  rw [denseWeightSgdB_eq_grad, GradNodeB.denseWGradB_den xN cotN x W b cot i j]
  simp only [Mat.flatten, Equiv.symm_apply_apply]

/-- **Batched dense bias op denotes the certified Σ_n batched bias gradient** (`Σ_{n} cotₙ` per
    output) — `Σ_n` of `denseBiasGrad_correct`. Covers the SE `b₁`/`b₂` and the head bias. -/
theorem denseBB_den {N c : Nat}
    (bN lrStr cotN : String) (W : Mat c c) (x : Vec c) (b : Vec c) (cot : Vec (N * c))
    (lr : ℝ) (j : Fin c) :
    den (SHlo.denseBiasSgdB bN lrStr b lr (.operand cotN cot)) j
      = b j - lr * ∑ n : Fin N, ∑ k : Fin c,
          pdiv (fun b' : Vec c => dense W b' x) b j k * batchSlice N c cot n k := by
  rw [denseBiasSgdB_eq_grad, GradNodeB.denseBGradB_den cotN W x b cot j]

-- ════════════════════════════════════════════════════════════════
-- § Batch-norm γ / β — the `den` folds N into the per-channel reduction `m = N·(h·w)` (delegation)
-- ════════════════════════════════════════════════════════════════

/-- **Batched BN γ op denotes the certified per-channel γ gradient over the merged batch+spatial
    axis `m = N·(h·w)`.** True batch-norm's γ grad is per-channel BN's γ grad at `m = N·h·w`
    (γ enters affinely — no batch coupling in the *param* grad), so this is a direct delegation to
    the generic `bnPerChannel_gamma_sgd_certified` at `m = N·(h·w)` (the den's exact reduction count,
    via the network→oc-major reindex `bnchwFwd`). Generic in the free `β`. -/
theorem bnGammaB_den {N oc h w : Nat}
    (gN vN epsStr lrStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (N * (oc * (h * w)))) (cot : Vec (N * (oc * (h * w)))) (lr : ℝ) (idx : Fin oc) :
    den (SHlo.bnGammaSgdB gN vN epsStr lrStr ε γ v lr (.operand cotN cot)) idx
      = γ idx - lr * ∑ j : Fin (oc * (N * (h * w))),
          pdiv (fun γ' : Vec oc =>
                  bnPerChannelFlat oc (N * (h * w)) ε γ' β (bnchwFwd N oc h w v))
               γ idx j * bnchwFwd N oc h w cot j := by
  rw [bnGammaSgdB_eq_grad, GradNodeB.bnGammaGradB_den]

/-- **Batched BN β op denotes the certified per-channel β gradient `Σ_{batch,spatial} cot`** at
    `m = N·(h·w)`. Used for every BN β AND (as the channel-sum) every conv/depthwise bias. Direct
    delegation to `bnPerChannel_beta_sgd_certified`. The pdiv form carries a free `v`/`γ` (β's grad
    is the channel-sum, independent of them). -/
theorem bnBetaB_den {N oc h w : Nat}
    (bN lrStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (oc * (N * (h * w)))) (cot : Vec (N * (oc * (h * w)))) (lr : ℝ) (idx : Fin oc) :
    den (SHlo.bnBetaSgdB bN lrStr β lr (.operand cotN cot)) idx
      = β idx - lr * ∑ j : Fin (oc * (N * (h * w))),
          pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) ε γ β' v)
               β idx j * bnchwFwd N oc h w cot j := by
  rw [bnBetaSgdB_eq_grad, GradNodeB.bnBetaGradB_den cotN ε γ β v cot idx]

/-- **One batched BN layer's fused γ and β SGD nodes, tied** — `GradNodeB.BnPairTiedB` under
    `θ − lr·`: the emitted `bnGammaSgdB` / `bnBetaSgdB` denote the certified per-channel γ and β
    steps at the layer's pre-BN activation `v` and output cotangent `cot`. -/
def BnSgdPairTiedB (N oc h w : Nat) (gN vN epsStr bN lrStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v cot : Vec (N * (oc * (h * w)))) (lr : ℝ) : Prop :=
  (∀ k : Fin oc,
      den (SHlo.bnGammaSgdB gN vN epsStr lrStr ε γ v lr (.operand cotN cot)) k
        = γ k - lr * ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (N * (h * w)) ε γ' β (bnchwFwd N oc h w v))
                 γ k j * bnchwFwd N oc h w cot j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnBetaSgdB bN lrStr β lr (.operand cotN cot)) k
        = β k - lr * ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) ε γ β' (bnchwFwd N oc h w v))
                 β k j * bnchwFwd N oc h w cot j)

theorem bnSgdPairTiedB_holds {N oc h w : Nat} {gN vN epsStr bN lrStr cotN : String} {ε : ℝ}
    {γ β : Vec oc} {v cot : Vec (N * (oc * (h * w)))} {lr : ℝ} :
    BnSgdPairTiedB N oc h w gN vN epsStr bN lrStr cotN ε γ β v cot lr :=
  ⟨fun k => bnGammaB_den gN vN epsStr lrStr cotN ε γ β v cot lr k,
   fun k => bnBetaB_den bN lrStr cotN ε γ β (bnchwFwd N oc h w v) cot lr k⟩

-- ════════════════════════════════════════════════════════════════
-- § Depthwise weight (stride-1 / strided) — Σ_n of the per-example depthwise VJP (HasVJP3, flattened)
-- ════════════════════════════════════════════════════════════════

/-- **Batched stride-1 depthwise weight op denotes the certified Σ_n batched weight gradient.**
    `Σ_n` of the flattened `depthwiseWeightGradHasVJP3.correct` (the per-slice grad bridge from
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
  rw [depthwiseWeightSgdB_eq_grad, GradNodeB.depthwiseWGradB_den]

/-- **Batched strided depthwise weight op denotes the certified Σ_n batched weight gradient.** The
    strided VJP is already flat, so `Σ_n` of `depthwiseStride2WeightGradHasVJP.correct`. -/
theorem depthwiseStridedWB_den {N c h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec c) (x : Vec (N * (c * (2*h) * (2*w))))
    (W : DepthwiseKernel c kH kW) (cot : Vec (N * (c * h * w))) (lr : ℝ) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseStridedWeightSgdB xN wN lrStr b x W lr (.operand cotN cot)) idx
      = Tensor3.flatten W idx - lr * ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  depthwiseStride2Flat (Tensor3.unflatten v') b (batchSlice N (c * (2*h) * (2*w)) x n))
               (Tensor3.flatten W) idx j * batchSlice N (c * h * w) cot n j := by
  rw [depthwiseStridedWeightSgdB_eq_grad, GradNodeB.depthwiseStridedWGradB_den]

-- ════════════════════════════════════════════════════════════════
-- § Tie clauses — one fused SGD node each (the `ResNet34PoCB.*TiedB` clauses under `θ − lr·`)
-- ════════════════════════════════════════════════════════════════

/-- A stride-1 conv weight SGD node, tied (`convWB_den`). -/
def ConvWSgdTiedB (N h w : Nat) {ic oc kH kW : Nat} (xN wN lrStr cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w)))
    (lr : ℝ) : Prop :=
  ∀ idx : Fin (oc * ic * kH * kW),
    den (SHlo.convWeightSgdB xN wN lrStr b x W lr (.operand cotN cot)) idx
      = Kernel4.flatten W idx - lr * ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                    (Tensor3.unflatten (batchSlice N (ic * h * w) x n))))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j

/-- A stride-1 depthwise weight SGD node, tied (`depthwiseWB_den`). -/
def DepthwiseWSgdTiedB (N h w : Nat) {c kH kW : Nat} (xN wN lrStr cotN : String) (b : Vec c)
    (x : Vec (N * (c * h * w))) (W : DepthwiseKernel c kH kW) (cot : Vec (N * (c * h * w)))
    (lr : ℝ) : Prop :=
  ∀ idx : Fin (c * kH * kW),
    den (SHlo.depthwiseWeightSgdB xN wN lrStr b x W lr (.operand cotN cot)) idx
      = Tensor3.flatten W idx - lr * ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b
                    (Tensor3.unflatten (batchSlice N (c * h * w) x n))))
               (Tensor3.flatten W) idx j * batchSlice N (c * h * w) cot n j

/-- A dense weight SGD node, tied (`denseWB_den`). -/
def DenseWSgdTiedB (N : Nat) {a c : Nat} (xN wN lrStr cotN : String) (x : Vec (N * a))
    (W : Mat a c) (b : Vec c) (cot : Vec (N * c)) (lr : ℝ) : Prop :=
  ∀ (i : Fin a) (j : Fin c),
    den (SHlo.denseWeightSgdB xN wN lrStr x W lr (.operand cotN cot)) (finProdFinEquiv (i, j))
      = W i j - lr * ∑ n : Fin N, ∑ k : Fin c,
          pdiv (fun v : Vec (a * c) => dense (Mat.unflatten v) b (batchSlice N a x n))
               (Mat.flatten W) (finProdFinEquiv (i, j)) k * batchSlice N c cot n k

/-- A dense bias SGD node, tied — free in `W` and `x`, which `b`'s gradient ignores. -/
def DenseBSgdTiedB (N : Nat) {a c : Nat} (bN lrStr cotN : String) (W : Mat a c) (x : Vec a)
    (b : Vec c) (cot : Vec (N * c)) (lr : ℝ) : Prop :=
  ∀ j : Fin c,
    den (SHlo.denseBiasSgdB bN lrStr b lr (.operand cotN cot)) j
      = b j - lr * ∑ n : Fin N, ∑ k : Fin c,
          pdiv (fun b' : Vec c => dense W b' x) b j k * batchSlice N c cot n k

/-! Each clause holds, every argument implicit (read off the goal by a step tie's constructor). -/

theorem convWSgdTiedB_holds {N h w ic oc kH kW : Nat} {xN wN lrStr cotN : String} {b : Vec oc}
    {x : Vec (N * (ic * h * w))} {W : Kernel4 oc ic kH kW} {cot : Vec (N * (oc * h * w))} {lr : ℝ} :
    ConvWSgdTiedB N h w xN wN lrStr cotN b x W cot lr := fun idx =>
  convWB_den xN wN lrStr cotN b x W cot lr idx

theorem depthwiseWSgdTiedB_holds {N h w c kH kW : Nat} {xN wN lrStr cotN : String} {b : Vec c}
    {x : Vec (N * (c * h * w))} {W : DepthwiseKernel c kH kW} {cot : Vec (N * (c * h * w))}
    {lr : ℝ} : DepthwiseWSgdTiedB N h w xN wN lrStr cotN b x W cot lr := fun idx =>
  depthwiseWB_den xN wN lrStr cotN b x W cot lr idx

theorem denseWSgdTiedB_holds {N a c : Nat} {xN wN lrStr cotN : String} {x : Vec (N * a)}
    {W : Mat a c} {b : Vec c} {cot : Vec (N * c)} {lr : ℝ} :
    DenseWSgdTiedB N xN wN lrStr cotN x W b cot lr := fun i j =>
  denseWB_den xN wN lrStr cotN x W b cot lr i j

/-- At a square witness, as every use is (the gradient ignores `W` and `x`). -/
theorem denseBSgdTiedB_holds {N c : Nat} {bN lrStr cotN : String} {W : Mat c c} {x : Vec c}
    {b : Vec c} {cot : Vec (N * c)} {lr : ℝ} : DenseBSgdTiedB N bN lrStr cotN W x b cot lr :=
  fun j => denseBB_den bN lrStr cotN W x b cot lr j

end Proofs.EnetPoC
