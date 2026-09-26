import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Architectures.PerChannelBNGrad
import LeanMlir.Proofs.Architectures.ConvGrad

/-! # The batched f32 gradient nodes — one `*GradB_den` per op kind, shared by every conv net

Every batched train step the suite renders emits the RAW gradient (`*GradB`) and hands it to an
optimizer tail (SGD, heavy-ball, Adam/AdamW, RMSProp, EMA, the data-parallel all-reduce), so one
lemma per op kind — "this node denotes the certified `Σ_n` gradient" — certifies every optimizer
variant of every net at once. Each proof is `Finset.sum_congr rfl` over the batch and then the
per-example VJP at `batchSlice n`. The bf16 kinds (`*GradBBf16`) are a different real number and
are folded in `Bf16GradNodes`; the fused `*SgdB` ops are these through
`StableHLO.lean`'s `*SgdB_eq_grad` family (`rfl`).

Namespaces are the net that first needed the op (kept so that every citation keeps its name):

| op kinds | namespace | emitted by |
|---|---|---|
| conv / strided-conv (symmetric) W and b, BN γ/β, dense W/b, the `*TiedB` clause Props | `ResNet34PoCB` | every conv net |
| XLA-`SAME` strided conv W, depthwise W, symmetric strided depthwise W, rectangular dense b | `EnetPoCG` | EfficientNet-B0, MobileNetV2/V4, ConvNeXt |
| XLA-`SAME` strided conv b, depthwise b, XLA-`SAME` strided depthwise W and b | `Mnv2PaperPoCG` | MobileNetV2, ConvNeXt |
| stride-4 patchify conv W | `CnxPoCGB` | ConvNeXt |
| vector-LN γ / β and their `*TiedB` clauses, the classifier head W and b | `ViTPoCGB` | ViT, ConvNeXt |

The other ViT nodes (per-token dense, patch embed, position, CLS) live in `ViTFoldGB`;
ConvNeXt's channel-LN and layer-scale nodes in `ConvNeXtFoldGB`.

Note: Padding is invisible in the types: the symmetric and XLA-`SAME` strided kinds have identical
types and identical emitted shapes, and only the certificate tells them apart.

Every lemma is `∀ cot`; pinning each cotangent to the emitted backward subgraph is each net's
`*StepTie*` file.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ResNet34PoCB

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Stride-1 conv weight / bias (block convs)
-- ════════════════════════════════════════════════════════════════

/-- **Batched stride-1 conv weight GRADIENT denotes the certified `Σ_n` weight gradient.** The
    un-fused peer of `EnetPoC.convWB_den`: same `Σ_n` of `conv_weight_grad_bridge`, with no
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
  simp only [denStep, denStepApp]
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
  simp only [denStep, denStepApp]
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
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride2WeightGradHasVJP b (batchSlice N (ic * (2 * h) * (2 * w)) x n)).correct
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
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride2BiasGradHasVJP W (batchSlice N (ic * (2 * h) * (2 * w)) x n)).correct
    b (batchSlice N (oc * h * w) cot n) o

-- ════════════════════════════════════════════════════════════════
-- § BatchNorm gamma / beta — the `den` folds N into the reduction `m = N*(h*w)`
-- ════════════════════════════════════════════════════════════════

/-- **Batched BN γ GRADIENT denotes the certified per-channel γ gradient over the merged
    batch+spatial axis `m = N·(h·w)`.** γ enters affinely, so there is no batch coupling in the
    PARAM gradient and this is `bnPerChannelGradGamma_correct` at that width, through the
    network→oc-major reindex `bnchwFwd`. Generic in the free `β`. -/
theorem bnGammaGradB_den {N oc h w : Nat}
    (vN epsStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (N * (oc * (h * w)))) (cot : Vec (N * (oc * (h * w)))) (c : Fin oc) :
    den (SHlo.bnGammaGradB vN epsStr ε v (.operand cotN cot)) c
      = ∑ j : Fin (oc * (N * (h * w))),
          pdiv (fun γ' : Vec oc =>
                  bnPerChannelFlat oc (N * (h * w)) ε γ' β (bnchwFwd N oc h w v))
               γ c j * bnchwFwd N oc h w cot j := by
  simp only [denStep, denStepApp]
  exact bnPerChannelGradGamma_correct oc (N * (h * w)) ε γ β
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
  simp only [denStep, denStepApp]
  exact bnPerChannelGradBeta_correct oc (N * (h * w)) ε γ β v (bnchwFwd N oc h w cot) c

/-- **One batched BN layer's γ and β gradient nodes, tied** — the pair every step tie states per
    BatchNorm: the emitted `bnGammaGradB` / `bnBetaGradB` denote the certified per-channel γ and β
    gradients over the merged batch+spatial axis, at the layer's pre-BN activation `v` and its
    output cotangent `cot` (both in the network layout). -/
def BnPairTiedB (N oc h w : Nat) (vN epsStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v cot : Vec (N * (oc * (h * w)))) : Prop :=
  (∀ k : Fin oc,
      den (SHlo.bnGammaGradB vN epsStr ε v (.operand cotN cot)) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (N * (h * w)) ε γ' β (bnchwFwd N oc h w v))
                 γ k j * bnchwFwd N oc h w cot j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) (.operand cotN cot)) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) ε γ β' (bnchwFwd N oc h w v))
                 β k j * bnchwFwd N oc h w cot j)

theorem bnPairTiedB_holds {N oc h w : Nat} {vN epsStr cotN : String} {ε : ℝ} {γ β : Vec oc}
    {v cot : Vec (N * (oc * (h * w)))} : BnPairTiedB N oc h w vN epsStr cotN ε γ β v cot :=
  ⟨fun k => bnGammaGradB_den vN epsStr cotN ε γ β v cot k,
   fun k => bnBetaGradB_den cotN ε γ β (bnchwFwd N oc h w v) cot k⟩

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
  simp only [denStep, denStepApp, Mat.flatten, Equiv.symm_apply_apply]
  apply Finset.sum_congr rfl
  intro n _
  exact denseWeightGrad_correct W b (batchSlice N a x n) (batchSlice N c cot n) i j

/-- **Batched dense bias GRADIENT denotes the certified `Σ_n` cotangent sum.** -/
theorem denseBGradB_den {N c : Nat}
    (cotN : String) (W : Mat c c) (x : Vec c) (b : Vec c) (cot : Vec (N * c)) (j : Fin c) :
    den (SHlo.denseBiasGradB (N := N) (.operand cotN cot)) j
      = ∑ n : Fin N, ∑ k : Fin c,
          pdiv (fun b' : Vec c => dense W b' x) b j k * batchSlice N c cot n k := by
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact denseBiasGrad_correct W b x (batchSlice N c cot n) j

-- ════════════════════════════════════════════════════════════════
-- § Tie clauses — one gradient node each
--   What a step tie states per conv / depthwise / dense parameter: the emitted `*GradB` node
--   denotes the certified `Σ_n` gradient at the layer's input `x` and output cotangent `cot`.
--   Each is its `_den` lemma's statement with the index bound; `…TiedB_holds` (end of file,
--   every argument implicit) proves it, so a step tie is an anonymous constructor of them.
-- ════════════════════════════════════════════════════════════════

/-- A stride-1 conv weight gradient node, tied (`convWGradB_den`). -/
def ConvWTiedB (N h w : Nat) {ic oc kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w))) :
    Prop :=
  ∀ idx : Fin (oc * ic * kH * kW),
    den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                    (Tensor3.unflatten (batchSlice N (ic * h * w) x n))))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j

/-- A stride-1 conv bias gradient node, tied (`convBGradB_den`). -/
def ConvBTiedB (N h w : Nat) {ic oc kH kW : Nat} (cotN : String) (W : Kernel4 oc ic kH kW)
    (x : Vec (N * (ic * h * w))) (b : Vec oc) (cot : Vec (N * (oc * h * w))) : Prop :=
  ∀ o : Fin oc,
    den (SHlo.convBiasGradB (h := h) (w := w) W x b (.operand cotN cot)) o
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc =>
                  Tensor3.flatten (conv2d W b'
                    (Tensor3.unflatten (batchSlice N (ic * h * w) x n))))
               b o j * batchSlice N (oc * h * w) cot n j

/-- A stride-2 (symmetric-pad) conv weight gradient node, tied (`convStridedWGradB_den`). -/
def ConvStridedWTiedB (N h w : Nat) {ic oc kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) : Prop :=
  ∀ idx : Fin (oc * ic * kH * kW),
    den (SHlo.convStridedWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  flatConvStride2 (Kernel4.unflatten v') b
                    (batchSlice N (ic * (2 * h) * (2 * w)) x n))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j

/-- A stride-2 (symmetric-pad) conv bias gradient node, tied (`convStridedBGradB_den`). -/
def ConvStridedBTiedB (N h w : Nat) {ic oc kH kW : Nat} (cotN : String) (W : Kernel4 oc ic kH kW)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (b : Vec oc) (cot : Vec (N * (oc * h * w))) : Prop :=
  ∀ o : Fin oc,
    den (SHlo.convStridedBiasGradB (h := h) (w := w) W x b (.operand cotN cot)) o
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc =>
                  flatConvStride2 W b' (batchSlice N (ic * (2 * h) * (2 * w)) x n))
               b o j * batchSlice N (oc * h * w) cot n j

/-- A stride-2 XLA-`SAME` conv weight gradient node, tied (`EnetPoCG.convStridedXlaWGradB_den`). -/
def ConvStridedXlaWTiedB (N h w : Nat) {ic oc kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) : Prop :=
  ∀ idx : Fin (oc * ic * kH * kW),
    den (SHlo.convStridedXlaWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  flatConvStride2Xla (Kernel4.unflatten v') b
                    (batchSlice N (ic * (2 * h) * (2 * w)) x n))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j

/-- A stride-1 depthwise weight gradient node, tied (`EnetPoCG.depthwiseWGradB_den`). -/
def DepthwiseWTiedB (N h w : Nat) {c kH kW : Nat} (xN cotN : String) (b : Vec c)
    (x : Vec (N * (c * h * w))) (W : DepthwiseKernel c kH kW) (cot : Vec (N * (c * h * w))) :
    Prop :=
  ∀ idx : Fin (c * kH * kW),
    den (SHlo.depthwiseWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b
                    (Tensor3.unflatten (batchSlice N (c * h * w) x n))))
               (Tensor3.flatten W) idx j * batchSlice N (c * h * w) cot n j

/-- A stride-1 depthwise bias gradient node, tied (`Mnv2PaperPoCG.depthwiseBGradB_den`). -/
def DepthwiseBTiedB (N h w : Nat) {c kH kW : Nat} (cotN : String) (W : DepthwiseKernel c kH kW)
    (x : Vec (N * (c * h * w))) (b : Vec c) (cot : Vec (N * (c * h * w))) : Prop :=
  ∀ o : Fin c,
    den (SHlo.depthwiseBiasGradB W x b (.operand cotN cot)) o
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c =>
                  Tensor3.flatten (depthwiseConv2d W b'
                    (Tensor3.unflatten (batchSlice N (c * h * w) x n))))
               b o j * batchSlice N (c * h * w) cot n j

/-- A stride-2 depthwise weight gradient node, tied (`EnetPoCG.depthwiseStridedWGradB_den`). -/
def DepthwiseStridedWTiedB (N h w : Nat) {c kH kW : Nat} (xN cotN : String) (b : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    (cot : Vec (N * (c * h * w))) : Prop :=
  ∀ idx : Fin (c * kH * kW),
    den (SHlo.depthwiseStridedWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  depthwiseStride2Flat (Tensor3.unflatten v') b
                    (batchSlice N (c * (2 * h) * (2 * w)) x n))
               (Tensor3.flatten W) idx j * batchSlice N (c * h * w) cot n j

/-- A dense weight gradient node, tied (`denseWGradB_den`). -/
def DenseWTiedB (N : Nat) {a c : Nat} (xN cotN : String) (x : Vec (N * a)) (W : Mat a c)
    (b : Vec c) (cot : Vec (N * c)) : Prop :=
  ∀ (i : Fin a) (j : Fin c),
    den (SHlo.denseWeightGradB (c := c) xN x (.operand cotN cot)) (finProdFinEquiv (i, j))
      = ∑ n : Fin N, ∑ k : Fin c,
          pdiv (fun v : Vec (a * c) => dense (Mat.unflatten v) b (batchSlice N a x n))
               (Mat.flatten W) (finProdFinEquiv (i, j)) k * batchSlice N c cot n k

/-- A dense bias gradient node, tied — free in `W` and `x`, which `b`'s gradient ignores. -/
def DenseBTiedB (N : Nat) {a c : Nat} (cotN : String) (W : Mat a c) (x : Vec a) (b : Vec c)
    (cot : Vec (N * c)) : Prop :=
  ∀ j : Fin c,
    den (SHlo.denseBiasGradB (N := N) (.operand cotN cot)) j
      = ∑ n : Fin N, ∑ k : Fin c,
          pdiv (fun b' : Vec c => dense W b' x) b j k * batchSlice N c cot n k

end Proofs.ResNet34PoCB

namespace Proofs.EnetPoCG

open scoped BigOperators

/-- **Batched dense bias GRADIENT denotes the certified `Σ_n` cotangent sum.** r34's peer at a
    RECTANGULAR witness `Mat a c`, which the SE's `c → r` squeeze needs; the gradient is the
    channel sum and depends on neither `W` nor `x`, so the widening is free. -/
theorem denseBGradB_den {N a c : Nat}
    (cotN : String) (W : Mat a c) (x : Vec a) (b : Vec c) (cot : Vec (N * c)) (j : Fin c) :
    den (SHlo.denseBiasGradB (N := N) (.operand cotN cot)) j
      = ∑ n : Fin N, ∑ k : Fin c,
          pdiv (fun b' : Vec c => dense W b' x) b j k * batchSlice N c cot n k := by
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact denseBiasGrad_correct W b x (batchSlice N c cot n) j

-- ════════════════════════════════════════════════════════════════
-- § New: the XLA-`SAME` strided stem
--   ⚠ `flatConvStride2Xla`, NOT r34's symmetric `flatConvStride2`. Identical types.
-- ════════════════════════════════════════════════════════════════

/-- **Batched XLA-`SAME` strided conv weight GRADIENT denotes the certified `Σ_n` weight
    gradient.** B0's 3×3/s2 stem, the net's one XLA-phase site. `Σ_n` of
    `flatConvStride2XlaWeightGradHasVJP.correct` — the odd-phase weight VJP, so the certified
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
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride2XlaWeightGradHasVJP b
    (batchSlice N (ic * (2 * h) * (2 * w)) x n)).correct
    (Kernel4.flatten W) (batchSlice N (oc * h * w) cot n) idx

-- ════════════════════════════════════════════════════════════════
-- § New: the MBConv depthwise kernels (3×3 and 5×5), stride 1 and stride 2
--   ⚠ SYMMETRIC padding at the strided sites — the render's forward is `.depthwiseStrided` too.
-- ════════════════════════════════════════════════════════════════

/-- **Batched stride-1 depthwise weight GRADIENT denotes the certified `Σ_n` weight gradient.**
    `Σ_n` of the flattened `depthwiseWeightGradHasVJP3.correct`. Generic in the kernel size, so
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
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  rw [← (HasVJP3.toHasVJP (depthwiseWeightGradHasVJP3 b
      (Tensor3.unflatten (batchSlice N (c * h * w) x n)))).correct
      (Tensor3.flatten W) (batchSlice N (c * h * w) cot n) idx]
  simp only [HasVJP3.toHasVJP, Tensor3.flatten, Tensor3.unflatten_flatten]

/-- **Batched strided depthwise weight GRADIENT denotes the certified `Σ_n` weight gradient.** The
    strided VJP is already flat, so this is `Σ_n` of
    `depthwiseStride2WeightGradHasVJP.correct`. -/
theorem depthwiseStridedWGradB_den {N c h w kH kW : Nat}
    (xN cotN : String) (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w))))
    (W : DepthwiseKernel c kH kW) (cot : Vec (N * (c * h * w))) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseStridedWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  depthwiseStride2Flat (Tensor3.unflatten v') b
                    (batchSlice N (c * (2 * h) * (2 * w)) x n))
               (Tensor3.flatten W) idx j * batchSlice N (c * h * w) cot n j := by
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact (depthwiseStride2WeightGradHasVJP b
    (batchSlice N (c * (2 * h) * (2 * w)) x n)).correct
    (Tensor3.flatten W) (batchSlice N (c * h * w) cot n) idx

end Proofs.EnetPoCG

namespace Proofs.Mnv2PaperPoCG

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The four new op kinds
--   ⚠ XLA-`SAME` at every strided site.
-- ════════════════════════════════════════════════════════════════

/-- **Batched XLA-`SAME` strided conv bias GRADIENT denotes the certified `Σ_n` bias gradient.**
    The stem's bias slot, at `convBias := true`. Same `reduce` text as the stride-1 bias grad; the
    `den` is the odd-phase bias VJP. -/
theorem convStridedXlaBGradB_den {N ic oc h w kH kW : Nat} (cotN : String)
    (W : Kernel4 oc ic kH kW) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (b : Vec oc)
    (cot : Vec (N * (oc * h * w))) (o : Fin oc) :
    den (SHlo.convStridedXlaBiasGradB (h := h) (w := w) W x b (.operand cotN cot)) o
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc =>
                  flatConvStride2Xla W b' (batchSlice N (ic * (2 * h) * (2 * w)) x n))
               b o j * batchSlice N (oc * h * w) cot n j := by
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride2XlaBiasGradHasVJP W
    (batchSlice N (ic * (2 * h) * (2 * w)) x n)).correct b (batchSlice N (oc * h * w) cot n) o

/-- **Batched stride-1 depthwise bias GRADIENT denotes the certified `Σ_n` bias gradient.** The
    depthwise bias slots at `convBias := true`. EfficientNet has no instance of this op — its
    depthwise convs are followed by BatchNorm, so their bias is always folded. -/
theorem depthwiseBGradB_den {N c h w kH kW : Nat} (cotN : String)
    (W : DepthwiseKernel c kH kW) (x : Vec (N * (c * h * w))) (b : Vec c)
    (cot : Vec (N * (c * h * w))) (o : Fin c) :
    den (SHlo.depthwiseBiasGradB W x b (.operand cotN cot)) o
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c =>
                  Tensor3.flatten (depthwiseConv2d W b'
                    (Tensor3.unflatten (batchSlice N (c * h * w) x n))))
               b o j * batchSlice N (c * h * w) cot n j := by
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact (depthwiseBiasGradHasVJP W
    (Tensor3.unflatten (batchSlice N (c * h * w) x n))).correct b
    (batchSlice N (c * h * w) cot n) o

/-- **Batched XLA-`SAME` strided depthwise weight GRADIENT denotes the certified `Σ_n` weight
    gradient.** The four stride-2 depthwises (b2/b4/b7/b14). Note: This is the `Xla` op — its
    weight-grad correlation keeps the `[p−1, p+1]` pad, the opposite asymmetry from the input-grad,
    and that asymmetry is the whole content of the variant. B0's strided depthwise is the
    SYMMETRIC op, so the two nets do not share this certificate. -/
theorem depthwiseStridedXlaWGradB_den {N c h w kH kW : Nat} (xN cotN : String)
    (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    (cot : Vec (N * (c * h * w))) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseStridedXlaWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  depthwiseStride2FlatXla (Tensor3.unflatten v') b
                    (batchSlice N (c * (2 * h) * (2 * w)) x n))
               (Tensor3.flatten W) idx j * batchSlice N (c * h * w) cot n j := by
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact (depthwiseStride2XlaWeightGradHasVJP b
    (batchSlice N (c * (2 * h) * (2 * w)) x n)).correct
    (Tensor3.flatten W) (batchSlice N (c * h * w) cot n) idx

/-- **Batched XLA-`SAME` strided depthwise bias GRADIENT denotes the certified `Σ_n` bias
    gradient.** At `convBias := true`. -/
theorem depthwiseStridedXlaBGradB_den {N c h w kH kW : Nat} (cotN : String)
    (W : DepthwiseKernel c kH kW) (x : Vec (N * (c * (2 * h) * (2 * w)))) (b : Vec c)
    (cot : Vec (N * (c * h * w))) (o : Fin c) :
    den (SHlo.depthwiseStridedXlaBiasGradB (h := h) (w := w) W x b (.operand cotN cot)) o
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c =>
                  depthwiseStride2FlatXla W b' (batchSlice N (c * (2 * h) * (2 * w)) x n))
               b o j * batchSlice N (c * h * w) cot n j := by
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact (depthwiseStride2XlaBiasGradHasVJP W
    (batchSlice N (c * (2 * h) * (2 * w)) x n)).correct b
    (batchSlice N (c * h * w) cot n) o

end Proofs.Mnv2PaperPoCG

namespace Proofs.CnxPoCGB

open scoped BigOperators

/-- **Batched patchify-stem weight GRADIENT denotes the certified `Σ_n` weight gradient.**
    Note: The emitted convolution contracts the batch axis itself (the transpose trick), so the outer
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
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride4WeightGradHasVJP b
    (batchSlice N (ic * (2 * (2 * h)) * (2 * (2 * w))) x n)).correct
    (Kernel4.flatten W) (batchSlice N (oc * h * w) cot n) idx

end Proofs.CnxPoCGB

namespace Proofs.ResNet34PoCB

/-! ## Each tie clause holds

One lemma per clause above, every argument implicit: a step tie's conjunction of clauses is then
an anonymous constructor of these, its arguments read off the goal. -/

theorem convWTiedB_holds {N h w ic oc kH kW : Nat} {xN cotN : String} {b : Vec oc}
    {x : Vec (N * (ic * h * w))} {W : Kernel4 oc ic kH kW} {cot : Vec (N * (oc * h * w))} :
    ConvWTiedB N h w xN cotN b x W cot := fun idx => convWGradB_den xN cotN b x W cot idx

theorem convBTiedB_holds {N h w ic oc kH kW : Nat} {cotN : String} {W : Kernel4 oc ic kH kW}
    {x : Vec (N * (ic * h * w))} {b : Vec oc} {cot : Vec (N * (oc * h * w))} :
    ConvBTiedB N h w cotN W x b cot := fun o => convBGradB_den cotN W x b cot o

theorem convStridedWTiedB_holds {N h w ic oc kH kW : Nat} {xN cotN : String} {b : Vec oc}
    {x : Vec (N * (ic * (2 * h) * (2 * w)))} {W : Kernel4 oc ic kH kW}
    {cot : Vec (N * (oc * h * w))} : ConvStridedWTiedB N h w xN cotN b x W cot :=
  fun idx => convStridedWGradB_den xN cotN b x W cot idx

theorem convStridedBTiedB_holds {N h w ic oc kH kW : Nat} {cotN : String}
    {W : Kernel4 oc ic kH kW} {x : Vec (N * (ic * (2 * h) * (2 * w)))} {b : Vec oc}
    {cot : Vec (N * (oc * h * w))} : ConvStridedBTiedB N h w cotN W x b cot :=
  fun o => convStridedBGradB_den cotN W x b cot o

theorem convStridedXlaWTiedB_holds {N h w ic oc kH kW : Nat} {xN cotN : String} {b : Vec oc}
    {x : Vec (N * (ic * (2 * h) * (2 * w)))} {W : Kernel4 oc ic kH kW}
    {cot : Vec (N * (oc * h * w))} : ConvStridedXlaWTiedB N h w xN cotN b x W cot :=
  fun idx => EnetPoCG.convStridedXlaWGradB_den xN cotN b x W cot idx

theorem depthwiseWTiedB_holds {N h w c kH kW : Nat} {xN cotN : String} {b : Vec c}
    {x : Vec (N * (c * h * w))} {W : DepthwiseKernel c kH kW} {cot : Vec (N * (c * h * w))} :
    DepthwiseWTiedB N h w xN cotN b x W cot :=
  fun idx => EnetPoCG.depthwiseWGradB_den xN cotN b x W cot idx

theorem depthwiseBTiedB_holds {N h w c kH kW : Nat} {cotN : String} {W : DepthwiseKernel c kH kW}
    {x : Vec (N * (c * h * w))} {b : Vec c} {cot : Vec (N * (c * h * w))} :
    DepthwiseBTiedB N h w cotN W x b cot :=
  fun o => Mnv2PaperPoCG.depthwiseBGradB_den cotN W x b cot o

theorem depthwiseStridedWTiedB_holds {N h w c kH kW : Nat} {xN cotN : String} {b : Vec c}
    {x : Vec (N * (c * (2 * h) * (2 * w)))} {W : DepthwiseKernel c kH kW}
    {cot : Vec (N * (c * h * w))} : DepthwiseStridedWTiedB N h w xN cotN b x W cot :=
  fun idx => EnetPoCG.depthwiseStridedWGradB_den xN cotN b x W cot idx

theorem denseWTiedB_holds {N a c : Nat} {xN cotN : String} {x : Vec (N * a)} {W : Mat a c}
    {b : Vec c} {cot : Vec (N * c)} : DenseWTiedB N xN cotN x W b cot :=
  fun i j => denseWGradB_den xN cotN x W b cot i j

theorem denseBTiedB_holds {N a c : Nat} {cotN : String} {W : Mat a c} {x : Vec a} {b : Vec c}
    {cot : Vec (N * c)} : DenseBTiedB N cotN W x b cot :=
  fun j => EnetPoCG.denseBGradB_den cotN W x b cot j

end Proofs.ResNet34PoCB

namespace Proofs.ViTPoCGB

open scoped BigOperators

/-! The batched vector-LayerNorm γ / β nodes, their tie clauses, and the classifier head's weight
and bias nodes. An LN β gradient and a per-token dense bias gradient are the same two-level reduce,
so `rowDenseBiasGradB` appears here against the LN forward. -/

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
  simp only [denStep, denStepApp]
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
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact vit_veclnBeta_grad_bridge ε γv β (X n) (batchSlice N (R * D) dy n) i

/-- **Batched classifier weight GRADIENT denotes the certified `Σ_n` outer product.** -/
theorem headWGradB_den {N D nC : Nat} (aN cotN : String)
    (a : Vec (N * D)) (Wc : Mat D nC) (bc : Vec nC) (cot : Vec (N * nC))
    (i : Fin D) (j : Fin nC) :
    den (SHlo.weightGradB (N := N) (m := D) (n := nC) aN a (.operand cotN cot))
        (finProdFinEquiv (i, j))
      = ∑ n : Fin N, ∑ k : Fin nC,
          pdiv (fun v : Vec (D * nC) => dense (Mat.unflatten v) bc (batchSlice N D a n))
               (Mat.flatten Wc) (finProdFinEquiv (i, j)) k * batchSlice N nC cot n k := by
  simp only [denStep, denStepApp, Mat.flatten, Equiv.symm_apply_apply]
  apply Finset.sum_congr rfl
  intro n _
  exact denseWeightGrad_correct Wc bc (batchSlice N D a n) (batchSlice N nC cot n) i j

/-- **Batched classifier bias GRADIENT denotes the certified cotangent, PER EXAMPLE.**

    Note: `biasGradB` is the identity on its operand — the reduce over the batch is in the emitted
    text, outside the AST — so the statement this node supports is the per-example one at every
    `batchSlice n`, and it is the per-example `biasGrad` carve-out carried over rather than a new
    one. `StableHLO.lean`'s constructor comment records the same thing on the emitter side. -/
theorem headBGradB_den {N D nC : Nat} (cotN : String)
    (Wc : Mat D nC) (a : Vec D) (bc : Vec nC) (cot : Vec (N * nC)) (n : Fin N) (i : Fin nC) :
    batchSlice N nC (den (SHlo.biasGradB (N := N) (n := nC) (.operand cotN cot))) n i
      = ∑ j : Fin nC, pdiv (fun b' : Vec nC => dense Wc b' a) bc i j * batchSlice N nC cot n j := by
  simp only [denStep]
  exact denseBiasGrad_correct Wc bc a (batchSlice N nC cot n) i

/-- A batched vector-LN γ gradient node, tied (`veclnGammaGradB_den`). -/
def VecLNGammaTiedB (N R : Nat) {D : Nat} (xN epsStr cotN : String) (ε : ℝ) (βv : Vec D)
    (x : Vec (N * (R * D))) (γ : Vec D) (dy : Vec (N * (R * D))) : Prop :=
  ∀ k : Fin D,
    den (SHlo.veclnGammaGradB (N := N) (R := R) (D := D) xN epsStr ε x (.operand cotN dy)) k
      = ∑ n : Fin N, ∑ o : Fin (R * D),
          pdiv (fun gv : Vec D =>
                  Mat.flatten (fun r =>
                    layerNormVec D ε gv βv (Mat.unflatten (batchSlice N (R * D) x n) r))) γ k o
            * batchSlice N (R * D) dy n o

/-- A batched vector-LN β gradient node, tied (`rowDenseBiasGradB_den_lnbeta`). -/
def VecLNBetaTiedB (N R : Nat) {D : Nat} (cotN : String) (ε : ℝ) (γv : Vec D)
    (x : Vec (N * (R * D))) (β : Vec D) (dy : Vec (N * (R * D))) : Prop :=
  ∀ i : Fin D,
    den (SHlo.rowDenseBiasGradB (N := N) (R := R) (c := D) (.operand cotN dy)) i
      = ∑ n : Fin N, ∑ o : Fin (R * D),
          pdiv (fun bv : Vec D =>
                  Mat.flatten (fun r =>
                    layerNormVec D ε γv bv (Mat.unflatten (batchSlice N (R * D) x n) r))) β i o
            * batchSlice N (R * D) dy n o

theorem vecLNGammaTiedB_holds {N R D : Nat} {xN epsStr cotN : String} {ε : ℝ} {βv : Vec D}
    {x : Vec (N * (R * D))} {γ : Vec D} {dy : Vec (N * (R * D))} :
    VecLNGammaTiedB N R xN epsStr cotN ε βv x γ dy := fun k =>
  veclnGammaGradB_den xN epsStr cotN ε βv x γ dy k

theorem vecLNBetaTiedB_holds {N R D : Nat} {cotN : String} {ε : ℝ} {γv : Vec D}
    {x : Vec (N * (R * D))} {β : Vec D} {dy : Vec (N * (R * D))} :
    VecLNBetaTiedB N R cotN ε γv x β dy := fun i =>
  rowDenseBiasGradB_den_lnbeta cotN ε γv (fun n => Mat.unflatten (batchSlice N (R * D) x n)) β dy i

end Proofs.ViTPoCGB

