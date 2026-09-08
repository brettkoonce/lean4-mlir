import LeanMlir.Proofs.Architectures.ConvNeXtFaithfulPoCGB

/-! # The bf16 gradient nodes — every `*GradBBf16` kind the suite emits, folded once

A bf16 render does NOT consume the f32 gradient node. It emits its own `*GradBBf16` constructor,
whose `den` rounds the operands going in and — for every kind but one — rounds the result ONCE,
outside the batch sum: the emitted convolution contracts the batch inside a single op and stores
its bf16 result once, so a rounding per summand would claim a coarser computation than the
hardware performs. That is a different real number from the f32 node's, and it needs its own
certificate. "The bf16 twins consume the same node" was written in three fold headers and is
false.

This file is the whole bf16 op table, stated per op kind: each lemma says the node denotes the
certified `Σ_n` gradient at the ROUNDED operands, rounded. The proofs are the f32 fold's — `congr 1`
peels the outer rounding, `Finset.sum_congr` the batch, and the inner equality is the per-example
certificate at rounded slices.

| kind | f32 peer | emitted by |
|---|---|---|
| `convWeightGradBBf16` | `ResNet34PoCB.convWGradB_den` | every net |
| `convStridedWeightGradBBf16` (symmetric) | `ResNet34PoCB.convStridedWGradB_den` | ResNet-34/50, MobileNetV4's fused stage, ConvNeXt's downsamples |
| `convStridedXlaWeightGradBBf16` (XLA-`SAME`) | `EnetPoCG.convStridedXlaWGradB_den` | EfficientNet-B0's, MobileNetV2's and MobileNetV4's stems |
| `convStride4WeightGradBBf16` | `CnxPoCGB.psWGradB_den` | ConvNeXt's patchify stem |
| `depthwiseWeightGradBBf16` | `EnetPoCG.depthwiseWGradB_den` | B0, MobileNetV2, MobileNetV4, ConvNeXt |
| `depthwiseStridedWeightGradBBf16` (symmetric) | `EnetPoCG.depthwiseStridedWGradB_den` | B0, MobileNetV4 |
| `depthwiseStridedXlaWeightGradBBf16` (XLA-`SAME`) | `Mnv2PaperPoCG.depthwiseStridedXlaWGradB_den` | MobileNetV2 |
| `rowDenseWeightGradBBf16` | `ViTPoCGB.rowDenseWeightGradB_den` | ViT's Q/K/V/O and MLP denses |
| `patchEmbedWeightGradBBf16` | `ViTPoCGB.patchEmbedWeightGradB_den` | ViT's patch embed |

⚠ **`rowDenseWeightGradBBf16` has NO outer rounding**, and that is the measurement rather than
an omission: its `dot_general` contracts batch and token in one op and keeps its f32-typed result
deliberately (`StableHLO.lean`'s constructor says why), so only the two leaf reads round.

⚠ Padding rides along invisibly, as in the f32 folds: the symmetric and XLA-`SAME` strided kinds
have identical types and identical emitted shapes, and only the certificate tells them apart.

⛔ BatchNorm, LayerNorm and the dense head have no bf16 twin here or anywhere: every bf16 net in the
suite keeps them in f32, so their γ/β and weight nodes are the f32 folds' in both worlds. -/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.Bf16PoC

open scoped BigOperators

/-- **bf16 conv weight GRADIENT denotes the certified `Σ_n` weight gradient at the rounded
    operands, rounded once.** Every 1×1 and 3×3 in every bf16 artifact. -/
theorem convWGradBBf16_den {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (xN cotN : String)
    (b : Vec oc) (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convWeightGradBBf16 rnd xN b x W (.operand cotN cot)) idx
      = rnd (∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                    (Tensor3.unflatten (fun j => rnd (batchSlice N (ic * h * w) x n j)))))
               (Kernel4.flatten W) idx j * rnd (batchSlice N (oc * h * w) cot n j)) := by
  simp only [den]
  congr 1
  apply Finset.sum_congr rfl
  intro n _
  exact conv_weight_grad_bridge b
    (Tensor3.unflatten (fun j => rnd (batchSlice N (ic * h * w) x n j)))
    (Kernel4.flatten W) (fun j => rnd (batchSlice N (oc * h * w) cot n j)) idx

/-- **bf16 SYMMETRIC strided conv weight GRADIENT**, rounded once. ResNet's downsamples and 7×7
    stem, MobileNetV4's fused 3×3/s2, ConvNeXt's 2×2/s2 downsamples. -/
theorem convStridedWGradBBf16_den {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (xN cotN : String)
    (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStridedWeightGradBBf16 rnd xN b x W (.operand cotN cot)) idx
      = rnd (∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  flatConvStride2 (Kernel4.unflatten v') b
                    (fun j => rnd (batchSlice N (ic * (2 * h) * (2 * w)) x n j)))
               (Kernel4.flatten W) idx j * rnd (batchSlice N (oc * h * w) cot n j)) := by
  simp only [den]
  congr 1
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride2_weight_grad_has_vjp b
    (fun j => rnd (batchSlice N (ic * (2 * h) * (2 * w)) x n j))).correct
    (Kernel4.flatten W) (fun j => rnd (batchSlice N (oc * h * w) cot n j)) idx

/-- **bf16 XLA-`SAME` strided conv weight GRADIENT**, rounded once. The TF-origin stems
    (EfficientNet-B0, MobileNetV2, MobileNetV4). ⚠ `flatConvStride2Xla`, not `flatConvStride2`. -/
theorem convStridedXlaWGradBBf16_den {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (xN cotN : String)
    (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStridedXlaWeightGradBBf16 rnd xN b x W (.operand cotN cot)) idx
      = rnd (∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  flatConvStride2Xla (Kernel4.unflatten v') b
                    (fun j => rnd (batchSlice N (ic * (2 * h) * (2 * w)) x n j)))
               (Kernel4.flatten W) idx j * rnd (batchSlice N (oc * h * w) cot n j)) := by
  simp only [den]
  congr 1
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride2Xla_weight_grad_has_vjp b
    (fun j => rnd (batchSlice N (ic * (2 * h) * (2 * w)) x n j))).correct
    (Kernel4.flatten W) (fun j => rnd (batchSlice N (oc * h * w) cot n j)) idx

/-- **bf16 4×4/s4 patchify-stem weight GRADIENT**, rounded once. ConvNeXt's stem. -/
theorem convStride4WGradBBf16_den {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (xN cotN : String)
    (b : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStride4WeightGradBBf16 rnd xN b x W (.operand cotN cot)) idx
      = rnd (∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  flatConvStride4 (Kernel4.unflatten v') b
                    (fun j => rnd (batchSlice N (ic * (2 * (2 * h)) * (2 * (2 * w))) x n j)))
               (Kernel4.flatten W) idx j * rnd (batchSlice N (oc * h * w) cot n j)) := by
  simp only [den]
  congr 1
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride4_weight_grad_has_vjp b
    (fun j => rnd (batchSlice N (ic * (2 * (2 * h)) * (2 * (2 * w))) x n j))).correct
    (Kernel4.flatten W) (fun j => rnd (batchSlice N (oc * h * w) cot n j)) idx

/-- **bf16 depthwise weight GRADIENT**, rounded once. Every stride-1 depthwise. -/
theorem depthwiseWGradBBf16_den {N c h w kH kW : Nat} (rnd : ℝ → ℝ) (xN cotN : String)
    (b : Vec c) (x : Vec (N * (c * h * w))) (W : DepthwiseKernel c kH kW)
    (cot : Vec (N * (c * h * w))) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseWeightGradBBf16 rnd xN b x W (.operand cotN cot)) idx
      = rnd (∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b
                    (Tensor3.unflatten (fun j => rnd (batchSlice N (c * h * w) x n j)))))
               (Tensor3.flatten W) idx j * rnd (batchSlice N (c * h * w) cot n j)) := by
  simp only [den]
  congr 1
  apply Finset.sum_congr rfl
  intro n _
  rw [← (hasVJP3_to_hasVJP (depthwise_weight_grad_has_vjp3 b
      (Tensor3.unflatten (fun j => rnd (batchSlice N (c * h * w) x n j))))).correct
      (Tensor3.flatten W) (fun j => rnd (batchSlice N (c * h * w) cot n j)) idx]
  simp only [hasVJP3_to_hasVJP, Tensor3.flatten, Tensor3.unflatten_flatten]

/-- **bf16 SYMMETRIC strided depthwise weight GRADIENT**, rounded once. B0's stride-2 MBConvs and
    MobileNetV4's rows 1, 3, 11. -/
theorem depthwiseStridedWGradBBf16_den {N c h w kH kW : Nat} (rnd : ℝ → ℝ) (xN cotN : String)
    (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    (cot : Vec (N * (c * h * w))) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseStridedWeightGradBBf16 rnd xN b x W (.operand cotN cot)) idx
      = rnd (∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  depthwiseStride2Flat (Tensor3.unflatten v') b
                    (fun j => rnd (batchSlice N (c * (2 * h) * (2 * w)) x n j)))
               (Tensor3.flatten W) idx j * rnd (batchSlice N (c * h * w) cot n j)) := by
  simp only [den]
  congr 1
  apply Finset.sum_congr rfl
  intro n _
  exact (depthwiseStride2_weight_grad_has_vjp b
    (fun j => rnd (batchSlice N (c * (2 * h) * (2 * w)) x n j))).correct
    (Tensor3.flatten W) (fun j => rnd (batchSlice N (c * h * w) cot n j)) idx

/-- **bf16 XLA-`SAME` strided depthwise weight GRADIENT**, rounded once. MobileNetV2's four
    stride-2 depthwises, and no other net's. ⚠ `depthwiseStride2FlatXla`. -/
theorem depthwiseStridedXlaWGradBBf16_den {N c h w kH kW : Nat} (rnd : ℝ → ℝ) (xN cotN : String)
    (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    (cot : Vec (N * (c * h * w))) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseStridedXlaWeightGradBBf16 rnd xN b x W (.operand cotN cot)) idx
      = rnd (∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  depthwiseStride2FlatXla (Tensor3.unflatten v') b
                    (fun j => rnd (batchSlice N (c * (2 * h) * (2 * w)) x n j)))
               (Tensor3.flatten W) idx j * rnd (batchSlice N (c * h * w) cot n j)) := by
  simp only [den]
  congr 1
  apply Finset.sum_congr rfl
  intro n _
  exact (depthwiseStride2Xla_weight_grad_has_vjp b
    (fun j => rnd (batchSlice N (c * (2 * h) * (2 * w)) x n j))).correct
    (Tensor3.flatten W) (fun j => rnd (batchSlice N (c * h * w) cot n j)) idx

/-- **bf16 per-token dense weight GRADIENT at the rounded operands — with NO outer rounding**:
    the emitted `dot_general` keeps its f32-typed result, so only the two leaf reads round.
    ViT's `Wq/Wk/Wv/Wo/Wfc1/Wfc2`. -/
theorem rowDenseWGradBBf16_den {N tk a c : Nat} (rnd : ℝ → ℝ) (xN cotN : String)
    (bb : Vec c) (x : Vec (N * (tk * a))) (W : Mat a c) (dy : Vec (N * (tk * c)))
    (i : Fin a) (j : Fin c) :
    den (SHlo.rowDenseWeightGradBBf16 (N := N) (tk := tk) (a := a) (c := c) rnd xN x
          (.operand cotN dy))
        (finProdFinEquiv (i, j))
      = ∑ n : Fin N, ∑ o : Fin (tk * c),
          pdiv (fun v : Vec (a * c) =>
                  Mat.flatten (fun r =>
                    dense (Mat.unflatten v) bb
                      (fun k => rnd (Mat.unflatten (batchSlice N (tk * a) x n) r k))))
               (Mat.flatten W) (finProdFinEquiv (i, j)) o
            * rnd (batchSlice N (tk * c) dy n o) := by
  simp only [den, Mat.flatten, Equiv.symm_apply_apply]
  apply Finset.sum_congr rfl
  intro n _
  exact vit_rowDenseW_grad_bridge bb
    (fun r k => rnd (Mat.unflatten (batchSlice N (tk * a) x n) r k)) W
    (fun o => rnd (batchSlice N (tk * c) dy n o)) i j

/-- **bf16 patch-embed conv weight GRADIENT at the rounded operands, rounded once.** ViT's
    16×16/s16 stem; the outer `rnd` wraps the whole batch sum because the emit contracts the
    batch inside one bf16-typed convolution. -/
theorem patchEmbedWGradBBf16_den {ic H W P tk D N : Nat} (rnd : ℝ → ℝ) (xN cotN : String)
    (bc cls : Vec D) (pos : Mat (tk + 1) D) (img : Vec (N * (ic * H * W)))
    (Wp : Kernel4 D ic P P) (dy : Vec (N * ((tk + 1) * D)))
    (d : Fin D) (c : Fin ic) (kh kw : Fin P) :
    den (SHlo.patchEmbedWeightGradBBf16 (N := N) (ic := ic) (H := H) (W := W) (P := P)
            (tk := tk) (D := D) rnd xN img (.operand cotN dy))
        (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw))
      = rnd (∑ n : Fin N, ∑ o : Fin ((tk + 1) * D),
          pdiv (fun v : Vec (D * ic * P * P) =>
                  patchEmbed_flat ic H W P tk D (Kernel4.unflatten v) bc cls pos
                    (fun j => rnd (batchSlice N (ic * H * W) img n j)))
            (Kernel4.flatten Wp)
            (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw)) o
            * rnd (batchSlice N ((tk + 1) * D) dy n o)) := by
  simp only [den, patchEmbedWeightGradFlat, Kernel4.flatten, Equiv.symm_apply_apply]
  congr 1
  apply Finset.sum_congr rfl
  intro n _
  exact vit_patchW_grad_bridge Wp bc cls pos
    (fun j => rnd (batchSlice N (ic * H * W) img n j))
    (fun j => rnd (batchSlice N ((tk + 1) * D) dy n j)) d c kh kw

end Proofs.Bf16PoC
