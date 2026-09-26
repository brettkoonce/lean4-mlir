import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Foundation.BatchedStageLayers

/-! # Conv / dense / GAP leaf ties — each per-op backward map IS the certified VJP

The hand-composed backward chains (`ResNetBackChains`, the MobileNetV2 / EfficientNet / ConvNeXt /
ViT whole-back ties) are written in the per-op backward maps of `BackwardMaps`. This file ties each
conv-family leaf to its certified VJP, so the whole-net ties close on named maps:

| leaf | theorem |
|---|---|
| conv (odd kernel), reversed-kernel conv = input-VJP | `convFlatBack_eq_vjp_backward` |
| stride-2 conv, symmetric padding / XLA-`SAME` | `flatConvStride2Back_eq_vjp_backward` / `flatConvStride2XlaBack_eq_vjp_backward` |
| dense, `Wᵀ·dy` | `dense_transpose_eq_vjp_backward` |
| global average pool, broadcast ÷ | `gapBack_eq_vjp_backward` |
| batched conv-BN-relu stage at batch BN, stride 2 (symmetric) / stride 1 | `cbReluStridedBBack_eq_vjp_backward` / `cbReluBBack_eq_vjp_backward` |

The depthwise twins are in `DepthwiseBackCertifiedTie`, the even-kernel conv in
`EvenKernelConvBack`, the 3×3/s2 max-pool leaf (`maxPool3s2FlatBack_eq_vjp_backward`) in
`BackwardMaps`. -/

namespace Proofs


/-- **Conv input-VJP leaf tie.** The backward map `convFlatBack W` (= reversed-kernel forward
    conv) IS the certified conv input-VJP `(flatConvHasVJP W b).backward x` (conv is linear,
    so the saved activation `x` is ignored), for odd kernels. Routes through the general
    `IR.convBackDenote_eq_input_grad_formula`; the leaf every conv slot of the ResNet chains reduces to. -/
theorem convFlatBack_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Vec (ic * h * w)) :
    convFlatBack (h := h) (w := w) W = (flatConvHasVJP W b).backward x := by
  funext dy
  simp only [convFlatBack, flatConv, flatConvHasVJP, HasVJP3.toHasVJP, conv2dHasVJP3]
  rw [IR.convBackDenote_eq_input_grad_formula hkH hkW W (Tensor3.unflatten dy)]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The strided-conv leaf ties — symmetric and XLA-`SAME` stride-2
-- ════════════════════════════════════════════════════════════════

/-- **Strided conv input-VJP leaf tie.** `flatConvStride2Back W` (= `convFlatBack ∘ decimateBack`)
    IS the certified strided conv input-VJP `(flatConvStride2HasVJP W b).backward x`, for odd
    kernels. Decomposes into the conv leaf tie (`convFlatBack_eq_vjp_backward`) and the decimate
    leaf (`decimateBack_eq_vjp`, `rfl`), matching `flatConvStride2 = decimateFlat ∘ flatConv`. -/
theorem flatConvStride2Back_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w))) :
    flatConvStride2Back (h := h) (w := w) W = (flatConvStride2HasVJP W b).backward x := by
  funext dy
  show convFlatBack (h := 2*h) (w := 2*w) W (decimateBack oc h w dy) = _
  rw [convFlatBack_eq_vjp_backward hkH hkW W b x]
  rfl

/-- **XLA-`SAME` strided conv input-VJP leaf tie.** `flatConvStride2XlaBack W`
    (= `convFlatBack ∘ decimateOddBack`) IS the certified `(flatConvStride2XlaHasVJP W b).backward x`,
    for odd kernels: the conv leaf tie and the odd-scatter leaf (`decimateOddBack_eq_vjp`, `rfl`),
    matching `flatConvStride2Xla = decimateOddFlat ∘ flatConv`. The TF-origin stems' (B0,
    MobileNetV2) leaf. This is the theorem that fixes the odd-phase backward's DIRECTION: the
    emitted transposed-conv pad `[p+1, p-1]` (opposite to the weight grads' `[p-1, p+1]`) denotes
    this map, so a backward derived "by symmetry" with the weight grads cannot be tied here. -/
theorem flatConvStride2XlaBack_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w))) :
    flatConvStride2XlaBack (h := h) (w := w) W = (flatConvStride2XlaHasVJP W b).backward x := by
  funext dy
  show convFlatBack (h := 2*h) (w := 2*w) W (decimateOddBack oc h w dy) = _
  rw [convFlatBack_eq_vjp_backward hkH hkW W b x]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The ENDPOINT leaf ties — dense head, GAP
--   (the stem's strided conv is `flatConvStride2Back_eq_vjp_backward` above)
-- ════════════════════════════════════════════════════════════════

/-- **Dense head input-VJP leaf tie.** The chain's dense backward `dense (Wᵀ) 0` (= `Wᵀ·dy`)
    IS the certified dense input-VJP `(denseHasVJP W b).backward x` (= `Mat.mulVec W dy`); dense
    is linear in its input, so the activation `x` is ignored. One `mul_comm` per term. -/
theorem dense_transpose_eq_vjp_backward {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m) :
    dense (Mat.transpose W) (0 : Vec m) = (denseHasVJP W b).backward x := by
  funext dy i
  simp only [dense, denseHasVJP, Mat.transpose, Mat.mulVec, Pi.zero_apply, add_zero]
  exact Finset.sum_congr rfl fun j _ => mul_comm _ _

/-- **GAP input-VJP leaf tie.** The backward map `gapBack c h w` (broadcast `dy(channel)/(h·w)`)
    IS the certified GAP input-VJP `(globalAvgPoolFlatHasVJP c h w).backward x` — definitionally
    the same broadcast-÷ map (the VJP ignores its primal argument). -/
theorem gapBack_eq_vjp_backward (c h w : Nat) (x : Vec (c * h * w)) :
    gapBack c h w = (globalAvgPoolFlatHasVJP c h w).backward x := rfl

/-- **The conv-BN-relu STAGE tie.** The hand-written
    `batchMap (flatConvStride2Back) ∘ bnBack ∘ reluMaskBack` IS `cbReluStridedB`'s certified
    backward at a smooth point. One `rw` of the odd-kernel strided leaf tie, then `rfl` — the
    stage's VJP is `vjpCompAt`-built so its backward is already the composition, and the conv
    leaf's backward is input-independent (a convolution is linear), so the row-wise `batchMap`
    lift matches at every saved input.

    SYMMETRIC padding (`flatConvStride2Back`), not the XLA-`SAME` phase B0's and MobileNetV2's
    stems take. Identical types, different certificates. -/
theorem cbReluStridedBBack_eq_vjp_backward {N ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hrelu : ∀ k, StableHLO.bnBatchLA N oc h w ε γ β
      (StableHLO.batchMap N (flatConvStride2 Ws bs) x) k ≠ 0) :
    (StableHLO.batchMap N (flatConvStride2Back (h := h) (w := w) Ws)
        ∘ (bnBatchLAHasVJP N oc h w ε hε γ β).backward
            (StableHLO.batchMap N (flatConvStride2 Ws bs) x)
        ∘ reluMaskBack (fun i => StableHLO.bnBatchLA N oc h w ε γ β
            (StableHLO.batchMap N (flatConvStride2 Ws bs) x) i > 0))
      = (StableHLO.cbReluStridedBHasVJPAt N Ws bs ε hε γ β x hrelu).backward := by
  rw [flatConvStride2Back_eq_vjp_backward hkH hkW Ws bs (fun _ => 0)]
  rfl

/-- **The HEAD CONV tie.** `batchMap (convFlatBack) ∘ bnBack ∘ reluMaskBack` IS `cbReluB`'s
    certified backward at a smooth point — the stride-1, plain-convolution peer of the stem's.

    The kernel extent is a binder, so a 1×1 conv-BN-relu is one instance of this stage and not
    a new one: MobileNetV4's two head convs (`%h1W` 256 → 960, `%hW` 960 → 1280) tie by applying
    this twice, and its head is therefore not hypothesis-free the way ResNet-34's is. -/
theorem cbReluBBack_eq_vjp_backward {N ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (v : Vec (N * (ic * h * w)))
    (hs : ∀ k, StableHLO.bnBatchLA N oc h w ε γ β
           (StableHLO.batchMap N (flatConv W b) v) k ≠ 0) :
    (StableHLO.batchMap N (convFlatBack (h := h) (w := w) W)
        ∘ (bnBatchLAHasVJP N oc h w ε hε γ β).backward
            (StableHLO.batchMap N (flatConv W b) v)
        ∘ reluMaskBack (fun i => StableHLO.bnBatchLA N oc h w ε γ β
            (StableHLO.batchMap N (flatConv W b) v) i > 0))
      = (StableHLO.cbReluBHasVJPAt N W b ε hε γ β v hs).backward := by
  rw [convFlatBack_eq_vjp_backward hkH hkW W b (fun _ => 0)]
  rfl

end Proofs
