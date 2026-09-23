import LeanMlir.Proofs.Nets.Small.CifarCNN
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetChainClose
import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Nets.ResNet.ResNetBackChains
import LeanMlir.Proofs.Foundation.IR

/-! # The ResNet leaf ties — the per-op backward maps ARE the certified VJPs

The hand-composed backward chains of `ResNetBackChains.lean` are written in the per-op backward
maps of `BackwardMaps.lean`. This file ties each ResNet leaf to its certified VJP, so that the
whole-net ties (`ResNet34BackCertifiedTieB`, `ResNet50WholeBackCertifiedTieB`) close on named
maps rather than on re-spellings:

* `convFlatBack_eq_vjp_backward` — the conv leaf: `convFlatBack W` (reversed-kernel conv) IS the
  certified conv input-VJP, via the general odd-kernel `IR.convBackDenote_eq_input_grad_formula`;
* `flatConvStride2Back_eq_vjp_backward` / `flatConvStride2XlaBack_eq_vjp_backward` — the two
  stride-2 leaves (symmetric padding, and the XLA-`SAME` twin the TF-origin stems take), each the
  conv leaf plus the `decimateBack` `rfl`;
* `dense_transpose_eq_vjp_backward` (the dense head, `Wᵀ·dy` = certified `Mat.mulVec W`) and
  `gapBack_eq_vjp_backward` (GAP broadcast-÷, `rfl`) — the endpoints.

The 3×3/s2 stem pool's leaf is `maxPool3s2FlatBack_eq_vjp_backward` (`BackwardMaps.lean`), and
`ResNetBackChains.lean` equates the render's scatter with the chain's.

History: this file also held the per-example ResNet-34 tier — the per-channel-BN block VJPs, the
stem leaf and the whole-net fold of the per-example chain onto `resnet34_has_vjp_at`'s backward
(closed 2026-09-03, at the forward the retired `ResNet34Render.lean` emitted). That tie is what
found the pool drift: the chain reversed the 2×2 pool while the committed forward pooled 3×3/s2,
and the missing leaf became `maxPool3s2FlatBack`. The tier was retired on 2026-09-19 once
`ResNet34BackCertifiedTieB` stated the same result at the batch-BN net every shipped artifact
runs. -/

namespace Proofs


/-- **Conv input-VJP leaf tie.** The backward map `convFlatBack W` (= reversed-kernel forward
    conv) IS the certified conv input-VJP `(flatConv_has_vjp W b).backward x` (conv is linear,
    so the saved activation `x` is ignored), for odd kernels. Routes through the general
    `IR.convBackDenote_eq_input_grad_formula`; the leaf every conv slot of the ResNet chains reduces to. -/
theorem convFlatBack_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Vec (ic * h * w)) :
    convFlatBack (h := h) (w := w) W = (flatConv_has_vjp W b).backward x := by
  funext dy
  simp only [convFlatBack, flatConv, flatConv_has_vjp, hasVJP3_to_hasVJP, conv2d_has_vjp3]
  rw [IR.convBackDenote_eq_input_grad_formula hkH hkW W (Tensor3.unflatten dy)]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The strided-conv leaf ties — symmetric and XLA-`SAME` stride-2
-- ════════════════════════════════════════════════════════════════

/-- **Strided conv input-VJP leaf tie.** `flatConvStride2Back W` (= `convFlatBack ∘ decimateBack`)
    IS the certified strided conv input-VJP `(flatConvStride2_has_vjp W b).backward x`, for odd
    kernels. Decomposes into the conv leaf tie (`convFlatBack_eq_vjp_backward`) and the decimate
    leaf (`decimateBack_eq_vjp`, `rfl`), matching `flatConvStride2 = decimateFlat ∘ flatConv`. -/
theorem flatConvStride2Back_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w))) :
    flatConvStride2Back (h := h) (w := w) W = (flatConvStride2_has_vjp W b).backward x := by
  funext dy
  show convFlatBack (h := 2*h) (w := 2*w) W (decimateBack oc h w dy) = _
  rw [convFlatBack_eq_vjp_backward hkH hkW W b x]
  rfl

/-- **XLA-`SAME` strided conv input-VJP leaf tie.** `flatConvStride2XlaBack W`
    (= `convFlatBack ∘ decimateOddBack`) IS the certified `(flatConvStride2Xla_has_vjp W b).backward x`,
    for odd kernels: the conv leaf tie and the odd-scatter leaf (`decimateOddBack_eq_vjp`, `rfl`),
    matching `flatConvStride2Xla = decimateOddFlat ∘ flatConv`. The TF-origin stems' (B0,
    MobileNetV2) leaf. ⚠ This is the theorem that fixes the odd-phase backward's DIRECTION: the
    emitted transposed-conv pad `[p+1, p-1]` (opposite to the weight grads' `[p-1, p+1]`) denotes
    this map, so a backward derived "by symmetry" with the weight grads cannot be tied here. -/
theorem flatConvStride2XlaBack_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w))) :
    flatConvStride2XlaBack (h := h) (w := w) W = (flatConvStride2Xla_has_vjp W b).backward x := by
  funext dy
  show convFlatBack (h := 2*h) (w := 2*w) W (decimateOddBack oc h w dy) = _
  rw [convFlatBack_eq_vjp_backward hkH hkW W b x]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The ENDPOINT leaf ties — dense head, GAP
--   (the stem's strided conv is `flatConvStride2Back_eq_vjp_backward` above)
-- ════════════════════════════════════════════════════════════════

/-- **Dense head input-VJP leaf tie.** The chain's dense backward `dense (Wᵀ) 0` (= `Wᵀ·dy`)
    IS the certified dense input-VJP `(dense_has_vjp W b).backward x` (= `Mat.mulVec W dy`), conv is
    linear so the activation `x` is ignored. One `mul_comm` per term. -/
theorem dense_transpose_eq_vjp_backward {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m) :
    dense (Mat.transpose W) (0 : Vec m) = (dense_has_vjp W b).backward x := by
  funext dy i
  simp only [dense, dense_has_vjp, Mat.transpose, Mat.mulVec, Pi.zero_apply, add_zero]
  exact Finset.sum_congr rfl fun j _ => mul_comm _ _

/-- **GAP input-VJP leaf tie.** The backward map `gapBack c h w` (broadcast `dy(channel)/(h·w)`)
    IS the certified GAP input-VJP `(globalAvgPoolFlat_has_vjp c h w).backward x` — definitionally
    the same broadcast-÷ map (the VJP ignores its primal argument). -/
theorem gapBack_eq_vjp_backward (c h w : Nat) (x : Vec (c * h * w)) :
    gapBack c h w = (globalAvgPoolFlat_has_vjp c h w).backward x := rfl

end Proofs
