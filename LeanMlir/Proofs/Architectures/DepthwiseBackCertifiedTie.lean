import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Foundation.IR

/-! # §B shared prerequisite: the DEPTHWISE adjoint gate + leaf ties

The §B certified-VJP ties for the three CNNs (convnext / mnv2 / efficientnet) all reverse a
**depthwise** convolution, so they all need the depthwise twin of the conv adjoint gate
`IR.convBackDenote_eq_input_grad_formula`. This file builds it once.

* `depthwiseConv2d_dwReverse_eq_input_grad_formula` — the gate: the emitted reversed-kernel forward
  depthwise conv `depthwiseConv2d (dwReverse W) 0` equals the certified depthwise input-gradient
  `depthwiseConv2d_input_grad_formula W`, for arbitrary dims with odd kernels. The exact depthwise
  analogue of the conv gate — same `(kh,kw) ↦ (kh+hi-pH, kw+wi-pW)` partial bijection on the pad
  supports, MINUS the `Σ co` channel sum (depthwise has no cross-channel mixing, so the input channel
  reads only from its own kernel/gradient channel `ch`).
* `depthwiseFlatBack_eq_vjp_backward` — the stride-1 leaf tie: the backward map `depthwiseFlatBack W`
  (= `depthwiseFlat (dwReverse W) 0`) IS the certified depthwise input-VJP
  `(depthwiseFlat_has_vjp W b).backward x` (depthwise conv is linear ⇒ the saved activation `x` is
  ignored). The depthwise peer of `convFlatBack_eq_vjp_backward`.
* `depthwiseStride2FlatXlaBack_eq_vjp_backward` — the XLA-`SAME` strided leaf tie:
  `depthwiseStride2FlatXlaBack W` (= `depthwiseFlatBack ∘ decimateOddBack`) IS
  `(depthwiseStride2FlatXla_has_vjp W b).backward x`, for odd kernels.
-/

namespace Proofs

/-- **The depthwise conv-adjoint identity (odd kernels), all dims.** The emitted reversed-kernel
    forward depthwise conv `depthwiseConv2d (dwReverse W) 0` equals the certified depthwise
    input-gradient `depthwiseConv2d_input_grad_formula W`, for arbitrary `c h w kH kW` with odd
    kernels. The depthwise twin of `IR.convBackDenote_eq_input_grad_formula`: no `Σ co` (depthwise
    channel `ch` is fixed), so per output coordinate it is `IR.reverseSlab_eq_gradSlab` at the
    channel's own slabs `W ch`, `dy ch`. The leaf the depthwise §B ties (convnext/mnv2/enet) stand
    on. -/
theorem depthwiseConv2d_dwReverse_eq_input_grad_formula {c h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : DepthwiseKernel c kH kW) (dy : Tensor3 c h w) :
    depthwiseConv2d (dwReverse W) (fun _ => 0) dy = depthwiseConv2d_input_grad_formula W dy := by
  funext ch hi wi
  simp only [depthwiseConv2d, dwReverse, zero_add, depthwiseConv2d_input_grad_formula]
  exact IR.reverseSlab_eq_gradSlab hkH hkW (W ch) (dy ch) hi wi

/-- **Depthwise conv input-VJP leaf tie.** The backward map `depthwiseFlatBack W` (= reversed-kernel
    forward depthwise conv) IS the certified depthwise input-VJP `(depthwiseFlat_has_vjp W b).backward
    x` (depthwise conv is linear, so the saved activation `x` is ignored), for odd kernels. Routes
    through `depthwiseConv2d_dwReverse_eq_input_grad_formula`; the depthwise peer of
    `convFlatBack_eq_vjp_backward`. -/
theorem depthwiseFlatBack_eq_vjp_backward {c h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (x : Vec (c * h * w)) :
    depthwiseFlatBack (h := h) (w := w) W = (depthwiseFlat_has_vjp W b).backward x := by
  funext dy
  simp only [depthwiseFlatBack, depthwiseFlat, depthwiseFlat_has_vjp, hasVJP3_to_hasVJP,
    depthwise_has_vjp3]
  rw [depthwiseConv2d_dwReverse_eq_input_grad_formula hkH hkW W (Tensor3.unflatten dy)]
  rfl

/-- **XLA-`SAME` strided depthwise input-VJP leaf tie.** `depthwiseStride2FlatXlaBack W`
    (= `depthwiseFlatBack ∘ decimateOddBack`) IS the certified
    `(depthwiseStride2FlatXla_has_vjp W b).backward x`, for odd kernels. MobileNetV2's four
    strided depthwises and B0's downsample depthwise, at the TF-origin convention. -/
theorem depthwiseStride2FlatXlaBack_eq_vjp_backward {c h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (x : Vec (c * (2 * h) * (2 * w))) :
    depthwiseStride2FlatXlaBack (h := h) (w := w) W
      = (depthwiseStride2FlatXla_has_vjp W b).backward x := by
  funext dy
  show depthwiseFlatBack (h := 2 * h) (w := 2 * w) W (decimateOddBack c h w dy) = _
  rw [depthwiseFlatBack_eq_vjp_backward hkH hkW W b x]
  rfl

end Proofs
