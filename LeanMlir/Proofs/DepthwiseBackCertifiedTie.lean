import LeanMlir.Proofs.DepthwiseBackFloatBridge
import LeanMlir.Proofs.StridedConvBackFloatBridge
import LeanMlir.Proofs.IR

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
* `depthwiseFlatBack_eq_vjp_backward` — the stride-1 leaf tie: the float-bridge `depthwiseFlatBack W`
  (= `depthwiseFlat (dwReverse W) 0`) IS the certified depthwise input-VJP
  `(depthwiseFlat_has_vjp W b).backward x` (depthwise conv is linear ⇒ the saved activation `x` is
  ignored). The depthwise peer of `convFlatBack_eq_vjp_backward`.
* `depthwiseStride2FlatBack_eq_vjp_backward` — the strided leaf tie: `depthwiseStride2FlatBack W`
  (= `depthwiseFlatBack ∘ decimateBack`) IS `(depthwiseStride2Flat_has_vjp W b).backward x`. The
  depthwise peer of `flatConvStride2Back_eq_vjp_backward` (conv leaf + the `decimateBack` `rfl`).
-/

namespace Proofs

open Classical

/-- **The depthwise conv-adjoint identity (odd kernels), all dims.** The emitted reversed-kernel
    forward depthwise conv `depthwiseConv2d (dwReverse W) 0` equals the certified depthwise
    input-gradient `depthwiseConv2d_input_grad_formula W`, for arbitrary `c h w kH kW` with odd
    kernels. The depthwise twin of `IR.convBackDenote_eq_input_grad_formula`: per output coordinate
    both sides sum over the SAME valid alignments via `(kh,kw) ↦ (kh+hi-pH, kw+wi-pW)`; under oddness
    `2·pH = kH-1` the reversed-kernel index `kH-1-kh` matches the formula's `hi+pH-ho`. No `Σ co`
    (depthwise channel `ch` is fixed). `Finset.sum_bij'` over the pad-filtered supports; all index
    arithmetic by `omega`. The load-bearing leaf for the depthwise §B ties (convnext/mnv2/enet). -/
theorem depthwiseConv2d_dwReverse_eq_input_grad_formula {c h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : DepthwiseKernel c kH kW) (dy : Tensor3 c h w) :
    depthwiseConv2d (dwReverse W) (fun _ => 0) dy = depthwiseConv2d_input_grad_formula W dy := by
  funext ch hi wi
  simp only [depthwiseConv2d, dwReverse, Proofs.IR.kRev, zero_add, depthwiseConv2d_input_grad_formula]
  rw [← Finset.sum_product', ← Finset.sum_product', Finset.univ_product_univ,
      Finset.univ_product_univ]
  rw [← Finset.sum_subset (Finset.filter_subset
        (fun p : Fin kH × Fin kW => (kH-1)/2 ≤ p.1.val + hi.val ∧ p.1.val + hi.val - (kH-1)/2 < h ∧
             (kW-1)/2 ≤ p.2.val + wi.val ∧ p.2.val + wi.val - (kW-1)/2 < w) Finset.univ) ?lv,
      ← Finset.sum_subset (Finset.filter_subset
        (fun q : Fin h × Fin w => q.1.val ≤ hi.val + (kH-1)/2 ∧ hi.val + (kH-1)/2 - q.1.val < kH ∧
             q.2.val ≤ wi.val + (kW-1)/2 ∧ wi.val + (kW-1)/2 - q.2.val < kW) Finset.univ) ?rv]
  case lv =>
    intro p _ hp
    rw [Finset.mem_filter] at hp
    rw [dif_neg (fun hpr => hp ⟨Finset.mem_univ p, hpr⟩), mul_zero]
  case rv =>
    intro q _ hq
    rw [Finset.mem_filter] at hq
    rw [dif_neg (fun hpr => hq ⟨Finset.mem_univ q, hpr⟩)]
  refine Finset.sum_bij'
    (fun p hp => ((⟨p.1.val + hi.val - (kH-1)/2, by
        have := (Finset.mem_filter.mp hp).2; omega⟩ : Fin h),
       (⟨p.2.val + wi.val - (kW-1)/2, by
        have := (Finset.mem_filter.mp hp).2; omega⟩ : Fin w)))
    (fun q _ => ((⟨kH - 1 - (hi.val + (kH-1)/2 - q.1.val), by omega⟩ : Fin kH),
       (⟨kW - 1 - (wi.val + (kW-1)/2 - q.2.val), by omega⟩ : Fin kW)))
    ?hi ?hj ?linv ?rinv ?heq
  case hi =>
    intro p hp
    have hb := (Finset.mem_filter.mp hp).2
    have := p.1.isLt; have := p.2.isLt
    rw [Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_, ?_, ?_, ?_⟩ <;> simp only <;> omega
  case hj =>
    intro q hq
    have hb := (Finset.mem_filter.mp hq).2
    have := q.1.isLt; have := q.2.isLt
    rw [Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_, ?_, ?_, ?_⟩ <;> simp only <;> omega
  case linv =>
    intro p hp
    have hb := (Finset.mem_filter.mp hp).2
    have := p.1.isLt; have := p.2.isLt
    apply Prod.ext <;> apply Fin.ext <;> simp only <;> omega
  case rinv =>
    intro q hq
    have hb := (Finset.mem_filter.mp hq).2
    have := q.1.isLt; have := q.2.isLt
    apply Prod.ext <;> apply Fin.ext <;> simp only <;> omega
  case heq =>
    intro p hp
    have hb := (Finset.mem_filter.mp hp).2
    have h1 := p.1.isLt; have h2 := p.2.isLt
    rw [dif_pos hb, dif_pos (by refine ⟨?_, ?_, ?_, ?_⟩ <;> simp only <;> omega)]
    dsimp only
    have ea : kH - 1 - p.1.val = hi.val + (kH - 1) / 2 - (p.1.val + hi.val - (kH - 1) / 2) := by omega
    have eb : kW - 1 - p.2.val = wi.val + (kW - 1) / 2 - (p.2.val + wi.val - (kW - 1) / 2) := by omega
    simp only [ea, eb]

/-- **Depthwise conv input-VJP leaf tie.** The float-bridge `depthwiseFlatBack W` (= reversed-kernel
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

/-- **Strided depthwise conv input-VJP leaf tie.** `depthwiseStride2FlatBack W` (= `depthwiseFlatBack
    ∘ decimateBack`) IS the certified strided depthwise input-VJP `(depthwiseStride2Flat_has_vjp W
    b).backward x`, for odd kernels. Decomposes into the stride-1 depthwise leaf tie
    (`depthwiseFlatBack_eq_vjp_backward`) and the decimate leaf (`decimateBack_eq_vjp`, `rfl`),
    matching `depthwiseStride2Flat = decimateFlat ∘ depthwiseFlat`. The depthwise peer of
    `flatConvStride2Back_eq_vjp_backward`; unlocks the mnv2 stride-2 inverted-residual downsample. -/
theorem depthwiseStride2FlatBack_eq_vjp_backward {c h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (x : Vec (c * (2 * h) * (2 * w))) :
    depthwiseStride2FlatBack (h := h) (w := w) W
      = (depthwiseStride2Flat_has_vjp W b).backward x := by
  funext dy
  show depthwiseFlatBack (h := 2 * h) (w := 2 * w) W (decimateBack c h w dy) = _
  rw [depthwiseFlatBack_eq_vjp_backward hkH hkW W b x]
  rfl

end Proofs
