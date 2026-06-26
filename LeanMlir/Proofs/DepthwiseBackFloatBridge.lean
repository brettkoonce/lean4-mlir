import LeanMlir.Proofs.DepthwiseFloatBridge
import LeanMlir.Proofs.StridedConvBackFloatBridge

/-! # ℝ→Float32 bridge for the DEPTHWISE-conv backward (mnv2 / efficientnet / convnext)

A3 (planning/a3_backward_deepnet_assembly.md §1e): the depthwise input-VJP, the conv-backward
analogue for the depthwise stage. Exactly as the regular conv backward `convFlatBack W =
flatConv (reverseSwap W) 0` reused the forward conv bridge **for free**
(`CnnBackFloatBridge.floatBridges_convBack`), the depthwise backward is a *forward depthwise conv at
the spatially-reversed kernel*:

  `dx[c,h,w] = Σ_{kh,kw} W[c, kH−1−kh, kW−1−kw] · dy[c, h+kh−p, w+kw−p]`
            = `depthwiseConv2d (dwReverse W) 0 dy`

(the MLIR-aligned reversed-kernel formula `depthwise_has_vjp3`'s `depthwiseConv2d_input_grad_formula`
denotes; the codegen emits `convolution(dy, reverse(W))` with `feature_group_count = c`, i.e. NO
channel transpose since depthwise has no cross-channel mixing — only the two spatial axes reverse).
So `depthwiseFlatBack W = depthwiseFlat (dwReverse W) 0` float-bridges via the forward
`floatBridges_depthwise` at the reversed kernel (`|dwReverse W| = |W|`): **no new proof** — the
depthwise twin of `floatBridges_convBack`.

The **strided** depthwise backward (mnv2's stride-2 inverted-residual downsample) is then
`depthwiseFlatBack ∘ decimateBack` — the zero-upsample scatter (`StridedConvBackFloatBridge`,
exact, modulus `id`) then the reversed-kernel depthwise conv — the depthwise twin of
`floatBridges_flatConvStride2Back`. Both `.comp` over already-bridged pieces.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The spatially-reversed depthwise kernel
-- ════════════════════════════════════════════════════════════════

/-- **Spatial reversal of a depthwise kernel** — reverse both spatial axes (`kRev k = kH−1−k`),
    keeping the channel axis (depthwise has no cross-channel mixing, so no transpose, unlike the
    regular conv's `reverseSwap`). The kernel the codegen feeds to the backward depthwise
    `stablehlo.convolution` (`reverse [2,3]`, `feature_group_count = c`). -/
noncomputable def dwReverse {c kH kW : Nat} (W : DepthwiseKernel c kH kW) :
    DepthwiseKernel c kH kW :=
  fun ch kh kw => W ch (Proofs.IR.kRev kh) (Proofs.IR.kRev kw)

/-- Reversing the spatial axes leaves the uniform kernel-magnitude bound unchanged. -/
theorem dwReverse_abs_le {c kH kW : Nat} {W : DepthwiseKernel c kH kW} {w' : ℝ}
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w') (ch : Fin c) (kh : Fin kH) (kw : Fin kW) :
    |dwReverse W ch kh kw| ≤ w' := hW ch (Proofs.IR.kRev kh) (Proofs.IR.kRev kw)

-- ════════════════════════════════════════════════════════════════
-- § Depthwise input-VJP: a reversed-kernel forward depthwise conv (FREE)
-- ════════════════════════════════════════════════════════════════

/-- **Depthwise conv backward in flat `Vec` space** — `dx = depthwise_has_vjp3.backward W dy`. The
    emitted reversed-kernel depthwise convolution denotes a forward `depthwiseConv2d (dwReverse W) 0`,
    which in flat space is `depthwiseFlat (dwReverse W) 0`. The backward of `depthwiseFlat W b`
    (`Vec (c·h·w) → Vec (c·h·w)`, channels preserved). -/
noncomputable def depthwiseFlatBack {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) :
    Vec (c * h * w) → Vec (c * h * w) :=
  depthwiseFlat (h := h) (w := w) (dwReverse W) (fun _ => 0)

/-- **The depthwise conv input-VJP float-bridges** — `depthwiseFlatBack W = depthwiseFlat
    (dwReverse W) 0`, so this is `floatBridges_depthwise` at the spatially-reversed kernel
    (`|dwReverse W ch kh kw| = |W ch (kRev kh) (kRev kw)| ≤ w'`) with zero bias. No new proof — the
    depthwise analogue of `floatBridges_convBack`; the budget rides the depthwise fan-in `kH·kW`. -/
theorem floatBridges_depthwiseBack {c h w kH kW : Nat} (M : FloatModel)
    (W : DepthwiseKernel c kH kW) {w' : ℝ} (hw' : 0 ≤ w') (hn : 0 < c * h * w)
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w') :
    FloatBridges (depthwiseFlatBack (h := h) (w := w) W) := by
  unfold depthwiseFlatBack
  exact floatBridges_depthwise (h := h) (w := w) M (dwReverse W) (fun _ => 0)
    hw' le_rfl hn (fun ch kh kw => dwReverse_abs_le hW ch kh kw) (fun _ => by simp)

-- ════════════════════════════════════════════════════════════════
-- § The strided depthwise backward: `depthwiseFlatBack ∘ decimateBack`
-- ════════════════════════════════════════════════════════════════

/-- **Strided (stride-2) depthwise conv backward in flat `Vec` space** — the input-VJP of
    `depthwiseStride2Flat W b = decimateFlat ∘ depthwiseFlat`: zero-upsample the cotangent
    (`decimateBack`, channels preserved), then run the reversed-kernel depthwise conv
    (`depthwiseFlatBack`). `Vec (c·h·w) → Vec (c·2h·2w)`. The depthwise twin of
    `flatConvStride2Back`. -/
noncomputable def depthwiseStride2FlatBack {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) :
    Vec (c * h * w) → Vec (c * (2 * h) * (2 * w)) :=
  depthwiseFlatBack (h := 2 * h) (w := 2 * w) W ∘ decimateBack c h w

/-- **The strided depthwise input-VJP float-bridges.** One `.comp`: the decimation scatter
    (`floatBridges_decimateBack`, exact, modulus `id`) then the reversed-kernel depthwise conv
    (`floatBridges_depthwiseBack`). The strided sibling of `floatBridges_depthwiseBack`; unlocks the
    mnv2 stride-2 inverted-residual downsample blocks. -/
theorem floatBridges_depthwiseStride2Back {c h w kH kW : Nat} (M : FloatModel)
    (W : DepthwiseKernel c kH kW) {w' : ℝ} (hw' : 0 ≤ w') (hn : 0 < c * (2 * h) * (2 * w))
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w') :
    FloatBridges (depthwiseStride2FlatBack (h := h) (w := w) W) := by
  unfold depthwiseStride2FlatBack
  exact (floatBridges_decimateBack c h w).comp
    (floatBridges_depthwiseBack (h := 2 * h) (w := 2 * w) M W hw' hn hW)

end Proofs
