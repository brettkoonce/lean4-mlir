import LeanMlir.Proofs.Float.DepthwiseFloatBridge
import LeanMlir.Proofs.Float.ConvMixedFloatBridge

/-! # `depthwise_close_mixed` — the bf16-mixed DEPTHWISE convolution against exact ℝ

The depthwise peer of `FloatModel.conv_close_mixed`: the same instantiation at a much smaller
fan-in, not a new result. **A depthwise output is a dot product of length `kH·kW`**: one channel,
no `ic` sum (`depthwiseConv2d_eq_dw_dot`). So it is `dot_close_mixed_uniform` at that fan-in, plus
one leaf rounding for the bf16 store and one accumulate rounding for the bias — three terms, one
per rounding the emit performs.

**The fan-in shrinks from thousands to nine and the bound barely moves**, because the fan-in
rides the accumulate roundoff and the leaf roundoff enters flat. At `u_acc = 2⁻²⁴` /
`u_leaf = 2⁻⁸` (arithmetic outside Lean, quoted as illustration):

    layer                              fan-in n   fan-in term   leaf term   dwBr
    depthwise 3×3 (every MNv2 block)          9     6.01e-07    7.83e-03    0.0078
    R50 3×3, ic=512 (for contrast)         4608     2.77e-04    7.83e-03    0.0081

The fan-in term drops by 461× and `dwBr` moves by 3.5% — the fan-in rides the ACCUMULATE
precision, which stays fp32, while the flat leaf term is what actually costs. A depthwise layer
is not more accurate than a dense conv in bf16 in any way that matters; it is the same 0.8%.

Note: Like `conv_close_mixed`, this bounds ONE layer against exact ℝ at an exactly-represented input.
Composition needs the error-modulus form — see `ConvMixedComposeBridge` for the conv version of
that argument, which transfers verbatim because `FloatClose` is precision- and layer-agnostic.
-/

namespace Proofs

open Finset BigOperators

/-- Channel `ch`'s flattened filter as a plain `Vec (kH·kW)` — `dwKernelMat`'s single column. -/
-- Note: `dwWindow`, `dwKernelMat` and `depthwiseConv2d_eq_dense` are reused from
-- `DepthwiseFloatBridge`, not rebuilt here. Redefining one of them (same name, same type) makes
-- `lake build LeanMlir` fail on the import with "environment already contains"; when a
-- `dw*`/`conv*` helper seems to be missing, grep before defining.
noncomputable def dwSlice {c kH kW : Nat} (W : DepthwiseKernel c kH kW) (ch : Fin c) :
    Vec (kH * kW) :=
  fun idx => dwKernelMat W ch idx 0

/-- **A depthwise output IS a dot product** of length `kH·kW` — one channel, no `ic` sum.
    `depthwiseConv2d_eq_dense` with `dense` unfolded; the depthwise peer of `conv2d_eq_flat_dot`. -/
theorem depthwiseConv2d_eq_dw_dot {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) (ch : Fin c) (hi : Fin h) (wi : Fin w) :
    depthwiseConv2d W b x ch hi wi
      = (∑ k, dwWindow kH kW x ch hi wi k * dwSlice W ch k) + b ch := by
  rw [depthwiseConv2d_eq_dense]; rfl

/-- The Higham bracket at the depthwise fan-in — `convBr` itself, with `n = kH·kW`. -/
noncomputable def dwBr (M L : FloatModel) (n : Nat) : ℝ := convBr M L n

/-- `Σ|kernel·window|` over the receptive field — the magnitude the bound scales. -/
noncomputable def dwFanS {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (x : Tensor3 c h w) (ch : Fin c) (hi : Fin h) (wi : Fin w) : ℝ :=
  ∑ k, |dwWindow kH kW x ch hi wi k * dwSlice W ch k|

namespace FloatModel
variable (M : FloatModel)

/-- **The mixed-precision depthwise convolution, as the emitted graph computes it.** Operands
    rounded to the leaf precision `L` and accumulated at `M`, the accumulator then rounded to `L`
    again — the **bf16-typed result**, i.e. the store — and only then the bias added at `M`.

    Note: The second `L.rnd` is not optional: `BatchableOp.depthwiseBf16` must give the convolution a
    bf16-typed result or XLA deletes the casts and cuDNN gets f32 parameters. Measured on a real
    MNv2 layer (c=144, 56², fgc=144) — grouping buys no exemption from that. -/
noncomputable def depthwiseMixed (L : FloatModel) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (x : Tensor3 c h w) : Tensor3 c h w :=
  fun ch hi wi =>
    M.add (L.rnd (M.dotMixed L (dwWindow kH kW x ch hi wi) (dwSlice W ch))) (b ch)

/-- **Mixed-precision DEPTHWISE forward error.** Three terms, one per rounding the emitted
    graph performs: the dot (`dwBr`, fan-in `kH·kW`), the bf16 STORE of the accumulator (`L.u`),
    and the f32 bias add (`M.u`). Structurally identical to `conv_close_mixed`; only the fan-in
    differs. -/
theorem depthwise_close_mixed (L : FloatModel) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (x : Tensor3 c h w)
    (ch : Fin c) (hi : Fin h) (wi : Fin w) :
    |M.depthwiseMixed L W b x ch hi wi - depthwiseConv2d W b x ch hi wi| ≤
      M.u * ((1 + L.u) * (1 + dwBr M L (kH*kW)) * dwFanS W x ch hi wi + |b ch|)
        + L.u * (1 + dwBr M L (kH*kW)) * dwFanS W x ch hi wi
        + dwBr M L (kH*kW) * dwFanS W x ch hi wi := by
  rw [depthwiseConv2d_eq_dw_dot]
  exact M.storeBias_close L _ _ (b ch)

end FloatModel
end Proofs
