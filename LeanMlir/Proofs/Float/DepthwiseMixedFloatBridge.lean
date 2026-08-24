import LeanMlir.Proofs.Float.FloatBridge
import LeanMlir.Proofs.Architectures.Depthwise
import LeanMlir.Proofs.Float.DepthwiseFloatBridge

/-! # `depthwise_close_mixed` — the bf16-mixed DEPTHWISE convolution against exact ℝ

The depthwise peer of `ConvMixedFloatBridge.conv_close_mixed`, and — as
`planning/bf16_renderer.md` §10.2 predicted — it is the same instantiation at a much smaller
fan-in, not a new result. **A depthwise output is a dot product of length `kH·kW`**: one channel,
no `ic` sum (`depthwiseConv2d_eq_dw_dot`). So it is `dot_close_mixed_uniform` at that fan-in, plus
one leaf rounding for the bf16 store and one accumulate rounding for the bias — three terms, one
per rounding the emit performs.

⭐ **The fan-in shrinks from thousands to NINE and the bound barely moves, which is the whole
point of the §9.3 separation.** At `u_acc = 2⁻²⁴` / `u_leaf = 2⁻⁸`:

    layer                              fan-in n   fan-in term   leaf term   dwBr
    depthwise 3×3 (every MNv2 block)          9     6.01e-07    7.83e-03    0.0078
    R50 3×3, ic=512 (for contrast)         4608     2.77e-04    7.83e-03    0.0081

The fan-in term drops by 461× and `dwBr` moves by 3.5% — the fan-in rides the ACCUMULATE
precision, which stays fp32, while the flat leaf term is what actually costs. ▶ A depthwise layer
is not more accurate than a dense conv in bf16 in any way that matters; it is the same 0.8%.

⚠ Like `conv_close_mixed`, this bounds ONE layer against exact ℝ at an exactly-represented input.
Composition needs the error-modulus form — see `ConvMixedComposeBridge` for the conv version of
that argument, which transfers verbatim because `FloatClose` is precision- and layer-agnostic.
-/

namespace Proofs

open Finset BigOperators

/-- Channel `ch`'s flattened filter as a plain `Vec (kH·kW)` — `dwKernelMat`'s single column.

    ⚠⚠ **`dwWindow`, `dwKernelMat` and `depthwiseConv2d_eq_dense` are REUSED from
    `DepthwiseFloatBridge`, not rebuilt here.** The first draft of this file defined its own
    `dwWindow` with the identical type and meaning, and `lake build LeanMlir` refused the import
    (`environment already contains 'Proofs.dwWindow'`) — the same collision that had `lake build
    LeanMlir` broken for three commits over `Proofs.convWindow`. ▶ When a `dw*`/`conv*` helper
    seems to be missing, grep before defining: on this evidence it usually already exists. -/
noncomputable def dwSlice {c kH kW : Nat} (W : DepthwiseKernel c kH kW) (ch : Fin c) :
    Vec (kH * kW) :=
  fun idx => dwKernelMat W ch idx 0

/-- ⭐ **A depthwise output IS a dot product** of length `kH·kW` — one channel, no `ic` sum.
    `depthwiseConv2d_eq_dense` with `dense` unfolded; the depthwise peer of `conv2d_eq_flat_dot`,
    and the reason §10.2's "expect it to be easier" was right. -/
theorem depthwiseConv2d_eq_dw_dot {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) (ch : Fin c) (hi : Fin h) (wi : Fin w) :
    depthwiseConv2d W b x ch hi wi
      = (∑ k, dwWindow kH kW x ch hi wi k * dwSlice W ch k) + b ch := by
  rw [depthwiseConv2d_eq_dense]; rfl

/-- The Higham bracket at the depthwise fan-in — `convBr`'s peer, with `n = kH·kW`. -/
noncomputable def dwBr (M L : FloatModel) (n : Nat) : ℝ :=
  ((1 + M.u) ^ (n + 1) - 1) * (1 + L.u) ^ 2 + (2 * L.u + L.u ^ 2)

/-- `Σ|kernel·window|` over the receptive field — the magnitude the bound scales. -/
noncomputable def dwFanS {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (x : Tensor3 c h w) (ch : Fin c) (hi : Fin h) (wi : Fin w) : ℝ :=
  ∑ k, |dwWindow kH kW x ch hi wi k * dwSlice W ch k|

namespace FloatModel
variable (M : FloatModel)

/-- **The mixed-precision depthwise convolution, as the emitted graph computes it.** Operands
    rounded to the leaf precision `L` and accumulated at `M`, the accumulator then rounded to `L`
    again — the **bf16-typed result**, i.e. the store — and only then the bias added at `M`.

    ⚠ The second `L.rnd` is not optional: `BatchableOp.depthwiseBf16` must give the convolution a
    bf16-typed result or XLA deletes the casts and cuDNN gets f32 parameters. Measured on a real
    MNv2 layer (c=144, 56², fgc=144) — grouping buys no exemption from §9.2. -/
noncomputable def depthwiseMixed (L : FloatModel) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (x : Tensor3 c h w) : Tensor3 c h w :=
  fun ch hi wi =>
    M.add (L.rnd (M.dotMixed L (dwWindow kH kW x ch hi wi) (dwSlice W ch))) (b ch)

/-- ⭐⭐ **Mixed-precision DEPTHWISE forward error.** Three terms, one per rounding the emitted
    graph performs: the dot (`dwBr`, fan-in `kH·kW`), the bf16 STORE of the accumulator (`L.u`),
    and the f32 bias add (`M.u`). Structurally identical to `conv_close_mixed`; only the fan-in
    differs, which is exactly what §10.2 predicted. -/
theorem depthwise_close_mixed (L : FloatModel) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (x : Tensor3 c h w)
    (ch : Fin c) (hi : Fin h) (wi : Fin w) :
    |M.depthwiseMixed L W b x ch hi wi - depthwiseConv2d W b x ch hi wi| ≤
      M.u * ((1 + L.u) * (1 + dwBr M L (kH*kW)) * dwFanS W x ch hi wi + |b ch|)
        + L.u * (1 + dwBr M L (kH*kW)) * dwFanS W x ch hi wi
        + dwBr M L (kH*kW) * dwFanS W x ch hi wi := by
  have hMu := M.u_nonneg
  have hLu := L.u_nonneg
  set win := dwWindow kH kW x ch hi wi with hwin
  set ker := dwSlice W ch with hker
  set S := dwFanS W x ch hi wi with hS
  set br := dwBr M L (kH*kW) with hbr
  set p := M.dotMixed L win ker with hp
  set P := ∑ k, win k * ker k with hP
  have hS0 : (0:ℝ) ≤ S := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hbr0 : (0:ℝ) ≤ br := by
    have h1 : (0:ℝ) ≤ (1 + M.u) ^ (kH*kW + 1) - 1 :=
      sub_nonneg.mpr (one_le_pow₀ (by linarith))
    have h2 : (0:ℝ) ≤ (1 + L.u) ^ 2 := sq_nonneg _
    have h3 : (0:ℝ) ≤ 2 * L.u + L.u ^ 2 := by nlinarith
    simp only [hbr, dwBr]; nlinarith
  -- the dot
  have hD : |p - P| ≤ br * S := by
    have h := M.dot_close_mixed_uniform L win ker
    simpa [hp, hP, hS, hbr, dwBr, dwFanS, hker, hwin] using h
  have hPS : |P| ≤ S := by
    simpa [hP, hS, dwFanS, hker, hwin] using Finset.abs_sum_le_sum_abs (fun k => win k * ker k) _
  have hpb : |p| ≤ (1 + br) * S := by
    have := abs_sub_abs_le_abs_sub p P
    nlinarith [abs_nonneg p, abs_nonneg P]
  -- the store
  have hstore : |L.rnd p - p| ≤ L.u * |p| := L.err p
  have hLp : |L.rnd p| ≤ (1 + L.u) * (1 + br) * S := by
    have h1 : |L.rnd p| ≤ |L.rnd p - p| + |p| := by simpa using abs_sub_le (L.rnd p) p 0
    nlinarith [abs_nonneg p]
  have hLpP : |L.rnd p - P| ≤ L.u * (1 + br) * S + br * S := by
    have h : |L.rnd p - P| ≤ |L.rnd p - p| + |p - P| := abs_sub_le _ _ _
    nlinarith [abs_nonneg p]
  -- the bias add, at the accumulate precision
  have hadd : |M.rnd (L.rnd p + b ch) - (L.rnd p + b ch)| ≤ M.u * |L.rnd p + b ch| := M.err _
  have hy : |L.rnd p + b ch| ≤ (1 + L.u) * (1 + br) * S + |b ch| := by
    have := abs_add_le (L.rnd p) (b ch); linarith
  rw [depthwiseConv2d_eq_dw_dot]
  show |M.add (L.rnd p) (b ch) - (P + b ch)| ≤ _
  simp only [FloatModel.add]
  have hsplit : |M.rnd (L.rnd p + b ch) - (P + b ch)|
      ≤ |M.rnd (L.rnd p + b ch) - (L.rnd p + b ch)| + |L.rnd p - P| := by
    have : M.rnd (L.rnd p + b ch) - (P + b ch)
         = (M.rnd (L.rnd p + b ch) - (L.rnd p + b ch)) + (L.rnd p - P) := by ring
    rw [this]; exact abs_add_le _ _
  have hMy : M.u * |L.rnd p + b ch| ≤ M.u * ((1 + L.u) * (1 + br) * S + |b ch|) :=
    mul_le_mul_of_nonneg_left hy hMu
  linarith

end FloatModel
end Proofs
