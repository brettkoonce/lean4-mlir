/-
**Mixed-precision CONVOLUTION forward error** — the accuracy half of the bf16 conv ops in
`Proofs.StableHLO` (`BatchableOp.convBf16`, `flatConvFBf16`, and the dgrad/wgrad twins).

`FloatBridge` stops at dense: it has `dot_close_mixed` / `dense_close_mixed` and no conv peer,
and `planning/bf16_renderer.md` calls that gap "the single biggest item in this document",
because every net where bf16 pays is conv-dominated.

▶ It turns out not to be a new hard theorem. A convolution output **is** a dot product over its
flattened receptive field (`conv2d_eq_flat_dot`), so the whole thing is
`dot_close_mixed_uniform` instantiated at fan-in `ic·kH·kW`, plus one leaf rounding for the
bf16 STORE and one accumulate rounding for the bias add.

⚠ The store term is the part that has no analogue in `dense_close_mixed`, and it is forced by
the hardware rather than chosen: a conv with bf16 operands and an f32-typed result has its
casts deleted by XLA and runs entirely in fp32 (measured — see `BatchableOp.convBf16`). The
only emit shape that reaches the tensor cores gives the convolution a **bf16-typed result**, so
the accumulator is rounded on store and the error model has to say so.

⭐ **Non-vacuous, and the fan-in is not what costs.** `convBr` at fp32 accumulate
(`M.u = 2⁻²⁴`) and bf16 leaf (`L.u = 2⁻⁸`), evaluated at ResNet-34's own layers — arithmetic
outside Lean, quoted as illustration, not proved here:

    layer                  fan-in n   fan-in term   leaf term   convBr
    stem 7×7, ic=3              147     8.9e-06     7.83e-03    0.0078
    stage-1 3×3, ic=64          576     3.5e-05     7.83e-03    0.0079
    stage-4 3×3, ic=512        4608     2.8e-04     7.83e-03    0.0081
    1×1 projection, ic=512      512     3.1e-05     7.83e-03    0.0079

Under 1% everywhere, and the LEAF term dominates the fan-in term by ~28× even at the widest
layer. Set the accumulate to bf16 as well and `((1+u)^(n+1) − 1)` at n=4608 is **6.4e7** — a
bound that says nothing at all. That contrast is the whole argument for bf16-mixed over bf16:
the `1/u` fan-in wall sits at the ACCUMULATE precision, which stays fp32.
-/
import LeanMlir.Proofs.Float.FloatBridge
import LeanMlir.Proofs.Architectures.CNN

open Finset BigOperators
namespace Proofs

/-- Summing a flattened `Tensor3` is the triple sum. -/
theorem Tensor3.sum_flatten {c h w : Nat} (T : Tensor3 c h w) :
    ∑ k, Tensor3.flatten T k = ∑ i, ∑ j, ∑ l, T i j l := by
  rw [← Equiv.sum_comp (finProdFinEquiv (m := c*h) (n := w))]
  rw [Fintype.sum_prod_type]
  rw [← Equiv.sum_comp (finProdFinEquiv (m := c) (n := h))]
  rw [Fintype.sum_prod_type]
  simp only [Tensor3.flatten, Equiv.symm_apply_apply]

/-- Flattening commutes with a pointwise product: both sides look up the same index. -/
theorem Tensor3.flatten_mul {c h w : Nat} (A B : Tensor3 c h w) (k : Fin (c*h*w)) :
    Tensor3.flatten A k * Tensor3.flatten B k
      = Tensor3.flatten (fun i j l => A i j l * B i j l) k := rfl

/-- The `kH × kW` receptive field `conv2d` reads at output pixel `(hi, wi)`, zero outside —
    `conv2d`'s own `if hpad …` branch, lifted out so the conv's fan-in is a `Tensor3`.

    ⚠⚠ **The `3` suffix is NOT decoration — it is what makes this file importable.**
    `SgdDescentCnn.lean` already declares `Proofs.convWindow` for the SAME receptive field at
    the FLAT type `Vec (ic*kH*kW)`. Two constants cannot share a full name, so while this one
    was also called `convWindow` the two could not coexist in one environment: `lake build
    LeanMlir` failed outright at
    `import … ConvMixedFloatBridge failed, environment already contains 'Proofs.convWindow'`,
    which is exactly the import a whole-net bound has to make. ▶ The `Tensor3` shape is
    deliberate and stays — `conv2d_eq_flat_dot` needs `Tensor3.sum_flatten` — so the name
    moved rather than the type. -/
noncomputable def convWindow3 {ic h w : Nat} (kH kW : Nat) (x : Tensor3 ic h w)
    (hi : Fin h) (wi : Fin w) : Tensor3 ic kH kW :=
  fun c kh kw =>
    let pH := (kH - 1) / 2
    let pW := (kW - 1) / 2
    let hh := kh.val + hi.val
    let ww := kw.val + wi.val
    if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
      x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
    else 0

/-- The output channel's kernel slice, as a `Tensor3`. -/
noncomputable def convSlice {ic oc kH kW : Nat} (W : Kernel4 oc ic kH kW) (o : Fin oc) :
    Tensor3 ic kH kW := fun c kh kw => W o c kh kw

/-- ⭐ **A convolution output is a DOT PRODUCT** of length `ic·kH·kW` over the flattened
    receptive field. This is the whole reason `conv_close_mixed` is not a new hard theorem:
    it lets the conv reuse `dot_close_mixed_uniform` at that fan-in. -/
theorem conv2d_eq_flat_dot {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Tensor3 ic h w) (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    conv2d W b x o hi wi
      = b o + ∑ k, Tensor3.flatten (convSlice W o) k
                   * Tensor3.flatten (convWindow3 kH kW x hi wi) k := by
  simp only [conv2d]
  congr 1
  simp only [Tensor3.flatten_mul]
  rw [Tensor3.sum_flatten]
  rfl

end Proofs

namespace Proofs

/-- The Higham-style bracket `dot_close_mixed_uniform` produces at fan-in `n`: fan-in
    amplification rides the ACCUMULATE precision `M.u`, the leaf precision contributes a flat
    per-leaf term. -/
noncomputable def convBr (M L : FloatModel) (n : Nat) : ℝ :=
  ((1 + M.u) ^ (n + 1) - 1) * (1 + L.u) ^ 2 + (2 * L.u + L.u ^ 2)

/-- `Σ|kernel·window|` over the receptive field — the magnitude the bound scales. -/
noncomputable def convFanS {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (x : Tensor3 ic h w) (o : Fin oc) (hi : Fin h) (wi : Fin w) : ℝ :=
  ∑ k, |Tensor3.flatten (convSlice W o) k * Tensor3.flatten (convWindow3 kH kW x hi wi) k|

namespace FloatModel
variable (M : FloatModel)

/-- **The mixed-precision convolution, as the emitted graph computes it.** Operands rounded to
    the leaf precision `L` and accumulated at `M` (`dotMixed`), the accumulator then rounded to
    `L` again — the **bf16-typed result**, i.e. the store — and only then the bias added at `M`.

    ⚠ The second `L.rnd` is what distinguishes this from `denseMixed`, and it is not optional:
    `BatchableOp.convBf16` must give the convolution a bf16-typed result or XLA deletes the
    casts and runs the whole conv in f32. The store is a consequence of the only emit shape
    that reaches the tensor cores, so the error model has to carry it. -/
noncomputable def convMixed (L : FloatModel) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Tensor3 ic h w) : Tensor3 oc h w :=
  fun o hi wi =>
    M.add (L.rnd (M.dotMixed L (Tensor3.flatten (convSlice W o))
                               (Tensor3.flatten (convWindow3 kH kW x hi wi)))) (b o)

/-- ⭐⭐ **Mixed-precision convolution forward error.** Three terms, one per rounding the
    emitted graph performs: the dot (`convBr`, fan-in `ic·kH·kW`), the bf16 STORE of the
    accumulator (`L.u`), and the f32 bias add (`M.u`).

    ▶ It is `dot_close_mixed_uniform` instantiated at the conv's fan-in, because a convolution
    output IS a dot product over its flattened receptive field (`conv2d_eq_flat_dot`). The
    fan-in wall therefore still sits at `1/M.u = 2²⁴` and not at the leaf precision — the same
    reason bf16-mixed is non-vacuous for dense. -/
theorem conv_close_mixed (L : FloatModel) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Tensor3 ic h w)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    |M.convMixed L W b x o hi wi - conv2d W b x o hi wi| ≤
      M.u * ((1 + L.u) * (1 + convBr M L (ic*kH*kW)) * convFanS W x o hi wi + |b o|)
        + L.u * (1 + convBr M L (ic*kH*kW)) * convFanS W x o hi wi
        + convBr M L (ic*kH*kW) * convFanS W x o hi wi := by
  have hMu := M.u_nonneg
  have hLu := L.u_nonneg
  set ker := Tensor3.flatten (convSlice W o) with hker
  set win := Tensor3.flatten (convWindow3 kH kW x hi wi) with hwin
  set S := convFanS W x o hi wi with hS
  set br := convBr M L (ic*kH*kW) with hbr
  set p := M.dotMixed L ker win with hp
  set P := ∑ k, ker k * win k with hP
  have hS0 : (0:ℝ) ≤ S := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hbr0 : (0:ℝ) ≤ br := by
    have h1 : (0:ℝ) ≤ (1 + M.u) ^ (ic*kH*kW + 1) - 1 :=
      sub_nonneg.mpr (one_le_pow₀ (by linarith))
    have h2 : (0:ℝ) ≤ (1 + L.u) ^ 2 := sq_nonneg _
    have h3 : (0:ℝ) ≤ 2 * L.u + L.u ^ 2 := by nlinarith
    simp only [hbr, convBr]; nlinarith
  -- the dot
  have hD : |p - P| ≤ br * S := by
    have h := M.dot_close_mixed_uniform L ker win
    simpa [hp, hP, hS, hbr, convBr, convFanS, hker, hwin] using h
  have hPS : |P| ≤ S := by
    simpa [hP, hS, convFanS, hker, hwin] using Finset.abs_sum_le_sum_abs (fun k => ker k * win k) _
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
  have hadd : |M.rnd (L.rnd p + b o) - (L.rnd p + b o)| ≤ M.u * |L.rnd p + b o| := M.err _
  have hy : |L.rnd p + b o| ≤ (1 + L.u) * (1 + br) * S + |b o| := by
    have := abs_add_le (L.rnd p) (b o); linarith
  rw [conv2d_eq_flat_dot]
  show |M.add (L.rnd p) (b o) - (b o + P)| ≤ _
  simp only [FloatModel.add]
  have hsplit : |M.rnd (L.rnd p + b o) - (b o + P)|
      ≤ |M.rnd (L.rnd p + b o) - (L.rnd p + b o)| + |L.rnd p - P| := by
    have : M.rnd (L.rnd p + b o) - (b o + P)
         = (M.rnd (L.rnd p + b o) - (L.rnd p + b o)) + (L.rnd p - P) := by ring
    rw [this]; exact abs_add_le _ _
  have hMy : M.u * |L.rnd p + b o| ≤ M.u * ((1 + L.u) * (1 + br) * S + |b o|) :=
    mul_le_mul_of_nonneg_left hy hMu
  linarith

end FloatModel
end Proofs
