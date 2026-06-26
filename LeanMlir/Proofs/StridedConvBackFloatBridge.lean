import LeanMlir.Proofs.CnnBackFloatBridge
import LeanMlir.Proofs.ResNet34

/-! # ℝ→Float32 bridge for the STRIDED-conv backward (r34 down-blocks + stem)

The stride-2 conv decomposes as `flatConvStride2 = decimateFlat ∘ flatConv` (StridedConv.lean), so
its input-gradient VJP is, by the chain rule, `flatConv.back ∘ decimate.back` — "zero-upsample the
cotangent, then run the reversed-kernel conv" (StableHLO `lhs_dilation=[2,2]`). The reversed-kernel
conv backward is already bridged (`floatBridges_convBack`); the new piece is the **decimation VJP**:
the zero-upsampling **scatter** that routes each cotangent cell back to its even spatial position
and fills zeros elsewhere.

`decimateBack` = the certified `decimateFlat_has_vjp.backward` (a `Σ (if idx = decimateIdx k then 1
else 0)·dy k`). Because `decimateIdx` is **injective** (distinct output cells → distinct even
positions), each output index receives from at most one input cell — so the scatter is exact in
float (pure data movement, like `gather`/`reluMaskBack`/`maxPoolBack`), magnitude-nonincreasing, and
1-Lipschitz (modulus `id`). The strided-conv backward then float-bridges as
`floatBridges_convBack.comp floatBridges_decimateBack`.

This unlocks the r34 **down-blocks** (strided body + projection) and the **stem** (7×7 stride-2),
the pieces `Resnet34BackFloatBridge`'s identity-block left open.
-/

namespace Proofs

open Proofs.IR

-- ════════════════════════════════════════════════════════════════
-- § The decimation VJP: the zero-upsampling scatter
--   (`decimateIdx_injective` is reused from `ResNet34`)
-- ════════════════════════════════════════════════════════════════

/-- **Decimation backward (zero-upsampling scatter)** — the certified `decimateFlat` VJP: route
    `dy k` to the even position `decimateIdx k`, 0 elsewhere. `Vec (oc·h·w) → Vec (oc·2h·2w)`. -/
noncomputable def decimateBack (oc h w : Nat) (dy : Vec (oc * h * w)) :
    Vec (oc * (2 * h) * (2 * w)) :=
  fun idx => ∑ k : Fin (oc * h * w), (if idx = decimateIdx oc h w k then (1 : ℝ) else 0) * dy k

/-- `decimateBack` is exactly the certified `decimateFlat` VJP backward (faithful by definition —
    the VJP backward ignores its primal argument). -/
theorem decimateBack_eq_vjp (oc h w : Nat) (v : Vec (oc * (2 * h) * (2 * w)))
    (dy : Vec (oc * h * w)) :
    decimateBack oc h w dy = (decimateFlat_has_vjp oc h w).backward v dy := rfl

/-- The scatter as a filtered sum: `decimateBack dy idx = Σ_{k : idx = decimateIdx k} dy k`. -/
theorem decimateBack_eq_filter (oc h w : Nat) (dy : Vec (oc * h * w))
    (idx : Fin (oc * (2 * h) * (2 * w))) :
    decimateBack oc h w dy idx
      = ∑ k ∈ Finset.univ.filter (fun k => idx = decimateIdx oc h w k), dy k := by
  unfold decimateBack
  rw [Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro k _
  by_cases hc : idx = decimateIdx oc h w k <;> simp [hc]

/-- At most one input cell scatters to a given output index (`decimateIdx` injective). -/
theorem decimateBack_filter_card_le (oc h w : Nat) (idx : Fin (oc * (2 * h) * (2 * w))) :
    (Finset.univ.filter (fun k => idx = decimateIdx oc h w k)).card ≤ 1 :=
  Finset.card_le_one.mpr (fun x hx y hy => by
    simp only [Finset.mem_filter] at hx hy
    exact decimateIdx_injective oc h w (hx.2.symm.trans hy.2))

/-- A real input index exists whenever an output index does (the dims are linked by ×4). -/
private theorem decimate_dom_pos {oc h w : Nat} (idx : Fin (oc * (2 * h) * (2 * w))) :
    0 < oc * h * w := by
  have hbig : 0 < oc * (2 * h) * (2 * w) := (Nat.zero_le _).trans_lt idx.isLt
  have h4 : oc * (2 * h) * (2 * w) = 4 * (oc * h * w) := by ring
  omega

/-- **Decimation backward is `FloatClose` with modulus `id`** — exact in float (real = float map,
    pure data movement), magnitude-nonincreasing and 1-Lipschitz (each output cell receives from at
    most one input cell, by `decimateIdx` injectivity). The scatter peer of `floatClose_gather`. -/
theorem floatClose_decimateBack (oc h w : Nat) (A : ℝ) :
    FloatClose A A (decimateBack oc h w) (decimateBack oc h w) (id : ℝ → ℝ) := by
  refine ⟨fun v hv idx => ?_, fun vt va e _ _ hd idx => ?_⟩
  · have hpos := decimate_dom_pos idx
    have hAnn : 0 ≤ A := (abs_nonneg _).trans (hv ⟨0, hpos⟩)
    have hcard := decimateBack_filter_card_le oc h w idx
    have hb : |decimateBack oc h w v idx| ≤ A := by
      rw [decimateBack_eq_filter]
      calc |∑ k ∈ Finset.univ.filter (fun k => idx = decimateIdx oc h w k), v k|
          ≤ ∑ k ∈ Finset.univ.filter (fun k => idx = decimateIdx oc h w k), |v k| :=
            Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ _k ∈ Finset.univ.filter (fun k => idx = decimateIdx oc h w k), A :=
            Finset.sum_le_sum (fun k _ => hv k)
        _ = (Finset.univ.filter (fun k => idx = decimateIdx oc h w k)).card * A := by
            rw [Finset.sum_const, nsmul_eq_mul]
        _ ≤ 1 * A := by
            apply mul_le_mul_of_nonneg_right _ hAnn; exact_mod_cast hcard
        _ = A := one_mul A
    exact ⟨hb, hb⟩
  · have hpos := decimate_dom_pos idx
    have he : 0 ≤ e := (abs_nonneg _).trans (hd ⟨0, hpos⟩)
    have hcard := decimateBack_filter_card_le oc h w idx
    rw [decimateBack_eq_filter, decimateBack_eq_filter, ← Finset.sum_sub_distrib]
    calc |∑ k ∈ Finset.univ.filter (fun k => idx = decimateIdx oc h w k), (vt k - va k)|
        ≤ ∑ k ∈ Finset.univ.filter (fun k => idx = decimateIdx oc h w k), |vt k - va k| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _k ∈ Finset.univ.filter (fun k => idx = decimateIdx oc h w k), e :=
          Finset.sum_le_sum (fun k _ => hd k)
      _ = (Finset.univ.filter (fun k => idx = decimateIdx oc h w k)).card * e := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ 1 * e := by apply mul_le_mul_of_nonneg_right _ he; exact_mod_cast hcard
      _ = e := one_mul e

/-- Decimation backward float-bridges (magnitude-stable, exact). -/
theorem floatBridges_decimateBack (oc h w : Nat) : FloatBridges (decimateBack oc h w) :=
  fun A hA => ⟨A, _, _, hA, floatClose_decimateBack oc h w A⟩

-- ════════════════════════════════════════════════════════════════
-- § The strided-conv backward: `convFlatBack ∘ decimateBack`
-- ════════════════════════════════════════════════════════════════

/-- **Strided (stride-2) conv backward in flat `Vec` space** — the rendered input-VJP of
    `flatConvStride2 W b = decimateFlat ∘ flatConv`: zero-upsample the cotangent (`decimateBack`),
    then run the reversed-kernel conv (`convFlatBack`). `Vec (oc·h·w) → Vec (ic·2h·2w)`. -/
noncomputable def flatConvStride2Back {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) :
    Vec (oc * h * w) → Vec (ic * (2 * h) * (2 * w)) :=
  convFlatBack (h := 2 * h) (w := 2 * w) W ∘ decimateBack oc h w

/-- **The strided-conv input-VJP float-bridges.** One `.comp`: the decimation scatter
    (`floatBridges_decimateBack`, exact, modulus `id`) then the reversed-kernel conv
    (`floatBridges_convBack`). The strided sibling of `floatBridges_convBack`; unlocks the r34
    down-blocks and the stem. -/
theorem floatBridges_flatConvStride2Back {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) {w' : ℝ} (hw' : 0 ≤ w') (hn : 0 < oc * (2 * h) * (2 * w))
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') :
    FloatBridges (flatConvStride2Back (h := h) (w := w) W) := by
  unfold flatConvStride2Back
  exact (floatBridges_decimateBack oc h w).comp
    (floatBridges_convBack (h := 2 * h) (w := 2 * w) M W hw' hn hW)

end Proofs
