import LeanMlir.Proofs.Float.Resnet34DownBackFloatBridge

/-! # ℝ→Float32 bridge: the WHOLE ResNet-34 backward — the [3,4,6,3] input-gradient fold

The capstone of the r34 backward: the whole-net input-gradient VJP at a smooth point float-bridges.
The forward `resnet34Forward_full_pc` is `dense ∘ GAP ∘ [3,4,6,3] blocks ∘ maxpool ∘ stem`; the
backward is its exact reverse, each op replaced by its backward, threaded through one `.comp` chain.

Every op is already bridged: `linBack` (dense), `gapBack` (GAP, built here), the 16 block backwards
(`floatBridges_r34IdBlockBack` ×13 / `floatBridges_r34DownBlockBack` ×3), `maxPoolFlatBack`, and the
stem `flatConvStride2Back ∘ bnBack ∘ reluMaskBack`. The stem/GAP/maxpool/dense **endpoints are
concrete** (so the certificate runs from the loss logits to the image gradient); the 16 **blocks are
supplied as `FloatBridges` facts** — exactly as `cifarBn_floatBridges`/`cifarBn_grad_floatBridges`
supply their BNs — discharged by the per-block bridges. The `[3,4,6,3]` stage structure is encoded in
the block maps' dimensions (the down-blocks change channels×spatial; the identity blocks preserve).

`gapBack` (the one new op): the certified GAP VJP is `dy(channel)/(h·w)` — broadcast the cotangent
to every spatial cell, divide by `h·w`. A scaled broadcast: magnitude-nonincreasing (`h·w ≥ 1`),
one rounding (the reciprocal multiply), modulus `mulErr(A) + (1/(h·w))·e`.
-/

namespace Proofs

open scoped Real

-- ════════════════════════════════════════════════════════════════
-- § GAP backward (broadcast ÷ h·w)
-- ════════════════════════════════════════════════════════════════

/-- **Global-average-pool backward** — the certified GAP VJP: route `dy(channel)` to every spatial
    cell of that channel, divided by `h·w`. `Vec c → Vec (c·h·w)`. -/
noncomputable def gapBack (c h w : Nat) (dy : Vec c) : Vec (c * h * w) :=
  fun idx => dy (flatChannel c h w idx) / ((h : ℝ) * (w : ℝ))

/-- Float GAP backward: broadcast then multiply by the precomputed reciprocal `1/(h·w)`. -/
noncomputable def gapBackF (M : FloatModel) (c h w : Nat) (dy : Vec c) : Vec (c * h * w) :=
  fun idx => M.mul (1 / ((h : ℝ) * (w : ℝ))) (dy (flatChannel c h w idx))

/-- **GAP backward is `FloatClose`** — a scaled broadcast: the float `fl((1/(h·w))·dy(channel))` is
    within `mulErr(A) + (1/(h·w))·e` of the certified `dy(channel)/(h·w)`, output magnitude
    `(1/(h·w))·A + mulErr(A)` (magnitude-nonincreasing since `h·w ≥ 1`). One `mul_close` per
    coordinate. -/
theorem floatClose_gapBack (M : FloatModel) (c h w : Nat) (hh : 0 < h) (hw : 0 < w) (A : ℝ) :
    FloatClose A
      (1 / ((h : ℝ) * (w : ℝ)) * A + FloatModel.mulErr M.u (1 / ((h : ℝ) * (w : ℝ))) A 0 0)
      (gapBack c h w) (gapBackF M c h w)
      (fun e => FloatModel.mulErr M.u (1 / ((h : ℝ) * (w : ℝ))) A 0 0
                + 1 / ((h : ℝ) * (w : ℝ)) * e) := by
  have hu := M.u_nonneg
  have hD : (0 : ℝ) < (h : ℝ) * (w : ℝ) := by
    have : (0 : ℝ) < (h : ℝ) := by exact_mod_cast hh
    have : (0 : ℝ) < (w : ℝ) := by exact_mod_cast hw
    positivity
  have hinv : (0 : ℝ) ≤ 1 / ((h : ℝ) * (w : ℝ)) := by positivity
  refine ⟨fun v hv idx => ?_, fun vt va e _ hvt hd idx => ?_⟩
  · have hAge : 0 ≤ A := (abs_nonneg _).trans (hv (flatChannel c h w idx))
    have hmerr : 0 ≤ FloatModel.mulErr M.u (1 / ((h : ℝ) * (w : ℝ))) A 0 0 := by
      unfold FloatModel.mulErr; positivity
    have hreal : |gapBack c h w v idx| ≤ 1 / ((h : ℝ) * (w : ℝ)) * A := by
      show |v (flatChannel c h w idx) / ((h : ℝ) * (w : ℝ))| ≤ 1 / ((h : ℝ) * (w : ℝ)) * A
      rw [abs_div, abs_of_pos hD]
      calc |v (flatChannel c h w idx)| / ((h : ℝ) * (w : ℝ))
          ≤ A / ((h : ℝ) * (w : ℝ)) := by gcongr; exact hv _
        _ = 1 / ((h : ℝ) * (w : ℝ)) * A := by ring
    have hclose : |gapBackF M c h w v idx - gapBack c h w v idx|
        ≤ FloatModel.mulErr M.u (1 / ((h : ℝ) * (w : ℝ))) A 0 0 := by
      show |M.mul (1 / ((h : ℝ) * (w : ℝ))) (v (flatChannel c h w idx))
          - v (flatChannel c h w idx) / ((h : ℝ) * (w : ℝ))| ≤ _
      rw [show v (flatChannel c h w idx) / ((h : ℝ) * (w : ℝ))
            = 1 / ((h : ℝ) * (w : ℝ)) * v (flatChannel c h w idx) from by ring]
      exact M.mul_close (by simp) (by simp) (by rw [abs_of_nonneg hinv]) (hv _)
    refine ⟨hreal.trans (le_add_of_nonneg_right hmerr), ?_⟩
    calc |gapBackF M c h w v idx|
        ≤ |gapBackF M c h w v idx - gapBack c h w v idx| + |gapBack c h w v idx| := by
          simpa using abs_sub_le (gapBackF M c h w v idx) (gapBack c h w v idx) 0
      _ ≤ FloatModel.mulErr M.u (1 / ((h : ℝ) * (w : ℝ))) A 0 0 + 1 / ((h : ℝ) * (w : ℝ)) * A :=
          add_le_add hclose hreal
      _ = 1 / ((h : ℝ) * (w : ℝ)) * A + FloatModel.mulErr M.u (1 / ((h : ℝ) * (w : ℝ))) A 0 0 := by
          ring
  · have hclose : |gapBackF M c h w vt idx - gapBack c h w vt idx|
        ≤ FloatModel.mulErr M.u (1 / ((h : ℝ) * (w : ℝ))) A 0 0 := by
      show |M.mul (1 / ((h : ℝ) * (w : ℝ))) (vt (flatChannel c h w idx))
          - vt (flatChannel c h w idx) / ((h : ℝ) * (w : ℝ))| ≤ _
      rw [show vt (flatChannel c h w idx) / ((h : ℝ) * (w : ℝ))
            = 1 / ((h : ℝ) * (w : ℝ)) * vt (flatChannel c h w idx) from by ring]
      exact M.mul_close (by simp) (by simp) (by rw [abs_of_nonneg hinv]) (hvt _)
    have hdiff : |gapBack c h w vt idx - gapBack c h w va idx| ≤ 1 / ((h : ℝ) * (w : ℝ)) * e := by
      show |vt (flatChannel c h w idx) / ((h : ℝ) * (w : ℝ))
          - va (flatChannel c h w idx) / ((h : ℝ) * (w : ℝ))| ≤ 1 / ((h : ℝ) * (w : ℝ)) * e
      rw [div_sub_div_same, abs_div, abs_of_pos hD]
      calc |vt (flatChannel c h w idx) - va (flatChannel c h w idx)| / ((h : ℝ) * (w : ℝ))
          ≤ e / ((h : ℝ) * (w : ℝ)) := by gcongr; exact hd _
        _ = 1 / ((h : ℝ) * (w : ℝ)) * e := by ring
    calc |gapBackF M c h w vt idx - gapBack c h w va idx|
        ≤ |gapBackF M c h w vt idx - gapBack c h w vt idx|
          + |gapBack c h w vt idx - gapBack c h w va idx| := abs_sub_le _ _ _
      _ ≤ FloatModel.mulErr M.u (1 / ((h : ℝ) * (w : ℝ))) A 0 0 + 1 / ((h : ℝ) * (w : ℝ)) * e :=
          add_le_add hclose hdiff

theorem floatBridges_gapBack (M : FloatModel) (c h w : Nat) (hc : 0 < c) (hh : 0 < h) (hw : 0 < w) :
    FloatBridges (gapBack c h w) := fun A hA =>
  ⟨_, _, _, (floatClose_gapBack M c h w hh hw A).cod_nonneg hA (by positivity),
    floatClose_gapBack M c h w hh hw A⟩

-- ════════════════════════════════════════════════════════════════
-- § The whole-net input-gradient VJP (the [3,4,6,3] fold)
-- ════════════════════════════════════════════════════════════════

/-- The whole ResNet-34 input-gradient VJP at a smooth point — the **exact reverse of
    `resnet34Forward_full_pc`**: `dense ∘ GAP ∘ [3,4,6,3] blocks ∘ maxpool ∘ stem` reversed. The
    stem/GAP/maxpool/dense endpoints are concrete (`flatConvStride2Back`/`gapBack`/`maxPoolFlatBack`/
    `dense (transposeᵀ) 0`); the 16 block backwards `a0B..e1B` are supplied (each
    `floatBridges_r34IdBlockBack`/`floatBridges_r34DownBlockBack`). The `[3,4,6,3]` stage structure
    is in the block maps' dims (down-blocks change channels×spatial; identity blocks preserve). -/
noncomputable def r34InputGrad (Ws : Kernel4 64 3 7 7) (Wd : Mat 512 10)
    (bnBs : Vec (64 * 112 * 112) → Vec (64 * 112 * 112))
    (e1B e0B : Vec (512 * 7 * 7) → Vec (512 * 7 * 7))
    (d4B : Vec (512 * 7 * 7) → Vec (256 * 14 * 14))
    (c4B c3B c2B c1B c0B : Vec (256 * 14 * 14) → Vec (256 * 14 * 14))
    (d3B : Vec (256 * 14 * 14) → Vec (128 * 28 * 28))
    (b2B b1B b0B : Vec (128 * 28 * 28) → Vec (128 * 28 * 28))
    (d2B : Vec (128 * 28 * 28) → Vec (64 * 56 * 56))
    (a2B a1B a0B : Vec (64 * 56 * 56) → Vec (64 * 56 * 56))
    (xmp : Tensor3 64 112 112)
    (m_stem : Fin (64 * 112 * 112) → Prop) [DecidablePred m_stem] :
    Vec 10 → Vec (3 * 224 * 224) :=
  (flatConvStride2Back (h := 112) (w := 112) Ws ∘ bnBs ∘ reluMaskBack m_stem)
  ∘ maxPoolFlatBack xmp
  ∘ a0B ∘ a1B ∘ a2B
  ∘ d2B
  ∘ b0B ∘ b1B ∘ b2B
  ∘ d3B
  ∘ c0B ∘ c1B ∘ c2B ∘ c3B ∘ c4B
  ∘ d4B
  ∘ e0B ∘ e1B
  ∘ gapBack 512 7 7
  ∘ dense (Mat.transpose Wd) (0 : Vec 512)

set_option maxRecDepth 100000 in
/-- **The whole ResNet-34 input-gradient VJP float-bridges** — the first Imagenette whole-net
    backward. One `.comp` chain over the per-op backward bridges: `linBack` (dense), `gapBack`, the
    16 supplied block backwards, `maxPoolFlatBack`, and the stem `flatConvStride2Back ∘ bnBack ∘
    reluMaskBack`. The deployed float backward of the whole 34-layer net is within an explicit budget
    of the certified `ℝ` backward — the backward peer of the r34 forward. Closes under
    `[propext, Classical.choice, Quot.sound]`. -/
theorem r34_grad_floatBridges (M : FloatModel)
    (Ws : Kernel4 64 3 7 7) (Wd : Mat 512 10)
    (bnBs : Vec (64 * 112 * 112) → Vec (64 * 112 * 112))
    (e1B e0B : Vec (512 * 7 * 7) → Vec (512 * 7 * 7))
    (d4B : Vec (512 * 7 * 7) → Vec (256 * 14 * 14))
    (c4B c3B c2B c1B c0B : Vec (256 * 14 * 14) → Vec (256 * 14 * 14))
    (d3B : Vec (256 * 14 * 14) → Vec (128 * 28 * 28))
    (b2B b1B b0B : Vec (128 * 28 * 28) → Vec (128 * 28 * 28))
    (d2B : Vec (128 * 28 * 28) → Vec (64 * 56 * 56))
    (a2B a1B a0B : Vec (64 * 56 * 56) → Vec (64 * 56 * 56))
    (xmp : Tensor3 64 112 112)
    (m_stem : Fin (64 * 112 * 112) → Prop) [DecidablePred m_stem]
    {ws wd : ℝ} (hws : 0 ≤ ws) (hwd : 0 ≤ wd)
    (hWs : ∀ o c kh kw, |Ws o c kh kw| ≤ ws) (hWd : ∀ i j, |Wd i j| ≤ wd)
    (hbnBs : FloatBridges bnBs)
    (he1B : FloatBridges e1B) (he0B : FloatBridges e0B) (hd4B : FloatBridges d4B)
    (hc4B : FloatBridges c4B) (hc3B : FloatBridges c3B) (hc2B : FloatBridges c2B)
    (hc1B : FloatBridges c1B) (hc0B : FloatBridges c0B) (hd3B : FloatBridges d3B)
    (hb2B : FloatBridges b2B) (hb1B : FloatBridges b1B) (hb0B : FloatBridges b0B)
    (hd2B : FloatBridges d2B)
    (ha2B : FloatBridges a2B) (ha1B : FloatBridges a1B) (ha0B : FloatBridges a0B) :
    FloatBridges (r34InputGrad Ws Wd bnBs e1B e0B d4B c4B c3B c2B c1B c0B d3B
      b2B b1B b0B d2B a2B a1B a0B xmp m_stem) := by
  unfold r34InputGrad
  have hstem : FloatBridges
      (flatConvStride2Back (h := 112) (w := 112) Ws ∘ bnBs ∘ reluMaskBack m_stem) :=
    ((floatBridges_reluMaskBack m_stem).comp hbnBs).comp
      (floatBridges_flatConvStride2Back (h := 112) (w := 112) M Ws hws (by norm_num) hWs)
  have h0 := (floatBridges_linBack M Wd hwd (by norm_num) hWd).comp
    (floatBridges_gapBack M 512 7 7 (by norm_num) (by norm_num) (by norm_num))
  have hE := (h0.comp he1B).comp he0B
  have hD4 := hE.comp hd4B
  have hC := ((((hD4.comp hc4B).comp hc3B).comp hc2B).comp hc1B).comp hc0B
  have hD3 := hC.comp hd3B
  have hB := ((hD3.comp hb2B).comp hb1B).comp hb0B
  have hD2 := hB.comp hd2B
  have hA := ((hD2.comp ha2B).comp ha1B).comp ha0B
  have hMP := hA.comp (floatBridges_maxPoolBack xmp)
  exact hMP.comp hstem

end Proofs
