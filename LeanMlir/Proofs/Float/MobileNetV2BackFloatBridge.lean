import LeanMlir.Proofs.Float.DepthwiseBackFloatBridge
import LeanMlir.Proofs.Float.Resnet34WholeBackFloatBridge

/-! # ℝ→Float32 bridge: the WHOLE MobileNetV2 backward — the inverted-residual input-gradient fold

A3 (planning/a3_backward_deepnet_assembly.md Part 2): the first whole-net backward to consume the
§1e depthwise input-VJP. The forward `mobilenetv2Forward_full_pc` (ch7 6-block per-channel render) is
`dense ∘ GAP ∘ head ∘ b6 ∘ b5 ∘ residual b4 ∘ b3 ∘ residual b2 ∘ b1 ∘ stem`; the backward is its exact
reverse, each op replaced by its backward, threaded through one `FloatBridges.comp` chain — the
`r34InputGrad` blueprint at the MobileNetV2 topology.

The new piece is the **inverted-residual body backward** (`floatBridges_invresBodyBackPC` /
`…StridedBackPC`): the reverse of `project ∘ depthwise ∘ expand` is
`expandBack ∘ depthwiseBack ∘ projectBack`, where the depthwise stage reverses through the §1e
`depthwiseFlatBack` (stride-1) / `depthwiseStride2FlatBack` (stride-2 downsample) — MobileNetV2 has
**no SE**, so the depthwise input-VJP is the whole novelty. The skip blocks (`b2`/`b4`) reverse to
`residual (bodyBack)` (the additive skip contributes the cotangent verbatim — `FloatBridges.residual`).
Each conv/BN/relu6 is already bridged: `convFlatBack`, the supplied per-channel `bnBack`s, and the
relu6 kink as a fixed-mask `reluMaskBack` (`0 < preact < 6` at the smooth point).

The stem/head/GAP/dense **endpoints are concrete**; the 6 inverted-residual block backwards (and the
stem/head BN-backs) are **supplied as `FloatBridges` facts** — exactly as `r34_grad_floatBridges`
supplies its 16 blocks — discharged by the per-block bridges here (and `floatBridges_bnPerChannelBack`).
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The inverted-residual body backward (where §1e depthwiseBack lands)
-- ════════════════════════════════════════════════════════════════

/-- The stride-1 inverted-residual body input-gradient VJP at a smooth point — the **reverse of
    `invresBodyPC = project ∘ depthwise ∘ expand`**: `expandBack ∘ depthwiseBack ∘ projectBack`.
    `projectBack = convFlatBack Wp ∘ bnBp` (no relu6); `depthwiseBack = depthwiseFlatBack Wd ∘ bnBd ∘
    reluMaskBack m_d`; `expandBack = convFlatBack We ∘ bnBe ∘ reluMaskBack m_e`. The BN-backs are the
    per-channel BatchNorm backwards (supplied). -/
noncomputable def invresBodyBackPC {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (Wd : DepthwiseKernel mid kHd kWd) (Wp : Kernel4 oc mid kHp kWp)
    (bnBe bnBd : Vec (mid * h * w) → Vec (mid * h * w))
    (bnBp : Vec (oc * h * w) → Vec (oc * h * w))
    (m_e m_d : Fin (mid * h * w) → Prop) [DecidablePred m_e] [DecidablePred m_d] :
    Vec (oc * h * w) → Vec (ic * h * w) :=
  (convFlatBack (h := h) (w := w) We ∘ bnBe ∘ reluMaskBack m_e)
  ∘ (depthwiseFlatBack (h := h) (w := w) Wd ∘ bnBd ∘ reluMaskBack m_d)
  ∘ (convFlatBack (h := h) (w := w) Wp ∘ bnBp)

/-- **The stride-1 inverted-residual body backward float-bridges.** One `.comp` chain over
    `convFlatBack` (expand/project 1×1), the §1e `depthwiseFlatBack`, the supplied per-channel
    BN-backs, and the relu6 masks (`reluMaskBack`). The MobileNetV2 analogue of
    `floatBridges_r34IdBlockBack`'s body. -/
theorem floatBridges_invresBodyBackPC {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat} (M : FloatModel)
    (We : Kernel4 mid ic kHe kWe) (Wd : DepthwiseKernel mid kHd kWd) (Wp : Kernel4 oc mid kHp kWp)
    (bnBe bnBd : Vec (mid * h * w) → Vec (mid * h * w))
    (bnBp : Vec (oc * h * w) → Vec (oc * h * w))
    (m_e m_d : Fin (mid * h * w) → Prop) [DecidablePred m_e] [DecidablePred m_d]
    {we wd wp : ℝ} (hwe : 0 ≤ we) (hwd : 0 ≤ wd) (hwp : 0 ≤ wp)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ we) (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ wd)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hmid : 0 < mid) (hoc : 0 < oc) (hh : 0 < h) (hw : 0 < w)
    (hbnBe : FloatBridges bnBe) (hbnBd : FloatBridges bnBd) (hbnBp : FloatBridges bnBp) :
    FloatBridges (invresBodyBackPC We Wd Wp bnBe bnBd bnBp m_e m_d) := by
  unfold invresBodyBackPC
  exact ((hbnBp.comp (floatBridges_convBack (h := h) (w := w) M Wp hwp (by positivity) hWp)).comp
    (((floatBridges_reluMaskBack m_d).comp hbnBd).comp
      (floatBridges_depthwiseBack (h := h) (w := w) M Wd hwd (by positivity) hWd))).comp
    (((floatBridges_reluMaskBack m_e).comp hbnBe).comp
      (floatBridges_convBack (h := h) (w := w) M We hwe (by positivity) hWe))

/-- The stride-2 (downsample) inverted-residual body input-gradient VJP — the **reverse of
    `invresBodyStridedPC = project ∘ depthwiseStrided ∘ expand(2h×2w)`**:
    `expandBack(2h×2w) ∘ depthwiseStridedBack ∘ projectBack`, where the depthwise reverses through the
    §1e `depthwiseStride2FlatBack` (zero-upsample scatter then reversed-kernel depthwise). -/
noncomputable def invresBodyStridedBackPC {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (Wd : DepthwiseKernel mid kHd kWd) (Wp : Kernel4 oc mid kHp kWp)
    (bnBe : Vec (mid * (2 * h) * (2 * w)) → Vec (mid * (2 * h) * (2 * w)))
    (bnBd : Vec (mid * h * w) → Vec (mid * h * w))
    (bnBp : Vec (oc * h * w) → Vec (oc * h * w))
    (m_e : Fin (mid * (2 * h) * (2 * w)) → Prop) [DecidablePred m_e]
    (m_d : Fin (mid * h * w) → Prop) [DecidablePred m_d] :
    Vec (oc * h * w) → Vec (ic * (2 * h) * (2 * w)) :=
  (convFlatBack (h := 2 * h) (w := 2 * w) We ∘ bnBe ∘ reluMaskBack m_e)
  ∘ (depthwiseStride2FlatBack (h := h) (w := w) Wd ∘ bnBd ∘ reluMaskBack m_d)
  ∘ (convFlatBack (h := h) (w := w) Wp ∘ bnBp)

/-- **The stride-2 inverted-residual body backward float-bridges.** Same `.comp` shape as the
    stride-1 body, with the depthwise stage threading the §1e `depthwiseStride2FlatBack` and the
    expand back at the `2h×2w` grid. Unlocks the MobileNetV2 downsample blocks (`b1/b3/b5/b6`). -/
theorem floatBridges_invresBodyStridedBackPC {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (M : FloatModel)
    (We : Kernel4 mid ic kHe kWe) (Wd : DepthwiseKernel mid kHd kWd) (Wp : Kernel4 oc mid kHp kWp)
    (bnBe : Vec (mid * (2 * h) * (2 * w)) → Vec (mid * (2 * h) * (2 * w)))
    (bnBd : Vec (mid * h * w) → Vec (mid * h * w))
    (bnBp : Vec (oc * h * w) → Vec (oc * h * w))
    (m_e : Fin (mid * (2 * h) * (2 * w)) → Prop) [DecidablePred m_e]
    (m_d : Fin (mid * h * w) → Prop) [DecidablePred m_d]
    {we wd wp : ℝ} (hwe : 0 ≤ we) (hwd : 0 ≤ wd) (hwp : 0 ≤ wp)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ we) (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ wd)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hmid : 0 < mid) (hoc : 0 < oc) (hh : 0 < h) (hw : 0 < w)
    (hbnBe : FloatBridges bnBe) (hbnBd : FloatBridges bnBd) (hbnBp : FloatBridges bnBp) :
    FloatBridges (invresBodyStridedBackPC We Wd Wp bnBe bnBd bnBp m_e m_d) := by
  unfold invresBodyStridedBackPC
  exact ((hbnBp.comp (floatBridges_convBack (h := h) (w := w) M Wp hwp (by positivity) hWp)).comp
    (((floatBridges_reluMaskBack m_d).comp hbnBd).comp
      (floatBridges_depthwiseStride2Back (h := h) (w := w) M Wd hwd (by positivity) hWd))).comp
    (((floatBridges_reluMaskBack m_e).comp hbnBe).comp
      (floatBridges_convBack (h := 2 * h) (w := 2 * w) M We hwe (by positivity) hWe))

-- ════════════════════════════════════════════════════════════════
-- § The whole-net input-gradient VJP (the 6-block fold)
-- ════════════════════════════════════════════════════════════════

/-- The whole MobileNetV2 input-gradient VJP at a smooth point — the **exact reverse of
    `mobilenetv2Forward_full_pc`**: `dense ∘ GAP ∘ head ∘ b6 ∘ b5 ∘ residual b4 ∘ b3 ∘ residual b2 ∘
    b1 ∘ stem` reversed. The stem/head/GAP/dense endpoints are concrete (`flatConvStride2Back ∘ bnBs ∘
    reluMaskBack` / `convFlatBack ∘ bnBh ∘ reluMaskBack` / `gapBack` / `dense (transposeᵀ) 0`); the 6
    inverted-residual block backwards `b1B..b6B` are supplied (each `floatBridges_invresBody*BackPC`,
    the skip blocks `b2`/`b4` wrapped by `FloatBridges.residual`). Channel/spatial schedule encoded in
    the block maps' dims (the strided blocks halve spatial; the skip blocks preserve). -/
noncomputable def mnv2InputGrad
    (Ws : Kernel4 16 3 3 3) (Wh : Kernel4 128 64 1 1) (Wfc : Mat 128 10)
    (bnBs : Vec (16 * 112 * 112) → Vec (16 * 112 * 112))
    (bnBh : Vec (128 * 7 * 7) → Vec (128 * 7 * 7))
    (b1B : Vec (24 * 56 * 56) → Vec (16 * 112 * 112))
    (b2B : Vec (24 * 56 * 56) → Vec (24 * 56 * 56))
    (b3B : Vec (32 * 28 * 28) → Vec (24 * 56 * 56))
    (b4B : Vec (32 * 28 * 28) → Vec (32 * 28 * 28))
    (b5B : Vec (64 * 14 * 14) → Vec (32 * 28 * 28))
    (b6B : Vec (64 * 7 * 7) → Vec (64 * 14 * 14))
    (m_stem : Fin (16 * 112 * 112) → Prop) [DecidablePred m_stem]
    (m_head : Fin (128 * 7 * 7) → Prop) [DecidablePred m_head] :
    Vec 10 → Vec (3 * 224 * 224) :=
  (flatConvStride2Back (h := 112) (w := 112) Ws ∘ bnBs ∘ reluMaskBack m_stem)
  ∘ b1B ∘ b2B ∘ b3B ∘ b4B ∘ b5B ∘ b6B
  ∘ (convFlatBack (h := 7) (w := 7) Wh ∘ bnBh ∘ reluMaskBack m_head)
  ∘ gapBack 128 7 7
  ∘ dense (Mat.transpose Wfc) (0 : Vec 128)

set_option maxRecDepth 100000 in
/-- **The whole MobileNetV2 input-gradient VJP float-bridges** — the first whole-net backward to
    consume the §1e depthwise input-VJP. One `.comp` chain over the per-op backward bridges: `linBack`
    (dense), `gapBack`, the head `convFlatBack ∘ bnBh ∘ reluMaskBack`, the 6 supplied inverted-residual
    block backwards, and the stem `flatConvStride2Back ∘ bnBs ∘ reluMaskBack`. The deployed float
    backward of the whole net is within an explicit budget of the certified `ℝ` backward — the
    backward peer of `mobilenetv2Forward_full_pc`. Closes under `[propext, Classical.choice,
    Quot.sound]`. -/
theorem mnv2_grad_floatBridges (M : FloatModel)
    (Ws : Kernel4 16 3 3 3) (Wh : Kernel4 128 64 1 1) (Wfc : Mat 128 10)
    (bnBs : Vec (16 * 112 * 112) → Vec (16 * 112 * 112))
    (bnBh : Vec (128 * 7 * 7) → Vec (128 * 7 * 7))
    (b1B : Vec (24 * 56 * 56) → Vec (16 * 112 * 112))
    (b2B : Vec (24 * 56 * 56) → Vec (24 * 56 * 56))
    (b3B : Vec (32 * 28 * 28) → Vec (24 * 56 * 56))
    (b4B : Vec (32 * 28 * 28) → Vec (32 * 28 * 28))
    (b5B : Vec (64 * 14 * 14) → Vec (32 * 28 * 28))
    (b6B : Vec (64 * 7 * 7) → Vec (64 * 14 * 14))
    (m_stem : Fin (16 * 112 * 112) → Prop) [DecidablePred m_stem]
    (m_head : Fin (128 * 7 * 7) → Prop) [DecidablePred m_head]
    {ws wh wfc : ℝ} (hws : 0 ≤ ws) (hwh : 0 ≤ wh) (hwfc : 0 ≤ wfc)
    (hWs : ∀ o c kh kw, |Ws o c kh kw| ≤ ws) (hWh : ∀ o c kh kw, |Wh o c kh kw| ≤ wh)
    (hWfc : ∀ i j, |Wfc i j| ≤ wfc)
    (hbnBs : FloatBridges bnBs) (hbnBh : FloatBridges bnBh)
    (hb1B : FloatBridges b1B) (hb2B : FloatBridges b2B) (hb3B : FloatBridges b3B)
    (hb4B : FloatBridges b4B) (hb5B : FloatBridges b5B) (hb6B : FloatBridges b6B) :
    FloatBridges (mnv2InputGrad Ws Wh Wfc bnBs bnBh b1B b2B b3B b4B b5B b6B m_stem m_head) := by
  unfold mnv2InputGrad
  have hstem : FloatBridges
      (flatConvStride2Back (h := 112) (w := 112) Ws ∘ bnBs ∘ reluMaskBack m_stem) :=
    ((floatBridges_reluMaskBack m_stem).comp hbnBs).comp
      (floatBridges_flatConvStride2Back (h := 112) (w := 112) M Ws hws (by norm_num) hWs)
  have hhead : FloatBridges
      (convFlatBack (h := 7) (w := 7) Wh ∘ bnBh ∘ reluMaskBack m_head) :=
    ((floatBridges_reluMaskBack m_head).comp hbnBh).comp
      (floatBridges_convBack (h := 7) (w := 7) M Wh hwh (by norm_num) hWh)
  have h0 := (floatBridges_linBack M Wfc hwfc (by norm_num) hWfc).comp
    (floatBridges_gapBack M 128 7 7 (by norm_num) (by norm_num) (by norm_num))
  have hH := h0.comp hhead
  have hB6 := hH.comp hb6B
  have hB5 := hB6.comp hb5B
  have hB4 := hB5.comp hb4B
  have hB3 := hB4.comp hb3B
  have hB2 := hB3.comp hb2B
  have hB1 := hB2.comp hb1B
  exact hB1.comp hstem

end Proofs
