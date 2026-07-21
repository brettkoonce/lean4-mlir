import LeanMlir.Proofs.Float.EnetFloatBridge
import LeanMlir.Proofs.Float.Resnet34WholeFloatBridge
import LeanMlir.Proofs.Float.EfficientNetWholeFloatBridge
import LeanMlir.Proofs.MobileNetV2RenderPC

/-! # ℝ→Float32 bridge: the WHOLE MobileNetV2 FORWARD — the 6-block fold

The forward peer of `mnv2_grad_floatBridges` (`MobileNetV2BackFloatBridge.lean`), and the mnv2 entry
in the whole-net forward sweep (`Resnet34WholeFloatBridge.lean` is the worked example). The repo had
the mnv2 forward float story only at *op* level; this folds the whole ch7 6-block per-channel render
(`mobilenetv2Forward_full_pc`) — the SAME target the r34/mnv2 backward used (NOT the 17-block paper
trainer).

MobileNetV2 has **no squeeze-excite** — the inverted-residual block is `project ∘ depthwise ∘ expand`,
each stage a conv/depthwise with per-channel BN and a **relu6** clamp (the project stage is the linear
bottleneck, no relu6). So the one new forward op-bridge is `floatBridges_relu6`: relu6 = `min(max(·,0),6)`
is exact in float (clamp by exact constants, no rounding) and 1-Lipschitz (modulus id) — a clean mirror
of `floatClose_relu`, via the mathlib clamp lemmas `abs_max_sub_max_le_abs` / `abs_min_sub_min_le_max`.
The strided depthwise reuses `floatBridges_depthwiseStride2Flat` (built for efficientnet).

`mnv2Forward` is the `∘` skeleton of `mobilenetv2Forward_full_pc` (concrete stem/head/GAP/dense
endpoints; the stem/head BNs and the 6 inverted-residual blocks supplied as `FloatBridges` — exactly
as the backward `mnv2InputGrad` supplies its `b1B..b6B`), and the named per-block bridges
`floatBridges_invresBodyPC` / `floatBridges_invresBodyStridedPC` discharge the block hyps by name
(the forward peers of `floatBridges_invresBodyBackPC` / `…StridedBackPC`) — so the mnv2 forward whole
net stands exactly as strong as the backward (full forward/backward parity, like r34).
-/

namespace Proofs

open scoped Real
open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § Forward relu6 as a `FloatBridges`  (the mnv2 activation)
-- ════════════════════════════════════════════════════════════════

/-- **relu6 is magnitude-nonincreasing.** `|relu6 v i| ≤ |v i|`: the clamp `min(max(v i,0),6)` is in
    `[0, max(v i,0)]` and `max(v i,0) ≤ |v i|`. -/
theorem relu6_abs_le {n : Nat} (v : Vec n) (i : Fin n) : |relu6 n v i| ≤ |v i| := by
  simp only [relu6]
  rw [abs_of_nonneg (le_min (le_max_right _ _) (by norm_num))]
  exact (min_le_left _ _).trans (max_le (le_abs_self _) (abs_nonneg _))

/-- **relu6 is 1-Lipschitz.** `|relu6 vt i - relu6 va i| ≤ |vt i - va i| ≤ e`: the clamp is a
    composition of the 1-Lipschitz `max(·,0)` and `min(·,6)` (mathlib `abs_max_sub_max_le_abs` /
    `abs_min_sub_min_le_max`). The relu6 peer of `relu_close`. -/
theorem relu6_close {n : Nat} (xt xa : Vec n) (e : ℝ)
    (hx : ∀ i, |xt i - xa i| ≤ e) (i : Fin n) :
    |relu6 n xt i - relu6 n xa i| ≤ e := by
  simp only [relu6]
  calc |min (max (xt i) 0) 6 - min (max (xa i) 0) 6|
      ≤ max |max (xt i) 0 - max (xa i) 0| |(6 : ℝ) - 6| := abs_min_sub_min_le_max _ _ _ _
    _ = |max (xt i) 0 - max (xa i) 0| := by
          rw [sub_self, abs_zero, max_eq_left (abs_nonneg _)]
    _ ≤ |xt i - xa i| := abs_max_sub_max_le_abs _ _ _
    _ ≤ e := hx i

/-- **relu6 is `FloatClose`** — exact in float (no rounding) and 1-Lipschitz, so magnitude `A` is
    preserved and the modulus is the identity. The relu6 peer of `floatClose_relu`. -/
theorem floatClose_relu6 {n : Nat} (A : ℝ) :
    FloatClose A A (relu6 n) (relu6 n) (fun e => e) :=
  ⟨fun v hv i => ⟨(relu6_abs_le v i).trans (hv i), (relu6_abs_le v i).trans (hv i)⟩,
   fun vt va e _ _ hd i => relu6_close vt va e hd i⟩

/-- **relu6 float-bridges** (magnitude-preserving, modulus id). -/
theorem floatBridges_relu6 {n : Nat} : FloatBridges (relu6 n) :=
  fun A hA => ⟨A, _, _, hA, floatClose_relu6 A⟩

-- ════════════════════════════════════════════════════════════════
-- § The inverted-residual stage bridges (per-channel BN supplied)
-- ════════════════════════════════════════════════════════════════

/-- Expand stage `relu6 ∘ bnPC ∘ flatConv` float-bridges (per-channel BN supplied). -/
theorem floatBridges_ivExpandPC {ic mid h w kHe kWe : Nat} (M : FloatModel)
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    {w' bb : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hni : 0 < ic * h * w)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ w') (hbe : ∀ o, |be o| ≤ bb)
    (hbnE : FloatBridges (bnPerChannelTensor3 mid h w εe γe βe)) :
    FloatBridges (ivExpandPC (h := h) (w := w) We be εe γe βe) := by
  unfold ivExpandPC
  exact ((floatBridges_flatConv (h := h) (w := w) M We be hw' hbb hni hWe hbe).comp hbnE).comp
    floatBridges_relu6

/-- Depthwise stage (stride-1) `relu6 ∘ bnPC ∘ depthwise` float-bridges. -/
theorem floatBridges_ivDepthwisePC {mid h w kHd kWd : Nat} (M : FloatModel)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    {w' bb : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hnm : 0 < mid * h * w)
    (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ w') (hbd : ∀ ch, |bd ch| ≤ bb)
    (hbnD : FloatBridges (bnPerChannelTensor3 mid h w εd γd βd)) :
    FloatBridges (ivDepthwisePC (h := h) (w := w) Wd bd εd γd βd) := by
  unfold ivDepthwisePC
  exact ((floatBridges_depthwise (h := h) (w := w) M Wd bd hw' hbb hnm hWd hbd).comp hbnD).comp
    floatBridges_relu6

/-- Depthwise stage (stride-2 downsample) `relu6 ∘ bnPC ∘ depthwiseStride2` float-bridges. -/
theorem floatBridges_ivDepthwiseStridedPC {mid h w kHd kWd : Nat} (M : FloatModel)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    {w' bb : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hnm : 0 < mid * (2 * h) * (2 * w))
    (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ w') (hbd : ∀ ch, |bd ch| ≤ bb)
    (hbnD : FloatBridges (bnPerChannelTensor3 mid h w εd γd βd)) :
    FloatBridges (ivDepthwiseStridedPC (h := h) (w := w) Wd bd εd γd βd) := by
  unfold ivDepthwiseStridedPC
  exact ((floatBridges_depthwiseStride2Flat (h := h) (w := w) M Wd bd hw' hbb hnm hWd hbd).comp
    hbnD).comp floatBridges_relu6

/-- Project (linear bottleneck) stage `bnPC ∘ flatConv` float-bridges (no relu6). -/
theorem floatBridges_ivProjectPC {mid oc h w kHp kWp : Nat} (M : FloatModel)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    {w' bb : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hnm : 0 < mid * h * w)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ w') (hbp : ∀ o, |bp o| ≤ bb)
    (hbnP : FloatBridges (bnPerChannelTensor3 oc h w εp γp βp)) :
    FloatBridges (ivProjectPC (h := h) (w := w) Wp bp εp γp βp) := by
  unfold ivProjectPC
  exact (floatBridges_flatConv (h := h) (w := w) M Wp bp hw' hbb hnm hWp hbp).comp hbnP

-- ════════════════════════════════════════════════════════════════
-- § The named per-block forward bridges (peers of floatBridges_invresBody*BackPC)
-- ════════════════════════════════════════════════════════════════

/-- **The mnv2 stride-1 inverted-residual body float-bridges** — the forward peer of
    `floatBridges_invresBodyBackPC`. `invresBodyPC = project ∘ depthwise ∘ expand`, one `.comp` chain
    over the three stage bridges; the three per-channel BNs supplied as `FloatBridges` facts (discharge
    with `floatBridges_bnPerChannelTensor3`). The matched-channel skip is `FloatBridges.residual` over
    this body (the `b2`/`b4` blocks). -/
theorem floatBridges_invresBodyPC {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat} (M : FloatModel)
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    {w' bb : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb)
    (hni : 0 < ic * h * w) (hnm : 0 < mid * h * w)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ w') (hbe : ∀ o, |be o| ≤ bb)
    (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ w') (hbd : ∀ ch, |bd ch| ≤ bb)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ w') (hbp : ∀ o, |bp o| ≤ bb)
    (hbnE : FloatBridges (bnPerChannelTensor3 mid h w εe γe βe))
    (hbnD : FloatBridges (bnPerChannelTensor3 mid h w εd γd βd))
    (hbnP : FloatBridges (bnPerChannelTensor3 oc h w εp γp βp)) :
    FloatBridges (invresBodyPC (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp) := by
  unfold invresBodyPC
  exact ((floatBridges_ivExpandPC M We be εe γe βe hw' hbb hni hWe hbe hbnE).comp
    (floatBridges_ivDepthwisePC M Wd bd εd γd βd hw' hbb hnm hWd hbd hbnD)).comp
    (floatBridges_ivProjectPC M Wp bp εp γp βp hw' hbb hnm hWp hbp hbnP)

/-- **The mnv2 stride-2 (downsample) inverted-residual body float-bridges** — the forward peer of
    `floatBridges_invresBodyStridedBackPC`. `invresBodyStridedPC = project ∘ depthwiseStrided ∘
    expand(at 2h×2w)`; the strided depthwise reuses `floatBridges_depthwiseStride2Flat`. The three
    per-channel BNs supplied. The `b1`/`b3`/`b5`/`b6` blocks (no skip — channels/spatial change). -/
theorem floatBridges_invresBodyStridedPC {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat} (M : FloatModel)
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    {w' bb : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb)
    (hni : 0 < ic * (2 * h) * (2 * w)) (hnm2 : 0 < mid * (2 * h) * (2 * w)) (hnm : 0 < mid * h * w)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ w') (hbe : ∀ o, |be o| ≤ bb)
    (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ w') (hbd : ∀ ch, |bd ch| ≤ bb)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ w') (hbp : ∀ o, |bp o| ≤ bb)
    (hbnE : FloatBridges (bnPerChannelTensor3 mid (2 * h) (2 * w) εe γe βe))
    (hbnD : FloatBridges (bnPerChannelTensor3 mid h w εd γd βd))
    (hbnP : FloatBridges (bnPerChannelTensor3 oc h w εp γp βp)) :
    FloatBridges (invresBodyStridedPC (h := h) (w := w)
      We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp) := by
  unfold invresBodyStridedPC
  exact ((floatBridges_ivExpandPC (h := 2 * h) (w := 2 * w) M We be εe γe βe hw' hbb hni hWe hbe
    hbnE).comp
    (floatBridges_ivDepthwiseStridedPC M Wd bd εd γd βd hw' hbb hnm2 hWd hbd hbnD)).comp
    (floatBridges_ivProjectPC M Wp bp εp γp βp hw' hbb hnm hWp hbp hbnP)

-- ════════════════════════════════════════════════════════════════
-- § The whole-net forward (the 6-block fold)
-- ════════════════════════════════════════════════════════════════

/-- The whole MobileNetV2 forward — the structural skeleton of `mobilenetv2Forward_full_pc`:
    `dense ∘ GAP ∘ head ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ (relu6 ∘ bn ∘ stride-2-conv)`. The stem
    conv/relu6, head conv/relu6, GAP, and dense endpoints are concrete; the stem/head BNs `bnS`/`bnH`
    and the 6 inverted-residual blocks `b1..b6` are supplied (each `FloatBridges`, discharged by the
    per-block bridges; the skip blocks `b2`/`b4` via `FloatBridges.residual`). The forward peer of
    `mnv2InputGrad`. -/
noncomputable def mnv2Forward (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (Wh : Kernel4 128 64 1 1)
    (bh : Vec 128) (Wfc : Mat 128 10) (bfc : Vec 10)
    (bnS : Vec (16 * 112 * 112) → Vec (16 * 112 * 112))
    (bnH : Vec (128 * 7 * 7) → Vec (128 * 7 * 7))
    (b1 : Vec (16 * 112 * 112) → Vec (24 * 56 * 56))
    (b2 : Vec (24 * 56 * 56) → Vec (24 * 56 * 56))
    (b3 : Vec (24 * 56 * 56) → Vec (32 * 28 * 28))
    (b4 : Vec (32 * 28 * 28) → Vec (32 * 28 * 28))
    (b5 : Vec (32 * 28 * 28) → Vec (64 * 14 * 14))
    (b6 : Vec (64 * 14 * 14) → Vec (64 * 7 * 7)) :
    Vec (3 * 224 * 224) → Vec 10 :=
  dense Wfc bfc
  ∘ globalAvgPoolFlat 128 7 7
  ∘ (relu6 (128 * 7 * 7) ∘ bnH ∘ flatConv (h := 7) (w := 7) Wh bh)
  ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1
  ∘ (relu6 (16 * 112 * 112) ∘ bnS ∘ flatConvStride2 (h := 112) (w := 112) Ws bs)

set_option maxRecDepth 100000 in
/-- **The whole MobileNetV2 forward float-bridges** — the forward peer of `mnv2_grad_floatBridges`.
    One `.comp` chain over the per-op forward bridges: the concrete stem (`relu6 ∘ bn ∘
    flatConvStride2`), the 6 supplied inverted-residual blocks, the concrete head (`relu6 ∘ bn ∘
    flatConv`), `globalAvgPoolFlat`, and `dense`. The deployed float forward of the whole net is
    within an explicit budget of the certified `ℝ` forward. Closes under `[propext, Classical.choice,
    Quot.sound]`. -/
theorem mnv2Forward_floatBridges (M : FloatModel)
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (Wh : Kernel4 128 64 1 1) (bh : Vec 128)
    (Wfc : Mat 128 10) (bfc : Vec 10)
    (bnS : Vec (16 * 112 * 112) → Vec (16 * 112 * 112))
    (bnH : Vec (128 * 7 * 7) → Vec (128 * 7 * 7))
    (b1 : Vec (16 * 112 * 112) → Vec (24 * 56 * 56))
    (b2 : Vec (24 * 56 * 56) → Vec (24 * 56 * 56))
    (b3 : Vec (24 * 56 * 56) → Vec (32 * 28 * 28))
    (b4 : Vec (32 * 28 * 28) → Vec (32 * 28 * 28))
    (b5 : Vec (32 * 28 * 28) → Vec (64 * 14 * 14))
    (b6 : Vec (64 * 14 * 14) → Vec (64 * 7 * 7))
    {ws bsβ wh bhβ wfc bfcβ : ℝ} (hws : 0 ≤ ws) (hbsβ : 0 ≤ bsβ) (hwh : 0 ≤ wh) (hbhβ : 0 ≤ bhβ)
    (hwfc : 0 ≤ wfc) (hbfcβ : 0 ≤ bfcβ)
    (hWs : ∀ o c kh kw, |Ws o c kh kw| ≤ ws) (hbs : ∀ o, |bs o| ≤ bsβ)
    (hWh : ∀ o c kh kw, |Wh o c kh kw| ≤ wh) (hbh : ∀ o, |bh o| ≤ bhβ)
    (hWfc : ∀ i j, |Wfc i j| ≤ wfc) (hbfc : ∀ j, |bfc j| ≤ bfcβ)
    (hbnS : FloatBridges bnS) (hbnH : FloatBridges bnH)
    (hb1 : FloatBridges b1) (hb2 : FloatBridges b2) (hb3 : FloatBridges b3)
    (hb4 : FloatBridges b4) (hb5 : FloatBridges b5) (hb6 : FloatBridges b6) :
    FloatBridges (mnv2Forward Ws bs Wh bh Wfc bfc bnS bnH b1 b2 b3 b4 b5 b6) := by
  unfold mnv2Forward
  have hstem : FloatBridges
      (relu6 (16 * 112 * 112) ∘ bnS ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) :=
    ((floatBridges_flatConvStride2 (h := 112) (w := 112) M Ws bs hws hbsβ (by norm_num) hWs hbs).comp
      hbnS).comp floatBridges_relu6
  have h1 := hstem.comp hb1
  have h2 := h1.comp hb2
  have h3 := h2.comp hb3
  have h4 := h3.comp hb4
  have h5 := h4.comp hb5
  have h6 := h5.comp hb6
  have hhead : FloatBridges (relu6 (128 * 7 * 7) ∘ bnH ∘ flatConv (h := 7) (w := 7) Wh bh) :=
    ((floatBridges_flatConv (h := 7) (w := 7) M Wh bh hwh hbhβ (by norm_num) hWh hbh).comp
      hbnH).comp floatBridges_relu6
  have hH := h6.comp hhead
  have hGAP := hH.comp (floatBridges_gap (c := 128) (h := 7) (w := 7) M (by norm_num) (by norm_num))
  exact hGAP.comp (floatBridges_dense M Wfc bfc hwfc hbfcβ (by norm_num) hWfc hbfc)

end Proofs
