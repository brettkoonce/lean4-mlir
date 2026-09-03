import LeanMlir.Proofs.Float.EnetFloatBridge
import LeanMlir.Proofs.Float.Resnet34WholeFloatBridge
import LeanMlir.Proofs.Float.EfficientNetWholeFloatBridge
import LeanMlir.Proofs.Codegen.MobileNetV2RenderPC

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

/-- ⭐ **relu6 is bounded by 6 whatever its input.** `min (max (v i) 0) 6 ∈ [0, 6]` — the half of
    the clamp `relu` does not have, and the reason a MobileNetV2 window does not grow: every
    relu6 site RESETS the certified magnitude to `6` rather than passing the incoming one
    through. Measured with `scripts/float_budget_envelope.py`'s `mnv2_eval_chain`, that is the
    difference between a certified output window of `10¹⁰⁰` and one of `10³`. -/
theorem relu6_le_six {n : Nat} (v : Vec n) (i : Fin n) : |relu6 n v i| ≤ 6 := by
  simp only [relu6]
  rw [abs_of_nonneg (le_min (le_max_right _ _) (by norm_num))]
  exact min_le_right _ _

/-- **relu6 is `FloatClose`** — exact in float (no rounding) and 1-Lipschitz, so the modulus is
    the identity; the output magnitude is `min A 6`, since relu6 is both magnitude-nonincreasing
    (`relu6_abs_le`) and clamped at `6` (`relu6_le_six`). The relu6 peer of `floatClose_relu`,
    ⚠ with the clamp — `floatClose_relu`'s `FloatClose A A` is the best `relu` can do, and
    stating `FloatClose A A` here would throw the clamp away. -/
theorem floatClose_relu6 {n : Nat} (A : ℝ) :
    FloatClose A (min A 6) (relu6 n) (relu6 n) (fun e => e) :=
  ⟨fun v hv i =>
     have h : |relu6 n v i| ≤ min A 6 :=
       le_min ((relu6_abs_le v i).trans (hv i)) (relu6_le_six v i)
     ⟨h, h⟩,
   fun vt va e _ _ hd i => relu6_close vt va e hd i⟩

/-- **relu6 float-bridges** (magnitude `A ↦ min A 6`, modulus id). -/
theorem floatBridges_relu6 {n : Nat} : FloatBridges (relu6 n) :=
  fun A hA => ⟨min A 6, _, _, le_min hA (by norm_num), floatClose_relu6 A⟩

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
    within a budget of the certified `ℝ` forward (⚠ one `FloatBridges` does not name — §4d;
    `mnv2Forward_floatBridgesTo` carries it as `.mod`). Closes under `[propext, Classical.choice,
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


-- ════════════════════════════════════════════════════════════════
-- § The same fold, with the float net NAMED (`FloatBridgesTo` migration)
-- ════════════════════════════════════════════════════════════════

/-- ReLU6 float-bridges to itself (clamp-and-select rounds nothing), with the clamp kept:
    `mag A = min A 6`, `mod = id`. -/
noncomputable def floatBridgesTo_relu6 {n : Nat} : FloatBridgesTo (relu6 n) (relu6 n) :=
  ⟨fun A => min A 6, fun _ e => e, fun A hA => ⟨le_min hA (by norm_num), floatClose_relu6 A⟩⟩

-- ════════════════════════════════════════════════════════════════
-- § The inverted-residual body with its NORMALISATION abstract
--   (the `rblkGen` pattern: ONE set of block bridges for training AND inference)
-- ════════════════════════════════════════════════════════════════

/-! `ivExpandPC`/`ivDepthwisePC`/`ivDepthwiseStridedPC`/`ivProjectPC` hard-wire
`bnPerChannelTensor3` — training-mode BatchNorm, whose modulus is quadratic in the window
(`planning/float_budget_numbers.md` §0.1). The deployed inference net is the same skeleton at
`bnPerChannelEvalTensor3`. Rather than a second copy of every stage, bridge and envelope, the
stages below take the normalisation as an argument, exactly as `rblkGen` does for ResNet-34:
`*PC_eq_gen` is `rfl` in each direction, so one set of block bridges serves both modes and the
choice of BN becomes an argument rather than a second proof. -/

/-- Expand stage with its normalisation abstract: `relu6 ∘ bn ∘ conv(1×1)`. -/
@[reducible] noncomputable def ivExpandGen {ic mid h w kHe kWe : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid)
    (bne : Vec (mid * h * w) → Vec (mid * h * w)) :
    Vec (ic * h * w) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bne ∘ flatConv We be

/-- Depthwise stage (stride-1) with its normalisation abstract. -/
@[reducible] noncomputable def ivDepthwiseGen {mid h w kHd kWd : Nat}
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid)
    (bnd : Vec (mid * h * w) → Vec (mid * h * w)) :
    Vec (mid * h * w) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnd ∘ depthwiseFlat Wd bd

/-- Depthwise stage (stride-2 downsample) with its normalisation abstract. -/
@[reducible] noncomputable def ivDepthwiseStridedGen {mid h w kHd kWd : Nat}
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid)
    (bnd : Vec (mid * h * w) → Vec (mid * h * w)) :
    Vec (mid * (2 * h) * (2 * w)) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnd ∘ depthwiseStride2Flat Wd bd

/-- Project (linear bottleneck) stage with its normalisation abstract — no relu6, so this is
    the one stage of the block that does NOT reset the window. -/
@[reducible] noncomputable def ivProjectGen {mid oc h w kHp kWp : Nat}
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc)
    (bnp : Vec (oc * h * w) → Vec (oc * h * w)) :
    Vec (mid * h * w) → Vec (oc * h * w) :=
  bnp ∘ flatConv Wp bp

/-- **The stride-1 inverted-residual body with its normalisations abstract.** -/
@[reducible] noncomputable def invresBodyGen {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid)
    (bne : Vec (mid * h * w) → Vec (mid * h * w))
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid)
    (bnd : Vec (mid * h * w) → Vec (mid * h * w))
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc)
    (bnp : Vec (oc * h * w) → Vec (oc * h * w)) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  ivProjectGen (h := h) (w := w) Wp bp bnp ∘
    (ivDepthwiseGen (h := h) (w := w) Wd bd bnd ∘ ivExpandGen (h := h) (w := w) We be bne)

/-- **The stride-2 inverted-residual body with its normalisations abstract** — the expand runs
    at `2h×2w`, the depthwise decimates. -/
@[reducible] noncomputable def invresBodyStridedGen {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid)
    (bne : Vec (mid * (2 * h) * (2 * w)) → Vec (mid * (2 * h) * (2 * w)))
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid)
    (bnd : Vec (mid * h * w) → Vec (mid * h * w))
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc)
    (bnp : Vec (oc * h * w) → Vec (oc * h * w)) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  ivProjectGen (h := h) (w := w) Wp bp bnp ∘
    (ivDepthwiseStridedGen (h := h) (w := w) Wd bd bnd ∘
      ivExpandGen (h := 2 * h) (w := 2 * w) We be bne)

/-- `invresBodyPC` IS `invresBodyGen` at the training-mode per-channel BN. -/
theorem invresBodyPC_eq_gen {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) :
    invresBodyPC (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp
      = invresBodyGen (h := h) (w := w) We be (bnPerChannelTensor3 mid h w εe γe βe)
          Wd bd (bnPerChannelTensor3 mid h w εd γd βd)
          Wp bp (bnPerChannelTensor3 oc h w εp γp βp) := rfl

/-- `invresBodyStridedPC` IS `invresBodyStridedGen` at the training-mode per-channel BN. -/
theorem invresBodyStridedPC_eq_gen {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) :
    invresBodyStridedPC (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp
      = invresBodyStridedGen (h := h) (w := w) We be
          (bnPerChannelTensor3 mid (2 * h) (2 * w) εe γe βe)
          Wd bd (bnPerChannelTensor3 mid h w εd γd βd)
          Wp bp (bnPerChannelTensor3 oc h w εp γp βp) := rfl

/-- **The float stride-1 inverted-residual body** — three rounded convolutions, three supplied
    float normalisations, `relu6` unchanged (exact in float). -/
noncomputable def invresBodyGenF {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat} (M : FloatModel)
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid)
    (bneF : Vec (mid * h * w) → Vec (mid * h * w))
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid)
    (bndF : Vec (mid * h * w) → Vec (mid * h * w))
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc)
    (bnpF : Vec (oc * h * w) → Vec (oc * h * w)) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  (bnpF ∘ M.flatConvF (h := h) (w := w) Wp bp) ∘
    ((relu6 (mid * h * w) ∘ bndF ∘ M.depthwiseFlatF (h := h) (w := w) Wd bd) ∘
      (relu6 (mid * h * w) ∘ bneF ∘ M.flatConvF (h := h) (w := w) We be))

/-- **The float stride-2 inverted-residual body.** -/
noncomputable def invresBodyStridedGenF {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (M : FloatModel) (We : Kernel4 mid ic kHe kWe) (be : Vec mid)
    (bneF : Vec (mid * (2 * h) * (2 * w)) → Vec (mid * (2 * h) * (2 * w)))
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid)
    (bndF : Vec (mid * h * w) → Vec (mid * h * w))
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc)
    (bnpF : Vec (oc * h * w) → Vec (oc * h * w)) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  (bnpF ∘ M.flatConvF (h := h) (w := w) Wp bp) ∘
    ((relu6 (mid * h * w) ∘ bndF ∘ M.depthwiseStride2FlatF (h := h) (w := w) Wd bd) ∘
      (relu6 (mid * (2 * h) * (2 * w)) ∘ bneF ∘ M.flatConvF (h := 2 * h) (w := 2 * w) We be))

/-- **The stride-1 inverted-residual body float-bridges TO its float body**, generic in the
    three normalisations — the mnv2 peer of `floatBridgesTo_r34IdBlock`. Eight `.comp` steps:
    conv, BN, relu6, depthwise, BN, relu6, conv, BN. -/
noncomputable def floatBridgesTo_invresBodyGen {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (M : FloatModel) (We : Kernel4 mid ic kHe kWe) (be : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc)
    (bne bneF bnd bndF : Vec (mid * h * w) → Vec (mid * h * w))
    (bnp bnpF : Vec (oc * h * w) → Vec (oc * h * w))
    {w' β' : ℝ} (hw' : 0 ≤ w') (hβ' : 0 ≤ β')
    (hni : 0 < ic * h * w) (hnm : 0 < mid * h * w)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ w') (hbe : ∀ o, |be o| ≤ β')
    (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ w') (hbd : ∀ ch, |bd ch| ≤ β')
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ w') (hbp : ∀ o, |bp o| ≤ β')
    (hbne : FloatBridgesTo bne bneF) (hbnd : FloatBridgesTo bnd bndF)
    (hbnp : FloatBridgesTo bnp bnpF) :
    FloatBridgesTo (invresBodyGen (h := h) (w := w) We be bne Wd bd bnd Wp bp bnp)
      (invresBodyGenF M We be bneF Wd bd bndF Wp bp bnpF) :=
  ((((((floatBridgesTo_flatConv (h := h) (w := w) M We be hw' hβ' hni hWe hbe).comp hbne).comp
        floatBridgesTo_relu6).comp
      (floatBridgesTo_depthwise (h := h) (w := w) M Wd bd hw' hβ' hnm hWd hbd)).comp hbnd).comp
    floatBridgesTo_relu6).comp
    ((floatBridgesTo_flatConv (h := h) (w := w) M Wp bp hw' hβ' hnm hWp hbp).comp hbnp)

/-- **The stride-2 inverted-residual body float-bridges TO its float body**, generic in the
    three normalisations. Same eight steps with the expand at `2h×2w` and the strided
    depthwise. -/
noncomputable def floatBridgesTo_invresBodyStridedGen
    {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (M : FloatModel) (We : Kernel4 mid ic kHe kWe) (be : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc)
    (bne bneF : Vec (mid * (2 * h) * (2 * w)) → Vec (mid * (2 * h) * (2 * w)))
    (bnd bndF : Vec (mid * h * w) → Vec (mid * h * w))
    (bnp bnpF : Vec (oc * h * w) → Vec (oc * h * w))
    {w' β' : ℝ} (hw' : 0 ≤ w') (hβ' : 0 ≤ β')
    (hni : 0 < ic * (2 * h) * (2 * w)) (hnm2 : 0 < mid * (2 * h) * (2 * w))
    (hnm : 0 < mid * h * w)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ w') (hbe : ∀ o, |be o| ≤ β')
    (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ w') (hbd : ∀ ch, |bd ch| ≤ β')
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ w') (hbp : ∀ o, |bp o| ≤ β')
    (hbne : FloatBridgesTo bne bneF) (hbnd : FloatBridgesTo bnd bndF)
    (hbnp : FloatBridgesTo bnp bnpF) :
    FloatBridgesTo (invresBodyStridedGen (h := h) (w := w) We be bne Wd bd bnd Wp bp bnp)
      (invresBodyStridedGenF M We be bneF Wd bd bndF Wp bp bnpF) :=
  ((((((floatBridgesTo_flatConv (h := 2 * h) (w := 2 * w) M We be hw' hβ' hni hWe hbe).comp
          hbne).comp floatBridgesTo_relu6).comp
      (floatBridgesTo_depthwiseStride2Flat (h := h) (w := w) M Wd bd hw' hβ' hnm2
        hWd hbd)).comp hbnd).comp floatBridgesTo_relu6).comp
    ((floatBridgesTo_flatConv (h := h) (w := w) M Wp bp hw' hβ' hnm hWp hbp).comp hbnp)

/-- **The float MobileNetV2 forward skeleton** — `mnv2Forward` with each concrete
    slot replaced by the model's rounded peer and each supplied slot by that
    block's float map. `relu6` is exact in float and is unchanged. -/
noncomputable def mnv2ForwardF (M : FloatModel)
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (Wh : Kernel4 128 64 1 1)
    (bh : Vec 128) (Wfc : Mat 128 10) (bfc : Vec 10)
    (bnSF : Vec (16 * 112 * 112) → Vec (16 * 112 * 112))
    (bnHF : Vec (128 * 7 * 7) → Vec (128 * 7 * 7))
    (b1F : Vec (16 * 112 * 112) → Vec (24 * 56 * 56))
    (b2F : Vec (24 * 56 * 56) → Vec (24 * 56 * 56))
    (b3F : Vec (24 * 56 * 56) → Vec (32 * 28 * 28))
    (b4F : Vec (32 * 28 * 28) → Vec (32 * 28 * 28))
    (b5F : Vec (32 * 28 * 28) → Vec (64 * 14 * 14))
    (b6F : Vec (64 * 14 * 14) → Vec (64 * 7 * 7)) :
    Vec (3 * 224 * 224) → Vec 10 :=
  M.dense Wfc bfc
  ∘ M.gapFlatF
  ∘ (relu6 (128 * 7 * 7) ∘ bnHF ∘ M.flatConvF (h := 7) (w := 7) Wh bh)
  ∘ b6F ∘ b5F ∘ b4F ∘ b3F ∘ b2F ∘ b1F
  ∘ (relu6 (16 * 112 * 112) ∘ bnSF ∘ M.flatConvStride2F (h := 112) (w := 112) Ws bs)

set_option maxRecDepth 100000 in
/-- **The whole MobileNetV2 forward float-bridges TO its float skeleton.** Same
    `.comp` chain as `mnv2Forward_floatBridges`, with every float map named — so
    this is the statement that carries "the deployed float forward of the whole net
    is within the bridge's `.mod` budget of the certified `ℝ` forward"
    (`formalization.yaml` fidelity §4d). -/
noncomputable def mnv2Forward_floatBridgesTo (M : FloatModel)
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (Wh : Kernel4 128 64 1 1) (bh : Vec 128)
    (Wfc : Mat 128 10) (bfc : Vec 10)
    (bnS bnSF : Vec (16 * 112 * 112) → Vec (16 * 112 * 112))
    (bnH bnHF : Vec (128 * 7 * 7) → Vec (128 * 7 * 7))
    (b1 b1F : Vec (16 * 112 * 112) → Vec (24 * 56 * 56))
    (b2 b2F : Vec (24 * 56 * 56) → Vec (24 * 56 * 56))
    (b3 b3F : Vec (24 * 56 * 56) → Vec (32 * 28 * 28))
    (b4 b4F : Vec (32 * 28 * 28) → Vec (32 * 28 * 28))
    (b5 b5F : Vec (32 * 28 * 28) → Vec (64 * 14 * 14))
    (b6 b6F : Vec (64 * 14 * 14) → Vec (64 * 7 * 7))
    {ws bsβ wh bhβ wfc bfcβ : ℝ} (hws : 0 ≤ ws) (hbsβ : 0 ≤ bsβ) (hwh : 0 ≤ wh) (hbhβ : 0 ≤ bhβ)
    (hwfc : 0 ≤ wfc) (hbfcβ : 0 ≤ bfcβ)
    (hWs : ∀ o c kh kw, |Ws o c kh kw| ≤ ws) (hbs : ∀ o, |bs o| ≤ bsβ)
    (hWh : ∀ o c kh kw, |Wh o c kh kw| ≤ wh) (hbh : ∀ o, |bh o| ≤ bhβ)
    (hWfc : ∀ i j, |Wfc i j| ≤ wfc) (hbfc : ∀ j, |bfc j| ≤ bfcβ)
    (hbnS : FloatBridgesTo bnS bnSF) (hbnH : FloatBridgesTo bnH bnHF)
    (hb1 : FloatBridgesTo b1 b1F) (hb2 : FloatBridgesTo b2 b2F)
    (hb3 : FloatBridgesTo b3 b3F) (hb4 : FloatBridgesTo b4 b4F)
    (hb5 : FloatBridgesTo b5 b5F) (hb6 : FloatBridgesTo b6 b6F) :
    FloatBridgesTo (mnv2Forward Ws bs Wh bh Wfc bfc bnS bnH b1 b2 b3 b4 b5 b6)
      (mnv2ForwardF M Ws bs Wh bh Wfc bfc bnSF bnHF b1F b2F b3F b4F b5F b6F) := by
  unfold mnv2Forward mnv2ForwardF
  have hstem : FloatBridgesTo
      (relu6 (16 * 112 * 112) ∘ bnS ∘ flatConvStride2 (h := 112) (w := 112) Ws bs)
      (relu6 (16 * 112 * 112) ∘ bnSF ∘ M.flatConvStride2F (h := 112) (w := 112) Ws bs) :=
    ((floatBridgesTo_flatConvStride2 (h := 112) (w := 112) M Ws bs hws hbsβ (by norm_num)
      hWs hbs).comp hbnS).comp floatBridgesTo_relu6
  have h1 := hstem.comp hb1
  have h2 := h1.comp hb2
  have h3 := h2.comp hb3
  have h4 := h3.comp hb4
  have h5 := h4.comp hb5
  have h6 := h5.comp hb6
  have hhead : FloatBridgesTo
      (relu6 (128 * 7 * 7) ∘ bnH ∘ flatConv (h := 7) (w := 7) Wh bh)
      (relu6 (128 * 7 * 7) ∘ bnHF ∘ M.flatConvF (h := 7) (w := 7) Wh bh) :=
    ((floatBridgesTo_flatConv (h := 7) (w := 7) M Wh bh hwh hbhβ (by norm_num)
      hWh hbh).comp hbnH).comp floatBridgesTo_relu6
  have hH := h6.comp hhead
  have hGAP := hH.comp (floatBridgesTo_gap (c := 128) (h := 7) (w := 7) M (by norm_num) (by norm_num))
  exact hGAP.comp (floatBridgesTo_dense M Wfc bfc hwfc hbfcβ (by norm_num) hWfc hbfc)

end Proofs
