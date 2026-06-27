import LeanMlir.Proofs.EfficientNetBackFloatBridge
import LeanMlir.Proofs.Resnet34WholeBackFloatBridge

/-! # ℝ→Float32 bridge: the WHOLE-NET EfficientNet input-gradient backward (the last capstone)

The backward peer of `efficientnetForwardB_floatBridges` (`EfficientNetWholeFloatBridge.lean`), and the
**final** entry in the 5-net × {forward, backward} whole-net `FloatBridges` matrix — every other net
(r34/mnv2/convnext/vit + the cifar pair) already has both; EfficientNet had only the forward.

The deployed forward `efficientnetForwardB` is the batched 3-block reduced B0
`head ∘ mbResid ∘ mbStrided ∘ mbNoExp ∘ stem`; its input-gradient backward is the reverse composition,
which — at a smooth point — is itself a *forward* composition of maps on the cotangent, so it threads
through the SAME `FloatBridges.comp` backbone (the `r34_grad_floatBridges` / `mnv2_grad_floatBridges`
blueprint). Exactly as `mnv2_grad_floatBridges`, the per-block backwards (`b1B`/`b2B`/`b3B`) and the
stem/head batch-norm + swish backwards enter as **supplied** `FloatBridges` hypotheses (each separately
dischargeable — the BNs by `floatBridges_bnBack`, the swishes by `floatBridges_diagBack`, the residual
block `b3B` by `floatBridges_mbconvBatchedResidBack`), around the concrete `batchMap`-lifted endpoints:
the strided-stem conv (`flatConvStride2Back`), the head conv (`convFlatBack`), the GAP scatter
(`gapBack`), and the classifier (`linBack`). Pure `.comp` assembly — A3 = closeness at a smooth point.
3-axiom-clean.
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § The whole-net EfficientNet input-gradient backward skeleton
-- ════════════════════════════════════════════════════════════════

/-- **The batched whole-net EfficientNet input-gradient backward** — the reverse of
    `efficientnetForwardB = head ∘ mbResid ∘ mbStrided ∘ mbNoExp ∘ stem` (the representative 3-block
    batched B0): classifier-back → GAP-back → head-conv-bn-swish-back → the three MBConv block backs →
    stem-conv-bn-swish-back. The block backs `b1B`/`b2B`/`b3B` and the stem/head BN+swish backs are
    supplied (the `mnv2InputGrad` discipline); the conv/GAP/dense leaves are concrete, `batchMap`-lifted
    over the `N` examples. -/
noncomputable def efficientnetInputGradB (N : Nat)
    (Ws : Kernel4 32 3 3 3) (Wh : Kernel4 1280 24 1 1) (Wfc : Mat 1280 10)
    (bnBs swBs : Vec (N * (32 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (bnBh swBh : Vec (N * (1280 * 56 * 56)) → Vec (N * (1280 * 56 * 56)))
    (b1B : Vec (N * (16 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (b2B : Vec (N * (24 * 56 * 56)) → Vec (N * (16 * 112 * 112)))
    (b3B : Vec (N * (24 * 56 * 56)) → Vec (N * (24 * 56 * 56))) :
    Vec (N * 10) → Vec (N * (3 * 224 * 224)) :=
  (StableHLO.batchMap N (flatConvStride2Back (h := 112) (w := 112) Ws) ∘ bnBs ∘ swBs)
  ∘ b1B ∘ b2B ∘ b3B
  ∘ (StableHLO.batchMap N (convFlatBack (h := 56) (w := 56) Wh) ∘ bnBh ∘ swBh)
  ∘ StableHLO.batchMap N (gapBack 1280 56 56)
  ∘ StableHLO.batchMap N (Proofs.dense (Mat.transpose Wfc) (0 : Vec 1280))

/-- **THE WHOLE-NET EfficientNet BACKWARD FLOAT-BRIDGES** — the final whole-net `FloatBridges` capstone.
    One `.comp` thread over the concrete classifier/GAP/head-conv/stem-conv endpoints (`batchMap`-lifted
    by `FloatBridges.batchMap`), the supplied stem/head BN+swish backwards, and the three supplied MBConv
    block backwards. The `mnv2_grad_floatBridges` blueprint, batched. With this, all five Imagenette nets
    have BOTH forward and backward whole-net float bridges. -/
theorem efficientnet_grad_floatBridges (N : Nat) (M : FloatModel)
    (Ws : Kernel4 32 3 3 3) (Wh : Kernel4 1280 24 1 1) (Wfc : Mat 1280 10)
    (bnBs swBs : Vec (N * (32 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (bnBh swBh : Vec (N * (1280 * 56 * 56)) → Vec (N * (1280 * 56 * 56)))
    (b1B : Vec (N * (16 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (b2B : Vec (N * (24 * 56 * 56)) → Vec (N * (16 * 112 * 112)))
    (b3B : Vec (N * (24 * 56 * 56)) → Vec (N * (24 * 56 * 56)))
    {ws wh wfc : ℝ} (hws : 0 ≤ ws) (hwh : 0 ≤ wh) (hwfc : 0 ≤ wfc)
    (hWs : ∀ o c kh kw, |Ws o c kh kw| ≤ ws) (hWh : ∀ o c kh kw, |Wh o c kh kw| ≤ wh)
    (hWfc : ∀ i j, |Wfc i j| ≤ wfc)
    (hbnBs : FloatBridges bnBs) (hswBs : FloatBridges swBs)
    (hbnBh : FloatBridges bnBh) (hswBh : FloatBridges swBh)
    (hb1B : FloatBridges b1B) (hb2B : FloatBridges b2B) (hb3B : FloatBridges b3B) :
    FloatBridges (efficientnetInputGradB N Ws Wh Wfc bnBs swBs bnBh swBh b1B b2B b3B) := by
  unfold efficientnetInputGradB
  have hstem : FloatBridges
      (StableHLO.batchMap N (flatConvStride2Back (h := 112) (w := 112) Ws) ∘ bnBs ∘ swBs) :=
    (hswBs.comp hbnBs).comp
      (FloatBridges.batchMap N (floatBridges_flatConvStride2Back M Ws hws (by positivity) hWs))
  have hhead : FloatBridges
      (StableHLO.batchMap N (convFlatBack (h := 56) (w := 56) Wh) ∘ bnBh ∘ swBh) :=
    (hswBh.comp hbnBh).comp
      (FloatBridges.batchMap N (floatBridges_convBack (h := 56) (w := 56) M Wh hwh (by positivity) hWh))
  have h0 := (FloatBridges.batchMap N (floatBridges_linBack M Wfc hwfc (by norm_num) hWfc)).comp
    (FloatBridges.batchMap N (floatBridges_gapBack M 1280 56 56 (by norm_num) (by norm_num) (by norm_num)))
  have hH := h0.comp hhead
  have hB3 := hH.comp hb3B
  have hB2 := hB3.comp hb2B
  have hB1 := hB2.comp hb1B
  exact hB1.comp hstem

-- ════════════════════════════════════════════════════════════════
-- § The two non-residual MBConv block backwards (discharge `b1B`/`b2B` of the capstone)
--   `mbconvBatchedResidBack` already discharges the residual `b3B`; these are the no-expand
--   (`b1B`) and strided (`b2B`) peers, so every supplied block back is dischargeable (mnv2 parity).
-- ════════════════════════════════════════════════════════════════

/-- **The no-expand MBConv body backward** — the reverse of `mbNoExpFwdB`'s body `project ∘ SE ∘
    (swish∘bn∘depthwise)` (no expand conv): `depthwiseBack ∘ seBack ∘ projectBack`. `mbconvBodyBack`
    with the expand arm dropped. -/
noncomputable def mbNoExpBodyBack {cin cout h w kHd kWd kHp kWp : Nat}
    (Wd : DepthwiseKernel cin kHd kWd) (Wp : Kernel4 cout cin kHp kWp)
    (bnBd swBd seB : Vec (cin * h * w) → Vec (cin * h * w))
    (bnBp : Vec (cout * h * w) → Vec (cout * h * w)) :
    Vec (cout * h * w) → Vec (cin * h * w) :=
  (depthwiseFlatBack (h := h) (w := w) Wd ∘ bnBd ∘ swBd)
  ∘ seB
  ∘ (convFlatBack (h := h) (w := w) Wp ∘ bnBp)

/-- The no-expand MBConv body backward float-bridges (the `floatBridges_mbconvBodyBack` recipe minus
    the expand arm). Batched by `FloatBridges.batchMap` ⇒ discharges the capstone's `b1B`. -/
theorem floatBridges_mbNoExpBodyBack {cin cout h w kHd kWd kHp kWp : Nat} (M : FloatModel)
    (Wd : DepthwiseKernel cin kHd kWd) (Wp : Kernel4 cout cin kHp kWp)
    (bnBd swBd seB : Vec (cin * h * w) → Vec (cin * h * w))
    (bnBp : Vec (cout * h * w) → Vec (cout * h * w))
    {wd wp : ℝ} (hwd : 0 ≤ wd) (hwp : 0 ≤ wp)
    (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ wd) (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hcin : 0 < cin) (hcout : 0 < cout) (hh : 0 < h) (hw : 0 < w)
    (hbnBd : FloatBridges bnBd) (hswBd : FloatBridges swBd) (hseB : FloatBridges seB)
    (hbnBp : FloatBridges bnBp) :
    FloatBridges (mbNoExpBodyBack Wd Wp bnBd swBd seB bnBp) := by
  unfold mbNoExpBodyBack
  exact (((hbnBp.comp (floatBridges_convBack (h := h) (w := w) M Wp hwp (by positivity) hWp)).comp
    hseB).comp
    ((hswBd.comp hbnBd).comp (floatBridges_depthwiseBack (h := h) (w := w) M Wd hwd
      (by positivity) hWd)))

/-- **The strided MBConv body backward** — the reverse of `mbStridedFwdB`'s body `project ∘ SE ∘
    (swish∘bn∘depthwise-s2) ∘ (swish∘bn∘expand)`: `expandBack(@2h×2w) ∘ depthwiseStride2Back ∘ seBack
    ∘ projectBack`. `mbconvBodyBack` with the stride-1 depthwise replaced by the stride-2 (upsampling)
    backward and the expand arm at the pre-downsample `2h×2w`. -/
noncomputable def mbStridedBodyBack {cin cmid cout h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 cmid cin kHe kWe) (Wd : DepthwiseKernel cmid kHd kWd)
    (Wp : Kernel4 cout cmid kHp kWp)
    (bnBe swBe : Vec (cmid * (2 * h) * (2 * w)) → Vec (cmid * (2 * h) * (2 * w)))
    (bnBd swBd seB : Vec (cmid * h * w) → Vec (cmid * h * w))
    (bnBp : Vec (cout * h * w) → Vec (cout * h * w)) :
    Vec (cout * h * w) → Vec (cin * (2 * h) * (2 * w)) :=
  (convFlatBack (h := 2 * h) (w := 2 * w) We ∘ bnBe ∘ swBe)
  ∘ (depthwiseStride2FlatBack (h := h) (w := w) Wd ∘ bnBd ∘ swBd)
  ∘ seB
  ∘ (convFlatBack (h := h) (w := w) Wp ∘ bnBp)

/-- The strided MBConv body backward float-bridges (the `floatBridges_mbconvBodyBack` recipe with the
    stride-2 depthwise backward). Batched by `FloatBridges.batchMap` ⇒ discharges the capstone's `b2B`. -/
theorem floatBridges_mbStridedBodyBack {cin cmid cout h w kHe kWe kHd kWd kHp kWp : Nat} (M : FloatModel)
    (We : Kernel4 cmid cin kHe kWe) (Wd : DepthwiseKernel cmid kHd kWd)
    (Wp : Kernel4 cout cmid kHp kWp)
    (bnBe swBe : Vec (cmid * (2 * h) * (2 * w)) → Vec (cmid * (2 * h) * (2 * w)))
    (bnBd swBd seB : Vec (cmid * h * w) → Vec (cmid * h * w))
    (bnBp : Vec (cout * h * w) → Vec (cout * h * w))
    {we wd wp : ℝ} (hwe : 0 ≤ we) (hwd : 0 ≤ wd) (hwp : 0 ≤ wp)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ we) (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ wd)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hcmid : 0 < cmid) (hcout : 0 < cout) (hh : 0 < h) (hw : 0 < w)
    (hbnBe : FloatBridges bnBe) (hswBe : FloatBridges swBe)
    (hbnBd : FloatBridges bnBd) (hswBd : FloatBridges swBd) (hseB : FloatBridges seB)
    (hbnBp : FloatBridges bnBp) :
    FloatBridges (mbStridedBodyBack We Wd Wp bnBe swBe bnBd swBd seB bnBp) := by
  unfold mbStridedBodyBack
  exact ((((hbnBp.comp (floatBridges_convBack (h := h) (w := w) M Wp hwp (by positivity) hWp)).comp
    hseB).comp
    ((hswBd.comp hbnBd).comp (floatBridges_depthwiseStride2Back (h := h) (w := w) M Wd hwd
      (by positivity) hWd))).comp
    ((hswBe.comp hbnBe).comp (floatBridges_convBack (h := 2 * h) (w := 2 * w) M We hwe
      (by positivity) hWe)))

end Proofs
