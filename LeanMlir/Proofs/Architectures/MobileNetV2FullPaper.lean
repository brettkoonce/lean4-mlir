import LeanMlir.Proofs.MobileNetV2RenderPC

/-! # The PAPER-SPEC MobileNetV2 — all 17 bottleneck blocks, forward graph + faithfulness

Scales `MobileNetV2RenderPC.lean`'s reduced ch7 net (strided stem + 6 inverted-residual
blocks + 1×1 head) to the real MobileNetV2 `[t,c,n,s]` table — 17 bottleneck blocks at
224² — closing the "honest caveat on full" in `planning/mobilenetv2_close.md`. Pure
enumeration + chaining of the per-channel stage machinery (`ivExpandPC`/`ivDepthwisePC`/
`ivDepthwiseStridedPC`/`ivProjectPC`), the `EfficientNetFullB0` recipe; the only
genuinely-new block shape is the **t=1 first bottleneck** (no expand conv — depthwise →
BN → relu6 → project → BN, the torchvision/official layout).

Paper `[t,c,n,s]` spec (stem 3×3-s2 3→32; head 1×1 320→1280 → GAP → dense):
  (1, 16,1,1) (6, 24,2,2) (6, 32,3,2) (6, 64,4,2) (6, 96,3,1) (6,160,3,2) (6,320,1,1)
Per-block (ic→oc, mid=t·ic, spatial, kind):
  b1  32→16   mid32  @112 noExp(t=1)     b10 64→64   mid384 @14  resid
  b2  16→24   mid96  112→56 strided      b11 64→96   mid384 @14  exp(no-resid, s=1)
  b3  24→24   mid144 @56  resid          b12 96→96   mid576 @14  resid
  b4  24→32   mid144 56→28 strided       b13 96→96   mid576 @14  resid
  b5  32→32   mid192 @28  resid          b14 96→160  mid576 14→7 strided
  b6  32→32   mid192 @28  resid          b15 160→160 mid960 @7   resid
  b7  32→64   mid192 28→14 strided       b16 160→160 mid960 @7   resid
  b8  64→64   mid384 @14  resid          b17 160→320 mid960 @7   exp(no-resid, s=1)
  b9  64→64   mid384 @14  resid

Like ResNet-34's full net (`ResNet34RenderPC`), the deliverable is forward + graph +
faithfulness — relu6 is kinked, so the whole-net input-VJP stays pointwise-only (the
repo standard for relu-family nets); the param-grad close is already covered: every
`MobileNetV2Close`/`MobileNetV2ChainClose` bridge is dim-polymorphic and applies at the
paper shapes verbatim.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-block weight bundles (the 17-block net has 158 block params + stem/head/fc)
-- ════════════════════════════════════════════════════════════════

/-- Weights of one MobileNetV2 bottleneck (expand `ic→mid` 1×1, depthwise 3×3,
    project `mid→oc` 1×1, per-channel BN after each). -/
structure IVW (ic mid oc : Nat) where
  eW : Kernel4 mid ic 1 1
  eb : Vec mid
  eε : ℝ
  eγ : Vec mid
  eβ : Vec mid
  dW : DepthwiseKernel mid 3 3
  db : Vec mid
  dε : ℝ
  dγ : Vec mid
  dβ : Vec mid
  pW : Kernel4 oc mid 1 1
  pb : Vec oc
  pε : ℝ
  pγ : Vec oc
  pβ : Vec oc

/-- Weights of the t=1 first bottleneck (NO expand conv): depthwise 3×3 on `ic`,
    project `ic→oc` 1×1, per-channel BN after each. -/
structure IVWNoExp (ic oc : Nat) where
  dW : DepthwiseKernel ic 3 3
  db : Vec ic
  dε : ℝ
  dγ : Vec ic
  dβ : Vec ic
  pW : Kernel4 oc ic 1 1
  pb : Vec oc
  pε : ℝ
  pγ : Vec oc
  pβ : Vec oc

/-- All paper-spec MobileNetV2 parameters: stem (3×3-s2 3→32) + the 17 bottlenecks of
    the `[t,c,n,s]` table + head (1×1 320→1280) + dense (1280→10). -/
structure MNV2PaperWeights where
  sW : Kernel4 32 3 3 3
  sb : Vec 32
  sε : ℝ
  sγ : Vec 32
  sβ : Vec 32
  b1 : IVWNoExp 32 16
  b2 : IVW 16 96 24
  b3 : IVW 24 144 24
  b4 : IVW 24 144 32
  b5 : IVW 32 192 32
  b6 : IVW 32 192 32
  b7 : IVW 32 192 64
  b8 : IVW 64 384 64
  b9 : IVW 64 384 64
  b10 : IVW 64 384 64
  b11 : IVW 64 384 96
  b12 : IVW 96 576 96
  b13 : IVW 96 576 96
  b14 : IVW 96 576 160
  b15 : IVW 160 960 160
  b16 : IVW 160 960 160
  b17 : IVW 160 960 320
  hW : Kernel4 1280 320 1 1
  hb : Vec 1280
  hε : ℝ
  hγ : Vec 1280
  hβ : Vec 1280
  fcW : Mat 1280 10
  fcb : Vec 10

-- ════════════════════════════════════════════════════════════════
-- § Weight-bundle wrappers (forward ℝ-fns) — `(h w)` explicit, block dims from the bundle
-- ════════════════════════════════════════════════════════════════

/-- t=1 bottleneck (no expand, stride-1, no skip — `ic ≠ oc`): `project ∘ depthwise`. -/
noncomputable def ivNoExpW (h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  ivProjectPC (h := h) (w := w) p.pW p.pb p.pε p.pγ p.pβ ∘
    ivDepthwisePC (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ

/-- Stride-1 bottleneck WITHOUT skip (`ic ≠ oc`, the stage-first blocks of the s=1
    stages): plain `invresBodyPC`. -/
noncomputable def ivExpOnlyW (h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  invresBodyPC (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ p.dW p.db p.dε p.dγ p.dβ
    p.pW p.pb p.pε p.pγ p.pβ

/-- Stride-1 bottleneck WITH the identity skip (`s = 1 ∧ ic = oc`). -/
noncomputable def ivResidW (h w : Nat) {c mid : Nat} (p : IVW c mid c) :
    Vec (c * h * w) → Vec (c * h * w) :=
  residual (invresBodyPC (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ p.dW p.db p.dε p.dγ p.dβ
    p.pW p.pb p.pε p.pγ p.pβ)

/-- Stride-2 downsampling bottleneck (no skip). -/
noncomputable def ivStridedW (h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  invresBodyStridedPC (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ p.dW p.db p.dε p.dγ p.dβ
    p.pW p.pb p.pε p.pγ p.pβ

-- ════════════════════════════════════════════════════════════════
-- § The full paper-spec ℝ-forward — all 17 bottlenecks, nested-application form
--   stem(224→112) → b1@112 → b2(112→56) → b3@56 → b4(56→28) → b5,b6@28 → b7(28→14)
--   → b8..b13@14 → b14(14→7) → b15,b16,b17@7 → head@7 → GAP → dense
-- ════════════════════════════════════════════════════════════════

noncomputable def mobilenetv2ForwardPaper (w : MNV2PaperWeights)
    (x : Vec (3 * 224 * 224)) : Vec 10 :=
  dense w.fcW w.fcb
    (globalAvgPoolFlat 1280 7 7
      (relu6 (1280 * 7 * 7) (bnPerChannelTensor3 1280 7 7 w.hε w.hγ w.hβ
        (flatConv (h := 7) (w := 7) w.hW w.hb
          (ivExpOnlyW 7 7 w.b17
            (ivResidW 7 7 w.b16
              (ivResidW 7 7 w.b15
                (ivStridedW 7 7 w.b14
                  (ivResidW 14 14 w.b13
                    (ivResidW 14 14 w.b12
                      (ivExpOnlyW 14 14 w.b11
                        (ivResidW 14 14 w.b10
                          (ivResidW 14 14 w.b9
                            (ivResidW 14 14 w.b8
                              (ivStridedW 14 14 w.b7
                                (ivResidW 28 28 w.b6
                                  (ivResidW 28 28 w.b5
                                    (ivStridedW 28 28 w.b4
                                      (ivResidW 56 56 w.b3
                                        (ivStridedW 56 56 w.b2
                                          (ivNoExpW 112 112 w.b1
                                            (relu6 (32 * 112 * 112)
                                              (bnPerChannelTensor3 32 112 112 w.sε w.sγ w.sβ
                                                (flatConvStride2 (h := 112) (w := 112)
                                                  w.sW w.sb x))))))))))))))))))))))))

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Per-block-kind typed `SHlo` graphs + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- t=1 bottleneck graph: `bnPC ∘ conv1×1 ∘ relu6 ∘ bnPC ∘ depthwise`. -/
def ivNoExpGraphW (pfx epsStr : String) (h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (e : SHlo (ic * h * w)) : SHlo (oc * h * w) :=
  .bnPerChannelF (oc := oc) (h := h) (w := w) s!"%{pfx}gp" s!"%{pfx}btp" epsStr p.pε p.pγ p.pβ
    (.flatConvF (h := h) (w := w) s!"%{pfx}Wp" s!"%{pfx}bp" p.pW p.pb
      (.relu6F (.bnPerChannelF (oc := ic) (h := h) (w := w) s!"%{pfx}gd" s!"%{pfx}btd" epsStr
          p.dε p.dγ p.dβ
        (.depthwiseF (h := h) (w := w) s!"%{pfx}Wd" s!"%{pfx}bd" p.dW p.db e))))

theorem ivNoExpGraphW_faithful (pfx epsStr : String) (h w : Nat) {ic oc : Nat}
    (p : IVWNoExp ic oc) (e : SHlo (ic * h * w)) :
    den (ivNoExpGraphW pfx epsStr h w p e) = ivNoExpW h w p (den e) := by
  unfold ivNoExpGraphW ivNoExpW
  simp only [bnPerChannelF_faithful, flatConvF_faithful, relu6F_faithful, depthwiseF_faithful]
  simp only [ivProjectPC, ivDepthwisePC, Function.comp_apply]

/-- Stride-1 no-skip bottleneck graph: expand → depthwise → project (per-channel BN). -/
def ivExpOnlyGraphW (pfx epsStr : String) (h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (e : SHlo (ic * h * w)) : SHlo (oc * h * w) :=
  .bnPerChannelF (oc := oc) (h := h) (w := w) s!"%{pfx}gp" s!"%{pfx}btp" epsStr p.pε p.pγ p.pβ
    (.flatConvF (h := h) (w := w) s!"%{pfx}Wp" s!"%{pfx}bp" p.pW p.pb
      (.relu6F (.bnPerChannelF (oc := mid) (h := h) (w := w) s!"%{pfx}gd" s!"%{pfx}btd" epsStr
          p.dε p.dγ p.dβ
        (.depthwiseF (h := h) (w := w) s!"%{pfx}Wd" s!"%{pfx}bd" p.dW p.db
          (.relu6F (.bnPerChannelF (oc := mid) (h := h) (w := w) s!"%{pfx}ge" s!"%{pfx}bte" epsStr
              p.eε p.eγ p.eβ
            (.flatConvF (h := h) (w := w) s!"%{pfx}We" s!"%{pfx}be" p.eW p.eb e)))))))

theorem ivExpOnlyGraphW_faithful (pfx epsStr : String) (h w : Nat) {ic mid oc : Nat}
    (p : IVW ic mid oc) (e : SHlo (ic * h * w)) :
    den (ivExpOnlyGraphW pfx epsStr h w p e) = ivExpOnlyW h w p (den e) := by
  unfold ivExpOnlyGraphW ivExpOnlyW
  simp only [bnPerChannelF_faithful, flatConvF_faithful, relu6F_faithful, depthwiseF_faithful]
  simp only [invresBodyPC, ivExpandPC, ivDepthwisePC, ivProjectPC, Function.comp_apply]

/-- Stride-1 skip bottleneck graph: the body + the `addV` identity skip (input subtree
    shared between both arms). -/
def ivResidGraphW (pfx epsStr : String) (h w : Nat) {c mid : Nat} (p : IVW c mid c)
    (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  .addV (ivExpOnlyGraphW pfx epsStr h w p e) e

theorem ivResidGraphW_faithful (pfx epsStr : String) (h w : Nat) {c mid : Nat}
    (p : IVW c mid c) (e : SHlo (c * h * w)) :
    den (ivResidGraphW pfx epsStr h w p e) = ivResidW h w p (den e) := by
  unfold ivResidGraphW ivResidW
  simp only [den_addV, ivExpOnlyGraphW_faithful]
  unfold ivExpOnlyW residual biPath
  rfl

/-- Stride-2 downsampling bottleneck graph: expand (at `2h×2w`) → strided depthwise →
    project (per-channel BN). -/
def ivStridedGraphW (pfx epsStr : String) (h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (e : SHlo (ic * (2 * h) * (2 * w))) : SHlo (oc * h * w) :=
  .bnPerChannelF (oc := oc) (h := h) (w := w) s!"%{pfx}gp" s!"%{pfx}btp" epsStr p.pε p.pγ p.pβ
    (.flatConvF (h := h) (w := w) s!"%{pfx}Wp" s!"%{pfx}bp" p.pW p.pb
      (.relu6F (.bnPerChannelF (oc := mid) (h := h) (w := w) s!"%{pfx}gd" s!"%{pfx}btd" epsStr
          p.dε p.dγ p.dβ
        (.depthwiseStridedF (h := h) (w := w) s!"%{pfx}Wd" s!"%{pfx}bd" p.dW p.db
          (.relu6F (.bnPerChannelF (oc := mid) (h := 2 * h) (w := 2 * w) s!"%{pfx}ge"
              s!"%{pfx}bte" epsStr p.eε p.eγ p.eβ
            (.flatConvF (h := 2 * h) (w := 2 * w) s!"%{pfx}We" s!"%{pfx}be" p.eW p.eb e)))))))

theorem ivStridedGraphW_faithful (pfx epsStr : String) (h w : Nat) {ic mid oc : Nat}
    (p : IVW ic mid oc) (e : SHlo (ic * (2 * h) * (2 * w))) :
    den (ivStridedGraphW pfx epsStr h w p e) = ivStridedW h w p (den e) := by
  unfold ivStridedGraphW ivStridedW
  simp only [bnPerChannelF_faithful, flatConvF_faithful, relu6F_faithful,
             depthwiseStridedF_faithful]
  simp only [invresBodyStridedPC, ivExpandPC, ivDepthwiseStridedPC, ivProjectPC,
             Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The full paper-spec forward graph + faithfulness (all 17 bottlenecks)
-- ════════════════════════════════════════════════════════════════

/-- The full **paper-spec MobileNetV2 forward graph** (3×224² → 10): strided stem →
    the 17 bottlenecks of the `[t,c,n,s]` table (4 stride-2 downsamples, 10 identity
    skips, 2 stage-first s=1 widenings, the t=1 no-expand first block) → 1×1
    conv-bn-relu6 head → GAP → dense. Per-channel BN throughout. -/
def mobilenetv2FwdGraphPaper (epsStr : String) (w : MNV2PaperWeights)
    (x : Vec (3 * 224 * 224)) : SHlo 10 :=
  denseF "%Wfc" "%bfc" w.fcW w.fcb
    (.gapF (c := 1280) (h := 7) (w := 7)
      (.relu6F (.bnPerChannelF (oc := 1280) (h := 7) (w := 7) "%gh" "%bth" epsStr
          w.hε w.hγ w.hβ
        (.flatConvF (h := 7) (w := 7) "%Wh" "%bh" w.hW w.hb
          (ivExpOnlyGraphW "b17" epsStr 7 7 w.b17
            (ivResidGraphW "b16" epsStr 7 7 w.b16
              (ivResidGraphW "b15" epsStr 7 7 w.b15
                (ivStridedGraphW "b14" epsStr 7 7 w.b14
                  (ivResidGraphW "b13" epsStr 14 14 w.b13
                    (ivResidGraphW "b12" epsStr 14 14 w.b12
                      (ivExpOnlyGraphW "b11" epsStr 14 14 w.b11
                        (ivResidGraphW "b10" epsStr 14 14 w.b10
                          (ivResidGraphW "b9" epsStr 14 14 w.b9
                            (ivResidGraphW "b8" epsStr 14 14 w.b8
                              (ivStridedGraphW "b7" epsStr 14 14 w.b7
                                (ivResidGraphW "b6" epsStr 28 28 w.b6
                                  (ivResidGraphW "b5" epsStr 28 28 w.b5
                                    (ivStridedGraphW "b4" epsStr 28 28 w.b4
                                      (ivResidGraphW "b3" epsStr 56 56 w.b3
                                        (ivStridedGraphW "b2" epsStr 56 56 w.b2
                                          (ivNoExpGraphW "b1" epsStr 112 112 w.b1
                                            (.relu6F (.bnPerChannelF (oc := 32) (h := 112)
                                                (w := 112) "%gs" "%bts" epsStr w.sε w.sγ w.sβ
                                              (.flatConvStridedF (h := 112) (w := 112)
                                                "%Ws" "%bs" w.sW w.sb
                                                (.operand "%x" x)))))))))))))))))))))))))

/-- **Full paper-spec MobileNetV2 forward faithfulness.** The 17-bottleneck graph
    denotes `mobilenetv2ForwardPaper` — chained from the per-block-kind `*GraphW_faithful`
    lemmas (the `EfficientNetFullB0` recipe), then a structural `rfl` (the forward is
    nested-application form, blocks opaque). -/
theorem mobilenetv2FwdGraphPaper_faithful (epsStr : String) (w : MNV2PaperWeights)
    (x : Vec (3 * 224 * 224)) :
    den (mobilenetv2FwdGraphPaper epsStr w x) = mobilenetv2ForwardPaper w x := by
  simp only [mobilenetv2FwdGraphPaper, denseF_faithful, gapF_faithful, relu6F_faithful,
             bnPerChannelF_faithful, flatConvF_faithful, flatConvStridedF_faithful,
             ivExpOnlyGraphW_faithful, ivResidGraphW_faithful, ivStridedGraphW_faithful,
             ivNoExpGraphW_faithful, den_operand]
  rfl

end StableHLO
end Proofs
