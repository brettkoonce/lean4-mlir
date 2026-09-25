import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetChainClose

/-! # The FULL EfficientNet-B0 — all 16 MBConv blocks, batched forward graph + faithfulness

Scales `EfficientNetRenderPC.lean`'s representative (stem + 3 MBConv + head) to the real B0
`[t,c,n,s,k]` spec — 16 MBConv layers — at the batched index `N·(c·h·w)`. Pure enumeration + chaining
of the generic per-block machinery; the only genuinely-new piece is the 4th block shape
(`mbExp`: expand + stride-1 + **no** residual, used by stage-5/stage-7 first blocks where `ic ≠ oc`).

B0 stage spec `[t,c,n,s,k]`:
  s1 (1,16,1,1,3) s2 (6,24,2,2,3) s3 (6,40,2,2,5) s4 (6,80,3,2,3)
  s5 (6,112,3,1,5) s6 (6,192,4,2,5) s7 (6,320,1,1,3); stem 3×3-s2 (3→32), head 1×1 (320→1280)→GAP→dense.
Per-block (ic, mid=t·ic, oc, r=⌈ic/4⌉, k, spatial, kind):
  b1  32→16   mid32  r8  k3 @112  noExp           b9  80→112  mid480  r20 k5 @14  exp(no-resid)
  b2  16→24   mid96  r4  k3 112→56 strided         b10 112→112 mid672  r28 k5 @14  resid
  b3  24→24   mid144 r6  k3 @56    resid            b11 112→112 mid672  r28 k5 @14  resid
  b4  24→40   mid144 r6  k5 56→28  strided          b12 112→192 mid672  r28 k5 14→7 strided
  b5  40→40   mid240 r10 k5 @28    resid            b13 192→192 mid1152 r48 k5 @7   resid
  b6  40→80   mid240 r10 k3 28→14  strided          b14 192→192 mid1152 r48 k5 @7   resid
  b7  80→80   mid480 r20 k3 @14    resid            b15 192→192 mid1152 r48 k5 @7   resid
  b8  80→80   mid480 r20 k3 @14    resid            b16 192→320 mid1152 r48 k3 @7   exp(no-resid)
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-block weight bundles (so the 262-param net stays manageable)
-- ════════════════════════════════════════════════════════════════

/-- Weights of one MBConv6 block (expand `ic→mid`, depthwise `k×k`, SE `mid→r→mid`, project `mid→oc`). -/
structure MBW (ic mid oc r kh kw : Nat) where
  eW : Kernel4 mid ic 1 1
  eb : Vec mid
  eε : ℝ
  eγ : Vec mid
  eβ : Vec mid
  dW : DepthwiseKernel mid kh kw
  db : Vec mid
  dε : ℝ
  dγ : Vec mid
  dβ : Vec mid
  z1 : Mat mid r
  zb1 : Vec r
  z2 : Mat r mid
  zb2 : Vec mid
  pW : Kernel4 oc mid 1 1
  pb : Vec oc
  pε : ℝ
  pγ : Vec oc
  pβ : Vec oc

/-- Weights of the MBConv1 block (`t=1`, no expand; depthwise on `ic`, SE `ic→r→ic`, project `ic→oc`). -/
structure MBWNoExp (ic oc r kh kw : Nat) where
  dW : DepthwiseKernel ic kh kw
  db : Vec ic
  dε : ℝ
  dγ : Vec ic
  dβ : Vec ic
  z1 : Mat ic r
  zb1 : Vec r
  z2 : Mat r ic
  zb2 : Vec ic
  pW : Kernel4 oc ic 1 1
  pb : Vec oc
  pε : ℝ
  pγ : Vec oc
  pβ : Vec oc

/-- All 262 EfficientNet-B0 parameters: stem (3×3-s2 3→32) + 16 MBConv blocks (the real `[t,c,n,s,k]`
    spec) + head (1×1 320→1280) + dense (1280→10). -/
structure B0Weights where
  sW : Kernel4 32 3 3 3
  sb : Vec 32
  sε : ℝ
  sγ : Vec 32
  sβ : Vec 32
  b1 : MBWNoExp 32 16 8 3 3
  b2 : MBW 16 96 24 4 3 3
  b3 : MBW 24 144 24 6 3 3
  b4 : MBW 24 144 40 6 5 5
  b5 : MBW 40 240 40 10 5 5
  b6 : MBW 40 240 80 10 3 3
  b7 : MBW 80 480 80 20 3 3
  b8 : MBW 80 480 80 20 3 3
  b9 : MBW 80 480 112 20 5 5
  b10 : MBW 112 672 112 28 5 5
  b11 : MBW 112 672 112 28 5 5
  b12 : MBW 112 672 192 28 5 5
  b13 : MBW 192 1152 192 48 5 5
  b14 : MBW 192 1152 192 48 5 5
  b15 : MBW 192 1152 192 48 5 5
  b16 : MBW 192 1152 320 48 3 3
  hW : Kernel4 1280 320 1 1
  hb : Vec 1280
  hε : ℝ
  hγ : Vec 1280
  hβ : Vec 1280
  fcW : Mat 1280 10
  fcb : Vec 10

/-- The three BatchNorm `ε`s of an MBConv6 block are positive. -/
structure MBW.EpsPos {ic mid oc r kh kw : Nat} (b : MBW ic mid oc r kh kw) : Prop where
  e : 0 < b.eε
  d : 0 < b.dε
  p : 0 < b.pε

/-- The two BatchNorm `ε`s of the MBConv1 block are positive. -/
structure MBWNoExp.EpsPos {ic oc r kh kw : Nat} (b : MBWNoExp ic oc r kh kw) : Prop where
  d : 0 < b.dε
  p : 0 < b.pε

/-- All 49 BatchNorm `ε`s of EfficientNet-B0 are positive: stem, the 16 blocks, head. -/
structure B0Weights.EpsPos (w : B0Weights) : Prop where
  s : 0 < w.sε
  b1 : w.b1.EpsPos
  b2 : w.b2.EpsPos
  b3 : w.b3.EpsPos
  b4 : w.b4.EpsPos
  b5 : w.b5.EpsPos
  b6 : w.b6.EpsPos
  b7 : w.b7.EpsPos
  b8 : w.b8.EpsPos
  b9 : w.b9.EpsPos
  b10 : w.b10.EpsPos
  b11 : w.b11.EpsPos
  b12 : w.b12.EpsPos
  b13 : w.b13.EpsPos
  b14 : w.b14.EpsPos
  b15 : w.b15.EpsPos
  b16 : w.b16.EpsPos
  h : 0 < w.hε

-- ════════════════════════════════════════════════════════════════
-- § The 4th block shape — MBConv6 expand + stride-1 + NO residual (`ic ≠ oc`, stages 5/7 first
--   block); `mbExpFwdB` and its VJP are in `EfficientNetChainClose`
-- ════════════════════════════════════════════════════════════════

namespace StableHLO

/-- MBConv6 expand + stride-1 + no-residual graph (the `mbResidGraphB` body without the `addV` skip). -/
def mbExpGraphB (p epsStr : String) {N ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (e : SHlo (N * (ic * h * w))) : SHlo (N * (oc * h * w)) :=
  .bnBatchF s!"%{p}pg" s!"%{p}pbt" epsStr εp γp βp
    (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" (biasName false "" oc) Wp bp)
      (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zW1" s!"%{p}zb1" s!"%{p}zW2" s!"%{p}zb2"
          Wz₁ bz₁ Wz₂ bz₂)
        (.batchOp (N := N) .swish (.bnBatchF s!"%{p}dg" s!"%{p}dbt" epsStr εd γd βd
          (.batchOp (N := N) (.depthwise (h := h) (w := w) s!"%{p}dW" (biasName false "" mid) Wd bd)
            (.batchOp (N := N) .swish (.bnBatchF s!"%{p}eg" s!"%{p}ebt" epsStr εe γe βe
              (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}eW" (biasName false "" mid) We be) e))))))))

theorem mbExpGraphB_faithful (p epsStr : String) {N ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (e : SHlo (N * (ic * h * w))) :
    den (mbExpGraphB p epsStr We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp e)
      = mbExpFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂
          Wp bp εp γp βp (den e) := by
  unfold mbExpGraphB mbExpFwdB projB seB dwbsB cbsB
  simp only [den_batchOp, denOp, den_bnBatchF,
             ↓den_batchOp_swish_eq_swishF, swishF_faithful, Function.comp_apply]

end StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Weight-bundle wrappers (forward ℝ-fns) — `(N h w)` explicit, block dims from the bundle
-- ════════════════════════════════════════════════════════════════

noncomputable def mbNoExpW (N h w : Nat) {ic oc kh kw r : Nat} (p : MBWNoExp ic oc r kh kw) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  mbNoExpFwdB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε p.pγ p.pβ

noncomputable def mbStridedW (N h w : Nat) {ic mid oc kh kw r : Nat} (p : MBW ic mid oc r kh kw) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  mbStridedFwdB N (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ p.dW p.db p.dε p.dγ p.dβ
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε p.pγ p.pβ

noncomputable def mbResidW (N h w : Nat) {c mid kh kw r : Nat} (p : MBW c mid c r kh kw) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  mbResidFwdB N (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ p.dW p.db p.dε p.dγ p.dβ
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε p.pγ p.pβ

noncomputable def mbExpW (N h w : Nat) {ic mid oc kh kw r : Nat} (p : MBW ic mid oc r kh kw) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  mbExpFwdB N (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ p.dW p.db p.dε p.dγ p.dβ
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε p.pγ p.pβ

-- § Weight-bundle wrappers — differentiability + VJP (each = the per-block lemma at the bundle's fields)

theorem mbNoExpW_differentiable (N h w : Nat) {ic oc kh kw r : Nat} (p : MBWNoExp ic oc r kh kw)
    (hd : 0 < p.dε) (hp : 0 < p.pε) : Differentiable ℝ (mbNoExpW N h w p) := by
  unfold mbNoExpW
  exact mbNoExpFwdB_differentiable N (h := h) (w := w) p.dW p.db p.dε hd p.dγ p.dβ
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ
noncomputable def mbNoExpWHasVJP (N h w : Nat) {ic oc kh kw r : Nat} (p : MBWNoExp ic oc r kh kw)
    (hd : 0 < p.dε) (hp : 0 < p.pε) : HasVJP (mbNoExpW N h w p) := by
  unfold mbNoExpW
  exact mbNoExpFwdBHasVJP N (h := h) (w := w) p.dW p.db p.dε hd p.dγ p.dβ
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ

theorem mbStridedW_differentiable (N h w : Nat) {ic mid oc kh kw r : Nat} (p : MBW ic mid oc r kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) : Differentiable ℝ (mbStridedW N h w p) := by
  unfold mbStridedW
  exact mbStridedFwdB_differentiable N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ
    p.dW p.db p.dε hd p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ
noncomputable def mbStridedWHasVJP (N h w : Nat) {ic mid oc kh kw r : Nat} (p : MBW ic mid oc r kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) : HasVJP (mbStridedW N h w p) := by
  unfold mbStridedW
  exact mbStridedFwdBHasVJP N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ
    p.dW p.db p.dε hd p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ

theorem mbResidW_differentiable (N h w : Nat) {c mid kh kw r : Nat} (p : MBW c mid c r kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) : Differentiable ℝ (mbResidW N h w p) := by
  unfold mbResidW
  exact mbResidFwdB_differentiable N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ
    p.dW p.db p.dε hd p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ
noncomputable def mbResidWHasVJP (N h w : Nat) {c mid kh kw r : Nat} (p : MBW c mid c r kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) : HasVJP (mbResidW N h w p) := by
  unfold mbResidW
  exact mbResidFwdBHasVJP N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ
    p.dW p.db p.dε hd p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ

theorem mbExpW_differentiable (N h w : Nat) {ic mid oc kh kw r : Nat} (p : MBW ic mid oc r kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) : Differentiable ℝ (mbExpW N h w p) := by
  unfold mbExpW
  exact mbExpFwdB_differentiable N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ
    p.dW p.db p.dε hd p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ
noncomputable def mbExpWHasVJP (N h w : Nat) {ic mid oc kh kw r : Nat} (p : MBW ic mid oc r kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) : HasVJP (mbExpW N h w p) := by
  unfold mbExpW
  exact mbExpFwdBHasVJP N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ
    p.dW p.db p.dε hd p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ

-- ════════════════════════════════════════════════════════════════
-- § The full B0 ℝ-forward — all 16 MBConv blocks, nested-application form
--   stem(224→112) → b1@112 → b2(112→56) → b3@56 → b4(56→28) → b5@28 → b6(28→14) → b7,b8,b9,b10,b11@14
--   → b12(14→7) → b13,b14,b15,b16@7 → head@7 → GAP → dense
-- ════════════════════════════════════════════════════════════════

noncomputable def efficientnetForwardBFull (N : Nat) (w : B0Weights)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * 10) :=
  headFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb
    (mbExpW N 7 7 w.b16
      (mbResidW N 7 7 w.b15
        (mbResidW N 7 7 w.b14
          (mbResidW N 7 7 w.b13
            (mbStridedW N 7 7 w.b12
              (mbResidW N 14 14 w.b11
                (mbResidW N 14 14 w.b10
                  (mbExpW N 14 14 w.b9
                    (mbResidW N 14 14 w.b8
                      (mbResidW N 14 14 w.b7
                        (mbStridedW N 14 14 w.b6
                          (mbResidW N 28 28 w.b5
                            (mbStridedW N 28 28 w.b4
                              (mbResidW N 56 56 w.b3
                                (mbStridedW N 56 56 w.b2
                                  (mbNoExpW N 112 112 w.b1
                                    (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ x)))))))))))))))))

namespace StableHLO

-- § Weight-bundle wrappers (graph) + faithfulness (each = the per-block lemma at the bundle's fields)

def mbNoExpGraphW (pfx epsStr : String) (N h w : Nat) {ic oc kh kw r : Nat} (p : MBWNoExp ic oc r kh kw)
    (e : SHlo (N * (ic * h * w))) : SHlo (N * (oc * h * w)) :=
  mbNoExpGraphB pfx epsStr (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2
    p.pW p.pb p.pε p.pγ p.pβ e
theorem mbNoExpGraphW_faithful (pfx epsStr : String) (N h w : Nat) {ic oc kh kw r : Nat}
    (p : MBWNoExp ic oc r kh kw) (e : SHlo (N * (ic * h * w))) :
    den (mbNoExpGraphW pfx epsStr N h w p e) = mbNoExpW N h w p (den e) := by
  unfold mbNoExpGraphW mbNoExpW
  exact mbNoExpGraphB_faithful pfx epsStr p.dW p.db p.dε p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2
    p.pW p.pb p.pε p.pγ p.pβ e

def mbStridedGraphW (pfx epsStr : String) (N h w : Nat) {ic mid oc kh kw r : Nat}
    (p : MBW ic mid oc r kh kw) (e : SHlo (N * (ic * (2 * h) * (2 * w)))) : SHlo (N * (oc * h * w)) :=
  mbStridedGraphB pfx epsStr (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ p.dW p.db p.dε p.dγ p.dβ
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε p.pγ p.pβ e
theorem mbStridedGraphW_faithful (pfx epsStr : String) (N h w : Nat) {ic mid oc kh kw r : Nat}
    (p : MBW ic mid oc r kh kw) (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (mbStridedGraphW pfx epsStr N h w p e) = mbStridedW N h w p (den e) := by
  unfold mbStridedGraphW mbStridedW
  exact mbStridedGraphB_faithful pfx epsStr p.eW p.eb p.eε p.eγ p.eβ p.dW p.db p.dε p.dγ p.dβ
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε p.pγ p.pβ e

def mbResidGraphW (pfx epsStr : String) (N h w : Nat) {c mid kh kw r : Nat} (p : MBW c mid c r kh kw)
    (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  mbResidGraphB pfx epsStr (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ p.dW p.db p.dε p.dγ p.dβ
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε p.pγ p.pβ e
theorem mbResidGraphW_faithful (pfx epsStr : String) (N h w : Nat) {c mid kh kw r : Nat}
    (p : MBW c mid c r kh kw) (e : SHlo (N * (c * h * w))) :
    den (mbResidGraphW pfx epsStr N h w p e) = mbResidW N h w p (den e) := by
  unfold mbResidGraphW mbResidW
  exact mbResidGraphB_faithful pfx epsStr p.eW p.eb p.eε p.eγ p.eβ p.dW p.db p.dε p.dγ p.dβ
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε p.pγ p.pβ e

def mbExpGraphW (pfx epsStr : String) (N h w : Nat) {ic mid oc kh kw r : Nat} (p : MBW ic mid oc r kh kw)
    (e : SHlo (N * (ic * h * w))) : SHlo (N * (oc * h * w)) :=
  mbExpGraphB pfx epsStr (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ p.dW p.db p.dε p.dγ p.dβ
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε p.pγ p.pβ e
theorem mbExpGraphW_faithful (pfx epsStr : String) (N h w : Nat) {ic mid oc kh kw r : Nat}
    (p : MBW ic mid oc r kh kw) (e : SHlo (N * (ic * h * w))) :
    den (mbExpGraphW pfx epsStr N h w p e) = mbExpW N h w p (den e) := by
  unfold mbExpGraphW mbExpW
  exact mbExpGraphB_faithful pfx epsStr p.eW p.eb p.eε p.eγ p.eβ p.dW p.db p.dε p.dγ p.dβ
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε p.pγ p.pβ e

-- ════════════════════════════════════════════════════════════════
-- § The full B0 batched forward graph + faithfulness (all 16 MBConv blocks)
-- ════════════════════════════════════════════════════════════════

/-- The full **batched EfficientNet-B0 forward graph** at the batched index `N·(c·h·w)`: stem → 16
    MBConv blocks (the real `[t,c,n,s,k]` spec, 3×3 and 5×5 depthwise, true batch-norm, squeeze-excite,
    4 stride-2 downsamples, identity residuals where `s=1 ∧ ic=oc`) → head → GAP → dense. -/
def efficientnetFwdGraphBFull (N : Nat) (epsStr : String) (w : B0Weights)
    (x : Vec (N * (3 * 224 * 224))) : SHlo (N * 10) :=
  headGraphB epsStr (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb
    (mbExpGraphW "b16" epsStr N 7 7 w.b16
      (mbResidGraphW "b15" epsStr N 7 7 w.b15
        (mbResidGraphW "b14" epsStr N 7 7 w.b14
          (mbResidGraphW "b13" epsStr N 7 7 w.b13
            (mbStridedGraphW "b12" epsStr N 7 7 w.b12
              (mbResidGraphW "b11" epsStr N 14 14 w.b11
                (mbResidGraphW "b10" epsStr N 14 14 w.b10
                  (mbExpGraphW "b9" epsStr N 14 14 w.b9
                    (mbResidGraphW "b8" epsStr N 14 14 w.b8
                      (mbResidGraphW "b7" epsStr N 14 14 w.b7
                        (mbStridedGraphW "b6" epsStr N 14 14 w.b6
                          (mbResidGraphW "b5" epsStr N 28 28 w.b5
                            (mbStridedGraphW "b4" epsStr N 28 28 w.b4
                              (mbResidGraphW "b3" epsStr N 56 56 w.b3
                                (mbStridedGraphW "b2" epsStr N 56 56 w.b2
                                  (mbNoExpGraphW "b1" epsStr N 112 112 w.b1
                                    (stemGraphB epsStr (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ
                                      (.operand "%x" x))))))))))))))))))

/-- **Full batched EfficientNet-B0 forward faithfulness.** The full 16-MBConv batched graph (true
    batch-norm + SE) denotes `efficientnetForwardBFull`. Chained from the per-block `*GraphW_faithful`
    lemmas (one `rw` per block, outermost→innermost), then a structural `rfl` (the forward is
    nested-application form, blocks opaque) — the `ResNet34RenderPC.lean` recipe (since retired) at full depth. -/
theorem efficientnetFwdGraphBFull_faithful (N : Nat) (epsStr : String) (w : B0Weights)
    (x : Vec (N * (3 * 224 * 224))) :
    den (efficientnetFwdGraphBFull N epsStr w x) = efficientnetForwardBFull N w x := by
  rw [efficientnetFwdGraphBFull, headGraphB_faithful,
      mbExpGraphW_faithful, mbResidGraphW_faithful, mbResidGraphW_faithful, mbResidGraphW_faithful,
      mbStridedGraphW_faithful, mbResidGraphW_faithful, mbResidGraphW_faithful, mbExpGraphW_faithful,
      mbResidGraphW_faithful, mbResidGraphW_faithful, mbStridedGraphW_faithful, mbResidGraphW_faithful,
      mbStridedGraphW_faithful, mbResidGraphW_faithful, mbStridedGraphW_faithful, mbNoExpGraphW_faithful,
      stemGraphB_faithful, den_operand]
  rfl

end StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The full B0 VJP — all 16 MBConv blocks (the full-depth analogue of `efficientnetHasVJP`)
-- ════════════════════════════════════════════════════════════════

/-- **The full EfficientNet-B0 has a (correct) VJP.** Chained from the per-block gradients (stem → 16
    MBConv blocks → head) via `vjpComp`. Stated on the `∘`-composition of the blocks (= the full
    forward by construction; keeps the blocks opaque so the chain closes structurally). The full-depth,
    batched, true-batch-norm + SE analogue of `efficientnetHasVJP`. -/
noncomputable def efficientnetForwardBFullHasVJP (N : Nat) (w : B0Weights)
    (hεw : w.EpsPos) :
    HasVJP
      (headFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb ∘
        mbExpW N 7 7 w.b16 ∘ mbResidW N 7 7 w.b15 ∘ mbResidW N 7 7 w.b14 ∘ mbResidW N 7 7 w.b13 ∘
        mbStridedW N 7 7 w.b12 ∘ mbResidW N 14 14 w.b11 ∘ mbResidW N 14 14 w.b10 ∘
        mbExpW N 14 14 w.b9 ∘ mbResidW N 14 14 w.b8 ∘ mbResidW N 14 14 w.b7 ∘
        mbStridedW N 14 14 w.b6 ∘ mbResidW N 28 28 w.b5 ∘ mbStridedW N 28 28 w.b4 ∘
        mbResidW N 56 56 w.b3 ∘ mbStridedW N 56 56 w.b2 ∘ mbNoExpW N 112 112 w.b1 ∘
        stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) := by
  have dS := stemB_differentiable N (h := 112) (w := 112) w.sW w.sb w.sε hεw.s w.sγ w.sβ
  have vS := stemBHasVJP N (h := 112) (w := 112) w.sW w.sb w.sε hεw.s w.sγ w.sβ
  have d1 := mbNoExpW_differentiable N 112 112 w.b1 hεw.b1.d hεw.b1.p
  have v1 := mbNoExpWHasVJP N 112 112 w.b1 hεw.b1.d hεw.b1.p
  have d2 := mbStridedW_differentiable N 56 56 w.b2 hεw.b2.e hεw.b2.d hεw.b2.p
  have v2 := mbStridedWHasVJP N 56 56 w.b2 hεw.b2.e hεw.b2.d hεw.b2.p
  have d3 := mbResidW_differentiable N 56 56 w.b3 hεw.b3.e hεw.b3.d hεw.b3.p
  have v3 := mbResidWHasVJP N 56 56 w.b3 hεw.b3.e hεw.b3.d hεw.b3.p
  have d4 := mbStridedW_differentiable N 28 28 w.b4 hεw.b4.e hεw.b4.d hεw.b4.p
  have v4 := mbStridedWHasVJP N 28 28 w.b4 hεw.b4.e hεw.b4.d hεw.b4.p
  have d5 := mbResidW_differentiable N 28 28 w.b5 hεw.b5.e hεw.b5.d hεw.b5.p
  have v5 := mbResidWHasVJP N 28 28 w.b5 hεw.b5.e hεw.b5.d hεw.b5.p
  have d6 := mbStridedW_differentiable N 14 14 w.b6 hεw.b6.e hεw.b6.d hεw.b6.p
  have v6 := mbStridedWHasVJP N 14 14 w.b6 hεw.b6.e hεw.b6.d hεw.b6.p
  have d7 := mbResidW_differentiable N 14 14 w.b7 hεw.b7.e hεw.b7.d hεw.b7.p
  have v7 := mbResidWHasVJP N 14 14 w.b7 hεw.b7.e hεw.b7.d hεw.b7.p
  have d8 := mbResidW_differentiable N 14 14 w.b8 hεw.b8.e hεw.b8.d hεw.b8.p
  have v8 := mbResidWHasVJP N 14 14 w.b8 hεw.b8.e hεw.b8.d hεw.b8.p
  have d9 := mbExpW_differentiable N 14 14 w.b9 hεw.b9.e hεw.b9.d hεw.b9.p
  have v9 := mbExpWHasVJP N 14 14 w.b9 hεw.b9.e hεw.b9.d hεw.b9.p
  have d10 := mbResidW_differentiable N 14 14 w.b10 hεw.b10.e hεw.b10.d hεw.b10.p
  have v10 := mbResidWHasVJP N 14 14 w.b10 hεw.b10.e hεw.b10.d hεw.b10.p
  have d11 := mbResidW_differentiable N 14 14 w.b11 hεw.b11.e hεw.b11.d hεw.b11.p
  have v11 := mbResidWHasVJP N 14 14 w.b11 hεw.b11.e hεw.b11.d hεw.b11.p
  have d12 := mbStridedW_differentiable N 7 7 w.b12 hεw.b12.e hεw.b12.d hεw.b12.p
  have v12 := mbStridedWHasVJP N 7 7 w.b12 hεw.b12.e hεw.b12.d hεw.b12.p
  have d13 := mbResidW_differentiable N 7 7 w.b13 hεw.b13.e hεw.b13.d hεw.b13.p
  have v13 := mbResidWHasVJP N 7 7 w.b13 hεw.b13.e hεw.b13.d hεw.b13.p
  have d14 := mbResidW_differentiable N 7 7 w.b14 hεw.b14.e hεw.b14.d hεw.b14.p
  have v14 := mbResidWHasVJP N 7 7 w.b14 hεw.b14.e hεw.b14.d hεw.b14.p
  have d15 := mbResidW_differentiable N 7 7 w.b15 hεw.b15.e hεw.b15.d hεw.b15.p
  have v15 := mbResidWHasVJP N 7 7 w.b15 hεw.b15.e hεw.b15.d hεw.b15.p
  have d16 := mbExpW_differentiable N 7 7 w.b16 hεw.b16.e hεw.b16.d hεw.b16.p
  have v16 := mbExpWHasVJP N 7 7 w.b16 hεw.b16.e hεw.b16.d hεw.b16.p
  have dH := headFwdB_differentiable N (h := 7) (w := 7) w.hW w.hb w.hε hεw.h w.hγ w.hβ w.fcW w.fcb
  have vH := headFwdBHasVJP N (h := 7) (w := 7) w.hW w.hb w.hε hεw.h w.hγ w.hβ w.fcW w.fcb
  have e1 := vjpComp _ _ dS d1 vS v1;            have f1 := d1.comp dS
  have e2 := vjpComp _ _ f1 d2 e1 v2;            have f2 := d2.comp f1
  have e3 := vjpComp _ _ f2 d3 e2 v3;            have f3 := d3.comp f2
  have e4 := vjpComp _ _ f3 d4 e3 v4;            have f4 := d4.comp f3
  have e5 := vjpComp _ _ f4 d5 e4 v5;            have f5 := d5.comp f4
  have e6 := vjpComp _ _ f5 d6 e5 v6;            have f6 := d6.comp f5
  have e7 := vjpComp _ _ f6 d7 e6 v7;            have f7 := d7.comp f6
  have e8 := vjpComp _ _ f7 d8 e7 v8;            have f8 := d8.comp f7
  have e9 := vjpComp _ _ f8 d9 e8 v9;            have f9 := d9.comp f8
  have e10 := vjpComp _ _ f9 d10 e9 v10;         have f10 := d10.comp f9
  have e11 := vjpComp _ _ f10 d11 e10 v11;       have f11 := d11.comp f10
  have e12 := vjpComp _ _ f11 d12 e11 v12;       have f12 := d12.comp f11
  have e13 := vjpComp _ _ f12 d13 e12 v13;       have f13 := d13.comp f12
  have e14 := vjpComp _ _ f13 d14 e13 v14;       have f14 := d14.comp f13
  have e15 := vjpComp _ _ f14 d15 e14 v15;       have f15 := d15.comp f14
  have e16 := vjpComp _ _ f15 d16 e15 v16;       have f16 := d16.comp f15
  exact vjpComp _ _ f16 dH e16 vH

/-- **`efficientnetForwardBFull` = the `∘`-chain of the VJP's statement** — the
    kernel-checked bridge between the nested-application and composition forms,
    closing the form-gap this file shipped with. PROOF-SHAPE MATTERS (the ConvNeXt-T
    `convNextForwardTCh_eq_chain` lesson): equation-lemma `rw` + 17 `comp_apply`
    rewrites close syntactically; a `simp`/`rfl` proof of the same statement makes
    the kernel reduce the block bodies (no reducibility, no defeq cache) and
    deterministically time out. -/
theorem efficientnetForwardBFull_eq_chain (N : Nat) (w : B0Weights)
    (x : Vec (N * (3 * 224 * 224))) :
    efficientnetForwardBFull N w x =
      (headFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb ∘
        mbExpW N 7 7 w.b16 ∘ mbResidW N 7 7 w.b15 ∘ mbResidW N 7 7 w.b14 ∘ mbResidW N 7 7 w.b13 ∘
        mbStridedW N 7 7 w.b12 ∘ mbResidW N 14 14 w.b11 ∘ mbResidW N 14 14 w.b10 ∘
        mbExpW N 14 14 w.b9 ∘ mbResidW N 14 14 w.b8 ∘ mbResidW N 14 14 w.b7 ∘
        mbStridedW N 14 14 w.b6 ∘ mbResidW N 28 28 w.b5 ∘ mbStridedW N 28 28 w.b4 ∘
        mbResidW N 56 56 w.b3 ∘ mbStridedW N 56 56 w.b2 ∘ mbNoExpW N 112 112 w.b1 ∘
        stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) x := by
  rw [efficientnetForwardBFull]
  rw [Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply]

/-- **Public correctness theorem for `efficientnetForwardBFullHasVJP`** — the full
    B0's backward equals the `pdiv`-contracted Jacobian of `efficientnetForwardBFull`
    itself at every input, tying the chain-stated VJP back to the nested forward via
    `efficientnetForwardBFull_eq_chain`. -/
theorem efficientnetForwardBFullHasVJP_correct (N : Nat) (w : B0Weights)
    (hεw : w.EpsPos)
    (x : Vec (N * (3 * 224 * 224))) (dy : Vec (N * 10)) (i : Fin (N * (3 * 224 * 224))) :
    (efficientnetForwardBFullHasVJP N w hεw).backward x dy i =
      ∑ j : Fin (N * 10), pdiv (efficientnetForwardBFull N w) x i j * dy j := by
  have h := (efficientnetForwardBFullHasVJP N w hεw).correct x dy i
  rwa [show efficientnetForwardBFull N w =
        (headFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb ∘
          mbExpW N 7 7 w.b16 ∘ mbResidW N 7 7 w.b15 ∘ mbResidW N 7 7 w.b14 ∘ mbResidW N 7 7 w.b13 ∘
          mbStridedW N 7 7 w.b12 ∘ mbResidW N 14 14 w.b11 ∘ mbResidW N 14 14 w.b10 ∘
          mbExpW N 14 14 w.b9 ∘ mbResidW N 14 14 w.b8 ∘ mbResidW N 14 14 w.b7 ∘
          mbStridedW N 14 14 w.b6 ∘ mbResidW N 28 28 w.b5 ∘ mbStridedW N 28 28 w.b4 ∘
          mbResidW N 56 56 w.b3 ∘ mbStridedW N 56 56 w.b2 ∘ mbNoExpW N 112 112 w.b1 ∘
          stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ)
      from funext (efficientnetForwardBFull_eq_chain N w)]

end Proofs
