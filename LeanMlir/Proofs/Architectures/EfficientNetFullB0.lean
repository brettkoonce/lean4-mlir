import LeanMlir.Proofs.Architectures.EfficientNetChainClose

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

-- ════════════════════════════════════════════════════════════════
-- § The 4th block shape — MBConv6 expand + stride-1 + NO residual (`ic ≠ oc`, stages 5/7 first block)
-- ════════════════════════════════════════════════════════════════

/-- MBConv6 expand, stride-1, NO residual: `project-bn ∘ SE ∘ dw-bn-swish ∘ expand-bn-swish` (the
    `mbResidFwdB` body without the identity skip). -/
noncomputable def mbExpFwdB (N : Nat) {ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  projB N (h := h) (w := w) Wp bp εp γp βp ∘
    seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ ∘
    dwbsB N (h := h) (w := w) Wd bd εd γd βd ∘
    cbsB N (h := h) (w := w) We be εe γe βe

theorem mbExpFwdB_differentiable (N : Nat) {ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    Differentiable ℝ (mbExpFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbExpFwdB
  exact (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp).comp
    ((seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂).comp
      ((dwbsB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd).comp
        (cbsB_differentiable N (h := h) (w := w) We be εe hεe γe βe)))
noncomputable def mbExpFwdB_has_vjp (N : Nat) {ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    HasVJP (mbExpFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbExpFwdB
  have dE := cbsB_differentiable N (h := h) (w := w) We be εe hεe γe βe
  have dDw := dwbsB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd
  have dSe := seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂
  have vEdw : HasVJP _ := vjp_comp _ _ dE dDw (cbsB_has_vjp N (h := h) (w := w) We be εe hεe γe βe)
    (dwbsB_has_vjp N (h := h) (w := w) Wd bd εd hεd γd βd)
  exact vjp_comp _ _ (dSe.comp (dDw.comp dE)) (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp)
    (vjp_comp _ _ (dDw.comp dE) dSe vEdw (seB_has_vjp N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂))
    (projB_has_vjp N (h := h) (w := w) Wp bp εp hεp γp βp)

namespace StableHLO

/-- MBConv6 expand + stride-1 + no-residual graph (the `mbResidGraphB` body without the `addV` skip). -/
def mbExpGraphB (p epsStr : String) {N ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (e : SHlo (N * (ic * h * w))) : SHlo (N * (oc * h * w)) :=
  .bnBatchF s!"%{p}pg" s!"%{p}pbt" epsStr εp γp βp
    (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" s!"%{p}pb" Wp bp)
      (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zWa" s!"%{p}zba" s!"%{p}zWb" s!"%{p}zbb"
          Wz₁ bz₁ Wz₂ bz₂)
        (.swishF (.bnBatchF s!"%{p}dg" s!"%{p}dbt" epsStr εd γd βd
          (.batchOp (N := N) (.depthwise (h := h) (w := w) s!"%{p}dW" s!"%{p}db" Wd bd)
            (.swishF (.bnBatchF s!"%{p}eg" s!"%{p}ebt" epsStr εe γe βe
              (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}eW" s!"%{p}eb" We be) e))))))))

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
  simp only [den_batchOp_conv, den_batchOp_seBlock, den_batchOp_depthwise, den_bnBatchF,
             swishF_faithful, Function.comp_apply]

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
noncomputable def mbNoExpW_has_vjp (N h w : Nat) {ic oc kh kw r : Nat} (p : MBWNoExp ic oc r kh kw)
    (hd : 0 < p.dε) (hp : 0 < p.pε) : HasVJP (mbNoExpW N h w p) := by
  unfold mbNoExpW
  exact mbNoExpFwdB_has_vjp N (h := h) (w := w) p.dW p.db p.dε hd p.dγ p.dβ
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ

theorem mbStridedW_differentiable (N h w : Nat) {ic mid oc kh kw r : Nat} (p : MBW ic mid oc r kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) : Differentiable ℝ (mbStridedW N h w p) := by
  unfold mbStridedW
  exact mbStridedFwdB_differentiable N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ
    p.dW p.db p.dε hd p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ
noncomputable def mbStridedW_has_vjp (N h w : Nat) {ic mid oc kh kw r : Nat} (p : MBW ic mid oc r kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) : HasVJP (mbStridedW N h w p) := by
  unfold mbStridedW
  exact mbStridedFwdB_has_vjp N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ
    p.dW p.db p.dε hd p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ

theorem mbResidW_differentiable (N h w : Nat) {c mid kh kw r : Nat} (p : MBW c mid c r kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) : Differentiable ℝ (mbResidW N h w p) := by
  unfold mbResidW
  exact mbResidFwdB_differentiable N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ
    p.dW p.db p.dε hd p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ
noncomputable def mbResidW_has_vjp (N h w : Nat) {c mid kh kw r : Nat} (p : MBW c mid c r kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) : HasVJP (mbResidW N h w p) := by
  unfold mbResidW
  exact mbResidFwdB_has_vjp N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ
    p.dW p.db p.dε hd p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ

theorem mbExpW_differentiable (N h w : Nat) {ic mid oc kh kw r : Nat} (p : MBW ic mid oc r kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) : Differentiable ℝ (mbExpW N h w p) := by
  unfold mbExpW
  exact mbExpFwdB_differentiable N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ
    p.dW p.db p.dε hd p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ
noncomputable def mbExpW_has_vjp (N h w : Nat) {ic mid oc kh kw r : Nat} (p : MBW ic mid oc r kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) : HasVJP (mbExpW N h w p) := by
  unfold mbExpW
  exact mbExpFwdB_has_vjp N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ
    p.dW p.db p.dε hd p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ

-- ════════════════════════════════════════════════════════════════
-- § The full B0 ℝ-forward — all 16 MBConv blocks, nested-application form
--   stem(224→112) → b1@112 → b2(112→56) → b3@56 → b4(56→28) → b5@28 → b6(28→14) → b7,b8,b9,b10,b11@14
--   → b12(14→7) → b13,b14,b15,b16@7 → head@7 → GAP → dense
-- ════════════════════════════════════════════════════════════════

noncomputable def efficientnetForwardB_full (N : Nat) (w : B0Weights)
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
def efficientnetFwdGraphB_full (N : Nat) (epsStr : String) (w : B0Weights)
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
    batch-norm + SE) denotes `efficientnetForwardB_full`. Chained from the per-block `*GraphW_faithful`
    lemmas (one `rw` per block, outermost→innermost), then a structural `rfl` (the forward is
    nested-application form, blocks opaque) — the `ResNet34RenderPC` recipe at full depth. -/
theorem efficientnetFwdGraphB_full_faithful (N : Nat) (epsStr : String) (w : B0Weights)
    (x : Vec (N * (3 * 224 * 224))) :
    den (efficientnetFwdGraphB_full N epsStr w x) = efficientnetForwardB_full N w x := by
  rw [efficientnetFwdGraphB_full, headGraphB_faithful,
      mbExpGraphW_faithful, mbResidGraphW_faithful, mbResidGraphW_faithful, mbResidGraphW_faithful,
      mbStridedGraphW_faithful, mbResidGraphW_faithful, mbResidGraphW_faithful, mbExpGraphW_faithful,
      mbResidGraphW_faithful, mbResidGraphW_faithful, mbStridedGraphW_faithful, mbResidGraphW_faithful,
      mbStridedGraphW_faithful, mbResidGraphW_faithful, mbStridedGraphW_faithful, mbNoExpGraphW_faithful,
      stemGraphB_faithful, den_operand]
  rfl

end StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The full B0 VJP — all 16 MBConv blocks (the full-depth analogue of `efficientnet_has_vjp`)
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 20000 in
/-- **The full EfficientNet-B0 has a (correct) VJP.** Chained from the per-block gradients (stem → 16
    MBConv blocks → head) via `vjp_comp`. Stated on the `∘`-composition of the blocks (= the full
    forward by construction; keeps the blocks opaque so the chain closes structurally). The full-depth,
    batched, true-batch-norm + SE analogue of `efficientnet_has_vjp`. -/
noncomputable def efficientnetForwardB_full_has_vjp (N : Nat) (w : B0Weights)
    (hsε : 0 < w.sε)
    (hb1d : 0 < w.b1.dε) (hb1p : 0 < w.b1.pε)
    (hb2e : 0 < w.b2.eε) (hb2d : 0 < w.b2.dε) (hb2p : 0 < w.b2.pε)
    (hb3e : 0 < w.b3.eε) (hb3d : 0 < w.b3.dε) (hb3p : 0 < w.b3.pε)
    (hb4e : 0 < w.b4.eε) (hb4d : 0 < w.b4.dε) (hb4p : 0 < w.b4.pε)
    (hb5e : 0 < w.b5.eε) (hb5d : 0 < w.b5.dε) (hb5p : 0 < w.b5.pε)
    (hb6e : 0 < w.b6.eε) (hb6d : 0 < w.b6.dε) (hb6p : 0 < w.b6.pε)
    (hb7e : 0 < w.b7.eε) (hb7d : 0 < w.b7.dε) (hb7p : 0 < w.b7.pε)
    (hb8e : 0 < w.b8.eε) (hb8d : 0 < w.b8.dε) (hb8p : 0 < w.b8.pε)
    (hb9e : 0 < w.b9.eε) (hb9d : 0 < w.b9.dε) (hb9p : 0 < w.b9.pε)
    (hb10e : 0 < w.b10.eε) (hb10d : 0 < w.b10.dε) (hb10p : 0 < w.b10.pε)
    (hb11e : 0 < w.b11.eε) (hb11d : 0 < w.b11.dε) (hb11p : 0 < w.b11.pε)
    (hb12e : 0 < w.b12.eε) (hb12d : 0 < w.b12.dε) (hb12p : 0 < w.b12.pε)
    (hb13e : 0 < w.b13.eε) (hb13d : 0 < w.b13.dε) (hb13p : 0 < w.b13.pε)
    (hb14e : 0 < w.b14.eε) (hb14d : 0 < w.b14.dε) (hb14p : 0 < w.b14.pε)
    (hb15e : 0 < w.b15.eε) (hb15d : 0 < w.b15.dε) (hb15p : 0 < w.b15.pε)
    (hb16e : 0 < w.b16.eε) (hb16d : 0 < w.b16.dε) (hb16p : 0 < w.b16.pε)
    (hhε : 0 < w.hε) :
    HasVJP
      (headFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb ∘
        mbExpW N 7 7 w.b16 ∘ mbResidW N 7 7 w.b15 ∘ mbResidW N 7 7 w.b14 ∘ mbResidW N 7 7 w.b13 ∘
        mbStridedW N 7 7 w.b12 ∘ mbResidW N 14 14 w.b11 ∘ mbResidW N 14 14 w.b10 ∘
        mbExpW N 14 14 w.b9 ∘ mbResidW N 14 14 w.b8 ∘ mbResidW N 14 14 w.b7 ∘
        mbStridedW N 14 14 w.b6 ∘ mbResidW N 28 28 w.b5 ∘ mbStridedW N 28 28 w.b4 ∘
        mbResidW N 56 56 w.b3 ∘ mbStridedW N 56 56 w.b2 ∘ mbNoExpW N 112 112 w.b1 ∘
        stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) := by
  have dS := stemB_differentiable N (h := 112) (w := 112) w.sW w.sb w.sε hsε w.sγ w.sβ
  have vS := stemB_has_vjp N (h := 112) (w := 112) w.sW w.sb w.sε hsε w.sγ w.sβ
  have d1 := mbNoExpW_differentiable N 112 112 w.b1 hb1d hb1p
  have v1 := mbNoExpW_has_vjp N 112 112 w.b1 hb1d hb1p
  have d2 := mbStridedW_differentiable N 56 56 w.b2 hb2e hb2d hb2p
  have v2 := mbStridedW_has_vjp N 56 56 w.b2 hb2e hb2d hb2p
  have d3 := mbResidW_differentiable N 56 56 w.b3 hb3e hb3d hb3p
  have v3 := mbResidW_has_vjp N 56 56 w.b3 hb3e hb3d hb3p
  have d4 := mbStridedW_differentiable N 28 28 w.b4 hb4e hb4d hb4p
  have v4 := mbStridedW_has_vjp N 28 28 w.b4 hb4e hb4d hb4p
  have d5 := mbResidW_differentiable N 28 28 w.b5 hb5e hb5d hb5p
  have v5 := mbResidW_has_vjp N 28 28 w.b5 hb5e hb5d hb5p
  have d6 := mbStridedW_differentiable N 14 14 w.b6 hb6e hb6d hb6p
  have v6 := mbStridedW_has_vjp N 14 14 w.b6 hb6e hb6d hb6p
  have d7 := mbResidW_differentiable N 14 14 w.b7 hb7e hb7d hb7p
  have v7 := mbResidW_has_vjp N 14 14 w.b7 hb7e hb7d hb7p
  have d8 := mbResidW_differentiable N 14 14 w.b8 hb8e hb8d hb8p
  have v8 := mbResidW_has_vjp N 14 14 w.b8 hb8e hb8d hb8p
  have d9 := mbExpW_differentiable N 14 14 w.b9 hb9e hb9d hb9p
  have v9 := mbExpW_has_vjp N 14 14 w.b9 hb9e hb9d hb9p
  have d10 := mbResidW_differentiable N 14 14 w.b10 hb10e hb10d hb10p
  have v10 := mbResidW_has_vjp N 14 14 w.b10 hb10e hb10d hb10p
  have d11 := mbResidW_differentiable N 14 14 w.b11 hb11e hb11d hb11p
  have v11 := mbResidW_has_vjp N 14 14 w.b11 hb11e hb11d hb11p
  have d12 := mbStridedW_differentiable N 7 7 w.b12 hb12e hb12d hb12p
  have v12 := mbStridedW_has_vjp N 7 7 w.b12 hb12e hb12d hb12p
  have d13 := mbResidW_differentiable N 7 7 w.b13 hb13e hb13d hb13p
  have v13 := mbResidW_has_vjp N 7 7 w.b13 hb13e hb13d hb13p
  have d14 := mbResidW_differentiable N 7 7 w.b14 hb14e hb14d hb14p
  have v14 := mbResidW_has_vjp N 7 7 w.b14 hb14e hb14d hb14p
  have d15 := mbResidW_differentiable N 7 7 w.b15 hb15e hb15d hb15p
  have v15 := mbResidW_has_vjp N 7 7 w.b15 hb15e hb15d hb15p
  have d16 := mbExpW_differentiable N 7 7 w.b16 hb16e hb16d hb16p
  have v16 := mbExpW_has_vjp N 7 7 w.b16 hb16e hb16d hb16p
  have dH := headFwdB_differentiable N (h := 7) (w := 7) w.hW w.hb w.hε hhε w.hγ w.hβ w.fcW w.fcb
  have vH := headFwdB_has_vjp N (h := 7) (w := 7) w.hW w.hb w.hε hhε w.hγ w.hβ w.fcW w.fcb
  have e1 := vjp_comp _ _ dS d1 vS v1;            have f1 := d1.comp dS
  have e2 := vjp_comp _ _ f1 d2 e1 v2;            have f2 := d2.comp f1
  have e3 := vjp_comp _ _ f2 d3 e2 v3;            have f3 := d3.comp f2
  have e4 := vjp_comp _ _ f3 d4 e3 v4;            have f4 := d4.comp f3
  have e5 := vjp_comp _ _ f4 d5 e4 v5;            have f5 := d5.comp f4
  have e6 := vjp_comp _ _ f5 d6 e5 v6;            have f6 := d6.comp f5
  have e7 := vjp_comp _ _ f6 d7 e6 v7;            have f7 := d7.comp f6
  have e8 := vjp_comp _ _ f7 d8 e7 v8;            have f8 := d8.comp f7
  have e9 := vjp_comp _ _ f8 d9 e8 v9;            have f9 := d9.comp f8
  have e10 := vjp_comp _ _ f9 d10 e9 v10;         have f10 := d10.comp f9
  have e11 := vjp_comp _ _ f10 d11 e10 v11;       have f11 := d11.comp f10
  have e12 := vjp_comp _ _ f11 d12 e11 v12;       have f12 := d12.comp f11
  have e13 := vjp_comp _ _ f12 d13 e12 v13;       have f13 := d13.comp f12
  have e14 := vjp_comp _ _ f13 d14 e13 v14;       have f14 := d14.comp f13
  have e15 := vjp_comp _ _ f14 d15 e14 v15;       have f15 := d15.comp f14
  have e16 := vjp_comp _ _ f15 d16 e15 v16;       have f16 := d16.comp f15
  exact vjp_comp _ _ f16 dH e16 vH

/-- **`efficientnetForwardB_full` = the `∘`-chain of the VJP's statement** — the
    kernel-checked bridge between the nested-application and composition forms,
    closing the form-gap this file shipped with. PROOF-SHAPE MATTERS (the ConvNeXt-T
    `convNextForwardT_eq_chain` lesson): equation-lemma `rw` + 17 `comp_apply`
    rewrites close syntactically; a `simp`/`rfl` proof of the same statement makes
    the kernel reduce the block bodies (no reducibility, no defeq cache) and
    deterministically time out. -/
theorem efficientnetForwardB_full_eq_chain (N : Nat) (w : B0Weights)
    (x : Vec (N * (3 * 224 * 224))) :
    efficientnetForwardB_full N w x =
      (headFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb ∘
        mbExpW N 7 7 w.b16 ∘ mbResidW N 7 7 w.b15 ∘ mbResidW N 7 7 w.b14 ∘ mbResidW N 7 7 w.b13 ∘
        mbStridedW N 7 7 w.b12 ∘ mbResidW N 14 14 w.b11 ∘ mbResidW N 14 14 w.b10 ∘
        mbExpW N 14 14 w.b9 ∘ mbResidW N 14 14 w.b8 ∘ mbResidW N 14 14 w.b7 ∘
        mbStridedW N 14 14 w.b6 ∘ mbResidW N 28 28 w.b5 ∘ mbStridedW N 28 28 w.b4 ∘
        mbResidW N 56 56 w.b3 ∘ mbStridedW N 56 56 w.b2 ∘ mbNoExpW N 112 112 w.b1 ∘
        stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) x := by
  rw [efficientnetForwardB_full]
  rw [Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply]

/-- **Public correctness theorem for `efficientnetForwardB_full_has_vjp`** — the full
    B0's backward equals the `pdiv`-contracted Jacobian of `efficientnetForwardB_full`
    itself at every input, tying the chain-stated VJP back to the nested forward via
    `efficientnetForwardB_full_eq_chain`. -/
theorem efficientnetForwardB_full_has_vjp_correct (N : Nat) (w : B0Weights)
    (hsε : 0 < w.sε)
    (hb1d : 0 < w.b1.dε) (hb1p : 0 < w.b1.pε)
    (hb2e : 0 < w.b2.eε) (hb2d : 0 < w.b2.dε) (hb2p : 0 < w.b2.pε)
    (hb3e : 0 < w.b3.eε) (hb3d : 0 < w.b3.dε) (hb3p : 0 < w.b3.pε)
    (hb4e : 0 < w.b4.eε) (hb4d : 0 < w.b4.dε) (hb4p : 0 < w.b4.pε)
    (hb5e : 0 < w.b5.eε) (hb5d : 0 < w.b5.dε) (hb5p : 0 < w.b5.pε)
    (hb6e : 0 < w.b6.eε) (hb6d : 0 < w.b6.dε) (hb6p : 0 < w.b6.pε)
    (hb7e : 0 < w.b7.eε) (hb7d : 0 < w.b7.dε) (hb7p : 0 < w.b7.pε)
    (hb8e : 0 < w.b8.eε) (hb8d : 0 < w.b8.dε) (hb8p : 0 < w.b8.pε)
    (hb9e : 0 < w.b9.eε) (hb9d : 0 < w.b9.dε) (hb9p : 0 < w.b9.pε)
    (hb10e : 0 < w.b10.eε) (hb10d : 0 < w.b10.dε) (hb10p : 0 < w.b10.pε)
    (hb11e : 0 < w.b11.eε) (hb11d : 0 < w.b11.dε) (hb11p : 0 < w.b11.pε)
    (hb12e : 0 < w.b12.eε) (hb12d : 0 < w.b12.dε) (hb12p : 0 < w.b12.pε)
    (hb13e : 0 < w.b13.eε) (hb13d : 0 < w.b13.dε) (hb13p : 0 < w.b13.pε)
    (hb14e : 0 < w.b14.eε) (hb14d : 0 < w.b14.dε) (hb14p : 0 < w.b14.pε)
    (hb15e : 0 < w.b15.eε) (hb15d : 0 < w.b15.dε) (hb15p : 0 < w.b15.pε)
    (hb16e : 0 < w.b16.eε) (hb16d : 0 < w.b16.dε) (hb16p : 0 < w.b16.pε)
    (hhε : 0 < w.hε)
    (x : Vec (N * (3 * 224 * 224))) (dy : Vec (N * 10)) (i : Fin (N * (3 * 224 * 224))) :
    (efficientnetForwardB_full_has_vjp N w hsε hb1d hb1p hb2e hb2d hb2p hb3e hb3d hb3p
        hb4e hb4d hb4p hb5e hb5d hb5p hb6e hb6d hb6p hb7e hb7d hb7p hb8e hb8d hb8p
        hb9e hb9d hb9p hb10e hb10d hb10p hb11e hb11d hb11p hb12e hb12d hb12p
        hb13e hb13d hb13p hb14e hb14d hb14p hb15e hb15d hb15p hb16e hb16d hb16p
        hhε).backward x dy i =
      ∑ j : Fin (N * 10), pdiv (efficientnetForwardB_full N w) x i j * dy j := by
  have h := (efficientnetForwardB_full_has_vjp N w hsε hb1d hb1p hb2e hb2d hb2p hb3e hb3d hb3p
        hb4e hb4d hb4p hb5e hb5d hb5p hb6e hb6d hb6p hb7e hb7d hb7p hb8e hb8d hb8p
        hb9e hb9d hb9p hb10e hb10d hb10p hb11e hb11d hb11p hb12e hb12d hb12p
        hb13e hb13d hb13p hb14e hb14d hb14p hb15e hb15d hb15p hb16e hb16d hb16p
        hhε).correct x dy i
  rwa [show efficientnetForwardB_full N w =
        (headFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb ∘
          mbExpW N 7 7 w.b16 ∘ mbResidW N 7 7 w.b15 ∘ mbResidW N 7 7 w.b14 ∘ mbResidW N 7 7 w.b13 ∘
          mbStridedW N 7 7 w.b12 ∘ mbResidW N 14 14 w.b11 ∘ mbResidW N 14 14 w.b10 ∘
          mbExpW N 14 14 w.b9 ∘ mbResidW N 14 14 w.b8 ∘ mbResidW N 14 14 w.b7 ∘
          mbStridedW N 14 14 w.b6 ∘ mbResidW N 28 28 w.b5 ∘ mbStridedW N 28 28 w.b4 ∘
          mbResidW N 56 56 w.b3 ∘ mbStridedW N 56 56 w.b2 ∘ mbNoExpW N 112 112 w.b1 ∘
          stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ)
      from funext (efficientnetForwardB_full_eq_chain N w)]

end Proofs
