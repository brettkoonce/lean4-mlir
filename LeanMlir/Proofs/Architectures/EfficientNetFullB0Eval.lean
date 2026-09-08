import LeanMlir.Proofs.Codegen.EfficientNetRenderPCEval

/-! # The FULL EfficientNet-B0 at INFERENCE — all 16 MBConv blocks, eval forward + graph + faithfulness

The eval twin of `EfficientNetFullB0.lean`. That file states the sixteen-block `[t,c,n,s,k]` net at
TRAINING BatchNorm (`bnBatchLA`), the world its VJP and its typed graph live in; this file states
the same ladder at INFERENCE BatchNorm — frozen running statistics at all **49** sites, one shared
`ε`, as `efficientnetForwardBEval` and the shipped `efficientnet_fwd_eval` both do — and proves
its typed `SHlo` graph denotes it — T2 at inference BatchNorm for the paper net, the graph of
`efficientnet_fwd_eval.mlir` and its 1000-class twin. (Built 2026-09-05 so the whole-net float
budget could end at a graph; the budget was deleted 2026-09-08 and the graph statement stays —
`planning/proofs_tier_to_paper_nets.md` 3.3(e).)

Pure enumeration and chaining of `EfficientNetRenderPCEval.lean`'s per-block machinery, at the
batched index `N·(c·h·w)` and generic in the class count. The one genuinely new piece is the
fourth block shape at inference — `mbExpFwdBEval` / `mbExpGraphBEval`: expand, stride 1, **no**
residual (`ic ≠ oc`; the stage-5 and stage-7 first blocks `b9`/`b16`) — which the three-block
representative has no instance of and `EfficientNetFullB0.lean` added at training BN as `mbExpFwdB`.

**What it is tied to.** `efficientnet_fwd_eval.mlir` is THIS net: `%x` plus 213 parameters (the
render folds each conv bias into the BatchNorm that follows it, so `%sb`/`%b1db`/… have no slot)
plus 98 statistic slots — 312 inputs at ten classes, and `efficientnetin_fwd_eval.mlir` its
1000-class twin. ⚠ The typed graph below inherits the three-block eval graph's SSA names, and
they differ from the artifact's in four ways, none of which enters `den` (names are
pretty-printing metadata): the graph carries a bias slot per conv (`"%sb"`, `s!"%{p}db"`, …) that
the render folds away; it names the statistic slots `%smu`/`%svar`, `%b{k}{e,d,p}mu`/`var`,
`%hmu`/`%hvar` where the artifact has `%stnmu`/`%stnvar`, `%b{k}{e,d,p}nmu`/`nvar`,
`%hnmu`/`%hnvar`; it names the SE denses `zWa/zba/zWb/zbb` where the artifact has
`zW1/zb1/zW2/zb2`; and its classifier is `%Wfc`/`%bfc` where the artifact's is `%Wd`/`%bd`. The
`den`-level statement is what the number needs; matching the text is a separate, cosmetic pass
over `EfficientNetRenderPCEval.lean`.

B0 stage spec `[t,c,n,s,k]`: s1 (1,16,1,1,3) s2 (6,24,2,2,3) s3 (6,40,2,2,5) s4 (6,80,3,2,3)
s5 (6,112,3,1,5) s6 (6,192,4,2,5) s7 (6,320,1,1,3); stem 3×3/s2 (3→32) at the XLA-`SAME` phase,
head 1×1 (320→1280) → GAP → dense.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The fourth block shape at inference: expand + stride-1 + NO residual
-- ════════════════════════════════════════════════════════════════

/-- MBConv6 expand, stride 1, no residual, at inference — `mbResidFwdBEval`'s body without the
    skip (the eval twin of `mbExpFwdB`). -/
noncomputable def mbExpFwdBEval (N : Nat) {ic mid oc h w kHd kWd r : Nat} (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (γe βe μe ve : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (γd βd μd vd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  projBEval N (h := h) (w := w) Wp bp ε γp βp μp vp ∘
    seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ ∘
    dwbsBEval N (h := h) (w := w) Wd bd ε γd βd μd vd ∘
    cbsBEval N (h := h) (w := w) We be ε γe βe μe ve

namespace StableHLO

/-- MBConv6 expand + stride-1 + no-residual inference graph — `mbResidGraphBEval` without the
    `addV` skip. -/
def mbExpGraphBEval (p epsStr : String) {N ic mid oc h w kHd kWd r : Nat} (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (γe βe μe ve : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (γd βd μd vd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc)
    (e : SHlo (N * (ic * h * w))) : SHlo (N * (oc * h * w)) :=
  .batchOp (N := N) (.bnEval (h := h) (w := w) s!"%{p}pg" s!"%{p}pbt" s!"%{p}pmu" s!"%{p}pvar"
      epsStr ε γp βp μp vp)
    (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" s!"%{p}pb" Wp bp)
      (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zWa" s!"%{p}zba" s!"%{p}zWb" s!"%{p}zbb"
          Wz₁ bz₁ Wz₂ bz₂)
        (.swishF (.batchOp (N := N) (.bnEval (h := h) (w := w) s!"%{p}dg" s!"%{p}dbt"
            s!"%{p}dmu" s!"%{p}dvar" epsStr ε γd βd μd vd)
          (.batchOp (N := N) (.depthwise (h := h) (w := w) s!"%{p}dW" s!"%{p}db" Wd bd)
            (.swishF (.batchOp (N := N) (.bnEval (h := h) (w := w) s!"%{p}eg" s!"%{p}ebt"
                s!"%{p}emu" s!"%{p}evar" epsStr ε γe βe μe ve)
              (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}eW" s!"%{p}eb" We be) e))))))))

theorem mbExpGraphBEval_faithful (p epsStr : String) {N ic mid oc h w kHd kWd r : Nat} (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (γe βe μe ve : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (γd βd μd vd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc)
    (e : SHlo (N * (ic * h * w))) :
    den (mbExpGraphBEval p epsStr ε We be γe βe μe ve Wd bd γd βd μd vd Wz₁ bz₁ Wz₂ bz₂
          Wp bp γp βp μp vp e)
      = mbExpFwdBEval N (h := h) (w := w) ε We be γe βe μe ve Wd bd γd βd μd vd
          Wz₁ bz₁ Wz₂ bz₂ Wp bp γp βp μp vp (den e) := by
  unfold mbExpGraphBEval mbExpFwdBEval projBEval seB dwbsBEval cbsBEval
  simp only [den_batchOp_conv, den_batchOp_seBlock, den_batchOp_depthwise, den_batchOp_bnEval,
             swishF_faithful, Function.comp_apply]

end StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Per-block weight bundles at inference (γ, β and the two frozen statistics per BN site)
-- ════════════════════════════════════════════════════════════════

/-- Weights and running statistics of one MBConv6 block at inference. -/
structure MBWEval (ic mid oc r kh kw : Nat) where
  eW : Kernel4 mid ic 1 1
  eb : Vec mid
  eγ : Vec mid
  eβ : Vec mid
  eμ : Vec mid
  ev : Vec mid
  dW : DepthwiseKernel mid kh kw
  db : Vec mid
  dγ : Vec mid
  dβ : Vec mid
  dμ : Vec mid
  dv : Vec mid
  z1 : Mat mid r
  zb1 : Vec r
  z2 : Mat r mid
  zb2 : Vec mid
  pW : Kernel4 oc mid 1 1
  pb : Vec oc
  pγ : Vec oc
  pβ : Vec oc
  pμ : Vec oc
  pv : Vec oc

/-- Weights and running statistics of the MBConv1 block (`t = 1`, no expand) at inference. -/
structure MBWNoExpEval (ic oc r kh kw : Nat) where
  dW : DepthwiseKernel ic kh kw
  db : Vec ic
  dγ : Vec ic
  dβ : Vec ic
  dμ : Vec ic
  dv : Vec ic
  z1 : Mat ic r
  zb1 : Vec r
  z2 : Mat r ic
  zb2 : Vec ic
  pW : Kernel4 oc ic 1 1
  pb : Vec oc
  pγ : Vec oc
  pβ : Vec oc
  pμ : Vec oc
  pv : Vec oc

/-- All of EfficientNet-B0's parameters and running statistics at inference: stem (3×3/s2 3→32)
    + 16 MBConv blocks (the real `[t,c,n,s,k]` spec, `B0Weights`'s widths) + head (1×1 320→1280)
    + dense (1280→`nCls`). 49 BatchNorm sites, each with `μ`/`v`. -/
structure B0WeightsEval (nCls : Nat) where
  sW : Kernel4 32 3 3 3
  sb : Vec 32
  sγ : Vec 32
  sβ : Vec 32
  sμ : Vec 32
  sv : Vec 32
  b1 : MBWNoExpEval 32 16 8 3 3
  b2 : MBWEval 16 96 24 4 3 3
  b3 : MBWEval 24 144 24 6 3 3
  b4 : MBWEval 24 144 40 6 5 5
  b5 : MBWEval 40 240 40 10 5 5
  b6 : MBWEval 40 240 80 10 3 3
  b7 : MBWEval 80 480 80 20 3 3
  b8 : MBWEval 80 480 80 20 3 3
  b9 : MBWEval 80 480 112 20 5 5
  b10 : MBWEval 112 672 112 28 5 5
  b11 : MBWEval 112 672 112 28 5 5
  b12 : MBWEval 112 672 192 28 5 5
  b13 : MBWEval 192 1152 192 48 5 5
  b14 : MBWEval 192 1152 192 48 5 5
  b15 : MBWEval 192 1152 192 48 5 5
  b16 : MBWEval 192 1152 320 48 3 3
  hW : Kernel4 1280 320 1 1
  hb : Vec 1280
  hγ : Vec 1280
  hβ : Vec 1280
  hμ : Vec 1280
  hv : Vec 1280
  fcW : Mat 1280 nCls
  fcb : Vec nCls

variable {nCls : Nat}

-- ════════════════════════════════════════════════════════════════
-- § Weight-bundle wrappers (forward ℝ-fns) — `(N h w)` explicit, block dims from the bundle
-- ════════════════════════════════════════════════════════════════

noncomputable def mbNoExpEvalW (N h w : Nat) (ε : ℝ) {ic oc kh kw r : Nat}
    (p : MBWNoExpEval ic oc r kh kw) : Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  mbNoExpFwdBEval N (h := h) (w := w) ε p.dW p.db p.dγ p.dβ p.dμ p.dv p.z1 p.zb1 p.z2 p.zb2
    p.pW p.pb p.pγ p.pβ p.pμ p.pv

noncomputable def mbStridedEvalW (N h w : Nat) (ε : ℝ) {ic mid oc kh kw r : Nat}
    (p : MBWEval ic mid oc r kh kw) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  mbStridedFwdBEval N (h := h) (w := w) ε p.eW p.eb p.eγ p.eβ p.eμ p.ev p.dW p.db p.dγ p.dβ p.dμ p.dv
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ p.pμ p.pv

noncomputable def mbResidEvalW (N h w : Nat) (ε : ℝ) {c mid kh kw r : Nat}
    (p : MBWEval c mid c r kh kw) : Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  mbResidFwdBEval N (h := h) (w := w) ε p.eW p.eb p.eγ p.eβ p.eμ p.ev p.dW p.db p.dγ p.dβ p.dμ p.dv
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ p.pμ p.pv

noncomputable def mbExpEvalW (N h w : Nat) (ε : ℝ) {ic mid oc kh kw r : Nat}
    (p : MBWEval ic mid oc r kh kw) : Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  mbExpFwdBEval N (h := h) (w := w) ε p.eW p.eb p.eγ p.eβ p.eμ p.ev p.dW p.db p.dγ p.dβ p.dμ p.dv
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ p.pμ p.pv

-- ════════════════════════════════════════════════════════════════
-- § The full B0 inference ℝ-forward — all 16 MBConv blocks, nested-application form
-- ════════════════════════════════════════════════════════════════

/-- **The sixteen-block EfficientNet-B0 inference forward** — `efficientnetForwardB_full`'s ladder
    with frozen running statistics at all 49 BatchNorm sites, at one shared `ε`. Nested-application
    form (NOT `∘`), as the training twin, so the faithfulness proof closes by `rw` and `rfl`. -/
noncomputable def efficientnetForwardB_fullEval (N : Nat) (ε : ℝ) (w : B0WeightsEval nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * nCls) :=
  headFwdBEval N (h := 7) (w := 7) ε w.hW w.hb w.hγ w.hβ w.hμ w.hv w.fcW w.fcb
    (mbExpEvalW N 7 7 ε w.b16
      (mbResidEvalW N 7 7 ε w.b15
        (mbResidEvalW N 7 7 ε w.b14
          (mbResidEvalW N 7 7 ε w.b13
            (mbStridedEvalW N 7 7 ε w.b12
              (mbResidEvalW N 14 14 ε w.b11
                (mbResidEvalW N 14 14 ε w.b10
                  (mbExpEvalW N 14 14 ε w.b9
                    (mbResidEvalW N 14 14 ε w.b8
                      (mbResidEvalW N 14 14 ε w.b7
                        (mbStridedEvalW N 14 14 ε w.b6
                          (mbResidEvalW N 28 28 ε w.b5
                            (mbStridedEvalW N 28 28 ε w.b4
                              (mbResidEvalW N 56 56 ε w.b3
                                (mbStridedEvalW N 56 56 ε w.b2
                                  (mbNoExpEvalW N 112 112 ε w.b1
                                    (stemBEval N (h := 112) (w := 112) w.sW w.sb ε w.sγ w.sβ w.sμ w.sv x)))))))))))))))))

namespace StableHLO

-- § Weight-bundle wrappers (graph) + faithfulness (each = the per-block lemma at the bundle's fields)

def mbNoExpGraphEvalW (pfx epsStr : String) (N h w : Nat) (ε : ℝ) {ic oc kh kw r : Nat}
    (p : MBWNoExpEval ic oc r kh kw) (e : SHlo (N * (ic * h * w))) : SHlo (N * (oc * h * w)) :=
  mbNoExpGraphBEval pfx epsStr (h := h) (w := w) ε p.dW p.db p.dγ p.dβ p.dμ p.dv p.z1 p.zb1 p.z2 p.zb2
    p.pW p.pb p.pγ p.pβ p.pμ p.pv e
theorem mbNoExpGraphEvalW_faithful (pfx epsStr : String) (N h w : Nat) (ε : ℝ) {ic oc kh kw r : Nat}
    (p : MBWNoExpEval ic oc r kh kw) (e : SHlo (N * (ic * h * w))) :
    den (mbNoExpGraphEvalW pfx epsStr N h w ε p e) = mbNoExpEvalW N h w ε p (den e) := by
  unfold mbNoExpGraphEvalW mbNoExpEvalW
  exact mbNoExpGraphBEval_faithful pfx epsStr ε p.dW p.db p.dγ p.dβ p.dμ p.dv p.z1 p.zb1 p.z2 p.zb2
    p.pW p.pb p.pγ p.pβ p.pμ p.pv e

def mbStridedGraphEvalW (pfx epsStr : String) (N h w : Nat) (ε : ℝ) {ic mid oc kh kw r : Nat}
    (p : MBWEval ic mid oc r kh kw) (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    SHlo (N * (oc * h * w)) :=
  mbStridedGraphBEval pfx epsStr (h := h) (w := w) ε p.eW p.eb p.eγ p.eβ p.eμ p.ev
    p.dW p.db p.dγ p.dβ p.dμ p.dv p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ p.pμ p.pv e
theorem mbStridedGraphEvalW_faithful (pfx epsStr : String) (N h w : Nat) (ε : ℝ)
    {ic mid oc kh kw r : Nat} (p : MBWEval ic mid oc r kh kw)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (mbStridedGraphEvalW pfx epsStr N h w ε p e) = mbStridedEvalW N h w ε p (den e) := by
  unfold mbStridedGraphEvalW mbStridedEvalW
  exact mbStridedGraphBEval_faithful pfx epsStr ε p.eW p.eb p.eγ p.eβ p.eμ p.ev
    p.dW p.db p.dγ p.dβ p.dμ p.dv p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ p.pμ p.pv e

def mbResidGraphEvalW (pfx epsStr : String) (N h w : Nat) (ε : ℝ) {c mid kh kw r : Nat}
    (p : MBWEval c mid c r kh kw) (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  mbResidGraphBEval pfx epsStr (h := h) (w := w) ε p.eW p.eb p.eγ p.eβ p.eμ p.ev
    p.dW p.db p.dγ p.dβ p.dμ p.dv p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ p.pμ p.pv e
theorem mbResidGraphEvalW_faithful (pfx epsStr : String) (N h w : Nat) (ε : ℝ) {c mid kh kw r : Nat}
    (p : MBWEval c mid c r kh kw) (e : SHlo (N * (c * h * w))) :
    den (mbResidGraphEvalW pfx epsStr N h w ε p e) = mbResidEvalW N h w ε p (den e) := by
  unfold mbResidGraphEvalW mbResidEvalW
  exact mbResidGraphBEval_faithful pfx epsStr ε p.eW p.eb p.eγ p.eβ p.eμ p.ev
    p.dW p.db p.dγ p.dβ p.dμ p.dv p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ p.pμ p.pv e

def mbExpGraphEvalW (pfx epsStr : String) (N h w : Nat) (ε : ℝ) {ic mid oc kh kw r : Nat}
    (p : MBWEval ic mid oc r kh kw) (e : SHlo (N * (ic * h * w))) : SHlo (N * (oc * h * w)) :=
  mbExpGraphBEval pfx epsStr (h := h) (w := w) ε p.eW p.eb p.eγ p.eβ p.eμ p.ev
    p.dW p.db p.dγ p.dβ p.dμ p.dv p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ p.pμ p.pv e
theorem mbExpGraphEvalW_faithful (pfx epsStr : String) (N h w : Nat) (ε : ℝ) {ic mid oc kh kw r : Nat}
    (p : MBWEval ic mid oc r kh kw) (e : SHlo (N * (ic * h * w))) :
    den (mbExpGraphEvalW pfx epsStr N h w ε p e) = mbExpEvalW N h w ε p (den e) := by
  unfold mbExpGraphEvalW mbExpEvalW
  exact mbExpGraphBEval_faithful pfx epsStr ε p.eW p.eb p.eγ p.eβ p.eμ p.ev
    p.dW p.db p.dγ p.dβ p.dμ p.dv p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ p.pμ p.pv e

-- ════════════════════════════════════════════════════════════════
-- § The full B0 inference graph + faithfulness (all 16 MBConv blocks)
-- ════════════════════════════════════════════════════════════════

/-- The **sixteen-block EfficientNet-B0 inference forward graph** at the batched index
    `N·(c·h·w)`: stem → 16 MBConv blocks → head → GAP → dense, every one of the 49 BatchNorm sites
    reading frozen running statistics through the `bnEval` descriptor. The eval twin of
    `efficientnetFwdGraphB_full`, and the typed form of the shipped `efficientnet_fwd_eval`. -/
def efficientnetFwdGraphB_fullEval (N : Nat) (epsStr : String) (ε : ℝ) (w : B0WeightsEval nCls)
    (x : Vec (N * (3 * 224 * 224))) : SHlo (N * nCls) :=
  headGraphBEval epsStr (h := 7) (w := 7) ε w.hW w.hb w.hγ w.hβ w.hμ w.hv w.fcW w.fcb
    (mbExpGraphEvalW "b16" epsStr N 7 7 ε w.b16
      (mbResidGraphEvalW "b15" epsStr N 7 7 ε w.b15
        (mbResidGraphEvalW "b14" epsStr N 7 7 ε w.b14
          (mbResidGraphEvalW "b13" epsStr N 7 7 ε w.b13
            (mbStridedGraphEvalW "b12" epsStr N 7 7 ε w.b12
              (mbResidGraphEvalW "b11" epsStr N 14 14 ε w.b11
                (mbResidGraphEvalW "b10" epsStr N 14 14 ε w.b10
                  (mbExpGraphEvalW "b9" epsStr N 14 14 ε w.b9
                    (mbResidGraphEvalW "b8" epsStr N 14 14 ε w.b8
                      (mbResidGraphEvalW "b7" epsStr N 14 14 ε w.b7
                        (mbStridedGraphEvalW "b6" epsStr N 14 14 ε w.b6
                          (mbResidGraphEvalW "b5" epsStr N 28 28 ε w.b5
                            (mbStridedGraphEvalW "b4" epsStr N 28 28 ε w.b4
                              (mbResidGraphEvalW "b3" epsStr N 56 56 ε w.b3
                                (mbStridedGraphEvalW "b2" epsStr N 56 56 ε w.b2
                                  (mbNoExpGraphEvalW "b1" epsStr N 112 112 ε w.b1
                                    (stemGraphBEval epsStr (h := 112) (w := 112) w.sW w.sb ε w.sγ w.sβ w.sμ w.sv
                                      (.operand "%x" x))))))))))))))))))

/-- ⭐ **Sixteen-block inference EfficientNet-B0 forward faithfulness.** The typed graph denotes
    `efficientnetForwardB_fullEval`: one `rw` per block with the `*GraphEvalW_faithful` lemmas
    (outermost → innermost), then a structural `rfl` — the training twin's recipe. -/
theorem efficientnetFwdGraphB_fullEval_faithful (N : Nat) (epsStr : String) (ε : ℝ)
    (w : B0WeightsEval nCls) (x : Vec (N * (3 * 224 * 224))) :
    den (efficientnetFwdGraphB_fullEval N epsStr ε w x) = efficientnetForwardB_fullEval N ε w x := by
  rw [efficientnetFwdGraphB_fullEval, headGraphBEval_faithful,
      mbExpGraphEvalW_faithful, mbResidGraphEvalW_faithful, mbResidGraphEvalW_faithful, mbResidGraphEvalW_faithful, mbStridedGraphEvalW_faithful, mbResidGraphEvalW_faithful, mbResidGraphEvalW_faithful, mbExpGraphEvalW_faithful, mbResidGraphEvalW_faithful, mbResidGraphEvalW_faithful, mbStridedGraphEvalW_faithful, mbResidGraphEvalW_faithful, mbStridedGraphEvalW_faithful, mbResidGraphEvalW_faithful, mbStridedGraphEvalW_faithful, mbNoExpGraphEvalW_faithful,
      stemGraphBEval_faithful, den_operand]
  rfl

end StableHLO

end Proofs
