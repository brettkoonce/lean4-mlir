import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0Eval
import LeanMlir.Proofs.Codegen.StableHLO.Pretty

/-! # EfficientNet-B0 with stochastic depth and classifier dropout — forward, graph, faithfulness

`EfficientNetFullB0` and `EfficientNetFullB0Eval` state the sixteen-block net without its two
regularisers; the `*drop*` / `*do*` artifacts render them. This file states both forwards WITH
them, as the renderer (`enetFwdChain`) emits them, and proves the typed graphs denote them:

* **stochastic depth** (`sd`, the `%dp<i>` inputs) — a per-example scale `dropPath` on the
  residual BRANCH, before the skip add, at the nine blocks that carry a skip (`b3 b5 b7 b8 b10
  b11 b13 b14 b15`, the renderer's `enetDropIdxs` `[2, 4, 6, 7, 9, 10, 12, 13, 14]`, block-indexed);
* **classifier dropout** (`cd`, the `%do` input) — a per-element `dropout` between the GAP and
  the dense, width 1280 at every class count.

Each is `Option`al, as the renderer's `sd` / `cd` flags are: `none` renders no node, so one
statement covers `efficientnet_drop_fwd` (`sd` only), `efficientnet_do_fwd` (`cd` only) and
`efficientnetin_dropdo_fwd` (both), and their `_eval` twins at inference BatchNorm.

* `efficientnetFwdGraphBFullDrop_faithful` / `efficientnetFwdGraphBFullEvalDrop_faithful` — the
  graph denotes the forward, at every mask;
* `efficientnetForwardBFullDrop_none` / `…_ones` — with no site, or at the all-ones masks the
  driver passes at eval, the forward IS `efficientnetForwardBFull` (likewise the eval twins). The
  identity is exact: the keep probability is folded into the mask (`Training/DropPath`).

The residual-block-with-drop and the dropout head are text-guarded against the renderer in
`Codegen/FwdGraphTextTies` at training BatchNorm (the eval graphs carry the three-block eval
graph's SSA names, as `EfficientNetFullB0Eval` records). These artifacts are f32. The train steps'
backward through the drop sites is outside this statement, as it is outside
`EfficientNetStepTieG`.

## References

- Huang et al. 2016, *Deep Networks with Stochastic Depth*. <https://arxiv.org/abs/1603.09382>
- Srivastava et al. 2014, *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. <https://jmlr.org/papers/v15/srivastava14a.html>
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Optional drop sites — `none` is the identity, as the renderer emits nothing
-- ════════════════════════════════════════════════════════════════

/-- Drop-path at a site that may be absent: `none` is the identity. -/
noncomputable def dropPathOpt (N n : Nat) : Option (Vec N) → Vec (N * n) → Vec (N * n)
  | none => id
  | some s => dropPath N n s

/-- Dropout at a site that may be absent: `none` is the identity. -/
noncomputable def dropoutOpt {m : Nat} : Option (Vec m) → Vec m → Vec m
  | none => id
  | some mk => dropout mk

/-- At the all-ones scale drop-path is the identity. -/
theorem dropPathOpt_ones (N n : Nat) : dropPathOpt N n (some fun _ => 1) = id :=
  funext (dropPath_ones_id N n)

/-- At the all-ones mask dropout is the identity. -/
theorem dropoutOpt_ones {m : Nat} : dropoutOpt (some (fun _ => 1 : Vec m)) = id :=
  funext dropout_ones_id

-- ════════════════════════════════════════════════════════════════
-- § The two sites, at training BatchNorm
-- ════════════════════════════════════════════════════════════════

/-- **A residual MBConv6 block with its drop site**: the scale on the branch, then the skip add
    (`eFwd`'s placement). -/
noncomputable def mbResidDropW (N h w : Nat) {c mid kh kw r : Nat} (p : MBW c mid c r kh kw)
    (s : Option (Vec N)) : Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  residual (dropPathOpt N (c * h * w) s ∘ (projB N (h := h) (w := w) p.pW p.pb p.pε p.pγ p.pβ ∘
    seB N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2 ∘
    dwbsB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ ∘
    cbsB N (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ))

theorem mbResidDropW_none (N h w : Nat) {c mid kh kw r : Nat} (p : MBW c mid c r kh kw) :
    mbResidDropW N h w p none = mbResidW N h w p :=
  -- not `rfl`: a `rfl` lemma is a dsimp lemma, and the net-level `simp only` would then hand
  -- the kernel the whole ladder to unfold
  funext fun _ => rfl

theorem mbResidDropW_ones (N h w : Nat) {c mid kh kw r : Nat} (p : MBW c mid c r kh kw) :
    mbResidDropW N h w p (some fun _ => 1) = mbResidW N h w p := by
  rw [mbResidDropW, dropPathOpt_ones]; rfl

/-- **The head with classifier dropout**: 1×1 conv-bn-swish → GAP → dropout → dense. -/
noncomputable def headDoFwdB (N : Nat) {c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC) (m : Option (Vec (N * oc))) :
    Vec (N * (c * h * w)) → Vec (N * nC) :=
  StableHLO.batchMap N (dense Wfc bfc) ∘ dropoutOpt m ∘
    StableHLO.batchMap N (globalAvgPoolFlat oc h w) ∘ cbsB N (h := h) (w := w) Wh bh εh γh βh

theorem headDoFwdB_none (N : Nat) {c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc) (Wfc : Mat oc nC) (bfc : Vec nC) :
    headDoFwdB N (h := h) (w := w) Wh bh εh γh βh Wfc bfc none
      = headFwdB N (h := h) (w := w) Wh bh εh γh βh Wfc bfc :=
  -- not `rfl`: a `rfl` lemma is a dsimp lemma, and the net-level `simp only` would then hand
  -- the kernel the whole ladder to unfold
  funext fun _ => rfl

theorem headDoFwdB_ones (N : Nat) {c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc) (Wfc : Mat oc nC) (bfc : Vec nC) :
    headDoFwdB N (h := h) (w := w) Wh bh εh γh βh Wfc bfc (some fun _ => 1)
      = headFwdB N (h := h) (w := w) Wh bh εh γh βh Wfc bfc := by
  rw [headDoFwdB, dropoutOpt_ones]; rfl

-- ════════════════════════════════════════════════════════════════
-- § The two sites, at inference BatchNorm
-- ════════════════════════════════════════════════════════════════

/-- `mbResidDropW` at inference BatchNorm. -/
noncomputable def mbResidEvalDropW (N h w : Nat) (ε : ℝ) {c mid kh kw r : Nat}
    (p : MBWEval c mid c r kh kw) (s : Option (Vec N)) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  residual (dropPathOpt N (c * h * w) s ∘ (projBEval N (h := h) (w := w) p.pW p.pb ε p.pγ p.pβ p.pμ p.pv ∘
    seB N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2 ∘
    dwbsBEval N (h := h) (w := w) p.dW p.db ε p.dγ p.dβ p.dμ p.dv ∘
    cbsBEval N (h := h) (w := w) p.eW p.eb ε p.eγ p.eβ p.eμ p.ev))

theorem mbResidEvalDropW_none (N h w : Nat) (ε : ℝ) {c mid kh kw r : Nat}
    (p : MBWEval c mid c r kh kw) : mbResidEvalDropW N h w ε p none = mbResidEvalW N h w ε p :=
  -- not `rfl`: a `rfl` lemma is a dsimp lemma, and the net-level `simp only` would then hand
  -- the kernel the whole ladder to unfold
  funext fun _ => rfl

theorem mbResidEvalDropW_ones (N h w : Nat) (ε : ℝ) {c mid kh kw r : Nat}
    (p : MBWEval c mid c r kh kw) :
    mbResidEvalDropW N h w ε p (some fun _ => 1) = mbResidEvalW N h w ε p := by
  rw [mbResidEvalDropW, dropPathOpt_ones]; rfl

/-- `headDoFwdB` at inference BatchNorm. -/
noncomputable def headDoFwdBEval (N : Nat) {c oc h w nC : Nat} (ε : ℝ)
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (γh βh μh vh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC) (m : Option (Vec (N * oc))) :
    Vec (N * (c * h * w)) → Vec (N * nC) :=
  StableHLO.batchMap N (dense Wfc bfc) ∘ dropoutOpt m ∘
    StableHLO.batchMap N (globalAvgPoolFlat oc h w) ∘ cbsBEval N (h := h) (w := w) Wh bh ε γh βh μh vh

theorem headDoFwdBEval_none (N : Nat) {c oc h w nC : Nat} (ε : ℝ)
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (γh βh μh vh : Vec oc) (Wfc : Mat oc nC) (bfc : Vec nC) :
    headDoFwdBEval N (h := h) (w := w) ε Wh bh γh βh μh vh Wfc bfc none
      = headFwdBEval N (h := h) (w := w) ε Wh bh γh βh μh vh Wfc bfc :=
  -- not `rfl`: a `rfl` lemma is a dsimp lemma, and the net-level `simp only` would then hand
  -- the kernel the whole ladder to unfold
  funext fun _ => rfl

theorem headDoFwdBEval_ones (N : Nat) {c oc h w nC : Nat} (ε : ℝ)
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (γh βh μh vh : Vec oc) (Wfc : Mat oc nC) (bfc : Vec nC) :
    headDoFwdBEval N (h := h) (w := w) ε Wh bh γh βh μh vh Wfc bfc (some fun _ => 1)
      = headFwdBEval N (h := h) (w := w) ε Wh bh γh βh μh vh Wfc bfc := by
  rw [headDoFwdBEval, dropoutOpt_ones]; rfl

-- ════════════════════════════════════════════════════════════════
-- § The whole net with both regularisers
-- ════════════════════════════════════════════════════════════════

/-- **EfficientNet-B0 with stochastic depth and classifier dropout** at training BatchNorm:
    `efficientnetForwardBFull` with `sd`'s nine per-example scales on the skip blocks' branches
    (site `k` = the `k`-th of `b3 b5 b7 b8 b10 b11 b13 b14 b15`) and `cd`'s mask before the
    classifier. -/
noncomputable def efficientnetForwardBFullDrop (N : Nat) {nCls : Nat} (w : B0Weights nCls)
    (sd : Option (Fin 9 → Vec N)) (cd : Option (Vec (N * 1280))) (x : Vec (N * (3 * 224 * 224))) :
    Vec (N * nCls) :=
  headDoFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb cd
    (mbExpW N 7 7 w.b16
      (mbResidDropW N 7 7 w.b15 (sd.map (· 8))
      (mbResidDropW N 7 7 w.b14 (sd.map (· 7))
      (mbResidDropW N 7 7 w.b13 (sd.map (· 6))
      (mbStridedW N 7 7 w.b12
      (mbResidDropW N 14 14 w.b11 (sd.map (· 5))
      (mbResidDropW N 14 14 w.b10 (sd.map (· 4))
      (mbExpW N 14 14 w.b9
      (mbResidDropW N 14 14 w.b8 (sd.map (· 3))
      (mbResidDropW N 14 14 w.b7 (sd.map (· 2))
      (mbStridedW N 14 14 w.b6
      (mbResidDropW N 28 28 w.b5 (sd.map (· 1))
      (mbStridedW N 28 28 w.b4
      (mbResidDropW N 56 56 w.b3 (sd.map (· 0))
      (mbStridedW N 56 56 w.b2
      (mbNoExpW N 112 112 w.b1
      (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ x)))))))))))))))))

/-- With neither site rendered, the forward is `efficientnetForwardBFull`. -/
theorem efficientnetForwardBFullDrop_none (N : Nat) {nCls : Nat} (w : B0Weights nCls)
    (x : Vec (N * (3 * 224 * 224))) :
    efficientnetForwardBFullDrop N w none none x = efficientnetForwardBFull N w x := by
  simp only [efficientnetForwardBFullDrop, Option.map_none, mbResidDropW_none, headDoFwdB_none]
  rfl

/-- **At the all-ones masks the forward is `efficientnetForwardBFull`**, exactly — the masks the
    driver passes to the forward artifacts. -/
theorem efficientnetForwardBFullDrop_ones (N : Nat) {nCls : Nat} (w : B0Weights nCls)
    (x : Vec (N * (3 * 224 * 224))) :
    efficientnetForwardBFullDrop N w (some fun _ _ => 1) (some fun _ => 1) x
      = efficientnetForwardBFull N w x := by
  simp only [efficientnetForwardBFullDrop, Option.map_some, mbResidDropW_ones, headDoFwdB_ones]
  rfl

/-- `efficientnetForwardBFullDrop` at inference BatchNorm. -/
noncomputable def efficientnetForwardBFullEvalDrop (N : Nat) (ε : ℝ) {nCls : Nat}
    (w : B0WeightsEval nCls) (sd : Option (Fin 9 → Vec N)) (cd : Option (Vec (N * 1280)))
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * nCls) :=
  headDoFwdBEval N (h := 7) (w := 7) ε w.hW w.hb w.hγ w.hβ w.hμ w.hv w.fcW w.fcb cd
    (mbExpEvalW N 7 7 ε w.b16
      (mbResidEvalDropW N 7 7 ε w.b15 (sd.map (· 8))
      (mbResidEvalDropW N 7 7 ε w.b14 (sd.map (· 7))
      (mbResidEvalDropW N 7 7 ε w.b13 (sd.map (· 6))
      (mbStridedEvalW N 7 7 ε w.b12
      (mbResidEvalDropW N 14 14 ε w.b11 (sd.map (· 5))
      (mbResidEvalDropW N 14 14 ε w.b10 (sd.map (· 4))
      (mbExpEvalW N 14 14 ε w.b9
      (mbResidEvalDropW N 14 14 ε w.b8 (sd.map (· 3))
      (mbResidEvalDropW N 14 14 ε w.b7 (sd.map (· 2))
      (mbStridedEvalW N 14 14 ε w.b6
      (mbResidEvalDropW N 28 28 ε w.b5 (sd.map (· 1))
      (mbStridedEvalW N 28 28 ε w.b4
      (mbResidEvalDropW N 56 56 ε w.b3 (sd.map (· 0))
      (mbStridedEvalW N 56 56 ε w.b2
      (mbNoExpEvalW N 112 112 ε w.b1
      (stemBEval N (h := 112) (w := 112) w.sW w.sb ε w.sγ w.sβ w.sμ w.sv x)))))))))))))))))

theorem efficientnetForwardBFullEvalDrop_none (N : Nat) (ε : ℝ) {nCls : Nat}
    (w : B0WeightsEval nCls) (x : Vec (N * (3 * 224 * 224))) :
    efficientnetForwardBFullEvalDrop N ε w none none x = efficientnetForwardBFullEval N ε w x := by
  simp only [efficientnetForwardBFullEvalDrop, Option.map_none, mbResidEvalDropW_none,
    headDoFwdBEval_none]
  rfl

/-- **At the all-ones masks the inference forward is `efficientnetForwardBFullEval`**, exactly. -/
theorem efficientnetForwardBFullEvalDrop_ones (N : Nat) (ε : ℝ) {nCls : Nat}
    (w : B0WeightsEval nCls) (x : Vec (N * (3 * 224 * 224))) :
    efficientnetForwardBFullEvalDrop N ε w (some fun _ _ => 1) (some fun _ => 1) x
      = efficientnetForwardBFullEval N ε w x := by
  simp only [efficientnetForwardBFullEvalDrop, Option.map_some, mbResidEvalDropW_ones,
    headDoFwdBEval_ones]
  rfl

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The graphs — a `dropPathB` / `dropoutB` node exactly where the site is `some`
-- ════════════════════════════════════════════════════════════════

/-- A `dropPathB` node when the site is rendered, nothing otherwise. -/
def dropPathOptG (mN : String) {N n : Nat} : Option (Vec N) → SHlo (N * n) → SHlo (N * n)
  | none, e => e
  | some s, e => .dropPathB mN s e

theorem den_dropPathOptG (mN : String) {N n : Nat} (s : Option (Vec N)) (e : SHlo (N * n)) :
    den (dropPathOptG mN s e) = dropPathOpt N n s (den e) := by
  cases s <;> rfl

/-- A `dropoutB` node when the site is rendered, nothing otherwise. -/
def dropoutOptG (mN : String) {N n : Nat} : Option (Vec (N * n)) → SHlo (N * n) → SHlo (N * n)
  | none, e => e
  | some m, e => .dropoutB mN m e

theorem den_dropoutOptG (mN : String) {N n : Nat} (m : Option (Vec (N * n))) (e : SHlo (N * n)) :
    den (dropoutOptG mN m e) = dropoutOpt m (den e) := by
  cases m <;> rfl

/-- **Residual MBConv6 with its drop site**: `mbResidGraphB` with `dropPathOptG` on the branch
    before the `addVB` — the node sequence `eFwd … (drop := some i)` emits. -/
def mbResidDropGraphB (p epsStr mN : String) {N c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (γp βp : Vec c) (s : Option (Vec N))
    (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  .addVB
    (dropPathOptG mN s
      (.bnBatchF s!"%{p}pg" s!"%{p}pbt" epsStr εp γp βp
        (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" (biasName false "" c) Wp bp)
          (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zW1" s!"%{p}zb1" s!"%{p}zW2" s!"%{p}zb2"
              Wz₁ bz₁ Wz₂ bz₂)
            (.batchOp (N := N) .swish (.bnBatchF s!"%{p}dg" s!"%{p}dbt" epsStr εd γd βd
              (.batchOp (N := N) (.depthwise (h := h) (w := w) s!"%{p}dW" (biasName false "" mid) Wd bd)
                (.batchOp (N := N) .swish (.bnBatchF s!"%{p}eg" s!"%{p}ebt" epsStr εe γe βe
                  (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}eW" (biasName false "" mid) We be)
                    e)))))))))) e

/-- With no drop site it is `mbResidGraphB`. -/
theorem mbResidDropGraphB_none (p epsStr mN : String) {N c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (γp βp : Vec c) (e : SHlo (N * (c * h * w))) :
    mbResidDropGraphB p epsStr mN We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp none e
      = mbResidGraphB p epsStr We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp e := rfl

/-- The graph with its drop site as a weight-bundle wrapper. -/
def mbResidDropGraphW (pfx epsStr mN : String) (N h w : Nat) {c mid kh kw r : Nat}
    (p : MBW c mid c r kh kw) (s : Option (Vec N)) (e : SHlo (N * (c * h * w))) :
    SHlo (N * (c * h * w)) :=
  mbResidDropGraphB pfx epsStr mN (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ p.dW p.db p.dε p.dγ p.dβ
    p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε p.pγ p.pβ s e

theorem mbResidDropGraphW_faithful (pfx epsStr mN : String) (N h w : Nat) {c mid kh kw r : Nat}
    (p : MBW c mid c r kh kw) (s : Option (Vec N)) (e : SHlo (N * (c * h * w))) :
    den (mbResidDropGraphW pfx epsStr mN N h w p s e) = mbResidDropW N h w p s (den e) := by
  unfold mbResidDropGraphW mbResidDropGraphB mbResidDropW projB seB dwbsB cbsB residual biPath
  simp only [den_addVB, den_dropPathOptG, den_batchOp, denOp, den_bnBatchF,
    ↓den_batchOp_swish_eq_swishF, swishF_faithful, Function.comp_apply]

/-- **Head with classifier dropout**: `headGraphB` with `dropoutOptG` between the GAP and the
    dense, its mask the input `mN` (the renderer's is `doName`). -/
def headGraphBDo (epsStr mN : String) {N c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC) (m : Option (Vec (N * oc)))
    (e : SHlo (N * (c * h * w))) : SHlo (N * nC) :=
  .batchOp (N := N) (.dense "%Wd" "%bd" Wfc bfc)
    (dropoutOptG mN m
      (.batchOp (N := N) (.gap (c := oc) (h := h) (w := w))
        (.batchOp (N := N) .swish (.bnBatchF "%hg" "%hbt" epsStr εh γh βh
          (.batchOp (N := N) (.conv (h := h) (w := w) "%hW" (biasName false "" oc) Wh bh) e)))))

theorem headGraphBDo_faithful (epsStr mN : String) {N c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC) (m : Option (Vec (N * oc))) (e : SHlo (N * (c * h * w))) :
    den (headGraphBDo epsStr mN Wh bh εh γh βh Wfc bfc m e)
      = headDoFwdB N (h := h) (w := w) Wh bh εh γh βh Wfc bfc m (den e) := by
  unfold headGraphBDo headDoFwdB cbsB
  simp only [den_batchOp, denOp, den_dropoutOptG, den_bnBatchF, ↓den_batchOp_swish_eq_swishF,
    swishF_faithful, Function.comp_apply]

/-- `mbResidDropGraphB` at inference BatchNorm (`mbResidGraphBEval`'s nodes). -/
def mbResidDropGraphBEval (p epsStr mN : String) {N c mid h w kHd kWd r : Nat} (ε : ℝ)
    (We : Kernel4 mid c 1 1) (be : Vec mid) (γe βe μe ve : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (γd βd μd vd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (γp βp μp vp : Vec c) (s : Option (Vec N))
    (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  .addV
    (dropPathOptG mN s
      (.batchOp (N := N) (.bnEval (h := h) (w := w) s!"%{p}pg" s!"%{p}pbt" s!"%{p}pmu" s!"%{p}pvar"
          epsStr ε γp βp μp vp)
        (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" s!"%{p}pb" Wp bp)
          (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zWa" s!"%{p}zba" s!"%{p}zWb"
              s!"%{p}zbb" Wz₁ bz₁ Wz₂ bz₂)
            (.swishF (.batchOp (N := N) (.bnEval (h := h) (w := w) s!"%{p}dg" s!"%{p}dbt"
                s!"%{p}dmu" s!"%{p}dvar" epsStr ε γd βd μd vd)
              (.batchOp (N := N) (.depthwise (h := h) (w := w) s!"%{p}dW" s!"%{p}db" Wd bd)
                (.swishF (.batchOp (N := N) (.bnEval (h := h) (w := w) s!"%{p}eg" s!"%{p}ebt"
                    s!"%{p}emu" s!"%{p}evar" epsStr ε γe βe μe ve)
                  (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}eW" s!"%{p}eb" We be)
                    e)))))))))) e

/-- The inference graph with its drop site as a weight-bundle wrapper. -/
def mbResidDropGraphEvalW (pfx epsStr mN : String) (N h w : Nat) (ε : ℝ) {c mid kh kw r : Nat}
    (p : MBWEval c mid c r kh kw) (s : Option (Vec N)) (e : SHlo (N * (c * h * w))) :
    SHlo (N * (c * h * w)) :=
  mbResidDropGraphBEval pfx epsStr mN (h := h) (w := w) ε p.eW p.eb p.eγ p.eβ p.eμ p.ev
    p.dW p.db p.dγ p.dβ p.dμ p.dv p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ p.pμ p.pv s e

theorem mbResidDropGraphEvalW_faithful (pfx epsStr mN : String) (N h w : Nat) (ε : ℝ)
    {c mid kh kw r : Nat} (p : MBWEval c mid c r kh kw) (s : Option (Vec N))
    (e : SHlo (N * (c * h * w))) :
    den (mbResidDropGraphEvalW pfx epsStr mN N h w ε p s e) = mbResidEvalDropW N h w ε p s (den e) := by
  unfold mbResidDropGraphEvalW mbResidDropGraphBEval mbResidEvalDropW projBEval seB dwbsBEval
    cbsBEval residual biPath
  simp only [den_addV, den_dropPathOptG, den_batchOp, denOp, swishF_faithful, Function.comp_apply]

/-- `headGraphBDo` at inference BatchNorm (`headGraphBEval`'s nodes). -/
def headGraphBEvalDo (epsStr mN : String) {N c oc h w nC : Nat} (ε : ℝ)
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (γh βh μh vh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC) (m : Option (Vec (N * oc)))
    (e : SHlo (N * (c * h * w))) : SHlo (N * nC) :=
  .batchOp (N := N) (.dense "%Wfc" "%bfc" Wfc bfc)
    (dropoutOptG mN m
      (.batchOp (N := N) (.gap (c := oc) (h := h) (w := w))
        (.swishF (.batchOp (N := N) (.bnEval (h := h) (w := w) "%hg" "%hbt" "%hmu" "%hvar" epsStr
            ε γh βh μh vh)
          (.batchOp (N := N) (.conv (h := h) (w := w) "%hW" "%hb" Wh bh) e)))))

theorem headGraphBEvalDo_faithful (epsStr mN : String) {N c oc h w nC : Nat} (ε : ℝ)
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (γh βh μh vh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC) (m : Option (Vec (N * oc))) (e : SHlo (N * (c * h * w))) :
    den (headGraphBEvalDo epsStr mN ε Wh bh γh βh μh vh Wfc bfc m e)
      = headDoFwdBEval N (h := h) (w := w) ε Wh bh γh βh μh vh Wfc bfc m (den e) := by
  unfold headGraphBEvalDo headDoFwdBEval cbsBEval
  simp only [den_batchOp, denOp, den_dropoutOptG, swishF_faithful, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The whole-net graphs + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- **The B0 forward graph with stochastic depth and classifier dropout** — the typed form of
    `efficientnet_drop_fwd` / `efficientnet_do_fwd` / `efficientnetin_dropdo_fwd`: each skip
    block's drop site reads `%dp<i>` at its BLOCK index `i`, the classifier dropout `%do`. -/
def efficientnetFwdGraphBFullDrop (N : Nat) (epsStr : String) {nCls : Nat} (w : B0Weights nCls)
    (sd : Option (Fin 9 → Vec N)) (cd : Option (Vec (N * 1280))) (x : Vec (N * (3 * 224 * 224))) :
    SHlo (N * nCls) :=
  headGraphBDo epsStr doName (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb cd
    (mbExpGraphW "b16" epsStr N 7 7 w.b16
      (mbResidDropGraphW "b15" epsStr (dpName 14) N 7 7 w.b15 (sd.map (· 8))
      (mbResidDropGraphW "b14" epsStr (dpName 13) N 7 7 w.b14 (sd.map (· 7))
      (mbResidDropGraphW "b13" epsStr (dpName 12) N 7 7 w.b13 (sd.map (· 6))
      (mbStridedGraphW "b12" epsStr N 7 7 w.b12
      (mbResidDropGraphW "b11" epsStr (dpName 10) N 14 14 w.b11 (sd.map (· 5))
      (mbResidDropGraphW "b10" epsStr (dpName 9) N 14 14 w.b10 (sd.map (· 4))
      (mbExpGraphW "b9" epsStr N 14 14 w.b9
      (mbResidDropGraphW "b8" epsStr (dpName 7) N 14 14 w.b8 (sd.map (· 3))
      (mbResidDropGraphW "b7" epsStr (dpName 6) N 14 14 w.b7 (sd.map (· 2))
      (mbStridedGraphW "b6" epsStr N 14 14 w.b6
      (mbResidDropGraphW "b5" epsStr (dpName 4) N 28 28 w.b5 (sd.map (· 1))
      (mbStridedGraphW "b4" epsStr N 28 28 w.b4
      (mbResidDropGraphW "b3" epsStr (dpName 2) N 56 56 w.b3 (sd.map (· 0))
      (mbStridedGraphW "b2" epsStr N 56 56 w.b2
      (mbNoExpGraphW "b1" epsStr N 112 112 w.b1
      (stemGraphB epsStr (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ (.operand "%x" x))))))))))))))))))

/-- **The graph denotes the forward, at every mask** — one `rw` per block, then `rfl`. -/
theorem efficientnetFwdGraphBFullDrop_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : B0Weights nCls) (sd : Option (Fin 9 → Vec N)) (cd : Option (Vec (N * 1280)))
    (x : Vec (N * (3 * 224 * 224))) :
    den (efficientnetFwdGraphBFullDrop N epsStr w sd cd x) = efficientnetForwardBFullDrop N w sd cd x := by
  rw [efficientnetFwdGraphBFullDrop, headGraphBDo_faithful,
      mbExpGraphW_faithful, mbResidDropGraphW_faithful, mbResidDropGraphW_faithful, mbResidDropGraphW_faithful, mbStridedGraphW_faithful, mbResidDropGraphW_faithful, mbResidDropGraphW_faithful, mbExpGraphW_faithful, mbResidDropGraphW_faithful, mbResidDropGraphW_faithful, mbStridedGraphW_faithful, mbResidDropGraphW_faithful, mbStridedGraphW_faithful, mbResidDropGraphW_faithful, mbStridedGraphW_faithful, mbNoExpGraphW_faithful,
      stemGraphB_faithful, den_operand]
  rfl

/-- The inference twin — the typed form of `efficientnet_drop_fwd_eval` / `efficientnet_do_fwd_eval`. -/
def efficientnetFwdGraphBFullEvalDrop (N : Nat) (epsStr : String) (ε : ℝ) {nCls : Nat}
    (w : B0WeightsEval nCls) (sd : Option (Fin 9 → Vec N)) (cd : Option (Vec (N * 1280)))
    (x : Vec (N * (3 * 224 * 224))) : SHlo (N * nCls) :=
  headGraphBEvalDo epsStr doName (h := 7) (w := 7) ε w.hW w.hb w.hγ w.hβ w.hμ w.hv w.fcW w.fcb cd
    (mbExpGraphEvalW "b16" epsStr N 7 7 ε w.b16
      (mbResidDropGraphEvalW "b15" epsStr (dpName 14) N 7 7 ε w.b15 (sd.map (· 8))
      (mbResidDropGraphEvalW "b14" epsStr (dpName 13) N 7 7 ε w.b14 (sd.map (· 7))
      (mbResidDropGraphEvalW "b13" epsStr (dpName 12) N 7 7 ε w.b13 (sd.map (· 6))
      (mbStridedGraphEvalW "b12" epsStr N 7 7 ε w.b12
      (mbResidDropGraphEvalW "b11" epsStr (dpName 10) N 14 14 ε w.b11 (sd.map (· 5))
      (mbResidDropGraphEvalW "b10" epsStr (dpName 9) N 14 14 ε w.b10 (sd.map (· 4))
      (mbExpGraphEvalW "b9" epsStr N 14 14 ε w.b9
      (mbResidDropGraphEvalW "b8" epsStr (dpName 7) N 14 14 ε w.b8 (sd.map (· 3))
      (mbResidDropGraphEvalW "b7" epsStr (dpName 6) N 14 14 ε w.b7 (sd.map (· 2))
      (mbStridedGraphEvalW "b6" epsStr N 14 14 ε w.b6
      (mbResidDropGraphEvalW "b5" epsStr (dpName 4) N 28 28 ε w.b5 (sd.map (· 1))
      (mbStridedGraphEvalW "b4" epsStr N 28 28 ε w.b4
      (mbResidDropGraphEvalW "b3" epsStr (dpName 2) N 56 56 ε w.b3 (sd.map (· 0))
      (mbStridedGraphEvalW "b2" epsStr N 56 56 ε w.b2
      (mbNoExpGraphEvalW "b1" epsStr N 112 112 ε w.b1
      (stemGraphBEval epsStr (h := 112) (w := 112) w.sW w.sb ε w.sγ w.sβ w.sμ w.sv (.operand "%x" x))))))))))))))))))

theorem efficientnetFwdGraphBFullEvalDrop_faithful (N : Nat) (epsStr : String) (ε : ℝ) {nCls : Nat}
    (w : B0WeightsEval nCls) (sd : Option (Fin 9 → Vec N)) (cd : Option (Vec (N * 1280)))
    (x : Vec (N * (3 * 224 * 224))) :
    den (efficientnetFwdGraphBFullEvalDrop N epsStr ε w sd cd x)
      = efficientnetForwardBFullEvalDrop N ε w sd cd x := by
  rw [efficientnetFwdGraphBFullEvalDrop, headGraphBEvalDo_faithful,
      mbExpGraphEvalW_faithful, mbResidDropGraphEvalW_faithful, mbResidDropGraphEvalW_faithful, mbResidDropGraphEvalW_faithful, mbStridedGraphEvalW_faithful, mbResidDropGraphEvalW_faithful, mbResidDropGraphEvalW_faithful, mbExpGraphEvalW_faithful, mbResidDropGraphEvalW_faithful, mbResidDropGraphEvalW_faithful, mbStridedGraphEvalW_faithful, mbResidDropGraphEvalW_faithful, mbStridedGraphEvalW_faithful, mbResidDropGraphEvalW_faithful, mbStridedGraphEvalW_faithful, mbNoExpGraphEvalW_faithful,
      stemGraphBEval_faithful, den_operand]
  rfl

end StableHLO

end Proofs
