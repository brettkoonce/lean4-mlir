import LeanMlir.Proofs.Codegen.EfficientNetRenderPC

/-! # EfficientNet-B0 — the INFERENCE forward, and its graph

The eval twin of `EfficientNetRenderPC.lean`, and the B0 peer of `ResNet34RenderPCEval.lean` /
`MobileNetV2RenderPCEval.lean`. That file proves `den (efficientnetFwdGraphB …) =
efficientnetForwardB …` for the **training** BN chain — `bnBatchLA`, which reduces μ/var over the
batch and spatial axes and genuinely COUPLES the batch. The deployed forward reads frozen running
statistics, and had no such theorem.

⭐⭐ **At inference the batch decouples, and that is what makes this net foldable.**
`bnBatchLA` is the one op in `efficientnetForwardB` that is not `batchMap N` of a per-example op —
it reduces μ/var across examples, which is why it has its own constructor. Frozen statistics are
constants, so the eval BN *is* per-example — `batchMap N (bnPerChannelEvalTensor3 oc h w ε γ β μ v)`
— and `den_batchOp_bnEval` proves the `bnEval` descriptor denotes exactly that, by `rfl`. With that
one site replaced, **every stage below is `batchMap N` of a per-example op or a pointwise map**
(read the five stage defs: conv, depthwise, SE and GAP/dense are `batchMap`, swish is pointwise,
the residual is a coordinatewise add). So every leaf the float budget needs is a per-example leaf
it already has.

⚠ That last paragraph is a statement about the SHAPE of these definitions, checkable by reading
them; the whole-net factorisation `efficientnetForwardBEval N = batchMap N (per-example forward)`
is NOT stated as a theorem here. Only the per-site claim is proved (`den_batchOp_bnEval`). Proving
the whole-net one needs a `batchMap N f ∘ batchMap N g = batchMap N (f ∘ g)` lemma, which the repo
does not yet have, plus a per-example B0 def to be the witness — worth doing, not done.

⚠ **And it is the only BN mode a whole-net float NUMBER can be stated at.** Training-mode BN's
error modulus is quadratic in the window, so the fold squares at each of this net's ten BN sites
(`planning/float_budget_numbers.md` §0.1). `EfficientNetFloatBudget.lean` states its number about
exactly the forward defined here.

Same rungs as the training file, one for one: the batched stage abbreviations at frozen
statistics (`cbsBEval` / `stemBEval` / `dwbsBEval` / `dwbsSBEval` / `projBEval` — `seB` is
unchanged, it has no BN), the four blocks, then `efficientnetForwardBEval` +
`efficientnetFwdGraphBEval` + `efficientnetFwdGraphBEval_faithful`. No new tokens: `bnEval` and
its `den` lemma were already there, they just had no whole-net B0 chain to sit in.

⚠ One ε for the whole net, as the render emits and as `resnet34Forward_full_pc_eval` and
`mobilenetv2Forward_full_pc_eval` do, where the training def carries a separate `ε` per site.
Each BN site instead grows by its frozen mean and variance, so the signature goes 64 stored
parameters → 75. The SSA names extend the training graph's (`%sg`/`%sbt` → `%smu`/`%svar`,
`%b1dg`/`%b1dbt` → `%b1dmu`/`%b1dvar`, …); names are pretty-printing metadata and do not enter
`den`. 3-axiom clean.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The batched INFERENCE stage abbreviations (eval mirrors of cbsB / … / projB)
-- ════════════════════════════════════════════════════════════════

/-- Batched conv → inference bn → swish (1×1 expand / generic stride-1 conv). -/
@[reducible] noncomputable def cbsBEval (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β μ v : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  swish (N * (oc * h * w)) ∘ StableHLO.batchMap N (bnPerChannelEvalTensor3 oc h w ε γ β μ v)
    ∘ StableHLO.batchMap N (flatConv W b)

/-- Batched strided (3×3 s2) stem conv → inference bn → swish (halves spatial). -/
noncomputable def stemBEval (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β μ v : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  swish (N * (oc * h * w)) ∘ StableHLO.batchMap N (bnPerChannelEvalTensor3 oc h w ε γ β μ v)
    ∘ StableHLO.batchMap N (flatConvStride2 W b)

/-- Batched depthwise (stride-1, k×k) → inference bn → swish. -/
@[reducible] noncomputable def dwbsBEval (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β μ v : Vec c) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  swish (N * (c * h * w)) ∘ StableHLO.batchMap N (bnPerChannelEvalTensor3 c h w ε γ β μ v)
    ∘ StableHLO.batchMap N (depthwiseFlat W b)

/-- Batched depthwise (stride-2 downsample, k×k) → inference bn → swish. -/
@[reducible] noncomputable def dwbsSBEval (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β μ v : Vec c) :
    Vec (N * (c * (2 * h) * (2 * w))) → Vec (N * (c * h * w)) :=
  swish (N * (c * h * w)) ∘ StableHLO.batchMap N (bnPerChannelEvalTensor3 c h w ε γ β μ v)
    ∘ StableHLO.batchMap N (depthwiseStride2Flat W b)

/-- Batched project: 1×1 conv → inference bn (no swish — the linear bottleneck). -/
@[reducible] noncomputable def projBEval (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β μ v : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  StableHLO.batchMap N (bnPerChannelEvalTensor3 oc h w ε γ β μ v)
    ∘ StableHLO.batchMap N (flatConv W b)

-- ════════════════════════════════════════════════════════════════
-- § Block inference ℝ-forwards (`seB` is shared — squeeze-excite has no BN)
-- ════════════════════════════════════════════════════════════════

/-- MBConv1 (`t=1`, no expand) at inference: dw-bn-swish → SE → project-bn. -/
noncomputable def mbNoExpFwdBEval (N : Nat) {ic oc h w kHd kWd r : Nat} (ε : ℝ)
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (γd βd μd vd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  projBEval N (h := h) (w := w) Wp bp ε γp βp μp vp ∘
    seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ ∘
    dwbsBEval N (h := h) (w := w) Wd bd ε γd βd μd vd

/-- MBConv6 with a stride-2 downsample at inference. -/
noncomputable def mbStridedFwdBEval (N : Nat) {ic mid oc h w kHd kWd r : Nat} (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (γe βe μe ve : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (γd βd μd vd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  projBEval N (h := h) (w := w) Wp bp ε γp βp μp vp ∘
    seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ ∘
    dwbsSBEval N (h := h) (w := w) Wd bd ε γd βd μd vd ∘
    cbsBEval N (h := 2 * h) (w := 2 * w) We be ε γe βe μe ve

/-- MBConv6 with an identity residual skip at inference. -/
noncomputable def mbResidFwdBEval (N : Nat) {c mid h w kHd kWd r : Nat} (ε : ℝ)
    (We : Kernel4 mid c 1 1) (be : Vec mid) (γe βe μe ve : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (γd βd μd vd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (γp βp μp vp : Vec c) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  residual (projBEval N (h := h) (w := w) Wp bp ε γp βp μp vp ∘
    seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ ∘
    dwbsBEval N (h := h) (w := w) Wd bd ε γd βd μd vd ∘
    cbsBEval N (h := h) (w := w) We be ε γe βe μe ve)

/-- Head at inference: 1×1 conv-bn-swish → global-avg-pool → dense classifier. -/
noncomputable def headFwdBEval (N : Nat) {c oc h w nC : Nat} (ε : ℝ)
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (γh βh μh vh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC) :
    Vec (N * (c * h * w)) → Vec (N * nC) :=
  StableHLO.batchMap N (dense Wfc bfc) ∘ StableHLO.batchMap N (globalAvgPoolFlat oc h w) ∘
    cbsBEval N (h := h) (w := w) Wh bh ε γh βh μh vh

-- ════════════════════════════════════════════════════════════════
-- § The representative batched EfficientNet-B0 INFERENCE ℝ-forward
-- ════════════════════════════════════════════════════════════════

/-- **The full inference EfficientNet-B0 forward** — `efficientnetForwardB` with frozen running
    statistics at all ten BN sites, at one shared ε. The map the deployed eval render computes,
    and the map `EfficientNetFloatBudget.lean` states its number about. Nested-application form
    (NOT `∘`), as the training twin, so the faithfulness proof closes by pure delta. -/
noncomputable def efficientnetForwardBEval
    (N : Nat) (ε : ℝ)
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (γs βs μs vs : Vec 32)
    (Wd1 : DepthwiseKernel 32 3 3) (bd1 : Vec 32) (γd1 βd1 μd1 vd1 : Vec 32)
    (Wz1a : Mat 32 8) (bz1a : Vec 8) (Wz1b : Mat 8 32) (bz1b : Vec 32)
    (Wp1 : Kernel4 16 32 1 1) (bp1 : Vec 16) (γp1 βp1 μp1 vp1 : Vec 16)
    (We2 : Kernel4 96 16 1 1) (be2 : Vec 96) (γe2 βe2 μe2 ve2 : Vec 96)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (γd2 βd2 μd2 vd2 : Vec 96)
    (Wz2a : Mat 96 4) (bz2a : Vec 4) (Wz2b : Mat 4 96) (bz2b : Vec 96)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (γp2 βp2 μp2 vp2 : Vec 24)
    (We3 : Kernel4 144 24 1 1) (be3 : Vec 144) (γe3 βe3 μe3 ve3 : Vec 144)
    (Wd3 : DepthwiseKernel 144 5 5) (bd3 : Vec 144) (γd3 βd3 μd3 vd3 : Vec 144)
    (Wz3a : Mat 144 6) (bz3a : Vec 6) (Wz3b : Mat 6 144) (bz3b : Vec 144)
    (Wp3 : Kernel4 24 144 1 1) (bp3 : Vec 24) (γp3 βp3 μp3 vp3 : Vec 24)
    (Wh : Kernel4 1280 24 1 1) (bh : Vec 1280) (γh βh μh vh : Vec 1280)
    (Wfc : Mat 1280 10) (bfc : Vec 10)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * 10) :=
  headFwdBEval N (h := 56) (w := 56) ε Wh bh γh βh μh vh Wfc bfc
    (mbResidFwdBEval N (h := 56) (w := 56) ε We3 be3 γe3 βe3 μe3 ve3 Wd3 bd3 γd3 βd3 μd3 vd3
        Wz3a bz3a Wz3b bz3b Wp3 bp3 γp3 βp3 μp3 vp3
      (mbStridedFwdBEval N (h := 56) (w := 56) ε We2 be2 γe2 βe2 μe2 ve2 Wd2 bd2 γd2 βd2 μd2 vd2
          Wz2a bz2a Wz2b bz2b Wp2 bp2 γp2 βp2 μp2 vp2
        (mbNoExpFwdBEval N (h := 112) (w := 112) ε Wd1 bd1 γd1 βd1 μd1 vd1 Wz1a bz1a Wz1b bz1b
            Wp1 bp1 γp1 βp1 μp1 vp1
          (stemBEval N (h := 112) (w := 112) Ws bs ε γs βs μs vs x))))

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Block inference `SHlo` graphs + faithfulness
--   `.bnBatchF` (a genuine batch reduction) is replaced everywhere by the batched
--   `.bnEval` DESCRIPTOR — legal precisely because frozen statistics do not reduce.
-- ════════════════════════════════════════════════════════════════

/-- Stem 3×3-s2 conv → inference bn → swish, batched. -/
def stemGraphBEval (epsStr : String) {N ic oc h w : Nat}
    (Ws : Kernel4 oc ic 3 3) (bs : Vec oc) (ε : ℝ) (γs βs μs vs : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) : SHlo (N * (oc * h * w)) :=
  .swishF (.batchOp (N := N) (.bnEval (h := h) (w := w) "%sg" "%sbt" "%smu" "%svar" epsStr
      ε γs βs μs vs)
    (.batchOp (N := N) (.convStrided (h := h) (w := w) "%sW" "%sb" Ws bs) e))

theorem stemGraphBEval_faithful (epsStr : String) {N ic oc h w : Nat}
    (Ws : Kernel4 oc ic 3 3) (bs : Vec oc) (ε : ℝ) (γs βs μs vs : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (stemGraphBEval epsStr Ws bs ε γs βs μs vs e)
      = stemBEval N (h := h) (w := w) Ws bs ε γs βs μs vs (den e) := by
  unfold stemGraphBEval stemBEval
  simp only [den_batchOp_convStrided, den_batchOp_bnEval, swishF_faithful, Function.comp_apply]

/-- MBConv1 (no expand) at inference: dw-bn-swish → SE → project-bn, batched. -/
def mbNoExpGraphBEval (p epsStr : String) {N ic oc h w kHd kWd r : Nat} (ε : ℝ)
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (γd βd μd vd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc)
    (e : SHlo (N * (ic * h * w))) : SHlo (N * (oc * h * w)) :=
  .batchOp (N := N) (.bnEval (h := h) (w := w) s!"%{p}pg" s!"%{p}pbt" s!"%{p}pmu" s!"%{p}pvar"
      epsStr ε γp βp μp vp)
    (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" s!"%{p}pb" Wp bp)
      (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zWa" s!"%{p}zba" s!"%{p}zWb" s!"%{p}zbb"
          Wz₁ bz₁ Wz₂ bz₂)
        (.swishF (.batchOp (N := N) (.bnEval (h := h) (w := w) s!"%{p}dg" s!"%{p}dbt"
            s!"%{p}dmu" s!"%{p}dvar" epsStr ε γd βd μd vd)
          (.batchOp (N := N) (.depthwise (h := h) (w := w) s!"%{p}dW" s!"%{p}db" Wd bd) e)))))

theorem mbNoExpGraphBEval_faithful (p epsStr : String) {N ic oc h w kHd kWd r : Nat} (ε : ℝ)
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (γd βd μd vd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc)
    (e : SHlo (N * (ic * h * w))) :
    den (mbNoExpGraphBEval p epsStr ε Wd bd γd βd μd vd Wz₁ bz₁ Wz₂ bz₂ Wp bp γp βp μp vp e)
      = mbNoExpFwdBEval N (h := h) (w := w) ε Wd bd γd βd μd vd Wz₁ bz₁ Wz₂ bz₂
          Wp bp γp βp μp vp (den e) := by
  unfold mbNoExpGraphBEval mbNoExpFwdBEval projBEval seB dwbsBEval
  simp only [den_batchOp_conv, den_batchOp_seBlock, den_batchOp_depthwise, den_batchOp_bnEval,
             swishF_faithful, Function.comp_apply]

/-- MBConv6 strided at inference: expand-bn-swish (at `2h×2w`) → strided dw-bn-swish → SE →
    project-bn, batched. -/
def mbStridedGraphBEval (p epsStr : String) {N ic mid oc h w kHd kWd r : Nat} (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (γe βe μe ve : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (γd βd μd vd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) : SHlo (N * (oc * h * w)) :=
  .batchOp (N := N) (.bnEval (h := h) (w := w) s!"%{p}pg" s!"%{p}pbt" s!"%{p}pmu" s!"%{p}pvar"
      epsStr ε γp βp μp vp)
    (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" s!"%{p}pb" Wp bp)
      (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zWa" s!"%{p}zba" s!"%{p}zWb" s!"%{p}zbb"
          Wz₁ bz₁ Wz₂ bz₂)
        (.swishF (.batchOp (N := N) (.bnEval (h := h) (w := w) s!"%{p}dg" s!"%{p}dbt"
            s!"%{p}dmu" s!"%{p}dvar" epsStr ε γd βd μd vd)
          (.batchOp (N := N) (.depthwiseStrided (h := h) (w := w) s!"%{p}dW" s!"%{p}db" Wd bd)
            (.swishF (.batchOp (N := N) (.bnEval (h := 2 * h) (w := 2 * w) s!"%{p}eg" s!"%{p}ebt"
                s!"%{p}emu" s!"%{p}evar" epsStr ε γe βe μe ve)
              (.batchOp (N := N) (.conv (h := 2 * h) (w := 2 * w) s!"%{p}eW" s!"%{p}eb" We be)
                e))))))))

theorem mbStridedGraphBEval_faithful (p epsStr : String) {N ic mid oc h w kHd kWd r : Nat} (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (γe βe μe ve : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (γd βd μd vd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (mbStridedGraphBEval p epsStr ε We be γe βe μe ve Wd bd γd βd μd vd Wz₁ bz₁ Wz₂ bz₂
          Wp bp γp βp μp vp e)
      = mbStridedFwdBEval N (h := h) (w := w) ε We be γe βe μe ve Wd bd γd βd μd vd
          Wz₁ bz₁ Wz₂ bz₂ Wp bp γp βp μp vp (den e) := by
  unfold mbStridedGraphBEval mbStridedFwdBEval projBEval seB dwbsSBEval cbsBEval
  simp only [den_batchOp_conv, den_batchOp_seBlock, den_batchOp_depthwiseStrided,
             den_batchOp_bnEval, swishF_faithful, Function.comp_apply]

/-- MBConv6 with identity residual at inference: `addV body skip`. -/
def mbResidGraphBEval (p epsStr : String) {N c mid h w kHd kWd r : Nat} (ε : ℝ)
    (We : Kernel4 mid c 1 1) (be : Vec mid) (γe βe μe ve : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (γd βd μd vd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (γp βp μp vp : Vec c)
    (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  .addV
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
                  e))))))))) e

theorem mbResidGraphBEval_faithful (p epsStr : String) {N c mid h w kHd kWd r : Nat} (ε : ℝ)
    (We : Kernel4 mid c 1 1) (be : Vec mid) (γe βe μe ve : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (γd βd μd vd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (γp βp μp vp : Vec c)
    (e : SHlo (N * (c * h * w))) :
    den (mbResidGraphBEval p epsStr ε We be γe βe μe ve Wd bd γd βd μd vd Wz₁ bz₁ Wz₂ bz₂
          Wp bp γp βp μp vp e)
      = mbResidFwdBEval N (h := h) (w := w) ε We be γe βe μe ve Wd bd γd βd μd vd
          Wz₁ bz₁ Wz₂ bz₂ Wp bp γp βp μp vp (den e) := by
  unfold mbResidGraphBEval mbResidFwdBEval projBEval seB dwbsBEval cbsBEval residual biPath
  simp only [den_batchOp_conv, den_batchOp_seBlock, den_batchOp_depthwise, den_batchOp_bnEval,
             swishF_faithful, den_addV, Function.comp_apply]

/-- Head at inference: 1×1 conv-bn-swish → GAP → dense, batched. -/
def headGraphBEval (epsStr : String) {N c oc h w nC : Nat} (ε : ℝ)
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (γh βh μh vh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC)
    (e : SHlo (N * (c * h * w))) : SHlo (N * nC) :=
  .batchOp (N := N) (.dense "%Wfc" "%bfc" Wfc bfc)
    (.batchOp (N := N) (.gap (c := oc) (h := h) (w := w))
      (.swishF (.batchOp (N := N) (.bnEval (h := h) (w := w) "%hg" "%hbt" "%hmu" "%hvar" epsStr
          ε γh βh μh vh)
        (.batchOp (N := N) (.conv (h := h) (w := w) "%hW" "%hb" Wh bh) e))))

theorem headGraphBEval_faithful (epsStr : String) {N c oc h w nC : Nat} (ε : ℝ)
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (γh βh μh vh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC)
    (e : SHlo (N * (c * h * w))) :
    den (headGraphBEval epsStr ε Wh bh γh βh μh vh Wfc bfc e)
      = headFwdBEval N (h := h) (w := w) ε Wh bh γh βh μh vh Wfc bfc (den e) := by
  unfold headGraphBEval headFwdBEval cbsBEval
  simp only [den_batchOp_dense, den_batchOp_gap, den_batchOp_conv, den_batchOp_bnEval,
             swishF_faithful, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The full inference graph + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- The representative **inference EfficientNet-B0 forward** graph at the batched index
    `N·(c·h·w)`: stem → MBConv1(no-exp) → MBConv6(strided 3×3) → MBConv6(5×5, residual) → head,
    every BN site reading frozen running statistics through the `bnEval` descriptor. The eval twin
    of `efficientnetFwdGraphB`. -/
def efficientnetFwdGraphBEval
    (N : Nat) (epsStr : String) (ε : ℝ)
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (γs βs μs vs : Vec 32)
    (Wd1 : DepthwiseKernel 32 3 3) (bd1 : Vec 32) (γd1 βd1 μd1 vd1 : Vec 32)
    (Wz1a : Mat 32 8) (bz1a : Vec 8) (Wz1b : Mat 8 32) (bz1b : Vec 32)
    (Wp1 : Kernel4 16 32 1 1) (bp1 : Vec 16) (γp1 βp1 μp1 vp1 : Vec 16)
    (We2 : Kernel4 96 16 1 1) (be2 : Vec 96) (γe2 βe2 μe2 ve2 : Vec 96)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (γd2 βd2 μd2 vd2 : Vec 96)
    (Wz2a : Mat 96 4) (bz2a : Vec 4) (Wz2b : Mat 4 96) (bz2b : Vec 96)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (γp2 βp2 μp2 vp2 : Vec 24)
    (We3 : Kernel4 144 24 1 1) (be3 : Vec 144) (γe3 βe3 μe3 ve3 : Vec 144)
    (Wd3 : DepthwiseKernel 144 5 5) (bd3 : Vec 144) (γd3 βd3 μd3 vd3 : Vec 144)
    (Wz3a : Mat 144 6) (bz3a : Vec 6) (Wz3b : Mat 6 144) (bz3b : Vec 144)
    (Wp3 : Kernel4 24 144 1 1) (bp3 : Vec 24) (γp3 βp3 μp3 vp3 : Vec 24)
    (Wh : Kernel4 1280 24 1 1) (bh : Vec 1280) (γh βh μh vh : Vec 1280)
    (Wfc : Mat 1280 10) (bfc : Vec 10)
    (x : Vec (N * (3 * 224 * 224))) : SHlo (N * 10) :=
  headGraphBEval epsStr (h := 56) (w := 56) ε Wh bh γh βh μh vh Wfc bfc
    (mbResidGraphBEval "b3" epsStr (h := 56) (w := 56) ε We3 be3 γe3 βe3 μe3 ve3
        Wd3 bd3 γd3 βd3 μd3 vd3 Wz3a bz3a Wz3b bz3b Wp3 bp3 γp3 βp3 μp3 vp3
      (mbStridedGraphBEval "b2" epsStr (h := 56) (w := 56) ε We2 be2 γe2 βe2 μe2 ve2
          Wd2 bd2 γd2 βd2 μd2 vd2 Wz2a bz2a Wz2b bz2b Wp2 bp2 γp2 βp2 μp2 vp2
        (mbNoExpGraphBEval "b1" epsStr (h := 112) (w := 112) ε Wd1 bd1 γd1 βd1 μd1 vd1
            Wz1a bz1a Wz1b bz1b Wp1 bp1 γp1 βp1 μp1 vp1
          (stemGraphBEval epsStr (h := 112) (w := 112) Ws bs ε γs βs μs vs (.operand "%x" x)))))

/-- ⭐ **Full inference EfficientNet-B0 forward faithfulness.**
    `den (efficientnetFwdGraphBEval …) = efficientnetForwardBEval …`, chained from the per-block
    `*GraphBEval_faithful` lemmas exactly as the training twin is. The eval half of "text = render
    of a proven graph" for this net, and the tie `EfficientNetFloatBudget.lean`'s number needs. -/
theorem efficientnetFwdGraphBEval_faithful
    (N : Nat) (epsStr : String) (ε : ℝ)
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (γs βs μs vs : Vec 32)
    (Wd1 : DepthwiseKernel 32 3 3) (bd1 : Vec 32) (γd1 βd1 μd1 vd1 : Vec 32)
    (Wz1a : Mat 32 8) (bz1a : Vec 8) (Wz1b : Mat 8 32) (bz1b : Vec 32)
    (Wp1 : Kernel4 16 32 1 1) (bp1 : Vec 16) (γp1 βp1 μp1 vp1 : Vec 16)
    (We2 : Kernel4 96 16 1 1) (be2 : Vec 96) (γe2 βe2 μe2 ve2 : Vec 96)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (γd2 βd2 μd2 vd2 : Vec 96)
    (Wz2a : Mat 96 4) (bz2a : Vec 4) (Wz2b : Mat 4 96) (bz2b : Vec 96)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (γp2 βp2 μp2 vp2 : Vec 24)
    (We3 : Kernel4 144 24 1 1) (be3 : Vec 144) (γe3 βe3 μe3 ve3 : Vec 144)
    (Wd3 : DepthwiseKernel 144 5 5) (bd3 : Vec 144) (γd3 βd3 μd3 vd3 : Vec 144)
    (Wz3a : Mat 144 6) (bz3a : Vec 6) (Wz3b : Mat 6 144) (bz3b : Vec 144)
    (Wp3 : Kernel4 24 144 1 1) (bp3 : Vec 24) (γp3 βp3 μp3 vp3 : Vec 24)
    (Wh : Kernel4 1280 24 1 1) (bh : Vec 1280) (γh βh μh vh : Vec 1280)
    (Wfc : Mat 1280 10) (bfc : Vec 10)
    (x : Vec (N * (3 * 224 * 224))) :
    den (efficientnetFwdGraphBEval N epsStr ε Ws bs γs βs μs vs Wd1 bd1 γd1 βd1 μd1 vd1 Wz1a bz1a Wz1b bz1b Wp1 bp1 γp1 βp1 μp1 vp1 We2 be2 γe2 βe2 μe2 ve2 Wd2 bd2 γd2 βd2 μd2 vd2 Wz2a bz2a Wz2b bz2b Wp2 bp2 γp2 βp2 μp2 vp2 We3 be3 γe3 βe3 μe3 ve3 Wd3 bd3 γd3 βd3 μd3 vd3 Wz3a bz3a Wz3b bz3b Wp3 bp3 γp3 βp3 μp3 vp3 Wh bh γh βh μh vh Wfc bfc x)
      = efficientnetForwardBEval N ε Ws bs γs βs μs vs Wd1 bd1 γd1 βd1 μd1 vd1 Wz1a bz1a Wz1b bz1b Wp1 bp1 γp1 βp1 μp1 vp1 We2 be2 γe2 βe2 μe2 ve2 Wd2 bd2 γd2 βd2 μd2 vd2 Wz2a bz2a Wz2b bz2b Wp2 bp2 γp2 βp2 μp2 vp2 We3 be3 γe3 βe3 μe3 ve3 Wd3 bd3 γd3 βd3 μd3 vd3 Wz3a bz3a Wz3b bz3b Wp3 bp3 γp3 βp3 μp3 vp3 Wh bh γh βh μh vh Wfc bfc x := by
  rw [efficientnetFwdGraphBEval, headGraphBEval_faithful, mbResidGraphBEval_faithful,
      mbStridedGraphBEval_faithful, mbNoExpGraphBEval_faithful, stemGraphBEval_faithful,
      den_operand]
  rfl

end StableHLO
end Proofs
