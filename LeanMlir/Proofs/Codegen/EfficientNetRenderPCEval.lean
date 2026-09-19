import LeanMlir.Proofs.Codegen.EfficientNetRenderPC

/-! # EfficientNet-B0 — the inference (frozen-statistics) stages and block graphs

The eval twin of `EfficientNetRenderPC.lean`: the batched stage abbreviations at frozen statistics
(`cbsBEval` / `stemBEval` / `dwbsBEval` / `dwbsSBEval` / `projBEval` — `seB` is unchanged, it has
no BN) and the five block graphs with their faithfulness (`*GraphBEval_faithful`).
`EfficientNetFullB0Eval` chains them into the shipped sixteen-block eval forward.

⭐ At inference the batch decouples: frozen statistics are constants, so the eval BN *is*
per-example — `batchMap N (bnPerChannelEvalTensor3 oc h w ε γ β μ v)`, `denOp`'s `bnEval` arm, read
off by `den_batchOp` — and every stage is `batchMap N` of a per-example op or a pointwise map.

⚠ One ε for the whole net, as the render emits, where the training def carries a separate `ε` per
site. The SSA names extend the training graph's (`%sg`/`%sbt` → `%smu`/`%svar`, `%b1dg`/`%b1dbt` →
`%b1dmu`/`%b1dvar`, …); names are pretty-printing metadata and do not enter `den`. 3-axiom clean.
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

/-- Batched strided (3×3 s2) stem conv → inference bn → swish (halves spatial). At the
    XLA-`SAME` phase, as `stemB` (`EfficientNetRenderPC.lean`) and the shipped
    `efficientnet_fwd_eval`. -/
noncomputable def stemBEval (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β μ v : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  swish (N * (oc * h * w)) ∘ StableHLO.batchMap N (bnPerChannelEvalTensor3 oc h w ε γ β μ v)
    ∘ StableHLO.batchMap N (flatConvStride2Xla W b)

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
    (.batchOp (N := N) (.convStridedXla (h := h) (w := w) "%sW" "%sb" Ws bs) e))

theorem stemGraphBEval_faithful (epsStr : String) {N ic oc h w : Nat}
    (Ws : Kernel4 oc ic 3 3) (bs : Vec oc) (ε : ℝ) (γs βs μs vs : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (stemGraphBEval epsStr Ws bs ε γs βs μs vs e)
      = stemBEval N (h := h) (w := w) Ws bs ε γs βs μs vs (den e) := by
  unfold stemGraphBEval stemBEval
  simp only [den_batchOp, denOp, swishF_faithful, Function.comp_apply]

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
  simp only [den_batchOp, denOp,
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
  simp only [den_batchOp, denOp, swishF_faithful, Function.comp_apply]

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
  simp only [den_batchOp, denOp,
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
  simp only [den_batchOp, denOp,
             swishF_faithful, Function.comp_apply]

end StableHLO
end Proofs
