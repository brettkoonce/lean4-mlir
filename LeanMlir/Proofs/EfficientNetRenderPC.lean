import LeanMlir.Proofs.StableHLO

/-! # Item A — the BATCHED EfficientNet-B0 forward graph (true batch-norm, matches the render)

The EfficientNet peer of `MobileNetV2RenderPC.lean` / `ResNet34RenderPC.lean` — but EfficientNet's
operational render (`tests/TestEfficientNetFwd.lean`) emits **true batch-norm** (reduce μ/var over the
batch+spatial axes `[0,2,3]` per channel — `bnBatchTensor4`), which **couples the batch**. MNV2/r34
get away with a batch-1 `den` because their per-channel BN reduces `[2,3]` (per-example, separable);
EfficientNet's does not. So the forward graph here genuinely lives at the **batched index**
`N·(c·h·w)` (`StableHLO.batchOp`/`StableHLO.bnBatchF`, `StableHLO.lean`):

* every batch-separable op (conv / strided conv / depthwise / strided depthwise / dense / GAP / the
  whole SE block) is `batchMap N` of the proven per-example op (`SHlo.batchOp` + `BatchableOp`/`denOp`);
* the pointwise ops (swish, sigmoid, relu, residual `addV`) reuse their EXISTING tokens at the batched
  index — they are already block-diagonal there, no new token needed;
* the one batch-coupled op, true batch-norm, is `SHlo.bnBatchF`, denoting `bnBatchLA` (= the proven
  `bnBatchTensor4`, reindexed to the network's left-assoc `N·(oc·h·w)` flat layout).

We prove the FORWARD half — `den (graph) = forward` — for a representative EfficientNet-B0 that
structurally exercises **every** element of B0: the stride-2 stem conv-bn-swish, an MBConv1 (`t=1`,
**no expand**) SE block, an MBConv6 expand SE block with a **stride-2** downsample (3×3 depthwise), an
MBConv6 expand SE block with a **5×5** depthwise and an **identity residual** skip, the 1×1 conv-bn-swish
head, GAP and the dense classifier — all with **true batch-norm** and the squeeze-excite gate
(`seBlockFull`). Squeeze-excite is the genuinely-new structure; here it enters as `BatchableOp.seBlock`
(= `batchMap N seBlockFull`).

Like `ResNet34RenderPC`, faithfulness is **per-block** (`*GraphB_faithful`: `den (block graph) = block
forward (den input)`), then chained — so the kernel never reduces the whole net at once. 3-axiom clean.
(The full 16-MBConv `[t,c,n,s,k]` enumeration is mechanical repetition of these block abbreviations;
the structural + batched-infra content — the batched graph, true batch-norm, SE — is here. The
structured render is Item B; the SE/BN cotangent chain is Item D.)
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Batched stage abbreviations (ℝ-forward), all at the batched index `N·(c·h·w)`.
--   Each is `batchMap N` of a proven per-example op (+ true batch-norm + pointwise swish).
-- ════════════════════════════════════════════════════════════════

/-- Batched conv → bn → swish (1×1 expand / generic stride-1 conv). -/
@[reducible] noncomputable def cbsB (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  swish (N * (oc * h * w)) ∘ StableHLO.bnBatchLA N oc h w ε γ β ∘ StableHLO.batchMap N (flatConv W b)

/-- Batched strided (3×3 s2) stem conv → bn → swish (halves spatial). -/
noncomputable def stemB (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  swish (N * (oc * h * w)) ∘ StableHLO.bnBatchLA N oc h w ε γ β ∘
    StableHLO.batchMap N (flatConvStride2 W b)

/-- Batched depthwise (stride-1, k×k) → bn → swish. -/
@[reducible] noncomputable def dwbsB (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  swish (N * (c * h * w)) ∘ StableHLO.bnBatchLA N c h w ε γ β ∘ StableHLO.batchMap N (depthwiseFlat W b)

/-- Batched depthwise (stride-2 downsample, k×k) → bn → swish. -/
@[reducible] noncomputable def dwbsSB (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) :
    Vec (N * (c * (2 * h) * (2 * w))) → Vec (N * (c * h * w)) :=
  swish (N * (c * h * w)) ∘ StableHLO.bnBatchLA N c h w ε γ β ∘
    StableHLO.batchMap N (depthwiseStride2Flat W b)

/-- Batched squeeze-excite block `x ⊙ gate(x)` (the proven `seBlockFull`, per example). -/
@[reducible] noncomputable def seB (N : Nat) {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  StableHLO.batchMap N (seBlockFull (h := h) (w := w) W₁ b₁ W₂ b₂)

/-- Batched project: 1×1 conv → bn (no swish — the linear bottleneck). -/
@[reducible] noncomputable def projB (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  StableHLO.bnBatchLA N oc h w ε γ β ∘ StableHLO.batchMap N (flatConv W b)

-- ════════════════════════════════════════════════════════════════
-- § Block ℝ-forwards: MBConv (no-expand / strided / residual) + head, all batched.
-- ════════════════════════════════════════════════════════════════

/-- MBConv1 (`t=1`, no expand): depthwise-bn-swish → SE → project-bn. No residual (`ic ≠ oc`). -/
noncomputable def mbNoExpFwdB (N : Nat) {ic oc h w kHd kWd r : Nat}
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (εd : ℝ) (γd βd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  projB N (h := h) (w := w) Wp bp εp γp βp ∘
    seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ ∘
    dwbsB N (h := h) (w := w) Wd bd εd γd βd

/-- MBConv6 with a stride-2 downsample: expand-bn-swish (at `2h×2w`) → strided depthwise-bn-swish
    → SE → project-bn. No residual (spatial changes). -/
noncomputable def mbStridedFwdB (N : Nat) {ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  projB N (h := h) (w := w) Wp bp εp γp βp ∘
    seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ ∘
    dwbsSB N (h := h) (w := w) Wd bd εd γd βd ∘
    cbsB N (h := 2 * h) (w := 2 * w) We be εe γe βe

/-- MBConv6 with an identity residual skip (`s=1 ∧ ic=oc=c`): `x + (project ∘ SE ∘ depthwise ∘ expand)(x)`. -/
noncomputable def mbResidFwdB (N : Nat) {c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (γp βp : Vec c) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  residual (projB N (h := h) (w := w) Wp bp εp γp βp ∘
    seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ ∘
    dwbsB N (h := h) (w := w) Wd bd εd γd βd ∘
    cbsB N (h := h) (w := w) We be εe γe βe)

/-- Head: 1×1 conv-bn-swish → global-avg-pool → dense classifier, all batched. -/
noncomputable def headFwdB (N : Nat) {c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC) :
    Vec (N * (c * h * w)) → Vec (N * nC) :=
  StableHLO.batchMap N (dense Wfc bfc) ∘ StableHLO.batchMap N (globalAvgPoolFlat oc h w) ∘
    cbsB N (h := h) (w := w) Wh bh εh γh βh

-- ════════════════════════════════════════════════════════════════
-- § The representative batched EfficientNet-B0 ℝ-forward (true batch-norm + SE)
--   stem(3×3 s2) → MBConv1(no-exp,SE) → MBConv6(exp,s2,3×3,SE) → MBConv6(exp,5×5,SE,+residual)
--   → head(1×1,bn,swish,GAP,dense).  Channels 3→32→16→24(→24)→1280→10; spatial 224→112→56.
-- ════════════════════════════════════════════════════════════════

noncomputable def efficientnetForwardB
    (N : Nat)
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (εs : ℝ) (γs βs : Vec 32)
    (Wd1 : DepthwiseKernel 32 3 3) (bd1 : Vec 32) (εd1 : ℝ) (γd1 βd1 : Vec 32)
    (Wz1a : Mat 32 8) (bz1a : Vec 8) (Wz1b : Mat 8 32) (bz1b : Vec 32)
    (Wp1 : Kernel4 16 32 1 1) (bp1 : Vec 16) (εp1 : ℝ) (γp1 βp1 : Vec 16)
    (We2 : Kernel4 96 16 1 1) (be2 : Vec 96) (εe2 : ℝ) (γe2 βe2 : Vec 96)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 : ℝ) (γd2 βd2 : Vec 96)
    (Wz2a : Mat 96 4) (bz2a : Vec 4) (Wz2b : Mat 4 96) (bz2b : Vec 96)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 : ℝ) (γp2 βp2 : Vec 24)
    (We3 : Kernel4 144 24 1 1) (be3 : Vec 144) (εe3 : ℝ) (γe3 βe3 : Vec 144)
    (Wd3 : DepthwiseKernel 144 5 5) (bd3 : Vec 144) (εd3 : ℝ) (γd3 βd3 : Vec 144)
    (Wz3a : Mat 144 6) (bz3a : Vec 6) (Wz3b : Mat 6 144) (bz3b : Vec 144)
    (Wp3 : Kernel4 24 144 1 1) (bp3 : Vec 24) (εp3 : ℝ) (γp3 βp3 : Vec 24)
    (Wh : Kernel4 1280 24 1 1) (bh : Vec 1280) (εh : ℝ) (γh βh : Vec 1280)
    (Wfc : Mat 1280 10) (bfc : Vec 10)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * 10) :=
  -- nested-application form (NOT `∘`) so the faithfulness proof closes by pure delta — the per-block
  -- `rw` chain produces exactly this term, and no composition needs reducing.
  headFwdB N (h := 56) (w := 56) Wh bh εh γh βh Wfc bfc
    (mbResidFwdB N (h := 56) (w := 56) We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3
        Wz3a bz3a Wz3b bz3b Wp3 bp3 εp3 γp3 βp3
      (mbStridedFwdB N (h := 56) (w := 56) We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2
          Wz2a bz2a Wz2b bz2b Wp2 bp2 εp2 γp2 βp2
        (mbNoExpFwdB N (h := 112) (w := 112) Wd1 bd1 εd1 γd1 βd1 Wz1a bz1a Wz1b bz1b
            Wp1 bp1 εp1 γp1 βp1
          (stemB N (h := 112) (w := 112) Ws bs εs γs βs x))))

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Block `SHlo` graphs (take the input subgraph `e`) + their faithfulness lemmas.
--   Each `*GraphB_faithful` proves `den (block graph e) = block forward (den e)` with the small
--   per-block recipe (`simp` with the batched-token `den` lemmas) — bounded kernel work per block.
-- ════════════════════════════════════════════════════════════════

/-- Stem 3×3-s2 conv → bn → swish, batched. -/
def stemGraphB (epsStr : String) {N ic oc h w : Nat}
    (Ws : Kernel4 oc ic 3 3) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) : SHlo (N * (oc * h * w)) :=
  .swishF (.bnBatchF "%sg" "%sbt" epsStr εs γs βs
    (.batchOp (N := N) (.convStrided (h := h) (w := w) "%sW" "%sb" Ws bs) e))

theorem stemGraphB_faithful (epsStr : String) {N ic oc h w : Nat}
    (Ws : Kernel4 oc ic 3 3) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (stemGraphB epsStr Ws bs εs γs βs e) = stemB N (h := h) (w := w) Ws bs εs γs βs (den e) := by
  unfold stemGraphB stemB
  simp only [den_batchOp_convStrided, den_bnBatchF, swishF_faithful, Function.comp_apply]

/-- MBConv1 (no expand): dw-bn-swish → SE → project-bn, batched. -/
def mbNoExpGraphB (p epsStr : String) {N ic oc h w kHd kWd r : Nat}
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (εd : ℝ) (γd βd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (e : SHlo (N * (ic * h * w))) : SHlo (N * (oc * h * w)) :=
  .bnBatchF s!"%{p}pg" s!"%{p}pbt" epsStr εp γp βp
    (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" s!"%{p}pb" Wp bp)
      (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zWa" s!"%{p}zba" s!"%{p}zWb" s!"%{p}zbb"
          Wz₁ bz₁ Wz₂ bz₂)
        (.swishF (.bnBatchF s!"%{p}dg" s!"%{p}dbt" epsStr εd γd βd
          (.batchOp (N := N) (.depthwise (h := h) (w := w) s!"%{p}dW" s!"%{p}db" Wd bd) e)))))

theorem mbNoExpGraphB_faithful (p epsStr : String) {N ic oc h w kHd kWd r : Nat}
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (εd : ℝ) (γd βd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (e : SHlo (N * (ic * h * w))) :
    den (mbNoExpGraphB p epsStr Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp e)
      = mbNoExpFwdB N (h := h) (w := w) Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp (den e) := by
  unfold mbNoExpGraphB mbNoExpFwdB projB seB dwbsB
  simp only [den_batchOp_conv, den_batchOp_seBlock, den_batchOp_depthwise, den_bnBatchF,
             swishF_faithful, Function.comp_apply]

/-- MBConv6 strided: expand-bn-swish (at `2h×2w`) → strided dw-bn-swish → SE → project-bn, batched. -/
def mbStridedGraphB (p epsStr : String) {N ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) : SHlo (N * (oc * h * w)) :=
  .bnBatchF s!"%{p}pg" s!"%{p}pbt" epsStr εp γp βp
    (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" s!"%{p}pb" Wp bp)
      (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zWa" s!"%{p}zba" s!"%{p}zWb" s!"%{p}zbb"
          Wz₁ bz₁ Wz₂ bz₂)
        (.swishF (.bnBatchF s!"%{p}dg" s!"%{p}dbt" epsStr εd γd βd
          (.batchOp (N := N) (.depthwiseStrided (h := h) (w := w) s!"%{p}dW" s!"%{p}db" Wd bd)
            (.swishF (.bnBatchF s!"%{p}eg" s!"%{p}ebt" epsStr εe γe βe
              (.batchOp (N := N) (.conv (h := 2 * h) (w := 2 * w) s!"%{p}eW" s!"%{p}eb" We be) e))))))))

theorem mbStridedGraphB_faithful (p epsStr : String) {N ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (mbStridedGraphB p epsStr We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp e)
      = mbStridedFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂
          Wp bp εp γp βp (den e) := by
  unfold mbStridedGraphB mbStridedFwdB projB seB dwbsSB cbsB
  simp only [den_batchOp_conv, den_batchOp_seBlock, den_batchOp_depthwiseStrided, den_bnBatchF,
             swishF_faithful, Function.comp_apply]

/-- MBConv6 with identity residual: `addV body skip`, body = project ∘ SE ∘ dw ∘ expand, batched. -/
def mbResidGraphB (p epsStr : String) {N c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (γp βp : Vec c)
    (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  .addV
    (.bnBatchF s!"%{p}pg" s!"%{p}pbt" epsStr εp γp βp
      (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" s!"%{p}pb" Wp bp)
        (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zWa" s!"%{p}zba" s!"%{p}zWb" s!"%{p}zbb"
            Wz₁ bz₁ Wz₂ bz₂)
          (.swishF (.bnBatchF s!"%{p}dg" s!"%{p}dbt" epsStr εd γd βd
            (.batchOp (N := N) (.depthwise (h := h) (w := w) s!"%{p}dW" s!"%{p}db" Wd bd)
              (.swishF (.bnBatchF s!"%{p}eg" s!"%{p}ebt" epsStr εe γe βe
                (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}eW" s!"%{p}eb" We be) e))))))))) e

theorem mbResidGraphB_faithful (p epsStr : String) {N c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (γp βp : Vec c)
    (e : SHlo (N * (c * h * w))) :
    den (mbResidGraphB p epsStr We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp e)
      = mbResidFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂
          Wp bp εp γp βp (den e) := by
  unfold mbResidGraphB mbResidFwdB projB seB dwbsB cbsB residual biPath
  simp only [den_batchOp_conv, den_batchOp_seBlock, den_batchOp_depthwise, den_bnBatchF,
             swishF_faithful, den_addV, Function.comp_apply]

/-- Head: 1×1 conv-bn-swish → GAP → dense, batched. -/
def headGraphB (epsStr : String) {N c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC)
    (e : SHlo (N * (c * h * w))) : SHlo (N * nC) :=
  .batchOp (N := N) (.dense "%Wfc" "%bfc" Wfc bfc)
    (.batchOp (N := N) (.gap (c := oc) (h := h) (w := w))
      (.swishF (.bnBatchF "%hg" "%hbt" epsStr εh γh βh
        (.batchOp (N := N) (.conv (h := h) (w := w) "%hW" "%hb" Wh bh) e))))

theorem headGraphB_faithful (epsStr : String) {N c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC)
    (e : SHlo (N * (c * h * w))) :
    den (headGraphB epsStr Wh bh εh γh βh Wfc bfc e)
      = headFwdB N (h := h) (w := w) Wh bh εh γh βh Wfc bfc (den e) := by
  unfold headGraphB headFwdB cbsB
  simp only [den_batchOp_dense, den_batchOp_gap, den_batchOp_conv, den_bnBatchF, swishF_faithful,
             Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The full batched graph + faithfulness (chaining the per-block lemmas)
-- ════════════════════════════════════════════════════════════════

/-- The representative **batched EfficientNet-B0 forward** graph at the batched index `N·(c·h·w)`:
    stem → MBConv1(no-exp) → MBConv6(strided 3×3) → MBConv6(5×5, residual) → head. Every spatial op
    is `batchOp`; **true batch-norm** is `bnBatchF`; pointwise swish is `swishF`; the residual is
    `addV`. Built by composing the per-block graphs; denotes `efficientnetForwardB`. -/
def efficientnetFwdGraphB
    (N : Nat) (epsStr : String)
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (εs : ℝ) (γs βs : Vec 32)
    (Wd1 : DepthwiseKernel 32 3 3) (bd1 : Vec 32) (εd1 : ℝ) (γd1 βd1 : Vec 32)
    (Wz1a : Mat 32 8) (bz1a : Vec 8) (Wz1b : Mat 8 32) (bz1b : Vec 32)
    (Wp1 : Kernel4 16 32 1 1) (bp1 : Vec 16) (εp1 : ℝ) (γp1 βp1 : Vec 16)
    (We2 : Kernel4 96 16 1 1) (be2 : Vec 96) (εe2 : ℝ) (γe2 βe2 : Vec 96)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 : ℝ) (γd2 βd2 : Vec 96)
    (Wz2a : Mat 96 4) (bz2a : Vec 4) (Wz2b : Mat 4 96) (bz2b : Vec 96)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 : ℝ) (γp2 βp2 : Vec 24)
    (We3 : Kernel4 144 24 1 1) (be3 : Vec 144) (εe3 : ℝ) (γe3 βe3 : Vec 144)
    (Wd3 : DepthwiseKernel 144 5 5) (bd3 : Vec 144) (εd3 : ℝ) (γd3 βd3 : Vec 144)
    (Wz3a : Mat 144 6) (bz3a : Vec 6) (Wz3b : Mat 6 144) (bz3b : Vec 144)
    (Wp3 : Kernel4 24 144 1 1) (bp3 : Vec 24) (εp3 : ℝ) (γp3 βp3 : Vec 24)
    (Wh : Kernel4 1280 24 1 1) (bh : Vec 1280) (εh : ℝ) (γh βh : Vec 1280)
    (Wfc : Mat 1280 10) (bfc : Vec 10)
    (x : Vec (N * (3 * 224 * 224))) : SHlo (N * 10) :=
  headGraphB epsStr (h := 56) (w := 56) Wh bh εh γh βh Wfc bfc
    (mbResidGraphB "b3" epsStr (h := 56) (w := 56) We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3
        Wz3a bz3a Wz3b bz3b Wp3 bp3 εp3 γp3 βp3
      (mbStridedGraphB "b2" epsStr (h := 56) (w := 56) We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2
          Wz2a bz2a Wz2b bz2b Wp2 bp2 εp2 γp2 βp2
        (mbNoExpGraphB "b1" epsStr (h := 112) (w := 112) Wd1 bd1 εd1 γd1 βd1 Wz1a bz1a Wz1b bz1b
            Wp1 bp1 εp1 γp1 βp1
          (stemGraphB epsStr (h := 112) (w := 112) Ws bs εs γs βs (.operand "%x" x)))))

/-- **Batched EfficientNet-B0 forward faithfulness.** The batched render graph (true batch-norm + SE,
    at index `N·(c·h·w)`) denotes `efficientnetForwardB`. Chained from the per-block `*GraphB_faithful`
    lemmas (each fires as a `simp` rewrite, so the kernel never reduces the whole net at once) — the
    `ResNet34RenderPC` recipe lifted to the batched index. The "text = render of a proven forward graph"
    half for EfficientNet at the render's genuine (batch-coupled) BN flavor. -/
theorem efficientnetFwdGraphB_faithful
    (N : Nat) (epsStr : String)
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (εs : ℝ) (γs βs : Vec 32)
    (Wd1 : DepthwiseKernel 32 3 3) (bd1 : Vec 32) (εd1 : ℝ) (γd1 βd1 : Vec 32)
    (Wz1a : Mat 32 8) (bz1a : Vec 8) (Wz1b : Mat 8 32) (bz1b : Vec 32)
    (Wp1 : Kernel4 16 32 1 1) (bp1 : Vec 16) (εp1 : ℝ) (γp1 βp1 : Vec 16)
    (We2 : Kernel4 96 16 1 1) (be2 : Vec 96) (εe2 : ℝ) (γe2 βe2 : Vec 96)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 : ℝ) (γd2 βd2 : Vec 96)
    (Wz2a : Mat 96 4) (bz2a : Vec 4) (Wz2b : Mat 4 96) (bz2b : Vec 96)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 : ℝ) (γp2 βp2 : Vec 24)
    (We3 : Kernel4 144 24 1 1) (be3 : Vec 144) (εe3 : ℝ) (γe3 βe3 : Vec 144)
    (Wd3 : DepthwiseKernel 144 5 5) (bd3 : Vec 144) (εd3 : ℝ) (γd3 βd3 : Vec 144)
    (Wz3a : Mat 144 6) (bz3a : Vec 6) (Wz3b : Mat 6 144) (bz3b : Vec 144)
    (Wp3 : Kernel4 24 144 1 1) (bp3 : Vec 24) (εp3 : ℝ) (γp3 βp3 : Vec 24)
    (Wh : Kernel4 1280 24 1 1) (bh : Vec 1280) (εh : ℝ) (γh βh : Vec 1280)
    (Wfc : Mat 1280 10) (bfc : Vec 10)
    (x : Vec (N * (3 * 224 * 224))) :
    den (efficientnetFwdGraphB N epsStr Ws bs εs γs βs Wd1 bd1 εd1 γd1 βd1 Wz1a bz1a Wz1b bz1b
          Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wz2a bz2a Wz2b bz2b
          Wp2 bp2 εp2 γp2 βp2 We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wz3a bz3a Wz3b bz3b
          Wp3 bp3 εp3 γp3 βp3 Wh bh εh γh βh Wfc bfc x)
      = efficientnetForwardB N Ws bs εs γs βs Wd1 bd1 εd1 γd1 βd1 Wz1a bz1a Wz1b bz1b
          Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wz2a bz2a Wz2b bz2b
          Wp2 bp2 εp2 γp2 βp2 We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wz3a bz3a Wz3b bz3b
          Wp3 bp3 εp3 γp3 βp3 Wh bh εh γh βh Wfc bfc x := by
  rw [efficientnetFwdGraphB, headGraphB_faithful, mbResidGraphB_faithful,
      mbStridedGraphB_faithful, mbNoExpGraphB_faithful, stemGraphB_faithful, den_operand]
  rfl

end StableHLO
end Proofs
