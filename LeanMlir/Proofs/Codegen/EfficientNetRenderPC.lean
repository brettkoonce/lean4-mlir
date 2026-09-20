import LeanMlir.Proofs.Codegen.StableHLO

/-! # The BATCHED EfficientNet-B0 block forwards and graphs (true batch-norm, matches the render)

The EfficientNet peer of `MobileNetV2RenderPC.lean` / the retired `ResNet34RenderPC.lean` — but EfficientNet's
operational render ([`tests/TestEfficientNetFwd.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/TestEfficientNetFwd.lean)) emits **true batch-norm** (reduce μ/var over the
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

We prove the FORWARD half — `den (graph) = forward` — for every block form B0 has: the stride-2
stem conv-bn-swish, an MBConv1 (`t=1`, **no expand**) SE block, an MBConv6 expand SE block with a
**stride-2** downsample, an MBConv6 expand SE block with an **identity residual** skip, and the 1×1
conv-bn-swish head, GAP and the dense classifier — all with **true batch-norm** and the
squeeze-excite gate (`seBlockFull`, entering as `BatchableOp.seBlock` = `batchMap N seBlockFull`).
Faithfulness is **per-block** (`*GraphB_faithful`: `den (block graph) = block forward (den input)`),
so `EfficientNetFullB0` chains the sixteen-block net without the kernel reducing it at once.
3-axiom clean.
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

/-- Batched strided (3×3 s2) stem conv → bn → swish (halves spatial). ⚠ At the XLA-`SAME` phase
    (`flatConvStride2Xla` = `decimateOddFlat ∘ flatConv`): the TF-origin B0 pads its stem `(0,1)`,
    and the shipped render has emitted `convStridedXla` there since 2026-08-08. The symmetric
    `flatConvStride2` has the same type and output shape; nothing structural would notice the
    wrong one (re-spelled 2026-09-05, `planning/archive/xla_same_respell_and_blueprint_audit.md`). -/
noncomputable def stemB (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  swish (N * (oc * h * w)) ∘ StableHLO.bnBatchLA N oc h w ε γ β ∘
    StableHLO.batchMap N (flatConvStride2Xla W b)

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
    (.batchOp (N := N) (.convStridedXla (h := h) (w := w) "%sW" "%sb" Ws bs) e))

theorem stemGraphB_faithful (epsStr : String) {N ic oc h w : Nat}
    (Ws : Kernel4 oc ic 3 3) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (stemGraphB epsStr Ws bs εs γs βs e) = stemB N (h := h) (w := w) Ws bs εs γs βs (den e) := by
  unfold stemGraphB stemB
  simp only [den_batchOp, denOp, den_bnBatchF, swishF_faithful, Function.comp_apply]

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
  simp only [den_batchOp, denOp, den_bnBatchF,
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
  simp only [den_batchOp, denOp, den_bnBatchF,
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
  simp only [den_batchOp, denOp, den_bnBatchF,
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
  simp only [den_batchOp, denOp, den_bnBatchF, swishF_faithful,
             Function.comp_apply]

end StableHLO
end Proofs
