import LeanMlir.Proofs.Foundation.BatchedStages
import LeanMlir.Proofs.Codegen.StableHLO

/-! # The BATCHED EfficientNet-B0 block forwards and graphs (true batch-norm, matches the render)

The EfficientNet peer of `MobileNetV2StagesPC`. EfficientNet's render (`EfficientNetRender`) emits
**true batch-norm** (reduce μ/var over the batch+spatial axes `[0,2,3]` per channel —
`bnBatchTensor4`), which **couples the batch**. So the forward graph here lives at the **batched
index**
`N·(c·h·w)` (`StableHLO.batchOp`/`StableHLO.bnBatchF`, `StableHLO.lean`):

* every batch-separable op (conv / strided conv / depthwise / strided depthwise / dense / GAP / the
  whole SE block) is `batchMap N` of the proven per-example op (`SHlo.batchOp` + `BatchableOp`/`denOp`);
* swish is the batched pointwise descriptor (`.batchOp .swish`, denoting the flat `swishF` by
  `den_batchOp_swish_eq_swishF`) and the residual add is `.addVB` — the tokens the render emits, so
  `pretty` of each block graph is the rendered block's text (`Codegen/FwdGraphTextTies`), names
  included: SE denses `zW1/zb1/zW2/zb2`, conv biases `biasName false`, classifier `%Wd`/`%bd`;
* the one batch-coupled op, true batch-norm, is `SHlo.bnBatchF`, denoting `bnBatchLA` (= the proven
  `bnBatchTensor4`, reindexed to the network's left-assoc `N·(oc·h·w)` flat layout).

This file proves the FORWARD half — `den (graph) = forward` — for the stride-2 stem
conv-bn-swish (`stemGraphB_faithful`), an MBConv1 (`t=1`, **no expand**) SE block
(`mbNoExpGraphB_faithful`), an MBConv6 expand SE block with a **stride-2** downsample
(`mbStridedGraphB_faithful`), an MBConv6 expand SE block with an **identity residual** skip
(`mbResidGraphB_faithful`), and the 1×1 conv-bn-swish head, GAP and the dense classifier
(`headGraphB_faithful`) — all with **true batch-norm** and the squeeze-excite gate (`seBlockFull`,
entering as `BatchableOp.seBlock` = `batchMap N seBlockFull`). B0's fifth block form, the
stride-1 expand block with no skip, is `mbExpGraphB_faithful` in `EfficientNetFullB0`.
Faithfulness is **per-block** (`*GraphB_faithful`: `den (block graph) = block forward (den input)`),
so `EfficientNetFullB0` chains the sixteen-block net without the kernel reducing it at once.
All five theorems are in [`tests/AuditAxioms.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/AuditAxioms.lean)'s `#print axioms` list.
-/

namespace Proofs

open scoped BigOperators

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
  .batchOp (N := N) .swish (.bnBatchF "%sg" "%sbt" epsStr εs γs βs
    (.batchOp (N := N) (.convStridedXla (h := h) (w := w) "%sW" (biasName false "" oc) Ws bs) e))

theorem stemGraphB_faithful (epsStr : String) {N ic oc h w : Nat}
    (Ws : Kernel4 oc ic 3 3) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (stemGraphB epsStr Ws bs εs γs βs e) = stemB N (h := h) (w := w) Ws bs εs γs βs (den e) := by
  unfold stemGraphB stemB
  simp only [den_batchOp, denOp, den_bnBatchF, ↓den_batchOp_swish_eq_swishF, swishF_faithful, Function.comp_apply]

/-- MBConv1 (no expand): dw-bn-swish → SE → project-bn, batched. -/
def mbNoExpGraphB (p epsStr : String) {N ic oc h w kHd kWd r : Nat}
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (εd : ℝ) (γd βd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (e : SHlo (N * (ic * h * w))) : SHlo (N * (oc * h * w)) :=
  .bnBatchF s!"%{p}pg" s!"%{p}pbt" epsStr εp γp βp
    (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" (biasName false "" oc) Wp bp)
      (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zW1" s!"%{p}zb1" s!"%{p}zW2" s!"%{p}zb2"
          Wz₁ bz₁ Wz₂ bz₂)
        (.batchOp (N := N) .swish (.bnBatchF s!"%{p}dg" s!"%{p}dbt" epsStr εd γd βd
          (.batchOp (N := N) (.depthwise (h := h) (w := w) s!"%{p}dW" (biasName false "" ic) Wd bd) e)))))

theorem mbNoExpGraphB_faithful (p epsStr : String) {N ic oc h w kHd kWd r : Nat}
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (εd : ℝ) (γd βd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (e : SHlo (N * (ic * h * w))) :
    den (mbNoExpGraphB p epsStr Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp e)
      = mbNoExpFwdB N (h := h) (w := w) Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp (den e) := by
  unfold mbNoExpGraphB mbNoExpFwdB projB seB dwbsB
  simp only [den_batchOp, denOp, den_bnBatchF,
             ↓den_batchOp_swish_eq_swishF, swishF_faithful, Function.comp_apply]

/-- MBConv6 strided: expand-bn-swish (at `2h×2w`) → strided dw-bn-swish → SE → project-bn, batched. -/
def mbStridedGraphB (p epsStr : String) {N ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) : SHlo (N * (oc * h * w)) :=
  .bnBatchF s!"%{p}pg" s!"%{p}pbt" epsStr εp γp βp
    (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" (biasName false "" oc) Wp bp)
      (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zW1" s!"%{p}zb1" s!"%{p}zW2" s!"%{p}zb2"
          Wz₁ bz₁ Wz₂ bz₂)
        (.batchOp (N := N) .swish (.bnBatchF s!"%{p}dg" s!"%{p}dbt" epsStr εd γd βd
          (.batchOp (N := N) (.depthwiseStrided (h := h) (w := w) s!"%{p}dW" (biasName false "" mid) Wd bd)
            (.batchOp (N := N) .swish (.bnBatchF s!"%{p}eg" s!"%{p}ebt" epsStr εe γe βe
              (.batchOp (N := N) (.conv (h := 2 * h) (w := 2 * w) s!"%{p}eW" (biasName false "" mid) We be) e))))))))

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
             ↓den_batchOp_swish_eq_swishF, swishF_faithful, Function.comp_apply]

/-- MBConv6 with identity residual: `addVB body skip`, body = project ∘ SE ∘ dw ∘ expand, batched. -/
def mbResidGraphB (p epsStr : String) {N c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (γp βp : Vec c)
    (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  .addVB
    (.bnBatchF s!"%{p}pg" s!"%{p}pbt" epsStr εp γp βp
      (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" (biasName false "" c) Wp bp)
        (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zW1" s!"%{p}zb1" s!"%{p}zW2" s!"%{p}zb2"
            Wz₁ bz₁ Wz₂ bz₂)
          (.batchOp (N := N) .swish (.bnBatchF s!"%{p}dg" s!"%{p}dbt" epsStr εd γd βd
            (.batchOp (N := N) (.depthwise (h := h) (w := w) s!"%{p}dW" (biasName false "" mid) Wd bd)
              (.batchOp (N := N) .swish (.bnBatchF s!"%{p}eg" s!"%{p}ebt" epsStr εe γe βe
                (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}eW" (biasName false "" mid) We be) e))))))))) e

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
             ↓den_batchOp_swish_eq_swishF, swishF_faithful, den_addVB, Function.comp_apply]

/-- Head: 1×1 conv-bn-swish → GAP → dense, batched. -/
def headGraphB (epsStr : String) {N c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC)
    (e : SHlo (N * (c * h * w))) : SHlo (N * nC) :=
  .batchOp (N := N) (.dense "%Wd" "%bd" Wfc bfc)
    (.batchOp (N := N) (.gap (c := oc) (h := h) (w := w))
      (.batchOp (N := N) .swish (.bnBatchF "%hg" "%hbt" epsStr εh γh βh
        (.batchOp (N := N) (.conv (h := h) (w := w) "%hW" (biasName false "" oc) Wh bh) e))))

theorem headGraphB_faithful (epsStr : String) {N c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC)
    (e : SHlo (N * (c * h * w))) :
    den (headGraphB epsStr Wh bh εh γh βh Wfc bfc e)
      = headFwdB N (h := h) (w := w) Wh bh εh γh βh Wfc bfc (den e) := by
  unfold headGraphB headFwdB cbsB
  simp only [den_batchOp, denOp, den_bnBatchF, ↓den_batchOp_swish_eq_swishF, swishF_faithful,
             Function.comp_apply]

end StableHLO
end Proofs
