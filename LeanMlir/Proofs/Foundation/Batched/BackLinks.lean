import LeanMlir.Proofs.Foundation.Batched.Stages
import LeanMlir.Proofs.Codegen.StableHLO.Basic

/-! # Batched backward links — the backward graphs and cotangent steps at the batched index

The vocabulary every batched whole-net backward proof and every step tie is written in, at the flat
batched index `N·(c·h·w)`. Each graph or cotangent here denotes the `.backward` of a proven VJP
(`Batched.Stages`), so a chain built from them is the loss-driven backward, not a free cotangent.

| what | names | namespace |
|---|---|---|
| residual fan-in backward graph | `residualBackGraph` / `_faithful` | `StableHLO` |
| batched op backwards: true BN, conv, strided conv, depthwise (stride 1 / symmetric / XLA-`SAME` stride 2), SE | `bnBatchBack_faithful`, `bnBatchLABack_faithful`, `convBackBatched_faithful`, … `seBackBatched_faithful` | `StableHLO` |
| stage backward graphs | `cbsBackBatchedGraph`, `dwbsBackBatchedGraph`, `dwbsSBackBatchedGraph`, `projBackBatchedGraph` (+ `_faithful`) | `StableHLO` |
| cotangent steps: BN, swish, sigmoid, conv / depthwise input-VJPs, GAP, SE, SE gate; the `c·h·w ↔ c·(h·w)` reindex | `bnBackB`, `swBackB`, `sigBackB`, `cInB`, `dInB`, `dStridedInB`, `gapInB`, `seInB`, `gateCotB`, `reassocB` | `BackLinks` |
| cotangent steps: relu mask, strided conv input-VJP, BN as emitted, 3×3/s2 max-pool; the one-row head casts | `reluMaskB`, `cStridedInB`, `bnInB`, `mpInB`, `rowB` / `unrowB` | `BackLinks` |

Every conv net's tie cites these names.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.StableHLO

open scoped BigOperators


/-- Backward graph for a residual block `x ↦ x + f x`, given a subgraph
    `fBack` that renders the body `f`'s input-cotangent. The identity skip
    contributes the cotangent verbatim (`%dy`); `addV` sums the two paths.
    This is the renderable image of `residualHasVJP`'s `biPath` backward. -/
def residualBackGraph {n : Nat} (fBack ecot : SHlo n) : SHlo n :=
  .addV fBack ecot

/-- **Residual additive-fan-in backward faithfulness (general).**
    If `fBack` denotes the body's VJP backward (`den fBack = hf.backward x dy`),
    then the residual backward graph denotes the proven `residualHasVJP`
    backward, which is `f.backward x dy + dy`. The proof is structural — the
    only definitional facts are `den (addV a b) = den a + den b` and the
    identity skip's `backward = dy` — so it composes without a whole-net
    terminal `rfl`. -/
theorem residualBackGraph_faithful {n : Nat}
    (f : Vec n → Vec n) (hf_diff : Differentiable ℝ f) (hf : HasVJP f)
    (x : Vec n) (ecot fBack : SHlo n)
    (hfb : den fBack = hf.backward x (den ecot)) :
    den (residualBackGraph fBack ecot)
      = (residualHasVJP f hf_diff hf).backward x (den ecot) := by
  funext i
  have hsum : den (residualBackGraph fBack ecot) i = den fBack i + den ecot i := rfl
  rw [hsum, hfb]
  -- RHS = `biPathHasVJP f id`'s backward = `f.backward x dy i + id.backward x dy i`
  -- with `id.backward x dy = dy`; defeq but needs full-transparency unfolding.
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Batched lifting (start): true batch-norm backward primitive
-- ════════════════════════════════════════════════════════════════

/-- **The renderable batch-norm input-grad IS the certified backward** — both equal the
    `pdiv`-contracted Jacobian (`bnBatchTensor4GradInput_correct`,
    `bnBatchTensor4HasVJP_correct`). -/
theorem bnBatchTensor4GradInput_eq_backward (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε)
    (γ β : Vec oc) (x dy : Vec (N * (oc * (h * w)))) :
    bnBatchTensor4GradInput N oc h w ε γ x dy
      = (bnBatchTensor4HasVJP N oc h w ε hε γ β).backward x dy :=
  funext fun i => by
    rw [bnBatchTensor4GradInput_correct N oc h w ε hε γ β,
      ← bnBatchTensor4HasVJP_correct N oc h w ε hε γ β]

/-- **`bnBatchBack` (true batch-norm backward) faithfulness.**
    `bnBatchBack` denotes the proven
    `bnBatchTensor4` VJP backward (batch-COUPLED batch-norm on `[N,C,H,W]`,
    reduce over `[0,2,3]` per channel) via the renderable three-term
    `bnBatchTensor4GradInput`. This is the genuinely-new op the batched MBConv
    stages need (their bn is `bnBatchLA`, not a per-example `batchMap`); the
    other batched stages (conv/depthwise/SE) are `batchMap` of the per-example
    backwards already proven above. The network-layout wrapper is
    `bnBatchLABack_faithful`. -/
theorem bnBatchBack_faithful {N oc h w : Nat} (gN xN es : String)
    (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (x : Vec (N * (oc * (h * w)))) (e : SHlo (N * (oc * (h * w)))) :
    den (SHlo.bnBatchBack gN xN es ε γ x e)
      = (bnBatchTensor4HasVJP N oc h w ε hε γ β).backward x (den e) := by
  -- `den (.bnBatchBack …)` is the grad-input at `den e` by definition (its `den` arm).
  show bnBatchTensor4GradInput N oc h w ε γ x (den e) = _
  exact bnBatchTensor4GradInput_eq_backward N oc h w ε hε γ β x (den e)

/-- **Batched conv input-VJP faithfulness.** `convBackBatched` denotes the proven
    VJP of the batched conv `batchMap N (flatConv W b)` — i.e. the per-example
    conv input-grad applied independently across the batch. Conv is linear, so
    its backward ignores the forward activation; the batched backward is a plain
    `batchMap` of the per-example backward, matching `batchMapHasVJP`. With
    `bnBatchBack` and the SE and depthwise peers, one of the batched MBConv's
    per-stage backward pieces. -/
theorem convBackBatched_faithful {N ic oc h w kH kW : Nat} (wN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (v : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) :
    den (SHlo.convBackBatched (N := N) wN W b e)
      = (batchMapHasVJP (flatConv W b) (flatConvHasVJP W b)
          (flatConv_differentiable W b)).backward v (den e) := by
  funext idx
  -- The transport in `batchMapHasVJP` unfolds to the per-example conv backward
  -- on each row; both sides then differ only in the (discarded) forward
  -- activation arg, since conv is linear (`conv2dHasVJP3.backward` ignores it).
  simp only [denStepApp, batchMap, batchMapHasVJP, flatConvHasVJP, HasVJPMat.toHasVJP,
    rowwiseHasVJPMat, HasVJP3.toHasVJP, conv2dHasVJP3]
  rfl

/-- **Batched STRIDE-2 conv input-VJP faithfulness.** The stride-2 analogue of
    `convBackBatched_faithful`: `convStridedBackBatched` denotes the proven VJP of
    the batched strided conv `batchMap N (flatConvStride2 W b)` — i.e. the
    per-example strided-conv input-grad (`flatConvStride2HasVJP` = zero-upsample
    the cotangent then the reversed-kernel conv) applied independently across the
    batch. Strided conv (`decimate ∘ conv`) is linear, so its backward ignores the
    forward activation; the batched backward is a plain `batchMap` of the
    per-example backward, matching `batchMapHasVJP`. The downsample basic-block's
    stride-2 conv1 backward brick. -/
theorem convStridedBackBatched_faithful {N ic oc h w kH kW : Nat} (wN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    den (SHlo.convStridedBackBatched (N := N) wN W b e)
      = (batchMapHasVJP (flatConvStride2 W b) (flatConvStride2HasVJP W b)
          (flatConvStride2_differentiable W b)).backward v (den e) := by
  funext idx
  -- The transport in `batchMapHasVJP` unfolds to the per-example strided-conv
  -- backward on each row; both sides then differ only in the (discarded) forward
  -- activation arg, since strided conv is linear (its backward ignores it).
  simp only [denStepApp, batchMap, batchMapHasVJP, HasVJPMat.toHasVJP, rowwiseHasVJPMat]
  rfl

/-- **Batched STRIDE-2 depthwise input-VJP faithfulness.** The stride-2 analogue
    of `depthwiseBackBatched_faithful` (and the depthwise analogue of
    `convStridedBackBatched_faithful`): `depthwiseStridedBackBatched` denotes the
    proven VJP of the batched strided depthwise `batchMap N (depthwiseStride2Flat W b)`
    — i.e. the per-example strided-depthwise input-grad (`depthwiseStride2FlatHasVJP`
    = zero-upsample the cotangent then the reversed-kernel per-channel depthwise)
    applied independently across the batch. Strided depthwise (`decimate ∘ depthwise`)
    is linear, so its backward ignores the forward activation; the batched backward
    is a plain `batchMap` of the per-example backward, matching `batchMapHasVJP`.
    The EfficientNet downsample MBConv's stride-2 depthwise backward brick. -/
theorem depthwiseStridedBackBatched_faithful {N c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (v : Vec (N * (c * (2 * h) * (2 * w)))) (e : SHlo (N * (c * h * w))) :
    den (SHlo.depthwiseStridedBackBatched (N := N) wN W b e)
      = (batchMapHasVJP (depthwiseStride2Flat W b) (depthwiseStride2FlatHasVJP W b)
          (depthwiseStride2Flat_differentiable W b)).backward v (den e) := by
  funext idx
  -- The transport in `batchMapHasVJP` unfolds to the per-example strided-depthwise
  -- backward on each row; both sides then differ only in the (discarded) forward
  -- activation arg, since strided depthwise is linear (its backward ignores it).
  simp only [denStepApp, batchMap, batchMapHasVJP, HasVJPMat.toHasVJP, rowwiseHasVJPMat]
  rfl

/-- **Batched XLA-`SAME` STRIDE-2 depthwise input-VJP faithfulness.** The odd-phase peer of
    `depthwiseStridedBackBatched_faithful`: `depthwiseStridedXlaBackBatched` (pad `[p+1, p-1]`,
    the token MobileNetV2's Adam render emits at its four strided depthwises) denotes the proven
    VJP of `batchMap N (depthwiseStride2FlatXla W b)`. Same proof: a scatter onto the odd
    positions is as linear as one onto the even ones. -/
theorem depthwiseStridedXlaBackBatched_faithful {N c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (v : Vec (N * (c * (2 * h) * (2 * w)))) (e : SHlo (N * (c * h * w))) :
    den (SHlo.depthwiseStridedXlaBackBatched (N := N) wN W b e)
      = (batchMapHasVJP (depthwiseStride2FlatXla W b) (depthwiseStride2FlatXlaHasVJP W b)
          (depthwiseStride2FlatXla_differentiable W b)).backward v (den e) := by
  funext idx
  simp only [denStepApp, batchMap, batchMapHasVJP, HasVJPMat.toHasVJP, rowwiseHasVJPMat]
  rfl

/-- **Batched depthwise input-VJP faithfulness.** The depthwise analogue of
    `convBackBatched_faithful`: `depthwiseBackBatched` denotes the proven VJP of
    the batched depthwise `batchMap N (depthwiseFlat W b)`. Depthwise conv is
    linear, so its backward is activation-independent and the batched backward is
    a plain `batchMap` of the per-example backward. The MBConv depthwise stage's
    batch-separable backward brick. -/
theorem depthwiseBackBatched_faithful {N c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (v : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) :
    den (SHlo.depthwiseBackBatched (N := N) wN W b e)
      = (batchMapHasVJP (depthwiseFlat W b) (depthwiseFlatHasVJP W b)
          (depthwiseFlat_differentiable W b)).backward v (den e) := by
  funext idx
  simp only [denStepApp, batchMap, batchMapHasVJP, depthwiseFlatHasVJP, HasVJPMat.toHasVJP,
    rowwiseHasVJPMat, HasVJP3.toHasVJP, depthwiseHasVJP3]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § bn-layout wrapper: true-batch-norm backward on the NETWORK layout
-- ════════════════════════════════════════════════════════════════

/-- **`bnBatchLA` backward = reindex-conjugated `bnBatchTensor4` backward.**
    The network indexes at `N·(oc·h·w)` (left-assoc) but the proven true-BN
    `bnBatchTensor4` lives at `N·(oc·(h·w))`; `bnBatchLA` bridges by conjugating
    with the associativity-cast reindexes (`bnBatchLA_eq_comp`). Its VJP backward
    is therefore: scatter the cotangent into `[N,C,(H·W)]`, run the renderable
    three-term `bnBatchTensor4GradInput` at the reindexed activation, scatter
    back. This is what a network-layout `bnBatchLABack` op denotes. -/
theorem bnBatchLA_back_conj {N oc h w : Nat} (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (x dy : Vec (N * (oc * h * w))) :
    (reindexHasVJP (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm)).backward x
      (bnBatchTensor4GradInput N oc h w ε γ
        (reindexCLM (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm) x)
        ((reindexHasVJP (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)))).backward
          (reindexCLM (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm) x) dy))
      = (bnBatchLAHasVJP N oc h w ε hε γ β).backward x dy := by
  rw [bnBatchTensor4GradInput_eq_backward N oc h w ε hε γ β]
  simp only [bnBatchLAHasVJP, eq_mpr_eq_cast]
  rfl

/-- **`bnBatchLABack` (network-layout true batch-norm backward) faithfulness.**
    The `den` (inline scatter-conjugated `bnBatchTensor4GradInput`) equals the
    proven `bnBatchLAHasVJP` backward — the bn backward at the network's
    `N·(oc·h·w)` index. This is the
    layout wrapper that lets `bnBatchBack` compose with `convBackBatched` /
    `depthwiseBackBatched` (all on the left-assoc index) into batched stages. -/
theorem bnBatchLABack_faithful {N oc h w : Nat} (gN xN es : String)
    (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (x : Vec (N * (oc * h * w))) (e : SHlo (N * (oc * h * w))) :
    den (SHlo.bnBatchLABack gN xN es ε γ x e)
      = (bnBatchLAHasVJP N oc h w ε hε γ β).backward x (den e) :=
  bnBatchLA_back_conj ε γ β hε x (den e)

/-- **`seBackBatched` (batched squeeze-excite backward) faithfulness.** The `den`
    (rowwise application of the proven per-example `seBlockFull` VJP) equals the
    proven batched `seBHasVJP` backward. SE is non-linear, so — unlike the
    linear `convBackBatched`/`depthwiseBackBatched` — the backward threads each
    example's forward activation `v`; the rowwise `batchMapHasVJP` structure
    handles that. -/
theorem seBackBatched_faithful {N c h w r : Nat} (w1N b1N w2N b2N vN : String)
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    (v : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) :
    den (SHlo.seBackBatched (N := N) w1N b1N w2N b2N vN W₁ b₁ W₂ b₂ v e)
      = (seBHasVJP N (h := h) (w := w) W₁ b₁ W₂ b₂).backward v (den e) := by
  funext idx
  simp only [denStepApp, seBHasVJP, batchMapHasVJP, HasVJPMat.toHasVJP, rowwiseHasVJPMat]

-- ════════════════════════════════════════════════════════════════
-- § Batched MBConv stage backward graphs (the bn wrapper unblocks these)
-- ════════════════════════════════════════════════════════════════

/-- Batched **conv → bn → swish** stage backward graph (MBConv expand), at the
    network layout: `convBackBatched ∘ bnBatchLABack ∘ swishBack`, each at its
    cumulative forward activation. -/
noncomputable def cbsBackBatchedGraph {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) : SHlo (N * (ic * h * w)) :=
  .convBackBatched (N := N) "%cbsW" W b
    (.bnBatchLABack "%cbsG" "%cbsX" "cbsE" ε γ (batchMap N (flatConv W b) x)
      (.swishBack "%cbsSw" (bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x)) e))

theorem cbsBackBatchedGraph_faithful {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) :
    den (cbsBackBatchedGraph W b ε γ β x e)
      = (cbsBHasVJP N W b ε hε γ β).backward x (den e) := by
  rw [cbsBackBatchedGraph, convBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε), swishBack_faithful]
  simp only [cbsBHasVJP, bnSwishStageHasVJP, vjpComp_backward, Function.comp_apply]

/-- Batched **depthwise → bn → swish** stage backward graph (MBConv depthwise). -/
noncomputable def dwbsBackBatchedGraph {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  .depthwiseBackBatched (N := N) "%dwsW" W b
    (.bnBatchLABack "%dwsG" "%dwsX" "dwsE" ε γ (batchMap N (depthwiseFlat W b) x)
      (.swishBack "%dwsSw" (bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x)) e))

theorem dwbsBackBatchedGraph_faithful {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) :
    den (dwbsBackBatchedGraph W b ε γ β x e)
      = (dwbsBHasVJP N W b ε hε γ β).backward x (den e) := by
  rw [dwbsBackBatchedGraph, depthwiseBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε), swishBack_faithful]
  simp only [dwbsBHasVJP, bnSwishStageHasVJP, vjpComp_backward, Function.comp_apply]

/-- Batched **STRIDE-2 depthwise → bn → swish** stage backward graph (the
    EfficientNet downsample MBConv's depthwise). The stride-2 analogue of
    `dwbsBackBatchedGraph`: the bn/swish run at the OUTPUT spatial `h×w`, then
    `depthwiseStridedBackBatched` maps the bn-cotangent back to the larger input
    `c·(2h)·(2w)` (zero-upsample + reversed-kernel per-channel depthwise). -/
noncomputable def dwbsSBackBatchedGraph {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (e : SHlo (N * (c * h * w))) :
    SHlo (N * (c * (2 * h) * (2 * w))) :=
  .depthwiseStridedBackBatched (N := N) "%dwssW" W b
    (.bnBatchLABack "%dwssG" "%dwssX" "dwssE" ε γ (batchMap N (depthwiseStride2Flat W b) x)
      (.swishBack "%dwssSw" (bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2Flat W b) x)) e))

theorem dwbsSBackBatchedGraph_faithful {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (e : SHlo (N * (c * h * w))) :
    den (dwbsSBackBatchedGraph W b ε γ β x e)
      = (dwbsSBHasVJP N W b ε hε γ β).backward x (den e) := by
  rw [dwbsSBackBatchedGraph, depthwiseStridedBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε), swishBack_faithful]
  simp only [dwbsSBHasVJP, bnSwishStageHasVJP, vjpComp_backward, Function.comp_apply]

/-- Batched **conv → bn** stage backward graph (MBConv project, no swish). -/
noncomputable def projBackBatchedGraph {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ _β : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) : SHlo (N * (ic * h * w)) :=
  .convBackBatched (N := N) "%pbW" W b
    (.bnBatchLABack "%pbG" "%pbX" "pbE" ε γ (batchMap N (flatConv W b) x) e)

theorem projBackBatchedGraph_faithful {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) :
    den (projBackBatchedGraph W b ε γ β x e)
      = (projBHasVJP N W b ε hε γ β).backward x (den e) := by
  rw [projBackBatchedGraph, convBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε)]
  simp only [projBHasVJP, bnStageHasVJP, vjpComp_backward]

end Proofs.StableHLO

namespace Proofs.BackLinks

open scoped BigOperators

/-! ## Chain-cotangent helpers — the per-op batched backward steps (built fresh, HasVJP-style)

`EfficientNetChainClose` proves the per-block VJPs by `vjpComp` of the per-op VJPs but exposes no
explicit cotangent-vector defs (unlike mnv2's `invresCot*`). So the tie BUILDS the chain cotangents
from the proven per-op backwards: `bnBackB` (true-BN, the batch-coupled `bnBatchLA` VJP), `swBackB`
(swish, smooth), `cInB`/`dInB` (the batched conv/depthwise input-VJP = `den convBackBatched`/
`depthwiseBackBatched`), `seInB` (the fused SE input-cot = `den seBackBatched`), `gateCotB` (the SE
gate cotangent = `den seReduceB`), `sigBackB`, `rowDenseBackFlat` (the SE excite/reduce backs). Every
helper IS a `.backward` of a proven VJP (or the exact `den` of the emitted backward op), so the
cotangents are the genuine loss-driven backward, not a free `∀c`. `reassocB` bridges the conv/swish
index `(oc·h·w)` to the BN param-op index `(oc·(h·w))`. -/

/-- `(oc·h·w) → (oc·(h·w))` batched reassociation reindex — bridges the conv/swish chain index to the
    BN γ/β + conv-bias op index (`EnetPoC.bn{Gamma,Beta}B_den` consume `Vec (N·(oc·(h·w)))`). -/
noncomputable def reassocB (N oc h w : Nat) (v : Vec (N * (oc * h * w))) : Vec (N * (oc * (h * w))) :=
  fun i => v (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm i)

/-- Batched **true-BN** input-cotangent (`bnBatchLA` VJP — batch-coupled). -/
noncomputable def bnBackB (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x dy : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  (bnBatchLAHasVJP N oc h w ε hε γ β).backward x dy

/-- **The tie's BN node and the emitted BN node denote one map.** Every batched render emits
    `.bnBatchBack`, typed at `N·(oc·(h·w))`; the ties state the BN input cotangent at
    `.bnBatchLABack`, its network-layout `N·(oc·h·w)` twin (`BackLinks.bnInB`). The two print
    the same text, and their `den`s differ only by the associativity relabelling `reassocB`: the
    two scatters inside `bnBatchLABack`'s `den` collapse because `Fin.cast` is a bijection. -/
theorem den_bnBatchLABack_eq_bnBatchBack {N oc h w : Nat} (gN xN es : String) (ε : ℝ) (γ : Vec oc)
    (x : Vec (N * (oc * h * w))) (e : SHlo (N * (oc * h * w))) :
    den (SHlo.bnBatchLABack gN xN es ε γ x e)
      = fun i => den (SHlo.bnBatchBack gN xN es ε γ (reassocB N oc h w x)
          (.operand "" (reassocB N oc h w (den e))))
          (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)) i) := by
  funext i
  have h1 : ∀ k : Fin (N * (oc * (h * w))),
      (i = Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm k)
        ↔ (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)) i = k) := fun k => by
    simp only [Fin.ext_iff, Fin.val_cast]
  have hin : (fun i' : Fin (N * (oc * (h * w))) => ∑ k' : Fin (N * (oc * h * w)),
        if i' = Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)) k' then den e k' else 0)
      = reassocB N oc h w (den e) := by
    funext i'
    have h2 : ∀ k' : Fin (N * (oc * h * w)),
        (i' = Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)) k')
          ↔ (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm i' = k') := fun k' => by
      simp only [Fin.ext_iff, Fin.val_cast]
    simp_rw [h2]
    simp only [Finset.sum_ite_eq, Finset.mem_univ, ite_true, reassocB]
  simp only [denStep, denStepApp]
  simp_rw [h1]
  simp only [Finset.sum_ite_eq, Finset.mem_univ, ite_true]
  rw [hin]
  rfl

/-- **The certified BN input cotangent every batched step tie threads IS the emitted `bnBatchBack`
    node's `den`**, read back through `reassocB`. This is the missing half of
    `bnBatchLABack_faithful`: that lemma certifies the tie's node, this one says the render's node
    computes the same numbers. -/
theorem bnBackB_eq_den_bnBatchBack (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x dy : Vec (N * (oc * h * w))) :
    bnBackB N oc h w ε hε γ β x dy
      = fun i => den (SHlo.bnBatchBack "" "" "" ε γ (reassocB N oc h w x)
          (.operand "" (reassocB N oc h w dy)))
          (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)) i) := by
  show (bnBatchLAHasVJP N oc h w ε hε γ β).backward x (den (.operand "" dy)) = _
  rw [← bnBatchLABack_faithful "" "" "" ε γ β hε x (.operand "" dy),
      den_bnBatchLABack_eq_bnBatchBack]
  rfl

/-- Batched **swish** mask-back (smooth, no kink). -/
noncomputable def swBackB (n : Nat) (x dy : Vec n) : Vec n := (swishHasVJP n).backward x dy

/-- Batched **sigmoid** back (the SE gate excite-dense output cotangent). -/
noncomputable def sigBackB (n : Nat) (x dy : Vec n) : Vec n := (sigmoidHasVJP n).backward x dy

/-- Batched **1×1/conv input-VJP** (= `den convBackBatched`; conv is linear, `x` unused). -/
noncomputable def cInB (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (dy : Vec (N * (oc * h * w))) : Vec (N * (ic * h * w)) :=
  batchMap N (fun d => (flatConvHasVJP W b).backward (fun _ => 0) d) dy

/-- Batched **depthwise input-VJP** (= `den depthwiseBackBatched`). -/
noncomputable def dInB (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (dy : Vec (N * (c * h * w))) : Vec (N * (c * h * w)) :=
  batchMap N (fun d => (depthwiseFlatHasVJP W b).backward (fun _ => 0) d) dy

/-- Batched **strided depthwise input-VJP** (= `den depthwiseStridedBackBatched`; upsamples `h→2h`). -/
noncomputable def dStridedInB (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (dy : Vec (N * (c * h * w))) : Vec (N * (c * (2 * h) * (2 * w))) :=
  batchMap N (fun d => (depthwiseStride2FlatHasVJP W b).backward (fun _ => 0) d) dy

/-- Batched **GAP input-VJP** (= `den gapBackBatched`; the head's GAP backward, broadcast÷(h·w)). -/
noncomputable def gapInB (N c h w : Nat) (dy : Vec (N * c)) : Vec (N * (c * h * w)) :=
  batchMap N (fun d => (globalAvgPoolFlatHasVJP c h w).backward (fun _ => 0) d) dy

/-- Batched **fused SE input-cotangent** (= `den seBackBatched`, the `x⊙gate` VJP). -/
noncomputable def seInB (N : Nat) {c h w r : Nat} (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    (x dy : Vec (N * (c * h * w))) : Vec (N * (c * h * w)) :=
  (seBHasVJP N (h := h) (w := w) W₁ b₁ W₂ b₂).backward x dy

/-- Batched **SE gate cotangent** `dgate[n,c] = Σ_{h,w}(x⊙dy)` (= `den seReduceB`, the broadcast-adjoint
    of `x ⊙ dy` — the FIRST step of the SE gate backward, feeding the SE dense param grads). -/
noncomputable def gateCotB (N c h w : Nat) (x dy : Vec (N * (c * h * w))) : Vec (N * c) :=
  fun idx => ∑ q : Fin (c * h * w),
    if flatChannel c h w q = (finProdFinEquiv.symm idx).2 then
      batchSlice N (c * h * w) x (finProdFinEquiv.symm idx).1 q
        * batchSlice N (c * h * w) dy (finProdFinEquiv.symm idx).1 q
    else 0

end Proofs.BackLinks

namespace Proofs.BackLinks

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Cotangent steps first needed by ResNet-34: the relu mask, the strided conv input-VJP,
--   true BN as emitted, the 3×3/s2 pool backward (and, below, the one-row head casts)
-- ════════════════════════════════════════════════════════════════

/-- **The relu backward mask** — `den (.selectPosB _ pre e) = fun i => if pre i > 0 then e i else 0`.
    r34 applies it twice per block (the body's mid-relu and the post-residual outer one) and once at
    the stem. -/
noncomputable def reluMaskB (n : Nat) (pre dy : Vec n) : Vec n :=
  fun i => if pre i > 0 then dy i else 0

/-- **Batched STRIDED conv input-VJP** (= `den convStridedBackBatched`; upsamples `h → 2h`). The
    strided peer of EfficientNet's `cInB`. Note: SYMMETRIC padding — `flatConvStride2`, not the
    XLA-`SAME` twin. -/
noncomputable def cStridedInB (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (b : Vec oc) (dy : Vec (N * (oc * h * w))) : Vec (N * (ic * (2 * h) * (2 * w))) :=
  batchMap N (fun d => (flatConvStride2HasVJP W b).backward (fun _ => 0) d) dy

/-- **Batched true-BN input-cotangent, as the EMITTED backward computes it.** Written as the `den`
    of the backward op rather than as the certified VJP's `.backward`, because that is the form the
    render's chain is in and `den` ignores the name strings — so every cotangent below is literally
    what the artifact's bytes compute. Note: The render's node is `.bnBatchBack`, typed at
    `N·(oc·(h·w))`; this is its network-layout twin, and `bnInB_eq_den_bnBatchBack` below says
    the two denote one map up to `reassocB`. It takes no `β`: the BatchNorm input-gradient does not
    depend on the shift, which `bnInB_eq_bnBackB` records by holding for every `β`. -/
noncomputable def bnInB (N oc h w : Nat) (ε : ℝ) (γ : Vec oc)
    (x dy : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  den (SHlo.bnBatchLABack (N := N) (oc := oc) (h := h) (w := w) "" "" "" ε γ x (.operand "" dy))

/-- **…and it IS the certified `bnBatchLA` VJP**, for every `β` and every `0 < ε`. This is
    `bnBatchLABack_faithful`, and it is the only step in this file's cotangent chain that is not
    `rfl` — everything else (the relu masks, the conv and strided-conv input-VJPs, the pool
    backward) denotes its certified backward definitionally. -/
theorem bnInB_eq_bnBackB (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x dy : Vec (N * (oc * h * w))) :
    bnInB N oc h w ε γ x dy = bnBackB N oc h w ε hε γ β x dy :=
  bnBatchLABack_faithful "" "" "" ε γ β hε x (.operand "" dy)

/-- **…and it IS the `den` of the node the render emits**, `.bnBatchBack` at the `N·(oc·(h·w))`
    index, read back through `reassocB` (`BackLinks.den_bnBatchLABack_eq_bnBatchBack`). -/
theorem bnInB_eq_den_bnBatchBack (N oc h w : Nat) (ε : ℝ) (γ : Vec oc)
    (x dy : Vec (N * (oc * h * w))) :
    bnInB N oc h w ε γ x dy
      = fun i => den (SHlo.bnBatchBack "" "" "" ε γ (reassocB N oc h w x)
          (.operand "" (reassocB N oc h w dy)))
          (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)) i) :=
  BackLinks.den_bnBatchLABack_eq_bnBatchBack "" "" "" ε γ x (.operand "" dy)

/-- **Batched 3×3/s2 max-pool backward** (= `den maxPool3s2BackB`): the `select_and_scatter`
    denotation, per example on that example's own saved activation — which is why it is
    `batchMapAux` and not `batchMap`. -/
noncomputable def mpInB (N c h w : Nat) (x : Vec (N * (c * (2 * h) * (2 * w))))
    (dy : Vec (N * (c * h * w))) : Vec (N * (c * (2 * h) * (2 * w))) :=
  batchMapAux N (maxPool3s2BackFlat c h w) x dy


/-- `Vec (N·(1·K)) → Vec (N·K)`: the loss chain runs at one ROW per example (`softmaxRow` needs a
    row index) and the dense parameter ops at the plain per-example width. The render writes one
    SSA name for both, because `1 * K = K` as an emitted shape; in Lean the two indices are
    propositionally but not definitionally equal, so the cast is explicit. -/
noncomputable def unrowB (N K : Nat) (v : Vec (N * (1 * K))) : Vec (N * K) :=
  fun i => v (Fin.cast (congrArg (N * ·) (Nat.one_mul K)).symm i)

/-- The inverse cast of `unrowB`: the head's logits, at the one-row-per-example index the loss
    chain's `softmaxRow` consumes. -/
noncomputable def rowB (N K : Nat) (v : Vec (N * K)) : Vec (N * (1 * K)) :=
  fun i => v (Fin.cast (congrArg (N * ·) (Nat.one_mul K)) i)

end Proofs.BackLinks
