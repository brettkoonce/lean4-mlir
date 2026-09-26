import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFoldG

/-! # The gradient-node fold for ConvNeXt-T at the batched index

`ConvNeXtFoldG.lean` holds the three per-example lemmas specific to this net (layer-scale γ,
channel-LN γ/β); the other node kinds are shared (`GradNodesB`). This file is the batched peer, at
`ConvNeXtRenderB.convNextBackAllB`'s constructors, from which every ConvNeXt Adam artifact
renders.

Against the per-example chain, the batched `convBackBatched` emits the conv input-VJP's
`transpose`/`reverse` in the other order from the per-example `convBack` — commuting ops on
disjoint axes — and tests/TestConvNeXtFwdBTie.lean allows exactly that pair and nothing else.

## The op table of `convnext_adam_train_step.mlir` and every `convnextin_*` train step

| emitted node | lemma | per-example peer it batches |
|---|---|---|
| `layerScaleChGammaGradB` (18 block γ) | `layerScaleChGammaGradB_den` | `CnxPoCG.layerScaleChGammaGrad_den` |
| `convWeightGradB` / `convBiasGradB` (18 expand + 18 project 1×1, + the stem bias) | `ResNet34PoCB.convWGradB_den` / `convBGradB_den` (`GradNodesB`) | — |
| `depthwiseWeightGradB` / `depthwiseBiasGradB` (18 × 7×7) | `EnetPoCG.depthwiseWGradB_den` / `Mnv2PaperPoCG.depthwiseBGradB_den` (`GradNodesB`) | — |
| `convStridedWeightGradB` / `convStridedBiasGradB` (3 × 2×2/s2 downsample) | `ResNet34PoCB.convStridedWGradB_den` / `convStridedBGradB_den` (`GradNodesB`) | — |
| `convStride4WeightGradB` (patchify stem) | `CnxPoCGB.psWGradB_den` (`GradNodesB`) | `flatConvStride4WeightGradHasVJP`, per example |
| `veclnGammaGradB` / `rowDenseBiasGradB` at `R = h·w` (22 spatial LN sites) | `chanLnGammaGradB_den` / `chanLnBetaGradB_den` | `CnxPoCG.chanLnGammaGrad_den` / `chanLnBetaGrad_den` |
| `veclnGammaGradB` / `rowDenseBiasGradB` at `R = 1` (the head LN, after GAP) | `ViTPoCGB.veclnGammaGradB_den` / `rowDenseBiasGradB_den_lnbeta` | — |
| `weightGradB` / `biasGradB` (the classifier) | `ViTPoCGB.headWGradB_den` / `headBGradB_den` | — |
| `convWeightGradBBf16` / `depthwiseWeightGradBBf16` / `convStridedWeightGradBBf16` / `convStride4WeightGradBBf16` (the bf16 artifacts) | `Bf16PoC.convWGradBBf16_den` and its siblings, [`Foundation/Bf16GradNodes.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Foundation/Bf16GradNodes.lean) | none — a bf16 node is its own op kind |

**No new mathematics.** Every proof is `Finset.sum_congr rfl` over the batch and then the
per-example bridge at `batchSlice n` — `ResNet34PoCB.denseWGradB_den`'s shape — because
each batched `den` arm is literally the per-example one under a batch sum. The channel-LN sites
add one step: the batched render hands the LN ops `batchMap N (chanLNRows c h w)` of the saved
input and of the cotangent (the `[h·w, c]` transposed views, lifted per example), and
`batchSlice_batchMap` peels the lift so `ChannelLN`'s permutation argument applies at each
slice.

**The bf16 artifacts (`convnextin_adamwxclipdropbf16`, the S/B twins) emit `*GradBBf16`
constructors, not these nodes**: their `den` rounds the operands and the result once, outside the
batch sum. Those are their own op kinds, folded once for every net in
[`Foundation/Bf16GradNodes.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Foundation/Bf16GradNodes.lean).

**One lemma per f32 node kind.** Every lemma is `∀ cot`, so it holds at whichever cotangent
the chain delivers: the f32 AdamW, `wx`/`clip`, EMA, drop-path and data-parallel artifacts all
emit these `*GradB` kinds, and the optimizer update that consumes the node is outside these
lemmas.

## Scope
* **`biasGradB` is the identity on its operand** and the classifier bias's batch reduce is in
  the emitted text, outside the AST — so `ViTPoCGB.headBGradB_den` is stated PER EXAMPLE at `batchSlice n`,
  the per-example `biasGrad` carve-out carried over unchanged (as in `ViTFoldGB`).
* Every lemma is `∀ cot`. The tie at these nodes, with the cotangents the emitted backward chain
  delivers and the smoothed loss, is `CnxTiePoCGB.cnx_net_tiedGB`.
* `convnextin_adamdp*` is four replicas: the all-reduce is its own `allReduceMeanF` node after each
  gradient node (`DataParallel.Node`), so these lemmas are about the per-replica gradient node
  it averages.
* Symmetric padding at the three 2×2/s2 downsamples and the 4×4/s4 stem (`flatConvStride2`,
  `flatConvStride4`); ConvNeXt is PyTorch-origin and has no XLA-`SAME` site.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CnxPoCGB

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-channel layer scale — the one op kind unique to this net
-- ════════════════════════════════════════════════════════════════

/-- **Batched per-channel layer-scale γ GRADIENT denotes the certified `Σ_n` gradient.** The
    emitted reduce contracts batch and spatial in one op; `den` reads it as the batch sum of the
    per-example `dγ_c = Σ_{k : chanIdx k = c} x_k·dy_k`. All 18 blocks. -/
theorem layerScaleChGammaGradB_den {N c h w : Nat} (xN cotN : String)
    (x : Vec (N * (c * h * w))) (γ : Vec c) (dy : Vec (N * (c * h * w))) (cc : Fin c) :
    den (SHlo.layerScaleChGammaGradB (N := N) (c := c) (h := h) (w := w) xN x (.operand cotN dy)) cc
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c =>
                  layerScale (fun k => γ' (chanIdx c h w k)) (batchSlice N (c * h * w) x n))
               γ cc j * batchSlice N (c * h * w) dy n j := by
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  have h := Proofs.CnxPoCG.layerScaleChGammaGrad_den (h := h) (w := w) xN cotN
    (batchSlice N (c * h * w) x n) γ (batchSlice N (c * h * w) dy n) cc
  simp only [denStepApp] at h
  exact h

-- ════════════════════════════════════════════════════════════════
-- § The 22 spatial LayerNorm sites — the CHANNEL-LN form, batched
--   The render hands the two-level ops `batchMap N (chanLNRows c h w)` of the saved LN input and
--   of the cotangent — the per-example `[h·w, c]` view, lifted — and the certified Jacobian is
--   `chanLNTensor3`'s in the `c·h·w` layout at each `batchSlice n`.
-- ════════════════════════════════════════════════════════════════

/-- **Batched channel-LN γ GRADIENT denotes the certified `Σ_n` γ gradient.** All 22 spatial sites
    (1 stem + 18 block + 3 downsample). Two levels: the outer sum is the batch, the inner the
    `h·w` rows within one example. -/
theorem chanLnGammaGradB_den {N c h w : Nat} (xN epsStr cotN : String)
    (ε : ℝ) (β : Vec c) (x : Vec (N * (c * h * w))) (γ : Vec c) (cot : Vec (N * (c * h * w)))
    (k : Fin c) :
    den (SHlo.veclnGammaGradB (N := N) (R := h * w) (D := c) xN epsStr ε
          (batchMap N (chanLNRows c h w) x)
          (.operand cotN (batchMap N (chanLNRows c h w) cot))) k
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c => chanLNTensor3 c h w ε γ' β (batchSlice N (c * h * w) x n)) γ k j
            * batchSlice N (c * h * w) cot n j := by
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  rw [batchSlice_batchMap, batchSlice_batchMap]
  have h := Proofs.CnxPoCG.chanLnGammaGrad_den xN epsStr cotN ε β
    (batchSlice N (c * h * w) x n) γ (batchSlice N (c * h * w) cot n) k
  simp only [denStep, denStepApp] at h
  exact h

/-- **Batched channel-LN β GRADIENT denotes the certified `Σ_n` β gradient.** The β gradient is
    the plain two-level row reduce, so the render uses the same `rowDenseBiasGradB` op ViT's LN β
    does. -/
theorem chanLnBetaGradB_den {N c h w : Nat} (cotN : String)
    (ε : ℝ) (γ : Vec c) (x : Vec (N * (c * h * w))) (β : Vec c) (cot : Vec (N * (c * h * w)))
    (k : Fin c) :
    den (SHlo.rowDenseBiasGradB (N := N) (R := h * w) (c := c)
          (.operand cotN (batchMap N (chanLNRows c h w) cot))) k
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun β' : Vec c => chanLNTensor3 c h w ε γ β' (batchSlice N (c * h * w) x n)) β k j
            * batchSlice N (c * h * w) cot n j := by
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  rw [batchSlice_batchMap]
  have h := Proofs.CnxPoCG.chanLnBetaGrad_den cotN ε γ (batchSlice N (c * h * w) x n) β
    (batchSlice N (c * h * w) cot n) k
  simp only [denStep, denStepApp] at h
  exact h

-- ════════════════════════════════════════════════════════════════
-- § Tie clauses — one batched channel-LN gradient node each (each its `_den` lemma's statement with the index
--   bound, over the flat input `x`; each `…_holds` below proves it)
-- ════════════════════════════════════════════════════════════════

/-- A batched channel-LN γ gradient node, tied (`chanLnGammaGradB_den`). -/
def ChanLNGammaTiedB (N h w : Nat) {c : Nat} (xN epsStr cotN : String) (ε : ℝ) (β : Vec c)
    (x : Vec (N * (c * h * w))) (γ : Vec c) (cot : Vec (N * (c * h * w))) : Prop :=
  ∀ k : Fin c,
    den (SHlo.veclnGammaGradB (N := N) (R := h * w) (D := c) xN epsStr ε
          (batchMap N (chanLNRows c h w) x)
          (.operand cotN (batchMap N (chanLNRows c h w) cot))) k
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c => chanLNTensor3 c h w ε γ' β (batchSlice N (c * h * w) x n)) γ k j
            * batchSlice N (c * h * w) cot n j

/-- A batched channel-LN β gradient node, tied (`chanLnBetaGradB_den`). -/
def ChanLNBetaTiedB (N h w : Nat) {c : Nat} (cotN : String) (ε : ℝ) (γ : Vec c)
    (x : Vec (N * (c * h * w))) (β : Vec c) (cot : Vec (N * (c * h * w))) : Prop :=
  ∀ k : Fin c,
    den (SHlo.rowDenseBiasGradB (N := N) (R := h * w) (c := c)
          (.operand cotN (batchMap N (chanLNRows c h w) cot))) k
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun β' : Vec c => chanLNTensor3 c h w ε γ β' (batchSlice N (c * h * w) x n)) β k j
            * batchSlice N (c * h * w) cot n j

/-! Each clause holds, every argument implicit (read off the goal by a step tie's constructor). -/

theorem chanLNGammaTiedB_holds {N h w c : Nat} {xN epsStr cotN : String} {ε : ℝ} {β : Vec c}
    {x : Vec (N * (c * h * w))} {γ : Vec c} {cot : Vec (N * (c * h * w))} :
    ChanLNGammaTiedB N h w xN epsStr cotN ε β x γ cot := fun k =>
  chanLnGammaGradB_den xN epsStr cotN ε β x γ cot k

theorem chanLNBetaTiedB_holds {N h w c : Nat} {cotN : String} {ε : ℝ} {γ : Vec c}
    {x : Vec (N * (c * h * w))} {β : Vec c} {cot : Vec (N * (c * h * w))} :
    ChanLNBetaTiedB N h w cotN ε γ x β cot := fun k => chanLnBetaGradB_den cotN ε γ x β cot k

end Proofs.CnxPoCGB
