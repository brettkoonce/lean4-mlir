import LeanMlir.Proofs.Architectures.ConvNeXtFaithfulPoCGB
import LeanMlir.Proofs.Architectures.ConvNeXtTiePoC
import LeanMlir.Proofs.Foundation.SmoothedLossCot

/-! # ConvNeXt-T's T3 §1a TIE at the BATCHED index, the UN-FUSED gradient and the SMOOTHED loss

`ConvNeXtTiePoC.lean` ties all 182 parameters of the SGD-inline `convnext_train_step.mlir`: each
fused `θ − lr·g` op `den`s to the certified step at the cotangent the emitted backward chain
delivers, per example, at a hard label. This file is that statement re-pointed along the THREE
axes 4b and 4c left open (`planning/proofs_tier_to_paper_nets.md` §4b, §4c-quater) — and unlike
EfficientNet-B0's (`EfficientNetTiePoCG.lean`, two axes), ConvNeXt's per-example capstone was at
a single image with the batch outside the AST, so the index is the third.

⭐ **Axis 1 — the OPTIMIZER FORM.** Every conjunct is at the RAW gradient node (`*GradB`), which is
what `convnext_adam_train_step.mlir` and every `convnextin_*` artifact emit; the fused op appears
only in the SGD-inline file. One statement covers AdamW, the `wx`/`clip` variants, EMA and the
data-parallel twins, because they all consume this node. `ConvNeXtFaithfulPoCGB.lean` (§4c-quater)
is the fold each conjunct delegates to.

⭐ **Axis 2 — the LOSS.** The capstone's top-of-chain cotangent is `smoothedLossCotGraphDiv`'s, at
a GENERAL target: the six-op chain `expe → softmaxDiv → subB → scaleB → addVB → shiftB →
divConstB` this render emits, with the target arriving as the graph input `%onehot` — a soft
vector under mixup or cutmix. The fused file pins it to `softmax − oneHot`, the gradient of plain
cross-entropy at a hard label, which no ImageNet artifact computes. ⚠ ConvNeXt's chain runs at the
plain width `N·K` (no `1·` row index), so there is no `rowB`/`unrowB` cast anywhere here.

⭐ **Axis 3 — the INDEX.** `N` is a binder. Every forward activation is `batchMap N` of the
per-example prefix the fused file threads (`cnxStemFwdO`, `cnxBlockFwdChO`, `cnxDownFwdChO`),
and every cotangent is `batchMapAux N` of the per-example chain (`cnxBlockCotInChAt`,
`cnxDownCotInChAt`, `ConvNeXtChainClose`'s `cnxCotP/E/N`, `chanLNTensor3Back`). That lift is
honest for this net and for no BatchNorm net: LayerNorm, GELU, the convolutions, layer scale and
the residual add are all batch-separable, so the batched op IS the per-example op under
`batchMap` — which is what the `*B` constructors' `den` arms say. `nC` is a binder too: the
Imagenette artifact is `nC = 10`, the ImageNet ones `nC = 1000`.

## What is NOT new, and why the file is a transformation rather than a proof

Every activation, every Jacobian witness and every chain cotangent is `ConvNeXtTiePoC.lean`'s,
lifted; every conjunct's proof is one `CnxPoCGB.*_den` lemma. The `wN`, `bN`, `gN`, `lrStr` and
`lr` binders disappear with the fused wrapper, exactly as they did for B0. ⚠ **The head takes `g`
as a PARAMETER** where the fused `cnxHeadChTied` computed it from a label — that is the whole of
axis 2 at the block level; the per-block ties were already loss-agnostic.

⛔ **Conventions carried unchanged from the fused file:** channel LayerNorm (`chanLNTensor3`, a
`Vec c` affine, `h·w` statistics per example) at all 22 spatial sites, the head at ViT's vector LN
on one row, per-channel layer scale, GELU (no kink — no smoothness hypothesis anywhere),
SYMMETRIC padding at the 4×4/s4 stem and the three 2×2/s2 downsamples. Stated at the literal
widths of ConvNeXt-T; S and B are other nets. ⛔ ONE REPLICA: in `convnextin_adamdp*` every
gradient node feeds `allReduceMeanF`, the collective as an AST node since 4d piece 2 (2026-09-07);
`DataParallelNode.lean` composes the per-replica statement with the replica mean.
⛔ The `%dgi…%dgapf` GAP backward is hand-written text on both chains (a declared carve-out); its
value here is `globalAvgPoolFlat_has_vjp.backward`, as in the fused file.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CnxTiePoCGB

open scoped BigOperators
open Proofs.CnxTiePoC (cnxStemFwdO cnxBlockFwdChO cnxDownFwdChO cnxBlockCotInChAt cnxDownCotInChAt)

/-! ## Per-example internal cotangents as functions of a block's INPUT

`ConvNeXtChainClose`'s `cnxCotE/N` and `chanLNTensor3Back` take the saved activations as
arguments; `batchMapAux` lifts a function of (one saved value, one input), so these recompute the
activations from the block input, exactly as `cnxBlockCotInChAt` does. -/

/-- Cotangent at the expand output (pre-GELU), from the block input and output cotangent. -/
noncomputable def blkCotE {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin dyOut : Vec (c*h*w)) : Vec (cExp*h*w) :=
  let γlsB : Vec (c*h*w) := fun k => lg (chanIdx c h w k)
  let nl := chanLNTensor3 c h w ε ng nbt (depthwiseFlat (h := h) (w := w) Wdw bdw xin)
  let e := flatConv (h := h) (w := w) Wex bex nl
  cnxCotE γlsB Wpr bpr (gelu (cExp*h*w) e) e dyOut

/-- Cotangent at the channel-LN output, from the block input and output cotangent. -/
noncomputable def blkCotN {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin dyOut : Vec (c*h*w)) : Vec (c*h*w) :=
  let γlsB : Vec (c*h*w) := fun k => lg (chanIdx c h w k)
  let nl := chanLNTensor3 c h w ε ng nbt (depthwiseFlat (h := h) (w := w) Wdw bdw xin)
  let e := flatConv (h := h) (w := w) Wex bex nl
  cnxCotN γlsB Wex bex Wpr bpr nl (gelu (cExp*h*w) e) e dyOut

/-- Cotangent at the depthwise output (the channel-LN input-VJP of `blkCotN`). -/
noncomputable def blkCotD {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin dyOut : Vec (c*h*w)) : Vec (c*h*w) :=
  chanLNTensor3Back c h w ε ng (depthwiseFlat (h := h) (w := w) Wdw bdw xin)
    (blkCotN ε Wdw bdw ng nbt Wex bex Wpr bpr lg xin dyOut)

/-- Downsample: cotangent at the LN output, i.e. the strided conv's input-VJP. -/
noncomputable def dnCotN {ci co h w : Nat} (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (ci*(2*h)*(2*w))) (dyOut : Vec (co*h*w)) : Vec (ci*(2*h)*(2*w)) :=
  (flatConvStride2_has_vjp Wd bd).backward (chanLNTensor3 ci (2*h) (2*w) ε dng dnbt xin) dyOut

/-- Stem: cotangent at the patchify output, the stem LN's input-VJP of `dyStem`. -/
noncomputable def stemCotPatch {c h w : Nat} (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng : Vec c)
    (x : Vec (3*(2*(2*h))*(2*(2*w)))) (dyStem : Vec (c*h*w)) : Vec (c*h*w) :=
  chanLNTensor3Back c h w ε psng (flatConvStride4 Wst psb x) dyStem

/-- Head: the dense backward at one example's LN output and loss cotangent. -/
noncomputable def headCotHn {nC : Nat} (Wfc : Mat 768 nC) (bfc : Vec nC)
    (hn : Vec 768) (g : Vec nC) : Vec (1*768) :=
  (dense_has_vjp Wfc bfc).backward hn g

/-- The cotangent at the last block output, per example, at a GENERAL class count —
    `CnxTiePoC.cnxHeadDyXheadCh` with `nC` a binder (that one is at the literal 10). -/
@[irreducible] noncomputable def cnxHeadDyXheadChN {h w nC : Nat} (ε : ℝ)
    (hng hnbt : Vec 768) (Wfc : Mat 768 nC) (bfc : Vec nC)
    (xhead : Vec (768*h*w)) (g : Vec nC) : Vec (768*h*w) :=
  let gap : Vec (1*768) := globalAvgPoolFlat 768 h w xhead
  let hn : Vec 768 := rowLNVecFlat 1 768 ε hng hnbt gap
  let cotHn : Vec (1*768) := (dense_has_vjp Wfc bfc).backward hn g
  let cotGap : Vec 768 := rowLNVecFlatBack 1 768 ε hng gap cotHn
  (globalAvgPoolFlat_has_vjp 768 h w).backward xhead cotGap

/-! ## ConvNeXt block — all 9 gradient nodes, batched -/

/-- **ConvNeXt block, tied at the batched gradient nodes.** All 9 params (depthwise 7×7 `W`+`b`,
    channel-LN γ/β at `Vec c`, expand/project 1×1 `W`+`b`, per-channel layer-scale γ) denote the
    certified `Σ_n` gradient at the real batched block forward and the chain cotangents driven by
    `dyOut`. -/
def cnxBlockChTiedGB (N : Nat) {c cExp h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin dyOut : Vec (N * (c*h*w))) : Prop :=
  let γlsB : Vec (c*h*w) := fun k => lg (chanIdx c h w k)
  -- forward activations — each `batchMap` of the per-example op the emitted node lifts
  let dB  : Vec (N * (c*h*w))    := batchMap N (depthwiseFlat (h := h) (w := w) Wdw bdw) xin
  let nlB : Vec (N * (c*h*w))    := batchMap N (chanLNTensor3 c h w ε ng nbt) dB
  let gB  : Vec (N * (cExp*h*w)) :=
    batchMap N (fun nl => gelu (cExp*h*w) (flatConv (h := h) (w := w) Wex bex nl)) nlB
  let pB  : Vec (N * (c*h*w))    := batchMap N (flatConv (h := h) (w := w) Wpr bpr) gB
  -- backward chain cotangents — `batchMapAux` of the per-example chain
  let cotPB : Vec (N * (c*h*w))    := batchMap N (cnxCotP γlsB) dyOut
  let cotEB : Vec (N * (cExp*h*w)) :=
    batchMapAux N (blkCotE ε Wdw bdw ng nbt Wex bex Wpr bpr lg) xin dyOut
  let cotNB : Vec (N * (c*h*w))    :=
    batchMapAux N (blkCotN ε Wdw bdw ng nbt Wex bex Wpr bpr lg) xin dyOut
  let cotDB : Vec (N * (c*h*w))    :=
    batchMapAux N (blkCotD ε Wdw bdw ng nbt Wex bex Wpr bpr lg) xin dyOut
  -- depthwise 7×7 W/b  (cot = cotDB, input = xin)
  (∀ idx : Fin (c*7*7),
      den (SHlo.depthwiseWeightGradB xN bdw xin Wdw (.operand cotN cotDB)) idx
        = ∑ n : Fin N, ∑ j : Fin (c*h*w),
            pdiv (fun v' : Vec (c*7*7) =>
                    Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') bdw
                      (Tensor3.unflatten (batchSlice N (c*h*w) xin n))))
                 (Tensor3.flatten Wdw) idx j * batchSlice N (c*h*w) cotDB n j)
  ∧ (∀ o : Fin c,
      den (SHlo.depthwiseBiasGradB Wdw xin bdw (.operand cotN cotDB)) o
        = ∑ n : Fin N, ∑ j : Fin (c*h*w),
            pdiv (fun b' : Vec c =>
                    Tensor3.flatten (depthwiseConv2d Wdw b'
                      (Tensor3.unflatten (batchSlice N (c*h*w) xin n))))
                 bdw o j * batchSlice N (c*h*w) cotDB n j)
  -- channel-LN γ/β  (cot = cotNB, LN input = dB; the ops see both as their batched [h·w, c] views)
  ∧ (∀ k : Fin c,
      den (SHlo.veclnGammaGradB (N := N) (R := h*w) (D := c) xN epsStr ε
            (batchMap N (chanLNRows c h w) dB)
            (.operand cotN (batchMap N (chanLNRows c h w) cotNB))) k
        = ∑ n : Fin N, ∑ j : Fin (c*h*w),
            pdiv (fun γ' : Vec c => chanLNTensor3 c h w ε γ' nbt (batchSlice N (c*h*w) dB n))
                 ng k j * batchSlice N (c*h*w) cotNB n j)
  ∧ (∀ k : Fin c,
      den (SHlo.rowDenseBiasGradB (N := N) (R := h*w) (c := c)
            (.operand cotN (batchMap N (chanLNRows c h w) cotNB))) k
        = ∑ n : Fin N, ∑ j : Fin (c*h*w),
            pdiv (fun β' : Vec c => chanLNTensor3 c h w ε ng β' (batchSlice N (c*h*w) dB n))
                 nbt k j * batchSlice N (c*h*w) cotNB n j)
  -- expand 1×1 conv (c → cExp) W/b  (cot = cotEB, conv input = nlB)
  ∧ (∀ idx : Fin (cExp*c*1*1),
      den (SHlo.convWeightGradB xN bex nlB Wex (.operand cotN cotEB)) idx
        = ∑ n : Fin N, ∑ j : Fin (cExp*h*w),
            pdiv (fun v' : Vec (cExp*c*1*1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') bex
                      (Tensor3.unflatten (batchSlice N (c*h*w) nlB n))))
                 (Kernel4.flatten Wex) idx j * batchSlice N (cExp*h*w) cotEB n j)
  ∧ (∀ o : Fin cExp,
      den (SHlo.convBiasGradB (h := h) (w := w) Wex nlB bex (.operand cotN cotEB)) o
        = ∑ n : Fin N, ∑ j : Fin (cExp*h*w),
            pdiv (fun b' : Vec cExp =>
                    Tensor3.flatten (conv2d Wex b'
                      (Tensor3.unflatten (batchSlice N (c*h*w) nlB n))))
                 bex o j * batchSlice N (cExp*h*w) cotEB n j)
  -- project 1×1 conv (cExp → c) W/b  (cot = cotPB, conv input = gB)
  ∧ (∀ idx : Fin (c*cExp*1*1),
      den (SHlo.convWeightGradB xN bpr gB Wpr (.operand cotN cotPB)) idx
        = ∑ n : Fin N, ∑ j : Fin (c*h*w),
            pdiv (fun v' : Vec (c*cExp*1*1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') bpr
                      (Tensor3.unflatten (batchSlice N (cExp*h*w) gB n))))
                 (Kernel4.flatten Wpr) idx j * batchSlice N (c*h*w) cotPB n j)
  ∧ (∀ o : Fin c,
      den (SHlo.convBiasGradB (h := h) (w := w) Wpr gB bpr (.operand cotN cotPB)) o
        = ∑ n : Fin N, ∑ j : Fin (c*h*w),
            pdiv (fun b' : Vec c =>
                    Tensor3.flatten (conv2d Wpr b'
                      (Tensor3.unflatten (batchSlice N (cExp*h*w) gB n))))
                 bpr o j * batchSlice N (c*h*w) cotPB n j)
  -- per-channel layer-scale γ  (cot = dyOut directly, layer input = pB)
  ∧ (∀ cc : Fin c,
      den (SHlo.layerScaleChGammaGradB (N := N) (c := c) (h := h) (w := w) xN pB
            (.operand cotN dyOut)) cc
        = ∑ n : Fin N, ∑ j : Fin (c*h*w),
            pdiv (fun γ' : Vec c =>
                    layerScale (fun k => γ' (chanIdx c h w k)) (batchSlice N (c*h*w) pB n))
                 lg cc j * batchSlice N (c*h*w) dyOut n j)

theorem cnx_block_ch_tiedGB (N : Nat) {c cExp h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin dyOut : Vec (N * (c*h*w))) :
    cnxBlockChTiedGB N xN epsStr cotN ε Wdw bdw ng nbt Wex bex Wpr bpr lg xin dyOut := by
  unfold cnxBlockChTiedGB
  intro γlsB dB nlB gB pB cotPB cotEB cotNB cotDB
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact CnxPoCGB.depthwiseWGradB_den xN cotN bdw xin Wdw cotDB idx
  · intro o;   exact CnxPoCGB.depthwiseBGradB_den cotN Wdw xin bdw cotDB o
  · intro k;   exact CnxPoCGB.chanLnGammaGradB_den xN epsStr cotN ε nbt dB ng cotNB k
  · intro k;   exact CnxPoCGB.chanLnBetaGradB_den cotN ε ng dB nbt cotNB k
  · intro idx; exact CnxPoCGB.convWGradB_den xN cotN bex nlB Wex cotEB idx
  · intro o;   exact CnxPoCGB.convBGradB_den cotN Wex nlB bex cotEB o
  · intro idx; exact CnxPoCGB.convWGradB_den xN cotN bpr gB Wpr cotPB idx
  · intro o;   exact CnxPoCGB.convBGradB_den cotN Wpr gB bpr cotPB o
  · intro cc;  exact CnxPoCGB.layerScaleChGammaGradB_den xN cotN pB lg dyOut cc

/-! ## Downsample — channel-LN → 2×2/s2 conv, all 4 gradient nodes, batched -/

/-- **Downsample, tied at the batched gradient nodes.** Channel-LN γ/β at the `ci·(2h)·(2w)` input
    grid, plus the strided conv's weight and bias. -/
def cnxDownChTiedGB (N : Nat) {ci co h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (N * (ci*(2*h)*(2*w)))) (dyOut : Vec (N * (co*h*w))) : Prop :=
  let nB : Vec (N * (ci*(2*h)*(2*w))) := batchMap N (chanLNTensor3 ci (2*h) (2*w) ε dng dnbt) xin
  let cotNB : Vec (N * (ci*(2*h)*(2*w))) := batchMapAux N (dnCotN ε dng dnbt Wd bd) xin dyOut
  (∀ k : Fin ci,
      den (SHlo.veclnGammaGradB (N := N) (R := (2*h)*(2*w)) (D := ci) xN epsStr ε
            (batchMap N (chanLNRows ci (2*h) (2*w)) xin)
            (.operand cotN (batchMap N (chanLNRows ci (2*h) (2*w)) cotNB))) k
        = ∑ n : Fin N, ∑ j : Fin (ci*(2*h)*(2*w)),
            pdiv (fun γ' : Vec ci =>
                    chanLNTensor3 ci (2*h) (2*w) ε γ' dnbt (batchSlice N (ci*(2*h)*(2*w)) xin n))
                 dng k j * batchSlice N (ci*(2*h)*(2*w)) cotNB n j)
  ∧ (∀ k : Fin ci,
      den (SHlo.rowDenseBiasGradB (N := N) (R := (2*h)*(2*w)) (c := ci)
            (.operand cotN (batchMap N (chanLNRows ci (2*h) (2*w)) cotNB))) k
        = ∑ n : Fin N, ∑ j : Fin (ci*(2*h)*(2*w)),
            pdiv (fun β' : Vec ci =>
                    chanLNTensor3 ci (2*h) (2*w) ε dng β' (batchSlice N (ci*(2*h)*(2*w)) xin n))
                 dnbt k j * batchSlice N (ci*(2*h)*(2*w)) cotNB n j)
  ∧ (∀ idx : Fin (co*ci*2*2),
      den (SHlo.convStridedWeightGradB xN bd nB Wd (.operand cotN dyOut)) idx
        = ∑ n : Fin N, ∑ j : Fin (co*h*w),
            pdiv (fun v' : Vec (co*ci*2*2) =>
                    flatConvStride2 (Kernel4.unflatten v') bd (batchSlice N (ci*(2*h)*(2*w)) nB n))
                 (Kernel4.flatten Wd) idx j * batchSlice N (co*h*w) dyOut n j)
  ∧ (∀ o : Fin co,
      den (SHlo.convStridedBiasGradB (h := h) (w := w) Wd nB bd (.operand cotN dyOut)) o
        = ∑ n : Fin N, ∑ j : Fin (co*h*w),
            pdiv (fun b' : Vec co => flatConvStride2 Wd b' (batchSlice N (ci*(2*h)*(2*w)) nB n))
                 bd o j * batchSlice N (co*h*w) dyOut n j)

theorem cnx_down_ch_tiedGB (N : Nat) {ci co h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (N * (ci*(2*h)*(2*w)))) (dyOut : Vec (N * (co*h*w))) :
    cnxDownChTiedGB N xN epsStr cotN ε dng dnbt Wd bd xin dyOut := by
  unfold cnxDownChTiedGB
  intro nB cotNB
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro k;   exact CnxPoCGB.chanLnGammaGradB_den xN epsStr cotN ε dnbt xin dng cotNB k
  · intro k;   exact CnxPoCGB.chanLnBetaGradB_den cotN ε dng xin dnbt cotNB k
  · intro idx; exact CnxPoCGB.convStridedWGradB_den xN cotN bd nB Wd dyOut idx
  · intro o;   exact CnxPoCGB.convStridedBGradB_den cotN Wd nB bd dyOut o

/-! ## Stem — 4×4/s4 patchify conv → channel-LN, all 4 gradient nodes, batched

The bias grad is a pure cotangent reduce, so the render emits it as a stride-1 `convBiasGradB` at
the OUTPUT resolution and the carried `W`/`x` are generic (`xstem` is a free parameter here, as
`Tensor3 3 h w` is in the fused file). The weight is `convStride4WeightGradB`, at its gradient
on both chains. -/

/-- **Stem, tied at the batched gradient nodes.** Channel-LN γ/β, the conv bias, the conv weight. -/
def cnxStemChTiedGB (N : Nat) {c h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c)
    (x : Vec (N * (3*(2*(2*h))*(2*(2*w))))) (xstem : Vec (N * (3*h*w)))
    (dyStem : Vec (N * (c*h*w))) : Prop :=
  let patchB : Vec (N * (c*h*w)) := batchMap N (flatConvStride4 Wst psb) x
  let cotPatchB : Vec (N * (c*h*w)) := batchMapAux N (stemCotPatch ε Wst psb psng) x dyStem
  (∀ k : Fin c,
      den (SHlo.veclnGammaGradB (N := N) (R := h*w) (D := c) xN epsStr ε
            (batchMap N (chanLNRows c h w) patchB)
            (.operand cotN (batchMap N (chanLNRows c h w) dyStem))) k
        = ∑ n : Fin N, ∑ j : Fin (c*h*w),
            pdiv (fun γ' : Vec c => chanLNTensor3 c h w ε γ' psnbt (batchSlice N (c*h*w) patchB n))
                 psng k j * batchSlice N (c*h*w) dyStem n j)
  ∧ (∀ k : Fin c,
      den (SHlo.rowDenseBiasGradB (N := N) (R := h*w) (c := c)
            (.operand cotN (batchMap N (chanLNRows c h w) dyStem))) k
        = ∑ n : Fin N, ∑ j : Fin (c*h*w),
            pdiv (fun β' : Vec c => chanLNTensor3 c h w ε psng β' (batchSlice N (c*h*w) patchB n))
                 psnbt k j * batchSlice N (c*h*w) dyStem n j)
  ∧ (∀ o : Fin c,
      den (SHlo.convBiasGradB (N := N) (ic := 3) (oc := c) (h := h) (w := w) (kH := 4) (kW := 4)
            Wst xstem psb (.operand cotN cotPatchB)) o
        = ∑ n : Fin N, ∑ j : Fin (c*h*w),
            pdiv (fun b' : Vec c =>
                    Tensor3.flatten (conv2d Wst b' (Tensor3.unflatten (batchSlice N (3*h*w) xstem n))))
                 psb o j * batchSlice N (c*h*w) cotPatchB n j)
  ∧ (∀ idx : Fin (c*3*4*4),
      den (SHlo.convStride4WeightGradB xN psb x Wst (.operand cotN cotPatchB)) idx
        = ∑ n : Fin N, ∑ j : Fin (c*h*w),
            pdiv (fun v' : Vec (c*3*4*4) =>
                    flatConvStride4 (Kernel4.unflatten v') psb
                      (batchSlice N (3*(2*(2*h))*(2*(2*w))) x n))
                 (Kernel4.flatten Wst) idx j * batchSlice N (c*h*w) cotPatchB n j)

theorem cnx_stem_ch_tiedGB (N : Nat) {c h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c)
    (x : Vec (N * (3*(2*(2*h))*(2*(2*w))))) (xstem : Vec (N * (3*h*w)))
    (dyStem : Vec (N * (c*h*w))) :
    cnxStemChTiedGB N xN epsStr cotN ε Wst psb psng psnbt x xstem dyStem := by
  unfold cnxStemChTiedGB
  intro patchB cotPatchB
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro k;   exact CnxPoCGB.chanLnGammaGradB_den xN epsStr cotN ε psnbt patchB psng dyStem k
  · intro k;   exact CnxPoCGB.chanLnBetaGradB_den cotN ε psng patchB psnbt dyStem k
  · intro o;   exact CnxPoCGB.convBGradB_den cotN Wst xstem psb cotPatchB o
  · intro idx; exact CnxPoCGB.psWGradB_den xN cotN psb x Wst cotPatchB idx

/-! ## Head — GAP → vector-LN at one row → dense, all 4 gradient nodes, batched

Stated at the LITERAL 768 for the fused file's reason: `1 * m` does not reduce at a variable `m`.
`g` is a PARAMETER — the loss cotangent arrives from `smoothedLossCotGraphDiv` in the capstone. -/

/-- **Head, tied at the batched gradient nodes.** The head-LN γ/β at the pooled row, the
    classifier weight at the LN output, the classifier bias PER EXAMPLE (`biasGradB` is the
    identity on its operand; the batch reduce is emitted text — `CnxPoCGB.headBGradB_den`). -/
def cnxHeadChTiedGB (N : Nat) {h w nC : Nat} (xN epsStr cotN dN : String) (ε : ℝ)
    (hng hnbt : Vec 768) (Wfc : Mat 768 nC) (bfc : Vec nC)
    (xhead : Vec (N * (768*h*w))) (g : Vec (N * nC)) : Prop :=
  let gapB   : Vec (N * (1*768)) := batchMap N (globalAvgPoolFlat 768 h w) xhead
  let hnB    : Vec (N * 768)     := batchMap N (rowLNVecFlat 1 768 ε hng hnbt) gapB
  let cotHnB : Vec (N * (1*768)) := batchMapAux N (headCotHn Wfc bfc) hnB g
  (∀ k : Fin 768,
      den (SHlo.veclnGammaGradB (N := N) (R := 1) (D := 768) xN epsStr ε gapB
            (.operand cotN cotHnB)) k
        = ∑ n : Fin N, ∑ o : Fin (1*768),
            pdiv (fun gv : Vec 768 =>
                    Mat.flatten (fun r =>
                      layerNormVec 768 ε gv hnbt (Mat.unflatten (batchSlice N (1*768) gapB n) r)))
                 hng k o * batchSlice N (1*768) cotHnB n o)
  ∧ (∀ k : Fin 768,
      den (SHlo.rowDenseBiasGradB (N := N) (R := 1) (c := 768) (.operand cotN cotHnB)) k
        = ∑ n : Fin N, ∑ o : Fin (1*768),
            pdiv (fun bv : Vec 768 =>
                    Mat.flatten (fun r =>
                      layerNormVec 768 ε hng bv (Mat.unflatten (batchSlice N (1*768) gapB n) r)))
                 hnbt k o * batchSlice N (1*768) cotHnB n o)
  ∧ (∀ (i : Fin 768) (j : Fin nC),
      den (SHlo.weightGradB (N := N) (m := 768) (n := nC) dN hnB (.operand cotN g))
          (finProdFinEquiv (i, j))
        = ∑ n : Fin N, ∑ k : Fin nC,
            pdiv (fun v : Vec (768 * nC) => dense (Mat.unflatten v) bfc (batchSlice N 768 hnB n))
                 (Mat.flatten Wfc) (finProdFinEquiv (i, j)) k * batchSlice N nC g n k)
  ∧ (∀ (n : Fin N) (i : Fin nC),
      batchSlice N nC (den (SHlo.biasGradB (N := N) (n := nC) (.operand cotN g))) n i
        = ∑ j : Fin nC,
            pdiv (fun b' : Vec nC => dense Wfc b' (batchSlice N 768 hnB n)) bfc i j
              * batchSlice N nC g n j)

theorem cnx_head_ch_tiedGB (N : Nat) {h w nC : Nat} (xN epsStr cotN dN : String) (ε : ℝ)
    (hng hnbt : Vec 768) (Wfc : Mat 768 nC) (bfc : Vec nC)
    (xhead : Vec (N * (768*h*w))) (g : Vec (N * nC)) :
    cnxHeadChTiedGB N xN epsStr cotN dN ε hng hnbt Wfc bfc xhead g := by
  unfold cnxHeadChTiedGB
  intro gapB hnB cotHnB
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro k;   exact CnxPoCGB.headLnGammaGradB_den xN epsStr cotN ε hnbt gapB hng cotHnB k
  · intro k;
    exact CnxPoCGB.headLnBetaGradB_den cotN ε hng
      (fun n => Mat.unflatten (batchSlice N (1*768) gapB n)) hnbt cotHnB k
  · intro i j; exact CnxPoCGB.headWGradB_den dN cotN hnB Wfc bfc g i j
  · intro n i; exact CnxPoCGB.headBGradB_den cotN Wfc (batchSlice N 768 hnB n) bfc g n i

/-! ## `@[irreducible]` wrappers — keep the 22-deep capstone thread opaque (the r34/mnv2 heartbeat lesson) -/

@[irreducible] def cnxBlockChTiedGBAt (N : Nat) {c cExp h w : Nat} (xN epsStr cotN : String)
    (ε : ℝ) (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin dyOut : Vec (N * (c*h*w))) : Prop :=
  cnxBlockChTiedGB N xN epsStr cotN ε Wdw bdw ng nbt Wex bex Wpr bpr lg xin dyOut

theorem cnx_block_ch_tiedGBAt (N : Nat) {c cExp h w : Nat} (xN epsStr cotN : String)
    (ε : ℝ) (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin dyOut : Vec (N * (c*h*w))) :
    cnxBlockChTiedGBAt N xN epsStr cotN ε Wdw bdw ng nbt Wex bex Wpr bpr lg xin dyOut := by
  unfold cnxBlockChTiedGBAt
  exact cnx_block_ch_tiedGB N xN epsStr cotN ε Wdw bdw ng nbt Wex bex Wpr bpr lg xin dyOut

@[irreducible] def cnxDownChTiedGBAt (N : Nat) {ci co h w : Nat} (xN epsStr cotN : String)
    (ε : ℝ) (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (N * (ci*(2*h)*(2*w)))) (dyOut : Vec (N * (co*h*w))) : Prop :=
  cnxDownChTiedGB N xN epsStr cotN ε dng dnbt Wd bd xin dyOut

theorem cnx_down_ch_tiedGBAt (N : Nat) {ci co h w : Nat} (xN epsStr cotN : String)
    (ε : ℝ) (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (N * (ci*(2*h)*(2*w)))) (dyOut : Vec (N * (co*h*w))) :
    cnxDownChTiedGBAt N xN epsStr cotN ε dng dnbt Wd bd xin dyOut := by
  unfold cnxDownChTiedGBAt
  exact cnx_down_ch_tiedGB N xN epsStr cotN ε dng dnbt Wd bd xin dyOut

@[irreducible] def cnxStemChTiedGBAt (N : Nat) {c h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c)
    (x : Vec (N * (3*(2*(2*h))*(2*(2*w))))) (xstem : Vec (N * (3*h*w)))
    (dyStem : Vec (N * (c*h*w))) : Prop :=
  cnxStemChTiedGB N xN epsStr cotN ε Wst psb psng psnbt x xstem dyStem

theorem cnx_stem_ch_tiedGBAt (N : Nat) {c h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c)
    (x : Vec (N * (3*(2*(2*h))*(2*(2*w))))) (xstem : Vec (N * (3*h*w)))
    (dyStem : Vec (N * (c*h*w))) :
    cnxStemChTiedGBAt N xN epsStr cotN ε Wst psb psng psnbt x xstem dyStem := by
  unfold cnxStemChTiedGBAt
  exact cnx_stem_ch_tiedGB N xN epsStr cotN ε Wst psb psng psnbt x xstem dyStem

@[irreducible] def cnxHeadChTiedGBAt (N : Nat) {h w nC : Nat} (xN epsStr cotN dN : String)
    (ε : ℝ) (hng hnbt : Vec 768) (Wfc : Mat 768 nC) (bfc : Vec nC)
    (xhead : Vec (N * (768*h*w))) (g : Vec (N * nC)) : Prop :=
  cnxHeadChTiedGB N xN epsStr cotN dN ε hng hnbt Wfc bfc xhead g

theorem cnx_head_ch_tiedGBAt (N : Nat) {h w nC : Nat} (xN epsStr cotN dN : String)
    (ε : ℝ) (hng hnbt : Vec 768) (Wfc : Mat 768 nC) (bfc : Vec nC)
    (xhead : Vec (N * (768*h*w))) (g : Vec (N * nC)) :
    cnxHeadChTiedGBAt N xN epsStr cotN dN ε hng hnbt Wfc bfc xhead g := by
  unfold cnxHeadChTiedGBAt
  exact cnx_head_ch_tiedGB N xN epsStr cotN dN ε hng hnbt Wfc bfc xhead g

/-! ## The whole-net capstone — all 182 params through the REAL batched forward + composed cotangent

The fused file's thread, lifted: block inputs are `batchMap N` of the forward prefixes, and the
backward cotangents are `batchMapAux N` of the per-example chain, composed from the smoothed loss
`g` down through the head, every block's backward with the residual fan-in `+ dyOut` at each of
the eighteen identity-skip merges, the channel-LN-back at each of the three downsamples, and the
stem LN's own back before the patchify conv's gradients. -/

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 400000 in
/-- ⭐⭐ **The whole [3,3,9,3] ConvNeXt-T train step, tied at the BATCHED index, the GRADIENT
    nodes and the SMOOTHED loss.** Threading the real channel-LN / per-channel layer-scale forward
    as `batchMap N` of the per-example prefixes, and the label-smoothed loss cotangent
    (`smoothedLossCotGraphDiv`, at a general target `t`) down through the head and every block's
    certified cotangent chain as `batchMapAux N` of the per-example chain — GELU masks, the residual
    fan-in at every identity skip, the channel-LN-back at every downsample and at the stem — the
    18 ConvNeXt blocks, the 3 downsamples, the 4×4/s4 stem with its LN and the GAP → LN → dense
    head all denote the certified batched `Σ_n` gradient. All 182 parameters, at the nodes
    `convnext_adam_train_step.mlir` and every `convnextin_*` train step emit.

    ⭐ **`N` and `nC` are binders and there is no smoothness hypothesis**: the folds are `∀ cot`
    statements instantiated at explicitly constructed cotangents, and ConvNeXt has no kink. The
    batch enters only through `batchMap`/`batchMapAux`, which is honest because no ConvNeXt op
    couples examples. ⛔ ONE REPLICA: in `convnextin_adamdp*` every gradient node feeds
    `allReduceMeanF` (an AST node since 4d piece 2; `DataParallelNode.lean` composes the
    per-replica statement with the replica mean). ⛔ Stated at the drop-free chain;
    the `*drop*` artifacts' parameter nodes are the same `*GradB` constructors (the folds are
    `∀ cot`) but their cotangent chain carries the `dropPathB` sites, which this thread does not
    name. -/
theorem cnx_net_tiedGB (N : Nat) {nC : Nat}
    (xN epsStr cotN dN aStr negAK bStr logN ohN : String) (ε α B : ℝ)
    -- stem (c=96): patchify conv + its channel-LN
    (Wst : Kernel4 96 3 4 4) (psb psng psnbt : Vec 96) (xstem : Vec (N * (3*56*56)))
    -- stage 0 (c=96, cExp=384): blocks 1,2,3
    (aW1 : DepthwiseKernel 96 7 7) (aB1 : Vec 96) (nG1 nB1 : Vec 96) (eW1 : Kernel4 384 96 1 1) (eB1 : Vec 384) (pW1 : Kernel4 96 384 1 1) (pB1 sL1 : Vec 96)
    (aW2 : DepthwiseKernel 96 7 7) (aB2 : Vec 96) (nG2 nB2 : Vec 96) (eW2 : Kernel4 384 96 1 1) (eB2 : Vec 384) (pW2 : Kernel4 96 384 1 1) (pB2 sL2 : Vec 96)
    (aW3 : DepthwiseKernel 96 7 7) (aB3 : Vec 96) (nG3 nB3 : Vec 96) (eW3 : Kernel4 384 96 1 1) (eB3 : Vec 384) (pW3 : Kernel4 96 384 1 1) (pB3 sL3 : Vec 96)
    (dG0 dT0 : Vec 96) (dW0 : Kernel4 192 96 2 2) (dB0 : Vec 192)
    -- stage 1 (c=192, cExp=768): blocks 4,5,6
    (aW4 : DepthwiseKernel 192 7 7) (aB4 : Vec 192) (nG4 nB4 : Vec 192) (eW4 : Kernel4 768 192 1 1) (eB4 : Vec 768) (pW4 : Kernel4 192 768 1 1) (pB4 sL4 : Vec 192)
    (aW5 : DepthwiseKernel 192 7 7) (aB5 : Vec 192) (nG5 nB5 : Vec 192) (eW5 : Kernel4 768 192 1 1) (eB5 : Vec 768) (pW5 : Kernel4 192 768 1 1) (pB5 sL5 : Vec 192)
    (aW6 : DepthwiseKernel 192 7 7) (aB6 : Vec 192) (nG6 nB6 : Vec 192) (eW6 : Kernel4 768 192 1 1) (eB6 : Vec 768) (pW6 : Kernel4 192 768 1 1) (pB6 sL6 : Vec 192)
    (dG1 dT1 : Vec 192) (dW1 : Kernel4 384 192 2 2) (dB1 : Vec 384)
    -- stage 2 (c=384, cExp=1536): blocks 7..15
    (aW7 : DepthwiseKernel 384 7 7) (aB7 : Vec 384) (nG7 nB7 : Vec 384) (eW7 : Kernel4 1536 384 1 1) (eB7 : Vec 1536) (pW7 : Kernel4 384 1536 1 1) (pB7 sL7 : Vec 384)
    (aW8 : DepthwiseKernel 384 7 7) (aB8 : Vec 384) (nG8 nB8 : Vec 384) (eW8 : Kernel4 1536 384 1 1) (eB8 : Vec 1536) (pW8 : Kernel4 384 1536 1 1) (pB8 sL8 : Vec 384)
    (aW9 : DepthwiseKernel 384 7 7) (aB9 : Vec 384) (nG9 nB9 : Vec 384) (eW9 : Kernel4 1536 384 1 1) (eB9 : Vec 1536) (pW9 : Kernel4 384 1536 1 1) (pB9 sL9 : Vec 384)
    (aW10 : DepthwiseKernel 384 7 7) (aB10 : Vec 384) (nG10 nB10 : Vec 384) (eW10 : Kernel4 1536 384 1 1) (eB10 : Vec 1536) (pW10 : Kernel4 384 1536 1 1) (pB10 sL10 : Vec 384)
    (aW11 : DepthwiseKernel 384 7 7) (aB11 : Vec 384) (nG11 nB11 : Vec 384) (eW11 : Kernel4 1536 384 1 1) (eB11 : Vec 1536) (pW11 : Kernel4 384 1536 1 1) (pB11 sL11 : Vec 384)
    (aW12 : DepthwiseKernel 384 7 7) (aB12 : Vec 384) (nG12 nB12 : Vec 384) (eW12 : Kernel4 1536 384 1 1) (eB12 : Vec 1536) (pW12 : Kernel4 384 1536 1 1) (pB12 sL12 : Vec 384)
    (aW13 : DepthwiseKernel 384 7 7) (aB13 : Vec 384) (nG13 nB13 : Vec 384) (eW13 : Kernel4 1536 384 1 1) (eB13 : Vec 1536) (pW13 : Kernel4 384 1536 1 1) (pB13 sL13 : Vec 384)
    (aW14 : DepthwiseKernel 384 7 7) (aB14 : Vec 384) (nG14 nB14 : Vec 384) (eW14 : Kernel4 1536 384 1 1) (eB14 : Vec 1536) (pW14 : Kernel4 384 1536 1 1) (pB14 sL14 : Vec 384)
    (aW15 : DepthwiseKernel 384 7 7) (aB15 : Vec 384) (nG15 nB15 : Vec 384) (eW15 : Kernel4 1536 384 1 1) (eB15 : Vec 1536) (pW15 : Kernel4 384 1536 1 1) (pB15 sL15 : Vec 384)
    (dG2 dT2 : Vec 384) (dW2 : Kernel4 768 384 2 2) (dB2 : Vec 768)
    -- stage 3 (c=768, cExp=3072): blocks 16,17,18
    (aW16 : DepthwiseKernel 768 7 7) (aB16 : Vec 768) (nG16 nB16 : Vec 768) (eW16 : Kernel4 3072 768 1 1) (eB16 : Vec 3072) (pW16 : Kernel4 768 3072 1 1) (pB16 sL16 : Vec 768)
    (aW17 : DepthwiseKernel 768 7 7) (aB17 : Vec 768) (nG17 nB17 : Vec 768) (eW17 : Kernel4 3072 768 1 1) (eB17 : Vec 3072) (pW17 : Kernel4 768 3072 1 1) (pB17 sL17 : Vec 768)
    (aW18 : DepthwiseKernel 768 7 7) (aB18 : Vec 768) (nG18 nB18 : Vec 768) (eW18 : Kernel4 3072 768 1 1) (eB18 : Vec 3072) (pW18 : Kernel4 768 3072 1 1) (pB18 sL18 : Vec 768)
    -- head
    (hG hT : Vec 768) (Wfc : Mat 768 nC) (bfc : Vec nC)
    (x : Vec (N * (3*224*224))) (t : Vec (N * nC)) :
    -- forward block inputs (the prefixes of the committed render's forward)
    let ib1 : Vec (N * (96*56*56)) := batchMap N (cnxStemFwdO (h := 56) (w := 56) ε Wst psb psng psnbt) x
    let ib2 : Vec (N * (96*56*56)) := batchMap N (cnxBlockFwdChO ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 ) ib1
    let ib3 : Vec (N * (96*56*56)) := batchMap N (cnxBlockFwdChO ε aW2 aB2 nG2 nB2 eW2 eB2 pW2 pB2 sL2 ) ib2
    let ibD0 : Vec (N * (96*56*56)) := batchMap N (cnxBlockFwdChO ε aW3 aB3 nG3 nB3 eW3 eB3 pW3 pB3 sL3 ) ib3
    let ib4 : Vec (N * (192*28*28)) := batchMap N (cnxDownFwdChO (h := 28) (w := 28) ε dG0 dT0 dW0 dB0) ibD0
    let ib5 : Vec (N * (192*28*28)) := batchMap N (cnxBlockFwdChO ε aW4 aB4 nG4 nB4 eW4 eB4 pW4 pB4 sL4 ) ib4
    let ib6 : Vec (N * (192*28*28)) := batchMap N (cnxBlockFwdChO ε aW5 aB5 nG5 nB5 eW5 eB5 pW5 pB5 sL5 ) ib5
    let ibD1 : Vec (N * (192*28*28)) := batchMap N (cnxBlockFwdChO ε aW6 aB6 nG6 nB6 eW6 eB6 pW6 pB6 sL6 ) ib6
    let ib7 : Vec (N * (384*14*14)) := batchMap N (cnxDownFwdChO (h := 14) (w := 14) ε dG1 dT1 dW1 dB1) ibD1
    let ib8 : Vec (N * (384*14*14)) := batchMap N (cnxBlockFwdChO ε aW7 aB7 nG7 nB7 eW7 eB7 pW7 pB7 sL7 ) ib7
    let ib9 : Vec (N * (384*14*14)) := batchMap N (cnxBlockFwdChO ε aW8 aB8 nG8 nB8 eW8 eB8 pW8 pB8 sL8 ) ib8
    let ib10 : Vec (N * (384*14*14)) := batchMap N (cnxBlockFwdChO ε aW9 aB9 nG9 nB9 eW9 eB9 pW9 pB9 sL9 ) ib9
    let ib11 : Vec (N * (384*14*14)) := batchMap N (cnxBlockFwdChO ε aW10 aB10 nG10 nB10 eW10 eB10 pW10 pB10 sL10 ) ib10
    let ib12 : Vec (N * (384*14*14)) := batchMap N (cnxBlockFwdChO ε aW11 aB11 nG11 nB11 eW11 eB11 pW11 pB11 sL11 ) ib11
    let ib13 : Vec (N * (384*14*14)) := batchMap N (cnxBlockFwdChO ε aW12 aB12 nG12 nB12 eW12 eB12 pW12 pB12 sL12 ) ib12
    let ib14 : Vec (N * (384*14*14)) := batchMap N (cnxBlockFwdChO ε aW13 aB13 nG13 nB13 eW13 eB13 pW13 pB13 sL13 ) ib13
    let ib15 : Vec (N * (384*14*14)) := batchMap N (cnxBlockFwdChO ε aW14 aB14 nG14 nB14 eW14 eB14 pW14 pB14 sL14 ) ib14
    let ibD2 : Vec (N * (384*14*14)) := batchMap N (cnxBlockFwdChO ε aW15 aB15 nG15 nB15 eW15 eB15 pW15 pB15 sL15 ) ib15
    let ib16 : Vec (N * (768*7*7)) := batchMap N (cnxDownFwdChO (h := 7) (w := 7) ε dG2 dT2 dW2 dB2) ibD2
    let ib17 : Vec (N * (768*7*7)) := batchMap N (cnxBlockFwdChO ε aW16 aB16 nG16 nB16 eW16 eB16 pW16 pB16 sL16 ) ib16
    let ib18 : Vec (N * (768*7*7)) := batchMap N (cnxBlockFwdChO ε aW17 aB17 nG17 nB17 eW17 eB17 pW17 pB17 sL17 ) ib17
    let xhead : Vec (N * (768*7*7)) := batchMap N (cnxBlockFwdChO ε aW18 aB18 nG18 nB18 eW18 eB18 pW18 pB18 sL18 ) ib18
    -- head forward + the SMOOTHED loss cotangent, at a general target `t`
    let gapB    : Vec (N * (1*768)) := batchMap N (globalAvgPoolFlat 768 7 7) xhead
    let hnB     : Vec (N * 768)     := batchMap N (rowLNVecFlat 1 768 ε hG hT) gapB
    let logitsB : Vec (N * nC)      := batchMap N (dense Wfc bfc) hnB
    let g       : Vec (N * nC)      :=
      den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN logitsB t)
    -- backward cotangents (composed from the loss; residual fan-in at each skip, LN-back at each
    -- downsample and at the stem)
    let dyO18 : Vec (N * (768*7*7)) := batchMapAux N (cnxHeadDyXheadChN (h := 7) (w := 7) ε hG hT Wfc bfc) xhead g
    let dyO17 : Vec (N * (768*7*7)) := batchMapAux N (cnxBlockCotInChAt ε aW18 aB18 nG18 nB18 eW18 eB18 pW18 pB18 sL18 ) ib18 dyO18
    let dyO16 : Vec (N * (768*7*7)) := batchMapAux N (cnxBlockCotInChAt ε aW17 aB17 nG17 nB17 eW17 eB17 pW17 pB17 sL17 ) ib17 dyO17
    let dyD2 : Vec (N * (768*7*7)) := batchMapAux N (cnxBlockCotInChAt ε aW16 aB16 nG16 nB16 eW16 eB16 pW16 pB16 sL16 ) ib16 dyO16
    let dyO15 : Vec (N * (384*14*14)) := batchMapAux N (cnxDownCotInChAt (h := 7) (w := 7) ε dG2 dT2 dW2 dB2) ibD2 dyD2
    let dyO14 : Vec (N * (384*14*14)) := batchMapAux N (cnxBlockCotInChAt ε aW15 aB15 nG15 nB15 eW15 eB15 pW15 pB15 sL15 ) ib15 dyO15
    let dyO13 : Vec (N * (384*14*14)) := batchMapAux N (cnxBlockCotInChAt ε aW14 aB14 nG14 nB14 eW14 eB14 pW14 pB14 sL14 ) ib14 dyO14
    let dyO12 : Vec (N * (384*14*14)) := batchMapAux N (cnxBlockCotInChAt ε aW13 aB13 nG13 nB13 eW13 eB13 pW13 pB13 sL13 ) ib13 dyO13
    let dyO11 : Vec (N * (384*14*14)) := batchMapAux N (cnxBlockCotInChAt ε aW12 aB12 nG12 nB12 eW12 eB12 pW12 pB12 sL12 ) ib12 dyO12
    let dyO10 : Vec (N * (384*14*14)) := batchMapAux N (cnxBlockCotInChAt ε aW11 aB11 nG11 nB11 eW11 eB11 pW11 pB11 sL11 ) ib11 dyO11
    let dyO9 : Vec (N * (384*14*14)) := batchMapAux N (cnxBlockCotInChAt ε aW10 aB10 nG10 nB10 eW10 eB10 pW10 pB10 sL10 ) ib10 dyO10
    let dyO8 : Vec (N * (384*14*14)) := batchMapAux N (cnxBlockCotInChAt ε aW9 aB9 nG9 nB9 eW9 eB9 pW9 pB9 sL9 ) ib9 dyO9
    let dyO7 : Vec (N * (384*14*14)) := batchMapAux N (cnxBlockCotInChAt ε aW8 aB8 nG8 nB8 eW8 eB8 pW8 pB8 sL8 ) ib8 dyO8
    let dyD1 : Vec (N * (384*14*14)) := batchMapAux N (cnxBlockCotInChAt ε aW7 aB7 nG7 nB7 eW7 eB7 pW7 pB7 sL7 ) ib7 dyO7
    let dyO6 : Vec (N * (192*28*28)) := batchMapAux N (cnxDownCotInChAt (h := 14) (w := 14) ε dG1 dT1 dW1 dB1) ibD1 dyD1
    let dyO5 : Vec (N * (192*28*28)) := batchMapAux N (cnxBlockCotInChAt ε aW6 aB6 nG6 nB6 eW6 eB6 pW6 pB6 sL6 ) ib6 dyO6
    let dyO4 : Vec (N * (192*28*28)) := batchMapAux N (cnxBlockCotInChAt ε aW5 aB5 nG5 nB5 eW5 eB5 pW5 pB5 sL5 ) ib5 dyO5
    let dyD0 : Vec (N * (192*28*28)) := batchMapAux N (cnxBlockCotInChAt ε aW4 aB4 nG4 nB4 eW4 eB4 pW4 pB4 sL4 ) ib4 dyO4
    let dyO3 : Vec (N * (96*56*56)) := batchMapAux N (cnxDownCotInChAt (h := 28) (w := 28) ε dG0 dT0 dW0 dB0) ibD0 dyD0
    let dyO2 : Vec (N * (96*56*56)) := batchMapAux N (cnxBlockCotInChAt ε aW3 aB3 nG3 nB3 eW3 eB3 pW3 pB3 sL3 ) ib3 dyO3
    let dyO1 : Vec (N * (96*56*56)) := batchMapAux N (cnxBlockCotInChAt ε aW2 aB2 nG2 nB2 eW2 eB2 pW2 pB2 sL2 ) ib2 dyO2
    let dyStem : Vec (N * (96*56*56)) := batchMapAux N (cnxBlockCotInChAt ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 ) ib1 dyO1
    -- the stem, every block, every downsample, the head, the dense total-loss fold + loss cot
    cnxStemChTiedGBAt N xN epsStr cotN ε Wst psb psng psnbt x xstem dyStem
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 ib1 dyO1
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW2 aB2 nG2 nB2 eW2 eB2 pW2 pB2 sL2 ib2 dyO2
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW3 aB3 nG3 nB3 eW3 eB3 pW3 pB3 sL3 ib3 dyO3
  ∧ cnxDownChTiedGBAt N xN epsStr cotN ε dG0 dT0 dW0 dB0 ibD0 dyD0
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW4 aB4 nG4 nB4 eW4 eB4 pW4 pB4 sL4 ib4 dyO4
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW5 aB5 nG5 nB5 eW5 eB5 pW5 pB5 sL5 ib5 dyO5
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW6 aB6 nG6 nB6 eW6 eB6 pW6 pB6 sL6 ib6 dyO6
  ∧ cnxDownChTiedGBAt N xN epsStr cotN ε dG1 dT1 dW1 dB1 ibD1 dyD1
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW7 aB7 nG7 nB7 eW7 eB7 pW7 pB7 sL7 ib7 dyO7
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW8 aB8 nG8 nB8 eW8 eB8 pW8 pB8 sL8 ib8 dyO8
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW9 aB9 nG9 nB9 eW9 eB9 pW9 pB9 sL9 ib9 dyO9
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW10 aB10 nG10 nB10 eW10 eB10 pW10 pB10 sL10 ib10 dyO10
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW11 aB11 nG11 nB11 eW11 eB11 pW11 pB11 sL11 ib11 dyO11
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW12 aB12 nG12 nB12 eW12 eB12 pW12 pB12 sL12 ib12 dyO12
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW13 aB13 nG13 nB13 eW13 eB13 pW13 pB13 sL13 ib13 dyO13
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW14 aB14 nG14 nB14 eW14 eB14 pW14 pB14 sL14 ib14 dyO14
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW15 aB15 nG15 nB15 eW15 eB15 pW15 pB15 sL15 ib15 dyO15
  ∧ cnxDownChTiedGBAt N xN epsStr cotN ε dG2 dT2 dW2 dB2 ibD2 dyD2
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW16 aB16 nG16 nB16 eW16 eB16 pW16 pB16 sL16 ib16 dyO16
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW17 aB17 nG17 nB17 eW17 eB17 pW17 pB17 sL17 ib17 dyO17
  ∧ cnxBlockChTiedGBAt N xN epsStr cotN ε aW18 aB18 nG18 nB18 eW18 eB18 pW18 pB18 sL18 ib18 dyO18
  ∧ cnxHeadChTiedGBAt N xN epsStr cotN dN ε hG hT Wfc bfc xhead g := by
  intro ib1 ib2 ib3 ibD0 ib4 ib5 ib6 ibD1 ib7 ib8 ib9 ib10 ib11 ib12 ib13 ib14 ib15 ibD2 ib16 ib17 ib18 xhead gapB hnB logitsB g dyO18 dyO17 dyO16 dyD2 dyO15 dyO14 dyO13 dyO12 dyO11 dyO10 dyO9 dyO8 dyO7 dyD1 dyO6 dyO5 dyO4 dyD0 dyO3 dyO2 dyO1 dyStem
  refine ⟨cnx_stem_ch_tiedGBAt N xN epsStr cotN ε Wst psb psng psnbt x xstem dyStem,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 ib1 dyO1
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW2 aB2 nG2 nB2 eW2 eB2 pW2 pB2 sL2 ib2 dyO2
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW3 aB3 nG3 nB3 eW3 eB3 pW3 pB3 sL3 ib3 dyO3
  · exact cnx_down_ch_tiedGBAt N xN epsStr cotN ε dG0 dT0 dW0 dB0 ibD0 dyD0
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW4 aB4 nG4 nB4 eW4 eB4 pW4 pB4 sL4 ib4 dyO4
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW5 aB5 nG5 nB5 eW5 eB5 pW5 pB5 sL5 ib5 dyO5
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW6 aB6 nG6 nB6 eW6 eB6 pW6 pB6 sL6 ib6 dyO6
  · exact cnx_down_ch_tiedGBAt N xN epsStr cotN ε dG1 dT1 dW1 dB1 ibD1 dyD1
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW7 aB7 nG7 nB7 eW7 eB7 pW7 pB7 sL7 ib7 dyO7
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW8 aB8 nG8 nB8 eW8 eB8 pW8 pB8 sL8 ib8 dyO8
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW9 aB9 nG9 nB9 eW9 eB9 pW9 pB9 sL9 ib9 dyO9
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW10 aB10 nG10 nB10 eW10 eB10 pW10 pB10 sL10 ib10 dyO10
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW11 aB11 nG11 nB11 eW11 eB11 pW11 pB11 sL11 ib11 dyO11
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW12 aB12 nG12 nB12 eW12 eB12 pW12 pB12 sL12 ib12 dyO12
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW13 aB13 nG13 nB13 eW13 eB13 pW13 pB13 sL13 ib13 dyO13
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW14 aB14 nG14 nB14 eW14 eB14 pW14 pB14 sL14 ib14 dyO14
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW15 aB15 nG15 nB15 eW15 eB15 pW15 pB15 sL15 ib15 dyO15
  · exact cnx_down_ch_tiedGBAt N xN epsStr cotN ε dG2 dT2 dW2 dB2 ibD2 dyD2
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW16 aB16 nG16 nB16 eW16 eB16 pW16 pB16 sL16 ib16 dyO16
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW17 aB17 nG17 nB17 eW17 eB17 pW17 pB17 sL17 ib17 dyO17
  · exact cnx_block_ch_tiedGBAt N xN epsStr cotN ε aW18 aB18 nG18 nB18 eW18 eB18 pW18 pB18 sL18 ib18 dyO18
  · exact cnx_head_ch_tiedGBAt N xN epsStr cotN dN ε hG hT Wfc bfc xhead g

end Proofs.CnxTiePoCGB
