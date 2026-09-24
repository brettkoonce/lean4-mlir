import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFoldGB
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtStepTie
import LeanMlir.Proofs.Foundation.SmoothedLossCot

/-! # ConvNeXt-T's T3 §1a TIE at the BATCHED index, the UN-FUSED gradient and the SMOOTHED loss

`ConvNeXtStepTie.lean` ties all 182 parameters of the SGD-inline `convnext_train_step.mlir`: each
fused `θ − lr·g` op `den`s to the certified step at the cotangent the emitted backward chain
delivers, per example, at a hard label. This file is that statement re-pointed along the THREE
axes 4b and 4c left open (`planning/archive/proofs_tier_to_paper_nets.md` §4b, §4c-quater) — and unlike
EfficientNet-B0's (`EfficientNetStepTieG.lean`, two axes), ConvNeXt's per-example capstone was at
a single image with the batch outside the AST, so the index is the third.

⭐ **Axis 1 — the OPTIMIZER FORM.** Every conjunct is at the RAW gradient node (`*GradB`), which is
what `convnext_adam_train_step.mlir` and every `convnextin_*` artifact emit; the fused op appears
only in the SGD-inline file. One statement covers AdamW, the `wx`/`clip` variants, EMA and the
data-parallel twins, because they all consume this node. `ConvNeXtFoldGB.lean` (§4c-quater)
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

Every activation, every Jacobian witness and every chain cotangent is `ConvNeXtStepTie.lean`'s,
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
open Proofs.CnxTiePoC (cnxStemFwdO cnxBlockFwdChO cnxDownFwdChO cnxBlockCotInChAt cnxDownCotInChAt
  CnxTieWeights)

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
  ResNet34PoCB.DepthwiseWTiedB N h w xN cotN bdw xin Wdw cotDB
  ∧ ResNet34PoCB.DepthwiseBTiedB N h w cotN Wdw xin bdw cotDB
  -- channel-LN γ/β  (cot = cotNB, LN input = dB; the ops see both as their batched [h·w, c] views)
  ∧ CnxPoCGB.ChanLNGammaTiedB N h w xN epsStr cotN ε nbt dB ng cotNB
  ∧ CnxPoCGB.ChanLNBetaTiedB N h w cotN ε ng dB nbt cotNB
  -- expand 1×1 conv (c → cExp) W/b  (cot = cotEB, conv input = nlB)
  ∧ ResNet34PoCB.ConvWTiedB N h w xN cotN bex nlB Wex cotEB
  ∧ ResNet34PoCB.ConvBTiedB N h w cotN Wex nlB bex cotEB
  -- project 1×1 conv (cExp → c) W/b  (cot = cotPB, conv input = gB)
  ∧ ResNet34PoCB.ConvWTiedB N h w xN cotN bpr gB Wpr cotPB
  ∧ ResNet34PoCB.ConvBTiedB N h w cotN Wpr gB bpr cotPB
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
  · intro idx; exact EnetPoCG.depthwiseWGradB_den xN cotN bdw xin Wdw cotDB idx
  · intro o;   exact Mnv2PaperPoCG.depthwiseBGradB_den cotN Wdw xin bdw cotDB o
  · intro k;   exact CnxPoCGB.chanLnGammaGradB_den xN epsStr cotN ε nbt dB ng cotNB k
  · intro k;   exact CnxPoCGB.chanLnBetaGradB_den cotN ε ng dB nbt cotNB k
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN bex nlB Wex cotEB idx
  · intro o;   exact ResNet34PoCB.convBGradB_den cotN Wex nlB bex cotEB o
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN bpr gB Wpr cotPB idx
  · intro o;   exact ResNet34PoCB.convBGradB_den cotN Wpr gB bpr cotPB o
  · intro cc;  exact CnxPoCGB.layerScaleChGammaGradB_den xN cotN pB lg dyOut cc

/-! ## Downsample — channel-LN → 2×2/s2 conv, all 4 gradient nodes, batched -/

/-- **Downsample, tied at the batched gradient nodes.** Channel-LN γ/β at the `ci·(2h)·(2w)` input
    grid, plus the strided conv's weight and bias. -/
def cnxDownChTiedGB (N : Nat) {ci co h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (N * (ci*(2*h)*(2*w)))) (dyOut : Vec (N * (co*h*w))) : Prop :=
  let nB : Vec (N * (ci*(2*h)*(2*w))) := batchMap N (chanLNTensor3 ci (2*h) (2*w) ε dng dnbt) xin
  let cotNB : Vec (N * (ci*(2*h)*(2*w))) := batchMapAux N (dnCotN ε dng dnbt Wd bd) xin dyOut
  CnxPoCGB.ChanLNGammaTiedB N (2 * h) (2 * w) xN epsStr cotN ε dnbt xin dng cotNB
  ∧ CnxPoCGB.ChanLNBetaTiedB N (2 * h) (2 * w) cotN ε dng xin dnbt cotNB
  ∧ ResNet34PoCB.ConvStridedWTiedB N h w xN cotN bd nB Wd dyOut
  ∧ ResNet34PoCB.ConvStridedBTiedB N h w cotN Wd nB bd dyOut

theorem cnx_down_ch_tiedGB (N : Nat) {ci co h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (N * (ci*(2*h)*(2*w)))) (dyOut : Vec (N * (co*h*w))) :
    cnxDownChTiedGB N xN epsStr cotN ε dng dnbt Wd bd xin dyOut := by
  unfold cnxDownChTiedGB
  intro nB cotNB
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro k;   exact CnxPoCGB.chanLnGammaGradB_den xN epsStr cotN ε dnbt xin dng cotNB k
  · intro k;   exact CnxPoCGB.chanLnBetaGradB_den cotN ε dng xin dnbt cotNB k
  · intro idx; exact ResNet34PoCB.convStridedWGradB_den xN cotN bd nB Wd dyOut idx
  · intro o;   exact ResNet34PoCB.convStridedBGradB_den cotN Wd nB bd dyOut o

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
  CnxPoCGB.ChanLNGammaTiedB N h w xN epsStr cotN ε psnbt patchB psng dyStem
  ∧ CnxPoCGB.ChanLNBetaTiedB N h w cotN ε psng patchB psnbt dyStem
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
  · intro o;   exact ResNet34PoCB.convBGradB_den cotN Wst xstem psb cotPatchB o
  · intro idx; exact CnxPoCGB.psWGradB_den xN cotN psb x Wst cotPatchB idx

/-! ## Head — GAP → vector-LN at one row → dense, all 4 gradient nodes, batched

Stated at the LITERAL 768 for the fused file's reason: `1 * m` does not reduce at a variable `m`.
`g` is a PARAMETER — the loss cotangent arrives from `smoothedLossCotGraphDiv` in the capstone. -/

/-- **Head, tied at the batched gradient nodes.** The head-LN γ/β at the pooled row, the
    classifier weight at the LN output, the classifier bias PER EXAMPLE (`biasGradB` is the
    identity on its operand; the batch reduce is emitted text — `ViTPoCGB.headBGradB_den`). -/
def cnxHeadChTiedGB (N : Nat) {h w nC : Nat} (xN epsStr cotN dN : String) (ε : ℝ)
    (hng hnbt : Vec 768) (Wfc : Mat 768 nC) (bfc : Vec nC)
    (xhead : Vec (N * (768*h*w))) (g : Vec (N * nC)) : Prop :=
  let gapB   : Vec (N * (1*768)) := batchMap N (globalAvgPoolFlat 768 h w) xhead
  let hnB    : Vec (N * 768)     := batchMap N (rowLNVecFlat 1 768 ε hng hnbt) gapB
  let cotHnB : Vec (N * (1*768)) := batchMapAux N (headCotHn Wfc bfc) hnB g
  ViTPoCGB.VecLNGammaTiedB N 1 xN epsStr cotN ε hnbt gapB hng cotHnB
  ∧ ViTPoCGB.VecLNBetaTiedB N 1 cotN ε hng gapB hnbt cotHnB
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
  · intro k;   exact ViTPoCGB.veclnGammaGradB_den xN epsStr cotN ε hnbt gapB hng cotHnB k
  · intro k;
    exact ViTPoCGB.rowDenseBiasGradB_den_lnbeta cotN ε hng
      (fun n => Mat.unflatten (batchSlice N (1*768) gapB n)) hnbt cotHnB k
  · intro i j; exact ViTPoCGB.headWGradB_den dN cotN hnB Wfc bfc g i j
  · intro n i; exact ViTPoCGB.headBGradB_den cotN Wfc (batchSlice N 768 hnB n) bfc g n i

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

/-- The block's batched tie (`cnxBlockChTiedGBAt`), over its `CnxTieBlk` record. -/
abbrev _root_.Proofs.CnxTiePoC.CnxTieBlk.TiedGB {c cExp h w : Nat} (p : CnxTiePoC.CnxTieBlk c cExp)
    (N : Nat) (xN epsStr cotN : String) (ε : ℝ) (xin dyOut : Vec (N * (c*h*w))) : Prop :=
  cnxBlockChTiedGBAt N xN epsStr cotN ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL xin dyOut

theorem _root_.Proofs.CnxTiePoC.CnxTieBlk.tiedGB {c cExp h w : Nat} (p : CnxTiePoC.CnxTieBlk c cExp)
    (N : Nat) (xN epsStr cotN : String) (ε : ℝ) (xin dyOut : Vec (N * (c*h*w))) :
    p.TiedGB N xN epsStr cotN ε xin dyOut :=
  cnx_block_ch_tiedGBAt N xN epsStr cotN ε _ _ _ _ _ _ _ _ _ xin dyOut

/-- The downsample's batched tie (`cnxDownChTiedGBAt`), over its `CnxTieDown` record. -/
abbrev _root_.Proofs.CnxTiePoC.CnxTieDown.TiedGB {ci co h w : Nat} (p : CnxTiePoC.CnxTieDown ci co)
    (N : Nat) (xN epsStr cotN : String) (ε : ℝ) (xin : Vec (N * (ci*(2*h)*(2*w))))
    (dyOut : Vec (N * (co*h*w))) : Prop :=
  cnxDownChTiedGBAt N xN epsStr cotN ε p.G p.T p.W p.B xin dyOut

theorem _root_.Proofs.CnxTiePoC.CnxTieDown.tiedGB {ci co h w : Nat} (p : CnxTiePoC.CnxTieDown ci co)
    (N : Nat) (xN epsStr cotN : String) (ε : ℝ) (xin : Vec (N * (ci*(2*h)*(2*w))))
    (dyOut : Vec (N * (co*h*w))) : p.TiedGB N xN epsStr cotN ε xin dyOut :=
  cnx_down_ch_tiedGBAt N xN epsStr cotN ε _ _ _ _ xin dyOut

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
    (w : CnxTieWeights nC) (xstem : Vec (N * (3*56*56)))
    (x : Vec (N * (3*224*224))) (t : Vec (N * nC)) :
    -- forward block inputs (the prefixes of the committed render's forward)
    let ib1 : Vec (N * (96*56*56)) := batchMap N (cnxStemFwdO (h := 56) (w := 56) ε w.sW w.sb w.sγ w.sβ) x
    let ib2 : Vec (N * (96*56*56)) := batchMap N (w.b1.fwdO ε) ib1
    let ib3 : Vec (N * (96*56*56)) := batchMap N (w.b2.fwdO ε) ib2
    let ibD0 : Vec (N * (96*56*56)) := batchMap N (w.b3.fwdO ε) ib3
    let ib4 : Vec (N * (192*28*28)) := batchMap N (w.d0.fwdO (h := 28) (w := 28) ε) ibD0
    let ib5 : Vec (N * (192*28*28)) := batchMap N (w.b4.fwdO ε) ib4
    let ib6 : Vec (N * (192*28*28)) := batchMap N (w.b5.fwdO ε) ib5
    let ibD1 : Vec (N * (192*28*28)) := batchMap N (w.b6.fwdO ε) ib6
    let ib7 : Vec (N * (384*14*14)) := batchMap N (w.d1.fwdO (h := 14) (w := 14) ε) ibD1
    let ib8 : Vec (N * (384*14*14)) := batchMap N (w.b7.fwdO ε) ib7
    let ib9 : Vec (N * (384*14*14)) := batchMap N (w.b8.fwdO ε) ib8
    let ib10 : Vec (N * (384*14*14)) := batchMap N (w.b9.fwdO ε) ib9
    let ib11 : Vec (N * (384*14*14)) := batchMap N (w.b10.fwdO ε) ib10
    let ib12 : Vec (N * (384*14*14)) := batchMap N (w.b11.fwdO ε) ib11
    let ib13 : Vec (N * (384*14*14)) := batchMap N (w.b12.fwdO ε) ib12
    let ib14 : Vec (N * (384*14*14)) := batchMap N (w.b13.fwdO ε) ib13
    let ib15 : Vec (N * (384*14*14)) := batchMap N (w.b14.fwdO ε) ib14
    let ibD2 : Vec (N * (384*14*14)) := batchMap N (w.b15.fwdO ε) ib15
    let ib16 : Vec (N * (768*7*7)) := batchMap N (w.d2.fwdO (h := 7) (w := 7) ε) ibD2
    let ib17 : Vec (N * (768*7*7)) := batchMap N (w.b16.fwdO ε) ib16
    let ib18 : Vec (N * (768*7*7)) := batchMap N (w.b17.fwdO ε) ib17
    let xhead : Vec (N * (768*7*7)) := batchMap N (w.b18.fwdO ε) ib18
    -- head forward + the SMOOTHED loss cotangent, at a general target `t`
    let gapB    : Vec (N * (1*768)) := batchMap N (globalAvgPoolFlat 768 7 7) xhead
    let hnB     : Vec (N * 768)     := batchMap N (rowLNVecFlat 1 768 ε w.hG w.hT) gapB
    let logitsB : Vec (N * nC)      := batchMap N (dense w.Wfc w.bfc) hnB
    let g       : Vec (N * nC)      :=
      den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN logitsB t)
    -- backward cotangents (composed from the loss; residual fan-in at each skip, LN-back at each
    -- downsample and at the stem)
    let dyO18 : Vec (N * (768*7*7)) := batchMapAux N (cnxHeadDyXheadChN (h := 7) (w := 7) ε w.hG w.hT w.Wfc w.bfc) xhead g
    let dyO17 : Vec (N * (768*7*7)) := batchMapAux N (w.b18.cotIn ε) ib18 dyO18
    let dyO16 : Vec (N * (768*7*7)) := batchMapAux N (w.b17.cotIn ε) ib17 dyO17
    let dyD2 : Vec (N * (768*7*7)) := batchMapAux N (w.b16.cotIn ε) ib16 dyO16
    let dyO15 : Vec (N * (384*14*14)) := batchMapAux N (w.d2.cotIn (h := 7) (w := 7) ε) ibD2 dyD2
    let dyO14 : Vec (N * (384*14*14)) := batchMapAux N (w.b15.cotIn ε) ib15 dyO15
    let dyO13 : Vec (N * (384*14*14)) := batchMapAux N (w.b14.cotIn ε) ib14 dyO14
    let dyO12 : Vec (N * (384*14*14)) := batchMapAux N (w.b13.cotIn ε) ib13 dyO13
    let dyO11 : Vec (N * (384*14*14)) := batchMapAux N (w.b12.cotIn ε) ib12 dyO12
    let dyO10 : Vec (N * (384*14*14)) := batchMapAux N (w.b11.cotIn ε) ib11 dyO11
    let dyO9 : Vec (N * (384*14*14)) := batchMapAux N (w.b10.cotIn ε) ib10 dyO10
    let dyO8 : Vec (N * (384*14*14)) := batchMapAux N (w.b9.cotIn ε) ib9 dyO9
    let dyO7 : Vec (N * (384*14*14)) := batchMapAux N (w.b8.cotIn ε) ib8 dyO8
    let dyD1 : Vec (N * (384*14*14)) := batchMapAux N (w.b7.cotIn ε) ib7 dyO7
    let dyO6 : Vec (N * (192*28*28)) := batchMapAux N (w.d1.cotIn (h := 14) (w := 14) ε) ibD1 dyD1
    let dyO5 : Vec (N * (192*28*28)) := batchMapAux N (w.b6.cotIn ε) ib6 dyO6
    let dyO4 : Vec (N * (192*28*28)) := batchMapAux N (w.b5.cotIn ε) ib5 dyO5
    let dyD0 : Vec (N * (192*28*28)) := batchMapAux N (w.b4.cotIn ε) ib4 dyO4
    let dyO3 : Vec (N * (96*56*56)) := batchMapAux N (w.d0.cotIn (h := 28) (w := 28) ε) ibD0 dyD0
    let dyO2 : Vec (N * (96*56*56)) := batchMapAux N (w.b3.cotIn ε) ib3 dyO3
    let dyO1 : Vec (N * (96*56*56)) := batchMapAux N (w.b2.cotIn ε) ib2 dyO2
    let dyStem : Vec (N * (96*56*56)) := batchMapAux N (w.b1.cotIn ε) ib1 dyO1
    -- the stem, every block, every downsample, the head, the dense total-loss fold + loss cot
    cnxStemChTiedGBAt N xN epsStr cotN ε w.sW w.sb w.sγ w.sβ x xstem dyStem
  ∧ w.b1.TiedGB N xN epsStr cotN ε ib1 dyO1
  ∧ w.b2.TiedGB N xN epsStr cotN ε ib2 dyO2
  ∧ w.b3.TiedGB N xN epsStr cotN ε ib3 dyO3
  ∧ w.d0.TiedGB N xN epsStr cotN ε ibD0 dyD0
  ∧ w.b4.TiedGB N xN epsStr cotN ε ib4 dyO4
  ∧ w.b5.TiedGB N xN epsStr cotN ε ib5 dyO5
  ∧ w.b6.TiedGB N xN epsStr cotN ε ib6 dyO6
  ∧ w.d1.TiedGB N xN epsStr cotN ε ibD1 dyD1
  ∧ w.b7.TiedGB N xN epsStr cotN ε ib7 dyO7
  ∧ w.b8.TiedGB N xN epsStr cotN ε ib8 dyO8
  ∧ w.b9.TiedGB N xN epsStr cotN ε ib9 dyO9
  ∧ w.b10.TiedGB N xN epsStr cotN ε ib10 dyO10
  ∧ w.b11.TiedGB N xN epsStr cotN ε ib11 dyO11
  ∧ w.b12.TiedGB N xN epsStr cotN ε ib12 dyO12
  ∧ w.b13.TiedGB N xN epsStr cotN ε ib13 dyO13
  ∧ w.b14.TiedGB N xN epsStr cotN ε ib14 dyO14
  ∧ w.b15.TiedGB N xN epsStr cotN ε ib15 dyO15
  ∧ w.d2.TiedGB N xN epsStr cotN ε ibD2 dyD2
  ∧ w.b16.TiedGB N xN epsStr cotN ε ib16 dyO16
  ∧ w.b17.TiedGB N xN epsStr cotN ε ib17 dyO17
  ∧ w.b18.TiedGB N xN epsStr cotN ε ib18 dyO18
  ∧ cnxHeadChTiedGBAt N xN epsStr cotN dN ε w.hG w.hT w.Wfc w.bfc xhead g := by
  intro ib1 ib2 ib3 ibD0 ib4 ib5 ib6 ibD1 ib7 ib8 ib9 ib10 ib11 ib12 ib13 ib14 ib15 ibD2 ib16 ib17 ib18 xhead gapB hnB logitsB g dyO18 dyO17 dyO16 dyD2 dyO15 dyO14 dyO13 dyO12 dyO11 dyO10 dyO9 dyO8 dyO7 dyD1 dyO6 dyO5 dyO4 dyD0 dyO3 dyO2 dyO1 dyStem
  refine ⟨cnx_stem_ch_tiedGBAt N xN epsStr cotN ε w.sW w.sb w.sγ w.sβ x xstem dyStem,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact w.b1.tiedGB N xN epsStr cotN ε ib1 dyO1
  · exact w.b2.tiedGB N xN epsStr cotN ε ib2 dyO2
  · exact w.b3.tiedGB N xN epsStr cotN ε ib3 dyO3
  · exact w.d0.tiedGB N xN epsStr cotN ε ibD0 dyD0
  · exact w.b4.tiedGB N xN epsStr cotN ε ib4 dyO4
  · exact w.b5.tiedGB N xN epsStr cotN ε ib5 dyO5
  · exact w.b6.tiedGB N xN epsStr cotN ε ib6 dyO6
  · exact w.d1.tiedGB N xN epsStr cotN ε ibD1 dyD1
  · exact w.b7.tiedGB N xN epsStr cotN ε ib7 dyO7
  · exact w.b8.tiedGB N xN epsStr cotN ε ib8 dyO8
  · exact w.b9.tiedGB N xN epsStr cotN ε ib9 dyO9
  · exact w.b10.tiedGB N xN epsStr cotN ε ib10 dyO10
  · exact w.b11.tiedGB N xN epsStr cotN ε ib11 dyO11
  · exact w.b12.tiedGB N xN epsStr cotN ε ib12 dyO12
  · exact w.b13.tiedGB N xN epsStr cotN ε ib13 dyO13
  · exact w.b14.tiedGB N xN epsStr cotN ε ib14 dyO14
  · exact w.b15.tiedGB N xN epsStr cotN ε ib15 dyO15
  · exact w.d2.tiedGB N xN epsStr cotN ε ibD2 dyD2
  · exact w.b16.tiedGB N xN epsStr cotN ε ib16 dyO16
  · exact w.b17.tiedGB N xN epsStr cotN ε ib17 dyO17
  · exact w.b18.tiedGB N xN epsStr cotN ε ib18 dyO18
  · exact cnx_head_ch_tiedGBAt N xN epsStr cotN dN ε w.hG w.hT w.Wfc w.bfc xhead g

end Proofs.CnxTiePoCGB
