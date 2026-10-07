import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFoldGB
import LeanMlir.Proofs.Foundation.GradNodesBAt
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtDropBlock

/-! # ConvNeXt-T's step tie at the batched index, the un-fused gradient and the smoothed loss

`ConvNeXtStepTie.lean` ties all 182 parameters of the SGD-inline `convnext_train_step.mlir`: each
fused `θ − lr·g` op `den`s to the certified step at the cotangent the emitted backward chain
delivers, per example, at a hard label. This file is that statement re-pointed along three axes;
unlike EfficientNet-B0's (`EfficientNetStepTieG.lean`, two axes), ConvNeXt's per-example capstone
was at a single image with the batch outside the AST, so the index is the third.

**Axis 1 — the gradient node.** Every conjunct is at the raw gradient node (`*GradB`), which is
what `convnext_adam_train_step.mlir` and every f32 `convnextin_*` artifact emit; the fused op
appears only in the SGD-inline file. The optimizer update that consumes the node (AdamW, the
`wx`/`clip` variants, EMA) is outside this statement. `ConvNeXtFoldGB.lean` is the fold each
conjunct delegates to.

**Precision is a flag on the statement.** The bf16 artifacts (`convnextin_adamwxclipdropbf16` and
its EMA, data-parallel, S and B twins) emit the bf16 conv, depthwise, strided-conv and patchify
weight-gradient kinds (`Foundation.Bf16GradNodes`). Every such node below is stated on the
renderers' switch — `ConvWTiedBAt bf16` / `DepthwiseWTiedBAt bf16` / `ConvStridedWTiedBAt bf16`
(`Foundation.GradNodesBAt`), the stem's `convStride4WeightGradBAt bf16 id …` spelled inline —
`bf16 := false` the f32 node, `bf16 := true` the bf16 one, and the capstone `cnx_net_tiedGB` takes
`bf16`, so it reaches the bf16 artifacts' gradient nodes read over ℝ exactly as it reads the f32
ones (`Bf16Erasure`: at the identity rounding the bf16 kind denotes what its f32 peer does). The
bias nodes, LayerNorm, layer scale and the classifier carry no flag because no render switches
them. The forward graph is a different matter: ConvNeXt's typed forward graph
(`convNextFwdGraphTCh`) is the per-example one, tied to `ConvNeXtRender`, which has no bf16; the
batched chain the bf16 artifacts render from has no typed graph at either precision.

**Axis 2 — the loss.** The capstone's top-of-chain cotangent is `smoothedLossCotGraphDiv`'s, at
a GENERAL target: the six-op chain `expe → softmaxDiv → subB → scaleB → addVB → shiftB →
divConstB` this render emits, with the target arriving as the graph input `%onehot` — a soft
vector under mixup or cutmix. The fused file pins it to `softmax − oneHot`, the gradient of plain
cross-entropy at a hard label, which no ImageNet artifact computes. ConvNeXt's chain runs at the
plain width `N·K` (no `1·` row index), so there is no `rowB`/`unrowB` cast anywhere here.

**Axis 3 — the index.** `N` is a binder. Every forward activation is a per-example lift of the
prefix the fused file threads (`cnxStemFwdO`, the block forward, `cnxDownFwdChO`), and every
cotangent a per-example lift of the chain (the block's input cotangent, `cnxDownCotInChAt`,
`ConvNeXtChainClose`'s `cnxCotP/E/N`, `chanLNTensor3Back`); a block's lift is the INDEXED one
(`batchMapIdx` / `batchMapAuxIdx`, `Foundation.Batched.Indexed`), since with stochastic depth
example `n` runs the block at its own mask entry. That lift is
honest for this net and for no BatchNorm net: LayerNorm, GELU, the convolutions, layer scale and
the residual add are all batch-separable, so the batched op IS the per-example op under
`batchMap` — which is what the `*B` constructors' `den` arms say. `nC` is a binder too: the
Imagenette artifact is `nC = 10`, the ImageNet ones `nC = 1000`.

## Relation to the fused file

Every activation, every Jacobian witness and every chain cotangent is `ConvNeXtStepTie.lean`'s,
lifted; every conjunct's proof is one `CnxFoldGB.*_den` lemma. The `wN`, `bN`, `gN`, `lrStr` and
`lr` binders disappear with the fused wrapper, as they did for B0. The head takes `g` as a
parameter where the fused `cnxHeadChTied` computed it from a label; the per-block ties were
already loss-agnostic.

**Conventions carried unchanged from the fused file:** channel LayerNorm (`chanLNTensor3`, a
`Vec c` affine, `h·w` statistics per example) at all 22 spatial sites, the head at ViT's vector LN
on one row, per-channel layer scale, GELU (no kink — no smoothness hypothesis anywhere),
SYMMETRIC padding at the 4×4/s4 stem and the three 2×2/s2 downsamples. Stated at the literal
widths of ConvNeXt-T; S and B are other nets.

**Scope.** One replica: in `convnextin_adamdp*` every gradient node feeds `allReduceMeanF`;
`DataParallel.Node` composes the per-replica statement with the replica mean.

**Stochastic depth is a binder.** `sd : Option (Fin 18 → Vec N)` is the render's `drop` flag:
`none` the drop-free artifacts' chain (every family constant, the statement the drop-free one by
`rfl`), `some` the `*drop*` artifacts' — every shipped `convnextin_*bf16` artifact, the book's
`convnextin_adamdpwxclipdroperfbf16` among them. Block `k`'s site (`cnxSd`) sits between
LayerScale and the skip add (`CnxTieBlk.fwdOD`), so every one of the block's nine nodes, LayerScale
γ's included, reads the dropped cotangent (`dropPathOpt`, where `ConvNeXtRenderB.bwdBlockB` emits
`dropPathB` on it) and only the skip fan-in the raw one (`CnxTieBlk.cotInD`, `ConvNeXtDropBlock`).
The forward saves the nodes read are the drop-free ones: the site is after them. Note:
the `%dgi…%dgapf` GAP backward is hand-written text on both chains; its value here is
`globalAvgPoolFlatHasVJP.backward`, as in the fused file.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CnxTieGB

open scoped BigOperators
open Proofs.CnxTie (cnxStemFwdO cnxBlockFwdChO cnxDownFwdChO cnxBlockCotInChAt cnxDownCotInChAt
  CnxTieWeights)
open Proofs.GradNodeB (convBTiedB_holds convStridedBTiedB_holds convStridedWTiedBAt_holds
  convWTiedBAt_holds depthwiseBTiedB_holds depthwiseWTiedBAt_holds)
open Proofs.GradNodeB (vecLNGammaTiedB_holds)
open Proofs.CnxFoldGB (chanLNBetaTiedB_holds chanLNGammaTiedB_holds)

/-! ## Per-example internal cotangents as functions of a block's INPUT

`ConvNeXtChainClose`'s `cnxCotE/N` and `chanLNTensor3Back` take the saved activations as
arguments; `batchMapAux` lifts a function of (one saved value, one input), so these recompute the
activations from the block input, exactly as `cnxBlockCotInChAt` does. -/

/-- Cotangent at the expand output (pre-GELU), from the block input and output cotangent. -/
noncomputable def blkCotE (gf : GeluForm) {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin dyOut : Vec (c*h*w)) : Vec (cExp*h*w) :=
  let γlsB : Vec (c*h*w) := fun k => lg (chanIdx c h w k)
  let nl := chanLNTensor3 c h w ε ng nbt (depthwiseFlat (h := h) (w := w) Wdw bdw xin)
  let e := flatConv (h := h) (w := w) Wex bex nl
  cnxCotE gf γlsB Wpr bpr (gf.map (cExp*h*w) e) e dyOut

/-- Cotangent at the channel-LN output, from the block input and output cotangent. -/
noncomputable def blkCotN (gf : GeluForm) {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin dyOut : Vec (c*h*w)) : Vec (c*h*w) :=
  let γlsB : Vec (c*h*w) := fun k => lg (chanIdx c h w k)
  let nl := chanLNTensor3 c h w ε ng nbt (depthwiseFlat (h := h) (w := w) Wdw bdw xin)
  let e := flatConv (h := h) (w := w) Wex bex nl
  cnxCotN gf γlsB Wex bex Wpr bpr nl (gf.map (cExp*h*w) e) e dyOut

/-- Cotangent at the depthwise output (the channel-LN input-VJP of `blkCotN`). -/
noncomputable def blkCotD (gf : GeluForm) {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin dyOut : Vec (c*h*w)) : Vec (c*h*w) :=
  chanLNTensor3Back c h w ε ng (depthwiseFlat (h := h) (w := w) Wdw bdw xin)
    (blkCotN gf ε Wdw bdw ng nbt Wex bex Wpr bpr lg xin dyOut)

/-- Downsample: cotangent at the LN output, i.e. the strided conv's input-VJP. -/
noncomputable def dnCotN {ci co h w : Nat} (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (ci*(2*h)*(2*w))) (dyOut : Vec (co*h*w)) : Vec (ci*(2*h)*(2*w)) :=
  (flatConvStride2HasVJP Wd bd).backward (chanLNTensor3 ci (2*h) (2*w) ε dng dnbt xin) dyOut

/-- Stem: cotangent at the patchify output, the stem LN's input-VJP of `dyStem`. -/
noncomputable def stemCotPatch {c h w : Nat} (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng : Vec c)
    (x : Vec (3*(2*(2*h))*(2*(2*w)))) (dyStem : Vec (c*h*w)) : Vec (c*h*w) :=
  chanLNTensor3Back c h w ε psng (flatConvStride4 Wst psb x) dyStem

/-- Head: the dense backward at one example's LN output and loss cotangent. -/
noncomputable def headCotHn {nC : Nat} (Wfc : Mat 768 nC) (bfc : Vec nC)
    (hn : Vec 768) (g : Vec nC) : Vec (1*768) :=
  (denseHasVJP Wfc bfc).backward hn g

/-- The cotangent at the last block output, per example, at a GENERAL class count —
    `CnxTie.cnxHeadDyXheadCh` with `nC` a binder (that one is at the literal 10). -/
noncomputable def cnxHeadDyXheadChN {h w nC : Nat} (ε : ℝ)
    (hng hnbt : Vec 768) (Wfc : Mat 768 nC) (bfc : Vec nC)
    (xhead : Vec (768*h*w)) (g : Vec nC) : Vec (768*h*w) :=
  let gap : Vec (1*768) := globalAvgPoolFlat 768 h w xhead
  let hn : Vec 768 := rowLNVecFlat 1 768 ε hng hnbt gap
  let cotHn : Vec (1*768) := (denseHasVJP Wfc bfc).backward hn g
  let cotGap : Vec 768 := rowLNVecFlatBack 1 768 ε hng gap cotHn
  (globalAvgPoolFlatHasVJP 768 h w).backward xhead cotGap

/-! ## ConvNeXt block — all 9 gradient nodes, batched -/

/-- **ConvNeXt block, tied at the batched gradient nodes.** All 9 params (depthwise 7×7 `W`+`b`,
    channel-LN γ/β at `Vec c`, expand/project 1×1 `W`+`b`, per-channel layer-scale γ) denote the
    certified `Σ_n` gradient at the real batched block forward and the chain cotangents driven by
    `dyOut`. -/
def cnxBlockChTiedGB (gf : GeluForm) (N : Nat) {c cExp h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (bf16 : Bool) (sd : Option (Vec N)) (xin dyOut : Vec (N * (c*h*w))) : Prop :=
  let γlsB : Vec (c*h*w) := fun k => lg (chanIdx c h w k)
  -- the drop site, between LayerScale and the skip add: the whole branch reads `s ⊙ dyOut`
  let dyOutD : Vec (N * (c*h*w)) := dropPathOpt N (c*h*w) sd dyOut
  -- forward activations — each `batchMap` of the per-example op the emitted node lifts
  let dB  : Vec (N * (c*h*w))    := batchMap N (depthwiseFlat (h := h) (w := w) Wdw bdw) xin
  let nlB : Vec (N * (c*h*w))    := batchMap N (chanLNTensor3 c h w ε ng nbt) dB
  let gB  : Vec (N * (cExp*h*w)) :=
    batchMap N (fun nl => gf.map (cExp*h*w) (flatConv (h := h) (w := w) Wex bex nl)) nlB
  let pB  : Vec (N * (c*h*w))    := batchMap N (flatConv (h := h) (w := w) Wpr bpr) gB
  -- backward chain cotangents — `batchMapAux` of the per-example chain
  let cotPB : Vec (N * (c*h*w))    := batchMap N (cnxCotP γlsB) dyOutD
  let cotEB : Vec (N * (cExp*h*w)) :=
    batchMapAux N (blkCotE gf ε Wdw bdw ng nbt Wex bex Wpr bpr lg) xin dyOutD
  let cotNB : Vec (N * (c*h*w))    :=
    batchMapAux N (blkCotN gf ε Wdw bdw ng nbt Wex bex Wpr bpr lg) xin dyOutD
  let cotDB : Vec (N * (c*h*w))    :=
    batchMapAux N (blkCotD gf ε Wdw bdw ng nbt Wex bex Wpr bpr lg) xin dyOutD
  -- depthwise 7×7 W/b  (cot = cotDB, input = xin)
  GradNodeB.DepthwiseWTiedBAt bf16 N h w xN cotN bdw xin Wdw cotDB
  ∧ GradNodeB.DepthwiseBTiedB N h w cotN Wdw xin bdw cotDB
  -- channel-LN γ/β  (cot = cotNB, LN input = dB; the ops see both as their batched [h·w, c] views)
  ∧ CnxFoldGB.ChanLNGammaTiedB N h w xN epsStr cotN ε nbt dB ng cotNB
  ∧ CnxFoldGB.ChanLNBetaTiedB N h w cotN ε ng dB nbt cotNB
  -- expand 1×1 conv (c → cExp) W/b  (cot = cotEB, conv input = nlB)
  ∧ GradNodeB.ConvWTiedBAt bf16 N h w xN cotN bex nlB Wex cotEB
  ∧ GradNodeB.ConvBTiedB N h w cotN Wex nlB bex cotEB
  -- project 1×1 conv (cExp → c) W/b  (cot = cotPB, conv input = gB)
  ∧ GradNodeB.ConvWTiedBAt bf16 N h w xN cotN bpr gB Wpr cotPB
  ∧ GradNodeB.ConvBTiedB N h w cotN Wpr gB bpr cotPB
  -- per-channel layer-scale γ  (cot = the dropped block-output cotangent, layer input = pB)
  ∧ (∀ cc : Fin c,
      den (SHlo.layerScaleChGammaGradB (N := N) (c := c) (h := h) (w := w) xN pB
            (.operand cotN dyOutD)) cc
        = ∑ n : Fin N, ∑ j : Fin (c*h*w),
            pdiv (fun γ' : Vec c =>
                    layerScale (fun k => γ' (chanIdx c h w k)) (batchSlice N (c*h*w) pB n))
                 lg cc j * batchSlice N (c*h*w) dyOutD n j)

theorem cnx_block_ch_tiedGB {gf : GeluForm} (N : Nat) {c cExp h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (bf16 : Bool) (sd : Option (Vec N)) (xin dyOut : Vec (N * (c*h*w))) :
    cnxBlockChTiedGB gf N xN epsStr cotN ε Wdw bdw ng nbt Wex bex Wpr bpr lg bf16 sd xin dyOut := by
  unfold cnxBlockChTiedGB
  intro γlsB dyOutD dB nlB gB pB cotPB cotEB cotNB cotDB
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact depthwiseWTiedBAt_holds bf16
  · exact depthwiseBTiedB_holds
  · exact chanLNGammaTiedB_holds
  · exact chanLNBetaTiedB_holds
  · exact convWTiedBAt_holds bf16
  · exact convBTiedB_holds
  · exact convWTiedBAt_holds bf16
  · exact convBTiedB_holds
  · intro cc;  exact CnxFoldGB.layerScaleChGammaGradB_den xN cotN pB lg dyOutD cc

/-! ## Downsample — channel-LN → 2×2/s2 conv, all 4 gradient nodes, batched -/

/-- **Downsample, tied at the batched gradient nodes.** Channel-LN γ/β at the `ci·(2h)·(2w)` input
    grid, plus the strided conv's weight and bias. -/
def cnxDownChTiedGB (N : Nat) {ci co h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co) (bf16 : Bool)
    (xin : Vec (N * (ci*(2*h)*(2*w)))) (dyOut : Vec (N * (co*h*w))) : Prop :=
  let nB : Vec (N * (ci*(2*h)*(2*w))) := batchMap N (chanLNTensor3 ci (2*h) (2*w) ε dng dnbt) xin
  let cotNB : Vec (N * (ci*(2*h)*(2*w))) := batchMapAux N (dnCotN ε dng dnbt Wd bd) xin dyOut
  CnxFoldGB.ChanLNGammaTiedB N (2 * h) (2 * w) xN epsStr cotN ε dnbt xin dng cotNB
  ∧ CnxFoldGB.ChanLNBetaTiedB N (2 * h) (2 * w) cotN ε dng xin dnbt cotNB
  ∧ GradNodeB.ConvStridedWTiedBAt bf16 N h w xN cotN bd nB Wd dyOut
  ∧ GradNodeB.ConvStridedBTiedB N h w cotN Wd nB bd dyOut

theorem cnx_down_ch_tiedGB (N : Nat) {ci co h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co) (bf16 : Bool)
    (xin : Vec (N * (ci*(2*h)*(2*w)))) (dyOut : Vec (N * (co*h*w))) :
    cnxDownChTiedGB N xN epsStr cotN ε dng dnbt Wd bd bf16 xin dyOut := by
  unfold cnxDownChTiedGB
  exact ⟨chanLNGammaTiedB_holds, chanLNBetaTiedB_holds, convStridedWTiedBAt_holds bf16,
    convStridedBTiedB_holds⟩

/-! ## Stem — 4×4/s4 patchify conv → channel-LN, all 4 gradient nodes, batched

The bias grad is a pure cotangent reduce, so the render emits it as a stride-1 `convBiasGradB` at
the OUTPUT resolution; the node carries an input it never reads (stated at zero), and the clause
is at the real stride-4 stem on each example's image through
`GradNodeB.pdiv_flatConvStride4_bias_eq_conv2d`. The weight is `convStride4WeightGradB`, at its gradient
on both chains. -/

/-- **Stem, tied at the batched gradient nodes.** Channel-LN γ/β, the conv bias, the conv weight. -/
def cnxStemChTiedGB (N : Nat) {c h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c) (bf16 : Bool)
    (x : Vec (N * (3*(2*(2*h))*(2*(2*w)))))
    (dyStem : Vec (N * (c*h*w))) : Prop :=
  let patchB : Vec (N * (c*h*w)) := batchMap N (flatConvStride4 Wst psb) x
  let cotPatchB : Vec (N * (c*h*w)) := batchMapAux N (stemCotPatch ε Wst psb psng) x dyStem
  CnxFoldGB.ChanLNGammaTiedB N h w xN epsStr cotN ε psnbt patchB psng dyStem
  ∧ CnxFoldGB.ChanLNBetaTiedB N h w cotN ε psng patchB psnbt dyStem
  ∧ (∀ o : Fin c,
      den (SHlo.convBiasGradB (N := N) (ic := 3) (oc := c) (h := h) (w := w) (kH := 4) (kW := 4)
            Wst 0 psb (.operand cotN cotPatchB)) o
        = ∑ n : Fin N, ∑ j : Fin (c*h*w),
            pdiv (fun b' : Vec c =>
                    (flatConvStride4 Wst b' (batchSlice N (3*(2*(2*h))*(2*(2*w))) x n) : Vec (c*h*w)))
                 psb o j * batchSlice N (c*h*w) cotPatchB n j)
  ∧ (∀ idx : Fin (c*3*4*4),
      den (SHlo.convStride4WeightGradBAt bf16 id xN psb x Wst (.operand cotN cotPatchB)) idx
        = ∑ n : Fin N, ∑ j : Fin (c*h*w),
            pdiv (fun v' : Vec (c*3*4*4) =>
                    flatConvStride4 (Kernel4.unflatten v') psb
                      (batchSlice N (3*(2*(2*h))*(2*(2*w))) x n))
                 (Kernel4.flatten Wst) idx j * batchSlice N (c*h*w) cotPatchB n j)

theorem cnx_stem_ch_tiedGB (N : Nat) {c h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c) (bf16 : Bool)
    (x : Vec (N * (3*(2*(2*h))*(2*(2*w)))))
    (dyStem : Vec (N * (c*h*w))) :
    cnxStemChTiedGB N xN epsStr cotN ε Wst psb psng psnbt bf16 x dyStem := by
  unfold cnxStemChTiedGB
  intro patchB cotPatchB
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact chanLNGammaTiedB_holds
  · exact chanLNBetaTiedB_holds
  · intro o
    refine (convBTiedB_holds o).trans (Finset.sum_congr rfl fun n _ => Finset.sum_congr rfl
      fun j _ => ?_)
    rw [GradNodeB.pdiv_flatConvStride4_bias_eq_conv2d Wst _
      (Tensor3.unflatten (batchSlice N (3*h*w) (0 : Vec (N * (3*h*w))) n))]
  · intro idx
    rw [Bf16Fold.den_convStride4WeightGradBAt_id]
    exact GradNodeB.psWGradB_den xN cotN psb x Wst cotPatchB idx

/-! ## Head — GAP → vector-LN at one row → dense, all 4 gradient nodes, batched

Stated at the LITERAL 768 for the fused file's reason: `1 * m` does not reduce at a variable `m`.
`g` is a PARAMETER — the loss cotangent arrives from `smoothedLossCotGraphDiv` in the capstone. -/

/-- **Head, tied at the batched gradient nodes.** The head-LN γ/β at the pooled row, the
    classifier weight at the LN output, and the classifier bias summed over the batch
    (`GradNodeB.headBGradB_den`). -/
def cnxHeadChTiedGB (N : Nat) {h w nC : Nat} (xN epsStr cotN dN : String) (ε : ℝ)
    (hng hnbt : Vec 768) (Wfc : Mat 768 nC) (bfc : Vec nC)
    (xhead : Vec (N * (768*h*w))) (g : Vec (N * nC)) : Prop :=
  let gapB   : Vec (N * (1*768)) := batchMap N (globalAvgPoolFlat 768 h w) xhead
  let hnB    : Vec (N * 768)     := batchMap N (rowLNVecFlat 1 768 ε hng hnbt) gapB
  let cotHnB : Vec (N * (1*768)) := batchMapAux N (headCotHn Wfc bfc) hnB g
  GradNodeB.VecLNGammaTiedB N 1 xN epsStr cotN ε hnbt gapB hng cotHnB
  ∧ GradNodeB.VecLNBetaTiedB N 1 cotN ε hng gapB hnbt cotHnB
  ∧ (∀ (i : Fin 768) (j : Fin nC),
      den (SHlo.weightGradB (N := N) (m := 768) (n := nC) dN hnB (.operand cotN g))
          (finProdFinEquiv (i, j))
        = ∑ n : Fin N, ∑ k : Fin nC,
            pdiv (fun v : Vec (768 * nC) => dense (Mat.unflatten v) bfc (batchSlice N 768 hnB n))
                 (Mat.flatten Wfc) (finProdFinEquiv (i, j)) k * batchSlice N nC g n k)
  ∧ (∀ i : Fin nC,
      den (SHlo.biasGradB (N := N) (n := nC) (.operand cotN g)) i
        = ∑ n : Fin N, ∑ j : Fin nC,
            pdiv (fun b' : Vec nC => dense Wfc b' (batchSlice N 768 hnB n)) bfc i j
              * batchSlice N nC g n j)

theorem cnx_head_ch_tiedGB (N : Nat) {h w nC : Nat} (xN epsStr cotN dN : String) (ε : ℝ)
    (hng hnbt : Vec 768) (Wfc : Mat 768 nC) (bfc : Vec nC)
    (xhead : Vec (N * (768*h*w))) (g : Vec (N * nC)) :
    cnxHeadChTiedGB N xN epsStr cotN dN ε hng hnbt Wfc bfc xhead g := by
  unfold cnxHeadChTiedGB
  intro gapB hnB cotHnB
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact vecLNGammaTiedB_holds
  · intro k;
    exact GradNodeB.rowDenseBiasGradB_den_lnbeta cotN ε hng
      (fun n => Mat.unflatten (batchSlice N (1*768) gapB n)) hnbt cotHnB k
  · intro i j; exact GradNodeB.headWGradB_den dN cotN hnB Wfc bfc g i j
  · intro i; exact GradNodeB.headBGradB_den cotN Wfc (batchSlice N 768 hnB) bfc g i

/-! ## The stem wrapper — `@[irreducible]`

Without it the capstone's `refine ⟨cnx_stem_ch_tiedGBAt …, ?_, …⟩` times out in `whnf` at the
default budget. The block, downsample and head ties need no wrapper: the capstone names
`cnxBlockChTiedGB` / `cnxDownChTiedGB` / `cnxHeadChTiedGB` directly. -/

@[irreducible] def cnxStemChTiedGBAt (N : Nat) {c h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c) (bf16 : Bool)
    (x : Vec (N * (3*(2*(2*h))*(2*(2*w)))))
    (dyStem : Vec (N * (c*h*w))) : Prop :=
  cnxStemChTiedGB N xN epsStr cotN ε Wst psb psng psnbt bf16 x dyStem

private theorem cnx_stem_ch_tiedGBAt (N : Nat) {c h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c) (bf16 : Bool)
    (x : Vec (N * (3*(2*(2*h))*(2*(2*w)))))
    (dyStem : Vec (N * (c*h*w))) :
    cnxStemChTiedGBAt N xN epsStr cotN ε Wst psb psng psnbt bf16 x dyStem := by
  unfold cnxStemChTiedGBAt
  exact cnx_stem_ch_tiedGB N xN epsStr cotN ε Wst psb psng psnbt bf16 x dyStem

/-! ## Every cotangent the capstone threads is a certified VJP backward

The capstone below threads three per-example constructors: the head's `cnxHeadDyXheadChN`, each
block's `cotInD` at its site and each downsample's `cnxDownCotInChAt`. Per example, each is the
backward of its stage's certified VJP (`CnxTieBlk.fwdODHasVJP`, `cnxDownChWHasVJP`, and
`cnxHeadHasVJP` below); lifted by `batchMapAux_eq_batchMapHasVJPAt` (the block's by
`batchMapAuxIdx_eq_batchMapIdxHasVJPAt`), the batched cotangent is the backward of the lifted
stage. ConvNeXt has no kink, so the only
hypothesis is the LayerNorm's `0 < ε`. -/

/-- **A downsample's input cotangent is its certified VJP's backward.** One rewrite:
    `chanLNTensor3Back` is the channel-LN VJP's backward. -/
theorem cnxDownCotInChAt_eq_vjp {ci co h w : Nat} (ε : ℝ) (hε : 0 < ε)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (ci*(2*h)*(2*w))) (dyOut : Vec (co*h*w)) :
    cnxDownCotInChAt ε dng dnbt Wd bd xin dyOut
      = (cnxDownChWHasVJP h w ⟨ε, dng, dnbt, Wd, bd⟩ hε).backward xin dyOut := by
  unfold cnxDownCotInChAt
  rw [chanLNTensor3Back_eq_chanLN_vjp (β := dnbt) ε hε dng xin]
  rfl

/-- The head `dense ∘ head LN ∘ GAP` as one certified VJP, at any class count. -/
noncomputable def cnxHeadHasVJP (h w : Nat) {nC : Nat} (ε : ℝ) (hε : 0 < ε) (hng hnbt : Vec 768)
    (Wfc : Mat 768 nC) (bfc : Vec nC) :
    HasVJP (dense Wfc bfc ∘ rowLNVecFlat 1 768 ε hng hnbt ∘ globalAvgPoolFlat 768 h w) :=
  vjpComp (globalAvgPoolFlat 768 h w) (dense Wfc bfc ∘ rowLNVecFlat 1 768 ε hng hnbt)
    (globalAvgPoolFlat_differentiable 768 h w)
    ((dense_differentiable Wfc bfc).comp (rowLNVecFlat_differentiable 1 768 ε hng hnbt hε))
    (globalAvgPoolFlatHasVJP 768 h w)
    (vjpComp (rowLNVecFlat 1 768 ε hng hnbt) (dense Wfc bfc)
      (rowLNVecFlat_differentiable 1 768 ε hng hnbt hε) (dense_differentiable Wfc bfc)
      (rowLNVecFlatHasVJP 1 768 ε hng hnbt hε) (denseHasVJP Wfc bfc))

/-- **The head's input cotangent — the last block's `dyOut` — is its certified VJP's backward**, at
    any loss cotangent `g`. -/
theorem cnxHeadDyXheadChN_eq_vjp {h w nC : Nat} (ε : ℝ) (hε : 0 < ε) (hng hnbt : Vec 768)
    (Wfc : Mat 768 nC) (bfc : Vec nC) (xhead : Vec (768*h*w)) (g : Vec nC) :
    cnxHeadDyXheadChN ε hng hnbt Wfc bfc xhead g
      = (cnxHeadHasVJP h w ε hε hng hnbt Wfc bfc).backward xhead g := by
  simp only [cnxHeadDyXheadChN, cnxHeadHasVJP, vjpComp_backward]
  rw [rowLNVecFlatHasVJP_backward_eq_fun (β := hnbt) ε hε hng]

/-- **Batched: a block's cotangent at its drop site is the lifted block VJP's backward.** Example
    `n` runs the block at its own mask entry, so the lift is the indexed one; `cotInD_eq_vjp` per
    example. At `none` it is the drop-free block's (`cotInD_none`, `fwdOD_none`). -/
theorem cnxBlockCotInB_eq_vjp {gf : GeluForm} (N : Nat) {c cExp h w : Nat} (ε : ℝ) (hε : 0 < ε)
    (p : CnxTie.CnxTieBlk c cExp) (sd : Option (Vec N)) (xin : Vec (N * (c*h*w))) :
    batchMapAuxIdx N (fun n => p.cotInD gf (h := h) (w := w) ε (exampleSite sd n)) xin
      = (batchMapIdxHasVJPAt (fun n => p.fwdOD gf (h := h) (w := w) ε (exampleSite sd n)) xin
          (fun n => (p.fwdODHasVJP gf ε hε (exampleSite sd n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε p _ _)).backward :=
  batchMapAuxIdx_eq_batchMapIdxHasVJPAt _ _ xin _ _ fun _ => by
    funext dy; exact cotInD_eq_vjp ε hε p _ _ dy

/-- **Batched: a downsample's `batchMapAux` cotangent is the lifted downsample VJP's backward.** -/
theorem cnxDownCotInB_eq_vjp (N : Nat) {ci co h w : Nat} (ε : ℝ) (hε : 0 < ε)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (N * (ci*(2*h)*(2*w)))) :
    batchMapAux N (cnxDownCotInChAt (h := h) (w := w) ε dng dnbt Wd bd) xin
      = (batchMapHasVJPAt (cnxDownChW h w ⟨ε, dng, dnbt, Wd, bd⟩) xin
          (fun _ => (cnxDownChWHasVJP h w ⟨ε, dng, dnbt, Wd, bd⟩ hε).toHasVJPAt _)
          (fun _ => (cnxDownChW_differentiable h w ⟨ε, dng, dnbt, Wd, bd⟩ hε) _)).backward :=
  batchMapAux_eq_batchMapHasVJPAt _ _ xin _ _ fun _ => by
    funext dy; exact cnxDownCotInChAt_eq_vjp ε hε dng dnbt Wd bd _ dy

/-- **Batched: the head's `batchMapAux` cotangent is the lifted head VJP's backward.** -/
theorem cnxHeadDyB_eq_vjp (N : Nat) {h w nC : Nat} (ε : ℝ) (hε : 0 < ε) (hng hnbt : Vec 768)
    (Wfc : Mat 768 nC) (bfc : Vec nC) (xhead : Vec (N * (768*h*w))) :
    batchMapAux N (cnxHeadDyXheadChN (h := h) (w := w) ε hng hnbt Wfc bfc) xhead
      = (batchMapHasVJPAt (dense Wfc bfc ∘ rowLNVecFlat 1 768 ε hng hnbt ∘
            globalAvgPoolFlat 768 h w) xhead
          (fun _ => (cnxHeadHasVJP h w ε hε hng hnbt Wfc bfc).toHasVJPAt _)
          (fun _ => ((dense_differentiable Wfc bfc).comp
            ((rowLNVecFlat_differentiable 1 768 ε hng hnbt hε).comp
              (globalAvgPoolFlat_differentiable 768 h w))) _)).backward :=
  batchMapAux_eq_batchMapHasVJPAt _ _ xhead _ _ fun _ => by
    funext g; exact cnxHeadDyXheadChN_eq_vjp ε hε hng hnbt Wfc bfc _ g

/-! ## The whole-net capstone — all 182 params through the REAL batched forward + composed cotangent

The fused file's thread, lifted: block inputs are per-example lifts of the forward prefixes (a
block at its site, `batchMapIdx`), and the backward cotangents lifts of the per-example chain,
composed from the smoothed loss
`g` down through the head, every block's backward with the residual fan-in `+ dyOut` at each of
the eighteen identity-skip merges, the channel-LN-back at each of the three downsamples, and the
stem LN's own back before the patchify conv's gradients. -/

/-- Block `k`'s stochastic-depth masks (of the eighteen, in render order; the downsamples carry
    none), when the render carries stochastic depth. -/
def cnxSd {N : Nat} (sd : Option (Fin 18 → Vec N)) (k : Fin 18) : Option (Vec N) :=
  sd.map fun f => f k

@[simp] theorem cnxSd_none {N : Nat} (k : Fin 18) : cnxSd (none : Option (Fin 18 → Vec N)) k = none :=
  rfl

/-- The block's batched tie (`cnxBlockChTiedGB`), over its `CnxTieBlk` record. -/
abbrev _root_.Proofs.CnxTie.CnxTieBlk.TiedGB (gf : GeluForm) {c cExp h w : Nat} (p : CnxTie.CnxTieBlk c cExp)
    (N : Nat) (xN epsStr cotN : String) (ε : ℝ) (bf16 : Bool) (sd : Option (Vec N))
    (xin dyOut : Vec (N * (c*h*w))) : Prop :=
  cnxBlockChTiedGB gf N xN epsStr cotN ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL bf16 sd xin dyOut

theorem _root_.Proofs.CnxTie.CnxTieBlk.tied_gb {gf : GeluForm} {c cExp h w : Nat} (p : CnxTie.CnxTieBlk c cExp)
    (N : Nat) (xN epsStr cotN : String) (ε : ℝ) (bf16 : Bool) (sd : Option (Vec N))
    (xin dyOut : Vec (N * (c*h*w))) : p.TiedGB gf N xN epsStr cotN ε bf16 sd xin dyOut :=
  cnx_block_ch_tiedGB N xN epsStr cotN ε _ _ _ _ _ _ _ _ _ bf16 sd xin dyOut

/-- The downsample's batched tie (`cnxDownChTiedGB`), over its `CnxTieDown` record. -/
abbrev _root_.Proofs.CnxTie.CnxTieDown.TiedGB {ci co h w : Nat} (p : CnxTie.CnxTieDown ci co)
    (N : Nat) (xN epsStr cotN : String) (ε : ℝ) (bf16 : Bool) (xin : Vec (N * (ci*(2*h)*(2*w))))
    (dyOut : Vec (N * (co*h*w))) : Prop :=
  cnxDownChTiedGB N xN epsStr cotN ε p.G p.T p.W p.B bf16 xin dyOut

theorem _root_.Proofs.CnxTie.CnxTieDown.tied_gb {ci co h w : Nat} (p : CnxTie.CnxTieDown ci co)
    (N : Nat) (xN epsStr cotN : String) (ε : ℝ) (bf16 : Bool) (xin : Vec (N * (ci*(2*h)*(2*w))))
    (dyOut : Vec (N * (co*h*w))) : p.TiedGB N xN epsStr cotN ε bf16 xin dyOut :=
  cnx_down_ch_tiedGB N xN epsStr cotN ε _ _ _ _ bf16 xin dyOut

/-- **The whole [3,3,9,3] ConvNeXt-T train step, tied at the batched index, the gradient
    nodes and the smoothed loss.** Threading the real channel-LN / per-channel layer-scale forward
    as per-example lifts of the prefixes, and the label-smoothed loss cotangent
    (`smoothedLossCotGraphDiv`, at a general target `t`) down through the head and every block's
    cotangent chain as the lift of the per-example chain — each the backward of its stage's
    lifted certified VJP (`cnxHeadDyB_eq_vjp`, `cnxBlockCotInB_eq_vjp`, `cnxDownCotInB_eq_vjp`) —
    GELU masks, the residual
    fan-in at every identity skip, the channel-LN-back at every downsample and at the stem — the
    18 ConvNeXt blocks, the 3 downsamples, the 4×4/s4 stem with its LN and the GAP → LN → dense
    head all denote the certified batched `Σ_n` gradient. All 182 parameters, at the gradient
    nodes `convnext_adam_train_step.mlir` and every `convnextin_*` train step emit — `bf16` selects
    the conv, depthwise, strided-conv and patchify weight nodes' kind, `false` the f32 artifacts',
    `true` the `*bf16` ones', read over ℝ at the identity rounding (`Bf16Erasure`); the right-hand
    side is the same certified gradient at either value.

    `N` and `nC` are binders and there is no smoothness hypothesis: the folds are `∀ cot`
    statements instantiated at explicitly constructed cotangents, and ConvNeXt has no kink. The
    batch enters only through the per-example lifts, because no ConvNeXt op couples examples.
    The statement is at one replica: in `convnextin_adamdp*` every gradient node feeds
    `allReduceMeanF`, and `DataParallel.Node` composes the per-replica statement with the
    replica mean. `sd` is stochastic depth: `none` the drop-free artifacts' chain, `some` the
    `*drop*` artifacts' — `convnextin_adamdpwxclipdroperfbf16`'s, the book's run — each block's
    nodes at the dropped cotangent and its skip at the raw one, so the book's ConvNeXt job is
    reached at every gradient node. -/
theorem cnx_net_tiedGB {gf : GeluForm} (N : Nat) {nC : Nat}
    (xN epsStr cotN dN aStr negAK bStr logN ohN : String) (ε α B : ℝ)
    (w : CnxTieWeights nC) (bf16 : Bool) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3*224*224))) (t : Vec (N * nC)) :
    -- forward block inputs (the prefixes of the committed render's forward)
    let ib1 : Vec (N * (96*56*56)) := batchMap N (cnxStemFwdO (h := 56) (w := 56) ε w.sW w.sb w.sγ w.sβ) x
    let ib2 : Vec (N * (96*56*56)) := batchMapIdx N (fun n => w.b1.fwdOD gf ε (exampleSite (cnxSd sd 0) n)) ib1
    let ib3 : Vec (N * (96*56*56)) := batchMapIdx N (fun n => w.b2.fwdOD gf ε (exampleSite (cnxSd sd 1) n)) ib2
    let ibD0 : Vec (N * (96*56*56)) := batchMapIdx N (fun n => w.b3.fwdOD gf ε (exampleSite (cnxSd sd 2) n)) ib3
    let ib4 : Vec (N * (192*28*28)) := batchMap N (w.d0.fwdO (h := 28) (w := 28) ε) ibD0
    let ib5 : Vec (N * (192*28*28)) := batchMapIdx N (fun n => w.b4.fwdOD gf ε (exampleSite (cnxSd sd 3) n)) ib4
    let ib6 : Vec (N * (192*28*28)) := batchMapIdx N (fun n => w.b5.fwdOD gf ε (exampleSite (cnxSd sd 4) n)) ib5
    let ibD1 : Vec (N * (192*28*28)) := batchMapIdx N (fun n => w.b6.fwdOD gf ε (exampleSite (cnxSd sd 5) n)) ib6
    let ib7 : Vec (N * (384*14*14)) := batchMap N (w.d1.fwdO (h := 14) (w := 14) ε) ibD1
    let ib8 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b7.fwdOD gf ε (exampleSite (cnxSd sd 6) n)) ib7
    let ib9 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b8.fwdOD gf ε (exampleSite (cnxSd sd 7) n)) ib8
    let ib10 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b9.fwdOD gf ε (exampleSite (cnxSd sd 8) n)) ib9
    let ib11 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b10.fwdOD gf ε (exampleSite (cnxSd sd 9) n)) ib10
    let ib12 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b11.fwdOD gf ε (exampleSite (cnxSd sd 10) n)) ib11
    let ib13 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b12.fwdOD gf ε (exampleSite (cnxSd sd 11) n)) ib12
    let ib14 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b13.fwdOD gf ε (exampleSite (cnxSd sd 12) n)) ib13
    let ib15 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b14.fwdOD gf ε (exampleSite (cnxSd sd 13) n)) ib14
    let ibD2 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b15.fwdOD gf ε (exampleSite (cnxSd sd 14) n)) ib15
    let ib16 : Vec (N * (768*7*7)) := batchMap N (w.d2.fwdO (h := 7) (w := 7) ε) ibD2
    let ib17 : Vec (N * (768*7*7)) := batchMapIdx N (fun n => w.b16.fwdOD gf ε (exampleSite (cnxSd sd 15) n)) ib16
    let ib18 : Vec (N * (768*7*7)) := batchMapIdx N (fun n => w.b17.fwdOD gf ε (exampleSite (cnxSd sd 16) n)) ib17
    let xhead : Vec (N * (768*7*7)) := batchMapIdx N (fun n => w.b18.fwdOD gf ε (exampleSite (cnxSd sd 17) n)) ib18
    -- head forward + the SMOOTHED loss cotangent, at a general target `t`
    let gapB    : Vec (N * (1*768)) := batchMap N (globalAvgPoolFlat 768 7 7) xhead
    let hnB     : Vec (N * 768)     := batchMap N (rowLNVecFlat 1 768 ε w.hG w.hT) gapB
    let logitsB : Vec (N * nC)      := batchMap N (dense w.Wfc w.bfc) hnB
    let g       : Vec (N * nC)      :=
      den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN logitsB t)
    -- backward cotangents (composed from the loss; residual fan-in at each skip, LN-back at each
    -- downsample and at the stem)
    let dyO18 : Vec (N * (768*7*7)) := batchMapAux N (cnxHeadDyXheadChN (h := 7) (w := 7) ε w.hG w.hT w.Wfc w.bfc) xhead g
    let dyO17 : Vec (N * (768*7*7)) := batchMapAuxIdx N (fun n => w.b18.cotInD gf ε (exampleSite (cnxSd sd 17) n)) ib18 dyO18
    let dyO16 : Vec (N * (768*7*7)) := batchMapAuxIdx N (fun n => w.b17.cotInD gf ε (exampleSite (cnxSd sd 16) n)) ib17 dyO17
    let dyD2 : Vec (N * (768*7*7)) := batchMapAuxIdx N (fun n => w.b16.cotInD gf ε (exampleSite (cnxSd sd 15) n)) ib16 dyO16
    let dyO15 : Vec (N * (384*14*14)) := batchMapAux N (w.d2.cotIn (h := 7) (w := 7) ε) ibD2 dyD2
    let dyO14 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b15.cotInD gf ε (exampleSite (cnxSd sd 14) n)) ib15 dyO15
    let dyO13 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b14.cotInD gf ε (exampleSite (cnxSd sd 13) n)) ib14 dyO14
    let dyO12 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b13.cotInD gf ε (exampleSite (cnxSd sd 12) n)) ib13 dyO13
    let dyO11 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b12.cotInD gf ε (exampleSite (cnxSd sd 11) n)) ib12 dyO12
    let dyO10 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b11.cotInD gf ε (exampleSite (cnxSd sd 10) n)) ib11 dyO11
    let dyO9 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b10.cotInD gf ε (exampleSite (cnxSd sd 9) n)) ib10 dyO10
    let dyO8 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b9.cotInD gf ε (exampleSite (cnxSd sd 8) n)) ib9 dyO9
    let dyO7 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b8.cotInD gf ε (exampleSite (cnxSd sd 7) n)) ib8 dyO8
    let dyD1 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b7.cotInD gf ε (exampleSite (cnxSd sd 6) n)) ib7 dyO7
    let dyO6 : Vec (N * (192*28*28)) := batchMapAux N (w.d1.cotIn (h := 14) (w := 14) ε) ibD1 dyD1
    let dyO5 : Vec (N * (192*28*28)) := batchMapAuxIdx N (fun n => w.b6.cotInD gf ε (exampleSite (cnxSd sd 5) n)) ib6 dyO6
    let dyO4 : Vec (N * (192*28*28)) := batchMapAuxIdx N (fun n => w.b5.cotInD gf ε (exampleSite (cnxSd sd 4) n)) ib5 dyO5
    let dyD0 : Vec (N * (192*28*28)) := batchMapAuxIdx N (fun n => w.b4.cotInD gf ε (exampleSite (cnxSd sd 3) n)) ib4 dyO4
    let dyO3 : Vec (N * (96*56*56)) := batchMapAux N (w.d0.cotIn (h := 28) (w := 28) ε) ibD0 dyD0
    let dyO2 : Vec (N * (96*56*56)) := batchMapAuxIdx N (fun n => w.b3.cotInD gf ε (exampleSite (cnxSd sd 2) n)) ib3 dyO3
    let dyO1 : Vec (N * (96*56*56)) := batchMapAuxIdx N (fun n => w.b2.cotInD gf ε (exampleSite (cnxSd sd 1) n)) ib2 dyO2
    let dyStem : Vec (N * (96*56*56)) := batchMapAuxIdx N (fun n => w.b1.cotInD gf ε (exampleSite (cnxSd sd 0) n)) ib1 dyO1
    -- the stem, every block, every downsample, the head, the dense total-loss fold + loss cot
    cnxStemChTiedGBAt N xN epsStr cotN ε w.sW w.sb w.sγ w.sβ bf16 x dyStem
  ∧ w.b1.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 0) ib1 dyO1
  ∧ w.b2.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 1) ib2 dyO2
  ∧ w.b3.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 2) ib3 dyO3
  ∧ w.d0.TiedGB N xN epsStr cotN ε bf16 ibD0 dyD0
  ∧ w.b4.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 3) ib4 dyO4
  ∧ w.b5.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 4) ib5 dyO5
  ∧ w.b6.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 5) ib6 dyO6
  ∧ w.d1.TiedGB N xN epsStr cotN ε bf16 ibD1 dyD1
  ∧ w.b7.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 6) ib7 dyO7
  ∧ w.b8.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 7) ib8 dyO8
  ∧ w.b9.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 8) ib9 dyO9
  ∧ w.b10.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 9) ib10 dyO10
  ∧ w.b11.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 10) ib11 dyO11
  ∧ w.b12.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 11) ib12 dyO12
  ∧ w.b13.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 12) ib13 dyO13
  ∧ w.b14.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 13) ib14 dyO14
  ∧ w.b15.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 14) ib15 dyO15
  ∧ w.d2.TiedGB N xN epsStr cotN ε bf16 ibD2 dyD2
  ∧ w.b16.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 15) ib16 dyO16
  ∧ w.b17.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 16) ib17 dyO17
  ∧ w.b18.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 17) ib18 dyO18
  ∧ cnxHeadChTiedGB N xN epsStr cotN dN ε w.hG w.hT w.Wfc w.bfc xhead g := by
  intro ib1 ib2 ib3 ibD0 ib4 ib5 ib6 ibD1 ib7 ib8 ib9 ib10 ib11 ib12 ib13 ib14 ib15 ibD2 ib16 ib17 ib18 xhead gapB hnB logitsB g dyO18 dyO17 dyO16 dyD2 dyO15 dyO14 dyO13 dyO12 dyO11 dyO10 dyO9 dyO8 dyO7 dyD1 dyO6 dyO5 dyO4 dyD0 dyO3 dyO2 dyO1 dyStem
  refine ⟨cnx_stem_ch_tiedGBAt N xN epsStr cotN ε w.sW w.sb w.sγ w.sβ bf16 x dyStem,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact w.b1.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 0) ib1 dyO1
  · exact w.b2.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 1) ib2 dyO2
  · exact w.b3.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 2) ib3 dyO3
  · exact w.d0.tied_gb N xN epsStr cotN ε bf16 ibD0 dyD0
  · exact w.b4.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 3) ib4 dyO4
  · exact w.b5.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 4) ib5 dyO5
  · exact w.b6.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 5) ib6 dyO6
  · exact w.d1.tied_gb N xN epsStr cotN ε bf16 ibD1 dyD1
  · exact w.b7.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 6) ib7 dyO7
  · exact w.b8.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 7) ib8 dyO8
  · exact w.b9.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 8) ib9 dyO9
  · exact w.b10.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 9) ib10 dyO10
  · exact w.b11.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 10) ib11 dyO11
  · exact w.b12.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 11) ib12 dyO12
  · exact w.b13.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 12) ib13 dyO13
  · exact w.b14.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 13) ib14 dyO14
  · exact w.b15.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 14) ib15 dyO15
  · exact w.d2.tied_gb N xN epsStr cotN ε bf16 ibD2 dyD2
  · exact w.b16.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 15) ib16 dyO16
  · exact w.b17.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 16) ib17 dyO17
  · exact w.b18.tied_gb N xN epsStr cotN ε bf16 (cnxSd sd 17) ib18 dyO18
  · exact cnx_head_ch_tiedGB N xN epsStr cotN dN ε w.hG w.hT w.Wfc w.bfc xhead g

end Proofs.CnxTieGB
