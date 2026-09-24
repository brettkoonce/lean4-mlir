import LeanMlir.Proofs.Foundation.GradNodesB
import LeanMlir.Proofs.Foundation.SmoothedLossCot
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetStepTie

/-! # EfficientNet-B0's T3 §1a TIE at the UN-FUSED gradient and the SMOOTHED loss

`EfficientNetStepTie.lean` ties all 262 parameters of the SGD-inline `efficientnet_train_step.mlir`:
each fused `theta - lr * g` op `den`s to the certified step at the cotangent the emitted backward
chain delivers. This file is that statement re-pointed along the two axes 4b left open
(`planning/archive/proofs_tier_to_paper_nets.md`, "What is NOT done, and is the honest boundary").

⭐ **Axis 1 — the OPTIMIZER FORM.** Every conjunct is at the RAW gradient node (`*GradB`), which is
what `efficientnet_adam_train_step.mlir` and every ImageNet artifact emit; the fused op appears only
in the SGD-inline file. One statement therefore covers AdamW, RMSProp, EMA, the clipped and
drop-path variants and their data-parallel and bf16 twins, because they all consume this node.
4b.1's `GradNodesB` is the fold each conjunct delegates to.

⭐ **Axis 2 — the LOSS.** The capstone's top-of-chain cotangent is
[`Foundation/SmoothedLossCot.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Foundation/SmoothedLossCot.lean)'s, at a GENERAL target: the six-op chain
`softmaxRow → subB → scaleB → addVB → shiftB → divConstB` the batched renders emit, with the target
arriving as the graph input `%onehot` — a soft vector under mixup or cutmix. The fused file pins it
to `softmax − oneHot`, the gradient of plain cross-entropy at a hard label, which no ImageNet
artifact computes. `unrowB` / `rowB` are ResNet-34's casts between the loss chain's one-row-per-
example index and the dense ops' plain per-example width.

## What is NOT new, and why the file is a transformation rather than a proof

Every cotangent chain, every forward activation and every Jacobian witness is
`EfficientNetStepTie.lean`'s, unchanged. The fusion is `rfl` — `*SgdB_eq_grad` says the fused op IS
`theta - lr *` applied to the un-fused one — so each conjunct's proof is the fused file's with the
wrapper peeling dropped, exactly as 4b's folds were. The `lr`, `wN`, `bN`, `gN` and `lrStr` binders
disappear with the wrapper.

⚠ **The head takes `g` as a PARAMETER here.** The fused `enetHeadTied` computes
`g := rowSoftmax(logits) − onehot` internally, which is what pinned that file to the hard label.
Making it a binder is the whole of axis 2: the per-block ties are `forall cot` statements and were
already loss-agnostic, so only the head and the capstone had to move.

⛔ **Conventions carried unchanged from the fused file**: batch BatchNorm (`bnBatchLA`), XLA-`SAME`
at the 3x3/s2 stem and SYMMETRIC at the strided depthwises, swish (no kink, so no smoothness
hypothesis anywhere), and the SE gate's fan-in folded into the block VJPs. ⛔ ONE REPLICA: in
`efficientnetin_emarmsdp64dropdo` every gradient node feeds `allReduceMeanF` — the collective as an
AST node since 4d piece 2 (2026-09-07), until then emitted text and a declared carve-out. Every
statement here is at the per-replica node; `DataParallelNode.lean` composes it with the replica
mean and the tail (`adamW_at_allReduceMeanF`). For the sync-BN data-parallel render (2026-09-21)
`EfficientNetSyncStepTieG.lean` is the whole step: its `efficientnet_net_syncTiedG` says each
all-reduced gradient IS this file's node at `N := R·N` (without drop-path and dropout).
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.EnetTiePoCG

open scoped BigOperators
open Proofs.EnetTiePoC (reassocB bnBackB swBackB sigBackB cInB dInB dStridedInB gapInB seInB
  gateCotB)

/-- **A conv or depthwise bias, tied.** A bias gradient is the channel sum of the conv-output
    cotangent `cot` (network layout), which is what the emitted `bnBetaGradB` node computes; the
    right side states it as a per-channel BatchNorm's β-Jacobian, at the `γ` and activation (both
    zero here) that β's gradient ignores. -/
def ConvBBetaTiedB (N h w : Nat) {oc : Nat} (cotN : String) (ε : ℝ) (b : Vec oc)
    (cot : Vec (N * (oc * h * w))) : Prop :=
  ∀ o : Fin oc,
    den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) (.operand cotN (reassocB N oc h w cot))) o
      = ∑ j : Fin (oc * (N * (h * w))),
          pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) ε (fun _ => 0) β' (fun _ => 0))
               b o j * bnchwFwd N oc h w (reassocB N oc h w cot) j

theorem convBBetaTiedB_holds {N h w oc : Nat} (cotN : String) (ε : ℝ) (b : Vec oc)
    (cot : Vec (N * (oc * h * w))) : ConvBBetaTiedB N h w cotN ε b cot := fun o =>
  ResNet34PoCB.bnBetaGradB_den cotN ε (fun _ => 0) b (fun _ => 0) (reassocB N oc h w cot) o

def enetExpTiedG {N ic mid oc h w r kHd kWd : Nat}
    (xN vN epsStr cotN : String)
    (εe : ℝ) (hεe : 0 < εe) (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd γd βd : Vec mid)
    (Wz1 : Mat mid r) (bz1 : Vec r) (Wz2 : Mat r mid) (bz2 : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) : Prop :=
  -- forward activations
  let ec : Vec (N * (mid * h * w)) := batchMap N (flatConv We be) xin
  let en : Vec (N * (mid * h * w)) := bnBatchLA N mid h w εe γe βe ec
  let er : Vec (N * (mid * h * w)) := swish (N * (mid * h * w)) en
  let dc : Vec (N * (mid * h * w)) := batchMap N (depthwiseFlat Wd bd) er
  let dn : Vec (N * (mid * h * w)) := bnBatchLA N mid h w εd γd βd dc
  let dr : Vec (N * (mid * h * w)) := swish (N * (mid * h * w)) dn
  let s  : Vec (N * mid) := batchMap N (globalAvgPoolFlat mid h w) dr
  let e1 : Vec (N * r) := batchMap N (dense Wz1 bz1) s
  let z  : Vec (N * r) := swish (N * r) e1
  let e2 : Vec (N * mid) := batchMap N (dense Wz2 bz2) z
  let se : Vec (N * (mid * h * w)) := seB N (h := h) (w := w) Wz1 bz1 Wz2 bz2 dr
  let pc : Vec (N * (oc * h * w)) := batchMap N (flatConv Wp bp) se
  -- backward chain cotangents (composed from dyOut)
  let cotPbn : Vec (N * (oc * h * w)) := bnBackB N oc h w εp hεp γp βp pc dyOut
  let cotSeOut : Vec (N * (mid * h * w)) := cInB N Wp bp cotPbn
  let dgate : Vec (N * mid) := gateCotB N mid h w dr cotSeOut
  let cotE2 : Vec (N * mid) := sigBackB (N * mid) e2 dgate
  let cotZ : Vec (N * r) := rowDenseBackFlat N r mid Wz2 cotE2
  let cotE1 : Vec (N * r) := swBackB (N * r) e1 cotZ
  let cotDxSe : Vec (N * (mid * h * w)) := seInB N (h := h) (w := w) Wz1 bz1 Wz2 bz2 dr cotSeOut
  let cotDn : Vec (N * (mid * h * w)) := swBackB (N * (mid * h * w)) dn cotDxSe
  let cotDc : Vec (N * (mid * h * w)) := bnBackB N mid h w εd hεd γd βd dc cotDn
  let cotEr : Vec (N * (mid * h * w)) := dInB N Wd bd cotDc
  let cotEn : Vec (N * (mid * h * w)) := swBackB (N * (mid * h * w)) en cotEr
  let cotEc : Vec (N * (mid * h * w)) := bnBackB N mid h w εe hεe γe βe ec cotEn
  -- expand 1×1 conv (c → mid), cot = cotEc
  ResNet34PoCB.ConvWTiedB N h w xN cotN be xin We cotEc
  ∧ ConvBBetaTiedB N h w cotN εe be cotEc
  ∧ ResNet34PoCB.BnPairTiedB N mid h w vN epsStr cotN εe γe βe (reassocB N mid h w ec)
        (reassocB N mid h w cotEn)
  -- depthwise (stride-1, kHd×kWd), cot = cotDc
  ∧ ResNet34PoCB.DepthwiseWTiedB N h w xN cotN bd er Wd cotDc
  ∧ ConvBBetaTiedB N h w cotN εd bd cotDc
  ∧ ResNet34PoCB.BnPairTiedB N mid h w vN epsStr cotN εd γd βd (reassocB N mid h w dc)
        (reassocB N mid h w cotDn)
  -- SE reduce dense W₁/b₁ (mid → r), cot = cotE1; excite dense W₂/b₂ (r → mid), cot = cotE2
  ∧ ResNet34PoCB.DenseWTiedB N xN cotN s Wz1 bz1 cotE1
  ∧ ResNet34PoCB.DenseBTiedB N cotN (0 : Mat r r) (0 : Vec r) bz1 cotE1
  ∧ ResNet34PoCB.DenseWTiedB N xN cotN z Wz2 bz2 cotE2
  ∧ ResNet34PoCB.DenseBTiedB N cotN (0 : Mat mid mid) (0 : Vec mid) bz2 cotE2
  -- project 1×1 conv (mid → oc), cot = cotPbn
  ∧ ResNet34PoCB.ConvWTiedB N h w xN cotN bp se Wp cotPbn
  ∧ ConvBBetaTiedB N h w cotN εp bp cotPbn
  ∧ ResNet34PoCB.BnPairTiedB N oc h w vN epsStr cotN εp γp βp (reassocB N oc h w pc)
        (reassocB N oc h w dyOut)

theorem enet_exp_tiedG {N ic mid oc h w r kHd kWd : Nat}
    (xN vN epsStr cotN : String)
    (εe : ℝ) (hεe : 0 < εe) (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd γd βd : Vec mid)
    (Wz1 : Mat mid r) (bz1 : Vec r) (Wz2 : Mat r mid) (bz2 : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    enetExpTiedG xN vN epsStr cotN εe hεe εd hεd εp hεp
      We be γe βe Wd bd γd βd Wz1 bz1 Wz2 bz2 Wp bp γp βp xin dyOut := by
  unfold enetExpTiedG
  intro ec en er dc dn dr s e1 z e2 se pc cotPbn cotSeOut dgate cotE2 cotZ cotE1
        cotDxSe cotDn cotDc cotEr cotEn cotEc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN be xin We cotEc idx
  · exact convBBetaTiedB_holds cotN εe be cotEc
  · exact ResNet34PoCB.bnPairTiedB_holds vN epsStr cotN εe γe βe (reassocB N mid h w ec)
          (reassocB N mid h w cotEn)
  · intro idx; exact EnetPoCG.depthwiseWGradB_den xN cotN bd er Wd cotDc idx
  · exact convBBetaTiedB_holds cotN εd bd cotDc
  · exact ResNet34PoCB.bnPairTiedB_holds vN epsStr cotN εd γd βd (reassocB N mid h w dc)
          (reassocB N mid h w cotDn)
  · intro i j; exact ResNet34PoCB.denseWGradB_den xN cotN s Wz1 bz1 cotE1 i j
  · intro j;   exact EnetPoCG.denseBGradB_den cotN (0 : Mat r r) (0 : Vec r) bz1 cotE1 j
  · intro i j; exact ResNet34PoCB.denseWGradB_den xN cotN z Wz2 bz2 cotE2 i j
  · intro j;   exact EnetPoCG.denseBGradB_den cotN (0 : Mat mid mid) (0 : Vec mid) bz2 cotE2 j
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN bp se Wp cotPbn idx
  · exact convBBetaTiedB_holds cotN εp bp cotPbn
  · exact ResNet34PoCB.bnPairTiedB_holds vN epsStr cotN εp γp βp (reassocB N oc h w pc)
          (reassocB N oc h w dyOut)

/-! ## Strided downsampling MBConv block — all 16 params tied (b2/b4/b6/b12)

Same as the expand block EXCEPT the expand stage lives at the block-input grid `2h×2w` and the
depthwise is strided (`depthwiseStridedWeightSgdB`, the expand-side cotangent `cotEr` upsamples `h→2h`
via `dStridedInB`). No skip (spatial+channels change). -/

/-- **Strided downsampling MBConv block, tied.** All 16 params at the real forward (expand at `2h×2w`,
    strided depthwise `2h→h`) + the chain cotangents driven by `dyOut`. -/
def enetStridedTiedG {N ic mid oc h w r kHd kWd : Nat}
    (xN vN epsStr cotN : String)
    (εe : ℝ) (hεe : 0 < εe) (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd γd βd : Vec mid)
    (Wz1 : Mat mid r) (bz1 : Vec r) (Wz2 : Mat r mid) (bz2 : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) : Prop :=
  -- forward activations (expand at 2h×2w, depthwise downsamples to h×w)
  let ec : Vec (N * (mid * (2 * h) * (2 * w))) := batchMap N (flatConv We be) xin
  let en : Vec (N * (mid * (2 * h) * (2 * w))) := bnBatchLA N mid (2 * h) (2 * w) εe γe βe ec
  let er : Vec (N * (mid * (2 * h) * (2 * w))) := swish (N * (mid * (2 * h) * (2 * w))) en
  let dc : Vec (N * (mid * h * w)) := batchMap N (depthwiseStride2Flat Wd bd) er
  let dn : Vec (N * (mid * h * w)) := bnBatchLA N mid h w εd γd βd dc
  let dr : Vec (N * (mid * h * w)) := swish (N * (mid * h * w)) dn
  let s  : Vec (N * mid) := batchMap N (globalAvgPoolFlat mid h w) dr
  let e1 : Vec (N * r) := batchMap N (dense Wz1 bz1) s
  let z  : Vec (N * r) := swish (N * r) e1
  let e2 : Vec (N * mid) := batchMap N (dense Wz2 bz2) z
  let se : Vec (N * (mid * h * w)) := seB N (h := h) (w := w) Wz1 bz1 Wz2 bz2 dr
  let pc : Vec (N * (oc * h * w)) := batchMap N (flatConv Wp bp) se
  -- backward chain cotangents
  let cotPbn : Vec (N * (oc * h * w)) := bnBackB N oc h w εp hεp γp βp pc dyOut
  let cotSeOut : Vec (N * (mid * h * w)) := cInB N Wp bp cotPbn
  let dgate : Vec (N * mid) := gateCotB N mid h w dr cotSeOut
  let cotE2 : Vec (N * mid) := sigBackB (N * mid) e2 dgate
  let cotZ : Vec (N * r) := rowDenseBackFlat N r mid Wz2 cotE2
  let cotE1 : Vec (N * r) := swBackB (N * r) e1 cotZ
  let cotDxSe : Vec (N * (mid * h * w)) := seInB N (h := h) (w := w) Wz1 bz1 Wz2 bz2 dr cotSeOut
  let cotDn : Vec (N * (mid * h * w)) := swBackB (N * (mid * h * w)) dn cotDxSe
  let cotDc : Vec (N * (mid * h * w)) := bnBackB N mid h w εd hεd γd βd dc cotDn
  let cotEr : Vec (N * (mid * (2 * h) * (2 * w))) := dStridedInB N Wd bd cotDc
  let cotEn : Vec (N * (mid * (2 * h) * (2 * w))) := swBackB (N * (mid * (2 * h) * (2 * w))) en cotEr
  let cotEc : Vec (N * (mid * (2 * h) * (2 * w))) := bnBackB N mid (2 * h) (2 * w) εe hεe γe βe ec cotEn
  -- expand 1×1 conv (ic → mid, at 2h×2w), cot = cotEc
  ResNet34PoCB.ConvWTiedB N (2 * h) (2 * w) xN cotN be xin We cotEc
  ∧ ConvBBetaTiedB N (2 * h) (2 * w) cotN εe be cotEc
  ∧ ResNet34PoCB.BnPairTiedB N mid (2 * h) (2 * w) vN epsStr cotN εe γe βe
        (reassocB N mid (2 * h) (2 * w) ec) (reassocB N mid (2 * h) (2 * w) cotEn)
  -- strided depthwise (kHd×kWd, 2h→h), cot = cotDc
  ∧ ResNet34PoCB.DepthwiseStridedWTiedB N h w xN cotN bd er Wd cotDc
  ∧ ConvBBetaTiedB N h w cotN εd bd cotDc
  ∧ ResNet34PoCB.BnPairTiedB N mid h w vN epsStr cotN εd γd βd (reassocB N mid h w dc)
        (reassocB N mid h w cotDn)
  -- SE reduce/excite dense (mid → r → mid)
  ∧ ResNet34PoCB.DenseWTiedB N xN cotN s Wz1 bz1 cotE1
  ∧ ResNet34PoCB.DenseBTiedB N cotN (0 : Mat r r) (0 : Vec r) bz1 cotE1
  ∧ ResNet34PoCB.DenseWTiedB N xN cotN z Wz2 bz2 cotE2
  ∧ ResNet34PoCB.DenseBTiedB N cotN (0 : Mat mid mid) (0 : Vec mid) bz2 cotE2
  -- project 1×1 conv (mid → oc), cot = cotPbn
  ∧ ResNet34PoCB.ConvWTiedB N h w xN cotN bp se Wp cotPbn
  ∧ ConvBBetaTiedB N h w cotN εp bp cotPbn
  ∧ ResNet34PoCB.BnPairTiedB N oc h w vN epsStr cotN εp γp βp (reassocB N oc h w pc)
        (reassocB N oc h w dyOut)

theorem enet_strided_tiedG {N ic mid oc h w r kHd kWd : Nat}
    (xN vN epsStr cotN : String)
    (εe : ℝ) (hεe : 0 < εe) (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd γd βd : Vec mid)
    (Wz1 : Mat mid r) (bz1 : Vec r) (Wz2 : Mat r mid) (bz2 : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    enetStridedTiedG xN vN epsStr cotN εe hεe εd hεd εp hεp
      We be γe βe Wd bd γd βd Wz1 bz1 Wz2 bz2 Wp bp γp βp xin dyOut := by
  unfold enetStridedTiedG
  intro ec en er dc dn dr s e1 z e2 se pc cotPbn cotSeOut dgate cotE2 cotZ cotE1
        cotDxSe cotDn cotDc cotEr cotEn cotEc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN be xin We cotEc idx
  · exact convBBetaTiedB_holds cotN εe be cotEc
  · exact ResNet34PoCB.bnPairTiedB_holds vN epsStr cotN εe γe βe
          (reassocB N mid (2 * h) (2 * w) ec) (reassocB N mid (2 * h) (2 * w) cotEn)
  · intro idx; exact EnetPoCG.depthwiseStridedWGradB_den xN cotN bd er Wd cotDc idx
  · exact convBBetaTiedB_holds cotN εd bd cotDc
  · exact ResNet34PoCB.bnPairTiedB_holds vN epsStr cotN εd γd βd (reassocB N mid h w dc)
          (reassocB N mid h w cotDn)
  · intro i j; exact ResNet34PoCB.denseWGradB_den xN cotN s Wz1 bz1 cotE1 i j
  · intro j;   exact EnetPoCG.denseBGradB_den cotN (0 : Mat r r) (0 : Vec r) bz1 cotE1 j
  · intro i j; exact ResNet34PoCB.denseWGradB_den xN cotN z Wz2 bz2 cotE2 i j
  · intro j;   exact EnetPoCG.denseBGradB_den cotN (0 : Mat mid mid) (0 : Vec mid) bz2 cotE2 j
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN bp se Wp cotPbn idx
  · exact convBBetaTiedB_holds cotN εp bp cotPbn
  · exact ResNet34PoCB.bnPairTiedB_holds vN epsStr cotN εp γp βp (reassocB N oc h w pc)
          (reassocB N oc h w dyOut)

/-! ## No-expand MBConv block (b1, t=1) — all 12 params tied (depthwise on `ic` → SE → project)

NO expand conv: the depthwise runs directly on the block input (`ic` channels). 12 params (4 depthwise+BN,
4 SE, 4 project). The SE squeeze/excite is on `ic` channels (`ic → r → ic`). -/

/-- **No-expand MBConv block, tied.** All 12 params at the real forward + chain cotangents. -/
def enetNoExpTiedG {N ic oc h w r kHd kWd : Nat}
    (xN vN epsStr cotN : String)
    (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (Wd : DepthwiseKernel ic kHd kWd) (bd γd βd : Vec ic)
    (Wz1 : Mat ic r) (bz1 : Vec r) (Wz2 : Mat r ic) (bz2 : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) : Prop :=
  -- forward activations (depthwise on the block input ic, no expand)
  let dc : Vec (N * (ic * h * w)) := batchMap N (depthwiseFlat Wd bd) xin
  let dn : Vec (N * (ic * h * w)) := bnBatchLA N ic h w εd γd βd dc
  let dr : Vec (N * (ic * h * w)) := swish (N * (ic * h * w)) dn
  let s  : Vec (N * ic) := batchMap N (globalAvgPoolFlat ic h w) dr
  let e1 : Vec (N * r) := batchMap N (dense Wz1 bz1) s
  let z  : Vec (N * r) := swish (N * r) e1
  let e2 : Vec (N * ic) := batchMap N (dense Wz2 bz2) z
  let se : Vec (N * (ic * h * w)) := seB N (h := h) (w := w) Wz1 bz1 Wz2 bz2 dr
  let pc : Vec (N * (oc * h * w)) := batchMap N (flatConv Wp bp) se
  -- backward chain cotangents
  let cotPbn : Vec (N * (oc * h * w)) := bnBackB N oc h w εp hεp γp βp pc dyOut
  let cotSeOut : Vec (N * (ic * h * w)) := cInB N Wp bp cotPbn
  let dgate : Vec (N * ic) := gateCotB N ic h w dr cotSeOut
  let cotE2 : Vec (N * ic) := sigBackB (N * ic) e2 dgate
  let cotZ : Vec (N * r) := rowDenseBackFlat N r ic Wz2 cotE2
  let cotE1 : Vec (N * r) := swBackB (N * r) e1 cotZ
  let cotDxSe : Vec (N * (ic * h * w)) := seInB N (h := h) (w := w) Wz1 bz1 Wz2 bz2 dr cotSeOut
  let cotDn : Vec (N * (ic * h * w)) := swBackB (N * (ic * h * w)) dn cotDxSe
  let cotDc : Vec (N * (ic * h * w)) := bnBackB N ic h w εd hεd γd βd dc cotDn
  -- depthwise (stride-1, kHd×kWd, on ic), cot = cotDc
  ResNet34PoCB.DepthwiseWTiedB N h w xN cotN bd xin Wd cotDc
  ∧ ConvBBetaTiedB N h w cotN εd bd cotDc
  ∧ ResNet34PoCB.BnPairTiedB N ic h w vN epsStr cotN εd γd βd (reassocB N ic h w dc)
        (reassocB N ic h w cotDn)
  -- SE reduce/excite dense (ic → r → ic)
  ∧ ResNet34PoCB.DenseWTiedB N xN cotN s Wz1 bz1 cotE1
  ∧ ResNet34PoCB.DenseBTiedB N cotN (0 : Mat r r) (0 : Vec r) bz1 cotE1
  ∧ ResNet34PoCB.DenseWTiedB N xN cotN z Wz2 bz2 cotE2
  ∧ ResNet34PoCB.DenseBTiedB N cotN (0 : Mat ic ic) (0 : Vec ic) bz2 cotE2
  -- project 1×1 conv (ic → oc), cot = cotPbn
  ∧ ResNet34PoCB.ConvWTiedB N h w xN cotN bp se Wp cotPbn
  ∧ ConvBBetaTiedB N h w cotN εp bp cotPbn
  ∧ ResNet34PoCB.BnPairTiedB N oc h w vN epsStr cotN εp γp βp (reassocB N oc h w pc)
        (reassocB N oc h w dyOut)

theorem enet_noexp_tiedG {N ic oc h w r kHd kWd : Nat}
    (xN vN epsStr cotN : String)
    (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (Wd : DepthwiseKernel ic kHd kWd) (bd γd βd : Vec ic)
    (Wz1 : Mat ic r) (bz1 : Vec r) (Wz2 : Mat r ic) (bz2 : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    enetNoExpTiedG xN vN epsStr cotN εd hεd εp hεp
      Wd bd γd βd Wz1 bz1 Wz2 bz2 Wp bp γp βp xin dyOut := by
  unfold enetNoExpTiedG
  intro dc dn dr s e1 z e2 se pc cotPbn cotSeOut dgate cotE2 cotZ cotE1 cotDxSe cotDn cotDc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact EnetPoCG.depthwiseWGradB_den xN cotN bd xin Wd cotDc idx
  · exact convBBetaTiedB_holds cotN εd bd cotDc
  · exact ResNet34PoCB.bnPairTiedB_holds vN epsStr cotN εd γd βd (reassocB N ic h w dc)
          (reassocB N ic h w cotDn)
  · intro i j; exact ResNet34PoCB.denseWGradB_den xN cotN s Wz1 bz1 cotE1 i j
  · intro j;   exact EnetPoCG.denseBGradB_den cotN (0 : Mat r r) (0 : Vec r) bz1 cotE1 j
  · intro i j; exact ResNet34PoCB.denseWGradB_den xN cotN z Wz2 bz2 cotE2 i j
  · intro j;   exact EnetPoCG.denseBGradB_den cotN (0 : Mat ic ic) (0 : Vec ic) bz2 cotE2 j
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN bp se Wp cotPbn idx
  · exact convBBetaTiedB_holds cotN εp bp cotPbn
  · exact ResNet34PoCB.bnPairTiedB_holds vN epsStr cotN εp γp βp (reassocB N oc h w pc)
          (reassocB N oc h w dyOut)

/-! ## Stem — the 3×3/s2 conv-bn-swish (4 params), feeding block 1

`swish(bn(convStride2Xla Ws bs x))`, 3→32 at 224→112, at the XLA-`SAME` phase the shipped stem
uses. The cotangent block 1 delivers at the stem swish output (`dyStem`) lifts through swish-back
+ true-BN-back to the conv-out cotangent (the `convStridedXlaWeightSgdB` consumes it; NO conv-back
past `%x`). 4 params. -/

/-- **Stem, tied.** The 3×3/s2 conv (`Ws`/`bs`) + its true-BN (`γs`/`βs`) at the real stem forward +
    the cotangent through the stem swish (no maxpool, no conv-back). -/
def enetStemTiedG {N ic oc h w kHs kWs : Nat}
    (xN vN epsStr cotN : String) (εs : ℝ) (hεs : 0 < εs)
    (Ws : Kernel4 oc ic kHs kWs) (bs γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (dyStem : Vec (N * (oc * h * w))) : Prop :=
  let stc : Vec (N * (oc * h * w)) := batchMap N (flatConvStride2Xla Ws bs) x
  let stn : Vec (N * (oc * h * w)) := bnBatchLA N oc h w εs γs βs stc
  let cotBnS : Vec (N * (oc * h * w)) := swBackB (N * (oc * h * w)) stn dyStem
  let cotStc : Vec (N * (oc * h * w)) := bnBackB N oc h w εs hεs γs βs stc cotBnS
  ResNet34PoCB.ConvStridedXlaWTiedB N h w xN cotN bs x Ws cotStc
  ∧ ConvBBetaTiedB N h w cotN εs bs cotStc
  ∧ ResNet34PoCB.BnPairTiedB N oc h w vN epsStr cotN εs γs βs (reassocB N oc h w stc)
        (reassocB N oc h w cotBnS)

theorem enet_stem_tiedG {N ic oc h w kHs kWs : Nat}
    (xN vN epsStr cotN : String) (εs : ℝ) (hεs : 0 < εs)
    (Ws : Kernel4 oc ic kHs kWs) (bs γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (dyStem : Vec (N * (oc * h * w))) :
    enetStemTiedG xN vN epsStr cotN εs hεs Ws bs γs βs x dyStem := by
  unfold enetStemTiedG
  intro stc stn cotBnS cotStc
  refine ⟨?_, ?_, ?_⟩
  · intro idx; exact EnetPoCG.convStridedXlaWGradB_den xN cotN bs x Ws cotStc idx
  · exact convBBetaTiedB_holds cotN εs bs cotStc
  · exact ResNet34PoCB.bnPairTiedB_holds vN epsStr cotN εs γs βs (reassocB N oc h w stc)
          (reassocB N oc h w cotBnS)

/-! ## Head — the 1×1 conv-bn-swish (4 params) → GAP → dense (Wfc/bfc), + the loss cotangent

`dense(GAP(swish(bn(conv Wh bh)))))` (320→1280 conv, GAP, 1280→nClasses dense), then the batched
per-row softmax-CE gradient `g = rowSoftmax(logits) − onehot`. The head conv params tie at the chain
cotangent (loss → dense-back → GAP-back → swish/BN-back); the dense Wfc/bfc tie at the loss cotangent
`g` directly. -/

/-- **Head, tied.** The 4 head conv-bn params + the 2 dense params (Wfc/bfc) denote the certified step
    at the real head forward + the loss-driven cotangent `g = rowSoftmax(logits) − onehot`. -/
def enetHeadTiedG {N c oc h w nC : Nat}
    (xN vN epsStr cotN dN : String) (εh : ℝ) (hεh : 0 < εh)
    (Wh : Kernel4 oc c 1 1) (bh γh βh : Vec oc) (Wfc : Mat oc nC) (bfc : Vec nC)
    (xhead : Vec (N * (c * h * w))) (g : Vec (N * nC)) : Prop :=
  let hc : Vec (N * (oc * h * w)) := batchMap N (flatConv Wh bh) xhead
  let hn : Vec (N * (oc * h * w)) := bnBatchLA N oc h w εh γh βh hc
  let hr : Vec (N * (oc * h * w)) := swish (N * (oc * h * w)) hn
  let a_gap : Vec (N * oc) := batchMap N (globalAvgPoolFlat oc h w) hr
  -- ⚠ the logits are no longer read here: `g` is a PARAMETER, which is axis 2.
  let _logits : Vec (N * nC) := batchMap N (dense Wfc bfc) a_gap
  let cotGapIn : Vec (N * oc) := rowDenseBackFlat N oc nC Wfc g
  let cotHr : Vec (N * (oc * h * w)) := gapInB N oc h w cotGapIn
  let cotHsw : Vec (N * (oc * h * w)) := swBackB (N * (oc * h * w)) hn cotHr
  let cotHbn : Vec (N * (oc * h * w)) := bnBackB N oc h w εh hεh γh βh hc cotHsw
  -- head 1×1 conv (c → oc), cot = cotHbn
  ResNet34PoCB.ConvWTiedB N h w xN cotN bh xhead Wh cotHbn
  ∧ ConvBBetaTiedB N h w cotN εh bh cotHbn
  ∧ ResNet34PoCB.BnPairTiedB N oc h w vN epsStr cotN εh γh βh (reassocB N oc h w hc)
        (reassocB N oc h w cotHsw)
  -- dense classifier (oc → nC), cot = g (the batched softmax-CE gradient)
  ∧ ResNet34PoCB.DenseWTiedB N dN cotN a_gap Wfc bfc g
  ∧ ResNet34PoCB.DenseBTiedB N cotN (0 : Mat nC nC) (0 : Vec nC) bfc g

theorem enet_head_tiedG {N c oc h w nC : Nat}
    (xN vN epsStr cotN dN : String) (εh : ℝ) (hεh : 0 < εh)
    (Wh : Kernel4 oc c 1 1) (bh γh βh : Vec oc) (Wfc : Mat oc nC) (bfc : Vec nC)
    (xhead : Vec (N * (c * h * w))) (g : Vec (N * nC)) :
    enetHeadTiedG xN vN epsStr cotN dN εh hεh Wh bh γh βh Wfc bfc xhead g := by
  unfold enetHeadTiedG
  intro hc hn hr a_gap _logits cotGapIn cotHr cotHsw cotHbn
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN bh xhead Wh cotHbn idx
  · exact convBBetaTiedB_holds cotN εh bh cotHbn
  · exact ResNet34PoCB.bnPairTiedB_holds vN epsStr cotN εh γh βh (reassocB N oc h w hc)
          (reassocB N oc h w cotHsw)
  · intro i j; exact ResNet34PoCB.denseWGradB_den dN cotN a_gap Wfc bfc g i j
  · intro j;   exact EnetPoCG.denseBGradB_den cotN (0 : Mat nC nC) (0 : Vec nC) bfc g j

/-! ## Bundle-taking `*TiedAt` wrappers — one per block type, for the whole-net thread

Each takes the `B0Weights` block bundle (`MBW`/`MBWNoExp`) + its ε-positivity + the block input + the
downstream cotangent `dyOut`, and delegates to the per-block-type tie. -/

def enetExpTiedGAt (xN vN epsStr cotN : String) {N ic mid oc r kh kw : Nat}
    (h w : Nat) (p : MBW ic mid oc r kh kw) (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) : Prop :=
  enetExpTiedG xN vN epsStr cotN p.eε he p.dε hd p.pε hp
    p.eW p.eb p.eγ p.eβ p.dW p.db p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ xin dyOut

theorem enet_exp_tiedGAt (xN vN epsStr cotN : String) {N ic mid oc r kh kw : Nat}
    (h w : Nat) (p : MBW ic mid oc r kh kw) (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    enetExpTiedGAt xN vN epsStr cotN h w p he hd hp xin dyOut := by
  unfold enetExpTiedGAt
  exact enet_exp_tiedG xN vN epsStr cotN p.eε he p.dε hd p.pε hp
    p.eW p.eb p.eγ p.eβ p.dW p.db p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ xin dyOut

def enetStridedTiedGAt (xN vN epsStr cotN : String) {N ic mid oc r kh kw : Nat}
    (h w : Nat) (p : MBW ic mid oc r kh kw) (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) : Prop :=
  enetStridedTiedG xN vN epsStr cotN p.eε he p.dε hd p.pε hp
    p.eW p.eb p.eγ p.eβ p.dW p.db p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ xin dyOut

theorem enet_strided_tiedGAt (xN vN epsStr cotN : String) {N ic mid oc r kh kw : Nat}
    (h w : Nat) (p : MBW ic mid oc r kh kw) (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    enetStridedTiedGAt xN vN epsStr cotN h w p he hd hp xin dyOut := by
  unfold enetStridedTiedGAt
  exact enet_strided_tiedG xN vN epsStr cotN p.eε he p.dε hd p.pε hp
    p.eW p.eb p.eγ p.eβ p.dW p.db p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ xin dyOut

def enetNoExpTiedGAt (xN vN epsStr cotN : String) {N ic oc r kh kw : Nat}
    (h w : Nat) (p : MBWNoExp ic oc r kh kw) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) : Prop :=
  enetNoExpTiedG xN vN epsStr cotN p.dε hd p.pε hp
    p.dW p.db p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ xin dyOut

theorem enet_noexp_tiedGAt (xN vN epsStr cotN : String) {N ic oc r kh kw : Nat}
    (h w : Nat) (p : MBWNoExp ic oc r kh kw) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    enetNoExpTiedGAt xN vN epsStr cotN h w p hd hp xin dyOut := by
  unfold enetNoExpTiedGAt
  exact enet_noexp_tiedG xN vN epsStr cotN p.dε hd p.pε hp
    p.dW p.db p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ xin dyOut

/-- **The whole 16-MBConv EfficientNet-B0 train step, tied at the GRADIENT nodes and the
    SMOOTHED loss.** Threading the real batched (true-BN + SE) forward
    `efficientnetForwardB_full` and the backward cotangent chain (swish masks, the SE
    gate fan-in, true-BN backs, the residual fan-in folded into the block VJPs), the stem, all 16
    MBConv blocks, the conv-bn-swish head, and the dense head all denote the certified batched Σ_n
    loss-descent step. -/
theorem efficientnet_net_tiedG (xN vN epsStr cotN dN : String) (N : Nat) (w : B0Weights)
    (hεw : w.EpsPos)
    (aStr negAK bStr logN ohN : String) (α B : ℝ)
    (x : Vec (N * (3 * 224 * 224))) (t : Vec (N * (1 * 10))) :
    -- forward block inputs (the prefixes of efficientnetForwardB_full)
    let a0  : Vec (N * (32 * 112 * 112)) := stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ x
    let a1  : Vec (N * (16 * 112 * 112)) := mbNoExpW N 112 112 w.b1 a0
    let a2  : Vec (N * (24 * 56 * 56))   := mbStridedW N 56 56 w.b2 a1
    let a3  : Vec (N * (24 * 56 * 56))   := mbResidW N 56 56 w.b3 a2
    let a4  : Vec (N * (40 * 28 * 28))   := mbStridedW N 28 28 w.b4 a3
    let a5  : Vec (N * (40 * 28 * 28))   := mbResidW N 28 28 w.b5 a4
    let a6  : Vec (N * (80 * 14 * 14))   := mbStridedW N 14 14 w.b6 a5
    let a7  : Vec (N * (80 * 14 * 14))   := mbResidW N 14 14 w.b7 a6
    let a8  : Vec (N * (80 * 14 * 14))   := mbResidW N 14 14 w.b8 a7
    let a9  : Vec (N * (112 * 14 * 14))  := mbExpW N 14 14 w.b9 a8
    let a10 : Vec (N * (112 * 14 * 14))  := mbResidW N 14 14 w.b10 a9
    let a11 : Vec (N * (112 * 14 * 14))  := mbResidW N 14 14 w.b11 a10
    let a12 : Vec (N * (192 * 7 * 7))    := mbStridedW N 7 7 w.b12 a11
    let a13 : Vec (N * (192 * 7 * 7))    := mbResidW N 7 7 w.b13 a12
    let a14 : Vec (N * (192 * 7 * 7))    := mbResidW N 7 7 w.b14 a13
    let a15 : Vec (N * (192 * 7 * 7))    := mbResidW N 7 7 w.b15 a14
    let a16 : Vec (N * (320 * 7 * 7))    := mbExpW N 7 7 w.b16 a15
    -- loss cotangent + backward block-output cotangents (composed top-down by the block VJPs)
    let g    : Vec (N * 10) :=
      Proofs.ResNet34TieB.unrowB N 10 (den (smoothedLossCotGraph N 10 α B aStr negAK bStr logN ohN
        (Proofs.ResNet34TieB.rowB N 10
          (headFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb a16)) t))
    let dy16 : Vec (N * (320 * 7 * 7))   := (headFwdB_has_vjp N (h := 7) (w := 7) w.hW w.hb w.hε hεw.h w.hγ w.hβ w.fcW w.fcb).backward a16 g
    let dy15 : Vec (N * (192 * 7 * 7))   := (mbExpW_has_vjp N 7 7 w.b16 hεw.b16.e hεw.b16.d hεw.b16.p).backward a15 dy16
    let dy14 : Vec (N * (192 * 7 * 7))   := (mbResidW_has_vjp N 7 7 w.b15 hεw.b15.e hεw.b15.d hεw.b15.p).backward a14 dy15
    let dy13 : Vec (N * (192 * 7 * 7))   := (mbResidW_has_vjp N 7 7 w.b14 hεw.b14.e hεw.b14.d hεw.b14.p).backward a13 dy14
    let dy12 : Vec (N * (192 * 7 * 7))   := (mbResidW_has_vjp N 7 7 w.b13 hεw.b13.e hεw.b13.d hεw.b13.p).backward a12 dy13
    let dy11 : Vec (N * (112 * 14 * 14)) := (mbStridedW_has_vjp N 7 7 w.b12 hεw.b12.e hεw.b12.d hεw.b12.p).backward a11 dy12
    let dy10 : Vec (N * (112 * 14 * 14)) := (mbResidW_has_vjp N 14 14 w.b11 hεw.b11.e hεw.b11.d hεw.b11.p).backward a10 dy11
    let dy9  : Vec (N * (112 * 14 * 14)) := (mbResidW_has_vjp N 14 14 w.b10 hεw.b10.e hεw.b10.d hεw.b10.p).backward a9 dy10
    let dy8  : Vec (N * (80 * 14 * 14))  := (mbExpW_has_vjp N 14 14 w.b9 hεw.b9.e hεw.b9.d hεw.b9.p).backward a8 dy9
    let dy7  : Vec (N * (80 * 14 * 14))  := (mbResidW_has_vjp N 14 14 w.b8 hεw.b8.e hεw.b8.d hεw.b8.p).backward a7 dy8
    let dy6  : Vec (N * (80 * 14 * 14))  := (mbResidW_has_vjp N 14 14 w.b7 hεw.b7.e hεw.b7.d hεw.b7.p).backward a6 dy7
    let dy5  : Vec (N * (40 * 28 * 28))  := (mbStridedW_has_vjp N 14 14 w.b6 hεw.b6.e hεw.b6.d hεw.b6.p).backward a5 dy6
    let dy4  : Vec (N * (40 * 28 * 28))  := (mbResidW_has_vjp N 28 28 w.b5 hεw.b5.e hεw.b5.d hεw.b5.p).backward a4 dy5
    let dy3  : Vec (N * (24 * 56 * 56))  := (mbStridedW_has_vjp N 28 28 w.b4 hεw.b4.e hεw.b4.d hεw.b4.p).backward a3 dy4
    let dy2  : Vec (N * (24 * 56 * 56))  := (mbResidW_has_vjp N 56 56 w.b3 hεw.b3.e hεw.b3.d hεw.b3.p).backward a2 dy3
    let dy1  : Vec (N * (16 * 112 * 112)) := (mbStridedW_has_vjp N 56 56 w.b2 hεw.b2.e hεw.b2.d hεw.b2.p).backward a1 dy2
    let dy0  : Vec (N * (32 * 112 * 112)) := (mbNoExpW_has_vjp N 112 112 w.b1 hεw.b1.d hεw.b1.p).backward a0 dy1
    -- every block + stem + head tied at its real input + threaded output cotangent
    enetStemTiedG xN vN epsStr cotN w.sε hεw.s w.sW w.sb w.sγ w.sβ x dy0
  ∧ enetNoExpTiedGAt xN vN epsStr cotN 112 112 w.b1 hεw.b1.d hεw.b1.p a0 dy1
  ∧ enetStridedTiedGAt xN vN epsStr cotN 56 56 w.b2 hεw.b2.e hεw.b2.d hεw.b2.p a1 dy2
  ∧ enetExpTiedGAt xN vN epsStr cotN 56 56 w.b3 hεw.b3.e hεw.b3.d hεw.b3.p a2 dy3
  ∧ enetStridedTiedGAt xN vN epsStr cotN 28 28 w.b4 hεw.b4.e hεw.b4.d hεw.b4.p a3 dy4
  ∧ enetExpTiedGAt xN vN epsStr cotN 28 28 w.b5 hεw.b5.e hεw.b5.d hεw.b5.p a4 dy5
  ∧ enetStridedTiedGAt xN vN epsStr cotN 14 14 w.b6 hεw.b6.e hεw.b6.d hεw.b6.p a5 dy6
  ∧ enetExpTiedGAt xN vN epsStr cotN 14 14 w.b7 hεw.b7.e hεw.b7.d hεw.b7.p a6 dy7
  ∧ enetExpTiedGAt xN vN epsStr cotN 14 14 w.b8 hεw.b8.e hεw.b8.d hεw.b8.p a7 dy8
  ∧ enetExpTiedGAt xN vN epsStr cotN 14 14 w.b9 hεw.b9.e hεw.b9.d hεw.b9.p a8 dy9
  ∧ enetExpTiedGAt xN vN epsStr cotN 14 14 w.b10 hεw.b10.e hεw.b10.d hεw.b10.p a9 dy10
  ∧ enetExpTiedGAt xN vN epsStr cotN 14 14 w.b11 hεw.b11.e hεw.b11.d hεw.b11.p a10 dy11
  ∧ enetStridedTiedGAt xN vN epsStr cotN 7 7 w.b12 hεw.b12.e hεw.b12.d hεw.b12.p a11 dy12
  ∧ enetExpTiedGAt xN vN epsStr cotN 7 7 w.b13 hεw.b13.e hεw.b13.d hεw.b13.p a12 dy13
  ∧ enetExpTiedGAt xN vN epsStr cotN 7 7 w.b14 hεw.b14.e hεw.b14.d hεw.b14.p a13 dy14
  ∧ enetExpTiedGAt xN vN epsStr cotN 7 7 w.b15 hεw.b15.e hεw.b15.d hεw.b15.p a14 dy15
  ∧ enetExpTiedGAt xN vN epsStr cotN 7 7 w.b16 hεw.b16.e hεw.b16.d hεw.b16.p a15 dy16
  ∧ enetHeadTiedG xN vN epsStr cotN dN w.hε hεw.h w.hW w.hb w.hγ w.hβ w.fcW w.fcb a16 g := by
  intro a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16
        g dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 dy0
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact enet_stem_tiedG xN vN epsStr cotN w.sε hεw.s w.sW w.sb w.sγ w.sβ x dy0
  · exact enet_noexp_tiedGAt xN vN epsStr cotN 112 112 w.b1 hεw.b1.d hεw.b1.p a0 dy1
  · exact enet_strided_tiedGAt xN vN epsStr cotN 56 56 w.b2 hεw.b2.e hεw.b2.d hεw.b2.p a1 dy2
  · exact enet_exp_tiedGAt xN vN epsStr cotN 56 56 w.b3 hεw.b3.e hεw.b3.d hεw.b3.p a2 dy3
  · exact enet_strided_tiedGAt xN vN epsStr cotN 28 28 w.b4 hεw.b4.e hεw.b4.d hεw.b4.p a3 dy4
  · exact enet_exp_tiedGAt xN vN epsStr cotN 28 28 w.b5 hεw.b5.e hεw.b5.d hεw.b5.p a4 dy5
  · exact enet_strided_tiedGAt xN vN epsStr cotN 14 14 w.b6 hεw.b6.e hεw.b6.d hεw.b6.p a5 dy6
  · exact enet_exp_tiedGAt xN vN epsStr cotN 14 14 w.b7 hεw.b7.e hεw.b7.d hεw.b7.p a6 dy7
  · exact enet_exp_tiedGAt xN vN epsStr cotN 14 14 w.b8 hεw.b8.e hεw.b8.d hεw.b8.p a7 dy8
  · exact enet_exp_tiedGAt xN vN epsStr cotN 14 14 w.b9 hεw.b9.e hεw.b9.d hεw.b9.p a8 dy9
  · exact enet_exp_tiedGAt xN vN epsStr cotN 14 14 w.b10 hεw.b10.e hεw.b10.d hεw.b10.p a9 dy10
  · exact enet_exp_tiedGAt xN vN epsStr cotN 14 14 w.b11 hεw.b11.e hεw.b11.d hεw.b11.p a10 dy11
  · exact enet_strided_tiedGAt xN vN epsStr cotN 7 7 w.b12 hεw.b12.e hεw.b12.d hεw.b12.p a11 dy12
  · exact enet_exp_tiedGAt xN vN epsStr cotN 7 7 w.b13 hεw.b13.e hεw.b13.d hεw.b13.p a12 dy13
  · exact enet_exp_tiedGAt xN vN epsStr cotN 7 7 w.b14 hεw.b14.e hεw.b14.d hεw.b14.p a13 dy14
  · exact enet_exp_tiedGAt xN vN epsStr cotN 7 7 w.b15 hεw.b15.e hεw.b15.d hεw.b15.p a14 dy15
  · exact enet_exp_tiedGAt xN vN epsStr cotN 7 7 w.b16 hεw.b16.e hεw.b16.d hεw.b16.p a15 dy16
  · exact enet_head_tiedG xN vN epsStr cotN dN w.hε hεw.h w.hW w.hb w.hγ w.hβ w.fcW w.fcb a16 g

end Proofs.EnetTiePoCG
