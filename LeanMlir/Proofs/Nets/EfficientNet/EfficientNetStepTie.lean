import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFold
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetBackB0

/-! # EfficientNet-B0 train-step tie (§1a, fused SGD) — `efficientnet_net_tied`

Every one of B0's 262 parameters' SGD ops, at the cotangent the rendered net's own backward chain
hands it: the real forward (`efficientnetForwardB_full`) threaded through every parameter op, the
loss cotangent composed back through all 16 MBConv blocks, the residual fan-in at each stride-1
skip and the squeeze-excite gate fan-in included. The batched, un-fused peer (tied at the
`*GradB` nodes) is `EnetTiePoCG.efficientnet_net_tiedG` in `EfficientNetStepTieG`.

## What B0 adds over MobileNetV2's tie

* **swish** masks (`swBackB`) instead of relu6's two-kink `selectMid`, at every conv-bn-swish and
  depthwise-bn-swish stage;
* the **SE gate fan-in**: the cotangent at an SE input is `gate ⊙ dyOut` (the fused
  `seBackBatched` value), and the SE dense parameters' cotangents come from `seReduceB` (the gate
  cotangent `Σ_{h,w}(x⊙dy)`) through `sigmoidBack → denseRowBack(W₂) → swishBack`;
* **true batch-norm** backward (`bnBackB`, the `bnBatchLA` VJP), batch-coupled;
* the strided depthwise back (`dStridedInB`); `reassocB` bridges the conv/swish `(oc·h·w)` and
  BN `(oc·(h·w))` layouts.

`EfficientNetChainClose`'s whole-net backward is `vjp_comp` of the per-block `_has_vjp`s, so the
tie builds explicit chain-cotangent constructors rather than reading them off.

## Contents

* Per block type, every parameter op at the chain cotangent — each a delegation to the §1-fold
  generics `EnetPoC.*`:
  - `enet_exp_tied` (16 params) — the stride-1 expand block: the 9 residual blocks and the two
    widenings b9/b16 (the parameter ops are skip-agnostic; the fan-in lives in the thread);
  - `enet_strided_tied` (16) — b2/4/6/12: expand at `2h×2w`, strided depthwise;
  - `enet_noexp_tied` (12) — b1 (`t = 1`: depthwise on `ic` → SE → project);
  - `enet_stem_tied` (4) — the 3×3/s2 conv-bn-swish stem;
  - `enet_head_tied` (6) — the 1×1 conv-bn-swish head and the dense, which ties at the loss
    cotangent `g`.
* `efficientnet_net_tied` — the whole-net thread. Block inputs are the forward's prefixes
  (`a0..a16`); the per-block output cotangents (`dy0..dy16`) are composed top-down by the proven
  block VJPs (`headFwdB_has_vjp`, `mb{Exp,Resid,Strided,NoExp}W_has_vjp`) from `g`. The residual
  fan-in is in `mbResidW`'s own VJP (it includes the `+ x`). `@[irreducible]` `*TiedAt` wrappers
  keep the 16-deep thread opaque to the elaborator.

Not done: the dense head's total-loss fold (`Wfc → ∂CE/∂Wfc`, the batched `Σ_n` analogue of
`mlp_output_total_loss_grad`); the head dense ties at `g` directly. -/

open Proofs Proofs.StableHLO

namespace Proofs.EnetTiePoC

open scoped BigOperators

/-! ## Residual stride-1 MBConv block — all 16 params tied (expand → dw → SE → project + skip)

The centerpiece: exercises the genuinely-new content vs mnv2 — swish masks (smooth), the SE gate
fan-in (`gateCotB → sigBackB → {zW₂,zb₂} → rowDenseBackFlat → swBackB → {zW₁,zb₁}`), and true
batch-norm backward (`bnBackB`), all at the batched index `N·(c·h·w)`. Backward from `dyOut` (cot at
project-BN out): project-BN-back → project-conv-back (cot at SE out) → SE backward (fused `dx` for the
depthwise side; un-fused gate-cot for the SE params) → depthwise swish/BN/conv backs → expand
swish/BN/conv backs. Residual (`ic=oc=c`): the block-input cotangent fan-in `+ dyOut` lives in the
whole-net thread, not here (the param ops are skip-agnostic — identical to the no-skip widenings). -/

/-- **Residual stride-1 MBConv block, tied.** All 16 params (expand/project 1×1 conv W+b, depthwise
    W+b, SE reduce/excite dense W₁/b₁/W₂/b₂, three true-BN γ/β) denote the certified batched Σ_n
    loss-descent step at the real block forward activations + the chain cotangents driven by `dyOut`. -/
def enetExpTied {N ic mid oc h w r kHd kWd : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (εe : ℝ) (hεe : 0 < εe) (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd γd βd : Vec mid)
    (Wz1 : Mat mid r) (bz1 : Vec r) (Wz2 : Mat r mid) (bz2 : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) : Prop :=
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
  EnetPoC.ConvWSgdTiedB N h w xN wN lrStr cotN be xin We cotEc lr
  ∧ (∀ o : Fin mid,
        den (SHlo.bnBetaSgdB bN lrStr be lr (.operand cotN (reassocB N mid h w cotEc))) o
          = be o - lr * ∑ j : Fin (mid * (N * (h * w))),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (N * (h * w)) εe (fun _ => 0) β' (fun _ => 0))
                   be o j * bnchwFwd N mid h w (reassocB N mid h w cotEc) j)
  ∧ EnetPoC.BnSgdPairTiedB N mid h w gN vN epsStr bN lrStr cotN εe γe βe (reassocB N mid h w ec)
        (reassocB N mid h w cotEn) lr
  -- depthwise (stride-1, kHd×kWd), cot = cotDc
  ∧ EnetPoC.DepthwiseWSgdTiedB N h w xN wN lrStr cotN bd er Wd cotDc lr
  ∧ (∀ o : Fin mid,
        den (SHlo.bnBetaSgdB bN lrStr bd lr (.operand cotN (reassocB N mid h w cotDc))) o
          = bd o - lr * ∑ j : Fin (mid * (N * (h * w))),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (N * (h * w)) εd (fun _ => 0) β' (fun _ => 0))
                   bd o j * bnchwFwd N mid h w (reassocB N mid h w cotDc) j)
  ∧ EnetPoC.BnSgdPairTiedB N mid h w gN vN epsStr bN lrStr cotN εd γd βd (reassocB N mid h w dc)
        (reassocB N mid h w cotDn) lr
  -- SE reduce dense W₁/b₁ (mid → r), cot = cotE1; excite dense W₂/b₂ (r → mid), cot = cotE2
  ∧ EnetPoC.DenseWSgdTiedB N xN wN lrStr cotN s Wz1 bz1 cotE1 lr
  ∧ EnetPoC.DenseBSgdTiedB N bN lrStr cotN (0 : Mat r r) (0 : Vec r) bz1 cotE1 lr
  ∧ EnetPoC.DenseWSgdTiedB N xN wN lrStr cotN z Wz2 bz2 cotE2 lr
  ∧ EnetPoC.DenseBSgdTiedB N bN lrStr cotN (0 : Mat mid mid) (0 : Vec mid) bz2 cotE2 lr
  -- project 1×1 conv (mid → oc), cot = cotPbn
  ∧ EnetPoC.ConvWSgdTiedB N h w xN wN lrStr cotN bp se Wp cotPbn lr
  ∧ (∀ o : Fin oc,
        den (SHlo.bnBetaSgdB bN lrStr bp lr (.operand cotN (reassocB N oc h w cotPbn))) o
          = bp o - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εp (fun _ => 0) β' (fun _ => 0))
                   bp o j * bnchwFwd N oc h w (reassocB N oc h w cotPbn) j)
  ∧ EnetPoC.BnSgdPairTiedB N oc h w gN vN epsStr bN lrStr cotN εp γp βp (reassocB N oc h w pc)
        (reassocB N oc h w dyOut) lr

theorem enet_exp_tied {N ic mid oc h w r kHd kWd : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (εe : ℝ) (hεe : 0 < εe) (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd γd βd : Vec mid)
    (Wz1 : Mat mid r) (bz1 : Vec r) (Wz2 : Mat r mid) (bz2 : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) :
    enetExpTied xN wN bN gN vN epsStr lrStr cotN εe hεe εd hεd εp hεp
      We be γe βe Wd bd γd βd Wz1 bz1 Wz2 bz2 Wp bp γp βp xin dyOut lr := by
  unfold enetExpTied
  intro ec en er dc dn dr s e1 z e2 se pc cotPbn cotSeOut dgate cotE2 cotZ cotE1
        cotDxSe cotDn cotDc cotEr cotEn cotEc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact EnetPoC.convWB_den xN wN lrStr cotN be xin We cotEc lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εe (fun _ => 0) be (fun _ => 0) (reassocB N mid h w cotEc) lr o
  · exact EnetPoC.bnSgdPairTiedB_holds gN vN epsStr bN lrStr cotN εe γe βe
          (reassocB N mid h w ec) (reassocB N mid h w cotEn) lr
  · intro idx; exact EnetPoC.depthwiseWB_den xN wN lrStr cotN bd er Wd cotDc lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εd (fun _ => 0) bd (fun _ => 0) (reassocB N mid h w cotDc) lr o
  · exact EnetPoC.bnSgdPairTiedB_holds gN vN epsStr bN lrStr cotN εd γd βd
          (reassocB N mid h w dc) (reassocB N mid h w cotDn) lr
  · intro i j; exact EnetPoC.denseWB_den xN wN lrStr cotN s Wz1 bz1 cotE1 lr i j
  · intro j;   exact EnetPoC.denseBB_den bN lrStr cotN (0 : Mat r r) (0 : Vec r) bz1 cotE1 lr j
  · intro i j; exact EnetPoC.denseWB_den xN wN lrStr cotN z Wz2 bz2 cotE2 lr i j
  · intro j;   exact EnetPoC.denseBB_den bN lrStr cotN (0 : Mat mid mid) (0 : Vec mid) bz2 cotE2 lr j
  · intro idx; exact EnetPoC.convWB_den xN wN lrStr cotN bp se Wp cotPbn lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εp (fun _ => 0) bp (fun _ => 0) (reassocB N oc h w cotPbn) lr o
  · exact EnetPoC.bnSgdPairTiedB_holds gN vN epsStr bN lrStr cotN εp γp βp
          (reassocB N oc h w pc) (reassocB N oc h w dyOut) lr

/-! ## Strided downsampling MBConv block — all 16 params tied (b2/b4/b6/b12)

Same as the expand block EXCEPT the expand stage lives at the block-input grid `2h×2w` and the
depthwise is strided (`depthwiseStridedWeightSgdB`, the expand-side cotangent `cotEr` upsamples `h→2h`
via `dStridedInB`). No skip (spatial+channels change). -/

/-- **Strided downsampling MBConv block, tied.** All 16 params at the real forward (expand at `2h×2w`,
    strided depthwise `2h→h`) + the chain cotangents driven by `dyOut`. -/
def enetStridedTied {N ic mid oc h w r kHd kWd : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (εe : ℝ) (hεe : 0 < εe) (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd γd βd : Vec mid)
    (Wz1 : Mat mid r) (bz1 : Vec r) (Wz2 : Mat r mid) (bz2 : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) : Prop :=
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
  EnetPoC.ConvWSgdTiedB N (2 * h) (2 * w) xN wN lrStr cotN be xin We cotEc lr
  ∧ (∀ o : Fin mid,
        den (SHlo.bnBetaSgdB bN lrStr be lr (.operand cotN (reassocB N mid (2 * h) (2 * w) cotEc))) o
          = be o - lr * ∑ j : Fin (mid * (N * ((2 * h) * (2 * w)))),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (N * ((2 * h) * (2 * w))) εe (fun _ => 0) β' (fun _ => 0))
                   be o j * bnchwFwd N mid (2 * h) (2 * w) (reassocB N mid (2 * h) (2 * w) cotEc) j)
  ∧ EnetPoC.BnSgdPairTiedB N mid (2 * h) (2 * w) gN vN epsStr bN lrStr cotN εe γe βe
        (reassocB N mid (2 * h) (2 * w) ec) (reassocB N mid (2 * h) (2 * w) cotEn) lr
  -- strided depthwise (kHd×kWd, 2h→h), cot = cotDc
  ∧ (∀ idx : Fin (mid * kHd * kWd),
        den (SHlo.depthwiseStridedWeightSgdB xN wN lrStr bd er Wd lr (.operand cotN cotDc)) idx
          = Tensor3.flatten Wd idx - lr * ∑ n : Fin N, ∑ j : Fin (mid * h * w),
              pdiv (fun v' : Vec (mid * kHd * kWd) =>
                      depthwiseStride2Flat (Tensor3.unflatten v') bd (batchSlice N (mid * (2 * h) * (2 * w)) er n))
                   (Tensor3.flatten Wd) idx j * batchSlice N (mid * h * w) cotDc n j)
  ∧ (∀ o : Fin mid,
        den (SHlo.bnBetaSgdB bN lrStr bd lr (.operand cotN (reassocB N mid h w cotDc))) o
          = bd o - lr * ∑ j : Fin (mid * (N * (h * w))),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (N * (h * w)) εd (fun _ => 0) β' (fun _ => 0))
                   bd o j * bnchwFwd N mid h w (reassocB N mid h w cotDc) j)
  ∧ EnetPoC.BnSgdPairTiedB N mid h w gN vN epsStr bN lrStr cotN εd γd βd (reassocB N mid h w dc)
        (reassocB N mid h w cotDn) lr
  -- SE reduce/excite dense (mid → r → mid)
  ∧ EnetPoC.DenseWSgdTiedB N xN wN lrStr cotN s Wz1 bz1 cotE1 lr
  ∧ EnetPoC.DenseBSgdTiedB N bN lrStr cotN (0 : Mat r r) (0 : Vec r) bz1 cotE1 lr
  ∧ EnetPoC.DenseWSgdTiedB N xN wN lrStr cotN z Wz2 bz2 cotE2 lr
  ∧ EnetPoC.DenseBSgdTiedB N bN lrStr cotN (0 : Mat mid mid) (0 : Vec mid) bz2 cotE2 lr
  -- project 1×1 conv (mid → oc), cot = cotPbn
  ∧ EnetPoC.ConvWSgdTiedB N h w xN wN lrStr cotN bp se Wp cotPbn lr
  ∧ (∀ o : Fin oc,
        den (SHlo.bnBetaSgdB bN lrStr bp lr (.operand cotN (reassocB N oc h w cotPbn))) o
          = bp o - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εp (fun _ => 0) β' (fun _ => 0))
                   bp o j * bnchwFwd N oc h w (reassocB N oc h w cotPbn) j)
  ∧ EnetPoC.BnSgdPairTiedB N oc h w gN vN epsStr bN lrStr cotN εp γp βp (reassocB N oc h w pc)
        (reassocB N oc h w dyOut) lr

theorem enet_strided_tied {N ic mid oc h w r kHd kWd : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (εe : ℝ) (hεe : 0 < εe) (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd γd βd : Vec mid)
    (Wz1 : Mat mid r) (bz1 : Vec r) (Wz2 : Mat r mid) (bz2 : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) :
    enetStridedTied xN wN bN gN vN epsStr lrStr cotN εe hεe εd hεd εp hεp
      We be γe βe Wd bd γd βd Wz1 bz1 Wz2 bz2 Wp bp γp βp xin dyOut lr := by
  unfold enetStridedTied
  intro ec en er dc dn dr s e1 z e2 se pc cotPbn cotSeOut dgate cotE2 cotZ cotE1
        cotDxSe cotDn cotDc cotEr cotEn cotEc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact EnetPoC.convWB_den xN wN lrStr cotN be xin We cotEc lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εe (fun _ => 0) be (fun _ => 0) (reassocB N mid (2 * h) (2 * w) cotEc) lr o
  · exact EnetPoC.bnSgdPairTiedB_holds gN vN epsStr bN lrStr cotN εe γe βe
          (reassocB N mid (2 * h) (2 * w) ec) (reassocB N mid (2 * h) (2 * w) cotEn) lr
  · intro idx; exact EnetPoC.depthwiseStridedWB_den xN wN lrStr cotN bd er Wd cotDc lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εd (fun _ => 0) bd (fun _ => 0) (reassocB N mid h w cotDc) lr o
  · exact EnetPoC.bnSgdPairTiedB_holds gN vN epsStr bN lrStr cotN εd γd βd
          (reassocB N mid h w dc) (reassocB N mid h w cotDn) lr
  · intro i j; exact EnetPoC.denseWB_den xN wN lrStr cotN s Wz1 bz1 cotE1 lr i j
  · intro j;   exact EnetPoC.denseBB_den bN lrStr cotN (0 : Mat r r) (0 : Vec r) bz1 cotE1 lr j
  · intro i j; exact EnetPoC.denseWB_den xN wN lrStr cotN z Wz2 bz2 cotE2 lr i j
  · intro j;   exact EnetPoC.denseBB_den bN lrStr cotN (0 : Mat mid mid) (0 : Vec mid) bz2 cotE2 lr j
  · intro idx; exact EnetPoC.convWB_den xN wN lrStr cotN bp se Wp cotPbn lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εp (fun _ => 0) bp (fun _ => 0) (reassocB N oc h w cotPbn) lr o
  · exact EnetPoC.bnSgdPairTiedB_holds gN vN epsStr bN lrStr cotN εp γp βp
          (reassocB N oc h w pc) (reassocB N oc h w dyOut) lr

/-! ## No-expand MBConv block (b1, t=1) — all 12 params tied (depthwise on `ic` → SE → project)

NO expand conv: the depthwise runs directly on the block input (`ic` channels). 12 params (4 depthwise+BN,
4 SE, 4 project). The SE squeeze/excite is on `ic` channels (`ic → r → ic`). -/

/-- **No-expand MBConv block, tied.** All 12 params at the real forward + chain cotangents. -/
def enetNoExpTied {N ic oc h w r kHd kWd : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (Wd : DepthwiseKernel ic kHd kWd) (bd γd βd : Vec ic)
    (Wz1 : Mat ic r) (bz1 : Vec r) (Wz2 : Mat r ic) (bz2 : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) : Prop :=
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
  EnetPoC.DepthwiseWSgdTiedB N h w xN wN lrStr cotN bd xin Wd cotDc lr
  ∧ (∀ o : Fin ic,
        den (SHlo.bnBetaSgdB bN lrStr bd lr (.operand cotN (reassocB N ic h w cotDc))) o
          = bd o - lr * ∑ j : Fin (ic * (N * (h * w))),
              pdiv (fun β' : Vec ic => bnPerChannelFlat ic (N * (h * w)) εd (fun _ => 0) β' (fun _ => 0))
                   bd o j * bnchwFwd N ic h w (reassocB N ic h w cotDc) j)
  ∧ EnetPoC.BnSgdPairTiedB N ic h w gN vN epsStr bN lrStr cotN εd γd βd (reassocB N ic h w dc)
        (reassocB N ic h w cotDn) lr
  -- SE reduce/excite dense (ic → r → ic)
  ∧ EnetPoC.DenseWSgdTiedB N xN wN lrStr cotN s Wz1 bz1 cotE1 lr
  ∧ EnetPoC.DenseBSgdTiedB N bN lrStr cotN (0 : Mat r r) (0 : Vec r) bz1 cotE1 lr
  ∧ EnetPoC.DenseWSgdTiedB N xN wN lrStr cotN z Wz2 bz2 cotE2 lr
  ∧ EnetPoC.DenseBSgdTiedB N bN lrStr cotN (0 : Mat ic ic) (0 : Vec ic) bz2 cotE2 lr
  -- project 1×1 conv (ic → oc), cot = cotPbn
  ∧ EnetPoC.ConvWSgdTiedB N h w xN wN lrStr cotN bp se Wp cotPbn lr
  ∧ (∀ o : Fin oc,
        den (SHlo.bnBetaSgdB bN lrStr bp lr (.operand cotN (reassocB N oc h w cotPbn))) o
          = bp o - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εp (fun _ => 0) β' (fun _ => 0))
                   bp o j * bnchwFwd N oc h w (reassocB N oc h w cotPbn) j)
  ∧ EnetPoC.BnSgdPairTiedB N oc h w gN vN epsStr bN lrStr cotN εp γp βp (reassocB N oc h w pc)
        (reassocB N oc h w dyOut) lr

theorem enet_noexp_tied {N ic oc h w r kHd kWd : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (Wd : DepthwiseKernel ic kHd kWd) (bd γd βd : Vec ic)
    (Wz1 : Mat ic r) (bz1 : Vec r) (Wz2 : Mat r ic) (bz2 : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) :
    enetNoExpTied xN wN bN gN vN epsStr lrStr cotN εd hεd εp hεp
      Wd bd γd βd Wz1 bz1 Wz2 bz2 Wp bp γp βp xin dyOut lr := by
  unfold enetNoExpTied
  intro dc dn dr s e1 z e2 se pc cotPbn cotSeOut dgate cotE2 cotZ cotE1 cotDxSe cotDn cotDc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact EnetPoC.depthwiseWB_den xN wN lrStr cotN bd xin Wd cotDc lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εd (fun _ => 0) bd (fun _ => 0) (reassocB N ic h w cotDc) lr o
  · exact EnetPoC.bnSgdPairTiedB_holds gN vN epsStr bN lrStr cotN εd γd βd
          (reassocB N ic h w dc) (reassocB N ic h w cotDn) lr
  · intro i j; exact EnetPoC.denseWB_den xN wN lrStr cotN s Wz1 bz1 cotE1 lr i j
  · intro j;   exact EnetPoC.denseBB_den bN lrStr cotN (0 : Mat r r) (0 : Vec r) bz1 cotE1 lr j
  · intro i j; exact EnetPoC.denseWB_den xN wN lrStr cotN z Wz2 bz2 cotE2 lr i j
  · intro j;   exact EnetPoC.denseBB_den bN lrStr cotN (0 : Mat ic ic) (0 : Vec ic) bz2 cotE2 lr j
  · intro idx; exact EnetPoC.convWB_den xN wN lrStr cotN bp se Wp cotPbn lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εp (fun _ => 0) bp (fun _ => 0) (reassocB N oc h w cotPbn) lr o
  · exact EnetPoC.bnSgdPairTiedB_holds gN vN epsStr bN lrStr cotN εp γp βp
          (reassocB N oc h w pc) (reassocB N oc h w dyOut) lr

/-! ## Stem — the 3×3/s2 conv-bn-swish (4 params), feeding block 1

`swish(bn(convStride2Xla Ws bs x))`, 3→32 at 224→112, at the XLA-`SAME` phase the shipped stem
uses. The cotangent block 1 delivers at the stem swish output (`dyStem`) lifts through swish-back
+ true-BN-back to the conv-out cotangent (the `convStridedXlaWeightSgdB` consumes it; NO conv-back
past `%x`). 4 params. -/

/-- **Stem, tied.** The 3×3/s2 conv (`Ws`/`bs`) + its true-BN (`γs`/`βs`) at the real stem forward +
    the cotangent through the stem swish (no maxpool, no conv-back). -/
def enetStemTied {N ic oc h w kHs kWs : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (εs : ℝ) (hεs : 0 < εs)
    (Ws : Kernel4 oc ic kHs kWs) (bs γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (dyStem : Vec (N * (oc * h * w))) (lr : ℝ) : Prop :=
  let stc : Vec (N * (oc * h * w)) := batchMap N (flatConvStride2Xla Ws bs) x
  let stn : Vec (N * (oc * h * w)) := bnBatchLA N oc h w εs γs βs stc
  let cotBnS : Vec (N * (oc * h * w)) := swBackB (N * (oc * h * w)) stn dyStem
  let cotStc : Vec (N * (oc * h * w)) := bnBackB N oc h w εs hεs γs βs stc cotBnS
  (∀ idx : Fin (oc * ic * kHs * kWs),
        den (SHlo.convStridedXlaWeightSgdB xN wN lrStr bs x Ws lr (.operand cotN cotStc)) idx
          = Kernel4.flatten Ws idx - lr * ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * ic * kHs * kWs) =>
                      flatConvStride2Xla (Kernel4.unflatten v') bs (batchSlice N (ic * (2 * h) * (2 * w)) x n))
                   (Kernel4.flatten Ws) idx j * batchSlice N (oc * h * w) cotStc n j)
  ∧ (∀ o : Fin oc,
        den (SHlo.bnBetaSgdB bN lrStr bs lr (.operand cotN (reassocB N oc h w cotStc))) o
          = bs o - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εs (fun _ => 0) β' (fun _ => 0))
                   bs o j * bnchwFwd N oc h w (reassocB N oc h w cotStc) j)
  ∧ EnetPoC.BnSgdPairTiedB N oc h w gN vN epsStr bN lrStr cotN εs γs βs (reassocB N oc h w stc)
        (reassocB N oc h w cotBnS) lr

theorem enet_stem_tied {N ic oc h w kHs kWs : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (εs : ℝ) (hεs : 0 < εs)
    (Ws : Kernel4 oc ic kHs kWs) (bs γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (dyStem : Vec (N * (oc * h * w))) (lr : ℝ) :
    enetStemTied xN wN bN gN vN epsStr lrStr cotN εs hεs Ws bs γs βs x dyStem lr := by
  unfold enetStemTied
  intro stc stn cotBnS cotStc
  refine ⟨?_, ?_, ?_⟩
  · intro idx; exact EnetPoC.convStridedWB_den xN wN lrStr cotN bs x Ws cotStc lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εs (fun _ => 0) bs (fun _ => 0) (reassocB N oc h w cotStc) lr o
  · exact EnetPoC.bnSgdPairTiedB_holds gN vN epsStr bN lrStr cotN εs γs βs
          (reassocB N oc h w stc) (reassocB N oc h w cotBnS) lr

/-! ## Head — the 1×1 conv-bn-swish (4 params) → GAP → dense (Wfc/bfc), + the loss cotangent

`dense(GAP(swish(bn(conv Wh bh)))))` (320→1280 conv, GAP, 1280→nClasses dense), then the batched
per-row softmax-CE gradient `g = rowSoftmax(logits) − onehot`. The head conv params tie at the chain
cotangent (loss → dense-back → GAP-back → swish/BN-back); the dense Wfc/bfc tie at the loss cotangent
`g` directly. -/

/-- **Head, tied.** The 4 head conv-bn params + the 2 dense params (Wfc/bfc) denote the certified step
    at the real head forward + the loss-driven cotangent `g = rowSoftmax(logits) − onehot`. -/
def enetHeadTied {N c oc h w nC : Nat}
    (xN wN bN gN vN epsStr lrStr cotN dN : String) (εh : ℝ) (hεh : 0 < εh)
    (Wh : Kernel4 oc c 1 1) (bh γh βh : Vec oc) (Wfc : Mat oc nC) (bfc : Vec nC)
    (xhead : Vec (N * (c * h * w))) (onehot : Vec (N * nC)) (lr : ℝ) : Prop :=
  let hc : Vec (N * (oc * h * w)) := batchMap N (flatConv Wh bh) xhead
  let hn : Vec (N * (oc * h * w)) := bnBatchLA N oc h w εh γh βh hc
  let hr : Vec (N * (oc * h * w)) := swish (N * (oc * h * w)) hn
  let a_gap : Vec (N * oc) := batchMap N (globalAvgPoolFlat oc h w) hr
  let logits : Vec (N * nC) := batchMap N (dense Wfc bfc) a_gap
  let g : Vec (N * nC) := fun idx => rowSoftmaxFlat N nC logits idx - onehot idx
  let cotGapIn : Vec (N * oc) := rowDenseBackFlat N oc nC Wfc g
  let cotHr : Vec (N * (oc * h * w)) := gapInB N oc h w cotGapIn
  let cotHsw : Vec (N * (oc * h * w)) := swBackB (N * (oc * h * w)) hn cotHr
  let cotHbn : Vec (N * (oc * h * w)) := bnBackB N oc h w εh hεh γh βh hc cotHsw
  -- head 1×1 conv (c → oc), cot = cotHbn
  EnetPoC.ConvWSgdTiedB N h w xN wN lrStr cotN bh xhead Wh cotHbn lr
  ∧ (∀ o : Fin oc,
        den (SHlo.bnBetaSgdB bN lrStr bh lr (.operand cotN (reassocB N oc h w cotHbn))) o
          = bh o - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εh (fun _ => 0) β' (fun _ => 0))
                   bh o j * bnchwFwd N oc h w (reassocB N oc h w cotHbn) j)
  ∧ EnetPoC.BnSgdPairTiedB N oc h w gN vN epsStr bN lrStr cotN εh γh βh (reassocB N oc h w hc)
        (reassocB N oc h w cotHsw) lr
  -- dense classifier (oc → nC), cot = g (the batched softmax-CE gradient)
  ∧ EnetPoC.DenseWSgdTiedB N dN wN lrStr cotN a_gap Wfc bfc g lr
  ∧ EnetPoC.DenseBSgdTiedB N dN lrStr cotN (0 : Mat nC nC) (0 : Vec nC) bfc g lr

theorem enet_head_tied {N c oc h w nC : Nat}
    (xN wN bN gN vN epsStr lrStr cotN dN : String) (εh : ℝ) (hεh : 0 < εh)
    (Wh : Kernel4 oc c 1 1) (bh γh βh : Vec oc) (Wfc : Mat oc nC) (bfc : Vec nC)
    (xhead : Vec (N * (c * h * w))) (onehot : Vec (N * nC)) (lr : ℝ) :
    enetHeadTied xN wN bN gN vN epsStr lrStr cotN dN εh hεh Wh bh γh βh Wfc bfc xhead onehot lr := by
  unfold enetHeadTied
  intro hc hn hr a_gap logits g cotGapIn cotHr cotHsw cotHbn
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact EnetPoC.convWB_den xN wN lrStr cotN bh xhead Wh cotHbn lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εh (fun _ => 0) bh (fun _ => 0) (reassocB N oc h w cotHbn) lr o
  · exact EnetPoC.bnSgdPairTiedB_holds gN vN epsStr bN lrStr cotN εh γh βh
          (reassocB N oc h w hc) (reassocB N oc h w cotHsw) lr
  · intro i j; exact EnetPoC.denseWB_den dN wN lrStr cotN a_gap Wfc bfc g lr i j
  · intro j;   exact EnetPoC.denseBB_den dN lrStr cotN (0 : Mat nC nC) (0 : Vec nC) bfc g lr j

/-! ## `@[irreducible]` bundle-taking `*TiedAt` wrappers — one per block type, for the whole-net thread

Each takes the `B0Weights` block bundle (`MBW`/`MBWNoExp`) + its ε-positivity + the block input + the
downstream cotangent `dyOut`, and delegates to the per-block-type tie. `@[irreducible]` keeps the
16-deep capstone thread opaque to the elaborator (the r34/mnv2 heartbeat lesson). -/

@[irreducible] def enetExpTiedAt (xN wN bN gN vN epsStr lrStr cotN : String) {N ic mid oc r kh kw : Nat}
    (h w : Nat) (p : MBW ic mid oc r kh kw) (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) : Prop :=
  enetExpTied xN wN bN gN vN epsStr lrStr cotN p.eε he p.dε hd p.pε hp
    p.eW p.eb p.eγ p.eβ p.dW p.db p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ xin dyOut lr

theorem enet_exp_tiedAt (xN wN bN gN vN epsStr lrStr cotN : String) {N ic mid oc r kh kw : Nat}
    (h w : Nat) (p : MBW ic mid oc r kh kw) (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) :
    enetExpTiedAt xN wN bN gN vN epsStr lrStr cotN h w p he hd hp xin dyOut lr := by
  unfold enetExpTiedAt
  exact enet_exp_tied xN wN bN gN vN epsStr lrStr cotN p.eε he p.dε hd p.pε hp
    p.eW p.eb p.eγ p.eβ p.dW p.db p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ xin dyOut lr

@[irreducible] def enetStridedTiedAt (xN wN bN gN vN epsStr lrStr cotN : String) {N ic mid oc r kh kw : Nat}
    (h w : Nat) (p : MBW ic mid oc r kh kw) (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) : Prop :=
  enetStridedTied xN wN bN gN vN epsStr lrStr cotN p.eε he p.dε hd p.pε hp
    p.eW p.eb p.eγ p.eβ p.dW p.db p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ xin dyOut lr

theorem enet_strided_tiedAt (xN wN bN gN vN epsStr lrStr cotN : String) {N ic mid oc r kh kw : Nat}
    (h w : Nat) (p : MBW ic mid oc r kh kw) (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) :
    enetStridedTiedAt xN wN bN gN vN epsStr lrStr cotN h w p he hd hp xin dyOut lr := by
  unfold enetStridedTiedAt
  exact enet_strided_tied xN wN bN gN vN epsStr lrStr cotN p.eε he p.dε hd p.pε hp
    p.eW p.eb p.eγ p.eβ p.dW p.db p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ xin dyOut lr

@[irreducible] def enetNoExpTiedAt (xN wN bN gN vN epsStr lrStr cotN : String) {N ic oc r kh kw : Nat}
    (h w : Nat) (p : MBWNoExp ic oc r kh kw) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) : Prop :=
  enetNoExpTied xN wN bN gN vN epsStr lrStr cotN p.dε hd p.pε hp
    p.dW p.db p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ xin dyOut lr

theorem enet_noexp_tiedAt (xN wN bN gN vN epsStr lrStr cotN : String) {N ic oc r kh kw : Nat}
    (h w : Nat) (p : MBWNoExp ic oc r kh kw) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) :
    enetNoExpTiedAt xN wN bN gN vN epsStr lrStr cotN h w p hd hp xin dyOut lr := by
  unfold enetNoExpTiedAt
  exact enet_noexp_tied xN wN bN gN vN epsStr lrStr cotN p.dε hd p.pε hp
    p.dW p.db p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pγ p.pβ xin dyOut lr

/-! ## The whole-net thread — all 262 params tied through the REAL `efficientnetForwardB_full`

The capstone: `efficientnetForwardB_full`'s prefixes are the block inputs (`a0..a16` = stem, then the
16 MBConv blocks), and the per-block output cotangents (`dy0..dy16`) are composed TOP-DOWN by the
proven block VJPs (`headFwdB_has_vjp`, `mb{Exp,Resid,Strided,NoExp}W_has_vjp`) from the loss cotangent
`g = rowSoftmax(logits) − onehot`. Each block's tie then holds at its real input + threaded `dyOut`.
The full §1a tie: the WHOLE 16-MBConv (262-param) EfficientNet-B0 train step is den-composed
forward→loss→backward, no free activations, no symbolic cotangent. The residual fan-in at the 9
identity skips is folded into `mbResidW`'s own VJP (it includes the `+ x`), so it is automatic. -/

/-- **The whole 16-MBConv EfficientNet-B0 train step, tied.** Threading the real batched (true-BN + SE)
    forward `efficientnetForwardB_full` and the loss-driven backward cotangent chain (swish masks, SE
    gate fan-in, true-BN backs, the residual fan-in folded into the block VJPs), the stem, all 16
    MBConv blocks, the conv-bn-swish head, and the dense head all denote the certified batched Σ_n
    loss-descent step. -/
theorem efficientnet_net_tied (xN wN bN gN vN epsStr lrStr cotN dN : String) (N : Nat) (w : B0Weights)
    (hεw : w.EpsPos)
    (x : Vec (N * (3 * 224 * 224))) (onehot : Vec (N * 10)) (lr : ℝ) :
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
    let g    : Vec (N * 10) := fun idx =>
      rowSoftmaxFlat N 10 (headFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb a16) idx
        - onehot idx
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
    enetStemTied xN wN bN gN vN epsStr lrStr cotN w.sε hεw.s w.sW w.sb w.sγ w.sβ x dy0 lr
  ∧ enetNoExpTiedAt xN wN bN gN vN epsStr lrStr cotN 112 112 w.b1 hεw.b1.d hεw.b1.p a0 dy1 lr
  ∧ enetStridedTiedAt xN wN bN gN vN epsStr lrStr cotN 56 56 w.b2 hεw.b2.e hεw.b2.d hεw.b2.p a1 dy2 lr
  ∧ enetExpTiedAt xN wN bN gN vN epsStr lrStr cotN 56 56 w.b3 hεw.b3.e hεw.b3.d hεw.b3.p a2 dy3 lr
  ∧ enetStridedTiedAt xN wN bN gN vN epsStr lrStr cotN 28 28 w.b4 hεw.b4.e hεw.b4.d hεw.b4.p a3 dy4 lr
  ∧ enetExpTiedAt xN wN bN gN vN epsStr lrStr cotN 28 28 w.b5 hεw.b5.e hεw.b5.d hεw.b5.p a4 dy5 lr
  ∧ enetStridedTiedAt xN wN bN gN vN epsStr lrStr cotN 14 14 w.b6 hεw.b6.e hεw.b6.d hεw.b6.p a5 dy6 lr
  ∧ enetExpTiedAt xN wN bN gN vN epsStr lrStr cotN 14 14 w.b7 hεw.b7.e hεw.b7.d hεw.b7.p a6 dy7 lr
  ∧ enetExpTiedAt xN wN bN gN vN epsStr lrStr cotN 14 14 w.b8 hεw.b8.e hεw.b8.d hεw.b8.p a7 dy8 lr
  ∧ enetExpTiedAt xN wN bN gN vN epsStr lrStr cotN 14 14 w.b9 hεw.b9.e hεw.b9.d hεw.b9.p a8 dy9 lr
  ∧ enetExpTiedAt xN wN bN gN vN epsStr lrStr cotN 14 14 w.b10 hεw.b10.e hεw.b10.d hεw.b10.p a9 dy10 lr
  ∧ enetExpTiedAt xN wN bN gN vN epsStr lrStr cotN 14 14 w.b11 hεw.b11.e hεw.b11.d hεw.b11.p a10 dy11 lr
  ∧ enetStridedTiedAt xN wN bN gN vN epsStr lrStr cotN 7 7 w.b12 hεw.b12.e hεw.b12.d hεw.b12.p a11 dy12 lr
  ∧ enetExpTiedAt xN wN bN gN vN epsStr lrStr cotN 7 7 w.b13 hεw.b13.e hεw.b13.d hεw.b13.p a12 dy13 lr
  ∧ enetExpTiedAt xN wN bN gN vN epsStr lrStr cotN 7 7 w.b14 hεw.b14.e hεw.b14.d hεw.b14.p a13 dy14 lr
  ∧ enetExpTiedAt xN wN bN gN vN epsStr lrStr cotN 7 7 w.b15 hεw.b15.e hεw.b15.d hεw.b15.p a14 dy15 lr
  ∧ enetExpTiedAt xN wN bN gN vN epsStr lrStr cotN 7 7 w.b16 hεw.b16.e hεw.b16.d hεw.b16.p a15 dy16 lr
  ∧ enetHeadTied xN wN bN gN vN epsStr lrStr cotN dN w.hε hεw.h w.hW w.hb w.hγ w.hβ w.fcW w.fcb a16 onehot lr := by
  intro a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16
        g dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 dy0
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact enet_stem_tied xN wN bN gN vN epsStr lrStr cotN w.sε hεw.s w.sW w.sb w.sγ w.sβ x dy0 lr
  · exact enet_noexp_tiedAt xN wN bN gN vN epsStr lrStr cotN 112 112 w.b1 hεw.b1.d hεw.b1.p a0 dy1 lr
  · exact enet_strided_tiedAt xN wN bN gN vN epsStr lrStr cotN 56 56 w.b2 hεw.b2.e hεw.b2.d hεw.b2.p a1 dy2 lr
  · exact enet_exp_tiedAt xN wN bN gN vN epsStr lrStr cotN 56 56 w.b3 hεw.b3.e hεw.b3.d hεw.b3.p a2 dy3 lr
  · exact enet_strided_tiedAt xN wN bN gN vN epsStr lrStr cotN 28 28 w.b4 hεw.b4.e hεw.b4.d hεw.b4.p a3 dy4 lr
  · exact enet_exp_tiedAt xN wN bN gN vN epsStr lrStr cotN 28 28 w.b5 hεw.b5.e hεw.b5.d hεw.b5.p a4 dy5 lr
  · exact enet_strided_tiedAt xN wN bN gN vN epsStr lrStr cotN 14 14 w.b6 hεw.b6.e hεw.b6.d hεw.b6.p a5 dy6 lr
  · exact enet_exp_tiedAt xN wN bN gN vN epsStr lrStr cotN 14 14 w.b7 hεw.b7.e hεw.b7.d hεw.b7.p a6 dy7 lr
  · exact enet_exp_tiedAt xN wN bN gN vN epsStr lrStr cotN 14 14 w.b8 hεw.b8.e hεw.b8.d hεw.b8.p a7 dy8 lr
  · exact enet_exp_tiedAt xN wN bN gN vN epsStr lrStr cotN 14 14 w.b9 hεw.b9.e hεw.b9.d hεw.b9.p a8 dy9 lr
  · exact enet_exp_tiedAt xN wN bN gN vN epsStr lrStr cotN 14 14 w.b10 hεw.b10.e hεw.b10.d hεw.b10.p a9 dy10 lr
  · exact enet_exp_tiedAt xN wN bN gN vN epsStr lrStr cotN 14 14 w.b11 hεw.b11.e hεw.b11.d hεw.b11.p a10 dy11 lr
  · exact enet_strided_tiedAt xN wN bN gN vN epsStr lrStr cotN 7 7 w.b12 hεw.b12.e hεw.b12.d hεw.b12.p a11 dy12 lr
  · exact enet_exp_tiedAt xN wN bN gN vN epsStr lrStr cotN 7 7 w.b13 hεw.b13.e hεw.b13.d hεw.b13.p a12 dy13 lr
  · exact enet_exp_tiedAt xN wN bN gN vN epsStr lrStr cotN 7 7 w.b14 hεw.b14.e hεw.b14.d hεw.b14.p a13 dy14 lr
  · exact enet_exp_tiedAt xN wN bN gN vN epsStr lrStr cotN 7 7 w.b15 hεw.b15.e hεw.b15.d hεw.b15.p a14 dy15 lr
  · exact enet_exp_tiedAt xN wN bN gN vN epsStr lrStr cotN 7 7 w.b16 hεw.b16.e hεw.b16.d hεw.b16.p a15 dy16 lr
  · exact enet_head_tied xN wN bN gN vN epsStr lrStr cotN dN w.hε hεw.h w.hW w.hb w.hγ w.hβ w.fcW w.fcb a16 onehot lr

end Proofs.EnetTiePoC
