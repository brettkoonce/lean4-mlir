import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtChainClose
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFold
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2Fold
import LeanMlir.Proofs.Nets.ResNet.ResNet34Fold
import LeanMlir.Proofs.Nets.Small.MlpTrainStep
import LeanMlir.Proofs.Nets.ViT.ViTFold
import LeanMlir.Proofs.Architectures.ChannelLNBack

/-! # PoC: the FULL [3,3,9,3] ConvNeXt-T §1a TIE — the whole net tied through the real forward

The Chapter-7 §1a tie: mnv2's whole-net thread (now `MobileNetV2TieB.mnv2_net_tiedB`; the
per-example original was deleted 2026-09-08) for the
ConvNeXt-T schedule. The §1 fold (`ConvNeXtFold` + `ConvNeXtClose`/M2/M3/ViT) already makes
every rendered param op `den = certified ∀ cotangent`; this file feeds each consumer the **real
forward activations** of the `convNextTrainStepFaithfulV` render and the **loss-driven
backward-chain cotangent** that net delivers — so the whole 18-block train step is den-composed
forward → loss → backward, no free activations, no symbolic cotangent.

## The net this is about

`verified_mlir/convnext_train_step.mlir`, measured, not assumed:

| convention | this file | read from |
|---|---|---|
| depth / widths | `[3,3,9,3]`, 96 → 192 → 384 → 768 | `ConvNeXtRender.cnxTiny` |
| stem | 4×4/s4 patchify conv **then channel-LN** | `convNextFwdChain` |
| normalisation | `chanLNTensor3` (per-channel `[c]` affine, `h·w` statistics per example) at all 22 spatial sites: 1 stem + 18 block + 3 downsample | `ConvNeXtRender.lnFwdSite` |
| head | GAP → **vector-LN at one row** (`rowLNVecFlat 1 768`) → dense | `headLnFwdSite`, restored 2026-08-30 |
| activation | GELU (smooth — no kink mask anywhere) | `fwdBlock` |
| layer scale | per-channel `Vec c`, broadcast by `chanIdx` | `layerScaleChF` |
| padding | symmetric; ConvNeXt is a PyTorch-origin net and has no XLA-`SAME` site | — |
| params | 182 | `allParams`, and the artifact's 184 func args (`%x` + 182 + `%onehot`) |

> **Superseded scope note.** Until 2026-09-05 this file tied the **scalar-LN** ConvNeXt-T — the
> retired whole-map `bnForward` spelling, with no stem LN and a scalar `Vec 1` head LN. §2m
> flipped the renderer to the real per-channel `channel_layer_norm` and added the stem LN, §2n
> deleted the flag that had selected the old spelling, and 2026-08-30 restored the head LN. Every
> theorem in the old file was true and none of them was about the committed bytes. These are: the
> 22 spatial LN sites are `chanLNTensor3` with `Vec c` γ/β, the head is ViT's vector-LN at
> `N = 1`, and the stem LN is here.

## What is new against the scalar-LN version

* **channel-LN γ/β** (`Vec c`) at every spatial site, through `CnxPoC.chanLn{Gamma,Beta}Sgd_den`
  — the render re-emits the `[h·w, c]` transposes and runs ViT's `veclnGammaSgd` /
  `rowDenseBiasSgd` on that view, so the op operands here are `chanLNRows` of the saved LN input
  and of the chain cotangent, while the certified Jacobian is `chanLNTensor3`'s in the `c·h·w`
  activation layout (`ConvNeXtChannelLN`'s permutation argument bridges the two).
* **the channel-LN input-VJP** in the cotangent chain: `chanLNTensor3Back` where the scalar
  version had `bn_grad_input`. It is the certified VJP —
  `ConvNeXtBackCertifiedTie.chanLNTensor3Back_eq_chanLN_vjp`.
* **the stem LN**, which the scalar version did not have at all: `psng`/`psnbt` tie at the
  stem-LN output cotangent, and the stem conv's own gradients now see the LN input-VJP of it.
* **the head at the vector LN**: `hng`/`hnbt : Vec 768` through `ViTPoC.veclnGammaSgd_den` /
  `rowDenseBiasSgd_den_lnbeta` at `N = 1`. Stated at the literal 768 because `1 * m` does not
  reduce at a variable `m` — the render's own documented trap, in the proof this time.
* **the four even-kernel weight grads are no longer a gap.** The three downsample 2×2/s2 weights
  are `convStridedWeightSgd` (`ResNet34PoC.convStridedW_den` is kernel-generic), and the stem
  4×4/s4 weight is `convStride4WeightGrad`, whose `den` is
  `flatConvStride4_weight_grad_has_vjp`.

## Coverage / honest residual

All **182** parameters are tied. 181 of them at the full `θ − lr·(certified ∂Loss/∂θ)` step; the
stem weight `psW` at its **gradient**, because the render emits `convStride4WeightGrad` and wraps
it in the hand-written `sgd` text (a declared §5 carve-out — there is no fused
`convStride4WeightSgd` op to be the `den` of). What remains outside: the block backward is
rendered hand-written, so the cotangent SSA ↔ chain-cot correspondence is the per-op trust the
whole suite carries; plus per-op `pretty` lexing; LN `0 < ε` smoothness; ℝ → Float32 — the
boundary every prior fold carries.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CnxTiePoC

open scoped BigOperators

/-! ## ConvNeXt block — all 9 params tied (depthwise → channel-LN → expand → GELU → project → layer-scale → +skip)

Forward: `out = addV( layerScaleCh lg (conv₁ₓ₁ₚᵣ( gelu( conv₁ₓ₁ₑₓ( chanLN( dw₇ₓ₇(xin) ))))), xin )`.
Backward from the block-output cotangent `dyOut` (the residual `addV` is the outermost op and there
is no post-add activation, so it passes `dyOut` straight to the layer-scale output): layer-scale-back
(`cnxCotP`) → project-conv-back → GELU mask (`cnxCotE`) → expand-conv-back (`cnxCotN`) → the
channel-LN input-VJP (`chanLNTensor3Back`) → depthwise-back. Only that last-but-one step differs
from the scalar-LN thread; `cnxCotP`/`cnxCotE`/`cnxCotN` are LN-form-agnostic and are reused
verbatim from `ConvNeXtChainClose`. -/

/-- **ConvNeXt block, tied.** All 9 params (depthwise 7×7 `W`+`b`, channel-LN γ/β at `Vec c`,
    expand/project 1×1 conv `W`+`b`, per-channel layer-scale γ) denote the certified loss-descent
    step at the real block forward activations + the chain cotangents driven by `dyOut`. -/
def cnxBlockChTied {c cExp h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c)
    (xin d nl p : Vec (c*h*w)) (e g : Vec (cExp*h*w))
    (dyOut : Vec (c*h*w)) (lr : ℝ) : Prop :=
    let γlsB : Vec (c*h*w) := fun k => lg (chanIdx c h w k)
    let cotP : Vec (c*h*w) := cnxCotP γlsB dyOut
    let cotE : Vec (cExp*h*w) := cnxCotE γlsB Wpr bpr g e dyOut
    let cotN' : Vec (c*h*w) := cnxCotN γlsB Wex bex Wpr bpr nl g e dyOut
    let cotD : Vec (c*h*w) := chanLNTensor3Back c h w ε ng d cotN'
    -- depthwise 7×7 W/b  (cot = cotD)
    (∀ idx : Fin (c*7*7),
        den (SHlo.depthwiseWeightSgd xN wN lrStr bdw (Tensor3.unflatten xin) Wdw lr (.operand cotN cotD)) idx
          = Tensor3.flatten Wdw idx - lr * ∑ j : Fin (c*h*w),
              pdiv (fun v' : Vec (c*7*7) => Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') bdw (Tensor3.unflatten xin)))
                   (Tensor3.flatten Wdw) idx j * cotD j)
  ∧ (∀ o : Fin c,
        den (SHlo.depthwiseBiasSgd bN lrStr Wdw (Tensor3.unflatten xin) bdw lr (.operand cotN cotD)) o
          = bdw o - lr * ∑ j : Fin (c*h*w),
              pdiv (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d Wdw b' (Tensor3.unflatten xin))) bdw o j * cotD j)
    -- channel-LN γ/β  (cot = cotN', LN input = d; the op sees both as their [h·w, c] views)
  ∧ (∀ k : Fin c,
        den (SHlo.veclnGammaSgd (N := h*w) (D := c) gN xN epsStr lrStr ε
              (chanLNRows c h w d) ng lr (.operand cotN (chanLNRows c h w cotN'))) k
          = ng k - lr * ∑ j : Fin (c*h*w),
              pdiv (fun γ' : Vec c => chanLNTensor3 c h w ε γ' nbt d) ng k j * cotN' j)
  ∧ (∀ k : Fin c,
        den (SHlo.rowDenseBiasSgd (N := h*w) (c := c) bN lrStr nbt lr
              (.operand cotN (chanLNRows c h w cotN'))) k
          = nbt k - lr * ∑ j : Fin (c*h*w),
              pdiv (fun β' : Vec c => chanLNTensor3 c h w ε ng β' d) nbt k j * cotN' j)
    -- expand 1×1 conv (c → cExp) W/b  (cot = cotE, conv input = nl)
  ∧ (∀ idx : Fin (cExp*c*1*1),
        den (SHlo.convWeightSgd xN wN lrStr bex (Tensor3.unflatten nl) Wex lr (.operand cotN cotE)) idx
          = Kernel4.flatten Wex idx - lr * ∑ j : Fin (cExp*h*w),
              pdiv (fun v' : Vec (cExp*c*1*1) => Tensor3.flatten (conv2d (Kernel4.unflatten v') bex (Tensor3.unflatten nl)))
                   (Kernel4.flatten Wex) idx j * cotE j)
  ∧ (∀ o : Fin cExp,
        den (SHlo.convBiasSgd bN lrStr Wex (Tensor3.unflatten nl) bex lr (.operand cotN cotE)) o
          = bex o - lr * ∑ j : Fin (cExp*h*w),
              pdiv (fun b' : Vec cExp => Tensor3.flatten (conv2d Wex b' (Tensor3.unflatten nl))) bex o j * cotE j)
    -- project 1×1 conv (cExp → c) W/b  (cot = cotP, conv input = g)
  ∧ (∀ idx : Fin (c*cExp*1*1),
        den (SHlo.convWeightSgd xN wN lrStr bpr (Tensor3.unflatten g) Wpr lr (.operand cotN cotP)) idx
          = Kernel4.flatten Wpr idx - lr * ∑ j : Fin (c*h*w),
              pdiv (fun v' : Vec (c*cExp*1*1) => Tensor3.flatten (conv2d (Kernel4.unflatten v') bpr (Tensor3.unflatten g)))
                   (Kernel4.flatten Wpr) idx j * cotP j)
  ∧ (∀ o : Fin c,
        den (SHlo.convBiasSgd bN lrStr Wpr (Tensor3.unflatten g) bpr lr (.operand cotN cotP)) o
          = bpr o - lr * ∑ j : Fin (c*h*w),
              pdiv (fun b' : Vec c => Tensor3.flatten (conv2d Wpr b' (Tensor3.unflatten g))) bpr o j * cotP j)
    -- per-channel layer-scale γ  (cot = dyOut directly, layer input = p)
  ∧ (∀ cc : Fin c,
        den (SHlo.layerScaleChGammaSgd gN xN lrStr p lg lr (.operand cotN dyOut)) cc
          = lg cc - lr * ∑ j : Fin (c*h*w),
              pdiv (fun γ' : Vec c => layerScale (fun k => γ' (chanIdx c h w k)) p) lg cc j * dyOut j)

theorem cnx_block_ch_tied {c cExp h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c)
    (xin d nl p : Vec (c*h*w)) (e g : Vec (cExp*h*w))
    (dyOut : Vec (c*h*w)) (lr : ℝ) :
    cnxBlockChTied xN wN bN gN epsStr lrStr cotN ε Wdw bdw ng nbt Wex bex Wpr bpr lg
      xin d nl p e g dyOut lr := by
  unfold cnxBlockChTied
  intro γlsB cotP cotE cotN' cotD
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact Mnv2PoC.depthwiseW_den xN wN lrStr cotN bdw (Tensor3.unflatten xin) Wdw cotD lr idx
  · intro o;   exact Mnv2PoC.depthwiseB_den bN lrStr cotN Wdw (Tensor3.unflatten xin) bdw cotD lr o
  · intro k;   exact CnxPoC.chanLnGammaSgd_den gN xN epsStr lrStr cotN ε nbt d ng cotN' lr k
  · intro k;   exact CnxPoC.chanLnBetaSgd_den bN lrStr cotN ε ng d nbt cotN' lr k
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN bex (Tensor3.unflatten nl) Wex cotE lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN Wex (Tensor3.unflatten nl) bex cotE lr o
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN bpr (Tensor3.unflatten g) Wpr cotP lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN Wpr (Tensor3.unflatten g) bpr cotP lr o
  · intro cc;  exact CnxPoC.layerScaleChGammaSgd_den gN xN lrStr cotN p lg dyOut lr cc

/-! ## Downsample — channel-LN → 2×2/s2 conv (all 4 params tied)

Forward: `o = convˢ²(chanLN(xin))` (LN over the block-input grid `2h×2w` with a `Vec ci` affine,
then a 2×2/s2 conv `ci → co`). No skip. Backward from `dyOut`: strided-conv-back (`cotN'`) →
channel-LN-back. The strided **weight** is no longer a gap: `convStridedWeightSgd` is emitted at
2×2 since `sWGradGeom` split the odd/even padding cases, and `ResNet34PoC.convStridedW_den` is
kernel-generic. -/

/-- **Downsample, tied.** Channel-LN γ/β at the `ci·(2h)·(2w)` input grid, plus the strided conv's
    weight and bias, at the real forward + the chain cotangents. -/
def cnxDownChTied {ci co h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin n : Vec (ci*(2*h)*(2*w))) (dyOut : Vec (co*h*w)) (lr : ℝ) : Prop :=
    let cotN' : Vec (ci*(2*h)*(2*w)) := (flatConvStride2_has_vjp Wd bd).backward n dyOut
    (∀ k : Fin ci,
        den (SHlo.veclnGammaSgd (N := (2*h)*(2*w)) (D := ci) gN xN epsStr lrStr ε
              (chanLNRows ci (2*h) (2*w) xin) dng lr
              (.operand cotN (chanLNRows ci (2*h) (2*w) cotN'))) k
          = dng k - lr * ∑ j : Fin (ci*(2*h)*(2*w)),
              pdiv (fun γ' : Vec ci => chanLNTensor3 ci (2*h) (2*w) ε γ' dnbt xin) dng k j * cotN' j)
  ∧ (∀ k : Fin ci,
        den (SHlo.rowDenseBiasSgd (N := (2*h)*(2*w)) (c := ci) bN lrStr dnbt lr
              (.operand cotN (chanLNRows ci (2*h) (2*w) cotN'))) k
          = dnbt k - lr * ∑ j : Fin (ci*(2*h)*(2*w)),
              pdiv (fun β' : Vec ci => chanLNTensor3 ci (2*h) (2*w) ε dng β' xin) dnbt k j * cotN' j)
  ∧ (∀ idx : Fin (co*ci*2*2),
        den (SHlo.convStridedWeightSgd xN wN lrStr bd n Wd lr (.operand cotN dyOut)) idx
          = Kernel4.flatten Wd idx - lr * ∑ j : Fin (co*h*w),
              pdiv (fun v' : Vec (co*ci*2*2) => flatConvStride2 (Kernel4.unflatten v') bd n)
                   (Kernel4.flatten Wd) idx j * dyOut j)
  ∧ (∀ o : Fin co,
        den (SHlo.convStridedBiasSgd bN lrStr Wd n bd lr (.operand cotN dyOut)) o
          = bd o - lr * ∑ j : Fin (co*h*w),
              pdiv (fun b' : Vec co => flatConvStride2 Wd b' n) bd o j * dyOut j)

theorem cnx_down_ch_tied {ci co h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin n : Vec (ci*(2*h)*(2*w))) (dyOut : Vec (co*h*w)) (lr : ℝ) :
    cnxDownChTied xN wN bN gN epsStr lrStr cotN ε dng dnbt Wd bd xin n dyOut lr := by
  unfold cnxDownChTied
  intro cotN'
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro k; exact CnxPoC.chanLnGammaSgd_den gN xN epsStr lrStr cotN ε dnbt xin dng cotN' lr k
  · intro k; exact CnxPoC.chanLnBetaSgd_den bN lrStr cotN ε dng xin dnbt cotN' lr k
  · intro idx; exact ResNet34PoC.convStridedW_den xN wN lrStr cotN bd n Wd dyOut lr idx
  · intro o; exact ResNet34PoC.convStridedB_den bN lrStr cotN Wd n bd dyOut lr o

/-! ## Stem — 4×4/s4 patchify conv → channel-LN (all 4 params)

`psng`/`psnbt` tie at `dyStem`, the cotangent block 1 delivers at the stem-LN output; the conv's
own two parameters then see `cotPatch`, the LN input-VJP of it. The bias grad is a pure cotangent
reduce, so the render emits it as a stride-1 `convBiasSgd` at the OUTPUT resolution and the
carried `W`/`x` are generic — the same modelling the mnv2/r34 stems use. The weight is the
`convStride4WeightGrad` op, and it ties at the **gradient**: its SGD wrap is hand-written text. -/

/-- **Stem, tied.** Channel-LN γ/β, the conv bias, and the conv weight's gradient. -/
def cnxStemChTied {c h w : Nat}
    (xN _wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c)
    (x : Vec (3*(2*(2*h))*(2*(2*w)))) (xstem : Tensor3 3 h w) (patch : Vec (c*h*w))
    (dyStem : Vec (c*h*w)) (lr : ℝ) : Prop :=
    let cotPatch : Vec (c*h*w) := chanLNTensor3Back c h w ε psng patch dyStem
    (∀ k : Fin c,
        den (SHlo.veclnGammaSgd (N := h*w) (D := c) gN xN epsStr lrStr ε
              (chanLNRows c h w patch) psng lr (.operand cotN (chanLNRows c h w dyStem))) k
          = psng k - lr * ∑ j : Fin (c*h*w),
              pdiv (fun γ' : Vec c => chanLNTensor3 c h w ε γ' psnbt patch) psng k j * dyStem j)
  ∧ (∀ k : Fin c,
        den (SHlo.rowDenseBiasSgd (N := h*w) (c := c) bN lrStr psnbt lr
              (.operand cotN (chanLNRows c h w dyStem))) k
          = psnbt k - lr * ∑ j : Fin (c*h*w),
              pdiv (fun β' : Vec c => chanLNTensor3 c h w ε psng β' patch) psnbt k j * dyStem j)
  ∧ (∀ o : Fin c,
        den (SHlo.convBiasSgd bN lrStr Wst xstem psb lr (.operand cotN cotPatch)) o
          = psb o - lr * ∑ j : Fin (c*h*w),
              pdiv (fun b' : Vec c => Tensor3.flatten (conv2d Wst b' xstem)) psb o j * cotPatch j)
  ∧ (∀ idx : Fin (c*3*4*4),
        den (SHlo.convStride4WeightGrad xN psb x Wst (.operand cotN cotPatch)) idx
          = ∑ j : Fin (c*h*w),
              pdiv (fun v' : Vec (c*3*4*4) => flatConvStride4 (Kernel4.unflatten v') psb x)
                   (Kernel4.flatten Wst) idx j * cotPatch j)

theorem cnx_stem_ch_tied {c h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c)
    (x : Vec (3*(2*(2*h))*(2*(2*w)))) (xstem : Tensor3 3 h w) (patch : Vec (c*h*w))
    (dyStem : Vec (c*h*w)) (lr : ℝ) :
    cnxStemChTied xN wN bN gN epsStr lrStr cotN ε Wst psb psng psnbt x xstem patch dyStem lr := by
  unfold cnxStemChTied
  intro cotPatch
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro k; exact CnxPoC.chanLnGammaSgd_den gN xN epsStr lrStr cotN ε psnbt patch psng dyStem lr k
  · intro k; exact CnxPoC.chanLnBetaSgd_den bN lrStr cotN ε psng patch psnbt dyStem lr k
  · intro o; exact CifarPoC.convB_den bN lrStr cotN Wst xstem psb cotPatch lr o
  · intro idx; exact flatConvStride4_weight_grad_has_vjp_correct psb x (Kernel4.flatten Wst) cotPatch idx

/-! ## Head — GAP → vector-LN at one row → dense

After GAP the tensor is a single `[768]` row, so "normalise each spatial row over its channels"
and "normalise the feature vector" are the same function at `m = 1`: the head is ViT's per-token
LN with one token, and `headLnFwdSite` is `lnFwdSite` with the two transposes deleted. Stated at
the LITERAL 768 — `1 * m` does not reduce at a variable `m` (`Nat.mul` recurses on its second
argument), which is the annotation trap the render carries in its own comment. -/

/-- **Head LN + dense bias, tied.** The head-LN γ/β (at the pooled row `gap`, cot = the dense-back
    `cotHn`) and the dense bias (cot = the loss `g`) at the real forward. -/
def cnxHeadChTied (gN xN bN bdN epsStr lrStr cotN dyN : String) (ε : ℝ)
    (hng hnbt : Vec 768) (Wfc : Mat 768 10) (bfc : Vec 10)
    (gap : Vec (1*768)) (hn : Vec 768) (g : Vec 10) (lr : ℝ) : Prop :=
    let cotHn : Vec (1*768) := (dense_has_vjp Wfc bfc).backward hn g
    (∀ k : Fin 768,
        den (SHlo.veclnGammaSgd (N := 1) (D := 768) gN xN epsStr lrStr ε gap hng lr
              (.operand cotN cotHn)) k
          = hng k - lr * ∑ o : Fin (1*768),
              pdiv (fun γ' : Vec 768 =>
                      Mat.flatten (fun r => layerNormVec 768 ε γ' hnbt (Mat.unflatten gap r)))
                   hng k o * cotHn o)
  ∧ (∀ k : Fin 768,
        den (SHlo.rowDenseBiasSgd (N := 1) (c := 768) bN lrStr hnbt lr (.operand cotN cotHn)) k
          = hnbt k - lr * ∑ o : Fin (1*768),
              pdiv (fun β' : Vec 768 =>
                      Mat.flatten (fun r => layerNormVec 768 ε hng β' (Mat.unflatten gap r)))
                   hnbt k o * cotHn o)
  ∧ (∀ i : Fin 10,
        den (SHlo.biasSgd bdN lrStr bfc lr (.operand dyN g)) i
          = bfc i - lr * ∑ j : Fin 10, pdiv (fun b' : Vec 10 => dense Wfc b' hn) bfc i j * g j)

theorem cnx_head_ch_tied (gN xN bN bdN epsStr lrStr cotN dyN : String) (ε : ℝ)
    (hng hnbt : Vec 768) (Wfc : Mat 768 10) (bfc : Vec 10)
    (gap : Vec (1*768)) (hn : Vec 768) (g : Vec 10) (lr : ℝ) :
    cnxHeadChTied gN xN bN bdN epsStr lrStr cotN dyN ε hng hnbt Wfc bfc gap hn g lr := by
  unfold cnxHeadChTied
  intro cotHn
  refine ⟨?_, ?_, ?_⟩
  · intro k; exact ViTPoC.veclnGammaSgd_den gN xN epsStr lrStr cotN ε hnbt gap hng cotHn lr k
  · intro k;
    exact ViTPoC.rowDenseBiasSgd_den_lnbeta bN lrStr cotN ε hng (Mat.unflatten gap) hnbt cotHn lr k
  · intro i; exact Cifar8PoC.denseB_den bdN lrStr dyN Wfc hn bfc g lr i

/-- **Dense head weight `Wd`, tied to the WHOLE softmax-CE loss** — `Wd − lr·∂(CE ∘ dense)/∂Wd`. -/
theorem cnx_dense_tied_totalloss {m : Nat} (aN wN lrStr dyN : String)
    (Wd : Mat m 10) (bd : Vec 10) (a : Vec m) (label : Fin 10)
    (lr : ℝ) (i : Fin m) (j : Fin 10) :
    den (SHlo.weightSgd aN wN lrStr a Wd lr
          (.operand dyN (fun k => softmax 10 (mnistLinear Wd bd a) k - oneHot 10 label k)))
        (finProdFinEquiv (i, j))
      = Wd i j - lr * pdiv (fun v : Vec (m * 10) => fun _ : Fin 1 =>
            crossEntropy 10 (dense (Mat.unflatten v) bd a) label)
          (Mat.flatten Wd) (finProdFinEquiv (i, j)) 0 := by
  rw [Cifar8PoC.denseW_den aN wN lrStr dyN a Wd bd
        (fun k => softmax 10 (mnistLinear Wd bd a) k - oneHot 10 label k) lr i j,
      mlp_output_total_loss_grad Wd bd a label i j]

/-- **The emitted loss-cotangent graph denotes the softmax-CE gradient at the logits.** -/
theorem cnxLossCot_den (nlogN ohN : String) (logits : Vec 10) (label : Fin 10) :
    den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe (.operand nlogN logits)))
          (.operand ohN (oneHot 10 label)))
      = fun j => softmax 10 logits j - oneHot 10 label j := by
  funext j; simp only [den, softmax]

/-! ## Forward aliases (`@[irreducible]`) — thread block inputs through the real forward

`@[irreducible]` so the 18-deep nested composition stays opaque during the capstone's dimension
inference (the r34/mnv2 heartbeat lesson). -/

/-- The stem's patchify conv output — the stem LN's input, and the activation `psW`/`psb` see. -/
@[irreducible] noncomputable def cnxStemPatchO {c h w : Nat}
    (Wst : Kernel4 c 3 4 4) (bst : Vec c) (x : Vec (3*(2*(2*h))*(2*(2*w)))) : Vec (c*h*w) :=
  flatConvStride4 Wst bst x

/-- The stem output: patchify conv **then** channel-LN (§2m — the pre-§2m render had no stem LN). -/
@[irreducible] noncomputable def cnxStemFwdO {c h w : Nat} (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (bst psng psnbt : Vec c)
    (x : Vec (3*(2*(2*h))*(2*(2*w)))) : Vec (c*h*w) :=
  chanLNTensor3 c h w ε psng psnbt (flatConvStride4 Wst bst x)

@[irreducible] noncomputable def cnxBlockBodyChO {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin : Vec (c*h*w)) : Vec (c*h*w) :=
  layerScale (fun k => lg (chanIdx c h w k))
    (flatConv (h := h) (w := w) Wpr bpr (gelu (cExp*h*w) (flatConv (h := h) (w := w) Wex bex
      (chanLNTensor3 c h w ε ng nbt (depthwiseFlat (h := h) (w := w) Wdw bdw xin)))))

@[irreducible] noncomputable def cnxBlockFwdChO {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin : Vec (c*h*w)) : Vec (c*h*w) :=
  fun i => cnxBlockBodyChO ε Wdw bdw ng nbt Wex bex Wpr bpr lg xin i + xin i

@[irreducible] noncomputable def cnxDownFwdChO {ci co h w : Nat} (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (ci*(2*h)*(2*w))) : Vec (co*h*w) :=
  flatConvStride2 Wd bd (chanLNTensor3 ci (2*h) (2*w) ε dng dnbt xin)

/-! ## Backward cot-in constructors (`@[irreducible]`) — thread block dyOuts (the residual fan-in) -/

/-- ConvNeXt block input cotangent: `depthwise-back(cotD) + dyOut` (the identity-skip fan-in),
    with `cotD` through the channel-LN input-VJP. -/
@[irreducible] noncomputable def cnxBlockCotInChAt {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin dyOut : Vec (c*h*w)) : Vec (c*h*w) :=
  let γlsB : Vec (c*h*w) := fun k => lg (chanIdx c h w k)
  let d := depthwiseFlat (h := h) (w := w) Wdw bdw xin
  let nl := chanLNTensor3 c h w ε ng nbt d
  let e := flatConv (h := h) (w := w) Wex bex nl
  let g := gelu (cExp*h*w) e
  let cotD := chanLNTensor3Back c h w ε ng d (cnxCotN γlsB Wex bex Wpr bpr nl g e dyOut)
  fun i => (depthwiseFlat_has_vjp (h := h) (w := w) Wdw bdw).backward xin cotD i + dyOut i

/-- Downsample input cotangent (at `ci·(2h)·(2w)`): the channel-LN input-VJP of the
    strided-conv-back. No skip. -/
@[irreducible] noncomputable def cnxDownCotInChAt {ci co h w : Nat} (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (ci*(2*h)*(2*w))) (dyOut : Vec (co*h*w)) : Vec (ci*(2*h)*(2*w)) :=
  let n := chanLNTensor3 ci (2*h) (2*w) ε dng dnbt xin
  let cotN := (flatConvStride2_has_vjp Wd bd).backward n dyOut
  chanLNTensor3Back ci (2*h) (2*w) ε dng xin cotN

/-- The cotangent at the last block output `xhead` (= s3b2's `dyOut`): `gap-back(headLN-back(
    dense-back(g)))`. The head LN's input-VJP is the render's `rowScaleF γ` then `lnRowBack` at
    γ = 1, which is `rowLNVecFlatBack` (`ConvNeXtBackB0.rowLNBack_affine_eq`). -/
@[irreducible] noncomputable def cnxHeadDyXheadCh {h w : Nat} (ε : ℝ)
    (hng hnbt : Vec 768) (Wfc : Mat 768 10) (bfc : Vec 10)
    (xhead : Vec (768*h*w)) (g : Vec 10) : Vec (768*h*w) :=
  let gap : Vec (1*768) := globalAvgPoolFlat 768 h w xhead
  let hn : Vec 768 := rowLNVecFlat 1 768 ε hng hnbt gap
  let cotHn : Vec (1*768) := (dense_has_vjp Wfc bfc).backward hn g
  let cotGap : Vec 768 := rowLNVecFlatBack 1 768 ε hng gap cotHn
  (globalAvgPoolFlat_has_vjp 768 h w).backward xhead cotGap

/-! ## Input-only `TiedAt` wrappers (`@[irreducible]`) — compute internals from a block's input -/

@[irreducible] def cnxBlockChTiedAt {c cExp h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin : Vec (c*h*w)) (dyOut : Vec (c*h*w)) (lr : ℝ) : Prop :=
  let d := depthwiseFlat (h := h) (w := w) Wdw bdw xin
  let nl := chanLNTensor3 c h w ε ng nbt d
  let e := flatConv (h := h) (w := w) Wex bex nl
  let g := gelu (cExp*h*w) e
  let p := flatConv (h := h) (w := w) Wpr bpr g
  cnxBlockChTied xN wN bN gN epsStr lrStr cotN ε Wdw bdw ng nbt Wex bex Wpr bpr lg
    xin d nl p e g dyOut lr

theorem cnx_block_ch_tiedAt {c cExp h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin : Vec (c*h*w)) (dyOut : Vec (c*h*w)) (lr : ℝ) :
    cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε Wdw bdw ng nbt Wex bex Wpr bpr lg xin dyOut lr := by
  unfold cnxBlockChTiedAt
  intro d nl e g p
  exact cnx_block_ch_tied xN wN bN gN epsStr lrStr cotN ε Wdw bdw ng nbt Wex bex Wpr bpr lg
    xin d nl p e g dyOut lr

@[irreducible] def cnxDownChTiedAt {ci co h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (ci*(2*h)*(2*w))) (dyOut : Vec (co*h*w)) (lr : ℝ) : Prop :=
  let n := chanLNTensor3 ci (2*h) (2*w) ε dng dnbt xin
  cnxDownChTied xN wN bN gN epsStr lrStr cotN ε dng dnbt Wd bd xin n dyOut lr

theorem cnx_down_ch_tiedAt {ci co h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (ci*(2*h)*(2*w))) (dyOut : Vec (co*h*w)) (lr : ℝ) :
    cnxDownChTiedAt xN wN bN gN epsStr lrStr cotN ε dng dnbt Wd bd xin dyOut lr := by
  unfold cnxDownChTiedAt
  intro n
  exact cnx_down_ch_tied xN wN bN gN epsStr lrStr cotN ε dng dnbt Wd bd xin n dyOut lr

@[irreducible] def cnxStemChTiedAt {c h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c)
    (x : Vec (3*(2*(2*h))*(2*(2*w)))) (xstem : Tensor3 3 h w)
    (dyStem : Vec (c*h*w)) (lr : ℝ) : Prop :=
  let patch := flatConvStride4 Wst psb x
  cnxStemChTied xN wN bN gN epsStr lrStr cotN ε Wst psb psng psnbt x xstem patch dyStem lr

theorem cnx_stem_ch_tiedAt {c h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c)
    (x : Vec (3*(2*(2*h))*(2*(2*w)))) (xstem : Tensor3 3 h w)
    (dyStem : Vec (c*h*w)) (lr : ℝ) :
    cnxStemChTiedAt xN wN bN gN epsStr lrStr cotN ε Wst psb psng psnbt x xstem dyStem lr := by
  unfold cnxStemChTiedAt
  intro patch
  exact cnx_stem_ch_tied xN wN bN gN epsStr lrStr cotN ε Wst psb psng psnbt x xstem patch dyStem lr

@[irreducible] def cnxHeadChTiedAt {h w : Nat}
    (gN xN bN bdN epsStr lrStr cotN dyN : String) (ε : ℝ)
    (hng hnbt : Vec 768) (Wfc : Mat 768 10) (bfc : Vec 10)
    (xhead : Vec (768*h*w)) (g : Vec 10) (lr : ℝ) : Prop :=
  let gap : Vec (1*768) := globalAvgPoolFlat 768 h w xhead
  let hn : Vec 768 := rowLNVecFlat 1 768 ε hng hnbt gap
  cnxHeadChTied gN xN bN bdN epsStr lrStr cotN dyN ε hng hnbt Wfc bfc gap hn g lr

theorem cnx_head_ch_tiedAt {h w : Nat}
    (gN xN bN bdN epsStr lrStr cotN dyN : String) (ε : ℝ)
    (hng hnbt : Vec 768) (Wfc : Mat 768 10) (bfc : Vec 10)
    (xhead : Vec (768*h*w)) (g : Vec 10) (lr : ℝ) :
    cnxHeadChTiedAt gN xN bN bdN epsStr lrStr cotN dyN ε hng hnbt Wfc bfc xhead g lr := by
  unfold cnxHeadChTiedAt
  intro gap hn
  exact cnx_head_ch_tied gN xN bN bdN epsStr lrStr cotN dyN ε hng hnbt Wfc bfc gap hn g lr

/-! ## The whole-net capstone — all 182 params through the REAL forward + composed cotangent

The `convNextTrainStepFaithfulV` forward threaded: block inputs are the forward prefixes
(`cnxStemFwdO` / `cnxBlockFwdChO` / `cnxDownFwdChO`), and the backward cotangents are composed
from the loss `g = softmax(logits) − onehot` down through dense (`dense_has_vjp`) + the head LN +
GAP (`globalAvgPoolFlat_has_vjp`) + every block's backward, with the residual fan-in `+ dyOut` at
each of the eighteen identity-skip merges, the channel-LN-back at each of the three downsamples,
and the stem LN's own back before the patchify conv's gradients. Each stem / block / down / head
tie then holds at its real input + threaded cotangent. The full §1a tie: the whole [3,3,9,3]
182-parameter ConvNeXt-T train step is den-composed forward → loss → backward, no free
activations, no symbolic cotangent. -/

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 400000 in
/-- **The whole [3,3,9,3] ConvNeXt-T train step, tied.** Threading the real (channel-LN,
    per-channel layer-scale) forward and the loss-driven backward cotangent chain (GELU masks, the
    residual fan-in at every identity skip, the channel-LN-back at every downsample and at the
    stem), the 18 ConvNeXt blocks, the 3 downsamples, the 4×4/s4 stem with its LN, the
    GAP → LN → dense head, and the dense total-loss fold + loss-cotangent graph all denote the
    certified loss-descent step. All 182 parameters; `psW` at its gradient (§5 carve-out). -/
theorem cnx_net_tied_certified
    (xN wN bN gN epsStr lrStr cotN dN nlogN ohN : String) (ε : ℝ)
    -- stem (c=96): patchify conv + its channel-LN
    (Wst : Kernel4 96 3 4 4) (psb psng psnbt : Vec 96) (xstem : Tensor3 3 56 56)
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
    (hG hT : Vec 768) (Wfc : Mat 768 10) (bfc : Vec 10)
    (x : Vec (3*224*224)) (label : Fin 10) (lr : ℝ) :
    -- forward block inputs (the prefixes of the committed render's forward)
    let ib1   : Vec (96*56*56)         := cnxStemFwdO (h := 56) (w := 56) ε Wst psb psng psnbt x
    let ib2   : Vec (96*56*56)         := cnxBlockFwdChO ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 ib1
    let ib3   : Vec (96*56*56)         := cnxBlockFwdChO ε aW2 aB2 nG2 nB2 eW2 eB2 pW2 pB2 sL2 ib2
    let ibD0  : Vec (96*56*56)         := cnxBlockFwdChO ε aW3 aB3 nG3 nB3 eW3 eB3 pW3 pB3 sL3 ib3
    let ib4   : Vec (192*28*28)        := cnxDownFwdChO (h := 28) (w := 28) ε dG0 dT0 dW0 dB0 ibD0
    let ib5   : Vec (192*28*28)        := cnxBlockFwdChO ε aW4 aB4 nG4 nB4 eW4 eB4 pW4 pB4 sL4 ib4
    let ib6   : Vec (192*28*28)        := cnxBlockFwdChO ε aW5 aB5 nG5 nB5 eW5 eB5 pW5 pB5 sL5 ib5
    let ibD1  : Vec (192*28*28)        := cnxBlockFwdChO ε aW6 aB6 nG6 nB6 eW6 eB6 pW6 pB6 sL6 ib6
    let ib7   : Vec (384*14*14)        := cnxDownFwdChO (h := 14) (w := 14) ε dG1 dT1 dW1 dB1 ibD1
    let ib8   : Vec (384*14*14)        := cnxBlockFwdChO ε aW7 aB7 nG7 nB7 eW7 eB7 pW7 pB7 sL7 ib7
    let ib9   : Vec (384*14*14)        := cnxBlockFwdChO ε aW8 aB8 nG8 nB8 eW8 eB8 pW8 pB8 sL8 ib8
    let ib10  : Vec (384*14*14)        := cnxBlockFwdChO ε aW9 aB9 nG9 nB9 eW9 eB9 pW9 pB9 sL9 ib9
    let ib11  : Vec (384*14*14)        := cnxBlockFwdChO ε aW10 aB10 nG10 nB10 eW10 eB10 pW10 pB10 sL10 ib10
    let ib12  : Vec (384*14*14)        := cnxBlockFwdChO ε aW11 aB11 nG11 nB11 eW11 eB11 pW11 pB11 sL11 ib11
    let ib13  : Vec (384*14*14)        := cnxBlockFwdChO ε aW12 aB12 nG12 nB12 eW12 eB12 pW12 pB12 sL12 ib12
    let ib14  : Vec (384*14*14)        := cnxBlockFwdChO ε aW13 aB13 nG13 nB13 eW13 eB13 pW13 pB13 sL13 ib13
    let ib15  : Vec (384*14*14)        := cnxBlockFwdChO ε aW14 aB14 nG14 nB14 eW14 eB14 pW14 pB14 sL14 ib14
    let ibD2  : Vec (384*14*14)        := cnxBlockFwdChO ε aW15 aB15 nG15 nB15 eW15 eB15 pW15 pB15 sL15 ib15
    let ib16  : Vec (768*7*7)          := cnxDownFwdChO (h := 7) (w := 7) ε dG2 dT2 dW2 dB2 ibD2
    let ib17  : Vec (768*7*7)          := cnxBlockFwdChO ε aW16 aB16 nG16 nB16 eW16 eB16 pW16 pB16 sL16 ib16
    let ib18  : Vec (768*7*7)          := cnxBlockFwdChO ε aW17 aB17 nG17 nB17 eW17 eB17 pW17 pB17 sL17 ib17
    let xhead : Vec (768*7*7)          := cnxBlockFwdChO ε aW18 aB18 nG18 nB18 eW18 eB18 pW18 pB18 sL18 ib18
    -- head forward + the loss cotangent
    let gap : Vec (1*768) := globalAvgPoolFlat 768 7 7 xhead
    let hn  : Vec 768     := rowLNVecFlat 1 768 ε hG hT gap
    let g   : Vec 10      := fun k => softmax 10 (mnistLinear Wfc bfc hn) k - oneHot 10 label k
    -- backward cotangents (composed from the loss; residual fan-in at each skip, LN-back at each
    -- downsample and at the stem)
    let dyO18  : Vec (768*7*7)         := cnxHeadDyXheadCh (h := 7) (w := 7) ε hG hT Wfc bfc xhead g
    let dyO17  : Vec (768*7*7)         := cnxBlockCotInChAt ε aW18 aB18 nG18 nB18 eW18 eB18 pW18 pB18 sL18 ib18 dyO18
    let dyO16  : Vec (768*7*7)         := cnxBlockCotInChAt ε aW17 aB17 nG17 nB17 eW17 eB17 pW17 pB17 sL17 ib17 dyO17
    let dyD2   : Vec (768*7*7)         := cnxBlockCotInChAt ε aW16 aB16 nG16 nB16 eW16 eB16 pW16 pB16 sL16 ib16 dyO16
    let dyO15  : Vec (384*14*14)       := cnxDownCotInChAt (h := 7) (w := 7) ε dG2 dT2 dW2 dB2 ibD2 dyD2
    let dyO14  : Vec (384*14*14)       := cnxBlockCotInChAt ε aW15 aB15 nG15 nB15 eW15 eB15 pW15 pB15 sL15 ib15 dyO15
    let dyO13  : Vec (384*14*14)       := cnxBlockCotInChAt ε aW14 aB14 nG14 nB14 eW14 eB14 pW14 pB14 sL14 ib14 dyO14
    let dyO12  : Vec (384*14*14)       := cnxBlockCotInChAt ε aW13 aB13 nG13 nB13 eW13 eB13 pW13 pB13 sL13 ib13 dyO13
    let dyO11  : Vec (384*14*14)       := cnxBlockCotInChAt ε aW12 aB12 nG12 nB12 eW12 eB12 pW12 pB12 sL12 ib12 dyO12
    let dyO10  : Vec (384*14*14)       := cnxBlockCotInChAt ε aW11 aB11 nG11 nB11 eW11 eB11 pW11 pB11 sL11 ib11 dyO11
    let dyO9   : Vec (384*14*14)       := cnxBlockCotInChAt ε aW10 aB10 nG10 nB10 eW10 eB10 pW10 pB10 sL10 ib10 dyO10
    let dyO8   : Vec (384*14*14)       := cnxBlockCotInChAt ε aW9 aB9 nG9 nB9 eW9 eB9 pW9 pB9 sL9 ib9 dyO9
    let dyO7   : Vec (384*14*14)       := cnxBlockCotInChAt ε aW8 aB8 nG8 nB8 eW8 eB8 pW8 pB8 sL8 ib8 dyO8
    let dyD1   : Vec (384*14*14)       := cnxBlockCotInChAt ε aW7 aB7 nG7 nB7 eW7 eB7 pW7 pB7 sL7 ib7 dyO7
    let dyO6   : Vec (192*28*28)       := cnxDownCotInChAt (h := 14) (w := 14) ε dG1 dT1 dW1 dB1 ibD1 dyD1
    let dyO5   : Vec (192*28*28)       := cnxBlockCotInChAt ε aW6 aB6 nG6 nB6 eW6 eB6 pW6 pB6 sL6 ib6 dyO6
    let dyO4   : Vec (192*28*28)       := cnxBlockCotInChAt ε aW5 aB5 nG5 nB5 eW5 eB5 pW5 pB5 sL5 ib5 dyO5
    let dyD0   : Vec (192*28*28)       := cnxBlockCotInChAt ε aW4 aB4 nG4 nB4 eW4 eB4 pW4 pB4 sL4 ib4 dyO4
    let dyO3   : Vec (96*56*56)        := cnxDownCotInChAt (h := 28) (w := 28) ε dG0 dT0 dW0 dB0 ibD0 dyD0
    let dyO2   : Vec (96*56*56)        := cnxBlockCotInChAt ε aW3 aB3 nG3 nB3 eW3 eB3 pW3 pB3 sL3 ib3 dyO3
    let dyO1   : Vec (96*56*56)        := cnxBlockCotInChAt ε aW2 aB2 nG2 nB2 eW2 eB2 pW2 pB2 sL2 ib2 dyO2
    let dyStem : Vec (96*56*56)        := cnxBlockCotInChAt ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 ib1 dyO1
    -- the stem, every block, every downsample, the head, the dense total-loss fold + loss cot
    cnxStemChTiedAt xN wN bN gN epsStr lrStr cotN ε Wst psb psng psnbt x xstem dyStem lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 ib1 dyO1 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW2 aB2 nG2 nB2 eW2 eB2 pW2 pB2 sL2 ib2 dyO2 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW3 aB3 nG3 nB3 eW3 eB3 pW3 pB3 sL3 ib3 dyO3 lr
  ∧ cnxDownChTiedAt xN wN bN gN epsStr lrStr cotN ε dG0 dT0 dW0 dB0 ibD0 dyD0 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW4 aB4 nG4 nB4 eW4 eB4 pW4 pB4 sL4 ib4 dyO4 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW5 aB5 nG5 nB5 eW5 eB5 pW5 pB5 sL5 ib5 dyO5 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW6 aB6 nG6 nB6 eW6 eB6 pW6 pB6 sL6 ib6 dyO6 lr
  ∧ cnxDownChTiedAt xN wN bN gN epsStr lrStr cotN ε dG1 dT1 dW1 dB1 ibD1 dyD1 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW7 aB7 nG7 nB7 eW7 eB7 pW7 pB7 sL7 ib7 dyO7 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW8 aB8 nG8 nB8 eW8 eB8 pW8 pB8 sL8 ib8 dyO8 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW9 aB9 nG9 nB9 eW9 eB9 pW9 pB9 sL9 ib9 dyO9 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW10 aB10 nG10 nB10 eW10 eB10 pW10 pB10 sL10 ib10 dyO10 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW11 aB11 nG11 nB11 eW11 eB11 pW11 pB11 sL11 ib11 dyO11 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW12 aB12 nG12 nB12 eW12 eB12 pW12 pB12 sL12 ib12 dyO12 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW13 aB13 nG13 nB13 eW13 eB13 pW13 pB13 sL13 ib13 dyO13 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW14 aB14 nG14 nB14 eW14 eB14 pW14 pB14 sL14 ib14 dyO14 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW15 aB15 nG15 nB15 eW15 eB15 pW15 pB15 sL15 ib15 dyO15 lr
  ∧ cnxDownChTiedAt xN wN bN gN epsStr lrStr cotN ε dG2 dT2 dW2 dB2 ibD2 dyD2 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW16 aB16 nG16 nB16 eW16 eB16 pW16 pB16 sL16 ib16 dyO16 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW17 aB17 nG17 nB17 eW17 eB17 pW17 pB17 sL17 ib17 dyO17 lr
  ∧ cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε aW18 aB18 nG18 nB18 eW18 eB18 pW18 pB18 sL18 ib18 dyO18 lr
  ∧ cnxHeadChTiedAt gN xN bN dN epsStr lrStr cotN cotN ε hG hT Wfc bfc xhead g lr
  ∧ (∀ i : Fin 768, ∀ j : Fin 10,
        den (SHlo.weightSgd xN wN lrStr hn Wfc lr
              (.operand cotN (fun k => softmax 10 (mnistLinear Wfc bfc hn) k - oneHot 10 label k)))
            (finProdFinEquiv (i, j))
          = Wfc i j - lr * pdiv (fun v : Vec (768 * 10) => fun _ : Fin 1 =>
                crossEntropy 10 (dense (Mat.unflatten v) bfc hn) label)
              (Mat.flatten Wfc) (finProdFinEquiv (i, j)) 0)
  ∧ den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe (.operand nlogN (mnistLinear Wfc bfc hn))))
          (.operand ohN (oneHot 10 label)))
      = g := by
  intro ib1 ib2 ib3 ibD0 ib4 ib5 ib6 ibD1 ib7 ib8 ib9 ib10 ib11 ib12 ib13 ib14 ib15 ibD2 ib16 ib17 ib18 xhead gap hn g dyO18 dyO17 dyO16 dyD2 dyO15 dyO14 dyO13 dyO12 dyO11 dyO10 dyO9 dyO8 dyO7 dyD1 dyO6 dyO5 dyO4 dyD0 dyO3 dyO2 dyO1 dyStem
  refine ⟨cnx_stem_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε Wst psb psng psnbt x xstem dyStem lr,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 ib1 dyO1 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW2 aB2 nG2 nB2 eW2 eB2 pW2 pB2 sL2 ib2 dyO2 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW3 aB3 nG3 nB3 eW3 eB3 pW3 pB3 sL3 ib3 dyO3 lr
  · exact cnx_down_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε dG0 dT0 dW0 dB0 ibD0 dyD0 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW4 aB4 nG4 nB4 eW4 eB4 pW4 pB4 sL4 ib4 dyO4 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW5 aB5 nG5 nB5 eW5 eB5 pW5 pB5 sL5 ib5 dyO5 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW6 aB6 nG6 nB6 eW6 eB6 pW6 pB6 sL6 ib6 dyO6 lr
  · exact cnx_down_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε dG1 dT1 dW1 dB1 ibD1 dyD1 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW7 aB7 nG7 nB7 eW7 eB7 pW7 pB7 sL7 ib7 dyO7 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW8 aB8 nG8 nB8 eW8 eB8 pW8 pB8 sL8 ib8 dyO8 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW9 aB9 nG9 nB9 eW9 eB9 pW9 pB9 sL9 ib9 dyO9 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW10 aB10 nG10 nB10 eW10 eB10 pW10 pB10 sL10 ib10 dyO10 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW11 aB11 nG11 nB11 eW11 eB11 pW11 pB11 sL11 ib11 dyO11 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW12 aB12 nG12 nB12 eW12 eB12 pW12 pB12 sL12 ib12 dyO12 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW13 aB13 nG13 nB13 eW13 eB13 pW13 pB13 sL13 ib13 dyO13 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW14 aB14 nG14 nB14 eW14 eB14 pW14 pB14 sL14 ib14 dyO14 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW15 aB15 nG15 nB15 eW15 eB15 pW15 pB15 sL15 ib15 dyO15 lr
  · exact cnx_down_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε dG2 dT2 dW2 dB2 ibD2 dyD2 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW16 aB16 nG16 nB16 eW16 eB16 pW16 pB16 sL16 ib16 dyO16 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW17 aB17 nG17 nB17 eW17 eB17 pW17 pB17 sL17 ib17 dyO17 lr
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW18 aB18 nG18 nB18 eW18 eB18 pW18 pB18 sL18 ib18 dyO18 lr
  · exact cnx_head_ch_tiedAt gN xN bN dN epsStr lrStr cotN cotN ε hG hT Wfc bfc xhead g lr
  · exact fun i j => cnx_dense_tied_totalloss xN wN lrStr cotN Wfc bfc hn label lr i j
  · exact cnxLossCot_den nlogN ohN (mnistLinear Wfc bfc hn) label

end Proofs.CnxTiePoC
