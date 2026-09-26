import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtChainClose
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFold
import LeanMlir.Proofs.Architectures.ChannelLNBack
import LeanMlir.Proofs.Foundation.SgdNodes
import LeanMlir.Proofs.Foundation.SmoothedLossCot

/-! # The full [3,3,9,3] ConvNeXt-T step tie — the whole net tied through the real forward

The whole-net thread for the ConvNeXt-T schedule (the batched MobileNetV2 peer is
`MobileNetV2TieB.mnv2_net_tiedB`). The folds (`ConvNeXtFold` and the MobileNetV2 / ResNet-34 / ViT
folds it imports) make every rendered param op `den = certified ∀ cotangent`; this file feeds each
consumer the forward activations of the `convNextTrainStepFaithfulV` render and the loss-driven
backward-chain cotangent that net delivers, so the 18-block train step is den-composed
forward → loss → backward with no free activations and no symbolic cotangent. The statement is
per example, at one image and a hard label; the artifact runs a batch of 32, and that batch and
its mean lie outside this statement (the batched form is `ConvNeXtStepTieGB.lean`).

## The net this is about

`verified_mlir/convnext_train_step.mlir`, measured, not assumed:

| convention | this file | read from |
|---|---|---|
| depth / widths | `[3,3,9,3]`, 96 → 192 → 384 → 768 | `ConvNeXtRender.cnxTiny` |
| stem | 4×4/s4 patchify conv **then channel-LN** | `convNextFwdChain` |
| normalisation | `chanLNTensor3` (per-channel `[c]` affine, `h·w` statistics per example) at all 22 spatial sites: 1 stem + 18 block + 3 downsample | `ConvNeXtRender.lnFwdSite` |
| head | GAP → **vector-LN at one row** (`rowLNVecFlat 1 768`) → dense | `headLnFwdSite` |
| activation | GELU (smooth — no kink mask anywhere) | `fwdBlock` |
| layer scale | per-channel `Vec c`, broadcast by `chanIdx` | `layerScaleChF` |
| padding | symmetric; ConvNeXt is a PyTorch-origin net and has no XLA-`SAME` site | — |
| params | 182 | `allParams`, and the artifact's 184 func args (`%x` + 182 + `%onehot`) |

## The channel-LN specifics

* **channel-LN γ/β** (`Vec c`) at every spatial site, through `CnxPoC.chanLn{Gamma,Beta}Sgd_den`
  — the render re-emits the `[h·w, c]` transposes and runs ViT's `veclnGammaSgd` /
  `rowDenseBiasSgd` on that view, so the op operands here are `chanLNRows` of the saved LN input
  and of the chain cotangent, while the certified Jacobian is `chanLNTensor3`'s in the `c·h·w`
  activation layout (`ChannelLN`'s permutation argument bridges the two).
* **the channel-LN input-VJP** in the cotangent chain: `chanLNTensor3Back`. It is the certified
  VJP — `chanLNTensor3Back_eq_chanLN_vjp`.
* **the stem LN**: `psng`/`psnbt` tie at the stem-LN output cotangent, and the stem conv's own
  gradients see the LN input-VJP of it.
* **the head at the vector LN**: `hng`/`hnbt : Vec 768` through `SgdNode.veclnGammaSgd_den` /
  `rowDenseBiasSgd_den_lnbeta` at `N = 1`. Stated at the literal 768 because `1 * m` does not
  reduce at a variable `m` — the render's own documented trap, in the proof this time.
* **the four even-kernel weight grads.** The three downsample 2×2/s2 weights
  are `convStridedWeightSgd` (`SgdNode.convStridedW_den` is kernel-generic), and the stem
  4×4/s4 weight is `convStride4WeightGrad`, whose `den` is
  `flatConvStride4WeightGradHasVJP`.

## Coverage

All **182** parameters are tied. 181 of them at the full `θ − lr·(certified ∂Loss/∂θ)` step; the
stem weight `psW` at its **gradient**, because the render emits `convStride4WeightGrad` and wraps
it in hand-written `sgd` text (there is no fused `convStride4WeightSgd` op to be the `den` of). What
remains outside: the block backward is rendered hand-written, so the cotangent SSA ↔ chain-cot
correspondence is the per-op trust the whole suite carries; plus per-op `pretty` lexing; LN `0 < ε`
smoothness; ℝ → Float32 — the boundary every prior fold carries.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CnxTiePoC

open scoped BigOperators
open Proofs.SgdNode (vecLNGammaSgdTied_holds)
open Proofs.CnxPoC (chanLNBetaSgdTied_holds chanLNGammaSgdTied_holds)

/-! ## ConvNeXt block — all 9 params tied (depthwise → channel-LN → expand → GELU → project → layer-scale → +skip)

Forward: `out = addV( layerScaleCh lg (conv₁ₓ₁ₚᵣ( gelu( conv₁ₓ₁ₑₓ( chanLN( dw₇ₓ₇(xin) ))))), xin )`.
Backward from the block-output cotangent `dyOut` (the residual `addV` is the outermost op and there
is no post-add activation, so it passes `dyOut` straight to the layer-scale output): layer-scale-back
(`cnxCotP`) → project-conv-back → GELU mask (`cnxCotE`) → expand-conv-back (`cnxCotN`) → the
channel-LN input-VJP (`chanLNTensor3Back`) → depthwise-back. `cnxCotP`/`cnxCotE`/`cnxCotN` are
LN-form-agnostic and are reused verbatim from `ConvNeXtChainClose`. -/

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
  ∧ CnxPoC.ChanLNGammaSgdTied h w gN xN epsStr lrStr cotN ε nbt d ng cotN' lr
  ∧ CnxPoC.ChanLNBetaSgdTied h w bN lrStr cotN ε ng d nbt cotN' lr
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
  · intro idx; exact SgdNode.depthwiseW_den xN wN lrStr cotN bdw (Tensor3.unflatten xin) Wdw cotD lr idx
  · intro o;   exact SgdNode.depthwiseB_den bN lrStr cotN Wdw (Tensor3.unflatten xin) bdw cotD lr o
  · exact chanLNGammaSgdTied_holds
  · exact chanLNBetaSgdTied_holds
  · exact convWSgdTied_holds
  · exact convBSgdTied_holds
  · exact convWSgdTied_holds
  · exact convBSgdTied_holds
  · intro cc;  exact CnxPoC.layerScaleChGammaSgd_den gN xN lrStr cotN p lg dyOut lr cc

/-! ## Downsample — channel-LN → 2×2/s2 conv (all 4 params tied)

Forward: `o = convˢ²(chanLN(xin))` (LN over the block-input grid `2h×2w` with a `Vec ci` affine,
then a 2×2/s2 conv `ci → co`). No skip. Backward from `dyOut`: strided-conv-back (`cotN'`) →
channel-LN-back. The strided **weight** is no longer a gap: `convStridedWeightSgd` is emitted at
2×2 since `sWGradGeom` split the odd/even padding cases, and `SgdNode.convStridedW_den` is
kernel-generic. -/

/-- **Downsample, tied.** Channel-LN γ/β at the `ci·(2h)·(2w)` input grid, plus the strided conv's
    weight and bias, at the real forward + the chain cotangents. -/
def cnxDownChTied {ci co h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin n : Vec (ci*(2*h)*(2*w))) (dyOut : Vec (co*h*w)) (lr : ℝ) : Prop :=
    let cotN' : Vec (ci*(2*h)*(2*w)) := (flatConvStride2HasVJP Wd bd).backward n dyOut
    CnxPoC.ChanLNGammaSgdTied (2 * h) (2 * w) gN xN epsStr lrStr cotN ε dnbt xin dng cotN' lr
  ∧ CnxPoC.ChanLNBetaSgdTied (2 * h) (2 * w) bN lrStr cotN ε dng xin dnbt cotN' lr
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
  · exact chanLNGammaSgdTied_holds
  · exact chanLNBetaSgdTied_holds
  · intro idx; exact SgdNode.convStridedW_den xN wN lrStr cotN bd n Wd dyOut lr idx
  · intro o; exact SgdNode.convStridedB_den bN lrStr cotN Wd n bd dyOut lr o

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
    CnxPoC.ChanLNGammaSgdTied h w gN xN epsStr lrStr cotN ε psnbt patch psng dyStem lr
  ∧ CnxPoC.ChanLNBetaSgdTied h w bN lrStr cotN ε psng patch psnbt dyStem lr
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
  · exact chanLNGammaSgdTied_holds
  · exact chanLNBetaSgdTied_holds
  · exact convBSgdTied_holds
  · intro idx; exact flatConvStride4WeightGradHasVJP_correct psb x (Kernel4.flatten Wst) cotPatch idx

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
    let cotHn : Vec (1*768) := (denseHasVJP Wfc bfc).backward hn g
    SgdNode.VecLNGammaSgdTied 1 gN xN epsStr lrStr cotN ε hnbt gap hng cotHn lr
  ∧ SgdNode.VecLNBetaSgdTied 1 bN lrStr cotN ε hng gap hnbt cotHn lr
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
  · exact vecLNGammaSgdTied_holds
  · intro k;
    exact SgdNode.rowDenseBiasSgd_den_lnbeta bN lrStr cotN ε hng (Mat.unflatten gap) hnbt cotHn lr k
  · intro i; exact SgdNode.denseB_den bdN lrStr dyN Wfc hn bfc g lr i

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
  rw [SgdNode.denseW_den aN wN lrStr dyN a Wd bd
        (fun k => softmax 10 (mnistLinear Wd bd a) k - oneHot 10 label k) lr i j,
      StableHLO.lossWeightGrad_eq_sum Wd bd a label i j]

/-- **The emitted loss-cotangent graph denotes the softmax-CE gradient at the logits.** -/
theorem cnxLossCot_den (nlogN ohN : String) (logits : Vec 10) (label : Fin 10) :
    den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe (.operand nlogN logits)))
          (.operand ohN (oneHot 10 label)))
      = fun j => softmax 10 logits j - oneHot 10 label j :=
  StableHLO.softmaxCELossCot_den nlogN ohN _ label

/-! ## Forward aliases — thread block inputs through the real forward -/

/-- The stem output: patchify conv **then** channel-LN. -/
noncomputable def cnxStemFwdO {c h w : Nat} (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (bst psng psnbt : Vec c)
    (x : Vec (3*(2*(2*h))*(2*(2*w)))) : Vec (c*h*w) :=
  chanLNTensor3 c h w ε psng psnbt (flatConvStride4 Wst bst x)

noncomputable def cnxBlockBodyChO {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin : Vec (c*h*w)) : Vec (c*h*w) :=
  layerScale (fun k => lg (chanIdx c h w k))
    (flatConv (h := h) (w := w) Wpr bpr (gelu (cExp*h*w) (flatConv (h := h) (w := w) Wex bex
      (chanLNTensor3 c h w ε ng nbt (depthwiseFlat (h := h) (w := w) Wdw bdw xin)))))

noncomputable def cnxBlockFwdChO {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin : Vec (c*h*w)) : Vec (c*h*w) :=
  fun i => cnxBlockBodyChO ε Wdw bdw ng nbt Wex bex Wpr bpr lg xin i + xin i

noncomputable def cnxDownFwdChO {ci co h w : Nat} (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (ci*(2*h)*(2*w))) : Vec (co*h*w) :=
  flatConvStride2 Wd bd (chanLNTensor3 ci (2*h) (2*w) ε dng dnbt xin)

/-! ## Backward cot-in constructors — thread block dyOuts (the residual fan-in) -/

/-- ConvNeXt block input cotangent: `depthwise-back(cotD) + dyOut` (the identity-skip fan-in),
    with `cotD` through the channel-LN input-VJP. -/
noncomputable def cnxBlockCotInChAt {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin dyOut : Vec (c*h*w)) : Vec (c*h*w) :=
  let γlsB : Vec (c*h*w) := fun k => lg (chanIdx c h w k)
  let d := depthwiseFlat (h := h) (w := w) Wdw bdw xin
  let nl := chanLNTensor3 c h w ε ng nbt d
  let e := flatConv (h := h) (w := w) Wex bex nl
  let g := gelu (cExp*h*w) e
  let cotD := chanLNTensor3Back c h w ε ng d (cnxCotN γlsB Wex bex Wpr bpr nl g e dyOut)
  fun i => (depthwiseFlatHasVJP (h := h) (w := w) Wdw bdw).backward xin cotD i + dyOut i

/-- Downsample input cotangent (at `ci·(2h)·(2w)`): the channel-LN input-VJP of the
    strided-conv-back. No skip. -/
noncomputable def cnxDownCotInChAt {ci co h w : Nat} (ε : ℝ)
    (dng dnbt : Vec ci) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (ci*(2*h)*(2*w))) (dyOut : Vec (co*h*w)) : Vec (ci*(2*h)*(2*w)) :=
  let n := chanLNTensor3 ci (2*h) (2*w) ε dng dnbt xin
  let cotN := (flatConvStride2HasVJP Wd bd).backward n dyOut
  chanLNTensor3Back ci (2*h) (2*w) ε dng xin cotN

/-- The cotangent at the last block output `xhead` (= s3b2's `dyOut`): `gap-back(headLN-back(
    dense-back(g)))`. The head LN's input-VJP is the render's `rowScaleF γ` then `lnRowBack` at
    γ = 1, which is `rowLNVecFlatBack` (`ConvNeXtBackB0.rowLNBack_affine_eq`). -/
noncomputable def cnxHeadDyXheadCh {h w : Nat} (ε : ℝ)
    (hng hnbt : Vec 768) (Wfc : Mat 768 10) (bfc : Vec 10)
    (xhead : Vec (768*h*w)) (g : Vec 10) : Vec (768*h*w) :=
  let gap : Vec (1*768) := globalAvgPoolFlat 768 h w xhead
  let hn : Vec 768 := rowLNVecFlat 1 768 ε hng hnbt gap
  let cotHn : Vec (1*768) := (denseHasVJP Wfc bfc).backward hn g
  let cotGap : Vec 768 := rowLNVecFlatBack 1 768 ε hng gap cotHn
  (globalAvgPoolFlatHasVJP 768 h w).backward xhead cotGap

/-! ## Input-only `*TiedAt` wrappers — compute internals from a block's input

Note: only `cnxStemChTiedAt` is `@[irreducible]`: without it the capstone's
`refine ⟨cnx_stem_ch_tiedAt …, ?_, …⟩` times out in `whnf`, and moving it to its own goal does not
help. Every other definition in this section and the two above is a plain `def`. -/

def cnxBlockChTiedAt {c cExp h w : Nat}
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

def cnxDownChTiedAt {ci co h w : Nat}
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

def cnxHeadChTiedAt {h w : Nat}
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

/-! ## The ties' weight records

The whole-net ties share ONE `ε` across the stem, every block, downsample and head LN, so these
records carry no `ε` (unlike `CnxBlockParamsCh`, whose `εn` is per block). The dot-notation
wrappers below are `abbrev`s over the constructors above: a statement over a
record unfolds to exactly the unpacked one. -/

/-- One ConvNeXt block's weights: depthwise 7×7, channel-LN affine, the two 1×1 convs, layer
    scale. -/
structure CnxTieBlk (c cExp : Nat) where
  aW : DepthwiseKernel c 7 7
  aB : Vec c
  nG : Vec c
  nB : Vec c
  eW : Kernel4 cExp c 1 1
  eB : Vec cExp
  pW : Kernel4 c cExp 1 1
  pB : Vec c
  sL : Vec c

/-- One stage-boundary downsample's weights: channel-LN affine, then the 2×2/s2 conv. -/
structure CnxTieDown (ci co : Nat) where
  G : Vec ci
  T : Vec ci
  W : Kernel4 co ci 2 2
  B : Vec co

/-- ConvNeXt-T as the ties bind it: patchify stem + its LN, the `[3,3,9,3]` blocks, three
    downsamples, head LN + dense. -/
structure CnxTieWeights (nC : Nat) where
  sW : Kernel4 96 3 4 4
  sb : Vec 96
  sγ : Vec 96
  sβ : Vec 96
  b1 : CnxTieBlk 96 384
  b2 : CnxTieBlk 96 384
  b3 : CnxTieBlk 96 384
  d0 : CnxTieDown 96 192
  b4 : CnxTieBlk 192 768
  b5 : CnxTieBlk 192 768
  b6 : CnxTieBlk 192 768
  d1 : CnxTieDown 192 384
  b7 : CnxTieBlk 384 1536
  b8 : CnxTieBlk 384 1536
  b9 : CnxTieBlk 384 1536
  b10 : CnxTieBlk 384 1536
  b11 : CnxTieBlk 384 1536
  b12 : CnxTieBlk 384 1536
  b13 : CnxTieBlk 384 1536
  b14 : CnxTieBlk 384 1536
  b15 : CnxTieBlk 384 1536
  d2 : CnxTieDown 384 768
  b16 : CnxTieBlk 768 3072
  b17 : CnxTieBlk 768 3072
  b18 : CnxTieBlk 768 3072
  hG : Vec 768
  hT : Vec 768
  Wfc : Mat 768 nC
  bfc : Vec nC

namespace CnxTieBlk
variable {c cExp : Nat} (p : CnxTieBlk c cExp)

/-- The block's forward (`cnxBlockFwdChO`). -/
noncomputable abbrev fwdO {h w : Nat} (ε : ℝ) : Vec (c*h*w) → Vec (c*h*w) :=
  cnxBlockFwdChO ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL

/-- The block's input cotangent (`cnxBlockCotInChAt`). -/
noncomputable abbrev cotIn {h w : Nat} (ε : ℝ) : Vec (c*h*w) → Vec (c*h*w) → Vec (c*h*w) :=
  cnxBlockCotInChAt ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL

/-- The block's per-example tie (`cnxBlockChTiedAt`). -/
abbrev TiedAt {h w : Nat} (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (xin dyOut : Vec (c*h*w)) (lr : ℝ) : Prop :=
  cnxBlockChTiedAt xN wN bN gN epsStr lrStr cotN ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL
    xin dyOut lr

theorem tied_at {h w : Nat} (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (xin dyOut : Vec (c*h*w)) (lr : ℝ) : p.TiedAt xN wN bN gN epsStr lrStr cotN ε xin dyOut lr :=
  cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε _ _ _ _ _ _ _ _ _ xin dyOut lr

end CnxTieBlk

namespace CnxTieDown
variable {ci co : Nat} (p : CnxTieDown ci co)

/-- The downsample's forward (`cnxDownFwdChO`). -/
noncomputable abbrev fwdO {h w : Nat} (ε : ℝ) : Vec (ci*(2*h)*(2*w)) → Vec (co*h*w) :=
  cnxDownFwdChO ε p.G p.T p.W p.B

/-- The downsample's input cotangent (`cnxDownCotInChAt`). -/
noncomputable abbrev cotIn {h w : Nat} (ε : ℝ) : Vec (ci*(2*h)*(2*w)) → Vec (co*h*w) → Vec (ci*(2*h)*(2*w)) :=
  cnxDownCotInChAt ε p.G p.T p.W p.B

/-- The downsample's per-example tie (`cnxDownChTiedAt`). -/
abbrev TiedAt {h w : Nat} (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (xin : Vec (ci*(2*h)*(2*w))) (dyOut : Vec (co*h*w)) (lr : ℝ) : Prop :=
  cnxDownChTiedAt xN wN bN gN epsStr lrStr cotN ε p.G p.T p.W p.B xin dyOut lr

theorem tied_at {h w : Nat} (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (xin : Vec (ci*(2*h)*(2*w))) (dyOut : Vec (co*h*w)) (lr : ℝ) :
    p.TiedAt xN wN bN gN epsStr lrStr cotN ε xin dyOut lr :=
  cnx_down_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε _ _ _ _ xin dyOut lr

end CnxTieDown

/-! ## The whole-net capstone — all 182 params through the REAL forward + composed cotangent

The `convNextTrainStepFaithfulV` forward threaded: block inputs are the forward prefixes
(`cnxStemFwdO` / `cnxBlockFwdChO` / `cnxDownFwdChO`), and the backward cotangents are composed
from the loss `g = softmax(logits) − onehot` down through dense (`denseHasVJP`) + the head LN +
GAP (`globalAvgPoolFlatHasVJP`) + every block's backward, with the residual fan-in `+ dyOut` at
each of the eighteen identity-skip merges, the channel-LN-back at each of the three downsamples,
and the stem LN's own back before the patchify conv's gradients. Each stem / block / down / head
tie then holds at its real input + threaded cotangent. The whole [3,3,9,3]
182-parameter ConvNeXt-T train step is den-composed forward → loss → backward, no free
activations, no symbolic cotangent. -/

/-- **The whole [3,3,9,3] ConvNeXt-T train step, tied.** Threading the real (channel-LN,
    per-channel layer-scale) forward and the loss-driven backward cotangent chain (GELU masks, the
    residual fan-in at every identity skip, the channel-LN-back at every downsample and at the
    stem), the 18 ConvNeXt blocks, the 3 downsamples, the 4×4/s4 stem with its LN, the
    GAP → LN → dense head, and the dense total-loss fold + loss-cotangent graph all denote the
    certified loss-descent step. All 182 parameters; `psW` at its gradient (its SGD wrap is
    hand-written text). The statement is per example, at one image `x` and a hard label `label`;
    the artifact's batch of 32 and its mean lie outside it (the batched form is
    `CnxTiePoCGB.cnx_net_tiedGB`). -/
theorem cnx_net_tied_certified
    (xN wN bN gN epsStr lrStr cotN dN nlogN ohN : String) (ε : ℝ)
    (w : CnxTieWeights 10) (xstem : Tensor3 3 56 56)
    (x : Vec (3*224*224)) (label : Fin 10) (lr : ℝ) :
    -- forward block inputs (the prefixes of the committed render's forward)
    let ib1   : Vec (96*56*56)         := cnxStemFwdO (h := 56) (w := 56) ε w.sW w.sb w.sγ w.sβ x
    let ib2   : Vec (96*56*56)         := w.b1.fwdO ε ib1
    let ib3   : Vec (96*56*56)         := w.b2.fwdO ε ib2
    let ibD0  : Vec (96*56*56)         := w.b3.fwdO ε ib3
    let ib4   : Vec (192*28*28)        := w.d0.fwdO (h := 28) (w := 28) ε ibD0
    let ib5   : Vec (192*28*28)        := w.b4.fwdO ε ib4
    let ib6   : Vec (192*28*28)        := w.b5.fwdO ε ib5
    let ibD1  : Vec (192*28*28)        := w.b6.fwdO ε ib6
    let ib7   : Vec (384*14*14)        := w.d1.fwdO (h := 14) (w := 14) ε ibD1
    let ib8   : Vec (384*14*14)        := w.b7.fwdO ε ib7
    let ib9   : Vec (384*14*14)        := w.b8.fwdO ε ib8
    let ib10  : Vec (384*14*14)        := w.b9.fwdO ε ib9
    let ib11  : Vec (384*14*14)        := w.b10.fwdO ε ib10
    let ib12  : Vec (384*14*14)        := w.b11.fwdO ε ib11
    let ib13  : Vec (384*14*14)        := w.b12.fwdO ε ib12
    let ib14  : Vec (384*14*14)        := w.b13.fwdO ε ib13
    let ib15  : Vec (384*14*14)        := w.b14.fwdO ε ib14
    let ibD2  : Vec (384*14*14)        := w.b15.fwdO ε ib15
    let ib16  : Vec (768*7*7)          := w.d2.fwdO (h := 7) (w := 7) ε ibD2
    let ib17  : Vec (768*7*7)          := w.b16.fwdO ε ib16
    let ib18  : Vec (768*7*7)          := w.b17.fwdO ε ib17
    let xhead : Vec (768*7*7)          := w.b18.fwdO ε ib18
    -- head forward + the loss cotangent
    let gap : Vec (1*768) := globalAvgPoolFlat 768 7 7 xhead
    let hn  : Vec 768     := rowLNVecFlat 1 768 ε w.hG w.hT gap
    let g   : Vec 10      := fun k => softmax 10 (mnistLinear w.Wfc w.bfc hn) k - oneHot 10 label k
    -- backward cotangents (composed from the loss; residual fan-in at each skip, LN-back at each
    -- downsample and at the stem)
    let dyO18  : Vec (768*7*7)         := cnxHeadDyXheadCh (h := 7) (w := 7) ε w.hG w.hT w.Wfc w.bfc xhead g
    let dyO17  : Vec (768*7*7)         := w.b18.cotIn ε ib18 dyO18
    let dyO16  : Vec (768*7*7)         := w.b17.cotIn ε ib17 dyO17
    let dyD2   : Vec (768*7*7)         := w.b16.cotIn ε ib16 dyO16
    let dyO15  : Vec (384*14*14)       := w.d2.cotIn (h := 7) (w := 7) ε ibD2 dyD2
    let dyO14  : Vec (384*14*14)       := w.b15.cotIn ε ib15 dyO15
    let dyO13  : Vec (384*14*14)       := w.b14.cotIn ε ib14 dyO14
    let dyO12  : Vec (384*14*14)       := w.b13.cotIn ε ib13 dyO13
    let dyO11  : Vec (384*14*14)       := w.b12.cotIn ε ib12 dyO12
    let dyO10  : Vec (384*14*14)       := w.b11.cotIn ε ib11 dyO11
    let dyO9   : Vec (384*14*14)       := w.b10.cotIn ε ib10 dyO10
    let dyO8   : Vec (384*14*14)       := w.b9.cotIn ε ib9 dyO9
    let dyO7   : Vec (384*14*14)       := w.b8.cotIn ε ib8 dyO8
    let dyD1   : Vec (384*14*14)       := w.b7.cotIn ε ib7 dyO7
    let dyO6   : Vec (192*28*28)       := w.d1.cotIn (h := 14) (w := 14) ε ibD1 dyD1
    let dyO5   : Vec (192*28*28)       := w.b6.cotIn ε ib6 dyO6
    let dyO4   : Vec (192*28*28)       := w.b5.cotIn ε ib5 dyO5
    let dyD0   : Vec (192*28*28)       := w.b4.cotIn ε ib4 dyO4
    let dyO3   : Vec (96*56*56)        := w.d0.cotIn (h := 28) (w := 28) ε ibD0 dyD0
    let dyO2   : Vec (96*56*56)        := w.b3.cotIn ε ib3 dyO3
    let dyO1   : Vec (96*56*56)        := w.b2.cotIn ε ib2 dyO2
    let dyStem : Vec (96*56*56)        := w.b1.cotIn ε ib1 dyO1
    -- the stem, every block, every downsample, the head, the dense total-loss fold + loss cot
    cnxStemChTiedAt xN wN bN gN epsStr lrStr cotN ε w.sW w.sb w.sγ w.sβ x xstem dyStem lr
  ∧ w.b1.TiedAt xN wN bN gN epsStr lrStr cotN ε ib1 dyO1 lr
  ∧ w.b2.TiedAt xN wN bN gN epsStr lrStr cotN ε ib2 dyO2 lr
  ∧ w.b3.TiedAt xN wN bN gN epsStr lrStr cotN ε ib3 dyO3 lr
  ∧ w.d0.TiedAt xN wN bN gN epsStr lrStr cotN ε ibD0 dyD0 lr
  ∧ w.b4.TiedAt xN wN bN gN epsStr lrStr cotN ε ib4 dyO4 lr
  ∧ w.b5.TiedAt xN wN bN gN epsStr lrStr cotN ε ib5 dyO5 lr
  ∧ w.b6.TiedAt xN wN bN gN epsStr lrStr cotN ε ib6 dyO6 lr
  ∧ w.d1.TiedAt xN wN bN gN epsStr lrStr cotN ε ibD1 dyD1 lr
  ∧ w.b7.TiedAt xN wN bN gN epsStr lrStr cotN ε ib7 dyO7 lr
  ∧ w.b8.TiedAt xN wN bN gN epsStr lrStr cotN ε ib8 dyO8 lr
  ∧ w.b9.TiedAt xN wN bN gN epsStr lrStr cotN ε ib9 dyO9 lr
  ∧ w.b10.TiedAt xN wN bN gN epsStr lrStr cotN ε ib10 dyO10 lr
  ∧ w.b11.TiedAt xN wN bN gN epsStr lrStr cotN ε ib11 dyO11 lr
  ∧ w.b12.TiedAt xN wN bN gN epsStr lrStr cotN ε ib12 dyO12 lr
  ∧ w.b13.TiedAt xN wN bN gN epsStr lrStr cotN ε ib13 dyO13 lr
  ∧ w.b14.TiedAt xN wN bN gN epsStr lrStr cotN ε ib14 dyO14 lr
  ∧ w.b15.TiedAt xN wN bN gN epsStr lrStr cotN ε ib15 dyO15 lr
  ∧ w.d2.TiedAt xN wN bN gN epsStr lrStr cotN ε ibD2 dyD2 lr
  ∧ w.b16.TiedAt xN wN bN gN epsStr lrStr cotN ε ib16 dyO16 lr
  ∧ w.b17.TiedAt xN wN bN gN epsStr lrStr cotN ε ib17 dyO17 lr
  ∧ w.b18.TiedAt xN wN bN gN epsStr lrStr cotN ε ib18 dyO18 lr
  ∧ cnxHeadChTiedAt gN xN bN dN epsStr lrStr cotN cotN ε w.hG w.hT w.Wfc w.bfc xhead g lr
  ∧ (∀ i : Fin 768, ∀ j : Fin 10,
        den (SHlo.weightSgd xN wN lrStr hn w.Wfc lr
              (.operand cotN (fun k => softmax 10 (mnistLinear w.Wfc w.bfc hn) k - oneHot 10 label k)))
            (finProdFinEquiv (i, j))
          = w.Wfc i j - lr * pdiv (fun v : Vec (768 * 10) => fun _ : Fin 1 =>
                crossEntropy 10 (dense (Mat.unflatten v) w.bfc hn) label)
              (Mat.flatten w.Wfc) (finProdFinEquiv (i, j)) 0)
  ∧ den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe (.operand nlogN (mnistLinear w.Wfc w.bfc hn))))
          (.operand ohN (oneHot 10 label)))
      = g := by
  intro ib1 ib2 ib3 ibD0 ib4 ib5 ib6 ibD1 ib7 ib8 ib9 ib10 ib11 ib12 ib13 ib14 ib15 ibD2 ib16 ib17 ib18 xhead gap hn g dyO18 dyO17 dyO16 dyD2 dyO15 dyO14 dyO13 dyO12 dyO11 dyO10 dyO9 dyO8 dyO7 dyD1 dyO6 dyO5 dyO4 dyD0 dyO3 dyO2 dyO1 dyStem
  refine ⟨cnx_stem_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε w.sW w.sb w.sγ w.sβ x xstem dyStem lr,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact w.b1.tied_at xN wN bN gN epsStr lrStr cotN ε ib1 dyO1 lr
  · exact w.b2.tied_at xN wN bN gN epsStr lrStr cotN ε ib2 dyO2 lr
  · exact w.b3.tied_at xN wN bN gN epsStr lrStr cotN ε ib3 dyO3 lr
  · exact w.d0.tied_at xN wN bN gN epsStr lrStr cotN ε ibD0 dyD0 lr
  · exact w.b4.tied_at xN wN bN gN epsStr lrStr cotN ε ib4 dyO4 lr
  · exact w.b5.tied_at xN wN bN gN epsStr lrStr cotN ε ib5 dyO5 lr
  · exact w.b6.tied_at xN wN bN gN epsStr lrStr cotN ε ib6 dyO6 lr
  · exact w.d1.tied_at xN wN bN gN epsStr lrStr cotN ε ibD1 dyD1 lr
  · exact w.b7.tied_at xN wN bN gN epsStr lrStr cotN ε ib7 dyO7 lr
  · exact w.b8.tied_at xN wN bN gN epsStr lrStr cotN ε ib8 dyO8 lr
  · exact w.b9.tied_at xN wN bN gN epsStr lrStr cotN ε ib9 dyO9 lr
  · exact w.b10.tied_at xN wN bN gN epsStr lrStr cotN ε ib10 dyO10 lr
  · exact w.b11.tied_at xN wN bN gN epsStr lrStr cotN ε ib11 dyO11 lr
  · exact w.b12.tied_at xN wN bN gN epsStr lrStr cotN ε ib12 dyO12 lr
  · exact w.b13.tied_at xN wN bN gN epsStr lrStr cotN ε ib13 dyO13 lr
  · exact w.b14.tied_at xN wN bN gN epsStr lrStr cotN ε ib14 dyO14 lr
  · exact w.b15.tied_at xN wN bN gN epsStr lrStr cotN ε ib15 dyO15 lr
  · exact w.d2.tied_at xN wN bN gN epsStr lrStr cotN ε ibD2 dyD2 lr
  · exact w.b16.tied_at xN wN bN gN epsStr lrStr cotN ε ib16 dyO16 lr
  · exact w.b17.tied_at xN wN bN gN epsStr lrStr cotN ε ib17 dyO17 lr
  · exact w.b18.tied_at xN wN bN gN epsStr lrStr cotN ε ib18 dyO18 lr
  · exact cnx_head_ch_tiedAt gN xN bN dN epsStr lrStr cotN cotN ε w.hG w.hT w.Wfc w.bfc xhead g lr
  · exact fun i j => cnx_dense_tied_totalloss xN wN lrStr cotN w.Wfc w.bfc hn label lr i j
  · exact cnxLossCot_den nlogN ohN (mnistLinear w.Wfc w.bfc hn) label

end Proofs.CnxTiePoC
