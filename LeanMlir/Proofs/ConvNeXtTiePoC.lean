import LeanMlir.Proofs.ConvNeXtChainClose
import LeanMlir.Proofs.ConvNeXtFaithfulPoC
import LeanMlir.Proofs.MobileNetV2FaithfulPoC
import LeanMlir.Proofs.ResNet34FaithfulPoC
import LeanMlir.Proofs.MlpTrainStep

/-! # PoC: the FULL [3,3,9,3] ConvNeXt-T §1a TIE — the whole net tied through the real forward

The Chapter-8 §1a tie: mnv2's whole-net thread (`Mnv2TiePoC.mnv2_net_tied_certified`) for the
ConvNeXt-T schedule. The §1 fold (`ConvNeXtFaithfulPoC` + `ConvNeXtClose`/M2/M3) already makes every
rendered param op `den = certified ∀ cotangent`; this file feeds each consumer the **real forward
activations** of the committed `convNextTrainStepFaithfulV` render and the **loss-driven
backward-chain cotangent** the ConvNeXt net actually delivers — so the whole 18-block train step is
den-composed forward→loss→backward, no free activations, no symbolic cotangent.

**What's new vs mnv2's tie:**
* **GELU mask** (smooth, no kink) where mnv2 had the relu6 two-kink mask: the expand-output cotangent
  carries `geluScalarDeriv` (`cnxCotE`), not a `selectMid`.
* **scalar LayerNorm γ/β** (`Vec 1`) at every normalization site (block / downsample / head), via
  `CnxPoC.lnGammaSgd_den` / `lnBetaSgd_den` — mnv2 had per-channel BN.
* **per-channel layer-scale γ** (`Vec c`, broadcast by `chanIdx`), via
  `CnxPoC.layerScaleChGammaSgd_den` — a ConvNeXt-signature family mnv2/r34 had no analogue of.
* **single block type** (every ConvNeXt block keeps resolution → identity skip always present), one
  **downsample** type (LN → 2×2/s2 conv, no skip), a bare **4×4/s4 stem** (no LN), and a
  **GAP → LN → dense** head.

**Reuse, no new core ops, no new bridges.** Each block-type tie is pure instantiation of the generic
`den = certified` lemmas at the `ConvNeXtChainClose` chain cotangents
(`cnxCotP`/`cnxCotE`/`cnxCotN`/`cnxCotD`):
* depthwise 7×7 → `Mnv2PoC.depthwiseW_den`/`depthwiseB_den` (kernel-generic);
* expand/project/stem/head 1×1 convs + stem bias → `CifarPoC.convW_den`/`convB_den`;
* scalar-LN γ/β → `CnxPoC.lnGammaSgd_den`/`lnBetaSgd_den`;
* per-channel layer-scale γ → `CnxPoC.layerScaleChGammaSgd_den`;
* downsample strided-conv bias → `ResNet34PoC.convStridedB_den`;
* dense head → `Cifar8PoC.denseW_den`/`denseB_den` (+ the total-loss fold).

## Coverage / honest residual
The four **even-kernel weight grads** (`psW` 4×4/s4 stem + `d0W`/`d1W`/`d2W` 2×2/s2 downsamples) are
the documented §1 render gap — hand-written, NOT `den(SHlo op)` — so they are outside this den-tie
(their *bias* grads ARE tied). Every other rendered param (176 of them) is tied here. The block
backward is rendered hand-written, so the cotangent SSA ↔ chain-cot correspondence is the per-op
trust the whole suite carries; plus per-op `pretty` lexing; LN `0 < ε` smoothness; ℝ → Float32 — the
boundary every prior fold carries.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CnxTiePoC

open scoped BigOperators

/-! ## ConvNeXt block — all 9 params tied (depthwise → LN → expand → GELU → project → layer-scale → +skip)

Forward (scalar-LN, per-channel layer-scale):
`out = addV( layerScaleCh lg (conv₁ₓ₁ₚᵣ( gelu( conv₁ₓ₁ₑₓ( LN( dw₇ₓ₇(xin) ))))), xin )`.
Backward from the block-output cotangent `dyOut` (the residual `addV` is the outermost op and there is
no post-add activation, so it passes `dyOut` straight to the layer-scale output): layer-scale-back
(`cnxCotP`) → project-conv-back → GELU mask (`cnxCotE`) → expand-conv-back (`cnxCotN`) → scalar-LN
input-VJP (`cnxCotD`) → depthwise-back. -/

/-- **ConvNeXt block, tied.** All 9 params (depthwise 7×7 `W`+`b`, scalar-LN γ/β, expand/project 1×1
    conv `W`+`b`, per-channel layer-scale γ) denote the certified loss-descent step at the real block
    forward activations + the `ConvNeXtChainClose` chain cotangents driven by `dyOut`. -/
def cnxBlockTied {c cExp h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec 1)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c)
    (xin d nl p : Vec (c*h*w)) (e g : Vec (cExp*h*w))
    (dyOut : Vec (c*h*w)) (lr : ℝ) : Prop :=
    let γlsB : Vec (c*h*w) := fun k => lg (chanIdx c h w k)
    let cotP : Vec (c*h*w) := cnxCotP γlsB dyOut
    let cotE : Vec (cExp*h*w) := cnxCotE γlsB Wpr bpr g e dyOut
    let cotN' : Vec (c*h*w) := cnxCotN γlsB Wex bex Wpr bpr nl g e dyOut
    let cotD : Vec (c*h*w) := cnxCotD ε (ng 0) γlsB Wex bex Wpr bpr d nl g e dyOut
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
    -- scalar-LN γ/β  (cot = cotN', LN input = d)
  ∧ (∀ i : Fin 1,
        den (SHlo.lnGammaSgd gN xN epsStr lrStr ε d ng lr (.operand cotN cotN')) i
          = ng 0 - lr * ∑ j : Fin (c*h*w),
              pdiv (fun γ' : Vec 1 => layerNormForward (c*h*w) ε (γ' 0) (nbt 0) d) ng 0 j * cotN' j)
  ∧ (∀ i : Fin 1,
        den (SHlo.lnBetaSgd bN lrStr nbt lr (.operand cotN cotN')) i
          = nbt 0 - lr * ∑ j : Fin (c*h*w),
              pdiv (fun β' : Vec 1 => layerNormForward (c*h*w) ε (ng 0) (β' 0) d) nbt 0 j * cotN' j)
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

theorem cnx_block_tied {c cExp h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec 1)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c)
    (xin d nl p : Vec (c*h*w)) (e g : Vec (cExp*h*w))
    (dyOut : Vec (c*h*w)) (lr : ℝ) :
    cnxBlockTied xN wN bN gN epsStr lrStr cotN ε Wdw bdw ng nbt Wex bex Wpr bpr lg
      xin d nl p e g dyOut lr := by
  unfold cnxBlockTied
  intro γlsB cotP cotE cotN' cotD
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact Mnv2PoC.depthwiseW_den xN wN lrStr cotN bdw (Tensor3.unflatten xin) Wdw cotD lr idx
  · intro o;   exact Mnv2PoC.depthwiseB_den bN lrStr cotN Wdw (Tensor3.unflatten xin) bdw cotD lr o
  · intro i;   exact CnxPoC.lnGammaSgd_den gN xN epsStr lrStr cotN ε (nbt 0) d ng cotN' lr i
  · intro i;   exact CnxPoC.lnBetaSgd_den bN lrStr cotN ε (ng 0) nbt d cotN' lr i
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN bex (Tensor3.unflatten nl) Wex cotE lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN Wex (Tensor3.unflatten nl) bex cotE lr o
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN bpr (Tensor3.unflatten g) Wpr cotP lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN Wpr (Tensor3.unflatten g) bpr cotP lr o
  · intro cc;  exact CnxPoC.layerScaleChGammaSgd_den gN xN lrStr cotN p lg dyOut lr cc

/-! ## Downsample — LN → 2×2/s2 conv (3 tied params: γ/β + strided bias; the strided weight is the gap)

Forward: `o = convˢ²(LN(xin))` (LN over the block-input grid `2h×2w`, then a 2×2/s2 conv `ci → co`).
No skip. Backward from `dyOut`: strided-conv-back (`cotN`) → LN-back. The strided **weight** grad
(`d{·}W`) is the documented even-kernel render gap — not an SHlo op, so not tied here. -/

/-- **Downsample, tied.** The LN γ/β (at the `ci·(2h)·(2w)` input grid) + the strided-conv bias, at
    the real forward + the chain cotangents. (The strided weight is the gap.) -/
def cnxDownTied {ci co h w : Nat}
    (_wN bN gN xN epsStr lrStr cotN : String) (ε : ℝ)
    (dng dnbt : Vec 1) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (ci*(2*h)*(2*w))) (n : Vec (ci*(2*h)*(2*w)))
    (dyOut : Vec (co*h*w)) (lr : ℝ) : Prop :=
    let cotN' : Vec (ci*(2*h)*(2*w)) := (flatConvStride2_has_vjp Wd bd).backward n dyOut
    -- LN γ/β  (cot = cotN', LN input = xin)
    (∀ i : Fin 1,
        den (SHlo.lnGammaSgd gN xN epsStr lrStr ε xin dng lr (.operand cotN cotN')) i
          = dng 0 - lr * ∑ j : Fin (ci*(2*h)*(2*w)),
              pdiv (fun γ' : Vec 1 => layerNormForward (ci*(2*h)*(2*w)) ε (γ' 0) (dnbt 0) xin) dng 0 j * cotN' j)
  ∧ (∀ i : Fin 1,
        den (SHlo.lnBetaSgd bN lrStr dnbt lr (.operand cotN cotN')) i
          = dnbt 0 - lr * ∑ j : Fin (ci*(2*h)*(2*w)),
              pdiv (fun β' : Vec 1 => layerNormForward (ci*(2*h)*(2*w)) ε (dng 0) (β' 0) xin) dnbt 0 j * cotN' j)
    -- strided-conv bias (ci → co)  (cot = dyOut, conv input = n = LN output)
  ∧ (∀ o : Fin co,
        den (SHlo.convStridedBiasSgd bN lrStr Wd n bd lr (.operand cotN dyOut)) o
          = bd o - lr * ∑ j : Fin (co*h*w),
              pdiv (fun b' : Vec co => flatConvStride2 Wd b' n) bd o j * dyOut j)

theorem cnx_down_tied {ci co h w : Nat}
    (wN bN gN xN epsStr lrStr cotN : String) (ε : ℝ)
    (dng dnbt : Vec 1) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (ci*(2*h)*(2*w))) (n : Vec (ci*(2*h)*(2*w)))
    (dyOut : Vec (co*h*w)) (lr : ℝ) :
    cnxDownTied wN bN gN xN epsStr lrStr cotN ε dng dnbt Wd bd xin n dyOut lr := by
  unfold cnxDownTied
  intro cotN'
  refine ⟨?_, ?_, ?_⟩
  · intro i; exact CnxPoC.lnGammaSgd_den gN xN epsStr lrStr cotN ε (dnbt 0) xin dng cotN' lr i
  · intro i; exact CnxPoC.lnBetaSgd_den bN lrStr cotN ε (dng 0) dnbt xin cotN' lr i
  · intro o; exact ResNet34PoC.convStridedB_den bN lrStr cotN Wd n bd dyOut lr o

/-! ## Stem bias — the 4×4/s4 patchify conv bias (the weight `psW` is the even-kernel gap)

The stem is a bare 4×4/s4 conv (no LN). Its bias grad is the spatial reduce of the stem-output
cotangent (= block s0b0's input cotangent), modelled as a 1×1-conv-at-output-resolution bias grad
(stride-independent) — generic in the carried `W,x`. -/

/-- **Stem bias, tied.** The 4×4/s4 stem bias denotes the certified step at the stem-output cotangent
    `dyStem`; `W`/`x` are carried generically (the bias grad is a pure cotangent reduce). -/
def cnxStemBiasTied {ic c h w : Nat}
    (bN lrStr cotN : String) (Wst : Kernel4 c ic 4 4) (xstem : Tensor3 ic h w) (psb : Vec c)
    (dyStem : Vec (c*h*w)) (lr : ℝ) : Prop :=
    ∀ o : Fin c,
      den (SHlo.convBiasSgd bN lrStr Wst xstem psb lr (.operand cotN dyStem)) o
        = psb o - lr * ∑ j : Fin (c*h*w),
            pdiv (fun b' : Vec c => Tensor3.flatten (conv2d Wst b' xstem)) psb o j * dyStem j

theorem cnx_stem_bias_tied {ic c h w : Nat}
    (bN lrStr cotN : String) (Wst : Kernel4 c ic 4 4) (xstem : Tensor3 ic h w) (psb : Vec c)
    (dyStem : Vec (c*h*w)) (lr : ℝ) :
    cnxStemBiasTied bN lrStr cotN Wst xstem psb dyStem lr := by
  intro o; exact CifarPoC.convB_den bN lrStr cotN Wst xstem psb dyStem lr o

/-! ## Head — GAP → scalar-LN → dense (LN γ/β + dense W/b; the dense weight folds to the total loss) -/

/-- **Head LN + dense bias, tied.** The head-LN γ/β (at the pooled `gap`, cot = the dense-back
    `cot_hn`) and the dense bias (cot = the loss `g`) at the real forward. -/
def cnxHeadTied {m : Nat}
    (gN xN bN bdN epsStr lrStr cotN dyN : String) (ε : ℝ)
    (hng hnbt : Vec 1) (Wd : Mat m 10) (bd : Vec 10)
    (gap hn : Vec m) (g : Vec 10) (lr : ℝ) : Prop :=
    let cotHn : Vec m := (dense_has_vjp Wd bd).backward hn g
    (∀ i : Fin 1,
        den (SHlo.lnGammaSgd gN xN epsStr lrStr ε gap hng lr (.operand cotN cotHn)) i
          = hng 0 - lr * ∑ j : Fin m,
              pdiv (fun γ' : Vec 1 => layerNormForward m ε (γ' 0) (hnbt 0) gap) hng 0 j * cotHn j)
  ∧ (∀ i : Fin 1,
        den (SHlo.lnBetaSgd bN lrStr hnbt lr (.operand cotN cotHn)) i
          = hnbt 0 - lr * ∑ j : Fin m,
              pdiv (fun β' : Vec 1 => layerNormForward m ε (hng 0) (β' 0) gap) hnbt 0 j * cotHn j)
  ∧ (∀ i : Fin 10,
        den (SHlo.biasSgd bdN lrStr bd lr (.operand dyN g)) i
          = bd i - lr * ∑ j : Fin 10, pdiv (fun b' : Vec 10 => dense Wd b' hn) bd i j * g j)

theorem cnx_head_tied {m : Nat}
    (gN xN bN bdN epsStr lrStr cotN dyN : String) (ε : ℝ)
    (hng hnbt : Vec 1) (Wd : Mat m 10) (bd : Vec 10)
    (gap hn : Vec m) (g : Vec 10) (lr : ℝ) :
    cnxHeadTied gN xN bN bdN epsStr lrStr cotN dyN ε hng hnbt Wd bd gap hn g lr := by
  unfold cnxHeadTied
  intro cotHn
  refine ⟨?_, ?_, ?_⟩
  · intro i; exact CnxPoC.lnGammaSgd_den gN xN epsStr lrStr cotN ε (hnbt 0) gap hng cotHn lr i
  · intro i; exact CnxPoC.lnBetaSgd_den bN lrStr cotN ε (hng 0) hnbt gap cotHn lr i
  · intro i; exact Cifar8PoC.denseB_den bdN lrStr dyN Wd hn bd g lr i

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

/-! ## Forward block aliases (`@[irreducible]`) — thread block inputs through the real forward

`cnxBlockBodyO` is the ConvNeXt block body (scalar-LN, per-channel layer-scale = what the render
emits and what the proof's `convNextBlockBody` computes when the layer-scale is the per-channel
broadcast); `cnxBlockFwdO` adds the identity skip. `@[irreducible]` so the 18-deep nested composition
stays opaque during the capstone's dimension inference (the r34/mnv2 heartbeat lesson). -/

@[irreducible] noncomputable def cnxBlockBodyO {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec 1)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin : Vec (c*h*w)) : Vec (c*h*w) :=
  layerScale (fun k => lg (chanIdx c h w k))
    (flatConv (h := h) (w := w) Wpr bpr (gelu (cExp*h*w) (flatConv (h := h) (w := w) Wex bex
      (layerNormForward (c*h*w) ε (ng 0) (nbt 0) (depthwiseFlat (h := h) (w := w) Wdw bdw xin)))))

@[irreducible] noncomputable def cnxBlockFwdO {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec 1)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin : Vec (c*h*w)) : Vec (c*h*w) :=
  fun i => cnxBlockBodyO ε Wdw bdw ng nbt Wex bex Wpr bpr lg xin i + xin i

@[irreducible] noncomputable def cnxDownFwdO {ci co h w : Nat} (ε : ℝ)
    (dng dnbt : Vec 1) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (ci*(2*h)*(2*w))) : Vec (co*h*w) :=
  flatConvStride2 Wd bd (layerNormForward (ci*(2*h)*(2*w)) ε (dng 0) (dnbt 0) xin)

@[irreducible] noncomputable def cnxStemFwdO {c h w : Nat}
    (Wst : Kernel4 c 3 4 4) (bst : Vec c) (x : Vec (3*(2*(2*h))*(2*(2*w)))) : Vec (c*h*w) :=
  flatConvStride4 Wst bst x

/-! ## Backward cot-in constructors (`@[irreducible]`) — thread block dyOuts (the residual fan-in) -/

/-- ConvNeXt block input cotangent: `depthwise-back(cnxCotD) + dyOut` (the identity-skip fan-in). -/
@[irreducible] noncomputable def cnxBlockCotInAt {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec 1)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin dyOut : Vec (c*h*w)) : Vec (c*h*w) :=
  let γlsB : Vec (c*h*w) := fun k => lg (chanIdx c h w k)
  let d := depthwiseFlat (h := h) (w := w) Wdw bdw xin
  let nl := layerNormForward (c*h*w) ε (ng 0) (nbt 0) d
  let e := flatConv (h := h) (w := w) Wex bex nl
  let g := gelu (cExp*h*w) e
  let cotD := cnxCotD ε (ng 0) γlsB Wex bex Wpr bpr d nl g e dyOut
  fun i => (depthwiseFlat_has_vjp (h := h) (w := w) Wdw bdw).backward xin cotD i + dyOut i

/-- Downsample input cotangent (at `ci·(2h)·(2w)`): LN input-VJP of the strided-conv-back. No skip. -/
@[irreducible] noncomputable def cnxDownCotInAt {ci co h w : Nat} (ε : ℝ)
    (dng dnbt : Vec 1) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (ci*(2*h)*(2*w))) (dyOut : Vec (co*h*w)) : Vec (ci*(2*h)*(2*w)) :=
  let n := layerNormForward (ci*(2*h)*(2*w)) ε (dng 0) (dnbt 0) xin
  let cotN := (flatConvStride2_has_vjp Wd bd).backward n dyOut
  bn_grad_input (ci*(2*h)*(2*w)) ε (dng 0) xin cotN

/-- The cotangent at the last block output `xhead` (= s3b2's `dyOut`): `gap-back(LN-back(dense-back(g)))`. -/
@[irreducible] noncomputable def cnxHeadDyXhead {c h w : Nat} (ε : ℝ)
    (hng hnbt : Vec 1) (Wd : Mat c 10) (bd : Vec 10) (xhead : Vec (c*h*w)) (g : Vec 10) : Vec (c*h*w) :=
  let gap := globalAvgPoolFlat c h w xhead
  let hn := layerNormForward c ε (hng 0) (hnbt 0) gap
  let cotHn := (dense_has_vjp Wd bd).backward hn g
  let cotGap := bn_grad_input c ε (hng 0) gap cotHn
  (globalAvgPoolFlat_has_vjp c h w).backward xhead cotGap

/-! ## Input-only `TiedAt` wrappers (`@[irreducible]`) — compute internals from a block's input -/

@[irreducible] def cnxBlockTiedAt {c cExp h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec 1)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin : Vec (c*h*w)) (dyOut : Vec (c*h*w)) (lr : ℝ) : Prop :=
  let d := depthwiseFlat (h := h) (w := w) Wdw bdw xin
  let nl := layerNormForward (c*h*w) ε (ng 0) (nbt 0) d
  let e := flatConv (h := h) (w := w) Wex bex nl
  let g := gelu (cExp*h*w) e
  let p := flatConv (h := h) (w := w) Wpr bpr g
  cnxBlockTied xN wN bN gN epsStr lrStr cotN ε Wdw bdw ng nbt Wex bex Wpr bpr lg
    xin d nl p e g dyOut lr

theorem cnx_block_tiedAt {c cExp h w : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec 1)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin : Vec (c*h*w)) (dyOut : Vec (c*h*w)) (lr : ℝ) :
    cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε Wdw bdw ng nbt Wex bex Wpr bpr lg xin dyOut lr := by
  unfold cnxBlockTiedAt
  intro d nl e g p
  exact cnx_block_tied xN wN bN gN epsStr lrStr cotN ε Wdw bdw ng nbt Wex bex Wpr bpr lg
    xin d nl p e g dyOut lr

@[irreducible] def cnxDownTiedAt {ci co h w : Nat}
    (wN bN gN xN epsStr lrStr cotN : String) (ε : ℝ)
    (dng dnbt : Vec 1) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (ci*(2*h)*(2*w))) (dyOut : Vec (co*h*w)) (lr : ℝ) : Prop :=
  let n := layerNormForward (ci*(2*h)*(2*w)) ε (dng 0) (dnbt 0) xin
  cnxDownTied wN bN gN xN epsStr lrStr cotN ε dng dnbt Wd bd xin n dyOut lr

theorem cnx_down_tiedAt {ci co h w : Nat}
    (wN bN gN xN epsStr lrStr cotN : String) (ε : ℝ)
    (dng dnbt : Vec 1) (Wd : Kernel4 co ci 2 2) (bd : Vec co)
    (xin : Vec (ci*(2*h)*(2*w))) (dyOut : Vec (co*h*w)) (lr : ℝ) :
    cnxDownTiedAt wN bN gN xN epsStr lrStr cotN ε dng dnbt Wd bd xin dyOut lr := by
  unfold cnxDownTiedAt
  intro n
  exact cnx_down_tied wN bN gN xN epsStr lrStr cotN ε dng dnbt Wd bd xin n dyOut lr

@[irreducible] def cnxHeadTiedAt {c h w : Nat}
    (gN xN bN bdN epsStr lrStr cotN dyN : String) (ε : ℝ)
    (hng hnbt : Vec 1) (Wd : Mat c 10) (bd : Vec 10)
    (xhead : Vec (c*h*w)) (g : Vec 10) (lr : ℝ) : Prop :=
  let gap := globalAvgPoolFlat c h w xhead
  let hn := layerNormForward c ε (hng 0) (hnbt 0) gap
  cnxHeadTied gN xN bN bdN epsStr lrStr cotN dyN ε hng hnbt Wd bd gap hn g lr

theorem cnx_head_tiedAt {c h w : Nat}
    (gN xN bN bdN epsStr lrStr cotN dyN : String) (ε : ℝ)
    (hng hnbt : Vec 1) (Wd : Mat c 10) (bd : Vec 10)
    (xhead : Vec (c*h*w)) (g : Vec 10) (lr : ℝ) :
    cnxHeadTiedAt gN xN bN bdN epsStr lrStr cotN dyN ε hng hnbt Wd bd xhead g lr := by
  unfold cnxHeadTiedAt
  intro gap hn
  exact cnx_head_tied gN xN bN bdN epsStr lrStr cotN dyN ε hng hnbt Wd bd gap hn g lr

/-! ## The whole-net capstone — all 176 tied params through the REAL forward + composed cotangent

The committed `convNextTrainStepFaithfulV` forward threaded: block inputs are the forward prefixes
(`cnxStemFwdO` / `cnxBlockFwdO` / `cnxDownFwdO`), and the backward cotangents are composed from the
loss `g = softmax(logits) − onehot` down through dense (`dense_has_vjp`) + the head LN + GAP
(`globalAvgPoolFlat_has_vjp`) + every block's backward, with the residual fan-in `+ dyOut` at each of
the eighteen identity-skip merges and the LN-back at each of the three downsamples. Each block / down /
head / stem-bias tie then holds at its real input + threaded cotangent. The full §1a tie: the whole
[3,3,9,3] (176-tied-param) ConvNeXt-T train step is den-composed forward→loss→backward, no free
activations, no symbolic cotangent. The four even-kernel weight grads (stem 4×4/s4 + 3 downsample
2×2/s2) are the documented §1 render gap, outside this den-tie. -/

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 400000 in
/-- **The whole [3,3,9,3] ConvNeXt-T train step, tied.** Threading the real (scalar-LN, per-channel
    layer-scale) forward and the loss-driven backward cotangent chain (GELU masks, the residual fan-in
    at every identity skip, the LN-back at every downsample), the 18 ConvNeXt blocks, the 3
    downsamples, the GAP→LN→dense head, the dense total-loss fold + loss-cotangent graph, and the
    stem bias all denote the certified loss-descent step. -/
theorem cnx_net_tied_certified
    (xN wN bN gN epsStr lrStr cotN dN nlogN ohN : String) (ε : ℝ)
    (Wst : Kernel4 96 3 4 4) (psb : Vec 96) (xstem : Tensor3 3 56 56)
    -- stage 0 (c=96, cExp=384): blocks 1,2,3
    (aW1 : DepthwiseKernel 96 7 7) (aB1 : Vec 96) (nG1 nB1 : Vec 1) (eW1 : Kernel4 384 96 1 1) (eB1 : Vec 384) (pW1 : Kernel4 96 384 1 1) (pB1 sL1 : Vec 96)
    (aW2 : DepthwiseKernel 96 7 7) (aB2 : Vec 96) (nG2 nB2 : Vec 1) (eW2 : Kernel4 384 96 1 1) (eB2 : Vec 384) (pW2 : Kernel4 96 384 1 1) (pB2 sL2 : Vec 96)
    (aW3 : DepthwiseKernel 96 7 7) (aB3 : Vec 96) (nG3 nB3 : Vec 1) (eW3 : Kernel4 384 96 1 1) (eB3 : Vec 384) (pW3 : Kernel4 96 384 1 1) (pB3 sL3 : Vec 96)
    (dG0 dT0 : Vec 1) (dW0 : Kernel4 192 96 2 2) (dB0 : Vec 192)
    -- stage 1 (c=192, cExp=768): blocks 4,5,6
    (aW4 : DepthwiseKernel 192 7 7) (aB4 : Vec 192) (nG4 nB4 : Vec 1) (eW4 : Kernel4 768 192 1 1) (eB4 : Vec 768) (pW4 : Kernel4 192 768 1 1) (pB4 sL4 : Vec 192)
    (aW5 : DepthwiseKernel 192 7 7) (aB5 : Vec 192) (nG5 nB5 : Vec 1) (eW5 : Kernel4 768 192 1 1) (eB5 : Vec 768) (pW5 : Kernel4 192 768 1 1) (pB5 sL5 : Vec 192)
    (aW6 : DepthwiseKernel 192 7 7) (aB6 : Vec 192) (nG6 nB6 : Vec 1) (eW6 : Kernel4 768 192 1 1) (eB6 : Vec 768) (pW6 : Kernel4 192 768 1 1) (pB6 sL6 : Vec 192)
    (dG1 dT1 : Vec 1) (dW1 : Kernel4 384 192 2 2) (dB1 : Vec 384)
    -- stage 2 (c=384, cExp=1536): blocks 7..15
    (aW7 : DepthwiseKernel 384 7 7) (aB7 : Vec 384) (nG7 nB7 : Vec 1) (eW7 : Kernel4 1536 384 1 1) (eB7 : Vec 1536) (pW7 : Kernel4 384 1536 1 1) (pB7 sL7 : Vec 384)
    (aW8 : DepthwiseKernel 384 7 7) (aB8 : Vec 384) (nG8 nB8 : Vec 1) (eW8 : Kernel4 1536 384 1 1) (eB8 : Vec 1536) (pW8 : Kernel4 384 1536 1 1) (pB8 sL8 : Vec 384)
    (aW9 : DepthwiseKernel 384 7 7) (aB9 : Vec 384) (nG9 nB9 : Vec 1) (eW9 : Kernel4 1536 384 1 1) (eB9 : Vec 1536) (pW9 : Kernel4 384 1536 1 1) (pB9 sL9 : Vec 384)
    (aW10 : DepthwiseKernel 384 7 7) (aB10 : Vec 384) (nG10 nB10 : Vec 1) (eW10 : Kernel4 1536 384 1 1) (eB10 : Vec 1536) (pW10 : Kernel4 384 1536 1 1) (pB10 sL10 : Vec 384)
    (aW11 : DepthwiseKernel 384 7 7) (aB11 : Vec 384) (nG11 nB11 : Vec 1) (eW11 : Kernel4 1536 384 1 1) (eB11 : Vec 1536) (pW11 : Kernel4 384 1536 1 1) (pB11 sL11 : Vec 384)
    (aW12 : DepthwiseKernel 384 7 7) (aB12 : Vec 384) (nG12 nB12 : Vec 1) (eW12 : Kernel4 1536 384 1 1) (eB12 : Vec 1536) (pW12 : Kernel4 384 1536 1 1) (pB12 sL12 : Vec 384)
    (aW13 : DepthwiseKernel 384 7 7) (aB13 : Vec 384) (nG13 nB13 : Vec 1) (eW13 : Kernel4 1536 384 1 1) (eB13 : Vec 1536) (pW13 : Kernel4 384 1536 1 1) (pB13 sL13 : Vec 384)
    (aW14 : DepthwiseKernel 384 7 7) (aB14 : Vec 384) (nG14 nB14 : Vec 1) (eW14 : Kernel4 1536 384 1 1) (eB14 : Vec 1536) (pW14 : Kernel4 384 1536 1 1) (pB14 sL14 : Vec 384)
    (aW15 : DepthwiseKernel 384 7 7) (aB15 : Vec 384) (nG15 nB15 : Vec 1) (eW15 : Kernel4 1536 384 1 1) (eB15 : Vec 1536) (pW15 : Kernel4 384 1536 1 1) (pB15 sL15 : Vec 384)
    (dG2 dT2 : Vec 1) (dW2 : Kernel4 768 384 2 2) (dB2 : Vec 768)
    -- stage 3 (c=768, cExp=3072): blocks 16,17,18
    (aW16 : DepthwiseKernel 768 7 7) (aB16 : Vec 768) (nG16 nB16 : Vec 1) (eW16 : Kernel4 3072 768 1 1) (eB16 : Vec 3072) (pW16 : Kernel4 768 3072 1 1) (pB16 sL16 : Vec 768)
    (aW17 : DepthwiseKernel 768 7 7) (aB17 : Vec 768) (nG17 nB17 : Vec 1) (eW17 : Kernel4 3072 768 1 1) (eB17 : Vec 3072) (pW17 : Kernel4 768 3072 1 1) (pB17 sL17 : Vec 768)
    (aW18 : DepthwiseKernel 768 7 7) (aB18 : Vec 768) (nG18 nB18 : Vec 1) (eW18 : Kernel4 3072 768 1 1) (eB18 : Vec 3072) (pW18 : Kernel4 768 3072 1 1) (pB18 sL18 : Vec 768)
    -- head
    (hG hT : Vec 1) (Wfc : Mat 768 10) (bfc : Vec 10)
    (x : Vec (3*224*224)) (label : Fin 10) (lr : ℝ) :
    -- forward block inputs (the prefixes of the committed render's forward)
    let ib1   : Vec (96*56*56)  := cnxStemFwdO (h := 56) (w := 56) Wst psb x
    let ib2   : Vec (96*56*56)  := cnxBlockFwdO ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 ib1
    let ib3   : Vec (96*56*56)  := cnxBlockFwdO ε aW2 aB2 nG2 nB2 eW2 eB2 pW2 pB2 sL2 ib2
    let ibD0  : Vec (96*56*56)  := cnxBlockFwdO ε aW3 aB3 nG3 nB3 eW3 eB3 pW3 pB3 sL3 ib3
    let ib4   : Vec (192*28*28) := cnxDownFwdO (h := 28) (w := 28) ε dG0 dT0 dW0 dB0 ibD0
    let ib5   : Vec (192*28*28) := cnxBlockFwdO ε aW4 aB4 nG4 nB4 eW4 eB4 pW4 pB4 sL4 ib4
    let ib6   : Vec (192*28*28) := cnxBlockFwdO ε aW5 aB5 nG5 nB5 eW5 eB5 pW5 pB5 sL5 ib5
    let ibD1  : Vec (192*28*28) := cnxBlockFwdO ε aW6 aB6 nG6 nB6 eW6 eB6 pW6 pB6 sL6 ib6
    let ib7   : Vec (384*14*14) := cnxDownFwdO (h := 14) (w := 14) ε dG1 dT1 dW1 dB1 ibD1
    let ib8   : Vec (384*14*14) := cnxBlockFwdO ε aW7 aB7 nG7 nB7 eW7 eB7 pW7 pB7 sL7 ib7
    let ib9   : Vec (384*14*14) := cnxBlockFwdO ε aW8 aB8 nG8 nB8 eW8 eB8 pW8 pB8 sL8 ib8
    let ib10  : Vec (384*14*14) := cnxBlockFwdO ε aW9 aB9 nG9 nB9 eW9 eB9 pW9 pB9 sL9 ib9
    let ib11  : Vec (384*14*14) := cnxBlockFwdO ε aW10 aB10 nG10 nB10 eW10 eB10 pW10 pB10 sL10 ib10
    let ib12  : Vec (384*14*14) := cnxBlockFwdO ε aW11 aB11 nG11 nB11 eW11 eB11 pW11 pB11 sL11 ib11
    let ib13  : Vec (384*14*14) := cnxBlockFwdO ε aW12 aB12 nG12 nB12 eW12 eB12 pW12 pB12 sL12 ib12
    let ib14  : Vec (384*14*14) := cnxBlockFwdO ε aW13 aB13 nG13 nB13 eW13 eB13 pW13 pB13 sL13 ib13
    let ib15  : Vec (384*14*14) := cnxBlockFwdO ε aW14 aB14 nG14 nB14 eW14 eB14 pW14 pB14 sL14 ib14
    let ibD2  : Vec (384*14*14) := cnxBlockFwdO ε aW15 aB15 nG15 nB15 eW15 eB15 pW15 pB15 sL15 ib15
    let ib16  : Vec (768*7*7)   := cnxDownFwdO (h := 7) (w := 7) ε dG2 dT2 dW2 dB2 ibD2
    let ib17  : Vec (768*7*7)   := cnxBlockFwdO ε aW16 aB16 nG16 nB16 eW16 eB16 pW16 pB16 sL16 ib16
    let ib18  : Vec (768*7*7)   := cnxBlockFwdO ε aW17 aB17 nG17 nB17 eW17 eB17 pW17 pB17 sL17 ib17
    let xhead : Vec (768*7*7)   := cnxBlockFwdO ε aW18 aB18 nG18 nB18 eW18 eB18 pW18 pB18 sL18 ib18
    -- head forward + the loss cotangent
    let gap : Vec 768 := globalAvgPoolFlat 768 7 7 xhead
    let hn  : Vec 768 := layerNormForward 768 ε (hG 0) (hT 0) gap
    let g   : Vec 10  := fun k => softmax 10 (mnistLinear Wfc bfc hn) k - oneHot 10 label k
    -- backward cotangents (composed from the loss; residual fan-in at each skip, LN-back at each downsample)
    let dyO18 : Vec (768*7*7)   := cnxHeadDyXhead (h := 7) (w := 7) ε hG hT Wfc bfc xhead g
    let dyO17 : Vec (768*7*7)   := cnxBlockCotInAt ε aW18 aB18 nG18 nB18 eW18 eB18 pW18 pB18 sL18 ib18 dyO18
    let dyO16 : Vec (768*7*7)   := cnxBlockCotInAt ε aW17 aB17 nG17 nB17 eW17 eB17 pW17 pB17 sL17 ib17 dyO17
    let dyD2  : Vec (768*7*7)   := cnxBlockCotInAt ε aW16 aB16 nG16 nB16 eW16 eB16 pW16 pB16 sL16 ib16 dyO16
    let dyO15 : Vec (384*14*14) := cnxDownCotInAt (h := 7) (w := 7) ε dG2 dT2 dW2 dB2 ibD2 dyD2
    let dyO14 : Vec (384*14*14) := cnxBlockCotInAt ε aW15 aB15 nG15 nB15 eW15 eB15 pW15 pB15 sL15 ib15 dyO15
    let dyO13 : Vec (384*14*14) := cnxBlockCotInAt ε aW14 aB14 nG14 nB14 eW14 eB14 pW14 pB14 sL14 ib14 dyO14
    let dyO12 : Vec (384*14*14) := cnxBlockCotInAt ε aW13 aB13 nG13 nB13 eW13 eB13 pW13 pB13 sL13 ib13 dyO13
    let dyO11 : Vec (384*14*14) := cnxBlockCotInAt ε aW12 aB12 nG12 nB12 eW12 eB12 pW12 pB12 sL12 ib12 dyO12
    let dyO10 : Vec (384*14*14) := cnxBlockCotInAt ε aW11 aB11 nG11 nB11 eW11 eB11 pW11 pB11 sL11 ib11 dyO11
    let dyO9  : Vec (384*14*14) := cnxBlockCotInAt ε aW10 aB10 nG10 nB10 eW10 eB10 pW10 pB10 sL10 ib10 dyO10
    let dyO8  : Vec (384*14*14) := cnxBlockCotInAt ε aW9 aB9 nG9 nB9 eW9 eB9 pW9 pB9 sL9 ib9 dyO9
    let dyO7  : Vec (384*14*14) := cnxBlockCotInAt ε aW8 aB8 nG8 nB8 eW8 eB8 pW8 pB8 sL8 ib8 dyO8
    let dyD1  : Vec (384*14*14) := cnxBlockCotInAt ε aW7 aB7 nG7 nB7 eW7 eB7 pW7 pB7 sL7 ib7 dyO7
    let dyO6  : Vec (192*28*28) := cnxDownCotInAt (h := 14) (w := 14) ε dG1 dT1 dW1 dB1 ibD1 dyD1
    let dyO5  : Vec (192*28*28) := cnxBlockCotInAt ε aW6 aB6 nG6 nB6 eW6 eB6 pW6 pB6 sL6 ib6 dyO6
    let dyO4  : Vec (192*28*28) := cnxBlockCotInAt ε aW5 aB5 nG5 nB5 eW5 eB5 pW5 pB5 sL5 ib5 dyO5
    let dyD0  : Vec (192*28*28) := cnxBlockCotInAt ε aW4 aB4 nG4 nB4 eW4 eB4 pW4 pB4 sL4 ib4 dyO4
    let dyO3  : Vec (96*56*56)  := cnxDownCotInAt (h := 28) (w := 28) ε dG0 dT0 dW0 dB0 ibD0 dyD0
    let dyO2  : Vec (96*56*56)  := cnxBlockCotInAt ε aW3 aB3 nG3 nB3 eW3 eB3 pW3 pB3 sL3 ib3 dyO3
    let dyO1  : Vec (96*56*56)  := cnxBlockCotInAt ε aW2 aB2 nG2 nB2 eW2 eB2 pW2 pB2 sL2 ib2 dyO2
    let dyStem : Vec (96*56*56) := cnxBlockCotInAt ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 ib1 dyO1
    -- the stem bias, every block, every downsample, the head, the dense total-loss fold + loss cot
    cnxStemBiasTied bN lrStr cotN Wst xstem psb dyStem lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 ib1 dyO1 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW2 aB2 nG2 nB2 eW2 eB2 pW2 pB2 sL2 ib2 dyO2 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW3 aB3 nG3 nB3 eW3 eB3 pW3 pB3 sL3 ib3 dyO3 lr
  ∧ cnxDownTiedAt wN bN gN xN epsStr lrStr cotN ε dG0 dT0 dW0 dB0 ibD0 dyD0 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW4 aB4 nG4 nB4 eW4 eB4 pW4 pB4 sL4 ib4 dyO4 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW5 aB5 nG5 nB5 eW5 eB5 pW5 pB5 sL5 ib5 dyO5 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW6 aB6 nG6 nB6 eW6 eB6 pW6 pB6 sL6 ib6 dyO6 lr
  ∧ cnxDownTiedAt wN bN gN xN epsStr lrStr cotN ε dG1 dT1 dW1 dB1 ibD1 dyD1 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW7 aB7 nG7 nB7 eW7 eB7 pW7 pB7 sL7 ib7 dyO7 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW8 aB8 nG8 nB8 eW8 eB8 pW8 pB8 sL8 ib8 dyO8 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW9 aB9 nG9 nB9 eW9 eB9 pW9 pB9 sL9 ib9 dyO9 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW10 aB10 nG10 nB10 eW10 eB10 pW10 pB10 sL10 ib10 dyO10 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW11 aB11 nG11 nB11 eW11 eB11 pW11 pB11 sL11 ib11 dyO11 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW12 aB12 nG12 nB12 eW12 eB12 pW12 pB12 sL12 ib12 dyO12 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW13 aB13 nG13 nB13 eW13 eB13 pW13 pB13 sL13 ib13 dyO13 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW14 aB14 nG14 nB14 eW14 eB14 pW14 pB14 sL14 ib14 dyO14 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW15 aB15 nG15 nB15 eW15 eB15 pW15 pB15 sL15 ib15 dyO15 lr
  ∧ cnxDownTiedAt wN bN gN xN epsStr lrStr cotN ε dG2 dT2 dW2 dB2 ibD2 dyD2 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW16 aB16 nG16 nB16 eW16 eB16 pW16 pB16 sL16 ib16 dyO16 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW17 aB17 nG17 nB17 eW17 eB17 pW17 pB17 sL17 ib17 dyO17 lr
  ∧ cnxBlockTiedAt xN wN bN gN epsStr lrStr cotN ε aW18 aB18 nG18 nB18 eW18 eB18 pW18 pB18 sL18 ib18 dyO18 lr
  ∧ cnxHeadTiedAt gN xN bN dN epsStr lrStr cotN cotN ε hG hT Wfc bfc xhead g lr
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
  intro ib1 ib2 ib3 ibD0 ib4 ib5 ib6 ibD1 ib7 ib8 ib9 ib10 ib11 ib12 ib13 ib14 ib15 ibD2
        ib16 ib17 ib18 xhead gap hn g dyO18 dyO17 dyO16 dyD2 dyO15 dyO14 dyO13 dyO12 dyO11
        dyO10 dyO9 dyO8 dyO7 dyD1 dyO6 dyO5 dyO4 dyD0 dyO3 dyO2 dyO1 dyStem
  refine ⟨cnx_stem_bias_tied bN lrStr cotN Wst xstem psb dyStem lr, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 ib1 dyO1 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW2 aB2 nG2 nB2 eW2 eB2 pW2 pB2 sL2 ib2 dyO2 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW3 aB3 nG3 nB3 eW3 eB3 pW3 pB3 sL3 ib3 dyO3 lr
  · exact cnx_down_tiedAt wN bN gN xN epsStr lrStr cotN ε dG0 dT0 dW0 dB0 ibD0 dyD0 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW4 aB4 nG4 nB4 eW4 eB4 pW4 pB4 sL4 ib4 dyO4 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW5 aB5 nG5 nB5 eW5 eB5 pW5 pB5 sL5 ib5 dyO5 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW6 aB6 nG6 nB6 eW6 eB6 pW6 pB6 sL6 ib6 dyO6 lr
  · exact cnx_down_tiedAt wN bN gN xN epsStr lrStr cotN ε dG1 dT1 dW1 dB1 ibD1 dyD1 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW7 aB7 nG7 nB7 eW7 eB7 pW7 pB7 sL7 ib7 dyO7 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW8 aB8 nG8 nB8 eW8 eB8 pW8 pB8 sL8 ib8 dyO8 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW9 aB9 nG9 nB9 eW9 eB9 pW9 pB9 sL9 ib9 dyO9 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW10 aB10 nG10 nB10 eW10 eB10 pW10 pB10 sL10 ib10 dyO10 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW11 aB11 nG11 nB11 eW11 eB11 pW11 pB11 sL11 ib11 dyO11 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW12 aB12 nG12 nB12 eW12 eB12 pW12 pB12 sL12 ib12 dyO12 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW13 aB13 nG13 nB13 eW13 eB13 pW13 pB13 sL13 ib13 dyO13 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW14 aB14 nG14 nB14 eW14 eB14 pW14 pB14 sL14 ib14 dyO14 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW15 aB15 nG15 nB15 eW15 eB15 pW15 pB15 sL15 ib15 dyO15 lr
  · exact cnx_down_tiedAt wN bN gN xN epsStr lrStr cotN ε dG2 dT2 dW2 dB2 ibD2 dyD2 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW16 aB16 nG16 nB16 eW16 eB16 pW16 pB16 sL16 ib16 dyO16 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW17 aB17 nG17 nB17 eW17 eB17 pW17 pB17 sL17 ib17 dyO17 lr
  · exact cnx_block_tiedAt xN wN bN gN epsStr lrStr cotN ε aW18 aB18 nG18 nB18 eW18 eB18 pW18 pB18 sL18 ib18 dyO18 lr
  · exact cnx_head_tiedAt gN xN bN dN epsStr lrStr cotN cotN ε hG hT Wfc bfc xhead g lr
  · exact fun i j => cnx_dense_tied_totalloss xN wN lrStr cotN Wfc bfc hn label lr i j
  · exact cnxLossCot_den nlogN ohN (mnistLinear Wfc bfc hn) label

end Proofs.CnxTiePoC
