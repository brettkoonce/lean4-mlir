import LeanMlir.Proofs.MobileNetV2ChainClose
import LeanMlir.Proofs.MobileNetV2FaithfulPoC
import LeanMlir.Proofs.ResNet34FaithfulPoC
import LeanMlir.Proofs.MobileNetV2FullPaper

/-! # PoC: the FULL 17-block paper MobileNetV2 §1a TIE — the whole net tied through the real forward

The Chapter-7 §1a tie: r34's whole-net thread (`ResNet34TiePoC.r34_net_tied_certified`) scaled to the
full `[t,c,n,s]` MobileNetV2. The paper §1 fold (`MobileNetV2FaithfulPoCPaper`) already makes every
one of the 210 params `den = certified ∀ c`; this file feeds each consumer the **real forward
activations** of `mobilenetv2ForwardPaper` and the **loss-driven backward-chain cotangent** the
inverted-residual net actually delivers — so the WHOLE 17-block train step is den-composed
forward→loss→backward, no free activations, no symbolic cotangent.

**What's new vs r34's tie:**
* **relu6 two-kink mask.** Where r34 masks with `if a i > 0` (relu), mnv2 masks with
  `if 0 < x i ∧ x i < 6` (the `selectMid` two-sided kink) at both the expand and depthwise BN outputs.
* **linear project bottleneck.** The project 1×1 conv has NO relu6 after the `addV`, so the
  project-BN-output cotangent is `dyOut` *directly* (skip and no-skip alike) — `invresCotPc` is a
  plain BN-back (see `MobileNetV2ChainClose`).
* **3 block variants** (no-expand b1, stride-1 expand b3/5/6/8-10/12/13/15/16 + the no-skip widenings
  b11/b17, stride-2 downsample b2/4/7/14) and a **conv-bn-relu6 head** (not r34's bare GAP→dense).

**Reuse, no new core ops, no new bridges.** Each block-type tie is pure instantiation of the generic
`den = certified` lemmas at the `MobileNetV2ChainClose` chain cotangents:
* expand/project 1×1 convs → `CifarPoC.convW_den`/`convB_den`;
* depthwise (stride-1) → `Mnv2PoC.depthwiseW_den`/`depthwiseB_den`; (stride-2) → `…Strided…`;
* per-channel BN γ/β → `CifarBnPoC.bnGamma_den`/`bnBeta_den`;
* stem 3×3/s2 conv → `ResNet34PoC.convStridedW_den`/`convStridedB_den`;
* final dense → `Cifar8PoC.denseW_den`/`denseB_den`.

## Honest residual (the boundary every prior fold carries)
* The block backward is rendered hand-written, so the cotangent SSA ↔ chain-cot correspondence is the
  per-op trust the whole suite carries; per-op `pretty` lexing; relu6 two-kink + BN `0 < ε` smoothness;
  ℝ → Float32.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.Mnv2TiePoC

open scoped BigOperators

/-! ## Stride-1 inverted-residual block — all 12 params tied (expand → depthwise(s1) → project)

Forward (per-channel BN): `body = bn(convₚ(relu6(bn(dwconv(relu6(bn(convₑ x))))))`; the block output is
`addV(body, x)` (skip) or `body` (no-skip widening) — IDENTICAL param ops either way, only the block
*input* cotangent (the fan-in) differs, handled in the cot-in constructors. Backward from the block
output cotangent `dyOut`: project-BN-back (linear, cot = `dyOut`) → project-conv-back → depthwise
relu6 mask → depthwise-BN-back → depthwise-back → expand relu6 mask → expand-BN-back → expand-conv-back. -/

/-- **Stride-1 inverted-residual block, tied.** All 12 params (expand/project 1×1 conv `W`+`b`,
    depthwise `W`+`b`, three per-channel BN γ/β) denote the certified loss-descent step at the real
    block forward activations + the `MobileNetV2ChainClose` chain cotangents driven by `dyOut`. -/
def ivS1Tied {ic mid oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid 3 3) (bd : Vec mid) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp : Vec oc)
    (xin : Tensor3 ic h w) (ec en er dc dn dr : Vec (mid*h*w)) (pc : Vec (oc*h*w))
    (dyOut : Vec (oc*h*w)) (lr : ℝ) : Prop :=
    let cotPc : Vec (oc*h*w) := invresCotPc ε γp pc dyOut
    let cotDr : Vec (mid*h*w) := (hasVJP3_to_hasVJP (conv2d_has_vjp3 Wp bp)).backward dr cotPc
    let dyBnD : Vec (mid*h*w) := fun i => if 0 < dn i ∧ dn i < 6 then cotDr i else 0
    let cotDc : Vec (mid*h*w) := invresCotDc ε γd γp Wp bp dr dn dc pc dyOut
    let cotEr : Vec (mid*h*w) := (depthwiseFlat_has_vjp (h := h) (w := w) Wd bd).backward er cotDc
    let dyBnE : Vec (mid*h*w) := fun i => if 0 < en i ∧ en i < 6 then cotEr i else 0
    let cotEc : Vec (mid*h*w) := invresCotEcS1 ε γe γd γp Wd bd Wp bp er en ec dr dn dc pc dyOut
    -- expand 1×1 conv (ic → mid)
    (∀ idx : Fin (mid*ic*1*1),
        den (SHlo.convWeightSgd xN wN lrStr be xin We lr (.operand cotN cotEc)) idx
          = Kernel4.flatten We idx - lr * ∑ j : Fin (mid*h*w),
              pdiv (fun v' : Vec (mid*ic*1*1) => Tensor3.flatten (conv2d (Kernel4.unflatten v') be xin))
                   (Kernel4.flatten We) idx j * cotEc j)
  ∧ (∀ o : Fin mid,
        den (SHlo.convBiasSgd bN lrStr We xin be lr (.operand cotN cotEc)) o
          = be o - lr * ∑ j : Fin (mid*h*w),
              pdiv (fun b' : Vec mid => Tensor3.flatten (conv2d We b' xin)) be o j * cotEc j)
    -- expand BN γ/β  (cot = dyBnE, the expand relu6-masked cotangent)
  ∧ (∀ idx : Fin mid,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γe ec lr (.operand cotN dyBnE)) idx
          = γe idx - lr * ∑ j : Fin (mid*(h*w)),
              pdiv (fun γ' : Vec mid => bnPerChannelFlat mid (h*w) ε γ' βe (reassocFwd mid h w ec))
                   γe idx j * reassocFwd mid h w dyBnE j)
  ∧ (∀ idx : Fin mid,
        den (SHlo.bnBetaSgd bN lrStr βe lr (.operand cotN dyBnE)) idx
          = βe idx - lr * ∑ j : Fin (mid*(h*w)),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (h*w) ε γe β' (reassocFwd mid h w ec))
                   βe idx j * reassocFwd mid h w dyBnE j)
    -- depthwise (stride-1) W/b  (cot = cotDc)
  ∧ (∀ idx : Fin (mid*3*3),
        den (SHlo.depthwiseWeightSgd xN wN lrStr bd (Tensor3.unflatten er) Wd lr (.operand cotN cotDc)) idx
          = Tensor3.flatten Wd idx - lr * ∑ j : Fin (mid*h*w),
              pdiv (fun v' : Vec (mid*3*3) => Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') bd (Tensor3.unflatten er)))
                   (Tensor3.flatten Wd) idx j * cotDc j)
  ∧ (∀ o : Fin mid,
        den (SHlo.depthwiseBiasSgd bN lrStr Wd (Tensor3.unflatten er) bd lr (.operand cotN cotDc)) o
          = bd o - lr * ∑ j : Fin (mid*h*w),
              pdiv (fun b' : Vec mid => Tensor3.flatten (depthwiseConv2d Wd b' (Tensor3.unflatten er))) bd o j * cotDc j)
    -- depthwise BN γ/β  (cot = dyBnD, the depthwise relu6-masked cotangent)
  ∧ (∀ idx : Fin mid,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γd dc lr (.operand cotN dyBnD)) idx
          = γd idx - lr * ∑ j : Fin (mid*(h*w)),
              pdiv (fun γ' : Vec mid => bnPerChannelFlat mid (h*w) ε γ' βd (reassocFwd mid h w dc))
                   γd idx j * reassocFwd mid h w dyBnD j)
  ∧ (∀ idx : Fin mid,
        den (SHlo.bnBetaSgd bN lrStr βd lr (.operand cotN dyBnD)) idx
          = βd idx - lr * ∑ j : Fin (mid*(h*w)),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (h*w) ε γd β' (reassocFwd mid h w dc))
                   βd idx j * reassocFwd mid h w dyBnD j)
    -- project 1×1 conv (mid → oc)  (cot = cotPc)
  ∧ (∀ idx : Fin (oc*mid*1*1),
        den (SHlo.convWeightSgd xN wN lrStr bp (Tensor3.unflatten dr) Wp lr (.operand cotN cotPc)) idx
          = Kernel4.flatten Wp idx - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*mid*1*1) => Tensor3.flatten (conv2d (Kernel4.unflatten v') bp (Tensor3.unflatten dr)))
                   (Kernel4.flatten Wp) idx j * cotPc j)
  ∧ (∀ o : Fin oc,
        den (SHlo.convBiasSgd bN lrStr Wp (Tensor3.unflatten dr) bp lr (.operand cotN cotPc)) o
          = bp o - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d Wp b' (Tensor3.unflatten dr))) bp o j * cotPc j)
    -- project BN γ/β  (cot = dyOut directly — linear bottleneck, no relu6)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γp pc lr (.operand cotN dyOut)) idx
          = γp idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (h*w) ε γ' βp (reassocFwd oc h w pc))
                   γp idx j * reassocFwd oc h w dyOut j)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnBetaSgd bN lrStr βp lr (.operand cotN dyOut)) idx
          = βp idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (h*w) ε γp β' (reassocFwd oc h w pc))
                   βp idx j * reassocFwd oc h w dyOut j)

theorem mnv2_ivS1_tied {ic mid oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid 3 3) (bd : Vec mid) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp : Vec oc)
    (xin : Tensor3 ic h w) (ec en er dc dn dr : Vec (mid*h*w)) (pc : Vec (oc*h*w))
    (dyOut : Vec (oc*h*w)) (lr : ℝ) :
    ivS1Tied xN wN bN gN vN epsStr lrStr cotN ε We be γe βe Wd bd γd βd Wp bp γp βp
      xin ec en er dc dn dr pc dyOut lr := by
  unfold ivS1Tied
  intro cotPc cotDr dyBnD cotDc cotEr dyBnE cotEc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN be xin We cotEc lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN We xin be cotEc lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γe βe ec dyBnE lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γe βe ec dyBnE lr idx
  · intro idx; exact Mnv2PoC.depthwiseW_den xN wN lrStr cotN bd (Tensor3.unflatten er) Wd cotDc lr idx
  · intro o;   exact Mnv2PoC.depthwiseB_den bN lrStr cotN Wd (Tensor3.unflatten er) bd cotDc lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γd βd dc dyBnD lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γd βd dc dyBnD lr idx
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN bp (Tensor3.unflatten dr) Wp cotPc lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN Wp (Tensor3.unflatten dr) bp cotPc lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γp βp pc dyOut lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γp βp pc dyOut lr idx

/-! ## Stride-2 downsampling inverted-residual block — all 12 params tied

Same backward as stride-1 EXCEPT the depthwise is strided (`depthwiseStrided{Weight,Bias}Sgd`) so the
expand side (conv `We`, BN γe/βe) lives at the block-input grid `2h×2w`; the expand-output cotangent is
`invresCotEcS2` (the strided depthwise input-VJP zero-upsamples). No skip (spatial+channels change). -/

/-- **Stride-2 downsampling block, tied.** All 12 params at the real forward + the stride-2 chain
    cotangents (`invresCotEcS2` at the `2h×2w` expand grid). -/
def ivS2Tied {ic mid oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid 3 3) (bd : Vec mid) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp : Vec oc)
    (xin : Tensor3 ic (2*h) (2*w)) (ec en er : Vec (mid*(2*h)*(2*w)))
    (dc dn dr : Vec (mid*h*w)) (pc : Vec (oc*h*w))
    (dyOut : Vec (oc*h*w)) (lr : ℝ) : Prop :=
    let cotPc : Vec (oc*h*w) := invresCotPc ε γp pc dyOut
    let cotDr : Vec (mid*h*w) := (hasVJP3_to_hasVJP (conv2d_has_vjp3 Wp bp)).backward dr cotPc
    let dyBnD : Vec (mid*h*w) := fun i => if 0 < dn i ∧ dn i < 6 then cotDr i else 0
    let cotDc : Vec (mid*h*w) := invresCotDc ε γd γp Wp bp dr dn dc pc dyOut
    let cotEr : Vec (mid*(2*h)*(2*w)) := (depthwiseStride2Flat_has_vjp (h := h) (w := w) Wd bd).backward er cotDc
    let dyBnE : Vec (mid*(2*h)*(2*w)) := fun i => if 0 < en i ∧ en i < 6 then cotEr i else 0
    let cotEc : Vec (mid*(2*h)*(2*w)) := invresCotEcS2 ε γe γd γp Wd bd Wp bp er en ec dr dn dc pc dyOut
    -- expand 1×1 conv (ic → mid) at 2h×2w
    (∀ idx : Fin (mid*ic*1*1),
        den (SHlo.convWeightSgd xN wN lrStr be xin We lr (.operand cotN cotEc)) idx
          = Kernel4.flatten We idx - lr * ∑ j : Fin (mid*(2*h)*(2*w)),
              pdiv (fun v' : Vec (mid*ic*1*1) => Tensor3.flatten (conv2d (Kernel4.unflatten v') be xin))
                   (Kernel4.flatten We) idx j * cotEc j)
  ∧ (∀ o : Fin mid,
        den (SHlo.convBiasSgd bN lrStr We xin be lr (.operand cotN cotEc)) o
          = be o - lr * ∑ j : Fin (mid*(2*h)*(2*w)),
              pdiv (fun b' : Vec mid => Tensor3.flatten (conv2d We b' xin)) be o j * cotEc j)
    -- expand BN γ/β  (at 2h×2w)
  ∧ (∀ idx : Fin mid,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γe ec lr (.operand cotN dyBnE)) idx
          = γe idx - lr * ∑ j : Fin (mid*((2*h)*(2*w))),
              pdiv (fun γ' : Vec mid => bnPerChannelFlat mid ((2*h)*(2*w)) ε γ' βe (reassocFwd mid (2*h) (2*w) ec))
                   γe idx j * reassocFwd mid (2*h) (2*w) dyBnE j)
  ∧ (∀ idx : Fin mid,
        den (SHlo.bnBetaSgd bN lrStr βe lr (.operand cotN dyBnE)) idx
          = βe idx - lr * ∑ j : Fin (mid*((2*h)*(2*w))),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid ((2*h)*(2*w)) ε γe β' (reassocFwd mid (2*h) (2*w) ec))
                   βe idx j * reassocFwd mid (2*h) (2*w) dyBnE j)
    -- depthwise (STRIDED) W/b  (cot = cotDc at h×w)
  ∧ (∀ idx : Fin (mid*3*3),
        den (SHlo.depthwiseStridedWeightSgd xN wN lrStr bd er Wd lr (.operand cotN cotDc)) idx
          = Tensor3.flatten Wd idx - lr * ∑ j : Fin (mid*h*w),
              pdiv (fun v' : Vec (mid*3*3) => depthwiseStride2Flat (Tensor3.unflatten v') bd er)
                   (Tensor3.flatten Wd) idx j * cotDc j)
  ∧ (∀ o : Fin mid,
        den (SHlo.depthwiseStridedBiasSgd bN lrStr Wd er bd lr (.operand cotN cotDc)) o
          = bd o - lr * ∑ j : Fin (mid*h*w),
              pdiv (fun b' : Vec mid => depthwiseStride2Flat Wd b' er) bd o j * cotDc j)
    -- depthwise BN γ/β  (at h×w)
  ∧ (∀ idx : Fin mid,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γd dc lr (.operand cotN dyBnD)) idx
          = γd idx - lr * ∑ j : Fin (mid*(h*w)),
              pdiv (fun γ' : Vec mid => bnPerChannelFlat mid (h*w) ε γ' βd (reassocFwd mid h w dc))
                   γd idx j * reassocFwd mid h w dyBnD j)
  ∧ (∀ idx : Fin mid,
        den (SHlo.bnBetaSgd bN lrStr βd lr (.operand cotN dyBnD)) idx
          = βd idx - lr * ∑ j : Fin (mid*(h*w)),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (h*w) ε γd β' (reassocFwd mid h w dc))
                   βd idx j * reassocFwd mid h w dyBnD j)
    -- project 1×1 conv (mid → oc)  (cot = cotPc)
  ∧ (∀ idx : Fin (oc*mid*1*1),
        den (SHlo.convWeightSgd xN wN lrStr bp (Tensor3.unflatten dr) Wp lr (.operand cotN cotPc)) idx
          = Kernel4.flatten Wp idx - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*mid*1*1) => Tensor3.flatten (conv2d (Kernel4.unflatten v') bp (Tensor3.unflatten dr)))
                   (Kernel4.flatten Wp) idx j * cotPc j)
  ∧ (∀ o : Fin oc,
        den (SHlo.convBiasSgd bN lrStr Wp (Tensor3.unflatten dr) bp lr (.operand cotN cotPc)) o
          = bp o - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d Wp b' (Tensor3.unflatten dr))) bp o j * cotPc j)
    -- project BN γ/β  (cot = dyOut)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γp pc lr (.operand cotN dyOut)) idx
          = γp idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (h*w) ε γ' βp (reassocFwd oc h w pc))
                   γp idx j * reassocFwd oc h w dyOut j)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnBetaSgd bN lrStr βp lr (.operand cotN dyOut)) idx
          = βp idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (h*w) ε γp β' (reassocFwd oc h w pc))
                   βp idx j * reassocFwd oc h w dyOut j)

theorem mnv2_ivS2_tied {ic mid oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid 3 3) (bd : Vec mid) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp : Vec oc)
    (xin : Tensor3 ic (2*h) (2*w)) (ec en er : Vec (mid*(2*h)*(2*w)))
    (dc dn dr : Vec (mid*h*w)) (pc : Vec (oc*h*w))
    (dyOut : Vec (oc*h*w)) (lr : ℝ) :
    ivS2Tied xN wN bN gN vN epsStr lrStr cotN ε We be γe βe Wd bd γd βd Wp bp γp βp
      xin ec en er dc dn dr pc dyOut lr := by
  unfold ivS2Tied
  intro cotPc cotDr dyBnD cotDc cotEr dyBnE cotEc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN be xin We cotEc lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN We xin be cotEc lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γe βe ec dyBnE lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γe βe ec dyBnE lr idx
  · intro idx; exact Mnv2PoC.depthwiseStridedW_den xN wN lrStr cotN bd er Wd cotDc lr idx
  · intro o;   exact Mnv2PoC.depthwiseStridedB_den bN lrStr cotN Wd er bd cotDc lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γd βd dc dyBnD lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γd βd dc dyBnD lr idx
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN bp (Tensor3.unflatten dr) Wp cotPc lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN Wp (Tensor3.unflatten dr) bp cotPc lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γp βp pc dyOut lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γp βp pc dyOut lr idx

/-! ## No-expand block (b1, t=1) — all 8 params tied (depthwise(s1, on ic ch) → BN → relu6 → project → BN)

NO expand conv: the depthwise runs directly on the block input (`ic` channels). The "mid" role of
`invresCotDc` is played by `ic`. 8 params. -/

/-- **No-expand block, tied.** All 8 params (depthwise `W`+`b`, depthwise BN γ/β on `ic` channels,
    project 1×1 conv `W`+`b`, project BN γ/β) at the real forward + chain cotangents. -/
def ivNoExpTied {ic oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (Wd : DepthwiseKernel ic 3 3) (bd : Vec ic) (γd βd : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (γp βp : Vec oc)
    (xin : Tensor3 ic h w) (dc dn dr : Vec (ic*h*w)) (pc : Vec (oc*h*w))
    (dyOut : Vec (oc*h*w)) (lr : ℝ) : Prop :=
    let cotPc : Vec (oc*h*w) := invresCotPc ε γp pc dyOut
    let cotDr : Vec (ic*h*w) := (hasVJP3_to_hasVJP (conv2d_has_vjp3 Wp bp)).backward dr cotPc
    let dyBnD : Vec (ic*h*w) := fun i => if 0 < dn i ∧ dn i < 6 then cotDr i else 0
    let cotDc : Vec (ic*h*w) := invresCotDc ε γd γp Wp bp dr dn dc pc dyOut
    -- depthwise (stride-1) W/b on the block input  (cot = cotDc)
    (∀ idx : Fin (ic*3*3),
        den (SHlo.depthwiseWeightSgd xN wN lrStr bd xin Wd lr (.operand cotN cotDc)) idx
          = Tensor3.flatten Wd idx - lr * ∑ j : Fin (ic*h*w),
              pdiv (fun v' : Vec (ic*3*3) => Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') bd xin))
                   (Tensor3.flatten Wd) idx j * cotDc j)
  ∧ (∀ o : Fin ic,
        den (SHlo.depthwiseBiasSgd bN lrStr Wd xin bd lr (.operand cotN cotDc)) o
          = bd o - lr * ∑ j : Fin (ic*h*w),
              pdiv (fun b' : Vec ic => Tensor3.flatten (depthwiseConv2d Wd b' xin)) bd o j * cotDc j)
    -- depthwise BN γ/β  (on ic, cot = dyBnD)
  ∧ (∀ idx : Fin ic,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γd dc lr (.operand cotN dyBnD)) idx
          = γd idx - lr * ∑ j : Fin (ic*(h*w)),
              pdiv (fun γ' : Vec ic => bnPerChannelFlat ic (h*w) ε γ' βd (reassocFwd ic h w dc))
                   γd idx j * reassocFwd ic h w dyBnD j)
  ∧ (∀ idx : Fin ic,
        den (SHlo.bnBetaSgd bN lrStr βd lr (.operand cotN dyBnD)) idx
          = βd idx - lr * ∑ j : Fin (ic*(h*w)),
              pdiv (fun β' : Vec ic => bnPerChannelFlat ic (h*w) ε γd β' (reassocFwd ic h w dc))
                   βd idx j * reassocFwd ic h w dyBnD j)
    -- project 1×1 conv (ic → oc)  (cot = cotPc)
  ∧ (∀ idx : Fin (oc*ic*1*1),
        den (SHlo.convWeightSgd xN wN lrStr bp (Tensor3.unflatten dr) Wp lr (.operand cotN cotPc)) idx
          = Kernel4.flatten Wp idx - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*ic*1*1) => Tensor3.flatten (conv2d (Kernel4.unflatten v') bp (Tensor3.unflatten dr)))
                   (Kernel4.flatten Wp) idx j * cotPc j)
  ∧ (∀ o : Fin oc,
        den (SHlo.convBiasSgd bN lrStr Wp (Tensor3.unflatten dr) bp lr (.operand cotN cotPc)) o
          = bp o - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d Wp b' (Tensor3.unflatten dr))) bp o j * cotPc j)
    -- project BN γ/β  (cot = dyOut)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γp pc lr (.operand cotN dyOut)) idx
          = γp idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (h*w) ε γ' βp (reassocFwd oc h w pc))
                   γp idx j * reassocFwd oc h w dyOut j)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnBetaSgd bN lrStr βp lr (.operand cotN dyOut)) idx
          = βp idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (h*w) ε γp β' (reassocFwd oc h w pc))
                   βp idx j * reassocFwd oc h w dyOut j)

theorem mnv2_ivNoExp_tied {ic oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (Wd : DepthwiseKernel ic 3 3) (bd : Vec ic) (γd βd : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (γp βp : Vec oc)
    (xin : Tensor3 ic h w) (dc dn dr : Vec (ic*h*w)) (pc : Vec (oc*h*w))
    (dyOut : Vec (oc*h*w)) (lr : ℝ) :
    ivNoExpTied xN wN bN gN vN epsStr lrStr cotN ε Wd bd γd βd Wp bp γp βp
      xin dc dn dr pc dyOut lr := by
  unfold ivNoExpTied
  intro cotPc cotDr dyBnD cotDc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact Mnv2PoC.depthwiseW_den xN wN lrStr cotN bd xin Wd cotDc lr idx
  · intro o;   exact Mnv2PoC.depthwiseB_den bN lrStr cotN Wd xin bd cotDc lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γd βd dc dyBnD lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γd βd dc dyBnD lr idx
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN bp (Tensor3.unflatten dr) Wp cotPc lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN Wp (Tensor3.unflatten dr) bp cotPc lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γp βp pc dyOut lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γp βp pc dyOut lr idx

/-! ## Stem — the 3×3/s2 conv + BN (4 params), NO maxpool (unlike r34)

Forward: `relu6(bn(convˢ x))`, feeding block 1. The cotangent block 1 delivers at the stem relu6
output (`dyStem`) lifts through the relu6 two-kink mask + BN-back to `mnv2StemCot` at the conv output. -/

/-- **Stem, tied.** The 3×3/s2 conv (`Ws`/`bs`) + its BN (`γs`/`βs`) at the real stem forward + the
    cotangent through the stem relu6 (no maxpool). -/
def mnv2StemTied {ic oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (Ws : Kernel4 oc ic 3 3) (bs : Vec oc) (γs βs : Vec oc)
    (x : Vec (ic*(2*h)*(2*w))) (stn stc dyStem : Vec (oc*h*w)) (lr : ℝ) : Prop :=
    let dyBnS : Vec (oc*h*w) := fun i => if 0 < stn i ∧ stn i < 6 then dyStem i else 0
    let cotStem : Vec (oc*h*w) := mnv2StemCot ε γs stn stc dyStem
    (∀ idx : Fin (oc*ic*3*3),
        den (SHlo.convStridedWeightSgd xN wN lrStr bs x Ws lr (.operand cotN cotStem)) idx
          = Kernel4.flatten Ws idx - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*ic*3*3) => flatConvStride2 (Kernel4.unflatten v') bs x)
                   (Kernel4.flatten Ws) idx j * cotStem j)
  ∧ (∀ o : Fin oc,
        den (SHlo.convStridedBiasSgd bN lrStr Ws x bs lr (.operand cotN cotStem)) o
          = bs o - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun b' : Vec oc => flatConvStride2 Ws b' x) bs o j * cotStem j)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γs stc lr (.operand cotN dyBnS)) idx
          = γs idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (h*w) ε γ' βs (reassocFwd oc h w stc))
                   γs idx j * reassocFwd oc h w dyBnS j)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnBetaSgd bN lrStr βs lr (.operand cotN dyBnS)) idx
          = βs idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (h*w) ε γs β' (reassocFwd oc h w stc))
                   βs idx j * reassocFwd oc h w dyBnS j)

theorem mnv2_stem_tied {ic oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (Ws : Kernel4 oc ic 3 3) (bs : Vec oc) (γs βs : Vec oc)
    (x : Vec (ic*(2*h)*(2*w))) (stn stc dyStem : Vec (oc*h*w)) (lr : ℝ) :
    mnv2StemTied xN wN bN gN vN epsStr lrStr cotN ε Ws bs γs βs x stn stc dyStem lr := by
  unfold mnv2StemTied
  intro dyBnS cotStem
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro idx; exact ResNet34PoC.convStridedW_den xN wN lrStr cotN bs x Ws cotStem lr idx
  · intro o;   exact ResNet34PoC.convStridedB_den bN lrStr cotN Ws x bs cotStem lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γs βs stc dyBnS lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γs βs stc dyBnS lr idx

/-! ## Head — the 1×1 conv-bn-relu6 (4 params) feeding GAP → dense

Structurally a single-conv block: input `x_b17`, cotangent `dyHr` at its relu6 output (which the
GAP→dense backward delivers). 1×1 conv W/b consume the BN-back of the relu6-masked `dyHr`; BN γ/β
consume the relu6-masked `dyHr`. -/

/-- **Head conv-bn-relu6, tied.** The 4 head params (`Wh`/`bh` 1×1 conv, `γh`/`βh` BN) at the real
    head forward + the cotangent `dyHr` the GAP/dense backward delivers at the head relu6 output. -/
def mnv2HeadTied {ic oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (γh βh : Vec oc)
    (xin : Tensor3 ic h w) (hc hn dyHr : Vec (oc*h*w)) (lr : ℝ) : Prop :=
    let dyHn : Vec (oc*h*w) := fun i => if 0 < hn i ∧ hn i < 6 then dyHr i else 0
    let cotHc : Vec (oc*h*w) := bnPerChannelTensor3_grad_input oc h w ε γh hc dyHn
    (∀ idx : Fin (oc*ic*1*1),
        den (SHlo.convWeightSgd xN wN lrStr bh xin Wh lr (.operand cotN cotHc)) idx
          = Kernel4.flatten Wh idx - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*ic*1*1) => Tensor3.flatten (conv2d (Kernel4.unflatten v') bh xin))
                   (Kernel4.flatten Wh) idx j * cotHc j)
  ∧ (∀ o : Fin oc,
        den (SHlo.convBiasSgd bN lrStr Wh xin bh lr (.operand cotN cotHc)) o
          = bh o - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d Wh b' xin)) bh o j * cotHc j)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γh hc lr (.operand cotN dyHn)) idx
          = γh idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (h*w) ε γ' βh (reassocFwd oc h w hc))
                   γh idx j * reassocFwd oc h w dyHn j)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnBetaSgd bN lrStr βh lr (.operand cotN dyHn)) idx
          = βh idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (h*w) ε γh β' (reassocFwd oc h w hc))
                   βh idx j * reassocFwd oc h w dyHn j)

theorem mnv2_head_tied {ic oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (γh βh : Vec oc)
    (xin : Tensor3 ic h w) (hc hn dyHr : Vec (oc*h*w)) (lr : ℝ) :
    mnv2HeadTied xN wN bN gN vN epsStr lrStr cotN ε Wh bh γh βh xin hc hn dyHr lr := by
  unfold mnv2HeadTied
  intro dyHn cotHc
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN bh xin Wh cotHc lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN Wh xin bh cotHc lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γh βh hc dyHn lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γh βh hc dyHn lr idx

/-! ## Loss cotangent + dense head (pin the top of the chain) -/

/-- **The emitted loss-cotangent graph denotes the softmax-CE gradient at the logits.** -/
theorem mnv2LossCot_den (nlogN ohN : String) (logits : Vec 10) (label : Fin 10) :
    den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe (.operand nlogN logits)))
          (.operand ohN (oneHot 10 label)))
      = fun j => softmax 10 logits j - oneHot 10 label j := by
  funext j; simp only [den, softmax]

/-- **Dense head weight `Wfc`, tied to the WHOLE softmax-CE loss** — `Wfc − lr·∂(CE ∘ dense)/∂Wfc`. -/
theorem mnv2_dense_tied_totalloss (aN lrStr dyN : String)
    (Wfc : Mat 1280 10) (bfc : Vec 10) (a_gap : Vec 1280) (label : Fin 10)
    (lr : ℝ) (i : Fin 1280) (j : Fin 10) :
    den (SHlo.weightSgd aN "%Wfc" lrStr a_gap Wfc lr
          (.operand dyN (fun k => softmax 10 (mnistLinear Wfc bfc a_gap) k - oneHot 10 label k)))
        (finProdFinEquiv (i, j))
      = Wfc i j - lr * pdiv (fun v : Vec (1280 * 10) => fun _ : Fin 1 =>
            crossEntropy 10 (dense (Mat.unflatten v) bfc a_gap) label)
          (Mat.flatten Wfc) (finProdFinEquiv (i, j)) 0 := by
  rw [Cifar8PoC.denseW_den aN "%Wfc" lrStr dyN a_gap Wfc bfc
        (fun k => softmax 10 (mnistLinear Wfc bfc a_gap) k - oneHot 10 label k) lr i j,
      mlp_output_total_loss_grad Wfc bfc a_gap label i j]

/-- **Dense head bias `bfc` = certified step.** -/
theorem mnv2_dense_bias_den (bN lrStr dyN : String)
    (Wfc : Mat 1280 10) (bfc : Vec 10) (a_gap : Vec 1280) (c : Vec 10) (lr : ℝ) (i : Fin 10) :
    den (SHlo.biasSgd bN lrStr bfc lr (.operand dyN c)) i
      = bfc i - lr * ∑ j : Fin 10, pdiv (fun b' : Vec 10 => dense Wfc b' a_gap) bfc i j * c j :=
  Cifar8PoC.denseB_den bN lrStr dyN Wfc a_gap bfc c lr i

/-! ## Input-only `TiedAt` wrappers — compute a block's internal activations from its input

For the whole-net thread each block is identified by its **input** + its downstream cotangent `dyOut`.
These wrappers compute the block's real forward activations (`ec`/`en`/`er`/… via the per-channel-BN
building blocks the forward uses) and delegate to the per-block-type tie — so the capstone threads only
block inputs + dyOuts. `@[irreducible]` keeps the 17-deep chain opaque to the elaborator (the r34
heartbeat lesson, more acute here). -/

/-- Stride-1 expand block tie, INPUT-only (covers skip AND no-skip widenings — identical param ops). -/
@[irreducible] def ivS1TiedAt {ic mid oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid) (Wd : DepthwiseKernel mid 3 3) (bd γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (ic*h*w)) (dyOut : Vec (oc*h*w)) (lr : ℝ) : Prop :=
  let ec := flatConv We be xin
  let en := bnPerChannelTensor3 mid h w ε γe βe ec
  let er := relu6 (mid*h*w) en
  let dc := depthwiseFlat Wd bd er
  let dn := bnPerChannelTensor3 mid h w ε γd βd dc
  let dr := relu6 (mid*h*w) dn
  let pc := flatConv Wp bp dr
  ivS1Tied xN wN bN gN vN epsStr lrStr cotN ε We be γe βe Wd bd γd βd Wp bp γp βp
    (Tensor3.unflatten xin) ec en er dc dn dr pc dyOut lr

theorem mnv2_ivS1_tiedAt {ic mid oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid) (Wd : DepthwiseKernel mid 3 3) (bd γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (ic*h*w)) (dyOut : Vec (oc*h*w)) (lr : ℝ) :
    ivS1TiedAt xN wN bN gN vN epsStr lrStr cotN ε We be γe βe Wd bd γd βd Wp bp γp βp xin dyOut lr := by
  unfold ivS1TiedAt
  intro ec en er dc dn dr pc
  exact mnv2_ivS1_tied xN wN bN gN vN epsStr lrStr cotN ε We be γe βe Wd bd γd βd Wp bp γp βp
    (Tensor3.unflatten xin) ec en er dc dn dr pc dyOut lr

/-- Stride-2 downsampling block tie, INPUT-only (block input at `2h×2w`). -/
@[irreducible] def ivS2TiedAt {ic mid oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid) (Wd : DepthwiseKernel mid 3 3) (bd γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (ic*(2*h)*(2*w))) (dyOut : Vec (oc*h*w)) (lr : ℝ) : Prop :=
  let ec := flatConv We be xin
  let en := bnPerChannelTensor3 mid (2*h) (2*w) ε γe βe ec
  let er := relu6 (mid*(2*h)*(2*w)) en
  let dc := depthwiseStride2Flat Wd bd er
  let dn := bnPerChannelTensor3 mid h w ε γd βd dc
  let dr := relu6 (mid*h*w) dn
  let pc := flatConv Wp bp dr
  ivS2Tied xN wN bN gN vN epsStr lrStr cotN ε We be γe βe Wd bd γd βd Wp bp γp βp
    (Tensor3.unflatten xin) ec en er dc dn dr pc dyOut lr

theorem mnv2_ivS2_tiedAt {ic mid oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid) (Wd : DepthwiseKernel mid 3 3) (bd γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (ic*(2*h)*(2*w))) (dyOut : Vec (oc*h*w)) (lr : ℝ) :
    ivS2TiedAt xN wN bN gN vN epsStr lrStr cotN ε We be γe βe Wd bd γd βd Wp bp γp βp xin dyOut lr := by
  unfold ivS2TiedAt
  intro ec en er dc dn dr pc
  exact mnv2_ivS2_tied xN wN bN gN vN epsStr lrStr cotN ε We be γe βe Wd bd γd βd Wp bp γp βp
    (Tensor3.unflatten xin) ec en er dc dn dr pc dyOut lr

/-- No-expand block tie, INPUT-only (depthwise runs on the block input). -/
@[irreducible] def ivNoExpTiedAt {ic oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (Wd : DepthwiseKernel ic 3 3) (bd γd βd : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp γp βp : Vec oc)
    (xin : Vec (ic*h*w)) (dyOut : Vec (oc*h*w)) (lr : ℝ) : Prop :=
  let dc := depthwiseFlat Wd bd xin
  let dn := bnPerChannelTensor3 ic h w ε γd βd dc
  let dr := relu6 (ic*h*w) dn
  let pc := flatConv Wp bp dr
  ivNoExpTied xN wN bN gN vN epsStr lrStr cotN ε Wd bd γd βd Wp bp γp βp
    (Tensor3.unflatten xin) dc dn dr pc dyOut lr

theorem mnv2_ivNoExp_tiedAt {ic oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (Wd : DepthwiseKernel ic 3 3) (bd γd βd : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp γp βp : Vec oc)
    (xin : Vec (ic*h*w)) (dyOut : Vec (oc*h*w)) (lr : ℝ) :
    ivNoExpTiedAt xN wN bN gN vN epsStr lrStr cotN ε Wd bd γd βd Wp bp γp βp xin dyOut lr := by
  unfold ivNoExpTiedAt
  intro dc dn dr pc
  exact mnv2_ivNoExp_tied xN wN bN gN vN epsStr lrStr cotN ε Wd bd γd βd Wp bp γp βp
    (Tensor3.unflatten xin) dc dn dr pc dyOut lr

/-- Stem tie, INPUT-only. -/
@[irreducible] def mnv2StemTiedAt {ic oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (Ws : Kernel4 oc ic 3 3) (bs γs βs : Vec oc)
    (x : Vec (ic*(2*h)*(2*w))) (dyStem : Vec (oc*h*w)) (lr : ℝ) : Prop :=
  let stc := flatConvStride2 Ws bs x
  let stn := bnPerChannelTensor3 oc h w ε γs βs stc
  mnv2StemTied xN wN bN gN vN epsStr lrStr cotN ε Ws bs γs βs x stn stc dyStem lr

theorem mnv2_stem_tiedAt {ic oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (Ws : Kernel4 oc ic 3 3) (bs γs βs : Vec oc)
    (x : Vec (ic*(2*h)*(2*w))) (dyStem : Vec (oc*h*w)) (lr : ℝ) :
    mnv2StemTiedAt xN wN bN gN vN epsStr lrStr cotN ε Ws bs γs βs x dyStem lr := by
  unfold mnv2StemTiedAt
  intro stc stn
  exact mnv2_stem_tied xN wN bN gN vN epsStr lrStr cotN ε Ws bs γs βs x stn stc dyStem lr

/-- Head conv-bn-relu6 tie, INPUT-only. -/
@[irreducible] def mnv2HeadTiedAt {ic oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (Wh : Kernel4 oc ic 1 1) (bh γh βh : Vec oc)
    (xin : Vec (ic*h*w)) (dyHr : Vec (oc*h*w)) (lr : ℝ) : Prop :=
  let hc := flatConv Wh bh xin
  let hn := bnPerChannelTensor3 oc h w ε γh βh hc
  mnv2HeadTied xN wN bN gN vN epsStr lrStr cotN ε Wh bh γh βh (Tensor3.unflatten xin) hc hn dyHr lr

theorem mnv2_head_tiedAt {ic oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (Wh : Kernel4 oc ic 1 1) (bh γh βh : Vec oc)
    (xin : Vec (ic*h*w)) (dyHr : Vec (oc*h*w)) (lr : ℝ) :
    mnv2HeadTiedAt xN wN bN gN vN epsStr lrStr cotN ε Wh bh γh βh xin dyHr lr := by
  unfold mnv2HeadTiedAt
  intro hc hn
  exact mnv2_head_tied xN wN bN gN vN epsStr lrStr cotN ε Wh bh γh βh (Tensor3.unflatten xin) hc hn dyHr lr

/-! ## Forward block aliases (`@[irreducible]`) — thread block INPUTS through the real forward

Each is the per-channel-BN block-output function with a single shared ε (= what the renderer emits and
what `mobilenetv2ForwardPaper` computes when all per-layer ε coincide). Irreducible so the 17-deep
nested composition is opaque during the capstone's dimension inference. -/

@[irreducible] noncomputable def stemFwdO {ic oc h w : Nat} (ε : ℝ)
    (Ws : Kernel4 oc ic 3 3) (bs γs βs : Vec oc) (x : Vec (ic*(2*h)*(2*w))) : Vec (oc*h*w) :=
  relu6 (oc*h*w) (bnPerChannelTensor3 oc h w ε γs βs (flatConvStride2 Ws bs x))

/-- Stride-1 expand block BODY output (no skip). -/
@[irreducible] noncomputable def ivS1BodyO {ic mid oc h w : Nat} (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid) (Wd : DepthwiseKernel mid 3 3) (bd γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc) (xin : Vec (ic*h*w)) : Vec (oc*h*w) :=
  bnPerChannelTensor3 oc h w ε γp βp (flatConv Wp bp
    (relu6 (mid*h*w) (bnPerChannelTensor3 mid h w ε γd βd (depthwiseFlat Wd bd
      (relu6 (mid*h*w) (bnPerChannelTensor3 mid h w ε γe βe (flatConv We be xin)))))))

/-- Stride-1 block WITH identity skip (`ic = oc`). -/
@[irreducible] noncomputable def ivS1SkipFwdO {c mid h w : Nat} (ε : ℝ)
    (We : Kernel4 mid c 1 1) (be γe βe : Vec mid) (Wd : DepthwiseKernel mid 3 3) (bd γd βd : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp γp βp : Vec c) (xin : Vec (c*h*w)) : Vec (c*h*w) :=
  fun i => ivS1BodyO ε We be γe βe Wd bd γd βd Wp bp γp βp xin i + xin i

@[irreducible] noncomputable def ivNoExpFwdO {ic oc h w : Nat} (ε : ℝ)
    (Wd : DepthwiseKernel ic 3 3) (bd γd βd : Vec ic) (Wp : Kernel4 oc ic 1 1) (bp γp βp : Vec oc)
    (xin : Vec (ic*h*w)) : Vec (oc*h*w) :=
  bnPerChannelTensor3 oc h w ε γp βp (flatConv Wp bp
    (relu6 (ic*h*w) (bnPerChannelTensor3 ic h w ε γd βd (depthwiseFlat Wd bd xin))))

@[irreducible] noncomputable def ivS2FwdO {ic mid oc h w : Nat} (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid) (Wd : DepthwiseKernel mid 3 3) (bd γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc) (xin : Vec (ic*(2*h)*(2*w))) : Vec (oc*h*w) :=
  bnPerChannelTensor3 oc h w ε γp βp (flatConv Wp bp
    (relu6 (mid*h*w) (bnPerChannelTensor3 mid h w ε γd βd (depthwiseStride2Flat Wd bd
      (relu6 (mid*(2*h)*(2*w)) (bnPerChannelTensor3 mid (2*h) (2*w) ε γe βe (flatConv We be xin)))))))

/-! ## Backward cot-in constructors (`@[irreducible]`) — thread block dyOuts (the residual fan-in)

Each gives the cotangent the backward delivers at a block's *input* (= the previous block's `dyOut`):
the body branch (expand/depthwise input-VJP of the deepest in-block chain cotangent) plus, for the
stride-1 skip blocks, the identity-skip branch (`+ dyOut`). -/

/-- Stride-1 skip block input cotangent: `expand-conv-back(invresCotEcS1) + dyOut` (the fan-in sum). -/
@[irreducible] noncomputable def ivS1SkipCotInAt {c mid h w : Nat} (ε : ℝ)
    (We : Kernel4 mid c 1 1) (be γe βe : Vec mid) (Wd : DepthwiseKernel mid 3 3) (bd γd βd : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp γp _βp : Vec c) (xin dyOut : Vec (c*h*w)) : Vec (c*h*w) :=
  let ec := flatConv We be xin
  let en := bnPerChannelTensor3 mid h w ε γe βe ec
  let er := relu6 (mid*h*w) en
  let dc := depthwiseFlat Wd bd er
  let dn := bnPerChannelTensor3 mid h w ε γd βd dc
  let dr := relu6 (mid*h*w) dn
  let pc := flatConv Wp bp dr
  let cotEc := invresCotEcS1 ε γe γd γp Wd bd Wp bp er en ec dr dn dc pc dyOut
  fun i => (hasVJP3_to_hasVJP (conv2d_has_vjp3 We be)).backward xin cotEc i + dyOut i

/-- Stride-1 no-skip (widening b11/b17) block input cotangent: body branch only. -/
@[irreducible] noncomputable def ivS1NoSkipCotInAt {ic mid oc h w : Nat} (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid) (Wd : DepthwiseKernel mid 3 3) (bd γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp _βp : Vec oc) (xin : Vec (ic*h*w)) (dyOut : Vec (oc*h*w)) :
    Vec (ic*h*w) :=
  let ec := flatConv We be xin
  let en := bnPerChannelTensor3 mid h w ε γe βe ec
  let er := relu6 (mid*h*w) en
  let dc := depthwiseFlat Wd bd er
  let dn := bnPerChannelTensor3 mid h w ε γd βd dc
  let dr := relu6 (mid*h*w) dn
  let pc := flatConv Wp bp dr
  let cotEc := invresCotEcS1 ε γe γd γp Wd bd Wp bp er en ec dr dn dc pc dyOut
  (hasVJP3_to_hasVJP (conv2d_has_vjp3 We be)).backward xin cotEc

/-- Stride-2 downsampling block input cotangent (at `2h×2w`): body branch only. -/
@[irreducible] noncomputable def ivS2CotInAt {ic mid oc h w : Nat} (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid) (Wd : DepthwiseKernel mid 3 3) (bd γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp _βp : Vec oc) (xin : Vec (ic*(2*h)*(2*w))) (dyOut : Vec (oc*h*w)) :
    Vec (ic*(2*h)*(2*w)) :=
  let ec := flatConv We be xin
  let en := bnPerChannelTensor3 mid (2*h) (2*w) ε γe βe ec
  let er := relu6 (mid*(2*h)*(2*w)) en
  let dc := depthwiseStride2Flat Wd bd er
  let dn := bnPerChannelTensor3 mid h w ε γd βd dc
  let dr := relu6 (mid*h*w) dn
  let pc := flatConv Wp bp dr
  let cotEc := invresCotEcS2 ε γe γd γp Wd bd Wp bp er en ec dr dn dc pc dyOut
  (hasVJP3_to_hasVJP (conv2d_has_vjp3 We be)).backward xin cotEc

/-- No-expand block input cotangent: depthwise input-VJP of `invresCotDc`. -/
@[irreducible] noncomputable def ivNoExpCotInAt {ic oc h w : Nat} (ε : ℝ)
    (Wd : DepthwiseKernel ic 3 3) (bd γd βd : Vec ic) (Wp : Kernel4 oc ic 1 1) (bp γp _βp : Vec oc)
    (xin : Vec (ic*h*w)) (dyOut : Vec (oc*h*w)) : Vec (ic*h*w) :=
  let dc := depthwiseFlat Wd bd xin
  let dn := bnPerChannelTensor3 ic h w ε γd βd dc
  let dr := relu6 (ic*h*w) dn
  let pc := flatConv Wp bp dr
  let cotDc := invresCotDc ε γd γp Wp bp dr dn dc pc dyOut
  (depthwiseFlat_has_vjp (h := h) (w := w) Wd bd).backward xin cotDc

/-- The cotangent at the head relu6 output (= what `mnv2HeadTiedAt` consumes): `gap-back(dense-back(g))`. -/
@[irreducible] noncomputable def headDyHr {ic oc h w : Nat} (ε : ℝ)
    (Wh : Kernel4 oc ic 1 1) (bh γh βh : Vec oc) (Wfc : Mat oc 10) (bfc : Vec 10)
    (xin : Vec (ic*h*w)) (g : Vec 10) : Vec (oc*h*w) :=
  let hc := flatConv Wh bh xin
  let hn := bnPerChannelTensor3 oc h w ε γh βh hc
  let hr := relu6 (oc*h*w) hn
  let a_gap := globalAvgPoolFlat oc h w hr
  (globalAvgPoolFlat_has_vjp oc h w).backward hr ((dense_has_vjp Wfc bfc).backward a_gap g)

/-- The block-input cotangent the head delivers to b17 (= b17's `dyOut`): head-conv input-VJP of the
    BN-back of the relu6-masked `headDyHr`. -/
@[irreducible] noncomputable def headCotIn {ic oc h w : Nat} (ε : ℝ)
    (Wh : Kernel4 oc ic 1 1) (bh γh βh : Vec oc) (Wfc : Mat oc 10) (bfc : Vec 10)
    (xin : Vec (ic*h*w)) (g : Vec 10) : Vec (ic*h*w) :=
  let hc := flatConv Wh bh xin
  let hn := bnPerChannelTensor3 oc h w ε γh βh hc
  let dyHr := headDyHr ε Wh bh γh βh Wfc bfc xin g
  let dyHn : Vec (oc*h*w) := fun i => if 0 < hn i ∧ hn i < 6 then dyHr i else 0
  let cotHc := bnPerChannelTensor3_grad_input oc h w ε γh hc dyHn
  (hasVJP3_to_hasVJP (conv2d_has_vjp3 Wh bh)).backward xin cotHc

/-! ## The whole-net capstone — all 210 params tied through the REAL forward + composed cotangent

`mobilenetv2ForwardPaper` (single-ε) threaded: block inputs are the forward prefixes (`stemFwdO` /
`ivNoExpFwdO` / `ivS2FwdO` / `ivS1SkipFwdO` / `ivS1BodyO`), and the backward cotangents are composed
from the loss `g = softmax(logits) − onehot` down through dense (`dense_has_vjp`) + GAP
(`globalAvgPoolFlat_has_vjp`) + the conv-bn-relu6 head + every block's backward, with the residual
fan-in `+ dyOut` at each of the ten stride-1 skip merges. Each block's tie then holds at its real input
+ threaded dyOut. The full §1a tie: the WHOLE 17-block (210-param) MobileNetV2 train step is
den-composed forward→loss→backward, no free activations, no symbolic cotangent. -/

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 200000 in
/-- **The whole 17-block paper MobileNetV2 train step, tied.** Threading the real per-channel-BN
    forward and the loss-driven backward cotangent chain (relu6 two-kink masks, the residual fan-in at
    every stride-1 skip), the stem, all 17 inverted-residual blocks, the conv-bn-relu6 head, and the
    dense head (total-loss fold + loss-cotangent graph) denote the certified loss-descent step. -/
theorem mnv2_net_tied_certified
    (xN wN bN gN vN epsStr lrStr cotN dN nlogN ohN : String) (ε : ℝ)
    (sW : Kernel4 32 3 3 3) (sb sg st : Vec 32)
    (d1W : DepthwiseKernel 32 3 3) (d1b d1g d1t : Vec 32) (p1W : Kernel4 16 32 1 1) (p1b p1g p1t : Vec 16)
    (e2W : Kernel4 96 16 1 1) (e2b e2g e2t : Vec 96) (d2W : DepthwiseKernel 96 3 3) (d2b d2g d2t : Vec 96) (p2W : Kernel4 24 96 1 1) (p2b p2g p2t : Vec 24)
    (e3W : Kernel4 144 24 1 1) (e3b e3g e3t : Vec 144) (d3W : DepthwiseKernel 144 3 3) (d3b d3g d3t : Vec 144) (p3W : Kernel4 24 144 1 1) (p3b p3g p3t : Vec 24)
    (e4W : Kernel4 144 24 1 1) (e4b e4g e4t : Vec 144) (d4W : DepthwiseKernel 144 3 3) (d4b d4g d4t : Vec 144) (p4W : Kernel4 32 144 1 1) (p4b p4g p4t : Vec 32)
    (e5W : Kernel4 192 32 1 1) (e5b e5g e5t : Vec 192) (d5W : DepthwiseKernel 192 3 3) (d5b d5g d5t : Vec 192) (p5W : Kernel4 32 192 1 1) (p5b p5g p5t : Vec 32)
    (e6W : Kernel4 192 32 1 1) (e6b e6g e6t : Vec 192) (d6W : DepthwiseKernel 192 3 3) (d6b d6g d6t : Vec 192) (p6W : Kernel4 32 192 1 1) (p6b p6g p6t : Vec 32)
    (e7W : Kernel4 192 32 1 1) (e7b e7g e7t : Vec 192) (d7W : DepthwiseKernel 192 3 3) (d7b d7g d7t : Vec 192) (p7W : Kernel4 64 192 1 1) (p7b p7g p7t : Vec 64)
    (e8W : Kernel4 384 64 1 1) (e8b e8g e8t : Vec 384) (d8W : DepthwiseKernel 384 3 3) (d8b d8g d8t : Vec 384) (p8W : Kernel4 64 384 1 1) (p8b p8g p8t : Vec 64)
    (e9W : Kernel4 384 64 1 1) (e9b e9g e9t : Vec 384) (d9W : DepthwiseKernel 384 3 3) (d9b d9g d9t : Vec 384) (p9W : Kernel4 64 384 1 1) (p9b p9g p9t : Vec 64)
    (e10W : Kernel4 384 64 1 1) (e10b e10g e10t : Vec 384) (d10W : DepthwiseKernel 384 3 3) (d10b d10g d10t : Vec 384) (p10W : Kernel4 64 384 1 1) (p10b p10g p10t : Vec 64)
    (e11W : Kernel4 384 64 1 1) (e11b e11g e11t : Vec 384) (d11W : DepthwiseKernel 384 3 3) (d11b d11g d11t : Vec 384) (p11W : Kernel4 96 384 1 1) (p11b p11g p11t : Vec 96)
    (e12W : Kernel4 576 96 1 1) (e12b e12g e12t : Vec 576) (d12W : DepthwiseKernel 576 3 3) (d12b d12g d12t : Vec 576) (p12W : Kernel4 96 576 1 1) (p12b p12g p12t : Vec 96)
    (e13W : Kernel4 576 96 1 1) (e13b e13g e13t : Vec 576) (d13W : DepthwiseKernel 576 3 3) (d13b d13g d13t : Vec 576) (p13W : Kernel4 96 576 1 1) (p13b p13g p13t : Vec 96)
    (e14W : Kernel4 576 96 1 1) (e14b e14g e14t : Vec 576) (d14W : DepthwiseKernel 576 3 3) (d14b d14g d14t : Vec 576) (p14W : Kernel4 160 576 1 1) (p14b p14g p14t : Vec 160)
    (e15W : Kernel4 960 160 1 1) (e15b e15g e15t : Vec 960) (d15W : DepthwiseKernel 960 3 3) (d15b d15g d15t : Vec 960) (p15W : Kernel4 160 960 1 1) (p15b p15g p15t : Vec 160)
    (e16W : Kernel4 960 160 1 1) (e16b e16g e16t : Vec 960) (d16W : DepthwiseKernel 960 3 3) (d16b d16g d16t : Vec 960) (p16W : Kernel4 160 960 1 1) (p16b p16g p16t : Vec 160)
    (e17W : Kernel4 960 160 1 1) (e17b e17g e17t : Vec 960) (d17W : DepthwiseKernel 960 3 3) (d17b d17g d17t : Vec 960) (p17W : Kernel4 320 960 1 1) (p17b p17g p17t : Vec 320)
    (hW : Kernel4 1280 320 1 1) (hb hg ht : Vec 1280) (Wfc : Mat 1280 10) (bfc : Vec 10)
    (x : Vec (3*224*224)) (label : Fin 10) (lr : ℝ) :
    -- forward block inputs (the prefixes of `mobilenetv2ForwardPaper`, single-ε)
    let ib1  : Vec (32*112*112)  := stemFwdO ε sW sb sg st x
    let ib2  : Vec (16*112*112)  := ivNoExpFwdO ε d1W d1b d1g d1t p1W p1b p1g p1t ib1
    let ib3  : Vec (24*56*56)    := ivS2FwdO ε e2W e2b e2g e2t d2W d2b d2g d2t p2W p2b p2g p2t ib2
    let ib4  : Vec (24*56*56)    := ivS1SkipFwdO ε e3W e3b e3g e3t d3W d3b d3g d3t p3W p3b p3g p3t ib3
    let ib5  : Vec (32*28*28)    := ivS2FwdO ε e4W e4b e4g e4t d4W d4b d4g d4t p4W p4b p4g p4t ib4
    let ib6  : Vec (32*28*28)    := ivS1SkipFwdO ε e5W e5b e5g e5t d5W d5b d5g d5t p5W p5b p5g p5t ib5
    let ib7  : Vec (32*28*28)    := ivS1SkipFwdO ε e6W e6b e6g e6t d6W d6b d6g d6t p6W p6b p6g p6t ib6
    let ib8  : Vec (64*14*14)    := ivS2FwdO ε e7W e7b e7g e7t d7W d7b d7g d7t p7W p7b p7g p7t ib7
    let ib9  : Vec (64*14*14)    := ivS1SkipFwdO ε e8W e8b e8g e8t d8W d8b d8g d8t p8W p8b p8g p8t ib8
    let ib10 : Vec (64*14*14)    := ivS1SkipFwdO ε e9W e9b e9g e9t d9W d9b d9g d9t p9W p9b p9g p9t ib9
    let ib11 : Vec (64*14*14)    := ivS1SkipFwdO ε e10W e10b e10g e10t d10W d10b d10g d10t p10W p10b p10g p10t ib10
    let ib12 : Vec (96*14*14)    := ivS1BodyO ε e11W e11b e11g e11t d11W d11b d11g d11t p11W p11b p11g p11t ib11
    let ib13 : Vec (96*14*14)    := ivS1SkipFwdO ε e12W e12b e12g e12t d12W d12b d12g d12t p12W p12b p12g p12t ib12
    let ib14 : Vec (96*14*14)    := ivS1SkipFwdO ε e13W e13b e13g e13t d13W d13b d13g d13t p13W p13b p13g p13t ib13
    let ib15 : Vec (160*7*7)     := ivS2FwdO ε e14W e14b e14g e14t d14W d14b d14g d14t p14W p14b p14g p14t ib14
    let ib16 : Vec (160*7*7)     := ivS1SkipFwdO ε e15W e15b e15g e15t d15W d15b d15g d15t p15W p15b p15g p15t ib15
    let ib17 : Vec (160*7*7)     := ivS1SkipFwdO ε e16W e16b e16g e16t d16W d16b d16g d16t p16W p16b p16g p16t ib16
    let xhead : Vec (320*7*7)    := ivS1BodyO ε e17W e17b e17g e17t d17W d17b d17g d17t p17W p17b p17g p17t ib17
    -- head forward + the loss cotangent
    let a_gap : Vec 1280 := globalAvgPoolFlat 1280 7 7 (relu6 (1280*7*7)
      (bnPerChannelTensor3 1280 7 7 ε hg ht (flatConv hW hb xhead)))
    let g : Vec 10 := fun k => softmax 10 (mnistLinear Wfc bfc a_gap) k - oneHot 10 label k
    -- backward cotangents (composed from the loss; residual fan-in at each skip)
    let dyHr  : Vec (1280*7*7) := headDyHr ε hW hb hg ht Wfc bfc xhead g
    let dyO17 : Vec (320*7*7)  := headCotIn ε hW hb hg ht Wfc bfc xhead g
    let dyO16 : Vec (160*7*7)  := ivS1NoSkipCotInAt ε e17W e17b e17g e17t d17W d17b d17g d17t p17W p17b p17g p17t ib17 dyO17
    let dyO15 : Vec (160*7*7)  := ivS1SkipCotInAt ε e16W e16b e16g e16t d16W d16b d16g d16t p16W p16b p16g p16t ib16 dyO16
    let dyO14 : Vec (160*7*7)  := ivS1SkipCotInAt ε e15W e15b e15g e15t d15W d15b d15g d15t p15W p15b p15g p15t ib15 dyO15
    let dyO13 : Vec (96*14*14) := ivS2CotInAt ε e14W e14b e14g e14t d14W d14b d14g d14t p14W p14b p14g p14t ib14 dyO14
    let dyO12 : Vec (96*14*14) := ivS1SkipCotInAt ε e13W e13b e13g e13t d13W d13b d13g d13t p13W p13b p13g p13t ib13 dyO13
    let dyO11 : Vec (96*14*14) := ivS1SkipCotInAt ε e12W e12b e12g e12t d12W d12b d12g d12t p12W p12b p12g p12t ib12 dyO12
    let dyO10 : Vec (64*14*14) := ivS1NoSkipCotInAt ε e11W e11b e11g e11t d11W d11b d11g d11t p11W p11b p11g p11t ib11 dyO11
    let dyO9  : Vec (64*14*14) := ivS1SkipCotInAt ε e10W e10b e10g e10t d10W d10b d10g d10t p10W p10b p10g p10t ib10 dyO10
    let dyO8  : Vec (64*14*14) := ivS1SkipCotInAt ε e9W e9b e9g e9t d9W d9b d9g d9t p9W p9b p9g p9t ib9 dyO9
    let dyO7  : Vec (64*14*14) := ivS1SkipCotInAt ε e8W e8b e8g e8t d8W d8b d8g d8t p8W p8b p8g p8t ib8 dyO8
    let dyO6  : Vec (32*28*28) := ivS2CotInAt ε e7W e7b e7g e7t d7W d7b d7g d7t p7W p7b p7g p7t ib7 dyO7
    let dyO5  : Vec (32*28*28) := ivS1SkipCotInAt ε e6W e6b e6g e6t d6W d6b d6g d6t p6W p6b p6g p6t ib6 dyO6
    let dyO4  : Vec (32*28*28) := ivS1SkipCotInAt ε e5W e5b e5g e5t d5W d5b d5g d5t p5W p5b p5g p5t ib5 dyO5
    let dyO3  : Vec (24*56*56) := ivS2CotInAt ε e4W e4b e4g e4t d4W d4b d4g d4t p4W p4b p4g p4t ib4 dyO4
    let dyO2  : Vec (24*56*56) := ivS1SkipCotInAt ε e3W e3b e3g e3t d3W d3b d3g d3t p3W p3b p3g p3t ib3 dyO3
    let dyO1  : Vec (16*112*112) := ivS2CotInAt ε e2W e2b e2g e2t d2W d2b d2g d2t p2W p2b p2g p2t ib2 dyO2
    let dyStem : Vec (32*112*112) := ivNoExpCotInAt ε d1W d1b d1g d1t p1W p1b p1g p1t ib1 dyO1
    -- every block + stem + head tied at its real input + threaded cotangent
    mnv2StemTiedAt xN wN bN gN vN epsStr lrStr cotN ε sW sb sg st x dyStem lr
  ∧ ivNoExpTiedAt xN wN bN gN vN epsStr lrStr cotN ε d1W d1b d1g d1t p1W p1b p1g p1t ib1 dyO1 lr
  ∧ ivS2TiedAt xN wN bN gN vN epsStr lrStr cotN ε e2W e2b e2g e2t d2W d2b d2g d2t p2W p2b p2g p2t ib2 dyO2 lr
  ∧ ivS1TiedAt xN wN bN gN vN epsStr lrStr cotN ε e3W e3b e3g e3t d3W d3b d3g d3t p3W p3b p3g p3t ib3 dyO3 lr
  ∧ ivS2TiedAt xN wN bN gN vN epsStr lrStr cotN ε e4W e4b e4g e4t d4W d4b d4g d4t p4W p4b p4g p4t ib4 dyO4 lr
  ∧ ivS1TiedAt xN wN bN gN vN epsStr lrStr cotN ε e5W e5b e5g e5t d5W d5b d5g d5t p5W p5b p5g p5t ib5 dyO5 lr
  ∧ ivS1TiedAt xN wN bN gN vN epsStr lrStr cotN ε e6W e6b e6g e6t d6W d6b d6g d6t p6W p6b p6g p6t ib6 dyO6 lr
  ∧ ivS2TiedAt xN wN bN gN vN epsStr lrStr cotN ε e7W e7b e7g e7t d7W d7b d7g d7t p7W p7b p7g p7t ib7 dyO7 lr
  ∧ ivS1TiedAt xN wN bN gN vN epsStr lrStr cotN ε e8W e8b e8g e8t d8W d8b d8g d8t p8W p8b p8g p8t ib8 dyO8 lr
  ∧ ivS1TiedAt xN wN bN gN vN epsStr lrStr cotN ε e9W e9b e9g e9t d9W d9b d9g d9t p9W p9b p9g p9t ib9 dyO9 lr
  ∧ ivS1TiedAt xN wN bN gN vN epsStr lrStr cotN ε e10W e10b e10g e10t d10W d10b d10g d10t p10W p10b p10g p10t ib10 dyO10 lr
  ∧ ivS1TiedAt xN wN bN gN vN epsStr lrStr cotN ε e11W e11b e11g e11t d11W d11b d11g d11t p11W p11b p11g p11t ib11 dyO11 lr
  ∧ ivS1TiedAt xN wN bN gN vN epsStr lrStr cotN ε e12W e12b e12g e12t d12W d12b d12g d12t p12W p12b p12g p12t ib12 dyO12 lr
  ∧ ivS1TiedAt xN wN bN gN vN epsStr lrStr cotN ε e13W e13b e13g e13t d13W d13b d13g d13t p13W p13b p13g p13t ib13 dyO13 lr
  ∧ ivS2TiedAt xN wN bN gN vN epsStr lrStr cotN ε e14W e14b e14g e14t d14W d14b d14g d14t p14W p14b p14g p14t ib14 dyO14 lr
  ∧ ivS1TiedAt xN wN bN gN vN epsStr lrStr cotN ε e15W e15b e15g e15t d15W d15b d15g d15t p15W p15b p15g p15t ib15 dyO15 lr
  ∧ ivS1TiedAt xN wN bN gN vN epsStr lrStr cotN ε e16W e16b e16g e16t d16W d16b d16g d16t p16W p16b p16g p16t ib16 dyO16 lr
  ∧ ivS1TiedAt xN wN bN gN vN epsStr lrStr cotN ε e17W e17b e17g e17t d17W d17b d17g d17t p17W p17b p17g p17t ib17 dyO17 lr
  ∧ mnv2HeadTiedAt xN wN bN gN vN epsStr lrStr cotN ε hW hb hg ht xhead dyHr lr
  ∧ (∀ i : Fin 1280, ∀ j : Fin 10,
        den (SHlo.weightSgd dN "%Wfc" lrStr a_gap Wfc lr
              (.operand cotN (fun k => softmax 10 (mnistLinear Wfc bfc a_gap) k - oneHot 10 label k)))
            (finProdFinEquiv (i, j))
          = Wfc i j - lr * pdiv (fun v : Vec (1280 * 10) => fun _ : Fin 1 =>
                crossEntropy 10 (dense (Mat.unflatten v) bfc a_gap) label)
              (Mat.flatten Wfc) (finProdFinEquiv (i, j)) 0)
  ∧ den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe (.operand nlogN (mnistLinear Wfc bfc a_gap))))
          (.operand ohN (oneHot 10 label)))
      = g := by
  intro ib1 ib2 ib3 ib4 ib5 ib6 ib7 ib8 ib9 ib10 ib11 ib12 ib13 ib14 ib15 ib16 ib17 xhead a_gap g
        dyHr dyO17 dyO16 dyO15 dyO14 dyO13 dyO12 dyO11 dyO10 dyO9 dyO8 dyO7 dyO6 dyO5 dyO4 dyO3 dyO2 dyO1 dyStem
  exact ⟨mnv2_stem_tiedAt xN wN bN gN vN epsStr lrStr cotN ε sW sb sg st x dyStem lr,
    mnv2_ivNoExp_tiedAt xN wN bN gN vN epsStr lrStr cotN ε d1W d1b d1g d1t p1W p1b p1g p1t ib1 dyO1 lr,
    mnv2_ivS2_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e2W e2b e2g e2t d2W d2b d2g d2t p2W p2b p2g p2t ib2 dyO2 lr,
    mnv2_ivS1_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e3W e3b e3g e3t d3W d3b d3g d3t p3W p3b p3g p3t ib3 dyO3 lr,
    mnv2_ivS2_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e4W e4b e4g e4t d4W d4b d4g d4t p4W p4b p4g p4t ib4 dyO4 lr,
    mnv2_ivS1_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e5W e5b e5g e5t d5W d5b d5g d5t p5W p5b p5g p5t ib5 dyO5 lr,
    mnv2_ivS1_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e6W e6b e6g e6t d6W d6b d6g d6t p6W p6b p6g p6t ib6 dyO6 lr,
    mnv2_ivS2_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e7W e7b e7g e7t d7W d7b d7g d7t p7W p7b p7g p7t ib7 dyO7 lr,
    mnv2_ivS1_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e8W e8b e8g e8t d8W d8b d8g d8t p8W p8b p8g p8t ib8 dyO8 lr,
    mnv2_ivS1_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e9W e9b e9g e9t d9W d9b d9g d9t p9W p9b p9g p9t ib9 dyO9 lr,
    mnv2_ivS1_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e10W e10b e10g e10t d10W d10b d10g d10t p10W p10b p10g p10t ib10 dyO10 lr,
    mnv2_ivS1_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e11W e11b e11g e11t d11W d11b d11g d11t p11W p11b p11g p11t ib11 dyO11 lr,
    mnv2_ivS1_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e12W e12b e12g e12t d12W d12b d12g d12t p12W p12b p12g p12t ib12 dyO12 lr,
    mnv2_ivS1_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e13W e13b e13g e13t d13W d13b d13g d13t p13W p13b p13g p13t ib13 dyO13 lr,
    mnv2_ivS2_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e14W e14b e14g e14t d14W d14b d14g d14t p14W p14b p14g p14t ib14 dyO14 lr,
    mnv2_ivS1_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e15W e15b e15g e15t d15W d15b d15g d15t p15W p15b p15g p15t ib15 dyO15 lr,
    mnv2_ivS1_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e16W e16b e16g e16t d16W d16b d16g d16t p16W p16b p16g p16t ib16 dyO16 lr,
    mnv2_ivS1_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e17W e17b e17g e17t d17W d17b d17g d17t p17W p17b p17g p17t ib17 dyO17 lr,
    mnv2_head_tiedAt xN wN bN gN vN epsStr lrStr cotN ε hW hb hg ht xhead dyHr lr,
    (fun i j => mnv2_dense_tied_totalloss dN lrStr cotN Wfc bfc a_gap label lr i j),
    mnv2LossCot_den nlogN ohN (mnistLinear Wfc bfc a_gap) label⟩

end Proofs.Mnv2TiePoC
