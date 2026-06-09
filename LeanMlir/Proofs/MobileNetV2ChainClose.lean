import LeanMlir.Proofs.MobileNetV2Close

/-! # MobileNetV2 Item D — pinning the inverted-residual cotangent chain

`MobileNetV2Close.lean` (Item C) certifies each MobileNetV2 conv/depthwise param output for *any*
cotangent `c` at that layer's output. This file pins `c` to the cotangent the **actual backward chain
delivers** — the MobileNetV2 analogue of `ResNet34ChainClose` / `CnnChainClose`, "the genuinely-new,
fiddliest piece" (`planning/mobilenetv2_close.md` Item D).

The chain through an inverted-residual block composes the *rendered* backward denotations — the relu6
two-sided-kink mask (`selectMid`, `if 0<x<6`), the per-channel BN input-VJP (`bnPerChannelTensor3_grad_input`,
= `bnPerChannelBack`'s denotation), the 1×1 conv input-VJP (`conv2d_has_vjp3` via the flatten bridge,
= `convBack`'s denotation), and the depthwise input-VJP (`depthwiseFlat_has_vjp` / `depthwiseStride2Flat_has_vjp`,
= `depthwiseBack` / `depthwiseStridedBack`'s denotation) — back through `project → depthwise → expand`:

  block:  o = [ addV( bn(conv₁ₓ₁ₚ( relu6(bn(dwconv( relu6(bn(conv₁ₓ₁ₑ x)) ))) )), x )  if skip ]
              [        bn(conv₁ₓ₁ₚ( relu6(bn(dwconv( relu6(bn(conv₁ₓ₁ₑ x)) ))) ))         else      ]

The project bottleneck is **linear** (no relu6 after the `addV`), so — unlike r34's `relu(add(…))` —
the cotangent at the project-BN output is `dyOut` *directly* (skip and no-skip alike); that is what
makes `invresCotPc` a plain BN-back. The depthwise is stride-1 (skip blocks) or stride-2 (downsampling
blocks), so the expand-side cotangent lives at the input spatial (2h×2w) for downsampling blocks — the
`_s1` / `_s2` split below. Each conv/depthwise θ output then denotes `θ − lr·(certified ∂/∂θ · the
actual-chain-cotangent)`. Pins the cotangent; the `= ∂loss/∂θ` fold stays separate, as for the CNN.
3-axiom clean.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The cotangent the inverted-residual backward chain delivers at each conv output
--   (activations carried flat; bridges receive `Tensor3.unflatten` where they want Tensor3)
-- ════════════════════════════════════════════════════════════════

/-- Cotangent at the **project conv output** (`oc` ch @h): `bn-back(dyOut)`. The project bottleneck
    is linear (no relu6 after the residual `addV`), so the project-BN output cotangent is the block
    output cotangent `dyOut` directly — skip and no-skip alike. -/
noncomputable def invresCotPc {oc h w : Nat} (ε : ℝ) (γp : Vec oc) (pc dyOut : Vec (oc * h * w)) :
    Vec (oc * h * w) :=
  bnPerChannelTensor3_grad_input oc h w ε γp pc dyOut

/-- Cotangent at the **depthwise conv output** (`mid` ch @h): continue through the project 1×1
    conv-back, the depthwise relu6 mask (`selectMid` on `dn`), and bn-back(γd). -/
noncomputable def invresCotDc {mid oc h w : Nat} (ε : ℝ) (γd : Vec mid) (γp : Vec oc)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc)
    (dr dn dc : Vec (mid * h * w)) (pc dyOut : Vec (oc * h * w)) : Vec (mid * h * w) :=
  let cotPc := invresCotPc ε γp pc dyOut
  let cotDr := (hasVJP3_to_hasVJP (conv2d_has_vjp3 Wp bp)).backward dr cotPc
  bnPerChannelTensor3_grad_input mid h w ε γd dc (fun i => if 0 < dn i ∧ dn i < 6 then cotDr i else 0)

/-- Cotangent at the **expand conv output**, stride-1 block (`mid` ch @h): continue through the
    stride-1 depthwise conv-back, the expand relu6 mask (`selectMid` on `en`), bn-back(γe). -/
noncomputable def invresCotEcS1 {mid oc h w : Nat} (ε : ℝ) (γe γd : Vec mid) (γp : Vec oc)
    (Wd : DepthwiseKernel mid 3 3) (bd : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc)
    (er en ec dr dn dc : Vec (mid * h * w)) (pc dyOut : Vec (oc * h * w)) : Vec (mid * h * w) :=
  let cotDc := invresCotDc ε γd γp Wp bp dr dn dc pc dyOut
  let cotEr := (depthwiseFlat_has_vjp (h := h) (w := w) Wd bd).backward er cotDc
  bnPerChannelTensor3_grad_input mid h w ε γe ec (fun i => if 0 < en i ∧ en i < 6 then cotEr i else 0)

/-- Cotangent at the **expand conv output**, stride-2 (downsampling) block (`mid` ch @2h, since the
    expand acts at the block-input spatial): the depthwise is strided, so its input-VJP
    zero-upsamples; `er`/`en`/`ec` live at 2h×2w. -/
noncomputable def invresCotEcS2 {mid oc h w : Nat} (ε : ℝ) (γe γd : Vec mid) (γp : Vec oc)
    (Wd : DepthwiseKernel mid 3 3) (bd : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc)
    (er en ec : Vec (mid * (2 * h) * (2 * w))) (dr dn dc : Vec (mid * h * w))
    (pc dyOut : Vec (oc * h * w)) : Vec (mid * (2 * h) * (2 * w)) :=
  let cotDc := invresCotDc ε γd γp Wp bp dr dn dc pc dyOut
  let cotEr := (depthwiseStride2Flat_has_vjp (h := h) (w := w) Wd bd).backward er cotDc
  bnPerChannelTensor3_grad_input mid (2 * h) (2 * w) ε γe ec
    (fun i => if 0 < en i ∧ en i < 6 then cotEr i else 0)

/-- Cotangent at the **stem conv output** (`16` ch @112): `bn-back( relu6'(stn) ⊙ dyStem )`, where
    `dyStem` is the cotangent block-1 delivers at the stem's relu6 output. (MobileNetV2's stem has no
    maxpool — just conv→bn→relu6 — so this is simpler than r34's stem.) -/
noncomputable def mnv2StemCot {oc h w : Nat} (ε : ℝ) (γs : Vec oc)
    (stn stc dyStem : Vec (oc * h * w)) : Vec (oc * h * w) :=
  bnPerChannelTensor3_grad_input oc h w ε γs stc
    (fun i => if 0 < stn i ∧ stn i < 6 then dyStem i else 0)

-- ════════════════════════════════════════════════════════════════
-- § The chain-pinned closes — the Item C bridges at the actual cotangents
-- ════════════════════════════════════════════════════════════════

/-- **Project 1×1 conv weight, chain-certified.** `Wpⁿ` denotes `Wp − lr·(certified ∂conv/∂Wp ·
    bn-back(dyOut))`. -/
theorem invres_render_projW_chain_certified {mid oc h w : Nat}
    (bp : Vec oc) (dr : Vec (mid * h * w)) (ε : ℝ) (γp : Vec oc) (pc dyOut : Vec (oc * h * w))
    (v : Vec (oc * mid * 1 * 1)) (lr : ℝ) (idx : Fin (oc * mid * 1 * 1)) :
    v idx - lr * (conv2d_weight_grad_has_vjp bp (Tensor3.unflatten dr)).backward v
        (invresCotPc ε γp pc dyOut) idx
      = v idx - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') bp (Tensor3.unflatten dr))) v idx j
            * invresCotPc ε γp pc dyOut j :=
  cnn_render_convW_certified bp (Tensor3.unflatten dr) v (invresCotPc ε γp pc dyOut) lr idx

/-- **Project 1×1 conv bias, chain-certified.** -/
theorem invres_render_projb_chain_certified {mid oc h w : Nat}
    (Wp : Kernel4 oc mid 1 1) (dr : Vec (mid * h * w)) (bp : Vec oc)
    (ε : ℝ) (γp : Vec oc) (pc dyOut : Vec (oc * h * w)) (lr : ℝ) (o : Fin oc) :
    bp o - lr * (conv2d_bias_grad_has_vjp Wp (Tensor3.unflatten dr)).backward bp
        (invresCotPc ε γp pc dyOut) o
      = bp o - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d Wp b' (Tensor3.unflatten dr))) bp o j
            * invresCotPc ε γp pc dyOut j :=
  cnn_render_convb_certified Wp (Tensor3.unflatten dr) bp (invresCotPc ε γp pc dyOut) lr o

/-- **Depthwise weight, chain-certified (stride-1 block).** `Wdⁿ` denotes `Wd − lr·(certified
    ∂(depthwiseConv2d)/∂Wd · the chain cotangent at the depthwise output)`. -/
theorem invres_render_dwW_s1_chain_certified {mid oc h w : Nat}
    (bd : Vec mid) (er : Tensor3 mid h w) (Wd : DepthwiseKernel mid 3 3) (γd : Vec mid) (γp : Vec oc)
    (ε : ℝ) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (dr dn dc : Vec (mid * h * w))
    (pc dyOut : Vec (oc * h * w)) (lr : ℝ) (ci : Fin mid) (hi : Fin 3) (wi : Fin 3) :
    Wd ci hi wi - lr * (depthwise_weight_grad_has_vjp3 bd er).backward Wd
        (Tensor3.unflatten (invresCotDc ε γd γp Wp bp dr dn dc pc dyOut)) ci hi wi
      = Wd ci hi wi - lr * ∑ co : Fin mid, ∑ ho : Fin h, ∑ wo : Fin w,
          pdiv3 (fun W' : DepthwiseKernel mid 3 3 => depthwiseConv2d W' bd er) Wd ci hi wi co ho wo
            * (Tensor3.unflatten (invresCotDc ε γd γp Wp bp dr dn dc pc dyOut) : Tensor3 mid h w) co ho wo :=
  mnv2_render_depthwiseW_certified bd er Wd
    (Tensor3.unflatten (invresCotDc ε γd γp Wp bp dr dn dc pc dyOut)) lr ci hi wi

/-- **Depthwise bias, chain-certified (stride-1 block).** -/
theorem invres_render_dwb_s1_chain_certified {mid oc h w : Nat}
    (Wd : DepthwiseKernel mid 3 3) (er : Tensor3 mid h w) (bd : Vec mid) (γd : Vec mid) (γp : Vec oc)
    (ε : ℝ) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (dr dn dc : Vec (mid * h * w))
    (pc dyOut : Vec (oc * h * w)) (lr : ℝ) (cc : Fin mid) :
    bd cc - lr * (depthwise_bias_grad_has_vjp Wd er).backward bd
        (invresCotDc ε γd γp Wp bp dr dn dc pc dyOut) cc
      = bd cc - lr * ∑ j : Fin (mid * h * w),
          pdiv (fun b' : Vec mid => Tensor3.flatten (depthwiseConv2d Wd b' er)) bd cc j
            * invresCotDc ε γd γp Wp bp dr dn dc pc dyOut j :=
  mnv2_render_depthwiseb_certified Wd er bd (invresCotDc ε γd γp Wp bp dr dn dc pc dyOut) lr cc

/-- **Depthwise weight, chain-certified (stride-2 downsampling block).** The strided depthwise weight
    bridge at the chain cotangent. -/
theorem invres_render_dwW_s2_chain_certified {mid oc h w : Nat}
    (bd : Vec mid) (er : Vec (mid * (2 * h) * (2 * w))) (γd : Vec mid) (γp : Vec oc)
    (ε : ℝ) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (dr dn dc : Vec (mid * h * w))
    (pc dyOut : Vec (oc * h * w)) (v : Vec (mid * 3 * 3)) (lr : ℝ) (i : Fin (mid * 3 * 3)) :
    v i - lr * (depthwiseStride2_weight_grad_has_vjp bd er).backward v
        (invresCotDc ε γd γp Wp bp dr dn dc pc dyOut) i
      = v i - lr * ∑ j : Fin (mid * h * w),
          pdiv (fun v' : Vec (mid * 3 * 3) =>
                  depthwiseStride2Flat (Tensor3.unflatten v' : DepthwiseKernel mid 3 3) bd er) v i j
            * invresCotDc ε γd γp Wp bp dr dn dc pc dyOut j :=
  mnv2_render_depthwiseW_strided_certified bd er v (invresCotDc ε γd γp Wp bp dr dn dc pc dyOut) lr i

/-- **Depthwise bias, chain-certified (stride-2 downsampling block).** -/
theorem invres_render_dwb_s2_chain_certified {mid oc h w : Nat}
    (Wd : DepthwiseKernel mid 3 3) (er : Vec (mid * (2 * h) * (2 * w))) (bd : Vec mid) (γd : Vec mid)
    (γp : Vec oc) (ε : ℝ) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (dr dn dc : Vec (mid * h * w))
    (pc dyOut : Vec (oc * h * w)) (lr : ℝ) (o : Fin mid) :
    bd o - lr * (depthwiseStride2_bias_grad_has_vjp Wd er).backward bd
        (invresCotDc ε γd γp Wp bp dr dn dc pc dyOut) o
      = bd o - lr * ∑ j : Fin (mid * h * w),
          pdiv (fun b' : Vec mid => depthwiseStride2Flat Wd b' er) bd o j
            * invresCotDc ε γd γp Wp bp dr dn dc pc dyOut j :=
  mnv2_render_depthwiseb_strided_certified Wd er bd (invresCotDc ε γd γp Wp bp dr dn dc pc dyOut) lr o

/-- **Expand 1×1 conv weight, chain-certified (stride-1 block).** `Weⁿ` denotes `We − lr·(certified
    ∂conv/∂We · the deepest in-block cotangent)` — the generic 1×1 bridge at `invresCotEcS1`. -/
theorem invres_render_expW_s1_chain_certified {ic mid oc h w : Nat}
    (be : Vec mid) (xin : Tensor3 ic h w) (ε : ℝ) (γe γd : Vec mid) (γp : Vec oc)
    (Wd : DepthwiseKernel mid 3 3) (bd : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc)
    (er en ec dr dn dc : Vec (mid * h * w)) (pc dyOut : Vec (oc * h * w))
    (v : Vec (mid * ic * 1 * 1)) (lr : ℝ) (idx : Fin (mid * ic * 1 * 1)) :
    v idx - lr * (conv2d_weight_grad_has_vjp be xin).backward v
        (invresCotEcS1 ε γe γd γp Wd bd Wp bp er en ec dr dn dc pc dyOut) idx
      = v idx - lr * ∑ j : Fin (mid * h * w),
          pdiv (fun v' : Vec (mid * ic * 1 * 1) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') be xin)) v idx j
            * invresCotEcS1 ε γe γd γp Wd bd Wp bp er en ec dr dn dc pc dyOut j :=
  cnn_render_convW_certified be xin v
    (invresCotEcS1 ε γe γd γp Wd bd Wp bp er en ec dr dn dc pc dyOut) lr idx

/-- **Expand 1×1 conv weight, chain-certified (stride-2 downsampling block).** The block input is at
    2h×2w; the chain cotangent is `invresCotEcS2` (the strided depthwise input-VJP). -/
theorem invres_render_expW_s2_chain_certified {ic mid oc h w : Nat}
    (be : Vec mid) (xin : Tensor3 ic (2 * h) (2 * w)) (ε : ℝ) (γe γd : Vec mid) (γp : Vec oc)
    (Wd : DepthwiseKernel mid 3 3) (bd : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc)
    (er en ec : Vec (mid * (2 * h) * (2 * w))) (dr dn dc : Vec (mid * h * w))
    (pc dyOut : Vec (oc * h * w)) (v : Vec (mid * ic * 1 * 1)) (lr : ℝ) (idx : Fin (mid * ic * 1 * 1)) :
    v idx - lr * (conv2d_weight_grad_has_vjp be xin).backward v
        (invresCotEcS2 ε γe γd γp Wd bd Wp bp er en ec dr dn dc pc dyOut) idx
      = v idx - lr * ∑ j : Fin (mid * (2 * h) * (2 * w)),
          pdiv (fun v' : Vec (mid * ic * 1 * 1) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') be xin)) v idx j
            * invresCotEcS2 ε γe γd γp Wd bd Wp bp er en ec dr dn dc pc dyOut j :=
  cnn_render_convW_certified be xin v
    (invresCotEcS2 ε γe γd γp Wd bd Wp bp er en ec dr dn dc pc dyOut) lr idx

/-- **Stem 3×3 strided conv weight, chain-certified.** `sWⁿ` denotes `sW − lr·(certified
    ∂(flatConvStride2)/∂sW · bn-back(relu6'(stn) ⊙ the block-1 input cotangent))`. -/
theorem mnv2_stem_render_convW_chain_certified {ic oc h w : Nat}
    (bs : Vec oc) (x : Vec (ic * (2 * h) * (2 * w))) (ε : ℝ) (γs : Vec oc)
    (stn stc dyStem : Vec (oc * h * w)) (v : Vec (oc * ic * 3 * 3)) (lr : ℝ)
    (i : Fin (oc * ic * 3 * 3)) :
    v i - lr * (flatConvStride2_weight_grad_has_vjp bs x).backward v (mnv2StemCot ε γs stn stc dyStem) i
      = v i - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * 3 * 3) => flatConvStride2 (Kernel4.unflatten v') bs x) v i j
            * mnv2StemCot ε γs stn stc dyStem j :=
  mnv2_render_stem_convW_certified bs x v (mnv2StemCot ε γs stn stc dyStem) lr i

end Proofs
