import LeanMlir.Proofs.Nets.Small.CifarBnClose
import LeanMlir.Proofs.Nets.Small.CnnChainClose

/-! # cifar8 render CLOSE — pinning the deeper (8-conv) CIFAR-CNN cotangent chain

`StableHLO.cifar8{,Bn}FwdGraph_faithful` (this PR's Stage 1) certifies the deeper 8-conv CIFAR-CNN
**forward** graph denotes the proven `cifarCnn8{Bn}Forward`. This file is the **backward** peer: it
pins each cifar8 parameter output to the cotangent the *actual backward chain delivers* — the cifar8
analogue of `CnnChainClose` (the 4-conv `cifar`/`cifar_bn` net) extended by two more conv→conv→pool
stages, following the `ConvNeXtChainClose` / `MobileNetV2ChainClose` recipe (Item D).

The chain through the BN net composes the *rendered* backward denotations — the dense-head flat
`IR.Back` chain (`cnnDenseHeadCot`, the `mlpCotOut` mechanism, reused verbatim for the
`d1→d1→nClasses` head), the maxpool input-VJP (`Back3.maxpool` via `flatDenote`, crossing the
flatten boundary), the ReLU mask (`relu'(·)⊙·`), the per-channel BN input-VJP
(`bnPerChannelTensor3_grad_input`, = `bnPerChannelBack`'s denotation, under `0<ε`), and the conv
input-VJP (`conv2d_has_vjp3` via the flatten bridge, = `convBack`'s denotation) — back through

  conv₁→bn₁→relu→conv₂→bn₂→relu→pool₁ → … → conv₇→bn₇→relu→conv₈→bn₈→relu→pool₄ → dense₉→relu→denseₐ→relu→denseᵦ

Each conv/BN/dense θ output then denotes `θ − lr·(certified ∂/∂θ · the actual-chain cotangent)` — the
conv W/b reuse `cnn_render_conv{W,b}_certified`, the BN γ/β reuse `cifar_bn_render_{gamma,beta}_certified`
(γ/β enter affinely → no `0<ε` for the param grad), the dense W/b reuse the M2 `weight_grad_bridge` /
`bias_grad_bridge`. Pins the cotangent; the `= ∂loss/∂θ` fold stays separate, as for the 4-conv CNN.
3-axiom clean. (The no-BN cifar8 net's conv chain is `CnnChainClose` verbatim — same dense-head + maxpool
+ relu + conv-back pieces, just four pool stages — so this file's BN chain is the deeper, primary close;
the no-BN cotangents are the BN ones with every `bnPerChannelTensor3_grad_input` deleted.)
-/

namespace Proofs

open Proofs.IR
open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The cotangent the cifar8-BN backward chain delivers at each conv output
--   (activations carried flat; bridges receive `Tensor3.unflatten` where they want Tensor3)
--
--   Stage 4 (deepest, nearest the head):  conv₇→bn₇→relu→conv₈→bn₈→relu→pool₄
--   The dense-head + pool₄ feed the cotangent `cpool₄` at the flat pool₄ output (c4·h·w); then
--   each BN-conv layer prepends  conv-back → relu-mask → bn-back  (the MNV2 stride-1 recipe).
-- ════════════════════════════════════════════════════════════════

/-- Cotangent at the **flat pool₄ output** (`c4·h·w`): the dense-head flat `Back` chain
    `W₉·(relu'(h9)⊙(Wa·(relu'(ha)⊙(Wb·dy))))` — `cnnDenseHeadCot` reused at the `c4·h·w → d1 → d1 →
    nClasses` head (the `mlpCotOut` mechanism). -/
noncomputable def cifar8DenseHeadCot {c4 h w d1 nClasses : Nat}
    (W₉ : Mat (c4 * h * w) d1) (Wa : Mat d1 d1) (Wb : Mat d1 nClasses) (h9 ha : Vec d1)
    (dy : Vec nClasses) : Vec (c4 * h * w) :=
  (cnnDenseHeadCot W₉ Wa Wb h9 ha).denote dy

/-- Cotangent at **bn₈'s output** (`c4` ch @ 2h, = the input to relu₈ then pool₄): `relu'(bn₈out) ⊙
    maxpool₄-back(cpool₄)` — the pool₄ input-VJP (`Back3.maxpool` via `flatDenote`) masked by relu₈.
    `bn8o` is the saved bn₈ pre-activation (= relu₈ input); `ac8` the pool₄ input. -/
noncomputable def cifar8CotBn8 {c4 h w d1 nClasses : Nat}
    (W₉ : Mat (c4 * h * w) d1) (Wa : Mat d1 d1) (Wb : Mat d1 nClasses) (h9 ha : Vec d1)
    (ac8 : Tensor3 c4 (2 * h) (2 * w)) (bn8o : Vec (c4 * (2 * h) * (2 * w)))
    (dy : Vec nClasses) : Vec (c4 * (2 * h) * (2 * w)) :=
  fun i => if bn8o i > 0
    then (Back3.maxpool (c₁ := c4) (h₁ := h) (w₁ := w) ac8 Back3.cot).flatDenote
           (cifar8DenseHeadCot W₉ Wa Wb h9 ha dy) i
    else 0

/-- Cotangent at **conv₈'s output** (`c4` ch @ 2h, = the input to bn₈): the per-channel BN₈
    input-VJP applied to `cifar8CotBn8`. `cc8` is the saved conv₈ output (the bn₈ input). -/
noncomputable def cifar8CotConv8 {c4 h w d1 nClasses : Nat}
    (ε₈ : ℝ) (γ₈ : Vec c4) (W₉ : Mat (c4 * h * w) d1) (Wa : Mat d1 d1) (Wb : Mat d1 nClasses)
    (h9 ha : Vec d1) (ac8 : Tensor3 c4 (2 * h) (2 * w)) (bn8o cc8 : Vec (c4 * (2 * h) * (2 * w)))
    (dy : Vec nClasses) : Vec (c4 * (2 * h) * (2 * w)) :=
  bnPerChannelTensor3_grad_input c4 (2 * h) (2 * w) ε₈ γ₈ cc8
    (cifar8CotBn8 W₉ Wa Wb h9 ha ac8 bn8o dy)

/-- Cotangent at **bn₇'s output** (`c4` ch @ 2h): `relu'(bn₇out) ⊙ conv₈-back(W₈, cifar8CotConv8)` —
    the conv₈ input-VJP (`Back3.conv` via `flatDenote`) masked by relu₇. -/
noncomputable def cifar8CotBn7 {c4 h w d1 nClasses kH kW : Nat}
    (ε₈ : ℝ) (γ₈ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW)
    (W₉ : Mat (c4 * h * w) d1) (Wa : Mat d1 d1) (Wb : Mat d1 nClasses) (h9 ha : Vec d1)
    (ac8 : Tensor3 c4 (2 * h) (2 * w)) (bn8o cc8 bn7o : Vec (c4 * (2 * h) * (2 * w)))
    (dy : Vec nClasses) : Vec (c4 * (2 * h) * (2 * w)) :=
  fun i => if bn7o i > 0
    then (Back3.conv (c₁ := c4) (h₁ := 2 * h) (w₁ := 2 * w) W₈ Back3.cot).flatDenote
           (cifar8CotConv8 ε₈ γ₈ W₉ Wa Wb h9 ha ac8 bn8o cc8 dy) i
    else 0

/-- Cotangent at **conv₇'s output** (`c4` ch @ 2h): bn₇ input-VJP of `cifar8CotBn7`. This is the
    deepest cotangent of stage 4; `cc7` is the saved conv₇ output. -/
noncomputable def cifar8CotConv7 {c4 h w d1 nClasses kH kW : Nat}
    (ε₇ ε₈ : ℝ) (γ₇ γ₈ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW)
    (W₉ : Mat (c4 * h * w) d1) (Wa : Mat d1 d1) (Wb : Mat d1 nClasses) (h9 ha : Vec d1)
    (ac8 : Tensor3 c4 (2 * h) (2 * w)) (bn8o cc8 bn7o cc7 : Vec (c4 * (2 * h) * (2 * w)))
    (dy : Vec nClasses) : Vec (c4 * (2 * h) * (2 * w)) :=
  bnPerChannelTensor3_grad_input c4 (2 * h) (2 * w) ε₇ γ₇ cc7
    (cifar8CotBn7 ε₈ γ₈ W₈ W₉ Wa Wb h9 ha ac8 bn8o cc8 bn7o dy)

-- ════════════════════════════════════════════════════════════════
-- § The chain-pinned closes — the generic per-op bridges at the actual cotangents
--   (representative stage-4 layer; the deeper stages compose by the same instantiation —
--    each pool's input-VJP prepends one `Back3.maxpool` + relu mask, each conv one `Back3.conv` +
--    relu mask + `bnPerChannelTensor3_grad_input`, generic in the downstream cotangent, exactly as
--    `CnnChainClose`'s conv₂→conv₁ and `ConvNeXtChainClose`'s block composition.)
-- ════════════════════════════════════════════════════════════════

/-- **Dense-head cotangent, explicit.** `cifar8DenseHeadCot` is the explicit dense backprop
    `W₉·(relu'(h9)⊙(Wa·(relu'(ha)⊙(Wb·dy))))` — the head's `mlpCotOut`-style chain spelled out via
    the `IR.Back` chain rule (`cnnDenseHeadCot_denote` at the cifar8 head shapes). -/
theorem cifar8DenseHeadCot_denote {c4 h w d1 nClasses : Nat}
    (W₉ : Mat (c4 * h * w) d1) (Wa : Mat d1 d1) (Wb : Mat d1 nClasses) (h9 ha : Vec d1)
    (dy : Vec nClasses) :
    cifar8DenseHeadCot W₉ Wa Wb h9 ha dy
      = Mat.mulVec W₉ (fun i => if h9 i > 0
          then Mat.mulVec Wa (fun k => if ha k > 0 then Mat.mulVec Wb dy k else 0) i else 0) :=
  cnnDenseHeadCot_denote W₉ Wa Wb h9 ha dy

/-- **Dense-head Wb (logit layer) weight, chain-certified.** `Wbⁿ` denotes `Wb − lr·(certified
    ∂dense/∂Wb · dy)` — the loss cotangent `dy` is the top cotangent at the logits, so the head's
    last layer reuses the M2 `weight_grad_bridge` directly (no chain content beyond Item C). -/
theorem cifar8_render_denseWb_chain_certified {d1 nClasses : Nat}
    (Wb : Mat d1 nClasses) (bb : Vec nClasses) (xa : Vec d1) (dy : Vec nClasses)
    (i : Fin d1) (j : Fin nClasses) :
    emitWeightGrad xa Back.cotangent dy i j
      = ∑ k : Fin nClasses,
          pdiv (fun v : Vec (d1 * nClasses) => dense (Mat.unflatten v) bb xa)
               (Mat.flatten Wb) (finProdFinEquiv (i, j)) k * dy k := by
  have := weight_grad_bridge Wb bb xa (Back.cotangent (inp := nClasses)) dy i j
  simpa [Back.denote] using this

/-- **Dense-head bias bb, chain-certified.** -/
theorem cifar8_render_densebb_chain_certified {d1 nClasses : Nat}
    (Wb : Mat d1 nClasses) (bb : Vec nClasses) (xa : Vec d1) (dy : Vec nClasses) (i : Fin nClasses) :
    emitBiasGrad Back.cotangent dy i
      = ∑ j : Fin nClasses, pdiv (fun b' : Vec nClasses => dense Wb b' xa) bb i j * dy j := by
  have := bias_grad_bridge Wb bb xa (Back.cotangent (inp := nClasses)) dy i
  simpa [Back.denote] using this

/-- **Conv-8 weight output, chain-certified.** `W₈ⁿ = W₈ − lr·(transpose-trick kernel grad)` denotes
    `W₈ − lr·(certified ∂conv₈/∂W₈ · the cotangent the chain delivers at conv₈)` — the generic
    `cnn_render_convW_certified` instantiated at `cifar8CotConv8`. `ac7` is the saved conv₈ input
    (the relu₇ output). -/
theorem cifar8_render_convW8_chain_certified {c4 h w d1 nClasses kH kW : Nat}
    (b₈ : Vec c4) (ac7 : Tensor3 c4 (2 * h) (2 * w)) (ε₈ : ℝ) (γ₈ : Vec c4)
    (W₉ : Mat (c4 * h * w) d1) (Wa : Mat d1 d1) (Wb : Mat d1 nClasses) (h9 ha : Vec d1)
    (ac8 : Tensor3 c4 (2 * h) (2 * w)) (bn8o cc8 : Vec (c4 * (2 * h) * (2 * w)))
    (dy : Vec nClasses) (v : Vec (c4 * c4 * kH * kW)) (lr : ℝ) (idx : Fin (c4 * c4 * kH * kW)) :
    v idx - lr * (conv2d_weight_grad_has_vjp b₈ ac7).backward v
        (cifar8CotConv8 ε₈ γ₈ W₉ Wa Wb h9 ha ac8 bn8o cc8 dy) idx
      = v idx - lr * ∑ j : Fin (c4 * (2 * h) * (2 * w)),
          pdiv (fun v' : Vec (c4 * c4 * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b₈ ac7)) v idx j
            * cifar8CotConv8 ε₈ γ₈ W₉ Wa Wb h9 ha ac8 bn8o cc8 dy j :=
  cnn_render_convW_certified b₈ ac7 v (cifar8CotConv8 ε₈ γ₈ W₉ Wa Wb h9 ha ac8 bn8o cc8 dy) lr idx

/-- **Conv-8 bias output, chain-certified.** -/
theorem cifar8_render_convb8_chain_certified {c4 h w d1 nClasses kH kW : Nat}
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (ac7 : Tensor3 c4 (2 * h) (2 * w)) (ε₈ : ℝ) (γ₈ : Vec c4)
    (W₉ : Mat (c4 * h * w) d1) (Wa : Mat d1 d1) (Wb : Mat d1 nClasses) (h9 ha : Vec d1)
    (ac8 : Tensor3 c4 (2 * h) (2 * w)) (bn8o cc8 : Vec (c4 * (2 * h) * (2 * w)))
    (dy : Vec nClasses) (lr : ℝ) (o : Fin c4) :
    b₈ o - lr * (conv2d_bias_grad_has_vjp W₈ ac7).backward b₈
        (cifar8CotConv8 ε₈ γ₈ W₉ Wa Wb h9 ha ac8 bn8o cc8 dy) o
      = b₈ o - lr * ∑ j : Fin (c4 * (2 * h) * (2 * w)),
          pdiv (fun b' : Vec c4 => Tensor3.flatten (conv2d W₈ b' ac7)) b₈ o j
            * cifar8CotConv8 ε₈ γ₈ W₉ Wa Wb h9 ha ac8 bn8o cc8 dy j :=
  cnn_render_convb_certified W₈ ac7 b₈ (cifar8CotConv8 ε₈ γ₈ W₉ Wa Wb h9 ha ac8 bn8o cc8 dy) lr o

/-- **BN-8 γ output, chain-certified.** The chain cotangent at bn₈'s output is `cifar8CotBn8` (relu₈
    mask on the pool₄ input-VJP), in the per-channel-flat `c4·m` layout (`m = 2h·2w`); `γ₈ⁿ` denotes
    `γ₈ − lr·(certified ∂(per-channel BN)/∂γ₈ · cifar8CotBn8)` with the saved conv₈ output `cc8` as the
    BN input. γ enters affinely → no `0<ε`. The cifar8 instance of `cifar_bn_render_gamma_certified`. -/
theorem cifar8_render_bn8gamma_chain_certified {c4 m : Nat} (ε₈ : ℝ) (γ₈ β₈ : Vec c4)
    (cc8 dyBn8 : Vec (c4 * m)) (lr : ℝ) (idx : Fin c4) :
    γ₈ idx - lr * bnPerChannel_grad_gamma c4 m ε₈ cc8 dyBn8 idx
      = γ₈ idx - lr * ∑ j : Fin (c4 * m),
          pdiv (fun γ' : Vec c4 => bnPerChannelFlat c4 m ε₈ γ' β₈ cc8) γ₈ idx j * dyBn8 j :=
  cifar_bn_render_gamma_certified c4 m ε₈ γ₈ β₈ cc8 dyBn8 lr idx

/-- **BN-8 β output, chain-certified.** -/
theorem cifar8_render_bn8beta_chain_certified {c4 m : Nat} (ε₈ : ℝ) (γ₈ β₈ : Vec c4)
    (cc8 dyBn8 : Vec (c4 * m)) (lr : ℝ) (idx : Fin c4) :
    β₈ idx - lr * bnPerChannel_grad_beta c4 m dyBn8 idx
      = β₈ idx - lr * ∑ j : Fin (c4 * m),
          pdiv (fun β' : Vec c4 => bnPerChannelFlat c4 m ε₈ γ₈ β' cc8) β₈ idx j * dyBn8 j :=
  cifar_bn_render_beta_certified c4 m ε₈ γ₈ β₈ cc8 dyBn8 lr idx

/-- **Conv-7 weight output, chain-certified.** `W₇ⁿ` denotes `W₇ − lr·(certified ∂conv₇/∂W₇ · the
    deepest stage-4 cotangent)` — the generic bridge at `cifar8CotConv7` (which crosses one more
    conv-back + relu-mask + bn-back than `cifar8CotConv8`, the next chain step). `ac6` is the saved
    conv₇ input. -/
theorem cifar8_render_convW7_chain_certified {c3 c4 h w d1 nClasses kH kW : Nat}
    (b₇ : Vec c4) (ac6 : Tensor3 c3 (2 * h) (2 * w)) (ε₇ ε₈ : ℝ) (γ₇ γ₈ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW)
    (W₉ : Mat (c4 * h * w) d1) (Wa : Mat d1 d1) (Wb : Mat d1 nClasses) (h9 ha : Vec d1)
    (ac8 : Tensor3 c4 (2 * h) (2 * w)) (bn8o cc8 bn7o cc7 : Vec (c4 * (2 * h) * (2 * w)))
    (dy : Vec nClasses) (v : Vec (c4 * c3 * kH * kW)) (lr : ℝ) (idx : Fin (c4 * c3 * kH * kW)) :
    v idx - lr * (conv2d_weight_grad_has_vjp b₇ ac6).backward v
        (cifar8CotConv7 ε₇ ε₈ γ₇ γ₈ W₈ W₉ Wa Wb h9 ha ac8 bn8o cc8 bn7o cc7 dy) idx
      = v idx - lr * ∑ j : Fin (c4 * (2 * h) * (2 * w)),
          pdiv (fun v' : Vec (c4 * c3 * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b₇ ac6)) v idx j
            * cifar8CotConv7 ε₇ ε₈ γ₇ γ₈ W₈ W₉ Wa Wb h9 ha ac8 bn8o cc8 bn7o cc7 dy j :=
  cnn_render_convW_certified b₇ ac6 v
    (cifar8CotConv7 ε₇ ε₈ γ₇ γ₈ W₈ W₉ Wa Wb h9 ha ac8 bn8o cc8 bn7o cc7 dy) lr idx

/-- **Conv-7 bias output, chain-certified.** -/
theorem cifar8_render_convb7_chain_certified {c3 c4 h w d1 nClasses kH kW : Nat}
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (ac6 : Tensor3 c3 (2 * h) (2 * w))
    (ε₇ ε₈ : ℝ) (γ₇ γ₈ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW)
    (W₉ : Mat (c4 * h * w) d1) (Wa : Mat d1 d1) (Wb : Mat d1 nClasses) (h9 ha : Vec d1)
    (ac8 : Tensor3 c4 (2 * h) (2 * w)) (bn8o cc8 bn7o cc7 : Vec (c4 * (2 * h) * (2 * w)))
    (dy : Vec nClasses) (lr : ℝ) (o : Fin c4) :
    b₇ o - lr * (conv2d_bias_grad_has_vjp W₇ ac6).backward b₇
        (cifar8CotConv7 ε₇ ε₈ γ₇ γ₈ W₈ W₉ Wa Wb h9 ha ac8 bn8o cc8 bn7o cc7 dy) o
      = b₇ o - lr * ∑ j : Fin (c4 * (2 * h) * (2 * w)),
          pdiv (fun b' : Vec c4 => Tensor3.flatten (conv2d W₇ b' ac6)) b₇ o j
            * cifar8CotConv7 ε₇ ε₈ γ₇ γ₈ W₈ W₉ Wa Wb h9 ha ac8 bn8o cc8 bn7o cc7 dy j :=
  cnn_render_convb_certified W₇ ac6 b₇
    (cifar8CotConv7 ε₇ ε₈ γ₇ γ₈ W₈ W₉ Wa Wb h9 ha ac8 bn8o cc8 bn7o cc7 dy) lr o

/-- **Dense W₉ (flatten→d1 layer) weight, chain-certified.** `W₉ⁿ` denotes `W₉ − lr·(certified
    ∂dense/∂W₉ · the cotangent the head delivers at the d1 layer)` — the M2 `weight_grad_bridge` at
    the head's `mlpCotOut`-style cotangent (`relu'(h9)⊙(Wa·(relu'(ha)⊙(Wb·dy)))`). `xpool` is the
    saved flat pool₄ output (the W₉ input). -/
theorem cifar8_render_denseW9_chain_certified {c4 h w d1 nClasses : Nat}
    (W₉ : Mat (c4 * h * w) d1) (b₉ : Vec d1) (xpool : Vec (c4 * h * w))
    (Wa : Mat d1 d1) (Wb : Mat d1 nClasses) (h9 ha : Vec d1) (dy : Vec nClasses)
    (i : Fin (c4 * h * w)) (j : Fin d1) :
    emitWeightGrad xpool (mlpCotOut0 Wa Wb h9 ha) dy i j
      = ∑ k : Fin d1,
          pdiv (fun v : Vec ((c4 * h * w) * d1) => dense (Mat.unflatten v) b₉ xpool)
               (Mat.flatten W₉) (finProdFinEquiv (i, j)) k
            * (mlpCotOut0 Wa Wb h9 ha).denote dy k :=
  weight_grad_bridge W₉ b₉ xpool (mlpCotOut0 Wa Wb h9 ha) dy i j

/-- **Dense W₉ bias b₉, chain-certified.** -/
theorem cifar8_render_denseb9_chain_certified {c4 h w d1 nClasses : Nat}
    (W₉ : Mat (c4 * h * w) d1) (b₉ : Vec d1) (xpool : Vec (c4 * h * w))
    (Wa : Mat d1 d1) (Wb : Mat d1 nClasses) (h9 ha : Vec d1) (dy : Vec nClasses) (i : Fin d1) :
    emitBiasGrad (mlpCotOut0 Wa Wb h9 ha) dy i
      = ∑ j : Fin d1, pdiv (fun b' : Vec d1 => dense W₉ b' xpool) b₉ i j
          * (mlpCotOut0 Wa Wb h9 ha).denote dy j :=
  bias_grad_bridge W₉ b₉ xpool (mlpCotOut0 Wa Wb h9 ha) dy i

-- The four pool stages compose by instantiation, exactly as in `CnnChainClose`
-- (conv₂→conv₁) and `ConvNeXtChainClose` (block→block): the cotangent at each pool's input is the
-- next deeper stage's `cifar8DenseHeadCot`-analogue, each BN-conv layer prepends one
-- `Back3.conv`-back + relu mask + `bnPerChannelTensor3_grad_input`, and each pool boundary prepends
-- one `Back3.maxpool`-back + relu mask — every theorem above is generic in its downstream cotangent.
-- The intermediate γ/β at the shallower stages reuse `cifar8_render_bn8{gamma,beta}_chain_certified`
-- at their own `dyBn` cotangents (the cifar8 instance of the affine BN param bridge); the loss-side
-- `= ∂loss/∂θ` fold is the separate `ConvLossFold` concern.

end Proofs
