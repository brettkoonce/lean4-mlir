import LeanMlir.Proofs.ResNet34Close
import LeanMlir.Proofs.StableHLO

/-! # r34 Item D — pinning the ResNet-34 cotangent chain (the `CnnChainClose` analogue)

`ResNet34Close.lean` (Item C) certifies each r34 conv param output for *any* cotangent `c` at that
conv layer's output. This file pins `c` to the cotangent the **actual backward chain delivers** —
the r34 analogue of `CnnChainClose` (`cnn_render_conv{W,b}1/2_chain_certified`). The chain through one
**basic block** composes the rendered backward denotations: the block-output relu mask
(`selectPos`, `if a>0`), the per-channel BN input-VJP (`bnPerChannelTensor3_grad_input`, the
`bnPerChannelBack` token's denotation, faithful under `0<ε`), and the conv input-VJP
(`conv2d_has_vjp3` through the flatten bridge, the `convBack` token's denotation).

This file does the **identity block** (13 of r34's 16 blocks); the downsample block adds the strided
conv backward + the projection-path fan-in (`ResNet34ChainClose` could be extended with
`downBlockCot*` the same way — same mechanism, strided VJP). Per `planning/render_close_handoff.md`
§1: this **pins the cotangent**; the further `= ∂loss/∂θ` fold (composing the per-block cotangents
through all 16 blocks + maxpool + stem to the loss) stays separate, exactly as for the CNN. 3-axiom
clean (the BN backward's `0<ε` enters only via `bnPerChannelTensor3_grad_input_correct`, used in
Item C, not here — these are pure instantiations of the generic bridges).
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The cotangent the identity-block backward chain delivers at each conv output
--   block:  o = relu( addV( bn₂(conv₂(relu(bn₁(conv₁ x)))), x ) )
-- ════════════════════════════════════════════════════════════════

/-- Cotangent at **conv₂'s output** in an identity block, given the block-output cotangent `dyOut`,
    the block-output relu pre-activation `a` (= the `addV` output), and the saved conv₂ output `c2`
    (= the BN₂ input). `bn₂-back( relu'(a) ⊙ dyOut )` — the rendered backward (`selectPos` then
    `bnPerChannelBack`). The cotangent `conv2d_weight_grad`/`bias_grad` contract for `W₂`/`b₂`. -/
noncomputable def idBlockCotC2 {c h w : Nat} (ε : ℝ) (γ₂ : Vec c) (a c2 dyOut : Vec (c * h * w)) :
    Vec (c * h * w) :=
  bnPerChannelTensor3_grad_input c h w ε γ₂ c2 (fun i => if a i > 0 then dyOut i else 0)

/-- Cotangent at **conv₁'s output** in an identity block: continue the chain through conv₂-back
    (`conv2d_has_vjp3` via the flatten bridge) and the relu₁ mask (`if n₁ > 0`), then bn₁-back. The
    cotangent for `W₁`/`b₁`. Builds on `idBlockCotC2` exactly as `cnnChainCotW1` builds on
    `cnnChainCotW2`. -/
noncomputable def idBlockCotC1 {c h w kH kW : Nat} (ε : ℝ) (γ₁ γ₂ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (r1 : Vec (c * h * w))
    (a c1 c2 n1 dyOut : Vec (c * h * w)) : Vec (c * h * w) :=
  let cC2 := idBlockCotC2 ε γ₂ a c2 dyOut
  let cR1 := (hasVJP3_to_hasVJP (conv2d_has_vjp3 W₂ b₂)).backward r1 cC2
  bnPerChannelTensor3_grad_input c h w ε γ₁ c1 (fun i => if n1 i > 0 then cR1 i else 0)

-- ════════════════════════════════════════════════════════════════
-- § The chain-pinned identity-block conv closes — the generic bridges at the actual cotangents
-- ════════════════════════════════════════════════════════════════

/-- **Identity-block conv₂ weight, chain-certified.** `W₂ⁿ = W₂ − lr·(transpose-trick kernel grad)`
    denotes `W₂ − lr·(certified ∂conv₂/∂W₂ · the cotangent the block backward delivers at conv₂)` —
    `cnn_render_convW_certified` (Item C) instantiated at `idBlockCotC2`. -/
theorem idBlock_render_convW2_chain_certified {c h w kH kW : Nat}
    (b₂ : Vec c) (r1 : Tensor3 c h w) (ε : ℝ) (γ₂ : Vec c) (a c2 dyOut : Vec (c * h * w))
    (v : Vec (c * c * kH * kW)) (lr : ℝ) (idx : Fin (c * c * kH * kW)) :
    v idx - lr * (conv2d_weight_grad_has_vjp b₂ r1).backward v (idBlockCotC2 ε γ₂ a c2 dyOut) idx
      = v idx - lr * ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * c * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ r1)) v idx j
            * idBlockCotC2 ε γ₂ a c2 dyOut j :=
  cnn_render_convW_certified b₂ r1 v (idBlockCotC2 ε γ₂ a c2 dyOut) lr idx

/-- **Identity-block conv₂ bias, chain-certified.** -/
theorem idBlock_render_convb2_chain_certified {c h w kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (r1 : Tensor3 c h w) (b₂ : Vec c)
    (ε : ℝ) (γ₂ : Vec c) (a c2 dyOut : Vec (c * h * w)) (lr : ℝ) (o : Fin c) :
    b₂ o - lr * (conv2d_bias_grad_has_vjp W₂ r1).backward b₂ (idBlockCotC2 ε γ₂ a c2 dyOut) o
      = b₂ o - lr * ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c => Tensor3.flatten (conv2d W₂ b' r1)) b₂ o j
            * idBlockCotC2 ε γ₂ a c2 dyOut j :=
  cnn_render_convb_certified W₂ r1 b₂ (idBlockCotC2 ε γ₂ a c2 dyOut) lr o

/-- **Identity-block conv₁ weight, chain-certified.** `W₁ⁿ` denotes `W₁ − lr·(certified ∂conv₁/∂W₁ ·
    the deepest block cotangent)` — the generic bridge at `idBlockCotC1` (which crosses one more
    conv-back + relu mask + BN-back than `idBlockCotC2`). -/
theorem idBlock_render_convW1_chain_certified {c h w kH kW : Nat}
    (b₁ : Vec c) (xin : Tensor3 c h w) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (r1 : Vec (c * h * w))
    (ε : ℝ) (γ₁ γ₂ : Vec c) (a c1 c2 n1 dyOut : Vec (c * h * w))
    (v : Vec (c * c * kH * kW)) (lr : ℝ) (idx : Fin (c * c * kH * kW)) :
    v idx - lr * (conv2d_weight_grad_has_vjp b₁ xin).backward v
        (idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1 a c1 c2 n1 dyOut) idx
      = v idx - lr * ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * c * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b₁ xin)) v idx j
            * idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1 a c1 c2 n1 dyOut j :=
  cnn_render_convW_certified b₁ xin v (idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1 a c1 c2 n1 dyOut) lr idx

/-- **Identity-block conv₁ bias, chain-certified.** -/
theorem idBlock_render_convb1_chain_certified {c h w kH kW : Nat}
    (W₁ : Kernel4 c c kH kW) (xin : Tensor3 c h w) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (r1 : Vec (c * h * w))
    (ε : ℝ) (γ₁ γ₂ : Vec c) (a c1 c2 n1 dyOut : Vec (c * h * w)) (lr : ℝ) (o : Fin c) :
    b₁ o - lr * (conv2d_bias_grad_has_vjp W₁ xin).backward b₁
        (idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1 a c1 c2 n1 dyOut) o
      = b₁ o - lr * ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c => Tensor3.flatten (conv2d W₁ b' xin)) b₁ o j
            * idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1 a c1 c2 n1 dyOut j :=
  cnn_render_convb_certified W₁ xin b₁ (idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1 a c1 c2 n1 dyOut) lr o

-- ════════════════════════════════════════════════════════════════
-- § Downsample block — the strided main `W₁` + the strided projection `Wp`
--   block:  o = relu( addV( bn₂(conv₂(relu(bn₁(conv₁ˢ x)))), bnₚ(convₚˢ x) ) )
--   conv₂ is stride-1 (c→c) ⇒ its cotangent is `idBlockCotC2` and its weight close reuses the
--   identity-block theorems; conv₁ and convₚ are stride-2 (ic→c), so their weight closes use the
--   strided bridge (`mnv2_render_stem_conv{W,b}_certified`). The projection cotangent is
--   `idBlockCotC2` with `(γₚ, cp)` (`bnₚ-back( relu'(a) ⊙ dyOut )`); the main `conv₁` cotangent is
--   `idBlockCotC1` (conv₂ is stride-1, so the chain through it is identical to the identity block).
-- ════════════════════════════════════════════════════════════════

/-- **Downsample-block strided conv₁ weight, chain-certified.** `W₁ⁿ` (stride-2, `ic→c`) denotes
    `W₁ − lr·(certified ∂(flatConvStride2)/∂W₁ · idBlockCotC1)`. -/
theorem downBlock_render_convW1_chain_certified {ic c h w kH kW : Nat}
    (b₁ : Vec c) (xin : Vec (ic * (2 * h) * (2 * w)))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (r1 : Vec (c * h * w))
    (ε : ℝ) (γ₁ γ₂ : Vec c) (a c1 c2 n1 dyOut : Vec (c * h * w))
    (v : Vec (c * ic * 3 * 3)) (lr : ℝ) (i : Fin (c * ic * 3 * 3)) :
    v i - lr * (flatConvStride2_weight_grad_has_vjp b₁ xin).backward v
        (idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1 a c1 c2 n1 dyOut) i
      = v i - lr * ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * ic * 3 * 3) => flatConvStride2 (Kernel4.unflatten v') b₁ xin) v i j
            * idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1 a c1 c2 n1 dyOut j :=
  mnv2_render_stem_convW_certified b₁ xin v (idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1 a c1 c2 n1 dyOut) lr i

/-- **Downsample-block strided conv₁ bias, chain-certified.** -/
theorem downBlock_render_convb1_chain_certified {ic c h w kH kW : Nat}
    (W₁ : Kernel4 c ic 3 3) (xin : Vec (ic * (2 * h) * (2 * w)))
    (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (r1 : Vec (c * h * w))
    (ε : ℝ) (γ₁ γ₂ : Vec c) (a c1 c2 n1 dyOut : Vec (c * h * w)) (lr : ℝ) (o : Fin c) :
    b₁ o - lr * (flatConvStride2_bias_grad_has_vjp W₁ xin).backward b₁
        (idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1 a c1 c2 n1 dyOut) o
      = b₁ o - lr * ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c => flatConvStride2 W₁ b' xin) b₁ o j
            * idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1 a c1 c2 n1 dyOut j :=
  mnv2_render_stem_convb_certified W₁ xin b₁ (idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1 a c1 c2 n1 dyOut) lr o

/-- **Downsample-block strided projection `Wp` weight, chain-certified.** `Wpⁿ` (stride-2, `ic→c`)
    denotes `Wp − lr·(certified ∂(flatConvStride2)/∂Wp · bnₚ-back(relu'(a)⊙dyOut))`. The projection
    cotangent is `idBlockCotC2` with the projection's `(γₚ, cp)`. -/
theorem downBlock_render_convWp_chain_certified {ic c h w : Nat}
    (bp : Vec c) (xin : Vec (ic * (2 * h) * (2 * w)))
    (ε : ℝ) (γp : Vec c) (a cp dyOut : Vec (c * h * w))
    (v : Vec (c * ic * 3 * 3)) (lr : ℝ) (i : Fin (c * ic * 3 * 3)) :
    v i - lr * (flatConvStride2_weight_grad_has_vjp bp xin).backward v
        (idBlockCotC2 ε γp a cp dyOut) i
      = v i - lr * ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * ic * 3 * 3) => flatConvStride2 (Kernel4.unflatten v') bp xin) v i j
            * idBlockCotC2 ε γp a cp dyOut j :=
  mnv2_render_stem_convW_certified bp xin v (idBlockCotC2 ε γp a cp dyOut) lr i

/-- **Downsample-block strided projection `bp` bias, chain-certified.** -/
theorem downBlock_render_convbp_chain_certified {ic c h w : Nat}
    (Wp : Kernel4 c ic 3 3) (xin : Vec (ic * (2 * h) * (2 * w)))
    (bp : Vec c) (ε : ℝ) (γp : Vec c) (a cp dyOut : Vec (c * h * w)) (lr : ℝ) (o : Fin c) :
    bp o - lr * (flatConvStride2_bias_grad_has_vjp Wp xin).backward bp
        (idBlockCotC2 ε γp a cp dyOut) o
      = bp o - lr * ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c => flatConvStride2 Wp b' xin) bp o j
            * idBlockCotC2 ε γp a cp dyOut j :=
  mnv2_render_stem_convb_certified Wp xin bp (idBlockCotC2 ε γp a cp dyOut) lr o

-- (conv₂/b₂ of the downsample block are stride-1 c→c, so they reuse
-- `idBlock_render_convW2/b2_chain_certified` at `idBlockCotC2` — no separate theorem needed.)

-- ════════════════════════════════════════════════════════════════
-- § Stem — the 7×7 strided conv, whose cotangent comes through the maxpool backward
--   stem:  maxpool( relu( bn( convˢ x ) ) ),  feeding block 1.
-- ════════════════════════════════════════════════════════════════

/-- Cotangent at the **stem conv output** (@112), given the cotangent `cotPool` the first block
    delivers at the maxpool output (@56), the saved pre-pool stem-relu output `str` and stem-bn
    output `stn` (the relu pre-act) and stem-conv output `stc` (the BN input). The chain is
    `bn-back( relu'(stn) ⊙ maxpool-back(str, cotPool) )` — the maxpool backward is the
    `select_and_scatter` denotation `maxPoolBackFlat`. -/
noncomputable def stemCot {h w : Nat} (ε : ℝ) (γs : Vec 64)
    (str : Vec (64 * (2 * h) * (2 * w))) (stn stc : Vec (64 * (2 * h) * (2 * w)))
    (cotPool : Vec (64 * h * w)) : Vec (64 * (2 * h) * (2 * w)) :=
  bnPerChannelTensor3_grad_input 64 (2 * h) (2 * w) ε γs stc
    (fun i => if stn i > 0 then StableHLO.maxPoolBackFlat 64 h w str cotPool i else 0)

/-- **Stem 7×7 strided conv weight, chain-certified.** `sWⁿ` (7×7 stride-2, 3→64) denotes
    `sW − lr·(certified ∂(flatConvStride2)/∂sW · the cotangent through the maxpool backward)`. The
    deepest pin — the cotangent crosses the maxpool `select_and_scatter` and the stem relu/BN. -/
theorem stem_render_convW_chain_certified {h w : Nat}
    (bs : Vec 64) (x : Vec (3 * (2 * (2 * h)) * (2 * (2 * w))))
    (ε : ℝ) (γs : Vec 64) (str stn stc : Vec (64 * (2 * h) * (2 * w))) (cotPool : Vec (64 * h * w))
    (v : Vec (64 * 3 * 7 * 7)) (lr : ℝ) (i : Fin (64 * 3 * 7 * 7)) :
    v i - lr * (flatConvStride2_weight_grad_has_vjp bs x).backward v (stemCot ε γs str stn stc cotPool) i
      = v i - lr * ∑ j : Fin (64 * (2 * h) * (2 * w)),
          pdiv (fun v' : Vec (64 * 3 * 7 * 7) => flatConvStride2 (Kernel4.unflatten v') bs x) v i j
            * stemCot ε γs str stn stc cotPool j :=
  mnv2_render_stem_convW_certified bs x v (stemCot ε γs str stn stc cotPool) lr i

/-- **Stem 7×7 strided conv bias, chain-certified.** -/
theorem stem_render_convb_chain_certified {h w : Nat}
    (Ws : Kernel4 64 3 7 7) (x : Vec (3 * (2 * (2 * h)) * (2 * (2 * w))))
    (bs : Vec 64) (ε : ℝ) (γs : Vec 64) (str stn stc : Vec (64 * (2 * h) * (2 * w)))
    (cotPool : Vec (64 * h * w)) (lr : ℝ) (o : Fin 64) :
    bs o - lr * (flatConvStride2_bias_grad_has_vjp Ws x).backward bs (stemCot ε γs str stn stc cotPool) o
      = bs o - lr * ∑ j : Fin (64 * (2 * h) * (2 * w)),
          pdiv (fun b' : Vec 64 => flatConvStride2 Ws b' x) bs o j
            * stemCot ε γs str stn stc cotPool j :=
  mnv2_render_stem_convb_certified Ws x bs (stemCot ε γs str stn stc cotPool) lr o

end Proofs
