import LeanMlir.Proofs.ResNet34FaithfulPoC
import LeanMlir.Proofs.ResNet34ChainClose
import LeanMlir.Proofs.ResNet34RenderPC

/-! # PoC: the ResNet-34 §1a TIE — the whole net tied through the real forward

The Chapter-6 §1a tie: the cnn/cifar tie (`cnn_conv_tied_certified` / `cifar_conv_tied_certified`)
scaled to the full `[3,4,6,3]` ResNet-34. r34's §1 fold (`ResNet34FaithfulPoC`) already makes every
param op `den = certified ∀ c`; this file feeds each consumer the **real forward activations** and the
**loss-driven backward-chain cotangent** the residual net actually delivers — so the WHOLE 146-param
train step is den-composed forward→loss→backward, no free activations, no symbolic cotangent.

**The structural novelty vs cnn/cifar: residual skip-add fan-in sums.** Each block is
`relu(addV(F(x), skip))`; the block-output cotangent flows to BOTH the body `F` and the skip, and at
the block *input* the two contributions **sum** (`idBlockCotIn`/`downBlockCotIn`). So the backward is
not a linear chain — at every one of the 16 skip merges the cotangents add. This is the step cnn/cifar
(no residuals) never had; the per-block backward (`ResNet34ChainClose`) is reused, the cross-block
fan-in sum is the new content.

**Reuse, no new core ops, no new bridges.** Each block-type tie lemma is pure instantiation of the
generic `den = certified` lemmas at the `ResNet34ChainClose` chain cotangents:
* stride-1 block convs → `CifarPoC.convW_den`/`convB_den`;
* strided convs (downsample `W₁`, projection `Wp`, 7×7 stem) → `ResNet34PoC.convStridedW_den`/`convStridedB_den`;
* per-channel BN γ/β → `CifarBnPoC.bnGamma_den`/`bnBeta_den`;
* final dense → `Cifar8PoC.denseW_den`/`denseB_den`.

The block-type tie lemmas (`r34_idblock_tied` etc.) are proven once and applied at each of the 16
blocks in the whole-net capstone, threading the real `resnet34Forward_full_pc` activations.

## Honest residual (the boundary every prior fold carries)
* The block backward is rendered hand-written, so the cotangent SSA ↔ chain-cot correspondence is the
  per-op trust the whole suite carries; per-op `pretty` lexing; BN `0 < ε` smoothness; ℝ → Float32.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ResNet34PoC

/-! ## Identity block — all 8 params tied (the basic block `relu(F(x) + x)`)

Forward (per-channel BN): `o = relu( addV( bn₂(conv₂(relu(bn₁(conv₁ x)))), x ) )`. The block-output
cotangent `dyOut`, masked by the outer relu (`relu'(a)⊙dyOut`, `a` = the `addV` output), is the bn₂
output cotangent (and — via the identity skip — flows verbatim into the block-input sum). The conv
cotangents are `ResNet34ChainClose`'s `idBlockCotC2`/`idBlockCotC1`; the bn cotangents are the
relu-masked forms those defs are built from. -/

/-- **Identity block, tied.** All 8 params (conv₁/conv₂ `W`+`b`, bn₁/bn₂ `γ`+`β`) of an identity
    residual block denote the certified loss-descent step, at the real block forward activations and
    the block-backward chain cotangent driven by the block-output cotangent `dyOut`. -/
theorem r34_idblock_tied {c h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (ε : ℝ) (W₁ : Kernel4 c c 3 3) (b₁ : Vec c) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c 3 3) (b₂ : Vec c) (γ₂ β₂ : Vec c)
    (xin : Tensor3 c h w) (r1v : Vec (c*h*w))
    (c1 n1 c2 a : Vec (c*h*w)) (dyOut : Vec (c*h*w)) (lr : ℝ) :
    let r1 : Tensor3 c h w := Tensor3.unflatten r1v
    let dyBn2 : Vec (c*h*w) := fun i => if a i > 0 then dyOut i else 0
    let cotC2 : Vec (c*h*w) := idBlockCotC2 ε γ₂ a c2 dyOut
    let cR1 : Vec (c*h*w) := (hasVJP3_to_hasVJP (conv2d_has_vjp3 W₂ b₂)).backward r1v cotC2
    let dyBn1 : Vec (c*h*w) := fun i => if n1 i > 0 then cR1 i else 0
    let cotC1 : Vec (c*h*w) := idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1v a c1 c2 n1 dyOut
    -- conv₂ (stride-1, c→c)
    (∀ idx : Fin (c*c*3*3),
        den (SHlo.convWeightSgd xN wN lrStr b₂ r1 W₂ lr (.operand cotN cotC2)) idx
          = Kernel4.flatten W₂ idx - lr * ∑ j : Fin (c*h*w),
              pdiv (fun v' : Vec (c*c*3*3) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ r1))
                   (Kernel4.flatten W₂) idx j * cotC2 j)
  ∧ (∀ o : Fin c,
        den (SHlo.convBiasSgd bN lrStr W₂ r1 b₂ lr (.operand cotN cotC2)) o
          = b₂ o - lr * ∑ j : Fin (c*h*w),
              pdiv (fun b' : Vec c => Tensor3.flatten (conv2d W₂ b' r1)) b₂ o j * cotC2 j)
    -- bn₂ (γ₂/β₂)
  ∧ (∀ idx : Fin c,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γ₂ c2 lr (.operand cotN dyBn2)) idx
          = γ₂ idx - lr * ∑ j : Fin (c*(h*w)),
              pdiv (fun γ' : Vec c => bnPerChannelFlat c (h*w) ε γ' β₂ (reassocFwd c h w c2))
                   γ₂ idx j * reassocFwd c h w dyBn2 j)
  ∧ (∀ idx : Fin c,
        den (SHlo.bnBetaSgd bN lrStr β₂ lr (.operand cotN dyBn2)) idx
          = β₂ idx - lr * ∑ j : Fin (c*(h*w)),
              pdiv (fun β' : Vec c => bnPerChannelFlat c (h*w) ε γ₂ β' (reassocFwd c h w c2))
                   β₂ idx j * reassocFwd c h w dyBn2 j)
    -- conv₁ (stride-1, c→c)
  ∧ (∀ idx : Fin (c*c*3*3),
        den (SHlo.convWeightSgd xN wN lrStr b₁ xin W₁ lr (.operand cotN cotC1)) idx
          = Kernel4.flatten W₁ idx - lr * ∑ j : Fin (c*h*w),
              pdiv (fun v' : Vec (c*c*3*3) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₁ xin))
                   (Kernel4.flatten W₁) idx j * cotC1 j)
  ∧ (∀ o : Fin c,
        den (SHlo.convBiasSgd bN lrStr W₁ xin b₁ lr (.operand cotN cotC1)) o
          = b₁ o - lr * ∑ j : Fin (c*h*w),
              pdiv (fun b' : Vec c => Tensor3.flatten (conv2d W₁ b' xin)) b₁ o j * cotC1 j)
    -- bn₁ (γ₁/β₁)
  ∧ (∀ idx : Fin c,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γ₁ c1 lr (.operand cotN dyBn1)) idx
          = γ₁ idx - lr * ∑ j : Fin (c*(h*w)),
              pdiv (fun γ' : Vec c => bnPerChannelFlat c (h*w) ε γ' β₁ (reassocFwd c h w c1))
                   γ₁ idx j * reassocFwd c h w dyBn1 j)
  ∧ (∀ idx : Fin c,
        den (SHlo.bnBetaSgd bN lrStr β₁ lr (.operand cotN dyBn1)) idx
          = β₁ idx - lr * ∑ j : Fin (c*(h*w)),
              pdiv (fun β' : Vec c => bnPerChannelFlat c (h*w) ε γ₁ β' (reassocFwd c h w c1))
                   β₁ idx j * reassocFwd c h w dyBn1 j) := by
  intro r1 dyBn2 cotC2 cR1 dyBn1 cotC1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₂ r1 W₂ cotC2 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₂ r1 b₂ cotC2 lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γ₂ β₂ c2 dyBn2 lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γ₂ β₂ c2 dyBn2 lr idx
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₁ xin W₁ cotC1 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₁ xin b₁ cotC1 lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γ₁ β₁ c1 dyBn1 lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γ₁ β₁ c1 dyBn1 lr idx

/-! ## Downsample block — all 12 params tied (`relu(addV(body, proj))`, strided)

Forward: `o = relu( addV( bnₚ(convₚˢ x), bn₂(conv₂(relu(bn₁(conv₁ˢ x)))) ) )`, `ic→oc`, spatial halved.
conv₂ is stride-1 (`oc→oc`) so its cotangent + close are the identity-block ones (`idBlockCotC2`/
`idBlockCotC1`, `CifarPoC`); conv₁ and the projection convₚ are stride-2, so their weights use the
strided den lemma (`ResNet34PoC.convStridedW_den`). Both `bn₂` and `bnₚ` feed the SAME `addV`, so they
share the masked block cotangent `m = relu'(a)⊙dyOut`. -/

/-- **Downsample block, tied.** All 12 params (strided conv₁, stride-1 conv₂, strided projection convₚ,
    each with bias + per-channel BN γ/β) denote the certified step at the real forward + chain
    cotangent. -/
theorem r34_downblock_tied {ic oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (ε : ℝ) (W₁ : Kernel4 oc ic 3 3) (b₁ : Vec oc) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc 3 3) (b₂ : Vec oc) (γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic 3 3) (bp : Vec oc) (γp βp : Vec oc)
    (xin : Vec (ic*(2*h)*(2*w))) (r1v : Vec (oc*h*w))
    (c1 n1 c2 cp a : Vec (oc*h*w)) (dyOut : Vec (oc*h*w)) (lr : ℝ) :
    let r1 : Tensor3 oc h w := Tensor3.unflatten r1v
    let m : Vec (oc*h*w) := fun i => if a i > 0 then dyOut i else 0
    let cotC2 : Vec (oc*h*w) := idBlockCotC2 ε γ₂ a c2 dyOut
    let cR1 : Vec (oc*h*w) := (hasVJP3_to_hasVJP (conv2d_has_vjp3 W₂ b₂)).backward r1v cotC2
    let dyBn1 : Vec (oc*h*w) := fun i => if n1 i > 0 then cR1 i else 0
    let cotC1 : Vec (oc*h*w) := idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1v a c1 c2 n1 dyOut
    let cotCp : Vec (oc*h*w) := idBlockCotC2 ε γp a cp dyOut
    -- conv₂ (stride-1, oc→oc)
    (∀ idx : Fin (oc*oc*3*3),
        den (SHlo.convWeightSgd xN wN lrStr b₂ r1 W₂ lr (.operand cotN cotC2)) idx
          = Kernel4.flatten W₂ idx - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*oc*3*3) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ r1))
                   (Kernel4.flatten W₂) idx j * cotC2 j)
  ∧ (∀ o : Fin oc,
        den (SHlo.convBiasSgd bN lrStr W₂ r1 b₂ lr (.operand cotN cotC2)) o
          = b₂ o - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W₂ b' r1)) b₂ o j * cotC2 j)
    -- bn₂
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γ₂ c2 lr (.operand cotN m)) idx
          = γ₂ idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (h*w) ε γ' β₂ (reassocFwd oc h w c2))
                   γ₂ idx j * reassocFwd oc h w m j)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnBetaSgd bN lrStr β₂ lr (.operand cotN m)) idx
          = β₂ idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (h*w) ε γ₂ β' (reassocFwd oc h w c2))
                   β₂ idx j * reassocFwd oc h w m j)
    -- conv₁ (STRIDED, ic→oc)
  ∧ (∀ idx : Fin (oc*ic*3*3),
        den (SHlo.convStridedWeightSgd xN wN lrStr b₁ xin W₁ lr (.operand cotN cotC1)) idx
          = Kernel4.flatten W₁ idx - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*ic*3*3) => flatConvStride2 (Kernel4.unflatten v') b₁ xin)
                   (Kernel4.flatten W₁) idx j * cotC1 j)
  ∧ (∀ o : Fin oc,
        den (SHlo.convStridedBiasSgd bN lrStr W₁ xin b₁ lr (.operand cotN cotC1)) o
          = b₁ o - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun b' : Vec oc => flatConvStride2 W₁ b' xin) b₁ o j * cotC1 j)
    -- bn₁
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γ₁ c1 lr (.operand cotN dyBn1)) idx
          = γ₁ idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (h*w) ε γ' β₁ (reassocFwd oc h w c1))
                   γ₁ idx j * reassocFwd oc h w dyBn1 j)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnBetaSgd bN lrStr β₁ lr (.operand cotN dyBn1)) idx
          = β₁ idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (h*w) ε γ₁ β' (reassocFwd oc h w c1))
                   β₁ idx j * reassocFwd oc h w dyBn1 j)
    -- projection convₚ (STRIDED, ic→oc)
  ∧ (∀ idx : Fin (oc*ic*3*3),
        den (SHlo.convStridedWeightSgd xN wN lrStr bp xin Wp lr (.operand cotN cotCp)) idx
          = Kernel4.flatten Wp idx - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*ic*3*3) => flatConvStride2 (Kernel4.unflatten v') bp xin)
                   (Kernel4.flatten Wp) idx j * cotCp j)
  ∧ (∀ o : Fin oc,
        den (SHlo.convStridedBiasSgd bN lrStr Wp xin bp lr (.operand cotN cotCp)) o
          = bp o - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun b' : Vec oc => flatConvStride2 Wp b' xin) bp o j * cotCp j)
    -- bnₚ (shares the masked cotangent `m` with bn₂)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γp cp lr (.operand cotN m)) idx
          = γp idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (h*w) ε γ' βp (reassocFwd oc h w cp))
                   γp idx j * reassocFwd oc h w m j)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnBetaSgd bN lrStr βp lr (.operand cotN m)) idx
          = βp idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (h*w) ε γp β' (reassocFwd oc h w cp))
                   βp idx j * reassocFwd oc h w m j) := by
  intro r1 m cotC2 cR1 dyBn1 cotC1 cotCp
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₂ r1 W₂ cotC2 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₂ r1 b₂ cotC2 lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γ₂ β₂ c2 m lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γ₂ β₂ c2 m lr idx
  · intro idx; exact convStridedW_den xN wN lrStr cotN b₁ xin W₁ cotC1 lr idx
  · intro o;   exact convStridedB_den bN lrStr cotN W₁ xin b₁ cotC1 lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γ₁ β₁ c1 dyBn1 lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γ₁ β₁ c1 dyBn1 lr idx
  · intro idx; exact convStridedW_den xN wN lrStr cotN bp xin Wp cotCp lr idx
  · intro o;   exact convStridedB_den bN lrStr cotN Wp xin bp cotCp lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γp βp cp m lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γp βp cp m lr idx

/-! ## Stem — the 7×7 strided conv + BN (4 params), cotangent through the maxpool backward

Forward: `maxpool( relu( bn( convˢ x ) ) )`, feeding block 1. The cotangent the first block delivers at
the maxpool output (`cotPool`) lifts through the maxpool backward (`maxPoolBackFlat`, the
`select_and_scatter` denotation) + the stem relu/BN to `stemCot` at the conv output. -/

/-- **Stem, tied.** The 7×7 strided conv (`sW`/`sb`) + its BN (`γs`/`βs`) denote the certified step at
    the real stem forward + the cotangent through the maxpool backward. -/
theorem r34_stem_tied {h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (ε : ℝ) (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (γs βs : Vec 64)
    (x : Vec (3*(2*(2*h))*(2*(2*w))))
    (str stn stc : Vec (64*(2*h)*(2*w))) (cotPool : Vec (64*h*w)) (lr : ℝ) :
    let dyBnStem : Vec (64*(2*h)*(2*w)) :=
      fun i => if stn i > 0 then StableHLO.maxPoolBackFlat 64 h w str cotPool i else 0
    let cotStem : Vec (64*(2*h)*(2*w)) := stemCot ε γs str stn stc cotPool
    (∀ idx : Fin (64*3*7*7),
        den (SHlo.convStridedWeightSgd xN wN lrStr bs x Ws lr (.operand cotN cotStem)) idx
          = Kernel4.flatten Ws idx - lr * ∑ j : Fin (64*(2*h)*(2*w)),
              pdiv (fun v' : Vec (64*3*7*7) => flatConvStride2 (Kernel4.unflatten v') bs x)
                   (Kernel4.flatten Ws) idx j * cotStem j)
  ∧ (∀ o : Fin 64,
        den (SHlo.convStridedBiasSgd bN lrStr Ws x bs lr (.operand cotN cotStem)) o
          = bs o - lr * ∑ j : Fin (64*(2*h)*(2*w)),
              pdiv (fun b' : Vec 64 => flatConvStride2 Ws b' x) bs o j * cotStem j)
  ∧ (∀ idx : Fin 64,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γs stc lr (.operand cotN dyBnStem)) idx
          = γs idx - lr * ∑ j : Fin (64*((2*h)*(2*w))),
              pdiv (fun γ' : Vec 64 => bnPerChannelFlat 64 ((2*h)*(2*w)) ε γ' βs (reassocFwd 64 (2*h) (2*w) stc))
                   γs idx j * reassocFwd 64 (2*h) (2*w) dyBnStem j)
  ∧ (∀ idx : Fin 64,
        den (SHlo.bnBetaSgd bN lrStr βs lr (.operand cotN dyBnStem)) idx
          = βs idx - lr * ∑ j : Fin (64*((2*h)*(2*w))),
              pdiv (fun β' : Vec 64 => bnPerChannelFlat 64 ((2*h)*(2*w)) ε γs β' (reassocFwd 64 (2*h) (2*w) stc))
                   βs idx j * reassocFwd 64 (2*h) (2*w) dyBnStem j) := by
  intro dyBnStem cotStem
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro idx; exact convStridedW_den xN wN lrStr cotN bs x Ws cotStem lr idx
  · intro o;   exact convStridedB_den bN lrStr cotN Ws x bs cotStem lr o
  · intro idx; exact CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γs βs stc dyBnStem lr idx
  · intro idx; exact CifarBnPoC.bnBeta_den bN lrStr cotN ε γs βs stc dyBnStem lr idx

/-! ## The residual fan-in sum — the block-INPUT cotangent (r34's structural novelty)

`relu(addV(F(x), skip))`: the masked block-output cotangent flows to BOTH branches, and at the block
input the two contributions **sum**. These constructors give the cotangent the backward delivers at a
block's *input* — i.e. the previous (deeper-in-data-flow) block's `dyOut`. This is the cross-block
glue cnn/cifar (no residuals) never needed; the per-block backward is `ResNet34ChainClose`'s. -/

/-- Identity-block input cotangent = **skip branch** (the masked block-output cotangent `m`, identity
    backward) **+ body branch** (conv₁ input-VJP of the block's conv₁ cotangent `idBlockCotC1`). -/
noncomputable def idBlockCotIn {c h w : Nat} (ε : ℝ) (γ₁ γ₂ : Vec c)
    (W₁ : Kernel4 c c 3 3) (b₁ : Vec c) (W₂ : Kernel4 c c 3 3) (b₂ : Vec c)
    (xin r1v c1 n1 c2 a dyOut : Vec (c*h*w)) : Vec (c*h*w) :=
  fun i => (if a i > 0 then dyOut i else 0)
    + (hasVJP3_to_hasVJP (conv2d_has_vjp3 W₁ b₁)).backward xin
        (idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1v a c1 c2 n1 dyOut) i

/-- The identity-block input cotangent is the explicit fan-in sum
    `(relu'(a)⊙dyOut) + conv₁-back(idBlockCotC1)`. -/
theorem idBlockCotIn_eq {c h w : Nat} (ε : ℝ) (γ₁ γ₂ : Vec c)
    (W₁ : Kernel4 c c 3 3) (b₁ : Vec c) (W₂ : Kernel4 c c 3 3) (b₂ : Vec c)
    (xin r1v c1 n1 c2 a dyOut : Vec (c*h*w)) :
    idBlockCotIn ε γ₁ γ₂ W₁ b₁ W₂ b₂ xin r1v c1 n1 c2 a dyOut
      = fun i => (if a i > 0 then dyOut i else 0)
          + (hasVJP3_to_hasVJP (conv2d_has_vjp3 W₁ b₁)).backward xin
              (idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1v a c1 c2 n1 dyOut) i := rfl

/-- Downsample-block input cotangent = **projection branch** (strided `convₚ` input-VJP of `cotCp`)
    **+ body branch** (strided `conv₁ˢ` input-VJP of `idBlockCotC1`) — both at the higher resolution
    `2h×2w`. The projected-residual fan-in sum. -/
noncomputable def downBlockCotIn {ic oc h w : Nat} (ε : ℝ) (γ₁ γ₂ γp : Vec oc)
    (W₁ : Kernel4 oc ic 3 3) (b₁ : Vec oc) (W₂ : Kernel4 oc oc 3 3) (b₂ : Vec oc)
    (Wp : Kernel4 oc ic 3 3) (bp : Vec oc)
    (xin : Vec (ic*(2*h)*(2*w))) (r1v c1 n1 c2 cp a dyOut : Vec (oc*h*w)) :
    Vec (ic*(2*h)*(2*w)) :=
  fun i => (flatConvStride2_has_vjp W₁ b₁).backward xin
              (idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1v a c1 c2 n1 dyOut) i
    + (flatConvStride2_has_vjp Wp bp).backward xin (idBlockCotC2 ε γp a cp dyOut) i

/-- The downsample-block input cotangent is the explicit projected-residual fan-in sum. -/
theorem downBlockCotIn_eq {ic oc h w : Nat} (ε : ℝ) (γ₁ γ₂ γp : Vec oc)
    (W₁ : Kernel4 oc ic 3 3) (b₁ : Vec oc) (W₂ : Kernel4 oc oc 3 3) (b₂ : Vec oc)
    (Wp : Kernel4 oc ic 3 3) (bp : Vec oc)
    (xin : Vec (ic*(2*h)*(2*w))) (r1v c1 n1 c2 cp a dyOut : Vec (oc*h*w)) :
    downBlockCotIn ε γ₁ γ₂ γp W₁ b₁ W₂ b₂ Wp bp xin r1v c1 n1 c2 cp a dyOut
      = fun i => (flatConvStride2_has_vjp W₁ b₁).backward xin
                  (idBlockCotC1 ε γ₁ γ₂ W₂ b₂ r1v a c1 c2 n1 dyOut) i
        + (flatConvStride2_has_vjp Wp bp).backward xin (idBlockCotC2 ε γp a cp dyOut) i := rfl

/-! ## Loss cotangent + dense head (pin the top of the chain) -/

/-- **The emitted loss-cotangent graph denotes the softmax-CE gradient at the logits**
    (`softmax(logits) − onehot = ∂CE/∂logits`). Generic in `logits`; in the whole-net tie `logits` is
    `resnet34Forward_full_pc … x`. The r34 peer of `CifarPoC.cifarLossCot_den`. -/
theorem r34LossCot_den (nlogN ohN : String) (logits : Vec 10) (label : Fin 10) :
    den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe (.operand nlogN logits)))
          (.operand ohN (oneHot 10 label)))
      = fun j => softmax 10 logits j - oneHot 10 label j := by
  funext j; simp only [den, softmax]

/-- **Dense head weight `Wd`, tied to the WHOLE softmax-CE loss.** With the head input = the real GAP
    output `a_gap` and the cotangent the emitted loss graph denotes, the `weightSgd` for `Wd` denotes
    `Wd − lr·∂(crossEntropy ∘ dense)/∂Wd`. Generic in `a_gap` (the GAP output the forward delivers); the
    r34 peer of `CifarPoC.cifar_W7_tied_totalloss`, here for a single (GAP→dense) head layer. -/
theorem r34_dense_tied_totalloss (aN lrStr dyN : String)
    (Wd : Mat 512 10) (bd : Vec 10) (a_gap : Vec 512) (label : Fin 10)
    (lr : ℝ) (i : Fin 512) (j : Fin 10) :
    den (SHlo.weightSgd aN "%Wd" lrStr a_gap Wd lr
          (.operand dyN (fun k => softmax 10 (mnistLinear Wd bd a_gap) k - oneHot 10 label k)))
        (finProdFinEquiv (i, j))
      = Wd i j - lr * pdiv (fun v : Vec (512 * 10) => fun _ : Fin 1 =>
            crossEntropy 10 (dense (Mat.unflatten v) bd a_gap) label)
          (Mat.flatten Wd) (finProdFinEquiv (i, j)) 0 := by
  rw [Cifar8PoC.denseW_den aN "%Wd" lrStr dyN a_gap Wd bd
        (fun k => softmax 10 (mnistLinear Wd bd a_gap) k - oneHot 10 label k) lr i j,
      mlp_output_total_loss_grad Wd bd a_gap label i j]

/-- **Dense head bias `bd` = certified step.** -/
theorem r34_dense_bias_den (bN lrStr dyN : String)
    (Wd : Mat 512 10) (bd : Vec 10) (a_gap : Vec 512) (c : Vec 10) (lr : ℝ) (i : Fin 10) :
    den (SHlo.biasSgd bN lrStr bd lr (.operand dyN c)) i
      = bd i - lr * ∑ j : Fin 10, pdiv (fun b' : Vec 10 => dense Wd b' a_gap) bd i j * c j :=
  Cifar8PoC.denseB_den bN lrStr dyN Wd a_gap bd c lr i

end Proofs.ResNet34PoC
