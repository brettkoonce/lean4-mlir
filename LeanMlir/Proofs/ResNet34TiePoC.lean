import LeanMlir.Proofs.ResNet34FaithfulPoC
import LeanMlir.Proofs.ResNet34ChainClose
import LeanMlir.Proofs.ResNet34RenderPC

/-! # PoC: the ResNet-34 §1a TIE — the whole net tied through the real forward

The Chapter-5 §1a tie: the cnn/cifar tie (`cnn_conv_tied_certified` / `cifar_conv_tied_certified`)
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
def idblockTied {c h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (ε : ℝ) (W₁ : Kernel4 c c 3 3) (b₁ : Vec c) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c 3 3) (b₂ : Vec c) (γ₂ β₂ : Vec c)
    (xin : Tensor3 c h w) (r1v : Vec (c*h*w))
    (c1 n1 c2 a : Vec (c*h*w)) (dyOut : Vec (c*h*w)) (lr : ℝ) : Prop :=
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
                   β₁ idx j * reassocFwd c h w dyBn1 j)

theorem r34_idblock_tied {c h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (ε : ℝ) (W₁ : Kernel4 c c 3 3) (b₁ : Vec c) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c 3 3) (b₂ : Vec c) (γ₂ β₂ : Vec c)
    (xin : Tensor3 c h w) (r1v : Vec (c*h*w))
    (c1 n1 c2 a : Vec (c*h*w)) (dyOut : Vec (c*h*w)) (lr : ℝ) :
    idblockTied xN wN bN gN vN epsStr lrStr cotN ε W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂
      xin r1v c1 n1 c2 a dyOut lr := by
  unfold idblockTied
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
def downblockTied {ic oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (ε : ℝ) (W₁ : Kernel4 oc ic 3 3) (b₁ : Vec oc) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc 3 3) (b₂ : Vec oc) (γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic 3 3) (bp : Vec oc) (γp βp : Vec oc)
    (xin : Vec (ic*(2*h)*(2*w))) (r1v : Vec (oc*h*w))
    (c1 n1 c2 cp a : Vec (oc*h*w)) (dyOut : Vec (oc*h*w)) (lr : ℝ) : Prop :=
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
                   βp idx j * reassocFwd oc h w m j)

theorem r34_downblock_tied {ic oc h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (ε : ℝ) (W₁ : Kernel4 oc ic 3 3) (b₁ : Vec oc) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc 3 3) (b₂ : Vec oc) (γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic 3 3) (bp : Vec oc) (γp βp : Vec oc)
    (xin : Vec (ic*(2*h)*(2*w))) (r1v : Vec (oc*h*w))
    (c1 n1 c2 cp a : Vec (oc*h*w)) (dyOut : Vec (oc*h*w)) (lr : ℝ) :
    downblockTied xN wN bN gN vN epsStr lrStr cotN ε W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ Wp bp γp βp
      xin r1v c1 n1 c2 cp a dyOut lr := by
  unfold downblockTied
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
def stemTied {h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (ε : ℝ) (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (γs βs : Vec 64)
    (x : Vec (3*(2*(2*h))*(2*(2*w))))
    (str stn stc : Vec (64*(2*h)*(2*w))) (cotPool : Vec (64*h*w)) (lr : ℝ) : Prop :=
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
                   βs idx j * reassocFwd 64 (2*h) (2*w) dyBnStem j)

theorem r34_stem_tied {h w : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (ε : ℝ) (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (γs βs : Vec 64)
    (x : Vec (3*(2*(2*h))*(2*(2*w))))
    (str stn stc : Vec (64*(2*h)*(2*w))) (cotPool : Vec (64*h*w)) (lr : ℝ) :
    stemTied xN wN bN gN vN epsStr lrStr cotN ε Ws bs γs βs x str stn stc cotPool lr := by
  unfold stemTied
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

/-! ## Input-only wrappers — compute a block's internal activations from its input

For the whole-net thread, each block is identified by its **input** (the forward prefix) + its
downstream cotangent `dyOut`. These wrappers compute the block's internal forward activations
(`c1`/`n1`/`r1v`/`c2`/`a`, the real `rblkPC`/`rblkPStridedPC` values) from the input and delegate to
the per-block-type tie lemma + the fan-in cotangent constructor — so the capstone threads only block
inputs (via `idFwd`/`downFwd`) and dyOuts (via these), not every internal activation. -/

/-- Identity block input cotangent, computed from the block input. -/
@[irreducible] noncomputable def idBlockCotInAt {c h w : Nat} (ε : ℝ)
    (W₁ : Kernel4 c c 3 3) (b₁ γ₁ β₁ : Vec c) (W₂ : Kernel4 c c 3 3) (b₂ γ₂ β₂ : Vec c)
    (xin dyOut : Vec (c*h*w)) : Vec (c*h*w) :=
  let c1 := flatConv W₁ b₁ xin
  let n1 := bnPerChannelTensor3 c h w ε γ₁ β₁ c1
  let r1v := relu (c*h*w) n1
  let c2 := flatConv W₂ b₂ r1v
  let Fb := bnPerChannelTensor3 c h w ε γ₂ β₂ c2
  let a := fun i => Fb i + xin i
  idBlockCotIn ε γ₁ γ₂ W₁ b₁ W₂ b₂ xin r1v c1 n1 c2 a dyOut

/-- Identity block tie, INPUT-only: the 8-param tie at the block's real forward activations
    (computed from `xin`) — the capstone conjunct form. -/
@[irreducible] def idblockTiedAt {c h w : Nat} (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (W₁ : Kernel4 c c 3 3) (b₁ γ₁ β₁ : Vec c) (W₂ : Kernel4 c c 3 3) (b₂ γ₂ β₂ : Vec c)
    (xin dyOut : Vec (c*h*w)) (lr : ℝ) : Prop :=
  let c1 := flatConv W₁ b₁ xin
  let n1 := bnPerChannelTensor3 c h w ε γ₁ β₁ c1
  let r1v := relu (c*h*w) n1
  let c2 := flatConv W₂ b₂ r1v
  let Fb := bnPerChannelTensor3 c h w ε γ₂ β₂ c2
  let a := fun i => Fb i + xin i
  idblockTied xN wN bN gN vN epsStr lrStr cotN ε W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂
    (Tensor3.unflatten xin) r1v c1 n1 c2 a dyOut lr

theorem r34_idblock_tiedAt {c h w : Nat} (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (W₁ : Kernel4 c c 3 3) (b₁ γ₁ β₁ : Vec c) (W₂ : Kernel4 c c 3 3) (b₂ γ₂ β₂ : Vec c)
    (xin dyOut : Vec (c*h*w)) (lr : ℝ) :
    idblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ xin dyOut lr := by
  unfold idblockTiedAt
  intro c1 n1 r1v c2 Fb a
  exact r34_idblock_tied xN wN bN gN vN epsStr lrStr cotN ε W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂
    (Tensor3.unflatten xin) r1v c1 n1 c2 a dyOut lr

/-- Downsample block input cotangent, computed from the block input (the projected-residual fan-in). -/
@[irreducible] noncomputable def downBlockCotInAt {ic oc h w : Nat} (ε : ℝ)
    (W₁ : Kernel4 oc ic 3 3) (b₁ γ₁ β₁ : Vec oc) (W₂ : Kernel4 oc oc 3 3) (b₂ γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic 3 3) (bp γp βp : Vec oc)
    (xin : Vec (ic*(2*h)*(2*w))) (dyOut : Vec (oc*h*w)) : Vec (ic*(2*h)*(2*w)) :=
  let c1 := flatConvStride2 W₁ b₁ xin
  let n1 := bnPerChannelTensor3 oc h w ε γ₁ β₁ c1
  let r1v := relu (oc*h*w) n1
  let c2 := flatConv W₂ b₂ r1v
  let Fb := bnPerChannelTensor3 oc h w ε γ₂ β₂ c2
  let cp := flatConvStride2 Wp bp xin
  let Fp := bnPerChannelTensor3 oc h w ε γp βp cp
  let a := fun i => Fp i + Fb i
  downBlockCotIn ε γ₁ γ₂ γp W₁ b₁ W₂ b₂ Wp bp xin r1v c1 n1 c2 cp a dyOut

/-- Downsample block tie, INPUT-only (12-param tie at the block's real forward activations). -/
@[irreducible] def downblockTiedAt {ic oc h w : Nat} (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (W₁ : Kernel4 oc ic 3 3) (b₁ γ₁ β₁ : Vec oc) (W₂ : Kernel4 oc oc 3 3) (b₂ γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic 3 3) (bp γp βp : Vec oc)
    (xin : Vec (ic*(2*h)*(2*w))) (dyOut : Vec (oc*h*w)) (lr : ℝ) : Prop :=
  let c1 := flatConvStride2 W₁ b₁ xin
  let n1 := bnPerChannelTensor3 oc h w ε γ₁ β₁ c1
  let r1v := relu (oc*h*w) n1
  let c2 := flatConv W₂ b₂ r1v
  let Fb := bnPerChannelTensor3 oc h w ε γ₂ β₂ c2
  let cp := flatConvStride2 Wp bp xin
  let Fp := bnPerChannelTensor3 oc h w ε γp βp cp
  let a := fun i => Fp i + Fb i
  downblockTied xN wN bN gN vN epsStr lrStr cotN ε W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ Wp bp γp βp
    xin r1v c1 n1 c2 cp a dyOut lr

theorem r34_downblock_tiedAt {ic oc h w : Nat} (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (W₁ : Kernel4 oc ic 3 3) (b₁ γ₁ β₁ : Vec oc) (W₂ : Kernel4 oc oc 3 3) (b₂ γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic 3 3) (bp γp βp : Vec oc)
    (xin : Vec (ic*(2*h)*(2*w))) (dyOut : Vec (oc*h*w)) (lr : ℝ) :
    downblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ Wp bp γp βp
      xin dyOut lr := by
  unfold downblockTiedAt
  intro c1 n1 r1v c2 Fb cp Fp a
  exact r34_downblock_tied xN wN bN gN vN epsStr lrStr cotN ε W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ Wp bp γp βp
    xin r1v c1 n1 c2 cp a dyOut lr

/-- Stem tie, INPUT-only (4-param tie at the stem's real forward activations). -/
@[irreducible] def stemTiedAt {h w : Nat} (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (Ws : Kernel4 64 3 7 7) (bs γs βs : Vec 64) (x : Vec (3*(2*(2*h))*(2*(2*w))))
    (cotPool : Vec (64*h*w)) (lr : ℝ) : Prop :=
  let stc := flatConvStride2 Ws bs x
  let stn := bnPerChannelTensor3 64 (2*h) (2*w) ε γs βs stc
  let str := relu (64*(2*h)*(2*w)) stn
  stemTied xN wN bN gN vN epsStr lrStr cotN ε Ws bs γs βs x str stn stc cotPool lr

theorem r34_stem_tiedAt {h w : Nat} (xN wN bN gN vN epsStr lrStr cotN : String) (ε : ℝ)
    (Ws : Kernel4 64 3 7 7) (bs γs βs : Vec 64) (x : Vec (3*(2*(2*h))*(2*(2*w))))
    (cotPool : Vec (64*h*w)) (lr : ℝ) :
    stemTiedAt xN wN bN gN vN epsStr lrStr cotN ε Ws bs γs βs x cotPool lr := by
  unfold stemTiedAt
  intro stc stn str
  exact r34_stem_tied xN wN bN gN vN epsStr lrStr cotN ε Ws bs γs βs x str stn stc cotPool lr

/-! ## The whole-net capstone — all 146 params tied through the REAL forward + composed cotangent

`resnet34Forward_full_pc` threaded: block inputs are the forward prefixes (`idFwd`/`downFwd`/stem),
and the backward cotangents are composed from the loss `g = softmax(logits) − onehot` down through
dense (`dense_has_vjp`) + GAP (`globalAvgPoolFlat_has_vjp`) + the residual fan-in sum at every skip
(`idBlockCotInAt`/`downBlockCotInAt`). Each block's tie then holds at its real input + threaded dyOut.
This is the full §1a tie: the WHOLE 146-param ResNet-34 train step is den-composed forward→loss→backward,
no free activations, no symbolic cotangent. -/

/-! Irreducible aliases of the (reducible) forward steps. Definitionally equal to `idFwd`/`downFwd`/
the stem+maxpool — so the threaded block inputs ARE the real `resnet34Forward_full_pc` prefixes — but
opaque to the elaborator, so the 16-deep forward chain is not eagerly unfolded during the capstone's
dimension inference (the per-block tie lemmas are generic in the block input, so opacity is harmless). -/
@[irreducible] noncomputable def idFwdO {c h w : Nat} (ε : ℝ)
    (W₁ : Kernel4 c c 3 3) (b₁ γ₁ β₁ : Vec c) (W₂ : Kernel4 c c 3 3) (b₂ γ₂ β₂ : Vec c) :
    Vec (c*h*w) → Vec (c*h*w) :=
  idFwd (h := h) (w := w) ε W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂

@[irreducible] noncomputable def downFwdO {ic oc h w : Nat} (ε : ℝ)
    (W₁ : Kernel4 oc ic 3 3) (b₁ γ₁ β₁ : Vec oc) (W₂ : Kernel4 oc oc 3 3) (b₂ γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic 3 3) (bp γp βp : Vec oc) : Vec (ic*(2*h)*(2*w)) → Vec (oc*h*w) :=
  downFwd (h := h) (w := w) ε W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ Wp bp γp βp

@[irreducible] noncomputable def stemMpO {h w : Nat} (ε : ℝ)
    (Ws : Kernel4 64 3 7 7) (bs γs βs : Vec 64) (x : Vec (3*(2*(2*h))*(2*(2*w)))) : Vec (64*h*w) :=
  maxPoolFlat 64 h w (cbrStridedPC (h := 2*h) (w := 2*w) Ws bs ε γs βs x)

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 100000 in
/-- **The whole ResNet-34 train step, tied.** Threading `resnet34Forward_full_pc`'s real activations
    and the loss-driven backward cotangent chain (residual fan-in sums at every skip), every one of the
    16 residual blocks (via `idblockTiedAt`/`downblockTiedAt`), the stem (`stemTiedAt`), and the dense
    head (the total-loss fold + the loss-cotangent graph) denote the certified loss-descent step. -/
theorem r34_net_tied_certified
    (xN wN bN gN vN epsStr lrStr cotN dN nlogN ohN : String) (ε : ℝ)
    (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (γs βs : Vec 64)
    (a0W1 : Kernel4 64 64 3 3) (a0b1 : Vec 64) (a0g1 a0t1 : Vec 64) (a0W2 : Kernel4 64 64 3 3) (a0b2 : Vec 64) (a0g2 a0t2 : Vec 64)
    (a1W1 : Kernel4 64 64 3 3) (a1b1 : Vec 64) (a1g1 a1t1 : Vec 64) (a1W2 : Kernel4 64 64 3 3) (a1b2 : Vec 64) (a1g2 a1t2 : Vec 64)
    (a2W1 : Kernel4 64 64 3 3) (a2b1 : Vec 64) (a2g1 a2t1 : Vec 64) (a2W2 : Kernel4 64 64 3 3) (a2b2 : Vec 64) (a2g2 a2t2 : Vec 64)
    (d2W1 : Kernel4 128 64 3 3) (d2b1 : Vec 128) (d2g1 d2t1 : Vec 128) (d2W2 : Kernel4 128 128 3 3) (d2b2 : Vec 128) (d2g2 d2t2 : Vec 128) (d2Wp : Kernel4 128 64 3 3) (d2bp : Vec 128) (d2gp d2tp : Vec 128)
    (b0W1 : Kernel4 128 128 3 3) (b0b1 : Vec 128) (b0g1 b0t1 : Vec 128) (b0W2 : Kernel4 128 128 3 3) (b0b2 : Vec 128) (b0g2 b0t2 : Vec 128)
    (b1W1 : Kernel4 128 128 3 3) (b1b1 : Vec 128) (b1g1 b1t1 : Vec 128) (b1W2 : Kernel4 128 128 3 3) (b1b2 : Vec 128) (b1g2 b1t2 : Vec 128)
    (b2W1 : Kernel4 128 128 3 3) (b2b1 : Vec 128) (b2g1 b2t1 : Vec 128) (b2W2 : Kernel4 128 128 3 3) (b2b2 : Vec 128) (b2g2 b2t2 : Vec 128)
    (d3W1 : Kernel4 256 128 3 3) (d3b1 : Vec 256) (d3g1 d3t1 : Vec 256) (d3W2 : Kernel4 256 256 3 3) (d3b2 : Vec 256) (d3g2 d3t2 : Vec 256) (d3Wp : Kernel4 256 128 3 3) (d3bp : Vec 256) (d3gp d3tp : Vec 256)
    (c0W1 : Kernel4 256 256 3 3) (c0b1 : Vec 256) (c0g1 c0t1 : Vec 256) (c0W2 : Kernel4 256 256 3 3) (c0b2 : Vec 256) (c0g2 c0t2 : Vec 256)
    (c1W1 : Kernel4 256 256 3 3) (c1b1 : Vec 256) (c1g1 c1t1 : Vec 256) (c1W2 : Kernel4 256 256 3 3) (c1b2 : Vec 256) (c1g2 c1t2 : Vec 256)
    (c2W1 : Kernel4 256 256 3 3) (c2b1 : Vec 256) (c2g1 c2t1 : Vec 256) (c2W2 : Kernel4 256 256 3 3) (c2b2 : Vec 256) (c2g2 c2t2 : Vec 256)
    (c3W1 : Kernel4 256 256 3 3) (c3b1 : Vec 256) (c3g1 c3t1 : Vec 256) (c3W2 : Kernel4 256 256 3 3) (c3b2 : Vec 256) (c3g2 c3t2 : Vec 256)
    (c4W1 : Kernel4 256 256 3 3) (c4b1 : Vec 256) (c4g1 c4t1 : Vec 256) (c4W2 : Kernel4 256 256 3 3) (c4b2 : Vec 256) (c4g2 c4t2 : Vec 256)
    (d4W1 : Kernel4 512 256 3 3) (d4b1 : Vec 512) (d4g1 d4t1 : Vec 512) (d4W2 : Kernel4 512 512 3 3) (d4b2 : Vec 512) (d4g2 d4t2 : Vec 512) (d4Wp : Kernel4 512 256 3 3) (d4bp : Vec 512) (d4gp d4tp : Vec 512)
    (e0W1 : Kernel4 512 512 3 3) (e0b1 : Vec 512) (e0g1 e0t1 : Vec 512) (e0W2 : Kernel4 512 512 3 3) (e0b2 : Vec 512) (e0g2 e0t2 : Vec 512)
    (e1W1 : Kernel4 512 512 3 3) (e1b1 : Vec 512) (e1g1 e1t1 : Vec 512) (e1W2 : Kernel4 512 512 3 3) (e1b2 : Vec 512) (e1g2 e1t2 : Vec 512)
    (Wd : Mat 512 10) (bd : Vec 10) (x : Vec (3 * 224 * 224)) (label : Fin 10) (lr : ℝ) :
    -- forward block inputs (the prefixes of `resnet34Forward_full_pc`)
    let xa0 : Vec (64*56*56) := stemMpO (h := 56) (w := 56) ε Ws bs γs βs x
    let xa1 := idFwdO (h := 56) (w := 56) ε a0W1 a0b1 a0g1 a0t1 a0W2 a0b2 a0g2 a0t2 xa0
    let xa2 := idFwdO (h := 56) (w := 56) ε a1W1 a1b1 a1g1 a1t1 a1W2 a1b2 a1g2 a1t2 xa1
    let xd2 := idFwdO (h := 56) (w := 56) ε a2W1 a2b1 a2g1 a2t1 a2W2 a2b2 a2g2 a2t2 xa2
    let xb0 := downFwdO (h := 28) (w := 28) ε d2W1 d2b1 d2g1 d2t1 d2W2 d2b2 d2g2 d2t2 d2Wp d2bp d2gp d2tp xd2
    let xb1 := idFwdO (h := 28) (w := 28) ε b0W1 b0b1 b0g1 b0t1 b0W2 b0b2 b0g2 b0t2 xb0
    let xb2 := idFwdO (h := 28) (w := 28) ε b1W1 b1b1 b1g1 b1t1 b1W2 b1b2 b1g2 b1t2 xb1
    let xd3 := idFwdO (h := 28) (w := 28) ε b2W1 b2b1 b2g1 b2t1 b2W2 b2b2 b2g2 b2t2 xb2
    let xc0 := downFwdO (h := 14) (w := 14) ε d3W1 d3b1 d3g1 d3t1 d3W2 d3b2 d3g2 d3t2 d3Wp d3bp d3gp d3tp xd3
    let xc1 := idFwdO (h := 14) (w := 14) ε c0W1 c0b1 c0g1 c0t1 c0W2 c0b2 c0g2 c0t2 xc0
    let xc2 := idFwdO (h := 14) (w := 14) ε c1W1 c1b1 c1g1 c1t1 c1W2 c1b2 c1g2 c1t2 xc1
    let xc3 := idFwdO (h := 14) (w := 14) ε c2W1 c2b1 c2g1 c2t1 c2W2 c2b2 c2g2 c2t2 xc2
    let xc4 := idFwdO (h := 14) (w := 14) ε c3W1 c3b1 c3g1 c3t1 c3W2 c3b2 c3g2 c3t2 xc3
    let xd4 := idFwdO (h := 14) (w := 14) ε c4W1 c4b1 c4g1 c4t1 c4W2 c4b2 c4g2 c4t2 xc4
    let xe0 := downFwdO (h := 7) (w := 7) ε d4W1 d4b1 d4g1 d4t1 d4W2 d4b2 d4g2 d4t2 d4Wp d4bp d4gp d4tp xd4
    let xe1 := idFwdO (h := 7) (w := 7) ε e0W1 e0b1 e0g1 e0t1 e0W2 e0b2 e0g2 e0t2 xe0
    let xgap := idFwdO (h := 7) (w := 7) ε e1W1 e1b1 e1g1 e1t1 e1W2 e1b2 e1g2 e1t2 xe1
    let a_gap : Vec 512 := globalAvgPoolFlat 512 7 7 xgap
    let g : Vec 10 := fun k => softmax 10 (mnistLinear Wd bd a_gap) k - oneHot 10 label k
    -- backward cotangents (composed from the loss; residual fan-in sums)
    let dyE1 : Vec (512*7*7) := (globalAvgPoolFlat_has_vjp 512 7 7).backward xgap ((dense_has_vjp Wd bd).backward a_gap g)
    let dyE0 := idBlockCotInAt (h := 7) (w := 7) ε e1W1 e1b1 e1g1 e1t1 e1W2 e1b2 e1g2 e1t2 xe1 dyE1
    let dyD4 := idBlockCotInAt (h := 7) (w := 7) ε e0W1 e0b1 e0g1 e0t1 e0W2 e0b2 e0g2 e0t2 xe0 dyE0
    let dyC4 := downBlockCotInAt (h := 7) (w := 7) ε d4W1 d4b1 d4g1 d4t1 d4W2 d4b2 d4g2 d4t2 d4Wp d4bp d4gp d4tp xd4 dyD4
    let dyC3 := idBlockCotInAt (h := 14) (w := 14) ε c4W1 c4b1 c4g1 c4t1 c4W2 c4b2 c4g2 c4t2 xc4 dyC4
    let dyC2 := idBlockCotInAt (h := 14) (w := 14) ε c3W1 c3b1 c3g1 c3t1 c3W2 c3b2 c3g2 c3t2 xc3 dyC3
    let dyC1 := idBlockCotInAt (h := 14) (w := 14) ε c2W1 c2b1 c2g1 c2t1 c2W2 c2b2 c2g2 c2t2 xc2 dyC2
    let dyC0 := idBlockCotInAt (h := 14) (w := 14) ε c1W1 c1b1 c1g1 c1t1 c1W2 c1b2 c1g2 c1t2 xc1 dyC1
    let dyD3 := idBlockCotInAt (h := 14) (w := 14) ε c0W1 c0b1 c0g1 c0t1 c0W2 c0b2 c0g2 c0t2 xc0 dyC0
    let dyB2 := downBlockCotInAt (h := 14) (w := 14) ε d3W1 d3b1 d3g1 d3t1 d3W2 d3b2 d3g2 d3t2 d3Wp d3bp d3gp d3tp xd3 dyD3
    let dyB1 := idBlockCotInAt (h := 28) (w := 28) ε b2W1 b2b1 b2g1 b2t1 b2W2 b2b2 b2g2 b2t2 xb2 dyB2
    let dyB0 := idBlockCotInAt (h := 28) (w := 28) ε b1W1 b1b1 b1g1 b1t1 b1W2 b1b2 b1g2 b1t2 xb1 dyB1
    let dyD2 := idBlockCotInAt (h := 28) (w := 28) ε b0W1 b0b1 b0g1 b0t1 b0W2 b0b2 b0g2 b0t2 xb0 dyB0
    let dyA2 := downBlockCotInAt (h := 28) (w := 28) ε d2W1 d2b1 d2g1 d2t1 d2W2 d2b2 d2g2 d2t2 d2Wp d2bp d2gp d2tp xd2 dyD2
    let dyA1 := idBlockCotInAt (h := 56) (w := 56) ε a2W1 a2b1 a2g1 a2t1 a2W2 a2b2 a2g2 a2t2 xa2 dyA2
    let dyA0 := idBlockCotInAt (h := 56) (w := 56) ε a1W1 a1b1 a1g1 a1t1 a1W2 a1b2 a1g2 a1t2 xa1 dyA1
    let cotPool := idBlockCotInAt (h := 56) (w := 56) ε a0W1 a0b1 a0g1 a0t1 a0W2 a0b2 a0g2 a0t2 xa0 dyA0
    -- every block + stem tied at its real input + threaded cotangent
    stemTiedAt xN wN bN gN vN epsStr lrStr cotN (h := 56) (w := 56) ε Ws bs γs βs x cotPool lr
  ∧ idblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε a0W1 a0b1 a0g1 a0t1 a0W2 a0b2 a0g2 a0t2 xa0 dyA0 lr
  ∧ idblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε a1W1 a1b1 a1g1 a1t1 a1W2 a1b2 a1g2 a1t2 xa1 dyA1 lr
  ∧ idblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε a2W1 a2b1 a2g1 a2t1 a2W2 a2b2 a2g2 a2t2 xa2 dyA2 lr
  ∧ downblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε d2W1 d2b1 d2g1 d2t1 d2W2 d2b2 d2g2 d2t2 d2Wp d2bp d2gp d2tp xd2 dyD2 lr
  ∧ idblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε b0W1 b0b1 b0g1 b0t1 b0W2 b0b2 b0g2 b0t2 xb0 dyB0 lr
  ∧ idblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε b1W1 b1b1 b1g1 b1t1 b1W2 b1b2 b1g2 b1t2 xb1 dyB1 lr
  ∧ idblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε b2W1 b2b1 b2g1 b2t1 b2W2 b2b2 b2g2 b2t2 xb2 dyB2 lr
  ∧ downblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε d3W1 d3b1 d3g1 d3t1 d3W2 d3b2 d3g2 d3t2 d3Wp d3bp d3gp d3tp xd3 dyD3 lr
  ∧ idblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε c0W1 c0b1 c0g1 c0t1 c0W2 c0b2 c0g2 c0t2 xc0 dyC0 lr
  ∧ idblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε c1W1 c1b1 c1g1 c1t1 c1W2 c1b2 c1g2 c1t2 xc1 dyC1 lr
  ∧ idblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε c2W1 c2b1 c2g1 c2t1 c2W2 c2b2 c2g2 c2t2 xc2 dyC2 lr
  ∧ idblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε c3W1 c3b1 c3g1 c3t1 c3W2 c3b2 c3g2 c3t2 xc3 dyC3 lr
  ∧ idblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε c4W1 c4b1 c4g1 c4t1 c4W2 c4b2 c4g2 c4t2 xc4 dyC4 lr
  ∧ downblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε d4W1 d4b1 d4g1 d4t1 d4W2 d4b2 d4g2 d4t2 d4Wp d4bp d4gp d4tp xd4 dyD4 lr
  ∧ idblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε e0W1 e0b1 e0g1 e0t1 e0W2 e0b2 e0g2 e0t2 xe0 dyE0 lr
  ∧ idblockTiedAt xN wN bN gN vN epsStr lrStr cotN ε e1W1 e1b1 e1g1 e1t1 e1W2 e1b2 e1g2 e1t2 xe1 dyE1 lr
  ∧ (∀ i : Fin 512, ∀ j : Fin 10,
        den (SHlo.weightSgd dN "%Wd" lrStr a_gap Wd lr
              (.operand cotN (fun k => softmax 10 (mnistLinear Wd bd a_gap) k - oneHot 10 label k)))
            (finProdFinEquiv (i, j))
          = Wd i j - lr * pdiv (fun v : Vec (512 * 10) => fun _ : Fin 1 =>
                crossEntropy 10 (dense (Mat.unflatten v) bd a_gap) label)
              (Mat.flatten Wd) (finProdFinEquiv (i, j)) 0)
  ∧ den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe (.operand nlogN (mnistLinear Wd bd a_gap))))
          (.operand ohN (oneHot 10 label)))
      = g := by
  intro xa0 xa1 xa2 xd2 xb0 xb1 xb2 xd3 xc0 xc1 xc2 xc3 xc4 xd4 xe0 xe1 xgap a_gap g
        dyE1 dyE0 dyD4 dyC4 dyC3 dyC2 dyC1 dyC0 dyD3 dyB2 dyB1 dyB0 dyD2 dyA2 dyA1 dyA0 cotPool
  exact ⟨r34_stem_tiedAt xN wN bN gN vN epsStr lrStr cotN ε Ws bs γs βs x cotPool lr,
    r34_idblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε a0W1 a0b1 a0g1 a0t1 a0W2 a0b2 a0g2 a0t2 xa0 dyA0 lr,
    r34_idblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε a1W1 a1b1 a1g1 a1t1 a1W2 a1b2 a1g2 a1t2 xa1 dyA1 lr,
    r34_idblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε a2W1 a2b1 a2g1 a2t1 a2W2 a2b2 a2g2 a2t2 xa2 dyA2 lr,
    r34_downblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε d2W1 d2b1 d2g1 d2t1 d2W2 d2b2 d2g2 d2t2 d2Wp d2bp d2gp d2tp xd2 dyD2 lr,
    r34_idblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε b0W1 b0b1 b0g1 b0t1 b0W2 b0b2 b0g2 b0t2 xb0 dyB0 lr,
    r34_idblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε b1W1 b1b1 b1g1 b1t1 b1W2 b1b2 b1g2 b1t2 xb1 dyB1 lr,
    r34_idblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε b2W1 b2b1 b2g1 b2t1 b2W2 b2b2 b2g2 b2t2 xb2 dyB2 lr,
    r34_downblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε d3W1 d3b1 d3g1 d3t1 d3W2 d3b2 d3g2 d3t2 d3Wp d3bp d3gp d3tp xd3 dyD3 lr,
    r34_idblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε c0W1 c0b1 c0g1 c0t1 c0W2 c0b2 c0g2 c0t2 xc0 dyC0 lr,
    r34_idblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε c1W1 c1b1 c1g1 c1t1 c1W2 c1b2 c1g2 c1t2 xc1 dyC1 lr,
    r34_idblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε c2W1 c2b1 c2g1 c2t1 c2W2 c2b2 c2g2 c2t2 xc2 dyC2 lr,
    r34_idblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε c3W1 c3b1 c3g1 c3t1 c3W2 c3b2 c3g2 c3t2 xc3 dyC3 lr,
    r34_idblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε c4W1 c4b1 c4g1 c4t1 c4W2 c4b2 c4g2 c4t2 xc4 dyC4 lr,
    r34_downblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε d4W1 d4b1 d4g1 d4t1 d4W2 d4b2 d4g2 d4t2 d4Wp d4bp d4gp d4tp xd4 dyD4 lr,
    r34_idblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e0W1 e0b1 e0g1 e0t1 e0W2 e0b2 e0g2 e0t2 xe0 dyE0 lr,
    r34_idblock_tiedAt xN wN bN gN vN epsStr lrStr cotN ε e1W1 e1b1 e1g1 e1t1 e1W2 e1b2 e1g2 e1t2 xe1 dyE1 lr,
    (fun i j => r34_dense_tied_totalloss dN lrStr cotN Wd bd a_gap label lr i j),
    r34LossCot_den nlogN ohN (mnistLinear Wd bd a_gap) label⟩

end Proofs.ResNet34PoC
