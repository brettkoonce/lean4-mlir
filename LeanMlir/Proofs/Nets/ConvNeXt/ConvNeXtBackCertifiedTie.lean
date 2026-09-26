import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtBackChains
import LeanMlir.Proofs.Architectures.ChannelLNBack
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFullT
import LeanMlir.Proofs.Architectures.DepthwiseBackCertifiedTie
import LeanMlir.Proofs.Architectures.ConvBackCertifiedTie

/-! # The ConvNeXt block-body backward chain IS the certified VJP

`cnxBlockBodyBack` (`ConvNeXtBackChains.lean`) is the hand-composed reverse of the ConvNeXt block
body, written in the per-op backward maps of `BackwardMaps.lean`. This file ties it for the net
the repo ships, whose LayerNorm is the CHANNEL LN `chanLNTensor3`: the chain IS the certified
input-gradient VJP, in the SAME non-batched vocabulary.

The block body is `layerScale ∘ project ∘ GELU ∘ expand ∘ LN ∘ depthwise`, whose certified VJP
backward applies the reverses in order `LS.back → PR.back → GE.back → EX.back → LN.back → D.back`.
`cnxBlockBodyBack` is the exact peer chain
`depthwiseFlatBack ∘ lnB ∘ convFlatBack Wex ∘ geluB ∘ convFlatBack Wpr ∘ lsB`. The tie pins the
layer-scale and GELU backs (`lsB`/`geluB`) to the certified backwards at the exact saved
activations, fills the LN slot with the concrete `chanLNTensor3Back` (tied by
`chanLNTensor3Back_eq_chanLN_vjp`), and ties the two 1×1 convs + the depthwise to their certified
input-VJPs via the leaf gates (`convFlatBack_eq_vjp_backward`, `depthwiseFlatBack_eq_vjp_backward`).
b1-free: the per-example body is the non-batched object the chain reverses, so there is no
`batchMap` reconciliation.
-/

namespace Proofs


-- ════════════════════════════════════════════════════════════════
-- § §B at ConvNeXt's REAL channel LayerNorm (§2n) — the LN op itself
-- ════════════════════════════════════════════════════════════════

/-! `chanLNTensor3Back` (`ChannelLNBack.lean`) is not an abstract slot — it is a concrete
five-factor chain, the row map conjugated by the forward's four layout permutations. So it owes a
tie of its own: that the chain IS `chanLNTensor3HasVJP`'s backward. That is what this section
proves.

The proof is piecewise, and every piece is already in the repo:

* the two re-associations collapse by `reassoc{Fwd,Back}HasVJP_backward_eq` (a permutation's
  scatter has exactly one surviving delta);
* the transpose collapses by `rfl` — `transposeHasVJP`'s backward is `fun i j => dY j i`, which
  through `HasVJPMat.toHasVJP` is the flat transpose back;
* the row map is ViT's vector-LN, whose VJP is `(+β)` (identity backward) after `layerScale γ`
  (`diagBack γ`) after `LN(1,0)` — and the LN backward meets the concrete three-term
  `bnGradInput` through the canonical `∑ pdiv` form, NOT by `rfl` (the `bnHasVJP` witness is
  built through a `rw [bnForward_eq_compose]` cast — the trap `bnBack_faithful_fn` documents).

**The tie is β-free**: the certified backward does not depend on the LN bias, and neither does
the chain — the `+β` translation's VJP is the identity, which is why `chanLNTensor3Back` never
took a `β` in the first place. -/

/-- **The flat transpose's VJP backward is the flat transpose back** — `transposeHasVJP`'s
    backward is `fun i j => dY j i`, and `HasVJPMat.toHasVJP` reads it at the row-major split, so
    this is definitional. The permutation adjoint the channel-LN conjugation needs, alongside
    `reassoc{Fwd,Back}HasVJP_backward_eq`. -/
theorem transposeFlatHasVJP_backward_eq (m n : Nat) (v : Vec (m * n)) (dy : Vec (n * m)) :
    (transposeFlatHasVJP m n).backward v dy = StableHLO.transposeFlat n m dy := rfl

/-- **The channel-LN backward tie: the chain is the certified VJP.** `chanLNTensor3Back` —
    the hand-composed reverse of `chanLNTensor3` — equals `(chanLNTensor3HasVJP …).backward` at
    every saved input and cotangent, so the chain is **the certified gradient**.

    Proof: the witness is a term-mode `vjpComp` chain, so its backward unfolds to the nested
    chain; rewrite its five factors (two reassoc collapses, two transposes by `rfl`, the row map
    through `bnGradInput`). -/
theorem chanLNTensor3Back_eq_chanLN_vjp {c h w : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (c * h * w)) :
    chanLNTensor3Back c h w ε γ x = (chanLNTensor3HasVJP c h w ε γ β hε).backward x := by
  funext dy
  simp only [chanLNTensor3HasVJP, vjpComp_backward]
  rw [reassocBackHasVJP_backward_eq, transposeFlatHasVJP_backward_eq,
      rowLNVecFlatHasVJP_backward_eq (β := β) ε hε,
      transposeFlatHasVJP_backward_eq, reassocFwdHasVJP_backward_eq]
  rfl

/-- **The channel-LN body tie: the block-body backward chain = the certified VJP**, for the net
    the repo ships. `cnxBlockBodyBack` with
    its LayerNorm slot filled by the CONCRETE `chanLNTensor3Back` (at the saved post-depthwise
    activation) and its
    layer-scale / GELU slots pinned to the certified backwards equals
    `(cnxBodyWithHasVJP (chanLNTensor3 …) …).backward`.

    Note what fills the LN slot: not a certified object but the concrete five-factor chain, which
    has to go through `chanLNTensor3Back_eq_chanLN_vjp` to earn its place. The proof rewrites the
    two 1×1 conv leaves and the depthwise leaf through their gates, rewrites the LN chain through
    its tie, and the rest matches definitionally. -/
theorem cnxBodyWithChanLNBack_eq_vjp {c cExp h w kHd kWd : Nat}
    (hkHd : 2 * ((kHd - 1) / 2) + 1 = kHd) (hkWd : 2 * ((kWd - 1) / 2) + 1 = kWd)
    (Wdw : DepthwiseKernel c kHd kWd) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) (v : Vec (c * h * w)) :
    cnxBlockBodyBack Wdw Wex Wpr
      (chanLNTensor3Back c h w εn γn (depthwiseFlat (h := h) (w := w) Wdw bdw v))
      ((layerScaleHasVJP γls).backward
        ((flatConv (h := h) (w := w) Wpr bpr ∘ gelu (cExp * h * w) ∘
          flatConv (h := h) (w := w) Wex bex ∘ chanLNTensor3 c h w εn γn βn ∘
          depthwiseFlat (h := h) (w := w) Wdw bdw) v))
      ((geluHasVJP (cExp * h * w)).backward
        ((flatConv (h := h) (w := w) Wex bex ∘ chanLNTensor3 c h w εn γn βn ∘
          depthwiseFlat (h := h) (w := w) Wdw bdw) v))
      = (cnxBodyWithHasVJP (chanLNTensor3_differentiable c h w εn γn βn hεn)
          (chanLNTensor3HasVJP c h w εn γn βn hεn)
          Wdw bdw Wex bex Wpr bpr γls).backward v := by
  funext dy
  unfold cnxBlockBodyBack
  rw [chanLNTensor3Back_eq_chanLN_vjp (β := βn) εn hεn γn
        (depthwiseFlat (h := h) (w := w) Wdw bdw v),
      convFlatBack_eq_vjp_backward (W := Wex) (b := bex)
        (x := (chanLNTensor3 c h w εn γn βn ∘ depthwiseFlat (h := h) (w := w) Wdw bdw) v)
        (by decide) (by decide),
      convFlatBack_eq_vjp_backward (W := Wpr) (b := bpr)
        (x := (gelu (cExp * h * w) ∘ flatConv (h := h) (w := w) Wex bex ∘
          chanLNTensor3 c h w εn γn βn ∘ depthwiseFlat (h := h) (w := w) Wdw bdw) v)
        (by decide) (by decide),
      depthwiseFlatBack_eq_vjp_backward hkHd hkWd Wdw bdw v]
  rfl

/-- **The channel-LN block tie (residual-wrapped).** `cnxBlockChW` is `residual` of the body, so
    the block backward chain is `residual (cnxBlockBodyBack …)` and equals
    `(cnxBlockChWHasVJP …).backward` — the additive skip's backward being `dy`. Immediate from the
    body tie. With `chanLNTensor3Back_eq_chanLN_vjp` and `cnxBodyWithChanLNBack_eq_vjp`, the
    channel-LN net's body, block and LayerNorm backward are each tied. -/
theorem cnxBlockChBack_eq_vjp {c cExp h w kHd kWd : Nat}
    (hkHd : 2 * ((kHd - 1) / 2) + 1 = kHd) (hkWd : 2 * ((kWd - 1) / 2) + 1 = kWd)
    (p : CnxBlockParamsCh c cExp h w kHd kWd) (hε : 0 < p.εn) (v : Vec (c * h * w)) :
    Proofs.residual (cnxBlockBodyBack p.Wdw p.Wex p.Wpr
      (chanLNTensor3Back c h w p.εn p.γn (depthwiseFlat (h := h) (w := w) p.Wdw p.bdw v))
      ((layerScaleHasVJP (cnxGlsCh p)).backward
        ((flatConv (h := h) (w := w) p.Wpr p.bpr ∘ gelu (cExp * h * w) ∘
          flatConv (h := h) (w := w) p.Wex p.bex ∘ chanLNTensor3 c h w p.εn p.γn p.βn ∘
          depthwiseFlat (h := h) (w := w) p.Wdw p.bdw) v))
      ((geluHasVJP (cExp * h * w)).backward
        ((flatConv (h := h) (w := w) p.Wex p.bex ∘ chanLNTensor3 c h w p.εn p.γn p.βn ∘
          depthwiseFlat (h := h) (w := w) p.Wdw p.bdw) v)))
      = (cnxBlockChWHasVJP p hε).backward v := by
  rw [cnxBodyWithChanLNBack_eq_vjp hkHd hkWd p.Wdw p.bdw p.εn hε p.γn p.βn p.Wex p.bex
        p.Wpr p.bpr (cnxGlsCh p) v]
  rfl

end Proofs
