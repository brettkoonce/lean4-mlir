import LeanMlir.Proofs.Float.ConvNeXtBackFloatBridge
import LeanMlir.Proofs.Architectures.ConvNeXtFullT
import LeanMlir.Proofs.Architectures.DepthwiseBackCertifiedTie
import LeanMlir.Proofs.Foundation.Resnet34BackCertifiedTie

/-! # §B: the ConvNeXt block-body backward float bridge targets the CERTIFIED VJP

The A3 backward float bridge `cnxBlockBodyBack` (`ConvNeXtBackFloatBridge.lean`) proves
**deployed-float ≈ a hand-assembled reverse-mode transcription** of the ConvNeXt block body. This file
closes §B for that body: the transcription IS the certified input-gradient VJP
`convNextBlockBody_has_vjp` (`ConvNeXt.lean`), in the SAME non-batched vocabulary — so the float
bridge's closeness is now closeness to **the certified gradient**.

The ConvNeXt block body is `convNextBlockBody = layerScale ∘ project ∘ GELU ∘ expand ∘ LN ∘ depthwise`,
whose certified VJP backward applies the reverses in order
`LS.back → PR.back → GE.back → EX.back → LN.back → D.back`. The float `cnxBlockBodyBack` is the exact
peer chain `depthwiseFlatBack ∘ lnB ∘ convFlatBack Wex ∘ geluB ∘ convFlatBack Wpr ∘ lsB`. The tie pins
the three smooth/diagonal/norm backs (`lsB`/`geluB`/`lnB`) to the certified `layerScale`/`gelu`/`LN`
backwards at the exact saved activations, and ties the two 1×1 convs + the depthwise to their certified
input-VJPs via the leaf gates (`convFlatBack_eq_vjp_backward`, `depthwiseFlatBack_eq_vjp_backward`).
The conv/depthwise backwards ignore their primal (linear), the pinned backs match the certified saved
activations definitionally, so the whole tie closes by rewriting the three convolution leaves + `rfl`.

This is the convnext analogue of `r34IdBlockBack_eq_rblkPC_vjp`; b1-free (the per-example body is the
non-batched object the float bridge reverses, no `batchMap` reconciliation). The certified
`convNextBlockBody_has_vjp` already existed, so the work is the depthwise leaf gate (shared, in
`DepthwiseBackCertifiedTie`) + this per-block tie. 3-axiom-clean.
-/

namespace Proofs

open Classical

/-- **The §B ConvNeXt body tie: float-bridge backward = certified VJP.** `cnxBlockBodyBack`, with its
    abstract layer-scale / GELU / LayerNorm backs pinned to the certified `layerScale` / `gelu` /
    `layerNorm` backwards at the exact saved forward activations (`depthwiseFlat … v` for LN, the deeper
    forward partials for GELU/LS), equals `(convNextBlockBody_has_vjp_at …).backward`.

    Both sides apply the six op-reverses in the order `LS → PR → GE → EX → LN → D`. The two 1×1 convs
    tie via `convFlatBack_eq_vjp_backward` (1×1 is odd) and the depthwise via
    `depthwiseFlatBack_eq_vjp_backward`; the conv/depthwise backwards ignore their (linear) primal and
    the pinned smooth backs carry the certified saved activations, so after rewriting the three leaves
    everything matches definitionally. Closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem cnxBlockBodyBack_eq_convNextBlockBody_vjp {c cExp h w kHd kWd : Nat}
    (hkHd : 2 * ((kHd - 1) / 2) + 1 = kHd) (hkWd : 2 * ((kWd - 1) / 2) + 1 = kWd)
    (Wdw : DepthwiseKernel c kHd kWd) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) (v : Vec (c * h * w)) :
    cnxBlockBodyBack Wdw Wex Wpr
      ((layerNorm_has_vjp (c * h * w) εn γn βn hεn).backward (depthwiseFlat Wdw bdw v))
      ((layerScale_has_vjp γls).backward
        ((flatConv (h := h) (w := w) Wpr bpr ∘ gelu (cExp * h * w) ∘
          flatConv (h := h) (w := w) Wex bex ∘ layerNormForward (c * h * w) εn γn βn ∘
          depthwiseFlat (h := h) (w := w) Wdw bdw) v))
      ((gelu_has_vjp (cExp * h * w)).backward
        ((flatConv (h := h) (w := w) Wex bex ∘ layerNormForward (c * h * w) εn γn βn ∘
          depthwiseFlat (h := h) (w := w) Wdw bdw) v))
      = (convNextBlockBody_has_vjp_at Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls v).backward := by
  funext dy
  unfold cnxBlockBodyBack
  rw [convFlatBack_eq_vjp_backward (W := Wex) (b := bex)
        (x := (layerNormForward (c * h * w) εn γn βn ∘ depthwiseFlat (h := h) (w := w) Wdw bdw) v)
        (by decide) (by decide),
      convFlatBack_eq_vjp_backward (W := Wpr) (b := bpr)
        (x := (gelu (cExp * h * w) ∘ flatConv (h := h) (w := w) Wex bex ∘
          layerNormForward (c * h * w) εn γn βn ∘ depthwiseFlat (h := h) (w := w) Wdw bdw) v)
        (by decide) (by decide),
      depthwiseFlatBack_eq_vjp_backward hkHd hkWd Wdw bdw v]
  rfl

/-- **The §B ConvNeXt block tie (residual-wrapped).** The full block is `residual (body)`, so the
    float block backward `residual (cnxBlockBodyBack …)` (the `dy ↦ bodyBack(dy) + dy` additive skip,
    as `floatBridges_cnxBlockBack` wraps it) equals `(convNextBlock_has_vjp_at …).backward`. Immediate
    from the body tie + the residual fan-in (`residual_has_vjp = biPath_has_vjp body id`, the skip's
    backward is `dy`): rewrite the body tie, then `rfl`. 3-axiom-clean. -/
theorem cnxBlockBack_eq_convNextBlock_vjp {c cExp h w kHd kWd : Nat}
    (hkHd : 2 * ((kHd - 1) / 2) + 1 = kHd) (hkWd : 2 * ((kWd - 1) / 2) + 1 = kWd)
    (Wdw : DepthwiseKernel c kHd kWd) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) (v : Vec (c * h * w)) :
    Proofs.residual (cnxBlockBodyBack Wdw Wex Wpr
      ((layerNorm_has_vjp (c * h * w) εn γn βn hεn).backward (depthwiseFlat Wdw bdw v))
      ((layerScale_has_vjp γls).backward
        ((flatConv (h := h) (w := w) Wpr bpr ∘ gelu (cExp * h * w) ∘
          flatConv (h := h) (w := w) Wex bex ∘ layerNormForward (c * h * w) εn γn βn ∘
          depthwiseFlat (h := h) (w := w) Wdw bdw) v))
      ((gelu_has_vjp (cExp * h * w)).backward
        ((flatConv (h := h) (w := w) Wex bex ∘ layerNormForward (c * h * w) εn γn βn ∘
          depthwiseFlat (h := h) (w := w) Wdw bdw) v)))
      = (convNextBlock_has_vjp_at Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls v).backward := by
  rw [cnxBlockBodyBack_eq_convNextBlockBody_vjp hkHd hkWd Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls v]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § §B at ConvNeXt's REAL channel LayerNorm (§2n) — the LN op itself
-- ════════════════════════════════════════════════════════════════

/-! The two ties above pin the block body's ABSTRACT `lnB` slot to a certified backward, which is
why they never had to look inside a LayerNorm. `chanLNTensor3Back` (`ChannelLNFloatBridge.lean`)
is not abstract — it is a concrete five-factor chain, written so `floatBridges_chanLNTensor3Back`
can run `floatClose_bnBack` in its middle. So it owes the tie the block ties did not: that the
chain IS `chanLNTensor3_has_vjp`'s backward. That is what this section proves.

The proof is piecewise, and every piece is already in the repo:

* the two re-associations collapse by `reassoc{Fwd,Back}_has_vjp_backward_eq` (a permutation's
  scatter has exactly one surviving delta);
* the transpose collapses by `rfl` — `transpose_has_vjp`'s backward is `fun i j => dY j i`, which
  through `hasVJPMat_to_hasVJP` is the flat transpose back;
* the row map is ViT's vector-LN, whose VJP is `(+β)` (identity backward) after `layerScale γ`
  (`diagBack γ`) after `LN(1,0)` — and the LN backward meets the concrete three-term
  `bn_grad_input` through the canonical `∑ pdiv` form, NOT by `rfl` (the `bn_has_vjp` witness is
  built through a `rw [bnForward_eq_compose]` cast — the trap `bnBack_faithful_fn` documents).

Two things worth reading off the statement. **The tie is β-free**: the certified backward does not
depend on the LN bias, and neither does the float chain — the `+β` translation's VJP is the
identity, which is why `chanLNTensor3Back` never took a `β` in the first place. And the transfer
to the committed witness goes through `HasVJP.backward_unique`, so it does not matter that
`chanLNTensor3_has_vjp` is tactic-built: any two witnesses for one map have one backward. -/

/-- **Any two VJP witnesses for the same map have the same backward.** Both `.correct` to the same
    `∑ pdiv f x i j * dy j`, so the backward is a property of `f`, not of how the witness was
    assembled. Lets a hand-written chain be tied to a tactic-built witness without unfolding it. -/
theorem HasVJP.backward_unique {m n : Nat} {f : Vec m → Vec n} (h₁ h₂ : HasVJP f)
    (x : Vec m) (dy : Vec n) : h₁.backward x dy = h₂.backward x dy := by
  funext i; rw [h₁.correct, h₂.correct]

/-- **The concrete three-term BN/LN input gradient IS the certified VJP backward.** `bn_grad_input`
    is not `rfl`-equal to `(bn_has_vjp …).backward` — the witness is built through a
    `rw [bnForward_eq_compose]` cast — but both reduce to the canonical `∑ pdiv` form
    (`bn_input_grad_correct` and `.correct`). The function-level peer of `bnBack_faithful_fn`. -/
theorem bn_grad_input_eq_vjp_backward {n : Nat} (ε γ β : ℝ) (hε : 0 < ε) (x dy : Vec n) :
    bn_grad_input n ε γ x dy = (bn_has_vjp n ε γ β hε).backward x dy := by
  funext i
  rw [bn_input_grad_correct n ε γ β hε x dy i]
  exact ((bn_has_vjp n ε γ β hε).correct x dy i).symm

/-- **The flat transpose's VJP backward is the flat transpose back** — `transpose_has_vjp`'s
    backward is `fun i j => dY j i`, and `hasVJPMat_to_hasVJP` reads it at the row-major split, so
    this is definitional. The permutation adjoint the channel-LN conjugation needs, alongside
    `reassoc{Fwd,Back}_has_vjp_backward_eq`. -/
theorem transposeFlat_has_vjp_backward_eq (m n : Nat) (v : Vec (m * n)) (dy : Vec (n * m)) :
    (transposeFlat_has_vjp m n).backward v dy = StableHLO.transposeFlat n m dy := rfl

/-- **The vector-LN row backward is `bn_grad_input` after the `γ` scale.** `layerNormVec` is
    `(+β) ∘ layerScale γ ∘ LN(1,0)`, so its VJP applies: the bias translation's identity backward,
    then `diagBack γ`, then the LN input gradient at `γ = 1`. The `+β` drops out — this is where
    the whole channel-LN backward story becomes β-free. -/
theorem layerNormVec_has_vjp_backward_eq {D : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec D)
    (x dy : Vec D) :
    (layerNormVec_has_vjp D ε γ β hε).backward x dy
      = bn_grad_input D ε 1 x (diagBack γ dy) := by
  rw [bn_grad_input_eq_vjp_backward ε 1 0 hε x (diagBack γ dy)]
  rfl

/-- **The rowwise vector-LN backward is `rowLNVecFlatBack`.** `rowLNVecFlat_has_vjp` is the
    `rowwise_has_vjp_mat` lift of the row VJP through `hasVJPMat_to_hasVJP`, and
    `rowLNVecFlatBack` is `perRowIdxFlat` of the row's closed form — the same per-row map at the
    same row of the saved input, so this is the row lemma read at each `(row, col)`. -/
theorem rowLNVecFlat_has_vjp_backward_eq {s c : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (X dy : Vec (s * c)) :
    (rowLNVecFlat_has_vjp s c ε γ β hε).backward X dy = rowLNVecFlatBack s c ε γ X dy := by
  funext idx
  show (layerNormVec_has_vjp c ε γ β hε).backward
      (Mat.unflatten X (finProdFinEquiv.symm idx).1)
      (Mat.unflatten dy (finProdFinEquiv.symm idx).1) (finProdFinEquiv.symm idx).2 = _
  rw [layerNormVec_has_vjp_backward_eq ε hε γ β]
  rfl

/-- The channel-LN VJP as a TERM-mode `vjp_comp` chain — the same five factors
    `chanLNTensor3_has_vjp` composes, but assembled without the leading `unfold`, so its backward
    reduces definitionally to the nested chain. Only a stepping stone: `HasVJP.backward_unique`
    transfers the result to the committed witness. -/
noncomputable def chanLNTensor3_vjp_chain (c h w : Nat) (ε : ℝ) (γ β : Vec c) (hε : 0 < ε) :
    HasVJP (chanLNTensor3 c h w ε γ β) :=
  let d0 := reassocFwd_differentiable c h w
  let d1 := transposeFlat_diff c (h * w)
  let d2 := rowLNVecFlat_diff (h * w) c ε γ β hε
  let d3 := transposeFlat_diff (h * w) c
  let d4 := reassocBack_differentiable c h w
  vjp_comp _ _ (d3.comp (d2.comp (d1.comp d0))) d4
    (vjp_comp _ _ (d2.comp (d1.comp d0)) d3
      (vjp_comp _ _ (d1.comp d0) d2
        (vjp_comp _ _ d0 d1 (reassocFwd_has_vjp c h w) (transposeFlat_has_vjp c (h * w)))
        (rowLNVecFlat_has_vjp (h * w) c ε γ β hε))
      (transposeFlat_has_vjp (h * w) c))
    (reassocBack_has_vjp c h w)

/-- **THE §2n §B TIE: the channel-LN float backward IS the certified VJP.** `chanLNTensor3Back` —
    the hand-composed reverse chain `floatBridges_chanLNTensor3Back` bridges — equals
    `(chanLNTensor3_has_vjp …).backward` at every saved input and cotangent. So the float bridge's
    closeness is closeness to **the certified gradient**, and the ⚠ in
    `ChannelLNFloatBridge.lean`'s docstring is discharged.

    Proof: compute the term-mode chain's backward by rewriting its five factors (two reassoc
    collapses, two transposes by `rfl`, the row map through `bn_grad_input`), then transfer to the
    committed tactic-built witness by `HasVJP.backward_unique`. The channel-LN peer of
    `cnxBlockBodyBack_eq_convNextBlockBody_vjp`; 3-axiom-clean. -/
theorem chanLNTensor3Back_eq_chanLN_vjp {c h w : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (c * h * w)) :
    chanLNTensor3Back c h w ε γ x = (chanLNTensor3_has_vjp c h w ε γ β hε).backward x := by
  funext dy
  rw [HasVJP.backward_unique (chanLNTensor3_has_vjp c h w ε γ β hε)
        (chanLNTensor3_vjp_chain c h w ε γ β hε) x dy]
  simp only [chanLNTensor3_vjp_chain, vjp_comp]
  rw [reassocBack_has_vjp_backward_eq, transposeFlat_has_vjp_backward_eq,
      rowLNVecFlat_has_vjp_backward_eq (β := β) ε hε,
      transposeFlat_has_vjp_backward_eq, reassocFwd_has_vjp_backward_eq]
  rfl

/-- **The §B channel-LN BODY tie: the float block-body backward = the certified VJP.** The peer of
    `cnxBlockBodyBack_eq_convNextBlockBody_vjp` for the net the repo ships. `cnxBlockBodyBack` with
    its LayerNorm slot filled by the CONCRETE `chanLNTensor3Back` (the chain
    `floatBridges_chanLNTensor3Back` bridges, at the saved post-depthwise activation) and its
    layer-scale / GELU slots pinned to the certified backwards equals
    `(cnxBodyWith_has_vjp (chanLNTensor3 …) …).backward`.

    Note what fills the LN slot here: not a certified object but the float bridge's own five-factor
    chain. That is the point — the scalar tie could pin an abstract `lnB` to whatever it liked,
    while this one has to go through `chanLNTensor3Back_eq_chanLN_vjp` to earn it. Otherwise the
    proof is the scalar one: rewrite the two 1×1 conv leaves and the depthwise leaf through their
    gates, rewrite the LN chain through its tie, and the rest matches definitionally.
    3-axiom-clean. -/
theorem cnxBodyWithChanLNBack_eq_vjp {c cExp h w kHd kWd : Nat}
    (hkHd : 2 * ((kHd - 1) / 2) + 1 = kHd) (hkWd : 2 * ((kWd - 1) / 2) + 1 = kWd)
    (Wdw : DepthwiseKernel c kHd kWd) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) (v : Vec (c * h * w)) :
    cnxBlockBodyBack Wdw Wex Wpr
      (chanLNTensor3Back c h w εn γn (depthwiseFlat (h := h) (w := w) Wdw bdw v))
      ((layerScale_has_vjp γls).backward
        ((flatConv (h := h) (w := w) Wpr bpr ∘ gelu (cExp * h * w) ∘
          flatConv (h := h) (w := w) Wex bex ∘ chanLNTensor3 c h w εn γn βn ∘
          depthwiseFlat (h := h) (w := w) Wdw bdw) v))
      ((gelu_has_vjp (cExp * h * w)).backward
        ((flatConv (h := h) (w := w) Wex bex ∘ chanLNTensor3 c h w εn γn βn ∘
          depthwiseFlat (h := h) (w := w) Wdw bdw) v))
      = (cnxBodyWith_has_vjp (chanLNTensor3_diff c h w εn γn βn hεn)
          (chanLNTensor3_has_vjp c h w εn γn βn hεn)
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

/-- **The §B channel-LN BLOCK tie (residual-wrapped).** `cnxBlockChW` is `residual` of the body, so
    the float block backward is `residual (cnxBlockBodyBack …)` and equals
    `(cnxBlockChW_has_vjp …).backward` — the additive skip's backward being `dy`. Immediate from the
    body tie. With this, the channel-LN net's §B coverage matches the scalar net's: body, block, and
    (new, because the LN slot is no longer abstract) the LayerNorm op itself. -/
theorem cnxBlockChBack_eq_vjp {c cExp h w kHd kWd : Nat}
    (hkHd : 2 * ((kHd - 1) / 2) + 1 = kHd) (hkWd : 2 * ((kWd - 1) / 2) + 1 = kWd)
    (p : CnxBlockParamsCh c cExp h w kHd kWd) (hε : 0 < p.εn) (v : Vec (c * h * w)) :
    Proofs.residual (cnxBlockBodyBack p.Wdw p.Wex p.Wpr
      (chanLNTensor3Back c h w p.εn p.γn (depthwiseFlat (h := h) (w := w) p.Wdw p.bdw v))
      ((layerScale_has_vjp (cnxGlsCh p)).backward
        ((flatConv (h := h) (w := w) p.Wpr p.bpr ∘ gelu (cExp * h * w) ∘
          flatConv (h := h) (w := w) p.Wex p.bex ∘ chanLNTensor3 c h w p.εn p.γn p.βn ∘
          depthwiseFlat (h := h) (w := w) p.Wdw p.bdw) v))
      ((gelu_has_vjp (cExp * h * w)).backward
        ((flatConv (h := h) (w := w) p.Wex p.bex ∘ chanLNTensor3 c h w p.εn p.γn p.βn ∘
          depthwiseFlat (h := h) (w := w) p.Wdw p.bdw) v)))
      = (cnxBlockChW_has_vjp p hε).backward v := by
  rw [cnxBodyWithChanLNBack_eq_vjp hkHd hkWd p.Wdw p.bdw p.εn hε p.γn p.βn p.Wex p.bex
        p.Wpr p.bpr (cnxGlsCh p) v]
  rfl

end Proofs
