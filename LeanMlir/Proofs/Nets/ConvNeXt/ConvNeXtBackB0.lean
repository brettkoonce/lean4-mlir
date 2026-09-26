import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtBackCertifiedTie

/-! # ConvNeXt whole-block backward-graph faithfulness (per-example / batch-1)

The ConvNeXt analogue of `mbResidBlockBackBatchedGraph_faithful` (EfficientNet) and
`r34BasicBlockBackBatchedGraph_faithful` / `r34DownBlockBackBatchedGraph_faithful` (ResNet-34):
a *backward* StableHLO graph that denotes the proven whole-block VJP.

This file's graphs are per-example: ConvNeXt's channel LayerNorm (`chanLNTensor3`) is
per-example separable, so no batch-coupled machinery (EfficientNet's `bnBatchLA`) is needed, and
the batched ties (`ConvNeXtStepTieGB.lean`, `ConvNeXtWholeBackCertifiedTieB.lean`) are plain
`batchMap`s of these per-example maps. The file is modelled on the per-example section of
`EfficientNetBackB0.lean` (`residualBackGraph`, `convBnSwishBackGraph`).

`chanLNBackGraph` and its faithfulness come first, then the block, residual-block and downsample
capstones over it. `chanLNBackGraph_faithful` is the backward peer of `chanLNGraph_faithful`, and
`chanLNBackGraph_eq_vjp` chains it through `chanLNTensor3Back_eq_chanLN_vjp`
(`ConvNeXtBackCertifiedTie.lean`), so every capstone lands on the certified VJP rather than on a
hand-composed reverse chain. The block is `residual (block body)`
with an identity skip, so the brick is `residualBackGraph (bodyBack …) dy`, closed via
`residualBackGraph_faithful`.

The block body is `layerScale ∘ project(1×1) ∘ gelu ∘ expand(1×1) ∘ LN ∘
depthwise(7×7)`; everything is smooth (GELU is smooth, conv/layerScale linear, LN
smooth given `ε>0`), so the body VJP is the unconditional `vjpComp` chain
`convNextBlockBodyHasVJP` and the only side condition is the LayerNorm positivity
`0 < εn`. The LN backward is the one non-`rfl` op, closed by `chanLNBackGraph_eq_vjp`.
-/

open Proofs Proofs.StableHLO

namespace Proofs

/-- **The backward peer of `rowLN_affine_eq`** (`ChannelLN.lean`). Forward, the emitted
    subtree normalises at the scalar identities `%one`/`%zero` and only then applies the real `[c]`
    affine, so three denotations collapse onto `rowLNVecFlat`. Backward it is the same fold one
    step earlier: the emitted `rowScaleF γ` applied to the COTANGENT is exactly the per-row
    `diagBack γ` that `rowLNVecFlatBack` folds in, and the LN input gradient then runs at `γ = 1`.

    `β` does not appear on either side — the translation's adjoint is the identity, the same
    β-freeness `chanLNTensor3Back_eq_chanLN_vjp` has for the certified backward. -/
theorem rowLNBack_affine_eq (s c : Nat) (ε : ℝ) (γ : Vec c) (X dy : Vec (s * c)) :
    StableHLO.rowLNBackFlat s c ε 1 X (StableHLO.rowScaleFlat s c γ dy)
      = rowLNVecFlatBack s c ε γ X dy := by
  unfold StableHLO.rowLNBackFlat StableHLO.rowScaleFlat rowLNVecFlatBack perRowFlatPR
         layerScale diagBack
  simp only [Mat.unflatten_flatten]
  rfl

end Proofs

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § §2o Part A — the channel-LN BACKWARD graph, and the capstones over it
-- ════════════════════════════════════════════════════════════════

/-- **One channel-LN backward site**, mirroring `ConvNeXtRender.lnBackSite` at `chLN := true`
    op-for-op: transpose the cotangent to `[h·w, c]`, scale it by the real `[c]` γ, run the rowwise
    LN input gradient at `γ = 1` against the saved input's transposed view, transpose back. The two
    `▸` transports are the same `Nat`-associativity casts `chanLNGraph` uses.

    The saved LN input enters as a VALUE (`chanLNRows` — its `[h·w, c]` view) alongside its SSA
    name, as `bnBack` carries its own: `lnRowBack` recomputes x̂/istd from the
    input rather than saving them. The backward peer of `ConvNeXtFullT.chanLNGraph`. -/
noncomputable def chanLNBackGraph (gN xN epsStr : String) {c h w : Nat} (ε : ℝ) (γ : Vec c)
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  (Nat.mul_assoc c h w).symm ▸
    (.transposeF (m := h * w) (n := c)
      (.lnRowBack (m := h * w) (n := c) "%one" xN epsStr ε 1 (chanLNRows c h w x)
        (.rowScaleF (m := h * w) (n := c) gN γ
          (.transposeF (m := c) (n := h * w) ((Nat.mul_assoc c h w) ▸ e)))))

/-- **Channel-LN backward-graph faithfulness** — the `den`-level peer of `chanLNGraph_faithful`.
    Same six-step shape as the forward: the two `▸`
    transports through `den_{un,re}assocS`, the three permutation/scale ops and the row backward
    through their `rfl` gates, and the graph's `rowScaleF`-then-`lnRowBack` pair collapsed onto
    `rowLNVecFlatBack` by `rowLNBack_affine_eq`. -/
theorem chanLNBackGraph_faithful (gN xN epsStr : String) {c h w : Nat} (ε : ℝ) (γ : Vec c)
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) :
    den (chanLNBackGraph gN xN epsStr ε γ x e) = chanLNTensor3Back c h w ε γ x (den e) := by
  unfold chanLNBackGraph chanLNTensor3Back
  rw [den_unassocS, transposeF_faithful, lnRowBack_faithful, rowScaleF_faithful,
      transposeF_faithful, den_reassocS, rowLNBack_affine_eq]
  rfl

/-- **The channel-LN backward graph denotes the CERTIFIED VJP.** `chanLNBackGraph_faithful` lands
    on `chanLNTensor3Back`, the hand-composed reverse chain;
    `chanLNTensor3Back_eq_chanLN_vjp` carries it the last step onto
    `(chanLNTensor3HasVJP …).backward`. Every capstone below is built on this statement, so each
    ties its graph to the certified gradient rather than to another hand-written chain. β-free on
    both sides. -/
theorem chanLNBackGraph_eq_vjp (gN xN epsStr : String) {c h w : Nat} (ε : ℝ) (hε : 0 < ε)
    (γ β : Vec c) (x : Vec (c * h * w)) (e : SHlo (c * h * w)) :
    den (chanLNBackGraph gN xN epsStr ε γ x e)
      = (chanLNTensor3HasVJP c h w ε γ β hε).backward x (den e) := by
  rw [chanLNBackGraph_faithful, chanLNTensor3Back_eq_chanLN_vjp (β := β) ε hε γ x]

/-- The channel-LN block-body backward graph — the block body's reverse chain with
    `chanLNBackGraph` for the LayerNorm and the LN affine at `Vec c`. -/
noncomputable def cnxBlockBodyChBackGraph {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (γn βn : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w))
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  let d  := depthwiseFlat (h := h) (w := w) Wdw bdw x              -- LN's input
  let nl := chanLNTensor3 c h w εn γn βn d                         -- EX's input
  let ex := flatConv (h := h) (w := w) Wex bex nl                  -- GE's input
  let ge := gelu (cExp * h * w) ex                                 -- PR's input
  .depthwiseBack "%cnxWdw" Wdw bdw x
    (chanLNBackGraph "%cnxGn" "%cnxXn" "cnxE" εn γn d
      (.convBack "%cnxWex" Wex bex nl
        (.geluBack "%cnxGe" ex
          (.convBack "%cnxWpr" Wpr bpr ge
            (.layerScaleF "%cnxGls" γls e)))))

/-- **Channel-LN block-body backward-graph faithfulness.** The reverse-order graph denotes
    `cnxBodyWithHasVJP`'s backward at the shipped LayerNorm, under `0 < εn`. The LN is the one
    non-`rfl` op (a whole subtree, closed by `chanLNBackGraph_eq_vjp`); the depthwise, 1×1 conv,
    GELU and layer-scale backs are their `*_faithful` lemmas. -/
theorem cnxBlockBodyChBackGraph_faithful {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w))
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) :
    den (cnxBlockBodyChBackGraph Wdw bdw εn γn βn Wex bex Wpr bpr γls x e)
      = (cnxBodyWithHasVJP (chanLNTensor3_differentiable c h w εn γn βn hεn)
          (chanLNTensor3HasVJP c h w εn γn βn hεn)
          Wdw bdw Wex bex Wpr bpr γls).backward x (den e) := by
  unfold cnxBlockBodyChBackGraph
  rw [depthwiseBack_faithful, chanLNBackGraph_eq_vjp (β := βn) (hε := hεn)]
  simp only [cnxBodyWithHasVJP, convBack_faithful, geluBack_faithful,
    layerScaleF_faithful]
  rfl

/-- The whole channel-LN residual block backward graph (block body + identity skip). -/
noncomputable def cnxResidBlockChBackGraph {c cExp h w kH kW : Nat}
    (p : CnxBlockParamsCh c cExp h w kH kW) (x : Vec (c * h * w)) (ecot : SHlo (c * h * w)) : SHlo (c * h * w) :=
  residualBackGraph
    (cnxBlockBodyChBackGraph p.Wdw p.bdw p.εn p.γn p.βn p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p) x
      ecot) ecot

/-- **The whole channel-LN ConvNeXt residual block: backward graph ↔ proven VJP**, at the block
    the shipped stages are built from. Assembles the
    body backward graph + the identity skip into `cnxBlockChWHasVJP`'s backward via
    `residualBackGraph_faithful`, no hypotheses beyond `0 < p.εn`. -/
theorem cnxResidBlockChBackGraph_faithful {c cExp h w kH kW : Nat}
    (p : CnxBlockParamsCh c cExp h w kH kW) (hε : 0 < p.εn) (x : Vec (c * h * w)) (ecot : SHlo (c * h * w)) :
    den (cnxResidBlockChBackGraph p x ecot) = (cnxBlockChWHasVJP p hε).backward x (den ecot) :=
  residualBackGraph_faithful
    (cnxBodyWith (chanLNTensor3 c h w p.εn p.γn p.βn) p.Wdw p.bdw p.Wex p.bex p.Wpr p.bpr
      (cnxGlsCh p))
    (cnxBodyWith_differentiable (chanLNTensor3_differentiable c h w p.εn p.γn p.βn hε)
      p.Wdw p.bdw p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p))
    (cnxBodyWithHasVJP (chanLNTensor3_differentiable c h w p.εn p.γn p.βn hε)
      (chanLNTensor3HasVJP c h w p.εn p.γn p.βn hε)
      p.Wdw p.bdw p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p))
    x ecot
    (cnxBlockBodyChBackGraph p.Wdw p.bdw p.εn p.γn p.βn p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p) x
      ecot)
    (cnxBlockBodyChBackGraph_faithful p.Wdw p.bdw p.εn hε p.γn p.βn p.Wex p.bex p.Wpr p.bpr
      (cnxGlsCh p) x ecot)

/-- The channel-LN stage-boundary downsample backward graph. Forward is `flatConvStride2(2×2) ∘ chanLNTensor3`, so the VJP in reverse order is
    `chanLNBackGraph ∘ convStridedBack`, each at its forward input: LN is the outer backward at
    `x`, and the strided conv's input is `chanLNTensor3 … x`. -/
noncomputable def cnxDownChBackGraph (h w : Nat) {cin cout : Nat}
    (p : CnxDownParamsCh cin cout) (x : Vec (cin * (2 * h) * (2 * w)))
    (e : SHlo (cout * h * w)) : SHlo (cin * (2 * h) * (2 * w)) :=
  chanLNBackGraph "%cnxdG" "%cnxdX" "cnxdE" p.ε p.γ x
    (.convStridedBack "%cnxdW" p.W p.b
      (chanLNTensor3 cin (2 * h) (2 * w) p.ε p.γ p.β x) e)

/-- **The channel-LN downsample: backward graph ↔ proven VJP**, under `0 < p.ε`, at the
    LayerNorm the net uses. `convStridedBack` is `rfl`-faithful; the LN goes through
    `chanLNBackGraph_eq_vjp`. -/
theorem cnxDownChBackGraph_faithful (h w : Nat) {cin cout : Nat}
    (p : CnxDownParamsCh cin cout) (hε : 0 < p.ε)
    (x : Vec (cin * (2 * h) * (2 * w))) (e : SHlo (cout * h * w)) :
    den (cnxDownChBackGraph h w p x e) = (cnxDownChWHasVJP h w p hε).backward x (den e) := by
  unfold cnxDownChBackGraph
  rw [chanLNBackGraph_eq_vjp (β := p.β) (hε := hε), convStridedBack_faithful]
  simp only [cnxDownChWHasVJP]
  rfl

/-- The **channel-LN** ConvNeXt block as a `CertLayer` — the form the shipped net's stages are
    built from, with `cnxResidBlockChBackGraph_faithful` as its `faithful` field. This is the one to
    chain for a real ConvNeXt stage. -/
noncomputable def cnxBlockChLayer {c cExp h w kH kW : Nat}
    (p : CnxBlockParamsCh c cExp h w kH kW) (hε : 0 < p.εn) :
    CertLayer (c * h * w) (c * h * w) where
  fwd := cnxBlockChW p
  ok := fun _ => True
  diff := fun x _ => (cnxBlockChW_differentiable p hε) x
  vjp := fun x _ => (cnxBlockChWHasVJP p hε).toHasVJPAt x
  graph := fun x e => cnxResidBlockChBackGraph p x e
  faithful := fun x _ e => cnxResidBlockChBackGraph_faithful p hε x e

end Proofs.StableHLO
