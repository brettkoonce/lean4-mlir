import LeanMlir.Proofs.Architectures.ConvNeXt
import LeanMlir.Proofs.Architectures.ConvNeXtFullT
import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Architectures.EfficientNetBackB0
import LeanMlir.Proofs.Architectures.ConvNeXtBackCertifiedTie

/-! # ConvNeXt whole-block backward-graph faithfulness (per-example / batch-1)

The ConvNeXt analogue of `mbconvResidual_backGraph_faithful` (EfficientNet) and
`r34*BackBatchedGraph_faithful` (ResNet-34): a *backward* StableHLO graph that
denotes the proven whole-block VJP.

ConvNeXt's whole verified stack is **per-example / batch-1** — LayerNorm here is
the per-example separable `layerNormForward` (= `bnForward` on the feature axis),
so NONE of EfficientNet's `batchMap`/`bnBatchLA` batched machinery is needed
(`ConvNeXtChainClose.lean:8`). So this file targets the per-example VJP
`convNextBlock_has_vjp` directly, modeled on the per-example section of
`EfficientNetBackB0.lean` (`residualBackGraph`, `convBnSwishBackGraph`,
`mbconvResidual_backGraph_faithful`).

Two worlds live here, and the distinction is the whole point of the file:

  * **§1 — the ch9 representative (SCALAR LN).** `cnxBlockBodyBackGraph` /
    `cnxResidBlockBackGraph` denote `convNextBlockBody_has_vjp` / `convNextBlock_has_vjp`.
    The block is `residual (block body)` with an identity skip, so the brick is
    `residualBackGraph (bodyBack …) dy`, closed via `residualBackGraph_faithful`.

  * **§2 (§2o Part A, 2026-07-31) — the SHIPPED net (CHANNEL LN).** `chanLNBackGraph` and its
    faithfulness, then the block, residual-block and downsample capstones over it. This is what
    the §2n drop left uncovered.

**What §2o Part A fixed.** §2n's commit message said the channel-LN *downsample* had no graph-side
backward capstone. Measured afterwards it was wider than that: the two capstones that SURVIVED the
drop (§1 above) are over `convNextBlockBody` — the ch9 representative's SCALAR LN — so the shipped
channel-LN net had **no `den`-level backward capstone at any level**, only the five FORWARD graph
theorems from §2m. The drop did not cause that; it removed the scalar capstone that was making the
column look populated. §2 closes it: `chanLNBackGraph_faithful` is the backward peer of §2m's
`chanLNGraph_faithful`, and `chanLNBackGraph_eq_vjp` chains it through §B
(`ConvNeXtBackCertifiedTie.chanLNTensor3Back_eq_chanLN_vjp`) so every capstone in §2 lands on the
CERTIFIED VJP rather than on a hand-composed reverse chain.

The block body is `layerScale ∘ project(1×1) ∘ gelu ∘ expand(1×1) ∘ LN ∘
depthwise(7×7)`; everything is smooth (GELU is smooth, conv/layerScale linear, LN
smooth given `ε>0`), so the body VJP is the unconditional `vjp_comp` chain
`convNextBlockBody_has_vjp` and the only side condition is the LayerNorm positivity
`0 < εn`. The LN backward is the one non-`rfl` op: `layerNorm_has_vjp` is
*definitionally* `bn_has_vjp`, so `bnBack_faithful_fn` (the `∑ pdiv` bridge) closes it.

Same scalar-LN representation caveat as `convNextBlockBody`'s doc-comment.
-/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § ConvNeXt block body backward graph (per-example)
-- ════════════════════════════════════════════════════════════════

/-- The ConvNeXt block-body backward graph `D⁻¹ ∘ LN⁻¹ ∘ EX⁻¹ ∘ GE⁻¹ ∘ PR⁻¹ ∘ LS⁻¹`,
    each op's backward applied at its forward input activation (outermost backward
    token = earliest forward op `depthwise`; innermost child applied to the cotangent
    subgraph `e`). The reverse-order chain of `convNextBlockBody`'s VJP:

      `depthwiseBack ∘ bnBack(LN) ∘ convBack(expand) ∘ geluBack ∘ convBack(project)
        ∘ layerScaleF`

    `layerScaleF` is the (input-independent diagonal) layer-scale backward; the two
    `convBack`s are the 1×1 expand/project; `bnBack` is the LayerNorm backward
    (LN = BN on the feature axis). -/
noncomputable def cnxBlockBodyBackGraph {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w))
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  let d  := depthwiseFlat (h := h) (w := w) Wdw bdw x              -- LN's input
  let nl := layerNormForward (c * h * w) εn γn βn d                -- EX's input
  let ex := flatConv (h := h) (w := w) Wex bex nl                  -- GE's input
  let ge := gelu (cExp * h * w) ex                                 -- PR's input
  .depthwiseBack "%cnxWdw" Wdw bdw x
    (.bnBack "%cnxGn" "%cnxXn" "cnxE" εn γn d
      (.convBack "%cnxWex" Wex bex nl
        (.geluBack "%cnxGe" ex
          (.convBack "%cnxWpr" Wpr bpr ge
            (.layerScaleF "%cnxGls" γls e)))))

/-- **ConvNeXt block-body backward-graph faithfulness.** The reverse-order graph
    denotes the proven `convNextBlockBody_has_vjp` backward, under `0 < εn`. The LN
    backward is the one non-`rfl` op (`bnBack_faithful_fn`); the rest
    (`depthwiseBack`/`convBack`/`geluBack`/`layerScaleF`) are `rfl`-faithful per-op
    tokens, and `layerScaleF` denotes the input-independent diagonal layer-scale
    backward. -/
theorem cnxBlockBodyBackGraph_faithful {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w))
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) :
    den (cnxBlockBodyBackGraph Wdw bdw εn γn βn Wex bex Wpr bpr γls x e)
      = (convNextBlockBody_has_vjp Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls).backward
          x (den e) := by
  simp only [cnxBlockBodyBackGraph, convNextBlockBody_has_vjp, vjp_comp,
    depthwiseBack_faithful, bnBack_faithful_fn (β := βn) (hε := hεn),
    convBack_faithful, geluBack_faithful, layerScaleF_faithful, Function.comp_apply]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Identity/residual block capstone (per-example)
-- ════════════════════════════════════════════════════════════════

/-- The whole ConvNeXt residual block backward graph (block body + identity skip):
    `residualBackGraph (bodyBack … (%dy)) dy`. The identity skip contributes the
    cotangent verbatim; `addV` sums the two paths. -/
noncomputable def cnxResidBlockBackGraph {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w))
    (x : Vec (c * h * w)) (ecot : SHlo (c * h * w)) : SHlo (c * h * w) :=
  residualBackGraph
    (cnxBlockBodyBackGraph Wdw bdw εn γn βn Wex bex Wpr bpr γls x ecot) ecot

/-- **The whole per-example ConvNeXt residual block: backward graph ↔ proven VJP**
    (identity/residual capstone), no hypotheses beyond `0 < εn`. Assembles the body
    backward graph (`cnxBlockBodyBackGraph`) + the identity skip into the proven
    `convNextBlock_has_vjp` backward via `residualBackGraph_faithful`. -/
theorem cnxResidBlockBackGraph_faithful {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w))
    (x : Vec (c * h * w)) (ecot : SHlo (c * h * w)) :
    den (cnxResidBlockBackGraph Wdw bdw εn γn βn Wex bex Wpr bpr γls x ecot)
      = (convNextBlock_has_vjp Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls).backward x (den ecot) :=
  residualBackGraph_faithful
    (convNextBlockBody Wdw bdw εn γn βn Wex bex Wpr bpr γls)
    (convNextBlockBody_differentiable Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls)
    (convNextBlockBody_has_vjp Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls)
    x ecot
    (cnxBlockBodyBackGraph Wdw bdw εn γn βn Wex bex Wpr bpr γls x ecot)
    (cnxBlockBodyBackGraph_faithful Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls x
      ecot)

end Proofs.StableHLO

namespace Proofs

/-- **The backward peer of `rowLN_affine_eq`** (`ConvNeXtChannelLN.lean`). Forward, the emitted
    subtree normalises at the scalar identities `%one`/`%zero` and only then applies the real `[c]`
    affine, so three denotations collapse onto `rowLNVecFlat`. Backward it is the same fold one
    step earlier: the emitted `rowScaleF γ` applied to the COTANGENT is exactly the per-row
    `diagBack γ` that `rowLNVecFlatBack` folds in, and the LN input gradient then runs at `γ = 1`.

    `β` does not appear on either side — the translation's adjoint is the identity, which is the
    same β-freeness §B proved for the certified backward. -/
theorem rowLNBack_affine_eq (s c : Nat) (ε : ℝ) (γ : Vec c) (X dy : Vec (s * c)) :
    StableHLO.rowLNBackFlat s c ε 1 X (StableHLO.rowScaleFlat s c γ dy)
      = rowLNVecFlatBack s c ε γ X dy := by
  unfold StableHLO.rowLNBackFlat StableHLO.rowScaleFlat rowLNVecFlatBack perRowIdxFlat
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
    name, exactly as the scalar `bnBack` carries its own: `lnRowBack` recomputes x̂/istd from the
    input rather than saving them. The backward peer of `ConvNeXtFullT.chanLNGraph`. -/
noncomputable def chanLNBackGraph (gN xN epsStr : String) {c h w : Nat} (ε : ℝ) (γ : Vec c)
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  (Nat.mul_assoc c h w).symm ▸
    (.transposeF (m := h * w) (n := c)
      (.lnRowBack (m := h * w) (n := c) "%one" xN epsStr ε 1 (chanLNRows c h w x)
        (.rowScaleF (m := h * w) (n := c) gN γ
          (.transposeF (m := c) (n := h * w) ((Nat.mul_assoc c h w) ▸ e)))))

/-- **Channel-LN backward-graph faithfulness** — the `den`-level peer of `chanLNGraph_faithful`,
    and the keystone the §2n drop left uncovered. Same six-step shape as the forward: the two `▸`
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
    on `chanLNTensor3Back`, the hand-composed reverse chain; §B's
    `chanLNTensor3Back_eq_chanLN_vjp` carries it the last step onto
    `(chanLNTensor3_has_vjp …).backward`. This is the statement every capstone below is built on,
    and the reason landing §B first was worth doing — without it these would tie the graph to
    another hand-written chain rather than to the certified gradient. β-free on both sides. -/
theorem chanLNBackGraph_eq_vjp (gN xN epsStr : String) {c h w : Nat} (ε : ℝ) (hε : 0 < ε)
    (γ β : Vec c) (x : Vec (c * h * w)) (e : SHlo (c * h * w)) :
    den (chanLNBackGraph gN xN epsStr ε γ x e)
      = (chanLNTensor3_has_vjp c h w ε γ β hε).backward x (den e) := by
  rw [chanLNBackGraph_faithful, chanLNTensor3Back_eq_chanLN_vjp (β := β) ε hε γ x]

/-- The channel-LN block-body backward graph — `cnxBlockBodyBackGraph`'s shape with the scalar
    `bnBack` replaced by `chanLNBackGraph` and the LN affine widened from `ℝ` to `Vec c`, which is
    exactly what §2m's flip did to the forward. -/
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
    `cnxBodyWith_has_vjp`'s backward at the shipped LayerNorm, under `0 < εn`. Same proof as the
    scalar peer with `chanLNBackGraph_eq_vjp` where `bnBack_faithful_fn` was — the LN is still the
    one non-`rfl` op, it is just a whole subtree now instead of a token. -/
theorem cnxBlockBodyChBackGraph_faithful {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w))
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) :
    den (cnxBlockBodyChBackGraph Wdw bdw εn γn βn Wex bex Wpr bpr γls x e)
      = (cnxBodyWith_has_vjp (chanLNTensor3_diff c h w εn γn βn hεn)
          (chanLNTensor3_has_vjp c h w εn γn βn hεn)
          Wdw bdw Wex bex Wpr bpr γls).backward x (den e) := by
  unfold cnxBlockBodyChBackGraph
  rw [depthwiseBack_faithful, chanLNBackGraph_eq_vjp (β := βn) (hε := hεn)]
  simp only [cnxBodyWith_has_vjp, vjp_comp, convBack_faithful, geluBack_faithful,
    layerScaleF_faithful, Function.comp_apply]
  rfl

/-- The whole channel-LN residual block backward graph (block body + identity skip). -/
noncomputable def cnxResidBlockChBackGraph {c cExp h w kH kW : Nat}
    (p : CnxBlockParamsCh c cExp h w kH kW) (x : Vec (c * h * w)) (ecot : SHlo (c * h * w)) : SHlo (c * h * w) :=
  residualBackGraph
    (cnxBlockBodyChBackGraph p.Wdw p.bdw p.εn p.γn p.βn p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p) x
      ecot) ecot

/-- **The whole channel-LN ConvNeXt residual block: backward graph ↔ proven VJP** — the capstone
    the shipped net was missing, at the block the shipped stages are built from. Assembles the
    body backward graph + the identity skip into `cnxBlockChW_has_vjp`'s backward via
    `residualBackGraph_faithful`, no hypotheses beyond `0 < p.εn`. -/
theorem cnxResidBlockChBackGraph_faithful {c cExp h w kH kW : Nat}
    (p : CnxBlockParamsCh c cExp h w kH kW) (hε : 0 < p.εn) (x : Vec (c * h * w)) (ecot : SHlo (c * h * w)) :
    den (cnxResidBlockChBackGraph p x ecot) = (cnxBlockChW_has_vjp p hε).backward x (den ecot) :=
  residualBackGraph_faithful
    (cnxBodyWith (chanLNTensor3 c h w p.εn p.γn p.βn) p.Wdw p.bdw p.Wex p.bex p.Wpr p.bpr
      (cnxGlsCh p))
    (cnxBodyWith_diff (chanLNTensor3_diff c h w p.εn p.γn p.βn hε)
      p.Wdw p.bdw p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p))
    (cnxBodyWith_has_vjp (chanLNTensor3_diff c h w p.εn p.γn p.βn hε)
      (chanLNTensor3_has_vjp c h w p.εn p.γn p.βn hε)
      p.Wdw p.bdw p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p))
    x ecot
    (cnxBlockBodyChBackGraph p.Wdw p.bdw p.εn p.γn p.βn p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p) x
      ecot)
    (cnxBlockBodyChBackGraph_faithful p.Wdw p.bdw p.εn hε p.γn p.βn p.Wex p.bex p.Wpr p.bpr
      (cnxGlsCh p) x ecot)

/-- The channel-LN stage-boundary downsample backward graph — the port of the `cnxDownBackGraph`
    §2n dropped. Forward is `flatConvStride2(2×2) ∘ chanLNTensor3`, so the VJP in reverse order is
    `chanLNBackGraph ∘ convStridedBack`, each at its forward input: LN is the outer backward at
    `x`, and the strided conv's input is `chanLNTensor3 … x`. -/
noncomputable def cnxDownChBackGraph (h w : Nat) {cin cout : Nat}
    (p : CnxDownParamsCh cin cout) (x : Vec (cin * (2 * h) * (2 * w)))
    (e : SHlo (cout * h * w)) : SHlo (cin * (2 * h) * (2 * w)) :=
  chanLNBackGraph "%cnxdG" "%cnxdX" "cnxdE" p.ε p.γ x
    (.convStridedBack "%cnxdW" p.W p.b
      (chanLNTensor3 cin (2 * h) (2 * w) p.ε p.γ p.β x) e)

/-- **The channel-LN downsample: backward graph ↔ proven VJP**, under `0 < p.ε`. The theorem §2n's
    commit message named as the gap — restored at the LayerNorm the net actually uses.
    `convStridedBack` is `rfl`-faithful; the LN goes through `chanLNBackGraph_eq_vjp`. -/
theorem cnxDownChBackGraph_faithful (h w : Nat) {cin cout : Nat}
    (p : CnxDownParamsCh cin cout) (hε : 0 < p.ε)
    (x : Vec (cin * (2 * h) * (2 * w))) (e : SHlo (cout * h * w)) :
    den (cnxDownChBackGraph h w p x e) = (cnxDownChW_has_vjp h w p hε).backward x (den e) := by
  unfold cnxDownChBackGraph
  rw [chanLNBackGraph_eq_vjp (β := p.β) (hε := hε), convStridedBack_faithful]
  simp only [cnxDownChW_has_vjp, vjp_comp]
  rfl

end Proofs.StableHLO
