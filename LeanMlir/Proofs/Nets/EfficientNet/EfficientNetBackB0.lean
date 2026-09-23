import LeanMlir.Proofs.Foundation.BatchedBackLinks
import LeanMlir.Proofs.Architectures.Residual
import LeanMlir.Proofs.Architectures.SE
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNet
import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetChainClose

/-! # Spike: fan-in backward-graph faithfulness (EfficientNet-B0 assembly brick)

The first reusable brick toward an `efficientnet*_back_faithful` theorem:
a *backward* StableHLO graph that denotes the proven whole-net VJP, the way
`mlpVerified_back_faithful` does for the MLP.

EfficientNet-B0 has two *branching* ops the MLP/dense-chain nets don't: the
MBConv **residual** skip (additive fan-in) and the squeeze-excite **gate**
(multiplicative fan-in). The `SHlo` backward inductive only has *unary*
backward constructors (`convBack`, `swishBack`, `denseRowBack`, …), but the
fan-ins are expressible with the existing *forward* elementwise combinators:
`addV` (`den (.addV a b) = den a + den b`) for the residual here, and
`layerScaleF` (Hadamard by a known activation vector) + `addV` for SE.

This file proves the residual case in general, then closes a fully concrete
instance (a dense body) end-to-end with no remaining hypothesis. -/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO
-- ════════════════════════════════════════════════════════════════
-- § Squeeze-excite (multiplicative fan-in) — the harder branch
-- ════════════════════════════════════════════════════════════════

/-- Backward graph for an SE block `x ↦ x ⊙ gate x`, given a subgraph
    `gateBack` rendering the gate sub-network's input-cotangent **at the
    SE-specific cotangent `x ⊙ dy`**. The main (identity) path contributes
    `gate x ⊙ dy`, rendered as a Hadamard (`layerScaleF`) of the cotangent
    by the gate activation; `addV` sums the two paths. The renderable image
    of `seBlock_has_vjp`'s `elemwiseProduct` (bi-cotangent) backward. -/
def seBlockBackGraph {n : Nat} (gateBack : SHlo n) (gateVal dy : Vec n) : SHlo n :=
  .addV (.layerScaleF "%segate" gateVal (.operand "%dy" dy)) gateBack

/-- **SE multiplicative fan-in backward faithfulness (general).**
    If `gateBack` denotes the gate path's VJP backward at the cotangent
    `x ⊙ dy` (`den gateBack = hg.backward x (x ⊙ dy)`), then the SE backward
    graph denotes the proven `seBlock_has_vjp` backward, which is
    `gate x ⊙ dy + gate.backward x (x ⊙ dy)`. Like the residual brick the
    proof is structural — the composition is delegated to `gateBack`, so the
    only definitional facts are `layerScaleF`/`addV` denotation and the
    identity main-path backward. -/
theorem seBlockBackGraph_faithful {n : Nat}
    (gate : Vec n → Vec n) (hg_diff : Differentiable ℝ gate) (hg : HasVJP gate)
    (x dy : Vec n) (gateBack : SHlo n)
    (hgb : den gateBack = hg.backward x (fun j => x j * dy j)) :
    den (seBlockBackGraph gateBack (gate x) dy)
      = (seBlock_has_vjp gate hg_diff hg).backward x dy := by
  funext i
  have hsum : den (seBlockBackGraph gateBack (gate x) dy) i
      = gate x i * dy i + den gateBack i := rfl
  rw [hsum, hgb]
  -- RHS = `elemwiseProduct_has_vjp id gate`'s backward
  --     = `id.backward x (gate x ⊙ dy) i + gate.backward x (x ⊙ dy) i`
  -- with `id.backward x u = u`; defeq under full transparency.
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The CONCRETE EfficientNet SE gate, assembled from per-op bricks
-- ════════════════════════════════════════════════════════════════

/-- The SE gate's backward graph. The gate is
    `broadcastFlat ∘ sigmoid ∘ dense W₂ ∘ swish ∘ dense W₁ ∘ GAP`, so its VJP
    backward (reverse order) is
    `gapBack ∘ denseᵀW₁ ∘ swishBack ∘ denseᵀW₂ ∘ sigmoidBack ∘ broadcastBack`,
    each per-op back applied at the matching forward activation. -/
noncomputable def seGateBackGraph {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    (x u : Vec (c * h * w)) : SHlo (c * h * w) :=
  .gapBack (c := c) (h := h) (w := w)
    (.dotOut "%seW1" W₁
      (.swishBack "%seSw" (dense W₁ b₁ (globalAvgPoolFlat c h w x))
        (.dotOut "%seW2" W₂
          (.sigmoidBack "%seSg"
              (dense W₂ b₂ (swish r (dense W₁ b₁ (globalAvgPoolFlat c h w x))))
            (.broadcastBack (c := c) (h := h) (w := w)
              (.operand "%seU" u))))))

/-- **The concrete SE gate's backward graph is faithful to `seGate_has_vjp`.**
    Assembles `gapBack`/`swishBack`/`sigmoidBack`/`broadcastBack` + the two dense
    `dotOut` backs into the gate's whole VJP. Closes the `gateBack` hypothesis of
    `seBlockBackGraph_faithful` for the real EfficientNet gate. -/
theorem seGate_backGraph_faithful {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    (x dy : Vec (c * h * w)) :
    den (seGateBackGraph W₁ b₁ W₂ b₂ x (fun j => x j * dy j))
      = (seGate_has_vjp (h := h) (w := w) W₁ b₁ W₂ b₂).backward x
          (fun j => x j * dy j) := rfl

-- ════════════════════════════════════════════════════════════════
-- § MBConv stage bricks: conv/depthwise-bn-swish + conv-bn (proj)
-- ════════════════════════════════════════════════════════════════

/-- **Function-level BatchNorm backward bridge.** `bnBack` denotes `bn_grad_input`,
    which is NOT rfl-equal to `(bn_has_vjp …).backward` (the witness is built via
    a `rw [bnForward_eq_compose]` cast). They agree through the canonical VJP sum:
    `bnBack_faithful` gives the `∑ pdiv` form and `bn_has_vjp.correct` matches it.
    This lemma is the one non-`rfl` bridge the bn-containing stages need. -/
theorem bnBack_faithful_fn {n : Nat} (gN xN es : String) (ε γ β : ℝ) (hε : 0 < ε)
    (x : Vec n) (e : SHlo n) :
    den (SHlo.bnBack gN xN es ε γ x e) = (bn_has_vjp n ε γ β hε).backward x (den e) := by
  funext i
  rw [bnBack_faithful gN xN es ε γ β hε x e i]
  exact ((bn_has_vjp n ε γ β hε).correct x (den e) i).symm

/-- conv → bn → swish backward graph (the MBConv expand stage), at input `x`,
    cotangent subgraph `e`: `convBack ∘ bnBack ∘ swishBack`. -/
noncomputable def convBnSwishBackGraph {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ)
    (x : Vec (ic * h * w)) (e : SHlo (oc * h * w)) : SHlo (ic * h * w) :=
  .convBack "%cbsW" W b x
    (.bnBack "%cbsG" "%cbsX" "cbsE" ε γ (flatConv W b x)
      (.swishBack "%cbsSw" (bnForward (oc * h * w) ε γ β (flatConv W b x)) e))

theorem convBnSwishBackGraph_faithful {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε)
    (x : Vec (ic * h * w)) (e : SHlo (oc * h * w)) :
    den (convBnSwishBackGraph W b ε γ β x e)
      = (convBnSwish_has_vjp W b ε γ β hε).backward x (den e) := by
  -- `vjp_comp` itself, not `vjp_comp_backward`: `convBn_has_vjp`'s conv witness is typed at
  -- `fun v => (conv2d W b (Tensor3.unflatten v)).flatten`, only defeq to `flatConv W b`, so the
  -- backward lemma's pattern does not instantiate.
  simp only [convBnSwishBackGraph, convBnSwish_has_vjp, convBn_has_vjp, vjp_comp,
    convBack_faithful, swishBack_faithful, bnBack_faithful_fn (β := β) (hε := hε),
    Function.comp_apply]

/-- depthwise → bn → swish backward graph (the MBConv depthwise stage). -/
noncomputable def dwBnSwishBackGraph {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε γ β : ℝ)
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  .depthwiseBack "%dwW" W b x
    (.bnBack "%dwG" "%dwX" "dwE" ε γ (depthwiseFlat W b x)
      (.swishBack "%dwSw" (bnForward (c * h * w) ε γ β (depthwiseFlat W b x)) e))

theorem dwBnSwishBackGraph_faithful {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε γ β : ℝ) (hε : 0 < ε)
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) :
    den (dwBnSwishBackGraph W b ε γ β x e)
      = (dwBnSwish_has_vjp W b ε γ β hε).backward x (den e) := by
  simp only [dwBnSwishBackGraph, dwBnSwish_has_vjp, vjp_comp_backward,
    depthwiseBack_faithful, swishBack_faithful, bnBack_faithful_fn (β := β) (hε := hε),
    Function.comp_apply]

/-- conv → bn backward graph (the MBConv project stage, no swish). -/
noncomputable def convBnBackGraph {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ _β : ℝ)
    (x : Vec (ic * h * w)) (e : SHlo (oc * h * w)) : SHlo (ic * h * w) :=
  .convBack "%pW" W b x
    (.bnBack "%pG" "%pX" "pE" ε γ (flatConv W b x) e)

theorem convBnBackGraph_faithful {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε)
    (x : Vec (ic * h * w)) (e : SHlo (oc * h * w)) :
    den (convBnBackGraph W b ε γ β x e)
      = (convBn_has_vjp W b ε γ β hε).backward x (den e) := by
  simp only [convBnBackGraph, convBn_has_vjp, vjp_comp, convBack_faithful,
    bnBack_faithful_fn (β := β) (hε := hε)]  -- `vjp_comp`: see `convBnSwishBackGraph_faithful`

-- ════════════════════════════════════════════════════════════════
-- § SE, subgraph-cotangent form (for mid-chain use inside the block)
-- ════════════════════════════════════════════════════════════════

/-- SE gate backward graph taking a cotangent **subgraph** `e` (not a `%dy`
    operand), so it can sit mid-chain inside the MBConv body. -/
noncomputable def seGateBackGraphE {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  .gapBack (c := c) (h := h) (w := w)
    (.dotOut "%seW1" W₁
      (.swishBack "%seSw" (dense W₁ b₁ (globalAvgPoolFlat c h w x))
        (.dotOut "%seW2" W₂
          (.sigmoidBack "%seSg"
              (dense W₂ b₂ (swish r (dense W₁ b₁ (globalAvgPoolFlat c h w x))))
            (.broadcastBack (c := c) (h := h) (w := w) e)))))

theorem seGateBackGraphE_faithful {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) :
    den (seGateBackGraphE W₁ b₁ W₂ b₂ x e)
      = (seGate_has_vjp (h := h) (w := w) W₁ b₁ W₂ b₂).backward x (den e) := rfl

/-- SE block backward graph, subgraph-cotangent form: main path
    `gate(x) ⊙ (den e)` via `layerScaleF`, gate path fed `x ⊙ (den e)`. -/
noncomputable def seBlockFullBackGraphE {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  .addV (.layerScaleF "%segate" (seGate (h := h) (w := w) W₁ b₁ W₂ b₂ x) e)
    (seGateBackGraphE W₁ b₁ W₂ b₂ x (.layerScaleF "%seInput" x e))

theorem seBlockFullBackGraphE_faithful {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) :
    den (seBlockFullBackGraphE W₁ b₁ W₂ b₂ x e)
      = (seBlockFull_has_vjp (h := h) (w := w) W₁ b₁ W₂ b₂).backward x (den e) := by
  have hg : den (seGateBackGraphE W₁ b₁ W₂ b₂ x (.layerScaleF "%seInput" x e))
      = (seGate_has_vjp (h := h) (w := w) W₁ b₁ W₂ b₂).backward x
          (fun j => x j * den e j) := seGateBackGraphE_faithful W₁ b₁ W₂ b₂ x _
  funext i
  have hsum : den (seBlockFullBackGraphE W₁ b₁ W₂ b₂ x e) i
      = seGate (h := h) (w := w) W₁ b₁ W₂ b₂ x i * den e i
        + den (seGateBackGraphE W₁ b₁ W₂ b₂ x (.layerScaleF "%seInput" x e)) i := rfl
  rw [hsum, hg]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Whole MBConv residual block: assemble all stages + skip
-- ════════════════════════════════════════════════════════════════

/-- The MBConv body backward graph `E⁻¹ ∘ D⁻¹ ∘ S⁻¹ ∘ P⁻¹`, each stage at its
    cumulative forward activation. `cin = cout = c` (stride-1 residual block). -/
noncomputable def mbconvBodyBackGraph {c cmid h w kHe kWe kHd kWd kHp kWp r : Nat}
    (We : Kernel4 cmid c kHe kWe) (be : Vec cmid) (εe γe βe : ℝ)
    (Wd : DepthwiseKernel cmid kHd kWd) (bd : Vec cmid) (εd γd βd : ℝ)
    (Ws₁ : Mat cmid r) (bs₁ : Vec r) (Ws₂ : Mat r cmid) (bs₂ : Vec cmid)
    (Wp : Kernel4 c cmid kHp kWp) (bp : Vec c) (εp γp βp : ℝ)
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  let eOut := swish (cmid*h*w) (bnForward (cmid*h*w) εe γe βe (flatConv We be x))
  let dOut := swish (cmid*h*w) (bnForward (cmid*h*w) εd γd βd (depthwiseFlat Wd bd eOut))
  let sOut := seBlockFull (h := h) (w := w) Ws₁ bs₁ Ws₂ bs₂ dOut
  convBnSwishBackGraph We be εe γe βe x
    (dwBnSwishBackGraph Wd bd εd γd βd eOut
      (seBlockFullBackGraphE Ws₁ bs₁ Ws₂ bs₂ dOut
        (convBnBackGraph Wp bp εp γp βp sOut e)))

theorem mbconvBodyBackGraph_faithful {c cmid h w kHe kWe kHd kWd kHp kWp r : Nat}
    (We : Kernel4 cmid c kHe kWe) (be : Vec cmid) (εe γe βe : ℝ) (hεe : 0 < εe)
    (Wd : DepthwiseKernel cmid kHd kWd) (bd : Vec cmid) (εd γd βd : ℝ) (hεd : 0 < εd)
    (Ws₁ : Mat cmid r) (bs₁ : Vec r) (Ws₂ : Mat r cmid) (bs₂ : Vec cmid)
    (Wp : Kernel4 c cmid kHp kWp) (bp : Vec c) (εp γp βp : ℝ) (hεp : 0 < εp)
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) :
    den (mbconvBodyBackGraph We be εe γe βe Wd bd εd γd βd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp x e)
      = (mbconvBody_has_vjp We be εe γe βe hεe Wd bd εd γd βd hεd
          Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp hεp).backward x (den e) := by
  rw [mbconvBodyBackGraph, convBnSwishBackGraph_faithful (hε := hεe),
      dwBnSwishBackGraph_faithful (hε := hεd),
      seBlockFullBackGraphE_faithful,
      convBnBackGraph_faithful (hε := hεp)]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Capstone: the whole batched MBConv residual block
-- ════════════════════════════════════════════════════════════════

/-- The batched MBConv body's VJP — `projB ∘ seB ∘ dwbsB ∘ cbsB`, reconstructed
    as the exact `vjp_comp` chain `mbResidFwdB_has_vjp` builds inline (`vBody`). -/
noncomputable def mbBodyB_has_vjp (N : Nat) {c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c) :
    HasVJP (projB N (h := h) (w := w) Wp bp εp γp βp ∘ seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ ∘
            dwbsB N (h := h) (w := w) Wd bd εd γd βd ∘ cbsB N (h := h) (w := w) We be εe γe βe) :=
  let dE := cbsB_differentiable N (h := h) (w := w) We be εe hεe γe βe
  let dDw := dwbsB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd
  let dSe := seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂
  vjp_comp _ _ (dSe.comp (dDw.comp dE)) (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp)
    (vjp_comp _ _ (dDw.comp dE) dSe
      (vjp_comp _ _ dE dDw (cbsB_has_vjp N (h := h) (w := w) We be εe hεe γe βe)
        (dwbsB_has_vjp N (h := h) (w := w) Wd bd εd hεd γd βd))
      (seB_has_vjp N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂))
    (projB_has_vjp N (h := h) (w := w) Wp bp εp hεp γp βp)

/-- The batched MBConv body backward graph: the four stage graphs chained at their
    cumulative forward activations (`cbsB⁻¹ ∘ dwbsB⁻¹ ∘ seB⁻¹ ∘ projB⁻¹`). -/
noncomputable def mbBodyBackBatchedGraph {N c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (γp βp : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  let xE := cbsB N (h := h) (w := w) We be εe γe βe x
  let xD := dwbsB N (h := h) (w := w) Wd bd εd γd βd xE
  let xS := seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ xD
  cbsBackBatchedGraph We be εe γe βe x
    (dwbsBackBatchedGraph Wd bd εd γd βd xE
      (.seBackBatched (N := N) "%seW1" "%seb1" "%seW2" "%seb2" "%seX" Wz₁ bz₁ Wz₂ bz₂ xD
        (projBackBatchedGraph Wp bp εp γp βp xS e)))

theorem mbBodyBackBatchedGraph_faithful {N c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) :
    den (mbBodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp x e)
      = (mbBodyB_has_vjp N We be εe hεe γe βe Wd bd εd hεd γd βd
          Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp).backward x (den e) := by
  rw [mbBodyBackBatchedGraph, cbsBackBatchedGraph_faithful (hε := hεe),
      dwbsBackBatchedGraph_faithful (hε := hεd), seBackBatched_faithful,
      projBackBatchedGraph_faithful (hε := hεp)]
  simp only [mbBodyB_has_vjp, vjp_comp_backward, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § Capstone: the batched DOWNSAMPLE MBConv body (strided depthwise, NO residual)
-- ════════════════════════════════════════════════════════════════

/-- The batched downsample MBConv body's VJP — `projB ∘ seB ∘ dwbsSB ∘ cbsB`, the
    stride-2 analogue of `mbBodyB_has_vjp` (swaps the stride-1 `dwbsB` depthwise
    stage for the STRIDED `dwbsSB`). The expand `cbsB` runs at the larger `2h×2w`,
    the strided depthwise then halves spatial to `h×w`; the rest at `h×w`. No
    residual (spatial/channels change), so this is the body alone — reconstructed
    as the exact `vjp_comp` chain `mbStridedFwdB_has_vjp` builds inline. -/
noncomputable def mbDownBodyB_has_vjp (N : Nat) {ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    HasVJP (projB N (h := h) (w := w) Wp bp εp γp βp ∘ seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ ∘
            dwbsSB N (h := h) (w := w) Wd bd εd γd βd ∘
            cbsB N (h := 2 * h) (w := 2 * w) We be εe γe βe) :=
  let dE := cbsB_differentiable N (h := 2 * h) (w := 2 * w) We be εe hεe γe βe
  let dDw := dwbsSB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd
  let dSe := seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂
  vjp_comp _ _ (dSe.comp (dDw.comp dE)) (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp)
    (vjp_comp _ _ (dDw.comp dE) dSe
      (vjp_comp _ _ dE dDw (cbsB_has_vjp N (h := 2 * h) (w := 2 * w) We be εe hεe γe βe)
        (dwbsSB_has_vjp N (h := h) (w := w) Wd bd εd hεd γd βd))
      (seB_has_vjp N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂))
    (projB_has_vjp N (h := h) (w := w) Wp bp εp hεp γp βp)

/-- The batched downsample MBConv body backward graph: the four stage graphs chained
    at their cumulative forward activations (`cbsB⁻¹ ∘ dwbsSB⁻¹ ∘ seB⁻¹ ∘ projB⁻¹`).
    Stride-2 analogue of `mbBodyBackBatchedGraph` (strided depthwise stage graph). -/
noncomputable def mbDownBodyBackBatchedGraph {N ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    SHlo (N * (ic * (2 * h) * (2 * w))) :=
  let xE := cbsB N (h := 2 * h) (w := 2 * w) We be εe γe βe x
  let xD := dwbsSB N (h := h) (w := w) Wd bd εd γd βd xE
  let xS := seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ xD
  cbsBackBatchedGraph We be εe γe βe x
    (dwbsSBackBatchedGraph Wd bd εd γd βd xE
      (.seBackBatched (N := N) "%seW1" "%seb1" "%seW2" "%seb2" "%seX" Wz₁ bz₁ Wz₂ bz₂ xD
        (projBackBatchedGraph Wp bp εp γp βp xS e)))

/-- **CAPSTONE — the batched EfficientNet DOWNSAMPLE MBConv body: backward graph ↔
    the proven `mbDownBodyB_has_vjp`.** The four batched stage backward graphs
    (`cbsB`/`dwbsSB`/`seB`/`projB`) chained at their forward activations, proven
    equal to the downsample-body VJP. The stride-2 analogue of
    `mbBodyBackBatchedGraph_faithful` (no residual skip — the downsample block
    changes spatial/channels, so the body alone is the block). EfficientNet uses
    swish (a global VJP), so this stays in the clean global `HasVJP`/`vjp_comp`
    form (no `_at` recompute, unlike r34/mnv2's relu blocks). -/
theorem mbDownBodyBackBatchedGraph_faithful {N ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    den (mbDownBodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp x e)
      = (mbDownBodyB_has_vjp N We be εe hεe γe βe Wd bd εd hεd γd βd
          Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp).backward x (den e) := by
  rw [mbDownBodyBackBatchedGraph, cbsBackBatchedGraph_faithful (hε := hεe),
      dwbsSBackBatchedGraph_faithful (hε := hεd), seBackBatched_faithful,
      projBackBatchedGraph_faithful (hε := hεp)]
  simp only [mbDownBodyB_has_vjp, vjp_comp_backward, Function.comp_apply]

/-- The whole batched MBConv residual block backward graph (body + identity skip). -/
noncomputable def mbResidBlockBackBatchedGraph {N c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (γp βp : Vec c)
    (x : Vec (N * (c * h * w))) (ecot : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  residualBackGraph
    (mbBodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp
      x ecot) ecot

/-- **CAPSTONE — the whole batched EfficientNet MBConv residual block: backward
    graph ↔ the proven `mbResidFwdB_has_vjp`.** The four batched stage backward
    graphs (`cbsB`/`dwbsB`/`seB`/`projB`) chained at their forward activations +
    the identity skip, proven equal to the repo's batched MBConv block VJP. -/
theorem mbResidBlockBackBatchedGraph_faithful {N c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c)
    (x : Vec (N * (c * h * w))) (ecot : SHlo (N * (c * h * w))) :
    den (mbResidBlockBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp x ecot)
      = (mbResidFwdB_has_vjp N We be εe hεe γe βe Wd bd εd hεd γd βd
          Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp).backward x (den ecot) :=
  residualBackGraph_faithful
    (projB N (h := h) (w := w) Wp bp εp γp βp ∘ seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ ∘
      dwbsB N (h := h) (w := w) Wd bd εd γd βd ∘ cbsB N (h := h) (w := w) We be εe γe βe)
    ((projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp).comp
      ((seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂).comp
        ((dwbsB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd).comp
          (cbsB_differentiable N (h := h) (w := w) We be εe hεe γe βe))))
    (mbBodyB_has_vjp N We be εe hεe γe βe Wd bd εd hεd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp)
    x ecot
    (mbBodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp
      x ecot)
    (mbBodyBackBatchedGraph_faithful We be εe hεe γe βe Wd bd εd hεd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp x ecot)

end Proofs.StableHLO
