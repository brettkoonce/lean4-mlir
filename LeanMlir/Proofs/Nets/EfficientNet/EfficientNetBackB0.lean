import LeanMlir.Proofs.Nets.EfficientNet.EfficientNet
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetChainClose

/-! # EfficientNet-B0 block backward graphs — the residual and squeeze-excite fan-ins

Backward StableHLO graphs for B0's MBConv blocks and the theorems that they denote the proven
VJPs. B0 has two branching ops the dense-chain nets do not: the residual skip (additive fan-in)
and the squeeze-excite gate (multiplicative fan-in). `SHlo`'s backward constructors are unary, but
both fan-ins are spelled with forward elementwise combinators: `addV` for the residual, and
`layerScaleF` (Hadamard by a known activation) + `addV` for the SE gate.

## Contents

* SE: `seBlockBackGraph`, the generic multiplicative fan-in;
* `bnBack_faithful_fn`, the function-level BatchNorm backward bridge;
* batched capstones: `mbBodyBackBatchedGraph` (stride-1 body), `mbDownBodyBackBatchedGraph`
  (strided depthwise, no residual) and `mbResidBlockBackBatchedGraph` (body + skip).

Each `…_faithful` states `den (graph) = ` the certified backward. -/

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
    of `seBlockHasVJP`'s `elemwiseProduct` (bi-cotangent) backward. -/
def seBlockBackGraph {n : Nat} (gateBack : SHlo n) (gateVal dy : Vec n) : SHlo n :=
  .addV (.layerScaleF "%segate" gateVal (.operand "%dy" dy)) gateBack

/-- **SE multiplicative fan-in backward faithfulness (general).**
    If `gateBack` denotes the gate path's VJP backward at the cotangent
    `x ⊙ dy` (`den gateBack = hg.backward x (x ⊙ dy)`), then the SE backward
    graph denotes the proven `seBlockHasVJP` backward, which is
    `gate x ⊙ dy + gate.backward x (x ⊙ dy)`. Like the residual brick the
    proof is structural — the composition is delegated to `gateBack`, so the
    only definitional facts are `layerScaleF`/`addV` denotation and the
    identity main-path backward. -/
theorem seBlockBackGraph_faithful {n : Nat}
    (gate : Vec n → Vec n) (hg_diff : Differentiable ℝ gate) (hg : HasVJP gate)
    (x dy : Vec n) (gateBack : SHlo n)
    (hgb : den gateBack = hg.backward x (fun j => x j * dy j)) :
    den (seBlockBackGraph gateBack (gate x) dy)
      = (seBlockHasVJP gate hg_diff hg).backward x dy := by
  funext i
  have hsum : den (seBlockBackGraph gateBack (gate x) dy) i
      = gate x i * dy i + den gateBack i := rfl
  rw [hsum, hgb]
  -- RHS = `elemwiseProductHasVJP id gate`'s backward
  --     = `id.backward x (gate x ⊙ dy) i + gate.backward x (x ⊙ dy) i`
  -- with `id.backward x u = u`; defeq under full transparency.
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The function-level BatchNorm backward bridge
-- ════════════════════════════════════════════════════════════════

/-- **Function-level BatchNorm backward bridge.** `bnBack` denotes `bnGradInput`,
    which is NOT rfl-equal to `(bnHasVJP …).backward` (the witness is built via
    a `rw [bnForward_eq_compose]` cast). They agree through the canonical VJP sum:
    `bnBack_faithful` gives the `∑ pdiv` form and `bnHasVJP.correct` matches it.
    This lemma is the one non-`rfl` bridge the bn-containing stages need. -/
theorem bnBack_faithful_fn {n : Nat} (gN xN es : String) (ε γ β : ℝ) (hε : 0 < ε)
    (x : Vec n) (e : SHlo n) :
    den (SHlo.bnBack gN xN es ε γ x e) = (bnHasVJP n ε γ β hε).backward x (den e) := by
  funext i
  rw [bnBack_faithful gN xN es ε γ β hε x e i]
  exact ((bnHasVJP n ε γ β hε).correct x (den e) i).symm

-- ════════════════════════════════════════════════════════════════
-- § Capstone: the whole batched MBConv residual block
-- ════════════════════════════════════════════════════════════════

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
      = (mbExpFwdBHasVJP N We be εe hεe γe βe Wd bd εd hεd γd βd
          Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp).backward x (den e) := by
  rw [mbBodyBackBatchedGraph, cbsBackBatchedGraph_faithful (hε := hεe),
      dwbsBackBatchedGraph_faithful (hε := hεd), seBackBatched_faithful,
      projBackBatchedGraph_faithful (hε := hεp)]
  simp only [mbExpFwdBHasVJP, vjpComp_backward, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § Capstone: the batched DOWNSAMPLE MBConv body (strided depthwise, NO residual)
-- ════════════════════════════════════════════════════════════════

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
    the proven `mbStridedFwdBHasVJP`.** The four batched stage backward graphs
    (`cbsB`/`dwbsSB`/`seB`/`projB`) chained at their forward activations, proven
    equal to the downsample-body VJP. The stride-2 analogue of
    `mbBodyBackBatchedGraph_faithful` (no residual skip — the downsample block
    changes spatial/channels, so the body alone is the block). EfficientNet uses
    swish (a global VJP), so this stays in the clean global `HasVJP`/`vjpComp`
    form (no `_at` recompute, unlike r34/mnv2's relu blocks). -/
theorem mbDownBodyBackBatchedGraph_faithful {N ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    den (mbDownBodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp x e)
      = (mbStridedFwdBHasVJP N We be εe hεe γe βe Wd bd εd hεd γd βd
          Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp).backward x (den e) := by
  rw [mbDownBodyBackBatchedGraph, cbsBackBatchedGraph_faithful (hε := hεe),
      dwbsSBackBatchedGraph_faithful (hε := hεd), seBackBatched_faithful,
      projBackBatchedGraph_faithful (hε := hεp)]
  simp only [mbStridedFwdBHasVJP, vjpComp_backward, Function.comp_apply]

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
    graph ↔ the proven `mbResidFwdBHasVJP`.** The four batched stage backward
    graphs (`cbsB`/`dwbsB`/`seB`/`projB`) chained at their forward activations +
    the identity skip, proven equal to the repo's batched MBConv block VJP. -/
theorem mbResidBlockBackBatchedGraph_faithful {N c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c)
    (x : Vec (N * (c * h * w))) (ecot : SHlo (N * (c * h * w))) :
    den (mbResidBlockBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp x ecot)
      = (mbResidFwdBHasVJP N We be εe hεe γe βe Wd bd εd hεd γd βd
          Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp).backward x (den ecot) :=
  residualBackGraph_faithful
    (projB N (h := h) (w := w) Wp bp εp γp βp ∘ seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ ∘
      dwbsB N (h := h) (w := w) Wd bd εd γd βd ∘ cbsB N (h := h) (w := w) We be εe γe βe)
    ((projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp).comp
      ((seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂).comp
        ((dwbsB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd).comp
          (cbsB_differentiable N (h := h) (w := w) We be εe hεe γe βe))))
    (mbExpFwdBHasVJP N We be εe hεe γe βe Wd bd εd hεd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp)
    x ecot
    (mbBodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp
      x ecot)
    (mbBodyBackBatchedGraph_faithful We be εe hεe γe βe Wd bd εd hεd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp x ecot)

end Proofs.StableHLO
