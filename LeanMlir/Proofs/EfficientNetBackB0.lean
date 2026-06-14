import LeanMlir.Proofs.Residual
import LeanMlir.Proofs.SE
import LeanMlir.Proofs.EfficientNet
import LeanMlir.Proofs.StableHLO
import LeanMlir.Proofs.EfficientNetChainClose

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

/-- Backward graph for a residual block `x ↦ x + f x`, given a subgraph
    `fBack` that renders the body `f`'s input-cotangent. The identity skip
    contributes the cotangent verbatim (`%dy`); `addV` sums the two paths.
    This is the renderable image of `residual_has_vjp`'s `biPath` backward. -/
def residualBackGraph {n : Nat} (fBack : SHlo n) (dy : Vec n) : SHlo n :=
  .addV fBack (.operand "%dy" dy)

/-- **Residual additive-fan-in backward faithfulness (general).**
    If `fBack` denotes the body's VJP backward (`den fBack = hf.backward x dy`),
    then the residual backward graph denotes the proven `residual_has_vjp`
    backward, which is `f.backward x dy + dy`. The proof is structural — the
    only definitional facts are `den (addV a b) = den a + den b` and the
    identity skip's `backward = dy` — so it composes without a whole-net
    terminal `rfl`. -/
theorem residualBackGraph_faithful {n : Nat}
    (f : Vec n → Vec n) (hf_diff : Differentiable ℝ f) (hf : HasVJP f)
    (x dy : Vec n) (fBack : SHlo n)
    (hfb : den fBack = hf.backward x dy) :
    den (residualBackGraph fBack dy)
      = (residual_has_vjp f hf_diff hf).backward x dy := by
  funext i
  have hsum : den (residualBackGraph fBack dy) i = den fBack i + dy i := rfl
  rw [hsum, hfb]
  -- RHS = `biPath_has_vjp f id`'s backward = `f.backward x dy i + id.backward x dy i`
  -- with `id.backward x dy = dy`; defeq but needs full-transparency unfolding.
  rfl

/-- **Fully concrete instance — no remaining hypothesis.** A residual block
    `x ↦ x + dense W b x` (square `W`): its backward graph `addV (backGraph W dy) %dy`
    denotes the proven `residual_has_vjp (dense W b)` backward. The body hypothesis
    of `residualBackGraph_faithful` is discharged by the existing per-op
    `backGraph_faithful`. This is the smallest end-to-end fan-in backward
    faithfulness: a *branching* backward graph proven equal to a composed VJP. -/
theorem residual_dense_backGraph_faithful {n : Nat}
    (W : Mat n n) (b x dy : Vec n) :
    den (residualBackGraph (backGraph W dy) dy)
      = (residual_has_vjp (dense W b) (dense_differentiable W b)
          (dense_has_vjp W b)).backward x dy :=
  residualBackGraph_faithful (dense W b) (dense_differentiable W b)
    (dense_has_vjp W b) x dy (backGraph W dy)
    (backGraph_faithful (W := W) (b := b) (x := x) dy)

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

/-- **Fully concrete SE instance — no remaining hypothesis** (linear gate).
    `seBlock (dense W b)` (square `W`): its multiplicative-fan-in backward
    graph denotes the proven `seBlock_has_vjp` backward. The gate-path
    hypothesis is discharged by `backGraph_faithful` at the SE cotangent
    `x ⊙ dy`. Exercises the product-rule bi-cotangent structure end-to-end;
    the GAP+sigmoid gate of EfficientNet additionally needs `sigmoidBack`
    (present) and a `gapBack` op (the remaining IR primitive). -/
theorem se_dense_backGraph_faithful {n : Nat}
    (W : Mat n n) (b x dy : Vec n) :
    den (seBlockBackGraph (backGraph W (fun j => x j * dy j)) (dense W b x) dy)
      = (seBlock_has_vjp (dense W b) (dense_differentiable W b)
          (dense_has_vjp W b)).backward x dy :=
  seBlockBackGraph_faithful (dense W b) (dense_differentiable W b)
    (dense_has_vjp W b) x dy (backGraph W (fun j => x j * dy j))
    (backGraph_faithful (W := W) (b := b) (x := x) (fun j => x j * dy j))

-- ════════════════════════════════════════════════════════════════
-- § `gapBack`: the squeeze's input-cotangent (new core-IR primitive)
-- ════════════════════════════════════════════════════════════════

/-- **GAP-backward (VJP) faithfulness.** The new `gapBack` SHlo op denotes the
    proven `globalAvgPoolFlat_has_vjp` backward: the per-channel cotangent
    broadcast back over the H×W grid and scaled by `1/(h·w)`. This is the last
    *per-op* brick the EfficientNet SE **gate** sub-network needs — its squeeze
    is a global-average-pool, whose backward had no renderable op before.
    Combined with the existing `sigmoidBack`/`denseRowBack`, the concrete
    GAP→dense→sigmoid gate's backward is now expressible, which discharges the
    `gateBack` hypothesis of `seBlockBackGraph_faithful` for the real SE gate. -/
theorem gapBack_faithful {c h w : Nat} (v : Vec (c * h * w)) (e : SHlo c) :
    den (SHlo.gapBack (c := c) (h := h) (w := w) e)
      = (globalAvgPoolFlat_has_vjp c h w).backward v (den e) := rfl

/-- **broadcastBack (VJP = sum-over-spatial) faithfulness.** Denotes the
    adjoint of `broadcastFlat` — the gate's outermost backward op. -/
theorem broadcastBack_faithful {c h w : Nat} (v : Vec c) (e : SHlo (c * h * w)) :
    den (SHlo.broadcastBack (c := c) (h := h) (w := w) e)
      = (broadcastFlat_has_vjp c h w).backward v (den e) := rfl

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

/-- **Concrete EfficientNet SE block, whole backward graph ↔ proven VJP.**
    `seBlockFull = x ⊙ seGate(x)` with the real GAP→reduce→swish→expand→sigmoid
    gate: its multiplicative-fan-in backward graph (main path `seGate(x) ⊙ dy`
    via `layerScaleF`, plus the assembled gate-path `seGateBackGraph` at the
    cotangent `x ⊙ dy`) denotes the proven `seBlockFull_has_vjp` backward. No
    remaining hypothesis. -/
theorem seBlockFull_backGraph_faithful {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    (x dy : Vec (c * h * w)) :
    den (seBlockBackGraph (seGateBackGraph W₁ b₁ W₂ b₂ x (fun j => x j * dy j))
          (seGate (h := h) (w := w) W₁ b₁ W₂ b₂ x) dy)
      = (seBlockFull_has_vjp (h := h) (w := w) W₁ b₁ W₂ b₂).backward x dy :=
  seBlockBackGraph_faithful (seGate (h := h) (w := w) W₁ b₁ W₂ b₂)
    (seGate_differentiable W₁ b₁ W₂ b₂) (seGate_has_vjp W₁ b₁ W₂ b₂)
    x dy (seGateBackGraph W₁ b₁ W₂ b₂ x (fun j => x j * dy j))
    (seGate_backGraph_faithful W₁ b₁ W₂ b₂ x dy)

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
  simp only [dwBnSwishBackGraph, dwBnSwish_has_vjp, vjp_comp,
    depthwiseBack_faithful, swishBack_faithful, bnBack_faithful_fn (β := β) (hε := hε),
    Function.comp_apply]

/-- conv → bn backward graph (the MBConv project stage, no swish). -/
noncomputable def convBnBackGraph {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ)
    (x : Vec (ic * h * w)) (e : SHlo (oc * h * w)) : SHlo (ic * h * w) :=
  .convBack "%pW" W b x
    (.bnBack "%pG" "%pX" "pE" ε γ (flatConv W b x) e)

theorem convBnBackGraph_faithful {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε)
    (x : Vec (ic * h * w)) (e : SHlo (oc * h * w)) :
    den (convBnBackGraph W b ε γ β x e)
      = (convBn_has_vjp W b ε γ β hε).backward x (den e) := by
  simp only [convBnBackGraph, convBn_has_vjp, vjp_comp,
    convBack_faithful, bnBack_faithful_fn (β := β) (hε := hε)]

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

/-- The whole MBConv residual block backward graph (body + identity skip). -/
noncomputable def mbconvResidualBackGraph {c cmid h w kHe kWe kHd kWd kHp kWp r : Nat}
    (We : Kernel4 cmid c kHe kWe) (be : Vec cmid) (εe γe βe : ℝ)
    (Wd : DepthwiseKernel cmid kHd kWd) (bd : Vec cmid) (εd γd βd : ℝ)
    (Ws₁ : Mat cmid r) (bs₁ : Vec r) (Ws₂ : Mat r cmid) (bs₂ : Vec cmid)
    (Wp : Kernel4 c cmid kHp kWp) (bp : Vec c) (εp γp βp : ℝ)
    (x dy : Vec (c * h * w)) : SHlo (c * h * w) :=
  residualBackGraph
    (mbconvBodyBackGraph We be εe γe βe Wd bd εd γd βd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp x
      (.operand "%dy" dy)) dy

/-- **The whole per-example EfficientNet MBConv residual block: backward graph
    ↔ proven VJP, no hypotheses beyond `ε>0`.** Assembles all stage bricks
    (`convBnSwish`/`dwBnSwish`/`SE`/`convBn`) + the identity skip into the proven
    `mbconvResidual_has_vjp` backward. -/
theorem mbconvResidual_backGraph_faithful {c cmid h w kHe kWe kHd kWd kHp kWp r : Nat}
    (We : Kernel4 cmid c kHe kWe) (be : Vec cmid) (εe γe βe : ℝ) (hεe : 0 < εe)
    (Wd : DepthwiseKernel cmid kHd kWd) (bd : Vec cmid) (εd γd βd : ℝ) (hεd : 0 < εd)
    (Ws₁ : Mat cmid r) (bs₁ : Vec r) (Ws₂ : Mat r cmid) (bs₂ : Vec cmid)
    (Wp : Kernel4 c cmid kHp kWp) (bp : Vec c) (εp γp βp : ℝ) (hεp : 0 < εp)
    (x dy : Vec (c * h * w)) :
    den (mbconvResidualBackGraph We be εe γe βe Wd bd εd γd βd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp x dy)
      = (mbconvResidual_has_vjp We be εe γe βe hεe Wd bd εd γd βd hεd
          Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp hεp).backward x dy :=
  residualBackGraph_faithful
    (mbconvBody We be εe γe βe Wd bd εd γd βd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp)
    (mbconvBody_differentiable We be εe γe βe hεe Wd bd εd γd βd hεd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp hεp)
    (mbconvBody_has_vjp We be εe γe βe hεe Wd bd εd γd βd hεd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp hεp)
    x dy
    (mbconvBodyBackGraph We be εe γe βe Wd bd εd γd βd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp x (.operand "%dy" dy))
    (mbconvBodyBackGraph_faithful We be εe γe βe hεe Wd bd εd γd βd hεd
      Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp hεp x (.operand "%dy" dy))

-- ════════════════════════════════════════════════════════════════
-- § Batched lifting (start): true batch-norm backward primitive
-- ════════════════════════════════════════════════════════════════

/-- **`bnBatchBack` (true batch-norm backward) faithfulness.** The first
    batched-backward primitive: `bnBatchBack` denotes the proven
    `bnBatchTensor4` VJP backward (batch-COUPLED batch-norm on `[N,C,H,W]`,
    reduce over `[0,2,3]` per channel) via the renderable three-term
    `bnBatchTensor4_grad_input`. This is the genuinely-new op the batched MBConv
    stages need (their bn is `bnBatchLA`, not a per-example `batchMap`); the
    other batched stages (conv/depthwise/SE) are `batchMap` of the per-example
    backwards already proven above. The `bnBatchLA` layout-reindex wrapper to the
    network's `N·(oc·h·w)` index is a thin remaining layer. -/
theorem bnBatchBack_faithful {N oc h w : Nat} (gN xN es : String)
    (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (x : Vec (N * (oc * (h * w)))) (e : SHlo (N * (oc * (h * w)))) :
    den (SHlo.bnBatchBack gN xN es ε γ x e)
      = (bnBatchTensor4_has_vjp N oc h w ε hε γ β).backward x (den e) := by
  funext i
  show bnBatchTensor4_grad_input N oc h w ε γ x (den e) i = _
  rw [bnBatchTensor4_grad_input_correct N oc h w ε hε γ β x (den e) i,
      ← bnBatchTensor4_has_vjp_correct N oc h w ε hε γ β x (den e) i]

/-- **Batched conv input-VJP faithfulness.** `convBackBatched` denotes the proven
    VJP of the batched conv `batchMap N (flatConv W b)` — i.e. the per-example
    conv input-grad applied independently across the batch. Conv is linear, so
    its backward ignores the forward activation; the batched backward is a plain
    `batchMap` of the per-example backward, matching `batchMap_has_vjp`. The
    second batch-separable stage brick (after `seB`); together with `bnBatchBack`
    these are the batched MBConv's per-stage backward pieces. -/
theorem convBackBatched_faithful {N ic oc h w kH kW : Nat} (wN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (v : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) :
    den (SHlo.convBackBatched (N := N) wN W b e)
      = (batchMap_has_vjp (flatConv W b) (flatConv_has_vjp W b)
          (flatConv_differentiable W b)).backward v (den e) := by
  funext idx
  -- The transport in `batchMap_has_vjp` unfolds to the per-example conv backward
  -- on each row; both sides then differ only in the (discarded) forward
  -- activation arg, since conv is linear (`conv2d_has_vjp3.backward` ignores it).
  simp only [den, batchMap, batchMap_has_vjp, flatConv_has_vjp, hasVJPMat_to_hasVJP,
    rowwise_has_vjp_mat, hasVJP3_to_hasVJP, conv2d_has_vjp3]
  rfl

/-- **Batched STRIDE-2 conv input-VJP faithfulness.** The stride-2 analogue of
    `convBackBatched_faithful`: `convStridedBackBatched` denotes the proven VJP of
    the batched strided conv `batchMap N (flatConvStride2 W b)` — i.e. the
    per-example strided-conv input-grad (`flatConvStride2_has_vjp` = zero-upsample
    the cotangent then the reversed-kernel conv) applied independently across the
    batch. Strided conv (`decimate ∘ conv`) is linear, so its backward ignores the
    forward activation; the batched backward is a plain `batchMap` of the
    per-example backward, matching `batchMap_has_vjp`. The downsample basic-block's
    stride-2 conv1 backward brick. -/
theorem convStridedBackBatched_faithful {N ic oc h w kH kW : Nat} (wN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    den (SHlo.convStridedBackBatched (N := N) wN W b e)
      = (batchMap_has_vjp (flatConvStride2 W b) (flatConvStride2_has_vjp W b)
          (flatConvStride2_differentiable W b)).backward v (den e) := by
  funext idx
  -- The transport in `batchMap_has_vjp` unfolds to the per-example strided-conv
  -- backward on each row; both sides then differ only in the (discarded) forward
  -- activation arg, since strided conv is linear (its backward ignores it).
  simp only [den, batchMap, batchMap_has_vjp, hasVJPMat_to_hasVJP, rowwise_has_vjp_mat]
  rfl

/-- **Batched depthwise input-VJP faithfulness.** The depthwise analogue of
    `convBackBatched_faithful`: `depthwiseBackBatched` denotes the proven VJP of
    the batched depthwise `batchMap N (depthwiseFlat W b)`. Depthwise conv is
    linear, so its backward is activation-independent and the batched backward is
    a plain `batchMap` of the per-example backward. The MBConv depthwise stage's
    batch-separable backward brick. -/
theorem depthwiseBackBatched_faithful {N c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (v : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) :
    den (SHlo.depthwiseBackBatched (N := N) wN W b e)
      = (batchMap_has_vjp (depthwiseFlat W b) (depthwiseFlat_has_vjp W b)
          (depthwiseFlat_differentiable W b)).backward v (den e) := by
  funext idx
  simp only [den, batchMap, batchMap_has_vjp, depthwiseFlat_has_vjp, hasVJPMat_to_hasVJP,
    rowwise_has_vjp_mat, hasVJP3_to_hasVJP, depthwise_has_vjp3]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § bn-layout wrapper: true-batch-norm backward on the NETWORK layout
-- ════════════════════════════════════════════════════════════════

/-- **`bnBatchLA` backward = reindex-conjugated `bnBatchTensor4` backward.**
    The network indexes at `N·(oc·h·w)` (left-assoc) but the proven true-BN
    `bnBatchTensor4` lives at `N·(oc·(h·w))`; `bnBatchLA` bridges by conjugating
    with the associativity-cast reindexes (`bnBatchLA_eq_comp`). Its VJP backward
    is therefore: scatter the cotangent into `[N,C,(H·W)]`, run the renderable
    three-term `bnBatchTensor4_grad_input` at the reindexed activation, scatter
    back. This is what a network-layout `bnBatchLABack` op denotes. -/
theorem bnBatchLA_back_conj {N oc h w : Nat} (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (x dy : Vec (N * (oc * h * w))) :
    (reindex_has_vjp (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm)).backward x
      (bnBatchTensor4_grad_input N oc h w ε γ
        (reindexCLM (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm) x)
        ((reindex_has_vjp (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)))).backward
          (reindexCLM (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm) x) dy))
      = (bnBatchLA_has_vjp N oc h w ε hε γ β).backward x dy := by
  have hb : bnBatchTensor4_grad_input N oc h w ε γ
        (reindexCLM (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm) x)
        ((reindex_has_vjp (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)))).backward
          (reindexCLM (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm) x) dy)
      = (bnBatchTensor4_has_vjp N oc h w ε hε γ β).backward
        (reindexCLM (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm) x)
        ((reindex_has_vjp (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)))).backward
          (reindexCLM (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm) x) dy) := by
    funext i
    rw [bnBatchTensor4_grad_input_correct N oc h w ε hε γ β,
        ← bnBatchTensor4_has_vjp_correct N oc h w ε hε γ β]
  rw [hb]
  simp only [bnBatchLA_has_vjp, vjp_comp, eq_mpr_eq_cast]
  rfl

/-- **`bnBatchLABack` (network-layout true batch-norm backward) faithfulness.**
    The `den` (inline scatter-conjugated `bnBatchTensor4_grad_input`) equals the
    proven `bnBatchLA_has_vjp` backward — the bn backward at the network's
    `N·(oc·h·w)` index, which is what renderBody's `bnBatch` emits. This is the
    layout wrapper that lets `bnBatchBack` compose with `convBackBatched` /
    `depthwiseBackBatched` (all on the left-assoc index) into batched stages. -/
theorem bnBatchLABack_faithful {N oc h w : Nat} (gN xN es : String)
    (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (x : Vec (N * (oc * h * w))) (e : SHlo (N * (oc * h * w))) :
    den (SHlo.bnBatchLABack gN xN es ε γ x e)
      = (bnBatchLA_has_vjp N oc h w ε hε γ β).backward x (den e) :=
  bnBatchLA_back_conj ε γ β hε x (den e)

/-- **`seBackBatched` (batched squeeze-excite backward) faithfulness.** The `den`
    (rowwise application of the proven per-example `seBlockFull` VJP) equals the
    proven batched `seB_has_vjp` backward. SE is non-linear, so — unlike the
    linear `convBackBatched`/`depthwiseBackBatched` — the backward threads each
    example's forward activation `v`; the rowwise `batchMap_has_vjp` structure
    handles that. The fourth (and last) MBConv stage's batch-separable backward. -/
theorem seBackBatched_faithful {N c h w r : Nat} (w1N b1N w2N b2N : String)
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    (v : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) :
    den (SHlo.seBackBatched (N := N) w1N b1N w2N b2N W₁ b₁ W₂ b₂ v e)
      = (seB_has_vjp N (h := h) (w := w) W₁ b₁ W₂ b₂).backward v (den e) := by
  funext idx
  simp only [den, seB_has_vjp, batchMap_has_vjp, hasVJPMat_to_hasVJP, rowwise_has_vjp_mat]

-- ════════════════════════════════════════════════════════════════
-- § Batched MBConv stage backward graphs (the bn wrapper unblocks these)
-- ════════════════════════════════════════════════════════════════

/-- Batched **conv → bn → swish** stage backward graph (MBConv expand), at the
    network layout: `convBackBatched ∘ bnBatchLABack ∘ swishBack`, each at its
    cumulative forward activation. -/
noncomputable def cbsBackBatchedGraph {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) : SHlo (N * (ic * h * w)) :=
  .convBackBatched (N := N) "%cbsW" W b
    (.bnBatchLABack "%cbsG" "%cbsX" "cbsE" ε γ (batchMap N (flatConv W b) x)
      (.swishBack "%cbsSw" (bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x)) e))

theorem cbsBackBatchedGraph_faithful {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) :
    den (cbsBackBatchedGraph W b ε γ β x e)
      = (cbsB_has_vjp N W b ε hε γ β).backward x (den e) := by
  rw [cbsBackBatchedGraph, convBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε), swishBack_faithful]
  simp only [cbsB_has_vjp, bnSwishStage_has_vjp, vjp_comp, Function.comp_apply]

/-- Batched **depthwise → bn → swish** stage backward graph (MBConv depthwise). -/
noncomputable def dwbsBackBatchedGraph {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  .depthwiseBackBatched (N := N) "%dwsW" W b
    (.bnBatchLABack "%dwsG" "%dwsX" "dwsE" ε γ (batchMap N (depthwiseFlat W b) x)
      (.swishBack "%dwsSw" (bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x)) e))

theorem dwbsBackBatchedGraph_faithful {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) :
    den (dwbsBackBatchedGraph W b ε γ β x e)
      = (dwbsB_has_vjp N W b ε hε γ β).backward x (den e) := by
  rw [dwbsBackBatchedGraph, depthwiseBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε), swishBack_faithful]
  simp only [dwbsB_has_vjp, bnSwishStage_has_vjp, vjp_comp, Function.comp_apply]

/-- Batched **conv → bn** stage backward graph (MBConv project, no swish). -/
noncomputable def projBackBatchedGraph {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) : SHlo (N * (ic * h * w)) :=
  .convBackBatched (N := N) "%pbW" W b
    (.bnBatchLABack "%pbG" "%pbX" "pbE" ε γ (batchMap N (flatConv W b) x) e)

theorem projBackBatchedGraph_faithful {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) :
    den (projBackBatchedGraph W b ε γ β x e)
      = (projB_has_vjp N W b ε hε γ β).backward x (den e) := by
  rw [projBackBatchedGraph, convBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε)]
  simp only [projB_has_vjp, bnStage_has_vjp, vjp_comp]

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
      (.seBackBatched (N := N) "%seW1" "%seb1" "%seW2" "%seb2" Wz₁ bz₁ Wz₂ bz₂ xD
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
  simp only [mbBodyB_has_vjp, vjp_comp, Function.comp_apply]

/-- The whole batched MBConv residual block backward graph (body + identity skip). -/
noncomputable def mbResidBlockBackBatchedGraph {N c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (γp βp : Vec c)
    (x dy : Vec (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  residualBackGraph
    (mbBodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp
      x (.operand "%dy" dy)) dy

/-- **CAPSTONE — the whole batched EfficientNet MBConv residual block: backward
    graph ↔ the proven `mbResidFwdB_has_vjp`.** The four batched stage backward
    graphs (`cbsB`/`dwbsB`/`seB`/`projB`) chained at their forward activations +
    the identity skip, proven equal to the repo's batched MBConv block VJP. The
    batched analogue of the per-example `mbconvResidual_backGraph_faithful`. -/
theorem mbResidBlockBackBatchedGraph_faithful {N c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c)
    (x dy : Vec (N * (c * h * w))) :
    den (mbResidBlockBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp x dy)
      = (mbResidFwdB_has_vjp N We be εe hεe γe βe Wd bd εd hεd γd βd
          Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp).backward x dy :=
  residualBackGraph_faithful
    (projB N (h := h) (w := w) Wp bp εp γp βp ∘ seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ ∘
      dwbsB N (h := h) (w := w) Wd bd εd γd βd ∘ cbsB N (h := h) (w := w) We be εe γe βe)
    ((projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp).comp
      ((seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂).comp
        ((dwbsB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd).comp
          (cbsB_differentiable N (h := h) (w := w) We be εe hεe γe βe))))
    (mbBodyB_has_vjp N We be εe hεe γe βe Wd bd εd hεd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp)
    x dy
    (mbBodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp
      x (.operand "%dy" dy))
    (mbBodyBackBatchedGraph_faithful We be εe hεe γe βe Wd bd εd hεd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp x (.operand "%dy" dy))

end Proofs.StableHLO
