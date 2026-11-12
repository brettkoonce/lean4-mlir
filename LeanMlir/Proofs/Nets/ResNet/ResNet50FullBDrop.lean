module

public import LeanMlir.Proofs.Nets.ResNet.ResNet50FullBVJP
public import LeanMlir.Proofs.Foundation.DropLayers
public import LeanMlir.Proofs.Codegen.StableHLO.Pretty
import LeanMlir.Proofs.Foundation.Bf16Erasure
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # ResNet-50 with stochastic depth — the forward, block VJPs and graph at optional drop sites

`ResNet50RenderB`'s `drop` renders (`resnet50in_emalambacc4x128wxclipdropbce` and its
data-parallel, weight-decay and bf16 twins) scale each bottleneck's residual BRANCH by a
host-drawn per-example mask before the skip add: `relu(dropPath s (F x) + skip x)`, sixteen sites
`%dp0 … %dp15`, one per block in forward order. The projection is never dropped.

This file states that forward with each site an `Option` (`Foundation.DropSites`): `none` emits
no node and is `ResNet50FullB`'s forward by `rfl`, `some s` is the `dropPathB` node at the mask
`s`.

* `r50IdBD` / `r50ProjBD` / `r50DownBD` — the three block forwards at a site; `_none` is `rfl`,
  and `_ones` says the all-ones mask the driver passes at eval is the drop-free block.
* `r50IdDropLayer` / `r50ProjDropLayer` / `r50DownDropLayer` — each as a `CertLayer`: the body,
  then `dropPathOptLayer`, inside the residual, then the outer relu. The VJP
  (`r50IdBDHasVJPAt`, …) and the backward graph's faithfulness are the layer's fields.
* `R50IdSmoothAtD` / `R50ProjSmoothAtD` / `R50DownSmoothAtD` — the block's three relu clauses.
  The two interior ones are the drop-free block's; the outer one is at the DROPPED pre-relu sum.
* `r50PreD1 … r50PreD16` — the activation entering each later block, with the sites of the
  blocks before it; the stem is `r50Pre0`, which has no site. `r50Sd sd k` is block `k`'s site.
* `resnet50ForwardBFullDrop` — the whole net; `resnet50ForwardBFullDrop_none` and `_ones` are
  the drop-free forward.
* `resnet50FwdGraphBFullDrop` and `_faithful` — the typed forward graph at the render's tokens,
  `dropPathB` at `dpName k` exactly where the site is `some`. `FwdGraphTextTies` checks each
  block form against the render. The projection forms name their branch and skip
  (`r50{Proj,Down}BodyGraphB`, `r50{Proj,Down}SkipGraphB`) because the render writes the
  projection between the branch and its `dropPathB`, so those two are tied piece by piece in the
  render's order.

The masks are binders. How the driver draws them is not modelled here.

The step ties over this forward are `ResNet50StepTieB` (one replica), `ResNet50SyncStepTieB`
(sync-BN data parallel) and `ResNet50ParamGrad` (each node is the loss's derivative). The
data-parallel forward at the masks is `ResNet50SyncBDrop`.

## References

- Huang, Sun, Liu, Sedra, Weinberger 2016, *Deep Networks with Stochastic Depth*. <https://arxiv.org/abs/1603.09382>
- Wightman, Touvron, Jégou 2021, *ResNet strikes back* (RSB A1/A2: stochastic depth 0.05). <https://arxiv.org/abs/2110.00476>
-/

namespace Proofs

open scoped BigOperators
open StableHLO

/-- Block `k`'s stochastic-depth site out of the net's sixteen. -/
def r50Sd {N : Nat} (sd : Option (Fin 16 → Vec N)) (k : Fin 16) : Option (Vec N) :=
  sd.map fun m => m k

@[simp] theorem r50Sd_none {N : Nat} (k : Fin 16) :
    r50Sd (none : Option (Fin 16 → Vec N)) k = none := rfl

@[simp] theorem r50Sd_some {N : Nat} (m : Fin 16 → Vec N) (k : Fin 16) :
    r50Sd (some m) k = some (m k) := rfl

-- ════════════════════════════════════════════════════════════════
-- § Block forwards at a site
-- ════════════════════════════════════════════════════════════════

/-- Identity bottleneck with a stochastic-depth site on the residual branch:
    `relu(dropPath s (F x) + x)`. -/
@[reducible] noncomputable def r50IdBD (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc)
    (s : Option (Vec N)) : Vec (N * (oc * h * w)) → Vec (N * (oc * h * w)) :=
  relu (N * (oc * h * w)) ∘ residual
    (dropPathOpt N (oc * h * w) s ∘
      projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
      cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁)

/-- Stride-1 projection bottleneck with a site on the residual branch. The projection is not
    dropped. -/
@[reducible] noncomputable def r50ProjBD (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (s : Option (Vec N)) : Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  relu (N * (oc * h * w)) ∘ residualProj
    (projB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp)
    (dropPathOpt N (oc * h * w) s ∘
      projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
      cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁)

/-- Strided projection bottleneck with a site on the residual branch. The projection is not
    dropped. -/
@[reducible] noncomputable def r50DownBD (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (s : Option (Vec N)) : Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  relu (N * (oc * h * w)) ∘ residualProj
    (projStridedB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp)
    (dropPathOpt N (oc * h * w) s ∘
      projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
      cbReluStridedB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      cbReluB N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁)

theorem r50IdBD_none (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc) :
    r50IdBD N h w p none = r50IdB N h w p := rfl

theorem r50ProjBD_none (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc) :
    r50ProjBD N h w p none = r50ProjB N h w p := rfl

theorem r50DownBD_none (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc) :
    r50DownBD N h w p none = r50DownB N h w p := rfl

/-- At the all-ones mask — what the driver passes at eval — the block is the drop-free one. -/
theorem r50IdBD_ones (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc) :
    r50IdBD N h w p (some fun _ => 1) = r50IdB N h w p := by
  simp only [r50IdBD, dropPathOpt_ones]; rfl

theorem r50ProjBD_ones (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc) :
    r50ProjBD N h w p (some fun _ => 1) = r50ProjB N h w p := by
  simp only [r50ProjBD, dropPathOpt_ones]; rfl

theorem r50DownBD_ones (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc) :
    r50DownBD N h w p (some fun _ => 1) = r50DownB N h w p := by
  simp only [r50DownBD, dropPathOpt_ones]; rfl

-- ════════════════════════════════════════════════════════════════
-- § The relu clauses at a site
-- ════════════════════════════════════════════════════════════════

/-- The three relu sites of an identity bottleneck with a drop site. The interior two are the
    drop-free block's (the site is after them); the outer one is at the dropped pre-relu sum. -/
structure R50IdSmoothAtD (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc) (s : Option (Vec N))
    (v : Vec (N * (oc * h * w))) : Prop where
  hm1 : ∀ k, bnBatchLA N mid h w p.ε₁ p.γ₁ p.β₁ (batchMap N (flatConv p.W₁ p.b₁) v) k ≠ 0
  hm2 : ∀ k, bnBatchLA N mid h w p.ε₂ p.γ₂ p.β₂
    (batchMap N (flatConv p.W₂ p.b₂)
      (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v)) k ≠ 0
  hout : ∀ k, residual (dropPathOpt N (oc * h * w) s ∘
    projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
    cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
    cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) v k ≠ 0

/-- The three relu sites of the stride-1 projection bottleneck with a drop site. -/
structure R50ProjSmoothAtD (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (s : Option (Vec N)) (v : Vec (N * (ic * h * w))) : Prop where
  hm1 : ∀ k, bnBatchLA N mid h w p.ε₁ p.γ₁ p.β₁ (batchMap N (flatConv p.W₁ p.b₁) v) k ≠ 0
  hm2 : ∀ k, bnBatchLA N mid h w p.ε₂ p.γ₂ p.β₂
    (batchMap N (flatConv p.W₂ p.b₂)
      (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v)) k ≠ 0
  hout : ∀ k, residualProj (projB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp)
    (dropPathOpt N (oc * h * w) s ∘
      projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
      cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) v k ≠ 0

/-- The three relu sites of the strided projection bottleneck with a drop site. -/
structure R50DownSmoothAtD (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (s : Option (Vec N)) (v : Vec (N * (ic * (2 * h) * (2 * w)))) : Prop where
  hm1 : ∀ k, bnBatchLA N mid (2 * h) (2 * w) p.ε₁ p.γ₁ p.β₁
    (batchMap N (flatConv p.W₁ p.b₁) v) k ≠ 0
  hm2 : ∀ k, bnBatchLA N mid h w p.ε₂ p.γ₂ p.β₂
    (batchMap N (flatConvStride2 p.W₂ p.b₂)
      (cbReluB N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v)) k ≠ 0
  hout : ∀ k, residualProj (projStridedB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp)
    (dropPathOpt N (oc * h * w) s ∘
      projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
      cbReluStridedB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      cbReluB N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) v k ≠ 0

/-- At an absent site the clauses are the drop-free block's. -/
theorem r50IdSmoothAtD_none {N h w mid oc : Nat} {p : R50IdW mid oc} {v : Vec (N * (oc * h * w))}
    (hs : R50IdSmoothAt N h w p v) : R50IdSmoothAtD N h w p none v :=
  ⟨hs.hm1, hs.hm2, hs.hout⟩

theorem r50ProjSmoothAtD_none {N h w ic mid oc : Nat} {p : R50ProjW ic mid oc}
    {v : Vec (N * (ic * h * w))} (hs : R50ProjSmoothAt N h w p v) :
    R50ProjSmoothAtD N h w p none v :=
  ⟨hs.hm1, hs.hm2, hs.hout⟩

theorem r50DownSmoothAtD_none {N h w ic mid oc : Nat} {p : R50ProjW ic mid oc}
    {v : Vec (N * (ic * (2 * h) * (2 * w)))} (hs : R50DownSmoothAt N h w p v) :
    R50DownSmoothAtD N h w p none v :=
  ⟨hs.hm1, hs.hm2, hs.hout⟩

-- ════════════════════════════════════════════════════════════════
-- § The blocks as certified layers, and their VJPs
-- ════════════════════════════════════════════════════════════════

/-- The identity bottleneck with a drop site as a `CertLayer`: the body, then the site, inside the
    identity residual, then the outer relu. -/
noncomputable def r50IdDropLayer (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc)
    (hq : R50IdPos p) (s : Option (Vec N)) : CertLayer (N * (oc * h * w)) (N * (oc * h * w)) :=
  (CertLayer.residual ((((cbReluLayer N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁).comp
    (cbReluLayer N p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂)).comp
    (projLayer N p.W₃ p.b₃ p.ε₃ hq.h3 p.γ₃ p.β₃)).comp
    (dropPathOptLayer N (oc * h * w) s))).comp (CertLayer.reluOut _)

/-- The stride-1 projection bottleneck with a drop site as a `CertLayer`. -/
noncomputable def r50ProjDropLayer (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) (s : Option (Vec N)) : CertLayer (N * (ic * h * w)) (N * (oc * h * w)) :=
  (CertLayer.residualProj (projLayer N (h := h) (w := w) p.Wp p.bp p.εp hq.hp p.γp p.βp)
    ((((cbReluLayer N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁).comp
      (cbReluLayer N p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂)).comp
      (projLayer N p.W₃ p.b₃ p.ε₃ hq.h3 p.γ₃ p.β₃)).comp
      (dropPathOptLayer N (oc * h * w) s))).comp (CertLayer.reluOut _)

/-- The strided projection bottleneck with a drop site as a `CertLayer`. -/
noncomputable def r50DownDropLayer (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) (s : Option (Vec N)) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) :=
  (CertLayer.residualProj (projStridedLayer N (h := h) (w := w) p.Wp p.bp p.εp hq.hp p.γp p.βp)
    ((((cbReluLayer N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁).comp
      (cbReluStridedLayer N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂)).comp
      (projLayer N p.W₃ p.b₃ p.ε₃ hq.h3 p.γ₃ p.β₃)).comp
      (dropPathOptLayer N (oc * h * w) s))).comp (CertLayer.reluOut _)

theorem r50IdDropLayer_fwd (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc) (hq : R50IdPos p)
    (s : Option (Vec N)) : (r50IdDropLayer N h w p hq s).fwd = r50IdBD N h w p s := rfl

theorem r50ProjDropLayer_fwd (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) (s : Option (Vec N)) :
    (r50ProjDropLayer N h w p hq s).fwd = r50ProjBD N h w p s := rfl

theorem r50DownDropLayer_fwd (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) (s : Option (Vec N)) :
    (r50DownDropLayer N h w p hq s).fwd = r50DownBD N h w p s := rfl

/-- The block's relu clauses are the layer's certification domain. -/
theorem r50IdDropLayer_ok {N h w mid oc : Nat} {p : R50IdW mid oc} (hq : R50IdPos p)
    {s : Option (Vec N)} {v : Vec (N * (oc * h * w))} (hs : R50IdSmoothAtD N h w p s v) :
    (r50IdDropLayer N h w p hq s).ok v :=
  ⟨⟨⟨⟨hs.hm1, hs.hm2⟩, trivial⟩, trivial⟩, hs.hout⟩

theorem r50ProjDropLayer_ok {N h w ic mid oc : Nat} {p : R50ProjW ic mid oc} (hq : R50ProjPos p)
    {s : Option (Vec N)} {v : Vec (N * (ic * h * w))} (hs : R50ProjSmoothAtD N h w p s v) :
    (r50ProjDropLayer N h w p hq s).ok v :=
  ⟨⟨trivial, ⟨⟨hs.hm1, hs.hm2⟩, trivial⟩, trivial⟩, hs.hout⟩

theorem r50DownDropLayer_ok {N h w ic mid oc : Nat} {p : R50ProjW ic mid oc} (hq : R50ProjPos p)
    {s : Option (Vec N)} {v : Vec (N * (ic * (2 * h) * (2 * w)))}
    (hs : R50DownSmoothAtD N h w p s v) : (r50DownDropLayer N h w p hq s).ok v :=
  ⟨⟨trivial, ⟨⟨hs.hm1, hs.hm2⟩, trivial⟩, trivial⟩, hs.hout⟩

/-- Identity bottleneck VJP at a drop site: `r50IdDropLayer`'s. -/
noncomputable def r50IdBDHasVJPAt (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc)
    (hq : R50IdPos p) (s : Option (Vec N)) (v : Vec (N * (oc * h * w)))
    (hs : R50IdSmoothAtD N h w p s v) : HasVJPAt (r50IdBD N h w p s) v :=
  (r50IdDropLayer N h w p hq s).vjp v (r50IdDropLayer_ok hq hs)

/-- Stride-1 projection bottleneck VJP at a drop site. -/
noncomputable def r50ProjBDHasVJPAt (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) (s : Option (Vec N)) (v : Vec (N * (ic * h * w)))
    (hs : R50ProjSmoothAtD N h w p s v) : HasVJPAt (r50ProjBD N h w p s) v :=
  (r50ProjDropLayer N h w p hq s).vjp v (r50ProjDropLayer_ok hq hs)

/-- Strided projection bottleneck VJP at a drop site. -/
noncomputable def r50DownBDHasVJPAt (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) (s : Option (Vec N)) (v : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : R50DownSmoothAtD N h w p s v) : HasVJPAt (r50DownBD N h w p s) v :=
  (r50DownDropLayer N h w p hq s).vjp v (r50DownDropLayer_ok hq hs)

-- ════════════════════════════════════════════════════════════════
-- § The running activations with the sites — `r50PreD{k}` enters block `k`
--   The stem has no site, so block 0 reads `r50Pre0`.
-- ════════════════════════════════════════════════════════════════

noncomputable def r50PreD1 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) :=
  r50ProjBD N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0 (r50Sd sd 0) ∘ r50Pre0 N q w
noncomputable def r50PreD2 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) :=
  r50IdBD N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1 (r50Sd sd 1) ∘ r50PreD1 N q w sd
noncomputable def r50PreD3 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) :=
  r50IdBD N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2 (r50Sd sd 2) ∘ r50PreD2 N q w sd
noncomputable def r50PreD4 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) :=
  r50DownBD N (2 * (2 * q)) (2 * (2 * q)) w.s2b0 (r50Sd sd 3) ∘ r50PreD3 N q w sd
noncomputable def r50PreD5 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) :=
  r50IdBD N (2 * (2 * q)) (2 * (2 * q)) w.s2b1 (r50Sd sd 4) ∘ r50PreD4 N q w sd
noncomputable def r50PreD6 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) :=
  r50IdBD N (2 * (2 * q)) (2 * (2 * q)) w.s2b2 (r50Sd sd 5) ∘ r50PreD5 N q w sd
noncomputable def r50PreD7 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) :=
  r50IdBD N (2 * (2 * q)) (2 * (2 * q)) w.s2b3 (r50Sd sd 6) ∘ r50PreD6 N q w sd
noncomputable def r50PreD8 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (1024 * (2 * q) * (2 * q))) :=
  r50DownBD N (2 * q) (2 * q) w.s3b0 (r50Sd sd 7) ∘ r50PreD7 N q w sd
noncomputable def r50PreD9 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (1024 * (2 * q) * (2 * q))) :=
  r50IdBD N (2 * q) (2 * q) w.s3b1 (r50Sd sd 8) ∘ r50PreD8 N q w sd
noncomputable def r50PreD10 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (1024 * (2 * q) * (2 * q))) :=
  r50IdBD N (2 * q) (2 * q) w.s3b2 (r50Sd sd 9) ∘ r50PreD9 N q w sd
noncomputable def r50PreD11 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (1024 * (2 * q) * (2 * q))) :=
  r50IdBD N (2 * q) (2 * q) w.s3b3 (r50Sd sd 10) ∘ r50PreD10 N q w sd
noncomputable def r50PreD12 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (1024 * (2 * q) * (2 * q))) :=
  r50IdBD N (2 * q) (2 * q) w.s3b4 (r50Sd sd 11) ∘ r50PreD11 N q w sd
noncomputable def r50PreD13 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (1024 * (2 * q) * (2 * q))) :=
  r50IdBD N (2 * q) (2 * q) w.s3b5 (r50Sd sd 12) ∘ r50PreD12 N q w sd
noncomputable def r50PreD14 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (2048 * q * q)) :=
  r50DownBD N q q w.s4b0 (r50Sd sd 13) ∘ r50PreD13 N q w sd
noncomputable def r50PreD15 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (2048 * q * q)) :=
  r50IdBD N q q w.s4b1 (r50Sd sd 14) ∘ r50PreD14 N q w sd
noncomputable def r50PreD16 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (2048 * q * q)) :=
  r50IdBD N q q w.s4b2 (r50Sd sd 15) ∘ r50PreD15 N q w sd
theorem r50PreD1_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD1 N q w sd x = r50ProjBD N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0 (r50Sd sd 0) (r50Pre0 N q w x) := by
  rw [r50PreD1, Function.comp_apply]
theorem r50PreD2_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD2 N q w sd x = r50IdBD N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1 (r50Sd sd 1) (r50PreD1 N q w sd x) := by
  rw [r50PreD2, Function.comp_apply]
theorem r50PreD3_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD3 N q w sd x = r50IdBD N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2 (r50Sd sd 2) (r50PreD2 N q w sd x) := by
  rw [r50PreD3, Function.comp_apply]
theorem r50PreD4_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD4 N q w sd x = r50DownBD N (2 * (2 * q)) (2 * (2 * q)) w.s2b0 (r50Sd sd 3) (r50PreD3 N q w sd x) := by
  rw [r50PreD4, Function.comp_apply]
theorem r50PreD5_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD5 N q w sd x = r50IdBD N (2 * (2 * q)) (2 * (2 * q)) w.s2b1 (r50Sd sd 4) (r50PreD4 N q w sd x) := by
  rw [r50PreD5, Function.comp_apply]
theorem r50PreD6_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD6 N q w sd x = r50IdBD N (2 * (2 * q)) (2 * (2 * q)) w.s2b2 (r50Sd sd 5) (r50PreD5 N q w sd x) := by
  rw [r50PreD6, Function.comp_apply]
theorem r50PreD7_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD7 N q w sd x = r50IdBD N (2 * (2 * q)) (2 * (2 * q)) w.s2b3 (r50Sd sd 6) (r50PreD6 N q w sd x) := by
  rw [r50PreD7, Function.comp_apply]
theorem r50PreD8_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD8 N q w sd x = r50DownBD N (2 * q) (2 * q) w.s3b0 (r50Sd sd 7) (r50PreD7 N q w sd x) := by
  rw [r50PreD8, Function.comp_apply]
theorem r50PreD9_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD9 N q w sd x = r50IdBD N (2 * q) (2 * q) w.s3b1 (r50Sd sd 8) (r50PreD8 N q w sd x) := by
  rw [r50PreD9, Function.comp_apply]
theorem r50PreD10_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD10 N q w sd x = r50IdBD N (2 * q) (2 * q) w.s3b2 (r50Sd sd 9) (r50PreD9 N q w sd x) := by
  rw [r50PreD10, Function.comp_apply]
theorem r50PreD11_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD11 N q w sd x = r50IdBD N (2 * q) (2 * q) w.s3b3 (r50Sd sd 10) (r50PreD10 N q w sd x) := by
  rw [r50PreD11, Function.comp_apply]
theorem r50PreD12_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD12 N q w sd x = r50IdBD N (2 * q) (2 * q) w.s3b4 (r50Sd sd 11) (r50PreD11 N q w sd x) := by
  rw [r50PreD12, Function.comp_apply]
theorem r50PreD13_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD13 N q w sd x = r50IdBD N (2 * q) (2 * q) w.s3b5 (r50Sd sd 12) (r50PreD12 N q w sd x) := by
  rw [r50PreD13, Function.comp_apply]
theorem r50PreD14_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD14 N q w sd x = r50DownBD N q q w.s4b0 (r50Sd sd 13) (r50PreD13 N q w sd x) := by
  rw [r50PreD14, Function.comp_apply]
theorem r50PreD15_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD15 N q w sd x = r50IdBD N q q w.s4b1 (r50Sd sd 14) (r50PreD14 N q w sd x) := by
  rw [r50PreD15, Function.comp_apply]
theorem r50PreD16_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50PreD16 N q w sd x = r50IdBD N q q w.s4b2 (r50Sd sd 15) (r50PreD15 N q w sd x) := by
  rw [r50PreD16, Function.comp_apply]

/-! At no sites the prefixes are `ResNet50FullBVJP`'s. -/
theorem r50PreD1_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD1 N q w none = r50Pre1 N q w := by
  rw [r50PreD1, r50Pre1]; rfl
theorem r50PreD2_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD2 N q w none = r50Pre2 N q w := by
  rw [r50PreD2, r50Pre2, r50PreD1_none]; rfl
theorem r50PreD3_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD3 N q w none = r50Pre3 N q w := by
  rw [r50PreD3, r50Pre3, r50PreD2_none]; rfl
theorem r50PreD4_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD4 N q w none = r50Pre4 N q w := by
  rw [r50PreD4, r50Pre4, r50PreD3_none]; rfl
theorem r50PreD5_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD5 N q w none = r50Pre5 N q w := by
  rw [r50PreD5, r50Pre5, r50PreD4_none]; rfl
theorem r50PreD6_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD6 N q w none = r50Pre6 N q w := by
  rw [r50PreD6, r50Pre6, r50PreD5_none]; rfl
theorem r50PreD7_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD7 N q w none = r50Pre7 N q w := by
  rw [r50PreD7, r50Pre7, r50PreD6_none]; rfl
theorem r50PreD8_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD8 N q w none = r50Pre8 N q w := by
  rw [r50PreD8, r50Pre8, r50PreD7_none]; rfl
theorem r50PreD9_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD9 N q w none = r50Pre9 N q w := by
  rw [r50PreD9, r50Pre9, r50PreD8_none]; rfl
theorem r50PreD10_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD10 N q w none = r50Pre10 N q w := by
  rw [r50PreD10, r50Pre10, r50PreD9_none]; rfl
theorem r50PreD11_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD11 N q w none = r50Pre11 N q w := by
  rw [r50PreD11, r50Pre11, r50PreD10_none]; rfl
theorem r50PreD12_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD12 N q w none = r50Pre12 N q w := by
  rw [r50PreD12, r50Pre12, r50PreD11_none]; rfl
theorem r50PreD13_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD13 N q w none = r50Pre13 N q w := by
  rw [r50PreD13, r50Pre13, r50PreD12_none]; rfl
theorem r50PreD14_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD14 N q w none = r50Pre14 N q w := by
  rw [r50PreD14, r50Pre14, r50PreD13_none]; rfl
theorem r50PreD15_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD15 N q w none = r50Pre15 N q w := by
  rw [r50PreD15, r50Pre15, r50PreD14_none]; rfl
theorem r50PreD16_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD16 N q w none = r50Pre16 N q w := by
  rw [r50PreD16, r50Pre16, r50PreD15_none]; rfl

/-! At the all-ones masks the prefixes are `ResNet50FullBVJP`'s. -/

theorem r50PreD1_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD1 N q w (some fun _ _ => 1) = r50Pre1 N q w := by
  rw [r50PreD1, r50Pre1]
  exact congrArg (· ∘ r50Pre0 N q w) (r50ProjBD_ones N _ _ w.s1b0)

theorem r50PreD2_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD2 N q w (some fun _ _ => 1) = r50Pre2 N q w := by
  rw [r50PreD2, r50Pre2, r50PreD1_ones]
  exact congrArg (· ∘ r50Pre1 N q w) (r50IdBD_ones N _ _ w.s1b1)

theorem r50PreD3_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD3 N q w (some fun _ _ => 1) = r50Pre3 N q w := by
  rw [r50PreD3, r50Pre3, r50PreD2_ones]
  exact congrArg (· ∘ r50Pre2 N q w) (r50IdBD_ones N _ _ w.s1b2)

theorem r50PreD4_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD4 N q w (some fun _ _ => 1) = r50Pre4 N q w := by
  rw [r50PreD4, r50Pre4, r50PreD3_ones]
  exact congrArg (· ∘ r50Pre3 N q w) (r50DownBD_ones N _ _ w.s2b0)

theorem r50PreD5_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD5 N q w (some fun _ _ => 1) = r50Pre5 N q w := by
  rw [r50PreD5, r50Pre5, r50PreD4_ones]
  exact congrArg (· ∘ r50Pre4 N q w) (r50IdBD_ones N _ _ w.s2b1)

theorem r50PreD6_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD6 N q w (some fun _ _ => 1) = r50Pre6 N q w := by
  rw [r50PreD6, r50Pre6, r50PreD5_ones]
  exact congrArg (· ∘ r50Pre5 N q w) (r50IdBD_ones N _ _ w.s2b2)

theorem r50PreD7_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD7 N q w (some fun _ _ => 1) = r50Pre7 N q w := by
  rw [r50PreD7, r50Pre7, r50PreD6_ones]
  exact congrArg (· ∘ r50Pre6 N q w) (r50IdBD_ones N _ _ w.s2b3)

theorem r50PreD8_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD8 N q w (some fun _ _ => 1) = r50Pre8 N q w := by
  rw [r50PreD8, r50Pre8, r50PreD7_ones]
  exact congrArg (· ∘ r50Pre7 N q w) (r50DownBD_ones N _ _ w.s3b0)

theorem r50PreD9_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD9 N q w (some fun _ _ => 1) = r50Pre9 N q w := by
  rw [r50PreD9, r50Pre9, r50PreD8_ones]
  exact congrArg (· ∘ r50Pre8 N q w) (r50IdBD_ones N _ _ w.s3b1)

theorem r50PreD10_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD10 N q w (some fun _ _ => 1) = r50Pre10 N q w := by
  rw [r50PreD10, r50Pre10, r50PreD9_ones]
  exact congrArg (· ∘ r50Pre9 N q w) (r50IdBD_ones N _ _ w.s3b2)

theorem r50PreD11_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD11 N q w (some fun _ _ => 1) = r50Pre11 N q w := by
  rw [r50PreD11, r50Pre11, r50PreD10_ones]
  exact congrArg (· ∘ r50Pre10 N q w) (r50IdBD_ones N _ _ w.s3b3)

theorem r50PreD12_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD12 N q w (some fun _ _ => 1) = r50Pre12 N q w := by
  rw [r50PreD12, r50Pre12, r50PreD11_ones]
  exact congrArg (· ∘ r50Pre11 N q w) (r50IdBD_ones N _ _ w.s3b4)

theorem r50PreD13_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD13 N q w (some fun _ _ => 1) = r50Pre13 N q w := by
  rw [r50PreD13, r50Pre13, r50PreD12_ones]
  exact congrArg (· ∘ r50Pre12 N q w) (r50IdBD_ones N _ _ w.s3b5)

theorem r50PreD14_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD14 N q w (some fun _ _ => 1) = r50Pre14 N q w := by
  rw [r50PreD14, r50Pre14, r50PreD13_ones]
  exact congrArg (· ∘ r50Pre13 N q w) (r50DownBD_ones N _ _ w.s4b0)

theorem r50PreD15_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD15 N q w (some fun _ _ => 1) = r50Pre15 N q w := by
  rw [r50PreD15, r50Pre15, r50PreD14_ones]
  exact congrArg (· ∘ r50Pre14 N q w) (r50IdBD_ones N _ _ w.s4b1)

theorem r50PreD16_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    r50PreD16 N q w (some fun _ _ => 1) = r50Pre16 N q w := by
  rw [r50PreD16, r50Pre16, r50PreD15_ones]
  exact congrArg (· ∘ r50Pre15 N q w) (r50IdBD_ones N _ _ w.s4b2)

-- ════════════════════════════════════════════════════════════════
-- § The whole net
-- ════════════════════════════════════════════════════════════════

/-- **The batch-BN ResNet-50 forward with stochastic depth**: sixteen optional sites, one per
    bottleneck, each on the residual branch before the skip add. `sd := none` is
    `resnet50ForwardBFull` (`resnet50ForwardBFullDrop_none`); `sd := some m` is the forward the
    `drop` renders emit at the masks `m`. The stem and head carry no site. -/
noncomputable def resnet50ForwardBFullDrop (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (sd : Option (Fin 16 → Vec N)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) : Vec (N * nCls) :=
  r34HeadB N q q w.Wd w.bd (r50PreD16 N q w sd x)

theorem resnet50ForwardBFullDrop_none (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    resnet50ForwardBFullDrop N q w none x = resnet50ForwardBFull N q w x := by
  rw [resnet50ForwardBFullDrop, r50PreD16_none, resnet50ForwardBFull_eq_chain]; rfl
/-- **Eval is the drop-free forward**: at the all-ones masks the driver passes when scoring,
    every site is the identity. -/
theorem resnet50ForwardBFullDrop_ones (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    resnet50ForwardBFullDrop N q w (some fun _ _ => 1) x = resnet50ForwardBFull N q w x := by
  rw [resnet50ForwardBFullDrop, r50PreD16_ones, resnet50ForwardBFull_eq_chain]; rfl

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The typed forward graph with the sites
--   `ResNet50FullB`'s per-block graphs with a `dropPathOptG` on the residual branch, between
--   bn₃ and the `addVB`, where `bnk{Id,Proj,Strided}FwdB` emit `dropPathB`.
-- ════════════════════════════════════════════════════════════════

/-- Identity bottleneck graph with a drop site: `r50IdGraphB` with `dropPathOptG dpN s` on the
    branch operand of the `addVB`. -/
def r50IdGraphBD (p epsStr dpN : String) (N h w : Nat) {mid oc : Nat} (pw : R50IdW mid oc)
    (bf16 : Bool) (s : Option (Vec N)) (e : SHlo (N * (oc * h * w))) : SHlo (N * (oc * h * w)) :=
  .batchOp (N := N) (.relu (n := oc * h * w))
    (.addVB
      (dropPathOptG dpN s
        (.bnBatchF s!"%{p}g3" s!"%{p}bt3" epsStr pw.ε₃ pw.γ₃ pw.β₃
          (.batchOp (N := N)
            (.convAt bf16 id (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃)
            (.batchOp (N := N) (.relu (n := mid * h * w))
              (.bnBatchF s!"%{p}g2" s!"%{p}bt2" epsStr pw.ε₂ pw.γ₂ pw.β₂
                (.batchOp (N := N)
                  (.convAt bf16 id (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂)
                  (.batchOp (N := N) (.relu (n := mid * h * w))
                    (.bnBatchF s!"%{p}g1" s!"%{p}bt1" epsStr pw.ε₁ pw.γ₁ pw.β₁
                      (.batchOp (N := N)
                        (.convAt bf16 id (h := h) (w := w) s!"%{p}W1" (biasName false "" mid)
                          pw.W₁ pw.b₁)
                        e)))))))))
      e)

theorem r50IdGraphBD_faithful (p epsStr dpN : String) (N h w : Nat) {mid oc : Nat}
    (pw : R50IdW mid oc) (bf16 : Bool) (s : Option (Vec N)) (e : SHlo (N * (oc * h * w))) :
    den (r50IdGraphBD p epsStr dpN N h w pw bf16 s e) = r50IdBD N h w pw s (den e) := by
  unfold r50IdGraphBD r50IdBD projB cbReluB residual biPath
  simp only [↓den_batchOp_relu_eq_reluF, reluF_faithful, den_batchOp, Bf16Fold.denOp_convAt_id,
    den_bnBatchF, den_addVB, den_dropPathOptG]
  simp only [denOp, Function.comp_apply]

/-- The residual branch of the stride-1 projection bottleneck up to bn₃, as its own graph: what
    the drop site reads. Named so `FwdGraphTextTies` can tie the render's emission order, which
    writes the projection between this branch and its `dropPathB`. -/
def r50ProjBodyGraphB (p epsStr : String) (N h w : Nat) {ic mid oc : Nat} (pw : R50ProjW ic mid oc)
    (bf16 : Bool) (e : SHlo (N * (ic * h * w))) : SHlo (N * (oc * h * w)) :=
  .bnBatchF s!"%{p}g3" s!"%{p}bt3" epsStr pw.ε₃ pw.γ₃ pw.β₃
    (.batchOp (N := N)
      (.convAt bf16 id (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃)
      (.batchOp (N := N) (.relu (n := mid * h * w))
        (.bnBatchF s!"%{p}g2" s!"%{p}bt2" epsStr pw.ε₂ pw.γ₂ pw.β₂
          (.batchOp (N := N)
            (.convAt bf16 id (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂)
            (.batchOp (N := N) (.relu (n := mid * h * w))
              (.bnBatchF s!"%{p}g1" s!"%{p}bt1" epsStr pw.ε₁ pw.γ₁ pw.β₁
                (.batchOp (N := N)
                  (.convAt bf16 id (h := h) (w := w) s!"%{p}W1" (biasName false "" mid)
                    pw.W₁ pw.b₁)
                  e)))))))

/-- The stride-1 projection skip, `bn_p(conv_p e)`, as its own graph. Never dropped. -/
def r50ProjSkipGraphB (p epsStr : String) (N h w : Nat) {ic mid oc : Nat} (pw : R50ProjW ic mid oc)
    (bf16 : Bool) (e : SHlo (N * (ic * h * w))) : SHlo (N * (oc * h * w)) :=
  .bnBatchF s!"%{p}gp" s!"%{p}btp" epsStr pw.εp pw.γp pw.βp
    (.batchOp (N := N)
      (.convAt bf16 id (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp) e)

/-- Stride-1 projection bottleneck graph with a drop site on the branch:
    `relu(addVB(dropPathOptG dpN s (body e), skip e))`. The projection operand is undropped. -/
def r50ProjGraphBD (p epsStr dpN : String) (N h w : Nat) {ic mid oc : Nat} (pw : R50ProjW ic mid oc)
    (bf16 : Bool) (s : Option (Vec N)) (e : SHlo (N * (ic * h * w))) :
    SHlo (N * (oc * h * w)) :=
  .batchOp (N := N) (.relu (n := oc * h * w))
    (.addVB
      (dropPathOptG dpN s (r50ProjBodyGraphB p epsStr N h w pw bf16 e))
      (r50ProjSkipGraphB p epsStr N h w pw bf16 e))

theorem r50ProjGraphBD_faithful (p epsStr dpN : String) (N h w : Nat) {ic mid oc : Nat}
    (pw : R50ProjW ic mid oc) (bf16 : Bool) (s : Option (Vec N))
    (e : SHlo (N * (ic * h * w))) :
    den (r50ProjGraphBD p epsStr dpN N h w pw bf16 s e) = r50ProjBD N h w pw s (den e) := by
  unfold r50ProjGraphBD r50ProjBodyGraphB r50ProjSkipGraphB r50ProjBD projB cbReluB residualProj biPath
  simp only [↓den_batchOp_relu_eq_reluF, reluF_faithful, den_batchOp, Bf16Fold.denOp_convAt_id,
    den_bnBatchF, den_addVB, den_dropPathOptG]
  simp only [denOp, Function.comp_apply]
  congr 1
  funext i
  ring

/-- The residual branch of the strided projection bottleneck up to bn₃, as its own graph: what
    the drop site reads. Named so `FwdGraphTextTies` can tie the render's emission order, which
    writes the projection between this branch and its `dropPathB`. -/
def r50DownBodyGraphB (p epsStr : String) (N h w : Nat) {ic mid oc : Nat} (pw : R50ProjW ic mid oc)
    (bf16 : Bool) (e : SHlo (N * (ic * (2 * h) * (2 * w)))) : SHlo (N * (oc * h * w)) :=
  .bnBatchF s!"%{p}g3" s!"%{p}bt3" epsStr pw.ε₃ pw.γ₃ pw.β₃
    (.batchOp (N := N)
      (.convAt bf16 id (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃)
      (.batchOp (N := N) (.relu (n := mid * h * w))
        (.bnBatchF s!"%{p}g2" s!"%{p}bt2" epsStr pw.ε₂ pw.γ₂ pw.β₂
          (.batchOp (N := N)
            (.convStridedAt bf16 id (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂)
            (.batchOp (N := N) (.relu (n := mid * (2 * h) * (2 * w)))
              (.bnBatchF s!"%{p}g1" s!"%{p}bt1" epsStr pw.ε₁ pw.γ₁ pw.β₁
                (.batchOp (N := N)
                  (.convAt bf16 id (h := 2 * h) (w := 2 * w) s!"%{p}W1" (biasName false "" mid)
                    pw.W₁ pw.b₁)
                  e)))))))

/-- The strided projection skip, `bn_p(conv_p e)`, as its own graph. Never dropped. -/
def r50DownSkipGraphB (p epsStr : String) (N h w : Nat) {ic mid oc : Nat} (pw : R50ProjW ic mid oc)
    (bf16 : Bool) (e : SHlo (N * (ic * (2 * h) * (2 * w)))) : SHlo (N * (oc * h * w)) :=
  .bnBatchF s!"%{p}gp" s!"%{p}btp" epsStr pw.εp pw.γp pw.βp
    (.batchOp (N := N)
      (.convStridedAt bf16 id (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp) e)

/-- Strided projection bottleneck graph with a drop site on the branch:
    `relu(addVB(dropPathOptG dpN s (body e), skip e))`. The projection operand is undropped. -/
def r50DownGraphBD (p epsStr dpN : String) (N h w : Nat) {ic mid oc : Nat} (pw : R50ProjW ic mid oc)
    (bf16 : Bool) (s : Option (Vec N)) (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    SHlo (N * (oc * h * w)) :=
  .batchOp (N := N) (.relu (n := oc * h * w))
    (.addVB
      (dropPathOptG dpN s (r50DownBodyGraphB p epsStr N h w pw bf16 e))
      (r50DownSkipGraphB p epsStr N h w pw bf16 e))

theorem r50DownGraphBD_faithful (p epsStr dpN : String) (N h w : Nat) {ic mid oc : Nat}
    (pw : R50ProjW ic mid oc) (bf16 : Bool) (s : Option (Vec N))
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (r50DownGraphBD p epsStr dpN N h w pw bf16 s e) = r50DownBD N h w pw s (den e) := by
  unfold r50DownGraphBD r50DownBodyGraphB r50DownSkipGraphB r50DownBD projB projStridedB cbReluB cbReluStridedB residualProj biPath
  simp only [↓den_batchOp_relu_eq_reluF, reluF_faithful, den_batchOp, Bf16Fold.denOp_convAt_id,
    Bf16Fold.denOp_convStridedAt_id,
    den_bnBatchF, den_addVB, den_dropPathOptG]
  simp only [denOp, Function.comp_apply]
  congr 1
  funext i
  ring

/-- **The batch-BN ResNet-50 forward graph with stochastic depth**: `resnet50FwdGraphBFull` with
    block `k`'s branch through `dropPathOptG (dpName k)` at its site, the mask operand the render
    names `%dp{k}`. At `sd := none` no site emits a node. -/
def resnet50FwdGraphBFullDrop (N q : Nat) (epsStr : String) {nCls : Nat} (w : R50BWeights nCls)
    (bf16 : Bool) (sd : Option (Fin 16 → Vec N)) (e : SHlo (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) : SHlo (N * nCls) :=
  r34HeadGraphB N q q w.Wd w.bd
    (r50IdGraphBD "s4b2" epsStr (dpName 15) N q q w.s4b2 bf16 (r50Sd sd 15)
      (r50IdGraphBD "s4b1" epsStr (dpName 14) N q q w.s4b1 bf16 (r50Sd sd 14)
        (r50DownGraphBD "s4b0" epsStr (dpName 13) N q q w.s4b0 bf16 (r50Sd sd 13)
          (r50IdGraphBD "s3b5" epsStr (dpName 12) N (2 * q) (2 * q) w.s3b5 bf16 (r50Sd sd 12)
            (r50IdGraphBD "s3b4" epsStr (dpName 11) N (2 * q) (2 * q) w.s3b4 bf16 (r50Sd sd 11)
              (r50IdGraphBD "s3b3" epsStr (dpName 10) N (2 * q) (2 * q) w.s3b3 bf16 (r50Sd sd 10)
                (r50IdGraphBD "s3b2" epsStr (dpName 9) N (2 * q) (2 * q) w.s3b2 bf16 (r50Sd sd 9)
                  (r50IdGraphBD "s3b1" epsStr (dpName 8) N (2 * q) (2 * q) w.s3b1 bf16 (r50Sd sd 8)
                    (r50DownGraphBD "s3b0" epsStr (dpName 7) N (2 * q) (2 * q) w.s3b0 bf16 (r50Sd sd 7)
                      (r50IdGraphBD "s2b3" epsStr (dpName 6) N (2 * (2 * q)) (2 * (2 * q)) w.s2b3 bf16 (r50Sd sd 6)
                        (r50IdGraphBD "s2b2" epsStr (dpName 5) N (2 * (2 * q)) (2 * (2 * q)) w.s2b2 bf16 (r50Sd sd 5)
                          (r50IdGraphBD "s2b1" epsStr (dpName 4) N (2 * (2 * q)) (2 * (2 * q)) w.s2b1 bf16 (r50Sd sd 4)
                            (r50DownGraphBD "s2b0" epsStr (dpName 3) N (2 * (2 * q)) (2 * (2 * q)) w.s2b0 bf16 (r50Sd sd 3)
                              (r50IdGraphBD "s1b2" epsStr (dpName 2) N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2 bf16 (r50Sd sd 2)
                                (r50IdGraphBD "s1b1" epsStr (dpName 1) N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1 bf16 (r50Sd sd 1)
                                  (r50ProjGraphBD "s1b0" epsStr (dpName 0) N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0 bf16 (r50Sd sd 0)
                                    (r50StemGraphB epsStr N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.sW w.sb w.sε w.sγ w.sβ bf16
                                      e)))))))))))))))))

/-- **Forward-graph faithfulness with stochastic depth**: the typed graph denotes
    `resnet50ForwardBFullDrop` at the same sites, at either precision. -/
theorem resnet50FwdGraphBFullDrop_faithful (N q : Nat) (epsStr : String) {nCls : Nat}
    (w : R50BWeights nCls) (bf16 : Bool) (sd : Option (Fin 16 → Vec N)) (e : SHlo (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    den (resnet50FwdGraphBFullDrop N q epsStr w bf16 sd e)
      = resnet50ForwardBFullDrop N q w sd (den e) := by
  rw [resnet50ForwardBFullDrop, r50PreD16_apply, r50PreD15_apply, r50PreD14_apply, r50PreD13_apply, r50PreD12_apply, r50PreD11_apply, r50PreD10_apply, r50PreD9_apply, r50PreD8_apply, r50PreD7_apply, r50PreD6_apply, r50PreD5_apply, r50PreD4_apply, r50PreD3_apply, r50PreD2_apply, r50PreD1_apply, r50Pre0_apply]
  unfold resnet50FwdGraphBFullDrop
  rw [r34HeadGraphB_faithful, r50IdGraphBD_faithful, r50IdGraphBD_faithful, r50DownGraphBD_faithful, r50IdGraphBD_faithful, r50IdGraphBD_faithful, r50IdGraphBD_faithful, r50IdGraphBD_faithful, r50IdGraphBD_faithful, r50DownGraphBD_faithful, r50IdGraphBD_faithful, r50IdGraphBD_faithful, r50IdGraphBD_faithful, r50DownGraphBD_faithful, r50IdGraphBD_faithful, r50IdGraphBD_faithful, r50ProjGraphBD_faithful,
    r50StemGraphB_faithful]

end StableHLO

end Proofs
