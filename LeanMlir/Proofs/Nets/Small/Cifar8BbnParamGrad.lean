module

public import LeanMlir.Proofs.Nets.Small.Cifar8BbnWholeBackB
public import LeanMlir.Proofs.Foundation.Batched.PoolSelB
public import LeanMlir.Proofs.Foundation.SmoothedBatchLoss
public import LeanMlir.Proofs.Nets.Small.Cifar8BbnStepTieB
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # The batch-statistics cifar8-BN net — every gradient node IS the loss's derivative

`Cifar8BbnStepTieB.cifar8Bbn_train_step_tiedB` ties the 38 gradient nodes of the `cifar8wb_bbn_*`
steps, each at the cotangent the rendered chain threads to it; `Cifar8BbnWholeBackB` says that
chain is the certified whole-net backward. This file is the rung above both: at those cotangents,
the loss of the WHOLE net with one parameter varied is differentiable in it and the node is its
gradient (`HasGradAt`), for any loss `L` of the logits with gradient `g` there
(`cifar8Bbn_net_lossGrad`). The batched peer of `Cifar8BnParamGrad.cifar8Bn_net_lossGrad`
(the per-image net) and of `ResNet34ParamGrad.r34_net_lossGrad`.

## The pools

A 2×2 max pool has no derivative at a window whose maximum is attained twice, and after a ReLU
every dead window is such a tie. So, as the per-image rung and ResNet-34's stem do, the pools are
frozen at a **selection** `σ` (one cell per window per example): the pool is then the gather at
`σ`, and `PoolSelB.maxPoolB_relu_eventuallyEq_sel` says the pooled ReLU IS that gather along any
parameter, near a point where every window is dead or tied only between **twins** — cells equal at
every value of the weights upstream of that pool (`C8PoolTwin1 … C8PoolTwin4`). The gradient is
then of the gather model, which is smooth, and the chain's pool cotangent is the scatter along `σ`
(`selScatter`). At `σ = argmax` that scatter is the render's `maxPoolBackB`
(`PoolSelB.batchMapAux_maxPoolBackFlat_eq_selScatter`), which is how `cifar8Bbn_net_lossGrad`'s
chain is the node tie's.

## How it is built

* `c8Z1 … c8Z8` — the eight pre-ReLU activations (`c8Aₖ = relu c8Zₖ`); `C8PoolTwinⱼ` — twins of
  pool `j`, stated on `c8Z₂ⱼ` at every record; `Cifar8BbnLossSmoothAt` — the hypotheses.
* `c8S₁ … c8S₈` — the loss read from stage `k`'s pre-activation through the REAL net;
  `c8G₁ … c8G₈` — the same through the gather model. `c8_germ_stageₖ`: along any parameter path
  through stage `k`'s weights, the two agree near the point.
* `cifar8Bbn_net_lossGrad` — the head by `hasGradAt_denseB`/`hasGradAt_relu`, the body by the
  gather model pulled back one op at a time (`hasGradAt_gatherReluB`, `hasGradAt_bnBatchLA`,
  `hasGradAt_conv`, `hasGradAt_relu`), each node by its `ParamGradNodes` lemma, and each
  transported to the real net by its stage's germ.

## Scope

One device; the conv weight nodes on the renderers' `bf16` switch (`convWAt_hasGradAt`), the
others carry no flag. The smoothed-CE instantiation at the artifacts' own cotangent is
`cifar8Bbn_net_lossGrad_CE`. -/

open Proofs Proofs.StableHLO Proofs.BackLinks Proofs.SmallParamGrad Proofs.GradNodeB
open scoped BigOperators

namespace Proofs

variable {ic c1 c2 c3 c4 h w d1 nC kH kW : Nat}

-- ════════════════════════════════════════════════════════════════
-- § The pre-ReLU activations
-- ════════════════════════════════════════════════════════════════

/-- Stage 1's pre-activation: BN₁ of conv₁. -/
noncomputable def c8Z1 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))) :=
  bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₁ p.β₁
    (batchMap N (flatConv p.W₁ p.b₁) x)

/-- Stage 2's pre-activation, pool 1's input before its ReLU. -/
noncomputable def c8Z2 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))) :=
  bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₂ p.β₂
    (batchMap N (flatConv p.W₂ p.b₂) (c8A1 N p ε x))

/-- Stage 3's pre-activation. -/
noncomputable def c8Z3 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w))))) :=
  bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₃ p.β₃
    (batchMap N (flatConv p.W₃ p.b₃) (c8P1 N p ε x))

/-- Stage 4's pre-activation, pool 2's input before its ReLU. -/
noncomputable def c8Z4 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w))))) :=
  bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₄ p.β₄
    (batchMap N (flatConv p.W₄ p.b₄) (c8A3 N p ε x))

/-- Stage 5's pre-activation. -/
noncomputable def c8Z5 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w)))) :=
  bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₅ p.β₅
    (batchMap N (flatConv p.W₅ p.b₅) (c8P2 N p ε x))

/-- Stage 6's pre-activation, pool 3's input before its ReLU. -/
noncomputable def c8Z6 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w)))) :=
  bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₆ p.β₆
    (batchMap N (flatConv p.W₆ p.b₆) (c8A5 N p ε x))

/-- Stage 7's pre-activation. -/
noncomputable def c8Z7 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c4 * (2 * h) * (2 * w))) :=
  bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₇ p.β₇ (batchMap N (flatConv p.W₇ p.b₇) (c8P3 N p ε x))

/-- Stage 8's pre-activation, pool 4's input before its ReLU. -/
noncomputable def c8Z8 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c4 * (2 * h) * (2 * w))) :=
  bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₈ p.β₈ (batchMap N (flatConv p.W₈ p.b₈) (c8A7 N p ε x))

-- ════════════════════════════════════════════════════════════════
-- § Twins of each pool
-- ════════════════════════════════════════════════════════════════

/-- Example `r`'s cell `(ci, p)` in the batched `[N, c, H, W]` index. -/
abbrev bIdx (N : Nat) {c H W : Nat} (r : Fin N) (ci : Fin c) (q : Fin H × Fin W) :
    Fin (N * (c * H * W)) :=
  finProdFinEquiv (r, t3Idx ci q.1 q.2)

/-- **Twins of pool 1** (example `r`): equal in every channel at every record of weights. Pool 1's
    pre-activation reads only stages 1 and 2, so this is "equal at every weight upstream"; `p` fixes
    the widths. -/
def C8PoolTwin1 (N : Nat) (_p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (r : Fin N) (q q' : Fin (2 * (2 * (2 * (2 * h)))) × Fin (2 * (2 * (2 * (2 * w))))) : Prop :=
  ∀ (p' : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ci : Fin c1),
    c8Z2 N p' ε x (bIdx N r ci q) = c8Z2 N p' ε x (bIdx N r ci q')

/-- **Twins of pool 2** (example `r`). -/
def C8PoolTwin2 (N : Nat) (_p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (r : Fin N) (q q' : Fin (2 * (2 * (2 * h))) × Fin (2 * (2 * (2 * w)))) : Prop :=
  ∀ (p' : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ci : Fin c2),
    c8Z4 N p' ε x (bIdx N r ci q) = c8Z4 N p' ε x (bIdx N r ci q')

/-- **Twins of pool 3** (example `r`). -/
def C8PoolTwin3 (N : Nat) (_p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (r : Fin N) (q q' : Fin (2 * (2 * h)) × Fin (2 * (2 * w))) : Prop :=
  ∀ (p' : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ci : Fin c3),
    c8Z6 N p' ε x (bIdx N r ci q) = c8Z6 N p' ε x (bIdx N r ci q')

/-- **Twins of pool 4** (example `r`). -/
def C8PoolTwin4 (N : Nat) (_p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (r : Fin N) (q q' : Fin (2 * h) × Fin (2 * w)) : Prop :=
  ∀ (p' : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ci : Fin c4),
    c8Z8 N p' ε x (bIdx N r ci q) = c8Z8 N p' ε x (bIdx N r ci q')

-- ════════════════════════════════════════════════════════════════
-- § The smooth-point hypotheses
-- ════════════════════════════════════════════════════════════════

/-- **What the loss gradient needs at `x`**: every pre-ReLU activation off zero (the eight stages
    and the two head layers); every pool's windows, in every example, dead or tied only between
    that pool's twins; each selection naming a maximum of every window of its pool's input. -/
structure Cifar8BbnLossSmoothAt (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (σ₁ : Fin N → Fin c1 → Fin (2 * (2 * (2 * h))) → Fin (2 * (2 * (2 * w))) → Fin 2 × Fin 2)
    (σ₂ : Fin N → Fin c2 → Fin (2 * (2 * h)) → Fin (2 * (2 * w)) → Fin 2 × Fin 2)
    (σ₃ : Fin N → Fin c3 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₄ : Fin N → Fin c4 → Fin h → Fin w → Fin 2 × Fin 2) : Prop where
  z1 : ∀ k, c8Z1 N p ε x k ≠ 0
  z2 : ∀ k, c8Z2 N p ε x k ≠ 0
  pool1 : MaxPool2SmoothUpToB N (C8PoolTwin1 N p ε x) (c8Z2 N p ε x)
  sel1 : PoolSelDomB N σ₁ (c8A2 N p ε x)
  z3 : ∀ k, c8Z3 N p ε x k ≠ 0
  z4 : ∀ k, c8Z4 N p ε x k ≠ 0
  pool2 : MaxPool2SmoothUpToB N (C8PoolTwin2 N p ε x) (c8Z4 N p ε x)
  sel2 : PoolSelDomB N σ₂ (c8A4 N p ε x)
  z5 : ∀ k, c8Z5 N p ε x k ≠ 0
  z6 : ∀ k, c8Z6 N p ε x k ≠ 0
  pool3 : MaxPool2SmoothUpToB N (C8PoolTwin3 N p ε x) (c8Z6 N p ε x)
  sel3 : PoolSelDomB N σ₃ (c8A6 N p ε x)
  z7 : ∀ k, c8Z7 N p ε x k ≠ 0
  z8 : ∀ k, c8Z8 N p ε x k ≠ 0
  pool4 : MaxPool2SmoothUpToB N (C8PoolTwin4 N p ε x) (c8Z8 N p ε x)
  sel4 : PoolSelDomB N σ₄ (c8A8 N p ε x)
  z9 : ∀ k, c8H9 N p ε x k ≠ 0
  za : ∀ k, c8HA N p ε x k ≠ 0

-- ════════════════════════════════════════════════════════════════
-- § The loss read from each stage's pre-activation: through the real net, and through the
--   gather model
-- ════════════════════════════════════════════════════════════════

section Suffixes
variable (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ) (L : Vec (N * nC) → Vec 1)

/-- The head: dense₉, relu, denseₐ, relu, dense_b, then `L`. -/
noncomputable def c8Gp4 (u : Vec (N * (c4 * h * w))) : Vec 1 :=
  L (batchMap N (dense p.Wb p.bb) (relu (N * d1) (batchMap N (dense p.Wa p.ba)
    (relu (N * d1) (batchMap N (dense p.W₉ p.b₉) u)))))

/-- Real: from stage 8's pre-activation. -/
noncomputable def c8S8 (y : Vec (N * (c4 * (2 * h) * (2 * w)))) : Vec 1 :=
  c8Gp4 N p L (batchMap N (maxPoolFlat c4 h w) (relu _ y))

/-- Real: from stage 7's pre-activation. -/
noncomputable def c8S7 (y : Vec (N * (c4 * (2 * h) * (2 * w)))) : Vec 1 :=
  c8S8 N p L (bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₈ p.β₈ (batchMap N (flatConv p.W₈ p.b₈) (relu _ y)))

/-- Real: from stage 6's pre-activation. -/
noncomputable def c8S6 (y : Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w))))) : Vec 1 :=
  c8S7 N p ε L (bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₇ p.β₇ (batchMap N (flatConv p.W₇ p.b₇)
    (batchMap N (maxPoolFlat c3 (2 * h) (2 * w)) (relu _ y))))

/-- Real: from stage 5's pre-activation. -/
noncomputable def c8S5 (y : Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w))))) : Vec 1 :=
  c8S6 N p ε L (bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₆ p.β₆
    (batchMap N (flatConv p.W₆ p.b₆) (relu _ y)))

/-- Real: from stage 4's pre-activation. -/
noncomputable def c8S4 (y : Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))))) : Vec 1 :=
  c8S5 N p ε L (bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₅ p.β₅
    (batchMap N (flatConv p.W₅ p.b₅) (batchMap N (maxPoolFlat c2 (2 * (2 * h)) (2 * (2 * w)))
      (relu _ y))))

/-- Real: from stage 3's pre-activation. -/
noncomputable def c8S3 (y : Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))))) : Vec 1 :=
  c8S4 N p ε L (bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₄ p.β₄
    (batchMap N (flatConv p.W₄ p.b₄) (relu _ y)))

/-- Real: from stage 2's pre-activation. -/
noncomputable def c8S2 (y : Vec (N * (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec 1 :=
  c8S3 N p ε L (bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₃ p.β₃
    (batchMap N (flatConv p.W₃ p.b₃)
      (batchMap N (maxPoolFlat c1 (2 * (2 * (2 * h))) (2 * (2 * (2 * w)))) (relu _ y))))

/-- Real: from stage 1's pre-activation. -/
noncomputable def c8S1 (y : Vec (N * (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec 1 :=
  c8S2 N p ε L (bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₂ p.β₂
    (batchMap N (flatConv p.W₂ p.b₂) (relu _ y)))

variable (σ₁ : Fin N → Fin c1 → Fin (2 * (2 * (2 * h))) → Fin (2 * (2 * (2 * w))) → Fin 2 × Fin 2)
  (σ₂ : Fin N → Fin c2 → Fin (2 * (2 * h)) → Fin (2 * (2 * w)) → Fin 2 × Fin 2)
  (σ₃ : Fin N → Fin c3 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
  (σ₄ : Fin N → Fin c4 → Fin h → Fin w → Fin 2 × Fin 2)

/-- Gather model: from stage 8's pre-activation, pool 4 frozen at `σ₄`. -/
noncomputable def c8G8 (y : Vec (N * (c4 * (2 * h) * (2 * w)))) : Vec 1 :=
  c8Gp4 N p L (fun k => relu _ y (poolSelIdxB N σ₄ k))

/-- Gather model: from stage 7's pre-activation. -/
noncomputable def c8G7 (y : Vec (N * (c4 * (2 * h) * (2 * w)))) : Vec 1 :=
  c8G8 N p L σ₄ (bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₈ p.β₈ (batchMap N (flatConv p.W₈ p.b₈) (relu _ y)))

/-- Gather model: from stage 6's pre-activation, pool 3 frozen at `σ₃`. -/
noncomputable def c8G6 (y : Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w))))) : Vec 1 :=
  c8G7 N p ε L σ₄ (bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₇ p.β₇ (batchMap N (flatConv p.W₇ p.b₇)
    (fun k => relu _ y (poolSelIdxB N σ₃ k))))

/-- Gather model: from stage 5's pre-activation. -/
noncomputable def c8G5 (y : Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w))))) : Vec 1 :=
  c8G6 N p ε L σ₃ σ₄ (bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₆ p.β₆
    (batchMap N (flatConv p.W₆ p.b₆) (relu _ y)))

/-- Gather model: from stage 4's pre-activation, pool 2 frozen at `σ₂`. -/
noncomputable def c8G4 (y : Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))))) : Vec 1 :=
  c8G5 N p ε L σ₃ σ₄ (bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₅ p.β₅
    (batchMap N (flatConv p.W₅ p.b₅) (fun k => relu _ y (poolSelIdxB N σ₂ k))))

/-- Gather model: from stage 3's pre-activation. -/
noncomputable def c8G3 (y : Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))))) : Vec 1 :=
  c8G4 N p ε L σ₂ σ₃ σ₄ (bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₄ p.β₄
    (batchMap N (flatConv p.W₄ p.b₄) (relu _ y)))

/-- Gather model: from stage 2's pre-activation, pool 1 frozen at `σ₁`. -/
noncomputable def c8G2 (y : Vec (N * (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec 1 :=
  c8G3 N p ε L σ₂ σ₃ σ₄ (bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₃ p.β₃
    (batchMap N (flatConv p.W₃ p.b₃) (fun k => relu _ y (poolSelIdxB N σ₁ k))))

/-- Gather model: from stage 1's pre-activation. -/
noncomputable def c8G1 (y : Vec (N * (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec 1 :=
  c8G2 N p ε L σ₁ σ₂ σ₃ σ₄ (bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₂ p.β₂
    (batchMap N (flatConv p.W₂ p.b₂) (relu _ y)))

end Suffixes

-- ════════════════════════════════════════════════════════════════
-- § From one stage's pre-activation to the next (the real net's ops, `p`'s weights)
-- ════════════════════════════════════════════════════════════════

section Up
variable (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)

/-- Stage 7's pre-activation to stage 8's: ReLU, conv₈, BN₈. -/
noncomputable def c8Up8of7 (y : Vec (N * (c4 * (2 * h) * (2 * w)))) : Vec (N * (c4 * (2 * h) * (2 * w))) :=
  bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₈ p.β₈ (batchMap N (flatConv p.W₈ p.b₈) (relu _ y))

/-- Stage 6's to stage 7's: ReLU, pool 3, conv₇, BN₇. -/
noncomputable def c8Up7of6 (y : Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w))))) :
    Vec (N * (c4 * (2 * h) * (2 * w))) :=
  bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₇ p.β₇ (batchMap N (flatConv p.W₇ p.b₇)
    (batchMap N (maxPoolFlat c3 (2 * h) (2 * w)) (relu _ y)))

/-- Stage 5's to stage 6's. -/
noncomputable def c8Up6of5 (y : Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w))))) :
    Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w)))) :=
  bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₆ p.β₆ (batchMap N (flatConv p.W₆ p.b₆) (relu _ y))

/-- Stage 4's to stage 5's: ReLU, pool 2, conv₅, BN₅. -/
noncomputable def c8Up5of4 (y : Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))))) :
    Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w)))) :=
  bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₅ p.β₅ (batchMap N (flatConv p.W₅ p.b₅)
    (batchMap N (maxPoolFlat c2 (2 * (2 * h)) (2 * (2 * w))) (relu _ y)))

/-- Stage 3's to stage 4's. -/
noncomputable def c8Up4of3 (y : Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))))) :
    Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w))))) :=
  bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₄ p.β₄
    (batchMap N (flatConv p.W₄ p.b₄) (relu _ y))

/-- Stage 2's to stage 3's: ReLU, pool 1, conv₃, BN₃. -/
noncomputable def c8Up3of2 (y : Vec (N * (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w))))) :=
  bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₃ p.β₃ (batchMap N (flatConv p.W₃ p.b₃)
    (batchMap N (maxPoolFlat c1 (2 * (2 * (2 * h))) (2 * (2 * (2 * w)))) (relu _ y)))

/-- Stage 1's to stage 2's. -/
noncomputable def c8Up2of1 (y : Vec (N * (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))) :=
  bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₂ p.β₂
    (batchMap N (flatConv p.W₂ p.b₂) (relu _ y))

theorem c8Up8of7_continuous (hε : 0 < ε) : Continuous (c8Up8of7 N p ε) :=
  ((bnBatchLA_continuous N c4 (2 * h) (2 * w) ε hε p.γ₈ p.β₈).comp
    (batchMap_continuous _ (flatConv_differentiable p.W₈ p.b₈).continuous)).comp (relu_continuous _)

theorem c8Up7of6_continuous (hε : 0 < ε) : Continuous (c8Up7of6 N p ε) :=
  ((bnBatchLA_continuous N c4 (2 * h) (2 * w) ε hε p.γ₇ p.β₇).comp
    (batchMap_continuous _ (flatConv_differentiable p.W₇ p.b₇).continuous)).comp
    ((batchMap_continuous _ (maxPoolFlat_continuous c3 (2 * h) (2 * w))).comp (relu_continuous _))

theorem c8Up6of5_continuous (hε : 0 < ε) : Continuous (c8Up6of5 N p ε) :=
  ((bnBatchLA_continuous N c3 (2 * (2 * h)) (2 * (2 * w)) ε hε p.γ₆ p.β₆).comp
    (batchMap_continuous _ (flatConv_differentiable p.W₆ p.b₆).continuous)).comp (relu_continuous _)

theorem c8Up5of4_continuous (hε : 0 < ε) : Continuous (c8Up5of4 N p ε) :=
  ((bnBatchLA_continuous N c3 (2 * (2 * h)) (2 * (2 * w)) ε hε p.γ₅ p.β₅).comp
    (batchMap_continuous _ (flatConv_differentiable p.W₅ p.b₅).continuous)).comp
    ((batchMap_continuous _ (maxPoolFlat_continuous c2 (2 * (2 * h)) (2 * (2 * w)))).comp
      (relu_continuous _))

theorem c8Up4of3_continuous (hε : 0 < ε) : Continuous (c8Up4of3 N p ε) :=
  ((bnBatchLA_continuous N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε hε p.γ₄ p.β₄).comp
    (batchMap_continuous _ (flatConv_differentiable p.W₄ p.b₄).continuous)).comp (relu_continuous _)

theorem c8Up3of2_continuous (hε : 0 < ε) : Continuous (c8Up3of2 N p ε) :=
  ((bnBatchLA_continuous N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε hε p.γ₃ p.β₃).comp
    (batchMap_continuous _ (flatConv_differentiable p.W₃ p.b₃).continuous)).comp
    ((batchMap_continuous _ (maxPoolFlat_continuous c1 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))))).comp
      (relu_continuous _))

theorem c8Up2of1_continuous (hε : 0 < ε) : Continuous (c8Up2of1 N p ε) :=
  ((bnBatchLA_continuous N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε hε p.γ₂ p.β₂).comp
    (batchMap_continuous _ (flatConv_differentiable p.W₂ p.b₂).continuous)).comp (relu_continuous _)

end Up

-- ════════════════════════════════════════════════════════════════
-- § The germs, stage by stage
--   `Z θ` is stage `k`'s pre-activation along a parameter, passing through `c8Zₖ` at `θ₀`; the
--   twin facts for each pool downstream are about the REAL pre-activation of that pool along `Z`
--   (`c8Up…`), which at a one-field record path is that record's `c8Z₂ⱼ` definitionally. Near `θ₀`
--   the real suffix and the gather model agree.
-- ════════════════════════════════════════════════════════════════

section Germs
variable (N : Nat) {P : Nat} (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
  (L : Vec (N * nC) → Vec 1)
  (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
  (σ₁ : Fin N → Fin c1 → Fin (2 * (2 * (2 * h))) → Fin (2 * (2 * (2 * w))) → Fin 2 × Fin 2)
  (σ₂ : Fin N → Fin c2 → Fin (2 * (2 * h)) → Fin (2 * (2 * w)) → Fin 2 × Fin 2)
  (σ₃ : Fin N → Fin c3 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
  (σ₄ : Fin N → Fin c4 → Fin h → Fin w → Fin 2 × Fin 2)

/-- Twins of pool `j`, along a pre-activation path: equal at every `θ`. -/
abbrev TwinAlong {c H W : Nat} (T : Fin N → Fin H × Fin W → Fin H × Fin W → Prop)
    (Z : Vec P → Vec (N * (c * H * W))) : Prop :=
  ∀ θ (r : Fin N) (ci : Fin c) (q q' : Fin H × Fin W), T r q q' →
    Z θ (bIdx N r ci q) = Z θ (bIdx N r ci q')

theorem c8_germ_stage8 (hε : 0 < ε) (Z : Vec P → Vec (N * (c4 * (2 * h) * (2 * w)))) (θ₀ : Vec P)
    (hZc : ContinuousAt Z θ₀) (hZ0 : Z θ₀ = c8Z8 N p ε x)
    (hT4 : TwinAlong N (C8PoolTwin4 N p ε x) Z)
    (hx : Cifar8BbnLossSmoothAt N p ε x σ₁ σ₂ σ₃ σ₄) :
    (fun θ => c8S8 N p L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G8 N p L σ₄ (Z θ)) := by
  have _ := hε
  have hev := maxPoolB_relu_eventuallyEq_sel Z σ₄ (C8PoolTwin4 N p ε x) hT4 θ₀ hZc
    (by rw [hZ0]; exact hx.z8) (by rw [hZ0]; exact hx.pool4) (by rw [hZ0]; exact hx.sel4)
  filter_upwards [hev] with θ hθ
  simp only [c8S8, c8G8]
  rw [hθ]

theorem c8_germ_stage7 (hε : 0 < ε) (Z : Vec P → Vec (N * (c4 * (2 * h) * (2 * w)))) (θ₀ : Vec P)
    (hZc : ContinuousAt Z θ₀) (hZ0 : Z θ₀ = c8Z7 N p ε x)
    (hT4 : TwinAlong N (C8PoolTwin4 N p ε x) (fun θ => c8Up8of7 N p ε (Z θ)))
    (hx : Cifar8BbnLossSmoothAt N p ε x σ₁ σ₂ σ₃ σ₄) :
    (fun θ => c8S7 N p ε L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G7 N p ε L σ₄ (Z θ)) :=
  c8_germ_stage8 N p ε L x σ₁ σ₂ σ₃ σ₄ hε (fun θ => c8Up8of7 N p ε (Z θ)) θ₀
    ((c8Up8of7_continuous N p ε hε).continuousAt.comp hZc) (by rw [hZ0]; rfl) hT4 hx

theorem c8_germ_stage6 (hε : 0 < ε) (Z : Vec P → Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w)))))
    (θ₀ : Vec P) (hZc : ContinuousAt Z θ₀) (hZ0 : Z θ₀ = c8Z6 N p ε x)
    (hT3 : TwinAlong N (C8PoolTwin3 N p ε x) Z)
    (hT4 : TwinAlong N (C8PoolTwin4 N p ε x) (fun θ => c8Up8of7 N p ε (c8Up7of6 N p ε (Z θ))))
    (hx : Cifar8BbnLossSmoothAt N p ε x σ₁ σ₂ σ₃ σ₄) :
    (fun θ => c8S6 N p ε L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G6 N p ε L σ₃ σ₄ (Z θ)) := by
  have h7 := c8_germ_stage7 N p ε L x σ₁ σ₂ σ₃ σ₄ hε (fun θ => c8Up7of6 N p ε (Z θ)) θ₀
    ((c8Up7of6_continuous N p ε hε).continuousAt.comp hZc) (by rw [hZ0]; rfl) hT4 hx
  have h6 := maxPoolB_relu_eventuallyEq_sel Z σ₃ (C8PoolTwin3 N p ε x) hT3 θ₀ hZc
    (by rw [hZ0]; exact hx.z6) (by rw [hZ0]; exact hx.pool3) (by rw [hZ0]; exact hx.sel3)
  filter_upwards [h7, h6] with θ h7θ h6θ
  show c8S7 N p ε L (c8Up7of6 N p ε (Z θ)) = c8G6 N p ε L σ₃ σ₄ (Z θ)
  rw [h7θ]
  simp only [c8G7, c8G6, c8Up7of6]
  rw [h6θ]

theorem c8_germ_stage5 (hε : 0 < ε) (Z : Vec P → Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w)))))
    (θ₀ : Vec P) (hZc : ContinuousAt Z θ₀) (hZ0 : Z θ₀ = c8Z5 N p ε x)
    (hT3 : TwinAlong N (C8PoolTwin3 N p ε x) (fun θ => c8Up6of5 N p ε (Z θ)))
    (hT4 : TwinAlong N (C8PoolTwin4 N p ε x)
      (fun θ => c8Up8of7 N p ε (c8Up7of6 N p ε (c8Up6of5 N p ε (Z θ)))))
    (hx : Cifar8BbnLossSmoothAt N p ε x σ₁ σ₂ σ₃ σ₄) :
    (fun θ => c8S5 N p ε L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G5 N p ε L σ₃ σ₄ (Z θ)) :=
  c8_germ_stage6 N p ε L x σ₁ σ₂ σ₃ σ₄ hε (fun θ => c8Up6of5 N p ε (Z θ)) θ₀
    ((c8Up6of5_continuous N p ε hε).continuousAt.comp hZc) (by rw [hZ0]; rfl) hT3 hT4 hx

theorem c8_germ_stage4 (hε : 0 < ε)
    (Z : Vec P → Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w))))))
    (θ₀ : Vec P) (hZc : ContinuousAt Z θ₀) (hZ0 : Z θ₀ = c8Z4 N p ε x)
    (hT2 : TwinAlong N (C8PoolTwin2 N p ε x) Z)
    (hT3 : TwinAlong N (C8PoolTwin3 N p ε x) (fun θ => c8Up6of5 N p ε (c8Up5of4 N p ε (Z θ))))
    (hT4 : TwinAlong N (C8PoolTwin4 N p ε x)
      (fun θ => c8Up8of7 N p ε (c8Up7of6 N p ε (c8Up6of5 N p ε (c8Up5of4 N p ε (Z θ))))))
    (hx : Cifar8BbnLossSmoothAt N p ε x σ₁ σ₂ σ₃ σ₄) :
    (fun θ => c8S4 N p ε L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G4 N p ε L σ₂ σ₃ σ₄ (Z θ)) := by
  have h5 := c8_germ_stage5 N p ε L x σ₁ σ₂ σ₃ σ₄ hε (fun θ => c8Up5of4 N p ε (Z θ)) θ₀
    ((c8Up5of4_continuous N p ε hε).continuousAt.comp hZc) (by rw [hZ0]; rfl) hT3 hT4 hx
  have h4 := maxPoolB_relu_eventuallyEq_sel Z σ₂ (C8PoolTwin2 N p ε x) hT2 θ₀ hZc
    (by rw [hZ0]; exact hx.z4) (by rw [hZ0]; exact hx.pool2) (by rw [hZ0]; exact hx.sel2)
  filter_upwards [h5, h4] with θ h5θ h4θ
  show c8S5 N p ε L (c8Up5of4 N p ε (Z θ)) = c8G4 N p ε L σ₂ σ₃ σ₄ (Z θ)
  rw [h5θ]
  simp only [c8G5, c8G4, c8Up5of4]
  rw [h4θ]

theorem c8_germ_stage3 (hε : 0 < ε)
    (Z : Vec P → Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w))))))
    (θ₀ : Vec P) (hZc : ContinuousAt Z θ₀) (hZ0 : Z θ₀ = c8Z3 N p ε x)
    (hT2 : TwinAlong N (C8PoolTwin2 N p ε x) (fun θ => c8Up4of3 N p ε (Z θ)))
    (hT3 : TwinAlong N (C8PoolTwin3 N p ε x)
      (fun θ => c8Up6of5 N p ε (c8Up5of4 N p ε (c8Up4of3 N p ε (Z θ)))))
    (hT4 : TwinAlong N (C8PoolTwin4 N p ε x)
      (fun θ => c8Up8of7 N p ε (c8Up7of6 N p ε (c8Up6of5 N p ε (c8Up5of4 N p ε
        (c8Up4of3 N p ε (Z θ)))))))
    (hx : Cifar8BbnLossSmoothAt N p ε x σ₁ σ₂ σ₃ σ₄) :
    (fun θ => c8S3 N p ε L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G3 N p ε L σ₂ σ₃ σ₄ (Z θ)) :=
  c8_germ_stage4 N p ε L x σ₁ σ₂ σ₃ σ₄ hε (fun θ => c8Up4of3 N p ε (Z θ)) θ₀
    ((c8Up4of3_continuous N p ε hε).continuousAt.comp hZc) (by rw [hZ0]; rfl) hT2 hT3 hT4 hx

theorem c8_germ_stage2 (hε : 0 < ε)
    (Z : Vec P → Vec (N * (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (θ₀ : Vec P) (hZc : ContinuousAt Z θ₀) (hZ0 : Z θ₀ = c8Z2 N p ε x)
    (hT1 : TwinAlong N (C8PoolTwin1 N p ε x) Z)
    (hT2 : TwinAlong N (C8PoolTwin2 N p ε x) (fun θ => c8Up4of3 N p ε (c8Up3of2 N p ε (Z θ))))
    (hT3 : TwinAlong N (C8PoolTwin3 N p ε x)
      (fun θ => c8Up6of5 N p ε (c8Up5of4 N p ε (c8Up4of3 N p ε (c8Up3of2 N p ε (Z θ))))))
    (hT4 : TwinAlong N (C8PoolTwin4 N p ε x)
      (fun θ => c8Up8of7 N p ε (c8Up7of6 N p ε (c8Up6of5 N p ε (c8Up5of4 N p ε
        (c8Up4of3 N p ε (c8Up3of2 N p ε (Z θ))))))))
    (hx : Cifar8BbnLossSmoothAt N p ε x σ₁ σ₂ σ₃ σ₄) :
    (fun θ => c8S2 N p ε L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G2 N p ε L σ₁ σ₂ σ₃ σ₄ (Z θ)) := by
  have h3 := c8_germ_stage3 N p ε L x σ₁ σ₂ σ₃ σ₄ hε (fun θ => c8Up3of2 N p ε (Z θ)) θ₀
    ((c8Up3of2_continuous N p ε hε).continuousAt.comp hZc) (by rw [hZ0]; rfl) hT2 hT3 hT4 hx
  have h2 := maxPoolB_relu_eventuallyEq_sel Z σ₁ (C8PoolTwin1 N p ε x) hT1 θ₀ hZc
    (by rw [hZ0]; exact hx.z2) (by rw [hZ0]; exact hx.pool1) (by rw [hZ0]; exact hx.sel1)
  filter_upwards [h3, h2] with θ h3θ h2θ
  show c8S3 N p ε L (c8Up3of2 N p ε (Z θ)) = c8G2 N p ε L σ₁ σ₂ σ₃ σ₄ (Z θ)
  rw [h3θ]
  simp only [c8G3, c8G2, c8Up3of2]
  rw [h2θ]

theorem c8_germ_stage1 (hε : 0 < ε)
    (Z : Vec P → Vec (N * (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (θ₀ : Vec P) (hZc : ContinuousAt Z θ₀) (hZ0 : Z θ₀ = c8Z1 N p ε x)
    (hT1 : TwinAlong N (C8PoolTwin1 N p ε x) (fun θ => c8Up2of1 N p ε (Z θ)))
    (hT2 : TwinAlong N (C8PoolTwin2 N p ε x)
      (fun θ => c8Up4of3 N p ε (c8Up3of2 N p ε (c8Up2of1 N p ε (Z θ)))))
    (hT3 : TwinAlong N (C8PoolTwin3 N p ε x)
      (fun θ => c8Up6of5 N p ε (c8Up5of4 N p ε (c8Up4of3 N p ε (c8Up3of2 N p ε
        (c8Up2of1 N p ε (Z θ)))))))
    (hT4 : TwinAlong N (C8PoolTwin4 N p ε x)
      (fun θ => c8Up8of7 N p ε (c8Up7of6 N p ε (c8Up6of5 N p ε (c8Up5of4 N p ε
        (c8Up4of3 N p ε (c8Up3of2 N p ε (c8Up2of1 N p ε (Z θ)))))))))
    (hx : Cifar8BbnLossSmoothAt N p ε x σ₁ σ₂ σ₃ σ₄) :
    (fun θ => c8S1 N p ε L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G1 N p ε L σ₁ σ₂ σ₃ σ₄ (Z θ)) :=
  c8_germ_stage2 N p ε L x σ₁ σ₂ σ₃ σ₄ hε (fun θ => c8Up2of1 N p ε (Z θ)) θ₀
    ((c8Up2of1_continuous N p ε hε).continuousAt.comp hZc) (by rw [hZ0]; rfl) hT1 hT2 hT3 hT4 hx

end Germs

-- ════════════════════════════════════════════════════════════════
-- § A stage's pre-activation moves continuously with each of its four parameters
-- ════════════════════════════════════════════════════════════════

section ParamCont
variable (N : Nat) (ε : ℝ) {a oc H W kH' kW' : Nat}

theorem c8_contW (hε : 0 < ε) (Wc : Kernel4 oc a kH' kW') (bc : Vec oc) (γ β : Vec oc) (y : Vec (N * (a * H * W))) :
    ContinuousAt (fun θ => bnBatchLA N oc H W ε γ β
      (batchMap N (flatConv (Kernel4.unflatten θ) bc) y)) (Kernel4.flatten Wc) :=
  (bnBatchLA_continuous N oc H W ε hε γ β).continuousAt.comp
    (batchMap_param_differentiableAt (fun θ y => flatConv (Kernel4.unflatten θ) bc y) y
      (Kernel4.flatten Wc) (fun y => (conv2d_weight_differentiable bc (Tensor3.unflatten y)) _)).continuousAt

theorem c8_contb (hε : 0 < ε) (Wc : Kernel4 oc a kH' kW') (bc : Vec oc) (γ β : Vec oc) (y : Vec (N * (a * H * W))) :
    ContinuousAt (fun θ => bnBatchLA N oc H W ε γ β (batchMap N (flatConv Wc θ) y)) bc :=
  (bnBatchLA_continuous N oc H W ε hε γ β).continuousAt.comp
    (batchMap_param_differentiableAt (fun θ y => flatConv Wc θ y) y bc
      (fun y => (conv2d_bias_differentiable Wc (Tensor3.unflatten y)) _)).continuousAt

theorem c8_contγ (hε : 0 < ε) (γ β : Vec oc) (v : Vec (N * (oc * H * W))) :
    ContinuousAt (fun θ => bnBatchLA N oc H W ε θ β v) γ := by
  have _ := hε
  exact (bnBatchLA_gamma_continuous N oc H W ε β v).continuousAt

theorem c8_contβ (hε : 0 < ε) (γ β : Vec oc) (v : Vec (N * (oc * H * W))) :
    ContinuousAt (fun θ => bnBatchLA N oc H W ε γ θ v) β := by
  have _ := hε
  exact (bnBatchLA_beta_continuous N oc H W ε γ v).continuousAt

end ParamCont

-- ════════════════════════════════════════════════════════════════
-- § The statement, and the theorem
-- ════════════════════════════════════════════════════════════════

/-- **Every batch-statistics cifar8-BN gradient node is the gradient of `L` in that parameter**:
    the 38 nodes `Cifar8BbnStepTieB` ties, each at the cotangent the chain threads to its layer
    with every pool routed at its selection, stated against `L` of `cifar8BbnForwardB` with that
    one parameter varied. The conv weight nodes are on the renderers' `bf16` switch. -/
def Cifar8BbnNetLossTiedB (N : Nat) (xN cotN vN epsStr : String)
    (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ) (bf16 : Bool)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (σ₁ : Fin N → Fin c1 → Fin (2 * (2 * (2 * h))) → Fin (2 * (2 * (2 * w))) → Fin 2 × Fin 2)
    (σ₂ : Fin N → Fin c2 → Fin (2 * (2 * h)) → Fin (2 * (2 * w)) → Fin 2 × Fin 2)
    (σ₃ : Fin N → Fin c3 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₄ : Fin N → Fin c4 → Fin h → Fin w → Fin 2 × Fin 2)
    (L : Vec (N * nC) → Vec 1) (g : Vec (N * nC)) : Prop :=
  -- the conv outputs (BN inputs)
  let hc1 := batchMap N (flatConv p.W₁ p.b₁) x
  let hc2 := batchMap N (flatConv p.W₂ p.b₂) (c8A1 N p ε x)
  let hc3 := batchMap N (flatConv p.W₃ p.b₃) (c8P1 N p ε x)
  let hc4 := batchMap N (flatConv p.W₄ p.b₄) (c8A3 N p ε x)
  let hc5 := batchMap N (flatConv p.W₅ p.b₅) (c8P2 N p ε x)
  let hc6 := batchMap N (flatConv p.W₆ p.b₆) (c8A5 N p ε x)
  let hc7 := batchMap N (flatConv p.W₇ p.b₇) (c8P3 N p ε x)
  let hc8 := batchMap N (flatConv p.W₈ p.b₈) (c8A7 N p ε x)
  -- the chain: head
  let dRA := rowDenseBackFlat N d1 nC p.Wb g
  let dHA := reluMaskB (N * d1) (c8HA N p ε x) dRA
  let dR9 := rowDenseBackFlat N d1 d1 p.Wa dHA
  let dH9 := reluMaskB (N * d1) (c8H9 N p ε x) dR9
  let dP4 := rowDenseBackFlat N (c4 * h * w) d1 p.W₉ dH9
  -- stage 4
  let dA8 := selScatter (poolSelIdxB N σ₄) dP4
  let dbn8 := reluMaskB _ (c8Z8 N p ε x) dA8
  let dhc8 := bnInB N c4 (2 * h) (2 * w) ε p.γ₈ hc8 dbn8
  let dA7 := cInB N (h := 2 * h) (w := 2 * w) p.W₈ p.b₈ dhc8
  let dbn7 := reluMaskB _ (c8Z7 N p ε x) dA7
  let dhc7 := bnInB N c4 (2 * h) (2 * w) ε p.γ₇ hc7 dbn7
  let dP3 := cInB N (h := 2 * h) (w := 2 * w) p.W₇ p.b₇ dhc7
  -- stage 3
  let dA6 := selScatter (poolSelIdxB N σ₃) dP3
  let dbn6 := reluMaskB _ (c8Z6 N p ε x) dA6
  let dhc6 := bnInB N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₆ hc6 dbn6
  let dA5 := cInB N (h := 2 * (2 * h)) (w := 2 * (2 * w)) p.W₆ p.b₆ dhc6
  let dbn5 := reluMaskB _ (c8Z5 N p ε x) dA5
  let dhc5 := bnInB N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₅ hc5 dbn5
  let dP2 := cInB N (h := 2 * (2 * h)) (w := 2 * (2 * w)) p.W₅ p.b₅ dhc5
  -- stage 2
  let dA4 := selScatter (poolSelIdxB N σ₂) dP2
  let dbn4 := reluMaskB _ (c8Z4 N p ε x) dA4
  let dhc4 := bnInB N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₄ hc4 dbn4
  let dA3 := cInB N (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) p.W₄ p.b₄ dhc4
  let dbn3 := reluMaskB _ (c8Z3 N p ε x) dA3
  let dhc3 := bnInB N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₃ hc3 dbn3
  let dP1 := cInB N (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) p.W₃ p.b₃ dhc3
  -- stage 1
  let dA2 := selScatter (poolSelIdxB N σ₁) dP1
  let dbn2 := reluMaskB _ (c8Z2 N p ε x) dA2
  let dhc2 := bnInB N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₂ hc2 dbn2
  let dA1 := cInB N (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) p.W₂ p.b₂ dhc2
  let dbn1 := reluMaskB _ (c8Z1 N p ε x) dA1
  let dhc1 := bnInB N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₁ hc1 dbn1
  -- ═══ the 38 nodes, in `cifar8BnSigList` order ═══
  -- layer 1
  HasGradAt (fun θ => L (cifar8BbnForwardB N { p with W₁ := Kernel4.unflatten θ } ε x))
      (Kernel4.flatten p.W₁)
      (den (SHlo.convWeightGradBAt bf16 (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w))))
        id xN p.b₁ x p.W₁ (.operand cotN dhc1)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with b₁ := θ } ε x)) p.b₁
      (den (SHlo.convBiasGradB (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w))))
        p.W₁ x p.b₁ (.operand cotN dhc1)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with γ₁ := θ } ε x)) p.γ₁
      (den (SHlo.bnGammaGradB vN epsStr ε (reassocB N c1 _ _ hc1)
        (.operand cotN (reassocB N c1 _ _ dbn1))))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with β₁ := θ } ε x)) p.β₁
      (den (SHlo.bnBetaGradB (N := N) (oc := c1) (h := 2 * (2 * (2 * (2 * h))))
        (w := 2 * (2 * (2 * (2 * w)))) (.operand cotN (reassocB N c1 _ _ dbn1))))
  -- layer 2
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with W₂ := Kernel4.unflatten θ } ε x))
      (Kernel4.flatten p.W₂)
      (den (SHlo.convWeightGradBAt bf16 (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w))))
        id xN p.b₂ (c8A1 N p ε x) p.W₂ (.operand cotN dhc2)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with b₂ := θ } ε x)) p.b₂
      (den (SHlo.convBiasGradB (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w))))
        p.W₂ (c8A1 N p ε x) p.b₂ (.operand cotN dhc2)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with γ₂ := θ } ε x)) p.γ₂
      (den (SHlo.bnGammaGradB vN epsStr ε (reassocB N c1 _ _ hc2)
        (.operand cotN (reassocB N c1 _ _ dbn2))))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with β₂ := θ } ε x)) p.β₂
      (den (SHlo.bnBetaGradB (N := N) (oc := c1) (h := 2 * (2 * (2 * (2 * h))))
        (w := 2 * (2 * (2 * (2 * w)))) (.operand cotN (reassocB N c1 _ _ dbn2))))
  -- layer 3
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with W₃ := Kernel4.unflatten θ } ε x))
      (Kernel4.flatten p.W₃)
      (den (SHlo.convWeightGradBAt bf16 (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w)))
        id xN p.b₃ (c8P1 N p ε x) p.W₃ (.operand cotN dhc3)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with b₃ := θ } ε x)) p.b₃
      (den (SHlo.convBiasGradB (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w)))
        p.W₃ (c8P1 N p ε x) p.b₃ (.operand cotN dhc3)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with γ₃ := θ } ε x)) p.γ₃
      (den (SHlo.bnGammaGradB vN epsStr ε (reassocB N c2 _ _ hc3)
        (.operand cotN (reassocB N c2 _ _ dbn3))))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with β₃ := θ } ε x)) p.β₃
      (den (SHlo.bnBetaGradB (N := N) (oc := c2) (h := 2 * (2 * (2 * h)))
        (w := 2 * (2 * (2 * w))) (.operand cotN (reassocB N c2 _ _ dbn3))))
  -- layer 4
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with W₄ := Kernel4.unflatten θ } ε x))
      (Kernel4.flatten p.W₄)
      (den (SHlo.convWeightGradBAt bf16 (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w)))
        id xN p.b₄ (c8A3 N p ε x) p.W₄ (.operand cotN dhc4)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with b₄ := θ } ε x)) p.b₄
      (den (SHlo.convBiasGradB (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w)))
        p.W₄ (c8A3 N p ε x) p.b₄ (.operand cotN dhc4)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with γ₄ := θ } ε x)) p.γ₄
      (den (SHlo.bnGammaGradB vN epsStr ε (reassocB N c2 _ _ hc4)
        (.operand cotN (reassocB N c2 _ _ dbn4))))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with β₄ := θ } ε x)) p.β₄
      (den (SHlo.bnBetaGradB (N := N) (oc := c2) (h := 2 * (2 * (2 * h)))
        (w := 2 * (2 * (2 * w))) (.operand cotN (reassocB N c2 _ _ dbn4))))
  -- layer 5
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with W₅ := Kernel4.unflatten θ } ε x))
      (Kernel4.flatten p.W₅)
      (den (SHlo.convWeightGradBAt bf16 (h := 2 * (2 * h)) (w := 2 * (2 * w))
        id xN p.b₅ (c8P2 N p ε x) p.W₅ (.operand cotN dhc5)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with b₅ := θ } ε x)) p.b₅
      (den (SHlo.convBiasGradB (h := 2 * (2 * h)) (w := 2 * (2 * w))
        p.W₅ (c8P2 N p ε x) p.b₅ (.operand cotN dhc5)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with γ₅ := θ } ε x)) p.γ₅
      (den (SHlo.bnGammaGradB vN epsStr ε (reassocB N c3 _ _ hc5)
        (.operand cotN (reassocB N c3 _ _ dbn5))))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with β₅ := θ } ε x)) p.β₅
      (den (SHlo.bnBetaGradB (N := N) (oc := c3) (h := 2 * (2 * h))
        (w := 2 * (2 * w)) (.operand cotN (reassocB N c3 _ _ dbn5))))
  -- layer 6
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with W₆ := Kernel4.unflatten θ } ε x))
      (Kernel4.flatten p.W₆)
      (den (SHlo.convWeightGradBAt bf16 (h := 2 * (2 * h)) (w := 2 * (2 * w))
        id xN p.b₆ (c8A5 N p ε x) p.W₆ (.operand cotN dhc6)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with b₆ := θ } ε x)) p.b₆
      (den (SHlo.convBiasGradB (h := 2 * (2 * h)) (w := 2 * (2 * w))
        p.W₆ (c8A5 N p ε x) p.b₆ (.operand cotN dhc6)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with γ₆ := θ } ε x)) p.γ₆
      (den (SHlo.bnGammaGradB vN epsStr ε (reassocB N c3 _ _ hc6)
        (.operand cotN (reassocB N c3 _ _ dbn6))))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with β₆ := θ } ε x)) p.β₆
      (den (SHlo.bnBetaGradB (N := N) (oc := c3) (h := 2 * (2 * h))
        (w := 2 * (2 * w)) (.operand cotN (reassocB N c3 _ _ dbn6))))
  -- layer 7
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with W₇ := Kernel4.unflatten θ } ε x))
      (Kernel4.flatten p.W₇)
      (den (SHlo.convWeightGradBAt bf16 (h := 2 * h) (w := 2 * w)
        id xN p.b₇ (c8P3 N p ε x) p.W₇ (.operand cotN dhc7)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with b₇ := θ } ε x)) p.b₇
      (den (SHlo.convBiasGradB (h := 2 * h) (w := 2 * w) p.W₇ (c8P3 N p ε x) p.b₇
        (.operand cotN dhc7)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with γ₇ := θ } ε x)) p.γ₇
      (den (SHlo.bnGammaGradB vN epsStr ε (reassocB N c4 _ _ hc7)
        (.operand cotN (reassocB N c4 _ _ dbn7))))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with β₇ := θ } ε x)) p.β₇
      (den (SHlo.bnBetaGradB (N := N) (oc := c4) (h := 2 * h) (w := 2 * w)
        (.operand cotN (reassocB N c4 _ _ dbn7))))
  -- layer 8
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with W₈ := Kernel4.unflatten θ } ε x))
      (Kernel4.flatten p.W₈)
      (den (SHlo.convWeightGradBAt bf16 (h := 2 * h) (w := 2 * w)
        id xN p.b₈ (c8A7 N p ε x) p.W₈ (.operand cotN dhc8)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with b₈ := θ } ε x)) p.b₈
      (den (SHlo.convBiasGradB (h := 2 * h) (w := 2 * w) p.W₈ (c8A7 N p ε x) p.b₈
        (.operand cotN dhc8)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with γ₈ := θ } ε x)) p.γ₈
      (den (SHlo.bnGammaGradB vN epsStr ε (reassocB N c4 _ _ hc8)
        (.operand cotN (reassocB N c4 _ _ dbn8))))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with β₈ := θ } ε x)) p.β₈
      (den (SHlo.bnBetaGradB (N := N) (oc := c4) (h := 2 * h) (w := 2 * w)
        (.operand cotN (reassocB N c4 _ _ dbn8))))
  -- the dense head
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with W₉ := Mat.unflatten θ } ε x))
      (Mat.flatten p.W₉)
      (den (SHlo.denseWeightGradB (c := d1) xN (c8P4 N p ε x) (.operand cotN dH9)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with b₉ := θ } ε x)) p.b₉
      (den (SHlo.denseBiasGradB (N := N) (.operand cotN dH9)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with Wa := Mat.unflatten θ } ε x))
      (Mat.flatten p.Wa)
      (den (SHlo.denseWeightGradB (c := d1) xN (c8R9 N p ε x) (.operand cotN dHA)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with ba := θ } ε x)) p.ba
      (den (SHlo.denseBiasGradB (N := N) (.operand cotN dHA)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with Wb := Mat.unflatten θ } ε x))
      (Mat.flatten p.Wb)
      (den (SHlo.denseWeightGradB (c := nC) xN (c8RA N p ε x) (.operand cotN g)))
  ∧ HasGradAt (fun θ => L (cifar8BbnForwardB N { p with bb := θ } ε x)) p.bb
      (den (SHlo.denseBiasGradB (N := N) (.operand cotN g)))

/-- **Every batch-statistics cifar8-BN gradient node is the gradient of `L` in that parameter**,
    whenever `g` is `L`'s gradient at the logits. Hypotheses: `0 < ε`; every ReLU off its kink,
    every pool window dead or tied only between that pool's twins, each selection naming a maximum
    of every window (`Cifar8BbnLossSmoothAt`); `hL`. -/
theorem cifar8Bbn_net_lossGrad (N : Nat) (xN cotN vN epsStr : String)
    (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ) (hε : 0 < ε) (bf16 : Bool)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (σ₁ : Fin N → Fin c1 → Fin (2 * (2 * (2 * h))) → Fin (2 * (2 * (2 * w))) → Fin 2 × Fin 2)
    (σ₂ : Fin N → Fin c2 → Fin (2 * (2 * h)) → Fin (2 * (2 * w)) → Fin 2 × Fin 2)
    (σ₃ : Fin N → Fin c3 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₄ : Fin N → Fin c4 → Fin h → Fin w → Fin 2 × Fin 2)
    (hx : Cifar8BbnLossSmoothAt N p ε x σ₁ σ₂ σ₃ σ₄)
    {L : Vec (N * nC) → Vec 1} {g : Vec (N * nC)}
    (hL : HasGradAt L (cifar8BbnForwardB N p ε x) g) :
    Cifar8BbnNetLossTiedB N xN cotN vN epsStr p ε bf16 x σ₁ σ₂ σ₃ σ₄ L g := by
  unfold Cifar8BbnNetLossTiedB
  intro hc1 hc2 hc3 hc4 hc5 hc6 hc7 hc8 dRA dHA dR9 dH9 dP4 dA8 dbn8 dhc8 dA7 dbn7 dhc7 dP3
    dA6 dbn6 dhc6 dA5 dbn5 dhc5 dP2 dA4 dbn4 dhc4 dA3 dbn3 dhc3 dP1 dA2 dbn2 dhc2 dA1 dbn1 dhc1
  -- ── the head, pulled back to the pooled features ──
  have hRA : HasGradAt (fun y => L (batchMap N (dense p.Wb p.bb) y)) (c8RA N p ε x) dRA :=
    hasGradAt_denseB p.Wb p.bb _ hL
  have hHA : HasGradAt (fun u => L (batchMap N (dense p.Wb p.bb) (relu _ u))) (c8HA N p ε x) dHA :=
    GradNodeB.hasGradAt_relu _ hx.za hRA
  have hR9 : HasGradAt (fun y => L (batchMap N (dense p.Wb p.bb) (relu _
      (batchMap N (dense p.Wa p.ba) y)))) (c8R9 N p ε x) dR9 :=
    hasGradAt_denseB p.Wa p.ba _ hHA
  have hH9 : HasGradAt (fun u => L (batchMap N (dense p.Wb p.bb) (relu _
      (batchMap N (dense p.Wa p.ba) (relu _ u))))) (c8H9 N p ε x) dH9 :=
    GradNodeB.hasGradAt_relu _ hx.z9 hR9
  have hP4 : HasGradAt (c8Gp4 N p L) (c8P4 N p ε x) dP4 := hasGradAt_denseB p.W₉ p.b₉ _ hH9
  -- ── stage 4 (layers 8, 7) through the gather at σ₄ ──
  have hpt4 : c8P4 N p ε x = fun k => relu _ (c8Z8 N p ε x) (poolSelIdxB N σ₄ k) :=
    batchMap_maxPoolFlat_eq_gatherB σ₄ _ hx.sel4
  have hB8 : HasGradAt (c8G8 N p L σ₄) (c8Z8 N p ε x) dbn8 :=
    hasGradAt_gatherReluB (poolSelIdxB N σ₄) (c8Z8 N p ε x) hx.z8 (hP4.congr_point hpt4)
  have hC8 : HasGradAt (fun z => c8G8 N p L σ₄ (bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₈ p.β₈ z))
      hc8 dhc8 := hasGradAt_bnBatchLA ε hε p.γ₈ p.β₈ hc8 hB8
  have hA7 : HasGradAt (fun y => c8G8 N p L σ₄ (bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₈ p.β₈
      (batchMap N (flatConv p.W₈ p.b₈) y))) (c8A7 N p ε x) dA7 :=
    hasGradAt_conv p.W₈ p.b₈ _ hC8
  have hB7 : HasGradAt (c8G7 N p ε L σ₄) (c8Z7 N p ε x) dbn7 := GradNodeB.hasGradAt_relu _ hx.z7 hA7
  have hC7 : HasGradAt (fun z => c8G7 N p ε L σ₄ (bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₇ p.β₇ z))
      hc7 dhc7 := hasGradAt_bnBatchLA ε hε p.γ₇ p.β₇ hc7 hB7
  have hP3 : HasGradAt (fun y => c8G7 N p ε L σ₄ (bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₇ p.β₇
      (batchMap N (flatConv p.W₇ p.b₇) y))) (c8P3 N p ε x) dP3 :=
    hasGradAt_conv p.W₇ p.b₇ _ hC7
  -- ── stage 3 (layers 6, 5) through the gather at σ₃ ──
  have hpt3 : c8P3 N p ε x = fun k => relu _ (c8Z6 N p ε x) (poolSelIdxB N σ₃ k) :=
    batchMap_maxPoolFlat_eq_gatherB σ₃ _ hx.sel3
  have hB6 : HasGradAt (c8G6 N p ε L σ₃ σ₄) (c8Z6 N p ε x) dbn6 :=
    hasGradAt_gatherReluB (poolSelIdxB N σ₃) (c8Z6 N p ε x) hx.z6 (hP3.congr_point hpt3)
  have hC6 : HasGradAt (fun z => c8G6 N p ε L σ₃ σ₄
      (bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₆ p.β₆ z)) hc6 dhc6 :=
    hasGradAt_bnBatchLA ε hε p.γ₆ p.β₆ hc6 hB6
  have hA5 : HasGradAt (fun y => c8G6 N p ε L σ₃ σ₄
      (bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₆ p.β₆ (batchMap N (flatConv p.W₆ p.b₆) y)))
      (c8A5 N p ε x) dA5 := hasGradAt_conv p.W₆ p.b₆ _ hC6
  have hB5 : HasGradAt (c8G5 N p ε L σ₃ σ₄) (c8Z5 N p ε x) dbn5 := GradNodeB.hasGradAt_relu _ hx.z5 hA5
  have hC5 : HasGradAt (fun z => c8G5 N p ε L σ₃ σ₄
      (bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₅ p.β₅ z)) hc5 dhc5 :=
    hasGradAt_bnBatchLA ε hε p.γ₅ p.β₅ hc5 hB5
  have hP2 : HasGradAt (fun y => c8G5 N p ε L σ₃ σ₄
      (bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₅ p.β₅ (batchMap N (flatConv p.W₅ p.b₅) y)))
      (c8P2 N p ε x) dP2 := hasGradAt_conv p.W₅ p.b₅ _ hC5
  -- ── stage 2 (layers 4, 3) through the gather at σ₂ ──
  have hpt2 : c8P2 N p ε x = fun k => relu _ (c8Z4 N p ε x) (poolSelIdxB N σ₂ k) :=
    batchMap_maxPoolFlat_eq_gatherB σ₂ _ hx.sel2
  have hB4 : HasGradAt (c8G4 N p ε L σ₂ σ₃ σ₄) (c8Z4 N p ε x) dbn4 :=
    hasGradAt_gatherReluB (poolSelIdxB N σ₂) (c8Z4 N p ε x) hx.z4 (hP2.congr_point hpt2)
  have hC4 : HasGradAt (fun z => c8G4 N p ε L σ₂ σ₃ σ₄
      (bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₄ p.β₄ z)) hc4 dhc4 :=
    hasGradAt_bnBatchLA ε hε p.γ₄ p.β₄ hc4 hB4
  have hA3 : HasGradAt (fun y => c8G4 N p ε L σ₂ σ₃ σ₄
      (bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₄ p.β₄
        (batchMap N (flatConv p.W₄ p.b₄) y))) (c8A3 N p ε x) dA3 :=
    hasGradAt_conv p.W₄ p.b₄ _ hC4
  have hB3 : HasGradAt (c8G3 N p ε L σ₂ σ₃ σ₄) (c8Z3 N p ε x) dbn3 :=
    GradNodeB.hasGradAt_relu _ hx.z3 hA3
  have hC3 : HasGradAt (fun z => c8G3 N p ε L σ₂ σ₃ σ₄
      (bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₃ p.β₃ z)) hc3 dhc3 :=
    hasGradAt_bnBatchLA ε hε p.γ₃ p.β₃ hc3 hB3
  have hP1 : HasGradAt (fun y => c8G3 N p ε L σ₂ σ₃ σ₄
      (bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₃ p.β₃
        (batchMap N (flatConv p.W₃ p.b₃) y))) (c8P1 N p ε x) dP1 :=
    hasGradAt_conv p.W₃ p.b₃ _ hC3
  -- ── stage 1 (layers 2, 1) through the gather at σ₁ ──
  have hpt1 : c8P1 N p ε x = fun k => relu _ (c8Z2 N p ε x) (poolSelIdxB N σ₁ k) :=
    batchMap_maxPoolFlat_eq_gatherB σ₁ _ hx.sel1
  have hB2 : HasGradAt (c8G2 N p ε L σ₁ σ₂ σ₃ σ₄) (c8Z2 N p ε x) dbn2 :=
    hasGradAt_gatherReluB (poolSelIdxB N σ₁) (c8Z2 N p ε x) hx.z2 (hP1.congr_point hpt1)
  have hC2 : HasGradAt (fun z => c8G2 N p ε L σ₁ σ₂ σ₃ σ₄
      (bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₂ p.β₂ z)) hc2 dhc2 :=
    hasGradAt_bnBatchLA ε hε p.γ₂ p.β₂ hc2 hB2
  have hA1 : HasGradAt (fun y => c8G2 N p ε L σ₁ σ₂ σ₃ σ₄
      (bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₂ p.β₂
        (batchMap N (flatConv p.W₂ p.b₂) y))) (c8A1 N p ε x) dA1 :=
    hasGradAt_conv p.W₂ p.b₂ _ hC2
  have hB1 : HasGradAt (c8G1 N p ε L σ₁ σ₂ σ₃ σ₄) (c8Z1 N p ε x) dbn1 :=
    GradNodeB.hasGradAt_relu _ hx.z1 hA1
  have hC1 : HasGradAt (fun z => c8G1 N p ε L σ₁ σ₂ σ₃ σ₄
      (bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₁ p.β₁ z)) hc1 dhc1 :=
    hasGradAt_bnBatchLA ε hε p.γ₁ p.β₁ hc1 hB1
  -- ── the germs along each parameter path, one per stage; the twin facts are the record's ──
  -- stage 8
  have g8 : ∀ {P : Nat} (Z : Vec P → _) (θ₀ : Vec P), ContinuousAt Z θ₀ → Z θ₀ = c8Z8 N p ε x →
      TwinAlong N (C8PoolTwin4 N p ε x) Z →
      (fun θ => c8S8 N p L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G8 N p L σ₄ (Z θ)) :=
    fun Z θ₀ hZc hZ0 hT4 => c8_germ_stage8 N p ε L x σ₁ σ₂ σ₃ σ₄ hε Z θ₀ hZc hZ0 hT4 hx
  have g7 : ∀ {P : Nat} (Z : Vec P → _) (θ₀ : Vec P), ContinuousAt Z θ₀ → Z θ₀ = c8Z7 N p ε x →
      TwinAlong N (C8PoolTwin4 N p ε x) (fun θ => c8Up8of7 N p ε (Z θ)) →
      (fun θ => c8S7 N p ε L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G7 N p ε L σ₄ (Z θ)) :=
    fun Z θ₀ hZc hZ0 hT4 => c8_germ_stage7 N p ε L x σ₁ σ₂ σ₃ σ₄ hε Z θ₀ hZc hZ0 hT4 hx
  have g6 : ∀ {P : Nat} (Z : Vec P → Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w))))) (θ₀ : Vec P),
      ContinuousAt Z θ₀ → Z θ₀ = c8Z6 N p ε x →
      TwinAlong N (C8PoolTwin3 N p ε x) (fun θ => (Z θ)) →
      TwinAlong N (C8PoolTwin4 N p ε x) (fun θ => (c8Up8of7 N p ε (c8Up7of6 N p ε (Z θ)))) →
      (fun θ => c8S6 N p ε L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G6 N p ε L σ₃ σ₄ (Z θ)) :=
    fun Z θ₀ hZc hZ0 hT3 hT4 => c8_germ_stage6 N p ε L x σ₁ σ₂ σ₃ σ₄ hε Z θ₀ hZc hZ0 hT3 hT4 hx
  have g5 : ∀ {P : Nat} (Z : Vec P → Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w))))) (θ₀ : Vec P),
      ContinuousAt Z θ₀ → Z θ₀ = c8Z5 N p ε x →
      TwinAlong N (C8PoolTwin3 N p ε x) (fun θ => (c8Up6of5 N p ε (Z θ))) →
      TwinAlong N (C8PoolTwin4 N p ε x) (fun θ => (c8Up8of7 N p ε (c8Up7of6 N p ε (c8Up6of5 N p ε (Z θ))))) →
      (fun θ => c8S5 N p ε L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G5 N p ε L σ₃ σ₄ (Z θ)) :=
    fun Z θ₀ hZc hZ0 hT3 hT4 => c8_germ_stage5 N p ε L x σ₁ σ₂ σ₃ σ₄ hε Z θ₀ hZc hZ0 hT3 hT4 hx
  have g4 : ∀ {P : Nat} (Z : Vec P → Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))))) (θ₀ : Vec P),
      ContinuousAt Z θ₀ → Z θ₀ = c8Z4 N p ε x →
      TwinAlong N (C8PoolTwin2 N p ε x) (fun θ => (Z θ)) →
      TwinAlong N (C8PoolTwin3 N p ε x) (fun θ => (c8Up6of5 N p ε (c8Up5of4 N p ε (Z θ)))) →
      TwinAlong N (C8PoolTwin4 N p ε x) (fun θ => (c8Up8of7 N p ε (c8Up7of6 N p ε (c8Up6of5 N p ε (c8Up5of4 N p ε (Z θ)))))) →
      (fun θ => c8S4 N p ε L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G4 N p ε L σ₂ σ₃ σ₄ (Z θ)) :=
    fun Z θ₀ hZc hZ0 hT2 hT3 hT4 => c8_germ_stage4 N p ε L x σ₁ σ₂ σ₃ σ₄ hε Z θ₀ hZc hZ0 hT2 hT3 hT4 hx
  have g3 : ∀ {P : Nat} (Z : Vec P → Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))))) (θ₀ : Vec P),
      ContinuousAt Z θ₀ → Z θ₀ = c8Z3 N p ε x →
      TwinAlong N (C8PoolTwin2 N p ε x) (fun θ => (c8Up4of3 N p ε (Z θ))) →
      TwinAlong N (C8PoolTwin3 N p ε x) (fun θ => (c8Up6of5 N p ε (c8Up5of4 N p ε (c8Up4of3 N p ε (Z θ))))) →
      TwinAlong N (C8PoolTwin4 N p ε x) (fun θ => (c8Up8of7 N p ε (c8Up7of6 N p ε (c8Up6of5 N p ε (c8Up5of4 N p ε (c8Up4of3 N p ε (Z θ))))))) →
      (fun θ => c8S3 N p ε L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G3 N p ε L σ₂ σ₃ σ₄ (Z θ)) :=
    fun Z θ₀ hZc hZ0 hT2 hT3 hT4 => c8_germ_stage3 N p ε L x σ₁ σ₂ σ₃ σ₄ hε Z θ₀ hZc hZ0 hT2 hT3 hT4 hx
  have g2 : ∀ {P : Nat} (Z : Vec P → Vec (N * (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) (θ₀ : Vec P),
      ContinuousAt Z θ₀ → Z θ₀ = c8Z2 N p ε x →
      TwinAlong N (C8PoolTwin1 N p ε x) (fun θ => (Z θ)) →
      TwinAlong N (C8PoolTwin2 N p ε x) (fun θ => (c8Up4of3 N p ε (c8Up3of2 N p ε (Z θ)))) →
      TwinAlong N (C8PoolTwin3 N p ε x) (fun θ => (c8Up6of5 N p ε (c8Up5of4 N p ε (c8Up4of3 N p ε (c8Up3of2 N p ε (Z θ)))))) →
      TwinAlong N (C8PoolTwin4 N p ε x) (fun θ => (c8Up8of7 N p ε (c8Up7of6 N p ε (c8Up6of5 N p ε (c8Up5of4 N p ε (c8Up4of3 N p ε (c8Up3of2 N p ε (Z θ)))))))) →
      (fun θ => c8S2 N p ε L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G2 N p ε L σ₁ σ₂ σ₃ σ₄ (Z θ)) :=
    fun Z θ₀ hZc hZ0 hT1 hT2 hT3 hT4 => c8_germ_stage2 N p ε L x σ₁ σ₂ σ₃ σ₄ hε Z θ₀ hZc hZ0 hT1 hT2 hT3 hT4 hx
  have g1 : ∀ {P : Nat} (Z : Vec P → Vec (N * (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) (θ₀ : Vec P),
      ContinuousAt Z θ₀ → Z θ₀ = c8Z1 N p ε x →
      TwinAlong N (C8PoolTwin1 N p ε x) (fun θ => (c8Up2of1 N p ε (Z θ))) →
      TwinAlong N (C8PoolTwin2 N p ε x) (fun θ => (c8Up4of3 N p ε (c8Up3of2 N p ε (c8Up2of1 N p ε (Z θ))))) →
      TwinAlong N (C8PoolTwin3 N p ε x) (fun θ => (c8Up6of5 N p ε (c8Up5of4 N p ε (c8Up4of3 N p ε (c8Up3of2 N p ε (c8Up2of1 N p ε (Z θ))))))) →
      TwinAlong N (C8PoolTwin4 N p ε x) (fun θ => (c8Up8of7 N p ε (c8Up7of6 N p ε (c8Up6of5 N p ε (c8Up5of4 N p ε (c8Up4of3 N p ε (c8Up3of2 N p ε (c8Up2of1 N p ε (Z θ))))))))) →
      (fun θ => c8S1 N p ε L (Z θ)) =ᶠ[nhds θ₀] (fun θ => c8G1 N p ε L σ₁ σ₂ σ₃ σ₄ (Z θ)) :=
    fun Z θ₀ hZc hZ0 hT1 hT2 hT3 hT4 => c8_germ_stage1 N p ε L x σ₁ σ₂ σ₃ σ₄ hε Z θ₀ hZc hZ0 hT1 hT2 hT3 hT4 hx
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  -- layer 1: W
  · exact (convWAt_hasGradAt bf16 xN cotN p.b₁ x p.W₁ hC1).congr_of_eventuallyEq
      (g1 (fun θ => bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₁ p.β₁ (batchMap N (flatConv (Kernel4.unflatten θ) p.b₁) x))
        (Kernel4.flatten p.W₁) (c8_contW N ε hε p.W₁ p.b₁ p.γ₁ p.β₁ x)
        (congrArg (fun W => bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₁ p.β₁ (batchMap N (flatConv W p.b₁) x))
          (Kernel4.unflatten_flatten p.W₁))
        (fun θ r ci q q' htw => htw { p with W₁ := Kernel4.unflatten θ } ci)
        (fun θ r ci q q' htw => htw { p with W₁ := Kernel4.unflatten θ } ci)
        (fun θ r ci q q' htw => htw { p with W₁ := Kernel4.unflatten θ } ci)
        (fun θ r ci q q' htw => htw { p with W₁ := Kernel4.unflatten θ } ci)).symm
  -- layer 1: b
  · exact (convB_hasGradAt cotN p.W₁ x p.b₁ hC1).congr_of_eventuallyEq
      (g1 (fun θ => bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₁ p.β₁ (batchMap N (flatConv p.W₁ θ) x))
        p.b₁ (c8_contb N ε hε p.W₁ p.b₁ p.γ₁ p.β₁ x) rfl
        (fun θ r ci q q' htw => htw { p with b₁ := θ } ci)
        (fun θ r ci q q' htw => htw { p with b₁ := θ } ci)
        (fun θ r ci q q' htw => htw { p with b₁ := θ } ci)
        (fun θ r ci q q' htw => htw { p with b₁ := θ } ci)).symm
  -- layer 1: γ
  · exact (bnGamma_hasGradAt vN epsStr cotN ε p.γ₁ p.β₁ hc1 hB1).congr_of_eventuallyEq
      (g1 (fun θ => bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε θ p.β₁ hc1) p.γ₁ (c8_contγ N ε hε p.γ₁ p.β₁ hc1) rfl
        (fun θ r ci q q' htw => htw { p with γ₁ := θ } ci)
        (fun θ r ci q q' htw => htw { p with γ₁ := θ } ci)
        (fun θ r ci q q' htw => htw { p with γ₁ := θ } ci)
        (fun θ r ci q q' htw => htw { p with γ₁ := θ } ci)).symm
  -- layer 1: β
  · exact (bnBeta_hasGradAt cotN ε p.γ₁ p.β₁ hc1 hB1).congr_of_eventuallyEq
      (g1 (fun θ => bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₁ θ hc1) p.β₁ (c8_contβ N ε hε p.γ₁ p.β₁ hc1) rfl
        (fun θ r ci q q' htw => htw { p with β₁ := θ } ci)
        (fun θ r ci q q' htw => htw { p with β₁ := θ } ci)
        (fun θ r ci q q' htw => htw { p with β₁ := θ } ci)
        (fun θ r ci q q' htw => htw { p with β₁ := θ } ci)).symm
  -- layer 2: W
  · exact (convWAt_hasGradAt bf16 xN cotN p.b₂ (c8A1 N p ε x) p.W₂ hC2).congr_of_eventuallyEq
      (g2 (fun θ => bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₂ p.β₂ (batchMap N (flatConv (Kernel4.unflatten θ) p.b₂) (c8A1 N p ε x)))
        (Kernel4.flatten p.W₂) (c8_contW N ε hε p.W₂ p.b₂ p.γ₂ p.β₂ (c8A1 N p ε x))
        (congrArg (fun W => bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₂ p.β₂ (batchMap N (flatConv W p.b₂) (c8A1 N p ε x)))
          (Kernel4.unflatten_flatten p.W₂))
        (fun θ r ci q q' htw => htw { p with W₂ := Kernel4.unflatten θ } ci)
        (fun θ r ci q q' htw => htw { p with W₂ := Kernel4.unflatten θ } ci)
        (fun θ r ci q q' htw => htw { p with W₂ := Kernel4.unflatten θ } ci)
        (fun θ r ci q q' htw => htw { p with W₂ := Kernel4.unflatten θ } ci)).symm
  -- layer 2: b
  · exact (convB_hasGradAt cotN p.W₂ (c8A1 N p ε x) p.b₂ hC2).congr_of_eventuallyEq
      (g2 (fun θ => bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₂ p.β₂ (batchMap N (flatConv p.W₂ θ) (c8A1 N p ε x)))
        p.b₂ (c8_contb N ε hε p.W₂ p.b₂ p.γ₂ p.β₂ (c8A1 N p ε x)) rfl
        (fun θ r ci q q' htw => htw { p with b₂ := θ } ci)
        (fun θ r ci q q' htw => htw { p with b₂ := θ } ci)
        (fun θ r ci q q' htw => htw { p with b₂ := θ } ci)
        (fun θ r ci q q' htw => htw { p with b₂ := θ } ci)).symm
  -- layer 2: γ
  · exact (bnGamma_hasGradAt vN epsStr cotN ε p.γ₂ p.β₂ hc2 hB2).congr_of_eventuallyEq
      (g2 (fun θ => bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε θ p.β₂ hc2) p.γ₂ (c8_contγ N ε hε p.γ₂ p.β₂ hc2) rfl
        (fun θ r ci q q' htw => htw { p with γ₂ := θ } ci)
        (fun θ r ci q q' htw => htw { p with γ₂ := θ } ci)
        (fun θ r ci q q' htw => htw { p with γ₂ := θ } ci)
        (fun θ r ci q q' htw => htw { p with γ₂ := θ } ci)).symm
  -- layer 2: β
  · exact (bnBeta_hasGradAt cotN ε p.γ₂ p.β₂ hc2 hB2).congr_of_eventuallyEq
      (g2 (fun θ => bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₂ θ hc2) p.β₂ (c8_contβ N ε hε p.γ₂ p.β₂ hc2) rfl
        (fun θ r ci q q' htw => htw { p with β₂ := θ } ci)
        (fun θ r ci q q' htw => htw { p with β₂ := θ } ci)
        (fun θ r ci q q' htw => htw { p with β₂ := θ } ci)
        (fun θ r ci q q' htw => htw { p with β₂ := θ } ci)).symm
  -- layer 3: W
  · exact (convWAt_hasGradAt bf16 xN cotN p.b₃ (c8P1 N p ε x) p.W₃ hC3).congr_of_eventuallyEq
      (g3 (fun θ => bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₃ p.β₃ (batchMap N (flatConv (Kernel4.unflatten θ) p.b₃) (c8P1 N p ε x)))
        (Kernel4.flatten p.W₃) (c8_contW N ε hε p.W₃ p.b₃ p.γ₃ p.β₃ (c8P1 N p ε x))
        (congrArg (fun W => bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₃ p.β₃ (batchMap N (flatConv W p.b₃) (c8P1 N p ε x)))
          (Kernel4.unflatten_flatten p.W₃))
        (fun θ r ci q q' htw => htw { p with W₃ := Kernel4.unflatten θ } ci)
        (fun θ r ci q q' htw => htw { p with W₃ := Kernel4.unflatten θ } ci)
        (fun θ r ci q q' htw => htw { p with W₃ := Kernel4.unflatten θ } ci)).symm
  -- layer 3: b
  · exact (convB_hasGradAt cotN p.W₃ (c8P1 N p ε x) p.b₃ hC3).congr_of_eventuallyEq
      (g3 (fun θ => bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₃ p.β₃ (batchMap N (flatConv p.W₃ θ) (c8P1 N p ε x)))
        p.b₃ (c8_contb N ε hε p.W₃ p.b₃ p.γ₃ p.β₃ (c8P1 N p ε x)) rfl
        (fun θ r ci q q' htw => htw { p with b₃ := θ } ci)
        (fun θ r ci q q' htw => htw { p with b₃ := θ } ci)
        (fun θ r ci q q' htw => htw { p with b₃ := θ } ci)).symm
  -- layer 3: γ
  · exact (bnGamma_hasGradAt vN epsStr cotN ε p.γ₃ p.β₃ hc3 hB3).congr_of_eventuallyEq
      (g3 (fun θ => bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε θ p.β₃ hc3) p.γ₃ (c8_contγ N ε hε p.γ₃ p.β₃ hc3) rfl
        (fun θ r ci q q' htw => htw { p with γ₃ := θ } ci)
        (fun θ r ci q q' htw => htw { p with γ₃ := θ } ci)
        (fun θ r ci q q' htw => htw { p with γ₃ := θ } ci)).symm
  -- layer 3: β
  · exact (bnBeta_hasGradAt cotN ε p.γ₃ p.β₃ hc3 hB3).congr_of_eventuallyEq
      (g3 (fun θ => bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₃ θ hc3) p.β₃ (c8_contβ N ε hε p.γ₃ p.β₃ hc3) rfl
        (fun θ r ci q q' htw => htw { p with β₃ := θ } ci)
        (fun θ r ci q q' htw => htw { p with β₃ := θ } ci)
        (fun θ r ci q q' htw => htw { p with β₃ := θ } ci)).symm
  -- layer 4: W
  · exact (convWAt_hasGradAt bf16 xN cotN p.b₄ (c8A3 N p ε x) p.W₄ hC4).congr_of_eventuallyEq
      (g4 (fun θ => bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₄ p.β₄ (batchMap N (flatConv (Kernel4.unflatten θ) p.b₄) (c8A3 N p ε x)))
        (Kernel4.flatten p.W₄) (c8_contW N ε hε p.W₄ p.b₄ p.γ₄ p.β₄ (c8A3 N p ε x))
        (congrArg (fun W => bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₄ p.β₄ (batchMap N (flatConv W p.b₄) (c8A3 N p ε x)))
          (Kernel4.unflatten_flatten p.W₄))
        (fun θ r ci q q' htw => htw { p with W₄ := Kernel4.unflatten θ } ci)
        (fun θ r ci q q' htw => htw { p with W₄ := Kernel4.unflatten θ } ci)
        (fun θ r ci q q' htw => htw { p with W₄ := Kernel4.unflatten θ } ci)).symm
  -- layer 4: b
  · exact (convB_hasGradAt cotN p.W₄ (c8A3 N p ε x) p.b₄ hC4).congr_of_eventuallyEq
      (g4 (fun θ => bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₄ p.β₄ (batchMap N (flatConv p.W₄ θ) (c8A3 N p ε x)))
        p.b₄ (c8_contb N ε hε p.W₄ p.b₄ p.γ₄ p.β₄ (c8A3 N p ε x)) rfl
        (fun θ r ci q q' htw => htw { p with b₄ := θ } ci)
        (fun θ r ci q q' htw => htw { p with b₄ := θ } ci)
        (fun θ r ci q q' htw => htw { p with b₄ := θ } ci)).symm
  -- layer 4: γ
  · exact (bnGamma_hasGradAt vN epsStr cotN ε p.γ₄ p.β₄ hc4 hB4).congr_of_eventuallyEq
      (g4 (fun θ => bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε θ p.β₄ hc4) p.γ₄ (c8_contγ N ε hε p.γ₄ p.β₄ hc4) rfl
        (fun θ r ci q q' htw => htw { p with γ₄ := θ } ci)
        (fun θ r ci q q' htw => htw { p with γ₄ := θ } ci)
        (fun θ r ci q q' htw => htw { p with γ₄ := θ } ci)).symm
  -- layer 4: β
  · exact (bnBeta_hasGradAt cotN ε p.γ₄ p.β₄ hc4 hB4).congr_of_eventuallyEq
      (g4 (fun θ => bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₄ θ hc4) p.β₄ (c8_contβ N ε hε p.γ₄ p.β₄ hc4) rfl
        (fun θ r ci q q' htw => htw { p with β₄ := θ } ci)
        (fun θ r ci q q' htw => htw { p with β₄ := θ } ci)
        (fun θ r ci q q' htw => htw { p with β₄ := θ } ci)).symm
  -- layer 5: W
  · exact (convWAt_hasGradAt bf16 xN cotN p.b₅ (c8P2 N p ε x) p.W₅ hC5).congr_of_eventuallyEq
      (g5 (fun θ => bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₅ p.β₅ (batchMap N (flatConv (Kernel4.unflatten θ) p.b₅) (c8P2 N p ε x)))
        (Kernel4.flatten p.W₅) (c8_contW N ε hε p.W₅ p.b₅ p.γ₅ p.β₅ (c8P2 N p ε x))
        (congrArg (fun W => bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₅ p.β₅ (batchMap N (flatConv W p.b₅) (c8P2 N p ε x)))
          (Kernel4.unflatten_flatten p.W₅))
        (fun θ r ci q q' htw => htw { p with W₅ := Kernel4.unflatten θ } ci)
        (fun θ r ci q q' htw => htw { p with W₅ := Kernel4.unflatten θ } ci)).symm
  -- layer 5: b
  · exact (convB_hasGradAt cotN p.W₅ (c8P2 N p ε x) p.b₅ hC5).congr_of_eventuallyEq
      (g5 (fun θ => bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₅ p.β₅ (batchMap N (flatConv p.W₅ θ) (c8P2 N p ε x)))
        p.b₅ (c8_contb N ε hε p.W₅ p.b₅ p.γ₅ p.β₅ (c8P2 N p ε x)) rfl
        (fun θ r ci q q' htw => htw { p with b₅ := θ } ci)
        (fun θ r ci q q' htw => htw { p with b₅ := θ } ci)).symm
  -- layer 5: γ
  · exact (bnGamma_hasGradAt vN epsStr cotN ε p.γ₅ p.β₅ hc5 hB5).congr_of_eventuallyEq
      (g5 (fun θ => bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε θ p.β₅ hc5) p.γ₅ (c8_contγ N ε hε p.γ₅ p.β₅ hc5) rfl
        (fun θ r ci q q' htw => htw { p with γ₅ := θ } ci)
        (fun θ r ci q q' htw => htw { p with γ₅ := θ } ci)).symm
  -- layer 5: β
  · exact (bnBeta_hasGradAt cotN ε p.γ₅ p.β₅ hc5 hB5).congr_of_eventuallyEq
      (g5 (fun θ => bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₅ θ hc5) p.β₅ (c8_contβ N ε hε p.γ₅ p.β₅ hc5) rfl
        (fun θ r ci q q' htw => htw { p with β₅ := θ } ci)
        (fun θ r ci q q' htw => htw { p with β₅ := θ } ci)).symm
  -- layer 6: W
  · exact (convWAt_hasGradAt bf16 xN cotN p.b₆ (c8A5 N p ε x) p.W₆ hC6).congr_of_eventuallyEq
      (g6 (fun θ => bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₆ p.β₆ (batchMap N (flatConv (Kernel4.unflatten θ) p.b₆) (c8A5 N p ε x)))
        (Kernel4.flatten p.W₆) (c8_contW N ε hε p.W₆ p.b₆ p.γ₆ p.β₆ (c8A5 N p ε x))
        (congrArg (fun W => bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₆ p.β₆ (batchMap N (flatConv W p.b₆) (c8A5 N p ε x)))
          (Kernel4.unflatten_flatten p.W₆))
        (fun θ r ci q q' htw => htw { p with W₆ := Kernel4.unflatten θ } ci)
        (fun θ r ci q q' htw => htw { p with W₆ := Kernel4.unflatten θ } ci)).symm
  -- layer 6: b
  · exact (convB_hasGradAt cotN p.W₆ (c8A5 N p ε x) p.b₆ hC6).congr_of_eventuallyEq
      (g6 (fun θ => bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₆ p.β₆ (batchMap N (flatConv p.W₆ θ) (c8A5 N p ε x)))
        p.b₆ (c8_contb N ε hε p.W₆ p.b₆ p.γ₆ p.β₆ (c8A5 N p ε x)) rfl
        (fun θ r ci q q' htw => htw { p with b₆ := θ } ci)
        (fun θ r ci q q' htw => htw { p with b₆ := θ } ci)).symm
  -- layer 6: γ
  · exact (bnGamma_hasGradAt vN epsStr cotN ε p.γ₆ p.β₆ hc6 hB6).congr_of_eventuallyEq
      (g6 (fun θ => bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε θ p.β₆ hc6) p.γ₆ (c8_contγ N ε hε p.γ₆ p.β₆ hc6) rfl
        (fun θ r ci q q' htw => htw { p with γ₆ := θ } ci)
        (fun θ r ci q q' htw => htw { p with γ₆ := θ } ci)).symm
  -- layer 6: β
  · exact (bnBeta_hasGradAt cotN ε p.γ₆ p.β₆ hc6 hB6).congr_of_eventuallyEq
      (g6 (fun θ => bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₆ θ hc6) p.β₆ (c8_contβ N ε hε p.γ₆ p.β₆ hc6) rfl
        (fun θ r ci q q' htw => htw { p with β₆ := θ } ci)
        (fun θ r ci q q' htw => htw { p with β₆ := θ } ci)).symm
  -- layer 7: W
  · exact (convWAt_hasGradAt bf16 xN cotN p.b₇ (c8P3 N p ε x) p.W₇ hC7).congr_of_eventuallyEq
      (g7 (fun θ => bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₇ p.β₇ (batchMap N (flatConv (Kernel4.unflatten θ) p.b₇) (c8P3 N p ε x)))
        (Kernel4.flatten p.W₇) (c8_contW N ε hε p.W₇ p.b₇ p.γ₇ p.β₇ (c8P3 N p ε x))
        (congrArg (fun W => bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₇ p.β₇ (batchMap N (flatConv W p.b₇) (c8P3 N p ε x)))
          (Kernel4.unflatten_flatten p.W₇))
        (fun θ r ci q q' htw => htw { p with W₇ := Kernel4.unflatten θ } ci)).symm
  -- layer 7: b
  · exact (convB_hasGradAt cotN p.W₇ (c8P3 N p ε x) p.b₇ hC7).congr_of_eventuallyEq
      (g7 (fun θ => bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₇ p.β₇ (batchMap N (flatConv p.W₇ θ) (c8P3 N p ε x)))
        p.b₇ (c8_contb N ε hε p.W₇ p.b₇ p.γ₇ p.β₇ (c8P3 N p ε x)) rfl
        (fun θ r ci q q' htw => htw { p with b₇ := θ } ci)).symm
  -- layer 7: γ
  · exact (bnGamma_hasGradAt vN epsStr cotN ε p.γ₇ p.β₇ hc7 hB7).congr_of_eventuallyEq
      (g7 (fun θ => bnBatchLA N c4 (2 * h) (2 * w) ε θ p.β₇ hc7) p.γ₇ (c8_contγ N ε hε p.γ₇ p.β₇ hc7) rfl
        (fun θ r ci q q' htw => htw { p with γ₇ := θ } ci)).symm
  -- layer 7: β
  · exact (bnBeta_hasGradAt cotN ε p.γ₇ p.β₇ hc7 hB7).congr_of_eventuallyEq
      (g7 (fun θ => bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₇ θ hc7) p.β₇ (c8_contβ N ε hε p.γ₇ p.β₇ hc7) rfl
        (fun θ r ci q q' htw => htw { p with β₇ := θ } ci)).symm
  -- layer 8: W
  · exact (convWAt_hasGradAt bf16 xN cotN p.b₈ (c8A7 N p ε x) p.W₈ hC8).congr_of_eventuallyEq
      (g8 (fun θ => bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₈ p.β₈ (batchMap N (flatConv (Kernel4.unflatten θ) p.b₈) (c8A7 N p ε x)))
        (Kernel4.flatten p.W₈) (c8_contW N ε hε p.W₈ p.b₈ p.γ₈ p.β₈ (c8A7 N p ε x))
        (congrArg (fun W => bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₈ p.β₈ (batchMap N (flatConv W p.b₈) (c8A7 N p ε x)))
          (Kernel4.unflatten_flatten p.W₈))
        (fun θ r ci q q' htw => htw { p with W₈ := Kernel4.unflatten θ } ci)).symm
  -- layer 8: b
  · exact (convB_hasGradAt cotN p.W₈ (c8A7 N p ε x) p.b₈ hC8).congr_of_eventuallyEq
      (g8 (fun θ => bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₈ p.β₈ (batchMap N (flatConv p.W₈ θ) (c8A7 N p ε x)))
        p.b₈ (c8_contb N ε hε p.W₈ p.b₈ p.γ₈ p.β₈ (c8A7 N p ε x)) rfl
        (fun θ r ci q q' htw => htw { p with b₈ := θ } ci)).symm
  -- layer 8: γ
  · exact (bnGamma_hasGradAt vN epsStr cotN ε p.γ₈ p.β₈ hc8 hB8).congr_of_eventuallyEq
      (g8 (fun θ => bnBatchLA N c4 (2 * h) (2 * w) ε θ p.β₈ hc8) p.γ₈ (c8_contγ N ε hε p.γ₈ p.β₈ hc8) rfl
        (fun θ r ci q q' htw => htw { p with γ₈ := θ } ci)).symm
  -- layer 8: β
  · exact (bnBeta_hasGradAt cotN ε p.γ₈ p.β₈ hc8 hB8).congr_of_eventuallyEq
      (g8 (fun θ => bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₈ θ hc8) p.β₈ (c8_contβ N ε hε p.γ₈ p.β₈ hc8) rfl
        (fun θ r ci q q' htw => htw { p with β₈ := θ } ci)).symm
  -- the dense head: no pool downstream, so no germ
  · exact denseW_hasGradAt xN cotN (c8P4 N p ε x) p.W₉ p.b₉ hH9
  · exact denseB_hasGradAt cotN p.W₉ (fun _ => 0) (c8P4 N p ε x) p.b₉ hH9
  · exact denseW_hasGradAt xN cotN (c8R9 N p ε x) p.Wa p.ba hHA
  · exact denseB_hasGradAt cotN p.Wa (fun _ => 0) (c8R9 N p ε x) p.ba hHA
  · exact denseW_hasGradAt xN cotN (c8RA N p ε x) p.Wb p.bb hL
  · exact denseB_hasGradAt cotN p.Wb (fun _ => 0) (c8RA N p ε x) p.bb hL


-- ════════════════════════════════════════════════════════════════
-- § The artifact's loss: the batch-mean softmax cross-entropy at a soft target, and its emitted
--   cotangent `scaleB invB (subB (softmaxRow logits) t)`
-- ════════════════════════════════════════════════════════════════

/-- The step tie's one-row casts are `BackLinks`' (`rowB`, `unrowB`), by definition. -/
theorem castB1_eq_rowB (N n : Nat) : Cifar8BbnTieB.castB1 N n = rowB N n := rfl
theorem unB1_eq_unrowB (N n : Nat) : Cifar8BbnTieB.unB1 N n = unrowB N n := rfl

/-- **The emitted cifar8-bbn loss cotangent is the smoothed chain at zero smoothing**: the
    render's `scaleB invB (subB (unrow softmaxRow) t)` is `smoothedLossCotGraph` at `α = 0` and
    divisor `invB⁻¹`, read back through `unrowB`. -/
theorem cifar8BbnLossCot_eq (N K : Nat) (sStr vN cotN aStr negAK bStr logN ohN : String)
    (invB : ℝ) (z t : Vec (N * K)) :
    den (SHlo.scaleB sStr invB (.subB (.operand vN (unrowB N K
        (den (.batchOp (N := N) (.softmaxRow (m := 1) (n := K)) (.operand vN (rowB N K z))))))
      (.operand cotN t)))
      = unrowB N K (den (smoothedLossCotGraph N K 0 invB⁻¹ aStr negAK bStr logN ohN (rowB N K z)
          (rowB N K t))) := by
  funext J
  simp only [den_scaleB, den_subB, den_operand, unrowB, smoothedLossCotGraph_den, den_batchOp,
    denOp, rowB]
  rw [show ∀ (h₁ : N * K = N * (1 * K)) (h₂ : N * (1 * K) = N * K),
      Fin.cast h₂ (Fin.cast h₁ J) = J from fun _ _ => Fin.ext rfl]
  rw [mul_zero, add_zero, zero_div, neg_zero, add_zero, div_inv_eq_mul]

/-- **The artifact's loss has the emitted cotangent as its gradient**, at any target whose rows
    each sum to one and any positive class count. -/
theorem cifar8BbnLoss_hasGradAt (N K : Nat) (sStr vN cotN : String) (invB : ℝ)
    (z t : Vec (N * K)) (ht : ∀ n, ∑ k : Fin K, targetRow N K (rowB N K t) n k = 1) :
    HasGradAt (smoothedBatchLoss N K 0 invB⁻¹ (rowB N K t)) z
      (den (SHlo.scaleB sStr invB (.subB (.operand vN (unrowB N K
        (den (.batchOp (N := N) (.softmaxRow (m := 1) (n := K)) (.operand vN (rowB N K z))))))
      (.operand cotN t)))) := by
  refine ⟨smoothedBatchLoss_differentiable N K 0 invB⁻¹ (rowB N K t) z, fun J => ?_⟩
  rw [smoothedBatchLoss_grad N K 0 invB⁻¹ "" "" "" "" "" (rowB N K t) z ht J,
    cifar8BbnLossCot_eq N K sStr vN cotN "" "" "" "" "" invB z t]

/-- **At the artifact's loss**: every node is the gradient of the batch-mean softmax
    cross-entropy at the soft target `t` (the graph input `%onehot`), with `g` the cotangent the
    render composes from `softmaxRow`, `subB` and `scaleB` — the chain `cifar8Bbn_train_step_tiedB`
    reads, at `castB1 = rowB` and `unB1 = unrowB`. -/
theorem cifar8Bbn_net_lossGrad_CE (N : Nat) (xN cotN vN epsStr sStr : String)
    (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ) (hε : 0 < ε) (bf16 : Bool) (invB : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (t : Vec (N * nC)) (ht : ∀ n, ∑ k : Fin nC, targetRow N nC (rowB N nC t) n k = 1)
    (σ₁ : Fin N → Fin c1 → Fin (2 * (2 * (2 * h))) → Fin (2 * (2 * (2 * w))) → Fin 2 × Fin 2)
    (σ₂ : Fin N → Fin c2 → Fin (2 * (2 * h)) → Fin (2 * (2 * w)) → Fin 2 × Fin 2)
    (σ₃ : Fin N → Fin c3 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₄ : Fin N → Fin c4 → Fin h → Fin w → Fin 2 × Fin 2)
    (hx : Cifar8BbnLossSmoothAt N p ε x σ₁ σ₂ σ₃ σ₄) :
    Cifar8BbnNetLossTiedB N xN cotN vN epsStr p ε bf16 x σ₁ σ₂ σ₃ σ₄
      (smoothedBatchLoss N nC 0 invB⁻¹ (rowB N nC t))
      (den (SHlo.scaleB sStr invB (.subB (.operand vN (unrowB N nC
        (den (.batchOp (N := N) (.softmaxRow (m := 1) (n := nC))
          (.operand vN (rowB N nC (cifar8BbnForwardB N p ε x)))))))
        (.operand cotN t)))) :=
  cifar8Bbn_net_lossGrad N xN cotN vN epsStr p ε hε bf16 x σ₁ σ₂ σ₃ σ₄ hx
    (cifar8BbnLoss_hasGradAt N nC sStr vN cotN invB _ t ht)

end Proofs
