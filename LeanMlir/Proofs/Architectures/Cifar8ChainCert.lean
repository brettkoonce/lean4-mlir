import LeanMlir.Proofs.Codegen.AdjointChainBridge
import LeanMlir.Proofs.Codegen.AdjointChainBridgeHet

/-!
# P5 — the whole-net float certificate as a Lean theorem (CIFAR-8 capstone)

The numerical probe (`scripts/adjoint_chain_probe.py` §3) certifies the committed
CIFAR-8 net: the float-evaluated logits sit within adjoint chainBudget ≈ 2.6 of
the real logits, below the logit magnitude ≈ 4.6 — argmax-safe. This file
assembles that certificate as a Lean THEOREM: `chain_adjointClose`
(`AdjointChainBridge.lean`) instantiated at the CIFAR-8 layer chain, with

- **fresh budgets from the PROVEN per-op modulus** — each stage is a
  `layerCert_reluDense` whose budget is `layerBudget M.u m w' β A 0` (the exact
  `FloatClose` modulus at input error 0, for a He-bounded relu∘dense layer);
- **tail gains supplied as NAMED HYPOTHESES** (`hH : TailGains …`) — the measured
  `Hᵢ` from the §3 backward/VJP sweep, quarantined exactly like `esig`/`egelu`:
  an ordinary argument with its provenance stated, never an axiom, so the
  statement stays 3-axiom clean;
- **the decision guarantee** `chain_argmaxSafe`: once the real margin exceeds
  `2·chainBudget`, the float net makes the SAME prediction as the exact net.

⚠⚠ **Scope — the v1 theorems below (`cifar8_chain_cert`,
`cifar8_chain_argmaxSafe`) have NO CIFAR-8 instance and cannot have one as
stated.** They are generic in `(m, w', β, A, layers)`; nothing in them is CIFAR-8
but the name. Two separate gaps, and the second is the binding one:

1. *Dimension.* `chain_adjointClose` is uniform-width (the `towerBack`-shape
   dim-preserving fold), so this is the uniform relu∘dense form. The committed
   net's conv trunk changes spatial/channel dims between stages; a fully
   dim-heterogeneous chain (sigma-typed layers) is the noted `AdjointChainBridge`
   v2 generalization, and heterogeneous stems/heads compose at the ends via
   `FloatClose.comp`.

2. ⛔ *Window (the one that actually bites).* `LayerCert m A` carries ONE window
   that every layer must map into ITSELF, so `hfit` unfolds to
   `(m·w'·A + β)·(1+u)^(m+2) ≤ A`, which forces **`m·w' ≤ 1`** whenever `A > 0`
   — and `A = 0` collapses the window to `{0}`, making the `hmargin` hypothesis
   (`2·chainBudget < 0`) unsatisfiable. At the committed `cifar8Verified`
   profile (`scripts/adjoint_chain_probe.py` §3: conv 3→16,16,16,16,32,32,32,32
   + dense 128/64/64/10, He init) `m·w'` runs **24 … 114**. So `hfit` is
   violated by one to two orders of magnitude at EVERY stage, and no amount of
   dimension generalisation repairs that. `hfit` is discharged nowhere in the
   repo — it appears only as a binder.

   The fix is not to satisfy `hfit` but to delete it: a heterogeneous-window
   `LayerCert m Ain Aout` with `Aout :=` the propagated bound discharges the
   obligation by construction, exactly as `FloatClose.comp` already threads
   `A → B → C`. That is `AdjointChainBridgeHet.lean`, and BOTH gaps close with it
   — indexing the chain by `(dim, window)` handles (1) in the same move, since
   `FloatClose` was always dimension-heterogeneous.

**✅ The v2 section at the bottom of this file carries the instance.**
`cifar8ChainH` is the committed `cifar8Verified` layer list op for op, built at
any `0 ≤ w'`, `0 ≤ β`, `0 ≤ A`, and `cifar8_chain_certH` /
`cifar8_chain_argmaxSafeH` are the certificate and the decision guarantee over
it. The v1 theorems are kept as the depth-generic statements they are.

The tail gains remain a measured hypothesis in BOTH versions, not a proven one
(`AdjointChainBridge.lean` §"Honest scope": `LipOnWindow` wants the gain over
the whole window, which a measured Jacobian underestimates). That is the
remaining quarantine, and it is unaffected by the window fix.

Everything here reuses proven bridges + the quarantine pattern; 3-axiom clean.
-/

namespace Proofs

open FloatModel

/-- One He-bounded dense layer: weights within `w'`, bias within `β`. The
    building block whose relu∘dense certificate is `layerCert_reluDense`. -/
structure BoundedDense (m : Nat) (w' β : ℝ) where
  W : Mat m m
  b : Vec m
  hW : ∀ i j, |W i j| ≤ w'
  hb : ∀ j, |b j| ≤ β

/-- The CIFAR-8 chain in uniform relu∘dense form: a `LayerCert` per stage, each
    with the PROVEN fresh budget `layerBudget M.u m w' β A 0` (independent of the
    specific weights — it depends only on the He bounds `w', β` and window `A`).
    Depth-generic — the committed net's 15 stages are just a longer list. -/
noncomputable def reluDenseTower (M : FloatModel) {m : Nat} {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (hm : 0 < m)
    (hfit : layerAct m w' β A + layerBudget M.u m w' β A 0 ≤ A) :
    List (BoundedDense m w' β) → List (LayerCert m A)
  | [] => []
  | d :: ds =>
      layerCert_reluDense M d.W d.b hw' hβ hA hm d.hW d.hb hfit
      :: reluDenseTower M hw' hβ hA hm hfit ds

/-- **The CIFAR-8 whole-net float certificate, as a theorem.** For the CIFAR-8
    relu∘dense tower with He-bounded weights, if the measured tail gains `Hs`
    hold (`hH`, provenance: probe §3) then the float net is within the
    depth-LINEAR `chainBudget = Σᵢ Hᵢ·bᵢ` of the real net — no gain products,
    the per-op Higham budgets amplified once each by their own measured tail. -/
theorem cifar8_chain_cert (M : FloatModel) {m : Nat} {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (hm : 0 < m)
    (hfit : layerAct m w' β A + layerBudget M.u m w' β A 0 ≤ A)
    (layers : List (BoundedDense m w' β)) (Hs : List ℝ)
    (hH : TailGains (reluDenseTower M hw' hβ hA hm hfit layers) Hs)
    (x : Vec m) (hx : ∀ k, |x k| ≤ A) (j : Fin m) :
    |chainF (reluDenseTower M hw' hβ hA hm hfit layers) x j
        - chainR (reluDenseTower M hw' hβ hA hm hfit layers) x j|
      ≤ chainBudget (reluDenseTower M hw' hβ hA hm hfit layers) Hs :=
  chain_adjointClose _ Hs hH x hx j

/-- **The decision guarantee: rounding cannot flip the CIFAR-8 prediction.**
    If the exact net's logit at `j₀` beats every other by more than twice the
    adjoint chainBudget, the float-evaluated CIFAR-8 net has the SAME argmax —
    the certificate turns the measured margin (§3: logits ≈ 4.6 vs budget ≈ 2.6,
    so a per-class margin > 2·2.6 is what §3 checks) into a proof that binary32
    rounding preserves the prediction. -/
theorem cifar8_chain_argmaxSafe (M : FloatModel) {m : Nat} {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (hm : 0 < m)
    (hfit : layerAct m w' β A + layerBudget M.u m w' β A 0 ≤ A)
    (layers : List (BoundedDense m w' β)) (Hs : List ℝ)
    (hH : TailGains (reluDenseTower M hw' hβ hA hm hfit layers) Hs)
    (x : Vec m) (hx : ∀ k, |x k| ≤ A) (j₀ : Fin m)
    (hmargin : ∀ j, j ≠ j₀ →
      2 * chainBudget (reluDenseTower M hw' hβ hA hm hfit layers) Hs
        < chainR (reluDenseTower M hw' hβ hA hm hfit layers) x j₀
          - chainR (reluDenseTower M hw' hβ hA hm hfit layers) x j) :
    ∀ j, j ≠ j₀ →
      chainF (reluDenseTower M hw' hβ hA hm hfit layers) x j
        < chainF (reluDenseTower M hw' hβ hA hm hfit layers) x j₀ :=
  chain_argmaxSafe _ Hs hH x hx j₀ hmargin

-- ════════════════════════════════════════════════════════════════
-- § v2 — the SAME certificate at the committed CIFAR-8 shape
-- ════════════════════════════════════════════════════════════════

/-! Everything above is the v1 chain, whose `hfit` no real net satisfies (§2 of the
header). What follows is the same certificate over `AdjointChainBridgeHet`'s
per-layer windows and dimensions, at the **actual committed `cifar8Verified`
shape** — conv 3→16,16,16,16,32,32,32,32 (3×3 SAME) with a max-pool after every
second conv (32→16→8→4→2), flatten 128, dense 64/64/10 — with no window side
condition anywhere. The tail gains remain a named hypothesis, as in v1. -/

/-- The committed CIFAR-8 net's weights, carrying their magnitude bounds. He init
    at these fan-ins gives `|W| ≈ 0.3…0.5`; nothing here constrains `w'` further,
    which is the point (v1 needed `m·w' ≤ 1`). -/
structure Cifar8Weights (w' β : ℝ) where
  cW1 : Kernel4 16 3 3 3
  cb1 : Vec 16
  cW2 : Kernel4 16 16 3 3
  cb2 : Vec 16
  cW3 : Kernel4 16 16 3 3
  cb3 : Vec 16
  cW4 : Kernel4 16 16 3 3
  cb4 : Vec 16
  cW5 : Kernel4 32 16 3 3
  cb5 : Vec 32
  cW6 : Kernel4 32 32 3 3
  cb6 : Vec 32
  cW7 : Kernel4 32 32 3 3
  cb7 : Vec 32
  cW8 : Kernel4 32 32 3 3
  cb8 : Vec 32
  dW1 : Mat 128 64
  db1 : Vec 64
  dW2 : Mat 64 64
  db2 : Vec 64
  dW3 : Mat 64 10
  db3 : Vec 10
  hcW1 : ∀ o c kh kw, |cW1 o c kh kw| ≤ w'
  hcb1 : ∀ o, |cb1 o| ≤ β
  hcW2 : ∀ o c kh kw, |cW2 o c kh kw| ≤ w'
  hcb2 : ∀ o, |cb2 o| ≤ β
  hcW3 : ∀ o c kh kw, |cW3 o c kh kw| ≤ w'
  hcb3 : ∀ o, |cb3 o| ≤ β
  hcW4 : ∀ o c kh kw, |cW4 o c kh kw| ≤ w'
  hcb4 : ∀ o, |cb4 o| ≤ β
  hcW5 : ∀ o c kh kw, |cW5 o c kh kw| ≤ w'
  hcb5 : ∀ o, |cb5 o| ≤ β
  hcW6 : ∀ o c kh kw, |cW6 o c kh kw| ≤ w'
  hcb6 : ∀ o, |cb6 o| ≤ β
  hcW7 : ∀ o c kh kw, |cW7 o c kh kw| ≤ w'
  hcb7 : ∀ o, |cb7 o| ≤ β
  hcW8 : ∀ o c kh kw, |cW8 o c kh kw| ≤ w'
  hcb8 : ∀ o, |cb8 o| ≤ β
  hdW1 : ∀ i j, |dW1 i j| ≤ w'
  hdb1 : ∀ j, |db1 j| ≤ β
  hdW2 : ∀ i j, |dW2 i j| ≤ w'
  hdb2 : ∀ j, |db2 j| ≤ β
  hdW3 : ∀ i j, |dW3 i j| ≤ w'
  hdb3 : ∀ j, |db3 j| ≤ β

/-- The per-stage fan-ins of the committed net, in chain order (max-pools carry the
    window unchanged and contribute no entry): eight receptive fields `ic·kH·kW`
    then the three dense input widths. -/
def cifar8FanIns : List Nat :=
  [3 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3,
   16 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3,
   128, 64, 64]

/-- **The committed CIFAR-8 net as a `LayerChain`** — 11 certified stages and 4
    max-pools, input `Vec (3·32·32)`, output `Vec 10`, windows propagating along
    `cifar8FanIns`. Builds for ANY `0 ≤ w'`, `0 ≤ β`, `0 ≤ A`: there is no `hfit`,
    and `cifar8_stage_defeats_hfit` shows v1 could not have accepted even one of
    these stages at He magnitudes. -/
noncomputable def cifar8ChainH (M : FloatModel) {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (W : Cifar8Weights w' β) :
    LayerChain (3 * 32 * 32) A 10 (windowFold M w' β cifar8FanIns A) :=
  let n := windowFold_nonneg M hw' hβ
  -- stage 1-2 @ 32×32, then pool to 16×16
  .cons (layerCertH_reluConv (h := 32) (w := 32) M W.cW1 W.cb1 hw' hβ
          (n [] hA) (by norm_num) W.hcW1 W.hcb1)
  (.cons (layerCertH_reluConv (h := 32) (w := 32) M W.cW2 W.cb2 hw' hβ
          (n [3 * 3 * 3] hA) (by norm_num) W.hcW2 W.hcb2)
  (.cons (layerCertH_maxPool 16 16 16 _)
  -- stage 3-4 @ 16×16, then pool to 8×8
  (.cons (layerCertH_reluConv (h := 16) (w := 16) M W.cW3 W.cb3 hw' hβ
          (n [3 * 3 * 3, 16 * 3 * 3] hA) (by norm_num) W.hcW3 W.hcb3)
  (.cons (layerCertH_reluConv (h := 16) (w := 16) M W.cW4 W.cb4 hw' hβ
          (n [3 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3] hA) (by norm_num) W.hcW4 W.hcb4)
  (.cons (layerCertH_maxPool 16 8 8 _)
  -- stage 5-6 @ 8×8, then pool to 4×4
  (.cons (layerCertH_reluConv (h := 8) (w := 8) M W.cW5 W.cb5 hw' hβ
          (n [3 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3] hA) (by norm_num)
          W.hcW5 W.hcb5)
  (.cons (layerCertH_reluConv (h := 8) (w := 8) M W.cW6 W.cb6 hw' hβ
          (n [3 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3] hA)
          (by norm_num) W.hcW6 W.hcb6)
  (.cons (layerCertH_maxPool 32 4 4 _)
  -- stage 7-8 @ 4×4, then pool to 2×2 (flatten 32·2·2 = 128)
  (.cons (layerCertH_reluConv (h := 4) (w := 4) M W.cW7 W.cb7 hw' hβ
          (n [3 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3,
              32 * 3 * 3] hA) (by norm_num) W.hcW7 W.hcb7)
  (.cons (layerCertH_reluConv (h := 4) (w := 4) M W.cW8 W.cb8 hw' hβ
          (n [3 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3,
              32 * 3 * 3, 32 * 3 * 3] hA) (by norm_num) W.hcW8 W.hcb8)
  (.cons (layerCertH_maxPool 32 2 2 _)
  -- the dense head: relu after the first two, bare dense for the logits
  -- (cifar8Verified.layers: `.dense 128 64, .relu, .dense 64 64, .relu, .dense 64 10`)
  (.cons (layerCertH_reluDense M W.dW1 W.db1 hw' hβ
          (n [3 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3,
              32 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3] hA) (by norm_num) W.hdW1 W.hdb1)
  (.cons (layerCertH_reluDense M W.dW2 W.db2 hw' hβ
          (n [3 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3,
              32 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3, 128] hA) (by norm_num)
          W.hdW2 W.hdb2)
  (.cons (layerCertH_dense M W.dW3 W.db3 hw' hβ
          (n [3 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3,
              32 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3, 128, 64] hA) (by norm_num)
          W.hdW3 W.hdb3)
    .nil))))))))))))))


/-- **The CIFAR-8 whole-net float certificate — at the committed shape, no window
    side condition.** The v2 peer of `cifar8_chain_cert`. The layer list of
    `cifar8ChainH` is `cifar8Verified.layers` op for op (8 convs `[16,16,32,32]`
    with SAME 3×3 kernels, relu after each, a max-pool after every second conv
    taking 32→16→8→4→2, flatten to 128, dense 64/64/10), so this is a statement
    about the committed net rather than a look-alike tower.

    What is still assumed, unchanged from v1: the tail gains `Hs` arrive as a named
    `TailGainsH` hypothesis with its provenance in `scripts/adjoint_chain_probe.py`
    §3, and `LipOnWindow` asks for the gain over the whole window while a measured
    Jacobian underestimates that supremum. What is no longer assumed: anything
    about `w'`, `β` or `A` — the windows propagate instead of having to be
    forward-invariant.

    ⚠ Not yet proved: that `chainRH (cifar8ChainH …)` is *definitionally* the
    committed `cifar8Verified` render. It is that net's ops in that order; the
    `rfl`-style tie is the peer of `WholeNetForwardTies` and is the next step. -/
theorem cifar8_chain_certH (M : FloatModel) {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (W : Cifar8Weights w' β)
    (Hs : List ℝ) (hH : TailGainsH (cifar8ChainH M hw' hβ hA W) Hs)
    (x : Vec (3 * 32 * 32)) (hx : ∀ k, |x k| ≤ A) (j : Fin 10) :
    |chainFH (cifar8ChainH M hw' hβ hA W) x j
        - chainRH (cifar8ChainH M hw' hβ hA W) x j|
      ≤ chainBudgetH (cifar8ChainH M hw' hβ hA W) Hs :=
  chain_adjointCloseH _ Hs hH x hx j

/-- **Rounding cannot flip the committed CIFAR-8 net's prediction.** The v2 peer of
    `cifar8_chain_argmaxSafe`, and — unlike it — instantiable: `cifar8ChainH` builds
    at any nonnegative `w'`, `β`, `A`, so He-init magnitudes are admissible. Once the
    exact net's logit margin at `j₀` exceeds twice the adjoint-chain budget, the
    float-evaluated net has the same argmax. -/
theorem cifar8_chain_argmaxSafeH (M : FloatModel) {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (W : Cifar8Weights w' β)
    (Hs : List ℝ) (hH : TailGainsH (cifar8ChainH M hw' hβ hA W) Hs)
    (x : Vec (3 * 32 * 32)) (hx : ∀ k, |x k| ≤ A) (j₀ : Fin 10)
    (hmargin : ∀ j, j ≠ j₀ →
      2 * chainBudgetH (cifar8ChainH M hw' hβ hA W) Hs
        < chainRH (cifar8ChainH M hw' hβ hA W) x j₀
          - chainRH (cifar8ChainH M hw' hβ hA W) x j) :
    ∀ j, j ≠ j₀ →
      chainFH (cifar8ChainH M hw' hβ hA W) x j
        < chainFH (cifar8ChainH M hw' hβ hA W) x j₀ :=
  chain_argmaxSafeH _ Hs hH x hx j₀ hmargin

/-- **The instance v1 could not have.** At the widest committed stage's magnitudes
    (fan-in 288, `|W| ≤ 1/2`) the v1 tower is contradictory for every positive window
    (`cifar8_stage_defeats_hfit`); the v2 chain builds at exactly those numbers. -/
noncomputable example (M : FloatModel) (W : Cifar8Weights (1/2) 1) :
    LayerChain (3 * 32 * 32) 1 10 (windowFold M (1/2) 1 cifar8FanIns 1) :=
  cifar8ChainH M (by norm_num) (by norm_num) (by norm_num) W


/-- ⭐ **The tie: the chain's real side IS the committed CIFAR-8 forward.** `chainRH`
    of `cifar8ChainH` is definitionally `cifarCnn8Forward` at the committed config
    (`ic=3, c1=c2=16, c3=c4=32, h=w=2, d1=64, nClasses=10, 3×3` kernels) — the same
    def `StableHLO.cifar8FwdGraph_faithful` certifies the emitted forward graph
    denotes. So `cifar8_chain_certH` / `cifar8_chain_argmaxSafeH` are statements
    about the committed net, not about a look-alike tower: the peer of
    `WholeNetForwardTies` for the adjoint-chain tier.

    ⚠ Writing this tie is what caught the chain's dense head being three BARE denses
    when `cifar8Verified.layers` has `.dense 128 64, .relu, .dense 64 64, .relu,
    .dense 64 10`. A chain that merely type-checks does not have to be the net; this
    `rfl` is the thing that makes "op for op" checkable rather than asserted. -/
theorem cifar8ChainH_chainRH_eq (M : FloatModel) {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (W : Cifar8Weights w' β) :
    chainRH (cifar8ChainH M hw' hβ hA W)
      = cifarCnn8Forward (ic := 3) (c1 := 16) (c2 := 16) (c3 := 32) (c4 := 32)
          (h := 2) (w := 2) (d1 := 64) (nClasses := 10) (kH := 3) (kW := 3)
          W.cW1 W.cb1 W.cW2 W.cb2 W.cW3 W.cb3 W.cW4 W.cb4
          W.cW5 W.cb5 W.cW6 W.cb6 W.cW7 W.cb7 W.cW8 W.cb8
          W.dW1 W.db1 W.dW2 W.db2 W.dW3 W.db3 := rfl

/-- **The certificate, restated on the committed forward.** `cifar8_chain_certH` with
    `chainRH` rewritten to `cifarCnn8Forward` — the float chain is within the
    adjoint-chain budget of the CIFAR-8 net the repo actually trains and emits. -/
theorem cifar8_chain_cert_committed (M : FloatModel) {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (W : Cifar8Weights w' β)
    (Hs : List ℝ) (hH : TailGainsH (cifar8ChainH M hw' hβ hA W) Hs)
    (x : Vec (3 * 32 * 32)) (hx : ∀ k, |x k| ≤ A) (j : Fin 10) :
    |chainFH (cifar8ChainH M hw' hβ hA W) x j
        - cifarCnn8Forward (ic := 3) (c1 := 16) (c2 := 16) (c3 := 32) (c4 := 32)
            (h := 2) (w := 2) (d1 := 64) (nClasses := 10) (kH := 3) (kW := 3)
            W.cW1 W.cb1 W.cW2 W.cb2 W.cW3 W.cb3 W.cW4 W.cb4
            W.cW5 W.cb5 W.cW6 W.cb6 W.cW7 W.cb7 W.cW8 W.cb8
            W.dW1 W.db1 W.dW2 W.db2 W.dW3 W.db3 x j|
      ≤ chainBudgetH (cifar8ChainH M hw' hβ hA W) Hs := by
  rw [← cifar8ChainH_chainRH_eq M hw' hβ hA W]
  exact cifar8_chain_certH M hw' hβ hA W Hs hH x hx j

end Proofs
