import LeanMlir.Proofs.Codegen.AdjointChainBridge

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

Scope (honest): `chain_adjointClose` is uniform-width (the `towerBack`-shape
dim-preserving fold), so this capstone is CIFAR-8 in its uniform relu∘dense
form (depth-generic over its stages via `reluDenseTower`). The committed net's
conv trunk changes spatial/channel dims between stages; a fully dim-heterogeneous
chain (sigma-typed layers) is the noted `AdjointChainBridge` v2 generalization,
and the heterogeneous stems/heads compose at the ends via `FloatClose.comp`.
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

end Proofs
