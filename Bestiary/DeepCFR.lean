import LeanMlir.Spec

/-! # Deep CFR — Bestiary entry

Deep Counterfactual Regret Minimization (Brown, Lerer, Gross & Sandholm, ICML 2019,
arXiv:1811.00164) is CFR with the two tables replaced by networks: an *advantage* network per
player regresses the sampled counterfactual regrets of an information set, and regret matching on
its positive outputs is the current strategy; an *average-strategy* network regresses the
strategies played. Each iteration runs external-sampling traversals against the advantage
networks, writes the sampled regrets and strategies into reservoir-sampled memories, and retrains
the traverser's advantage network from scratch on its memory. Single Deep CFR (Steinberger 2019,
arXiv:1901.07621) drops the strategy network: the average strategy is read exactly from the
stored advantage networks, each weighted by its own reach.

The architecture is not the point and the paper says so ("not highly tuned"): it is the loss. The
inputs are cards and bets; the paper's network (its Fig. 1) is two branches — the cards through
summed rank / suit / card embeddings per group of permutation-invariant cards and three dense
layers, the bets (one occurred-flag and one size per betting position) through two — concatenated
into three more dense layers with skip connections where the widths agree, a normalisation of the
last layer's features, and a dense head of one slot per action. Its heads-up flop hold'em network
has 98,948 parameters. Steinberger's Leduc experiments use three dense layers of 64 units, which
"adds up to more parameters than Leduc Hold'em has states".

Trained here: `demos/MainDeepCfrLeduc.lean` runs the loop on the Lean Leduc hold'em
(`LeanMlir/Leduc.lean`) with `deepCfrLeducNet` — the Leduc papers' three dense layers of 64 on
a feature row of one-hot cards, rank scalars and the betting history — and scores every arm by
exact exploitability; SD-CFR comes for free from the stored profiles. The loss is the weighted
squared error on the legal slots, delivered through the rank-2 DDPM MSE block as a host-built
target: zero new codegen.

## Variants

- `deepCfrLeducNet r` — the demo's advantage / strategy network for an `r`-rank Leduc deck
  (`2r + 24` input features)
- `deepCfrTrunk dim` — the paper's trunk after the two input branches meet: three dense layers
  of `dim` and the head, at the paper's default width 256. The branches, skips and the
  normalisation are what a sequential NetSpec does not carry; see the note.
-/

/-- The Leduc advantage / strategy network: `F = 2r + 24` features (private one-hot and rank
    scalar, public one-hot with a none slot and rank scalar, pair and private-above-public
    flags, 2 rounds × 4 action slots × {call, raise}, round flag, two contributions) → 64 → 64
    → 64 → 3 (fold, call, raise). No BatchNorm: the eval forward the loss reads and the train
    step's forward are the same function. -/
def deepCfrLeducNet (r : Nat := 3) : NetSpec where
  name := s!"Deep CFR Leduc r{r}"
  imageH := 1
  imageW := 1
  layers := [
    .dense (2 * r + 24) 64 .relu,
    .dense 64 64 .relu,
    .dense 64 64 .relu,
    .dense 64 3 .identity
  ]

/-- The paper's trunk after the card branch and the bet branch are concatenated (each `dim`
    wide): three dense layers of `dim` — the second and third carry a skip, `x_{i+1} = ReLU(A x
    + x)`, in the paper — then the action head, 3 slots for a fold / call / raise game. The
    appendix code's default width is 256. -/
def deepCfrTrunk (dim : Nat := 256) (actions : Nat := 3) : NetSpec where
  name := s!"Deep CFR trunk dim{dim}"
  imageH := 1
  imageW := 1
  layers := [
    .dense (2 * dim) dim .relu,
    .dense dim dim .relu,
    .dense dim dim .relu,
    .dense dim actions .identity
  ]

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — Deep CFR"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  CFR with networks in place of the tables: an advantage net per"
  IO.println "  player regresses sampled counterfactual regrets, an average-"
  IO.println "  strategy net regresses the strategies played (SD-CFR drops it)."

  (deepCfrLeducNet 3).summarize
  (deepCfrLeducNet 13).summarize
  (deepCfrTrunk 256).summarize

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives: dense + ReLU stacks. The paper's two"
  IO.println "    input branches (summed card embeddings; bet flags and sizes),"
  IO.println "    the skips on the equal-width layers and the last-layer"
  IO.println "    normalisation are not in a sequential NetSpec; the trunk is."
  IO.println "  • The object is the loop, not the net: external-sampling"
  IO.println "    traversals write instantaneous regrets to a reservoir, the"
  IO.println "    advantage net is retrained from scratch every iteration, and"
  IO.println "    regret matching on its positive outputs is the strategy."
  IO.println "  • Trained on Leduc hold'em in demos/MainDeepCfrLeduc.lean and"
  IO.println "    scored by exact exploitability (LeanMlir/Leduc.lean); the"
  IO.println "    matched-budget bracket is tabular external-sampling MCCFR."
