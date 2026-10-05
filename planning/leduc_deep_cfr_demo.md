# leduc_deep_cfr_demo.md — Deep CFR on Leduc hold'em, scored by exact exploitability

**Opened 2026-10-05.** Goal: the Player of games section's imperfect-information demo, between
Pong and tic-tac-toe, replacing blackjack. Pong is DQN where no table can go; tic-tac-toe is
self-play search on a perfect-information game. Poker is the third kind: the opponent's card
is hidden, a good policy has to be *mixed*, and the score is not a win rate but
**exploitability** — what a best responder with full knowledge of the policy wins against it
per hand, computed exactly by one walk of the game tree. Zero is Nash. Nothing in the main
table is sampled.

The question the section asks: **does a network trained by regret regression learn to
bluff, and what is bluffing worth?** The second half is exact: remove the bluffs from an
equilibrium policy and the best responder's winnings rise by a number we can print.

Once the section lands, blackjack is deleted (§8) — user's call, 2026-10-05: "if we have that
can kill my darling".

## 0. Why this game

- **Small enough to solve exactly, big enough to need mixing.** Leduc hold'em (Southey et
  al. 2005, *Bayes' Bluff*, UAI) is the field's standard small poker: the benchmark in Deep
  CFR (Brown, Lerer, Gross & Sandholm 2019, arXiv:1811.00164), Single Deep CFR (Steinberger
  2019, arXiv:1901.07621), NFSP (Heinrich & Silver 2016) and OpenSpiel's test suite.
- **The instrument is a tree walk.** Best response to a fixed policy is one bottom-up pass
  over the information sets; exploitability = (BR value as P0 + BR value as P1) / 2
  (OpenSpiel's convention, NashConv / 2). Head-to-head value between two policies is one
  exact expectation over the 30 deals and the public card. This is blackjack's value
  iteration and tic-tac-toe's solver for this game.
- **A scale knob, like tic-tac-toe's n.** The deck is r ranks × 2 suits; r = 3 is Leduc.
  The rules, solver, BR and trainer are parametric in r (§5, the scale arm).
- **Zero new codegen.** Deep CFR's two losses are weighted squared errors, which is the
  squared-error block with a host-written target — the move the tic-tac-toe and Pong
  trainers already make (§3).

### Verified 2026-10-05 (Python prototype, ~150 lines, not in the tree)

| | value |
|---|---|
| information sets, suit-aware / rank-level | **936 / 288** (suits never matter in Leduc: no flushes) |
| game-tree nodes, r = 3 / 6 / 13 | 9,450 / 100,980 / 1,179,750 |
| rank-level information sets, r = 3 / 6 / 13 | 288 / 1,116 / 5,148 |
| CFR+ (alternating, linear averaging): exploitability at 10 / 100 / 300 / 1,000 iterations | 0.61 / 0.0134 / 0.0023 / **0.00025** chips/hand (71 s in Python) |
| game value to P0 at that profile | **−0.0856** chips/hand (the published Leduc value) |
| exploitability of the uniform-random policy | 2.3736 chips/hand |
| bluffs at that profile: P0 raises round 1 with J / Q / K | 7.3% / 74.1% / 75.3% |
| P0 opens round 2 after check-check, no pair, raise freq J/Q · J/K · Q/K | 19.5% · 7.5% · 58.5% |

The jack under a queen is the worst holding at the table (it beats nothing a showdown can
produce), and the equilibrium raises it one time in five. That is the figure's cell.

Rules, as OpenSpiel's `leduc_poker` spells them (prototype matched its 936): 6 cards J Q K ×
2 suits, ante 1 each, one private card each, two betting rounds, bet size 2 then 4, at most
two raises per round, fold legal only when facing a bet, P0 acts first in both rounds, one
public card dealt between rounds; showdown: pair with the public card wins, else the higher
card, equal ranks split.

## 1. The game and the instrument — `LeanMlir/Leduc.lean`, C in `ffi/f32_helpers.c`

- `LeanMlir/Leduc.lean`: pure Lean like `Blackjack.lean` / `Pong.lean` / `TicTacToe.lean`
  (shares `FloatFmt`): `State` (r, cards, public, round, per-round action strings, pot),
  `legal`, `step`, `payoff`, `infoKey` (player, private rank, public rank or none, the two
  strings), deterministic dealing from a `StdGen`.
- C, section `lean_leduc_*` (Lean's host push costs ~1.5 µs/float, `lean_host_push_cost`;
  every tree walk goes in C, as tic-tac-toe's solver does):
  - `enumerate` — the information-set table for r, a dense index per key, legal masks,
    the feature rows of §2 for every set (one batched forward scores the whole game).
  - `cfr_plus` — the reference solver, alternating updates, linear averaging; and `dcfr`
    (Brown & Sandholm 2019, discounted CFR) as the second exact solver for §4's
    non-uniqueness check.
  - `best_response`, `exploitability`, `head_to_head` — exact, over a strategy table
    `[nInfo, 3]`.
  - `es_mccfr` — tabular external-sampling MCCFR (Lanctot et al. 2009), the matched-budget
    bracket, counting nodes touched.
  - `traverse` — Deep CFR's external-sampling traversal against a strategy table, writing
    (features, instantaneous regrets, legal mask, iteration) rows to the advantage
    reservoir and (features, σ, iteration) rows to the strategy reservoir.
- `lake exe leduc-env [r=3]`: the counts, CFR+ to 1e-4, the game value, every scripted
  arm's exploitability and head-to-head, and the gates of §7.

## 2. The network

A dense stack, no BatchNorm (tic-tac-toe §2's reason: with BN the eval forward the loss
trick reads and the train step's forward are different functions):

```
.dense F 64 .relu, .dense 64 64 .relu, .dense 64 64 .relu, .dense 64 3 .identity
```

Output slots = fold, call/check, raise; illegal slots are masked on the host. Features
(F = 2r + 24 at the current spelling; fix in Phase 1): private rank one-hot (r) and as a
scalar rank / (r − 1); public rank one-hot (r + 1, slot r = not dealt) and scalar; pair
flag; private-above-public flag; per round, 4 action slots × {call, raise} one-hot (16);
round flag; pot contributions / max pot (2). The scalars and the two flags are what let the
net generalise across ranks in the scale arm (§5); they are an inductive bias and the
section says so.

Three parameter sets share one compiled graph: the advantage net per player (2) and the
average-strategy net (1). ~10.5k parameters each at r = 3 (F = 30).

## 3. The training loop — `demos/MainDeepCfrLeduc.lean`

Deep CFR as published, with one shortcut that does not change the algorithm:

1. Iteration t = 1..T, player p ∈ {0, 1}:
   - One batched forward of p's advantage net over **every** information set → regret
     matching on the positive part (argmax of the advantages if none is positive, as the
     paper does) → σ_t as a table. σ_t is fixed within an iteration, so the table is
     exactly what per-node queries would return; at r = 13 it is 5,148 rows, one call.
     (The shortcut is what Leduc's size buys; it is stated in the section.)
   - K external-sampling traversals in C (`traverse`) against σ_t for both seats.
   - Re-initialise p's advantage net on the host (the paper retrains from scratch each
     iteration; compile once, re-init params) and train it on the advantage reservoir.
2. After T iterations, train the average-strategy net on the strategy reservoir.

**The losses through the squared-error block** (`trainStepAdamF32Ddpm`, `useDdpm := true`,
`ddpmOutShape := [B, 3, 1, 1]`). The block's gradient is (2 / (nOut·B))·(out − y). The host
runs the eval forward on the batch and writes

  y = out − (nOut / 2) · g,  g = w ⊙ m ⊙ (out − target)

with m the legal mask and w = t / mean(t over the batch) (linear CFR weighting), so the
block's gradient is g / B: the weighted MSE on legal slots, zero on illegal ones. Advantage
net: target = instantaneous regrets. Strategy net: target = σ, weight t. This is exactly
`MainAlphaZeroTtt.lean`'s `targets` (`y = out − nOut·g/2`) with a different g.

Starting hyperparameters (tuned in Phase 2, logged in §11): T = 100 iterations, K = 1,000
traversals per player per iteration, reservoirs 1M rows, 1,000 Adam steps per advantage
net at batch 512, lr 1e-3; 4,000 steps for the strategy net.

## 4. Arms and the table

Every arm is a strategy table `[nInfo, 3]`, scored by the C instrument. r = 3, three seeds
for every learned arm (`LEAN_MLIR_SEED`):

| arm | exploitability (chips/hand) | head-to-head vs CFR+, seat-averaged | worst-hand raise freq |
|---|---|---|---|
| uniform random | 2.3736 | | |
| scripted "honest": raise with a pair or a K, call otherwise, fold J to a bet | Phase 0 | | 0 |
| CFR+ equilibrium with the bluffs removed (worst-hand raises → call, renormalised) | Phase 1 — **the price of honesty** | | 0 |
| tabular ES-MCCFR at Deep CFR's budget (nodes touched) | Phase 1 | | |
| Deep CFR, strategy net | Phase 2 | | |
| SD-CFR: the exact average of the stored advantage nets, no strategy net | Phase 3 | | |
| CFR+, 1,000 iterations | 0.00025 | 0 | 19.5% (J/Q) |

- **Head-to-head** is the value an ordinary opponent would see; exploitability is the worst
  case. A policy can be close to Nash in one and far in the other, which is why both are
  columns.
- **SD-CFR** (Steinberger 2019) is affordable here because the average strategy over T stored
  nets is computed exactly from T batched forwards over the whole game, weighted by t and
  by each net's own reach. It separates the advantage nets' error from the strategy net's.
- **Non-uniqueness.** Leduc's equilibria are not unique, so a bluff frequency compared with
  one solver's is not a fact about the game. Phase 1 runs CFR+ and DCFR from two inits each
  to exploitability < 1e-3 and reports the worst-hand raise frequency as a range. If the
  range is wide, the column shows the range and the section says the frequency is free.
  The price-of-honesty row is computed from each exact solver; if those disagree, both go in.

## 5. The scale arm: matched budget, r = 3 / 6 / 13

The obvious objection is blackjack's: at r = 3 a table is enough, and tabular CFR+ reaches
2.5e-4 in 71 s of Python. A table is enough at every r here (1.18M nodes at r = 13 is
nothing to C). So the comparison is not network against table at convergence. It is the
Deep CFR paper's own axis: **exploitability against nodes touched**, sampled table
(ES-MCCFR) against network, at r = 3, 6, 13. A sampled table leaves the information sets it
never visited at uniform; the network has to say something about them, and the rank scalars
let it. The prediction is that the table wins at r = 3 and the network closes the gap or
passes it as r grows at a small budget. Report coverage (fraction of information sets
visited at the budget) beside each point, as tic-tac-toe reports that self-play visits 1.2%
of the 4×4 positions.

If the prediction fails, the section says the sampled table wins at every size, and that the
network's job in this section is imperfect-information play, not scale.

## 5a. Figure and section

Figure (rule: first panel = the object itself):
- (a) One hand: the six cards, the deal, round 2's betting tree from P0's seat holding J
  under a Q, with the trained net's probability on each edge and the bluff edge marked.
- (b) Exploitability against nodes touched, log–log: ES-MCCFR, Deep CFR, SD-CFR, the CFR+
  line, uniform dashed; r = 3 solid, r = 13 faded.
- (c) Raise probability per (private, public) at round 2's first decision: the net beside
  CFR+, worst-hand cells ringed.

Section `\subsection{Poker --- Deep CFR against the exact best response}` between Pong and
tic-tac-toe, one table (§4), the scale result as a paragraph, "what this is not" at the end:
a two-player, fixed-limit, one-card game; no-limit and multi-way poker are another scale
(Pluribus), and exploitability is not computable there.

Bestiary entries: Deep CFR (the net is a few dense layers; the paper's card-embedding body
can be spelled in `Bestiary/DeepCFR.lean` at 0 new primitives), Player of Games / ReBeL in
prose.

## 6. Phases

```
Phase 0 (CPU, 1 session):  §1 C instrument + LeanMlir/Leduc.lean + `lake exe leduc-env`;
                            Gate 0 (§7). OpenSpiel in its own venv (.venv-poker, pinned,
                            never the main .venv).
Phase 1 (CPU, ½ session):  scripted arm, price of honesty, CFR+/DCFR non-uniqueness range,
                            ES-MCCFR curves at r = 3, 6, 13.
Phase 2 (GPU, 1 session):  §3 trainer at r = 3; Gates A and B; tune §3's numbers; one seed.
Phase 3 (GPU, ½ session):  three seeds; SD-CFR; the r = 6 / 13 arms at matched budgets.
Phase 4 (½ session):       figure + section + bestiary entries + demos/README.md section.
Phase 5 (½ session):       §8, after the section is in and the user has read it.
```

Expected cost: under an hour of GPU for r = 3 (blackjack's 200k updates took ten minutes;
here T × 2 × 1,000 = 200k advantage steps plus the strategy net). The r = 13 arm is longer;
ask before any run over ~30 min.

## 7. Gates that fail loudly

- **Gate 0 — the instrument is the game.** Against OpenSpiel's `leduc_poker`: 936
  information sets; `exploitability` of the uniform policy and of a saved CFR+ table equal
  to ours to 1e-6; game value −0.0856 to 1e-3. Every terminal payoff of the Lean `step` /
  `payoff` equals the C tree's (one exhaustive walk). Best response checked by brute force
  over all pure strategies of one seat at a reduced r, if cheap; otherwise against OpenSpiel
  only.
- **Gate A — the loss trick is exact.** On one batch, the train step's parameter gradient
  equals the host-computed weighted-MSE gradient to float tolerance (the NQS and tic-tac-toe
  demos ran the same check).
- **Gate B — the traversal is right.** With the advantage net replaced by a lookup table
  (σ from accumulated sampled regrets), the trainer's own traversal reproduces ES-MCCFR's
  exploitability curve. Separates sampling bugs from function-approximation error before
  any network trains.
- **Gate C — it learns.** Deep CFR's exploitability falls monotonically (up to seed noise)
  across iterations and ends below the scripted arm's. The bar against the published Leduc
  curves is read from the Deep CFR and SD-CFR papers' figures at Phase 2, not quoted from
  memory.

## 8. Removing blackjack (Phase 5, after the poker section is in)

Delete, not archive (`prune_rule_toc_canonical`): `demos/MainBlackjackDqn.lean`,
`demos/MainBlackjackEnv.lean`, `LeanMlir/Blackjack.lean`, the `blackjack-env` and
`blackjack-dqn` targets (`lakefile.lean` ~:717–730), `scripts/demos/blackjack_figure.py`,
the figure `demos/blackjack_chart.png`, the README section; move
`planning/blackjack_dqn_demo.md` to `planning/archive/`. `runs/2026-09-11-blackjack-dqn/`
stays as the record.

Text that cites it (`content.tex` line numbers as of 2026-10-05):
- :330 "three games — blackjack, Pong, tic-tac-toe" and :17017 "blackjack's" ceiling.
- :17140 Pong "exactly as in blackjack": **the Bellman-target-into-the-squared-error-block
  explanation moves into Pong**, since blackjack is where it was introduced.
- :17210, :17222 tic-tac-toe: "the block blackjack's Bellman target went through",
  "Blackjack and Pong made the same choice".
- :16542 GW "the way blackjack has value iteration"; :16757 NQS (being replaced by CASP).
- Docstrings: `demos/MainPongDqn.lean:9`, `MainGwDetect.lean:11`, `MainRsBands.lean:16`,
  `MainNqsIsing.lean:23`, `LeanMlir/FloatFmt.lean:2`; `LeanMlir/README.md`, `README.md`,
  `CHANGELOG.md` (history, leave it).
- Gates: `check_target_names.sh`, `gen_mlir_manifest --check`, the blueprint `\uses`
  regeneration, `docstring-checkrefs`.

## 9. Out of scope

No-limit, multi-way, or any game without an exact best response; ReBeL / Player of Games'
search at test time (prose in the bestiary); opponent modelling (*Bayes' Bluff*'s own
topic); a verified-render tier for the dense stack.

## 10. Open questions

- The worst-hand definition for the bluff column: "loses to every holding a showdown can
  produce except a tie" (J/Q, J/K at r = 3; J in round 1). Check it reads sensibly at r = 13
  before the scale arm reports it.
- Whether the strategy net or SD-CFR is the demo's headline row. Decide on the Phase 3
  numbers; SD-CFR is the better algorithm on paper and the cheaper one here.

## 11. Work log

- 2026-10-05: plan opened. Python prototype: 936 / 288 information sets, CFR+ 2.5e-4 at
  1,000 iterations, game value −0.0856, uniform 2.3736, worst-hand raise 19.5% (J/Q). First
  prototype bug worth keeping: summing instead of averaging over the public card stalls CFR
  at 0.33 exploitability with nothing else looking wrong — Gate 0's value check catches it.
