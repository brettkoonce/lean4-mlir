# blackjack_dqn_demo.md — tabular Q-learning to DQN on blackjack

Status 2026-09-11: Phases 0–2 done; Phase 3's figure and content.tex section
done; `Bestiary/DQN.lean` and the Gymnasium cross-check (§1) are open. Numbers and files in `runs/2026-09-11-blackjack-dqn/README.md`:
optimum -0.0431, tabular Q -0.0440 (192/200, 10^6 hands), Double DQN -0.0476
(188/200, 200k updates) — Gate 2 passes for all four ablation arms. The
environment moved to `LeanMlir/Blackjack.lean` so both exes share it; the DQN is
`lake exe blackjack-dqn`. The book's figure is the two charts alone,
`blueprint/src/figures/demos/blackjack_chart.png` (`scripts/blackjack_figure.py … chart`);
the curve version stays in the run directory.

Goal: the reinforcement-learning demo, in two rungs on one loop. Rung 1 ports
the old Swift/gym blackjack demo to pure Lean: the environment, tabular
Q-learning, and its three comparison arms. Rung 2 replaces the Q table with
an MLP trained through the stack, which is DQN, on the existing DDPM MSE
block with zero new codegen. Both rungs are scored by exact dynamic
programming over the game's 200 decision states, so the ceiling is a theorem
about the rules rather than a number somebody measured.

The old demo already had the book's instrument ethic: random was the floor,
the published table the ceiling, Q-learning scored between. This keeps that
four-arm structure and adds the exact optimum above the published table.

## 0. The one-paragraph version

Blackjack under the Sutton & Barto Example 5.1 rules is a 200-state, two-
action MDP with known transition probabilities, so its optimal policy and
the exact value of *any* policy are a value iteration that runs in
milliseconds. That makes every arm exactly scorable: random, a threshold
heuristic, the published casino-rules table, tabular Q, DQN, and the optimum.
The DQN is a three-layer `.dense` net whose loss is the rank-2 DDPM MSE
block: the host builds the Bellman target and copies the net's own prediction
into the untaken action's slot so its gradient is zero. The figure is the
classic hit/stand chart, learned versus exact, cell by cell, and a training
curve whose ceiling is the exact optimum.

## 1. Rules — Gymnasium's Blackjack-v1 with `sab=True`

The de-facto specification is Gymnasium's `Blackjack-v1`, and its
`sab=True` mode is Sutton & Barto exactly:

- Infinite deck: cards drawn with replacement, values 1–10, ten with
  probability 4/13 and each other value 1/13.
- An ace counts 11 when that does not bust ("usable"), otherwise 1.
- Dealer shows one card; the hole card is an independent draw. Dealer hits
  below 17 and stands on 17 or more (soft 17 included).
- Player acts first: hit or stick. Bust is -1. After sticking, dealer plays
  out; +1 / -1 / 0 by comparison. A natural wins +1 unless the dealer also has
  one (0). No doubling, splitting or insurance.
- Observation: (player sum 4–21, dealer showing 1–10, usable ace 0/1) — the
  triple the Swift code unpacked.

Implemented as `Blackjack.lean` inside the demo: `Card`, `Hand`, `State`,
`Obs`, `draw : StdGen → Card × StdGen`, `reset : StdGen → State × Obs ×
StdGen`, `step : State → Action → StdGen → (Obs × Float × Bool × StdGen)`.
Pure functions with the RNG threaded through, so an episode replays from a
seed and the tests are deterministic.

Cross-check against Gymnasium (one-off, throwaway venv, never the pinned
one): 10^6 hands under the published table in both environments; the mean
rewards must agree within three standard errors (σ/√n ≈ 0.001).

## 2. The exact instrument — `Blackjack.DP`

Decision states are (sum 12–21) × (dealer 1–10) × (usable ace), 200 of them;
below 12 hitting cannot bust and is always taken. Transitions are exact:
hit draws a card from the fixed distribution and updates (sum, ace); stick
ends the hand with the dealer's outcome distribution conditional on the
showing card, computed once by recursion over the dealer's draws.

- `optimalPolicy` and `V*` by backward induction: ace = false states from
  high sum down, then ace = true states (a bust with a usable ace becomes a
  hard hand at sum - 10, so the order is acyclic).
- `evaluate (π : Obs → Action) : Float` — the exact mean reward per hand of
  any policy, including the initial deal and naturals. This is the column
  every arm is scored on.
- `agreement (π) : Nat` — decision states where π matches the optimum, of
  200.

Gate 0: the DP's policy must match the Monte Carlo ES chart in Sutton &
Barto chapter 5 (Figure 5.2 in the second edition) cell for cell. Those are
the same rules; any difference is a bug in §1.

## 3. Rung 1 — tabular Q and the four arms, pure Lean

Arms, all `Obs → Action` after training:

1. random — the floor.
2. threshold heuristic — the old `markovStrategy`: hit with probability 0.8
   below 18, stick with probability 0.8 at 18 and above.
3. published table — the old `normalStrategyLookup` strings verbatim. ⚠ It
   is a casino-rules table (doubling, splitting, dealer rules differ), so it
   will score below the exact optimum for these rules. That is a row, not a
   bug: "the published table is for a different game".
4. tabular Q — Q[32][11][2][2] as before, ε-greedy, γ = 1 (the hand is short
   and undiscounted; the old γ = 0.2 discounted the terminal reward through
   each hit), 10^6 hands. Step size max(0.001, 1/(1+N)) with N the visit
   count: a constant α stalls at 177/200 because the bootstrap target keeps
   moving; the visit-count average reaches 192/200 at 10^6 hands.
   ⛔ The old update was `Q += (1-α)·Q + α·target`, i.e. `Q ← (2-α)Q + α·target`,
   which grows geometrically at α = 0.5. The port writes `Q ← (1-α)Q + α·target`.
5. the DP optimum — the ceiling.

Each arm reports the exact value from §2 and a Monte Carlo estimate over
10^6 hands with its standard error. The two must agree; that is the check
that the environment and the DP describe the same game.

## 4. Rung 2 — DQN through the stack, zero new codegen

**Net.** `.dense 29 64 .relu, .dense 64 64 .relu, .dense 64 2 .identity` with
`imageH := imageW := 1`, the 2-D toy's shape. Input is a one-hot of the sum
(4–21, 18 slots), a one-hot of the dealer card (10) and the ace bit: 29
floats, so a linear function of the input can already represent any table
policy and the net has nothing to invent.

**Loss.** `generateTrainStep … (useDdpm := true) (ddpmOutShape := [B, 2, 1,
1])` and `LowererSession.trainStepAdamF32Ddpm`, exactly the 2-D toy's call.
Per step the host runs the eval graph at batch B on the replay batch to get
`Q(s, ·)`, builds `y = Q(s, ·)` and overwrites `y[a] = r + γ (1 - done)
max_a' Q_target(s', a')`, and hands `y` in as the target. The MSE gradient is
then zero on the untaken slot and `(Q(s,a) - y[a]) / B` on the taken one: a
per-sample TD loss. If the extra forward per step is ever the cost, a
ten-line masked-MSE branch in the loss block replaces it; not needed at this
scale.

**Loop.** Replay ring of 20k transitions (s, a, r, s', done) as flat f32
arrays; batch 128; Adam lr 1e-3; ε from 1.0 to 0.05 over the first 20k hands
then held; target parameters are a second `evalParams` buffer copied from
the online parameters every 500 steps; γ = 1; 50k gradient steps (hands are
one to three steps, so about 10^5 hands). Double DQN behind a flag: argmax
from the online net, value from the target.

**Instrument while training.** Every 1000 steps, read the greedy policy off
the online net over the 200 states with 200 batched forwards and call
`Blackjack.DP.evaluate` — the exact value of the current policy, for free.
That curve, with the optimum and the published table as horizontal lines, is
the second panel of the figure. Tabular Q gets the same curve from its table.

## 5. Figure and tables

**Figure.** Left: the hit/stand charts, usable ace and not, player 12–21
against dealer A–10, three columns (exact, tabular Q, DQN), cells coloured
hit/stick with disagreements outlined. Right: exact value of the greedy
policy against training hands for tabular Q and DQN, with the optimum and
the published table as lines. Produced by `scripts/blackjack_figure.py` from
a `.bin` dump of the three policies and the two curves.

**Table 1 (bracketed).** Rows: random, threshold, published table, tabular Q,
DQN, exact optimum. Columns: exact mean reward per hand, Monte Carlo over
10^6 hands ± s.e., agreement with the optimum of 200 states. The lead-in
sentence is that the ceiling is a theorem about the rules.

**Table 2 (the DQN's own).** ε schedule and target-update period ablation, or
Double vs plain, whichever moves the exact value; two or three rows, no more.

## 6. Section and bestiary

`blueprint/src/content.tex`: the RL subsection becomes "Reinforcement
learning — demo: DQN on blackjack", in the demo shape: three things that
change (there is no dataset, the target is the model's own prediction one
step later, and the instrument is a solved game), figure, Table 1 with
lead-in and caveat, Table 2. Bestiary: `Bestiary/DQN.lean` — the Nature DQN
(84×84×4, three strided convs, dense 512, actions; 1.7M) with Double, Dueling
and C51 as prose variants. ⚠ `.conv2d` has no stride argument and `.convBn`
does, so the entry either carries BN or notes the approximation the DCGAN
entry makes for transposed convs. Demo-datasets table: no row, the data is
self-generated; say so in the text.

## 7. Phases

```
Phase 0 (½ session, no GPU):  §1 env + §2 DP + Gate 0 (S&B chart) + Gymnasium cross-check
Phase 1 (½ session, no GPU):  §3 arms + tabular Q; Gate 1: tabular Q within 0.01 of the
                               optimum's exact value and ≥ 190/200 agreement
Phase 2 (1 session):           §4 DQN trainer on the DDPM path; Gate 2: DQN matches
                               tabular Q's exact value within 0.01
                               DONE 2026-09-11: 50k updates -0.0578 (misses by 0.014);
                               200k updates plain -0.0503 / Double -0.0476 / +lr decay
                               -0.0530, -0.0503 — all pass; Double is Table 2's winner
                               at one seed, lr decay does not reduce the readout jitter
Phase 3 (½ session):           §5 figure + tables; §6 section + bestiary entry
```

Every run is seconds to minutes on one card; nothing needs asking about.

## 8. Gates that fail loudly

- Gate 0 is the whole foundation: a wrong dealer rule shifts every row by
  the same amount and nothing else would notice.
- The Monte Carlo and DP columns must agree for every arm. If they agree for
  the optimum and not for tabular Q, the policy readout is wrong, not the
  learner.
- The published-table row must sit below the optimum. If it does not, §1's
  rules have drifted toward the casino game.

## 9. Out of scope, for now

Rung 3, DQN from pixels on a Lean-written Pong, has its own plan:
`pong_dqn_demo.md` (the Ch 3 CNN as the Q-function, this doc's loop reused,
the game the only new code). AlphaZero-style self-play and policy-gradient
methods are out.
