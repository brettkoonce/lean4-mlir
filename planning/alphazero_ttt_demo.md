# alphazero_ttt_demo.md — AlphaZero on (n×n, k-in-a-row) tic-tac-toe, scored against the solved game

Goal: the RL subsection's third demo, after blackjack (rung 2) and Pong (rung 3): the
self-play + tree-search loop the subsection's closing sentence currently only promises,
on a game written in Lean, with the bestiary's own `tinyAlphaZero` body trained rather
than read, and every arm scored against the solved game the way blackjack's arms are
scored against value iteration. The board size is the one knob: 3×3 first, then 4×4 by
re-running the same binary, which is the claim the section measures.

The user's 2018 talk (brettkoonce.com/talks/solving-go-with-alpha-go-and-alpha-zero)
ran Surag Nair's alpha-zero-general through Evgeny Tyurin's tic-tac-toe port: 3×3 in
minutes, 4×4 estimated at "20–24 hours" in that code. That estimate is the bracket row.

## 0. Why this game

- Solved, cheaply and completely. A position is a base-3 number over the cells, so a
  memoised minimax fills one byte per index — 19,683 at 3×3, 43 MB at 4×4 — in seconds
  of C, giving the exact value AND the optimal-move set of every reachable position.
  That is blackjack's DP for this game: agreement over all reachable decision
  positions, a perfect opponent with random tie-breaking, and the value head scored
  against the true minimax value.
- Two boards from one binary. n is the only knob; the solver, the plane builder, the
  symmetry augmentation, MCTS and the instrument are all parametric in it.
- The bestiary already spells the net (`Bestiary/AlphaZero.lean`, `tinyAlphaZero*`).

## 1. The game and the instrument — `LeanMlir/TicTacToe.lean`, C in `ffi/f32_helpers.c`

- `Pos`: n, k, n² cells in {0, X, O}, stone count, last cell played. X moves first; the
  mover is the stone count's parity. Win check through the last cell only.
- Table entry from the MOVER's view: bits 0–1 value (0 loss, 1 draw, 2 win), bit 2
  terminal, 255 unreached. `lean_ttt_solve` walks every legal line (no pruning, so
  "reachable" means by any play, not optimal play). `lean_ttt_reachable` lists the
  indices, `lean_ttt_planes` builds the canonical `[2, n, n]` planes (mine, theirs)
  for an index list, `lean_ttt_score` argmax-checks a logits block against the
  optimal sets and the value head against the exact value — all in C, so the 4×4
  sweep over millions of positions costs seconds.
- Players: random; win-or-block heuristic; perfect (uniform over the optimal set).
- `lake exe ttt-env [n=3] [k=3] [games=1000]`: the counts, the empty board's value,
  W/D/L of every scripted pairing, and the gates of §8.

## 2. The network — AlphaGo's plain conv stack, one merged head

```
.conv2d 2 64 3 .same .relu, .conv2d 64 64 3 .same .relu, .conv2d 64 64 3 .same .relu,
.conv2d 64 4 1 .same .relu, .flatten, .dense (4·n²) 64 .relu, .dense 64 (n² + 1) .identity
```

`imageH := imageW := n`. Output slots 0..n²−1 are policy logits, slot n² the value
pre-tanh. NetSpec is a linear list, so the two heads become one dense head over the
shared 1×1 features; tanh is applied on the host, as the bestiary entry's own note says.

Not the bestiary's conv-BN-residual `tinyAlphaZero` body, which was the first choice:
that run diverged at iteration 11 of 20 (loss 1.26 → 2.63 → 3.76 → 10.8, sweep
agreement 96.8% → 56%, `runs/2026-09-29-alphazero-ttt/n3_convbn_collapse.txt`) after
being unbeaten from iteration 4. BatchNorm's batch statistics over nine binary cells
are the suspect — a channel with a tiny batch variance is divided by √1e-5 — and, with
BN, the eval forward the loss trick reads (running stats) and the train step's forward
(batch stats) are different functions, so the delivered cotangent is off by their
disagreement. Blackjack and Pong made the same call for the same reason. Without BN
the two forwards coincide and the trick is exact, as in the NQS demo.

## 3. The loss — the two AlphaZero terms through the MSE block, zero codegen

`(z − v)² − πᵀ log p`, delivered as the DDPM MSE block's target the way
`MainNqsIsing.lean` does it: with output cotangent g, `y = out − nOut·g/2` makes the
block's gradient g/B, the gradient of the batch-mean loss. g = softmax(p) − π_mcts on
the policy slots (π is zero on illegal cells, so those logits are pushed down, as in the
paper); g = (tanh v − z)·(1 − tanh² v) on the value slot. The printed loss is
meaningless, as in blackjack; the section says so. Every forward is the eval graph,
held on the device with a generation token (Pong's pattern); the train step's own
forward, whose parameters move every step, takes the copying path so the held one is
re-seeded once per iteration.

## 4. The loop — alpha-zero-general's, in lockstep

- Self-play: G games in parallel. Each move: S simulations of PUCT (c_puct 1.5),
  Dirichlet noise at the root, one batched forward over every game's pending leaf per
  simulation step, terminal leaves scored exactly. τ = 1 for the first t plies, then
  greedy. Record (s, π_mcts, z) with z from the mover's view.
- Augmentation: the eight dihedral views via `F32.dihedralGather` on the stacked
  [planes ‖ π] tensor, so one transform hits board and target alike.
- Replay: the last W iterations; E epochs of Adam at batch B per iteration.
- Instrument every iteration: (a) one batched sweep over the reachable decision
  positions — agreement of the masked argmax with the optimal set, value MSE and sign
  agreement; (b) M games vs perfect as X and as O, net alone and net + MCTS; (c) the
  value the net gives the empty board.
- 3×3 defaults from alpha-zero-general (25 sims, 100 games, ~20 iterations); 4×4 raises
  sims and games. Every default is a knob.

## 5. The one table

Rows: random, win-or-block, net alone, net + MCTS, perfect (and the talk's 4×4 estimate
as a note). Columns per board: agreement over reachable decision positions, W/D/L vs
perfect as X and as O, vs random, iterations, wall clock.

## 6. The one figure

Left: a board with the net's policy as a heatmap over the empty cells at a position a
reader can check (X centre, O to move), the optimal moves ringed as in the blackjack
chart. Middle: agreement-vs-iteration, 3×3 and 4×4 on one axis. Right: value head vs
exact value over the reachable positions.

## 7. Phases

```
Phase 0 (no GPU):  game, solver, players, ttt-env, lakefile row.
                   Gate: perfect vs perfect all draws (if the empty board is a draw);
                   perfect never loses to anyone as either side.
Phase 1:           demos/MainAlphaZeroTtt.lean; smoke convBn/residualBlock at imageH = 3.
                   Gate A: 3×3 all draws vs perfect both sides; net-alone agreement high.
Phase 2:           n=4, no code change. Gate B: never loses to perfect as either side.
Phase 3:           figure, section, README, bestiary cross-link.
```

## 8. Gates that fail loudly

- The solver is checked by play before anything trains: perfect vs perfect draws 100%
  when the root is a draw, and perfect loses to nobody. A table that fails either is not
  the game.
- The net-alone row is scored over ALL reachable decision positions, not the self-play
  distribution. The expected finding at 4×4 is a lower number there with an unbeaten
  match record: most positions never occur under competent play.
- The batched-forward shapes are checked at startup: a `[B, 2, n, n]` input and an
  `n² + 1` output, or the run stops.

## 9. Out of scope

Go, Othello, Connect-Four; MuZero's learned model; a boards-larger-than-4 solver
(3^25 entries is 847 GB); an arena acceptance step (the instrument replaces it).

## 10. Log

- 2026-09-29, Phase 0: `lake exe ttt-env` — 3×3: 5,478 reachable positions (the classic
  count), 4,520 decisions, table in 0 ms; 4×4: 9,722,011 reachable, 9,062,619 decisions,
  43 MB in 603 ms. Both roots draw. Perfect vs perfect 1000/1000 draws both ways; random
  and win-or-block never win a game against perfect; win-or-block loses 169/1000 as X and
  785/1000 as O at 3×3, 70 and 160 at 4×4. "X centre, O to move" reads corners 0, edges
  −1. Gates pass on both boards.
- Phase 1, first run (bestiary conv-BN body, 20 iterations, 25 sims, 256 games): unbeaten
  by perfect from iteration 4 (MCTS) / 6 (net alone), sweep 96.8% at iteration 10, then
  divergence — loss 1.26 → 2.63 → 3.76 → 10.8, sweep 56%, root value saturating at ±1.
  `runs/2026-09-29-alphazero-ttt/n3_convbn_collapse.txt`. Net changed to AlphaGo's plain
  conv stack (§2); the cotangent delivered as the batch-mean gradient (§3).
- Phase 1, plain-conv run (`n3_run2.log`; run1 identical to every printed digit): 20
  iterations in 286 s on one 4060 Ti; sweep 57.9 → 87.2 (iter 1) → 95.4 (4) → 97.9%
  (20); value sign agreement 92.2%, MSE 0.093; net alone unbeaten from iteration 6, net +
  MCTS from 4; the root valued +0.26 for X (self-play's X-bias, not the theorem's 0).
  Self-play stood at 2,010 of the 4,520 decision positions (44.5%). The 95 misses: 0 at
  lost positions, 29 at drawn, 66 at won (missed forced wins), concentrated at 3–6
  stones; 33 among visited positions, 62 among never-visited — the gap is not purely
  off-distribution. Gate A passes.
- Bugs found on the way: the match loop took the side to move from game 0's position,
  which lies once game 0 has finished (net "won" 8 games against perfect); the held
  forward re-seeded (and printed its banner) every training step, so the train step's
  forward moved to a copying-path session; the curve CSV carried the iteration twice, so
  the figure read `agree` one column off (repaired in the trainer and in the written file).
- 4×4 timing probe (1 iteration, 100 sims, 256 games): self-play 17.7 s, matches 18.3 s,
  the full 9.06M-position sweep 14.2 s; after one iteration 252/256 draws as X with MCTS.
- Phase 2 launched 22:38: `n=4 iters=40 sims=100 sweep=50000 epochs=5`, card 1.
- Phase 2 (`n4_run1.log`): 40 iterations in 2,091 s (34.9 min) on one 4060 Ti, the same
  binary. Sweep (50k subsample) 60.3 → 88.8 (iter 3) → 97.0 (7) → 99.1% (40); the full
  9,062,619-position sweep at the end: **99.20%** agree, value MSE 0.054, sign 94.3%. Net +
  MCTS unbeaten from iteration 4, net alone from 38. Self-play stood at 107,153 decision
  positions (1.2%). Misses in the subsample: 444 (0.89%), none among the 568 visited, 411
  missed forced wins + 33 losing moves, at 7–14 stones. Root value +0.03. Gate B passes.
- Bracket rows (`ttt-env`, `lean_ttt_scripted_agreement`): expected agreement random 57.97 /
  60.41%, win-or-block 93.95 / 95.61% — the untrained net's iteration-0 sweep (57.90 /
  60.26%) is the random row.
- Phase 3: figure `demos/figures/alphazero_ttt.png` (+ the blueprint copy), run README,
  demos README section, book paragraph + table in the RL subsection (retitled), bestiary
  cross-links, tree entries, `certs.yml` entry-point count 111 → 113.
- Phase 4, the search in C (`lean_mcts_*`: flat per-game arrays in one Lean-owned arena —
  nodes, a position-index hash, the pending path; select / expand-backup / root noise /
  root visits; the lockstep stays in Lean): same configs, `n3_run3_ctree.log` 151.7 s
  (was 285.9), 97.30%, unbeaten alone from 11 / MCTS from 5; `n4_run2_ctree.log` 598.3 s
  (was 2,091.5), full sweep 99.42% / MSE 0.037 / sign 95.6%, unbeaten alone from 10 /
  MCTS from 4, self-play stood at 108,483 positions. Per 4×4 iteration: self-play 18.5 →
  0.7 s, matches 17.8 → 0.8 s, training 18–20 s at the full window (the remainder).
  The Lean-tree runs stay in the run directory as the before-row.
