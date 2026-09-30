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

## 11. 5×5 — the instrument without the table

The dense table ends at 4×4 (3²⁵ bytes = 847 GB). Three things replace it, and each
is gated against the table where the table exists:

- **An on-demand exact solver** (`lean_ttt_solver_*`): negamax to the end of the game
  with a transposition table keyed by the symmetry-canonical index (min over the eight
  dihedral views), exact values only — the sole cut-offs are the ones that keep the
  value exact: an immediate win is +1 without recursion, two opponent threats are −1,
  one threat forces the block, and the child loop stops at the first win. Centre-out
  move order. The arena is a Lean ByteArray (keys u64 + values u8, open addressing,
  capacity a knob), so a solved cache is saved and loaded with plain file IO. `Table`
  dispatches: the dense table at n ≤ 4, the solver above — and `ttt-env` checks the
  solver against the table on every reachable 3×3 and 4×4 position before anything
  trusts it at 5×5.
- **A sampled sweep**: `sweep` positions drawn by random play (fixed seed), each solved
  exactly; the scorer and the scripted players' expected agreement run over that set.
  "Agree" at 5×5 is therefore exact per position but not exhaustive, and the table
  says so. Self-play coverage moves from a byte per index to a hash set.
- **64-bit indices** through the plane builder, the scorers, the search's leaf records
  and the Lean pushes (`pushU64LE` / `readU64`), since 3²⁵ > 2³².

Board: (5, 5, 4), the interesting one (the literature says draw; the solver decides —
the root's solve time is the unknown, cached to disk once found). (5, 5, 5) is a
trivial draw. The MCTS arena's node budget n²·sims a game holds.

Phases: 5a u64 + solver + the cross-check gate + the (5,5,4) root timed; 5b the
sampled sweep and the trainer's n ≥ 5 path; 5c a one-iteration timing probe, then the
run (ask first).
- 2026-09-30, §11 5a: u64 index lists end to end; `Table` dispatches dense / solver;
  `lean_ttt_solver_check` gates the solver against the table on every reachable position.
  Solver v1 (plain negamax, first-win cut, memo, no pruning) passed the gate — 3×3 0
  mismatches, 4×4 0 mismatches in 3.0 s (1.14M canonical entries) — but the (5,5,4) root
  did not finish in 12 min. v2 (alpha-beta with bounded entries) first failed the gate:
  40 mismatches at 3×3, 16,512 at 4×4, from one line that promoted an upper bound to
  exact on a forced block; removed, 0 mismatches. Its root solve ran 51 min without
  finishing: the transposition table stopped caching at 80% of 2²⁴ slots, so past ~13M
  entries the search ran memo-less (calibration was misleading too — the 2-stone position
  timed at 11 s was an X win, cheap to prove; a draw at the root is the expensive proof).
  v3: replace-always caching with 8-slot probe runs, u16 entries carrying the best move,
  winning lines as bitmasks (one popcount a line for "immediate win?" and "threats?"),
  move order = table move, killers, then lines the mover owns unopposed (threat-makers
  first), centre-out on ties. Gate: 0 mismatches on both boards (4×4 in 16.8 s, 80M
  nodes — bound re-searches cost more than the plain memo on an exhaustive check, which
  is not the workload). Timing the root with 2²⁷ slots (`n5_solver_probe2.log`).
- v3's null-window search from the root ran 16 min without returning on (5,5,4) while the
  25 openings solved one at a time with full windows take 3.5 min in all (centre 2.7 s
  with four drawing replies — the diagonal neighbours; mid-adjacent 12 s, four; inner
  corner 6.8 s, ONE drawing reply; edge-middle 51 s, seventeen; edge-off-middle 61 s,
  thirteen; corner 76 s, all twenty-four): every opening draws, so (5,5,4) is a draw and
  every first move keeps it. The empty board's entry is therefore the max over its
  children, each solved full-window and cached (`ttt_solver_entry`), and the perfect
  player's opening set is exact; the centre-only fallback that was wired for an hour is
  gone. Gate still 0 mismatches on both boards.
- The hour of "root won't solve" was a log artefact: `ttt-env` printed with `IO.println`,
  which is block-buffered under redirection, so the log stayed empty while the process was
  past the root and inside the 20k-position sampled agreement. With flushing prints: the
  (5,5,4) root (max over its 25 full-window-solved openings) is **33.2 s**, 43.7M entries,
  99M nodes, a draw; the cache is 1.3 GB at 2²⁷. The sampled agreement is the cost: with
  every child valued, 2-stone positions cost seconds each (a fresh 3-stone subtree), so the
  sweep sample now starts at six stones (`sweepMin` / `sampleMin`, default 6): 2,000
  positions with all children in 65 s (32 ms each); the trainer's scorer needs two solves a
  position. Baselines at ≥ 6 stones: random 43.1%, win-or-block 75.4%. Runs use 2²⁸ slots
  (the 2²⁷ table reached 105M of 134M).
- The match play was the next wall: the perfect player valued EVERY child exactly to draw
  uniformly from the optimal set, and at 5×5 a fresh child is a full solve — the trainer's
  iteration 0 and `ttt-env`'s random-vs-perfect both sat 50 min without a line. Now it
  shuffles the legal moves and takes the first whose value matches the position's — the
  first optimal move of a uniform permutation is a uniform draw from the optimal set, so
  the distribution is unchanged and the cost is about two child solves a move. Gates
  unchanged on both boards.
- Solver v4: entries carry both bounds (lo, hi; merged on store, exact when equal), which
  ends the ≥1 / ≤1 re-search churn — the exhaustive 4×4 check drops 80M → 13.3M nodes
  (16.8 → 3.8 s) — and the principal move in canonical coordinates. The perfect player
  takes that move from the third stone on (one solve, the position's) and keeps the
  uniform draw over the optimal set for each side's first move, where the opening book
  makes it free. A second gate checks the principal move against the table's optimal set
  on every decision position: it caught two cases that stored no move — lost positions
  (no child ever beats 0) and the double-threat shortcut — both now store a legal move
  (any move is optimal in a lost position). 0 / 0 on both boards.
- 5×5 probe (`n5_probe.log`, 200 sims, 256 games, 20k sweep at ≥ 6 stones): iteration 0
  ~4 min (first-time solves); iteration 1 self-play 2.3 s, train 0.95 s, score 26 s; the
  untrained net loses all 512 games and self-play is decisive (X 149 / draw 15 / O 92 —
  (5,5,4) punishes bad play). Root 36.6 s in the same table. Run launched:
  `n=5 k=4 iters=60 sims=200 sweep=20000 sweepMin=6 epochs=5 cap=28 tag=run1`, card 1.
- `ttt-env n=5 k=4 games=200 samples=20000 sampleMin=6 cap=28` (`n5_env.log`, 12.7 min,
  2.6 GB RSS): root a draw in 36.6 s; expected agreement over the 20k sample (≥ 6 stones)
  random 45.04%, win-or-block 77.03% (6.9 min); random vs perfect 0/0/200 both ways (perfect
  wins every game — (5,5,4) punishes bad play), win-or-block vs perfect 0/7/193 as X and
  0/1/199 as O, perfect vs perfect 200/200 draws both ways; random vs random X 113/10/77.
  The 2²⁸ table saturated (268.4M of 268.4M slots, replace-always from there); cache file
  2.56 GB. cap=29 (5 GB) if a run's table thrashes.
- 5×5 run1 (`n5_run1.log`, 60 × 200 sims, 25.4 min): sample 42.4 → 85.9%, sign 64.7%; net +
  search unbeaten as X from iteration 5, 243/13 as O at 60; net alone 198/58 X, 256/0 O at 60
  — the O side not closed at 200 sims. Misses 14.1%: 2,070 missed forced wins + 758 losing
  moves, flat over 6–23 stones; self-play stood at 3 of the 20k sample. Opening policy: 76%
  centre from the empty board; 24/24/24/23% on the four diagonal replies after X centre —
  the solver's optimal set. run2 (100 × 400 sims, card 0) launched as the quoted row: at
  iteration 34 unbeaten both sides alone and with search, 86.0%.
- 5×5 run2 (`n5_run2.log`, 100 × 400 sims, 63.8 min): sample 87.38%, MSE 0.334, sign
  70.7%; net + search unbeaten from 59 to 100; net alone clean for 24 iterations from 68,
  237/19 as O at 100; root +0.04; misses 12.6% (1,849 missed wins, 676 losing moves);
  self-play 304,169 distinct positions, 4 in the sample; opening 26% centre, 27/24/23/22%
  on the four diagonals. Table, README, book, figure (stacked 3×3 / 5×5 boards, three
  curves) updated. Phase 5 done.
- Book: §10.4 Player of games first rendered as ".1 Game theory" — the section had been inserted
  after the `\appendix` switch (and the TOC-depth restores) that precede the Data-availability
  chapter, so the appendix counter printed blank. The three lines now sit after the MuZero
  entry, just before `\chapter{Data availability}`; preview: 10.4 / 10.4.1–3 / Appendix A.
- Independent evaluations (`alphazero-ttt … iters=0 params=<file>`, a new knob: with no
  iterations the run is the instrument alone on a saved net): 3×3 and 4×4 nets, seeds 2–4,
  three readings each — every cell 0/256/0, alone and with search, both sides (768 games per
  cell); the sweeps reproduce the runs' last readings to the digit (97.30 / 99.42). The 5×5
  net on the exhaustive opening (`opening=3`: 7,526 decision positions with ≤ 3 stones,
  two solves each): **92.11%**, value sign 82.8% — five points above the ≥ 6-stone sample's
  87.38% — and on a fresh evaluation alone X 256/0 · O 224/32, with search X 256/0 · O
  241/15: the run's own last reading (256/0 with search) was one draw of the perfect
  player's random lines; three more seeds pooled for the table.
- The scripted players' expected agreement over the exhaustive opening needs every child of
  every position (a fresh 4-stone subtree each, ~0.5 s): `ttt-env … opening=3` ran 40 min
  and was stopped; `opening=4` (76k positions) is out of reach. The opening line is the
  net's alone; the baselines stay on the ≥ 6-stone sample.
- Pooled 5×5 evaluations (seeds 2–4, 768 games a cell): alone X 0/768/0 · O 0/708/60,
  with search X 0/768/0 · O 0/745/23. Table, book, README, run README carry these; the
  "unbeaten from 59" sentence is now the run's reading against the evaluations' truth.
