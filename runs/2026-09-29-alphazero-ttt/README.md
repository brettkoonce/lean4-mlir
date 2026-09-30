# AlphaZero on tic-tac-toe, 2026-09-29 (planning/alphazero_ttt_demo.md Phases 0–2)

`lake exe alphazero-ttt n=3 tag=run3` (20 iterations, 25 sims, defaults) and
`lake exe alphazero-ttt n=4 iters=40 sims=100 sweep=50000 epochs=5 tag=run2`, XLA, one
RTX 4060 Ti each, with the search in C (`lean_mcts_*`); the same configs under the
first, Lean-side tree are `n3_run2.log` / `n4_run1.log`, the before-row below. 256 self-play games per iteration in lockstep, PUCT c = 1.5, Dir(1.0) at
0.25 on the root, τ = 1 for every ply, the last 20 iterations as the replay, Adam 1e-3,
batch 64, every sample under a random dihedral view. The net is AlphaGo's plain conv +
ReLU stack (three 3×3 convs at 64, a 1×1 at 4, dense 64, dense n² + 1), 78,350 params at
3×3 and 154,777 at 4×4. The loss `(z − v)² − πᵀ log p` goes through the rank-2 DDPM MSE
block as a host-built target (`lean_ttt_targets`); zero new codegen.

The instrument is the solved game (`lake exe ttt-env`): a minimax table over every
reachable position — 5,478 / 4,520 decisions at 3×3, 9,722,011 / 9,062,619 at 4×4 (43 MB,
0.6 s) — so every arm is scored exactly: the argmax of the net's masked logits against the
optimal set over every decision position ("agree"), the value head against the exact
value, and matches against a perfect player that draws uniformly from the optimal set.

## The table

Net rows: 256 games as X and 256 as O per reading; scripted rows: 1,000 each way
(`ttt-env`). "Agree" for the scripted players is their expected agreement over the same
positions.

| arm | 3×3 agree | vs perfect as X · as O (W/D/L) | 4×4 agree | vs perfect as X · as O |
|---|---|---|---|---|
| random | 58.0% | 0/207/793 · 0/33/967 | 60.4% | 0/487/513 · 0/341/659 |
| win-or-block | 94.0% | 0/831/169 · 0/215/785 | 95.6% | 0/930/70 · 0/840/160 |
| **net alone** | **97.30%** (4,520) | 0/256/0 · 0/256/0 from iteration 11 | **99.42%** (9,062,619) | 0/256/0 · 0/256/0 from iteration 10 |
| net + MCTS | — | 0/256/0 · 0/256/0 from iteration 5 | — | 0/256/0 · 0/256/0 from iteration 4 |
| perfect | 100% | 0/1000/0 · 0/1000/0 | 100% | 0/1000/0 · 0/1000/0 |
| untrained net (iteration 0) | 57.9% | 0/0/256 · 0/0/256 | 60.3% | 0/150/106 · 0/87/169 |

3×3: 20 iterations, 62,241 steps, **2.5 min**. 4×4: 40 iterations, 186,877 steps,
**10.0 min** (the talk's alpha-zero-general estimate for this board was 20–24 h). With the
tree in Lean (`n3_run2.log`, `n4_run1.log`, same configs, ~90 µs a descent) the same runs
took 4.8 and 34.9 min for 97.90% / 99.20%, unbeaten alone from 14 / 38: the search was
the wall clock — 18.5 s of self-play and 17.8 s of matches per 4×4 iteration, now 0.7 and
0.8 s — and training is what is left (18–20 s of every 4×4 iteration at the full window).
Value head: sign agreement 90.3% / MSE 0.100 at 3×3, 95.6% / 0.037 at 4×4 over every
decision position. The root's value at the end: +0.32 at 3×3, +0.03 at 4×4 (the theorem
says 0 for both; self-play at 3×3 stays X-favoured, 99 X wins / 141 draws / 16 O wins in
the last iteration, while 4×4 self-play draws 242 of 256).

## Where the misses are (`scripts/demos/ttt_sweep_stats.py`)

- Self-play stood at **2,035 of the 4,520** 3×3 decision positions (45.0%) and at
  **108,483 of the 9,062,619** 4×4 ones (1.2%); the sweep scores every position.
- 3×3, 122 misses: 0 at lost positions (every move is optimal there), 24 at drawn ones
  (a move that loses), 98 at won ones (a forced win not taken), all at 3–6 stones; 42
  among positions self-play visited, 80 among the rest.
- 4×4, 308 misses in the 50,000-position subsample (0.62%; 0.58% over all 9.06M): 1
  among the 594 visited positions, 289 missed forced wins and 19 losing moves among the
  rest, at 7–14 stones.
- Value sign agreement by the exact value — 3×3: lost 91.0%, drawn 93.4%, won 89.0%;
  4×4: lost 58.5%, drawn 99.3%, won 91.4%. Lost positions are what 4×4 self-play almost
  never produces.

## 5×5, four in a row — the instrument without the table

No dense table exists at 5×5 (3²⁵ bytes). `lake exe ttt-env n=5 k=4` and the trainer use
the on-demand solver instead (`lean_ttt_solver_*`: alpha-beta over the values {loss, draw,
win} with a symmetry-canonical transposition table whose entries carry both bounds and the
principal move; gated against the dense table on every reachable 3×3 and 4×4 position —
0 value mismatches, 0 principal moves outside the optimal set). Facts it establishes:

- **(5,5,4) is a draw**, the root as the max over its 25 openings each solved full-window:
  33–37 s, 44M entries, 99M nodes. Every opening draws. After X centre, O's only drawing
  replies are the four diagonal neighbours (6, 8, 16, 18); the inner-corner opening
  (cell 6) leaves O exactly one drawing reply; a corner opening leaves all 24.
- The sweep is a fixed random-play sample of 20,000 decision positions with at least six
  stones (`sweepMin=6`): a fresh 3-stone subtree costs seconds to solve, a 6-stone one
  milliseconds. Agreement at 5×5 is therefore exact per position but not exhaustive, and
  the sample is off the self-play distribution (self-play stood at 3 of the 20,000).
- Expected agreement over that sample: random 45.04%, win-or-block 77.03%. Scripted
  pairings over 200 games each way (`n5_env.log`, 12.7 min): random vs perfect 0/0/200
  and 0/0/200 — perfect wins every game, (5,5,4) punishes bad play; win-or-block vs
  perfect 0/7/193 as X, 0/1/199 as O; perfect vs perfect 200/200 draws both ways.
- The perfect player draws uniformly from the optimal set for each side's first move (the
  opening book makes it free) and takes the search's principal move from the third stone
  on: one solve a move instead of one per child, exact either way.

`lake exe alphazero-ttt n=5 k=4 iters=60 sims=200 sweep=20000 sweepMin=6 epochs=5 cap=28
tag=run1` (`n5_run1.log`), one 4060 Ti, 83,486 params: **25.4 min** for 60 iterations
(self-play 3 s, training up to 20 s, the instrument 3–8 s per iteration once its cache is
warm; iteration 0 pays ~4 min of first-time solves). Sample agreement 42.4 → 85.9%, value
sign 64.7%; net + search unbeaten as X from iteration 5 and 243/13 as O at iteration 60;
net alone 198/58 as X and 256/0 as O at the end — the O side had not closed at 200 sims.
Sweep misses (14.1%): 2,070 forced wins not taken and 758 losing moves, spread evenly
over 6–23 stones; value sign lost 61% / drawn 75% / won 62%. The net's opening policy:
76% on the centre from the empty board, and at X-centre-O-to-move 24 / 24 / 24 / 23% on
the four diagonal neighbours — the solver's four optimal replies.

`… iters=100 sims=400 … tag=run2` (`n5_run2.log`), the row the table quotes: **63.8 min**
for 100 iterations (self-play ~7 s, training up to 29 s, the instrument ~6 s per
iteration). Sample agreement 53.6 → 84.7 (iter 25) → 87.4% (100), value MSE 0.334, sign
70.7% (lost 53% / drawn 84% / won 71%); in the run's own readings net + search unbeaten
from iteration 59 to the end, net alone unbeaten for 24 iterations from 68 and 237/19 as O
at the last reading; root value +0.04. **Independent evaluations** of the saved net
(`iters=0 params=n5_params.bin`, seeds 2–4, 256 games each; the table's cells): alone
X 0/768/0 · O 0/708/60 (28, 20, 12 losses), with search X 0/768/0 · O 0/745/23 (23, 0, 0);
a first evaluation at seed 1 read 224/32 and 241/15. Over the **exhaustive opening**
(`opening=3`, 7,526 decision positions with ≤ 3 stones, two solves each): agree **92.11%**,
value sign 82.8%. The 3×3 and 4×4 nets under the same protocol: 0/768/0 in every cell. Misses 12.6%: 1,849 forced wins not taken, 676 losing moves; self-play
stood at 304,169 distinct positions, 4 of them in the sample. Opening policy: 26% on the
centre from the empty board (every opening draws), 27 / 24 / 23 / 22% on the four diagonal
replies after X centre. Between the two runs the difference is the search: 400 sims at
100 iterations holds the O side, 200 at 60 never did.

## Files

- `n3_run3_ctree.log`, `n4_run2_ctree.log` — the runs (search in C). `n3_run2.log`,
  `n4_run1.log` — the same configs with the tree in Lean (`n3_run1.log` is that run under
  the pre-coverage binary, identical to every printed digit). `n3_convbn_collapse.txt` — the first attempt, on the
  bestiary's conv-BN tower, which diverged at iteration 11 (planning doc §2): an excerpt of its
  instrument lines, since the log itself was overwritten by the relaunch under the same tag.
- `n{3,4}_curve.csv` — one row per iteration: loss, sweep agreement, value MSE / sign
  agreement / MAE, the four W/D/L readings, the root value, positions seen, wall clock.
- `n{3,4}_sweep.csv` — one row per swept position (all 4,520 at 3×3; the 50,000-position
  subsample at 4×4): exact value, the net's value, its move, whether optimal, seen by self-play,
  and its softmax over the cells. `n4_full_sweep.txt` — the 9,062,619-position numbers.
- `n{3,4,5}_policy.csv` — the probe positions (the empty board; X centre — cell 4, 10, 12).
- `n5_*` — the 5×5 runs (`n5_run1_curve.csv` the 200-sim run, `n5_curve.csv` / `n5_sweep.csv`
  the quoted run's); `n5_env.log` the solved-game facts and the scripted pairings;
  `n5_probe.log` the two-iteration timing probe; `n5_open_*.log` / `n5_centre_probe.log`
  the six opening classes solved one at a time; `n5_root.log` the root alone.
- `n{3,4}_params.bin` — the final parameters, on disk only (`.gitignore` keeps `runs/**/*.bin`
  out). `n4_probe.log` — the one-iteration timing probe.
- `alphazero_ttt.png` — `scripts/demos/ttt_figure.py` on this directory.
