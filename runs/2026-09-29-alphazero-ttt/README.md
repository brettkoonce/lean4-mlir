# AlphaZero on tic-tac-toe, 2026-09-29 (planning/alphazero_ttt_demo.md Phases 0–2)

`lake exe alphazero-ttt n=3 tag=run2` (20 iterations, 25 sims, defaults) and
`lake exe alphazero-ttt n=4 iters=40 sims=100 sweep=50000 epochs=5 tag=run1`, XLA, one
RTX 4060 Ti each. 256 self-play games per iteration in lockstep, PUCT c = 1.5, Dir(1.0) at
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
| **net alone** | **97.90%** (4,520) | 0/256/0 · 0/256/0 from iteration 14 | **99.20%** (9,062,619) | 0/256/0 · 0/256/0 from iteration 38 |
| net + MCTS | — | 0/256/0 · 0/256/0 from iteration 4 | — | 0/256/0 · 0/256/0 from iteration 4 |
| perfect | 100% | 0/1000/0 · 0/1000/0 | 100% | 0/1000/0 · 0/1000/0 |
| untrained net (iteration 0) | 57.9% | 0/0/256 · 0/0/256 | 60.3% | 0/150/106 · 0/87/169 |

3×3: 20 iterations, 63,470 steps, **4.8 min**. 4×4: 40 iterations, 186,649 steps,
**34.9 min** (the talk's alpha-zero-general estimate for this board was 20–24 h). Value
head: sign agreement 92.2% / MSE 0.093 at 3×3, 94.3% / 0.054 at 4×4 over every decision
position. The root's value at the end: +0.26 at 3×3, +0.03 at 4×4 (the theorem says 0
for both; self-play at 3×3 stays X-favoured, 83 X wins / 145 draws / 28 O wins in the last
iteration, while 4×4 self-play draws 234 of 256).

## Where the misses are (`scripts/demos/ttt_sweep_stats.py`)

- Self-play stood at **2,010 of the 4,520** 3×3 decision positions (44.5%) and at
  **107,153 of the 9,062,619** 4×4 ones (1.2%); the sweep scores every position.
- 3×3, 95 misses: 0 at lost positions (every move is optimal there), 29 at drawn ones
  (a move that loses), 66 at won ones (a forced win not taken); 32 + 27 of them at 3 and 4
  stones; 33 among positions self-play visited, 62 among the rest.
- 4×4, 444 misses in the 50,000-position subsample (0.89%; 0.80% over all 9.06M): 0
  among the 568 visited positions, 411 missed forced wins and 33 losing moves among the
  rest, at 7–14 stones.
- Value sign agreement by the exact value — 3×3: lost 83.2%, drawn 93.2%, won 93.9%;
  4×4: lost 64.6%, drawn 99.5%, won 85.9%. Lost positions are what 4×4 self-play almost
  never produces.

## Files

- `n3_run2.log`, `n4_run1.log` — the runs (run1 at 3×3 is the same run under the pre-coverage
  binary, identical to every printed digit). `n3_convbn_collapse.txt` — the first attempt, on the
  bestiary's conv-BN tower, which diverged at iteration 11 (planning doc §2): an excerpt of its
  instrument lines, since the log itself was overwritten by the relaunch under the same tag.
- `n{3,4}_curve.csv` — one row per iteration: loss, sweep agreement, value MSE / sign
  agreement / MAE, the four W/D/L readings, the root value, positions seen, wall clock.
- `n{3,4}_sweep.csv` — one row per swept position (all 4,520 at 3×3; the 50,000-position
  subsample at 4×4): exact value, the net's value, its move, whether optimal, seen by self-play,
  and its softmax over the cells. `n4_full_sweep.txt` — the 9,062,619-position numbers.
- `n{3,4}_policy.csv` — the probe positions (the empty board; X centre / cell 10 for 4×4).
- `n{3,4}_params.bin` — the final parameters, on disk only (`.gitignore` keeps `runs/**/*.bin`
  out). `n4_probe.log` — the one-iteration timing probe.
- `alphazero_ttt.png` — `scripts/demos/ttt_figure.py` on this directory.
