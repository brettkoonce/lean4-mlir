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
- `n{3,4}_policy.csv` — the probe positions (the empty board; X centre / cell 10 for 4×4).
- `n{3,4}_params.bin` — the final parameters, on disk only (`.gitignore` keeps `runs/**/*.bin`
  out). `n4_probe.log` — the one-iteration timing probe.
- `alphazero_ttt.png` — `scripts/demos/ttt_figure.py` on this directory.
