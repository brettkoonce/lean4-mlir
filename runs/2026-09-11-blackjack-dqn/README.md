# 2026-09-11 blackjack DQN — rung 2 of planning/blackjack_dqn_demo.md

Environment, DP instrument and tabular Q: `LeanMlir/Blackjack.lean`, driven by
`lake exe blackjack-env` (default table, `dump`, `curve`, `play`). The DQN is
`lake exe blackjack-dqn [steps] [seed] [double] [lrdecay] [tag=…]`, XLA backend,
one GPU, `LEAN_MLIR_MEM_FRACTION=0.1` (6,210 params). ~2.9 s per 1000 updates.

Net `.dense 29 64 .relu, .dense 64 64 .relu, .dense 64 2 .identity`; loss is the
rank-2 DDPM MSE block with the host writing y = Q(s,·) and y[a] = r + γ·max Q_target(s',·);
replay 20k, batch 128, Adam 1e-3, ε 1.0→0.05 over 20k hands, target copy every 500
updates, γ = 1. The greedy policy is read off the online net every 50 updates
(280-state batched forward) for acting and scored exactly every 1000 for the curve.

## Scores (exact = value iteration; MC = 10^6 hands, seed 7; agreement over the 200 decision states)

| arm                              | updates | hands   | exact   | Monte Carlo       | agree |
|----------------------------------|---------|---------|---------|-------------------|-------|
| plain, seed 1                    | 49k     | 34k     | -0.0578 | -0.0570 ± 0.0010  | 175   |
| plain                            | 200k    | 133k    | -0.0503 | -0.0497 ± 0.0010  | 185   |
| Double                           | 200k    | 132k    | -0.0476 | -0.0471 ± 0.0010  | 188   |
| plain + lr decay (1e-3 → 1e-4)   | 200k    | 132k    | -0.0530 | -0.0526 ± 0.0010  | 178   |
| Double + lr decay                | 200k    | 132k    | -0.0503 | -0.0494 ± 0.0010  | 187   |
| tabular Q, 10^6 hands            |         | 1000k   | -0.0440 | -0.0425 ± 0.0010  | 192   |
| exact optimum                    |         |         | -0.0431 | -0.0419 ± 0.0010  | 200   |

Gate 2 (DQN within 0.01 of tabular Q's exact value) passes for every 200k arm.
One seed per arm; the 1000-update readouts jitter by ±0.005 late in training
(knife-edge cells flipping), so the 0.003 between Double and plain is not a
seed-robust ranking. Learning-rate decay did not reduce the jitter: the noise is
in the greedy readout, not the step size.

Files: `dqn_200k_<arm>.{out,log}` (chart + scores, training log), `dqn_200k_<arm>_{curve,policy}.csv`;
`dqn_curve.csv` / `dqn_policy.csv` are the Double arm (the figure); `tabq_curve.csv` is
`blackjack-env curve 1000000 1000`; `states.csv` is `blackjack-env dump`; `arms.csv` the table.
Figures: `python scripts/blackjack_figure.py runs/2026-09-11-blackjack-dqn blackjack_dqn.png`
(curve + charts, here only) and `… blackjack_chart.png chart` (the two charts, the book's copy at
`blueprint/src/figures/demos/blackjack_chart.png`). Needs matplotlib; ~/loiter-venv has it.
