# CIFAR per-image binaries + the MNIST three, five seeds — mars (4× RTX 3060, CUDA/XLA), 2026-10-29

`SUITE=small scripts/sweeps/seed_sweep.sh 0,1,2,3 "1 2 3 4 5"` (planning/runs_queue_2026-10-29.md
Tier 1b): `cifar8w-ablation` (no BN) and `cifar8w-bn-ablation` (BN, per image), three optimizer arms
each at the constant lr, 40 epochs; `mnist-{linear,mlp,cnn}-verified` at 12 epochs. 25/25 jobs OK,
no AER. Final-epoch test accuracy on all 10,000 (`sweep.log` OK lines and the per-arm `epoch 40`
lines of each log; per-arm bitmaps in `bitmaps/`, gitignored). `*` = the log printed `loss=NaN`.

The September comparison is `runs/2026-09-01-cifar8w-6arm-constlr/` (`nobn_s*`, `bn_s*`; ares,
4060 Ti), the directory `tab:cifar_optimizers` cites for these two rows.

## cifar8w, no BN

| seed | SGD (lr 0.1) | Nesterov (μ .9, lr 0.02) | AdamW (lr 1e-3) |
|---|---|---|---|
| 1 | 71.61 | 72.31 | 72.39 |
| 2 | 68.97 | **10.00** | 73.87* |
| 3 | 69.03 | 73.69 | 72.77* |
| 4 | 69.17 | 73.72 | 73.07* |
| 5 | 72.00 | 74.00 | 72.79* |
| **median (range)** | **69.17** (68.97–72.00) | **73.69** (10.00–74.00) | **72.79** (72.39–73.87) |
| Sept 09-01 | 68.75 (68.03–70.73) | 72.15 (71.05–74.36) | 72.78 (71.04–73.82) |

## cifar8w-bn, BN per image

| seed | SGD | Nesterov | AdamW |
|---|---|---|---|
| 1 | 75.47 | 76.57 | 74.13 |
| 2 | 74.65 | 76.09 | 74.52 |
| 3 | 73.37 | 76.57 | 73.40 |
| 4 | 74.80 | 75.93 | 74.21 |
| 5 | 75.14 | 76.41 | 74.55 |
| **median (range)** | **74.80** (73.37–75.47) | **76.41** (75.93–76.57) | **74.21** (73.40–74.55) |
| Sept 09-01 | 74.50 (74.14–75.21) | 76.35 (75.75–76.73) | 74.29 (74.24–74.46) |

No NaN in any BN log.

## MNIST, 12 epochs

| seed | linear | MLP | CNN |
|---|---|---|---|
| 1 | 92.10 | 97.81 | 98.74 |
| 2 | 92.10 | 97.65 | 98.79 |
| 3 | 92.10 | 97.69 | 98.78 |
| 4 | 92.10 | 97.83 | 98.78 |
| 5 | 92.10 | 97.67 | 98.64 |
| **median** | 92.10 | 97.69 | 98.78 |

MNIST-linear is 92.10 on every seed: the seed does not reach that trainer (zero init, fixed order),
as in August. Not new.

## Read

- **Every median lands inside or at the edge of its September range**; the largest move is no-BN
  Nesterov +1.5 (73.69 vs 72.15), within the two ranges' overlap. Both the 09-01 rows of
  `tab:cifar_optimizers` hold on the new date.
- **No-BN Nesterov seed 2 never trained**: 10.00 % (chance) from epoch 1 to 40, loss 2.27 at
  epoch 1, then pinned at ln 10 = 2.303 through epoch 40 (a uniform output). A dead seed, not a NaN. September had the same thing once, in the un-normalized
  bf16 arm (`runs/2026-09-01-cifar8w-6arm-constlr/bf16_s1.log`, Nesterov 10.00), never in the
  per-image no-BN arm. With the table's "range from lowest run to highest", this cell's range would
  read (10.00–74.00); the median is unaffected.
- **No-BN AdamW `loss=NaN`: 4 of 5 seeds** (first NaN at epochs 32, 40, 37, 37) against 3 of 5 in
  September (37, 37, 39). In every case the test accuracy stays at ~73 %: the NaN is in the logged
  training loss late in the run, not in the weights the test forward reads.
