# CIFAR un-normalized fp32 vs bf16, five seeds — mars (4× RTX 3060, CUDA/XLA), 2026-10-29

`SUITE=nobnprec scripts/sweeps/seed_sweep.sh 0,1,2,3 "1 2 3 4 5"` (planning/runs_queue_2026-10-29.md
Tier 1b; the `nobnprec` case was added to `seed_sweep.sh` for this run, on the `bnprec` pattern):
`cifar8wb-ablation` (f32) and `cifar8wb-bf16-ablation`, the un-normalized wide net, three optimizer
arms each at the constant lr, 40 epochs. 10/10 jobs OK, no AER. Final-epoch test accuracy on all
10,000; `*` = the log printed `loss=NaN`.

September comparison: `runs/2026-09-01-cifar8w-6arm-constlr/` `fp32b_s*` and `bf16_s*` (ares,
4060 Ti), the directory behind the book's un-normalized fp32-vs-bf16 comparison.

## f32

| seed | SGD (lr 0.1) | Nesterov (μ .9, lr 0.02) | AdamW (lr 1e-3) |
|---|---|---|---|
| 1 | 69.41 | 74.52 | 73.96 |
| 2 | 70.36 | 73.37 | 74.47* |
| 3 | 69.56 | 72.90 | 72.60* |
| 4 | 70.11 | 73.92 | 72.96* |
| 5 | 69.88 | 72.52 | 73.67* |
| **median (range)** | **69.88** (69.41–70.36) | **73.37** (72.52–74.52) | **73.67** (72.60–74.47) |
| Sept 09-01 | 70.74 (69.02–71.31) | 73.85 (71.14–74.25) | 73.53 (71.56–74.23) |

## bf16

| seed | SGD | Nesterov | AdamW |
|---|---|---|---|
| 1 | 71.44 | 72.86 | 73.37* |
| 2 | 69.47 | 73.86 | 73.13* |
| 3 | 70.68 | 74.48 | 72.93 |
| 4 | 69.88 | 71.93 | 73.55* |
| 5 | 70.84 | 73.45 | 74.48 |
| **median (range)** | **70.68** (69.47–71.44) | **73.45** (71.93–74.48) | **73.37** (72.93–74.48) |
| Sept 09-01 | 69.34 (68.89–70.72) | 73.29 (10.00–74.47) | 73.44 (72.23–74.00) |

## Read

- Every median is inside its September range. bf16 against f32 on the new date: SGD +0.80,
  Nesterov +0.08, AdamW −0.30, all inside the per-arm seed ranges, so neither precision is resolved
  ahead of the other at five seeds.
- The September bf16 Nesterov 10.00 (seed 1, a run that never trained) does not recur: all ten
  Nesterov runs here trained. (The same failure appeared once in the same-date `small` sweep, no-BN
  per-image Nesterov seed 2; see `runs/2026-10-29-small-seeds/RESULTS.md`.)
- AdamW `loss=NaN`: f32 4/5 (first at epochs 39, 38, 38, 38), bf16 3/5 (38, 39, 33); September
  f32 1/5, bf16 4/5. Test accuracy stays at ~73 % in every case: the NaN is in the logged training
  loss in the last few epochs, not in the weights the test forward reads.
