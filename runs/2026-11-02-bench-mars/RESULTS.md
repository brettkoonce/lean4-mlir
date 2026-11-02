# Real-data bench, first pass — mars (4× RTX 3060, Xeon w5-2455X, 172 GB), 2026-11-02

`lake run bench r34 a3 vit` at 71e4fe9c (`scripts/bench/`): each row sources the job conf the book's run
used, trains 2 epochs on real ImageNet (`$HOME/tensorflow_datasets`), and stops. Launched 09:14 UTC after
the A3 seed queue left the GPUs; done 12:50 UTC. `bench.tsv` has every row; one directory per row holds
its timestamped `train.log` and the `env.json` read from the live process. Bench checkpoints were deleted
after each row; no conf's lineage was touched.

## Epoch 2 (epoch 1 carries the compile)

| net | path | job | s/epoch | step ms | img/s | GPU util | host CPU busy | known pace |
|---|---|---|---|---|---|---|---|---|
| R34 | JAX | `r34-default-jax-4gpu` | 601 | 117 | 2,198 | 97 % | 0.63 | 560 (14.0 h/90 ep, Aug sweep) |
| R34 | verified | `r34-default-bf16-4gpu` | 885 | 167 | 1,448 | 80 % | 0.44 | ≈21 h/90 ep |
| R50 A3 | JAX | `r50-a3-jax-4gpu` | 511 | 789 | 2,599 | 99 % | 0.78 | ≈520 (8.7 min) |
| R50 A3 | verified | `r50-a3-wxclip4x128-bf16-4gpu` | 757 | 279 | 1,693 | 75 % | 0.62 | 770–800 (12.8–13.3 min) |
| ViT-Ti | JAX | `vit-default-jax-4gpu` | 556 | 445 | 2,306 | — | — | 551 (45.9 h/300 ep) |
| ViT-Ti | verified | `vit-default-emabf16-4gpu` | 754 | 568 | 1,699 | 51 % | 0.95 | 779 (64.9 h/300 ep) |

s/epoch is wall between epoch ends (JAX includes its ~18 s val; the verified rows' train/val split reads 0
because their val is inside the epoch line). Step ms is from in-epoch step 200 on; img/s is 1,281,167 over
train seconds.

## Reading

* Five of six rows land on their known pace within a few percent; R34 JAX is ~7 % slower than the August
  sweep. The harness measures what the long runs measure.
* Every verified row is feed-bound: 51–80 % GPU utilization against 95–99 % on JAX, and the ViT verified
  row keeps the host 95 % busy. The vCPU count matters more than the card for a cloud box on this path.

## Harness notes

* ViT-Ti's JAX trainer validates every 5 epochs; the other epochs end `[Xs train, val skipped]`, which the
  first pass's epoch regex did not match. The row trained 10 epochs (its two validated ones counted) and
  wrote one empty line. `bench_run.py` now takes the skipped form; the two ViT JAX rows in `bench.tsv` were
  re-parsed from `vit-jax/train.log` with the fixed parser (`status ok:reparsed`; the telemetry columns are
  blank, it is not kept on disk).
* The verified ViT row's PRECHECK warned that the conf's own checkpoint (epoch 300) would resume. The bench
  appends its own `LEAN_MLIR_CKPT_TAG` after the PRECHECK, and the log starts at `Epoch 1/300`.
