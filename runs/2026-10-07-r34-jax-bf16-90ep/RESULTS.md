# ResNet-34, `default` (2018) recipe, bf16 with an fp32 head — JAX reference path, 90 epochs

**✅ 74.140 % top-1 / 91.820 % top-5** (live weights, no EMA), epoch 90/90.
* Landed 2026-10-07 17:38:07 UTC.
* **14.81 h** of trainer time, on attempt 1.
* Zero restarts, zero thermal rests, EDAC 0 / 0 throughout.

This is the rerun the book's R34 table owes (`content.tex` :6600, "[TODO: rerun the reference with
its classifier head in fp32.]"). The landed reference, 74.16 / 91.92 (August 2026), scored through a
bf16 classifier head. This trainer's head is the fp32 matmul (X3), as the verified render's always
was.
* Job: `scripts/jobs/r34-default-jax-4gpu.conf`. Its precheck asserts the torchvision crop, no
  RandAugment, 90 epochs, lr 0.1 and the fp32 head.
* Unit `r34-jax`. Launched 2026-10-07 02:49:10 UTC on the 4× 3060 box (mars).
* Fresh start: there was no checkpoint under `/home/skoonce/r34_2018_jax_2610/`.
* First of three chained ResNet JAX reruns (`planning/next_session_resnet_jax_reruns.md`). The A3
  run started from `runs/2026-10-07-r50-a3-jax-bf16-100ep/chain_after.sh` 55 s after this one
  completed.

## The run, from its own output

* `lr=0.100000  batch_size=256 (4 devices x 64)  epochs=90  params=21797672`
* `steps_per_epoch=5004  total_steps=450360`
* bf16 matmuls with an fp32 classifier head, SGD momentum 0.9, cosine with a 5-epoch warmup, label
  smoothing 0.1, BN decay 0.99 (as on the verified path)

Trainer: `jax/.lake/build/generated_resnet34_imagenet.py`, byte-equal to the committed
`jax/generated/` copy (`regen_jax_generated.sh check` + `sync`, 0 stale), on
`/home/skoonce/.venv-cuda` (jax 0.11.1, cuda13). Tree at `9d7def4d`.

## Curve (top-1 / top-5)

| epoch | 1 | 5 | 10 | 30 | 60 | 75 | 85 | 89 | **90** |
|---|---|---|---|---|---|---|---|---|---|
| top-1 | 11.63 | 35.50 | 41.59 | 49.68 | 59.84 | 67.90 | 73.54 | 74.11 | **74.14** |
| top-5 | 28.59 | 62.14 | 68.96 | 75.68 | 83.34 | 88.46 | 91.55 | 91.84 | **91.82** |

## Against the book's table

| | top-1 | top-5 |
|---|---|---|
| reference, landed (bf16 head, August) | 74.16 | 91.92 |
| **reference, this run (fp32 head)** | **74.14** | **91.82** |
| verified path (`runs/2026-09-22-r34-syncbn-bf16-90ep`) | 74.17 (74.168) | 91.89 (91.894) |

* **Against the verified path: −0.03 top-1 / −0.07 top-5.** That is a tenth of the ±0.38 95 %
  Wilson interval at n = 50,000, so the pair still agrees.
* The fp32 head moved the reference by −0.02 / −0.10, which is noise at one seed.
* The book's sentence "agree to 0.01 points on top-1 and 0.03 on top-5" would become 0.03 / 0.07.
  No book text is changed here.
* A per-image pairing (McNemar) is possible: the verified run has
  `bitmaps/r34_momdp64bf16_e90.bin`, and this run's e90 train state is kept. Re-scoring it per image
  (template: `runs/2026-10-01-vit-jax-bf16-300ep/score_arms.py`) would pair them. That has not been
  done yet.

## Pace

* **592 s an epoch, eval included** (min 590, max 593; ~575 s train + ~18 s val).
* That is 14.81 h in all, against the conf's ares ETA of 13.3 h and the handoff's mars estimate of
  13.5–14 h: about 6 % slower than estimated.
* `epoch_clock.tsv` stops at e89: the watcher exits with the unit, before it could log e90's state.

## Artifacts

* Checkpoints, outside the repo in `/home/skoonce/r34_2018_jax_2610/`:
  * the weights for every epoch (`resnet34_imagenet_e{1..90}.bin`)
  * the full train state for e88–90
  * the final `resnet34_imagenet.state.npz`
* `attempt.log` / `full.log` / `master.log`, `epoch_clock.tsv`, `edac.tsv`, and the two watcher
  scripts
* `r34_jax_curve.csv`, written by `summarize.sh`

## Reproduce

    bash runs/2026-10-07-r34-jax-bf16-90ep/summarize.sh
