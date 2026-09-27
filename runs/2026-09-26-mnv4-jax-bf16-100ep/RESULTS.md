# MobileNetV4-Conv-M, `default` recipe, bf16 — JAX reference path, 100 epochs

**✅ 76.572 % top-1 / 92.980 % top-5** (EMA shadow), epoch 100/100. Landed 2026-09-26 17:24:17 UTC.
**16.17 h** trainer time, `attempt 1`, zero restarts, zero thermal rests, EDAC 0 / 0 throughout.

The JAX half of the MNv4 100-epoch pair (`planning/imagenet_parity.md` §7 R1, the 2026-09-25
rescope that retired the 50-epoch `half` pair). Job `scripts/jobs/mnv4-default-jax-4gpu.conf`,
unit `mnv4-jax`, launched 2026-09-26 01:13:33 UTC on the 4× 3060 box, fresh start (no checkpoint
under `/home/skoonce/mnv4_timm_100ep/`). The verified half is
`runs/2026-09-26-mnv4-verified-bf16-100ep/` (76.678 / 93.144).

## The run, from its own output

`lr=0.004000  batch_size=4096 (4 devices x 1024)  epochs=100  params=9715512` ·
`grad-accum=8x  micro_batch=512 (4 devices x 128) -> effective_batch=4096` ·
`steps_per_epoch=312  total_steps=31200` · bf16 matmuls · RandAugment N2 m9 (the conf's precheck
asserts the call) · EMA shadow scored every 5 epochs and at 100.

Trainer: `jax/.lake/build/generated_mobilenet_v4_imagenet.py`, synced from the committed
`jax/generated/` copy before launch, on `/home/skoonce/.venv-cuda` (jax 0.11.1, cuda13) — the
3060 box's stack; `scripts/lib/jax_job.sh` could not start a JAX conf on this box until it learned
to pick the python per box (committed separately).

## Curve (top-1 / top-5, EMA shadow)

| epoch | 5 | 10 | 25 | 50 | 75 | 90 | 95 | **100** |
|---|---|---|---|---|---|---|---|---|
| top-1 | 36.96 | 57.33 | 69.71 | 74.17 | 75.99 | 76.49 | 76.51 | **76.57** |
| top-5 | 62.23 | 80.58 | 89.04 | 91.74 | 92.73 | 92.92 | 92.96 | **92.98** |

Flat over the anneal: e90–100 moves +0.09.

## Against what came before

* The book's MNv4 100-epoch JAX tier, **75.48 / 92.37** (`content.tex` :7972, :7978, :8020),
  trained the **pre-timm-parity** transcription of the net. This run trains the timm-parity net
  (9,715,512 parameters) on the current `default` recipe: **+1.09 / +0.61**, but that is a
  different network, not a re-run — the two numbers do not pair. No book text is changed here.
* The paper's 79.9 is a 500-epoch, 256² recipe with a heavier RandAugment (m15 p0.7), drop-path,
  wd 0.1 and dropout 0.2 (`planning/imagenet_parity.md` §2.1); this is the 100-epoch tier.

## Pace

582 s/epoch mean (570–608), eval included — against the conf's `~16-18 h … UNMEASURED` ETA,
16.17 h. The 2026-09-25 probe (`~1,825 ms/step`) predicted it to within 2 %.

## Artifacts

* checkpoints outside the repo, `/home/skoonce/mnv4_timm_100ep/`: the EMA weights for every epoch
  (`mobilenet_v4_imagenet_e{1..100}.bin`), the full train state for e98–100 (the pruner keeps the
  three newest `.state.npz`), and the final `mobilenet_v4_imagenet.{bin,state.npz}`
* `attempt.log` / `full.log` / `master.log`, `epoch_clock.tsv` (one row per new `.state.npz`),
  `edac.tsv`, and the two watcher scripts
* `mnv4_jax_curve.csv` — written by `summarize.sh`
* no per-image record: the JAX trainer writes no bitmaps, so the pair with the verified run is
  unpaired as it stands. The e100 EMA weights are kept, so re-scoring them per image would pair it.

## Reproduce

    bash runs/2026-09-26-mnv4-jax-bf16-100ep/summarize.sh
