# MobileNetV2, `full` recipe, bf16 — JAX reference path, 350 epochs

**✅ 71.634 % top-1 / 90.460 % top-5** (35,817 / 50,000, raw weights), epoch 350/350. Landed
2026-09-29 06:27:52 UTC. **38.82 h** trainer time, `attempt 1`, zero restarts, zero thermal rests,
EDAC 0 / 0 throughout.

The JAX half of the MobileNetV2 350-epoch pair (`planning/imagenet_parity.md` §7 R4). Job
`scripts/jobs/mnv2-full-jax-4gpu.conf`, unit `mnv2-jax`, launched 2026-09-27 15:38:12 UTC on the
4× 3060 box, fresh start (`/home/skoonce/mnv2_full350_relu6/` did not exist). The verified half is
`runs/2026-09-27-mnv2-verified-bf16-350ep/` (71.124 / 90.036), chained behind this run.

## The run, from its own output

`lr=0.045000  batch_size=256 (4 devices x 64)  epochs=350  params=3504872` ·
`steps_per_epoch=5004  total_steps=1751400` · bf16 convs and matmuls · the recipe is
`mobilenetV2ImagenetConfigFull`: TF-RMSProp (ρ 0.9, μ 0.9, ε 1.0 inside the root, mean-square
initialised to 1.0), ×0.98 per-epoch STAIRCASE from step 0 with no warmup (epoch-1 lr 0.045,
epoch-2 lr 0.0441), coupled L2 4e-5 off BN γ/β and biases, classifier dropout 0.2, no label
smoothing, TF-slim BN (decay 0.997, ε 1e-3), ReLU6 throughout, crop + flip only. The conf's
precheck asserted the BN constants, the staircase and the wd mask against the emitted file.

Trainer: `jax/.lake/build/generated_mobilenet_v2_imagenet_full.py`, synced from the committed
`jax/generated/` copy before launch (`scripts/regen_jax_generated.sh sync` then `box` ✅), on
`/home/skoonce/.venv-cuda` (jax 0.11.1, cuda13).

## Curve (top-1 / top-5, raw weights, every epoch)

| epoch | 1 | 10 | 25 | 50 | 100 | 150 | 200 | 250 | 300 | **350** |
|---|---|---|---|---|---|---|---|---|---|---|
| top-1 | 12.52 | 49.79 | 56.69 | 62.43 | 67.32 | 70.17 | 70.98 | 71.35 | 71.49 | **71.63** |
| top-5 | 29.91 | 75.63 | 80.70 | 84.70 | 87.99 | 89.58 | 90.20 | 90.39 | 90.36 | **90.46** |

(full per-epoch table: `mnv2_jax_curve.csv`, with lr and train loss). The tail is flat: e326–350
average **71.539 / 90.431**, top-1 range 71.444–71.634, so the endpoint is not a lucky epoch.

## Against what came before

* The book's MobileNetV2 reference, **71.90 / 90.41** (`content.tex` :7572, :7746, :7789), is the
  retired 2026-07-28 run. It most likely trained at label smoothing 0.1 from a stale emit
  (planning/imagenet_parity.md H3/M2-2), before the ReLU6 stem/head fix (2026-08-30), at BN ε 1e-5
  / decay 0.99, without the staircase-from-step-0. This run is the recipe the paper describes, so
  **−0.27** against that number compares two different recipes, not a re-run.
* Paper (Sandler et al.): 72.0. This run: −0.37.

## Pace

**399 s/epoch**, flat from e3 to e350 (min 399, max 400), eval of all 50,000 included (~18 s). That
is ~3 % slower than the 2026-09-27 probe, whose 403 s train + 19 s val was epoch 1 with compile
warmup. The conf's `~38 h` (the retired run's wall clock) held: 38.82 h.

## Artifacts

* checkpoints outside the repo, `/home/skoonce/mnv2_full350_relu6/`: weights for every epoch
  (`mobilenet_v2_imagenet_e{1..350}.bin`, ~14 MB each), the full train state for e348–350, and the
  final `mobilenet_v2_imagenet.{bin,state.npz}`
* `attempt.log` / `full.log` / `master.log`, `epoch_clock.tsv`, `edac.tsv`, the two watcher scripts
* `mnv2_jax_curve.csv` — written by `summarize.sh`
* no per-image record: the JAX trainer writes no bitmaps. Its e350 weights are kept, so a per-image
  re-score would pair it with the verified run's bitmaps.

## Reproduce

    bash runs/2026-09-27-mnv2-jax-bf16-350ep/summarize.sh
