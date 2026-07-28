# cifar8 optimizer ablation: {SGD, Nesterov momentum, AdamW} × {no-BN, BN} (2026-06-19/20)

Optimizer ablation on the deeper 8-conv CIFAR-10 CNN (`cifar8Verified` /
`cifar8BnVerified`; [16,16,32,32], 4 pools 32→2, → 128→64→64→10). All cells
share the proof-rendered forward + backward + parameter-gradient body — **only
the optimizer update op differs** (SGD `θ−lr·∇`, Nesterov momentum
`v←μv+∇; θ←θ−lr·(μv+∇)`, AdamW). The updates are `emitSgd` / `emitMomentum` /
`ViTRender.emitAdamV` swapped onto the same certified gradient; cotangent divided
by B (mean gradients), no label smoothing.

## Setup

- Net: no-BN = 22 params / 52,858 floats; BN = 38 params (8× per-channel γ/β added)
- 40 epochs each, batch=128, He init
- **Shared modern pipeline** (the controlled comparison): per-epoch shuffle + random
  hflip + cosine-warmup(3) schedule, all via `trainAdamSched` (variants `sgd`/`mom`/`adam`)
- Per-optimizer tuned lr: SGD 0.1, momentum μ=0.9/lr 0.02, AdamW lr 1e-3 (wd 1e-4)
- Backend: IREE + ROCm/HIP on RX 7900 XTX (gfx1100)
- Renders: `tests/TestCifar8AdamTrain.lean` (all six train steps); `tests/RenderCifar8Sgd02.lean`
  (lr sweep — **removed 2026-07-28**: it re-rendered the committed
  `verified_mlir/cifar8_train_step.mlir` at lr 0.02 rather than the committed 0.1, i.e. it silently
  overwrote a certified artifact with different hyperparameters. Recover with
  `git show 57e7a12:tests/RenderCifar8Sgd02.lean` if this sweep is ever re-run, and `git checkout
  verified_mlir/cifar8_train_step.mlir` afterwards.)

## Controlled results — modern pipeline, optimizer is the only variable (final % / best %)

| net | SGD (lr 0.1) | Nesterov momentum (μ.9, lr.02) | AdamW (lr 1e-3) |
|---|---|---|---|
| no-BN | 73.78 / 74.06 | **76.99 / 77.20** | 74.13 / 74.32 |
| BN    | 73.98 / 74.15 | **76.08 / 76.26** | 73.52 / 73.74 |

Findings:
- **Nesterov momentum wins both rows** — ~2–3 pts over SGD and over Adam. On a small vision
  net, well-tuned SGD+momentum edges out Adam's per-parameter adaptive scaling.
- **SGD ≈ Adam** once the pipeline is controlled — Adam's adaptivity earns nothing it keeps.
- **BN is ~neutral at convergence** (a fraction of a point either way; down under momentum/Adam,
  even under SGD) but a clear *early* accelerator — see `{nobn,bn}_sgdsched.log` per-epoch:
  BN leads from epoch 2, both converge to ~74% (this is the BN-vs-no-BN curve in Ch5 §5.1).

## Methodology note: the confound we caught

The FIRST cut (the `nobn_sgd.log`/`bn_sgd.log` SGD baselines, run via `VerifiedNet.train`)
had **NO per-epoch shuffle and NO hflip**, while the momentum/Adam runs (`trainAdamSched`)
**did**. That made the naive table below look like a ~10-pt optimizer gap that was really
mostly the data pipeline:

| net | SGD naive (no shuffle/aug, flat lr 0.1) |
|---|---|
| no-BN | 66.72 / 68.36 |
| BN    | 66.03 / 66.83 |
(extra: no-BN flat lr=0.02 = 65.00 / 65.71 — *below* lr=0.1, so the flat-SGD optimum is ≥0.1.)

Holding the pipeline fixed (the `*_sgdsched.log` runs, plain SGD on the SAME shuffle+hflip+cosine
path) lifts SGD to ~74% and collapses the gap to the real ~2–3 optimizer points. Lesson: an
ablation only measures the variable you changed if everything else is genuinely held constant —
and a verified gradient does not catch an experiment-design slip.

## Files

- `nobn_sgdsched.log` / `bn_sgdsched.log` — **SGD on the controlled pipeline** (`cifar8{,-bn}-verified-sgdsched`)
- `nobn_mom.log` / `bn_mom.log` — Nesterov momentum (`cifar8{,-bn}-verified-momentum`)
- `nobn_adam.log` / `bn_adam.log` — AdamW (`cifar8{,-bn}-verified-adam`)
- `nobn_sgd.log` / `bn_sgd.log` — SGD naive baseline (no shuffle/aug, `cifar8{,-bn}-verified`)
- `nobn_sgd_lr02.log` — SGD naive, lr 0.02, no-BN (re-rendered train step; committed artifact restored after)
