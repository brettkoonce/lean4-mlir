# ResNet-50 RSB-A3 — ImageNet, 100 epochs, **VERIFIED path** (phase 4)

The A3 recipe on the Lean → StableHLO → XLA/PJRT path: the rendered MLIR the proofs reason
about, lowered through `ffi/pjrt_ffi.c`, no Python at run time. Companion to
`runs/2026-07-10-r50-rsb-a3-rerun/`, which is the same recipe on the phase-2 (Lean→JAX) trainer.

- **Artifact**: `verified_mlir/resnet50in160_lambaccdp8x64bce_train_step.mlir` — LAMB ×
  BCE-with-logits × gradient accumulation k=8, 4 replicas × per-replica batch 64 =
  effective batch 2048, train@160 / eval@224, mixup α0.1 / cutmix α1.0, RandAugment
  m6-mstd0.5-inc1, lr 0.008, wd 0.02, cosine + 5-epoch warmup.
- **Job**: `scripts/supervise.sh r50-imagenet-4gpu` (that config is `r50-a3-4gpu` since the 2026-08-27 naming pass; the name here is what the log recorded) on 4× RTX 4060 Ti (CUDA, **fp32**).
- **Result: 77.43% top-1 / 93.60% top-5** at ep100 (peak ep99, 77.49%), against the phase-2
  reference's 77.22% / 93.34% and paper RSB-A3's 78.1%.

## Wall-clock

Ran in two legs, 2026-08-06 03:47 → 2026-08-07 14:01:

| leg | epochs | wall-clock |
|---|---|---|
| 1 | 0–29 | 10 h 20 m |
| 2 | 30–99 | 23 h 44 m |
| | | **34.06 h compute** (34.22 h end-to-end) |

20.6 min/epoch (240 ms/step real, 222 synth, ~54 s/epoch eval). **One launch per leg, zero
thermal duty-cycle cooldowns, zero PCIe/AER interruptions, zero crashes**; peak ~53 °C against
the 78 °C trip.

⭐ The split was free, not a restart: `totalSteps` reads the *configured* epoch count while
`LEAN_MLIR_MAX_EPOCHS` caps only the loop, so leg 1 is a genuine prefix of the 100-epoch cosine
and leg 2 resumed onto the same curve. Verified from the run itself — lr at ep25 was 0.007157,
matching both the closed form and the phase-2 reference's ep25 to six decimals.

## Milestones, against the phase-2 reference

| epoch | ours top-1 | JAX rerun | JAX Jul-4 |
|---|---|---|---|
| 25 | 50.49 | 37.98 | 40.02 |
| 50 | 62.10 | 55.81 | 56.60 |
| 75 | 73.28 | 71.62 | 71.81 |
| 100 | **77.43** | **77.22** | **76.66** |

The verified run led by 10.5 points at ep25, the reference recovered nearly all of it during
the anneal, and the gap then stabilised at +0.2–0.4 through the last twenty epochs.

## ⚠ This is NOT "RSB-A3 reproduced"

It finished ahead of the phase-2 reference **while missing** ingredients that reference has —
`wdExcludeNormBias`, a 512-image BN group (ours is 64), and accumulation-compensated BN
running-stat momentum. Full ledger, with evidence and remediation cost per item, in
**`planning/a3_paper_fidelity.md` §2**. Quote a number against that list.

## Files

- `train.log` — full trainer stdout, both legs concatenated (`full.log` from the supervisor,
  which appends across attempts rather than truncating — so unlike the phase-2 run's log, the
  early epochs survive here).
- `epochs.tsv` — per-epoch loss, lr, and val top-1/top-5 for all 100 epochs.
- `supervisor.log` — supervisor narration across both legs.
- Checkpoint `.lake/build/resnet50in160_lambaccdp8x64bce_ckpt_xla.bin` (390 MB, not in repo).
