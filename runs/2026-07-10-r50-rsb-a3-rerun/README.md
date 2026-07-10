# ResNet-50 RSB-A3 (rsb-faithful) — ImageNet, 100 epochs

Re-run on ares (4x RTX 4060 Ti, CUDA, bf16 matmul+conv), fresh from init after
the fan repair. Supervised via `jax/scripts/supervise_r50_a3_rsbfaithful_100ep.sh`
(AER watchdog + thermal duty-cycle cooldowns @ ep 25/50/75).

- Config: `resnet50ImagenetConfigRSBFaithful` — LAMB, eff-batch 2048 (512 micro x4
  grad-accum, Ghost-BN over 512), BCE over Mixup/CutMix, RandAugment m6, wd 0.02
  (skip BN gamma/beta+bias), train@160 / eval@224.
- Wall-clock: 2026-07-09 15:40:37 -> 2026-07-10 08:35:25 (~16h55m incl. 3x 30min cooldowns).
- **Result: 77.22% top-1 / 93.34% top-5** (ep100), reproducing + slightly beating
  the Jul-4 76.66%/93.03% run. Zero AER interruptions across all 100 epochs.
- Final ckpt `~/r50_a3_rerun/ckpt.bin` (weights not in repo).

## This run's milestones (from supervisor.log)
    ep25  top1 0.3798  top5 0.6282
    ep50  top1 0.5581  top5 0.7942
    ep75  top1 0.7162  top5 0.9024
    ep100 top1 0.7722  top5 0.9334

## Files
- `train.log` — full trainer stdout. NOTE: the supervisor truncates its runlog on
  every resume (`: > RUNLOG` per attempt), so this holds only the FINAL attempt
  (epochs 76-100). Earlier attempts' per-epoch stdout was overwritten at each
  cooldown resume and is not recoverable from disk.
- `epochs_final_attempt_76-100.tsv` — per-epoch val lines for that final segment.
- `supervisor.log` — supervisor narration across all attempts (fresh start + 3
  cooldown resumes). Includes stale Jul-4 milestone/COMPLETE lines at the top from
  the persistent master log; the Jul-9->10 rerun starts at the "START ... ckpt=
  /home/skoonce/r50_a3_rerun/ckpt" line.
