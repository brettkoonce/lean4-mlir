# CIFAR-10 batch-statistics BN, five seeds — minerva's 7900 XTX (ROCm 7.2.4), 2026-10-25/26

`SUITE=bbn scripts/sweeps/seed_sweep.sh 0 '1 2 3 4 5'`: `cifar8wb-bbn-ablation`, six arms, 40 epochs,
seeds 1–5, one card, in ≤ 55-min blocks (`../2026-10-25-xtx-ladder/`). The CUDA control is
`../2026-10-24-bbn-seeds/` (ares, 4060 Ti), same commit's node list.

Final-epoch test accuracy on all 10,000, mean and range over the five seeds:

| arm | 7900 XTX mean | 7900 XTX range | CUDA mean | CUDA range |
|---|---|---|---|---|
| sgd | 76.64 | 75.25–77.71 | 76.70 | 74.93–77.89 |
| mom | 77.83 | 77.48–78.30 | 76.98 | 75.73–77.91 |
| adam | 76.13 | 75.46–77.07 | 75.94 | 75.00–77.25 |
| bf16sgd | 76.57 | 72.89–78.13 | 76.95 | 76.29–78.08 |
| bf16mom | 76.95 | 75.78–77.85 | 78.01 | 77.76–78.37 |
| bf16adam | 75.65 | 74.66–76.04 | 75.29 | 72.22–77.47 |

Per seed (sgd / mom / adam / bf16sgd / bf16mom / bf16adam):

| seed | 7900 XTX |
|---|---|
| 1 | 75.25 / 77.91 / 75.46 / 76.43 / 76.55 / 74.66 |
| 2 | 77.32 / 77.81 / 75.66 / 77.72 / 77.85 / 75.59 |
| 3 | 75.99 / 78.30 / 76.89 / 72.89 / 77.01 / 76.04 |
| 4 | 76.92 / 77.48 / 77.07 / 78.13 / 77.58 / 76.01 |
| 5 | 77.71 / 77.67 / 75.56 / 77.70 / 75.78 / 75.94 |

Seeds 2 and 4 were cut by a block's time limit and resumed from their per-epoch checkpoints; the
resume truncated their logs to the last arm, so their first five arms are counted from the
epoch-40 correctness bitmaps (`bitmaps/*_e40.bin`, one byte per test image). Seed 3's bitmaps
reproduce its logged accuracies exactly.

The sweep wrote into `../2026-10-24-bbn-seeds/` (the newest `*-bbn-seeds` directory, as
`seed_sweep.sh` resumes); its files were moved here and the CUDA directory restored from git.

Wall time: each f32 arm ~2 min 20 s (8.6 ms/step), each bf16 arm ~7 min — bf16 on these 32×32
convolutions is 3× slower than f32 in MIOpen (`../../deploy/ROCM_7900XTX.md`, What to expect).
