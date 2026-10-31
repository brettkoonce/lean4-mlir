# CIFAR-10 batch-statistics BN, head-width grid — minerva's 7900 XTX (ROCm 7.2.4), 2026-10-25/26

`GPUS=0 scripts/sweeps/run_cifar8wb_bbn_grid_sweep.sh 25` (`../2026-10-25-xtx-ladder/`, block 3):
`cifar8wb-bbn-grid`, AdamW, cosine, 25 epochs, one seed per width. fc8 finished on 10-25; the box
reset two minutes into fc4096 (23:12:57, `../../deploy/ROCM_7900XTX.md` §2), and the other nine
widths ran on 10-26 02:30–02:44 by calling the binary per width (`6g1.out` in the ladder dir), so
`results.tsv` was assembled from the ten logs afterwards. Test accuracy on all 10,000, Wilson 95 %:

| fc width | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|
| 7900 XTX | 74.97 | 76.87 | 76.94 | 76.10 | 76.69 | 76.89 | 77.88 | **78.16** | 77.74 | 77.42 |
| CUDA (`../2026-10-24-cifar8wb-bbn-grid-xla/`) | 75.98 | 76.99 | 77.35 | 76.40 | 76.49 | 77.60 | **77.85** | 77.84 | 77.30 | 77.62 |

Every width is within one interval (~±0.83) of CUDA's; the shape is the same — fc8 lowest, nothing
past fc32 separable.
