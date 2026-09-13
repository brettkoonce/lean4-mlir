# cifar8-BN dense-head width sweep, XLA path, all 10,000 test images

`scripts/run_cifar8bn_grid_sweep.sh 25 runs/2026-09-13-cifar8bn-grid-xla`, GPUs 1–3 of the 4× 4060 Ti
box, 22 minutes wall clock. Replaces `runs/cifar8bn_grid_results.tsv` (2026-07-24), which was run on
the retired ROCm pair through IREE and scored the pre-fix 9,984-image denominator (78 × 128). This
sweep scores all 10,000 and records the Wilson 95% interval the trainer prints.

Same experiment otherwise: 8-conv [16,16,32,32] BN backbone held fixed, dense head 128→d→d→10 swept,
AdamW 1e-3, cosine with 3-epoch warmup, 25 epochs, batch 128, one run per width, each on its own
width-slugged render (`.lake/build/cifar8_bn_<d>_adam_train_step.mlir`, rendered 2026-09-02).

| d    | acc    | 95% CI          |
|------|--------|-----------------|
| 8    | 70.38  | [69.48, 71.27]  |
| 16   | 72.06  | [71.17, 72.93]  |
| 32   | 73.61  | [72.74, 74.46]  |
| 64   | 73.13  | [72.25, 73.99]  |
| 128  | 73.60  | [72.73, 74.45]  |
| 256  | 73.66  | [72.79, 74.51]  |
| 512  | 74.09  | [73.22, 74.94]  |
| 1024 | 73.34  | [72.46, 74.20]  |
| 2048 | 73.38  | [72.50, 74.24]  |
| 4096 | 73.07  | [72.19, 73.93]  |

Read: the plateau starts at d = 32 — every width from 32 to 4096 spans 73.07–74.09, and all eight
intervals overlap (the lowest upper bound, 73.93 at d = 4096, clears the highest lower bound,
73.22 at d = 512). d = 16 sits below it — its interval [71.17, 72.93] does not reach d = 512's — and
d = 8 is clearly below everything (70.38, upper bound 71.27 against plateau lower bounds ≥ 72.19).
The canonical d = 64 (73.13) sits on the plateau. The 4096-wide head (17M-parameter classifier)
buys nothing the test set can resolve.

Against the July table the whole curve sits about two points higher (71.1–72.2 → 73.1–74.1 on the
plateau) and d = 8 gains 3.6. Two things moved between the runs besides the box and the lowerer: the
2026-08-04 init change (fan-out conv init, Glorot dense, matching the JAX reference) and the
denominator. The shape — sharp knee, flat plateau — is the same.

No per-epoch timing or parameter-count columns: the grid driver's transcript carries neither. Whole-run
wall clock can be read off `sweep.out`'s start stamps (each GPU runs its widths back to back): d = 64 took
1 m 56 s (11:38:23 → 11:40:19), d = 512 2 m 25 s, d = 4096 18 m 02 s (11:29:46 → 11:47:48) — the 17M-parameter
head is nine times the wall clock of the 64-wide one. Three jobs shared the host, so treat these as ratios.
