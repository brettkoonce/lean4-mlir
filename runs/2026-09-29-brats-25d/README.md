# 2026-09-29 — BraTS: per-volume scoring, the 2.5D A/B, and the 3D gate

Three things the plan (`planning/brats_25d_3d.md`) had scoped and none of which existed in
code: a per-volume scorer (Stage G), the 2.5D variant of the ResNet-34 UNet with its data
build (Stage B / Gate D), and the 3D UNet's Gate 0 measured by running it rather than compiling
it (Stage C). Plus a JAX 3D UNet trainer on the whole volumes, as the reference a Lean 3D UNet
would tie to.

```bash
# data: the 2.5D build (±1 slice → 12 channels) and whole-volume exports for per-volume scoring
python3 scripts/datasets/preprocess_brats.py data/brats/Task01_BrainTumour data/brats224c3 --size 224 --seed 0 --context 1 --val-full
python3 scripts/datasets/preprocess_brats.py data/brats/Task01_BrainTumour data/brats224 --size 224 --seed 0 --only-val-full
python3 scripts/datasets/preprocess_brats.py data/brats/Task01_BrainTumour data/brats224 --size 224 --seed 0 --only-train-full
python3 scripts/datasets/preprocess_brats.py data/brats/Task01_BrainTumour data/brats --seed 0 --only-val-full

# the 2.5D A/B: r34 on GPU 0, scratch on GPU 1, same seed/schedule/records as the 2D run of 09-25
./scripts/sweeps/run_brats_r34_ab.sh 10 data/brats224c3 skip 1
python3 scripts/probes/brats_r34_ab.py runs/2026-09-25-brats-r34-xla/brats_skip_r34_gpu0.log \
        brats_skip_c1_r34_gpu0.log --names "2D r34,2.5D r34" --arm r34

# per-volume Dice (the literature's protocol) of any checkpoint
CUDA_VISIBLE_DEVICES=2 lake exe brats-eval net=r34 arm=r34 best out=pervol_2d_r34.csv
CUDA_VISIBLE_DEVICES=2 lake exe brats-eval net=r34 ctx=1 arm=r34 best out=pervol_25d_r34.csv

# 3D: Gate 0 (throughput), then the JAX 3D UNet on 128³ patches
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u jax/scripts/unet3d_gate0.py
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u jax/scripts/unet3d_brats.py --steps 4000 --out runs/2026-09-29-brats-25d/unet3d
```

Logs here are the trainers' stderr with the XLA/ptxas chatter stripped; `regen_*.log` are the
data builds (the 2.5D train split is 14,415 records, the same count and order as the 2D one).

## Gate 0 — rank-5 runs, at 77% of the 2D per-voxel rate

`gate0.log`. The whole 3D UNet (unetBrats's shape with 3³ kernels, 23.5M params), forward +
backward + update through XLA on one 4060 Ti, against the 2D twin at 16 × 240² from the same
harness:

| net | input | ms/step | Mvox/s | per voxel vs 2D | peak GiB |
|---|---|---|---|---|---|
| 2D UNet | 16 × 240² | 155 | 5.95 | 1× | 2.9 |
| 3D UNet | 128³ × B2 | 897 | 4.68 | 1.3× slower | 8.6 |
| 3D UNet | 128³ × B1 | 452 | 4.64 | 1.3× | |
| 3D UNet | 96³ × B2 | 378 | 4.67 | 1.3× | |
| 3D UNet | 64³ × B4 | 221 | 4.75 | 1.3× | |

The gate allowed 3–5×. nnU-Net's 128³ × B2 patch fits with 6 GiB to spare.

## Per-volume Dice of the 09-25 2D checkpoints

`pervol_2d*.csv`, one row per validation patient. Best-by-val checkpoints; the trainer's pooled
protocol reproduced first (it agrees with the trainer's log to 0.0004), then every slice, then
the literature's per-volume mean:

| checkpoint | pooled, tumour-bearing slices | pooled, every slice | per-volume mean ± sd (median) |
|---|---|---|---|
| r34 (ImageNet bootstrap) | WT 0.912 · TC 0.869 · ET 0.855 | 0.910 · 0.868 · 0.854 | WT 0.893 ± 0.074 (0.914) · TC 0.821 ± 0.152 (0.867) · ET 0.790 ± 0.185 (0.858) |
| scratch (He-init) | WT 0.910 · TC 0.867 · ET 0.855 | 0.907 · 0.866 · 0.854 | WT 0.889 ± 0.077 (0.917) · TC 0.819 ± 0.158 (0.869) · ET 0.783 ± 0.186 (0.851) |
| scratch, no skips | WT 0.871 · TC 0.814 · ET 0.733 | 0.869 · 0.814 · 0.732 | WT 0.843 ± 0.107 (0.873) · TC 0.749 ± 0.195 (0.801) · ET 0.620 ± 0.233 (0.679) |
| from-scratch UNet 240², dicece (3 ep) | WT 0.903 · TC 0.856 · ET 0.854 | 0.901 · 0.855 · 0.853 | WT 0.882 ± 0.082 (0.908) · TC 0.802 ± 0.164 (0.850) · ET 0.782 ± 0.174 (0.839) |
| from-scratch UNet 240², ce (3 ep) | WT 0.898 · TC 0.857 · ET 0.844 | 0.897 · 0.857 · 0.843 | WT 0.870 ± 0.099 (0.905) · TC 0.801 ± 0.163 (0.844) · ET 0.773 ± 0.178 (0.835) |

Per volume the skips are worth more than the pooled table said: +0.05 WT, +0.07 TC, +0.17 ET
against +0.04 / +0.06 / +0.12 pooled. The `dicece`/`ce` gap on the from-scratch net, 0.006
pooled, is 0.012 per volume on WT — still inside one seed's noise, but the per-volume number is
the one that separates them.

The tumour-free slices cost almost nothing (every-slice within 0.003 of tumour-slice), so the
slice model does not paint tumour where there is none. The per-volume mean is 2–7 points below
the pooled number with a wide spread: a tail of small-tumour patients pulls the mean down while
the medians sit at the pooled values. One of the 73 patients has no enhancing tumour (scored by
the BraTS convention, 1 or 0 outright).

## The 2.5D A/B — Gate D fails: ±1 slice is worth nothing

`brats_skip_c1_r34_gpu0.log`, `brats_skip_c1_scratch_gpu1.log`; per-epoch tables from
`scripts/probes/brats_r34_ab.py` (with `--names`). Same 14,415 records in the same order, same
seed, schedule and loss as the 09-25 2D run; the only change is the stem, `[64,12,7,7]` for
`[64,4,7,7]`, fed the slice above and below as eight extra channels. 240 ms/step against 252.

Pooled over the 2,569 tumour-bearing val slices, peak of ten epochs:

| arm | mIoU | WT | TC | ET | 2D peak (mIoU / WT / TC / ET) |
|---|---|---|---|---|---|
| 2.5D r34 | 0.744 | 0.913 | 0.874 | 0.858 | 0.744 / 0.913 / 0.869 / 0.857 |
| 2.5D scratch | 0.744 | 0.911 | 0.871 | 0.858 | 0.741 / 0.910 / 0.867 / 0.857 |

Deltas of +0.001 to +0.004, on both arms. Gate D asked for +0.02 mIoU or +0.03 ET.

Per patient (best-by-val checkpoints, `pervol_25d_*.csv`):

| checkpoint | WT | TC | ET | 2D twin |
|---|---|---|---|---|
| 2.5D r34 | 0.893 ± 0.074 (0.916) | 0.824 ± 0.154 (0.868) | 0.788 ± 0.190 (0.850) | 0.893 / 0.821 / 0.790 |
| 2.5D scratch | 0.891 ± 0.081 (0.921) | 0.819 ± 0.163 (0.868) | 0.790 ± 0.183 (0.848) | 0.889 / 0.819 / 0.783 |

The curves sit on top of each other from epoch 2 on (the 2.5D bootstrap arm led by 0.024 mIoU at
epoch 1 and by nothing after epoch 6); epochs-to-target are identical on every metric. The
render (`brats_25d_pred.png`) is the 2D figure with the same boundaries.

What this does and does not say. The slices are 1 mm apart and skull-stripped brain changes
little over 1 mm, so the two neighbours are very nearly the centre slice again: the finding is
that **adjacent-slice context adds nothing a 2D model at this resolution cannot already infer**,
not that through-plane information is useless. A wider context (`--context 2 --context-step 2`,
±4 mm, 20 channels, 58 GB of host RAM) is untested, and the 3D probe below is the direct test of
volumetric context.

## The 3D UNet in JAX — parity with the 2D anchor after one hour, still descending

`unet3d_log.txt` (the trainer's own log), `unet3d_pervol.csv`, `unet3d_params.npz` (94 MB, not
committed). `jax/scripts/unet3d_brats.py`: unetBrats's shape with 3³ kernels (23.5M params),
random 128³ patches at batch 2 from the whole training volumes (`train_full.bin`), a third of
them centred on a tumour voxel, mirrored along each axis, Dice + CE, Adam 3e-4 with 200 warmup
steps and cosine decay to zero over 4,000 steps; BatchNorm with running stats, because that is
what a Lean port would have. 909 ms/step, 63.5 min on one 4060 Ti; scoring is `brats-eval`'s
protocol on two 128-deep z-windows with the overlap averaged.

| | pooled, tumour-bearing slices | per-patient mean ± sd (median) |
|---|---|---|
| step 2,000 (32 min) | WT 0.905 · TC 0.844 · ET 0.842 · mIoU 0.723 | WT 0.882 ± 0.110 (0.912) · TC 0.783 ± 0.189 (0.847) · ET 0.781 ± 0.176 (0.845) |
| step 4,000 (64 min) | WT 0.910 · TC 0.859 · ET 0.848 · mIoU 0.737 | WT 0.894 ± 0.063 (0.914) · TC 0.810 ± 0.162 (0.858) · ET 0.790 ± 0.171 (0.854) |
| 2D R34 anchor, 10 ep (50 min) | WT 0.912 · TC 0.869 · ET 0.855 · mIoU 0.743 | WT 0.893 ± 0.074 (0.914) · TC 0.821 ± 0.152 (0.867) · ET 0.790 ± 0.185 (0.858) |

Per patient the one-hour 3D net ties the 2D anchor on WT and ET and trails it by 0.011 on TC,
with a tighter WT spread (sd 0.063 against 0.074); pooled it trails by 0.002 to 0.010. It was
not done: between the two evals every per-patient number rose (+0.012 / +0.027 / +0.010) and the
loss was still falling (0.37 → 0.30). A from-scratch volumetric net reaching the bootstrapped
slice model's per-patient Dice in an hour is the most encouraging number of the day; whether it
goes past it is a longer-schedule question (nnU-Net trains roughly a hundred times longer), and
that experiment is a JAX run, not a codegen investment.

## Where this leaves the 3D question

* Gate 0 (affordable): passed, 1.3× per voxel.
* Gate D (warranted, via 2.5D): failed as written; ±1 mm of context is nothing.
* The direct test (a 3D UNet, one hour): parity per patient, still improving.

The next experiment is a longer 3D run in JAX (≈20,000 steps, ~5 h on one card, or 4 cards with
a data-parallel loop) with a per-patient eval every 2,000 steps. If it clears the 2D anchor by
more than the seed noise, the Lean port (rank-generic codegen, five ops behind FD probes, the
patch loader) has its reason; if it plateaus at parity, the slice model is the right model for
this data and the 3D port stays unbuilt.
