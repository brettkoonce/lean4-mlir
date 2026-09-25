# 2026-09-25 — the BraTS R34 UNet on XLA: the A/B re-run, and the no-skip arm

Every published BraTS number (demos/README.md, the book's §Semantic segmentation) was from the
IREE-era runs of July. This re-runs both arms of the transfer A/B, 10 epochs each, on XLA, and
the skipless-decoder arm the book's skips table compares against.

```bash
./scripts/run_brats_r34_ab.sh 10 data/brats224            # r34 on GPU 0, scratch on GPU 1
CUDA_VISIBLE_DEVICES=0 ./.lake/build/bin/unet-brats-r34 data/brats224 10 scratch noskip
python3 scripts/brats_r34_ab.py brats_skip_r34_gpu0.log brats_skip_scratch_gpu1.log  # → ab_score.txt
lake exe brats-predict net=r34 arm=scratch,r34 best out.ppm   # the figure, best-by-val checkpoints
```

Logs here are the trainer's stderr with the XLA/ptxas chatter stripped.

⛔ **The first two launches segfaulted at the end of epoch 1**, in the first per-epoch seg eval
the trainer had run since the Lean 4.34 bump: the C side of the FFI still declared the IO world
parameter Lean no longer passes, and `lean_iree_forward_f32` wrote into its caller's frame.
Fixed in 7c14fbac; these logs are from the fixed binary.

## The A/B

Best-by-val checkpoint (mean region Dice), epoch 9 in both arms:

| arm | mIoU | WT | TC | ET |
|---|---|---|---|---|
| `r34` (ImageNet bootstrap) | 0.743 | 0.912 | 0.869 | 0.856 |
| `scratch` (He-init) | 0.741 | 0.910 | 0.867 | 0.856 |
| IREE era, `r34` | 0.742 | 0.911 | 0.870 | 0.858 |
| IREE era, `scratch` | 0.740 | 0.910 | 0.869 | 0.856 |

⭐ **Both arms reproduce the IREE numbers to within 0.003 on every metric**, and the peaks are
still a tie (+0.002 mIoU, noise at n = 1).

⚠ **The epoch-1 gap is much smaller than the July write-up said.** July: ET Dice 0.818
bootstrapped against 0.184 for the control, "still collapsed on the hard classes". Here: 0.805
against 0.742, mIoU 0.631 against 0.616; by epoch 2 the control is level (ET 0.833 vs 0.841, mIoU
0.719 vs 0.709). The transfer still buys the first epoch, but the control no longer collapses at
epoch 1, so "one epoch sooner" is now a few points at epoch 1 rather than a rescue.

Per-epoch curves, the epochs-to-target table and the guards: `ab_score.txt`.

Speed: median 252 ms/step (r34 arm) and 259 ms/step (scratch), batch 16 at 224², one RTX 4060 Ti
each, ~4.5–5.5 min per epoch including the val pass.

## The skips table

Same backbone, data and schedule (`scratch` arm), decoder with and without the encoder concat
(`noskip`); best-by-val checkpoint (epoch 10 without skips, epoch 9 with):

| decoder | mIoU | WT | TC | ET |
|---|---|---|---|---|
| upsample only (no skips) | 0.633 | 0.871 | 0.814 | 0.733 |
| `.unetUp` (skips) | **0.741** | **0.910** | **0.867** | **0.856** |
| IREE era, no skips | 0.635 | 0.872 | 0.817 | 0.733 |

+0.108 mIoU, +0.123 on ET — the IREE-era "roughly ten points, largest on ET (+0.12)" holds.
The skipless arm is still improving at epoch 10 (its best is the last epoch); median 237 ms/step.
