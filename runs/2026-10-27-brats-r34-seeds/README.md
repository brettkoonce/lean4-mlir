# 2026-10-27 — BraTS, the three 10-epoch arms at seeds 1 and 2 (planning/part2_ship_audit.md §2.3)

`runs/2026-09-25-brats-r34-xla/README.md` is the unset seed (0) of the R34-bootstrap / He-init A/B
and of the skipless-decoder arm; its peaks were "still a tie (+0.002 mIoU, noise at n = 1)" and
"+0.108 mIoU, +0.123 on ET" for the skips. These are the same three arms — `unet-brats-r34`,
`data/brats224`, 10 epochs, batch 16 at 224², the 09-25 binary — at `LEAN_MLIR_SEED=1` and `2`
(`LeanMlir/Train.lean:651–654`: the He-init, the per-epoch shuffle and the augmentation draws;
the R34 bootstrap bytes are the same on every seed), one card per seed, one arm at a time
(cards 1 and 2 of the grind after the VisDrone aff30 seeds; `queue.log`: r34 52 min, scratch
59–62, noskip 51–54, beside the other chains). `tag=s<k>` keeps each seed's checkpoints apart and
puts the seed into the log's `arm:` line (`r34_s1`, `scratch_s1`, `scratch_noskip_s1`), which the
scorer's arm guard reads.

```
LEAN_MLIR_SEED=$s LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$G \
  .lake/build/bin/unet-brats-r34 data/brats224 10 r34 tag=s$s             # run.sh: r34 | scratch | scratch noskip
python3 scripts/probes/brats_r34_ab.py brats_r34_s$s.log brats_scratch_s$s.log            # ab_score_s$s.txt
python3 scripts/probes/brats_r34_ab.py brats_noskip_s$s.log brats_scratch_s$s.log \
  --names "scratch noskip,scratch" --arm scratch                                          # noskip_score_s$s.txt
```

Both guards (distinct consecutive eval rows; arm/tag agreement) pass on all four scorings. The
logs are the trainer's stderr, unstripped (the XLA compile lines are in them).

## The A/B, three seeds

Best-by-val checkpoint as the 09-25 README chose it (the epoch with the highest mean region
Dice); seed 0 is the 09-25 run.

| arm | seed | best epoch | mIoU | WT | TC | ET |
|---|---|---|---|---|---|---|
| `r34` (ImageNet bootstrap) | 0 | 9 | 0.743 | 0.912 | 0.869 | 0.856 |
| | 1 | 7 | 0.740 | 0.913 | 0.870 | 0.853 |
| | 2 | 8 | 0.739 | 0.911 | 0.862 | 0.858 |
| | **mean ± sd** | | **0.741 ± 0.002** | **0.912 ± 0.001** | **0.867 ± 0.004** | **0.856 ± 0.003** |
| `scratch` (He-init) | 0 | 9 | 0.741 | 0.910 | 0.867 | 0.856 |
| | 1 | 7 | 0.736 | 0.910 | 0.866 | 0.854 |
| | 2 | 10 | 0.734 | 0.907 | 0.858 | 0.855 |
| | **mean ± sd** | | **0.737 ± 0.004** | **0.909 ± 0.002** | **0.864 ± 0.005** | **0.855 ± 0.001** |
| `r34` − `scratch`, paired | 0 / 1 / 2 | | +0.002 / +0.004 / +0.005 | +0.003 / +0.002 / +0.005 | +0.003 / +0.004 / +0.004 | −0.000 / −0.001 / +0.003 |

Epoch 1 and epoch 2, the "transfer buys the first epoch" rows:

| seed | r34 e1 mIoU / ET | scratch e1 mIoU / ET | e1 ET gap | r34 e2 mIoU / ET | scratch e2 mIoU / ET |
|---|---|---|---|---|---|
| 0 | 0.631 / 0.805 | 0.616 / 0.742 | +0.064 | 0.709 / 0.841 | 0.719 / 0.833 |
| 1 | 0.648 / 0.784 | 0.651 / 0.819 | −0.035 | 0.721 / 0.837 | 0.714 / 0.831 |
| 2 | 0.690 / 0.816 | 0.687 / 0.818 | −0.003 | 0.649 / 0.825 | 0.632 / 0.818 |

## The skips table, three seeds

Same `scratch` arm, decoder with and without the encoder concat; the skipless arm's best is
epoch 10 on every seed (still improving at the end of the schedule, as at seed 0).

| decoder | seed | mIoU | WT | TC | ET |
|---|---|---|---|---|---|
| upsample only (`noskip`) | 0 | 0.633 | 0.871 | 0.814 | 0.733 |
| | 1 | 0.637 | 0.871 | 0.819 | 0.732 |
| | 2 | 0.632 | 0.870 | 0.816 | 0.728 |
| | **mean ± sd** | **0.634 ± 0.003** | **0.871 ± 0.001** | **0.816 ± 0.003** | **0.731 ± 0.003** |
| `.unetUp` (skips) | mean ± sd | 0.737 ± 0.004 | 0.909 ± 0.002 | 0.864 ± 0.005 | 0.855 ± 0.001 |
| skips − no skips, paired | 0 / 1 / 2 | +0.108 / +0.099 / +0.102 | +0.039 / +0.039 / +0.036 | +0.053 / +0.048 / +0.042 | +0.123 / +0.122 / +0.128 |

## What it says

- **The bootstrap's peak is a small lead with the same sign on every seed, not a tie.** mIoU
  +0.002 / +0.004 / +0.005 (mean +0.004, sd of the paired difference 0.0015), WT and TC the same
  size; ET is nothing (−0.000 / −0.001 / +0.003). It is the size of one arm's own seed spread
  (0.002–0.004), so three seeds put it at the edge of what they can resolve: "a few thousandths,
  consistently" is the honest sentence, and "a tie" and "the bootstrap wins" are both too strong.
- **"The transfer buys the first epoch" does not survive the seeds.** Seed 0's epoch-1 ET gap of
  +0.064 is −0.035 at seed 1 (the control ahead) and −0.003 at seed 2; mIoU at epoch 1 is
  +0.015 / −0.004 / +0.003. The epochs-to-target table (`ab_score_s*.txt`) reaches every 50 / 80 /
  90 % target at the same epoch on both arms at seed 2 and differs by one epoch on one row at
  seeds 0 and 1. The 09-25 README had already cut July's "collapsed on the hard classes" down to
  "a few points at epoch 1"; at three seeds it is zero, and the sample-efficiency story the scorer
  was written to measure is not there at this data size. What the bootstrap buys is the 0.004 at
  the end.
- **Seed 2 is the rough seed on both arms.** Its epoch-2 val drops 0.04–0.06 mIoU below epoch 1
  on `r34` and `scratch` alike (0.690 → 0.649, 0.687 → 0.632), and `scratch` dips again at epoch
  5 (0.719 → 0.683); seeds 0 and 1 move by about 0.01 between consecutive epochs after epoch 2. The
  dips are shared by the two inits at that seed, so they come from the shuffle and augmentation
  draws, not the weights — one more reason the single best-by-val row carries ~0.005 of noise on
  its own.
- **The skips number is a measurement now:** +0.103 ± 0.005 mIoU and +0.124 ± 0.003 ET at three
  seeds, the skipless arm's sd 0.003 on every metric. "Roughly ten points, largest on ET" holds
  exactly as written.
- For the book (`tab:brats_dice`, which prints the seed-0 `scratch` row 0.741 / 0.910 / 0.867 /
  0.856): the row becomes 0.737 ± 0.004 / 0.909 ± 0.002 / 0.864 ± 0.005 / 0.855 ± 0.001 (three
  seeds, best-by-val), or keeps the seed-0 numbers marked as one of three. The A/B paragraph
  loses its epoch-1 claim and gains the paired +0.004.

Files: `brats_{r34,scratch,noskip}_s{1,2}.log`, `ab_score_s{1,2}.txt`, `noskip_score_s{1,2}.txt`,
`queue.log`, `run.sh`. Checkpoints (`.lake/build/…__r34_s1_params.bin` etc.) stay on ares.
