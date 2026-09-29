# EuroSAT ladder, 2026-09-29 — chapter 4's CNN under five band-subset stems

Plan: `planning/remote_sensing_wavelengths_demo.md` §3–§4 (Gate 1). Data: `data/rs/eurosat_*.bin`
from `scripts/datasets/download_rs.sh` (Gate 0 in `runs/2026-09-29-rs-gate0/gate0.log`). Trainer:
`lake exe rs-bands` (`demos/MainRsBands.lean`): chapter 4's CIFAR-CNN8-wide-BN with a `C`-plane stem
(`cifar8w`, 0.57M parameters at C = 3), Adam 1e-3, one-epoch warmup, cosine to zero, weight decay
1e-4, batch 64, label smoothing 0, and the dihedral augmentation (`F32.dihedralGather`: each
training chip under a random one of the eight symmetries of the square). One 4060 Ti; an
80-epoch arm is 20,240 steps in ~5–6 min (3 s per epoch, the 13-plane host gather included).

## The table — EuroSAT test (5,400 chips, torchgeo's list), 10-way accuracy, 80 epochs

| arm | planes | seed 1 | seed 2 | seed 3 | logits |
|---|---|---|---|---|---|
| rgb  | B04 B03 B02 | 97.06 | 97.06 | 97.17 | `rs_cifar8w_rgb_s<k>_logits_eurosat_test.bin` |
| rgbn | + B08 | 97.89 | 98.19 | 98.06 | |
| ms10 | the 10 m + 20 m bands (B02–B08, B8A, B11, B12) | 98.52 | 98.35 | 98.33 | |
| all  | 13 | 98.20 | 98.31 | 98.48 | |
| ir   | B05 B06 B07 B08 B8A B11 B12 — no visible light | 98.00 | 97.81 | 98.02 | |

Published tie: Helber et al. 2019, Table IV, ResNet-50 on a random 80/20 split — RGB 98.57,
colour-infrared (B08/B04/B03) 98.30, SWIR 97.05. The 10 m + 20 m arm of a 0.57M-parameter
net from scratch lands on the pretrained ResNet-50's RGB number; every invisible-band arm
beats RGB by ~1 point, and the arm with no visible light at all sits within a point of it.

## How the schedule was chosen (`ep20/`, `noaug/`, `e40/`, `e80/`, `e160/`)

The GW trainer this copies has no augmentation and the plain ladder memorised (train loss
0.03 at 20 epochs): rgb 92.24 / rgbn 95.33 / ms10 96.85 / all 95.96 / ir 95.09 (`noaug/`,
seed 1). With the dihedral gather, 20 epochs, three seeds (`ep20/`): rgb 94.30 / 94.39 / 94.30,
rgbn 96.46 / 96.41 / 96.50, ms10 97.04 / 97.19 / 97.09, all 97.30 / 97.09 / 97.13, ir 96.20 /
96.02 / 96.06 — the same ordering, ±0.2 across seeds, but RGB was not converged: rgb at 40 /
80 / 160 epochs = 96.20 / 97.19 / 97.30 (train loss 0.16 → 0.03 → 0.007), all at 40 / 80 =
97.70 / 98.26. So the ladder's schedule is 80 epochs for every arm — RGB has converged there
(+0.1 at 160) and the gap that remains is spectrum, not schedule. The 20-epoch numbers stay
here as the record of why.

## Reading it

- In Europe, in-domain, the ten invisible bands are worth about a point to a small net
  trained from scratch, and the bands a camera cannot see are enough on their own
  (`ir` 98.0). Helber's pretrained ResNet-50 ordering (RGB best) does not transfer to a net
  that has not seen ImageNet: RGB's advantage there was the pretraining, not the spectrum.
- NIR alone (`rgbn`) buys ~1 point over RGB; the red-edge / SWIR bands of `ms10` buy the rest.
- The plane lists are in `RsBands.planesOf`; the order was derived from the chips at Gate 0
  (B08's most-correlated plane is 12 = B8A, B10 ≈ 11 DN), not from the documentation.

## Files

`<arm>_s<k>.log` (per-epoch loss and val accuracy), `rs_cifar8w_<arm>_s<k>_{params,bn_stats}.bin`,
`_curve.csv`, `_logits_eurosat_{val,test}.bin` and, after Phase 3, `_logits_{amazon_dry,
cerrado_dry,cerrado_wet}.bin`; `ladder80.sh` / `ladder80.log` (the run script and its log);
`relaunch_aug.sh` (the waiter that stopped the plain ladder after seed 1 and rebuilt with the
gather). Scored by `scripts/demos/rs_score.py <logits> --part eurosat_test`; the seed table by
`scripts/demos/rs_table.py runs/2026-09-29-rs-phase3` once Phase 3's JSONs exist.
