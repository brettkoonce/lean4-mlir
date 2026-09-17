# ArASL, 2026-09-17 — chapter 4's CNN under a random and a capture-order-blocked split

Plan: `planning/arasl_people_watching_demo.md` (§11 is the log). Book:
`\subsection{People watching}` after Industrial inspection. This directory holds
Gate 0, the queue scripts and the scores; each arm's logs, curves and checkpoints
are in `runs/2026-09-17-arasl-<net>-<split>/` (`train_s<seed>.log`,
`*_curve.csv`, `*_params.bin`, `*_logits_{val,test}.bin`; the size-32 arm is
`train_size32_s1.log` in the cifar8w dirs).

## Gate 0 — `gate0.log`, `data/arasl/manifest.json`

- 54,049 images, 32 classes, 1,293 (`yaa`) – 2,114 (`ain`); files numbered 1..n
  with no gap in every class. 658 not 64×64 L: `kaaf` 320 + `meem` 318 at
  256×256, `haa` 10 at 768×1024, 10 RGB (`ra` 7, `seen`/`thaa`/`toot` 1 each).
- Bursts: median consecutive |Δ| 3.3 grey levels (random pair within a class
  32.9); 73.5% of consecutive pairs under 6; 14,336 chains, 3.8 frames each,
  longest 58; weakest class `al` 63.9% — premise holds (≥ 60% everywhere).
- Splits, 80/10/10 per class, seed 0: random 43,241 / 5,404 / 5,404; blocked
  43,238 / 5,416 / 5,395 (cuts snapped to chain boundaries; 0 chains straddle).
- Leak audit (test → nearest train image of any class):

  | | random | blocked |
  |---|---|---|
  | test images with a train image within 6 grey levels, 16×16 | **92.3%** | **6.4%** |
  | same, at 64×64 | 88.3% | 3.7% |
  | median nearest |Δ| (16×16) | 1.2 | 13.3 |
  | nearest train image has the same label (= 1-NN accuracy) | 95.7% | 28.6% |
  | same-class leaks: median capture-order gap | 1 file | **306 files** |

  The blocked residual is the same hand returning in a later sitting, not a
  straddled burst. The **val tenth is 37.5% leaked** under blocked (it
  neighbours train in capture order) against the test tenth's 6.4% —
  `per_class_audit.log` — which is why every blocked log shows val ≈ 92% and
  test ≈ 78%. The epoch is chosen on val regardless; stated in the section.

## Table 1 — test at the val-peak epoch

| arm | random | blocked | best-val epochs (random / blocked) |
|---|---|---|---|
| **cifar8w** 64×64, seeds 1–3 | 98.52, 98.72, 98.61 → **98.62 ± 0.10**, pooled [98.43, 98.79] | 78.78, 77.85, 77.18 → **77.94 ± 0.80**, pooled [77.29, 78.57] | 23, 27, 26 / 24, 26, 26 |
| — on leaked test images (16×16 < 6) | 99.73 (n = 4,988) | 98.46 (n = 347) | |
| — on the rest | 85.26 (n = 416) | 76.53 (n = 5,048) | |
| cifar8w at 32×32 (`size=32`), seed 1 | 98.21 [97.82, 98.53] | 71.03 [69.80, 72.22] | 28 / 20 |
| MLP 4096-512-512-32, seed 1 | 94.86 [94.23, 95.41] | 42.09 [40.78, 43.42] | 30 / 24 |
| linear 4096-32, seed 1 | 53.20 [51.87, 54.53] | 15.37 [14.43, 16.35] | 30 / 21 |
| 1-NN on 16×16 thumbnails | 95.65 [95.07, 96.16] | 28.60 [27.41, 29.82] | — |

Wilson 95% intervals; the chapter net's pooled interval is over 3 × 5,4xx trials.
Gate 1 (≥ 96% on random) was met at epoch 5 of seed 1 (96.78%). MLP and linear
were still rising at epoch 30 on the random split (best epoch = 30); they are the
bracket, not a result, and were not extended.

## Table 2 — the blocked split, per class and confused pairs (`score_cifar8w_blocked.log`, 3 seeds pooled)

Per class: `fa` 16.5, `gaaf` 36.7, `thaa` 42.9, `kaaf` 51.3, `dhad` 55.0, `ha`
57.4, `taa` 63.6, `la` 64.0, `yaa` 67.5, `saad` 68.6, `toot` 74.2, `nun` 77.0,
`zay` 79.8, `waw` 80.0, `bb` 81.6, `laam` 84.5, `meem` 85.9, `jeem` 86.2, `ain`
86.9, `dal` 88.6, `aleff` 89.7, `haa` 93.7, `al` 94.4, `ghain` 95.1, `thal` 96.7,
`khaa` 96.7, `dha` 97.6, `ra` 98.2, `ya` 99.2, `seen` 99.6, `ta` 99.8, `sheen` 100.

Rank correlation between class accuracy and the class's median nearest-train
|Δ| is −0.55 (pooled; −0.51 on seed 1): the far classes are among the worst
(`fa`, `gaaf`, `thaa`, `kaaf` have 0% leak and median |Δ| 15–17) but `al` (18.1)
and `seen` (14.5) are as far and score 94–100 — the rest is what the new hand's
shape resembles.

Pairs (per run, both directions): `fa`↔`gaaf` 60.7 (`fa`→`gaaf` 181, reverse 1);
`kaaf`↔`thaa` 56.0 (76 / 92); `taa`↔`thaa` 52.7 (34 / 124); `dhad`↔`gaaf` 39.0
(0 / 117); `fa`↔`waw` 37.7 (113 / 0); `dhad`↔`saad` 34.3; `fa`↔`saad` 31.7;
`aleff`↔`kaaf` 31.3 (0 / 94). Random split, for contrast: the top pair is
`dha`↔`ta` at 8.7 per run. The plan's guessed pairs (`ta`/`taa`, `dal`/`thal`,
`ra`/`zay`, `seen`/`sheen`, `ha`/`haa`/`khaa`) are NOT the ones that confuse.

## How it ran

- `preprocess_arasl.py --stats`: 107 s on CPU (16 s decode, 2 × 44 s audit).
  `--size 32`: 166 s, same splits (chains/split from the 64×64 images, then
  downsampled; the first cut recomputed chains on 32×32 and got a different
  blocked split — fixed before any 32 run).
- `lake build arasl-signs`: 15 s. Train step 8 ms/step alone (581,424 params,
  384 BN floats), 9–10 s/epoch with three cards busy, ~5.5 min per 30-epoch run;
  MLP 16–18 s/epoch (host-bound). ⚠ With four cards running and a numpy audit
  on the host, epochs went to 40 s for a few minutes (the box-wide stall) and
  recovered when the audit ended.
- Queues: `queue.sh` (GPU 0: cifar8w random ×3; GPU 1: cifar8w blocked ×3; GPU 2:
  mlp + linear × both splits), `queue32.sh` (GPU 3: size-32, both splits).
- Recipe = the GW demo's: Adam 1e-3, one-epoch linear warmup then cosine, wd
  1e-4, batch 64, label smoothing 0, no augmentation. Input f32 in [0, 1] (what
  the chapter's CIFAR loader feeds; the trainer prints the first image's range
  and the label range at start).

## Reproduce

```bash
./download_arasl.sh                                   # or: .venv/bin/python preprocess_arasl.py data/arasl data/arasl --stats
lake build arasl-signs
for s in 1 2 3; do CUDA_VISIBLE_DEVICES=0 .lake/build/bin/arasl-signs net=cifar8w split=random  seed=$s tag=s$s out=runs/2026-09-17-arasl-cifar8w-random;  done
for s in 1 2 3; do CUDA_VISIBLE_DEVICES=1 .lake/build/bin/arasl-signs net=cifar8w split=blocked seed=$s tag=s$s out=runs/2026-09-17-arasl-cifar8w-blocked; done
.venv/bin/python scripts/arasl_score.py runs/2026-09-17-arasl-cifar8w-blocked/arasl_cifar8w_blocked_s{1,2,3}_logits_test.bin --split=blocked --top 8 --json runs/2026-09-17-arasl/score_cifar8w_blocked.json
.venv/bin/python scripts/arasl_figure.py --score runs/2026-09-17-arasl/score_cifar8w_blocked.json --logits runs/2026-09-17-arasl-cifar8w-blocked/arasl_cifar8w_blocked_s1_logits_test.bin
```

Not run: the optional ImageNet-R34 bootstrap arm — the blocked column is a
result about the data, not a number to raise.
