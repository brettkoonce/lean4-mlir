# Which wavelengths travel, 2026-09-29 — the European arms on the Brazil chips

Plan: `planning/remote_sensing_wavelengths_demo.md` §4–§5. Weights: the 80-epoch EuroSAT ladder
(`runs/2026-09-29-rs-phase1/`, three seeds). Chips: `runs/2026-09-29-rs-phase2/` (its README is
how the dataset was made). Scorer: `scripts/demos/rs_score.py`; the tables here are
`scripts/demos/rs_table.py runs/2026-09-29-rs-phase3` verbatim (`tables_final.txt`). Drivers:
`score_zero_shot.sh` (Table 1–2), `ceiling.sh` (the Brazil-trained rows), `finetune.sh` (the
label-budget rows). Shapley: `../2026-09-29-rs-phase4/shapley_all_s1_<part>.log`.

## Table 1 — accuracy, mean ± sd over three seeds

EuroSAT is 10-way on the test list; a Brazil part is 7-way after the collapse (built = max of
Industrial and Residential, water = max of River and SeaLake, Highway masked out), scored on
the chips whose MapBiomas class has a European name; the savanna and plantation chips are
aside (Table 2).

| arm | EuroSAT test | Amazon, June | Cerrado, September | Cerrado, March | same answer both seasons |
|---|---|---|---|---|---|
| rgb  | 97.09 ± 0.05 | 87.37 ± 0.92 | 33.15 ± 0.46 | 37.66 ± 0.34 | 38% |
| rgbn | 98.04 ± 0.12 | 89.30 ± 1.35 | 34.61 ± 0.55 | 39.04 ± 0.53 | 41% |
| ms10 | 98.40 ± 0.08 | 96.95 ± 0.55 | 32.76 ± 2.25 | 59.21 ± 0.50 | 27% |
| all  | 98.33 ± 0.11 | 97.02 ± 0.46 | 35.39 ± 2.45 | 53.82 ± 4.27 | 24% |
| ir   | 97.94 ± 0.09 | 91.71 ± 1.48 | 29.03 ± 2.52 | 39.84 ± 3.05 | 39% |

Pasture recall in the Amazon (forest is ≥ 99.8 for every arm): rgb 74.4, rgbn 78.1, ms10 95.0,
all 95.4, ir 83.0. Macro-F1 (Amazon / Sept / March): rgb 54.6 / 29.1 / 34.5; rgbn 55.0 / 31.9 /
34.5; ms10 60.9 / 24.1 / 45.2; all 63.5 / 27.5 / 40.3; ir 52.8 / 24.7 / 36.0 — the Amazon
macro-F1 is dragged by the 41 crop, 7 grassland, 2 urban and 1 water chips.

**Grass merged** (herbaceous ∪ pasture as one class either side — a European meadow and a
grazed Cerrado paddock are both "grass"): Amazon rgb 93.9 / ms10 97.1 / all 97.2 / ir 92.8;
Cerrado September rgb **76.0** / rgbn 71.4 / ms10 64.7 / all 65.8 / ir 73.2; March rgb 56.3 /
ms10 61.8 / all 59.1 / ir 57.9. The September collapse of the 7-way table is largely the
pasture/herbaceous line: rgb calls 666 of 884 dry pastures "herbaceous"; the multispectral
arms send a share of them to "annual crop" as well (bare, dry ground reads as a ploughed field
in SWIR), which the merge does not forgive.

## Table 2 — what each arm calls the classes Europe does not have (seed mean, % of the 2,000 savanna chips)

| arm | September: herbaceous / pasture / forest / crop | March: herbaceous / pasture / forest / crop |
|---|---|---|
| rgb  | 81 / 5 / 1 / 8 | 28 / 13 / **42** / 9 |
| rgbn | 77 / 5 / 1 / 14 | 25 / 19 / 35 / 14 |
| ms10 | 63 / 30 / 0 / 6 | 6 / 34 / 39 / 11 |
| all  | 51 / 43 / 0 / 4 | 9 / 42 / 32 / 4 |
| ir   | 71 / 17 / 0 / 7 | 18 / 22 / 41 / 6 |

The one forest-plantation chip is called perennial crop by most arms in September; too few to
say more.

## Table 3 — trained or fine-tuned on Brazil labels, five folds by chip id (`brazil_all`, 7,064 chips)

| row | rgb | ms10 | all | ir |
|---|---|---|---|---|
| from scratch, all labels (the ceiling) | 94.19 ± 0.72 | 95.65 ± 0.91 | 95.83 ± 0.64 | — |
| fine-tuned from EuroSAT seed 1, 300 labels | 90.02 ± 0.99 | — | 91.02 ± 1.29 | 89.20 ± 0.84 |
| fine-tuned from EuroSAT seed 1, all labels | 94.33 ± 0.54 | — | 95.07 ± 0.71 | 94.91 ± 0.40 |

`brazil_all` is the union of the scored chips of the three parts (the cerrado twins are two
records of one chip id); a fold holds a chip id in both seasons or neither. The ceiling says
the labels are learnable to 96% and that, trained in Brazil, 13 bands still beat RGB by 1.6 —
the same margin as in Europe. Fine-tuning from Europe with 300 labels recovers ~90% for every
arm, about a point better with the 13 bands; with all labels the European start is worth
nothing over scratch.

## Shapley over five band groups, the `all` arm, seed 1 (`../2026-09-29-rs-phase4/`)

Players visible (B02 B03 B04), red edge (B05 B06 B07), NIR (B08 B8A), SWIR (B11 B12),
atmospheric (B01 B09 B10); a removed group is set to its EuroSAT training mean; value = log
p(target); exact over 32 coalitions; efficiency residual ≤ 2e-6. Share of |φ| and mean φ:

| part | visible | red edge | NIR | SWIR | atmospheric (mean φ) |
|---|---|---|---|---|---|
| EuroSAT test (300) | 33.5% | 14.9% | 24.2% | 13.2% | 14.1% (−0.07) |
| Amazon (151) | 35.0% | 12.1% | 15.0% | 22.5% | 15.5% (**−2.96**) |
| Cerrado, September (218) | 32.3% | 19.6% | 12.3% | 21.5% | 14.3% (−0.43) |
| Cerrado, March (218) | 34.8% | 16.4% | 16.3% | 20.3% | 12.1% (−1.46) |

Visible light is the largest player everywhere; SWIR's share rises from 13% at home to 22%
abroad; the three atmospheric bands have negative value in Brazil — they carry Europe's
atmosphere — which is why `ms10` (without them) is the best zero-shot arm in the Amazon and in
March, and matches `all` in-domain (ceiling 95.65 vs 95.83).

## Reading it

1. **The invisible bands travel where the classes exist on both sides.** Rondônia is forest
   and pasture; RGB finds the forest (99.9) and loses a quarter of the pasture; the ten
   invisible bands find the pasture (95). NIR alone (`rgbn`) buys four points; red edge and
   SWIR buy the other six. The arm with no visible light at all (`ir`) reaches 91.7.
2. **The dry Cerrado breaks every arm on the 7-way map,** and the merge shows why: a
   September paddock is "herbaceous" to a European net, and the biome's own class has no
   European name. That is a taxonomy problem before it is a spectral one.
3. **No arm is season-stable.** The same 3,620 chips get the same answer in September and
   March 24–41% of the time; the multispectral arms are better in both seasons, not stabler.
   The spectrum of a Cerrado pasture genuinely doubles in red and SWIR between March and
   September (the band means in the Phase 2 README), and no European chip taught that.
4. Savanna is "herbaceous" in September and 32–42% "forest" in March for every arm.
5. The gap is domain, not labels: trained in Brazil the same net reads 94–96%.

## Files

`eval_<arm>_s<k>.log`, `score_<arm>_s<k>_<part>.json`, `pair_<arm>_s<k>.json` (Table 1–2);
`ceil_<arm>_fold<k>.log`, `score_ceil_*.json`, `rs_cifar8w_<arm>_ceil_fold<k>_*` (ceiling);
`ft_<arm>_n<N>_fold<k>.log`, `score_ft_*.json`, `rs_cifar8w_<arm>_ft_fold<k>[_n300]_*` (Table 3);
`tables_final.txt`. The figure's dense windows: `../2026-09-29-rs-figure/` (`draft_s1.png` is
`demos/figures/remote_sensing_wavelengths.jpg`; the fig_* parts under data/rs/).
