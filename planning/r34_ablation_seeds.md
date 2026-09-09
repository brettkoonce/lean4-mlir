# ResNet-34 recipe ablation: error bars over seeds

**Opened 2026-09-09, estimate only — nothing launched.** `planning/tour_realignment.md` §9 item 6
already lists this; the decision there (2026-09-08) is that a tour number is **mean ± 95% CI over
seeds**. This doc costs it. ⚠ Ask before launching.

## §0 What exists

* Eight arms (`full nowd nowarm nols noadam nocos noaug bare`) × two precisions = 16 runs, each
  run ONCE (`runs/2026-09-01-r34-ablation/RESULTS.md`, `…-bf16/`). The book's §5.6 figure plots
  deltas from `full` with a Wilson band of ±0.92 at n = 3,925 — a single-run resolution, not a
  spread over seeds. Three fp32 arms lost their per-epoch curves (not their results) to a
  pre-guard truncation.
* Driver: `scripts/run_r34_ablation.sh` — a work queue over the four cards with the checkpoint-tag
  trap guarded; it sets **neither `LEAN_MLIR_SEED` nor `PJRT_FFI_RESIDENT`**. `scripts/seed_sweep.sh`
  sets both (`:155-156`). That is why the sweep's ResNet-34 seeds ran in **45 min** where the
  ablation arms took **80**: same net, same recipe, host residency on vs off.
* Seed spread, ResNet-34 full recipe, `runs/2026-08-31-imagenette-n3/` five seeds: 90.04, 90.37,
  90.17, 89.68, 90.14 → mean 90.08, **sd 0.25**.

## §1 What the bars would resolve

| seeds | CI half-width per arm | CI on a difference from full |
|---|---|---|
| 3 | ±0.63 | ±0.89 |
| 5 | ±0.31 | ±0.44 |

n = 3 reproduces the current ±0.92 band and buys nothing visible. n = 5 resolves `nowd` (−0.64
fp32), `nowarm` (−1.04), `nols` (−1.50); the bf16 `nowd` (−0.10) and `nols` (−0.46) rows stay
inside the bar — the honest outcome, not a failure. Arms sharing a seed share init and data
order, so a paired difference could tighten the second column; measure, do not assume.

## §2 Cost

The existing 16 runs used the default seed (1) and can stand as seed 1.

| plan | per run | GPU-hours | wall on 4 cards |
|---|---|---|---|
| +4 seeds, resident on | 45 min | 48 | ~12 h |
| +4 seeds, as the driver runs today | 80 min | 85 | ~21 h |
| all 80 fresh, resident | 45 min | 60 | ~15 h |

Residency does not change the arithmetic; keep the existing sixteen unless one-vintage purity is
wanted for the figure.

## §3 Work (~half a day)

1. Seeds in the driver: `SEEDS` loop, `LEAN_MLIR_SEED=$seed`, `PJRT_FFI_RESIDENT=1`,
   `LEAN_MLIR_CKPT_TAG=<arm>_<prec>_s<seed>`, per-seed log names, the overwrite guard kept. Or a
   `SUITE=r34ablation` in `seed_sweep.sh`, which has all three already and the AER watchdog.
2. Summarizer: per arm mean, t-based 95% interval, Δ vs full with its own interval; emits the
   `RESULTS.md` table and the pgfplots coordinates.
3. Figure: explicit x error bars on the two dot plots, drop the Wilson band, rewrite the prose
   around "a point inside the band is not resolved" to say what a bar means; both panels.
4. `RESULTS.md`, the chapter's two single-run references, `tour_realignment.md` §9.6.

Gates: `lake exe blueprint-checkdecls blueprint/lean_decls` after the book edit; the PDF renders;
every seed's OK line present in the sweep log before the summarizer runs.
