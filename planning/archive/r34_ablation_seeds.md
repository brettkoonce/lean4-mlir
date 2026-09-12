# ResNet-34 recipe ablation: error bars over seeds

**Opened 2026-09-09. ✅ CLOSED 2026-09-12** — both precisions seeded, and both §5.6 figures
carry paired 95% bars. Driver `scripts/run_r34_ablation.sh`, summarizer
`scripts/r34_ablation_ci.py`.

| half | runs | results |
|---|---|---|
| bf16 | 8 arms × 5 seeds | `runs/2026-09-11-r34-ablation-bf16-seeds/RESULTS.md` |
| fp32 | 8 arms × 5 seeds | `runs/2026-09-12-r34-ablation-fp32-seeds/RESULTS.md` |
| step-matched SGD | `sgd10` × 5 seeds | same directory, as a ninth arm |

**What the seeds settled.**

* §1 predicted n = 5 would resolve `nowd`, `nowarm`, `nols`. Two of the three: `nowarm`
  (−0.55 ± 0.46) and `nols` (−0.69 ± 0.42) resolve, **`nowd` does not** (−0.02 ± 0.61) — and its
  effect collapsed from the single run's −0.64 to −0.02. Unresolved in bf16 too (+0.14 ± 0.45),
  so the chapter no longer claims weight decay earns anything on this recipe.
* §2b's opening question — the cosine arm's spread — is answered: fp32 sd **1.93** against
  bf16's 2.18, so the width belongs to the arm and is not a bf16 artifact.
* Every fp32 Δ falls inside its bf16 interval and every bf16 Δ inside the fp32 one.
* ⭐ The endpoints ladder REVERSED. Step-matched SGD finishes **ahead** of momentum, paired
  **+0.403 ± 0.162**, all five seeds agreeing in sign, where single runs had put it 0.31 behind
  and called the pair indistinguishable. `sgd10` was the only new configuration: the ladder's
  other two rungs are the `full` and `noadam` arms reused, not re-run.

⚠ `run_r34_ablation.sh` refuses a leftover checkpoint — four seed-1 arms silently resumed one on
2026-09-11 and had to be redone. ⚠ Its built-in `PJRT_PLUGIN` default
(`~/.venv-cuda/…/xla_cuda13/`) does not exist on this box; pass the path §2b names. ⚠⚠ The driver
blocks on GPU occupancy but **not** on host load, so it will launch happily into a CPU-saturated
box: with an emulated build running alongside, arms took 7,061 s against a clean 2,800 s. The
trainers draw only ~0.5 core each, so the penalty is run-queue latency, not CPU share.

Originally: `planning/tour_realignment.md` §9 item 6 already lists this; the decision there
(2026-09-08) is that a tour number is **mean ± 95% CI over seeds**. This doc costed it.

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

## §2b The fp32 half — the handoff (2026-09-11) — ✅ RAN 2026-09-12, 7.7 h, 40/40 clean

One command, the same one the bf16 half ran; ~45–50 min per run resident, 40 runs over four
cards ≈ 8 h. The driver waits for each card to be idle, refuses to overwrite a log, and refuses
a leftover `abl-fp32-<arm>-s<seed>` checkpoint (the 2026-09-01 fp32 tags had no seed suffix, so
they do not collide). `DRY_RUN=1` prints the queue and exits.

    PREC=fp32 SEEDS="1 2 3 4 5" OUT=runs/$(date +%F)-r34-ablation-fp32-seeds \
      PJRT_PLUGIN=$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so \
      bash scripts/run_r34_ablation.sh
    python3 scripts/r34_ablation_ci.py runs/<that dir> --write     # RESULTS.md + the pgfplots block

Then in `blueprint/src/content.tex` §5.6: give the fp32 axis the same `error bars` `\addplot`
the bf16 axis has (the block in RESULTS.md, minus the `bare` row), drop its Wilson band, retitle
it "fp32, 5 seeds", rewrite the marks paragraph for two seeded panels, and delete the
`[TODO: more fp32 samples.]` line. `leanblueprint pdf` to check the page (~10 s here).
⚠ Five FRESH fp32 seeds, not seed 1 + four: the 2026-09-01 fp32 logs did not survive either.
The question to answer first when it lands: the cosine arm's spread — bf16 gave sd 2.18.

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
