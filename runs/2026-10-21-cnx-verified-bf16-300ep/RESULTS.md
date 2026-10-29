# ConvNeXt-T, the official recipe (facebookresearch/ConvNeXt's own), bf16 — verified path, 300 epochs

**✅ 82.146 % top-1 / 95.910 % top-5** (41,073 / 47,955 of 50,000, EMA shadow, f32 eval forward),
epoch 300/300, 95 % CI 81.81–82.48.
* Landed 2026-10-28 22:38:01 UTC.
* **102.66 h** wall clock from the first supervisor start. That includes the chain's forced resume
  test (one stop + relaunch mid-epoch 3); there were no other restarts.
* Zero crashes, zero thermal rests, EDAC 0 / 0 throughout, no NaN.

The verified half of the official-recipe ConvNeXt-T pair (`planning/recipe_prep.md` §4; ch 8's marker).
Its JAX half is `runs/2026-10-21-cnx-full-jax-bf16-300ep/` (81.990 / 95.87), which `summarize.sh` reads
in place.
* Job `scripts/jobs/cnx-default-emabf16-4gpu.conf`, variant `emaaccdp16xwxdroperfbf16`, unit
  `cnx-off-ver`. Chain-launched by `chain_after_jax.sh` 2026-10-24 15:58:39 UTC on the 4× 3060 box
  (mars), after the JAX half's ✅ COMPLETE; tree at `b7af4de3`, fresh checkpoint name.
* Render `verified_mlir/convnextin_emaaccdp16xwxdroperfbf16_train_step.mlir`: 5 regions
  [θ|m|v|G|ema], 935 outputs. Eval forward `convnextin_erf_fwd` (sharded over 4 replicas).

## ⭐ The pair: a tie

| | verified (this run) | JAX reference | Δ |
|---|---|---|---|
| top-1 / top-5, e300, EMA | **82.146 / 95.910** | 81.990 / 95.87 | **+0.16 / +0.04** |
| mean of e276–300 | 82.136 / 95.934 | 81.933 / 95.892 | +0.20 / +0.04 |

The endpoint Δ is under half of one Wilson half-width (±0.34). Per 50-epoch window (`summarize.sh`,
both runs score every epoch):

| epochs | verified top-1 | JAX top-1 | Δtop-1 | Δtop-5 | Δtrain loss |
|---|---|---|---|---|---|
| 1–50 | 0.53 | 1.10 | −0.56 | −1.32 | −0.125 |
| 51–100 | 47.34 | 50.59 | −3.25 | −3.09 | −0.042 |
| 101–150 | 77.05 | 76.68 | +0.37 | +0.16 | −0.031 |
| 151–200 | 80.00 | 79.76 | +0.24 | +0.16 | −0.018 |
| 201–250 | 81.44 | 81.15 | +0.29 | −0.00 | −0.016 |
| 251–300 | 82.08 | 81.84 | +0.24 | +0.04 | −0.015 |

* **The early deficit is EMA lag, not an offset.** Both runs score a 0.9999 shadow that starts at the
  init and moves once per optimizer step (312 an epoch), so both sit near chance to ~e50 and then climb
  steeply. On that climb the verified shadow trailed JAX's by ~2.4 epochs (e60 −6.61, e80 −2.41,
  e90 −0.99, e100 −0.24). It drew level at e103 and was ahead on every epoch after it (203 ahead, 97
  behind, the last behind at e102).
* After the crossover the lead settles at +0.2 to +0.4 and narrows toward the end. Train loss runs
  0.015–0.03 lower in every window after e100.
* The one-seed caveat holds: which RNG stream (init, augmentation, Mixup λ) put this run's EMA climb
  two epochs behind is not separated.

### McNemar, per image (`scripts/demos/mcnemar.py`)

The JAX run's final `.state.npz` was re-scored through `jax/scripts/eval_full50k.py`
(`EVAL_DTYPE=f32 PROTOCOL=train BATCH=256`, `DUMP_CORRECT`; logs `rescore_{ema,live}.log` in the
JAX folder). Its EMA arm reproduces 40,994 against the logged 40,995. This run's EMA arm is its own
e300 bitmap. Its live arm is `score-checkpoint convnext-in` with `LEAN_MLIR_REGION=live` (region 0
of 5; `rescore_live.log`).

| arm | JAX | verified | only JAX right | only verified right | p (exact) |
|---|---|---|---|---|---|
| EMA (scored) | 40,994 (81.988) | 41,073 (82.146) | 1,647 | 1,726 | **0.18** |
| live weights | 40,961 (81.922) | 40,990 (81.980) | 1,801 | 1,830 | **0.64** |

Neither arm separates the two checkpoints. The EMA is worth +0.07 (JAX) and +0.17 (verified) at the
end of this recipe; the bitmaps are local only.

## Against the 09-18 verified run (the global-256 form of the recipe)

| | e300 top-1 / top-5 | Δ vs its JAX |
|---|---|---|
| verified, this run (official recipe, batch 4096) | 82.146 / 95.910 | **+0.16** |
| verified, `runs/2026-09-18-cnx-verified-300ep/` | 81.302 / 95.608 | −0.23 (vs 81.53) |
| JAX, `runs/2026-10-21-cnx-full-jax-bf16-300ep/` | 81.990 / 95.87 | |
| JAX, the 09-04 run | 81.53 / 95.50 | |

The verified side gained +0.84 and the JAX side +0.46 on the official recipe. The two changes are not
separated (one seed each, and the recipe changed several rows at once on both paths; see the JAX
RESULTS.md).

## Verified from the run's own output

* `train 1281167, test 50000; bs 64, ConvNeXt-T (ImageNet-1k) emaaccdp16xwxdroperfbf16 (cosine→0.000001+warmup 20ep, baseLR 0.004000)`
* `▸ GRADIENT ACCUMULATION: k = 16, blob region 3 of 5 [θ|m|v|G|ema]. Micro-batch 256 x 16 = EFFECTIVE BATCH 4096. 4992 micro-batches/epoch = 312 updates/epoch`
* `▸ EMA: region 4 of 5 [θ|m|v|G|ema], shadow starts AT the weights, decay 0.999900 constant from step 0 (timm ModelEma …)`; it advances on apply micro-batches only
* `▸ STOCHASTIC DEPTH: 18 drop sites` (keeps 1.0 … 0.9)
* `▸ INIT: ConvNeXt _init_weights (σ=0.02 on every conv AND the head; biases 0, LN γ 1, LayerScale γ 1e-6)`
* `▸ MIXUP/CUTMIX: … SHIM_MIX=both; ON` (λ from numpy, distributional agreement only)
* `DATA-PARALLEL: 4 replicas x bs 64 = global batch 256, 4992 steps/epoch`
* the relaunch: `▸ resuming from checkpoint at epoch 2` (1 of 1)

⚠ The header still reads `He init`: the stale unconditional label `Train.lean` documents; the
`▸ INIT` banner is the true one.

## The forced resume test (`CHAIN_EVENT`)

This variant is the first 5-region accumulating ConvNeXt blob, and it had never been resumed. The
chain stopped the trainer mid-epoch 3 (SIGTERM at micro-step ~1000/4992, 2026-10-24 16:43:55 UTC),
relaunched the same supervisor and watchers, and checked the relaunch: `✅ resume test: the relaunch
resumed from the epoch-2 checkpoint`. The partial epoch 3 was lost and retrained (e3 took 1,543 s on
the epoch clock). The curve shows no step at e3, and the run landed tied with JAX.

## Pace and loaders

* **1,232 s an epoch** (min 1,206, max 1,543 at the resume) over 299 gaps, eval of all 50,000 every
  epoch included. The conf's ~109 h was the ares estimate; 102.7 h landed on the 3060 box. The JAX
  half took 74.07 h (888.7 s an epoch), so the verified path ran at 1.39× its time.
* Loaders were healthy throughout: the peak arena-class memory was 4.09 GiB (e55), and no producer
  passed 6 GiB. `▸ SHIM RESPAWN` replaced one producer every 10 epochs (29 respawns); `fault_watch`
  never fired.

## Artifacts

* Checkpoint, outside git: `.lake/build/convnextin_emaaccdp16xwxdroperfbf16_ckpt_xla.bin{,.epoch}`
  (marker 300, 571.8 MB, all five regions).
* Per-image top-1 bitmaps for all 300 epochs plus `ver_e300_live.bin`, `bitmaps/` (local only;
  `runs/**/*.bin` is gitignored).
* `attempt.log` / `full.log` / `master.log`, `log_ts.log`, `epoch_clock.tsv`, `loader_rss.tsv`,
  `edac.tsv`, `CHAIN_EVENT`, `WATCH_EVENT`, `rescore_live.log`, the five watcher scripts and
  `chain_after_jax.sh`.
* `cnx_verified_curve.csv`, written by `summarize.sh`.

## Reproduce

    bash runs/2026-10-21-cnx-full-jax-bf16-300ep/summarize.sh
    bash runs/2026-10-21-cnx-verified-bf16-300ep/summarize.sh
