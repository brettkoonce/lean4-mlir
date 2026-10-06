# ViT-Ti (DeiT-Ti), bf16 — verified path, 300 epochs

**✅ 72.416 % top-1 / 91.212 % top-5** (36,208 / 45,606 of 50,000, EMA shadow), epoch 300/300,
95 % CI 72.02–72.81. Landed 2026-10-06 07:05:31 UTC. **64.92 h** wall, `attempt 1`, zero restarts,
zero thermal rests, EDAC 0 / 0 throughout.

This is the verified half of the post-C6 ViT-Ti pair (`planning/vit_parity_todo.md`).
* **Job:** `scripts/jobs/vit-default-emabf16-4gpu.conf`, variant `emadp128x4wxclipdropeps0000001bf16`,
  unit `vit-verified`.
* **Render:** `verified_mlir/vitin_emadp128x4wxclipdropeps0000001bf16_train_step.mlir`. That is
  4 regions [θ|m|v|ema], 831 operands and 200 all-reduces. It is the 72.35 run's layout, with
  LayerNorm ε 1e-6 at all 75 sites.
* **Eval partner:** `vitin_emadp128x4wxclipdropeps0000001bf16_fwd`, ε 1e-6 at all 25 sites, fp32,
  sharded over 4 replicas.
* **Launch:** chain-launched 2026-10-03 14:10:35 UTC by `chain_after_jax.sh`, 50 s after the JAX
  run's ✅ COMPLETE.
* **Partner run:** `runs/2026-10-01-vit-jax-bf16-300ep/` (72.072 / 91.000), which `summarize.sh`
  reads in place.

## ⭐ The pair: a TIE at the recipe level, with the verified checkpoint measurably ahead

| | verified (this run) | JAX reference | Δ |
|---|---|---|---|
| top-1 / top-5, e300, EMA (as logged) | **72.416 / 91.212** | 72.072 / 91.000 | **+0.34 / +0.21** |
| mean of e280, 285, …, 300 | 72.35 / 91.15 | 71.95 / 90.92 | +0.39 / +0.23 |
| e300 **live** weights (DeiT's protocol) | 72.148 / 91.108 | 71.860 / 90.878 (fp32) | +0.29 / +0.23 |

The endpoint Δ is inside one Wilson half-width (±0.39). The per-image test is the better
question, though, because both runs score the same 50,000 images:

| pairing (per-image, e300) | only JAX right | only verified right | McNemar exact p |
|---|---|---|---|
| EMA vs EMA (the run's own bitmap; JAX bf16 eval) | 2,257 | 2,438 | **0.009** |
| EMA vs EMA (the run's own bitmap; JAX fp32 eval) | 2,252 | 2,429 | **0.010** |
| live vs live (re-score; both fp32) | 2,417 | 2,561 | **0.043** |

The two bitmaps agree on 90.6 % of images. A misaligned order would agree on ~60 %, so the
pairing is real.

⇒ **These two checkpoints differ, and the verified one is better on this val set.** That is a
statement about these two trained models, not the recipe. With one seed per arm, "the verified
path trains ViT-Ti better" is not shown. What is shown: matched row for row, the verified path does
**not** trail its reference. That is the question the pair was rerun to settle.

The lead was built during training and holds steady (`summarize.sh`, 25-epoch windows, top-1 on
the JAX run's scored epochs):

| epochs | Δtop-1 | Δtop-5 | Δ**train** loss |
|---|---|---|---|
| 1–25 | −0.11 | −0.30 | +0.011 |
| 26–50 | +0.20 | −0.05 | +0.002 |
| 51–75 | +0.40 | +0.30 | −0.012 |
| 76–100 | +0.35 | +0.31 | −0.022 |
| 126–150 | +0.42 | +0.32 | −0.020 |
| 201–225 | +0.28 | +0.20 | −0.031 |
| 251–275 | +0.44 | +0.28 | −0.031 |
| 276–300 | +0.39 | +0.23 | −0.032 |

Tied through e35 (e5 13.89 vs 13.99, e25 48.27 vs 48.24, e35 52.95 vs 52.85), then +0.2–0.5 from
e40 to the end. It never reverses. The verified **train** loss is lower from e51 on, by a margin
that widens to −0.03. So the verified side fits the training stream better AND generalises better.
This is the mirror image of MNv2 (`runs/2026-09-27-mnv2-verified-bf16-350ep/`: +0.02 train loss,
−0.5 top-1).

What still differs between the arms (`vit_parity_todo.md` §2):
* **G2 (mixup/cutmix granularity):** one λ / partner / box per 128-row replica shard here, one
  over the global 512 on JAX. This is the leading candidate for a train-loss offset, because it
  changes the mixed-target stream itself.
* **G3 (RNG streams):** different random streams on the two sides.
* **G4 (init sampler):** Bates-3 here vs Gaussian on JAX.
* **G5 (repeated-aug placement):** different placement of the repeated-augmentation copies.

**G1 (eval precision) is now measured** and is worth 0.01, so it is closed.

## Against what came before

* The previous verified run, 72.350 / 91.216 (`runs/2026-09-08-vit-verified-300ep-det0/`, §9.6),
  differs from this one in:
  * the pre-C6 augmentation
  * ε 1e-5
  * no LR floor
  * CLS/pos-embed initialised to 0
  * the cosine one step ahead

  This run: +0.07 / −0.00.
* DeiT-Ti paper: 72.2 / 91.1, on the live model. This run's live weights: **72.148 / 91.108**, i.e.
  −0.05 / +0.01. This run's EMA beats the paper's number by +0.22.
* **EMA is worth +0.26** here (live 72.148 → EMA 72.412 on the re-score, McNemar p = 0.0004). On
  JAX it is worth +0.23. For comparison: B0 +0.82, ConvNeXt +0.03.

## Verified from the run's own output

* `cosine→0.000010+warmup 5ep, baseLR 0.000500`
* `▸ INIT: timm/DeiT (σ=0.02 weights + CLS/pos-embed, patch-embed on PyTorch Conv2d default)`
* `▸ EMA: region 3 of 4 [θ|m|v|ema], … decay min(0.999960, (1+t)/(10+t))` (the same warmup-corrected
  EMA as the JAX trainer's `ema_update`)
* `▸ STOCHASTIC DEPTH: 24 drop sites`
* `DATA-PARALLEL: 4 replicas x bs 128 = global batch 512, 2502 steps/epoch`
* train step RESIDENT (800 tensors, 87.2 MB), forward RESIDENT (200 tensors)
* `SHIM RESPAWN: one producer every 10 epoch(s)` (29 respawns)
* step-0 loss 6.937

⚠ The header line still reads `He init`. It is the stale unconditional label `Train.lean` already
documents (:2008), and the `▸ INIT` banner is the true one.

## Re-score (2026-10-06, `score-checkpoint vit-in`, 4 replicas)

`LEAN_MLIR_REGION=ema` gives 36,206 / 45,607. The training run logged 36,208 / 45,606 for the same
blob, so 4 images flip. The equality gate is off by 4 images (0.004 pt). That is the class `scripts/gates/sharded_eval_gate.sh`
documents (9 images on the 2026-09-18 ViT checkpoint, from XLA's per-process compile, not
reproducible): each process compiles its own 4-replica executable. It is not a factoring
difference. `LEAN_MLIR_REGION=live` gives 36,074 / 45,554. The bitmaps for both are in `bitmaps/`
as `ver_e300_{ema,live}.bin`.

## Loaders and pace

* **Pace:** **779 s/epoch** mean over 299 gaps (758–839; one epoch over 800, e246). Eval of all
  50,000 runs every epoch. That is ~299 ms/step, 8 % over the 2026-09-26 probe's 277, which matches
  the 2026-10-01 re-probe (298). The conf's `~61–62 h` was the old probe; 64.92 h landed.
* **Loaders:** the respawn retired a producer every 10 epochs. No producer's arena-class memory
  passed 4.91 GiB (peak at e115), so the B0/MNv2 runaway never started. `fault_watch` never fired.
* **Swap:** host swap rose from 1.5 to 5.8 GiB over e1–7, then drifted back to 3.8 GiB by e299. Free memory
  held at ~138 GiB and there was no pace effect.

## Artifacts

* Checkpoint `.lake/build/vitin_emadp128x4wxclipdropeps0000001bf16_ckpt_xla.bin{,.epoch}` (marker
  300, 91.5 MB, all four regions).
* Per-image top-1 bitmaps for all 300 epochs, `bitmaps/vit_emadp128x4wxclipdropeps0000001bf16_e{1..300}.bin`,
  plus the e300 re-score's `ver_e300_{ema,live}.bin`. LOCAL ONLY.
* `attempt.log` / `full.log` / `master.log`, `log_ts.log`, `epoch_clock.tsv`, `loader_rss.tsv`,
  `edac.tsv`.
* The five watcher scripts, `chain_after_jax.sh`, and `restart_after_ckpt.sh` (never needed).
* `vit_verified_curve.csv`, written by `summarize.sh`.

## Book rows this bears on (not edited here)

§9.6's pair (72.31 JAX / 72.35 verified) and its curves, and K1. This pair supersedes both arms,
matched row for row: **72.072 / 91.000 (JAX) vs 72.416 / 91.212 (verified)**, live 71.86 / 72.15.

## Reproduce

    bash runs/2026-10-01-vit-jax-bf16-300ep/summarize.sh
    bash runs/2026-10-01-vit-verified-bf16-300ep/summarize.sh
    scripts/demos/mcnemar.py runs/2026-10-01-vit-jax-bf16-300ep/bitmaps/jax_e300_ema_bf16.bin \
      runs/2026-10-01-vit-verified-bf16-300ep/bitmaps/vit_emadp128x4wxclipdropeps0000001bf16_e300.bin
