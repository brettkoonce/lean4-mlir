# MobileNetV2, bf16, sync-BN — verified path, 350 epochs

**✅ 71.124 % top-1 / 90.036 % top-5** (35,562 / 45,018 of 50,000, raw weights), epoch 350/350,
95 % CI 70.73–71.52. Landed 2026-10-01 14:16:26 UTC. **55.79 h** wall, `attempt 1`, zero restarts,
zero thermal rests, EDAC 0 / 0 throughout.

The verified half of the MobileNetV2 350-epoch pair (`planning/imagenet_parity.md` §7 R4). Job
`scripts/jobs/mnv2-default-4gpu.conf`, variant `rmsdp64wxdols0eps0001bf16` (render
`verified_mlir/mobilenetv2in_rmsdp64wxdols0eps0001bf16_train_step.mlir`, 314 all-reduces =
sync-BN; eval partner `mobilenetv2in_fwd_eval_eps0001`), unit `mnv2-verified`, chain-launched
2026-09-29 06:29:19 UTC by `chain_after_jax.sh` two minutes after the JAX run's ✅ COMPLETE. Its
partner is `runs/2026-09-27-mnv2-jax-bf16-350ep/` (71.634 / 90.460), read in place by
`summarize.sh`.

## ⭐ The pair — a GAP, not a tie

| | verified (this run) | JAX reference | Δ |
|---|---|---|---|
| top-1, e350 | **71.124** | 71.634 | **−0.51** |
| top-5, e350 | **90.036** | 90.460 | **−0.42** |
| mean over e326–350 | 71.12 / 90.04 | 71.54 / 90.43 | **−0.42 / −0.39** |

The endpoint Δ is beyond one CI half-width (±0.40), and the offset is not endpoint noise. It holds
in every 25-epoch window from e126 on (`summarize.sh`):

| epochs | Δtop-1 | Δtop-5 | Δ**train** loss |
|---|---|---|---|
| 1–25 | −0.14 | −0.60 | **−0.083** (verified LOWER) |
| 26–50 | −0.34 | −0.32 | +0.016 |
| 51–75 | −0.74 | −0.59 | +0.026 |
| 76–100 | −0.64 | −0.58 | +0.027 |
| 126–150 | −0.55 | −0.39 | +0.024 |
| 201–225 | −0.52 | −0.42 | +0.021 |
| 276–300 | −0.42 | −0.39 | +0.018 |
| 326–350 | −0.42 | −0.39 | +0.018 |

⇒ The **train** loss is higher by +0.018–0.027 in every window from e26 on, so the gap is made in
TRAINING, not at eval or in the BN running statistics. And it **reverses** in the first 25 epochs,
where verified trains faster.

Both paths score the RAW weights (this recipe has no EMA), on all 50,000, every epoch.

## What the pair was meant to retire, and what it did

The old pair (book `content.tex` :7746–7747, 71.90 / 71.91) differed in FOUR ways besides the
lowerer (BN group 64 vs 256, classifier dropout, stem/head activation, label smoothing), and its
JAX arm most likely trained at ls 0.1 from a stale emit (imagenet_parity.md H2/H3). This pair
matches all four, and the TF-slim BN, the staircase and the wd mask, row for row. So it says
something the old one could not: **with the recipe matched, the verified arm trails by ~0.4–0.5.**

## Audit of the gap (2026-09-30 / 10-01, read-only; full notes in memory + `planning/init_parity.md`)

Checked and **matching** between the render and `generated_mobilenet_v2_imagenet_full.py`:
* the RMSProp update op for op (coupled L2 before the accumulator, ε 1.0 inside the root, ms 1.0)
* loss /64 per replica + 158 gradient all-reduces ÷4 = the global-batch mean
* sync-BN forward (μ, Chan variance) AND backward (`gdst`: both dy-reductions all-reduced)
* XLA-SAME `[[0,1],[0,1]]` at all 5 stride-2 sites; BN ε 1e-3; the LR staircase (0.000808 @ e200)
* the shim is byte-identical to `_full_shim`, and `build_imagenet_iter` to the JAX trainer's
* dropout: the mask is drawn at the global 256 × 1280 and split per replica; the generator is
  statistically clean over full epochs (keep 0.80000, step-to-step agreement 0.680 = independent,
  per-position / feature / row spread = binomial)
* θ | m | v slot threading: all 158 parameters, every output slot from its own inputs, in order
* one-step gradient tie on SHARED weights (`tests/mnv2_bf16_grad_tie.py`, **4× 3060 GPU**, JAX
  e150 weights, 256 val images, dropout off; four arms, each against its own sub-ULP-nudge floor):

  | | rel L2 | own floors |
  |---|---|---|
  | verified f32 (twin, same emitter, bf16 off) vs JAX f32: the structural tie | 2.29e-2 | 2.28e-2 / 2.37e-2 |
  | verified bf16 vs JAX bf16 | 8.95e-2 | 8.83e-2 / 8.71e-2 |
  | each side's own bf16 error (vs its f32) | 0.1279 (verified) / 0.1288 (JAX) | |

  Both at their floors; bias against f32 −0.66 % (verified) vs −0.52 % (JAX). f32 step-0 losses
  1.32799 / 1.32798. **bf16 placement is ruled out on production numerics.** The f32 twin is a gate
  input (`.lake/build/mnv2bf16tie/`); rendering it rebuilt `MobileNetV2RenderB` (stale olean) and
  every committed `verified_mlir/` file came back byte-identical.

**Leading suspect: INIT.** The one input that differs by construction, and the one a shared-weight
tie cannot see. Every depthwise kernel starts **√C = 6–31× narrower** on the verified side (`mkParam`
He fan-out `2/(C·k²)` vs the JAX emitters' `2/k²`). Under ε = 1 TF-RMSProp the step is ≈ g, so a
BN-followed weight's RELATIVE step scales as 1/‖W‖². That fits the early reversal. By e279 the
depthwise weight RMS has converged (verified/JAX 0.87–1.04), so the imprint is in the trajectory.
Full per-net audit and the parity plan: `planning/init_parity.md`.

⚠ **Against that hypothesis:** the OLD pair carried the same depthwise mismatch, its verified arm
led early (51.36 vs 46.69 at e10, `content.tex` :7789–7790) and it ended tied. That pair differed
in four other ways, so it is confounded, but init alone is not shown to make this gap. Nothing
here is a re-run; the confirming test (a matched-init verified run) was deferred.

## Verified from the run's own output

`rmsdp64wxdols0eps0001bf16` · `exp x0.980000/1.000000ep staircase+warmup 0ep, baseLR 0.045000` ·
`RMSPROP … v = running MEAN-SQUARE (init 1.0, TF convention)` · `CLASSIFIER DROPOUT: keep 0.800000,
mask tensor<256x1280xf32> (PER-ELEMENT …)` · `running-stats BN: 52 layers, 34112 stat floats → eval
via @mobilenetv2in_fwd_eval_eps0001` · `decay 0.997000 …, new-batch weight 0.003000` ·
`DATA-PARALLEL: 4 replicas x bs 64 = global batch 256, 5004 steps/epoch` · train step RESIDENT
(474 tensors, 40.1 MB) · `SHIM RESPAWN: one producer every 10 epoch(s)` (34 respawns) · step-0
loss 6.982 (ln 1000 = 6.908, no smoothing).

## Loaders, and the respawn's first full run

`LEAN_MLIR_SHIM_RESPAWN_EPOCHS=10` (`37b4df0b`) was on for its first long run. The first-generation
producer (pid 2854229) still ran away: arena-class memory reached **8.43 GiB** by e39 (age 6.4 h),
and epochs e31–40 ran 597–868 s (e32 685, e33 773, e35 868) against a 570 s baseline. The respawn retired it at
the end of its 40-epoch life and pace returned to 570 s at e41 **with no intervention**. `fault_watch`
correctly did not fire (never three over threshold in a row). No later producer passed 7 GiB.
⇒ respawn-every-10 is sufficient here; it bounds the fault rather than preventing its onset.

The other slow epoch, e226 (681 s; e225 619 s), was **self-inflicted**: a CPU-only test of
`tests/mnv2_bf16_grad_tie.py` pinned to cores 20–23 competed with the shim producers. It was
stopped as soon as it showed in the epoch clock.

## Pace

**574 s/epoch** mean over 348 gaps (570–572 typical), eval included. The 2026-09-27 probe measured
104.4 ms/step (~522 s train/epoch). The ~50 s difference is eval, checkpoint and respawn. 55.79 h
wall against the probe-based ~56 h.

## Artifacts

* checkpoint `.lake/build/mobilenetv2in_rmsdp64wxdols0eps0001bf16_ckpt_xla.bin{,.bn,.epoch}` (marker 350)
* per-image top-1 bitmaps for all 350 epochs, `bitmaps/mnv2_rmsdp64wxdols0eps0001bf16_e{1..350}.bin` — LOCAL ONLY (`runs/**/*.bin` is
  gitignored)
* `attempt.log` / `full.log` / `master.log`, `log_ts.log`, `epoch_clock.tsv`, `loader_rss.tsv`,
  `edac.tsv`, the five watcher scripts, `chain_after_jax.sh`, `restart_after_ckpt.sh` (pre-retargeted,
  never needed), `WATCH_EVENT` (fault_watch's end-of-run note)
* `mnv2_verified_curve.csv` — written by `summarize.sh` (per-epoch train loss, top-1, top-5)

## Book rows this bears on (not edited here)

`content.tex` :7719 / :7746–7747 / :7790 / :7816 (the verified 71.91 / 90.52, its curve) and :7572 /
:7789 / :7812 (the JAX 71.90 / 90.41). This pair supersedes both arms with a matched recipe:
**71.634 / 90.460 (JAX) vs 71.124 / 90.036 (verified)**.

## Reproduce

    bash runs/2026-09-27-mnv2-jax-bf16-350ep/summarize.sh
    bash runs/2026-09-27-mnv2-verified-bf16-350ep/summarize.sh
