# mobilenetv2-imagenet-verified — MobileNetV2 / ImageNet-1k, VERIFIED (Lean render → XLA/PJRT), 350 epochs

**Launched** 2026-09-10 15:07:16 UTC · **Finished** 2026-09-12 21:20:48 UTC
**Status** ✅ **COMPLETE — 71.912 / 90.520 in 54 h 13 m 32 s, ONE attempt, zero resumes, zero rests**

## ⭐ RESULT — a dead heat with its own reference

| | top-1 | top-5 | wall clock | steps/epoch |
|---|---|---|---|---|
| MobileNetV2, paper | 72.0 | — | — | — |
| phase-2 JAX reference (§6.5, `85daffbc`) | 71.90 | 90.41 | 38.4 h | 5004 |
| **phase-4 verified (Lean render → XLA/PJRT)** | **71.912** | **90.520** | **54.23 h** | 5004 |

**+0.01 top-1 / +0.11 top-5 over the reference**, from a different lowerer on the same recipe.
95% Wilson interval on the verified final: [71.52, 72.30] — the reference sits inside it.

Best epochs: verified top-1 **72.024 @ ep 346**; reference best **71.99 @ ep 331**. Also a dead heat.

⚠ **The wall clocks are NOT comparable as a lowerer statement** — see Throughput. ~1/3 of this run
executed under the mimalloc RSS bug fixed upstream in `13d90e68`, which this binary predates.

## Pair tracking — Δ = verified − reference, top-1

| epoch | 10 | 20 | 50 | 100 | 150 | 200 | 250 | 300 | 340 | 350 |
|---|---|---|---|---|---|---|---|---|---|---|
| reference | 46.69 | 56.27 | 61.71 | 67.40 | 69.78 | 71.23 | 71.63 | 71.85 | 71.91 | 71.90 |
| verified | 51.36 | 56.47 | 63.82 | 67.74 | 69.97 | 70.95 | 71.70 | 71.84 | 71.93 | 71.91 |
| **Δ** | +4.67 | +0.20 | +2.11 | +0.34 | +0.19 | −0.28 | +0.07 | −0.01 | +0.02 | **+0.01** |

⭐ The curves **converge and then oscillate about zero**, they do not merely end together. The +4.67
at epoch 10 is a warmup artifact (warmup ends at epoch 5) and is gone by epoch 20. The +2.11 at
epoch 50 is the REFERENCE's slow epoch, not a verified lead: the reference gains only 1.18 points
over 40→50 and then 2.57 over 50→60. From epoch 100 on, every point is inside ±0.35.

⛔ **Do NOT read the pair off the train loss** — the ViT run's audit applies here unchanged.
State ties on **val top-1/top-5**.

## ⛔ THE CONFOUND THIS PAIR HAS AND ViT's DID NOT — BatchNorm

§9.6 could say "what is left between the two columns is the lowerer" because ViT has no BN and both
sides were bf16. **That sentence must not be reused here.** The verified path runs four replicas of
64 with no collective touching the batch statistics, so its **BN statistic group is 64**; the JAX
reference under a `NamedSharding` mesh reduces globally, so its group is **256**. §5.7 measured that
same fourfold difference as worth 0.02 points on ResNet-34 at convergence. The +0.01 here is the
same order, so the honest statement is that the two agree to within the BN-group effect — not that
the lowerer is proven neutral.

## Throughput — and why the wall clock is not the headline

| window (epochs) | s/epoch |
|---|---|
| 1→10 | 533 |
| 10→20 | 529 |
| 20→50 | 529 |
| 50→80 | 528 |
| 80→120 | 527 |
| 120→160 | 527 |
| 160→200 | 526 |
| 200→247 | 527 |
| **247→337** | **596** |
| **337→340** | **856** |
| **340→350** | **858** |
| overall | 557.5 |

⛔ **The step change at ~epoch 247 is `13d90e68`'s mimalloc bug, not this net.** The trainer's
anonymous RSS grows because a batch buffer allocated on the prefetch pool thread and freed on the
main thread is only `madvise(MADV_FREE)`d; the pages stay in RSS as LazyFree until the kernel is
under pressure, after which every step pays reclaim. Measured live at epoch 338 on this run:
trainer RSS 60.4 GiB of which **LazyFree 23.3 GiB**, MemFree 2.4 GB, MemAvailable 111.7 GB.
Ruled out at the same time: GPU throttling (`throttle_reasons` 0x0, boost clocks, 46–59 °C),
restarts (1 attempt), producer churn (all four originals at 51 h elapsed), and stale-shim
determinism (this net's shim is post-`4a0a2781` and its producers ran ~3.5 cores each, not the
~1.7 a stale shim caps at).

▶ **The number to quote for §6 is the clean pace: 527 s/epoch ⇒ 51.2 h for 350 epochs.** The
54.23 h actual includes ~3 h of bug. Both are honest; neither is a renderer measurement.

⚠ **Everything probed on 2026-09-10 is contaminated**, including `runs/2026-09-10-mnv2rms-sweep.tsv`
and the `ETA=` string written into `scripts/jobs/mnv2-default-4gpu.conf`. Re-probe post-`13d90e68`.
That sweep chose `SHIM_WORKERS=4`; because more producers meant more memory pressure under the bug,
the sweep was biased AGAINST higher worker counts, and the post-fix optimum may be higher (upstream
found 6 best for ViT).

## Config

`scripts/jobs/mnv2-default-4gpu.conf`, variant `rmsdp64bf16`, 4× RTX 3060, global batch 256
(64 × 4 replicas), 5004 steps/epoch, RMSProp + exponential decay, baseLR 0.045, 5-epoch warmup,
×0.98/epoch, He init, `PJRT_FFI_RESIDENT=1`, `SHIM_WORKERS=4`, tf.data determinism OFF (the
default since `4a0a2781`).

Verified from the run's own output, not the conf: `mobilenetv2in_rmsdp64bf16_train_step.mlir`,
4 replicas, 581 outputs, train 1281167 / test 50000, 474 resident parameter tensors (40.1 MB).

⚠ Four blockers were fixed before this could launch — the conf shipped the **f32** arm as its
default, its `CKPT_EPOCH_FILE` was keyed to that variant, its `PJRT_PLUGIN` pointed at ares'
`xla_cuda12` (absent here) with no `SHIM_PYTHON`, and its PRECHECK validated the **f32** render so
it passed green on the wrong arm. The exe was also 11 days stale. See the conf's header.

## Files

* `mnv2_verified_curve.csv` — 350 rows, epoch/train_loss/lr/top1/top5
* `attempt.log` / `full.log` — the run's own output, every epoch
* `master.log` — supervisor narration (5 lines: launch + COMPLETE, no events)
