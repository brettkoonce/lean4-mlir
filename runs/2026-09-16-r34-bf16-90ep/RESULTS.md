# ResNet-34 / ImageNet-1k — verified path, bf16, 90 epochs

**74.064 % top-1 / 91.754 % top-5 in 22 h 10 m, one attempt, zero restarts, zero rests.**

| | |
|---|---|
| job | `scripts/jobs/r34-default-bf16-4gpu.conf` (new; `r34-default-4gpu.conf` stays fp32) |
| artifact | `verified_mlir/resnet34in_momdp64bf16_train_step.mlir`, 4 replicas |
| box | 4× RTX 3060, CUDA 13.3, `PJRT_FFI_RESIDENT=1`, `SHIM_WORKERS=4` |
| schedule | 90 ep, global batch 256 (64/replica), lr 0.1 cosine + warmup, 5,004 steps/epoch |
| launched | 2026-09-16 03:28:58 UTC → 2026-09-17 01:39:28 UTC |
| binary | built 2026-09-16 01:34:41, i.e. carrying the respawn (`63b21d84`) |

## The number

| | top-1 | top-5 |
|---|---|---|
| **verified (PJRT), bf16, 4× 3060** — this run | **74.064** | **91.754** |
| reference (JAX), bf16, 4× 4060 Ti | 74.16 | 91.92 |
| **Δ** | **−0.10** | **−0.17** |

Per-epoch convergence, 15-epoch windows (mean of per-epoch evals):

| window | verified | reference | Δ |
|---|---|---|---|
| e1–15 | 35.01 / 60.19 | 36.32 / 61.71 | −1.31 / −1.52 |
| e16–30 | 46.15 / 72.33 | 45.05 / 71.35 | +1.10 / +0.98 |
| e31–45 | 50.75 / 76.21 | 49.75 / 75.55 | +1.00 / +0.66 |
| e46–60 | 56.56 / 80.86 | 55.90 / 80.37 | +0.67 / +0.49 |
| e61–75 | 63.94 / 85.79 | 64.12 / 86.06 | −0.18 / −0.27 |
| **e76–90** | **72.32 / 90.82** | **72.28 / 90.94** | **+0.04 / −0.12** |

The curves converge — the final window is within a tenth on both metrics.

## ⛔ What this result may and may not claim

⚠⚠ **It does NOT replace §5's verified row.** That row is **fp32 on 4× 4060 Ti** (74.14 / 91.86).
This run moves **two axes at once** — precision *and* box — so the two cannot be tabulated as if
one superseded the other. What this is: a **second verified datapoint, bf16, on this box, against
the same reference number** (74.16 / 91.92).

For the record, and *not* as a controlled comparison: verified-bf16-3060 sits −0.08 / −0.11 from
verified-fp32-4060Ti. Two axes moved; attribute nothing to precision from that.

## ⭐⭐ The respawn verdict — the reason this run exists

**Arm A: `LEAN_MLIR_SHIM_RESPAWN_EPOCHS` unset (respawn OFF) and `REST_EPOCHS=""` for the entire
90 epochs. Zero respawns, zero rests, zero thermal events. The loaders ran 21.9 h continuously.**

**No degradation.** Per-epoch pace bucketed by loader age:

| loader age | epochs | mean s/ep |
|---|---|---|
| 0–2 h | 7 | 884 (warm-up) |
| 2–4 h | 8 | 873 |
| 4–6 h | 8 | 873 |
| 6–8 h | 8 | 872 |
| 8–10 h | 9 | 872 |
| 10–12 h | 8 | 873 |
| **12–14 h** | 8 | **900** ← external, see below |
| **14–16 h** | 7 | **967** ← external, see below |
| 16–18 h | 9 | 882 |
| 18–20 h | 8 | 883 |
| 20–22 h | 8 | 881 |

Loader memory at 21.9 h — arena class is the signal, and it is the same plateau reached at hour 2:

    pid 3521420  age 21.9 h  rss 5.97 GiB  arena 3.82 GiB  maps 325
    pid 3521608  age 21.9 h  rss 6.42 GiB  arena 4.31 GiB  maps 322
    pid 3521788  age 21.9 h  rss 5.72 GiB  arena 4.36 GiB  maps 334
    pid 3521996  age 21.9 h  rss 5.69 GiB  arena 4.24 GiB  maps 322

Growth was **uniform across all four** and **front-loaded**: ~3.2–3.45 GiB at spawn → ~3.9–4.2 GiB
by hour 2 → flat for the following twenty hours (±0.1 GiB oscillation). Compare EfficientNet, where
**one** producer went 5 → 7–11 GiB while the other three sat still, and the round-robin feed ran at
its pace.

## ⛔ The 12–16 h bump was NOT the loaders

Five epochs exceeded 950 s: **e57 (1055), e59 (997), e60 (1037), e61 (1030), e62 (972)**, with
e54–e56 and e63 shouldering up to them. This is **not** loader degradation, and three independent
facts say so:

1. **It tracks host memory exactly.** `epoch_clock.tsv` logs `MemAvailable`:

   | epoch | avail | swap | s/ep |
   |---|---|---|---|
   | e53 | 125.4 GiB | 5.44 | 872 |
   | e56 | **93.0** | 7.87 | 900 |
   | e57–e61 | **92.5 → 87.5** | ~7.9 | 1030–1055 |
   | e62 | 110.9 | 7.42 | 972 |
   | e63 | 125.7 | 7.98 | 902 |
   | e64+ | ~126 | ~7.2 | 881–886 |

   ~35 GiB of host RAM went elsewhere for ~2.5 h (17:00–19:00 UTC), evicting the TFRecord shards
   from page cache and pushing the feed onto real disk reads.
2. **It recovered on its own**, with no restart and no respawn. EfficientNet's fault never
   self-cleared — it only ever went away on a kill.
3. **Loader RSS FELL through it** (5.40/5.67/5.71/5.75 GiB at 11.2 h → 4.89/5.22/5.15/5.24 at
   13.9 h), the opposite of the signature.

▶ **Cause confirmed by Brett: another job of his on the same box.** Attributed, not merely
correlated. Exclude these five epochs from any pace statement about the loaders.

⚠ The residual baseline drift — 872 s before the bump, 881–883 s after — tracks swap sitting at
~7.2 GiB rather than ~5.5, not loader growth. It is +1.1 %.

## ⭐ Why the negative result is the expected one, mechanistically

The `SHIM_WORKERS` sweep run immediately before this job
(`runs/2026-09-16-r34-bf16-sweep/`, 800 measured steps/arm) gives the reason:

| | R34 (this net) | EfficientNet-B0 |
|---|---|---|
| shim work | **flip only** | AutoAugment + RandAugment per image |
| step, fed | 163 ms | 134 ms |
| step, synth (compute floor) | 144 ms | 97 ms |
| **feed cost** | **19 ms (12 %)** | **37 ms (28 %)** |

A shim doing a quarter of B0's per-image work allocates to steady state early and stays there.
**There is very little feed here to degrade.**

## ▶ What this does and does not settle

* ✅ **On a flip-only shim the fault does not appear in 21.9 h** — past the ≥16 h bar §5 asked for,
  and past the ~13 h at which B0's producer diverged.
* ⛔ **This does NOT exonerate the multi-process loader arrangement.** It *bounds* the fault to
  heavy-augmentation shims. The respawn was never switched on, so this run says nothing about
  whether the mitigation works in production — only that this net did not need it.
* ⛔ **The root cause inside the loader is still unidentified.** §2's allocator soak remains the
  experiment that would find it.

## Files

| | |
|---|---|
| `RESULTS.md` | this file |
| `r34_bf16_verified_curve.csv` | epoch, train_loss, lr, top1, top5 — 90 rows |
| `reference_curve.tsv` | ⚠ extracted from `blueprint/src/content.tex`, **not** a log — see below |
| `epoch_clock.tsv` | per-epoch wall clock, trainer RssAnon, MemAvailable, swap |
| `loader_rss.tsv` | per-loader RSS + arena-class maps, per epoch — **the evidence** |
| `log_ts.log`, `full.log`, `attempt.log`, `master.log` | logs |
| `summarize.sh` | rebuilds every number above from the logs |
| `WATCH_EVENT.e57-hostmem` | the watcher's capture of the external-memory incident |

⛔ **`reference_curve.tsv` is not derived from a reference log, and that is deliberate.**
`/home/skoonce/r34_2018_90ep/` — the path `planning/archive/imagenet_rerun_sweep.md` and
`planning/archive/inflight_r50_queue.md` both name — **does not exist on this box**. The log that
*is* present, `/home/skoonce/r34_90ep_eu_logs/r34_90ep.log`, is a **different run**: a 49,152
denominator (pre-C4), ending **73.95**, not 74.16. Pairing against it would have shifted the
reference by −0.21 silently. The curve here was extracted once from `content.tex`'s published
pgfplots series and asserted to be 90 points ending at exactly 74.16 / 91.92.
