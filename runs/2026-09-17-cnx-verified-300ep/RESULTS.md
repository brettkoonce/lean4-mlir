# ConvNeXt-T / ImageNet-1k — verified path (Lean → StableHLO → XLA/PJRT), bf16, 300 epochs

**Launched 2026-09-17 04:03:37 UTC** (2026-09-16 23:03 CDT) on the 4× RTX 3060 box.
Job `cnx-default-4gpu` · variant `adamdpwxclipdropbf16` · **batch 4 × 64 = global 256**.
ETA **~101 h MEASURED** ⇒ finish ≈ **Mon 2026-09-21 08:33 UTC / 03:33 CDT**.
⚠ Revised up from the ~96 h forecast at e3, the first clean epoch. The probe was right about the
STEPS — 220 ms/step held exactly (4,998 steps in 1,118 s = 223.7 ms) — and the overhead
convention was wrong: eval + checkpoint is **105 s/epoch**, not the ~5% (55 s) MNv2, B0 and R34
realised on this box. 782 eval batches at bs 64 over the 50,000 split plus a 343 MB checkpoint
write costs more than those nets pay. Per-epoch **1,209 s = 20.2 min** (mean of 62 clean epochs; the pace has drifted slightly DOWN, e3 1,224 → last-20 mean 1,204).

> Operational history lives HERE, not in the book. `RESULTS.md` is evidence; §8 is the copy
> session's and quotes from this file rather than restating it.

---

## 0. Status

> ⛔⛔ **THIS RUN WAS KILLED AT EPOCH 67 AND ITS ACCURACY IS NOT A RESULT.** It was stopped
> deliberately once §7.0 established that the verified and reference arms initialise differently
> (He fan-in vs `trunc_normal(0.02)`, 2.6–10.2× wider), which confounds exactly the lowering
> question it existed to answer. **Do not quote 73.874 / 92.092 anywhere.**
>
> ⭐ **What it DID establish, and none of it needed 300 epochs:**
> * the batch-64 rescope works end to end — the pair's batch, LR, warmup, schedule and SPE all match
> * the **`bStr` loss-divisor** defect, found and fixed, now gated fleet-wide (193/193)
> * **resume on a LayerNorm net** at ImageNet scale — 5/5 checks, §4. First evidence ever
> * the **shim respawn** in production — 6 fired on schedule, a full round-robin cycle, §5. First
>   evidence ever for `63b21d84`. And the loader drift is REAL but sub-pathological here
> * the pace: **1,209 s/epoch**, 220 ms/step, 6.8% feed starvation
> * that **eval runs on 1 of 4 GPUs** (~100 s/epoch idle on three cards) —
>   `planning/eval_parallelism.md`
> * and §7.0 itself: the init asymmetry, which nothing else had surfaced
>
> ⛔⛔ **BEFORE ANY RELAUNCH: `.lake/build/convnextin_adamdpwxclipdropbf16_ckpt_xla.bin` is still on
> disk at epoch 67, holding the WRONG-INIT weights.** A relaunch RESUMES from it and the `cnxInit`
> fix would be silently defeated — the new init is only ever applied on a fresh start. The precheck
> WARNS about an existing checkpoint but does not refuse. Move it aside first.

| | |
|---|---|
| run | ⛔ **KILLED at e67/300 on 2026-09-18 02:48 UTC**, 22.7 h in, by decision — the two arms do not share a weight init (§7.0), so this run could not answer the question it was queued for. Re-run with a verified-side `cnxInit`. |
| reference | `/home/skoonce/convnext_t300_3060/`, 2026-09-04→09-07, 76.5 h — rescored **EMA 81.53 / 95.50, raw 81.51 / 95.50**; training log ends 0.8153 / 0.9551 (§8) |
| ⚠ pairing target | **raw 81.51 / 95.50** — this run is `ema := false`, so it pairs against the RAW arm |
| resume test (§4) | ✅ **PASSED all 5 checks** — see §4 |
| respawn test (§3) | ✅ **6 fired on schedule** — e10/20/30/40/50/60, a full round-robin cycle and into the second. See §5 |
| ⛔ blocker for §8 | **the two arms do not share an INIT** — see §7.0. Not a lowering result as it stands |
| final verified | **e67 73.874 / 92.092** (reference e67 77.29 / 93.63) — abandoned, NOT a result |

---

## 1. ⛔⛔ THE PAIR WAS NOT A PAIR UNTIL TODAY — the batch-64 rescope

As shipped, this job ran **4 × 32 = global 128 at 10,009 steps/epoch** against the reference's
**global 256 at 5,004**, with a `baseLR` that was the bs-256 value. The reference's own banner:

    [sup] START ConvNeXt-T ImageNet 300ep · GPUs 0,1,2,3 · bf16 · batch 256 (4x64) · SPE 5004
    lr=0.000250  batch_size=256 (4 devices x 64)  epochs=300  params=28589128

Half the batch, twice the updates, an LR off the linear-scaling rule. That is a **recipe**
difference, not a lowering difference — and it would have forfeited the only thing this net is in
the book for: **ConvNeXt has no BatchNorm**, so it is the net that isolates the LOWERER rather than
a statistic group. A tie says the lowerer is clean; an offset implicates the lowerer or the feed
fleet-wide. Neither reading survives a batch mismatch.

### What moved

`cBS` (per-example renderer, 111 uses) and `bB` (batched renderer, 152 uses) were `private def … := 32`,
unreachable from a `#eval`. Both are now **threaded parameters**, ViT's `vbB` method:

* private helpers take the batch **first, with no default** — the compiler then enumerates every
  call site instead of a reviewer;
* public entry points take it **last, defaulted to 32**, per the §2m rule the file already states
  ("a parameter inserted mid-list captures an existing positional argument at every call site");
* `ConvNeXtRenderB.cnxInBS : Nat := 64` spells the ImageNet batch **once** for all 13 emits.

The two files **had to move together**: `convNextAdamTrainStepFaithfulB` renders the BODY at
`N := bB` and delegates the WRAPPER (`%x`'s shape, the `%bsc` divisor, the drop-path signature) to
`convNextAdamTrainStepFaithful`'s `cBS`. It now spells the batch once and hands the same Nat to
both halves — the discipline `sd`, `V` and `bf16` already had there.

### ⭐ Inertness, measured not asserted

Threading alone, before any `#eval` changed: **`git diff verified_mlir/` EMPTY**. Then with the 13
`convnextin_*` emits moved to 64:

* exactly **13 artifacts changed, all `convnextin_*`**; insertions == deletions (52,195 each way) —
  same graph, new shapes;
* **zero `tensor<32x` remain** in any of them; leading dims are exactly ConvNeXt-T's widths
  (96/192/384/768, 1536/3072 expand) plus 64;
* `all_reduce` counts preserved (183 DP / 0 single-device);
* `convnext_*` (Imagenette), `convnextsin_*`, `convnextbin_*` byte-identical;
* `scripts/regen_verified_mlir.sh check` — all audits pass, **including the prefix audits**:
  `convnextin_fwd` is still a byte-identical 1,662-line prefix of `convnextin_adamdp_train_step`
  and `convnextin_drop_fwd` a 1,734-line prefix of `convnextin_adamdpwxclipdrop`. The forward and
  the train steps moved **together**;
* `convnext-fwd-b-tie` — **byte-identical**, gradMap identical (182 params, same names, same SSA,
  same order), SSA numbering unmoved. The refactor is semantically inert.

### ⛔⛔ THE CATCH — a second, independent spelling of the batch

`bStr`, the third positional argument to `convNextAdamTrainStepFaithfulB`, was the string `"32.0"`.
It feeds `.divConstB … bStr` — **the loss-gradient batch divisor**, which on the no-smoothing path
is the same division as `divide dyr, %bsc` with `%bsc = dense<cBS.0>`. So the batch had TWO
spellings: one derived from the shapes, one a hand-passed literal.

Moving the shapes alone produced this, in the artifact the run loads:

    %v1745 = stablehlo.constant dense<32.0> : tensor<64x1000xf32>
    %v1746 = stablehlo.divide %v1744, %v1745 : tensor<64x1000xf32>

**A 64-row tensor divided by 32.0 — every gradient exactly 2× too large, and nothing fails.** It
reads as a 2× learning rate: the run trains, descends, reports a plausible loss, and pairs against
nothing. Corroborated against a net whose batch already varies — R34's `momdp64` render carries
`dense<64.0>` and `momdp128` carries `dense<128.0>`, so the divisor tracks the batch exactly.

Fixed by **deriving it**: the 11 train-step emits now pass `s!"{cnxInBS}.0"`, so the shapes and the
divisor cannot drift. All three batch-shaped constants in the artifact are now `64.0`, and the
`PRECHECK` greps for `dense<64.0> : tensor<64x1000xf32>` so it can never regress silently.

> This is the repo's own recurring lesson one layer down: *a parameter no audit reads is a
> parameter that can be wrong forever.* The shapes were audited; the divisor was a string.

### ⭐ Two things the rescope fixed for free

1. **The LR is now correct.** `baseLR` default 2.5e-4 is the bs-256 value; at global 256 it is both
   the reference's own knob and the linear-scaling value. The driver's comment said the opposite
   ("this run is at global 128, so the rule would put it near 1.25e-4 … left as the thing to tune
   first"). Now set explicitly as `LEAN_MLIR_BASE_LR_U=250` and asserted by the precheck.
   ⚠ Warmup also matches without change: reference e1 `lr=1.2e-05` = 2.5e-4/20 ⇒ 20-epoch warmup,
   which is `convNeXtTinyImagenetConfig.warmupEpochs` and what the driver passes.
2. **Steps per epoch halved**, 10,009 → 5,004, matching the reference's SPE exactly.

### ⚠ Left deliberately at 32

`convnextsin_*` and `convnextbin_*` (ConvNeXt-S and -B at ImageNet) keep the 32 default: neither
has ever been trained or paired, so moving them would churn nine artifacts and nine gate runs for
nothing. If either is ever launched, check its `LEAN_MLIR_BATCH` against its render — the new
`PRECHECK` in `cnx-default-4gpu.conf` greps the artifact for its own baked batch and is the template.

---

## 2. Throughput — measured on this box before quoting any ETA

`runs/2026-09-17-cnx-sweep/` · bf16, 4×64, `SHIM_WORKERS=4`, 400 measured steps (WARM=200 STEPS=600),
`PJRT_FFI_RESIDENT=1`, 5,004 steps/epoch.

| arm | median | mean | min | p90 | starvation |
|---|---|---|---|---|---|
| **fed** | **220** | 219 | 203 | 233 | **15 ms (6.8%)** |
| synth (compute floor) | 205 | 205 | 202 | 207 | — |

⭐⭐ **The plan predicted ~28% starvation and it is 6.8%** — better than EfficientNet's 28% *and*
R34's 12%. The batch rescope is why: per-step compute doubled (205 ms at bs64 vs ~102 at bs32)
while the per-step feed cost did not, so the same AutoAugment + RandAugment + random-erasing work
now amortises over twice the arithmetic.
⚠ Read that as a **ratio** result. The absolute feed cost per image is unchanged, and §3's
loader-degradation risk is drift over HOURS, which a 400-step probe cannot see at all.

**ETA** 5,004 × 220 ms = 1,101 s/epoch of steps; +~5% for eval + checkpoint (the convention MNv2,
B0 and R34 all realised on this box; the reference's own eval is 17.7 s) ⇒ ~1,156 s/epoch ⇒
**~96 h ≈ 4.0 d**.

**For scale**, what the two changes bought:

| shape | ms/step | steps/ep | wall clock |
|---|---|---|---|
| f32 4×32 (the only previously measured shape on this box) | 220 | 10,009 | 183.7 h |
| **bf16 4×64 (this run)** | **220** | **5,004** | **~96 h** |
| JAX reference, same box, same shape | 179 | 5,004 | 76.5 h (actual) |

Same ms/step, half the steps ⇒ **1.9× wall-clock win**. Against the reference on the same box and
the same shape the verified path is **1.23×** — the honest lowering overhead at this batch.

⚠ Peak memory read 9,181–9,343 MiB/GPU, but that is XLA's BFC preallocation (8.72 GiB requested,
75% of 12 GB), **not** true usage. The meaningful fact is that it did not OOM: batch-64 bf16 fits.
The 4-replica train step compiled in **22.4 s** (567 outputs), the eval forward in 3.4 s.

---

## 3. Launch configuration

| | |
|---|---|
| conf | `scripts/jobs/cnx-default-4gpu.conf` (rewritten 2026-09-17) |
| launcher | `run_cnx_verified.sh` under `systemd-run --user --unit=cnx-verified` |
| variant | `adamdpwxclipdropbf16` → `verified_mlir/convnextin_adamdpwxclipdropbf16_train_step.mlir` |
| batch | `LEAN_MLIR_BATCH=64` per replica, 4 replicas |
| LR | `LEAN_MLIR_BASE_LR_U=250` (2.5e-4), 20-epoch warmup, cosine over `LEAN_MLIR_EPOCHS=300` |
| feed | `SHIM_WORKERS=4`, `PJRT_FFI_RESIDENT=1`, `LEAN_MLIR_SHIM_RESPAWN_EPOCHS=10` |
| thermal | `TEMP_MAX=78` / `TEMP_RESUME=62`; `REST_EPOCHS=""` |
| exe | rebuilt 2026-09-17 03:53, newer than `VerifiedTrain.lean`, `pjrt_ffi.c`, `f32_helpers.c`, `MainConvNeXtImagenet.lean` |

The precheck was rewritten on `r34-default-bf16-4gpu.conf`'s model. The gate it replaced asserted
**none** of: the variant, the ckpt/variant agreement, the exe's freshness, the render's baked batch,
the loss divisor, the shim's freshness, the LR, or an idle GPU — and it tested `.venv/bin/python3`,
which on this box is a wrapper into a retired ROCm venv, so it passed while naming the wrong
interpreter. Dry-run via `lake run cnx-default-4gpu plan`: **passes** (one bug in the new precheck
itself was caught by that dry run — `grep -c` prints `0` *and* exits 1, so `|| echo 0` yielded
`"0\n0"`; now `|| true`).

Watchers, all as systemd units (a bare `&` gets SIGKILLed by the agent harness under memory
pressure): `cnx-clock`, `cnx-loaders`, `cnx-logts`, `cnx-fault`, `cnx-resume-test`.

---

## 4. ⛔ The LayerNorm resume test (§4) — ARMED

Resume has **never** been exercised on a LayerNorm net at ImageNet scale. The `367bb28b` fix and
all eight of its live resumes were EfficientNet — a BatchNorm net with a `.bn` companion. ConvNeXt
writes no companion (no BN, `ema := false` here), so its resume surface is the `[θ|m|v]` blob plus
the epoch marker, and every ViT/ConvNeXt verified run so far was one attempt with zero restarts.
Every thermal rest, every `REST_EPOCHS` fallback and every respawn-failure recovery rides on it.

`resume_test.sh` waits for the epoch-1 marker, hashes the checkpoint, SIGTERMs the **trainer** (not
the supervisor), then checks: (a) a resume is announced **at epoch 1**, (b) the checkpoint bytes are
unchanged across the restart, (c) no `.bn` companion appeared, (d) it steps again, (e) epoch 2 is
continuous with epoch 1, read against the reference's own e1→e2 (0.0064 → 0.0068).

⛔ If it does not resume cleanly: **stop the run**, do R2 of the shim plan §4 on Imagenette
(`convnext-verified-adam`, variant `ema`, rebuild the stale exe), relaunch after.

### ✅ RESULT — PASSED, 2026-09-17 04:48 UTC

    epoch-1 marker seen 04:25:40
      ckpt: 343069536 bytes  sha256=97411ff875719f274377f1195125d0e3
        epoch 1: test_acc = 253/50000 = 0.506000%  top5 = 2.198000%
    SIGTERM -> trainer pid=3823944 (NOT the supervisor)
    resume line:   ▸ resuming from checkpoint at epoch 1   (04:27:10)
      ✅ resumed AT EPOCH 1
      ✅ checkpoint BYTE-IDENTICAL across the restart (343069536 bytes, 97411ff8…)
      ✅ still no .bn companion (correct for LayerNorm)
      ✅ stepping again
        epoch 2: test_acc = 303/50000 = 0.606000%  top5 = 2.574000%
      ✅ CONTINUOUS — epoch 2 above epoch 1

The supervisor handled it cleanly: attempt 1 `ended (exited) at epoch 1; 15s then retry`, attempt 2
`resuming at epoch 1/300`. **So resume works on a LayerNorm net at ImageNet scale** — the path
nothing had ever exercised, and the one every thermal rest, every `REST_EPOCHS` fallback and every
respawn-failure recovery depends on. Pace was unchanged across the restart (220–225 ms/step).
Cost: one epoch's restart overhead (~2 min: recompile 22 s + val preload).

⚠ What it does NOT establish: this variant is `ema := false` and ConvNeXt has no BN, so the blob is
just `[θ|m|v]`. It says nothing about the `ema_bn` half of `367bb28b`, which remains
EfficientNet-only evidence.

---

## 5. Respawn — ON from step 0, first production test

`LEAN_MLIR_SHIM_RESPAWN_EPOCHS=10` (63b21d84): every 10 epochs one of the 4 loaders is replaced,
staggered, so no loader lives past ~40 epochs (~12 h at this pace).

Arm B rather than R34's bare Arm A, deliberately: R34 already supplied the Arm-A evidence (its
flip-only shim never degraded in 21.9 h), while ConvNeXt's shim is the heavy kind — EfficientNet's
class, which degraded at 5.6–13 h with one of four producers drifting to 7–11 GiB RSS and pacing the
whole round-robin feed 1.8× slow. On the book's longest schedule that is not a risk worth running
bare, and running it ON *is* the A/B `planning/shim_loader_health_and_resume_tests.md` §2 asked for.

`fault_watch.sh` **self-calibrates** (R34's hardcoded 950 s fired on a forecast error): baseline =
median of epochs 4–12, trigger = 3 consecutive epochs > 1.25× baseline **while host MemAvailable is
healthy**. Both conditions are from the R34 lesson — its first watcher fired on one slow epoch that
was tracking host memory falling 126 → 87 GiB and recovered on its own.

⚠ With the respawn ON, a flat `loader_rss.tsv` is evidence about the **mitigation**, not about the
fault. The fault question is answered by `etime_s`: no loader should exceed ~40 epochs of age.

### ✅ RESULT SO FAR — the mitigation works in production

    04:05:04  ▸ SHIM RESPAWN: one producer every 10 epoch(s), round-robin over 4
    07:31:07  ▸ shim respawn: producer 0 of 4 replaced after epoch 10 (generation 1, seed 5)
    10:53:28  ▸ shim respawn: producer 1 of 4 replaced after epoch 20 (generation 2, seed 10)
    14:14:51  ▸ shim respawn: producer 2 of 4 replaced after epoch 30 (generation 3, seed 15)

Round-robin, on schedule, ~3.4 h apart, each announced with its slot and seed. **This is the first
production evidence for `63b21d84`** — previously built, inert-when-off, smoke-tested, and never
run against the fault it was written for.

⭐⭐ **AND THE FAULT IS REAL, BUT SUB-PATHOLOGICAL — the loaders DO drift with age:**

| loader age | RSS | arena-class |
|---|---|---|
| 0.6 h | 5.11 GiB | 3.32 GiB |
| 3.9 h | 5.79 | 3.82 |
| 7.3 h | 5.75 | 3.92 |
| 10.4 h | 5.81 | 4.17 |

Monotone, ~+0.07 GiB/h, in the arena class — **the same signature EfficientNet showed**. But B0's
went 5 → **7–11 GiB** with the feed pacing 1.8× slow; this tops out under 6 GiB and **the pace is
flat-to-improving** (e3 1,224 s → e30 1,196 s, mean 1,214). So on this net the drift exists and has
not reached the regime that costs throughput — and the respawn is capping lifetime at ~40 epochs
(≈13.5 h) before it can.
⚠ This does NOT settle whether the respawn is *necessary* here; it settles that it works and that
the drift is real. The counterfactual (Arm A on this shim) was not run, deliberately — see above.

---

## 6. Pair tracking — 50-epoch windows

Rebuild with `./summarize.sh`. ⚠ `reference_curve.tsv` stores FRACTIONS (0.8153); the verified log
prints PERCENTAGES (81.53) — `summarize.sh` scales by 100.

| window | verified top1 / top5 | reference top1 / top5 | Δ |
|---|---|---|---|
| e1–50 | ▶ in progress (e31 67.96 / 88.38) | 41.60 / 68.43 (e31 71.58 / 90.43) | — |
| e51–100 | — | — | — |
| e101–150 | — | — | — |
| e151–200 | — | — | — |
| e201–250 | — | — | — |
| e251–300 | — | — | — |

⚠ **The first two epochs read slightly BELOW the reference** — 0.506 / 0.606 against 0.64 / 0.68,
and 322/50000 sits just outside the CI on 253/50000. Do not read anything into it. At epoch 1–2 of
a 20-epoch warmup the LR is 1.2–2.5e-5, the net is barely trained, and both known NON-lowering
differences live exactly here: drop-path masks are host-drawn vs `jax.random`, and the verified init
is `heInit` (a Bates-3 sum of three uniforms — variance matched to JAX's single uniform, shape not).
ViT's pair converged by epoch 100. The verified arm was also climbing faster over those two points
(+0.100 vs +0.04), which is exactly why two points meant nothing — and **at e3 it went ABOVE the
reference on both metrics: 1.410 / 5.230 against 1.300 / 4.740.** The early gap closed and
reversed inside three epochs, which is the expected shape and not yet a result either way.

### ⚠ THE TRAJECTORY THROUGH e31 — a CROSSOVER, not an offset

| epoch | verified | reference | Δ |
|---|---|---|---|
| 1 | 0.506 | 0.64 | −0.13 |
| 3 | 1.410 | 1.30 | +0.11 |
| 5 | 5.108 | 3.83 | **+1.28** |
| 10 | 29.996 | 25.63 | **+4.37** |
| 15 | 50.412 | 49.22 | +1.19 |
| 20 | 59.576 | 61.51 | −1.93 |
| 24 | 63.658 | 67.16 | −3.50 |
| 31 | 67.962 | 71.58 | −3.62 |

**15 of 31 epochs above.** The verified arm LEADS through the warmup, peaks at +4.4 around e10,
crosses over near e16–18 (the end of the 20-epoch warmup), and now trails.

⭐ **The schedule is NOT the cause and this was checked rather than assumed.** The two LR curves
agree to the printed digit at every epoch sampled — 1.2e-05 / 6.3e-05 / 0.000125 / 0.000188 /
0.000250 (peak, e20) / 0.000249 (cosine begun, e31). Warmup length, peak rate and decay all match.

⚠ **Read it as a PHASE LAG, not a final-accuracy gap.** The reference is gaining ~0.6 pts/epoch
here, so −3.6 points is the verified arm sitting where the reference was ~6 epochs earlier:
**verified e31 = 67.96% ≈ reference e25 = 68.02%.** Mid-training gaps on a steep curve exaggerate
small differences; both arms saturate later, where a 6-epoch lag is worth a fraction of a point.
ViT's pair converged by e100.

⚠ Train loss is much closer than the val gap suggests: verified sits **+0.04 nats** of the
reference (e20 4.588 vs 4.533, e31 4.138 vs 4.096), and was BELOW it at e10. So this is not a
"training is broken" signature — the arms are learning at nearly the same rate and generalising
slightly differently, which is where the two known non-lowering differences live (host-drawn
drop-path masks vs `jax.random`, and `heInit`'s Bates-3 shape vs JAX's single uniform).
⛔ Do not quote that loss comparison as a curve — this net's `%loss` is a report-only carve-out.

▶ **What decides it:** whether the lag closes as both saturate (ViT's shape) or persists (the
EfficientNet shape, which was −0.27 at the END with 0 of 300 epochs above). Too early at e31/300,
with the cosine barely started.

⭐ The number to watch is **epochs above the reference**. EfficientNet's tell was **0 of 300**, a
persistent −0.27 offset with the BN group the named suspect. `summarize.sh` prints this count.

---

## 7. What this pair does and does not isolate

## ⛔⛔ 7.0 — THE TWO ARMS DO NOT SHARE AN INIT (found e65; MEASURED 2026-09-18)

**This pair does not isolate the lowerer, and §8 must not claim it does.** The run announces the
problem in its own banner:

    train 1281167, test 50000; bs 64, ConvNeXt-T (ImageNet-1k) adamdpwxclipdropbf16
    (cosine+warmup 20ep, baseLR 0.000250), He init          <-- HERE

* **verified**: `mkParam`'s He default — rank-4 is fan-**OUT**, `2/(oc·kh·kw)`; rank-2 is Glorot
* **reference**: `jax/MainConvNeXtImagenet.lean:65` sets **`cnxInit := true`** — ConvNeXt
  `_init_weights`, `trunc_normal(0.02)` on **every conv AND the head**

### ⚠⚠ THE NUMBERS BELOW ARE MEASURED, AND AN EARLIER READING OF THEM WAS WRONG

The first version of this section reported 2.6×–10.2× "wider", read off
`SpecHelpers.heInitLayer` — which is He fan-**IN** and is **not the function this trainer calls**.
`mkParam` is fan-**OUT**. Same net, same flag, different rule, ratios wrong by up to 20× and in the
wrong DIRECTION at several shapes. ⭐ So `tests/TestCnxInit.lean` (`lake exe cnx-init-check`) now
emits the real 182-spec layout both ways and MEASURES σ. Init is host-side, never reaches a
committed artifact, and therefore has no drift guard — measurement is the only reliable reading.

**What the default actually emitted, per distinct weight shape** (ratio to the reference's 0.02):

| shape | role | σ | ratio |
|---|---|---|---|
| `[96,384,1,1]` | 1×1 project, stage 1 | 0.1447 | **7.23×** |
| `[192,768,1,1]` | 1×1 project, stage 2 | 0.1023 | 5.11× |
| `[384,96,1,1]` | 1×1 expand, stage 1 | 0.0726 | 3.63× |
| `[384,1536,1,1]` | 1×1 project, stage 3 | 0.0721 | 3.61× |
| `[768,3072,1,1]` | 1×1 project, stage 4 | 0.0510 | 2.55× |
| `[96,3,4,4]` | 4×4/s4 patchify stem | 0.0363 | 1.81× |
| `[768,1000]` | classifier head | 0.0337 | 1.68× |
| `[96,1,7,7]` | 7×7 depthwise, stage 1 | 0.0203 | **1.01× — correct** |
| `[192,1,7,7]` | 7×7 depthwise, stage 2 | 0.0147 | 0.74× |
| `[384,1,7,7]` | 7×7 depthwise, stage 3 | 0.0103 | **0.52×** |
| `[768,1,7,7]` | 7×7 depthwise, stage 4 | 0.0073 | **0.37×** |

**56 of 59 weight specs differ from the reference. The spread is 0.37×–7.23× and it goes BOTH
WAYS** — the 1×1 projections start far too wide while the deep depthwise kernels start far too
narrow, monotonically with channel count, and one shape happens to land right.

⚠ That is arguably worse for training dynamics than a uniform scale error would be: it distorts the
RELATIVE scale between layer types within every block, so the depthwise→expand→project path starts
badly conditioned rather than merely hot or cold.

⚠⚠ **A KNOWN, PRE-EXISTING gap, not a regression from this session** — it is open item 2 of the
`cnxInit` thread ("the verified path has a DIFFERENT init from the JAX path, today … needs deciding
as its own question"). The batch rescope closed the batch axis; this one was already open.

**How much is it worth?** On the JAX side, `cnxInit` + the head LN together moved 81.10 → **81.53
(+0.43 / +0.13)** at 300 epochs. So at CONVERGENCE the init is worth a few tenths, and the −3.5
mid-training gap should narrow substantially by e300. ⚠ But that A/B moved a Xavier init to 0.02,
i.e. one wrong scale to the right one; this is a per-shape spread in both directions, so it is not
the same experiment and +0.43 is not a safe extrapolation either up or down.

▶ **What §8 may say:** one architecture, two independent lowerings, **and two different weight
initialisations** — lowering-plus-init, not separable from this run. ⛔ Do NOT attribute a residual
offset to the lowerer, and do NOT use this pair to adjudicate EfficientNet's −0.27 BN question,
which was the reason it was queued.
▶ **Fixed 2026-09-18**: `VerifiedConfig.cnxInit`, host-side, no artifact moves, gated by
`cnx-init-check` (59/59 weights at σ=0.02, worst relative error 1.8%, with a control that fires).

---

## 7. What this pair does and does not isolate

## ⛔⛔ 7.0 CORRECTION (2026-09-18, e65) — THE TWO ARMS DO NOT SHARE AN INIT

**This pair does NOT currently isolate the lowerer, and §8 must not claim it does.** Found while
chasing the persistent ~-3.5 gap; the run announces it in its own banner:

    train 1281167, test 50000; bs 64, ConvNeXt-T (ImageNet-1k) adamdpwxclipdropbf16
    (cosine+warmup 20ep, baseLR 0.000250), He init          <-- HERE

* **verified**: `F32.heInit` at `std = sqrt(2 / fan_in)` (`SpecHelpers.lean:234`)
* **reference**: `jax/MainConvNeXtImagenet.lean:65` sets **`cnxInit := true`** — ConvNeXt
  `_init_weights`, `trunc_normal(0.02)` on **every conv AND the head**

| layer | fan_in | verified He std | paper / reference | ratio |
|---|---|---|---|---|
| 4x4/s4 patchify stem | 48 | 0.2041 | 0.02 | **10.2x** |
| 7x7 depthwise | 49 | 0.2020 | 0.02 | **10.1x** |
| block 1x1 expand (stage 1) | 96 | 0.1443 | 0.02 | 7.2x |
| block 1x1 project / 2x2 downsample | 384 | 0.0722 | 0.02 | 3.6x |
| head 768->1000 | 768 | 0.0510 | 0.02 | 2.6x |

So the verified arm initialises **2.6x-10.2x wider than the recipe its reference implements**, and
widest exactly at the stem and the depthwise kernels.

⚠⚠ **This is a KNOWN, PRE-EXISTING gap, not a regression from this session's work** — it is open
item 2 of the `cnxInit` thread ("the verified path has a DIFFERENT init from the JAX path, today
... this is the convention-audit failure mode and needs deciding as its own question"). The batch
rescope closed the batch axis; this axis was already open, and I did not see it until the curve
forced the question.

**How much is it worth?** On the JAX side, `cnxInit` + the head LN together moved 81.10 -> **81.53
(+0.43 / +0.13)** at 300 epochs. So at CONVERGENCE the init is worth a few tenths, not 3.5 points,
and the mid-training gap should narrow substantially by e300. ⚠ But do not read +0.43 as the size
of THIS handicap: the JAX pre-fix init was Xavier (0.118 at the stem) and the verified init is He
fan-in (**0.204**), i.e. wider still, so the verified arm's disadvantage is larger than that A/B
measured.

▶ **What §8 may say:** one architecture, two independent lowerings, **and two different weight
initialisations** — the result is lowering-plus-init and the two are not separable from this run.
⛔ Do NOT attribute a residual offset here to the lowerer, and do NOT use this pair to adjudicate
EfficientNet's -0.27 BN question, which was the whole reason it was queued. That reading needs the
inits matched first.
▶ **The fix** is a verified-side `cnxInit` equivalent (trunc-normal 0.02 for ConvNeXt's convs and
head). It is HOST-SIDE and changes no committed artifact — then a re-run. Cost: 102 h.

---

## 7. What this pair does and does not isolate

**Does:** the lowering **and the init together** — see §7.0. Same architecture, same box
(4x RTX 3060), same batch (4x64 = 256), same LR (2.5e-4) and schedule (20ep warmup + cosine over
300), same augmentation including mixup/cutmix, same 50,000-image eval protocol. Say *"one
architecture, two independent lowerings"*; never *"proven"* — the
proof-carrying tier stops at Imagenette.

**Does not:**
* ⛔ **the INIT** — see §7.0. This is the big one, and it is a BIAS, not variance.
* **drop-path and dropout masks** are host-drawn here and `jax.random`'s there. Variance, not bias
  — do not read a small gap as a lowering fact.
* ✅ **mixup/cutmix IS on** and matches the reference recipe (the run announces `SHIM_MIX=both`,
  wire v4 soft float32 targets) — the handoff's "ViT/ConvNeXt run WITHOUT their reference's
  mixup/cutmix" caveat is CLOSED, checked on this run's own log rather than assumed. ⚠ λ comes from
  numpy's Generator, not `jax.random`, so agreement there is distributional, never per-step.
* **no BN statistic-group row**, and that is the point of this pair rather than a gap in it.
* **the `%loss` curve.** ConvNeXt's initial loss is 10.42 where every other net starts near
  ln(1000) = 6.91. It descends and its baked divisor is correct, but §5 lists this net's `%loss` as
  a report-only carve-out outside every faithfulness theorem, and R34 shipped a wrong `%loss` once,
  caught only by a numeric tie. `cnx_verified_curve.csv` carries `train_loss`; **do not quote it as
  a curve.**

---

## 8. ✅ BOTH OPEN QUESTIONS RESOLVED — and the answer MOVES THE PAIRING TARGET

### 1. 95.50 vs 95.51 — not an error, two eval paths

The book's **95.50 comes from the canonical full-50k rescore** (`jax/scripts/eval_convnext_full50k.py`
over the `.bin`), not from the training log's last epoch (95.51). Those two paths run on
**bit-identical weights** (verified max|diff| = 0 between the `.bin` and the npz's `ema_params`)
and disagree by **5 images in 50,000 — 0.01%** — sharding ⇒ reduction order ⇒ borderline flips.
⚠ The in-training val already covers all 50,000 (the generated pipeline batches validation with
`drop_remainder=training` = False), so the rescore CONFIRMS the number rather than correcting it.
`reference_curve.tsv` is the training-log curve and legitimately ends 0.9551; the book's endpoint
is the rescore. Say which path a quoted number came from.

### 2. ⛔⛔ 81.53 IS THE **EMA** ARM — AND THIS RUN HAS NO EMA

The reference's rescore gives, on the final checkpoint:

| arm | top-1 | top-5 |
|---|---|---|
| EMA (what the book prints) | **81.53** | 95.50 |
| **raw** | **81.51** | 95.50 |

⭐ EMA is worth only **+0.02 / +0.00** on this net — essentially nothing, and the opposite of
EfficientNet-B0's +0.82. Visible in the log's own shape: the last ~40 epochs run at lr ≈ 0 on a
fully annealed cosine, so the raw weights have stopped moving and the EMA shadow has nothing left
to average. ⛔ Do NOT carry Ch. 7's "EMA was the most quotable thing" framing into §8.

**⚠ THE CONSEQUENCE FOR THIS PAIR.** The verified arm runs `adamdpwxclipdropbf16`, i.e.
`ema := false`. So the honest comparison is **verified-raw against reference-RAW = 81.51 / 95.50**,
not against the 81.53 the book prints. That is a −0.02 shift in the target. Small, but this pair
exists to read an offset of about that size against EfficientNet's −0.27, so pairing the raw
verified arm against the EMA reference would bias the one number §8 is for.

▶ Re-derivable rather than taken on trust: `jax/scripts/eval_convnext_arms_full50k.py` prints both
arms from `/home/skoonce/convnext_t300_3060/convnext_tiny_imagenet_e300.state.npz` (present, 457 MB
— the `.bin` alone is `ema_params`, so the raw weights exist only inside the npz). ⚠ That script
guards itself with a raw-vs-EMA max|diff| print: if it is 0 the comparison is fake.

### 3. Still genuinely open

* Nothing blocking. If §8 wants an EMA-to-EMA comparison instead, the verified side would need a
  re-run at an `ema*` variant — not worth 96 h for +0.02.
