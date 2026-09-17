# ConvNeXt-T / ImageNet-1k — verified path (Lean → StableHLO → XLA/PJRT), bf16, 300 epochs

**Launched 2026-09-17 04:03:37 UTC** (2026-09-16 23:03 CDT) on the 4× RTX 3060 box.
Job `cnx-default-4gpu` · variant `adamdpwxclipdropbf16` · **batch 4 × 64 = global 256**.
ETA ~96 h ⇒ expected finish ≈ **2026-09-21 04:00 UTC / 2026-09-20 23:00 CDT**.

> Operational history lives HERE, not in the book. `RESULTS.md` is evidence; §8 is the copy
> session's and quotes from this file rather than restating it.

---

## 0. Status

| | |
|---|---|
| run | ▶ IN FLIGHT |
| reference | `/home/skoonce/convnext_t300_3060/`, 2026-09-04→09-07, 76.5 h, **81.53 / 95.51** |
| resume test (§4) | ▶ armed, fires at the epoch-1 marker |
| respawn test (§3) | ▶ ON from step 0, `LEAN_MLIR_SHIM_RESPAWN_EPOCHS=10` |
| final verified | — |

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

**Result: ▶ pending.**

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

**Result: ▶ pending.**

---

## 6. Pair tracking — 50-epoch windows

Rebuild with `./summarize.sh`. ⚠ `reference_curve.tsv` stores FRACTIONS (0.8153); the verified log
prints PERCENTAGES (81.53) — `summarize.sh` scales by 100.

| window | verified top1 / top5 | reference top1 / top5 | Δ |
|---|---|---|---|
| e1–50 | — | 41.60 / 68.43 | — |
| e51–100 | — | — | — |
| e101–150 | — | — | — |
| e151–200 | — | — | — |
| e201–250 | — | — | — |
| e251–300 | — | — | — |

⭐ The number to watch is **epochs above the reference**. EfficientNet's tell was **0 of 300**, a
persistent −0.27 offset with the BN group the named suspect. `summarize.sh` prints this count.

---

## 7. What this pair does and does not isolate

**Does:** the lowering. Same architecture, same recipe, same box (4× RTX 3060), same batch, same LR,
same schedule, same 50,000-image eval protocol. One axis moves — Lean/StableHLO/PJRT against
JAX/XLA. Say *"one architecture, one recipe, two independent lowerings"*; never *"proven"* — the
proof-carrying tier stops at Imagenette.

**Does not:**
* **drop-path and dropout masks** are host-drawn here and `jax.random`'s there. Variance, not bias
  — do not read a small gap as a lowering fact.
* **no BN statistic-group row**, and that is the point of this pair rather than a gap in it.
* **the `%loss` curve.** ConvNeXt's initial loss is 10.42 where every other net starts near
  ln(1000) = 6.91. It descends and its baked divisor is correct, but §5 lists this net's `%loss` as
  a report-only carve-out outside every faithfulness theorem, and R34 shipped a wrong `%loss` once,
  caught only by a numeric tie. `cnx_verified_curve.csv` carries `train_loss`; **do not quote it as
  a curve.**

---

## 8. Open questions for §8

1. ⚠ **95.50 vs 95.51.** The book prints 81.53 / 95.50; the reference log's last epoch is
   **0.8153 / 0.9551**, and e298 is 0.9550. `extract_reference.sh` asserts the log's value. Find
   where 95.50 came from (rescoring? truncation of 0.95506? a different epoch?) before §8 quotes
   either.
2. ⚠ **Which weights is 81.53?** The reference log has no EMA line. Memory says EMA was worth only
   +0.03 on this net (vs +0.82 on B0) — so it probably does not matter, but confirm rather than
   assume. ⚠ And do **not** reuse Ch. 7's EMA framing here.
3. The verified arm runs `ema := false`, so if 81.53 is an EMA number the comparison needs saying.
