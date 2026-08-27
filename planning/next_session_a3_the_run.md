# next_session_a3_the_run.md — everything for RSB-A3 is built and gated. What is left is RUNNING it.

**Written 2026-08-06.** Supersedes `next_session_rsb_a3.md`, which is now the *record* of how the
composition got built and gated; this file is the *plan* for the run and the list of what is still
owed.

▶ **The frame has inverted.** The previous brief's question was "does the composition type-check,
does it pass the gates, is the run affordable". All three are answered: it composes, both
accumulation gates certify it, and it fits ~31.7 h against a 40 h bar. **What remains is almost
entirely operational** — measure the thing that is still extrapolated, write the supervisor config,
shake it out, then run. Plus one real correctness gap that is older than this work.

---

## 0. WHERE IT STANDS

| RSB-A3 ingredient | state |
|---|---|
| the R50 gradient | ✅ `r50-gradcheck` (CE) — 53 conv + 33 BN identities, tier 2 fit |
| gradient accumulation | ✅ `r50-accum-tie` + `r50-accum-shard-tie` |
| LAMB | ✅ `r50-lamb-tie` |
| BCE-with-logits | ✅ `r50-bce-tie` (the LOSS; ⚠ its GRADIENT — see §3.1) |
| mixup / cutmix | ✅ `soft-target-tie`, wire v2 |
| train @160 / eval @224 | ✅ **runs end to end**, both sides of the data path |
| **the composition** | ✅ `resnet50in160_lambaccdp8x64bce_train_step` — rendered, compiled, run, and CERTIFIED by both accumulation gates |
| `wdExcludeNormBias` | ⛔ still absent — a recipe delta, see §4 |

Artifacts added 2026-08-06 (all `verified_mlir/`):
`resnet50in160_{lambaccdp8x64bce,lambacc8x64bce,lambacc4x64bce,lambdp64bce}_train_step`.

The measured numbers, all `PJRT_FFI_RESIDENT=1`, `SHIM_WORKERS=8`, median of steps 9..120:

| config | ms/step |
|---|---|
| 224, 4×bs64 | 376 |
| 224, 1×bs64 | 317 (real) / 317 (synth, LAMB+BCE) |
| 224, 4×bs64 synth | 367 |
| **160, 1×bs64** | **169 real / 166 synth** |

---

## 1. ⭐⭐ THE ONE MEASUREMENT STILL EXTRAPOLATED — do this FIRST

▶ **Every 4-GPU 160 number in the record is `166 + 59`**: measured 1-GPU 160 compute plus the
allreduce measured *at 224*. Both terms are measured, but never together, and the artifact that
would let them be measured together **did not exist until now**.

⭐ `resnet50in160_lambaccdp8x64bce_train_step` is a 4-replica 160 render. **Probe it directly:**

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 \
  LEAN_MLIR_VARIANT=lambaccdp8x64bce LEAN_MLIR_RES=160 \
  LEAN_MLIR_MAX_STEPS=120 PJRT_FFI_RESIDENT=1 LEAN_MLIR_SKIP_EVAL=1 SHIM_WORKERS=8 \
  .lake/build/bin/resnet50-imagenet-verified data
```

⚠ **AND RUN IT TWICE — with and without `LEAN_MLIR_BENCH_SYNTH=1`.** That gap is the whole point:

* **the producer question.** 1 GPU at 169 ms/step is **379 img/s**. Four replicas need
  256 img / ~0.225 s = **~1,140 img/s, 3× more**, and A3's shim runs **RandAugment m6** where the 224
  shim is RRC+hflip only. `scripts/jobs/enet-default-4gpu.conf` records EfficientNet at
  **2,061 ms/step on one producer against 203 on eight** — entirely the data pipeline, and it *"read
  exactly like a correct one"*. ▶ If real ≫ synth, raise `SHIM_WORKERS` before anything else.
* **the allreduce question.** synth at 4×160 gives the collective term at the right resolution
  instead of borrowing 224's 59 ms.

⚠ The startup transient lies: GPU sat at 0% with all 8 producers alive for ~a minute at 160 and that
was NOT a stall. Read the `PROBE:` line, not `nvidia-smi`.

▶ **Expected ~225–235 ms/step ⇒ ~31–33 h for 100 epochs.** If it comes in above **288 ms** the run
does not fit 40 h and the answer is bf16 or fewer epochs — decide there, not after starting.

---

## 2. THE SUPERVISOR CONFIG — `scripts/jobs/r50-a3-4gpu.conf` DOES NOT EXIST

`scripts/supervise.sh` is the one engine (AER watchdog, thermal duty cycle, stall guard,
crash-resume off `.bin.epoch`, `PRECHECK`). Configs exist for cnx/enet/mnv2/vit/r34 — **not R50.**
Copy `enet-default-4gpu.conf`; it is the best-commented one. What this job must carry:

```bash
DEVS="0,2,3,4"                       # idx1 (bus 02) and idx5 (bus 62) threw BadTLP
EPOCHS=100
CMD=(.lake/build/bin/resnet50-imagenet-verified data)
ENV_EXTRA=(
  PJRT_REPLICAS=4  LEAN_MLIR_REPLICAS=4
  LEAN_MLIR_VARIANT=lambaccdp8x64bce
  LEAN_MLIR_RES=160                  # ⚠ selects the NET, not a suffix
  LEAN_MLIR_BATCH=64                 # PER REPLICA
  LEAN_MLIR_G2_STEPS=5000            # ⚠⚠ SEE BELOW — not optional
  PJRT_FFI_RESIDENT=1                # silent 10x if absent
  SHIM_WORKERS=8                     # silent 10x if absent
)
TEMP_MAX=78; TEMP_RESUME=62; REST_EPOCHS=""      # temperature governs; the box has no fans
CKPT_EPOCH_FILE=".lake/build/resnet50in160_lambaccdp8x64bce_ckpt_xla.bin.epoch"
```

⚠⚠ **`LEAN_MLIR_G2_STEPS=5000` IS LOAD-BEARING AND THE RUN REFUSES WITHOUT IT.** ImageNet gives
5,004 micro-batches/epoch at global batch 256, and **5004 = 2²·3²·139 is not divisible by 8** —
while k = 8 is exactly what effective batch 2048 requires. The driver throws rather than applying a
short cycle. 5000 = 8 × 625 drops **4 micro-batches = 1,024 images/epoch = 0.08%**.
▶ **Do NOT instead pick a k that divides 5004** — k = 6 gives effective batch 1536 and k = 9 gives
2304, i.e. it silently changes the recipe's batch to make the arithmetic tidy.

⭐ **The cap also makes epochs align with cycles**: 5000/8 = **625 complete cycles per epoch**, so
an epoch boundary is always a cycle boundary and the per-epoch checkpoint never lands mid-cycle.
That is what makes the supervisor's crash-resume safe with accumulation on. ✅ And the LR schedule
follows the cap — `totalSteps := cfg.epochs * nb / accK` reads the *capped* `nb`
(`VerifiedTrain.lean` 1004 → 1166), so 100 epochs is 62,500 updates.

⚠ `PRECHECK` should refuse on a missing `PJRT_FFI_RESIDENT` / `SHIM_WORKERS` / `G2_STEPS`, exactly
as enet's does. All three are silent-failure knobs; the third now fails loudly, the first two do not.

---

## 3. WHAT IS STILL OWED ON CORRECTNESS

### 3.1 ⛔⛔ `r50-gradcheck` tier 1 B does not transfer to BCE — the one real gap

**Pre-existing, not the composition's**, and localised to a single variable by a 2×2
(`next_session_rsb_a3.md` §1.4c):

| res | loss | opt | accum | tier 1 B |
|---|---|---|---|---|
| 224 | CE | AdamW | — | 0.000084 ✅ |
| 224 | CE | LAMB | — | 0.000061 ✅ |
| 224 | **BCE** | LAMB | — | **0.000757** ❌ |
| 160 | BCE | LAMB | k=4 | 0.000571 ❌ |

CE → BCE moves it 12×; LAMB, resolution and accumulation are each exonerated by their own row.
Mechanism: `‖g‖` is 197.5 under CE and **2.18 under BCE**, and tier 1 B measures a CANCELLATION
residual, so it degrades against a 90× smaller gradient.

⚠⚠ **The gate cannot decide at BCE**: its CE calibration has passing sites at 0.000084 and the
weakest *deliberate* control violation at 0.000545, with the tolerance between them — but BCE's
passing sites reach 0.000757, **above** that weakest violation. The populations overlap, so no
tolerance separates them. §6's rule: *a control that cannot fire is not a control.*
▶ **DO NOT raise `R50_GC_EXACT_U` until it goes green.** Options, in preference order:
1. a conditioning-robust tier 1 B (normalise by ‖g_γ‖·‖γ‖ per-site rather than globally, or run the
   identity at a scaled loss so ‖g‖ is comparable to CE's);
2. failing that, **record explicitly that tier 1 B is CE-only** and that BCE artifacts rest on
   tier 1 A + tier 2 — both of which DO pass on the composition (A at 0.000017; tier 2 at rel ≤ 0.230
   against 0.30, control live at 0.997).

⚠ This affects `adam64bce` and `lamb64bce` too, which have shipped since 2026-08-05.

### 3.2 `r50_dp_render_tie.py` — the composed pair is not in it

§3 of the previous brief asked for the (1-replica, 4-replica) pair; `lambacc8x64bce` /
`lambaccdp8x64bce` now exist and are not wired into it.

### 3.3 §1.2's `bce` axis — hygiene, still live

`bce` is a trailing `Bool` on `resnet50TrainStepFaithfulB` while `lamb`/`accum` are constructors of
`R34Opt`, and the variant string is `r34AdamVariant` **plus** a hand-passed `vSuffix := "bce"`. So
the name and the flag can still disagree with nothing noticing — and the artifact this run depends
on ends in `bce`. Mitigated by `#guard`s on the produced names, **not closed**. Folding `bce` into
the optimizer type is what §1.1 wanted and is still the right shape.

⚠ Note the accumulation mechanism itself is now single-writer (`accumScalarConsts`), so the
duplication argument for the full `{base, accumK, bce}` restructure is weaker than it was — `bce` is
the remaining reason to do it.

---

## 4. ⚠ WHAT A COMPLETED RUN WOULD AND WOULD NOT LICENSE — write this BEFORE the number exists

It is **not** the reference's 78.1%, and the deltas are known in advance. Quote against this list:

* **`wdExcludeNormBias = false`** — timm's A3 excludes BN γ/β and biases from weight decay; this
  render decays them, at `wd = 0.02`. Unmeasured effect, but it is the largest single recipe delta.
* **Ghost-BN over 64-image micro-batches**, not a genuine bs2048 forward. The effective batch is
  2048 for the OPTIMIZER; BN normalises over 64. That is what the reference's accumulation does too,
  so it is faithful — but it is not "batch norm at 2048".
* **the 0.08% dropped tail** per epoch from `G2_STEPS=5000`.
* **fp32, not bf16.**
* ⚠ `planning/rsb_a2_resnet50.md` records LAMB at bs512 giving **40.8% against 78.1%** — batch is
  not a detail at this recipe, which is the entire reason k = 8 is in the name.

---

## 5. THE ORDER

1. ⭐ **§1's 4-GPU 160 probe, real AND synth.** Decides the wall-clock and the producer count.
   *~30 min.* Do not skip the synth half.
2. **Write `scripts/jobs/r50-a3-4gpu.conf`** (§2), and `DRY_RUN=1 scripts/supervise.sh` it.
   *~30 min.*
3. ⭐ **A 30-epoch shakeout at 160** — ~9–10 h, overnight, through the supervisor. It is the same
   validation tier R34/ImageNet ran, so it is directly comparable, and it exercises resume, the
   thermal duty cycle and the eval split on real weights rather than on a 3-step checkpoint.
   ▶ **This is also the first run whose eval accuracy MEANS anything** — every eval so far has been
   at chance by construction.
4. **Then A3 proper**, 100 epochs, ~31–33 h — ⚠ **and only after asking.**
5. §3.1's gradcheck gap, at any point; it blocks nothing but it is the honest debt.

⚠ Steps 1–3 are a session. **Do not let step 4 pull step 3 into being skipped** — a 31 h run that
dies at hour 20 on something a 9 h run would have surfaced is the expensive failure here.

---

## 6. THINGS THIS SESSION LEARNED THAT APPLY DIRECTLY

* ⭐⭐ **A shape constant that appears N times is N defects, and the count is not visible from the one
  you are looking at.** The 160 train path needed THREE independent hardcoded `3*224*224`s to fall
  (shim spawn width, driver read width, `loadData`'s `trainPix`), and fixing them one at a time
  surfaced the identical refusal three times. `grep -n '3 \* 224 \* 224'` was worth more than reading
  the call chain. ⚠ A fourth occurrence was CORRECT (`evalFlat`, a property of the val split) — the
  sweep tells you where to look, not what to change.
* ⭐ **"The marker leads, so nothing can spell it accidentally" is an assumption about which OTHER
  markers exist.** `accOn`'s prefix test was documented as safe for exactly that reason, and
  `lamb` ++ `acc` falsified it. Collisions live in CONCATENATIONS — the fourth instance.
* ⭐ **A justification can be circular and still read as sound.** `accOn "emaacc…" == false` was
  defended as "the driver refuses ema × acc anyway" — but that refusal is `if emaOn && accOn`, so
  with `accOn` false it could never fire. Check that the mechanism you are deferring to is reachable.
* ⭐⭐ **When a gate fails on new inputs, localise before you re-tolerance.** Tier 1 B failing on the
  composition looked like the composition's fault; a 2×2 over (loss × resolution × accumulation)
  showed CE→BCE moved it 12× and the other two moved it not at all. **The tolerance was never
  touched.**
* ⭐ **A refactor's safety net should be an equality you can run.** Extracting `accumScalarConsts`
  was licensed by all **27** committed R50/R34 artifacts re-rendering byte-identical, and pointing
  `trainPix := net.d0` at six nets was licensed by a `#guard` block proving `net.d0 == 3*224*224`
  for all of them. Neither needed per-net re-validation.
* ⚠ **A parameterisation is not done when it elaborates.** The 160 path passed a clean build, every
  `#guard`, the contract gate, the wiring gate AND an XLA compile of both artifacts — and could not
  read one batch. Second instance of this exact lesson.
