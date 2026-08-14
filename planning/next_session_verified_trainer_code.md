# next_session_verified_trainer_code.md — get the TRAINER CODE down; runs come later

**Written 2026-08-14** at the end of the session that landed `fc0a492 … 3732a11`. Successor to
`next_session_a3_the_run.md` (spent — its run finished at 77.43%).

▶ **THE FRAMING, and it decides every priority below.** Every net has to be re-run anyway
(§1), so this session is **not** about producing numbers. It is about getting the CODE for the
verified trainers complete and gated, so that when the compute question is answered — a cloud box,
a `cBS` refactor, or both — the runs are a scheduling decision and not a development one.

⚠ **Do not start a long run out of this document.** Every wall-clock figure here exists to size a
decision, not to license a burn. `planning/vit_convnext_sb_scaleup.md` has the cloud sizing.

---

## 0. WHERE THINGS STAND (2026-08-14, four commits)

| commit | what |
|---|---|
| `fc0a492` | ConvNeXt-**S** and ConvNeXt-**B** on the verified path — 50.2 M / 88.6 M, 4 artifacts each |
| `ccca380` | three A3 fidelity deltas closed: `wdExcludeNormBias`, BN-momentum ÷ accum, the 80 dropped val images |
| `d96c7fa` | RandAugment **Posterize** was one bit too aggressive — a transcription bug, found by diffing timm 1.0.28 |
| `3732a11` | measured the BILINEAR→BICUBIC flip and **did not make it** — it would have been ~60× worse |

Green at HEAD: `lake build LeanMlir`/`Proofs`, `regen_verified_mlir.sh check` (185 artifacts),
`check_render_coverage` (172/154 guarded), `shim_wiring_gate`, `randaug_timm_diff`,
`geo_aug_pil_diff`, `docstring-checkrefs`.

⚠ **8 commits ahead of `origin/main` and unpushed.** That is why the shim-wiring gate's red streak
(§5) never showed up in CI.

---

## 1. ⚠⚠ EVERY NET NEEDS RE-RUNNING, AND HERE IS EXACTLY WHY

`d96c7fa` changed **Posterize**'s magnitude mapping. At RSB's own m=7 timm keeps **2** MSBs and we
were keeping **1** — a whole bit more posterisation, on one of the fifteen RandAugment ops, on
every image that drew it. That is a real change to the training distribution.

▶ **So no accuracy number produced before `d96c7fa` is comparable to one produced after it.** Not
wrong — it was a real run of a real recipe — but a curve spanning the commit is two experiments.

⚠ It is in `jax/Jax/Codegen.lean`, the **shared** emitter, so it moved BOTH the verified path's
shim and the JAX reference trainer's own augmentation. The two sides stay comparable *to each
other*; both moved relative to their own history.

⚠ The on-disk `jax/.lake/build/generated_*_imagenet*.py` TRAINERS still carry the old formula —
only the shims were regenerated. Harmless (each exe rewrites its `.py` before spawning Python) but
do not read those files as current.

⭐ **This is the licence to restructure freely.** Nothing downstream is being protected, so take
the format changes and interface changes now rather than after a fresh set of numbers exists.

---

## 2. ▶ ITEM 1 — the full-val eval tool (start here; it is the only one that unblocks others)

**What it is:** score a CHECKPOINT, standalone, without retraining. The JAX side has six
`eval_*_full50k.py`; the verified side has none, so a verified number can only be produced
in-training and only for the weights that were live at that moment.

### 2a. ✅ LANDED 2026-08-14 — `VerifiedNet.scoreCheckpoint` + `lake build score-checkpoint`

    LEAN_MLIR_VARIANT=<v> .lake/build/bin/score-checkpoint <net> [dataDir]
    # LEAN_MLIR_CKPT (default: ckptPathFor, i.e. the file the run just wrote)
    # LEAN_MLIR_REGION  auto | live | ema

The factoring below, exactly as planned — no new MLIR, no new ops, no renderer work. What it
turned into beyond the plan:

* ⭐ **`VerifiedVariant`** — the five variant-axis predicates (`emaOn`/`rmsOn`/`sdOn`/`cdOn`/
  `accOn`) plus `accK`/`nRegions`/`nScalars` now live once, in `VerifiedTrain.lean`.
  `trainAdamSched` computed all five inline and `tests/TestVariantPredicates.lean` declared its own
  `private def` of each — so that table gated a **transcription** of the driver, not the driver, and
  an edit to the real predicate could not turn it red. It now opens the shared namespace: 61
  spellings against the definitions that actually run. ▶ §5's lesson one level up.
* ⭐ **`ckptPathFor`** — the backend-scoped, `$LEAN_MLIR_CKPT_TAG`-suffixed path is one function,
  shared with the trainer, so "score the checkpoint the run just wrote" cannot land on a different
  name than the one that wrote it.
* `loadData` takes `evalOnly`, which skips a 7.4 GB Imagenette train read for a job that scores
  3,925 val images. Inert on `.imagenet` (it streams).

⚠ **The equality gate is written but NOT YET RUN**: this box's pinned venv broke on 2026-08-14
(`nvidia-cudnn-cu13 9.20.0.48` installed over `nvidia-cudnn-cu12 9.23.2.1` — same
`nvidia/cudnn/lib/` directory, clobbered in place), and JAX cannot execute a conv on it at all,
so nothing GPU-side ran. The three refusal paths were exercised and are correct. Repair is
`pip install --force-reinstall nvidia-cudnn-cu12==9.23.2.1` in `.venv`.

▶ The gate, when the box is back — ConvNeXt-T Imagenette, `.lake/build/convnext_adam_ckpt_xla.bin`
at epoch 80, whose training run printed
`epoch 80: val_acc = 3339/3925 = 85.070064%  top5 = 3819/3925 = 97.299363%`
(`runs/2026-08-12-convnext-imagenette-xla-cuda/`). `score-checkpoint convnext data` must print
those same counts.

### 2a (original plan). Part 2 first — the script, which is a factoring job

Everything needed already exists inside `trainAdamSched`'s eval half:

| need | already there |
|---|---|
| drain the val split | `loadData` (all 50,000 as of `ccca380`) |
| the eval graph | `mkSession` on `<slug>_fwd_eval`, or `<slug>_fwd` for the LN nets |
| batching a short tail | `F32.sliceImagesPad` + `min bs (nEval − bi·bs)` |
| forward | `LowererSession.forwardF32` |
| metrics | `F32.argmaxN` (top-1), `F32.rankOf` (top-5) |

So: lift the loop into `VerifiedNet.scoreCheckpoint (net, variant, ckptPath, region)`.
**No new MLIR, no new ops, no renderer work.**

⭐ **Gate it on ConvNeXt or ViT.** They have `nBnStats = 0`, so the tool works on their EXISTING
checkpoints today with nothing else changed — and the standalone number must equal the
in-training number for the same checkpoint. That is a real equality gate, not a smoke test, and it
is available before any of §2b lands.

### 2b. Part 1 — the blocker for the BN nets, and it is a DATA problem

⚠⚠ **The checkpoint is exactly `thetamv`** — `[θ|m|v]`, or four regions with EMA/accum — and
**the BN running mean/var are NOT in it** (`VerifiedTrain.lean:1328`: "reset per process and
rebuilt within an epoch"). In-training eval works because the stats have been accumulating all
epoch. A fresh process reading a `.bin` has ZEROS, and `@<slug>_fwd_eval` then normalises with
zeros — garbage, not a slightly-off number.

**So R50, R34, MNv2, EfficientNet and MNv4 cannot be re-scored from a checkpoint at all today.**
Two exits, and they are not exclusive:

* **(a) append the `nBnStats` floats to the checkpoint.** Cleanest going forward. ⭐ The size guard
  at `VerifiedTrain.lean:1367` already makes the format change LOUD — an old checkpoint refuses
  with a byte count rather than misaligning every parameter. ⚠ **It buys nothing retroactively:
  A3's finished checkpoint does not contain them**, so the run you would most want to re-score is
  exactly the one this does not reach.
* **(b) BN recalibration** — ~100–200 training batches forward to re-accumulate, then eval. Works
  on existing checkpoints INCLUDING A3's. ⚠ It is a different estimate from the run's own and has
  to be said out loud when quoting.

▶ Recommendation: **(a) for the format, (b) as a `--recalibrate` fallback.** Together they cover
both futures, and (b) is what lets A3 be re-scored at all.

### 2c. What the tool buys once it exists

* Re-score a finished run without retraining — the whole point.
* ⭐ **Score the EMA shadow separately.** Today the driver picks live-or-shadow at TRAIN time
  (`emaLiveBn`); one checkpoint cannot yield both numbers. timm reports the shadow, and RSB-A2 sets
  `emaDecay := 0.9999` — so without this, an A2 result is not quotable the way its reference is.
* Score at a different resolution or crop (FixRes-style test-time tweaks) with no re-render.
* Rank checkpoints across epochs post hoc.

---

## 3. ▶ ITEM 2 — the three R50 renderer features A2/A1 need

`resnet50TrainStepFaithfulB` takes `(B nClasses epsStr replicas opt slug bce vSuffix q wdExclude)`.
Of the four recipe knobs RSB-A2 needs beyond A3, **one landed today** and three have not:

| knob | A3 | A2 | verified status |
|---|---|---|---|
| `wdExcludeNormBias` | (missing) | required | ✅ **done `ccca380`** — `wx` renders exist |
| stochastic depth | 0.0 | **0.05** | ⛔ **R50 has no `sd` flag at all** |
| EMA | off | **0.9999** | ⛔ **R50 has no `ema` flag** (no 4th region) |
| weight decay | 0.02 | 0.02 (**A1: 0.01**) | ⛔ `%wd` is a BAKED constant — A1 needs `wdStr` |

⭐ **All three have working precedents to copy**: ConvNeXt and ViT both carry `sd` and `ema`;
ConvNeXt carries `wdStr`. None needs a new `SHlo` op.

⚠ **Stochastic depth is the biggest of the three** and the one with a real design question: where
the site goes in a bottleneck, and whether every block gets one. R50 has 16 blocks, so the ramp
denominator is 15 — but the stage-first blocks have a PROJECTION shortcut, which is exactly the
case EfficientNet's skip-guard exists for. Read `planning/stochastic_depth.md` §7b (the
misplacement control) before placing a single site: an all-ones-mask gate is structurally blind to
placement, and `scripts/misplace_drop_sites.py` is what makes a green run mean anything.

### 3b. The A2 train step itself is ONE `#eval`

`resnet50in_lambaccdp8x64bce` (224, LAMB, BCE, k=8, 4 replicas) does not exist; the identical
config at 160 does. Same for its `wx` and (once §3 lands) `drop`/`ema` peers. The machinery is all
there — it is a slug and a `q`.

⭐ **The 224 DATA side is already done and this surprised me**: `generated_resnet50_imagenet_shim.py`
already carries RandAugment **m7** (A3 is m6) and **repeated augmentation 3×**. Checked, not
assumed.

---

## 4. ▶ ITEM 3 — `cBS`, which is now the top infrastructure item

`ConvNeXtRender.lean:41 private def cBS : Nat := 32`, and R50/R34 bake their batch the same way.
`planning/vit_convnext_sb_scaleup.md` was updated this session to promote this from hygiene to the
main blocker, for two independent measured reasons:

1. **It starves big hardware.** bs32 kernels cannot fill an A100/H100, so most of what you would
   rent is unusable — and the TF32 result says memory bandwidth is the ONLY axis that matters for
   these nets, which is exactly the part bs32 fails to collect.
2. **It is the only lever on the all-reduce.** The collective cost is per STEP and independent of
   batch, while step count falls linearly with it. bs32 → bs128 cuts all-reduces per epoch 4×. On
   this box that is 24% of wall clock; on an 8×4090 rental it is 56%.

▶ Do it **before** ConvNeXt-B or A2 ever run, not after.

---

## 5. ⚠ THE GATE THAT WAS RED, AND THE LESSON UNDER IT

`shim_wiring_gate.py` asserted a hardcoded **8** `.imagenet` nets. ViT-S/B took it to 10 (`b9ea36f`)
and ConvNeXt-S/B to 12 (`fc0a492`); it had been failing locally since, and CI never saw it because
those commits are unpushed. Fixed in `ccca380` — but **not by bumping the number**: "all resolve to
DISTINCT shims" is the wrong invariant once size variants exist, since ViT-S/B and ConvNeXt-S/B
correctly SHARE their base net's pipeline. It is now "base nets pairwise distinct AND each variant
names exactly its declared base's shim", which is strictly stronger.

⭐⭐ **And the deeper lesson, which is what actually cost us a bit of Posterize:** that gate compares
the shim to the reference by augmentation **PARTITION** — *which* ops are on — and never by the
**ARGUMENT** each op is called with. The partition was right the whole time. **A gate on "is the
feature enabled" is not a gate on "is the feature correct."** `scripts/randaug_timm_diff.py` is now
the second kind; look for other places that have only the first.

---

## 6. MEASURED THIS SESSION — numbers to plan against, not to quote

4060 Ti, fp32, `*wxclipdrop` / `lamb64bce`, single GPU unless stated.

| | 1×32 | 4×32 (DP) | notes |
|---|---|---|---|
| ConvNeXt-T | 167 ms | 220 ms | control |
| ConvNeXt-S | 280 ms | 366 ms | 61 min/epoch, 305 h @300ep |
| ConvNeXt-B | 408 ms | 539 ms | 90 min/epoch, 450 h @300ep; **fits at bs32**, unlike ViT-B |
| R50 @160 (A3) | 168 ms | — | the shipped A3 config measured 240 ms/micro-step at 4×64×k8 |
| R50 @224 (A2) | 317 ms | — | **1.89× the 160 cost**; A2 ≈ 193 h, A1 ≈ 387 h on this box |

⭐ **The DP tax is one bandwidth line**: ring traffic `2·(N−1)/N·(params × 4 B)` over a **3.5 GB/s**
effective link. Fitted on ConvNeXt-S alone, it predicted T to 8% and B to 14%. On NVLink it goes to
~1 ms. ▶ Interconnect, not FLOPs, is the thing to buy.

⚠ **TF32 is worth only 10%** (280 → 308 with `NVIDIA_TF32_OVERRIDE=0`), so these nets are
memory-bound, not tensor-FLOP bound. Rank hardware by GB/s.

⚠ **Host RAM peaks at 36.5 GB**, almost all of it the unconditional val drain. A 32 GB rental OOMs
before step 0 whatever its GPU is.

---

## 7. WHAT IS STILL DIFFERENT: verified A3 vs its JAX reference

Four, down from seven this morning.

| | JAX | verified |
|---|---|---|
| **BN group** | 512 (batch axis mesh-sharded ⇒ XLA all-reduces) | 64 per replica |
| precision | bf16 | fp32 (~2× throughput) |
| train tail | — | 4 micro-batches/epoch, 0.08% |
| mixup λ RNG | `jax.random` | numpy `Generator` — distributional only |

⭐ **Only the BN group has a plausible accuracy effect**, and its SIGN is unknown: Hoffer et al. is
usually cited for *smaller* ghost batches helping at large effective batch, which would make our 64
a benefit. It remains a live hypothesis for why the verified path came out AHEAD (77.43 vs 77.22).

**Not gaps:** EMA (A3 sets `useEMA := false` on both sides). Posterize/Solarize/interpolation are
BOTH-sides-vs-timm, not verified-vs-JAX — they come from the shared emitter.

---

## 8. SUGGESTED ORDER

1. **Full-val eval Part 2**, gated on ConvNeXt/ViT (§2a). Unblocks everything else's measurement.
2. **Full-val eval Part 1(a)+(b)** (§2b) — the checkpoint format plus `--recalibrate`.
3. **`cBS` threading** (§4). Do it before anything is rendered at a new size.
4. **R50 `sd` / `ema` / `wdStr`** (§3), stochastic depth last of the three and with the
   misplacement control.
5. The A2 `#eval`s (§3b) — trivial once 3 and 4 land.
6. Only then: talk about compute.

⚠ Steps 1–5 are all CODE. None of them needs a GPU beyond a `LEAN_MLIR_MAX_STEPS` probe, and none
produces a quotable number. That is the point of this session.
