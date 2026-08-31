# imagenette_error_intervals.md — n=3 across every Imagenette trainer, and error bars in the book

**Opened 2026-08-30**, at the end of the session that closed the three §7 paper diffs and then
watched a ConvNeXt re-run produce a result it could not interpret.

**This is self-contained.** A session that has read nothing else can execute it.

---

## 0. Why, in one block

The book publishes a five-row Imagenette table (`content.tex` ~line 10645) of **single runs**, to two
decimal places, with no interval. Today's ConvNeXt re-run is the argument for fixing that:

| | pre-B1 | with the head LN | |
|---|---|---|---|
| final, epoch 80 | 85.07% | **85.45%** | +0.38 pt |
| best epoch | 85.22 (e75) | 85.71 (e70) | |
| 95% Wilson CI | 83.92–86.15 | 84.31–86.52 | fully overlapping |

⛔ **+0.38 pt is 15 net label flips out of 3,925, and it is not resolvable.** Not marginally: even a
PAIRED McNemar would need the two models to disagree on fewer than **58** images (1.5%) for 15 flips
to clear p<0.05, and two independently-trained nets disagree on far more. At n=1 there is nothing to
settle.

⚠⚠ **And the mid-run trend reversed twice.** The gap read +5.2 at epoch 20, +3.3 at 40, +0.7 at 60,
+0.38 at 80. Two confident readings were given during that run and both were withdrawn. That is the
case for error bars better than any argument about statistics: **the quantity being reported has a
spread comparable to the effects being claimed, and nothing in the book says so.**

▶ So: run every Imagenette trainer **3×**, publish **mean ± std** beside the existing Wilson CI, and
stop quoting single runs to 2 dp.

✅ **DONE 2026-08-31 — and at n=5, not n=3. Results and what they change are in §8.** The short
version: σ ranges 0.14–0.64 across the seven nets (so there is no single error bar), the n=3 σ
estimates were wrong by up to 75% (§8.1), ResNet-34 does not reproduce its published number (§8.2),
and GPUs 1 and 5 are now cleared under load (§8.4).

---

## 1. The machinery already exists — it landed today, don't rebuild it

* **`wilson95 correct nEval`** (`LeanMlir/VerifiedTrain.lean`) appends `[95% CI lo–hi]` to every eval
  line, in `trainAdamSched` and `scoreCheckpoint` both. Wilson, not the normal approximation, which
  collapses to ±0 at 0/n. Already in the log format — nothing to add.
* **`LEAN_MLIR_DUMP_CORRECT=<prefix>`** writes one byte per val image (1 = top-1 correct, eval
  order). Off by default.
* **`scripts/mcnemar.py`** — the paired test over two of those bitmaps. Exact binomial.
  ⚠ Useful for "are these two CHECKPOINTS different"; **silent** on "is this recipe better", which is
  what this document is about. Do not substitute one for the other.
* **`LEAN_MLIR_SEED`** (`VerifiedTrain.lean:1546`, default **1**) seeds parameter init, incrementing
  once per tensor. This is the knob that makes n=3 meaningful.
* **`LEAN_MLIR_CKPT_TAG`** (`VerifiedNet.ckptPathFor`, `VerifiedTrain.lean:933`) appends a run-scoped
  suffix to the checkpoint path. Its docstring was written for exactly this case, so §3.3's
  "move the checkpoint between seeds" is unnecessary — tag per seed and nothing is moved or deleted.
* ⭐ **`scripts/seed_sweep.sh`** (added 2026-08-31) is the runner: LPT packing over N GPUs,
  every §3 trap handled, `DRY_RUN=1` to print the plan, resumable via per-job done-markers, and it
  refuses to start if its AER source is unreadable. **Do not run these binaries by hand** — the
  environment is the experiment.

---

## 2. The run matrix

Seven trainers, all `80 epochs / bs 32`, all `apps/imagenette/Main*VerifiedAdam.lean`:

| exe | step | one run | in the book's table? |
|---|---|---|---|
| `resnet34-verified-adam` | 220 ms | 1.5 h | ✅ 89.71 / 98.27 |
| `mobilenetv2-verified-adam` | 90 ms | 35 min | ✅ 89.25 / 98.68 |
| `efficientnet-verified-adam` | 103 ms | 41 min | ✅ 89.96 / 98.45 |
| `convnext-verified-adam` | 196 ms | 1.3 h | ✅ 85.07 → **85.45 today** |
| `vit-verified-adam` | 61 ms | 24 min | ✅ 68.74 / 90.42 |
| `resnet50-verified-adam` | `[see below]` | **~1.4 h** | ⛔ no published number |
| `mobilenetv4-verified-adam` | `[see below]` | **~42 min** | ⛔ no published number |

⭐ **R50 and MNv4 were MEASURED 2026-08-31**, one epoch each with `LEAN_MLIR_MAX_EPOCHS=1` and
residency on, which is what this row asked for. Both estimates were high:

| | est. | wall, 1 epoch | compile | ⇒ 80 epochs |
|---|---|---|---|---|
| `resnet50-verified-adam` | ~2.5 h | 73.1 s | 10.4 s | **~1.4 h** |
| `mobilenetv4-verified-adam` | ~50 min | 41.5 s | 10.2 s | **~42 min** |

⚠ Those wall figures include eval over all 3,925 val images and one-off startup, so they bound the
epoch cost from ABOVE — but they do **not** yield a `step` ms for the book's table, which is train
steps only. That column stays unmeasured for these two.

**Cost.** 4.47 h/seed for the five + ~2.1 h measured for the other two ≈ **6.6 h/seed**, so
**~19.7 GPU-hours** at n=3 — down from the 23.5 this section first estimated. LPT-packed, one
trainer per GPU:

| GPUs | makespan |
|---|---|
| 4 (the AER-clean `0,2,3,4`) | **~4.9 h** |
| 6 | **~3.3 h** |

⚠ **The "6" was unverified**; established practice is the four in `scripts/supervise.sh`'s job
confs (⚠ this section said `scripts/aer_watchdog_epoch.sh`, which does not exist). The kernel
journal showed zero AER events in 14 days, which proved nothing, because the two excluded cards
had not been under load to throw any. You cannot clear a card by watching it idle.

▶ **Done 2026-08-31**: the two probes above ran ON GPUs 1 and 5 specifically, with the watchdog
armed. Both completed clean, `RESIDENT:` lines present, no AER. That is one epoch of evidence, not
a clearance — the watchdog stays armed for the sweep and stops it rather than finishing degraded.

---

## 3. ⚠⚠ Seven traps, every one of which has already cost someone a run here

1. **`PJRT_FFI_RESIDENT=1` — SET IT.** Off by default (`ffi/pjrt_ffi.c:284`, and the comment calls
   the default deliberate). Today's ConvNeXt run did not set it and produced **no `RESIDENT:` line
   and 2 h 14 m against the book's 1 h 19 m**. ⛔ Note what that means for the book: §8.1 prints four
   commands with no environment and then shows a transcript containing a RESIDENT line — **the
   printed commands do not reproduce the printed output**, and have not for some time. Fix the
   section by adding the variable, not by deleting the line.

2. **`LEAN_MLIR_MAX_EPOCHS` is `min n cfg.epochs` — it can only LOWER.** A net whose checkpoint is
   already at `cfg.epochs` resumes, prints `done`, and exits **with no epoch line at all**. That is
   completion, not failure, and it makes a capped sweep a silent no-op on every finished net.

3. **Checkpoints must be isolated per seed or the runs resume each other.** `trainAdamSched` writes
   `.lake/build/<slug>_<variant>_ckpt_xla.bin` — a fixed path with no seed in it. Seed 2 started
   after seed 1 finishes will **resume seed 1's epoch 80 and exit immediately**. ▶ Move or delete the
   checkpoint between seeds, and **back it up first**: this session's §1.2d records an
   `efficientnet_adam` epoch-80 state deleted while chasing a phantom bug.
   ⭐ **Superseded 2026-08-31** — `LEAN_MLIR_CKPT_TAG=s<seed>` (§1) gives each seed its own path,
   so nothing is moved, deleted, or backed up, and the existing untagged checkpoints are untouched.

4. **`efficientnet-verified` ≠ `efficientnet-verified-adam`**, and the plain one ignores
   `LEAN_MLIR_VARIANT`. Every published number comes from the `-adam` binary. Quote it exactly.

5. **ConvNeXt checkpoints are incompatible pre/post-B1** (180 vs 182 param tensors). The size guard
   refuses loudly, which is the good failure — but the pre-B1 epoch-80 state at
   `~/ckpt_backups/convnext_adam_ckpt_xla.bin.pre-B1-e80` **cannot be scored by the current binary**,
   so no paired comparison against it is possible. It is history, not a baseline.

6. **One trainer per GPU.** XLA preallocates ~75% of a card; two will not fit. And ⚠ **the box
   crashes on long runs** — supervise, don't fire-and-forget.

7. ⚠⚠ **`dmesg` is unreadable here, and fails SILENTLY.** `kernel.dmesg_restrict=1`, there is no
   passwordless sudo, and `dmesg` exits non-zero with **empty output** — so the obvious watchdog,
   `dmesg | grep -c AER`, reports "0 events" forever and looks perfectly healthy. That false green
   was live in this sweep's own runner for one revision. **`journalctl -k` is the source that
   works** (it is what `scripts/supervise.sh` has always used), and it must be filtered with
   `grep -ivE "no action required"` or benign corrected-error chatter trips it. ▶ A watchdog that
   cannot read its source must REFUSE, not pass; the runner prechecks exactly that.

⚠ **Do not run anything CPU-heavy beside the sweep.** Today's wall clock was also contaminated by a
3,923-job `lake build Certs` and two IREE forward ties running alongside; the accuracy was unharmed
but the timing column was destroyed. If the table's Step time / Total columns are being refreshed,
the box must be otherwise idle.

---

## 4. What to publish

For each net, over the 3 seeds:

* **mean ± std** of final-epoch top-1 and top-5 — the seed-variance number, which is what a reader
  needs to compare two rows of the table.
* **the Wilson CI** on any single run quoted — the measurement number, already printed.
* ⭐ **State which epoch is being quoted.** Final vs best differs by ~0.3 pt on ConvNeXt (85.45 vs
  85.71) and the choice is currently unstated. Both runs also wobble ~0.1 pt across epochs 75–80, so
  "best" is partly a max over noise. **Final epoch** is the defensible choice; say so once.

⚠⚠ **The two intervals mean different things and must not be added or conflated.** Wilson answers
"how well do 3,925 images pin THIS model"; the std answers "where would another seed land". A table
carrying only one of them invites the wrong inference — which is the inference this whole document
exists to prevent.

▶ **n=3 gives 2 degrees of freedom**, so the std is itself crudely estimated. Report it as a spread,
not as a precise σ, and do not run significance tests on it.

---

## 5. ▶ DEFERRED — the §8.1 "Run it first" repair (drafted 2026-08-30, then withdrawn)

Held back deliberately: it wants to be done ONCE, from a clean run, not twice. Three separate
defects, only one of which needs the sweep.

**(a) The printed commands do not reproduce the printed output.** §8.1 lists

```
lake build convnext-verified-adam
./.lake/build/bin/convnext-verified-adam data
```

and then shows a transcript containing `[pjrt_ffi] RESIDENT: … holds 540 parameter tensors
(318.4 MB) on 1 device`. Residency is **off by default** (`ffi/pjrt_ffi.c:284`; the comment calls
the default deliberate), so the reader who runs those four commands gets a different run.
⭐ **It is a §8.1 defect, not a book-wide one**: `content.tex:5404` and `:5737` — the ImageNet job
listings — both set `PJRT_FFI_RESIDENT=1` correctly. Only the Imagenette section omits it.
▶ Fix: add the variable to the listing plus one sentence on what it does. An edit adding it was
written and reverted on 2026-08-30; redo it as part of the whole-section pass.

**(b) The accuracy figures are superseded.** 85.07 / 97.30 is the pre-B1 net. Today's current run is
**85.45 / 97.35**, best epoch 85.71 (e70–71), epoch 1 at 39.03, `549` train-step outputs (was 543),
PJRT 0.114 (was 0.112), and the description line now reads `→ GAP → LN → dense`. Per `85daffb`'s
rule, the superseded number is **deleted, not annotated**.

**(c) The timing claim cannot be refreshed from today's run.** "About 59 seconds each, one hour and
nineteen minutes" — today's took 2 h 14 m with residency OFF and a `lake build Certs` plus two IREE
ties running alongside. Both confounds hit wall clock only; accuracy is unaffected. ▶ A clean
`PJRT_FFI_RESIDENT=1` run on an idle box is what closes this, which is exactly what the sweep
provides if ConvNeXt's seed 1 is run that way.

⚠ **Anchored edits only, and check after each.** `planning/vit_verified_run.md` records slicing to
the first match of a heading four chapters share, which duplicated ~4,100 lines. The 4-line command
block and the `85.07` occurrences were each verified unique before editing on 2026-08-30; verify
again, the file has moved since.

▶ **Also stale beyond §8.1**, from the same run: `content.tex:8805` and `:10648` both carry
ConvNeXt-T's `85.07\% & 97.30\%` in comparison tables, and `:8816` argues from `85.07\%` against
EfficientNet-B0's `89.96\%`. The sweep's mean ± std replaces all of them at once — which is the
argument for not touching any of them now.

---

## 6. Open decisions for whoever picks this up

* **Refresh the timing columns too, or accuracy only?** The timing needs an idle box and
  `PJRT_FFI_RESIDENT=1`; the accuracy does not. Doing both at once is cheaper than twice.
* **Does the comparison table become 7 rows?** R50 and MNv4 have trainers and no published numbers.
  Adding them is nearly free once the sweep runs.

---

## 8. ✅ RESULTS — the sweep ran 2026-08-31, n=5

**35/35 jobs, zero failures, zero AER.** Two passes into `runs/2026-08-31-imagenette-n3/`:
seeds 1–3 (21 jobs, 03:46→06:37, **2 h 51 m**) then seeds 4–5 (14 jobs, 12:37→14:34, **1 h 57 m**),
six GPUs, `PJRT_FFI_RESIDENT=1`, one `LEAN_MLIR_CKPT_TAG` per seed. Runner:
`scripts/seed_sweep.sh`.

⭐ **All figures are FINAL EPOCH 80**, per §4's ruling. 95% CI is the t-interval on the seed mean
(t=2.776, df=4) — the *seed* interval, NOT the Wilson CI, which answers a different question (§4).

| net | seeds 1..5 | mean | std | 95% seed CI | book (n=1) |
|---|---|---|---|---|---|
| ResNet-34 | 90.17 90.37 90.04 90.14 89.68 | **90.08** | 0.25 | [89.77, 90.39] | 89.71 ⚠ |
| ResNet-50 | 89.68 89.96 89.73 89.58 89.71 | **89.73** | 0.14 | [89.56, 89.91] | ⛔ none |
| EfficientNet-B0 | 90.06 89.50 90.50 89.45 89.91 | **89.89** | 0.43 | [89.35, 90.42] | 89.96 ✅ |
| ConvNeXt-T | 85.40 85.66 85.38 85.35 85.78 | **85.51** | 0.19 | [85.27, 85.75] | 85.45 ✅ |
| MobileNetV2 | 88.76 88.74 89.32 89.32 89.07 | **89.04** | 0.29 | [88.69, 89.40] | 89.25 ✅ |
| MNv4-Conv-M | 86.39 85.15 86.90 86.24 86.09 | **86.16** | 0.64 | [85.36, 86.95] | ⛔ none |
| ViT-Tiny | 68.87 69.17 68.54 69.15 67.82 | **68.71** | 0.56 | [68.01, 69.40] | 68.74 ✅ |

Top-5 means (n=5, same order): 98.46, 98.76, 98.34, 97.24, 98.63, 97.58, 90.07.

### 8.1 ⭐⭐ Why n=3 was not enough — and this is the section to quote

§4 warned the n=3 std is crudely estimated. It is worse than "crude". Recomputing σ from the first
three seeds against all five:

| | R34 | R50 | B0 | CNX | MNv2 | MNv4 | ViT |
|---|---|---|---|---|---|---|---|
| σ at n=3 | 0.17 | 0.15 | 0.50 | 0.15 | 0.33 | **0.90** | 0.32 |
| σ at n=5 | 0.25 | 0.14 | 0.43 | 0.19 | 0.29 | **0.64** | **0.56** |
| change | +51% | −6% | −14% | +26% | −13% | −29% | **+75%** |

⛔ **Four of seven moved more than 25%; ViT's nearly doubled** (its seed 5 came in at 67.82, well
below the other four) and MNv4's fell 29%. An error bar published off n=3 would have been wrong for
at least two nets, and MNv4's — the net with no published number at all — would have been
**overstated by 40%**. ▶ n=5 is the floor for this table. Do not publish n=3 spreads.

⚠ **σ is not one number**: 0.14 (R50) to 0.64 (MNv4), a 4.6× spread. A single global error bar on
the table would be wrong. Each row carries its own.

### 8.2 ⚠ ResNet-34 does not reproduce, and it is the only one

R34's 95% seed interval **[89.77, 90.39] excludes the book's 89.71**. Every other published net
brackets its book value. The extra seeds narrowed it (seed 5's 89.68 pulled the mean 90.19 → 90.08)
but did not close it, so this is not a small-sample artifact.

⭐ **It pairs with a timing miss in the same direction.** R34 ran **44.7 min** here against a
published **1.5 h** — while ConvNeXt (74.9 min), B0 (39.5), MNv2 (36.5) and ViT (23.8) all reproduce
their book timings *despite* six-way contention. Two independent misses on one net point at the
published row coming from an OLDER CODE STATE, not at anything wrong today.
▶ Next step is `git log` on the R34 Imagenette path, not more seeds.

### 8.3 The ConvNeXt head-LN question this document opened with

ConvNeXt is one of the tightest nets, σ = 0.19. The pre-B1 **85.07** sits **2.3σ** below the
post-B1 mean and outside [85.27, 85.75] — suggestive of a real improvement, and no longer the
"nothing to settle" of §0.

⛔ **But it is still not settled, and more post-B1 seeds cannot settle it.** 85.07 is a single
observation compared against an interval for a mean; the pre-B1 arm has n=1 and no σ of its own.
Closing this needs pre-B1 SEEDS — i.e. reverting the head LN and retraining 3–5×. Whether that is
worth ~4 GPU-hours to confirm a +0.4 pt effect is a judgement call, not a measurement.

### 8.4 ✅ GPUs 1 and 5 are clean — the six-way question, settled

§2 said you cannot clear a card by watching it idle. Across **both** sweeps, GPUs 1 and 5 each
carried **six full 80-epoch runs** under sustained load: `journalctl -k` reports **zero** AER events
(filtered for the benign `no action required` chatter). That is real evidence, and the 6-GPU
makespan row is no longer marked unverified.

### 8.6 ▶ TODO — the n=5 CHECKPOINTS are gone; only the numbers survive

⛔ **All 35 Imagenette checkpoints were deleted 2026-08-31** by an over-broad
`rm .lake/build/*_ckpt_xla_s[1-5].bin` while clearing the small-suite tags before a re-run.
▶ §3.3's "back it up first" is there for exactly this and was not followed.

**What survives, and it is most of the value:** every accuracy (the per-seed logs in
`runs/2026-08-31-imagenette-n3/`, and §8's tables), and all **1,680 per-example bitmaps** in
`bitmaps/` — which is what `scripts/mcnemar.py` actually consumes, so paired tests are unaffected.

**What is lost:** the trained weights, so none of those 35 models can be re-scored at a different
eval setting or used as a bootstrap.

▶ **Regenerated at n=1** (seed 1, all seven nets) on 2026-08-31 into
`runs/2026-08-31-imagenette-seeds/` — enough to have a usable checkpoint per net again.
▶ **TODO: the full n=5 regeneration**, ~4 h 48 m on six GPUs, whenever weights for seeds 2–5 are
actually needed. Nothing in §8's published numbers depends on it.

---

### 8.5 ⚠⚠ The timings here do NOT close §5(c)

These ran **six at a time**, so they are contended and cannot refresh the book's Step time / Total
columns — §3's own warning ("the box must be otherwise idle"). They bound the cost from above and
are useful as a cross-check (which is how §8.2 uses them), nothing more. The clean single-job
timing run §5(c) needs has still not happened.

---

## 7. State of the tree as this was written

* Everything through `c205cea` is on `origin/main`; `85daffb` (MNv2 §6.5) pulled in on top.
* ConvNeXt's Imagenette checkpoint is **fresh at epoch 80 with the head LN**, 85.45 / 97.35,
  `.lake/build/convnext_adam_ckpt_xla.bin`. Log at
  `runs/2026-08-30-convnext-imagenette-headln/train.log` (untracked).
* ⚠ That run is **current** — B1 is in it, B3 is inert (ConvNeXt has no BatchNorm), and B2 is inert
  (this config runs neither mixup nor cutmix). It can serve as seed 1 for ConvNeXt **only if** the
  sweep also runs without `PJRT_FFI_RESIDENT`; otherwise re-run it for consistency.
