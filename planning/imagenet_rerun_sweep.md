# imagenet_rerun_sweep.md — every ImageNet run that needs redoing, and why

**Opened 2026-08-15.** Between 2026-08-14 and 2026-08-15 five changes landed that move either the
training distribution, the eval denominator, or the optimizer. **No ImageNet number in this repo is
comparable across that boundary**, and for seven of the configs no run was *possible* at all. This
file is the inventory: what changed, what it touched, and what has to be re-run before anything is
quoted again.

▶ Read §1 before deciding the order. Read §3 before starting the sweep — two of the prerequisites
are cheap and one of them gates everything.

---

## ⭐ STATUS 2026-08-22 — the ResNet half of the sweep is DONE

Four runs landed 2026-08-16 → 08-22, all post-C2/C3/C4/C5/C6 and all scored over **50,000**.
These are directly comparable to each other and supersede every number below that predates them.

| run | ours | reference | Δ |
|---|---|---|---|
| R50 `2018`, JAX, 90 ep | **76.95 / 93.44** | ~76.1 torchvision | +0.85 |
| R50 RSB-A3, **JAX reference**, 100 ep | **78.26 / 93.79** | 78.052 / 93.780 (`timm resnet50.a3_in1k`) | +0.21 |
| R50 RSB-A3, **VERIFIED** `wxclip`, 100 ep | **77.91 / 93.84** | — | −0.35 vs the reference above |
| R34 `2018`, JAX, 90 ep | **74.16 / 91.92** | 73.298 / 91.422 (`timm resnet34.tv_in1k`) | +0.86 |

⚠ **The verified-vs-reference −0.35 is NOT one-variable.** The reference normalises BN over 512,
the verified path over 64 — an 8× difference in the Ghost-BN group. See
`inflight_r50_queue.md` §4.

⚠ **§5's ordering and §6's table below are the pre-run state** and are kept for the reasoning,
not the numbers. Blueprint §5.7 and §5.8 carry the current figures.

▶ Still open: the **verified** 2018 peers (R50 and R34), MNv4/B0/MNv2 re-trains, and the whole
C1-blocked ViT/ConvNeXt family.

---

## 0. THE ONE-LINE VERSION

`d96c7fa` (Posterize) moved the training distribution for every RandAugment net, and its Solarize
sibling **crashed the shim outright** for every net with `mstd > 0` — so those seven have not been
trainable since 2026-08-14 and nobody noticed, because nothing tried to train until 2026-08-15.
Separately, `f8cd3a9` + `ccca380` changed how *every* net is evaluated.

---

## 1. WHAT CHANGED, AND WHAT EACH ONE TOUCHES

| # | change | commit | blast radius | effect |
|---|---|---|---|---|
| C1 | **Solarize `min()` on a symbolic tensor** | `d96c7fa` broke it, `8f8fdd1` fixed it | RandAugment nets with **`mstd > 0`** | ⛔ **the shim emitted ZERO bytes** — training could not start at all |
| C2 | **Posterize kept one MSB too few** | `d96c7fa` | every **RandAugment** net | training distribution moved (a whole bit at RSB's own magnitude) |
| C3 | **timm validation protocol + resampler** | `f8cd3a9` | **every** ImageNet net | resize/crop/antialias changed ⇒ eval moved |
| C4 | **eval denominator 49,920 → 50,000** | `ccca380` | **every** ImageNet net | every quoted top-1 was over the wrong denominator |
| C5 | **wire framing v1/v2 → v3/v4** | `db61adf` | **every** ImageNet net | the val drain could not read a partial batch; the run died at startup |
| C6 | **strided conv `SAME` → symmetric `(k-1)//2` padding** | `d078a6d` (2026-08-04) | **every** net with a strided conv | ⛔ **the emitted NETWORK changed** — every checkpoint written before 2026-08-04 is a different architecture from what the emitter builds today |

⚠ **C1 and C5 are not fidelity deltas, they are hard blockers.** A run either started or it did not.
C2/C3/C4 are the ones that make old numbers incomparable rather than impossible.

⚠⚠ **C6 was missing from this table when it was opened, and it is the one that decides whether a
re-score is legitimate at all** (§5.4). It is not an eval change and not a training-distribution
change: it changes the forward graph, so old weights scored through today's emitter are being run
in *a network they were never trained in*. `f8cd3a9`'s own commit message measured it on R50-A3 —
**77.150% through the matching 2026-08-03 forward vs 75.200% through the regenerated one, a silent
−1.95 pt** — and concluded that "the checkpoint outlives the artifact it was trained on" is
unguarded on the JAX path. Measured again 2026-08-15, see §6.

### 1a. Who is in C1's blast radius, measured not assumed

`_randaugment` only makes the magnitude symbolic when `mstd > 0`:
`mg = m if mstd <= 0.0 else tf.clip_by_value(tf.random.normal([], m, mstd), 0.0, _AA_MAX)`.

| net | augmentation | `mstd` | C1 (crash)? | C2 (distribution)? |
|---|---|---|---|---|
| **R50** (A3/A2/A1/rsb-faithful) | RandAugment | 0.5 | ⛔ **yes** | yes |
| **ViT** T / S / B | RandAugment | 0.5 | ⛔ **yes** | yes |
| **ConvNeXt** T / S / B | RandAugment | 0.5 | ⛔ **yes** | yes |
| **MNv4** | RandAugment | **0.0** | ✅ no — `mg` stays a Python float | yes |
| **EfficientNet-B0** | **AutoAugment** | — | ✅ no — policy magnitudes are constants | no |
| **MNv2** | none | — | ✅ no | no |
| **R34** | none | — | ✅ no | no |
| **R50 `2018` recipe** | none (`useRandAugment := false`) | — | ✅ no | no |

⭐ So the "everything is broken" reading is wrong and worth stating: **seven configs could not
train; four could, and were merely evaluated differently.**

---

## 2. THE INVENTORY

### 2.1 ⛔ Blocked outright (C1) — never ran since 2026-08-14

| net | recipe | status |
|---|---|---|
| R50 | `rsb-faithful`, `a2-accum`, `a1`, `default` (A2) | ⛔ C1 + C2 + C3 + C4 |
| ViT-T / ViT-S / ViT-B | ImageNet | ⛔ C1 + C2 + C3 + C4 |
| ConvNeXt-T / S / B | ImageNet | ⛔ C1 + C2 + C3 + C4 |

### 2.2 ⚠ Ran, but the numbers are not comparable

| net | last number | invalidated by |
|---|---|---|
| **R50 A3 verified** `lambaccdp8x64bce` | **77.43 / 93.60** (2026-08-07, 34.06 h) | C2, C3, C4 — and it also lacked `wdExcludeNormBias` and D1's clip |
| **R50 A3 JAX reference** | 77.22 / 93.34 (Jul-9→10), 76.66 / 93.03 (Jul-4) | C2, C3, C4 |
| **MNv4-Conv-M** 100 ep | 75.51 (lives OUTSIDE the repo, `/home/skoonce/mnv4_convm_100ep`) | C2, C3, C4 |
| **EfficientNet-B0** | `runs/enet_*_80ep_aug02.log` | C3, C4 only |
| **MNv2** | `runs/mnv2_*_80ep_aug02.log` | C3, C4 only |
| **R34 ImageNet** | `runs/r34in_30ep_4gpu_sup*.log` | C3, C4 only |

⚠ **Every top-1 in the blueprint is over 49,920**, not timm's 50,000 (C4). That is stated in
`blueprint/src/content.tex` but it applies to the whole results chapter, not one row.

### 2.3 ⭐ The verified path has NEW artifacts that have never run at all

Not a re-run — a first run. From `planning/verified_optimizer_parity.md`:

* `resnet50in160_lambaccdp8x64wxbce` — `wdExcludeNormBias`
* `resnet50in160_lambaccdp8x64wxclipbce` — **+ D1's gradient clip**; the artifact that most nearly
  implements RSB-A3 on paper. **Profiled 2026-08-15 at 206 ms/step (4×bs64, 160², synth) — identical
  to the baseline, so the clip is free.** ETA ≈ 31–34 h.

---

## 3. ⚠ PREREQUISITES — do these before the sweep, not during it

1. ⭐⭐ **The `.venv` is one `pip install` from breaking multi-GPU again.** timm had been installed
   into it, pulling `torch` and the whole **CUDA 13** stack; `nvidia-nccl-cu13` overwrote the pinned
   cu12 `libnccl.so.2` (they share a path), and driver 575.57.08 caps at CUDA 12.9 — so NCCL could
   not init and every DP run died with a misleading *"CUDA driver version is insufficient"*.
   ✅ Cleaned 2026-08-15 and verified (4-way `psum` correct, 6 devices).
   ⚠ **timm belongs in `.venv-timm`.** Re-installing it into `.venv` re-breaks this silently, and
   the failure looks like a driver problem rather than a package problem.
2. ⚠ **`iree-base-compiler==3.12.0rc20260428` is in `requirements-cuda-lock.txt` and no longer on
   the index**, so the lockfile cannot be satisfied from scratch. Nothing on the JAX/PJRT path needs
   it (it is `vjp_oracle`'s differential compiler), but the pin needs re-pointing.
3. **Regenerate before measuring.** `jax/.lake/build/generated_*.py` are build products and went
   stale twice this week — once silently. `scripts/gen_shims.sh` for the shims; the per-recipe exes
   for the references.

---

## 4. ⭐ WHAT IS ALREADY VERIFIED WORKING (2026-08-15)

Do not re-litigate these; they were measured today.

* **R50 `2018` recipe, JAX, 4 GPU, end to end** — trains, evals, checkpoints:
  `[Epoch 1] val_top1=3006/50000 (6.01%)` → `[Epoch 2] 6604/50000 (13.21%)`, 182 ms/step,
  **899 s/epoch ⇒ 22.5 h for 90 epochs**. ⭐ `/50000` confirms C4's denominator is live.
* **The verified R50 path, 1 and 4 GPU** — 165 / 206 ms/step, `wx`+clip free.
* **The val drain** — `val ready — 50000 images, 28710 MB`, C5 fixed.

---

## 5. SUGGESTED ORDER

The box is 6× RTX 4060 Ti and the recorded runs are 15–35 h each, so this is weeks of wall clock,
not a batch job. Order by what unblocks a *claim*, not by net.

1. **R50 `2018`, 90 ep, ~22.5 h.** Already validated end to end today; it is the controlled
   comparison the blueprint's A3-vs-2018 section needs and the only row there with no number.
2. **R50 A3 verified `lambaccdp8x64wxclipbce`, ~31–34 h.** The headline artifact, and the one whose
   deltas closed. Replaces 77.43% against a recipe that is now materially closer to the paper.
3. **R50 A3 JAX reference, ~17 h.** Needed as (2)'s peer — the verified-vs-reference comparison is
   meaningless if one side moved and the other did not.
4. **MNv4-Conv-M, EfficientNet-B0, MNv2, R34** — C3/C4 only, so their *training* is unchanged and in
   principle only eval moved. ⚠ But C4 changes the denominator, so a re-score is not optional; if a
   checkpoint survives, **re-scoring is far cheaper than re-training** — see
   `score-checkpoint` (`f62a3a9`). ▶ Check for surviving checkpoints before booking GPU time.

   ⛔ **CORRECTED 2026-08-15 — this step as written does not produce the number it promises.** Every
   surviving checkpoint predates C6 (R50-A3 Jul 10, B0 Jul 18, MNv2 Jul 30, MNv4 Jul 31; C6 landed
   Aug 4), so scoring one through today's emitter measures C3+C4+C6 together and reports it as
   C3/C4. Worse, a *clean* C3/C4-only re-score is not reachable by regenerating at any commit:
   C6 (Aug 4) precedes C3 (Aug 14), so no emitter revision exists that has the new eval protocol
   and the old network. **The numbers in §6 are what a re-score can actually deliver; a
   C3/C4-only figure needs a re-train.** This is the same conclusion `f8cd3a9` reached for the
   blueprint ("the change moves the training distribution, so it needs a re-run rather than a
   re-score") — it applies to the eval half too, for a different reason.
5. **ViT / ConvNeXt families** — C1-blocked, so nothing exists to compare against; these are first
   runs in all but name and should go last unless a specific claim needs them.

⚠ **Do not start (2) before (3)'s reference is scheduled.** The whole value of the verified number
is that it sits beside a reference measured under the same rules.

---

## 6. ⭐ THE RE-SCORE PASS, RUN 2026-08-15

Three surviving checkpoints scored over all 50,000 with `jax/scripts/eval_full50k.py` (new; the
one writer, replacing six drifted `eval_<net>_full50k.py` copies — three still carried ResNet-34's
docstring, `eval_enet_full50k.py` had never learned the `bn`/`training` args, and none had learned
`take`). It reads the state-tuple layout out of the generating module's own `save_train_state`
call and checks the derived leaf count against the `.npz`, so a checkpoint/module mismatch refuses
instead of mis-assigning arrays.

| net | ckpt | in-training (49,920) | re-scored (50,000) | Δ |
|---|---|---|---|---|
| **MNv2** 350 ep | Jul 30 | 71.46 / 90.33 | **71.26 / 90.12** | −0.20 |
| **EfficientNet-B0** 350 ep | Jul 18 | 76.80 / 93.26 | **76.75 / 93.17** | −0.05 |
| **R50 RSB-A3 JAX ref** | Jul 10 | 77.22 / 93.34 | **74.62 / 91.75** | **−2.60** |

⚠ **Each Δ is C3+C4+C6 together, not C3/C4** — see §5.4's correction. The R50 outlier is the
signature: `f8cd3a9` measured C6 alone at −1.95 pt on this exact checkpoint and the sound resize
figure at −0.90, and −1.95 − 0.90 ≈ −2.85 brackets the −2.60 observed here. MNv2 and B0 barely
move, so C6 costs *them* little — but "little" is measured, not assumed, and only for these two.

### 6a. ⭐ The equality gate, which is why the above is trustworthy

`generated_mobilenet_v2_imagenet_full.py` (2026-07-28) survived in `.lake/build/` — a **pre-C6,
pre-C3 module matching MNv2's checkpoint**. Scored through it, the checkpoint gives
**71.46 / 90.33**, against the run's own `[Epoch 350] val_top1=0.7146 val_top5=0.9033`. Exact on
both metrics. That is the gate `f8cd3a9` said the JAX path lacks, and it is what separates "the
re-score pipeline is correct and C6 is real" from "something in the loader is broken".

▶ `eval_full50k.py` tolerates a 4-argument `eval_batch` specifically so this gate can be run
against stale modules; it refuses a batch width that would leave an unmaskable tail.

### 6b. What could not be re-scored

* **R34** — ✅ **RESOLVED 2026-08-22 by a re-train**, which is what this bullet said it needed.
  The JAX `.bin` (May 31) has **no companion `.state.npz`**, so there were no BN running stats to
  eval with, and it predated C6 besides. ⚠ Note §2.2 credits R34's number to
  `runs/r34in_30ep_4gpu_sup*.log`, but that is the **verified** path (70.71% over 49,920), not a
  JAX run — and `score-checkpoint` refuses BN nets by construction.
  ▶ Re-run on the **same 90-epoch 2018 recipe** under the current emitter:
  **74.16 / 91.92**, 14 h 49 m, artifacts at `/home/skoonce/r34_2018_90ep/` (this one HAS its
  `.state.npz`). That is **+2.14** over the superseded 72.02 %, on an unchanged recipe — the whole
  delta is C3/C4/C6. `jax/runs/r34_imagenet_bf16_90ep/RESULTS.md` is marked superseded.

* **MNv4-Conv-M** — checkpoint and state both survive at `/home/skoonce/mnv4_convm_100ep/`, and
  `eval_full50k.py` should drive it as-is, but it was descoped from this pass. It is C2-affected
  anyway, so a re-score was never going to be its final number.
