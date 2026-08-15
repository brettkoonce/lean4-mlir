# imagenet_rerun_sweep.md — every ImageNet run that needs redoing, and why

**Opened 2026-08-15.** Between 2026-08-14 and 2026-08-15 five changes landed that move either the
training distribution, the eval denominator, or the optimizer. **No ImageNet number in this repo is
comparable across that boundary**, and for seven of the configs no run was *possible* at all. This
file is the inventory: what changed, what it touched, and what has to be re-run before anything is
quoted again.

▶ Read §1 before deciding the order. Read §3 before starting the sweep — two of the prerequisites
are cheap and one of them gates everything.

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

⚠ **C1 and C5 are not fidelity deltas, they are hard blockers.** A run either started or it did not.
C2/C3/C4 are the ones that make old numbers incomparable rather than impossible.

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
5. **ViT / ConvNeXt families** — C1-blocked, so nothing exists to compare against; these are first
   runs in all but name and should go last unless a specific claim needs them.

⚠ **Do not start (2) before (3)'s reference is scheduled.** The whole value of the verified number
is that it sits beside a reference measured under the same rules.
