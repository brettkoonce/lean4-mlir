# resize_eval_reconciliation.md — closing blueprint §5.8's `[TODO, resize/eval reconciliation]`

**Opened 2026-08-14**, scoping the one open `[TODO]` in `blueprint/src/content.tex` §5.8
("Side quest: ResNet-50, RSB-A3"). Companion to `planning/recipe_fidelity_diffs.md` C1, which
owns the measurement, and to `planning/a3_paper_fidelity.md` §4b, which owns the timm diffs.

## 0. THE TODO, VERBATIM

> **[TODO, resize/eval reconciliation.]** Put `true-2048` and the timm reference on one ruler,
> either a single fixed resampler or a bit-exact `tf.image.resize` match in the reference eval,
> before reading anything into the 77.00% vs 78.1% gap.

The prose it guards (`content.tex` ~5370):

> Both rows here are scored through this repo's `tf.image.resize` preprocessing, whereas the timm
> reference reports 78.1% through PIL's bicubic resize… Port these weights into a standard
> PIL-resize eval and they fall to **~74.4%**. Port the reference `a3_in1k` weights *into this
> pipeline* and *they* fall to **~74%**. The ranking flips with the resize library.

---

## 1. ⚠⚠ THE REPO CONTAINS TWO ANSWERS TO THIS AND THEY DIFFER BY ~10×

| source | claim | instrument |
|---|---|---|
| `content.tex` §5.8 | the resampler is worth **~2.6–4 pt** on R50 | ⛔ none found |
| `recipe_fidelity_diffs.md` C1 | the eval preprocessing is worth **~0.2 pt** | ✅ `jax/scripts/eval_preproc_ab.py`, full 50k, two nets |
| **§2e below, 2026-08-14** | **0.90 pt on R50** (77.15 → 76.25) | ✅ the same script, fixed, 2,000 paired images |

▶ **Measured, the answer is nearer C1's than §5.8's** — and R50 genuinely is ~3.6× more
resize-sensitive than the two nets C1 covered, which is the half of §5.8's instinct that was right.

C1's measured table (ViT-Ti / ConvNeXt-T, all 50,000):

| | ViT-Ti | ConvNeXt-T |
|---|---|---|
| ours (aliased both sides) | 65.642 | 78.144 |
| eval antialiased only | 65.398 | 77.918 |
| full timm protocol | 65.436 | 77.940 |

Total spread: **0.25 pt**. ⚠ `grep -rn "74.4"` across the repo returns the §5.8 prose and nothing
else — no script, no log, no run directory. §1 below explains why: **the script that would have
produced it could not run on R50 at all.**

---

## 2. WHAT WAS ACTUALLY WRONG — four findings, all verified 2026-08-14

### 2a. ⛔ `eval_preproc_ab.py` could not run on ANY BatchNorm net — FIXED

The emitter gives a LayerNorm net `forward(params, x, drop_key=None)` and a BatchNorm net
`forward(params, x, bn, training, drop_key=None)`. The script called `m.forward(params, x)`.

So it worked on **ConvNeXt-T and ViT-Ti** — which is exactly and only what C1 measured — and
raised `TypeError` on R50, MobileNetV2, EfficientNet and MNv4.

▶ **§5.8's ~74.4% therefore did not come from this repo's instrument.** Nothing here could
produce an R50 preprocessing number until today.

⭐ The running statistics turned out to be recoverable, and the asymmetry is worth carrying: the
JAX trainer's `.state.npz` is `jax.tree.leaves((params, opt_state, bn_state))`, so **the BN
buffers ARE checkpointed on the JAX side** — where the verified path's `.bin` is exactly
`[θ|m|v]` and drops them (`next_session_verified_trainer_code.md` §2b). Same gap, opposite
outcome: this side can re-score a finished run today, that side cannot until the format changes.

### 2b. ⛔ …and it scored R50 at the wrong CROP — FIXED

`CROP_PCT = S / (S + m._CROP_PADDING)` = 0.875. But `Jax/Codegen.lean` emits **two** branches for
the eval crop, and a config that sets `testCropRatio` takes the other one and never reads
`_CROP_PADDING`. RSB-A3 sets **0.95**.

Census over every generated ImageNet trainer — exactly two take the explicit branch, and both are
the RSB-A3 R50:

| crop | via | trainers |
|---|---|---|
| 0.875 | `_CROP_PADDING` | all 23 others (ConvNeXt ×6, ViT ×6, R34, R50 default/a1/a2accum, ENet, MNv2 ×2, MNv4, …) |
| 0.875 | `testCropRatio` (explicit) | `resnet50_imagenet_2018` |
| **0.95** | `testCropRatio` (explicit) | **`resnet50_imagenet_rsbfaithful`, `resnet50_imagenet_short`** |

So the formula was right for the nets C1 measured and wrong for the net §5.8 is about. An 8%
narrower field of view is a different measurement, not an error bar, and on a FixRes recipe
(train@160 / eval@224) object scale at test time is precisely what the recipe is exploiting.

▶ Now recovered by **reading the emitted function** (`_crop_pct_of`), with a refusal if it matches
neither branch — not recomputed from a constant that one branch ignores.

### 2c. ⭐ The `timm` arm IS PIL — measured, so C1's ruler is valid

That arm calls TF (`tf.image.resize(BICUBIC, antialias=True)`) while claiming to be timm's
protocol, so the whole claim rests on TF's antialiased bicubic agreeing with PIL's. Diffed across
the downscale range ImageNet val actually spans, onto a 256 shorter side:

| shorter side | ratio | TF aa=True vs PIL | TF aa=False vs PIL | PIL bilinear vs PIL bicubic |
|---|---|---|---|---|
| 180 | 0.70× | 0.31 | 0.31 | 0.38 |
| 375 | 1.46× | 0.30 | 0.39 | 0.72 |
| 750 | 2.93× | 0.29 | 1.05 | 1.87 |
| 1000 | 3.91× | 0.30 | 2.54 | 2.38 |
| 2000 | 7.81× | 0.31 | 11.20 | 4.91 |
| 3000 | 11.72× | 0.29 | 13.73 | 7.25 |

(mean |Δ| of 255, smooth-image-with-a-hard-edge method from `geo_aug_pil_diff.py` — *never
validate a resampler on noise*.)

⭐ **`antialias=True` tracks PIL FLAT at ~0.30 across the whole range** — the same resampler to
within a quantisation step. So C1's instrument is a genuine PIL ruler and its ~0.2 pt is the
answer for the nets it measured.

⚠ **The control is the interesting half.** `antialias=False` — what we ship — tracks PIL at 0.31
with no downscale and then diverges *with the ratio*: 1.05 at 2.9×, 2.54 at 3.9×, 13.73 at 11.7×.
▶ So the aliasing penalty is concentrated on the LARGE images in the val set, and any single
scalar for it is an average over a heavily skewed distribution rather than a per-image constant.
That is the one respect in which §5.8's instinct was right: the effect is real and it is not
uniform. It is the *magnitude* that has no support.

### 2d. ⚠ Found on the way: an A1/A2 RECIPE gap, not just a script one — FIXED

`timm.get_pretrained_cfg('resnet50.a{1,2,3}_in1k').crop_pct` is **0.95** for all three RSB tiers
(read from timm 1.0.28, not assumed; only `tv_in1k`, the 2018 torchvision weights, is 0.875 —
which is why our 2018 recipe correctly sets that value).

Our A3 config sets 0.95. **`default` (A2 @ bs512), `a2-true-2048`, `a2-accum` and — through it —
`a1` did not**: they inherit `resnet50ImagenetConfig`, which left it unset, i.e. 0.875.

Fixed at the base config, so it is one line and it is inert for A3 and 2018 (both explicit).
⭐ None of those recipes has ever been run, so no number is invalidated — and
`next_session_verified_trainer_code.md` §3b is about to render and run A2.

---

## 2e. ⭐ THE FIRST R50 PREPROCESSING NUMBERS THIS REPO HAS EVER PRODUCED

With §2a and §2b fixed, run on `/home/skoonce/resnet/r50_a3_rerun/ckpt_e100.bin` (the **77.22%**
run, the §5.8 row) over the **first 2,000** val images — the same 2,000 in all three arms, since
the tfds validation iterator neither shuffles nor repeats, so this is a PAIRED comparison:

| arm | top-1 | top-5 | Δ top-1 |
|---|---|---|---|
| `tf-current` — ours, aliased both sides | **77.150** | 93.000 | — |
| `tf-aa` — eval antialiased only | 76.700 | 92.800 | −0.45 |
| `timm` — the full PIL protocol | **76.250** | 92.750 | **−0.90** |

⚠ A 2,000-image prefix, not the 50,000 a quotable number needs — but it settles the question the
TODO is blocked on, because the claim under test is 2.6 pt and this is 0.90.

**Two things it establishes and one it corrects.**

1. ▶ **§5.8's ~74.4% is not reproducible.** The full timm protocol costs **0.90 pt** here
   (77.15 → 76.25), where §5.8 asserts ~2.6. The 77.22 run scored through PIL is ~76.3, not ~74.4.
2. ⭐ **R50 really is more resize-sensitive than the nets C1 measured** — 0.90 pt against ViT-Ti's
   and ConvNeXt-T's 0.25. So §5.8's instinct that this net is different was right; only the
   magnitude was wrong. The plausible mechanism is §2b's own subject: crop_pct **0.95** keeps 8%
   more of the frame than 0.875, so more of the image survives to be resampled, on a FixRes recipe
   whose whole trick is test-time object scale.
3. The **direction** reproduces C1 exactly: fixing eval *alone* makes the number **worse**, because
   the model was trained under the aliased resampler. `tf-aa` sits between the two, as it did on
   both other nets.

⚠ **This is one side of the ruler.** It puts OUR weights on timm's preprocessing; it does not put
timm's weights on ours, which is Tier 2 and still unbuilt. "The ranking flips with the resize
library" remains untested.

---

## 3. WHAT IT WOULD TAKE TO CLOSE THE TODO

### Tier 1 — ✅ MOSTLY DONE (§2e). What remains is one full-50k pass, ≈1 GPU-hour

Everything needed exists as of today.

* **Weights: present**, though `grep` in the repo misses them (the MNv4 lesson again). The JAX-side
  A3 checkpoints live OUTSIDE the repo:
  * `/home/skoonce/resnet/r50_a3_rerun/ckpt_e100.bin` + `.state.npz` — the 2026-07-10 rerun, the
    **77.22%** row in §5.8;
  * `/home/skoonce/resnet/r50_rsb_a3_jul4_archive/r50_rsb_a3_rsbfaithful_e100.bin` — the 76.66%
    first run.
  Both are 102,228,128 B = 25,557,032 params × 4, R50's exact count.
* **Instrument: fixed** (§2a, §2b). Smoked end-to-end on CPU against `ckpt_e100.bin`: crop_pct
  resolves to 0.950000 off the trainer, 106 BN arrays load from the `.state.npz`, and the first 256
  val images score **77.34 / 92.58** — tracking the run's own 77.22, i.e. the path is right.

```bash
cd jax
for M in tf-current tf-aa timm; do
  TFDS_DATA_DIR=/home/skoonce/tensorflow_datasets \
  GEN=.lake/build/generated_resnet50_imagenet_rsbfaithful.py \
  CKPT=/home/skoonce/resnet/r50_a3_rerun/ckpt_e100.bin \
  MODE=$M BATCH=250 ../.venv/bin/python scripts/eval_preproc_ab.py
done
```

▶ **Run at `LIMIT=0` (all 50,000), on the GPU, it is ~20 min per arm.** §2e already ran it at
`LIMIT=2000` on CPU (~4 min per arm) and got the answer; the full pass is what makes it quotable.
⚠ The CPU run needs `JAX_PLATFORMS=cpu`; on GPU drop that and raise `BATCH` to 250.

⚠ **`true-2048`'s own weights are NOT on this box.** §5.8's table quotes it at 77.00% from a
rented A100 and only the log came back. So the TODO as literally worded — "put `true-2048` and the
timm reference on one ruler" — cannot be executed on the checkpoint it names; the `rsb-faithful`
4×4060 Ti run (77.22%) is the available stand-in, and it differs from `true-2048` only in the BN
group, which §5.8 itself measures at ~0.2 pt and of unknown sign.

### Tier 2 — the reverse direction (the harder half, ~1 day)

"Port the reference `a3_in1k` weights into this pipeline and *they* fall to ~74%" needs a
**torch → our-layout weight port**, and there is none in the repo (`grep -rl "state_dict"` over
`jax/` and `scripts/` returns nothing). What it involves:

* `timm.create_model('resnet50.a3_in1k', pretrained=True)` — downloads, and timm is installed.
* Map ~161 torch tensors onto `init_params_from_file`'s flat f32 order. Mechanical but exacting:
  torchvision's R50 and ours agree on carrying **no conv biases** (`rsb_a3_r50_verified.md` §2m),
  and both land on 25,557,032, so the port is a permutation with no shape surgery.
* BN running mean/var must come across too, into the `bn_state` tree §2a now reads.
* ⭐ **Gate it before believing it**: the ported weights scored through *timm's own* eval must
  reproduce timm's published 78.06 to within a rounding step. Without that control, "they fall to
  ~74%" is indistinguishable from a bad port — which is the most likely explanation for the number
  §5.8 already quotes.

### Tier 3 — what "one ruler" should actually mean

Three candidates, and the choice is a judgement rather than a measurement:

| ruler | cost | what it says |
|---|---|---|
| (a) score both through **timm's `validate.py`** | needs Tier 2 *reversed* (our weights → torch) | the number the reference world would quote |
| (b) score both through **our pipeline** | needs Tier 2 | the number this repo would quote |
| (c) make our eval **antialiased** and re-run | a full retrain per net | removes the mismatch instead of measuring it |

▶ Recommendation: **(a) and (b) together, reported as a 2×2**, which is what "the ranking flips
with the resize library" is actually asserting and what neither number alone can show. ⚠ (c) is
the expensive one and C1 already bounds its value at ~0.2 pt — but note §2c: that bound is an
average over image sizes, and it is loosest exactly where the val set has its long tail.

⚠ **Do not run (c) blind.** `recipe_fidelity_diffs.md` C1 already records that fixing eval *alone*
makes the number **worse**, because the model is trained under the aliased resampler. A genuine
fix is `antialias=True` on both sides plus a retrain, and the matched-antialiased arm has never
been run, so its sign is unknown.

### Tier 0 — the free half, do it regardless

Whatever Tiers 1–3 conclude, §5.8's prose currently states an unmeasured number as a measured one.
The minimum honest edit is to mark ~74.4% and ~74% as **estimates without an instrument**, or
delete them, until Tier 1 reports. That costs nothing and is true today.

---

## 4. WHAT CHANGED TODAY

| file | change |
|---|---|
| `jax/Jax/Codegen.lean` | ⛔ **the `jax/` package had not built since `d96c7fa`** — an unescaped `"` inside the `autoAugmentPy` Lean string literal (line 37, `"timm's default geometric interpolation"`), which terminated the string and left the rest of the Python parsed as Lean. Escaped. Same shape as §5's shim-wiring gate: 8 commits unpushed, so CI never saw it, and the handoff's green list covers `lake build LeanMlir`/`Proofs` but not the `jax/` package |
| `jax/MainResnet50Imagenet.lean` | `testCropRatio := 0.95` on the base config — the A1/A2 recipe fix (§2d) |
| `jax/scripts/eval_preproc_ab.py` | crop ratio read off the emitted function (§2b); BN nets supported via the `.state.npz` (§2a); the `timm` arm's PIL equivalence recorded as measured (§2c) |

Gates re-run green after all of it: `lake build LeanMlir Proofs`, `regen_verified_mlir.sh check`
(185 artifacts), `check_render_coverage` (172, 154 guarded), `shim_wiring_gate`,
`randaug_timm_diff`, `geo_aug_pil_diff`, `docstring-checkrefs`, `TestVariantPredicates`.

---

## 5. THE GENERAL SHAPE, because it is the third instance

`a3_paper_fidelity.md` §4b already records it: *"our X is bilinear, theirs is bicubic" sounds like
a fidelity gap and is really a bound on one — before paying for a change, measure what the change
could possibly buy.* This is that lesson again with a new twist:

⭐⭐ **An instrument that cannot run on a net will not tell you so in the number — it will tell you
in a stack trace you never saw, and the number will come from somewhere else.** C1's table is
correct and its scope was ViT-Ti and ConvNeXt-T; the moment its conclusion was carried to R50, it
was carried past the two nets the tool could actually load. Neither the doc nor the script said
which nets it covered, because nobody had to choose — the coverage was decided by an arity
mismatch.

▶ The check that would have caught it: **run the instrument on every net it claims to cover, once,
with `LIMIT=8`.** That is under a minute per net and it partitions "measured" from "assumed".
