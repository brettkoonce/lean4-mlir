# imagenet_parity.md — paper ↔ JAX ↔ verified for the five remaining ImageNet chapters

Started 2026-09-25 from a six-way read-only audit (MNv2, MNv4-Conv-M, EfficientNet-B0, ConvNeXt-T,
ViT-Ti, plus a cross-cutting pass). ResNet (ch 5) is the template: R50-2018 is running on the other
box with the current patterns, and after it lands that chapter is done. This doc brings the other
five up to that standard in three stages, in order:

1. **Code** (§3–§5): the emitter, renderers, drivers, confs and gates say what each recipe
   should be, and the JAX and verified paths agree on it.
2. **Book** (§6): brought into sync with the code, and with the runs that already exist, where
   those runs still stand.
3. **Runs** (§7): scheduled once the code for that net is closed, so that each long run absorbs
   every recipe change at once.

Every finding was read at file:line. Line numbers are as of `e38e56ab`. "Unverified" marks claims
the audit could not confirm.

## 0. Rules for this thread

* The standard is a pair of runs that differ in one variable: JAX reference vs verified. A
  second, disclosed axis is paper vs JAX. The book may call a pair one-variable only when §2's
  table for that net has a single row left.
* A recipe change touches both paths in the same commit: the `Codegen.lean` emit, the renderer,
  and the driver. A change on one side only is a new row in §2.
* A render change is checked by strip-and-compile plus byte-identity of every artifact it is
  not meant to move, as in `proof_cleanup.md` §0. A gate fix lands with its control (shown red on
  a broken input).
* Nothing here launches a GPU job without the user's word ([[lake_run_job_launches]]).
  Smoke runs and probes are listed in §7 with the long runs.
* No commits or pushes without approval; stage, then stop.

## 1. Headline findings (the book is wrong today)

| # | net | finding | evidence |
|---|---|---|---|
| H1 | ViT-Ti | The pair is **not one-variable**. The JAX 72.31/91.12 run finished 08-29 at an **effective label smoothing of 0**: the fix, 4b765be7, landed 08-30, and its own message says "ViT-Ti's 72.31/91.12 are now stale". The verified 72.35/91.22 trained at 0.1, because the render smooths the mixed target. | `content.tex` ~12106 (loss "matches"), ~12215, ~12270 ("one-variable"); render constants 0.9 / 1e-4 |
| H2 | MNv2 | The pair differs in **four** ways, not one: the BN group (64 vs 256), classifier dropout (J 0.2, V none: `rmsdp64bf16` has no `do`), stem/head activation (J ReLU, since the run predates the ReLU6 fix; V ReLU6), and **label smoothing (J 0.0, V 0.1)**. The render hard-codes α = 0.1. | `MobileNetV2RenderB.lean:683`; the `%lomac` 0.9 constant at `mobilenetv2in_rmsdp64bf16_train_step.mlir:13575`; `jax/MainMobilenetV2Imagenet.lean:82`; book ~7690 ("held equal"), 7705–7740 |
| H3 | MNv2 | Contradiction to settle: `mnv2-default-4gpu.conf:155-157` says "this net's published 350ep run got labelSmoothing 0.1" from a stale shim, while the book's ledger (7619) says "Label smoothing 0". Which run, and which path, is unverified. | conf, book |
| H4 | MNv2, B0 | The finished verified runs trained **per-replica BN (group 64)**. The committed renders have been sync-BN since cad811ac (09-21): MNv2 has 314 all-reduces, B0 361, against 214 before. The next launch of either conf is therefore the sync-BN run, not a repeat. The book discloses this with `\globalbntodo`. | renders, `runs/2026-09-1{0,2}-*/RESULTS.md` |
| H5 | ConvNeXt-T | The verified run had no EMA, while the reference scores the EMA shadow. Its EMA peer (`emadpwxclipdropbf16`, e38e56ab) has **never run on a GPU**. Book 10128–10137 ("no committed artifact combines EMA…") went false with e38e56ab. | `runs/2026-09-18-cnx-verified-300ep/RESULTS.md` §1 |
| H6 | MNv4 | The book says the paper uses EMA. **Conv-M does not** (arXiv 2404.10518, Table 10). The "half" pair (JAX `half` × `emaaccdp8x128wxdowd005bf16`) matches to within noise. | book 7893–8083 |

## 2. The deltas

### 2.1 Paper → JAX reference

| net | JAX / paper top-1 | deltas that matter | smaller deltas |
|---|---|---|---|
| MNv2 | 71.90 / 72.0 | lr 0.045 at 256 sync (paper: 96×16 async); coupled wd 4e-5 on **all** tensors, BN and depthwise included (slim skips both); warmup 5 + continuous ×0.98 (paper: staircase, no warmup) | BN eps/decay 1e-5/0.99 vs TF 1e-3/0.997; He-uniform vs truncated normal σ .09; no colour distortion; the run predates the ReLU6 stem/head |
| MNv4-Conv-M | 75.48 (pre-timm-parity net) / 79.9 | 100/50 ep vs 500; 224 train vs 256; eval crop 0.875 vs timm r224's 0.95; wd 0.05 vs 0.1; dropout 0.1 vs 0.2; no drop-path (paper 0.075); RandAugment N2 m9 p0.5 vs m15 p0.7 | AdamW ε 1e-8 vs 1e-7; BN decay 0.99 (timm 0.9); drop-path ramp /21 vs timm's /22; **both paths run EMA, the paper doesn't** |
| B0 | 77.15 / 77.1 (timm 77.7) | **wd on every parameter**, BN γ/β and biases included: the RMSProp branch ignores `wdExcludeNormBias` (`Codegen.lean:2643-2660`); decay continuous and offset by warmup vs staircase from step 0 (LR runs ~6.5% high) | BN eps 1e-5 vs 1e-3 (never measured); drop-connect ramp i/15 vs i/16; SAME stem + symmetric strided depthwise (an undocumented hybrid); dense Xavier init; TF puts lr inside RMSProp's momentum |
| ConvNeXt-T | 81.53 EMA / 82.1 | batch 256 @ 2.5e-4 vs 4096 @ 4e-3; grad clip 1.0 (paper: none); tanh GELU (paper: erf) | erasing zero-fill + uniform aspect vs `pixel` + log-uniform; TF inception crop; mixup/cutmix alternate by step instead of a p=0.5 switch |
| ViT-Ti | 72.31 / 72.2 | **ls effectively 0** (H1); clip 1.0 (DeiT: none; the clip-off arm on `vitInit` never ran); scores the EMA shadow where DeiT scores the live model | LN eps 1e-5 vs 1e-6; tanh GELU; no cooldown or min_lr; repeated-aug copies spread by a shuffle window, not placed in one step; erasing zero-fill; batch 512 (scaled) |

Fleet-wide, identical on both paths: RandAugment/AutoAugment geometry uses **bilinear**
interpolation where timm uses bicubic (`Codegen.lean:50-65`); random erasing fills **zero**
(`:430-443`).

### 2.2 JAX reference → verified

| net | verified / JAX | shipping conf → variant | rows left |
|---|---|---|---|
| MNv2 | 71.91 / 71.90 | `mnv2-default-4gpu` → `rmsdp64bf16` | ls (H2), dropout (H2), stem/head act (JAX-side stale), BN group (render already fixed) |
| MNv4 | run in progress (67.08 EMA @ e15) | `mnv4-half-4gpu` → `emaaccdp8x128wxdowd005bf16` | none that matter; f32 eval forward, `bnFirst`, cosine off by one step, RNG streams |
| B0 | 76.88 / 77.15 | `enet-default-4gpu` → `emarmsdp64dropdobf16` | BN group (render already fixed); `bnFirst` running-stat seeding (eval only); host vs PRNG masks |
| ConvNeXt-T | 81.30 raw / 81.51 raw (p = 0.10) | `cnx-default-4gpu` → `adamdpwxclipdropbf16` | EMA; the peer exists, unrun (H5) |
| ViT-Ti | 72.35 / 72.31 | `vit-default-emabf16-4gpu` → `emadp128x4wxclipdropbf16` | ls (H1); mixup λ drawn per 128-row shard from numpy (agreement is distributional only) |

### 2.3 Feature matrix against the R50 template (V = verified, J = JAX)

| feature | R50-A3 | MNv2 | MNv4 half | B0 | CNX-T | ViT-Ti |
|---|---|---|---|---|---|---|
| sync-BN (Chan) | V | V render, not the run | V, group 512 | V render, not the run | n/a | n/a |
| bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (`vit-default` f32) |
| EMA | none (recipe) | none | ✓ both | ✓ both | J ✓ / V unrun | ✓ both |
| `wx` (no decay on norm/bias) | ✓ | ✗ both | ✓ | ✗ both | ✓ | ✓ |
| clip | ✓ | – | – | – | ✓ | ✓ |
| accumulation | k4 | – | k8 | – | – | – |
| classifier dropout | – | **J only** | ✓ | ✓ | – | – |
| label smoothing | 0 (BCE) | **J 0 / V 0.1** | 0.1 | 0.1 | 0.1 | 0.1 (**J run 0**) |
| sharded-batch gate | ✓ | ✓ | ✗ (only the `adamdp64` variant is covered) | ✓ | ✓ | ✗ (`vit-dp-check` duplicates rows) |
| JAX-path job conf | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| parity gate vs timm/torchvision | – | ✗ | ✓ (not in CI) | ✗ | ✗ | ✗ |

## 3. Code: shared (fix once, every net benefits)

| # | where | problem | fix | size |
|---|---|---|---|---|
| C1 | every `*RenderB.lean` loss block (e.g. `MobileNetV2RenderB.lean:683`) | α = 0.1 hard-coded in the cotangent and the report-only loss | α becomes a renderer parameter, defaulting to 0.1 so every existing artifact renders byte-identical; the MNv2 variant passes 0 | S–M |
| C2 | `Codegen.lean:2643-2660` (RMSProp branch); MNv2/B0 renderers | `wdExcludeNormBias` is ignored for RMSProp; no `wx` in `MobileNetV2RenderB`/`EfficientNetRender` | honour the mask in the RMSProp emit; add `wx` to both renderers; set it in the MNv2/B0 recipes (a recipe change, so both paths rerun: §7) | M |
| C3 | `supervise.sh:131`, `VerifiedTrain` | recipe → shim selection works for R50 only, so MNv4's `full` shim (RandAugment m15) cannot be reached on the verified path | `LEAN_MLIR_RECIPE` → shim for every net, in `VerifiedTrain` | S–M |
| C4 | `VerifiedTrain.lean:200` | BN running-stat decay is 0.99 for every net except R50; timm-ported MNv4 wants 0.9 | per-net setting on both paths; decide per net in §5 | S |
| C5 | `VerifiedTrain.lean` ~2778 (`scoreCheckpoint`) | refuses every BN net, although the `.bn` companion has been written since 09-12; the comment "~30 GB drain" is stale | read `.bn`; score BN nets | S–M |
| C6 | `Codegen.lean:50-65, 430-443` + the shim (✅ 2026-09-25, see §3 status) | bilinear geometry ops; zero-fill erasing with uniform aspect; the erase box is sized from `_IMG_SIZE`, so any `trainRes` recipe with erasing breaks | bicubic geometry, `pixel` erasing with log-uniform aspect, a size taken from the actual crop. Lands **with the next reruns**, since it moves every reference | M |
| C7 | every `scripts/jobs/*.conf` | prechecks diverge: only `mnv4-half` builds its exe; mnv2, mnv4-default and vit-default have no freshness check; vit-emabf16 tests `-x` only; the ConvNeXt AutoAugment grep matches the `def _autoaugment` line (always passes, and AutoAugment isn't in that recipe) | `scripts/lib/precheck.sh`, sourced by every conf: build the exe, `regen_jax_generated.sh box`, GPU idle, render exists with the expected replica/all-reduce count, `CKPT_EPOCH_FILE` matches the variant, shim present and fresh, aug call-site grep on the call not the def | S |
| C8 | `scripts/gates/gen_mlir_manifest.py:56-171` | `wd<n>` not decoded; `x` means k×B after `acc` but B×replicas in `128x4`; batch suppressed whenever `acc` is present | decode `wd`; one `x` grammar (rename `dp128x4` if needed, predicates in `TestVariantPredicates` pin it) | S |
| C9 | JAX resume (`shuffle(seed=42)`, unseeded aug) | the MNv4 JAX conf says "resumes bit for bit"; it doesn't | seed the shuffle by epoch and the aug by step, or fix the claim | S |

### §3 status (2026-09-25, non-render half)

* ✅ C7: `scripts/lib/precheck.sh` — `pc_box`, `pc_exe` (`lake build <exe>` at launch; under
  `DRY_RUN` a ⚠ from mtimes, no build), `pc_pjrt_so` (compiles `pjrt_ffi.c` to a scratch file and
  compares bytes: the mtime test read this box's current .so as stale after a checkout),
  `pc_jax_box`, `pc_shim`, `pc_aug_call`/`pc_aug_no_call` (call lines only, `def` and comments
  dropped), `pc_render` (replica marker + exact all-reduce count), `pc_ckpt`, `pc_gpu_idle`, `pc_env`.
  Wired into mnv2, enet, cnx, cnx-emabf16 (now sources cnx-default and swaps `VARIANT`), vit,
  vit-emabf16, cnxs/cnxb, vits/vitb. Counts pinned: MNv2 314, B0 **360** (not 361), ConvNeXt-T 182,
  ConvNeXt-S/B 344, ViT-Ti/S/B 200, MNv4 464. The pre-sync-BN renders had MNv2 **158** and B0
  **213** (not 214). Controls: the old ConvNeXt `_autoaugment(` grep passes on the `def` line and
  `pc_aug_call` fails on it; `pc_render … 158` fails on the committed MNv2 render. MNv4 confs left
  to the MNv4 rescope.
* ✅ C8: `wd<d…>`/`ls<d…>` decoded (first digit is the integer part, checked against `%wd` /
  `%lomac`); one `x` grammar (`acc[dp]<k>x<B>` vs `dp<B>x<R>`), batch no longer suppressed under
  `acc`, and `wd00` no longer reads as "batch 00". Seven new `--selftest` rows. MANIFEST regenerated.
* ✅ C9 (claim side): the MNv4 JAX conf now says the train state resumes exactly but the data stream
  does not (`shuffle(8192, seed=42)` restarts, augmentation unseeded). The generated trainer's
  `save_train_state` docstring ("continues bit-for-bit") comes from `jax/Jax/Codegen.lean` and is
  unchanged.
* ✅ C1 (MNv2): α is a `mobilenetv2AdamTrainStepFaithfulB` parameter (default 0.1, byte-identical);
  at α = 0 the smoothing ops are not emitted. The other renderers keep 0.1, which every one of
  their recipes uses.
* ✅ C2: the JAX RMSProp and SGD-momentum branches honour `wdExcludeNormBias` through `WD_MASK`;
  `rmsOne`/`adamOne` take the decay operand name; `r34WdDecays`/`r34WdName`/`wdzConst` moved to
  `RenderKit` (same names). MNv2 turns it on (both paths); B0 does not (decision, §5.3).
* ✅ C4: already a per-net knob (`VerifiedConfig.bnMomentum`, `TrainConfig.bnMomentum`); only
  the per-net values are open.
* ✅ C9: the emitter's "bit-for-bit" comment fixed; the resume semantics live in
  `scripts/lib/jax_job.sh`, the one launch/resume block the JAX-path confs share.
* ✅ C5: `scoreCheckpoint` scores BN nets through `@<slug>_fwd_eval` with the `.bn` companion
  (EMA shadow ↔ `ema_bn`); equality gate on a trained MNv4 checkpoint: in-training eval and
  `score-checkpoint` both 33,552 / 43,783 of 50,000.
* Open: C3 (needed only for MNv4's 500-epoch tier).
* ✅ C6 (2026-09-25), opt-in per recipe — `TrainConfig.augBicubic` / `erasingPixel`, both default off
  so R50 (running on the other box) and MNv2 (no geometric aug, no erasing) are byte-identical. On
  for B0, MNv4-Conv-M, ConvNeXt-T/S/B, ViT-Ti/S/B (erasing only where the recipe erases: CNX, ViT);
  48 generated files move, trainers and shims alike.
  - Bicubic: TF has no bicubic projective warp (`ImageProjectiveTransformV3` takes NEAREST/BILINEAR
    and only LOGS on BICUBIC), so the emitter writes PIL's affine sampler out: a = −1 (PIL's
    transform kernel; its resize uses −0.5), border clamp, truncation to uint8, and shear/translate
    as 4-tap 1-D warps (bit-equal to 16 taps there), Rotate at 16 taps. Timm's PIL shear data is in
    pixel-centre coordinates, so the index-space shear gains `+f/2`.
    `scripts/gates/aug_bicubic_pil_check.py`: worst mean |Δ| 0.094 / 255, max 1, over the five ops at three
    magnitudes; control (the bilinear block) 8.0.
  - Erasing: timm `RandomErasing(mode='pixel')` — N(0,1) fill, aspect log-uniform [0.3, 1/0.3], first
    of 10 draws that fits, box sized from the image erased (fixes `trainRes`). Sampled: fires 0.253
    at p 0.25, erased area mean 0.175 (U(0.02, 1/3) → 0.177), fill mean ≈ 0, std ≈ 1, at 224² and 160².
  - ⚠ FEED COST, measured with 4 concurrent producers in `SHIM_HASH` mode on ares: B0 1,551 → 1,558
    img/s (decode-bound); ViT-Ti ~1,960 → ~1,450 (−26%, repeated-aug 3× makes aug the bottleneck) on
    the clean repeats, but the bicubic shims are BIMODAL under contention — some ViT/ConvNeXt/MNv4
    runs came out 2–3.5× slower. The user chose to ship it everywhere anyway (2026-09-25); the next
    ms/step probe of each C6 net is the real measure. Every conf ETA predates C6.

## 4. Code: gates

| # | net | gap | fix | size |
|---|---|---|---|---|
| G1 | ViT | `vit-dp-check` feeds every replica the same rows, so it can't see a shard-offset bug (book ~12205 admits it) | add `vitin` to `shard-check` / the genuinely sharded check | S, short GPU |
| G2 | MNv4 | `mnv4-dp-check` uses a duplicated batch; `imagenet-syncbn-check mnv4` is hard-wired to `adamdp64` (`tests/TestImagenetSyncBnCheck.lean:167-173`) | a `renderDp` like R50's (`:141-147`): 4×128 tied against 1×512 | S, short GPU |
| G3 | MNv2, B0, ViT (**go**, 2026-09-25) | no parity gate like `scripts/parity/mnv4_timm_parity.py`; `enet_forward_tie.py:32` ties the render to the **Imagenette** JAX file, which is how the ImageNet ReLU stem/head bug went unseen | timm/torchvision parity per net (MNv2 needs a pad-mode option for SAME vs symmetric; ViT pins LN eps and GELU); an ImageNet-spec forward tie for B0 | M, CPU |
| G4 | MNv4 | `mnv4_timm_parity.py` and `mnv4_forward_tie.py` not in CI | add to `jax.yml` | S |
| G5 | ConvNeXt | `opt_step_tie.py` has only ResNet-50 fixtures; the EMA variant has no numeric optimizer-step tie | AdamW + wx + clip (+EMA) row against `generated_convnext_tiny_imagenet_full.py`; run `TestConvNeXtDpCheck` on the EMA variant | M, short GPU |
| G6 | MNv2 | the precheck doesn't assert the sync-BN render, so a relaunch silently trains different BN semantics from the book's run | covered by C7's all-reduce count (314) | S |
| G7 | B0 | the sync-BN probe has never covered `emarmsdp64dropdobf16` (EMA + host-fed drop masks), per `global_bn_verified.md:67` | add it to `imagenet-syncbn-check` | S, short GPU |

### §4 status (2026-09-25)

* ✅ G6: `mnv2-default-4gpu` asserts 314 all-reduces (C7); `enet-default-4gpu` asserts 360.
* ✅ G3 for MNv2: `scripts/parity/mnv2_timm_parity.py` (+ `_mnv2_timm_dump.py`, run from `.venv/bin/python`;
  it drives `.venv-timm` itself) ties both JAX emitters to timm's `mobilenetv2_100` at
  `pad_type='same'` on shared weights — Imagenette train 4.4e-6, ImageNet train 4.7e-6 / eval 3.1e-6
  at ε 1e-3. `--controls` shows it red on symmetric padding (1.5e-1) and on ε 1e-5 (1.7e-2).
  `scripts/parity/mnv2_forward_tie.py --imagenet` carries it to the artifact the run scores through:
  `@mobilenetv2in_fwd_eval_eps0001` against the ImageNet reference in its own eval mode, max |Δ|
  3.0e-5 (2.1e-6 of scale); control `--mlir verified_mlir/mobilenetv2in_fwd_eval.mlir` (ε 1e-5)
  fails at 1.4e-1. Neither script is in CI yet (G4's question for MNv4 too: CI has no `.venv-timm`).
  ⚠ Pre-existing, not touched: the Imagenette `--eval` tie reports FAIL at max |Δ| 1.3e-4 against its
  absolute 1e-4 tolerance on logits spanning ±49 (2.6e-6 of scale), identically before this change.
* ✅ G3 for B0: `scripts/parity/enet_timm_parity.py` (+ `_enet_timm_dump.py`) against timm's `efficientnet_b0`
  with a SAME stem — the paths' hybrid (SAME stem, symmetric strided depthwise): Imagenette 2.3e-6,
  ImageNet train 2.0e-6 / eval 7.3e-7; controls red (symmetric everywhere 6.0e-3, ε 1e-5 6.0e-3).
  ⚠ `--pad tf` against TF's all-SAME `tf_efficientnet_b0`: train 3.7e-1 / eval 4.8e-2 of scale. The
  strided-depthwise padding is a LARGE deviation from the TF net the recipe now follows; fixing it
  moves both paths (JAX `mbconv_block` + the verified render's four strided depthwise sites) and is
  a decision for the user, not taken here.
* ✅ G3 for ViT: `scripts/parity/vit_timm_parity.py` (+ `_vit_timm_dump.py`) against `deit_tiny_patch16_224`
  at the reference's tanh GELU / LN ε 1e-5: Imagenette 4.7e-7, ImageNet 6.9e-7 at tol 1e-5 (tanh vs
  erf is only ~4e-5 of scale, so the CNN gates' 1e-3 would be blind to it); controls red (k/v swapped
  1.3e-1, erf GELU 3.8e-5). `--deit` (erf, 1e-6): 4.6e-5 / 3.9e-5 at random init.
* ✅ G4: `jax.yml` job `timm-parity` runs all four parity gates (with controls) and the two IREE
  forward ties (`mnv4_forward_tie.py`, `mnv2_forward_tie.py --imagenet` + its ε control) on CPU:
  jax at the lockfile's version, the pinned timm env torch-first from the CPU index, IREE 3.11.0 from
  PyPI (both ties checked on it locally). Trigger paths extended to the gates, dumps, `_iree.py` and
  the timm lockfile. Rehearsed locally step for step; first CI run is on the next push.
  `aug_bicubic_pil_check.py` is not in CI (it needs TensorFlow, which no CI job installs).
* ✅ C6 does not reach MNv2: its trainer and shim call no geometric aug or erasing, and both resizes
  are already bicubic + antialias. **The MNv2 pair (R4) is code-complete.**

## 5. Code: per net

Each block ends with the **recipe decisions** to make before its long run. Every run is costly,
so paper-fidelity knobs are chosen once, per net, here.

### 5.1 MobileNetV2
| # | fix | size |
|---|---|---|
| M2-1 | render `rmsdp64dobf16` (classifier dropout 0.2; the op and gate exist from B0), with α = 0 (C1); flip the conf variant, `CKPT_EPOCH_FILE` and precheck | M |
| M2-2 | settle H3: which run got ls 0.1 from a stale shim, and does that change the book's JAX number? | S (read logs) |
| M2-3 | JAX-path conf `mnv2-full-jax-4gpu.conf`, modelled on `mnv4-half-jax-4gpu` | S |
| M2-4 | docstrings: `apps/imagenette/MainMobileNetV2Imagenet.lean:5-10` ("AdamW here"), `VerifiedNetsCore.lean:795` ("AdamW + cosine"), `jax/MainMobilenetV2Imagenet.lean:27-28, 45-65` (68.77, "30-epoch validation recipe", stale `TODO(recipe)`) | S |

Decisions: (a) `wx` (C2): the paper's TF recipe skips BN and depthwise decay; the gap is already
inside the CI, so optional. (b) BN decay 0.997 / eps 1e-3 (TF) or keep 0.99 / 1e-5? (c) Staircase
×0.98 without warmup, or keep the warmup?

**Status (2026-09-25):** ✅ M2-4 docstrings (the driver says the optimizer follows the variant and
names the ls 0.1 / no-dropout gaps of `rmsdp64bf16`; the JAX main's 68.77 / 30-epoch / TODO text is
rewritten). The conf's ETA is marked pre-sync-BN.
✅ M2-1: `mobilenetv2in_rmsdp64wxdols0bf16` (wx, dropout keep 0.8, α = 0, sync-BN 314, bf16),
the JAX `full` recipe row for row; `mobilenetv2ImagenetVerified.dropoutKeep`; conf flipped;
`dropout-tie --net` gate W green on it (fault control red); predicate row added; 40-step 4-GPU
smoke + resume green (loss 6.97 at step 0 = ln 1000 with no smoothing). ✅ M2-2: the stale-emit
note dates to the JAX run's launch (07-28), so the 71.90 most likely trained at ls 0.1 like the
verified run; both reruns are at 0. ✅ M2-3: `mnv2-full-jax-4gpu.conf`. Decision (a) taken: `wx`
on both paths (free, both runs are redone anyway). (b)/(c) still open.
**Decisions (b)/(c) taken, 2026-09-25 (the user): TF-slim on both.** BN decay 0.997, ε 1e-3;
×0.98 per epoch as a staircase from step 0, no warmup. ✅ Landed: JAX `TrainConfig.bnEps` /
`expLRStaircase` (defaults byte-identical; only the two MNv2 generated files move); verified
`mobilenetv2in_rmsdp64wxdols0eps0001bf16` + its eval partner `mobilenetv2in_fwd_eval_eps0001`
(ε is baked, so a new pair; each differs from its 1e-5 sibling only in the ε constants and the
entry), `VerifiedVariant.evalTag` so the trainer and `score-checkpoint` score an `eps` variant
through its own eval graph, `mnv2ImagenetRmsSchedule` (warmup 0, staircase) and
`bnMomentum := 0.997` in the driver; `mnv2-default-4gpu` flipped, `mnv2-full-jax-4gpu` checks the
emitted ε/decay/floor. The MNv2 pair's code is closed apart from C6 and G3.

### 5.2 MobileNetV4-Conv-M
| # | fix | size |
|---|---|---|
| M4-1 | `mnv4-half-4gpu.conf`: `EPOCH_SECS=960` and the ETA "11–15 h" are wrong. Measured ~1,500 s/epoch, so ~21 h for 50 epochs, about three overnight chunks; the JAX conf's 9–10 h is probably optimistic for the same reason | S |
| M4-2 | retire `mnv4-default-4gpu` (500 ep, f32 `adamdp64`, precheck needs the dead `shard-check` exe, "AutoAugment AND RandAugment") or re-point it to the recipe variant; fix `lakefile.lean:2381` | S |
| M4-3 | score the **live** weights beside the EMA shadow at the end of a run: τ = 10k updates is 1.56 τ over 50 epochs, so the shadow lags | S code |
| M4-4 | verified UIB drop-path is not rendered (`sdOn`/`dropKeeps` empty for `mnv4in`, `VerifiedNetsCore.lean:1646-1651`); JAX has it on `full` only | M |
| M4-5 | RandAugment m15 exceeds `_AA_MAX = 10` (extrapolated); no p0.7 knob | S–M |
| M4-6 | docstrings: `apps/imagenette/MainMobilenetV4Imagenet.lean:10-43`, `MobileNetV4RenderB.lean` ~862 (pre-timm stem/stage/head), `jax/MainMobilenetV4Imagenet.lean:13-16, 122` ("not yet wired into UIB" is false), `proofs.yml:280` ("Conv-S"), the driver banner "NOT rsb-faithful — LAMB and BCE", `VerifiedTrain.lean:2755` | S |

Decisions: (a) EMA stays for the 50-epoch pair or goes, to match the paper? (b) eval crop 0.95 /
test at 256 (a shim change, so both paths)? (c) what the paper tier needs: 256 train, m15 p0.7,
drop-path 0.075, wd 0.1, dropout 0.2, 500 ep. Is a paper-tier run in scope at all?

**Status (2026-09-25):** M4-1 measured and landed (~1,516 s/epoch over the first 16 epochs, from
the master log's launch time to the epoch-16 checkpoint), then overtaken by the 100-epoch rescope,
which owns the MNv4 confs, the lakefile row and the MNv4 driver docstrings. M4-6: `proofs.yml`
Conv-S → Conv-M done; the "not yet wired into UIB" comment was already false (UIB drop-path is in
the `full` trainer); the "NOT rsb-faithful — LAMB and BCE" banner is at `VerifiedTrain.lean:1750`,
unchanged; `MobileNetV4RenderB.lean` ~869-875 still describes the pre-timm net.
**Rescope (2026-09-25, the user):** the 50-epoch `half` pair is retired; the 100-epoch `default`
recipe is the pair. `mnv4-default-4gpu` (the old half conf, EPOCHS=100, checkpoint tag `e100` so the
retired run's epoch-16 checkpoint is never resumed; ~42 h) replaces the stale `adamdp64` conf;
`mnv4-default-jax-4gpu` (RECIPE=default, `~/mnv4_timm_100ep`); JAX `half` recipe and its generated
files deleted; `planning/mnv4_half_pair.md` archived. Book: after the pair, the phase-2 warning goes
and "TODO: 500 epochs" replaces it. ✅ The RSB banner now prints only for the ResNet-50 family;
the render docstring describes the timm net.

### 5.3 EfficientNet-B0
| # | fix | size |
|---|---|---|
| B0-1 | `wx` on both paths (C2): TF and timm exclude BN (and timm biases) | M |
| B0-2 | re-probe ms/step on the sync-BN render before the rerun: the conf's "68 h, 134 ms" is from the per-replica render, and the run took 73.4 h | short GPU |
| B0-3 | `bnFirst` running-stat seeding vs JAX's 0/1: a checkable candidate for the "unexplained" epoch-1 20.62 vs 10.09 | S, investigation |
| B0-4 | docstrings: `apps/imagenette/MainEfficientNetImagenet.lean:9-10, 22-23, 50-51`, `VerifiedNetsCore.lean:915-918`, `jax/MainEfficientNetImagenet.lean:53-86, 107` (lr 0.045, "0.01" peak where it is 0.016); confs and planning say "AutoAugment + RandAugment", but only AutoAugment is called | S |
| B0-5 | conf: `RECIPE=default` (the 80-ep tier name) beside `EPOCHS=350`; trim the withdrawn-measurement header | S |

Decisions: (a) staircase decay from step 0 (paper) or keep continuous-after-warmup and fix the
book? (b) BN eps 1e-3? (c) drop-connect i/16? All three are cheap to render and ride on the same
rerun.
**Decisions taken, 2026-09-25 (the user): follow TF.** (a) staircase ×0.97 / 2.4 epochs counted
from step 0 (the 5-epoch warmup stays, layered on top as in the TF code); (b) BN ε 1e-3 (TF; timm's
own `efficientnet_b0` is 1e-5, `tf_efficientnet_b0` 1e-3); (c) drop-connect i/16; and B0-1 `wx` ON
(the C2 note's "B0 does not" had no decision behind it). A JAX rerun rides along (R5).
✅ Landed (branch `mnv2-tf-recipe`): JAX `wdExcludeNormBias`, `bnEps := 1e-3`, `expLRStaircase`
(now TF's form: floored on the GLOBAL step, the warmup overriding it only while it runs) and
`dropPathOverN` (i/16 on the MBConv path; ramp ends at keep 0.8125); verified
`efficientnetin_emarmsdp64dropdowxeps0001bf16` (131 1-D params on `%wdz`, 196 ε sites; otherwise
byte-equal to `emarmsdp64dropdobf16`) + `efficientnetin_fwd_eval_eps0001`, `bnEpsMarker` /
`fwdEvalEntry` moved to `RenderKit` (shared with MNv2), `enetImagenetRmsSchedule`, the ImageNet
`dropKeeps` at i/16 (pinned in `TestDropPathRamp`); `enet-default-4gpu` flipped; new JAX-path conf
`enet-full-jax-4gpu` (fresh `~/enet_tf350`, precheck greps ε, floor, i/16, wd mask).
⚠ G7 still open: `imagenet-syncbn-check` cannot feed EMA or the host drop/dropout masks, so it
covers `rmsdp64bf16`'s scope, not the shipping variant's.

**Status (2026-09-25):** ✅ B0-4 docstrings (peak lr 0.016, AutoAugment only, continuous decay,
decay on every parameter). ✅ B0-5 header trimmed to the 2026-09-12 table plus a short history;
the ETA is now the measured 73 h, marked pre-sync-BN. ⚠ The `RECIPE=default` vs 350-epoch
(`full`) mismatch is annotated, not renamed: a rename moves the lakefile row and the book.

### 5.4 ConvNeXt-T (and S/B)
| # | fix | size |
|---|---|---|
| CX-1 | `cnx-default-emabf16-4gpu.conf` header lines 38–42 were pasted from the other conf ("flipped IN PLACE", "Ch. 8 has never run"); the ETA is the non-EMA 3060 number; both ConvNeXt confs say "AutoAugment + RandAugment" | S |
| CX-2 | `scripts/jobs/run_cnx_verified.sh` hard-codes `RUNDIR=runs/2026-09-18-cnx-verified-300ep` and the non-EMA job, so it would resume into a finished run | S |
| CX-3 | `summarize.sh` says the curve is "unaffected" (`RESULTS.md` §3 notes it is wrong) | S |
| CX-4 | S/B: JAX mains lack `cnxInit` (Xavier-uniform, `generated_convnext_s_imagenet.py:424`); `cnxs/cnxb` confs run f32 at 4×32 (global 128, 10,009 steps/epoch) at 2.5e-4, with no `LEAN_MLIR_EPOCHS`/`BASE_LR_U` and the old `.venv` precheck; they need T's 64-per-replica rescope, bf16 variants and C7 | M |

Decisions: (a) exact-erf GELU on both paths (touches the proof-side GELU node: cost it first)?
(b) drop the clip for the 300-ep recipe? (c) `pixel` erasing lands with C6 regardless.

**Status (2026-09-25):** ✅ CX-1: `cnx-default-emabf16-4gpu` is a short conf that sources
`cnx-default-4gpu` and swaps the variant; its ETA says it carries the non-EMA number. Both confs'
"AutoAugment + RandAugment" became RandAugment + random erasing. ✅ CX-2: `scripts/jobs/run_cnx_verified.sh [JOB]`
defaults to the EMA job and no longer pins a RUNDIR. S/B confs use the shared checks; CX-3/CX-4 open.

### 5.5 ViT-Ti (and S/B)
| # | fix | size |
|---|---|---|
| VT-1 | JAX-path conf for `deit-init`; make `deit-init` the `default` recipe, or rename (`jax/MainVitImagenet.lean:84` still defaults to Xavier) | S |
| VT-2 | worker counts contradict: `vit-default-4gpu` has `SHIM_WORKERS=8` and calls the 08-31 sweep superseded; the emabf16 conf enforces 4, citing that sweep. Pick one by a median probe; retire `vit-default-4gpu` or move it to bf16+EMA | S + short GPU |
| VT-3 | S/B drivers (`apps/imagenette/MainViT{S,B}Imagenet.lean`) don't set `vitInit` (Glorot, the very gap the book credits for Tiny); vits/vitb confs launch f32 with no EMA; no EMA render for S/B | S (init, confs) / M (renders) |
| VT-4 | docstrings: `VerifiedNetsCore.lean:1320-1325` ("mixup/EMA/clip don't exist on the verified path"), `apps/imagenette/MainViTImagenet.lean:13-26`, `jax/MainVitImagenet.lean:3-12` ("2-GPU", "gfx1100"), `ViTFoldGB.lean:46` (names the non-EMA render), the shim's "RSB-A2 3×" repeated-aug comment, the emabf16 conf's "box-specific, cuda13" header | S |
| VT-5 | proofs: `ViTStepTieGB` has no mention of bf16. Confirm the bf16 render the book's number comes from is covered by the bf16 fold nodes | S, read-only |

Decisions: (a) the clip-off arm (DeiT has none): short probe first? (b) LN eps 1e-6 + erf GELU
(re-render + pair rerun; G3 pins them either way)? (c) cooldown + min_lr 1e-5? (d) score the
live weights, as DeiT does, or keep EMA and say so?

**Status (2026-09-25):** VT-2 partial: both confs keep their worker counts and name the open
probe instead of contradicting each other; `vit-default-4gpu` says it is the f32 sibling and not
the shipping job; `vit-default-emabf16-4gpu` has the measured 48 h ETA and no longer calls itself
3060-only. ✅ VT-4 docstrings; the "RSB-A2 3×" comment is in `jax/Jax/Codegen.lean:487`, unchanged.
✅ VT-1: `vitInit := true` in the Ti/S/B base configs, so `default`/`short`/`accum` carry DeiT init
(each new `default` is byte-identical to the old `deit-init` file); `deit-init` recipes deleted; Ti
keeps `xavier` for the A/B. `vit-default-jax-4gpu.conf` checks the smoothing fix and the init.
⚠ The "RSB-A2 3×" comment is emitted text: fixing it moves R50's generated trainers while
R50-2018 runs on the other box, so it waits. The book still says `deit-init` (K1).

## 5.6 Scoring: timm's validation protocol, on both paths

The rule (the user, 2026-09-25): score the way timm validates. timm scores at the pretrained
config's `test_input_size` / `test_crop_pct`, which is often not the training size.
`scripts/parity/timm_eval_protocols.py` (run from `.venv-timm`) reads them from the pinned timm and writes
`jax/timm_eval_protocols.json`, keyed by generated trainer:

| net (our recipe) | timm tag | train | timm test |
|---|---|---|---|
| R34, R50 2018 | `resnet{34,50}.tv_in1k` | 224 / 0.875 | 224 / 0.875 (bilinear) |
| R50 RSB-A3 | `resnet50.a3_in1k` | 160 / 0.95 | 224 / 0.95 |
| R50 RSB-A2 / A1 | `resnet50.a{2,1}_in1k` | 224 / 0.95 | **288 / 1.0** |
| MNv2 | `mobilenetv2_100.ra_in1k` | 224 / 0.875 | 224 / 0.875 |
| MNv4-Conv-M (ours, r224) | `mobilenetv4_conv_medium.e500_r224_in1k` | 224 / 0.95 | **256 / 1.0** |
| B0 | `tf_efficientnet_b0.in1k` | 224 / 0.875 | 224 / 0.875 |
| ConvNeXt-T/S/B | `convnext_*.fb_in1k` | 224 / 0.875 | **288 / 1.0** |
| DeiT-Ti/S/B | `deit_*_patch16_224.fb_in1k` | 224 / 0.9 | 224 / **0.9** (we crop 0.875) |

| # | item | status |
|---|---|---|
| S1 | protocol table from the pinned timm, with `--check` | ✅ `scripts/parity/timm_eval_protocols.py`, `jax/timm_eval_protocols.json` |
| S2 | JAX forwards score at any size: every conv stem infers its square side (`_s = …`, the A3 idiom), ViT stays fixed (pos-embed tied to the grid; timm scores DeiT at 224) | ✅ `Codegen.lean`; 24 trainers moved one line each |
| S3 | `jax/scripts/eval_full50k.py`: `PROTOCOL=train\|timm\|both`, `EVAL_SIZE`/`EVAL_CROP`; refuses a fixed forward at a foreign size | ✅ A/B on the RSB-A3 rerun: committed scorer 74.64 / 91.75, new 74.62 / 91.75 (XLA noise); B0 `both` gives two identical passes |
| S4 | the verified path at timm's test size, scored by a VERIFIED graph (the user's call): eval renders at the test size (`mnv4in_fwd_eval_s256`, `convnextin_fwd_s288`; `f`/`s` parameters on the MNv4 and ConvNeXt forward chains, default byte-identical, proofs untouched), `SHIM_EVAL_SIZE`/`SHIM_EVAL_CROP` in every shim, `score-checkpoint` under `LEAN_MLIR_EVAL_SIZE`/`_CROP`, `scripts/parity/score_timm.sh <net>` | ✅ MNv4 epoch-17 checkpoint: 67.10 at 224 / 0.875, **68.02** at 256 / 1.0; ConvNeXt 288 plumbing smoke green. Open: RSB-A1/A2 at 288 (R50 renders) |
| S4b | the other box: the generated trainers and shims moved (forward reshape, eval-size override). Run `scripts/regen_jax_generated.sh sync` there after pulling, or its prechecks refuse | note |
| S5 | DeiT's 0.9 crop in the in-training eval (shim constant, both paths, no render) | open |
| S6 | retire the six per-net `jax/scripts/eval_<net>_full50k.py` copies for the generic one | open |

⚠ Found while testing: checkpoints trained before the timm-protocol preprocessing change do not
reproduce their book number under today's eval — the July RSB-A3 rerun (book 77.22) scores 74.62
at 224 / 0.95 through both the old and the new scorer, and the July B0 checkpoint here scores
71.70. Book numbers are each run's own in-training eval at the time; rescoring an old checkpoint
is not a like-for-like check. (The canonical B0 77.15 and ConvNeXt 81.53 checkpoints are on the
3060 box.)

**Notes (the user, 2026-09-25):**
* MNv4: run the 100-epoch pair at 224 as configured, to get a number on the board first; 256
  training and the 500-epoch paper tier come after. Then bring the MNv4 section into line with the
  rest of the ImageNet chapters (reference → verified → side-by-side, like ch 5/6/9), with the
  phase-2 warning replaced by "TODO: 500 epochs".
* Book appendix A (Data availability, the "four things are matched" timm paragraph): add a line
  for the test-resolution column — each net is also scored at timm's `test_input_size` /
  `test_crop_pct` from the pinned timm.

## 6. Book sync (after §3–§5 close for that net)

| # | where in `content.tex` | fix |
|---|---|---|
| K1 | ViT ~12102–12110, 12215–12275 | caveat H1 now; the pair's claim returns with the JAX rerun (§7); add EMA-vs-live, cooldown/min_lr, repeated-aug placement |
| K2 | MNv2 7705–7740, ~7690, 7498, 7619–7620 | the side-by-side table lists all four rows (H2), or three after M2-1; `epochs := 90 -- the real run` is the 350-ep `full` recipe; the BN-decay row compares against PyTorch 0.9 where TF's 0.997 is the paper's; resolve the ls ledger row (H3) |
| K3 | MNv4 7893–8083, 16742–16743 | EMA "none" for Conv-M; ε 1e-7; add a resolution row; "ships two tiers" ignores `half`/`probe`; label the 49,664-image in-loop curve against the 50k 75.48; "+3.58 over MNv2" is pre-fix; the Phase-4 prose (`adamdp64` table, "TODO: run mnv4-default-4gpu") is superseded; Track 4 points at `mnv4-default` |
| K4 | B0 8392–8394 | `thm:efficientnet_step_tie` claims 262 gradients of "every `efficientnetin_*` artifact"; the Lean theorem is pinned at 10 classes (`EfficientNetStepTieG.lean:414,434`; sync twin `SyncStepTieG.lean:1548`), and the ImageNet renders have 213 tensors at 1000 classes. Restate the scope |
| K5 | B0 9153, ~8896, 9030–9031 | "staircase" (code is continuous); "194 logistic" is now 178, "147 splats of 1e-5" now 196: pin to the run's commit or update |
| K6 | ConvNeXt 9974–9975, 10030–10036, 10128–10137, 10208–10210, 10360–10378 | "no deviation remains" omits the batch, clip, tanh GELU and erasing mode; "four of six knobs / no artifact combines EMA" is false since e38e56ab; "7.4% discordance is the run-to-run spread" was never measured (no two-seed pair); the S/B side quest shows global 128; settle top-5 95.50 vs the log's 95.51 |

## 7. Runs (schedule once the net's code is closed; each needs the user's word)

| # | run | absorbs | est. | box |
|---|---|---|---|---|
| R1 | MNv4 100-ep pair: `mnv4-default-4gpu` + `mnv4-default-jax-4gpu` | the rescope | ~42 h + ~16–18 h | ares, chunks |
| R2 | ConvNeXt-T EMA: smoke + resume, then the 300-ep run | CX-1, G5 | smoke ≤1 h; ~100 h | |
| R3 | ViT-Ti JAX `vit-default-jax-4gpu` rerun with smoothing | §5.5 decisions, C6 if landed | ~34 h (3060 box) | |
| R4 | MNv2 JAX rerun (ReLU6 stem/head) + verified rerun (sync-BN, dropout, α = 0) | M2-1, §5.1 decisions, C6 | ~38 h + ~51 h | |
| R5 | B0 verified sync-BN rerun (and JAX if `wx`/staircase land) | B0-1, B0-2, §5.3 decisions, C6 | ~75 h (+ ~47 h JAX) | |
| R6 | short probes: ETA re-probes on the sync-BN renders (every BN net; only `r50-a3-4x128` has been done), VT-2 worker probe, G1/G2/G5/G7 | — | ~30 min each | |

**Pre-launch checks, 2026-09-25 (all green):** sync-BN split-batch `imagenet-syncbn-check`
mobilenetv2 / efficientnet (4×64 = 1×256); every shipping verified variant (cnx EMA, B0 sync-BN,
MNv2 wx/do/ls0, ViT EMA) probed at 600 steps, scored at epoch 1 and resumed into epoch 2; the new
ViT JAX `default` trainer smoked 15 min. Probed on 4× 4060 Ti: cnx 237 ms (~107 h), B0 148 (~80 h),
MNv2 123 (~66 h). ViT is FEED-BOUND here (verified 288 median / 691 mean; JAX ~380 ms, ~79 h vs
34 h on the 3060 box): both ViT runs go on the 3060 box.

Order: R1 is already queued; R2 needs no code change beyond CX-1. R3–R5 wait on their §5
decisions and on C6, so one rerun per net takes every change.

## 8. Housekeeping

* Archive `planning/next_session_{mnv2,enet,convnext}_verified_run.md` (all done; only enet says
  CLOSED).
* Run records missing on this box: ViT 72.31 and 72.35 (both ran on the 3060 box; the verified
  archive `runs/2026-09-08-vit-verified-300ep-det0/` is named in ea96760c), and the MNv2 JAX 71.90
  (only the book and the commit message record it). `jax/runs/mnv2_imagenet_bf16_90ep/RESULTS.md`
  is the retired SGD run and is not marked superseded.
* Stale ETAs: enet/cnx/r34 quote 4×3060 only; every BN-net ETA predates sync-BN; `mnv4-default`
  is pre-timm; `vit-default` cites ares 08-05 ("32 cores", "AutoAugment"); `vit-default-emabf16`
  has no ETA despite a measured 48 h.

### §8 status (2026-09-25)

* ✅ `next_session_{mnv2,enet,convnext}_verified_run.md` moved to `planning/archive/`.
* ✅ `jax/runs/mnv2_imagenet_bf16_90ep/RESULTS.md` carries a SUPERSEDED banner.
* Stale ETAs: every BN-net conf ETA now says pre-sync-BN, re-probe owed; `vit-default` rewritten.
