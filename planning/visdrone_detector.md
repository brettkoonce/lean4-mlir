# visdrone_detector.md — R34 + FPN head on VisDrone, the single source of truth

**Written 2026-07-23.** Consolidates the scattered detector docs into one current
plan. Supersedes the forward-looking parts of `yolo_fpn.md`, `yolo_drone.md`,
`yolo_final.md`, and retires `yolo_scoring.md` + `yolo_assignment.md` (both carry
⛔ banners — their analysis was measured on scrambled data, see below). Those
stay as historical reference; **work from this doc.**

Live companions: memory `yolo-fpn-thread` (the investigation history),
`edge-deploy-orin` (the deployment half), `brats-demo-thread` (the "clever loss
levers don't beat plain training" lesson that applies here too), and
`post_shuffle_fix.md` (the bug ledger this grew out of).

---

## 0. TL;DR — where it stands

> **▶ RESUMED 2026-08-28. The tree was empty; it has been rebuilt and re-verified.**
> The code and these docs survived the pause. The data, every checkpoint, the
> backbone file, and the PyTorch twin's venv did not — they were deleted to free
> disk. §12's resume checklist is therefore **obsolete**: its step 1 (re-score
> `long30`) cannot run, because those checkpoints are gone. The `long30` result
> stays VOID and "train longer" is now being answered by a fresh 50-epoch run
> rather than a re-score.
>
> **What changed while this was parked, all of it load-bearing:**
> - **The trainer got 9.47× faster** (`planning/detector_pjrt_port.md`, DONE
>   2026-08-07). A 12-epoch run was 4.5 h and is now ~45 min. Every cost estimate
>   in this doc and in `coco_visdrone_two_stage.md` predates that.
> - **The box is now 6× RTX 4060 Ti on CUDA.** Every `IREE_BACKEND=rocm` and
>   `HIP_VISIBLE_DEVICES` in these runbooks is dead weight; use
>   `CUDA_VISIBLE_DEVICES`. Arms can run in parallel, one per card — though the
>   **host data loader is the shared bottleneck**: epochs stretch from 208 s to
>   ~400 s once five trainers are live.
> - **Aug pack is confirmed ACTIVE, finally.** `FPN_AUG=1` raises epoch-1 train
>   loss (356.4 vs 354.5) and epoch time (305 s vs 208 s). "Is it even on" was a
>   live question; it is on. Still opt-in and still off by default.
>
> **Rebuild recipe (verified faithful — reproduce these counts or stop):**
> `./download_visdrone.sh`, then `preprocess_visdrone.py data/visdrone
> data/visdrone_fpn --size 448 --grid 14 --fpn data/visdrone`, then the same
> without `--fpn` into `data/visdrone448` for the scoring sidecar. ⚠ Do **not**
> regenerate the anchor priors with k-means — write
> `data/visdrone/anchors_fpn_{p3,p4,p5}.txt` from the values hardcoded in
> `demos/MainYolov1VisdroneFpn.lean`, so encoder and model cannot disagree.
>
> | quantity | historical | rebuilt |
> |---|---|---|
> | train GT boxes | 343,204 | 343,204 |
> | val GT boxes | 38,759 | 38,759 |
> | val boxes/img | 70.7 | 70.7 |
> | encodability | 88.2% | 88.2% |
>
> **Backbones — the file situation is the trap now.**
> - `jax_r34_imagenet.bin` was deleted and only a full ImageNet run regenerates
>   it. But it is exactly the **first 85,138,688 bytes** of the packed
>   `.lake/build/resnet34in_momdp64_ckpt_xla.bin` (θ is 21,797,672 floats =
>   21,284,672 body + 513,000 fc). Reconstructed that way it starts at loss 717.7
>   and reaches epoch-1 354.7 — **better** than the historical 391.1, because it
>   is the 30-epoch verified checkpoint rather than the older ~72% one. That is a
>   confound on any comparison to 0.1386, which is why a matched 12-epoch arm runs
>   alongside.
> - ⛔ **R50-A3's `ckpt.bin` cannot be bootstrapped raw.** Its conv weights are
>   perfect (std 0.17, every segment matching `ckpt_e100.state.npz`) but **every
>   per-channel BN slot runs to 5.2e6**, against R34's γ∈[0,0.60] / β∈[−0.68,0.77].
>   Loaded as-is the detector starts at loss **5.8e8** where R34 starts at 717.7.
>   Those values are presumably meaningful to the render that wrote them (it did
>   score 77.2%) — most likely a scale folded against running statistics kept in
>   the `.state.npz`, which the generic bootstrap loads as zeros, so nothing
>   cancels. **Not a codegen bug**: the same spec under `FPN_NOBOOTSTRAP=1` starts
>   at 8,284.8 and trains normally. Workaround in place — keep the conv weights,
>   reset BN to γ=1/β=0 (`r50_a3convonly_params.bin`); that starts at 8,567.6 and
>   is ahead of He init by step 100. Clean fix if full transfer is ever wanted:
>   re-export A3 through `LEAN_MLIR_PARAMS_OUT`, the path that produced R34's file.
>
> **Codegen: two limits found putting `.fpnDetect` on a bottleneck backbone**
> (both fixed/worked around, see `LeanMlir/MlirCodegen.lean`):
> - `.fpnDetect` taps `fpnStages`, which only `.residualBlock` populated. On R50
>   the list was empty, the head emitted a comment instead of a graph, and the
>   backward then referenced `%d_logits` — a classifier gradient that does not
>   exist under a detector. Fixed: bottleneck stages now register as taps in both
>   walks. Additive; nothing but `.fpnDetect` reads that list.
> - ⚠ **`.maxPool` with size > stride has no correct backward.** The tile-compare-
>   select routes gradient to the argmax and is valid only for non-overlapping
>   windows; with overlap one input can be the max of several and its gradient is
>   a sum the tiling never forms. R50's `.maxPool 3 2` *also* fails to parse first
>   (forward pads 224→225, backward reads the unpadded `inShape`), so fixing that
>   type error alone would trade a loud failure for a silent one. Use
>   `.maxPool 2 2` — pooling is parameter-free, so a pretrained prefix still aligns.
> - Incidental: `convPadStyle` is declared in `Types.lean` and **read nowhere in
>   `MlirCodegen.lean`**. Setting it on a spec that trains through the generic walk
>   does nothing.

A ResNet-34 + FPN anchor detector, trained on VisDrone, **works**: full-GT
**mAP@0.5 = 0.1386**, recall 0.676, class-agnostic AP 0.376 at 12 epochs. That is
**90.5% of its faithful PyTorch twin** (0.1532), with recall essentially tied
(0.676 vs 0.677) — so the ~10% gap is **ranking/classification, not
localization**. The detector is **undertrained** (e12 loss still descending); the
0.1386 arm ran with **no augmentation**, though an online HSV + horizontal-flip
pack now exists as an opt-in A/B arm (`FPN_AUG=1`, §6.2). The whole stack is self-hosted: the R34 backbone
was trained *by this stack* on ImageNet (72% top-1), the head trains on VisDrone,
no borrowed weights. Part-3 goal = a practical, self-hosted detector that trains
from free data and **deploys to an edge device** (Orin path already validated).
Verification/proofs are NOT required here — codegen + FD only.

**One thing gates every external number:** score against the full-GT sidecar
(§5), not the 56-box-capped tail. `0.1386` is the real number; `0.1167` was the
capped artifact.

---

## 1. Architecture

Spec `r34FpnDetT` in `demos/MainYolov1VisdroneFpn.lean`; layer
`.fpnDetect 256 128 256 512 14 3 tower` (oc=256; C3/C4/C5 = 128/256/512; coarsest
grid 14; A=3 anchors/scale; tower depth via `FPN_TOWER`, default 0).

- **Input:** 448×448 RGB, ImageNet-normalized `(px/255 − mean)/std`
  (mean `[.485,.456,.406]`, std `[.229,.224,.225]`). The backbone was trained in
  this normalized space, so it is not optional.
- **Backbone: ResNet-34**, self-trained on ImageNet (`jax_r34_imagenet.bin`,
  21,284,672 floats, 72% top-1). Emits C3/C4/C5 at strides 8/16/32.
  **Padding is TF-style ASYMMETRIC SAME** (`MlirCodegen.samePad`) — do NOT drop
  in torchvision weights (symmetric-trained ⇒ 1-px grid shift, compounds through
  4 stages; this is what broke the twin forward until `pad="lean"`).
- **Neck: FPN** — 3× 1×1 lateral conv + top-down bilinear upsample + add ⇒
  P3 (56²), P4 (28²), P5 (14²).
- **Head: per-scale 1×1 conv**, anchor-based, **15 channels/anchor** =
  4 box + 1 objectness + 10 class. With a RetinaNet **prior bias** on the obj
  channel (`detPriorPi := 0.01`; a zero-init bias is a byte-exact no-op, so it
  composes cleanly). `FPN_TOWER=N` adds N 3×3+ReLU convs per scale (no norm) —
  measured to do **nothing** (see §7), off by default.
- **Output width Ntot = 185,220** = Σ_s A·15·g_s² over g∈{56,28,14}. Aligned
  slot-for-slot with the training target.

---

## 2. Data pipeline

VisDrone-DET2019: 6,471 train / 548 val images, 10 classes, ~70 objects/image,
mostly **tiny** (a 20–25 px source object is ~2–5 px after the 448 resize). This
tininess IS the difficulty and the reason for the multi-scale head.

**Preprocess** (`preprocess_visdrone.py --size 448 --grid 14 --fpn data/visdrone
<visdrone_dir> data/visdrone_fpn`): each image → 448×448 uint8 CHW + a flat
185,220-float target (`process_split_fpn`). Per record = 1,342,992 bytes;
train.bin ≈ 8.3 GB.

**Target encoding** (`encode_targets_fpn`), per GT box:
1. **scale by size** — `max(w,h)·448` → P3 (tiny) / P4 / P5 (big).
2. **cell by center**, **anchor by best size match**.
3. write 15 numbers into that (scale, cell, anchor) slot: `[0:4]` box (in-cell
   center offset + w/h), `[4]=1` objectness, `[5:15]` class one-hot. Rest = 0.
4. collisions = last-write-wins (the residual vs the 88.2% coverage ceiling).

**Eval GT: the full-GT sidecar.** `pack_raw_boxes` caps at MAX_BBOXES=56, but val
averages 70.7 boxes/img, so the training record's box tail drops 34.9% of val GT.
Eval instead reads `data/visdrone448/val.full_gt.bin` — an uncapped,
variable-length sidecar written in the SAME loop as `val.bin` (order aligned by
construction). The scorer prefers it automatically. See §5.

**Pairing is load-bearing and was the great bug of July 2026.** The image at slot
k must pair with target k; `lean_f32_shuffle` used to permute images by a full
record but labels by a hardcoded 4 bytes, scrambling every det/seg batch every
epoch (mAP 0.0001). Fixed (label stride threaded through); guarded (exact-size
check both directions); pinned (`tests/TestShufflePairing.lean`,
`tests/TestDatasetRecordSizes.lean`). Full story: `post_shuffle_fix.md`.

---

## 3. Loss

`emitMultiScaleYoloLoss` (MlirCodegen ~6888), summed over the 3 scales:
- **Box** — DIoU, λ_box = 5.0, only on assigned cells (masked). `tw,th` capped
  at 8 before `exp` (the unbounded op; uncapped it hit `inf·0=NaN` under the
  global-norm clip — see `post_shuffle_fix.md` gotchas).
- **Objectness** — focal, γ=2.0, on **ALL** cells (~12,348/img, ~45 positive).
  This imbalance is the hard term and where the twin gap lives.
- **Class** — weighted CE (sqrt-inverse frequency, `fpnClsWeights`), only on
  assigned cells. Weights are target-only ⇒ still exactly FD-checkable.

**Cross-check FD-verified** (`fpn_loss_probe`, `fpn_detect_probe` at γ=0). The
loss and its VJP are correct — that was never the problem; the *data* fed to them
was (§2).

---

## 4. Training

`r34FpnDetConfig`: lr 4e-4, batch 8, Adam, wd 5e-4, cosine decay, warmup 3,
gradClip 4.0, focal γ=2, class weights on, prior bias π=0.01, backbone bootstrap.
Default 12 epochs, checkpoint every 2. Loop: load train.bin into RAM →
**shuffle image+target together** each epoch → batch forward (R34→FPN→heads) →
multi-scale loss → backprop through head→neck→backbone (`fpnTapGrad` seam) →
Adam+wd step. Train step is `iree-compile`'d to a vmfb; ~1.65 s/step, ~24
min/epoch on gfx1100 (conv-weak).

**Env knobs:** `FPN_TAG` (checkpoint prefix — NOT optional, an untagged probe
overwrites a live arm), `FPN_EPOCHS`, `FPN_CKPT_EVERY`, `FPN_TOWER`,
`FPN_LR_MULT` (0 = frozen-param probe), `FPN_CLIP`, `FPN_AUG` (1 = the
augmentation pack, §6.2; off by default so the baseline stays byte-reproducible —
run it under its own `FPN_TAG`). `IREE_BACKEND=rocm` is **required** on the train
step or the loss reduce fails to distribute.

---

## 5. Current results & the twin gap

12-epoch arm (`runs/yolo_fpn_shuffix`), full GT unless noted:

| metric | Lean FPN | twin (`runs/twin_r34_12ep`) | Lean/twin |
|---|---|---|---|
| mAP@0.5 (full GT) | **0.1386** | 0.1532 | 90.5% |
| recall (full GT) | 0.6756 | 0.6772 | ~tied |
| class-agnostic AP | 0.3763 | 0.3997 | 94% |
| mAP@0.5 (capped GT) | 0.1167 | 0.1299 | — |
| recall (capped GT) | 0.7353 | 0.7378 | — |

Reads: recall/localization essentially **match the twin** ⇒ the ~10% gap is the
per-class head/ranking (Lean's Adam+coupled-wd vs the twin's decoupled AdamW; the
class head is a softmax masked to positives ⇒ never trained on background). BOTH
are undertrained (recall ~0.68 ≪ a converged RetinaNet-class detector). **The
yardstick is the twin, not YOLOv8** — YOLOv8 is a different, stronger
architecture; comparing to it conflates architecture with implementation, which
is the confound the twin exists to remove.

---

## 6. The plan — in priority order

1. **Train longer (RAN, but the result is VOID — must re-score).** The 30-epoch
   probe (`run_fpn_long.sh`, `FPN_TAG=long30`) completed on 2026-07-23: tagged
   checkpoints `…__long30_params_e{5,10,15,20,25,30}.bin` exist on disk. **But its
   mAP is not usable.** The eval reported mAP@0.50 = 0.0002 with byte-identical
   rows at e5/e10/e15/e20/e25/e30 (same TP=4190, same 541949 dets) — the tell that
   all six passes scored the *same* file. Root cause: `run_fpn_long.sh:50` ran
   `… infer …` **without** `FPN_TAG=long30`, so the untagged spec loaded
   `…__visdrone__params.bin` (the baseline pointer, itself stale/bad) instead of
   `…__long30_params.bin`. This is exactly the trap the `inferDump` comment warns
   about ("six identical rows was the only tell"). **Fixed** (the infer line now
   carries `FPN_TAG=$TAG`), but the corrected re-score has NOT been run.
   - Secondary, UNCONFIRMED: the `long30` *training* loss (`runs/fpn_long30_gpu0.log`)
     never visibly descended — per-batch loss thrashes 33–96 and sits at 66 at e30
     (lr→0). That's noisy single-batch loss, not epoch means, so it is *not* proof
     of a broken train; but with the eval also broken, treat "30 epochs helped" as
     an open question. **The re-score decides it:** e30 mAP ≥ the 12-ep 0.1386 ⇒
     only the eval was broken, training is fine, keep pushing epochs; e30 mAP ≈ 0
     or ≪ 0.1386 ⇒ the longer/full-data run regressed and needs its own root-cause
     (suspect LR schedule, the `fpnTapGrad` seam, or a full-data-only path).
   - **Do NOT trust any long-run number until a tagged re-score confirms it.** The
     original read ("still climbing / plateaus early ⇒ decide aug") is unanswered.
2. **Augmentation for the FPN path (COMMITTED `9280a3d`, NOT yet A/B'd — `FPN_AUG=1`).**
   The pack is two *online* augmenters, no dataset regeneration:
   - **HSV jitter** (photometric, image-only) — YOLO-style multiplicative h/s/v
     gains (0.015/0.7/0.4), one draw per image, in the FPN DataIO's `augmentBatch`
     hook (which fires but was a no-op before). Runs in [0,1] sRGB with the
     ImageNet de-norm/re-norm round-trip (`lean_f32_hsv_jitter`).
   - **Horizontal flip** (geometric, paired) — mirrors the image AND the flat
     `[P3|P4|P5]` target together, p=0.5/image (`lean_f32_fpn_hflip`, a new
     `useFpnRun && cfg.augment` block in the train loop). The flip is
     *shape-invariant*, so every GT keeps its scale AND best-shape anchor: only
     the grid columns mirror and the in-cell `tx` → `1-tx` on assigned cells. This
     is EXACT — proven equal to a full re-encode of the flipped boxes over 300
     random box sets (`scratchpad/check_fpn_flip.py`) — so no boxes-on-disk are
     needed (the FPN record stores none). Off by default; A/B under its own tag.

   What's still open (the doc's original "then …"): a box-aware **affine**
   (scale/translate) that re-encodes the target — the real build, and the one that
   changes object *scale* (the axis VisDrone's 2–5px objects live on). **Mosaic is
   deliberately deferred**: 4-into-1 halves every object, pushing an already-tiny
   distribution further into sub-P3 territory — likely counterproductive here
   (unlike Pets, where it fixed a central-marginal collapse). Worth ~1.7× was the
   twin study's number for the full pack; measure HSV+flip alone first. No proofs
   needed — plain host/codegen.
3. **Re-test the levers, but expect little.** T1b class-weight (on), focal,
   prior-bias, tower were all "refuted" on scrambled data ⇒ untested on valid
   data. BUT BraTS just showed the clever loss fix (weighted CE) was *mildly
   worse* than plain CE once the data was fixed. Strong prior the same holds
   here. Honest minimum: a plain-loss control vs the current arm on the longer
   schedule. Don't invest ahead of that result.
4. **Backbone: keep R34 for now, swap later.** It is the known-good, non-issue
   variable (forward matches the twin, 72% top-1, self-hosted). Options when the
   pipeline is solid (§8).

---

## 7. What is already ruled out (don't re-run without new reason)

All measured this session (some on scrambled data — those are noted void):
- **Capacity is not the constraint** — `FPN_TOWER=4` (+7.08M) didn't even lower
  the *train* loss. Off by default.
- **The four levers (T1a loss-balance, T1b class-weight, T2 prior-bias, T2a
  tower) were measured on SCRAMBLED data** ⇒ their refutations are void, but the
  BraTS analogy says re-testing them is low-value. `yolo_scoring.md` /
  `yolo_assignment.md` hold the (now void) analysis.
- **Backbone is not the bottleneck** — recall ties the twin; "backbone not
  learning" refuted (weights update). Inferred, not proven; the check is
  recall-vs-twin each epoch of the longer run.
- **Aspect-squash / letterbox is refuted** — squash beats letterbox 0.1399 vs
  0.1140 (pixel budget dominates distortion).

---

## 8. Backbone roadmap (the "swap later" options)

Measured/estimated this session. Backbone is downstream of getting the training
right (§6.1); do not swap on an undertrained detector.

- **R34 (current)** — 72% top-1, 21M, self-hosted, forward-verified. Keep.
- **MnV2** — available (`mnv2_imagenet_bf16.bin`, 68.3% top-1, 3.5M). ~6× lighter
  (edge win) but ~4pt worse top-1 AND thin P3 features (~24–32 ch vs R34's 128) —
  the exact axis VisDrone's 2–5px objects need. Likely *lower* mAP, bought for
  latency. Best use: a **second deployable** to plot mAP-vs-latency on the device.
- **MnV4-Conv-M** — **ready to train** (`MainMobilenetV4Imagenet.lean`, faithful
  9.7M, UIB + running-BN wired ✅). The one that wins both axes (paper 79.9% AND
  edge-native, makes the Pi5 CPU path viable). BUT the repo default is the
  **~100-epoch tier**, which underfits (paper needs 500ep). 100ep ≈ R34-level
  (~low-70s) at half the params = still an edge win; **500ep = 79.9% = the
  decisive upgrade** (~2 days rented A100 / ~week local 4-GPU). Cost is
  schedule/compute, not code.
- **R50-RSB-A2** — no checkpoint (needs a full ImageNet run), and *heavier* than
  R34 ⇒ wrong direction for edge. ~+10–25% relative mAP at most, and not the
  bottleneck. Skip unless a well-trained detector proves feature-limited.

Order of operations for any swap: rewire the FPN neck lateral-conv channels + the
bootstrap size for the new backbone's stage widths, retrain the detector, measure
**tiny-object/P3 recall specifically** (aggregate mAP hides a small-object
regression).

---

## 9. Recipes

**Train (12ep, ~4.4h):**
```
IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=1 FPN_TAG=<tag> \
  .lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn > runs/<log> 2>&1
```

**Train WITH the aug pack (HSV + hflip; add `FPN_AUG=1`, own tag):**
```
IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=1 FPN_TAG=aug FPN_AUG=1 \
  .lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn > runs/fpn_aug.log 2>&1
```
Aug is host-side (before the GPU graph), so the train-step vmfb is byte-identical
to the no-aug arm — the `aug`-tagged run recompiles only because the checkpoint
prefix changed. The honest A/B is this vs the same-schedule `FPN_AUG=0` control.

**Score (full-GT sidecar used automatically; `--gt-capped` for A/B vs pre-fix):**
```
FPN_TAG=<tag> FPN_TOWER=0 IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 \
  .lake/build/bin/yolov1-visdrone-fpn infer data/visdrone_fpn runs/<out>
visdrone/.venv/bin/python3 scripts/yolo_map_visdrone.py \
  runs/<out>/logits.bin data/visdrone448/val.bin --fpn data/visdrone --grid 14
```

**Regenerate the full-GT sidecar** (~1s, val only, val.bin byte-identical):
```
visdrone/.venv/bin/python3 preprocess_visdrone.py --size 448 --grid 14 \
  --val-only data/visdrone data/visdrone448
```

**8-image overfit gate** (always run before a full run; fixed arm hits 3.145
total / 0.526 obj at e2000):
```
IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 FPN_TAG=<tag> \
  FPN_EPOCHS=2000 FPN_CKPT_EVERY=500 \
  .lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn_of8
```

**Frozen-param determinism probe** (run FIRST on any "cannot descend" — this is
what caught the shuffle bug in ~2 min; loss MUST be bit-stable):
```
FPN_LR_MULT=0 FPN_EPOCHS=3 FPN_CKPT_EVERY=99 FPN_TAG=<tag> ...
```

**Validate the PyTorch twin** (from `visdrone/`; `pad="lean"` + `pool="lean"`
are required or the forward won't match):
```
.venv/bin/python3 -m bespoke.validate_oracle --ckpt <params.bin> \
  --dump <logits.bin> --data <train.bin> [--bn-stats <bn_stats.bin>]
```

---

## 10. Edge deployment (the part-3 payoff)

The back half is **validated on a Jetson Orin** (memory `edge-deploy-orin`):
stack MLIR → `iree-compile` (cuda sm_87 / llvm-cpu) → `iree-run-module` → exact
output, both backends. Runtime install was one line
(`pip install "iree-base-runtime==<rc>" --find-links https://iree.dev/pip-release-links.html`),
no Jetson source build. Targets: **Orin** GPU (cuda sm_87) or CPU; **Pi5** CPU
(llvm-cpu, aarch64) — MnV4 is what makes the Pi5 path fast enough to matter.

**The one remaining gap** = a standalone deployable artifact. Stack forward MLIR
takes weights as *inputs*, not baked constants; for a self-contained `.vmfb` you
bake the trained weights in as constants (or feed via `--input=@file`). A ~20-line
script closes it. MNIST classifier (`mnist_mlp_fwd.vmfb` + `mnist_mlp_params.bin`
on disk) is the smallest real end-to-end test before the detector.

**Flow to build:** `train checkpoint → bake weights → iree-compile → push →
run on device`, as a repeatable pipeline. Generalizes across models (MNIST,
detector) and devices (Orin, Pi5).

---

## 11. Decisions that are yours

1. **Is VisDrone the right demo target?** 70 obj/img at 2–5px is punishing;
   RetinaNet-class expectation is ~0.15–0.25, not YOLOv8's 0.38. An easier
   dataset (Pets det, COCO subset) would land "verified stack trains a real
   detector" sooner, with VisDrone as the stretch.
2. **From-scratch scope.** Current = ImageNet-backbone + head-from-random (the
   standard, honest "from scratch" for a detector, fully self-hosted). Fully
   random-everything or COCO-detection-pretrain are bigger asks; COCO needs a
   streaming loader (150 GB preprocessed) our stack lacks.
3. **How far to push the number vs move on.** 0.1386 → ~0.20 is longer-run +
   aug (weeks cheaper than architecture). The edge deploy is the differentiator,
   not the mAP.

---

## 12b. In flight / next (2026-08-28)

**Three arms running**, one per card, all on the rebuilt `data/visdrone_fpn`, all
R34, all under distinct `FPN_TAG`s so no two can share a checkpoint prefix:

| arm | epochs | aug | what it answers |
|---|---|---|---|
| `ctrl12` | 12 | off | matched control vs the 0.1386 baseline — quantifies the backbone confound |
| `long50` | 50 | off | "does training longer help", which `long30` never actually answered |
| `aug50` | 50 | on | the `FPN_AUG` A/B committed in `9280a3d` and never run at scale |

⚠ `ctrl12` and `long50` have **different cosine schedules**, so `long50` at its
epoch 12 is NOT a control for `ctrl12`. Each gets a complete annealed schedule,
which is the right design, but it means the only valid comparison against 0.1386
is `ctrl12`'s final. Mid-flight the two legitimately diverge — at e5, `ctrl12`
scored 0.1286 while `long50` scored 0.1114, purely because `ctrl12` had begun
annealing and `long50` had not.

**First measured result, and the rebuild is sound.** `ctrl12` @ e5 of 12:
mAP@0.5 **0.1286**, recall 0.655, class-agnostic AP 0.349, against all 38,759
uncapped GT boxes — i.e. 93% of the old *final* number at under half the schedule.
Per-class it is car 0.550 carrying the mean, against bicycle 0.006 and
awning-tricycle 0.025: the headline is a story about nine weak classes, not a
uniformly mediocre detector.

**Scoring is one command**, and it handles the mid-run case. Scoring a live arm's
checkpoint under its own tag would race the trainer's writes, so the helper copies
the checkpoint to a `<tag>ev` name and copies the eval graph across with its module
renamed (the tag appears only in a comment and the module header). Note the eval
graph is written by *training*, not by `infer` — pointing `infer` at a fresh tag
fails with "no eval graph; train first".

**Next, in order:**
1. Score `ctrl12` final against 0.1386. That settles whether the rebuild
   reproduces, and how much of any delta is the better R34 checkpoint.
2. Score `long50` at e10/20/30/40/50 for the plateau curve. `long30` plateaued in
   val mAP at e10 while train loss halved through e30 — this is the re-test.
3. Score `aug50` against `long50` at matched epochs. Same schedule, same seed,
   aug the only difference.
4. R50 with the conv-only bootstrap, 50 epochs, matched to `long50`. It is behind
   early (epoch-1 666.8 vs R34's 354.7) because BN is reset and must re-learn;
   whether its stronger features overtake that is the question. Give it the box
   rather than adding a fifth trainer — the host loader is the bottleneck.
5. Only then revisit `coco_visdrone_two_stage.md`, whose data (94 GB) is also gone
   and whose own §10 predicts a tie as the most likely outcome.

---

## 12. State on pause / resume checklist (2026-07-24) — ⛔ OBSOLETE, see §12b

**Step 1 below cannot run: the `long30` checkpoints were deleted.** Kept for the
reasoning and the gotchas, which still apply.

Parked to pivot to the R34→BraTS retraining demo. When YOLO resumes, in order:

1. **Re-score `long30` — the ONE blocking unknown.** The tagged checkpoints exist;
   the script bug is fixed. Confirm whether 30 epochs helped or regressed:
   ```
   FPN_TAG=long30 FPN_TOWER=0 IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=1 \
     .lake/build/bin/yolov1-visdrone-fpn infer data/visdrone_fpn runs/long30_rescore_e30
   visdrone/.venv/bin/python3 scripts/yolo_map_visdrone.py \
     runs/long30_rescore_e30/logits.bin data/visdrone448/val.bin --fpn data/visdrone --grid 14
   ```
   (The infer log must show `params : …__long30_params.bin`; if it says
   `…__visdrone__params.bin`, the tag didn't take — the whole point of this fix.)
   Repeat for e5..e25 (copy `…_long30_params_e${EP}.bin` → `…_long30_params.bin`
   first) for the curve. ≥0.1386 ⇒ eval-only bug, keep pushing epochs; ≈0 ⇒
   training regressed, root-cause before anything else (§6.1).

2. **Then the aug A/B** (`9280a3d` is committed, unvalidated at scale). Same
   schedule, `FPN_AUG=1 FPN_TAG=aug` vs `FPN_AUG=0 FPN_TAG=noaug`, full GT. Only
   worth running once #1 gives a trustworthy control.

3. Backlog unchanged: box-aware affine aug (§6.2), backbone swap (§8), standalone
   edge `.vmfb` (§10).

**Committed/changed this session:** aug pack `9280a3d` (4 code files). Uncommitted:
`run_fpn_long.sh` tag fix (untracked helper); this doc (untracked); pre-existing
edits to `post_shuffle_fix.md` / `yolo_{drone,final,fpn}.md` (not this session's).

**The meta-lesson (applies to the BraTS pivot too — see memory
`results-need-guards`):** every recent bug here — shuffle-pairing, capped-GT,
now missing-`FPN_TAG` — was a *silent plumbing* error: the code ran, emitted a
plausible number, and measured the wrong thing. None were math/compute bugs. The
long30 eval was wrong for a full day because nothing asserted "these six rows
can't be identical" or "the loaded checkpoint path must contain the tag." The
R34→BraTS retraining will have the identical risk surface (is the backbone
actually loaded? is eval scoring the right checkpoint? is image↔mask paired?).
**Cheap, fast, CPU-side known-answer guards are worth more than more GPU hours.**
