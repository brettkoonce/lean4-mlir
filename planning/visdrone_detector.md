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

> **▶ STATE 2026-08-29. Paused here; the next session picks up at §13.**
>
> **mAP@0.5 = 0.1961**, from the **0.1386** this thread resumed at — **+41.5%**,
> for one 12-epoch run (~47 min on one RTX 4060 Ti). The recipe is
> `FPN_AUG=1 FPN_CLSW=none FPN_CLSFOCAL=2 FPN_EPOCHS=12`, scored with
> `--multilabel`. Argmax-to-argmax against the PyTorch replica's 0.1532 it is
> **0.1774 (+15.8%)**.
>
> **Deployment is unblocked and measured on real hardware**: Jetson Orin Nano
> 8GB, 25 W, TensorRT fp16 — **35.7 fps end to end**, up from 14.9 (§12b).
>
> ⚠ **Four things that were believed at the last pause and are now false.** Each
> cost a run to disprove and none is obvious from the code:
> 1. *"Aug moves the optimum to ~30 epochs."* No — **12 epochs wins** (0.1798 at
>    e10 of a 12-ep cosine vs 0.1782 at the 30-ep arm's best). The annealing was
>    doing the work, not the length.
> 2. *"The T1b class weights help class spread."* They are **net harmful**; the
>    unweighted arm is best and full inverse-frequency costs 23%.
> 3. *"The class head overfits on longer schedules."* Its aggregate accuracy is
>    **flat** (top-1 67.58 → 67.32); it *redistributes* toward the priors, and
>    mAP is charged for what average CE is indifferent to.
> 4. *"The replica validates the export."* It never did — the ONNX shipped a
>    **different model** for a day. Root cause was one missing `pad="lean"`.
>
> **The tree was rebuilt from nothing on 2026-08-28** — data, checkpoints, the
> backbone file and the twin's venv had all been deleted for disk. It reproduces:
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

A ResNet-34 + FPN anchor detector, trained on VisDrone, **works**. As of
2026-08-28 the best measured arm is **mAP@0.5 = 0.1731** (recall 0.709,
class-agnostic AP 0.435) at 30 epochs with `FPN_AUG=1` — up **25%** on the
0.1386 this thread paused at, and **13% past** the architecture-matched PyTorch
replica (0.1532) that it used to trail by 10%. **Full results, curves and the
resume plan are in **§13**, with the workings in §12b. The paragraphs below
predate all of it and are kept only for the reasoning.**

⚠ The single most important operational finding: **`FPN_AUG=1` is not optional
on any schedule longer than ~12 epochs.** Without it, 50 epochs scores 0.1243,
*worse* than 12 epochs' 0.1526. With it, 0.1674 at 50 and 0.1731 at 30. And both
arms peak then decline, so "run longer" is wrong too — aug moves the optimum from
~15 to ~30 epochs, it does not remove overfitting.

The historical framing below (that the gap to the twin is ranking, not
localization) was written at 0.1386 and is now stale: recall has moved 0.676 →
0.709 and class-agnostic AP 0.376 → 0.435, so localization improved as well. So
is "the detector is undertrained" — it is now measurably *over*trained past ~30
epochs without aug.

<!-- historical, pre-2026-08-28 -->
The detector is **undertrained** (e12 loss still descending); the
0.1386 arm ran with **no augmentation**, though an online HSV + horizontal-flip
pack now exists as an opt-in A/B arm (`FPN_AUG=1`, §6.2). The whole stack is self-hosted: the R34 backbone
was trained *by this stack* on ImageNet (72% top-1), the head trains on VisDrone,
no borrowed weights. Part-3 goal = a practical, self-hosted detector that trains
from free data and **deploys to an edge device** (Orin path already validated).
Verification/proofs are NOT required here — codegen + FD only.

**One thing gates every external number:** score against the full-GT sidecar
(§5), not the 56-box-capped tail. `0.1386` was the real number at the time and
`0.1167` was the
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

## 5. Results as of 2026-07 — ⛔ SUPERSEDED by §12b/§13, kept for the twin analysis

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

   - **Box-aware affine** (scale + translate) — **BUILT 2026-08-28, not yet A/B'd.**
     `F32.fpnAffine`, `FPN_AFFINE=<percent>` (plus `_SCALE`, `_TRANSLATE`), off by
     default. Unlike the flip this is not shape-invariant — scaling changes
     `max(w,h)`, so a GT moves between FPN levels and between anchors — so the
     target is rebuilt from boxes decoded out of the assigned slots (every slot is
     an exact encoding of one; `fpn_decode_boxes` inverts `encode_targets_fpn`
     term for term). Gated by `scripts/check_fpn_affine.py`, which **compiles
     `ffi/f32_helpers.c` itself** and checks it against
     `preprocess_visdrone.encode_targets_fpn` **itself** — no re-implementation on
     either side, which is exactly the mistake that shipped a wrong ONNX. Identity
     is byte-identical on 12 real records; the re-encode is exact (max|d| = 0.0)
     over 48 random draws.

     ⚠ **The knob costs are measured, and they invert the obvious prior**
     (`scripts/fpn_affine_knob_cost.py`, 16 val records, GT surviving the transform):

     | setting | survive | survivors under 2 px |
     |---|---|---|
     | control | 100% | 2.8% |
     | translate ±0.05 | 93.8% | 2.4% |
     | translate ±0.10 | 92.3% | 2.4% |
     | translate ±0.20 | 89.6% | 2.3% |
     | scale ±0.10 | 93.6% | 2.7% |
     | scale ±0.25 | 90.0% | 3.4% |
     | **scale ±0.50** (Ultralytics default) | **82.7%** | **5.3%** |
     | scale ±0.25 + translate ±0.10 (our default) | 89.2% | 3.1% |

     **Translate is the expensive knob, not scale** — ±0.05 alone costs 6% of all
     GT, and the cost then saturates. That is a property of this data (≈70
     objects/image, spread to the edges) *and* of augmenting the 448-px RECORD
     rather than the source image: content that leaves the frame is replaced by
     the mean fill, so boxes lost at one edge are not repaid at the other.
     **Scale ±0.50 is confirmed wrong here**: 17% of GT gone and the sub-2px share
     nearly doubled — the Ultralytics setting really is calibrated for a dataset
     with the opposite size distribution.

   **Mosaic is deliberately deferred**: 4-into-1 halves every object, pushing an
   already-tiny distribution further into sub-P3 territory — likely
   counterproductive here (unlike Pets, where it fixed a central-marginal
   collapse). Worth ~1.7× was the twin study's number for the full pack. No proofs
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

## 12b. Results ledger (2026-08-28 → 29) — the workings behind §0 and §13

### The four arms, all finished and scored

Identical data, seed and spec; only the schedule and `FPN_AUG` differ. Scored
against all 38,759 uncapped val GT boxes.

| arm | epochs | aug | train loss | mAP@0.5 | recall | ca-AP |
|---|---|---|---|---|---|---|
| baseline on record (pre-pause) | 12 | off | — | 0.1386 | 0.676 | 0.376 |
| `ctrl12` | 12 | off | 112.7 | 0.1526 | 0.682 | 0.393 |
| `long50` | 50 | off | 47.9 | 0.1243 | 0.669 | 0.369 |
| `aug50` final | 50 | on | 88.5 | 0.1674 | 0.703 | 0.429 |
| **`aug50` @ e30 (best seen)** | 30 | on | — | **0.1731** | **0.709** | **0.435** |
| PyTorch replica, same architecture | 12 | off | — | 0.1532 | 0.677 | 0.400 |

### The finding: augmentation moves the optimum, it does not license "longer"

| epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| `long50` no aug | 0.1114 | 0.1223 | **0.1320** | 0.1313 | 0.1315 | 0.1283 | 0.1261 | 0.1268 | 0.1244 | 0.1243 |
| `aug50` +aug | 0.1207 | — | — | 0.1703 | — | **0.1731** | — | 0.1694 | — | 0.1674 |

- **Longer WITHOUT aug costs 19%.** 4× the schedule, less than half the train
  loss, detections 437k→463k while recall falls. Overfitting on 6,471 images.
- **Aug inverts it**: −19% becomes +10% over the best short run.
- ⚠ **But both arms peak and then decline.** Aug does not remove overfitting, it
  delays and softens it: the optimum moves ~15 → ~30 epochs and rises 31%.
  **Do not run past ~30 with this aug pack.**
- 8/10 classes improve under aug, hardest where the detector was weakest
  (bus 0.165→0.227, truck 0.095→0.108). Car 0.573→0.605 no longer carries the mean.

Retroactively vindicates two discarded things: §6.1's `long30` "plateaus at e10"
observation (voided for being measured in the shuffle-bug era — the plateau was
real), and `coco_visdrone_two_stage.md` §4's untested prediction that "the
schedule decision is really an aug decision".

### Next, in order — ⛔ SUPERSEDED, see §13

Items 1 and 2 (30 epochs annealed; aug at 12) were run and **both are answered**:
12 epochs wins and longer loses. Kept only so the reasoning is legible. The live
list is §13.

### ⭐ Class focal (T1c) — helps, but NOT the way it was predicted to (2026-08-29)

`FPN_CLSFOCAL=γ` on top of the best config (aug, 12 ep, `FPN_CLSW=none`):

| γ | decode | mAP@0.5 | ca-AP | recall |
|---|---|---|---|---|
| 0 | argmax | **0.1774** | 0.4285 | 0.6993 |
| 1 | argmax | 0.1767 | 0.4322 | 0.7045 |
| 2 | argmax | 0.1762 | 0.4409 | 0.7075 |
| 0 | multilabel | 0.1909 | 0.4317 | 0.7430 |
| 1 | multilabel | 0.1952 | 0.4349 | 0.7489 |
| **2** | **multilabel** | **0.1961** | **0.4414** | 0.7485 |

⚠ **The prediction was wrong and the measurement is worth more than the win.**
Focal was chosen to recover rare-class AP where static weights failed. It does
not do that: awning-tricycle top-1 is 13.4% at γ=0 and 13.4% at γ=2, tricycle
32.2 → 33.8, and per-class mAP under the argmax decode is FLAT to slightly down
(0.1774 → 0.1762). The class head's discrimination is unchanged — top-1
70.50 → 70.07, top-3 93.74 → 93.77.

**What actually improved is score CALIBRATION.** Class-agnostic AP rises
monotonically with γ (0.4285 → 0.4322 → 0.4409) and so does recall
(0.6993 → 0.7075), while class accuracy sits still. Down-weighting easy examples
stops the model driving p_t to 1 on the cells it already has, which compresses
the confident end of `obj·p_c` and spreads the uncertain end — a better RANKING
of the same decisions.

**That is why focal and multilabel are complementary rather than redundant.**
Focal improves the probabilities; multilabel is the decode that reads all of them
instead of only the argmax. Neither alone gets there — focal alone under argmax
is flat (0.1762), multilabel alone is 0.1909, together 0.1961.

✅ **Focal does avoid the failure that killed static weighting.** Detection counts
stay sane: tricycle 7,419 → 8,597 across the γ ladder, against the static-weight
ladder's 7,419 → **31,322**. The predicted mechanism — a dynamic weight that
tracks p_t cannot buy recall with a permanent precision tax — held exactly. It
just does not buy rare-class AP either.

### ⛔ The class weights are NET HARMFUL — turn them off (2026-08-29)

`fpnClsWeights` (T1b, sqrt-inverse frequency) has been on since the arm was
built, justified by "better class spread". A three-rung ladder at 12 epochs with
aug, everything else identical (`FPN_CLSW=none|sqrt|inv`):

| weights | range | mAP (argmax) | mAP (multilabel) | head top-1 |
|---|---|---|---|---|
| **none** | — | **0.1774** | **0.1909** | **70.50%** |
| sqrt (current default) | 6.7× | 0.1771 | 0.1869 | 67.58% |
| inv | 44.7× | 0.1368 | 0.1469 | 58.60% |

**Monotonic, and in the wrong direction.** More weighting ⇒ worse head accuracy
⇒ worse mAP. The current default buys nothing on the argmax decode (0.1771 vs
0.1774) and costs 2% on the better one. Full inverse frequency costs 23%.

⚠ **And the weights DO work as designed — that is what makes this instructive.**
Rare-class top-1 rises monotonically with weighting: tricycle 32.2 → 44.8 →
52.5, awning-tricycle 13.4 → 29.9 → 30.6, van 22.5 → 43.6. It is *AP* that does
not follow, because the weighted models also hand rare labels to a flood of cells
that are not that class: tricycle detections 7,419 → 17,225 → **31,322** against
1,045 GT. Weighting buys recall by spending precision, and AP is precision-
sensitive. **Accuracy on positives is the wrong proxy for AP** — that is the
whole lesson, and it is why the "better class spread" justification survived
this long without being wrong exactly, just irrelevant.

Retroactively this is the BraTS result again (`brats_demo.md`: on fixed data
plain CE beat every weighted arm). Two datasets, same answer: **the loss-weighting
lever does not beat plain training.** §7's "expect little from the levers" was
right and can now be closed rather than re-litigated.

**Default NOT changed here** — the spec name embeds `wcls` and every existing
checkpoint prefix depends on it, so flipping it silently would orphan the whole
checkpoint set. `FPN_CLSW=none` is one flag. Renaming the arm is a deliberate
call, not a side effect.

### ⭐ Multilabel decode is worth +7.6% for free (2026-08-29)

`scripts/yolo_map_visdrone.py --multilabel` emits one detection per (cell, class)
for the top-3 classes instead of the argmax alone. Under argmax a cell whose GT
class loses the argmax is charged **twice** — a false negative for the true class
and a false positive for the winner — and per-class AP is an unweighted mean, so
that lands on the rare classes. COCO protocol does not evaluate this way; argmax-
only was the non-standard, self-handicapping choice.

Unweighted arm: **0.1774 → 0.1909**, recall 0.699 → 0.743. No retraining.

⚠ **Do not mix decodes in a comparison.** The PyTorch replica's 0.1532 and every
number recorded before today are argmax. Replica-to-Lean, like for like, is
0.1532 vs **0.1774 = +15.8%**.

### The current best arm

**aug, 12 epochs, `FPN_CLSW=none`, `FPN_CLSFOCAL=2`, multilabel decode
= mAP@0.5 0.1961.** +41.5% on the 0.1386 this thread resumed at. Argmax-to-argmax
against the PyTorch replica's 0.1532 it is 0.1774, +15.8%. Cost is one 12-epoch
run (~47 min on one 4060 Ti).

### ✅ Deployment: the correctness gate is CLEARED (fixed 2026-08-28)

The Orin path works and is fast — **229 fps** forward under TensorRT fp16 (4.4 ms;
`trtexec` GPU compute 4.16 ms), fp16 worth 2.15× over fp32 with identical
detections. It shipped the wrong model; that is now fixed and gated.

- ✅ **The mismatch was `pad="lean"`, omitted in `export_onnx.py`.** The replica ran
  torchvision's SYMMETRIC conv padding against a Lean spec emitting
  `MlirCodegen.samePad` — TF-style ASYMMETRIC SAME. Shapes, parameter counts and
  `iree-compile` are identical either way, so nothing structural could catch it,
  but the sampling grid shifts half an output pixel per stride-2 conv and
  **compounds through the 3/4/5 downsamples feeding C3/C4/C5** — exactly why the
  objectness correlation fell 0.90/0.80/**0.62** with tap depth instead of being
  uniform. With `pad="lean"`: **1.0000 at all three scales**, max relative
  difference 5.0e-4, and the same **238 detections at top 0.7108** against the Lean
  stack's 238 / 0.7109. Both of the device's original figures (279 / 0.7656) were
  reproduced on the training box first, so the diagnosis is confirmed, not fitted.
- ⚠ **Every "ruled out" was empty for the same reason.** The BN-layout, BN-walk,
  BN-eps, `pool`, and channel-grouping probes all built the replica without `pad=`,
  so each compared two wrong models. Note `grad_dump.py`, `validate_oracle.py` and
  `layout_hunt.py` all take `--pad` and default it to `lean`; the only two places
  that hardcoded it away were the one that shipped (`export_onnx.py`) and the one
  cited as the validation (`bespoke/diff_lean.py`).
- ⚠⚠ **The validation gap was real and is also fixed.** `bespoke/diff_lean.py` never
  compared elementwise — it printed a scalar loss beside *hardcoded* pre-shuffle-fix
  Lean numbers. It now takes `--lean-logits <infer dump>` and compares logit for
  logit, plus `--bn-stats` so `--eval-mode` means something. Verified over 64 val
  records: 1.4e-3 relative, objectness r = 1.0000. This never invalidated the
  replica's 0.1532 as an architecture-matched yardstick (independently trained,
  fair comparison) — only its use as an elementwise stand-in, which export demands.
- ⚠ **Tolerances are RELATIVE, and this matters.** Logit magnitude varies by two
  orders of magnitude across records: the reference frame spans ±16, val record 40
  spans −976 .. +1287. An absolute threshold is flaky or vacuous depending on which
  record it was tuned on. Relative, the regimes are stable and 3 decades apart —
  correct 5e-4, broken ~1.0.
- **The 5e-4 residual is TF32, measured not assumed.** The Lean render carries no
  `precision_config`, so XLA uses TF32 on Ada. Re-running the identical graph under
  `NVIDIA_TF32_OVERRIDE=0` moves the residual to 1.6e-4, and the Lean graph
  disagrees with *itself* by 7.8e-3 absolute across that switch.
- ⚠ **CPU decode is the real bottleneck: 57 ms, 6× the network.** 43 ms of that is
  pure-Python per-class NMS over 300 boxes. Vectorizing it, or an EfficientNMS
  plugin in the engine, is worth more than any architecture change.
- ⚠ **Before anyone tries int8:** those ±1287 logits are CLASS channels on
  background cells. The class head is a softmax masked to positives, so background
  is never trained and those logits are unconstrained; the decode discards the cells
  on objectness anyway. Harmless in fp32/fp16 (max 65504), fatal to an int8
  calibration.
- ⚠ IREE on the Orin is **0.5 fps** — correct, just slow, because its CUDA backend
  generates its own conv kernels. **Never project throughput across compilers**:
  scaling the 65 fps XLA number by fp32 TFLOPS predicted 16 fps, wrong by 32× under
  IREE and by 14× the other way under TensorRT.
- **Still open:** the 229 fps was measured on the wrong graph. Padding should not
  move throughput, but it has not been re-measured on the device.

See `deploy/ORIN_SMOKE_TEST.md` and `runs/2026-08-28-visdrone-fpn-rebuild/`.


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


---

## 13. NEXT SESSION STARTS HERE (paused 2026-08-29)

### The one-line state

`FPN_AUG=1 FPN_CLSW=none FPN_CLSFOCAL=2 FPN_EPOCHS=12` → **mAP@0.5 0.1961**
under `--multilabel`, 0.1774 under argmax. Deployed at 35.7 fps on an Orin Nano.

### The full ladder, every number from this thread

| arm | ep | aug | clsw | γ_cls | decode | mAP | ca-AP | recall |
|---|---|---|---|---|---|---|---|---|
| baseline on record | 12 | off | sqrt | 0 | argmax | 0.1386 | 0.376 | 0.676 |
| PyTorch replica | 12 | off | — | — | argmax | 0.1532 | 0.400 | 0.677 |
| `ctrl12` | 12 | off | sqrt | 0 | argmax | 0.1526 | 0.393 | 0.682 |
| `long50` | 50 | off | sqrt | 0 | argmax | 0.1243 | 0.369 | 0.669 |
| `aug50` | 50 | on | sqrt | 0 | argmax | 0.1674 | 0.429 | 0.703 |
| `aug30` best (e20) | 30 | on | sqrt | 0 | argmax | 0.1782 | — | — |
| `aug12` best (e10) | 12 | on | sqrt | 0 | argmax | 0.1798 | 0.415 | 0.699 |
| `clswinv` | 12 | on | inv | 0 | argmax | 0.1368 | 0.346 | 0.696 |
| `clswnone` | 12 | on | none | 0 | argmax | 0.1774 | 0.429 | 0.699 |
| `cfoc2` | 12 | on | none | 2 | argmax | 0.1762 | 0.441 | 0.708 |
| `clswnone` | 12 | on | none | 0 | **multilabel** | 0.1909 | 0.432 | 0.743 |
| **`cfoc2`** | 12 | on | none | 2 | **multilabel** | **0.1961** | 0.441 | 0.749 |

⚠ Never mix decodes in a comparison. Everything before 2026-08-29 is argmax.

### ⛔ The wall: rare classes, and it is probably NOT the loss

Three separate levers have now failed to move rare-class discrimination, and the
head sits at ~13% top-1 on awning-tricycle and ~19% on bicycle under all of them:

| lever | rare-class top-1 | verdict |
|---|---|---|
| sqrt-inverse weights | rises (awning-tri 13.4 → 29.9) | but AP falls — FP flood |
| full-inverse weights | rises more (→ 30.6) | mAP −23%, worse still |
| class focal γ=2 | **does not move** (13.4 → 13.4) | helps calibration instead |

Static weights raise rare-class *recall* by flooding those classes with false
positives (tricycle detections 7,419 → **31,322** against 1,045 GT) and AP is
precision-sensitive. Focal avoids that flood (7,419 → 8,597) but does not improve
discrimination at all. **Stop treating this as a loss-function problem.** With
291 and 367 val instances against car's 6,345, the next honest hypothesis is
data, not objective.

### What to run first, in order

1. ⭐ **The box-aware affine A/B — BUILT, GATED, NEVER RUN.** It is the only
   untried lever that adds information rather than reweighting what is there, and
   it is the one that moves object SCALE, which is the axis this dataset lives on.
   `FPN_AUG=1 FPN_CLSW=none FPN_CLSFOCAL=2 FPN_AFFINE=50 FPN_TAG=aff50` against
   the same config without `FPN_AFFINE` (which is `cfoc2`, already on the board at
   0.1961 — so the control is free). Start at the measured default (scale ±0.25,
   translate ±0.10, ~11% GT cost); ⚠ **not** Ultralytics' ±0.50, which costs 17%
   of GT here and nearly doubles the sub-2px share
   (`scripts/fpn_affine_knob_cost.py`).
2. **Mosaic**, deferred all thread on the grounds that 4-into-1 halves
   already-tiny objects. Worth re-testing only if (1) shows scale aug helps.
3. **`--ml-k` / `--ml-floor` sweep.** Multilabel is worth +7.6% at the default
   top-3 / p≥0.05 and has never been tuned. Free — it is decode-only, no retrain.
4. **The backbone** (§8), last. Do not swap on a recipe still being tuned.

### Deployment: what is done and what is not

✅ Export correctness gated (`export_onnx.py --verify-frame`, and
`bespoke/diff_lean.py --lean-logits` for an elementwise check over many records).
✅ Decode vectorized: 49.1 → 10.0 ms on device, gated against the old
implementation by `orin_detect.py --gate-decode`.
✅ Measured: Orin Nano 8GB / 25 W / TensorRT fp16 / JetPack 6.2 / TRT 10.3 —
preprocess 11.9, forward 6.3, decode 10.0, **total 28.0 ms = 35.7 fps**.

⛔ **Not yet run on device:** `--fold-preprocess u8`. It is built and passes the
frame gate at an identical 4.959e-04, and should take preprocess from 11.9 ms to
near zero (input becomes `[1,448,448,3]` UINT8 — what `np.asarray(pil)` already
returns — so the host does no arithmetic and the H2D copy drops 4×). Needs a
TensorRT that accepts a UINT8 network input; fall back to `f32` if not. **This is
the largest remaining easy win on the device: ~28 → ~17 ms, ~59 fps.**

### ⚠ Environment traps that will cost an hour each

- **`scripts/fpn_loss_probe_check.py` needs IREE, which is NOT in the repo
  `.venv` and must never be** — that venv is the pinned JAX/cuDNN environment and
  installing into it breaks every bf16 conv. Use a throwaway venv with a MATCHED
  `iree-base-compiler`/`iree-base-runtime` pair and pass `IREE_COMPILE=`; the
  recipe is in that script's header.
- **`deploy/export_onnx.py` needs torch**, likewise never in the pinned `.venv`
  (torch drags its own CUDA wheels over the pinned cuDNN). Throwaway CPU-only
  venv; recipe in that script's header.
- **Scoring a checkpoint from a RUNNING arm** must go through the `<tag>ev`
  copy-and-rename dance, or the live trainer's `_params.bin` gets clobbered.
- **`FPN_TAG` is not optional on `infer`.** An untagged eval silently scores a
  different arm; the only tell is identical rows across an epoch sweep. This
  voided a full day once.

### Everything added this session

| thing | where |
|---|---|
| box-aware affine | `ffi/f32_helpers.c`, `F32.fpnAffine`, `FPN_AFFINE` |
| its gate | `scripts/check_fpn_affine.py` (compiles the real C, refs the real encoder) |
| its knob costs | `scripts/fpn_affine_knob_cost.py` |
| class focal | `emitAnchorYoloLoss`, `yoloClsFocalGamma`, `FPN_CLSFOCAL` |
| its FD gate | `scripts/fpn_loss_probe_check.py` (2 new arms) |
| class-weight ladder | `FPN_CLSW=none\|sqrt\|inv` |
| multilabel decode | `yolo_map_visdrone.py --multilabel --ml-k --ml-floor` |
| the padding fix | `deploy/export_onnx.py` (`pad="lean"`) |
| fast decode | `deploy/orin_detect.py` (`decode_reference` is the oracle) |
| preprocess folding | `export_onnx.py --fold-preprocess {none,f32,u8}` |
