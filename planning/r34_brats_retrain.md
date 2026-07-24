# r34_brats_retrain.md — the self-hosted R34 base, transferred to BraTS segmentation

**Written 2026-07-24.** Scoping doc for the pivot off the VisDrone detector
(`planning/visdrone_detector.md`, now paused). Companions: memory
`brats-demo-thread` (the from-scratch UNet result), `results-need-guards` (the
verification discipline this doc bakes in from day one), `edge-deploy-orin`,
`rocm-is-the-transformer-box` / `cross-box-throughput-anchors` (where to run it).

---

## 0. The thesis — why this is the high-ROI move

The stack already trained **one ResNet-34 on ImageNet by itself** (72% top-1,
`.lake/build/jax_r34_imagenet.bin`, 21,284,672 floats, no borrowed weights). The
VisDrone detector demo transfers that base to **object detection** (R34 → FPN
head). This demo transfers the *same base* to **medical image segmentation** (R34
→ UNet decoder → 4-class brain-tumour masks). That is the story: **one
self-hosted backbone, two unrelated real downstream tasks, no external weights
anywhere.** It's a stronger claim than either demo alone, and it's mostly
assembly of proven parts — the reason it's higher-ROI than pushing the detector's
mAP another few points.

"Retraining" = start the encoder from the ImageNet R34 weights (bootstrap) and
retrain end-to-end on BraTS, rather than the current from-scratch UNet.

---

## 1. What already exists (don't rebuild)

- **A from-scratch BraTS UNet that WORKS.** `demos/MainUnetBratsTrain.lean` /
  `unetBrats`: 4-modality 240×240 input → 4 `unetDown` → convBn bottleneck → 4
  `unetUp` → 4-class head. Post the July shuffle-bug fix, **plain CE segments**
  (mIoU ~0.69, WT/TC/ET Dice 0.875/0.813/0.837 — memory `brats-demo-thread`). This
  is the **baseline to beat** and the control: R34-transfer has to clear it to
  justify itself.
- **The bootstrap mechanism, proven on the detector.** `TrainConfig.bootstrapBackbone
  : Option (String × Nat)` → `patchInitWithPretrainedPrefix` (SpecHelpers.lean:646)
  overwrites the first `prefixFloats` of the He-init with the pretrained
  checkpoint, and Train.lean:671 does the companion BN-stats bootstrap. The
  detector uses `some (".lake/build/jax_r34_imagenet.bin", 21284672)` verbatim.
  **Key limitation: it patches a PREFIX only** (`head ++ tail`), so the
  bootstrappable params must be contiguous at the FRONT of the layout — this
  drives §3.
- **The R34 encoder, as a layer list** (from the detector spec): `convBn 3 64 7 2
  .same` (stem, /2) → `maxPool 2 2` (/4) → `residualBlock 64 64 3 1` (/4, 64ch) →
  `residualBlock 64 128 4 2` (/8, 128ch = C3) → `residualBlock 128 256 6 2` (/16,
  256ch = C4) → `residualBlock 256 512 3 2` (/32, 512ch = C5). Total stride /32.
- **UNet decoder layers.** `.unetUp ic oc` (transposed-conv upsample + skip concat)
  and `.bilinearUpsample scale` already exist and are used by `unetBrats`. The
  detector's FPN neck already taps R34's C3/C4/C5 for skips, so tapping the R34
  feature pyramid for a decoder is a solved pattern.
- **Data + loader.** `preprocess_brats.py`, `F32.loadBrats (path) (imgSize)`,
  `bratsIO` (4×240×240 image + 240×240 uint8 mask, `DatasetKind.brats`), the
  `.perPixelCE` seg training path, `F32.segHflipPair` mask-aware aug, `segConfusion`
  / `segRegions` (WT/TC/ET) eval. All reusable unchanged.

---

## 2. Architecture — R34 encoder + UNet decoder

A U-shape with the R34 stages as the contracting path and a symmetric decoder:

```
input → [R34: stem/pool → stage1 → stage2(C3) → stage3(C4) → stage4(C5)]
                 |skip64    |skip64   |skip128     |skip256      ↓ bottleneck 512
        decoder: up → cat C4 → up → cat C3 → up → cat stage1 → up → up → 4-class head
```

The decoder mirrors the current `unetBrats` decoder (it already goes 512→256→128
→64→32→4), just fed by R34 tap points instead of `unetDown` outputs, and it needs
**5 upsamples** to undo R34's /32 (the from-scratch UNet only had 4 — one extra
stage, because R34's stem downsamples before its first residual stage). Skip
channel counts come from the R34 taps (64/64/128/256), not the decoder's own.

**New codegen surface:** likely a `.resnetUnetDecoder`-style layer (or reuse
`.unetUp` with explicit skip-source wiring), analogous to how `.fpnDetect` taps
C3/C4/C5. Estimate this is the main build — the encoder, decoder primitives,
bootstrap, loss, and eval all exist; the wiring of R34 taps → decoder skips is the
new part. FD-verify the new backward seam (the tap-gradient join) at tiny scale
before any GPU run — this is the exact class of seam that hid the detector's
`fpnTapGrad` bug.

---

## 3. The three real design decisions (with recommendations)

### 3a. Channel mismatch: R34 stem is 3→64, BraTS is 4 modalities
The prefix-only bootstrap (§1) means the R34 stem must be the first layer AND must
match the pretrained 3-channel stem to bootstrap it. Options:
- **(v0, RECOMMENDED) Use 3 of the 4 modalities → 3-channel input.** The ENTIRE
  R34 bootstraps as the existing prefix, **zero new bootstrap code**. Pick the 3
  most informative (candidate: FLAIR + T1gd + T2w — FLAIR defines edema, T1gd
  defines enhancing tumour; drop T1w, the least tumour-specific). Fastest path to a
  working, honest transfer demo. Cost: drops one modality (may lower ceiling).
- **(v1) Fresh 4→64 stem, bootstrap R34-minus-stem.** Uses all 4 modalities (the
  faithful medical setup), but the fresh stem sits at the FRONT, so the
  bootstrappable weights are no longer a prefix → **needs an offset-aware
  bootstrap** (patch a byte RANGE, or a per-layer bootstrap map — a small,
  well-scoped extension to `patchInitWithPretrainedPrefix`). The stem is only
  ~12.5k params and MRI≠RGB anyway, so re-learning it is defensible.
- **(rejected) 4→3 adapter conv before a frozen 3-ch stem** — puts an
  un-bootstrapped layer at the front (same prefix problem as v1) with none of v1's
  faithfulness. Skip.

**Do v0 first** (validation ladder: get a number on the board), then v1 for the
full-modality claim.

### 3b. Resolution / stride alignment: 240 doesn't divide by /32
R34's /32 stride wants an input divisible by 32 for clean feature maps and skip
concatenation. 240/32 = 7.5 (not integer) → the last stage needs ceil-padding and
the decoder skips won't line up cleanly. The from-scratch UNet used 240 native
precisely because its 4 halvings divide it (240 = 16×15); R34's 5 downsamples
don't. **RECOMMENDED: resize BraTS to 224** (bilinear image, nearest-neighbour
mask) — 224/32 = 7 clean, AND 224 is R34's ImageNet pretraining resolution, so the
backbone sees its native scale (least domain shift). 256 also works (256/32=8) if
the ~7% downsample of 240→224 loses too much small-lesion detail; measure. The
loader already takes an `imgSize` arg.

### 3c. Bootstrap = warm-start, not freeze
Retrain end-to-end (encoder unfrozen) — the ImageNet features are a starting point,
not fixed. This matches the detector (which trains through the backbone). A
frozen-encoder + train-decoder-only arm is a cheap ablation worth running to
measure how much the backbone actually adapts, but the headline demo is full
fine-tuning.

---

## 4. Plan — validation ladder

1. **v0 build (3-modality, 224, whole-R34 prefix bootstrap).** Wire R34 taps →
   UNet decoder. FD-verify the new tap-gradient backward seam at tiny scale (CPU,
   before any GPU). This gates everything.
2. **Overfit gate (the fast known-answer smoke).** Overfit 8–16 slices to near-zero
   loss with a hard threshold. Catches a broken train/tap seam in minutes, not a
   40-min epoch. Bakes in `results-need-guards` — the detector's `long30` eval was
   void for a day for lack of exactly this discipline.
3. **Full run vs the from-scratch control.** Same schedule, R34-bootstrap vs
   He-init `unetBrats`, plain CE (the post-shuffle-fix baseline — do NOT reopen the
   weighted-CE/focal chapter; `brats-demo-thread` closed it). Score WT/TC/ET Dice +
   mIoU. **Success = clears the ~0.69 mIoU baseline** (and ideally converges faster
   / from fewer epochs — the transfer win is usually sample-efficiency, not just
   peak).
4. **v1 (4-modality fresh stem)** only if v0 clears the bar and the dropped modality
   is measurably costing Dice. Needs the offset-aware bootstrap (§3a).
5. **Edge deploy** reuses `edge-deploy-orin` unchanged (same forward→iree-compile→
   device flow); a 4-class MRI segmenter on an Orin is a clean part-3 payoff.

---

## 5. Guards to build FROM THE START (results-need-guards)

Every recent bug here was silent plumbing, not math. This demo's specific risk
surface — assert each, cheaply, before trusting any Dice number:
- **Backbone actually loaded?** After bootstrap, assert the first `prefixFloats` of
  the init params byte-equal the checkpoint (not silently He-init). A one-line
  check; the whole demo is a lie if this is off.
- **Image↔mask pairing.** This is *literally* the bug that broke BraTS before (the
  shuffle permuted images by record, labels by 4 bytes). The `lean_f32_shuffle`
  label-stride fix + `TestShufflePairing.lean` cover it now — but add a decode-time
  assert (a known slice's mask overlaps its image's non-zero brain region) as a
  belt.
- **Eval scores the right checkpoint.** The detector's `long30` bug: infer ran
  without the arm tag and scored the wrong params six times (identical rows). Assert
  the loaded checkpoint path contains the expected arm tag, and that consecutive-
  epoch eval rows are not byte-identical.
- **Modality order.** If v0 selects 3 of 4 channels, assert the SAME 3 in the same
  order at train and eval (a silent reorder is invisible in the loss). MSD permuted
  BraTS's native labels (memory `brats-demo-thread` warns of the 1/2/4→2/1/3 remap)
  — the same class of silent index bug applies to modality selection.

These are CPU-fast. Build them as the demo is built, not after being fooled once.

---

## 6. Compute / box

BraTS UNet training is CNN-heavy and gfx1100 is MIOpen-conv-weak (~40 min/epoch for
the from-scratch UNet per `unetBratsConfig`; the heavier R34-UNet will be slower).
Per `rocm-is-the-transformer-box` / `cross-box-throughput-anchors`, **run the real
training on ares** (the CNN box), keep gfx1100 for the CPU-FD checks and the
overfit smoke. Confirm the box's throughput anchor before committing to an epoch
budget.

---

## 7. Decisions that are yours

1. **Which 3 modalities for v0** (recommend FLAIR/T1gd/T2w), or jump straight to the
   v1 4-channel fresh stem (more faithful, needs the bootstrap extension first).
2. **Resolution: 224 vs 256** (224 = backbone-native + clean stride; 256 keeps more
   of the 240 native detail). Cheap to A/B once wired.
3. **Scope of "retraining"**: full fine-tune (headline) vs a frozen-encoder ablation
   alongside it (measures transfer strength for ~free).
4. **Is the R34-transfer worth it if it only ties the from-scratch baseline?** The
   transfer story's usual payoff is sample-efficiency (fewer epochs / less data to
   the same Dice), not a higher peak — decide whether that's the claim, and design
   the run to measure epochs-to-target, not just final Dice.

---

## 8. First concrete steps (for the clean session)

- Read `demos/MainUnetBratsTrain.lean` + the detector's R34 layer list + the
  `.fpnDetect` tap wiring (the closest precedent for R34-tap → head).
- Add the R34-encoder + UNet-decoder spec (v0: 3-ch, 224). Reuse `unetUp` /
  `bilinearUpsample`; new part is the tap→skip wiring + its backward seam.
- FD-verify the seam (tiny scale, CPU) → overfit gate (8 slices) → full run vs
  control. Guards from §5 in place before the full run.
```
# skeleton (fill spec/exe names once built):
lake build unet-brats-r34         # the new exe
<FD probe>                        # gate 1: seam correct
<overfit 8 slices, hard thresh>   # gate 2: train path works
# full run on ares, plain CE, R34-bootstrap vs He-init control, WT/TC/ET Dice
```
