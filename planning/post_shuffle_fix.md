# post_shuffle_fix.md — what to do now that the detector works

**Written 2026-07-22 for a fresh session.** Prereq: `planning/jax_gradient_oracle.md` §0
(the root cause and how it was found). This doc does not re-derive that; it is the ledger of
what the fix invalidated and what is worth doing next.

**State:** commit `e164958` on `main`, **ahead 1, NOT pushed**. Working tree otherwise carries
only long-standing untracked `runs/`, `figures/`, `verified_mlir/` noise — leave it alone.

---

## 0. The one-paragraph version

`ffi/f32_helpers.c: lean_f32_shuffle` permuted images by a full record and labels by a
**hardcoded 4 bytes** — one float, a classification scalar. Every detector and segmentation
trainer therefore saw mismatched image/target pairs on every epoch and could only learn the
marginal target distribution. Fixed by threading `label_stride` through `F32.shuffle` from
`dio.labelBytesPerRecord`. The FPN detector went **mAP@0.5 0.0001 → 0.1167**, recall
**0.118 → 0.7353**, 8-image-overfit objectness **27.550 → 0.526**.

**Nothing else has been re-measured yet.** That is what this doc is for.

---

## 1. The invalidation ledger — do these in order

Every trainer with `labelBytesPerRecord ≠ 4` ate the corruption. Classification is stride-4,
exactly the hardcoded value, so **all classifier work is unaffected** — the five verified
nets, ImageNet, CIFAR, ViT, the width sweeps, every proof-side artifact. Do not re-run those.

| # | arm | `labelBytes` | status |
|---|---|---|---|
| — | FPN detector | 740,880 | ✅ fixed + re-run |
| **1** | **BraTS UNet** (`bratsIO`) | 240² | ❌ every trained result void |
| **2** | **YOLO v2 / mosaic** | 30·gH·gW·4 + … | ❌ void |
| **3** | **Pets segmentation** (`petsIO`) | 224² | ❌ void |
| **4** | anchor YOLO (pre-FPN) | A·15·gH·gW·4 | ❌ void, low value |
| **5** | YOLOv1 pets detection | 7,200 | ❌ void, low value |

### 1a. BraTS is the highest-value re-run — a diagnosis may be wrong

`brats-demo-thread` concluded **Gates A and B both fail**, that `ce` == `dicece` == a trivial
predictor, and root-caused it to **Dice's gradient vanishing ∝ p_i**, prescribing
class-weighted CE. **Mispaired image/mask data also produces exactly a trivial predictor,
and it is the simpler explanation.**

Careful about what is and is not void: the *gradient measurements* (`seg_grad_scorecard.py`)
are statements about the loss functions in isolation and still stand. What is void is every
conclusion drawn from a **trained** arm — Gates A/B, the wce 196× amplification verdict, the
focal no-op-at-init verdict, and the whole ablation series in
`runs/brats_ablation_*.log` / `runs/brats_wcesqrtpbcos_*.log`.

Re-run the ablation arms on the fixed shuffle **before** crediting or dismissing weighted CE.
If plain `ce` now segments, the entire loss-design thread was chasing a data bug and the demo
gets much simpler.

### 1b. YOLO v2 — the "transfer gap" is suspect

`yolo-v2-thread` records mAP@0.5 **0.041 mosaic vs ≈0.0002 single-frame** and reads that as
"the transfer gap". Both arms were mispaired. Re-measure before treating that gap as real.

### 1c. Docs already corrected, content not

`planning/yolo_scoring.md` and `planning/yolo_assignment.md` now open with ⛔ banners marking
everything below them void. The banners are honest but they are not a re-measurement. In
particular these four are **untested, not refuted** — each may now work:

- **T1a** objectness `/num_pos`
- **T1b** class-weighted CE (currently ON; never actually A/B'd on valid data)
- **T2-bias** RetinaNet prior init (the "washed out by training" reading is void)
- **T2a** head tower, +7.08M params (`FPN_TOWER=4`)

Also void: the "target assignment is the constraint" thesis, the ring/neighbour analysis, the
resolution / readout / IoU-target probe conclusions, and the "converged equilibrium at
p≈0.14" story. The numpy probes themselves are sound instruments — rerun them, don't rewrite
them.

---

## 2. The FPN detector's real remaining work

Now that it trains, these are ordinary engineering rather than bug hunts.

1. **Fix the eval-GT truncation before quoting any absolute number.** `pack_raw_boxes` caps at
   `MAX_BBOXES = 56`; VisDrone val averages 70.7 boxes/image, so **34.9% of val GT is silently
   dropped** and 59.1% of val images are over the cap. Arm-vs-arm comparisons are unaffected
   (same truncated GT both sides), but **0.1167 is not VisDrone protocol** and must not be
   published as such. This gates every external claim.
2. **Close the 84% gap to the twin** (Lean 0.1167 vs twin 0.1391; recall already matches at
   0.7353 vs 0.738). Two known divergences, both cheap to test: Lean's **Adam + coupled wd**
   vs the twin's **decoupled AdamW**, and the **jax-ImageNet bootstrap** vs torchvision init.
   Recall matching while mAP lags points at ranking, i.e. the optimizer/regularization side.
3. **Augmentation via `augmentBatch` is off for detection and segmentation — but that is
   not the same as "no augmentation", and the original wording here was wrong.**
   `DatasetIO.augmentBatch : raw → batch → seed → IO ByteArray` receives **only the images**,
   so it cannot transform a mask or a box set, and every seg/det dataset correctly sets
   `augmentBatch := fun raw _ _ => return raw`. But both paths have a *pair-aware* augmenter
   wired in the batch loop instead, bypassing the hook entirely:
   - **Segmentation** — `F32.segHflipPair` (Train.lean ~747, gated on `useSeg && cfg.augment`)
     flips the f32 image and the u8 mask under one coin per image. Verified correct.
   - **Single-box detection** — `F32.yoloAugment` (gated on `cfg.augment`) does a bbox-aware
     hflip + crop and re-encodes target/mask from the transformed raw boxes. Verified correct,
     and it derives its strides from `gridH/gridW` rather than hardcoding them.

   What is genuinely unaugmented is the **FPN and anchor** detection path: both set
   `augment := false`, and no pair-aware augmenter exists for the multi-scale flat target.
   That is where the unclaimed ~1.7× lives. Note also that on the single-box path
   `yoloAugment` re-encodes from the **capped** box list (`n_boxes > 56 → 56`), so on
   VisDrone an augmented epoch trains on 73.1% of the GT boxes — the `MAX_BBOXES` cap in
   item 1 is not only an eval-side problem.
4. **Re-test the four levers** (§1c) now that a lever can actually move the metric.
5. **Longer schedule.** The 12-epoch loss was **still descending** at e12 (112.62, no plateau).
   The old arm flattened at ~300 by e4, which is why 12 epochs was ever considered enough.

---

## 3. Prevention — the actual structural gap

**The host data path has no differential reference.** Every verification instrument this repo
has — FD probes, `iree-compile`, the FloatBridge ties, the §1a spec ties, the proof
obligations — inspects *emitted code*. A bug in the C that assembles the batch is invisible to
all of them simultaneously, and a same-`NetSpec` JAX oracle would have been fed the same
mispaired batch, so it would not have caught this either.

Worth building, cheapest first:

1. ✅ **DONE — and it immediately caught that the 430ba2c guard did not work.**
   `tests/TestShufflePairing.lean` (`lake exe test-shuffle-pairing`, hermetic, no data, no
   GPU) builds records where image *k* is `pixels` copies of *k* and label *k* is
   `labelFloats` copies of *k*, shuffles, and checks per element that label slot *k* still
   describes image slot *k* — plus that the result is a real permutation and not the
   identity, so it cannot pass vacuously.

   **What it found.** The guard 430ba2c added was
   `lean_sarray_size(lbl_obj) < n * label_stride`, which fires only when the stride is too
   *large*. Under-reporting — a caller claiming 4 bytes for a 740,880-byte record, i.e. the
   original bug stated as a call — sails straight through, and the test reproduced the
   corruption verbatim through the "fixed" API. The C comment directly above it claimed it
   refused exactly that case. Now an exact-equality check on **both** buffers, so a stride
   that disagrees with its buffer in either direction is an error; `n == 0` is rejected too
   (it underflowed `n - 1` into a `SIZE_MAX` loop), and the error path no longer leaks its
   two arguments.

   Companion: `tests/TestDatasetRecordSizes.lean` (`lake exe test-dataset-record-sizes`)
   loads each dataset present on disk and asserts `images = n × trainPixels × 4` and
   `labels = n × labelBytesPerRecord` — the same invariant, against real loaders. All 10
   dataset configurations pass, which is what licenses the stricter guard. Run it whenever a
   dataset or its preprocessing changes; it skips what is absent, so it is a pre-flight
   check rather than a CI job.
2. **Adopt the frozen-parameter probe as a standard gate.** Set LR to 0, run N steps, demand a
   bit-stable loss. Two minutes, no instrument, and it would have ended this thread in July.
   Memory: `frozen-param-determinism-probe`.
3. ✅ **DONE — the full host data path swept.** All 56 `LEAN_EXPORT`s in `f32_helpers.c`, every
   batch-slicing loop in `Train.lean` / `VerifiedTrain.lean` / the demo mains, and every
   `preprocess_*.py` writer against its C reader. Two defects found beyond the guard above;
   both are fixed, and the rest of the sweep is recorded in §6.
4. **Never accept summary statistics as a tie.** The twin's "forward exonerated" check compared
   mean/std/min — all permutation-invariant — and reported ~2% on a forward whose per-element
   disagreement was rel 0.327. Compare **per element**, and compare **sorted** values as well:
   matching sorted values with mismatched positions separates a layout/alignment bug from a
   numerical one.

---

## 4. Recipes

**Re-run the FPN arm** (~4.4 h, 24 min/epoch):
```
IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=1 FPN_TAG=<distinct> \
  .lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn > runs/<log> 2>&1
```

**The 8-image overfit gate** (~55 min, 1.65 s/epoch) — always gate on this before a full run:
```
IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 FPN_TAG=<distinct> \
  FPN_EPOCHS=2000 FPN_CKPT_EVERY=500 \
  .lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn_of8
```
Reference: fixed arm reaches **3.145** total / **0.526** objectness at e2000.

**Score:**
```
FPN_TAG=<tag> FPN_TOWER=0 IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 \
  .lake/build/bin/yolov1-visdrone-fpn infer data/visdrone_fpn figures/<out>
visdrone/.venv/bin/python3 scripts/yolo_map_visdrone.py \
  figures/<out>/logits.bin data/visdrone448/val.bin --fpn data/visdrone --grid 14
```

**The frozen-parameter determinism probe** — run this FIRST on any "cannot descend" report:
```
FPN_LR_MULT=0 FPN_EPOCHS=3 FPN_CKPT_EVERY=99 FPN_TAG=<distinct> ...
```
Loss must be identical every step. A ~1e-4 residual is fp32 reduction-order noise; anything
larger means the input is changing.

**Validate the PyTorch twin against Lean** (from `visdrone/`, run before trusting any diff):
```
.venv/bin/python3 -m bespoke.validate_oracle --ckpt <lean_params.bin> \
  --dump <logits.bin> --data <train.bin> [--bn-stats <bn_stats.bin>]
```

### Gotchas that have each cost real time

- **`FPN_TAG` is not optional.** The spec name IS the on-disk checkpoint prefix; an untagged
  probe silently overwrites a live arm's checkpoints. Nothing errors.
- **`FPN_TOWER` selects the SPEC, so set it on `infer` too.** Cost a full 12-epoch eval sweep.
  The tell is zero variation between checkpoints.
- **`IREE_BACKEND=rocm` on the train step**, or the loss reduce dies with
  `'func.func' op failed to distribute`.
- **A new `FPN_TAG` means a new vmfb — ~6.5 min compile**, and two concurrent compiles are
  slower still. Budget for it.
- **`pad="lean"` is required** on the PyTorch twin for any checkpoint comparison
  (`MlirCodegen.samePad` is asymmetric). `pool="lean"` too.
- **`cd` persists between Bash calls** in this harness; it bit twice this session. Use
  absolute paths.
- **`pkill -f <pattern>` kills the calling shell** when the pattern matches your own command
  line. Collect PIDs with `pgrep` and kill those.

---

## 5. Decisions that are yours

1. **Push `e164958`?** It is one squashed commit on top of the two pulled remote commits.
   Reflog recovery point for the pre-rebase history: `a99c650b48`.
2. **Is VisDrone still the demo target?** 70 objects/image at 2–5 px is punishing for a
   "verified stack trains a real detector" claim. Now that the stack demonstrably works, an
   easier dataset would make the claim land sooner, with VisDrone as the stretch. Worth
   deciding explicitly rather than by default. For calibration: VisDrone-DET2019 lists
   RetinaNet as the **weakest** of its standard baselines, and this is a RetinaNet-class
   detector at 448 px, so **0.15–0.25 is the right expectation for a fully-working arm**, not
   YOLOv8's 0.38.
3. **How much of the refuted analysis to re-run?** §1c is five arms × 4.4 h. The honest
   minimum is T1b (it is currently ON and was never validly A/B'd) and one control with all
   levers off. The rest can wait for a reason.

---

## 6. The audit — what else was wrong (2026-07-22, uncommitted)

Swept for one bug class only: **anything that can make the image at slot *k* and the label at
slot *k* stop referring to the same example.** Three things were fixed; the rest of this
section is the negative result, which is the part worth keeping.

### 6a. Fixed

1. **The 430ba2c guard did not guard.** `<` instead of `==`, so it accepted the exact call it
   was written to reject. See §3.1. `ffi/f32_helpers.c`.
2. **TTA eval ran the *training* augmentation on *validation* data.** `Train.lean` called
   `dio.augmentBatch` on the val batch under `cfg.useTTA`. Imagenette stores train at 256 and
   val at 224, so `randomCrop 256→224` strode a 196,608-float record across a 150,528-float
   buffer: slot *i* drifted by 46,080 floats, and at `evalBatch = 32` the last 8 slots read
   ~5.9 MB past the end of the allocation, while `lblSlice.data[i*4]` still supplied slot
   *i*'s original label. **This is a live bug, not a latent one** — it fired on the
   `vit-tiny-tta` and `vit-tiny-kitchensink` ablation arms, whose reported TTA accuracy was
   measured on mispaired data plus out-of-bounds heap reads. The comment on
   `imagenetteIO.augmentBatch` warned about precisely this ("The eval pipeline must NOT call
   this on val data"); the TTA path, added later, did it anyway.
   Fixed by adding `DatasetIO.valAugmentBatch`, which receives **val-sized** records and
   defaults to `valPreprocessBatch` (so TTA degrades to a no-op rather than a corruption).
   Imagenette overrides it with a 224-native hflip. Those two arms need re-running.
3. **`lean_voc_split_batch` hardcoded the Pets 7×7 record** (7200 B) while its caller sliced
   at `dio.labelBytesPerRecord` — 25,428 at VisDrone-448/14×14, 98,340 at 28×28. Reading
   record *i* at a 7200 stride out of a 25,428-stride buffer pairs an image with a target
   lifted from the middle of some other record, and the output shape is still exactly what
   the train step expects, so nothing downstream could see it. Not currently reachable: all
   three single-box det configs set `augment := true` and take the `yoloAugment` arm instead.
   It was one boolean away. Now parameterized by `gridH/gridW/perCell` and it rejects a
   stride that disagrees with the buffer.

### 6b. Cleared — the parts that turned out to be right

- **Every `.bin` writer.** All 14 dataset/format combinations pair image and label by
  construction (keyed by stem, or written as one concatenated record), and every writer's
  bytes-per-record matches its C reader and its `labelBytesPerRecord`, confirmed by on-disk
  arithmetic on all 24 files. Independently verified that `visdrone_fpn/val.bin` record *k*
  and `visdrone448/val.bin` record *k* hold byte-identical images (the scorer relies on this
  cross-file assumption) and that record *k* is `sorted(glob(...))[k]` against the source JPEGs.
- **Mixup / CutMix / KNN-mixup.** These are the one place pairing is established by *two*
  independent derivations from a shared seed rather than in a single call — the fragile
  shape. Traced instruction-by-instruction: each pair uses the same seed constant, the same
  `f32_beta_symmetric → f32_permutation` call order and the same λ-flip, so the permutation
  is bit-identical, and `Train.lean` passes matching seeds and alphas. They do hardcode a
  4-byte label read, which is unreachable with a non-4 stride only because `compileVmfbs`
  throws on seg/yolov1 × mixup. Correct today, unguarded in C.
- **All 13 `VerifiedTrain.lean` loops** — classification-only by construction, 4-byte labels,
  drop-last on both sides of every batch slice.
- **Train and eval batch slicing** — image and label always sliced over the same record range
  with their own correct per-record size; `bpE := nTrain / batchN` is drop-last everywhere, so
  no tail batch can pad the two sides differently.
- **`segHflipPair`, `yoloAugment`, the DDPM `stepInputs` path, and every detection inference
  dump** (which pads the tail by repeating the last real image and then truncates, so
  `logits.bin` row *k* is val record *k*).

### 6c. Latent — correct today, unguarded, worth knowing

- **No loader checks its file size against `4 + count × rec_bytes`.** Over-specifying a stride
  short-reads and errors; **under**-specifying slides every record silently. This is the
  general form of the whole class. `visdrone/bespoke/data.py` asserts exactly this and is the
  pattern to copy.
- **Geometry hardcoded away from its source of truth**: `loadPets` fixes 224, `loadBrats`
  fixes 4 channels while `preprocess_brats.py` exposes `--size`, `preprocess_visdrone.py`'s
  `FPN_GRIDS` is 448-only and a mismatch only warns. A regenerated `.bin` at another size
  would be read at the wrong stride with no error.
- **`sliceLabels`' `bytesPerLabel : Nat := 4` default** is the same "a label is one float"
  assumption that caused this. Every `Train.lean` site passes it explicitly.
- **`segHflipPair` takes `augH/augW` from the spec, not from `dio`**, and validates neither
  buffer — a seg spec whose input size differed from the stored mask size would read out of
  bounds. The Imagenette 256/224 split is exactly that shape, so this is not hypothetical.
- **SWAG eval is not gated by `useSeg`/`useYolov1Run`** the way the regular eval loops are; it
  would `argmax10` over a detection output and read a label at `i*4` out of a 50,176-byte mask
  record. No config currently pairs `useSWAG` with a non-classification dataset.
- **`MAX_BBOXES = 56` truncation is worse than §2.1 says.** Val: 34.9% of GT dropped, 59.1% of
  images over the cap. **Train: 26.9% dropped, and `yoloAugment` re-encodes from the capped
  list**, so an augmented single-box epoch never sees those boxes at all.
