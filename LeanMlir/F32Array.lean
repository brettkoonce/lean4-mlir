/-! Float32-in-ByteArray utilities.

    All tensor data (params, images, gradients) stored as raw float32 bytes
    in `ByteArray`. Zero conversion at the FFI boundary — IREE sees the same
    bytes Lean wrote. Lean-side reads widen to `Float` (f64) only when needed
    (loss printing, argmax, debugging).

    Heavy-lift operations (He init, const fill, image loading) are @[extern]
    to C for speed — avoids millions of Lean-level push calls. -/

namespace F32

/-- Number of float32 elements in a ByteArray. -/
def size (ba : ByteArray) : Nat := ba.size / 4

/-- Read a float32 at `idx` (element index, not byte index), widened to Float. -/
@[extern "lean_f32_read"]
opaque read (ba : @& ByteArray) (idx : USize) : Float

/-- Allocate `n` float32 values filled with `v` as a ByteArray. -/
@[extern "lean_f32_const"]
opaque const (n : USize) (v : Float) : IO ByteArray

/-- He-init: `n` float32 values ~ N(0, scale²), packed in ByteArray.
    Uses xorshift + 3-uniform-sum approximation (same as existing randnFA). -/
@[extern "lean_f32_he_init"]
opaque heInit (seed : USize) (n : USize) (scale : Float) : IO ByteArray

/-- Tile one flattened image `base[off .. off+d0)` (element offset `off`, `d0` floats)
    into `m` copies, each with independent `N(0, σ²)` **exact** Gaussian noise added
    (Box-Muller). Returns `m*d0` float32. NO clipping — the randomized-smoothing
    certificate (Cohen–Rosenfeld–Kolter 2019) lives in raw input L2 space. With `m=1`,
    `off=0`, `d0=bs·pix` it noises a whole training batch (Gaussian data augmentation). -/
@[extern "lean_f32_add_gaussian_tiled"]
opaque addGaussianTiled (base : @& ByteArray) (off d0 m : USize)
  (sigma : Float) (seed : USize) : IO ByteArray

/-- Perturb one image `base[off .. off+d0)` by `r·u` for a uniformly-random unit vector `u`
    (so `‖r·u‖₂ = r` exactly). Returns the `d0`-vector `x + r·u`. For the Lipschitz-hypothesis
    probe: shift the input by a known L2 amount and watch `Φ⁻¹(P[f(x+η)=c])` respond. -/
@[extern "lean_f32_perturb_unit"]
opaque perturbUnit (base : @& ByteArray) (off d0 : USize) (r : Float) (seed : USize) : IO ByteArray

/-- Write three consecutive f32 values starting at float index `idx`, **in place**
    when the array is unshared (it copies otherwise, so the result never depends
    on a refcount). Used to patch the `lr`/`bc₁`/`bc₂` slots of the Adam step
    buffer without rebuilding it — see `planning/xla_pjrt_ladder.md` §8. -/
@[extern "lean_f32_write3"]
opaque write3 (ba : ByteArray) (idx : USize) (a b c : Float) : IO ByteArray

/-- Copy `count` f32 values from `src[srcOff..]` into `dst[dstOff..]`, **in place**
    when `dst` is unshared. Used to patch the BN running-stat region of the Adam
    step buffer. -/
@[extern "lean_f32_blit"]
opaque blit (dst : ByteArray) (dstOff : USize) (src : @& ByteArray)
  (srcOff count : USize) : IO ByteArray

/-- `dst[dstOff + i] += a · src[srcOff + i]` for `i < count`, **in place** when `dst` is
    unshared. The perturbation primitive of the adjoint gradcheck
    (`tests/TestR50GradCheck.lean`): parameter tensors are packed in func-arg order, so a
    direction supported on one BLOCK is a contiguous slice and needs no mask. -/
@[extern "lean_f32_axpy_slice"]
opaque axpySlice (dst : ByteArray) (dstOff : USize) (src : @& ByteArray)
  (srcOff count : USize) (a : Float) : IO ByteArray

/-- `Σ_{i<count} a[aOff+i] · b[bOff+i]`, accumulated in **f64**.

    ⚠ The wide accumulator is the point. This computes the predicted directional derivative
    `⟨g, δ⟩` over as many as 4.7M same-sign terms; an f32 running sum would lose ~log₂ n bits to
    the accumulation and report it as disagreement between the gradient and the finite
    difference — i.e. as a failure of the thing under test. Returns `NaN` on an out-of-range
    slice rather than clamping, so a caller that mis-computes an offset sees it. -/
@[extern "lean_f32_dot_slice"]
opaque dotSlice (a : @& ByteArray) (aOff : USize) (b : @& ByteArray)
  (bOff count : USize) : Float

/-- **Per-site, per-example stochastic-depth scales** — `bernoulli(keep_i)/keep_i` for each of
    `keeps.size` drop sites × `bs` examples, laid out site-major to match the render's
    `%dp<i>: tensor<Bxf32>` inputs in signature order. Returns `keeps.size * bs` float32.

    **Pure Lean on purpose, in two ways.**

    *Not in the graph.* `stablehlo.rng` is disqualified: every numeric gate in this repo is a
    bit-exactness or known-answer argument over a DETERMINISTIC graph — the tie harnesses' A-vs-A
    floor, `residency_gate.sh`'s bit-identity, the duplicated-batch DP identity, the cross-lowerer
    IREE-vs-XLA agreement. A graph that draws its own randomness makes each of those either
    impossible or contingent on seeding an XLA RNG identically across two lowerers and two vendors.

    *Not in C either.* This is `keeps.size * bs` floats per step — 288 at EfficientNet's 9 sites and
    batch 32, against a ~310 ms step — so the C round trip buys nothing measurable, and keeping the
    draw in Lean keeps the one piece of genuine randomness in the training loop readable and seeded
    where it can be audited. `heInit` is extern because it fills millions of values; this does not.

    ⚠ **`seed` must be derived from the GLOBAL STEP**, like `augSeed`, or no run is reproducible and
    every gate that replays a step breaks. ⚠ `1/keep` is folded in HERE rather than baked into the
    graph — see `VerifiedNet.dropKeeps`. At `keep = 1` the scale is exactly `1.0` for every example,
    so a site with `keep = 1` is the identity in IEEE, not merely close. -/
def dropScales (keeps : Array Float) (bs : Nat) (seed : USize) : IO ByteArray := do
  let n := keeps.size * bs
  let mut out ← const n.toUSize 1.0
  -- xorshift64*, the same family `heInit` uses. Seeded per call from the global step, and
  -- advanced once per (site, example) so the draw order is fixed by the layout rather than by
  -- evaluation order.
  let mut st : UInt64 := (seed.toUInt64 * 2654435761) ||| 1
  let mut buf : Array Float := #[]
  for si in [0:keeps.size] do
    let k := keeps[si]!
    for _ in [0:bs] do
      st := st ^^^ (st <<< 13); st := st ^^^ (st >>> 7); st := st ^^^ (st <<< 17)
      -- u ∈ [0,1) from the top 24 bits, which is exactly float32's mantissa width.
      let u := (st >>> 40).toNat.toFloat / 16777216.0
      -- keep w.p. `k`, then scale survivors by 1/k so the expectation is preserved and eval
      -- (all-ones, k = 1) is the identity. `k ≥ 1` ⇒ always keep, scale 1.0.
      buf := buf.push (if k ≥ 1.0 then 1.0 else if u < k then 1.0 / k else 0.0)
  -- `write3` is the only in-place float writer; `keeps.size * bs` is divisible by 3 whenever the
  -- site count is (9 on EfficientNet-B0), so this tiles exactly with no tail.
  let mut i := 0
  while i + 3 ≤ buf.size do
    out ← write3 out i.toUSize buf[i]! buf[i+1]! buf[i+2]!
    i := i + 3
  for j in [i:buf.size] do
    out ← write3 out (j - 2).toUSize buf[j-2]! buf[j-1]! buf[j]!
  pure out

/-- ▶ **The classifier-dropout mask** (`recipe_gaps.md` gap C) — `n` INDEPENDENT Bernoulli draws at
    keep probability `keep`, each survivor scaled by `1/keep`. Fills one `%do: tensor<B×w×f32>`
    graph input, so `n = B * w`.

    ⚠⚠ **`n` DRAWS, NOT `B` — this is `dropScales`' per-example loop replaced by a per-ELEMENT one,
    and that single difference is the whole distinction between the two regularisers.** The
    reference draws `bernoulli(key, keep, x.shape)` for the classifier
    (`jax/Jax/Codegen.lean:1971`) against `(branch.shape[0],) + (1,)*(ndim-1)` for stochastic depth
    (`:1037`). A mask built by drawing `B` values and repeating each `w` times type-checks, fills
    the same buffer, trains and descends — it is stochastic depth on the classifier. Nothing
    downstream can tell: the shapes agree, the emitted graph is identical, and only the
    DISTRIBUTION differs. `Proofs.dropPath_scales_uniformly` is the statement of what would be
    wrong; on this side it is the loop bound, and there is no gate but reading it.

    ⚠ **Seed it from a stream DISJOINT from `dropScales`'.** The reference offsets by `999983`
    (`fold_in(drop_key, 999983)` against `fold_in(drop_key, block_index)`) exactly so a net running
    both regularisers does not correlate them; the caller adds that offset. Sharing a seed here
    would make the classifier mask a function of block 0's drop decisions, every step.

    ⚠ Same `1/keep` folding as `dropScales`, for the same reason: the graph bakes no constant, so a
    ones mask (`keep ≥ 1`, or eval) is the EXACT identity rather than a rescale.

    Pure Lean for `dropScales`' reasons — `stablehlo.rng` would make every bit-exactness gate in the
    repo contingent on seeding an XLA RNG identically across two lowerers. ⚠ It is bigger than
    `dropScales` (40,960 floats at B=32×1280, against 288) but still ~2 ULP of a ~310 ms step, and
    keeping the draw here keeps the randomness readable and seeded where it can be audited. -/
def dropoutMask (keep : Float) (n : Nat) (seed : USize) : IO ByteArray := do
  let mut out ← const n.toUSize 1.0
  if keep ≥ 1.0 || n == 0 then
    return out
  -- ⚠ REFUSE below 3 rather than return the all-ones buffer. `write3` is the only in-place float
  -- writer, so it cannot tile 1 or 2 elements, and the overlap-backwards tail below would index
  -- past the start. The failure mode if this were silent is the bad one: `out` is already filled
  -- with 1.0, so a too-small mask would come back as the exact IDENTITY — dropout switched off,
  -- with the render, the arity and every shape still correct. Real call sites are `B * width`
  -- (≥ 1280 here), so this only fires on a mis-sized caller, which is precisely when it should.
  if n < 3 then
    throw (IO.userError s!"dropoutMask: n = {n} < 3 cannot be written by write3; a silent \
all-ones return would be the identity, i.e. dropout switched off")
  -- xorshift64*, the same family `dropScales` and `heInit` use.
  let mut st : UInt64 := (seed.toUInt64 * 2654435761) ||| 1
  let inv := 1.0 / keep
  let mut buf : Array Float := #[]
  for _ in [0:n] do
    st := st ^^^ (st <<< 13); st := st ^^^ (st >>> 7); st := st ^^^ (st <<< 17)
    -- u ∈ [0,1) from the top 24 bits — float32's mantissa width, as in `dropScales`.
    let u := (st >>> 40).toNat.toFloat / 16777216.0
    buf := buf.push (if u < keep then inv else 0.0)
  -- `write3` is the only in-place float writer, so tile in threes and finish the tail by
  -- overlapping backwards from the end (`dropScales`' idiom — every element is written, and the
  -- overlap rewrites already-correct values rather than skipping any).
  let mut i := 0
  while i + 3 ≤ buf.size do
    out ← write3 out i.toUSize buf[i]! buf[i+1]! buf[i+2]!
    i := i + 3
  for j in [i:buf.size] do
    out ← write3 out (j - 2).toUSize buf[j-2]! buf[j-1]! buf[j]!
  pure out

/-- Concatenate multiple ByteArrays. Fast (memcpy per chunk). -/
def concat (arrays : Array ByteArray) : ByteArray := Id.run do
  let mut out : ByteArray := .empty
  for a in arrays do out := out.append a
  return out

/-- Slice `count` float32 elements starting at element index `start`. -/
def slice (ba : ByteArray) (start count : Nat) : ByteArray :=
  ba.extract (start * 4) ((start + count) * 4)

/-- Extract the loss (last float32) from a train_step output. -/
def extractLoss (out : ByteArray) (lossIdx : Nat) : Float :=
  read out lossIdx.toUSize

/-- Drop the trailing loss float from train_step output. -/
def dropLoss (out : ByteArray) (nParams : Nat) : ByteArray :=
  out.extract 0 (nParams * 4)

/-- Decode the little-endian **int32** label at record `i` of a packed label buffer.

    ⚠ Replaces `lbl.get! (4 * i)`, which returned a `UInt8` — **byte 0 only**, i.e. `label % 256`.
    Every net this repo GATES is 10-class (Imagenette / CIFAR / MNIST), so byte 0 *is* the label
    there and the truncation was invisible; on 1000-class ImageNet it silently discarded the high
    byte, and since a correct prediction can then only match on classes 0..255, it capped every
    reported top-1 at roughly a quarter of the truth. Measured off the val wire 2026-08-05: the
    first batch carries labels 1..988 with **193 of 256 (75.4%) above 255**. -/
def readLabel (lbl : ByteArray) (i : Nat) : Nat :=
  (lbl.get! (4 * i)).toNat
    ||| ((lbl.get! (4 * i + 1)).toNat <<< 8)
    ||| ((lbl.get! (4 * i + 2)).toNat <<< 16)
    ||| ((lbl.get! (4 * i + 3)).toNat <<< 24)

/-- Argmax over `n` float32 values starting at element offset `off`.

    ⚠ Replaced `argmax10`, which took no `n` and scanned a literal 10 entries. Every net this
    repo GATES is 10-class (Imagenette / CIFAR / MNIST), so the constant was right everywhere it
    was ever checked and wrong on exactly the un-gated tier — the 1000-class ImageNet trainers,
    where it confined every prediction to labels 0..9. Pass the net's own class count at the call
    site; it is already the multiplier in the `off` expression, so the two cannot disagree. -/
@[extern "lean_f32_argmax_n"]
opaque argmaxN (ba : @& ByteArray) (off : USize) (n : USize) : USize

/-- Rank of `label` in a row of `n` logits at element offset `off`: the count of entries
    STRICTLY GREATER than the label's own logit. The label is in the top-`k` iff `rank < k`.

    Deliberately the same construction as the JAX reference's
    `jnp.sum(logits > true_logit, axis=1) < 5` rather than a sort or `top_k` — that side records
    `jax.lax.top_k`'s indices as broken on ROCm/gfx1100, and matching the formulation makes the
    two paths' top-5 comparable by construction, ties included (strictly-greater means a tie
    resolves in the label's favour on both sides). -/
@[extern "lean_f32_rank_of"]
opaque rankOf (ba : @& ByteArray) (off : USize) (n : USize) (label : USize) : USize

/-- Load MNIST images from IDX file directly into f32 ByteArray (normalized to [0,1]).
    Returns (images ByteArray, count as Nat). -/
@[extern "lean_f32_load_idx_images"]
opaque loadIdxImages (path : @& String) : IO (ByteArray × Nat)

/-- Load MNIST labels from IDX file into int32 LE ByteArray. -/
@[extern "lean_f32_load_idx_labels"]
opaque loadIdxLabels (path : @& String) : IO (ByteArray × Nat)

/-- Slice a batch of images: `count` images × `pixelsPerImage` floats. Zero-copy. -/
def sliceImages (images : ByteArray) (start count pixelsPerImage : Nat) : ByteArray :=
  images.extract (start * pixelsPerImage * 4) ((start + count) * pixelsPerImage * 4)

/-- `sliceImages`, zero-padding past the end of the dataset (`total` images).
    The batch dimension is baked into the compiled `.vmfb`, so a final partial
    eval batch must still be a full `count` images; the caller scores only the
    first `total - start` rows of the result. -/
def sliceImagesPad (images : ByteArray) (start count pixelsPerImage total : Nat) : ByteArray :=
  let avail := min count (total - start)
  if avail == count then sliceImages images start count pixelsPerImage
  else sliceImages images start avail pixelsPerImage
       ++ ByteArray.mk (Array.replicate ((count - avail) * pixelsPerImage * 4) 0)

/-- Slice a batch of labels: `count` records of `bytesPerLabel` bytes
    each. Defaults to 4 (int32 LE) for classification. Per-pixel
    segmentation masks pass `bytesPerLabel := H * W` (e.g. 224*224 = 50176
    for Pets). Zero-copy. -/
def sliceLabels (labels : ByteArray) (start count : Nat) (bytesPerLabel : Nat := 4) : ByteArray :=
  labels.extract (start * bytesPerLabel) ((start + count) * bytesPerLabel)

/-- Convert a batch of CIFAR-10 raw records to f32 ByteArray.
    `raw` is the concatenated batch file bytes (3073 bytes per record).
    Returns `count × 3072` float32 values normalized to [0,1]. -/
@[extern "lean_f32_cifar_batch"]
opaque cifarBatch (raw : @& ByteArray) (start : USize) (count : USize) : IO ByteArray

/-- Load Imagenette binary file. Returns (images f32 ByteArray, labels i32 ByteArray, count).
    Images are normalized with ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]. -/
@[extern "lean_f32_load_imagenette"]
opaque loadImagenette (path : @& String) : IO (ByteArray × ByteArray × Nat)

/-- Load Imagenette with explicit image size (e.g. 256 for train, 224 for val). -/
@[extern "lean_f32_load_imagenette_sized"]
opaque loadImagenetteSized (path : @& String) (imgSize : USize) : IO (ByteArray × ByteArray × Nat)

/-- Load Oxford-IIIT Pets binary file. Returns
    (images f32 ByteArray, masks uint8 ByteArray, count).
    Images are 224×224×3, channel-first, normalized with ImageNet mean/std.
    Masks are 224×224 uint8 per-pixel class labels (0=fg, 1=bg, 2=boundary). -/
@[extern "lean_f32_load_pets"]
opaque loadPets (path : @& String) : IO (ByteArray × ByteArray × Nat)

/-- Load a BraTS (MSD Task01_BrainTumour) binary file at the given in-plane
    size. Returns (images f32 ByteArray, masks uint8 ByteArray, count).
    Images are `imgSize`×`imgSize`×4 (FLAIR / T1w / T1gd / T2w), channel-first.
    Unlike the RGB datasets these carry no ImageNet normalization: the loader
    inverts the uint8 quantization `preprocess_brats.py` applied, yielding the
    per-volume, per-modality z-scored intensities the preprocessor computed over
    brain voxels. Masks are `imgSize`×`imgSize` uint8 per-pixel class labels
    (0=background, 1=edema, 2=non-enhancing tumour, 3=enhancing tumour). -/
@[extern "lean_f32_load_brats"]
opaque loadBrats (path : @& String) (imgSize : USize) : IO (ByteArray × ByteArray × Nat)

/-- YOLOv1 detection-bin loader (target+mask format; used by Pets). Returns `(images_f32_normalized,
    yLabels_concat, count)` where `yLabels_concat` carries **7200** bytes
    per image: 30×7×7 float32 target (5880), then 7×7 float32 mask (196),
    then numBoxes (4), then raw_boxes 56×20 (1120) — the Phase 3b format,
    matching `petsDetIO.labelBytesPerRecord`. (This docstring said 6076,
    the pre-Phase-3b target+mask size, long after the record grew the bbox
    tail; a stale stride in the docs is what this whole bug class feeds on.)
    The Lean dispatcher (`runTraining`) splits this into target + mask before
    calling `trainStepAdamF32Yolov1`. See `preprocess_pets_mosaic.py` for the
    on-disk format. -/
@[extern "lean_f32_load_voc"]
opaque loadDetBin (path : @& String) : IO (ByteArray × ByteArray × Nat)

/-- Dimension-parameterized detection-bin loader (same record format as
    `loadDetBin`, but for an arbitrary square input `imgSize` and grid
    `gridH`×`gridW`). Used for the higher-resolution VisDrone path (448 input /
    14×14 grid); `loadDetBin` is the fixed 224/7×7 Pets path. -/
@[extern "lean_f32_load_voc_dims"]
opaque loadDetBinDims (path : @& String) (imgSize gridH gridW : USize)
    : IO (ByteArray × ByteArray × Nat)

/-- Anchor-format detection loader (brick #2). Returns `(images_f32_normalized,
    target_only_concat, count)` with `numAnchors·15·gridH·gridW` f32 target per
    image — the anchor loss derives its per-anchor mask from the target's
    objectness channels, so the on-disk mask/numBoxes/raw_boxes are skipped. -/
@[extern "lean_f32_load_voc_anchor"]
opaque loadDetBinAnchor (path : @& String) (imgSize gridH gridW numAnchors : USize)
    : IO (ByteArray × ByteArray × Nat)

/-- FPN multi-scale detection loader (brick #3). Returns `(images_f32_normalized,
    target_only_concat, count)` with `ntot` f32 target per image — the flat
    `[P3|P4|P5]` block (`ntot = Σ_s numAnchorsₛ·15·g_s²`). Like the anchor loader,
    the loss derives per-anchor masks from the target's objectness channels, so
    the on-disk record is just image + flat target (no mask/boxes). Eval GT comes
    from the single-box val.bin geometry, as in the anchor path. -/
@[extern "lean_f32_load_voc_fpn"]
opaque loadDetBinFpn (path : @& String) (imgSize ntot : USize)
    : IO (ByteArray × ByteArray × Nat)

/-- Split an interleaved YOLOv1 batch slice into separately-contiguous target
    and mask tensors suitable for the `trainStepAdamF32Yolov1` FFI.

    The per-record layout is `target (perCell*gH*gW*4) || mask (gH*gW*4) ||
    numBoxes (4) || raw_boxes (56*20)`; only the target and mask are extracted.
    Returns `(target_concat, mask_concat)` sized `batch * perCell*gH*gW*4` and
    `batch * gH*gW*4`.

    **Pass the caller's real grid.** This used to hardcode the Pets 7×7 record
    (7200 bytes/record) while the caller sliced at `dio.labelBytesPerRecord` —
    25428 at VisDrone-448/14×14. Reading at the wrong stride pairs each image
    with a target lifted out of a different record, which nothing downstream can
    see because the output shape is still correct. The FFI now rejects a stride
    that disagrees with the buffer. -/
@[extern "lean_voc_split_batch"]
opaque detSplitBatch (interleaved : @& ByteArray) (batch : USize)
    (gridH gridW perCell : USize)
    : IO (ByteArray × ByteArray)

/-- Bbox-aware horizontal flip for a YOLOv1 batch. Per-image p=0.5
    coin (xorshift64 seeded by `seed`); when flipped, reverses image
    along W, target along gridW, mask along gridW, and replaces the
    x_cell channel with `1 - x_cell` on cells where mask=1 (since the
    cell itself mirrors). Returns the augmented (images, target, mask)
    triple as fresh ByteArrays; inputs are not modified.
    See `planning/yolo_final.md` Phase 3. LEGACY — superseded by
    `yoloAugment` (Phase 3b) which operates on raw bboxes. -/
@[extern "lean_f32_yolo_hflip"]
opaque yoloHflip (images : @& ByteArray) (target : @& ByteArray) (mask : @& ByteArray)
    (batch : USize) (channels : USize) (imgH : USize) (imgW : USize)
    (gridH : USize) (gridW : USize) (perCell : USize) (seed : USize)
    : IO (ByteArray × ByteArray × ByteArray)

/-- Unified bbox-aware augmentation for YOLOv1: per-image hflip + random
    crop, with target+mask re-encoded from the transformed raw bboxes
    so the geometric correspondence is exact. Replaces `yoloHflip` for
    Phase 3b once preprocessor stores raw bboxes alongside the
    pre-encoded target.

    * `images`: f32 image batch `[B, C, H, W]`
    * `boxes`: per-record YOLOv1 label block (target 5880 + mask 196 +
      numBoxes 4 + raw_boxes 1120 = 7200 bytes/record). Only the
      numBoxes + raw_boxes tail is read.
    * `hflipProb`, `cropProb`: per-image Bernoulli probabilities.
    * `cropMinScale`: crop side ∈ `[cropMinScale, 1.0] × imgW`
      (paper's ±20% jitter → 0.8).
    * `seed`: xorshift seed.

    Returns `(new_image, new_target, new_mask)` as fresh ByteArrays.
    See `planning/yolo_final.md` Phase 3. -/
@[extern "lean_f32_yolo_augment"]
opaque yoloAugment (images : @& ByteArray) (boxes : @& ByteArray)
    (batch : USize) (channels : USize) (imgH : USize) (imgW : USize)
    (gridH : USize) (gridW : USize) (perCell : USize) (numClasses : USize)
    (hflipProb : Float) (cropProb : Float) (cropMinScale : Float)
    (seed : USize)
    : IO (ByteArray × ByteArray × ByteArray)

/-- Photometric HSV jitter (YOLO-style) for a normalized image batch `[B,C,H,W]`.
    Three multiplicative gains `1 + U(-1,1)·gain` on hue (mod 360°), saturation,
    and value, one draw per image. Image-only — touches no labels, so it composes
    with any detector target. `imagenetNorm=1` de-norms to [0,1] sRGB, applies the
    gains in HSV space, clamps, and re-norms (the FPN loader stores images
    ImageNet-normalized). Returns a fresh image batch; the input is not modified. -/
@[extern "lean_f32_hsv_jitter"]
opaque hsvJitter (images : @& ByteArray) (batch channels height width : USize)
    (hGain sGain vGain : Float) (imagenetNorm : USize) (seed : USize) : IO ByteArray

/-- Horizontal flip of an FPN image `[B,C,H,W]` + its flat `[P3|P4|P5]`
    multi-scale target, one p=`prob` coin per image. A flip is shape-invariant, so
    every GT keeps its scale AND best-shape anchor: the image and each scale's grid
    mirror columns, and the in-cell x-offset `tx` becomes `1-tx` on assigned cells
    (obj=1). This matches `encode_targets_fpn` exactly — no re-encode from boxes is
    needed (the FPN record stores none). `scalesFlat` is int32-LE pairs `[g_s, A_s]`
    per scale; perAnchor=15 fixed. Returns `(image', target')` as fresh ByteArrays. -/
@[extern "lean_f32_fpn_hflip"]
opaque fpnHflip (images target : @& ByteArray)
    (batch channels height width : USize)
    (scalesFlat : @& ByteArray) (nScales : USize) (prob : Float) (seed : USize)
    : IO (ByteArray × ByteArray)

/-- Convert a uint8 mask ByteArray (one byte per pixel) into a little-endian
    int32 ByteArray of 4× the size. Pets `loadPets` returns masks as packed
    uint8; `trainStepAdamF32Seg` expects int32 per-pixel class labels. -/
@[extern "lean_f32_mask_u8_to_i32"]
opaque maskU8ToI32 (mask : @& ByteArray) : IO ByteArray

/-- Paired horizontal flip for segmentation: flips the f32 image `[B,C,H,W]`
    and the uint8 per-pixel mask `[B,H,W]` together, one coin per image, so the
    pixel correspondence survives. A flip is a pure column permutation, so the
    mask stays exact (no label interpolation). Returns `(image', mask')`.
    See `lean_f32_seg_hflip_pair`. -/
@[extern "lean_f32_seg_hflip_pair"]
opaque segHflipPair (img mask : @& ByteArray)
    (batch channels height width seed : USize) : IO (ByteArray × ByteArray)

/-- Per-batch segmentation confusion matrix. `logits` is f32 `[B,NC,H,W]`,
    `masks` is u8 `[B,H,W]` (per-pixel class). Returns int64 LE `[NC*NC]`
    counts `conf[true*NC + pred]` (argmax over channels), for mIoU
    accumulation across batches. planning/unet_demo_v2.md Workstream A. -/
@[extern "lean_f32_seg_confusion"]
opaque segConfusion (logits masks : @& ByteArray)
    (B NC H W : USize) : IO ByteArray

/-- Convert little-endian int32 token IDs to f32, element for element.
    Feeds the `idsInput` tokenPositionEmbed path: model input is `[B, T]`
    f32 ids, one-hot built in-graph — the host-side `[B, V·T]` one-hot
    buffer disappears. Exact for ids < 2²⁴. -/
@[extern "lean_f32_ids_to_floats"]
opaque idsToFloats (ids : @& ByteArray) : IO ByteArray

/-- Shuffle images and labels in-place (Fisher-Yates), applying the SAME
    permutation to both. Returns (shuffled images, shuffled labels).

    `labelBytes` is the label's bytes per record — 4 for a classification
    scalar, but a whole tensor for detection/segmentation (the FPN detector's
    is 185220 floats = 740880 bytes). It used to be hardcoded to 4 in the FFI,
    which permuted the images while leaving multi-float targets in place and so
    destroyed the image/target pairing every epoch on every detector and
    segmentation trainer. Pass `dio.labelBytesPerRecord`; never a literal. -/
@[extern "lean_f32_shuffle"]
opaque shuffle (images : ByteArray) (labels : ByteArray)
    (n : USize) (pixelsPerImage : USize) (labelBytes : USize) (seed : USize)
    : IO (ByteArray × ByteArray)

/-- Affine map of every element: `out[i] = scale * in[i] + shift`.
    Used to center [0,1] data to [-1,1] (scale=2, shift=-1) for DDPM
    training, and to invert for rendering. -/
@[extern "lean_f32_scale_shift"]
opaque scaleShift (ba : @& ByteArray) (scale : Float) (shift : Float) : IO ByteArray

/-- EMA update: running = (1-momentum)*running + momentum*batch. -/
@[extern "lean_f32_ema"]
opaque ema (running : @& ByteArray) (batch : @& ByteArray) (momentum : Float) : IO ByteArray

/-- Per-image horizontal flip of an NCHW f32 batch (independent p=0.5
    coin per image). Plain image aug for unconditional DDPM —
    planning/ddpm_demo_v2.md Workstream B3. -/
@[extern "lean_f32_hflip_nchw"]
opaque hflipNCHW (images : @& ByteArray) (batch : USize) (channels : USize)
    (H : USize) (W : USize) (seed : USize) : IO ByteArray

/-- Random crop: batch of NCHW images from src_size to crop_size. -/
@[extern "lean_f32_random_crop"]
opaque randomCrop (images : @& ByteArray) (batch : USize) (channels : USize)
    (srcH : USize) (srcW : USize) (cropH : USize) (cropW : USize)
    (seed : USize) : IO ByteArray

/-- Deterministic center crop: same window (y0=x0=max/2) for every image in
    the batch. No RNG. Used as the augment=false preprocessing fallback. -/
@[extern "lean_f32_center_crop"]
opaque centerCrop (images : @& ByteArray) (batch : USize) (channels : USize)
    (srcH : USize) (srcW : USize) (cropH : USize) (cropW : USize) : IO ByteArray

/-- Random horizontal flip for a batch of NCHW images (50% per image). -/
@[extern "lean_f32_random_hflip"]
opaque randomHFlip (images : @& ByteArray) (batch : USize) (channels : USize)
    (height : USize) (width : USize) (seed : USize) : IO ByteArray

/-- Mixup (Zhang et al. 2017) — λ ~ Beta(α, α), x_mixed[i] =
    λ·x[i] + (1-λ)·x[π(i)]. Returns the mixed image batch.
    Pair with `mixupSoftLabels` using the SAME seed + alpha. -/
@[extern "lean_f32_mixup_images"]
opaque mixupImages (images : @& ByteArray) (batch : USize) (channels : USize)
    (height : USize) (width : USize) (alpha : Float) (seed : USize) : IO ByteArray

/-- Soft labels for the mixup. Pair with `mixupImages` (same seed + alpha).
    Output shape: [batch, nClasses] f32, with label smoothing applied. -/
@[extern "lean_f32_mixup_soft_labels"]
opaque mixupSoftLabels (intLabels : @& ByteArray) (batch : USize) (nClasses : USize)
    (alpha : Float) (smooth : Float) (seed : USize) : IO ByteArray

/-- CutMix (Yun et al. 2019) — paste a random rectangle from x[π(i)]
    onto x[i]. Pair with `cutmixSoftLabels` (same seed + alpha). -/
@[extern "lean_f32_cutmix_images"]
opaque cutmixImages (images : @& ByteArray) (batch : USize) (channels : USize)
    (height : USize) (width : USize) (alpha : Float) (seed : USize) : IO ByteArray

/-- KNN-Mixup — like Mixup but pair[i] is the nearest neighbor of i in
    pixel-space L2 distance, not a random permutation. Mixes each sample
    with its closest manifold sibling in the batch. Pair with
    `knnMixupSoftLabels` using SAME images + seed + alpha. -/
@[extern "lean_f32_knn_mixup_images"]
opaque knnMixupImages (images : @& ByteArray) (batch : USize) (channels : USize)
    (height : USize) (width : USize) (alpha : Float) (seed : USize) : IO ByteArray

/-- KNN-Mixup soft labels. Needs the original images to recompute the
    same KNN pairing the `_images` call used. -/
@[extern "lean_f32_knn_mixup_soft_labels"]
opaque knnMixupSoftLabels (intLabels : @& ByteArray) (images : @& ByteArray)
    (batch : USize) (nClasses : USize) (channels : USize)
    (height : USize) (width : USize) (alpha : Float) (smooth : Float)
    (seed : USize) : IO ByteArray

/-- Soft labels for CutMix. λ_actual is recomputed from rectangle area. -/
@[extern "lean_f32_cutmix_soft_labels"]
opaque cutmixSoftLabels (intLabels : @& ByteArray) (batch : USize) (nClasses : USize)
    (height : USize) (width : USize) (alpha : Float) (smooth : Float)
    (seed : USize) : IO ByteArray

/-- Random Erasing (Zhong et al. 2017) — with probability `prob`, fill a
    random rectangle (relative area 2–33%, aspect 0.3–3.3) with N(0,1)
    noise. Per-image independent. Labels unchanged. -/
@[extern "lean_f32_random_erasing"]
opaque randomErasing (images : @& ByteArray) (batch : USize) (channels : USize)
    (height : USize) (width : USize) (prob : Float) (seed : USize) : IO ByteArray

/-- RandAugment-Color (Cubuk et al. 2019, color-only subset). Per image,
    apply `nOps` random ops drawn from {identity, brightness, contrast,
    color, autocontrast} with magnitude `m` (0–10, paper default 9).
    Geometric ops (rotate / shear / translate) are TODO.

    `imagenetNorm = 1` tells the kernel the incoming images are
    ImageNet-mean/std normalized (Imagenette / Imagewoof); the kernel
    de-normalizes to [0,1] sRGB, applies ops, then re-normalizes. Pass
    `0` for already-in-[0,1] datasets (CIFAR, MNIST). -/
@[extern "lean_f32_rand_augment"]
opaque randAugment (images : @& ByteArray) (batch : USize) (channels : USize)
    (height : USize) (width : USize) (nOps : USize) (m : Float)
    (imagenetNorm : USize) (seed : USize) : IO ByteArray

/-- EMA on squared values: out = (1−mom)·running + mom·batch². Used by
    SWAG to maintain a running E[θ²] alongside SWA's running E[θ]. -/
@[extern "lean_f32_ema_sq"]
opaque emaSq (running : @& ByteArray) (batch : @& ByteArray) (momentum : Float) : IO ByteArray

/-- Element-wise subtract: a − b. Used by SWAG for per-epoch deviation
    snapshots `p − swaMean`. -/
@[extern "lean_f32_subtract"]
opaque subtract (a : @& ByteArray) (b : @& ByteArray) : IO ByteArray

/-- Load a flat int32 LE token-stream file (e.g. `data/shakespeare/train.bin`).
    Returns (raw token bytes, token count). -/
@[extern "lean_f32_load_token_stream"]
opaque loadTokenStream (path : @& String) : IO (ByteArray × USize)

/-- Sample `batch` random sequences of length `seqLen` from a token stream.
    Returns a single flat ByteArray of size `2 * batch * seqLen * 4`
    containing input IDs followed by next-token target IDs (both int32 LE). -/
@[extern "lean_f32_sample_chunks"]
opaque sampleChunks (tokens : @& ByteArray) (nTokens batch seqLen seed : USize)
    : IO ByteArray

/-- One-hot encode `[batch, seqLen]` int32 token IDs into a flat f32
    tensor of shape `[batch, seqLen * vocab]` row-major in (b, t, v). -/
@[extern "lean_f32_token_one_hot"]
opaque tokenOneHot (ids : @& ByteArray) (batch seqLen vocab : USize) : IO ByteArray

/-- GradCAM closed-form (Zhou 2016 CAM). For nets ending GAP+dense,
    `heat[i,j] = ReLU(Σ_k W[k, tgt] · A[k, i, j])`, max-normalized to
    [0, 1]. `denseW` is `[C, NC]` row-major, `lastConv` is `[B, C, H, W]`
    NCHW. Returns `[H, W]` f32 for the chosen `batchIdx`. -/
@[extern "lean_f32_cam_compute"]
opaque camCompute (denseW : @& ByteArray) (lastConv : @& ByteArray)
    (batchIdx : USize) (C H W NC : USize) (tgt : USize) : IO ByteArray

/-- Recompute logits from a pre-GAP activation. Returns `[NC]` f32 for
    a single image (batchIdx). Used so the GradCAM exe can pick a class
    via argmax without running the full forward a second time. -/
@[extern "lean_f32_cam_logits"]
opaque camLogits (denseW : @& ByteArray) (denseB : @& ByteArray)
    (lastConv : @& ByteArray) (batchIdx : USize) (C H W NC : USize)
    : IO ByteArray

/-- Bilinear upsample a single 2D plane `[Hin, Win]` to `[Hout, Wout]`,
    align-corners. Returns the upsampled f32 ByteArray. -/
@[extern "lean_f32_bilinear_upsample_2d"]
opaque bilinearUpsample2D (img : @& ByteArray)
    (Hin Win Hout Wout : USize) : IO ByteArray

/-- SWAG sample weights (Maddox et al. 2019). Given the SWA mean,
    SWA-of-θ² (for diagonal variance), and the K most-recent per-epoch
    deviation snapshots packed row-major as `K × nParams` f32, draw
    one sample from the SWAG posterior `N(μ, ½Σ_diag + ½ Σ_low)`. -/
@[extern "lean_f32_swag_sample"]
opaque swagSample (swaMean : @& ByteArray) (swaSq : @& ByteArray)
    (deviations : @& ByteArray) (nParams : USize) (k : USize)
    (seed : USize) : IO ByteArray

end F32
