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

/-- Argmax over 10 float32 values starting at element offset `off`. -/
@[extern "lean_f32_argmax10"]
opaque argmax10 (ba : @& ByteArray) (off : USize) : USize

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
    yLabels_concat, count)` where `yLabels_concat` carries 6076 bytes
    per image: 30×7×7 float32 target (5880 bytes), then 7×7 float32
    mask (196 bytes). The Lean dispatcher (`runTraining`) splits this
    into target + mask before calling `trainStepAdamF32Yolov1`. See
    `preprocess_pets_mosaic.py` for the on-disk format. -/
@[extern "lean_f32_load_voc"]
opaque loadDetBin (path : @& String) : IO (ByteArray × ByteArray × Nat)

/-- Split an interleaved YOLOv1 batch slice (per-record `[target||mask]`,
    6076 bytes/record) into separately-contiguous target + mask tensors
    suitable for the `trainStepAdamF32Yolov1` FFI. Returns
    `(target_concat, mask_concat)` with sizes `batch * 5880` and
    `batch * 196` bytes respectively. -/
@[extern "lean_voc_split_batch"]
opaque detSplitBatch (interleaved : @& ByteArray) (batch : USize)
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

/-- Convert a uint8 mask ByteArray (one byte per pixel) into a little-endian
    int32 ByteArray of 4× the size. Pets `loadPets` returns masks as packed
    uint8; `trainStepAdamF32Seg` expects int32 per-pixel class labels. -/
@[extern "lean_f32_mask_u8_to_i32"]
opaque maskU8ToI32 (mask : @& ByteArray) : IO ByteArray

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

/-- Shuffle images and labels in-place (Fisher-Yates). Returns (shuffled images, shuffled labels). -/
@[extern "lean_f32_shuffle"]
opaque shuffle (images : ByteArray) (labels : ByteArray)
    (n : USize) (pixelsPerImage : USize) (seed : USize) : IO (ByteArray × ByteArray)

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
