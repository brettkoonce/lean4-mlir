import LeanMlir

/-! YOLOv1 cat/dog head detector on Oxford-IIIT Pets, R34-ImageNet backbone bootstrap.

    A ResNet-34 backbone + a deep convolutional detection head, trained on Pets
    head boxes tiled into 2×2 mosaics (see `preprocess_pets_mosaic.py`). The
    backbone weights load from the 1000-class ImageNet checkpoint at
    `.lake/build/jax_r34_imagenet.bin` (the `resnet34-imagenet` JAX trainer —
    69.26% top-1, 30 epochs) into the first 21,284,672 floats of the init; the
    deep head (conv 512→256 3×3 → 256→30 1×1) keeps its He-init.

    BN running stats: no `_bn_stats.bin` companion for this checkpoint, so the
    loader falls back to zeros (fresh BN) — fine for fine-tuning, which retrains
    them. (One-off size-mismatch WARN as it tries the companion path; harmless.)

    See `planning/yolo_final.md`. Usage:
      lake build yolov1-pets-train-bootstrap
      ./download_pets.sh && python3 preprocess_pets_mosaic.py data/pets data/pets_mosaic_bal
      IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 \
        .lake/build/bin/yolov1-pets-train-bootstrap data/pets_mosaic_bal

    Bootstrap prefix: R34-ImageNet total 21,797,672 floats − the 512×1000 head
    (513,000) = **21,284,672** backbone floats; the YOLOv1 spec shares that
    backbone before its detection head, so the bootstrap prefix = 21,284,672. -/

def r34Yolov1 : NetSpec where
  -- Deep detection head: a single 1×1-linear conf head can't decode "object
  -- here" from the 512-d feature column (and weight decay zeros its sparse-signal
  -- weights). A 3×3 conv 512→256 + ReLU (nonlinear capacity + spatial context)
  -- then 1×1 → 30 gives it the read it needs. Backbone prefix (21,284,672)
  -- unchanged (the two head convs are He-init, after the prefix). See
  -- planning/yolo_final.md for why localization is really a data problem (mosaic).
  name := "ResNet-34 + YOLOv1 deep-head (Pets)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .conv2d 512 256 3 .same .relu,      -- deep head L1: 3×3, nonlinear, spatial context
    .conv2d 256 30 1 .same .identity,   -- deep head L2: 1×1 → [B,30,7,7]
    .flatten                            -- → [B,1470] for the YOLOv1 masked loss
  ]

def r34Yolov1BootstrapConfig : TrainConfig where
  -- Fresh bootstrap (R34 backbone prefix + He-init head), full-finetune. The
  -- He-init head needs a hotter LR than a plain pretrained fine-tune, which
  -- diverges without gradient clipping (raw LR 7e-4 spiked the loss to ~66 and
  -- 1.5e-3 plateaued worse). gradClipNorm tames the early large-gradient
  -- batches → 7e-4 trains stably (val: 271→~10 over 3 epochs, no spike at peak
  -- LR). See the grad-clip codegen in LeanMlir/MlirCodegen.lean.
  learningRate := 7.0e-4
  batchSize    := 16
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.0          -- WD off: WD 5e-4 zeroed the conf head's input-dependent
                               -- weights faster than the ~2 foreground cells/image could
                               -- build them → flat center-prior.
  cosineDecay  := true
  warmupEpochs := 3
  gradClipNorm := 4.0          -- global-L2-norm gradient clip (essential at this LR)
  headLrMult   := 1.0          -- uniform LR: the conv head is small and conv structure
                               -- localizes naturally, so no dense-head boost needed.
                               -- (headLrMult only applies to .dense layers anyway.)
  checkpointEveryNEpochs := 2  -- frequent ckpts: mars segfaults ~ep4-11; auto-resume needs recent state
  augment      := true        -- Phase 3: bbox-aware hflip + random crop
  lossKind     := LossKind.yolov1Masked
  -- Focal-BCE objectness (planning/yolo_final.md §3): sigmoid + focal-BCE on the
  -- conf channel instead of raw-MSE, so the ~1-2 foreground cells aren't drowned
  -- by ~47 background cells. This is the fix for the objectness collapse-to-
  -- center-prior the conv head otherwise decays into by ~ep20. γ=2 (RetinaNet).
  useFocal     := true
  focalGamma   := 2.0
  bootstrapBackbone := some (".lake/build/jax_r34_imagenet.bin", 21284672)

def main (args : List String) : IO Unit := do
  -- Balanced Pets mosaic dir (data/pets_mosaic_bal); pass another as argv[0].
  let dataDir := args.head?.getD "data/pets_mosaic_bal"
  IO.println s!"YOLOv1 Pets cat/dog detector with R34 bootstrap (focal objectness) — data dir: {dataDir}"
  r34Yolov1.train r34Yolov1BootstrapConfig dataDir DatasetKind.petsDet
