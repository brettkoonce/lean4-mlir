import LeanMlir

/-! YOLOv1 on Pascal VOC 2007 with R34-ImageNet (1000-class) backbone bootstrap.

    Builds on `MainYolov1VocTrain` (Phase 2 + Phase 3 smoke trainer) by
    loading the pretrained R34 backbone weights from the 1000-class
    ImageNet checkpoint at `.lake/build/jax_r34_imagenet.bin` (produced by
    the `resnet34-imagenet` JAX trainer — 69.26% top-1, 30 epochs) into
    the first 21,284,672 floats of the YOLOv1 init. The YOLOv1 head
    (dense 25088→1470, ~37M params) keeps its He-init.

    BN running stats: there is no `_bn_stats.bin` companion for this
    checkpoint, so the loader falls back to zeros (fresh BN) — fine for
    fine-tuning, which retrains them. (It will log a one-off size-mismatch
    WARN as it tries the companion path; harmless.)

    Phase 4 of `planning/yolo_demo_v3.md`. Usage:
      lake build yolov1-voc-train-bootstrap
      IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 \
        .lake/build/bin/yolov1-voc-train-bootstrap data/voc2007       # ~6-10 hr

    Spec is the full ResNet-34 + YOLOv1 head from Phase 1 (D11). batch=16
    for proper BN-stat estimation; cosine LR + warmup; full augment.

    Math for the bootstrap prefix:
      R34-ImageNet total = 21,797,672 floats
      R34-ImageNet head  = dense 512×1000 + bias = 513,000 floats
      Backbone           = 21,797,672 − 513,000 = **21,284,672** floats
    YOLOv1 spec has the same backbone (first 21,284,672 floats) before
    the new dense 25088→1470 head, so prefix = 21,284,672. (Coincidentally
    identical to the old Imagenette prefix, since 21,289,802 − 5,130 also
    = 21,284,672.) -/

def r34Yolov1 : NetSpec where
  -- Deep detection head (attempt #1): the 1×1-linear conf head collapsed to a
  -- center-prior by ~e16 even with focal — a single linear per-cell readout
  -- can't decode "person here" from the 512-d feature column, and WD zeros its
  -- sparse-signal weights (verdict in planning/yolo_v5.md). Replace it with a
  -- 3×3 conv 512→256 + ReLU (nonlinear capacity + 3×3 spatial context) then a
  -- 1×1 → 30. New name → own checkpoints. Backbone prefix (21,284,672) unchanged
  -- (the two head convs are He-init, after the prefix).
  name := "ResNet-34 + YOLOv1 deep-head (person VOC)"
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
  weightDecay  := 0.0          -- attempt #1: WD off. WD 5e-4 zeroed the conf head's
                               -- input-dependent weights faster than the ~2 person
                               -- cells/image could build them → flat center-prior.
  cosineDecay  := true
  warmupEpochs := 3
  gradClipNorm := 4.0          -- global-L2-norm gradient clip (essential at this LR)
  headLrMult   := 1.0          -- uniform LR: the conv head is small (~15k params) and conv
                               -- structure localizes naturally, so no dense-head boost needed.
                               -- (headLrMult only applies to .dense layers anyway.) Run on the
                               -- person-only data dir (data/voc2007_person).
  checkpointEveryNEpochs := 2  -- frequent ckpts: mars segfaults ~ep4-11; auto-resume needs recent state
  augment      := true        -- Phase 3: bbox-aware hflip + random crop
  lossKind     := LossKind.yolov1Masked
  -- Focal-BCE objectness (planning/yolo_v5.md §3): sigmoid + focal-BCE on the
  -- conf channel instead of raw-MSE, so the ~1-2 foreground cells aren't drowned
  -- by ~47 background cells. This is the fix for the objectness collapse-to-
  -- center-prior the conv head otherwise decays into by ~ep20. γ=2 (RetinaNet).
  useFocal     := true
  focalGamma   := 2.0
  bootstrapBackbone := some (".lake/build/jax_r34_imagenet.bin", 21284672)

def main (args : List String) : IO Unit := do
  -- Person-only data dir (data/voc2007_person) is the v5 recipe; pass it as argv[0].
  let dataDir := args.head?.getD "data/voc2007_person"
  IO.println s!"YOLOv1 VOC trainer with R34 bootstrap (focal objectness) — data dir: {dataDir}"
  r34Yolov1.train r34Yolov1BootstrapConfig dataDir DatasetKind.pascalVoc
