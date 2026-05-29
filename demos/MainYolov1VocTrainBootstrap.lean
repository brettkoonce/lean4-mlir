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
  name := "ResNet-34 + YOLOv1 head (VOC bootstrap)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .flatten,
    .dense 25088 1470 .identity
  ]

def r34Yolov1BootstrapConfig : TrainConfig where
  -- Lower LR since we're fine-tuning a pretrained backbone; full-finetune
  -- the whole network (no freeze schedule for v3 simplicity).
  learningRate := 1.0e-4
  batchSize    := 16
  epochs       := 80
  useAdam      := true
  weightDecay  := 5.0e-4
  cosineDecay  := true
  warmupEpochs := 3
  augment      := true        -- Phase 3: bbox-aware hflip + random crop
  lossKind     := LossKind.yolov1Masked
  bootstrapBackbone := some (".lake/build/jax_r34_imagenet.bin", 21284672)

def main (args : List String) : IO Unit := do
  let dataDir := args.head?.getD "data/voc2007"
  IO.println s!"YOLOv1 VOC trainer with R34 bootstrap — data dir: {dataDir}"
  r34Yolov1.train r34Yolov1BootstrapConfig dataDir DatasetKind.pascalVoc
