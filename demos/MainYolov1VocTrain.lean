import LeanMlir

/-! YOLOv1 on Pascal VOC 2007 — Phase 2 smoke trainer.

    Loads real VOC 2007 data via `pascalVocIO` (preprocess via
    `preprocess_voc.py`), runs `NetSpec.train` end-to-end through the
    unified `compileVmfbs` + `runTraining` path (R1 enables this), and
    verifies the train step runs + loss drops on real data.

    Phase 2 ships WITHOUT bbox-aware augmentation (cfg.augment := false)
    and WITHOUT mAP evaluation (eval block is skipped for yolov1Masked).
    Both land in Phase 3 / Phase 5. See
    `planning/yolo_demo_v3.md`.

    Usage:
      preprocess_voc.py first to make data/voc2007/{train,val}.bin
      lake build yolov1-voc-train
      .lake/build/bin/yolov1-voc-train [data_dir]   -- default data/voc2007

    Spec is the full ResNet-34 + YOLOv1 head pinned by D11; backbone is
    untrained (He init — pretrained-weight loading is Phase 4 work).
    With He init on a 58M-param model on raw VOC, expect a noisy loss
    that trends down but doesn't reach paper-faithful mAP without
    transfer learning. The point of this trainer is "does the
    end-to-end pipeline run cleanly on real data". -/

def r34Yolov1 : NetSpec where
  name := "ResNet-34 + YOLOv1 head (VOC train)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,                  -- 2/2 IREE-compat, matches MainResnetTrain
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .flatten,
    .dense 25088 1470 .identity
  ]

def r34Yolov1Config : TrainConfig where
  learningRate := 1.0e-5     -- conservative for batch=1 Adam on a noisy loss landscape
  batchSize    := 1
  epochs       := 5
  useAdam      := true
  weightDecay  := 0.0
  -- Explicitly set lossKind for clarity (could also leave default and
  -- rely on DatasetKind.pascalVoc derivation in compileVmfbs).
  lossKind     := LossKind.yolov1Masked

def main (args : List String) : IO Unit := do
  let dataDir := args.head?.getD "data/voc2007"
  IO.println s!"YOLOv1 VOC trainer — data dir: {dataDir}"
  r34Yolov1.train r34Yolov1Config dataDir DatasetKind.pascalVoc
