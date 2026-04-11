import LeanMlir

/-! CIFAR-10 plain CNN (no batch norm) — phase-3 unified pipeline.
    32×32 RGB, 10 classes. The "F32" in the filename is historical
    (predates the all-float32 pipeline). -/

def cifarCnn : NetSpec where
  name := "CIFAR-10-CNN"
  imageH := 32
  imageW := 32
  layers := [
    .conv2d  3 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .conv2d 32 64 3 .same .relu,
    .conv2d 64 64 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense 4096 512 .relu,
    .dense 512 512 .relu,
    .dense 512 10 .identity
  ]

def cifarCnnConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 128
  epochs       := 25
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 2
  augment      := true

def main (args : List String) : IO Unit :=
  cifarCnn.train cifarCnnConfig (args.head?.getD "data") .cifar10
