import LeanMlir

/-! CIFAR-10 BN-CNN — phase-3 unified training pipeline.
    32×32 RGB input, 10 classes, ~3.7M params. Reuses the same
    `LeanMlir.Train.train` function as the bigger Imagenette models —
    only the spec, the config, and the dataset enum differ. -/

def cifarCnn : NetSpec where
  name := "CIFAR-10-BN"
  imageH := 32
  imageW := 32
  layers := [
    .convBn  3 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense 4096 512 .relu,
    .dense 512 512 .relu,
    .dense 512 10 .identity
  ]

def cifarCnnConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 128
  epochs       := 30
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 2
  augment      := true

def main (args : List String) : IO Unit :=
  cifarCnn.train cifarCnnConfig (args.head?.getD "data") .cifar10
