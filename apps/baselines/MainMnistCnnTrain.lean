import LeanMlir

/-! MNIST CNN — phase-3 unified training pipeline.
    Conv-BN-ReLU architecture, ~1.7M params, 28×28 grayscale input,
    10 classes. Trained via the same `LeanMlir.Train.train` function
    as ResNet/MobileNet/EfficientNet/ViT — just pointed at MNIST. -/

def mnistCnn : NetSpec where
  name := "MNIST-CNN"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense 3136 512 .relu,
    .dense 512 10 .identity
  ]

def mnistCnnConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 128
  epochs       := 15
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 1
  augment      := false

def main (args : List String) : IO Unit :=
  mnistCnn.train mnistCnnConfig (args.head?.getD "data") .mnist
