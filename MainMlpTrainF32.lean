import LeanMlir

/-! MNIST MLP — phase-3 unified training pipeline.
    Pure dense network, no convolutions, ~670K params. The "f32" in the
    filename is historical (it predates everything being f32). -/

def mnistMlp : NetSpec where
  name := "MNIST-MLP"
  imageH := 28
  imageW := 28
  layers := [
    .dense 784 512 .relu,
    .dense 512 512 .relu,
    .dense 512  10 .identity
  ]

def mnistMlpConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 128
  epochs       := 12
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 1
  augment      := false

def main (args : List String) : IO Unit :=
  mnistMlp.train mnistMlpConfig (args.head?.getD "data") .mnist
