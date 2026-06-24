import LeanMlir

/-! MNIST linear classifier — phase-3 unified training pipeline.

    Single dense layer (784 → 10, identity activation). The smallest
    network in the book — every gradient is one outer product, every
    backward step traceable on one page. Used as the worked example in
    Ch 2 ("You Are Here") to introduce the `.train` pipeline before
    layering hidden units / nonlinearities on top.

    s4tfBaseline config: 12 epochs, SGD lr=0.1, batch=128, no weight
    decay, no cosine, no augmentation — matches the Ch 2 worked-example
    invocation `mnistLinear.train (s4tfBaseline 12) "data" .mnist`. -/

def mnistLinear : NetSpec where
  name := "MNIST-Linear"
  imageH := 28
  imageW := 28
  layers := [
    .dense 784 10 .identity
  ]

def mnistLinearConfig : TrainConfig where
  learningRate := 0.1
  batchSize    := 128
  epochs       := 12
  useAdam      := false
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false
  labelSmoothing := 0.0

def main (args : List String) : IO Unit :=
  mnistLinear.train mnistLinearConfig (args.head?.getD "data") .mnist
