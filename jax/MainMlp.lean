import Jax

/-! MNIST MLP — S4TF book Ch. 1
    784 → 512 (ReLU) → 512 (ReLU) → 10 -/

def mnistMlp : NetSpec where
  name := "MNIST-MLP"
  imageH := 28
  imageW := 28
  layers := [
    .dense 784 512 .relu,
    .dense 512 512 .relu,
    .dense 512  10 .identity
  ]

-- Matches phase 3 MainMlpTrainF32.lean config so the two phases' traces
-- can be diffed against each other via tests/diff_traces.py.
def mnistConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 128
  epochs       := 12
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 1
  augment      := false

#eval mnistMlp.validate!

def main (args : List String) : IO Unit :=
  runJax mnistMlp mnistConfig .mnist (args.head? |>.getD "data") "generated_mnist_mlp.py"
