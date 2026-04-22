import Jax

def bneckNet : NetSpec where
  name   := "vjp-oracle-bottleneck"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 8 3 1 .same,
    .bottleneckBlock 8 8 1 1,
    .flatten,
    .dense 6272 10 .identity
  ]

def vjpCfg : TrainConfig where
  learningRate := 0.001
  batchSize    := 4
  epochs       := 1
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false

#eval bneckNet.validate!

def main (args : List String) : IO Unit :=
  runJax bneckNet vjpCfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_bottleneck.py"
