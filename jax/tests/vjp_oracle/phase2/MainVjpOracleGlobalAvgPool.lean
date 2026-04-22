import Jax

def gapNet : NetSpec where
  name   := "vjp-oracle-global-avg-pool"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 4 3 1 .same,
    .globalAvgPool,
    .dense 4 10 .identity
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

#eval gapNet.validate!

def main (args : List String) : IO Unit :=
  runJax gapNet vjpCfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_global_avg_pool.py"
