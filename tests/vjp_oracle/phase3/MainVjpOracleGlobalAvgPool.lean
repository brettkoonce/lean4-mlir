import LeanMlir

/-! VJP oracle: globalAvgPool — tests spatial-mean VJP in isolation. -/

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

def main (args : List String) : IO Unit :=
  gapNet.train vjpCfg (args.head?.getD "data") .mnist
