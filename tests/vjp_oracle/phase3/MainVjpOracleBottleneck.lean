import LeanMlir

/-! VJP oracle: bottleneckBlock — ResNet-50 building block.
    1×1 reduce + 3×3 + 1×1 expand + skip. Tests the same biPath VJP
    as residual but through a 3-conv composition. -/

def bneckNet : NetSpec where
  name   := "vjp-oracle-bottleneck"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 8 3 1 .same,           -- stem: 8×28×28 (oc must be divisible by 4)
    .bottleneckBlock 8 8 1 1,         -- mid = 2, 1 block, no proj
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

def main (args : List String) : IO Unit :=
  bneckNet.train vjpCfg (args.head?.getD "data") .mnist
