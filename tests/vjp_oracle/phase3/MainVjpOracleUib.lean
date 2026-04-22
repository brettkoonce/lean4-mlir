import LeanMlir

/-! VJP oracle: uib — MobileNet V4 Universal Inverted Bottleneck.
    Optional pre-DW + 1×1 expand + optional post-DW + 1×1 project.
    Here we use preDW=3, postDW=5 (ExtraDW config) so both conditional
    paths fire. -/

def uibNet : NetSpec where
  name   := "vjp-oracle-uib"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 4 3 1 .same,          -- stem
    .uib 4 4 2 1 3 5,                -- ic=4, oc=4, expand=2, stride=1, preDW=3, postDW=5
    .flatten,
    .dense 3136 10 .identity
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
  uibNet.train vjpCfg (args.head?.getD "data") .mnist
