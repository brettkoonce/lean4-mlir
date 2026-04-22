import LeanMlir

/-! VJP oracle: fusedMbConv — EfficientNet V2 block. k×k regular conv
    replaces (1×1 expand + k×k depthwise) of MBConv. Same axiom family,
    different composition — tests the path where the fused conv is an
    expanding convBn rather than a factored expand+DW pair. -/

def fusedMbNet : NetSpec where
  name   := "vjp-oracle-fused-mbconv"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 4 3 1 .same,                  -- stem
    .fusedMbConv 4 4 2 3 1 1 false,          -- ic=4, oc=4, expand=2, k=3, s=1, n=1, SE off
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
  fusedMbNet.train vjpCfg (args.head?.getD "data") .mnist
