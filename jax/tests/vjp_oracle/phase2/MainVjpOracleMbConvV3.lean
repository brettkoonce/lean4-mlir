import Jax

def mbConvV3Net : NetSpec where
  name   := "vjp-oracle-mbconv-v3"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 4 3 1 .same,
    .mbConvV3 4 4 8 3 1 true true,
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

#eval mbConvV3Net.validate!

def main (args : List String) : IO Unit :=
  runJax mbConvV3Net vjpCfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_mbconv_v3.py"
