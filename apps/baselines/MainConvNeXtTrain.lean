import LeanMlir
import LeanMlir.ReferenceNets

/-! ConvNeXt-Tiny on Imagenette — pure-CNN modernization recipe.

    Liu et al. 2022 — "A ConvNet for the 2020s". Compute ratio
    (3, 3, 9, 3), channels (96, 192, 384, 768), depthwise-7×7 + LN +
    inverted-bottleneck + GELU + LayerScale + residual blocks.
    ~28M params at 224×224, 10 classes.

    Stem uses BN-equivalent `convBn` (paper uses LN; same param count,
    same expressiveness). Each `convNextDownsample` is `LN + 2×2 conv
    stride 2`, dedicated between stages (not fused with the first block
    of a stage like ResNet). -/

/-- `ReferenceNets.convNextTinyGelu`, trained under the prefix `ConvNeXt-T`. -/
def convNextTiny : NetSpec := { ReferenceNets.convNextTinyGelu with name := "ConvNeXt-T" }

def convNextTinyConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 32
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 3
  augment      := true
  labelSmoothing := 0.1

def main (args : List String) : IO Unit :=
  convNextTiny.train convNextTinyConfig (args.head?.getD "data/imagenette")
