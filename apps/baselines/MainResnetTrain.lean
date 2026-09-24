import LeanMlir
import LeanMlir.ReferenceNets

/-! ResNet-34 on Imagenette — full training pipeline.
    Generates train_step MLIR → compiles with IREE → Adam training loop.
    ~21.3M params, 224×224 input, 10 classes.

    Architecture and training recipe are the only things this file
    defines; everything else (param shapes, MLIR codegen, vmfb compile,
    init, training loop, val eval, save) lives in `LeanMlir/Train.lean`. -/

def resnet34Config : TrainConfig where
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
  ReferenceNets.resnet34.train resnet34Config (args.head?.getD "data/imagenette")
