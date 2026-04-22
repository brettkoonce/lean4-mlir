import LeanMlir

/-! CIFAR-10 plain CNN (no batch norm) — phase-3 unified pipeline.
    32×32 RGB, 10 classes.

    **Pedagogical trainer — read this if it's your first run.** This
    is the CIFAR CNN *before* batch norm, kept around as a pair with
    `MainCifarCnnBnTrain.lean` to show what BN buys you. Expect
    slower convergence, more sensitivity to learning rate, and a
    lower final accuracy ceiling (~75-80% val) vs the BN variant
    (~85%+). If your goal is "best CIFAR result," run
    `cifar-bn-train` instead; this file exists so the book chapter
    on BN can demonstrate the improvement side by side. -/

def cifarCnn : NetSpec where
  name := "CIFAR-10-CNN"
  imageH := 32
  imageW := 32
  layers := [
    .conv2d  3 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .conv2d 32 64 3 .same .relu,
    .conv2d 64 64 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense 4096 512 .relu,
    .dense 512 512 .relu,
    .dense 512 10 .identity
  ]

def cifarCnnConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 128
  epochs       := 25
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 2
  augment      := true

def main (args : List String) : IO Unit :=
  cifarCnn.train cifarCnnConfig (args.head?.getD "data") .cifar10
