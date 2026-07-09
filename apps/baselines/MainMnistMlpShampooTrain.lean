import LeanMlir

/-! MNIST MLP with the **Shampoo** optimizer (UNVERIFIED perf path) — the
    fast smoke test for the Shampoo codegen (`planning/shampoo.md`).

    Same architecture + recipe as `MainMnistMlpTrain` (the AdamW baseline), so
    the two are a compute-matched A/B. The middle `dense 512 512` weight is
    SQUARE, so it is routed to Shampoo (Kronecker `L^{-1/4}·G·R^{-1/4}`, L/R in
    the m/v slots); the `784×512` and `512×10` weights are non-square and the
    biases are 1-D, so they fall back to AdamW. Signature-identical to the Adam
    module (Shampoo's L/R reuse the m/v slots for the square weight), so it
    drives through the existing Adam FFI — hence `useAdam := true` AND
    `useShampoo := true`. Fast enough to compile in seconds — the smoke check
    that validates the codegen before the ViT A/B. -/

def mnistMlp : NetSpec where
  name := "MNIST-MLP-Shampoo"
  imageH := 28
  imageW := 28
  layers := [
    .dense 784 512 .relu,
    .dense 512 512 .relu,   -- SQUARE → Shampoo
    .dense 512  10 .identity
  ]

def mnistMlpShampooConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 128
  epochs       := 12
  useAdam      := true      -- host state path (m+v, %t) — the Shampoo module is Adam-signature
  useShampoo   := true      -- codegen routes SQUARE 2D weights to Shampoo
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 1
  augment      := false

def main (args : List String) : IO Unit :=
  mnistMlp.train mnistMlpShampooConfig (args.head?.getD "data") .mnist
