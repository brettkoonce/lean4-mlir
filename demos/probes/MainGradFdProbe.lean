import LeanMlir

/-! **Gradient bisection probe**: which layer type makes the analytic gradient
    disagree with the loss surface?

    Probing `r34UnetBrats` turned up a systematic ~15% gap between the analytic
    gradient (what backprop emits) and finite differences of the loss (ground
    truth). It is flat in ε from 1e-3 down to 3e-5, so it is not truncation
    error, and it appears in the skipless net too, so it predates the skip work.

    That measurement came from one 22M-parameter net with convBn, residual
    blocks, maxPool, bilinear upsample, channel concat and per-pixel CE all
    stacked together. It names no culprit.

    So: run the same check up a ladder of MNIST-scale specs that each add ONE
    ingredient. The rung where the gap first appears is the answer.

      mlp      dense only — no conv, no BN, no pooling
      conv     conv2d + dense (still no BN)
      pool     conv2d + maxPool + dense
      convbn   convBn + dense            ← BatchNorm enters
      res      convBn + residualBlock + dense
      gap      conv2d + globalAvgPool + dense

    **`mlp` is the control on the harness itself.** Its backward is the
    best-established path in the repo; if `mlp` shows a 15% gap then the probe
    methodology is what is broken, not the codegen, and everything downstream of
    that conclusion changes. Run it first and believe nothing until it passes.

    Each spec is deliberately small (seconds per step) so the whole ladder is
    minutes. Usage mirrors the BraTS probe — plain SGD, no decay/clip/warmup, so
    one step is exactly `θ' = θ − η·g` and `scripts/grad_fd_bisect.py` can
    recover `g` and difference the loss against it.

      lake exe grad-fd-probe <rung> <lr> [data]
-/

/-- Dense-only. The control: if this fails, suspect the probe, not the codegen. -/
def fdMlp : NetSpec where
  name := "fdprobe mlp"
  imageH := 28
  imageW := 28
  layers := [ .flatten, .dense 784 64 .relu, .dense 64 10 .identity ]

/-- Adds convolution, still no normalization. -/
def fdConv : NetSpec where
  name := "fdprobe conv"
  imageH := 28
  imageW := 28
  layers := [ .conv2d 1 8 3 .same .relu, .flatten, .dense 6272 10 .identity ]

/-- Adds max pooling — whose backward is a scatter through an argmax. -/
def fdPool : NetSpec where
  name := "fdprobe pool"
  imageH := 28
  imageW := 28
  layers := [ .conv2d 1 8 3 .same .relu, .maxPool 2 2, .flatten, .dense 1568 10 .identity ]

/-- Adds BatchNorm. In train mode the batch statistics are themselves functions
    of the parameters, which is the single most likely place for a backward to
    drop a term. -/
def fdConvBn : NetSpec where
  name := "fdprobe convbn"
  imageH := 28
  imageW := 28
  layers := [ .convBn 1 8 3 1 .same, .flatten, .dense 6272 10 .identity ]

/-- Adds a residual block — the skip-add join, and BN inside it. -/
def fdRes : NetSpec where
  name := "fdprobe res"
  imageH := 28
  imageW := 28
  layers := [ .convBn 1 8 3 1 .same, .residualBlock 8 8 1 1,
              .flatten, .dense 6272 10 .identity ]

/-- Adds global average pooling, the other reduction in the stack. -/
def fdGap : NetSpec where
  name := "fdprobe gap"
  imageH := 28
  imageW := 28
  layers := [ .conv2d 1 8 3 .same .relu, .globalAvgPool, .dense 8 10 .identity ]

/-- `gap` with NO ReLU. A central difference straddles a ReLU kink whenever a
    perturbation flips a unit across zero, and `fdGap` has only 170 parameters —
    a tiny true gradient, so a few flipped kinks are large in relative terms.
    Removing the nonlinearity makes the net piecewise-linear-free and isolates
    whether `globalAvgPool` itself is at fault or the kinks were. -/
def fdGapId : NetSpec where
  name := "fdprobe gapid"
  imageH := 28
  imageW := 28
  layers := [ .conv2d 1 8 3 .same .identity, .globalAvgPool, .dense 8 10 .identity ]

/-- `gap` with 8× the channels: same structure, much larger true gradient, so
    kink error shrinks in relative terms. -/
def fdGapBig : NetSpec where
  name := "fdprobe gapbig"
  imageH := 28
  imageW := 28
  layers := [ .conv2d 1 64 3 .same .relu, .globalAvgPool, .dense 64 10 .identity ]

/-- Residual block WITH a projection shortcut (ic ≠ oc, stride 2). `fdRes` uses
    8→8 stride 1, which takes the identity-skip branch and never exercises the
    1×1 projection convBn — and every downsampling stage of R34 has one. -/
def fdResProj : NetSpec where
  name := "fdprobe resproj"
  imageH := 28
  imageW := 28
  layers := [ .convBn 1 8 3 1 .same, .residualBlock 8 16 1 2,
              .flatten, .dense 3136 10 .identity ]

/-- Bilinear upsample — in every decoder of the BraTS nets, and absent from the
    rest of this ladder. -/
def fdUpsample : NetSpec where
  name := "fdprobe upsample"
  imageH := 28
  imageW := 28
  layers := [ .conv2d 1 4 3 .same .relu, .bilinearUpsample 2,
              .flatten, .dense 12544 10 .identity ]

/-- Deep ReLU stacks. Width REDUCED the kink error (`gapbig` < `gap`), but the
    BraTS net is deep rather than wide — 40+ ReLU layers. Each layer contributes
    its own set of near-zero units that a perturbation can flip, so if kink
    error compounds with depth these should degrade monotonically while every
    individual layer stays exact. That is the difference between "the codegen
    has a bug" and "finite differences do not work on deep ReLU networks". -/
def fdDeepN (n : Nat) : NetSpec where
  name := s!"fdprobe deep{n}"
  imageH := 28
  imageW := 28
  layers := [ .convBn 1 8 3 1 .same ]
            ++ (List.replicate n (Layer.convBn 8 8 3 1 .same))
            ++ [ .flatten, .dense 6272 10 .identity ]

def rungs : List (String × NetSpec) :=
  [ ("mlp", fdMlp), ("conv", fdConv), ("pool", fdPool)
  , ("convbn", fdConvBn), ("res", fdRes), ("gap", fdGap)
  , ("gapid", fdGapId), ("gapbig", fdGapBig)
  , ("resproj", fdResProj), ("upsample", fdUpsample)
  , ("deep2", fdDeepN 2), ("deep6", fdDeepN 6)
  , ("deep12", fdDeepN 12), ("deep24", fdDeepN 24) ]

def fdProbeConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 16
  epochs       := 1
  -- Everything that would bend the update away from θ − η·g is off, so the
  -- gradient is exactly recoverable from the parameter delta. Momentum is fine
  -- because its velocity starts at zero: step 1 is plain SGD either way.
  useAdam      := false
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  gradClipNorm := 0.0
  labelSmoothing := 0.0
  augment      := false
  evalEveryNEpochs := 0
  checkpointEveryNEpochs := 0

def main (args : List String) : IO Unit := do
  let rung := args[0]?.getD "mlp"
  let lrN := (args[1]?.bind String.toNat?).getD 10
  let dataDir := args[2]?.getD "data"
  match rungs.lookup rung with
  | none =>
    IO.eprintln s!"unknown rung {rung}; expected one of {rungs.map (·.1)}"
    IO.Process.exit 1
  | some spec =>
    IO.eprintln s!"  rung: {rung}"
    IO.eprintln s!"  artifacts: {(spec.withBuildTag "fd").buildPrefix}_*"
    (spec.withBuildTag "fd").train
      { fdProbeConfig with learningRate := lrN.toFloat * 0.0001 }
      dataDir .mnist
