import LeanMlir

/-! UNet on BraTS brain-tumour MRI (MSD Task01_BrainTumour), 2D axial slices.

    The segmentation demo one rung up the ladder from `MainUnetPetsTrain`.
    Same UNet, same skip codegen, same per-pixel CE — three things change,
    and each one is the point:

    * **4 input channels, not 3.** The channels are co-registered MRI
      modalities (FLAIR / T1w / T1gd / T2w), not RGB. The tumour sub-regions
      are *defined* by which modalities light up: enhancing tumour is bright
      on T1gd, edema is bright on FLAIR. So the input is genuinely
      multi-modal rather than three correlated views of one thing — the
      `.unetDown 3 32` -> `.unetDown 4 32` edit is the whole architectural
      diff.

    * **4 output classes, and they are brutally imbalanced.** Enhancing
      tumour is on the order of 1% of pixels. The pets demo already collapsed
      its thin class (boundary IoU 0.000 at 3 epochs, RESULTS.md) and we
      shrugged, because a trimap boundary is not what the demo was about.
      Here the thin classes *are* the task, so the collapse becomes
      unignorable — which is exactly why medical segmentation invented Dice.
      This demo is the rung where per-pixel CE is expected to break.

    * **240×240, not 224×224.** Native BraTS in-plane size. 240 = 16*15, so
      the depth-4 UNet's four halvings still divide evenly and nothing needs
      resizing.

    Data: ./download_brats.sh (MSD Task01, openly downloadable; BraTS 2021
    itself is gated behind a Synapse agreement). Volumes are split by patient
    before slicing — see preprocess_brats.py.

    Usage:
      ./download_brats.sh
      lake exe unet-brats-train [data/brats] [epochs] [ce|dice|dicece|wce|focal]

    The loss arg is the demo's ablation axis (planning/brats_demo.md
    Workstream B/B'). `ce` and `dicece` both reproduce the collapse — that is
    the finding, not a misconfiguration. `wce` and `focal` are the two arms
    that should not, and they fail differently if they fail:
    `scripts/seg_grad_scorecard.py` measures wce's advantage as a constant
    ~196× from step 0, and focal's as nil at init rising to ~4e6× once the net
    is confident — so focal's open question is whether it engages before the
    collapse is decided (~100 steps).
-/

/-- Inverse-frequency class weights, measured over `data/brats/train.bin` by
    `scripts/brats_class_weights.py` (14,415 slices; background 97.46% / edema
    1.60% / non-enhancing 0.44% / enhancing 0.50% of voxels).

    Why inverse frequency, stated plainly, because it is the demo's argument:
    it makes every class contribute **exactly 25%** of the loss. That is the
    goal soft Dice was reaching for — "a ratio per class, so every class carries
    equal weight no matter how few pixels it owns" — and Dice fails to deliver
    it, because its gradient carries a `p_i` factor and vanishes on precisely
    the class that has collapsed. CE's gradient is flat at `p → 0`. So this is
    Dice's own objective, pursued with a gradient that still exists where it
    matters. Under plain CE the corresponding shares are 97.46 / 1.60 / 0.44 /
    0.50 — background owns the loss, and predicting background is the cheapest
    descent direction available. Which is exactly what the net does.

    The stock objection to inverse frequency is that a ~200× dynamic range
    destabilizes training. That objection is about a `/N` reduction, where the
    weights inflate the gradient scale outright. `perPixelWeightedCE` divides by
    `Σ_k w_{y_k}` over the batch, so the loss stays a weighted *mean* — same
    scale as unweighted CE, self-normalizing per batch. -/
def unetBratsClassWeights : List Float :=
  [1.0, 60.9033, 220.0868, 195.5835]

def unetBrats : NetSpec where
  name := "UNet (BraTS, 240×240 4-modality MRI → 4-class tumour)"
  imageH := 240
  imageW := 240
  layers := [
    .unetDown 4   32,
    .unetDown 32  64,
    .unetDown 64  128,
    .unetDown 128 256,
    .convBn 256 512 3 1 .same,
    .convBn 512 512 3 1 .same,
    .unetUp 512 256,
    .unetUp 256 128,
    .unetUp 128 64,
    .unetUp 64  32,
    .conv2d 32 4 1 .same .identity
  ]

def unetBratsConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 16
  epochs       := 3
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false

def main (args : List String) : IO Unit := do
  -- Optional 2nd arg overrides epochs; the default 3 is a smoke test, same
  -- convention as unet-pets-train. mIoU + per-class IoU print every 10
  -- epochs and at the end.
  let epochs := (args[1]?.bind String.toNat?).getD unetBratsConfig.epochs
  -- 3rd arg picks the loss — the demo's ablation axis.
  --
  --   ce     collapses every tumour class (mIoU 0.243 ≈ the trivial
  --          background-only predictor, planning/brats_demo.md Workstream A)
  --   dicece collapses too, identically to four decimals — the Gate B result,
  --          and the reason `dicece` is no longer the default
  --   wce    amplify the rare class. Constant ~196× edge, live from step 0.
  --   focal  defund the easy majority. No edge at init, ~4e6× once confident.
  --
  -- `wce` is the default because it is the arm with no timing risk. The others
  -- are the pedagogy: run them to watch the collapse, or to find out whether
  -- focal's feedback is quick enough to prevent it.
  let lossKind : LossKind :=
    if args.any (· == "ce") then .perPixelCE
    else if args.any (· == "dicece") then .perPixelDiceCE
    else if args.any (· == "dice") then .perPixelDice
    else if args.any (· == "focal") then .perPixelFocalCE 2.0
    else .perPixelWeightedCE unetBratsClassWeights
  IO.eprintln s!"  loss: {repr lossKind}"
  unetBrats.train { unetBratsConfig with epochs, lossKind }
    (args.head?.getD "data/brats") .brats
