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
      lake exe unet-brats-train [data/brats] [epochs] [ce|dice|dicece]

    The loss arg is the demo's ablation axis (planning/brats_demo.md
    Workstream B). Default is `dicece`; pass `ce` to reproduce the collapse.
-/

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
  -- 3rd arg picks the loss — the demo's ablation axis. Default `dicece`:
  -- per-pixel CE alone collapses every tumour class on this data (mIoU 0.243
  -- ≈ the trivial background-only predictor, see planning/brats_demo.md), so
  -- it is the pedagogical baseline rather than the recommended setting.
  let lossKind : LossKind :=
    if args.any (· == "ce") then .perPixelCE
    else if args.any (· == "dice") then .perPixelDice
    else .perPixelDiceCE
  IO.eprintln s!"  loss: {repr lossKind}"
  unetBrats.train { unetBratsConfig with epochs, lossKind }
    (args.head?.getD "data/brats") .brats
