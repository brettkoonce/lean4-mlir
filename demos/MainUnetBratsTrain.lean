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
      Here the thin classes *are* the task, so a collapse would be
      unignorable — which is why medical segmentation invented Dice.

      ⚠ Historical note, because this demo's original thesis was exactly the
      opposite: it was built to show per-pixel CE breaking here, and for a
      while it appeared to. That collapse was a data bug (see the loss-arm
      commentary in `main`). On correct data every arm segments and plain CE
      is competitive, so the imbalance is real but it is not fatal at this
      scale.

    * **240×240, not 224×224.** Native BraTS in-plane size. 240 = 16*15, so
      the depth-4 UNet's four halvings still divide evenly and nothing needs
      resizing.

    Data: ./download_brats.sh (MSD Task01, openly downloadable; BraTS 2021
    itself is gated behind a Synapse agreement). Volumes are split by patient
    before slicing — see preprocess_brats.py.

    Usage — turn-key. Once the data is prepared, no arguments are needed:

      ./download_brats.sh
      python3 preprocess_brats.py data/brats/Task01_BrainTumour data/brats
      lake exe unet-brats-train

    That trains 3 epochs of `dicece` on data/brats and prints mIoU + per-class
    IoU + region Dice (WT/TC/ET) every epoch. Expect val mIoU ≈ 0.73,
    WT Dice ≈ 0.90 at 3 epochs.

    Everything else is optional and order-free — a data dir, an epoch count,
    and an arm can appear in any order:

      lake exe unet-brats-train ce 10
      lake exe unet-brats-train data/brats 3 dice

    Arms:      ce | dice | dicece | wce | wcesqrt | wceb b=<pct> | focal g=<n>
    Modifiers: pb (prior-bias init) · cos (cosine LR) · aug (paired hflip)
               lr=<n, in units of 1e-4> · tag=<s> (keep two runs of one arm
               from clobbering each other's checkpoints)
-/

/-- Inverse-frequency class weights, measured over `data/brats/train.bin` by
    `scripts/brats_class_weights.py` (14,415 slices; background 97.46% / edema
    1.60% / non-enhancing 0.44% / enhancing 0.50% of voxels).

    Inverse frequency makes every class contribute **exactly 25%** of the
    loss; under plain CE the shares are instead 97.46 / 1.60 / 0.44 / 0.50.

    ⚠⚠ The argument this docstring used to make — that Dice cannot deliver the
    same thing because its gradient carries a `p_i` factor and vanishes on a
    collapsed class — is RETRACTED (2026-08-26). It was inferred from runs on
    mismatched image/mask data. Re-measured on correct data, `dice` and
    `dicece` are the two BEST arms (mIoU 0.736 / 0.734) and this
    inverse-frequency `wce` is the WORST and the only unstable one (0.640, and
    it over-paints 1.5–2.4×). Equalizing the loss shares turns out to be an
    over-correction here, not the fix. Kept as a selectable arm and as the
    β = 1 endpoint of `unetBratsClassWeightsBeta`; no longer the default.

    The stock objection to inverse frequency is that a ~200× dynamic range
    destabilizes training. That objection is about a `/N` reduction, where the
    weights inflate the gradient scale outright. `perPixelWeightedCE` divides by
    `Σ_k w_{y_k}` over the batch, so the loss stays a weighted *mean* — same
    scale as unweighted CE, self-normalizing per batch. -/
def unetBratsClassWeights : List Float :=
  [1.0, 60.9033, 220.0868, 195.5835]

/-- Inverse-**sqrt**-frequency weights, same histogram (`brats_class_weights.py`).
    The fallback this doc named before the run, and the run made necessary.

    `unetBratsClassWeights` (inverse frequency) equalizes the per-class loss
    share at 25% each, and at matched 10 epochs that **inverted the collapse**:
    99.39% enhancing recall at 1.78% precision, painting 28.95% of every brain
    as tumour against a 0.52% truth — a 56× over-prediction (Gate B' result).

    The cause is an exchange rate, not a bug. `w₃/w₀ = 195.6` prices one false
    negative on enhancing tumour at 196 false positives on background, so a
    capacity-limited net rationally over-predicts. These weights drop that rate
    to **13.99 : 1** and tumour's share of the loss from 75% to ~21% — still an
    enormous thumb on the scale versus CE's 0.50%, without instructing the net
    that a miss costs two hundred false alarms.

    The falsifiable prediction: precision comes off 1.78% without recall
    returning to CE's 0.00%. If it lands in between, the knob is real and the
    demo has an axis. If it snaps back to the collapse, the whole
    amplify-the-minority family is a false lead and focal is the story. -/
def unetBratsClassWeightsSqrt : List Float :=
  [1.0, 7.8041, 14.8353, 13.9851]

/-- The measured class prior over `data/brats/train.bin` — background / edema /
    non-enhancing / enhancing, as fractions of all voxels
    (`scripts/brats_class_weights.py`). `unetBratsClassWeights` is literally the
    reciprocal of this; both come from the same histogram.

    Fed to `TrainConfig.headPriorBias` (the `pb` flag) to start the head at
    `log π_c` instead of a uniform softmax. Needs no normalization — a constant
    added to every logit is a no-op under softmax. -/
def unetBratsClassPriors : List Float :=
  [0.9746, 0.0160, 0.0044, 0.0050]

/-- Class weights as `π_c^(-β)`, normalized to `w₀ = 1`. **The knob that
    unifies every weighted-CE arm into one axis**, which is what the ablation
    turned the discrete arms into:

      β = 0    → all ones = plain CE            (mIoU 0.728)
      β = 0.5  → `unetBratsClassWeightsSqrt`    (mIoU 0.709)
      β = 1    → `unetBratsClassWeights`        (mIoU 0.640, over-paints)

    ⚠ Re-measured 2026-08-26 on fixed data, and the axis now runs the other
    way: β = 0 is the *best* of the three and increasing β monotonically hurts.
    The previous reading (β = 0 collapses, 0.5 finds the tumour, 1 over-
    predicts) came from runs on mismatched image/mask pairs. Focal's
    "collapse at every γ" is likewise retracted — `focal g=2` scores 0.719.
    The axis is still real and still worth sweeping; its sign is just not what
    this demo originally reported. -/
def unetBratsClassWeightsBeta (beta : Float) : List Float :=
  let w0 := Float.exp (-beta * Float.log unetBratsClassPriors.head!)
  unetBratsClassPriors.map (fun p => Float.exp (-beta * Float.log p) / w0)

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
  -- Eval EVERY epoch, against the framework default of 10. An epoch here is
  -- ~40 min and the eval is a forward pass over 2,569 val slices — a couple of
  -- minutes, call it 5% overhead. That is cheap insurance for the thing this
  -- demo exists to measure: the per-class IoU is the ONLY instrument that can
  -- see a collapsed class (the loss curve provably cannot — Workstream A), so
  -- at the default cadence a 10-epoch arm reports nothing until it is over and
  -- a collapse is indistinguishable from progress for seven hours.
  evalEveryNEpochs := 1

/-- ⛔ RETRACTED 2026-08-26 — this docstring used to carry a "cost of the class
    prior relative to uniform" table and called it **"the single number that has
    predicted every arm we have run"**, monotone across all four arms.

    It predicted nothing. Two independent strikes:

    1. It was already refuted in July by `focal g=8`, whose ratio (1.30×) is
       indistinguishable from `wcesqrt`'s (1.35×) while landing on the opposite
       outcome. `planning/brats_demo.md` recorded that refutation; this comment
       was never updated, and went on asserting the claim for a month.
    2. Its four "measured outcomes" are void anyway — all four arms trained on
       mismatched image/mask pairs. On correct data every arm segments and the
       ordering the table predicted does not appear.

    Kept as a marker, because the failure mode is the reusable lesson: a tidy
    scalar fitted post-hoc to four points, promoted to a predictor, and left
    in the source after the doc that spawned it had already withdrawn it. -/
def main (args : List String) : IO Unit := do
  -- Turn-key arg parsing. Options are matched by keyword ANYWHERE in `args`;
  -- the data dir and epoch count are then taken from whatever is left over, so
  -- all of these work and none of them need a manual:
  --   unet-brats-train | … ce | … 5 | … ce 5 | … data/brats 3 ce
  -- ⚠ Before this the data dir was `args[0]` unconditionally while the arms
  -- were matched with `args.any`, so the natural `unet-brats-train ce` silently
  -- looked for slices in a directory literally named "ce" and died there.
  let optionWords : List String :=
    ["ce", "dice", "dicece", "focal", "wce", "wcesqrt", "wceb", "pb", "cos", "aug"]
  let optionPrefixes : List String := ["g=", "b=", "lr=", "tag="]
  let positionals := args.filter (fun a =>
    !(optionWords.contains a || optionPrefixes.any (fun q => a.startsWith q)))
  -- Epoch count: the first bare number. Default 3 is a smoke test, same
  -- convention as unet-pets-train. mIoU + per-class IoU print EVERY epoch
  -- (`evalEveryNEpochs := 1` below) and at the end.
  let epochs := (positionals.findSome? String.toNat?).getD unetBratsConfig.epochs
  -- Data dir: the first non-numeric leftover.
  let dataDir := (positionals.find? (fun a => (String.toNat? a).isNone)).getD "data/brats"

  -- `g=<n>` sets focal's γ. Default 2 is the RetinaNet paper's, and on this
  -- data that is a **collapse setting**: γ=2 puts the prior/uniform ratio at
  -- 6.84×, barely off CE's 9.79×, and it collapsed exactly like CE. γ is the
  -- only knob focal has (α is deliberately omitted — that is wce's mechanism),
  -- and raising it walks the ratio down: γ=4 → 3.94×, γ=8 → 1.30×, γ=9 → 0.99×.
  -- So γ≈8 is where the framework says focal should land on wcesqrt's 1.35×,
  -- reaching the same place by defunding the majority instead of amplifying the
  -- minority. Running γ=2 and concluding "focal doesn't work" would have been
  -- testing the default, not the loss.
  let focalGamma : Float :=
    ((args.filter (·.startsWith "g=")).head?.bind
      (fun a => (a.drop 2).toNat?)).map Nat.toFloat |>.getD 2.0
  -- `b=<n>` sets the weighted-CE exponent β as a PERCENT (b=70 → β=0.70), so
  -- the `wceb` arm sweeps the one axis wave 2 established: β=0 collapse, 0.5
  -- finds the tumour, 1.0 over-predicts. Percent because the arg parser has
  -- only `toNat?` (Lean core has no `String.toFloat?`).
  let wceBeta : Float :=
    (((args.filter (·.startsWith "b=")).head?.bind
      (fun a => (a.drop 2).toNat?)).map Nat.toFloat |>.getD 70.0) / 100.0
  -- The loss arg picks the arm — the demo's ablation axis.
  --
  -- ⭐ Measured 2026-08-26, 3 epochs each on fixed data, best val mIoU:
  --
  --   dice 0.736 · dicece 0.734 · ce 0.728 · focal 0.719 · wcesqrt 0.709
  --   wce 0.640
  --
  -- Five of the six land in a tight band; only `wce` (β=1) separates, and it
  -- is also the only arm with an unstable trajectory (0.640 → 0.494 → 0.611,
  -- where every other arm is monotone). It over-paints by 1.5–2.4×.
  --
  -- ⚠⚠ This REVERSES the pre-2026-07-22 finding that `ce`/`dicece`/`focal`
  -- collapse to a trivial predictor. Those runs trained on mismatched
  -- image/mask pairs: `lean_f32_shuffle` permuted images by a full record but
  -- labels by a hardcoded 4 bytes, and a BraTS label is 240². Fixed in
  -- `430ba2c`/`ca83835`. Every collapse this demo was built to exhibit was an
  -- artifact of that bug — see the STOP banner in planning/brats_demo.md and
  -- planning/post_shuffle_fix.md §1a.
  --
  -- `dicece` is the default: joint-best WT Dice (0.903), best endpoint mIoU,
  -- the most monotone trajectory, and Dice+CE is the standard compound loss in
  -- medical segmentation. ⚠ `ce` and `dice` are within noise of it — this is a
  -- single-seed 3-epoch sweep, not a tuned comparison, so treat the ordering
  -- among the top five as unresolved.
  let lossKind : LossKind :=
    if args.any (· == "ce") then .perPixelCE
    else if args.any (· == "dicece") then .perPixelDiceCE
    else if args.any (· == "dice") then .perPixelDice
    else if args.any (· == "focal") then .perPixelFocalCE focalGamma
    else if args.any (· == "wcesqrt") then .perPixelWeightedCE unetBratsClassWeightsSqrt
    else if args.any (· == "wceb") then .perPixelWeightedCE (unetBratsClassWeightsBeta wceBeta)
    else if args.any (· == "wce") then .perPixelWeightedCE unetBratsClassWeights
    else .perPixelDiceCE
  IO.eprintln s!"  loss: {repr lossKind}"
  -- `pb` adds RetinaNet prior-bias init. Orthogonal to the loss on purpose —
  -- it composes with any arm, and `focal pb` is the pairing with an actual
  -- mechanistic argument behind it: focal is a no-op at a uniform softmax, and
  -- this is what gives it confidence to suppress at step 0.
  let priorBias := if args.any (· == "pb") then unetBratsClassPriors else []
  -- `cos` turns on cosine LR decay (to ~0 over the run). The weighted arms
  -- OSCILLATE at the constant 0.001 default — wcesqrt rotated which class it
  -- predicted every epoch (enhancing @1-2, edema @3-4) with WT Dice bouncing
  -- 0.40/0.51/0.045: heavy weights carve a narrow basin and a flat LR
  -- slingshots through it. Cosine decay is the standard damping — high LR early
  -- to find the basin, low LR late to settle in it.
  let cosine := args.any (· == "cos")
  -- `aug` turns on mask-aware augmentation: paired horizontal flip of image and
  -- mask (F32.segHflipPair, one coin per image). Brains are near-symmetric so
  -- hflip is label-safe, and it is the cheapest real-data-variety lever on a
  -- 14k-slice set. This is the ceiling-raiser the best-checkpoint peak (WT
  -- 0.329, capacity-bound) motivated — not another re-roll of the same recipe.
  let aug := args.any (· == "aug")
  -- `lr=<n>` overrides the base learning rate as n×1e-4 (default 0.001 = lr=10).
  -- The aug run peaks high (WT 0.66) but oscillates violently (0.66→0.43→0.06→
  -- 0.26) — heavy weights + cosine + aug noise slingshot the narrow basin. A
  -- gentler LR (lr=5 → 5e-4) is the direct test of whether it can HOLD a high
  -- number instead of spiking to it.
  let baseLr : Float :=
    (((args.filter (·.startsWith "lr=")).head?.bind
      (fun a => (a.drop 3).toNat?)).map Nat.toFloat |>.getD 10.0) * 0.0001
  -- `tag=<s>` appends an extra suffix to the artifact tag, so two runs of the
  -- SAME arm at different budgets/schedules (e.g. a 10-epoch and a 30-epoch
  -- `wcesqrt cos pb`) don't clobber each other's checkpoints or race on the
  -- vmfb. Without it both resolve to the same `wcesqrt_pb_cos` prefix.
  let extraTag : String :=
    match (args.filter (·.startsWith "tag=")).head? with
    | some a => "_" ++ (a.drop 4).toString
    | none => ""
  -- Tag the build artifacts with the arm. Without this every arm writes the
  -- same `_params.bin` and the same `_train_step.vmfb`, so a sequential
  -- ablation silently overwrites itself and a parallel one (which is the point
  -- of having two GPUs) races mid-compile. With it, `ce` and `wce` can run
  -- concurrently and each keeps its own checkpoint for `brats-predict`.
  let armName := (if args.any (· == "ce") then "ce"
                  else if args.any (· == "dicece") then "dicece"
                  else if args.any (· == "dice") then "dice"
                  else if args.any (· == "focal") then s!"focal_g{focalGamma.toUInt64}"
                  else if args.any (· == "wcesqrt") then "wcesqrt"
                  else if args.any (· == "wceb") then s!"wceb{(wceBeta * 100.0).toUInt64}"
                  else if args.any (· == "wce") then "wce"
                  else "dicece") ++ (if args.any (· == "pb") then "_pb" else "")
  let lrTag := if baseLr == 0.001 then "" else s!"_lr{(baseLr * 10000.0).toUInt64}"
  let fullTag := armName ++ (if cosine then "_cos" else "") ++ (if aug then "_aug" else "") ++ lrTag ++ extraTag
  IO.eprintln s!"  arm: {fullTag}  (artifacts: {(unetBrats.withBuildTag fullTag).buildPrefix}_*)"
  (unetBrats.withBuildTag fullTag).train
    { unetBratsConfig with epochs, lossKind, headPriorBias := priorBias, cosineDecay := cosine, augment := aug, learningRate := baseLr }
    dataDir .brats
