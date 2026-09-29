import LeanMlir
import LeanMlir.ReferenceNets

open ReferenceNets (r34UnetBratsOf)

/-! **Transfer learning**: the self-hosted ImageNet ResNet-34, retrained on
    BraTS brain-tumour segmentation.

    The claim this demo exists to support is not "we can segment tumours" —
    `MainUnetBratsTrain` already does that from scratch. It is that **one
    backbone this stack trained itself, with no borrowed weights anywhere,
    transfers to two unrelated real tasks**: object detection (VisDrone, see
    `MainYolov1VisdroneFpn`) and now medical image segmentation.

    The backbone is `.lake/build/jax_r34_imagenet.bin` — 21,284,672 floats,
    72% top-1, trained by this stack on ImageNet. Nothing here is downloaded.

    ## What is measured, and against what

    The arm switch (`r34` vs `scratch`) changes **exactly one thing**: whether
    the encoder starts from those ImageNet weights or from He-init. Same net,
    same data, same schedule, same loss, same seed. That matters, and it is
    the reason this demo does not simply race the from-scratch `unetBrats`:
    that comparison confounds the initialization with a different architecture,
    and could not tell you which one moved the number.

    The from-scratch `unetBrats` result (mIoU 0.736, 3 epochs) is a
    separate reference point — "is this architecture competitive at all" — not
    the transfer measurement.

    **The expected payoff is sample-efficiency, not peak Dice.** Transfer
    usually buys you the same number in fewer epochs rather than a better
    number eventually, so `evalEveryNEpochs := 1` and the epoch-by-epoch
    WT/TC/ET curve is the deliverable. A final-epoch-only comparison would
    measure the wrong thing.

    ## Two architectural notes, both deliberate

    * **224, not BraTS's native 240 — and it is a crop, not a resize.** R34 is
      /32, and 240/32 = 7.5, so 240 cannot produce clean feature maps or a
      decoder that lands back on the input size. 224/32 = 7 exactly:
      7 → 14 → 28 → 56 → 112 → 224 on five upsamples. 224 is also R34's own
      ImageNet resolution, so the backbone sees the scale it was trained at.

      `scripts/datasets/preprocess_brats.py --size 224` **center-crops** (`fit_plane`); it does
      not interpolate. That matters more than it sounds: a bilinear resize of a
      label mask invents classes that never existed, silently. A crop leaves
      intensities exact and the mask exactly {0,1,2,3}. It is also lossless
      here — across 4,000 sampled slices the 8-pixel border it removes holds
      **zero** tumour voxels and **zero** brain voxels, because MSD's volumes
      are skull-stripped and centered.

      Dropping the encoder's maxPool to make the stride /16 (240/16 = 15, also
      exact) needs no new data, but runs every encoder stage at 2× resolution —
      4× the FLOPs.

    * **The stem is fresh, and that is correct.** BraTS is 4 co-registered MRI
      modalities (FLAIR / T1w / T1gd / T2w); ImageNet is 3-channel RGB. So the
      stem is `[64,4,7,7]` here and `[64,3,7,7]` in the checkpoint, and the
      transferable weights are no longer a prefix — hence
      `bootstrapBackboneRange` (`NetSpec.patchInitWithPretrainedRange`) rather
      than the detector's prefix bootstrap. The stem is 12,544 of 21,284,672
      floats — **0.06%** — and MRI intensities are not RGB intensities, so
      re-learning that one layer is the right behaviour rather than a
      concession. Every other backbone weight transfers intact.

      Keeping all 4 modalities also means this net sees the same modalities the
      from-scratch control sees, and that `brats-predict net=r34` renders it
      with no changes beyond the spec swap.

    ## 2.5D — `ctx=<k>`

    A slice model never sees the slice above or below. `ctx=k` feeds it the `k`
    axial neighbours on each side as extra input channels (`4·(2k+1)` in all,
    `DatasetKind.brats224Ctx k`, from `preprocess_brats.py --context k`), still
    predicting the centre slice's mask. Because the stem is the fresh layer
    anyway, the bootstrap skips `stemFloats (4·(2k+1))` instead of
    `stemFloats 4` and nothing else changes: same records, same order, same
    schedule, same seed as the 2D run, so the 2D and 2.5D arms are an A/B on
    through-plane context alone. That number decides whether a 3D UNet is
    worth its codegen (planning/brats_25d_3d.md).

    Usage:
      python3 scripts/datasets/preprocess_brats.py data/brats/Task01_BrainTumour data/brats224 \
              --size 224 --seed 0                 # same split as data/brats
      lake exe unet-brats-r34 data/brats224 10 r34       # bootstrapped arm
      lake exe unet-brats-r34 data/brats224 10 scratch   # control arm

      python3 scripts/datasets/preprocess_brats.py data/brats/Task01_BrainTumour data/brats224c3 \
              --size 224 --seed 0 --context 1     # 2.5D, ±1 slice, 12 channels
      lake exe unet-brats-r34 10 r34 ctx=1               # reads data/brats224c3
-/

/-- Floats in a `convBn ic 64 7 2` stem, all three tensors: W `[64,ic,7,7]`
    plus BN gamma `[64]` and beta `[64]`. The two offsets the range bootstrap
    needs are just this at `ic = 4` (this spec) and `ic = 3` (the checkpoint). -/
private def stemFloats (ic : Nat) : Nat := 64 * ic * 7 * 7 + 64 + 64

/-- Backbone floats in `jax_r34_imagenet.bin` — the R34 conv body, stem
    through stage 4, excluding the 1000-class classifier. Same constant the
    VisDrone detector bootstraps with, and it is not a magic number: it is
    `stemFloats 3` (9,536) + 221,952 + 1,116,416 + 6,822,400 + 13,114,368 for
    the four residual stages. The file itself is 21,797,672 floats; the
    difference, 513,000, is exactly the `512×1000 + 1000` classifier. -/
private def r34BackboneFloats : Nat := 21284672

/-- **Equivalence probe.** `unetDown ic oc` is defined as
    `convBn(ic→oc) + convBn(oc→oc) + maxPool 2`, pushing the pre-pool feature as
    the skip. So these two specs are the SAME network — same layers, same
    parameter shapes in the same order, same skip — but they reach the decoder's
    concat by different code paths:

      `equiv false` → `unetDown` … `unetUp`   (the `unetDown` path)
      `equiv true`  → `convBn, convBn, maxPool` … `unetUp`  (the tap path)

    Run one training step of each from the same He-init and the losses and
    updated parameters must agree bit-for-bit. That is a known-answer test for
    the tap wiring which does NOT depend on the absolute accuracy of the
    gradient — which matters here, because this architecture family carries
    a ~15% analytic-vs-finite-difference gap (measured by the FD probe) that
    swamps any skip-specific error.

    It exercises the maxPool→`addSkipGrad` half of the tap path directly. The
    residual-stage half differs only in which field carries the gradient
    (`fpnTapGrad`), the consumption and tagging logic being shared. -/
def r34EquivProbe (viaTap : Bool) : NetSpec where
  name := if viaTap then "equiv probe TAP (BraTS 224)" else "equiv probe UNETDOWN (BraTS 224)"
  imageH := 224
  imageW := 224
  layers :=
    (if viaTap then
      [ .convBn 4 32 3 1 .same, .convBn 32 32 3 1 .same, .maxPool 2 2 ]
     else
      [ .unetDown 4 32 ]) ++
    [ .unetUp 32 32, .conv2d 32 4 1 .same .identity ]

def r34UnetBratsConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 16             -- matches the from-scratch unetBrats arms, so
                                 -- the LR and schedule carry over unreinterpreted
  epochs       := 10
  optimizer    := .adam
  weightDecay  := 0.0001
  cosineDecay  := true           -- the from-scratch BraTS arms oscillated hard
                                 -- at a flat LR (see MainUnetBratsTrain)
  warmupEpochs := 1              -- a bootstrapped encoder should not eat a
                                 -- full-LR step before the fresh stem and
                                 -- decoder have any signal at all
  augment      := false
  lossKind     := some .perPixelCE    -- plain CE, which segments. Do not reopen
                                 -- weighted-CE/focal here.
  evalEveryNEpochs := 1          -- epochs-to-target IS the transfer claim
  checkpointEveryNEpochs := 2

def main (args : List String) : IO Unit := do
  -- `ctx=<k>`: the 2.5D context (0 = the 2D net). Picks the dataset kind, its
  -- channel count and its default directory in one place.
  let ctx : Nat :=
    ((args.filter (·.startsWith "ctx=")).head?.bind (fun a => (a.drop 4).toNat?)).getD 0
  let kind : DatasetKind := if ctx == 0 then .brats224 else .brats224Ctx ctx
  let inCh := kind.bratsChannels
  -- Positionals: the data dir is the first argument that is not a number or a
  -- keyword, the epoch count the first bare number — order-free, so
  -- `unet-brats-r34 10 r34 ctx=1` needs no directory spelled out.
  let keywords := ["r34", "scratch", "noskip", "equivdown", "equivtap", "fdprobe"]
  let positionals := args.filter (fun a =>
    !(keywords.contains a || a.startsWith "lr=" || a.startsWith "tag=" || a.startsWith "ctx="))
  let epochs := (positionals.findSome? String.toNat?).getD r34UnetBratsConfig.epochs
  let dataDir := (positionals.find? (fun a => (String.toNat? a).isNone)).getD kind.bratsDataDir
  -- The arm switch. `r34` bootstraps the encoder; `scratch` is the control.
  -- Default is the control on purpose: an unlabelled run should be the boring
  -- one, so a forgotten flag understates the result rather than inventing it.
  let useR34 := args.any (· == "r34")
  -- `noskip` selects the skipless decoder. Skips are the default because they are
  -- strictly the better segmenter; the flag exists so the published no-skip
  -- numbers stay reproducible from this exe.
  let skips := !(args.any (· == "noskip"))
  -- `equivdown` / `equivtap` select the two halves of the equivalence probe.
  let spec :=
    if args.any (· == "equivdown") then r34EquivProbe false
    else if args.any (· == "equivtap") then r34EquivProbe true
    else r34UnetBratsOf skips ctx
  -- `fdprobe` makes one training step an exactly-invertible function of the
  -- gradient, so `scripts/probes/brats_r34_fd_probe.py` can recover g = (θ−θ′)/η and
  -- check it against finite differences of the loss.
  --
  -- Every knob that would bend the update away from `θ − η·g` is switched off:
  -- Adam (rescales by √v), cosine decay and warmup (scale η), grad clipping
  -- (rescales g), weight decay (adds λθ), and augmentation (makes the batch
  -- non-deterministic across runs). With those off, and BN in train mode using
  -- batch statistics, the loss is a deterministic function of (θ, batch).
  let fdProbe := args.any (· == "fdprobe")
  -- `lr=<n>` as n×1e-4, matching unet-brats-train's convention.
  let baseLr : Float :=
    (((args.filter (·.startsWith "lr=")).head?.bind
      (fun a => (a.drop 3).toNat?)).map Nat.toFloat |>.getD 10.0) * 0.0001
  let extraTag : String :=
    match (args.filter (·.startsWith "tag=")).head? with
    | some a => "_" ++ (a.drop 4).toString
    | none => ""
  let lrTag := if baseLr == 0.001 then "" else s!"_lr{(baseLr * 10000.0).toUInt64}"
  -- Tag artifacts by arm. Both arms share a NetSpec (that is the point), so
  -- without this they would share `_params.bin` and `_train_step.vmfb` too —
  -- a sequential A/B would overwrite itself and a parallel one would race
  -- mid-compile.
  let fullTag := (if useR34 then "r34" else "scratch")
                 ++ (if skips then "" else "_noskip") ++ lrTag ++ extraTag

  let bootstrap : Option (String × Nat × Nat × Nat) :=
    if !useR34 then none
    else
      -- dst: skip THIS spec's stem (4 channels in 2D, 4·(2k+1) in 2.5D). src:
      -- skip the checkpoint's 3-channel stem. count: the rest of the backbone.
      let dstOff := stemFloats inCh       -- 12,672 at 4 channels
      let srcOff := stemFloats 3          --  9,536
      some (".lake/build/jax_r34_imagenet.bin", dstOff, srcOff,
            r34BackboneFloats - srcOff)   -- 21,275,136

  IO.eprintln s!"  arm: {fullTag}  ({if useR34 then "R34 ImageNet bootstrap" else "He-init control"})"
  if ctx > 0 then
    IO.eprintln s!"  2.5D: ctx=±{ctx} → {inCh} input channels, records from {dataDir}"
  IO.eprintln s!"  artifacts: {(spec.withBuildTag fullTag).buildPrefix}_*"
  if useR34 then
    let ckpt := ".lake/build/jax_r34_imagenet.bin"
    if !(← System.FilePath.pathExists ckpt) then
      IO.eprintln s!"missing backbone checkpoint: {ckpt}"
      IO.eprintln "  (this arm is the whole demo — refusing to silently fall back to He-init)"
      IO.Process.exit 1

  let cfg := { r34UnetBratsConfig with
                 epochs, learningRate := baseLr, bootstrapBackboneRange := bootstrap }
  let cfg := if !fdProbe then cfg else
    { cfg with optimizer := .sgd, cosineDecay := false, warmupEpochs := 0
             , weightDecay := 0.0, gradClipNorm := 0.0, augment := false
             , evalEveryNEpochs := 0, checkpointEveryNEpochs := 0 }
  (spec.withBuildTag fullTag).train cfg dataDir kind
