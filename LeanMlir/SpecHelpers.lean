import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.IreeRuntime
import LeanMlir.MlirCodegen
import LeanMlir.F32Array

/-! Spec → tensor-shape / packed-bytes / parameter-init helpers.

These were copy-pasted into every `Main*Train.lean` file. Centralizing
them here means each trainer just imports `LeanMlir` and asks for
`spec.paramShapes`, `spec.bnShapesBA`, etc.

Anything new added here should be a pure function of `NetSpec` — no IO,
no architecture-specific specialization. Adding a new layer type means
adding a case here once and every trainer picks it up. -/

namespace NetSpec

/-- Per-parameter tensor shapes for the entire spec, in the same order
    `MlirCodegen.emitTrainStepSig` walks them. Used to pack/unpack the
    `params ++ m ++ v` ByteArray that goes into and out of every
    `trainStepAdamF32` call. -/
def paramShapes (spec : NetSpec) : Array (Array Nat) :=
  (spec.layers.flatMap fun l => (l.paramSlots.getD []).map (·.shape.toArray)).toArray

/-- Packed `params ++ m ++ v` shape array (3× the param shapes), as int32 LE
    `ByteArray`. This is what gets passed to `trainStepAdamF32`. -/
def shapesBA (spec : NetSpec) : ByteArray :=
  packShapes (spec.paramShapes ++ spec.paramShapes ++ spec.paramShapes)

/-- BN-layer (pidx, oc) pairs as discovered by the codegen. ViT-style
    transformer specs return an empty array. -/
def bnLayers (spec : NetSpec) : Array (Nat × Nat) :=
  MlirCodegen.collectBnLayers spec

/-- Total float count needed to store running BN stats (mean + var per
    BN layer). 0 for ViT and any non-BN architecture. -/
def nBnStats (spec : NetSpec) : Nat :=
  spec.bnLayers.foldl (fun acc (_, oc) => acc + oc * 2) 0

/-- BN shapes packed for the FFI: `[n_bn_layers, oc0, oc1, ...]` as int32 LE.
    The `trainStepAdamF32` FFI uses this to know how many BN-stat outputs
    to pop after the params/loss. -/
def bnShapesBA (spec : NetSpec) : ByteArray := Id.run do
  let push := pushU32LE
  let bn := spec.bnLayers
  let mut ba := push .empty bn.size
  for (_, oc) in bn do ba := push ba oc
  return ba

/-- Param shapes for the eval forward pass: regular params followed by
    one `[oc]` mean and `[oc]` var per BN layer. ViT-style specs collapse
    this to just `paramShapes` because there are no BN layers. -/
def evalShapes (spec : NetSpec) : Array (Array Nat) := Id.run do
  let mut shapes := spec.paramShapes
  for (_, oc) in spec.bnLayers do
    shapes := shapes.push #[oc] |>.push #[oc]
  return shapes

/-- Packed `evalShapes` as int32 LE for the eval forward FFI. -/
def evalShapesBA (spec : NetSpec) : ByteArray :=
  packShapes spec.evalShapes

/-- Packed input-tensor shape for a flat-image batch (NCHW collapsed
    to `[batch, channels*H*W]`). Channel count comes from the first
    conv-style layer; defaults to 1 for pure-MLP specs. -/
def xShape (spec : NetSpec) (batch : Nat) : ByteArray :=
  packXShape #[batch, MlirCodegen.inputFlatDim spec]

/-- Sanitized base name for the spec — same transformation the codegen
    applies when generating MLIR module names. -/
def sanitizedName (spec : NetSpec) : String :=
  MlirCodegen.sanitize spec.name

/-- The eval forward function name to pass to `forwardF32`. The codegen
    emits modules of the form `@<sanitized_name>_eval` containing
    `func.func @forward_eval`, so the FFI call wants the qualified
    `"<sanitized_name>_eval.forward_eval"`. Computing this from the
    spec means trainers can never get the spelling wrong by hand. -/
def evalFnName (spec : NetSpec) : String :=
  spec.sanitizedName ++ "_eval.forward_eval"

/-- Initialize one layer's parameters from its `Layer.paramSlots`: a random draw takes the current
    seed and advances it, a constant does not (`zeroSeeded` advances it without drawing). Layers
    with no slots contribute nothing. -/
private def heInitLayer (l : Layer) (seed : USize) : IO (Array ByteArray × USize) := do
  let mut parts : Array ByteArray := #[]
  let mut s := seed
  for sl in l.paramSlots.getD [] do
    let n := (sl.shape.foldl (· * ·) 1).toUSize
    match sl.init with
    | .he fanIn =>
      parts := parts.push (← F32.heInit s n (Float.sqrt (2.0 / fanIn.toFloat))); s := s + 1
    | .normal σ => parts := parts.push (← F32.heInit s n σ); s := s + 1
    | .const v => parts := parts.push (← F32.const n v)
    | .zeroSeeded => parts := parts.push (← F32.const n 0.0); s := s + 1
  return (parts, s)

/-- He-initialize all parameters for a spec, walking layer-by-layer. -/
def heInitParams (spec : NetSpec) : IO ByteArray := do
  let mut paramParts : Array ByteArray := #[]
  let mut seed : USize := 42
  for l in spec.layers do
    let (parts, s') ← heInitLayer l seed
    paramParts := paramParts ++ parts
    seed := s'
  return F32.concat paramParts

/-- Overwrite the head's bias with `log priors[c]` — RetinaNet's prior-bias
    init (Lin et al. §3.3, "prior"), for segmentation heads.

    `heInitParams` lays layers out in order and emits each conv's bias directly
    after its weights, so the head's bias is the **final `NC` floats** of the
    buffer. That is what this patches; it is a no-op on everything else.

    **What it buys, and why it is focal's other half.** A zero-bias head starts
    at a uniform softmax: every class at `1/NC`, background included. The net's
    first job is therefore to discover the class prior, and on BraTS it does
    that by walking straight into the trivial predictor — the collapse is
    decided in the first ~100 steps (`planning/archive/brats_demo.md` Workstream A). A
    `log π_c` bias hands it the prior at step 0 instead, so the first gradient
    step is spent on the actual task.

    The quantitative version is in `scripts/probes/seg_grad_scorecard.py`, whose sweep
    lands on this exact row. Prior-bias init starts the net at
    `z₀ - z₃ = log(π₀/π₃) = log(0.9746/0.0050) = 5.27` (verified against the
    emitted checkpoint: `softmax(head bias) == π` to 2e-09). Its measured
    balance ratio — rare-class gradient over majority gradient — reads:

    | (C) at | ce | dice | wce | focal |
    |---|---|---|---|---|
    | `z0 = 0` (uniform) | 5.09e-03 | 2.84e-02 | 9.96e-01 | **5.15e-03** |
    | `z0 = 5.27` (this) | 2.60e-01 | 1.12e-01 | 5.08e+01 | **9.90e+01** |

    **One bias vector is worth ~19,000× to focal, at step 0** — and flips it
    from the worst arm (tied with CE, a literal no-op) to the best (~2× wce).
    That is the whole content of "focal needs confidence to suppress": this
    manufactures the confidence up front instead of waiting for training to
    produce it too late. It is why Lin et al. ship the two together; they are
    one idea, and reading them as separate tricks is how the pairing gets lost.

    It helps every arm — wce's ratio rises 51× too — because starting at the
    prior is simply a better starting point than uniform. focal is the one that
    goes from *inert* to *leading*.

    `priors` need not be normalized: adding a constant to every logit is a
    no-op under softmax, so an overall scale on `priors` shifts every bias by
    `log k` and changes nothing. Only the ratios matter — the same
    scale-invariance that `perPixelWeightedCE`'s reduction enjoys, for the same
    reason.

    Returns a NEW ByteArray. -/
def applyHeadPriorBias (spec : NetSpec) (params : ByteArray)
    (priors : List Float) : IO ByteArray := do
  let nc := (spec.layers.getLast?.map (·.outChannels)).getD 0
  if priors.length != nc then
    throw <| IO.userError s!"headPriorBias: got {priors.length} priors for a {nc}-class head — they must match 1:1"
  if priors.any (· <= 0.0) then
    throw <| IO.userError "headPriorBias: priors must be strictly positive (log 0 = -inf would hard-zero the class, which is the collapse we are trying to prevent, installed by hand)"
  let total := params.size / 4
  if total < nc then
    throw <| IO.userError s!"headPriorBias: {total} params is smaller than the {nc}-float head bias"
  -- No scalar-store primitive on the F32 side, and none is worth adding for
  -- one vector written once per run: build the bias block and splice it onto
  -- the truncated buffer.
  let mut biasParts : Array ByteArray := #[]
  for pi in priors do
    biasParts := biasParts.push (← F32.const (1 : USize) (Float.log pi))
  return (params.extract 0 ((total - nc) * 4)).append (F32.concat biasParts)

/-- RetinaNet prior-bias init for the **FPN detector head** (planning/archive/yolo_fpn.md
    Tier 2). Sets every objectness logit's bias to `−log((1−π)/π)` so the head
    starts predicting `sigmoid = π` (π ≈ 0.01) on every cell; box and class biases
    stay at zero.

    This is the classifier trick of `applyHeadPriorBias` transposed to a
    sigmoid/one-vs-all head, and it is aimed at a measured failure rather than a
    guess. On the e12 run every objectness logit sat in ≈[−2.7, −1.2] (p5..p95)
    with pos/neg means −1.549/−1.803: the head had real signal (AUC 0.742) but
    almost no dynamic range, because a bias-free 1×1 conv has to synthesize the
    constant background offset out of weights that also have to discriminate. The
    bias hands it that constant for free, which is the whole point — and per-class
    mAP is bounded by objectness *ranking*, so this is the lever that can move it
    (both Tier-1 levers were measured out; see the T1a/T1b write-ups).

    Assumes the `.fpnDetect` layer is last, so its 3 `[A·15]` biases are the final
    `3·A·15` floats of the buffer — the same tail-splice `applyHeadPriorBias` does.
    Within each `[A·15]` block, anchor `a`'s objectness is channel `a·15 + 4`
    (`emitAnchorYoloLoss` slices box at `base..base+4`, obj at `base+4`, class at
    `base+5..base+15`).

    Returns a NEW ByteArray. -/
def applyDetPriorBias (spec : NetSpec) (params : ByteArray) (pi : Float) : IO ByteArray := do
  if pi <= 0.0 || pi >= 1.0 then
    throw <| IO.userError s!"detPriorBias: π must be in (0,1), got {pi}"
  let A ← match spec.layers.getLast? with
    | some (.fpnDetect _ _ _ _ _ A _) => pure A
    | _ => throw <| IO.userError "detPriorBias: last layer is not .fpnDetect (the head this init targets)"
  let ap := A * 15
  let total := params.size / 4
  if total < 3 * ap then
    throw <| IO.userError s!"detPriorBias: {total} params is smaller than the {3 * ap}-float head-bias tail"
  let b0 := -(Float.log ((1.0 - pi) / pi))
  -- One [A·15] block: b0 on each anchor's objectness channel, 0 elsewhere.
  let mut blockParts : Array ByteArray := #[]
  for c in [0:ap] do
    blockParts := blockParts.push (← F32.const (1 : USize) (if c % 15 == 4 then b0 else 0.0))
  let block := F32.concat blockParts
  return (params.extract 0 ((total - 3 * ap) * 4)).append (F32.concat #[block, block, block])

/-- Patch the first `prefixBytes` of `initParams` with bytes read from
    `pretrainedPath`. Used to bootstrap a fresh init from a pretrained
    backbone — e.g. load R34-Imagenette weights into a YOLOv1 init,
    keeping the YOLOv1 head's He-init untouched.

    `prefixBytes` is computed by the caller (usually
    `4 * (spec.totalParams - <last layer fanOut + fanIn*fanOut>)` for a
    spec whose final dense layer differs from the pretrained source).

    Returns a NEW ByteArray; `initParams` is not modified. -/
def patchInitWithPretrainedPrefix (initParams : ByteArray) (pretrainedPath : String)
    (prefixBytes : Nat) : IO ByteArray := do
  let pre ← IO.FS.readBinFile pretrainedPath
  if pre.size < prefixBytes then
    throw <| IO.userError s!"pretrained checkpoint {pretrainedPath} has {pre.size} bytes; need at least {prefixBytes}"
  if initParams.size < prefixBytes then
    throw <| IO.userError s!"init params has {initParams.size} bytes; can't patch {prefixBytes}"
  let head := pre.extract 0 prefixBytes
  let tail := initParams.extract prefixBytes initParams.size
  return head ++ tail

/-- Offset-aware sibling of `patchInitWithPretrainedPrefix`: copy `countBytes`
    from `pretrainedPath` starting at `srcOffBytes` into `initParams` starting
    at `dstOffBytes`, leaving everything outside that window at its He-init.

    The prefix version can only bootstrap weights that are contiguous at the
    FRONT of the layout, which forces the first layer to match the checkpoint's
    first layer exactly. That is the wrong constraint the moment the input
    channel count differs: an R34 trained on 3-channel RGB has a `[64,3,7,7]`
    stem, and a 4-modality MRI net needs `[64,4,7,7]`. The shapes disagree by
    3,136 floats, so a prefix patch would land every subsequent weight at the
    wrong offset — silently, since the sizes still "fit".

    With a range, the fresh stem keeps its He-init and the rest of the backbone
    lands where it belongs:
      dstOff = stem floats in THIS spec, srcOff = stem floats in the CHECKPOINT,
      count  = backbone floats − srcOff.

    Both windows are bounds-checked, because the failure mode this exists to
    prevent is a silent misalignment rather than a crash. -/
def patchInitWithPretrainedRange (initParams : ByteArray) (pretrainedPath : String)
    (dstOffBytes srcOffBytes countBytes : Nat) : IO ByteArray := do
  let pre ← IO.FS.readBinFile pretrainedPath
  if pre.size < srcOffBytes + countBytes then
    throw <| IO.userError s!"pretrained checkpoint {pretrainedPath} has {pre.size} bytes; need at least {srcOffBytes + countBytes} (srcOff {srcOffBytes} + count {countBytes})"
  if initParams.size < dstOffBytes + countBytes then
    throw <| IO.userError s!"init params has {initParams.size} bytes; can't patch {countBytes} at offset {dstOffBytes}"
  let head := initParams.extract 0 dstOffBytes
  let mid  := pre.extract srcOffBytes (srcOffBytes + countBytes)
  let tail := initParams.extract (dstOffBytes + countBytes) initParams.size
  return head ++ mid ++ tail

end NetSpec
