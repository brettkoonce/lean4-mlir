import LeanMlir
import LeanMlir.CliArgs
import LeanMlir.ReferenceNets
import LeanMlir.SmallClassifier

/-! Which wavelengths travel: chapter 4's CNN on EuroSAT's thirteen Sentinel-2 bands.

    EuroSAT is 27,000 64 × 64 Sentinel-2 L1C chips over 34 European countries, ten
    land-cover classes, all 13 bands (443–2190 nm); `scripts/datasets/preprocess_rs_eurosat.py`
    writes them as f32 `[N, 13, 64, 64]` records, per-band standardised. The net is
    chapter 4's CIFAR-CNN8-wide-BN (`cifar8w`, the GW demo's) with a `C`-channel stem,
    where `C` and which planes feed it are the ARM: `rgb` (B04 B03 B02), `rgbn` (+ B08),
    `ms10` (the ten 10 m / 20 m bands), `all` (13), `ir` (red edge, NIR and SWIR — no
    visible light). Every arm gathers its planes out of the same 13-plane records by
    `ByteArray.extract`, so one file per part serves all five; the ordinary
    cross-entropy train step with int32 labels, the blackjack/2-D host loop, no
    `DatasetKind`, no new codegen. The Brazil parts (`scripts/datasets/preprocess_rs_brazil.py`) are
    scored with the same weights and written as logits for `scripts/demos/rs_score.py`, which
    collapses the ten European classes into the seven the two continents share.

    XLA backend only.
    `lake exe rs-bands [arm=rgb|rgbn|ms10|all|ir] [train=eurosat_train] [val=eurosat_val]
                       [score=eurosat_test[,amazon_dry,...]] [classes=10] [epochs=20] [batch=64]
                       [lr=0.001] [ls=0.0] [seed=1] [aug=1] [fold=<k>] [labels=<N>] [init=<prefix>] [tag=<name>]
                       [out=<dir>] [eval]`
    `fold=k` trains on the chips of `train` whose `folds_<train>.bin` id is not k and scores
    the ones whose id is k (the Brazil-trained ceiling and the fine-tune rows; `val` and a
    `score` part named like `train` take the test side); `labels=N` trains on N chips of the
    train side drawn from the seed (the label-budget rows).
    `aug=1` (default) trains on each chip under a random one of the eight symmetries of
    the square — flips and 90° rotations, all label-preserving for a nadir chip
    (`F32.dihedralGather`, in C); `aug=0` is the plain gather.
    `train` / `val` / `score` name parts under `data/rs/` (`<part>.bin` + `labels_<part>.bin`);
    `init` continues from a saved `<prefix>_params.bin` + `_bn_stats.bin` (the Brazil
    fine-tune arm); `eval` reloads this run's own checkpoint and only writes logits.
    Writes `<prefix>_curve.csv`, `_params.bin`, `_bn_stats.bin` and `_logits_<part>.bin`
    (f32 `[N, classes]`) under `out` (default `.lake/build`). -/

open CliArgs SmallClassifier

namespace RsBands

def H : Nat := 64
def W : Nat := 64
/-- Planes in the record: B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B8A. -/
def nPlanes : Nat := 13
def hw : Nat := H * W

/-- The arm's planes into the 13-plane record (0-based; B8A is plane 12). -/
def planesOf (arm : String) : Option (Array Nat) :=
  match arm with
  | "rgb"  => some #[3, 2, 1]
  | "rgbn" => some #[3, 2, 1, 7]
  | "ms10" => some #[1, 2, 3, 4, 5, 6, 7, 12, 10, 11]
  | "all"  => some #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  | "ir"   => some #[4, 5, 6, 7, 12, 10, 11]
  | _ => none

/-- Chapter 4's CIFAR-CNN8-wide-BN verbatim: eight convolutions in four conv-conv-pool
    stages at 16, 16, 32, 32 channels into the 2 × 512 head; only the stem's input
    channels and the output width change (flatten = 32 × 4 × 4 = 512). -/
def cifar8w (arm : String) (C nOut : Nat) : NetSpec :=
  ReferenceNets.cifar8wOf s!"rs cifar8w {arm}" C H W nOut

/-- Re-lay a 13-plane part as the arm's `C`-plane part: one memcpy per (chip, plane). -/
def selectPlanes (src : ByteArray) (n : Nat) (planes : Array Nat) : ByteArray := Id.run do
  let planeBytes := hw * 4
  let mut dst := ByteArray.emptyWithCapacity (n * planes.size * planeBytes)
  for k in [:n] do
    for p in planes do
      dst := dst ++ src.extract ((k * nPlanes + p) * planeBytes) ((k * nPlanes + p + 1) * planeBytes)
  return dst

/-- Gather a batch of `B` chips by index from the arm-layout f32 part, and their labels;
    with `aug`, each chip under a random one of the eight symmetries of the square
    (`F32.dihedralGather`, seeded by the step). -/
def gatherChips (img lbl : ByteArray) (idx : Array Nat) (start B C : Nat) (aug : Bool) (seed : UInt64) :
    IO (ByteArray × ByteArray) := do
  if aug then
    let mut y := ByteArray.emptyWithCapacity (B * 4)
    let mut ib := ByteArray.emptyWithCapacity (B * 4)
    for i in [:B] do
      let k := idx[start + i]!
      y := y ++ F32.sliceLabels lbl k 1
      ib := pushU32LE ib k
    return (← F32.dihedralGather img ib B.toUSize C.toUSize H.toUSize seed, y)
  else
    return gather img lbl idx start B (C * hw)

/-- Which chips of a part a fold keeps: `fold=k` trains on `folds_<part>.bin ≠ k` and
    scores on `= k`; a part without a folds file is used whole. -/
inductive FoldSide | train | test | whole

/-- Load `data/rs/<part>.bin` (13 planes) re-laid to the arm, with its int32 labels; the
    label check is the ArASL one (every label < classes). With a fold, only that side's
    chips are copied (`folds_<part>.bin`, int32 per chip, written by `scripts/datasets/rs_folds.py`
    from a hash of the chip id, so a wet/dry twin sits in one fold). -/
def loadPart (dataDir part : String) (planes : Array Nat) (nOut : Nat) (fold : Option Nat := none)
    (side : FoldSide := .whole) (labels : Nat := 0) (seed : Nat := 1) : IO (ByteArray × ByteArray × Nat) := do
  let imgPath := s!"{dataDir}/{part}.bin"
  let lblPath := s!"{dataDir}/labels_{part}.bin"
  for f in [imgPath, lblPath] do
    unless ← System.FilePath.pathExists f do
      throw <| IO.userError s!"{f} missing — run the matching scripts/datasets/preprocess_rs_*.py"
  let lbl ← IO.FS.readBinFile lblPath
  let n := lbl.size / 4
  let raw ← IO.FS.readBinFile imgPath
  unless raw.size == n * nPlanes * hw * 4 do
    throw <| IO.userError s!"{imgPath}: {raw.size} bytes, expected {n} chips × 13 × {hw} × 4 = {n * nPlanes * hw * 4}"
  for i in [:n] do
    let l := lbl.data[i * 4]!.toNat + 256 * lbl.data[i * 4 + 1]!.toNat
    unless l < nOut do
      throw <| IO.userError s!"{lblPath}: label {l} at chip {i} but classes={nOut}"
  let keep : Array Nat ← match fold, side with
    | some k, .train | some k, .test => do
      let fPath := s!"{dataDir}/folds_{part}.bin"
      unless ← System.FilePath.pathExists fPath do
        throw <| IO.userError s!"{fPath} missing — scripts/datasets/rs_folds.py writes it"
      let fb ← IO.FS.readBinFile fPath
      unless fb.size == n * 4 do
        throw <| IO.userError s!"{fPath}: {fb.size / 4} fold ids for {n} chips"
      let mut a : Array Nat := #[]
      for i in [:n] do
        let f := fb.data[i * 4]!.toNat
        if (match side with | .train => f != k | _ => f == k) then a := a.push i
      pure a
    | _, _ => pure (Array.range n)
  -- a label budget: `labels` chips of the kept set, drawn from the seed (the fine-tune rows)
  let keep := if labels == 0 || labels >= keep.size then keep else
    (permutation keep.size (seed.toUInt64 * 6364136223846793005 + 1442695040888963407)).toList.take labels
      |>.toArray.qsort (· < ·) |>.map (keep[·]!)
  if keep.size == n then
    return (selectPlanes raw n planes, lbl, n)
  let planeBytes := hw * 4
  let mut sub := ByteArray.emptyWithCapacity (keep.size * nPlanes * planeBytes)
  let mut lsub := ByteArray.emptyWithCapacity (keep.size * 4)
  for i in keep do
    sub := sub ++ raw.extract (i * nPlanes * planeBytes) ((i + 1) * nPlanes * planeBytes)
    lsub := lsub ++ lbl.extract (i * 4) ((i + 1) * 4)
  return (selectPlanes sub keep.size planes, lsub, keep.size)

end RsBands

open RsBands in
def main (args : List String) : IO Unit := do
  let arm := parseArg args "arm" "all"
  let trainPart := parseArg args "train" "eurosat_train"
  let valPart := parseArg args "val" "eurosat_val"
  let scoreParts := (parseArg args "score" "eurosat_test").splitOn "," |>.filter (· ≠ "")
  let nOut := (parseArg args "classes" "10").toNat!
  let epochs := (parseArg args "epochs" "20").toNat!
  let B := (parseArg args "batch" "64").toNat!
  let lr := floatArg args "lr" 0.001
  let ls := floatArg args "ls" 0.0
  let seed := (parseArg args "seed" "1").toNat!
  let init := parseArg args "init" ""
  let fold : Option Nat := (parseArg args "fold" "").toNat?
  let labels := (parseArg args "labels" "0").toNat!
  let tag := parseArg args "tag" ""
  let outDir := parseArg args "out" ".lake/build"
  let evalOnly := args.contains "eval"
  let aug := parseArg args "aug" "1" != "0"
  let maxSteps := (parseArg args "steps" "0").toNat!      -- debugging: stop after this many
  let dataDir := "data/rs"
  let some planes := planesOf arm
    | throw <| IO.userError s!"arm={arm}: expected rgb, rgbn, ms10, all or ir"
  let C := planes.size
  let nPix := C * hw
  let spec := cifar8w arm C nOut
  unless (← LowererSession.backendName) == "xla" do
    throw <| IO.userError "rs-bands runs on the XLA backend only"
  IO.FS.createDirAll outDir
  let pfx := s!"{outDir}/{spec.sanitizedName}" ++ (if tag == "" then "" else s!"_{tag}")
    ++ (match fold with | some k => s!"_fold{k}" | none => "") ++ (if labels > 0 then s!"_n{labels}" else "")
  IO.eprintln s!"{spec.name}: {C} planes {planes}, {spec.bnLayers.size} BN layers, \
train on {trainPart}{match fold with | some k => s!" fold {k}" | none => ""}{if labels > 0 then s!" ({labels} labels)" else ""}, {epochs} epochs, batch {B}, lr {lr}, label smoothing {ls}, seed {seed}\
{if aug then ", dihedral augmentation" else ", no augmentation"}{if init == "" then "" else s!", init {init}"}{if evalOnly then " (eval only)" else ""}"

  -- ── graphs ──
  IO.FS.createDirAll ".lake/build"
  let gpfx := spec.buildPrefix                  -- already under .lake/build/
  let trainMlir := MlirCodegen.generateTrainStep spec B
    ("jit_" ++ spec.sanitizedName ++ "_train_step")
    (labelSmoothing := ls) (weightDecay := 0.0001) (useAdam := true)
  IO.FS.writeFile s!"{gpfx}_train_step.mlir" trainMlir
  IO.FS.writeFile s!"{gpfx}_fwd_eval.mlir" (MlirCodegen.generateEval spec B)
  let evalSess ← LowererSession.create (← NetSpec.graphArtifact gpfx "fwd_eval")
  -- sized from the initialised buffer, which the emitted graph follows; a wrong nP reads the
  -- loss from inside a weight tensor.
  let p0 ← spec.heInitParams
  let nP := F32.size p0
  let nT := 3 * nP
  let nBn := spec.nBnStats
  IO.eprintln s!"  {nP} params ({spec.totalParams} by NetSpec.totalParams), {nBn} BN stat floats"
  let allShapes := spec.shapesBA
  let evalShapes := spec.evalShapesBA
  let bnShapes := spec.bnShapesBA
  let xSh := spec.xShape B

  -- ── data: every part as one arm-layout ByteArray, batches gathered by index ──
  let t0 ← IO.monoMsNow
  let (imgVa, lblVa, nVa) ← loadPart dataDir valPart planes nOut fold .test
  let t1 ← IO.monoMsNow
  IO.eprintln s!"  val {valPart}: {nVa} chips × {C} planes ({t1 - t0} ms)"

  let mut p := p0
  let mut bn ← F32.const nBn.toUSize 0.0
  if init != "" then
    p ← IO.FS.readBinFile s!"{init}_params.bin"
    bn ← IO.FS.readBinFile s!"{init}_bn_stats.bin"
    unless F32.size p == nP && F32.size bn == nBn do
      throw <| IO.userError s!"{init}: {F32.size p} params / {F32.size bn} BN floats, this arm has {nP} / {nBn}"
    IO.eprintln s!"  init from {init}_params.bin"
  if evalOnly then
    p ← IO.FS.readBinFile s!"{pfx}_params.bin"
    bn ← IO.FS.readBinFile s!"{pfx}_bn_stats.bin"
    IO.eprintln s!"  loaded {pfx}_params.bin"
  else
    let (imgTr, lblTr, nTr) ← loadPart dataDir trainPart planes nOut fold .train labels seed
    let t2 ← IO.monoMsNow
    IO.eprintln s!"  train {trainPart}: {nTr} chips ({t2 - t1} ms)"
    let sess ← LowererSession.create (← NetSpec.graphArtifact gpfx "train_step")
    let mut m ← F32.const nP.toUSize 0.0
    let mut v ← F32.const nP.toUSize 0.0
    let bpE := nTr / B
    let total := epochs * bpE
    let warm := bpE
    let mut step : Nat := 0
    let mut curve : Array String := #[]
    let tStart ← IO.monoMsNow
    for epoch in [:epochs] do
      let idx := permutation nTr (seed.toUInt64 * 1000003 + epoch.toUInt64 + 1)
      let mut lossAcc : Float := 0.0
      let tE ← IO.monoMsNow
      for bi in [:bpE] do
        step := step + 1
        -- linear warmup over the first epoch, cosine to zero after
        let lrNow := if step <= warm then lr * step.toFloat / warm.toFloat
          else lr * 0.5 * (1.0 + Float.cos (3.14159265358979 * (step - warm).toFloat / (total - warm).toFloat))
        let (xba, yb) ← gatherChips imgTr lblTr idx (bi * B) B C aug (seed.toUInt64 * 7919 + step.toUInt64)
        let packed := (p.append m).append v
        let out ← LowererSession.trainStepAdamF32 sess spec.trainFnName
                    packed allShapes xba xSh yb lrNow step.toFloat bnShapes B.toUSize
        if step == 1 then
          unless out.size / 4 == nT + 1 + nBn do
            throw <| IO.userError s!"train step returned {out.size / 4} floats, expected 3*{nP} + 1 + {nBn} = {nT + 1 + nBn}: the packed layout does not match the graph"
        lossAcc := lossAcc + F32.read out nT.toUSize
        (p, m, v) := F32.unpackAdam out nP
        let batchBn := out.extract ((nT + 1) * 4) ((nT + 1 + nBn) * 4)
        bn ← F32.ema bn batchBn (if step == 1 && init == "" then 1.0 else 0.1)
        if epoch == 0 && bi < 3 then
          let tb ← IO.monoMsNow
          IO.eprintln s!"    step {bi}: loss {fmt (F32.read out nT.toUSize) 4} ({tb - tE} ms so far)"
        if maxSteps > 0 && step >= maxSteps then
          throw <| IO.userError s!"stopped after {step} steps (steps={maxSteps})"
      let tV ← IO.monoMsNow
      let evalParams := p.append bn
      let (_, acc) ← scoreSet evalSess spec evalParams evalShapes xSh imgVa lblVa nVa B nPix
      let tD ← IO.monoMsNow
      let trainLoss := lossAcc / bpE.toFloat
      IO.eprintln s!"  epoch {epoch + 1}/{epochs}: loss {fmt trainLoss 4}  val acc {fmt acc 2}%  \
({tV - tE} ms train, {tD - tV} ms eval)"
      curve := curve.push s!"{epoch + 1},{fmt trainLoss 5},{fmt acc 3}"
    let tEnd ← IO.monoMsNow
    IO.eprintln s!"trained: {step} steps in {(tEnd - tStart) / 1000} s"
    IO.FS.writeBinFile s!"{pfx}_params.bin" p
    IO.FS.writeBinFile s!"{pfx}_bn_stats.bin" bn
    IO.FS.writeFile s!"{pfx}_curve.csv" ("epoch,loss,val_acc\n" ++
      String.intercalate "\n" curve.toList ++ "\n")

  -- ── logits for the scorer: the val part and every score part ──
  let evalParams := p.append bn
  for part in valPart :: scoreParts do
    let (img, lbl, n) ← if part == valPart then pure (imgVa, lblVa, nVa)
      else loadPart dataDir part planes nOut fold (if part == trainPart then .test else .whole)
    let (logits, acc) ← scoreSet evalSess spec evalParams evalShapes xSh img lbl n B nPix
    IO.FS.writeBinFile s!"{pfx}_logits_{part}.bin" logits
    IO.println s!"{spec.name} on {part}: {n} chips, {nOut}-way argmax accuracy {fmt acc 2}%  \
-> {pfx}_logits_{part}.bin"
  IO.eprintln s!"score: .venv-rs/bin/python scripts/demos/rs_score.py {pfx}_logits_<part>.bin --part <part>"
