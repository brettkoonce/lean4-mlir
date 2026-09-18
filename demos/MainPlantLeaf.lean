import LeanMlir

/-! Chapter 6's ResNet-34 from lab leaves to field leaves —
    planning/plant_lab_to_field_demo.md.

    PlantVillage is 54,305 lab photographs of single picked leaves on a grey background,
    38 classes; PlantDoc is 2,578 field photographs of the same crops whose 28 classes all
    map into PlantVillage's 38. `preprocess_plant.py` writes both in the Imagenette record
    format under two PlantVillage splits (`random`, the literature's; `grouped`, the
    maintainers' own leaf-grouped split) with the test tenth's `segmented` twin, its
    background complement and its leaf mask beside it, plus a composite and an augmented
    twin of each training part and five PlantDoc folds. This trains the R34 from the
    ImageNet prefix (`jax_r34_imagenet.bin`, the BraTS/VisDrone/NEU bootstrap) with a
    38-way head through the ordinary cross-entropy train step — `demos/MainAraslSigns.lean`
    with the Imagenette loader and the bootstrap — and writes `[N, 38]` logits for every
    evaluation part, which `scripts/plant_score.py` turns into the two columns of the
    section's table. The u8 records stay resident and a C helper converts one batch at a
    time (`F32.imagenetteGather`), because the f32 form of a 43k-image part is 34 GB.

    XLA backend only.
    `lake exe plant-leaf [split=grouped|random] [train=base|comp|aug|compaug]
                         [init=imagenet|scratch|<run prefix>] [field=<fold 0..4>] [fieldn=<N>]
                         [epochs=10] [batch=64] [lr=0.001] [seed=1] [tag=<name>] [out=<dir>]
                         [extra=<part.bin>] [cam=1] [eval]`
    `train` picks the PlantVillage training parts (base ∪ composites / augmented twin);
    `field=k` instead fine-tunes on PlantDoc fold k's training images (`fieldn` of them,
    stratified) from the checkpoint `init` names, a fixed schedule with no selection, and
    scores fold k's held-out images. Otherwise the epoch is chosen on the PlantVillage val
    tenth. `extra` names one more 224-side part to score (the Shapley probe
    `scripts/plant_shapley.py grid` writes); `cam=1` also dumps the closed-form CAM of every
    image of the test tenth and the field part (`camDump`). Writes `<prefix>_curve.csv`, `_params.bin`, `_bn_stats.bin` and
    `_logits_<part>.bin` for `pv{,g}_test`, `_test_seg`, `_test_bg`, `_test_leaf`, `_test_none`
    and `pd_all` (or `pd_fold<k>_test`) under `out` (default `.lake/build`). -/

namespace PlantLeaf

def nClasses : Nat := 38
def S : Nat := 224
def sTrain : Nat := 256
def nPix : Nat := 3 * S * S
def nPixTrain : Nat := 3 * sTrain * sTrain
/-- The ImageNet R34 backbone: every conv and BN of the net, no head, no BN statistics. -/
def imagenetPrefix : String := ".lake/build/jax_r34_imagenet.bin"
def imagenetPrefixFloats : Nat := 21284672

/-- ResNet-34 as the chapter and the GradCAM probe spell it, with the 38-way head. -/
def resnet34 : NetSpec where
  name := "plant resnet34"
  imageH := S
  imageW := S
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 nClasses .identity
  ]

def fmt (x : Float) (d : Nat) : String :=
  let m := (10.0 : Float) ^ d.toFloat
  let r := (x * m).round / m
  let s := toString r
  if s.length > 10 then (s.toRawSubstring.take 10).toString else s

@[inline] def xs (s : UInt64) : UInt64 :=
  let s := s ^^^ (s <<< 13)
  let s := s ^^^ (s >>> 7)
  s ^^^ (s <<< 17)

/-- Fisher–Yates permutation of `0..n-1` from a seed. -/
def permutation (n : Nat) (seed : UInt64) : Array Nat := Id.run do
  let mut a : Array Nat := Array.range n
  let mut s := if seed == 0 then 0x9E3779B97F4A7C15 else seed
  for i in [1:n] do
    let j := n - i
    s := xs s
    let k := (s % (j + 1).toUInt64).toNat
    let tmp := a[j]!
    a := a.set! j a[k]!
    a := a.set! k tmp
  return a

/-- A part: the raw Imagenette-format bytes, its int32 labels, its size, and its stored
    image side (256 for training parts, 224 for evaluation parts). -/
structure Part where
  name : String
  raw : ByteArray
  lbl : ByteArray
  n : Nat
  side : Nat
  deriving Inhabited

def loadPart (dataDir name : String) (side : Nat) : IO Part := do
  let path := s!"{dataDir}/{name}.bin"
  unless ← System.FilePath.pathExists path do
    throw <| IO.userError s!"{path} missing — run: ./download_plant.sh"
  let raw ← IO.FS.readBinFile path
  let lbl ← F32.imagenetteLabels raw side.toUSize
  let n := lbl.size / 4
  unless raw.size == 4 + n * (1 + 3 * side * side) do
    throw <| IO.userError s!"{path}: {raw.size} bytes for {n} records of side {side}"
  return { name, raw, lbl, n, side }

def u32le (k : Nat) : ByteArray :=
  ByteArray.mk #[(k % 256).toUInt8, ((k / 256) % 256).toUInt8, ((k / 65536) % 256).toUInt8, ((k / 16777216) % 256).toUInt8]

/-- One image of a part as a normalised f32 buffer. -/
def image (p : Part) (k : Nat) : IO ByteArray :=
  F32.imagenetteGather p.raw (u32le k) 1 p.side.toUSize

/-- A training batch gathered by global index over a list of parts (their concatenation), at
    the stored 256, then random-cropped to 224 and flipped — the Imagenette recipe. -/
def gatherTrain (parts : Array Part) (idx : Array Nat) (start B : Nat) (seed : Nat) : IO (ByteArray × ByteArray) := do
  let mut x := ByteArray.emptyWithCapacity (B * nPixTrain * 4)
  let mut y := ByteArray.emptyWithCapacity (B * 4)
  for i in [:B] do
    let mut k := idx[start + i]!
    let mut pi := 0
    while k >= parts[pi]!.n do
      k := k - parts[pi]!.n
      pi := pi + 1
    let p := parts[pi]!
    x := x ++ (← image p k)
    y := y ++ F32.sliceLabels p.lbl k 1
  let cropped ← F32.randomCrop x B.toUSize 3 sTrain.toUSize sTrain.toUSize S.toUSize S.toUSize seed.toUSize
  let flipped ← F32.randomHFlip cropped B.toUSize 3 S.toUSize S.toUSize (seed + 7777).toUSize
  return (flipped, y)

/-- Run the eval graph over a 224-side part at batch `evalB` (tail zero-padded) and return the
    logits as f32 `[n, 38]` plus the accuracy. -/
def scoreSet (sess : LowererSession) (spec : NetSpec) (evalParams evalShapes xSh : ByteArray)
    (p : Part) (evalB : Nat) : IO (ByteArray × Float) := do
  let nC := nClasses
  let mut logits := ByteArray.emptyWithCapacity (p.n * nC * 4)
  let mut correct : Nat := 0
  let nb := (p.n + evalB - 1) / evalB
  for bi in [:nb] do
    let avail := min evalB (p.n - bi * evalB)
    let mut idx := ByteArray.emptyWithCapacity (evalB * 4)
    for i in [:evalB] do
      idx := idx ++ u32le (bi * evalB + (min i (avail - 1)))     -- pad with the last real image
    let xba ← F32.imagenetteGather p.raw idx evalB.toUSize p.side.toUSize
    let out ← LowererSession.forwardF32 sess spec.evalFnName evalParams evalShapes xba xSh
                evalB.toUSize nC.toUSize
    logits := logits ++ out.extract 0 (avail * nC * 4)
    for i in [:avail] do
      let pred := F32.argmaxN out (i * nC).toUSize nC.toUSize
      let label := p.lbl.data[(bi * evalB + i) * 4]!.toNat
      if pred.toNat == label then correct := correct + 1
  return (logits, correct.toFloat / p.n.toFloat * 100.0)

/-- Elements of the packed params before shape slot `targetIdx` (the GradCAM probe's rule for
    finding the final dense W and b: a `.dense fi fo _` is the last two slots, `[fi, fo]`, `[fo]`). -/
def offsetBefore (spec : NetSpec) (targetIdx : Nat) : Nat := Id.run do
  let shapes := spec.paramShapes
  let mut acc : Nat := 0
  for i in [:targetIdx] do
    let mut sz : Nat := 1
    for d in shapes[i]! do sz := sz * d
    acc := acc + sz
  return acc

/-- The closed-form CAM (Zhou 2016) over a whole 224-side part: the pre-GAP `[C, 7, 7]` map
    through `forward_cam`, the class's dense row as the channel weights, ReLU, max-normalised —
    `F32.camCompute` — for the TRUE class and for the PREDICTED class of every image. Writes
    `<pfx>_cam_<part>.bin` as f32 `[n, 2, 7, 7]` and `<pfx>_campred_<part>.bin` as int32 `[n]`;
    `scripts/plant_cam.py` reads them against the leaf masks. No autodiff, no new codegen:
    `MlirCodegen.generateForwardCam` is the eval graph cut before the pool. -/
def camDump (spec : NetSpec) (pfx : String) (params bn : ByteArray) (parts : Array Part) (B : Nat) : IO Unit := do
  let gpfx := spec.buildPrefix
  IO.FS.writeFile s!"{gpfx}_fwd_cam.mlir" (MlirCodegen.generateForwardCam spec B)
  let camSess ← LowererSession.create (← NetSpec.graphArtifact gpfx "fwd_cam")
  let camFn := s!"{spec.sanitizedName}_cam.forward_cam"
  let (c, h, w) ← match MlirCodegen.preGAPShape spec with
    | some t => pure t
    | none => throw <| IO.userError "spec has no globalAvgPool — not CAM-eligible"
  let nShapes := spec.paramShapes.size
  let wOff := offsetBefore spec (nShapes - 2)
  let denseW := F32.slice params wOff (512 * nClasses)
  let denseB := F32.slice params (wOff + 512 * nClasses) nClasses
  let evalParams := params.append bn
  let evalShapes := spec.evalShapesBA
  let xSh := spec.xShape B
  for p in parts do
    let mut cams := ByteArray.emptyWithCapacity (p.n * 2 * h * w * 4)
    let mut preds := ByteArray.emptyWithCapacity (p.n * 4)
    let nb := (p.n + B - 1) / B
    let t0 ← IO.monoMsNow
    for bi in [:nb] do
      let avail := min B (p.n - bi * B)
      let mut idx := ByteArray.emptyWithCapacity (B * 4)
      for i in [:B] do
        idx := idx ++ u32le (bi * B + (min i (avail - 1)))
      let xba ← F32.imagenetteGather p.raw idx B.toUSize p.side.toUSize
      let lastConv ← LowererSession.forwardF32 camSess camFn evalParams evalShapes xba xSh B.toUSize (c * h * w).toUSize
      for i in [:avail] do
        let logits ← F32.camLogits denseW denseB lastConv i.toUSize c.toUSize h.toUSize w.toUSize nClasses.toUSize
        let pred := (F32.argmaxN logits 0 nClasses.toUSize).toNat
        let label := p.lbl.data[(bi * B + i) * 4]!.toNat
        let camTrue ← F32.camCompute denseW lastConv i.toUSize c.toUSize h.toUSize w.toUSize nClasses.toUSize label.toUSize
        let camPred ← F32.camCompute denseW lastConv i.toUSize c.toUSize h.toUSize w.toUSize nClasses.toUSize pred.toUSize
        cams := cams ++ camTrue ++ camPred
        preds := preds ++ u32le pred
    let t1 ← IO.monoMsNow
    IO.FS.writeBinFile s!"{pfx}_cam_{p.name}.bin" cams
    IO.FS.writeBinFile s!"{pfx}_campred_{p.name}.bin" preds
    IO.println s!"cam: {p.name} {p.n} images → {pfx}_cam_{p.name}.bin ([n, 2, {h}, {w}] f32) ({(t1 - t0) / 1000} s)"

def parseArg (args : List String) (key : String) (dflt : String) : String :=
  match args.find? (·.startsWith (key ++ "=")) with
  | some a => (a.toRawSubstring.drop (key.length + 1)).toString
  | none => dflt

/-- `fieldn` images of a part, stratified by class in a seeded order — the "a few hundred
    field labels" arm. Returns indices into the part. -/
def stratifiedSubset (p : Part) (want : Nat) (seed : Nat) : Array Nat := Id.run do
  if want == 0 || want >= p.n then return Array.range p.n
  let order := permutation p.n (seed.toUInt64 * 7919 + 13)
  -- round-robin over classes in shuffled order until `want` are taken
  let mut byClass : Array (Array Nat) := Array.replicate nClasses #[]
  for k in order do
    let c := p.lbl.data[k * 4]!.toNat
    byClass := byClass.modify c (·.push k)
  let mut out : Array Nat := #[]
  let mut round := 0
  while out.size < want && round < p.n do
    for c in [:nClasses] do
      let bucket := byClass[c]!
      if out.size < want && round < bucket.size then
        out := out.push bucket[round]!
    round := round + 1
  return out

end PlantLeaf

open PlantLeaf in
def main (args : List String) : IO Unit := do
  let split := parseArg args "split" "grouped"
  let trainSel := parseArg args "train" "base"
  let init := parseArg args "init" "imagenet"
  let field := (parseArg args "field" "none")
  let fieldN := (parseArg args "fieldn" "0").toNat!
  let epochs := (parseArg args "epochs" "10").toNat!
  let B := (parseArg args "batch" "64").toNat!
  let lr := (ViTGradcheck.parseFloat? (parseArg args "lr" (if field == "none" then "0.001" else "0.0001"))).getD 0.001
  let seed := (parseArg args "seed" "1").toNat!
  let tag := parseArg args "tag" ""
  let outDir := parseArg args "out" ".lake/build"
  let evalOnly := args.contains "eval"
  let maxSteps := (parseArg args "steps" "0").toNat!
  let dataDir := "data/plant"
  unless split == "random" || split == "grouped" do
    throw <| IO.userError s!"split={split}: expected random or grouped"
  unless ["base", "comp", "aug", "compaug"].contains trainSel do
    throw <| IO.userError s!"train={trainSel}: expected base, comp, aug or compaug"
  let pv := if split == "random" then "pv" else "pvg"
  let spec := resnet34
  unless (← LowererSession.backendName) == "xla" do
    throw <| IO.userError "plant-leaf runs on the XLA backend only"
  IO.FS.createDirAll outDir
  let initTag := if init == "imagenet" || init == "scratch" then init else "ckpt"
  let pfx := s!"{outDir}/{spec.sanitizedName}_{split}_{trainSel}_{initTag}"
    ++ (if field == "none" then "" else s!"_field{field}" ++ (if fieldN == 0 then "" else s!"n{fieldN}"))
    ++ (if tag == "" then "" else s!"_{tag}")
  IO.eprintln s!"{spec.name}: {spec.bnLayers.size} BN layers, {nClasses} classes, split {split}, \
train {trainSel}, init {init}, field {field}{if fieldN > 0 then s!" (n={fieldN})" else ""}, \
{epochs} epochs, batch {B}, lr {lr}, seed {seed}{if evalOnly then " (eval only)" else ""}"

  -- ── graphs ──
  IO.FS.createDirAll ".lake/build"
  let gpfx := spec.buildPrefix
  let trainMlir := MlirCodegen.generateTrainStep spec B
    ("jit_" ++ spec.sanitizedName ++ "_train_step")
    (labelSmoothing := 0.0) (weightDecay := 0.0001) (useAdam := true)
  IO.FS.writeFile s!"{gpfx}_train_step.mlir" trainMlir
  IO.FS.writeFile s!"{gpfx}_fwd_eval.mlir" (MlirCodegen.generateEval spec B)
  let evalSess ← LowererSession.create (← NetSpec.graphArtifact gpfx "fwd_eval")
  -- ⚠ sized from the initialised buffer, not `spec.totalParams` (the SE-net disagreement)
  let p0 ← spec.heInitParams
  let nP := F32.size p0
  let nT := 3 * nP
  let nBn := spec.nBnStats
  let headFloats := 512 * nClasses + nClasses
  IO.eprintln s!"  {nP} params ({spec.totalParams} by NetSpec.totalParams), {nBn} BN stat floats, head {headFloats}"
  let allShapes := spec.shapesBA
  let evalShapes := spec.evalShapesBA
  let bnShapes := spec.bnShapesBA
  let xSh := spec.xShape B

  -- ── init: the ImageNet prefix, He, or a saved run ──
  let mut p := p0
  let mut bn ← F32.const nBn.toUSize 0.0
  if init == "imagenet" then
    -- the checkpoint is every conv and BN of the R34 and nothing else: the prefix must be
    -- exactly our packed layout minus the head, or the offsets are silently wrong
    unless nP - headFloats == imagenetPrefixFloats do
      throw <| IO.userError s!"packed layout has {nP - headFloats} backbone floats, the ImageNet prefix has {imagenetPrefixFloats}"
    p ← NetSpec.patchInitWithPretrainedPrefix p0 imagenetPrefix (imagenetPrefixFloats * 4)
    IO.eprintln s!"  bootstrap: {imagenetPrefixFloats} floats from {imagenetPrefix} (BN statistics start fresh)"
  else if init != "scratch" then
    p ← IO.FS.readBinFile s!"{init}_params.bin"
    bn ← IO.FS.readBinFile s!"{init}_bn_stats.bin"
    unless F32.size p == nP && bn.size / 4 == nBn do
      throw <| IO.userError s!"{init}_params.bin: {F32.size p} floats, expected {nP}"
    IO.eprintln s!"  init: {init}_params.bin + _bn_stats.bin"

  -- ── data ──
  let t0 ← IO.monoMsNow
  let valPart ← loadPart dataDir s!"{pv}_val" S
  let mut evalParts : Array Part := #[]
  -- the test tenth, its segmented twin, and the two-player Shapley's three counterfactuals
  for nm in [s!"{pv}_test", s!"{pv}_test_seg", s!"{pv}_test_bg", s!"{pv}_test_leaf", s!"{pv}_test_none"] do
    evalParts := evalParts.push (← loadPart dataDir nm S)
  let fieldPart := if field == "none" then "pd_all" else s!"pd_fold{field}_test"
  evalParts := evalParts.push (← loadPart dataDir fieldPart S)
  -- `extra=<path>`: any further 224-side Imagenette-format part (the Shapley probe), scored
  -- and written like the others under its own basename
  let extra := parseArg args "extra" ""
  if extra != "" then
    let dir := (System.FilePath.mk extra).parent.map toString |>.getD "."
    let base := (System.FilePath.mk extra).fileStem.getD "extra"
    evalParts := evalParts.push (← loadPart dir base S)
  let mut lmax : Nat := 0
  for i in [:valPart.n] do
    lmax := max lmax valPart.lbl.data[i * 4]!.toNat
  unless lmax + 1 == nClasses do
    throw <| IO.userError s!"labels run to {lmax}, expected {nClasses - 1}"
  let t1 ← IO.monoMsNow
  IO.eprintln s!"  val {valPart.n}; eval parts {evalParts.map (fun q => s!"{q.name} {q.n}")} ({t1 - t0} ms)"

  if evalOnly then
    p ← IO.FS.readBinFile s!"{pfx}_params.bin"
    bn ← IO.FS.readBinFile s!"{pfx}_bn_stats.bin"
    IO.eprintln s!"  loaded {pfx}_params.bin"
  else
    -- training parts: PlantVillage (with its composite / augmented twin), or a PlantDoc fold
    let mut trainParts : Array Part := #[]
    let mut trainIdx : Array Nat := #[]
    if field == "none" then
      trainParts := trainParts.push (← loadPart dataDir s!"{pv}_train" sTrain)
      if trainSel == "comp" || trainSel == "compaug" then
        trainParts := trainParts.push (← loadPart dataDir s!"{pv}_train_comp" sTrain)
      if trainSel == "aug" || trainSel == "compaug" then
        trainParts := trainParts.push (← loadPart dataDir s!"{pv}_train_aug" sTrain)
      trainIdx := Array.range (trainParts.foldl (· + ·.n) 0)
    else
      let fp ← loadPart dataDir s!"pd_fold{field}_train" sTrain
      trainParts := #[fp]
      trainIdx := stratifiedSubset fp fieldN seed
    let nTr := trainIdx.size
    let t2 ← IO.monoMsNow
    IO.eprintln s!"  train {nTr} images from {trainParts.map (·.name)} ({t2 - t1} ms)"
    let sess ← LowererSession.create (← NetSpec.graphArtifact gpfx "train_step")
    let mut m ← F32.const nP.toUSize 0.0
    let mut v ← F32.const nP.toUSize 0.0
    let bpE := nTr / B
    let total := epochs * bpE
    let warm := bpE
    let mut step : Nat := 0
    let mut curve : Array String := #[]
    -- the epoch is chosen on the PlantVillage val tenth; a field fine-tune is a fixed schedule
    let select := field == "none"
    let mut bestVal : Float := -1.0
    let mut bestEpoch : Nat := 0
    let mut bestP := p
    let mut bestBn := bn
    let tStart ← IO.monoMsNow
    for epoch in [:epochs] do
      let perm := permutation nTr (seed.toUInt64 * 1000003 + epoch.toUInt64 + 1)
      let idx := perm.map (trainIdx[·]!)
      let mut lossAcc : Float := 0.0
      let tE ← IO.monoMsNow
      for bi in [:bpE] do
        step := step + 1
        let lrNow := if step <= warm then lr * step.toFloat / warm.toFloat
          else lr * 0.5 * (1.0 + Float.cos (3.14159265358979 * (step - warm).toFloat / (total - warm).toFloat))
        let (xba, yb) ← gatherTrain trainParts idx (bi * B) B (seed * 1000003 + step)
        let packed := (p.append m).append v
        let out ← LowererSession.trainStepAdamF32 sess spec.trainFnName
                    packed allShapes xba xSh yb lrNow step.toFloat bnShapes B.toUSize
        if step == 1 then
          unless out.size / 4 == nT + 1 + nBn do
            throw <| IO.userError s!"train step returned {out.size / 4} floats, expected 3*{nP} + 1 + {nBn}"
        lossAcc := lossAcc + F32.read out nT.toUSize
        p := F32.slice out 0 nP
        m := F32.slice out nP nP
        v := F32.slice out (2 * nP) nP
        let batchBn := out.extract ((nT + 1) * 4) ((nT + 1 + nBn) * 4)
        -- a bootstrapped backbone has no statistics of its own: take the first batch's whole
        bn ← F32.ema bn batchBn (if step == 1 && init != "ckpt" then 1.0 else 0.1)
        if epoch == 0 && bi < 3 then
          let tb ← IO.monoMsNow
          IO.eprintln s!"    step {bi}: loss {fmt (F32.read out nT.toUSize) 4} ({tb - tE} ms so far)"
        if maxSteps > 0 && step >= maxSteps then
          throw <| IO.userError s!"stopped after {step} steps (steps={maxSteps})"
      let tV ← IO.monoMsNow
      let evalParams := p.append bn
      let (_, accVa) ← scoreSet evalSess spec evalParams evalShapes xSh valPart B
      let (_, accField) ← scoreSet evalSess spec evalParams evalShapes xSh evalParts.back! B
      let tD ← IO.monoMsNow
      let trainLoss := lossAcc / bpE.toFloat
      let better := select && accVa > bestVal
      if better then
        bestVal := accVa; bestEpoch := epoch + 1; bestP := p; bestBn := bn
      IO.eprintln s!"  epoch {epoch + 1}/{epochs}: loss {fmt trainLoss 4}  \
val acc {fmt accVa 2}%  {fieldPart} acc {fmt accField 2}%  \
({(tV - tE) / 1000} s train, {(tD - tV) / 1000} s eval){if better then " *" else ""}"
      curve := curve.push s!"{epoch + 1},{fmt trainLoss 5},{fmt accVa 3},{fmt accField 3}"
    let tEnd ← IO.monoMsNow
    if select then
      IO.eprintln s!"trained: {step} steps in {(tEnd - tStart) / 1000} s; best val epoch {bestEpoch} ({fmt bestVal 2}%)"
      p := bestP
      bn := bestBn
    else
      IO.eprintln s!"trained: {step} steps in {(tEnd - tStart) / 1000} s; fixed schedule, last epoch kept"
    IO.FS.writeBinFile s!"{pfx}_params.bin" p
    IO.FS.writeBinFile s!"{pfx}_bn_stats.bin" bn
    IO.FS.writeFile s!"{pfx}_curve.csv" (s!"epoch,loss,val_acc,{fieldPart}_acc\n" ++
      String.intercalate "\n" curve.toList ++ "\n")

  -- ── logits for the scorer, every evaluation part, from the chosen weights ──
  let evalParams := p.append bn
  for q in evalParts do
    let (logits, acc) ← scoreSet evalSess spec evalParams evalShapes xSh q B
    IO.FS.writeBinFile s!"{pfx}_logits_{q.name}.bin" logits
    IO.println s!"{spec.name} [{split}/{trainSel}/{initTag}{if field == "none" then "" else s!"/field{field}"}] \
{q.name}: accuracy {fmt acc 2}%  -> {pfx}_logits_{q.name}.bin"
  IO.eprintln s!"score: .venv/bin/python scripts/plant_score.py {pfx}_logits_{fieldPart}.bin --part {fieldPart} --restrict"
  -- ── cam=1: the CAM of every image of the test tenth and the field part ──
  if parseArg args "cam" "0" == "1" then
    camDump spec pfx p bn #[evalParts[0]!, evalParts.back!] B
