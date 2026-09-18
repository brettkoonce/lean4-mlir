import LeanMlir

/-! Chapter 4's CNN on Arabic sign-language letters, under two splits of the same
    images — planning/arasl_people_watching_demo.md.

    ArASL is 54,049 grey 64 × 64 crops of hands spelling the 32 letters of the Arabic
    alphabet, captured as video bursts. `preprocess_arasl.py` writes the same images
    twice: `random` is the stratified permutation every published number uses, `blocked`
    keeps each class's capture order together and cuts at the 80 % and 90 % marks. The
    net is chapter 4's CIFAR-CNN8-wide-BN with a one-channel stem and a 32-way head —
    the two edits the gravitational-wave demo made — through the ordinary cross-entropy
    train step with int32 labels; the chapter's linear and MLP rungs run through the
    same loop as the bracket. Zero new codegen: `demos/MainGwDetect.lean` with the specs
    and the file names swapped.

    The epoch is chosen on val and test is reported at it; the logits this writes for
    the test part are what `scripts/arasl_score.py` turns into the Wilson interval, the
    per-class table, the confused pairs and the leak audit.

    XLA backend only.
    `lake exe arasl-signs [net=cifar8w|mlp|linear] [split=blocked|random] [epochs=30]
                          [batch=64] [lr=0.001] [seed=1] [size=64] [tag=<name>]
                          [out=<dir>] [eval]`
    `eval` reloads `<out>/<prefix>_params.bin` + `_bn_stats.bin` and only writes the
    logits. Writes `<prefix>_curve.csv`, `_params.bin`, `_bn_stats.bin` and
    `_logits_{val,test}.bin` (f32 [N, 32]) under `out` (default `.lake/build`). -/

namespace AraslSigns

def nClasses : Nat := 32

/-- Chapter 4's CIFAR-CNN8-wide-BN verbatim: eight convolutions in four conv-conv-pool
    stages at 16, 16, 32, 32 channels into the 2 × 512 head; only the stem's input
    channels (1) and the output width (32) change. On the 64 × 64 native input the four
    pools end at 4 × 4, so the flatten is 32 × 4 × 4 = 512; `size=32` is the chapter's
    own input and its own 128-wide flatten. -/
def cifar8w (s : Nat) : NetSpec where
  name := if s == 64 then "arasl cifar8w" else s!"arasl cifar8w {s}"
  imageH := s
  imageW := s
  layers := [
    .convBn 1 16 3 1 .same,
    .convBn 16 16 3 1 .same,
    .maxPool 2 2,
    .convBn 16 16 3 1 .same,
    .convBn 16 16 3 1 .same,
    .maxPool 2 2,
    .convBn 16 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense (32 * (s / 16) * (s / 16)) 512 .relu,
    .dense 512 512 .relu,
    .dense 512 nClasses .identity
  ]

/-- The chapter 2–3 MLP on the flat image: 4096 → 512 → 512 → 32. -/
def mlp (s : Nat) : NetSpec where
  name := if s == 64 then "arasl mlp" else s!"arasl mlp {s}"
  imageH := s
  imageW := s
  layers := [.dense (s * s) 512 .relu, .dense 512 512 .relu, .dense 512 nClasses .identity]

/-- The chapter 1 linear classifier: 4096 → 32. -/
def linear (s : Nat) : NetSpec where
  name := if s == 64 then "arasl linear" else s!"arasl linear {s}"
  imageH := s
  imageW := s
  layers := [.dense (s * s) nClasses .identity]

def fmt (x : Float) (d : Nat) : String :=
  let m := (10.0 : Float) ^ d.toFloat
  let r := (x * m).round / m
  let s := toString r
  if s.length > 10 then (s.toRawSubstring.take 10).toString else s

/-- xorshift64 step. -/
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

/-- Gather a batch of `B` images by index from the flat f32 set, and their labels. -/
def gather (img lbl : ByteArray) (idx : Array Nat) (start B nPix : Nat) : ByteArray × ByteArray := Id.run do
  let mut x := ByteArray.emptyWithCapacity (B * nPix * 4)
  let mut y := ByteArray.emptyWithCapacity (B * 4)
  for i in [:B] do
    let k := idx[start + i]!
    x := x ++ F32.sliceImages img k 1 nPix
    y := y ++ F32.sliceLabels lbl k 1
  return (x, y)

/-- Run the eval graph over a whole part at batch `evalB` (tail zero-padded) and return
    the logits as f32 `[n, 32]` plus the accuracy. -/
def scoreSet (sess : LowererSession) (spec : NetSpec) (evalParams evalShapes xSh : ByteArray)
    (img lbl : ByteArray) (n evalB nPix : Nat) : IO (ByteArray × Float) := do
  let nC := nClasses
  let mut logits := ByteArray.emptyWithCapacity (n * nC * 4)
  let mut correct : Nat := 0
  let nb := (n + evalB - 1) / evalB
  for bi in [:nb] do
    let xba := F32.sliceImagesPad img (bi * evalB) evalB nPix n
    let out ← LowererSession.forwardF32 sess spec.evalFnName evalParams evalShapes xba xSh
                evalB.toUSize nC.toUSize
    let avail := min evalB (n - bi * evalB)
    logits := logits ++ out.extract 0 (avail * nC * 4)
    for i in [:avail] do
      let pred := F32.argmaxN out (i * nC).toUSize nC.toUSize
      let label := lbl.data[(bi * evalB + i) * 4]!.toNat
      if pred.toNat == label then correct := correct + 1
  return (logits, correct.toFloat / n.toFloat * 100.0)

def parseArg (args : List String) (key : String) (dflt : String) : String :=
  match args.find? (·.startsWith (key ++ "=")) with
  | some a => (a.toRawSubstring.drop (key.length + 1)).toString
  | none => dflt

/-- Read a part: images `[n, 1, s, s]` f32 and int32 labels, sizes cross-checked. -/
def loadPart (dataDir split part sfx : String) (nPix : Nat) : IO (ByteArray × ByteArray × Nat) := do
  let ip := s!"{dataDir}/{split}_{part}{sfx}.bin"
  let lp := s!"{dataDir}/labels_{split}_{part}{sfx}.bin"
  for f in [ip, lp] do
    unless ← System.FilePath.pathExists f do
      throw <| IO.userError s!"{f} missing — run: .venv/bin/python preprocess_arasl.py data/arasl data/arasl --stats"
  let lbl ← IO.FS.readBinFile lp
  let n := lbl.size / 4
  let img ← IO.FS.readBinFile ip
  unless img.size == n * nPix * 4 do
    throw <| IO.userError s!"{ip}: {img.size} bytes, expected {n} × {nPix} × 4 = {n * nPix * 4}"
  return (img, lbl, n)

end AraslSigns

open AraslSigns in
def main (args : List String) : IO Unit := do
  let netName := parseArg args "net" "cifar8w"
  let split := parseArg args "split" "blocked"
  let epochs := (parseArg args "epochs" "30").toNat!
  let B := (parseArg args "batch" "64").toNat!
  let lr := (ViTGradcheck.parseFloat? (parseArg args "lr" "0.001")).getD 0.001
  let seed := (parseArg args "seed" "1").toNat!
  let size := (parseArg args "size" "64").toNat!
  let tag := parseArg args "tag" ""
  let outDir := parseArg args "out" ".lake/build"
  let evalOnly := args.contains "eval"
  let maxSteps := (parseArg args "steps" "0").toNat!      -- debugging: stop after this many
  let dataDir := "data/arasl"
  unless split == "random" || split == "blocked" do
    throw <| IO.userError s!"split={split}: expected random or blocked"
  let spec := match netName with
    | "mlp" => mlp size
    | "linear" => linear size
    | _ => cifar8w size
  let nPix := size * size
  let sfx := if size == 64 then "" else s!"_{size}"
  unless (← LowererSession.backendName) == "xla" do
    throw <| IO.userError "arasl-signs runs on the XLA backend only"
  IO.FS.createDirAll outDir
  let pfx := s!"{outDir}/{spec.sanitizedName}_{split}" ++ (if tag == "" then "" else s!"_{tag}")
  IO.eprintln s!"{spec.name}: {spec.bnLayers.size} BN layers, {nClasses} classes, \
split {split}, {epochs} epochs, batch {B}, lr {lr}, seed {seed}\
{if evalOnly then " (eval only)" else ""}"

  -- ── graphs ──
  IO.FS.createDirAll ".lake/build"
  let gpfx := spec.buildPrefix                  -- already under .lake/build/
  let trainMlir := MlirCodegen.generateTrainStep spec B
    ("jit_" ++ spec.sanitizedName ++ "_train_step")
    (labelSmoothing := 0.0) (weightDecay := 0.0001) (useAdam := true)
  IO.FS.writeFile s!"{gpfx}_train_step.mlir" trainMlir
  IO.FS.writeFile s!"{gpfx}_fwd_eval.mlir" (MlirCodegen.generateEval spec B)
  let evalSess ← LowererSession.create (← NetSpec.graphArtifact gpfx "fwd_eval")
  -- ⚠ sized from the initialised buffer, not `spec.totalParams`: the two disagree on SE
  -- nets (totalParams counts squeeze-excite off the block input, `paramShapes` — which
  -- `heInitParams`, `shapesBA` and the emitted graph all follow — off the expanded width;
  -- 4.0M vs 7.1M for B0), and a wrong nP reads the loss from inside a weight tensor.
  let p0 ← spec.heInitParams
  let nP := F32.size p0
  let nT := 3 * nP
  let nBn := spec.nBnStats
  IO.eprintln s!"  {nP} params ({spec.totalParams} by NetSpec.totalParams), {nBn} BN stat floats"
  let allShapes := spec.shapesBA
  let evalShapes := spec.evalShapesBA
  let bnShapes := spec.bnShapesBA
  let xSh := spec.xShape B

  -- ── data: each part as one ByteArray, batches gathered by index ──
  let t0 ← IO.monoMsNow
  let (imgVa, lblVa, nVa) ← loadPart dataDir split "val" sfx nPix
  let (imgTe, lblTe, nTe) ← loadPart dataDir split "test" sfx nPix
  -- the two places a wrong constant reads as a low number rather than a crash: the
  -- input range (the chapter net was designed for [0, 1]) and the label range (0..31)
  let mut lo : Float := 1.0e9
  let mut hi : Float := -1.0e9
  for i in [:nPix] do
    let v := F32.read imgVa i.toUSize
    lo := min lo v
    hi := max hi v
  let mut lmax : Nat := 0
  for i in [:nVa] do
    lmax := max lmax lblVa.data[i * 4]!.toNat
  unless lmax + 1 == nClasses do
    throw <| IO.userError s!"labels run to {lmax}, expected {nClasses - 1}"
  let t1 ← IO.monoMsNow
  IO.eprintln s!"  val {nVa} / test {nTe} images of {size}×{size}, first image in [{fmt lo 3}, {fmt hi 3}], \
labels 0..{lmax} ({t1 - t0} ms)"

  let mut p := p0
  let mut bn ← F32.const nBn.toUSize 0.0
  if evalOnly then
    p ← IO.FS.readBinFile s!"{pfx}_params.bin"
    bn ← IO.FS.readBinFile s!"{pfx}_bn_stats.bin"
    IO.eprintln s!"  loaded {pfx}_params.bin"
  else
    let (imgTr, lblTr, nTr) ← loadPart dataDir split "train" sfx nPix
    let t2 ← IO.monoMsNow
    IO.eprintln s!"  train {nTr} images ({t2 - t1} ms)"
    let sess ← LowererSession.create (← NetSpec.graphArtifact gpfx "train_step")
    let mut m ← F32.const nP.toUSize 0.0
    let mut v ← F32.const nP.toUSize 0.0
    let bpE := nTr / B
    let total := epochs * bpE
    let warm := bpE
    let mut step : Nat := 0
    let mut curve : Array String := #[]
    -- the epoch is chosen on val: keep the best snapshot, report test at it
    let mut bestVal : Float := -1.0
    let mut bestEpoch : Nat := 0
    let mut bestP := p
    let mut bestBn := bn
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
        let (xba, yb) := gather imgTr lblTr idx (bi * B) B nPix
        let packed := (p.append m).append v
        let out ← LowererSession.trainStepAdamF32 sess spec.trainFnName
                    packed allShapes xba xSh yb lrNow step.toFloat bnShapes B.toUSize
        if step == 1 then
          unless out.size / 4 == nT + 1 + nBn do
            throw <| IO.userError s!"train step returned {out.size / 4} floats, expected 3*{nP} + 1 + {nBn} = {nT + 1 + nBn}: the packed layout does not match the graph"
        lossAcc := lossAcc + F32.read out nT.toUSize
        p := F32.slice out 0 nP
        m := F32.slice out nP nP
        v := F32.slice out (2 * nP) nP
        if nBn > 0 then
          let batchBn := out.extract ((nT + 1) * 4) ((nT + 1 + nBn) * 4)
          bn ← F32.ema bn batchBn (if step == 1 then 1.0 else 0.1)
        if epoch == 0 && bi < 3 then
          let tb ← IO.monoMsNow
          IO.eprintln s!"    step {bi}: loss {fmt (F32.read out nT.toUSize) 4} ({tb - tE} ms so far)"
        if maxSteps > 0 && step >= maxSteps then
          throw <| IO.userError s!"stopped after {step} steps (steps={maxSteps})"
      let tV ← IO.monoMsNow
      let evalParams := p.append bn
      let (_, accVa) ← scoreSet evalSess spec evalParams evalShapes xSh imgVa lblVa nVa B nPix
      let (_, accTe) ← scoreSet evalSess spec evalParams evalShapes xSh imgTe lblTe nTe B nPix
      let tD ← IO.monoMsNow
      let trainLoss := lossAcc / bpE.toFloat
      let star := if accVa > bestVal then " *" else ""
      if accVa > bestVal then
        bestVal := accVa; bestEpoch := epoch + 1; bestP := p; bestBn := bn
      IO.eprintln s!"  epoch {epoch + 1}/{epochs}: loss {fmt trainLoss 4}  \
val acc {fmt accVa 2}%  test acc {fmt accTe 2}%  \
({tV - tE} ms train, {tD - tV} ms eval){star}"
      curve := curve.push s!"{epoch + 1},{fmt trainLoss 5},{fmt accVa 3},{fmt accTe 3}"
    let tEnd ← IO.monoMsNow
    IO.eprintln s!"trained: {step} steps in {(tEnd - tStart) / 1000} s; best val epoch {bestEpoch} ({fmt bestVal 2}%)"
    p := bestP
    bn := bestBn
    IO.FS.writeBinFile s!"{pfx}_params.bin" p
    IO.FS.writeBinFile s!"{pfx}_bn_stats.bin" bn
    IO.FS.writeFile s!"{pfx}_curve.csv" ("epoch,loss,val_acc,test_acc\n" ++
      String.intercalate "\n" curve.toList ++ "\n")

  -- ── logits for the scorer, val and test, from the chosen epoch ──
  let evalParams := p.append bn
  for (nm, img, lbl, n) in [("val", imgVa, lblVa, nVa), ("test", imgTe, lblTe, nTe)] do
    let (logits, acc) ← scoreSet evalSess spec evalParams evalShapes xSh img lbl n B nPix
    IO.FS.writeBinFile s!"{pfx}_logits_{nm}.bin" logits
    IO.println s!"{spec.name} on the {split} split, {nm}: accuracy {fmt acc 2}%  -> {pfx}_logits_{nm}.bin"
  IO.eprintln s!"score: .venv/bin/python scripts/arasl_score.py {pfx}_logits_test.bin --split={split}"
