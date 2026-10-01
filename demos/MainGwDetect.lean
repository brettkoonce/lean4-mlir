import LeanMlir
import LeanMlir.CliArgs
import LeanMlir.ReferenceNets
import LeanMlir.SmallClassifier

/-! A CNN against the matched filter on LIGO strain.

    Two-channel (H1, L1) 64 × 128 constant-Q spectrograms of 2-s windows, half of
    them carrying an injected binary-black-hole chirp, from `scripts/datasets/preprocess_gw.py`. The
    net is chapter 4's CIFAR-CNN8-wide-BN (`cifar8w`) with a 2-channel stem and a two-class head,
    and the ordinary cross-entropy train step with int32 labels — the blackjack/2-D pattern of a
    host loop around the standard step, no `DatasetKind`, no new codegen. The score is not the accuracy printed here but the
    logits this writes for both val sets, which `scripts/demos/gw_metrics.py` thresholds at a
    false-alarm rate and bins by injected SNR beside the matched filter's closed form.

    XLA backend only.
    `lake exe gw-detect [arm=gauss|real] [epochs=20] [net=cifar8w|cnn|cifarbn|cifarbn3|cifarbn4] [batch=64] [lr=0.001] [ls=0.0]
                        [seed=1] [tag=<name>] [out=<dir>] [eval]`
    `arm` picks the TRAINING noise set; both val sets are always scored. `eval`
    reloads `<out>/<prefix>_params.bin` + `_bn_stats.bin` and only writes the logits.
    Writes `<prefix>_curve.csv`, `_params.bin`, `_bn_stats.bin` and
    `_logits_{gauss,real}_val.bin` (f32 [N, 2]) under `out` (default `.lake/build`). -/

open CliArgs SmallClassifier

namespace GwDetect

def H : Nat := 64
def W : Nat := 128
def C : Nat := 2
def nPix : Nat := C * H * W

/-- The small arm: the CIFAR CNN's body with a global-average-pool head, so no
    fan-in is tied to the input size. -/
def cnn : NetSpec where
  name := "gw cnn"
  imageH := H
  imageW := W
  layers := [
    .convBn C 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .maxPool 2 2,
    .convBn 64 128 3 1 .same,
    .convBn 128 128 3 1 .same,
    .maxPool 2 2,
    .globalAvgPool,
    .dense 128 2 .identity
  ]

/-- Chapter 4's CIFAR-10-BN on this input: two conv-conv-pool stacks, flatten, 512, 512,
    out — the diffs from the CIFAR-10 net are the 2-channel
    stem, the flatten width and the 2-way output. The flatten fan-in is C·H·W after the
    two pools (64 × 16 × 32 = 32768), which `NetSpec.validate` does not check, so it is
    spelled out here. -/
def cifarbn : NetSpec where
  name := "gw cifarbn"
  imageH := H
  imageW := W
  layers := [
    .convBn C 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense (64 * (H / 4) * (W / 4)) 512 .relu,
    .dense 512 512 .relu,
    .dense 512 2 .identity
  ]

/-- The same with a third stack and pool before the flatten (128 × 8 × 16 = 16384). -/
def cifarbn3 : NetSpec where
  name := "gw cifarbn3"
  imageH := H
  imageW := W
  layers := [
    .convBn C 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .maxPool 2 2,
    .convBn 64 128 3 1 .same,
    .convBn 128 128 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense (128 * (H / 8) * (W / 8)) 512 .relu,
    .dense 512 512 .relu,
    .dense 512 2 .identity
  ]

/-- Four stacks (32, 64, 128, 256 channels), each conv-conv-pool; the window ends at
    4 × 8 with 256 channels, so the flatten is 8192 wide. -/
def cifarbn4 : NetSpec where
  name := "gw cifarbn4"
  imageH := H
  imageW := W
  layers := [
    .convBn C 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .maxPool 2 2,
    .convBn 64 128 3 1 .same,
    .convBn 128 128 3 1 .same,
    .maxPool 2 2,
    .convBn 128 256 3 1 .same,
    .convBn 256 256 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense (256 * (H / 16) * (W / 16)) 512 .relu,
    .dense 512 512 .relu,
    .dense 512 2 .identity
  ]

/-- Chapter 4's CIFAR-CNN8-wide-BN verbatim: eight convolutions in four conv-conv-pool
    stages at 16, 16, 32, 32 channels into the 2 × 512 head; only the stem's input
    channels and the output width change (flatten = 32 × 4 × 8 = 1024). -/
def cifar8w : NetSpec := ReferenceNets.cifar8wOf "gw cifar8w" C H W 2

end GwDetect

open GwDetect in
def main (args : List String) : IO Unit := do
  let arm := parseArg args "arm" "gauss"
  let epochs := (parseArg args "epochs" "20").toNat!
  let netName := parseArg args "net" "cifar8w"
  let B := (parseArg args "batch" "64").toNat!
  let lr := floatArg args "lr" 0.001
  let seed := (parseArg args "seed" "1").toNat!
  let ls := floatArg args "ls" 0.0
  let tag := parseArg args "tag" ""
  let outDir := parseArg args "out" ".lake/build"
  let evalOnly := args.contains "eval"
  let maxSteps := (parseArg args "steps" "0").toNat!      -- debugging: stop after this many
  let dataDir := "data/gw"
  unless arm == "gauss" || arm == "real" do
    throw <| IO.userError s!"arm={arm}: expected gauss or real"
  let spec : NetSpec ← match netName with
    | "cnn" => pure cnn
    | "cifarbn" => pure cifarbn
    | "cifarbn3" => pure cifarbn3
    | "cifarbn4" => pure cifarbn4
    | "cifar8w" => pure cifar8w
    | other =>
      throw (IO.userError s!"net={other}: expected cifar8w, cnn, cifarbn, cifarbn3 or cifarbn4")
  unless (← LowererSession.backendName) == "xla" do
    throw <| IO.userError "gw-detect runs on the XLA backend only"
  IO.FS.createDirAll outDir
  let pfx := s!"{outDir}/{spec.sanitizedName}_{arm}" ++ (if tag == "" then "" else s!"_{tag}")
  IO.eprintln s!"{spec.name}: {spec.bnLayers.size} BN layers, \
train on {arm}, {epochs} epochs, batch {B}, lr {lr}, label smoothing {ls}, seed {seed}\
{if evalOnly then " (eval only)" else ""}"

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

  -- ── data: the whole set as one ByteArray, batches gathered by index ──
  let t0 ← IO.monoMsNow
  let need := [s!"{dataDir}/{arm}_train.bin", s!"{dataDir}/labels_train.bin",
               s!"{dataDir}/gauss_val.bin", s!"{dataDir}/real_val.bin", s!"{dataDir}/labels_val.bin"]
  for f in need do
    unless ← System.FilePath.pathExists f do
      throw <| IO.userError s!"{f} missing — run: .venv-gw/bin/python scripts/datasets/preprocess_gw.py"
  let lblTr ← IO.FS.readBinFile s!"{dataDir}/labels_train.bin"
  let lblVa ← IO.FS.readBinFile s!"{dataDir}/labels_val.bin"
  let nTr := lblTr.size / 4
  let nVa := lblVa.size / 4
  let valSets ← [("gauss", s!"{dataDir}/gauss_val.bin"), ("real", s!"{dataDir}/real_val.bin")].mapM
    fun (nm, path) => do
      let img ← IO.FS.readBinFile path
      unless img.size == nVa * nPix * 4 do
        throw <| IO.userError s!"{path}: {img.size} bytes, expected {nVa * nPix * 4}"
      pure (nm, img)
  let t1 ← IO.monoMsNow
  IO.eprintln s!"  val: {nVa} windows × 2 sets ({t1 - t0} ms)"

  let mut p := p0
  let mut bn ← F32.const nBn.toUSize 0.0
  if evalOnly then
    p ← IO.FS.readBinFile s!"{pfx}_params.bin"
    bn ← IO.FS.readBinFile s!"{pfx}_bn_stats.bin"
    IO.eprintln s!"  loaded {pfx}_params.bin"
  else
    let imgTr ← IO.FS.readBinFile s!"{dataDir}/{arm}_train.bin"
    unless imgTr.size == nTr * nPix * 4 do
      throw <| IO.userError s!"{arm}_train.bin: {imgTr.size} bytes, expected {nTr * nPix * 4}"
    let t2 ← IO.monoMsNow
    IO.eprintln s!"  train: {nTr} windows of {arm} ({t2 - t1} ms)"
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
        let (xba, yb) := gather imgTr lblTr idx (bi * B) B nPix
        let packed := (p.append m).append v
        let out ← LowererSession.trainStepAdamF32 sess spec.trainFnName
                    packed allShapes xba xSh yb lrNow step.toFloat bnShapes B.toUSize
        if step == 1 then
          unless out.size / 4 == nT + 1 + nBn do
            throw <| IO.userError s!"train step returned {out.size / 4} floats, expected 3*{nP} + 1 + {nBn} = {nT + 1 + nBn}: the packed layout does not match the graph"
        lossAcc := lossAcc + F32.read out nT.toUSize
        (p, m, v) := F32.unpackAdam out nP
        let batchBn := out.extract ((nT + 1) * 4) ((nT + 1 + nBn) * 4)
        bn ← F32.ema bn batchBn (if step == 1 then 1.0 else 0.1)
        if epoch == 0 && bi < 3 then
          let tb ← IO.monoMsNow
          IO.eprintln s!"    step {bi}: loss {fmt (F32.read out nT.toUSize) 4} ({tb - tE} ms so far)"
        if maxSteps > 0 && step >= maxSteps then
          throw <| IO.userError s!"stopped after {step} steps (steps={maxSteps})"
      let tV ← IO.monoMsNow
      let evalParams := p.append bn
      let mut accs : Array (String × Float) := #[]
      for (nm, img) in valSets do
        let (_, acc) ← scoreSet evalSess spec evalParams evalShapes xSh img lblVa nVa B nPix
        accs := accs.push (nm, acc)
      let tD ← IO.monoMsNow
      let trainLoss := lossAcc / bpE.toFloat
      IO.eprintln s!"  epoch {epoch + 1}/{epochs}: loss {fmt trainLoss 4}  \
val acc gauss {fmt (accs[0]!).2 2}%  real {fmt (accs[1]!).2 2}%  \
({tV - tE} ms train, {tD - tV} ms eval)"
      curve := curve.push s!"{epoch + 1},{fmt trainLoss 5},{fmt (accs[0]!).2 3},{fmt (accs[1]!).2 3}"
    let tEnd ← IO.monoMsNow
    IO.eprintln s!"trained: {step} steps in {(tEnd - tStart) / 1000} s"
    IO.FS.writeBinFile s!"{pfx}_params.bin" p
    IO.FS.writeBinFile s!"{pfx}_bn_stats.bin" bn
    IO.FS.writeFile s!"{pfx}_curve.csv" ("epoch,loss,val_acc_gauss,val_acc_real\n" ++
      String.intercalate "\n" curve.toList ++ "\n")

  -- ── logits for the scorer, both val sets ──
  let evalParams := p.append bn
  for (nm, img) in valSets do
    let (logits, acc) ← scoreSet evalSess spec evalParams evalShapes xSh img lblVa nVa B nPix
    IO.FS.writeBinFile s!"{pfx}_logits_{nm}_val.bin" logits
    IO.println s!"{spec.name} trained on {arm}, scored on {nm} val: accuracy {fmt acc 2}%  \
-> {pfx}_logits_{nm}_val.bin"
  IO.eprintln s!"score: .venv-gw/bin/python scripts/demos/gw_metrics.py table \
--logits=gauss:{pfx}_logits_gauss_val.bin,real:{pfx}_logits_real_val.bin"
