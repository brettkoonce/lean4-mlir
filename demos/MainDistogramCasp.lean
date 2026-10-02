import LeanMlir
import LeanMlir.CliArgs
import LeanMlir.SmallClassifier

/-! The CASP16 distogram demo (planning/casp16_distogram_demo.md): a residue-pair distance
    classifier on frozen ESM-2 35M features, scored against the CASP16 field.

    The net is `pairTile` (two dense maps and an outer sum, the AlphaFold-1 / trRosetta stem)
    into a 1×1 `convBn`, the chapter residual body at stride 1, and a 1×1 head over 66 classes:
    64 Cβ–Cβ distance bins on [2, 22) Å, ">22 Å", and "unobserved", which carries weight 0 in
    `perPixelWeightedCE` and so masks missing residues and the padding of short chains. Training
    crops are `crop × crop` windows anywhere in a chain's L × L map, gathered in C
    (`lean_casp_gather`) from the packed files `scripts/datasets/casp16_pack.py` writes;
    inference tiles the same window over a target at stride `crop/2` and averages the logits.
    The host loop is the GW / NQS pattern around the standard seg train step.

    XLA backend only.
    `lake exe distogram-casp smoke` — emit a tiny instance, run one step, finite-difference the
    gradient of the pairTile parameters (and a few others) against the step's Adam moment.
    `lake exe distogram-casp train [list=train_full|train] [val=val] [epochs=30] [batch=32]
        [lr=0.001] [ch=64] [units=16] [crop=64] [crops=1] [seed=1] [tag=<name>] [out=<dir>]
        [steps=N]`
      — `crops` windows per chain per epoch; val every epoch on one diagonal window per chain
      of the `val` index (masked CE and top-L/5 long-range contact precision); `val=tiny` on a
      training subset is the memorization probe. Writes `<prefix>_curve.csv`,
      `_params.bin`, `_bn_stats.bin` under `out` (default `.lake/build`).
    `lake exe distogram-casp predict [same net args] [stride=32] [pool=targets|valsub]` — reloads
      the params and writes, per evaluation unit, the summed window logits `<prefix>_<pool>/<EU>.acc.bin`
      (f32 [L, L, 66]) and window counts `.cnt.bin` (f32 [L, L]) for
      `scripts/demos/casp16_predict.py`. -/

open CliArgs SmallClassifier

namespace DistogramCasp

/-- Distance classes: 64 bins + ">22 Å" + "unobserved" (weight 0). -/
def nClasses : Nat := 66
def unobserved : Nat := 65
/-- Classes whose lower edge is under 8 Å: bins 0..19 (bin 19 starts at 7.94 Å). -/
def contactMaxClass : Nat := 19
/-- Long-range: |i − j| ≥ 24 (the CASP RR definition). -/
def longRangeSep : Nat := 24
def classWeights : List Float := List.replicate (nClasses - 1) 1.0 ++ [0.0]

/-- The distogram net at crop `L`, feature width `D`, `ch` channels, `units` residual units. -/
def distogramNet (name : String) (L D ch units : Nat) : NetSpec where
  name := name
  imageH := L
  imageW := L
  layers := [
    .pairTile L D ch,
    .convBn ch ch 1 1 .same,
    .residualBlock ch ch units 1,
    .conv2d ch nClasses 1 .same .identity ]

/-- `ffi/f32_helpers.c`: B crops from the packed set. `pos` = u32 [B, 3] (chain row, r1, r2);
    `exact = 0` reduces r1, r2 to offsets in [0, L − crop], `exact = 1` takes them as offsets.
    Returns the pairTile input f32 [B, 2·crop·D] and the int32 labels [B, crop, crop]. -/
@[extern "lean_casp_gather"]
opaque caspGather (feat lab idx pos : @& ByteArray) (B crop D unobs exact : USize)
  : IO (ByteArray × ByteArray)

/-- f32 [4] = (contact hits, contacts taken, Σ masked CE, observed pairs) over the first `B`
    crops of `logits` [B, NC, crop, crop] against `y` [B, crop, crop]. -/
@[extern "lean_casp_val_metrics"]
opaque caspValMetrics (logits y : @& ByteArray) (B NC crop sep contactMax unobs : USize)
  : IO ByteArray

/-- Add the first `nValid` windows' logits into a target's accumulators (in place when
    unshared): `acc` f32 [L, L, NC], `cnt` f32 [L, L]; `pos` are exact offsets. -/
@[extern "lean_casp_accumulate"]
opaque caspAccumulate (acc cnt : ByteArray) (logits pos : @& ByteArray) (nValid L crop NC : USize)
  : IO (ByteArray × ByteArray)

def readI64 (ba : ByteArray) (off : Nat) : Nat := Id.run do
  let mut v : Nat := 0
  for k in [:8] do
    v := v + ((ba.get! (off + k)).toNat <<< (8 * k))
  return v

def pushU32 (ba : ByteArray) (v : Nat) : ByteArray :=
  let u := v.toUInt32
  ba.push u.toUInt8 |>.push (u >>> 8).toUInt8 |>.push (u >>> 16).toUInt8 |>.push (u >>> 24).toUInt8

/-- One index file over a feature/label pool: 32 bytes per chain (feat offset, label offset,
    L, csv row). -/
structure ChainIdx where
  idx : ByteArray
  n : Nat

def ChainIdx.load (path : String) : IO ChainIdx := do
  let idx ← IO.FS.readBinFile path
  unless idx.size % 32 == 0 do
    throw <| IO.userError s!"{path}: {idx.size} bytes is not a multiple of 32"
  pure { idx, n := idx.size / 32 }

def ChainIdx.len (ci : ChainIdx) (c : Nat) : Nat := readI64 ci.idx (32 * c + 16)

def xorshift (s : UInt64) : UInt64 :=
  let s := s ^^^ (s <<< 13)
  let s := s ^^^ (s >>> 7)
  s ^^^ (s <<< 17)

/-- Random crops for one batch: rows `(chain, r1, r2)` from a permutation, the draws reduced to
    offsets in C. -/
def randomPos (perm : Array Nat) (start B : Nat) (seed : UInt64) : ByteArray := Id.run do
  let mut s := seed ||| 1
  let mut pos := ByteArray.empty
  for b in [:B] do
    let c := perm[(start + b) % perm.size]!
    s := xorshift s
    let r1 := (s >>> 11).toNat % 1000003
    s := xorshift s
    let r2 := (s >>> 11).toNat % 1000003
    pos := pushU32 (pushU32 (pushU32 pos c) r1) r2
  return pos

/-- The diagonal window at the start of chains `start, …, start + B − 1` (the val proxy); a
    short tail is padded by repeating the last chain, and `valid` says how many are real. -/
def diagPos (start B n : Nat) : ByteArray × Nat := Id.run do
  let mut pos := ByteArray.empty
  let mut valid := 0
  for b in [:B] do
    let real := start + b < n
    let c := if real then start + b else n - 1
    if real then valid := valid + 1
    pos := pushU32 (pushU32 (pushU32 pos c) 0) 0
  return (pos, valid)

/-- Window offsets covering `[0, L)` at `stride`, the last one clamped so the chain end is
    covered. -/
def windowOffsets (L crop stride : Nat) : List Nat :=
  if L <= crop then [0]
  else
    let steps := (L - crop + stride - 1) / stride
    let offs := (List.range (steps + 1)).map (fun k => min (k * stride) (L - crop))
    offs.eraseDups

structure Net where
  spec : NetSpec
  crop : Nat
  D : Nat
  pfx : String

def netFromArgs (args : List String) (listName : String) : Net :=
  let ch := (parseArg args "ch" "64").toNat!
  let units := (parseArg args "units" "16").toNat!
  let crop := (parseArg args "crop" "64").toNat!
  let D := (parseArg args "dim" "489").toNat!
  let tag := parseArg args "tag" ""
  let outDir := parseArg args "out" ".lake/build"
  let spec := distogramNet s!"distogram r{units}x{ch}" crop D ch units
  { spec, crop, D, pfx := s!"{outDir}/{spec.sanitizedName}_{listName}" ++ (if tag == "" then "" else s!"_{tag}") }

/-- Val pass: one diagonal window per val chain through the eval forward. Returns
    (masked CE, contact precision). -/
def valPass (evalSess : LowererSession) (net : Net) (evalParams evalShapes xSh : ByteArray)
    (feat lab : ByteArray) (val : ChainIdx) (B : Nat) : IO (Float × Float) := do
  let mut hits := 0.0; let mut taken := 0.0; let mut ce := 0.0; let mut nobs := 0.0
  let nb := (val.n + B - 1) / B
  for bi in [:nb] do
    let (pos, valid) := diagPos (bi * B) B val.n
    let (xba, yb) ← caspGather feat lab val.idx pos B.toUSize net.crop.toUSize net.D.toUSize
      unobserved.toUSize 1
    let logits ← LowererSession.forwardF32 evalSess net.spec.evalFnName evalParams evalShapes xba xSh
      B.toUSize (nClasses * net.crop * net.crop).toUSize
    let m ← caspValMetrics logits yb valid.toUSize nClasses.toUSize net.crop.toUSize
      longRangeSep.toUSize contactMaxClass.toUSize unobserved.toUSize
    hits := hits + F32.read m 0; taken := taken + F32.read m 1
    ce := ce + F32.read m 2; nobs := nobs + F32.read m 3
  pure (ce / max nobs 1.0, hits / max taken 1.0)

def train (args : List String) : IO Unit := do
  let listName := parseArg args "list" "train_full"
  let epochs := (parseArg args "epochs" "30").toNat!
  let B := (parseArg args "batch" "32").toNat!
  let lr := floatArg args "lr" 0.001
  let seed := (parseArg args "seed" "1").toNat!
  let dataDir := parseArg args "data" "data/casp16/packed"
  let maxSteps := (parseArg args "steps" "0").toNat!
  let cropsPerChain := (parseArg args "crops" "1").toNat!
  let net := netFromArgs args listName
  let spec := net.spec
  unless (← LowererSession.backendName) == "xla" do
    throw <| IO.userError "distogram-casp runs on the XLA backend only"
  IO.FS.createDirAll ".lake/build"
  spec.validate!
  IO.eprintln s!"{spec.name}: {spec.archStr}; {spec.bnLayers.size} BN layers; list {listName}, \
{epochs} epochs, batch {B}, lr {lr}, {cropsPerChain} crop(s) per chain per epoch, seed {seed}"
  -- ── graphs ──
  let gpfx := spec.buildPrefix
  IO.FS.writeFile s!"{gpfx}_train_step.mlir" <| MlirCodegen.generateTrainStep spec B
    ("jit_" ++ spec.sanitizedName ++ "_train_step") (labelSmoothing := 0.0) (weightDecay := 0.0001)
    (useAdam := true) (useSeg := true) (segLoss := .weightedCE classWeights) (gradClipNorm := 1.0)
  IO.FS.writeFile s!"{gpfx}_fwd_eval.mlir" (MlirCodegen.generateEval spec B)
  let sess ← LowererSession.create (← NetSpec.graphArtifact gpfx "train_step")
  let evalSess ← LowererSession.create (← NetSpec.graphArtifact gpfx "fwd_eval")
  let p0 ← spec.heInitParams
  let nP := F32.size p0
  let nT := 3 * nP
  let nBn := spec.nBnStats
  IO.eprintln s!"  {nP} params ({spec.totalParams} by NetSpec.totalParams), {nBn} BN stat floats"
  let allShapes := spec.shapesBA
  let evalShapes := spec.evalShapesBA
  let bnShapes := spec.bnShapesBA
  let xSh := spec.xShape B
  -- ── data: the packed pool and two index files ──
  let t0 ← IO.monoMsNow
  let feat ← IO.FS.readBinFile s!"{dataDir}/pool_feat.bin"
  let lab ← IO.FS.readBinFile s!"{dataDir}/pool_lab.bin"
  unless feat.size % (2 * net.D) == 0 do
    throw <| IO.userError s!"pool_feat.bin: {feat.size} bytes is not a multiple of 2·{net.D}; dim= is wrong"
  let tr ← ChainIdx.load s!"{dataDir}/{listName}_idx.bin"
  let va ← ChainIdx.load s!"{dataDir}/{parseArg args "val" "val"}_idx.bin"
  let t1 ← IO.monoMsNow
  IO.eprintln s!"  {tr.n} train chains, {va.n} val chains; features {feat.size / 1000000} MB, \
labels {lab.size / 1000000} MB ({t1 - t0} ms)"
  -- ── loop ──
  let mut p := p0
  let mut bn ← F32.const nBn.toUSize 0.0
  let mut m ← F32.const nP.toUSize 0.0
  let mut v ← F32.const nP.toUSize 0.0
  let bpE := (tr.n * cropsPerChain) / B
  let total := epochs * bpE
  let warm := bpE
  let mut step : Nat := 0
  let mut curve : Array String := #[]
  let tStart ← IO.monoMsNow
  for epoch in [:epochs] do
    let perm := permutation tr.n (seed.toUInt64 * 1000003 + epoch.toUInt64 + 1)
    let mut lossAcc : Float := 0.0
    let tE ← IO.monoMsNow
    for bi in [:bpE] do
      step := step + 1
      let lrNow := if step <= warm then lr * step.toFloat / warm.toFloat
        else lr * 0.5 * (1.0 + Float.cos (3.14159265358979 * (step - warm).toFloat / (total - warm).toFloat))
      let pos := randomPos perm (bi * B) B (seed.toUInt64 * 7919 + step.toUInt64 * 104729)
      let (xba, yb) ← caspGather feat lab tr.idx pos B.toUSize net.crop.toUSize net.D.toUSize
        unobserved.toUSize 0
      let packed := (p.append m).append v
      let out ← LowererSession.trainStepAdamF32Seg sess spec.trainFnName
        packed allShapes xba xSh yb lrNow step.toFloat bnShapes B.toUSize net.crop.toUSize net.crop.toUSize
      if step == 1 then
        unless out.size / 4 == nT + 1 + nBn do
          throw <| IO.userError s!"train step returned {out.size / 4} floats, expected 3·{nP} + 1 + {nBn}"
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
    let (valCe, valPrec) ← valPass evalSess net (p.append bn) evalShapes xSh feat lab va B
    let tD ← IO.monoMsNow
    let trainLoss := lossAcc / bpE.toFloat
    IO.eprintln s!"  epoch {epoch + 1}/{epochs}: loss {fmt trainLoss 4}  val CE {fmt valCe 4}  \
val top-L/5 LR precision {fmt (100.0 * valPrec) 2}%  ({(tV - tE) / 1000} s train, {tD - tV} ms val, \
{fmt ((tV - tE).toFloat / bpE.toFloat) 1} ms/step)"
    curve := curve.push s!"{epoch + 1},{fmt trainLoss 5},{fmt valCe 5},{fmt valPrec 4}"
    IO.FS.writeBinFile s!"{net.pfx}_params.bin" p
    IO.FS.writeBinFile s!"{net.pfx}_bn_stats.bin" bn
    IO.FS.writeFile s!"{net.pfx}_curve.csv" ("epoch,loss,val_ce,val_top_l5_lr_precision\n" ++
      String.intercalate "\n" curve.toList ++ "\n")
  let tEnd ← IO.monoMsNow
  IO.println s!"trained: {step} steps in {(tEnd - tStart) / 1000} s -> {net.pfx}_params.bin"
  IO.eprintln s!"predict: lake exe distogram-casp predict list={listName} {String.intercalate " " (args.filter fun a => a.startsWith "ch=" || a.startsWith "units=" || a.startsWith "tag=")}"

/-- Tiled inference over the evaluation units: every (i0, j0) window at `stride`, logits summed
    into a per-target accumulator the Python side averages and symmetrizes. -/
def predict (args : List String) : IO Unit := do
  let listName := parseArg args "list" "train_full"
  let B := (parseArg args "batch" "32").toNat!
  let stride := (parseArg args "stride" "32").toNat!
  let dataDir := parseArg args "data" "data/casp16/packed"
  let net := netFromArgs args listName
  let spec := net.spec
  unless (← LowererSession.backendName) == "xla" do
    throw <| IO.userError "distogram-casp runs on the XLA backend only"
  let gpfx := spec.buildPrefix
  spec.validate!
  IO.FS.writeFile s!"{gpfx}_fwd_eval.mlir" (MlirCodegen.generateEval spec B)
  let evalSess ← LowererSession.create (← NetSpec.graphArtifact gpfx "fwd_eval")
  let p ← IO.FS.readBinFile s!"{net.pfx}_params.bin"
  let bn ← IO.FS.readBinFile s!"{net.pfx}_bn_stats.bin"
  let evalParams := p.append bn
  let evalShapes := spec.evalShapesBA
  let xSh := spec.xShape B
  let pool := parseArg args "pool" "targets"      -- `targets` (the EUs) or `valsub` (val chains, for fold tuning)
  let feat ← IO.FS.readBinFile s!"{dataDir}/{pool}_feat.bin"
  let lab ← IO.FS.readBinFile s!"{dataDir}/{pool}_lab.bin"
  let tg ← ChainIdx.load s!"{dataDir}/{pool}_idx.bin"
  let names := ((← IO.FS.readFile s!"{dataDir}/{pool}_order.txt").splitOn "\n").filter (· != "")
  unless names.length == tg.n do
    throw <| IO.userError s!"{pool}_order.txt lists {names.length} entries, {pool}_idx.bin has {tg.n}"
  let outDir := s!"{net.pfx}_{pool}"
  IO.FS.createDirAll outDir
  let t0 ← IO.monoMsNow
  let mut nWin := 0
  for c in [:tg.n] do
    let L := tg.len c
    let offs := windowOffsets L net.crop stride
    let wins := (offs.flatMap fun i0 => offs.map fun j0 => (i0, j0)).toArray
    let mut acc ← F32.const (L * L * nClasses).toUSize 0.0
    let mut cnt ← F32.const (L * L).toUSize 0.0
    let nb := (wins.size + B - 1) / B
    for bi in [:nb] do
      let mut pos := ByteArray.empty
      let mut valid := 0
      for b in [:B] do
        let k := bi * B + b
        let (i0, j0) := if k < wins.size then wins[k]! else wins[wins.size - 1]!
        if k < wins.size then valid := valid + 1
        pos := pushU32 (pushU32 (pushU32 pos c) i0) j0
      let (xba, _) ← caspGather feat lab tg.idx pos B.toUSize net.crop.toUSize net.D.toUSize
        unobserved.toUSize 1
      let logits ← LowererSession.forwardF32 evalSess spec.evalFnName evalParams evalShapes xba xSh
        B.toUSize (nClasses * net.crop * net.crop).toUSize
      (acc, cnt) ← caspAccumulate acc cnt logits pos valid.toUSize L.toUSize net.crop.toUSize nClasses.toUSize
    IO.FS.writeBinFile s!"{outDir}/{names[c]!}.acc.bin" acc
    IO.FS.writeBinFile s!"{outDir}/{names[c]!}.cnt.bin" cnt
    nWin := nWin + wins.size
    IO.eprintln s!"  {names[c]!}: L {L}, {wins.size} windows"
  let t1 ← IO.monoMsNow
  IO.println s!"predicted {tg.n} EUs, {nWin} windows in {(t1 - t0) / 1000} s -> {outDir}/"
  IO.eprintln s!"assemble + score: .venv-casp/bin/python scripts/demos/casp16_predict.py {outDir}"

def pad (s : String) (n : Nat) : String := s ++ String.ofList (List.replicate (n - s.length) ' ')
def padL (s : String) (n : Nat) : String := String.ofList (List.replicate (n - s.length) ' ') ++ s

/-- int32 LE labels `[B, L, L]`, a fixed pattern over the 66 classes (so some pairs are masked). -/
def smokeLabels (B L : Nat) : ByteArray := Id.run do
  let mut yb := ByteArray.empty
  for k in [:B * L * L] do
    let c : UInt32 := ((k * 7 + 3) % nClasses).toUInt32
    yb := yb.push c.toUInt8 |>.push (c >>> 8).toUInt8 |>.push (c >>> 16).toUInt8 |>.push (c >>> 24).toUInt8
  return yb

/-- Emit a tiny instance (L = 8, D = 5, 4 channels, one residual unit, B = 2), run the seg
    train step once at zero moments — Adam's first moment is then `(1 − β₁)·g`, so `g = 10·m` —
    and compare that gradient with central differences of the step's own loss at ±ε on a
    handful of coordinates: the pairTile `W` and `Wj`, a residual-body weight, and the head
    bias. Tolerance 2 % + 2e-4 absolute (f32 differences of a loss near 5 carry ~1e-4 of
    noise). Also runs the eval forward and checks the logits' size. -/
def smoke : IO Unit := do
  let L := 8; let D := 5; let ch := 4; let B := 2
  let spec := distogramNet "distogram smoke" L D ch 1
  spec.validate!
  unless (← LowererSession.backendName) == "xla" do
    throw <| IO.userError "distogram-casp runs on the XLA backend only"
  IO.FS.createDirAll ".lake/build"
  let gpfx := spec.buildPrefix
  IO.FS.writeFile s!"{gpfx}_train_step.mlir" <| MlirCodegen.generateTrainStep spec B
    ("jit_" ++ spec.sanitizedName ++ "_train_step") (labelSmoothing := 0.0) (weightDecay := 0.0)
    (useAdam := true) (useSeg := true) (segLoss := .weightedCE classWeights)
  IO.FS.writeFile s!"{gpfx}_fwd_eval.mlir" (MlirCodegen.generateEval spec B)
  let sess ← LowererSession.create (← NetSpec.graphArtifact gpfx "train_step")
  let evalSess ← LowererSession.create (← NetSpec.graphArtifact gpfx "fwd_eval")
  let p0 ← spec.heInitParams
  let nP := F32.size p0
  let nT := 3 * nP
  let nBn := spec.nBnStats
  let allShapes := spec.shapesBA
  let bnShapes := spec.bnShapesBA
  let xSh := spec.xShape B
  let x ← F32.heInit 7 (B * 2 * L * D).toUSize 1.0
  let y := smokeLabels B L
  let m0 ← F32.const nP.toUSize 0.0
  let step (p : ByteArray) : IO (Float × ByteArray) := do
    let packed := (p.append m0).append m0
    let out ← LowererSession.trainStepAdamF32Seg sess spec.trainFnName
      packed allShapes x xSh y 0.001 1.0 bnShapes B.toUSize L.toUSize L.toUSize
    unless out.size / 4 == nT + 1 + nBn do
      throw <| IO.userError s!"train step returned {out.size / 4} floats, expected {nT + 1 + nBn}"
    pure (F32.read out nT.toUSize, (F32.unpackAdam out nP).2.1)
  let (loss0, m1) ← step p0
  IO.println s!"{spec.name}: {spec.archStr}; {nP} params ({spec.totalParams} by totalParams), \
{nBn} BN floats; loss at init {loss0} (ln 66 = 4.19)"
  -- parameter layout: pairTile W [D·ch] | Wj [D·ch] | convBn … | head W, b
  let coords : List (String × Nat) :=
    [("pairTile.W[0]", 0), ("pairTile.W[1]", 1), ("pairTile.W[last]", D * ch - 1),
     ("pairTile.Wj[0]", D * ch), ("pairTile.Wj[3]", D * ch + 3), ("pairTile.Wj[last]", 2 * D * ch - 1),
     ("body[nP/2]", nP / 2), ("head.b[last]", nP - 1), ("head.b[last-1]", nP - 2)]
  let eps := 0.01
  let unit ← F32.const 1 1.0
  let mut bad := 0
  IO.println s!"  {pad "coordinate" 20} {padL "10·m (grad)" 14} {padL "central FD" 14} {padL "|diff|" 10} {padL "tol" 10}"
  for (nm, k) in coords do
    let pPlus ← F32.axpySlice (p0.extract 0 p0.size) k.toUSize unit 0 1 eps
    let pMinus ← F32.axpySlice (p0.extract 0 p0.size) k.toUSize unit 0 1 (-eps)
    let (lp, _) ← step pPlus
    let (lm, _) ← step pMinus
    let fd := (lp - lm) / (2.0 * eps)
    let g := 10.0 * F32.read m1 k.toUSize
    let tol := 0.02 * g.abs + 0.0002
    if (fd - g).abs > tol then bad := bad + 1
    IO.println s!"  {pad nm 20} {padL (fmt g 6) 14} {padL (fmt fd 6) 14} {padL (fmt (fd - g).abs 6) 10} {padL (fmt tol 6) 10}\
{if (fd - g).abs > tol then "  ✗" else ""}"
  -- eval ≡ train: with the BN running stats set to this batch's own statistics (what the
  -- train step returns after its loss), the eval forward's masked CE on the same batch must
  -- equal the train step's loss.
  let packed0 := (p0.append m0).append m0
  let out0 ← LowererSession.trainStepAdamF32Seg sess spec.trainFnName
    packed0 allShapes x xSh y 0.001 1.0 bnShapes B.toUSize L.toUSize L.toUSize
  let batchBn := out0.extract ((nT + 1) * 4) ((nT + 1 + nBn) * 4)
  let evalParams := p0.append batchBn
  let logits ← LowererSession.forwardF32 evalSess spec.evalFnName evalParams spec.evalShapesBA x xSh
    B.toUSize (nClasses * L * L).toUSize
  let mtr ← caspValMetrics logits y B.toUSize nClasses.toUSize L.toUSize longRangeSep.toUSize
    contactMaxClass.toUSize unobserved.toUSize
  let evalCe := F32.read mtr 2 / F32.read mtr 3
  IO.println s!"  eval logits: {F32.size logits} floats (expected {B * nClasses * L * L}); \
eval-forward masked CE at batch BN stats {evalCe} vs train-step loss {loss0} (|diff| {(evalCe - loss0).abs})"
  if (evalCe - loss0).abs > 1e-3 * loss0 then
    throw <| IO.userError "eval forward disagrees with the train forward"
  if bad > 0 then
    throw <| IO.userError s!"pairTile gradient check FAILED on {bad} coordinate(s)"
  IO.println "smoke: pairTile gradient check passed; eval forward ≡ train forward"


end DistogramCasp

open DistogramCasp in
def main (args : List String) : IO Unit := do
  if args.contains "smoke" then smoke
  else if args.contains "predict" then predict args
  else train args
