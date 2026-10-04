import LeanMlir
import LeanMlir.ReferenceNets

open ReferenceNets (bratsNetOf)
open SegMetrics (regionCounts regionDice)

/-! Per-volume scoring of a trained BraTS checkpoint — the literature's number.

    The trainer's `val Dice WT/TC/ET` is pooled over the 2,569 tumour-bearing
    val slices at stride 2: every pixel of every kept slice into one confusion
    matrix, one Dice out. Published BraTS Dice is something else — **one Dice
    per patient volume, over every slice of it including the tumour-free ones,
    then the mean over patients** — and the two differ in both directions. A
    volume with a small tumour counts as much as one with a large tumour, which
    the pooled number cannot do; and the ~75 slices per volume the pooled number
    never saw are exactly where a slice model over-paints, since there is nothing
    there to paint. So the pooled number is the easier measurement by
    construction, and until this file existed nothing in the repo could produce
    the other one.

    This reads `val_full.bin` (every axial slice of every validation volume, in
    volume order, from `preprocess_brats.py --val-full`) and `val_full.idx`
    (the per-volume slice counts), runs the eval forward over all of it, and
    accumulates one confusion matrix per volume. From those:

    * **per-volume Dice**, WT / TC / ET, under the BraTS convention for a region
      the ground truth lacks: a volume with no enhancing tumour scores 1 if the
      model predicts none and 0 if it predicts any. Mean, median and standard
      deviation over volumes, and the mean over the volumes where the region
      exists, since the convention's 0/1 jumps are a large share of the spread
      on ET (roughly a fifth of Task01's cases are lower-grade gliomas with no
      enhancing tumour at all).
    * **the tail**: per region the mean over the worst tenth of the patients and
      how many score under 0.7 and under 0.5 — the patients a clinician would
      have to redo, which the mean hides.
    * **the per-slice ET false-alarm rate**: of the slices with no enhancing
      tumour in the ground truth, how many get at least 1 and at least 10
      predicted ET pixels. The CSV carries the per-volume counts
      (`ET_clear_slices`, `ET_fa1`, `ET_fa10`) so `scripts/probes/brats_tail.py`
      can recompute both under a minimum-ET post-process.
    * **pooled Dice over every slice**, the trainer's number recomputed on the
      whole volumes — same instrument, tumour-free slices included.
    * **pooled Dice over the tumour-bearing slices** at stride 1, which should
      land within a few thousandths of the trainer's own log line for the same
      checkpoint. That agreement is the check that this file and the trainer
      score the same thing when they look at the same slices.

    No sliding window is needed: the 224 build is a centre crop whose 8-pixel
    border holds no brain voxel on this data (`DatasetKind.brats224`), so a
    224² prediction covers the volume.

    Usage:
        lake exe brats-eval [net=r34|r34noskip] [ctx=<k>] [arm=<tag>] [best] [out=<csv>] [params.bin bn_stats.bin]

    `net=`, `ctx=`, `arm=` and `best` mean what they mean for `brats-predict`
    (default: the from-scratch UNet on `data/brats`); `out=` writes one row per
    volume. The explicit params/bn_stats paths override the arm's artifacts. -/

private def numClasses : Nat := 4

private def fmt (x : Float) : String :=
  let s := toString x
  if s.length > 6 then (s.take 6).toString else s

private def meanOf (xs : Array Float) : Float :=
  if xs.isEmpty then 0.0 else (xs.foldl (· + ·) 0.0) / xs.size.toFloat

private def stdOf (xs : Array Float) : Float :=
  if xs.size < 2 then 0.0 else
    let m := meanOf xs
    Float.sqrt ((xs.foldl (fun a x => a + (x - m) * (x - m)) 0.0) / (xs.size.toFloat - 1.0))

private def medianOf (xs : Array Float) : Float :=
  if xs.isEmpty then 0.0 else
    let s := xs.qsort (· < ·)
    if s.size % 2 == 1 then s[s.size / 2]!
    else (s[s.size / 2 - 1]! + s[s.size / 2]!) / 2.0

def main (args : List String) : IO Unit := do
  let noSkip := args.any (· == "net=r34noskip")
  let useR34 := noSkip || args.any (· == "net=r34")
  let ctx : Nat :=
    ((args.filter (·.startsWith "ctx=")).head?.bind (fun a => (a.drop 4).toNat?)).getD 0
  if ctx > 0 && !useR34 then
    IO.eprintln "ctx= is the ResNet-34 UNet's 2.5D variant — pass net=r34 with it"
    IO.Process.exit 1
  let (spec, kind) := bratsNetOf useR34 noSkip ctx
  let channels := kind.bratsChannels
  let dataDir := kind.bratsDataDir
  let arm : String :=
    match (args.filter (·.startsWith "arm=")).head? with
    | some a => (a.drop 4).toString
    | none => ""
  let suffix := if args.any (· == "best") then "_best" else ""
  let outCsv : Option String :=
    ((args.filter (·.startsWith "out=")).head?).map (fun a => (a.drop 4).toString)
  let positional := args.filter (fun a =>
    !(a.startsWith "arm=" || a.startsWith "net=" || a.startsWith "ctx=" || a.startsWith "out="
      || a == "best"))
  let pfx := (spec.withBuildTag arm).buildPrefix
  let paramPath := positional[0]?.getD s!"{pfx}{suffix}_params.bin"
  let bnPath := positional[1]?.getD s!"{pfx}{suffix}_bn_stats.bin"
  let graph ← NetSpec.graphArtifact pfx "fwd_eval"
  for p in [paramPath, bnPath, graph, s!"{dataDir}/val_full.bin", s!"{dataDir}/val_full.idx"] do
    if !(← System.FilePath.pathExists p) then
      IO.eprintln s!"missing: {p}"
      IO.eprintln "  (a checkpoint from the trainer, and val_full.* from preprocess_brats.py --val-full)"
      IO.Process.exit 1
  IO.eprintln s!"  net: {spec.name}"
  IO.eprintln s!"  checkpoint: {paramPath}"
  let params ← IO.FS.readBinFile paramPath
  let bnStats ← IO.FS.readBinFile bnPath
  let evalParams := params.append bnStats

  -- The whole validation volumes, and where each one starts.
  let idx ← IO.FS.readBinFile s!"{dataDir}/val_full.idx"
  let nVol := readU32LE idx 0
  let counts : Array Nat := Id.run do
    let mut a := #[]
    for v in [:nVol] do a := a.push (readU32LE idx (4 + 4 * v))
    return a
  IO.eprintln s!"  loading {dataDir}/val_full.bin at {spec.imageH}², {channels} channels ..."
  let (img, mask, nSlices) ←
    F32.loadBrats s!"{dataDir}/val_full.bin" spec.imageH.toUSize channels.toUSize
  if counts.foldl (· + ·) 0 != nSlices then
    IO.eprintln s!"val_full.idx sums to {counts.foldl (· + ·) 0} slices, val_full.bin has {nSlices}"
    IO.Process.exit 1
  IO.eprintln s!"  {nVol} volumes, {nSlices} slices"
  let volOf : Array Nat := Id.run do   -- slice index → volume index
    let mut a := Array.mkEmpty nSlices
    for v in [:nVol] do
      for _ in [:counts[v]!] do a := a.push v
    return a

  let H := spec.imageH
  let W := spec.imageW
  let plane := H * W
  let NC := numClasses
  let imgPixels := channels * plane
  let evalBatch : Nat := 16          -- the batch the eval graph is rendered at
  let rowBytes := NC * plane * 4
  let outElems : USize := (NC * plane).toUSize
  let xShape := spec.xShape evalBatch
  let shapesBA := spec.evalShapesBA
  let sess ← LowererSession.create graph
  let nBatches := (nSlices + evalBatch - 1) / evalBatch
  IO.eprintln s!"  {nBatches} eval forwards at batch {evalBatch} ..."

  -- One confusion matrix per volume, plus two pooled ones.
  let mut conf : Array (Array Nat) := Array.replicate nVol (Array.replicate (NC * NC) 0)
  let mut pooledAll : Array Nat := Array.replicate (NC * NC) 0
  let mut pooledTumour : Array Nat := Array.replicate (NC * NC) 0
  let mut nTumourSlices := 0
  -- Per volume: slices with no ground-truth ET, and those of them with ≥ 1 / ≥ 10 predicted ET pixels.
  let mut etClear : Array Nat := Array.replicate nVol 0
  let mut etFa1 : Array Nat := Array.replicate nVol 0
  let mut etFa10 : Array Nat := Array.replicate nVol 0
  let t0 ← IO.monoMsNow
  for bi in [:nBatches] do
    let start := bi * evalBatch
    let real := min evalBatch (nSlices - start)
    let xba := F32.sliceImagesPad img start evalBatch imgPixels nSlices
    let logits ← LowererSession.forwardF32 sess spec.evalFnName
                   evalParams shapesBA xba xShape evalBatch.toUSize outElems
    for b in [:real] do
      let s := start + b
      let row := logits.extract (b * rowBytes) ((b + 1) * rowBytes)
      let m := F32.sliceLabels mask s 1 plane
      let cb ← F32.segConfusion row m 1 NC.toUSize H.toUSize W.toUSize
      let v := volOf[s]!
      let mut cv := conf[v]!
      let mut tumourPx := 0
      for j in [:NC * NC] do
        let c := readU64LE cb (8 * j)
        cv := cv.set! j (cv[j]! + c)
        pooledAll := pooledAll.set! j (pooledAll[j]! + c)
        if j / NC != 0 then tumourPx := tumourPx + c
      conf := conf.set! v cv
      let mut gtEt := 0
      let mut prEt := 0
      for j in [:NC] do
        gtEt := gtEt + readU64LE cb (8 * (3 * NC + j))
        prEt := prEt + readU64LE cb (8 * (j * NC + 3))
      if gtEt == 0 then
        etClear := etClear.set! v (etClear[v]! + 1)
        if prEt ≥ 1 then etFa1 := etFa1.set! v (etFa1[v]! + 1)
        if prEt ≥ 10 then etFa10 := etFa10.set! v (etFa10[v]! + 1)
      if tumourPx > 0 then
        nTumourSlices := nTumourSlices + 1
        for j in [:NC * NC] do
          pooledTumour := pooledTumour.set! j (pooledTumour[j]! + readU64LE cb (8 * j))
    if bi % 100 == 0 then
      IO.eprintln s!"    batch {bi}/{nBatches}"
  let t1 ← IO.monoMsNow
  IO.eprintln s!"  scored in {(t1 - t0) / 1000} s"

  -- Pooled: the trainer's instrument, on all slices and on the tumour-bearing ones.
  let pooledLine := fun (label : String) (c : Array Nat) (n : Nat) => Id.run do
    let mut parts : List String := []
    for (name, cls) in kind.segRegions do
      let (i, g, p) := regionCounts c NC cls
      parts := parts ++ [s!"{name} {fmt (regionDice i g p)}"]
    let mut ious : Float := 0.0
    for k in [:NC] do
      let tp := c[k * NC + k]!
      let mut row := 0
      let mut col := 0
      for j in [:NC] do
        row := row + c[k * NC + j]!
        col := col + c[j * NC + k]!
      let uni := row + col - tp
      ious := ious + (if uni == 0 then 0.0 else tp.toFloat / uni.toFloat)
    s!"  pooled Dice, {label} ({n} slices): {String.intercalate "  " parts}  mIoU {fmt (ious / NC.toFloat)}"
  IO.println (pooledLine "tumour-bearing slices" pooledTumour nTumourSlices)
  IO.println (pooledLine "every slice" pooledAll nSlices)

  -- Per volume: the literature's protocol.
  IO.println s!"  per-volume Dice over {nVol} patients (mean ± sd, median; mean over volumes with the region; volumes without it):"
  let mut csv := "volume,slices"
  for (name, _) in kind.segRegions do
    csv := csv ++ s!",{name}_inter,{name}_gt,{name}_pred,{name}_dice"
  csv := csv ++ ",ET_clear_slices,ET_fa1,ET_fa10\n"
  let mut rows : Array String := Array.replicate nVol ""
  for v in [:nVol] do
    rows := rows.set! v s!"{v},{counts[v]!}"
  let mut tails : List (String × Array Float) := []
  for (name, cls) in kind.segRegions do
    let mut dices : Array Float := #[]
    let mut present : Array Float := #[]
    let mut absent := 0
    for v in [:nVol] do
      let (i, g, p) := regionCounts conf[v]! NC cls
      let d := regionDice i g p
      dices := dices.push d
      if g > 0 then present := present.push d else absent := absent + 1
      rows := rows.set! v (rows[v]! ++ s!",{i},{g},{p},{d}")
    IO.println s!"    {name}: {fmt (meanOf dices)} ± {fmt (stdOf dices)}  median {fmt (medianOf dices)}   present-only {fmt (meanOf present)} (n={present.size})   absent in {absent}"
    tails := tails ++ [(name, dices)]
  IO.println s!"  tail over {nVol} patients (worst-10% mean · n<0.7 · n<0.5):"
  for (name, dices) in tails do
    let sorted := dices.qsort (· < ·)
    let k := max 1 (nVol / 10)
    IO.println s!"    {name}: {fmt (meanOf (sorted.extract 0 k))} · {(dices.filter (· < 0.7)).size} · {(dices.filter (· < 0.5)).size}"
  let clear := etClear.foldl (· + ·) 0
  let fa1 := etFa1.foldl (· + ·) 0
  let fa10 := etFa10.foldl (· + ·) 0
  let rate := fun (n : Nat) => if clear == 0 then 0.0 else n.toFloat / clear.toFloat
  IO.println s!"  ET false alarms over {clear} slices with no ET: ≥1 px on {fa1} ({fmt (rate fa1)}), ≥10 px on {fa10} ({fmt (rate fa10)})"
  for v in [:nVol] do
    csv := csv ++ rows[v]! ++ s!",{etClear[v]!},{etFa1[v]!},{etFa10[v]!}\n"
  if let some path := outCsv then
    IO.FS.writeFile path csv
    IO.eprintln s!"  wrote {path}"
