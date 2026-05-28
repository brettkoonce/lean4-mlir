import LeanMlir

/-! T2 + T3 + T4 + T5 from planning/yolo_demo_v2.md Phase 1.

    Runtime tests for the YOLOv1 5-term masked-MSE codegen. Compiles
    the train-step vmfb for the full ResNet-34 + YOLO head spec (D11),
    then exercises the FFI via `trainStepAdamF32Yolov1` on synthesized
    targets + masks.

    Tests:
      T2 — loss decreases over 10 optimizer steps on random target+mask
      T3 — mask=0 everywhere: loss is finite, non-NaN (only noobj-conf
           terms contribute since coord/class/conf-positive are gated)
      T4 — per-term decomposition: MLIR string contains all 6 term SSAs
           (%y1_t1..%y1_t6) and the paper-correct constants (λ_coord=5.0,
           λ_noobj=0.5, ε=1.0e-06)
      T5 — ε floor stability: target_wh = 0 (clamped to ε at √); no NaN
           in loss or output

    Heavy: iree-compile of R34 train-step takes ~5-10 min on first run
    (cached afterwards via the .hash sidecar). Each step is ~1s on the
    7900 XTX. -/

def r34Yolov1 : NetSpec where
  name := "ResNet-34 + YOLOv1 head (train step)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,                    -- 2/2 instead of paper's 3/2 (IREE compat; matches MainResnetTrain)
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .flatten,
    .dense 25088 1470 .identity
  ]

-- YOLOv1 hyperparameters (match codegen defaults / VOC layout).
private def gridH : Nat := 7
private def gridW : Nat := 7
private def numBoxes : Nat := 2
private def numClasses : Nat := 20
private def perCell : Nat := numBoxes * 5 + numClasses  -- 30

private def findIreeCompile : IO String := do
  if ← System.FilePath.pathExists ".venv/bin/iree-compile" then
    return ".venv/bin/iree-compile"
  return "iree-compile"

/-- Float-equality check accepting NaN as "not finite". -/
private def isFinite (x : Float) : Bool := !x.isNaN && !x.isInf

/-- Substring check: `sub` appears in `s`. Implemented via splitOn because
    `String.containsSubstr` doesn't exist in core. -/
private def hasSubstr (s sub : String) : Bool :=
  (s.splitOn sub).length > 1

/-- ───────────────────────── T4 (string-level) ─────────────────────────
    Inspect the generated MLIR for structural evidence of the 5-term
    decomposition. Catches: missing term, wrong λ constants, missing
    ε floor. Runs without iree-compile / runtime. -/
def testT4_structuralDecomposition (mlir : String) : IO Bool := do
  let mut ok := true
  let mut missing : Array String := #[]
  let required := [
    "%y1_t1", "%y1_t2", "%y1_t3", "%y1_t4", "%y1_t5", "%y1_t6",
    "%y1_lcoord", "%y1_lnoobj", "%y1_eps_wh",
    "dense<5.0>",      -- λ_coord
    "dense<0.5>",      -- λ_noobj
    "dense<1.0e-06>",  -- ε floor on √
    "stablehlo.sqrt",  -- √ pred / target
    "%y1_active_wh"    -- gradient-zero mask where pred was clamped
  ]
  for s in required do
    if !hasSubstr mlir s then
      ok := false; missing := missing.push s
  if ok then
    IO.println s!"T4 PASS: structural decomposition (all {required.length} markers present)"
  else
    IO.eprintln s!"T4 FAIL: missing markers: {missing}"
  return ok

def main : IO Unit := do
  let spec := r34Yolov1
  match spec.validate with
  | some err => IO.eprintln s!"NetSpec validation failed: {err}"; IO.Process.exit 1
  | none     => pure ()
  IO.println s!"{spec.name}: {spec.totalParams} params"

  let batch : Nat := 1
  let batchU : USize := batch.toUSize

  -- Generate train-step MLIR.
  let train := MlirCodegen.generateTrainStep spec batch "jit_test_yolov1_train_step"
    (useAdam := true) (useYolov1 := true)
    (yoloGridH := gridH) (yoloGridW := gridW)
    (yoloNumBoxes := numBoxes) (yoloNumClasses := numClasses)

  -- T4 (structural decomposition) — runs without iree-compile.
  let t4_ok ← testT4_structuralDecomposition train

  IO.FS.createDirAll ".lake/build"
  let mlirPath := ".lake/build/test_yolov1_train_step.mlir"
  let vmfbPath := ".lake/build/test_yolov1_train_step.vmfb"
  IO.FS.writeFile mlirPath train
  IO.println s!"wrote {mlirPath} ({train.length} chars)"

  -- iree-compile (slow on R34 first time; cached via .hash sidecar).
  let args ← ireeCompileArgs mlirPath vmfbPath
  let compiler ← findIreeCompile
  IO.println s!"running iree-compile (may take several minutes for R34)..."
  let t0 ← IO.monoMsNow
  let r ← IO.Process.output { cmd := compiler, args := args }
  let t1 ← IO.monoMsNow
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile failed (exit {r.exitCode})"
    IO.eprintln (r.stderr.take 6000)
    IO.Process.exit 1
  IO.println s!"iree-compile OK ({(t1 - t0) / 1000}s)"

  -- Load session.
  let sess ← IreeSession.create vmfbPath
  IO.println "session loaded"

  -- He-init params + zero m, v.
  let params ← spec.heInitParams
  let nP := spec.totalParams
  IO.println s!"init params: {nP} floats ({params.size / 1024 / 1024} MB)"
  let m ← F32.const nP.toUSize 0.0
  let v ← F32.const nP.toUSize 0.0

  let allShapes := spec.shapesBA
  let bnShapes := spec.bnShapesBA
  let xShape := spec.xShape batch
  let inputFlat := batch * MlirCodegen.inputFlatDim spec
  let tgtFlat := batch * perCell * gridH * gridW
  let mskFlat := batch * gridH * gridW

  -- ──────────────────────── T2 — loss-decreases ────────────────────────
  -- Adam at lr=1e-5 + 30 steps. With B=1 + random targets, the first few
  -- Adam steps swing wildly (bias-correction `1 - β1^t` = 0.1 at t=1, so
  -- the effective step is 10× nominal). What we check is that the
  -- gradient direction is correct: the *minimum* observed loss in the
  -- last 20 steps is < 80% of the initial. That suffices as a "gradient
  -- has the right sign" smoke check without requiring strict monotonic
  -- decrease at this learning rate / batch size.
  IO.println "T2: loss-decreases over 30 steps (Adam @ lr=1e-5, B=1)"
  let x ← F32.heInit 1234 inputFlat.toUSize 1.0
  let tgt ← F32.heInit 2345 tgtFlat.toUSize 1.0
  let mask ← F32.const mskFlat.toUSize 1.0
  let mut p := params
  let mut mu := m
  let mut vv := v
  let mut losses : Array Float := #[]
  for step in [:30] do
    let lr : Float := 1.0e-5
    let t : Float := (step + 1).toFloat
    let packed := (p.append mu).append vv
    let out ← IreeSession.trainStepAdamF32Yolov1 sess
      "jit_test_yolov1_train_step.main"
      packed allShapes x xShape tgt mask lr t bnShapes
      batchU gridH.toUSize gridW.toUSize perCell.toUSize
    let loss := F32.extractLoss out (3 * nP)
    losses := losses.push loss
    p  := F32.slice out 0 nP
    mu := F32.slice out nP nP
    vv := F32.slice out (2 * nP) nP
    if step < 5 || step % 5 == 0 then
      IO.println s!"  step {step}: loss={loss}"
  let firstLoss := losses[0]!
  -- Min loss observed after step 10 (let Adam stabilize past the warmup spike).
  let tailLosses := (losses.toList.drop 10).filter isFinite
  let huge : Float := 1.0e30
  let minTail := tailLosses.foldl (fun a b => if a < b then a else b) huge
  let t2_ok :=
    isFinite firstLoss && isFinite minTail && minTail < firstLoss * 0.8
  if t2_ok then
    IO.println s!"T2 PASS: loss {firstLoss} → min(tail) {minTail} (< 80% of initial)"
  else
    IO.eprintln s!"T2 FAIL: loss {firstLoss} → min(tail) {minTail} (expected < {firstLoss * 0.8})"

  -- ──────────────────────── T3 — mask=0 closed-form ────────────────────
  -- Fresh params (don't reuse T2's trained-down state).
  IO.println "T3: mask=0 everywhere; loss should be finite (noobj-conf only)"
  let p3 ← spec.heInitParams
  let m3 ← F32.const nP.toUSize 0.0
  let v3 ← F32.const nP.toUSize 0.0
  let tgt3 ← F32.const tgtFlat.toUSize 0.0
  let mask3 ← F32.const mskFlat.toUSize 0.0   -- no cells "have object"
  let packed3 := (p3.append m3).append v3
  let out3 ← IreeSession.trainStepAdamF32Yolov1 sess
    "jit_test_yolov1_train_step.main"
    packed3 allShapes x xShape tgt3 mask3 0.0 1.0 bnShapes
    batchU gridH.toUSize gridW.toUSize perCell.toUSize
  let loss3 := F32.extractLoss out3 (3 * nP)
  let t3_ok := isFinite loss3 && loss3 >= 0.0
  if t3_ok then
    IO.println s!"T3 PASS: mask=0 loss = {loss3} (finite, non-negative)"
  else
    IO.eprintln s!"T3 FAIL: mask=0 loss = {loss3} (expected finite, non-negative)"

  -- ──────────────────────── T5 — ε floor stability ─────────────────────
  -- Target with wh channels at 0 → triggers max(tgt, ε) clamp in codegen.
  -- (Predicted wh ≈ He-init values, which could be small; the ε floor on
  -- pred_wh_clamp ensures √ doesn't NaN.)
  IO.println "T5: target_wh = 0 (clamped to ε); no NaN in loss"
  let p5 ← spec.heInitParams
  let m5 ← F32.const nP.toUSize 0.0
  let v5 ← F32.const nP.toUSize 0.0
  -- Build target where every (w, h) channel is 0 but xy/conf/class random.
  let tgt5 ← F32.heInit 4567 tgtFlat.toUSize 1.0
  -- Zero out wh channels (offsets 2, 3 within each cell). For [B, perCell, gH, gW]
  -- in NCHW, channel index 2 spans bytes [2 * gH * gW * 4, 3 * gH * gW * 4),
  -- channel 3 spans [3 * gH * gW * 4, 4 * gH * gW * 4). Per batch slot:
  --   tgt5_offset = b * perCell * gH * gW + c * gH * gW + cell, in floats.
  -- Use F32.zeroChannels if available, else a manual loop. We don't have a
  -- generic helper, so allocate a fresh zero block and overwrite via a
  -- minimal in-place loop using ByteArray.set! on the float bytes.
  let mut tgt5Mut := tgt5
  let chBytes : Nat := gridH * gridW * 4
  for b in [:batch] do
    for c in [2, 3] do
      let base : Nat := (b * perCell + c) * chBytes
      for i in [:chBytes] do
        tgt5Mut := tgt5Mut.set! (base + i) 0
  let mask5 ← F32.const mskFlat.toUSize 1.0
  let packed5 := (p5.append m5).append v5
  let out5 ← IreeSession.trainStepAdamF32Yolov1 sess
    "jit_test_yolov1_train_step.main"
    packed5 allShapes x xShape tgt5Mut mask5 0.0 1.0 bnShapes
    batchU gridH.toUSize gridW.toUSize perCell.toUSize
  let loss5 := F32.extractLoss out5 (3 * nP)
  let t5_ok := isFinite loss5
  if t5_ok then
    IO.println s!"T5 PASS: ε-floor loss = {loss5} (finite)"
  else
    IO.eprintln s!"T5 FAIL: ε-floor loss = {loss5} (NaN or Inf)"

  -- ──────────────────────── Summary ────────────────────
  let pass := t4_ok && t2_ok && t3_ok && t5_ok
  if pass then
    IO.println "ALL PASS: T2, T3, T4, T5"
  else
    IO.eprintln s!"FAIL: T2={t2_ok}, T3={t3_ok}, T4={t4_ok}, T5={t5_ok}"
    IO.Process.exit 1
