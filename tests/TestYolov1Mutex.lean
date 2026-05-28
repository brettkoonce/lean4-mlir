import LeanMlir

/-! T7 from planning/yolo_demo_v2.md Phase 1 — useYolov1 mutex checks.

    Verifies that compileVmfbs throws `IO.userError` for every forbidden
    combination of `useYolov1` with other loss-path flags. Also verifies
    the catch-all throw that documents Phase-1-smoke-test-only scope.

    Decision D8: inline mutex checks (no helper refactor yet). -/

def tinyYoloSpec : NetSpec where
  name := "tiny-yolo-mutex-test"
  imageH := 28
  imageW := 28
  layers := [
    .flatten,
    .dense 784 1470 .identity   -- 1470 = 7*7*30 so the shape is yolov1-compatible
  ]

def baseConfig : TrainConfig := {
  learningRate := 0.001
  batchSize    := 1
  epochs       := 1
  useAdam      := true
  useYolov1    := true
}

/-- Run `act`, expect it to throw `IO.userError`. Returns `none` on success
    (i.e. it threw), `some msg` on failure (it didn't throw or threw a
    non-userError). -/
private def expectThrow (label : String) (act : IO Unit) : IO (Option String) := do
  try
    act
    return some s!"FAIL [{label}]: expected throw, none happened"
  catch e =>
    let msg := toString e
    if !msg.contains "useYolov1" then
      return some s!"FAIL [{label}]: threw, but message didn't mention 'useYolov1': {msg}"
    return none

def main : IO Unit := do
  let mut failures : Array String := #[]

  -- C1: useYolov1 + useMixup → throw
  let c1 := { baseConfig with useMixup := true }
  match (← expectThrow "useYolov1 + useMixup" (do let _ ← tinyYoloSpec.compileVmfbs c1; pure ())) with
  | some f => failures := failures.push f
  | none => IO.println "OK [C1]: useYolov1 + useMixup → throws"

  -- C2: useYolov1 + useCutmix → throw
  let c2 := { baseConfig with useCutmix := true }
  match (← expectThrow "useYolov1 + useCutmix" (do let _ ← tinyYoloSpec.compileVmfbs c2; pure ())) with
  | some f => failures := failures.push f
  | none => IO.println "OK [C2]: useYolov1 + useCutmix → throws"

  -- C3: useYolov1 + useKnnMixup → throw
  let c3 := { baseConfig with useKnnMixup := true }
  match (← expectThrow "useYolov1 + useKnnMixup" (do let _ ← tinyYoloSpec.compileVmfbs c3; pure ())) with
  | some f => failures := failures.push f
  | none => IO.println "OK [C3]: useYolov1 + useKnnMixup → throws"

  -- C4: useYolov1 + useFocal → throw
  let c4 := { baseConfig with useFocal := true }
  match (← expectThrow "useYolov1 + useFocal" (do let _ ← tinyYoloSpec.compileVmfbs c4; pure ())) with
  | some f => failures := failures.push f
  | none => IO.println "OK [C4]: useYolov1 + useFocal → throws"

  -- C5: useYolov1 + labelSmoothing != 0 → throw
  let c5 := { baseConfig with labelSmoothing := 0.1 }
  match (← expectThrow "useYolov1 + labelSmoothing" (do let _ ← tinyYoloSpec.compileVmfbs c5; pure ())) with
  | some f => failures := failures.push f
  | none => IO.println "OK [C5]: useYolov1 + labelSmoothing → throws"

  -- C6: useYolov1 alone (no other forbidden combo) — still throws because
  -- Phase 1 doesn't wire YOLOv1 through compileVmfbs (smoke-test-only).
  match (← expectThrow "useYolov1 alone" (do let _ ← tinyYoloSpec.compileVmfbs baseConfig; pure ())) with
  | some f => failures := failures.push f
  | none => IO.println "OK [C6]: useYolov1 alone (Phase 1 not-yet-integrated catch-all) → throws"

  if failures.isEmpty then
    IO.println "T7 PASS: all 6 mutex checks throw as expected"
  else
    for f in failures do IO.eprintln f
    IO.eprintln s!"T7 FAIL: {failures.size}/6 checks failed"
    IO.Process.exit 1
