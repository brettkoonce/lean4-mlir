import LeanMlir

/-! `.yolov1Masked` mutex checks.

    Verifies that compileVmfbs throws `IO.userError` for every forbidden
    combination of the YOLOv1 loss with other loss-path flags, that the loss compiles on its
    own, and that a detection run with no explicit `lossKind` resolves to it
    (`TrainConfig.lossKindFor`), so the same checks apply. -/

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
  optimizer    := .adam
  lossKind     := some .yolov1Masked
}

/-- Run `act`, expect it to throw `IO.userError` mentioning yolov1.
    Returns `none` on success (it threw the right error), `some msg` on
    failure. The message must name `yolov1Masked`, as evidence the throw is from the YOLOv1
    mutex path. -/
private def expectThrow (label : String) (act : IO Unit) : IO (Option String) := do
  try
    act
    return some s!"FAIL [{label}]: expected throw, none happened"
  catch e =>
    let msg := toString e
    if !msg.contains "yolov1Masked" then
      return some s!"FAIL [{label}]: threw, but message didn't mention 'yolov1Masked': {msg}"
    return none

def main : IO Unit := do
  let mut failures : Array String := #[]

  -- C1: yolov1Masked + useMixup → throw
  let c1 := { baseConfig with useMixup := true }
  match (← expectThrow "yolov1Masked + useMixup" (do let _ ← tinyYoloSpec.compileVmfbs c1; pure ())) with
  | some f => failures := failures.push f
  | none => IO.println "OK [C1]: yolov1Masked + useMixup → throws"

  -- C2: yolov1Masked + useCutmix → throw
  let c2 := { baseConfig with useCutmix := true }
  match (← expectThrow "yolov1Masked + useCutmix" (do let _ ← tinyYoloSpec.compileVmfbs c2; pure ())) with
  | some f => failures := failures.push f
  | none => IO.println "OK [C2]: yolov1Masked + useCutmix → throws"

  -- C3: yolov1Masked + useKnnMixup → throw
  let c3 := { baseConfig with useKnnMixup := true }
  match (← expectThrow "yolov1Masked + useKnnMixup" (do let _ ← tinyYoloSpec.compileVmfbs c3; pure ())) with
  | some f => failures := failures.push f
  | none => IO.println "OK [C3]: yolov1Masked + useKnnMixup → throws"

  -- C4: yolov1Masked + useFocal → COMPILES. Focal selects the sigmoid
  -- focal-BCE objectness path, so this combo is
  -- valid and the train step should compile cleanly.
  let c4 := { baseConfig with useFocal := true, focalGamma := 2.0 }
  let c4_ok ← try
    let _ ← tinyYoloSpec.compileVmfbs c4
    pure true
  catch e =>
    IO.eprintln s!"FAIL [C4]: yolov1Masked + useFocal should compile (focal objectness), but threw: {e}"
    pure false
  if c4_ok then
    IO.println "OK [C4]: yolov1Masked + useFocal → focal-objectness train step compiles"
  else
    failures := failures.push "C4 failed"

  -- C5: yolov1Masked + labelSmoothing != 0 → throw
  let c5 := { baseConfig with labelSmoothing := 0.1 }
  match (← expectThrow "yolov1Masked + labelSmoothing" (do let _ ← tinyYoloSpec.compileVmfbs c5; pure ())) with
  | some f => failures := failures.push f
  | none => IO.println "OK [C5]: yolov1Masked + labelSmoothing → throws"

  -- C6: yolov1Masked alone (no other forbidden combo) — compileVmfbs
  -- integrates YOLOv1 and should return a vmfb path without throwing.
  let c6_ok ← try
    let _ ← tinyYoloSpec.compileVmfbs baseConfig
    pure true
  catch e =>
    IO.eprintln s!"FAIL [C6]: yolov1Masked alone should compile, but threw: {e}"
    pure false
  if c6_ok then
    IO.println "OK [C6]: yolov1Masked alone → compileVmfbs succeeds"
  else
    failures := failures.push "C6 failed"

  -- C7: detection with no explicit lossKind resolves to yolov1Masked, so its mutex applies.
  let c7 : TrainConfig := { baseConfig with lossKind := none, useMixup := true }
  match (← expectThrow "detection (derived) + useMixup"
      (do let _ ← tinyYoloSpec.compileVmfbs c7 .detection; pure ())) with
  | some f => failures := failures.push f
  | none => IO.println "OK [C7]: detection derives yolov1Masked + useMixup → throws"

  if failures.isEmpty then
    IO.println "T7 PASS: 4 mutex throws + 2 integration successes (incl. focal objectness)"
  else
    for f in failures do IO.eprintln f
    IO.eprintln s!"T7 FAIL: {failures.size}/6 checks failed"
    IO.Process.exit 1
