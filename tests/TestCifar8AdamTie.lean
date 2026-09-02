import LeanMlir.VerifiedNets

/-! # `@cifar8_adam_train_step` render tie — hand-written vs `pretty(provenGraph)`

`planning/xla_pjrt_handoff.md` §2a-ter. The artifact used to come from the hand-written emitter in
`tests/TestCifar8AdamTrain.lean` (whose optimizer was `ViTRender.emitAdamV`); it now renders from
`LeanMlir/Proofs/Codegen/CnnRender.lean` as `pretty(provenGraph)`, with the fused SGD tail replaced
by un-fused param gradients feeding the proven AdamW ops.

The signature is identical (71 inputs / 69 outputs, same names and types in order), so both take the
same packed `[θ|m|v|lr|bc1|bc2]` buffer. One step, then compare every returned float.

    lake build cifar8-adam-tie
    .lake/build/bin/cifar8-adam-tie <refRender.mlir> <newRender.mlir>

Exits non-zero if the renders disagree, or if the comparison is degenerate.
-/

def main (args : List String) : IO Unit := do
  let dflt := "verified_mlir/cifar8_adam_train_step.mlir"
  let (pathA, pathB) := match args with
    | a :: b :: _ => (a, b)
    | [a]         => (a, dflt)
    | []          => (dflt, dflt)
  let net := cifar8Verified.toNet
  let bs  := 128                       -- the baked batch of the committed render
  IO.println s!"@cifar8_adam_train_step tie: A={pathA}  B={pathB}"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), bs {bs}, backend {← LowererSession.backendName}"

  -- ── θ (driver init), m (centred noise), v (POSITIVE — it is under a sqrt) ──
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  let m ← F32.heInit 4242 net.nParams.toUSize 0.02
  -- v ≈ 0.05 ± 0.01, strictly positive: a zero v would make √v̂ + ε ≈ ε for every coordinate
  -- and hide any error in the second-moment path.
  let v ← F32.scaleShift (← F32.heInit 8484 net.nParams.toUSize 0.01) 1.0 0.05
  -- scalar tail: lr, bc1, bc2 (bias-correction denominators, host-computed per step)
  let tail ← F32.const 3 0.0
  let tail ← F32.write3 tail 0 0.001 0.19 0.002    -- lr, 1−β₁ᵗ, 1−β₂ᵗ at a mid-training step
  let pbuf := F32.concat #[θ, m, v, tail]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]])
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat (i % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0

  let runOne (path tag : String) : IO ByteArray := do
    let sess ← mkSession path
    LowererSession.mlpTrainStepV sess "m.cifar8_adam_train_step" x pbuf shapes y
      bs.toUSize net.d0.toUSize net.nClasses.toUSize
  let oa ← runOne pathA "a"
  let ob ← runOne pathB "b"

  if oa.size != ob.size then
    IO.eprintln s!"SIZE MISMATCH: {oa.size} vs {ob.size} bytes"; IO.Process.exit 1
  let n := oa.size / 4
  let mut maxAbs : Float := 0.0
  let mut maxRel : Float := 0.0
  let mut maxMag : Float := 0.0
  let mut nonFinite : Nat := 0
  let mut moved : Nat := 0
  for i in [0:n] do
    let a := F32.read oa i.toUSize
    let b := F32.read ob i.toUSize
    if !a.isFinite || !b.isFinite then nonFinite := nonFinite + 1
    let d := (a - b).abs
    let mg := max a.abs b.abs
    if mg > maxMag then maxMag := mg
    if mg > 1e-12 then moved := moved + 1
    if d > maxAbs then maxAbs := d
    let r := if mg > 1e-30 then d / mg else 0.0
    if r > maxRel then maxRel := r

  IO.println s!"  {n} returned floats; {moved} non-zero; |max| = {maxMag}"
  IO.println s!"  max abs diff = {maxAbs}   max rel diff = {maxRel}"
  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite}/{n} non-finite outputs"; IO.Process.exit 1
  if moved * 10 < n then
    IO.eprintln s!"DEGENERATE: only {moved}/{n} outputs are non-zero — the tie proves little"
    IO.Process.exit 1
  if maxRel > 1e-4 then
    IO.eprintln s!"TIE FAILED: max rel diff {maxRel} > 1e-4"; IO.Process.exit 1
  IO.println s!"✓ renders TIE (max rel {maxRel} ≤ 1e-4) on {moved} non-degenerate outputs"
