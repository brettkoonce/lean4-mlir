import LeanMlir.VerifiedNets

/-! # `@resnet34_fwd` render tie — do two renders compute the same function?

Migration guard for `planning/xla_pjrt_handoff.md` §2a. `verified_mlir/resnet34_fwd.mlir` used to be
written by a hand-written string emitter in `tests/TestResnet34Fwd.lean`, independent of the proven
graph; it is now `pretty(provenGraph)` out of `LeanMlir/Proofs/Codegen/ResNet34Render.lean`, sharing
its forward chain with the train step. The two emitters produce *textually* different MLIR by
construction, so the only meaningful check is numeric: feed both the same parameters and the same
input, and compare logits.

Both paths must be the same `@resnet34_fwd` signature (147 inputs — `%x` then the 146 params in
`net.paramShapes` order — returning `[BS,10]`); the signature itself is checked by arity/type diff,
not here.

    lake build resnet34-fwd-tie
    .lake/build/bin/resnet34-fwd-tie <refRender.mlir> <newRender.mlir>

Defaults to comparing `verified_mlir/resnet34_fwd.mlir` against itself (a self-tie smoke test).
Exits non-zero if the renders disagree, or if the comparison is **degenerate** — all-zero or
non-finite logits agree trivially and prove nothing.
-/

/-- The driver's own init (`VerifiedTrain.mkParam`, which is private): He(fan-in) weights,
    γ = 1, β/bias = 0. Using the real init matters — a constant-splat parameter set makes BN see
    zero variance, and two wrong renders would agree on the resulting garbage. -/
private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0
  | 2 => F32.const n.toUSize 0.0
  | _ =>
    let fanIn := if dims.size == 4 then dims[1]! * dims[2]! * dims[3]! else dims[0]!
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))

def main (args : List String) : IO Unit := do
  let (pathA, pathB) := match args with
    | a :: b :: _ => (a, b)
    | [a]         => (a, "verified_mlir/resnet34_fwd.mlir")
    | []          => ("verified_mlir/resnet34_fwd.mlir", "verified_mlir/resnet34_fwd.mlir")
  let net := resnet34Verified.toNet
  let bs  := 32                        -- the baked batch of the committed render
  IO.println s!"resnet34_fwd tie: A={pathA}  B={pathB}"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), bs {bs}, backend {← IreeSession.backendName}"

  -- ── one deterministic (θ, x) both renders see ──
  let mut parts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    parts := parts.push (← mkParam sd dims kind)
    sd := sd + 1
  let params := F32.concat parts
  let x ← F32.heInit 987654 (bs * net.d0).toUSize 1.0
  let shapes := net.shapesBA
  let xsh    := net.xShape bs

  let runOne (path tag : String) : IO ByteArray := do
    let sess ← mkSession path s!".lake/build/r34_fwd_tie_{tag}.vmfb"
    IreeSession.forwardF32 sess "m.resnet34_fwd" params shapes x xsh
      bs.toUSize net.nClasses.toUSize
  let la ← runOne pathA "a"
  let lb ← runOne pathB "b"

  -- ── compare, and refuse a degenerate agreement ──
  let n := bs * net.nClasses
  let mut maxAbs    : Float := 0.0
  let mut maxRel    : Float := 0.0
  let mut maxMag    : Float := 0.0
  let mut nonFinite : Nat   := 0
  for i in [0:n] do
    let a := F32.read la i.toUSize
    let b := F32.read lb i.toUSize
    if !a.isFinite || !b.isFinite then nonFinite := nonFinite + 1
    let d := (a - b).abs
    let m := max a.abs b.abs
    if m > maxMag then maxMag := m
    if d > maxAbs then maxAbs := d
    let r := if m > 1e-30 then d / m else 0.0
    if r > maxRel then maxRel := r

  IO.println s!"  logits |max| = {maxMag}   max abs diff = {maxAbs}   max rel diff = {maxRel}"
  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite}/{n} non-finite logits — the tie proves nothing"
    IO.Process.exit 1
  if maxMag < 1e-6 then
    IO.eprintln s!"DEGENERATE: logits are all ~0 (|max| = {maxMag}) — the tie proves nothing"
    IO.Process.exit 1
  if maxRel > 1e-4 then
    IO.eprintln s!"TIE FAILED: max rel diff {maxRel} > 1e-4 — the renders compute different functions"
    IO.Process.exit 1
  IO.println s!"✓ renders TIE (max rel {maxRel} ≤ 1e-4) on non-degenerate logits"
