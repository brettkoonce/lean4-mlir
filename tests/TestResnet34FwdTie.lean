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
    .lake/build/bin/resnet34-fwd-tie [--eval] <refRender.mlir> <newRender.mlir>

`--eval` ties `@resnet34_fwd_eval` instead: 219 inputs (the 146 params plus 72 running-stat
inputs, μ/var interleaved per BN layer in `bnChannels` order) and running-stats BN.

Defaults to comparing the artifact against itself (a self-tie smoke test). Exits non-zero if the
renders disagree, or if the comparison is **degenerate** — all-zero or non-finite logits agree
trivially and prove nothing.
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
  let isEval := args.contains "--eval"
  let paths  := args.filter (· != "--eval")
  let slug   := if isEval then "resnet34_fwd_eval" else "resnet34_fwd"
  let dflt   := s!"verified_mlir/{slug}.mlir"
  let (pathA, pathB) := match paths with
    | a :: b :: _ => (a, b)
    | [a]         => (a, dflt)
    | []          => (dflt, dflt)
  let net := resnet34Verified.toNet
  let bs  := 32                        -- the baked batch of the committed render
  IO.println s!"@{slug} tie: A={pathA}  B={pathB}"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), bs {bs}, backend {← IreeSession.backendName}"

  -- ── one deterministic (θ, x) both renders see ──
  let mut parts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    parts := parts.push (← mkParam sd dims kind)
    sd := sd + 1
  let mut params := F32.concat parts
  -- eval BN consumes frozen per-channel stats, appended after the params in `bnChannels` order,
  -- μ then var per layer. μ is centred noise; var must be POSITIVE (rsqrt) and must vary — a
  -- constant var would let a broadcast bug pass unnoticed.
  let mut shapes := net.shapesBA
  if isEval then
    let mut stats : Array ByteArray := #[]
    let mut ss := 5000
    for c in net.bnChannels do
      stats := stats.push (← F32.heInit ss.toUSize c.toUSize 0.30)                        -- μ
      stats := stats.push (← F32.scaleShift (← F32.heInit (ss+1).toUSize c.toUSize 0.20)
                               1.0 1.0)                                                   -- var ≈ 1 ± 0.2
      ss := ss + 2
    params := F32.concat (#[params] ++ stats)
    shapes := packShapes (net.paramShapes ++
                net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[])
    IO.println s!"  + {net.bnChannels.size} BN layers → {net.bnChannels.foldl (· + 2 * ·) 0} running-stat floats"
  let x ← F32.heInit 987654 (bs * net.d0).toUSize 1.0
  let xsh := net.xShape bs

  let runOne (path tag : String) : IO ByteArray := do
    let sess ← mkSession path s!".lake/build/r34_{slug}_tie_{tag}.vmfb"
    IreeSession.forwardF32 sess s!"m.{slug}" params shapes x xsh
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
