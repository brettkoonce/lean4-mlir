import LeanMlir.VerifiedNets

/-! # cifar8 optimizer-render tie — one harness for `adam` / `sgd` / `mom` (handoff §2i)

    lake build cifar8-opt-tie
    .lake/build/bin/cifar8-opt-tie <adam|sgd|mom> [<refRender.mlir> [<newRender.mlir>]]

The three variants share ONE forward, backward and un-fused-gradient body and one packed
`[θ|m|v|lr|bc1|bc2]` signature (71 in / 69 out for every one), so a single harness drives them all;
the slug selects the entry point `@cifar8_<slug>_train_step` and the gradient-recovery formula.

## It gates the RECOVERED GRADIENT, never θ′ — and for SGD that is the whole ballgame

§2a-quinquies found this the hard way on the SGD renders: a train step returns `θ' = θ − lr·g`, and
θ' is dominated by `θ`, **the same input on both sides**. At the lr used here (1e-3) a *wholly wrong*
gradient still lands within `lr·|g|/|θ|` of a match, so a θ'-based tie is close to meaningless.
Every variant's gradient is exactly recoverable from its own outputs, so that is what is compared:

| slug | recovery | from |
|---|---|---|
| `adam` | `g = (m' − β₁·m)/(1−β₁)` | the first moment |
| `sgd`  | `g = (θ − θ')/lr` | the only output that moves |
| `mom`  | `g = v' − μ·v` | the velocity, `v' = μv + g` |

All three are exact algebra on values this harness supplies, not approximations. (`momVNext_v_zero`
and `adamMNextF`'s `m' = (1−β₁)g` are the same observation, and it is what `convnext-shard-check`
relies on too.)

## The passthrough check, which is free and catches a real bug class

`sgd` must return `m' = m` and `v' = v` **bit-exactly**, and `mom` must return `m' = m` bit-exactly —
the packed signature is shared with AdamW precisely so the driver never changes, and a tail that
silently zeroed or dropped a moment slot would still produce a plausible θ'. Gated, not reported.

Deletes its `.vmfb` before every compile (§4's false-PASS hazard: `compileVmfb` keys on the OUTPUT
path plus an mtime, never the source, so running one binary twice with different candidates silently
reuses the first one's binary — which is exactly what comparing a staged render against an incumbent
looks like). `cifar8-adam-tie` still has that hazard; prefer this harness.
-/

private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0
  | 2 => F32.const n.toUSize 0.0
  | _ =>
    let fanIn := if dims.size == 4 then dims[1]! * dims[2]! * dims[3]! else dims[0]!
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))

def main (args : List String) : IO Unit := do
  let (slug, rest) := match args with
    | s :: r => (s, r)
    | []     => ("adam", [])
  if slug != "adam" && slug != "sgd" && slug != "mom" then
    IO.eprintln s!"unknown slug '{slug}' — expected adam | sgd | mom"; IO.Process.exit 1
  let dflt := s!"verified_mlir/cifar8_{slug}_train_step.mlir"
  let (pathA, pathB) := match rest with
    | a :: b :: _ => (a, b)
    | [a]         => (a, dflt)
    | []          => (dflt, dflt)
  let net := cifar8Verified.toNet
  let bs  := 128
  let nP  := net.nParams
  let lr  := 0.001; let β₁ := 0.9; let μ := 0.9
  IO.println s!"@cifar8_{slug}_train_step tie: A={pathA}  B={pathB}"
  IO.println s!"  {net.specs.size} params ({nP} floats), bs {bs}, backend {← IreeSession.backendName}"

  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  let m ← F32.heInit 4242 nP.toUSize 0.02
  -- v must be strictly positive for adam (it sits under a sqrt); for mom it is the incoming
  -- velocity and any value works, so one buffer serves all three variants.
  let v ← F32.scaleShift (← F32.heInit 8484 nP.toUSize 0.01) 1.0 0.05
  let tail ← F32.const 3 0.0
  let tail ← F32.write3 tail 0 lr 0.19 0.002
  let pbuf := F32.concat #[θ, m, v, tail]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]])
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat (i % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0

  let runOne (path tag : String) : IO ByteArray := do
    for p in [s!".lake/build/c8_opt_{slug}_{tag}.vmfb",
              s!".lake/build/c8_opt_{slug}_{tag}_{((← IO.getEnv "IREE_BACKEND").getD "cuda")}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
    let sess ← mkSession path s!".lake/build/c8_opt_{slug}_{tag}.vmfb"
    IreeSession.mlpTrainStepV sess s!"m.cifar8_{slug}_train_step" x pbuf shapes y
      bs.toUSize net.d0.toUSize net.nClasses.toUSize
  let oa ← runOne pathA "a"
  let ob ← runOne pathB "b"
  if oa.size != ob.size then
    IO.eprintln s!"SIZE MISMATCH: {oa.size} vs {ob.size} bytes"; IO.Process.exit 1

  -- Recover the gradient from one side's outputs — exact algebra, see the table above.
  let gradOf (o : ByteArray) (i : Nat) : Float :=
    match slug with
    | "adam" => (F32.read o (nP + i).toUSize - β₁ * F32.read m i.toUSize) / (1.0 - β₁)
    | "sgd"  => (F32.read θ i.toUSize - F32.read o i.toUSize) / lr
    | _      => F32.read o (2*nP + i).toUSize - μ * F32.read v i.toUSize

  let mut gAbs : Float := 0.0; let mut gMag : Float := 0.0
  let mut nonFinite : Nat := 0; let mut movedG : Nat := 0
  -- Count EXACT coordinates. `Float.toString` gives six decimals, so a genuine 3e-8 prints as
  -- "0.000000" and reads as bit-exact when it is not — §2e-bis hit exactly that, and the count is
  -- what distinguishes "bit-exact" from "prints as zero".
  let mut exactG : Nat := 0
  for i in [0:nP] do
    let ga := gradOf oa i; let gb := gradOf ob i
    if !ga.isFinite || !gb.isFinite then nonFinite := nonFinite + 1
    if ga.abs > 1e-9 then movedG := movedG + 1
    if ga.abs > gMag then gMag := ga.abs
    let d := (ga - gb).abs
    if d == 0.0 then exactG := exactG + 1
    if d > gAbs then gAbs := d
  let gRel := if gMag > 1e-30 then gAbs / gMag else 0.0

  -- passthrough slots: which of m'/v' must equal their INPUT bit-exactly
  let passthru : List (String × Nat × ByteArray) := match slug with
    | "sgd" => [("m", nP, m), ("v", 2*nP, v)]
    | "mom" => [("m", nP, m)]
    | _     => []
  let mut ptBad : List String := []
  for (nm, off, src) in passthru do
    let mut bad := 0
    for i in [0:nP] do
      if F32.read oa (off + i).toUSize != F32.read src i.toUSize then bad := bad + 1
    if bad != 0 then ptBad := ptBad ++ [s!"{nm} ({bad}/{nP} slots changed)"]

  let la := F32.read oa (3*nP).toUSize; let lb := F32.read ob (3*nP).toUSize
  IO.println s!"  ── recovered gradient ({nP} coords) ──"
  IO.println s!"    max|gA−gB| = {gAbs} ({gAbs * 1e9} e-9), max|gA| = {gMag}, \
norm-rel = {gRel} ({gRel * 1e9} e-9), bit-exact {exactG}/{nP}"
  IO.println s!"    non-zero gradients: {movedG}/{nP}"
  IO.println s!"  %loss  A = {la}  B = {lb}  (abs diff {(la-lb).abs})"
  if !passthru.isEmpty then
    IO.println s!"  passthrough slots checked: {passthru.map (·.1)} → \
{if ptBad.isEmpty then "all bit-exact" else toString ptBad}"

  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite}/{nP} non-finite recovered gradients"; IO.Process.exit 1
  if movedG * 10 < nP then
    IO.eprintln s!"DEGENERATE: only {movedG}/{nP} gradients are non-zero — the tie proves little"
    IO.Process.exit 1
  if !ptBad.isEmpty then
    IO.eprintln s!"TIE FAILED: passthrough slots are NOT bit-exact: {ptBad}. The packed \
[θ|m|v] signature is shared with AdamW so the driver never changes; a tail that drops or zeroes a \
moment slot still produces a plausible θ', which is why this is gated."
    IO.Process.exit 1
  if (la - lb).abs > 1e-4 then
    IO.eprintln s!"TIE FAILED: %loss differs by {(la-lb).abs} > 1e-4"; IO.Process.exit 1
  if gRel > 1e-4 then
    IO.eprintln s!"TIE FAILED: recovered-gradient norm-rel {gRel} > 1e-4"; IO.Process.exit 1
  IO.println s!"✓ {slug} renders TIE — recovered gradient norm-rel {gRel} ≤ 1e-4, %loss within \
1e-4, passthrough slots bit-exact, over {movedG} non-degenerate gradient coords"
