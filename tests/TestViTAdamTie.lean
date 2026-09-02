import LeanMlir.VerifiedNets

/-! # `@vit_adam_train_step` render tie — hand-written vs `pretty(provenGraph)`

`planning/xla_pjrt_handoff.md`, the ViT AdamW thread, step 3. `Proofs/Codegen/ViTRender.lean`'s
`vitAdamTrainStepFaithful` renders the same train step the hand-written
`LeanMlir/ViTRender.vitTrainStepModuleAdamSched` does — the one
`apps/imagenette/MainViTVerifiedAdam.lean` writes at startup and trains on. This harness is what
licenses swapping them; run it BEFORE deleting the driver's writer, because afterwards the
comparison no longer exists.

The interface is positionally identical (605 in / 603 out, arg and return types equal in order —
only the parameter NAMES differ, `%Wq_0` vs `%b0_Wq`, which the packed-buffer FFI never sees), so
both take the same `[θ|m|v | lr,bc1,bc2]` blob `trainAdamSched` builds.

**Why a numeric tie and not a text diff.** Same function, different graph: SSA naming differs, and
the certified render emits 14,881 ops against 7,700 because `pretty` has no CSE. Only running both
and comparing every returned float settles it.

**What this gates, and what it cannot.** ViT has **no BatchNorm**, so unlike `resnet34-adam-tie`
there is no `bnstat` region — nothing in the output depends on the forward alone, so a forward
disagreement and a backward one both land in `m` and cannot be separated. Two consequences:

* `%loss` earns its keep here. It is report-only and on no gradient path, so no theorem covers it —
  and §2b shipped exactly this wrong once (plain CE against a smoothed-CE cotangent). With no
  `bnstat` to pin the forward, `%loss` is the *only* output that reads the forward directly, so a
  loss mismatch against matching gradients is the signature of a forward/cotangent bug.
* the gradient gate is norm-relative, not per-coordinate (handoff §3).

    lake build vit-adam-tie
    .lake/build/bin/vit-adam-tie <refRender.mlir> <candRender.mlir>

Exits non-zero if the renders disagree or the comparison is degenerate.
-/

def main (args : List String) : IO Unit := do
  let dflt := "verified_mlir/vit_adam_train_step.mlir"
  let (pathA, pathB) := match args with
    | a :: b :: _ => (a, b)
    | [a]         => (dflt, a)
    | []          => (dflt, dflt)
  let net := vitVerified.toNet
  let bs  := 32
  IO.println "@vit_adam_train_step tie"
  IO.println s!"  A (reference, hand-written) = {pathA}"
  IO.println s!"  B (candidate, certified)    = {pathB}"
  if pathA == pathB then
    IO.println "  NOTE: both paths are the same file — an A-vs-A determinism run, NOT a migration \
check."
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), no BN, bs {bs}, \
backend {← LowererSession.backendName}"

  -- ── the packed [θ|m|v|lr,bc1,bc2] blob, byte-identical to both sides ──
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  let m ← F32.heInit 4242 net.nParams.toUSize 0.02
  -- v strictly positive: a zero v makes √v̂ + ε ≈ ε everywhere and would hide the second-moment path
  let v ← F32.scaleShift (← F32.heInit 8484 net.nParams.toUSize 0.01) 1.0 0.05
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
    LowererSession.mlpTrainStepV sess "m.vit_adam_train_step" x pbuf shapes y
      bs.toUSize net.d0.toUSize net.nClasses.toUSize
  IO.println "  running A…"; (← IO.getStdout).flush
  let oa ← runOne pathA "a"
  -- the determinism floor: without it "the difference is 1e-6" is an assertion, not a measurement
  IO.println "  running A again (determinism floor)…"; (← IO.getStdout).flush
  let oa2 ← runOne pathA "a2"
  let mut floor : Float := 0.0
  let mut fex : Nat := 0
  for i in [0:oa.size / 4] do
    let d := (F32.read oa i.toUSize - F32.read oa2 i.toUSize).abs
    if d == 0.0 then fex := fex + 1
    if d > floor then floor := d
  IO.println s!"  A-vs-A floor: max|a−a'| = {floor}, bit-exact {fex}/{oa.size / 4}"
  IO.println "  running B…"; (← IO.getStdout).flush
  let ob ← runOne pathB "b"

  if oa.size != ob.size then
    IO.eprintln s!"SIZE MISMATCH: {oa.size} vs {ob.size} bytes"; IO.Process.exit 1
  let n := oa.size / 4
  let nP := net.nParams
  if n != 3 * nP + 3 then
    IO.eprintln s!"ARITY: expected {3*nP+3} returned floats, got {n}"; IO.Process.exit 1

  -- ── per region ────────────────────────────────────────────────────────────────────────────
  -- `m' = β₁·m + (1−β₁)·g` off a shared `m`, so the `m` region IS the gradient. θ' is scale-free
  -- under Adam (§3: a near-zero-gradient coordinate flips sign on a 1-ULP difference and moves a
  -- full ±lr), so θ' is reported but NOT the gate.
  let regions : List (String × Nat × Nat) :=
    [("theta", 0, nP), ("m", nP, 2*nP), ("v", 2*nP, 3*nP), ("loss/bc", 3*nP, n)]
  let mut nonFinite : Nat := 0
  let mut moved : Nat := 0
  let mut gradNormRel : Float := 0.0
  let mut lossRel : Float := 0.0
  IO.println "  ── per region ──"
  for (nm, lo, hi) in regions do
    let mut ra : Float := 0.0
    let mut rm : Float := 0.0
    let mut exact : Nat := 0
    for i in [lo:hi] do
      let a := F32.read oa i.toUSize
      let b := F32.read ob i.toUSize
      if !a.isFinite || !b.isFinite then nonFinite := nonFinite + 1
      if max a.abs b.abs > 1e-12 then moved := moved + 1
      let d := (a - b).abs
      if d == 0.0 then exact := exact + 1
      if d > ra then ra := d
      if a.abs > rm then rm := a.abs
    let nr := if rm > 1e-30 then ra / rm else 0.0
    if nm == "m" then gradNormRel := nr
    if nm == "loss/bc" then lossRel := nr
    IO.println s!"    {nm}: max|a-b| = {ra}, max|a| = {rm}, norm-rel = {nr}, \
bit-exact {exact}/{hi-lo}"

  -- ── per-PARAMETER localisation over `m` — naming the layer beats one global max ──
  let mut bad : Nat := 0
  let mut worst := ""
  let mut worstR : Float := 0.0
  let mut base := nP
  for pi in [0:net.paramShapes.size] do
    let dims := net.paramShapes[pi]!
    let cnt := dims.foldl (· * ·) 1
    let mut pd : Float := 0.0
    let mut pm : Float := 0.0
    for k in [0:cnt] do
      let i := base + k
      let d := (F32.read oa i.toUSize - F32.read ob i.toUSize).abs
      if d > pd then pd := d
      let a := (F32.read oa i.toUSize).abs
      if a > pm then pm := a
    let nr := if pm > 1e-30 then pd / pm else 0.0
    if nr > 1e-4 then bad := bad + 1
    if nr > worstR then worstR := nr; worst := s!"p{pi} {dims}"
    base := base + cnt
  IO.println s!"  params whose gradient disagrees (>1e-4 norm-rel): {bad}/{net.paramShapes.size}\
   (worst {worst} at {worstR})"

  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite}/{n} non-finite outputs"; IO.Process.exit 1
  if moved * 10 < n then
    IO.eprintln s!"DEGENERATE: only {moved}/{n} outputs are non-zero — the tie proves little"
    IO.Process.exit 1
  -- ── the gate ──────────────────────────────────────────────────────────────────────────────
  -- No `bnstat` region exists (ViT has no BN), so the forward cannot be pinned bit-exactly the way
  -- resnet34-adam-tie pins it. `%loss` is the only output reading the forward directly, and it is
  -- exactly what §2b got wrong (plain CE vs the smoothed CE its own cotangent implied) — so it is
  -- gated, not merely reported.
  if lossRel > 1e-4 then
    IO.eprintln s!"TIE FAILED: the loss/bc region differs at norm-rel {lossRel} > 1e-4. With no BN \
statistics in the output, %loss is the only direct read of the forward — a mismatch here against \
matching gradients is the signature of a forward or cotangent bug (§2b shipped exactly that)."
    IO.Process.exit 1
  if gradNormRel > 1e-4 then
    IO.eprintln s!"TIE FAILED: gradient (m) norm-relative diff {gradNormRel} > 1e-4"
    IO.Process.exit 1
  IO.println s!"✓ renders TIE: gradient norm-rel {gradNormRel} ≤ 1e-4, loss {lossRel} ≤ 1e-4, \
over all {n} returned floats"
  IO.println "  (no BN ⇒ no forward-only region; this ties the GRADIENT and the LOSS, and does not \
separately pin the forward bit-exactly — see the file docstring)"
