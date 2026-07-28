import LeanMlir.VerifiedNets

/-! # `@resnet34_adam_train_step` render tie — hand-written vs `pretty(provenGraph)`

`planning/xla_pjrt_handoff.md` §2b step 5. The committed
`verified_mlir/resnet34_adam_train_step.mlir` **now** renders from
`LeanMlir/Proofs/Codegen/ResNet34RenderB.lean` as `pretty(provenGraph)` at the batched index
`N := B`, with the un-fused `*GradB` gradients feeding the proven AdamW ops. It used to come from
the hand-written string emitter in `tests/TestResnet34Train.lean`; this harness is what licensed
that swap, and the hand-written AdamW render is now retired (that emitter renders only the
data-parallel variant, to `…_dp.mlir`).

**Post-swap, the no-argument form compares the artifact against itself** — an A-vs-A run, which is
still worth having (it re-establishes the determinism floor the gate depends on) but is not a
migration check. To re-run the real tie, recover the retired render and pass it explicitly:

    git show b856deb:verified_mlir/resnet34_adam_train_step.mlir > /tmp/retired.mlir
    .lake/build/bin/resnet34-adam-tie /tmp/retired.mlir

**Why this has to be a numeric tie and not a text diff.** The two are the same function but not the
same graph. SSA names differ (tagged vs counter), and so does the op sequence: the hand-written
render fuses label smoothing and the softmax into one `[B,10]` block, while the kit composes
`softmaxRow → subB → scaleB → addVB → shiftB → divConstB` with the batched ops' reshape
round-trips. So neither a byte diff nor the SSA-name-independent verb-sequence trick
(`tests/TestAdamOpTie.lean`) applies — only running both and comparing every returned float does.

The interface IS identical (515 in / 513 out, types positionally equal), so both take the same
packed `[θ|m|v | lr,bc1,bc2 | bn stats]` buffer that `trainAdamSched` builds.

    lake build resnet34-adam-tie
    .lake/build/bin/resnet34-adam-tie [refRender.mlir] [newRender.mlir]

Exits non-zero if the renders disagree, or if the comparison is degenerate. The degeneracy guards
matter more here than usual: a batch-BN net at an untrained init can produce near-zero gradients
over most coordinates, and a tie on all-zeros proves nothing.
-/

private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0          -- BN γ
  | 2 => F32.const n.toUSize 0.0          -- BN β / biases
  | _ =>
    let fanIn := if dims.size == 4 then dims[1]! * dims[2]! * dims[3]! else dims[0]!
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))

def main (args : List String) : IO Unit := do
  let dflt := "verified_mlir/resnet34_adam_train_step.mlir"
  let (pathA, pathB) := match args with
    | a :: b :: _ => (a, b)
    | [a]         => (a, dflt)
    | []          => (dflt, dflt)
  let net := resnet34Verified.toNet
  let bs  := 32                            -- the baked batch of both renders
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  IO.println s!"@resnet34_adam_train_step tie"
  IO.println s!"  A (reference) = {pathA}"
  IO.println s!"  B (candidate) = {pathB}"
  if pathA == pathB then
    IO.println "  NOTE: both paths are the same file — this is an A-vs-A determinism run, NOT a \
migration check. Pass the retired render as the first argument for that."
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), {net.bnChannels.size} BN layers \
({nBnStats} stat floats), bs {bs}, backend {← IreeSession.backendName}"

  -- ── θ (driver init), m (centred noise), v (POSITIVE — it sits under a sqrt) ──
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  let m ← F32.heInit 4242 net.nParams.toUSize 0.02
  -- v ≈ 0.05 ± 0.01, strictly positive: a zero v makes √v̂ + ε ≈ ε everywhere and would hide any
  -- error in the second-moment path.
  let v ← F32.scaleShift (← F32.heInit 8484 net.nParams.toUSize 0.01) 1.0 0.05
  let tail ← F32.const 3 0.0
  let tail ← F32.write3 tail 0 0.001 0.19 0.002    -- lr, 1−β₁ᵗ, 1−β₂ᵗ at a mid-training step
  -- BN passthrough slots. The hand-written render ignores their VALUES (it recomputes the batch
  -- statistics), so any finite fill is fine; non-zero is used so a slot that is wrongly echoed
  -- rather than recomputed shows up as a mismatch instead of matching a zero by luck.
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let pbuf := F32.concat #[θ, m, v, tail, bnIn]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]] ++ bnStatShapes)
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat (i % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0

  let runOne (path tag : String) : IO ByteArray := do
    let sess ← mkSession path s!".lake/build/r34_adam_tie_{tag}.vmfb"
    IreeSession.mlpTrainStepV sess "m.resnet34_adam_train_step" x pbuf shapes y
      bs.toUSize net.d0.toUSize net.nClasses.toUSize
  IO.println "  running A…"; (← IO.getStdout).flush
  let oa ← runOne pathA "a"
  IO.println "  running B…"; (← IO.getStdout).flush
  let ob ← runOne pathB "b"

  if oa.size != ob.size then
    IO.eprintln s!"SIZE MISMATCH: {oa.size} vs {ob.size} bytes"; IO.Process.exit 1
  let n := oa.size / 4
  let nP := net.nParams
  -- Report per REGION, not just globally: θ' is scale-free under Adam (§3 — a near-zero-gradient
  -- coordinate flips sign on a 1-ULP difference and moves a full ±lr), so a θ' mismatch and an
  -- m' mismatch mean very different things and must not be averaged into one number.
  let region (i : Nat) : String :=
    if i < nP then "theta" else if i < 2*nP then "m" else if i < 3*nP then "v"
    else if i < 3*nP + 3 then "loss/bc" else "bnstat"
  let mut maxAbs : Float := 0.0
  let mut maxRel : Float := 0.0
  let mut maxMag : Float := 0.0
  let mut nonFinite : Nat := 0
  let mut moved : Nat := 0
  let mut worstIdx : Nat := 0
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
    if r > maxRel then maxRel := r; worstIdx := i

  IO.println s!"  {n} returned floats; {moved} non-zero; |max| = {maxMag}"
  IO.println s!"  max abs diff = {maxAbs}   max rel diff = {maxRel}  (worst at {worstIdx}, region {region worstIdx})"

  -- ── per-REGION bisection ──────────────────────────────────────────────────────────────────
  -- `bnstat` depends ONLY on the forward pass (batch μ/var of each BN input), so it separates a
  -- forward disagreement from a backward one in a single run. `m` is `(1−β₁)·g` off a shared `m`,
  -- so it is the gradient. Reported as max|a−b| and as max|a−b| / max|a| — the NORM-relative
  -- error, because a per-coordinate ratio on a near-zero gradient entry is meaningless (§3).
  let regions : List (String × Nat × Nat) :=
    [("theta", 0, nP), ("m", nP, 2*nP), ("v", 2*nP, 3*nP),
     ("loss/bc", 3*nP, 3*nP+3), ("bnstat", 3*nP+3, n)]
  IO.println "  ── per region ──"
  let mut fwdExact := true
  let mut worstNormRel : Float := 0.0
  for (nm, lo, hi) in regions do
    let mut ra : Float := 0.0
    let mut rm : Float := 0.0
    let mut rr : Float := 0.0
    let mut exact : Nat := 0
    for i in [lo:hi] do
      let a := F32.read oa i.toUSize
      let b := F32.read ob i.toUSize
      let d := (a - b).abs
      if d == 0.0 then exact := exact + 1
      if d > ra then ra := d
      if a.abs > rm then rm := a.abs
      let r := if max a.abs b.abs > 1e-30 then d / max a.abs b.abs else 0.0
      if r > rr then rr := r
    let normRel := if rm > 1e-30 then ra / rm else 0.0
    if normRel > worstNormRel then worstNormRel := normRel
    if nm == "bnstat" && exact != hi - lo then fwdExact := false
    IO.println s!"    {nm}: max|a-b| = {ra}, max|a| = {rm}, norm-rel = {normRel}, \
per-coord max rel = {rr}, bit-exact {exact}/{hi-lo}"

  -- ── per-PARAMETER localisation over the `m` region ────────────────────────────────────────
  -- `m' = β₁·m + (1−β₁)·g` off the same input `m`, so a disagreement here IS a gradient
  -- disagreement, scaled by (1−β₁). A single global max says "something is wrong"; naming the
  -- layer says what. Walk the param list in signature order and report the worst offenders.
  let mut offs : Array (String × Float × Float) := #[]
  let mut base := nP                       -- start of the m region
  for pi in [0:net.paramShapes.size] do
    let dims := net.paramShapes[pi]!
    let cnt := dims.foldl (· * ·) 1
    let mut pr : Float := 0.0
    let mut pa : Float := 0.0
    for k in [0:cnt] do
      let i := base + k
      let a := F32.read oa i.toUSize
      let b := F32.read ob i.toUSize
      let d := (a - b).abs
      let mg := max a.abs b.abs
      if d > pa then pa := d
      let r := if mg > 1e-30 then d / mg else 0.0
      if r > pr then pr := r
    offs := offs.push (s!"p{pi} {dims}", pr, pa)
    base := base + cnt
  let bad := offs.filter (fun (_, r, _) => r > 1e-4)
  IO.println s!"  params whose m' disagrees (>1e-4 rel): {bad.size}/{offs.size}"
  for (nm, r, a) in bad.toList.take 20 do
    IO.println s!"    {nm}: max rel {r}, max abs {a}"
  if bad.size > 20 then IO.println s!"    … and {bad.size - 20} more"
  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite}/{n} non-finite outputs"; IO.Process.exit 1
  if moved * 10 < n then
    IO.eprintln s!"DEGENERATE: only {moved}/{n} outputs are non-zero — the tie proves little"
    IO.Process.exit 1
  -- ── the GATE ──────────────────────────────────────────────────────────────────────────────
  -- NOT a per-coordinate relative gate. §3 of the handoff establishes that R34's gradient does not
  -- reproduce to better than ~6e-3 per-coordinate even XLA-vs-XLA under a sub-ULP nudge, so a 1e-4
  -- per-coordinate gate fails a correct render by 60× — it is dominated by near-zero gradient
  -- entries where a sub-ULP absolute difference is a huge ratio. Two gates that do mean something:
  --
  --   1. the FORWARD must be BIT-EXACT. `bnstat` is the batch μ/var of all 36 BN inputs, so it
  --      pins the entire forward chain — stem, every block, every BN — to the last bit. Any real
  --      mis-wiring of the forward shows up here as a hard failure, not a tolerance argument.
  --   2. the backward must agree to NORM-relative 1e-4: max|a−b| / max|a| over each region, which
  --      is scale-free in the right way (a tiny coordinate contributes a tiny numerator).
  --
  -- Run A against itself to see the floor: it is bit-exact on every one of the 68M outputs, so a
  -- non-zero A-vs-B difference is attributable to the graph, never to backend nondeterminism.
  if !fwdExact then
    IO.eprintln "TIE FAILED: the forward differs — `bnstat` (batch μ/var of all 36 BN inputs) is \
not bit-exact, so the two renders do not compute the same forward pass"
    IO.Process.exit 1
  if worstNormRel > 1e-4 then
    IO.eprintln s!"TIE FAILED: worst norm-relative diff {worstNormRel} > 1e-4"
    IO.Process.exit 1
  IO.println s!"✓ renders TIE: forward BIT-EXACT (bnstat), backward norm-rel {worstNormRel} ≤ 1e-4"
  IO.println s!"  (per-coordinate max rel {maxRel} is reported for information only — see §3; it is \
dominated by near-zero gradient entries and is NOT the gate)"
