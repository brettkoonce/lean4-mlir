import LeanMlir.VerifiedNets
import LeanMlir.GradcheckHelpers

/-! # SGD `@<slug>_train_step` render tie — the `tests/` emitter vs `pretty(provenGraph)`

`planning/xla_pjrt_handoff.md` §2a-quinquies. Four artifacts still have two writers, and for
`convnext_train_step` / `efficientnet_train_step` the two emitters have **positionally identical
interfaces** (182/180 and 264/262), so the R34 move applies: run both renders on one shared input
and compare every returned float, *before* deleting either emitter. Once the `tests/` writer is
gone the comparison cannot be run at all, so it is run first — that is the whole point of this file.

**What is compared is the GRADIENT, not θ'.** An SGD train step returns `θ' = θ − lr·g` and nothing
else — no loss, no BN statistics, 180 in / 180 out for convnext. Comparing θ' directly would be
close to meaningless: θ' is dominated by θ, which is *the same input on both sides*, so a wholly
wrong gradient still lands within `lr·|g| / |θ|` of a match. This harness therefore recovers

    g = (θ − θ') / lr

per side and ties **that**. Two consequences, both deliberate:

* the two sides may carry **different `lr`** and still be tied — which is not hypothetical:
  `tests/TestEfficientNetTrain.lean` bakes `LR = 0.1` while the committed
  `efficientnet_train_step.mlir` (from `Proofs/Codegen/EfficientNetRender.lean`) bakes `0.05`. The
  lr is passed per side on the command line rather than assumed, so that divergence is *recorded*
  instead of silently failing the tie for a reason that has nothing to do with the graph;
* the gate is **per-parameter** norm-relative, `max|gA−gB| / max|gA|` within each parameter. A
  global denominator would let an entire small-gradient layer be wrong and still pass, and a
  per-*coordinate* ratio is meaningless on a near-zero gradient entry (handoff §3).

**What this does NOT establish, unlike the R34 AdamW tie.** That harness could gate the forward pass
bit-exactly, because `@resnet34_adam_train_step` returns the batch μ/var of all 36 BN inputs and
those depend only on the forward. These SGD renders return no forward-only quantity, so there is no
forward/backward bisection here: a forward disagreement and a backward one both land in `g`. The
gradient is still a strong forward check — a mis-wiring at layer k perturbs the activations of every
layer ≥ k and the cotangent reaching every layer < k, so all 180 parameters see it — but it is a
tolerance argument, not a bit-exact one. Say "the two emitters compute the same gradient", not "the
same forward".

    lake build sgd-render-tie
    .lake/build/bin/sgd-render-tie <slug> <pathA> <lrA> <pathB> <lrB>

Exits non-zero if the renders disagree or if the comparison is degenerate. `TIE_SKIP_AA=1` skips the
A-against-itself run, which is the determinism floor: without it "the difference is 1e-6" is an
assertion rather than a measurement (handoff §4).
-/

private def netBySlug (slug : String) : IO VerifiedNetSpec :=
  match slug with
  | "convnext"     => pure convnextVerified
  | "efficientnet" => pure efficientnetVerified
  | "mobilenetv2"  => pure mobilenetv2Verified
  | "resnet34"     => pure resnet34Verified
  -- ViT is here so the XLA/MIOpen patch-embed blocker (handoff §2a) can be probed on the SGD
  -- graph, which carries the SAME two convolutions as the AdamW one. Its train step is the same
  -- (x, θ, onehot) -> θ' shape this harness drives: 202 in / 200 out.
  | "vit"          => pure vitVerified
  | _ => throw (IO.userError
      s!"unknown net slug '{slug}' — expected convnext | efficientnet | mobilenetv2 | resnet34 | vit")

def main (args : List String) : IO Unit := do
  let (slug, pathA, lrAs, pathB, lrBs) ← match args with
    | [s, a, la, b, lb] => pure (s, a, la, b, lb)
    | _ => throw (IO.userError
        "usage: sgd-render-tie <slug> <pathA> <lrA> <pathB> <lrB>\n\
         both learning rates are REQUIRED and are not defaulted: the gradient is recovered as \
         (θ − θ')/lr, so a wrong lr silently rescales one side.")
  -- `ViTGradcheck.parseFloat` yields 0.0 on anything it cannot read, which the next guard catches:
  -- a mistyped lr is the one input error that would silently rescale one side of the comparison.
  let lrA := ViTGradcheck.parseFloat lrAs
  let lrB := ViTGradcheck.parseFloat lrBs
  if lrA == 0.0 || lrB == 0.0 then
    throw (IO.userError s!"lr must be a non-zero float (got '{lrAs}' → {lrA}, '{lrBs}' → {lrB}); \
lr = 0 makes the gradient unrecoverable")
  let net := (← netBySlug slug).toNet
  let bs  := 32                            -- the baked batch of every one of these renders
  IO.println s!"@{slug}_train_step SGD render tie"
  IO.println s!"  A (reference) = {pathA}   lr {lrA}"
  IO.println s!"  B (candidate) = {pathB}   lr {lrB}"
  if lrA != lrB then
    IO.println s!"  NOTE: the two renders bake DIFFERENT learning rates ({lrA} vs {lrB}). They are \
therefore not the same function; the gradient is what is being tied, and it is recovered per side \
by dividing by that side's lr."
  if pathA == pathB && lrA == lrB then
    IO.println "  NOTE: both sides are the same file — this is an A-vs-A determinism run, NOT a \
migration check."
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), bs {bs}, \
backend {← LowererSession.backendName}"

  -- ── θ, in func-arg order, from the driver's init ──
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParamHeFanIn sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  let shapes := packShapes net.paramShapes
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat (i % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0

  let runOne (path tag : String) : IO ByteArray := do
    let sess ← mkSession path
    LowererSession.mlpTrainStepV sess s!"m.{slug}_train_step" x θ shapes y
      bs.toUSize net.d0.toUSize net.nClasses.toUSize

  -- ── the determinism floor: A against itself, in this same process ──
  -- XLA is bit-identical for a single step *within* a process (handoff §3), so this run is what
  -- turns any A-vs-B difference into something graph-attributable rather than backend noise.
  let skipAA := (← IO.getEnv "TIE_SKIP_AA").isSome
  IO.println "  running A…"; (← IO.getStdout).flush
  let oa ← runOne pathA "a"
  if !skipAA then
    IO.println "  running A again (determinism floor)…"; (← IO.getStdout).flush
    let oa2 ← runOne pathA "a2"
    let mut floorDiff : Float := 0.0
    let mut exact : Nat := 0
    for i in [0:oa.size / 4] do
      let d := (F32.read oa i.toUSize - F32.read oa2 i.toUSize).abs
      if d == 0.0 then exact := exact + 1
      if d > floorDiff then floorDiff := d
    IO.println s!"  A-vs-A floor: max|a−a'| = {floorDiff}, bit-exact {exact}/{oa.size / 4}"
    if floorDiff != 0.0 then
      IO.println "  WARNING: A is not bit-stable against itself — every number below is only \
meaningful above this floor."
  IO.println "  running B…"; (← IO.getStdout).flush
  let ob ← runOne pathB "b"

  if oa.size != ob.size then
    IO.eprintln s!"SIZE MISMATCH: {oa.size} vs {ob.size} bytes"; IO.Process.exit 1
  if oa.size / 4 != net.nParams then
    IO.eprintln s!"ARITY MISMATCH: render returned {oa.size / 4} floats, net has {net.nParams} \
params — these are not two spellings of one function"
    IO.Process.exit 1
  let n := oa.size / 4

  -- ── recover the gradient on each side and compare ──────────────────────────────────────────
  -- g = (θ − θ')/lr. Both sides consume the SAME θ, so gA − gB is exactly the graph difference,
  -- rescaled — no cancellation with the (identical, large) θ term.
  let mut maxAbs : Float := 0.0
  let mut maxMag : Float := 0.0
  let mut nonFinite : Nat := 0
  let mut moved : Nat := 0
  for i in [0:n] do
    let t  := F32.read θ  i.toUSize
    let ga := (t - F32.read oa i.toUSize) / lrA
    let gb := (t - F32.read ob i.toUSize) / lrB
    if !ga.isFinite || !gb.isFinite then nonFinite := nonFinite + 1
    let d := (ga - gb).abs
    let mg := max ga.abs gb.abs
    if mg > maxMag then maxMag := mg
    if mg > 1e-12 then moved := moved + 1
    if d > maxAbs then maxAbs := d
  IO.println s!"  {n} gradient coordinates; {moved} non-zero; max|g| = {maxMag}"
  IO.println s!"  global: max|gA−gB| = {maxAbs}, norm-rel = {if maxMag > 1e-30 then maxAbs / maxMag else 0.0}"

  -- ── per-PARAMETER localisation, and the gate ───────────────────────────────────────────────
  -- Per parameter, because a 180-param net's gradients span many orders of magnitude and a single
  -- global denominator would hide a whole wrong layer under the largest layer's scale.
  let mut worstNormRel : Float := 0.0
  let mut worstName := ""
  let mut deadParams : Nat := 0
  let mut offs : Array (String × Float × Float) := #[]
  let mut base := 0
  for pi in [0:net.paramShapes.size] do
    let dims := net.paramShapes[pi]!
    let cnt := dims.foldl (· * ·) 1
    let mut pd : Float := 0.0                -- max |gA − gB| in this param
    let mut pm : Float := 0.0                -- max |gA|      in this param
    for k in [0:cnt] do
      let i := base + k
      let t  := F32.read θ  i.toUSize
      let ga := (t - F32.read oa i.toUSize) / lrA
      let gb := (t - F32.read ob i.toUSize) / lrB
      let d := (ga - gb).abs
      if d > pd then pd := d
      if ga.abs > pm then pm := ga.abs
    let nr := if pm > 1e-30 then pd / pm else 0.0
    -- A parameter whose reference gradient is numerically zero cannot be tied — matching zeros
    -- proves nothing. Counted and reported rather than folded into the max.
    if pm <= 1e-30 then deadParams := deadParams + 1
    else if nr > worstNormRel then worstNormRel := nr; worstName := s!"p{pi} {dims}"
    offs := offs.push (s!"p{pi} {dims}", nr, pd)
    base := base + cnt
  let bad := offs.filter (fun (_, r, _) => r > 1e-4)
  IO.println s!"  params whose gradient disagrees (>1e-4 norm-rel): {bad.size}/{offs.size}\
{if deadParams > 0 then s!"   ({deadParams} have a numerically zero reference gradient and are not gated)" else ""}"
  for (nm, r, a) in bad.toList.take 20 do
    IO.println s!"    {nm}: norm-rel {r}, max abs {a}"
  if bad.size > 20 then IO.println s!"    … and {bad.size - 20} more"

  -- ── degeneracy guards: refuse to report a green tie on nothing ─────────────────────────────
  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite}/{n} non-finite gradient coordinates"; IO.Process.exit 1
  if moved * 10 < n then
    IO.eprintln s!"DEGENERATE: only {moved}/{n} gradient coordinates are non-zero — the tie proves \
little"
    IO.Process.exit 1
  if deadParams * 4 > net.paramShapes.size then
    IO.eprintln s!"DEGENERATE: {deadParams}/{net.paramShapes.size} parameters have a zero reference \
gradient — too much of the net is untested"
    IO.Process.exit 1
  if worstNormRel > 1e-4 then
    IO.eprintln s!"TIE FAILED: worst per-parameter norm-relative gradient diff {worstNormRel} \
(at {worstName}) > 1e-4"
    IO.Process.exit 1
  IO.println s!"✓ renders TIE: worst per-parameter gradient norm-rel {worstNormRel} \
(at {worstName}) ≤ 1e-4 over {offs.size - deadParams} gated parameters"
  IO.println "  (this ties the GRADIENT; unlike the R34 AdamW tie there is no forward-only output \
here, so it does not separately pin the forward pass bit-exactly — see the file docstring)"
