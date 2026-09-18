import LeanMlir.VerifiedNets

/-! # `cnx-init-check` — the known-answer gate for ConvNeXt's verified weight init

⛔ **THE DEFECT THIS EXISTS FOR.** The 2026-09-17 ConvNeXt/ImageNet pair run was killed at epoch 67
because the two arms did not share a weight init: the JAX reference sets `cnxInit := true`
(ConvNeXt `_init_weights`, `trunc_normal(0.02)` on every conv AND the head) while the verified path
used `mkParam`'s He default. Two arms with different inits cannot isolate the lowerer, which is the
one thing this BatchNorm-free net is in the book for.

⚠⚠ **AND THE INIT IS EXACTLY THE KIND OF THING NOBODY CAN READ OFF THE SOURCE RELIABLY** — it is
host-side, it never enters a committed artifact, and `mkParam`'s rank-4 default is He **fan-OUT**
(`2/(oc·kh·kw)`) while `SpecHelpers.heInitLayer` uses fan-**IN** (`2/(ic·kh·kw)`). Reading the wrong
one off the wrong file gives ratios that are wrong by an order of magnitude and in the wrong
DIRECTION for some layer shapes. So this gate MEASURES the emitted parameters instead.

What it asserts, on `convnextImagenetVerified`'s real 183-spec layout:

* with `cnxInit := true`, every WEIGHT spec (kind 0) lands at **σ = 0.02**;
* the other three kinds are untouched — LayerNorm γ exactly 1.0, biases exactly 0.0,
  LayerScale γ exactly 1e-6 — because ConvNeXt's `_init_weights` sets those and they already matched;
* ⚠ **CONTROL**: with `cnxInit := false` the weights must land somewhere ELSE, or the gate is
  reading a flag that does nothing. Reported per distinct shape so the real ratios are on the record.

Run:  lake exe cnx-init-check
-/



/-- Population mean and σ of an f32 blob. -/
private def stats (ba : ByteArray) : Float × Float := Id.run do
  let n := F32.size ba
  if n == 0 then return (0.0, 0.0)
  let mut s := 0.0
  for i in [0:n] do s := s + F32.read ba i.toUSize
  let m := s / n.toFloat
  let mut v := 0.0
  for i in [0:n] do
    let d := F32.read ba i.toUSize - m
    v := v + d * d
  return (m, Float.sqrt (v / n.toFloat))

private def fmt (x : Float) : String :=
  let r := (x * 1000000.0).round / 1000000.0
  toString r

/-- Largest |x − target| over the blob. -/
private def maxDev (ba : ByteArray) (target : Float) : Float := Id.run do
  let n := F32.size ba
  let mut m := 0.0
  for i in [0:n] do
    let d := (F32.read ba i.toUSize - target).abs
    if d > m then m := d
  return m

def main : IO Unit := do
  let net := convnextImagenetVerified.toNet
  let specs := net.specs
  IO.println s!"── ConvNeXt-T / ImageNet verified init — {specs.size} specs ──"

  let mut bad : Array String := #[]

  -- ═══ 1. cnxInit := true — every weight at σ = 0.02, everything else untouched ═══
  let mut nW := 0
  let mut worstW := 0.0
  let mut seed : Nat := 1
  let mut shapesOn : Array (String × Float) := #[]
  for spec in specs do
    let (dims, kind) := spec
    let ba ← mkParam seed dims kind false none false true
    let (_, sd) := stats ba
    match kind with
    | 0 =>
      nW := nW + 1
      let rel := ((sd - 0.02) / 0.02).abs
      if rel > worstW then worstW := rel
      if rel > 0.05 then
        bad := bad.push s!"weight {dims} σ={fmt sd} — wanted 0.02 (rel {fmt rel})"
      if !(shapesOn.any (fun p => p.1 == toString dims)) then
        shapesOn := shapesOn.push (toString dims, sd)
    | 1 => let d := maxDev ba 1.0
           if d > 0.0 then bad := bad.push s!"LayerNorm γ {dims} deviates {fmt d} from 1.0"
    | 2 => let d := maxDev ba 0.0
           if d > 0.0 then bad := bad.push s!"bias {dims} deviates {fmt d} from 0.0"
    | 3 => let d := maxDev ba 1e-6
           if d > 1e-12 then bad := bad.push s!"LayerScale γ {dims} deviates {fmt d} from 1e-6"
    | k => bad := bad.push s!"unexpected kind {k} at {dims}"
    seed := seed + 1
  IO.println s!"  cnxInit=true : {nW} weight specs, worst relative σ error {fmt worstW} (tol 0.05)"

  -- ═══ 2. ⚠ THE CONTROL — cnxInit := false must give something ELSE ═══
  IO.println "── CONTROL: cnxInit=false, the default the killed run trained under ──"
  IO.println "  (per distinct weight shape: what the default emits, and its ratio to 0.02)"
  seed := 1
  let mut seen : Array String := #[]
  let mut moved := 0
  for spec in specs do
    let (dims, kind) := spec
    if kind == 0 then
      let ba ← mkParam seed dims kind false none false false
      let (_, sd) := stats ba
      let key := toString dims
      if !(seen.any (fun s => s == key)) then
        seen := seen.push key
        IO.println s!"    {key}  σ={fmt sd}  = {fmt (sd / 0.02)}x the reference's 0.02"
      if ((sd - 0.02) / 0.02).abs > 0.05 then moved := moved + 1
    seed := seed + 1
  IO.println s!"  {moved} of {nW} weight specs differ from 0.02 under the default"
  if moved == 0 then
    bad := bad.push "CONTROL DEAD — cnxInit=false gives the SAME init as cnxInit=true, so this gate proves nothing"

  if bad.isEmpty then
    IO.println "✅ cnx-init-check PASS — every weight at σ=0.02, γ/β/LayerScale untouched, control fires"
  else
    IO.println s!"⛔ cnx-init-check FAIL — {bad.size} problem(s):"
    for b in bad do IO.println s!"   {b}"
    throw (IO.userError "cnx-init-check failed")
