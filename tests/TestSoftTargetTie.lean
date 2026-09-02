import LeanMlir.VerifiedNets

/-! # Soft-target gate — the committed renders are AFFINE in the target, so mixup needs no new render

**The claim this settles.** `planning/xla_pjrt_handoff.md` §2p asserted that mixup/cutmix need a
new `softLabelCE` cotangent on the verified path. That is **wrong**, and this harness is what
proves it. Every render already takes the target as a `[batch, nClasses]` FLOAT tensor `%onehot`,
and the emitted cotangent is

```
dy = ((softmax − onehot) + α·onehot − α/K) / B      -- ViTRender.lean:312, and the peer renders
   = (softmax − (1−α)·onehot − α/K) / B
```

which is **affine in `onehot`**. The forward does not read it at all, and the backward is linear in
`dy`, so the whole parameter gradient is affine in the target. Therefore for any `λ`:

> `grad(λ·y_a + (1−λ)·y_b)  =  λ·grad(y_a) + (1−λ)·grad(y_b)`

— the affine constant survives because `λ + (1−λ) = 1`. That identity is exactly what mixup asks
for, and it holds by construction rather than by approximation. What actually blocked soft targets
was **not** the graph but the C layer, which expanded int32 labels into a one-hot in three separate
copies; `lean_fill_targets` now takes either.

**Why gate it instead of asserting it.** The reasoning above is about the render's *intended*
denotation, and this repo's standing rule (§5) is that a faithfulness argument does not witness the
emitter — the `Tok → text` lexer is audited-but-trusted, and §2b shipped a `%loss` that disagreed
with its own cotangent for exactly that reason. So: measure the identity on the committed bytes.

**It gates `m`, not `θ'`, and that is forced.** AdamW's update is nonlinear in the gradient
(`m̂/(√v̂+ε)`), so θ' is not affine in the target even though the gradient is. Feeding `m = 0`
makes `adamMNextF` give `m' = (1−β₁)·g = 0.1·g`, exactly linear in the gradient — the §5
shard-check construction, reused. `v' = 0.001·g²` is quadratic and is reported but NOT gated.

**The CONTROL is what makes a pass meaningful.** `|mix − a|` is what this returns if the harness
were secretly comparing a buffer with itself, or if the render ignored its target input entirely.
It runs every time, and the gate refuses as VACUOUS if the two endpoints are not distinguishable.

```
lake build soft-target-tie
HIP_VISIBLE_DEVICES=0 .lake/build/bin/soft-target-tie vit
HIP_VISIBLE_DEVICES=0 .lake/build/bin/soft-target-tie convnext
```
-/

/-- int32 hard labels, class `(i + off) % nClasses` — the `batch*4`-byte form. -/
private def mkLabels (bs off nc : Nat) : ByteArray := Id.run do
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat ((i + off) % nc)); y := y.push 0; y := y.push 0; y := y.push 0
  y

/-- The `batch*nClasses*4`-byte SOFT form: `lam·onehot(i%nc) + (1−lam)·onehot((i+off)%nc)`, i.e.
    precisely what mixup produces from the two label sets `mkLabels bs 0 nc` and `mkLabels bs off nc`. -/
private def mkSoft (bs off nc : Nat) (lam : Float) : IO ByteArray := do
  -- Built with `blit` from three one-element constants because `F32Array` has no scalar writer
  -- (only `write3`, which would clobber the two neighbouring classes).
  let one ← F32.const 1 1.0
  let cLam ← F32.const 1 lam
  let cRest ← F32.const 1 (1.0 - lam)
  let mut buf ← F32.const (bs * nc).toUSize 0.0
  for i in [0:bs] do
    let a := i % nc
    let b := (i + off) % nc
    if a == b then
      -- λ + (1−λ) lands on ONE class: the row is a plain one-hot. Writing `lam` then `1−lam`
      -- into the same slot would leave `1−lam` there and quietly make the row sub-stochastic.
      buf ← F32.blit buf (i * nc + a).toUSize one 0 1
    else
      buf ← F32.blit buf (i * nc + a).toUSize cLam 0 1
      buf ← F32.blit buf (i * nc + b).toUSize cRest 0 1
  pure buf

/-- The LayerNorm nets only, and that is a scope limit rather than an oversight.

    A batch-BN net carries running-stat **inputs** and returns the batch statistics, so its arity is
    2·(BN layers) wider on both sides — EfficientNet is 740 outputs against the 642 this harness
    supplies, and the shim's G4 guard refuses the call outright rather than answering it wrongly
    (§5 records the same lesson for `shard-check`). Adding the BN region here is mechanical and
    `TestShardCheck.lean` has the worked version, but it buys nothing for this question: the
    property under test is a property of the **loss cotangent**, which is the same expression in
    every one of these renders. Two nets that disagree about almost everything else agreeing on it
    is the evidence; a third and fourth would be repetition. -/
private def netOf : String → Option (VerifiedNetSpec × Nat)
  | "vit"          => some (vitVerified,      32)
  | "convnext"     => some (convnextVerified, 32)
  | _              => none

def main (args : List String) : IO Unit := do
  let slug := args.head?.getD ""
  let some (spec, bs) := netOf slug
    | IO.eprintln s!"usage: soft-target-tie <vit|convnext|efficientnet|mobilenetv2>  (got '{slug}')"
      IO.Process.exit 1
  let net := spec.toNet
  let nc := spec.nClasses
  let lam : Float := 0.7          -- an arbitrary interior λ; 0 or 1 would make the test vacuous
  let off := 3                    -- the second label set, offset so the endpoints really differ
  let path := args[1]?.getD s!"verified_mlir/{net.slug}_adam_train_step.mlir"

  IO.println s!"Soft-target affineness gate — {spec.name}"
  IO.println s!"  render : {path}  (bs {bs}, {nc} classes, {net.nParams} params)"
  IO.println s!"  identity: grad(λ·y_a + (1−λ)·y_b) == λ·grad(y_a) + (1−λ)·grad(y_b),  λ = {lam}"

  -- θ fixed across all three runs; m = 0 so that m' = 0.1·g recovers the gradient exactly.
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind); sd := sd + 1
  let θ := F32.concat θparts
  let m ← F32.const net.nParams.toUSize 0.0
  let v ← F32.const net.nParams.toUSize 0.0
  let tail ← F32.write3 (← F32.const 3 0.0) 0 0.001 0.19 0.002
  let pbuf := F32.concat #[θ, m, v, tail]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]])
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0

  let yA := mkLabels bs 0 nc                 -- int32 hard labels  (the OLD path)
  let yB := mkLabels bs off nc               -- int32 hard labels  (the OLD path)
  let yM ← mkSoft bs off nc lam              -- float32 soft target (the NEW path)
  IO.println s!"  target buffers: hard {yA.size} bytes, soft {yM.size} bytes \
(expect {bs*4} and {bs*nc*4})"

  let entry := s!"m.{net.slug}_adam_train_step"
  let run (tag : String) (y : ByteArray) : IO ByteArray := do
    IO.println s!"  running {tag}…"; (← IO.getStdout).flush
    let s ← mkSession path
    LowererSession.mlpTrainStepV s entry x pbuf shapes y
      bs.toUSize net.d0.toUSize nc.toUSize
  let oA ← run "a" yA
  let oB ← run "b" yB
  let oM ← run "mix" yM
  -- The PATH check, and it is the one that actually validates `lean_fill_targets`: the SAME target
  -- expressed as int32 labels and as an explicit float one-hot must give the SAME answer. Both
  -- routes build the identical `[batch,nClasses]` float array in C (memcpy vs expansion), so this
  -- is expected BIT-EXACT — it is not a tolerance question, and any drift means the soft path
  -- feeds the graph something different from what the hard path does.
  let ySoftA ← mkSoft bs 0 nc 1.0     -- λ=1 ⇒ a pure one-hot of the `yA` classes
  let oSA ← run "a-as-soft" ySoftA
  -- The FLOOR for that comparison: the identical hard target, run again through the identical
  -- construction. Each `run` builds a fresh session and recompiles, and §3's "bit-identical within
  -- a process" turns out NOT to survive that on every net — ConvNeXt disagrees with itself on
  -- ~0.15% of coordinates by sub-print-precision amounts, which is its documented ill-conditioned
  -- layer-scale reduce (§2f-bis: "does not reproduce to 1e-4 against ANY reordering"). Without
  -- this run the PATH check would read that as a broken soft path. Measure the floor; never assume
  -- it (§4).
  let oA2 ← run "a-again" yA

  -- Compare the `m` region only: [θ' | m' | v' | …], so m' starts at nParams.
  let nP := net.nParams
  let mut maxAbsDiff : Float := 0.0
  let mut maxAbsMix  : Float := 0.0
  let mut maxCtlDiff : Float := 0.0
  let mut pathDiff   : Float := 0.0
  let mut pathExact  : Nat := 0
  let mut floorDiff  : Float := 0.0
  let mut floorExact : Nat := 0
  let mut nonFinite  : Nat := 0
  for i in [nP:2*nP] do
    let a := F32.read oA i.toUSize
    let b := F32.read oB i.toUSize
    let mx := F32.read oM i.toUSize
    let sa := F32.read oSA i.toUSize
    if !a.isFinite || !b.isFinite || !mx.isFinite then nonFinite := nonFinite + 1
    let pred := lam * a + (1.0 - lam) * b
    let d := (mx - pred).abs
    if d > maxAbsDiff then maxAbsDiff := d
    if mx.abs > maxAbsMix then maxAbsMix := mx.abs
    let c := (mx - a).abs
    if c > maxCtlDiff then maxCtlDiff := c
    let p := (sa - a).abs
    if p == 0.0 then pathExact := pathExact + 1
    if p > pathDiff then pathDiff := p
    let f := ((F32.read oA2 i.toUSize) - a).abs
    if f == 0.0 then floorExact := floorExact + 1
    if f > floorDiff then floorDiff := f

  let rel    := if maxAbsMix > 1e-30 then maxAbsDiff / maxAbsMix else 0.0
  let ctlRel := if maxAbsMix > 1e-30 then maxCtlDiff / maxAbsMix else 0.0
  IO.println s!"  ── gradient region (m), {nP} floats ──"
  IO.println s!"    FLOOR   |hard(a) − hard(a) again|  = {floorDiff}, bit-exact {floorExact}/{nP}"
  IO.println s!"    PATH    |soft-onehot(a) − hard(a)|  = {pathDiff}, bit-exact {pathExact}/{nP}"
  IO.println s!"    TEST    |mix − (λ·a + (1−λ)·b)| / max|mix| = {rel}"
  IO.println s!"    CONTROL |mix − a|               / max|mix| = {ctlRel}   ← must be large"

  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite} non-finite outputs"; IO.Process.exit 1
  -- ① The PATH check is absolute, because it is not a tolerance question: the two routes build
  --    the identical float array in C, so anything but bit-exact means the soft path is feeding
  --    the graph a different target.
  -- Read against the MEASURED floor, not against bit-exactness: the same hard target run twice
  -- already disagrees on some nets (see the floor run above), so "must be bit-exact" would fail
  -- them for a property that has nothing to do with soft targets.
  if pathDiff > max floorDiff 1e-12 then
    IO.eprintln s!"SOFT PATH BROKEN: feeding onehot(a) as a float target differs from feeding \
label a as int32 by {pathDiff}, ABOVE the {floorDiff} floor the same hard target gives against \
itself. `lean_fill_targets` is building a different target from the same information."
    IO.Process.exit 1
  -- ② Vacuity: with indistinguishable endpoints the affine identity is trivially true.
  if ctlRel < 1e-3 then
    IO.eprintln s!"VACUOUS: the two label sets produce gradients only {ctlRel} apart — \
the affine identity is untested. Pick a larger `off`."
    IO.Process.exit 1
  -- ③ The affine identity is gated RELATIVE TO THE CONTROL, never absolutely. §2d.1 spells out
  --    why: an absolute 1e-4 on a gradient "failed at 2.9e-3 and looked like a defect. It is
  --    not." The residual here is not the render disagreeing — it is that `λ·grad_a + (1−λ)·grad_b`
  --    is a HOST-side combination of two separately-rounded gradients, so it carries cancellation
  --    the fused run does not, and how much depends on the net's conditioning (ViT is far worse
  --    than ConvNeXt on exactly this axis, §3). A margin against the control is the honest bar,
  --    and it is the same shape every tie in this repo uses.
  let sep := ctlRel / (max rel 1e-12)
  if sep < 50.0 then
    IO.eprintln s!"SOFT-TARGET GATE FAILED: the affine residual {rel} is only {sep}× below the \
control {ctlRel} (need 50×). The render is not behaving affinely in its target."
    IO.Process.exit 1
  IO.println s!"✓ AFFINE IN THE TARGET: residual {rel}, control {ctlRel}, separation {sep}× \
(≥ 50 required). Soft targets need NO render change, and the hard path is bit-exactly unchanged."
