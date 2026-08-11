import LeanMlir.VerifiedNets

/-! # The **sharding** gate, for every net that has a data-parallel render

    lake build shard-check
    unset HIP_VISIBLE_DEVICES
    PJRT_REPLICAS=2 .lake/build/bin/shard-check <convnext|efficientnet|mobilenetv2> [<dpPath>]

Generalised from `tests/TestConvNeXtShardCheck.lean` on 2026-07-30 (handoff §5's "still open"
item). It exists because the `*-dp-check` gates have a hole: the duplicated-batch identity hands
both replicas the **same** rows, so a shard-offset bug — replica 1 reading `[0,b)` instead of
`[b,2b)` — leaves the two halves identical and those gates still pass **bit-exact**. They establish
*"the collective averages correctly"*, **not** *"the replicas saw different data"*.

This closes that. Give the replicas **different** data and check against two single-device steps:

    DP( [xA | xB] )  must equal  mean( single(xA), single(xB) )

A wrong shard offset gives `DP([xA|xB]) = single(xA)`, and the CONTROL below asserts that is a
*different* number — so the gate is sensitive to exactly the failure the duplicated-batch one is
blind to.

**Why it gates `m` and not `θ'`.** AdamW's update is NONLINEAR in the gradient (`m̂/(√v̂+ε)`), so
`(θ'_A + θ'_B)/2 ≠ θ'(ḡ)` and comparing θ' would be meaningless. `adamMNextF` is
`m' = β₁·m + (1−β₁)·g`, so feeding **m = 0** makes `m' = 0.1·g` — exactly linear in the gradient,
hence exactly averagable. `v' = 0.001·g²` is quadratic, so it is not compared. Same conclusion as
§3's "gate the gradient, never θ", reached from the other direction.

**Why this works on the BATCH-BN nets too**, which is the whole reason it generalises: each replica
normalises over its own `b` rows, and the single-device reference runs are at that same `b`, so
`single(xA)` reproduces replica 0's arithmetic exactly. That is *not* true of the cifar8 split
identity (`1×2b` vs `2×b`), which genuinely needs no BN — see §10.3b. The `bnstat` region is
deliberately **not** compared here: under an asymmetric batch the DP render returns replica 0's
statistics, which equal `single(xA)`'s and not the A/B mean, so it would need its own comparand and
adds nothing — `*-dp-check` already pins the forward bit-exactly.

**One harness, three nets, because the only per-net facts are the spec and the batch.** Everything
else derives from `net.slug`: `verified_mlir/<slug>_adam{,dp}_train_step.mlir` and the matching
`m.<slug>_adam{,dp}_train_step` entry names. Writing this per net would be the double-writer
disease one level down, in code — and the generic harness is gated by having to **reproduce
`convnext-shard-check`'s committed numbers** (TEST 8.2e-8 / CONTROL 0.137).

Needs TWO GPUs and the XLA backend (collectives do not exist on the IREE path).
-/

private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0
  | 2 => F32.const n.toUSize 0.0
  | _ =>
    let fanIn := if dims.size == 4 then dims[1]! * dims[2]! * dims[3]! else dims[0]!
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))

/-- Labels for one shard: class `(i + off) % nClasses`, packed as the driver's 4-byte records. -/
private def mkLabels (bs off nc : Nat) : ByteArray := Id.run do
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat ((i + off) % nc)); y := y.push 0; y := y.push 0; y := y.push 0
  y

/-- The nets with a DP render that this construction applies to. R34 is absent on purpose: its
    `_adam_train_step` is bs32 batch-BN like these, so the *test* would work, but its DP evidence
    is tracked separately (§2b-quater) and it has no `adamdp` peer at this batch to pair with. -/
private def netOf : String → Option (VerifiedNetSpec × Nat)
  | "convnext"     => some (convnextVerified,     32)
  -- the 1000-class ImageNet twins (§2p)
  | "convnextin"        => some (convnextImagenetVerified, 32)
  | "efficientnetin"       => some (efficientnetImagenetVerified, 64)
  | "mobilenetv2in"       => some (mobilenetv2ImagenetVerified, 64)
  | "efficientnet" => some (efficientnetVerified, 32)
  | "mobilenetv2"  => some (mobilenetv2Verified,  32)
  | _              => none

def main (args : List String) : IO Unit := do
  let slug := args.head?.getD ""
  let some (spec, bs) := netOf slug
    | do IO.eprintln s!"usage: shard-check <convnext|efficientnet|mobilenetv2> [<dpPath>]\n\
got: '{slug}'"; IO.Process.exit 1
  let net := spec.toNet
  -- $SHARD_REPLICAS generalises the construction: `ds.shard`-style, N shards each with genuinely
  -- different data, checked against the mean of N single-device steps. The identity
  -- `DP([x0|..|xN-1]) == mean(single(x0),..,single(xN-1))` holds for any N — 2 was never special,
  -- it was just the only DP render that existed when this was written. The ImageNet renders are
  -- 4-replica, which is what forced the generalisation.
  let replicas := ((← IO.getEnv "SHARD_REPLICAS").bind (·.toNat?)).getD 2
  -- $SHARD_VARIANT names the single-device and DP variants when they are not the bare
  -- `adam`/`adamdp` (EfficientNet's ImageNet pair is `adam64`/`adamdp64`, since `enetAdamVariant`
  -- appends the per-device batch).
  let vSg := (← IO.getEnv "SHARD_VARIANT").getD "adam"
  let vDp := (← IO.getEnv "SHARD_VARIANT_DP").getD "adamdp"
  -- argv[2] overrides the DP render so a deliberately broken one can be run through the identical
  -- harness (e.g. the sum-not-mean control: flip every `%arn… dense<2.0>` to 1.0).
  let dpPath := args[1]?.getD s!"verified_mlir/{net.slug}_{vDp}_train_step.mlir"
  let sgPath := s!"verified_mlir/{net.slug}_{vSg}_train_step.mlir"
  IO.println s!"{net.name} SHARDING gate — asymmetric batch"
  IO.println s!"  DP( [x0|..|x{replicas-1}] )  ==  mean of {replicas} single-device steps   ({replicas} replicas x bs {bs})"
  IO.println s!"  single   : {sgPath}"
  IO.println s!"  DP render: {dpPath}"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), {net.bnChannels.size} BN layers, \
backend {← LowererSession.backendName}"

  -- The BATCH-BN nets carry running-stat inputs AND return the batch statistics, so their arity is
  -- 2·(BN layers) wider on both sides than the `[θ|m|v|lr,bc1,bc2]` core. Omitting them is not a
  -- silent wrong answer — the shim's G4 guard refuses the call ("returns 887 outputs, caller
  -- supplied 789 destinations"), which is how this was caught. ConvNeXt is LayerNorm, so
  -- `bnChannels` is empty and every line below degrades to a no-op there: its numbers are
  -- unchanged by this generalisation, which is itself a check on it.
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  -- m = 0 is LOAD-BEARING: it makes m' = (1-β₁)·g exactly linear in the gradient, which is the
  -- whole reason this comparison is a known answer rather than an approximation.
  let m ← F32.const net.nParams.toUSize 0.0
  let v ← F32.scaleShift (← F32.heInit 8484 net.nParams.toUSize 0.01) 1.0 0.05
  let tail ← F32.const 3 0.0
  let tail ← F32.write3 tail 0 0.001 0.19 0.002
  let pbuf ← if nBnStats == 0 then pure (F32.concat #[θ, m, v, tail]) else do
      let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
      pure (F32.concat #[θ, m, v, tail, bnIn])
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]] ++ bnStatShapes)
  -- Two genuinely DIFFERENT shards — different pixels AND different labels, so a replica reading
  -- the wrong rows cannot coincidentally agree.
  let mut xs : Array ByteArray := #[]
  let mut ys : Array ByteArray := #[]
  for i in [0:replicas] do
    xs := xs.push (← F32.heInit (555 + 444 * i).toUSize (bs * net.d0).toUSize 1.0)
    ys := ys.push (mkLabels bs (5 * i) net.nClasses)
  let xAB := F32.concat xs
  let mut yAB : ByteArray := .empty
  for y in ys do yAB := yAB ++ y

  -- Delete both the bare and the backend-scoped .vmfb first (§4): `compileVmfb` keys its cache on
  -- the OUTPUT path plus an mtime, never the source, so a re-run with a different candidate under
  -- the same tag would silently reuse the first one and report a perfect match.
  for tag in [s!"{net.slug}_shard_a", s!"{net.slug}_shard_b"] do
    for p in [s!".lake/build/{tag}.vmfb",
              s!".lake/build/{tag}_{((← IO.getEnv "IREE_BACKEND").getD "cuda")}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
  let s1 ← mkSession sgPath
  let mut outs : Array ByteArray := #[]
  for i in [0:replicas] do
    IO.println s!"  single-device on shard {i}…"; (← IO.getStdout).flush
    outs := outs.push (← LowererSession.mlpTrainStepV s1 s!"m.{net.slug}_{vSg}_train_step"
      xs[i]! pbuf shapes ys[i]! bs.toUSize net.d0.toUSize net.nClasses.toUSize)
  let oA := outs[0]!
  IO.println s!"  data-parallel on the {replicas}-way shard…"; (← IO.getStdout).flush
  let s2 ← mkSession dpPath
  let oD ← LowererSession.mlpTrainStepVDP s2 s!"m.{net.slug}_{vDp}_train_step" xAB pbuf shapes yAB
             (bs * replicas).toUSize net.d0.toUSize net.nClasses.toUSize replicas.toUSize

  let nP := net.nParams
  for (o, i) in outs.zipIdx do
    if o.size != oD.size then
      IO.eprintln s!"SIZE MISMATCH: shard {i} gives {o.size}, DP gives {oD.size}"; IO.Process.exit 1
  -- `m` occupies [nP, 2nP) in the `[θ | m | v | loss/bc | bnstat]` layout every one of these nets
  -- returns. m' = 0.1·g, so mean(mA, mB) is 0.1·mean(gA, gB) = what a correct shard must produce.
  let mut relMean : Float := 0.0            -- TEST:    DP vs mean(A,B)
  let mut relA    : Float := 0.0            -- CONTROL: DP vs A alone (a broken shard)
  let mut denom   : Float := 0.0
  let mut nonFinite : Nat := 0
  let mut moved : Nat := 0
  let invN := 1.0 / replicas.toFloat
  for i in [nP:2*nP] do
    let a := F32.read oA i.toUSize
    let d := F32.read oD i.toUSize
    -- the mean over ALL N shards, not just two
    let mut acc : Float := 0.0
    let mut fin := true
    for o in outs do
      let vi := F32.read o i.toUSize
      if !vi.isFinite then fin := false
      acc := acc + vi
    if !fin || !d.isFinite then nonFinite := nonFinite + 1
    if d.abs > 1e-12 then moved := moved + 1
    let avg := invN * acc
    let e1 := (d - avg).abs
    let e2 := (d - a).abs
    if e1 > relMean then relMean := e1
    if e2 > relA then relA := e2
    if avg.abs > denom then denom := avg.abs
  let nrMean := if denom > 1e-30 then relMean / denom else 0.0
  let nrA    := if denom > 1e-30 then relA / denom else 0.0
  IO.println s!"  ── gradient proxy m' = 0.1·g, over {nP} coords ──"
  IO.println s!"    TEST    |DP − mean of {replicas}| / max|mean| = {nrMean} ({nrMean * 1e9} e-9)"
  IO.println s!"    CONTROL |DP − shard0|     / max|mean| = {nrA}  ← a broken shard would land HERE"

  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite} non-finite outputs"; IO.Process.exit 1
  if moved * 10 < nP then
    IO.eprintln "DEGENERATE: too few non-zero gradients — the check proves little"
    IO.Process.exit 1
  -- The control must be LARGE, or the two shards were not actually different and the test is
  -- vacuous — the same trap §2d.1 hit with a reversed-batch control that produced no difference.
  if nrA < 1e-3 then
    IO.eprintln s!"VACUOUS: shard A and the A/B mean agree to {nrA} — the two shards are not \
distinguishable, so passing the TEST would prove nothing. Use more different data."
    IO.Process.exit 1
  if nrMean > 1e-4 then
    IO.eprintln s!"SHARD CHECK FAILED: DP does not reproduce mean(single(A), single(B)) — \
norm-rel {nrMean} > 1e-4. Either the replicas are not receiving disjoint rows, or the collective \
is not averaging them."
    IO.Process.exit 1
  IO.println s!"✓ sharding CONFIRMED for {net.slug}: DP([A|B]) = mean(single(A), single(B)) to \
{nrMean}, while DP vs A alone is {nrA} — {nrA / (max nrMean 1e-12)}x apart, so the replicas \
provably saw DIFFERENT data"
