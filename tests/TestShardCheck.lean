import LeanMlir.Verified.NetsCore
import LeanMlir.Verified.Train

/-! # The **sharding** gate, for every net that has a data-parallel render

    lake build shard-check
    unset CUDA_VISIBLE_DEVICES
    PJRT_REPLICAS=2 .lake/build/bin/shard-check <convnext|vit> [<dpPath>]

⚠ A 4-replica render needs `SHARD_REPLICAS=4` and four GPUs (plus both `SHARD_VARIANT` knobs when
its variants are not the bare `adam`/`adamdp`), e.g. ConvNeXt's ImageNet pair:

    PJRT_REPLICAS=4 SHARD_REPLICAS=4 .lake/build/bin/shard-check convnextin

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

⛔ **It no longer covers the BATCH-BN nets — retired 2026-09-21.** It used to, because each replica
normalised over its own `b` rows and so `single(xA)` reproduced replica 0's arithmetic exactly.
Since `planning/global_bn_verified.md` §3.2–3.4 every BatchNorm net's DP render is SYNCHRONISED:
the replicas all-reduce their BN statistics, so `DP([xA|xB])` is `single_2b([xA|xB])`, and the
identity above is exactly what `<net>-syncbn-check`'s CONTROL now requires to FAIL (measured on
the committed renders: `efficientnet` 0.21, `mobilenetv2` 1.60, against this gate's 1e-4). The
rows `efficientnet`, `mobilenetv2`, `efficientnetin`, `mobilenetv2in` and `mnv4in` are therefore
retired, not loosened. Their replacement is the sync-BN gates' TEST column —
`DP_sync([x₀|…]) = single_Rb([x₀|…])` on different data per replica, which a shard-offset bug
breaks on the statistics — run by `mobilenetv2-syncbn-check`, `efficientnet-syncbn-check` and
`imagenet-syncbn-check <net>`. LayerNorm reduces within an example, so ConvNeXt and ViT keep the
per-replica identity and stay here.

**One harness, three nets, because the only per-net facts are the spec and the batch.** Everything
else derives from `net.slug`: `verified_mlir/<slug>_adam{,dp}_train_step.mlir` and the matching
`m.<slug>_adam{,dp}_train_step` entry names. Writing this per net would be the double-writer
disease one level down, in code — and the generic harness is gated by having to **reproduce
`convnext-shard-check`'s committed numbers** (TEST 8.2e-8 / CONTROL 0.137).

Needs TWO GPUs and the XLA backend (collectives do not exist on the IREE path).
-/

/-- The nets with a DP render that this construction applies to. R34 is absent on purpose: its
    `_adam_train_step` is bs32 batch-BN like these, so the *test* would work, but its DP evidence
    is tracked separately (§2b-quater) and it has no `adamdp` peer at this batch to pair with. -/
private def netOf : String → Option (VerifiedNetSpec × Nat)
  | "convnext"     => some (convnextVerified,     32)
  -- the 1000-class ImageNet twins (§2p)
  | "convnextin"        => some (convnextImagenetVerified, 32)
  -- ⛔ `efficientnet`, `mobilenetv2`, `efficientnetin`, `mobilenetv2in` and `mnv4in` were rows
  -- here until 2026-09-21; see the module docstring and `retired` below.
  -- ViT, added 2026-08-12. It had `tests/TestViTDpCheck.lean` and nothing else, which is the gate
  -- that hands both replicas the SAME rows: `all_reduce(add)/N` is an identity on a duplicated
  -- batch, so that check is structurally blind to a shard-offset bug. This row is what closes it,
  -- by giving the replicas genuinely different data. ⚠ ViT has no BatchNorm, so the identity holds
  -- exactly here rather than approximately (the reason R34 is absent, above).
  | "vit"          => some (vitVerified,          32)
  | _              => none

/-- The batch-BN rows, retired when their DP renders went sync-BN, and the gate that replaced each. -/
private def retired : String → Option String
  | "efficientnet"   => some "efficientnet-syncbn-check"
  | "mobilenetv2"    => some "mobilenetv2-syncbn-check"
  | "efficientnetin" => some "imagenet-syncbn-check efficientnet"
  | "mobilenetv2in"  => some "imagenet-syncbn-check mobilenetv2"
  | "mnv4in"         => some "imagenet-syncbn-check mnv4"
  | _                => none

def main (args : List String) : IO Unit := do
  let slug := args.head?.getD ""
  if let some gate := retired slug then
    IO.eprintln s!"shard-check {slug}: RETIRED 2026-09-21. This net's DP render is synchronised \
BatchNorm, so DP([x0|x1]) = single_2b([x0|x1]), not mean(single(x0), single(x1)) — the identity \
this gate asserts. Run `{gate}` instead."
    IO.Process.exit 2
  let some (spec, bs) := netOf slug
    | do IO.eprintln s!"usage: shard-check <convnext|vit|convnextin> [<dpPath>]\ngot: '{slug}'"
         IO.Process.exit 1
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
