import LeanMlir.VerifiedNets

/-! # ConvNeXt-T **sharding** gate — the asymmetric-batch known answer

The companion to `tests/TestConvNeXtDpCheck.lean`, and it exists because that gate has a hole
(handoff §2h-quater): the duplicated-batch identity hands both replicas the **same** rows, so a
shard-offset bug — replica 1 reading `[0,b)` instead of `[b,2b)` — leaves the two halves identical
and the gate still passes bit-exact. It establishes *"the collective averages correctly"*, **not**
*"the replicas saw different data"*.

This one closes that. Give the replicas **different** data and check the answer against two
single-device steps run separately:

    DP( [xA | xB] )  must equal  mean( single(xA), single(xB) )

A wrong shard offset gives `DP([xA|xB]) = single(xA)`, which the control below asserts is a
*different* number — so the gate is sensitive to exactly the failure the duplicated-batch one is
blind to.

**Why it gates `m` and not `θ'`.** AdamW's parameter update is NONLINEAR in the gradient
(`m̂/(√v̂+ε)`), so `(θ'_A + θ'_B)/2 ≠ θ'(ḡ)` and comparing θ' would be meaningless. But
`adamMNextF` is `m' = β₁·m + (1−β₁)·g`, so feeding **m = 0** makes `m' = 0.1·g` — exactly linear
in the gradient, hence exactly averagable. `v' = 0.001·g²` is quadratic and is therefore reported
but NOT gated. This is the same reasoning §3 gives for gating the gradient rather than θ, arrived
at from the other direction.

**Why ConvNeXt can do this at all.** LayerNorm reduces within one example, so a replica's arithmetic
on its own rows is what it would have been anywhere — `single(xA)` reproduces replica 0's work
exactly. On a **batch-BN** net this still holds for THIS test (each replica normalises over its own
b rows either way, and the single-device runs are at that same b), which is why the construction
transfers to efficientnet/mobilenetv2 unchanged — unlike the cifar8 split identity, which genuinely
needs no BN.

    lake build convnext-shard-check
    unset HIP_VISIBLE_DEVICES && PJRT_REPLICAS=2 .lake/build/bin/convnext-shard-check

Needs TWO GPUs and the XLA backend.
-/

/-- Labels for one shard: class `(i + off) % nClasses`, packed as the driver's 4-byte records. -/
private def mkLabels (bs off nc : Nat) : ByteArray := Id.run do
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat ((i + off) % nc)); y := y.push 0; y := y.push 0; y := y.push 0
  y

def main (args : List String) : IO Unit := do
  let net := convnextVerified.toNet
  let bs := 32
  let replicas := 2
  let dpPath := args.head?.getD "verified_mlir/convnext_adamdp_train_step.mlir"
  IO.println "ConvNeXt-T SHARDING gate — asymmetric batch"
  IO.println s!"  DP( [xA | xB] )  ==  mean( single(xA), single(xB) )   ({replicas} replicas x bs {bs})"
  IO.println s!"  DP render: {dpPath}"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), backend {← LowererSession.backendName}"

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
  let pbuf := F32.concat #[θ, m, v, tail]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]])
  -- Two genuinely DIFFERENT shards — different pixels AND different labels, so a replica reading
  -- the wrong rows cannot coincidentally agree.
  let xA ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let xB ← F32.heInit 999 (bs * net.d0).toUSize 1.0
  let yA := mkLabels bs 0 net.nClasses
  let yB := mkLabels bs 5 net.nClasses
  let xAB := F32.concat #[xA, xB]
  let yAB := yA ++ yB

  for tag in ["cnx_shard_a", "cnx_shard_b"] do
    for p in [s!".lake/build/{tag}.vmfb",
              s!".lake/build/{tag}_{((← IO.getEnv "IREE_BACKEND").getD "cuda")}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
  let s1 ← mkSession "verified_mlir/convnext_adam_train_step.mlir"
  IO.println "  single-device on shard A…"; (← IO.getStdout).flush
  let oA ← LowererSession.mlpTrainStepV s1 "m.convnext_adam_train_step" xA pbuf shapes yA
             bs.toUSize net.d0.toUSize net.nClasses.toUSize
  IO.println "  single-device on shard B…"; (← IO.getStdout).flush
  let oB ← LowererSession.mlpTrainStepV s1 "m.convnext_adam_train_step" xB pbuf shapes yB
             bs.toUSize net.d0.toUSize net.nClasses.toUSize
  IO.println "  data-parallel on [A | B]…"; (← IO.getStdout).flush
  let s2 ← mkSession dpPath
  let oD ← LowererSession.mlpTrainStepVDP s2 "m.convnext_adamdp_train_step" xAB pbuf shapes yAB
             (bs * replicas).toUSize net.d0.toUSize net.nClasses.toUSize replicas.toUSize

  let nP := net.nParams
  if oA.size != oD.size || oB.size != oD.size then
    IO.eprintln s!"SIZE MISMATCH: {oA.size} / {oB.size} / {oD.size}"; IO.Process.exit 1
  -- `m` occupies [nP, 2nP). m' = 0.1·g, so mean(mA, mB) is 0.1·mean(gA, gB) = what a correct
  -- shard must produce.
  let mut relMean : Float := 0.0            -- TEST:    DP vs mean(A,B)
  let mut relA    : Float := 0.0            -- CONTROL: DP vs A alone (a broken shard)
  let mut denom   : Float := 0.0
  let mut nonFinite : Nat := 0
  let mut moved : Nat := 0
  for i in [nP:2*nP] do
    let a := F32.read oA i.toUSize
    let b := F32.read oB i.toUSize
    let d := F32.read oD i.toUSize
    if !a.isFinite || !b.isFinite || !d.isFinite then nonFinite := nonFinite + 1
    if d.abs > 1e-12 then moved := moved + 1
    let avg := 0.5 * (a + b)
    let e1 := (d - avg).abs
    let e2 := (d - a).abs
    if e1 > relMean then relMean := e1
    if e2 > relA then relA := e2
    if avg.abs > denom then denom := avg.abs
  let nrMean := if denom > 1e-30 then relMean / denom else 0.0
  let nrA    := if denom > 1e-30 then relA / denom else 0.0
  IO.println s!"  ── gradient proxy m' = 0.1·g, over {nP} coords ──"
  IO.println s!"    TEST    |DP − mean(A,B)| / max|mean| = {nrMean} ({nrMean * 1e9} e-9)"
  IO.println s!"    CONTROL |DP − A|        / max|mean| = {nrA}  ← a broken shard would land HERE"

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
  IO.println s!"✓ sharding CONFIRMED: DP([A|B]) = mean(single(A), single(B)) to {nrMean}, while \
DP vs A alone is {nrA} — {nrA / (max nrMean 1e-12)}x apart, so the replicas provably saw \
DIFFERENT data"
