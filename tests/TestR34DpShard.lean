import LeanMlir.VerifiedNets

/-! # `r34-dp-shard` — do R34/ImageNet's four replicas actually SEE DIFFERENT DATA?

    lake build r34-dp-shard
    unset HIP_VISIBLE_DEVICES
    PJRT_PLUGIN=... PJRT_REPLICAS=4 .lake/build/bin/r34-dp-shard

**Why this file exists.** `tests/TestShardCheck.lean` says in its own docstring that *"R34 is
absent on purpose … it has no `adamdp` peer at this batch to pair with"*, and there is no
`resnet34-dp-check` either. So R34 — the net carrying the 30-epoch ImageNet run — is the ONE net
whose data-parallel path has no gate at all. Everything asserted about its sharding has been read
off a source comment.

That became load-bearing on 2026-08-04: the verified run sat at **10.57% top-1 at epoch 5** where
the reference curve reads **42.79%**, a ~4× deficit that held almost constant (6.0/4.9/4.6/4.3/4.05)
across the five warmup epochs — where both runs are recipe-identical by construction. A constant
factor is the signature of a systematic cause, and "each replica sees a quarter of the data it
should" is exactly such a cause.

**The construction, and why it is not `shard-check`'s.** The standard identity
`DP([x0|..|x3]) == mean(single(x0),..,single(x3))` needs a single-device render at the SAME
per-replica batch, and `resnet34in` has only `mom256` (bs 256), not a bs-64 peer — the
"single-GPU bs64 render, which does not exist" of `xla_pjrt_handoff.md`. Rendering one is a new
artifact; this asks the narrower question directly and needs only the DP artifact that is already
committed and already training:

  * **A** = `[s0 | s1 | s2 | s3]`
  * **B** = `[s0 | s1' | s2' | s3']` — replicas 1..3 get DIFFERENT pixels and labels, replica 0 is
    byte-identical to A.
  * **C** = `[s0' | s1 | s2 | s3]` — only replica 0 differs.

  | | requires | reads |
  |---|---|---|
  | **TEST** | `A ≠ B` | replicas 1..3 genuinely feed the update |
  | **CONTROL** | `A ≠ C` | the harness can see a change at all |

`A == B` while `A ≠ C` is the *positive identification* of replication: the collective would be
averaging four copies of replica 0's gradient, every step, and three quarters of every batch would
be discarded. ⚠ This does NOT verify the shard OFFSETS (a permutation of the four shards passes),
which is what the full `shard-check` identity would add. It answers only the question the accuracy
deficit poses.

Compared on the **`m` region** `[nP, 2nP)`: the momentum buffer is seeded to 0, so after one step
`m' = g + wd·θ`, and θ is identical across A/B/C — every difference in `m'` is a difference in the
gradient, with no optimizer state to launder it.
-/

private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0          -- γ = 1
  | 2 => F32.const n.toUSize 0.0          -- β / bias = 0
  | _ =>
    let fanOut := if dims.size == 4 then dims[0]! * dims[2]! * dims[3]! else dims[0]!
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanOut.toFloat))

/-- Labels for one shard: class `(i + off) % nClasses`, in the driver's 4-byte records. -/
private def mkLabels (bs off nc : Nat) : ByteArray := Id.run do
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat ((i + off) % nc)); y := y.push 0; y := y.push 0; y := y.push 0
  y

def main : IO Unit := do
  let net      := resnet34ImagenetVerified.toNet
  let bs       := ((← IO.getEnv "SHARD_BS").bind (·.toNat?)).getD 64
  let replicas := ((← IO.getEnv "PJRT_REPLICAS").bind (·.toNat?)).getD 4
  let variant  := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "momdp64"
  let dpPath   := s!"verified_mlir/{net.slug}_{variant}_train_step.mlir"
  let fn       := s!"m.{net.slug}_{variant}_train_step"

  IO.println s!"{net.name} — DP SHARD DISCRIMINATION ({replicas} replicas x bs {bs})"
  IO.println s!"  render : {dpPath}"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), {net.bnChannels.size} BN layers, \
backend {← IreeSession.backendName}"

  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0

  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind); sd := sd + 1
  let θ := F32.concat θparts
  -- m = 0 is LOAD-BEARING: it makes m' = g + wd·θ exactly linear in the gradient.
  let m ← F32.const net.nParams.toUSize 0.0
  let v ← F32.const net.nParams.toUSize 0.0
  let tail ← F32.const 3 0.0
  let tail ← F32.write3 tail 0 0.1 0.9 0.999          -- lr, and the two unused bc slots
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let pbuf := F32.concat #[θ, m, v, tail, bnIn]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]] ++ bnStatShapes)

  -- Four shards with genuinely different pixels AND labels, plus primed alternates.
  let mkShard (seed off : Nat) : IO (ByteArray × ByteArray) := do
    pure (← F32.heInit seed.toUSize (bs * net.d0).toUSize 1.0, mkLabels bs off net.nClasses)
  let mut s : Array (ByteArray × ByteArray) := #[]
  for i in [0:replicas] do s := s.push (← mkShard (555 + 444 * i) (5 * i))
  let s0'  ← mkShard 90001 700
  let mut sPrime : Array (ByteArray × ByteArray) := #[]
  for i in [1:replicas] do sPrime := sPrime.push (← mkShard (70000 + 313 * i) (11 * i + 3))

  let glue (parts : Array (ByteArray × ByteArray)) : ByteArray × ByteArray := Id.run do
    let mut xs : Array ByteArray := #[]
    let mut y : ByteArray := .empty
    for (a, b) in parts do xs := xs.push a; y := y ++ b
    (F32.concat xs, y)

  let (xA, yA) := glue s
  let (xB, yB) := glue (#[s[0]!] ++ sPrime)                       -- replicas 1..3 changed
  let (xC, yC) := glue (#[s0'] ++ s.extract 1 replicas)           -- only replica 0 changed

  -- One session, three invokes: same executable, same params, only the batch differs.
  for p in [s!".lake/build/{net.slug}_dpshard.vmfb",
            s!".lake/build/{net.slug}_dpshard_{((← IO.getEnv "IREE_BACKEND").getD "cuda")}.vmfb"] do
    if ← System.FilePath.pathExists p then IO.FS.removeFile p
  let sess ← mkSession dpPath
  let run (x y : ByteArray) : IO ByteArray :=
    IreeSession.mlpTrainStepVDP sess fn x pbuf shapes y
      (bs * replicas).toUSize net.d0.toUSize net.nClasses.toUSize replicas.toUSize
  IO.println "  invoking A…"; (← IO.getStdout).flush
  let oA ← run xA yA
  IO.println "  invoking B (replicas 1..3 differ)…"; (← IO.getStdout).flush
  let oB ← run xB yB
  IO.println "  invoking C (only replica 0 differs)…"; (← IO.getStdout).flush
  let oC ← run xC yC

  -- Compare the momentum region: every difference there is a gradient difference.
  let nP := net.nParams
  -- ⚠ Compare θ' — [0, nP) — NOT the `m` region. This is HEAVY-BALL, not Adam: the single
  -- velocity buffer lands in the THIRD region and region 2 is an untouched passthrough, so a
  -- comparison over [nP, 2nP) is identically zero for every input and silently proves nothing.
  -- The first draft of this file did exactly that and its own CONTROL caught it.
  let cmp (o1 o2 : ByteArray) : Nat × Float := Id.run do
    let mut diff := 0
    let mut rel : Float := 0.0
    let mut den : Float := 0.0
    for i in [0:nP] do
      let a := F32.read o1 i.toUSize
      let b := F32.read o2 i.toUSize
      if a != b then diff := diff + 1
      rel := max rel (Float.abs (a - b)); den := max den (Float.abs a)
    (diff, if den > 0.0 then rel / den else 0.0)
  let (dAB, rAB) := cmp oA oB
  let (dAC, rAC) := cmp oA oC

  -- ▶ DIAGNOSTIC: is `x` reaching the graph at all, and is the m-region index right?
  IO.println ""
  IO.println s!"  [diag] out size {oA.size} bytes = {oA.size / 4} floats; nP = {nP}; \
d0 = {net.d0}; expect >= {3 * nP + 3 + nBnStats}"
  IO.println s!"  [diag] loss slot ({3 * nP}): A={F32.read oA (3*nP).toUSize} \
B={F32.read oB (3*nP).toUSize} C={F32.read oC (3*nP).toUSize}"
  let dθ : Nat := Id.run do
    let mut d : Nat := 0
    for i in [0:nP] do
      if F32.read oA i.toUSize != F32.read oC i.toUSize then d := d + 1
    pure d
  IO.println s!"  [diag] θ' region A vs C: {dθ}/{nP} differ"
  IO.println ""
  IO.println s!"  TEST     A vs B (replicas 1..3 differ): {dAB}/{nP} floats differ, rel {rAB}"
  IO.println s!"  CONTROL  A vs C (only replica 0 differs): {dAC}/{nP} floats differ, rel {rAC}"
  IO.println ""
  if dAC == 0 then
    IO.println "⛔ CONTROL DID NOT FIRE — changing replica 0's data changed nothing, so this"
    IO.println "   harness proves NOTHING about the shard. Fix the harness before reading TEST."
    IO.Process.exit 1
  else if dAB == 0 then
    IO.println "⛔⛔ REPLICATION CONFIRMED. Replicas 1..3 do not affect the update: the four"
    IO.println "    replicas are being handed the SAME rows, so every step trains on bs"
    IO.println s!"    {bs} and discards {100 * (replicas - 1) / replicas}% of each batch."
    IO.Process.exit 1
  else
    IO.println "✅ SHARDING IS REAL — replicas 1..3 feed the update, and the control fires."
    IO.println "   ⚠ This does not verify shard OFFSETS; a permutation would also pass."
