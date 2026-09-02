import LeanMlir.VerifiedNets

/-! # EfficientNet-B0 data-parallel step-time bench — 1 GPU vs 2 GPUs on XLA/PJRT

`planning/xla_pjrt_handoff.md` §2e-bis. The first DP number for this net came from two short
end-to-end training runs (wall clock minus the shim's reported compile time) and was quoted as
"~1.2×, one run each" — good enough to state a direction, not good enough to quote. This is the
§2b-bis methodology applied to the data-parallel path.

**Method.** Both executables are compiled in ONE process and their steps are **interleaved**
(A,B,A,B,…) so clock and thermal drift hit both equally; the reported statistic is the **min**,
which is the robust one for a bench (noise only ever adds time). Inputs are synthetic and built
once in memory, so the data loader is out of the measurement entirely — §3's standing warning is
that a data-bound regime makes any throughput ratio meaningless, and the R34 "within 1.04× of JAX"
figure was measured in exactly that trap.

* **A** = `efficientnet_adam_train_step.mlir`, 1 replica, bs 32.
* **B** = `efficientnet_adamdp_train_step.mlir`, 2 replicas, global 64 (32 per replica).

The batches differ by construction, so **ms/IMAGE is the figure** and ms/step is not comparable.
Perfect scaling would be an unchanged ms/step at twice the images, i.e. 2.00× on ms/image.

**What limits it, and why the ratio is the wrong thing to blame.** Parameters are host-resident
(§2c), so *every* step pushes the whole packed `[θ|m|v]` — 213 params, 4,020,358 floats, ~48.2 MB (262 / 4,041,366 before §2m) —
to *every* replica. Compute halves across two devices while that transfer doubles. This bench
therefore also reports the implied per-step transfer share, because the interesting question is not
"is it 2×" (it cannot be) but "how much of the step is already transfer", which is the number that
says whether device-resident parameters (§2d.3) is worth doing.

    gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
    lake build efficientnet-dp-bench
    unset HIP_VISIBLE_DEVICES && PJRT_REPLICAS=2 .lake/build/bin/efficientnet-dp-bench [rounds]

Needs TWO GPUs and the XLA backend — collectives do not exist on the IREE path. One process holding
both a 1-replica and a 2-replica executable is fine: the replica count is per-GRAPH, not per-process
(§2c), which `efficientnet-dp-check` already relies on.
-/

private def opCount (path : String) : IO Nat := do
  let s ← IO.FS.readFile path
  pure ((s.splitOn "stablehlo.").length - 1)

/-- Read the entry-point name and the baked batch off a render: `func.func @NAME(%x: tensor<BxD…`.
    Both are properties of the artifact, so reading them beats assuming them — it lets one bench
    compare renders at different `B` (the bs128 pair, §2e-quater) with no new argument, and it
    cannot silently invoke the wrong entry. -/
private def entryAndBatch (path : String) : IO (String × Nat) := do
  let s ← IO.FS.readFile path
  let some i := (s.splitOn "func.func @")[1]? | throw (IO.userError s!"{path}: no `func.func @`")
  let name := (i.takeWhile (· != '(')).trimAscii.toString
  let some rest := (i.splitOn "%x: tensor<")[1]? | throw (IO.userError s!"{path}: no %x operand")
  let bstr := (rest.takeWhile (· != 'x')).toString
  let some b := bstr.toNat? | throw (IO.userError s!"{path}: batch {bstr} is not a number")
  pure (name, b)

/-- min / median / mean of a sample, in ms. -/
private def stats (xs : Array Float) : Float × Float × Float :=
  let s := xs.qsort (· < ·)
  let n := s.size
  let mean := (s.foldl (· + ·) 0.0) / n.toFloat
  (s[0]!, s[n / 2]!, mean)

def main (args : List String) : IO Unit := do
  let dfltA := "verified_mlir/efficientnet_adam_train_step.mlir"
  let dfltB := "verified_mlir/efficientnet_adamdp_train_step.mlir"
  let (pathA, pathB, rounds) := match args with
    | a :: b :: r :: _ => (a, b, r.toNat!)
    | [a, b]           => (a, b, 20)
    | [a]              => (dfltA, dfltB, a.toNat!)
    | []               => (dfltA, dfltB, 20)
  let warmup := 3
  let net := efficientnetVerified.toNet
  let replicas := 2
  -- Entry name and per-replica batch are read OFF each artifact rather than assumed, so this bench
  -- also compares renders at different `B` (§2e-quater's bs128 pair) with no new argument.
  let (fnA, bsA) ← entryAndBatch pathA
  let (fnB, bsB) ← entryAndBatch pathB
  -- Side B is data-parallel only if it actually contains collectives; otherwise this is a plain
  -- single-device A-vs-B comparison (e.g. bs32 vs bs128 on one GPU) and the invoke must not ask
  -- for replicas it was not compiled for — the shim refuses that outright.
  let bIsDP := ((← IO.FS.readFile pathB).splitOn "stablehlo.all_reduce").length > 1
  let repB := if bIsDP then replicas else 1
  let imgPerStepA := bsA
  let imgPerStepB := bsB * repB
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  let opsA ← opCount pathA
  let opsB ← opCount pathB
  IO.println "EfficientNet-B0 AdamW step-time bench"
  IO.println s!"  A = {pathA}  (@{fnA}, 1 replica, bs {bsA} = {imgPerStepA} img/step, {opsA} ops)"
  IO.println s!"  B = {pathB}  (@{fnB}, {repB} replica(s), bs {bsB} = {imgPerStepB} img/step, \
{opsB} ops{if bIsDP then s!" — the extra {opsB - opsA} are the {net.specs.size} collectives" else ""})"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats = {net.nParams * 3 * 4 / 1048576} \
MiB of [θ|m|v] per replica per step), {net.bnChannels.size} BN layers, \
backend {← LowererSession.backendName}"
  IO.println s!"  {warmup} warmup + {rounds} timed rounds, interleaved A,B,A,B…, SYNTHETIC inputs \
(no data loader — see §3)"
  (← IO.getStdout).flush

  -- ── inputs: byte-identical to tests/TestEfficientNetDpCheck.lean ──────────────────────────
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParamHeFanIn sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  let m ← F32.heInit 4242 net.nParams.toUSize 0.02
  let v ← F32.scaleShift (← F32.heInit 8484 net.nParams.toUSize 0.01) 1.0 0.05
  let tail ← F32.const 3 0.0
  let tail ← F32.write3 tail 0 0.001 0.19 0.002    -- lr, 1−β₁ᵗ, 1−β₂ᵗ
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let pbuf := F32.concat #[θ, m, v, tail, bnIn]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]] ++ bnStatShapes)
  let mkXY (b : Nat) : IO (ByteArray × ByteArray) := do
    let x ← F32.heInit 555 (b * net.d0).toUSize 1.0
    let mut y : ByteArray := .empty
    for i in [0:b] do
      y := y.push (UInt8.ofNat (i % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0
    pure (x, y)
  let (x1, y1) ← mkXY imgPerStepA
  let (x2, y2) ← mkXY imgPerStepB

  IO.println "  compiling A (1 replica)…"; (← IO.getStdout).flush
  let ca0 ← IO.monoMsNow
  let sessA ← mkSession pathA
  let compA := (← IO.monoMsNow) - ca0
  IO.println s!"  compiling B ({repB} replica(s))…"; (← IO.getStdout).flush
  let cb0 ← IO.monoMsNow
  let sessB ← mkSession pathB
  let compB := (← IO.monoMsNow) - cb0
  IO.println s!"  compile: A {compA} ms, B {compB} ms"

  let stepA : IO Nat := do
    let t0 ← IO.monoMsNow
    let out ← LowererSession.mlpTrainStepV sessA s!"m.{fnA}" x1 pbuf shapes y1
      imgPerStepA.toUSize net.d0.toUSize net.nClasses.toUSize
    let t1 ← IO.monoMsNow
    if out.size == 0 then throw (IO.userError "empty step output (A)")
    pure (t1 - t0)
  let stepB : IO Nat := do
    let t0 ← IO.monoMsNow
    let out ← if bIsDP then
        LowererSession.mlpTrainStepVDP sessB s!"m.{fnB}" x2 pbuf shapes y2
          imgPerStepB.toUSize net.d0.toUSize net.nClasses.toUSize repB.toUSize
      else
        LowererSession.mlpTrainStepV sessB s!"m.{fnB}" x2 pbuf shapes y2
          imgPerStepB.toUSize net.d0.toUSize net.nClasses.toUSize
    let t1 ← IO.monoMsNow
    if out.size == 0 then throw (IO.userError "empty step output (B)")
    pure (t1 - t0)

  IO.println "  warmup…"; (← IO.getStdout).flush
  for _ in [0:warmup] do
    let _ ← stepA
    let _ ← stepB

  let mut msA : Array Float := #[]
  let mut msB : Array Float := #[]
  IO.println "  timing…"; (← IO.getStdout).flush
  for _ in [0:rounds] do
    msA := msA.push (← stepA).toFloat
    msB := msB.push (← stepB).toFloat

  let (minA, medA, meanA) := stats msA
  let (minB, medB, meanB) := stats msB
  let imgA := minA / imgPerStepA.toFloat
  let imgB := minB / imgPerStepB.toFloat
  IO.println "  ── ms/step ──"
  IO.println s!"    A ({imgPerStepA} img): min {minA}  median {medA}  mean {meanA}"
  IO.println s!"    B ({imgPerStepB} img): min {minB}  median {medB}  mean {meanB}"
  IO.println "  ── ms/image (min) ──"
  IO.println s!"    A = {imgA}   B = {imgB}"
  IO.println "  ── verdict ──"
  IO.println s!"    THROUGHPUT = {(imgA / imgB)}× for B over A"
  if bIsDP then
    IO.println s!"    on {repB} GPUs, perfect scaling = {repB.toFloat}× ⇒ parallel efficiency \
{(100.0 * imgA / imgB / repB.toFloat)}%"
  -- The right decomposition, and NOT the textbook Amdahl one. Each replica of B does exactly the
  -- work A does — 32 images — so with zero DP overhead B's ms/STEP would EQUAL A's, at twice the
  -- images. The whole cost of going data-parallel is therefore the step-time excess `T_B − T_A`,
  -- not some serial fraction of A. (An Amdahl form written as `2·T_B/T_A − 1` assumes B halves A's
  -- work, which is the wrong model here and returns >100% nonsense.)
  -- The right decomposition for the DP comparison, and NOT the textbook Amdahl one. Each replica
  -- of B does exactly the work A does, so with zero DP overhead B's ms/STEP would EQUAL A's at
  -- twice the images. The whole cost of going data-parallel is the step-time excess `T_B − T_A`.
  -- (An Amdahl form `2·T_B/T_A − 1` assumes B halves A's work, which is the wrong model here and
  -- returns >100% nonsense.) It only means anything when the two sides share a per-replica batch.
  if bIsDP && bsA == bsB then
    let ovhMs := minB - minA
    -- What has to cross the bus that a single-device step does not: the all-reduce exchanges the
    -- whole gradient (one float per parameter) between replicas, and the host pushes [θ|m|v] to
    -- the SECOND replica too, because parameters are host-resident (§2c).
    let gradMiB := net.nParams.toFloat * 4.0 / 1048576.0
    let paramMiB := net.nParams.toFloat * 3.0 * 4.0 / 1048576.0
    IO.println s!"    DP overhead = {ovhMs} ms/step = {(100.0 * ovhMs / minA)}% of the 1-GPU step"
    IO.println s!"    extra bytes it covers ≈ {gradMiB} MiB all-reduced + {paramMiB} MiB [θ|m|v] \
pushed to replica {repB}  ⇒ ≈ {((gradMiB + paramMiB) / 1024.0 / (ovhMs / 1000.0))} GiB/s effective"
    IO.println "    Parameters are host-resident (§2c), so the [θ|m|v] push is per-replica and grows"
    IO.println "    with the replica count. Device-resident parameters (§2d.3) removes it."
  IO.println "    NOTE this is the ON-GPU number: inputs are synthetic and in-process. An"
  IO.println "    end-to-end training run also pays the data loader, which does NOT shrink with"
  IO.println "    replicas — measured 1.67× end-to-end at bs32 against 1.75× here, the gap being"
  IO.println "    a ~4.3 s/epoch host cost (shuffle + crop + hflip, no prefetch). Measure that as a"
  IO.println "    MARGINAL epoch (T3−T1)/2: wall-clock-minus-compile still carries the one-time"
  IO.println "    ~7.45 GiB dataset load and gave a wrong 1.43× twice over (§2e-ter)."
