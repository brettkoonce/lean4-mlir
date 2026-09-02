import LeanMlir.VerifiedNets

/-! # `@resnet34_adam_train_step` step-time bench — hand-written vs `pretty(provenGraph)`

`planning/xla_pjrt_handoff.md` §2b-bis. The batched render (then `…_b.mlir`, now the committed
`verified_mlir/resnet34_adam_train_step.mlir`) is **1.68× the ops** of the hand-written render it
replaced (10014 vs 5971): `pretty` has no CSE, the batched
backward ops are self-contained recomputes (`bnBatchF`, `bnBatchBack`, `bnGammaGradB` each rebuild
x̂ from the saved BN input, so `rsqrt` is 108 = 36 × 3 where the hand-written render saves `%{p}xh`
once), and the `[B,c·h·w] ↔ [B,c,h,w]` round-trips add ~621 reshapes.

The claim to be tested is that **XLA's own CSE collapses most of that** — identical subgraphs on
identical inputs — so the emitted op count is a codegen artifact and not a runtime cost. That was an
assumption; this measures it. Do not quote a step time for the batched render without running this.

**Method.** Both artifacts are compiled in ONE process and their steps are **interleaved**
(A,B,A,B,…) so clock/thermal drift hits both equally, and the reported statistic is the **min**,
which is the robust one for a bench (noise only ever adds time). Inputs are byte-identical to
`tests/TestResnet34AdamTie.lean` so the two harnesses measure the same two executables.

**What the ratio does and does not mean.** Each step round-trips the whole packed `[θ|m|v]` buffer
(~272 MB each way) over PCIe because parameters are still host-resident (§2c). That cost is
identical for both renders, so the honest comparison is the **absolute delta** `T_B − T_A`, which is
the compute difference in ms; the *ratio* of totals is diluted by the shared transfer and will
understate a real compute regression. Both are reported.

    lake build resnet34-adam-bench
    .lake/build/bin/resnet34-adam-bench [refRender.mlir] [newRender.mlir] [rounds]

Since the swap the hand-written render is retired, so the no-argument form benches the committed
artifact against itself. To reproduce the original comparison, recover the retired render:

    git show b856deb:verified_mlir/resnet34_adam_train_step.mlir > /tmp/retired.mlir
    .lake/build/bin/resnet34-adam-bench /tmp/retired.mlir \
      verified_mlir/resnet34_adam_train_step.mlir 20

To answer the CSE question directly rather than by wall clock, dump the post-optimisation HLO and
count what survived:

    XLA_FLAGS="--xla_dump_to=/tmp/hlo --xla_dump_hlo_pass_re=.*" .lake/build/bin/resnet34-adam-bench
-/

/-- Emitted `stablehlo.*` op count of a render. Counted from the file rather than hardcoded so the
    verdict stays correct when the artifacts change — or when the two paths are passed in the other
    order, which is the control run for compile-order effects. -/
private def opCount (path : String) : IO Nat := do
  let s ← IO.FS.readFile path
  pure ((s.splitOn "stablehlo.").length - 1)

/-- Read the entry-point name and the baked batch off a render: `func.func @NAME(%x: tensor<BxD…`.

    Both are properties of the artifact, so reading them beats assuming them — it lets one bench
    compare renders at *different* `B` (§2d.1), and it removes the chance of invoking the wrong
    entry (the shim refuses an entry mismatch, but failing here is a clearer message). -/
private def entryAndBatch (path : String) : IO (String × Nat) := do
  let s ← IO.FS.readFile path
  let some i := (s.splitOn "func.func @")[1]? | throw (IO.userError s!"{path}: no `func.func @`")
  -- `.toString` on both: `takeWhile`/`trim` yield `String.Slice` in this toolchain.
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
  let dfltA := "verified_mlir/resnet34_adam_train_step.mlir"
  let dfltB := "verified_mlir/resnet34_adam_train_step.mlir"
  let (pathA, pathB, rounds) := match args with
    | a :: b :: r :: _ => (a, b, r.toNat!)
    | [a, b]           => (a, b, 20)
    | [a]              => (a, dfltB, 20)
    | []               => (dfltA, dfltB, 20)
  let warmup := 3
  let net := resnet34Verified.toNet
  -- Entry name and batch are read OFF each artifact rather than assumed, so this bench also works
  -- across renders at different `B` (§2d.1's bs32-vs-bs256 measurement) without a new argument —
  -- and cannot silently invoke the wrong entry point, which the shim would refuse anyway.
  let (fnA, bsA) ← entryAndBatch pathA
  let (fnB, bsB) ← entryAndBatch pathB
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  let opsA ← opCount pathA
  let opsB ← opCount pathB
  IO.println s!"resnet34 AdamW step-time bench"
  IO.println s!"  A (first)  = {pathA}  (@{fnA}, bs {bsA}, {opsA} stablehlo ops)"
  IO.println s!"  B (second) = {pathB}  (@{fnB}, bs {bsB}, {opsB} stablehlo ops)"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), {net.bnChannels.size} BN layers, \
backend {← LowererSession.backendName}"
  IO.println s!"  {warmup} warmup + {rounds} timed rounds, interleaved A,B,A,B…"

  -- ── inputs: byte-identical to tests/TestResnet34AdamTie.lean ──────────────────────────────
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind)
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
  let (xA, yA) ← mkXY bsA
  let (xB, yB) ← mkXY bsB

  -- ── compile both (timed — B is 1.68× the ops, so this is a dev-loop cost worth naming) ────
  IO.println "  compiling A…"; (← IO.getStdout).flush
  let ca0 ← IO.monoMsNow
  let sessA ← mkSession pathA
  let compA := (← IO.monoMsNow) - ca0
  IO.println "  compiling B…"; (← IO.getStdout).flush
  let cb0 ← IO.monoMsNow
  let sessB ← mkSession pathB
  let compB := (← IO.monoMsNow) - cb0
  IO.println s!"  compile: A {compA} ms, B {compB} ms  ({(compB.toFloat / compA.toFloat)}×)"

  let step (sess : LowererSession) (fn : String) (x y : ByteArray) (b : Nat) : IO Nat := do
    let t0 ← IO.monoMsNow
    let out ← LowererSession.mlpTrainStepV sess s!"m.{fn}" x pbuf shapes y
      b.toUSize net.d0.toUSize net.nClasses.toUSize
    let t1 ← IO.monoMsNow
    -- touch the result so nothing can be elided, and guard against a silently empty return
    if out.size == 0 then throw (IO.userError "empty step output")
    pure (t1 - t0)
  let stepA : IO Nat := step sessA fnA xA yA bsA
  let stepB : IO Nat := step sessB fnB xB yB bsB

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
  IO.println "  ── ms/step ──"
  IO.println s!"    A (bs {bsA}, {opsA} ops): min {minA}  median {medA}  mean {meanA}"
  IO.println s!"    B (bs {bsB}, {opsB} ops): min {minB}  median {medB}  mean {meanB}"
  IO.println "  ── ms/image (min) ──"
  let imgA := minA / bsA.toFloat
  let imgB := minB / bsB.toFloat
  IO.println s!"    A = {imgA}   B = {imgB}"
  IO.println s!"  ── verdict ──"
  IO.println s!"    op ratio (emitted text)  = {(opsB.toFloat / opsA.toFloat)}×"
  IO.println s!"    step ratio (min)         = {(minB / minA)}×"
  if bsA == bsB then
    IO.println s!"    step delta (min)         = {(minB - minA)} ms"
    IO.println s!"    NOTE: both steps pay the same ~272 MB host↔device round-trip for [θ|m|v] \
(§2c), so the DELTA is the compute difference; the ratio is diluted by that shared transfer."
  else
    IO.println s!"    THROUGHPUT (ms/img)      = {(imgA / imgB)}× for B over A"
    IO.println s!"    NOTE: the batches differ ({bsA} vs {bsB}), so ms/step is not comparable — \
ms/IMAGE is the figure. Each step still pays the same ~272 MB host↔device round-trip for [θ|m|v] \
(§2c) regardless of batch, which is most of why the larger batch wins: the transfer is amortised \
over more images, not made faster."
