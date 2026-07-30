import LeanMlir.VerifiedNets

/-! # ViT data-parallel gate — the collective's semantics, on a duplicated batch

The ViT peer of `tests/TestCifar8DpCheck.lean`. `verified_mlir/vit_adamdp_train_step.mlir` inserts
one `all_reduce(add)/N` per parameter between the certified gradient and the certified AdamW triple
(handoff §2b-quater's pattern). That collective is a **trusted carve-out** — emitted text, outside
every faithfulness theorem — so it needs its own numeric check.

**The exact identity used here.** Give both replicas the **same** 32 examples. Each computes the
same gradient `g`, so `all_reduce(add)/2` returns `(g + g)/2 = g` — the mean is an identity on a
duplicated batch. The data-parallel step must therefore reproduce the **single-device** step on that
batch, output for output, and ViT has **no BatchNorm**, so nothing else couples the replicas and
this holds exactly rather than approximately. (R34 could not do this: batch BN makes N×b ≠ 1×(N·b)
by design, §10.3b, which is why its collective is gated on cifar8 instead.)

Two failure modes it separates, both of which have actually happened in this repo:

* **collective missing** → the shim's replica-count guard refuses the call before any numbers.
* **collective present but wrong** (sum not mean) → every gradient is 2× and `m` moves by ~1, five
  orders above the gate. §2b-quater verified exactly that by breaking the divisor.

```
lake build vit-dp-check
unset HIP_VISIBLE_DEVICES
MIOPEN_DEBUG_CONV_GEMM=0 PJRT_REPLICAS=2 .lake/build/bin/vit-dp-check
```

⚠ **`MIOPEN_DEBUG_CONV_GEMM=0` is REQUIRED on this box.** Without it every ViT graph with a
backward dies at *execution* — a fused interior-dilated pad+conv selects MIOpen's no-workspace
`GemmFwdRest` solver, whose `MIOpenIm2d2Col.cpp` fails to build under HIPRTC (it uses the OpenCL
builtin `get_global_id`). Diagnosis and a 20-line JAX reproducer:
`upstream-issues/2026-06-jax-rocm-miopen-im2col-hiprtc/README.md`. That is why this gate sat written
but unrun from 2026-07-28 to 2026-07-30.

⚠ **This gate reports BIT-EXACT, so it MUST be run against a control** — a tie that is bit-exact
everywhere is indistinguishable from a harness comparing a buffer with itself (§4). Pass a broken
render as `argv[1]`; the sum-not-mean control is built by flipping every collective's divisor:

```
sed -E 's/^(    %arn[A-Za-z0-9_]+ = stablehlo\.constant dense<)2\.0(>)/\11.0\2/' \
  verified_mlir/vit_adamdp_train_step.mlir > /tmp/vit_dp_sum.mlir     # 200 divisors 2.0 -> 1.0
MIOPEN_DEBUG_CONV_GEMM=0 PJRT_REPLICAS=2 .lake/build/bin/vit-dp-check /tmp/vit_dp_sum.mlir
```

*The `argv[1]` path did not exist until 2026-07-30 — the gate hardcoded both artifacts, so its
first green run could not be falsified. It was added for exactly that reason.*

Needs TWO GPUs and the XLA backend (collectives do not exist on the IREE path). No `.vmfb` cache
hazard here despite `mkSession` taking a path: on the XLA branch it compiles in-process and never
writes one.
-/

private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0
  | 2 => F32.const n.toUSize 0.0
  | _ =>
    let fanIn := if dims.size == 4 then dims[1]! * dims[2]! * dims[3]! else dims[0]!
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))

def main (args : List String) : IO Unit := do
  let net := vitVerified.toNet
  let bs := 32                                   -- the baked per-replica batch
  let replicas := 2
  -- argv[1] overrides the DP render, so a deliberately broken one (sum-not-mean) can be run
  -- through the identical harness. Without this the bit-exact PASS above is unfalsifiable (§4).
  let dpPath := args.head?.getD "verified_mlir/vit_adamdp_train_step.mlir"
  IO.println "ViT data-parallel gate — duplicated batch"
  IO.println s!"  single : verified_mlir/vit_adam_train_step.mlir   (bs {bs})"
  IO.println s!"  DP     : {dpPath} ({replicas} replicas, \
global {bs * replicas} = the same {bs} examples twice)"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), no BN, \
backend {← IreeSession.backendName}"

  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  let m ← F32.heInit 4242 net.nParams.toUSize 0.02
  let v ← F32.scaleShift (← F32.heInit 8484 net.nParams.toUSize 0.01) 1.0 0.05
  let tail ← F32.const 3 0.0
  let tail ← F32.write3 tail 0 0.001 0.19 0.002
  let pbuf := F32.concat #[θ, m, v, tail]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]])
  let x1 ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let x2 := F32.concat #[x1, x1]                 -- the SAME batch on both replicas
  let mut y1 : ByteArray := .empty
  for i in [0:bs] do
    y1 := y1.push (UInt8.ofNat (i % net.nClasses)); y1 := y1.push 0
    y1 := y1.push 0; y1 := y1.push 0
  let y2 := y1 ++ y1

  IO.println "  running single-device…"; (← IO.getStdout).flush
  let s1 ← mkSession "verified_mlir/vit_adam_train_step.mlir" ".lake/build/vit_dp_a.vmfb"
  let o1 ← IreeSession.mlpTrainStepV s1 "m.vit_adam_train_step" x1 pbuf shapes y1
             bs.toUSize net.d0.toUSize net.nClasses.toUSize
  IO.println "  running data-parallel…"; (← IO.getStdout).flush
  let s2 ← mkSession dpPath ".lake/build/vit_dp_b.vmfb"
  let o2 ← IreeSession.mlpTrainStepVDP s2 "m.vit_adamdp_train_step" x2 pbuf shapes y2
             (bs * replicas).toUSize net.d0.toUSize net.nClasses.toUSize replicas.toUSize

  if o1.size != o2.size then
    IO.eprintln s!"SIZE MISMATCH: {o1.size} vs {o2.size}"; IO.Process.exit 1
  let n := o1.size / 4
  let nP := net.nParams
  let regions : List (String × Nat × Nat) :=
    [("theta", 0, nP), ("m", nP, 2*nP), ("v", 2*nP, 3*nP), ("loss/bc", 3*nP, n)]
  let mut gradRel : Float := 0.0
  let mut nonFinite : Nat := 0
  let mut moved : Nat := 0
  IO.println "  ── per region ──"
  for (nm, lo, hi) in regions do
    let mut ra : Float := 0.0
    let mut rm : Float := 0.0
    let mut exact : Nat := 0
    for i in [lo:hi] do
      let a := F32.read o1 i.toUSize
      let b := F32.read o2 i.toUSize
      if !a.isFinite || !b.isFinite then nonFinite := nonFinite + 1
      if max a.abs b.abs > 1e-12 then moved := moved + 1
      let d := (a - b).abs
      if d == 0.0 then exact := exact + 1
      if d > ra then ra := d
      if a.abs > rm then rm := a.abs
    let nr := if rm > 1e-30 then ra / rm else 0.0
    if nm == "m" then gradRel := nr
    IO.println s!"    {nm}: max|a-b| = {ra}, max|a| = {rm}, norm-rel = {nr}, \
bit-exact {exact}/{hi-lo}"

  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite} non-finite outputs"; IO.Process.exit 1
  if moved * 10 < n then
    IO.eprintln "DEGENERATE: too few non-zero outputs — the check proves little"; IO.Process.exit 1
  -- Gate the GRADIENT (`m`), never θ: Adam's update is scale-free, so a near-zero-gradient
  -- parameter flips sign on a 1-ULP difference and θ lands at ~1e-4 whether or not anything is
  -- wrong (§3). §2b-quater measured this directly — a 2× gradient error moved θ by 2.7e-4 and `m`
  -- by 0.96.
  if gradRel > 1e-4 then
    IO.eprintln s!"DP CHECK FAILED: gradient (m) norm-rel {gradRel} > 1e-4. On a duplicated batch \
all_reduce(add)/2 is the identity, so the data-parallel step must reproduce the single-device one."
    IO.Process.exit 1
  IO.println s!"✓ DP step reproduces the single-device step on a duplicated batch: gradient \
norm-rel {gradRel} ≤ 1e-4, over all {n} returned floats"
