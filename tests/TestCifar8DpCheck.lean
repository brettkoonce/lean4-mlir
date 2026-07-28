import LeanMlir.VerifiedNets

/-! # cifar8 data-parallel EXACT check — 2×128 + all_reduce vs 1×256

`planning/xla_pjrt_handoff.md` §2b-quater. This is the gate that pins the **semantics** of the
collective carve-out, and it is the only one that can: cifar8 has **no BatchNorm**, so the loss is a
plain mean over examples and the batch decomposition is an identity,

    (1/2)·[ (1/128)·Σ_A + (1/128)·Σ_B ]  =  (1/256)·Σ_{A∪B}

so a correct data-parallel step must reproduce the single-device step at the global batch to fp
rounding. **ResNet-34 cannot be checked this way** — BN normalises per replica, so there
N×b ≠ 1×(N·b) *by design* (§10.3b) and no exact tie exists. Hence the ladder: pin the collective
here, then use it at R34 scale where only structural checks are available.

Both renders come from `LeanMlir/Proofs/Codegen/CnnRender.lean`, i.e. the same
`cifar8AdamTrainStepFaithfulV` at `replicas := 1` (B=256) and `replicas := 2` (B=128). 1/256 and
1/128 are both exact in binary32, so the loss scaling contributes no rounding of its own.

    unset HIP_VISIBLE_DEVICES
    lake build cifar8-dp-check
    PJRT_REPLICAS=2 .lake/build/bin/cifar8-dp-check

Needs **two** GPUs and the XLA/PJRT backend (`mlpTrainStepVDP` is XLA-only; the IREE build raises
rather than silently running single-device).

**Gate on `m`, not `θ`** (§3): Adam's update is scale-free, so a near-zero-gradient coordinate flips
sign on a 1-ULP difference and moves a full ±lr. `m' = β₁·m + (1−β₁)·g` off a shared `m`, so a
disagreement there IS a gradient disagreement, scaled by (1−β₁).

`%loss` is deliberately **excluded** from the gate: the DP render computes it per replica (÷128) and
the result is read back from replica 0, so it is the mean over that replica's own half — a different
quantity from the 256-example mean, and correctly so. It is reported for information.
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
  let (single, dp) := match args with
    | a :: b :: _ => (a, b)
    | [a]         => (a, "verified_mlir/cifar8_adamdp_train_step.mlir")
    | []          => ("verified_mlir/cifar8_adam256_train_step.mlir",
                      "verified_mlir/cifar8_adamdp_train_step.mlir")
  let net := cifar8Verified.toNet
  let gbs := 256                      -- GLOBAL batch: 1×256 vs 2×128
  let replicas := 2
  IO.println "cifar8 data-parallel exact check (no BN ⇒ the decomposition is an identity)"
  IO.println s!"  A = {single}   (1 device × {gbs})"
  IO.println s!"  B = {dp}   ({replicas} replicas × {gbs / replicas})"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), backend {← IreeSession.backendName}"

  -- ── identical inputs for both ─────────────────────────────────────────────────────────────
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  let m ← F32.heInit 4242 net.nParams.toUSize 0.02
  -- v strictly positive: a zero v makes √v̂ + ε ≈ ε everywhere and would hide a second-moment error.
  let v ← F32.scaleShift (← F32.heInit 8484 net.nParams.toUSize 0.01) 1.0 0.05
  let tail ← F32.const 3 0.0
  let tail ← F32.write3 tail 0 0.001 0.19 0.002    -- lr, 1−β₁ᵗ, 1−β₂ᵗ
  let pbuf := F32.concat #[θ, m, v, tail]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]])
  -- One global batch of 256. The shim hands replica 0 the first 128 and replica 1 the rest, so the
  -- union the DP run sees is exactly what the single-device run sees.
  let x ← F32.heInit 555 (gbs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:gbs] do
    y := y.push (UInt8.ofNat (i % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0

  IO.println "  running A (single device, batch 256)…"; (← IO.getStdout).flush
  let sessA ← mkSession single ".lake/build/c8_dp_single.vmfb"
  let oa ← IreeSession.mlpTrainStepV sessA "m.cifar8_adam_train_step" x pbuf shapes y
    gbs.toUSize net.d0.toUSize net.nClasses.toUSize
  IO.println s!"  running B (data-parallel, {replicas} × {gbs / replicas})…"; (← IO.getStdout).flush
  let sessB ← mkSession dp ".lake/build/c8_dp_dp.vmfb"
  let ob ← IreeSession.mlpTrainStepVDP sessB "m.cifar8_adamdp_train_step" x pbuf shapes y
    gbs.toUSize net.d0.toUSize net.nClasses.toUSize replicas.toUSize

  if oa.size != ob.size then
    IO.eprintln s!"SIZE MISMATCH: {oa.size} vs {ob.size} bytes"; IO.Process.exit 1
  let n := oa.size / 4
  let nP := net.nParams

  -- ── per region ────────────────────────────────────────────────────────────────────────────
  let regions : List (String × Nat × Nat) :=
    [("theta", 0, nP), ("m", nP, 2*nP), ("v", 2*nP, 3*nP), ("loss/bc", 3*nP, n)]
  let mut worstGated : Float := 0.0
  let mut mNormRel : Float := 0.0
  let mut nonFinite : Nat := 0
  let mut moved : Nat := 0
  IO.println "  ── per region (norm-relative = max|a−b| / max|a|) ──"
  for (nm, lo, hi) in regions do
    let mut ra : Float := 0.0
    let mut rm : Float := 0.0
    let mut exact : Nat := 0
    for i in [lo:hi] do
      let a := F32.read oa i.toUSize
      let b := F32.read ob i.toUSize
      if !a.isFinite || !b.isFinite then nonFinite := nonFinite + 1
      if max a.abs b.abs > 1e-12 then moved := moved + 1
      let d := (a - b).abs
      if d == 0.0 then exact := exact + 1
      if d > ra then ra := d
      if a.abs > rm then rm := a.abs
    let normRel := if rm > 1e-30 then ra / rm else 0.0
    if nm == "m" then mNormRel := normRel
    if nm != "loss/bc" && normRel > worstGated then worstGated := normRel
    let note := if nm == "loss/bc" then "   (NOT gated — replica-0 half-batch mean)" else ""
    IO.println s!"    {nm}: max|a-b| = {ra}, max|a| = {rm}, norm-rel = {normRel}, \
bit-exact {exact}/{hi-lo}{note}"

  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite}/{n} non-finite outputs"; IO.Process.exit 1
  if moved * 10 < n then
    IO.eprintln s!"DEGENERATE: only {moved}/{n} outputs are non-zero — the check proves little"
    IO.Process.exit 1

  -- The decomposition is EXACT in exact arithmetic, so the only budget is fp reassociation:
  -- summing 256 contributions on one device vs 128+128 then averaging. §2c measured 1.015e-06 for
  -- this comparison with the hand-written emitter; 1e-4 leaves two orders of headroom while still
  -- failing hard on a wrong collective (no collective at all would show ~0.5, a sum-not-mean ~1.0).
  if worstGated > 1e-4 then
    IO.eprintln s!"DP CHECK FAILED: worst norm-relative diff {worstGated} > 1e-4"
    IO.eprintln "  a missing collective reads as ~0.5 here; an unaveraged sum as ~1.0"
    IO.Process.exit 1
  IO.println s!"✓ data-parallel step MATCHES the single-device step at the global batch"
  IO.println s!"  gradient (m) norm-rel = {mNormRel}; worst gated region = {worstGated} ≤ 1e-4"
