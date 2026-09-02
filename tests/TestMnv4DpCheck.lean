import LeanMlir.VerifiedNets

/-! # MobileNetV4 data-parallel gate — the collective's semantics, on a duplicated batch

The MNv4 peer of `tests/TestMobilenetV2DpCheck.lean`, and the reason
`verified_mlir/mnv4in_adamdp64_train_step.mlir` stops carrying a caveat.

⛔ **WHY THIS FILE EXISTS.** `MobileNetV4RenderB.lean`'s `#eval` block rendered the 4-replica pair
for COSTING only, and said so in three places: *"nothing has tied MNv4's collectives"*, *"do not
train off these"*, *"what would lift the caveat is a DP tie for MNv4's collectives, the way
R34/R50/MNv2/ConvNeXt have one"*. An untied collective artifact looks exactly as trustworthy as a
tied one — it is the same bytes, in the same directory, named the same way — so the render was
deliberately unquotable rather than deliberately absent. This gate and the `mnv4in` row in
`tests/TestShardCheck.lean` are that tie.

**The exact identity, and why BatchNorm does not spoil it.** Give every replica the **same** 64
examples. BN normalises per replica, so each replica's BN group is those same 64 examples and all
four compute *identical* batch statistics; each therefore computes the same gradient `g`, and
`all_reduce(add)/4` returns `(4·g)/4 = g`. The mean is an identity on a duplicated batch, at any
replica count. The data-parallel step must reproduce the **single-device** step, output for output.

The §10.3b caveat that stopped R34 from having this gate is about **splitting** a batch — 4×64 is
genuinely not 1×256 under batch BN, because the shards get different statistics. Duplicating a
batch is the other case: the statistics are the same by construction, so nothing couples the
replicas and the identity is exact rather than approximate.

**This gate pins the forward.** MNv4-Conv-M returns batch statistics for every BN layer, so it has
a forward-only `bnstat` region that must come back **bit-exact**. Every replica saw the same data,
so any difference there is the DP path corrupting the forward, which no gradient tolerance would
have caught.

⚠⚠ **FOUR replicas, not two, and that is forced.** MNv4 renders `adamdp64` at 4 replicas only —
there is no 2-replica peer — so this needs four GPUs. `PJRT_REPLICAS=2` does not degrade to a
2-way run; the shim's replica-count guard refuses the call. Every other `*-dp-check` here defaults
to 2 because its net had a bs32 Imagenette DP render to pair with, and MNv4 has none.

⚠ **It is the ImageNet net, so this is a 1000-class 224² step.** There is no Imagenette-scale MNv4
DP render to gate more cheaply; `mnv4_adam_train_step.mlir` is single-device.

Two failure modes it separates, both of which have actually happened in this repo:

* **collective missing** → the shim's replica-count guard refuses the call before any numbers.
* **collective present but wrong** (sum not mean) → every gradient is 4× and `m` moves by ~1, five
  orders above the gate. Pass the broken render as `argv[1]` to run that control here.

    lake build mnv4-dp-check
    unset HIP_VISIBLE_DEVICES
    PJRT_REPLICAS=4 .lake/build/bin/mnv4-dp-check                  # fp32
    DP_VARIANT=adam64bf16 DP_VARIANT_DP=adamdp64bf16 \
      PJRT_REPLICAS=4 .lake/build/bin/mnv4-dp-check                # the bf16 pair

⭐ **Both precision arms need gating, not just fp32.** `mnv4in_adamdp64bf16` is on disk and is
exactly as untied as its f32 peer was; the precision axis does not get to quietly inherit a tie it
was not given. The `DP_VARIANT` knobs are what make that a re-run rather than a second file.

Needs FOUR GPUs and the XLA backend (collectives do not exist on the IREE path — the IREE shim
refuses a DP entry point outright rather than silently running single-device).
-/

def main (args : List String) : IO Unit := do
  -- Env-selected the way `TestMobilenetV2DpCheck` and `TestShardCheck` are, defaulting to EXACTLY
  -- the configuration the committed result was measured at — so it reproduces with no arguments.
  -- ⚠ Unlike those two the defaults are 4 replicas and the ImageNet spec, because MNv4's only DP
  -- renders are 4-replica and 1000-class. There is nothing cheaper to fall back to.
  let net := mnv4ImagenetVerified.toNet
  let bs := ((← IO.getEnv "DP_BATCH").bind (·.toNat?)).getD 64       -- the BAKED per-replica batch
  let replicas := ((← IO.getEnv "DP_REPLICAS").bind (·.toNat?)).getD 4
  -- `mnv4AdamVariant` appends the per-device batch, so the variant strings carry the 64 —
  -- `adam64`/`adamdp64` rather than the bare `adam`/`adamdp` the Imagenette-scale gates use.
  let vSg := (← IO.getEnv "DP_VARIANT").getD "adam64"
  let vDp := (← IO.getEnv "DP_VARIANT_DP").getD "adamdp64"
  let sgPath := s!"verified_mlir/{net.slug}_{vSg}_train_step.mlir"
  -- The DP render is overridable so a deliberately-broken one can be fed in. That is not a
  -- convenience: a gate nobody has seen go red is an assertion. The control to run is the `%arn`
  -- divisor 4.0 → 1.0, i.e. sum instead of mean.
  let dpPath := args.head?.getD s!"verified_mlir/{net.slug}_{vDp}_train_step.mlir"
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  IO.println "MobileNetV4-Conv-M data-parallel gate — duplicated batch"
  IO.println s!"  single : {sgPath}   (bs {bs})"
  IO.println s!"  DP     : {dpPath} ({replicas} replicas, \
global {bs * replicas} = the same {bs} examples {replicas} times)"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), {net.bnChannels.size} BN layers \
({nBnStats} stat floats), backend {← LowererSession.backendName}"

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
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let pbuf := F32.concat #[θ, m, v, tail, bnIn]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]] ++ bnStatShapes)
  let x1 ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  -- The SAME batch on EVERY replica. `all_reduce(add)/N` over N identical gradients is `(N·g)/N =
  -- g`, the identity at any N, so nothing here depends on the replica count but this concat.
  let x2 := F32.concat (Array.replicate replicas x1)
  let mut y1 : ByteArray := .empty
  for i in [0:bs] do
    y1 := y1.push (UInt8.ofNat (i % net.nClasses)); y1 := y1.push 0
    y1 := y1.push 0; y1 := y1.push 0
  let y2 := (Array.replicate replicas y1).foldl (· ++ ·) ByteArray.empty

  IO.println "  running single-device…"; (← IO.getStdout).flush
  -- Delete first on BOTH sides: `compileVmfb` keys on the OUTPUT path and an mtime, never the
  -- source, so a second run with a different candidate silently reuses the first one's binary
  -- (handoff §4) — which is exactly what running the sum-not-mean control looks like.
  for tag in ["mnv4_dp_a", "mnv4_dp_b"] do
    for p in [s!".lake/build/{tag}.vmfb",
              s!".lake/build/{tag}_{((← IO.getEnv "IREE_BACKEND").getD "cuda")}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
  let s1 ← mkSession sgPath
  let o1 ← LowererSession.mlpTrainStepV s1 s!"m.{net.slug}_{vSg}_train_step" x1 pbuf shapes y1
             bs.toUSize net.d0.toUSize net.nClasses.toUSize
  IO.println "  running data-parallel…"; (← IO.getStdout).flush
  let s2 ← mkSession dpPath
  let o2 ← LowererSession.mlpTrainStepVDP s2 s!"m.{net.slug}_{vDp}_train_step" x2 pbuf shapes y2
             (bs * replicas).toUSize net.d0.toUSize net.nClasses.toUSize replicas.toUSize

  if o1.size != o2.size then
    IO.eprintln s!"SIZE MISMATCH: {o1.size} vs {o2.size}"; IO.Process.exit 1
  let n := o1.size / 4
  let nP := net.nParams
  let regions : List (String × Nat × Nat) :=
    [("theta", 0, nP), ("m", nP, 2*nP), ("v", 2*nP, 3*nP),
     ("loss/bc", 3*nP, 3*nP+3), ("bnstat", 3*nP+3, n)]
  let mut gradRel : Float := 0.0
  let mut fwdExact := true
  let mut fwdRel : Float := 0.0
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
    if nm == "bnstat" then
      fwdRel := nr
      if exact != hi - lo then fwdExact := false
    -- Also in nano-units. `Float.toString` gives six decimals, so a genuine 3e-8 disagreement
    -- prints as "0.000000" and reads as bit-exact when it is not. The bit-exact COUNT is what
    -- distinguishes "bit-exact" from "prints as zero".
    IO.println s!"    {nm}: max|a-b| = {ra} ({ra * 1e9} e-9), max|a| = {rm}, \
norm-rel = {nr} ({nr * 1e9} e-9), bit-exact {exact}/{hi-lo}"

  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite} non-finite outputs"; IO.Process.exit 1
  if moved * 10 < n then
    IO.eprintln "DEGENERATE: too few non-zero outputs — the check proves little"; IO.Process.exit 1
  -- Every replica saw the SAME 64 examples, so their BN groups are identical and the batch
  -- statistics cannot legitimately differ. Anything here is the DP path corrupting the forward.
  if !fwdExact then
    IO.eprintln s!"DP CHECK FAILED: the forward differs — `bnstat` is not bit-exact \
(norm-rel {fwdRel}). Every replica was given the same batch, so their BN statistics are identical \
by construction; a difference here is the data-parallel path corrupting the forward."
    IO.Process.exit 1
  -- Gate the GRADIENT (`m`), never θ: Adam's update is scale-free, so a near-zero-gradient
  -- parameter flips sign on a 1-ULP difference and θ lands at ~1e-4 whether or not anything is
  -- wrong (§3).
  if gradRel > 1e-4 then
    IO.eprintln s!"DP CHECK FAILED: gradient (m) norm-rel {gradRel} > 1e-4. On a duplicated batch \
all_reduce(add)/N is the identity, so the data-parallel step must reproduce the single-device one."
    IO.Process.exit 1
  IO.println s!"✓ DP step reproduces the single-device step on a duplicated batch: forward \
BIT-EXACT (bnstat, {net.bnChannels.size} BN layers), gradient norm-rel {gradRel} ≤ 1e-4, \
over all {n} returned floats"
