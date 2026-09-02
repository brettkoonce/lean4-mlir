import LeanMlir.VerifiedNets

/-! # MobileNetV2 data-parallel gate — the collective's semantics, on a duplicated batch

The mnv2 peer of `tests/TestEfficientNetDpCheck.lean`, and gated by the same **exact** identity
rather than a tolerance argument. `verified_mlir/mobilenetv2_adamdp_train_step.mlir` inserts one
`all_reduce(add)/N` per parameter between the certified gradient and the certified AdamW triple
(handoff §2b-quater's pattern). That collective is a **trusted carve-out** — emitted text, outside
every faithfulness theorem — so it needs its own numeric check.

**The exact identity used here, and why BatchNorm does not spoil it.** Give both replicas the
**same** 32 examples. BN normalises per replica, so each replica's BN group is those same 32
examples and both compute *identical* batch statistics; each therefore computes the same gradient
`g`, and `all_reduce(add)/2` returns `(g + g)/2 = g`. The mean is an identity on a duplicated batch.
The data-parallel step must reproduce the **single-device** step on that batch, output for output.

The §10.3b caveat that stopped R34 from having this gate is about **splitting** a batch — 2×32 is
genuinely not 1×64 under batch BN, because the two halves get different statistics. Duplicating a
batch is the other case: the statistics are the same by construction, so nothing couples the
replicas and the identity is exact rather than approximate.

**Like EfficientNet's and unlike ViT's, this gate pins the forward.** mnv2 returns 104 BN batch
statistics (52 layers), so it has a forward-only `bnstat` region. On a duplicated batch that region
must come back **bit-exact** — every replica saw the same data, so any difference there is the DP
path corrupting the forward, which no gradient tolerance would have caught.

Two failure modes it separates, both of which have actually happened in this repo:

* **collective missing** → the shim's replica-count guard refuses the call before any numbers.
* **collective present but wrong** (sum not mean) → every gradient is 2× and `m` moves by ~1, five
  orders above the gate. §2b-quater and §2e-bis each verified exactly that by breaking the divisor;
  pass the broken render as `argv[1]` to run it here.

    lake build mobilenetv2-dp-check
    unset HIP_VISIBLE_DEVICES && PJRT_REPLICAS=2 .lake/build/bin/mobilenetv2-dp-check

Needs TWO GPUs and the XLA backend (collectives do not exist on the IREE path — the IREE shim
refuses a DP entry point outright rather than silently running single-device, which is why
`mobilenetv2-verified-adam` had to exist first, §2h).
-/

def main (args : List String) : IO Unit := do
  -- ▶ Which (net, batch, replica count, variant pair) this drives is env-selected, defaulting to
  -- EXACTLY the Imagenette/AdamW/2-replica configuration this harness was written for — so the
  -- committed result reproduces with no arguments, which is the gate on the generalisation itself
  -- (`TestShardCheck.lean` took the same route when it went N-replica).
  --
  -- It exists because the RMSProp DP render (`recipe_gaps.md` v1.2) is rendered ONLY at the
  -- ImageNet shape, `mobilenetv2in` B=64 × 4 replicas, and there is no bs32 `rmsdp` peer to pair with.
  -- ⚠ `shard-check` cannot stand in for this one: its known answer is
  -- `DP([A|B]) = mean(single(A), single(B))`, which needs the gated slot to be LINEAR in the
  -- gradient — true of AdamW's `m` at `m = 0` (`m' = (1−β₁)·g`) and **false of RMSProp's buffer**
  -- (`b' = μ·b + gw/√(ρ·s + (1−ρ)·gw² + ε)`). The duplicated-batch identity used here is
  -- optimizer-AGNOSTIC — both sides receive the identical gradient, so whatever nonlinear tail
  -- consumes it must agree — which is why it is the construction that transfers.
  let netSel := (← IO.getEnv "DP_NET").getD "imagenette"
  let (net, bsDefault, repDefault) := match netSel with
    | "imagenet" => (mobilenetv2ImagenetVerified.toNet, 64, 4)
    | _          => (mobilenetv2Verified.toNet, 32, 2)
  let bs := ((← IO.getEnv "DP_BATCH").bind (·.toNat?)).getD bsDefault      -- the BAKED per-replica batch
  let replicas := ((← IO.getEnv "DP_REPLICAS").bind (·.toNat?)).getD repDefault
  let vSg := (← IO.getEnv "DP_VARIANT").getD "adam"
  let vDp := (← IO.getEnv "DP_VARIANT_DP").getD "adamdp"
  let sgPath := s!"verified_mlir/{net.slug}_{vSg}_train_step.mlir"
  -- The DP render is overridable so a deliberately-broken one can be fed in. That is not a
  -- convenience: a gate nobody has seen go red is an assertion. §2b-quater's control — the `%arn`
  -- divisor 2.0 → 1.0, i.e. sum instead of mean — is the one to run.
  let dpPath := args.head?.getD s!"verified_mlir/{net.slug}_{vDp}_train_step.mlir"
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  IO.println "MobileNetV2 data-parallel gate — duplicated batch"
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
  -- The SAME batch on EVERY replica — `replicas` copies, not two. `all_reduce(add)/N` over N
  -- identical gradients is `(N·g)/N = g`, the identity at any N, so the construction does not
  -- depend on the replica count; only this concatenation did.
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
  for tag in ["mnv2_dp_a", "mnv2_dp_b"] do
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
    -- prints as "0.000000" and reads as bit-exact when it is not — EfficientNet's DP gate lands at
    -- exactly that (3.3e-8 on `m`), because the DP module is a different HLO program and XLA is
    -- free to order the backward reductions differently even though the collective itself is exact
    -- ((g+g)/2 = g holds to the bit in binary FP). Report the count too; that is what distinguishes
    -- "bit-exact" from "prints as zero".
    IO.println s!"    {nm}: max|a-b| = {ra} ({ra * 1e9} e-9), max|a| = {rm}, \
norm-rel = {nr} ({nr * 1e9} e-9), bit-exact {exact}/{hi-lo}"

  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite} non-finite outputs"; IO.Process.exit 1
  if moved * 10 < n then
    IO.eprintln "DEGENERATE: too few non-zero outputs — the check proves little"; IO.Process.exit 1
  -- Both replicas saw the SAME 32 examples, so their BN groups are identical and the batch
  -- statistics cannot legitimately differ. Anything here is the DP path corrupting the forward.
  if !fwdExact then
    IO.eprintln s!"DP CHECK FAILED: the forward differs — `bnstat` is not bit-exact \
(norm-rel {fwdRel}). Both replicas were given the same batch, so their BN statistics are identical \
by construction; a difference here is the data-parallel path corrupting the forward."
    IO.Process.exit 1
  -- Gate the GRADIENT (`m`), never θ: Adam's update is scale-free, so a near-zero-gradient
  -- parameter flips sign on a 1-ULP difference and θ lands at ~1e-4 whether or not anything is
  -- wrong (§3). Measured on mnv2 itself in §2f: a perturbed cotangent moved θ by 1e-6 while `m`
  -- moved 1.17e-2.
  if gradRel > 1e-4 then
    IO.eprintln s!"DP CHECK FAILED: gradient (m) norm-rel {gradRel} > 1e-4. On a duplicated batch \
all_reduce(add)/2 is the identity, so the data-parallel step must reproduce the single-device one."
    IO.Process.exit 1
  IO.println s!"✓ DP step reproduces the single-device step on a duplicated batch: forward \
BIT-EXACT (bnstat, {net.bnChannels.size} BN layers), gradient norm-rel {gradRel} ≤ 1e-4, \
over all {n} returned floats"
