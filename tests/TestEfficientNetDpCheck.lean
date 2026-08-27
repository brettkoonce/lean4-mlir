import LeanMlir.VerifiedNets

/-! # EfficientNet data-parallel gate — the collective's semantics, on a duplicated batch

The EfficientNet peer of `tests/TestViTDpCheck.lean` and `tests/TestCifar8DpCheck.lean`.
`verified_mlir/efficientnet_adamdp_train_step.mlir` inserts one `all_reduce(add)/N` per parameter
between the certified gradient and the certified AdamW triple (handoff §2b-quater's pattern). That
collective is a **trusted carve-out** — emitted text, outside every faithfulness theorem — so it
needs its own numeric check.

**The exact identity used here, and why BatchNorm does not spoil it.** Give both replicas the
**same** 32 examples. BN normalises per replica, so each replica's BN group is those same 32
examples and both compute *identical* batch statistics; each therefore computes the same gradient
`g`, and `all_reduce(add)/2` returns `(g + g)/2 = g`. The mean is an identity on a duplicated batch.
The data-parallel step must reproduce the **single-device** step on that batch, output for output.

The §10.3b caveat that stopped R34 from having this gate is about **splitting** a batch — 2×32 is
genuinely not 1×64 under batch BN, because the two halves get different statistics. Duplicating a
batch is the other case: the statistics are the same by construction, so nothing couples the
replicas and the identity is exact rather than approximate.

**This gate is stronger than ViT's**, for the same reason `efficientnet-adam-tie` is stronger than
`vit-adam-tie`: EfficientNet returns 98 BN batch statistics, so it has a forward-only `bnstat`
region. On a duplicated batch that region must come back **bit-exact** — every replica saw the same
data, so any difference there is the DP path corrupting the forward, which no gradient tolerance
would have caught.

Two failure modes it separates, both of which have actually happened in this repo:

* **collective missing** → the shim's replica-count guard refuses the call before any numbers.
* **collective present but wrong** (sum not mean) → every gradient is 2× and `m` moves by ~1, five
  orders above the gate. §2b-quater verified exactly that by breaking the divisor.

    lake build efficientnet-dp-check
    unset HIP_VISIBLE_DEVICES && PJRT_REPLICAS=2 .lake/build/bin/efficientnet-dp-check

Needs TWO GPUs and the XLA backend (collectives do not exist on the IREE path).
-/

private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0          -- BN γ
  | 2 => F32.const n.toUSize 0.0          -- BN β / biases
  | _ =>
    let fanIn := if dims.size == 4 then dims[1]! * dims[2]! * dims[3]! else dims[0]!
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))

def main (args : List String) : IO Unit := do
  -- ▶ Env-selected (net, batch, replicas, variant pair), defaulting to EXACTLY the Imagenette /
  -- AdamW / 2-replica configuration this harness was written for, so the committed result
  -- reproduces with no arguments — which is the gate on the generalisation itself. Added for the
  -- RMSProp DP render, which exists only at the ImageNet shape (`efficientnetin`, B=64 × 4 replicas).
  -- ⚠ `shard-check` cannot substitute: its `DP([A|B]) = mean(single(A), single(B))` needs the gated
  -- slot LINEAR in the gradient, which RMSProp's buffer is not. The duplicated-batch identity here
  -- is optimizer-agnostic — both sides see the identical gradient. Mirrors `TestMobilenetV2DpCheck`.
  let netSel := (← IO.getEnv "DP_NET").getD "imagenette"
  let (net, bsDefault, repDefault) := match netSel with
    | "imagenet" => (efficientnetImagenetVerified.toNet, 64, 4)
    | _          => (efficientnetVerified.toNet, 32, 2)
  let bs := ((← IO.getEnv "DP_BATCH").bind (·.toNat?)).getD bsDefault   -- the BAKED per-replica batch
  let replicas := ((← IO.getEnv "DP_REPLICAS").bind (·.toNat?)).getD repDefault
  let vSg := (← IO.getEnv "DP_VARIANT").getD "adam"
  let vDp := (← IO.getEnv "DP_VARIANT_DP").getD "adamdp"
  -- ⚠ `ema*` variants carry a FOURTH `[θ|m|v|ema]` region and a 5-slot scalar tail. The harness has
  -- to BUILD the blob it feeds rather than assume three regions — getting it wrong is not a
  -- tolerance question, PJRT refuses on the buffer count (§2m's `expected 265 buffers`).
  -- ⚠ `emarms` is this net's real recipe and it is why `rmsOn` is a SUBSTRING test — but `emaOn`
  -- is and always was the PREFIX one, which is what `emarms` satisfies. The two lessons are not
  -- the same lesson.
  -- ⚠⚠ **THESE THREE ARE THE DRIVER'S OWN, NOT A TRANSCRIPTION OF THEM (2026-08-27).** They read
  -- `let emaOn := …; let nRegions := if emaOn then 4 else 3` — this file's private copy of what
  -- `trainAdamSched` computes, which is precisely the drift `tests/TestVariantPredicates.lean`'s
  -- header warns about one level up: *a gate on a transcription is not a gate on the thing
  -- transcribed*. Two things had gone wrong by the time it was noticed:
  --   ⛔ the copy was a SUBSTRING test where `VerifiedVariant.emaOn` is a PREFIX one, so `adamema…`
  --      would have made this harness build four regions for a graph the driver feeds three;
  --   ⛔ the copy is frozen at `4 else 3`, and the region count became **3, 4 or 5** the day EMA and
  --      gradient accumulation stopped sharing a slot (`VerifiedVariant.nRegions`, RSB-A2/A1).
  -- Neither is reachable from the variants this gate runs today. Both are one variant string away.
  let emaOn := VerifiedVariant.emaOn vSg
  let nRegions := VerifiedVariant.nRegions vSg
  let nScalars := VerifiedVariant.nScalars vSg
  let sgPath := s!"verified_mlir/{net.slug}_{vSg}_train_step.mlir"
  -- The DP render is overridable so a deliberately-broken one can be fed in. That is not a
  -- convenience: a gate nobody has seen go red is an assertion. §2b-quater's control — the `%arn`
  -- divisor 2.0 → 1.0, i.e. sum instead of mean — is the one to run.
  let dpPath := args.head?.getD s!"verified_mlir/{net.slug}_{vDp}_train_step.mlir"
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  IO.println "EfficientNet-B0 data-parallel gate — duplicated batch"
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
  let tail ← F32.const nScalars.toUSize 0.0
  let tail ← F32.write3 tail 0 0.001 0.19 0.002
  -- The EMA decay pair, at a value that is neither 0 nor 1 so the shadow is a genuine blend of the
  -- old shadow and θ' — a degenerate `d` lets a region wired to the wrong slot agree by accident.
  let tail ← if emaOn then do
      let dPair ← F32.const 3 0.0
      let dPair ← F32.write3 dPair 0 0.9 0.1 0.0
      F32.blit tail 3 dPair 0 2
    else pure tail
  -- ⚠ The shadow starts from an arbitrary NON-θ state, deliberately: seeded at θ,
  -- `ema' = d·θ + (1−d)·θ'` would agree with a harness that had wired the region to the wrong slot.
  let e ← F32.scaleShift (← F32.heInit 9999 net.nParams.toUSize 0.03) 1.0 0.02
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let pbuf := F32.concat (#[θ, m, v] ++ (if emaOn then #[e] else #[]) ++ #[tail, bnIn])
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ (if emaOn then net.paramShapes else #[])
                            ++ Array.replicate nScalars #[] ++ bnStatShapes)
  let x1 ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  -- The SAME batch on EVERY replica — `replicas` copies, not two: `all_reduce(add)/N` over N
  -- identical gradients is the identity at any N, so only this concatenation was 2-specific.
  let x2 := F32.concat (Array.replicate replicas x1)
  let mut y1 : ByteArray := .empty
  for i in [0:bs] do
    y1 := y1.push (UInt8.ofNat (i % net.nClasses)); y1 := y1.push 0
    y1 := y1.push 0; y1 := y1.push 0
  let y2 := (Array.replicate replicas y1).foldl (· ++ ·) ByteArray.empty

  IO.println "  running single-device…"; (← IO.getStdout).flush
  let s1 ← mkSession sgPath
  let o1 ← LowererSession.mlpTrainStepV s1 s!"m.{net.slug}_{vSg}_train_step" x1 pbuf shapes y1
             bs.toUSize net.d0.toUSize net.nClasses.toUSize
  IO.println "  running data-parallel…"; (← IO.getStdout).flush
  -- Delete first: `compileVmfb` keys on the OUTPUT path and an mtime, never the source, so a
  -- second run with a different candidate silently reuses the first one's binary (handoff §4).
  for p in [".lake/build/enet_dp_b.vmfb",
            s!".lake/build/enet_dp_b_{((← IO.getEnv "IREE_BACKEND").getD "cuda")}.vmfb"] do
    if ← System.FilePath.pathExists p then IO.FS.removeFile p
  let s2 ← mkSession dpPath
  let o2 ← LowererSession.mlpTrainStepVDP s2 s!"m.{net.slug}_{vDp}_train_step" x2 pbuf shapes y2
             (bs * replicas).toUSize net.d0.toUSize net.nClasses.toUSize replicas.toUSize

  if o1.size != o2.size then
    IO.eprintln s!"SIZE MISMATCH: {o1.size} vs {o2.size}"; IO.Process.exit 1
  let n := o1.size / 4
  let nP := net.nParams
  -- ⚠ The EMA shadow is REPORTED, never GATED. `planning/ema.md`: the shadow is θ's low-pass
  -- filter, so gating it is §3's "gate the gradient, never θ" one step WORSE — measured, a
  -- sum-not-mean control moved `m` by 0.94, θ by 1.95e-4 and the shadow by 1.00e-4, i.e. exactly
  -- ON a 1e-4 gate. It is here so a mis-threaded 4th region is visible, not so it decides anything.
  let regions : List (String × Nat × Nat) :=
    [("theta", 0, nP), ("m", nP, 2*nP), ("v", 2*nP, 3*nP)]
    ++ (if emaOn then [("ema", 3*nP, 4*nP)] else [])
    ++ [("loss/bc", nRegions*nP, nRegions*nP+nScalars),
        ("bnstat", nRegions*nP+nScalars, n)]
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
    -- prints as "0.000000" and reads as bit-exact when it is not — and here it is not: the DP
    -- module is a different HLO program, so XLA is free to order the backward reductions
    -- differently even though the collective itself is exact ((g+g)/2 = g holds to the bit in
    -- binary floating point). Report the count too; that is what distinguishes the two.
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
  -- wrong (§3). §2b-quater measured this directly — a 2× gradient error moved θ by 2.7e-4 and `m`
  -- by 0.96.
  if gradRel > 1e-4 then
    IO.eprintln s!"DP CHECK FAILED: gradient (m) norm-rel {gradRel} > 1e-4. On a duplicated batch \
all_reduce(add)/2 is the identity, so the data-parallel step must reproduce the single-device one."
    IO.Process.exit 1
  IO.println s!"✓ DP step reproduces the single-device step on a duplicated batch: forward \
BIT-EXACT (bnstat, {net.bnChannels.size} BN layers), gradient norm-rel {gradRel} ≤ 1e-4, \
over all {n} returned floats"
