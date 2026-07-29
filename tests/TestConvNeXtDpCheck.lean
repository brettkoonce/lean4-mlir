import LeanMlir.VerifiedNets

/-! # ConvNeXt-T data-parallel gate — the collective's semantics, on a duplicated batch

The ConvNeXt peer of `tests/TestMobilenetV2DpCheck.lean` / `tests/TestEfficientNetDpCheck.lean`, and
gated by the same **exact** identity rather than a tolerance argument.
`verified_mlir/convnext_adamdp_train_step.mlir` inserts one `all_reduce(add)/N` per parameter
between the certified gradient and the certified AdamW triple (handoff §2b-quater's pattern). That
collective is a **trusted carve-out** — emitted text, outside every faithfulness theorem — so it
needs its own numeric check.

**The exact identity used here.** Give both replicas the **same** 32 examples. Each computes the
same gradient `g`, so `all_reduce(add)/2` returns `(g + g)/2 = g` — the mean is an identity on a
duplicated batch. The data-parallel step must therefore reproduce the **single-device** step on that
batch, output for output.

**ConvNeXt needs no BatchNorm caveat at all, and it is the only net here that does not.** The
§10.3b caveat — that 2×32 is genuinely not 1×64, because the halves get different batch statistics —
is a statement about batch BN. ConvNeXt normalises with **LayerNorm**, which reduces within one
example and never across the batch, so every replica's arithmetic on its own examples is what it
would have been in a single batch of 64. EfficientNet and mnv2 have to argue that *duplicating* a
batch keeps the two replicas' BN groups identical; here there is nothing to argue.

**The consequence for what this gate can see.** ConvNeXt returns no BN batch statistics, so unlike
the mnv2/EfficientNet gates there is no `bnstat` region pinning the forward bit-exactly. The only
forward-only output is **`%loss`** — report-only, on no gradient path, covered by no theorem, which
is exactly the configuration in which §2b shipped plain CE against a smoothed-CE cotangent. So this
harness **gates** `%loss` as well as the gradient, the same split `convnext-adam-tie` uses.

**44 of the 180 collectives are RANK-0** — the scalar LayerNorm γ/β at `tensor<f32>`. No other net's
DP render has an operand below rank 1, so this harness is also the first execution of a scalar
`stablehlo.all_reduce` anywhere in the repo.

Two failure modes it separates, both of which have actually happened here:

* **collective missing** → the shim's replica-count guard refuses the call before any numbers.
* **collective present but wrong** (sum not mean) → every gradient is 2× and `m` moves by ~1, five
  orders above the gate. §2b-quater, §2e-bis and §2h-bis each verified exactly that by breaking the
  divisor; pass the broken render as `argv[1]` to run it here.

    lake build convnext-dp-check
    unset HIP_VISIBLE_DEVICES && PJRT_REPLICAS=2 .lake/build/bin/convnext-dp-check

Needs TWO GPUs and the XLA backend (collectives do not exist on the IREE path — the IREE shim
refuses a DP entry point outright rather than silently running single-device, which is why
`convnext-verified-adam-xla` had to exist first, §2h).
-/

private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0          -- LayerNorm γ / layerScale
  | 2 => F32.const n.toUSize 0.0          -- LayerNorm β / biases
  | _ =>
    let fanIn := if dims.size == 4 then dims[1]! * dims[2]! * dims[3]! else dims[0]!
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))

def main (args : List String) : IO Unit := do
  let net := convnextVerified.toNet
  let bs := 32                                   -- the baked per-replica batch
  let replicas := 2
  -- The DP render is overridable so a deliberately-broken one can be fed in. That is not a
  -- convenience: a gate nobody has seen go red is an assertion. §2b-quater's control — the `%arn`
  -- divisor 2.0 → 1.0, i.e. sum instead of mean — is the one to run.
  let dpPath := args.head?.getD "verified_mlir/convnext_adamdp_train_step.mlir"
  IO.println "ConvNeXt-T data-parallel gate — duplicated batch"
  IO.println s!"  single : verified_mlir/convnext_adam_train_step.mlir   (bs {bs})"
  IO.println s!"  DP     : {dpPath} ({replicas} replicas, \
global {bs * replicas} = the same {bs} examples twice)"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), no BatchNorm (LayerNorm), \
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
  -- Delete first on BOTH sides: `compileVmfb` keys on the OUTPUT path and an mtime, never the
  -- source, so a second run with a different candidate silently reuses the first one's binary
  -- (handoff §4) — which is exactly what running the sum-not-mean control looks like.
  for tag in ["cnx_dp_a", "cnx_dp_b"] do
    for p in [s!".lake/build/{tag}.vmfb",
              s!".lake/build/{tag}_{((← IO.getEnv "IREE_BACKEND").getD "cuda")}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
  let s1 ← mkSession "verified_mlir/convnext_adam_train_step.mlir" ".lake/build/cnx_dp_a.vmfb"
  let o1 ← IreeSession.mlpTrainStepV s1 "m.convnext_adam_train_step" x1 pbuf shapes y1
             bs.toUSize net.d0.toUSize net.nClasses.toUSize
  IO.println "  running data-parallel…"; (← IO.getStdout).flush
  let s2 ← mkSession dpPath ".lake/build/cnx_dp_b.vmfb"
  let o2 ← IreeSession.mlpTrainStepVDP s2 "m.convnext_adamdp_train_step" x2 pbuf shapes y2
             (bs * replicas).toUSize net.d0.toUSize net.nClasses.toUSize replicas.toUSize

  if o1.size != o2.size then
    IO.eprintln s!"SIZE MISMATCH: {o1.size} vs {o2.size}"; IO.Process.exit 1
  let n := o1.size / 4
  let nP := net.nParams
  -- `%loss` gets its own region: it is the ONLY output that reads the forward alone, so folding it
  -- in with `%bc1`/`%bc2` (constant passthroughs, which agree trivially) would let a forward
  -- disagreement hide behind two exact values.
  let regions : List (String × Nat × Nat) :=
    [("theta", 0, nP), ("m", nP, 2*nP), ("v", 2*nP, 3*nP),
     ("loss", 3*nP, 3*nP+1), ("bc", 3*nP+1, n)]
  let mut gradRel : Float := 0.0
  let mut lossRel : Float := 0.0
  let mut lossAbs : Float := 0.0
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
    if nm == "loss" then lossRel := nr; lossAbs := ra
    -- Also in nano-units. `Float.toString` gives six decimals, so a genuine 3e-8 disagreement
    -- prints as "0.000000" and reads as bit-exact when it is not — EfficientNet's DP gate lands at
    -- exactly that (3.3e-8 on `m`) and mnv2's at 8.7e-8, because the DP module is a different HLO
    -- program and XLA is free to order the backward reductions differently even though the
    -- collective itself is exact ((g+g)/2 = g holds to the bit in binary FP). Report the count too;
    -- that is what distinguishes "bit-exact" from "prints as zero".
    IO.println s!"    {nm}: max|a-b| = {ra} ({ra * 1e9} e-9), max|a| = {rm}, \
norm-rel = {nr} ({nr * 1e9} e-9), bit-exact {exact}/{hi-lo}"

  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite} non-finite outputs"; IO.Process.exit 1
  if moved * 10 < n then
    IO.eprintln "DEGENERATE: too few non-zero outputs — the check proves little"; IO.Process.exit 1
  -- Gate `%loss`: with no BN statistics returned, this is the whole of the forward evidence. Both
  -- replicas saw the same 32 examples and the collective touches nothing upstream of the loss, so
  -- a difference here is the DP path disturbing the forward. 1e-4, matching `convnext-adam-tie`.
  if lossRel > 1e-4 then
    IO.eprintln s!"DP CHECK FAILED: %loss differs by {lossAbs} (rel {lossRel}) > 1e-4. Both \
replicas were given the same batch and the collective is downstream of the loss, so the forward \
must reproduce exactly."
    IO.Process.exit 1
  -- Gate the GRADIENT (`m`), never θ: Adam's update is scale-free, so a near-zero-gradient
  -- parameter flips sign on a 1-ULP difference and θ lands at ~1e-4 whether or not anything is
  -- wrong (§3). Measured five times now — on the mnv2 DP control a 2× gradient error moved θ by
  -- 8.4e-5, i.e. UNDER a 1e-4 θ gate, while `m` moved 1.037.
  if gradRel > 1e-4 then
    IO.eprintln s!"DP CHECK FAILED: gradient (m) norm-rel {gradRel} > 1e-4. On a duplicated batch \
all_reduce(add)/2 is the identity, so the data-parallel step must reproduce the single-device one."
    IO.Process.exit 1
  IO.println s!"✓ DP step reproduces the single-device step on a duplicated batch: %loss rel \
{lossRel} ≤ 1e-4, gradient norm-rel {gradRel} ≤ 1e-4, over all {n} returned floats"
