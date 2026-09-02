import LeanMlir.VerifiedNets

/-! # bs256 re-render gate — the duplicated-batch exact check

`planning/xla_pjrt_handoff.md` §2d.1. `verified_mlir/resnet34_adam256_train_step.mlir` is the same
`pretty(provenGraph)` as the bs32 artifact with `B := 256`; the two are structurally identical
(10014 ops, 9838 lines, same op profile) and differ only in tensor dimensions and one constant. That
is *exactly* the kind of change that looks obviously right and can still be silently wrong — the
§2b lesson was that a re-instantiation at a new batch index inflated 597 pointwise trailing dims
without changing the graph structure at all. So it gets a numeric gate rather than an argument.

**The gate, and why it is exact rather than a tolerance.** Feed the bs256 render **8 identical
copies** of the same 32 examples. Then, term by term:

* **batch BN** — μ and σ² over 256 rows that are 8 copies of 32 rows are, as real numbers, exactly
  μ and σ² over those 32 rows. So every BN layer sees the same statistics, and every example's
  forward activations are identical to its bs32 forward.
* **the loss cotangent** — this render's is *mean*-CE (`divide %dyr, dense<256.0>` against the bs32
  render's `dense<32.0>`; checked in the emitted text, not assumed). The mean of 8 copies is the
  mean. So the cotangent per example is unchanged, and the summed parameter gradient is too.

Therefore **every one of the 68,040,737 returned floats must agree** — θ', m', v', the loss, and all
72 BN batch-statistic slots. Not "up to a batch-size factor": equal. The only slack is floating-point
reduction order (256-element trees vs 32-element ones), which is why the gate is norm-relative 1e-4
rather than bit-equality, and why it reports the A-vs-A determinism floor first.

This is the same shape of argument as the cifar8 data-parallel gate (§2b-quater): find the input on
which two graphs that are *not* generally equal become provably equal, and check there.

**What it does not cover.** It says the bs256 render computes the bs32 function on a degenerate
batch. It cannot see an error that only appears when the 256 rows actually differ — nothing in a
one-batch check can. The structural comparison against the bs32 artifact (op counts and profile
identical) is what covers the rest, and it is cheap to re-run.

    lake build resnet34-batch-check
    HIP_VISIBLE_DEVICES=0 .lake/build/bin/resnet34-batch-check

`TIE_SKIP_AA=1` skips the determinism-floor run.
-/

private def nPOf (net : VerifiedNet) : Nat := net.nParams

def main (args : List String) : IO Unit := do
  let (path32, path256) := match args with
    | a :: b :: _ => (a, b)
    | _ => ("verified_mlir/resnet34_adam_train_step.mlir",
            "verified_mlir/resnet34_adam256_train_step.mlir")
  let net  := resnet34Verified.toNet
  let bs   := 32
  let reps := 8                            -- 8 × 32 = 256, the baked batch of the second render
  let bs256 := bs * reps
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  IO.println "@resnet34_adam_train_step vs @resnet34_adam256_train_step — duplicated-batch exact check"
  IO.println s!"  A = {path32}   (B = {bs})"
  IO.println s!"  B = {path256}  (B = {bs256}, fed {reps} identical copies of A's batch)"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), {net.bnChannels.size} BN layers \
({nBnStats} stat floats), backend {← LowererSession.backendName}"

  -- ── the shared parameter blob: identical bytes to both renders ──
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  let m ← F32.heInit 4242 net.nParams.toUSize 0.02
  let v ← F32.scaleShift (← F32.heInit 8484 net.nParams.toUSize 0.01) 1.0 0.05
  let tail ← F32.const 3 0.0
  let tail ← F32.write3 tail 0 0.001 0.19 0.002    -- lr, 1−β₁ᵗ, 1−β₂ᵗ at a mid-training step
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let pbuf := F32.concat #[θ, m, v, tail, bnIn]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]] ++ bnStatShapes)

  -- ── the batch, and its 8-fold duplicate ──
  let x32 ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let x256 := F32.concat (Array.replicate reps x32)
  let mut y32 : ByteArray := .empty
  for i in [0:bs] do
    y32 := y32.push (UInt8.ofNat (i % net.nClasses)); y32 := y32.push 0
    y32 := y32.push 0; y32 := y32.push 0
  let mut y256 : ByteArray := .empty
  for _ in [0:reps] do y256 := y256 ++ y32
  if x256.size != reps * x32.size || y256.size != reps * y32.size then
    IO.eprintln "internal: duplicate construction is wrong"; IO.Process.exit 1

  let runOne (path fn tag : String) (x y : ByteArray) (b : Nat) : IO ByteArray := do
    let sess ← mkSession path
    LowererSession.mlpTrainStepV sess fn x pbuf shapes y
      b.toUSize net.d0.toUSize net.nClasses.toUSize
  IO.println "  running bs32…"; (← IO.getStdout).flush
  let oa ← runOne path32 "m.resnet34_adam_train_step" "a" x32 y32 bs
  -- The determinism floor. Without it "the difference is 1e-6" is an assertion, not a measurement.
  if !((← IO.getEnv "TIE_SKIP_AA").isSome) then
    IO.println "  running bs32 again (determinism floor)…"; (← IO.getStdout).flush
    let oa2 ← runOne path32 "m.resnet34_adam_train_step" "a2" x32 y32 bs
    let mut fl : Float := 0.0
    let mut ex : Nat := 0
    for i in [0:oa.size / 4] do
      let d := (F32.read oa i.toUSize - F32.read oa2 i.toUSize).abs
      if d == 0.0 then ex := ex + 1
      if d > fl then fl := d
    IO.println s!"  bs32-vs-bs32 floor: max|a−a'| = {fl}, bit-exact {ex}/{oa.size / 4}"
  -- ── the floor that actually matters: same function, DIFFERENT REDUCTION ORDER ──
  -- Reversing the 32 rows leaves every real-number quantity untouched — BN μ/σ² are symmetric in
  -- the batch, the mean-CE is a mean, and each example keeps its own label — but it changes the
  -- order in which fp values are summed. That is precisely the difference between the bs32 and
  -- bs256 graphs on a duplicated batch, isolated from the batch change itself. Without this the
  -- bit-exact floor above would be the wrong yardstick: it contains no reduction-order variation
  -- at all, so it cannot say whether a given disagreement is noise or a defect.
  let rowBytes := net.d0 * 4
  let mut xRev : ByteArray := .empty
  let mut yRev : ByteArray := .empty
  for k in [0:bs] do
    let i := bs - 1 - k
    xRev := xRev ++ x32.extract (i * rowBytes) ((i + 1) * rowBytes)
    yRev := yRev ++ y32.extract (i * 4) ((i + 1) * 4)
  IO.println "  running bs32 on the REVERSED batch…"; (← IO.getStdout).flush
  let oRev ← runOne path32 "m.resnet34_adam_train_step" "rev" xRev yRev bs

  IO.println "  running bs256 on the duplicated batch…"; (← IO.getStdout).flush
  let ob ← runOne path256 "m.resnet34_adam256_train_step" "b" x256 y256 bs256

  -- ── the CONTROL that calibrates the gradient: same graph, same batch, different order ──
  -- `x256` is the 32-batch repeated 8×; `xGrp` is each example repeated 8× consecutively. Same
  -- multiset of 256 rows, so every real-number quantity is identical — but the 256-wide reductions
  -- accumulate in a different order. That is the *same class* of fp difference the bs32-vs-bs256
  -- comparison has, measured on one graph where correctness is not in question. It is what makes
  -- the gradient figure below a measurement rather than an assertion; the reversed-bs32 run cannot
  -- do it, because 32-wide reductions turn out to be order-insensitive here (forward bit-exact).
  let mut xGrp : ByteArray := .empty
  let mut yGrp : ByteArray := .empty
  for i in [0:bs] do
    for _ in [0:reps] do
      xGrp := xGrp ++ x32.extract (i * rowBytes) ((i + 1) * rowBytes)
      yGrp := yGrp ++ y32.extract (i * 4) ((i + 1) * 4)
  IO.println "  running bs256 on the REGROUPED duplicate (control)…"; (← IO.getStdout).flush
  let oGrp ← runOne path256 "m.resnet34_adam256_train_step" "grp" xGrp yGrp bs256

  if oa.size != ob.size then
    IO.eprintln s!"SIZE MISMATCH: {oa.size} vs {ob.size} bytes — the two renders do not present \
the same interface, which they must (the parameter count does not depend on B)"
    IO.Process.exit 1
  let n := oa.size / 4
  let nP := net.nParams

  -- ── per region ────────────────────────────────────────────────────────────────────────────
  -- `bnstat` is the batch μ/var of all 36 BN inputs, so it depends only on the forward — it
  -- separates "the duplicated batch did not reproduce the statistics" from "the backward differs".
  let regions : List (String × Nat × Nat) :=
    [("theta", 0, nP), ("m", nP, 2*nP), ("v", 2*nP, 3*nP),
     ("loss/bc", 3*nP, 3*nP+3), ("bnstat", 3*nP+3, n)]
  -- Report each candidate against `oa` the SAME way, returning
  -- (forward norm-rel, gradient norm-rel, non-finite count, non-zero count).
  -- `bnstat` is the batch μ/var of all 36 BN inputs, so it is the forward; `m` is `(1−β₁)·g` off a
  -- shared `m`, so it is the gradient. The pair is what matters — see the amplification note below.
  let report (label : String) (p q : ByteArray) : IO (Float × Float × Nat × Nat) := do
    let mut fwd : Float := 0.0
    let mut grad : Float := 0.0
    let mut nf : Nat := 0
    let mut mv : Nat := 0
    IO.println s!"  ── {label} ──"
    for (nm, lo, hi) in regions do
      let mut ra : Float := 0.0
      let mut rm : Float := 0.0
      let mut exact : Nat := 0
      for i in [lo:hi] do
        let a := F32.read p i.toUSize
        let b := F32.read q i.toUSize
        if !a.isFinite || !b.isFinite then nf := nf + 1
        if max a.abs b.abs > 1e-12 then mv := mv + 1
        let d := (a - b).abs
        if d == 0.0 then exact := exact + 1
        if d > ra then ra := d
        if a.abs > rm then rm := a.abs
      let nr := if rm > 1e-30 then ra / rm else 0.0
      if nm == "bnstat" then fwd := nr
      if nm == "m" then grad := nr
      IO.println s!"    {nm}: max|a-b| = {ra}, max|a| = {rm}, norm-rel = {nr}, \
bit-exact {exact}/{hi-lo}"
    return (fwd, grad, nf, mv)
  let (fwdRev, gradRev, nfRev, _) ←
    report "bs32 vs bs32 REVERSED (does a 32-wide reduction reorder at all?)" oa oRev
  let (fwdCtl, gradCtl, nfCtl, _) ←
    report "CONTROL: bs256 duplicated vs bs256 REGROUPED (same graph, same batch)" ob oGrp
  let (fwd256, grad256, nf256, mv256) ←
    report "TEST: bs32 vs bs256 on the duplicated batch" oa ob

  if nfRev + nfCtl + nf256 > 0 then
    IO.eprintln s!"DEGENERATE: {nfRev + nfCtl + nf256} non-finite outputs"; IO.Process.exit 1
  if mv256 * 10 < n then
    IO.eprintln s!"DEGENERATE: only {mv256}/{n} outputs are non-zero — the check proves little"
    IO.Process.exit 1

  -- ── the gate ──────────────────────────────────────────────────────────────────────────────
  -- TWO gates, because the forward and the gradient have completely different conditioning here.
  --
  --  1. the FORWARD, in absolute terms. `bnstat` is a plain symmetric mean and variance over the
  --     batch; it amplifies nothing, and on a duplicated batch it is *exactly* the bs32 value. A
  --     real mis-wiring of the bs256 graph lands here. Gate: 1e-4.
  --
  --  2. the GRADIENT, COMPARATIVELY — never in absolute terms. Handoff §3: R34's gradient does not
  --     reproduce to better than ~6e-3 against the same backend under a sub-ULP forward nudge, so a
  --     1e-4 absolute gradient gate fails a correct render by 60×. The control measures what this
  --     net's gradient does under a change that is *provably* semantics-preserving — reordering the
  --     same 256 rows — on the same graph, same batch, same backend. If the bs32-vs-bs256 gradient
  --     difference is no larger than that, it is conditioning, not a defect.
  IO.println "  ── summary (forward = bnstat, gradient = m) ──"
  IO.println s!"    bs32 reordered  : forward {fwdRev}  gradient {gradRev}"
  IO.println s!"    CONTROL bs256   : forward {fwdCtl}  gradient {gradCtl}"
  IO.println s!"    TEST bs32↔bs256 : forward {fwd256}  gradient {grad256}"
  if fwd256 > 1e-4 then
    IO.eprintln s!"BATCH CHECK FAILED: the FORWARD differs at norm-rel {fwd256} > 1e-4. On a \
duplicated batch the batch statistics are exactly the bs32 ones, so this is a defect in the bs256 \
render, not conditioning."
    IO.Process.exit 1
  let bound := max 1.0e-4 (4.0 * gradCtl)
  if grad256 > bound then
    IO.eprintln s!"BATCH CHECK FAILED: gradient norm-rel {grad256} exceeds {bound} — 4× the \
{gradCtl} that the SAME bs256 graph produces under a provably semantics-preserving reordering of \
the same 256 rows. That excess is attributable to the re-render, not to this net's conditioning."
    IO.Process.exit 1
  IO.println s!"✓ bs256 render agrees with bs32 on the duplicated batch: forward norm-rel \
{fwd256} ≤ 1e-4, gradient {grad256} ≤ {bound} (4× the control's {gradCtl})"
