import LeanMlir.VerifiedNets

/-! # `@efficientnet_adam_train_step` render tie — hand-written vs `pretty(provenGraph)`

`planning/xla_pjrt_handoff.md`, the EfficientNet AdamW thread, step 3.
`Proofs/Codegen/EfficientNetRender.lean`'s `efficientnetAdamTrainStepFaithful` renders the same
train step the hand-written emitter in `tests/TestEfficientNetTrain.lean` does — the one
`efficientnet-verified-adam` trains on. This harness is what licenses swapping them; run it BEFORE
retiring the hand-written emitter, because afterwards the comparison no longer exists.

The interface is positionally identical (**889 in / 887 out**, arg and return types equal in order —
only the 98 unused BN passthrough slot NAMES differ, `%stnmui` vs `%bnmu0i`, which the packed-buffer
FFI never sees), so both take the same `[θ|m|v | lr,bc1,bc2 | bn stats]` blob `trainAdamSched`
builds.

**Why a numeric tie and not a text diff.** Same function, different graph: SSA naming differs, and
the certified render emits 17,545 ops against 10,421 because `pretty` has no CSE (§2b-bis measured
the identical 1.68× on R34 and found it costs nothing after XLA optimisation). The cotangents are
also composed differently — the hand-written render fuses label smoothing into one `[B,10]` block
while the kit composes `softmaxRow → subB → scaleB → addVB → shiftB → divConstB`. Only running both
and comparing every returned float settles it.

**This tie is STRONGER than `vit-adam-tie`, and the reason is BatchNorm.** EfficientNet returns 98
batch statistics — μ and σ² of all 49 BN inputs — which depend on the **forward alone**. That
`bnstat` region pins the whole forward chain (stem, all 16 MBConv blocks including every expand,
depthwise and project BN, and the head) to the last bit, so a forward disagreement and a backward
one are separable in a single run. ViT had no such region and had to lean on `%loss`.

`%loss` is still gated rather than merely reported. It is report-only, on no gradient path, and
covered by no theorem — which is exactly the configuration in which §2b shipped plain CE against a
smoothed-CE cotangent, caught only by the numeric tie. Here it is a *cross-check* on `bnstat`
rather than the sole forward evidence.

    lake build efficientnet-adam-tie
    .lake/build/bin/efficientnet-adam-tie [refRender.mlir] [candRender.mlir]

Linked against **IREE**, not XLA/PJRT, for the same reason `vit-adam-tie` is: `efficientnet-verified-adam`
is an IREE binary, and a tie should run on the backend the trainer actually uses. (This box is also
MIOpen-conv-weak, and EfficientNet is all depthwise convolutions — see the ROCm note.)

Exits non-zero if the renders disagree or the comparison is degenerate.
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
  let handWritten := "verified_mlir/efficientnet_adam_train_step.mlir"
  let certified   := "verified_mlir/efficientnet_adam_train_step_b.mlir"
  -- Pre-swap the certified render lives at its own path, so the no-argument form IS the migration
  -- check. Post-swap that path is gone and the default degrades to an A-vs-A determinism run —
  -- still worth having (it re-establishes the floor the gate depends on), but say so out loud.
  let certExists ← System.FilePath.pathExists certified
  let (pathA, pathB) := match args with
    | a :: b :: _ => (a, b)
    | [a]         => (handWritten, a)
    | []          => (handWritten, if certExists then certified else handWritten)
  let net := efficientnetVerified.toNet
  let bs  := 32                            -- the baked batch of both renders
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  IO.println "@efficientnet_adam_train_step tie"
  IO.println s!"  A (reference, hand-written) = {pathA}"
  IO.println s!"  B (candidate, certified)    = {pathB}"
  if pathA == pathB then
    IO.println "  NOTE: both paths are the same file — an A-vs-A determinism run, NOT a migration \
check. Pass the retired render as the first argument for that."
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), {net.bnChannels.size} BN layers \
({nBnStats} stat floats), bs {bs}, backend {← IreeSession.backendName}"

  -- ── θ (driver init), m (centred noise), v (POSITIVE — it sits under a sqrt) ──
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  let m ← F32.heInit 4242 net.nParams.toUSize 0.02
  -- v ≈ 0.05 ± 0.01, strictly positive: a zero v makes √v̂ + ε ≈ ε everywhere and would hide any
  -- error in the second-moment path.
  let v ← F32.scaleShift (← F32.heInit 8484 net.nParams.toUSize 0.01) 1.0 0.05
  let tail ← F32.const 3 0.0
  let tail ← F32.write3 tail 0 0.001 0.19 0.002    -- lr, 1−β₁ᵗ, 1−β₂ᵗ at a mid-training step
  -- BN passthrough slots. BOTH renders ignore their values (each recomputes the batch statistics),
  -- so any finite fill is fine; non-zero is used so a slot that is wrongly echoed rather than
  -- recomputed shows up as a mismatch instead of matching a zero by luck.
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let pbuf := F32.concat #[θ, m, v, tail, bnIn]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]] ++ bnStatShapes)
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat (i % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0

  -- A tie must NEVER reuse a cached binary. `compileVmfb` keys on **mtime**, not on the source
  -- path, so running this harness twice with different candidates under the same tag silently
  -- reuses the FIRST candidate's `.vmfb` and reports the second as a perfect match. That is not
  -- hypothetical — it happened while building the negative controls below, and it produced a
  -- bit-exact "pass" for a render that had never been compiled. Delete before compiling.
  let runOne (path tag : String) : IO ByteArray := do
    let vmfb := s!".lake/build/enet_adam_tie_{tag}.vmfb"
    let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
    let vmfbT := s!".lake/build/enet_adam_tie_{tag}_{target}.vmfb"
    for p in [vmfb, vmfbT] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
    let sess ← mkSession path vmfb
    IreeSession.mlpTrainStepV sess "m.efficientnet_adam_train_step" x pbuf shapes y
      bs.toUSize net.d0.toUSize net.nClasses.toUSize
  IO.println "  running A…"; (← IO.getStdout).flush
  let oa ← runOne pathA "a"
  -- The determinism floor. Without it "the difference is 1e-6" is an assertion, not a measurement:
  -- it is what makes any A-vs-B difference graph-attributable rather than backend noise.
  IO.println "  running A again (determinism floor)…"; (← IO.getStdout).flush
  let oa2 ← runOne pathA "a2"
  let mut floor : Float := 0.0
  let mut fex : Nat := 0
  for i in [0:oa.size / 4] do
    let d := (F32.read oa i.toUSize - F32.read oa2 i.toUSize).abs
    if d == 0.0 then fex := fex + 1
    if d > floor then floor := d
  IO.println s!"  A-vs-A floor: max|a−a'| = {floor}, bit-exact {fex}/{oa.size / 4}"
  IO.println "  running B…"; (← IO.getStdout).flush
  let ob ← runOne pathB "b"

  if oa.size != ob.size then
    IO.eprintln s!"SIZE MISMATCH: {oa.size} vs {ob.size} bytes"; IO.Process.exit 1
  let n := oa.size / 4
  let nP := net.nParams
  if n != 3 * nP + 3 + nBnStats then
    IO.eprintln s!"ARITY: expected {3*nP+3+nBnStats} returned floats, got {n}"; IO.Process.exit 1

  -- ── per REGION ────────────────────────────────────────────────────────────────────────────
  -- `bnstat` depends ONLY on the forward (batch μ/var of all 49 BN inputs), so it separates a
  -- forward disagreement from a backward one in a single run. `m' = β₁·m + (1−β₁)·g` off a shared
  -- `m`, so the `m` region IS the gradient. θ' is scale-free under Adam (§3: a near-zero-gradient
  -- coordinate flips sign on a 1-ULP difference and moves a full ±lr), so θ' is reported, never
  -- gated. Errors are norm-relative (max|a−b| / max|a|) because a per-coordinate ratio on a
  -- near-zero gradient entry is meaningless.
  let regions : List (String × Nat × Nat) :=
    [("theta", 0, nP), ("m", nP, 2*nP), ("v", 2*nP, 3*nP),
     ("loss/bc", 3*nP, 3*nP+3), ("bnstat", 3*nP+3, n)]
  let mut nonFinite : Nat := 0
  let mut moved : Nat := 0
  let mut gradNormRel : Float := 0.0
  let mut lossRel : Float := 0.0
  let mut fwdExact := true
  let mut fwdRel : Float := 0.0
  IO.println "  ── per region ──"
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
    let nr := if rm > 1e-30 then ra / rm else 0.0
    if nm == "m" then gradNormRel := nr
    if nm == "loss/bc" then lossRel := nr
    if nm == "bnstat" then
      fwdRel := nr
      if exact != hi - lo then fwdExact := false
    IO.println s!"    {nm}: max|a-b| = {ra}, max|a| = {rm}, norm-rel = {nr}, \
bit-exact {exact}/{hi-lo}"

  -- ── per-PARAMETER localisation over `m` — naming the layer beats one global max ──
  let mut bad : Nat := 0
  let mut worst := ""
  let mut worstR : Float := 0.0
  let mut base := nP
  for pi in [0:net.paramShapes.size] do
    let dims := net.paramShapes[pi]!
    let cnt := dims.foldl (· * ·) 1
    let mut pd : Float := 0.0
    let mut pm : Float := 0.0
    for k in [0:cnt] do
      let i := base + k
      let d := (F32.read oa i.toUSize - F32.read ob i.toUSize).abs
      if d > pd then pd := d
      let a := (F32.read oa i.toUSize).abs
      if a > pm then pm := a
    let nr := if pm > 1e-30 then pd / pm else 0.0
    if nr > 1e-4 then bad := bad + 1
    if nr > worstR then worstR := nr; worst := s!"p{pi} {dims}"
    base := base + cnt
  IO.println s!"  params whose gradient disagrees (>1e-4 norm-rel): {bad}/{net.paramShapes.size}\
   (worst {worst} at {worstR})"

  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite}/{n} non-finite outputs"; IO.Process.exit 1
  if moved * 10 < n then
    IO.eprintln s!"DEGENERATE: only {moved}/{n} outputs are non-zero — the tie proves little"
    IO.Process.exit 1
  -- ── the GATE ──────────────────────────────────────────────────────────────────────────────
  -- 1. the FORWARD must be BIT-EXACT. `bnstat` is the batch μ/var of all 49 BN inputs, so it pins
  --    the entire forward chain. Any real mis-wiring lands here as a hard failure rather than a
  --    tolerance argument — and it also catches a BN running-stat slot wired to the wrong layer,
  --    which the arity check cannot see.
  -- 2. `%loss` must agree. Report-only, no theorem, no gradient path — §2b's standing reminder.
  -- 3. the backward must agree NORM-relative, never per-coordinate (handoff §3).
  if !fwdExact then
    IO.eprintln s!"TIE FAILED: the forward differs — `bnstat` (batch μ/var of all \
{net.bnChannels.size} BN inputs) is not bit-exact (norm-rel {fwdRel}), so the two renders do not \
compute the same forward pass"
    IO.Process.exit 1
  if lossRel > 1e-4 then
    IO.eprintln s!"TIE FAILED: the loss/bc region differs at norm-rel {lossRel} > 1e-4 against a \
bit-exact forward — the signature of a cotangent/loss mismatch (§2b shipped exactly that: plain CE \
against a smoothed-CE cotangent)"
    IO.Process.exit 1
  if gradNormRel > 1e-4 then
    IO.eprintln s!"TIE FAILED: gradient (m) norm-relative diff {gradNormRel} > 1e-4"
    IO.Process.exit 1
  IO.println s!"✓ renders TIE: forward BIT-EXACT (bnstat, {net.bnChannels.size} BN layers), \
%loss norm-rel {lossRel}, gradient norm-rel {gradNormRel} ≤ 1e-4, over all {n} returned floats"
