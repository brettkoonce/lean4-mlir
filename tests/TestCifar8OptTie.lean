import LeanMlir.VerifiedNets

/-! # cifar8 optimizer-render tie — six variants, one harness (handoff §2i)

    lake build cifar8-opt-tie
    .lake/build/bin/cifar8-opt-tie [w_][bn_]<adam|sgd|mom> \
      [<refRender.mlir> [<newRender.mlir>]]

Each variant shares ONE forward, backward and un-fused-gradient body with its siblings and one
packed `[θ|m|v|lr|bc1|bc2]` signature — **71 in / 69 out** with no BN, **119 in / 117 out** with it
(38 params rather than 22: + 8 BN γ and 8 BN β). The slug decomposes into three independent choices,
which is why one harness covers all **twelve** variants:

* an optional `w_` picks the **wide 2×512 dense head** — `cifar8w` is `cifar8` at `d1 = 512`, not a
  second net (§2i: the specs agree layer-for-layer up to the head width, and the committed wide BN
  AdamW artifact is byte-identical to the width sweep's `cifar8_bn_512` one);
* an optional `bn_` picks per-channel BatchNorm;
* the remainder picks the **gradient-recovery formula** below, which depends only on the optimizer.

Param count, shapes and the He/ones/zeros init kinds all come from the selected `VerifiedNetSpec`, so
nothing about a net is hardcoded here. The three together give the entry point and the default
artifact path, `@cifar8[w][_bn]_<opt>_train_step`.

## It gates the RECOVERED GRADIENT, never θ′ — and for SGD that is the whole ballgame

§2a-quinquies found this the hard way on the SGD renders: a train step returns `θ' = θ − lr·g`, and
θ' is dominated by `θ`, **the same input on both sides**. At the lr used here (1e-3) a *wholly wrong*
gradient still lands within `lr·|g|/|θ|` of a match, so a θ'-based tie is close to meaningless.
Every variant's gradient is exactly recoverable from its own outputs, so that is what is compared:

| slug | recovery | from |
|---|---|---|
| `adam` | `g = (m' − β₁·m)/(1−β₁)` | the first moment |
| `sgd`  | `g = (θ − θ')/lr` | the only output that moves |
| `mom`  | `g = v' − μ·v` | the velocity, `v' = μv + g` |

All three are exact algebra on values this harness supplies, not approximations. (`momVNext_v_zero`
and `adamMNextF`'s `m' = (1−β₁)g` are the same observation, and it is what `convnext-shard-check`
relies on too.)

## The passthrough check, which is free and catches a real bug class

`sgd` must return `m' = m` and `v' = v` **bit-exactly**, and `mom` must return `m' = m` bit-exactly —
the packed signature is shared with AdamW precisely so the driver never changes, and a tail that
silently zeroed or dropped a moment slot would still produce a plausible θ'. Gated, not reported.

Deletes its `.vmfb` before every compile (§4's false-PASS hazard: `compileVmfb` keys on the OUTPUT
path plus an mtime, never the source, so running one binary twice with different candidates silently
reuses the first one's binary — which is exactly what comparing a staged render against an incumbent
looks like). `cifar8-adam-tie` still has that hazard; prefer this harness.
-/

def main (args : List String) : IO Unit := do
  let (slug, rest) := match args with
    | s :: r => (s, r)
    | []     => ("adam", [])
  -- The slug decomposes into THREE independent choices, which is why one harness covers all twelve:
  -- an optional `w_` (the wide 2×512 head — `cifar8w` is `cifar8` at d1=512, §2i), an optional `bn_`,
  -- and the optimizer. Net, artifact path and entry point all follow.
  -- `String.drop` returns a `String.Slice` on this toolchain (Lean 4.32), hence the `.toString`.
  let wide := slug.startsWith "w_"
  let afterW := if wide then (slug.drop 2).toString else slug
  let bn := afterW.startsWith "bn_"
  let optName := if bn then (afterW.drop 3).toString else afterW
  if !["adam", "sgd", "mom"].contains optName then
    IO.eprintln s!"unknown slug '{slug}' — expected an optional w_ then an optional bn_ then \
adam | sgd | mom (e.g. adam, bn_mom, w_sgd, w_bn_adam)"
    IO.Process.exit 1
  let netSlug := "cifar8" ++ (if wide then "w" else "") ++ (if bn then "_bn" else "") ++ "_" ++ optName
  let dflt := s!"verified_mlir/{netSlug}_train_step.mlir"
  let (pathA, pathB) := match rest with
    | a :: b :: _ => (a, b)
    | [a]         => (a, dflt)
    | []          => (dflt, dflt)
  let net := (if wide then (if bn then cifar8wBnVerified else cifar8wVerified)
                      else (if bn then cifar8BnVerified else cifar8Verified)).toNet
  let bs  := 128
  let nP  := net.nParams
  let lr  := 0.001; let β₁ := 0.9; let μ := 0.9
  IO.println s!"@{netSlug}_train_step tie: A={pathA}  B={pathB}"
  IO.println s!"  {net.specs.size} params ({nP} floats), bs {bs}, backend {← LowererSession.backendName}"

  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  let m ← F32.heInit 4242 nP.toUSize 0.02
  -- v must be strictly positive for adam (it sits under a sqrt); for mom it is the incoming
  -- velocity and any value works, so one buffer serves all three variants.
  let v ← F32.scaleShift (← F32.heInit 8484 nP.toUSize 0.01) 1.0 0.05
  let tail ← F32.const 3 0.0
  let tail ← F32.write3 tail 0 lr 0.19 0.002
  let pbuf := F32.concat #[θ, m, v, tail]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]])
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat (i % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0

  -- The batch, reversed. Every parameter gradient here is a SUM over the batch and `%loss` is a
  -- mean over it, so as real arithmetic a row permutation cannot change either — but it changes the
  -- ORDER the 128-wide reductions accumulate in. Running the reference render against itself on this
  -- is the §2f-bis reorder control: the same CLASS of floating-point difference the two emitters
  -- have, measured where correctness is not in question.
  let xR := F32.concat ((Array.range bs).map (fun r => F32.sliceImages x (bs - 1 - r) 1 net.d0))
  let mut yR : ByteArray := .empty
  for r in [0:bs] do
    yR := yR.push (UInt8.ofNat ((bs - 1 - r) % net.nClasses)); yR := yR.push 0
    yR := yR.push 0; yR := yR.push 0

  let runOne (path tag : String) (xb yb : ByteArray) : IO ByteArray := do
    for p in [s!".lake/build/c8_opt_{slug}_{tag}.vmfb",
              s!".lake/build/c8_opt_{slug}_{tag}_{((← IO.getEnv "IREE_BACKEND").getD "cuda")}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
    let sess ← mkSession path
    LowererSession.mlpTrainStepV sess s!"m.{netSlug}_train_step" xb pbuf shapes yb
      bs.toUSize net.d0.toUSize net.nClasses.toUSize
  let oa ← runOne pathA "a" x y
  let ob ← runOne pathB "b" x y
  -- A against ITSELF on the reversed batch. Skippable, because it costs a third compile+run.
  let oc ← if (← IO.getEnv "TIE_SKIP_REORDER").isSome then pure oa
           else runOne pathA "c" xR yR
  if oa.size != ob.size then
    IO.eprintln s!"SIZE MISMATCH: {oa.size} vs {ob.size} bytes"; IO.Process.exit 1

  -- Recover the gradient from one side's outputs — exact algebra, see the table above.
  let gradOf (o : ByteArray) (i : Nat) : Float :=
    match optName with
    | "adam" => (F32.read o (nP + i).toUSize - β₁ * F32.read m i.toUSize) / (1.0 - β₁)
    | "sgd"  => (F32.read θ i.toUSize - F32.read o i.toUSize) / lr
    | _      => F32.read o (2*nP + i).toUSize - μ * F32.read v i.toUSize

  -- The RAW output slot each recovery reads, before the recovery's amplification. Reporting it is
  -- what makes the three variants' numbers comparable: recovery is exact algebra but it MULTIPLIES
  -- the output-level difference by `1/(1−β₁)` = 10 for adam, `1/lr` = 1000 for sgd, 1 for mom. So a
  -- single shared graph difference shows up 100× larger in sgd's recovered gradient than in adam's,
  -- and without this row sgd reads like a near-miss when it is the same disagreement seen through a
  -- bigger lens. Gating stays on the recovered gradient — this row is for interpreting it.
  let (slotNm, slotOff, slotGain) := match optName with
    | "adam" => ("m'", nP, 1.0 / (1.0 - β₁))
    | "sgd"  => ("θ'", 0, 1.0 / lr)
    | _      => ("v'", 2*nP, 1.0)
  let mut sAbs : Float := 0.0; let mut sMag : Float := 0.0; let mut sExact : Nat := 0
  for i in [0:nP] do
    let a := F32.read oa (slotOff + i).toUSize; let b := F32.read ob (slotOff + i).toUSize
    if a == b then sExact := sExact + 1
    if (a - b).abs > sAbs then sAbs := (a - b).abs
    if a.abs > sMag then sMag := a.abs

  -- One statistic function, used for BOTH the test and the reorder control: max norm-relative
  -- gradient difference, and the per-PARAMETER spread (how many params disagree above 1e-4 against
  -- their OWN scale). Measuring the control with different code would not be a control.
  let statsD (u w : ByteArray) : Float × Nat × List Nat := Id.run do
    let mut aMax : Float := 0.0; let mut mMax : Float := 0.0
    let mut sp : Nat := 0; let mut pOff : Nat := 0; let mut which : List Nat := []
    let mut pIdx : Nat := 0
    for sh in net.paramShapes do
      let sz := sh.foldl (· * ·) 1
      let mut pAbs : Float := 0.0; let mut pMag : Float := 0.0
      for i in [pOff:pOff+sz] do
        let ga := gradOf u i; let gb := gradOf w i
        if (ga - gb).abs > pAbs then pAbs := (ga - gb).abs
        if ga.abs > pMag then pMag := ga.abs
      if pAbs > aMax then aMax := pAbs
      if pMag > mMax then mMax := pMag
      if pMag > 1e-30 && pAbs / pMag > 1e-4 then
        sp := sp + 1; which := which ++ [pIdx]
      pOff := pOff + sz; pIdx := pIdx + 1
    pure (if mMax > 1e-30 then aMax / mMax else 0.0, sp, which)
  let stats (u w : ByteArray) : Float × Nat := let (r, sp, _) := statsD u w; (r, sp)
  let (ctlRel, ctlSpread, ctlWhich) := statsD oa oc

  let mut gAbs : Float := 0.0; let mut gMag : Float := 0.0
  let mut nonFinite : Nat := 0; let mut movedG : Nat := 0
  -- Count EXACT coordinates. `Float.toString` gives six decimals, so a genuine 3e-8 prints as
  -- "0.000000" and reads as bit-exact when it is not — §2e-bis hit exactly that, and the count is
  -- what distinguishes "bit-exact" from "prints as zero".
  let mut exactG : Nat := 0
  for i in [0:nP] do
    let ga := gradOf oa i; let gb := gradOf ob i
    if !ga.isFinite || !gb.isFinite then nonFinite := nonFinite + 1
    if ga.abs > 1e-9 then movedG := movedG + 1
    if ga.abs > gMag then gMag := ga.abs
    let d := (ga - gb).abs
    if d == 0.0 then exactG := exactG + 1
    if d > gAbs then gAbs := d
  let gRel := if gMag > 1e-30 then gAbs / gMag else 0.0

  let (_, spread, spreadWhich) := statsD oa ob

  -- passthrough slots: which of m'/v' must equal their INPUT bit-exactly
  let passthru : List (String × Nat × ByteArray) := match optName with
    | "sgd" => [("m", nP, m), ("v", 2*nP, v)]
    | "mom" => [("m", nP, m)]
    | _     => []
  let mut ptBad : List String := []
  for (nm, off, src) in passthru do
    let mut bad := 0
    for i in [0:nP] do
      if F32.read oa (off + i).toUSize != F32.read src i.toUSize then bad := bad + 1
    if bad != 0 then ptBad := ptBad ++ [s!"{nm} ({bad}/{nP} slots changed)"]

  let la := F32.read oa (3*nP).toUSize; let lb := F32.read ob (3*nP).toUSize
  IO.println s!"  ── recovered gradient ({nP} coords) ──"
  IO.println s!"    max|gA−gB| = {gAbs} ({gAbs * 1e9} e-9), max|gA| = {gMag}, \
norm-rel = {gRel} ({gRel * 1e9} e-9), bit-exact {exactG}/{nP}"
  IO.println s!"    non-zero gradients: {movedG}/{nP},  spread {spread}/{net.specs.size} params above 1e-4"
  IO.println s!"  ── raw {slotNm} slot (what the recovery reads, gain {slotGain}×) ──"
  IO.println s!"    max|{slotNm}A−{slotNm}B| = {sAbs} ({sAbs * 1e9} e-9), max|{slotNm}| = {sMag}, \
bit-exact {sExact}/{nP}"
  IO.println s!"  ── REORDER CONTROL (A vs A, batch reversed — semantics-preserving) ──"
  IO.println s!"    norm-rel = {ctlRel} ({ctlRel * 1e9} e-9), spread {ctlSpread}/{net.specs.size} \
params{if (← IO.getEnv "TIE_SKIP_REORDER").isSome then "   ⚠ SKIPPED (TIE_SKIP_REORDER)" else ""}"
  -- WHICH params, not just how many: the ill-conditioned ones should be the SAME set in the test and
  -- the control. A test set that is a subset of the control's is the strongest form of this evidence.
  if spread > 0 || ctlSpread > 0 then
    IO.println s!"    param indices above 1e-4 — test {spreadWhich}, control {ctlWhich}"
  IO.println s!"  %loss  A = {la}  B = {lb}  (abs diff {(la-lb).abs})"
  if !passthru.isEmpty then
    IO.println s!"  passthrough slots checked: {passthru.map (·.1)} → \
{if ptBad.isEmpty then "all bit-exact" else toString ptBad}"

  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite}/{nP} non-finite recovered gradients"; IO.Process.exit 1
  if movedG * 10 < nP then
    IO.eprintln s!"DEGENERATE: only {movedG}/{nP} gradients are non-zero — the tie proves little"
    IO.Process.exit 1
  if !ptBad.isEmpty then
    IO.eprintln s!"TIE FAILED: passthrough slots are NOT bit-exact: {ptBad}. The packed \
[θ|m|v] signature is shared with AdamW so the driver never changes; a tail that drops or zeroes a \
moment slot still produces a plausible θ', which is why this is gated."
    IO.Process.exit 1
  if (la - lb).abs > 1e-4 then
    IO.eprintln s!"TIE FAILED: %loss differs by {(la-lb).abs} > 1e-4"; IO.Process.exit 1
  -- Magnitude: the ABSOLUTE 1e-4 floor, or 4× the reorder control where the control is the larger
  -- — §2d.1's rule. A control that shows no difference calibrates nothing, hence the floor rather
  -- than a bare multiple: on the no-BN nets the control comes back bit-exact and 4×0 would demand
  -- bit-exactness of a merely-correct render.
  let magGate := max 1e-4 (4.0 * ctlRel)
  if gRel > magGate then
    IO.eprintln s!"TIE FAILED: recovered-gradient norm-rel {gRel} > {magGate} (max of the 1e-4 \
floor and 4× the reorder control's {ctlRel})"
    IO.Process.exit 1
  -- Spread: at most as many params as the reorder control disturbs. §2f-bis measured that a
  -- control-relative MAGNITUDE gate alone passes a deliberately perturbed cotangent, because
  -- floating-point conditioning is LOCAL to the ill-conditioned op while a different function is
  -- GLOBAL. This is the check that separates them, and it is why the gate is control-relative and
  -- not the absolute 1e-4 per-param bound a first draft of this harness used — that bound fails the
  -- REAL tie on the BN nets, whose BN γ/β gradients are cancelling reduces over 128×S terms.
  if spread > ctlSpread then
    IO.eprintln s!"TIE FAILED: {spread}/{net.specs.size} parameters disagree above 1e-4 against \
their own scale, where the reorder control disturbs only {ctlSpread}. Conditioning is LOCAL to the \
ill-conditioned op; a different function is GLOBAL."
    IO.Process.exit 1
  IO.println s!"✓ {slug} renders TIE — recovered gradient norm-rel {gRel} ≤ {magGate}, spread \
{spread}/{net.specs.size} params ≤ the control's {ctlSpread}, %loss within 1e-4, passthrough slots \
bit-exact, over {movedG} non-degenerate gradient coords"
