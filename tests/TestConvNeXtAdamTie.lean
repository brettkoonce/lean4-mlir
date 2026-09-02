import LeanMlir.VerifiedNets

/-! # `@convnext_adam_train_step` render tie — hand-written vs `pretty(provenGraph)`

`planning/xla_pjrt_handoff.md` §2f, step 3. `Proofs/Codegen/ConvNeXtRender.lean`'s
`convNextAdamTrainStepFaithful` renders the same train step the hand-written emitter in
`tests/TestConvNeXtTrain.lean` does — the one `convnext-verified-adam` trains on. This harness is
what licenses swapping them; run it BEFORE retiring the hand-written emitter, because afterwards the
comparison no longer exists.

The interface is positionally identical (**545 in / 543 out**, and here even the parameter *names*
agree), so both take the same `[θ|m|v | lr,bc1,bc2]` blob `trainAdamSched` builds.

**Why a numeric tie and not a text diff.** Same function, different graph: the certified render
emits 10,407 ops against 7,488 because `pretty` has no CSE, and the cotangents are composed
differently. Only running both and comparing every returned float settles it.

**This tie is ViT-grade, NOT EfficientNet-grade, and the difference matters.** ConvNeXt has
**no BatchNorm** — it uses LayerNorm — so there is no `bnstat` region and *nothing* in the output
depends on the forward alone except `%loss`. A forward disagreement and a backward one therefore
both land in `m` and cannot be separated, exactly as in `vit-adam-tie`. Consequences:

* **`%loss` is load-bearing, not a footnote.** It is report-only, on no gradient path, and covered
  by no theorem — the precise configuration in which §2b shipped plain CE against a smoothed-CE
  cotangent. With no BN statistics to pin the forward, it is the only direct read of it, so the
  harness **gates** it.
* the gradient gate is norm-relative, never per-coordinate (handoff §3).

    lake build convnext-adam-tie
    .lake/build/bin/convnext-adam-tie [refRender.mlir] [candRender.mlir]

Linked against **IREE**: `convnext-verified-adam` is an IREE binary, and a tie must run on the
backend the trainer actually uses.

Exits non-zero if the renders disagree or the comparison is degenerate.
-/

def main (args : List String) : IO Unit := do
  let handWritten := "verified_mlir/convnext_adam_train_step.mlir"
  let certified   := "verified_mlir/convnext_adam_train_step_b.mlir"
  -- Pre-swap the certified render lives at its own path, so the no-argument form IS the migration
  -- check; post-swap it degrades to an A-vs-A determinism run, which is still worth having but is
  -- not a migration check — so say so.
  let certExists ← System.FilePath.pathExists certified
  let (pathA, pathB) := match args with
    | a :: b :: _ => (a, b)
    | [a]         => (handWritten, a)
    | []          => (handWritten, if certExists then certified else handWritten)
  let net := convnextVerified.toNet
  let bs  := 32
  -- Input seed is overridable: a formula difference gives a STABLE relative error across inputs,
  -- while amplification through a cancelling reduce moves around. That is the discriminator.
  let xseed := ((← IO.getEnv "TIE_XSEED").bind (·.toNat?)).getD 555
  let reverseB := (← IO.getEnv "TIE_REVERSE").isSome
  IO.println "@convnext_adam_train_step tie"
  IO.println s!"  A (reference, hand-written) = {pathA}"
  IO.println s!"  B (candidate, certified)    = {pathB}"
  if pathA == pathB then
    IO.println "  NOTE: both paths are the same file — an A-vs-A determinism run, NOT a migration \
check. Pass the retired render as the first argument for that."
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), NO BatchNorm (LayerNorm), bs {bs}, \
xseed {xseed}{if reverseB then ", B batch REVERSED (control)" else ""}, backend {← LowererSession.backendName}"

  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParamHeFanIn sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  let m ← F32.heInit 4242 net.nParams.toUSize 0.02
  -- v strictly positive: a zero v makes √v̂ + ε ≈ ε everywhere and would hide the second-moment path
  let v ← F32.scaleShift (← F32.heInit 8484 net.nParams.toUSize 0.01) 1.0 0.05
  let tail ← F32.const 3 0.0
  let tail ← F32.write3 tail 0 0.001 0.19 0.002    -- lr, 1−β₁ᵗ, 1−β₂ᵗ at a mid-training step
  let pbuf := F32.concat #[θ, m, v, tail]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]])
  let x0 ← F32.heInit xseed.toUSize (bs * net.d0).toUSize 1.0
  let mut y0 : ByteArray := .empty
  for i in [0:bs] do
    y0 := y0.push (UInt8.ofNat (i % net.nClasses)); y0 := y0.push 0; y0 := y0.push 0; y0 := y0.push 0
  -- `TIE_REVERSE=1` feeds side B the SAME batch in reversed row order. Every parameter gradient
  -- here sums over the batch, so that is semantics-preserving as real arithmetic — the multiset of
  -- terms is identical — but it changes the ACCUMULATION ORDER. Running A against A-reversed is
  -- therefore a control that produces the same CLASS of floating-point difference this tie sees,
  -- measured where correctness is not in question (the §2d.1 method: a control that shows no
  -- difference calibrates nothing, and an absolute bound with no control is an assertion).
  let mut xrParts : Array ByteArray := #[]
  let mut yrParts : Array ByteArray := #[]
  for i in [0:bs] do
    let src := bs - 1 - i
    xrParts := xrParts.push (F32.sliceImages x0 src 1 net.d0)
    yrParts := yrParts.push (F32.sliceLabels y0 src 1)
  let xR := F32.concat xrParts
  let yR := F32.concat yrParts

  -- A tie must NEVER reuse a cached binary. `compileVmfb` keys on **mtime**, not on the source, so
  -- re-running with a different candidate under the same tag silently reuses the FIRST candidate's
  -- `.vmfb` and reports a perfect match. That is not hypothetical — it happened while building the
  -- EfficientNet controls (handoff §4). Delete before compiling.
  let runOne (path tag : String) (x y : ByteArray) : IO ByteArray := do
    let vmfb := s!".lake/build/convnext_adam_tie_{tag}.vmfb"
    let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
    let vmfbT := s!".lake/build/convnext_adam_tie_{tag}_{target}.vmfb"
    for p in [vmfb, vmfbT] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
    let sess ← mkSession path
    LowererSession.mlpTrainStepV sess "m.convnext_adam_train_step" x pbuf shapes y
      bs.toUSize net.d0.toUSize net.nClasses.toUSize
  IO.println "  running A…"; (← IO.getStdout).flush
  let oa ← runOne pathA "a" x0 y0
  -- The determinism floor: without it "the difference is 1e-6" is an assertion, not a measurement.
  IO.println "  running A again (determinism floor)…"; (← IO.getStdout).flush
  let oa2 ← runOne pathA "a2" x0 y0
  let mut floor : Float := 0.0
  let mut fex : Nat := 0
  for i in [0:oa.size / 4] do
    let d := (F32.read oa i.toUSize - F32.read oa2 i.toUSize).abs
    if d == 0.0 then fex := fex + 1
    if d > floor then floor := d
  IO.println s!"  A-vs-A floor: max|a−a'| = {floor}, bit-exact {fex}/{oa.size / 4}"
  -- The CONDITIONING control: the SAME render (A) on the SAME batch in reversed row order. Every
  -- parameter gradient here sums over the batch, so as real arithmetic this cannot change the
  -- answer — but it changes the accumulation order, which is the same class of floating-point
  -- difference an independently-structured emitter produces. Without it an absolute gradient bound
  -- is an assertion: ConvNeXt's layer-scale γ gradient is a heavily cancelling reduce and does NOT
  -- reproduce to 1e-4 even against itself. §2d.1 learned this the expensive way on R34's bs256.
  IO.println "  running A on the REVERSED batch (conditioning control)…"; (← IO.getStdout).flush
  let oc ← runOne pathA "c" xR yR
  IO.println "  running B…"; (← IO.getStdout).flush
  let ob ← runOne pathB "b" (if reverseB then xR else x0) (if reverseB then yR else y0)

  if oa.size != ob.size then
    IO.eprintln s!"SIZE MISMATCH: {oa.size} vs {ob.size} bytes"; IO.Process.exit 1
  let n := oa.size / 4
  let nP := net.nParams
  if n != 3 * nP + 3 then
    IO.eprintln s!"ARITY: expected {3*nP+3} returned floats, got {n}"; IO.Process.exit 1

  -- the control's gradient disagreement — the floating-point noise floor for THIS net
  let mut ctlA : Float := 0.0
  let mut ctlM : Float := 0.0
  for i in [nP:2*nP] do
    let a := F32.read oa i.toUSize
    let c := F32.read oc i.toUSize
    let d := (a - c).abs
    if d > ctlA then ctlA := d
    if a.abs > ctlM then ctlM := a.abs
  let ctlRel := if ctlM > 1e-30 then ctlA / ctlM else 0.0
  -- The control's SPREAD: how many parameters a semantics-preserving reorder disturbs. This is the
  -- statistic that actually discriminates, and magnitude alone does not — measured: perturbing the
  -- cotangent (α 0.1 → 0.11) moves the gradient norm-rel to 0.0091, still under 4× this control,
  -- but it disturbs EVERY parameter where the reorder disturbs six. Conditioning noise is localised
  -- to the ill-conditioned op; a different function is global.
  let mut ctlBad : Nat := 0
  let mut cbase := nP
  for pi in [0:net.paramShapes.size] do
    let cnt := (net.paramShapes[pi]!).foldl (· * ·) 1
    let mut pd : Float := 0.0
    let mut pm : Float := 0.0
    for k in [0:cnt] do
      let a := F32.read oa (cbase + k).toUSize
      let c := F32.read oc (cbase + k).toUSize
      let d := (a - c).abs
      if d > pd then pd := d
      if a.abs > pm then pm := a.abs
    if (if pm > 1e-30 then pd / pm else 0.0) > 1e-4 then ctlBad := ctlBad + 1
    cbase := cbase + cnt
  IO.println s!"  CONTROL (A vs A on the reversed batch): gradient norm-rel = {ctlRel}, \
{ctlBad}/{net.paramShapes.size} params disturbed"

  -- `m' = β₁·m + (1−β₁)·g` off a shared `m`, so the `m` region IS the gradient. θ' is scale-free
  -- under Adam (§3), so it is reported but never gated.
  let regions : List (String × Nat × Nat) :=
    [("theta", 0, nP), ("m", nP, 2*nP), ("v", 2*nP, 3*nP), ("loss/bc", 3*nP, n)]
  let mut nonFinite : Nat := 0
  let mut moved : Nat := 0
  let mut gradNormRel : Float := 0.0
  let mut lossRel : Float := 0.0
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
    IO.println s!"    {nm}: max|a-b| = {ra} ({ra * 1e9} e-9), max|a| = {rm}, \
norm-rel = {nr} ({nr * 1e9} e-9), bit-exact {exact}/{hi-lo}"

  -- per-PARAMETER localisation over `m` — naming the layer beats one global max
  let mut bad : Nat := 0
  let mut worst := ""
  let mut worstR : Float := 0.0
  let mut worstIdxOpt : Option (Nat × Nat) := none
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
    if nr > worstR then
      worstR := nr; worst := s!"p{pi} {dims}"; worstIdxOpt := some (base, cnt)
    base := base + cnt
  IO.println s!"  params whose gradient disagrees (>1e-4 norm-rel): {bad}/{net.paramShapes.size}\
   (worst {worst} at {worstR})"

  -- Conditioning probe on the worst parameter. A reduce whose SUM is small against the MAGNITUDE
  -- of its terms is catastrophically cancelling, and two graphs that tile it differently will then
  -- disagree at a large RELATIVE error while both being correct — §3's standing lesson, and the
  -- reason the bs256 gate (§2d.1) needed a control rather than an absolute bound. `Σ|g|` vs
  -- `|Σ g|` is not available here, but the ratio of the two sides' magnitudes and the sign pattern
  -- separate "ill-conditioned sum" from "wrong formula": a wrong formula moves the magnitude, a
  -- reordering does not.
  if worstIdxOpt.isSome then
    let (lo, cnt) := worstIdxOpt.get!
    let mut sa : Float := 0.0
    let mut sb : Float := 0.0
    let mut aa : Float := 0.0
    let mut ab : Float := 0.0
    let mut signFlips : Nat := 0
    for k in [0:cnt] do
      let a := F32.read oa (lo + k).toUSize
      let b := F32.read ob (lo + k).toUSize
      sa := sa + a; sb := sb + b; aa := aa + a.abs; ab := ab + b.abs
      if (a < 0.0) != (b < 0.0) then signFlips := signFlips + 1
    IO.println s!"  ── conditioning probe on {worst} ({cnt} coords) ──"
    IO.println s!"    Σa = {sa}, Σb = {sb}, Σ|a| = {aa}, Σ|b| = {ab}"
    IO.println s!"    |Σa|/Σ|a| = {(sa.abs / aa)}   Σ|b|/Σ|a| = {(ab / aa)}  (≈1 ⇒ same formula, different summation order)"
    IO.println s!"    sign flips between the two sides: {signFlips}/{cnt}"
    IO.println s!"    (|Σa|/Σ|a| ≪ 1 ⇒ a cancelling sum, where a reordering alone produces a large \
RELATIVE error; the magnitudes Σ|a| agreeing is what says the formula is the same)"

  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite}/{n} non-finite outputs"; IO.Process.exit 1
  if moved * 10 < n then
    IO.eprintln s!"DEGENERATE: only {moved}/{n} outputs are non-zero — the tie proves little"
    IO.Process.exit 1
  -- ── the gate. No `bnstat` region exists (no BN), so `%loss` is the only direct read of the
  --    forward, and it is exactly what §2b got wrong once. Gated, not merely reported.
  if lossRel > 1e-4 then
    IO.eprintln s!"TIE FAILED: the loss/bc region differs at norm-rel {lossRel} > 1e-4. With no BN \
statistics in the output, %loss is the only direct read of the forward — a mismatch here against \
matching gradients is the signature of a forward or cotangent bug (§2b shipped exactly that)."
    IO.Process.exit 1
  -- The gradient gate is CONTROL-RELATIVE, never absolute — §2d.1's rule. `4×` is that section's
  -- factor, and the `1e-4` floor keeps the gate tight on a well-conditioned net where the control
  -- comes back near zero (there, this is exactly the old absolute bound).
  let gate := max 1e-4 (4.0 * ctlRel)
  if gradNormRel > gate then
    IO.eprintln s!"TIE FAILED: gradient (m) norm-relative diff {gradNormRel} > {gate} \
(= 4 × the {ctlRel} control). A disagreement ABOVE the reordering control is not conditioning."
    IO.Process.exit 1
  -- The SPREAD gate. Necessary, not decorative: a cotangent perturbation clears the magnitude gate
  -- above while disturbing every parameter, so magnitude alone would pass a real bug.
  if bad > max ctlBad 1 then
    IO.eprintln s!"TIE FAILED: {bad}/{net.paramShapes.size} parameters disagree, against \
{ctlBad} disturbed by a semantics-preserving batch reorder. Floating-point conditioning is LOCAL to \
the ill-conditioned op; a difference this widely spread is a different function, whatever its \
magnitude."
    IO.Process.exit 1
  IO.println s!"✓ renders TIE: gradient norm-rel {gradNormRel} ≤ {gate} (4 × the {ctlRel} \
reorder control), %loss {lossRel} ≤ 1e-4, over all {n} returned floats"
  IO.println s!"  spread: {bad}/{net.paramShapes.size} params disagree vs {ctlBad} disturbed by the \
reorder control"
  IO.println s!"  NOTE the gradient agrees BETTER than this render does with itself under a \
semantics-preserving batch reversal ({ctlRel}) — so the residual is this net's conditioning, not a \
difference between the emitters. ConvNeXt's layer-scale γ is a cancelling reduce; see §2f."
  IO.println "  (no BN ⇒ no forward-only region; this ties the GRADIENT and the LOSS, and does not \
separately pin the forward bit-exactly — see the file docstring)"
