import LeanMlir
import LeanMlir.Proofs.Codegen.MobileNetV4RenderB

/-! # MNv4 AdamW train-step structural smoke (`planning/mnv4_verified.md` phase 2)

The backward peer of `TestMnv4FwdSmoke`. It checks three things and emits a **batch-2** copy of
the train step for `scripts/grad_tie.py --net mnv4` to run — the committed artifact is B=32, which is
more than a CPU gradient check wants to carry.

⭐⭐ **The load-bearing check here is the FORWARD-PREFIX one**, and it is the one the rest of the
repo does not have. `planning/mnv4_verified.md` §3d(b) measured `mobilenetv2_fwd.mlir` sitting in a
*different BN world* from the Adam train step that trains it, and `scripts/regen_verified_mlir.sh
check` reported **green** anyway — because it only ever pairs a forward with the **SGD** train step,
never the Adam one, and those two happen to share a world. The forward that scores every quoted
verified number was therefore unaudited against the graph that trains it.

Here `@mnv4_fwd`, `@mnv4_fwd_eval` and `@mnv4_adam_train_step` all come from one `mnv4FwdChainB`
call, so the train step's forward region is *literally* the forward module's body — and this test
asserts that as a string, at the same batch. It cannot drift without failing at `lake build`.

⚠ What this does NOT check is that the gradient is right. Op counts and arities are blind to a
backward that masks a swish with `selectPos`, or differentiates an ExtraDW block as an FFN — both
type-check, both descend. That is `scripts/grad_tie.py --net mnv4`'s job, and §3g's record of a
derived-by-symmetry backward pad that "type-checked, produced the right shape, and would have
trained and descended" is why the numeric gate is not optional.
-/

open Proofs.StableHLO

def main : IO Unit := do
  let B := 2
  let nClasses := 10
  let m := mobilenetv4AdamTrainStepFaithfulB B nClasses "1.0e-05"
  IO.FS.writeFile ".lake/build/mnv4_adam_train_step_b2.mlir" m
  let lines := m.splitOn "\n"
  IO.println s!"  rendered {lines.length} lines"
  let mut bad := 0
  let chk (what : String) (got want : Nat) : IO Bool := do
    if got == want then IO.println s!"  ✓ {what}: {got}"; pure true
    else IO.println s!"  ✗ {what}: got {got}, want {want}"; pure false

  -- ── the interface contract: 233 params × (θ, m, v) + the scalars + 154 stat slots ──
  let nP := (mnv4ShapeList nClasses).length
  let nS := mnv4StatShapeList.length
  if !(← chk "parameters" nP 233) then bad := bad + 1
  if !(← chk "BN stat slots" nS 154) then bad := bad + 1
  -- inputs = %x + 3·233 + lr/bc1/bc2 + 154 + %onehot ; outputs = 3·233 + 3 + 154
  let hdr := (m.splitOn "func.func @").getD 1 ""
  let argSig := (hdr.splitOn ") -> (").getD 0 ""
  let nIn := (argSig.splitOn ": tensor").length - 1
  if !(← chk "func inputs" nIn (1 + 3*nP + 3 + nS + 1)) then bad := bad + 1
  let retLine := (lines.filter (fun l => (l.splitOn "    return ").length > 1)).getD 0 ""
  let nRet := ((retLine.splitOn " : ").getD 0 "").splitOn ", " |>.length
  if !(← chk "return values" nRet (3*nP + 3 + nS)) then bad := bad + 1

  -- ── the entry name must match `mnv4AdamVariant`, or the shim refuses the call ──
  -- ⚠ B=2 ⇒ variant "adam2"; only B=32 is unsuffixed. Derived, not spelled, so the two cannot
  -- drift the way a hardcoded name would.
  let entry := s!"@mnv4_{mnv4AdamVariant B 1}_train_step("
  if (m.splitOn entry).length > 1 then
    IO.println s!"  ✓ entry point {entry.dropRight 1}"
  else
    IO.println s!"  ✗ entry point missing/renamed (expected {entry.dropRight 1})"; bad := bad + 1

  -- ── ⭐ THE FORWARD-PREFIX GATE (see the module docstring) ──
  -- `mnv4FwdChainB` is called first inside the train step, so its fresh-name counter starts at 0
  -- exactly as the standalone forward's does, and the two code strings must be character-equal.
  let fwd := (mnv4FwdChainB B nClasses "1.0e-05" .train).run' 0
  if (m.splitOn fwd.code).length > 1 then
    IO.println s!"  ✓ train step contains @mnv4_fwd's body VERBATIM ({fwd.code.splitOn "\n" |>.length} lines)"
  else
    IO.println "  ✗ the train step's forward region is NOT the forward module's body"
    IO.println "    ⇒ the net that trains and the net that scores have diverged (§3d(b))."
    bad := bad + 1

  -- ── ⭐ THE SPEC'S BN LIST vs THE RENDER'S STAT SLOTS — the two-lists tie for running stats ──
  -- `mobilenetv4Verified.bnChannels` drives the driver's running-stat threading; `mnv4StatShapeList`
  -- drives what the train step RETURNS and what `@mnv4_fwd_eval` READS. They are separate readings
  -- of one layout (the spec is a light module and cannot import the renderer), so nothing but this
  -- ties them. A mismatch is silent at run time: the arities agree and the wrong layer's statistics
  -- flow into the wrong eval slot.
  let specBn := mobilenetv4Verified.bnChannels.toList
  let renderBn := (mnv4StatShapeList.map (fun (_, ds) => ds.headD 0)).toArray.toList
  -- the render carries mu AND var per layer, so halve it by taking every other entry
  let renderBnLayers := (renderBn.zipIdx.filterMap
    (fun (c, i) => if i % 2 == 0 then some c else none))
  if !(← chk "spec bnChannels == render stat widths (77 layers)"
        (if specBn == renderBnLayers then 1 else 0) 1) then
    IO.println s!"    spec {specBn.length} entries vs render {renderBnLayers.length}"
    bad := bad + 1

  -- ── the eval forward must read the SAME stat slots the train step returns ──
  let ev := mnv4FwdEvalFaithfulV B nClasses "1.0e-05"
  let mut missing := 0
  for (n, _) in mnv4StatShapeList do
    if (ev.splitOn (n ++ ":")).length ≤ 1 then missing := missing + 1
  if !(← chk "eval forward binds every stat slot (0 missing)" missing 0) then bad := bad + 1

  -- ── ⭐ THE ACTIVATION-PAIRING GATE: every activation is masked by its OWN backward ──
  -- The forward has 54 relu + 1 swish (`mnv4-fwd-smoke` pins those). So the whole train step must
  -- show 54 `maximum` (the relus) paired with 54 `select` (their `selectPosB` masks), and
  -- `logistic` exactly twice — once forward in the swish, once in `swishBackB`.
  --
  -- ⚠ This is what catches masking the swish site with `selectPos`: that renders 37 selects and
  -- ONE logistic, keeps every shape and count elsewhere, type-checks, and descends. `mnv4-fwd-smoke`
  -- cannot see it (it never looks at a backward) and neither can the arity checks above.
  let n (pat : String) : Nat := (lines.filter (fun l => (l.splitOn pat).length > 1)).length
  let nRelu := n "stablehlo.maximum"
  if !(← chk "relu forwards" nRelu 54) then bad := bad + 1
  if !(← chk "selectPos masks (= one per relu)" (n "stablehlo.select") nRelu) then bad := bad + 1
  if !(← chk "logistic (swish fwd + swishBack)" (n "stablehlo.logistic") 2) then bad := bad + 1

  if bad == 0 then
    IO.println "  ✓ mnv4 train step: arity, entry point, forward-prefix and stat binding all hold"
    IO.println "    ⚠ NOT a gradient check — run scripts/grad_tie.py --net mnv4 for that."
  else
    IO.println s!"  ✗ {bad} check(s) failed"
    throw (IO.userError "mnv4 train-step smoke FAILED")
