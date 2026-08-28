import LeanMlir.VerifiedNets
import LeanMlir.Proofs.Codegen.EfficientNetRender

/-! # Classifier dropout — the two gates its own identity checks cannot make

`recipe_gaps.md` gap C. EfficientNet-B0's reference sets `dropout := 0.2`
(`jax/MainEfficientNetImagenet.lean:68`) and no verified render had it until 2026-08-03.

The feature ships with the usual endpoint gates — `verified_mlir/` re-renders byte-identically with
dropout off, the forward is a byte-prefix of the train step, `Proofs.dropout_ones_id` says a ones
mask is the exact identity — and **every one of them is an identity check at the neutral value.**
`xla_pjrt_handoff.md` §0.4 finding 1 is about exactly that class: *a gate whose input makes the
intervention inert cannot test the intervention.* So this file is the two things those cannot say:

| gate | what it establishes | why no endpoint gate can |
|---|---|---|
| **A — the known answer** | the mask multiplies **per ELEMENT**, so this is dropout and not stochastic depth applied to the classifier | at a ones mask the two regularisers are the SAME FUNCTION. Only a non-uniform mask against a host-computed answer separates them |
| **W — the weight-gradient operand** | `∂L/∂W_d` reads the **dropped** activation, not the pooled one | at a ones mask the two activations are the same buffer, so the keep = 1 tie, the prefix audit and the ones-identity endpoint ALL pass on a render that has this wrong |

⚠⚠ **Gate W is the one that matters, and it exists because the same defect shipped once already.**
ConvNeXt's LayerScale γ gradient read the cotangent at the drop site and was fed the undropped one:
18 of 180 gradients wrong by a per-example factor, on the very parameter stochastic depth acts
through, found by tracing operands **by hand** because nothing in that feature's gate set could see
it (handoff §0.10). Dropout has the identical shape one net over — `dnW`'s `xName` argument — and
`scripts/fault_dropout_wgrad.py` is that defect, mechanised, so the gate has a control it is
verified to fail against.

    lake build dropout-tie
    CUDA_VISIBLE_DEVICES=0 .lake/build/bin/dropout-tie          # both gates
    .lake/build/bin/dropout-tie --op                            # gate A only (tiny, no artifact)
    .lake/build/bin/dropout-tie --op --break                    # gate A is falsifiable
    .lake/build/bin/dropout-tie --net                           # gate W only (no GPU at all)
    scripts/fault_dropout_wgrad.py verified_mlir/efficientnet_adamdo_train_step.mlir /tmp/f.mlir
    .lake/build/bin/dropout-tie --net /tmp/f.mlir               # gate W goes red; expect rc=1

**Why gate W is STRUCTURAL and not numeric, deliberately.** The defect is an operand choice, and a
byte check on the operand localises it exactly — where a numeric tie would report "some parameter
moved" and leave the reader to find which. §0.2 increment 4 makes the same call for the same reason:
*a byte tie says which form did it in one run, where a tolerance argument cannot.* It also means
gate W needs no GPU, no det shim and no determinism floor, so it runs in milliseconds on any box —
which matters, because §0.1 says this one cannot do long runs.

**What this does NOT establish.** Gate A is SITE-LOCAL: it drives the op through the same `pretty`
emitter the render uses, at small dims, against a closed form. It says the op multiplies per
element; it does not say the real net wires it before the classifier. Gate W is the whole-net half
and says exactly that — and says nothing about the arithmetic. The two are independent and both are
needed, the same split `TestDropPathTie` records for stochastic depth.
-/

open Proofs Proofs.StableHLO

/-- A float32 buffer from explicit values. Masks are tiny and must be exact. -/
private def mkVec (vals : Array Float) : IO ByteArray := do
  let mut cells : Array ByteArray := #[]
  for v in vals do cells := cells.push (← F32.const 1 v)
  pure (F32.concat cells)

/-- `(max |a−b|, max(|a|,|b|), bit-exact count)`. The magnitude is over BOTH buffers — a control
    that drives one side to identically zero would otherwise give a zero denominator and report a
    total collapse as agreement (`TestDropPathTie`'s `cmpBufs`, learned there). -/
private def cmpBufs (a b : ByteArray) (n : Nat) : Float × Float × Nat := Id.run do
  let mut d := 0.0; let mut m := 0.0; let mut e := 0
  for i in [0:n] do
    let u := F32.read a i.toUSize
    let w := F32.read b i.toUSize
    if u == w then e := e + 1
    if (u - w).abs > d then d := (u - w).abs
    if max u.abs w.abs > m then m := max u.abs w.abs
  (d, m, e)

private def rel (d m : Float) : Float := if m > 1e-30 then d / m else 0.0

private def nonFinite (a : ByteArray) (n : Nat) : Nat := Id.run do
  let mut c : Nat := 0
  for i in [0:n] do
    if !(F32.read a i.toUSize).isFinite then c := c + 1
  return c

private def die (msg : String) : IO α := throw (IO.userError msg)

/-- Compile fresh — delete both the bare and the backend-scoped `.vmfb` first (§4): `compileVmfb`
    keys its cache on the output path and an mtime, never on the source, so a second run under one
    tag silently reuses the first binary. That bites exactly when running a control. -/
private def freshSession (path tag : String) : IO LowererSession := do
  let vmfb := s!".lake/build/dropout_tie_{tag}.vmfb"
  let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
  for p in [vmfb, s!".lake/build/dropout_tie_{tag}_{target}.vmfb"] do
    if ← System.FilePath.pathExists p then IO.FS.removeFile p
  mkSession path

-- ════════════════════════════════════════════════════════════════
-- § GATE A — the known answer. Does the mask multiply PER ELEMENT?
-- ════════════════════════════════════════════════════════════════

private def OB : Nat := 8      -- examples
private def ON : Nat := 5      -- per-example width. ⚠ ON ≠ OB — see `opModule`

private def zOX : Vec (OB * ON) := fun _ => 0

/-- **The op, through the SAME `pretty` emitter the render uses** — one `dropoutB` over `%x` with
    the mask as a graph input, which is exactly the subtree `enetFwdChain` splices before the dense.

    ⚠ `ON ≠ OB` on purpose, and here it does MORE work than in `TestDropPathTie`. There the point
    was that a wrong-axis broadcast becomes a type error; here there is no broadcast at all, so the
    asymmetry instead means the mask buffer (40 floats) cannot be confused with a per-example one
    (8 floats) by any accident of shape. What remains to catch is the substantive error — a mask
    that is CONSTANT WITHIN each example, i.e. stochastic depth wearing dropout's types — and that
    is control C1 below, which is why the mask is deliberately non-uniform within every example. -/
private def opModule : String :=
  let go : StateM Nat String := do
    let (c, o) ← pretty OB (.dropoutB (N := OB) (n := ON) "%do" (fun _ => 0 : Vec (OB * ON))
                              (.operand "%x" zOX))
    pure (c ++ s!"    return {o} : {ty [OB, ON]}\n")
  let body : String := go.run' 0
  "module @m {\n" ++
  s!"  func.func @do(%x: {ty [OB, ON]}, %do: {ty [OB, ON]}) -> {ty [OB, ON]} " ++ "{\n" ++
  body ++ "  }\n}\n"

/-- The closed form: `y[k] = m[k] · x[k]`, every coordinate independent.

    **Bit-exact is the right bar by argument.** Both operands are float32, so their exact product
    needs at most 48 mantissa bits and is therefore exact in the float64 Lean computes it in;
    rounding that to float32 is by definition what float32 multiplication returns. Any difference at
    all is the render computing something else. ⚠ The mask is read back out of the buffer the device
    receives, never from the literal, so both sides see identical float32 values. -/
private def dropoutRef (x m : ByteArray) : IO ByteArray := do
  let mut cells : Array ByteArray := #[]
  for k in [0:OB * ON] do
    cells := cells.push (← F32.const 1 (F32.read m k.toUSize * F32.read x k.toUSize))
  pure (F32.concat cells)

/-- ⭐⭐ **Control C1 — THE WRONG REGULARISER.** Every element of example `j` scaled by that
    example's FIRST mask value, i.e. what stochastic depth computes: a `(B, 1, …, 1)` mask
    broadcast over the feature axis.

    This is the control this whole gate exists for. A render that drew `B` values and repeated each
    `n` times fills the same buffer, has the same shape, emits the same MLIR and trains — and the
    only thing that separates it from dropout is that its mask is constant within each example.
    `Proofs.dropPath_scales_uniformly` is that statement on the denotation side; this is it on
    device. It must NOT match, which is why `opMask` varies within every example. -/
private def dropoutRefPerExample (x m : ByteArray) : IO ByteArray := do
  let mut cells : Array ByteArray := #[]
  for j in [0:OB] do
    let mj := F32.read m (j * ON).toUSize
    for i in [0:ON] do
      cells := cells.push (← F32.const 1 (mj * F32.read x (j * ON + i).toUSize))
  pure (F32.concat cells)

/-- The mask under test: `B × n` values, three kinds in one buffer so one invoke covers three
    claims.

* **`0` and `1/keep`** — the values `F32.dropoutMask` ACTUALLY supplies. It emits
  `bernoulli(keep)/keep`, i.e. `0` or `1/0.8 = 1.25`. A gate written over `(0,1)` would test a
  range the feature never uses.
* **the endpoints `1.0` and `0.0`** — `dropout_ones_id` and `dropout_zeros_zero`, on device.
* **interior `0.5 / 0.25 / 0.75`** — exactly representable in binary, so the known answer is not
  testing the harness's own rounding.

⚠⚠ **It VARIES WITHIN each example, and that is the load-bearing property.** A mask that happened
to be constant per example would make control C1 agree with the gate, and the gate would then be
passing on a render that computes stochastic depth. The first value of each example is deliberately
never `0.0`, for `TestDropPathTie`'s reason: C1 compares against "everything scaled by `m[j·n]`",
which a zero there would turn into the trivially-different all-zero vector. -/
private def opMask (keep : Float) : Array Float := Id.run do
  let inv := 1.0 / keep
  let pattern : Array Float := #[inv, 0.0, 1.0, 0.5, 0.25]
  let mut out : Array Float := #[]
  for j in [0:OB] do
    -- rotate per example so no two examples share a pattern and none is uniform
    for i in [0:ON] do
      out := out.push pattern[(i + j) % ON]!
  return out

private def gateOp (doBreak : Bool) : IO Unit := do
  IO.println "── GATE A — the known answer: does the mask multiply PER ELEMENT?"
  let keep := (efficientnetVerified.dropoutKeep.map (·.1)).getD 0.8
  let maskVals := opMask keep
  IO.println s!"  B {OB}, n {ON} (n ≠ B on purpose), keep {keep}, backend {← LowererSession.backendName}"
  IO.println s!"  mask varies WITHIN every example — 1/keep = {1.0 / keep}, both endpoints, two \
interior values, rotated per example"
  let x ← F32.heInit 20260803 (OB * ON).toUSize 1.0
  let m ← mkVec maskVals
  let path := "/tmp/dropout_op.mlir"
  IO.FS.writeFile path opModule
  let sess ← freshSession path "op"
  let y ← LowererSession.forwardF32 sess "m.do" m (packShapes #[#[OB, ON]]) x
            (packXShape #[OB, ON]) OB.toUSize ON.toUSize
  let n := OB * ON
  if nonFinite y n > 0 then
    die "DEGENERATE: non-finite output — the gate proves nothing"

  -- ── the gate ──
  let ref ← dropoutRef x m
  let (d, mg, e) := cmpBufs y ref n
  IO.println s!"  emitted vs host `m[k]·x[k]` : max abs {d}   |ref|max {mg}   bit-exact {e}/{n}"
  if mg < 1e-6 then die "DEGENERATE: the reference is ~0 — the gate proves nothing"
  if e != n then
    die s!"GATE A FAILED: only {e}/{n} coordinates are bit-exact against the host product \
(max abs {d}, rel {rel d mg}). An f32 product is exact in f64, so any difference is the render \
computing a different function."

  -- ── the endpoints, read out separately because they are named theorems ──
  let cnt (p : Nat → Bool) : Nat := Id.run do
    let mut c := 0
    for k in [0:n] do if p k then c := c + 1
    return c
  let onesIdx := cnt (fun k => maskVals[k]! == 1.0)
  let onesOk := cnt (fun k => maskVals[k]! == 1.0 && F32.read y k.toUSize == F32.read x k.toUSize)
  let zeroIdx := cnt (fun k => maskVals[k]! == 0.0)
  let zeroOk := cnt (fun k => maskVals[k]! == 0.0 && F32.read y k.toUSize == 0.0)
  IO.println s!"  `dropout_ones_id`   on device: {onesOk}/{onesIdx} coordinates at m=1 equal x"
  IO.println s!"  `dropout_zeros_zero` on device: {zeroOk}/{zeroIdx} coordinates at m=0 are zero"
  if onesOk != onesIdx || zeroOk != zeroIdx then
    die "GATE A FAILED: an endpoint theorem does not hold on device"

  -- ── ⭐ C1: the wrong regulariser must NOT match ──
  let refPE ← dropoutRefPerExample x m
  let (dP, _, eP) := cmpBufs y refPE n
  IO.println s!"  ⚠ CONTROL C1 (per-EXAMPLE mask — i.e. stochastic depth on the classifier): \
max abs {dP} (rel {rel dP mg}), bit-exact {eP}/{n}"
  if rel dP mg < 1e-3 then
    die s!"CONTROL C1 FAILED: a per-example mask agrees with the render (rel {rel dP mg}). Either \
the test mask is constant within examples (it must not be) or the render is applying the mask \
per example — which is stochastic depth, not dropout. Either way the gate above means nothing."

  -- ── C2: prove the gate can go red ──
  if doBreak then
    let bad := maskVals.set! 7 (maskVals[7]! * 1.01)
    let mBad ← mkVec bad
    let refBad ← dropoutRef x mBad
    let (dX, _, eX) := cmpBufs y refBad n
    IO.println s!"  ⚠ CONTROL C2 (one mask cell 1% wrong): max abs {dX}, bit-exact {eX}/{n}"
    if eX == n then
      die "CONTROL C2 FAILED: a 1% wrong mask value still matches — the comparison is not reading \
the mask at all."
    IO.println s!"    ✓ fires, and on exactly the {n - eX} coordinate(s) that cell reaches"
  IO.println "  ✓ GATE A"

-- ════════════════════════════════════════════════════════════════
-- § GATE W — the weight-gradient operand. The one no identity gate can make.
-- ════════════════════════════════════════════════════════════════

/-- Read a whole file. -/
private def slurp (p : String) : IO String := IO.FS.readFile p

/-- All `%name` operands of the first `dot_general` on `line`, in order. -/
private def dotOperands (line : String) : List String :=
  ((line.splitOn "stablehlo.dot_general").getD 1 "").splitOn ","
    |>.map (·.trimAscii.toString) |>.filter (·.startsWith "%") |>.map (fun s => (s.splitOn " ").getD 0 s)

/-- ⭐⭐ **GATE W.** In a classifier-dropout render, three things must hold about ONE value — the
    dropout site's output — and each is a different way for the render to be wrong:

    1. the dropout site exists at all, as `multiply` against `%do` with no broadcast;
    2. the classifier dense reads the DROPPED value (contracting the feature axis, `[1] x [0]`);
    3. ⭐ the classifier WEIGHT GRADIENT reads the DROPPED value too (contracting the BATCH axis,
       `[0] x [0]`) — `∂L/∂W_d = Σ_b (dense input)_b ⊗ dy_b`.

    (3) is the whole point. (1) and (2) are what any reading of the render would check; (3) is the
    consumer of the displaced value that is easy to miss, because it lives in the BACKWARD while the
    op was spliced into the FORWARD. Handoff §0.10's carry-forward, verbatim: *when an op is
    spliced into a chain, list every consumer of the value it displaced.*

    ⚠ And note what is NOT checked here: the bias gradient, which reads only the cotangent and must
    be UNAFFECTED. It is asserted below for the same reason the others are — a render that scaled it
    too would be wrong in the opposite direction, and nothing else would notice. -/
private def gateNet (path : String) : IO Unit := do
  IO.println s!"── GATE W — the classifier weight gradient's operand"
  IO.println s!"  artifact {path}"
  let text ← slurp path
  let lines := (text.splitOn "\n").toArray

  -- ── 1. the dropout site ──
  let siteIdx := (List.range lines.size).filter (fun i =>
    (lines[i]!.splitOn "stablehlo.multiply").length > 1 && (lines[i]!.splitOn "%do").length > 1)
  if siteIdx.length != 2 then
    die s!"GATE W: expected exactly 2 `multiply` against %do (the forward site and the cotangent \
scale), found {siteIdx.length}. This render is not the shape this gate assumes — refusing rather \
than reporting a pass on a render it cannot read."
  let fwdLine := lines[siteIdx[0]!]!
  -- `%D = stablehlo.multiply %do, %G : tensor<...>`
  let lhs := ((fwdLine.splitOn "=").getD 0 "").trimAscii.toString
  let ops := ((fwdLine.splitOn "stablehlo.multiply").getD 1 "").splitOn ":" |>.getD 0 ""
  let opNames := (ops.splitOn ",").map (·.trimAscii.toString) |>.filter (·.startsWith "%")
  let pooled := (opNames.filter (· != "%do")).getD 0 ""
  if lhs.isEmpty || pooled.isEmpty then die s!"GATE W: cannot parse the dropout site: {fwdLine}"
  if (fwdLine.splitOn "broadcast_in_dim").length > 1 then
    die s!"GATE W: the dropout site BROADCASTS its mask — that is stochastic depth, not dropout"
  IO.println s!"  ① dropout site        : {lhs} = {pooled} ⊙ %do   (no broadcast)"

  -- ── 2 & 3. the two dot_generals that must read it ──
  let dots := (List.range lines.size).filter (fun i =>
    (lines[i]!.splitOn "stablehlo.dot_general").length > 1)
  let fwdDot := dots.filter (fun i =>
    (lines[i]!.splitOn "contracting_dims = [1] x [0]").length > 1 &&
    (dotOperands lines[i]!).getD 0 "" == lhs)
  -- ⚠ Batch contraction ALONE is not enough to identify it — EfficientNet has 33
  -- `contracting_dims = [0] x [0]` sites, because every squeeze-excite dense's weight gradient
  -- contracts the batch too. What singles out the CLASSIFIER's is that its first operand is one of
  -- the two candidate values: the dropped activation (correct) or the pooled one (the defect). So
  -- the filter names both and then decides between them, rather than locating the site first and
  -- reading its operand — which is what lets it REFUSE when neither appears, instead of silently
  -- matching an SE gradient and reporting a pass about the wrong parameter.
  let wDot := dots.filter (fun i =>
    (lines[i]!.splitOn "contracting_dims = [0] x [0]").length > 1 &&
    ((dotOperands lines[i]!).getD 0 "" == lhs || (dotOperands lines[i]!).getD 0 "" == pooled))
  if fwdDot.isEmpty then
    die s!"GATE W FAILED (②): no classifier dense reads the dropped value {lhs}. The dropout site \
exists but nothing downstream consumes it — the mask reaches nothing."
  IO.println s!"  ② classifier dense    : reads {lhs} — the DROPPED activation"
  if wDot.length != 1 then
    die s!"GATE W: expected exactly 1 batch-contracting dot_general (the classifier weight \
gradient), found {wDot.length}. Refusing rather than guessing which."
  let wLine := lines[wDot[0]!]!
  let wOperand := (dotOperands wLine).getD 0 ""
  if wOperand == pooled then
    die s!"GATE W FAILED (③): the classifier WEIGHT GRADIENT reads {wOperand} — the POOLED \
activation — where it must read {lhs}, the DROPPED one.\n\
    ∂L/∂W_d = Σ_b (dense input)_b ⊗ dy_b, and with dropout on the dense's input IS the dropped \
activation.\n\
    ⚠ This render trains and descends, and at a ones mask it is BIT-IDENTICAL to a correct one — \
so the keep = 1 tie, the prefix audit and `dropout_ones_id` all pass on it. That is why this gate \
is structural and why it exists.\n\
    line {wDot[0]! + 1}: {wLine.trimAscii.toString}"
  if wOperand != lhs then
    die s!"GATE W FAILED (③): the classifier weight gradient reads {wOperand}, which is neither \
the dropped value {lhs} nor the pooled one {pooled}. line {wDot[0]! + 1}: {wLine.trimAscii.toString}"
  IO.println s!"  ③ classifier W grad   : reads {lhs} — the DROPPED activation ⭐"

  -- ── the bias gradient must be UNTOUCHED, the opposite error ──
  let biasReduce := (List.range lines.size).filter (fun i =>
    (lines[i]!.splitOn "stablehlo.reduce").length > 1 &&
    (lines[i]!.splitOn "across dimensions = [0]").length > 1 &&
    (lines[i]!.splitOn lhs).length > 1)
  if !biasReduce.isEmpty then
    die s!"GATE W FAILED: the classifier BIAS gradient reads the dropped activation. It is \
Σ_b dy_b and depends on the mask only through dy — scaling it again double-counts the mask."
  IO.println s!"  ④ classifier b grad   : does NOT read {lhs} — correct, it is Σ_b dy_b"

  -- ── the cotangent scale, on the way down ──
  let bwdLine := lines[siteIdx[1]!]!
  IO.println s!"  ⑤ cotangent scale     : {((bwdLine.splitOn "=").getD 0 "").trimAscii.toString} — the same op \
at the same mask (`Proofs.dropout_vjp_is_self`)"
  IO.println "  ✓ GATE W"

def main (args : List String) : IO Unit := do
  let onlyOp  := args.contains "--op"
  let onlyNet := args.contains "--net"
  let doBreak := args.contains "--break"
  let cand := (args.filter (fun a => !a.startsWith "--")).getD 0
                "verified_mlir/efficientnet_adamdo_train_step.mlir"
  IO.println "══ classifier dropout — the gates its identity checks cannot make ══"
  if !onlyNet then gateOp doBreak
  if !onlyOp then gateNet cand
  IO.println "══ all requested gates green ══"
