import LeanMlir.Proofs.Codegen.StableHLO

/-! # The RMSProp `SHlo` op emits TENSORFLOW's ε placement, and the guard is shown to catch the
    textbook one

`SHlo.rmsBufNextF` denotes `Proofs.rmsBufNext` (`rmsBufNextF_faithful`, by `rfl`), so the
*denotation* side of ε-inside-the-sqrt is closed by construction. This file closes the other half:
that the **emitted text** puts it there too.

**Why this needs its own guard rather than riding the numeric tie.** The two spellings —
`g/√(s'+ε)` and `g/(√s' + ε)` — differ by a factor of `1/√ε` when the running mean-square is small
(`Proofs.rmsBufNext_eps_placement_at_zero`). At **EfficientNet's ε = 1e-3 that is 31.6×**; at
**MobileNetV2's ε = 1.0 it is exactly 1×**. So a render that got the placement wrong would be
**invisible on MobileNetV2 and wrong on EfficientNet** — the two nets in this repo that use RMSProp
sit on opposite sides of the difference. A green mnv2 tie cannot license the enet render, and a
structural check is the cheap thing that covers both.

This is the emit-side twin of the denotation theorem, and the same split `tests/TestAdamOpTie.lean`
makes for AdamW: `den` says what the graph *means*, this says what the text *is*.

    lake env lean tests/TestRmsPropOpTie.lean

⚠ Fails via `throw`, never `IO.Process.exit` — under `#eval`/`lake env lean` the elaborator buffers
output and `exit` discards every diagnostic, so you get a bare status and no idea what broke (§4).
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 8
private def DS : List Nat := [4, 4]
private def N  : Nat := 16          -- DS.prod

private def z : Vec N := fun _ => 0

private def render (g : SHlo N) : String := (pretty BS g).run' (0, []) |>.1

private def gradOperand : SHlo N := .operand "%g" z

/-- One emitted line, parsed: `(lhs, verb, %-operands)`. Splitting on `" = stablehlo."` is what
    makes this safe against `dims = []`, which also contains `" = "`. -/
private def parseLine (l : String) : Option (String × String × List String) :=
  match l.splitOn " = stablehlo." with
  | [lhs, rest] =>
      let body := (rest.splitOn " : ").headD rest
      match body.splitOn " " with
      | verb :: argsRaw =>
          -- splitting on " " already yields whitespace-free tokens, so stripping the operand
          -- separator is the only cleanup needed (and `String.trim` is deprecated besides).
          let args := (argsRaw.map fun a => a.replace "," "").filter fun a => a.startsWith "%"
          -- the lhs arrives indented; take its one `%`-token rather than trimming.
          match (lhs.splitOn " ").filter (fun t => t.startsWith "%") with
          | [name] => some (name, verb, args)
          | _ => none
      | [] => none
  | _ => none

private def lines (s : String) : List (String × String × List String) :=
  (s.splitOn "\n").filterMap parseLine

/-- The `stablehlo.<verb>` sequence — the SSA-name-independent skeleton. -/
private def verbs (s : String) : List String := (lines s).map (fun t => t.2.1)

/-- The line defining an SSA name. -/
private def defOf (s : String) (ssa : String) : Option (String × String × List String) :=
  (lines s).find? fun t => t.1 == ssa

/-- ▶ **THE FIDELITY CHECK.** Walk back from the `sqrt` and insist its operand is an `add` whose
    right-hand side is the broadcast of `%eps`.

    That is exactly "ε went in BEFORE the root". The textbook spelling roots the mean-square first
    and adds ε to the result, so its `sqrt` operand is the mean-square `add` (whose right operand is
    a `multiply`, not the ε broadcast) — and this returns an error there. -/
private def epsIsInsideSqrt (s : String) : Except String Unit := do
  let some (_, _, sqrtArgs) := (lines s).find? (fun t => t.2.1 == "sqrt")
    | throw "no stablehlo.sqrt in the rendered block"
  let some rootArg := sqrtArgs.head?
    | throw "stablehlo.sqrt has no operand"
  let some (_, addVerb, addArgs) := defOf s rootArg
    | throw s!"the sqrt operand {rootArg} is a function argument, not a computed value"
  if addVerb != "add" then
    throw s!"the sqrt operand {rootArg} is defined by `{addVerb}`, expected `add` (of s' and ε)"
  let some epsCandidate := addArgs.getLast?
    | throw "the add feeding the sqrt has no operands"
  let some (_, bcVerb, bcArgs) := defOf s epsCandidate
    | throw s!"{epsCandidate} is not a computed value"
  if bcVerb != "broadcast_in_dim" then
    throw s!"expected ε broadcast feeding the sqrt's add, got `{bcVerb}` — this is the TEXTBOOK \
             spelling g/(√s' + ε), NOT TensorFlow's g/√(s' + ε)"
  if bcArgs != ["%eps"] then
    throw s!"the add feeding the sqrt broadcasts {bcArgs}, not %eps"
  pure ()

-- ════════════════════════════════════════════════════════════════
-- § The renders under test
-- ════════════════════════════════════════════════════════════════

private def rmsBuf : String :=
  render (.rmsBufNextF "%sq" "%buf" "%rho" "%orho" "%mu" "%eps" DS 0 0 0 z z gradOperand)

/-- The mean-square slot as the render actually emits it: the EXISTING Adam second-moment op at
    `β₂ := ρ` (`Proofs.rmsSqNext_eq_adamVNext`, and `adamVNextF_as_rmsSqNext` on the `den` side). -/
private def sqSlot : String :=
  render (.adamVNextF "%sq" "%rho" "%orho" DS 0 z gradOperand)

/-- ⚠ **THE NEGATIVE CONTROL — hand-built, because no vanilla op exists to render.** Textbook
    RMSProp: root the mean-square, THEN add ε. Byte-plausible MLIR with the same verb multiset as
    the real block; only the order of `sqrt` and the ε `add` differs. If `epsIsInsideSqrt` does not
    reject this, it is not checking anything. -/
private def vanillaControl : String :=
  "    %w1 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<4x4xf32>\n" ++
  "    %w2 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<4x4xf32>\n" ++
  "    %w3 = stablehlo.multiply %w1, %sq : tensor<4x4xf32>\n" ++
  "    %w4 = stablehlo.multiply %g, %g : tensor<4x4xf32>\n" ++
  "    %w5 = stablehlo.multiply %w2, %w4 : tensor<4x4xf32>\n" ++
  "    %w6 = stablehlo.add %w3, %w5 : tensor<4x4xf32>\n" ++
  "    %w7 = stablehlo.sqrt %w6 : tensor<4x4xf32>\n" ++
  "    %w8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<4x4xf32>\n" ++
  "    %w9 = stablehlo.add %w7, %w8 : tensor<4x4xf32>\n" ++
  "    %w10 = stablehlo.divide %g, %w9 : tensor<4x4xf32>\n" ++
  "    %w11 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<4x4xf32>\n" ++
  "    %w12 = stablehlo.multiply %w11, %buf : tensor<4x4xf32>\n" ++
  "    %w13 = stablehlo.add %w12, %w10 : tensor<4x4xf32>\n"

/-- The op sequence `rmsBufNextF` is contracted to emit, in order — `μ·b + g/√(ρ·s + (1−ρ)·g² + ε)`
    read left to right. Pinned literally so a reordering of the emitter is a hard failure here
    rather than a number in a tie three steps downstream. -/
private def expectedVerbs : List String :=
  ["broadcast_in_dim", "broadcast_in_dim", "multiply", "multiply", "multiply", "add",
   "broadcast_in_dim", "add", "sqrt", "divide", "broadcast_in_dim", "multiply", "add"]

def main : IO Unit := do
  let mut fails : Array String := #[]

  -- ── 1. the emitted op sequence is the contracted one ──
  let vs := verbs rmsBuf
  if vs == expectedVerbs then
    IO.println s!"  ✓ verb sequence ({vs.length} ops) matches the RMSProp contract"
  else
    fails := fails.push s!"verb sequence\n      expected: {expectedVerbs}\n      got:      {vs}"

  -- ── 2. ▶ ε is INSIDE the sqrt (the whole fidelity claim) ──
  match epsIsInsideSqrt rmsBuf with
  | .ok _ => IO.println "  ✓ ε enters BEFORE the sqrt — TensorFlow's placement"
  | .error e => fails := fails.push s!"ε placement in the rendered op: {e}"

  -- ── 3. the control: the guard must REJECT the textbook spelling ──
  match epsIsInsideSqrt vanillaControl with
  | .error e =>
      IO.println s!"  ✓ CONTROL fires on the textbook spelling — {e}"
  | .ok _ =>
      fails := fails.push
        "CONTROL DID NOT FIRE: the ε-placement check accepted textbook RMSProp, so it is \
         checking nothing. This guard is vacuous until it rejects `vanillaControl`."

  -- ── 4. the reuse claim, emit-side: the mean-square slot IS the Adam op, and its text is the
  --      prefix `rmsBufNextF` recomputes internally (SHlo is single-result; XLA CSEs the copy) ──
  let sqVs := verbs sqSlot
  if sqVs.isPrefixOf vs then
    IO.println s!"  ✓ adamVNextF at ρ is the mean-square prefix ({sqVs.length} ops) of rmsBufNextF"
  else
    fails := fails.push s!"adamVNextF at ρ is not a verb-prefix of rmsBufNextF\n      \
                           adamVNextF: {sqVs}\n      rmsBufNextF: {vs}"

  -- ── 5. the mean-square really reads %sq and the buffer %buf, not each other's slot ──
  let ops := lines rmsBuf
  let readsSq  := ops.any fun t => t.2.1 == "multiply" && t.2.2.contains "%sq"
  let readsBuf := ops.any fun t => t.2.1 == "multiply" && t.2.2.contains "%buf"
  if readsSq && readsBuf then
    IO.println "  ✓ reads %sq (mean-square) and %buf (momentum) in their own slots"
  else
    fails := fails.push s!"slot wiring: readsSq={readsSq} readsBuf={readsBuf}"

  if fails.isEmpty then
    IO.println "✓ RMSProp op emit tie: all 5 checks pass (control fired)"
  else
    throw <| IO.userError <|
      "✖ RMSProp op emit tie FAILED:\n" ++ String.intercalate "\n" (fails.toList.map ("    • " ++ ·))

#eval main
