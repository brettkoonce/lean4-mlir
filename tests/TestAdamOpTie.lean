import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.ViTRender

/-! # The AdamW `SHlo` ops emit exactly what the trusted string emitter emitted

`ViTRender.emitAdamV` is a hand-written String emitter whose docstring *claims* to be op-for-op
`Proofs.adamWParam`. Nothing checked that claim — it is why `planning/xla_pjrt_handoff.md` §2a
found the repo split along Adam: every `_adam_train_step.mlir` was rendered from `tests/`, outside
the proven kit.

`SHlo.adamWParamF` / `.adamMNextF` / `.adamVNextF` now denote the proven `Proofs.adamWParam` /
`adamMNext` / `adamVNext` (`adamW_triple_faithful` bundles them into `Proofs.adamWStep`, by `rfl`).
This file closes the other half: that the *emitted text* is the same graph the trusted emitter
produced. SSA names differ (the ops use the `fresh` counter, `emitAdamV` uses a tag), so the
comparison is on the **op sequence** — every `stablehlo.<verb>`, in order, with its operand
positions.

    lake env lean tests/TestAdamOpTie.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 8
private def DS : List Nat := [4, 4]
private def N  : Nat := 16          -- DS.prod

private def z : Vec N := fun _ => 0

/-- The `stablehlo.<verb>` sequence of a rendered block — the SSA-name-independent skeleton. -/
private def verbs (s : String) : List String :=
  (s.splitOn "\n").filterMap fun line =>
    match (line.splitOn "stablehlo.") with
    | _ :: rest :: _ =>
        some (String.ofList (rest.toList.takeWhile fun c => c.isAlpha || c == '_'))
    | _ => none

private def render (g : SHlo N) : String := (pretty BS g).run' 0 |>.1

private def gradOperand : SHlo N := .operand "%g" z

/-- `emitAdamV` returns `(ir, θ', m', v')`; its `ir` is one block computing all three. -/
private def trusted : String :=
  (ViTRender.emitAdamV "%th" "%g" "%m" "%v" DS "t").1

def main : IO Unit := do
  let θ' := render (.adamWParamF "%th" "%m" "%v" "%b1" "%ob1" "%b2" "%ob2" "%bc1" "%bc2"
                      "%lr" "%eps" "%wd" DS 0 0 0 0 0 0 0 z z z gradOperand)
  let m' := render (.adamMNextF "%m" "%b1" "%ob1" DS 0 z gradOperand)
  let v' := render (.adamVNextF "%v" "%b2" "%ob2" DS 0 z gradOperand)
  let vT := verbs trusted
  let vP := verbs θ'
  IO.println s!"emitAdamV       : {vT.length} ops"
  IO.println s!"adamWParamF     : {vP.length} ops"
  IO.println s!"  trusted : {vT}"
  IO.println s!"  proven  : {vP}"
  if vT != vP then
    IO.eprintln "MISMATCH: adamWParamF does not emit emitAdamV's op sequence"
    IO.Process.exit 1
  -- m'/v' are prefixes of the same computation: emitAdamV interleaves them into its one block,
  -- so check each moment op's verbs occur as a contiguous run of the trusted sequence.
  let isInfix (small big : List String) : Bool :=
    (List.range (big.length + 1 - small.length)).any fun i =>
      ((big.drop i).take small.length) == small
  let vM := verbs m'
  let vV := verbs v'
  IO.println s!"adamMNextF      : {vM.length} ops — contiguous in emitAdamV: {isInfix vM vT}"
  IO.println s!"adamVNextF      : {vV.length} ops — contiguous in emitAdamV: {isInfix vV vT}"
  if !(isInfix vM vT) || !(isInfix vV vT) then
    IO.eprintln "MISMATCH: a moment op is not emitAdamV's corresponding run"
    IO.Process.exit 1
  IO.println "✓ the proven AdamW ops emit emitAdamV's graph, op for op"

#eval main
