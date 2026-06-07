import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! Standalone render + `iree-compile` validation for the Chapter-7 C2 ReLU6 SHlo
    op pair (`relu6F` / `selectMid`). Renders a tiny forward (`clamp(·,0,6) =
    min(max(·,0),6)`) and backward (`select(0<x<6,·,0)`, the two-sided-kink mask)
    `func.func` from the VERIFIED `Proofs.StableHLO` emitter and compiles each to a
    ROCm `.vmfb` — the thin lexical boundary the proofs leave to `iree-compile`.
    The Lean operand/x values are render-irrelevant placeholders.

    Run (rocm):
      export PATH="$PWD/.venv/bin:$PATH"
      export IREE_BACKEND=rocm
      lake env lean tests/TestRelu6.lean
-/

open Proofs Proofs.StableHLO

private def N  : Nat := 64
private def BS : Nat := 2

-- render-irrelevant placeholder runtime values
private def xv : Vec N := fun _ => 0
private def dyv : Vec N := fun _ => 0

/-- `@relu6_fwd` from the verified AST: one `relu6F` over `%x`. -/
private def fwdModule : String :=
  renderModule "relu6_fwd" s!"%x: {ty [BS, N]}" BS N (.relu6F (.operand "%x" xv))

/-- `@relu6_back` from the verified AST: one `selectMid` over `%dy`, masking on the
    saved pre-activation `%xs` (route dy where `0 < xs < 6`). -/
private def backModule : String :=
  renderModule "relu6_back" s!"%dy: {ty [BS, N]}, %xs: {ty [BS, N]}"
    BS N (.selectMid "%xs" xv (.operand "%dy" dyv))

private def compileCheck (name body : String) : IO Unit := do
  IO.FS.createDirAll ".lake/build"
  let path := s!".lake/build/{name}.mlir"
  IO.FS.writeFile path body
  let cargs ← ireeCompileArgs path s!".lake/build/{name}.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"[{name}] iree-compile FAILED:\n{r.stderr.take 2000}"
  else
    IO.println s!"[{name}] iree-compile OK → .lake/build/{name}.vmfb"

def main : IO Unit := do
  IO.println "── @relu6_fwd ──"
  IO.println fwdModule
  IO.println "── @relu6_back ──"
  IO.println backModule
  compileCheck "relu6_fwd" fwdModule
  compileCheck "relu6_back" backModule

#eval main
