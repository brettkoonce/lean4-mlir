import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Types

/-! Standalone render + `iree-compile` validation for the Chapter-9 N1 GELU SHlo
    op pair (`geluF` / `geluBack`). Renders a tiny forward (tanh approximation,
    `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, via `stablehlo.tanh`) and backward
    (`dy ⊙ gelu'(x)`, the closed-form tanh-approx derivative, recomputed from the
    saved pre-activation `%xs`) `func.func` from the VERIFIED `Proofs.StableHLO`
    emitter and compiles each to a ROCm `.vmfb` — the thin lexical boundary the
    proofs leave to `iree-compile`. GELU is smooth everywhere (no kink, no select
    mask), like swish/sigmoid. The Lean operand/x values are render-irrelevant
    placeholders.

    Run (rocm):
      export PATH="$PWD/.venv/bin:$PATH"
      export IREE_BACKEND=rocm
      lake env lean tests/TestGelu.lean
-/

open Proofs Proofs.StableHLO

private def N  : Nat := 64
private def BS : Nat := 2

-- render-irrelevant placeholder runtime values
private def xv : Vec N := fun _ => 0
private def dyv : Vec N := fun _ => 0

/-- `@gelu_fwd` from the verified AST: one `geluF` over `%x`. -/
private def fwdModule : String :=
  renderModule "gelu_fwd" s!"%x: {ty [BS, N]}" BS N (.geluF (.operand "%x" xv))

/-- `@gelu_back` from the verified AST: one `geluBack` over `%dy`, using the saved
    pre-activation `%xs` (closed-form `dy ⊙ gelu'(xs)`). -/
private def backModule : String :=
  renderModule "gelu_back" s!"%dy: {ty [BS, N]}, %xs: {ty [BS, N]}"
    BS N (.geluBack "%xs" xv (.operand "%dy" dyv))

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
  IO.println "── @gelu_fwd ──"
  IO.println fwdModule
  IO.println "── @gelu_back ──"
  IO.println backModule
  compileCheck "gelu_fwd" fwdModule
  compileCheck "gelu_back" backModule

#eval main
