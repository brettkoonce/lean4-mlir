import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Types

/-! Standalone render + `iree-compile` validation for the Chapter-8 E2 sigmoid SHlo
    op pair (`sigmoidF` / `sigmoidBack`) — the squeeze-excite gate's output
    nonlinearity. Renders a tiny forward (`σ(x) = stablehlo.logistic`) and backward
    (`dy ⊙ σ(x)·(1−σ(x))`, recomputed from the saved pre-activation `%xs`)
    `func.func` from the VERIFIED `Proofs.StableHLO` emitter and compiles each to a
    ROCm `.vmfb` — the thin lexical boundary the proofs leave to `iree-compile`.
    Sigmoid is smooth everywhere (no kink, no mask), like swish but with a forward
    of just one `logistic`. The Lean operand/x values are render-irrelevant placeholders.

    Run (rocm):
      export PATH="$PWD/.venv/bin:$PATH"
      export IREE_BACKEND=rocm
      lake env lean tests/TestSigmoid.lean
-/

open Proofs Proofs.StableHLO

private def N  : Nat := 64
private def BS : Nat := 2

-- render-irrelevant placeholder runtime values
private def xv : Vec N := fun _ => 0
private def dyv : Vec N := fun _ => 0

/-- `@sigmoid_fwd` from the verified AST: one `sigmoidF` over `%x`. -/
private def fwdModule : String :=
  renderModule "sigmoid_fwd" s!"%x: {ty [BS, N]}" BS N (.sigmoidF (.operand "%x" xv))

/-- `@sigmoid_back` from the verified AST: one `sigmoidBack` over `%dy`, using the saved
    pre-activation `%xs` (closed-form `dy ⊙ σ(xs)·(1−σ(xs))`). -/
private def backModule : String :=
  renderModule "sigmoid_back" s!"%dy: {ty [BS, N]}, %xs: {ty [BS, N]}"
    BS N (.sigmoidBack "%xs" xv (.operand "%dy" dyv))

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
  IO.println "── @sigmoid_fwd ──"
  IO.println fwdModule
  IO.println "── @sigmoid_back ──"
  IO.println backModule
  compileCheck "sigmoid_fwd" fwdModule
  compileCheck "sigmoid_back" backModule

#eval main
