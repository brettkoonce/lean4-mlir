import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! Standalone render + `iree-compile` validation for the Chapter-6 B8b per-channel
    BatchNorm SHlo op pair (`bnPerChannelF` / `bnPerChannelBack`). Renders a tiny
    forward and backward `func.func` from the VERIFIED `Proofs.StableHLO` emitter and
    compiles each to a ROCm `.vmfb` — the thin lexical boundary the proofs leave to
    `iree-compile`. The Lean operand/γ/β/x values are render-irrelevant placeholders
    (only SSA names + the ε literal reach the text).

    Run (rocm):
      export PATH="$PWD/.venv/bin:$PATH"
      export IREE_BACKEND=rocm
      lake env lean tests/TestPerChannelBn.lean
-/

open Proofs Proofs.StableHLO

private def OC  : Nat := 4
private def Hh  : Nat := 8
private def Ww  : Nat := 8
private def BS  : Nat := 2
private def EPS : String := "1.0e-5"

-- render-irrelevant placeholder runtime values
private def gv : Vec OC := fun _ => 0
private def xv : Vec (OC*Hh*Ww) := fun _ => 0
private def dyv : Vec (OC*Hh*Ww) := fun _ => 0

/-- `@perchannel_bn_fwd` from the verified AST: one `bnPerChannelF` over `%x`. -/
private def fwdModule : String :=
  renderModule "perchannel_bn_fwd"
    s!"%x: {ty [BS, OC*Hh*Ww]}, %g: {ty [OC]}, %b: {ty [OC]}"
    BS (OC*Hh*Ww)
    (.bnPerChannelF (oc := OC) (h := Hh) (w := Ww) "%g" "%b" EPS 0 gv gv (.operand "%x" xv))

/-- `@perchannel_bn_back` from the verified AST: one `bnPerChannelBack` over `%dy`,
    recomputing μ/var from the saved input `%xs`. -/
private def backModule : String :=
  renderModule "perchannel_bn_back"
    s!"%dy: {ty [BS, OC*Hh*Ww]}, %g: {ty [OC]}, %xs: {ty [BS, OC*Hh*Ww]}"
    BS (OC*Hh*Ww)
    (.bnPerChannelBack (oc := OC) (h := Hh) (w := Ww) "%g" "%xs" EPS 0 gv xv (.operand "%dy" dyv))

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
  IO.println "── @perchannel_bn_fwd ──"
  IO.println fwdModule
  IO.println "── @perchannel_bn_back ──"
  IO.println backModule
  compileCheck "perchannel_bn_fwd" fwdModule
  compileCheck "perchannel_bn_back" backModule

#eval main
