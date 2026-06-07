import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! Standalone render + `iree-compile` validation for the Chapter-7 C1 depthwise
    conv SHlo op pair (`depthwiseF` / `depthwiseBack`). Renders a tiny forward and
    backward `func.func` from the VERIFIED `Proofs.StableHLO` emitter and compiles
    each to a ROCm `.vmfb` — the thin lexical boundary the proofs leave to
    `iree-compile`. The depthwise delta from a normal conv: `feature_group_count = c`
    plus a `[c,1,kH,kW]` kernel (one filter per channel). The Lean operand/W/b values
    are render-irrelevant placeholders (only SSA names + shapes reach the text).

    Run (rocm):
      export PATH="$PWD/.venv/bin:$PATH"
      export IREE_BACKEND=rocm
      lake env lean tests/TestDepthwise.lean
-/

open Proofs Proofs.StableHLO

private def C  : Nat := 4
private def Hh : Nat := 8
private def Ww : Nat := 8
private def KH : Nat := 3
private def KW : Nat := 3
private def BS : Nat := 2

-- render-irrelevant placeholder runtime values
private def wv : DepthwiseKernel C KH KW := fun _ _ _ => 0
private def bv : Vec C := fun _ => 0
private def xv : Vec (C*Hh*Ww) := fun _ => 0
private def dyv : Vec (C*Hh*Ww) := fun _ => 0

/-- `@depthwise_fwd` from the verified AST: one `depthwiseF` over `%x`. -/
private def fwdModule : String :=
  renderModule "depthwise_fwd"
    s!"%x: {ty [BS, C*Hh*Ww]}, %W: {ty [C,1,KH,KW]}, %b: {ty [C]}"
    BS (C*Hh*Ww)
    (.depthwiseF (c := C) (h := Hh) (w := Ww) (kH := KH) (kW := KW)
      "%W" "%b" wv bv (.operand "%x" xv))

/-- `@depthwise_back` from the verified AST: one `depthwiseBack` over `%dy`
    (reversed-kernel depthwise conv). -/
private def backModule : String :=
  renderModule "depthwise_back"
    s!"%dy: {ty [BS, C*Hh*Ww]}, %W: {ty [C,1,KH,KW]}"
    BS (C*Hh*Ww)
    (.depthwiseBack (c := C) (h := Hh) (w := Ww) (kH := KH) (kW := KW)
      "%W" wv bv xv (.operand "%dy" dyv))

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
  IO.println "── @depthwise_fwd ──"
  IO.println fwdModule
  IO.println "── @depthwise_back ──"
  IO.println backModule
  compileCheck "depthwise_fwd" fwdModule
  compileCheck "depthwise_back" backModule

#eval main
