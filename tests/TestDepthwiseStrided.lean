import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! Standalone render + `iree-compile` validation for the Chapter-7 C3 (D2)
    STRIDED depthwise conv SHlo op pair (`depthwiseStridedF` / `depthwiseStridedBack`).
    The stride-2 depthwise MobileNetV2 downsamples with. Forward halves spatial via
    `window_strides=[2,2]` (feature_group_count=c, [c,1,3,3] kernel); backward
    zero-upsamples the cotangent (`stablehlo.pad` interior=1) then runs the
    reversed-kernel stride-1 depthwise. den via the proven `depthwiseStride2Flat`
    / `depthwiseStride2Flat_has_vjp` (= decimate ∘ depthwise). Values are placeholders.

    Run (rocm):
      export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
      lake env lean tests/TestDepthwiseStrided.lean
-/

open Proofs Proofs.StableHLO

private def C  : Nat := 4
private def Ho : Nat := 8       -- output spatial (input = 16×16)
private def Wo : Nat := 8
private def KH : Nat := 3
private def KW : Nat := 3
private def BS : Nat := 2

-- render-irrelevant placeholder runtime values
private def wv : DepthwiseKernel C KH KW := fun _ _ _ => 0
private def bv : Vec C := fun _ => 0
private def xv : Vec (C*(2*Ho)*(2*Wo)) := fun _ => 0
private def dyv : Vec (C*Ho*Wo) := fun _ => 0

/-- `@dwstrided_fwd`: one `depthwiseStridedF` over `%x` (16×16 → 8×8). -/
private def fwdModule : String :=
  renderModule "dwstrided_fwd"
    s!"%x: {ty [BS, C*(2*Ho)*(2*Wo)]}, %W: {ty [C,1,KH,KW]}, %b: {ty [C]}"
    BS (C*Ho*Wo)
    (.depthwiseStridedF (c := C) (h := Ho) (w := Wo) (kH := KH) (kW := KW)
      "%W" "%b" wv bv (.operand "%x" xv))

/-- `@dwstrided_back`: one `depthwiseStridedBack` over `%dy` (8×8 → 16×16). -/
private def backModule : String :=
  renderModule "dwstrided_back"
    s!"%dy: {ty [BS, C*Ho*Wo]}, %W: {ty [C,1,KH,KW]}"
    BS (C*(2*Ho)*(2*Wo))
    (.depthwiseStridedBack (c := C) (h := Ho) (w := Wo) (kH := KH) (kW := KW)
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
  IO.println "── @dwstrided_fwd ──"
  IO.println fwdModule
  IO.println "── @dwstrided_back ──"
  IO.println backModule
  compileCheck "dwstrided_fwd" fwdModule
  compileCheck "dwstrided_back" backModule

#eval main
