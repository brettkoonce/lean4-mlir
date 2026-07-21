import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.ViTRender
import LeanMlir.GradcheckHelpers
import LeanMlir.Types

/-! # ch10 V4 — transformer block renderer (pre-norm) + Lean gradcheck

One ViT encoder block: `x → LN1 → MHSA → +x → LN2 → MLP(fc1→GELU→fc2) → +`. Uses
the shared `ViTRender` fragments (validated MHSA from TestMHSA + per-channel `[D]`
LayerNorm + GELU MLP). LayerNorm γ/β are per-channel `[D]` (the user's choice; the
proof witness uses scalar — this goes beyond it, faithful per-op: normalize =
`layerNormForward` γ=1/β=0, affine = `layerScale` + `[D]` bias). De-risked here on
a tiny config (B=2,N=3,D=4,heads=2,mlp=8), `iree-compile`d, and gradchecked over
EVERY input (x + all 16 block params) — this validates the new per-channel LN
backward AND the residual fan-in wiring in addition to the already-validated MHSA.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"
  export LD_LIBRARY_PATH="$PWD/ffi:/opt/rocm/lib:$LD_LIBRARY_PATH"
  export IREE_BACKEND=rocm
  lake env lean tests/TestViTBlock.lean
-/

open Proofs Proofs.StableHLO
open ViTRender ViTGradcheck

private def Bb : Nat := 2
private def Nn : Nat := 3
private def Dd : Nat := 4
private def Hh : Nat := 2
private def Dh : Nat := 2
private def Mm : Nat := 8
private def epsStr : String := "1.0e-5"
private def scaleStr : String := "0.7071067811865476"   -- 1/√2

private def bp : BlockParams :=
  { g1 := "%g1", b1 := "%b1",
    Wq := "%Wq", bq := "%bq", Wk := "%Wk", bk := "%bk",
    Wv := "%Wv", bv := "%bv", Wo := "%Wo", bo := "%bo",
    g2 := "%g2", b2 := "%b2",
    Wfc1 := "%Wfc1", bfc1 := "%bfc1", Wfc2 := "%Wfc2", bfc2 := "%bfc2" }

/-- Bare shape string for an `iree-run-module --input` (no `tensor<…>` wrapper). -/
private def shp (dims : List Nat) : String := String.intercalate "x" (dims.map toString ++ ["f32"])

/-- The 17 forward inputs (x + 16 params) in func-arg order: name, dims, flatLen. -/
private def inputs : List (String × List Nat × Nat) :=
  [("%x", [Bb,Nn,Dd], Bb*Nn*Dd),
   ("%g1", [Dd], Dd), ("%b1", [Dd], Dd),
   ("%Wq", [Dd,Dd], Dd*Dd), ("%bq", [Dd], Dd),
   ("%Wk", [Dd,Dd], Dd*Dd), ("%bk", [Dd], Dd),
   ("%Wv", [Dd,Dd], Dd*Dd), ("%bv", [Dd], Dd),
   ("%Wo", [Dd,Dd], Dd*Dd), ("%bo", [Dd], Dd),
   ("%g2", [Dd], Dd), ("%b2", [Dd], Dd),
   ("%Wfc1", [Dd,Mm], Dd*Mm), ("%bfc1", [Mm], Mm),
   ("%Wfc2", [Mm,Dd], Mm*Dd), ("%bfc2", [Dd], Dd)]

private def argSig : String :=
  String.intercalate ", " (inputs.map (fun (nm, dims, _) => s!"{nm}: {ty dims}"))

private def fwdModule : String :=
  "module @m {\n" ++
  s!"  func.func @block_fwd({argSig}) -> {ty [Bb,Nn,Dd]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  blockFwd "blk" "%x" bp Bb Nn Dd Mm Hh Dh epsStr scaleStr ++
  s!"    return %blkout : {ty [Bb,Nn,Dd]}\n" ++ "  }\n}\n"

/-- Backward returns dx + the 16 param grads (BlockParams order). -/
private def backModule : String :=
  let gradNames := s!"%blkdx" :: blockGradNames "blk"
  let retShapes := ty [Bb,Nn,Dd] :: (inputs.drop 1).map (fun (_, dims, _) => ty dims)
  let retTy := String.intercalate ", " retShapes
  "module @m {\n" ++
  s!"  func.func @block_back({argSig}, %dOut: {ty [Bb,Nn,Dd]}) -> ({retTy}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  blockFwd "blk" "%x" bp Bb Nn Dd Mm Hh Dh epsStr scaleStr ++
  blockBack "blk" "%dOut" bp Bb Nn Dd Mm Hh Dh scaleStr ++
  s!"    return {String.intercalate ", " gradNames} : {retTy}\n" ++ "  }\n}\n"

private def compileCheck (name body : String) : IO Bool := do
  IO.FS.createDirAll ".lake/build"
  let path := s!".lake/build/{name}.mlir"
  IO.FS.writeFile path body
  let cargs ← ireeCompileArgs path s!".lake/build/{name}.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"[{name}] iree-compile FAILED:\n{r.stderr.take 3000}"; return false
  else
    IO.println s!"[{name}] iree-compile OK → .lake/build/{name}.vmfb"; return true

def main : IO Unit := do
  IO.FS.writeFile ".lake/build/block_fwd_dump.mlir" fwdModule
  let okF ← compileCheck "block_fwd" fwdModule
  let okB ← compileCheck "block_back" backModule
  if okF && okB then
    let _ ← adjointGradcheck "vit-block" ".lake/build/block_fwd.vmfb" "block_fwd"
      ".lake/build/block_back.vmfb" "block_back"
      (inputs.map (fun (_, dims, _) => shp dims)) (inputs.map (fun (_, _, l) => l))
      (shp [Bb,Nn,Dd]) (Bb*Nn*Dd)
    pure ()

#eval main
