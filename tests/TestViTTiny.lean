import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.ViTRender
import LeanMlir.GradcheckHelpers
import LeanMlir.Types

/-! # ch10 V6a — WHOLE-ViT tiny-config gradcheck (image → logits → all param grads)

Validates the ENTIRE ViT codegen assembly end-to-end at a tiny config (so the
finite-difference adjoint test is accurate and cheap): patch-embed conv (k=s=4) +
weight-grad, CLS token, positional embed, 2 transformer blocks, final per-channel
LayerNorm, CLS-slice classifier head. The image `x` is a FIXED input (no image grad
— patch embed is the first layer); the gradcheck perturbs + checks EVERY learnable
param (patch W/b, CLS, pos, both blocks' 16 each, final-LN γ/β, head W/b).

A PASS here means the full `vitFwd`/`vitBack` (shared `ViTRender`) is correct; the
production 224²/depth-12 render (TestViTTrain) is the SAME fragments at scale.

Config: b=2, ic=3, image 8×8 (s=4 ⇒ 2×2=4 patches, +CLS = 5 tokens), d=8, heads=2,
mlp=16, classes=10, depth=2.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"
  export LD_LIBRARY_PATH="$PWD/ffi:/opt/rocm/lib:$LD_LIBRARY_PATH"
  export IREE_BACKEND=rocm
  lake env lean tests/TestViTTiny.lean
-/

open Proofs Proofs.StableHLO
open ViTRender ViTGradcheck

private def cfg : ViTConfig :=
  { b := 2, ic := 3, d := 8, ph := 2, pw := 2, s := 4, m := 16, h := 2, dh := 4,
    nc := 10, eps := "1.0e-5", scale := "0.5" }   -- 1/√dh = 1/√4 = 0.5

private def mkBP (i : Nat) : BlockParams :=
  { g1 := s!"%g1_{i}", b1 := s!"%b1_{i}",
    Wq := s!"%Wq_{i}", bq := s!"%bq_{i}", Wk := s!"%Wk_{i}", bk := s!"%bk_{i}",
    Wv := s!"%Wv_{i}", bv := s!"%bv_{i}", Wo := s!"%Wo_{i}", bo := s!"%bo_{i}",
    g2 := s!"%g2_{i}", b2 := s!"%b2_{i}",
    Wfc1 := s!"%Wfc1_{i}", bfc1 := s!"%bfc1_{i}", Wfc2 := s!"%Wfc2_{i}", bfc2 := s!"%bfc2_{i}" }

private def blocks : List BlockParams := [mkBP 0, mkBP 1]

private def shp (dims : List Nat) : String := String.intercalate "x" (dims.map toString ++ ["f32"])
private def prod (dims : List Nat) : Nat := dims.foldl (· * ·) 1

private def pnames : List String := vitParamNames blocks
private def pdims  : List (List Nat) := vitParamDims blocks cfg
private def imgDims : List Nat := [cfg.b, cfg.ic, cfg.s * cfg.ph, cfg.s * cfg.pw]

private def argSig : String :=
  s!"%x: {ty imgDims}, " ++
  String.intercalate ", " ((pnames.zip pdims).map (fun (nm, dims) => s!"{nm}: {ty dims}"))

private def fwdModule : String :=
  "module @m {\n" ++
  s!"  func.func @vit_fwd({argSig}) -> {ty [cfg.b, cfg.nc]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  vitFwd "vit" "%x" "%wConv" "%bConv" "%cls" "%pos" "%gF" "%bF" "%Wc" "%bc" blocks cfg ++
  s!"    return %vithdlogits : {ty [cfg.b, cfg.nc]}\n" ++ "  }\n}\n"

private def backModule : String :=
  let retTy := String.intercalate ", " (pdims.map (fun dims => ty dims))
  "module @m {\n" ++
  s!"  func.func @vit_back({argSig}, %dlog: {ty [cfg.b, cfg.nc]}) -> ({retTy}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  vitFwd "vit" "%x" "%wConv" "%bConv" "%cls" "%pos" "%gF" "%bF" "%Wc" "%bc" blocks cfg ++
  vitBack "vit" "%dlog" "%x" "%wConv" "%Wc" "%gF" blocks cfg ++
  s!"    return {String.intercalate ", " (vitGradNames "vit" blocks)} : {retTy}\n" ++ "  }\n}\n"

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
  let okF ← compileCheck "vit_fwd" fwdModule
  let okB ← compileCheck "vit_back" backModule
  if okF && okB then
    let xVals := randVec 999 (prod imgDims)
    let _ ← adjointGradcheckFixed "vit-tiny" ".lake/build/vit_fwd.vmfb" "vit_fwd"
      ".lake/build/vit_back.vmfb" "vit_back"
      [(shp imgDims, xVals)]
      (pdims.map shp) (pdims.map prod)
      (shp [cfg.b, cfg.nc]) (cfg.b * cfg.nc)
    pure ()

#eval main
