import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Types

/-! # ch9 N3 — patchify (4×4/s4) + downsample (2×2/s2) strided convs, render + iree

The two EVEN-kernel strided convs ConvNeXt needs, which the proven odd-kernel
`flatConvStride2` (= decimate∘SAME, pad (k-1)/2) does NOT cover (kernel 2 ⇒ pad 0,
shape mismatch). Hand-derived even-kernel (k = stride, pad 0, non-overlapping) backward,
verified on concrete tiny examples:

  forward     : conv stride s, pad 0, kernel s  ([B,ic,sH,sW] → [B,oc,H,W])
  input-grad  : dilate dy by s (interior s-1, high s-1 → sH), reverse(Wᵀ),
                stride-1 conv pad [[s-1,0],[s-1,0]]   →  [B,ic,sH,sW]
  weight-grad : dilate dy by s (interior s-1, NO high → sH-(s-1)), valid stride-1
                conv (x as lhs, dilated dy as rhs)    →  [oc,ic,s,s]
  bias-grad   : reduce dy over [0,2,3] → [oc]

Patchify (stem, first layer) needs only weight+bias grad. Compile-only shape check.

Run (rocm): export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestConvNeXtStrided.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 2

-- ════════════ patchify: k×k stride-k pad-0 conv (k = stride, non-overlapping) ════════════

/-- Patchify forward: s×s stride-s pad-0 conv, `[B,ic,s·Ho,s·Wo] → [B,oc,Ho,Wo]`. -/
private def patchify (o x w bnm : String) (oc ic Ho Wo s : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [{s}, {s}], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,s*Ho,s*Wo]}, {ty [oc,ic,s,s]}) -> {ty [BS,oc,Ho,Wo]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Ho,Wo]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Ho,Wo]}\n"

/-- Patchify weight-grad: dilate dy by s (interior s-1, no high → s·Ho-(s-1)), valid
    stride-1 conv (x lhs, dilated dy rhs). Result `%{o}` `[oc,ic,s,s]`. -/
private def patchifyWGrad (o inp dy : String) (ic oc Ho Wo s : Nat) : String :=
  let dilH := s*Ho - (s-1)
  let dilW := s*Wo - (s-1)
  s!"    %{o}u = stablehlo.pad {dy}, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, {s-1}, {s-1}] : ({ty [BS,oc,Ho,Wo]}, tensor<f32>) -> {ty [BS,oc,dilH,dilW]}\n" ++
  s!"    %{o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [BS,ic,s*Ho,s*Wo]}) -> {ty [ic,BS,s*Ho,s*Wo]}\n" ++
  s!"    %{o}dt = stablehlo.transpose %{o}u, dims = [1, 0, 2, 3] : ({ty [BS,oc,dilH,dilW]}) -> {ty [oc,BS,dilH,dilW]}\n" ++
  s!"    %{o}raw = stablehlo.convolution(%{o}xt, %{o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ic,BS,s*Ho,s*Wo]}, {ty [oc,BS,dilH,dilW]}) -> {ty [ic,oc,s,s]}\n" ++
  s!"    %{o} = stablehlo.transpose %{o}raw, dims = [1, 0, 2, 3] : ({ty [ic,oc,s,s]}) -> {ty [oc,ic,s,s]}\n"

private def convBiasGrad (o dy : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.reduce({dy} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"

-- ════════════ downsample: 2×2 stride-2 pad-0 conv ════════════

/-- Downsample forward: 2×2 stride-2 pad-0 conv, `[B,ic,2H,2W] → [B,oc,H,W]`. -/
private def convDown (o x w bnm : String) (oc ic H W : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,2*H,2*W]}, {ty [oc,ic,2,2]}) -> {ty [BS,oc,H,W]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,H,W]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,H,W]}\n"

/-- Downsample input-grad: dilate dy (interior 1, high 1 → 2H), reverse(Wᵀ), stride-1
    conv pad [[1,0],[1,0]]. Result `%{o}` `[B,ic,2H,2W]`. -/
private def convDownBack (o dy w : String) (ic oc H W : Nat) : String :=
  s!"    %{o}u = stablehlo.pad {dy}, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [BS,oc,H,W]}, tensor<f32>) -> {ty [BS,oc,2*H,2*W]}\n" ++
  s!"    %{o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,ic,2,2]}) -> {ty [ic,oc,2,2]}\n" ++
  s!"    %{o}r = stablehlo.reverse %{o}t, dims = [2, 3] : {ty [ic,oc,2,2]}\n" ++
  s!"    %{o} = stablehlo.convolution(%{o}u, %{o}r)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,oc,2*H,2*W]}, {ty [ic,oc,2,2]}) -> {ty [BS,ic,2*H,2*W]}\n"

/-- Downsample weight-grad: dilate dy (interior 1, NO high → 2H-1), valid stride-1
    conv (x lhs, dilated dy rhs). Result `%{o}` `[oc,ic,2,2]`. -/
private def convDownWGrad (o inp dy : String) (ic oc H W : Nat) : String :=
  s!"    %{o}u = stablehlo.pad {dy}, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : ({ty [BS,oc,H,W]}, tensor<f32>) -> {ty [BS,oc,2*H-1,2*W-1]}\n" ++
  s!"    %{o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [BS,ic,2*H,2*W]}) -> {ty [ic,BS,2*H,2*W]}\n" ++
  s!"    %{o}dt = stablehlo.transpose %{o}u, dims = [1, 0, 2, 3] : ({ty [BS,oc,2*H-1,2*W-1]}) -> {ty [oc,BS,2*H-1,2*W-1]}\n" ++
  s!"    %{o}raw = stablehlo.convolution(%{o}xt, %{o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ic,BS,2*H,2*W]}, {ty [oc,BS,2*H-1,2*W-1]}) -> {ty [ic,oc,2,2]}\n" ++
  s!"    %{o} = stablehlo.transpose %{o}raw, dims = [1, 0, 2, 3] : ({ty [ic,oc,2,2]}) -> {ty [oc,ic,2,2]}\n"

-- ════════════ standalone shape-check module ════════════

private def stridedModule : String := Id.run do
  -- patchify: 3→16, 32→8 (stride 4); downsample: 8→16, 16→8 (stride 2)
  let body :=
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %xr = stablehlo.reshape %x : (" ++ ty [BS,3*32*32] ++ ") -> " ++ ty [BS,3,32,32] ++ "\n" ++
    -- patchify forward + grads
    patchify "ps" "%xr" "%psW" "%psb" 16 3 8 8 4 ++
    patchifyWGrad "psdW" "%xr" "%psdy" 3 16 8 8 4 ++
    convBiasGrad "psdb" "%psdy" 16 8 8 ++
    -- downsample forward + grads (input %dsx : [B,8,16,16])
    convDown "ds" "%dsx" "%dsW" "%dsb" 16 8 8 8 ++
    convDownBack "dsdx" "%dsdy" "%dsW" 8 16 8 8 ++
    convDownWGrad "dsdW" "%dsx" "%dsdy" 8 16 8 8 ++
    convBiasGrad "dsdb" "%dsdy" 16 8 8
  let sig := String.intercalate ", "
    [s!"%x: {ty [BS,3*32*32]}", s!"%psW: {ty [16,3,4,4]}", s!"%psb: {ty [16]}",
     s!"%psdy: {ty [BS,16,8,8]}",
     s!"%dsx: {ty [BS,8,16,16]}", s!"%dsW: {ty [16,8,2,2]}", s!"%dsb: {ty [16]}",
     s!"%dsdy: {ty [BS,16,8,8]}"]
  let retTy := String.intercalate ", "
    [ty [BS,16,8,8], ty [16,3,4,4], ty [16], ty [BS,16,8,8], ty [BS,8,16,16], ty [16,8,2,2], ty [16]]
  let retV := "%ps, %psdW, %psdb, %ds, %dsdx, %dsdW, %dsdb"
  return "module @m {\n" ++ s!"  func.func @convnext_strided({sig}) -> ({retTy}) " ++ "{\n" ++
    body ++ s!"    return {retV} : {retTy}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := stridedModule
  IO.println s!"rendered ConvNeXt strided (patchify 4×4/s4 + downsample 2×2/s2): {mlir.length} chars"
  IO.FS.createDirAll ".lake/build"
  let path := ".lake/build/convnext_strided.mlir"
  IO.FS.writeFile path mlir
  let cargs ← ireeCompileArgs path ".lake/build/convnext_strided.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println "convnext_strided iree-compile OK → .lake/build/convnext_strided.vmfb"

#eval main
