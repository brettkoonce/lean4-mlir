import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Types

/-! # ch9 N5 — ConvNeXt-T forward renderer (faithful [3,3,9,3] config) + iree

Programmatic StableHLO for **ConvNeXt-T** (Liu et al. 2022) on **Imagenette 224², 10
classes** — the paper-native resolution:

  stem    : 4×4 conv stride 4 (3→96)  "patchify"             224→56
  stage 1 : 3× ConvNeXt block @ 96                           @56
  downsmpl: LN + 2×2 conv stride 2 (96→192)                  56→28
  stage 2 : 3× ConvNeXt block @ 192                          @28
  downsmpl: LN + 2×2 conv stride 2 (192→384)                 28→14
  stage 3 : 9× ConvNeXt block @ 384                          @14
  downsmpl: LN + 2×2 conv stride 2 (384→768)                 14→7
  stage 4 : 3× ConvNeXt block @ 768                          @7
  head    : globalAvgPool → LN(768) → dense 768→10

ConvNeXt block (dim c, expand 4c): depthwise 7×7 → LN → 1×1 expand c→4c → GELU
→ 1×1 project 4c→c → layerScale (per-channel γ) → + x (identity skip; stride-1 only).
LN = per-example GLOBAL scalar-γ/β batch-norm (`= layerNormForward`); GELU = `geluF`
(tanh approx). Every fragment is the StableHLO a VERIFIED per-op emitter produces.

Run (rocm): export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestConvNeXtFwd.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 32
private def IMG : Nat := 224
private def EPS : String := "1.0e-6"

-- ════════════ fragments (4-D [B,C,H,W]) ════════════

private def conv1 (o x w bnm : String) (oc ic Hh Ww : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,Hh,Ww]}, {ty [oc,ic,1,1]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Hh,Ww]}\n"

private def dwconv (o x w bnm : String) (c Hh Ww k : Nat) : String :=
  let p := (k-1)/2
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,Hh,Ww]}, {ty [c,1,k,k]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,c,Hh,Ww]}\n"

private def addOp (o a b : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.add {a}, {b} : {ty [BS,oc,Hh,Ww]}\n"

private def geluAct (o x : String) (c Hh Ww : Nat) : String :=
  let t := ty [BS,c,Hh,Ww]
  s!"    %{o}x2 = stablehlo.multiply {x}, {x} : {t}\n" ++
  s!"    %{o}x3 = stablehlo.multiply %{o}x2, {x} : {t}\n" ++
  s!"    %{o}ck = stablehlo.constant dense<0.044715> : {t}\n" ++
  s!"    %{o}kx3 = stablehlo.multiply %{o}ck, %{o}x3 : {t}\n" ++
  s!"    %{o}inn = stablehlo.add {x}, %{o}kx3 : {t}\n" ++
  s!"    %{o}cs = stablehlo.constant dense<0.7978845608028654> : {t}\n" ++
  s!"    %{o}u = stablehlo.multiply %{o}cs, %{o}inn : {t}\n" ++
  s!"    %{o}t = stablehlo.tanh %{o}u : {t}\n" ++
  s!"    %{o}one = stablehlo.constant dense<1.0> : {t}\n" ++
  s!"    %{o}opt = stablehlo.add %{o}one, %{o}t : {t}\n" ++
  s!"    %{o}half = stablehlo.constant dense<0.5> : {t}\n" ++
  s!"    %{o}hx = stablehlo.multiply %{o}half, {x} : {t}\n" ++
  s!"    %{o} = stablehlo.multiply %{o}hx, %{o}opt : {t}\n"

/-- LayerNorm = per-example GLOBAL scalar-γ/β batch-norm over flat c·h·w (`= bnF`). -/
private def lnFwd (o x g bt : String) (c Hh Ww : Nat) : String :=
  let n := c*Hh*Ww
  let tn := ty [BS,n]
  s!"    %{o}ri = stablehlo.reshape {x} : ({ty [BS,c,Hh,Ww]}) -> {tn}\n" ++
  s!"    %{o}nf = stablehlo.constant dense<{n}.0> : {tn}\n" ++
  s!"    %{o}ep = stablehlo.constant dense<{EPS}> : {tn}\n" ++
  s!"    %{o}smr = stablehlo.reduce(%{o}ri init: %sc) applies stablehlo.add across dimensions = [1] : ({tn}, tensor<f32>) -> {ty [BS]}\n" ++
  s!"    %{o}sm = stablehlo.broadcast_in_dim %{o}smr, dims = [0] : ({ty [BS]}) -> {tn}\n" ++
  s!"    %{o}mu = stablehlo.divide %{o}sm, %{o}nf : {tn}\n" ++
  s!"    %{o}xc = stablehlo.subtract %{o}ri, %{o}mu : {tn}\n" ++
  s!"    %{o}sq = stablehlo.multiply %{o}xc, %{o}xc : {tn}\n" ++
  s!"    %{o}vsr = stablehlo.reduce(%{o}sq init: %sc) applies stablehlo.add across dimensions = [1] : ({tn}, tensor<f32>) -> {ty [BS]}\n" ++
  s!"    %{o}vs = stablehlo.broadcast_in_dim %{o}vsr, dims = [0] : ({ty [BS]}) -> {tn}\n" ++
  s!"    %{o}vr = stablehlo.divide %{o}vs, %{o}nf : {tn}\n" ++
  s!"    %{o}ve = stablehlo.add %{o}vr, %{o}ep : {tn}\n" ++
  s!"    %{o}istd = stablehlo.rsqrt %{o}ve : {tn}\n" ++
  s!"    %{o}xh = stablehlo.multiply %{o}xc, %{o}istd : {tn}\n" ++
  s!"    %{o}gb = stablehlo.broadcast_in_dim {g}, dims = [] : (tensor<f32>) -> {tn}\n" ++
  s!"    %{o}btb = stablehlo.broadcast_in_dim {bt}, dims = [] : (tensor<f32>) -> {tn}\n" ++
  s!"    %{o}gx = stablehlo.multiply %{o}xh, %{o}gb : {tn}\n" ++
  s!"    %{o}fl = stablehlo.add %{o}gx, %{o}btb : {tn}\n" ++
  s!"    %{o} = stablehlo.reshape %{o}fl : ({tn}) -> {ty [BS,c,Hh,Ww]}\n"

private def layerScaleF (o x g : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.multiply {x}, %{o}gb : {ty [BS,c,Hh,Ww]}\n"

/-- Patchify forward: s×s stride-s pad-0 conv, `[B,ic,s·Ho,s·Wo] → [B,oc,Ho,Wo]`. -/
private def patchify (o x w bnm : String) (oc ic Ho Wo s : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [{s}, {s}], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,s*Ho,s*Wo]}, {ty [oc,ic,s,s]}) -> {ty [BS,oc,Ho,Wo]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Ho,Wo]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Ho,Wo]}\n"

/-- Downsample conv forward: 2×2 stride-2 pad-0 conv, `[B,ic,2H,2W] → [B,oc,H,W]`. -/
private def convDown (o x w bnm : String) (oc ic H W : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,2*H,2*W]}, {ty [oc,ic,2,2]}) -> {ty [BS,oc,H,W]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,H,W]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,H,W]}\n"

-- ════════════ block / downsample / stage ════════════

/-- One ConvNeXt block (stride-1, identity skip), dim `c`, expand `e=4c`, spatial `h`. -/
private def blockFwd (p x : String) (c e h : Nat) : String × String :=
  let code :=
    dwconv s!"{p}d" x s!"%{p}dW" s!"%{p}db" c h h 7 ++
    lnFwd s!"{p}n" s!"%{p}d" s!"%{p}ng" s!"%{p}nbt" c h h ++
    conv1 s!"{p}e" s!"%{p}n" s!"%{p}eW" s!"%{p}eb" e c h h ++
    geluAct s!"{p}g" s!"%{p}e" e h h ++
    conv1 s!"{p}p" s!"%{p}g" s!"%{p}pW" s!"%{p}pb" c e h h ++
    layerScaleF s!"{p}ls" s!"%{p}p" s!"%{p}lg" c h h ++
    addOp s!"{p}o" s!"%{p}ls" x c h h
  (code, s!"%{p}o")

/-- Downsample: LN(global scalar) over `[B,ci,hin,hin]` then 2×2/s2 conv `ci→co`. -/
private def downFwd (d x : String) (ci co hin : Nat) : String × String :=
  let code :=
    lnFwd s!"{d}n" x s!"%{d}ng" s!"%{d}nbt" ci hin hin ++
    convDown s!"{d}c" s!"%{d}n" s!"%{d}W" s!"%{d}b" co ci (hin/2) (hin/2)
  (code, s!"%{d}c")

private def depths : Array Nat := #[3, 3, 9, 3]
private def dims   : Array Nat := #[96, 192, 384, 768]
private def spats  : Array Nat := #[56, 28, 14, 7]

-- ════════════ param signature (forward order) ════════════

private def blockSig (p : String) (c e : Nat) : List String :=
  [s!"%{p}dW: {ty [c,1,7,7]}", s!"%{p}db: {ty [c]}",
   s!"%{p}ng: tensor<f32>", s!"%{p}nbt: tensor<f32>",
   s!"%{p}eW: {ty [e,c,1,1]}", s!"%{p}eb: {ty [e]}",
   s!"%{p}pW: {ty [c,e,1,1]}", s!"%{p}pb: {ty [c]}",
   s!"%{p}lg: {ty [c]}"]

private def downSig (d : String) (ci co : Nat) : List String :=
  [s!"%{d}ng: tensor<f32>", s!"%{d}nbt: tensor<f32>",
   s!"%{d}W: {ty [co,ci,2,2]}", s!"%{d}b: {ty [co]}"]

/-- Architecture param signature, in func-arg order (stem → stages+downsamples → head). -/
private def archSig : List String := Id.run do
  let mut sig : List String := [s!"%psW: {ty [96,3,4,4]}", s!"%psb: {ty [96]}"]
  for si in [0:4] do
    let c := dims[si]!
    let e := 4 * c
    for j in [0:depths[si]!] do
      sig := sig ++ blockSig s!"s{si}b{j}" c e
    if si < 3 then
      sig := sig ++ downSig s!"d{si}" c (dims[si+1]!)
  sig := sig ++ [s!"%hng: tensor<f32>", s!"%hnbt: tensor<f32>",
                 s!"%Wd: {ty [768,10]}", s!"%bd: {ty [10]}"]
  return sig

-- ════════════ whole net forward ════════════

private def convnextFwd : String := Id.run do
  let mut body := "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n"
    ++ s!"    %xr = stablehlo.reshape %x : ({ty [BS,3*IMG*IMG]}) -> {ty [BS,3,IMG,IMG]}\n"
    ++ patchify "ps" "%xr" "%psW" "%psb" 96 3 56 56 4
  let mut cur := "%ps"
  for si in [0:4] do
    let c := dims[si]!
    let e := 4 * c
    let h := spats[si]!
    for j in [0:depths[si]!] do
      let (code, out) := blockFwd s!"s{si}b{j}" cur c e h
      body := body ++ code; cur := out
    if si < 3 then
      let (code, out) := downFwd s!"d{si}" cur c (dims[si+1]!) h
      body := body ++ code; cur := out
  -- head: GAP → LN(768) → dense 768→10
  body := body
    ++ s!"    %gaps = stablehlo.reduce({cur} init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,768,7,7]}, tensor<f32>) -> {ty [BS,768]}\n"
    ++ s!"    %gapnf = stablehlo.constant dense<49.0> : {ty [BS,768]}\n"
    ++ s!"    %gap = stablehlo.divide %gaps, %gapnf : {ty [BS,768]}\n"
    ++ s!"    %gapr = stablehlo.reshape %gap : ({ty [BS,768]}) -> {ty [BS,768,1,1]}\n"
    ++ lnFwd "hn" "%gapr" "%hng" "%hnbt" 768 1 1
    ++ s!"    %hnf = stablehlo.reshape %hn : ({ty [BS,768,1,1]}) -> {ty [BS,768]}\n"
    ++ s!"    %ld = stablehlo.dot_general %hnf, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,768]}, {ty [768,10]}) -> {ty [BS,10]}\n"
    ++ s!"    %ldb = stablehlo.broadcast_in_dim %bd, dims = [1] : ({ty [10]}) -> {ty [BS,10]}\n"
    ++ s!"    %out = stablehlo.add %ld, %ldb : {ty [BS,10]}\n"
  let argSig := String.intercalate ", " (("%x: " ++ ty [BS,3*IMG*IMG]) :: archSig)
  return "module @m {\n" ++ s!"  func.func @convnext_fwd({argSig}) -> {ty [BS,10]} " ++ "{\n" ++
    body ++ s!"    return %out : {ty [BS,10]}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := convnextFwd
  IO.println s!"rendered @convnext_fwd ConvNeXt-T (BS={BS}, [3,3,9,3]): {mlir.length} chars"
  IO.FS.createDirAll "verified_mlir"
  IO.FS.writeFile "verified_mlir/convnext_fwd.mlir" mlir
  IO.FS.createDirAll ".lake/build"
  let cargs ← ireeCompileArgs "verified_mlir/convnext_fwd.mlir" ".lake/build/convnext_fwd_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println "convnext_fwd ConvNeXt-T iree-compile OK → .lake/build/convnext_fwd_v.vmfb"

#eval main
