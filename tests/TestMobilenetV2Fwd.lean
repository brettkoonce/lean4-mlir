import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! # C4a/D3 — MobileNetV2 forward renderer (real downsampling [t,c,n,s]) + iree

Programmatic StableHLO for a real, DOWNSAMPLING MobileNetV2 forward (CIFAR 3×32×32),
matching the reference architecture (inverted-residual `[t,c,n,s]` stages that shrink
spatial via STRIDE-2 DEPTHWISE — the C3 `depthwiseStridedF` op):

  stem  : 3×3 stride-2 conv (3→16, 32→16) + BN + relu6
  b1    : IR  16→24, t≈4 (mid 64),  stride 2  (16→8)   [no skip]
  b2    : IR  24→24, mid 96,        stride 1  (8×8)     [skip]
  b3    : IR  24→32, mid 96,        stride 2  (8→4)     [no skip]
  b4    : IR  32→32, mid 128,       stride 1  (4×4)     [skip]
  b5    : IR  32→64, mid 128,       stride 1  (4×4)     [no skip]
  b6    : IR  64→64, mid 256,       stride 1  (4×4)     [skip]
  head  : 1×1 conv (64→128) + BN + relu6  (MNv2 "features" layer @4×4)
  tail  : global-average-pool → dense(128→10)

Spatial: 32→16(stem)→8(b1)→4(b3)→4 — two strided-depthwise downsamples. Every
fragment is the StableHLO a VERIFIED per-op emitter produces: depthwise stride-1
(`depthwiseF`) / stride-2 (`depthwiseStridedF`), relu6 (`relu6F`), per-channel BN
(`bnPerChannelF`), 1×1/3×3 convs, residual `addV`, GAP, dense. A block downsamples
(stride 2, no skip) or keeps shape (stride 1, skip iff ic=oc). The head's relu6
before GAP is essential (per-example instance-norm zeroes the spatial mean, so GAP
of a raw linear-bottleneck BN is the constant β).

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestMobilenetV2Fwd.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 128
private def EPS : String := "1.0e-5"

-- ── 4-D fragment helpers ([B,C,H,W]; structured names, result `%{o}`) ──

/-- 3×3 SAME conv (stride `s`, pad 1) + bias — the stem. -/
private def conv3 (o x w bnm : String) (oc ic Hin Win Hout Wout s : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [{s}, {s}], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,Hin,Win]}, {ty [oc,ic,3,3]}) -> {ty [BS,oc,Hout,Wout]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hout,Wout]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Hout,Wout]}\n"

/-- 1×1 conv (pad 0, stride 1) + bias — expand/project/head. -/
private def conv1 (o x w bnm : String) (oc ic Hh Ww : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,Hh,Ww]}, {ty [oc,ic,1,1]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Hh,Ww]}\n"

/-- Depthwise 3×3 SAME conv, STRIDE 1 (feature_group_count=c, [c,1,3,3]). -/
private def dwconv (o x w bnm : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,Hh,Ww]}, {ty [c,1,3,3]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,c,Hh,Ww]}\n"

/-- Depthwise 3×3 SAME conv, STRIDE 2 (the C3 `depthwiseStridedF`): input
    `[B,c,2Hout,2Wout]` → output `[B,c,Hout,Wout]` (halves spatial). -/
private def dwconvStrided (o x w bnm : String) (c Hout Wout : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,2*Hout,2*Wout]}, {ty [c,1,3,3]}) -> {ty [BS,c,Hout,Wout]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hout,Wout]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,c,Hout,Wout]}\n"

/-- Per-channel BN forward (reduce μ/var over spatial [2,3], rank-1 γ/β dims=[1]). -/
private def bnPC (o x g bt : String) (oc Hh Ww m : Nat) : String :=
  s!"    %{o}nf = stablehlo.constant dense<{m}.0> : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}ep = stablehlo.constant dense<{EPS}> : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}smr = stablehlo.reduce({x} init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [BS,oc]}\n" ++
  s!"    %{o}sm = stablehlo.broadcast_in_dim %{o}smr, dims = [0, 1] : ({ty [BS,oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}mu = stablehlo.divide %{o}sm, %{o}nf : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xc = stablehlo.subtract {x}, %{o}mu : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}sq = stablehlo.multiply %{o}xc, %{o}xc : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}vsr = stablehlo.reduce(%{o}sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [BS,oc]}\n" ++
  s!"    %{o}vs = stablehlo.broadcast_in_dim %{o}vsr, dims = [0, 1] : ({ty [BS,oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}vr = stablehlo.divide %{o}vs, %{o}nf : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}ve = stablehlo.add %{o}vr, %{o}ep : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}istd = stablehlo.rsqrt %{o}ve : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xh = stablehlo.multiply %{o}xc, %{o}istd : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}btb = stablehlo.broadcast_in_dim {bt}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}gx = stablehlo.multiply %{o}xh, %{o}gb : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}gx, %{o}btb : {ty [BS,oc,Hh,Ww]}\n"

private def relu6 (o x : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}z = stablehlo.constant dense<0.0> : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}six = stablehlo.constant dense<6.0> : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}mx = stablehlo.maximum {x}, %{o}z : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.minimum %{o}mx, %{o}six : {ty [BS,c,Hh,Ww]}\n"

private def addOp (o a b : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.add {a}, {b} : {ty [BS,oc,Hh,Ww]}\n"

-- ── inverted-residual block (stride `s`): expand 1×1 → BN → relu6 → depthwise
--    (stride `s`) → BN → relu6 → project 1×1 → BN; + residual iff s=1 ∧ ic=oc ──

private def irBlockFwd (p x : String) (ic mid oc Hin s : Nat) : String × String :=
  let Hout := Hin / s
  let body :=
    conv1 s!"{p}e" x s!"%{p}eW" s!"%{p}eb" mid ic Hin Hin ++
    bnPC s!"{p}en" s!"%{p}e" s!"%{p}eg" s!"%{p}ebt" mid Hin Hin (Hin*Hin) ++
    relu6 s!"{p}er" s!"%{p}en" mid Hin Hin ++
    (if s == 2 then dwconvStrided s!"{p}d" s!"%{p}er" s!"%{p}dW" s!"%{p}db" mid Hout Hout
     else dwconv s!"{p}d" s!"%{p}er" s!"%{p}dW" s!"%{p}db" mid Hin Hin) ++
    bnPC s!"{p}dn" s!"%{p}d" s!"%{p}dg" s!"%{p}dbt" mid Hout Hout (Hout*Hout) ++
    relu6 s!"{p}dr" s!"%{p}dn" mid Hout Hout ++
    conv1 s!"{p}p" s!"%{p}dr" s!"%{p}pW" s!"%{p}pb" oc mid Hout Hout ++
    bnPC s!"{p}pn" s!"%{p}p" s!"%{p}pg" s!"%{p}pbt" oc Hout Hout (Hout*Hout)
  if s == 1 && ic == oc then
    (body ++ addOp s!"{p}o" s!"%{p}pn" x oc Hout Hout, s!"%{p}o")
  else
    (body, s!"%{p}pn")

-- ── block config (p, ic, mid, oc, s); spatial threaded from the stem (16×16) ──

private def blocks : List (String × Nat × Nat × Nat × Nat) :=
  [("b1", 16, 64,  24, 2),   -- 16→8
   ("b2", 24, 96,  24, 1),   -- skip
   ("b3", 24, 96,  32, 2),   -- 8→4
   ("b4", 32, 128, 32, 1),   -- skip
   ("b5", 32, 128, 64, 1),   -- 32→64 (no skip)
   ("b6", 64, 256, 64, 1)]   -- skip

private def bnSig (p : String) (oc : Nat) : List String :=
  [s!"%{p}g: {ty [oc]}", s!"%{p}bt: {ty [oc]}"]
private def irBlockSig (p : String) (ic mid oc : Nat) : List String :=
  [s!"%{p}eW: {ty [mid,ic,1,1]}", s!"%{p}eb: {ty [mid]}", s!"%{p}eg: {ty [mid]}", s!"%{p}ebt: {ty [mid]}",
   s!"%{p}dW: {ty [mid,1,3,3]}", s!"%{p}db: {ty [mid]}", s!"%{p}dg: {ty [mid]}", s!"%{p}dbt: {ty [mid]}",
   s!"%{p}pW: {ty [oc,mid,1,1]}", s!"%{p}pb: {ty [oc]}", s!"%{p}pg: {ty [oc]}", s!"%{p}pbt: {ty [oc]}"]

private def gapDense (o x : String) (c nC Hh Ww : Nat) : String :=
  s!"    %{o}gs = stablehlo.reduce({x} init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,c,Hh,Ww]}, tensor<f32>) -> {ty [BS,c]}\n" ++
  s!"    %{o}gnf = stablehlo.constant dense<{Hh*Ww}.0> : {ty [BS,c]}\n" ++
  s!"    %{o}g = stablehlo.divide %{o}gs, %{o}gnf : {ty [BS,c]}\n" ++
  s!"    %{o}dd = stablehlo.dot_general %{o}g, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,c]}, {ty [c,nC]}) -> {ty [BS,nC]}\n" ++
  s!"    %{o}db = stablehlo.broadcast_in_dim %bd, dims = [1] : ({ty [nC]}) -> {ty [BS,nC]}\n" ++
  s!"    %{o} = stablehlo.add %{o}dd, %{o}db : {ty [BS,nC]}\n"

-- ── whole net ──

private def mobilenetv2Fwd : String := Id.run do
  -- stem: 3×3 stride-2 conv (3→16, 32→16) + BN + relu6
  let stemCode :=
    s!"    %xr = stablehlo.reshape %x : ({ty [BS,3072]}) -> {ty [BS,3,32,32]}\n" ++
    conv3 "stc" "%xr" "%sW" "%sb" 16 3 32 32 16 16 2 ++
    bnPC "stn" "%stc" "%sg" "%sbt" 16 16 16 (16*16) ++
    relu6 "str" "%stn" 16 16 16
  let mut blkCode := ""
  let mut cur := "%str"
  let mut curH := 16
  for (p, ic, mid, oc, s) in blocks do
    let (c, out) := irBlockFwd p cur ic mid oc curH s
    blkCode := blkCode ++ c; cur := out; curH := curH / s
  -- head: 1×1 conv (64→128) + BN + relu6 @ curH×curH (=4×4)
  let head :=
    conv1 "h" cur "%hW" "%hb" 128 64 curH curH ++
    bnPC "hn" "%h" "%hg" "%hbt" 128 curH curH (curH*curH) ++
    relu6 "hr" "%hn" 128 curH curH
  let tail := gapDense "out" "%hr" 128 10 curH curH
  let body :=
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    stemCode ++ blkCode ++ head ++ tail
  let sig : List String :=
    ["%x: " ++ ty [BS,3072]]
    ++ [s!"%sW: {ty [16,3,3,3]}", s!"%sb: {ty [16]}"] ++ bnSig "s" 16
    ++ (blocks.map (fun (p, ic, mid, oc, _) => irBlockSig p ic mid oc)).flatten
    ++ [s!"%hW: {ty [128,64,1,1]}", s!"%hb: {ty [128]}"] ++ bnSig "h" 128
    ++ [s!"%Wd: {ty [128,10]}", s!"%bd: {ty [10]}"]
  let argSig := String.intercalate ", " sig
  return "module @m {\n" ++ s!"  func.func @mobilenetv2_fwd({argSig}) -> {ty [BS,10]} " ++ "{\n" ++
    body ++ s!"    return %out : {ty [BS,10]}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := mobilenetv2Fwd
  IO.println s!"rendered @mobilenetv2_fwd (BS={BS}, {blocks.length} IR blocks): {mlir.length} chars"
  IO.FS.createDirAll "verified_mlir"
  IO.FS.writeFile "verified_mlir/mobilenetv2_fwd.mlir" mlir
  IO.FS.createDirAll ".lake/build"
  let path := "verified_mlir/mobilenetv2_fwd.mlir"
  let cargs ← ireeCompileArgs path ".lake/build/mobilenetv2_fwd_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println "mobilenetv2_fwd iree-compile OK → .lake/build/mobilenetv2_fwd_v.vmfb"

#eval main
