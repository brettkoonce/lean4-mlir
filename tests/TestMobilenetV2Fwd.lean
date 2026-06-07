import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! # C4a — MobileNetV2 forward renderer + `iree-compile` validation

Programmatic StableHLO for a small but real MobileNetV2 forward (CIFAR 3×32×32),
matching the proven apex structure `dense ∘ GAP ∘ invresBody ∘ residual(invresBody)
∘ stem`:
  stem  : 3×3 stride-2 conv (3→32, 32→16) + BN + relu6, then 2×2 maxpool (16→8)
  IR-A  : inverted-residual WITH skip   (ic=oc=32, mid=t·ic=64, stride 1 @8×8)
  IR-B  : inverted-residual WITHOUT skip (ic=32→oc=64, mid=64, stride 1 @8×8)
  head  : 1×1 conv (64→128) + BN + relu6  (the MNv2 "features" layer @8×8)
  tail  : global-average-pool → dense(128→10)

The head's relu6 before GAP is ESSENTIAL: per-example instance-norm forces each
channel's spatial mean to 0, so GAP of a raw (linear-bottleneck) BN output is the
constant β — input-independent. The relu6 gives the pooled tensor a per-input mean.

Every fragment is the same StableHLO the VERIFIED per-op emitters produce: the
stem uses a regular stride-2 conv (ch6 `flatConvStridedF`), the inverted-residual
blocks use the C1 depthwise op (`depthwiseF`, feature_group_count=c, [c,1,3,3]
kernel), C2 relu6 (`relu6F`), per-channel BN (`bnPerChannelF`), 1×1 convs, GAP and
dense. Stride-1 IR blocks = the proven apex (no C3 strided depthwise needed). This
de-risks the whole MNv2 forward on `iree-compile` before wiring the train step.

Naming: helper `o` is a bare prefix; result SSA = `%{o}`, intermediates `%{o}…`.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestMobilenetV2Fwd.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 128
private def EPS : String := "1.0e-5"

-- ── 4-D fragment helpers ([B,C,H,W] throughout; structured names) ──

/-- 3×3 SAME conv (stride `s`, pad 1) + bias broadcast — for the stem. Result `%{o}`. -/
private def conv3 (o x w bnm : String) (oc ic Hin Win Hout Wout s : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [{s}, {s}], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,Hin,Win]}, {ty [oc,ic,3,3]}) -> {ty [BS,oc,Hout,Wout]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hout,Wout]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Hout,Wout]}\n"

/-- 1×1 conv (pad 0, stride 1) + bias — the expand/project convs. Result `%{o}`. -/
private def conv1 (o x w bnm : String) (oc ic Hh Ww : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,Hh,Ww]}, {ty [oc,ic,1,1]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Hh,Ww]}\n"

/-- Depthwise 3×3 SAME conv (stride 1, pad 1, feature_group_count=c, [c,1,3,3]
    kernel) + bias — the C1 `depthwiseF` op. Result `%{o}`. -/
private def dwconv (o x w bnm : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,Hh,Ww]}, {ty [c,1,3,3]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,c,Hh,Ww]}\n"

/-- Per-channel BatchNorm forward (4-D), reduce μ/var over spatial `[2,3]`, rank-1
    γ/β `dims=[1]`. The bnPerChannelF op pattern, kept 4-D. Result `%{o}`. -/
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

/-- ReLU6 forward: clamp to [0,6] as `min(max(x,0),6)` (the C2 `relu6F` op). -/
private def relu6 (o x : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}z = stablehlo.constant dense<0.0> : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}six = stablehlo.constant dense<6.0> : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}mx = stablehlo.maximum {x}, %{o}z : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.minimum %{o}mx, %{o}six : {ty [BS,c,Hh,Ww]}\n"

private def maxpool (o x : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}ni = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
  s!"    %{o} = \"stablehlo.reduce_window\"({x}, %{o}ni) (" ++ "{\n" ++
  "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
  "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
  "        stablehlo.return %pm : tensor<f32>\n" ++
  "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
  s!" : ({ty [BS,c,Hh,Ww]}, tensor<f32>) -> {ty [BS,c,Hh/2,Ww/2]}\n"

private def addOp (o a b : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.add {a}, {b} : {ty [BS,oc,Hh,Ww]}\n"

-- ── inverted-residual block: expand 1×1 → bn → relu6 → dw → bn → relu6 →
--    project 1×1 → bn (no relu6); + residual skip iff ic=oc (stride 1). ──

private def irBlockFwd (p x : String) (ic mid oc Hh Ww : Nat) (skip : Bool) : String × String :=
  let m := Hh*Ww
  let body :=
    conv1 s!"{p}e" x s!"%{p}eW" s!"%{p}eb" mid ic Hh Ww ++
    bnPC s!"{p}en" s!"%{p}e" s!"%{p}eg" s!"%{p}ebt" mid Hh Ww m ++
    relu6 s!"{p}er" s!"%{p}en" mid Hh Ww ++
    dwconv s!"{p}d" s!"%{p}er" s!"%{p}dW" s!"%{p}db" mid Hh Ww ++
    bnPC s!"{p}dn" s!"%{p}d" s!"%{p}dg" s!"%{p}dbt" mid Hh Ww m ++
    relu6 s!"{p}dr" s!"%{p}dn" mid Hh Ww ++
    conv1 s!"{p}p" s!"%{p}dr" s!"%{p}pW" s!"%{p}pb" oc mid Hh Ww ++
    bnPC s!"{p}pn" s!"%{p}p" s!"%{p}pg" s!"%{p}pbt" oc Hh Ww m
  if skip then
    (body ++ addOp s!"{p}o" s!"%{p}pn" x oc Hh Ww, s!"%{p}o")
  else
    (body, s!"%{p}pn")

-- ── parameter-signature generators (mirror the body's names + types) ──

private def bnSig (p : String) (oc : Nat) : List String :=
  [s!"%{p}g: {ty [oc]}", s!"%{p}bt: {ty [oc]}"]

/-- IR block params in func-arg order: expand(1×1) / depthwise(3×3) / project(1×1). -/
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
  -- stem: 3×3 stride-2 conv (3→32, 32→16) + BN + relu6, then 2×2 maxpool (16→8)
  let stemCode :=
    s!"    %xr = stablehlo.reshape %x : ({ty [BS,3072]}) -> {ty [BS,3,32,32]}\n" ++
    conv3 "stc" "%xr" "%sW" "%sb" 32 3 32 32 16 16 2 ++
    bnPC "stn" "%stc" "%sg" "%sbt" 32 16 16 (16*16) ++
    relu6 "str" "%stn" 32 16 16 ++
    maxpool "stp" "%str" 32 16 16
  let (a, oa) := irBlockFwd "ira" "%stp" 32 64 32 8 8 true   -- skip block (ic=oc=32)
  let (b, ob) := irBlockFwd "irb" oa 32 64 64 8 8 false      -- no-skip block (32→64)
  -- head: 1×1 conv (64→128) → BN → relu6 — the standard MNv2 "features" layer.
  -- ESSENTIAL: GAP of a per-example instance-normed BN is just β (constant across
  -- inputs); the relu6 here gives the pooled tensor an input-varying mean.
  let head :=
    conv1 "h" ob "%hW" "%hb" 128 64 8 8 ++
    bnPC "hn" "%h" "%hg" "%hbt" 128 8 8 (8*8) ++
    relu6 "hr" "%hn" 128 8 8
  let tail := gapDense "out" "%hr" 128 10 8 8
  let body :=
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    stemCode ++ a ++ b ++ head ++ tail
  let sig : List String :=
    ["%x: " ++ ty [BS,3072]]
    ++ [s!"%sW: {ty [32,3,3,3]}", s!"%sb: {ty [32]}"] ++ bnSig "s" 32
    ++ irBlockSig "ira" 32 64 32
    ++ irBlockSig "irb" 32 64 64
    ++ [s!"%hW: {ty [128,64,1,1]}", s!"%hb: {ty [128]}"] ++ bnSig "h" 128
    ++ [s!"%Wd: {ty [128,10]}", s!"%bd: {ty [10]}"]
  let argSig := String.intercalate ", " sig
  return "module @m {\n" ++ s!"  func.func @mobilenetv2_fwd({argSig}) -> {ty [BS,10]} " ++ "{\n" ++
    body ++ s!"    return %out : {ty [BS,10]}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := mobilenetv2Fwd
  IO.println s!"rendered @mobilenetv2_fwd (BS={BS}): {mlir.length} chars"
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
