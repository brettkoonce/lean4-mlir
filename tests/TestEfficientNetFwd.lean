import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! # E6 — EfficientNet-B0 forward renderer (faithful [t,c,n,s,k] config) + iree

Programmatic StableHLO for the **EfficientNet-B0** architecture (Tan & Le 2019), all-swish
with batch norm (E5), on **Imagenette 224×224, 10 classes** — the native B0 resolution
(stem stride 2, 5 downsamples 224→7; the real B0 strides/widths/kernels/expand-ratios):

  stem : 3×3 stride-2 conv (3→32) + BN + swish   (224→112)
  B0 stages [expand t, channels c, repeats n, stride s, kernel k]:
    s1: t1 c16  n1 s1 k3   (MBConv1, NO expand conv)   @112
    s2: t6 c24  n2 s2 k3                                112→56
    s3: t6 c40  n2 s2 k5                                56→28
    s4: t6 c80  n3 s2 k3                                28→14
    s5: t6 c112 n3 s1 k5                                @14
    s6: t6 c192 n4 s2 k5                                14→7
    s7: t6 c320 n1 s1 k3                                @7
  head : 1×1 conv (320→1280) + BN + swish  →  GAP → dense(1280→10)
  16 MBConv layers total; SE (ratio 0.25 of block-input ch) in every block.

MBConv: expand 1×1 conv→BN→swish (SKIPPED when t=1) → depthwise k×k (stride 1/2)→BN→swish
→ SE gate → project 1×1 conv→BN; + residual iff s=1 ∧ ic=oc. Every fragment is the StableHLO
a VERIFIED per-op emitter produces: swish (`swishF`), sigmoid (`sigmoidF`), batch norm
(`bnBatchTensor4`, E5), depthwise k×k stride-1/2 (`depthwiseF`/`depthwiseStridedF`, the op is
kernel-general), 1×1/3×3 convs, residual `addV`, GAP, dense; the SE mirrors
`seGate`/`seBlock`/`broadcastFlat`. Batch-norm keeps the final GAP non-degenerate (all-swish).

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestEfficientNetFwd.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 32      -- 224² B0 is memory-heavy; small batch
private def IMG : Nat := 224    -- Imagenette resolution (B0 native, stem stride 2 → 112)
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

/-- Depthwise k×k SAME conv, STRIDE 1 (feature_group_count=c, [c,1,k,k], pad (k-1)/2). -/
private def dwconv (o x w bnm : String) (c Hh Ww k : Nat) : String :=
  let p := (k-1)/2
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,Hh,Ww]}, {ty [c,1,k,k]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,c,Hh,Ww]}\n"

/-- Depthwise k×k SAME conv, STRIDE 2 ([c,1,k,k], pad (k-1)/2): `[B,c,2Hout,2Wout]` →
    `[B,c,Hout,Wout]` (halves spatial). -/
private def dwconvStrided (o x w bnm : String) (c Hout Wout k : Nat) : String :=
  let p := (k-1)/2
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [2, 2], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,2*Hout,2*Wout]}, {ty [c,1,k,k]}) -> {ty [BS,c,Hout,Wout]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hout,Wout]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,c,Hout,Wout]}\n"

/-- **Batch-norm per channel** (E5): reduce μ/var over batch+spatial [0,2,3] per channel
    (count N·H·W), rank-1 γ/β dims=[1]. The proven `bnBatchTensor4`. -/
private def bnBatch (o x g bt : String) (oc Hh Ww _m : Nat) : String :=
  let nf := BS * Hh * Ww
  s!"    %{o}nf = stablehlo.constant dense<{nf}.0> : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}ep = stablehlo.constant dense<{EPS}> : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}smr = stablehlo.reduce({x} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n" ++
  s!"    %{o}sm = stablehlo.broadcast_in_dim %{o}smr, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}mu = stablehlo.divide %{o}sm, %{o}nf : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xc = stablehlo.subtract {x}, %{o}mu : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}sq = stablehlo.multiply %{o}xc, %{o}xc : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}vsr = stablehlo.reduce(%{o}sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n" ++
  s!"    %{o}vs = stablehlo.broadcast_in_dim %{o}vsr, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}vr = stablehlo.divide %{o}vs, %{o}nf : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}ve = stablehlo.add %{o}vr, %{o}ep : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}istd = stablehlo.rsqrt %{o}ve : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xh = stablehlo.multiply %{o}xc, %{o}istd : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}btb = stablehlo.broadcast_in_dim {bt}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}gx = stablehlo.multiply %{o}xh, %{o}gb : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}gx, %{o}btb : {ty [BS,oc,Hh,Ww]}\n"

/-- Swish forward `y = x · σ(x)` (= emitTok swishF). -/
private def swishAct (o x : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}s = stablehlo.logistic {x} : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.multiply {x}, %{o}s : {ty [BS,c,Hh,Ww]}\n"

private def addOp (o a b : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.add {a}, {b} : {ty [BS,oc,Hh,Ww]}\n"

/-- SE forward (squeeze→dense₁→swish→dense₂→sigmoid→bcast×x). Produces `%{p}se`. -/
private def seFwd (p x Ws1 bs1 Ws2 bs2 : String) (c h w r : Nat) : String :=
  s!"    %{p}sqs = stablehlo.reduce({x} init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,c,h,w]}, tensor<f32>) -> {ty [BS,c]}\n" ++
  s!"    %{p}sqnf = stablehlo.constant dense<{h*w}.0> : {ty [BS,c]}\n" ++
  s!"    %{p}sq = stablehlo.divide %{p}sqs, %{p}sqnf : {ty [BS,c]}\n" ++
  s!"    %{p}exd = stablehlo.dot_general %{p}sq, {Ws1}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,c]}, {ty [c,r]}) -> {ty [BS,r]}\n" ++
  s!"    %{p}exbb = stablehlo.broadcast_in_dim {bs1}, dims = [1] : ({ty [r]}) -> {ty [BS,r]}\n" ++
  s!"    %{p}ex = stablehlo.add %{p}exd, %{p}exbb : {ty [BS,r]}\n" ++
  s!"    %{p}a1s = stablehlo.logistic %{p}ex : {ty [BS,r]}\n" ++
  s!"    %{p}a1 = stablehlo.multiply %{p}ex, %{p}a1s : {ty [BS,r]}\n" ++
  s!"    %{p}h2d = stablehlo.dot_general %{p}a1, {Ws2}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,r]}, {ty [r,c]}) -> {ty [BS,c]}\n" ++
  s!"    %{p}h2bb = stablehlo.broadcast_in_dim {bs2}, dims = [1] : ({ty [c]}) -> {ty [BS,c]}\n" ++
  s!"    %{p}h2 = stablehlo.add %{p}h2d, %{p}h2bb : {ty [BS,c]}\n" ++
  s!"    %{p}gate = stablehlo.logistic %{p}h2 : {ty [BS,c]}\n" ++
  s!"    %{p}gb = stablehlo.broadcast_in_dim %{p}gate, dims = [0, 1] : ({ty [BS,c]}) -> {ty [BS,c,h,w]}\n" ++
  s!"    %{p}se = stablehlo.multiply {x}, %{p}gb : {ty [BS,c,h,w]}\n"

-- ── MBConv block (stride `s`, kernel `k`): expand 1×1 → BN → swish (SKIPPED if t=1,
--    i.e. mid=ic) → depthwise k×k (stride `s`) → BN → swish → SE → project 1×1 → BN;
--    + residual iff s=1 ∧ ic=oc ──

private def mbconvFwd (p x : String) (ic mid oc Hin s r k : Nat) : String × String :=
  let Hout := Hin / s
  let hasExpand := mid != ic    -- t=1 (MBConv1) ⇒ mid=ic ⇒ no expand conv
  let (exC, dwIn) :=
    if hasExpand then
      (conv1 s!"{p}e" x s!"%{p}eW" s!"%{p}eb" mid ic Hin Hin ++
       bnBatch s!"{p}en" s!"%{p}e" s!"%{p}eg" s!"%{p}ebt" mid Hin Hin (Hin*Hin) ++
       swishAct s!"{p}es" s!"%{p}en" mid Hin Hin, s!"%{p}es")
    else ("", x)
  let body :=
    exC ++
    (if s == 2 then dwconvStrided s!"{p}d" dwIn s!"%{p}dW" s!"%{p}db" mid Hout Hout k
     else dwconv s!"{p}d" dwIn s!"%{p}dW" s!"%{p}db" mid Hin Hin k) ++
    bnBatch s!"{p}dn" s!"%{p}d" s!"%{p}dg" s!"%{p}dbt" mid Hout Hout (Hout*Hout) ++
    swishAct s!"{p}ds" s!"%{p}dn" mid Hout Hout ++
    seFwd s!"{p}z" s!"%{p}ds" s!"%{p}zW1" s!"%{p}zb1" s!"%{p}zW2" s!"%{p}zb2" mid Hout Hout r ++
    conv1 s!"{p}p" s!"%{p}zse" s!"%{p}pW" s!"%{p}pb" oc mid Hout Hout ++
    bnBatch s!"{p}pn" s!"%{p}p" s!"%{p}pg" s!"%{p}pbt" oc Hout Hout (Hout*Hout)
  if s == 1 && ic == oc then
    (body ++ addOp s!"{p}o" s!"%{p}pn" x oc Hout Hout, s!"%{p}o")
  else
    (body, s!"%{p}pn")

-- ── EfficientNet-B0 stage spec (t, c, n, s, k); stem out = 32, CIFAR stem stride 1 ──

private def stages : List (Nat × Nat × Nat × Nat × Nat) :=
  [(1, 16,  1, 1, 3), (6, 24,  2, 2, 3), (6, 40,  2, 2, 5), (6, 80,  3, 2, 3),
   (6, 112, 3, 1, 5), (6, 192, 4, 2, 5), (6, 320, 1, 1, 3)]

/-- Expand the stage spec to per-block `(p, ic, mid, oc, s, r, k)`: first layer of a stage
    does `prev→c` at stride `s`, the rest `c→c` at stride 1 (skip); mid = t·ic, SE r = ic/4. -/
private def blocks : List (String × Nat × Nat × Nat × Nat × Nat × Nat) := Id.run do
  let mut bs : List (String × Nat × Nat × Nat × Nat × Nat × Nat) := []
  let mut prev := 32
  let mut idx := 1
  for (t, c, n, s, k) in stages do
    for j in [0:n] do
      let ic := if j == 0 then prev else c
      let stride := if j == 0 then s else 1
      bs := bs ++ [(s!"b{idx}", ic, t * ic, c, stride, max 1 (ic / 4), k)]
      idx := idx + 1
    prev := c
  return bs

private def bnSig (p : String) (oc : Nat) : List String :=
  [s!"%{p}g: {ty [oc]}", s!"%{p}bt: {ty [oc]}"]
private def mbconvSig (p : String) (ic mid oc r k : Nat) : List String :=
  (if mid != ic then
    [s!"%{p}eW: {ty [mid,ic,1,1]}", s!"%{p}eb: {ty [mid]}", s!"%{p}eg: {ty [mid]}", s!"%{p}ebt: {ty [mid]}"]
   else []) ++
  [s!"%{p}dW: {ty [mid,1,k,k]}", s!"%{p}db: {ty [mid]}", s!"%{p}dg: {ty [mid]}", s!"%{p}dbt: {ty [mid]}",
   s!"%{p}zW1: {ty [mid,r]}", s!"%{p}zb1: {ty [r]}", s!"%{p}zW2: {ty [r,mid]}", s!"%{p}zb2: {ty [mid]}",
   s!"%{p}pW: {ty [oc,mid,1,1]}", s!"%{p}pb: {ty [oc]}", s!"%{p}pg: {ty [oc]}", s!"%{p}pbt: {ty [oc]}"]

private def gapDense (o x : String) (c nC Hh Ww : Nat) : String :=
  s!"    %{o}gs = stablehlo.reduce({x} init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,c,Hh,Ww]}, tensor<f32>) -> {ty [BS,c]}\n" ++
  s!"    %{o}gnf = stablehlo.constant dense<{Hh*Ww}.0> : {ty [BS,c]}\n" ++
  s!"    %{o}g = stablehlo.divide %{o}gs, %{o}gnf : {ty [BS,c]}\n" ++
  s!"    %{o}dd = stablehlo.dot_general %{o}g, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,c]}, {ty [c,nC]}) -> {ty [BS,nC]}\n" ++
  s!"    %{o}db = stablehlo.broadcast_in_dim %bd, dims = [1] : ({ty [nC]}) -> {ty [BS,nC]}\n" ++
  s!"    %{o} = stablehlo.add %{o}dd, %{o}db : {ty [BS,nC]}\n"

-- ── whole net ──

private def efficientnetFwd : String := Id.run do
  let stemCode :=
    s!"    %xr = stablehlo.reshape %x : ({ty [BS,3*IMG*IMG]}) -> {ty [BS,3,IMG,IMG]}\n" ++
    conv3 "stc" "%xr" "%sW" "%sb" 32 3 IMG IMG (IMG/2) (IMG/2) 2 ++   -- B0: stem stride 2 (224→112)
    bnBatch "stn" "%stc" "%sg" "%sbt" 32 (IMG/2) (IMG/2) ((IMG/2)*(IMG/2)) ++
    swishAct "str" "%stn" 32 (IMG/2) (IMG/2)
  let mut blkCode := ""
  let mut cur := "%str"
  let mut curH := IMG/2
  for (p, ic, mid, oc, s, r, k) in blocks do
    let (c, out) := mbconvFwd p cur ic mid oc curH s r k
    blkCode := blkCode ++ c; cur := out; curH := curH / s
  let head :=
    conv1 "h" cur "%hW" "%hb" 1280 320 curH curH ++
    bnBatch "hn" "%h" "%hg" "%hbt" 1280 curH curH (curH*curH) ++
    swishAct "hr" "%hn" 1280 curH curH
  let tail := gapDense "out" "%hr" 1280 10 curH curH
  let body :=
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    stemCode ++ blkCode ++ head ++ tail
  let sig : List String :=
    ["%x: " ++ ty [BS,3*IMG*IMG]]
    ++ [s!"%sW: {ty [32,3,3,3]}", s!"%sb: {ty [32]}"] ++ bnSig "s" 32
    ++ (blocks.map (fun (p, ic, mid, oc, _, r, k) => mbconvSig p ic mid oc r k)).flatten
    ++ [s!"%hW: {ty [1280,320,1,1]}", s!"%hb: {ty [1280]}"] ++ bnSig "h" 1280
    ++ [s!"%Wd: {ty [1280,10]}", s!"%bd: {ty [10]}"]
  let argSig := String.intercalate ", " sig
  return "module @m {\n" ++ s!"  func.func @efficientnet_fwd({argSig}) -> {ty [BS,10]} " ++ "{\n" ++
    body ++ s!"    return %out : {ty [BS,10]}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := efficientnetFwd
  IO.println s!"rendered @efficientnet_fwd B0 (BS={BS}, {blocks.length} MBConv layers): {mlir.length} chars"
  IO.FS.createDirAll "verified_mlir"
  IO.FS.writeFile "verified_mlir/efficientnet_fwd.mlir" mlir
  IO.FS.createDirAll ".lake/build"
  let path := "verified_mlir/efficientnet_fwd.mlir"
  let cargs ← ireeCompileArgs path ".lake/build/efficientnet_fwd_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println "efficientnet_fwd B0 iree-compile OK → .lake/build/efficientnet_fwd_v.vmfb"

#eval main
