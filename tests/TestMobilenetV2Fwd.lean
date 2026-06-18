import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! # C4a/D3 — MobileNetV2 forward renderer (real downsampling [t,c,n,s]) + iree

Programmatic StableHLO for a real, DOWNSAMPLING MobileNetV2 forward (IMAGENETTE
3×224×224 — the paper-native ImageNet resolution), matching the reference architecture
(inverted-residual `[t,c,n,s]` stages that shrink spatial via STRIDE-2 DEPTHWISE — the
C3 `depthwiseStridedF` op), at the real MobileNetV2 /32 spatial flow:

  stem  : 3×3 stride-2 conv (3→32, 224→112) + BN + relu6
  b1-b17: full-paper inverted-residual stack [t,c,n,s] (mid=t·ic, strided depthwise
          downsamples), channels 16→24→32→64→96→160→320, spatial 112→56→28→14→7
  head  : 1×1 conv (320→1280) + BN + relu6  (MNv2 "features" layer @7×7)
  tail  : global-average-pool → dense(1280→10)

Spatial: 224→112(stem)→56→28→14→7 — strided-depthwise
downsamples (+ the strided stem) = the real MobileNetV2 /32 flow. Every
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

private def BS : Nat := 32
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

/-- True batch-norm forward (reduce μ/var over batch+spatial `[0,2,3]`, `nf = BS·H·W`) — matches
    the reference's batch-norm (the SGD `mobilenetv2-verified` eval; the adam trainer evals through
    `@mobilenetv2_fwd_eval` with running stats instead). rank-1 γ/β dims=[1]. -/
private def bnPC (o x g bt : String) (oc Hh Ww m : Nat) : String :=
  s!"    %{o}nf = stablehlo.constant dense<{BS*m}.0> : {ty [BS,oc,Hh,Ww]}\n" ++
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
  -- t=1 (mid=ic): NO expand 1×1 — depthwise reads the block input directly (torchvision-canonical).
  let dwIn := if mid == ic then x else s!"%{p}er"
  let expandCode := if mid == ic then "" else
    conv1 s!"{p}e" x s!"%{p}eW" s!"%{p}eb" mid ic Hin Hin ++
    bnPC s!"{p}en" s!"%{p}e" s!"%{p}eg" s!"%{p}ebt" mid Hin Hin (Hin*Hin) ++
    relu6 s!"{p}er" s!"%{p}en" mid Hin Hin
  let body :=
    expandCode ++
    (if s == 2 then dwconvStrided s!"{p}d" dwIn s!"%{p}dW" s!"%{p}db" mid Hout Hout
     else dwconv s!"{p}d" dwIn s!"%{p}dW" s!"%{p}db" mid Hin Hin) ++
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
  -- (p, ic, mid, oc, s) — full paper MobileNetV2 (17 inverted-residual blocks)
  [("b1",  32,  32,  16, 1),
   ("b2",  16,  96,  24, 2), ("b3",  24, 144,  24, 1),
   ("b4",  24, 144,  32, 2), ("b5",  32, 192,  32, 1), ("b6",  32, 192,  32, 1),
   ("b7",  32, 192,  64, 2), ("b8",  64, 384,  64, 1), ("b9",  64, 384,  64, 1), ("b10", 64, 384,  64, 1),
   ("b11", 64, 384,  96, 1), ("b12", 96, 576,  96, 1), ("b13", 96, 576,  96, 1),
   ("b14", 96, 576, 160, 2), ("b15",160, 960, 160, 1), ("b16",160, 960, 160, 1),
   ("b17",160, 960, 320, 1)]

private def bnSig (p : String) (oc : Nat) : List String :=
  [s!"%{p}g: {ty [oc]}", s!"%{p}bt: {ty [oc]}"]
private def irBlockSig (p : String) (ic mid oc : Nat) : List String :=
  (if mid == ic then [] else
   [s!"%{p}eW: {ty [mid,ic,1,1]}", s!"%{p}eb: {ty [mid]}", s!"%{p}eg: {ty [mid]}", s!"%{p}ebt: {ty [mid]}"]) ++
  [s!"%{p}dW: {ty [mid,1,3,3]}", s!"%{p}db: {ty [mid]}", s!"%{p}dg: {ty [mid]}", s!"%{p}dbt: {ty [mid]}",
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
  -- stem: 3×3 stride-2 conv (3→16, 224→112) + BN + relu6
  let stemCode :=
    s!"    %xr = stablehlo.reshape %x : ({ty [BS,150528]}) -> {ty [BS,3,224,224]}\n" ++
    conv3 "stc" "%xr" "%sW" "%sb" 32 3 224 224 112 112 2 ++
    bnPC "stn" "%stc" "%sg" "%sbt" 32 112 112 (112*112) ++
    relu6 "str" "%stn" 32 112 112
  let mut blkCode := ""
  let mut cur := "%str"
  let mut curH := 112
  for (p, ic, mid, oc, s) in blocks do
    let (c, out) := irBlockFwd p cur ic mid oc curH s
    blkCode := blkCode ++ c; cur := out; curH := curH / s
  -- head: 1×1 conv (320→1280) + BN + relu6 @ curH×curH (=7×7)
  let head :=
    conv1 "h" cur "%hW" "%hb" 1280 320 curH curH ++
    bnPC "hn" "%h" "%hg" "%hbt" 1280 curH curH (curH*curH) ++
    relu6 "hr" "%hn" 1280 curH curH
  let tail := gapDense "out" "%hr" 1280 10 curH curH
  let body :=
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    stemCode ++ blkCode ++ head ++ tail
  let sig : List String :=
    ["%x: " ++ ty [BS,150528]]
    ++ [s!"%sW: {ty [32,3,3,3]}", s!"%sb: {ty [32]}"] ++ bnSig "s" 32
    ++ (blocks.map (fun (p, ic, mid, oc, _) => irBlockSig p ic mid oc)).flatten
    ++ [s!"%hW: {ty [1280,320,1,1]}", s!"%hb: {ty [1280]}"] ++ bnSig "h" 1280
    ++ [s!"%Wd: {ty [1280,10]}", s!"%bd: {ty [10]}"]
  let argSig := String.intercalate ", " sig
  return "module @m {\n" ++ s!"  func.func @mobilenetv2_fwd({argSig}) -> {ty [BS,10]} " ++ "{\n" ++
    body ++ s!"    return %out : {ty [BS,10]}\n" ++ "  }\n}\n"

-- ════════════ inference-BN (running-stats) eval forward ════════════
-- Affine-only BN consuming per-layer running mean/var (func inputs `%{o}mu`/`%{o}var`), instead of
-- computing batch stats. `@mobilenetv2_fwd_eval` is what the adam driver evals with once running
-- stats are threaded — class-batch-independent eval, unlike the degenerate batch-BN eval.

/-- Affine BN with running stats: `y = γ·(x − μ)·rsqrt(var + ε) + β`, μ/var from inputs `%{o}mu`/`%{o}var`. -/
private def bnEval (o x g bt : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o}mub = stablehlo.broadcast_in_dim %{o}mu, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xc = stablehlo.subtract {x}, %{o}mub : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}vb = stablehlo.broadcast_in_dim %{o}var, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}ep = stablehlo.constant dense<{EPS}> : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}ve = stablehlo.add %{o}vb, %{o}ep : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}istd = stablehlo.rsqrt %{o}ve : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xh = stablehlo.multiply %{o}xc, %{o}istd : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}btb = stablehlo.broadcast_in_dim {bt}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}gx = stablehlo.multiply %{o}xh, %{o}gb : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}gx, %{o}btb : {ty [BS,oc,Hh,Ww]}\n"

/-- Inverted-residual block forward with affine running-stats BN (the eval block). -/
private def irBlockEval (p x : String) (ic mid oc Hin s : Nat) : String × String :=
  let Hout := Hin / s
  let dwIn := if mid == ic then x else s!"%{p}er"
  let expandCode := if mid == ic then "" else
    conv1 s!"{p}e" x s!"%{p}eW" s!"%{p}eb" mid ic Hin Hin ++
    bnEval s!"{p}en" s!"%{p}e" s!"%{p}eg" s!"%{p}ebt" mid Hin Hin ++
    relu6 s!"{p}er" s!"%{p}en" mid Hin Hin
  let body :=
    expandCode ++
    (if s == 2 then dwconvStrided s!"{p}d" dwIn s!"%{p}dW" s!"%{p}db" mid Hout Hout
     else dwconv s!"{p}d" dwIn s!"%{p}dW" s!"%{p}db" mid Hin Hin) ++
    bnEval s!"{p}dn" s!"%{p}d" s!"%{p}dg" s!"%{p}dbt" mid Hout Hout ++
    relu6 s!"{p}dr" s!"%{p}dn" mid Hout Hout ++
    conv1 s!"{p}p" s!"%{p}dr" s!"%{p}pW" s!"%{p}pb" oc mid Hout Hout ++
    bnEval s!"{p}pn" s!"%{p}p" s!"%{p}pg" s!"%{p}pbt" oc Hout Hout
  if s == 1 && ic == oc then
    (body ++ addOp s!"{p}o" s!"%{p}pn" x oc Hout Hout, s!"%{p}o")
  else
    (body, s!"%{p}pn")

/-- BN running-stats input pair `(%{p}mu, %{p}var)`, both `[oc]` — canonical (forward) order. -/
private def bnStatSig (p : String) (oc : Nat) : List String :=
  [s!"%{p}mu: {ty [oc]}", s!"%{p}var: {ty [oc]}"]
private def irBlockStatSig (p : String) (ic mid oc : Nat) : List String :=
  (if mid == ic then [] else bnStatSig s!"{p}en" mid) ++ bnStatSig s!"{p}dn" mid ++ bnStatSig s!"{p}pn" oc

/-- `@mobilenetv2_fwd_eval` — the eval forward with affine running-stats BN. Same params as
    `@mobilenetv2_fwd`, plus per-BN-layer `%{p}mu`/`%{p}var` `[oc]` inputs in BN forward order
    (the driver passes `θ ++ runningBnStats`). Returns logits `[BS,10]`. -/
private def mobilenetv2FwdEval : String := Id.run do
  let stemCode :=
    s!"    %xr = stablehlo.reshape %x : ({ty [BS,150528]}) -> {ty [BS,3,224,224]}\n" ++
    conv3 "stc" "%xr" "%sW" "%sb" 32 3 224 224 112 112 2 ++
    bnEval "stn" "%stc" "%sg" "%sbt" 32 112 112 ++
    relu6 "str" "%stn" 32 112 112
  let mut blkCode := ""
  let mut cur := "%str"
  let mut curH := 112
  for (p, ic, mid, oc, s) in blocks do
    let (c, out) := irBlockEval p cur ic mid oc curH s
    blkCode := blkCode ++ c; cur := out; curH := curH / s
  let head :=
    conv1 "h" cur "%hW" "%hb" 1280 320 curH curH ++
    bnEval "hn" "%h" "%hg" "%hbt" 1280 curH curH ++
    relu6 "hr" "%hn" 1280 curH curH
  let tail := gapDense "out" "%hr" 1280 10 curH curH
  let body :=
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    stemCode ++ blkCode ++ head ++ tail
  let paramSig : List String :=
    ["%x: " ++ ty [BS,150528]]
    ++ [s!"%sW: {ty [32,3,3,3]}", s!"%sb: {ty [32]}"] ++ bnSig "s" 32
    ++ (blocks.map (fun (p, ic, mid, oc, _) => irBlockSig p ic mid oc)).flatten
    ++ [s!"%hW: {ty [1280,320,1,1]}", s!"%hb: {ty [1280]}"] ++ bnSig "h" 1280
    ++ [s!"%Wd: {ty [1280,10]}", s!"%bd: {ty [10]}"]
  -- BN running-stats inputs, BN forward order (stem; per block en/dn/pn; head) — the driver's
  -- runningBnStats layout, matching mobilenetv2Verified.bnChannels and the adam step's stat outputs.
  let statSig : List String :=
    bnStatSig "stn" 32
    ++ (blocks.map (fun (p, ic, mid, oc, _) => irBlockStatSig p ic mid oc)).flatten
    ++ bnStatSig "hn" 1280
  let argSig := String.intercalate ", " (paramSig ++ statSig)
  return "module @m {\n" ++ s!"  func.func @mobilenetv2_fwd_eval({argSig}) -> {ty [BS,10]} " ++ "{\n" ++
    body ++ s!"    return %out : {ty [BS,10]}\n" ++ "  }\n}\n"

private def tryCompile (src dst label : String) : IO Unit := do
  try
    let cargs ← ireeCompileArgs src dst
    let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
    if r.exitCode != 0 then IO.eprintln s!"iree-compile ({label}) FAILED:\n{r.stderr.take 3000}"
    else IO.println s!"{label} iree-compile OK → {src}"
  catch e => IO.eprintln s!"iree-compile ({label}) skipped (compiler unavailable): {e}"

def main : IO Unit := do
  IO.FS.createDirAll "verified_mlir"
  IO.FS.createDirAll ".lake/build"
  let mlir := mobilenetv2Fwd
  IO.println s!"rendered @mobilenetv2_fwd (BS={BS}, {blocks.length} IR blocks): {mlir.length} chars"
  IO.FS.writeFile "verified_mlir/mobilenetv2_fwd.mlir" mlir
  let evalMlir := mobilenetv2FwdEval
  IO.println s!"rendered @mobilenetv2_fwd_eval (BS={BS}): {evalMlir.length} chars"
  IO.FS.writeFile "verified_mlir/mobilenetv2_fwd_eval.mlir" evalMlir
  tryCompile "verified_mlir/mobilenetv2_fwd.mlir" ".lake/build/mobilenetv2_fwd_v.vmfb" "fwd"
  tryCompile "verified_mlir/mobilenetv2_fwd_eval.mlir" ".lake/build/mobilenetv2_fwd_eval_v.vmfb" "fwd_eval"

#eval main
