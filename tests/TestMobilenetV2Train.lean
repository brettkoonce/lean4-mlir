import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! # C4b/D3 — MobileNetV2 train-step renderer (real downsampling [t,c,n,s]) + iree

Full single-batch SGD train step for the downsampling MobileNetV2 of the forward
renderer (IMAGENETTE 3×224×224, the real MobileNetV2 /32 spatial flow). Data-driven
over one block list `(p, ic, mid, oc, s)`, threading spatial dims through forward AND
the reverse pass. Exercises every op's fwd+back+SGD:
  stem 3×3 stride-2 conv + BN + relu6 → 6 inverted-residual blocks (4 of them
  stride-2 downsampling via depthwise: 224→112→56→28→14→7) → head 1×1 conv (64→128)
  + BN + relu6 → GAP → dense, softmax-CE mean-loss cotangent, full reverse pass,
  SGD updates.

New strided backward fragments vs the stride-1 net:
  * depthwise stride-2 input-grad  (`dwconvStridedBack`) — zero-upsample the
    cotangent (pad interior=1) then reversed-kernel stride-1 depthwise; the C3
    `depthwiseStridedBack` op, validated against depthwiseStride2Flat_has_vjp.
  * depthwise stride-2 weight-grad (`dwconvWGradStrided`) — upsample dy then the
    stride-1 depthwise weight-grad (batch_group_count=c) on the [c,B,2H,2W] grid.
The stem (stride-2 regular conv) reuses ch6's strided conv weight-grad.

Run (rocm): export IREE_BACKEND=rocm; lake env lean tests/TestMobilenetV2Train.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 32
private def EPS : String := "1.0e-5"
private def LR : String := "0.3"

-- ════════════ forward fragments ════════════

private def conv3 (o x w bnm : String) (oc ic Hin Win Hout Wout s : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [{s}, {s}], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,Hin,Win]}, {ty [oc,ic,3,3]}) -> {ty [BS,oc,Hout,Wout]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hout,Wout]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Hout,Wout]}\n"

private def conv1 (o x w bnm : String) (oc ic Hh Ww : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,Hh,Ww]}, {ty [oc,ic,1,1]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Hh,Ww]}\n"

private def dwconv (o x w bnm : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,Hh,Ww]}, {ty [c,1,3,3]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,c,Hh,Ww]}\n"

private def dwconvStrided (o x w bnm : String) (c Hout Wout : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,2*Hout,2*Wout]}, {ty [c,1,3,3]}) -> {ty [BS,c,Hout,Wout]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hout,Wout]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,c,Hout,Wout]}\n"

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

-- ════════════ backward fragments ════════════

private def relu6Back (o pre dy : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}z = stablehlo.constant dense<0.0> : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}six = stablehlo.constant dense<6.0> : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}g0 = stablehlo.compare GT, {pre}, %{o}z : ({ty [BS,c,Hh,Ww]}, {ty [BS,c,Hh,Ww]}) -> {tyI1 [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}l6 = stablehlo.compare LT, {pre}, %{o}six : ({ty [BS,c,Hh,Ww]}, {ty [BS,c,Hh,Ww]}) -> {tyI1 [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}m = stablehlo.and %{o}g0, %{o}l6 : {tyI1 [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.select %{o}m, {dy}, %{o}z : {tyI1 [BS,c,Hh,Ww]}, {ty [BS,c,Hh,Ww]}\n"

private def bnBackPC (o bn dy : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o}dxh = stablehlo.multiply %{bn}gb, {dy} : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}sdxr = stablehlo.reduce(%{o}dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [BS,oc]}\n" ++
  s!"    %{o}sdx = stablehlo.broadcast_in_dim %{o}sdxr, dims = [0, 1] : ({ty [BS,oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xd = stablehlo.multiply %{bn}xh, %{o}dxh : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}sxdr = stablehlo.reduce(%{o}xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [BS,oc]}\n" ++
  s!"    %{o}sxd = stablehlo.broadcast_in_dim %{o}sxdr, dims = [0, 1] : ({ty [BS,oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}t1 = stablehlo.multiply %{o}dxh, %{bn}nf : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}i1 = stablehlo.subtract %{o}t1, %{o}sdx : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xs = stablehlo.multiply %{bn}xh, %{o}sxd : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}i2 = stablehlo.subtract %{o}i1, %{o}xs : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}sN = stablehlo.divide %{bn}istd, %{bn}nf : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.multiply %{o}sN, %{o}i2 : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}dgp = stablehlo.multiply {dy}, %{bn}xh : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}dg = stablehlo.reduce(%{o}dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n" ++
  s!"    %{o}db = stablehlo.reduce({dy} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"

/-- 1×1 conv input-grad (pad 0). Result `%{o}` [B,ic,H,W]. -/
private def conv1Back (o dy w : String) (ic oc Hh Ww : Nat) : String :=
  s!"    %{o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,ic,1,1]}) -> {ty [ic,oc,1,1]}\n" ++
  s!"    %{o} = stablehlo.convolution({dy}, %{o}t)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,oc,Hh,Ww]}, {ty [ic,oc,1,1]}) -> {ty [BS,ic,Hh,Ww]}\n"

/-- 1×1 conv weight-grad (pad 0). Result `%{o}` [oc,ic,1,1]. -/
private def conv1WGrad (o inp dy : String) (ic oc Hh Ww : Nat) : String :=
  s!"    %{o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [BS,ic,Hh,Ww]}) -> {ty [ic,BS,Hh,Ww]}\n" ++
  s!"    %{o}dt = stablehlo.transpose {dy}, dims = [1, 0, 2, 3] : ({ty [BS,oc,Hh,Ww]}) -> {ty [oc,BS,Hh,Ww]}\n" ++
  s!"    %{o}raw = stablehlo.convolution(%{o}xt, %{o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ic,BS,Hh,Ww]}, {ty [oc,BS,Hh,Ww]}) -> {ty [ic,oc,1,1]}\n" ++
  s!"    %{o} = stablehlo.transpose %{o}raw, dims = [1, 0, 2, 3] : ({ty [ic,oc,1,1]}) -> {ty [oc,ic,1,1]}\n"

/-- Depthwise stride-1 input-grad (reverse [2,3] + depthwise conv). `%{o}` [B,c,H,W]. -/
private def dwconvBack (o dy w : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}rev = stablehlo.reverse {w}, dims = [2, 3] : {ty [c,1,3,3]}\n" ++
  s!"    %{o} = stablehlo.convolution({dy}, %{o}rev)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,Hh,Ww]}, {ty [c,1,3,3]}) -> {ty [BS,c,Hh,Ww]}\n"

/-- Depthwise stride-1 weight-grad (batch_group_count=c). `%{o}` [c,1,3,3]. -/
private def dwconvWGrad (o inp dy : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [BS,c,Hh,Ww]}) -> {ty [c,BS,Hh,Ww]}\n" ++
  s!"    %{o}dt = stablehlo.transpose {dy}, dims = [1, 0, 2, 3] : ({ty [BS,c,Hh,Ww]}) -> {ty [c,BS,Hh,Ww]}\n" ++
  s!"    %{o}raw = stablehlo.convolution(%{o}xt, %{o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [c,BS,Hh,Ww]}, {ty [c,BS,Hh,Ww]}) -> {ty [1,c,3,3]}\n" ++
  s!"    %{o} = stablehlo.reshape %{o}raw : ({ty [1,c,3,3]}) -> {ty [c,1,3,3]}\n"

private def convBiasGrad (o dy : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.reduce({dy} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"

/-- Zero-upsample [B,c,Hh,Ww] → [B,c,2Hh,2Ww] (pad interior/high=1). -/
private def upsample (o dy : String) (c Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.pad {dy}, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [BS,c,Hh,Ww]}, tensor<f32>) -> {ty [BS,c,2*Hh,2*Ww]}\n"

/-- Depthwise STRIDE-2 input-grad (the C3 `depthwiseStridedBack`): zero-upsample dy
    [B,c,Hout,Wout] → [B,c,2Hout,2Wout], reverse [2,3], stride-1 depthwise. -/
private def dwconvStridedBack (o dy w : String) (c Hout Wout : Nat) : String :=
  upsample s!"{o}u" dy c Hout Wout ++
  s!"    %{o}rev = stablehlo.reverse {w}, dims = [2, 3] : {ty [c,1,3,3]}\n" ++
  s!"    %{o} = stablehlo.convolution(%{o}u, %{o}rev)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,2*Hout,2*Wout]}, {ty [c,1,3,3]}) -> {ty [BS,c,2*Hout,2*Wout]}\n"

/-- Depthwise STRIDE-2 weight-grad: upsample dy [B,c,Hout,Wout] → [B,c,2Hout,2Wout],
    then the stride-1 depthwise weight-grad between the dw input `inp` (at 2Hout) and
    the upsampled cotangent. `%{o}` [c,1,3,3]. -/
private def dwconvWGradStrided (o inp dy : String) (c Hout Wout : Nat) : String :=
  upsample s!"{o}u" dy c Hout Wout ++ dwconvWGrad o inp s!"%{o}u" c (2*Hout) (2*Wout)

/-- 3×3 conv weight-grad (stride 1, pad 1). `%{o}` [oc,ic,3,3]. -/
private def conv3WGrad (o inp dy : String) (ic oc Hh Ww : Nat) : String :=
  s!"    %{o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [BS,ic,Hh,Ww]}) -> {ty [ic,BS,Hh,Ww]}\n" ++
  s!"    %{o}dt = stablehlo.transpose {dy}, dims = [1, 0, 2, 3] : ({ty [BS,oc,Hh,Ww]}) -> {ty [oc,BS,Hh,Ww]}\n" ++
  s!"    %{o}raw = stablehlo.convolution(%{o}xt, %{o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ic,BS,Hh,Ww]}, {ty [oc,BS,Hh,Ww]}) -> {ty [ic,oc,3,3]}\n" ++
  s!"    %{o} = stablehlo.transpose %{o}raw, dims = [1, 0, 2, 3] : ({ty [ic,oc,3,3]}) -> {ty [oc,ic,3,3]}\n"

/-- Strided 3×3 conv weight-grad (stem): upsample dy then stride-1 weight-grad. -/
private def conv3WGradStrided (o inp dy : String) (ic oc Hh Ww : Nat) : String :=
  upsample s!"{o}u" dy oc Hh Ww ++ conv3WGrad o inp s!"%{o}u" ic oc (2*Hh) (2*Ww)

-- ════════════ inverted-residual block forward / backward (stride `s`) ════════════

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

/-- IR block backward (stride `s`). `dy` = cotangent of block output (at Hout);
    `xin` = block input (at Hin). Result dx `%{p}dx` (at Hin). -/
private def irBlockBack (p dy xin : String) (ic mid oc Hin s : Nat) : String × String :=
  let Hout := Hin / s
  let projDwExp :=
    -- project: BN back → 1×1 conv back (at Hout)
    bnBackPC s!"{p}dpn" s!"{p}pn" dy oc Hout Hout ++
    conv1Back s!"{p}dp" s!"%{p}dpn" s!"%{p}pW" mid oc Hout Hout ++
    conv1WGrad s!"{p}dpW" s!"%{p}dr" s!"%{p}dpn" mid oc Hout Hout ++
    convBiasGrad s!"{p}dpb" s!"%{p}dpn" oc Hout Hout ++
    -- depthwise: relu6 back (mask on dn @Hout) → BN back → depthwise back
    relu6Back s!"{p}ddr" s!"%{p}dn" s!"%{p}dp" mid Hout Hout ++
    bnBackPC s!"{p}ddn" s!"{p}dn" s!"%{p}ddr" mid Hout Hout ++
    (if s == 2 then
       dwconvStridedBack s!"{p}dd" s!"%{p}ddn" s!"%{p}dW" mid Hout Hout ++
       dwconvWGradStrided s!"{p}ddW" s!"%{p}er" s!"%{p}ddn" mid Hout Hout
     else
       dwconvBack s!"{p}dd" s!"%{p}ddn" s!"%{p}dW" mid Hin Hin ++
       dwconvWGrad s!"{p}ddW" s!"%{p}er" s!"%{p}ddn" mid Hin Hin) ++
    convBiasGrad s!"{p}ddb" s!"%{p}ddn" mid Hout Hout ++
    -- expand: relu6 back (mask on en @Hin) → BN back → 1×1 conv back (at Hin)
    relu6Back s!"{p}der" s!"%{p}en" s!"%{p}dd" mid Hin Hin ++
    bnBackPC s!"{p}den" s!"{p}en" s!"%{p}der" mid Hin Hin ++
    conv1Back s!"{p}de" s!"%{p}den" s!"%{p}eW" ic mid Hin Hin ++
    conv1WGrad s!"{p}deW" xin s!"%{p}den" ic mid Hin Hin ++
    convBiasGrad s!"{p}deb" s!"%{p}den" mid Hin Hin
  if s == 1 && ic == oc then
    (projDwExp ++ addOp s!"{p}dx" s!"%{p}de" dy ic Hin Hin, s!"%{p}dx")
  else
    (projDwExp, s!"%{p}de")

-- ════════════ block config + data-driven params ════════════

private def sgd (θ dθ ty' : String) : String :=
  s!"    %{θ}l = stablehlo.constant dense<{LR}> : {ty'}\n" ++
  s!"    %{θ}s = stablehlo.multiply {dθ}, %{θ}l : {ty'}\n" ++
  s!"    %{θ}n = stablehlo.subtract %{θ}, %{θ}s : {ty'}\n"

/-- (p, ic, mid, oc, s); spatial threaded from the stem (16×16). -/
private def blocks : List (String × Nat × Nat × Nat × Nat) :=
  [("b1", 16, 64,  24, 2),   -- 112→56
   ("b2", 24, 96,  24, 1),   -- skip @56
   ("b3", 24, 96,  32, 2),   -- 56→28
   ("b4", 32, 128, 32, 1),   -- skip @28
   ("b5", 32, 128, 64, 2),   -- 28→14 (no skip)
   ("b6", 64, 256, 64, 2)]   -- 14→7 (no skip)

/-- IR block param triples (name, gradSSA, type) in func-arg order. -/
private def irBlkParams (p : String) (ic mid oc : Nat) : List (String × String × String) :=
  [(s!"{p}eW", s!"%{p}deW", ty [mid,ic,1,1]), (s!"{p}eb", s!"%{p}deb", ty [mid]),
   (s!"{p}eg", s!"%{p}dendg", ty [mid]), (s!"{p}ebt", s!"%{p}dendb", ty [mid]),
   (s!"{p}dW", s!"%{p}ddW", ty [mid,1,3,3]), (s!"{p}db", s!"%{p}ddb", ty [mid]),
   (s!"{p}dg", s!"%{p}ddndg", ty [mid]), (s!"{p}dbt", s!"%{p}ddndb", ty [mid]),
   (s!"{p}pW", s!"%{p}dpW", ty [oc,mid,1,1]), (s!"{p}pb", s!"%{p}dpb", ty [oc]),
   (s!"{p}pg", s!"%{p}dpndg", ty [oc]), (s!"{p}pbt", s!"%{p}dpndb", ty [oc])]

/-- Whole net's param triples: stem, blocks, head, dense. -/
private def allParams : List (String × String × String) :=
  [("sW", "%dsW", ty [16,3,3,3]), ("sb", "%dsb", ty [16]),
   ("sg", "%dstndg", ty [16]), ("sbt", "%dstndb", ty [16])]
  ++ (blocks.map (fun (p, ic, mid, oc, _) => irBlkParams p ic mid oc)).flatten
  ++ [("hW", "%dhW", ty [128,64,1,1]), ("hb", "%dhb", ty [128]),
      ("hg", "%dhndg", ty [128]), ("hbt", "%dhndb", ty [128])]
  ++ [("Wd", "%dWd", ty [128,10]), ("bd", "%dbd", ty [10])]

private def trainStep : String := Id.run do
  -- ── forward: stem → blocks → head → GAP(7×7) → dense(128→10) ──
  let mut fwd := "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n"
    ++ s!"    %xr = stablehlo.reshape %x : ({ty [BS,150528]}) -> {ty [BS,3,224,224]}\n"
    ++ conv3 "stc" "%xr" "%sW" "%sb" 16 3 224 224 112 112 2
    ++ bnPC "stn" "%stc" "%sg" "%sbt" 16 112 112 (112*112)
    ++ relu6 "str" "%stn" 16 112 112
  let mut cur := "%str"
  let mut curH := 112
  let mut io : List ((String × Nat × Nat × Nat × Nat) × String × Nat) := []  -- (blk, xin, Hin)
  for blk in blocks do
    let (p, ic, mid, oc, s) := blk
    io := io ++ [(blk, cur, curH)]
    fwd := fwd ++ (irBlockFwd p cur ic mid oc curH s).1
    cur := if s == 1 && ic == oc then s!"%{p}o" else s!"%{p}pn"
    curH := curH / s
  -- head: 1×1 conv (64→128) + BN + relu6 @ curH
  let hd := curH
  fwd := fwd
    ++ conv1 "h" cur "%hW" "%hb" 128 64 hd hd
    ++ bnPC "hn" "%h" "%hg" "%hbt" 128 hd hd (hd*hd)
    ++ relu6 "hr" "%hn" 128 hd hd
    ++ s!"    %gaps = stablehlo.reduce(%hr init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,128,hd,hd]}, tensor<f32>) -> {ty [BS,128]}\n"
    ++ s!"    %gapnf = stablehlo.constant dense<{hd*hd}.0> : {ty [BS,128]}\n"
    ++ s!"    %gap = stablehlo.divide %gaps, %gapnf : {ty [BS,128]}\n"
    ++ s!"    %ld = stablehlo.dot_general %gap, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,128]}, {ty [128,10]}) -> {ty [BS,10]}\n"
    ++ s!"    %ldb = stablehlo.broadcast_in_dim %bd, dims = [1] : ({ty [10]}) -> {ty [BS,10]}\n"
    ++ s!"    %logits = stablehlo.add %ld, %ldb : {ty [BS,10]}\n"
  -- ── loss cotangent dy = (softmax(logits) − onehot) / B ──
  let cot :=
    s!"    %le = stablehlo.exponential %logits : {ty [BS,10]}\n"
    ++ s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [BS,10]}, tensor<f32>) -> {ty [BS]}\n"
    ++ s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [BS]}) -> {ty [BS,10]}\n"
    ++ s!"    %lsm = stablehlo.divide %le, %lsb : {ty [BS,10]}\n"
    ++ s!"    %dyr = stablehlo.subtract %lsm, %onehot : {ty [BS,10]}\n"
    ++ s!"    %bnc = stablehlo.constant dense<{BS}.0> : {ty [BS,10]}\n"
    ++ s!"    %dy = stablehlo.divide %dyr, %bnc : {ty [BS,10]}\n"
  -- ── backward: dense + GAP → head → blocks reversed → stem ──
  let mut bwd :=
    s!"    %dgap = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [BS,10]}, {ty [128,10]}) -> {ty [BS,128]}\n"
    ++ s!"    %dWd = stablehlo.dot_general %gap, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,128]}, {ty [BS,10]}) -> {ty [128,10]}\n"
    ++ s!"    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,10]}, tensor<f32>) -> {ty [10]}\n"
    ++ s!"    %dgnf = stablehlo.constant dense<{hd*hd}.0> : {ty [BS,128]}\n"
    ++ s!"    %dgs = stablehlo.divide %dgap, %dgnf : {ty [BS,128]}\n"
    ++ s!"    %dgapin = stablehlo.broadcast_in_dim %dgs, dims = [0, 1] : ({ty [BS,128]}) -> {ty [BS,128,hd,hd]}\n"
    -- head backward: relu6 (mask hn) → BN → 1×1 conv back (→ dy for last block)
    ++ relu6Back "dhr" "%hn" "%dgapin" 128 hd hd
    ++ bnBackPC "dhn" "hn" "%dhr" 128 hd hd
    ++ conv1Back "dh" "%dhn" "%hW" 64 128 hd hd
    ++ conv1WGrad "dhW" cur "%dhn" 64 128 hd hd
    ++ convBiasGrad "dhb" "%dhn" 128 hd hd
  let mut d := "%dh"
  for (blk, xin, hin) in io.reverse do
    let (p, ic, mid, oc, s) := blk
    let (code, out) := irBlockBack p d xin ic mid oc hin s
    bwd := bwd ++ code
    d := out
  -- stem backward: relu6 (mask stn) → BN → strided 3×3 weight-grad
  bwd := bwd
    ++ relu6Back "dstr" "%stn" d 16 112 112
    ++ bnBackPC "dstn" "stn" "%dstr" 16 112 112
    ++ convBiasGrad "dsb" "%dstn" 16 112 112
    ++ conv3WGradStrided "dsW" "%xr" "%dstn" 3 16 112 112
  -- ── SGD + signature/return, all from the single param list ──
  let upd := String.join (allParams.map (fun (nm, gr, t) => sgd nm gr t))
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [BS,150528]) :: allParams.map (fun (nm, _, t) => s!"%{nm}: {t}") ++ ["%onehot: " ++ ty [BS,10]])
  let retTyL := String.intercalate ", " (allParams.map (fun (_, _, t) => t))
  let retVals := String.intercalate ", " (allParams.map (fun (nm, _, _) => s!"%{nm}n"))
  return "module @m {\n" ++ s!"  func.func @mobilenetv2_train_step({argSig}) -> ({retTyL}) " ++ "{\n" ++
    fwd ++ cot ++ bwd ++ upd ++ s!"    return {retVals} : {retTyL}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := trainStep
  IO.println s!"rendered MobileNetV2 train step (BS={BS}, {blocks.length} IR blocks): {mlir.length} chars, {allParams.length} params"
  IO.FS.createDirAll "verified_mlir"
  IO.FS.writeFile "verified_mlir/mobilenetv2_train_step.mlir" mlir
  IO.FS.createDirAll ".lake/build"
  let cargs ← ireeCompileArgs "verified_mlir/mobilenetv2_train_step.mlir" ".lake/build/mobilenetv2_train_step_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println "mobilenetv2 FULL train step iree-compile OK → verified_mlir/mobilenetv2_train_step.mlir"

#eval main
