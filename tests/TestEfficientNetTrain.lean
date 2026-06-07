import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! # E4b — EfficientNet train-step renderer (MBConv = inverted-residual + SE + swish) + iree

Full single-batch SGD train step for the downsampling EfficientNet of the forward
renderer. Data-driven over one block list `(p, ic, mid, oc, s, r)`, threading spatial
dims through forward AND the reverse pass. The MBConv block = ch7's inverted residual
with relu6 → **swish** and a **squeeze-excite** gate before the project:

  stem 3×3 stride-2 conv + BN + swish → 6 MBConv blocks (2 stride-2 downsampling via
  depthwise; each with an SE gate) → head 1×1 conv (64→128) + BN + swish → GAP →
  dense, softmax-CE mean-loss cotangent, full reverse pass, SGD updates.

vs ch7's MobileNetV2 trainer: swish/swishBack in place of relu6/relu6Back, and the SE
module (seFwd forward + seBack 2-path backward) inserted between depthwise-swish and the
project. SE backward: the project's conv-input grad is the SE-output cotangent `dse`;
seBack returns the SE-input cotangent (→ depthwise swish-back) plus the two SE dense
weight+bias grads. Every fragment is the StableHLO of a proven-faithful per-op emitter
(swishF/swishBack, sigmoidF, depthwise stride-1/2, per-channel BN, convs, residual, GAP,
dense); the SE mirrors `seGate`/`seBlock`/`broadcastFlat`. The SE was gradcheck-validated
standalone (tests/TestSE.lean).

Run (rocm): export IREE_BACKEND=rocm; lake env lean tests/TestEfficientNetTrain.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 128
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

/-- Swish forward `y = x · σ(x)` (= emitTok swishF). -/
private def swishAct (o x : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}s = stablehlo.logistic {x} : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.multiply {x}, %{o}s : {ty [BS,c,Hh,Ww]}\n"

/-- ReLU6 forward `clamp(x,0,6)` (= emitTok relu6F) — HEAD only (breaks the GAP-of-
    instance-norm degeneracy that smooth swish leaves at the final pool; blocks use swish). -/
private def relu6 (o x : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}z = stablehlo.constant dense<0.0> : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}six = stablehlo.constant dense<6.0> : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}mx = stablehlo.maximum {x}, %{o}z : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.minimum %{o}mx, %{o}six : {ty [BS,c,Hh,Ww]}\n"

/-- ReLU6 backward mask `select(0<pre<6, dy, 0)` (= emitTok selectMid) — HEAD only. -/
private def relu6Back (o pre dy : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}z = stablehlo.constant dense<0.0> : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}six = stablehlo.constant dense<6.0> : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}g0 = stablehlo.compare GT, {pre}, %{o}z : ({ty [BS,c,Hh,Ww]}, {ty [BS,c,Hh,Ww]}) -> {tyI1 [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}l6 = stablehlo.compare LT, {pre}, %{o}six : ({ty [BS,c,Hh,Ww]}, {ty [BS,c,Hh,Ww]}) -> {tyI1 [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}m = stablehlo.and %{o}g0, %{o}l6 : {tyI1 [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.select %{o}m, {dy}, %{o}z : {tyI1 [BS,c,Hh,Ww]}, {ty [BS,c,Hh,Ww]}\n"

private def addOp (o a b : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.add {a}, {b} : {ty [BS,oc,Hh,Ww]}\n"

/-- SE forward (squeeze→dense₁→swish→dense₂→sigmoid→bcast×x). Produces `%{p}se`,
    saves `%{p}sq`,`%{p}ex`,`%{p}a1`,`%{p}gate`. -/
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

-- ════════════ backward fragments ════════════

/-- Swish backward `dx = dy ⊙ σ(pre)·(1 + pre·(1−σ(pre)))` (= emitTok swishBack). -/
private def swishBack (o pre dy : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}s = stablehlo.logistic {pre} : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}one = stablehlo.constant dense<1.0> : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}om = stablehlo.subtract %{o}one, %{o}s : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}xom = stablehlo.multiply {pre}, %{o}om : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}in = stablehlo.add %{o}one, %{o}xom : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}sp = stablehlo.multiply %{o}s, %{o}in : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.multiply {dy}, %{o}sp : {ty [BS,c,Hh,Ww]}\n"

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

private def conv1Back (o dy w : String) (ic oc Hh Ww : Nat) : String :=
  s!"    %{o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,ic,1,1]}) -> {ty [ic,oc,1,1]}\n" ++
  s!"    %{o} = stablehlo.convolution({dy}, %{o}t)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,oc,Hh,Ww]}, {ty [ic,oc,1,1]}) -> {ty [BS,ic,Hh,Ww]}\n"

private def conv1WGrad (o inp dy : String) (ic oc Hh Ww : Nat) : String :=
  s!"    %{o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [BS,ic,Hh,Ww]}) -> {ty [ic,BS,Hh,Ww]}\n" ++
  s!"    %{o}dt = stablehlo.transpose {dy}, dims = [1, 0, 2, 3] : ({ty [BS,oc,Hh,Ww]}) -> {ty [oc,BS,Hh,Ww]}\n" ++
  s!"    %{o}raw = stablehlo.convolution(%{o}xt, %{o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ic,BS,Hh,Ww]}, {ty [oc,BS,Hh,Ww]}) -> {ty [ic,oc,1,1]}\n" ++
  s!"    %{o} = stablehlo.transpose %{o}raw, dims = [1, 0, 2, 3] : ({ty [ic,oc,1,1]}) -> {ty [oc,ic,1,1]}\n"

private def dwconvBack (o dy w : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}rev = stablehlo.reverse {w}, dims = [2, 3] : {ty [c,1,3,3]}\n" ++
  s!"    %{o} = stablehlo.convolution({dy}, %{o}rev)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,Hh,Ww]}, {ty [c,1,3,3]}) -> {ty [BS,c,Hh,Ww]}\n"

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

private def upsample (o dy : String) (c Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.pad {dy}, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [BS,c,Hh,Ww]}, tensor<f32>) -> {ty [BS,c,2*Hh,2*Ww]}\n"

private def dwconvStridedBack (o dy w : String) (c Hout Wout : Nat) : String :=
  upsample s!"{o}u" dy c Hout Wout ++
  s!"    %{o}rev = stablehlo.reverse {w}, dims = [2, 3] : {ty [c,1,3,3]}\n" ++
  s!"    %{o} = stablehlo.convolution(%{o}u, %{o}rev)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,2*Hout,2*Wout]}, {ty [c,1,3,3]}) -> {ty [BS,c,2*Hout,2*Wout]}\n"

private def dwconvWGradStrided (o inp dy : String) (c Hout Wout : Nat) : String :=
  upsample s!"{o}u" dy c Hout Wout ++ dwconvWGrad o inp s!"%{o}u" c (2*Hout) (2*Wout)

private def conv3WGrad (o inp dy : String) (ic oc Hh Ww : Nat) : String :=
  s!"    %{o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [BS,ic,Hh,Ww]}) -> {ty [ic,BS,Hh,Ww]}\n" ++
  s!"    %{o}dt = stablehlo.transpose {dy}, dims = [1, 0, 2, 3] : ({ty [BS,oc,Hh,Ww]}) -> {ty [oc,BS,Hh,Ww]}\n" ++
  s!"    %{o}raw = stablehlo.convolution(%{o}xt, %{o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ic,BS,Hh,Ww]}, {ty [oc,BS,Hh,Ww]}) -> {ty [ic,oc,3,3]}\n" ++
  s!"    %{o} = stablehlo.transpose %{o}raw, dims = [1, 0, 2, 3] : ({ty [ic,oc,3,3]}) -> {ty [oc,ic,3,3]}\n"

private def conv3WGradStrided (o inp dy : String) (ic oc Hh Ww : Nat) : String :=
  upsample s!"{o}u" dy oc Hh Ww ++ conv3WGrad o inp s!"%{o}u" ic oc (2*Hh) (2*Ww)

/-- SE backward — input cotangent `%{p}dds` + dense grads `%{p}dWs1`,`%{p}dbs1`,
    `%{p}dWs2`,`%{p}dbs2`. Reuses `%{p}sq`,`%{p}ex`,`%{p}a1`,`%{p}gate`. -/
private def seBack (p x dse Ws1 Ws2 : String) (c h w r : Nat) : String :=
  s!"    %{p}gb2 = stablehlo.broadcast_in_dim %{p}gate, dims = [0, 1] : ({ty [BS,c]}) -> {ty [BS,c,h,w]}\n" ++
  s!"    %{p}dleft = stablehlo.multiply %{p}gb2, {dse} : {ty [BS,c,h,w]}\n" ++
  s!"    %{p}xdse = stablehlo.multiply {x}, {dse} : {ty [BS,c,h,w]}\n" ++
  s!"    %{p}dgate = stablehlo.reduce(%{p}xdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,c,h,w]}, tensor<f32>) -> {ty [BS,c]}\n" ++
  s!"    %{p}one = stablehlo.constant dense<1.0> : {ty [BS,c]}\n" ++
  s!"    %{p}omg = stablehlo.subtract %{p}one, %{p}gate : {ty [BS,c]}\n" ++
  s!"    %{p}sg = stablehlo.multiply %{p}gate, %{p}omg : {ty [BS,c]}\n" ++
  s!"    %{p}dh2 = stablehlo.multiply %{p}dgate, %{p}sg : {ty [BS,c]}\n" ++
  s!"    %{p}da1 = stablehlo.dot_general %{p}dh2, {Ws2}, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [BS,c]}, {ty [r,c]}) -> {ty [BS,r]}\n" ++
  s!"    %{p}dWs2 = stablehlo.dot_general %{p}a1, %{p}dh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,r]}, {ty [BS,c]}) -> {ty [r,c]}\n" ++
  s!"    %{p}dbs2 = stablehlo.reduce(%{p}dh2 init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,c]}, tensor<f32>) -> {ty [c]}\n" ++
  s!"    %{p}dexs = stablehlo.logistic %{p}ex : {ty [BS,r]}\n" ++
  s!"    %{p}dexone = stablehlo.constant dense<1.0> : {ty [BS,r]}\n" ++
  s!"    %{p}dexom = stablehlo.subtract %{p}dexone, %{p}dexs : {ty [BS,r]}\n" ++
  s!"    %{p}dexxom = stablehlo.multiply %{p}ex, %{p}dexom : {ty [BS,r]}\n" ++
  s!"    %{p}dexin = stablehlo.add %{p}dexone, %{p}dexxom : {ty [BS,r]}\n" ++
  s!"    %{p}dexsp = stablehlo.multiply %{p}dexs, %{p}dexin : {ty [BS,r]}\n" ++
  s!"    %{p}dex = stablehlo.multiply %{p}da1, %{p}dexsp : {ty [BS,r]}\n" ++
  s!"    %{p}dsq = stablehlo.dot_general %{p}dex, {Ws1}, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [BS,r]}, {ty [c,r]}) -> {ty [BS,c]}\n" ++
  s!"    %{p}dWs1 = stablehlo.dot_general %{p}sq, %{p}dex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,c]}, {ty [BS,r]}) -> {ty [c,r]}\n" ++
  s!"    %{p}dbs1 = stablehlo.reduce(%{p}dex init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,r]}, tensor<f32>) -> {ty [r]}\n" ++
  s!"    %{p}dsqnf = stablehlo.constant dense<{h*w}.0> : {ty [BS,c]}\n" ++
  s!"    %{p}dsqd = stablehlo.divide %{p}dsq, %{p}dsqnf : {ty [BS,c]}\n" ++
  s!"    %{p}dgsp = stablehlo.broadcast_in_dim %{p}dsqd, dims = [0, 1] : ({ty [BS,c]}) -> {ty [BS,c,h,w]}\n" ++
  s!"    %{p}dds = stablehlo.add %{p}dleft, %{p}dgsp : {ty [BS,c,h,w]}\n"

-- ════════════ MBConv block forward / backward (stride `s`, SE bottleneck `r`) ════════════

private def mbconvFwd (p x : String) (ic mid oc Hin s r : Nat) : String × String :=
  let Hout := Hin / s
  let body :=
    conv1 s!"{p}e" x s!"%{p}eW" s!"%{p}eb" mid ic Hin Hin ++
    bnPC s!"{p}en" s!"%{p}e" s!"%{p}eg" s!"%{p}ebt" mid Hin Hin (Hin*Hin) ++
    swishAct s!"{p}es" s!"%{p}en" mid Hin Hin ++
    (if s == 2 then dwconvStrided s!"{p}d" s!"%{p}es" s!"%{p}dW" s!"%{p}db" mid Hout Hout
     else dwconv s!"{p}d" s!"%{p}es" s!"%{p}dW" s!"%{p}db" mid Hin Hin) ++
    bnPC s!"{p}dn" s!"%{p}d" s!"%{p}dg" s!"%{p}dbt" mid Hout Hout (Hout*Hout) ++
    swishAct s!"{p}ds" s!"%{p}dn" mid Hout Hout ++
    seFwd s!"{p}z" s!"%{p}ds" s!"%{p}zW1" s!"%{p}zb1" s!"%{p}zW2" s!"%{p}zb2" mid Hout Hout r ++
    conv1 s!"{p}p" s!"%{p}zse" s!"%{p}pW" s!"%{p}pb" oc mid Hout Hout ++
    bnPC s!"{p}pn" s!"%{p}p" s!"%{p}pg" s!"%{p}pbt" oc Hout Hout (Hout*Hout)
  if s == 1 && ic == oc then
    (body ++ addOp s!"{p}o" s!"%{p}pn" x oc Hout Hout, s!"%{p}o")
  else
    (body, s!"%{p}pn")

/-- MBConv block backward (stride `s`). `dy` = cotangent of block output (at Hout);
    `xin` = block input (at Hin). Result dx `%{p}dx` (at Hin). -/
private def mbconvBack (p dy xin : String) (ic mid oc Hin s r : Nat) : String × String :=
  let Hout := Hin / s
  let body :=
    -- project back: BN back → 1×1 conv back (= SE-output cotangent `%{p}dp`)
    bnBackPC s!"{p}dpn" s!"{p}pn" dy oc Hout Hout ++
    conv1Back s!"{p}dp" s!"%{p}dpn" s!"%{p}pW" mid oc Hout Hout ++
    conv1WGrad s!"{p}dpW" s!"%{p}zse" s!"%{p}dpn" mid oc Hout Hout ++
    convBiasGrad s!"{p}dpb" s!"%{p}dpn" oc Hout Hout ++
    -- SE back (2-path fan-in): x = depthwise-swish `%{p}ds`, dse = `%{p}dp`
    --   → SE-input cotangent `%{p}zdds` + dense grads %{p}zdWs1/dbs1/dWs2/dbs2
    seBack s!"{p}z" s!"%{p}ds" s!"%{p}dp" s!"%{p}zW1" s!"%{p}zW2" mid Hout Hout r ++
    -- depthwise: swish back (pre = dn @Hout, dy = SE-input cotangent) → BN back → depthwise back
    swishBack s!"{p}ddr" s!"%{p}dn" s!"%{p}zdds" mid Hout Hout ++
    bnBackPC s!"{p}ddn" s!"{p}dn" s!"%{p}ddr" mid Hout Hout ++
    (if s == 2 then
       dwconvStridedBack s!"{p}dd" s!"%{p}ddn" s!"%{p}dW" mid Hout Hout ++
       dwconvWGradStrided s!"{p}ddW" s!"%{p}es" s!"%{p}ddn" mid Hout Hout
     else
       dwconvBack s!"{p}dd" s!"%{p}ddn" s!"%{p}dW" mid Hin Hin ++
       dwconvWGrad s!"{p}ddW" s!"%{p}es" s!"%{p}ddn" mid Hin Hin) ++
    convBiasGrad s!"{p}ddb" s!"%{p}ddn" mid Hout Hout ++
    -- expand: swish back (pre = en @Hin, dy = depthwise-input cotangent) → BN back → 1×1 conv back
    swishBack s!"{p}der" s!"%{p}en" s!"%{p}dd" mid Hin Hin ++
    bnBackPC s!"{p}den" s!"{p}en" s!"%{p}der" mid Hin Hin ++
    conv1Back s!"{p}de" s!"%{p}den" s!"%{p}eW" ic mid Hin Hin ++
    conv1WGrad s!"{p}deW" xin s!"%{p}den" ic mid Hin Hin ++
    convBiasGrad s!"{p}deb" s!"%{p}den" mid Hin Hin
  if s == 1 && ic == oc then
    (body ++ addOp s!"{p}dx" s!"%{p}de" dy ic Hin Hin, s!"%{p}dx")
  else
    (body, s!"%{p}de")

-- ════════════ block config + data-driven params ════════════

private def sgd (θ dθ ty' : String) : String :=
  s!"    %{θ}l = stablehlo.constant dense<{LR}> : {ty'}\n" ++
  s!"    %{θ}s = stablehlo.multiply {dθ}, %{θ}l : {ty'}\n" ++
  s!"    %{θ}n = stablehlo.subtract %{θ}, %{θ}s : {ty'}\n"

/-- (p, ic, mid, oc, s, r); spatial threaded from the stem (16×16). -/
private def blocks : List (String × Nat × Nat × Nat × Nat × Nat) :=
  [("b1", 16, 64,  24, 2, 4),    -- 16→8
   ("b2", 24, 96,  24, 1, 6),    -- skip
   ("b3", 24, 96,  32, 2, 6),    -- 8→4
   ("b4", 32, 128, 32, 1, 8),    -- skip
   ("b5", 32, 128, 64, 1, 8),    -- 32→64 (no skip)
   ("b6", 64, 256, 64, 1, 16)]   -- skip

/-- MBConv param triples (name, gradSSA, type) in func-arg order — expand, depthwise,
    SE (2 dense W+b), project — MUST match mbconvFwd/mbconvBack + the layout. -/
private def mbconvParams (p : String) (ic mid oc r : Nat) : List (String × String × String) :=
  [(s!"{p}eW", s!"%{p}deW", ty [mid,ic,1,1]), (s!"{p}eb", s!"%{p}deb", ty [mid]),
   (s!"{p}eg", s!"%{p}dendg", ty [mid]), (s!"{p}ebt", s!"%{p}dendb", ty [mid]),
   (s!"{p}dW", s!"%{p}ddW", ty [mid,1,3,3]), (s!"{p}db", s!"%{p}ddb", ty [mid]),
   (s!"{p}dg", s!"%{p}ddndg", ty [mid]), (s!"{p}dbt", s!"%{p}ddndb", ty [mid]),
   (s!"{p}zW1", s!"%{p}zdWs1", ty [mid,r]), (s!"{p}zb1", s!"%{p}zdbs1", ty [r]),
   (s!"{p}zW2", s!"%{p}zdWs2", ty [r,mid]), (s!"{p}zb2", s!"%{p}zdbs2", ty [mid]),
   (s!"{p}pW", s!"%{p}dpW", ty [oc,mid,1,1]), (s!"{p}pb", s!"%{p}dpb", ty [oc]),
   (s!"{p}pg", s!"%{p}dpndg", ty [oc]), (s!"{p}pbt", s!"%{p}dpndb", ty [oc])]

/-- Whole net's param triples: stem, blocks, head, dense. -/
private def allParams : List (String × String × String) :=
  [("sW", "%dsW", ty [16,3,3,3]), ("sb", "%dsb", ty [16]),
   ("sg", "%dstndg", ty [16]), ("sbt", "%dstndb", ty [16])]
  ++ (blocks.map (fun (p, ic, mid, oc, _, r) => mbconvParams p ic mid oc r)).flatten
  ++ [("hW", "%dhW", ty [128,64,1,1]), ("hb", "%dhb", ty [128]),
      ("hg", "%dhndg", ty [128]), ("hbt", "%dhndb", ty [128])]
  ++ [("Wd", "%dWd", ty [128,10]), ("bd", "%dbd", ty [10])]

private def trainStep : String := Id.run do
  -- ── forward: stem → blocks → head → GAP(4×4) → dense(128→10) ──
  let mut fwd := "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n"
    ++ s!"    %xr = stablehlo.reshape %x : ({ty [BS,3072]}) -> {ty [BS,3,32,32]}\n"
    ++ conv3 "stc" "%xr" "%sW" "%sb" 16 3 32 32 16 16 2
    ++ bnPC "stn" "%stc" "%sg" "%sbt" 16 16 16 (16*16)
    ++ swishAct "str" "%stn" 16 16 16
  let mut cur := "%str"
  let mut curH := 16
  let mut io : List ((String × Nat × Nat × Nat × Nat × Nat) × String × Nat) := []  -- (blk, xin, Hin)
  for blk in blocks do
    let (p, ic, mid, oc, s, r) := blk
    io := io ++ [(blk, cur, curH)]
    fwd := fwd ++ (mbconvFwd p cur ic mid oc curH s r).1
    cur := if s == 1 && ic == oc then s!"%{p}o" else s!"%{p}pn"
    curH := curH / s
  -- head: 1×1 conv (64→128) + BN + swish @ curH
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
    let (p, ic, mid, oc, s, r) := blk
    let (code, out) := mbconvBack p d xin ic mid oc hin s r
    bwd := bwd ++ code
    d := out
  -- stem backward: swish (pre stn) → BN → strided 3×3 weight-grad
  bwd := bwd
    ++ swishBack "dstr" "%stn" d 16 16 16
    ++ bnBackPC "dstn" "stn" "%dstr" 16 16 16
    ++ convBiasGrad "dsb" "%dstn" 16 16 16
    ++ conv3WGradStrided "dsW" "%xr" "%dstn" 3 16 16 16
  -- ── SGD + signature/return, all from the single param list ──
  let upd := String.join (allParams.map (fun (nm, gr, t) => sgd nm gr t))
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [BS,3072]) :: allParams.map (fun (nm, _, t) => s!"%{nm}: {t}") ++ ["%onehot: " ++ ty [BS,10]])
  let retTyL := String.intercalate ", " (allParams.map (fun (_, _, t) => t))
  let retVals := String.intercalate ", " (allParams.map (fun (nm, _, _) => s!"%{nm}n"))
  return "module @m {\n" ++ s!"  func.func @efficientnet_train_step({argSig}) -> ({retTyL}) " ++ "{\n" ++
    fwd ++ cot ++ bwd ++ upd ++ s!"    return {retVals} : {retTyL}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := trainStep
  IO.println s!"rendered EfficientNet train step (BS={BS}, {blocks.length} MBConv blocks): {mlir.length} chars, {allParams.length} params"
  IO.FS.createDirAll "verified_mlir"
  IO.FS.writeFile "verified_mlir/efficientnet_train_step.mlir" mlir
  IO.FS.createDirAll ".lake/build"
  let cargs ← ireeCompileArgs "verified_mlir/efficientnet_train_step.mlir" ".lake/build/efficientnet_train_step_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println "efficientnet FULL train step iree-compile OK → verified_mlir/efficientnet_train_step.mlir"

#eval main
