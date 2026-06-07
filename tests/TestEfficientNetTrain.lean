import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! # E6 — EfficientNet-B0 train-step renderer (faithful [t,c,n,s,k] config) + iree

Full single-batch SGD train step for the EfficientNet-B0 forward of TestEfficientNetFwd.lean
— all-swish, batch norm (E5), the real B0 stage spec (16 MBConv layers, channels
[16,24,40,80,112,192,320], kernels [3,3,5,3,5,5,3], expand [1,6,6,6,6,6,6] with the MBConv1
no-expand first stage), on Imagenette 224² (native B0 resolution, stem stride 2, 224→7).
Data-driven over the generated per-block `(p, ic, mid, oc, s, r, k)` list, threading spatial
dims fwd AND reverse.

vs E5's reduced 6-block net: depthwise fragments parameterized by kernel `k` (5×5 as well as
3×3); the MBConv1 (t=1, mid=ic) blocks skip the expand conv/BN/swish — both forward and the
expand-back (their dx is the depthwise input-grad directly); the B0 [t,c,n,s,k] stage spec
with repeats. Every fragment is the StableHLO of a proven-faithful per-op emitter (swish,
sigmoid, batch-norm 3-term backward, depthwise k×k stride-1/2 — the op is kernel-general,
convs, residual, GAP, dense); SE mirrors `seGate`/`seBlock`/`broadcastFlat` (gradcheck-validated).

Run (rocm): export IREE_BACKEND=rocm; lake env lean tests/TestEfficientNetTrain.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 32      -- 224² B0 is memory-heavy; small batch
private def IMG : Nat := 224    -- Imagenette resolution (B0 native, stem stride 2 → 112)
private def EPS : String := "1.0e-5"
private def LR : String := "0.1"

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

private def dwconvStrided (o x w bnm : String) (c Hout Wout k : Nat) : String :=
  let p := (k-1)/2
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [2, 2], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,2*Hout,2*Wout]}, {ty [c,1,k,k]}) -> {ty [BS,c,Hout,Wout]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hout,Wout]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,c,Hout,Wout]}\n"

/-- Batch-norm per channel (E5): reduce μ/var over batch+spatial [0,2,3], rank-1 γ/β dims=[1].
    Saves `%{o}gb`,`%{o}xh`,`%{o}nf`,`%{o}istd` for the backward. -/
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

private def swishAct (o x : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}s = stablehlo.logistic {x} : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.multiply {x}, %{o}s : {ty [BS,c,Hh,Ww]}\n"

private def addOp (o a b : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.add {a}, {b} : {ty [BS,oc,Hh,Ww]}\n"

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

private def swishBack (o pre dy : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}s = stablehlo.logistic {pre} : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}one = stablehlo.constant dense<1.0> : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}om = stablehlo.subtract %{o}one, %{o}s : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}xom = stablehlo.multiply {pre}, %{o}om : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}in = stablehlo.add %{o}one, %{o}xom : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}sp = stablehlo.multiply %{o}s, %{o}in : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.multiply {dy}, %{o}sp : {ty [BS,c,Hh,Ww]}\n"

/-- Batch-norm per-channel backward: batch-coupled 3-term (Σ over [0,2,3]) + γ/β grads.
    The render of the proven `bnBatchTensor4_grad_input` (E5). -/
private def bnBatchBack (o bn dy : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o}dxh = stablehlo.multiply %{bn}gb, {dy} : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}sdxr = stablehlo.reduce(%{o}dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n" ++
  s!"    %{o}sdx = stablehlo.broadcast_in_dim %{o}sdxr, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xd = stablehlo.multiply %{bn}xh, %{o}dxh : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}sxdr = stablehlo.reduce(%{o}xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n" ++
  s!"    %{o}sxd = stablehlo.broadcast_in_dim %{o}sxdr, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
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

/-- Depthwise k×k input-grad (reverse [2,3] + depthwise conv, pad (k-1)/2). -/
private def dwconvBack (o dy w : String) (c Hh Ww k : Nat) : String :=
  let p := (k-1)/2
  s!"    %{o}rev = stablehlo.reverse {w}, dims = [2, 3] : {ty [c,1,k,k]}\n" ++
  s!"    %{o} = stablehlo.convolution({dy}, %{o}rev)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,Hh,Ww]}, {ty [c,1,k,k]}) -> {ty [BS,c,Hh,Ww]}\n"

/-- Depthwise k×k weight-grad (batch_group_count=c, output [1,c,k,k] → [c,1,k,k], pad (k-1)/2). -/
private def dwconvWGrad (o inp dy : String) (c Hh Ww k : Nat) : String :=
  let p := (k-1)/2
  s!"    %{o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [BS,c,Hh,Ww]}) -> {ty [c,BS,Hh,Ww]}\n" ++
  s!"    %{o}dt = stablehlo.transpose {dy}, dims = [1, 0, 2, 3] : ({ty [BS,c,Hh,Ww]}) -> {ty [c,BS,Hh,Ww]}\n" ++
  s!"    %{o}raw = stablehlo.convolution(%{o}xt, %{o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [c,BS,Hh,Ww]}, {ty [c,BS,Hh,Ww]}) -> {ty [1,c,k,k]}\n" ++
  s!"    %{o} = stablehlo.reshape %{o}raw : ({ty [1,c,k,k]}) -> {ty [c,1,k,k]}\n"

private def convBiasGrad (o dy : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.reduce({dy} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"

private def upsample (o dy : String) (c Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.pad {dy}, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [BS,c,Hh,Ww]}, tensor<f32>) -> {ty [BS,c,2*Hh,2*Ww]}\n"

/-- Depthwise STRIDE-2 input-grad (k×k): zero-upsample dy, reverse [2,3], stride-1 depthwise. -/
private def dwconvStridedBack (o dy w : String) (c Hout Wout k : Nat) : String :=
  let p := (k-1)/2
  upsample s!"{o}u" dy c Hout Wout ++
  s!"    %{o}rev = stablehlo.reverse {w}, dims = [2, 3] : {ty [c,1,k,k]}\n" ++
  s!"    %{o} = stablehlo.convolution(%{o}u, %{o}rev)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,2*Hout,2*Wout]}, {ty [c,1,k,k]}) -> {ty [BS,c,2*Hout,2*Wout]}\n"

/-- Depthwise STRIDE-2 weight-grad (k×k): upsample dy then the stride-1 depthwise weight-grad. -/
private def dwconvWGradStrided (o inp dy : String) (c Hout Wout k : Nat) : String :=
  upsample s!"{o}u" dy c Hout Wout ++ dwconvWGrad o inp s!"%{o}u" c (2*Hout) (2*Wout) k

/-- 3×3 conv weight-grad (stride 1, pad 1). -/
private def conv3WGrad (o inp dy : String) (ic oc Hh Ww : Nat) : String :=
  s!"    %{o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [BS,ic,Hh,Ww]}) -> {ty [ic,BS,Hh,Ww]}\n" ++
  s!"    %{o}dt = stablehlo.transpose {dy}, dims = [1, 0, 2, 3] : ({ty [BS,oc,Hh,Ww]}) -> {ty [oc,BS,Hh,Ww]}\n" ++
  s!"    %{o}raw = stablehlo.convolution(%{o}xt, %{o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ic,BS,Hh,Ww]}, {ty [oc,BS,Hh,Ww]}) -> {ty [ic,oc,3,3]}\n" ++
  s!"    %{o} = stablehlo.transpose %{o}raw, dims = [1, 0, 2, 3] : ({ty [ic,oc,3,3]}) -> {ty [oc,ic,3,3]}\n"

/-- Strided 3×3 conv weight-grad (the B0 stem, stride 2): upsample dy then stride-1 wgrad. -/
private def conv3WGradStrided (o inp dy : String) (ic oc Hh Ww : Nat) : String :=
  upsample s!"{o}u" dy oc Hh Ww ++ conv3WGrad o inp s!"%{o}u" ic oc (2*Hh) (2*Ww)

/-- SE backward — input cotangent `%{p}dds` + dense grads. Reuses `%{p}sq`,`%{p}ex`,`%{p}a1`,`%{p}gate`. -/
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

-- ════════════ MBConv block forward / backward (stride `s`, SE `r`, kernel `k`) ════════════

private def mbconvFwd (p x : String) (ic mid oc Hin s r k : Nat) : String × String :=
  let Hout := Hin / s
  let hasExpand := mid != ic
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

private def mbconvBack (p dy xin : String) (ic mid oc Hin s r k : Nat) : String × String :=
  let Hout := Hin / s
  let hasExpand := mid != ic
  let dwInName := if hasExpand then s!"%{p}es" else xin   -- depthwise input (= block input if no expand)
  let proj :=
    bnBatchBack s!"{p}dpn" s!"{p}pn" dy oc Hout Hout ++
    conv1Back s!"{p}dp" s!"%{p}dpn" s!"%{p}pW" mid oc Hout Hout ++
    conv1WGrad s!"{p}dpW" s!"%{p}zse" s!"%{p}dpn" mid oc Hout Hout ++
    convBiasGrad s!"{p}dpb" s!"%{p}dpn" oc Hout Hout
  let se := seBack s!"{p}z" s!"%{p}ds" s!"%{p}dp" s!"%{p}zW1" s!"%{p}zW2" mid Hout Hout r
  let dw :=
    swishBack s!"{p}ddr" s!"%{p}dn" s!"%{p}zdds" mid Hout Hout ++
    bnBatchBack s!"{p}ddn" s!"{p}dn" s!"%{p}ddr" mid Hout Hout ++
    (if s == 2 then
       dwconvStridedBack s!"{p}dd" s!"%{p}ddn" s!"%{p}dW" mid Hout Hout k ++
       dwconvWGradStrided s!"{p}ddW" dwInName s!"%{p}ddn" mid Hout Hout k
     else
       dwconvBack s!"{p}dd" s!"%{p}ddn" s!"%{p}dW" mid Hin Hin k ++
       dwconvWGrad s!"{p}ddW" dwInName s!"%{p}ddn" mid Hin Hin k) ++
    convBiasGrad s!"{p}ddb" s!"%{p}ddn" mid Hout Hout
  let (exB, dxMain) :=
    if hasExpand then
      (swishBack s!"{p}der" s!"%{p}en" s!"%{p}dd" mid Hin Hin ++
       bnBatchBack s!"{p}den" s!"{p}en" s!"%{p}der" mid Hin Hin ++
       conv1Back s!"{p}de" s!"%{p}den" s!"%{p}eW" ic mid Hin Hin ++
       conv1WGrad s!"{p}deW" xin s!"%{p}den" ic mid Hin Hin ++
       convBiasGrad s!"{p}deb" s!"%{p}den" mid Hin Hin, s!"%{p}de")
    else ("", s!"%{p}dd")    -- no expand: dx = depthwise input-grad directly (= block-input cotangent)
  let body := proj ++ se ++ dw ++ exB
  if s == 1 && ic == oc then
    (body ++ addOp s!"{p}dx" dxMain dy ic Hin Hin, s!"%{p}dx")
  else
    (body, dxMain)

-- ════════════ EfficientNet-B0 config + data-driven params ════════════

private def sgd (θ dθ ty' : String) : String :=
  s!"    %{θ}l = stablehlo.constant dense<{LR}> : {ty'}\n" ++
  s!"    %{θ}s = stablehlo.multiply {dθ}, %{θ}l : {ty'}\n" ++
  s!"    %{θ}n = stablehlo.subtract %{θ}, %{θ}s : {ty'}\n"

private def stages : List (Nat × Nat × Nat × Nat × Nat) :=
  [(1, 16,  1, 1, 3), (6, 24,  2, 2, 3), (6, 40,  2, 2, 5), (6, 80,  3, 2, 3),
   (6, 112, 3, 1, 5), (6, 192, 4, 2, 5), (6, 320, 1, 1, 3)]

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

/-- MBConv param triples (name, gradSSA, type), conditional on expand (MBConv1 has none),
    kernel-`k` depthwise. MUST match mbconvFwd/mbconvBack + the layout. -/
private def mbconvParams (p : String) (ic mid oc r k : Nat) : List (String × String × String) :=
  (if mid != ic then
    [(s!"{p}eW", s!"%{p}deW", ty [mid,ic,1,1]), (s!"{p}eb", s!"%{p}deb", ty [mid]),
     (s!"{p}eg", s!"%{p}dendg", ty [mid]), (s!"{p}ebt", s!"%{p}dendb", ty [mid])]
   else []) ++
  [(s!"{p}dW", s!"%{p}ddW", ty [mid,1,k,k]), (s!"{p}db", s!"%{p}ddb", ty [mid]),
   (s!"{p}dg", s!"%{p}ddndg", ty [mid]), (s!"{p}dbt", s!"%{p}ddndb", ty [mid]),
   (s!"{p}zW1", s!"%{p}zdWs1", ty [mid,r]), (s!"{p}zb1", s!"%{p}zdbs1", ty [r]),
   (s!"{p}zW2", s!"%{p}zdWs2", ty [r,mid]), (s!"{p}zb2", s!"%{p}zdbs2", ty [mid]),
   (s!"{p}pW", s!"%{p}dpW", ty [oc,mid,1,1]), (s!"{p}pb", s!"%{p}dpb", ty [oc]),
   (s!"{p}pg", s!"%{p}dpndg", ty [oc]), (s!"{p}pbt", s!"%{p}dpndb", ty [oc])]

private def allParams : List (String × String × String) :=
  [("sW", "%dsW", ty [32,3,3,3]), ("sb", "%dsb", ty [32]),
   ("sg", "%dstndg", ty [32]), ("sbt", "%dstndb", ty [32])]
  ++ (blocks.map (fun (p, ic, mid, oc, _, r, k) => mbconvParams p ic mid oc r k)).flatten
  ++ [("hW", "%dhW", ty [1280,320,1,1]), ("hb", "%dhb", ty [1280]),
      ("hg", "%dhndg", ty [1280]), ("hbt", "%dhndb", ty [1280])]
  ++ [("Wd", "%dWd", ty [1280,10]), ("bd", "%dbd", ty [10])]

private def trainStep : String := Id.run do
  -- ── forward: stem (3→32, stride 1) → B0 blocks → head (320→1280) → GAP → dense ──
  let mut fwd := "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n"
    ++ s!"    %xr = stablehlo.reshape %x : ({ty [BS,3*IMG*IMG]}) -> {ty [BS,3,IMG,IMG]}\n"
    ++ conv3 "stc" "%xr" "%sW" "%sb" 32 3 IMG IMG (IMG/2) (IMG/2) 2
    ++ bnBatch "stn" "%stc" "%sg" "%sbt" 32 (IMG/2) (IMG/2) ((IMG/2)*(IMG/2))
    ++ swishAct "str" "%stn" 32 (IMG/2) (IMG/2)
  let mut cur := "%str"
  let mut curH := IMG/2
  let mut io : List ((String × Nat × Nat × Nat × Nat × Nat × Nat) × String × Nat) := []
  for blk in blocks do
    let (p, ic, mid, oc, s, r, k) := blk
    io := io ++ [(blk, cur, curH)]
    fwd := fwd ++ (mbconvFwd p cur ic mid oc curH s r k).1
    cur := if s == 1 && ic == oc then s!"%{p}o" else s!"%{p}pn"
    curH := curH / s
  -- head: 1×1 conv (320→1280) + BN + swish @ curH
  let hd := curH
  fwd := fwd
    ++ conv1 "h" cur "%hW" "%hb" 1280 320 hd hd
    ++ bnBatch "hn" "%h" "%hg" "%hbt" 1280 hd hd (hd*hd)
    ++ swishAct "hr" "%hn" 1280 hd hd
    ++ s!"    %gaps = stablehlo.reduce(%hr init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,1280,hd,hd]}, tensor<f32>) -> {ty [BS,1280]}\n"
    ++ s!"    %gapnf = stablehlo.constant dense<{hd*hd}.0> : {ty [BS,1280]}\n"
    ++ s!"    %gap = stablehlo.divide %gaps, %gapnf : {ty [BS,1280]}\n"
    ++ s!"    %ld = stablehlo.dot_general %gap, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,1280]}, {ty [1280,10]}) -> {ty [BS,10]}\n"
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
    s!"    %dgap = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [BS,10]}, {ty [1280,10]}) -> {ty [BS,1280]}\n"
    ++ s!"    %dWd = stablehlo.dot_general %gap, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,1280]}, {ty [BS,10]}) -> {ty [1280,10]}\n"
    ++ s!"    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,10]}, tensor<f32>) -> {ty [10]}\n"
    ++ s!"    %dgnf = stablehlo.constant dense<{hd*hd}.0> : {ty [BS,1280]}\n"
    ++ s!"    %dgs = stablehlo.divide %dgap, %dgnf : {ty [BS,1280]}\n"
    ++ s!"    %dgapin = stablehlo.broadcast_in_dim %dgs, dims = [0, 1] : ({ty [BS,1280]}) -> {ty [BS,1280,hd,hd]}\n"
    ++ swishBack "dhr" "%hn" "%dgapin" 1280 hd hd
    ++ bnBatchBack "dhn" "hn" "%dhr" 1280 hd hd
    ++ conv1Back "dh" "%dhn" "%hW" 320 1280 hd hd
    ++ conv1WGrad "dhW" cur "%dhn" 320 1280 hd hd
    ++ convBiasGrad "dhb" "%dhn" 1280 hd hd
  let mut d := "%dh"
  for (blk, xin, hin) in io.reverse do
    let (p, ic, mid, oc, s, r, k) := blk
    let (code, out) := mbconvBack p d xin ic mid oc hin s r k
    bwd := bwd ++ code
    d := out
  -- stem backward: swish (pre stn) → BN → 3×3 weight-grad (stride 2)
  let sH := IMG/2
  bwd := bwd
    ++ swishBack "dstr" "%stn" d 32 sH sH
    ++ bnBatchBack "dstn" "stn" "%dstr" 32 sH sH
    ++ convBiasGrad "dsb" "%dstn" 32 sH sH
    ++ conv3WGradStrided "dsW" "%xr" "%dstn" 3 32 sH sH
  -- ── SGD + signature/return, all from the single param list ──
  let upd := String.join (allParams.map (fun (nm, gr, t) => sgd nm gr t))
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [BS,3*IMG*IMG]) :: allParams.map (fun (nm, _, t) => s!"%{nm}: {t}") ++ ["%onehot: " ++ ty [BS,10]])
  let retTyL := String.intercalate ", " (allParams.map (fun (_, _, t) => t))
  let retVals := String.intercalate ", " (allParams.map (fun (nm, _, _) => s!"%{nm}n"))
  return "module @m {\n" ++ s!"  func.func @efficientnet_train_step({argSig}) -> ({retTyL}) " ++ "{\n" ++
    fwd ++ cot ++ bwd ++ upd ++ s!"    return {retVals} : {retTyL}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := trainStep
  IO.println s!"rendered EfficientNet-B0 train step (BS={BS}, {blocks.length} MBConv layers): {mlir.length} chars, {allParams.length} params"
  IO.FS.createDirAll "verified_mlir"
  IO.FS.writeFile "verified_mlir/efficientnet_train_step.mlir" mlir
  IO.FS.createDirAll ".lake/build"
  let cargs ← ireeCompileArgs "verified_mlir/efficientnet_train_step.mlir" ".lake/build/efficientnet_train_step_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println "efficientnet-B0 FULL train step iree-compile OK → verified_mlir/efficientnet_train_step.mlir"

#eval main
