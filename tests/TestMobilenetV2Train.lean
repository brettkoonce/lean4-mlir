import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! # C4b — MobileNetV2 train-step renderer (forward + backward + SGD) + iree

The full single-batch SGD train step for the small MobileNetV2 of C4a (ch6 B9b
analog). Exercises every op type's forward + backward + SGD:
  stem 3×3 stride-2 conv + BN + relu6 → maxpool → IR-A (skip) → IR-B (no-skip)
  → head 1×1 conv (64→128) + BN + relu6 → GAP → dense, softmax-CE mean-loss
  cotangent, full reverse pass, SGD updates.

The head's relu6 before GAP is ESSENTIAL (the MNv2 "features" layer): per-example
instance-norm zeroes each channel's spatial mean, so GAP of a raw linear-bottleneck
BN is the constant β; the relu6 restores a per-input pooled mean so the net learns.

The genuinely-new backward fragments vs ch6:
  * depthwise input-grad  = reverse the [c,1,3,3] filters over [2,3] + depthwise
    conv (feature_group_count=c) — the C1 `depthwiseBack` pattern.
  * depthwise weight-grad = batch_group_count=c convolution (the XLA depthwise
    filter-grad trick): transpose inp/dy to [c,B,H,W], conv → [1,c,3,3], reshape.
  * relu6 backward        = two-sided mask select(0<pre<6, dy, 0) — the C2 op.
  * 1×1 conv in/weight-grad = pad-0 transpose-trick convs (expand/project).
The stem (stride-2) reuses ch6's strided weight-grad (upsample + transpose conv).

If this iree-compiles, the trainer exe (C4c) wires it into the FFI SGD loop.

Run (rocm): export IREE_BACKEND=rocm; lake env lean tests/TestMobilenetV2Train.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 128
private def EPS : String := "1.0e-5"
private def LR : String := "0.1"

-- ════════════ forward fragments ([B,C,H,W]; `o` bare prefix → result `%{o}`) ════════════

/-- 3×3 SAME conv (stride `s`, pad 1) + bias — the stem. Result `%{o}`. -/
private def conv3 (o x w bnm : String) (oc ic Hin Win Hout Wout s : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [{s}, {s}], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,Hin,Win]}, {ty [oc,ic,3,3]}) -> {ty [BS,oc,Hout,Wout]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hout,Wout]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Hout,Wout]}\n"

/-- 1×1 conv (pad 0, stride 1) + bias — expand/project. Result `%{o}`. -/
private def conv1 (o x w bnm : String) (oc ic Hh Ww : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,Hh,Ww]}, {ty [oc,ic,1,1]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Hh,Ww]}\n"

/-- Depthwise 3×3 SAME conv (stride 1, pad 1, feature_group_count=c, [c,1,3,3]). Result `%{o}`. -/
private def dwconv (o x w bnm : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,Hh,Ww]}, {ty [c,1,3,3]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,c,Hh,Ww]}\n"

/-- Per-channel BN forward; saves `%{o}xh`, `%{o}istd`, `%{o}nf`, `%{o}gb` for backward. -/
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

-- ════════════ backward fragments ════════════

/-- ReLU6 backward: two-sided mask `0 < pre < 6` (the C2 selectMid op). Result `%{o}`. -/
private def relu6Back (o pre dy : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}z = stablehlo.constant dense<0.0> : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}six = stablehlo.constant dense<6.0> : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}g0 = stablehlo.compare GT, {pre}, %{o}z : ({ty [BS,c,Hh,Ww]}, {ty [BS,c,Hh,Ww]}) -> {tyI1 [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}l6 = stablehlo.compare LT, {pre}, %{o}six : ({ty [BS,c,Hh,Ww]}, {ty [BS,c,Hh,Ww]}) -> {tyI1 [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}m = stablehlo.and %{o}g0, %{o}l6 : {tyI1 [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.select %{o}m, {dy}, %{o}z : {tyI1 [BS,c,Hh,Ww]}, {ty [BS,c,Hh,Ww]}\n"

/-- Per-channel BN backward (reuses forward `bn` saves). Result dx `%{o}`; param
    grads `%{o}dg` (dγ : [oc]) and `%{o}db` (dβ : [oc]). -/
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

/-- 1×1 conv input-grad (pad 0): transposed kernel, conv. Result `%{o}` [B,ic,H,W]. -/
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

/-- Depthwise input-grad: reverse the [c,1,3,3] filters over [2,3], depthwise conv
    (feature_group_count=c). The C1 `depthwiseBack` pattern. Result `%{o}` [B,c,H,W]. -/
private def dwconvBack (o dy w : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}rev = stablehlo.reverse {w}, dims = [2, 3] : {ty [c,1,3,3]}\n" ++
  s!"    %{o} = stablehlo.convolution({dy}, %{o}rev)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,Hh,Ww]}, {ty [c,1,3,3]}) -> {ty [BS,c,Hh,Ww]}\n"

/-- Depthwise weight-grad: transpose inp/dy to [c,B,H,W], batch_group_count=c conv
    → [1,c,3,3], reshape to [c,1,3,3] (the XLA depthwise filter-grad trick). -/
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

/-- Zero-upsample [B,oc,Hh,Ww] → [B,oc,2Hh,2Ww] (lhs_dilation) for the strided stem. -/
private def upsample (o dy : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.pad {dy}, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [BS,oc,2*Hh,2*Ww]}\n"

/-- 3×3 conv weight-grad (stride 1, pad 1). Result `%{o}` [oc,ic,3,3]. -/
private def conv3WGrad (o inp dy : String) (ic oc Hh Ww : Nat) : String :=
  s!"    %{o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [BS,ic,Hh,Ww]}) -> {ty [ic,BS,Hh,Ww]}\n" ++
  s!"    %{o}dt = stablehlo.transpose {dy}, dims = [1, 0, 2, 3] : ({ty [BS,oc,Hh,Ww]}) -> {ty [oc,BS,Hh,Ww]}\n" ++
  s!"    %{o}raw = stablehlo.convolution(%{o}xt, %{o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ic,BS,Hh,Ww]}, {ty [oc,BS,Hh,Ww]}) -> {ty [ic,oc,3,3]}\n" ++
  s!"    %{o} = stablehlo.transpose %{o}raw, dims = [1, 0, 2, 3] : ({ty [ic,oc,3,3]}) -> {ty [oc,ic,3,3]}\n"

/-- Strided 3×3 conv weight-grad: upsample dy to 2Hh×2Ww, then stride-1 weight grad. -/
private def conv3WGradStrided (o inp dy : String) (ic oc Hh Ww : Nat) : String :=
  upsample s!"{o}u" dy oc Hh Ww ++ conv3WGrad o inp s!"%{o}u" ic oc (2*Hh) (2*Ww)

/-- Max-pool backward (select_and_scatter). src [B,c,2Hh,2Ww], dy [B,c,Hh,Ww]. Result `%{o}`. -/
private def maxpoolBack (o src dy : String) (c Hh Ww : Nat) : String :=
  s!"    %{o} = \"stablehlo.select_and_scatter\"({src}, {dy}, %sc) (" ++ "{\n" ++
  "      ^bb0(%qa: tensor<f32>, %qb: tensor<f32>):\n" ++
  "        %qge = stablehlo.compare GE, %qa, %qb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
  "        stablehlo.return %qge : tensor<i1>\n" ++
  "    }, " ++ "{\n" ++
  "      ^bb0(%qc: tensor<f32>, %qd: tensor<f32>):\n" ++
  "        %qs = stablehlo.add %qc, %qd : tensor<f32>\n" ++
  "        stablehlo.return %qs : tensor<f32>\n" ++
  "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
  s!" : ({ty [BS,c,2*Hh,2*Ww]}, {ty [BS,c,Hh,Ww]}, tensor<f32>) -> {ty [BS,c,2*Hh,2*Ww]}\n"

-- ════════════ inverted-residual block forward / backward ════════════

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

/-- IR block backward. `dy` = cotangent of the block output; `xin` = block input.
    Result dx `%{p}dx` (skip: main expand-input grad + dy; no-skip: `%{p}de`). -/
private def irBlockBack (p dy xin : String) (ic mid oc Hh Ww : Nat) (skip : Bool) : String × String :=
  let code :=
    -- project: BN back → 1×1 conv back (+ weight/bias grad)
    bnBackPC s!"{p}dpn" s!"{p}pn" dy oc Hh Ww ++
    conv1Back s!"{p}dp" s!"%{p}dpn" s!"%{p}pW" mid oc Hh Ww ++
    conv1WGrad s!"{p}dpW" s!"%{p}dr" s!"%{p}dpn" mid oc Hh Ww ++
    convBiasGrad s!"{p}dpb" s!"%{p}dpn" oc Hh Ww ++
    -- depthwise: relu6 back (mask on dn) → BN back → depthwise back (+ weight/bias grad)
    relu6Back s!"{p}ddr" s!"%{p}dn" s!"%{p}dp" mid Hh Ww ++
    bnBackPC s!"{p}ddn" s!"{p}dn" s!"%{p}ddr" mid Hh Ww ++
    dwconvBack s!"{p}dd" s!"%{p}ddn" s!"%{p}dW" mid Hh Ww ++
    dwconvWGrad s!"{p}ddW" s!"%{p}er" s!"%{p}ddn" mid Hh Ww ++
    convBiasGrad s!"{p}ddb" s!"%{p}ddn" mid Hh Ww ++
    -- expand: relu6 back (mask on en) → BN back → 1×1 conv back (+ weight/bias grad)
    relu6Back s!"{p}der" s!"%{p}en" s!"%{p}dd" mid Hh Ww ++
    bnBackPC s!"{p}den" s!"{p}en" s!"%{p}der" mid Hh Ww ++
    conv1Back s!"{p}de" s!"%{p}den" s!"%{p}eW" ic mid Hh Ww ++
    conv1WGrad s!"{p}deW" xin s!"%{p}den" ic mid Hh Ww ++
    convBiasGrad s!"{p}deb" s!"%{p}den" mid Hh Ww
  if skip then
    (code ++ addOp s!"{p}dx" s!"%{p}de" dy ic Hh Ww, s!"%{p}dx")
  else
    (code, s!"%{p}de")

-- ════════════ data-driven params (forward arg order) ════════════

private def sgd (θ dθ ty' : String) : String :=
  s!"    %{θ}l = stablehlo.constant dense<{LR}> : {ty'}\n" ++
  s!"    %{θ}s = stablehlo.multiply {dθ}, %{θ}l : {ty'}\n" ++
  s!"    %{θ}n = stablehlo.subtract %{θ}, %{θ}s : {ty'}\n"

/-- IR block param triples (name, gradSSA, type) in func-arg order: expand/dw/project. -/
private def irBlkParams (p : String) (ic mid oc : Nat) : List (String × String × String) :=
  [(s!"{p}eW", s!"%{p}deW", ty [mid,ic,1,1]), (s!"{p}eb", s!"%{p}deb", ty [mid]),
   (s!"{p}eg", s!"%{p}dendg", ty [mid]), (s!"{p}ebt", s!"%{p}dendb", ty [mid]),
   (s!"{p}dW", s!"%{p}ddW", ty [mid,1,3,3]), (s!"{p}db", s!"%{p}ddb", ty [mid]),
   (s!"{p}dg", s!"%{p}ddndg", ty [mid]), (s!"{p}dbt", s!"%{p}ddndb", ty [mid]),
   (s!"{p}pW", s!"%{p}dpW", ty [oc,mid,1,1]), (s!"{p}pb", s!"%{p}dpb", ty [oc]),
   (s!"{p}pg", s!"%{p}dpndg", ty [oc]), (s!"{p}pbt", s!"%{p}dpndb", ty [oc])]

/-- Whole net's param triples: stem, IR-A, IR-B, dense. -/
private def allParams : List (String × String × String) :=
  [("sW", "%dsW", ty [32,3,3,3]), ("sb", "%dsb", ty [32]),
   ("sg", "%dstndg", ty [32]), ("sbt", "%dstndb", ty [32])]
  ++ irBlkParams "ira" 32 64 32
  ++ irBlkParams "irb" 32 64 64
  ++ [("hW", "%dhW", ty [128,64,1,1]), ("hb", "%dhb", ty [128]),
      ("hg", "%dhndg", ty [128]), ("hbt", "%dhndb", ty [128])]
  ++ [("Wd", "%dWd", ty [128,10]), ("bd", "%dbd", ty [10])]

private def trainStep : String := Id.run do
  -- ── forward: stem → maxpool → IR-A → IR-B → GAP(8×8) → dense(64→10) ──
  let mut fwd := "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n"
    ++ s!"    %xr = stablehlo.reshape %x : ({ty [BS,3072]}) -> {ty [BS,3,32,32]}\n"
    ++ conv3 "stc" "%xr" "%sW" "%sb" 32 3 32 32 16 16 2
    ++ bnPC "stn" "%stc" "%sg" "%sbt" 32 16 16 (16*16)
    ++ relu6 "str" "%stn" 32 16 16
    ++ maxpool "stp" "%str" 32 16 16
  let (a, _) := irBlockFwd "ira" "%stp" 32 64 32 8 8 true
  let (b, ob) := irBlockFwd "irb" "%irao" 32 64 64 8 8 false
  -- head: 1×1 conv (64→128) → BN → relu6 (the standard MNv2 "features" layer).
  -- ESSENTIAL: GAP of a per-example instance-normed BN is just β (constant across
  -- inputs); the relu6 gives the pooled tensor an input-varying mean so the net learns.
  fwd := fwd ++ a ++ b
    ++ conv1 "h" ob "%hW" "%hb" 128 64 8 8
    ++ bnPC "hn" "%h" "%hg" "%hbt" 128 8 8 (8*8)
    ++ relu6 "hr" "%hn" 128 8 8
    ++ s!"    %gaps = stablehlo.reduce(%hr init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,128,8,8]}, tensor<f32>) -> {ty [BS,128]}\n"
    ++ s!"    %gapnf = stablehlo.constant dense<64.0> : {ty [BS,128]}\n"
    ++ s!"    %gap = stablehlo.divide %gaps, %gapnf : {ty [BS,128]}\n"
    ++ s!"    %ld = stablehlo.dot_general %gap, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,128]}, {ty [128,10]}) -> {ty [BS,10]}\n"
    ++ s!"    %ldb = stablehlo.broadcast_in_dim %bd, dims = [1] : ({ty [10]}) -> {ty [BS,10]}\n"
    ++ s!"    %logits = stablehlo.add %ld, %ldb : {ty [BS,10]}\n"
  -- ── loss cotangent dy = (softmax(logits) − onehot) / B (mean loss) ──
  let cot :=
    s!"    %le = stablehlo.exponential %logits : {ty [BS,10]}\n"
    ++ s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [BS,10]}, tensor<f32>) -> {ty [BS]}\n"
    ++ s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [BS]}) -> {ty [BS,10]}\n"
    ++ s!"    %lsm = stablehlo.divide %le, %lsb : {ty [BS,10]}\n"
    ++ s!"    %dyr = stablehlo.subtract %lsm, %onehot : {ty [BS,10]}\n"
    ++ s!"    %bnc = stablehlo.constant dense<{BS}.0> : {ty [BS,10]}\n"
    ++ s!"    %dy = stablehlo.divide %dyr, %bnc : {ty [BS,10]}\n"
  -- ── backward: dense + GAP → IR-B → IR-A → maxpool → stem ──
  let mut bwd :=
    s!"    %dgap = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [BS,10]}, {ty [128,10]}) -> {ty [BS,128]}\n"
    ++ s!"    %dWd = stablehlo.dot_general %gap, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,128]}, {ty [BS,10]}) -> {ty [128,10]}\n"
    ++ s!"    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,10]}, tensor<f32>) -> {ty [10]}\n"
    ++ s!"    %dgnf = stablehlo.constant dense<64.0> : {ty [BS,128]}\n"
    ++ s!"    %dgs = stablehlo.divide %dgap, %dgnf : {ty [BS,128]}\n"
    ++ s!"    %dgapin = stablehlo.broadcast_in_dim %dgs, dims = [0, 1] : ({ty [BS,128]}) -> {ty [BS,128,8,8]}\n"
  -- head backward: relu6 back (mask on %hn) → BN back → 1×1 conv back (→ dy for IR-B)
  bwd := bwd
    ++ relu6Back "dhr" "%hn" "%dgapin" 128 8 8
    ++ bnBackPC "dhn" "hn" "%dhr" 128 8 8
    ++ conv1Back "dh" "%dhn" "%hW" 64 128 8 8
    ++ conv1WGrad "dhW" "%irbpn" "%dhn" 64 128 8 8
    ++ convBiasGrad "dhb" "%dhn" 128 8 8
    ++ (irBlockBack "irb" "%dh" "%irao" 32 64 64 8 8 false).1
    ++ (irBlockBack "ira" "%irbde" "%stp" 32 64 32 8 8 true).1
    ++ maxpoolBack "dmp" "%str" "%iradx" 32 8 8
    ++ relu6Back "dstr" "%stn" "%dmp" 32 16 16
    ++ bnBackPC "dstn" "stn" "%dstr" 32 16 16
    ++ convBiasGrad "dsb" "%dstn" 32 16 16
    ++ conv3WGradStrided "dsW" "%xr" "%dstn" 3 32 16 16
  -- ── SGD + signature/return, all from the single param list ──
  let upd := String.join (allParams.map (fun (nm, gr, t) => sgd nm gr t))
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [BS,3072]) :: allParams.map (fun (nm, _, t) => s!"%{nm}: {t}") ++ ["%onehot: " ++ ty [BS,10]])
  let retTyL := String.intercalate ", " (allParams.map (fun (_, _, t) => t))
  let retVals := String.intercalate ", " (allParams.map (fun (nm, _, _) => s!"%{nm}n"))
  return "module @m {\n" ++ s!"  func.func @mobilenetv2_train_step({argSig}) -> ({retTyL}) " ++ "{\n" ++
    fwd ++ cot ++ bwd ++ upd ++ s!"    return {retVals} : {retTyL}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := trainStep
  IO.println s!"rendered MobileNetV2 train step (BS={BS}): {mlir.length} chars, {allParams.length} params"
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
