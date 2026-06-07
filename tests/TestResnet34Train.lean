import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! # B9b (de-risk) — ResNet-34 train-step BACKWARD helpers, small representative net

Before the full 34-layer train step, validate the backward machinery on a small net
that exercises every op type's forward+backward+SGD: stem(conv+BN+relu) → maxpool →
1 identity block → 1 strided downsample block → GAP → dense, softmax-CE cotangent,
full reverse pass (residual fan-ins, strided conv backward via zero-upsample, per-
channel BN backward, maxpool select_and_scatter, GAP/dense backward), and SGD updates
for every parameter. If this iree-compiles, the full depth is the same helpers in a
longer fold.

Run (rocm): export IREE_BACKEND=rocm; lake env lean tests/TestResnet34Train.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 2
private def EPS : String := "1.0e-5"
private def LR : String := "0.05"

-- ════════════ forward fragments ([B,C,H,W]; `o` bare prefix → result `%{o}`) ════════════

private def conv (o x w bnm : String) (oc ic Hin Win Hout Wout s : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [{s}, {s}], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,Hin,Win]}, {ty [oc,ic,3,3]}) -> {ty [BS,oc,Hout,Wout]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hout,Wout]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Hout,Wout]}\n"

/-- Per-channel BN forward; saves `%{o}xh` (x̂), `%{o}istd`, `%{o}nf`, `%{o}gb` for the backward. -/
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

private def relu (o x : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o}z = stablehlo.constant dense<0.0> : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.maximum {x}, %{o}z : {ty [BS,oc,Hh,Ww]}\n"

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

/-- ReLU backward: mask by the saved pre-activation `pre`. Result `%{o}`. -/
private def reluBack (o pre dy : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o}z = stablehlo.constant dense<0.0> : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}m = stablehlo.compare GT, {pre}, %{o}z : ({ty [BS,oc,Hh,Ww]}, {ty [BS,oc,Hh,Ww]}) -> {tyI1 [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.select %{o}m, {dy}, %{o}z : {tyI1 [BS,oc,Hh,Ww]}, {ty [BS,oc,Hh,Ww]}\n"

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

/-- Conv input-grad (stride 1): reversed-kernel SAME conv. Result `%{o}` [B,ic,H,W]. -/
private def convBack (o dy w : String) (ic oc Hh Ww : Nat) : String :=
  s!"    %{o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,ic,3,3]}) -> {ty [ic,oc,3,3]}\n" ++
  s!"    %{o}r = stablehlo.reverse %{o}t, dims = [2, 3] : {ty [ic,oc,3,3]}\n" ++
  s!"    %{o} = stablehlo.convolution({dy}, %{o}r)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,oc,Hh,Ww]}, {ty [ic,oc,3,3]}) -> {ty [BS,ic,Hh,Ww]}\n"

/-- Conv weight-grad (stride 1). Result `%{o}` [oc,ic,3,3]. -/
private def convWGrad (o inp dy : String) (ic oc Hh Ww : Nat) : String :=
  s!"    %{o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [BS,ic,Hh,Ww]}) -> {ty [ic,BS,Hh,Ww]}\n" ++
  s!"    %{o}dt = stablehlo.transpose {dy}, dims = [1, 0, 2, 3] : ({ty [BS,oc,Hh,Ww]}) -> {ty [oc,BS,Hh,Ww]}\n" ++
  s!"    %{o}raw = stablehlo.convolution(%{o}xt, %{o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ic,BS,Hh,Ww]}, {ty [oc,BS,Hh,Ww]}) -> {ty [ic,oc,3,3]}\n" ++
  s!"    %{o} = stablehlo.transpose %{o}raw, dims = [1, 0, 2, 3] : ({ty [ic,oc,3,3]}) -> {ty [oc,ic,3,3]}\n"

private def convBiasGrad (o dy : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.reduce({dy} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"

/-- Zero-upsample `[B,oc,Hh,Ww] → [B,oc,2Hh,2Ww]` (pad interior+high = 1) — the
    decimate-backward / `lhs_dilation` for the strided-conv backward. Result `%{o}`. -/
private def upsample (o dy : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.pad {dy}, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [BS,oc,2*Hh,2*Ww]}\n"

/-- Strided conv input-grad: upsample dy to 2Hh×2Ww, then reversed-kernel SAME conv.
    Result `%{o}` [B,ic,2Hh,2Ww]. -/
private def convBackStrided (o dy w : String) (ic oc Hh Ww : Nat) : String :=
  upsample s!"{o}u" dy oc Hh Ww ++
  s!"    %{o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,ic,3,3]}) -> {ty [ic,oc,3,3]}\n" ++
  s!"    %{o}r = stablehlo.reverse %{o}t, dims = [2, 3] : {ty [ic,oc,3,3]}\n" ++
  s!"    %{o} = stablehlo.convolution(%{o}u, %{o}r)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,oc,2*Hh,2*Ww]}, {ty [ic,oc,3,3]}) -> {ty [BS,ic,2*Hh,2*Ww]}\n"

/-- Strided conv weight-grad: weight grad of stride-1 conv between input [B,ic,2Hh,2Ww]
    and the upsampled cotangent. Result `%{o}` [oc,ic,3,3]. -/
private def convWGradStrided (o inp dy : String) (ic oc Hh Ww : Nat) : String :=
  upsample s!"{o}u" dy oc Hh Ww ++ convWGrad o inp s!"%{o}u" ic oc (2*Hh) (2*Ww)

/-- Max-pool backward (select_and_scatter): route dy to the window argmax of `src`.
    Result `%{o}` [B,c,2Hh,2Ww] (src is the pre-pool [B,c,2Hh,2Ww], dy [B,c,Hh,Ww]). -/
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

-- ════════════ block forward / backward ════════════

private def idBlockFwd (p x : String) (c Hh Ww : Nat) : String × String :=
  let m := Hh*Ww
  (conv s!"{p}c1" x s!"%{p}W1" s!"%{p}b1" c c Hh Ww Hh Ww 1 ++
   bnPC s!"{p}n1" s!"%{p}c1" s!"%{p}g1" s!"%{p}bt1" c Hh Ww m ++
   relu s!"{p}r1" s!"%{p}n1" c Hh Ww ++
   conv s!"{p}c2" s!"%{p}r1" s!"%{p}W2" s!"%{p}b2" c c Hh Ww Hh Ww 1 ++
   bnPC s!"{p}n2" s!"%{p}c2" s!"%{p}g2" s!"%{p}bt2" c Hh Ww m ++
   addOp s!"{p}a" s!"%{p}n2" x c Hh Ww ++
   relu s!"{p}o" s!"%{p}a" c Hh Ww, s!"%{p}o")

/-- Identity block backward. `dy` cotangent of `%{p}o`, `xin` block input. Result dx `%{p}dx`. -/
private def idBlockBack (p dy xin : String) (c Hh Ww : Nat) : String × String :=
  (reluBack s!"{p}da" s!"%{p}a" dy c Hh Ww ++
   bnBackPC s!"{p}dn2" s!"{p}n2" s!"%{p}da" c Hh Ww ++
   convBack s!"{p}dc2" s!"%{p}dn2" s!"%{p}W2" c c Hh Ww ++
   convWGrad s!"{p}dW2" s!"%{p}r1" s!"%{p}dn2" c c Hh Ww ++
   convBiasGrad s!"{p}db2" s!"%{p}dn2" c Hh Ww ++
   reluBack s!"{p}dr1" s!"%{p}n1" s!"%{p}dc2" c Hh Ww ++
   bnBackPC s!"{p}dn1" s!"{p}n1" s!"%{p}dr1" c Hh Ww ++
   convBack s!"{p}dc1" s!"%{p}dn1" s!"%{p}W1" c c Hh Ww ++
   convWGrad s!"{p}dW1" xin s!"%{p}dn1" c c Hh Ww ++
   convBiasGrad s!"{p}db1" s!"%{p}dn1" c Hh Ww ++
   addOp s!"{p}dx" s!"%{p}dc1" s!"%{p}da" c Hh Ww, s!"%{p}dx")

private def downBlockFwd (p x : String) (c oc Hh Ww : Nat) : String × String :=
  let m := Hh*Ww; let Hin := 2*Hh; let Win := 2*Ww
  (conv s!"{p}c1" x s!"%{p}W1" s!"%{p}b1" oc c Hin Win Hh Ww 2 ++
   bnPC s!"{p}n1" s!"%{p}c1" s!"%{p}g1" s!"%{p}bt1" oc Hh Ww m ++
   relu s!"{p}r1" s!"%{p}n1" oc Hh Ww ++
   conv s!"{p}c2" s!"%{p}r1" s!"%{p}W2" s!"%{p}b2" oc oc Hh Ww Hh Ww 1 ++
   bnPC s!"{p}n2" s!"%{p}c2" s!"%{p}g2" s!"%{p}bt2" oc Hh Ww m ++
   conv s!"{p}cp" x s!"%{p}Wp" s!"%{p}bp" oc c Hin Win Hh Ww 2 ++
   bnPC s!"{p}np" s!"%{p}cp" s!"%{p}gp" s!"%{p}btp" oc Hh Ww m ++
   addOp s!"{p}a" s!"%{p}n2" s!"%{p}np" oc Hh Ww ++
   relu s!"{p}o" s!"%{p}a" oc Hh Ww, s!"%{p}o")

/-- Downsample block backward. `dy` cotangent of `%{p}o`, `xin` block input [B,c,2Hh,2Ww].
    Result dx `%{p}dx` [B,c,2Hh,2Ww]. -/
private def downBlockBack (p dy xin : String) (c oc Hh Ww : Nat) : String × String :=
  (reluBack s!"{p}da" s!"%{p}a" dy oc Hh Ww ++
   -- main path
   bnBackPC s!"{p}dn2" s!"{p}n2" s!"%{p}da" oc Hh Ww ++
   convBack s!"{p}dc2" s!"%{p}dn2" s!"%{p}W2" oc oc Hh Ww ++
   convWGrad s!"{p}dW2" s!"%{p}r1" s!"%{p}dn2" oc oc Hh Ww ++
   convBiasGrad s!"{p}db2" s!"%{p}dn2" oc Hh Ww ++
   reluBack s!"{p}dr1" s!"%{p}n1" s!"%{p}dc2" oc Hh Ww ++
   bnBackPC s!"{p}dn1" s!"{p}n1" s!"%{p}dr1" oc Hh Ww ++
   convBackStrided s!"{p}dc1" s!"%{p}dn1" s!"%{p}W1" c oc Hh Ww ++
   convWGradStrided s!"{p}dW1" xin s!"%{p}dn1" c oc Hh Ww ++
   convBiasGrad s!"{p}db1" s!"%{p}dn1" oc Hh Ww ++
   -- projection skip
   bnBackPC s!"{p}dnp" s!"{p}np" s!"%{p}da" oc Hh Ww ++
   convBackStrided s!"{p}dcp" s!"%{p}dnp" s!"%{p}Wp" c oc Hh Ww ++
   convWGradStrided s!"{p}dWp" xin s!"%{p}dnp" c oc Hh Ww ++
   convBiasGrad s!"{p}dbp" s!"%{p}dnp" oc Hh Ww ++
   addOp s!"{p}dx" s!"%{p}dc1" s!"%{p}dcp" c (2*Hh) (2*Ww), s!"%{p}dx")

private def sgd (θ dθ ty' : String) : String :=
  s!"    %{θ}l = stablehlo.constant dense<{LR}> : {ty'}\n" ++
  s!"    %{θ}s = stablehlo.multiply {dθ}, %{θ}l : {ty'}\n" ++
  s!"    %{θ}n = stablehlo.subtract %{θ}, %{θ}s : {ty'}\n"

-- ════════════ data-driven full ResNet-34 [3,4,6,3] ════════════

/-- A residual block in forward order. `idB p c hh` = identity block (c ch @ hh×hh);
    `downB p cin c hh` = strided downsample (cin→c, 2hh×2hh → hh×hh). -/
inductive Blk where
  | idB   (p : String) (c hh : Nat)
  | downB (p : String) (cin c hh : Nat)

/-- The 16 basic blocks: stage depths [3,4,6,3], channels 64/128/256/512,
    spatial 16/8/4/2 (stages 2–4 open with a strided downsample). -/
private def blocks : List Blk :=
  [.idB "s1b0" 64 16, .idB "s1b1" 64 16, .idB "s1b2" 64 16,
   .downB "d2" 64 128 8, .idB "s2b0" 128 8, .idB "s2b1" 128 8, .idB "s2b2" 128 8,
   .downB "d3" 128 256 4, .idB "s3b0" 256 4, .idB "s3b1" 256 4, .idB "s3b2" 256 4,
     .idB "s3b3" 256 4, .idB "s3b4" 256 4,
   .downB "d4" 256 512 2, .idB "s4b0" 512 2, .idB "s4b1" 512 2]

private def blkOut : Blk → String
  | .idB p _ _ => s!"%{p}o" | .downB p _ _ _ => s!"%{p}o"
private def blkDx : Blk → String
  | .idB p _ _ => s!"%{p}dx" | .downB p _ _ _ => s!"%{p}dx"

/-- Per-block parameter list as `(name, gradSSA, type)` triples, in func-arg order. -/
private def blkParams : Blk → List (String × String × String)
  | .idB p c _ =>
    [(s!"{p}W1", s!"%{p}dW1", ty [c,c,3,3]), (s!"{p}b1", s!"%{p}db1", ty [c]),
     (s!"{p}g1", s!"%{p}dn1dg", ty [c]), (s!"{p}bt1", s!"%{p}dn1db", ty [c]),
     (s!"{p}W2", s!"%{p}dW2", ty [c,c,3,3]), (s!"{p}b2", s!"%{p}db2", ty [c]),
     (s!"{p}g2", s!"%{p}dn2dg", ty [c]), (s!"{p}bt2", s!"%{p}dn2db", ty [c])]
  | .downB p cin c _ =>
    [(s!"{p}W1", s!"%{p}dW1", ty [c,cin,3,3]), (s!"{p}b1", s!"%{p}db1", ty [c]),
     (s!"{p}g1", s!"%{p}dn1dg", ty [c]), (s!"{p}bt1", s!"%{p}dn1db", ty [c]),
     (s!"{p}W2", s!"%{p}dW2", ty [c,c,3,3]), (s!"{p}b2", s!"%{p}db2", ty [c]),
     (s!"{p}g2", s!"%{p}dn2dg", ty [c]), (s!"{p}bt2", s!"%{p}dn2db", ty [c]),
     (s!"{p}Wp", s!"%{p}dWp", ty [c,cin,3,3]), (s!"{p}bp", s!"%{p}dbp", ty [c]),
     (s!"{p}gp", s!"%{p}dnpdg", ty [c]), (s!"{p}btp", s!"%{p}dnpdb", ty [c])]

/-- Whole net's parameter triples in func-arg order: stem, blocks, dense. -/
private def allParams : List (String × String × String) :=
  [("sW", "%dsW", ty [64,3,3,3]), ("sb", "%dsb", ty [64]),
   ("sg", "%dstndg", ty [64]), ("sbt", "%dstndb", ty [64])]
  ++ (blocks.map blkParams).flatten
  ++ [("Wd", "%dWd", ty [512,10]), ("bd", "%dbd", ty [10])]

private def trainStep : String := Id.run do
  -- ── forward: stem → maxpool → blocks → GAP(2×2) → dense(512→10) ──
  let mut fwd := "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n"
    ++ conv "stc" "%x" "%sW" "%sb" 64 3 32 32 32 32 1
    ++ bnPC "stn" "%stc" "%sg" "%sbt" 64 32 32 (32*32)
    ++ relu "str" "%stn" 64 32 32
    ++ maxpool "stp" "%str" 64 32 32
  let mut cur := "%stp"
  let mut io : List (Blk × String) := []      -- (block, its input SSA name)
  for blk in blocks do
    io := io ++ [(blk, cur)]
    match blk with
    | .idB p c hh => fwd := fwd ++ (idBlockFwd p cur c hh hh).1; cur := s!"%{p}o"
    | .downB p cin c hh => fwd := fwd ++ (downBlockFwd p cur cin c hh hh).1; cur := s!"%{p}o"
  fwd := fwd
    ++ s!"    %gaps = stablehlo.reduce({cur} init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,512,2,2]}, tensor<f32>) -> {ty [BS,512]}\n"
    ++ s!"    %gapnf = stablehlo.constant dense<4.0> : {ty [BS,512]}\n"
    ++ s!"    %gap = stablehlo.divide %gaps, %gapnf : {ty [BS,512]}\n"
    ++ s!"    %ld = stablehlo.dot_general %gap, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,512]}, {ty [512,10]}) -> {ty [BS,10]}\n"
    ++ s!"    %ldb = stablehlo.broadcast_in_dim %bd, dims = [1] : ({ty [10]}) -> {ty [BS,10]}\n"
    ++ s!"    %logits = stablehlo.add %ld, %ldb : {ty [BS,10]}\n"
  -- ── loss cotangent dy = softmax(logits) − onehot ──
  let cot :=
    s!"    %le = stablehlo.exponential %logits : {ty [BS,10]}\n"
    ++ s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [BS,10]}, tensor<f32>) -> {ty [BS]}\n"
    ++ s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [BS]}) -> {ty [BS,10]}\n"
    ++ s!"    %lsm = stablehlo.divide %le, %lsb : {ty [BS,10]}\n"
    ++ s!"    %dy = stablehlo.subtract %lsm, %onehot : {ty [BS,10]}\n"
  -- ── backward: dense + GAP, blocks reversed, maxpool, stem ──
  let mut bwd :=
    s!"    %dgap = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [BS,10]}, {ty [512,10]}) -> {ty [BS,512]}\n"
    ++ s!"    %dWd = stablehlo.dot_general %gap, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,512]}, {ty [BS,10]}) -> {ty [512,10]}\n"
    ++ s!"    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,10]}, tensor<f32>) -> {ty [10]}\n"
    ++ s!"    %dgnf = stablehlo.constant dense<4.0> : {ty [BS,512]}\n"
    ++ s!"    %dgs = stablehlo.divide %dgap, %dgnf : {ty [BS,512]}\n"
    ++ s!"    %dgapin = stablehlo.broadcast_in_dim %dgs, dims = [0, 1] : ({ty [BS,512]}) -> {ty [BS,512,2,2]}\n"
  let mut d := "%dgapin"
  for (blk, xin) in io.reverse do
    match blk with
    | .idB p c hh => bwd := bwd ++ (idBlockBack p d xin c hh hh).1; d := s!"%{p}dx"
    | .downB p cin c hh => bwd := bwd ++ (downBlockBack p d xin cin c hh hh).1; d := s!"%{p}dx"
  bwd := bwd
    ++ maxpoolBack "dmp" "%str" d 64 16 16
    ++ reluBack "dstr" "%stn" "%dmp" 64 32 32
    ++ bnBackPC "dstn" "stn" "%dstr" 64 32 32
    ++ convBiasGrad "dsb" "%dstn" 64 32 32
    ++ convWGrad "dsW" "%x" "%dstn" 3 64 32 32
  -- ── SGD + signature/return, all from the single param list ──
  let upd := String.join (allParams.map (fun (nm, gr, t) => sgd nm gr t))
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [BS,3,32,32]) :: allParams.map (fun (nm, _, t) => s!"%{nm}: {t}") ++ ["%onehot: " ++ ty [BS,10]])
  let retTyL := String.intercalate ", " (allParams.map (fun (_, _, t) => t))
  let retVals := String.intercalate ", " (allParams.map (fun (nm, _, _) => s!"%{nm}n"))
  return "module @m {\n" ++ s!"  func.func @resnet34_train_step({argSig}) -> ({retTyL}) " ++ "{\n" ++
    fwd ++ cot ++ bwd ++ upd ++ s!"    return {retVals} : {retTyL}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := trainStep
  IO.println s!"rendered full ResNet-34 train step: {mlir.length} chars, {allParams.length} params"
  IO.FS.createDirAll ".lake/build"
  let path := ".lake/build/resnet34_train_step_v.mlir"
  IO.FS.writeFile path mlir
  let cargs ← ireeCompileArgs path ".lake/build/resnet34_train_step_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println "resnet34 FULL train step iree-compile OK → .lake/build/resnet34_train_step_v.vmfb"

#eval main
