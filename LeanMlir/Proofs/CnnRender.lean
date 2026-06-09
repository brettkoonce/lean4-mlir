import LeanMlir.Proofs.StableHLO

/-! # CNN + CIFAR render half — conv train-step text as a name-threaded render of proven graphs

The peer of `MlpRender.lean` (`mlpTrainStepStructured`) for the Chapter-4 MNIST CNN
(`cnnTrainStepStructured`), the Chapter-5 CIFAR CNN (`cifarTrainStepStructured`), and the
per-channel-BatchNorm CIFAR CNN (`cifarBnTrainStepStructured`).
The MLP render was all-flat, so `pretty`'s flat result names fed the backward/param-grad
templates directly. The CNN forward graph (`cnnFwdGraph`) is *also* rendered all-flat —
each `.flatConvF`/`.maxPoolF` token reshapes flat→NCHW *internally* and back to flat at
its boundary (`emitTok`, `StableHLO.lean`), so the names `pretty` exposes are flat. But
the conv-specific tail ops (`convWGrad`, `selMask4`, `select_and_scatter`) consume the
**4-D NCHW** activations. We bridge that with explicit `reshape` glue in the tail: capture
the flat pre-acts/acts from `pretty (cnnFwdGraph …)` (proof-rendered), reshape the four
the conv tail needs (`%hc1,%ac1,%hc2,%ac2`) plus `%xr` back to `[B,c,H,W]`, then emit the
GPU-validated backward/param-grad/SGD templates (the same op text as `cnnTrainStepText`)
around the captured names. `reshape` is a semantic/GPU no-op (flat and NCHW are the same
buffer), so the rendered module trains identically to the committed hand-written one.

The forward pieces (`flatConvF`/`reluF`/`maxPoolF`/`denseF`, loss cotangent) are denotable
and proven faithful (`flatConvF_faithful`/`reluF_faithful`/`maxPoolF_faithful`/
`denseF_faithful`/`lossCotGraph`-style); the denotation-side close (each conv/dense SGD
output denotes `θ − lr·certified`) is `cnn_render_conv{W,b}_certified` + the M2 dense
bridges (`CnnTrainStep.lean`). See `planning/render_close_handoff.md` §1.
-/

namespace Proofs.StableHLO

open Proofs

/-- Structured CNN train-step renderer (`@cnn_train_step`): forward conv/relu/maxpool/dense
    pre-acts/activations/logits/cotangent from the proven `cnnFwdGraph` pieces
    (name-threaded via `pretty`), then the backward + conv/dense param-grad + SGD ops
    referencing the captured names. The four 4-D activations the conv tail needs
    (`%hc1,%ac1,%hc2,%ac2`) and `%xr` are recovered by explicit flat→NCHW `reshape`s.
    Dim convention matches `cnnFwdGraph`/`cnnFwdModuleV`: `h,w` are the POST-pool spatial
    sizes, the image is `2h × 2w` (so `H := 2h`, `W := 2w`, flattened map `flat := c·h·w`). -/
def cnnTrainStepStructured (B ic c h w d1 nClasses kH kW : Nat) (lr : String)
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) (x : Vec (ic*(2*h)*(2*w))) : String :=
  let H := 2*h; let W := 2*w; let flat := c*h*w
  -- ── op templates (same op text as the GPU-validated `cnnTrainStepText`) ──
  let dg (o a wn cA cB tA tB tO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {wn}, contracting_dims = [{cA}] x [{cB}], precision = [DEFAULT, DEFAULT] : ({tA}, {tB}) -> {tO}\n"
  let reduce0 (o dyk : String) (nn : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dyk} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,nn]}, tensor<f32>) -> {ty [nn]}\n"
  -- relu-backward masks (`select(pre>0, dy, 0)`), 2-D (dense) and 4-D (conv) forms
  let selMask2 (o pre dgrad : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,nn]}, {ty [B,nn]}) -> {tyI1 [B,nn]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,nn]}, {ty [B,nn]}\n"
  let selMask4 (o pre dgrad : String) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,c,H,W]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,c,H,W]}, {ty [B,c,H,W]}) -> {tyI1 [B,c,H,W]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,c,H,W]}, {ty [B,c,H,W]}\n"
  -- conv input-VJP: transpose[1,0,2,3] + reverse[2,3] + convolution (= emitTok convBack)
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let convBack (o dh wn : String) (icc oc : Nat) : String :=
    s!"    {o}t = stablehlo.transpose {wn}, dims = [1, 0, 2, 3] : ({ty [oc,icc,kH,kW]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o}r = stablehlo.reverse {o}t, dims = [2, 3] : {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.convolution({dh}, {o}r)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,oc,H,W]}, {ty [icc,oc,kH,kW]}) -> {ty [B,icc,H,W]}\n"
  -- conv weight grad (transpose trick): a convolution with the batch axis as the
  -- contraction feature, then transpose back to `[oc,icc,kH,kW]`.
  let convWGrad (o inp grad : String) (icc oc : Nat) : String :=
    s!"    {o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [B,icc,H,W]}) -> {ty [icc,B,H,W]}\n" ++
    s!"    {o}dt = stablehlo.transpose {grad}, dims = [1, 0, 2, 3] : ({ty [B,oc,H,W]}) -> {ty [oc,B,H,W]}\n" ++
    s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [icc,B,H,W]}, {ty [oc,B,H,W]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [icc,oc,kH,kW]}) -> {ty [oc,icc,kH,kW]}\n"
  let convBiasGrad (o dh : String) (oc : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dh} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,H,W]}, tensor<f32>) -> {ty [oc]}\n"
  -- maxpool backward (`select_and_scatter`, route dy to the window argmax)
  let scatter (o src dgrad : String) : String :=
    s!"    {o} = \"stablehlo.select_and_scatter\"({src}, {dgrad}, %sc) (" ++ "{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, " ++ "{\n" ++
    "      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %su, %sv : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,c,H,W]}, {ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,H,W]}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  -- flat→NCHW reshape glue: recover a 4-D activation from `pretty`'s flat result name
  let reshape4 (o4 flatN : String) (chans : Nat) : String :=
    s!"    {o4} = stablehlo.reshape {flatN} : ({ty [B, chans*H*W]}) -> {ty [B,chans,H,W]}\n"
  -- ── forward pre-acts/activations from the proven `cnnFwdGraph` pieces; operand
  --    VALUES are placeholders (`pretty` renders names only — never an operand's value). ──
  let zC : Vec (c*(2*h)*(2*w)) := fun _ => 0
  let zPool : Vec (c*h*w) := fun _ => 0
  let zD1 : Vec d1 := fun _ => 0
  let zNC : Vec nClasses := fun _ => 0
  let go : StateM Nat String := do
    let (cHc1, nHc1) ← pretty B (.flatConvF (h := 2*h) (w := 2*w) "%W1" "%b1" W₁ b₁ (.operand "%x" x))
    let (cAc1, nAc1) ← pretty B (.reluF (.operand nHc1 zC))
    let (cHc2, nHc2) ← pretty B (.flatConvF (h := 2*h) (w := 2*w) "%W2" "%b2" W₂ b₂ (.operand nAc1 zC))
    let (cAc2, nAc2) ← pretty B (.reluF (.operand nHc2 zC))
    let (cPool, nPool) ← pretty B (.maxPoolF (c := c) (h := h) (w := w) (.operand nAc2 zC))
    let (cH3, nH3) ← pretty B (denseF "%W3" "%b3" W₃ b₃ (.operand nPool zPool))
    let (cA3, nA3) ← pretty B (.reluF (.operand nH3 zD1))
    let (cH4, nH4) ← pretty B (denseF "%W4" "%b4" W₄ b₄ (.operand nA3 zD1))
    let (cA4, nA4) ← pretty B (.reluF (.operand nH4 zD1))
    let (cLog, nLog) ← pretty B (denseF "%W5" "%b5" W₅ b₅ (.operand nA4 zD1))
    let (cDy, nDy) ← pretty B
      (.sub (.softmaxDiv (.expe (.operand nLog zNC))) (.operand "%onehot" zNC))
    pure <|
      "    // ── forward: conv→relu→conv→relu→maxpool→dense→relu→dense→relu→dense (proof-rendered, flat) ──\n" ++
      cHc1 ++ cAc1 ++ cHc2 ++ cAc2 ++ cPool ++ cH3 ++ cA3 ++ cH4 ++ cA4 ++ cLog ++
      "    // ── loss cotangent dy = softmax(logits) − onehot (lossCotGraph_isCEgrad) ──\n" ++ cDy ++
      "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      "    // ── flat→NCHW glue: the conv tail (selMask4 / scatter / convWGrad) reads 4-D acts ──\n" ++
      s!"    %xr = stablehlo.reshape %x : ({ty [B, ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
      reshape4 "%hc1" nHc1 c ++ reshape4 "%ac1" nAc1 c ++
      reshape4 "%hc2" nHc2 c ++ reshape4 "%ac2" nAc2 c ++
      "    // ── backward: dense (dotOut) + relu masks → reshape → select_and_scatter → convBack ──\n" ++
      dg "%dx5" nDy "%W5" "1" "1" (ty [B,nClasses]) (ty [d1,nClasses]) (ty [B,d1]) ++
      selMask2 "%dy4" nH4 "%dx5" d1 ++
      dg "%dx4" "%dy4" "%W4" "1" "1" (ty [B,d1]) (ty [d1,d1]) (ty [B,d1]) ++
      selMask2 "%dy3" nH3 "%dx4" d1 ++
      dg "%dx3" "%dy3" "%W3" "1" "1" (ty [B,d1]) (ty [flat,d1]) (ty [B,flat]) ++
      s!"    %dpool = stablehlo.reshape %dx3 : ({ty [B,flat]}) -> {ty [B,c,h,w]}\n" ++
      scatter "%dac2" "%ac2" "%dpool" ++
      selMask4 "%dhc2" "%hc2" "%dac2" ++
      convBack "%dac1" "%dhc2" "%W2" c c ++
      selMask4 "%dhc1" "%hc1" "%dac1" ++
      "    // ── param grads: dense W/b (dot_general/reduce); conv dW (transpose trick), db (reduce) ──\n" ++
      dg "%dW5" nA4 nDy "0" "0" (ty [B,d1]) (ty [B,nClasses]) (ty [d1,nClasses]) ++ reduce0 "%db5" nDy nClasses ++
      dg "%dW4" nA3 "%dy4" "0" "0" (ty [B,d1]) (ty [B,d1]) (ty [d1,d1]) ++ reduce0 "%db4" "%dy4" d1 ++
      dg "%dW3" nPool "%dy3" "0" "0" (ty [B,flat]) (ty [B,d1]) (ty [flat,d1]) ++ reduce0 "%db3" "%dy3" d1 ++
      convWGrad "%dW2" "%ac1" "%dhc2" c c ++ convBiasGrad "%db2" "%dhc2" c ++
      convWGrad "%dW1" "%xr" "%dhc1" ic c ++ convBiasGrad "%db1" "%dhc1" c ++
      "    // ── SGD θ' = θ − lr·∇ (all 10 params) ──\n" ++
      sgd "%W1" "%dW1" (ty [c,ic,kH,kW]) ++ sgd "%b1" "%db1" (ty [c]) ++
      sgd "%W2" "%dW2" (ty [c,c,kH,kW]) ++ sgd "%b2" "%db2" (ty [c]) ++
      sgd "%W3" "%dW3" (ty [flat,d1]) ++ sgd "%b3" "%db3" (ty [d1]) ++
      sgd "%W4" "%dW4" (ty [d1,d1]) ++ sgd "%b4" "%db4" (ty [d1]) ++
      sgd "%W5" "%dW5" (ty [d1,nClasses]) ++ sgd "%b5" "%db5" (ty [nClasses])
  let body : String := go.run' 0
  "module @m {\n" ++
  s!"  func.func @cnn_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c,ic,kH,kW]}, %b1: {ty [c]}, %W2: {ty [c,c,kH,kW]}, %b2: {ty [c]}, %W3: {ty [flat,d1]}, %b3: {ty [d1]}, %W4: {ty [d1,d1]}, %b4: {ty [d1]}, %W5: {ty [d1,nClasses]}, %b5: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c,ic,kH,kW]}, {ty [c]}, {ty [c,c,kH,kW]}, {ty [c]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  body ++
  s!"    return %W1n, %b1n, %W2n, %b2n, %W3n, %b3n, %W4n, %b4n, %W5n, %b5n : {ty [c,ic,kH,kW]}, {ty [c]}, {ty [c,c,kH,kW]}, {ty [c]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
  "  }\n}\n"

/-- Structured **CIFAR CNN** train-step renderer (`@cifar_train_step`): the Chapter-5 peer of
    `cnnTrainStepStructured`, a re-parameterization across two conv→conv→pool stages at two
    spatial scales (channels `ic→c1→c1` then `c1→c2→c2`; spatial `H×W → H/2 → H/4`). Forward
    rendered all-flat from the proven `cifarFwdGraph`; the conv tail's 4-D consumers are
    recovered by ten flat→NCHW `reshape`s (`%hc{1..4},%ac{1..4},%pool1,%xr`, at the two
    scales). Tail = `cifarTrainStepText`'s backward/grad/SGD templates wired to the captured
    names. Dim convention matches `cifarFwdGraph`/`cifarFwdModuleV`: `h,w` are the FINAL
    pooled spatial sizes, so the image is `4h × 4w` (`H := 4h`, stage-2 `H2 := 2h`,
    flattened map `flat := c2·h·w`). The close is `cnn_render_conv{W,b}_certified`
    (generic in dims — covers all four conv layers) + the M2 dense bridges. See
    `planning/render_close_handoff.md` §2a. -/
def cifarTrainStepStructured (B ic c1 c2 h w d1 nClasses kH kW : Nat) (lr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) (x : Vec (ic*(2*(2*h))*(2*(2*w)))) : String :=
  let H := 2*(2*h); let W := 2*(2*w)      -- stage-1 full spatial (32)
  let H2 := 2*h; let W2 := 2*w            -- stage-2 spatial after pool1 (16)
  let flat := c2*h*w                       -- final pooled, flattened
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  -- ── op templates (same op text as the GPU-validated `cifarTrainStepText`) ──
  let dg (o a wn cA cB tA tB tO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {wn}, contracting_dims = [{cA}] x [{cB}], precision = [DEFAULT, DEFAULT] : ({tA}, {tB}) -> {tO}\n"
  let reduce0 (o dyk : String) (nn : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dyk} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,nn]}, tensor<f32>) -> {ty [nn]}\n"
  let selMask2 (o pre dgrad : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,nn]}, {ty [B,nn]}) -> {tyI1 [B,nn]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,nn]}, {ty [B,nn]}\n"
  let selMask4 (o pre dgrad : String) (C Hh Ww : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,C,Hh,Ww]}, {ty [B,C,Hh,Ww]}) -> {tyI1 [B,C,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,C,Hh,Ww]}, {ty [B,C,Hh,Ww]}\n"
  let convBack (o dh wn : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}t = stablehlo.transpose {wn}, dims = [1, 0, 2, 3] : ({ty [oc,icc,kH,kW]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o}r = stablehlo.reverse {o}t, dims = [2, 3] : {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.convolution({dh}, {o}r)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,oc,Hh,Ww]}, {ty [icc,oc,kH,kW]}) -> {ty [B,icc,Hh,Ww]}\n"
  let convWGrad (o inp grad : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [B,icc,Hh,Ww]}) -> {ty [icc,B,Hh,Ww]}\n" ++
    s!"    {o}dt = stablehlo.transpose {grad}, dims = [1, 0, 2, 3] : ({ty [B,oc,Hh,Ww]}) -> {ty [oc,B,Hh,Ww]}\n" ++
    s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [icc,B,Hh,Ww]}, {ty [oc,B,Hh,Ww]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [icc,oc,kH,kW]}) -> {ty [oc,icc,kH,kW]}\n"
  let convBiasGrad (o dh : String) (oc Hh Ww : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dh} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"
  let scatter (o src dgrad : String) (C Hh Ww : Nat) : String :=
    s!"    {o} = \"stablehlo.select_and_scatter\"({src}, {dgrad}, %sc) (" ++ "{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, " ++ "{\n" ++
    "      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %su, %sv : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, {ty [B,C,Hh/2,Ww/2]}, tensor<f32>) -> {ty [B,C,Hh,Ww]}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  let reshape4 (o4 flatN : String) (C Hh Ww : Nat) : String :=
    s!"    {o4} = stablehlo.reshape {flatN} : ({ty [B, C*Hh*Ww]}) -> {ty [B,C,Hh,Ww]}\n"
  -- ── forward pieces from the proven `cifarFwdGraph` (placeholder operand values) ──
  let zC1full : Vec (c1*(2*(2*h))*(2*(2*w))) := fun _ => 0
  let zC1half : Vec (c1*(2*h)*(2*w)) := fun _ => 0
  let zC2half : Vec (c2*(2*h)*(2*w)) := fun _ => 0
  let zFlat : Vec (c2*h*w) := fun _ => 0
  let zD1 : Vec d1 := fun _ => 0
  let zNC : Vec nClasses := fun _ => 0
  let go : StateM Nat String := do
    let (cHc1, nHc1) ← pretty B (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W1" "%b1" W₁ b₁ (.operand "%x" x))
    let (cAc1, nAc1) ← pretty B (.reluF (.operand nHc1 zC1full))
    let (cHc2, nHc2) ← pretty B (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W2" "%b2" W₂ b₂ (.operand nAc1 zC1full))
    let (cAc2, nAc2) ← pretty B (.reluF (.operand nHc2 zC1full))
    let (cP1, nPool1) ← pretty B (.maxPoolF (c := c1) (h := 2*h) (w := 2*w) (.operand nAc2 zC1full))
    let (cHc3, nHc3) ← pretty B (.flatConvF (h := 2*h) (w := 2*w) "%W3" "%b3" W₃ b₃ (.operand nPool1 zC1half))
    let (cAc3, nAc3) ← pretty B (.reluF (.operand nHc3 zC2half))
    let (cHc4, nHc4) ← pretty B (.flatConvF (h := 2*h) (w := 2*w) "%W4" "%b4" W₄ b₄ (.operand nAc3 zC2half))
    let (cAc4, nAc4) ← pretty B (.reluF (.operand nHc4 zC2half))
    let (cP2, nPool2) ← pretty B (.maxPoolF (c := c2) (h := h) (w := w) (.operand nAc4 zC2half))
    let (cH5, nH5) ← pretty B (denseF "%W5" "%b5" W₅ b₅ (.operand nPool2 zFlat))
    let (cA5, nA5) ← pretty B (.reluF (.operand nH5 zD1))
    let (cH6, nH6) ← pretty B (denseF "%W6" "%b6" W₆ b₆ (.operand nA5 zD1))
    let (cA6, nA6) ← pretty B (.reluF (.operand nH6 zD1))
    let (cLog, nLog) ← pretty B (denseF "%W7" "%b7" W₇ b₇ (.operand nA6 zD1))
    let (cDy, nDy) ← pretty B
      (.sub (.softmaxDiv (.expe (.operand nLog zNC))) (.operand "%onehot" zNC))
    pure <|
      "    // ── forward: (conv→relu)×2→pool →(conv→relu)×2→pool →flatten→(dense→relu)×2→dense (proof-rendered, flat) ──\n" ++
      cHc1 ++ cAc1 ++ cHc2 ++ cAc2 ++ cP1 ++ cHc3 ++ cAc3 ++ cHc4 ++ cAc4 ++ cP2 ++ cH5 ++ cA5 ++ cH6 ++ cA6 ++ cLog ++
      "    // ── loss cotangent dy = softmax(logits) − onehot (lossCotGraph_isCEgrad) ──\n" ++ cDy ++
      "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      "    // ── flat→NCHW glue: the conv tail reads 4-D acts at both spatial scales ──\n" ++
      s!"    %xr = stablehlo.reshape %x : ({ty [B, ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
      reshape4 "%hc1" nHc1 c1 H W ++ reshape4 "%ac1" nAc1 c1 H W ++
      reshape4 "%hc2" nHc2 c1 H W ++ reshape4 "%ac2" nAc2 c1 H W ++
      reshape4 "%pool1" nPool1 c1 H2 W2 ++
      reshape4 "%hc3" nHc3 c2 H2 W2 ++ reshape4 "%ac3" nAc3 c2 H2 W2 ++
      reshape4 "%hc4" nHc4 c2 H2 W2 ++ reshape4 "%ac4" nAc4 c2 H2 W2 ++
      "    // ── backward: dense (dotOut)+relu masks → scatter → convBack, twice through ──\n" ++
      dg "%dx7" nDy "%W7" "1" "1" (ty [B,nClasses]) (ty [d1,nClasses]) (ty [B,d1]) ++
      selMask2 "%dy6" nH6 "%dx7" d1 ++
      dg "%dx6" "%dy6" "%W6" "1" "1" (ty [B,d1]) (ty [d1,d1]) (ty [B,d1]) ++
      selMask2 "%dy5" nH5 "%dx6" d1 ++
      dg "%dx5" "%dy5" "%W5" "1" "1" (ty [B,d1]) (ty [flat,d1]) (ty [B,flat]) ++
      s!"    %dpool2 = stablehlo.reshape %dx5 : ({ty [B,flat]}) -> {ty [B,c2,h,w]}\n" ++
      scatter "%dac4" "%ac4" "%dpool2" c2 H2 W2 ++
      selMask4 "%dhc4" "%hc4" "%dac4" c2 H2 W2 ++
      convBack "%dac3" "%dhc4" "%W4" c2 c2 H2 W2 ++
      selMask4 "%dhc3" "%hc3" "%dac3" c2 H2 W2 ++
      convBack "%dpool1" "%dhc3" "%W3" c1 c2 H2 W2 ++
      scatter "%dac2" "%ac2" "%dpool1" c1 H W ++
      selMask4 "%dhc2" "%hc2" "%dac2" c1 H W ++
      convBack "%dac1" "%dhc2" "%W2" c1 c1 H W ++
      selMask4 "%dhc1" "%hc1" "%dac1" c1 H W ++
      "    // ── param grads: dense W/b (dot_general/reduce); conv dW (transpose trick), db (reduce) ──\n" ++
      dg "%dW7" nA6 nDy "0" "0" (ty [B,d1]) (ty [B,nClasses]) (ty [d1,nClasses]) ++ reduce0 "%db7" nDy nClasses ++
      dg "%dW6" nA5 "%dy6" "0" "0" (ty [B,d1]) (ty [B,d1]) (ty [d1,d1]) ++ reduce0 "%db6" "%dy6" d1 ++
      dg "%dW5" nPool2 "%dy5" "0" "0" (ty [B,flat]) (ty [B,d1]) (ty [flat,d1]) ++ reduce0 "%db5" "%dy5" d1 ++
      convWGrad "%dW4" "%ac3" "%dhc4" c2 c2 H2 W2 ++ convBiasGrad "%db4" "%dhc4" c2 H2 W2 ++
      convWGrad "%dW3" "%pool1" "%dhc3" c1 c2 H2 W2 ++ convBiasGrad "%db3" "%dhc3" c2 H2 W2 ++
      convWGrad "%dW2" "%ac1" "%dhc2" c1 c1 H W ++ convBiasGrad "%db2" "%dhc2" c1 H W ++
      convWGrad "%dW1" "%xr" "%dhc1" ic c1 H W ++ convBiasGrad "%db1" "%dhc1" c1 H W ++
      "    // ── SGD θ' = θ − lr·∇ (all 14 params) ──\n" ++
      sgd "%W1" "%dW1" (ty [c1,ic,kH,kW]) ++ sgd "%b1" "%db1" (ty [c1]) ++
      sgd "%W2" "%dW2" (ty [c1,c1,kH,kW]) ++ sgd "%b2" "%db2" (ty [c1]) ++
      sgd "%W3" "%dW3" (ty [c2,c1,kH,kW]) ++ sgd "%b3" "%db3" (ty [c2]) ++
      sgd "%W4" "%dW4" (ty [c2,c2,kH,kW]) ++ sgd "%b4" "%db4" (ty [c2]) ++
      sgd "%W5" "%dW5" (ty [flat,d1]) ++ sgd "%b5" "%db5" (ty [d1]) ++
      sgd "%W6" "%dW6" (ty [d1,d1]) ++ sgd "%b6" "%db6" (ty [d1]) ++
      sgd "%W7" "%dW7" (ty [d1,nClasses]) ++ sgd "%b7" "%db7" (ty [nClasses])
  let body : String := go.run' 0
  "module @m {\n" ++
  s!"  func.func @cifar_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [flat,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  body ++
  s!"    return %W1n, %b1n, %W2n, %b2n, %W3n, %b3n, %W4n, %b4n, %W5n, %b5n, %W6n, %b6n, %W7n, %b7n : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
  "  }\n}\n"

/-- Structured **per-channel-BatchNorm CIFAR CNN** train-step renderer (`@cifar_bn_train_step`):
    the BN peer of `cifarTrainStepStructured` (conv→BN→relu ×4, 2 pools, 3 dense; 22 params
    incl. per-channel γ/β). Both the BN **forward** (`bnPerChannelF`, via `cifarBnFwdGraph`)
    and the BN input-grad **backward** (`bnPerChannelBack` token, which recomputes x̂/istd from
    the saved conv-output input — faithful under `0<ε`) are PROOF-RENDERED through `pretty`;
    only the BN parameter grads dγ=Σ_{b,h,w} dy·x̂, dβ=Σ_{b,h,w} dy are hand-emitted (with an
    x̂ recompute, since the token doesn't expose x̂). The conv/dense/relu forward, the conv
    input-grad / weight-grad / bias-grad tail, and the six flat→NCHW glue reshapes are exactly
    as in `cifarTrainStepStructured`. Text differs from the committed `cifarBnTrainStepText`
    (the proven tokens use a `[B,oc,h,w]` reduce-`[2,3]` recompute layout vs the committed
    `[B,C,S]` save-x̂ layout), so it trains EQUIVALENTLY (not bit-identically). The close adds
    the BN dγ/dβ bridges (the BN analogue of `bias_grad_bridge`, under `0<ε`); the conv/dense
    closes and the BN input-grad are already proven. See `planning/render_close_handoff.md` §2b. -/
def cifarBnTrainStepStructured (B ic c1 c2 h w d1 nClasses kH kW : Nat) (epsStr lr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) (x : Vec (ic*(2*(2*h))*(2*(2*w)))) : String :=
  let H := 2*(2*h); let W := 2*(2*w)      -- stage-1 full spatial (32)
  let H2 := 2*h; let W2 := 2*w            -- stage-2 spatial after pool1 (16)
  let flat := c2*h*w
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  -- ── tail op templates (same op text as `cifarBnTrainStepText`) ──
  let dg (o a wn cA cB tA tB tO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {wn}, contracting_dims = [{cA}] x [{cB}], precision = [DEFAULT, DEFAULT] : ({tA}, {tB}) -> {tO}\n"
  let reduce0 (o dyk : String) (nn : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dyk} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,nn]}, tensor<f32>) -> {ty [nn]}\n"
  let selMask2 (o pre dgrad : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,nn]}, {ty [B,nn]}) -> {tyI1 [B,nn]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,nn]}, {ty [B,nn]}\n"
  let convBack (o dh wn : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}t = stablehlo.transpose {wn}, dims = [1, 0, 2, 3] : ({ty [oc,icc,kH,kW]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o}r = stablehlo.reverse {o}t, dims = [2, 3] : {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.convolution({dh}, {o}r)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,oc,Hh,Ww]}, {ty [icc,oc,kH,kW]}) -> {ty [B,icc,Hh,Ww]}\n"
  let convWGrad (o inp grad : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [B,icc,Hh,Ww]}) -> {ty [icc,B,Hh,Ww]}\n" ++
    s!"    {o}dt = stablehlo.transpose {grad}, dims = [1, 0, 2, 3] : ({ty [B,oc,Hh,Ww]}) -> {ty [oc,B,Hh,Ww]}\n" ++
    s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [icc,B,Hh,Ww]}, {ty [oc,B,Hh,Ww]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [icc,oc,kH,kW]}) -> {ty [oc,icc,kH,kW]}\n"
  let convBiasGrad (o dh : String) (oc Hh Ww : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dh} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"
  let scatter (o src dgrad : String) (C Hh Ww : Nat) : String :=
    s!"    {o} = \"stablehlo.select_and_scatter\"({src}, {dgrad}, %sc) (" ++ "{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, " ++ "{\n" ++
    "      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %su, %sv : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, {ty [B,C,Hh/2,Ww/2]}, tensor<f32>) -> {ty [B,C,Hh,Ww]}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  let reshape4 (o4 flatN : String) (C Hh Ww : Nat) : String :=
    s!"    {o4} = stablehlo.reshape {flatN} : ({ty [B, C*Hh*Ww]}) -> {ty [B,C,Hh,Ww]}\n"
  let rsToFlat (o flatN : String) (C Hh Ww : Nat) : String :=
    s!"    {o} = stablehlo.reshape {flatN} : ({ty [B,C,Hh,Ww]}) -> {ty [B, C*Hh*Ww]}\n"
  -- hand-emitted BN per-channel param grads dγ_c=Σ_{b,h,w} dy·x̂, dβ_c=Σ_{b,h,w} dy.
  -- Recompute x̂ from `convOut` (the saved BN input = conv output, flat), reduce over [0,2,3].
  let bnParamGradPC (dgr dbe convOut dyf : String) (C Hh Ww : Nat) : String :=
    let nf := Hh * Ww
    s!"    {dgr}_xr = stablehlo.reshape {convOut} : ({ty [B, C*Hh*Ww]}) -> {ty [B,C,Hh,Ww]}\n" ++
    s!"    {dgr}_nf = stablehlo.constant dense<{nf}.0> : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {dgr}_ep = stablehlo.constant dense<{epsStr}> : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {dgr}_smr = stablehlo.reduce({dgr}_xr init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {dgr}_sm = stablehlo.broadcast_in_dim {dgr}_smr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,Hh,Ww]}\n" ++
    s!"    {dgr}_mu = stablehlo.divide {dgr}_sm, {dgr}_nf : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {dgr}_xc = stablehlo.subtract {dgr}_xr, {dgr}_mu : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {dgr}_sq = stablehlo.multiply {dgr}_xc, {dgr}_xc : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {dgr}_vsr = stablehlo.reduce({dgr}_sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {dgr}_vs = stablehlo.broadcast_in_dim {dgr}_vsr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,Hh,Ww]}\n" ++
    s!"    {dgr}_var = stablehlo.divide {dgr}_vs, {dgr}_nf : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {dgr}_ve = stablehlo.add {dgr}_var, {dgr}_ep : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {dgr}_istd = stablehlo.rsqrt {dgr}_ve : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {dgr}_xhat = stablehlo.multiply {dgr}_xc, {dgr}_istd : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {dgr}_dyr = stablehlo.reshape {dyf} : ({ty [B, C*Hh*Ww]}) -> {ty [B,C,Hh,Ww]}\n" ++
    s!"    {dgr}_p = stablehlo.multiply {dgr}_dyr, {dgr}_xhat : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {dgr} = stablehlo.reduce({dgr}_p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [C]}\n" ++
    s!"    {dbe} = stablehlo.reduce({dgr}_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [C]}\n"
  -- placeholder operand/param values (pretty renders names only — values are irrelevant)
  let zF1 : Vec (c1*(2*(2*h))*(2*(2*w))) := fun _ => 0
  let zP1 : Vec (c1*(2*h)*(2*w)) := fun _ => 0
  let zF2 : Vec (c2*(2*h)*(2*w)) := fun _ => 0
  let zFlat : Vec (c2*h*w) := fun _ => 0
  let zD1 : Vec d1 := fun _ => 0
  let zNC : Vec nClasses := fun _ => 0
  let zVc1 : Vec c1 := fun _ => 0
  let zVc2 : Vec c2 := fun _ => 0
  let go : StateM Nat String := do
    -- ═══ forward (proof-rendered, incl. BN forward) ═══
    let (cHc1, nHc1) ← pretty B (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W1" "%b1" W₁ b₁ (.operand "%x" x))
    let (cBn1, nBn1) ← pretty B (.bnPerChannelF (oc := c1) (h := 2*(2*h)) (w := 2*(2*w)) "%g1" "%bt1" epsStr 0 zVc1 zVc1 (.operand nHc1 zF1))
    let (cAc1, nAc1) ← pretty B (.reluF (.operand nBn1 zF1))
    let (cHc2, nHc2) ← pretty B (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W2" "%b2" W₂ b₂ (.operand nAc1 zF1))
    let (cBn2, nBn2) ← pretty B (.bnPerChannelF (oc := c1) (h := 2*(2*h)) (w := 2*(2*w)) "%g2" "%bt2" epsStr 0 zVc1 zVc1 (.operand nHc2 zF1))
    let (cAc2, nAc2) ← pretty B (.reluF (.operand nBn2 zF1))
    let (cP1, nPool1) ← pretty B (.maxPoolF (c := c1) (h := 2*h) (w := 2*w) (.operand nAc2 zF1))
    let (cHc3, nHc3) ← pretty B (.flatConvF (h := 2*h) (w := 2*w) "%W3" "%b3" W₃ b₃ (.operand nPool1 zP1))
    let (cBn3, nBn3) ← pretty B (.bnPerChannelF (oc := c2) (h := 2*h) (w := 2*w) "%g3" "%bt3" epsStr 0 zVc2 zVc2 (.operand nHc3 zF2))
    let (cAc3, nAc3) ← pretty B (.reluF (.operand nBn3 zF2))
    let (cHc4, nHc4) ← pretty B (.flatConvF (h := 2*h) (w := 2*w) "%W4" "%b4" W₄ b₄ (.operand nAc3 zF2))
    let (cBn4, nBn4) ← pretty B (.bnPerChannelF (oc := c2) (h := 2*h) (w := 2*w) "%g4" "%bt4" epsStr 0 zVc2 zVc2 (.operand nHc4 zF2))
    let (cAc4, nAc4) ← pretty B (.reluF (.operand nBn4 zF2))
    let (cP2, nPool2) ← pretty B (.maxPoolF (c := c2) (h := h) (w := w) (.operand nAc4 zF2))
    let (cH5, nH5) ← pretty B (denseF "%W5" "%b5" W₅ b₅ (.operand nPool2 zFlat))
    let (cA5, nA5) ← pretty B (.reluF (.operand nH5 zD1))
    let (cH6, nH6) ← pretty B (denseF "%W6" "%b6" W₆ b₆ (.operand nA5 zD1))
    let (cA6, nA6) ← pretty B (.reluF (.operand nH6 zD1))
    let (cLog, nLog) ← pretty B (denseF "%W7" "%b7" W₇ b₇ (.operand nA6 zD1))
    let (cDy, nDy) ← pretty B
      (.sub (.softmaxDiv (.expe (.operand nLog zNC))) (.operand "%onehot" zNC))
    -- ═══ BN input-grad pieces (proof-rendered via the bnPerChannelBack token) ═══
    -- each recomputes x̂/istd from the saved conv-output input (nHc{i}); operand is the
    -- relu-back cotangent %dbn{i} (a tail name, defined before the spliced text below).
    let (cBack4, nDhc4f) ← pretty B (.bnPerChannelBack (oc := c2) (h := 2*h) (w := 2*w) "%g4" nHc4 epsStr 0 zVc2 zF2 (.operand "%dbn4" zF2))
    let (cBack3, nDhc3f) ← pretty B (.bnPerChannelBack (oc := c2) (h := 2*h) (w := 2*w) "%g3" nHc3 epsStr 0 zVc2 zF2 (.operand "%dbn3" zF2))
    let (cBack2, nDhc2f) ← pretty B (.bnPerChannelBack (oc := c1) (h := 2*(2*h)) (w := 2*(2*w)) "%g2" nHc2 epsStr 0 zVc1 zF1 (.operand "%dbn2" zF1))
    let (cBack1, nDhc1f) ← pretty B (.bnPerChannelBack (oc := c1) (h := 2*(2*h)) (w := 2*(2*w)) "%g1" nHc1 epsStr 0 zVc1 zF1 (.operand "%dbn1" zF1))
    pure <|
      "    // ── forward: (conv→BN→relu)×2→pool ×2 →flatten→(dense→relu)×2→dense (proof-rendered, BN incl.) ──\n" ++
      cHc1 ++ cBn1 ++ cAc1 ++ cHc2 ++ cBn2 ++ cAc2 ++ cP1 ++
      cHc3 ++ cBn3 ++ cAc3 ++ cHc4 ++ cBn4 ++ cAc4 ++ cP2 ++ cH5 ++ cA5 ++ cH6 ++ cA6 ++ cLog ++
      "    // ── loss cotangent dy = softmax(logits) − onehot ──\n" ++ cDy ++
      "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      "    // ── flat→NCHW glue: conv tail (scatter / convWGrad) reads 4-D acts ──\n" ++
      s!"    %xr = stablehlo.reshape %x : ({ty [B, ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
      reshape4 "%ac1" nAc1 c1 H W ++ reshape4 "%ac2" nAc2 c1 H W ++
      reshape4 "%pool1" nPool1 c1 H2 W2 ++ reshape4 "%ac3" nAc3 c2 H2 W2 ++ reshape4 "%ac4" nAc4 c2 H2 W2 ++
      "    // ── backward: dense (dotOut)+relu → scatter → (relu→BN-back→convBack)×block ──\n" ++
      dg "%dx7" nDy "%W7" "1" "1" (ty [B,nClasses]) (ty [d1,nClasses]) (ty [B,d1]) ++
      selMask2 "%dy6" nH6 "%dx7" d1 ++
      dg "%dx6" "%dy6" "%W6" "1" "1" (ty [B,d1]) (ty [d1,d1]) (ty [B,d1]) ++
      selMask2 "%dy5" nH5 "%dx6" d1 ++
      dg "%dx5" "%dy5" "%W5" "1" "1" (ty [B,d1]) (ty [flat,d1]) (ty [B,flat]) ++
      s!"    %dpool2 = stablehlo.reshape %dx5 : ({ty [B,flat]}) -> {ty [B,c2,h,w]}\n" ++
      -- stage-2 block 4: scatter → relu-back → BN-back → conv-back
      scatter "%dac4" "%ac4" "%dpool2" c2 H2 W2 ++ rsToFlat "%dac4f" "%dac4" c2 H2 W2 ++
      selMask2 "%dbn4" nBn4 "%dac4f" (c2*H2*W2) ++
      cBack4 ++ bnParamGradPC "%dg4" "%dbt4" nHc4 "%dbn4" c2 H2 W2 ++
      reshape4 "%dhc4" nDhc4f c2 H2 W2 ++
      convBack "%dac3" "%dhc4" "%W4" c2 c2 H2 W2 ++ rsToFlat "%dac3f" "%dac3" c2 H2 W2 ++
      -- stage-2 block 3
      selMask2 "%dbn3" nBn3 "%dac3f" (c2*H2*W2) ++
      cBack3 ++ bnParamGradPC "%dg3" "%dbt3" nHc3 "%dbn3" c2 H2 W2 ++
      reshape4 "%dhc3" nDhc3f c2 H2 W2 ++
      convBack "%dpool1" "%dhc3" "%W3" c1 c2 H2 W2 ++
      -- stage-1 block 2: scatter → relu-back → BN-back → conv-back
      scatter "%dac2" "%ac2" "%dpool1" c1 H W ++ rsToFlat "%dac2f" "%dac2" c1 H W ++
      selMask2 "%dbn2" nBn2 "%dac2f" (c1*H*W) ++
      cBack2 ++ bnParamGradPC "%dg2" "%dbt2" nHc2 "%dbn2" c1 H W ++
      reshape4 "%dhc2" nDhc2f c1 H W ++
      convBack "%dac1" "%dhc2" "%W2" c1 c1 H W ++ rsToFlat "%dac1f" "%dac1" c1 H W ++
      -- stage-1 block 1
      selMask2 "%dbn1" nBn1 "%dac1f" (c1*H*W) ++
      cBack1 ++ bnParamGradPC "%dg1" "%dbt1" nHc1 "%dbn1" c1 H W ++
      reshape4 "%dhc1" nDhc1f c1 H W ++
      "    // ── param grads: dense W/b; conv dW (transpose trick), db (reduce) ──\n" ++
      dg "%dW7" nA6 nDy "0" "0" (ty [B,d1]) (ty [B,nClasses]) (ty [d1,nClasses]) ++ reduce0 "%db7" nDy nClasses ++
      dg "%dW6" nA5 "%dy6" "0" "0" (ty [B,d1]) (ty [B,d1]) (ty [d1,d1]) ++ reduce0 "%db6" "%dy6" d1 ++
      dg "%dW5" nPool2 "%dy5" "0" "0" (ty [B,flat]) (ty [B,d1]) (ty [flat,d1]) ++ reduce0 "%db5" "%dy5" d1 ++
      convWGrad "%dW4" "%ac3" "%dhc4" c2 c2 H2 W2 ++ convBiasGrad "%db4" "%dhc4" c2 H2 W2 ++
      convWGrad "%dW3" "%pool1" "%dhc3" c1 c2 H2 W2 ++ convBiasGrad "%db3" "%dhc3" c2 H2 W2 ++
      convWGrad "%dW2" "%ac1" "%dhc2" c1 c1 H W ++ convBiasGrad "%db2" "%dhc2" c1 H W ++
      convWGrad "%dW1" "%xr" "%dhc1" ic c1 H W ++ convBiasGrad "%db1" "%dhc1" c1 H W ++
      "    // ── SGD θ' = θ − lr·∇ (all 22 params, incl. per-channel γ/β) ──\n" ++
      sgd "%W1" "%dW1" (ty [c1,ic,kH,kW]) ++ sgd "%b1" "%db1" (ty [c1]) ++ sgd "%g1" "%dg1" (ty [c1]) ++ sgd "%bt1" "%dbt1" (ty [c1]) ++
      sgd "%W2" "%dW2" (ty [c1,c1,kH,kW]) ++ sgd "%b2" "%db2" (ty [c1]) ++ sgd "%g2" "%dg2" (ty [c1]) ++ sgd "%bt2" "%dbt2" (ty [c1]) ++
      sgd "%W3" "%dW3" (ty [c2,c1,kH,kW]) ++ sgd "%b3" "%db3" (ty [c2]) ++ sgd "%g3" "%dg3" (ty [c2]) ++ sgd "%bt3" "%dbt3" (ty [c2]) ++
      sgd "%W4" "%dW4" (ty [c2,c2,kH,kW]) ++ sgd "%b4" "%db4" (ty [c2]) ++ sgd "%g4" "%dg4" (ty [c2]) ++ sgd "%bt4" "%dbt4" (ty [c2]) ++
      sgd "%W5" "%dW5" (ty [flat,d1]) ++ sgd "%b5" "%db5" (ty [d1]) ++
      sgd "%W6" "%dW6" (ty [d1,d1]) ++ sgd "%b6" "%db6" (ty [d1]) ++
      sgd "%W7" "%dW7" (ty [d1,nClasses]) ++ sgd "%b7" "%db7" (ty [nClasses])
  let body : String := go.run' 0
  "module @m {\n" ++
  s!"  func.func @cifar_bn_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [flat,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  body ++
  s!"    return %W1n, %b1n, %g1n, %bt1n, %W2n, %b2n, %g2n, %bt2n, %W3n, %b3n, %g3n, %bt3n, %W4n, %b4n, %g4n, %bt4n, %W5n, %b5n, %W6n, %b6n, %W7n, %b7n : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
  "  }\n}\n"

end Proofs.StableHLO
