import LeanMlir.Proofs.StableHLO

/-! # CNN render half — the CNN train-step text as a name-threaded render of proven graphs

The peer of `MlpRender.lean` (`mlpTrainStepStructured`) for the Chapter-4 MNIST CNN.
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

end Proofs.StableHLO
