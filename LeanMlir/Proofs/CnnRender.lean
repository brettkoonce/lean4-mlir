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

/-- **MNIST-CNN train step rendered ENTIRELY from the verified AST.** The peer of
    `mlpTrainStepFaithfulV` for the conv net: like `cnnTrainStepStructured` for the
    forward, but the backward chain (`dotOut`/`selectPos`/`maxPoolBack`/`convBack`)
    and ALL ten parameter SGD updates are now `pretty` of denoted `SHlo` nodes too —
    the dense head via `weightSgd`/`biasSgd`, the conv layers via the new
    `convWeightSgd`/`convBiasSgd` ops. So every emitted line is `pretty(provenNode)`,
    and `CnnFaithfulPoC` proves each output's `den` = the certified loss-descent step.
    Cotangents (`%dy`/`dy4`/`dy3`/`dac2`/`dhc2`/`dac1`/`dhc1`) are rendered once and
    shared as operand leaves; operand/`lr`/weight VALUES are `skel`-erased, so these
    placeholders print identically to the live graphs the `den` theorems use. Dim
    convention matches `cnnTrainStepStructured` (`h,w` post-pool, image `2h×2w`). -/
def cnnTrainStepFaithfulV (B ic c h w d1 nClasses kH kW : Nat) (lrStr : String)
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) (x : Vec (ic*(2*h)*(2*w))) : String :=
  let flat := c*h*w
  let zC : Vec (c*(2*h)*(2*w)) := fun _ => 0
  let zPool : Vec (c*h*w) := fun _ => 0
  let zD1 : Vec d1 := fun _ => 0
  let zNC : Vec nClasses := fun _ => 0
  let zT2 : Tensor3 c (2*h) (2*w) := fun _ _ _ => 0      -- conv-2 input (ac1) placeholder
  let zTx : Tensor3 ic (2*h) (2*w) := fun _ _ _ => 0     -- conv-1 input (image) placeholder
  let act : StateM Nat (String × (String × String × String × String × String × String × String × String × String × String)) := do
    -- ═══ forward (proof-rendered, flat) ═══
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
    -- ═══ backward chain (dense head → maxpool → conv), proof-rendered ═══
    let (cDy4, nDy4) ← pretty B (.selectPos nH4 zD1 (.dotOut "%W5" W₅ (.operand nDy zNC)))
    let (cDy3, nDy3) ← pretty B (.selectPos nH3 zD1 (.dotOut "%W4" W₄ (.operand nDy4 zD1)))
    let (cDx3, nDx3) ← pretty B (.dotOut "%W3" W₃ (.operand nDy3 zD1))
    let (cDac2, nDac2) ← pretty B (.maxPoolBack (c := c) (h := h) (w := w) nAc2 zC (.operand nDx3 zPool))
    let (cDhc2, nDhc2) ← pretty B (.selectPos nHc2 zC (.operand nDac2 zC))
    let (cDac1, nDac1) ← pretty B
      (.convBack (h := 2*h) (w := 2*w) "%W2" W₂ b₂ zC (.operand nDhc2 zC))
    let (cDhc1, nDhc1) ← pretty B (.selectPos nHc1 zC (.operand nDac1 zC))
    -- ═══ param SGD updates: dense head (weightSgd/biasSgd), conv (convWeightSgd/convBiasSgd) ═══
    let (cW5, nW5) ← pretty B (SHlo.weightSgd nA4 "%W5" lrStr zD1 W₅ 0 (.operand nDy zNC))
    let (cb5, nb5) ← pretty B (SHlo.biasSgd "%b5" lrStr zNC 0 (.operand nDy zNC))
    let (cW4, nW4) ← pretty B (SHlo.weightSgd nA3 "%W4" lrStr zD1 W₄ 0 (.operand nDy4 zD1))
    let (cb4, nb4) ← pretty B (SHlo.biasSgd "%b4" lrStr zD1 0 (.operand nDy4 zD1))
    let (cW3, nW3) ← pretty B (SHlo.weightSgd nPool "%W3" lrStr zPool W₃ 0 (.operand nDy3 zD1))
    let (cb3, nb3) ← pretty B (SHlo.biasSgd "%b3" lrStr zD1 0 (.operand nDy3 zD1))
    let (cW2g, nW2g) ← pretty B (SHlo.convWeightSgd nAc1 "%W2" lrStr b₂ zT2 W₂ 0 (.operand nDhc2 zC))
    let (cb2g, nb2g) ← pretty B (SHlo.convBiasSgd "%b2" lrStr W₂ zT2 b₂ 0 (.operand nDhc2 zC))
    let (cW1g, nW1g) ← pretty B (SHlo.convWeightSgd "%x" "%W1" lrStr b₁ zTx W₁ 0 (.operand nDhc1 zC))
    let (cb1g, nb1g) ← pretty B (SHlo.convBiasSgd "%b1" lrStr W₁ zTx b₁ 0 (.operand nDhc1 zC))
    pure (cHc1 ++ cAc1 ++ cHc2 ++ cAc2 ++ cPool ++ cH3 ++ cA3 ++ cH4 ++ cA4 ++ cLog ++ cDy ++
            cDy4 ++ cDy3 ++ cDx3 ++ cDac2 ++ cDhc2 ++ cDac1 ++ cDhc1 ++
            cW1g ++ cb1g ++ cW2g ++ cb2g ++ cW3 ++ cb3 ++ cW4 ++ cb4 ++ cW5 ++ cb5,
          nW1g, nb1g, nW2g, nb2g, nW3, nb3, nW4, nb4, nW5, nb5)
  let (body, n1W, n1b, n2W, n2b, n3W, n3b, n4W, n4b, n5W, n5b) := act.run' 0
  "module @m {\n" ++
  s!"  func.func @cnn_train_step(%x: {ty [B,ic*(2*h)*(2*w)]}, %W1: {ty [c,ic,kH,kW]}, %b1: {ty [c]}, %W2: {ty [c,c,kH,kW]}, %b2: {ty [c]}, %W3: {ty [flat,d1]}, %b3: {ty [d1]}, %W4: {ty [d1,d1]}, %b4: {ty [d1]}, %W5: {ty [d1,nClasses]}, %b5: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c,ic,kH,kW]}, {ty [c]}, {ty [c,c,kH,kW]}, {ty [c]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  "    // ── cnn train step: every line is pretty(verified AST node) ──\n" ++
  body ++
  s!"    return {n1W}, {n1b}, {n2W}, {n2b}, {n3W}, {n3b}, {n4W}, {n4b}, {n5W}, {n5b} : {ty [c,ic,kH,kW]}, {ty [c]}, {ty [c,c,kH,kW]}, {ty [c]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
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

/-- **CIFAR-CNN (Chapter 5, no-BN) train step rendered ENTIRELY from the verified AST.**
    The deeper, two-spatial-scale peer of `cnnTrainStepFaithfulV`: like
    `cifarTrainStepStructured` for the forward, but the backward chain
    (`dotOut`/`selectPos`/`maxPoolBack`/`convBack`, twice through) and all 14 parameter
    SGD updates are now `pretty` of denoted `SHlo` nodes — the dense head via
    `weightSgd`/`biasSgd`, the four conv layers via the `convWeightSgd`/`convBiasSgd`
    ops (reused from cnn, NO new ops). Every emitted line is `pretty(provenNode)`, and
    `CifarFaithfulPoC` proves each output's `den` = the certified loss-descent step.
    Dim convention matches `cifarTrainStepStructured` (`h,w` final pooled; image `4h×4w`,
    stage-2 spatial `2h×2w`). -/
def cifarTrainStepFaithfulV (B ic c1 c2 h w d1 nClasses kH kW : Nat) (lrStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) (x : Vec (ic*(2*(2*h))*(2*(2*w)))) : String :=
  let flat := c2*h*w
  let zC1full : Vec (c1*(2*(2*h))*(2*(2*w))) := fun _ => 0
  let zC1half : Vec (c1*(2*h)*(2*w)) := fun _ => 0
  let zC2half : Vec (c2*(2*h)*(2*w)) := fun _ => 0
  let zFlat : Vec (c2*h*w) := fun _ => 0
  let zD1 : Vec d1 := fun _ => 0
  let zNC : Vec nClasses := fun _ => 0
  let zTic4 : Tensor3 ic (2*(2*h)) (2*(2*w)) := fun _ _ _ => 0   -- conv1 input (image)
  let zTc14 : Tensor3 c1 (2*(2*h)) (2*(2*w)) := fun _ _ _ => 0   -- conv2 input (ac1)
  let zTc12 : Tensor3 c1 (2*h) (2*w) := fun _ _ _ => 0           -- conv3 input (pool1)
  let zTc22 : Tensor3 c2 (2*h) (2*w) := fun _ _ _ => 0           -- conv4 input (ac3)
  let act : StateM Nat (String × String × String × String × String × String × String ×
                          String × String × String × String × String × String × String × String) := do
    -- ═══ forward (proof-rendered, flat) ═══
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
    -- ═══ backward chain (dense head → pool2 → conv stage 2 → pool1 → conv stage 1) ═══
    let (cDy6, nDy6) ← pretty B (.selectPos nH6 zD1 (.dotOut "%W7" W₇ (.operand nDy zNC)))
    let (cDy5, nDy5) ← pretty B (.selectPos nH5 zD1 (.dotOut "%W6" W₆ (.operand nDy6 zD1)))
    let (cDx5, nDx5) ← pretty B (.dotOut "%W5" W₅ (.operand nDy5 zD1))
    let (cDac4, nDac4) ← pretty B (.maxPoolBack (c := c2) (h := h) (w := w) nAc4 zC2half (.operand nDx5 zFlat))
    let (cDhc4, nDhc4) ← pretty B (.selectPos nHc4 zC2half (.operand nDac4 zC2half))
    let (cDac3, nDac3) ← pretty B (.convBack (h := 2*h) (w := 2*w) "%W4" W₄ b₄ zC2half (.operand nDhc4 zC2half))
    let (cDhc3, nDhc3) ← pretty B (.selectPos nHc3 zC2half (.operand nDac3 zC2half))
    let (cDpl1, nDpool1) ← pretty B (.convBack (h := 2*h) (w := 2*w) "%W3" W₃ b₃ zC1half (.operand nDhc3 zC2half))
    let (cDac2, nDac2) ← pretty B (.maxPoolBack (c := c1) (h := 2*h) (w := 2*w) nAc2 zC1full (.operand nDpool1 zC1half))
    let (cDhc2, nDhc2) ← pretty B (.selectPos nHc2 zC1full (.operand nDac2 zC1full))
    let (cDac1, nDac1) ← pretty B (.convBack (h := 2*(2*h)) (w := 2*(2*w)) "%W2" W₂ b₂ zC1full (.operand nDhc2 zC1full))
    let (cDhc1, nDhc1) ← pretty B (.selectPos nHc1 zC1full (.operand nDac1 zC1full))
    -- ═══ param SGD updates: conv (convWeightSgd/convBiasSgd) + dense head (weightSgd/biasSgd) ═══
    let (cW1g, nW1g) ← pretty B (SHlo.convWeightSgd "%x" "%W1" lrStr b₁ zTic4 W₁ 0 (.operand nDhc1 zC1full))
    let (cb1g, nb1g) ← pretty B (SHlo.convBiasSgd "%b1" lrStr W₁ zTic4 b₁ 0 (.operand nDhc1 zC1full))
    let (cW2g, nW2g) ← pretty B (SHlo.convWeightSgd nAc1 "%W2" lrStr b₂ zTc14 W₂ 0 (.operand nDhc2 zC1full))
    let (cb2g, nb2g) ← pretty B (SHlo.convBiasSgd "%b2" lrStr W₂ zTc14 b₂ 0 (.operand nDhc2 zC1full))
    let (cW3g, nW3g) ← pretty B (SHlo.convWeightSgd nPool1 "%W3" lrStr b₃ zTc12 W₃ 0 (.operand nDhc3 zC2half))
    let (cb3g, nb3g) ← pretty B (SHlo.convBiasSgd "%b3" lrStr W₃ zTc12 b₃ 0 (.operand nDhc3 zC2half))
    let (cW4g, nW4g) ← pretty B (SHlo.convWeightSgd nAc3 "%W4" lrStr b₄ zTc22 W₄ 0 (.operand nDhc4 zC2half))
    let (cb4g, nb4g) ← pretty B (SHlo.convBiasSgd "%b4" lrStr W₄ zTc22 b₄ 0 (.operand nDhc4 zC2half))
    let (cW5, nW5) ← pretty B (SHlo.weightSgd nPool2 "%W5" lrStr zFlat W₅ 0 (.operand nDy5 zD1))
    let (cb5, nb5) ← pretty B (SHlo.biasSgd "%b5" lrStr zD1 0 (.operand nDy5 zD1))
    let (cW6, nW6) ← pretty B (SHlo.weightSgd nA5 "%W6" lrStr zD1 W₆ 0 (.operand nDy6 zD1))
    let (cb6, nb6) ← pretty B (SHlo.biasSgd "%b6" lrStr zD1 0 (.operand nDy6 zD1))
    let (cW7, nW7) ← pretty B (SHlo.weightSgd nA6 "%W7" lrStr zD1 W₇ 0 (.operand nDy zNC))
    let (cb7, nb7) ← pretty B (SHlo.biasSgd "%b7" lrStr zNC 0 (.operand nDy zNC))
    pure (cHc1 ++ cAc1 ++ cHc2 ++ cAc2 ++ cP1 ++ cHc3 ++ cAc3 ++ cHc4 ++ cAc4 ++ cP2 ++
            cH5 ++ cA5 ++ cH6 ++ cA6 ++ cLog ++ cDy ++
            cDy6 ++ cDy5 ++ cDx5 ++ cDac4 ++ cDhc4 ++ cDac3 ++ cDhc3 ++ cDpl1 ++ cDac2 ++ cDhc2 ++ cDac1 ++ cDhc1 ++
            cW1g ++ cb1g ++ cW2g ++ cb2g ++ cW3g ++ cb3g ++ cW4g ++ cb4g ++
            cW5 ++ cb5 ++ cW6 ++ cb6 ++ cW7 ++ cb7,
          nW1g, nb1g, nW2g, nb2g, nW3g, nb3g, nW4g, nb4g, nW5, nb5, nW6, nb6, nW7, nb7)
  let (body, r1W, r1b, r2W, r2b, r3W, r3b, r4W, r4b, r5W, r5b, r6W, r6b, r7W, r7b) := act.run' 0
  "module @m {\n" ++
  s!"  func.func @cifar_train_step(%x: {ty [B,ic*(2*(2*h))*(2*(2*w))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [flat,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  "    // ── cifar train step: every line is pretty(verified AST node) ──\n" ++
  body ++
  s!"    return {r1W}, {r1b}, {r2W}, {r2b}, {r3W}, {r3b}, {r4W}, {r4b}, {r5W}, {r5b}, {r6W}, {r6b}, {r7W}, {r7b} : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
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

/-- **CIFAR-BN (Chapter 5, per-channel BatchNorm) train step rendered ENTIRELY from the
    verified AST.** The BN peer of `cifarTrainStepFaithfulV` (`conv→BN→relu ×4, 2 pools,
    3 dense`; 22 params). Forward + the BN input-grad backward were already proof-rendered
    (`bnPerChannelF`/`bnPerChannelBack`); now the whole backward chain
    (`dotOut`/`selectPos`/`maxPoolBack`/`convBack`/`bnPerChannelBack`) AND all 22 param SGD
    updates are `pretty` of denoted nodes — conv via `convWeightSgd`/`convBiasSgd`, dense via
    `weightSgd`/`biasSgd`, and the per-channel BN γ/β via the new `bnGammaSgd`/`bnBetaSgd`
    ops. `CifarBnFaithfulPoC` proves each output's `den` = certified. The whole module is
    built inside the `StateM` (so the fresh param-result names are in scope for the `return`).
    Dim convention matches `cifarBnTrainStepStructured`. -/
def cifarBnTrainStepFaithfulV (B ic c1 c2 h w d1 nClasses kH kW : Nat) (epsStr lrStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) (x : Vec (ic*(2*(2*h))*(2*(2*w)))) : String :=
  let flat := c2*h*w
  let zF1 : Vec (c1*(2*(2*h))*(2*(2*w))) := fun _ => 0
  let zP1 : Vec (c1*(2*h)*(2*w)) := fun _ => 0
  let zF2 : Vec (c2*(2*h)*(2*w)) := fun _ => 0
  let zFlat : Vec (c2*h*w) := fun _ => 0
  let zD1 : Vec d1 := fun _ => 0
  let zNC : Vec nClasses := fun _ => 0
  let zVc1 : Vec c1 := fun _ => 0
  let zVc2 : Vec c2 := fun _ => 0
  let zTic4 : Tensor3 ic (2*(2*h)) (2*(2*w)) := fun _ _ _ => 0
  let zTc14 : Tensor3 c1 (2*(2*h)) (2*(2*w)) := fun _ _ _ => 0
  let zTc12 : Tensor3 c1 (2*h) (2*w) := fun _ _ _ => 0
  let zTc22 : Tensor3 c2 (2*h) (2*w) := fun _ _ _ => 0
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
    -- ═══ backward chain: dense head → (scatter → relu-back → BN-back → conv-back) per block ═══
    let (cDy6, nDy6) ← pretty B (.selectPos nH6 zD1 (.dotOut "%W7" W₇ (.operand nDy zNC)))
    let (cDy5, nDy5) ← pretty B (.selectPos nH5 zD1 (.dotOut "%W6" W₆ (.operand nDy6 zD1)))
    let (cDx5, nDx5) ← pretty B (.dotOut "%W5" W₅ (.operand nDy5 zD1))
    -- stage-2 block 4
    let (cDac4, nDac4) ← pretty B (.maxPoolBack (c := c2) (h := h) (w := w) nAc4 zF2 (.operand nDx5 zFlat))
    let (cDbn4, nDbn4) ← pretty B (.selectPos nBn4 zF2 (.operand nDac4 zF2))
    let (cDhc4, nDhc4) ← pretty B (.bnPerChannelBack (oc := c2) (h := 2*h) (w := 2*w) "%g4" nHc4 epsStr 0 zVc2 zF2 (.operand nDbn4 zF2))
    -- stage-2 block 3
    let (cDac3, nDac3) ← pretty B (.convBack (h := 2*h) (w := 2*w) "%W4" W₄ b₄ zF2 (.operand nDhc4 zF2))
    let (cDbn3, nDbn3) ← pretty B (.selectPos nBn3 zF2 (.operand nDac3 zF2))
    let (cDhc3, nDhc3) ← pretty B (.bnPerChannelBack (oc := c2) (h := 2*h) (w := 2*w) "%g3" nHc3 epsStr 0 zVc2 zF2 (.operand nDbn3 zF2))
    let (cDpl1, nDpool1) ← pretty B (.convBack (h := 2*h) (w := 2*w) "%W3" W₃ b₃ zP1 (.operand nDhc3 zF2))
    -- stage-1 block 2
    let (cDac2, nDac2) ← pretty B (.maxPoolBack (c := c1) (h := 2*h) (w := 2*w) nAc2 zF1 (.operand nDpool1 zP1))
    let (cDbn2, nDbn2) ← pretty B (.selectPos nBn2 zF1 (.operand nDac2 zF1))
    let (cDhc2, nDhc2) ← pretty B (.bnPerChannelBack (oc := c1) (h := 2*(2*h)) (w := 2*(2*w)) "%g2" nHc2 epsStr 0 zVc1 zF1 (.operand nDbn2 zF1))
    -- stage-1 block 1
    let (cDac1, nDac1) ← pretty B (.convBack (h := 2*(2*h)) (w := 2*(2*w)) "%W2" W₂ b₂ zF1 (.operand nDhc2 zF1))
    let (cDbn1, nDbn1) ← pretty B (.selectPos nBn1 zF1 (.operand nDac1 zF1))
    let (cDhc1, nDhc1) ← pretty B (.bnPerChannelBack (oc := c1) (h := 2*(2*h)) (w := 2*(2*w)) "%g1" nHc1 epsStr 0 zVc1 zF1 (.operand nDbn1 zF1))
    -- ═══ param SGD updates: conv + BN γ/β + dense ═══
    let (cW1g, nW1g) ← pretty B (SHlo.convWeightSgd "%x" "%W1" lrStr b₁ zTic4 W₁ 0 (.operand nDhc1 zF1))
    let (cb1g, nb1g) ← pretty B (SHlo.convBiasSgd "%b1" lrStr W₁ zTic4 b₁ 0 (.operand nDhc1 zF1))
    let (cg1, ng1) ← pretty B (SHlo.bnGammaSgd "%g1" nHc1 epsStr lrStr 0 zVc1 zF1 0 (.operand nDbn1 zF1))
    let (cbt1, nbt1) ← pretty B (SHlo.bnBetaSgd "%bt1" lrStr zVc1 0 (.operand nDbn1 zF1))
    let (cW2g, nW2g) ← pretty B (SHlo.convWeightSgd nAc1 "%W2" lrStr b₂ zTc14 W₂ 0 (.operand nDhc2 zF1))
    let (cb2g, nb2g) ← pretty B (SHlo.convBiasSgd "%b2" lrStr W₂ zTc14 b₂ 0 (.operand nDhc2 zF1))
    let (cg2, ng2) ← pretty B (SHlo.bnGammaSgd "%g2" nHc2 epsStr lrStr 0 zVc1 zF1 0 (.operand nDbn2 zF1))
    let (cbt2, nbt2) ← pretty B (SHlo.bnBetaSgd "%bt2" lrStr zVc1 0 (.operand nDbn2 zF1))
    let (cW3g, nW3g) ← pretty B (SHlo.convWeightSgd nPool1 "%W3" lrStr b₃ zTc12 W₃ 0 (.operand nDhc3 zF2))
    let (cb3g, nb3g) ← pretty B (SHlo.convBiasSgd "%b3" lrStr W₃ zTc12 b₃ 0 (.operand nDhc3 zF2))
    let (cg3, ng3) ← pretty B (SHlo.bnGammaSgd "%g3" nHc3 epsStr lrStr 0 zVc2 zF2 0 (.operand nDbn3 zF2))
    let (cbt3, nbt3) ← pretty B (SHlo.bnBetaSgd "%bt3" lrStr zVc2 0 (.operand nDbn3 zF2))
    let (cW4g, nW4g) ← pretty B (SHlo.convWeightSgd nAc3 "%W4" lrStr b₄ zTc22 W₄ 0 (.operand nDhc4 zF2))
    let (cb4g, nb4g) ← pretty B (SHlo.convBiasSgd "%b4" lrStr W₄ zTc22 b₄ 0 (.operand nDhc4 zF2))
    let (cg4, ng4) ← pretty B (SHlo.bnGammaSgd "%g4" nHc4 epsStr lrStr 0 zVc2 zF2 0 (.operand nDbn4 zF2))
    let (cbt4, nbt4) ← pretty B (SHlo.bnBetaSgd "%bt4" lrStr zVc2 0 (.operand nDbn4 zF2))
    let (cW5, nW5) ← pretty B (SHlo.weightSgd nPool2 "%W5" lrStr zFlat W₅ 0 (.operand nDy5 zD1))
    let (cb5, nb5) ← pretty B (SHlo.biasSgd "%b5" lrStr zD1 0 (.operand nDy5 zD1))
    let (cW6, nW6) ← pretty B (SHlo.weightSgd nA5 "%W6" lrStr zD1 W₆ 0 (.operand nDy6 zD1))
    let (cb6, nb6) ← pretty B (SHlo.biasSgd "%b6" lrStr zD1 0 (.operand nDy6 zD1))
    let (cW7, nW7) ← pretty B (SHlo.weightSgd nA6 "%W7" lrStr zD1 W₇ 0 (.operand nDy zNC))
    let (cb7, nb7) ← pretty B (SHlo.biasSgd "%b7" lrStr zNC 0 (.operand nDy zNC))
    let body := cHc1 ++ cBn1 ++ cAc1 ++ cHc2 ++ cBn2 ++ cAc2 ++ cP1 ++
      cHc3 ++ cBn3 ++ cAc3 ++ cHc4 ++ cBn4 ++ cAc4 ++ cP2 ++ cH5 ++ cA5 ++ cH6 ++ cA6 ++ cLog ++ cDy ++
      cDy6 ++ cDy5 ++ cDx5 ++ cDac4 ++ cDbn4 ++ cDhc4 ++ cDac3 ++ cDbn3 ++ cDhc3 ++ cDpl1 ++
      cDac2 ++ cDbn2 ++ cDhc2 ++ cDac1 ++ cDbn1 ++ cDhc1 ++
      cW1g ++ cb1g ++ cg1 ++ cbt1 ++ cW2g ++ cb2g ++ cg2 ++ cbt2 ++
      cW3g ++ cb3g ++ cg3 ++ cbt3 ++ cW4g ++ cb4g ++ cg4 ++ cbt4 ++
      cW5 ++ cb5 ++ cW6 ++ cb6 ++ cW7 ++ cb7
    pure <|
      "    // ── cifar-bn train step: every line is pretty(verified AST node) ──\n" ++ body ++
      s!"    return {nW1g}, {nb1g}, {ng1}, {nbt1}, {nW2g}, {nb2g}, {ng2}, {nbt2}, {nW3g}, {nb3g}, {ng3}, {nbt3}, {nW4g}, {nb4g}, {ng4}, {nbt4}, {nW5}, {nb5}, {nW6}, {nb6}, {nW7}, {nb7} : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n"
  let inner : String := go.run' 0
  "module @m {\n" ++
  s!"  func.func @cifar_bn_train_step(%x: {ty [B,ic*(2*(2*h))*(2*(2*w))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [flat,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

set_option maxRecDepth 4000 in
/-- **Deeper 8-conv CIFAR (cifar8, no-BN) train step rendered ENTIRELY from the verified
    AST.** The 4-stage peer of `cifarTrainStepFaithfulV` (`(conv→relu)×2→pool` ×4, 3 dense;
    22 params). Backward chain (`dotOut`/`selectPos`/`maxPoolBack`/`convBack`, four stages)
    and all 22 param SGD ops are `pretty` of denoted nodes — conv via `convWeightSgd`/
    `convBiasSgd`, dense via `weightSgd`/`biasSgd` (NO new ops). `Cifar8FaithfulPoC` proves
    each output's `den` = certified. `h,w` are the final pooled sizes; stage spatials build
    up ×2 per pool (`s4=2h, s3=4h, s2=8h, s1=16h`; image `16h×16w`). -/
def cifar8TrainStepFaithfulV (B ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat) (lrStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses) (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) : String :=
  let s4h := 2*h; let s4w := 2*w
  let s3h := 2*s4h; let s3w := 2*s4w
  let s2h := 2*s3h; let s2w := 2*s3w
  let s1h := 2*s2h; let s1w := 2*s2w
  let flat := c4*h*w
  let zS1c1 : Vec (c1*s1h*s1w) := fun _ => 0
  let zS2c1 : Vec (c1*s2h*s2w) := fun _ => 0
  let zS2c2 : Vec (c2*s2h*s2w) := fun _ => 0
  let zS3c2 : Vec (c2*s3h*s3w) := fun _ => 0
  let zS3c3 : Vec (c3*s3h*s3w) := fun _ => 0
  let zS4c3 : Vec (c3*s4h*s4w) := fun _ => 0
  let zS4c4 : Vec (c4*s4h*s4w) := fun _ => 0
  let zPc4 : Vec (c4*h*w) := fun _ => 0
  let zD1 : Vec d1 := fun _ => 0
  let zNC : Vec nClasses := fun _ => 0
  let zTW1 : Tensor3 ic s1h s1w := fun _ _ _ => 0
  let zTW2 : Tensor3 c1 s1h s1w := fun _ _ _ => 0
  let zTW3 : Tensor3 c1 s2h s2w := fun _ _ _ => 0
  let zTW4 : Tensor3 c2 s2h s2w := fun _ _ _ => 0
  let zTW5 : Tensor3 c2 s3h s3w := fun _ _ _ => 0
  let zTW6 : Tensor3 c3 s3h s3w := fun _ _ _ => 0
  let zTW7 : Tensor3 c3 s4h s4w := fun _ _ _ => 0
  let zTW8 : Tensor3 c4 s4h s4w := fun _ _ _ => 0
  let go : StateM Nat String := do
    -- ═══ forward (proof-rendered, flat): (conv→relu)×2→pool ×4 → (dense→relu)×2→dense ═══
    let (cHc1, nHc1) ← pretty B (.flatConvF (h := s1h) (w := s1w) "%W1" "%b1" W₁ b₁ (.operand "%x" x))
    let (cAc1, nAc1) ← pretty B (.reluF (.operand nHc1 zS1c1))
    let (cHc2, nHc2) ← pretty B (.flatConvF (h := s1h) (w := s1w) "%W2" "%b2" W₂ b₂ (.operand nAc1 zS1c1))
    let (cAc2, nAc2) ← pretty B (.reluF (.operand nHc2 zS1c1))
    let (cP1, nPool1) ← pretty B (.maxPoolF (c := c1) (h := s2h) (w := s2w) (.operand nAc2 zS1c1))
    let (cHc3, nHc3) ← pretty B (.flatConvF (h := s2h) (w := s2w) "%W3" "%b3" W₃ b₃ (.operand nPool1 zS2c1))
    let (cAc3, nAc3) ← pretty B (.reluF (.operand nHc3 zS2c2))
    let (cHc4, nHc4) ← pretty B (.flatConvF (h := s2h) (w := s2w) "%W4" "%b4" W₄ b₄ (.operand nAc3 zS2c2))
    let (cAc4, nAc4) ← pretty B (.reluF (.operand nHc4 zS2c2))
    let (cP2, nPool2) ← pretty B (.maxPoolF (c := c2) (h := s3h) (w := s3w) (.operand nAc4 zS2c2))
    let (cHc5, nHc5) ← pretty B (.flatConvF (h := s3h) (w := s3w) "%W5" "%b5" W₅ b₅ (.operand nPool2 zS3c2))
    let (cAc5, nAc5) ← pretty B (.reluF (.operand nHc5 zS3c3))
    let (cHc6, nHc6) ← pretty B (.flatConvF (h := s3h) (w := s3w) "%W6" "%b6" W₆ b₆ (.operand nAc5 zS3c3))
    let (cAc6, nAc6) ← pretty B (.reluF (.operand nHc6 zS3c3))
    let (cP3, nPool3) ← pretty B (.maxPoolF (c := c3) (h := s4h) (w := s4w) (.operand nAc6 zS3c3))
    let (cHc7, nHc7) ← pretty B (.flatConvF (h := s4h) (w := s4w) "%W7" "%b7" W₇ b₇ (.operand nPool3 zS4c3))
    let (cAc7, nAc7) ← pretty B (.reluF (.operand nHc7 zS4c4))
    let (cHc8, nHc8) ← pretty B (.flatConvF (h := s4h) (w := s4w) "%W8" "%b8" W₈ b₈ (.operand nAc7 zS4c4))
    let (cAc8, nAc8) ← pretty B (.reluF (.operand nHc8 zS4c4))
    let (cP4, nPool4) ← pretty B (.maxPoolF (c := c4) (h := h) (w := w) (.operand nAc8 zS4c4))
    let (cH9, nH9) ← pretty B (denseF "%W9" "%b9" W₉ b₉ (.operand nPool4 zPc4))
    let (cA9, nA9) ← pretty B (.reluF (.operand nH9 zD1))
    let (cHa, nHa) ← pretty B (denseF "%Wa" "%ba" Wa ba (.operand nA9 zD1))
    let (cAa, nAa) ← pretty B (.reluF (.operand nHa zD1))
    let (cLog, nLog) ← pretty B (denseF "%Wb" "%bb" Wb bb (.operand nAa zD1))
    let (cDy, nDy) ← pretty B
      (.sub (.softmaxDiv (.expe (.operand nLog zNC))) (.operand "%onehot" zNC))
    -- ═══ backward chain: dense head → (scatter → relu-back → conv-back) per stage ═══
    let (cDyA, nDyA) ← pretty B (.selectPos nHa zD1 (.dotOut "%Wb" Wb (.operand nDy zNC)))
    let (cDy9, nDy9) ← pretty B (.selectPos nH9 zD1 (.dotOut "%Wa" Wa (.operand nDyA zD1)))
    let (cDx9, nDx9) ← pretty B (.dotOut "%W9" W₉ (.operand nDy9 zD1))
    -- stage 4
    let (cDac8, nDac8) ← pretty B (.maxPoolBack (c := c4) (h := h) (w := w) nAc8 zS4c4 (.operand nDx9 zPc4))
    let (cDhc8, nDhc8) ← pretty B (.selectPos nHc8 zS4c4 (.operand nDac8 zS4c4))
    let (cDac7, nDac7) ← pretty B (.convBack (h := s4h) (w := s4w) "%W8" W₈ b₈ zS4c4 (.operand nDhc8 zS4c4))
    let (cDhc7, nDhc7) ← pretty B (.selectPos nHc7 zS4c4 (.operand nDac7 zS4c4))
    let (cDpl3, nDpool3) ← pretty B (.convBack (h := s4h) (w := s4w) "%W7" W₇ b₇ zS4c3 (.operand nDhc7 zS4c4))
    -- stage 3
    let (cDac6, nDac6) ← pretty B (.maxPoolBack (c := c3) (h := s4h) (w := s4w) nAc6 zS3c3 (.operand nDpool3 zS4c3))
    let (cDhc6, nDhc6) ← pretty B (.selectPos nHc6 zS3c3 (.operand nDac6 zS3c3))
    let (cDac5, nDac5) ← pretty B (.convBack (h := s3h) (w := s3w) "%W6" W₆ b₆ zS3c3 (.operand nDhc6 zS3c3))
    let (cDhc5, nDhc5) ← pretty B (.selectPos nHc5 zS3c3 (.operand nDac5 zS3c3))
    let (cDpl2, nDpool2) ← pretty B (.convBack (h := s3h) (w := s3w) "%W5" W₅ b₅ zS3c2 (.operand nDhc5 zS3c3))
    -- stage 2
    let (cDac4, nDac4) ← pretty B (.maxPoolBack (c := c2) (h := s3h) (w := s3w) nAc4 zS2c2 (.operand nDpool2 zS3c2))
    let (cDhc4, nDhc4) ← pretty B (.selectPos nHc4 zS2c2 (.operand nDac4 zS2c2))
    let (cDac3, nDac3) ← pretty B (.convBack (h := s2h) (w := s2w) "%W4" W₄ b₄ zS2c2 (.operand nDhc4 zS2c2))
    let (cDhc3, nDhc3) ← pretty B (.selectPos nHc3 zS2c2 (.operand nDac3 zS2c2))
    let (cDpl1, nDpool1) ← pretty B (.convBack (h := s2h) (w := s2w) "%W3" W₃ b₃ zS2c1 (.operand nDhc3 zS2c2))
    -- stage 1
    let (cDac2, nDac2) ← pretty B (.maxPoolBack (c := c1) (h := s2h) (w := s2w) nAc2 zS1c1 (.operand nDpool1 zS2c1))
    let (cDhc2, nDhc2) ← pretty B (.selectPos nHc2 zS1c1 (.operand nDac2 zS1c1))
    let (cDac1, nDac1) ← pretty B (.convBack (h := s1h) (w := s1w) "%W2" W₂ b₂ zS1c1 (.operand nDhc2 zS1c1))
    let (cDhc1, nDhc1) ← pretty B (.selectPos nHc1 zS1c1 (.operand nDac1 zS1c1))
    -- ═══ param SGD updates: conv (×8) + dense (×3) ═══
    let (cW1g, nW1g) ← pretty B (SHlo.convWeightSgd "%x" "%W1" lrStr b₁ zTW1 W₁ 0 (.operand nDhc1 zS1c1))
    let (cb1g, nb1g) ← pretty B (SHlo.convBiasSgd "%b1" lrStr W₁ zTW1 b₁ 0 (.operand nDhc1 zS1c1))
    let (cW2g, nW2g) ← pretty B (SHlo.convWeightSgd nAc1 "%W2" lrStr b₂ zTW2 W₂ 0 (.operand nDhc2 zS1c1))
    let (cb2g, nb2g) ← pretty B (SHlo.convBiasSgd "%b2" lrStr W₂ zTW2 b₂ 0 (.operand nDhc2 zS1c1))
    let (cW3g, nW3g) ← pretty B (SHlo.convWeightSgd nPool1 "%W3" lrStr b₃ zTW3 W₃ 0 (.operand nDhc3 zS2c2))
    let (cb3g, nb3g) ← pretty B (SHlo.convBiasSgd "%b3" lrStr W₃ zTW3 b₃ 0 (.operand nDhc3 zS2c2))
    let (cW4g, nW4g) ← pretty B (SHlo.convWeightSgd nAc3 "%W4" lrStr b₄ zTW4 W₄ 0 (.operand nDhc4 zS2c2))
    let (cb4g, nb4g) ← pretty B (SHlo.convBiasSgd "%b4" lrStr W₄ zTW4 b₄ 0 (.operand nDhc4 zS2c2))
    let (cW5g, nW5g) ← pretty B (SHlo.convWeightSgd nPool2 "%W5" lrStr b₅ zTW5 W₅ 0 (.operand nDhc5 zS3c3))
    let (cb5g, nb5g) ← pretty B (SHlo.convBiasSgd "%b5" lrStr W₅ zTW5 b₅ 0 (.operand nDhc5 zS3c3))
    let (cW6g, nW6g) ← pretty B (SHlo.convWeightSgd nAc5 "%W6" lrStr b₆ zTW6 W₆ 0 (.operand nDhc6 zS3c3))
    let (cb6g, nb6g) ← pretty B (SHlo.convBiasSgd "%b6" lrStr W₆ zTW6 b₆ 0 (.operand nDhc6 zS3c3))
    let (cW7g, nW7g) ← pretty B (SHlo.convWeightSgd nPool3 "%W7" lrStr b₇ zTW7 W₇ 0 (.operand nDhc7 zS4c4))
    let (cb7g, nb7g) ← pretty B (SHlo.convBiasSgd "%b7" lrStr W₇ zTW7 b₇ 0 (.operand nDhc7 zS4c4))
    let (cW8g, nW8g) ← pretty B (SHlo.convWeightSgd nAc7 "%W8" lrStr b₈ zTW8 W₈ 0 (.operand nDhc8 zS4c4))
    let (cb8g, nb8g) ← pretty B (SHlo.convBiasSgd "%b8" lrStr W₈ zTW8 b₈ 0 (.operand nDhc8 zS4c4))
    let (cW9, nW9) ← pretty B (SHlo.weightSgd nPool4 "%W9" lrStr zPc4 W₉ 0 (.operand nDy9 zD1))
    let (cb9, nb9) ← pretty B (SHlo.biasSgd "%b9" lrStr zD1 0 (.operand nDy9 zD1))
    let (cWa, nWa) ← pretty B (SHlo.weightSgd nA9 "%Wa" lrStr zD1 Wa 0 (.operand nDyA zD1))
    let (cba, nba) ← pretty B (SHlo.biasSgd "%ba" lrStr zD1 0 (.operand nDyA zD1))
    let (cWb, nWb) ← pretty B (SHlo.weightSgd nAa "%Wb" lrStr zD1 Wb 0 (.operand nDy zNC))
    let (cbb, nbb) ← pretty B (SHlo.biasSgd "%bb" lrStr zNC 0 (.operand nDy zNC))
    let body := cHc1 ++ cAc1 ++ cHc2 ++ cAc2 ++ cP1 ++ cHc3 ++ cAc3 ++ cHc4 ++ cAc4 ++ cP2 ++
      cHc5 ++ cAc5 ++ cHc6 ++ cAc6 ++ cP3 ++ cHc7 ++ cAc7 ++ cHc8 ++ cAc8 ++ cP4 ++
      cH9 ++ cA9 ++ cHa ++ cAa ++ cLog ++ cDy ++
      cDyA ++ cDy9 ++ cDx9 ++ cDac8 ++ cDhc8 ++ cDac7 ++ cDhc7 ++ cDpl3 ++
      cDac6 ++ cDhc6 ++ cDac5 ++ cDhc5 ++ cDpl2 ++ cDac4 ++ cDhc4 ++ cDac3 ++ cDhc3 ++ cDpl1 ++
      cDac2 ++ cDhc2 ++ cDac1 ++ cDhc1 ++
      cW1g ++ cb1g ++ cW2g ++ cb2g ++ cW3g ++ cb3g ++ cW4g ++ cb4g ++
      cW5g ++ cb5g ++ cW6g ++ cb6g ++ cW7g ++ cb7g ++ cW8g ++ cb8g ++
      cW9 ++ cb9 ++ cWa ++ cba ++ cWb ++ cbb
    pure <|
      "    // ── cifar8 train step: every line is pretty(verified AST node) ──\n" ++ body ++
      s!"    return {nW1g}, {nb1g}, {nW2g}, {nb2g}, {nW3g}, {nb3g}, {nW4g}, {nb4g}, {nW5g}, {nb5g}, {nW6g}, {nb6g}, {nW7g}, {nb7g}, {nW8g}, {nb8g}, {nW9}, {nb9}, {nWa}, {nba}, {nWb}, {nbb} : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c3,c2,kH,kW]}, {ty [c3]}, {ty [c3,c3,kH,kW]}, {ty [c3]}, {ty [c4,c3,kH,kW]}, {ty [c4]}, {ty [c4,c4,kH,kW]}, {ty [c4]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n"
  let inner : String := go.run' 0
  "module @m {\n" ++
  s!"  func.func @cifar8_train_step(%x: {ty [B,ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %W9: {ty [flat,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c3,c2,kH,kW]}, {ty [c3]}, {ty [c3,c3,kH,kW]}, {ty [c3]}, {ty [c4,c3,kH,kW]}, {ty [c4]}, {ty [c4,c4,kH,kW]}, {ty [c4]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

set_option maxRecDepth 8000 in
/-- **Deeper 8-conv CIFAR-BN (cifar8-bn) train step rendered ENTIRELY from the verified
    AST.** The per-channel-BatchNorm peer of `cifar8TrainStepFaithfulV` (`(conv→BN→relu)×2→pool`
    ×4, 3 dense; 38 params). Pure reuse — NO new ops and NO new proof: conv via
    `convWeightSgd`/`convBiasSgd`, BN via `bnGammaSgd`/`bnBetaSgd`, dense via `weightSgd`/
    `biasSgd`; every output's `den` = certified by the existing generic lemmas
    (`CifarPoC.conv{W,B}_den`, `CifarBnPoC.bn{Gamma,Beta}_den`, `Cifar8PoC.dense{W,B}_den`)
    instantiated per layer. Forward + BN-back proof-rendered via `bnPerChannelF`/
    `bnPerChannelBack`. `h,w` final pooled; stage spatials `s4=2h…s1=16h`. -/
def cifar8BnTrainStepFaithfulV (B ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat) (epsStr lrStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses) (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) : String :=
  let s4h := 2*h; let s4w := 2*w
  let s3h := 2*s4h; let s3w := 2*s4w
  let s2h := 2*s3h; let s2w := 2*s3w
  let s1h := 2*s2h; let s1w := 2*s2w
  let flat := c4*h*w
  let zS1c1 : Vec (c1*s1h*s1w) := fun _ => 0
  let zS2c1 : Vec (c1*s2h*s2w) := fun _ => 0
  let zS2c2 : Vec (c2*s2h*s2w) := fun _ => 0
  let zS3c2 : Vec (c2*s3h*s3w) := fun _ => 0
  let zS3c3 : Vec (c3*s3h*s3w) := fun _ => 0
  let zS4c3 : Vec (c3*s4h*s4w) := fun _ => 0
  let zS4c4 : Vec (c4*s4h*s4w) := fun _ => 0
  let zPc4 : Vec (c4*h*w) := fun _ => 0
  let zD1 : Vec d1 := fun _ => 0
  let zNC : Vec nClasses := fun _ => 0
  let zVc1 : Vec c1 := fun _ => 0
  let zVc2 : Vec c2 := fun _ => 0
  let zVc3 : Vec c3 := fun _ => 0
  let zVc4 : Vec c4 := fun _ => 0
  let zTW1 : Tensor3 ic s1h s1w := fun _ _ _ => 0
  let zTW2 : Tensor3 c1 s1h s1w := fun _ _ _ => 0
  let zTW3 : Tensor3 c1 s2h s2w := fun _ _ _ => 0
  let zTW4 : Tensor3 c2 s2h s2w := fun _ _ _ => 0
  let zTW5 : Tensor3 c2 s3h s3w := fun _ _ _ => 0
  let zTW6 : Tensor3 c3 s3h s3w := fun _ _ _ => 0
  let zTW7 : Tensor3 c3 s4h s4w := fun _ _ _ => 0
  let zTW8 : Tensor3 c4 s4h s4w := fun _ _ _ => 0
  let go : StateM Nat String := do
    -- ═══ forward (proof-rendered, incl. BN): (conv→BN→relu)×2→pool ×4 → (dense→relu)×2→dense ═══
    let (cHc1, nHc1) ← pretty B (.flatConvF (h := s1h) (w := s1w) "%W1" "%b1" W₁ b₁ (.operand "%x" x))
    let (cBn1, nBn1) ← pretty B (.bnPerChannelF (oc := c1) (h := s1h) (w := s1w) "%g1" "%bt1" epsStr 0 zVc1 zVc1 (.operand nHc1 zS1c1))
    let (cAc1, nAc1) ← pretty B (.reluF (.operand nBn1 zS1c1))
    let (cHc2, nHc2) ← pretty B (.flatConvF (h := s1h) (w := s1w) "%W2" "%b2" W₂ b₂ (.operand nAc1 zS1c1))
    let (cBn2, nBn2) ← pretty B (.bnPerChannelF (oc := c1) (h := s1h) (w := s1w) "%g2" "%bt2" epsStr 0 zVc1 zVc1 (.operand nHc2 zS1c1))
    let (cAc2, nAc2) ← pretty B (.reluF (.operand nBn2 zS1c1))
    let (cP1, nPool1) ← pretty B (.maxPoolF (c := c1) (h := s2h) (w := s2w) (.operand nAc2 zS1c1))
    let (cHc3, nHc3) ← pretty B (.flatConvF (h := s2h) (w := s2w) "%W3" "%b3" W₃ b₃ (.operand nPool1 zS2c1))
    let (cBn3, nBn3) ← pretty B (.bnPerChannelF (oc := c2) (h := s2h) (w := s2w) "%g3" "%bt3" epsStr 0 zVc2 zVc2 (.operand nHc3 zS2c2))
    let (cAc3, nAc3) ← pretty B (.reluF (.operand nBn3 zS2c2))
    let (cHc4, nHc4) ← pretty B (.flatConvF (h := s2h) (w := s2w) "%W4" "%b4" W₄ b₄ (.operand nAc3 zS2c2))
    let (cBn4, nBn4) ← pretty B (.bnPerChannelF (oc := c2) (h := s2h) (w := s2w) "%g4" "%bt4" epsStr 0 zVc2 zVc2 (.operand nHc4 zS2c2))
    let (cAc4, nAc4) ← pretty B (.reluF (.operand nBn4 zS2c2))
    let (cP2, nPool2) ← pretty B (.maxPoolF (c := c2) (h := s3h) (w := s3w) (.operand nAc4 zS2c2))
    let (cHc5, nHc5) ← pretty B (.flatConvF (h := s3h) (w := s3w) "%W5" "%b5" W₅ b₅ (.operand nPool2 zS3c2))
    let (cBn5, nBn5) ← pretty B (.bnPerChannelF (oc := c3) (h := s3h) (w := s3w) "%g5" "%bt5" epsStr 0 zVc3 zVc3 (.operand nHc5 zS3c3))
    let (cAc5, nAc5) ← pretty B (.reluF (.operand nBn5 zS3c3))
    let (cHc6, nHc6) ← pretty B (.flatConvF (h := s3h) (w := s3w) "%W6" "%b6" W₆ b₆ (.operand nAc5 zS3c3))
    let (cBn6, nBn6) ← pretty B (.bnPerChannelF (oc := c3) (h := s3h) (w := s3w) "%g6" "%bt6" epsStr 0 zVc3 zVc3 (.operand nHc6 zS3c3))
    let (cAc6, nAc6) ← pretty B (.reluF (.operand nBn6 zS3c3))
    let (cP3, nPool3) ← pretty B (.maxPoolF (c := c3) (h := s4h) (w := s4w) (.operand nAc6 zS3c3))
    let (cHc7, nHc7) ← pretty B (.flatConvF (h := s4h) (w := s4w) "%W7" "%b7" W₇ b₇ (.operand nPool3 zS4c3))
    let (cBn7, nBn7) ← pretty B (.bnPerChannelF (oc := c4) (h := s4h) (w := s4w) "%g7" "%bt7" epsStr 0 zVc4 zVc4 (.operand nHc7 zS4c4))
    let (cAc7, nAc7) ← pretty B (.reluF (.operand nBn7 zS4c4))
    let (cHc8, nHc8) ← pretty B (.flatConvF (h := s4h) (w := s4w) "%W8" "%b8" W₈ b₈ (.operand nAc7 zS4c4))
    let (cBn8, nBn8) ← pretty B (.bnPerChannelF (oc := c4) (h := s4h) (w := s4w) "%g8" "%bt8" epsStr 0 zVc4 zVc4 (.operand nHc8 zS4c4))
    let (cAc8, nAc8) ← pretty B (.reluF (.operand nBn8 zS4c4))
    let (cP4, nPool4) ← pretty B (.maxPoolF (c := c4) (h := h) (w := w) (.operand nAc8 zS4c4))
    let (cH9, nH9) ← pretty B (denseF "%W9" "%b9" W₉ b₉ (.operand nPool4 zPc4))
    let (cA9, nA9) ← pretty B (.reluF (.operand nH9 zD1))
    let (cHa, nHa) ← pretty B (denseF "%Wa" "%ba" Wa ba (.operand nA9 zD1))
    let (cAa, nAa) ← pretty B (.reluF (.operand nHa zD1))
    let (cLog, nLog) ← pretty B (denseF "%Wb" "%bb" Wb bb (.operand nAa zD1))
    let (cDy, nDy) ← pretty B
      (.sub (.softmaxDiv (.expe (.operand nLog zNC))) (.operand "%onehot" zNC))
    -- ═══ backward: dense head → (scatter → relu-back → BN-back → conv-back) per block, 4 stages ═══
    let (cDyA, nDyA) ← pretty B (.selectPos nHa zD1 (.dotOut "%Wb" Wb (.operand nDy zNC)))
    let (cDy9, nDy9) ← pretty B (.selectPos nH9 zD1 (.dotOut "%Wa" Wa (.operand nDyA zD1)))
    let (cDx9, nDx9) ← pretty B (.dotOut "%W9" W₉ (.operand nDy9 zD1))
    -- stage 4
    let (cDac8, nDac8) ← pretty B (.maxPoolBack (c := c4) (h := h) (w := w) nAc8 zS4c4 (.operand nDx9 zPc4))
    let (cDbn8, nDbn8) ← pretty B (.selectPos nBn8 zS4c4 (.operand nDac8 zS4c4))
    let (cDhc8, nDhc8) ← pretty B (.bnPerChannelBack (oc := c4) (h := s4h) (w := s4w) "%g8" nHc8 epsStr 0 zVc4 zS4c4 (.operand nDbn8 zS4c4))
    let (cDac7, nDac7) ← pretty B (.convBack (h := s4h) (w := s4w) "%W8" W₈ b₈ zS4c4 (.operand nDhc8 zS4c4))
    let (cDbn7, nDbn7) ← pretty B (.selectPos nBn7 zS4c4 (.operand nDac7 zS4c4))
    let (cDhc7, nDhc7) ← pretty B (.bnPerChannelBack (oc := c4) (h := s4h) (w := s4w) "%g7" nHc7 epsStr 0 zVc4 zS4c4 (.operand nDbn7 zS4c4))
    let (cDpl3, nDpool3) ← pretty B (.convBack (h := s4h) (w := s4w) "%W7" W₇ b₇ zS4c3 (.operand nDhc7 zS4c4))
    -- stage 3
    let (cDac6, nDac6) ← pretty B (.maxPoolBack (c := c3) (h := s4h) (w := s4w) nAc6 zS3c3 (.operand nDpool3 zS4c3))
    let (cDbn6, nDbn6) ← pretty B (.selectPos nBn6 zS3c3 (.operand nDac6 zS3c3))
    let (cDhc6, nDhc6) ← pretty B (.bnPerChannelBack (oc := c3) (h := s3h) (w := s3w) "%g6" nHc6 epsStr 0 zVc3 zS3c3 (.operand nDbn6 zS3c3))
    let (cDac5, nDac5) ← pretty B (.convBack (h := s3h) (w := s3w) "%W6" W₆ b₆ zS3c3 (.operand nDhc6 zS3c3))
    let (cDbn5, nDbn5) ← pretty B (.selectPos nBn5 zS3c3 (.operand nDac5 zS3c3))
    let (cDhc5, nDhc5) ← pretty B (.bnPerChannelBack (oc := c3) (h := s3h) (w := s3w) "%g5" nHc5 epsStr 0 zVc3 zS3c3 (.operand nDbn5 zS3c3))
    let (cDpl2, nDpool2) ← pretty B (.convBack (h := s3h) (w := s3w) "%W5" W₅ b₅ zS3c2 (.operand nDhc5 zS3c3))
    -- stage 2
    let (cDac4, nDac4) ← pretty B (.maxPoolBack (c := c2) (h := s3h) (w := s3w) nAc4 zS2c2 (.operand nDpool2 zS3c2))
    let (cDbn4, nDbn4) ← pretty B (.selectPos nBn4 zS2c2 (.operand nDac4 zS2c2))
    let (cDhc4, nDhc4) ← pretty B (.bnPerChannelBack (oc := c2) (h := s2h) (w := s2w) "%g4" nHc4 epsStr 0 zVc2 zS2c2 (.operand nDbn4 zS2c2))
    let (cDac3, nDac3) ← pretty B (.convBack (h := s2h) (w := s2w) "%W4" W₄ b₄ zS2c2 (.operand nDhc4 zS2c2))
    let (cDbn3, nDbn3) ← pretty B (.selectPos nBn3 zS2c2 (.operand nDac3 zS2c2))
    let (cDhc3, nDhc3) ← pretty B (.bnPerChannelBack (oc := c2) (h := s2h) (w := s2w) "%g3" nHc3 epsStr 0 zVc2 zS2c2 (.operand nDbn3 zS2c2))
    let (cDpl1, nDpool1) ← pretty B (.convBack (h := s2h) (w := s2w) "%W3" W₃ b₃ zS2c1 (.operand nDhc3 zS2c2))
    -- stage 1
    let (cDac2, nDac2) ← pretty B (.maxPoolBack (c := c1) (h := s2h) (w := s2w) nAc2 zS1c1 (.operand nDpool1 zS2c1))
    let (cDbn2, nDbn2) ← pretty B (.selectPos nBn2 zS1c1 (.operand nDac2 zS1c1))
    let (cDhc2, nDhc2) ← pretty B (.bnPerChannelBack (oc := c1) (h := s1h) (w := s1w) "%g2" nHc2 epsStr 0 zVc1 zS1c1 (.operand nDbn2 zS1c1))
    let (cDac1, nDac1) ← pretty B (.convBack (h := s1h) (w := s1w) "%W2" W₂ b₂ zS1c1 (.operand nDhc2 zS1c1))
    let (cDbn1, nDbn1) ← pretty B (.selectPos nBn1 zS1c1 (.operand nDac1 zS1c1))
    let (cDhc1, nDhc1) ← pretty B (.bnPerChannelBack (oc := c1) (h := s1h) (w := s1w) "%g1" nHc1 epsStr 0 zVc1 zS1c1 (.operand nDbn1 zS1c1))
    -- ═══ param SGD updates: per block conv W/b + BN γ/β, then dense ═══
    let (cW1g, nW1g) ← pretty B (SHlo.convWeightSgd "%x" "%W1" lrStr b₁ zTW1 W₁ 0 (.operand nDhc1 zS1c1))
    let (cb1g, nb1g) ← pretty B (SHlo.convBiasSgd "%b1" lrStr W₁ zTW1 b₁ 0 (.operand nDhc1 zS1c1))
    let (cg1, ng1) ← pretty B (SHlo.bnGammaSgd "%g1" nHc1 epsStr lrStr 0 zVc1 zS1c1 0 (.operand nDbn1 zS1c1))
    let (cbt1, nbt1) ← pretty B (SHlo.bnBetaSgd "%bt1" lrStr zVc1 0 (.operand nDbn1 zS1c1))
    let (cW2g, nW2g) ← pretty B (SHlo.convWeightSgd nAc1 "%W2" lrStr b₂ zTW2 W₂ 0 (.operand nDhc2 zS1c1))
    let (cb2g, nb2g) ← pretty B (SHlo.convBiasSgd "%b2" lrStr W₂ zTW2 b₂ 0 (.operand nDhc2 zS1c1))
    let (cg2, ng2) ← pretty B (SHlo.bnGammaSgd "%g2" nHc2 epsStr lrStr 0 zVc1 zS1c1 0 (.operand nDbn2 zS1c1))
    let (cbt2, nbt2) ← pretty B (SHlo.bnBetaSgd "%bt2" lrStr zVc1 0 (.operand nDbn2 zS1c1))
    let (cW3g, nW3g) ← pretty B (SHlo.convWeightSgd nPool1 "%W3" lrStr b₃ zTW3 W₃ 0 (.operand nDhc3 zS2c2))
    let (cb3g, nb3g) ← pretty B (SHlo.convBiasSgd "%b3" lrStr W₃ zTW3 b₃ 0 (.operand nDhc3 zS2c2))
    let (cg3, ng3) ← pretty B (SHlo.bnGammaSgd "%g3" nHc3 epsStr lrStr 0 zVc2 zS2c2 0 (.operand nDbn3 zS2c2))
    let (cbt3, nbt3) ← pretty B (SHlo.bnBetaSgd "%bt3" lrStr zVc2 0 (.operand nDbn3 zS2c2))
    let (cW4g, nW4g) ← pretty B (SHlo.convWeightSgd nAc3 "%W4" lrStr b₄ zTW4 W₄ 0 (.operand nDhc4 zS2c2))
    let (cb4g, nb4g) ← pretty B (SHlo.convBiasSgd "%b4" lrStr W₄ zTW4 b₄ 0 (.operand nDhc4 zS2c2))
    let (cg4, ng4) ← pretty B (SHlo.bnGammaSgd "%g4" nHc4 epsStr lrStr 0 zVc2 zS2c2 0 (.operand nDbn4 zS2c2))
    let (cbt4, nbt4) ← pretty B (SHlo.bnBetaSgd "%bt4" lrStr zVc2 0 (.operand nDbn4 zS2c2))
    let (cW5g, nW5g) ← pretty B (SHlo.convWeightSgd nPool2 "%W5" lrStr b₅ zTW5 W₅ 0 (.operand nDhc5 zS3c3))
    let (cb5g, nb5g) ← pretty B (SHlo.convBiasSgd "%b5" lrStr W₅ zTW5 b₅ 0 (.operand nDhc5 zS3c3))
    let (cg5, ng5) ← pretty B (SHlo.bnGammaSgd "%g5" nHc5 epsStr lrStr 0 zVc3 zS3c3 0 (.operand nDbn5 zS3c3))
    let (cbt5, nbt5) ← pretty B (SHlo.bnBetaSgd "%bt5" lrStr zVc3 0 (.operand nDbn5 zS3c3))
    let (cW6g, nW6g) ← pretty B (SHlo.convWeightSgd nAc5 "%W6" lrStr b₆ zTW6 W₆ 0 (.operand nDhc6 zS3c3))
    let (cb6g, nb6g) ← pretty B (SHlo.convBiasSgd "%b6" lrStr W₆ zTW6 b₆ 0 (.operand nDhc6 zS3c3))
    let (cg6, ng6) ← pretty B (SHlo.bnGammaSgd "%g6" nHc6 epsStr lrStr 0 zVc3 zS3c3 0 (.operand nDbn6 zS3c3))
    let (cbt6, nbt6) ← pretty B (SHlo.bnBetaSgd "%bt6" lrStr zVc3 0 (.operand nDbn6 zS3c3))
    let (cW7g, nW7g) ← pretty B (SHlo.convWeightSgd nPool3 "%W7" lrStr b₇ zTW7 W₇ 0 (.operand nDhc7 zS4c4))
    let (cb7g, nb7g) ← pretty B (SHlo.convBiasSgd "%b7" lrStr W₇ zTW7 b₇ 0 (.operand nDhc7 zS4c4))
    let (cg7, ng7) ← pretty B (SHlo.bnGammaSgd "%g7" nHc7 epsStr lrStr 0 zVc4 zS4c4 0 (.operand nDbn7 zS4c4))
    let (cbt7, nbt7) ← pretty B (SHlo.bnBetaSgd "%bt7" lrStr zVc4 0 (.operand nDbn7 zS4c4))
    let (cW8g, nW8g) ← pretty B (SHlo.convWeightSgd nAc7 "%W8" lrStr b₈ zTW8 W₈ 0 (.operand nDhc8 zS4c4))
    let (cb8g, nb8g) ← pretty B (SHlo.convBiasSgd "%b8" lrStr W₈ zTW8 b₈ 0 (.operand nDhc8 zS4c4))
    let (cg8, ng8) ← pretty B (SHlo.bnGammaSgd "%g8" nHc8 epsStr lrStr 0 zVc4 zS4c4 0 (.operand nDbn8 zS4c4))
    let (cbt8, nbt8) ← pretty B (SHlo.bnBetaSgd "%bt8" lrStr zVc4 0 (.operand nDbn8 zS4c4))
    let (cW9, nW9) ← pretty B (SHlo.weightSgd nPool4 "%W9" lrStr zPc4 W₉ 0 (.operand nDy9 zD1))
    let (cb9, nb9) ← pretty B (SHlo.biasSgd "%b9" lrStr zD1 0 (.operand nDy9 zD1))
    let (cWa, nWa) ← pretty B (SHlo.weightSgd nA9 "%Wa" lrStr zD1 Wa 0 (.operand nDyA zD1))
    let (cba, nba) ← pretty B (SHlo.biasSgd "%ba" lrStr zD1 0 (.operand nDyA zD1))
    let (cWb, nWb) ← pretty B (SHlo.weightSgd nAa "%Wb" lrStr zD1 Wb 0 (.operand nDy zNC))
    let (cbb, nbb) ← pretty B (SHlo.biasSgd "%bb" lrStr zNC 0 (.operand nDy zNC))
    let body := cHc1 ++ cBn1 ++ cAc1 ++ cHc2 ++ cBn2 ++ cAc2 ++ cP1 ++
      cHc3 ++ cBn3 ++ cAc3 ++ cHc4 ++ cBn4 ++ cAc4 ++ cP2 ++
      cHc5 ++ cBn5 ++ cAc5 ++ cHc6 ++ cBn6 ++ cAc6 ++ cP3 ++
      cHc7 ++ cBn7 ++ cAc7 ++ cHc8 ++ cBn8 ++ cAc8 ++ cP4 ++
      cH9 ++ cA9 ++ cHa ++ cAa ++ cLog ++ cDy ++
      cDyA ++ cDy9 ++ cDx9 ++
      cDac8 ++ cDbn8 ++ cDhc8 ++ cDac7 ++ cDbn7 ++ cDhc7 ++ cDpl3 ++
      cDac6 ++ cDbn6 ++ cDhc6 ++ cDac5 ++ cDbn5 ++ cDhc5 ++ cDpl2 ++
      cDac4 ++ cDbn4 ++ cDhc4 ++ cDac3 ++ cDbn3 ++ cDhc3 ++ cDpl1 ++
      cDac2 ++ cDbn2 ++ cDhc2 ++ cDac1 ++ cDbn1 ++ cDhc1 ++
      cW1g ++ cb1g ++ cg1 ++ cbt1 ++ cW2g ++ cb2g ++ cg2 ++ cbt2 ++
      cW3g ++ cb3g ++ cg3 ++ cbt3 ++ cW4g ++ cb4g ++ cg4 ++ cbt4 ++
      cW5g ++ cb5g ++ cg5 ++ cbt5 ++ cW6g ++ cb6g ++ cg6 ++ cbt6 ++
      cW7g ++ cb7g ++ cg7 ++ cbt7 ++ cW8g ++ cb8g ++ cg8 ++ cbt8 ++
      cW9 ++ cb9 ++ cWa ++ cba ++ cWb ++ cbb
    pure <|
      "    // ── cifar8-bn train step: every line is pretty(verified AST node) ──\n" ++ body ++
      s!"    return {nW1g}, {nb1g}, {ng1}, {nbt1}, {nW2g}, {nb2g}, {ng2}, {nbt2}, {nW3g}, {nb3g}, {ng3}, {nbt3}, {nW4g}, {nb4g}, {ng4}, {nbt4}, {nW5g}, {nb5g}, {ng5}, {nbt5}, {nW6g}, {nb6g}, {ng6}, {nbt6}, {nW7g}, {nb7g}, {ng7}, {nbt7}, {nW8g}, {nb8g}, {ng8}, {nbt8}, {nW9}, {nb9}, {nWa}, {nba}, {nWb}, {nbb} : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c3,c2,kH,kW]}, {ty [c3]}, {ty [c3]}, {ty [c3]}, {ty [c3,c3,kH,kW]}, {ty [c3]}, {ty [c3]}, {ty [c3]}, {ty [c4,c3,kH,kW]}, {ty [c4]}, {ty [c4]}, {ty [c4]}, {ty [c4,c4,kH,kW]}, {ty [c4]}, {ty [c4]}, {ty [c4]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n"
  let inner : String := go.run' 0
  "module @m {\n" ++
  s!"  func.func @cifar8_bn_train_step(%x: {ty [B,ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %g5: {ty [c3]}, %bt5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %g6: {ty [c3]}, %bt6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %g7: {ty [c4]}, %bt7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %g8: {ty [c4]}, %bt8: {ty [c4]}, %W9: {ty [flat,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c3,c2,kH,kW]}, {ty [c3]}, {ty [c3]}, {ty [c3]}, {ty [c3,c3,kH,kW]}, {ty [c3]}, {ty [c3]}, {ty [c3]}, {ty [c4,c3,kH,kW]}, {ty [c4]}, {ty [c4]}, {ty [c4]}, {ty [c4,c4,kH,kW]}, {ty [c4]}, {ty [c4]}, {ty [c4]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

end Proofs.StableHLO

-- Regenerate `verified_mlir/cnn_train_step.mlir` (what MainMnistCnnVerified trains on)
-- from the faithful renderer; the den-certified proofs live in CnnFaithfulPoC.lean.
-- (cnnTrainStepText — the hand-written predecessor — is kept in StableHLO.lean for
-- reference.) Dims `128 1 32 14 14 512 10 3 3`: B=128, ic=1, c=32, h=w=14 (post-pool,
-- image 28×28), d1=512, nClasses=10, 3×3 kernels; lr = 0.1/128 (mean-loss equiv).
#eval IO.FS.writeFile "verified_mlir/cnn_train_step.mlir"
  (Proofs.StableHLO.cnnTrainStepFaithfulV 128 1 32 14 14 512 10 3 3 "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0))

-- Regenerate `verified_mlir/cifar_train_step.mlir` (what MainCifarVerified trains on)
-- from the faithful renderer; the den-certified proofs live in CifarFaithfulPoC.lean.
-- (cifarTrainStepText — the hand-written predecessor — is kept in StableHLO.lean for
-- reference.) Dims `128 3 32 64 8 8 512 10 3 3`: B=128, ic=3, c1=32, c2=64, h=w=8
-- (final pooled, image 32×32), d1=512, nClasses=10, 3×3 kernels; lr = 0.1/128.
#eval IO.FS.writeFile "verified_mlir/cifar_train_step.mlir"
  (Proofs.StableHLO.cifarTrainStepFaithfulV 128 3 32 64 8 8 512 10 3 3 "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0))

-- Regenerate `verified_mlir/cifar_bn_train_step.mlir` (what MainCifarBnVerified trains on)
-- from the faithful renderer; the den-certified proofs live in CifarBnFaithfulPoC.lean.
-- (cifarBnTrainStepText — the hand-written predecessor — is kept in StableHLO.lean for
-- reference.) Dims `128 3 32 64 8 8 512 10 3 3`, ε=1e-5, lr = 0.1/128.
#eval IO.FS.writeFile "verified_mlir/cifar_bn_train_step.mlir"
  (Proofs.StableHLO.cifarBnTrainStepFaithfulV 128 3 32 64 8 8 512 10 3 3 "1.0e-05" "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0))

-- Regenerate `verified_mlir/cifar8_train_step.mlir` (what MainCifar8Verified trains on)
-- from the faithful renderer; the den-certified proofs live in Cifar8FaithfulPoC.lean.
-- (cifar8TrainStepText — the hand-written predecessor — is kept in StableHLO.lean for
-- reference.) Dims `128 3 16 16 32 32 2 2 64 10 3 3`: h=w=2 (final pooled, image 32×32).
#eval IO.FS.writeFile "verified_mlir/cifar8_train_step.mlir"
  (Proofs.StableHLO.cifar8TrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3 "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0))

-- Regenerate `verified_mlir/cifar8_bn_train_step.mlir` (what MainCifar8BnVerified trains on)
-- from the faithful renderer; den-certified by the existing generics (CifarPoC.conv{W,B}_den,
-- CifarBnPoC.bn{Gamma,Beta}_den, Cifar8PoC.dense{W,B}_den). cifar8BnTrainStepText kept for ref.
#eval IO.FS.writeFile "verified_mlir/cifar8_bn_train_step.mlir"
  (Proofs.StableHLO.cifar8BnTrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3 "1.0e-05" "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0))
