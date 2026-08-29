import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.ViTRender

/-! # CNN + CIFAR render half — conv train-step text as a name-threaded render of proven graphs

The peer of `MlpRender.lean` (`mlpTrainStepStructured`) for the Chapter-3 MNIST CNN
(`cnnTrainStepStructured`), the Chapter-4 CIFAR CNN (`cifarTrainStepStructured`), and the
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
  let go : StateM Proofs.StableHLO.EmitS String := do
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
  let body : String := go.run' (0, [])
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
  let act : StateM Proofs.StableHLO.EmitS (String × String × String × String × String × String × String × String × String × String × String × String) := do
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
          nW1g, nb1g, nW2g, nb2g, nW3, nb3, nW4, nb4, nW5, nb5, nLog)
  let (body, n1W, n1b, n2W, n2b, n3W, n3b, n4W, n4b, n5W, n5b, nLog) := act.run' (0, [])
  -- ⚠⚠ `%loss` IS REPORT-ONLY — a DECLARED CARVE-OUT, exactly as in `MlpRender` and as
  -- ConvNeXt/EfficientNet/R50 already do. Hand-written text, not `pretty` of a `den`
  -- node. APPENDED, never woven in: it reads only the logits and `%onehot` and adds
  -- only `%l*` names, so the ten proven parameter outputs are byte-identical and
  -- `CnnFaithfulPoC` is untouched. `%lslot` is the unused input that keeps the C
  -- entry's single shape list symmetric (see `MlpRender` for the full argument).
  let lossCode :=
    "    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──\n" ++
    s!"    %lz = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    s!"    %lex = stablehlo.exponential {nLog} : {ty [B,nClasses]}\n" ++
    s!"    %lsum = stablehlo.reduce(%lex init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [B,nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
    s!"    %lsmb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,nClasses]}\n" ++
    s!"    %lsm = stablehlo.divide %lex, %lsmb : {ty [B,nClasses]}\n" ++
    s!"    %llog = stablehlo.log %lsm : {ty [B,nClasses]}\n" ++
    s!"    %lohll = stablehlo.multiply %onehot, %llog : {ty [B,nClasses]}\n" ++
    s!"    %lrow = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [B,nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
    s!"    %lsum2 = stablehlo.reduce(%lrow init: %lz) applies stablehlo.add across dimensions = [0] : ({ty [B]}, tensor<f32>) -> tensor<f32>\n" ++
    s!"    %lbf = stablehlo.constant dense<{B}.0> : tensor<f32>\n" ++
    s!"    %lossm = stablehlo.divide %lsum2, %lbf : tensor<f32>\n" ++
    s!"    %loss = stablehlo.negate %lossm : tensor<f32>\n"
  "module @m {\n" ++
  s!"  func.func @cnn_train_step(%x: {ty [B,ic*(2*h)*(2*w)]}, %W1: {ty [c,ic,kH,kW]}, %b1: {ty [c]}, %W2: {ty [c,c,kH,kW]}, %b2: {ty [c]}, %W3: {ty [flat,d1]}, %b3: {ty [d1]}, %W4: {ty [d1,d1]}, %b4: {ty [d1]}, %W5: {ty [d1,nClasses]}, %b5: {ty [nClasses]}, %lslot: tensor<f32>, %onehot: {ty [B,nClasses]}) -> ({ty [c,ic,kH,kW]}, {ty [c]}, {ty [c,c,kH,kW]}, {ty [c]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}, tensor<f32>) " ++ "{\n" ++
  "    // ── cnn train step: every line is pretty(verified AST node) ──\n" ++
  body ++
  lossCode ++
  s!"    return {n1W}, {n1b}, {n2W}, {n2b}, {n3W}, {n3b}, {n4W}, {n4b}, {n5W}, {n5b}, %loss : {ty [c,ic,kH,kW]}, {ty [c]}, {ty [c,c,kH,kW]}, {ty [c]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}, tensor<f32>\n" ++
  "  }\n}\n"

/-- Structured **CIFAR CNN** train-step renderer (`@cifar_train_step`): the Chapter-4 peer of
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
  let go : StateM Proofs.StableHLO.EmitS String := do
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
  let body : String := go.run' (0, [])
  "module @m {\n" ++
  s!"  func.func @cifar_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [flat,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  body ++
  s!"    return %W1n, %b1n, %W2n, %b2n, %W3n, %b3n, %W4n, %b4n, %W5n, %b5n, %W6n, %b6n, %W7n, %b7n : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
  "  }\n}\n"

/-- **CIFAR-CNN (Chapter 4, no-BN) train step rendered ENTIRELY from the verified AST.**
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
  let act : StateM Proofs.StableHLO.EmitS (String × String × String × String × String × String × String ×
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
  let (body, r1W, r1b, r2W, r2b, r3W, r3b, r4W, r4b, r5W, r5b, r6W, r6b, r7W, r7b) := act.run' (0, [])
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
  let go : StateM Proofs.StableHLO.EmitS String := do
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
  let body : String := go.run' (0, [])
  "module @m {\n" ++
  s!"  func.func @cifar_bn_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [flat,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  body ++
  s!"    return %W1n, %b1n, %g1n, %bt1n, %W2n, %b2n, %g2n, %bt2n, %W3n, %b3n, %g3n, %bt3n, %W4n, %b4n, %g4n, %bt4n, %W5n, %b5n, %W6n, %b6n, %W7n, %b7n : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
  "  }\n}\n"

/-- **CIFAR-BN (Chapter 4, per-channel BatchNorm) train step rendered ENTIRELY from the
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
  let go : StateM Proofs.StableHLO.EmitS String := do
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
  let inner : String := go.run' (0, [])
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
    (Wb : Mat d1 nClasses) (bb : Vec nClasses) (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) (bf16 : Bool := false) : String :=
  -- Identity, exactly as the ImageNet `*RenderB` renderers pass it (ResNet34RenderB l.81):
  -- the PROOF carries an arbitrary `rnd`; the bf16 claim lives in the EMIT shape.
  let zrnd : ℝ → ℝ := fun r => r
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
  let go : StateM Proofs.StableHLO.EmitS String := do
    -- ═══ forward (proof-rendered, flat): (conv→relu)×2→pool ×4 → (dense→relu)×2→dense ═══
    let (cHc1, nHc1) ← pretty B (if bf16 then .flatConvFBf16 (h := s1h) (w := s1w) zrnd "%W1" "%b1" W₁ b₁ (.operand "%x" x) else .flatConvF (h := s1h) (w := s1w) "%W1" "%b1" W₁ b₁ (.operand "%x" x))
    let (cAc1, nAc1) ← pretty B (.reluF (.operand nHc1 zS1c1))
    let (cHc2, nHc2) ← pretty B (if bf16 then .flatConvFBf16 (h := s1h) (w := s1w) zrnd "%W2" "%b2" W₂ b₂ (.operand nAc1 zS1c1) else .flatConvF (h := s1h) (w := s1w) "%W2" "%b2" W₂ b₂ (.operand nAc1 zS1c1))
    let (cAc2, nAc2) ← pretty B (.reluF (.operand nHc2 zS1c1))
    let (cP1, nPool1) ← pretty B (.maxPoolF (c := c1) (h := s2h) (w := s2w) (.operand nAc2 zS1c1))
    let (cHc3, nHc3) ← pretty B (if bf16 then .flatConvFBf16 (h := s2h) (w := s2w) zrnd "%W3" "%b3" W₃ b₃ (.operand nPool1 zS2c1) else .flatConvF (h := s2h) (w := s2w) "%W3" "%b3" W₃ b₃ (.operand nPool1 zS2c1))
    let (cAc3, nAc3) ← pretty B (.reluF (.operand nHc3 zS2c2))
    let (cHc4, nHc4) ← pretty B (if bf16 then .flatConvFBf16 (h := s2h) (w := s2w) zrnd "%W4" "%b4" W₄ b₄ (.operand nAc3 zS2c2) else .flatConvF (h := s2h) (w := s2w) "%W4" "%b4" W₄ b₄ (.operand nAc3 zS2c2))
    let (cAc4, nAc4) ← pretty B (.reluF (.operand nHc4 zS2c2))
    let (cP2, nPool2) ← pretty B (.maxPoolF (c := c2) (h := s3h) (w := s3w) (.operand nAc4 zS2c2))
    let (cHc5, nHc5) ← pretty B (if bf16 then .flatConvFBf16 (h := s3h) (w := s3w) zrnd "%W5" "%b5" W₅ b₅ (.operand nPool2 zS3c2) else .flatConvF (h := s3h) (w := s3w) "%W5" "%b5" W₅ b₅ (.operand nPool2 zS3c2))
    let (cAc5, nAc5) ← pretty B (.reluF (.operand nHc5 zS3c3))
    let (cHc6, nHc6) ← pretty B (if bf16 then .flatConvFBf16 (h := s3h) (w := s3w) zrnd "%W6" "%b6" W₆ b₆ (.operand nAc5 zS3c3) else .flatConvF (h := s3h) (w := s3w) "%W6" "%b6" W₆ b₆ (.operand nAc5 zS3c3))
    let (cAc6, nAc6) ← pretty B (.reluF (.operand nHc6 zS3c3))
    let (cP3, nPool3) ← pretty B (.maxPoolF (c := c3) (h := s4h) (w := s4w) (.operand nAc6 zS3c3))
    let (cHc7, nHc7) ← pretty B (if bf16 then .flatConvFBf16 (h := s4h) (w := s4w) zrnd "%W7" "%b7" W₇ b₇ (.operand nPool3 zS4c3) else .flatConvF (h := s4h) (w := s4w) "%W7" "%b7" W₇ b₇ (.operand nPool3 zS4c3))
    let (cAc7, nAc7) ← pretty B (.reluF (.operand nHc7 zS4c4))
    let (cHc8, nHc8) ← pretty B (if bf16 then .flatConvFBf16 (h := s4h) (w := s4w) zrnd "%W8" "%b8" W₈ b₈ (.operand nAc7 zS4c4) else .flatConvF (h := s4h) (w := s4w) "%W8" "%b8" W₈ b₈ (.operand nAc7 zS4c4))
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
  let inner : String := go.run' (0, [])
  "module @m {\n" ++
  s!"  func.func @cifar8_train_step(%x: {ty [B,ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %W9: {ty [flat,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c3,c2,kH,kW]}, {ty [c3]}, {ty [c3,c3,kH,kW]}, {ty [c3]}, {ty [c4,c3,kH,kW]}, {ty [c4]}, {ty [c4,c4,kH,kW]}, {ty [c4]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § cifar8 AdamW — the same forward/backward, optimizer swapped for the proven Adam ops
-- ════════════════════════════════════════════════════════════════

/-- Which optimizer tail the cifar8 render emits (handoff §2i). All three share ONE forward,
    backward and un-fused-gradient body, and one packed `[θ|m|v]` signature — 71 in / 69 out for
    every variant — so the ablation section's "SGD several ways" is genuinely the same net with the
    optimizer swapped, and a reader can diff the artifacts to see only the tail move. -/
inductive CifarOpt
  /-- `θ' = θ − lr·(m̂/(√v̂+ε)) − lr·wd·θ`, and both moments live. -/
  | adamw
  /-- `θ' = θ − lr·g`; `m`/`v` ride through untouched. -/
  | sgd
  /-- `v' = μ·v + g`, `θ' = θ − lr·(g + μ·v')`; velocity in the `v` slot, `m` untouched. -/
  | nesterov
deriving DecidableEq, Repr

/-- One parameter's optimizer tail. The gradient is emitted **once** and every optimizer output
    reads it back by SSA name (`.operand gradSSA`), so the outputs share one gradient subgraph
    rather than a copy each. Returns `(code, θ', m', v')` — the slots the packed protocol returns.

    * `.adamw` — the proven triple, denoting `Proofs.adamWStep` (`adamW_triple_faithful`).
    * `.sgd` — `sgdParamF`, denoting `Proofs.sgdParam`. `m'`/`v'` are the INPUT names: a
      passthrough, which is what keeps the signature identical to AdamW's.
    * `.nesterov` — `momParamF` + `momVNextF`, together denoting `Proofs.momStep`
      (`mom_pair_faithful`). Velocity occupies the `v` slot; `m` passes through.

    `%mu` is a baked constant and `%lr` a runtime arg, matching the retired emitter exactly. -/
private def optTail (opt : CifarOpt) (B replicas n : Nat) (pName : String) (ds : List Nat)
    (gradSSA : String) : StateM Proofs.StableHLO.EmitS (String × String × String × String) := do
  let z : Vec n := fun _ => 0
  -- At `replicas > 1`, average the gradient across devices first. Trusted carve-out, same as
  -- ResNet34RenderB's (handoff §2b-quater, §5). cifar8 is where that carve-out gets its EXACT
  -- gate: no BatchNorm, so the loss is a plain mean over examples and the batch decomposition
  --   (1/2)[(1/B)Σ_A + (1/B)Σ_B] = (1/2B)Σ_{A∪B}
  -- holds identically — 2×B with the collective must equal 1×2B to fp rounding. R34 cannot be
  -- checked this way: BN normalises per replica, so there N×b ≠ 1×(N·b) BY DESIGN.
  -- `pName` is "%W1"; the collective's SSA tag must not carry the '%'. `String.drop` returns a
  -- `String.Slice` on this toolchain (Lean 4.32), hence the explicit `.toString`.
  let (arS, gAvg) := ViTRender.emitGradAllReduce gradSSA ds (pName.drop 1).toString replicas
  match opt with
  | .adamw =>
    let (cT, nT) ← pretty B (SHlo.adamWParamF pName s!"{pName}m" s!"{pName}v"
        "%b1" "%ob1" "%b2" "%ob2" "%bc1" "%bc2" "%lr" "%eps" "%wd" ds
        0 0 0 0 0 0 0 z z z (.operand gAvg z))
    let (cM, nM) ← pretty B (SHlo.adamMNextF s!"{pName}m" "%b1" "%ob1" ds 0 z (.operand gAvg z))
    let (cV, nV) ← pretty B (SHlo.adamVNextF s!"{pName}v" "%b2" "%ob2" ds 0 z (.operand gAvg z))
    pure (arS ++ cT ++ cM ++ cV, nT, nM, nV)
  | .sgd =>
    let (cT, nT) ← pretty B (SHlo.sgdParamF pName "%lr" ds 0 z (.operand gAvg z))
    -- m/v are returned UNCHANGED. That is not laziness: the packed signature is shared with
    -- AdamW so the driver is byte-identical across variants, and a plain-SGD step genuinely
    -- has no state to carry.
    pure (arS ++ cT, nT, s!"{pName}m", s!"{pName}v")
  | .nesterov =>
    let (cT, nT) ← pretty B (SHlo.momParamF pName s!"{pName}v" "%mu" "%lr" ds 0 0 z z
        (.operand gAvg z))
    let (cV, nV) ← pretty B (SHlo.momVNextF s!"{pName}v" "%mu" ds 0 z (.operand gAvg z))
    pure (arS ++ cT ++ cV, nT, s!"{pName}m", nV)

set_option maxRecDepth 8000 in
/-- **cifar8 AdamW train step rendered ENTIRELY from the verified AST** — the optimizer half of
    `planning/xla_pjrt_handoff.md` §2a. Identical forward/backward to
    `cifar8TrainStepFaithfulV`; the 22 fused SGD ops are replaced by 22 un-fused param
    gradients (`convWeightGrad`/`convBiasGrad`/`weightGrad`/`biasGrad`) each feeding the three
    proven AdamW ops (`adamWParamF`/`adamMNextF`/`adamVNextF`, denoting `Proofs.adamWStep`).

    Signature matches the packed `trainAdamSched` protocol byte for byte:
    `(x, θ×22, m×22, v×22, %lr, %bc1, %bc2, onehot) → (θ'×22, m'×22, v'×22, loss, bc1, bc2)`.
    β₁/β₂/ε/wd are baked as in-body constants (`%b1 %ob1 %b2 %ob2 %eps %wd`), so the conv
    biases are named `%cb1…%cb8` — `%b1` is β₁.

    **Two lines are outside the proven surface**, both marked in the emitted text: the scalar
    `%loss` (report-only; the kit has no rank-0 loss op and it does not feed the update) and
    the `%bc1`/`%bc2` passthroughs. The mean-loss `1/B` on the cotangent IS proven — it is
    `scaleF`, not hand-written text. Unlike the SGD render it cannot be folded into `lr`,
    because `lr` is a runtime scalar here. -/
def cifar8AdamTrainStepFaithfulV (B ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat)
    (invBStr b1Str ob1Str b2Str ob2Str epsStr wdStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))))
    -- Trailing + defaulted so every existing positional call site is unchanged.
    (replicas : Nat := 1) (opt : CifarOpt := .adamw) (bf16 : Bool := false) : String :=
  let zrnd : ℝ → ℝ := fun r => r   -- identity, as the ImageNet renderers pass it
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
  let go : StateM Proofs.StableHLO.EmitS String := do
    -- ═══ forward — identical to cifar8TrainStepFaithfulV, conv biases renamed %cb* ═══
    let (cHc1, nHc1) ← pretty B (if bf16 then .flatConvFBf16 (h := s1h) (w := s1w) zrnd "%W1" "%cb1" W₁ b₁ (.operand "%x" x) else .flatConvF (h := s1h) (w := s1w) "%W1" "%cb1" W₁ b₁ (.operand "%x" x))
    let (cAc1, nAc1) ← pretty B (.reluF (.operand nHc1 zS1c1))
    let (cHc2, nHc2) ← pretty B (if bf16 then .flatConvFBf16 (h := s1h) (w := s1w) zrnd "%W2" "%cb2" W₂ b₂ (.operand nAc1 zS1c1) else .flatConvF (h := s1h) (w := s1w) "%W2" "%cb2" W₂ b₂ (.operand nAc1 zS1c1))
    let (cAc2, nAc2) ← pretty B (.reluF (.operand nHc2 zS1c1))
    let (cP1, nPool1) ← pretty B (.maxPoolF (c := c1) (h := s2h) (w := s2w) (.operand nAc2 zS1c1))
    let (cHc3, nHc3) ← pretty B (if bf16 then .flatConvFBf16 (h := s2h) (w := s2w) zrnd "%W3" "%cb3" W₃ b₃ (.operand nPool1 zS2c1) else .flatConvF (h := s2h) (w := s2w) "%W3" "%cb3" W₃ b₃ (.operand nPool1 zS2c1))
    let (cAc3, nAc3) ← pretty B (.reluF (.operand nHc3 zS2c2))
    let (cHc4, nHc4) ← pretty B (if bf16 then .flatConvFBf16 (h := s2h) (w := s2w) zrnd "%W4" "%cb4" W₄ b₄ (.operand nAc3 zS2c2) else .flatConvF (h := s2h) (w := s2w) "%W4" "%cb4" W₄ b₄ (.operand nAc3 zS2c2))
    let (cAc4, nAc4) ← pretty B (.reluF (.operand nHc4 zS2c2))
    let (cP2, nPool2) ← pretty B (.maxPoolF (c := c2) (h := s3h) (w := s3w) (.operand nAc4 zS2c2))
    let (cHc5, nHc5) ← pretty B (if bf16 then .flatConvFBf16 (h := s3h) (w := s3w) zrnd "%W5" "%cb5" W₅ b₅ (.operand nPool2 zS3c2) else .flatConvF (h := s3h) (w := s3w) "%W5" "%cb5" W₅ b₅ (.operand nPool2 zS3c2))
    let (cAc5, nAc5) ← pretty B (.reluF (.operand nHc5 zS3c3))
    let (cHc6, nHc6) ← pretty B (if bf16 then .flatConvFBf16 (h := s3h) (w := s3w) zrnd "%W6" "%cb6" W₆ b₆ (.operand nAc5 zS3c3) else .flatConvF (h := s3h) (w := s3w) "%W6" "%cb6" W₆ b₆ (.operand nAc5 zS3c3))
    let (cAc6, nAc6) ← pretty B (.reluF (.operand nHc6 zS3c3))
    let (cP3, nPool3) ← pretty B (.maxPoolF (c := c3) (h := s4h) (w := s4w) (.operand nAc6 zS3c3))
    let (cHc7, nHc7) ← pretty B (if bf16 then .flatConvFBf16 (h := s4h) (w := s4w) zrnd "%W7" "%cb7" W₇ b₇ (.operand nPool3 zS4c3) else .flatConvF (h := s4h) (w := s4w) "%W7" "%cb7" W₇ b₇ (.operand nPool3 zS4c3))
    let (cAc7, nAc7) ← pretty B (.reluF (.operand nHc7 zS4c4))
    let (cHc8, nHc8) ← pretty B (if bf16 then .flatConvFBf16 (h := s4h) (w := s4w) zrnd "%W8" "%cb8" W₈ b₈ (.operand nAc7 zS4c4) else .flatConvF (h := s4h) (w := s4w) "%W8" "%cb8" W₈ b₈ (.operand nAc7 zS4c4))
    let (cAc8, nAc8) ← pretty B (.reluF (.operand nHc8 zS4c4))
    let (cP4, nPool4) ← pretty B (.maxPoolF (c := c4) (h := h) (w := w) (.operand nAc8 zS4c4))
    let (cH9, nH9) ← pretty B (denseF "%W9" "%b9" W₉ b₉ (.operand nPool4 zPc4))
    let (cA9, nA9) ← pretty B (.reluF (.operand nH9 zD1))
    let (cHa, nHa) ← pretty B (denseF "%Wa" "%ba" Wa ba (.operand nA9 zD1))
    let (cAa, nAa) ← pretty B (.reluF (.operand nHa zD1))
    let (cLog, nLog) ← pretty B (denseF "%Wb" "%bb" Wb bb (.operand nAa zD1))
    -- ═══ MEAN-loss cotangent: softmax split out so the report-only %loss can read it ═══
    let (cSm, nSm) ← pretty B (.softmaxDiv (.expe (.operand nLog zNC)))
    let (cD0, nD0) ← pretty B (.sub (.operand nSm zNC) (.operand "%onehot" zNC))
    let (cDy, nDy) ← pretty B (.scaleF invBStr 0 (.operand nD0 zNC))
    -- ═══ backward chain — identical to the SGD render ═══
    let (cDyA, nDyA) ← pretty B (.selectPos nHa zD1 (.dotOut "%Wb" Wb (.operand nDy zNC)))
    let (cDy9, nDy9) ← pretty B (.selectPos nH9 zD1 (.dotOut "%Wa" Wa (.operand nDyA zD1)))
    let (cDx9, nDx9) ← pretty B (.dotOut "%W9" W₉ (.operand nDy9 zD1))
    let (cDac8, nDac8) ← pretty B (.maxPoolBack (c := c4) (h := h) (w := w) nAc8 zS4c4 (.operand nDx9 zPc4))
    let (cDhc8, nDhc8) ← pretty B (.selectPos nHc8 zS4c4 (.operand nDac8 zS4c4))
    let (cDac7, nDac7) ← pretty B (.convBack (h := s4h) (w := s4w) "%W8" W₈ b₈ zS4c4 (.operand nDhc8 zS4c4))
    let (cDhc7, nDhc7) ← pretty B (.selectPos nHc7 zS4c4 (.operand nDac7 zS4c4))
    let (cDpl3, nDpool3) ← pretty B (.convBack (h := s4h) (w := s4w) "%W7" W₇ b₇ zS4c3 (.operand nDhc7 zS4c4))
    let (cDac6, nDac6) ← pretty B (.maxPoolBack (c := c3) (h := s4h) (w := s4w) nAc6 zS3c3 (.operand nDpool3 zS4c3))
    let (cDhc6, nDhc6) ← pretty B (.selectPos nHc6 zS3c3 (.operand nDac6 zS3c3))
    let (cDac5, nDac5) ← pretty B (.convBack (h := s3h) (w := s3w) "%W6" W₆ b₆ zS3c3 (.operand nDhc6 zS3c3))
    let (cDhc5, nDhc5) ← pretty B (.selectPos nHc5 zS3c3 (.operand nDac5 zS3c3))
    let (cDpl2, nDpool2) ← pretty B (.convBack (h := s3h) (w := s3w) "%W5" W₅ b₅ zS3c2 (.operand nDhc5 zS3c3))
    let (cDac4, nDac4) ← pretty B (.maxPoolBack (c := c2) (h := s3h) (w := s3w) nAc4 zS2c2 (.operand nDpool2 zS3c2))
    let (cDhc4, nDhc4) ← pretty B (.selectPos nHc4 zS2c2 (.operand nDac4 zS2c2))
    let (cDac3, nDac3) ← pretty B (.convBack (h := s2h) (w := s2w) "%W4" W₄ b₄ zS2c2 (.operand nDhc4 zS2c2))
    let (cDhc3, nDhc3) ← pretty B (.selectPos nHc3 zS2c2 (.operand nDac3 zS2c2))
    let (cDpl1, nDpool1) ← pretty B (.convBack (h := s2h) (w := s2w) "%W3" W₃ b₃ zS2c1 (.operand nDhc3 zS2c2))
    let (cDac2, nDac2) ← pretty B (.maxPoolBack (c := c1) (h := s2h) (w := s2w) nAc2 zS1c1 (.operand nDpool1 zS2c1))
    let (cDhc2, nDhc2) ← pretty B (.selectPos nHc2 zS1c1 (.operand nDac2 zS1c1))
    let (cDac1, nDac1) ← pretty B (.convBack (h := s1h) (w := s1w) "%W2" W₂ b₂ zS1c1 (.operand nDhc2 zS1c1))
    let (cDhc1, nDhc1) ← pretty B (.selectPos nHc1 zS1c1 (.operand nDac1 zS1c1))
    -- ═══ per param: un-fused gradient, then the three proven AdamW outputs ═══
    let (gW1, sW1) ← pretty B (SHlo.convWeightGrad "%x" b₁ zTW1 W₁ (.operand nDhc1 zS1c1))
    let (aW1, tW1, mW1, vW1) ← optTail opt B replicas (c1*ic*kH*kW) "%W1" [c1,ic,kH,kW] sW1
    let (gb1, sb1) ← pretty B (SHlo.convBiasGrad W₁ zTW1 b₁ (.operand nDhc1 zS1c1))
    let (ab1, tb1, mb1, vb1) ← optTail opt B replicas c1 "%cb1" [c1] sb1
    let (gW2, sW2) ← pretty B (SHlo.convWeightGrad nAc1 b₂ zTW2 W₂ (.operand nDhc2 zS1c1))
    let (aW2, tW2, mW2, vW2) ← optTail opt B replicas (c1*c1*kH*kW) "%W2" [c1,c1,kH,kW] sW2
    let (gb2, sb2) ← pretty B (SHlo.convBiasGrad W₂ zTW2 b₂ (.operand nDhc2 zS1c1))
    let (ab2, tb2, mb2, vb2) ← optTail opt B replicas c1 "%cb2" [c1] sb2
    let (gW3, sW3) ← pretty B (SHlo.convWeightGrad nPool1 b₃ zTW3 W₃ (.operand nDhc3 zS2c2))
    let (aW3, tW3, mW3, vW3) ← optTail opt B replicas (c2*c1*kH*kW) "%W3" [c2,c1,kH,kW] sW3
    let (gb3, sb3) ← pretty B (SHlo.convBiasGrad W₃ zTW3 b₃ (.operand nDhc3 zS2c2))
    let (ab3, tb3, mb3, vb3) ← optTail opt B replicas c2 "%cb3" [c2] sb3
    let (gW4, sW4) ← pretty B (SHlo.convWeightGrad nAc3 b₄ zTW4 W₄ (.operand nDhc4 zS2c2))
    let (aW4, tW4, mW4, vW4) ← optTail opt B replicas (c2*c2*kH*kW) "%W4" [c2,c2,kH,kW] sW4
    let (gb4, sb4) ← pretty B (SHlo.convBiasGrad W₄ zTW4 b₄ (.operand nDhc4 zS2c2))
    let (ab4, tb4, mb4, vb4) ← optTail opt B replicas c2 "%cb4" [c2] sb4
    let (gW5, sW5) ← pretty B (SHlo.convWeightGrad nPool2 b₅ zTW5 W₅ (.operand nDhc5 zS3c3))
    let (aW5, tW5, mW5, vW5) ← optTail opt B replicas (c3*c2*kH*kW) "%W5" [c3,c2,kH,kW] sW5
    let (gb5, sb5) ← pretty B (SHlo.convBiasGrad W₅ zTW5 b₅ (.operand nDhc5 zS3c3))
    let (ab5, tb5, mb5, vb5) ← optTail opt B replicas c3 "%cb5" [c3] sb5
    let (gW6, sW6) ← pretty B (SHlo.convWeightGrad nAc5 b₆ zTW6 W₆ (.operand nDhc6 zS3c3))
    let (aW6, tW6, mW6, vW6) ← optTail opt B replicas (c3*c3*kH*kW) "%W6" [c3,c3,kH,kW] sW6
    let (gb6, sb6) ← pretty B (SHlo.convBiasGrad W₆ zTW6 b₆ (.operand nDhc6 zS3c3))
    let (ab6, tb6, mb6, vb6) ← optTail opt B replicas c3 "%cb6" [c3] sb6
    let (gW7, sW7) ← pretty B (SHlo.convWeightGrad nPool3 b₇ zTW7 W₇ (.operand nDhc7 zS4c4))
    let (aW7, tW7, mW7, vW7) ← optTail opt B replicas (c4*c3*kH*kW) "%W7" [c4,c3,kH,kW] sW7
    let (gb7, sb7) ← pretty B (SHlo.convBiasGrad W₇ zTW7 b₇ (.operand nDhc7 zS4c4))
    let (ab7, tb7, mb7, vb7) ← optTail opt B replicas c4 "%cb7" [c4] sb7
    let (gW8, sW8) ← pretty B (SHlo.convWeightGrad nAc7 b₈ zTW8 W₈ (.operand nDhc8 zS4c4))
    let (aW8, tW8, mW8, vW8) ← optTail opt B replicas (c4*c4*kH*kW) "%W8" [c4,c4,kH,kW] sW8
    let (gb8, sb8) ← pretty B (SHlo.convBiasGrad W₈ zTW8 b₈ (.operand nDhc8 zS4c4))
    let (ab8, tb8, mb8, vb8) ← optTail opt B replicas c4 "%cb8" [c4] sb8
    let (gW9, sW9) ← pretty B (SHlo.weightGrad nPool4 zPc4 (.operand nDy9 zD1))
    let (aW9, tW9, mW9, vW9) ← optTail opt B replicas (flat*d1) "%W9" [flat,d1] sW9
    let (gb9, sb9) ← pretty B (SHlo.biasGrad (n := d1) (.operand nDy9 zD1))
    let (ab9, tb9, mb9, vb9) ← optTail opt B replicas d1 "%b9" [d1] sb9
    let (gWa, sWa) ← pretty B (SHlo.weightGrad nA9 zD1 (.operand nDyA zD1))
    let (aWa, tWa, mWa, vWa) ← optTail opt B replicas (d1*d1) "%Wa" [d1,d1] sWa
    let (gba, sba) ← pretty B (SHlo.biasGrad (n := d1) (.operand nDyA zD1))
    let (aba, tba, mba, vba) ← optTail opt B replicas d1 "%ba" [d1] sba
    let (gWb, sWb) ← pretty B (SHlo.weightGrad nAa zD1 (.operand nDy zNC))
    let (aWb, tWb, mWb, vWb) ← optTail opt B replicas (d1*nClasses) "%Wb" [d1,nClasses] sWb
    let (gbb, sbb) ← pretty B (SHlo.biasGrad (n := nClasses) (.operand nDy zNC))
    let (abb, tbb, mbb, vbb) ← optTail opt B replicas nClasses "%bb" [nClasses] sbb
    -- ═══ report-only scalar loss — OUTSIDE the proven surface, does not feed the update ═══
    let lossCode :=
      "    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it\n" ++
      "    //    feeds no parameter, only the driver's progress line) ──\n" ++
      s!"    %llog = stablehlo.log {nSm} : {ty [B,nClasses]}\n" ++
      s!"    %ohll = stablehlo.multiply %onehot, %llog : {ty [B,nClasses]}\n" ++
      s!"    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,nClasses]}, tensor<f32>) -> tensor<f32>\n" ++
      s!"    %cneg = stablehlo.negate %csum : tensor<f32>\n" ++
      s!"    %lbf = stablehlo.constant dense<{B}.0> : tensor<f32>\n" ++
      s!"    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>\n"
    let body := cHc1 ++ cAc1 ++ cHc2 ++ cAc2 ++ cP1 ++ cHc3 ++ cAc3 ++ cHc4 ++ cAc4 ++ cP2 ++
      cHc5 ++ cAc5 ++ cHc6 ++ cAc6 ++ cP3 ++ cHc7 ++ cAc7 ++ cHc8 ++ cAc8 ++ cP4 ++
      cH9 ++ cA9 ++ cHa ++ cAa ++ cLog ++ cSm ++ cD0 ++ cDy ++ lossCode ++
      cDyA ++ cDy9 ++ cDx9 ++ cDac8 ++ cDhc8 ++ cDac7 ++ cDhc7 ++ cDpl3 ++
      cDac6 ++ cDhc6 ++ cDac5 ++ cDhc5 ++ cDpl2 ++ cDac4 ++ cDhc4 ++ cDac3 ++ cDhc3 ++ cDpl1 ++
      cDac2 ++ cDhc2 ++ cDac1 ++ cDhc1 ++
      gW1 ++ aW1 ++ gb1 ++ ab1 ++ gW2 ++ aW2 ++ gb2 ++ ab2 ++
      gW3 ++ aW3 ++ gb3 ++ ab3 ++ gW4 ++ aW4 ++ gb4 ++ ab4 ++
      gW5 ++ aW5 ++ gb5 ++ ab5 ++ gW6 ++ aW6 ++ gb6 ++ ab6 ++
      gW7 ++ aW7 ++ gb7 ++ ab7 ++ gW8 ++ aW8 ++ gb8 ++ ab8 ++
      gW9 ++ aW9 ++ gb9 ++ ab9 ++ gWa ++ aWa ++ gba ++ aba ++ gWb ++ aWb ++ gbb ++ abb
    let pTys := [ty [c1,ic,kH,kW], ty [c1], ty [c1,c1,kH,kW], ty [c1],
      ty [c2,c1,kH,kW], ty [c2], ty [c2,c2,kH,kW], ty [c2],
      ty [c3,c2,kH,kW], ty [c3], ty [c3,c3,kH,kW], ty [c3],
      ty [c4,c3,kH,kW], ty [c4], ty [c4,c4,kH,kW], ty [c4],
      ty [flat,d1], ty [d1], ty [d1,d1], ty [d1], ty [d1,nClasses], ty [nClasses]]
    let ths := [tW1, tb1, tW2, tb2, tW3, tb3, tW4, tb4, tW5, tb5, tW6, tb6, tW7, tb7, tW8, tb8,
                tW9, tb9, tWa, tba, tWb, tbb]
    let mns := [mW1, mb1, mW2, mb2, mW3, mb3, mW4, mb4, mW5, mb5, mW6, mb6, mW7, mb7, mW8, mb8,
                mW9, mb9, mWa, mba, mWb, mbb]
    let vns := [vW1, vb1, vW2, vb2, vW3, vb3, vW4, vb4, vW5, vb5, vW6, vb6, vW7, vb7, vW8, vb8,
                vW9, vb9, vWa, vba, vWb, vbb]
    pure <|
      "    // ── cifar8 AdamW train step: every line is pretty(verified AST node), except the\n" ++
      "    //    marked report-only loss + the %bc passthroughs ──\n" ++
      -- NB `%sc`/`%sa`/`%sb`/`%sd` are RESERVED: maxPoolBack's select_and_scatter emitter
      -- hardcodes them as region block arguments, and a top-level def of the same name is a
      -- redefinition error at parse time. Hence `%lzero`.
      "    %lzero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      s!"    %b1 = stablehlo.constant dense<{b1Str}> : tensor<f32>\n" ++
      s!"    %ob1 = stablehlo.constant dense<{ob1Str}> : tensor<f32>\n" ++
      s!"    %b2 = stablehlo.constant dense<{b2Str}> : tensor<f32>\n" ++
      s!"    %ob2 = stablehlo.constant dense<{ob2Str}> : tensor<f32>\n" ++
      s!"    %eps = stablehlo.constant dense<{epsStr}> : tensor<f32>\n" ++
      s!"    %wd = stablehlo.constant dense<{wdStr}> : tensor<f32>\n" ++
      -- Emitted ONLY for Nesterov, so the AdamW render re-renders byte-identical after this
      -- threading — the §0 gate-1 self-check that the generalisation is inert.
      (if opt == .nesterov then "    %mu = stablehlo.constant dense<0.9> : tensor<f32>\n" else "") ++
      body ++
      s!"    return {String.intercalate ", " (ths ++ mns ++ vns)}, %loss, %bc1, %bc2 : " ++
      s!"{String.intercalate ", " (pTys ++ pTys ++ pTys)}, tensor<f32>, tensor<f32>, tensor<f32>\n"
  let pSig := s!"%W1: {ty [c1,ic,kH,kW]}, %cb1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %cb2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %cb3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %cb4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %cb5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %cb6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %cb7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %cb8: {ty [c4]}, %W9: {ty [flat,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}"
  let sfx (s : String) : String :=
    s!"%W1{s}: {ty [c1,ic,kH,kW]}, %cb1{s}: {ty [c1]}, %W2{s}: {ty [c1,c1,kH,kW]}, %cb2{s}: {ty [c1]}, %W3{s}: {ty [c2,c1,kH,kW]}, %cb3{s}: {ty [c2]}, %W4{s}: {ty [c2,c2,kH,kW]}, %cb4{s}: {ty [c2]}, %W5{s}: {ty [c3,c2,kH,kW]}, %cb5{s}: {ty [c3]}, %W6{s}: {ty [c3,c3,kH,kW]}, %cb6{s}: {ty [c3]}, %W7{s}: {ty [c4,c3,kH,kW]}, %cb7{s}: {ty [c4]}, %W8{s}: {ty [c4,c4,kH,kW]}, %cb8{s}: {ty [c4]}, %W9{s}: {ty [flat,d1]}, %b9{s}: {ty [d1]}, %Wa{s}: {ty [d1,d1]}, %ba{s}: {ty [d1]}, %Wb{s}: {ty [d1,nClasses]}, %bb{s}: {ty [nClasses]}"
  let pTy := [ty [c1,ic,kH,kW], ty [c1], ty [c1,c1,kH,kW], ty [c1],
    ty [c2,c1,kH,kW], ty [c2], ty [c2,c2,kH,kW], ty [c2],
    ty [c3,c2,kH,kW], ty [c3], ty [c3,c3,kH,kW], ty [c3],
    ty [c4,c3,kH,kW], ty [c4], ty [c4,c4,kH,kW], ty [c4],
    ty [flat,d1], ty [d1], ty [d1,d1], ty [d1], ty [d1,nClasses], ty [nClasses]]
  let retTy := String.intercalate ", " (pTy ++ pTy ++ pTy) ++ ", tensor<f32>, tensor<f32>, tensor<f32>"
  let inner : String := go.run' (0, [])
  -- Entry name tracks the driver's `{slug}_{variant}_train_step` convention (see ResNet34RenderB:
  -- a mismatch here is refused by the shim as "entry mismatch", not silently mis-run).
  let fname := if replicas ≤ 1 then "cifar8_adam_train_step" else "cifar8_adamdp_train_step"
  let msfx := sfx "m"
  let vsfx := sfx "v"
  "module @m {\n" ++
  s!"  func.func @{fname}(%x: {ty [B,ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))]}, {pSig}, {msfx}, {vsfx}, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: {ty [B,nClasses]}) -> ({retTy}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

set_option maxRecDepth 8000 in

-- ════════════════════════════════════════════════════════════════
-- § cifar8 on the BATCHED op family — the unification
-- ════════════════════════════════════════════════════════════════

/-- **`cifar8AdamTrainStepFaithfulB` — the batched peer of `cifar8AdamTrainStepFaithfulV`.**

    Same net, same three optimizers, same packed `[θ|m|v]` signature. The difference is the op
    FAMILY: this render carries the batch **in the Lean type** (`SHlo (B*(c*h*w))`) and uses the
    batched constructors, where the `…V` render is per-example (`SHlo (c*h*w)`) with `pretty B`
    broadcasting. Naming follows the ImageNet renderers, where `…RenderB` is exactly this
    migration done once per net (`ResNet34RenderB`, `MobileNetV2RenderB`, …).

    ⭐⭐ **Why it exists: bf16, and rehearsal.** The 27 bf16 ops were built for ImageNet, which is
    entirely on the batched family — so bf16 twins exist for `convBackBatched`/`denseRowBack` and
    do NOT exist for the per-example `convBack`/`dotOut` that `…V` uses. Rather than write two
    CIFAR-only ops (`convBackBf16`, `dotOutBf16`) that ImageNet would never run, this moves CIFAR
    onto the ops ImageNet already uses. bf16 then drops in for the whole step, forward AND
    backward, with **zero new verified ops** — and CIFAR becomes a real rehearsal for ImageNet
    instead of a parallel dialect. See planning/cifar_lowprec_stability.md §4.1.

    ⭐ **The migration is semantically free, not a re-derivation.** Both families denote the SAME
    proven VJP — `StableHLO.lean` l.2016 vs l.2200 are `(conv2d_has_vjp3 W b).backward v …` and
    `batchMap N (… (conv2d_has_vjp3 W b)).backward (fun _ => 0) …`. The only difference is the
    primal argument, and l.2990 records why it is free: *conv is linear, so this is a global VJP*
    — the input-VJP ignores the primal. That is why `.convBack`'s primal argument is simply
    dropped below rather than threaded.

    ⚠ Faithful by CONSTRUCTION, like every render here: the AST is built only from verified
    constructors, so `pretty(provenGraph)` needs no new proof. Nothing in this function is
    hand-written MLIR except the report-only `%loss`, exactly as in the `…V` peer.

    ⚠ Parameters are NOT batched — only activations are. The optimizer tail (`optTail`) is
    therefore untouched and shared verbatim with `…V`. -/
def cifar8AdamTrainStepFaithfulB (B ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat)
    (invBStr b1Str ob1Str b2Str ob2Str epsStr wdStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (B*(ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))))
    -- Trailing + defaulted so every existing positional call site is unchanged.
    (replicas : Nat := 1) (opt : CifarOpt := .adamw) (bf16 : Bool := false) (fp8 : Bool := false) : String :=
  let zrnd : ℝ → ℝ := fun r => r   -- identity, as the ImageNet renderers pass it
  let s4h := 2*h; let s4w := 2*w
  let s3h := 2*s4h; let s3w := 2*s4w
  let s2h := 2*s3h; let s2w := 2*s3w
  let s1h := 2*s2h; let s1w := 2*s2w
  let flat := c4*h*w
  let _zS1c1 : Vec (c1*s1h*s1w) := fun _ => 0
  let _zS2c1 : Vec (c1*s2h*s2w) := fun _ => 0
  let _zS2c2 : Vec (c2*s2h*s2w) := fun _ => 0
  let _zS3c2 : Vec (c2*s3h*s3w) := fun _ => 0
  let _zS3c3 : Vec (c3*s3h*s3w) := fun _ => 0
  let _zS4c3 : Vec (c3*s4h*s4w) := fun _ => 0
  let _zS4c4 : Vec (c4*s4h*s4w) := fun _ => 0
  let _zPc4 : Vec (c4*h*w) := fun _ => 0
  let _zD1 : Vec d1 := fun _ => 0
  let _zNC : Vec nClasses := fun _ => 0
  -- ── batched peers: every ACTIVATION operand carries the batch in its type here, which is
  -- the whole difference between this render and the `…V` one. Parameters keep their
  -- per-example types (they are not batched), so the optimizer tail below is untouched.
  let bX    : Vec (B*(ic*s1h*s1w)) := fun _ => 0
  let bS1c1 : Vec (B*(c1*s1h*s1w)) := fun _ => 0
  let bS2c1 : Vec (B*(c1*s2h*s2w)) := fun _ => 0
  let bS2c2 : Vec (B*(c2*s2h*s2w)) := fun _ => 0
  let bS3c2 : Vec (B*(c2*s3h*s3w)) := fun _ => 0
  let bS3c3 : Vec (B*(c3*s3h*s3w)) := fun _ => 0
  let bS4c3 : Vec (B*(c3*s4h*s4w)) := fun _ => 0
  let bS4c4 : Vec (B*(c4*s4h*s4w)) := fun _ => 0
  let bPc4  : Vec (B*(c4*h*w)) := fun _ => 0
  let bD1   : Vec (B*d1) := fun _ => 0
  let bNC   : Vec (B*nClasses) := fun _ => 0
  -- ⚠ `1 * n` is NOT defeq to `n` in Lean, so the `rows := 1` head ops (`softmaxRow`,
  -- `denseRowBack`) need operands declared at exactly their type. Confined to the head:
  -- each `pretty` node is an independent tree, linked to the next only by the SSA name.
  let b1NC  : Vec (B*(1*nClasses)) := fun _ => 0
  let b1D1  : Vec (B*(1*d1)) := fun _ => 0
  let _zTW1 : Tensor3 ic s1h s1w := fun _ _ _ => 0
  let _zTW2 : Tensor3 c1 s1h s1w := fun _ _ _ => 0
  let _zTW3 : Tensor3 c1 s2h s2w := fun _ _ _ => 0
  let _zTW4 : Tensor3 c2 s2h s2w := fun _ _ _ => 0
  let _zTW5 : Tensor3 c2 s3h s3w := fun _ _ _ => 0
  let _zTW6 : Tensor3 c3 s3h s3w := fun _ _ _ => 0
  let _zTW7 : Tensor3 c3 s4h s4w := fun _ _ _ => 0
  let _zTW8 : Tensor3 c4 s4h s4w := fun _ _ _ => 0
  let go : StateM Proofs.StableHLO.EmitS String := do
    -- ═══ forward — identical to cifar8TrainStepFaithfulV, conv biases renamed %cb* ═══
    let (cHc1, nHc1) ← pretty B (if fp8 then .batchOp (N := B) (.convF8 (h := s1h) (w := s1w) zrnd "%W1" "%cb1" W₁ b₁) (.operand "%x" x) else if bf16 then .batchOp (N := B) (.convBf16 (h := s1h) (w := s1w) zrnd "%W1" "%cb1" W₁ b₁) (.operand "%x" x) else .batchOp (N := B) (.conv (h := s1h) (w := s1w) "%W1" "%cb1" W₁ b₁) (.operand "%x" x))
    let (cAc1, nAc1) ← pretty B (.batchOp (N := B) .relu (.operand nHc1 bS1c1))
    let (cHc2, nHc2) ← pretty B (if fp8 then .batchOp (N := B) (.convF8 (h := s1h) (w := s1w) zrnd "%W2" "%cb2" W₂ b₂) (.operand nAc1 bS1c1) else if bf16 then .batchOp (N := B) (.convBf16 (h := s1h) (w := s1w) zrnd "%W2" "%cb2" W₂ b₂) (.operand nAc1 bS1c1) else .batchOp (N := B) (.conv (h := s1h) (w := s1w) "%W2" "%cb2" W₂ b₂) (.operand nAc1 bS1c1))
    let (cAc2, nAc2) ← pretty B (.batchOp (N := B) .relu (.operand nHc2 bS1c1))
    let (cP1, nPool1) ← pretty B (.batchOp (N := B) (.maxPool (c := c1) (h := s2h) (w := s2w)) (.operand nAc2 bS1c1))
    let (cHc3, nHc3) ← pretty B (if fp8 then .batchOp (N := B) (.convF8 (h := s2h) (w := s2w) zrnd "%W3" "%cb3" W₃ b₃) (.operand nPool1 bS2c1) else if bf16 then .batchOp (N := B) (.convBf16 (h := s2h) (w := s2w) zrnd "%W3" "%cb3" W₃ b₃) (.operand nPool1 bS2c1) else .batchOp (N := B) (.conv (h := s2h) (w := s2w) "%W3" "%cb3" W₃ b₃) (.operand nPool1 bS2c1))
    let (cAc3, nAc3) ← pretty B (.batchOp (N := B) .relu (.operand nHc3 bS2c2))
    let (cHc4, nHc4) ← pretty B (if fp8 then .batchOp (N := B) (.convF8 (h := s2h) (w := s2w) zrnd "%W4" "%cb4" W₄ b₄) (.operand nAc3 bS2c2) else if bf16 then .batchOp (N := B) (.convBf16 (h := s2h) (w := s2w) zrnd "%W4" "%cb4" W₄ b₄) (.operand nAc3 bS2c2) else .batchOp (N := B) (.conv (h := s2h) (w := s2w) "%W4" "%cb4" W₄ b₄) (.operand nAc3 bS2c2))
    let (cAc4, nAc4) ← pretty B (.batchOp (N := B) .relu (.operand nHc4 bS2c2))
    let (cP2, nPool2) ← pretty B (.batchOp (N := B) (.maxPool (c := c2) (h := s3h) (w := s3w)) (.operand nAc4 bS2c2))
    let (cHc5, nHc5) ← pretty B (if fp8 then .batchOp (N := B) (.convF8 (h := s3h) (w := s3w) zrnd "%W5" "%cb5" W₅ b₅) (.operand nPool2 bS3c2) else if bf16 then .batchOp (N := B) (.convBf16 (h := s3h) (w := s3w) zrnd "%W5" "%cb5" W₅ b₅) (.operand nPool2 bS3c2) else .batchOp (N := B) (.conv (h := s3h) (w := s3w) "%W5" "%cb5" W₅ b₅) (.operand nPool2 bS3c2))
    let (cAc5, nAc5) ← pretty B (.batchOp (N := B) .relu (.operand nHc5 bS3c3))
    let (cHc6, nHc6) ← pretty B (if fp8 then .batchOp (N := B) (.convF8 (h := s3h) (w := s3w) zrnd "%W6" "%cb6" W₆ b₆) (.operand nAc5 bS3c3) else if bf16 then .batchOp (N := B) (.convBf16 (h := s3h) (w := s3w) zrnd "%W6" "%cb6" W₆ b₆) (.operand nAc5 bS3c3) else .batchOp (N := B) (.conv (h := s3h) (w := s3w) "%W6" "%cb6" W₆ b₆) (.operand nAc5 bS3c3))
    let (cAc6, nAc6) ← pretty B (.batchOp (N := B) .relu (.operand nHc6 bS3c3))
    let (cP3, nPool3) ← pretty B (.batchOp (N := B) (.maxPool (c := c3) (h := s4h) (w := s4w)) (.operand nAc6 bS3c3))
    let (cHc7, nHc7) ← pretty B (if fp8 then .batchOp (N := B) (.convF8 (h := s4h) (w := s4w) zrnd "%W7" "%cb7" W₇ b₇) (.operand nPool3 bS4c3) else if bf16 then .batchOp (N := B) (.convBf16 (h := s4h) (w := s4w) zrnd "%W7" "%cb7" W₇ b₇) (.operand nPool3 bS4c3) else .batchOp (N := B) (.conv (h := s4h) (w := s4w) "%W7" "%cb7" W₇ b₇) (.operand nPool3 bS4c3))
    let (cAc7, nAc7) ← pretty B (.batchOp (N := B) .relu (.operand nHc7 bS4c4))
    let (cHc8, nHc8) ← pretty B (if fp8 then .batchOp (N := B) (.convF8 (h := s4h) (w := s4w) zrnd "%W8" "%cb8" W₈ b₈) (.operand nAc7 bS4c4) else if bf16 then .batchOp (N := B) (.convBf16 (h := s4h) (w := s4w) zrnd "%W8" "%cb8" W₈ b₈) (.operand nAc7 bS4c4) else .batchOp (N := B) (.conv (h := s4h) (w := s4w) "%W8" "%cb8" W₈ b₈) (.operand nAc7 bS4c4))
    let (cAc8, nAc8) ← pretty B (.batchOp (N := B) .relu (.operand nHc8 bS4c4))
    let (cP4, nPool4) ← pretty B (.batchOp (N := B) (.maxPool (c := c4) (h := h) (w := w)) (.operand nAc8 bS4c4))
    let (cH9, nH9) ← pretty B (.batchOp (N := B) (.dense "%W9" "%b9" W₉ b₉) (.operand nPool4 bPc4))
    let (cA9, nA9) ← pretty B (.batchOp (N := B) .relu (.operand nH9 bD1))
    let (cHa, nHa) ← pretty B (.batchOp (N := B) (.dense "%Wa" "%ba" Wa ba) (.operand nA9 bD1))
    let (cAa, nAa) ← pretty B (.batchOp (N := B) .relu (.operand nHa bD1))
    let (cLog, nLog) ← pretty B (.batchOp (N := B) (.dense "%Wb" "%bb" Wb bb) (.operand nAa bD1))
    -- ═══ MEAN-loss cotangent: softmax split out so the report-only %loss can read it ═══
    let (cSm, nSm) ← pretty B (.batchOp (N := B) (.softmaxRow (m := 1) (n := nClasses)) (.operand nLog b1NC))
    let (cD0, nD0) ← pretty B (.subB (.operand nSm bNC) (.operand "%onehot" bNC))
    let (cDy, nDy) ← pretty B (.scaleB invBStr 0 (.operand nD0 bNC))
    -- ═══ backward chain — identical to the SGD render ═══
    let (cDyA, nDyA) ← pretty B (.selectPosB nHa b1D1 (.batchOp (N := B) (.denseRowBack (rows := 1) "%Wb" Wb) (.operand nDy b1NC)))
    let (cDy9, nDy9) ← pretty B (.selectPosB nH9 b1D1 (.batchOp (N := B) (.denseRowBack (rows := 1) "%Wa" Wa) (.operand nDyA b1D1)))
    let (cDx9, nDx9) ← pretty B (.batchOp (N := B) (.denseRowBack (rows := 1) "%W9" W₉) (.operand nDy9 b1D1))
    let (cDac8, nDac8) ← pretty B (.maxPoolBackB (N := B) (c := c4) (h := h) (w := w) nAc8 bS4c4 (.operand nDx9 bPc4))
    let (cDhc8, nDhc8) ← pretty B (.selectPosB nHc8 bS4c4 (.operand nDac8 bS4c4))
    let (cDac7, nDac7) ← pretty B ((if fp8 then SHlo.convBackBatchedF8 (N := B) (h := s4h) (w := s4w) zrnd "%W8" W₈ b₈ else if bf16 then SHlo.convBackBatchedBf16 (N := B) (h := s4h) (w := s4w) zrnd "%W8" W₈ b₈ else SHlo.convBackBatched (N := B) (h := s4h) (w := s4w) "%W8" W₈ b₈) (.operand nDhc8 bS4c4))
    let (cDhc7, nDhc7) ← pretty B (.selectPosB nHc7 bS4c4 (.operand nDac7 bS4c4))
    let (cDpl3, nDpool3) ← pretty B ((if fp8 then SHlo.convBackBatchedF8 (N := B) (h := s4h) (w := s4w) zrnd "%W7" W₇ b₇ else if bf16 then SHlo.convBackBatchedBf16 (N := B) (h := s4h) (w := s4w) zrnd "%W7" W₇ b₇ else SHlo.convBackBatched (N := B) (h := s4h) (w := s4w) "%W7" W₇ b₇) (.operand nDhc7 bS4c4))
    let (cDac6, nDac6) ← pretty B (.maxPoolBackB (N := B) (c := c3) (h := s4h) (w := s4w) nAc6 bS3c3 (.operand nDpool3 bS4c3))
    let (cDhc6, nDhc6) ← pretty B (.selectPosB nHc6 bS3c3 (.operand nDac6 bS3c3))
    let (cDac5, nDac5) ← pretty B ((if fp8 then SHlo.convBackBatchedF8 (N := B) (h := s3h) (w := s3w) zrnd "%W6" W₆ b₆ else if bf16 then SHlo.convBackBatchedBf16 (N := B) (h := s3h) (w := s3w) zrnd "%W6" W₆ b₆ else SHlo.convBackBatched (N := B) (h := s3h) (w := s3w) "%W6" W₆ b₆) (.operand nDhc6 bS3c3))
    let (cDhc5, nDhc5) ← pretty B (.selectPosB nHc5 bS3c3 (.operand nDac5 bS3c3))
    let (cDpl2, nDpool2) ← pretty B ((if fp8 then SHlo.convBackBatchedF8 (N := B) (h := s3h) (w := s3w) zrnd "%W5" W₅ b₅ else if bf16 then SHlo.convBackBatchedBf16 (N := B) (h := s3h) (w := s3w) zrnd "%W5" W₅ b₅ else SHlo.convBackBatched (N := B) (h := s3h) (w := s3w) "%W5" W₅ b₅) (.operand nDhc5 bS3c3))
    let (cDac4, nDac4) ← pretty B (.maxPoolBackB (N := B) (c := c2) (h := s3h) (w := s3w) nAc4 bS2c2 (.operand nDpool2 bS3c2))
    let (cDhc4, nDhc4) ← pretty B (.selectPosB nHc4 bS2c2 (.operand nDac4 bS2c2))
    let (cDac3, nDac3) ← pretty B ((if fp8 then SHlo.convBackBatchedF8 (N := B) (h := s2h) (w := s2w) zrnd "%W4" W₄ b₄ else if bf16 then SHlo.convBackBatchedBf16 (N := B) (h := s2h) (w := s2w) zrnd "%W4" W₄ b₄ else SHlo.convBackBatched (N := B) (h := s2h) (w := s2w) "%W4" W₄ b₄) (.operand nDhc4 bS2c2))
    let (cDhc3, nDhc3) ← pretty B (.selectPosB nHc3 bS2c2 (.operand nDac3 bS2c2))
    let (cDpl1, nDpool1) ← pretty B ((if fp8 then SHlo.convBackBatchedF8 (N := B) (h := s2h) (w := s2w) zrnd "%W3" W₃ b₃ else if bf16 then SHlo.convBackBatchedBf16 (N := B) (h := s2h) (w := s2w) zrnd "%W3" W₃ b₃ else SHlo.convBackBatched (N := B) (h := s2h) (w := s2w) "%W3" W₃ b₃) (.operand nDhc3 bS2c2))
    let (cDac2, nDac2) ← pretty B (.maxPoolBackB (N := B) (c := c1) (h := s2h) (w := s2w) nAc2 bS1c1 (.operand nDpool1 bS2c1))
    let (cDhc2, nDhc2) ← pretty B (.selectPosB nHc2 bS1c1 (.operand nDac2 bS1c1))
    let (cDac1, nDac1) ← pretty B ((if fp8 then SHlo.convBackBatchedF8 (N := B) (h := s1h) (w := s1w) zrnd "%W2" W₂ b₂ else if bf16 then SHlo.convBackBatchedBf16 (N := B) (h := s1h) (w := s1w) zrnd "%W2" W₂ b₂ else SHlo.convBackBatched (N := B) (h := s1h) (w := s1w) "%W2" W₂ b₂) (.operand nDhc2 bS1c1))
    let (cDhc1, nDhc1) ← pretty B (.selectPosB nHc1 bS1c1 (.operand nDac1 bS1c1))
    -- ═══ per param: un-fused gradient, then the three proven AdamW outputs ═══
    let (gW1, sW1) ← pretty B ((if bf16 then SHlo.convWeightGradBBf16 (N := B) zrnd "%x" b₁ bX W₁ else SHlo.convWeightGradB (N := B) "%x" b₁ bX W₁) (.operand nDhc1 bS1c1))
    let (aW1, tW1, mW1, vW1) ← optTail opt B replicas (c1*ic*kH*kW) "%W1" [c1,ic,kH,kW] sW1
    let (gb1, sb1) ← pretty B (.convBiasGradB (N := B) W₁ bX b₁ (.operand nDhc1 bS1c1))
    let (ab1, tb1, mb1, vb1) ← optTail opt B replicas c1 "%cb1" [c1] sb1
    let (gW2, sW2) ← pretty B ((if bf16 then SHlo.convWeightGradBBf16 (N := B) zrnd nAc1 b₂ bS1c1 W₂ else SHlo.convWeightGradB (N := B) nAc1 b₂ bS1c1 W₂) (.operand nDhc2 bS1c1))
    let (aW2, tW2, mW2, vW2) ← optTail opt B replicas (c1*c1*kH*kW) "%W2" [c1,c1,kH,kW] sW2
    let (gb2, sb2) ← pretty B (.convBiasGradB (N := B) W₂ bS1c1 b₂ (.operand nDhc2 bS1c1))
    let (ab2, tb2, mb2, vb2) ← optTail opt B replicas c1 "%cb2" [c1] sb2
    let (gW3, sW3) ← pretty B ((if bf16 then SHlo.convWeightGradBBf16 (N := B) zrnd nPool1 b₃ bS2c1 W₃ else SHlo.convWeightGradB (N := B) nPool1 b₃ bS2c1 W₃) (.operand nDhc3 bS2c2))
    let (aW3, tW3, mW3, vW3) ← optTail opt B replicas (c2*c1*kH*kW) "%W3" [c2,c1,kH,kW] sW3
    let (gb3, sb3) ← pretty B (.convBiasGradB (N := B) W₃ bS2c1 b₃ (.operand nDhc3 bS2c2))
    let (ab3, tb3, mb3, vb3) ← optTail opt B replicas c2 "%cb3" [c2] sb3
    let (gW4, sW4) ← pretty B ((if bf16 then SHlo.convWeightGradBBf16 (N := B) zrnd nAc3 b₄ bS2c2 W₄ else SHlo.convWeightGradB (N := B) nAc3 b₄ bS2c2 W₄) (.operand nDhc4 bS2c2))
    let (aW4, tW4, mW4, vW4) ← optTail opt B replicas (c2*c2*kH*kW) "%W4" [c2,c2,kH,kW] sW4
    let (gb4, sb4) ← pretty B (.convBiasGradB (N := B) W₄ bS2c2 b₄ (.operand nDhc4 bS2c2))
    let (ab4, tb4, mb4, vb4) ← optTail opt B replicas c2 "%cb4" [c2] sb4
    let (gW5, sW5) ← pretty B ((if bf16 then SHlo.convWeightGradBBf16 (N := B) zrnd nPool2 b₅ bS3c2 W₅ else SHlo.convWeightGradB (N := B) nPool2 b₅ bS3c2 W₅) (.operand nDhc5 bS3c3))
    let (aW5, tW5, mW5, vW5) ← optTail opt B replicas (c3*c2*kH*kW) "%W5" [c3,c2,kH,kW] sW5
    let (gb5, sb5) ← pretty B (.convBiasGradB (N := B) W₅ bS3c2 b₅ (.operand nDhc5 bS3c3))
    let (ab5, tb5, mb5, vb5) ← optTail opt B replicas c3 "%cb5" [c3] sb5
    let (gW6, sW6) ← pretty B ((if bf16 then SHlo.convWeightGradBBf16 (N := B) zrnd nAc5 b₆ bS3c3 W₆ else SHlo.convWeightGradB (N := B) nAc5 b₆ bS3c3 W₆) (.operand nDhc6 bS3c3))
    let (aW6, tW6, mW6, vW6) ← optTail opt B replicas (c3*c3*kH*kW) "%W6" [c3,c3,kH,kW] sW6
    let (gb6, sb6) ← pretty B (.convBiasGradB (N := B) W₆ bS3c3 b₆ (.operand nDhc6 bS3c3))
    let (ab6, tb6, mb6, vb6) ← optTail opt B replicas c3 "%cb6" [c3] sb6
    let (gW7, sW7) ← pretty B ((if bf16 then SHlo.convWeightGradBBf16 (N := B) zrnd nPool3 b₇ bS4c3 W₇ else SHlo.convWeightGradB (N := B) nPool3 b₇ bS4c3 W₇) (.operand nDhc7 bS4c4))
    let (aW7, tW7, mW7, vW7) ← optTail opt B replicas (c4*c3*kH*kW) "%W7" [c4,c3,kH,kW] sW7
    let (gb7, sb7) ← pretty B (.convBiasGradB (N := B) W₇ bS4c3 b₇ (.operand nDhc7 bS4c4))
    let (ab7, tb7, mb7, vb7) ← optTail opt B replicas c4 "%cb7" [c4] sb7
    let (gW8, sW8) ← pretty B ((if bf16 then SHlo.convWeightGradBBf16 (N := B) zrnd nAc7 b₈ bS4c4 W₈ else SHlo.convWeightGradB (N := B) nAc7 b₈ bS4c4 W₈) (.operand nDhc8 bS4c4))
    let (aW8, tW8, mW8, vW8) ← optTail opt B replicas (c4*c4*kH*kW) "%W8" [c4,c4,kH,kW] sW8
    let (gb8, sb8) ← pretty B (.convBiasGradB (N := B) W₈ bS4c4 b₈ (.operand nDhc8 bS4c4))
    let (ab8, tb8, mb8, vb8) ← optTail opt B replicas c4 "%cb8" [c4] sb8
    let (gW9, sW9) ← pretty B (.denseWeightGradB (N := B) nPool4 bPc4 (.operand nDy9 bD1))
    let (aW9, tW9, mW9, vW9) ← optTail opt B replicas (flat*d1) "%W9" [flat,d1] sW9
    let (gb9, sb9) ← pretty B (.denseBiasGradB (N := B) (.operand nDy9 bD1))
    let (ab9, tb9, mb9, vb9) ← optTail opt B replicas d1 "%b9" [d1] sb9
    let (gWa, sWa) ← pretty B (.denseWeightGradB (N := B) nA9 bD1 (.operand nDyA bD1))
    let (aWa, tWa, mWa, vWa) ← optTail opt B replicas (d1*d1) "%Wa" [d1,d1] sWa
    let (gba, sba) ← pretty B (.denseBiasGradB (N := B) (.operand nDyA bD1))
    let (aba, tba, mba, vba) ← optTail opt B replicas d1 "%ba" [d1] sba
    let (gWb, sWb) ← pretty B (.denseWeightGradB (N := B) nAa bD1 (.operand nDy bNC))
    let (aWb, tWb, mWb, vWb) ← optTail opt B replicas (d1*nClasses) "%Wb" [d1,nClasses] sWb
    let (gbb, sbb) ← pretty B (.denseBiasGradB (N := B) (.operand nDy bNC))
    let (abb, tbb, mbb, vbb) ← optTail opt B replicas nClasses "%bb" [nClasses] sbb
    -- ═══ report-only scalar loss — OUTSIDE the proven surface, does not feed the update ═══
    let lossCode :=
      "    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it\n" ++
      "    //    feeds no parameter, only the driver's progress line) ──\n" ++
      s!"    %llog = stablehlo.log {nSm} : {ty [B,nClasses]}\n" ++
      s!"    %ohll = stablehlo.multiply %onehot, %llog : {ty [B,nClasses]}\n" ++
      s!"    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,nClasses]}, tensor<f32>) -> tensor<f32>\n" ++
      s!"    %cneg = stablehlo.negate %csum : tensor<f32>\n" ++
      s!"    %lbf = stablehlo.constant dense<{B}.0> : tensor<f32>\n" ++
      s!"    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>\n"
    let body := cHc1 ++ cAc1 ++ cHc2 ++ cAc2 ++ cP1 ++ cHc3 ++ cAc3 ++ cHc4 ++ cAc4 ++ cP2 ++
      cHc5 ++ cAc5 ++ cHc6 ++ cAc6 ++ cP3 ++ cHc7 ++ cAc7 ++ cHc8 ++ cAc8 ++ cP4 ++
      cH9 ++ cA9 ++ cHa ++ cAa ++ cLog ++ cSm ++ cD0 ++ cDy ++ lossCode ++
      cDyA ++ cDy9 ++ cDx9 ++ cDac8 ++ cDhc8 ++ cDac7 ++ cDhc7 ++ cDpl3 ++
      cDac6 ++ cDhc6 ++ cDac5 ++ cDhc5 ++ cDpl2 ++ cDac4 ++ cDhc4 ++ cDac3 ++ cDhc3 ++ cDpl1 ++
      cDac2 ++ cDhc2 ++ cDac1 ++ cDhc1 ++
      gW1 ++ aW1 ++ gb1 ++ ab1 ++ gW2 ++ aW2 ++ gb2 ++ ab2 ++
      gW3 ++ aW3 ++ gb3 ++ ab3 ++ gW4 ++ aW4 ++ gb4 ++ ab4 ++
      gW5 ++ aW5 ++ gb5 ++ ab5 ++ gW6 ++ aW6 ++ gb6 ++ ab6 ++
      gW7 ++ aW7 ++ gb7 ++ ab7 ++ gW8 ++ aW8 ++ gb8 ++ ab8 ++
      gW9 ++ aW9 ++ gb9 ++ ab9 ++ gWa ++ aWa ++ gba ++ aba ++ gWb ++ aWb ++ gbb ++ abb
    let pTys := [ty [c1,ic,kH,kW], ty [c1], ty [c1,c1,kH,kW], ty [c1],
      ty [c2,c1,kH,kW], ty [c2], ty [c2,c2,kH,kW], ty [c2],
      ty [c3,c2,kH,kW], ty [c3], ty [c3,c3,kH,kW], ty [c3],
      ty [c4,c3,kH,kW], ty [c4], ty [c4,c4,kH,kW], ty [c4],
      ty [flat,d1], ty [d1], ty [d1,d1], ty [d1], ty [d1,nClasses], ty [nClasses]]
    let ths := [tW1, tb1, tW2, tb2, tW3, tb3, tW4, tb4, tW5, tb5, tW6, tb6, tW7, tb7, tW8, tb8,
                tW9, tb9, tWa, tba, tWb, tbb]
    let mns := [mW1, mb1, mW2, mb2, mW3, mb3, mW4, mb4, mW5, mb5, mW6, mb6, mW7, mb7, mW8, mb8,
                mW9, mb9, mWa, mba, mWb, mbb]
    let vns := [vW1, vb1, vW2, vb2, vW3, vb3, vW4, vb4, vW5, vb5, vW6, vb6, vW7, vb7, vW8, vb8,
                vW9, vb9, vWa, vba, vWb, vbb]
    pure <|
      "    // ── cifar8 AdamW train step: every line is pretty(verified AST node), except the\n" ++
      "    //    marked report-only loss + the %bc passthroughs ──\n" ++
      -- NB `%sc`/`%sa`/`%sb`/`%sd` are RESERVED: maxPoolBack's select_and_scatter emitter
      -- hardcodes them as region block arguments, and a top-level def of the same name is a
      -- redefinition error at parse time. Hence `%lzero`.
      "    %lzero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      s!"    %b1 = stablehlo.constant dense<{b1Str}> : tensor<f32>\n" ++
      s!"    %ob1 = stablehlo.constant dense<{ob1Str}> : tensor<f32>\n" ++
      s!"    %b2 = stablehlo.constant dense<{b2Str}> : tensor<f32>\n" ++
      s!"    %ob2 = stablehlo.constant dense<{ob2Str}> : tensor<f32>\n" ++
      s!"    %eps = stablehlo.constant dense<{epsStr}> : tensor<f32>\n" ++
      s!"    %wd = stablehlo.constant dense<{wdStr}> : tensor<f32>\n" ++
      -- Emitted ONLY for Nesterov, so the AdamW render re-renders byte-identical after this
      -- threading — the §0 gate-1 self-check that the generalisation is inert.
      (if opt == .nesterov then "    %mu = stablehlo.constant dense<0.9> : tensor<f32>\n" else "") ++
      body ++
      s!"    return {String.intercalate ", " (ths ++ mns ++ vns)}, %loss, %bc1, %bc2 : " ++
      s!"{String.intercalate ", " (pTys ++ pTys ++ pTys)}, tensor<f32>, tensor<f32>, tensor<f32>\n"
  let pSig := s!"%W1: {ty [c1,ic,kH,kW]}, %cb1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %cb2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %cb3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %cb4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %cb5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %cb6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %cb7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %cb8: {ty [c4]}, %W9: {ty [flat,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}"
  let sfx (s : String) : String :=
    s!"%W1{s}: {ty [c1,ic,kH,kW]}, %cb1{s}: {ty [c1]}, %W2{s}: {ty [c1,c1,kH,kW]}, %cb2{s}: {ty [c1]}, %W3{s}: {ty [c2,c1,kH,kW]}, %cb3{s}: {ty [c2]}, %W4{s}: {ty [c2,c2,kH,kW]}, %cb4{s}: {ty [c2]}, %W5{s}: {ty [c3,c2,kH,kW]}, %cb5{s}: {ty [c3]}, %W6{s}: {ty [c3,c3,kH,kW]}, %cb6{s}: {ty [c3]}, %W7{s}: {ty [c4,c3,kH,kW]}, %cb7{s}: {ty [c4]}, %W8{s}: {ty [c4,c4,kH,kW]}, %cb8{s}: {ty [c4]}, %W9{s}: {ty [flat,d1]}, %b9{s}: {ty [d1]}, %Wa{s}: {ty [d1,d1]}, %ba{s}: {ty [d1]}, %Wb{s}: {ty [d1,nClasses]}, %bb{s}: {ty [nClasses]}"
  let pTy := [ty [c1,ic,kH,kW], ty [c1], ty [c1,c1,kH,kW], ty [c1],
    ty [c2,c1,kH,kW], ty [c2], ty [c2,c2,kH,kW], ty [c2],
    ty [c3,c2,kH,kW], ty [c3], ty [c3,c3,kH,kW], ty [c3],
    ty [c4,c3,kH,kW], ty [c4], ty [c4,c4,kH,kW], ty [c4],
    ty [flat,d1], ty [d1], ty [d1,d1], ty [d1], ty [d1,nClasses], ty [nClasses]]
  let retTy := String.intercalate ", " (pTy ++ pTy ++ pTy) ++ ", tensor<f32>, tensor<f32>, tensor<f32>"
  let inner : String := go.run' (0, [])
  -- Entry name tracks the driver's `{slug}_{variant}_train_step` convention (see ResNet34RenderB:
  -- a mismatch here is refused by the shim as "entry mismatch", not silently mis-run).
  let fname := if replicas ≤ 1 then "cifar8b_adam_train_step" else "cifar8b_adamdp_train_step"
  let msfx := sfx "m"
  let vsfx := sfx "v"
  "module @m {\n" ++
  s!"  func.func @{fname}(%x: {ty [B,ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))]}, {pSig}, {msfx}, {vsfx}, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: {ty [B,nClasses]}) -> ({retTy}) " ++ "{\n" ++
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
    `bnPerChannelBack`. `h,w` final pooled; stage spatials `s4=2h…s1=16h`.

    **`opt` selects the optimizer tail (handoff §2i), and it changes the INTERFACE**, unlike the
    no-BN `cifar8AdamTrainStepFaithfulV` where all three variants share one packed signature:

    | `opt` | entry | interface | tail |
    |---|---|---|---|
    | `none` | `@cifar8_bn_train_step` | **40 in / 38 out** | the 38 fused `*Sgd` ops, `lr` a baked literal |
    | `some o` | `@cifar8_bn_{adam,mom,sgd}_train_step` | **119 in / 117 out** | un-fused `*Grad` + `optTail o`, packed `[θ|m|v]`, `%lr` runtime |

    Three things branch, not just the tail — this is why §2i scoped it as more than a tail swap:
    the **cotangent** (fused folds the batch mean into `lrStr`; packed cannot, `lr` is a runtime
    arg, so it emits an explicit `scaleF invB` plus a report-only `%loss`), the **conv bias names**
    (`%b1..%b8` fused, but AdamW bakes β₁/β₂ as `%b1`/`%b2`, so packed renames to `%cb1..%cb8` —
    the same collision-free naming the retired hand-written emitter used), and the
    **signature/return**. Everything else — the whole forward, the whole backward, all 38
    gradients — is shared verbatim, and at `none` the render is byte-identical to the incumbent
    (gate 1), which is what proves the threading inert. -/
def cifar8BnTrainStepFaithfulV (B ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat) (epsStr lrStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses) (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))))
    -- Trailing + defaulted so the existing positional `#eval` is unchanged and `none` keeps
    -- rendering the committed fused artifact. The packed constants are only read at `some _`;
    -- they are parameters rather than literals so the `#eval` states the hyperparameters it is
    -- committing to (§2a-quinquies: a render that bakes its own is how a 16× lr slip shipped).
    (opt : Option CifarOpt := none)
    (invBStr : String := "0.0078125") (b1Str : String := "0.9") (ob1Str : String := "0.1")
    (b2Str : String := "0.999") (ob2Str : String := "0.001") (aEpsStr : String := "1.0e-8")
    (wdStr : String := "0.0001") : String :=
  let s4h := 2*h; let s4w := 2*w
  let s3h := 2*s4h; let s3w := 2*s4w
  let s2h := 2*s3h; let s2w := 2*s3w
  let s1h := 2*s2h; let s1w := 2*s2w
  let flat := c4*h*w
  -- Conv-bias name, the ONE place the two worlds differ (see the docstring): `%b{i}` fused,
  -- `%cb{i}` packed, threaded to the forward, the param tail and the signature alike so they
  -- cannot disagree.
  let cbn : Nat → String := fun i => if opt.isSome then s!"cb{i}" else s!"b{i}"
  let cb : Nat → String := fun i => "%" ++ cbn i
  -- ONE source for the 38 params' ORDER and SHAPES — the arg signature, the packed `m`/`v` blocks
  -- and the return types all read it, so they cannot drift. Order is `cifar8BnVerified.toSpecs`:
  -- (conv W, conv b, BN γ, BN β) ×8, then the 3 dense (W, b).
  let bnSig : List (String × List Nat) :=
    [("W1",[c1,ic,kH,kW]), (cbn 1,[c1]), ("g1",[c1]), ("bt1",[c1]),
     ("W2",[c1,c1,kH,kW]), (cbn 2,[c1]), ("g2",[c1]), ("bt2",[c1]),
     ("W3",[c2,c1,kH,kW]), (cbn 3,[c2]), ("g3",[c2]), ("bt3",[c2]),
     ("W4",[c2,c2,kH,kW]), (cbn 4,[c2]), ("g4",[c2]), ("bt4",[c2]),
     ("W5",[c3,c2,kH,kW]), (cbn 5,[c3]), ("g5",[c3]), ("bt5",[c3]),
     ("W6",[c3,c3,kH,kW]), (cbn 6,[c3]), ("g6",[c3]), ("bt6",[c3]),
     ("W7",[c4,c3,kH,kW]), (cbn 7,[c4]), ("g7",[c4]), ("bt7",[c4]),
     ("W8",[c4,c4,kH,kW]), (cbn 8,[c4]), ("g8",[c4]), ("bt8",[c4]),
     ("W9",[flat,d1]), ("b9",[d1]), ("Wa",[d1,d1]), ("ba",[d1]),
     ("Wb",[d1,nClasses]), ("bb",[nClasses])]
  let pTys : List String := bnSig.map (fun (_, ds) => ty ds)
  let argBlk : String → String := fun s =>
    String.intercalate ", " (bnSig.map (fun (nm, ds) => s!"%{nm}{s}: {ty ds}"))
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
  let go : StateM Proofs.StableHLO.EmitS String := do
    -- ═══ forward (proof-rendered, incl. BN): (conv→BN→relu)×2→pool ×4 → (dense→relu)×2→dense ═══
    let (cHc1, nHc1) ← pretty B (.flatConvF (h := s1h) (w := s1w) "%W1" (cb 1) W₁ b₁ (.operand "%x" x))
    let (cBn1, nBn1) ← pretty B (.bnPerChannelF (oc := c1) (h := s1h) (w := s1w) "%g1" "%bt1" epsStr 0 zVc1 zVc1 (.operand nHc1 zS1c1))
    let (cAc1, nAc1) ← pretty B (.reluF (.operand nBn1 zS1c1))
    let (cHc2, nHc2) ← pretty B (.flatConvF (h := s1h) (w := s1w) "%W2" (cb 2) W₂ b₂ (.operand nAc1 zS1c1))
    let (cBn2, nBn2) ← pretty B (.bnPerChannelF (oc := c1) (h := s1h) (w := s1w) "%g2" "%bt2" epsStr 0 zVc1 zVc1 (.operand nHc2 zS1c1))
    let (cAc2, nAc2) ← pretty B (.reluF (.operand nBn2 zS1c1))
    let (cP1, nPool1) ← pretty B (.maxPoolF (c := c1) (h := s2h) (w := s2w) (.operand nAc2 zS1c1))
    let (cHc3, nHc3) ← pretty B (.flatConvF (h := s2h) (w := s2w) "%W3" (cb 3) W₃ b₃ (.operand nPool1 zS2c1))
    let (cBn3, nBn3) ← pretty B (.bnPerChannelF (oc := c2) (h := s2h) (w := s2w) "%g3" "%bt3" epsStr 0 zVc2 zVc2 (.operand nHc3 zS2c2))
    let (cAc3, nAc3) ← pretty B (.reluF (.operand nBn3 zS2c2))
    let (cHc4, nHc4) ← pretty B (.flatConvF (h := s2h) (w := s2w) "%W4" (cb 4) W₄ b₄ (.operand nAc3 zS2c2))
    let (cBn4, nBn4) ← pretty B (.bnPerChannelF (oc := c2) (h := s2h) (w := s2w) "%g4" "%bt4" epsStr 0 zVc2 zVc2 (.operand nHc4 zS2c2))
    let (cAc4, nAc4) ← pretty B (.reluF (.operand nBn4 zS2c2))
    let (cP2, nPool2) ← pretty B (.maxPoolF (c := c2) (h := s3h) (w := s3w) (.operand nAc4 zS2c2))
    let (cHc5, nHc5) ← pretty B (.flatConvF (h := s3h) (w := s3w) "%W5" (cb 5) W₅ b₅ (.operand nPool2 zS3c2))
    let (cBn5, nBn5) ← pretty B (.bnPerChannelF (oc := c3) (h := s3h) (w := s3w) "%g5" "%bt5" epsStr 0 zVc3 zVc3 (.operand nHc5 zS3c3))
    let (cAc5, nAc5) ← pretty B (.reluF (.operand nBn5 zS3c3))
    let (cHc6, nHc6) ← pretty B (.flatConvF (h := s3h) (w := s3w) "%W6" (cb 6) W₆ b₆ (.operand nAc5 zS3c3))
    let (cBn6, nBn6) ← pretty B (.bnPerChannelF (oc := c3) (h := s3h) (w := s3w) "%g6" "%bt6" epsStr 0 zVc3 zVc3 (.operand nHc6 zS3c3))
    let (cAc6, nAc6) ← pretty B (.reluF (.operand nBn6 zS3c3))
    let (cP3, nPool3) ← pretty B (.maxPoolF (c := c3) (h := s4h) (w := s4w) (.operand nAc6 zS3c3))
    let (cHc7, nHc7) ← pretty B (.flatConvF (h := s4h) (w := s4w) "%W7" (cb 7) W₇ b₇ (.operand nPool3 zS4c3))
    let (cBn7, nBn7) ← pretty B (.bnPerChannelF (oc := c4) (h := s4h) (w := s4w) "%g7" "%bt7" epsStr 0 zVc4 zVc4 (.operand nHc7 zS4c4))
    let (cAc7, nAc7) ← pretty B (.reluF (.operand nBn7 zS4c4))
    let (cHc8, nHc8) ← pretty B (.flatConvF (h := s4h) (w := s4w) "%W8" (cb 8) W₈ b₈ (.operand nAc7 zS4c4))
    let (cBn8, nBn8) ← pretty B (.bnPerChannelF (oc := c4) (h := s4h) (w := s4w) "%g8" "%bt8" epsStr 0 zVc4 zVc4 (.operand nHc8 zS4c4))
    let (cAc8, nAc8) ← pretty B (.reluF (.operand nBn8 zS4c4))
    let (cP4, nPool4) ← pretty B (.maxPoolF (c := c4) (h := h) (w := w) (.operand nAc8 zS4c4))
    let (cH9, nH9) ← pretty B (denseF "%W9" "%b9" W₉ b₉ (.operand nPool4 zPc4))
    let (cA9, nA9) ← pretty B (.reluF (.operand nH9 zD1))
    let (cHa, nHa) ← pretty B (denseF "%Wa" "%ba" Wa ba (.operand nA9 zD1))
    let (cAa, nAa) ← pretty B (.reluF (.operand nHa zD1))
    let (cLog, nLog) ← pretty B (denseF "%Wb" "%bb" Wb bb (.operand nAa zD1))
    -- ═══ loss cotangent — the first of the three things `opt` branches ═══
    -- FUSED: one nested node, plain `softmax − onehot`, with the batch mean folded into `lrStr`
    -- (0.00078125 = 0.1/128). PACKED: `lr` is a RUNTIME arg, so the mean cannot hide inside it —
    -- the softmax is split out (so the report-only `%loss` can read it back by name) and the mean
    -- becomes an explicit `scaleF invB`, which IS proven text, not hand-written. Identical to what
    -- `cifar8AdamTrainStepFaithfulV` does; the retired emitter spelled the same value as
    -- `divide by 128.0` where `scaleF` multiplies by 0.0078125 — exact in binary32 either way,
    -- which is why the tie is numeric rather than byte-for-byte (§2i).
    let (cCot, nDy, lossCode) ← match opt with
      | none => do
        let (c, n) ← pretty B
          (.sub (.softmaxDiv (.expe (.operand nLog zNC))) (.operand "%onehot" zNC))
        pure (c, n, "")
      | some _ => do
        let (cSm, nSm) ← pretty B (.softmaxDiv (.expe (.operand nLog zNC)))
        let (cD0, nD0) ← pretty B (.sub (.operand nSm zNC) (.operand "%onehot" zNC))
        let (cSc, nSc) ← pretty B (.scaleF invBStr 0 (.operand nD0 zNC))
        -- report-only scalar loss — OUTSIDE the proven surface (the kit has no rank-0 loss op),
        -- feeds no parameter. Rendered from the SAME softmax the cotangent uses: §2b shipped plain
        -- CE against a smoothed-CE cotangent here and only a numeric tie caught it, because no
        -- theorem covers a value on no gradient path.
        pure (cSm ++ cD0 ++ cSc, nSc,
          "    // ── report-only scalar loss (NOT pretty(AST): no rank-0 loss op; feeds no\n" ++
          "    //    parameter, only the driver's progress line) ──\n" ++
          s!"    %llog = stablehlo.log {nSm} : {ty [B,nClasses]}\n" ++
          s!"    %ohll = stablehlo.multiply %onehot, %llog : {ty [B,nClasses]}\n" ++
          s!"    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,nClasses]}, tensor<f32>) -> tensor<f32>\n" ++
          s!"    %cneg = stablehlo.negate %csum : tensor<f32>\n" ++
          s!"    %lbf = stablehlo.constant dense<{B}.0> : tensor<f32>\n" ++
          s!"    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>\n")
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
    -- ═══ param tails — the third and last thing `opt` branches ═══
    -- FUSED: the 38 `*Sgd` ops, each fusing the gradient with `θ − lr·g` at a baked literal `lr`.
    -- PACKED: those same 38 gradients as their un-fused `*Grad` peers — all six already exist from
    -- §2a, and `den (xSgd …) = θ − lr · den (xGrad …)` is `rfl` — each feeding `optTail`, which
    -- spends it on the selected optimizer. The param NAME, `n` and `ds` all come from the same
    -- `bnSig` entry the signature is built from, so a shape or a slot cannot drift between the two
    -- (§2e: a misaligned slot is silent, and the m/v slots a tail reads are derived from that name).
    let (cTails, ths, mns, vns) ← match opt with
      | none => do
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
        pure (cW1g ++ cb1g ++ cg1 ++ cbt1 ++ cW2g ++ cb2g ++ cg2 ++ cbt2 ++ cW3g ++ cb3g ++ cg3 ++ cbt3 ++ cW4g ++ cb4g ++ cg4 ++ cbt4 ++ cW5g ++ cb5g ++ cg5 ++ cbt5 ++ cW6g ++ cb6g ++ cg6 ++ cbt6 ++ cW7g ++ cb7g ++ cg7 ++ cbt7 ++ cW8g ++ cb8g ++ cg8 ++ cbt8 ++ cW9 ++ cb9 ++ cWa ++ cba ++ cWb ++ cbb,
          [nW1g, nb1g, ng1, nbt1, nW2g, nb2g, ng2, nbt2,
           nW3g, nb3g, ng3, nbt3, nW4g, nb4g, ng4, nbt4,
           nW5g, nb5g, ng5, nbt5, nW6g, nb6g, ng6, nbt6,
           nW7g, nb7g, ng7, nbt7, nW8g, nb8g, ng8, nbt8,
           nW9, nb9, nWa, nba, nWb, nbb],
          [], [])
      | some o => do
        let (pg0, ps0) ← pretty B (SHlo.convWeightGrad "%x" b₁ zTW1 W₁ (.operand nDhc1 zS1c1))
        let (pa0, pt0, pm0, pv0) ← optTail o B 1 (c1*ic*kH*kW) "%W1" [c1,ic,kH,kW] ps0
        let (pg1, ps1) ← pretty B (SHlo.convBiasGrad W₁ zTW1 b₁ (.operand nDhc1 zS1c1))
        let (pa1, pt1, pm1, pv1) ← optTail o B 1 (c1) (cb 1) [c1] ps1
        let (pg2, ps2) ← pretty B (SHlo.bnGammaGrad nHc1 epsStr 0 zS1c1 (.operand nDbn1 zS1c1))
        let (pa2, pt2, pm2, pv2) ← optTail o B 1 (c1) "%g1" [c1] ps2
        let (pg3, ps3) ← pretty B (SHlo.bnBetaGrad (.operand nDbn1 zS1c1))
        let (pa3, pt3, pm3, pv3) ← optTail o B 1 (c1) "%bt1" [c1] ps3
        let (pg4, ps4) ← pretty B (SHlo.convWeightGrad nAc1 b₂ zTW2 W₂ (.operand nDhc2 zS1c1))
        let (pa4, pt4, pm4, pv4) ← optTail o B 1 (c1*c1*kH*kW) "%W2" [c1,c1,kH,kW] ps4
        let (pg5, ps5) ← pretty B (SHlo.convBiasGrad W₂ zTW2 b₂ (.operand nDhc2 zS1c1))
        let (pa5, pt5, pm5, pv5) ← optTail o B 1 (c1) (cb 2) [c1] ps5
        let (pg6, ps6) ← pretty B (SHlo.bnGammaGrad nHc2 epsStr 0 zS1c1 (.operand nDbn2 zS1c1))
        let (pa6, pt6, pm6, pv6) ← optTail o B 1 (c1) "%g2" [c1] ps6
        let (pg7, ps7) ← pretty B (SHlo.bnBetaGrad (.operand nDbn2 zS1c1))
        let (pa7, pt7, pm7, pv7) ← optTail o B 1 (c1) "%bt2" [c1] ps7
        let (pg8, ps8) ← pretty B (SHlo.convWeightGrad nPool1 b₃ zTW3 W₃ (.operand nDhc3 zS2c2))
        let (pa8, pt8, pm8, pv8) ← optTail o B 1 (c2*c1*kH*kW) "%W3" [c2,c1,kH,kW] ps8
        let (pg9, ps9) ← pretty B (SHlo.convBiasGrad W₃ zTW3 b₃ (.operand nDhc3 zS2c2))
        let (pa9, pt9, pm9, pv9) ← optTail o B 1 (c2) (cb 3) [c2] ps9
        let (pg10, ps10) ← pretty B (SHlo.bnGammaGrad nHc3 epsStr 0 zS2c2 (.operand nDbn3 zS2c2))
        let (pa10, pt10, pm10, pv10) ← optTail o B 1 (c2) "%g3" [c2] ps10
        let (pg11, ps11) ← pretty B (SHlo.bnBetaGrad (.operand nDbn3 zS2c2))
        let (pa11, pt11, pm11, pv11) ← optTail o B 1 (c2) "%bt3" [c2] ps11
        let (pg12, ps12) ← pretty B (SHlo.convWeightGrad nAc3 b₄ zTW4 W₄ (.operand nDhc4 zS2c2))
        let (pa12, pt12, pm12, pv12) ← optTail o B 1 (c2*c2*kH*kW) "%W4" [c2,c2,kH,kW] ps12
        let (pg13, ps13) ← pretty B (SHlo.convBiasGrad W₄ zTW4 b₄ (.operand nDhc4 zS2c2))
        let (pa13, pt13, pm13, pv13) ← optTail o B 1 (c2) (cb 4) [c2] ps13
        let (pg14, ps14) ← pretty B (SHlo.bnGammaGrad nHc4 epsStr 0 zS2c2 (.operand nDbn4 zS2c2))
        let (pa14, pt14, pm14, pv14) ← optTail o B 1 (c2) "%g4" [c2] ps14
        let (pg15, ps15) ← pretty B (SHlo.bnBetaGrad (.operand nDbn4 zS2c2))
        let (pa15, pt15, pm15, pv15) ← optTail o B 1 (c2) "%bt4" [c2] ps15
        let (pg16, ps16) ← pretty B (SHlo.convWeightGrad nPool2 b₅ zTW5 W₅ (.operand nDhc5 zS3c3))
        let (pa16, pt16, pm16, pv16) ← optTail o B 1 (c3*c2*kH*kW) "%W5" [c3,c2,kH,kW] ps16
        let (pg17, ps17) ← pretty B (SHlo.convBiasGrad W₅ zTW5 b₅ (.operand nDhc5 zS3c3))
        let (pa17, pt17, pm17, pv17) ← optTail o B 1 (c3) (cb 5) [c3] ps17
        let (pg18, ps18) ← pretty B (SHlo.bnGammaGrad nHc5 epsStr 0 zS3c3 (.operand nDbn5 zS3c3))
        let (pa18, pt18, pm18, pv18) ← optTail o B 1 (c3) "%g5" [c3] ps18
        let (pg19, ps19) ← pretty B (SHlo.bnBetaGrad (.operand nDbn5 zS3c3))
        let (pa19, pt19, pm19, pv19) ← optTail o B 1 (c3) "%bt5" [c3] ps19
        let (pg20, ps20) ← pretty B (SHlo.convWeightGrad nAc5 b₆ zTW6 W₆ (.operand nDhc6 zS3c3))
        let (pa20, pt20, pm20, pv20) ← optTail o B 1 (c3*c3*kH*kW) "%W6" [c3,c3,kH,kW] ps20
        let (pg21, ps21) ← pretty B (SHlo.convBiasGrad W₆ zTW6 b₆ (.operand nDhc6 zS3c3))
        let (pa21, pt21, pm21, pv21) ← optTail o B 1 (c3) (cb 6) [c3] ps21
        let (pg22, ps22) ← pretty B (SHlo.bnGammaGrad nHc6 epsStr 0 zS3c3 (.operand nDbn6 zS3c3))
        let (pa22, pt22, pm22, pv22) ← optTail o B 1 (c3) "%g6" [c3] ps22
        let (pg23, ps23) ← pretty B (SHlo.bnBetaGrad (.operand nDbn6 zS3c3))
        let (pa23, pt23, pm23, pv23) ← optTail o B 1 (c3) "%bt6" [c3] ps23
        let (pg24, ps24) ← pretty B (SHlo.convWeightGrad nPool3 b₇ zTW7 W₇ (.operand nDhc7 zS4c4))
        let (pa24, pt24, pm24, pv24) ← optTail o B 1 (c4*c3*kH*kW) "%W7" [c4,c3,kH,kW] ps24
        let (pg25, ps25) ← pretty B (SHlo.convBiasGrad W₇ zTW7 b₇ (.operand nDhc7 zS4c4))
        let (pa25, pt25, pm25, pv25) ← optTail o B 1 (c4) (cb 7) [c4] ps25
        let (pg26, ps26) ← pretty B (SHlo.bnGammaGrad nHc7 epsStr 0 zS4c4 (.operand nDbn7 zS4c4))
        let (pa26, pt26, pm26, pv26) ← optTail o B 1 (c4) "%g7" [c4] ps26
        let (pg27, ps27) ← pretty B (SHlo.bnBetaGrad (.operand nDbn7 zS4c4))
        let (pa27, pt27, pm27, pv27) ← optTail o B 1 (c4) "%bt7" [c4] ps27
        let (pg28, ps28) ← pretty B (SHlo.convWeightGrad nAc7 b₈ zTW8 W₈ (.operand nDhc8 zS4c4))
        let (pa28, pt28, pm28, pv28) ← optTail o B 1 (c4*c4*kH*kW) "%W8" [c4,c4,kH,kW] ps28
        let (pg29, ps29) ← pretty B (SHlo.convBiasGrad W₈ zTW8 b₈ (.operand nDhc8 zS4c4))
        let (pa29, pt29, pm29, pv29) ← optTail o B 1 (c4) (cb 8) [c4] ps29
        let (pg30, ps30) ← pretty B (SHlo.bnGammaGrad nHc8 epsStr 0 zS4c4 (.operand nDbn8 zS4c4))
        let (pa30, pt30, pm30, pv30) ← optTail o B 1 (c4) "%g8" [c4] ps30
        let (pg31, ps31) ← pretty B (SHlo.bnBetaGrad (.operand nDbn8 zS4c4))
        let (pa31, pt31, pm31, pv31) ← optTail o B 1 (c4) "%bt8" [c4] ps31
        let (pg32, ps32) ← pretty B (SHlo.weightGrad nPool4 zPc4 (.operand nDy9 zD1))
        let (pa32, pt32, pm32, pv32) ← optTail o B 1 (flat*d1) "%W9" [flat,d1] ps32
        let (pg33, ps33) ← pretty B (SHlo.biasGrad (n := d1) (.operand nDy9 zD1))
        let (pa33, pt33, pm33, pv33) ← optTail o B 1 (d1) "%b9" [d1] ps33
        let (pg34, ps34) ← pretty B (SHlo.weightGrad nA9 zD1 (.operand nDyA zD1))
        let (pa34, pt34, pm34, pv34) ← optTail o B 1 (d1*d1) "%Wa" [d1,d1] ps34
        let (pg35, ps35) ← pretty B (SHlo.biasGrad (n := d1) (.operand nDyA zD1))
        let (pa35, pt35, pm35, pv35) ← optTail o B 1 (d1) "%ba" [d1] ps35
        let (pg36, ps36) ← pretty B (SHlo.weightGrad nAa zD1 (.operand nDy zNC))
        let (pa36, pt36, pm36, pv36) ← optTail o B 1 (d1*nClasses) "%Wb" [d1,nClasses] ps36
        let (pg37, ps37) ← pretty B (SHlo.biasGrad (n := nClasses) (.operand nDy zNC))
        let (pa37, pt37, pm37, pv37) ← optTail o B 1 (nClasses) "%bb" [nClasses] ps37
        pure (String.join [pg0, pa0, pg1, pa1, pg2, pa2, pg3, pa3, pg4, pa4, pg5, pa5, pg6, pa6, pg7, pa7, pg8, pa8, pg9, pa9, pg10, pa10, pg11, pa11, pg12, pa12, pg13, pa13, pg14, pa14, pg15, pa15, pg16, pa16, pg17, pa17, pg18, pa18, pg19, pa19, pg20, pa20, pg21, pa21, pg22, pa22, pg23, pa23, pg24, pa24, pg25, pa25, pg26, pa26, pg27, pa27, pg28, pa28, pg29, pa29, pg30, pa30, pg31, pa31, pg32, pa32, pg33, pa33, pg34, pa34, pg35, pa35, pg36, pa36, pg37, pa37],
          [pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10, pt11, pt12, pt13, pt14, pt15, pt16, pt17, pt18, pt19, pt20, pt21, pt22, pt23, pt24, pt25, pt26, pt27, pt28, pt29, pt30, pt31, pt32, pt33, pt34, pt35, pt36, pt37],
          [pm0, pm1, pm2, pm3, pm4, pm5, pm6, pm7, pm8, pm9, pm10, pm11, pm12, pm13, pm14, pm15, pm16, pm17, pm18, pm19, pm20, pm21, pm22, pm23, pm24, pm25, pm26, pm27, pm28, pm29, pm30, pm31, pm32, pm33, pm34, pm35, pm36, pm37],
          [pv0, pv1, pv2, pv3, pv4, pv5, pv6, pv7, pv8, pv9, pv10, pv11, pv12, pv13, pv14, pv15, pv16, pv17, pv18, pv19, pv20, pv21, pv22, pv23, pv24, pv25, pv26, pv27, pv28, pv29, pv30, pv31, pv32, pv33, pv34, pv35, pv36, pv37])
    let body := cHc1 ++ cBn1 ++ cAc1 ++ cHc2 ++ cBn2 ++ cAc2 ++ cP1 ++
      cHc3 ++ cBn3 ++ cAc3 ++ cHc4 ++ cBn4 ++ cAc4 ++ cP2 ++
      cHc5 ++ cBn5 ++ cAc5 ++ cHc6 ++ cBn6 ++ cAc6 ++ cP3 ++
      cHc7 ++ cBn7 ++ cAc7 ++ cHc8 ++ cBn8 ++ cAc8 ++ cP4 ++
      cH9 ++ cA9 ++ cHa ++ cAa ++ cLog ++ cCot ++ lossCode ++
      cDyA ++ cDy9 ++ cDx9 ++
      cDac8 ++ cDbn8 ++ cDhc8 ++ cDac7 ++ cDbn7 ++ cDhc7 ++ cDpl3 ++
      cDac6 ++ cDbn6 ++ cDhc6 ++ cDac5 ++ cDbn5 ++ cDhc5 ++ cDpl2 ++
      cDac4 ++ cDbn4 ++ cDhc4 ++ cDac3 ++ cDbn3 ++ cDhc3 ++ cDpl1 ++
      cDac2 ++ cDbn2 ++ cDhc2 ++ cDac1 ++ cDbn1 ++ cDhc1 ++ cTails
    -- The return list and its types, from the one `bnSig`-derived source. At `none` the m/v blocks
    -- are empty, which is what keeps this byte-identical to the incumbent.
    let retVals := match opt with
      | none   => ths
      | some _ => ths ++ mns ++ vns ++ ["%loss", "%bc1", "%bc2"]
    let retTys := match opt with
      | none   => pTys
      | some _ => pTys ++ pTys ++ pTys ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"]
    -- The packed constants block. `%lzero` because `%sc`/`%sa`/`%sb`/`%sd` are RESERVED — the
    -- `select_and_scatter` emitter hardcodes them as region block arguments and a top-level
    -- constant of the same name is a redefinition error at XLA parse time, not in Lean (§4).
    -- `%mu` is emitted ONLY for Nesterov, so the other two renders carry no dead constant.
    let constBlk := match opt with
      | none => ""
      | some o =>
        "    %lzero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    %b1 = stablehlo.constant dense<{b1Str}> : tensor<f32>\n" ++
        s!"    %ob1 = stablehlo.constant dense<{ob1Str}> : tensor<f32>\n" ++
        s!"    %b2 = stablehlo.constant dense<{b2Str}> : tensor<f32>\n" ++
        s!"    %ob2 = stablehlo.constant dense<{ob2Str}> : tensor<f32>\n" ++
        s!"    %eps = stablehlo.constant dense<{aEpsStr}> : tensor<f32>\n" ++
        s!"    %wd = stablehlo.constant dense<{wdStr}> : tensor<f32>\n" ++
        (if o == .nesterov then "    %mu = stablehlo.constant dense<0.9> : tensor<f32>\n" else "")
    let hdr := match opt with
      | none   => "    // ── cifar8-bn train step: every line is pretty(verified AST node) ──\n"
      | some _ =>
        "    // ── cifar8-bn train step: every line is pretty(verified AST node), except the\n" ++
        "    //    marked report-only loss + the %bc passthroughs ──\n"
    pure <|
      hdr ++ constBlk ++ body ++
      s!"    return {String.intercalate ", " retVals} : {String.intercalate ", " retTys}\n"
  let inner : String := go.run' (0, [])
  -- Entry name tracks the driver's `{slug}_{variant}_train_step` convention, and a mismatch is
  -- refused by the shim as "entry mismatch" rather than silently running the wrong graph.
  let fname := match opt with
    | none          => "cifar8_bn_train_step"
    | some .adamw   => "cifar8_bn_adam_train_step"
    | some .sgd     => "cifar8_bn_sgd_train_step"
    | some .nesterov => "cifar8_bn_mom_train_step"
  let retTyStr := String.intercalate ", " (match opt with
    | none   => pTys
    | some _ => pTys ++ pTys ++ pTys ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"])
  let argSig := match opt with
    | none   => argBlk "" ++ s!", %onehot: {ty [B,nClasses]}"
    | some _ => argBlk "" ++ ", " ++ argBlk "m" ++ ", " ++ argBlk "v" ++
                s!", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: {ty [B,nClasses]}"
  "module @m {\n" ++
  s!"  func.func @{fname}(%x: {ty [B,ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))]}, {argSig}) -> ({retTyStr}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

/-- The §2i **plain-SGD** cifar8 render: `cifar8AdamTrainStepFaithfulV` with `opt := .sgd`, so the
    forward, backward and all 22 un-fused gradients are shared verbatim with the AdamW render and
    only the tail differs. Entry `@cifar8_sgd_train_step`, 71 in / 69 out. -/
def cifar8SgdTrainStepFaithful : String :=
  cifar8AdamTrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 .sgd
    |>.replace "@cifar8_adam_train_step" "@cifar8_sgd_train_step"

/-- The §2i **Nesterov** cifar8 render (`opt := .nesterov`), μ baked at 0.9.
    Entry `@cifar8_mom_train_step`, 71 in / 69 out. -/
def cifar8MomTrainStepFaithful : String :=
  cifar8AdamTrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 .nesterov
    |>.replace "@cifar8_adam_train_step" "@cifar8_mom_train_step"

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
-- Regenerate `verified_mlir/cifar8_adam_train_step.mlir` — the AdamW peer, same forward/backward
-- with the fused SGD tail replaced by un-fused gradients + the proven AdamW ops
-- (planning/xla_pjrt_handoff.md §2a-ter). Hyperparameters match the retired tests render:
-- β₁ 0.9, β₂ 0.999, ε 1e-8, wd 1e-4; 1/B = 1/128 = 0.0078125 (exact in binary32).
#eval IO.FS.writeFile "verified_mlir/cifar8_adam_train_step.mlir"
  (Proofs.StableHLO.cifar8AdamTrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0))

-- ── §2i: the SGD and NESTEROV peers, staged to their OWN paths ──────────────────────────────
-- ✅ SWAPPED 2026-07-29. These were staged beside the incumbents, tied, and swapped — the
-- hand-written emitters in `tests/TestCifar8AdamTrain.lean` are retired, so each `#eval` below is
-- now its artifact's ONLY writer. `cifar8-opt-tie {sgd,mom}` came back BIT-EXACT on all 52,858
-- recovered gradient coordinates with the m/v passthrough slots bit-exact, and both negative
-- controls fire (÷B 0.0078125→0.008 ⇒ 0.024; μ 0.9→0.91 ⇒ 1.6e-4 with 0/52858 exact).
--
-- The interfaces are IDENTICAL to the AdamW render (71 in / 69 out) because the packed `[θ|m|v]`
-- signature is shared — only the tail moves. `%mu` is baked at 0.9 and `%lr` stays runtime, both
-- matching the retired emitters.
--
-- ⚠ These will NOT be byte-identical to the incumbents: `momVNextF`/`momParamF` are separate SHlo
-- nodes so `v'` is computed twice (SHlo is single-result — `adamWParamF` recomputes `m'`/`v'` the
-- same way), where the retired `emitMomentum` emitted one fused block. XLA's CSE folds it, and
-- §2b-bis measured exactly that pattern costing nothing on R34. Hence the tie is NUMERIC.
#eval IO.FS.writeFile "verified_mlir/cifar8_sgd_train_step.mlir"
  (Proofs.StableHLO.cifar8SgdTrainStepFaithful)
#eval IO.FS.writeFile "verified_mlir/cifar8_mom_train_step.mlir"
  (Proofs.StableHLO.cifar8MomTrainStepFaithful)

-- ── the data-parallel exact gate (handoff §2b-quater) ────────────────────────────────────────
-- cifar8 has NO BatchNorm, so the batch decomposition is an identity and 2 replicas × B=128 with
-- an all_reduce'd gradient must equal 1 device × B=256 to fp rounding. That is the ONLY check
-- that pins the collective's SEMANTICS rather than its syntax; R34 cannot be checked this way
-- (BN normalises per replica). Driven by `cifar8-dp-check`.
--
-- 1/256 = 0.00390625 and 1/128 = 0.0078125 are both exact in binary32, so the loss scaling
-- contributes no rounding of its own to the comparison.
#eval IO.FS.writeFile "verified_mlir/cifar8_adam256_train_step.mlir"
  (Proofs.StableHLO.cifar8AdamTrainStepFaithfulV 256 3 16 16 32 32 2 2 64 10 3 3
    "0.00390625" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0))

#eval IO.FS.writeFile "verified_mlir/cifar8_adamdp_train_step.mlir"
  (Proofs.StableHLO.cifar8AdamTrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) (replicas := 2))

#eval IO.FS.writeFile "verified_mlir/cifar8_train_step.mlir"
  (Proofs.StableHLO.cifar8TrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3 "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0))

-- ═══ bf16 CIFAR — the third precision of the §5.2 optimizer sweep ═══════════════════════════
-- SAME renderers, SAME arguments, `bf16 := true`. Each artifact therefore differs from its f32
-- peer ONLY in the eight forward convs' emit (`flatConvFBf16` vs `flatConvF`) — which is exactly
-- the claim: the OPTIMIZER ORDERING (SGD < AdamW < Nesterov) is invariant under precision. The
-- f32 peers stay byte-identical, which is the gate that the threading is a no-op at the default.
--
-- Slug is `cifar8_bf16`, so `VerifiedTrain.mkSession` resolves
--   `{slug}_train_step.mlir` / `{slug}_{variant}_train_step.mlir` / `{slug}_fwd.mlir`
-- exactly as it does for the f32 and fp8 arms. Func symbols are renamed to match, because the
-- driver calls `m.{slug}_{variant}_train_step` by name.
--
-- ⚠ FORWARD ONLY: cifar8's backward is on the PER-EXAMPLE `convBack`/`dotOut` and the 27 bf16
-- ops were built for ImageNet's BATCHED family, so `convBackBf16`/`dotOutBf16` do not exist.
-- planning/cifar_lowprec_stability.md §4.1 has why the fix is unification, not two new ops.
-- ⚠⚠ NO SPEEDUP, by design — §5.3 measured bf16 at 0.87× across cifar8's conv stack. These
-- artifacts demonstrate that the MATH scales across precision, never the throughput.
#eval IO.FS.writeFile "verified_mlir/cifar8_bf16_train_step.mlir"
  ((Proofs.StableHLO.cifar8TrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3 "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) (bf16 := true)).replace "@cifar8_train_step" "@cifar8_bf16_train_step")

#eval IO.FS.writeFile "verified_mlir/cifar8_bf16_mom_train_step.mlir"
  ((Proofs.StableHLO.cifar8AdamTrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 .nesterov (bf16 := true)).replace "@cifar8_adam_train_step" "@cifar8_bf16_mom_train_step")

-- ═══ the BATCHED render (`…FaithfulB`) — the unification, and FULL bf16 ═══════════════════
-- Emitted from `cifar8AdamTrainStepFaithfulB`, which is on ImageNet's batched op family. Unlike
-- the `…V` artifacts above (bf16 forward convs only), these carry bf16 through the BACKWARD as
-- well — `convBackBatchedBf16` + `convWeightGradBBf16` — because those twins exist for the
-- batched family and not for the per-example one. Zero new verified ops; see §4.1.
-- ⭐ THE FIRST ARTIFACT IN THIS REPO CONTAINING AN f8 TYPE. Forward convs only for now
-- (`convBackBatchedF8` / `convWeightGradBF8` do not exist yet — planning/fp8_in_graph.md §6
-- step 1), and UNSCALED, so this is a lowering probe rather than a trainable arm: E4M3 maxes
-- at 448 and XLA synthesises scale = 1.0 when given no scale operand (§4).
#eval IO.FS.writeFile "verified_mlir/cifar8b_fp8_adam_train_step.mlir"
  ((Proofs.StableHLO.cifar8AdamTrainStepFaithfulB 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 .adamw (bf16 := false) (fp8 := true)).replace
      "@cifar8b_adam_train_step" "@cifar8b_fp8_adam_train_step")

#eval IO.FS.writeFile "verified_mlir/cifar8b_adam_train_step.mlir"
  (Proofs.StableHLO.cifar8AdamTrainStepFaithfulB 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 .adamw)

-- ⚠ The entry symbol MUST match the file stem: `regen_verified_mlir.sh check` fails any
-- artifact whose declared `@name` differs from its path, because the driver resolves the
-- entry as `m.{slug}_{variant}_train_step` and so could never load it.
#eval IO.FS.writeFile "verified_mlir/cifar8b_bf16_adam_train_step.mlir"
  ((Proofs.StableHLO.cifar8AdamTrainStepFaithfulB 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 .adamw (bf16 := true)).replace
      "@cifar8b_adam_train_step" "@cifar8b_bf16_adam_train_step")

-- Eval forward for the batched slug. The forward graph is IDENTICAL either way (the batch was
-- always in the MLIR; only the Lean-side type changed), so this is the f32 `cifar8_fwd` renamed
-- rather than a re-render — it cannot drift from what the `…V` arms evaluate against, which is
-- exactly what makes the B-vs-V training comparison a controlled one.
#eval do
  let fwd ← IO.FS.readFile "verified_mlir/cifar8_fwd.mlir"
  IO.FS.writeFile "verified_mlir/cifar8b_fwd.mlir" (fwd.replace "@cifar8_fwd" "@cifar8b_fwd")

#eval IO.FS.writeFile "verified_mlir/cifar8_bf16_adam_train_step.mlir"
  ((Proofs.StableHLO.cifar8AdamTrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 .adamw (bf16 := true)).replace "@cifar8_adam_train_step" "@cifar8_bf16_adam_train_step")

-- The eval forward stays f32 — you train in bf16 and evaluate in f32, so this is the f32
-- `cifar8_fwd` renamed to the bf16 slug rather than a re-render. Making that a copy (not a
-- second renderer call) is deliberate: it cannot drift from the artifact the f32 arm evaluates.
#eval do
  let fwd ← IO.FS.readFile "verified_mlir/cifar8_fwd.mlir"
  IO.FS.writeFile "verified_mlir/cifar8_bf16_fwd.mlir"
    (fwd.replace "@cifar8_fwd" "@cifar8_bf16_fwd")

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

-- ── §2i: the three PACKED BN variants ───────────────────────────────────────────────────────
-- ✅ SWAPPED 2026-07-30. Staged to `_cert` paths, tied against the hand-written incumbents, then
-- swapped onto the canonical names; the three `IO.FS.writeFile` calls in
-- `tests/TestCifar8AdamTrain.lean` are retired (the renders stay as the ties' references), so each
-- `#eval` below is now its artifact's ONLY writer. `cifar8_bn_adam` is the one artifact of §2i's 13
-- that backs a REAL trainer (`cifar8-bn-verified-adam{,-xla}`).
--
-- `cifar8-opt-tie bn_{adam,mom,sgd}`, all three against a BIT-EXACT A-vs-A floor and a
-- semantics-preserving reorder control (the reference render vs itself on the reversed batch):
--   bn_adam  gradient norm-rel 1.0e-6, spread 8/38 — the control's own 8, the SAME param indices
--   bn_mom   gradient norm-rel 1.0e-6, spread 8/38 = the control's 8; `m` passthrough bit-exact
--   bn_sgd   gradient norm-rel 3.8e-5 vs the control's 1.9e-5 (2.0×), spread 11/38 ⊂ the control's 12
-- The 8 are the CONV BIASES, whose gradient `Σ_{b,h,w} dy` is a cancelling reduce over 128·H·W
-- terms — §2f-bis's finding on a second net, and the reason the spread gate is control-relative and
-- not absolute (an absolute 1e-4 per-param bound FAILS the real tie here).
--
-- `lr` is a RUNTIME arg for all three (the cosine+warmup schedule drives it), so unlike the fused
-- render the batch mean cannot fold into it: `invB` = 1/128 = 0.0078125, exact in binary32. The
-- AdamW constants match the retired emitter: β₁ 0.9, β₂ 0.999, ε 1e-8, wd 1e-4; Nesterov μ 0.9.
-- BN ε stays 1e-05, as in the fused render.
--
-- ⚠ These will NOT be byte-identical to the incumbents even where the arithmetic agrees: the
-- retired emitter spelled the batch mean `divide by 128.0` where `scaleF` multiplies by 0.0078125,
-- and `momVNextF`/`momParamF` are separate SHlo nodes so `v'` is computed twice (SHlo is
-- single-result). Hence the tie is NUMERIC — `cifar8-opt-tie bn_{adam,mom,sgd}`.
private def c8bnPacked (opt : Proofs.StableHLO.CifarOpt) : String :=
  Proofs.StableHLO.cifar8BnTrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3 "1.0e-05" "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) (some opt) "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"

#eval IO.FS.writeFile "verified_mlir/cifar8_bn_adam_train_step.mlir" (c8bnPacked .adamw)
#eval IO.FS.writeFile "verified_mlir/cifar8_bn_mom_train_step.mlir" (c8bnPacked .nesterov)
#eval IO.FS.writeFile "verified_mlir/cifar8_bn_sgd_train_step.mlir" (c8bnPacked .sgd)

-- ── §2i: the cifar8-WIDE family — `cifar8` at `d1 := 512`, NOT a second net ─────────────────
-- Measured 2026-07-30: `cifar8{,Bn}wVerified` agree with `cifar8{,Bn}Verified` layer-for-layer up to
-- the head width, and the committed `cifar8w_bn_adam_train_step.mlir` is **byte-identical modulo the
-- entry name** to the width-sweep's `cifar8_bn_512_adam_train_step.mlir`. So all six wide train steps
-- are these same two renderers at 512, and the interfaces match the committed artifacts exactly —
-- 71/69 and 119/117, arg + return types AND names positionally identical, 0 MALFORMED.
--
-- These back the Chapter-5 "bridge" table (`runs/ablation_cifar8w/README.md`): the wide-vs-narrow
-- comparison behind *"head width barely matters — 7.1× the params, accuracy within a point; the
-- depth, not the head, is the lever."* All six cells are load-bearing, and the `cifar8_bn_{d}` width
-- sweep is adam-only so it covers just one of them.
--
-- ✅ SWAPPED 2026-07-30, all six tied (no-BN three BIT-EXACT; BN three at 1e-6/3.4e-5 against a
-- reorder control they match or beat) and each `#eval` is now its artifact's ONLY writer. The entry rename is what the retired writer did too: both
-- renderers emit the narrow slug, and the wide drivers ask for `m.cifar8w[_bn]_<opt>_train_step`.
private def c8wPacked (opt : Proofs.StableHLO.CifarOpt) (entry : String) : String :=
  (Proofs.StableHLO.cifar8AdamTrainStepFaithfulV 128 3 16 16 32 32 2 2 512 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 opt).replace "@cifar8_adam_train_step" s!"@{entry}"

/-- Wide-head (d1=512) peer of `c8wPacked` on the **BATCHED** op family, with a bf16 switch.
    Same net, same hyperparameters, same packed signature as `c8wPacked`; the only differences are
    the op family (`…FaithfulB`) and the `bf16` flag. This is what the §4.3 "Lever 3: precision"
    sweep trains on, so f32 and bf16 come from ONE renderer and differ only in the emit — which is
    what makes that lever a controlled comparison rather than two nets.

    ⚠ Unlike the `…V` bf16 artifacts, bf16 here reaches the BACKWARD too (23/23 convolutions,
    vs 8/23), because the batched family is the one the 27 bf16 ops were built for. -/
private def c8wbPacked (opt : Proofs.StableHLO.CifarOpt) (bf16 : Bool) (entry : String) : String :=
  (Proofs.StableHLO.cifar8AdamTrainStepFaithfulB 128 3 16 16 32 32 2 2 512 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 opt (bf16 := bf16)).replace "@cifar8b_adam_train_step" s!"@{entry}"

#eval IO.FS.writeFile "verified_mlir/cifar8wb_adam_train_step.mlir"      (c8wbPacked .adamw    false "cifar8wb_adam_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8wb_mom_train_step.mlir"       (c8wbPacked .nesterov false "cifar8wb_mom_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8wb_sgd_train_step.mlir"       (c8wbPacked .sgd      false "cifar8wb_sgd_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8wb_bf16adam_train_step.mlir"  (c8wbPacked .adamw    true  "cifar8wb_bf16adam_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8wb_bf16mom_train_step.mlir"   (c8wbPacked .nesterov true  "cifar8wb_bf16mom_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8wb_bf16sgd_train_step.mlir"   (c8wbPacked .sgd      true  "cifar8wb_bf16sgd_train_step")

-- Eval forward for the batched wide slug: the f32 `cifar8w_fwd` renamed. The forward graph does
-- not change with the op family or with bf16 (you train low-precision and evaluate in f32), and
-- copying rather than re-rendering means it cannot drift from what the f32 arms evaluate against.
#eval do
  let fwd ← IO.FS.readFile "verified_mlir/cifar8w_fwd.mlir"
  IO.FS.writeFile "verified_mlir/cifar8wb_fwd.mlir" (fwd.replace "@cifar8w_fwd" "@cifar8wb_fwd")

private def c8wBnPacked (opt : Proofs.StableHLO.CifarOpt) (from_ entry : String) : String :=
  (Proofs.StableHLO.cifar8BnTrainStepFaithfulV 128 3 16 16 32 32 2 2 512 10 3 3
    "1.0e-05" "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) (some opt) "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
   ).replace s!"@{from_}" s!"@{entry}"

#eval IO.FS.writeFile "verified_mlir/cifar8w_adam_train_step.mlir"
  (c8wPacked .adamw "cifar8w_adam_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8w_mom_train_step.mlir"
  (c8wPacked .nesterov "cifar8w_mom_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8w_sgd_train_step.mlir"
  (c8wPacked .sgd "cifar8w_sgd_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8w_bn_adam_train_step.mlir"
  (c8wBnPacked .adamw "cifar8_bn_adam_train_step" "cifar8w_bn_adam_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8w_bn_mom_train_step.mlir"
  (c8wBnPacked .nesterov "cifar8_bn_mom_train_step" "cifar8w_bn_mom_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8w_bn_sgd_train_step.mlir"
  (c8wBnPacked .sgd "cifar8_bn_sgd_train_step" "cifar8w_bn_sgd_train_step")
