import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! # ConvNeXt-T render CAPSTONE — proof-rendered full [3,3,9,3] train step at the
     committed `convnext_train_step.mlir` signature

The ConvNeXt analogue of `tests/TestViTTrainPC.lean`: the FULL ConvNeXt-T train step
(BS=32, 3×224² → 10, 180 params), forward AND backward cotangent chain proof-rendered
through `pretty` over the very tokens of the proven `convNextFwdGraphTC`
(`ConvNeXtFullT.lean` — the committed-config graph: the committed render omits the
paper's stem-LN), at the committed function's EXACT signature (param order/shapes,
fn `@convnext_train_step`, eps 1.0e-6, lr 0.1, tanh-GELU, per-channel layer-scale
`tensor<c>`, scalar LN `tensor<f32>`).

Forward by token: `flatConvStride4F` (pad-0 left-aligned 4×4/s4 patchify) → 18×
[`depthwiseF` → `bnF` → `flatConvF` → `geluF` → `flatConvF` → `layerScaleChF` →
`addV`] with `bnF`+`flatConvStridedF` downsamples → `gapF` → `bnF` → `denseF`.
Backward by token: `dotOut`, `bnBack`, `layerScaleChF` (diagonal — forward token on
the cotangent IS the backward), `convBack`, `geluBack`, `depthwiseBack`,
`convStridedBack` (even-kernel transpose pad [[1,0],[1,0]]), `addV` fan-in.
Hand-emitted only: the GAP backward, conv/depthwise/dense W+b grads, per-channel
layer-scale `dγ_c = Σ_{b,h,w} x⊙dy`, scalar-LN `dγ/dβ`, the strided W-grads
(2×2/s2 and 4×4/s4 dilate-dy transpose convs — the committed formulations).

Validation (two-sided GPU parity vs the committed trainer):
  IREE_BACKEND=rocm lake env lean tests/TestConvNeXtTTrainPC.lean
  HIP_VISIBLE_DEVICES=0 scripts/render_parity.py --fn convnext_train_step \
    --ref verified_mlir/convnext_train_step.mlir --cand /tmp/cnxtpc/train_step.mlir
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 32
private def EPS : String := "1.0e-6"
private def LR : String := "0.1"

private def depths : Array Nat := #[3, 3, 9, 3]
private def dims   : Array Nat := #[96, 192, 384, 768]
private def spats  : Array Nat := #[56, 28, 14, 7]

-- placeholder values (pretty/emitTok render names only; values are irrelevant)
private def zK {o i kh kw : Nat} : Kernel4 o i kh kw := fun _ _ _ _ => 0
private def zD {c kh kw : Nat} : DepthwiseKernel c kh kw := fun _ _ _ => 0
private def zV {n : Nat} : Vec n := fun _ => 0
private def zM {a b : Nat} : Mat a b := fun _ _ => 0

-- ════════════ hand-emitted tail templates (param grads; NCHW only here) ════════════

/-- flat → NCHW reshape. -/
private def rs4 (o flatN : String) (Cc Hh Ww : Nat) : String :=
  s!"    {o} = stablehlo.reshape {flatN} : ({ty [BS, Cc*Hh*Ww]}) -> {ty [BS,Cc,Hh,Ww]}\n"

/-- 1×1 conv weight-grad (transpose trick), inputs flat. -/
private def convWGrad (o inpFlat dyFlat : String) (ic oc Hh : Nat) : String :=
  rs4 s!"{o}xi" inpFlat ic Hh Hh ++ rs4 s!"{o}di" dyFlat oc Hh Hh ++
  s!"    {o}xt = stablehlo.transpose {o}xi, dims = [1, 0, 2, 3] : ({ty [BS,ic,Hh,Hh]}) -> {ty [ic,BS,Hh,Hh]}\n" ++
  s!"    {o}dt = stablehlo.transpose {o}di, dims = [1, 0, 2, 3] : ({ty [BS,oc,Hh,Hh]}) -> {ty [oc,BS,Hh,Hh]}\n" ++
  s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ic,BS,Hh,Hh]}, {ty [oc,BS,Hh,Hh]}) -> {ty [ic,oc,1,1]}\n" ++
  s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [ic,oc,1,1]}) -> {ty [oc,ic,1,1]}\n"

/-- Depthwise 7×7 weight-grad (batch_group_count=c), inputs flat. -/
private def dwWGrad (o inpFlat dyFlat : String) (c Hh : Nat) : String :=
  rs4 s!"{o}xi" inpFlat c Hh Hh ++ rs4 s!"{o}di" dyFlat c Hh Hh ++
  s!"    {o}xt = stablehlo.transpose {o}xi, dims = [1, 0, 2, 3] : ({ty [BS,c,Hh,Hh]}) -> {ty [c,BS,Hh,Hh]}\n" ++
  s!"    {o}dt = stablehlo.transpose {o}di, dims = [1, 0, 2, 3] : ({ty [BS,c,Hh,Hh]}) -> {ty [c,BS,Hh,Hh]}\n" ++
  s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [c,BS,Hh,Hh]}, {ty [c,BS,Hh,Hh]}) -> {ty [1,c,7,7]}\n" ++
  s!"    {o} = stablehlo.reshape {o}raw : ({ty [1,c,7,7]}) -> {ty [c,1,7,7]}\n"

/-- conv/depthwise bias-grad: reduce flat cotangent over batch+spatial → [oc]. -/
private def biasGrad (o dyFlat : String) (oc Hh : Nat) : String :=
  rs4 s!"{o}i" dyFlat oc Hh Hh ++
  s!"    {o} = stablehlo.reduce({o}i init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Hh]}, tensor<f32>) -> {ty [oc]}\n"

/-- Per-channel layer-scale γ-grad `dγ_c = Σ_{b,h,w} x ⊙ dy`. -/
private def lsGradCh (o xFlat dyFlat : String) (c Hh : Nat) : String :=
  rs4 s!"{o}xi" xFlat c Hh Hh ++ rs4 s!"{o}di" dyFlat c Hh Hh ++
  s!"    {o}p = stablehlo.multiply {o}xi, {o}di : {ty [BS,c,Hh,Hh]}\n" ++
  s!"    {o} = stablehlo.reduce({o}p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,c,Hh,Hh]}, tensor<f32>) -> {ty [c]}\n"

/-- scalar-LN dγ = Σ_{b,k} dy·x̂, dβ = Σ_{b,k} dy; recompute x̂ from the saved LN
    input `inFlat` (per-example mean/var over [1] — `bnF`'s own emission text). -/
private def lnParamGrad (dgr dbe inFlat dyFlat : String) (n : Nat) : String :=
  let tn := ty [BS, n]
  s!"    {dgr}nf = stablehlo.constant dense<{n}.0> : {tn}\n" ++
  s!"    {dgr}ep = stablehlo.constant dense<{EPS}> : {tn}\n" ++
  s!"    {dgr}smr = stablehlo.reduce({inFlat} init: %sc) applies stablehlo.add across dimensions = [1] : ({tn}, tensor<f32>) -> {ty [BS]}\n" ++
  s!"    {dgr}sm = stablehlo.broadcast_in_dim {dgr}smr, dims = [0] : ({ty [BS]}) -> {tn}\n" ++
  s!"    {dgr}mu = stablehlo.divide {dgr}sm, {dgr}nf : {tn}\n" ++
  s!"    {dgr}xc = stablehlo.subtract {inFlat}, {dgr}mu : {tn}\n" ++
  s!"    {dgr}sq = stablehlo.multiply {dgr}xc, {dgr}xc : {tn}\n" ++
  s!"    {dgr}vsr = stablehlo.reduce({dgr}sq init: %sc) applies stablehlo.add across dimensions = [1] : ({tn}, tensor<f32>) -> {ty [BS]}\n" ++
  s!"    {dgr}vs = stablehlo.broadcast_in_dim {dgr}vsr, dims = [0] : ({ty [BS]}) -> {tn}\n" ++
  s!"    {dgr}vr = stablehlo.divide {dgr}vs, {dgr}nf : {tn}\n" ++
  s!"    {dgr}ve = stablehlo.add {dgr}vr, {dgr}ep : {tn}\n" ++
  s!"    {dgr}istd = stablehlo.rsqrt {dgr}ve : {tn}\n" ++
  s!"    {dgr}xh = stablehlo.multiply {dgr}xc, {dgr}istd : {tn}\n" ++
  s!"    {dgr}p = stablehlo.multiply {dyFlat}, {dgr}xh : {tn}\n" ++
  s!"    {dgr} = stablehlo.reduce({dgr}p init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({tn}, tensor<f32>) -> tensor<f32>\n" ++
  s!"    {dbe} = stablehlo.reduce({dyFlat} init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({tn}, tensor<f32>) -> tensor<f32>\n"

/-- 2×2/s2 downsample weight-grad: dilate dy (interior 1 → 2h−1), valid conv
    (x lhs, dilated dy rhs) — the committed `convDownWGrad` formulation. -/
private def downWGrad (o inFlat dyFlat : String) (ci co h2 : Nat) : String :=
  rs4 s!"{o}xi" inFlat ci (2*h2) (2*h2) ++ rs4 s!"{o}di" dyFlat co h2 h2 ++
  s!"    {o}u = stablehlo.pad {o}di, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : ({ty [BS,co,h2,h2]}, tensor<f32>) -> {ty [BS,co,2*h2-1,2*h2-1]}\n" ++
  s!"    {o}xt = stablehlo.transpose {o}xi, dims = [1, 0, 2, 3] : ({ty [BS,ci,2*h2,2*h2]}) -> {ty [ci,BS,2*h2,2*h2]}\n" ++
  s!"    {o}dt = stablehlo.transpose {o}u, dims = [1, 0, 2, 3] : ({ty [BS,co,2*h2-1,2*h2-1]}) -> {ty [co,BS,2*h2-1,2*h2-1]}\n" ++
  s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ci,BS,2*h2,2*h2]}, {ty [co,BS,2*h2-1,2*h2-1]}) -> {ty [ci,co,2,2]}\n" ++
  s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [ci,co,2,2]}) -> {ty [co,ci,2,2]}\n"

/-- 4×4/s4 patchify weight-grad: dilate dy by 4 (interior 3 → 221), valid conv —
    the committed `patchifyWGrad` formulation. -/
private def patchWGrad (o dyFlat : String) : String :=
  rs4 s!"{o}xi" "%x" 3 224 224 ++ rs4 s!"{o}di" dyFlat 96 56 56 ++
  s!"    {o}u = stablehlo.pad {o}di, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 3, 3] : ({ty [BS,96,56,56]}, tensor<f32>) -> {ty [BS,96,221,221]}\n" ++
  s!"    {o}xt = stablehlo.transpose {o}xi, dims = [1, 0, 2, 3] : ({ty [BS,3,224,224]}) -> {ty [3,BS,224,224]}\n" ++
  s!"    {o}dt = stablehlo.transpose {o}u, dims = [1, 0, 2, 3] : ({ty [BS,96,221,221]}) -> {ty [96,BS,221,221]}\n" ++
  s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [3,BS,224,224]}, {ty [96,BS,221,221]}) -> {ty [3,96,4,4]}\n" ++
  s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [3,96,4,4]}) -> {ty [96,3,4,4]}\n"

private def sgd (nm t : String) : String :=
  s!"    %{nm}l = stablehlo.constant dense<{LR}> : {t}\n" ++
  s!"    %{nm}s = stablehlo.multiply %d{nm}, %{nm}l : {t}\n" ++
  s!"    %{nm}n = stablehlo.subtract %{nm}, %{nm}s : {t}\n"

-- ════════════ captured forward names per ConvNeXt block ════════════
private structure FNames where  -- all flat SSA names from `pretty`
  xin : String   -- block input (= the residual skip)
  d : String     -- depthwise out (LN input)
  n : String     -- LN out (expand in)
  e : String     -- expand conv out (gelu pre-activation)
  g : String     -- gelu out (project in)
  p : String     -- project out (layer-scale in)
  bout : String  -- block out (addV)
  deriving Inhabited

/-- One ConvNeXt block forward via `pretty` — the `convNextFwdGraphTC` block tokens
    (`cnxBlockGraphW`'s shape) at the committed param names. -/
private def fwdBlock (pfx xin : String) (c e h : Nat) : StateM Nat (String × FNames) := do
  let (k1, d) ← pretty BS (.depthwiseF (h := h) (w := h) s!"%{pfx}dW" s!"%{pfx}db" (zD : DepthwiseKernel c 7 7) zV (.operand xin zV))
  let (k2, n) ← pretty BS (.bnF s!"%{pfx}ng" s!"%{pfx}nbt" EPS 0 0 0 (.operand d (zV : Vec (c*h*h))))
  let (k3, e') ← pretty BS (.flatConvF (h := h) (w := h) s!"%{pfx}eW" s!"%{pfx}eb" (zK : Kernel4 e c 1 1) zV (.operand n zV))
  let (k4, g) ← pretty BS (.geluF (.operand e' (zV : Vec (e*h*h))))
  let (k5, p) ← pretty BS (.flatConvF (h := h) (w := h) s!"%{pfx}pW" s!"%{pfx}pb" (zK : Kernel4 c e 1 1) zV (.operand g zV))
  let (k6, ls) ← pretty BS (.layerScaleChF (h := h) (w := h) s!"%{pfx}lg" (zV : Vec c) (.operand p zV))
  let (k7, bout) ← pretty BS (.addV (.operand ls (zV : Vec (c*h*h))) (.operand xin zV))
  pure (k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ k7, ⟨xin, d, n, e', g, p, bout⟩)

/-- One ConvNeXt block backward cotangent chain via `pretty`. `dy` = flat cotangent
    at block output (the residual add passes it through to the layer-scale output).
    Layer-scale back = `layerScaleChF` on the cotangent (diagonal). Returns
    (code, cot-at-block-input, cot_p, cot_e, cot_n, cot_d). -/
private def bwdBlock (pfx dy : String) (b : FNames) (c e h : Nat) :
    StateM Nat (String × String × String × String × String × String) := do
  let (k1, cot_p) ← pretty BS (.layerScaleChF (h := h) (w := h) s!"%{pfx}lg" (zV : Vec c) (.operand dy zV))
  let (k2, cot_g) ← pretty BS (.convBack (h := h) (w := h) s!"%{pfx}pW" (zK : Kernel4 c e 1 1) zV zV (.operand cot_p zV))
  let (k3, cot_e) ← pretty BS (.geluBack b.e (zV : Vec (e*h*h)) (.operand cot_g zV))
  let (k4, cot_n) ← pretty BS (.convBack (h := h) (w := h) s!"%{pfx}eW" (zK : Kernel4 e c 1 1) zV zV (.operand cot_e zV))
  let (k5, cot_d) ← pretty BS (.bnBack s!"%{pfx}ng" b.d EPS 0 0 zV (.operand cot_n (zV : Vec (c*h*h))))
  let (k6, cot_main) ← pretty BS (.depthwiseBack (h := h) (w := h) s!"%{pfx}dW" (zD : DepthwiseKernel c 7 7) zV zV (.operand cot_d zV))
  let (k7, cot_xin) ← pretty BS (.addV (.operand cot_main (zV : Vec (c*h*h))) (.operand dy zV))
  pure (k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ k7, cot_xin, cot_p, cot_e, cot_n, cot_d)

/-- block param-grad text (hand-emitted), given captured fwd names + cotangents. -/
private def blockParamGrads (pfx : String) (b : FNames)
    (cot_p cot_e cot_n cot_d dy : String) (c e h : Nat) : String :=
  lsGradCh s!"%d{pfx}lg" b.p dy c h ++
  convWGrad s!"%d{pfx}pW" b.g cot_p e c h ++ biasGrad s!"%d{pfx}pb" cot_p c h ++
  convWGrad s!"%d{pfx}eW" b.n cot_e c e h ++ biasGrad s!"%d{pfx}eb" cot_e e h ++
  lnParamGrad s!"%d{pfx}ng" s!"%d{pfx}nbt" b.d cot_n (c*h*h) ++
  dwWGrad s!"%d{pfx}dW" b.xin cot_d c h ++ biasGrad s!"%d{pfx}db" cot_d c h

/-- Downsample forward via `pretty`: scalar LN → 2×2/s2 widening conv
    (`cnxDownGraphW`'s tokens). Returns (code, LN-out, out). -/
private def fwdDown (pfx xin : String) (ci co h2 : Nat) : StateM Nat (String × String × String) := do
  let (k1, n) ← pretty BS (.bnF s!"%{pfx}ng" s!"%{pfx}nbt" EPS 0 0 0 (.operand xin (zV : Vec (ci*(2*h2)*(2*h2)))))
  let (k2, o) ← pretty BS (.flatConvStridedF (h := h2) (w := h2) s!"%{pfx}W" s!"%{pfx}b" (zK : Kernel4 co ci 2 2) zV (.operand n zV))
  pure (k1 ++ k2, n, o)

/-- Downsample backward via `pretty`: strided conv input-VJP (`convStridedBack`,
    even-kernel transpose pad) → scalar-LN input-VJP. Returns (code, cot-at-LN-out,
    cot-at-downsample-input). -/
private def bwdDown (pfx dy xin : String) (ci co h2 : Nat) :
    StateM Nat (String × String × String) := do
  let (k1, cot_n) ← pretty BS (.convStridedBack (h := h2) (w := h2) s!"%{pfx}W" (zK : Kernel4 co ci 2 2) zV zV (.operand dy (zV : Vec (co*h2*h2))))
  let (k2, cot_x) ← pretty BS (.bnBack s!"%{pfx}ng" xin EPS 0 0 (zV : Vec (ci*(2*h2)*(2*h2))) (.operand cot_n zV))
  pure (k1 ++ k2, cot_n, cot_x)

-- ════════════ param list (committed forward order) ════════════

private def blkParams (pfx : String) (c e : Nat) : List (String × String) :=
  [(s!"{pfx}dW", ty [c,1,7,7]), (s!"{pfx}db", ty [c]),
   (s!"{pfx}ng", "tensor<f32>"), (s!"{pfx}nbt", "tensor<f32>"),
   (s!"{pfx}eW", ty [e,c,1,1]), (s!"{pfx}eb", ty [e]),
   (s!"{pfx}pW", ty [c,e,1,1]), (s!"{pfx}pb", ty [c]),
   (s!"{pfx}lg", ty [c])]

private def allParams : List (String × String) := Id.run do
  let mut ps : List (String × String) := [("psW", ty [96,3,4,4]), ("psb", ty [96])]
  for si in [0:4] do
    let c := dims[si]!
    let e := 4 * c
    for j in [0:depths[si]!] do
      ps := ps ++ blkParams s!"s{si}b{j}" c e
    if si < 3 then
      ps := ps ++ [(s!"d{si}ng", "tensor<f32>"), (s!"d{si}nbt", "tensor<f32>"),
                   (s!"d{si}W", ty [dims[si+1]!, c, 2, 2]), (s!"d{si}b", ty [dims[si+1]!])]
  ps := ps ++ [("hng", "tensor<f32>"), ("hnbt", "tensor<f32>"),
               ("Wd", ty [768,10]), ("bd", ty [10])]
  return ps

-- ════════════ whole train step ════════════

private def trainStep : String := Id.run do
  let go : StateM Nat String := do
    -- ═══ forward (proof-rendered; the convNextFwdGraphTC tokens in graph order) ═══
    let (cS, stem) ← pretty BS (.flatConvStride4F (h := 56) (w := 56) "%psW" "%psb"
      (zK : Kernel4 96 3 4 4) zV (.operand "%x" (zV : Vec (3*(2*(2*56))*(2*(2*56))))))
    let mut fwd := cS
    let mut cur := stem
    let mut blksAll : Array (Array FNames) := #[]
    let mut downLn : Array String := #[]   -- LN-out per downsample
    let mut downIn : Array String := #[]   -- downsample input (stage out)
    for si in [0:4] do
      let c := dims[si]!
      let e := 4 * c
      let h := spats[si]!
      let mut blks : Array FNames := #[]
      for j in [0:depths[si]!] do
        let (code, bn) ← fwdBlock s!"s{si}b{j}" cur c e h
        fwd := fwd ++ code; cur := bn.bout; blks := blks.push bn
      blksAll := blksAll.push blks
      if si < 3 then
        downIn := downIn.push cur
        let (code, n, o) ← fwdDown s!"d{si}" cur c dims[si+1]! spats[si+1]!
        fwd := fwd ++ code; downLn := downLn.push n; cur := o
    -- head: GAP → LN(768) → dense 768→10
    let (cG, gap) ← pretty BS (.gapF (c := 768) (h := 7) (w := 7) (.operand cur zV))
    let (cHn, hn) ← pretty BS (.bnF "%hng" "%hnbt" EPS 0 0 0 (.operand gap (zV : Vec 768)))
    let (cLog, logits) ← pretty BS (denseF "%Wd" "%bd" (zM : Mat 768 10) zV (.operand hn zV))
    -- loss cotangent: (softmax(logits) − onehot)/BS
    let (cSub, dyr) ← pretty BS (.sub (.softmaxDiv (.expe (.operand logits (zV : Vec 10)))) (.operand "%onehot" zV))
    fwd := fwd ++ cG ++ cHn ++ cLog ++ cSub
    -- ═══ backward (proof-rendered chain; param grads hand-emitted inline) ═══
    let (cDd, cot_hn) ← pretty BS (.dotOut "%Wd" (zM : Mat 768 10) (.operand "%dy" zV))
    let (cHnB, cot_gap) ← pretty BS (.bnBack "%hng" gap EPS 0 0 zV (.operand cot_hn (zV : Vec 768)))
    let mut bwd :=
      s!"    %dy = stablehlo.divide {dyr}, %bsc : {ty [BS, 10]}\n" ++ cDd ++ cHnB ++
      s!"    %dWd = stablehlo.dot_general {hn}, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,768]}, {ty [BS,10]}) -> {ty [768,10]}\n" ++
      s!"    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,10]}, tensor<f32>) -> {ty [10]}\n" ++
      lnParamGrad "%dhng" "%dhnbt" gap cot_hn 768 ++
      s!"    %dgi = stablehlo.reshape {cot_gap} : ({ty [BS,768]}) -> {ty [BS,768,1,1]}\n" ++
      s!"    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : ({ty [BS,768,1,1]}) -> {ty [BS,768,7,7]}\n" ++
      s!"    %dgn = stablehlo.constant dense<49.0> : {ty [BS,768,7,7]}\n" ++
      s!"    %dgd = stablehlo.divide %dgb, %dgn : {ty [BS,768,7,7]}\n" ++
      s!"    %dgapf = stablehlo.reshape %dgd : ({ty [BS,768,7,7]}) -> {ty [BS, 768*7*7]}\n"
    let mut dy := "%dgapf"
    for si' in [0:4] do
      let si := 3 - si'
      let c := dims[si]!
      let e := 4 * c
      let h := spats[si]!
      -- blocks of stage si, last → first
      for j' in [0:depths[si]!] do
        let j := depths[si]! - 1 - j'
        let b := (blksAll[si]!)[j]!
        let (code, cot_xin, cot_p, cot_e, cot_n, cot_d) ← bwdBlock s!"s{si}b{j}" dy b c e h
        bwd := bwd ++ code ++ blockParamGrads s!"s{si}b{j}" b cot_p cot_e cot_n cot_d dy c e h
        dy := cot_xin
      -- downsample d{si−1} sits before stage si
      if si > 0 then
        let ci := dims[si-1]!
        let h2 := spats[si]!
        let (code, cot_n, cot_x) ← bwdDown s!"d{si-1}" dy (downIn[si-1]!) ci c h2
        bwd := bwd ++ code ++
          downWGrad s!"%dd{si-1}W" (downLn[si-1]!) dy ci c h2 ++
          biasGrad s!"%dd{si-1}b" dy c h2 ++
          lnParamGrad s!"%dd{si-1}ng" s!"%dd{si-1}nbt" (downIn[si-1]!) cot_n (ci*(2*h2)*(2*h2))
        dy := cot_x
    -- stem (4×4/s4 patchify): W+b grads (first layer — no input grad)
    bwd := bwd ++ patchWGrad "%dpsW" dy ++ biasGrad "%dpsb" dy 96 56
    pure (fwd ++ bwd)
  let body : String := go.run' 0
  let upd := String.join (allParams.map (fun (nm, t) => sgd nm t))
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [BS, 3*224*224]) :: allParams.map (fun (nm, t) => s!"%{nm}: {t}") ++ ["%onehot: " ++ ty [BS,10]])
  let retTyL := String.intercalate ", " (allParams.map (fun (_, t) => t))
  let retVals := String.intercalate ", " (allParams.map (fun (nm, _) => s!"%{nm}n"))
  return "module @m {\n" ++ s!"  func.func @convnext_train_step({argSig}) -> ({retTyL}) " ++ "{\n" ++
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    s!"    %bsc = stablehlo.constant dense<{BS}.0> : {ty [BS,10]}\n" ++
    body ++ upd ++
    s!"    return {retVals} : {retTyL}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := trainStep
  IO.println s!"rendered structured ConvNeXt-T FULL train step: {mlir.length} chars, {allParams.length} params"
  IO.FS.createDirAll "/tmp/cnxtpc"
  IO.FS.writeFile "/tmp/cnxtpc/train_step.mlir" mlir
  let cargs ← ireeCompileArgs "/tmp/cnxtpc/train_step.mlir" "/tmp/cnxtpc/train_step.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 5000}"
  else
    IO.println "structured ConvNeXt-T FULL train step iree-compile OK → /tmp/cnxtpc/train_step.mlir"

#eval main
