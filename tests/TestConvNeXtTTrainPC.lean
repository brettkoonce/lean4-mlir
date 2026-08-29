import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Types

/-! # ConvNeXt-T render CAPSTONE — proof-rendered full [3,3,9,3] train step at the
     committed `convnext_train_step.mlir` signature

The ConvNeXt analogue of `tests/TestViTTrainPC.lean`: the FULL ConvNeXt-T train step
(BS=32, 3×224² → 10, 180 params), forward AND backward cotangent chain proof-rendered
through `pretty` over the very tokens of the proven `convNextFwdGraphTCh`
(`ConvNeXtFullT.lean` — the §2m channel-LN graph), at the committed function's EXACT
signature (param order/shapes, fn `@convnext_train_step`, eps 1.0e-6, lr 0.1,
tanh-GELU, per-channel layer-scale `tensor<c>`, **per-channel LN `tensor<c>`**).

⚠ **§2m moved three things and this file had to move with all three** — it renders an
INDEPENDENT second spelling of the committed artifact, so a stale LN here is not a
harmless doc drift, it is a parity failure. (1) every LN is the real channel LN, on
`h·w` statistics per example over the `c` channels, with a `[c]` affine; (2) the
**stem LN is present**; (3) the **head LN is gone**. The reference's 22 sites are
1 stem + 18 block + 3 downsample.

Forward by token: `flatConvStride4F` (pad-0 left-aligned 4×4/s4 patchify) →
**channel-LN** → 18× [`depthwiseF` → **channel-LN** → `flatConvF` → `geluF` →
`flatConvF` → `layerScaleChF` → `addV`] with **channel-LN**+`flatConvStridedF`
downsamples → `gapF` → `denseF`. One channel-LN site is five tokens:
`transposeF` → `lnRowF`(γ=1,β=0) → `rowScaleF` → `rowBiasF` → `transposeF`.
Backward by token: `dotOut`, the channel-LN input-VJP (`transposeF`×2 → `rowScaleF`
→ `lnRowBack` → `transposeF`), `layerScaleChF` (diagonal — forward token on the
cotangent IS the backward), `convBack`, `geluBack`, `depthwiseBack`,
`convStridedBack` (even-kernel transpose pad [[1,0],[1,0]]), `addV` fan-in.

**The LN γ/β gradients are now PROOF-RENDERED too** — `veclnGammaGrad` and
`rowDenseBiasGrad` under transposes, where the scalar LN needed 16 lines of
hand-emitted x̂ recomputation per site. That is 44 params moving from hand-emitted
to `pretty`, so this capstone got STRICTLY stronger with the flip.
Hand-emitted only: the GAP backward, conv/depthwise/dense W+b grads, per-channel
layer-scale `dγ_c = Σ_{b,h,w} x⊙dy`, the strided W-grads
(2×2/s2 and 4×4/s4 dilate-dy transpose convs — the committed formulations).

Validation (two-sided GPU parity vs the committed trainer) — **PARITY ✓ 2026-07-31 on the
channel-LN artifact: 180 of 180 outputs BIT-IDENTICAL, worst rel-diff 0.0.** Verified to fail:
EPS 1.0e-6 → 1.0e-3 gives 0/180 bit-identical, worst rel 1.46e-2, rc=1 — so the harness
demonstrably separates, which a 180/180 bit-identical PASS otherwise cannot be distinguished
from (§4).

  IREE_BACKEND=rocm lake env lean tests/TestConvNeXtTTrainPC.lean
  HIP_VISIBLE_DEVICES=0 scripts/render_parity.py --fn convnext_train_step \
    --ref verified_mlir/convnext_train_step.mlir --cand /tmp/cnxtpc/train_step.mlir

  ⚠ `render_parity.py` needs `iree-run-module`, which is NOT in this repo's `.venv/bin` (only
  `iree-compile` is). It lives in `../lean4-mlir/.venv/bin` — put BOTH on PATH.
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

-- ════════════ channel LayerNorm (§2m) — five proof-rendered tokens per site ════════════

/-! ⚠ `Nat` multiplication is not definitionally associative and the ambient activation index is
`c*h*h = (c*h)*h`, while the transpose needs `c*(h*h)`. These transport along `Nat.mul_assoc`;
they are casts on the INDEX, not on the value, so `pretty` walks the same tree.
`ConvNeXtChannelLN.den_reassocS` is the proof that this transport is the math's Mat-split
bridge — i.e. that this spelling and `chanLNTensor3` are one function. -/
private def reassoc {c h : Nat} (e : SHlo (c*h*h)) : SHlo (c*(h*h)) := (Nat.mul_assoc c h h) ▸ e
private def unassoc {c h : Nat} (e : SHlo (c*(h*h))) : SHlo (c*h*h) := (Nat.mul_assoc c h h).symm ▸ e

/-- One channel-LN FORWARD site: transpose to `[h·w, c]`, normalise each spatial row over its
    channels at the scalar identities, apply the real `[c]` affine, transpose back. -/
private def lnFwdSite (gN btN xin : String) (c h : Nat) : StateM Proofs.StableHLO.EmitS (String × String) := do
  let (k1, t)  ← pretty BS (.transposeF (m := c) (n := h*h) (reassoc (.operand xin (zV : Vec (c*h*h)))))
  let (k2, n)  ← pretty BS (.lnRowF (m := h*h) (n := c) "%one" "%zero" EPS 0 1 0
                              (.operand t (zV : Vec (h*h*c))))
  let (k3, sc) ← pretty BS (.rowScaleF (m := h*h) (n := c) gN (zV : Vec c)
                              (.operand n (zV : Vec (h*h*c))))
  let (k4, bi) ← pretty BS (.rowBiasF (m := h*h) (n := c) btN (zV : Vec c)
                              (.operand sc (zV : Vec (h*h*c))))
  let (k5, o)  ← pretty BS (.transposeF (m := h*h) (n := c) (.operand bi (zV : Vec (h*h*c))))
  pure (k1 ++ k2 ++ k3 ++ k4 ++ k5, o)

/-- One channel-LN INPUT-VJP site. `xName` is the saved LN input — `lnRowBack` recomputes
    x̂/istd from it rather than saving them. -/
private def lnBackSite (gN xName cot : String) (c h : Nat) : StateM Proofs.StableHLO.EmitS (String × String) := do
  let (k1, xT)  ← pretty BS (.transposeF (m := c) (n := h*h) (reassoc (.operand xName (zV : Vec (c*h*h)))))
  let (k2, dT)  ← pretty BS (.transposeF (m := c) (n := h*h) (reassoc (.operand cot (zV : Vec (c*h*h)))))
  let (k3, da)  ← pretty BS (.rowScaleF (m := h*h) (n := c) gN (zV : Vec c)
                               (.operand dT (zV : Vec (h*h*c))))
  let (k4, dxT) ← pretty BS (.lnRowBack (m := h*h) (n := c) "%one" xT EPS 0 1 zV
                               (.operand da (zV : Vec (h*h*c))))
  let (k5, o)   ← pretty BS (.transposeF (m := h*h) (n := c) (.operand dxT (zV : Vec (h*h*c))))
  pure (k1 ++ k2 ++ k3 ++ k4 ++ k5, o)

/-- One channel-LN site's γ/β gradients — `veclnGammaGrad` / `rowDenseBiasGrad` under the same
    transposes. PROOF-RENDERED, unlike the scalar LN's hand-emitted x̂ recomputation. -/
private def lnParamGradCh (dgr dbe xName cot : String) (c h : Nat) : StateM Proofs.StableHLO.EmitS String := do
  let (k1, xT) ← pretty BS (.transposeF (m := c) (n := h*h) (reassoc (.operand xName (zV : Vec (c*h*h)))))
  let (k2, dT) ← pretty BS (.transposeF (m := c) (n := h*h) (reassoc (.operand cot (zV : Vec (c*h*h)))))
  let (k3, g)  ← pretty BS (.veclnGammaGrad (N := h*h) (D := c) xT EPS 0 (zV : Vec (h*h*c))
                               (.operand dT (zV : Vec (h*h*c))))
  let (k4, b)  ← pretty BS (.rowDenseBiasGrad (N := h*h) (c := c) (.operand dT (zV : Vec (h*h*c))))
  -- `pretty` returns fresh SSA names; `sgd` needs the grads bound to `%d<param>`. A same-shape
  -- reshape is the file's existing no-op rename idiom (cf. `rs4`).
  pure (k1 ++ k2 ++ k3 ++ k4 ++
        s!"    {dgr} = stablehlo.reshape {g} : ({ty [c]}) -> {ty [c]}\n" ++
        s!"    {dbe} = stablehlo.reshape {b} : ({ty [c]}) -> {ty [c]}\n")

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

/-- One ConvNeXt block forward via `pretty` — the `convNextFwdGraphTCh` block tokens
    (`cnxBlockChGraphW`'s shape) at the committed param names. -/
private def fwdBlock (pfx xin : String) (c e h : Nat) : StateM Proofs.StableHLO.EmitS (String × FNames) := do
  let (k1, d) ← pretty BS (.depthwiseF (h := h) (w := h) s!"%{pfx}dW" s!"%{pfx}db" (zD : DepthwiseKernel c 7 7) zV (.operand xin zV))
  let (k2, n) ← lnFwdSite s!"%{pfx}ng" s!"%{pfx}nbt" d c h
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
    StateM Proofs.StableHLO.EmitS (String × String × String × String × String × String) := do
  let (k1, cot_p) ← pretty BS (.layerScaleChF (h := h) (w := h) s!"%{pfx}lg" (zV : Vec c) (.operand dy zV))
  let (k2, cot_g) ← pretty BS (.convBack (h := h) (w := h) s!"%{pfx}pW" (zK : Kernel4 c e 1 1) zV zV (.operand cot_p zV))
  let (k3, cot_e) ← pretty BS (.geluBack b.e (zV : Vec (e*h*h)) (.operand cot_g zV))
  let (k4, cot_n) ← pretty BS (.convBack (h := h) (w := h) s!"%{pfx}eW" (zK : Kernel4 e c 1 1) zV zV (.operand cot_e zV))
  let (k5, cot_d) ← lnBackSite s!"%{pfx}ng" b.d cot_n c h
  let (k6, cot_main) ← pretty BS (.depthwiseBack (h := h) (w := h) s!"%{pfx}dW" (zD : DepthwiseKernel c 7 7) zV zV (.operand cot_d zV))
  let (k7, cot_xin) ← pretty BS (.addV (.operand cot_main (zV : Vec (c*h*h))) (.operand dy zV))
  pure (k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ k7, cot_xin, cot_p, cot_e, cot_n, cot_d)

/-- block param-grad text, given captured fwd names + cotangents. Monadic because the LN γ/β
    grads are now PROOF-RENDERED (`veclnGammaGrad`/`rowDenseBiasGrad`) rather than hand-emitted. -/
private def blockParamGrads (pfx : String) (b : FNames)
    (cot_p cot_e cot_n cot_d dy : String) (c e h : Nat) : StateM Proofs.StableHLO.EmitS String := do
  let ln ← lnParamGradCh s!"%d{pfx}ng" s!"%d{pfx}nbt" b.d cot_n c h
  pure (lsGradCh s!"%d{pfx}lg" b.p dy c h ++
    convWGrad s!"%d{pfx}pW" b.g cot_p e c h ++ biasGrad s!"%d{pfx}pb" cot_p c h ++
    convWGrad s!"%d{pfx}eW" b.n cot_e c e h ++ biasGrad s!"%d{pfx}eb" cot_e e h ++
    ln ++
    dwWGrad s!"%d{pfx}dW" b.xin cot_d c h ++ biasGrad s!"%d{pfx}db" cot_d c h)

/-- Downsample forward via `pretty`: channel LN → 2×2/s2 widening conv
    (`cnxDownChGraphW`'s tokens). Returns (code, LN-out, out). -/
private def fwdDown (pfx xin : String) (ci co h2 : Nat) : StateM Proofs.StableHLO.EmitS (String × String × String) := do
  let (k1, n) ← lnFwdSite s!"%{pfx}ng" s!"%{pfx}nbt" xin ci (2*h2)
  let (k2, o) ← pretty BS (.flatConvStridedF (h := h2) (w := h2) s!"%{pfx}W" s!"%{pfx}b" (zK : Kernel4 co ci 2 2) zV (.operand n zV))
  pure (k1 ++ k2, n, o)

/-- Downsample backward via `pretty`: strided conv input-VJP (`convStridedBack`,
    even-kernel transpose pad) → channel-LN input-VJP. Returns (code, cot-at-LN-out,
    cot-at-downsample-input). -/
private def bwdDown (pfx dy xin : String) (ci co h2 : Nat) :
    StateM Proofs.StableHLO.EmitS (String × String × String) := do
  let (k1, cot_n) ← pretty BS (.convStridedBack (h := h2) (w := h2) s!"%{pfx}W" (zK : Kernel4 co ci 2 2) zV zV (.operand dy (zV : Vec (co*h2*h2))))
  let (k2, cot_x) ← lnBackSite s!"%{pfx}ng" xin cot_n ci (2*h2)
  pure (k1 ++ k2, cot_n, cot_x)

-- ════════════ param list (committed forward order) ════════════

private def blkParams (pfx : String) (c e : Nat) : List (String × String) :=
  [(s!"{pfx}dW", ty [c,1,7,7]), (s!"{pfx}db", ty [c]),
   (s!"{pfx}ng", ty [c]), (s!"{pfx}nbt", ty [c]),
   (s!"{pfx}eW", ty [e,c,1,1]), (s!"{pfx}eb", ty [e]),
   (s!"{pfx}pW", ty [c,e,1,1]), (s!"{pfx}pb", ty [c]),
   (s!"{pfx}lg", ty [c])]

private def allParams : List (String × String) := Id.run do
  let mut ps : List (String × String) :=
    [("psW", ty [96,3,4,4]), ("psb", ty [96]), ("psng", ty [96]), ("psnbt", ty [96])]
  for si in [0:4] do
    let c := dims[si]!
    let e := 4 * c
    for j in [0:depths[si]!] do
      ps := ps ++ blkParams s!"s{si}b{j}" c e
    if si < 3 then
      ps := ps ++ [(s!"d{si}ng", ty [c]), (s!"d{si}nbt", ty [c]),
                   (s!"d{si}W", ty [dims[si+1]!, c, 2, 2]), (s!"d{si}b", ty [dims[si+1]!])]
  -- §2m: NO head LN — the reference's `forward` is patchify → LN → stages → GAP → dense.
  ps := ps ++ [("Wd", ty [768,10]), ("bd", ty [10])]
  return ps

-- ════════════ whole train step ════════════

private def trainStep : String := Id.run do
  let go : StateM Proofs.StableHLO.EmitS String := do
    -- ═══ forward (proof-rendered; the convNextFwdGraphTCh tokens in graph order) ═══
    let (cS, stem) ← pretty BS (.flatConvStride4F (h := 56) (w := 56) "%psW" "%psb"
      (zK : Kernel4 96 3 4 4) zV (.operand "%x" (zV : Vec (3*(2*(2*56))*(2*(2*56))))))
    -- §2m: the stem channel-LN, which the scalar-LN render omitted entirely
    let (cSln, stemN) ← lnFwdSite "%psng" "%psnbt" stem 96 56
    let mut fwd := cS ++ cSln
    let mut cur := stemN
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
    -- head: GAP → dense 768→10 (§2m: no head LN)
    let (cG, gap) ← pretty BS (.gapF (c := 768) (h := 7) (w := 7) (.operand cur zV))
    let (cLog, logits) ← pretty BS (denseF "%Wd" "%bd" (zM : Mat 768 10) zV (.operand gap (zV : Vec 768)))
    -- loss cotangent: (softmax(logits) − onehot)/BS
    let (cSub, dyr) ← pretty BS (.sub (.softmaxDiv (.expe (.operand logits (zV : Vec 10)))) (.operand "%onehot" zV))
    fwd := fwd ++ cG ++ cLog ++ cSub
    -- ═══ backward (proof-rendered chain; param grads hand-emitted inline) ═══
    let (cDd, cot_gap) ← pretty BS (.dotOut "%Wd" (zM : Mat 768 10) (.operand "%dy" zV))
    let mut bwd :=
      s!"    %dy = stablehlo.divide {dyr}, %bsc : {ty [BS, 10]}\n" ++ cDd ++
      s!"    %dWd = stablehlo.dot_general {gap}, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,768]}, {ty [BS,10]}) -> {ty [768,10]}\n" ++
      s!"    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,10]}, tensor<f32>) -> {ty [10]}\n" ++
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
        let pg ← blockParamGrads s!"s{si}b{j}" b cot_p cot_e cot_n cot_d dy c e h
        bwd := bwd ++ code ++ pg
        dy := cot_xin
      -- downsample d{si−1} sits before stage si
      if si > 0 then
        let ci := dims[si-1]!
        let h2 := spats[si]!
        let (code, cot_n, cot_x) ← bwdDown s!"d{si-1}" dy (downIn[si-1]!) ci c h2
        let dln ← lnParamGradCh s!"%dd{si-1}ng" s!"%dd{si-1}nbt" (downIn[si-1]!) cot_n ci (2*h2)
        bwd := bwd ++ code ++
          downWGrad s!"%dd{si-1}W" (downLn[si-1]!) dy ci c h2 ++
          biasGrad s!"%dd{si-1}b" dy c h2 ++ dln
        dy := cot_x
    -- §2m stem: channel-LN back, its γ/β grads, then the patchify W+b grads
    -- (first layer — no input grad). `stem` is the LN's saved input, i.e. the patchify output.
    let (cSb, cot_stem) ← lnBackSite "%psng" stem dy 96 56
    let sln ← lnParamGradCh "%dpsng" "%dpsnbt" stem dy 96 56
    bwd := bwd ++ cSb ++ sln ++
      patchWGrad "%dpsW" cot_stem ++ biasGrad "%dpsb" cot_stem 96 56
    pure (fwd ++ bwd)
  let body : String := go.run' (0, [])
  let upd := String.join (allParams.map (fun (nm, t) => sgd nm t))
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [BS, 3*224*224]) :: allParams.map (fun (nm, t) => s!"%{nm}: {t}") ++ ["%onehot: " ++ ty [BS,10]])
  let retTyL := String.intercalate ", " (allParams.map (fun (_, t) => t))
  let retVals := String.intercalate ", " (allParams.map (fun (nm, _) => s!"%{nm}n"))
  return "module @m {\n" ++ s!"  func.func @convnext_train_step({argSig}) -> ({retTyL}) " ++ "{\n" ++
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    -- §2m: the channel-LN chain normalises with lnRowF at γ=1/β=0 and applies the REAL
    -- per-channel affine with rowScaleF/rowBiasF, so these two are its scalar identities.
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
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
