import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Types
import LeanMlir.ViTRender      -- `emitGradAllReduce`, the data-parallel collective (a carve-out)

/-! # ConvNeXt-T train step rendered ENTIRELY from the verified AST (the §1 render)

The ConvNeXt peer of `MobileNetV2Render`/`EfficientNetRender`: the FULL [3,3,9,3] ConvNeXt-T train
step (BS=32, 3×224²→10) rendered as `pretty` of verified `SHlo` nodes — forward, backward-cotangent
chain, AND the param-SGD tail (the new `ConvNeXtFaithfulPoC` ops + the existing conv/depthwise/dense
ops). Adapted from the committed emitter `tests/TestConvNeXtTTrainPC.lean`: its forward + backward
cotangent chain were already `pretty(SHlo)`; here the hand-written param-GRAD strings are replaced by
the SHlo param-SGD ops, which BUNDLE the gradient + SGD wrap into one op (producing the updated param).

**The two weight-gradient residuals are CLOSED (2026-07-28); all 180 params are now SHlo ops.**
They were never the same kind of gap:
* the **stem 4×4/s4 weight** (`psW`) needed a genuinely missing cert. `flatConvStride4` (forward)
  and `flatConvStride4_has_vjp` (input) already existed; `flatConvStride4_weight_grad_has_vjp` is
  new — two `vjp_comp` steps over the stride-1 weight-VJP and the two decimations, mirroring the
  stride-2 sibling. It backs the new `.convStride4WeightGrad` op.
* the **2×2/s2 downsample** (`d{i}W`) needed NO new cert.
  `flatConvStride2_weight_grad_has_vjp` is kernel-generic and `.convStridedWeightGrad` already
  existed; the blocker was purely emit-side — `(kH−1)/2` symmetric SAME padding floors to 0 at
  `kH = 2` and emitted a 1×1 convolution against a declared 2×2 result, i.e. type-invalid MLIR.
  `StableHLO.sWGradGeom` splits odd/even (odd byte-for-byte unchanged) and the site is now certified.

*(A note here used to claim the **scalar-LN γ/β** params render as `tensor<1xf32>` against a
committed `tensor<f32>` signature. Checked 2026-07-29 and **retired: neither is true of any committed
artifact.** `ty [] = "tensor<f32>"`, and `grep -c 'tensor<1xf32>'` is **0** in both
`convnext_train_step.mlir` and `convnext_adam_train_step.mlir` — the scalar params are `tensor<f32>`
on both sides. Handoff §0b repeats the stale claim.)*

Every other param (depthwise-7×7 W/b, 1×1 expand/project W/b, per-channel layer-scale γ, scalar-LN
γ/β, downsample 2×2 W/b, dense W/b) denotes the certified loss-descent step (`ConvNeXtFaithfulPoC` +
`ConvNeXtClose`/M2/M3). Render is value-independent (`skel` erases values), so placeholders + `lr:=0`
are passed; the emitted `lrStr`/`epsStr` literals carry the real values. -/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO

private def cBS : Nat := 32
private def cEPS : String := "1.0e-6"
private def cLR : String := "0.1"
private def cDepths : Array Nat := #[3, 3, 9, 3]
private def cDims   : Array Nat := #[96, 192, 384, 768]
private def cSpats  : Array Nat := #[56, 28, 14, 7]

private def zK {o i kh kw : Nat} : Kernel4 o i kh kw := fun _ _ _ _ => 0
private def zD {c kh kw : Nat} : DepthwiseKernel c kh kw := fun _ _ _ => 0
private def zV {n : Nat} : Vec n := fun _ => 0
private def zM {a b : Nat} : Mat a b := fun _ _ => 0
private def zT {c h w : Nat} : Tensor3 c h w := fun _ _ _ => 0

-- ── The two hand-written weight-grad emitters that used to live here (`rs4`, `patchWGrad`
--    for the 4×4/s4 patchify stem, `downWGrad` for the even-kernel 2×2/s2 downsample) are
--    DELETED, not left dormant — a retired emitter that can still be called is one more
--    thing to drift (§2b-quater). Both are now certified `SHlo` ops:
--      * `psW` → `.convStride4WeightGrad`, `den` = the NEW `flatConvStride4_weight_grad_has_vjp`;
--      * `d{i}W` → `.convStridedWeightGrad` at 2×2, which needed no new cert at all — only
--        `StableHLO.sWGradGeom`, the emitter's odd/even padding split.
--    Recover from `git show 5920848:LeanMlir/Proofs/Codegen/ConvNeXtRender.lean` if needed.

/-- The SGD update wrap for ConvNeXt's stem weight, taking the gradient's SSA name explicitly.
    (Its predecessor `sgd` read a hardcoded `%d{nm}`, which only worked for the hand-written
    emitters that chose that name; the certified `SHlo` ops emit a fresh `%vN`. Deleted with them.) -/
private def sgdOf (gradN nm t : String) : String :=
  s!"    %{nm}l = stablehlo.constant dense<{cLR}> : {t}\n" ++
  s!"    %{nm}s = stablehlo.multiply {gradN}, %{nm}l : {t}\n" ++
  s!"    %{nm}n = stablehlo.subtract %{nm}, %{nm}s : {t}\n"

-- ════════════════════════════════════════════════════════════════
-- § CHANNEL LayerNorm (§2m) — the real one, assembled from ViT's proven row-LN family
-- ════════════════════════════════════════════════════════════════

/-! **What was wrong.** Every LN site here normalises with `.bnF` ⇒ `bnForward n ε γ β`, which takes
ONE mean and ONE variance over the whole `c·h·w` map per example and applies a **scalar** γ/β.
ConvNeXt's `channel_layer_norm` takes `h·w` statistics per example, each over the `c` channels at
one spatial position, with a **per-channel `[c]`** affine — a different function, on 21 of the 22
sites. (The 22nd, the head, runs after GAP where there is no spatial extent left, so its axis is
already right; only its affine is wrong.) §2m first recorded the axis as correct by matching the
literal `across dimensions = [1]` against the reference's `axis=1`, but the artifact's tensor is
rank-2 `[B, c·h·w]` and the reference's is rank-4 NCHW — §4's one-tensor-layout rule.

**Route A: no new `SHlo` op.** ConvNeXt's channel-LN IS ViT's row-LN under a transpose — view an
example as `[c, s]` with `s = h·w`, transpose to `[s, c]`, and each row is one spatial position
holding its `c` channels, which is exactly what `rowLNFlat` normalises. Settled on device before
any of this was written (`lake build channel-ln`): forward and all three backward pieces tie the
closed form at rel 0, the incumbent `.bnF` control fires at rel 0.82, and the transposes measure
**free** (Δ 0.00 ms on 16.1 ms of whole-net LN — XLA folds a transpose into the consumer's layout).

⚠ **`Nat` multiplication is not definitionally associative**, and the ambient index here is
`c*h*h = (c*h)*h` while the transpose needs `c*(h*h)`. `reassoc`/`unassoc` transport along
`Nat.mul_assoc`; they are casts on the index, not on the value, so `pretty` walks the same tree. -/

private def reassoc {c h : Nat} (e : SHlo (c*h*h)) : SHlo (c*(h*h)) := (Nat.mul_assoc c h h) ▸ e
private def unassoc {c h : Nat} (e : SHlo (c*(h*h))) : SHlo (c*h*h) := (Nat.mul_assoc c h h).symm ▸ e

/-- The `%one`/`%zero` constants the channel-LN chain binds `lnRowF`/`lnRowBack`'s SCALAR γ/β to —
    the real per-channel affine is `rowScaleF`/`rowBiasF` downstream, exactly as ViT does it.
    Emitted once per module body.

    ⚠ This is the enet `zeroBiasPrelude` defect one net over (§2m): wire the operand and forget the
    prelude, and the artifact uses an SSA name nothing defines — `iree-compile`/XLA say "use of
    undeclared SSA value name" and nothing before them says anything. It happened HERE too, on the
    first flag-on render, and `regen_verified_mlir.sh check`'s prelude audit is what generalises. -/
private def chLnPrelude : String :=
    "    // §2m: the channel-LN chain normalises with lnRowF at γ=1/β=0 and applies the REAL\n" ++
    "    // per-channel affine with rowScaleF/rowBiasF, so these two are its scalar identities.\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n"

/-- One **LayerNorm forward** site — ConvNeXt's real channel-LN (§2m): transpose to `[h·w, c]`,
    normalise each spatial row over its channels at the scalar identities `%one`/`%zero`, apply
    the real `[c]` affine, transpose back.

    §2n deleted the `chLN : Bool` flag this used to carry. Its `false` branch emitted the RETIRED
    scalar-global `.bnF` with rank-0 γ/β, and it had no caller: both `#eval` writers took the
    default, and `ConvNeXtTiePoC` — which ties that spelling — does not import this file, it works
    over its own math mirrors. It corresponded to the ch9 §1a tie; it was never used by it. -/
private def lnFwdSite (gN btN xin : String) (c h : Nat) :
    StateM Nat (String × String) := do
    let (k1, t)  ← pretty cBS (.transposeF (m := c) (n := h*h)
                                  (reassoc (.operand xin (zV : Vec (c*h*h)))))
    let (k2, n)  ← pretty cBS (.lnRowF (m := h*h) (n := c) "%one" "%zero" cEPS 0 1 0
                                  (.operand t (zV : Vec (h*h*c))))
    let (k3, sc) ← pretty cBS (.rowScaleF (m := h*h) (n := c) gN (zV : Vec c)
                                  (.operand n (zV : Vec (h*h*c))))
    let (k4, bi) ← pretty cBS (.rowBiasF (m := h*h) (n := c) btN (zV : Vec c)
                                  (.operand sc (zV : Vec (h*h*c))))
    let (k5, o)  ← pretty cBS (.transposeF (m := h*h) (n := c) (.operand bi (zV : Vec (h*h*c))))
    pure (k1 ++ k2 ++ k3 ++ k4 ++ k5, o)

/-- One **LayerNorm input-VJP** site: `dx = transposeᵀ (lnRowBack γ=1 (rowScale γ dyᵀ))`. `xName`
    is the saved LN INPUT — `lnRowBack` recomputes x̂/istd from it rather than saving them. -/
private def lnBackSite (gN xName cot : String) (c h : Nat) :
    StateM Nat (String × String) := do
    let (k1, xT)  ← pretty cBS (.transposeF (m := c) (n := h*h)
                                   (reassoc (.operand xName (zV : Vec (c*h*h)))))
    let (k2, dT)  ← pretty cBS (.transposeF (m := c) (n := h*h)
                                   (reassoc (.operand cot (zV : Vec (c*h*h)))))
    let (k3, da)  ← pretty cBS (.rowScaleF (m := h*h) (n := c) gN (zV : Vec c)
                                   (.operand dT (zV : Vec (h*h*c))))
    let (k4, dxT) ← pretty cBS (.lnRowBack (m := h*h) (n := c) "%one" xT cEPS 0 1 zV
                                   (.operand da (zV : Vec (h*h*c))))
    let (k5, o)   ← pretty cBS (.transposeF (m := h*h) (n := c) (.operand dxT (zV : Vec (h*h*c))))
    pure (k1 ++ k2 ++ k3 ++ k4 ++ k5, o)

/-- The **γ tail** for one LN site — the per-channel `veclnGamma{Grad,Sgd}`. ⚠ The transposes of `xName`/`cot` are re-emitted here rather
    than threaded from `lnBackSite`: `pretty` has no CSE (§4), but XLA does, and §2b-bis measured
    that it collapses exactly this kind of duplicated subtree. -/
private def lnGammaTail (adam : Bool) (gN xName cot : String) (c h : Nat) :
    StateM Nat (String × String) := do
    let (k1, xT) ← pretty cBS (.transposeF (m := c) (n := h*h)
                                  (reassoc (.operand xName (zV : Vec (c*h*h)))))
    let (k2, dT) ← pretty cBS (.transposeF (m := c) (n := h*h)
                                  (reassoc (.operand cot (zV : Vec (c*h*h)))))
    let (k3, o) ← if adam then
        pretty cBS (.veclnGammaGrad (N := h*h) (D := c) xT cEPS 0 (zV : Vec (h*h*c))
                       (.operand dT (zV : Vec (h*h*c))))
      else pretty cBS (.veclnGammaSgd (N := h*h) (D := c) gN xT cEPS cLR 0 (zV : Vec (h*h*c))
                          (zV : Vec c) 0 (.operand dT (zV : Vec (h*h*c))))
    pure (k1 ++ k2 ++ k3, o)

/-- The **β tail** — the per-channel `rowDenseBias{Grad,Sgd}`. It reduces `dims = [0,1]`, i.e. it contracts the
    BATCH as well as the spatial rows, which is what a shared `[c]` parameter needs. -/
private def lnBetaTail (adam : Bool) (btN cot : String) (c h : Nat) :
    StateM Nat (String × String) := do
    let (k1, dT) ← pretty cBS (.transposeF (m := c) (n := h*h)
                                  (reassoc (.operand cot (zV : Vec (c*h*h)))))
    let (k2, o) ← if adam then
        pretty cBS (.rowDenseBiasGrad (N := h*h) (c := c) (.operand dT (zV : Vec (h*h*c))))
      else pretty cBS (.rowDenseBiasSgd (N := h*h) (c := c) btN cLR (zV : Vec c) 0
                          (.operand dT (zV : Vec (h*h*c))))
    pure (k1 ++ k2, o)

-- ── captured forward names per ConvNeXt block ──
private structure FNames where
  xin : String   -- block input (residual skip / depthwise in)
  d : String     -- depthwise out (LN in)
  n : String     -- LN out (expand in)
  e : String     -- expand conv out (gelu pre-act)
  g : String     -- gelu out (project in)
  p : String     -- project out (layer-scale in)
  bout : String  -- block out (addV)
  deriving Inhabited

-- ── forward + backward-cotangent block helpers (verbatim pretty(SHlo), from the committed emitter) ──
private def fwdBlock (pfx xin : String) (c e h : Nat) : StateM Nat (String × FNames) := do
  let (k1, d) ← pretty cBS (.depthwiseF (h := h) (w := h) s!"%{pfx}dW" s!"%{pfx}db" (zD : DepthwiseKernel c 7 7) zV (.operand xin zV))
  let (k2, n) ← lnFwdSite s!"%{pfx}ng" s!"%{pfx}nbt" d c h
  let (k3, e') ← pretty cBS (.flatConvF (h := h) (w := h) s!"%{pfx}eW" s!"%{pfx}eb" (zK : Kernel4 e c 1 1) zV (.operand n zV))
  let (k4, g) ← pretty cBS (.geluF (.operand e' (zV : Vec (e*h*h))))
  let (k5, p) ← pretty cBS (.flatConvF (h := h) (w := h) s!"%{pfx}pW" s!"%{pfx}pb" (zK : Kernel4 c e 1 1) zV (.operand g zV))
  let (k6, ls) ← pretty cBS (.layerScaleChF (h := h) (w := h) s!"%{pfx}lg" (zV : Vec c) (.operand p zV))
  let (k7, bout) ← pretty cBS (.addV (.operand ls (zV : Vec (c*h*h))) (.operand xin zV))
  pure (k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ k7, ⟨xin, d, n, e', g, p, bout⟩)

private def bwdBlock (pfx dy : String) (b : FNames) (c e h : Nat) :
    StateM Nat (String × String × String × String × String × String) := do
  let (k1, cot_p) ← pretty cBS (.layerScaleChF (h := h) (w := h) s!"%{pfx}lg" (zV : Vec c) (.operand dy zV))
  let (k2, cot_g) ← pretty cBS (.convBack (h := h) (w := h) s!"%{pfx}pW" (zK : Kernel4 c e 1 1) zV zV (.operand cot_p zV))
  let (k3, cot_e) ← pretty cBS (.geluBack b.e (zV : Vec (e*h*h)) (.operand cot_g zV))
  let (k4, cot_n) ← pretty cBS (.convBack (h := h) (w := h) s!"%{pfx}eW" (zK : Kernel4 e c 1 1) zV zV (.operand cot_e zV))
  let (k5, cot_d) ← lnBackSite s!"%{pfx}ng" b.d cot_n c h
  let (k6, cot_main) ← pretty cBS (.depthwiseBack (h := h) (w := h) s!"%{pfx}dW" (zD : DepthwiseKernel c 7 7) zV zV (.operand cot_d zV))
  let (k7, cot_xin) ← pretty cBS (.addV (.operand cot_main (zV : Vec (c*h*h))) (.operand dy zV))
  pure (k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ k7, cot_xin, cot_p, cot_e, cot_n, cot_d)

private def fwdDown (pfx xin : String) (ci co h2 : Nat) : StateM Nat (String × String × String) := do
  let (k1, n) ← lnFwdSite s!"%{pfx}ng" s!"%{pfx}nbt" xin ci (2*h2)
  let (k2, o) ← pretty cBS (.flatConvStridedF (h := h2) (w := h2) s!"%{pfx}W" s!"%{pfx}b" (zK : Kernel4 co ci 2 2) zV (.operand n zV))
  pure (k1 ++ k2, n, o)

private def bwdDown (pfx dy xin : String) (ci co h2 : Nat) :
    StateM Nat (String × String × String) := do
  let (k1, cot_n) ← pretty cBS (.convStridedBack (h := h2) (w := h2) s!"%{pfx}W" (zK : Kernel4 co ci 2 2) zV zV (.operand dy (zV : Vec (co*h2*h2))))
  let (k2, cot_x) ← lnBackSite s!"%{pfx}ng" xin cot_n ci (2*h2)
  pure (k1 ++ k2, cot_n, cot_x)

-- ── param tails via the SHlo ops: the updated param at `adam := false` (the op output IS θ'),
--    the un-fused GRADIENT at `adam := true` (§2f) ──

/-! Every leaf below is one of the `*Sgd`/`*Grad` pairs whose `den`s differ by exactly `θ − lr · ·`
(`*Sgd_eq_grad`, all `rfl`) and whose emits differ by exactly the const-lr/multiply/subtract tail
(`tests/TestBatchedEmitTie.lean`, byte-PREFIX). So one traversal serves both renders.

ConvNeXt is the cheapest of the five nets to thread, because its param tails were **already
factored out** of the cotangent traversal — `bwdBlock` computes cotangents and nothing else, and is
untouched by this. `lrStr` is threaded but unused in `adam` mode: AdamW's learning rate is the
runtime `%lr` argument, not a baked literal. -/

private def blockParamSgd (adam : Bool) (pfx : String) (b : FNames)
    (cot_p cot_e cot_n cot_d dy : String) (c e h : Nat) :
    StateM Nat (String × List (String × String)) := do
  let (cLg, nLg) ← if adam then
      pretty cBS (.layerScaleChGammaGrad (c := c) (h := h) (w := h) b.p (zV : Vec (c*h*h)) (.operand dy zV))
    else pretty cBS (.layerScaleChGammaSgd s!"%{pfx}lg" b.p cLR (zV : Vec (c*h*h)) (zV : Vec c) 0 (.operand dy zV))
  let (cPw, nPw) ← if adam then
      pretty cBS (.convWeightGrad b.g (zV : Vec c) (zT : Tensor3 e h h) (zK : Kernel4 c e 1 1) (.operand cot_p zV))
    else pretty cBS (.convWeightSgd b.g s!"%{pfx}pW" cLR (zV : Vec c) (zT : Tensor3 e h h) (zK : Kernel4 c e 1 1) 0 (.operand cot_p zV))
  let (cPb, nPb) ← if adam then
      pretty cBS (.convBiasGrad (zK : Kernel4 c e 1 1) (zT : Tensor3 e h h) (zV : Vec c) (.operand cot_p zV))
    else pretty cBS (.convBiasSgd s!"%{pfx}pb" cLR (zK : Kernel4 c e 1 1) (zT : Tensor3 e h h) (zV : Vec c) 0 (.operand cot_p zV))
  let (cEw, nEw) ← if adam then
      pretty cBS (.convWeightGrad b.n (zV : Vec e) (zT : Tensor3 c h h) (zK : Kernel4 e c 1 1) (.operand cot_e zV))
    else pretty cBS (.convWeightSgd b.n s!"%{pfx}eW" cLR (zV : Vec e) (zT : Tensor3 c h h) (zK : Kernel4 e c 1 1) 0 (.operand cot_e zV))
  let (cEb, nEb) ← if adam then
      pretty cBS (.convBiasGrad (zK : Kernel4 e c 1 1) (zT : Tensor3 c h h) (zV : Vec e) (.operand cot_e zV))
    else pretty cBS (.convBiasSgd s!"%{pfx}eb" cLR (zK : Kernel4 e c 1 1) (zT : Tensor3 c h h) (zV : Vec e) 0 (.operand cot_e zV))
  let (cNg, nNg) ← lnGammaTail adam s!"%{pfx}ng" b.d cot_n c h
  let (cNb, nNb) ← lnBetaTail adam s!"%{pfx}nbt" cot_n c h
  let (cDw, nDw) ← if adam then
      pretty cBS (.depthwiseWeightGrad b.xin (zV : Vec c) (zT : Tensor3 c h h) (zD : DepthwiseKernel c 7 7) (.operand cot_d zV))
    else pretty cBS (.depthwiseWeightSgd b.xin s!"%{pfx}dW" cLR (zV : Vec c) (zT : Tensor3 c h h) (zD : DepthwiseKernel c 7 7) 0 (.operand cot_d zV))
  let (cDb, nDb) ← if adam then
      pretty cBS (.depthwiseBiasGrad (zD : DepthwiseKernel c 7 7) (zT : Tensor3 c h h) (zV : Vec c) (.operand cot_d zV))
    else pretty cBS (.depthwiseBiasSgd s!"%{pfx}db" cLR (zD : DepthwiseKernel c 7 7) (zT : Tensor3 c h h) (zV : Vec c) 0 (.operand cot_d zV))
  pure (cLg ++ cPw ++ cPb ++ cEw ++ cEb ++ cNg ++ cNb ++ cDw ++ cDb,
    [(s!"{pfx}dW", nDw), (s!"{pfx}db", nDb), (s!"{pfx}ng", nNg), (s!"{pfx}nbt", nNb),
     (s!"{pfx}eW", nEw), (s!"{pfx}eb", nEb), (s!"{pfx}pW", nPw), (s!"{pfx}pb", nPb), (s!"{pfx}lg", nLg)])

private def downParamSgd (adam : Bool) (pfx downLn downIn cot_n dy : String) (ci co h2 : Nat) :
    StateM Nat (String × List (String × String)) := do
  -- dXb (channel-sum) + dXng/dXnbt + dXW: ALL FOUR are now SHlo ops.
  --
  -- **`dXW` used to be the hand-written `downWGrad` — the "even-kernel gap".** It was never a
  -- missing certificate: `flatConvStride2_weight_grad_has_vjp {ic oc h w kH kW}` is kernel-generic
  -- (no parity assumption — it is `vjp_comp (conv2d_weight_grad_has_vjp) decimateFlat`), and this
  -- block's forward and input-VJP already used certified ops at 2×2. The blocker was that
  -- `convStridedWeightGrad`'s EMITTER hardcoded symmetric SAME padding `[[p,p]]` with
  -- `p = (kH−1)/2`, which floors to 0 at `kH = 2` and emitted a 1×1 convolution against a declared
  -- 2×2 result — type-invalid MLIR. `StableHLO.sWGradGeom` now splits odd/even (odd byte-for-byte
  -- unchanged), so this call site is certified like every other.
  let (cB, nB) ← if adam then
      pretty cBS (.convStridedBiasGrad (zK : Kernel4 co ci 2 2) (zV : Vec (ci*(2*h2)*(2*h2))) (zV : Vec co) (.operand dy zV))
    else pretty cBS (.convStridedBiasSgd s!"%{pfx}b" cLR (zK : Kernel4 co ci 2 2) (zV : Vec (ci*(2*h2)*(2*h2))) (zV : Vec co) 0 (.operand dy zV))
  let (cNg, nNg) ← lnGammaTail adam s!"%{pfx}ng" downIn cot_n ci (2*h2)
  let (cNb, nNb) ← lnBetaTail adam s!"%{pfx}nbt" cot_n ci (2*h2)
  let (wcode, nW) ← if adam then
      pretty cBS (.convStridedWeightGrad (ic := ci) (oc := co) (h := h2) (w := h2) (kH := 2) (kW := 2)
        downLn (zV : Vec co) (zV : Vec (ci*(2*h2)*(2*h2))) (zK : Kernel4 co ci 2 2)
        (.operand dy (zV : Vec (co*h2*h2))))
    else pretty cBS (.convStridedWeightSgd (ic := ci) (oc := co) (h := h2) (w := h2) (kH := 2) (kW := 2)
        downLn s!"%{pfx}W" cLR (zV : Vec co) (zV : Vec (ci*(2*h2)*(2*h2))) (zK : Kernel4 co ci 2 2) 0
        (.operand dy (zV : Vec (co*h2*h2))))
  pure (cB ++ cNg ++ cNb ++ wcode,
    [(s!"{pfx}ng", nNg), (s!"{pfx}nbt", nNb), (s!"{pfx}W", nW), (s!"{pfx}b", nB)])

-- ── full param signature (committed forward order), name + SHAPE ──
/-! Shapes are `List Nat` rather than rendered `tensor<…>` strings because the AdamW render needs
both forms: `%{nm}`/`%{nm}m`/`%{nm}v` for the moment slots and the raw dimensions for the emitted
Adam ops. The scalar-LN γ/β are **rank 0**, i.e. `[]` — and `ty [] = "tensor<f32>"`, exactly the
string that used to be hardcoded, so the SGD render's emitted text is unchanged. -/

/-- ⚠ The LN γ/β are `[c]` — the real channel-LN's per-channel affine (the retired scalar-global
    spelling had them rank-0 `[]`). This list is the func signature AND the return types, so
    getting it wrong is not silent — the body uses `%…ng` at `tensor<{c}xf32>` and the compiler rejects a signature
    that disagrees with itself (the §2l-A lesson: no gate found that one either, the type checker
    did). -/
private def blkParams (pfx : String) (c e : Nat) : List (String × List Nat) :=
  let lnSh : List Nat := [c]
  [(s!"{pfx}dW", [c,1,7,7]), (s!"{pfx}db", [c]),
   (s!"{pfx}ng", lnSh), (s!"{pfx}nbt", lnSh),
   (s!"{pfx}eW", [e,c,1,1]), (s!"{pfx}eb", [e]),
   (s!"{pfx}pW", [c,e,1,1]), (s!"{pfx}pb", [c]),
   (s!"{pfx}lg", [c])]

/-- `nClasses` defaults to 10 (Imagenette) so every existing call site is unchanged; the ImageNet
    renders pass 1000. Only the head moves — 180 of the 182 entries are class-independent. -/
private def allParams (nClasses : Nat := 10) : List (String × List Nat) := Id.run do
  let mut ps : List (String × List Nat) :=
    [("psW", [96,3,4,4]), ("psb", [96]), ("psng", [96]), ("psnbt", [96])]
  for si in [0:4] do
    let c := cDims[si]!
    let e := 4 * c
    for j in [0:cDepths[si]!] do
      ps := ps ++ blkParams s!"s{si}b{j}" c e
    if si < 3 then
      ps := ps ++ [(s!"d{si}ng", [c]), (s!"d{si}nbt", [c]),
                   (s!"d{si}W", [cDims[si+1]!, c, 2, 2]), (s!"d{si}b", [cDims[si+1]!])]
  ps := ps ++ [("Wd", [768,nClasses]), ("bd", [nClasses])]
  return ps

-- ════════════════════════════════════════════════════════════════
-- § The shared forward chain (the forward render and both train steps emit this)
-- ════════════════════════════════════════════════════════════════

/-- Every SSA name the ConvNeXt-T forward produces. `convNextFwdFaithfulV` returns just `logits`;
    the train steps additionally consume the block records, the two downsample names and the
    head's `gap`/`hn` on the way back. -/
private structure CFwd where
  code    : String                  -- stem → 4 stages → GAP → LN → dense, in emission order
  blksAll : Array (Array FNames)    -- the [3,3,9,3] block forwards, stage-major, forward order
  downLn  : Array String            -- the 3 downsample LN outputs (the strided conv's input)
  downIn  : Array String            -- the 3 downsample inputs (the LN's input)
  gap     : String                  -- global-average-pool output
  stemC   : String                  -- stem conv output (= the §2m stem LN's input)
  hn      : String                  -- dense input: `gap` itself (the reference has no head LN)
  logits  : String                  -- dense output
  deriving Inhabited

set_option maxRecDepth 8000 in
/-- **The full ConvNeXt-T `[3,3,9,3]` forward as `pretty` of the verified AST.** 4×4/s4 patchify
    stem (3→96, 224→56) → 4 stages at 56/28/14/7 with 2×2/s2 downsamples between them → GAP(7×7)
    → head LN → dense(768→10). Every emitted line is `pretty` of a verified `SHlo` node.

    **There is no train/eval mode here, and there should not be** — ConvNeXt normalises with
    LayerNorm, which reduces over the channel/spatial axes of ONE example and never over the batch.
    So the forward is already class-batch-independent: train == eval, and `@convnext_fwd` is the
    only forward artifact this net needs (unlike the BN nets, which need a frozen-stats peer). -/
private def convNextFwdChain (nClasses : Nat := 10) : StateM Nat CFwd := do
  let (cS, stemC) ← pretty cBS (.flatConvStride4F (h := 56) (w := 56) "%psW" "%psb"
    (zK : Kernel4 96 3 4 4) zV (.operand "%x" (zV : Vec (3*(2*(2*56))*(2*(2*56))))))
  -- §2m: the reference's `convnext_stem` is patchify conv → channel-LN. The PRE-§2m render had
  -- NO stem LN, and had a head LN the reference does not — the two nearly cancel in the parameter
  -- count (+2×768 − 2×96 = +1,344 out of 28.6M), which is why the count alone never caught it.
  let (cSln, stem) ← lnFwdSite "%psng" "%psnbt" stemC 96 56
  let mut fwd := cS ++ cSln
  let mut cur := stem
  let mut blksAll : Array (Array FNames) := #[]
  let mut downLn : Array String := #[]
  let mut downIn : Array String := #[]
  for si in [0:4] do
    let c := cDims[si]!; let e := 4 * c; let h := cSpats[si]!
    let mut blks : Array FNames := #[]
    for j in [0:cDepths[si]!] do
      let (code, bn) ← fwdBlock s!"s{si}b{j}" cur c e h
      fwd := fwd ++ code; cur := bn.bout; blks := blks.push bn
    blksAll := blksAll.push blks
    if si < 3 then
      downIn := downIn.push cur
      let (code, n, o) ← fwdDown s!"d{si}" cur c cDims[si+1]! cSpats[si+1]!
      fwd := fwd ++ code; downLn := downLn.push n; cur := o
  let (cG, gap) ← pretty cBS (.gapF (c := 768) (h := 7) (w := 7) (.operand cur zV))
  -- §2m: the reference goes GAP → dense with no norm between — the head LN is GONE
  -- (its 2×768 params move to the stem, at 2×96).
  let (cHn, hn) := ("", gap)
  let (cLog, logits) ← pretty cBS (denseF "%Wd" "%bd" (zM : Mat 768 nClasses) zV (.operand hn zV))
  pure { code := fwd ++ cG ++ cHn ++ cLog,
         blksAll := blksAll, downLn := downLn, downIn := downIn,
         gap := gap, stemC := stemC, hn := hn, logits := logits }

set_option maxRecDepth 8000 in
/-- **`@convnext_fwd` rendered ENTIRELY from the verified AST** — the peer of the train-step
    render, sharing its forward chain and its 180-parameter signature. Takes `%x` plus the 180
    params in `allParams` (= func-arg) order (181 inputs) and returns logits `[32, 10]`.

    This replaces the independent hand-written string emitter in `tests/TestConvNeXtFwd.lean`: the
    forward the driver evals is now the same graph the train step differentiates, **by construction
    rather than by inspection**. Because it shares the chain, the emitted body is a byte-identical
    PREFIX of `convnext_train_step.mlir`'s, ending exactly where the loss begins — which is what
    `scripts/regen_verified_mlir.sh check` audits. -/
def convNextFwdFaithfulV (funcName : String := "convnext_fwd") (nClasses : Nat := 10)
    : String := Id.run do
  let F : CFwd := (convNextFwdChain nClasses).run' 0
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [cBS, 3*224*224]) :: (allParams nClasses).map (fun (nm, d) => s!"%{nm}: {ty d}"))
  return "module @m {\n" ++ s!"  func.func @{funcName}({argSig}) -> {ty [cBS,nClasses]} " ++ "{\n" ++
    "    // ── ConvNeXt-T forward: every line is pretty(verified AST node) ──\n" ++
    chLnPrelude ++ F.code ++
    s!"    return {F.logits} : {ty [cBS,nClasses]}\n" ++ "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § The whole-net renderer
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 8000 in
/-- **The whole-net forward + cotangent + backward traversal, SHARED by the SGD and AdamW renders.**

    Returns `(code, params, softmax)`: every emitted line; one SSA per parameter — the **updated
    param** at `adam := false`, the **un-fused gradient** at `adam := true`; and the softmax SSA,
    which the AdamW render's report-only `%loss` reads.

    `smooth` is the cotangent recipe. `none` → the SGD render's `(softmax − onehot)/B`, unchanged
    down to the hand-written `%dy` divide. `some (α, −α/K, B)` → the label-smoothed
    `((softmax − onehot) + α·onehot − α/K)/B` composed from kit ops, which is what the AdamW recipe
    trains on. ConvNeXt already spells the ÷B explicitly (unlike ViT/R34, which fold the mean into
    lr), so here the smoothing is the *only* difference between the two cotangents.

    Gate on the refactor: `convnext_train_step.mlir` must come back byte-identical. It does — the
    softmax is now `pretty`d on its own line instead of nested inside the `.sub` so that `%loss` can
    read it, but `.operand` is a leaf that emits nothing, so the fresh-name sequence is unchanged. -/
private def convNextBackAll (adam : Bool) (smooth : Option (String × String × String) := none)
    (nClasses : Nat := 10) :
    StateM Nat (String × List (String × String) × String) := do
    -- ═══ forward — the SAME chain `convNextFwdFaithfulV` emits, so `@convnext_fwd` and the two
    --     train steps cannot drift into computing different functions (§2a) ═══
    let F : CFwd ← convNextFwdChain nClasses
    let (cSm, nSm) ← pretty cBS (.softmaxDiv (.expe (.operand F.logits (zV : Vec nClasses))))
    let (cSub, dyr) ← pretty cBS (.sub (.operand nSm (zV : Vec nClasses)) (.operand "%onehot" zV))
    let blksAll := F.blksAll
    let downLn := F.downLn
    let downIn := F.downIn
    let hn := F.hn
    let fwd := F.code ++ cSm ++ cSub
    -- ═══ the cotangent. `none` keeps the SGD render's hand-written `%dy` divide byte-for-byte;
    --     `some` appends the label-smoothing chain, every line `pretty` of a verified node. ═══
    let (cDyC, dyName) ← match smooth with
      | none => pure (s!"    %dy = stablehlo.divide {dyr}, %bsc : {ty [cBS, nClasses]}\n", "%dy")
      | some (aStr, negAK, bStr) => do
          let (c1, n1) ← pretty cBS (.scaleF (n := nClasses) aStr 0 (.operand "%onehot" (zV : Vec nClasses)))
          let (c2, n2) ← pretty cBS (.addV (.operand dyr (zV : Vec nClasses)) (.operand n1 (zV : Vec nClasses)))
          -- ⚠ the operands are annotated at `Vec (1 * nClasses)`, not `Vec nClasses`: `shiftB`/
          -- `divConstB` are indexed `SHlo (N*n)`, and `1 * n` reduces definitionally only when `n`
          -- is a literal. It did while this render was pinned at 10; it stops the moment `nClasses`
          -- is a variable. `ViTRender` carries the same annotation for the same reason.
          let (c3, n3) ← pretty cBS (.shiftB (N := 1) (n := nClasses) negAK 0 (.operand n2 (zV : Vec (1 * nClasses))))
          let (c4, n4) ← pretty cBS (.divConstB (N := 1) (n := nClasses) bStr 0 (.operand n3 (zV : Vec (1 * nClasses))))
          pure (c1 ++ c2 ++ c3 ++ c4, n4)
    -- ═══ backward: head cotangent chain + param-SGD ═══
    let (cDd, cot_hn) ← pretty cBS (.dotOut "%Wd" (zM : Mat 768 nClasses) (.operand dyName zV))
    let (cHnB, cot_gap) := ("", cot_hn)
    let (cWd, nWd) ← if adam then
        pretty cBS (.weightGrad (m := 768) (n := nClasses) hn (zV : Vec 768) (.operand dyName (zV : Vec nClasses)))
      else pretty cBS (.weightSgd hn "%Wd" cLR (zV : Vec 768) (zM : Mat 768 nClasses) 0 (.operand dyName zV))
    let (cBd, nBd) ← if adam then
        pretty cBS (.biasGrad (n := nClasses) (.operand dyName (zV : Vec nClasses)))
      else pretty cBS (.biasSgd "%bd" cLR (zV : Vec nClasses) 0 (.operand dyName zV))
    let mut updMap : List (String × String) := [("Wd", nWd), ("bd", nBd)]
    let mut bwd := cDyC ++ cDd ++ cHnB ++
      cWd ++ cBd ++
      s!"    %dgi = stablehlo.reshape {cot_gap} : ({ty [cBS,768]}) -> {ty [cBS,768,1,1]}\n" ++
      s!"    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : ({ty [cBS,768,1,1]}) -> {ty [cBS,768,7,7]}\n" ++
      s!"    %dgn = stablehlo.constant dense<49.0> : {ty [cBS,768,7,7]}\n" ++
      s!"    %dgd = stablehlo.divide %dgb, %dgn : {ty [cBS,768,7,7]}\n" ++
      s!"    %dgapf = stablehlo.reshape %dgd : ({ty [cBS,768,7,7]}) -> {ty [cBS, 768*7*7]}\n"
    let mut dy := "%dgapf"
    for si' in [0:4] do
      let si := 3 - si'
      let c := cDims[si]!; let e := 4 * c; let h := cSpats[si]!
      for j' in [0:cDepths[si]!] do
        let j := cDepths[si]! - 1 - j'
        let b := (blksAll[si]!)[j]!
        let (code, cot_xin, cot_p, cot_e, cot_n, cot_d) ← bwdBlock s!"s{si}b{j}" dy b c e h
        let (pcode, pairs) ← blockParamSgd adam s!"s{si}b{j}" b cot_p cot_e cot_n cot_d dy c e h
        bwd := bwd ++ code ++ pcode; updMap := updMap ++ pairs; dy := cot_xin
      if si > 0 then
        let ci := cDims[si-1]!; let h2 := cSpats[si]!
        let (code, cot_n, cot_x) ← bwdDown s!"d{si-1}" dy (downIn[si-1]!) ci c h2
        let (pcode, pairs) ← downParamSgd adam s!"d{si-1}" (downLn[si-1]!) (downIn[si-1]!) cot_n dy ci c h2
        bwd := bwd ++ code ++ pcode; updMap := updMap ++ pairs; dy := cot_x
    -- stem: psb via convBiasSgd (channel-sum), psW via the certified stride-4 weight grad.
    -- `psW` WAS the last hand-written weight gradient in this render (`patchWGrad`, "the stride-4
    -- gap"). It is now `.convStride4WeightGrad`, whose `den` is the proven
    -- `flatConvStride4_weight_grad_has_vjp` — the cert that was genuinely missing, unlike the
    -- downsample's (see `downParamSgd`). In `adam` mode the update is the proven AdamW triple; the
    -- SGD path still wraps it in the hand-written `sgd` helper, so SGD is certified-gradient +
    -- hand-written-update there.
    -- §2m: back through the stem LN before the stem conv's own gradients see the cotangent.
    let (cg, ng) ← lnGammaTail adam "%psng" F.stemC dy 96 56
    let (cb, nb) ← lnBetaTail adam "%psnbt" dy 96 56
    let (cx, dx) ← lnBackSite "%psng" F.stemC dy 96 56
    bwd := bwd ++ cg ++ cb ++ cx
    updMap := updMap ++ [("psng", ng), ("psnbt", nb)]
    dy := dx
    let (cPsb, nPsb) ← if adam then
        pretty cBS (.convBiasGrad (zK : Kernel4 96 3 4 4) (zT : Tensor3 3 56 56) (zV : Vec 96) (.operand dy zV))
      else pretty cBS (.convBiasSgd "%psb" cLR (zK : Kernel4 96 3 4 4) (zT : Tensor3 3 56 56) (zV : Vec 96) 0 (.operand dy zV))
    let (cPsW, nPsW) ← pretty cBS (.convStride4WeightGrad (ic := 3) (oc := 96) (h := 56) (w := 56)
      (kH := 4) (kW := 4) "%x" (zV : Vec 96) (zV : Vec (3*(2*(2*56))*(2*(2*56))))
      (zK : Kernel4 96 3 4 4) (.operand dy (zV : Vec (96*56*56))))
    bwd := bwd ++ cPsW ++ (if adam then "" else sgdOf nPsW "psW" (ty [96,3,4,4])) ++ cPsb
    updMap := updMap ++ [("psW", if adam then nPsW else "%psWn"), ("psb", nPsb)]
    pure (fwd ++ bwd, updMap, nSm)

set_option maxRecDepth 8000 in
/-- **ConvNeXt-T (full [3,3,9,3]) SGD train step rendered from the verified AST** (except the two
    documented weight-grad gaps — the stem 4×4/s4 patchify and the even-kernel 2×2/s2 downsample,
    neither of which has a VJP-cert `SHlo` op). Every other line is `pretty` of a verified node:
    forward + backward cotangent chain + the param-SGD ops, whose output IS the updated param.

    The cotangent is plain CE with an **explicit** ÷B — unlike ViT/R34, which fold the batch mean
    into `lr` — so the committed `cLR = 0.1` is an effective 0.1, the house convention spelled
    differently (§2a-quinquies). -/
def convNextTrainStepFaithfulV (funcName : String := "convnext_train_step")
    (nClasses : Nat := 10) : String := Id.run do
  let (body, updMap, _) := (convNextBackAll false none nClasses).run' 0
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [cBS, 3*224*224]) :: (allParams nClasses).map (fun (nm, d) => s!"%{nm}: {ty d}") ++ ["%onehot: " ++ ty [cBS,nClasses]])
  let retTyL := String.intercalate ", " ((allParams nClasses).map (fun p => ty p.2))
  let retVals := String.intercalate ", "
    ((allParams nClasses).map (fun (nm, _) => (updMap.lookup nm).getD s!"%{nm}n"))
  return "module @m {\n" ++ s!"  func.func @{funcName}({argSig}) -> ({retTyL}) " ++ "{\n" ++
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    s!"    %bsc = stablehlo.constant dense<{cBS}.0> : {ty [cBS,nClasses]}\n" ++
    chLnPrelude ++
    body ++
    s!"    return {retVals} : {retTyL}\n" ++ "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § The AdamW tail — one proven triple per parameter, folded in signature order
-- ════════════════════════════════════════════════════════════════

/-- `(θ', m', v')` for one parameter from its un-fused gradient — the proven
    `adamMNextF`/`adamVNextF`/`adamWParamF` triple (`adamW_triple_faithful` bundles their `den`s
    into `Proofs.adamWStep` by `rfl`). β₁/β₂/ε/wd are baked literals; `%lr`/`%bc1`/`%bc2` are
    runtime `tensor<f32>` args, so one render serves the whole schedule. Mirrors
    `ResNet34RenderB.adamOne` / `MobileNetV2RenderB.adamOneM`.

    At `replicas > 1` the gradient is first averaged by `ViTRender.emitGradAllReduce`. **That
    collective is a TRUSTED CARVE-OUT** (handoff §5) — emitted text, not `pretty` of an AST node,
    so outside every faithfulness theorem here. The AdamW triple consumes the averaged gradient as
    an `.operand` exactly as it consumed the raw one, so the `den` side does not shift. At
    `replicas ≤ 1` this emits nothing and threads the raw gradient, so the single-device render
    stays byte-identical — the cheap self-check that the insertion is inert.

    ⚠ **A claim this docstring carried is now FALSE and is retired.** It said *"ConvNeXt is the
    first net whose collectives include RANK-0 operands — its 44 scalar LayerNorm γ/β params have
    `ds = []`"*. That was true of the scalar-LN render; §2m's channel LN makes every LN γ/β a
    `Vec c`, so **0 of the 180 collectives are rank-0** (measured on the re-rendered artifact:
    the LN ones are `tensor<96xf32>` … `tensor<768xf32>`). The rank-0 `all_reduce` path this
    render used to be the only exerciser of is no longer exercised anywhere in the repo. -/
private def convnextAdamOne (replicas : Nat) (nm : String) (ds : List Nat) (gradSSA : String)
    (ema : Bool := false) :
    StateM Nat (String × String × String × String × String) := do
  let n := ds.foldl (· * ·) 1
  let z : Vec n := fun _ => 0
  let (arS, gAvg) := ViTRender.emitGradAllReduce gradSSA ds nm replicas
  let gr : SHlo n := .operand gAvg z
  let (cM, nM) ← pretty cBS (.adamMNextF s!"%{nm}m" "%b1" "%ob1" ds 0 z gr)
  let (cV, nV) ← pretty cBS (.adamVNextF s!"%{nm}v" "%b2" "%ob2" ds 0 z gr)
  let (cT, nT) ← pretty cBS (.adamWParamF s!"%{nm}" s!"%{nm}m" s!"%{nm}v" "%b1" "%ob1"
                    "%b2" "%ob2" "%bc1" "%bc2" "%lr" "%eps" "%wd" ds 0 0 0 0 0 0 0 z z z gr)
  -- ▶ THE EMA SHADOW, and it needs NO new op: `Proofs.adamMNext β₁ m g = β₁·m + (1−β₁)·g` IS the
  -- reference's `ema_update` (`jax/Jax/Codegen.lean:2459`) at `(β₁ := d, m := ema, g := θ')`, so
  -- `adamMNextF` renders it and `adamMNextF_faithful` closes the denotation side by `rfl`. Third
  -- time this reading has paid — `momVNextF` at `(μ := wd, v := θ)` is the coupled L2 (§2k) and
  -- `adamVNextF` at `β₂ := ρ` is RMSProp's mean-square (recipe_gaps v1.2).
  --
  -- ⚠ It consumes `nT`, the UPDATED parameter, not the gradient — the shadow averages weights.
  -- ⚠ `%emad`/`%oemad` are function ARGS, not constants, because the reference's decay is
  -- TIME-VARYING: `d = min(decay, (1+t)/(10+t))`, TF's warmup-corrected form. That correction is
  -- required at our scale rather than optional — see `planning/ema.md` §2, where the reference's
  -- own measurement has a shadow holding 12.8% of the random init and scoring 0.00% top-1.
  --
  -- At `ema := false` NO `pretty` call happens, so the fresh-name counter does not move and every
  -- committed artifact re-renders byte-identically. That is gate 1 in its strong form, for free.
  let (cE, nE) ← if ema then
      pretty cBS (.adamMNextF s!"%{nm}e" "%emad" "%oemad" ds 0 z (.operand nT z))
    else pure ("", "")
  pure (arS ++ cM ++ cV ++ cT ++ cE, nT, nM, nV, nE)

/-- The driver's **variant slug** for a given replica count: the artifact is
    `verified_mlir/convnext_<variant>_train_step.mlir`, the entry point is
    `@convnext_<variant>_train_step`, and `LEAN_MLIR_VARIANT` selects it. All three must agree —
    the shim checks the entry name and refuses a mismatch outright ("entry mismatch") rather than
    running the wrong graph. The `#guard`s at the bottom pin the literal `#eval` paths against this.

    ConvNeXt has only one batch (32), so unlike `mnv2AdamVariant`/`r34AdamVariant` there is no
    batch suffix — rendering another batch would need `cBS` to stop being a private constant.

    ⚠ The EMA renders get their OWN slugs (`ema`/`emadp`), for the reason the RMSProp ones do: a
    render carrying a fourth `[θ|m|v|ema]` region must never be able to overwrite the artifact the
    AdamW trainer runs, whose blob has three. That is §2a's last-writer-wins race, and here it would
    also be an arity mismatch the driver could not survive. -/
def cnxAdamVariant (replicas : Nat) (ema : Bool := false) : String :=
  (if ema then "ema" else "adam") ++ (if replicas ≤ 1 then "" else "dp")

/-- β₁/β₂/ε/wd as graph constants — the committed ConvNeXt-T AdamW recipe. -/
private def convnextAdamConsts : String :=
  "    %b1 = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
  "    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>\n" ++
  "    %b2 = stablehlo.constant dense<0.999> : tensor<f32>\n" ++
  "    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>\n" ++
  "    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>\n" ++
  "    %wd = stablehlo.constant dense<0.0001> : tensor<f32>\n"

set_option maxRecDepth 8000 in
/-- **ConvNeXt-T AdamW train step rendered from the verified AST.** The certified peer of the
    hand-written render in `tests/TestConvNeXtTrain.lean` that `convnext-verified-adam` trains on.

    Same backward as `convnext_train_step` (`convNextBackAll`, one traversal) but taking the
    **un-fused gradients**, each fed to the proven AdamW triple. The cotangent adds label smoothing
    (α = 0.1, K = 10); ConvNeXt already spelled the ÷B explicitly, so that is the only difference.

    Interface: 545 in (`%x`, 180 θ, 180 m, 180 v, `%lr`/`%bc1`/`%bc2`, `%onehot`) / 543 out
    (180 θ', 180 m', 180 v', `%loss`/`%bc1`/`%bc2`) — positionally identical to the hand-written
    render, so `trainAdamSched`'s packed `[θ|m|v]` protocol is unchanged.

    **What this certifies.** As of 2026-07-28 **all 180 params are `pretty(AST)` end to end** —
    the two weight-grad gaps this render used to carry (the stem 4×4/s4 patchify and the even-kernel
    2×2/s2 downsample) are closed, by a new cert (`flatConvStride4_weight_grad_has_vjp`) and an
    emit-side odd/even padding split (`StableHLO.sWGradGeom`) respectively. Licensed by
    `convnext-adam-tie` against the previously committed hand-written render: **bit-exact on all
    83,434,629 returned floats**, spread 0/180, against a bit-exact A-vs-A floor.

    Still outside the AST here, and unchanged: `%loss` (report-only, no gradient path).

    **`replicas > 1` renders the DATA-PARALLEL variant** (handoff §2h-quater) to its own entry name
    and artifact path via `cnxAdamVariant`, so producing it can never clobber the one the trainer
    runs. The only difference is one `all_reduce(add)/N` per parameter gradient, between the
    certified gradient and the certified AdamW triple: *certified gradient → trusted collective →
    certified AdamW*. See `convnextAdamOne` for the carve-out. -/
def convNextAdamTrainStepFaithful (alphaStr negAlphaKStr bStr : String)
    (replicas : Nat := 1) (nClasses : Nat := 10) (slug : String := "convnext")
    (ema : Bool := false)
    : String := Id.run do
  -- ⚠ `negAlphaKStr` is DERIVED from `nClasses` when the caller leaves it empty, and only honoured
  -- verbatim otherwise. Passing −α/K as a string independent of K is the two-writers-for-one-fact
  -- shape §2k removed from ViT on 2026-07-31, and it is not academic: the R34 ImageNet render
  -- shipped `α/K` hardcoded at the K=10 value, ON THE GRADIENT PATH, and only an implausible loss
  -- (≈87 against ln(1000)=6.9) caught it. The empty-string default keeps every existing call site
  -- byte-identical while making the K=1000 spelling impossible to get wrong.
  let negAK := if negAlphaKStr.isEmpty then "-" ++ alphaOverK nClasses 0.1 else negAlphaKStr
  let (body, gradMap, nSm) := (convNextBackAll true (some (alphaStr, negAK, bStr)) nClasses).run' 0
  let go : StateM Nat String := do
    let mut adamCode := ""
    let mut thetaN : List String := []
    let mut mN : List String := []
    let mut vN : List String := []
    let mut eN : List String := []
    for (nm, ds) in allParams nClasses do
      let g := (gradMap.lookup nm).getD s!"%d{nm}"
      let (c, nT, nM, nV, nE) ← convnextAdamOne replicas nm ds g ema
      adamCode := adamCode ++ c
      thetaN := thetaN ++ [nT]; mN := mN ++ [nM]; vN := vN ++ [nV]
      if ema then eN := eN ++ [nE]
    -- `%loss` is REPORT-ONLY: mean smoothed-CE for logging, on no gradient path, NOT `pretty` of an
    -- AST node, and covered by no theorem — which is exactly the configuration in which §2b shipped
    -- plain CE against a smoothed-CE cotangent and only the numeric tie caught it. Built from the
    -- SAME smoothed recipe the cotangent implies, and gated by `convnext-adam-tie`. ConvNeXt has no
    -- BN, so — as with ViT — this is the ONLY output that reads the forward directly.
    let lossCode :=
      "    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──\n" ++
      s!"    %lz = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      s!"    %llog = stablehlo.log {nSm} : {ty [cBS, nClasses]}\n" ++
      s!"    %lohll = stablehlo.multiply %onehot, %llog : {ty [cBS, nClasses]}\n" ++
      s!"    %lt1s = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [cBS, nClasses]}, tensor<f32>) -> {ty [cBS]}\n" ++
      s!"    %llsr = stablehlo.reduce(%llog init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [cBS, nClasses]}, tensor<f32>) -> {ty [cBS]}\n" ++
      -- ⚠ BOTH constants are DERIVED. `%laKc` is α/K and was hardcoded at the K=10 value
      -- (0.010000) until 2026-08-01, which made the ImageNet render report a loss of ~101 where
      -- 1000-class CE at init must be ≈ ln(1000) = 6.9 — §2k's bug, in a SECOND place. It hid from
      -- the check that caught the cotangent because that one greps for the NEGATIVE spelling
      -- `-0.010000`, and this copy is positive. `%lomac` is (1−α) and is K-independent, but it is
      -- derived too so that α has one spelling here rather than two.
      -- At K=10 both render byte-identically to the literals they replace, so the fix is inert on
      -- every committed Imagenette artifact — gated, not assumed.
      s!"    %lomac = stablehlo.constant dense<{oneMinusAlpha 0.1}> : {ty [cBS]}\n" ++
      s!"    %laKc = stablehlo.constant dense<{alphaOverK nClasses 0.1}> : {ty [cBS]}\n" ++
      s!"    %llt1 = stablehlo.multiply %lomac, %lt1s : {ty [cBS]}\n" ++
      s!"    %llt2 = stablehlo.multiply %laKc, %llsr : {ty [cBS]}\n" ++
      s!"    %llpe = stablehlo.add %llt1, %llt2 : {ty [cBS]}\n" ++
      s!"    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : ({ty [cBS]}, tensor<f32>) -> tensor<f32>\n" ++
      s!"    %lbfc = stablehlo.constant dense<{cBS}.0> : tensor<f32>\n" ++
      s!"    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>\n" ++
      s!"    %loss = stablehlo.negate %lossm : tensor<f32>\n"
    let pTy := (allParams nClasses).map (fun p => ty p.2)
    -- ⚠ THE RETURN LAYOUT MUST EQUAL THE INPUT LAYOUT, region for region and scalar for scalar.
    -- The driver does `pbuf := out` — each step's output IS the next step's input (§2d.3's no-copy
    -- handover) — so a return list that dropped the shadow, or carried fewer scalars than the
    -- signature takes, would silently re-interpret the blob from step 2 onward. It is also exactly
    -- what the resident shim checks: inputs `[res_in, res_in+n)` and outputs `[res_out, res_out+n)`
    -- must agree tensor for tensor, which is why a 4th region needs no C change at all.
    -- `%emad`/`%oemad` ride through unread, as `%bc1`/`%bc2` already do.
    let retVals := thetaN ++ mN ++ vN ++ eN ++ ["%loss", "%bc1", "%bc2"]
                     ++ (if ema then ["%emad", "%oemad"] else [])
    let retTys := pTy ++ pTy ++ pTy ++ (if ema then pTy else [])
                     ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"]
                     ++ (if ema then ["tensor<f32>", "tensor<f32>"] else [])
    pure <|
      (if replicas ≤ 1 then
        -- Updated 2026-07-29. This banner used to carve out the stem 4x4/s4 and the 2x2/s2
        -- downsample WEIGHT GRADIENTS as hand-written — true when it was written, false since
        -- `9bb00f5` (§2f-bis) closed both, so the emitted artifact was UNDER-describing itself.
        "    // ── ConvNeXt-T AdamW train step: gradients + optimizer are pretty(AST node) ──\n" ++
        "    // All 180 params, including the stem 4x4/s4 patchify and the 2x2/s2 downsample\n" ++
        "    // WEIGHT GRADIENTS — the two documented gaps, closed 2026-07-28 (new cert\n" ++
        "    // flatConvStride4_weight_grad_has_vjp; emit-side odd/even split sWGradGeom).\n"
       else
        s!"    // ── ConvNeXt-T AdamW train step, DATA-PARALLEL over {replicas} replicas ──\n" ++
        "    // Every line is pretty(verified AST node) EXCEPT the per-parameter `%arsum*`\n" ++
        "    // all_reduce / `%armean*` blocks: those are a TRUSTED CARVE-OUT (handoff §5), emitted\n" ++
        "    // text outside the faithfulness theorems. Each replica evaluates the same tied graph\n" ++
        "    // at the batch it was rendered for; the collective averages that function's gradients\n" ++
        "    // over disjoint equal batches. Unlike the BN nets, ConvNeXt normalises with LayerNorm\n" ++
        "    // — within one example, never across the batch — so N x b IS 1 x (N.b) here and the\n" ++
        "    // §10.3b caveat does not apply.\n") ++
      body ++ convnextAdamConsts ++ adamCode ++ lossCode ++
      s!"    return {String.intercalate ", " retVals} : {String.intercalate ", " retTys}\n"
  -- The AdamW body continues the SGD traversal's fresh-name counter. `convNextBackAll` consumed
  -- names 0..k, so the Adam ops must start at k — otherwise they collide with the backward's SSAs.
  let used := ((convNextBackAll true (some (alphaStr, negAlphaKStr, bStr))).run 0).2
  let inner : String := go.run' used
  let pSig := String.intercalate ", " ((allParams nClasses).map (fun (nm, d) => s!"%{nm}: {ty d}"))
  let mSig := String.intercalate ", " ((allParams nClasses).map (fun (nm, d) => s!"%{nm}m: {ty d}"))
  let vSig := String.intercalate ", " ((allParams nClasses).map (fun (nm, d) => s!"%{nm}v: {ty d}"))
  let eSig := String.intercalate ", " ((allParams nClasses).map (fun (nm, d) => s!"%{nm}e: {ty d}"))
  let argSig := ("%x: " ++ ty [cBS, 3*224*224]) ++ ", " ++ pSig ++ ", " ++ mSig ++ ", " ++ vSig ++
    (if ema then ", " ++ eSig else "") ++
    ", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>" ++
    (if ema then ", %emad: tensor<f32>, %oemad: tensor<f32>" else "") ++
    ", %onehot: " ++ ty [cBS,nClasses]
  let pTy := (allParams nClasses).map (fun p => ty p.2)
  let retTyL := String.intercalate ", "
    (pTy ++ pTy ++ pTy ++ (if ema then pTy else [])
       ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"]
       ++ (if ema then ["tensor<f32>", "tensor<f32>"] else []))
  -- ⚠ The slug is load-bearing exactly as it is on R34 (§2k) and ViT (§2p): a 1000-class render
  -- emitted under the `convnext` slug would collide with the artifacts the 84.41% Imagenette run,
  -- the prefix audit and every `convnext-adam-tie` invocation depend on.
  let funcName := s!"{slug}_{cnxAdamVariant replicas ema}_train_step"
  return "module @m {\n" ++ s!"  func.func @{funcName}({argSig}) -> ({retTyL}) " ++ "{\n" ++
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    s!"    %bsc = stablehlo.constant dense<{cBS}.0> : {ty [cBS,nClasses]}\n" ++
    chLnPrelude ++
    inner ++ "  }\n}\n"

end Proofs.StableHLO

-- Regenerate verified_mlir/convnext_train_step.mlir from the faithful renderer (BS=32, ε=1e-6, lr=0.1).
#eval IO.FS.writeFile "verified_mlir/convnext_train_step.mlir"
  (Proofs.StableHLO.convNextTrainStepFaithfulV "convnext_train_step")

-- Regenerate `verified_mlir/convnext_fwd.mlir` — what `convnext-smooth` certifies through, and the
-- eval forward for the ConvNeXt trainers — from the SAME `convNextFwdChain` the train steps
-- differentiate. This replaces the independent hand-written emitter in `tests/TestConvNeXtFwd.lean`;
-- that copy is retired to an `iree-compile` smoke over the committed bytes.
--
-- **ConvNeXt needs no `_fwd_eval` peer and must not grow one.** LayerNorm reduces within one
-- example, never over the batch, so this forward is already class-batch-independent — the very
-- property `@resnet34_fwd_eval` / `@efficientnet_fwd_eval` exist to recover for the BN nets.
#eval IO.FS.writeFile "verified_mlir/convnext_fwd.mlir"
  (Proofs.StableHLO.convNextFwdFaithfulV "convnext_fwd")

-- The **AdamW** train step — **the artifact `convnext-verified-adam` trains on**, and from
-- 2026-07-28 this `#eval` is its ONLY writer. The hand-written emitter in
-- `tests/TestConvNeXtTrain.lean` is retired; that file now only iree-compiles the committed bytes.
-- Literals: α = 0.1, −α/K = −0.01 (K = 10), batch 32.
--
-- The swap was licensed by `lake build convnext-adam-tie` (one AdamW step, all 83,434,629 returned
-- floats): `%loss` BIT-EXACT, 179 of 180 parameter gradients bit-exact, and the one that differs —
-- `s3b2lg`, the last block's layer-scale γ — agrees BETTER than this render does with itself under
-- a semantics-preserving batch reversal. That γ gradient is a cancelling reduce (|Σ|/Σ|·| ≈ 0.09)
-- and does not reproduce to 1e-4 against ANY reordering, so the gate is calibrated against that
-- control rather than an absolute bound, and gates the SPREAD as well as the magnitude — a
-- cotangent perturbation clears the magnitude gate while disturbing 178/180 params. To re-run:
--
--   git show b94e8e9:verified_mlir/convnext_adam_train_step.mlir > /tmp/retired.mlir
--   IREE_BACKEND=rocm .lake/build/bin/convnext-adam-tie /tmp/retired.mlir \
--     verified_mlir/convnext_adam_train_step.mlir
#eval IO.FS.writeFile "verified_mlir/convnext_adam_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithful "0.100000" "-0.010000" "32.0")

-- ── ▶ THE EMA VARIANT (`planning/ema.md`), selected by `LEAN_MLIR_VARIANT=ema` ────────────────
-- Same graph plus one `adamMNextF` per parameter on the UPDATED weight — `d·ema + (1−d)·θ'`, which
-- is `Proofs.adamMNext` at `(β₁ := d, m := ema, g := θ')`, so this costs **no new op, no new `den`,
-- no new faithfulness theorem and no new VJP**. It is the third time enumerating the reference's
-- update against existing ops AT THEIR OTHER READINGS has collapsed a scoped op family to zero
-- (§2k heavy-ball, recipe_gaps v1.2 RMSProp, here).
--
-- ⚠ THE BLOB GAINS A FOURTH REGION: `[θ|m|v|ema]`, and the scalar tail goes 3 → 5 (`%emad`,
-- `%oemad`). That is why it renders to its OWN slug — a 4-region graph fed a 3-region blob is not a
-- subtle numeric wrong answer, it is every parameter misaligned, and the AdamW artifact must stay
-- exactly what it is. The driver's checkpoint SIZE GUARD is the other half of that (`ema.md` §5b):
-- checkpoints carry no header, so a 3-region file read as 4 resumes silent garbage.
--
-- ⚠ `%emad`/`%oemad` are ARGS rather than constants because the reference's decay is time-varying,
-- `d = min(decay, (1+t)/(10+t))` — TF's warmup-corrected `ExponentialMovingAverage`. `ema.md` §2
-- has the reference's own measurement of what dropping that correction costs: a shadow still
-- holding 12.8% of the random init at epoch 66, scoring **0.00% top-1** while the live weights
-- scored 70.48%. An 80-epoch Imagenette run is 2.4 τ, i.e. inside that regime.
#eval IO.FS.writeFile "verified_mlir/convnext_ema_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithful "0.100000" "-0.010000" "32.0"
    (ema := true))

-- The **DATA-PARALLEL** render (handoff §2h-quater), selected at run time by
-- `LEAN_MLIR_VARIANT=adamdp`. ConvNeXt was the last large net with no DP path at all — its renderer
-- took no `replicas` and emitted no collective, so unlike mnv2's (§2h-bis, one `#eval`) this needed
-- the parameter threaded through `convnextAdamOne` first.
--
-- Same graph, plus one `all_reduce(add)/N` per parameter gradient between the certified gradient
-- and the certified AdamW triple: *certified gradient → trusted collective → certified AdamW*. The
-- collective is a DECLARED carve-out and the render says so in its own output banner at
-- `replicas > 1`, per the §5/§2b `%loss` lesson that an undeclared carve-out is how wrong things
-- ship. Claim ceiling is unchanged (§5): the gradient averaging is a proven identity; the collective
-- implementing it is trusted, exactly like the lowerer.
--
-- ⚠ §2h-quater's headline — *"44 of the 180 collectives are RANK-0"* — WAS this net's scalar
-- LayerNorm and is retired by §2m: per-channel γ/β makes them `tensor<{c}xf32>`, so no collective
-- here is rank-0 any more. Nothing in the repo exercises a rank-0 `all_reduce` now.
--
-- It renders to its OWN path, which is what stops the §2a race where producing a DP render meant
-- editing a knob and clobbering the artifact the trainer runs. `2` is the replica count these are
-- rendered at and it must match `PJRT_REPLICAS` at run time, because the graph bakes
-- `replica_groups`. Re-render here to change it.
--
-- It needs the XLA build (`convnext-verified-adam-xla`, §2h): collectives exist only on the PJRT
-- path, and the IREE shim refuses a DP entry point outright rather than silently running
-- single-device.
#eval IO.FS.writeFile "verified_mlir/convnext_adamdp_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithful "0.100000" "-0.010000" "32.0" 2)

-- ── ConvNeXt-T on FULL 1000-class ImageNet, slug `cnxin` — 2026-08-01 ──────────────────────────
-- The ConvNeXt peer of `resnet34in_*` (§2k) and `vitin_*` (§2p). `nClasses` is a renderer
-- parameter as of this change; `cBS` is NOT, so these render at the committed batch of 32 and the
-- four-replica variant is global batch 128.
--
-- That is a deliberate scope cut, not an oversight. `cBS` is a private constant in 96 places and
-- threading it is a separate refactor; global 128 is meanwhile a perfectly good ImageNet config —
-- §2d.2 measured accuracy tracking STEP COUNT, and at 1,281,167 images global 128 gives 10,009
-- steps/epoch against the reference's 5,004 at batch 256. Fewer images per step, more steps.
--
-- ⚠ `-α/K` is DERIVED here (empty string ⇒ `alphaOverK nClasses`), so the emitted constant is
-- -0.000100 at K=1000 rather than the K=10 literal the Imagenette renders carry. That hardcoding
-- was a REAL BUG on R34's first ImageNet render, on the gradient path, caught only because the
-- loss was implausible (§2k). Gated below by the artifact check, not assumed.
#eval IO.FS.writeFile "verified_mlir/cnxin_adam_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithful "0.100000" "" "32.0" 1 1000 "cnxin")
#eval IO.FS.writeFile "verified_mlir/cnxin_adamdp_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithful "0.100000" "" "32.0" 4 1000 "cnxin")
#eval IO.FS.writeFile "verified_mlir/cnxin_fwd.mlir"
  (Proofs.StableHLO.convNextFwdFaithfulV "cnxin_fwd" 1000)

-- The entry name, the artifact path and `LEAN_MLIR_VARIANT` must agree or the shim refuses the
-- call ("entry mismatch"). These pin the literal paths above against `cnxAdamVariant`, so a rename
-- fails at `lake build` rather than at run time. (The audit greps for the LITERAL string
-- `IO.FS.writeFile "verified_mlir/`, so those paths must stay literals — do not interpolate them.)
#guard Proofs.StableHLO.cnxAdamVariant 1 == "adam"
#guard Proofs.StableHLO.cnxAdamVariant 2 == "adamdp"
-- The EMA peers. Distinct slugs from the AdamW ones is the point: the EMA render carries a FOURTH
-- `[θ|m|v|ema]` region, so it and the AdamW render cannot share an artifact path, a checkpoint or a
-- driver invocation. `LEAN_MLIR_VARIANT=ema` selects it and the driver keys its 4-region layout off
-- the same `"ema"` prefix, exactly as it keys RMSProp's mean-square init off `"rms"`.
#guard Proofs.StableHLO.cnxAdamVariant 1 true == "ema"
#guard Proofs.StableHLO.cnxAdamVariant 2 true == "emadp"
