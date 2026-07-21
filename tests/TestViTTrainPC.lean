import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Types

/-! # ViT Item B — structured representative train-step render (proof-rendered)

The ViT peer of `tests/TestConvNeXtTrainPC.lean`, at the **PRODUCTION ViT-Tiny config**
(the capstone; the proven graph `vitFwdGraphKMHV`, `ViTDepthK.lean` / `ViTMultiHead.lean`):
16×16/stride-16 patch embed (+ CLS + pos) → **12 distinct-param pre-norm transformer
blocks** (data-driven loop, the `Fin k` param-function regime; **heads = 3, d_head = 64**)
→ final per-token LN → CLS slice → dense head. ImageNet-shaped: 3×224² in → 196 patches
+ CLS = 197 tokens, D = 192, mlpDim = 768, 10 classes (imagenette), BS = 32,
4 + 16·12 + 4 = 200 params. The signature MATCHES the committed
`verified_mlir/vit_train_step.mlir` exactly (same `@vit_train_step` name, same param
order/shapes — incl. cls as `[1,D]` via denotation-trivial reshape glue, eps 1e-5,
scale 1/√64, lr 0.1), enabling the TWO-SIDED `render_parity.py` check.
**Vector-[D] LN γ/β** (the production `ViTRender` form, proven in `ViTVecLN.lean`): each
LN site is the three-token decomposition `lnRowF`(1,0) → `rowScaleF γ` → `rowBiasF β`;
the LN backward reuses `rowScaleF` on the cotangent (diagonal — its own input-VJP) then
`lnRowBack`(γ=1) at the saved pre-LN input; the per-channel param grads keep the channel
axis (`dγ_k = Σ_{b,tokens} dy·x̂`, certified `vit_render_vecln{gamma,beta}_certified`).

Forward AND the whole backward cotangent chain are proof-rendered through `pretty` over
the very tokens of `vitFwdGraphMHV` — forward (`patchEmbedF`/`lnRowF`/`denseRowF`/
**per head: `headSliceF` → `matmulF`/`transposeF`/`scaleF`/`softmaxRowF` → `headPadF`**,
the pad-sum concat as `addV`/`clsSliceF`/`denseF`) and backward (`dotOut`, `clsPadF`,
`lnRowBack`, `denseRowBack`, `geluBack`, `softmaxRowBack`, and the **per-head SDPA 3-path
backward spelled with the forward `matmulF`/`transposeF`/`headSliceF`/`headPadF` on
cotangents** — matmul's VJP IS matmul and the slice/pad pair are each other's VJPs:
`dO_h = slice_h dO`, `dP_h = dO_h·V_hᵀ`, `dV_h = P_hᵀ·dO_h`, `dQ_h = s·dS_h·K_h`,
`dK_h = s·dS_hᵀ·Q_h`, then `dQ = Σ_h pad_h dQ_h` etc. — the proven `sdpa_back_{Q,K,V}`
shapes per head). Residual fan-ins are `addV`; the Q/K/V three-way fan-in at LN₁'s
output is two `addV`s.

Only the no-SHlo-constructor pieces are hand-emitted, each certified in
`ViTClose.lean` (Item C): per-token dense `dW = Σ_{b,tokens} x⊗dy` / `db = Σ dy`
(`vit_render_rowdense{W,b}_certified`), rowwise scalar-LN `dγ = Σ dy·x̂` / `dβ = Σ dy`
(`vit_render_rowln{gamma,beta}_certified`), `dPos = Σ_b dy` (`vit_render_pos_certified`),
`dCls = row-0 slice` (`vit_render_cls_certified`), the patchSize-1 patch-projection
`dWp`/`dbp` over the patch rows (`vit_render_patch{W,b}_certified`), and the head
`dWcls = clsᵀ·dy` / `dbcls = Σ dy` (M2).

Validation is the `scripts/render_parity.py` TWO-SIDED parity against the committed
GPU-trained renderer (same signature; expect equivalent-not-byte-identical — the
recompute-vs-save layouts differ, e.g. per-head slice/pad-sum vs rank-4 batched
attention, im2col vs dilate+conv patch W-grad, 3-token vs fused LN affine):
  `scripts/render_parity.py --fn vit_train_step --ref verified_mlir/vit_train_step.mlir \
     --cand /tmp/vitpc/train_step.mlir`

Run: `IREE_BACKEND=rocm lake env lean tests/TestViTTrainPC.lean`
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 32
private def IC : Nat := 3     -- input channels
private def HH : Nat := 224   -- spatial (ImageNet)
private def PP : Nat := 16    -- patch size (= stride)
private def PH : Nat := 14    -- patch grid (224/16)
private def NP : Nat := 196   -- patches (14×14)
private def NT : Nat := 197   -- tokens (patches + CLS)
private def DEPTH : Nat := 12 -- transformer blocks (the ViT-Tiny depth)
private def HD : Nat := 3     -- heads
private def DH : Nat := 64    -- d_head
private def DD : Nat := 192   -- embed dim (= heads · d_head)
private def MD : Nat := 768   -- MLP dim (4×)
private def NC : Nat := 10    -- classes (imagenette)
private def EPS : String := "1.0e-5"  -- the committed ViTRender eps
private def LR : String := "0.1"
private def SCALE : String := "0.125"  -- 1/√64 = 1/√d_head

-- placeholder values (pretty/emitTok render names only; values are irrelevant)
private def zK {o i kh kw : Nat} : Kernel4 o i kh kw := fun _ _ _ _ => 0
private def zV {n : Nat} : Vec n := fun _ => 0
private def zM {a b : Nat} : Mat a b := fun _ _ => 0

-- ════════════ hand-emitted param-grad templates (Item C certified forms) ════════════

/-- flat → [BS, tokens, feat] reshape. -/
private def rs3 (o flatN : String) (t f : Nat) : String :=
  s!"    {o} = stablehlo.reshape {flatN} : ({ty [BS, t*f]}) -> {ty [BS,t,f]}\n"

/-- per-token dense weight grad `dW = Σ_(b,tokens) x ⊗ dy` — one `dot_general`
    contracting batch+token axes (`vit_render_rowdenseW_certified`). -/
private def rowDenseWGrad (o xFlat dyFlat : String) (a c : Nat) : String :=
  rs3 s!"{o}xi" xFlat NT a ++ rs3 s!"{o}di" dyFlat NT c ++
  s!"    {o} = stablehlo.dot_general {o}xi, {o}di, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : ({ty [BS,NT,a]}, {ty [BS,NT,c]}) -> {ty [a,c]}\n"

/-- per-token dense bias grad `db = Σ_(b,tokens) dy` (`vit_render_rowdenseb_certified`). -/
private def rowDenseBGrad (o dyFlat : String) (c : Nat) : String :=
  rs3 s!"{o}i" dyFlat NT c ++
  s!"    {o} = stablehlo.reduce({o}i init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({ty [BS,NT,c]}, tensor<f32>) -> {ty [c]}\n"

/-- vector-LN dγ_k = Σ_(b,tok) dy·x̂ (KEEPS the channel axis), dβ_k = Σ_(b,tok) dy —
    `ViTRender`'s per-channel LN param reduces, off the SAVED normalize output x̂ (the
    `lnRowF`(1,0) SSA value — no recompute needed at the decomposed form). The rendered
    `vecLN_grad_gamma/beta` (`vit_render_vecln{gamma,beta}_certified`). -/
private def vecLNParamGrad (dgr dbe xhFlat dyFlat : String) (t f : Nat) : String :=
  let tn := ty [BS, t, f]
  rs3 s!"{dgr}xh" xhFlat t f ++ rs3 s!"{dgr}dyi" dyFlat t f ++
  s!"    {dgr}p = stablehlo.multiply {dgr}dyi, {dgr}xh : {tn}\n" ++
  s!"    {dgr} = stablehlo.reduce({dgr}p init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({tn}, tensor<f32>) -> {ty [f]}\n" ++
  s!"    {dbe} = stablehlo.reduce({dgr}dyi init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({tn}, tensor<f32>) -> {ty [f]}\n"

private def sgd (θ dθ ty' : String) : String :=
  s!"    {θ}l = stablehlo.constant dense<{LR}> : {ty'}\n" ++
  s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
  s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"

-- ════════════ captured forward names per ViT block ════════════
private structure HNames where  -- per-head saved SSA names
  qh : String    -- sliced Q (the dK matmul partner)
  kh : String    -- sliced K (the dQ matmul partner)
  vh : String    -- sliced V (the dP transpose partner)
  ss : String    -- scaled scores (pre-softmax; softmaxRowBack's saved input)
  p  : String    -- post-softmax weights (the dV transpose partner)
deriving Inhabited

private structure FNames where  -- flat SSA names from `pretty`
  xin : String   -- block input (= the attn residual skip; saved pre-LN1 input)
  xh1 : String   -- LN1 normalize out (x̂ — the dγ1 partner)
  ln1 : String   -- LN1 out (= Q/K/V dense input)
  hs : List HNames  -- per-head saved names (HD entries)
  att : String   -- Σ_h pad_h(P_h·V_h) — the head concat (out-proj input)
  h : String     -- attn-sublayer out (= the MLP residual skip; saved pre-LN2 input)
  xh2 : String   -- LN2 normalize out (x̂ — the dγ2 partner)
  ln2 : String   -- LN2 out (= fc1 input)
  m1 : String    -- fc1 out (pre-GELU)
  g : String     -- GELU out (fc2 input)
  bout : String  -- block out

/-- Left-fold `addV` of the per-head pad-scatters (= `headsSumG`'s emission). -/
private def sumPads (ns : List String) : StateM Nat (String × String) := do
  let mut code := ""
  let mut acc := ns.head!
  for pn in ns.tail! do
    let (ca, s) ← pretty BS (.addV (.operand acc (zV : Vec (NT*(HD*DH)))) (.operand pn zV))
    code := code ++ ca
    acc := s
  pure (code, acc)

/-- Per-head SDPA forward: slice Q/K/V, `Q_h·K_hᵀ` → `·1/√d_head` → row-softmax →
    `P_h·V_h`, pad-scatter back; the head concat is the `addV` fold of the pads
    (exactly `vitBlockGraphMHV`'s attention tokens). -/
private def fwdHeads (q k v : String) : StateM Nat (String × List HNames × String) := do
  let mut code := ""
  let mut hs : List HNames := []
  let mut pads : List String := []
  for h in List.finRange HD do
    let (c1, qh) ← pretty BS (.headSliceF h (.operand q (zV : Vec (NT*(HD*DH)))))
    let (c2, kh) ← pretty BS (.headSliceF h (.operand k (zV : Vec (NT*(HD*DH)))))
    let (c3, vh) ← pretty BS (.headSliceF h (.operand v (zV : Vec (NT*(HD*DH)))))
    let (c4, kT) ← pretty BS (.transposeF (.operand kh (zV : Vec (NT*DH))))
    let (c5, scs) ← pretty BS (.matmulF (.operand qh (zV : Vec (NT*DH)))
      (.operand kT (zV : Vec (DH*NT))))
    let (c6, ss) ← pretty BS (.scaleF SCALE 0 (.operand scs (zV : Vec (NT*NT))))
    let (c7, p) ← pretty BS (.softmaxRowF (.operand ss (zV : Vec (NT*NT))))
    let (c8, attH) ← pretty BS (.matmulF (.operand p (zV : Vec (NT*NT)))
      (.operand vh (zV : Vec (NT*DH))))
    let (c9, pad) ← pretty BS (.headPadF h (.operand attH (zV : Vec (NT*DH))))
    code := code ++ c1 ++ c2 ++ c3 ++ c4 ++ c5 ++ c6 ++ c7 ++ c8 ++ c9
    hs := hs ++ [⟨qh, kh, vh, ss, p⟩]
    pads := pads ++ [pad]
  let (cs, att) ← sumPads pads
  pure (code ++ cs, hs, att)

/-- Per-head SDPA backward — the forward `matmulF`/`transposeF`/`headSliceF`/`headPadF`
    on cotangents (slice and pad are each other's VJPs): per head `dO_h = slice_h dO`,
    `dP_h = dO_h·V_hᵀ`, `dV_h = P_hᵀ·dO_h`, `dS_h = softmaxRowBack`, undo-scale,
    `dQ_h = dS_h·K_h`, `dK_h = dS_hᵀ·Q_h`, then `dQ/dK/dV = Σ_h pad_h(·)` — the proven
    `sdpa_back_{Q,K,V}` shapes per head. Returns (code, dQ, dK, dV) at `[NT,D]` flat. -/
private def bwdHeads (cot_att : String) (hs : List HNames) :
    StateM Nat (String × String × String × String) := do
  let mut code := ""
  let mut dQpads : List String := []
  let mut dKpads : List String := []
  let mut dVpads : List String := []
  for (h, b) in (List.finRange HD).zip hs do
    let (c0, dOh) ← pretty BS (.headSliceF h (.operand cot_att (zV : Vec (NT*(HD*DH)))))
    let (c1, vT) ← pretty BS (.transposeF (.operand b.vh (zV : Vec (NT*DH))))
    let (c2, dP) ← pretty BS (.matmulF (.operand dOh (zV : Vec (NT*DH)))
      (.operand vT (zV : Vec (DH*NT))))
    let (c3, pT) ← pretty BS (.transposeF (.operand b.p (zV : Vec (NT*NT))))
    let (c4, dVh) ← pretty BS (.matmulF (.operand pT (zV : Vec (NT*NT)))
      (.operand dOh (zV : Vec (NT*DH))))
    let (c5, dS) ← pretty BS (.softmaxRowBack b.ss (zV : Vec (NT*NT)) (.operand dP zV))
    let (c6, dSs) ← pretty BS (.scaleF SCALE 0 (.operand dS (zV : Vec (NT*NT))))
    let (c7, dQh) ← pretty BS (.matmulF (.operand dSs (zV : Vec (NT*NT)))
      (.operand b.kh (zV : Vec (NT*DH))))
    let (c8, dSsT) ← pretty BS (.transposeF (.operand dSs (zV : Vec (NT*NT))))
    let (c9, dKh) ← pretty BS (.matmulF (.operand dSsT (zV : Vec (NT*NT)))
      (.operand b.qh (zV : Vec (NT*DH))))
    let (cq, dQp) ← pretty BS (.headPadF h (.operand dQh (zV : Vec (NT*DH))))
    let (ck, dKp) ← pretty BS (.headPadF h (.operand dKh (zV : Vec (NT*DH))))
    let (cv, dVp) ← pretty BS (.headPadF h (.operand dVh (zV : Vec (NT*DH))))
    code := code ++ c0 ++ c1 ++ c2 ++ c3 ++ c4 ++ c5 ++ c6 ++ c7 ++ c8 ++ c9 ++
            cq ++ ck ++ cv
    dQpads := dQpads ++ [dQp]
    dKpads := dKpads ++ [dKp]
    dVpads := dVpads ++ [dVp]
  let (csq, dQ) ← sumPads dQpads
  let (csk, dK) ← sumPads dKpads
  let (csv, dV) ← sumPads dVpads
  pure (code ++ csq ++ csk ++ csv, dQ, dK, dV)

/-- One ViT block forward via `pretty` — exactly the `vitBlockGraphMHV` tokens. -/
private def fwdBlock (i : Nat) (xin : String) : StateM Nat (String × FNames) := do
  let (k1a, xh1) ← pretty BS (.lnRowF "%one" "%sc" EPS 0 1 0
    (.operand xin (zV : Vec (NT*DD))))
  let (k1b, sc1) ← pretty BS (.rowScaleF s!"%g1_{i}" (zV : Vec DD)
    (.operand xh1 (zV : Vec (NT*DD))))
  let (k1, ln1) ← pretty BS (.rowBiasF s!"%bt1_{i}" (zV : Vec DD)
    (.operand sc1 (zV : Vec (NT*DD))))
  let (k2, q) ← pretty BS (.denseRowF s!"%Wq{i}" s!"%bq{i}" (zM : Mat DD DD) zV
    (.operand ln1 (zV : Vec (NT*DD))))
  let (k3, kk) ← pretty BS (.denseRowF s!"%Wk{i}" s!"%bk{i}" (zM : Mat DD DD) zV
    (.operand ln1 (zV : Vec (NT*DD))))
  let (k4, v) ← pretty BS (.denseRowF s!"%Wv{i}" s!"%bv{i}" (zM : Mat DD DD) zV
    (.operand ln1 (zV : Vec (NT*DD))))
  let (k5, hs, att) ← fwdHeads q kk v
  let (k10, o) ← pretty BS (.denseRowF s!"%Wo{i}" s!"%bo{i}" (zM : Mat DD DD) zV
    (.operand att (zV : Vec (NT*DD))))
  let (k11, h) ← pretty BS (.addV (.operand xin (zV : Vec (NT*DD))) (.operand o zV))
  let (k12a, xh2) ← pretty BS (.lnRowF "%one" "%sc" EPS 0 1 0
    (.operand h (zV : Vec (NT*DD))))
  let (k12b, sc2) ← pretty BS (.rowScaleF s!"%g2_{i}" (zV : Vec DD)
    (.operand xh2 (zV : Vec (NT*DD))))
  let (k12, ln2) ← pretty BS (.rowBiasF s!"%bt2_{i}" (zV : Vec DD)
    (.operand sc2 (zV : Vec (NT*DD))))
  let (k13, m1) ← pretty BS (.denseRowF s!"%Wfc1{i}" s!"%bfc1{i}" (zM : Mat DD MD) zV
    (.operand ln2 (zV : Vec (NT*DD))))
  let (k14, g) ← pretty BS (.geluF (.operand m1 (zV : Vec (NT*MD))))
  let (k15, m2) ← pretty BS (.denseRowF s!"%Wfc2{i}" s!"%bfc2{i}" (zM : Mat MD DD) zV
    (.operand g (zV : Vec (NT*MD))))
  let (k16, bout) ← pretty BS (.addV (.operand h (zV : Vec (NT*DD))) (.operand m2 zV))
  pure (k1a ++ k1b ++ k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k10 ++
        k11 ++ k12a ++ k12b ++ k12 ++ k13 ++ k14 ++ k15 ++ k16,
        ⟨xin, xh1, ln1, hs, att, h, xh2, ln2, m1, g, bout⟩)

/-- One ViT block backward via `pretty`. `dy` = flat cotangent at block output.
    The SDPA backward is the forward `matmulF`/`transposeF` on cotangents.
    Returns (code, cot-at-block-input, dQ, dK, dV, cot_h, cot_ln2, cot_m1, cot_g, cot_ln1). -/
private def bwdBlock (i : Nat) (dy : String) (b : FNames) :
    StateM Nat (String × String × String × String × String × String × String ×
                String × String × String) := do
  -- MLP sublayer back: bout = h + fc2(gelu(fc1(LN2 h)))
  let (k1, cot_g) ← pretty BS (.denseRowBack s!"%Wfc2{i}" (zM : Mat MD DD)
    (.operand dy (zV : Vec (NT*DD))))
  let (k2, cot_m1) ← pretty BS (.geluBack b.m1 (zV : Vec (NT*MD))
    (.operand cot_g zV))
  let (k3, cot_ln2) ← pretty BS (.denseRowBack s!"%Wfc1{i}" (zM : Mat DD MD)
    (.operand cot_m1 (zV : Vec (NT*MD))))
  let (k4a, cot_xh2) ← pretty BS (.rowScaleF s!"%g2_{i}" (zV : Vec DD)
    (.operand cot_ln2 (zV : Vec (NT*DD))))
  let (k4, cot_h_mlp) ← pretty BS (.lnRowBack "%one" b.h EPS 0 1
    (zV : Vec (NT*DD)) (.operand cot_xh2 zV))
  let (k5, cot_h) ← pretty BS (.addV (.operand dy (zV : Vec (NT*DD)))
    (.operand cot_h_mlp zV))
  -- attn sublayer back: h = xin + Wo·MHSA(q,k,v)
  let (k6, cot_att) ← pretty BS (.denseRowBack s!"%Wo{i}" (zM : Mat DD DD)
    (.operand cot_h (zV : Vec (NT*DD))))
  -- per-head SDPA 3-path backward + pad-sum (the proven sdpa_back_{Q,K,V}
  -- shapes per head; slice/pad are each other's VJPs)
  let (k7, dQ, dK, dV) ← bwdHeads cot_att b.hs
  -- Q/K/V dense backs fan IN at LN1's output (three cotangents sum)
  let (k16, cq) ← pretty BS (.denseRowBack s!"%Wq{i}" (zM : Mat DD DD)
    (.operand dQ (zV : Vec (NT*DD))))
  let (k17, ck) ← pretty BS (.denseRowBack s!"%Wk{i}" (zM : Mat DD DD)
    (.operand dK (zV : Vec (NT*DD))))
  let (k18, cv) ← pretty BS (.denseRowBack s!"%Wv{i}" (zM : Mat DD DD)
    (.operand dV (zV : Vec (NT*DD))))
  let (k19, s1) ← pretty BS (.addV (.operand cq (zV : Vec (NT*DD))) (.operand ck zV))
  let (k20, cot_ln1) ← pretty BS (.addV (.operand s1 (zV : Vec (NT*DD))) (.operand cv zV))
  let (k21a, cot_xh1) ← pretty BS (.rowScaleF s!"%g1_{i}" (zV : Vec DD)
    (.operand cot_ln1 (zV : Vec (NT*DD))))
  let (k21, cot_xin_attn) ← pretty BS (.lnRowBack "%one" b.xin EPS 0 1
    (zV : Vec (NT*DD)) (.operand cot_xh1 zV))
  let (k22, cot_xin) ← pretty BS (.addV (.operand cot_h (zV : Vec (NT*DD)))
    (.operand cot_xin_attn zV))
  pure (k1 ++ k2 ++ k3 ++ k4a ++ k4 ++ k5 ++ k6 ++ k7 ++
        k16 ++ k17 ++ k18 ++ k19 ++ k20 ++ k21a ++ k21 ++ k22,
        cot_xin, dQ, dK, dV, cot_h, cot_ln2, cot_m1, cot_g, cot_ln1)

/-- Block param grads (hand-emitted, Item C certified forms). -/
private def blockParamGrads (i : Nat) (b : FNames)
    (dQ dK dV cot_h cot_ln2 cot_m1 _cot_g cot_ln1 dy : String) : String :=
  -- Q/K/V/O per-token dense: X = LN1-out (Q/K/V) / att (O)
  rowDenseWGrad s!"%dWq{i}" b.ln1 dQ DD DD ++ rowDenseBGrad s!"%dbq{i}" dQ DD ++
  rowDenseWGrad s!"%dWk{i}" b.ln1 dK DD DD ++ rowDenseBGrad s!"%dbk{i}" dK DD ++
  rowDenseWGrad s!"%dWv{i}" b.ln1 dV DD DD ++ rowDenseBGrad s!"%dbv{i}" dV DD ++
  rowDenseWGrad s!"%dWo{i}" b.att cot_h DD DD ++ rowDenseBGrad s!"%dbo{i}" cot_h DD ++
  -- MLP fc1/fc2: X = LN2-out / GELU-out; dy at fc2-out = block-out cotangent
  rowDenseWGrad s!"%dWfc1{i}" b.ln2 cot_m1 DD MD ++ rowDenseBGrad s!"%dbfc1{i}" cot_m1 MD ++
  rowDenseWGrad s!"%dWfc2{i}" b.g dy MD DD ++ rowDenseBGrad s!"%dbfc2{i}" dy DD ++
  -- vector LN1/LN2 γ/β: per-channel reduces off the SAVED normalize outputs
  vecLNParamGrad s!"%dg1_{i}" s!"%dbt1_{i}" b.xh1 cot_ln1 NT DD ++
  vecLNParamGrad s!"%dg2_{i}" s!"%dbt2_{i}" b.xh2 cot_ln2 NT DD

/-- per-block param (name, type) list, forward order (matches `vitFwdGraph` arg order). -/
private def blkParams (i : Nat) : List (String × String) :=
  [(s!"g1_{i}", ty [DD]), (s!"bt1_{i}", ty [DD]),
   (s!"Wq{i}", ty [DD,DD]), (s!"bq{i}", ty [DD]),
   (s!"Wk{i}", ty [DD,DD]), (s!"bk{i}", ty [DD]),
   (s!"Wv{i}", ty [DD,DD]), (s!"bv{i}", ty [DD]),
   (s!"Wo{i}", ty [DD,DD]), (s!"bo{i}", ty [DD]),
   (s!"g2_{i}", ty [DD]), (s!"bt2_{i}", ty [DD]),
   (s!"Wfc1{i}", ty [DD,MD]), (s!"bfc1{i}", ty [MD]),
   (s!"Wfc2{i}", ty [MD,DD]), (s!"bfc2{i}", ty [DD])]

private def trainStep : String := Id.run do
  let go : StateM Nat String := do
    -- ═══ forward (proof-rendered; the vitFwdGraphKMHV tokens in graph order) ═══
    -- the committed signature carries cls as [1,D]; the patchEmbedF broadcast
    -- reads a [D] vector — denotation-trivial reshape glue
    let cClsR := s!"    %clsr = stablehlo.reshape %cls : ({ty [1,DD]}) -> {ty [DD]}\n"
    let (cE, embed) ← pretty BS (.patchEmbedF (ic := IC) (H := HH) (W := HH) (P := PP)
      (N := NP) (D := DD) "%Wp" "%bp" "%clsr" "%pos"
      (zK : Kernel4 DD IC PP PP) zV zV (zM : Mat (NP+1) DD) (.operand "%x" zV))
    let cE := cClsR ++ cE
    let mut fwd := cE
    let mut blocks : List FNames := []
    let mut xcur := embed
    for i in List.range DEPTH do
      let (c, b) ← fwdBlock (i + 1) xcur
      fwd := fwd ++ c
      blocks := blocks ++ [b]
      xcur := b.bout
    let (cFa, xhF) ← pretty BS (.lnRowF "%one" "%sc" EPS 0 1 0
      (.operand xcur (zV : Vec (NT*DD))))
    let (cFb, scF) ← pretty BS (.rowScaleF "%gF" (zV : Vec DD)
      (.operand xhF (zV : Vec (NT*DD))))
    let (cF, fl) ← pretty BS (.rowBiasF "%btF" (zV : Vec DD)
      (.operand scF (zV : Vec (NT*DD))))
    let (cCs, clsv) ← pretty BS (.clsSliceF (N := NP) (D := DD)
      (.operand fl (zV : Vec ((NP+1)*DD))))
    let (cLog, logits) ← pretty BS (denseF "%Wcls" "%bcls" (zM : Mat DD NC) zV
      (.operand clsv (zV : Vec DD)))
    -- loss cotangent: (softmax(logits) − onehot)/BS
    let (cSub, dyr) ← pretty BS (.sub (.softmaxDiv (.expe (.operand logits (zV : Vec NC))))
      (.operand "%onehot" zV))
    fwd := fwd ++ cFa ++ cFb ++ cF ++ cCs ++ cLog ++ cSub
    -- ═══ backward cotangent chain (proof-rendered) ═══
    let (cDd, cot_cls) ← pretty BS (.dotOut "%Wcls" (zM : Mat DD NC) (.operand "%dy" zV))
    let (cPad, cot_fl) ← pretty BS (.clsPadF (N := NP) (D := DD)
      (.operand cot_cls (zV : Vec DD)))
    let (cFsc, cot_xhF) ← pretty BS (.rowScaleF "%gF" (zV : Vec DD)
      (.operand cot_fl (zV : Vec (NT*DD))))
    let (cFbk, cot_bLout) ← pretty BS (.lnRowBack "%one" xcur EPS 0 1
      (zV : Vec (NT*DD)) (.operand cot_xhF zV))
    let mut bwd := s!"    %dy = stablehlo.divide {dyr}, %bsc : {ty [BS, NC]}\n" ++
      cDd ++ cPad ++ cFsc ++ cFbk
    -- blocks in reverse, threading the cotangent; per-block param grads as we go
    let mut paramG := ""
    let mut cot := cot_bLout
    for (i, b) in (((List.range DEPTH).map (· + 1)).zip blocks).reverse do
      let (c, cot_in, dQ, dK, dV, ch, cln2, cm1, cg, cln1) ← bwdBlock i cot b
      bwd := bwd ++ c
      paramG := paramG ++ blockParamGrads i b dQ dK dV ch cln2 cm1 cg cln1 cot
      cot := cot_in
    let cot_embed := cot
    -- ═══ remaining param grads (hand-emitted; Item C certified forms) ═══
    paramG := paramG ++
      -- final vector-LN γ/β: per-channel reduces off the saved normalize output
      vecLNParamGrad "%dgF" "%dbtF" xhF cot_fl NT DD ++
      -- pos-embed: dPos = Σ_batch dy at the embed output
      rs3 "%dposi" cot_embed NT DD ++
      s!"    %dpos = stablehlo.reduce(%dposi init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,NT,DD]}, tensor<f32>) -> {ty [NT,DD]}\n" ++
      -- CLS token: dCls = row-0 slice of the embed cotangent, batch-summed
      -- (final reshape to [1,D] — the committed signature's cls shape)
      s!"    %dclss = stablehlo.slice %dposi [0:{BS}, 0:1, 0:{DD}] : ({ty [BS,NT,DD]}) -> {ty [BS,1,DD]}\n" ++
      s!"    %dclsf = stablehlo.reshape %dclss : ({ty [BS,1,DD]}) -> {ty [BS,DD]}\n" ++
      s!"    %dclsr = stablehlo.reduce(%dclsf init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,DD]}, tensor<f32>) -> {ty [DD]}\n" ++
      s!"    %dcls = stablehlo.reshape %dclsr : ({ty [DD]}) -> {ty [1,DD]}\n" ++
      -- patch projection (stride-P non-overlapping conv = per-patch dense): im2col
      -- the image into patch-major rows [BS,NP,IC·P·P] (pure reshape/transpose —
      -- patches don't overlap), then ONE dot_general against the patch-token rows
      -- of the embed cotangent — the §E certified `dWp = Σ_p read·dy_(p+1,·)` form
      s!"    %dWpx6 = stablehlo.reshape %x : ({ty [BS, IC*HH*HH]}) -> {ty [BS,IC,PH,PP,PH,PP]}\n" ++
      s!"    %dWpxt = stablehlo.transpose %dWpx6, dims = [0, 2, 4, 1, 3, 5] : ({ty [BS,IC,PH,PP,PH,PP]}) -> {ty [BS,PH,PH,IC,PP,PP]}\n" ++
      s!"    %dWpxm = stablehlo.reshape %dWpxt : ({ty [BS,PH,PH,IC,PP,PP]}) -> {ty [BS, NP, IC*PP*PP]}\n" ++
      s!"    %dWpdy = stablehlo.slice %dposi [0:{BS}, 1:{NT}, 0:{DD}] : ({ty [BS,NT,DD]}) -> {ty [BS,NP,DD]}\n" ++
      s!"    %dWpr = stablehlo.dot_general %dWpxm, %dWpdy, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : ({ty [BS,NP,IC*PP*PP]}, {ty [BS,NP,DD]}) -> {ty [IC*PP*PP,DD]}\n" ++
      s!"    %dWpt = stablehlo.transpose %dWpr, dims = [1, 0] : ({ty [IC*PP*PP,DD]}) -> {ty [DD,IC*PP*PP]}\n" ++
      s!"    %dWp = stablehlo.reshape %dWpt : ({ty [DD,IC*PP*PP]}) -> {ty [DD,IC,PP,PP]}\n" ++
      s!"    %dbp = stablehlo.reduce(%dWpdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({ty [BS,NP,DD]}, tensor<f32>) -> {ty [DD]}\n" ++
      -- head: dWcls = clsᵀ·dy (M2 outer product, batch-contracted), dbcls = Σ dy
      s!"    %dWcls = stablehlo.dot_general {clsv}, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,DD]}, {ty [BS,NC]}) -> {ty [DD,NC]}\n" ++
      s!"    %dbcls = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,NC]}, tensor<f32>) -> {ty [NC]}\n"
    pure (fwd ++ bwd ++ paramG)
  let body : String := go.run' 0
  -- ═══ SGD over all 4 + 16·DEPTH + 4 params (forward order) + signature ═══
  let allParams : List (String × String) :=
    [("Wp", ty [DD,IC,PP,PP]), ("bp", ty [DD]), ("cls", ty [1,DD]), ("pos", ty [NT,DD])]
    ++ (List.range DEPTH).flatMap (fun i => blkParams (i + 1))
    ++ [("gF", ty [DD]), ("btF", ty [DD]),
        ("Wcls", ty [DD,NC]), ("bcls", ty [NC])]
  let upd := String.join (allParams.map (fun (nm, t) => sgd s!"%{nm}" s!"%d{nm}" t))
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [BS, IC*HH*HH]) :: allParams.map (fun (nm, t) => s!"%{nm}: {t}")
      ++ ["%onehot: " ++ ty [BS,NC]])
  let retTyL := String.intercalate ", " (allParams.map (fun (_, t) => t))
  let retVals := String.intercalate ", " (allParams.map (fun (nm, _) => s!"%{nm}n"))
  return "module @m {\n" ++ s!"  func.func @vit_train_step({argSig}) -> ({retTyL}) " ++ "{\n" ++
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %bsc = stablehlo.constant dense<{BS}.0> : {ty [BS,NC]}\n" ++
    body ++ upd ++
    s!"    return {retVals} : {retTyL}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := trainStep
  IO.println s!"rendered structured ViT (representative) train step: {mlir.length} chars"
  IO.FS.createDirAll "/tmp/vitpc"
  IO.FS.writeFile "/tmp/vitpc/train_step.mlir" mlir
  let cargs ← ireeCompileArgs "/tmp/vitpc/train_step.mlir" "/tmp/vitpc/train_step.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 5000}"
  else
    IO.println "structured ViT representative train step iree-compile OK → /tmp/vitpc/train_step.mlir"

#eval main
