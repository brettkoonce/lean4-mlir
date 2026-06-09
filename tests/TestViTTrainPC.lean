import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! # ViT Item B — structured representative train-step render (proof-rendered)

The ViT peer of `tests/TestConvNeXtTrainPC.lean`, at the **representative** `vitForward2`
config (the proven graph `vitFwdGraph`, Item A): patchSize-1 patch embed (per-pixel dense
+ CLS + pos) → 2 distinct-param pre-norm transformer blocks (heads = 1) → final per-token
LN → CLS slice → dense head. CIFAR-shaped: 3×8² in → 64 patches + CLS = 65 tokens,
D = 32, mlpDim = 128, 10 classes, BS = 32.

Forward AND the whole backward cotangent chain are proof-rendered through `pretty` over
the very tokens of `vitFwdGraph` — forward (`patchEmbedF`/`lnRowF`/`denseRowF`/`matmulF`/
`transposeF`/`scaleF`/`softmaxRowF`/`geluF`/`addV`/`clsSliceF`/`denseF`) and backward
(`dotOut`, `clsPadF`, `lnRowBack`, `denseRowBack`, `geluBack`, `softmaxRowBack`, and the
**SDPA 3-path backward spelled with the forward `matmulF`/`transposeF` on cotangents** —
matmul's VJP IS matmul: `dP = dO·Vᵀ`, `dV = Pᵀ·dO`, `dQ = s·dS·K`, `dK = s·dSᵀ·Q` — the
proven `sdpa_back_{Q,K,V}` shapes; no backward-only matmul token). Residual fan-ins are
`addV`; the Q/K/V three-way fan-in at LN₁'s output is two `addV`s.

Only the no-SHlo-constructor pieces are hand-emitted, each certified in
`ViTClose.lean` (Item C): per-token dense `dW = Σ_{b,tokens} x⊗dy` / `db = Σ dy`
(`vit_render_rowdense{W,b}_certified`), rowwise scalar-LN `dγ = Σ dy·x̂` / `dβ = Σ dy`
(`vit_render_rowln{gamma,beta}_certified`), `dPos = Σ_b dy` (`vit_render_pos_certified`),
`dCls = row-0 slice` (`vit_render_cls_certified`), the patchSize-1 patch-projection
`dWp`/`dbp` over the patch rows (`vit_render_patch{W,b}_certified`), and the head
`dWcls = clsᵀ·dy` / `dbcls = Σ dy` (M2).

No committed same-signature renderer exists (the committed `TestViTTrain.lean` is the
full ViT-Tiny: 224², depth-12, 3 heads, vector-LN), so validation is the
`scripts/render_parity.py` ref-only smoke: compile + run on the GPU, all 40 updated
params finite:
  `scripts/render_parity.py --fn vit_rep_train_step --ref /tmp/vitpc/train_step.mlir`

Run: `IREE_BACKEND=rocm lake env lean tests/TestViTTrainPC.lean`
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 32
private def IC : Nat := 3     -- input channels
private def HH : Nat := 8     -- spatial
private def NP : Nat := 64    -- patches (8×8 at patchSize 1)
private def NT : Nat := 65    -- tokens (patches + CLS)
private def DD : Nat := 32    -- embed dim (heads = 1)
private def MD : Nat := 128   -- MLP dim (4×)
private def EPS : String := "1.0e-6"
private def LR : String := "0.1"
private def SCALE : String := "0.17677669529663687"   -- 1/√32 = 1/√d_head

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

/-- rowwise scalar-LN dγ = Σ_(b,tok,D) dy·x̂, dβ = Σ dy; recompute x̂ per token row from
    the saved LN input (stats over the row axis [2] — `lnRowF`'s own emission text).
    The rendered `rowLN_grad_gamma/beta` (`vit_render_rowln{gamma,beta}_certified`). -/
private def lnRowParamGrad (dgr dbe inFlat dyFlat : String) (t f : Nat) : String :=
  let tn := ty [BS, t, f]
  rs3 s!"{dgr}xi" inFlat t f ++ rs3 s!"{dgr}dyi" dyFlat t f ++
  s!"    {dgr}nf = stablehlo.constant dense<{f}.0> : {tn}\n" ++
  s!"    {dgr}ep = stablehlo.constant dense<{EPS}> : {tn}\n" ++
  s!"    {dgr}smr = stablehlo.reduce({dgr}xi init: %sc) applies stablehlo.add across dimensions = [2] : ({tn}, tensor<f32>) -> {ty [BS, t]}\n" ++
  s!"    {dgr}sm = stablehlo.broadcast_in_dim {dgr}smr, dims = [0, 1] : ({ty [BS, t]}) -> {tn}\n" ++
  s!"    {dgr}mu = stablehlo.divide {dgr}sm, {dgr}nf : {tn}\n" ++
  s!"    {dgr}xc = stablehlo.subtract {dgr}xi, {dgr}mu : {tn}\n" ++
  s!"    {dgr}sq = stablehlo.multiply {dgr}xc, {dgr}xc : {tn}\n" ++
  s!"    {dgr}vsr = stablehlo.reduce({dgr}sq init: %sc) applies stablehlo.add across dimensions = [2] : ({tn}, tensor<f32>) -> {ty [BS, t]}\n" ++
  s!"    {dgr}vs = stablehlo.broadcast_in_dim {dgr}vsr, dims = [0, 1] : ({ty [BS, t]}) -> {tn}\n" ++
  s!"    {dgr}vr = stablehlo.divide {dgr}vs, {dgr}nf : {tn}\n" ++
  s!"    {dgr}ve = stablehlo.add {dgr}vr, {dgr}ep : {tn}\n" ++
  s!"    {dgr}istd = stablehlo.rsqrt {dgr}ve : {tn}\n" ++
  s!"    {dgr}xh = stablehlo.multiply {dgr}xc, {dgr}istd : {tn}\n" ++
  s!"    {dgr}p = stablehlo.multiply {dgr}dyi, {dgr}xh : {tn}\n" ++
  s!"    {dgr} = stablehlo.reduce({dgr}p init: %sc) applies stablehlo.add across dimensions = [0, 1, 2] : ({tn}, tensor<f32>) -> tensor<f32>\n" ++
  s!"    {dbe} = stablehlo.reduce({dgr}dyi init: %sc) applies stablehlo.add across dimensions = [0, 1, 2] : ({tn}, tensor<f32>) -> tensor<f32>\n"

private def sgd (θ dθ ty' : String) : String :=
  s!"    {θ}l = stablehlo.constant dense<{LR}> : {ty'}\n" ++
  s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
  s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"

-- ════════════ captured forward names per ViT block ════════════
private structure FNames where  -- flat SSA names from `pretty`
  xin : String   -- block input (= the attn residual skip; saved pre-LN1 input)
  ln1 : String   -- LN1 out (= Q/K/V dense input)
  q : String
  k : String
  v : String
  ss : String    -- scaled scores (pre-softmax; softmaxRowBack's saved input)
  p : String     -- post-softmax weights
  att : String   -- P·V (out-proj input)
  h : String     -- attn-sublayer out (= the MLP residual skip; saved pre-LN2 input)
  ln2 : String   -- LN2 out (= fc1 input)
  m1 : String    -- fc1 out (pre-GELU)
  g : String     -- GELU out (fc2 input)
  bout : String  -- block out

/-- One ViT block forward via `pretty` — exactly the `vitBlockGraph` tokens. -/
private def fwdBlock (i : Nat) (xin : String) : StateM Nat (String × FNames) := do
  let (k1, ln1) ← pretty BS (.lnRowF s!"%g1_{i}" s!"%bt1_{i}" EPS 0 0 0
    (.operand xin (zV : Vec (NT*DD))))
  let (k2, q) ← pretty BS (.denseRowF s!"%Wq{i}" s!"%bq{i}" (zM : Mat DD DD) zV
    (.operand ln1 (zV : Vec (NT*DD))))
  let (k3, kk) ← pretty BS (.denseRowF s!"%Wk{i}" s!"%bk{i}" (zM : Mat DD DD) zV
    (.operand ln1 (zV : Vec (NT*DD))))
  let (k4, v) ← pretty BS (.denseRowF s!"%Wv{i}" s!"%bv{i}" (zM : Mat DD DD) zV
    (.operand ln1 (zV : Vec (NT*DD))))
  let (k5, kT) ← pretty BS (.transposeF (.operand kk (zV : Vec (NT*DD))))
  let (k6, sc) ← pretty BS (.matmulF (.operand q (zV : Vec (NT*DD)))
    (.operand kT (zV : Vec (DD*NT))))
  let (k7, ss) ← pretty BS (.scaleF SCALE 0 (.operand sc (zV : Vec (NT*NT))))
  let (k8, p) ← pretty BS (.softmaxRowF (.operand ss (zV : Vec (NT*NT))))
  let (k9, att) ← pretty BS (.matmulF (.operand p (zV : Vec (NT*NT)))
    (.operand v (zV : Vec (NT*DD))))
  let (k10, o) ← pretty BS (.denseRowF s!"%Wo{i}" s!"%bo{i}" (zM : Mat DD DD) zV
    (.operand att (zV : Vec (NT*DD))))
  let (k11, h) ← pretty BS (.addV (.operand xin (zV : Vec (NT*DD))) (.operand o zV))
  let (k12, ln2) ← pretty BS (.lnRowF s!"%g2_{i}" s!"%bt2_{i}" EPS 0 0 0
    (.operand h (zV : Vec (NT*DD))))
  let (k13, m1) ← pretty BS (.denseRowF s!"%Wfc1{i}" s!"%bfc1{i}" (zM : Mat DD MD) zV
    (.operand ln2 (zV : Vec (NT*DD))))
  let (k14, g) ← pretty BS (.geluF (.operand m1 (zV : Vec (NT*MD))))
  let (k15, m2) ← pretty BS (.denseRowF s!"%Wfc2{i}" s!"%bfc2{i}" (zM : Mat MD DD) zV
    (.operand g (zV : Vec (NT*MD))))
  let (k16, bout) ← pretty BS (.addV (.operand h (zV : Vec (NT*DD))) (.operand m2 zV))
  pure (k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ k7 ++ k8 ++ k9 ++ k10 ++ k11 ++ k12 ++
        k13 ++ k14 ++ k15 ++ k16,
        ⟨xin, ln1, q, kk, v, ss, p, att, h, ln2, m1, g, bout⟩)

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
  let (k4, cot_h_mlp) ← pretty BS (.lnRowBack s!"%g2_{i}" b.h EPS 0 0
    (zV : Vec (NT*DD)) (.operand cot_ln2 zV))
  let (k5, cot_h) ← pretty BS (.addV (.operand dy (zV : Vec (NT*DD)))
    (.operand cot_h_mlp zV))
  -- attn sublayer back: h = xin + Wo·SDPA(q,k,v)
  let (k6, cot_att) ← pretty BS (.denseRowBack s!"%Wo{i}" (zM : Mat DD DD)
    (.operand cot_h (zV : Vec (NT*DD))))
  -- SDPA 3-path: dP = dO·Vᵀ, dV = Pᵀ·dO, dS = softmaxRowBack, undo scale,
  -- dQ = dS·K, dK = dSᵀ·Q  (the proven sdpa_back_{Q,K,V} shapes)
  let (k7, vT) ← pretty BS (.transposeF (.operand b.v (zV : Vec (NT*DD))))
  let (k8, dP) ← pretty BS (.matmulF (.operand cot_att (zV : Vec (NT*DD)))
    (.operand vT (zV : Vec (DD*NT))))
  let (k9, pT) ← pretty BS (.transposeF (.operand b.p (zV : Vec (NT*NT))))
  let (k10, dV) ← pretty BS (.matmulF (.operand pT (zV : Vec (NT*NT)))
    (.operand cot_att (zV : Vec (NT*DD))))
  let (k11, dS) ← pretty BS (.softmaxRowBack b.ss (zV : Vec (NT*NT))
    (.operand dP zV))
  let (k12, dSs) ← pretty BS (.scaleF SCALE 0 (.operand dS (zV : Vec (NT*NT))))
  let (k13, dQ) ← pretty BS (.matmulF (.operand dSs (zV : Vec (NT*NT)))
    (.operand b.k (zV : Vec (NT*DD))))
  let (k14, dSsT) ← pretty BS (.transposeF (.operand dSs (zV : Vec (NT*NT))))
  let (k15, dK) ← pretty BS (.matmulF (.operand dSsT (zV : Vec (NT*NT)))
    (.operand b.q (zV : Vec (NT*DD))))
  -- Q/K/V dense backs fan IN at LN1's output (three cotangents sum)
  let (k16, cq) ← pretty BS (.denseRowBack s!"%Wq{i}" (zM : Mat DD DD)
    (.operand dQ (zV : Vec (NT*DD))))
  let (k17, ck) ← pretty BS (.denseRowBack s!"%Wk{i}" (zM : Mat DD DD)
    (.operand dK (zV : Vec (NT*DD))))
  let (k18, cv) ← pretty BS (.denseRowBack s!"%Wv{i}" (zM : Mat DD DD)
    (.operand dV (zV : Vec (NT*DD))))
  let (k19, s1) ← pretty BS (.addV (.operand cq (zV : Vec (NT*DD))) (.operand ck zV))
  let (k20, cot_ln1) ← pretty BS (.addV (.operand s1 (zV : Vec (NT*DD))) (.operand cv zV))
  let (k21, cot_xin_attn) ← pretty BS (.lnRowBack s!"%g1_{i}" b.xin EPS 0 0
    (zV : Vec (NT*DD)) (.operand cot_ln1 zV))
  let (k22, cot_xin) ← pretty BS (.addV (.operand cot_h (zV : Vec (NT*DD)))
    (.operand cot_xin_attn zV))
  pure (k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ k7 ++ k8 ++ k9 ++ k10 ++ k11 ++ k12 ++
        k13 ++ k14 ++ k15 ++ k16 ++ k17 ++ k18 ++ k19 ++ k20 ++ k21 ++ k22,
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
  -- scalar LN1/LN2 γ/β from the saved pre-LN inputs
  lnRowParamGrad s!"%dg1_{i}" s!"%dbt1_{i}" b.xin cot_ln1 NT DD ++
  lnRowParamGrad s!"%dg2_{i}" s!"%dbt2_{i}" b.h cot_ln2 NT DD

/-- per-block param (name, type) list, forward order (matches `vitFwdGraph` arg order). -/
private def blkParams (i : Nat) : List (String × String) :=
  [(s!"g1_{i}", "tensor<f32>"), (s!"bt1_{i}", "tensor<f32>"),
   (s!"Wq{i}", ty [DD,DD]), (s!"bq{i}", ty [DD]),
   (s!"Wk{i}", ty [DD,DD]), (s!"bk{i}", ty [DD]),
   (s!"Wv{i}", ty [DD,DD]), (s!"bv{i}", ty [DD]),
   (s!"Wo{i}", ty [DD,DD]), (s!"bo{i}", ty [DD]),
   (s!"g2_{i}", "tensor<f32>"), (s!"bt2_{i}", "tensor<f32>"),
   (s!"Wfc1{i}", ty [DD,MD]), (s!"bfc1{i}", ty [MD]),
   (s!"Wfc2{i}", ty [MD,DD]), (s!"bfc2{i}", ty [DD])]

private def trainStep : String := Id.run do
  let go : StateM Nat String := do
    -- ═══ forward (proof-rendered; the vitFwdGraph tokens in graph order) ═══
    let (cE, embed) ← pretty BS (.patchEmbedF (ic := IC) (H := HH) (W := HH) (P := 1)
      (N := NP) (D := DD) "%Wp" "%bp" "%cls" "%pos"
      (zK : Kernel4 DD IC 1 1) zV zV (zM : Mat (NP+1) DD) (.operand "%x" zV))
    let (cB1, b1) ← fwdBlock 1 embed
    let (cB2, b2) ← fwdBlock 2 b1.bout
    let (cF, fl) ← pretty BS (.lnRowF "%gF" "%btF" EPS 0 0 0
      (.operand b2.bout (zV : Vec (NT*DD))))
    let (cCs, clsv) ← pretty BS (.clsSliceF (N := NP) (D := DD)
      (.operand fl (zV : Vec ((NP+1)*DD))))
    let (cLog, logits) ← pretty BS (denseF "%Wcls" "%bcls" (zM : Mat DD 10) zV
      (.operand clsv (zV : Vec DD)))
    -- loss cotangent: (softmax(logits) − onehot)/BS
    let (cSub, dyr) ← pretty BS (.sub (.softmaxDiv (.expe (.operand logits (zV : Vec 10))))
      (.operand "%onehot" zV))
    let fwd := cE ++ cB1 ++ cB2 ++ cF ++ cCs ++ cLog ++ cSub
    -- ═══ backward cotangent chain (proof-rendered) ═══
    let (cDd, cot_cls) ← pretty BS (.dotOut "%Wcls" (zM : Mat DD 10) (.operand "%dy" zV))
    let (cPad, cot_fl) ← pretty BS (.clsPadF (N := NP) (D := DD)
      (.operand cot_cls (zV : Vec DD)))
    let (cFb, cot_b2out) ← pretty BS (.lnRowBack "%gF" b2.bout EPS 0 0
      (zV : Vec (NT*DD)) (.operand cot_fl (zV : Vec ((NP+1)*DD))))
    let (cB2b, cot_b1out, dQ2, dK2, dV2, ch2, cln2_2, cm1_2, cg2, cln1_2) ←
      bwdBlock 2 cot_b2out b2
    let (cB1b, cot_embed, dQ1, dK1, dV1, ch1, cln2_1, cm1_1, cg1, cln1_1) ←
      bwdBlock 1 cot_b1out b1
    let bwd := s!"    %dy = stablehlo.divide {dyr}, %bsc : {ty [BS, 10]}\n" ++
      cDd ++ cPad ++ cFb ++ cB2b ++ cB1b
    -- ═══ param grads (hand-emitted; Item C certified forms) ═══
    let paramG :=
      blockParamGrads 2 b2 dQ2 dK2 dV2 ch2 cln2_2 cm1_2 cg2 cln1_2 cot_b2out ++
      blockParamGrads 1 b1 dQ1 dK1 dV1 ch1 cln2_1 cm1_1 cg1 cln1_1 cot_b1out ++
      -- final LN γ/β (saved block-2 out)
      lnRowParamGrad "%dgF" "%dbtF" b2.bout cot_fl NT DD ++
      -- pos-embed: dPos = Σ_batch dy at the embed output
      rs3 "%dposi" cot_embed NT DD ++
      s!"    %dpos = stablehlo.reduce(%dposi init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,NT,DD]}, tensor<f32>) -> {ty [NT,DD]}\n" ++
      -- CLS token: dCls = row-0 slice of the embed cotangent, batch-summed
      s!"    %dclss = stablehlo.slice %dposi [0:{BS}, 0:1, 0:{DD}] : ({ty [BS,NT,DD]}) -> {ty [BS,1,DD]}\n" ++
      s!"    %dclsf = stablehlo.reshape %dclss : ({ty [BS,1,DD]}) -> {ty [BS,DD]}\n" ++
      s!"    %dcls = stablehlo.reduce(%dclsf init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,DD]}, tensor<f32>) -> {ty [DD]}\n" ++
      -- patch projection (patchSize 1 = per-pixel dense over the 64 patch rows):
      -- X = image pixels token-major [BS,64,IC]; dY = embed-cot rows 1..65
      s!"    %dWpx4 = stablehlo.reshape %x : ({ty [BS, IC*HH*HH]}) -> {ty [BS,IC,NP]}\n" ++
      s!"    %dWpxt = stablehlo.transpose %dWpx4, dims = [0, 2, 1] : ({ty [BS,IC,NP]}) -> {ty [BS,NP,IC]}\n" ++
      s!"    %dWpdy = stablehlo.slice %dposi [0:{BS}, 1:{NT}, 0:{DD}] : ({ty [BS,NT,DD]}) -> {ty [BS,NP,DD]}\n" ++
      s!"    %dWpr = stablehlo.dot_general %dWpxt, %dWpdy, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : ({ty [BS,NP,IC]}, {ty [BS,NP,DD]}) -> {ty [IC,DD]}\n" ++
      s!"    %dWpt = stablehlo.transpose %dWpr, dims = [1, 0] : ({ty [IC,DD]}) -> {ty [DD,IC]}\n" ++
      s!"    %dWp = stablehlo.reshape %dWpt : ({ty [DD,IC]}) -> {ty [DD,IC,1,1]}\n" ++
      s!"    %dbp = stablehlo.reduce(%dWpdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({ty [BS,NP,DD]}, tensor<f32>) -> {ty [DD]}\n" ++
      -- head: dWcls = clsᵀ·dy (M2 outer product, batch-contracted), dbcls = Σ dy
      s!"    %dWcls = stablehlo.dot_general {clsv}, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,DD]}, {ty [BS,10]}) -> {ty [DD,10]}\n" ++
      s!"    %dbcls = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,10]}, tensor<f32>) -> {ty [10]}\n"
    pure (fwd ++ bwd ++ paramG)
  let body : String := go.run' 0
  -- ═══ SGD over all 40 params (forward order) + signature ═══
  let allParams : List (String × String) :=
    [("Wp", ty [DD,IC,1,1]), ("bp", ty [DD]), ("cls", ty [DD]), ("pos", ty [NT,DD])]
    ++ blkParams 1 ++ blkParams 2
    ++ [("gF", "tensor<f32>"), ("btF", "tensor<f32>"),
        ("Wcls", ty [DD,10]), ("bcls", ty [10])]
  let upd := String.join (allParams.map (fun (nm, t) => sgd s!"%{nm}" s!"%d{nm}" t))
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [BS, IC*HH*HH]) :: allParams.map (fun (nm, t) => s!"%{nm}: {t}")
      ++ ["%onehot: " ++ ty [BS,10]])
  let retTyL := String.intercalate ", " (allParams.map (fun (_, t) => t))
  let retVals := String.intercalate ", " (allParams.map (fun (nm, _) => s!"%{nm}n"))
  return "module @m {\n" ++ s!"  func.func @vit_rep_train_step({argSig}) -> ({retTyL}) " ++ "{\n" ++
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    s!"    %bsc = stablehlo.constant dense<{BS}.0> : {ty [BS,10]}\n" ++
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
