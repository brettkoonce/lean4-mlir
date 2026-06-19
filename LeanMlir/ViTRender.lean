import LeanMlir.Proofs.StableHLO

/-! # ch10 ViT — verified-faithful StableHLO render fragments (shared library)

Hand-rendered batched StableHLO string fragments for the Vision Transformer
(ch7/ch8/ch9 style), each line what the proven-faithful emitter produces — the
matmuls are `dot_general` (proven `dense`), the row-softmax is the V1 op pattern
(`softmaxRowF`/`softmaxRowBack`, plain exp/sum), GELU is the ch9 `geluF` tanh
approximation, LayerNorm is `layerNormForward` (per-token over D, γ=1/β=0) ∘ a
per-channel `[D]` affine (ConvNeXt `layerScale` + a `[D]` bias). NOT a single
`den(trainStep)` theorem — faithful PER-OP, validated by the Lean gradchecks
(TestSDPA/TestMHSA/TestViTBlock) and by training.

Every fragment is prefix-parameterized (`p`) so it can be instantiated many times
(e.g. 12 distinct blocks) without SSA collisions. A forward fragment KEEPS the
intermediate SSA values its backward needs; the matching backward fragment reuses
them (after a forward recompute), exactly like ch8's `seFwd`/`seBack`. All
fragments assume a `%sc` (f32 0) constant is in scope (the reduce init). -/

namespace ViTRender
open Proofs Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Per-channel [D] LayerNorm  (normalize per-token over D, then γ[D]·x̂+β[D])
-- ════════════════════════════════════════════════════════════════

/-- **LayerNorm forward**, prefix `p`, over `[b,n,d]` (normalize each token over
    the `d` axis [2], then per-channel affine). KEEPS `%{p}xhat` `[b,n,d]`,
    `%{p}istdb` `[b,n,d]`, `%{p}nf` `[b,n]` for the backward. Result `%{p}y`. -/
def lnFwd (p x g b : String) (bb n d : Nat) (eps : String) : String :=
  s!"    %{p}sum = stablehlo.reduce({x} init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [bb,n,d]}, tensor<f32>) -> {ty [bb,n]}\n" ++
  s!"    %{p}nf = stablehlo.constant dense<{d}.0> : {ty [bb,n]}\n" ++
  s!"    %{p}mu = stablehlo.divide %{p}sum, %{p}nf : {ty [bb,n]}\n" ++
  s!"    %{p}mub = stablehlo.broadcast_in_dim %{p}mu, dims = [0, 1] : ({ty [bb,n]}) -> {ty [bb,n,d]}\n" ++
  s!"    %{p}xc = stablehlo.subtract {x}, %{p}mub : {ty [bb,n,d]}\n" ++
  s!"    %{p}sq = stablehlo.multiply %{p}xc, %{p}xc : {ty [bb,n,d]}\n" ++
  s!"    %{p}vsum = stablehlo.reduce(%{p}sq init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [bb,n,d]}, tensor<f32>) -> {ty [bb,n]}\n" ++
  s!"    %{p}var = stablehlo.divide %{p}vsum, %{p}nf : {ty [bb,n]}\n" ++
  s!"    %{p}eps = stablehlo.constant dense<{eps}> : {ty [bb,n]}\n" ++
  s!"    %{p}ve = stablehlo.add %{p}var, %{p}eps : {ty [bb,n]}\n" ++
  s!"    %{p}istd = stablehlo.rsqrt %{p}ve : {ty [bb,n]}\n" ++
  s!"    %{p}istdb = stablehlo.broadcast_in_dim %{p}istd, dims = [0, 1] : ({ty [bb,n]}) -> {ty [bb,n,d]}\n" ++
  s!"    %{p}xhat = stablehlo.multiply %{p}xc, %{p}istdb : {ty [bb,n,d]}\n" ++
  s!"    %{p}gb = stablehlo.broadcast_in_dim {g}, dims = [2] : ({ty [d]}) -> {ty [bb,n,d]}\n" ++
  s!"    %{p}bbc = stablehlo.broadcast_in_dim {b}, dims = [2] : ({ty [d]}) -> {ty [bb,n,d]}\n" ++
  s!"    %{p}gx = stablehlo.multiply %{p}xhat, %{p}gb : {ty [bb,n,d]}\n" ++
  s!"    %{p}y = stablehlo.add %{p}gx, %{p}bbc : {ty [bb,n,d]}\n"

/-- **LayerNorm backward**, prefix `p`. Reuses `%{p}xhat`/`%{p}istdb`/`%{p}nf`
    from a preceding `lnFwd p`. `dy` is the output cotangent, `g` the γ `[d]`.
    Produces `%{p}dx` `[b,n,d]`, `%{p}dg` `[d]`, `%{p}db` `[d]`.
    Affine back: `dx̂ = dy⊙γ`, `dγ = Σ dy⊙x̂`, `dβ = Σ dy`. Normalize back (γ=1):
    `dx = istd·(dx̂ − mean_d(dx̂) − x̂·mean_d(dx̂⊙x̂))` (= `bn_grad_input` with γ=1).
    (No `x` arg — the LN input is not needed; `x̂`/`istd` come from the fwd recompute.) -/
def lnBack (p g dy : String) (bb n d : Nat) : String :=
  -- affine back
  s!"    %{p}gbk = stablehlo.broadcast_in_dim {g}, dims = [2] : ({ty [d]}) -> {ty [bb,n,d]}\n" ++
  s!"    %{p}dxhat = stablehlo.multiply {dy}, %{p}gbk : {ty [bb,n,d]}\n" ++
  s!"    %{p}dgpre = stablehlo.multiply {dy}, %{p}xhat : {ty [bb,n,d]}\n" ++
  s!"    %{p}dg = stablehlo.reduce(%{p}dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({ty [bb,n,d]}, tensor<f32>) -> {ty [d]}\n" ++
  s!"    %{p}db = stablehlo.reduce({dy} init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({ty [bb,n,d]}, tensor<f32>) -> {ty [d]}\n" ++
  -- normalize back (γ=1)
  s!"    %{p}m1s = stablehlo.reduce(%{p}dxhat init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [bb,n,d]}, tensor<f32>) -> {ty [bb,n]}\n" ++
  s!"    %{p}m1 = stablehlo.divide %{p}m1s, %{p}nf : {ty [bb,n]}\n" ++
  s!"    %{p}dxxh = stablehlo.multiply %{p}dxhat, %{p}xhat : {ty [bb,n,d]}\n" ++
  s!"    %{p}m2s = stablehlo.reduce(%{p}dxxh init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [bb,n,d]}, tensor<f32>) -> {ty [bb,n]}\n" ++
  s!"    %{p}m2 = stablehlo.divide %{p}m2s, %{p}nf : {ty [bb,n]}\n" ++
  s!"    %{p}m1b = stablehlo.broadcast_in_dim %{p}m1, dims = [0, 1] : ({ty [bb,n]}) -> {ty [bb,n,d]}\n" ++
  s!"    %{p}m2b = stablehlo.broadcast_in_dim %{p}m2, dims = [0, 1] : ({ty [bb,n]}) -> {ty [bb,n,d]}\n" ++
  s!"    %{p}t1 = stablehlo.subtract %{p}dxhat, %{p}m1b : {ty [bb,n,d]}\n" ++
  s!"    %{p}xm2 = stablehlo.multiply %{p}xhat, %{p}m2b : {ty [bb,n,d]}\n" ++
  s!"    %{p}t2 = stablehlo.subtract %{p}t1, %{p}xm2 : {ty [bb,n,d]}\n" ++
  s!"    %{p}dx = stablehlo.multiply %{p}istdb, %{p}t2 : {ty [bb,n,d]}\n"

-- ════════════════════════════════════════════════════════════════
-- § GELU (tanh approximation) — elementwise over [b,n,m]
-- ════════════════════════════════════════════════════════════════

/-- **GELU forward** (`geluF` lines) over `[b,n,m]`, prefix `p`, input SSA `inp`.
    Result `%{p}a`. -/
def geluActF (p inp : String) (bb n m : Nat) : String :=
  let t := ty [bb,n,m]
  s!"    %{p}x2 = stablehlo.multiply {inp}, {inp} : {t}\n" ++
  s!"    %{p}x3 = stablehlo.multiply %{p}x2, {inp} : {t}\n" ++
  s!"    %{p}ck = stablehlo.constant dense<0.044715> : {t}\n" ++
  s!"    %{p}kx3 = stablehlo.multiply %{p}ck, %{p}x3 : {t}\n" ++
  s!"    %{p}inn = stablehlo.add {inp}, %{p}kx3 : {t}\n" ++
  s!"    %{p}csqrt = stablehlo.constant dense<0.7978845608028654> : {t}\n" ++
  s!"    %{p}u = stablehlo.multiply %{p}csqrt, %{p}inn : {t}\n" ++
  s!"    %{p}t = stablehlo.tanh %{p}u : {t}\n" ++
  s!"    %{p}one = stablehlo.constant dense<1.0> : {t}\n" ++
  s!"    %{p}opt = stablehlo.add %{p}one, %{p}t : {t}\n" ++
  s!"    %{p}chalf = stablehlo.constant dense<0.5> : {t}\n" ++
  s!"    %{p}hx = stablehlo.multiply %{p}chalf, {inp} : {t}\n" ++
  s!"    %{p}a = stablehlo.multiply %{p}hx, %{p}opt : {t}\n"

/-- **GELU backward** (`geluBack` lines) over `[b,n,m]`, prefix `p`. Recomputes
    `gelu'(x)` from the saved pre-activation `xpre`; `dy` is the cotangent.
    Result `%{p}dx`. -/
def geluActBack (p xpre dy : String) (bb n m : Nat) : String :=
  let t := ty [bb,n,m]
  s!"    %{p}bx2 = stablehlo.multiply {xpre}, {xpre} : {t}\n" ++
  s!"    %{p}bx3 = stablehlo.multiply %{p}bx2, {xpre} : {t}\n" ++
  s!"    %{p}bck = stablehlo.constant dense<0.044715> : {t}\n" ++
  s!"    %{p}bkx3 = stablehlo.multiply %{p}bck, %{p}bx3 : {t}\n" ++
  s!"    %{p}binn = stablehlo.add {xpre}, %{p}bkx3 : {t}\n" ++
  s!"    %{p}bcsqrt = stablehlo.constant dense<0.7978845608028654> : {t}\n" ++
  s!"    %{p}bu = stablehlo.multiply %{p}bcsqrt, %{p}binn : {t}\n" ++
  s!"    %{p}bt = stablehlo.tanh %{p}bu : {t}\n" ++
  s!"    %{p}bone = stablehlo.constant dense<1.0> : {t}\n" ++
  s!"    %{p}bopt = stablehlo.add %{p}bone, %{p}bt : {t}\n" ++
  s!"    %{p}bchalf = stablehlo.constant dense<0.5> : {t}\n" ++
  s!"    %{p}bterm1 = stablehlo.multiply %{p}bchalf, %{p}bopt : {t}\n" ++
  s!"    %{p}bt2 = stablehlo.multiply %{p}bt, %{p}bt : {t}\n" ++
  s!"    %{p}bomt2 = stablehlo.subtract %{p}bone, %{p}bt2 : {t}\n" ++
  s!"    %{p}bhx = stablehlo.multiply %{p}bchalf, {xpre} : {t}\n" ++
  s!"    %{p}bhxo = stablehlo.multiply %{p}bhx, %{p}bomt2 : {t}\n" ++
  s!"    %{p}bc3b = stablehlo.constant dense<0.134145> : {t}\n" ++
  s!"    %{p}ba3x2 = stablehlo.multiply %{p}bc3b, %{p}bx2 : {t}\n" ++
  s!"    %{p}bin2 = stablehlo.add %{p}bone, %{p}ba3x2 : {t}\n" ++
  s!"    %{p}bup = stablehlo.multiply %{p}bcsqrt, %{p}bin2 : {t}\n" ++
  s!"    %{p}bterm2 = stablehlo.multiply %{p}bhxo, %{p}bup : {t}\n" ++
  s!"    %{p}bgp = stablehlo.add %{p}bterm1, %{p}bterm2 : {t}\n" ++
  s!"    %{p}dx = stablehlo.multiply {dy}, %{p}bgp : {t}\n"

-- ════════════════════════════════════════════════════════════════
-- § MLP sublayer  fc2 ∘ gelu ∘ fc1   over [b,n,d] (hidden m)
-- ════════════════════════════════════════════════════════════════

/-- **MLP forward**, prefix `p`. KEEPS `%{p}h1` `[b,n,m]` (pre-gelu) and
    `%{p}ga` `[b,n,m]` (post-gelu) for the backward. Result `%{p}y` `[b,n,d]`. -/
def mlpFwd (p x Wfc1 bfc1 Wfc2 bfc2 : String) (bb n d m : Nat) : String :=
  s!"    %{p}h1d = stablehlo.dot_general {x}, {Wfc1}, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : ({ty [bb,n,d]}, {ty [d,m]}) -> {ty [bb,n,m]}\n" ++
  s!"    %{p}h1bb = stablehlo.broadcast_in_dim {bfc1}, dims = [2] : ({ty [m]}) -> {ty [bb,n,m]}\n" ++
  s!"    %{p}h1 = stablehlo.add %{p}h1d, %{p}h1bb : {ty [bb,n,m]}\n" ++
  geluActF s!"{p}g" s!"%{p}h1" bb n m ++           -- result %{p}ga
  s!"    %{p}y2d = stablehlo.dot_general %{p}ga, {Wfc2}, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : ({ty [bb,n,m]}, {ty [m,d]}) -> {ty [bb,n,d]}\n" ++
  s!"    %{p}y2bb = stablehlo.broadcast_in_dim {bfc2}, dims = [2] : ({ty [d]}) -> {ty [bb,n,d]}\n" ++
  s!"    %{p}y = stablehlo.add %{p}y2d, %{p}y2bb : {ty [bb,n,d]}\n"

/-- **MLP backward**, prefix `p`. Reuses `%{p}h1`/`%{p}ga` from `mlpFwd p`. `dy`
    `[b,n,d]`. Produces `%{p}dx` `[b,n,d]` + `%{p}dWfc1` `[d,m]`, `%{p}dbfc1` `[m]`,
    `%{p}dWfc2` `[m,d]`, `%{p}dbfc2` `[d]`. -/
def mlpBack (p x Wfc1 Wfc2 dy : String) (bb n d m : Nat) : String :=
  -- fc2 back
  s!"    %{p}da1 = stablehlo.dot_general {dy}, {Wfc2}, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : ({ty [bb,n,d]}, {ty [m,d]}) -> {ty [bb,n,m]}\n" ++
  s!"    %{p}dWfc2 = stablehlo.dot_general %{p}ga, {dy}, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : ({ty [bb,n,m]}, {ty [bb,n,d]}) -> {ty [m,d]}\n" ++
  s!"    %{p}dbfc2 = stablehlo.reduce({dy} init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({ty [bb,n,d]}, tensor<f32>) -> {ty [d]}\n" ++
  -- gelu back (dh1 = da1 ⊙ gelu'(h1))
  geluActBack s!"{p}gb" s!"%{p}h1" s!"%{p}da1" bb n m ++   -- result %{p}gbdx
  -- fc1 back
  s!"    %{p}dx = stablehlo.dot_general %{p}gbdx, {Wfc1}, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : ({ty [bb,n,m]}, {ty [d,m]}) -> {ty [bb,n,d]}\n" ++
  s!"    %{p}dWfc1 = stablehlo.dot_general {x}, %{p}gbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : ({ty [bb,n,d]}, {ty [bb,n,m]}) -> {ty [d,m]}\n" ++
  s!"    %{p}dbfc1 = stablehlo.reduce(%{p}gbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({ty [bb,n,m]}, tensor<f32>) -> {ty [m]}\n"

-- ════════════════════════════════════════════════════════════════
-- § Multi-head self-attention  (validated verbatim in tests/TestMHSA.lean)
-- ════════════════════════════════════════════════════════════════

/-- **MHSA forward**, prefix `p`. Saves `%{p}W` `[b,h,n,n]`, `%{p}Qh`/`%{p}Kh`/
    `%{p}Vh` `[b,h,dh]`, `%{p}P` `[b,n,d]` for the backward. Result `%{p}O`. -/
def mhsaFwd (p x Wq bq Wk bk Wv bv Wo bo : String)
    (b n d h dh : Nat) (scale : String) : String :=
  let proj (nm W bias : String) : String :=
    s!"    %{p}{nm}d = stablehlo.dot_general {x}, {W}, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : ({ty [b,n,d]}, {ty [d,d]}) -> {ty [b,n,d]}\n" ++
    s!"    %{p}{nm}bb = stablehlo.broadcast_in_dim {bias}, dims = [2] : ({ty [d]}) -> {ty [b,n,d]}\n" ++
    s!"    %{p}{nm} = stablehlo.add %{p}{nm}d, %{p}{nm}bb : {ty [b,n,d]}\n"
  let split (src dst : String) : String :=
    s!"    %{p}{dst}r = stablehlo.reshape %{p}{src} : ({ty [b,n,d]}) -> {ty [b,n,h,dh]}\n" ++
    s!"    %{p}{dst} = stablehlo.transpose %{p}{dst}r, dims = [0, 2, 1, 3] : ({ty [b,n,h,dh]}) -> {ty [b,h,n,dh]}\n"
  proj "Q" Wq bq ++ proj "K" Wk bk ++ proj "V" Wv bv ++
  split "Q" "Qh" ++ split "K" "Kh" ++ split "V" "Vh" ++
  s!"    %{p}S = stablehlo.dot_general %{p}Qh, %{p}Kh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : ({ty [b,h,n,dh]}, {ty [b,h,n,dh]}) -> {ty [b,h,n,n]}\n" ++
  s!"    %{p}scl = stablehlo.constant dense<{scale}> : {ty [b,h,n,n]}\n" ++
  s!"    %{p}Ss = stablehlo.multiply %{p}S, %{p}scl : {ty [b,h,n,n]}\n" ++
  s!"    %{p}se = stablehlo.exponential %{p}Ss : {ty [b,h,n,n]}\n" ++
  s!"    %{p}sum = stablehlo.reduce(%{p}se init: %sc) applies stablehlo.add across dimensions = [3] : ({ty [b,h,n,n]}, tensor<f32>) -> {ty [b,h,n]}\n" ++
  s!"    %{p}sumb = stablehlo.broadcast_in_dim %{p}sum, dims = [0, 1, 2] : ({ty [b,h,n]}) -> {ty [b,h,n,n]}\n" ++
  s!"    %{p}W = stablehlo.divide %{p}se, %{p}sumb : {ty [b,h,n,n]}\n" ++
  s!"    %{p}A = stablehlo.dot_general %{p}W, %{p}Vh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : ({ty [b,h,n,n]}, {ty [b,h,n,dh]}) -> {ty [b,h,n,dh]}\n" ++
  s!"    %{p}AT = stablehlo.transpose %{p}A, dims = [0, 2, 1, 3] : ({ty [b,h,n,dh]}) -> {ty [b,n,h,dh]}\n" ++
  s!"    %{p}P = stablehlo.reshape %{p}AT : ({ty [b,n,h,dh]}) -> {ty [b,n,d]}\n" ++
  s!"    %{p}od = stablehlo.dot_general %{p}P, {Wo}, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : ({ty [b,n,d]}, {ty [d,d]}) -> {ty [b,n,d]}\n" ++
  s!"    %{p}obb = stablehlo.broadcast_in_dim {bo}, dims = [2] : ({ty [d]}) -> {ty [b,n,d]}\n" ++
  s!"    %{p}O = stablehlo.add %{p}od, %{p}obb : {ty [b,n,d]}\n"

/-- **MHSA backward**, prefix `p`. Reuses the `mhsaFwd p` saves. Produces
    `%{p}dx` `[b,n,d]` + 8 param grads `%{p}dWQ %{p}dbQ %{p}dWK %{p}dbK %{p}dWV
    %{p}dbV %{p}dWo %{p}dbo`. (Validated in tests/TestMHSA.lean, rel err 2e-3.) -/
def mhsaBack (p x Wq Wk Wv Wo dO : String)
    (b n d h dh : Nat) (scale : String) : String :=
  s!"    %{p}dP = stablehlo.dot_general {dO}, {Wo}, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : ({ty [b,n,d]}, {ty [d,d]}) -> {ty [b,n,d]}\n" ++
  s!"    %{p}dWo = stablehlo.dot_general %{p}P, {dO}, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : ({ty [b,n,d]}, {ty [b,n,d]}) -> {ty [d,d]}\n" ++
  s!"    %{p}dbo = stablehlo.reduce({dO} init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({ty [b,n,d]}, tensor<f32>) -> {ty [d]}\n" ++
  s!"    %{p}dPr = stablehlo.reshape %{p}dP : ({ty [b,n,d]}) -> {ty [b,n,h,dh]}\n" ++
  s!"    %{p}dA = stablehlo.transpose %{p}dPr, dims = [0, 2, 1, 3] : ({ty [b,n,h,dh]}) -> {ty [b,h,n,dh]}\n" ++
  s!"    %{p}dW = stablehlo.dot_general %{p}dA, %{p}Vh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : ({ty [b,h,n,dh]}, {ty [b,h,n,dh]}) -> {ty [b,h,n,n]}\n" ++
  s!"    %{p}dVh = stablehlo.dot_general %{p}W, %{p}dA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : ({ty [b,h,n,n]}, {ty [b,h,n,dh]}) -> {ty [b,h,n,dh]}\n" ++
  s!"    %{p}pdw = stablehlo.multiply %{p}W, %{p}dW : {ty [b,h,n,n]}\n" ++
  s!"    %{p}srow = stablehlo.reduce(%{p}pdw init: %sc) applies stablehlo.add across dimensions = [3] : ({ty [b,h,n,n]}, tensor<f32>) -> {ty [b,h,n]}\n" ++
  s!"    %{p}srowb = stablehlo.broadcast_in_dim %{p}srow, dims = [0, 1, 2] : ({ty [b,h,n]}) -> {ty [b,h,n,n]}\n" ++
  s!"    %{p}diff = stablehlo.subtract %{p}dW, %{p}srowb : {ty [b,h,n,n]}\n" ++
  s!"    %{p}dSs = stablehlo.multiply %{p}W, %{p}diff : {ty [b,h,n,n]}\n" ++
  s!"    %{p}sclb = stablehlo.constant dense<{scale}> : {ty [b,h,n,n]}\n" ++
  s!"    %{p}dS = stablehlo.multiply %{p}dSs, %{p}sclb : {ty [b,h,n,n]}\n" ++
  s!"    %{p}dQh = stablehlo.dot_general %{p}dS, %{p}Kh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : ({ty [b,h,n,n]}, {ty [b,h,n,dh]}) -> {ty [b,h,n,dh]}\n" ++
  s!"    %{p}dKh = stablehlo.dot_general %{p}dS, %{p}Qh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : ({ty [b,h,n,n]}, {ty [b,h,n,dh]}) -> {ty [b,h,n,dh]}\n" ++
  (["Q", "K", "V"].foldl (fun acc nm =>
    acc ++
    s!"    %{p}d{nm}T = stablehlo.transpose %{p}d{nm}h, dims = [0, 2, 1, 3] : ({ty [b,h,n,dh]}) -> {ty [b,n,h,dh]}\n" ++
    s!"    %{p}d{nm} = stablehlo.reshape %{p}d{nm}T : ({ty [b,n,h,dh]}) -> {ty [b,n,d]}\n") "") ++
  (([("Q", Wq), ("K", Wk), ("V", Wv)] : List (String × String)).foldl (fun acc (nm, W) =>
    acc ++
    s!"    %{p}dx{nm} = stablehlo.dot_general %{p}d{nm}, {W}, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : ({ty [b,n,d]}, {ty [d,d]}) -> {ty [b,n,d]}\n" ++
    s!"    %{p}dW{nm} = stablehlo.dot_general {x}, %{p}d{nm}, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : ({ty [b,n,d]}, {ty [b,n,d]}) -> {ty [d,d]}\n" ++
    s!"    %{p}db{nm} = stablehlo.reduce(%{p}d{nm} init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({ty [b,n,d]}, tensor<f32>) -> {ty [d]}\n") "") ++
  s!"    %{p}dxa = stablehlo.add %{p}dxQ, %{p}dxK : {ty [b,n,d]}\n" ++
  s!"    %{p}dx = stablehlo.add %{p}dxa, %{p}dxV : {ty [b,n,d]}\n"

-- ════════════════════════════════════════════════════════════════
-- § Transformer block (pre-norm):  LN1→MHSA→+x → LN2→MLP→+
-- ════════════════════════════════════════════════════════════════

/-- A block's 16 parameter SSA names, in layout order:
    LN1 γ/β `[d]`; MHSA Wq/bq/Wk/bk/Wv/bv/Wo/bo `[d,d]`/`[d]`; LN2 γ/β `[d]`;
    MLP Wfc1/bfc1/Wfc2/bfc2 `[d,m]`/`[m]`/`[m,d]`/`[d]`. -/
structure BlockParams where
  g1 : String
  b1 : String
  Wq : String
  bq : String
  Wk : String
  bk : String
  Wv : String
  bv : String
  Wo : String
  bo : String
  g2 : String
  b2 : String
  Wfc1 : String
  bfc1 : String
  Wfc2 : String
  bfc2 : String

/-- **Transformer block forward**, prefix `p`. `x` `[b,n,d]`. Result `%{p}out`.
    Keeps `%{p}r1` (the first residual) + all sub-fragment saves. The sub-prefixes
    are `{p}1` (LN1), `{p}m` (MHSA), `{p}2` (LN2), `{p}p` (MLP). -/
def blockFwd (p x : String) (bp : BlockParams) (b n d m h dh : Nat)
    (eps scale : String) : String :=
  lnFwd s!"{p}1" x bp.g1 bp.b1 b n d eps ++
  mhsaFwd s!"{p}m" s!"%{p}1y" bp.Wq bp.bq bp.Wk bp.bk bp.Wv bp.bv bp.Wo bp.bo b n d h dh scale ++
  s!"    %{p}r1 = stablehlo.add {x}, %{p}mO : {ty [b,n,d]}\n" ++
  lnFwd s!"{p}2" s!"%{p}r1" bp.g2 bp.b2 b n d eps ++
  mlpFwd s!"{p}p" s!"%{p}2y" bp.Wfc1 bp.bfc1 bp.Wfc2 bp.bfc2 b n d m ++
  s!"    %{p}out = stablehlo.add %{p}r1, %{p}py : {ty [b,n,d]}\n"

/-- **Transformer block backward**, prefix `p`. Reuses `blockFwd p` saves. `dOut`
    `[b,n,d]` is the block-output cotangent. Produces `%{p}dx` `[b,n,d]` and the 16
    param grads (see `blockGradNames`). -/
def blockBack (p dOut : String) (bp : BlockParams) (b n d m h dh : Nat)
    (scale : String) : String :=
  -- MLP + LN2 path (dOut → mlp → ln2)
  mlpBack s!"{p}p" s!"%{p}2y" bp.Wfc1 bp.Wfc2 dOut b n d m ++
  lnBack s!"{p}2" bp.g2 s!"%{p}pdx" b n d ++
  s!"    %{p}dr1 = stablehlo.add {dOut}, %{p}2dx : {ty [b,n,d]}\n" ++
  -- MHSA + LN1 path (dr1 → mhsa → ln1)
  mhsaBack s!"{p}m" s!"%{p}1y" bp.Wq bp.Wk bp.Wv bp.Wo s!"%{p}dr1" b n d h dh scale ++
  lnBack s!"{p}1" bp.g1 s!"%{p}mdx" b n d ++
  s!"    %{p}dx = stablehlo.add %{p}dr1, %{p}1dx : {ty [b,n,d]}\n"

/-- The 16 block param-grad SSA names produced by `blockBack p`, in the same
    (layout) order as `BlockParams`: g1,b1, Wq,bq,Wk,bk,Wv,bv,Wo,bo, g2,b2,
    Wfc1,bfc1,Wfc2,bfc2. -/
def blockGradNames (p : String) : List String :=
  [s!"%{p}1dg", s!"%{p}1db",
   s!"%{p}mdWQ", s!"%{p}mdbQ", s!"%{p}mdWK", s!"%{p}mdbK",
   s!"%{p}mdWV", s!"%{p}mdbV", s!"%{p}mdWo", s!"%{p}mdbo",
   s!"%{p}2dg", s!"%{p}2db",
   s!"%{p}pdWfc1", s!"%{p}pdbfc1", s!"%{p}pdWfc2", s!"%{p}pdbfc2"]

-- ════════════════════════════════════════════════════════════════
-- § Patch embed (k=s strided conv) + flatten + CLS token + positional embed
-- ════════════════════════════════════════════════════════════════

/-- **Patch-embed conv forward**, prefix `p`: non-overlapping `s×s`/stride-`s` conv
    `[b,ic,s·ph,s·pw]·[d,ic,s,s] + bias → [b,d,ph,pw]`, then flatten to tokens
    `[b, ph·pw, d]` (transpose `[0,2,3,1]` + reshape). Result `%{p}tok`. -/
def patchEmbedFwd (p x w bias : String) (b ic d ph pw s : Nat) : String :=
  s!"    %{p}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [{s}, {s}], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [b,ic,s*ph,s*pw]}, {ty [d,ic,s,s]}) -> {ty [b,d,ph,pw]}\n" ++
  s!"    %{p}cbb = stablehlo.broadcast_in_dim {bias}, dims = [1] : ({ty [d]}) -> {ty [b,d,ph,pw]}\n" ++
  s!"    %{p}pe = stablehlo.add %{p}c, %{p}cbb : {ty [b,d,ph,pw]}\n" ++
  s!"    %{p}pt = stablehlo.transpose %{p}pe, dims = [0, 2, 3, 1] : ({ty [b,d,ph,pw]}) -> {ty [b,ph,pw,d]}\n" ++
  s!"    %{p}tok = stablehlo.reshape %{p}pt : ({ty [b,ph,pw,d]}) -> {ty [b,ph*pw,d]}\n"

/-- **Patch-embed backward** (weight + bias grads; NO input grad — first layer),
    prefix `p`. `dtok` `[b,ph·pw,d]` is the token-grad. Reshapes/transposes back to
    `[b,d,ph,pw]`, then: bias grad `reduce[0,2,3]→[d]`; weight grad = dilate `dy`
    interior `s-1` (no high → `s·ph-(s-1)`), valid conv `→ [d,ic,s,s]`. Produces
    `%{p}dw` `[d,ic,s,s]`, `%{p}db` `[d]`. -/
def patchEmbedBack (p x dtok : String) (b ic d ph pw s : Nat) : String :=
  let dilH := s*ph - (s-1)
  let dilW := s*pw - (s-1)
  -- token-grad → [b,d,ph,pw]
  s!"    %{p}dtr = stablehlo.reshape {dtok} : ({ty [b,ph*pw,d]}) -> {ty [b,ph,pw,d]}\n" ++
  s!"    %{p}dy = stablehlo.transpose %{p}dtr, dims = [0, 3, 1, 2] : ({ty [b,ph,pw,d]}) -> {ty [b,d,ph,pw]}\n" ++
  -- bias grad
  s!"    %{p}db = stablehlo.reduce(%{p}dy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [b,d,ph,pw]}, tensor<f32>) -> {ty [d]}\n" ++
  -- weight grad (dilate-no-high + valid conv, ch9 patchifyWGrad with stride s)
  s!"    %{p}u = stablehlo.pad %{p}dy, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, {s-1}, {s-1}] : ({ty [b,d,ph,pw]}, tensor<f32>) -> {ty [b,d,dilH,dilW]}\n" ++
  s!"    %{p}xt = stablehlo.transpose {x}, dims = [1, 0, 2, 3] : ({ty [b,ic,s*ph,s*pw]}) -> {ty [ic,b,s*ph,s*pw]}\n" ++
  s!"    %{p}dt = stablehlo.transpose %{p}u, dims = [1, 0, 2, 3] : ({ty [b,d,dilH,dilW]}) -> {ty [d,b,dilH,dilW]}\n" ++
  s!"    %{p}raw = stablehlo.convolution(%{p}xt, %{p}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ic,b,s*ph,s*pw]}, {ty [d,b,dilH,dilW]}) -> {ty [ic,d,s,s]}\n" ++
  s!"    %{p}dw = stablehlo.transpose %{p}raw, dims = [1, 0, 2, 3] : ({ty [ic,d,s,s]}) -> {ty [d,ic,s,s]}\n"

/-- **CLS-token + positional-embed forward**, prefix `p`: prepend a learned `[d]`
    CLS at row 0 of the `n0` patch tokens → `[b,n0+1,d]`, then add a learned `[n0+1,d]`
    positional embedding (broadcast over batch). Result `%{p}z` `[b,n0+1,d]`. -/
def clsPosFwd (p tok cls pos : String) (b n0 d : Nat) : String :=
  s!"    %{p}clsb = stablehlo.broadcast_in_dim {cls}, dims = [2] : ({ty [d]}) -> {ty [b,1,d]}\n" ++
  s!"    %{p}cat = stablehlo.concatenate %{p}clsb, {tok}, dim = 1 : ({ty [b,1,d]}, {ty [b,n0,d]}) -> {ty [b,n0+1,d]}\n" ++
  s!"    %{p}posb = stablehlo.broadcast_in_dim {pos}, dims = [1, 2] : ({ty [n0+1,d]}) -> {ty [b,n0+1,d]}\n" ++
  s!"    %{p}z = stablehlo.add %{p}cat, %{p}posb : {ty [b,n0+1,d]}\n"

/-- **CLS + pos backward**, prefix `p`. `dz` `[b,n0+1,d]`. Produces `%{p}dtok`
    `[b,n0,d]` (patch-token grad), `%{p}dcls` `[d]` (Σ over batch of row 0),
    `%{p}dpos` `[n0+1,d]` (Σ over batch). -/
def clsPosBack (p dz : String) (b n0 d : Nat) : String :=
  -- pos: sum over batch
  s!"    %{p}dpos = stablehlo.reduce({dz} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [b,n0+1,d]}, tensor<f32>) -> {ty [n0+1,d]}\n" ++
  -- CLS: row 0 slice, sum over batch
  s!"    %{p}cslc = stablehlo.slice {dz} [0:{b}, 0:1, 0:{d}] : ({ty [b,n0+1,d]}) -> {ty [b,1,d]}\n" ++
  s!"    %{p}cr = stablehlo.reshape %{p}cslc : ({ty [b,1,d]}) -> {ty [b,d]}\n" ++
  s!"    %{p}dcls = stablehlo.reduce(%{p}cr init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [b,d]}, tensor<f32>) -> {ty [d]}\n" ++
  -- patch tokens: rows 1..n0
  s!"    %{p}dtok = stablehlo.slice {dz} [0:{b}, 1:{n0+1}, 0:{d}] : ({ty [b,n0+1,d]}) -> {ty [b,n0,d]}\n"

/-- **Classifier head forward**, prefix `p`: take the CLS token (row 0 of `[b,n,d]`)
    → `[b,d]`, dense `[d,nc] + bias → [b,nc]` logits. Result `%{p}logits`. -/
def headFwd (p z Wc bc : String) (b n d nc : Nat) : String :=
  s!"    %{p}cls = stablehlo.slice {z} [0:{b}, 0:1, 0:{d}] : ({ty [b,n,d]}) -> {ty [b,1,d]}\n" ++
  s!"    %{p}clsv = stablehlo.reshape %{p}cls : ({ty [b,1,d]}) -> {ty [b,d]}\n" ++
  s!"    %{p}hd = stablehlo.dot_general %{p}clsv, {Wc}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [b,d]}, {ty [d,nc]}) -> {ty [b,nc]}\n" ++
  s!"    %{p}hbb = stablehlo.broadcast_in_dim {bc}, dims = [1] : ({ty [nc]}) -> {ty [b,nc]}\n" ++
  s!"    %{p}logits = stablehlo.add %{p}hd, %{p}hbb : {ty [b,nc]}\n"

/-- **Classifier head backward**, prefix `p`. Reuses `%{p}clsv` from `headFwd`.
    `dlog` `[b,nc]`. Produces `%{p}dz` `[b,n,d]` (cotangent scattered into row 0,
    zero elsewhere), `%{p}dWc` `[d,nc]`, `%{p}dbc` `[nc]`. -/
def headBack (p Wc dlog : String) (b n d nc : Nat) : String :=
  s!"    %{p}dWc = stablehlo.dot_general %{p}clsv, {dlog}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [b,d]}, {ty [b,nc]}) -> {ty [d,nc]}\n" ++
  s!"    %{p}dbc = stablehlo.reduce({dlog} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [b,nc]}, tensor<f32>) -> {ty [nc]}\n" ++
  s!"    %{p}dclsv = stablehlo.dot_general {dlog}, {Wc}, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [b,nc]}, {ty [d,nc]}) -> {ty [b,d]}\n" ++
  s!"    %{p}dclsr = stablehlo.reshape %{p}dclsv : ({ty [b,d]}) -> {ty [b,1,d]}\n" ++
  -- scatter into row 0: pad the [b,1,d] grad with zeros to [b,n,d] (high pad on dim 1)
  s!"    %{p}dz = stablehlo.pad %{p}dclsr, %sc, low = [0, 0, 0], high = [0, {n-1}, 0], interior = [0, 0, 0] : ({ty [b,1,d]}, tensor<f32>) -> {ty [b,n,d]}\n"

-- ════════════════════════════════════════════════════════════════
-- § Whole ViT:  patch-embed → CLS+pos → k blocks → final LN → head
-- ════════════════════════════════════════════════════════════════

/-- ViT hyperparameters (one config). `n0 = ph·pw` patches, `n = n0+1` tokens. -/
structure ViTConfig where
  b : Nat        -- batch
  ic : Nat       -- input channels (3)
  d : Nat        -- model dim
  ph : Nat       -- patch grid height (= H/s)
  pw : Nat       -- patch grid width
  s : Nat        -- patch size (= stride)
  m : Nat        -- MLP hidden
  h : Nat        -- heads
  dh : Nat       -- head dim (= d/h)
  nc : Nat       -- classes
  eps : String
  scale : String

/-- **Whole-ViT forward**, prefix `p`: `x` `[b,ic,s·ph,s·pw]` → logits `[b,nc]`.
    Result `%{p}hdlogits`. Keeps every sub-fragment save for the backward. -/
def vitFwd (p x wConv bConv cls pos gF bF Wc bc : String)
    (blocks : List BlockParams) (cfg : ViTConfig) : String :=
  let n0 := cfg.ph * cfg.pw
  let n := n0 + 1
  -- block prefix `{p}b{i}_` (trailing `_` separates the index from sub-prefixes so
  -- block 1's LN1 `…b1_1` ≠ block 11 `…b11_`).
  let (blkCode, lastZ) := (blocks.zipIdx).foldl (fun (st : String × String) (bi : BlockParams × Nat) =>
      let (acc, prev) := st
      let (bpi, i) := bi
      (acc ++ blockFwd s!"{p}b{i}_" prev bpi cfg.b n cfg.d cfg.m cfg.h cfg.dh cfg.eps cfg.scale,
       s!"%{p}b{i}_out")) ("", s!"%{p}cpz")
  patchEmbedFwd s!"{p}pe" x wConv bConv cfg.b cfg.ic cfg.d cfg.ph cfg.pw cfg.s ++
  clsPosFwd s!"{p}cp" s!"%{p}petok" cls pos cfg.b n0 cfg.d ++
  blkCode ++
  lnFwd s!"{p}fln" lastZ gF bF cfg.b n cfg.d cfg.eps ++
  headFwd s!"{p}hd" s!"%{p}flny" Wc bc cfg.b n cfg.d cfg.nc

/-- **Whole-ViT backward**, prefix `p`. Reuses `vitFwd p` saves. `dlog` `[b,nc]` is
    the logits cotangent. `x` is the (fixed) input image (for the patch weight-grad).
    Produces all param grads (see `vitGradNames`); NO image grad (first layer). -/
def vitBack (p dlog x wConv Wc gF : String)
    (blocks : List BlockParams) (cfg : ViTConfig) : String :=
  let n0 := cfg.ph * cfg.pw
  let n := n0 + 1
  let (blkCode, firstDz) := (blocks.zipIdx.reverse).foldl (fun (st : String × String) (bi : BlockParams × Nat) =>
      let (acc, dnext) := st
      let (bpi, i) := bi
      (acc ++ blockBack s!"{p}b{i}_" dnext bpi cfg.b n cfg.d cfg.m cfg.h cfg.dh cfg.scale,
       s!"%{p}b{i}_dx")) ("", s!"%{p}flndx")
  headBack s!"{p}hd" Wc dlog cfg.b n cfg.d cfg.nc ++
  lnBack s!"{p}fln" gF s!"%{p}hddz" cfg.b n cfg.d ++
  blkCode ++
  clsPosBack s!"{p}cp" firstDz cfg.b n0 cfg.d ++
  patchEmbedBack s!"{p}pe" x s!"%{p}cpdtok" cfg.b cfg.ic cfg.d cfg.ph cfg.pw cfg.s

/-- Param SSA names in canonical (layout) order: wConv, bConv, cls, pos, then each
    block's 16 (via `BlockParams` fields), then gF, bF, Wc, bc. -/
def vitParamNames (blocks : List BlockParams) : List String :=
  ["%wConv", "%bConv", "%cls", "%pos"] ++
  (blocks.flatMap (fun bp =>
    [bp.g1, bp.b1, bp.Wq, bp.bq, bp.Wk, bp.bk, bp.Wv, bp.bv, bp.Wo, bp.bo,
     bp.g2, bp.b2, bp.Wfc1, bp.bfc1, bp.Wfc2, bp.bfc2])) ++
  ["%gF", "%bF", "%Wc", "%bc"]

/-- Param grad SSA names produced by `vitBack p`, SAME order as `vitParamNames`. -/
def vitGradNames (p : String) (blocks : List BlockParams) : List String :=
  [s!"%{p}pedw", s!"%{p}pedb", s!"%{p}cpdcls", s!"%{p}cpdpos"] ++
  ((List.range blocks.length).flatMap (fun i => blockGradNames s!"{p}b{i}_")) ++
  [s!"%{p}flndg", s!"%{p}flndb", s!"%{p}hddWc", s!"%{p}hddbc"]

/-- Param dims in canonical order (matches `vitParamNames`). -/
def vitParamDims (blocks : List BlockParams) (cfg : ViTConfig) : List (List Nat) :=
  let n := cfg.ph * cfg.pw + 1
  [[cfg.d, cfg.ic, cfg.s, cfg.s], [cfg.d], [cfg.d], [n, cfg.d]] ++   -- CLS [d] (1D — matches the proof render + ViTLayout)
  (blocks.flatMap (fun _ =>
    [[cfg.d], [cfg.d],
     [cfg.d, cfg.d], [cfg.d], [cfg.d, cfg.d], [cfg.d], [cfg.d, cfg.d], [cfg.d], [cfg.d, cfg.d], [cfg.d],
     [cfg.d], [cfg.d],
     [cfg.d, cfg.m], [cfg.m], [cfg.m, cfg.d], [cfg.d]])) ++
  [[cfg.d], [cfg.d], [cfg.d, cfg.nc], [cfg.nc]]

-- ════════════════════════════════════════════════════════════════
-- § Whole-net module builders (the production train-step + fwd renderers)
-- ════════════════════════════════════════════════════════════════

/-- The param func signature `%nm: tensor<…>` (canonical order). -/
def vitParamSig (blocks : List BlockParams) (cfg : ViTConfig) : String :=
  String.intercalate ", "
    (((vitParamNames blocks).zip (vitParamDims blocks cfg)).map (fun (nm, ds) => s!"{nm}: {ty ds}"))

/-- `@vit_fwd(%x flat, params…) → logits [b,nc]` — image→logits (for eval). The flat
    `%x` `[b, ic·H·W]` is reshaped to `[b,ic,H,W]` then run through `vitFwd`. -/
def vitFwdModule (cfg : ViTConfig) (blocks : List BlockParams) : String :=
  let h := cfg.s * cfg.ph; let w := cfg.s * cfg.pw; let d0 := cfg.ic * h * w
  "module @m {\n" ++
  s!"  func.func @vit_fwd(%x: {ty [cfg.b, d0]}, {vitParamSig blocks cfg}) -> {ty [cfg.b, cfg.nc]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %xr = stablehlo.reshape %x : ({ty [cfg.b, d0]}) -> {ty [cfg.b, cfg.ic, h, w]}\n" ++
  vitFwd "vit" "%xr" "%wConv" "%bConv" "%cls" "%pos" "%gF" "%bF" "%Wc" "%bc" blocks cfg ++
  s!"    return %vithdlogits : {ty [cfg.b, cfg.nc]}\n" ++ "  }\n}\n"

/-- `@vit_train_step(%x flat, params…, %onehot) → updated params` — one mean-loss-SGD
    step. Forward → softmax-CE cotangent `dy=(softmax(logits)−onehot)/b` → `vitBack` →
    per-param SGD `θ ← θ − lr·dθ` (baked in), returns the updated param list. -/
def vitTrainStepModule (cfg : ViTConfig) (blocks : List BlockParams) (lr : String) : String :=
  let h := cfg.s * cfg.ph; let w := cfg.s * cfg.pw; let d0 := cfg.ic * h * w
  let pnames := vitParamNames blocks
  let pdims := vitParamDims blocks cfg
  let grads := vitGradNames "vit" blocks
  let cot :=
    s!"    %le = stablehlo.exponential %vithdlogits : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [cfg.b, cfg.nc]}, tensor<f32>) -> {ty [cfg.b]}\n" ++
    s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [cfg.b]}) -> {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %lsm = stablehlo.divide %le, %lsb : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %dyr = stablehlo.subtract %lsm, %onehot : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %bnc = stablehlo.constant dense<{cfg.b}.0> : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %dy = stablehlo.divide %dyr, %bnc : {ty [cfg.b, cfg.nc]}\n"
  let upd := String.join (((pnames.zip grads).zip pdims).map (fun ((nm, gr), ds) =>
    s!"    {nm}_lr = stablehlo.constant dense<{lr}> : {ty ds}\n" ++
    s!"    {nm}_st = stablehlo.multiply {gr}, {nm}_lr : {ty ds}\n" ++
    s!"    {nm}n = stablehlo.subtract {nm}, {nm}_st : {ty ds}\n"))
  let retTy := String.intercalate ", " (pdims.map (fun ds => ty ds))
  let retVals := String.intercalate ", " (pnames.map (fun nm => s!"{nm}n"))
  "module @m {\n" ++
  s!"  func.func @vit_train_step(%x: {ty [cfg.b, d0]}, {vitParamSig blocks cfg}, %onehot: {ty [cfg.b, cfg.nc]}) -> ({retTy}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %xr = stablehlo.reshape %x : ({ty [cfg.b, d0]}) -> {ty [cfg.b, cfg.ic, h, w]}\n" ++
  vitFwd "vit" "%xr" "%wConv" "%bConv" "%cls" "%pos" "%gF" "%bF" "%Wc" "%bc" blocks cfg ++
  cot ++
  vitBack "vit" "%dy" "%xr" "%wConv" "%Wc" "%gF" blocks cfg ++
  upd ++
  s!"    return {retVals} : {retTy}\n" ++ "  }\n}\n"

/-- Production ViT-Tiny config @ Imagenette 224² (matches `ViTLayout`): depth-`k`
    blocks with distinct per-block param names `%<field>_<i>`. -/
def vitTinyBlocks (depth : Nat) : List BlockParams :=
  (List.range depth).map (fun i =>
    { g1 := s!"%g1_{i}", b1 := s!"%b1_{i}",
      Wq := s!"%Wq_{i}", bq := s!"%bq_{i}", Wk := s!"%Wk_{i}", bk := s!"%bk_{i}",
      Wv := s!"%Wv_{i}", bv := s!"%bv_{i}", Wo := s!"%Wo_{i}", bo := s!"%bo_{i}",
      g2 := s!"%g2_{i}", b2 := s!"%b2_{i}",
      Wfc1 := s!"%Wfc1_{i}", bfc1 := s!"%bfc1_{i}", Wfc2 := s!"%Wfc2_{i}", bfc2 := s!"%bfc2_{i}" })

def vitTinyConfig (b depth : Nat) : ViTConfig :=
  { b := b, ic := 3, d := 192, ph := 14, pw := 14, s := 16, m := 768, h := 3, dh := 64,
    nc := 10, eps := "1.0e-5", scale := "0.125" }   -- 1/√64 = 0.125

-- ════════════════════════════════════════════════════════════════
-- § AdamW optimizer render (Phase 3b of vit_train_to_vit_verified.md)
-- The proven-fragment-side analogue of `MlirCodegen.emitAdamUpdate`; its ℝ
-- spec is `Proofs.adamWParam` (LeanMlir/Proofs/AdamStep.lean). Scalar
-- hyperparameters arrive as `tensor<f32>` function args.
-- ════════════════════════════════════════════════════════════════

/-- **AdamW update for one parameter.** Emits the `m'/v'/θ'` block at shape `ds`
    (tag `t` keeps SSA names distinct), reading scalar args `%b1 %ob1 %b2 %ob2
    %bc1 %bc2 %lr %eps %wd`. Returns `(ir, θ'SSA, m'SSA, v'SSA)`. Op-for-op the
    coordinate formula `Proofs.adamWParam`:
    `θ' = θ − lr·((β₁m+(1−β₁)g)/bc₁)/(√((β₂v+(1−β₂)g²)/bc₂)+ε) − (wd·lr)·θ`. -/
def emitAdamV (θ g m v : String) (ds : List Nat) (t : String) : String × String × String × String :=
  let T := ty ds
  let s :=
    s!"    %adb1{t} = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> {T}\n" ++
    s!"    %adob1{t} = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> {T}\n" ++
    s!"    %adms{t} = stablehlo.multiply %adb1{t}, {m} : {T}\n" ++
    s!"    %admg{t} = stablehlo.multiply %adob1{t}, {g} : {T}\n" ++
    s!"    %admn{t} = stablehlo.add %adms{t}, %admg{t} : {T}\n" ++
    s!"    %adb2{t} = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> {T}\n" ++
    s!"    %adob2{t} = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> {T}\n" ++
    s!"    %advs{t} = stablehlo.multiply %adb2{t}, {v} : {T}\n" ++
    s!"    %adg2{t} = stablehlo.multiply {g}, {g} : {T}\n" ++
    s!"    %advg{t} = stablehlo.multiply %adob2{t}, %adg2{t} : {T}\n" ++
    s!"    %advn{t} = stablehlo.add %advs{t}, %advg{t} : {T}\n" ++
    s!"    %adbc1{t} = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> {T}\n" ++
    s!"    %adbc2{t} = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> {T}\n" ++
    s!"    %admh{t} = stablehlo.divide %admn{t}, %adbc1{t} : {T}\n" ++
    s!"    %advh{t} = stablehlo.divide %advn{t}, %adbc2{t} : {T}\n" ++
    s!"    %adlr{t} = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> {T}\n" ++
    s!"    %adeps{t} = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {T}\n" ++
    s!"    %adsq{t} = stablehlo.sqrt %advh{t} : {T}\n" ++
    s!"    %adden{t} = stablehlo.add %adsq{t}, %adeps{t} : {T}\n" ++
    s!"    %adrat{t} = stablehlo.divide %admh{t}, %adden{t} : {T}\n" ++
    s!"    %adst{t} = stablehlo.multiply %adlr{t}, %adrat{t} : {T}\n" ++
    s!"    %adsub{t} = stablehlo.subtract {θ}, %adst{t} : {T}\n" ++
    s!"    %adwd{t} = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> {T}\n" ++
    s!"    %adwdlr{t} = stablehlo.multiply %adwd{t}, %adlr{t} : {T}\n" ++
    s!"    %adwdp{t} = stablehlo.multiply %adwdlr{t}, {θ} : {T}\n" ++
    s!"    %adnew{t} = stablehlo.subtract %adsub{t}, %adwdp{t} : {T}\n"
  (s, s!"%adnew{t}", s!"%admn{t}", s!"%advn{t}")

/-- `@vit_train_step_adam` — the SGD train step's optimizer swapped for AdamW.
    Same forward/backward/softmax-CE cotangent as `vitTrainStepModule`; the per-
    param SGD `θ−lr·dθ` is replaced by `emitAdamV` (so the func also takes the
    per-param moments `%<nm>m`/`%<nm>v` and the scalar Adam hyperparameters).
    Returns the updated parameters (moment outputs elided for the smoke; a full
    step would also return `%admn`/`%advn`). -/
def vitTrainStepModuleAdam (cfg : ViTConfig) (blocks : List BlockParams) : String :=
  let h := cfg.s * cfg.ph; let w := cfg.s * cfg.pw; let d0 := cfg.ic * h * w
  let pnames := vitParamNames blocks
  let pdims := vitParamDims blocks cfg
  let grads := vitGradNames "vit" blocks
  let mnames := pnames.map (· ++ "m")
  let vnames := pnames.map (· ++ "v")
  let psig := vitParamSig blocks cfg
  let msig := String.intercalate ", " ((mnames.zip pdims).map (fun (nm, ds) => s!"{nm}: {ty ds}"))
  let vsig := String.intercalate ", " ((vnames.zip pdims).map (fun (nm, ds) => s!"{nm}: {ty ds}"))
  let scalarSig := "%b1: tensor<f32>, %ob1: tensor<f32>, %b2: tensor<f32>, %ob2: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %lr: tensor<f32>, %eps: tensor<f32>, %wd: tensor<f32>"
  let cot :=
    s!"    %le = stablehlo.exponential %vithdlogits : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [cfg.b, cfg.nc]}, tensor<f32>) -> {ty [cfg.b]}\n" ++
    s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [cfg.b]}) -> {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %lsm = stablehlo.divide %le, %lsb : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %dyr = stablehlo.subtract %lsm, %onehot : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %bnc = stablehlo.constant dense<{cfg.b}.0> : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %dy = stablehlo.divide %dyr, %bnc : {ty [cfg.b, cfg.nc]}\n"
  let updParts := (((pnames.zip grads).zip pdims).zip (mnames.zip vnames)).map
    (fun (((nm, gr), ds), (mm, vv)) => emitAdamV nm gr mm vv ds (String.ofList (nm.toList.drop 1)))
  let upd := String.join (updParts.map (·.1))
  let retTy := String.intercalate ", " (pdims.map (fun ds => ty ds))
  let retVals := String.intercalate ", " (updParts.map (·.2.1))
  "module @m {\n" ++
  s!"  func.func @vit_train_step_adam(%x: {ty [cfg.b, d0]}, {psig}, {msig}, {vsig}, {scalarSig}, %onehot: {ty [cfg.b, cfg.nc]}) -> ({retTy}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %xr = stablehlo.reshape %x : ({ty [cfg.b, d0]}) -> {ty [cfg.b, cfg.ic, h, w]}\n" ++
  vitFwd "vit" "%xr" "%wConv" "%bConv" "%cls" "%pos" "%gF" "%bF" "%Wc" "%bc" blocks cfg ++
  cot ++
  vitBack "vit" "%dy" "%xr" "%wConv" "%Wc" "%gF" blocks cfg ++
  upd ++
  s!"    return {retVals} : {retTy}\n" ++ "  }\n}\n"

/-- **Packed AdamW train step** for the FFI driver. Hyperparameters are baked as
    constants (so the func takes NO scalar args), and the parameters + both moment
    buffers thread as a single `[θ|m|v]` blob: arg order `(x, θ×k, m×k, v×k, onehot)`
    and return `(θ'×k, m'×k, v'×k)`. This matches `iree_ffi_train_step_generic`'s
    `(x, params, y) → params'` contract with `n_params = 3k` (the moments ride in
    the params blob — no `.so` change). Bias correction is omitted (`bc₁=bc₂=1`); a
    later rung host-passes the per-step `1−βᵗ`. Optimizer = `Proofs.adamWParam`. -/
def vitTrainStepModuleAdamPacked (cfg : ViTConfig) (blocks : List BlockParams)
    (lr β1 ob1 β2 ob2 eps wd : String) : String :=
  let h := cfg.s * cfg.ph; let w := cfg.s * cfg.pw; let d0 := cfg.ic * h * w
  let pnames := vitParamNames blocks
  let pdims := vitParamDims blocks cfg
  let grads := vitGradNames "vit" blocks
  let mnames := pnames.map (· ++ "m")
  let vnames := pnames.map (· ++ "v")
  let psig := vitParamSig blocks cfg
  let msig := String.intercalate ", " ((mnames.zip pdims).map (fun (nm, ds) => s!"{nm}: {ty ds}"))
  let vsig := String.intercalate ", " ((vnames.zip pdims).map (fun (nm, ds) => s!"{nm}: {ty ds}"))
  let consts :=
    s!"    %b1 = stablehlo.constant dense<{β1}> : tensor<f32>\n" ++
    s!"    %ob1 = stablehlo.constant dense<{ob1}> : tensor<f32>\n" ++
    s!"    %b2 = stablehlo.constant dense<{β2}> : tensor<f32>\n" ++
    s!"    %ob2 = stablehlo.constant dense<{ob2}> : tensor<f32>\n" ++
    s!"    %bc1 = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %bc2 = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %lr = stablehlo.constant dense<{lr}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %wd = stablehlo.constant dense<{wd}> : tensor<f32>\n"
  let cot :=
    s!"    %le = stablehlo.exponential %vithdlogits : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [cfg.b, cfg.nc]}, tensor<f32>) -> {ty [cfg.b]}\n" ++
    s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [cfg.b]}) -> {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %lsm = stablehlo.divide %le, %lsb : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %dyr = stablehlo.subtract %lsm, %onehot : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %bnc = stablehlo.constant dense<{cfg.b}.0> : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %dy = stablehlo.divide %dyr, %bnc : {ty [cfg.b, cfg.nc]}\n"
  let updParts := (((pnames.zip grads).zip pdims).zip (mnames.zip vnames)).map
    (fun (((nm, gr), ds), (mm, vv)) => emitAdamV nm gr mm vv ds (String.ofList (nm.toList.drop 1)))
  let upd := String.join (updParts.map (·.1))
  let allNames := (updParts.map (·.2.1)) ++ (updParts.map (·.2.2.1)) ++ (updParts.map (·.2.2.2))
  let allDims := pdims ++ pdims ++ pdims
  let retTy := String.intercalate ", " (allDims.map (fun ds => ty ds))
  let retVals := String.intercalate ", " allNames
  "module @m {\n" ++
  s!"  func.func @vit_adam_train_step(%x: {ty [cfg.b, d0]}, {psig}, {msig}, {vsig}, %onehot: {ty [cfg.b, cfg.nc]}) -> ({retTy}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  consts ++
  s!"    %xr = stablehlo.reshape %x : ({ty [cfg.b, d0]}) -> {ty [cfg.b, cfg.ic, h, w]}\n" ++
  vitFwd "vit" "%xr" "%wConv" "%bConv" "%cls" "%pos" "%gF" "%bF" "%Wc" "%bc" blocks cfg ++
  cot ++
  vitBack "vit" "%dy" "%xr" "%wConv" "%Wc" "%gF" blocks cfg ++
  upd ++
  s!"    return {retVals} : {retTy}\n" ++ "  }\n}\n"

/-- **Scheduled AdamW train step** (Phase 2): like `…AdamPacked`, but `lr`/`bc₁`/`bc₂`
    arrive as runtime rank-0 scalar *params* (smuggled in the packed blob's tail —
    the FFI takes no scalar slot) so the host can drive cosine+warmup and the
    per-step bias correction `1−βᵗ`. They are returned UNCHANGED (passthrough) so
    `#outputs = #inputs = 3k+3`, preserving the generic FFI's invariant. Arg order
    `(x, θ×k, m×k, v×k, lr, bc₁, bc₂, onehot)`; only `β₁,β₂,ε,wd` stay baked. -/
def vitTrainStepModuleAdamSched (cfg : ViTConfig) (blocks : List BlockParams)
    (β1 ob1 β2 ob2 eps wd : String) (ls : Float) : String :=
  let h := cfg.s * cfg.ph; let w := cfg.s * cfg.pw; let d0 := cfg.ic * h * w
  let lsK := ls / cfg.nc.toFloat   -- α/K, the off-class smoothing mass
  let pnames := vitParamNames blocks
  let pdims := vitParamDims blocks cfg
  let grads := vitGradNames "vit" blocks
  let mnames := pnames.map (· ++ "m")
  let vnames := pnames.map (· ++ "v")
  let psig := vitParamSig blocks cfg
  let msig := String.intercalate ", " ((mnames.zip pdims).map (fun (nm, ds) => s!"{nm}: {ty ds}"))
  let vsig := String.intercalate ", " ((vnames.zip pdims).map (fun (nm, ds) => s!"{nm}: {ty ds}"))
  let consts :=
    s!"    %b1 = stablehlo.constant dense<{β1}> : tensor<f32>\n" ++
    s!"    %ob1 = stablehlo.constant dense<{ob1}> : tensor<f32>\n" ++
    s!"    %b2 = stablehlo.constant dense<{β2}> : tensor<f32>\n" ++
    s!"    %ob2 = stablehlo.constant dense<{ob2}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %wd = stablehlo.constant dense<{wd}> : tensor<f32>\n"
  -- Label smoothing (α = ls): the soft-target CE gradient softmax − ((1−α)·onehot +
  -- α/K) = (softmax − onehot) + α·onehot − α/K, applied in-graph (the FFI still
  -- passes a hard onehot). α = 0 recovers the hard-label cotangent.
  let cot :=
    s!"    %le = stablehlo.exponential %vithdlogits : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [cfg.b, cfg.nc]}, tensor<f32>) -> {ty [cfg.b]}\n" ++
    s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [cfg.b]}) -> {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %lsm = stablehlo.divide %le, %lsb : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %dyr0 = stablehlo.subtract %lsm, %onehot : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %lsa = stablehlo.constant dense<{ls}> : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %lsaoh = stablehlo.multiply %lsa, %onehot : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %dyr1 = stablehlo.add %dyr0, %lsaoh : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %lsaik = stablehlo.constant dense<{lsK}> : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %dyr = stablehlo.subtract %dyr1, %lsaik : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %bnc = stablehlo.constant dense<{cfg.b}.0> : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %dy = stablehlo.divide %dyr, %bnc : {ty [cfg.b, cfg.nc]}\n" ++
    -- Smoothed-CE loss for logging: -mean_b( (1−α)·log p[true] + (α/K)·Σ_k log p_k ),
    -- emitted to the %loss output slot (replaces the lr passthrough — lr isn't read back).
    s!"    %llog = stablehlo.log %lsm : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %ohll = stablehlo.multiply %onehot, %llog : {ty [cfg.b, cfg.nc]}\n" ++
    s!"    %t1s = stablehlo.reduce(%ohll init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [cfg.b, cfg.nc]}, tensor<f32>) -> {ty [cfg.b]}\n" ++
    s!"    %lls = stablehlo.reduce(%llog init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [cfg.b, cfg.nc]}, tensor<f32>) -> {ty [cfg.b]}\n" ++
    s!"    %omac = stablehlo.constant dense<{1.0 - ls}> : {ty [cfg.b]}\n" ++
    s!"    %aKc = stablehlo.constant dense<{lsK}> : {ty [cfg.b]}\n" ++
    s!"    %lt1 = stablehlo.multiply %omac, %t1s : {ty [cfg.b]}\n" ++
    s!"    %lt2 = stablehlo.multiply %aKc, %lls : {ty [cfg.b]}\n" ++
    s!"    %lpe = stablehlo.add %lt1, %lt2 : {ty [cfg.b]}\n" ++
    s!"    %lsum2 = stablehlo.reduce(%lpe init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [cfg.b]}, tensor<f32>) -> tensor<f32>\n" ++
    s!"    %lbfc = stablehlo.constant dense<{cfg.b}.0> : tensor<f32>\n" ++
    s!"    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>\n" ++
    s!"    %loss = stablehlo.negate %lossm : tensor<f32>\n"
  let updParts := (((pnames.zip grads).zip pdims).zip (mnames.zip vnames)).map
    (fun (((nm, gr), ds), (mm, vv)) => emitAdamV nm gr mm vv ds (String.ofList (nm.toList.drop 1)))
  let upd := String.join (updParts.map (·.1))
  let allNames := (updParts.map (·.2.1)) ++ (updParts.map (·.2.2.1)) ++ (updParts.map (·.2.2.2))
    ++ ["%loss", "%bc1", "%bc2"]   -- the lr slot now carries the train loss for logging
  let allDims := pdims ++ pdims ++ pdims
  let retTy := String.intercalate ", " ((allDims.map (fun ds => ty ds)) ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"])
  let retVals := String.intercalate ", " allNames
  "module @m {\n" ++
  s!"  func.func @vit_adam_train_step(%x: {ty [cfg.b, d0]}, {psig}, {msig}, {vsig}, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: {ty [cfg.b, cfg.nc]}) -> ({retTy}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  consts ++
  s!"    %xr = stablehlo.reshape %x : ({ty [cfg.b, d0]}) -> {ty [cfg.b, cfg.ic, h, w]}\n" ++
  vitFwd "vit" "%xr" "%wConv" "%bConv" "%cls" "%pos" "%gF" "%bF" "%Wc" "%bc" blocks cfg ++
  cot ++
  vitBack "vit" "%dy" "%xr" "%wConv" "%Wc" "%gF" blocks cfg ++
  upd ++
  s!"    return {retVals} : {retTy}\n" ++ "  }\n}\n"

end ViTRender
