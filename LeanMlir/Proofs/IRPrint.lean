/-! # Phase 0 of `planning/verified_codegen.md` — `Back → StableHLO` printer

A small **computable** codegen AST (`Hlo`) + printer that renders a backward
graph to StableHLO text, in the exact form `MlirCodegen.lean` emits
(`dot_general … contracting_dims = [1] x [1]`, ReLU-back = `compare GT` +
`select`).

Why a separate AST and not `Back` directly: `Back` (in `IR.lean`) carries
abstract `Vec`/`Mat` (`Fin n → ℝ`, noncomputable), so its operand *values*
can't be printed and it can't be `#eval`'d. `Hlo` is the renderable mirror:
SSA names + shapes instead of values, **same structure** as `Back` (D1 in
the spec). The correspondence, per node:

    Hlo                Back                bridge (⟦Back⟧ = proven VJP)
    ───────            ──────────────      ─────────────────────────────
    .dot W m n         .dotGeneral W       dense_at_bridge      (= Mat.mulVec W)
    .reluBack p n      .selectPos p        relu_at_bridge       (= if p>0 then · else 0)
    .input "%dy"       .cotangent          —

So `emitMlpHlo` below mirrors `IR.emitMlpBack`, whose denotation is proven
equal to `mlp_has_vjp_at.backward` (`IR.mlp_whole_bridge`). The printed text
is therefore the rendering of a proof-backed computation — up to the printer
(this file, trusted), IREE, and float. (Phase 1: feed the output to IREE.)
-/

namespace Proofs.IRPrint

/-- StableHLO tensor type, row-major `f32`. -/
def tt (dims : List Nat) : String :=
  "tensor<" ++ String.intercalate "x" (dims.map toString) ++ "xf32>"

/-- `i1` (mask) tensor type. -/
def ti1 (dims : List Nat) : String :=
  "tensor<" ++ String.intercalate "x" (dims.map toString) ++ "xi1>"

/-- A backward graph in codegen form: SSA names + shapes, mirroring `Back`.
    `B` is the batch dimension (threaded by the printer). -/
inductive Hlo where
  /-- The cotangent input, at an externally-supplied SSA name. -/
  | input (ssa : String) : Hlo
  /-- Dense input-gradient: `dx = dot_general(·, W)`, `W : [m, n]`, takes a
      `[B, n]` cotangent to `[B, m]`. Mirrors `Back.dotGeneral`. -/
  | dot (wSSA : String) (m n : Nat) : Hlo → Hlo
  /-- ReLU backward at saved pre-activation `pSSA` (shape `[B, n]`):
      `compare GT 0` + `select`. Mirrors `Back.selectPos`. -/
  | reluBack (pSSA : String) (n : Nat) : Hlo → Hlo

/-- Fresh SSA name from a counter. -/
def fresh : StateM Nat String := do
  let n ← get; set (n + 1); pure s!"%bk{n}"

/-- Render an `Hlo` graph to a StableHLO op sequence; returns
    `(emitted code, result SSA)`. `B` = batch dim. -/
def Hlo.render (B : Nat) : Hlo → StateM Nat (String × String)
  | .input ssa => pure ("", ssa)
  | .dot wSSA m n e => do
      let (c, r) ← e.render B
      let o ← fresh
      pure (c ++
        s!"    {o} = stablehlo.dot_general {r}, {wSSA}, contracting_dims = [1] x [1],\n" ++
        s!"              precision = [DEFAULT, DEFAULT] : ({tt [B, n]}, {tt [m, n]}) -> {tt [B, m]}\n",
        o)
  | .reluBack pSSA n e => do
      let (c, r) ← e.render B
      let z ← fresh; let cmp ← fresh; let o ← fresh
      pure (c ++
        s!"    {z} = stablehlo.constant dense<0.0> : {tt [B, n]}\n" ++
        s!"    {cmp} = stablehlo.compare GT, {pSSA}, {z} : ({tt [B, n]}, {tt [B, n]}) -> {ti1 [B, n]}\n" ++
        s!"    {o} = stablehlo.select {cmp}, {r}, {z} : {ti1 [B, n]}, {tt [B, n]}\n",
        o)

/-- Render a backward graph into a labeled block (header + ops + the
    `dx` result), the way it would splice into a train-step function. -/
def renderBlock (name : String) (B : Nat) (h : Hlo) : String :=
  let (code, res) := (h.render B).run' 0
  s!"  // ── {name} backward (input-gradient / VJP chain) ──\n" ++
  s!"  //   inputs: %dy (cotangent), %W* (weights), %p* (saved ReLU pre-activations)\n" ++
  code ++ s!"  //   dx = {res}\n"

-- ════════════════════════════════════════════════════════════════
-- § Forward codegen AST — the renderable mirror of `IR.Fwd`
--
-- `Hlo` mirrors the backward IR `Back`; `HloF` mirrors the *forward* IR
-- `IR.Fwd` (whose denotation is proven `= mlpForward`, `IR.mlp_fwd_bridge`).
-- So the emitted forward StableHLO is `print (mlpFwdHlo)` by construction —
-- the forward enjoys the same render-from-proof-backed-IR status as the
-- backward, not a hand-written string. `dense` → `dot_general +
-- broadcast_in_dim + add`, `relu` → `maximum 0`.
-- ════════════════════════════════════════════════════════════════

/-- Forward graph in codegen form: SSA names + shapes, mirroring `IR.Fwd`. -/
inductive HloF where
  | input (ssa : String) : HloF
  | dense (wSSA bSSA : String) (m n : Nat) : HloF → HloF
  | relu (n : Nat) : HloF → HloF

/-- Render a forward graph to StableHLO. Threads `(dense#, relu#)`: dense
    outputs are named `%h{k}` (the saved pre-activations the backward reads
    for its ReLU masks), relu outputs `%a{k}` (the activations `dWℓ` reads).
    Returns `(code, resultSSA)`. -/
def HloF.render (B : Nat) : HloF → StateM (Nat × Nat) (String × String)
  | .input ssa => pure ("", ssa)
  | .dense wSSA bSSA m n e => do
      let (c, r) ← e.render B
      let (hk, ak) ← get; set (hk + 1, ak)
      let bb := s!"%hb{hk}"; let dd := s!"%hd{hk}"; let o := s!"%h{hk}"
      pure (c ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bSSA}, dims = [1] : ({tt [n]}) -> {tt [B, n]}\n" ++
        s!"    {dd} = stablehlo.dot_general {r}, {wSSA}, contracting_dims = [1] x [0],\n" ++
        s!"              precision = [DEFAULT, DEFAULT] : ({tt [B, m]}, {tt [m, n]}) -> {tt [B, n]}\n" ++
        s!"    {o} = stablehlo.add {dd}, {bb} : {tt [B, n]}\n",
        o)
  | .relu n e => do
      let (c, r) ← e.render B
      let (hk, ak) ← get; set (hk, ak + 1)
      let z := s!"%az{ak}"; let o := s!"%a{ak}"
      pure (c ++
        s!"    {z} = stablehlo.constant dense<0.0> : {tt [B, n]}\n" ++
        s!"    {o} = stablehlo.maximum {r}, {z} : {tt [B, n]}\n",
        o)

-- ════════════════════════════════════════════════════════════════
-- § Examples
-- ════════════════════════════════════════════════════════════════

/-- **Linear model** (a single dense `d₀ → d₁`): the whole input-gradient
    backward is one `dot_general`. Mirrors `IR.emitDenseBack`. -/
def linearHlo (d₀ d₁ : Nat) : Hlo := .dot "%W0" d₀ d₁ (.input "%dy")

/-- **2-hidden-layer MLP** `dense d₀→d₁ → relu → dense d₁→d₂ → relu → dense d₂→d₃`.
    Backward = `dot W₀ ∘ reluBack p₀ ∘ dot W₁ ∘ reluBack p₁ ∘ dot W₂` (applied
    to `%dy`). Mirrors `IR.emitMlpBack`. -/
def mlpHlo (d₀ d₁ d₂ d₃ : Nat) : Hlo :=
  .dot "%W0" d₀ d₁ (.reluBack "%p0" d₁
    (.dot "%W1" d₁ d₂ (.reluBack "%p1" d₂
      (.dot "%W2" d₂ d₃ (.input "%dy")))))

/-- Wrap the linear backward as a standalone `func.func` module. -/
def linearModule (B d₀ d₁ : Nat) : String :=
  let (body, res) := ((linearHlo d₀ d₁).render B).run' 0
  "module @m {\n" ++
  s!"  func.func @linear_back(%dy: {tt [B, d₁]}, %W0: {tt [d₀, d₁]}) -> {tt [B, d₀]} " ++ "{\n" ++
  body ++ s!"    return {res} : {tt [B, d₀]}\n" ++ "  }\n}\n"

/-- Wrap the MLP backward as a `func.func`: cotangent + weights + saved
    ReLU pre-activations in, `dx` out. (Input-gradient / VJP chain.) -/
def mlpModule (B d₀ d₁ d₂ d₃ : Nat) : String :=
  let (body, res) := ((mlpHlo d₀ d₁ d₂ d₃).render B).run' 0
  "module @m {\n" ++
  s!"  func.func @mlp_back(%dy: {tt [B, d₃]}, %W0: {tt [d₀, d₁]}, %W1: {tt [d₁, d₂]}, " ++
  s!"%W2: {tt [d₂, d₃]}, %p0: {tt [B, d₁]}, %p1: {tt [B, d₂]}) -> {tt [B, d₀]} " ++ "{\n" ++
  body ++ s!"    return {res} : {tt [B, d₀]}\n" ++ "  }\n}\n"

/-- **Forward activations prefix** `relu ∘ dense W₁ ∘ relu ∘ dense W₀`: the
    part of the forward whose outputs the backward consumes (`%h0,%h1`
    pre-activations; `%a0,%a1` activations; result `%a1`). Mirror of
    `IR.emitMlpFwd` minus the top dense; its denotation is the layer-1
    activation, with `%h0,%h1` proven `= IR.mlp_fwd_preact0/1`. -/
def mlpFwdActs (d₀ d₁ d₂ : Nat) : HloF :=
  .relu d₂ (.dense "%W1" "%b1" d₁ d₂ (.relu d₁ (.dense "%W0" "%b0" d₀ d₁ (.input "%x"))))

/-- **Whole MLP forward** `dense W₂ ∘ (mlpFwdActs)`. Mirror of `IR.emitMlpFwd`
    (`⟦emitMlpFwd⟧ = mlpForward`, `IR.mlp_fwd_bridge`). -/
def mlpFwdHlo (d₀ d₁ d₂ d₃ : Nat) : HloF :=
  .dense "%W2" "%b2" d₂ d₃ (mlpFwdActs d₀ d₁ d₂)

/-- Standalone forward `func.func @mlp_fwd`: `x` + weights in, logits out.
    The render-from-IR forward artifact, peer to `mlpModule` (the backward). -/
def mlpFwdModule (B d₀ d₁ d₂ d₃ : Nat) : String :=
  let (body, res) := ((mlpFwdHlo d₀ d₁ d₂ d₃).render B).run' (0, 0)
  "module @m {\n" ++
  s!"  func.func @mlp_fwd(%x: {tt [B,d₀]}, %W0: {tt [d₀,d₁]}, %b0: {tt [d₁]}, " ++
  s!"%W1: {tt [d₁,d₂]}, %b1: {tt [d₂]}, %W2: {tt [d₂,d₃]}, %b2: {tt [d₃]}) -> {tt [B,d₃]} " ++ "{\n" ++
  body ++ s!"    return {res} : {tt [B,d₃]}\n" ++ "  }\n}\n"

/-- Render the softmax-CE loss head `dy = softmax(logits) − onehot`:
    `exp` + `reduce`(add over classes) + `broadcast` + `divide` (= softmax),
    then `subtract` the target. Mirror of `IR.emitLossCot`, whose denotation
    is the proven `∂(crossEntropy)/∂logits` (`IR.lossCot_bridge`). -/
def renderLossCot (B c : Nat) (logits onehot dy : String) : String :=
  s!"    %le = stablehlo.exponential {logits} : {tt [B,c]}\n" ++
  s!"    %lz = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %lsum = stablehlo.reduce(%le init: %lz) applies stablehlo.add across dimensions = [1] : ({tt [B,c]}, tensor<f32>) -> {tt [B]}\n" ++
  s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({tt [B]}) -> {tt [B,c]}\n" ++
  s!"    %lsm = stablehlo.divide %le, %lsb : {tt [B,c]}\n" ++
  s!"    {dy} = stablehlo.subtract %lsm, {onehot} : {tt [B,c]}\n"

/-- Standalone softmax-CE loss-cotangent module: `logits` + target `onehot`
    in, `dy = ∂L/∂logits` out. The render-from-IR loss-head artifact. -/
def lossCotModule (B c : Nat) : String :=
  "module @m {\n" ++
  s!"  func.func @loss_cot(%logits: {tt [B,c]}, %onehot: {tt [B,c]}) -> {tt [B,c]} " ++ "{\n" ++
  renderLossCot B c "%logits" "%onehot" "%dy" ++
  s!"    return %dy : {tt [B,c]}\n" ++ "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § Full train step — forward + proof-backed backward + SGD
--
-- The backward above emits only `dx` (the input-gradient / VJP chain). A
-- train step also needs the *parameter* gradients and an optimizer update.
-- This module renders one full SGD step for the MLP:
--
--   forward  (PROOF-BACKED): rendered from `mlpFwdHlo`/`IR.emitMlpFwd`, whose
--                       denotation is the proven `mlpForward` (logits `%h2`);
--                       `%h0,%h1` are the pre-activations (IR.mlp_fwd_preact0/1)
--                       the backward reads, `%a0,%a1` the activations.
--   loss     (PROOF-BACKED): dy = softmax(%h2) − %onehot, rendered from
--                       `renderLossCot`/`IR.emitLossCot`, whose denotation is
--                       the proven softmax-CE gradient ∂L/∂logits
--                       (IR.lossCot_bridge). The cotangent is computed, not
--                       supplied.
--   backward (PROOF-BACKED): the dx chain is ⟦emitMlpBack⟧ = mlp_has_vjp_at
--                       .backward; each dWℓ = aₗ₋₁ᵀ·dyℓ (batch-contracting
--                       dot_general) and dbℓ = Σ_batch dyℓ (reduce-add) is
--                       IR.emitWeightGrad / IR.emitBiasGrad, bridged to the
--                       certified Jacobians by weight_grad_bridge /
--                       bias_grad_bridge.
--   SGD      (TRUSTED): θ' = θ − lr·dθ, elementwise.
--
-- So forward, loss cotangent, backward, AND the parameter gradients are ALL
-- renderings of proof-backed IR; only the SGD arithmetic (and the printer /
-- IREE / float) remain trusted.
-- ════════════════════════════════════════════════════════════════

/-- A full MLP SGD train step (`dense → relu → dense → relu → dense` + softmax
    cross-entropy), `dims d₀→d₁→d₂→d₃`, batch `B`, learning rate `lr` (a
    decimal literal). Inputs: `x`, the six parameters, and the target
    distribution `%onehot` (the labels). The loss cotangent
    `dy = softmax(logits) − onehot` is *computed* in-module, not supplied.
    Returns the six updated parameters. Forward / loss / backward / param-grads
    are all renderings of proof-backed IR; only the SGD arithmetic is the
    trusted frame. -/
def mlpTrainStepModule (B d₀ d₁ d₂ d₃ : Nat) (lr : String) : String :=
  let sgd (θ dθ θ' lrC sg ty : String) : String :=
    s!"    {lrC} = stablehlo.constant dense<{lr}> : {ty}\n" ++
    s!"    {sg} = stablehlo.multiply {dθ}, {lrC} : {ty}\n" ++
    s!"    {θ'} = stablehlo.subtract {θ}, {sg} : {ty}\n"
  let dg (o a b cdA cdB tyA tyB tyO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {b}, contracting_dims = [{cdA}] x [{cdB}],\n" ++
    s!"              precision = [DEFAULT, DEFAULT] : ({tyA}, {tyB}) -> {tyO}\n"
  -- full forward rendered from the forward IR mirror; logits = result handle
  let (fwd, logits) := ((mlpFwdHlo d₀ d₁ d₂ d₃).render B).run' (0, 0)
  "module @m {\n" ++
  s!"  func.func @mlp_train_step(%x: {tt [B,d₀]}, %W0: {tt [d₀,d₁]}, %b0: {tt [d₁]}, " ++
  s!"%W1: {tt [d₁,d₂]}, %b1: {tt [d₂]}, %W2: {tt [d₂,d₃]}, %b2: {tt [d₃]}, %onehot: {tt [B,d₃]}) -> " ++
  s!"({tt [d₀,d₁]}, {tt [d₁]}, {tt [d₁,d₂]}, {tt [d₂]}, {tt [d₂,d₃]}, {tt [d₃]}) " ++ "{\n" ++
  -- ── forward (proof-backed: rendered from mlpFwdHlo = emitMlpFwd) ──
  "    // ── forward (PROOF-BACKED: render mlpFwdHlo; ⟦emitMlpFwd⟧ = mlpForward,\n" ++
  "    //    %h0,%h1 = pre-activations IR.mlp_fwd_preact0/1, %a0,%a1 = activations, logits = result) ──\n" ++
  fwd ++
  -- ── loss cotangent (proof-backed: dy = softmax(logits) − onehot) ──
  "    // ── loss (PROOF-BACKED: dy = softmax(logits) − onehot = ⟦emitLossCot⟧ = ∂L/∂logits) ──\n" ++
  renderLossCot B d₃ logits "%onehot" "%dy" ++
  -- ── backward (proof-backed) ──
  "    // ── backward (PROOF-BACKED: dx chain = ⟦emitMlpBack⟧ = mlp_has_vjp_at.backward;\n" ++
  "    //    dWℓ/dbℓ = emitWeightGrad/emitBiasGrad, bridged to the certified Jacobians) ──\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %za = stablehlo.constant dense<0.0> : {tt [B,d₁]}\n" ++
  s!"    %zb = stablehlo.constant dense<0.0> : {tt [B,d₂]}\n" ++
  dg "%dW2" "%a1" "%dy" "0" "0" (tt [B,d₂]) (tt [B,d₃]) (tt [d₂,d₃]) ++
  s!"    %db2 = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : ({tt [B,d₃]}, tensor<f32>) -> {tt [d₃]}\n" ++
  dg "%dx2" "%dy" "%W2" "1" "1" (tt [B,d₃]) (tt [d₂,d₃]) (tt [B,d₂]) ++
  s!"    %m1 = stablehlo.compare GT, %h1, %zb : ({tt [B,d₂]}, {tt [B,d₂]}) -> {ti1 [B,d₂]}\n" ++
  s!"    %dy1 = stablehlo.select %m1, %dx2, %zb : {ti1 [B,d₂]}, {tt [B,d₂]}\n" ++
  dg "%dW1" "%a0" "%dy1" "0" "0" (tt [B,d₁]) (tt [B,d₂]) (tt [d₁,d₂]) ++
  s!"    %db1 = stablehlo.reduce(%dy1 init: %sc) applies stablehlo.add across dimensions = [0] : ({tt [B,d₂]}, tensor<f32>) -> {tt [d₂]}\n" ++
  dg "%dx1" "%dy1" "%W1" "1" "1" (tt [B,d₂]) (tt [d₁,d₂]) (tt [B,d₁]) ++
  s!"    %m0 = stablehlo.compare GT, %h0, %za : ({tt [B,d₁]}, {tt [B,d₁]}) -> {ti1 [B,d₁]}\n" ++
  s!"    %dy0 = stablehlo.select %m0, %dx1, %za : {ti1 [B,d₁]}, {tt [B,d₁]}\n" ++
  dg "%dW0" "%x" "%dy0" "0" "0" (tt [B,d₀]) (tt [B,d₁]) (tt [d₀,d₁]) ++
  s!"    %db0 = stablehlo.reduce(%dy0 init: %sc) applies stablehlo.add across dimensions = [0] : ({tt [B,d₁]}, tensor<f32>) -> {tt [d₁]}\n" ++
  -- ── SGD (trusted) ──
  "    // ── SGD update (trusted, elementwise): θ' = θ − lr·dθ ──\n" ++
  sgd "%W0" "%dW0" "%W0n" "%lW0" "%sW0" (tt [d₀,d₁]) ++
  sgd "%b0" "%db0" "%b0n" "%lb0" "%sb0" (tt [d₁]) ++
  sgd "%W1" "%dW1" "%W1n" "%lW1" "%sW1" (tt [d₁,d₂]) ++
  sgd "%b1" "%db1" "%b1n" "%lb1" "%sb1" (tt [d₂]) ++
  sgd "%W2" "%dW2" "%W2n" "%lW2" "%sW2" (tt [d₂,d₃]) ++
  sgd "%b2" "%db2" "%b2n" "%lb2" "%sb2" (tt [d₃]) ++
  s!"    return %W0n, %b0n, %W1n, %b1n, %W2n, %b2n : " ++
  s!"{tt [d₀,d₁]}, {tt [d₁]}, {tt [d₁,d₂]}, {tt [d₂]}, {tt [d₂,d₃]}, {tt [d₃]}\n" ++
  "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § CNN — conv forward + proof-backed conv backward (Phase 3, start)
--
-- The repo's `conv2d` is SAME-padding, stride-1 cross-correlation, which is
-- exactly `stablehlo.convolution` (XLA conv is cross-correlation, no flip).
-- The proven conv input-gradient is `IR.convBackDenote W = conv2d(reverseSwap
-- W, 0)` (`IR.conv3_node_bridge_1to2`, via the reversed-kernel identity
-- `conv_back_bridge_1to2`): swap in/out channels + flip both spatial axes,
-- then convolve. So the backward is `transpose [1,0,2,3]` + `reverse [2,3]` +
-- `convolution`. Layout: NCHW input/output `[B,C,H,W]`, OIHW kernel
-- `[oc,ic,kH,kW]` (= `Kernel4 oc ic kH kW`).
-- ════════════════════════════════════════════════════════════════

/-- A `stablehlo.convolution` op, SAME padding (`pH,pW`) + stride 1: NCHW
    in/out, OIHW kernel. (Explicit unit dilations — IREE's lowering wants
    them.) -/
def convOp (o lhs rhs tyL tyR tyO : String) (pH pW : Nat) : String :=
  s!"    {o} = stablehlo.convolution({lhs}, {rhs})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = " ++ "{" ++
    s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++
    "}\n" ++
  "      " ++ "{batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({tyL}, {tyR}) -> {tyO}\n"

/-- Wrap a single-function module. -/
def actMod (name argSig retTy body : String) : String :=
  "module @m {\n" ++ s!"  func.func @{name}({argSig}) -> {retTy} " ++ "{\n" ++ body ++ "  }\n}\n"

/-- `convOp` with an explicit `feature_group_count` (grouped/depthwise conv). -/
def convOpG (o lhs rhs tyL tyR tyO : String) (pH pW : Nat) (fgc : String) : String :=
  s!"    {o} = stablehlo.convolution({lhs}, {rhs})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = " ++ "{" ++
    s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++
    "}\n" ++
  "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {fgc} : i64" ++ "}" ++
    s!" : ({tyL}, {tyR}) -> {tyO}\n"

/-- Depthwise (per-channel) conv forward `depthwiseConv2d` as `@dw_fwd`:
    grouped convolution, `feature_group_count = c`, kernel `[c,1,kH,kW]`. -/
def depthwiseFwdM (c H W kH kW : Nat) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  actMod "dw_fwd" s!"%x: {tt [1,c,H,W]}, %W: {tt [c,kH,kW]}, %b: {tt [c]}" (tt [1,c,H,W])
    (s!"    %We = stablehlo.reshape %W : ({tt [c,kH,kW]}) -> {tt [c,1,kH,kW]}\n" ++
     convOpG "%cv" "%x" "%We" (tt [1,c,H,W]) (tt [c,1,kH,kW]) (tt [1,c,H,W]) pH pW (toString c) ++
     s!"    %bb = stablehlo.broadcast_in_dim %b, dims = [1] : ({tt [c]}) -> {tt [1,c,H,W]}\n" ++
     s!"    %o = stablehlo.add %cv, %bb : {tt [1,c,H,W]}\n    return %o : {tt [1,c,H,W]}\n")

/-- Depthwise conv input-gradient `depthwiseConv2d_input_grad_formula` as
    `@dw_back`: per-channel reversed-kernel grouped conv (no channel
    transpose — depthwise channels don't mix). -/
def depthwiseBackM (c H W kH kW : Nat) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  actMod "dw_back" s!"%dy: {tt [1,c,H,W]}, %W: {tt [c,kH,kW]}" (tt [1,c,H,W])
    (s!"    %We = stablehlo.reshape %W : ({tt [c,kH,kW]}) -> {tt [c,1,kH,kW]}\n" ++
     s!"    %Wr = stablehlo.reverse %We, dims = [2, 3] : {tt [c,1,kH,kW]}\n" ++
     convOpG "%dx" "%dy" "%Wr" (tt [1,c,H,W]) (tt [c,1,kH,kW]) (tt [1,c,H,W]) pH pW (toString c) ++
     s!"    return %dx : {tt [1,c,H,W]}\n")

/-- Conv forward `conv2d W b` as a `func.func @conv_fwd` (convolution + bias). -/
def convFwdModule (B ic oc H Wd kH kW : Nat) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  "module @m {\n" ++
  s!"  func.func @conv_fwd(%x: {tt [B,ic,H,Wd]}, %W: {tt [oc,ic,kH,kW]}, %b: {tt [oc]}) -> {tt [B,oc,H,Wd]} " ++ "{\n" ++
  convOp "%c" "%x" "%W" (tt [B,ic,H,Wd]) (tt [oc,ic,kH,kW]) (tt [B,oc,H,Wd]) pH pW ++
  s!"    %bb = stablehlo.broadcast_in_dim %b, dims = [1] : ({tt [oc]}) -> {tt [B,oc,H,Wd]}\n" ++
  s!"    %o = stablehlo.add %c, %bb : {tt [B,oc,H,Wd]}\n" ++
  s!"    return %o : {tt [B,oc,H,Wd]}\n" ++ "  }\n}\n"

/-- Conv input-gradient backward `IR.convBackDenote W` as `@conv_back`:
    `transpose` (swap channels) + `reverse` (flip spatial) + `convolution`.
    Denotes the proven conv input-VJP (`conv_back_bridge_1to2`). -/
def convBackModule (B ic oc H Wd kH kW : Nat) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  "module @m {\n" ++
  s!"  func.func @conv_back(%dy: {tt [B,oc,H,Wd]}, %W: {tt [oc,ic,kH,kW]}) -> {tt [B,ic,H,Wd]} " ++ "{\n" ++
  s!"    %Wt = stablehlo.transpose %W, dims = [1, 0, 2, 3] : ({tt [oc,ic,kH,kW]}) -> {tt [ic,oc,kH,kW]}\n" ++
  s!"    %Wr = stablehlo.reverse %Wt, dims = [2, 3] : {tt [ic,oc,kH,kW]}\n" ++
  convOp "%dx" "%dy" "%Wr" (tt [B,oc,H,Wd]) (tt [ic,oc,kH,kW]) (tt [B,ic,H,Wd]) pH pW ++
  s!"    return %dx : {tt [B,ic,H,Wd]}\n" ++ "  }\n}\n"

-- 2×2 stride-2 max pool: forward = `reduce_window`(max); backward =
-- `select_and_scatter` (route dy to each window's argmax), which is exactly
-- `IR.maxPoolBackDenote` and matches the proven maxpool VJP at smooth points
-- (unique argmax — `maxpool_back_bridge`/`maxpool3_node_bridge`; GE tie-break).

/-- Max-pool forward `IR.maxPool2` as `@maxpool_fwd`: `reduce_window` max,
    window/stride `[1,1,2,2]` over NCHW. -/
def maxpoolFwdModule (B c h w : Nat) : String :=
  "module @m {\n" ++
  s!"  func.func @maxpool_fwd(%x: {tt [B,c,2*h,2*w]}) -> {tt [B,c,h,w]} " ++ "{\n" ++
  "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
  "    %p = \"stablehlo.reduce_window\"(%x, %ninf) (" ++ "{\n" ++
  "      ^bb0(%a: tensor<f32>, %b: tensor<f32>):\n" ++
  "        %m = stablehlo.maximum %a, %b : tensor<f32>\n" ++
  "        stablehlo.return %m : tensor<f32>\n" ++
  "    }) " ++ "{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
  s!" : ({tt [B,c,2*h,2*w]}, tensor<f32>) -> {tt [B,c,h,w]}\n" ++
  s!"    return %p : {tt [B,c,h,w]}\n" ++ "  }\n}\n"

/-- Max-pool backward `IR.maxPoolBackDenote` as `@maxpool_back`:
    `select_and_scatter` (select = GE, scatter = add) routes `dy` to each
    window's argmax cell — the proven maxpool VJP at smooth points. -/
def maxpoolBackModule (B c h w : Nat) : String :=
  "module @m {\n" ++
  s!"  func.func @maxpool_back(%x: {tt [B,c,2*h,2*w]}, %dy: {tt [B,c,h,w]}) -> {tt [B,c,2*h,2*w]} " ++ "{\n" ++
  "    %z = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    %dx = \"stablehlo.select_and_scatter\"(%x, %dy, %z) (" ++ "{\n" ++
  "      ^bb0(%a: tensor<f32>, %b: tensor<f32>):\n" ++
  "        %ge = stablehlo.compare GE, %a, %b : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
  "        stablehlo.return %ge : tensor<i1>\n" ++
  "    }, " ++ "{\n" ++
  "      ^bb0(%a: tensor<f32>, %b: tensor<f32>):\n" ++
  "        %s = stablehlo.add %a, %b : tensor<f32>\n" ++
  "        stablehlo.return %s : tensor<f32>\n" ++
  "    }) " ++ "{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
  s!" : ({tt [B,c,2*h,2*w]}, {tt [B,c,h,w]}, tensor<f32>) -> {tt [B,c,2*h,2*w]}\n" ++
  s!"    return %dx : {tt [B,c,2*h,2*w]}\n" ++ "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § CNN capstone — conv → relu → maxpool → flatten → dense, fwd + dx
--
-- The CNN analogue of `mlpModule` (the whole backward chain): one
-- `func.func @cnn_back` that runs the forward far enough to save the
-- activations the backward reads (the conv pre-activation `%hconv` for the
-- ReLU mask, the ReLU output `%a` = maxpool's operand), then the full
-- input-gradient backward, composing EVERY proof-backed primitive:
--
--   dense_back  (dot_general)         dense_at_bridge
--   reshape     (flatten bijection)   conv_flatten_bridge / maxpool_flatten_bridge
--   maxpool_back(select_and_scatter)  maxpool_back_bridge   (route dy to argmax)
--   relu_back   (compare GT + select) relu_at_bridge
--   conv_back   (transpose+reverse+conv) conv_back_bridge_1to2
--
-- composed via the proven chain rules `denote_subst` / `denote_subst3`. The
-- Tensor3 flatten is C-order row-major (`Tensor3.flatten` = `stablehlo.reshape`),
-- so the reshape is proof-faithful. Layout NCHW / OIHW, batch 1. (dx only;
-- the conv weight-gradient + a full CNN train step is the next step.)
-- ════════════════════════════════════════════════════════════════

/-- A small CNN forward + input-gradient backward, `@cnn_back`:
    `conv (ic→oc, kH×kW SAME) → relu → maxpool 2×2 → flatten → dense
    (flat→nClass)`, dims from `(ic,oc,H,W,kH,kW,nClass)`, batch 1. Inputs:
    `x`, conv weights `%Wc,%bc`, dense weight `%Wd`, cotangent `%dy`; output
    `dx`. Every op is a rendered proof-backed bridge (see the section note). -/
def cnnModule (ic oc H W kH kW nClass : Nat) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2; let W2 := W / 2; let flat := oc * H2 * W2
  "module @m {\n" ++
  s!"  func.func @cnn_back(%x: {tt [1,ic,H,W]}, %Wc: {tt [oc,ic,kH,kW]}, %bc: {tt [oc]}, " ++
  s!"%Wd: {tt [flat,nClass]}, %dy: {tt [1,nClass]}) -> {tt [1,ic,H,W]} " ++ "{\n" ++
  -- forward (to the saved activations the backward reads)
  "    // ── forward: conv → relu (saves %hconv pre-act, %a = maxpool operand) ──\n" ++
  convOp "%cv" "%x" "%Wc" (tt [1,ic,H,W]) (tt [oc,ic,kH,kW]) (tt [1,oc,H,W]) pH pW ++
  s!"    %bcb = stablehlo.broadcast_in_dim %bc, dims = [1] : ({tt [oc]}) -> {tt [1,oc,H,W]}\n" ++
  s!"    %hconv = stablehlo.add %cv, %bcb : {tt [1,oc,H,W]}\n" ++
  s!"    %zc = stablehlo.constant dense<0.0> : {tt [1,oc,H,W]}\n" ++
  s!"    %a = stablehlo.maximum %hconv, %zc : {tt [1,oc,H,W]}\n" ++
  -- backward (the full dx chain)
  "    // ── backward (dx): dense → reshape → maxpool → relu → conv, all proof-backed ──\n" ++
  s!"    %dflat = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1],\n" ++
  s!"              precision = [DEFAULT, DEFAULT] : ({tt [1,nClass]}, {tt [flat,nClass]}) -> {tt [1,flat]}\n" ++
  s!"    %dp = stablehlo.reshape %dflat : ({tt [1,flat]}) -> {tt [1,oc,H2,W2]}\n" ++
  "    %zs = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    %da = \"stablehlo.select_and_scatter\"(%a, %dp, %zs) (" ++ "{\n" ++
  "      ^bb0(%u: tensor<f32>, %v: tensor<f32>):\n" ++
  "        %ge = stablehlo.compare GE, %u, %v : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
  "        stablehlo.return %ge : tensor<i1>\n" ++
  "    }, " ++ "{\n" ++
  "      ^bb0(%u: tensor<f32>, %v: tensor<f32>):\n" ++
  "        %s = stablehlo.add %u, %v : tensor<f32>\n" ++
  "        stablehlo.return %s : tensor<f32>\n" ++
  "    }) " ++ "{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
  s!" : ({tt [1,oc,H,W]}, {tt [1,oc,H2,W2]}, tensor<f32>) -> {tt [1,oc,H,W]}\n" ++
  s!"    %mc = stablehlo.compare GT, %hconv, %zc : ({tt [1,oc,H,W]}, {tt [1,oc,H,W]}) -> {ti1 [1,oc,H,W]}\n" ++
  s!"    %dhconv = stablehlo.select %mc, %da, %zc : {ti1 [1,oc,H,W]}, {tt [1,oc,H,W]}\n" ++
  s!"    %Wt = stablehlo.transpose %Wc, dims = [1, 0, 2, 3] : ({tt [oc,ic,kH,kW]}) -> {tt [ic,oc,kH,kW]}\n" ++
  s!"    %Wr = stablehlo.reverse %Wt, dims = [2, 3] : {tt [ic,oc,kH,kW]}\n" ++
  convOp "%dx" "%dhconv" "%Wr" (tt [1,oc,H,W]) (tt [ic,oc,kH,kW]) (tt [1,ic,H,W]) pH pW ++
  s!"    return %dx : {tt [1,ic,H,W]}\n" ++ "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § CNN train step — the CNN peer of mlpTrainStepModule (Phase 3, rest)
--
-- A full SGD step for the conv-net, every mathematical op proof-backed:
--   forward  conv → relu → maxpool → flatten → dense (logits)
--   loss     dy = softmax(logits) − onehot                 (lossCot_bridge)
--   backward dense_back → reshape → maxpool_back → relu_back → (dhconv)
--   grads    dWd/dbd (dense, weight_grad_bridge/bias_grad_bridge),
--            dWc = conv weight-grad via the **transpose trick** — the SAME
--                  `stablehlo.convolution` with input/gradient reshaped (input
--                  channels as batch, gradient as the kernel); proven formula
--                  `conv2d_weight_grad_has_vjp`, IREE-friendly (no exotic
--                  dim_numbers — iree#21955),
--            dbc = Σ_{batch,spatial} dhconv (conv2d_bias_grad_formula).
--   SGD      θ' = θ − lr·dθ (trusted).
-- (The transpose-trick render is numerically validated here, as the repo's
-- check_jacobians does; the graph-denotation bridge is the same expansion
-- as conv_back_bridge_1to2 — deferred, not a gap in the math.)
-- ════════════════════════════════════════════════════════════════

/-- A full CNN SGD train step `@cnn_train_step`. Inputs: `x`, conv `%Wc,%bc`,
    dense `%Wd,%bd`, target `%onehot`. Returns the four updated parameters.
    Forward/loss/backward/param-grads are renderings of proof-backed IR (conv
    weight-grad numerically validated); only SGD is the trusted frame. -/
def cnnTrainStepModule (ic oc H W kH kW nClass : Nat) (lr : String) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2; let W2 := W / 2; let flat := oc * H2 * W2
  let dg (o a b cdA cdB tyA tyB tyO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {b}, contracting_dims = [{cdA}] x [{cdB}],\n" ++
    s!"              precision = [DEFAULT, DEFAULT] : ({tyA}, {tyB}) -> {tyO}\n"
  let sgd (θ dθ θ' lrC sg ty : String) : String :=
    s!"    {lrC} = stablehlo.constant dense<{lr}> : {ty}\n" ++
    s!"    {sg} = stablehlo.multiply {dθ}, {lrC} : {ty}\n" ++
    s!"    {θ'} = stablehlo.subtract {θ}, {sg} : {ty}\n"
  "module @m {\n" ++
  s!"  func.func @cnn_train_step(%x: {tt [1,ic,H,W]}, %Wc: {tt [oc,ic,kH,kW]}, %bc: {tt [oc]}, " ++
  s!"%Wd: {tt [flat,nClass]}, %bd: {tt [nClass]}, %onehot: {tt [1,nClass]}) -> " ++
  s!"({tt [oc,ic,kH,kW]}, {tt [oc]}, {tt [flat,nClass]}, {tt [nClass]}) " ++ "{\n" ++
  -- forward
  "    // ── forward: conv → relu → maxpool → flatten → dense (PROOF-BACKED) ──\n" ++
  convOp "%cv" "%x" "%Wc" (tt [1,ic,H,W]) (tt [oc,ic,kH,kW]) (tt [1,oc,H,W]) pH pW ++
  s!"    %bcb = stablehlo.broadcast_in_dim %bc, dims = [1] : ({tt [oc]}) -> {tt [1,oc,H,W]}\n" ++
  s!"    %hconv = stablehlo.add %cv, %bcb : {tt [1,oc,H,W]}\n" ++
  s!"    %zc = stablehlo.constant dense<0.0> : {tt [1,oc,H,W]}\n" ++
  s!"    %a = stablehlo.maximum %hconv, %zc : {tt [1,oc,H,W]}\n" ++
  "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
  "    %p = \"stablehlo.reduce_window\"(%a, %ninf) (" ++ "{\n" ++
  "      ^bb0(%u: tensor<f32>, %v: tensor<f32>):\n" ++
  "        %m = stablehlo.maximum %u, %v : tensor<f32>\n" ++
  "        stablehlo.return %m : tensor<f32>\n" ++
  "    }) " ++ "{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
  s!" : ({tt [1,oc,H,W]}, tensor<f32>) -> {tt [1,oc,H2,W2]}\n" ++
  s!"    %flat = stablehlo.reshape %p : ({tt [1,oc,H2,W2]}) -> {tt [1,flat]}\n" ++
  dg "%xw" "%flat" "%Wd" "1" "0" (tt [1,flat]) (tt [flat,nClass]) (tt [1,nClass]) ++
  s!"    %bdb = stablehlo.broadcast_in_dim %bd, dims = [1] : ({tt [nClass]}) -> {tt [1,nClass]}\n" ++
  s!"    %logits = stablehlo.add %xw, %bdb : {tt [1,nClass]}\n" ++
  -- loss cotangent
  "    // ── loss: dlog = softmax(logits) − onehot (PROOF-BACKED) ──\n" ++
  renderLossCot 1 nClass "%logits" "%onehot" "%dlog" ++
  -- backward + parameter gradients
  "    // ── backward + param grads (PROOF-BACKED; conv dW = transpose trick) ──\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  dg "%dWd" "%flat" "%dlog" "0" "0" (tt [1,flat]) (tt [1,nClass]) (tt [flat,nClass]) ++
  s!"    %dbd = stablehlo.reduce(%dlog init: %sc) applies stablehlo.add across dimensions = [0] : ({tt [1,nClass]}, tensor<f32>) -> {tt [nClass]}\n" ++
  dg "%dflat" "%dlog" "%Wd" "1" "1" (tt [1,nClass]) (tt [flat,nClass]) (tt [1,flat]) ++
  s!"    %dp = stablehlo.reshape %dflat : ({tt [1,flat]}) -> {tt [1,oc,H2,W2]}\n" ++
  "    %da = \"stablehlo.select_and_scatter\"(%a, %dp, %sc) (" ++ "{\n" ++
  "      ^bb0(%u: tensor<f32>, %v: tensor<f32>):\n" ++
  "        %ge = stablehlo.compare GE, %u, %v : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
  "        stablehlo.return %ge : tensor<i1>\n" ++
  "    }, " ++ "{\n" ++
  "      ^bb0(%u: tensor<f32>, %v: tensor<f32>):\n" ++
  "        %s = stablehlo.add %u, %v : tensor<f32>\n" ++
  "        stablehlo.return %s : tensor<f32>\n" ++
  "    }) " ++ "{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
  s!" : ({tt [1,oc,H,W]}, {tt [1,oc,H2,W2]}, tensor<f32>) -> {tt [1,oc,H,W]}\n" ++
  s!"    %mc = stablehlo.compare GT, %hconv, %zc : ({tt [1,oc,H,W]}, {tt [1,oc,H,W]}) -> {ti1 [1,oc,H,W]}\n" ++
  s!"    %dhconv = stablehlo.select %mc, %da, %zc : {ti1 [1,oc,H,W]}, {tt [1,oc,H,W]}\n" ++
  s!"    %xt = stablehlo.transpose %x, dims = [1, 0, 2, 3] : ({tt [1,ic,H,W]}) -> {tt [ic,1,H,W]}\n" ++
  s!"    %dht = stablehlo.transpose %dhconv, dims = [1, 0, 2, 3] : ({tt [1,oc,H,W]}) -> {tt [oc,1,H,W]}\n" ++
  convOp "%dWcraw" "%xt" "%dht" (tt [ic,1,H,W]) (tt [oc,1,H,W]) (tt [ic,oc,kH,kW]) pH pW ++
  s!"    %dWc = stablehlo.transpose %dWcraw, dims = [1, 0, 2, 3] : ({tt [ic,oc,kH,kW]}) -> {tt [oc,ic,kH,kW]}\n" ++
  s!"    %dbc = stablehlo.reduce(%dhconv init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({tt [1,oc,H,W]}, tensor<f32>) -> {tt [oc]}\n" ++
  -- SGD
  "    // ── SGD update (trusted, elementwise): θ' = θ − lr·dθ ──\n" ++
  sgd "%Wc" "%dWc" "%Wcn" "%lWc" "%sWc" (tt [oc,ic,kH,kW]) ++
  sgd "%bc" "%dbc" "%bcn" "%lbc" "%sbc" (tt [oc]) ++
  sgd "%Wd" "%dWd" "%Wdn" "%lWd" "%sWd" (tt [flat,nClass]) ++
  sgd "%bd" "%dbd" "%bdn" "%lbd" "%sbd" (tt [nClass]) ++
  s!"    return %Wcn, %bcn, %Wdn, %bdn : " ++
  s!"{tt [oc,ic,kH,kW]}, {tt [oc]}, {tt [flat,nClass]}, {tt [nClass]}\n" ++
  "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § BatchNorm / LayerNorm — the reduce/broadcast chapter (Phase 3 sweep)
--
-- The repo's `bnForward` (Vec n → Vec n) normalizes over the feature axis:
--   μ = Σx/N, σ² = Σ(x−μ)²/N, x̂ = (x−μ)·istd, istd = 1/√(σ²+ε), y = γx̂+β.
-- Its proven backward (`bn_back_bridge`, the consolidated 3-term rank-1 form):
--   dx = (istd/N)·( N·dx̂ − Σⱼdx̂ⱼ − x̂·Σⱼ x̂ⱼdx̂ⱼ ),  dx̂ = γ·dy.
-- Rendered with `reduce`(add over the feature axis) + `broadcast_in_dim` +
-- `rsqrt`/`multiply`/`subtract` — the same renderers softmax, LayerNorm
-- (definitionally BN) and attention reuse. (γ,β scalar, per the Vec proof.)
-- ════════════════════════════════════════════════════════════════

/-- Reduce-sum over the feature axis [1] of `[B,n]`, broadcast back to `[B,n]`.
    `init %sc` (a `tensor<f32>` 0) must already be in scope. -/
def reduceSumBcast (o src : String) (B n : Nat) : String :=
  s!"    {o}_r = stablehlo.reduce({src} init: %sc) applies stablehlo.add across dimensions = [1] : ({tt [B,n]}, tensor<f32>) -> {tt [B]}\n" ++
  s!"    {o} = stablehlo.broadcast_in_dim {o}_r, dims = [0] : ({tt [B]}) -> {tt [B,n]}\n"

/-- BatchNorm/LayerNorm forward `bnForward` as `@bn_fwd` (γ,β scalar inputs). -/
def bnFwdModule (B n : Nat) (eps : String) : String :=
  "module @m {\n" ++
  s!"  func.func @bn_fwd(%x: {tt [B,n]}, %g: tensor<f32>, %b: tensor<f32>) -> {tt [B,n]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %nf = stablehlo.constant dense<{n}.0> : {tt [B,n]}\n" ++
  s!"    %eps = stablehlo.constant dense<{eps}> : {tt [B,n]}\n" ++
  reduceSumBcast "%sum" "%x" B n ++
  s!"    %mu = stablehlo.divide %sum, %nf : {tt [B,n]}\n" ++
  s!"    %xc = stablehlo.subtract %x, %mu : {tt [B,n]}\n" ++
  s!"    %sq = stablehlo.multiply %xc, %xc : {tt [B,n]}\n" ++
  reduceSumBcast "%vs" "%sq" B n ++
  s!"    %var = stablehlo.divide %vs, %nf : {tt [B,n]}\n" ++
  s!"    %vare = stablehlo.add %var, %eps : {tt [B,n]}\n" ++
  s!"    %istd = stablehlo.rsqrt %vare : {tt [B,n]}\n" ++
  s!"    %xhat = stablehlo.multiply %xc, %istd : {tt [B,n]}\n" ++
  s!"    %gb = stablehlo.broadcast_in_dim %g, dims = [] : (tensor<f32>) -> {tt [B,n]}\n" ++
  s!"    %bb = stablehlo.broadcast_in_dim %b, dims = [] : (tensor<f32>) -> {tt [B,n]}\n" ++
  s!"    %gx = stablehlo.multiply %xhat, %gb : {tt [B,n]}\n" ++
  s!"    %y = stablehlo.add %gx, %bb : {tt [B,n]}\n" ++
  s!"    return %y : {tt [B,n]}\n" ++ "  }\n}\n"

/-- BatchNorm/LayerNorm backward `bn_has_vjp.backward` (the proven 3-term
    rank-1 form, `bn_back_bridge`) as `@bn_back`: recompute x̂,istd, then
    `dx = (istd/N)·(N·dx̂ − Σdx̂ − x̂·Σ(x̂·dx̂))`, `dx̂ = γ·dy`. -/
def bnBackModule (B n : Nat) (eps : String) : String :=
  "module @m {\n" ++
  s!"  func.func @bn_back(%x: {tt [B,n]}, %g: tensor<f32>, %dy: {tt [B,n]}) -> {tt [B,n]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %nf = stablehlo.constant dense<{n}.0> : {tt [B,n]}\n" ++
  s!"    %eps = stablehlo.constant dense<{eps}> : {tt [B,n]}\n" ++
  reduceSumBcast "%sum" "%x" B n ++
  s!"    %mu = stablehlo.divide %sum, %nf : {tt [B,n]}\n" ++
  s!"    %xc = stablehlo.subtract %x, %mu : {tt [B,n]}\n" ++
  s!"    %sq = stablehlo.multiply %xc, %xc : {tt [B,n]}\n" ++
  reduceSumBcast "%vs" "%sq" B n ++
  s!"    %var = stablehlo.divide %vs, %nf : {tt [B,n]}\n" ++
  s!"    %vare = stablehlo.add %var, %eps : {tt [B,n]}\n" ++
  s!"    %istd = stablehlo.rsqrt %vare : {tt [B,n]}\n" ++
  s!"    %xhat = stablehlo.multiply %xc, %istd : {tt [B,n]}\n" ++
  s!"    %gb = stablehlo.broadcast_in_dim %g, dims = [] : (tensor<f32>) -> {tt [B,n]}\n" ++
  s!"    %dxhat = stablehlo.multiply %gb, %dy : {tt [B,n]}\n" ++
  reduceSumBcast "%sdx" "%dxhat" B n ++
  s!"    %xd = stablehlo.multiply %xhat, %dxhat : {tt [B,n]}\n" ++
  reduceSumBcast "%sxdx" "%xd" B n ++
  s!"    %t1 = stablehlo.multiply %dxhat, %nf : {tt [B,n]}\n" ++
  s!"    %i1 = stablehlo.subtract %t1, %sdx : {tt [B,n]}\n" ++
  s!"    %xs = stablehlo.multiply %xhat, %sxdx : {tt [B,n]}\n" ++
  s!"    %i2 = stablehlo.subtract %i1, %xs : {tt [B,n]}\n" ++
  s!"    %s = stablehlo.divide %istd, %nf : {tt [B,n]}\n" ++
  s!"    %dx = stablehlo.multiply %s, %i2 : {tt [B,n]}\n" ++
  s!"    return %dx : {tt [B,n]}\n" ++ "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § Softmax — the rank-1 chapter (Phase 3 sweep; the attention building block)
--
-- `softmax c z = exp(z)/Σexp(z)` (over the feature axis). Proven backward
-- (`softmax_back_bridge`, rank-1): `dz = p ⊙ (dy − ⟨p, dy⟩)`, one reduction
-- `⟨p,dy⟩` + broadcast-subtract + scale by `p` — same shape as BN. Reuses the
-- `reduceSumBcast` renderer; this is the core nonlinearity of attention.
-- ════════════════════════════════════════════════════════════════

/-- Softmax over the feature axis into `o`: `exp` + `reduce`(add) + `broadcast`
    + `divide`. `%sc` (a `tensor<f32>` 0) must be in scope. -/
def renderSoftmax (o z : String) (B c : Nat) : String :=
  s!"    %se = stablehlo.exponential {z} : {tt [B,c]}\n" ++
  s!"    %ssum_r = stablehlo.reduce(%se init: %sc) applies stablehlo.add across dimensions = [1] : ({tt [B,c]}, tensor<f32>) -> {tt [B]}\n" ++
  s!"    %ssumb = stablehlo.broadcast_in_dim %ssum_r, dims = [0] : ({tt [B]}) -> {tt [B,c]}\n" ++
  s!"    {o} = stablehlo.divide %se, %ssumb : {tt [B,c]}\n"

/-- Softmax forward `softmax c` as `@softmax_fwd`. -/
def softmaxFwdModule (B c : Nat) : String :=
  "module @m {\n" ++
  s!"  func.func @softmax_fwd(%z: {tt [B,c]}) -> {tt [B,c]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  renderSoftmax "%p" "%z" B c ++
  s!"    return %p : {tt [B,c]}\n" ++ "  }\n}\n"

/-- Softmax backward `softmax_has_vjp.backward` (rank-1, `softmax_back_bridge`)
    as `@softmax_back`: `dz = p ⊙ (dy − Σⱼ pⱼ·dyⱼ)`. -/
def softmaxBackModule (B c : Nat) : String :=
  "module @m {\n" ++
  s!"  func.func @softmax_back(%z: {tt [B,c]}, %dy: {tt [B,c]}) -> {tt [B,c]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  renderSoftmax "%p" "%z" B c ++
  s!"    %pdy = stablehlo.multiply %p, %dy : {tt [B,c]}\n" ++
  reduceSumBcast "%s" "%pdy" B c ++
  s!"    %d = stablehlo.subtract %dy, %s : {tt [B,c]}\n" ++
  s!"    %dz = stablehlo.multiply %p, %d : {tt [B,c]}\n" ++
  s!"    return %dz : {tt [B,c]}\n" ++ "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § Scaled dot-product attention — the apex (Phase 3 sweep, ViT core)
--
-- `sdpa Q K V = softmax(QKᵀ/√d)·V`. Proven backward (sdpa_back_Q/K/V_correct),
-- step by step:  dV = wᵀ·dOut,  dWeights = dOut·Vᵀ,  dScaled =
-- rowsoftmax-VJP(w, dWeights) = w⊙(dW − ⟨w,dW⟩),  dScores = dScaled/√d,
-- dQ = dScores·K,  dK = dScoresᵀ·Q. All `dot_general` + the softmax above +
-- a scalar scale — "no novel structural move" (Attention.lean): three dense
-- backwards, two matmuls, one row-softmax, one scale. (Q,K,V : Mat n d,
-- single head — the W_q/W_k/W_v/W_o projections are dense layers already done.)
-- ════════════════════════════════════════════════════════════════

/-- A `dot_general` matmul (no batch dims): contract `cdA`×`cdB`. -/
def matdg (o a b cdA cdB tyA tyB tyO : String) : String :=
  s!"    {o} = stablehlo.dot_general {a}, {b}, contracting_dims = [{cdA}] x [{cdB}],\n" ++
  s!"              precision = [DEFAULT, DEFAULT] : ({tyA}, {tyB}) -> {tyO}\n"

/-- SDPA forward `sdpa n d` as `@sdpa_fwd`: scores = QKᵀ, scale, rowsoftmax, ·V. -/
def sdpaFwdModule (n d : Nat) (scale : String) : String :=
  "module @m {\n" ++
  s!"  func.func @sdpa_fwd(%Q: {tt [n,d]}, %K: {tt [n,d]}, %V: {tt [n,d]}) -> {tt [n,d]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %scl = stablehlo.constant dense<{scale}> : {tt [n,n]}\n" ++
  matdg "%scores" "%Q" "%K" "1" "1" (tt [n,d]) (tt [n,d]) (tt [n,n]) ++
  s!"    %scaled = stablehlo.multiply %scores, %scl : {tt [n,n]}\n" ++
  renderSoftmax "%weights" "%scaled" n n ++
  matdg "%out" "%weights" "%V" "1" "0" (tt [n,n]) (tt [n,d]) (tt [n,d]) ++
  s!"    return %out : {tt [n,d]}\n" ++ "  }\n}\n"

/-- SDPA backward (the three proven input grads `sdpa_back_Q/K/V`) as
    `@sdpa_back`: recompute the softmax weights, then the matmul/softmax-VJP
    chain. Returns `(dQ, dK, dV)`. -/
def sdpaBackModule (n d : Nat) (scale : String) : String :=
  "module @m {\n" ++
  s!"  func.func @sdpa_back(%Q: {tt [n,d]}, %K: {tt [n,d]}, %V: {tt [n,d]}, %dOut: {tt [n,d]}) -> " ++
  s!"({tt [n,d]}, {tt [n,d]}, {tt [n,d]}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %scl = stablehlo.constant dense<{scale}> : {tt [n,n]}\n" ++
  matdg "%scores" "%Q" "%K" "1" "1" (tt [n,d]) (tt [n,d]) (tt [n,n]) ++
  s!"    %scaled = stablehlo.multiply %scores, %scl : {tt [n,n]}\n" ++
  renderSoftmax "%weights" "%scaled" n n ++
  matdg "%dWeights" "%dOut" "%V" "1" "1" (tt [n,d]) (tt [n,d]) (tt [n,n]) ++
  matdg "%dV" "%weights" "%dOut" "0" "0" (tt [n,n]) (tt [n,d]) (tt [n,d]) ++
  s!"    %pdw = stablehlo.multiply %weights, %dWeights : {tt [n,n]}\n" ++
  reduceSumBcast "%srow" "%pdw" n n ++
  s!"    %diff = stablehlo.subtract %dWeights, %srow : {tt [n,n]}\n" ++
  s!"    %dScaled = stablehlo.multiply %weights, %diff : {tt [n,n]}\n" ++
  s!"    %dScores = stablehlo.multiply %dScaled, %scl : {tt [n,n]}\n" ++
  matdg "%dQ" "%dScores" "%K" "1" "0" (tt [n,n]) (tt [n,d]) (tt [n,d]) ++
  matdg "%dK" "%dScores" "%Q" "0" "0" (tt [n,n]) (tt [n,d]) (tt [n,d]) ++
  s!"    return %dQ, %dK, %dV : {tt [n,d]}, {tt [n,d]}, {tt [n,d]}\n" ++ "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § Pointwise activations (Phase 3 sweep) — gelu, swish, sigmoid, relu6
--
-- Each has a diagonal Jacobian, so its proven backward is `dy ⊙ act'(x)`
-- (gelu/swish/sigmoid_back_bridge — a single multiply). Forward renders the
-- transcendental directly (`logistic`/`tanh`); the derivative is the
-- closed form matching the repo's `*ScalarDeriv = deriv …`. relu6 is the
-- two-sided clamp with mask `1[0<x<6]` (relu6_has_vjp_at). Length `m`. -/
-- ════════════════════════════════════════════════════════════════

/-- sigmoid: σ = logistic; σ' = σ(1−σ). -/
def sigmoidFwdM (m : Nat) : String :=
  actMod "sigmoid_fwd" s!"%x: {tt [m]}" (tt [m])
    (s!"    %s = stablehlo.logistic %x : {tt [m]}\n    return %s : {tt [m]}\n")
def sigmoidBackM (m : Nat) : String :=
  actMod "sigmoid_back" s!"%x: {tt [m]}, %dy: {tt [m]}" (tt [m])
    (s!"    %s = stablehlo.logistic %x : {tt [m]}\n" ++
     s!"    %one = stablehlo.constant dense<1.0> : {tt [m]}\n" ++
     s!"    %om = stablehlo.subtract %one, %s : {tt [m]}\n" ++
     s!"    %sp = stablehlo.multiply %s, %om : {tt [m]}\n" ++
     s!"    %dx = stablehlo.multiply %dy, %sp : {tt [m]}\n    return %dx : {tt [m]}\n")

/-- swish: y = x·σ(x); swish' = σ·(1 + x·(1−σ)). -/
def swishFwdM (m : Nat) : String :=
  actMod "swish_fwd" s!"%x: {tt [m]}" (tt [m])
    (s!"    %s = stablehlo.logistic %x : {tt [m]}\n    %y = stablehlo.multiply %x, %s : {tt [m]}\n    return %y : {tt [m]}\n")
def swishBackM (m : Nat) : String :=
  actMod "swish_back" s!"%x: {tt [m]}, %dy: {tt [m]}" (tt [m])
    (s!"    %s = stablehlo.logistic %x : {tt [m]}\n" ++
     s!"    %one = stablehlo.constant dense<1.0> : {tt [m]}\n" ++
     s!"    %om = stablehlo.subtract %one, %s : {tt [m]}\n" ++
     s!"    %xom = stablehlo.multiply %x, %om : {tt [m]}\n" ++
     s!"    %inner = stablehlo.add %one, %xom : {tt [m]}\n" ++
     s!"    %sp = stablehlo.multiply %s, %inner : {tt [m]}\n" ++
     s!"    %dx = stablehlo.multiply %dy, %sp : {tt [m]}\n    return %dx : {tt [m]}\n")

/-- relu6: clamp(x,0,6); deriv = 1[0<x<6]. -/
def relu6FwdM (m : Nat) : String :=
  actMod "relu6_fwd" s!"%x: {tt [m]}" (tt [m])
    (s!"    %z = stablehlo.constant dense<0.0> : {tt [m]}\n" ++
     s!"    %six = stablehlo.constant dense<6.0> : {tt [m]}\n" ++
     s!"    %m1 = stablehlo.maximum %x, %z : {tt [m]}\n" ++
     s!"    %y = stablehlo.minimum %m1, %six : {tt [m]}\n    return %y : {tt [m]}\n")
def relu6BackM (m : Nat) : String :=
  actMod "relu6_back" s!"%x: {tt [m]}, %dy: {tt [m]}" (tt [m])
    (s!"    %z = stablehlo.constant dense<0.0> : {tt [m]}\n" ++
     s!"    %six = stablehlo.constant dense<6.0> : {tt [m]}\n" ++
     s!"    %gt = stablehlo.compare GT, %x, %z : ({tt [m]}, {tt [m]}) -> {ti1 [m]}\n" ++
     s!"    %lt = stablehlo.compare LT, %x, %six : ({tt [m]}, {tt [m]}) -> {ti1 [m]}\n" ++
     s!"    %mask = stablehlo.and %gt, %lt : {ti1 [m]}\n" ++
     s!"    %dx = stablehlo.select %mask, %dy, %z : {ti1 [m]}, {tt [m]}\n    return %dx : {tt [m]}\n")

/-- gelu (tanh approx, c=√(2/π), a=0.044715): y = 0.5x(1+t), t=tanh(c(x+ax³));
    gelu' = 0.5(1+t) + 0.5x(1−t²)·c(1+3a·x²). -/
def geluFwdM (m : Nat) : String :=
  actMod "gelu_fwd" s!"%x: {tt [m]}" (tt [m])
    (s!"    %a = stablehlo.constant dense<0.044715> : {tt [m]}\n" ++
     s!"    %c = stablehlo.constant dense<0.7978845608> : {tt [m]}\n" ++
     s!"    %half = stablehlo.constant dense<0.5> : {tt [m]}\n" ++
     s!"    %one = stablehlo.constant dense<1.0> : {tt [m]}\n" ++
     s!"    %x2 = stablehlo.multiply %x, %x : {tt [m]}\n" ++
     s!"    %x3 = stablehlo.multiply %x2, %x : {tt [m]}\n" ++
     s!"    %ax3 = stablehlo.multiply %a, %x3 : {tt [m]}\n" ++
     s!"    %inn = stablehlo.add %x, %ax3 : {tt [m]}\n" ++
     s!"    %u = stablehlo.multiply %c, %inn : {tt [m]}\n" ++
     s!"    %t = stablehlo.tanh %u : {tt [m]}\n" ++
     s!"    %ot = stablehlo.add %one, %t : {tt [m]}\n" ++
     s!"    %hx = stablehlo.multiply %half, %x : {tt [m]}\n" ++
     s!"    %y = stablehlo.multiply %hx, %ot : {tt [m]}\n    return %y : {tt [m]}\n")
def geluBackM (m : Nat) : String :=
  actMod "gelu_back" s!"%x: {tt [m]}, %dy: {tt [m]}" (tt [m])
    (s!"    %a = stablehlo.constant dense<0.044715> : {tt [m]}\n" ++
     s!"    %a3 = stablehlo.constant dense<0.134145> : {tt [m]}\n" ++
     s!"    %c = stablehlo.constant dense<0.7978845608> : {tt [m]}\n" ++
     s!"    %half = stablehlo.constant dense<0.5> : {tt [m]}\n" ++
     s!"    %one = stablehlo.constant dense<1.0> : {tt [m]}\n" ++
     s!"    %x2 = stablehlo.multiply %x, %x : {tt [m]}\n" ++
     s!"    %x3 = stablehlo.multiply %x2, %x : {tt [m]}\n" ++
     s!"    %ax3 = stablehlo.multiply %a, %x3 : {tt [m]}\n" ++
     s!"    %inn = stablehlo.add %x, %ax3 : {tt [m]}\n" ++
     s!"    %u = stablehlo.multiply %c, %inn : {tt [m]}\n" ++
     s!"    %t = stablehlo.tanh %u : {tt [m]}\n" ++
     s!"    %ot = stablehlo.add %one, %t : {tt [m]}\n" ++
     s!"    %ta = stablehlo.multiply %half, %ot : {tt [m]}\n" ++
     s!"    %t2 = stablehlo.multiply %t, %t : {tt [m]}\n" ++
     s!"    %omt2 = stablehlo.subtract %one, %t2 : {tt [m]}\n" ++
     s!"    %hx = stablehlo.multiply %half, %x : {tt [m]}\n" ++
     s!"    %hxo = stablehlo.multiply %hx, %omt2 : {tt [m]}\n" ++
     s!"    %a3x2 = stablehlo.multiply %a3, %x2 : {tt [m]}\n" ++
     s!"    %in2 = stablehlo.add %one, %a3x2 : {tt [m]}\n" ++
     s!"    %cin2 = stablehlo.multiply %c, %in2 : {tt [m]}\n" ++
     s!"    %tb = stablehlo.multiply %hxo, %cin2 : {tt [m]}\n" ++
     s!"    %gp = stablehlo.add %ta, %tb : {tt [m]}\n" ++
     s!"    %dx = stablehlo.multiply %dy, %gp : {tt [m]}\n    return %dx : {tt [m]}\n")

-- ════════════════════════════════════════════════════════════════
-- § Residual + Squeeze-Excite (Phase 3 sweep) — the fan-in chapters
--
-- Residual `out = x + f(x)`: backward `dx = dy + f_back(dy)` — an `add`
-- fan-in (here f = dense n→n, so dx = dy + dy·Wᵀ). SE `out = x ⊙ gate(x)`:
-- backward (se_back_bridge) `dx = gate(x)⊙dy + gate_back(x⊙dy)` — `add`
-- fan-in of the gate output and the gate's own backward (here gate = σ, so
-- gate_back(v) = v⊙σ(1−σ)). Both show the chain rule composing through a
-- non-composition combinator (the fan-in `add`).
-- ════════════════════════════════════════════════════════════════

/-- Residual dense block forward `x + dense W b x`. -/
def residualFwdM (B n : Nat) : String :=
  actMod "residual_fwd" s!"%x: {tt [B,n]}, %W: {tt [n,n]}, %b: {tt [n]}" (tt [B,n])
    (matdg "%xw" "%x" "%W" "1" "0" (tt [B,n]) (tt [n,n]) (tt [B,n]) ++
     s!"    %bb = stablehlo.broadcast_in_dim %b, dims = [1] : ({tt [n]}) -> {tt [B,n]}\n" ++
     s!"    %h = stablehlo.add %xw, %bb : {tt [B,n]}\n" ++
     s!"    %out = stablehlo.add %x, %h : {tt [B,n]}\n    return %out : {tt [B,n]}\n")
/-- Residual backward `dx = dy + dy·Wᵀ` (the add fan-in: identity + dense). -/
def residualBackM (B n : Nat) : String :=
  actMod "residual_back" s!"%dy: {tt [B,n]}, %W: {tt [n,n]}" (tt [B,n])
    (matdg "%wd" "%dy" "%W" "1" "1" (tt [B,n]) (tt [n,n]) (tt [B,n]) ++
     s!"    %dx = stablehlo.add %dy, %wd : {tt [B,n]}\n    return %dx : {tt [B,n]}\n")

/-- SE block forward `x ⊙ σ(x)` (gate = sigmoid). -/
def seFwdM (m : Nat) : String :=
  actMod "se_fwd" s!"%x: {tt [m]}" (tt [m])
    (s!"    %g = stablehlo.logistic %x : {tt [m]}\n    %out = stablehlo.multiply %x, %g : {tt [m]}\n    return %out : {tt [m]}\n")
/-- SE backward (se_back_bridge): `dx = σ(x)⊙dy + (x⊙dy)⊙σ(x)(1−σ(x))`. -/
def seBackM (m : Nat) : String :=
  actMod "se_back" s!"%x: {tt [m]}, %dy: {tt [m]}" (tt [m])
    (s!"    %g = stablehlo.logistic %x : {tt [m]}\n" ++
     s!"    %one = stablehlo.constant dense<1.0> : {tt [m]}\n" ++
     s!"    %t1 = stablehlo.multiply %g, %dy : {tt [m]}\n" ++
     s!"    %xdy = stablehlo.multiply %x, %dy : {tt [m]}\n" ++
     s!"    %om = stablehlo.subtract %one, %g : {tt [m]}\n" ++
     s!"    %sp = stablehlo.multiply %g, %om : {tt [m]}\n" ++
     s!"    %t2 = stablehlo.multiply %xdy, %sp : {tt [m]}\n" ++
     s!"    %dx = stablehlo.add %t1, %t2 : {tt [m]}\n    return %dx : {tt [m]}\n")

-- ════════════════════════════════════════════════════════════════
-- § ViT transformer block — the apex assembly (Phase 3, whole-net)
--
-- `transformerBlock = MlpSublayer ∘ AttnSublayer`, each a residual:
--   AttnSublayer x = x + Wo·SDPA(LN₁(x)·{Wq,Wk,Wv})   (+biases)
--   MlpSublayer  y = y + Wfc2·gelu(Wfc1·LN₂(y))        (+biases)
-- The full block composes EVERY rendered op: per-token LayerNorm, dense
-- projections, scaled dot-product attention (with the proven dQ/dK/dV and
-- the three-way Q/K/V fan-in at the input), gelu, residual add. The backward
-- (input gradient `dx`) chains them via the proven bridges — the codegen
-- analogue of `transformerBlock_has_vjp_mat`. Single head (heads=1,
-- d_head=D); `N` tokens, model dim `D`, MLP hidden `F`, scale `1/√D`.
-- ════════════════════════════════════════════════════════════════

/-- Per-token LayerNorm forward into `out` (γ,β scalar SSA names); leaves
    `{out}_xhat` and `{out}_istd` in scope for the backward. `%sc` (f32 0),
    `%nf` (= n splat) and `%eps` must be in scope. -/
def renderLN (out x g b : String) (B n : Nat) : String :=
  reduceSumBcast s!"{out}_sm" x B n ++
  s!"    {out}_mu = stablehlo.divide {out}_sm, %nf : {tt [B,n]}\n" ++
  s!"    {out}_xc = stablehlo.subtract {x}, {out}_mu : {tt [B,n]}\n" ++
  s!"    {out}_sq = stablehlo.multiply {out}_xc, {out}_xc : {tt [B,n]}\n" ++
  reduceSumBcast s!"{out}_vs" s!"{out}_sq" B n ++
  s!"    {out}_var = stablehlo.divide {out}_vs, %nf : {tt [B,n]}\n" ++
  s!"    {out}_ve = stablehlo.add {out}_var, %eps : {tt [B,n]}\n" ++
  s!"    {out}_istd = stablehlo.rsqrt {out}_ve : {tt [B,n]}\n" ++
  s!"    {out}_xhat = stablehlo.multiply {out}_xc, {out}_istd : {tt [B,n]}\n" ++
  s!"    {out}_gb = stablehlo.broadcast_in_dim {g}, dims = [] : (tensor<f32>) -> {tt [B,n]}\n" ++
  s!"    {out}_bb = stablehlo.broadcast_in_dim {b}, dims = [] : (tensor<f32>) -> {tt [B,n]}\n" ++
  s!"    {out}_gx = stablehlo.multiply {out}_xhat, {out}_gb : {tt [B,n]}\n" ++
  s!"    {out} = stablehlo.add {out}_gx, {out}_bb : {tt [B,n]}\n"

/-- LayerNorm backward `dx` from cotangent `dy`, reusing `{ln}_xhat`/`{ln}_istd`
    + scalar γ `g`. (The proven 3-term form.) `%nf` in scope. -/
def renderLNBack (out ln g dy : String) (B n : Nat) : String :=
  s!"    {out}_gb = stablehlo.broadcast_in_dim {g}, dims = [] : (tensor<f32>) -> {tt [B,n]}\n" ++
  s!"    {out}_dxh = stablehlo.multiply {out}_gb, {dy} : {tt [B,n]}\n" ++
  reduceSumBcast s!"{out}_sdx" s!"{out}_dxh" B n ++
  s!"    {out}_xd = stablehlo.multiply {ln}_xhat, {out}_dxh : {tt [B,n]}\n" ++
  reduceSumBcast s!"{out}_sxd" s!"{out}_xd" B n ++
  s!"    {out}_t1 = stablehlo.multiply {out}_dxh, %nf : {tt [B,n]}\n" ++
  s!"    {out}_i1 = stablehlo.subtract {out}_t1, {out}_sdx : {tt [B,n]}\n" ++
  s!"    {out}_xs = stablehlo.multiply {ln}_xhat, {out}_sxd : {tt [B,n]}\n" ++
  s!"    {out}_i2 = stablehlo.subtract {out}_i1, {out}_xs : {tt [B,n]}\n" ++
  s!"    {out}_s = stablehlo.divide {ln}_istd, %nf : {tt [B,n]}\n" ++
  s!"    {out} = stablehlo.multiply {out}_s, {out}_i2 : {tt [B,n]}\n"

/-- Dense `out = in·W + b` ([B,m]·[m,n]+[n]) into `{out}`. -/
def renderDense (out inp W b : String) (B m n : Nat) : String :=
  matdg s!"{out}_w" inp W "1" "0" (tt [B,m]) (tt [m,n]) (tt [B,n]) ++
  s!"    {out}_bb = stablehlo.broadcast_in_dim {b}, dims = [1] : ({tt [n]}) -> {tt [B,n]}\n" ++
  s!"    {out} = stablehlo.add {out}_w, {out}_bb : {tt [B,n]}\n"

/-- GELU (tanh approx) forward into `{out}`; leaves `{out}_t/_ot/_x2/_hx` for
    the backward. gelu constants `%ga,%gc,%ghalf,%gone,%ga3` ([B,n]) in scope. -/
def renderGeluF (out x : String) (B n : Nat) : String :=
  s!"    {out}_x2 = stablehlo.multiply {x}, {x} : {tt [B,n]}\n" ++
  s!"    {out}_x3 = stablehlo.multiply {out}_x2, {x} : {tt [B,n]}\n" ++
  s!"    {out}_ax3 = stablehlo.multiply %ga, {out}_x3 : {tt [B,n]}\n" ++
  s!"    {out}_inn = stablehlo.add {x}, {out}_ax3 : {tt [B,n]}\n" ++
  s!"    {out}_u = stablehlo.multiply %gc, {out}_inn : {tt [B,n]}\n" ++
  s!"    {out}_t = stablehlo.tanh {out}_u : {tt [B,n]}\n" ++
  s!"    {out}_ot = stablehlo.add %gone, {out}_t : {tt [B,n]}\n" ++
  s!"    {out}_hx = stablehlo.multiply %ghalf, {x} : {tt [B,n]}\n" ++
  s!"    {out} = stablehlo.multiply {out}_hx, {out}_ot : {tt [B,n]}\n"

/-- GELU backward `{out} = dy ⊙ gelu'(x)`, reusing the forward `{g}_*`. -/
def renderGeluB (out g dy : String) (B n : Nat) : String :=
  s!"    {out}_ta = stablehlo.multiply %ghalf, {g}_ot : {tt [B,n]}\n" ++
  s!"    {out}_t2 = stablehlo.multiply {g}_t, {g}_t : {tt [B,n]}\n" ++
  s!"    {out}_omt2 = stablehlo.subtract %gone, {out}_t2 : {tt [B,n]}\n" ++
  s!"    {out}_hxo = stablehlo.multiply {g}_hx, {out}_omt2 : {tt [B,n]}\n" ++
  s!"    {out}_a3x2 = stablehlo.multiply %ga3, {g}_x2 : {tt [B,n]}\n" ++
  s!"    {out}_in2 = stablehlo.add %gone, {out}_a3x2 : {tt [B,n]}\n" ++
  s!"    {out}_cin2 = stablehlo.multiply %gc, {out}_in2 : {tt [B,n]}\n" ++
  s!"    {out}_tb = stablehlo.multiply {out}_hxo, {out}_cin2 : {tt [B,n]}\n" ++
  s!"    {out}_gp = stablehlo.add {out}_ta, {out}_tb : {tt [B,n]}\n" ++
  s!"    {out} = stablehlo.multiply {dy}, {out}_gp : {tt [B,n]}\n"

/-- The gelu tanh-approx constants as `[B,n]` splats. -/
def geluConsts (B n : Nat) : String :=
  s!"    %ga = stablehlo.constant dense<0.044715> : {tt [B,n]}\n" ++
  s!"    %ga3 = stablehlo.constant dense<0.134145> : {tt [B,n]}\n" ++
  s!"    %gc = stablehlo.constant dense<0.7978845608> : {tt [B,n]}\n" ++
  s!"    %ghalf = stablehlo.constant dense<0.5> : {tt [B,n]}\n" ++
  s!"    %gone = stablehlo.constant dense<1.0> : {tt [B,n]}\n"

/-- ViT **attention sublayer** forward `x + Wo·SDPA(LN(x)·{Wq,Wk,Wv})`
    (single head, +biases). `@attn_fwd`. -/
def attnSublayerFwdModule (N D : Nat) (eps scale : String) : String :=
  actMod "attn_fwd"
    (s!"%x: {tt [N,D]}, %Wq: {tt [D,D]}, %Wk: {tt [D,D]}, %Wv: {tt [D,D]}, %Wo: {tt [D,D]}, " ++
     s!"%bq: {tt [D]}, %bk: {tt [D]}, %bv: {tt [D]}, %bo: {tt [D]}, %g1: tensor<f32>, %b1: tensor<f32>")
    (tt [N,D])
    ("    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
     s!"    %nf = stablehlo.constant dense<{D}.0> : {tt [N,D]}\n" ++
     s!"    %eps = stablehlo.constant dense<{eps}> : {tt [N,D]}\n" ++
     s!"    %scl = stablehlo.constant dense<{scale}> : {tt [N,N]}\n" ++
     renderLN "%a" "%x" "%g1" "%b1" N D ++
     renderDense "%Q" "%a" "%Wq" "%bq" N D D ++
     renderDense "%K" "%a" "%Wk" "%bk" N D D ++
     renderDense "%V" "%a" "%Wv" "%bv" N D D ++
     matdg "%scores" "%Q" "%K" "1" "1" (tt [N,D]) (tt [N,D]) (tt [N,N]) ++
     s!"    %scaled = stablehlo.multiply %scores, %scl : {tt [N,N]}\n" ++
     renderSoftmax "%w" "%scaled" N N ++
     matdg "%attn" "%w" "%V" "1" "0" (tt [N,N]) (tt [N,D]) (tt [N,D]) ++
     renderDense "%o" "%attn" "%Wo" "%bo" N D D ++
     s!"    %out = stablehlo.add %x, %o : {tt [N,D]}\n    return %out : {tt [N,D]}\n")

/-- ViT attention sublayer **input-gradient backward** `dx` (chains LN-back,
    the three-way Q/K/V fan-in, SDPA dQ/dK/dV, residual). `@attn_back`. -/
def attnSublayerBackModule (N D : Nat) (eps scale : String) : String :=
  actMod "attn_back"
    (s!"%x: {tt [N,D]}, %Wq: {tt [D,D]}, %Wk: {tt [D,D]}, %Wv: {tt [D,D]}, %Wo: {tt [D,D]}, " ++
     s!"%bq: {tt [D]}, %bk: {tt [D]}, %bv: {tt [D]}, %bo: {tt [D]}, %g1: tensor<f32>, %b1: tensor<f32>, %dOut: {tt [N,D]}")
    (tt [N,D])
    ("    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
     s!"    %nf = stablehlo.constant dense<{D}.0> : {tt [N,D]}\n" ++
     s!"    %eps = stablehlo.constant dense<{eps}> : {tt [N,D]}\n" ++
     s!"    %scl = stablehlo.constant dense<{scale}> : {tt [N,N]}\n" ++
     -- recompute forward intermediates: a, Q, K, V, weights
     renderLN "%a" "%x" "%g1" "%b1" N D ++
     renderDense "%Q" "%a" "%Wq" "%bq" N D D ++
     renderDense "%K" "%a" "%Wk" "%bk" N D D ++
     renderDense "%V" "%a" "%Wv" "%bv" N D D ++
     matdg "%scores" "%Q" "%K" "1" "1" (tt [N,D]) (tt [N,D]) (tt [N,N]) ++
     s!"    %scaled = stablehlo.multiply %scores, %scl : {tt [N,N]}\n" ++
     renderSoftmax "%w" "%scaled" N N ++
     -- backward: o = attn·Wo ⇒ dattn = dOut·Woᵀ
     matdg "%dattn" "%dOut" "%Wo" "1" "1" (tt [N,D]) (tt [D,D]) (tt [N,D]) ++
     -- SDPA backward (dQ,dK,dV)
     matdg "%dWeights" "%dattn" "%V" "1" "1" (tt [N,D]) (tt [N,D]) (tt [N,N]) ++
     matdg "%dV" "%w" "%dattn" "0" "0" (tt [N,N]) (tt [N,D]) (tt [N,D]) ++
     s!"    %pdw = stablehlo.multiply %w, %dWeights : {tt [N,N]}\n" ++
     reduceSumBcast "%srow" "%pdw" N N ++
     s!"    %diff = stablehlo.subtract %dWeights, %srow : {tt [N,N]}\n" ++
     s!"    %dScaled = stablehlo.multiply %w, %diff : {tt [N,N]}\n" ++
     s!"    %dScores = stablehlo.multiply %dScaled, %scl : {tt [N,N]}\n" ++
     matdg "%dQ" "%dScores" "%K" "1" "0" (tt [N,N]) (tt [N,D]) (tt [N,D]) ++
     matdg "%dK" "%dScores" "%Q" "0" "0" (tt [N,N]) (tt [N,D]) (tt [N,D]) ++
     -- projections back ⇒ three-way fan-in da = dQ·Wqᵀ + dK·Wkᵀ + dV·Wvᵀ
     matdg "%daQ" "%dQ" "%Wq" "1" "1" (tt [N,D]) (tt [D,D]) (tt [N,D]) ++
     matdg "%daK" "%dK" "%Wk" "1" "1" (tt [N,D]) (tt [D,D]) (tt [N,D]) ++
     matdg "%daV" "%dV" "%Wv" "1" "1" (tt [N,D]) (tt [D,D]) (tt [N,D]) ++
     s!"    %daQK = stablehlo.add %daQ, %daK : {tt [N,D]}\n" ++
     s!"    %da = stablehlo.add %daQK, %daV : {tt [N,D]}\n" ++
     -- LN backward + residual fan-in
     renderLNBack "%dxa" "%a" "%g1" "%da" N D ++
     s!"    %dx = stablehlo.add %dOut, %dxa : {tt [N,D]}\n    return %dx : {tt [N,D]}\n")

/-- The full transformer-block parameter signature (shared by fwd/back). -/
def vitBlockSig (N D F : Nat) : String :=
  s!"%x: {tt [N,D]}, %Wq: {tt [D,D]}, %Wk: {tt [D,D]}, %Wv: {tt [D,D]}, %Wo: {tt [D,D]}, " ++
  s!"%bq: {tt [D]}, %bk: {tt [D]}, %bv: {tt [D]}, %bo: {tt [D]}, %g1: tensor<f32>, %b1: tensor<f32>, " ++
  s!"%g2: tensor<f32>, %b2: tensor<f32>, %Wfc1: {tt [D,F]}, %bfc1: {tt [F]}, %Wfc2: {tt [F,D]}, %bfc2: {tt [D]}"

/-- Shared block constants + the attention sublayer forward through `%x1`,
    and (for back) `%c,%h1,%hg` of the MLP sublayer. -/
def vitBlockFwdBody (N D F : Nat) (eps scale : String) : String :=
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %nf = stablehlo.constant dense<{D}.0> : {tt [N,D]}\n" ++
  s!"    %eps = stablehlo.constant dense<{eps}> : {tt [N,D]}\n" ++
  s!"    %scl = stablehlo.constant dense<{scale}> : {tt [N,N]}\n" ++
  geluConsts N F ++
  renderLN "%a" "%x" "%g1" "%b1" N D ++
  renderDense "%Q" "%a" "%Wq" "%bq" N D D ++
  renderDense "%K" "%a" "%Wk" "%bk" N D D ++
  renderDense "%V" "%a" "%Wv" "%bv" N D D ++
  matdg "%scores" "%Q" "%K" "1" "1" (tt [N,D]) (tt [N,D]) (tt [N,N]) ++
  s!"    %scaled = stablehlo.multiply %scores, %scl : {tt [N,N]}\n" ++
  renderSoftmax "%w" "%scaled" N N ++
  matdg "%attn" "%w" "%V" "1" "0" (tt [N,N]) (tt [N,D]) (tt [N,D]) ++
  renderDense "%o" "%attn" "%Wo" "%bo" N D D ++
  s!"    %x1 = stablehlo.add %x, %o : {tt [N,D]}\n" ++
  renderLN "%c" "%x1" "%g2" "%b2" N D ++
  renderDense "%h1" "%c" "%Wfc1" "%bfc1" N D F ++
  renderGeluF "%hg" "%h1" N F

/-- Full ViT **transformer block** forward `MlpSublayer ∘ AttnSublayer`. -/
def vitBlockFwdModule (N D F : Nat) (eps scale : String) : String :=
  actMod "vit_fwd" (vitBlockSig N D F) (tt [N,D])
    (vitBlockFwdBody N D F eps scale ++
     renderDense "%m" "%hg" "%Wfc2" "%bfc2" N F D ++
     s!"    %out = stablehlo.add %x1, %m : {tt [N,D]}\n    return %out : {tt [N,D]}\n")

/-- Full ViT transformer block **input-gradient backward** `dx`: MLP-sublayer
    back (gelu) then attention-sublayer back (3-way QKV fan-in), each a residual
    fan-in. The codegen analogue of `transformerBlock_has_vjp_mat`. -/
def vitBlockBackModule (N D F : Nat) (eps scale : String) : String :=
  actMod "vit_back" (vitBlockSig N D F ++ s!", %dOut: {tt [N,D]}") (tt [N,D])
    (vitBlockFwdBody N D F eps scale ++
     -- MLP sublayer backward
     matdg "%dhg" "%dOut" "%Wfc2" "1" "1" (tt [N,D]) (tt [F,D]) (tt [N,F]) ++
     renderGeluB "%dh1" "%hg" "%dhg" N F ++
     matdg "%dc" "%dh1" "%Wfc1" "1" "1" (tt [N,F]) (tt [D,F]) (tt [N,D]) ++
     renderLNBack "%dx1m" "%c" "%g2" "%dc" N D ++
     s!"    %dx1 = stablehlo.add %dOut, %dx1m : {tt [N,D]}\n" ++
     -- attention sublayer backward (cotangent %dx1)
     matdg "%dattn" "%dx1" "%Wo" "1" "1" (tt [N,D]) (tt [D,D]) (tt [N,D]) ++
     matdg "%dWeights" "%dattn" "%V" "1" "1" (tt [N,D]) (tt [N,D]) (tt [N,N]) ++
     matdg "%dV" "%w" "%dattn" "0" "0" (tt [N,N]) (tt [N,D]) (tt [N,D]) ++
     s!"    %pdw = stablehlo.multiply %w, %dWeights : {tt [N,N]}\n" ++
     reduceSumBcast "%srow" "%pdw" N N ++
     s!"    %diff = stablehlo.subtract %dWeights, %srow : {tt [N,N]}\n" ++
     s!"    %dScaled = stablehlo.multiply %w, %diff : {tt [N,N]}\n" ++
     s!"    %dScores = stablehlo.multiply %dScaled, %scl : {tt [N,N]}\n" ++
     matdg "%dQ" "%dScores" "%K" "1" "0" (tt [N,N]) (tt [N,D]) (tt [N,D]) ++
     matdg "%dK" "%dScores" "%Q" "0" "0" (tt [N,N]) (tt [N,D]) (tt [N,D]) ++
     matdg "%daQ" "%dQ" "%Wq" "1" "1" (tt [N,D]) (tt [D,D]) (tt [N,D]) ++
     matdg "%daK" "%dK" "%Wk" "1" "1" (tt [N,D]) (tt [D,D]) (tt [N,D]) ++
     matdg "%daV" "%dV" "%Wv" "1" "1" (tt [N,D]) (tt [D,D]) (tt [N,D]) ++
     s!"    %daQK = stablehlo.add %daQ, %daK : {tt [N,D]}\n" ++
     s!"    %da = stablehlo.add %daQK, %daV : {tt [N,D]}\n" ++
     renderLNBack "%dxa" "%a" "%g1" "%da" N D ++
     s!"    %dx = stablehlo.add %dx1, %dxa : {tt [N,D]}\n    return %dx : {tt [N,D]}\n")

-- ════════════════════════════════════════════════════════════════
-- § ResNet — the whole-network test case (Phase 3, deep assembly)
--
-- The repo's proven ResNet-style net (\texttt{cnn\_has\_vjp\_at}) is
--   dense ∘ globalAvgPool ∘ rblkP ∘ rblk ∘ maxPool ∘ cbr(stem),
-- where its "BN" is `bnForward` over the *flattened* feature vector --- i.e.
-- the per-token LN renderer (renderLN), reused at n = C·H·W. The basic block
-- is  out = relu( BN₂(conv₂(relu(BN₁(conv₁(x))))) + skip(x) )  (residual_has_vjp).
-- Convs have no bias (BN absorbs it — standard ResNet). We keep tensors NCHW
-- and reshape to [1,C·H·W] only across each BN. (Input gradient dx; the
-- generator below stacks these blocks into a full ResNet tower.)
-- ════════════════════════════════════════════════════════════════

/-- One conv→BN→relu→conv→BN forward residual function `F`, into `{p}out`
    (plus `{p}c1/{p}n1f/{p}c2/{p}n2` saved for the backward). NCHW;
    `%sc,%nf,%eps,%zc` in scope; reshapes across BN to `[1,M]`, `M=C·H·W`. -/
def renderResF (p x W1 W2 g1 b1 g2 b2 : String) (C H W kH kW : Nat) : String :=
  let pH := (kH-1)/2; let pW := (kW-1)/2; let M := C*H*W
  convOp s!"{p}c1" x W1 (tt [1,C,H,W]) (tt [C,C,kH,kW]) (tt [1,C,H,W]) pH pW ++
  s!"    {p}c1f = stablehlo.reshape {p}c1 : ({tt [1,C,H,W]}) -> {tt [1,M]}\n" ++
  renderLN s!"{p}n1f" s!"{p}c1f" g1 b1 1 M ++
  s!"    {p}n1 = stablehlo.reshape {p}n1f : ({tt [1,M]}) -> {tt [1,C,H,W]}\n" ++
  s!"    {p}r1 = stablehlo.maximum {p}n1, %zc : {tt [1,C,H,W]}\n" ++
  convOp s!"{p}c2" s!"{p}r1" W2 (tt [1,C,H,W]) (tt [C,C,kH,kW]) (tt [1,C,H,W]) pH pW ++
  s!"    {p}c2f = stablehlo.reshape {p}c2 : ({tt [1,C,H,W]}) -> {tt [1,M]}\n" ++
  renderLN s!"{p}n2f" s!"{p}c2f" g2 b2 1 M ++
  s!"    {p}out = stablehlo.reshape {p}n2f : ({tt [1,M]}) -> {tt [1,C,H,W]}\n"

/-- Backward of `renderResF` from cotangent `{p}dn2` ([1,C,H,W]) to `{p}dF`
    (the dx contribution of `F`), reusing the saved forward `{p}…`. -/
def renderResFBack (p W1 W2 g1 g2 dn2 : String) (C H W kH kW : Nat) : String :=
  let pH := (kH-1)/2; let pW := (kW-1)/2; let M := C*H*W
  s!"    {p}dn2f = stablehlo.reshape {dn2} : ({tt [1,C,H,W]}) -> {tt [1,M]}\n" ++
  renderLNBack s!"{p}dc2f" s!"{p}n2f" g2 s!"{p}dn2f" 1 M ++
  s!"    {p}dc2 = stablehlo.reshape {p}dc2f : ({tt [1,M]}) -> {tt [1,C,H,W]}\n" ++
  s!"    {p}W2t = stablehlo.transpose {W2}, dims = [1, 0, 2, 3] : ({tt [C,C,kH,kW]}) -> {tt [C,C,kH,kW]}\n" ++
  s!"    {p}W2r = stablehlo.reverse {p}W2t, dims = [2, 3] : {tt [C,C,kH,kW]}\n" ++
  convOp s!"{p}dr1" s!"{p}dc2" s!"{p}W2r" (tt [1,C,H,W]) (tt [C,C,kH,kW]) (tt [1,C,H,W]) pH pW ++
  s!"    {p}mn1 = stablehlo.compare GT, {p}n1, %zc : ({tt [1,C,H,W]}, {tt [1,C,H,W]}) -> {ti1 [1,C,H,W]}\n" ++
  s!"    {p}dn1 = stablehlo.select {p}mn1, {p}dr1, %zc : {ti1 [1,C,H,W]}, {tt [1,C,H,W]}\n" ++
  s!"    {p}dn1f = stablehlo.reshape {p}dn1 : ({tt [1,C,H,W]}) -> {tt [1,M]}\n" ++
  renderLNBack s!"{p}dc1f" s!"{p}n1f" g1 s!"{p}dn1f" 1 M ++
  s!"    {p}dc1 = stablehlo.reshape {p}dc1f : ({tt [1,M]}) -> {tt [1,C,H,W]}\n" ++
  s!"    {p}W1t = stablehlo.transpose {W1}, dims = [1, 0, 2, 3] : ({tt [C,C,kH,kW]}) -> {tt [C,C,kH,kW]}\n" ++
  s!"    {p}W1r = stablehlo.reverse {p}W1t, dims = [2, 3] : {tt [C,C,kH,kW]}\n" ++
  convOp s!"{p}dF" s!"{p}dc1" s!"{p}W1r" (tt [1,C,H,W]) (tt [C,C,kH,kW]) (tt [1,C,H,W]) pH pW

/-- Identity residual block forward `relu(F(x) + x)` as `@res_fwd`. -/
def resBlockFwdModule (C H W kH kW : Nat) (eps : String) : String :=
  let M := C*H*W
  actMod "res_fwd"
    (s!"%x: {tt [1,C,H,W]}, %W1: {tt [C,C,kH,kW]}, %W2: {tt [C,C,kH,kW]}, " ++
     s!"%g1: tensor<f32>, %b1: tensor<f32>, %g2: tensor<f32>, %b2: tensor<f32>")
    (tt [1,C,H,W])
    ("    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
     s!"    %nf = stablehlo.constant dense<{M}.0> : {tt [1,M]}\n" ++
     s!"    %eps = stablehlo.constant dense<{eps}> : {tt [1,M]}\n" ++
     s!"    %zc = stablehlo.constant dense<0.0> : {tt [1,C,H,W]}\n" ++
     renderResF "%" "%x" "%W1" "%W2" "%g1" "%b1" "%g2" "%b2" C H W kH kW ++
     s!"    %add = stablehlo.add %out, %x : {tt [1,C,H,W]}\n" ++
     s!"    %y = stablehlo.maximum %add, %zc : {tt [1,C,H,W]}\n    return %y : {tt [1,C,H,W]}\n")

/-- Identity residual block input-gradient backward `dx = F.back(dadd) + dadd`
    (the add fan-in), `dadd = dOut ⊙ relu'(F(x)+x)`. As `@res_back`. -/
def resBlockBackModule (C H W kH kW : Nat) (eps : String) : String :=
  let M := C*H*W
  actMod "res_back"
    (s!"%x: {tt [1,C,H,W]}, %W1: {tt [C,C,kH,kW]}, %W2: {tt [C,C,kH,kW]}, " ++
     s!"%g1: tensor<f32>, %b1: tensor<f32>, %g2: tensor<f32>, %b2: tensor<f32>, %dOut: {tt [1,C,H,W]}")
    (tt [1,C,H,W])
    ("    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
     s!"    %nf = stablehlo.constant dense<{M}.0> : {tt [1,M]}\n" ++
     s!"    %eps = stablehlo.constant dense<{eps}> : {tt [1,M]}\n" ++
     s!"    %zc = stablehlo.constant dense<0.0> : {tt [1,C,H,W]}\n" ++
     renderResF "%" "%x" "%W1" "%W2" "%g1" "%b1" "%g2" "%b2" C H W kH kW ++
     s!"    %add = stablehlo.add %out, %x : {tt [1,C,H,W]}\n" ++
     s!"    %madd = stablehlo.compare GT, %add, %zc : ({tt [1,C,H,W]}, {tt [1,C,H,W]}) -> {ti1 [1,C,H,W]}\n" ++
     s!"    %dadd = stablehlo.select %madd, %dOut, %zc : {ti1 [1,C,H,W]}, {tt [1,C,H,W]}\n" ++
     renderResFBack "%" "%W1" "%W2" "%g1" "%g2" "%dadd" C H W kH kW ++
     s!"    %dx = stablehlo.add %dF, %dadd : {tt [1,C,H,W]}\n    return %dx : {tt [1,C,H,W]}\n")

-- ════════════════════════════════════════════════════════════════
-- § ResNet tower generator — a WHOLE deep net from a loop (feasibility)
--
-- The harness, ResNet-specialized: a Lean fold stacks `k` residual blocks,
-- threading each block's output into the next, and emits the whole forward
-- then the whole input-gradient backward (reversed). ResNet-34 has exactly
-- 16 residual blocks (3+4+6+3), so `k = 16` matches its residual depth. This
-- is the "declare the structure, generate the net" move: depth is a loop
-- bound, not 16 hand-written blocks.
-- ════════════════════════════════════════════════════════════════

/-- Per-block parameter signature (6 params/block), prepended with `%x`. -/
def resTowerSig (k C H W kH kW : Nat) : String :=
  s!"%x: {tt [1,C,H,W]}" ++ String.join ((List.range k).map (fun i =>
    let p := s!"%b{i}"
    s!", {p}W1: {tt [C,C,kH,kW]}, {p}W2: {tt [C,C,kH,kW]}, {p}g1: tensor<f32>, " ++
    s!"{p}b1: tensor<f32>, {p}g2: tensor<f32>, {p}b2: tensor<f32>"))

/-- Forward fold: stack `k` identity residual blocks; returns (code, result SSA). -/
def resTowerFwdBody (k C H W kH kW : Nat) : String × String :=
  (List.range k).foldl (fun (acc : String × String) i =>
    let (code, xin) := acc
    let p := s!"%b{i}"
    (code ++ renderResF p xin s!"{p}W1" s!"{p}W2" s!"{p}g1" s!"{p}b1" s!"{p}g2" s!"{p}b2" C H W kH kW
       ++ s!"    {p}add = stablehlo.add {p}out, {xin} : {tt [1,C,H,W]}\n"
       ++ s!"    {p}y = stablehlo.maximum {p}add, %zc : {tt [1,C,H,W]}\n",
     s!"{p}y")) ("", "%x")

/-- Backward fold (reverse order): returns (code, dx SSA). -/
def resTowerBackBody (k C H W kH kW : Nat) : String × String :=
  ((List.range k).reverse).foldl (fun (acc : String × String) i =>
    let (code, dy) := acc
    let p := s!"%b{i}"
    (code
      ++ s!"    {p}madd = stablehlo.compare GT, {p}add, %zc : ({tt [1,C,H,W]}, {tt [1,C,H,W]}) -> {ti1 [1,C,H,W]}\n"
      ++ s!"    {p}dadd = stablehlo.select {p}madd, {dy}, %zc : {ti1 [1,C,H,W]}, {tt [1,C,H,W]}\n"
      ++ renderResFBack p s!"{p}W1" s!"{p}W2" s!"{p}g1" s!"{p}g2" s!"{p}dadd" C H W kH kW
      ++ s!"    {p}dx = stablehlo.add {p}dF, {p}dadd : {tt [1,C,H,W]}\n",
     s!"{p}dx")) ("", "%dOut")

/-- Shared constants for the tower. -/
def resTowerConsts (C H W : Nat) (eps : String) : String :=
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %nf = stablehlo.constant dense<{C*H*W}.0> : {tt [1,C*H*W]}\n" ++
  s!"    %eps = stablehlo.constant dense<{eps}> : {tt [1,C*H*W]}\n" ++
  s!"    %zc = stablehlo.constant dense<0.0> : {tt [1,C,H,W]}\n"

/-- A `k`-block ResNet residual tower, forward, as `@res_tower_fwd`. -/
def resTowerFwdModule (k C H W kH kW : Nat) (eps : String) : String :=
  let (body, res) := resTowerFwdBody k C H W kH kW
  actMod "res_tower_fwd" (resTowerSig k C H W kH kW) (tt [1,C,H,W])
    (resTowerConsts C H W eps ++ body ++ s!"    return {res} : {tt [1,C,H,W]}\n")

/-- The tower input gradient `dx`, as `@res_tower_back` (forward to save
    activations, then the reversed per-block backward). -/
def resTowerBackModule (k C H W kH kW : Nat) (eps : String) : String :=
  let (fwd, _) := resTowerFwdBody k C H W kH kW
  let (bwd, dx) := resTowerBackBody k C H W kH kW
  actMod "res_tower_back" (resTowerSig k C H W kH kW ++ s!", %dOut: {tt [1,C,H,W]}") (tt [1,C,H,W])
    (resTowerConsts C H W eps ++ fwd ++ bwd ++ s!"    return {dx} : {tt [1,C,H,W]}\n")

-- ════════════════════════════════════════════════════════════════
-- § ResNet train step — the full verified ResNet (Phase 3, capstone)
--
-- stem(cbr) → k residual blocks → global-avg-pool → FC → softmax-CE, with the
-- full backward computing dx AND every parameter gradient (conv weight grads
-- via the transpose trick, BN γ/β grads, FC W/b grads), then SGD. Every
-- gradient is a rendering of proof-backed IR; SGD arithmetic is the trusted
-- frame --- the same standard as the MLP and CNN train steps, now for a
-- residual net. (Constant width, no downsample, to keep dims uniform; the
-- dx-tower generator above shows depth scales to 16.)
-- ════════════════════════════════════════════════════════════════

/-- Global average pool `[1,C,H,W] → [1,C]`. `%sc` in scope. -/
def renderGAP (out x : String) (C H W : Nat) : String :=
  s!"    {out}_s = stablehlo.reduce({x} init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({tt [1,C,H,W]}, tensor<f32>) -> {tt [1,C]}\n" ++
  s!"    {out}_hw = stablehlo.constant dense<{H*W}.0> : {tt [1,C]}\n" ++
  s!"    {out} = stablehlo.divide {out}_s, {out}_hw : {tt [1,C]}\n"

/-- GAP backward: `dx[c,i,j] = dy[c]/(H·W)` broadcast over the spatial axes. -/
def renderGAPBack (out dy : String) (C H W : Nat) : String :=
  s!"    {out}_hw = stablehlo.constant dense<{H*W}.0> : {tt [1,C]}\n" ++
  s!"    {out}_d = stablehlo.divide {dy}, {out}_hw : {tt [1,C]}\n" ++
  s!"    {out} = stablehlo.broadcast_in_dim {out}_d, dims = [0, 1] : ({tt [1,C]}) -> {tt [1,C,H,W]}\n"

/-- Conv weight gradient via the transpose trick: input + output-cotangent → dW. -/
def renderConvWGrad (out inp grad : String) (ic oc H W kH kW : Nat) : String :=
  let pH := (kH-1)/2; let pW := (kW-1)/2
  s!"    {out}_xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({tt [1,ic,H,W]}) -> {tt [ic,1,H,W]}\n" ++
  s!"    {out}_dt = stablehlo.transpose {grad}, dims = [1, 0, 2, 3] : ({tt [1,oc,H,W]}) -> {tt [oc,1,H,W]}\n" ++
  convOp s!"{out}_raw" s!"{out}_xt" s!"{out}_dt" (tt [ic,1,H,W]) (tt [oc,1,H,W]) (tt [ic,oc,kH,kW]) pH pW ++
  s!"    {out} = stablehlo.transpose {out}_raw, dims = [1, 0, 2, 3] : ({tt [ic,oc,kH,kW]}) -> {tt [oc,ic,kH,kW]}\n"

/-- BN scalar γ,β grads: dγ = Σ(x̂⊙dy), dβ = Σ(dy) over the flattened map. -/
def renderBNParamGrad (dg db xhat dyf : String) (M : Nat) : String :=
  s!"    {dg}_p = stablehlo.multiply {xhat}, {dyf} : {tt [1,M]}\n" ++
  s!"    {dg} = stablehlo.reduce({dg}_p init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({tt [1,M]}, tensor<f32>) -> tensor<f32>\n" ++
  s!"    {db} = stablehlo.reduce({dyf} init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({tt [1,M]}, tensor<f32>) -> tensor<f32>\n"

/-- A full 2-block ResNet SGD train step (`@resnet_train_step`): stem(cbr) →
    2 residual blocks → GAP → FC → softmax-CE, backward with every parameter
    gradient, then SGD. Returns the 17 updated parameters. -/
def resnetTrainStepModule (C H W kH kW nCls : Nat) (eps lr : String) : String :=
  let M := C*H*W
  let sgd (θ dθ θ' lrC sg ty : String) : String :=
    s!"    {lrC} = stablehlo.constant dense<{lr}> : {ty}\n" ++
    s!"    {sg} = stablehlo.multiply {dθ}, {lrC} : {ty}\n" ++
    s!"    {θ'} = stablehlo.subtract {θ}, {sg} : {ty}\n"
  let ty4 := tt [C,C,kH,kW]; let f32 := "tensor<f32>"
  actMod "resnet_train_step"
    (s!"%x: {tt [1,C,H,W]}, %Ws: {tt [C,C,kH,kW]}, %gs: {f32}, %bs: {f32}, " ++
     s!"%W10: {ty4}, %W20: {ty4}, %g10: {f32}, %bb10: {f32}, %g20: {f32}, %bb20: {f32}, " ++
     s!"%W11: {ty4}, %W21: {ty4}, %g11: {f32}, %bb11: {f32}, %g21: {f32}, %bb21: {f32}, " ++
     s!"%Wd: {tt [C,nCls]}, %bd: {tt [nCls]}, %onehot: {tt [1,nCls]}")
    (s!"({tt [C,C,kH,kW]}, {f32}, {f32}, {ty4}, {ty4}, {f32}, {f32}, {f32}, {f32}, " ++
     s!"{ty4}, {ty4}, {f32}, {f32}, {f32}, {f32}, {tt [C,nCls]}, {tt [nCls]})")
    (-- consts
     "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
     s!"    %nf = stablehlo.constant dense<{M}.0> : {tt [1,M]}\n" ++
     s!"    %eps = stablehlo.constant dense<{eps}> : {tt [1,M]}\n" ++
     s!"    %zc = stablehlo.constant dense<0.0> : {tt [1,C,H,W]}\n" ++
     "    // ── forward: stem(cbr) → block0 → block1 → GAP → FC → logits ──\n" ++
     convOp "%sc1" "%x" "%Ws" (tt [1,C,H,W]) (tt [C,C,kH,kW]) (tt [1,C,H,W]) ((kH-1)/2) ((kW-1)/2) ++
     s!"    %sc1f = stablehlo.reshape %sc1 : ({tt [1,C,H,W]}) -> {tt [1,M]}\n" ++
     renderLN "%snf" "%sc1f" "%gs" "%bs" 1 M ++
     s!"    %sn = stablehlo.reshape %snf : ({tt [1,M]}) -> {tt [1,C,H,W]}\n" ++
     s!"    %sa = stablehlo.maximum %sn, %zc : {tt [1,C,H,W]}\n" ++
     renderResF "%B0" "%sa" "%W10" "%W20" "%g10" "%bb10" "%g20" "%bb20" C H W kH kW ++
     s!"    %B0add = stablehlo.add %B0out, %sa : {tt [1,C,H,W]}\n" ++
     s!"    %B0y = stablehlo.maximum %B0add, %zc : {tt [1,C,H,W]}\n" ++
     renderResF "%B1" "%B0y" "%W11" "%W21" "%g11" "%bb11" "%g21" "%bb21" C H W kH kW ++
     s!"    %B1add = stablehlo.add %B1out, %B0y : {tt [1,C,H,W]}\n" ++
     s!"    %B1y = stablehlo.maximum %B1add, %zc : {tt [1,C,H,W]}\n" ++
     renderGAP "%gap" "%B1y" C H W ++
     s!"    %xw = stablehlo.dot_general %gap, %Wd, contracting_dims = [1] x [0],\n" ++
     s!"              precision = [DEFAULT, DEFAULT] : ({tt [1,C]}, {tt [C,nCls]}) -> {tt [1,nCls]}\n" ++
     s!"    %bdb = stablehlo.broadcast_in_dim %bd, dims = [1] : ({tt [nCls]}) -> {tt [1,nCls]}\n" ++
     s!"    %logits = stablehlo.add %xw, %bdb : {tt [1,nCls]}\n" ++
     "    // ── loss + backward (PROOF-BACKED: dx + every param grad) ──\n" ++
     renderLossCot 1 nCls "%logits" "%onehot" "%dy" ++
     -- FC backward
     s!"    %dWd = stablehlo.dot_general %gap, %dy, contracting_dims = [0] x [0],\n" ++
     s!"              precision = [DEFAULT, DEFAULT] : ({tt [1,C]}, {tt [1,nCls]}) -> {tt [C,nCls]}\n" ++
     s!"    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : ({tt [1,nCls]}, tensor<f32>) -> {tt [nCls]}\n" ++
     s!"    %dgap = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1],\n" ++
     s!"              precision = [DEFAULT, DEFAULT] : ({tt [1,nCls]}, {tt [C,nCls]}) -> {tt [1,C]}\n" ++
     renderGAPBack "%dB1y" "%dgap" C H W ++
     -- block1 backward + param grads
     s!"    %B1madd = stablehlo.compare GT, %B1add, %zc : ({tt [1,C,H,W]}, {tt [1,C,H,W]}) -> {ti1 [1,C,H,W]}\n" ++
     s!"    %B1dadd = stablehlo.select %B1madd, %dB1y, %zc : {ti1 [1,C,H,W]}, {tt [1,C,H,W]}\n" ++
     renderResFBack "%B1" "%W11" "%W21" "%g11" "%g21" "%B1dadd" C H W kH kW ++
     s!"    %B1dx = stablehlo.add %B1dF, %B1dadd : {tt [1,C,H,W]}\n" ++
     renderConvWGrad "%dW11" "%B0y" "%B1dc1" C C H W kH kW ++
     renderConvWGrad "%dW21" "%B1r1" "%B1dc2" C C H W kH kW ++
     renderBNParamGrad "%dg11" "%dbb11" "%B1n1f_xhat" "%B1dn1f" M ++
     renderBNParamGrad "%dg21" "%dbb21" "%B1n2f_xhat" "%B1dn2f" M ++
     -- block0 backward + param grads
     s!"    %B0madd = stablehlo.compare GT, %B0add, %zc : ({tt [1,C,H,W]}, {tt [1,C,H,W]}) -> {ti1 [1,C,H,W]}\n" ++
     s!"    %B0dadd = stablehlo.select %B0madd, %B1dx, %zc : {ti1 [1,C,H,W]}, {tt [1,C,H,W]}\n" ++
     renderResFBack "%B0" "%W10" "%W20" "%g10" "%g20" "%B0dadd" C H W kH kW ++
     s!"    %B0dx = stablehlo.add %B0dF, %B0dadd : {tt [1,C,H,W]}\n" ++
     renderConvWGrad "%dW10" "%sa" "%B0dc1" C C H W kH kW ++
     renderConvWGrad "%dW20" "%B0r1" "%B0dc2" C C H W kH kW ++
     renderBNParamGrad "%dg10" "%dbb10" "%B0n1f_xhat" "%B0dn1f" M ++
     renderBNParamGrad "%dg20" "%dbb20" "%B0n2f_xhat" "%B0dn2f" M ++
     -- stem backward + param grads
     s!"    %smask = stablehlo.compare GT, %sn, %zc : ({tt [1,C,H,W]}, {tt [1,C,H,W]}) -> {ti1 [1,C,H,W]}\n" ++
     s!"    %dsn = stablehlo.select %smask, %B0dx, %zc : {ti1 [1,C,H,W]}, {tt [1,C,H,W]}\n" ++
     s!"    %dsnf = stablehlo.reshape %dsn : ({tt [1,C,H,W]}) -> {tt [1,M]}\n" ++
     renderLNBack "%dsc1f" "%snf" "%gs" "%dsnf" 1 M ++
     s!"    %dsc1 = stablehlo.reshape %dsc1f : ({tt [1,M]}) -> {tt [1,C,H,W]}\n" ++
     renderConvWGrad "%dWs" "%x" "%dsc1" C C H W kH kW ++
     renderBNParamGrad "%dgs" "%dbs" "%snf_xhat" "%dsnf" M ++
     "    // ── SGD update (trusted): θ' = θ − lr·dθ ──\n" ++
     sgd "%Ws" "%dWs" "%Wsn" "%lWs" "%sWs" (tt [C,C,kH,kW]) ++
     sgd "%gs" "%dgs" "%gsn" "%lgs" "%sgs" f32 ++
     sgd "%bs" "%dbs" "%bsn" "%lbs" "%sbs" f32 ++
     sgd "%W10" "%dW10" "%W10n" "%lW10" "%sW10" ty4 ++
     sgd "%W20" "%dW20" "%W20n" "%lW20" "%sW20" ty4 ++
     sgd "%g10" "%dg10" "%g10n" "%lg10" "%sg10" f32 ++
     sgd "%bb10" "%dbb10" "%bb10n" "%lbb10" "%sbb10" f32 ++
     sgd "%g20" "%dg20" "%g20n" "%lg20" "%sg20" f32 ++
     sgd "%bb20" "%dbb20" "%bb20n" "%lbb20" "%sbb20" f32 ++
     sgd "%W11" "%dW11" "%W11n" "%lW11" "%sW11" ty4 ++
     sgd "%W21" "%dW21" "%W21n" "%lW21" "%sW21" ty4 ++
     sgd "%g11" "%dg11" "%g11n" "%lg11" "%sg11" f32 ++
     sgd "%bb11" "%dbb11" "%bb11n" "%lbb11" "%sbb11" f32 ++
     sgd "%g21" "%dg21" "%g21n" "%lg21" "%sg21" f32 ++
     sgd "%bb21" "%dbb21" "%bb21n" "%lbb21" "%sbb21" f32 ++
     sgd "%Wd" "%dWd" "%Wdn" "%lWd" "%sWd" (tt [C,nCls]) ++
     sgd "%bd" "%dbd" "%bdn" "%lbd" "%sbd" (tt [nCls]) ++
     s!"    return %Wsn, %gsn, %bsn, %W10n, %W20n, %g10n, %bb10n, %g20n, %bb20n, " ++
     s!"%W11n, %W21n, %g11n, %bb11n, %g21n, %bb21n, %Wdn, %bdn : " ++
     s!"{tt [C,C,kH,kW]}, {f32}, {f32}, {ty4}, {ty4}, {f32}, {f32}, {f32}, {f32}, " ++
     s!"{ty4}, {ty4}, {f32}, {f32}, {f32}, {f32}, {tt [C,nCls]}, {tt [nCls]}\n")

-- ════════════════════════════════════════════════════════════════
-- § EfficientNet MBConv — the squeeze-excite inverted residual (Phase 3 sweep)
--
-- The headline EfficientNet block, rendered end-to-end:
--   expand(1×1 conv)→BN→swish → depthwise(k×k)→BN→swish
--     → SE(GAP→dense→swish→dense→sigmoid→broadcast→⊙)
--     → project(1×1 conv)→BN → + x        (stride-1, cin=cout: identity skip)
-- Every stage is an already-validated op; the genuine squeeze-excite gate is a
-- fan-out sub-network multiplied back into the main path, so its backward is a
-- product-rule fan-in (the proven `seBlockFull` / `se_back_bridge` shape):
--   d(in) = broadcast(gate)⊙d(se) + GAPback(gate.back(Σ_spatial(in⊙d(se)))).
-- BN here is the flat `bnForward` (LayerNorm over the flattened example, scalar
-- γ/β); conv biases fold into BN (the be=bd=bp=0 instance), as in `renderResF`.
-- Expand widens channels (cin→cmid) so the expand/dw BN run over cmid·h·w while
-- the project BN runs over cout·h·w — hence the nf/eps-parameterized flat BN.
-- ════════════════════════════════════════════════════════════════

/-- Flat-BN forward (= `renderLN`/`bnForward`) but with explicit `nf`,`eps` SSA
    names so one block can carry several BN widths (cmid·h·w vs cout·h·w). -/
def mbLNF (out x g b nf eps : String) (B n : Nat) : String :=
  reduceSumBcast s!"{out}_sm" x B n ++
  s!"    {out}_mu = stablehlo.divide {out}_sm, {nf} : {tt [B,n]}\n" ++
  s!"    {out}_xc = stablehlo.subtract {x}, {out}_mu : {tt [B,n]}\n" ++
  s!"    {out}_sq = stablehlo.multiply {out}_xc, {out}_xc : {tt [B,n]}\n" ++
  reduceSumBcast s!"{out}_vs" s!"{out}_sq" B n ++
  s!"    {out}_var = stablehlo.divide {out}_vs, {nf} : {tt [B,n]}\n" ++
  s!"    {out}_ve = stablehlo.add {out}_var, {eps} : {tt [B,n]}\n" ++
  s!"    {out}_istd = stablehlo.rsqrt {out}_ve : {tt [B,n]}\n" ++
  s!"    {out}_xhat = stablehlo.multiply {out}_xc, {out}_istd : {tt [B,n]}\n" ++
  s!"    {out}_gb = stablehlo.broadcast_in_dim {g}, dims = [] : (tensor<f32>) -> {tt [B,n]}\n" ++
  s!"    {out}_bb = stablehlo.broadcast_in_dim {b}, dims = [] : (tensor<f32>) -> {tt [B,n]}\n" ++
  s!"    {out}_gx = stablehlo.multiply {out}_xhat, {out}_gb : {tt [B,n]}\n" ++
  s!"    {out} = stablehlo.add {out}_gx, {out}_bb : {tt [B,n]}\n"

/-- Flat-BN backward (= `renderLNBack`) with explicit `nf` name; reuses
    `{ln}_xhat`/`{ln}_istd` + scalar γ `g`. -/
def mbLNB (out ln g dy nf : String) (B n : Nat) : String :=
  s!"    {out}_gb = stablehlo.broadcast_in_dim {g}, dims = [] : (tensor<f32>) -> {tt [B,n]}\n" ++
  s!"    {out}_dxh = stablehlo.multiply {out}_gb, {dy} : {tt [B,n]}\n" ++
  reduceSumBcast s!"{out}_sdx" s!"{out}_dxh" B n ++
  s!"    {out}_xd = stablehlo.multiply {ln}_xhat, {out}_dxh : {tt [B,n]}\n" ++
  reduceSumBcast s!"{out}_sxd" s!"{out}_xd" B n ++
  s!"    {out}_t1 = stablehlo.multiply {out}_dxh, {nf} : {tt [B,n]}\n" ++
  s!"    {out}_i1 = stablehlo.subtract {out}_t1, {out}_sdx : {tt [B,n]}\n" ++
  s!"    {out}_xs = stablehlo.multiply {ln}_xhat, {out}_sxd : {tt [B,n]}\n" ++
  s!"    {out}_i2 = stablehlo.subtract {out}_i1, {out}_xs : {tt [B,n]}\n" ++
  s!"    {out}_s = stablehlo.divide {ln}_istd, {nf} : {tt [B,n]}\n" ++
  s!"    {out} = stablehlo.multiply {out}_s, {out}_i2 : {tt [B,n]}\n"

/-- Elementwise swish forward `y = x·σ(x)` into `{out}` (StableHLO type `ty`). -/
def swishF (out x ty : String) : String :=
  s!"    {out}_s = stablehlo.logistic {x} : {ty}\n" ++
  s!"    {out} = stablehlo.multiply {x}, {out}_s : {ty}\n"

/-- Elementwise swish backward `{out} = dy ⊙ σ(1 + x(1−σ))` (`x` = pre-activation). -/
def swishB (out x dy ty : String) : String :=
  s!"    {out}_s = stablehlo.logistic {x} : {ty}\n" ++
  s!"    {out}_one = stablehlo.constant dense<1.0> : {ty}\n" ++
  s!"    {out}_om = stablehlo.subtract {out}_one, {out}_s : {ty}\n" ++
  s!"    {out}_xom = stablehlo.multiply {x}, {out}_om : {ty}\n" ++
  s!"    {out}_in = stablehlo.add {out}_one, {out}_xom : {ty}\n" ++
  s!"    {out}_sp = stablehlo.multiply {out}_s, {out}_in : {ty}\n" ++
  s!"    {out} = stablehlo.multiply {dy}, {out}_sp : {ty}\n"

/-- Squeeze-excite gate forward: `se = x ⊙ broadcast(σ(W₂·swish(W₁·GAP(x))+b₂))`.
    `x` is `[1,C,H,W]`; produces `{p}se` `[1,C,H,W]`, leaving `{p}gate` `[1,C]`
    and `{p}ex` (pre-mid-swish `[1,r]`) for the backward. `%sc` (f32 0) in scope. -/
def renderSEGate (p x Ws1 bs1 Ws2 bs2 : String) (C H W r : Nat) : String :=
  -- squeeze: global average pool over the spatial axes
  s!"    {p}sq_s = stablehlo.reduce({x} init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({tt [1,C,H,W]}, tensor<f32>) -> {tt [1,C]}\n" ++
  s!"    {p}sq_hw = stablehlo.constant dense<{H*W}.0> : {tt [1,C]}\n" ++
  s!"    {p}sq = stablehlo.divide {p}sq_s, {p}sq_hw : {tt [1,C]}\n" ++
  -- excite: dense C→r, swish, dense r→C, sigmoid
  renderDense s!"{p}ex" s!"{p}sq" Ws1 bs1 1 C r ++
  swishF s!"{p}a1" s!"{p}ex" (tt [1,r]) ++
  renderDense s!"{p}h2" s!"{p}a1" Ws2 bs2 1 r C ++
  s!"    {p}gate = stablehlo.logistic {p}h2 : {tt [1,C]}\n" ++
  -- broadcast the per-channel gate back to spatial and multiply
  s!"    {p}gb = stablehlo.broadcast_in_dim {p}gate, dims = [0, 1] : ({tt [1,C]}) -> {tt [1,C,H,W]}\n" ++
  s!"    {p}se = stablehlo.multiply {x}, {p}gb : {tt [1,C,H,W]}\n"

/-- Squeeze-excite backward: input-cotangent of `se = x ⊙ broadcast(gate(x))`,
    the product-rule fan-in. `dse` cotangent `[1,C,H,W]`; reuses `{p}gate`,`{p}ex`
    + the SE input `x`. Produces `{p}dds` `[1,C,H,W]`. `%sc` in scope. -/
def renderSEGateBack (p x dse Ws1 Ws2 : String) (C H W r : Nat) : String :=
  -- left factor: broadcast(gate) ⊙ dse
  s!"    {p}gb2 = stablehlo.broadcast_in_dim {p}gate, dims = [0, 1] : ({tt [1,C]}) -> {tt [1,C,H,W]}\n" ++
  s!"    {p}dleft = stablehlo.multiply {p}gb2, {dse} : {tt [1,C,H,W]}\n" ++
  -- gate path: dgate = Σ_spatial(x ⊙ dse) (adjoint of broadcast)
  s!"    {p}xdse = stablehlo.multiply {x}, {dse} : {tt [1,C,H,W]}\n" ++
  s!"    {p}dgate = stablehlo.reduce({p}xdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({tt [1,C,H,W]}, tensor<f32>) -> {tt [1,C]}\n" ++
  -- through sigmoid': gate(1−gate)
  s!"    {p}one = stablehlo.constant dense<1.0> : {tt [1,C]}\n" ++
  s!"    {p}omg = stablehlo.subtract {p}one, {p}gate : {tt [1,C]}\n" ++
  s!"    {p}sg = stablehlo.multiply {p}gate, {p}omg : {tt [1,C]}\n" ++
  s!"    {p}dh2 = stablehlo.multiply {p}dgate, {p}sg : {tt [1,C]}\n" ++
  -- dense₂ back: da1 = dh2 · Ws₂ᵀ
  matdg s!"{p}da1" s!"{p}dh2" Ws2 "1" "1" (tt [1,C]) (tt [r,C]) (tt [1,r]) ++
  -- swish back through the excite mid
  swishB s!"{p}dex" s!"{p}ex" s!"{p}da1" (tt [1,r]) ++
  -- dense₁ back: dsq = dex · Ws₁ᵀ
  matdg s!"{p}dsq" s!"{p}dex" Ws1 "1" "1" (tt [1,r]) (tt [C,r]) (tt [1,C]) ++
  -- GAP back: broadcast(dsq / (H·W)) over spatial
  s!"    {p}dsq_hw = stablehlo.constant dense<{H*W}.0> : {tt [1,C]}\n" ++
  s!"    {p}dsq_d = stablehlo.divide {p}dsq, {p}dsq_hw : {tt [1,C]}\n" ++
  s!"    {p}dgate_sp = stablehlo.broadcast_in_dim {p}dsq_d, dims = [0, 1] : ({tt [1,C]}) -> {tt [1,C,H,W]}\n" ++
  -- sum the two fan-in contributions
  s!"    {p}dds = stablehlo.add {p}dleft, {p}dgate_sp : {tt [1,C,H,W]}\n"

/-- Per-MBConv parameter signature (expand 1×1, depthwise k×k, SE squeeze C→r→C,
    project 1×1; conv biases folded into BN). `cin = cout = c` for the skip. -/
def mbconvSig (c cmid H W kHd kWd r : Nat) : String :=
  s!"%x: {tt [1,c,H,W]}, %We: {tt [cmid,c,1,1]}, %ge: tensor<f32>, %be: tensor<f32>, " ++
  s!"%Wd: {tt [cmid,kHd,kWd]}, %gd: tensor<f32>, %bd: tensor<f32>, " ++
  s!"%Ws1: {tt [cmid,r]}, %bs1: {tt [r]}, %Ws2: {tt [r,cmid]}, %bs2: {tt [cmid]}, " ++
  s!"%Wp: {tt [c,cmid,1,1]}, %gp: tensor<f32>, %bp: tensor<f32>"

/-- Shared constants: f32 zero + the two BN widths' nf/eps. -/
def mbconvConsts (c cmid H W : Nat) (eps : String) : String :=
  let Me := cmid*H*W; let Mp := c*H*W
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %nfe = stablehlo.constant dense<{Me}.0> : {tt [1,Me]}\n" ++
  s!"    %epse = stablehlo.constant dense<{eps}> : {tt [1,Me]}\n" ++
  s!"    %nfp = stablehlo.constant dense<{Mp}.0> : {tt [1,Mp]}\n" ++
  s!"    %epsp = stablehlo.constant dense<{eps}> : {tt [1,Mp]}\n"

/-- MBConv forward chain producing `%out`, leaving every saved activation
    (`%e_n`,`%d_n`,`%e_c`,`%d_c`,`%d_s4`, the SE `%g…`, `%p_n…`) for the backward. -/
def mbconvFwdBody (c cmid H W kHd kWd r : Nat) : String :=
  let pHd := (kHd-1)/2; let pWd := (kWd-1)/2; let Me := cmid*H*W; let Mp := c*H*W
  -- expand 1×1 conv → flat BN → swish
  convOp "%e_c" "%x" "%We" (tt [1,c,H,W]) (tt [cmid,c,1,1]) (tt [1,cmid,H,W]) 0 0 ++
  s!"    %e_cf = stablehlo.reshape %e_c : ({tt [1,cmid,H,W]}) -> {tt [1,Me]}\n" ++
  mbLNF "%e_n" "%e_cf" "%ge" "%be" "%nfe" "%epse" 1 Me ++
  swishF "%e_s" "%e_n" (tt [1,Me]) ++
  s!"    %e_s4 = stablehlo.reshape %e_s : ({tt [1,Me]}) -> {tt [1,cmid,H,W]}\n" ++
  -- depthwise (no bias) → flat BN → swish
  s!"    %dwe = stablehlo.reshape %Wd : ({tt [cmid,kHd,kWd]}) -> {tt [cmid,1,kHd,kWd]}\n" ++
  convOpG "%d_c" "%e_s4" "%dwe" (tt [1,cmid,H,W]) (tt [cmid,1,kHd,kWd]) (tt [1,cmid,H,W]) pHd pWd (toString cmid) ++
  s!"    %d_cf = stablehlo.reshape %d_c : ({tt [1,cmid,H,W]}) -> {tt [1,Me]}\n" ++
  mbLNF "%d_n" "%d_cf" "%gd" "%bd" "%nfe" "%epse" 1 Me ++
  swishF "%d_s" "%d_n" (tt [1,Me]) ++
  s!"    %d_s4 = stablehlo.reshape %d_s : ({tt [1,Me]}) -> {tt [1,cmid,H,W]}\n" ++
  -- squeeze-excite (genuine gate sub-network)
  renderSEGate "%g" "%d_s4" "%Ws1" "%bs1" "%Ws2" "%bs2" cmid H W r ++
  -- project 1×1 conv → flat BN
  convOp "%p_c" "%gse" "%Wp" (tt [1,cmid,H,W]) (tt [c,cmid,1,1]) (tt [1,c,H,W]) 0 0 ++
  s!"    %p_cf = stablehlo.reshape %p_c : ({tt [1,c,H,W]}) -> {tt [1,Mp]}\n" ++
  mbLNF "%p_n" "%p_cf" "%gp" "%bp" "%nfp" "%epsp" 1 Mp ++
  s!"    %p_n4 = stablehlo.reshape %p_n : ({tt [1,Mp]}) -> {tt [1,c,H,W]}\n" ++
  -- identity residual skip
  s!"    %out = stablehlo.add %x, %p_n4 : {tt [1,c,H,W]}\n"

/-- Residual MBConv forward as `@mbconv_fwd`. -/
def mbconvFwdModule (c cmid H W kHd kWd r : Nat) (eps : String) : String :=
  actMod "mbconv_fwd" (mbconvSig c cmid H W kHd kWd r) (tt [1,c,H,W])
    (mbconvConsts c cmid H W eps ++ mbconvFwdBody c cmid H W kHd kWd r ++
     s!"    return %out : {tt [1,c,H,W]}\n")

/-- Residual MBConv input-gradient backward `dx = dOut + body.back(dOut)` as
    `@mbconv_back`: project-back → SE fan-in → dw-bn-swish back → expand-bn-swish
    back, then the residual add. Recomputes the forward to save activations. -/
def mbconvBackModule (c cmid H W kHd kWd r : Nat) (eps : String) : String :=
  let pHd := (kHd-1)/2; let pWd := (kWd-1)/2; let Me := cmid*H*W; let Mp := c*H*W
  actMod "mbconv_back" (mbconvSig c cmid H W kHd kWd r ++ s!", %dOut: {tt [1,c,H,W]}") (tt [1,c,H,W])
    (mbconvConsts c cmid H W eps ++ mbconvFwdBody c cmid H W kHd kWd r ++
     "    // ── backward: input gradient dx = dOut + body.back(dOut) ──\n" ++
     -- project back: BN-back then 1×1 conv input-VJP
     s!"    %dOnf = stablehlo.reshape %dOut : ({tt [1,c,H,W]}) -> {tt [1,Mp]}\n" ++
     mbLNB "%dpcf" "%p_n" "%gp" "%dOnf" "%nfp" 1 Mp ++
     s!"    %dpc = stablehlo.reshape %dpcf : ({tt [1,Mp]}) -> {tt [1,c,H,W]}\n" ++
     s!"    %Wpt = stablehlo.transpose %Wp, dims = [1, 0, 2, 3] : ({tt [c,cmid,1,1]}) -> {tt [cmid,c,1,1]}\n" ++
     s!"    %Wpr = stablehlo.reverse %Wpt, dims = [2, 3] : {tt [cmid,c,1,1]}\n" ++
     convOp "%dse" "%dpc" "%Wpr" (tt [1,c,H,W]) (tt [cmid,c,1,1]) (tt [1,cmid,H,W]) 0 0 ++
     -- squeeze-excite back (fan-in)
     renderSEGateBack "%g" "%d_s4" "%dse" "%Ws1" "%Ws2" cmid H W r ++
     -- depthwise-bn-swish back
     s!"    %gddsf = stablehlo.reshape %gdds : ({tt [1,cmid,H,W]}) -> {tt [1,Me]}\n" ++
     swishB "%ddn" "%d_n" "%gddsf" (tt [1,Me]) ++
     mbLNB "%ddc" "%d_n" "%gd" "%ddn" "%nfe" 1 Me ++
     s!"    %ddc4 = stablehlo.reshape %ddc : ({tt [1,Me]}) -> {tt [1,cmid,H,W]}\n" ++
     s!"    %dwe2 = stablehlo.reshape %Wd : ({tt [cmid,kHd,kWd]}) -> {tt [cmid,1,kHd,kWd]}\n" ++
     s!"    %dwr = stablehlo.reverse %dwe2, dims = [2, 3] : {tt [cmid,1,kHd,kWd]}\n" ++
     convOpG "%des" "%ddc4" "%dwr" (tt [1,cmid,H,W]) (tt [cmid,1,kHd,kWd]) (tt [1,cmid,H,W]) pHd pWd (toString cmid) ++
     s!"    %desf = stablehlo.reshape %des : ({tt [1,cmid,H,W]}) -> {tt [1,Me]}\n" ++
     -- expand-bn-swish back
     swishB "%den" "%e_n" "%desf" (tt [1,Me]) ++
     mbLNB "%dec" "%e_n" "%ge" "%den" "%nfe" 1 Me ++
     s!"    %dec4 = stablehlo.reshape %dec : ({tt [1,Me]}) -> {tt [1,cmid,H,W]}\n" ++
     s!"    %Wet = stablehlo.transpose %We, dims = [1, 0, 2, 3] : ({tt [cmid,c,1,1]}) -> {tt [c,cmid,1,1]}\n" ++
     s!"    %Wer = stablehlo.reverse %Wet, dims = [2, 3] : {tt [c,cmid,1,1]}\n" ++
     convOp "%dxb" "%dec4" "%Wer" (tt [1,cmid,H,W]) (tt [c,cmid,1,1]) (tt [1,c,H,W]) 0 0 ++
     -- residual fan-in
     s!"    %dx = stablehlo.add %dOut, %dxb : {tt [1,c,H,W]}\n" ++
     s!"    return %dx : {tt [1,c,H,W]}\n")

-- ════════════════════════════════════════════════════════════════
-- § MobileNetV2 inverted residual — the linear-bottleneck block (Phase 3 sweep)
--
--   expand(1×1 conv)→BN→relu6 → depthwise(k×k)→BN→relu6 → project(1×1 conv)→BN
--     + x        (stride-1, cin=cout: identity skip; linear bottleneck — no
--                 activation after project)
-- The MBConv skeleton minus the squeeze-excite branch, relu6 in place of swish
-- (`invresBody = ivProject ∘ ivDepthwise ∘ ivExpand`). relu6 has a kink, so its
-- bridge holds at a smooth point (no pre-activation exactly 0 or 6) — the
-- measure-zero conditionality of the proven `relu6_has_vjp_at`. Reuses the
-- nf/eps-parameterized flat BN and the conv/depthwise machinery from MBConv.
-- ════════════════════════════════════════════════════════════════

/-- Elementwise relu6 forward `y = min(max(x,0),6)` into `{out}` (type `ty`). -/
def relu6F (out x ty : String) : String :=
  s!"    {out}_z = stablehlo.constant dense<0.0> : {ty}\n" ++
  s!"    {out}_six = stablehlo.constant dense<6.0> : {ty}\n" ++
  s!"    {out}_m1 = stablehlo.maximum {x}, {out}_z : {ty}\n" ++
  s!"    {out} = stablehlo.minimum {out}_m1, {out}_six : {ty}\n"

/-- Elementwise relu6 backward `{out} = dy ⊙ 1[0<x<6]` (`x` = pre-activation). -/
def relu6B (out x dy ty tyI1 : String) : String :=
  s!"    {out}_z = stablehlo.constant dense<0.0> : {ty}\n" ++
  s!"    {out}_six = stablehlo.constant dense<6.0> : {ty}\n" ++
  s!"    {out}_gt = stablehlo.compare GT, {x}, {out}_z : ({ty}, {ty}) -> {tyI1}\n" ++
  s!"    {out}_lt = stablehlo.compare LT, {x}, {out}_six : ({ty}, {ty}) -> {tyI1}\n" ++
  s!"    {out}_mask = stablehlo.and {out}_gt, {out}_lt : {tyI1}\n" ++
  s!"    {out} = stablehlo.select {out}_mask, {dy}, {out}_z : {tyI1}, {ty}\n"

/-- Inverted-residual parameter signature (expand 1×1, depthwise k×k, project
    1×1; conv biases folded into BN). `cin = cout = c` for the identity skip. -/
def invresSig (c cmid H W kHd kWd : Nat) : String :=
  s!"%x: {tt [1,c,H,W]}, %We: {tt [cmid,c,1,1]}, %ge: tensor<f32>, %be: tensor<f32>, " ++
  s!"%Wd: {tt [cmid,kHd,kWd]}, %gd: tensor<f32>, %bd: tensor<f32>, " ++
  s!"%Wp: {tt [c,cmid,1,1]}, %gp: tensor<f32>, %bp: tensor<f32>"

/-- Inverted-residual forward chain producing `%out`, leaving `%e_n`,`%d_n`,
    `%e_c`,`%d_c`,`%p_n…` for the backward. -/
def invresFwdBody (c cmid H W kHd kWd : Nat) : String :=
  let pHd := (kHd-1)/2; let pWd := (kWd-1)/2; let Me := cmid*H*W; let Mp := c*H*W
  -- expand 1×1 conv → flat BN → relu6
  convOp "%e_c" "%x" "%We" (tt [1,c,H,W]) (tt [cmid,c,1,1]) (tt [1,cmid,H,W]) 0 0 ++
  s!"    %e_cf = stablehlo.reshape %e_c : ({tt [1,cmid,H,W]}) -> {tt [1,Me]}\n" ++
  mbLNF "%e_n" "%e_cf" "%ge" "%be" "%nfe" "%epse" 1 Me ++
  relu6F "%e_r" "%e_n" (tt [1,Me]) ++
  s!"    %e_r4 = stablehlo.reshape %e_r : ({tt [1,Me]}) -> {tt [1,cmid,H,W]}\n" ++
  -- depthwise (no bias) → flat BN → relu6
  s!"    %dwe = stablehlo.reshape %Wd : ({tt [cmid,kHd,kWd]}) -> {tt [cmid,1,kHd,kWd]}\n" ++
  convOpG "%d_c" "%e_r4" "%dwe" (tt [1,cmid,H,W]) (tt [cmid,1,kHd,kWd]) (tt [1,cmid,H,W]) pHd pWd (toString cmid) ++
  s!"    %d_cf = stablehlo.reshape %d_c : ({tt [1,cmid,H,W]}) -> {tt [1,Me]}\n" ++
  mbLNF "%d_n" "%d_cf" "%gd" "%bd" "%nfe" "%epse" 1 Me ++
  relu6F "%d_r" "%d_n" (tt [1,Me]) ++
  s!"    %d_r4 = stablehlo.reshape %d_r : ({tt [1,Me]}) -> {tt [1,cmid,H,W]}\n" ++
  -- project 1×1 conv → flat BN (linear bottleneck — no activation)
  convOp "%p_c" "%d_r4" "%Wp" (tt [1,cmid,H,W]) (tt [c,cmid,1,1]) (tt [1,c,H,W]) 0 0 ++
  s!"    %p_cf = stablehlo.reshape %p_c : ({tt [1,c,H,W]}) -> {tt [1,Mp]}\n" ++
  mbLNF "%p_n" "%p_cf" "%gp" "%bp" "%nfp" "%epsp" 1 Mp ++
  s!"    %p_n4 = stablehlo.reshape %p_n : ({tt [1,Mp]}) -> {tt [1,c,H,W]}\n" ++
  -- identity residual skip
  s!"    %out = stablehlo.add %x, %p_n4 : {tt [1,c,H,W]}\n"

/-- Residual inverted-residual forward as `@invres_fwd`. -/
def invresFwdModule (c cmid H W kHd kWd : Nat) (eps : String) : String :=
  actMod "invres_fwd" (invresSig c cmid H W kHd kWd) (tt [1,c,H,W])
    (mbconvConsts c cmid H W eps ++ invresFwdBody c cmid H W kHd kWd ++
     s!"    return %out : {tt [1,c,H,W]}\n")

/-- Residual inverted-residual input-gradient backward `dx = dOut + body.back(dOut)`
    as `@invres_back`: project-back → relu6/dw → relu6/expand, then the skip. -/
def invresBackModule (c cmid H W kHd kWd : Nat) (eps : String) : String :=
  let pHd := (kHd-1)/2; let pWd := (kWd-1)/2; let Me := cmid*H*W; let Mp := c*H*W
  actMod "invres_back" (invresSig c cmid H W kHd kWd ++ s!", %dOut: {tt [1,c,H,W]}") (tt [1,c,H,W])
    (mbconvConsts c cmid H W eps ++ invresFwdBody c cmid H W kHd kWd ++
     "    // ── backward: input gradient dx = dOut + body.back(dOut) ──\n" ++
     -- project back: BN-back then 1×1 conv input-VJP
     s!"    %dOnf = stablehlo.reshape %dOut : ({tt [1,c,H,W]}) -> {tt [1,Mp]}\n" ++
     mbLNB "%dpcf" "%p_n" "%gp" "%dOnf" "%nfp" 1 Mp ++
     s!"    %dpc = stablehlo.reshape %dpcf : ({tt [1,Mp]}) -> {tt [1,c,H,W]}\n" ++
     s!"    %Wpt = stablehlo.transpose %Wp, dims = [1, 0, 2, 3] : ({tt [c,cmid,1,1]}) -> {tt [cmid,c,1,1]}\n" ++
     s!"    %Wpr = stablehlo.reverse %Wpt, dims = [2, 3] : {tt [cmid,c,1,1]}\n" ++
     convOp "%ddr" "%dpc" "%Wpr" (tt [1,c,H,W]) (tt [cmid,c,1,1]) (tt [1,cmid,H,W]) 0 0 ++
     -- depthwise-bn-relu6 back
     s!"    %ddrf = stablehlo.reshape %ddr : ({tt [1,cmid,H,W]}) -> {tt [1,Me]}\n" ++
     relu6B "%ddn" "%d_n" "%ddrf" (tt [1,Me]) (ti1 [1,Me]) ++
     mbLNB "%ddc" "%d_n" "%gd" "%ddn" "%nfe" 1 Me ++
     s!"    %ddc4 = stablehlo.reshape %ddc : ({tt [1,Me]}) -> {tt [1,cmid,H,W]}\n" ++
     s!"    %dwe2 = stablehlo.reshape %Wd : ({tt [cmid,kHd,kWd]}) -> {tt [cmid,1,kHd,kWd]}\n" ++
     s!"    %dwr = stablehlo.reverse %dwe2, dims = [2, 3] : {tt [cmid,1,kHd,kWd]}\n" ++
     convOpG "%des" "%ddc4" "%dwr" (tt [1,cmid,H,W]) (tt [cmid,1,kHd,kWd]) (tt [1,cmid,H,W]) pHd pWd (toString cmid) ++
     s!"    %desf = stablehlo.reshape %des : ({tt [1,cmid,H,W]}) -> {tt [1,Me]}\n" ++
     -- expand-bn-relu6 back
     relu6B "%den" "%e_n" "%desf" (tt [1,Me]) (ti1 [1,Me]) ++
     mbLNB "%dec" "%e_n" "%ge" "%den" "%nfe" 1 Me ++
     s!"    %dec4 = stablehlo.reshape %dec : ({tt [1,Me]}) -> {tt [1,cmid,H,W]}\n" ++
     s!"    %Wet = stablehlo.transpose %We, dims = [1, 0, 2, 3] : ({tt [cmid,c,1,1]}) -> {tt [c,cmid,1,1]}\n" ++
     s!"    %Wer = stablehlo.reverse %Wet, dims = [2, 3] : {tt [c,cmid,1,1]}\n" ++
     convOp "%dxb" "%dec4" "%Wer" (tt [1,cmid,H,W]) (tt [c,cmid,1,1]) (tt [1,c,H,W]) 0 0 ++
     -- residual fan-in
     s!"    %dx = stablehlo.add %dOut, %dxb : {tt [1,c,H,W]}\n" ++
     s!"    return %dx : {tt [1,c,H,W]}\n")

-- Dump (human view) + write compilable modules for the IREE loop
-- (run: `lake env lean LeanMlir/Proofs/IRPrint.lean`).
#eval IO.println (renderBlock "linear d₀=4 → d₁=3 (B=2)" 2 (linearHlo 4 3))
#eval IO.println (renderBlock "mlp 4→3→3→2 (B=2)" 2 (mlpHlo 4 3 3 2))
#eval IO.FS.writeFile "/tmp/linear_back.mlir" (linearModule 2 4 3)
#eval IO.FS.writeFile "/tmp/mlp_back.mlir" (mlpModule 2 4 3 3 2)
#eval IO.FS.writeFile "/tmp/mlp_fwd.mlir" (mlpFwdModule 2 4 3 3 2)
#eval IO.FS.writeFile "/tmp/loss_cot.mlir" (lossCotModule 2 2)
#eval IO.FS.writeFile "/tmp/mlp_train_step.mlir" (mlpTrainStepModule 2 4 3 3 2 "0.1")
-- CNN (Phase 3): conv forward + proof-backed conv backward, 1→2 ch, 4×4, 3×3.
#eval IO.FS.writeFile "/tmp/conv_fwd.mlir" (convFwdModule 1 1 2 4 4 3 3)
#eval IO.FS.writeFile "/tmp/conv_back.mlir" (convBackModule 1 1 2 4 4 3 3)
-- 2×2 max pool forward + proof-backed backward, 2 ch, 4×4 → 2×2.
#eval IO.FS.writeFile "/tmp/maxpool_fwd.mlir" (maxpoolFwdModule 1 2 2 2)
#eval IO.FS.writeFile "/tmp/maxpool_back.mlir" (maxpoolBackModule 1 2 2 2)
-- CNN capstone: conv(1→2,3×3) → relu → maxpool → flatten(8) → dense(8→3), fwd+dx.
#eval IO.FS.writeFile "/tmp/cnn_back.mlir" (cnnModule 1 2 4 4 3 3 3)
-- CNN full SGD train step: same net, 4 updated params (conv dW via transpose trick).
#eval IO.FS.writeFile "/tmp/cnn_train_step.mlir" (cnnTrainStepModule 1 2 4 4 3 3 3 "0.1")
-- BatchNorm/LayerNorm forward + proof-backed 3-term backward, B=2, n=4, ε=1e-5.
#eval IO.FS.writeFile "/tmp/bn_fwd.mlir" (bnFwdModule 2 4 "0.00001")
#eval IO.FS.writeFile "/tmp/bn_back.mlir" (bnBackModule 2 4 "0.00001")
-- Softmax forward + proven rank-1 backward, B=2, c=4.
#eval IO.FS.writeFile "/tmp/softmax_fwd.mlir" (softmaxFwdModule 2 4)
#eval IO.FS.writeFile "/tmp/softmax_back.mlir" (softmaxBackModule 2 4)
-- Scaled dot-product attention forward + proven backward (dQ,dK,dV), n=3, d=4, 1/√4=0.5.
#eval IO.FS.writeFile "/tmp/sdpa_fwd.mlir" (sdpaFwdModule 3 4 "0.5")
#eval IO.FS.writeFile "/tmp/sdpa_back.mlir" (sdpaBackModule 3 4 "0.5")
-- Pointwise activations (m=8): fwd + proven dy⊙act'(x) backward.
#eval IO.FS.writeFile "/tmp/sigmoid_fwd.mlir" (sigmoidFwdM 8)
#eval IO.FS.writeFile "/tmp/sigmoid_back.mlir" (sigmoidBackM 8)
#eval IO.FS.writeFile "/tmp/swish_fwd.mlir" (swishFwdM 8)
#eval IO.FS.writeFile "/tmp/swish_back.mlir" (swishBackM 8)
#eval IO.FS.writeFile "/tmp/relu6_fwd.mlir" (relu6FwdM 8)
#eval IO.FS.writeFile "/tmp/relu6_back.mlir" (relu6BackM 8)
#eval IO.FS.writeFile "/tmp/gelu_fwd.mlir" (geluFwdM 8)
#eval IO.FS.writeFile "/tmp/gelu_back.mlir" (geluBackM 8)
-- Residual (add fan-in) + Squeeze-Excite (gate-multiply fan-in).
#eval IO.FS.writeFile "/tmp/residual_fwd.mlir" (residualFwdM 2 4)
#eval IO.FS.writeFile "/tmp/residual_back.mlir" (residualBackM 2 4)
#eval IO.FS.writeFile "/tmp/se_fwd.mlir" (seFwdM 8)
#eval IO.FS.writeFile "/tmp/se_back.mlir" (seBackM 8)
-- Depthwise (per-channel grouped) conv forward + proof-backed backward, c=2, 4×4, 3×3.
#eval IO.FS.writeFile "/tmp/dw_fwd.mlir" (depthwiseFwdM 2 4 4 3 3)
#eval IO.FS.writeFile "/tmp/dw_back.mlir" (depthwiseBackM 2 4 4 3 3)
-- ViT attention sublayer (LN→QKV→SDPA→Wo→residual), N=2 tokens, D=4, ε=1e-5, 1/√4=0.5.
#eval IO.FS.writeFile "/tmp/attn_fwd.mlir" (attnSublayerFwdModule 2 4 "0.00001" "0.5")
#eval IO.FS.writeFile "/tmp/attn_back.mlir" (attnSublayerBackModule 2 4 "0.00001" "0.5")
-- Full ViT transformer block (attn + MLP sublayers), N=2, D=4, F=8.
#eval IO.FS.writeFile "/tmp/vit_fwd.mlir" (vitBlockFwdModule 2 4 8 "0.00001" "0.5")
#eval IO.FS.writeFile "/tmp/vit_back.mlir" (vitBlockBackModule 2 4 8 "0.00001" "0.5")
-- ResNet basic residual block (conv-BN-relu-conv-BN + skip + relu), C=2, 4×4, 3×3.
#eval IO.FS.writeFile "/tmp/res_fwd.mlir" (resBlockFwdModule 2 4 4 3 3 "0.00001")
#eval IO.FS.writeFile "/tmp/res_back.mlir" (resBlockBackModule 2 4 4 3 3 "0.00001")
-- ResNet-34-depth residual tower (16 blocks = 3+4+6+3), generated by a loop.
#eval IO.FS.writeFile "/tmp/res_tower_fwd.mlir" (resTowerFwdModule 16 2 4 4 3 3 "0.00001")
#eval IO.FS.writeFile "/tmp/res_tower_back.mlir" (resTowerBackModule 16 2 4 4 3 3 "0.00001")
-- Full ResNet SGD train step: stem + 2 blocks + GAP + FC + softmax-CE, 17 params.
#eval IO.FS.writeFile "/tmp/resnet_train_step.mlir" (resnetTrainStepModule 2 4 4 3 3 3 "0.00001" "0.1")

#eval IO.FS.writeFile "/tmp/mbconv_fwd.mlir" (mbconvFwdModule 2 4 4 4 3 3 2 "0.00001")
#eval IO.FS.writeFile "/tmp/mbconv_back.mlir" (mbconvBackModule 2 4 4 4 3 3 2 "0.00001")

#eval IO.FS.writeFile "/tmp/invres_fwd.mlir" (invresFwdModule 2 4 4 4 3 3 "0.00001")
#eval IO.FS.writeFile "/tmp/invres_back.mlir" (invresBackModule 2 4 4 4 3 3 "0.00001")

end Proofs.IRPrint
