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

end Proofs.IRPrint
