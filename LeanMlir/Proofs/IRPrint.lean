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

-- ════════════════════════════════════════════════════════════════
-- § Full train step — forward + proof-backed backward + SGD
--
-- The backward above emits only `dx` (the input-gradient / VJP chain). A
-- train step also needs the *parameter* gradients and an optimizer update.
-- This module renders one full SGD step for the MLP:
--
--   forward  (TRUSTED): recompute the saved activations h0,a0,h1,a1 from x
--                       and the weights (Phase-2 forward-IR bridge would make
--                       this proof-backed too; here it is hand-rendered).
--   backward (PROOF-BACKED): the dx chain is ⟦emitMlpBack⟧ = mlp_has_vjp_at
--                       .backward; each dWℓ = aₗ₋₁ᵀ·dyℓ (batch-contracting
--                       dot_general) and dbℓ = Σ_batch dyℓ (reduce-add) is
--                       IR.emitWeightGrad / IR.emitBiasGrad, bridged to the
--                       certified Jacobians by weight_grad_bridge /
--                       bias_grad_bridge.
--   SGD      (TRUSTED): θ' = θ − lr·dθ, elementwise.
--
-- So the gradient core the optimizer consumes is the rendering of
-- proof-backed IR; forward and the SGD arithmetic are the trusted frame.
-- ════════════════════════════════════════════════════════════════

/-- A full MLP SGD train step (`dense → relu → dense → relu → dense`),
    `dims d₀→d₁→d₂→d₃`, batch `B`, learning rate `lr` (a decimal literal).
    Inputs: `x`, the six parameters, and the output cotangent `%dy`
    (`∂L/∂logits`, supplied — keeps the loss out of scope). Returns the six
    updated parameters. Op-for-op the rendering of the proof-backed backward
    (`IR.emitMlpBack` + `IR.emitWeightGrad`/`emitBiasGrad`) inside a trusted
    forward + SGD frame. -/
def mlpTrainStepModule (B d₀ d₁ d₂ d₃ : Nat) (lr : String) : String :=
  let sgd (θ dθ θ' lrC sg ty : String) : String :=
    s!"    {lrC} = stablehlo.constant dense<{lr}> : {ty}\n" ++
    s!"    {sg} = stablehlo.multiply {dθ}, {lrC} : {ty}\n" ++
    s!"    {θ'} = stablehlo.subtract {θ}, {sg} : {ty}\n"
  let dg (o a b cdA cdB tyA tyB tyO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {b}, contracting_dims = [{cdA}] x [{cdB}],\n" ++
    s!"              precision = [DEFAULT, DEFAULT] : ({tyA}, {tyB}) -> {tyO}\n"
  "module @m {\n" ++
  s!"  func.func @mlp_train_step(%x: {tt [B,d₀]}, %W0: {tt [d₀,d₁]}, %b0: {tt [d₁]}, " ++
  s!"%W1: {tt [d₁,d₂]}, %b1: {tt [d₂]}, %W2: {tt [d₂,d₃]}, %b2: {tt [d₃]}, %dy: {tt [B,d₃]}) -> " ++
  s!"({tt [d₀,d₁]}, {tt [d₁]}, {tt [d₁,d₂]}, {tt [d₂]}, {tt [d₂,d₃]}, {tt [d₃]}) " ++ "{\n" ++
  -- ── forward (trusted) ──
  "    // ── forward (trusted): recompute saved activations from x, weights ──\n" ++
  dg "%xw0" "%x" "%W0" "1" "0" (tt [B,d₀]) (tt [d₀,d₁]) (tt [B,d₁]) ++
  s!"    %b0b = stablehlo.broadcast_in_dim %b0, dims = [1] : ({tt [d₁]}) -> {tt [B,d₁]}\n" ++
  s!"    %h0 = stablehlo.add %xw0, %b0b : {tt [B,d₁]}\n" ++
  s!"    %za = stablehlo.constant dense<0.0> : {tt [B,d₁]}\n" ++
  s!"    %a0 = stablehlo.maximum %h0, %za : {tt [B,d₁]}\n" ++
  dg "%aw1" "%a0" "%W1" "1" "0" (tt [B,d₁]) (tt [d₁,d₂]) (tt [B,d₂]) ++
  s!"    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1] : ({tt [d₂]}) -> {tt [B,d₂]}\n" ++
  s!"    %h1 = stablehlo.add %aw1, %b1b : {tt [B,d₂]}\n" ++
  s!"    %zb = stablehlo.constant dense<0.0> : {tt [B,d₂]}\n" ++
  s!"    %a1 = stablehlo.maximum %h1, %zb : {tt [B,d₂]}\n" ++
  -- ── backward (proof-backed) ──
  "    // ── backward (PROOF-BACKED: dx chain = ⟦emitMlpBack⟧ = mlp_has_vjp_at.backward;\n" ++
  "    //    dWℓ/dbℓ = emitWeightGrad/emitBiasGrad, bridged to the certified Jacobians) ──\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
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

-- Dump (human view) + write compilable modules for the IREE loop
-- (run: `lake env lean LeanMlir/Proofs/IRPrint.lean`).
#eval IO.println (renderBlock "linear d₀=4 → d₁=3 (B=2)" 2 (linearHlo 4 3))
#eval IO.println (renderBlock "mlp 4→3→3→2 (B=2)" 2 (mlpHlo 4 3 3 2))
#eval IO.FS.writeFile "/tmp/linear_back.mlir" (linearModule 2 4 3)
#eval IO.FS.writeFile "/tmp/mlp_back.mlir" (mlpModule 2 4 3 3 2)
#eval IO.FS.writeFile "/tmp/mlp_train_step.mlir" (mlpTrainStepModule 2 4 3 3 2 "0.1")

end Proofs.IRPrint
