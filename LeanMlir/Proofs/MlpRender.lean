import LeanMlir.Proofs.LinearTrainStep

/-! # MLP render half — the train-step text as a name-threaded render of proven graphs

The linear `renderModuleN` rendered ONE shared cotangent subgraph (`%dy`). The MLP
train step is a DAG with several shared intermediates: the backward `select`s read the
forward *pre-activations* (`%h0`,`%h1`); the parameter gradients read the *activations*
(`%a0`,`%a1`) and the per-layer cotangents. So the renderer threads names: render each
forward piece from its proven `SHlo` graph (via `pretty`, capturing the fresh result
SSA), then emit the backward / param-grad / SGD ops referencing the captured names.

The forward pieces (`denseF`/`reluF`, `lossCotGraph`) are denotable and proven faithful
(`denseF_faithful`/`reluF_faithful`/`lossCotGraph_isCEgrad`); the backward + param-grad +
SGD ops mirror the GPU-validated emitter (the same op text as `mlpTrainStepText`), now
assembled around proof-rendered forward SSA. The result is a valid MLP train-step module
generated from the proven forward graphs — the multi-intermediate generalization of the
linear render half (cf. `planning/verified_train_step.md`, Crux B).
-/

namespace Proofs.StableHLO

open Proofs

/-- Structured MLP train-step renderer: forward pre-acts/activations/logits/cotangent
    from proven `SHlo` graphs (name-threaded), then backward + param-grad + SGD ops
    referencing the captured names. Produces `@mlp_train_step`. -/
def mlpTrainStepStructured (B d₀ d₁ d₂ d₃ : Nat) (lrStr : String)
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) : String :=
  -- op templates (same as the GPU-validated `mlpTrainStepText`)
  let dg (o a w cA cB tA tB tO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {w}, contracting_dims = [{cA}] x [{cB}], precision = [DEFAULT, DEFAULT] : ({tA}, {tB}) -> {tO}\n"
  let reduce (o dyk : String) (nn : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dyk} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,nn]}, tensor<f32>) -> {ty [nn]}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lrStr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  -- forward pre-acts/activations from proven graphs; operand VALUES are placeholders
  -- (`pretty` renders names only — it never reads an operand's value), so the renderer
  -- stays computable while denotation is a separate concern.
  let z₁ : Vec d₁ := fun _ => 0
  let z₂ : Vec d₂ := fun _ => 0
  let z₃ : Vec d₃ := fun _ => 0
  let go : StateM Nat String := do
    let (cp0, np0) ← pretty B (denseF "%W0" "%b0" W₀ b₀ (.operand "%x" x))
    let (ca0, na0) ← pretty B (.reluF (.operand np0 z₁))
    let (cp1, np1) ← pretty B (denseF "%W1" "%b1" W₁ b₁ (.operand na0 z₁))
    let (ca1, na1) ← pretty B (.reluF (.operand np1 z₂))
    let (clog, nlog) ← pretty B (denseF "%W2" "%b2" W₂ b₂ (.operand na1 z₂))
    let (cdy, ndy) ← pretty B
      (.sub (.softmaxDiv (.expe (.operand nlog z₃))) (.operand "%onehot" z₃))
    pure <|
      "    // ── forward (denseF/reluF, proof-rendered) ──\n" ++ cp0 ++ ca0 ++ cp1 ++ ca1 ++ clog ++
      "    // ── loss cotangent dy = softmax(logits) − onehot ──\n" ++ cdy ++
      "    // ── backward: dotOut + select masks reading the forward pre-acts ──\n" ++
      dg "%dx2" ndy "%W2" "1" "1" (ty [B,d₃]) (ty [d₂,d₃]) (ty [B,d₂]) ++
      s!"    %bz1 = stablehlo.constant dense<0.0> : {ty [B,d₂]}\n" ++
      s!"    %bm1 = stablehlo.compare GT, {np1}, %bz1 : ({ty [B,d₂]}, {ty [B,d₂]}) -> {tyI1 [B,d₂]}\n" ++
      s!"    %dy1 = stablehlo.select %bm1, %dx2, %bz1 : {tyI1 [B,d₂]}, {ty [B,d₂]}\n" ++
      dg "%dx1" "%dy1" "%W1" "1" "1" (ty [B,d₂]) (ty [d₁,d₂]) (ty [B,d₁]) ++
      s!"    %bz0 = stablehlo.constant dense<0.0> : {ty [B,d₁]}\n" ++
      s!"    %bm0 = stablehlo.compare GT, {np0}, %bz0 : ({ty [B,d₁]}, {ty [B,d₁]}) -> {tyI1 [B,d₁]}\n" ++
      s!"    %dy0 = stablehlo.select %bm0, %dx1, %bz0 : {tyI1 [B,d₁]}, {ty [B,d₁]}\n" ++
      "    // ── param grads (dot over batch / reduce) reading the activations ──\n" ++
      "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      dg "%dW2" na1 ndy "0" "0" (ty [B,d₂]) (ty [B,d₃]) (ty [d₂,d₃]) ++ reduce "%db2" ndy d₃ ++
      dg "%dW1" na0 "%dy1" "0" "0" (ty [B,d₁]) (ty [B,d₂]) (ty [d₁,d₂]) ++ reduce "%db1" "%dy1" d₂ ++
      dg "%dW0" "%x" "%dy0" "0" "0" (ty [B,d₀]) (ty [B,d₁]) (ty [d₀,d₁]) ++ reduce "%db0" "%dy0" d₁ ++
      "    // ── SGD θ' = θ − lr·∇ ──\n" ++
      sgd "%W0" "%dW0" (ty [d₀,d₁]) ++ sgd "%b0" "%db0" (ty [d₁]) ++
      sgd "%W1" "%dW1" (ty [d₁,d₂]) ++ sgd "%b1" "%db1" (ty [d₂]) ++
      sgd "%W2" "%dW2" (ty [d₂,d₃]) ++ sgd "%b2" "%db2" (ty [d₃])
  let body : String := go.run' 0
  "module @m {\n" ++
  s!"  func.func @mlp_train_step(%x: {ty [B,d₀]}, %W0: {ty [d₀,d₁]}, %b0: {ty [d₁]}, %W1: {ty [d₁,d₂]}, %b1: {ty [d₂]}, %W2: {ty [d₂,d₃]}, %b2: {ty [d₃]}, %onehot: {ty [B,d₃]}) -> ({ty [d₀,d₁]}, {ty [d₁]}, {ty [d₁,d₂]}, {ty [d₂]}, {ty [d₂,d₃]}, {ty [d₃]}) " ++ "{\n" ++
  body ++
  s!"    return %W0n, %b0n, %W1n, %b1n, %W2n, %b2n : {ty [d₀,d₁]}, {ty [d₁]}, {ty [d₁,d₂]}, {ty [d₂]}, {ty [d₂,d₃]}, {ty [d₃]}\n" ++
  "  }\n}\n"

/-- **MLP train step rendered ENTIRELY from the verified AST.** Like
    `mlpTrainStepStructured` for the forward, but the backward chain
    (`dotOut`/`selectPos`) and the six parameter SGD updates (`weightSgd`/`biasSgd`)
    are now `pretty` of denoted `SHlo` nodes too — so every emitted line is
    `pretty(provenNode)`, and `MlpFaithfulPoC` proves each output's `den` = the
    certified loss-descent step. Cotangents `%dy`/`nc1`/`nc0` are rendered once and
    shared (operand leaves); operand/`lr`/weight VALUES are `skel`-erased, so these
    placeholders print identically to the live graphs the `den` theorems use. -/
def mlpTrainStepFaithfulV (B d₀ d₁ d₂ d₃ : Nat) (lrStr : String)
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) : String :=
  let z₀ : Vec d₀ := fun _ => 0
  let z₁ : Vec d₁ := fun _ => 0
  let z₂ : Vec d₂ := fun _ => 0
  let z₃ : Vec d₃ := fun _ => 0
  let act : StateM Nat (String × String × String × String × String × String × String) := do
    let (cp0, np0) ← pretty B (denseF "%W0" "%b0" W₀ b₀ (.operand "%x" x))
    let (ca0, na0) ← pretty B (.reluF (.operand np0 z₁))
    let (cp1, np1) ← pretty B (denseF "%W1" "%b1" W₁ b₁ (.operand na0 z₁))
    let (ca1, na1) ← pretty B (.reluF (.operand np1 z₂))
    let (clog, nlog) ← pretty B (denseF "%W2" "%b2" W₂ b₂ (.operand na1 z₂))
    let (cdy, ndy) ← pretty B (.sub (.softmaxDiv (.expe (.operand nlog z₃))) (.operand "%onehot" z₃))
    let (cc1, nc1) ← pretty B (.selectPos np1 z₂ (.dotOut "%W2" W₂ (.operand ndy z₃)))
    let (cc0, nc0) ← pretty B (.selectPos np0 z₁ (.dotOut "%W1" W₁ (.operand nc1 z₂)))
    let (cW2, nW2) ← pretty B (SHlo.weightSgd na1 "%W2" lrStr z₂ W₂ 0 (.operand ndy z₃))
    let (cb2, nb2) ← pretty B (SHlo.biasSgd "%b2" lrStr z₃ 0 (.operand ndy z₃))
    let (cW1, nW1) ← pretty B (SHlo.weightSgd na0 "%W1" lrStr z₁ W₁ 0 (.operand nc1 z₂))
    let (cb1, nb1) ← pretty B (SHlo.biasSgd "%b1" lrStr z₂ 0 (.operand nc1 z₂))
    let (cW0, nW0) ← pretty B (SHlo.weightSgd "%x" "%W0" lrStr z₀ W₀ 0 (.operand nc0 z₁))
    let (cb0, nb0) ← pretty B (SHlo.biasSgd "%b0" lrStr z₁ 0 (.operand nc0 z₁))
    pure (cp0 ++ ca0 ++ cp1 ++ ca1 ++ clog ++ cdy ++ cc1 ++ cc0 ++
            cW2 ++ cb2 ++ cW1 ++ cb1 ++ cW0 ++ cb0,
          nW0, nb0, nW1, nb1, nW2, nb2)
  let (body, nW0, nb0, nW1, nb1, nW2, nb2) := act.run' 0
  "module @m {\n" ++
  s!"  func.func @mlp_train_step(%x: {ty [B,d₀]}, %W0: {ty [d₀,d₁]}, %b0: {ty [d₁]}, %W1: {ty [d₁,d₂]}, %b1: {ty [d₂]}, %W2: {ty [d₂,d₃]}, %b2: {ty [d₃]}, %onehot: {ty [B,d₃]}) -> ({ty [d₀,d₁]}, {ty [d₁]}, {ty [d₁,d₂]}, {ty [d₂]}, {ty [d₂,d₃]}, {ty [d₃]}) " ++ "{\n" ++
  "    // ── mlp train step: every line is pretty(verified AST node) ──\n" ++
  body ++
  s!"    return {nW0}, {nb0}, {nW1}, {nb1}, {nW2}, {nb2} : {ty [d₀,d₁]}, {ty [d₁]}, {ty [d₁,d₂]}, {ty [d₂]}, {ty [d₂,d₃]}, {ty [d₃]}\n" ++
  "  }\n}\n"

-- Regenerate `verified_mlir/mlp_train_step.mlir` (what MainMnistMlpVerified trains on)
-- from the faithful renderer; the den-certified proofs live in MlpFaithfulPoC.lean.
#eval IO.FS.writeFile "verified_mlir/mlp_train_step.mlir"
  (mlpTrainStepFaithfulV 128 784 512 512 10 "0.00078125"
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ => 0))

end Proofs.StableHLO
