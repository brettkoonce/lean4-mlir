import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! # E3 — squeeze-excite module renderer (fwd + 2-path backward) + iree

The one genuinely-new EfficientNet combinator: a fan-out gate sub-network whose
scalar-per-channel output is multiplied back into the main path. De-risked here as
a standalone `@se_fwd` / `@se_back` pair on `iree-compile` BEFORE wiring into MBConv
(E4). Batched `[BS,C,H,W]` versions of IRPrint's `renderSEGate`/`renderSEGateBack`
(the GPU-validated reference), in the ch7 trainer's hand-rendered-fragment style —
each line is what the proven-faithful emitter (`emitTok`) produces for that op:

  forward  se = x ⊙ broadcast(σ(W₂·swish(W₁·GAP(x)+b₁)+b₂))   (`seGate`, `seBlock`)
    squeeze GAP[2,3]÷H·W → dense₁ C→r → swish → dense₂ r→C → sigmoid gate → bcast×x

  backward (the elemwise-product VJP, `seBlock_has_vjp`: gate⊙dy + gate.back(x,x⊙dy))
    main : broadcast(gate) ⊙ dse
    gate : Σ_spatial(x⊙dse) → sigmoid'(gate·(1−gate)) → dense₂ back (+dWs₂,+dbs₂)
           → swish back → dense₁ back (+dWs₁,+dbs₁) → GAP back (bcast ÷H·W)
    dx   = main + gate-path   (the two fan-in contributions summed)

The squeeze GAP-back (`÷H·W` then broadcast) and the gate broadcast-back (sum over
spatial) are adjoints of each other's forward steps (`broadcastFlat_has_vjp`).

Run (rocm): export IREE_BACKEND=rocm; lake env lean tests/TestSE.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 4
private def C  : Nat := 64
private def H  : Nat := 8
private def W  : Nat := 8
private def R  : Nat := 16   -- SE bottleneck width (r = C/4 typical)

/-- **SE forward** on `%x` `[BS,C,H,W]`, prefix `p`. Produces `%{p}se` and saves the
    backward's reused activations `%{p}sq` `[BS,C]`, `%{p}ex` `[BS,r]` (pre-swish),
    `%{p}a1` `[BS,r]` (post-swish), `%{p}gate` `[BS,C]`. `%sc` (f32 0) must be in scope. -/
private def seFwd (p x Ws1 bs1 Ws2 bs2 : String) (c h w r : Nat) : String :=
  -- squeeze: global average pool over the spatial axes [2,3]
  s!"    %{p}sqs = stablehlo.reduce({x} init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,c,h,w]}, tensor<f32>) -> {ty [BS,c]}\n" ++
  s!"    %{p}sqnf = stablehlo.constant dense<{h*w}.0> : {ty [BS,c]}\n" ++
  s!"    %{p}sq = stablehlo.divide %{p}sqs, %{p}sqnf : {ty [BS,c]}\n" ++
  -- excite dense₁ (C→r) + bias
  s!"    %{p}exd = stablehlo.dot_general %{p}sq, {Ws1}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,c]}, {ty [c,r]}) -> {ty [BS,r]}\n" ++
  s!"    %{p}exbb = stablehlo.broadcast_in_dim {bs1}, dims = [1] : ({ty [r]}) -> {ty [BS,r]}\n" ++
  s!"    %{p}ex = stablehlo.add %{p}exd, %{p}exbb : {ty [BS,r]}\n" ++
  -- swish (= emitTok swishF: logistic then multiply)
  s!"    %{p}a1s = stablehlo.logistic %{p}ex : {ty [BS,r]}\n" ++
  s!"    %{p}a1 = stablehlo.multiply %{p}ex, %{p}a1s : {ty [BS,r]}\n" ++
  -- excite dense₂ (r→C) + bias
  s!"    %{p}h2d = stablehlo.dot_general %{p}a1, {Ws2}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,r]}, {ty [r,c]}) -> {ty [BS,c]}\n" ++
  s!"    %{p}h2bb = stablehlo.broadcast_in_dim {bs2}, dims = [1] : ({ty [c]}) -> {ty [BS,c]}\n" ++
  s!"    %{p}h2 = stablehlo.add %{p}h2d, %{p}h2bb : {ty [BS,c]}\n" ++
  -- sigmoid gate (= emitTok sigmoidF: logistic)
  s!"    %{p}gate = stablehlo.logistic %{p}h2 : {ty [BS,c]}\n" ++
  -- broadcast the per-channel gate back to spatial and multiply into the main path
  s!"    %{p}gb = stablehlo.broadcast_in_dim %{p}gate, dims = [0, 1] : ({ty [BS,c]}) -> {ty [BS,c,h,w]}\n" ++
  s!"    %{p}se = stablehlo.multiply {x}, %{p}gb : {ty [BS,c,h,w]}\n"

/-- **SE backward** — input cotangent of `se = x ⊙ broadcast(gate(x))` (the product-
    rule fan-in) + the two SE dense weight/bias grads. `dse` `[BS,C,H,W]`; reuses the
    saved `%{p}sq`,`%{p}ex`,`%{p}a1`,`%{p}gate` + the weights `Ws1`,`Ws2` + `%sc`.
    Produces `%{p}dds` `[BS,C,H,W]`, `%{p}dWs1` `[C,r]`, `%{p}dbs1` `[r]`,
    `%{p}dWs2` `[r,C]`, `%{p}dbs2` `[C]`. -/
private def seBack (p x dse Ws1 Ws2 : String) (c h w r : Nat) : String :=
  -- main-path factor: broadcast(gate) ⊙ dse
  s!"    %{p}gb2 = stablehlo.broadcast_in_dim %{p}gate, dims = [0, 1] : ({ty [BS,c]}) -> {ty [BS,c,h,w]}\n" ++
  s!"    %{p}dleft = stablehlo.multiply %{p}gb2, {dse} : {ty [BS,c,h,w]}\n" ++
  -- gate path: dgate = Σ_spatial(x ⊙ dse)  (adjoint of the step-6 broadcast)
  s!"    %{p}xdse = stablehlo.multiply {x}, {dse} : {ty [BS,c,h,w]}\n" ++
  s!"    %{p}dgate = stablehlo.reduce(%{p}xdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,c,h,w]}, tensor<f32>) -> {ty [BS,c]}\n" ++
  -- sigmoid' via the saved gate: σ'(h2) = gate·(1−gate)  (= emitTok sigmoidBack, gate=σ(h2))
  s!"    %{p}one = stablehlo.constant dense<1.0> : {ty [BS,c]}\n" ++
  s!"    %{p}omg = stablehlo.subtract %{p}one, %{p}gate : {ty [BS,c]}\n" ++
  s!"    %{p}sg = stablehlo.multiply %{p}gate, %{p}omg : {ty [BS,c]}\n" ++
  s!"    %{p}dh2 = stablehlo.multiply %{p}dgate, %{p}sg : {ty [BS,c]}\n" ++
  -- dense₂ back: input grad da1 = dh2·Ws₂ᵀ, weight grad dWs₂ = a1ᵀ·dh2, bias grad dbs₂
  s!"    %{p}da1 = stablehlo.dot_general %{p}dh2, {Ws2}, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [BS,c]}, {ty [r,c]}) -> {ty [BS,r]}\n" ++
  s!"    %{p}dWs2 = stablehlo.dot_general %{p}a1, %{p}dh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,r]}, {ty [BS,c]}) -> {ty [r,c]}\n" ++
  s!"    %{p}dbs2 = stablehlo.reduce(%{p}dh2 init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,c]}, tensor<f32>) -> {ty [c]}\n" ++
  -- swish back through the excite mid `ex` (= emitTok swishBack: dy ⊙ σ(1+x(1−σ)))
  s!"    %{p}dexs = stablehlo.logistic %{p}ex : {ty [BS,r]}\n" ++
  s!"    %{p}dexone = stablehlo.constant dense<1.0> : {ty [BS,r]}\n" ++
  s!"    %{p}dexom = stablehlo.subtract %{p}dexone, %{p}dexs : {ty [BS,r]}\n" ++
  s!"    %{p}dexxom = stablehlo.multiply %{p}ex, %{p}dexom : {ty [BS,r]}\n" ++
  s!"    %{p}dexin = stablehlo.add %{p}dexone, %{p}dexxom : {ty [BS,r]}\n" ++
  s!"    %{p}dexsp = stablehlo.multiply %{p}dexs, %{p}dexin : {ty [BS,r]}\n" ++
  s!"    %{p}dex = stablehlo.multiply %{p}da1, %{p}dexsp : {ty [BS,r]}\n" ++
  -- dense₁ back: input grad dsq = dex·Ws₁ᵀ, weight grad dWs₁ = sqᵀ·dex, bias grad dbs₁
  s!"    %{p}dsq = stablehlo.dot_general %{p}dex, {Ws1}, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [BS,r]}, {ty [c,r]}) -> {ty [BS,c]}\n" ++
  s!"    %{p}dWs1 = stablehlo.dot_general %{p}sq, %{p}dex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,c]}, {ty [BS,r]}) -> {ty [c,r]}\n" ++
  s!"    %{p}dbs1 = stablehlo.reduce(%{p}dex init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,r]}, tensor<f32>) -> {ty [r]}\n" ++
  -- GAP back: broadcast(dsq / (H·W)) over spatial (the squeeze adjoint, ÷H·W on this side)
  s!"    %{p}dsqnf = stablehlo.constant dense<{h*w}.0> : {ty [BS,c]}\n" ++
  s!"    %{p}dsqd = stablehlo.divide %{p}dsq, %{p}dsqnf : {ty [BS,c]}\n" ++
  s!"    %{p}dgsp = stablehlo.broadcast_in_dim %{p}dsqd, dims = [0, 1] : ({ty [BS,c]}) -> {ty [BS,c,h,w]}\n" ++
  -- sum the two fan-in contributions → the SE input cotangent
  s!"    %{p}dds = stablehlo.add %{p}dleft, %{p}dgsp : {ty [BS,c,h,w]}\n"

private def seSig : String :=
  s!"%x: {ty [BS,C,H,W]}, %Ws1: {ty [C,R]}, %bs1: {ty [R]}, %Ws2: {ty [R,C]}, %bs2: {ty [C]}"

private def fwdModule : String :=
  "module @m {\n" ++
  s!"  func.func @se_fwd({seSig}) -> {ty [BS,C,H,W]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  seFwd "se" "%x" "%Ws1" "%bs1" "%Ws2" "%bs2" C H W R ++
  s!"    return %sese : {ty [BS,C,H,W]}\n" ++
  "  }\n}\n"

private def backModule : String :=
  let retTy := String.intercalate ", " [ty [BS,C,H,W], ty [C,R], ty [R], ty [R,C], ty [C]]
  "module @m {\n" ++
  s!"  func.func @se_back({seSig}, %dse: {ty [BS,C,H,W]}) -> ({retTy}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  -- recompute the forward to recover the saved activations (gate/ex/sq/a1)
  seFwd "se" "%x" "%Ws1" "%bs1" "%Ws2" "%bs2" C H W R ++
  seBack "se" "%x" "%dse" "%Ws1" "%Ws2" C H W R ++
  s!"    return %sedds, %sedWs1, %sedbs1, %sedWs2, %sedbs2 : {retTy}\n" ++
  "  }\n}\n"

private def compileCheck (name body : String) : IO Unit := do
  IO.FS.createDirAll ".lake/build"
  let path := s!".lake/build/{name}.mlir"
  IO.FS.writeFile path body
  let cargs ← ireeCompileArgs path s!".lake/build/{name}.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"[{name}] iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println s!"[{name}] iree-compile OK → .lake/build/{name}.vmfb"

def main : IO Unit := do
  IO.println "── @se_fwd ──"
  IO.println fwdModule
  IO.println "── @se_back ──"
  IO.println backModule
  compileCheck "se_fwd" fwdModule
  compileCheck "se_back" backModule

#eval main
