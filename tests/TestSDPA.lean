import LeanMlir.Proofs.Codegen.StableHLOPretty
import LeanMlir.Types
import LeanMlir.GradcheckHelpers

/-! # ch10 V2 — single-head scaled-dot-product-attention renderer (fwd + 3-path back)

The highest-risk hand-wiring of the ViT chapter: SDPA's backward threads three
paths (dQ/dK/dV) through a per-row softmax VJP. De-risked here STANDALONE on
`[n,d]` (no batch — the proof's `Mat n d`), `iree-compile`d, AND numerically
gradchecked (see `LeanMlir/Proofs/check_jacobians.py`, sdpaBackQ/K/V) BEFORE
wiring into multi-head (TestMHSA) — compile-clean ≠ correct for a transpose/axis bug.

Fragments mirror IRPrint's GPU-validated `sdpaFwdModule`/`sdpaBackModule`, each line
what the proven-faithful emitter produces: the matmuls are `dot_general` (proven
dense), the softmax is the V1 row-softmax pattern (`softmaxRowF`/`softmaxRowBack`),
the `1/√d` scale a `multiply`. The proven backward (`sdpa_back_{Q,K,V}_correct`,
Attention.lean):
  dWeights = dOut·Vᵀ,  dV = weightsᵀ·dOut,
  dScaled  = rowsoftmax-VJP(weights, dWeights) = weights⊙(dWeights − ⟨weights,dWeights⟩),
  dScores  = dScaled·scale,  dQ = dScores·K,  dK = dScoresᵀ·Q.

The gradcheck is the **adjoint/finite-difference dot-product test**, run entirely
in Lean4 (no numpy): it shells out to `iree-run-module` to execute the compiled
`@sdpa_fwd`/`@sdpa_back` `.vmfb`, then checks
  ⟨dQ,vQ⟩+⟨dK,vK⟩+⟨dV,vV⟩  ≈  (Φ(+ε) − Φ(−ε)) / 2ε,   Φ(s) := ⟨fwd(in + s·v), dOut⟩
for random directions v — validating all three backward paths at once (the VJP
is J·ᵀ, so ⟨Jᵀ dOut, v⟩ = ⟨dOut, J v⟩ = the directional derivative).

Run (needs iree-compile on PATH):
  lake env lean tests/TestSDPA.lean          # renders, iree-compiles, AND gradchecks
-/

open Proofs Proofs.StableHLO
open ViTGradcheck

private def Nn : Nat := 5    -- tokens (the N of attention)
private def Dd : Nat := 4    -- head dim
private def scaleStr : String := "0.5"   -- 1/√d = 1/√4 = 0.5 (exact)

/-- **Single-head SDPA forward** `softmax(QKᵀ·scale)·V`, prefix `p`, over `[n,d]`.
    Produces `%{p}out` `[n,d]` and saves `%{p}weights` `[n,n]` (the row-softmax
    probabilities) for the backward. `%sc` (f32 0) must be in scope. -/
private def sdpaFwd (p Q K V : String) (n d : Nat) (scale : String) : String :=
  -- scores = Q·Kᵀ  (contract the d axis)
  s!"    %{p}scores = stablehlo.dot_general {Q}, {K}, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [n,d]}, {ty [n,d]}) -> {ty [n,n]}\n" ++
  s!"    %{p}scl = stablehlo.constant dense<{scale}> : {ty [n,n]}\n" ++
  s!"    %{p}scaled = stablehlo.multiply %{p}scores, %{p}scl : {ty [n,n]}\n" ++
  -- row-softmax over the LAST axis [1] (V1 pattern, plain exp/sum)
  s!"    %{p}se = stablehlo.exponential %{p}scaled : {ty [n,n]}\n" ++
  s!"    %{p}ssum = stablehlo.reduce(%{p}se init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [n,n]}, tensor<f32>) -> {ty [n]}\n" ++
  s!"    %{p}ssumb = stablehlo.broadcast_in_dim %{p}ssum, dims = [0] : ({ty [n]}) -> {ty [n,n]}\n" ++
  s!"    %{p}weights = stablehlo.divide %{p}se, %{p}ssumb : {ty [n,n]}\n" ++
  -- out = weights·V  (contract the key axis)
  s!"    %{p}out = stablehlo.dot_general %{p}weights, {V}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [n,n]}, {ty [n,d]}) -> {ty [n,d]}\n"

/-- **Single-head SDPA backward** (the 3 proven input grads), prefix `p`. Requires
    `%{p}weights` already in scope (recompute via `sdpaFwd` first). Produces
    `%{p}dQ` `%{p}dK` `%{p}dV` `[n,d]`. `%sc` (f32 0) must be in scope. -/
private def sdpaBack (p Q K V dOut : String) (n d : Nat) (scale : String) : String :=
  -- dWeights = dOut·Vᵀ  (contract d);   dV = weightsᵀ·dOut  (contract the query axis 0)
  s!"    %{p}dWeights = stablehlo.dot_general {dOut}, {V}, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [n,d]}, {ty [n,d]}) -> {ty [n,n]}\n" ++
  s!"    %{p}dV = stablehlo.dot_general %{p}weights, {dOut}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [n,n]}, {ty [n,d]}) -> {ty [n,d]}\n" ++
  -- row-softmax VJP: dScaled = weights ⊙ (dWeights − ⟨weights,dWeights⟩_row)
  s!"    %{p}pdw = stablehlo.multiply %{p}weights, %{p}dWeights : {ty [n,n]}\n" ++
  s!"    %{p}srow = stablehlo.reduce(%{p}pdw init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [n,n]}, tensor<f32>) -> {ty [n]}\n" ++
  s!"    %{p}srowb = stablehlo.broadcast_in_dim %{p}srow, dims = [0] : ({ty [n]}) -> {ty [n,n]}\n" ++
  s!"    %{p}diff = stablehlo.subtract %{p}dWeights, %{p}srowb : {ty [n,n]}\n" ++
  s!"    %{p}dScaled = stablehlo.multiply %{p}weights, %{p}diff : {ty [n,n]}\n" ++
  -- undo the scale, then dQ = dScores·K,  dK = dScoresᵀ·Q
  s!"    %{p}sclb = stablehlo.constant dense<{scale}> : {ty [n,n]}\n" ++
  s!"    %{p}dScores = stablehlo.multiply %{p}dScaled, %{p}sclb : {ty [n,n]}\n" ++
  s!"    %{p}dQ = stablehlo.dot_general %{p}dScores, {K}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [n,n]}, {ty [n,d]}) -> {ty [n,d]}\n" ++
  s!"    %{p}dK = stablehlo.dot_general %{p}dScores, {Q}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [n,n]}, {ty [n,d]}) -> {ty [n,d]}\n"

private def sdpaSig : String := s!"%Q: {ty [Nn,Dd]}, %K: {ty [Nn,Dd]}, %V: {ty [Nn,Dd]}"

private def fwdModule : String :=
  "module @m {\n" ++
  s!"  func.func @sdpa_fwd({sdpaSig}) -> {ty [Nn,Dd]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  sdpaFwd "a" "%Q" "%K" "%V" Nn Dd scaleStr ++
  s!"    return %aout : {ty [Nn,Dd]}\n" ++ "  }\n}\n"

private def backModule : String :=
  let retTy := String.intercalate ", " [ty [Nn,Dd], ty [Nn,Dd], ty [Nn,Dd]]
  "module @m {\n" ++
  s!"  func.func @sdpa_back({sdpaSig}, %dOut: {ty [Nn,Dd]}) -> ({retTy}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  sdpaFwd "a" "%Q" "%K" "%V" Nn Dd scaleStr ++         -- recompute weights
  sdpaBack "a" "%Q" "%K" "%V" "%dOut" Nn Dd scaleStr ++
  s!"    return %adQ, %adK, %adV : {retTy}\n" ++ "  }\n}\n"


-- ════════════════════════════════════════════════════════════════
-- § Numerical gradcheck (all in Lean4; shells out to iree-run-module)
-- ════════════════════════════════════════════════════════════════

/-- The adjoint/finite-difference gradcheck of the compiled SDPA fwd/back: all three backward
    paths (dQ, dK, dV) against one directional derivative. -/
private def gradcheck : IO Unit := do
  let sh := s!"{Nn}x{Dd}xf32"
  let _ ← adjointGradcheck "sdpa gradcheck" ".lake/build/sdpa_fwd.vmfb" "sdpa_fwd"
    ".lake/build/sdpa_back.vmfb" "sdpa_back" [sh, sh, sh] [Nn*Dd, Nn*Dd, Nn*Dd] sh (Nn*Dd)

def main : IO Unit := do
  IO.println "── @sdpa_fwd ──"
  IO.println fwdModule
  IO.println "── @sdpa_back ──"
  IO.println backModule
  compileCheck "sdpa_fwd" fwdModule
  compileCheck "sdpa_back" backModule
  gradcheck

#eval main
