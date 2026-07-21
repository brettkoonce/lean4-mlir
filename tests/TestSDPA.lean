import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Types

/-! # ch10 V2 — single-head scaled-dot-product-attention renderer (fwd + 3-path back)

The highest-risk hand-wiring of the ViT chapter: SDPA's backward threads three
paths (dQ/dK/dV) through a per-row softmax VJP. De-risked here STANDALONE on
`[n,d]` (no batch — the proof's `Mat n d`), `iree-compile`d, AND numerically
gradchecked (see `tests/gradcheck_sdpa.py`, the adjoint dot-product test) BEFORE
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

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"
  export LD_LIBRARY_PATH="$PWD/ffi:/opt/rocm/lib:$LD_LIBRARY_PATH"
  export IREE_BACKEND=rocm
  lake env lean tests/TestSDPA.lean          # renders, iree-compiles, AND gradchecks
-/

open Proofs Proofs.StableHLO

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

-- ════════════════════════════════════════════════════════════════
-- § Numerical gradcheck (all in Lean4; shells out to iree-run-module)
-- ════════════════════════════════════════════════════════════════

/-- `10^k` as a `Float` (k may be negative). -/
private def pow10 (k : Int) : Float :=
  if k ≥ 0 then (List.range k.toNat).foldl (fun a _ => a * 10.0) 1.0
  else (List.range (-k).toNat).foldl (fun a _ => a / 10.0) 1.0

private def digitsToNat (cs : List Char) : Nat :=
  cs.foldl (fun acc c => acc * 10 + (c.toNat - '0'.toNat)) 0

/-- Split a `List Char` at the first occurrence of `c` (the separator dropped). -/
private def splitAtChar (c : Char) (xs : List Char) : (List Char × Option (List Char)) :=
  match xs.span (· != c) with
  | (pre, [])        => (pre, none)
  | (pre, _ :: post) => (pre, some post)

/-- Parse one iree-printed float token (`-0.00623606`, `1.3e-05`, `42`), entirely
    over `List Char` (robust to the String/Slice API). -/
private def parseFloat (tok : String) : Float := Id.run do
  let cs0 := (tok.toList).map (fun c => if c == 'E' then 'e' else c)
  let (neg, cs) := match cs0 with
    | '-' :: rest => (true, rest)
    | '+' :: rest => (false, rest)
    | _           => (false, cs0)
  let (mantCs, expCsOpt) := splitAtChar 'e' cs
  let expVal : Int := match expCsOpt with
    | none              => 0
    | some ('-' :: ds)  => -(Int.ofNat (digitsToNat ds))
    | some ('+' :: ds)  => Int.ofNat (digitsToNat ds)
    | some ds           => Int.ofNat (digitsToNat ds)
  let (intCs, fracCsOpt) := splitAtChar '.' mantCs
  let ip := Float.ofNat (digitsToNat intCs)
  let mant := match fracCsOpt with
    | none      => ip
    | some fracCs => ip + Float.ofNat (digitsToNat fracCs) * pow10 (-(fracCs.length : Int))
  let v := mant * pow10 expVal
  return (if neg then -v else v)

/-- Extract the parsed result buffers (in `result[i]` order) from an
    iree-run-module stdout: each value line is `…xf32=[a b][c d]…`. -/
private def parseResults (out : String) : Array (Array Float) := Id.run do
  let mut res : Array (Array Float) := #[]
  for line in out.splitOn "\n" do
    if let some idx := (line.splitOn "f32=")[1]? then
      -- strip brackets → whitespace, split, parse
      let cleaned := idx.map (fun c => if c == '[' || c == ']' then ' ' else c)
      let toks := (cleaned.splitOn " ").filter (fun t => !t.isEmpty)
      res := res.push ((toks.map parseFloat).toArray)
  return res

/-- Format a flat `Array Float` as an iree-run-module `--input=shape=v1 v2 …`. -/
private def fmtInput (shape : String) (xs : Array Float) : String :=
  s!"--input={shape}=" ++ String.intercalate " " (xs.toList.map toString)

/-- Run a compiled `.vmfb` function on flat `[n,d]` inputs, return result buffers. -/
private def runFn (vmfb fn : String) (inputs : Array (Array Float)) : IO (Array (Array Float)) := do
  let args := #[s!"--module={vmfb}", "--device=hip", s!"--function={fn}"] ++
              inputs.map (fmtInput s!"{Nn}x{Dd}xf32")
  let r ← IO.Process.output { cmd := "iree-run-module", args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"[run {fn}] FAILED:\n{r.stderr.take 1500}"
    return #[]
  return parseResults r.stdout

/-- Deterministic LCG pseudo-random `Array Float` in `[-1,1]`, length `n`. -/
private def randVec (seed n : Nat) : Array Float := Id.run do
  let mut s : Nat := seed * 2654435761 + 12345
  let mut out : Array Float := #[]
  for _ in [0:n] do
    s := (s * 1103515245 + 12345) % 2147483648
    out := out.push (2.0 * (Float.ofNat s / 2147483648.0) - 1.0)
  return out

private def dot (a b : Array Float) : Float :=
  (a.zip b).foldl (fun acc (x, y) => acc + x * y) 0.0

private def axpy (a : Float) (x y : Array Float) : Array Float :=  -- y + a·x
  (y.zip x).map (fun (yi, xi) => yi + a * xi)

/-- The adjoint/finite-difference gradcheck of the compiled SDPA fwd/back. -/
private def gradcheck : IO Unit := do
  let len := Nn * Dd
  let Q := randVec 1 len;  let K := randVec 2 len;  let V := randVec 3 len
  let dOut := randVec 4 len
  let vQ := randVec 5 len;  let vK := randVec 6 len;  let vV := randVec 7 len
  let eps : Float := 1.0e-3
  -- backward: dQ, dK, dV
  let back ← runFn ".lake/build/sdpa_back.vmfb" "sdpa_back" #[Q, K, V, dOut]
  if back.size != 3 then
    IO.eprintln s!"[gradcheck] expected 3 back results, got {back.size}"; return
  let dQ := back[0]!; let dK := back[1]!; let dV := back[2]!
  let lhs := dot dQ vQ + dot dK vK + dot dV vV
  -- Φ(s) = ⟨fwd(Q+s·vQ, K+s·vK, V+s·vV), dOut⟩
  let phi (s : Float) : IO Float := do
    let f ← runFn ".lake/build/sdpa_fwd.vmfb" "sdpa_fwd"
              #[axpy s vQ Q, axpy s vK K, axpy s vV V]
    if f.size != 1 then IO.eprintln "[gradcheck] fwd result missing"; return 0.0
    return dot f[0]! dOut
  let phiP ← phi eps
  let phiM ← phi (-eps)
  let rhs := (phiP - phiM) / (2.0 * eps)
  let absErr := Float.abs (lhs - rhs)
  let relErr := absErr / (Float.abs rhs + 1.0e-9)
  IO.println s!"[gradcheck] adjoint lhs = {lhs}"
  IO.println s!"[gradcheck] finite-diff rhs = {rhs}"
  IO.println s!"[gradcheck] abs err = {absErr}   rel err = {relErr}"
  if relErr < 1.0e-2 then
    IO.println "[gradcheck] ✅ PASS (SDPA 3-path backward matches finite differences)"
  else
    IO.eprintln "[gradcheck] ❌ FAIL — backward does NOT match finite differences"

def main : IO Unit := do
  IO.println "── @sdpa_fwd ──"
  IO.println fwdModule
  IO.println "── @sdpa_back ──"
  IO.println backModule
  compileCheck "sdpa_fwd" fwdModule
  compileCheck "sdpa_back" backModule
  gradcheck

#eval main
