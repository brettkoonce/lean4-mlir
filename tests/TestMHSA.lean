import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Types

/-! # ch10 V3 — multi-head self-attention renderer (fwd + full backward)

The big new ViT combinator (the analogue of ch8's SE module): MHSA = QKV proj →
reshape/transpose to H heads → BATCHED per-head SDPA (V1 row-softmax + V2 SDPA, now
with `batching_dims=[0,1]` over (batch, heads)) → concat heads → out-proj. De-risked
STANDALONE on `[B,N,D]`, `iree-compile`d, AND numerically gradchecked (adjoint test,
all in Lean4) over EVERY input (x + Wq/bq/Wk/bk/Wv/bv/Wo/bo) BEFORE wiring into the
block (TestViTBlock). Forward shapes mirror MlirCodegen's `emitMHSAForward` (the GPU
reference); softmax is the V1 plain exp/sum (NO max-shift) matching the proven `softmax`.

Backward (each piece proven per-op): out-proj dense back → reshape/transpose back →
A=W·Vh back (dW=dA·Vhᵀ, dVh=Wᵀ·dA) → row-softmax VJP → undo scale → S=Qh·Khᵀ back
(dQh=dS·Kh, dKh=dSᵀ·Qh) → reshape/transpose back → QKV dense backs → x fan-in (Q,K,V
all read x, so dx = dxQ+dxK+dxV). The reshape↔transpose pair (dims=[0,2,1,3], an
involution) must be exact inverses — a wrong perm compiles fine but trains dead.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"
  export LD_LIBRARY_PATH="$PWD/ffi:/opt/rocm/lib:$LD_LIBRARY_PATH"
  export IREE_BACKEND=rocm
  lake env lean tests/TestMHSA.lean          # renders, iree-compiles, AND gradchecks
-/

open Proofs Proofs.StableHLO

private def Bb : Nat := 2     -- batch
private def Nn : Nat := 3     -- tokens
private def Dd : Nat := 4     -- model dim
private def Hh : Nat := 2     -- heads
private def Dh : Nat := 2     -- head dim (= D/H)
private def scaleStr : String := "0.7071067811865476"   -- 1/√2

/-- **MHSA forward**, prefix `p`, over `[B,N,D]`. Saves `%{p}W` (softmax weights
    `[B,H,N,N]`), `%{p}Qh`/`%{p}Kh`/`%{p}Vh` `[B,H,N,dh]`, `%{p}P` `[B,N,D]` (the
    concat-heads pre-out-proj) for the backward. `%sc` (f32 0) must be in scope. -/
private def mhsaFwd (p x Wq bq Wk bk Wv bv Wo bo : String)
    (b n d h dh : Nat) (scale : String) : String :=
  let proj (nm W bias : String) : String :=  -- dense [B,N,d]·[d,d]+bias → [B,N,d]
    s!"    %{p}{nm}d = stablehlo.dot_general {x}, {W}, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : ({ty [b,n,d]}, {ty [d,d]}) -> {ty [b,n,d]}\n" ++
    s!"    %{p}{nm}bb = stablehlo.broadcast_in_dim {bias}, dims = [2] : ({ty [d]}) -> {ty [b,n,d]}\n" ++
    s!"    %{p}{nm} = stablehlo.add %{p}{nm}d, %{p}{nm}bb : {ty [b,n,d]}\n"
  let split (src dst : String) : String :=  -- [B,N,d] → [B,N,h,dh] → [B,h,N,dh]
    s!"    %{p}{dst}r = stablehlo.reshape %{p}{src} : ({ty [b,n,d]}) -> {ty [b,n,h,dh]}\n" ++
    s!"    %{p}{dst} = stablehlo.transpose %{p}{dst}r, dims = [0, 2, 1, 3] : ({ty [b,n,h,dh]}) -> {ty [b,h,n,dh]}\n"
  proj "Q" Wq bq ++ proj "K" Wk bk ++ proj "V" Wv bv ++
  split "Q" "Qh" ++ split "K" "Kh" ++ split "V" "Vh" ++
  -- scores = Qh·Khᵀ (batched over [b,h]), scale
  s!"    %{p}S = stablehlo.dot_general %{p}Qh, %{p}Kh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : ({ty [b,h,n,dh]}, {ty [b,h,n,dh]}) -> {ty [b,h,n,n]}\n" ++
  s!"    %{p}scl = stablehlo.constant dense<{scale}> : {ty [b,h,n,n]}\n" ++
  s!"    %{p}Ss = stablehlo.multiply %{p}S, %{p}scl : {ty [b,h,n,n]}\n" ++
  -- row-softmax over the LAST axis [3] (plain exp/sum)
  s!"    %{p}se = stablehlo.exponential %{p}Ss : {ty [b,h,n,n]}\n" ++
  s!"    %{p}sum = stablehlo.reduce(%{p}se init: %sc) applies stablehlo.add across dimensions = [3] : ({ty [b,h,n,n]}, tensor<f32>) -> {ty [b,h,n]}\n" ++
  s!"    %{p}sumb = stablehlo.broadcast_in_dim %{p}sum, dims = [0, 1, 2] : ({ty [b,h,n]}) -> {ty [b,h,n,n]}\n" ++
  s!"    %{p}W = stablehlo.divide %{p}se, %{p}sumb : {ty [b,h,n,n]}\n" ++
  -- A = W·Vh (batched), transpose+reshape back to [B,N,D]
  s!"    %{p}A = stablehlo.dot_general %{p}W, %{p}Vh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : ({ty [b,h,n,n]}, {ty [b,h,n,dh]}) -> {ty [b,h,n,dh]}\n" ++
  s!"    %{p}AT = stablehlo.transpose %{p}A, dims = [0, 2, 1, 3] : ({ty [b,h,n,dh]}) -> {ty [b,n,h,dh]}\n" ++
  s!"    %{p}P = stablehlo.reshape %{p}AT : ({ty [b,n,h,dh]}) -> {ty [b,n,d]}\n" ++
  -- out projection
  s!"    %{p}od = stablehlo.dot_general %{p}P, {Wo}, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : ({ty [b,n,d]}, {ty [d,d]}) -> {ty [b,n,d]}\n" ++
  s!"    %{p}obb = stablehlo.broadcast_in_dim {bo}, dims = [2] : ({ty [d]}) -> {ty [b,n,d]}\n" ++
  s!"    %{p}O = stablehlo.add %{p}od, %{p}obb : {ty [b,n,d]}\n"

/-- **MHSA backward**, prefix `p`. Requires the forward saves (`%{p}W`,`%{p}Qh`,
    `%{p}Kh`,`%{p}Vh`,`%{p}P`) in scope. Produces `%{p}dx` `[B,N,D]` and the 8 param
    grads `%{p}dWq %{p}dbq %{p}dWk %{p}dbk %{p}dWv %{p}dbv %{p}dWo %{p}dbo`. -/
private def mhsaBack (p x Wq Wk Wv Wo dO : String)
    (b n d h dh : Nat) (scale : String) : String :=
  -- out-proj dense back
  s!"    %{p}dP = stablehlo.dot_general {dO}, {Wo}, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : ({ty [b,n,d]}, {ty [d,d]}) -> {ty [b,n,d]}\n" ++
  s!"    %{p}dWo = stablehlo.dot_general %{p}P, {dO}, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : ({ty [b,n,d]}, {ty [b,n,d]}) -> {ty [d,d]}\n" ++
  s!"    %{p}dbo = stablehlo.reduce({dO} init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({ty [b,n,d]}, tensor<f32>) -> {ty [d]}\n" ++
  -- reshape/transpose back: dA = transpose(reshape(dP))
  s!"    %{p}dPr = stablehlo.reshape %{p}dP : ({ty [b,n,d]}) -> {ty [b,n,h,dh]}\n" ++
  s!"    %{p}dA = stablehlo.transpose %{p}dPr, dims = [0, 2, 1, 3] : ({ty [b,n,h,dh]}) -> {ty [b,h,n,dh]}\n" ++
  -- A = W·Vh back:  dW = dA·Vhᵀ (contract dh),  dVh = Wᵀ·dA (contract the key axis n)
  s!"    %{p}dW = stablehlo.dot_general %{p}dA, %{p}Vh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : ({ty [b,h,n,dh]}, {ty [b,h,n,dh]}) -> {ty [b,h,n,n]}\n" ++
  s!"    %{p}dVh = stablehlo.dot_general %{p}W, %{p}dA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : ({ty [b,h,n,n]}, {ty [b,h,n,dh]}) -> {ty [b,h,n,dh]}\n" ++
  -- row-softmax VJP over [3]:  dSs = W ⊙ (dW − ⟨W,dW⟩_row)
  s!"    %{p}pdw = stablehlo.multiply %{p}W, %{p}dW : {ty [b,h,n,n]}\n" ++
  s!"    %{p}srow = stablehlo.reduce(%{p}pdw init: %sc) applies stablehlo.add across dimensions = [3] : ({ty [b,h,n,n]}, tensor<f32>) -> {ty [b,h,n]}\n" ++
  s!"    %{p}srowb = stablehlo.broadcast_in_dim %{p}srow, dims = [0, 1, 2] : ({ty [b,h,n]}) -> {ty [b,h,n,n]}\n" ++
  s!"    %{p}diff = stablehlo.subtract %{p}dW, %{p}srowb : {ty [b,h,n,n]}\n" ++
  s!"    %{p}dSs = stablehlo.multiply %{p}W, %{p}diff : {ty [b,h,n,n]}\n" ++
  -- undo scale
  s!"    %{p}sclb = stablehlo.constant dense<{scale}> : {ty [b,h,n,n]}\n" ++
  s!"    %{p}dS = stablehlo.multiply %{p}dSs, %{p}sclb : {ty [b,h,n,n]}\n" ++
  -- S = Qh·Khᵀ back:  dQh = dS·Kh (contract j),  dKh = dSᵀ·Qh (contract i)
  s!"    %{p}dQh = stablehlo.dot_general %{p}dS, %{p}Kh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : ({ty [b,h,n,n]}, {ty [b,h,n,dh]}) -> {ty [b,h,n,dh]}\n" ++
  s!"    %{p}dKh = stablehlo.dot_general %{p}dS, %{p}Qh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : ({ty [b,h,n,n]}, {ty [b,h,n,dh]}) -> {ty [b,h,n,dh]}\n" ++
  -- transpose/reshape each of dQh,dKh,dVh back to [B,N,D]
  (["Q", "K", "V"].foldl (fun acc nm =>
    acc ++
    s!"    %{p}d{nm}T = stablehlo.transpose %{p}d{nm}h, dims = [0, 2, 1, 3] : ({ty [b,h,n,dh]}) -> {ty [b,n,h,dh]}\n" ++
    s!"    %{p}d{nm} = stablehlo.reshape %{p}d{nm}T : ({ty [b,n,h,dh]}) -> {ty [b,n,d]}\n") "") ++
  -- QKV dense backs (dx_* = dProj·Wᵀ, dW = xᵀ·dProj, db = Σ dProj)
  (([("Q", Wq), ("K", Wk), ("V", Wv)] : List (String × String)).foldl (fun acc (nm, W) =>
    acc ++
    s!"    %{p}dx{nm} = stablehlo.dot_general %{p}d{nm}, {W}, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : ({ty [b,n,d]}, {ty [d,d]}) -> {ty [b,n,d]}\n" ++
    s!"    %{p}dW{nm} = stablehlo.dot_general {x}, %{p}d{nm}, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : ({ty [b,n,d]}, {ty [b,n,d]}) -> {ty [d,d]}\n" ++
    s!"    %{p}db{nm} = stablehlo.reduce(%{p}d{nm} init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({ty [b,n,d]}, tensor<f32>) -> {ty [d]}\n") "") ++
  -- x fan-in: x feeds Q, K, V
  s!"    %{p}dxa = stablehlo.add %{p}dxQ, %{p}dxK : {ty [b,n,d]}\n" ++
  s!"    %{p}dx = stablehlo.add %{p}dxa, %{p}dxV : {ty [b,n,d]}\n"

private def sig : String :=
  s!"%x: {ty [Bb,Nn,Dd]}, %Wq: {ty [Dd,Dd]}, %bq: {ty [Dd]}, %Wk: {ty [Dd,Dd]}, %bk: {ty [Dd]}, " ++
  s!"%Wv: {ty [Dd,Dd]}, %bv: {ty [Dd]}, %Wo: {ty [Dd,Dd]}, %bo: {ty [Dd]}"

private def fwdModule : String :=
  "module @m {\n" ++
  s!"  func.func @mhsa_fwd({sig}) -> {ty [Bb,Nn,Dd]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  mhsaFwd "a" "%x" "%Wq" "%bq" "%Wk" "%bk" "%Wv" "%bv" "%Wo" "%bo" Bb Nn Dd Hh Dh scaleStr ++
  s!"    return %aO : {ty [Bb,Nn,Dd]}\n" ++ "  }\n}\n"

private def backModule : String :=
  let retTy := String.intercalate ", "
    [ty [Bb,Nn,Dd], ty [Dd,Dd], ty [Dd], ty [Dd,Dd], ty [Dd], ty [Dd,Dd], ty [Dd], ty [Dd,Dd], ty [Dd]]
  "module @m {\n" ++
  s!"  func.func @mhsa_back({sig}, %dO: {ty [Bb,Nn,Dd]}) -> ({retTy}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  mhsaFwd "a" "%x" "%Wq" "%bq" "%Wk" "%bk" "%Wv" "%bv" "%Wo" "%bo" Bb Nn Dd Hh Dh scaleStr ++
  mhsaBack "a" "%x" "%Wq" "%Wk" "%Wv" "%Wo" "%dO" Bb Nn Dd Hh Dh scaleStr ++
  s!"    return %adx, %adWQ, %adbQ, %adWK, %adbK, %adWV, %adbV, %adWo, %adbo : {retTy}\n" ++ "  }\n}\n"

private def compileCheck (name body : String) : IO Bool := do
  IO.FS.createDirAll ".lake/build"
  let path := s!".lake/build/{name}.mlir"
  IO.FS.writeFile path body
  let cargs ← ireeCompileArgs path s!".lake/build/{name}.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"[{name}] iree-compile FAILED:\n{r.stderr.take 3000}"; return false
  else
    IO.println s!"[{name}] iree-compile OK → .lake/build/{name}.vmfb"; return true

-- ════════════════════════════════════════════════════════════════
-- § Lean4 gradcheck helpers (generalized over heterogeneous input shapes)
-- ════════════════════════════════════════════════════════════════

private def pow10 (k : Int) : Float :=
  if k ≥ 0 then (List.range k.toNat).foldl (fun a _ => a * 10.0) 1.0
  else (List.range (-k).toNat).foldl (fun a _ => a / 10.0) 1.0

private def digitsToNat (cs : List Char) : Nat :=
  cs.foldl (fun acc c => acc * 10 + (c.toNat - '0'.toNat)) 0

private def splitAtChar (c : Char) (xs : List Char) : (List Char × Option (List Char)) :=
  match xs.span (· != c) with
  | (pre, [])        => (pre, none)
  | (pre, _ :: post) => (pre, some post)

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
    | none        => ip
    | some fracCs => ip + Float.ofNat (digitsToNat fracCs) * pow10 (-(fracCs.length : Int))
  let v := mant * pow10 expVal
  return (if neg then -v else v)

private def parseResults (out : String) : Array (Array Float) := Id.run do
  let mut res : Array (Array Float) := #[]
  for line in out.splitOn "\n" do
    if let some idx := (line.splitOn "f32=")[1]? then
      let cleaned := idx.map (fun c => if c == '[' || c == ']' then ' ' else c)
      let toks := (cleaned.splitOn " ").filter (fun t => !t.isEmpty)
      res := res.push ((toks.map parseFloat).toArray)
  return res

/-- Run a compiled `.vmfb` function; `inputs` are `(shapeStr, flatValues)`. -/
private def runFn (vmfb fn : String) (inputs : List (String × Array Float)) : IO (Array (Array Float)) := do
  let inArgs := inputs.map (fun (sh, xs) =>
    s!"--input={sh}=" ++ String.intercalate " " (xs.toList.map toString))
  let args := #[s!"--module={vmfb}", "--device=hip", s!"--function={fn}"] ++ inArgs.toArray
  let r ← IO.Process.output { cmd := "iree-run-module", args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"[run {fn}] FAILED:\n{r.stderr.take 1500}"; return #[]
  return parseResults r.stdout

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

/-- Adjoint/finite-difference gradcheck of the full MHSA over every input. -/
private def gradcheck : IO Unit := do
  -- input shapes in the @mhsa_fwd arg order
  let shapes : List String :=
    [s!"{Bb}x{Nn}x{Dd}xf32", s!"{Dd}x{Dd}xf32", s!"{Dd}xf32", s!"{Dd}x{Dd}xf32", s!"{Dd}xf32",
     s!"{Dd}x{Dd}xf32", s!"{Dd}xf32", s!"{Dd}x{Dd}xf32", s!"{Dd}xf32"]
  let lens : List Nat := [Bb*Nn*Dd, Dd*Dd, Dd, Dd*Dd, Dd, Dd*Dd, Dd, Dd*Dd, Dd]
  -- random params + perturbation directions (distinct seeds)
  let params := (lens.zipIdx).map (fun (l, i) => randVec (100 + i) l)
  let dirs   := (lens.zipIdx).map (fun (l, i) => randVec (200 + i) l)
  let dO := randVec 42 (Bb*Nn*Dd)
  let ins := shapes.zip params
  -- backward: 9 grads in the same order as the 9 inputs
  let back ← runFn ".lake/build/mhsa_back.vmfb" "mhsa_back" (ins ++ [(s!"{Bb}x{Nn}x{Dd}xf32", dO)])
  if back.size != 9 then
    IO.eprintln s!"[gradcheck] expected 9 back results, got {back.size}"; return
  let lhs := ((back.toList.zip dirs).map (fun (g, v) => dot g v)).foldl (· + ·) 0.0
  -- Φ(s) = ⟨fwd(params + s·dirs), dO⟩
  let phi (s : Float) : IO Float := do
    let pert := (params.zip dirs).map (fun (pv, vv) => axpy s vv pv)
    let f ← runFn ".lake/build/mhsa_fwd.vmfb" "mhsa_fwd" (shapes.zip pert)
    if f.size != 1 then IO.eprintln "[gradcheck] fwd result missing"; return 0.0
    return dot f[0]! dO
  let eps : Float := 1.0e-3
  let phiP ← phi eps
  let phiM ← phi (-eps)
  let rhs := (phiP - phiM) / (2.0 * eps)
  let absErr := Float.abs (lhs - rhs)
  let relErr := absErr / (Float.abs rhs + 1.0e-9)
  IO.println s!"[gradcheck] adjoint lhs = {lhs}"
  IO.println s!"[gradcheck] finite-diff rhs = {rhs}"
  IO.println s!"[gradcheck] abs err = {absErr}   rel err = {relErr}"
  if relErr < 1.0e-2 then
    IO.println "[gradcheck] ✅ PASS (full MHSA backward matches finite differences)"
  else
    IO.eprintln "[gradcheck] ❌ FAIL — MHSA backward does NOT match finite differences"

def main : IO Unit := do
  IO.println "── @mhsa_fwd ──"
  IO.println fwdModule
  let okF ← compileCheck "mhsa_fwd" fwdModule
  let okB ← compileCheck "mhsa_back" backModule
  if okF && okB then gradcheck

#eval main
