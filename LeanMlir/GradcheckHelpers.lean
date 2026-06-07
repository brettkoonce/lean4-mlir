/-! # Lean4 numerical gradcheck harness (no numpy)

Shells out to `iree-run-module` to execute compiled `@*_fwd`/`@*_back` `.vmfb`,
then runs the **adjoint / finite-difference dot-product test**: for a forward
`f` with VJP `J·ᵀ`, the backward gives `g_i = (Jᵀ dOut)_i`, and for random
perturbation directions `v_i`,
  Σ_i ⟨g_i, v_i⟩  =  ⟨Jᵀ dOut, v⟩  =  ⟨dOut, J v⟩  =  (Φ(+ε) − Φ(−ε)) / 2ε,
where `Φ(s) := ⟨f(inputs + s·v), dOut⟩`. One backward run + two forward runs
validate ALL input gradients at once — catching transpose/axis bugs that
`iree-compile` (type-checking only) cannot.

Used by the ch10 ViT de-risk tests (TestSDPA/TestMHSA/TestViTBlock). All Lean4. -/

namespace ViTGradcheck

/-- `10^k` as a `Float` (k may be negative). -/
def pow10 (k : Int) : Float :=
  if k ≥ 0 then (List.range k.toNat).foldl (fun a _ => a * 10.0) 1.0
  else (List.range (-k).toNat).foldl (fun a _ => a / 10.0) 1.0

def digitsToNat (cs : List Char) : Nat :=
  cs.foldl (fun acc c => acc * 10 + (c.toNat - '0'.toNat)) 0

/-- Split a `List Char` at the first occurrence of `c` (the separator dropped). -/
def splitAtChar (c : Char) (xs : List Char) : (List Char × Option (List Char)) :=
  match xs.span (· != c) with
  | (pre, [])        => (pre, none)
  | (pre, _ :: post) => (pre, some post)

/-- Parse one iree-printed float token (`-0.00623606`, `1.3e-05`, `42`), entirely
    over `List Char` (robust to the String/Slice API churn). -/
def parseFloat (tok : String) : Float := Id.run do
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

/-- Extract the parsed result buffers (in `result[i]` order) from an
    iree-run-module stdout: each value line is `…xf32=[a b][c d]…`. -/
def parseResults (out : String) : Array (Array Float) := Id.run do
  let mut res : Array (Array Float) := #[]
  for line in out.splitOn "\n" do
    if let some idx := (line.splitOn "f32=")[1]? then
      let cleaned := idx.map (fun c => if c == '[' || c == ']' then ' ' else c)
      let toks := (cleaned.splitOn " ").filter (fun t => !t.isEmpty)
      res := res.push ((toks.map parseFloat).toArray)
  return res

/-- Run a compiled `.vmfb` function; `inputs` are `(shapeStr, flatValues)`. -/
def runFn (vmfb fn : String) (inputs : List (String × Array Float)) : IO (Array (Array Float)) := do
  let inArgs := inputs.map (fun (sh, xs) =>
    s!"--input={sh}=" ++ String.intercalate " " (xs.toList.map toString))
  let args := #[s!"--module={vmfb}", "--device=hip", s!"--function={fn}"] ++ inArgs.toArray
  let r ← IO.Process.output { cmd := "iree-run-module", args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"[run {fn}] FAILED:\n{r.stderr.take 1500}"; return #[]
  return parseResults r.stdout

/-- Deterministic LCG pseudo-random `Array Float` in `[-1,1]`, length `n`. -/
def randVec (seed n : Nat) : Array Float := Id.run do
  let mut s : Nat := seed * 2654435761 + 12345
  let mut out : Array Float := #[]
  for _ in [0:n] do
    s := (s * 1103515245 + 12345) % 2147483648
    out := out.push (2.0 * (Float.ofNat s / 2147483648.0) - 1.0)
  return out

def dot (a b : Array Float) : Float :=
  (a.zip b).foldl (fun acc (x, y) => acc + x * y) 0.0

/-- `y + a·x` (elementwise). -/
def axpy (a : Float) (x y : Array Float) : Array Float :=
  (y.zip x).map (fun (yi, xi) => yi + a * xi)

/-- **Adjoint/finite-difference gradcheck** of a compiled fwd/back pair.
    `inShapes`/`inLens` describe the forward inputs (in arg order); the backward
    is expected to return one gradient per input in the same order. `outShape`/
    `outLen` describe the forward's single output (the `dOut` cotangent). Returns
    `true` iff the relative error is below `tol` (default 1e-2, for f32). -/
def adjointGradcheck (label fwdVmfb fwdFn backVmfb backFn : String)
    (inShapes : List String) (inLens : List Nat)
    (outShape : String) (outLen : Nat)
    (seedBase : Nat := 0) (eps : Float := 1.0e-3) (tol : Float := 1.0e-2) : IO Bool := do
  let params := (inLens.zipIdx).map (fun (l, i) => randVec (seedBase + 100 + i) l)
  let dirs   := (inLens.zipIdx).map (fun (l, i) => randVec (seedBase + 200 + i) l)
  let dO := randVec (seedBase + 42) outLen
  let ins := inShapes.zip params
  let back ← runFn backVmfb backFn (ins ++ [(outShape, dO)])
  if back.size != inShapes.length then
    IO.eprintln s!"[{label}] expected {inShapes.length} back results, got {back.size}"; return false
  let lhs := ((back.toList.zip dirs).map (fun (g, v) => dot g v)).foldl (· + ·) 0.0
  let phi (s : Float) : IO Float := do
    let pert := (params.zip dirs).map (fun (pv, vv) => axpy s vv pv)
    let f ← runFn fwdVmfb fwdFn (inShapes.zip pert)
    if f.size != 1 then IO.eprintln s!"[{label}] fwd result missing"; return 0.0
    return dot f[0]! dO
  let phiP ← phi eps
  let phiM ← phi (-eps)
  let rhs := (phiP - phiM) / (2.0 * eps)
  let absErr := Float.abs (lhs - rhs)
  let relErr := absErr / (Float.abs rhs + 1.0e-9)
  IO.println s!"[{label}] adjoint lhs = {lhs}   finite-diff rhs = {rhs}"
  IO.println s!"[{label}] abs err = {absErr}   rel err = {relErr}"
  if relErr < tol then
    IO.println s!"[{label}] ✅ PASS"; return true
  else
    IO.eprintln s!"[{label}] ❌ FAIL — backward does NOT match finite differences"; return false

/-- Like `adjointGradcheck` but with `fixed` inputs (concrete `(shape,values)`)
    that are passed to BOTH fwd and back, never perturbed, and have no expected
    gradient — e.g. a ViT input image (first layer ⇒ no image grad). The forward
    arg order is `fixed ++ params`; the backward is `fixed ++ params ++ dOut` and
    returns one grad per PARAM (in order). -/
def adjointGradcheckFixed (label fwdVmfb fwdFn backVmfb backFn : String)
    (fixed : List (String × Array Float))
    (inShapes : List String) (inLens : List Nat)
    (outShape : String) (outLen : Nat)
    (seedBase : Nat := 0) (eps : Float := 1.0e-3) (tol : Float := 1.0e-2) : IO Bool := do
  let params := (inLens.zipIdx).map (fun (l, i) => randVec (seedBase + 100 + i) l)
  let dirs   := (inLens.zipIdx).map (fun (l, i) => randVec (seedBase + 200 + i) l)
  let dO := randVec (seedBase + 42) outLen
  let ins := inShapes.zip params
  let back ← runFn backVmfb backFn (fixed ++ ins ++ [(outShape, dO)])
  if back.size != inShapes.length then
    IO.eprintln s!"[{label}] expected {inShapes.length} back results, got {back.size}"; return false
  let lhs := ((back.toList.zip dirs).map (fun (g, v) => dot g v)).foldl (· + ·) 0.0
  let phi (s : Float) : IO Float := do
    let pert := (params.zip dirs).map (fun (pv, vv) => axpy s vv pv)
    let f ← runFn fwdVmfb fwdFn (fixed ++ inShapes.zip pert)
    if f.size != 1 then IO.eprintln s!"[{label}] fwd result missing"; return 0.0
    return dot f[0]! dO
  let phiP ← phi eps
  let phiM ← phi (-eps)
  let rhs := (phiP - phiM) / (2.0 * eps)
  let absErr := Float.abs (lhs - rhs)
  let relErr := absErr / (Float.abs rhs + 1.0e-9)
  IO.println s!"[{label}] adjoint lhs = {lhs}   finite-diff rhs = {rhs}"
  IO.println s!"[{label}] abs err = {absErr}   rel err = {relErr}"
  if relErr < tol then
    IO.println s!"[{label}] ✅ PASS"; return true
  else
    IO.eprintln s!"[{label}] ❌ FAIL — backward does NOT match finite differences"; return false

end ViTGradcheck
