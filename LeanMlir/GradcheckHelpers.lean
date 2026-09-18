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

/-- Parse one iree-printed float token (`-0.00623606`, `1.3e-05`, `42`, `nan`, `-inf`); `none`
    on anything else. Decimal and scientific forms go through the elaborator's own literal decoder
    (`Lean.Syntax.decodeScientificLitVal?` + `Float.ofScientific`), so a token rounds exactly as the
    same literal written in Lean source would; a bare integer, which that decoder rejects, falls
    back to `String.toNat?`. `nan`/`inf` are kept as NaN/∞ so a non-finite output fails a gradcheck
    rather than reading as a number. -/
def parseFloat? (tok : String) : Option Float :=
  let (neg, body) :=
    if tok.startsWith "-" then (true, (tok.drop 1).toString)
    else if tok.startsWith "+" then (false, (tok.drop 1).toString)
    else (false, tok)
  let v? : Option Float :=
    if body == "nan" then some (0.0 / 0.0)
    else if body == "inf" then some (1.0 / 0.0)
    else match Lean.Syntax.decodeScientificLitVal? body with
      | some (m, s, e) => some (Float.ofScientific m s e)
      | none => body.toNat?.map Nat.toFloat
  v?.map fun v => if neg then -v else v

/-- `parseFloat?` with `0.0` for a token it cannot read. -/
def parseFloat (tok : String) : Float := (parseFloat? tok).getD 0.0

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
