/-! # Lean4 numerical gradcheck harness (no numpy)

Shells out to `iree-run-module` to execute compiled `@*_fwd`/`@*_back` `.vmfb`,
then runs the **adjoint / finite-difference dot-product test**: for a forward
`f` with VJP `J·ᵀ`, the backward gives `g_i = (Jᵀ dOut)_i`, and for random
perturbation directions `v_i`,
  Σ_i ⟨g_i, v_i⟩  =  ⟨Jᵀ dOut, v⟩  =  ⟨dOut, J v⟩  =  (Φ(+ε) − Φ(−ε)) / 2ε,
where `Φ(s) := ⟨f(inputs + s·v), dOut⟩`. One backward run + two forward runs
validate ALL input gradients at once — catching transpose/axis bugs that
`iree-compile` (type-checking only) cannot.

Used by the ViT gradcheck tests (tests/TestSDPA.lean, TestMHSA, TestViTBlock, TestViTTiny).
The float-token parser (`ViTGradcheck.parseFloat?`, `ViTGradcheck.parseFloat`) is also used by
TestSgdRenderTie and by three demos (MainAraslSigns, MainPlantLeaf, MainSegLossProbe) to read
float arguments. All Lean4. -/

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

/-- Write a flat array as raw little-endian `f32`, for `iree-run-module --input=<shape>=@file`.
    Inputs and outputs go through files, not text: `Float.toString` keeps six decimals and IREE
    prints six significant digits, and either rounding alone puts a ~3e-4 floor under a
    finite-difference quotient at ε = 1e-3. Through files the floor is f32's own. -/
private def writeBinF32 (path : String) (xs : Array Float) : IO Unit := do
  let mut b := ByteArray.emptyWithCapacity (4 * xs.size)
  for x in xs do
    let u := x.toFloat32.toBits
    b := b.push u.toUInt8 |>.push (u >>> 8).toUInt8 |>.push (u >>> 16).toUInt8
      |>.push (u >>> 24).toUInt8
  IO.FS.writeBinFile path b

/-- Read an `f32` `.npy` file (as `iree-run-module --output=@file.npy` writes it) into a flat
    array. -/
private def readNpyF32 (path : String) : IO (Array Float) := do
  let b ← IO.FS.readBinFile path
  unless b.size ≥ 10 && b[0]! == 0x93 && b[6]! == 1 do
    throw <| IO.userError s!"{path}: not a v1 .npy file"
  let hlen := b[8]!.toNat + 256 * b[9]!.toNat
  let header := String.fromUTF8! (b.extract 10 (10 + hlen))
  unless (header.splitOn "'<f4'").length > 1 do
    throw <| IO.userError s!"{path}: expected dtype '<f4', header {header}"
  let n := (b.size - 10 - hlen) / 4
  return (Array.range n).map fun i =>
    let o := 10 + hlen + 4 * i
    let bits := b[o]!.toUInt32 ||| (b[o+1]!.toUInt32 <<< 8) |||
      (b[o+2]!.toUInt32 <<< 16) ||| (b[o+3]!.toUInt32 <<< 24)
    (Float32.ofBits bits).toFloat

/-- The `iree-run-module` device for the `IREE_BACKEND` the `.vmfb` was compiled for (default
    `cuda`, as in `ireeCompileArgs`): `rocm` runs on `hip`, `llvm-cpu` on `local-task`. -/
def runDevice : IO String := do
  match (← IO.getEnv "IREE_BACKEND").getD "cuda" with
  | "rocm" => return "hip"
  | "llvm-cpu" => return "local-task"
  | b => return b

/-- Run a compiled `.vmfb` function with `nOut` results; `inputs` are `(shapeStr, flatValues)`. -/
private def runFn (vmfb fn : String) (inputs : List (String × Array Float)) (nOut : Nat) :
    IO (Array (Array Float)) := do
  let inArgs ← inputs.zipIdx.mapM fun ((sh, xs), i) => do
    let f := s!".lake/build/{fn}_in{i}.bin"
    writeBinF32 f xs
    return s!"--input={sh}=@{f}"
  let outs := (List.range nOut).map (fun i => s!".lake/build/{fn}_out{i}.npy")
  let args := #[s!"--module={vmfb}", s!"--device={← runDevice}", s!"--function={fn}"] ++
    inArgs.toArray ++ (outs.map (s!"--output=@" ++ ·)).toArray
  let r ← IO.Process.output { cmd := "iree-run-module", args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"[run {fn}] FAILED:\n{r.stderr.take 1500}"; return #[]
  outs.toArray.mapM readNpyF32

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
private def axpy (a : Float) (x y : Array Float) : Array Float :=
  (y.zip x).map (fun (yi, xi) => yi + a * xi)

/-- **Adjoint/finite-difference gradcheck** of a compiled fwd/back pair, checking
    ⟨back(in, dOut), v⟩ ≈ (Φ(+ε) − Φ(−ε))/2ε with Φ(s) = ⟨fwd(in + s·v), dOut⟩ for random `v`.
    `inShapes`/`inLens` describe the perturbed inputs (in arg order), `outShape`/`outLen` the
    forward's single output. `fixed` inputs (concrete `(shape,values)`) are passed to BOTH fwd and
    back, never perturbed, and have no expected gradient — e.g. a ViT input image. Forward arg
    order is `fixed ++ params`; the backward takes `fixed ++ params ++ dOut` and returns one grad
    per PARAM (in order). `true` iff the relative error is below `tol` (default 1e-2, for f32). -/
def adjointGradcheckFixed (label fwdVmfb fwdFn backVmfb backFn : String)
    (fixed : List (String × Array Float))
    (inShapes : List String) (inLens : List Nat)
    (outShape : String) (outLen : Nat)
    (seedBase : Nat := 0) (eps : Float := 1.0e-3) (tol : Float := 1.0e-2) : IO Bool := do
  let params := (inLens.zipIdx).map (fun (l, i) => randVec (seedBase + 100 + i) l)
  let dirs   := (inLens.zipIdx).map (fun (l, i) => randVec (seedBase + 200 + i) l)
  let dO := randVec (seedBase + 42) outLen
  let ins := inShapes.zip params
  let back ← runFn backVmfb backFn (fixed ++ ins ++ [(outShape, dO)]) inShapes.length
  if back.size != inShapes.length then
    IO.eprintln s!"[{label}] expected {inShapes.length} back results, got {back.size}"; return false
  let lhs := ((back.toList.zip dirs).map (fun (g, v) => dot g v)).foldl (· + ·) 0.0
  let phi (s : Float) : IO Float := do
    let pert := (params.zip dirs).map (fun (pv, vv) => axpy s vv pv)
    let f ← runFn fwdVmfb fwdFn (fixed ++ inShapes.zip pert) 1
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

/-- `adjointGradcheckFixed` with no fixed inputs: the backward returns one gradient per input. -/
def adjointGradcheck (label fwdVmfb fwdFn backVmfb backFn : String)
    (inShapes : List String) (inLens : List Nat)
    (outShape : String) (outLen : Nat)
    (seedBase : Nat := 0) (eps : Float := 1.0e-3) (tol : Float := 1.0e-2) : IO Bool :=
  adjointGradcheckFixed label fwdVmfb fwdFn backVmfb backFn [] inShapes inLens outShape outLen
    seedBase eps tol

end ViTGradcheck
