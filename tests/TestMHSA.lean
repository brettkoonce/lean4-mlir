import LeanMlir.Proofs.Codegen.StableHLOPretty
import LeanMlir.Types
import LeanMlir.GradcheckHelpers
import LeanMlir.ViTRender

/-! # ch10 V3 — multi-head self-attention renderer (fwd + full backward)

The big new ViT combinator (the analogue of ch8's SE module): MHSA = QKV proj →
reshape/transpose to H heads → BATCHED per-head SDPA (V1 row-softmax + V2 SDPA, now
with `batching_dims=[0,1]` over (batch, heads)) → concat heads → out-proj. De-risked
STANDALONE on `[B,N,D]`, `iree-compile`d, AND numerically gradchecked (adjoint test,
all in Lean4) over EVERY input (x + Wq/bq/Wk/bk/Wv/bv/Wo/bo) BEFORE wiring into the
block (TestViTBlock). The fragments are the shipped `ViTRender.mhsaFwd`/`mhsaBack`, so this
gradchecks the emitter the ViT renders use. Forward shapes mirror MlirCodegen's `emitMHSAForward`
(the GPU reference); softmax is the V1 plain exp/sum (NO max-shift) matching the proven `softmax`.

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
open ViTGradcheck

private def Bb : Nat := 2     -- batch
private def Nn : Nat := 3     -- tokens
private def Dd : Nat := 4     -- model dim
private def Hh : Nat := 2     -- heads
private def Dh : Nat := 2     -- head dim (= D/H)
private def scaleStr : String := "0.7071067811865476"   -- 1/√2

private def sig : String :=
  s!"%x: {ty [Bb,Nn,Dd]}, %Wq: {ty [Dd,Dd]}, %bq: {ty [Dd]}, %Wk: {ty [Dd,Dd]}, %bk: {ty [Dd]}, " ++
  s!"%Wv: {ty [Dd,Dd]}, %bv: {ty [Dd]}, %Wo: {ty [Dd,Dd]}, %bo: {ty [Dd]}"

private def fwdModule : String :=
  "module @m {\n" ++
  s!"  func.func @mhsa_fwd({sig}) -> {ty [Bb,Nn,Dd]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  ViTRender.mhsaFwd "a" "%x" "%Wq" "%bq" "%Wk" "%bk" "%Wv" "%bv" "%Wo" "%bo" Bb Nn Dd Hh Dh scaleStr ++
  s!"    return %aO : {ty [Bb,Nn,Dd]}\n" ++ "  }\n}\n"

private def backModule : String :=
  let retTy := String.intercalate ", "
    [ty [Bb,Nn,Dd], ty [Dd,Dd], ty [Dd], ty [Dd,Dd], ty [Dd], ty [Dd,Dd], ty [Dd], ty [Dd,Dd], ty [Dd]]
  "module @m {\n" ++
  s!"  func.func @mhsa_back({sig}, %dO: {ty [Bb,Nn,Dd]}) -> ({retTy}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  ViTRender.mhsaFwd "a" "%x" "%Wq" "%bq" "%Wk" "%bk" "%Wv" "%bv" "%Wo" "%bo" Bb Nn Dd Hh Dh scaleStr ++
  ViTRender.mhsaBack "a" "%x" "%Wq" "%Wk" "%Wv" "%Wo" "%dO" Bb Nn Dd Hh Dh scaleStr ++
  s!"    return %adx, %adWQ, %adbQ, %adWK, %adbK, %adWV, %adbV, %adWo, %adbo : {retTy}\n" ++ "  }\n}\n"

/-- Adjoint/finite-difference gradcheck of the full MHSA over every input (x + the 8 params),
    in the `@mhsa_fwd` arg order. -/
private def gradcheck : IO Unit := do
  let _ ← adjointGradcheck "mhsa gradcheck" ".lake/build/mhsa_fwd.vmfb" "mhsa_fwd"
    ".lake/build/mhsa_back.vmfb" "mhsa_back"
    [s!"{Bb}x{Nn}x{Dd}xf32", s!"{Dd}x{Dd}xf32", s!"{Dd}xf32", s!"{Dd}x{Dd}xf32", s!"{Dd}xf32",
     s!"{Dd}x{Dd}xf32", s!"{Dd}xf32", s!"{Dd}x{Dd}xf32", s!"{Dd}xf32"]
    [Bb*Nn*Dd, Dd*Dd, Dd, Dd*Dd, Dd, Dd*Dd, Dd, Dd*Dd, Dd]
    s!"{Bb}x{Nn}x{Dd}xf32" (Bb*Nn*Dd)

def main : IO Unit := do
  IO.println "── @mhsa_fwd ──"
  IO.println fwdModule
  let okF ← compileCheckB "mhsa_fwd" fwdModule
  let okB ← compileCheckB "mhsa_back" backModule
  if okF && okB then gradcheck

#eval main
