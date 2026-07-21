import LeanMlir.Proofs.StableHLO
import LeanMlir.Proofs.Architectures.Attention
import LeanMlir.Types

/-! Standalone render + `iree-compile` validation for the Chapter-10 V1 ROW-softmax
    SHlo op pair (`softmaxRowF` / `softmaxRowBack`) — the one genuinely new op of the
    ViT chapter. Renders a tiny forward (per-row `exp / reduce[last] / divide`, plain
    exp/sum — NO max-shift, matching the proven `softmax`) and backward (the closed
    form `p ⊙ (dy − ⟨p,dy⟩)` per row, recomputing `p` from the saved pre-softmax
    scores `%xs`) `func.func` from the VERIFIED `Proofs.StableHLO` emitter and
    compiles each to a ROCm `.vmfb` — the thin lexical boundary the proofs leave to
    `iree-compile`. Row-softmax is smooth everywhere (no kink). The Lean operand/x
    values are render-irrelevant placeholders.

    Also closes the **faithfulness tie**: the new ops' `den` equals the PROVEN
    `rowSoftmax` / `rowSoftmax_has_vjp_mat.backward` (Attention.lean) by `rfl` —
    `StableHLO` itself spells the den with MLP's `softmax` to avoid importing the
    capstone `Attention` (no import cycle); this file (which imports both) certifies
    the tie. rfl-faithful ⇒ stays OUT of the axiom audit (`roundtrip` covers it).

    Run (rocm):
      export PATH="$PWD/.venv/bin:$PATH"
      export IREE_BACKEND=rocm
      lake env lean tests/TestSoftmaxRow.lean
-/

open Proofs Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Faithfulness: the new ops denote the PROVEN row-softmax (+ its VJP)
-- ════════════════════════════════════════════════════════════════

/-- **`rowSoftmaxFlat` IS the proven `rowSoftmax`** (flattened). The `den` of
    `softmaxRowF` (spelled via MLP's `softmax`) equals `Mat.flatten ∘ rowSoftmax ∘
    Mat.unflatten`, the Attention.lean object whose smoothness/VJP are proven. -/
theorem rowSoftmaxFlat_eq_rowSoftmax (m n : Nat) (v : Vec (m*n)) :
    rowSoftmaxFlat m n v = Mat.flatten (rowSoftmax (Mat.unflatten v)) := rfl

/-- **`rowSoftmaxBackFlat` IS the proven `rowSoftmax_has_vjp_mat.backward`**
    (flattened). The closed form `p ⊙ (dy − ⟨p,dy⟩)` per row equals the proven
    block-diagonal row-softmax VJP backward applied to the unflattened matrices. -/
theorem rowSoftmaxBackFlat_eq_vjp (m n : Nat) (preAct dy : Vec (m*n)) :
    rowSoftmaxBackFlat m n preAct dy =
      Mat.flatten ((rowSoftmax_has_vjp_mat (m := m) (n := n)).backward
                     (Mat.unflatten preAct) (Mat.unflatten dy)) := rfl

/-- The forward op `den` denotes the proven `rowSoftmax`. -/
theorem softmaxRowF_den_rowSoftmax {m n : Nat} (e : SHlo (m*n)) :
    den (.softmaxRowF e) = Mat.flatten (rowSoftmax (Mat.unflatten (den e))) := rfl

/-- The backward op `den` denotes the proven `rowSoftmax_has_vjp_mat.backward`. -/
theorem softmaxRowBack_den_vjp {m n : Nat} (xN : String) (preAct : Vec (m*n)) (e : SHlo (m*n)) :
    den (.softmaxRowBack xN preAct e) =
      Mat.flatten ((rowSoftmax_has_vjp_mat (m := m) (n := n)).backward
                     (Mat.unflatten preAct) (Mat.unflatten (den e))) := rfl

-- ════════════════════════════════════════════════════════════════
-- § Render + iree-compile
-- ════════════════════════════════════════════════════════════════

private def M  : Nat := 3   -- rows (tokens, the N of attention)
private def Nn : Nat := 3   -- cols (the keys, also N — scores are [N,N])
private def BS : Nat := 2

-- render-irrelevant placeholder runtime values
private def xv  : Vec (M*Nn) := fun _ => 0
private def dyv : Vec (M*Nn) := fun _ => 0

/-- `@softmax_row_fwd` from the verified AST: one `softmaxRowF` over `%x`
    (`[BS, M*Nn]` flat, row-softmax over each of the M rows of Nn cols). -/
private def fwdModule : String :=
  renderModule "softmax_row_fwd" s!"%x: {ty [BS, M*Nn]}"
    BS (M*Nn) (.softmaxRowF (m := M) (n := Nn) (.operand "%x" xv))

/-- `@softmax_row_back` from the verified AST: one `softmaxRowBack` over `%dy`,
    recomputing `p` from the saved pre-softmax scores `%xs`. -/
private def backModule : String :=
  renderModule "softmax_row_back" s!"%dy: {ty [BS, M*Nn]}, %xs: {ty [BS, M*Nn]}"
    BS (M*Nn) (.softmaxRowBack (m := M) (n := Nn) "%xs" xv (.operand "%dy" dyv))

private def compileCheck (name body : String) : IO Unit := do
  IO.FS.createDirAll ".lake/build"
  let path := s!".lake/build/{name}.mlir"
  IO.FS.writeFile path body
  let cargs ← ireeCompileArgs path s!".lake/build/{name}.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"[{name}] iree-compile FAILED:\n{r.stderr.take 2000}"
  else
    IO.println s!"[{name}] iree-compile OK → .lake/build/{name}.vmfb"

def main : IO Unit := do
  IO.println "── @softmax_row_fwd ──"
  IO.println fwdModule
  IO.println "── @softmax_row_back ──"
  IO.println backModule
  compileCheck "softmax_row_fwd" fwdModule
  compileCheck "softmax_row_back" backModule

#eval main
