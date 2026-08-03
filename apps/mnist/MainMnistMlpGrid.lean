import LeanMlir.VerifiedNets
import LeanMlir.Proofs.Codegen.MlpRender
import LeanMlir.Proofs.Codegen.StableHLO

/-! # `mnist-mlp-grid` — the width-parametric MNIST MLP demo

One parametric trainer for the whole `784→d₁→d₂→10` size grid. Reads the two hidden
widths `d₁ d₂` from argv, renders `verified_mlir/mlp_{d₁}x{d₂}_{train_step,fwd}.mlir`
from the **faithful** renderers (`mlpTrainStepFaithfulV` — every line is `pretty` of a
den-certified verified AST node — and `mlpFwdModuleV`, the forward AST), then trains on
that render through the shared `VerifiedNet.train` driver (Lean → IREE FFI → GPU).

The architecture is `mlpG d₁ d₂` (in `LeanMlir.VerifiedNets`); its math VJP is the
polymorphic `mlp_has_vjp {d₀ d₁ d₂ d₃}` (SpecVJP/MLP.lean) instantiated at these dims —
so every grid point is one theorem, not a new proof. The canonical Chapter-2 demo is
exactly `mnist-mlp-grid 512 512`.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-mlp-grid 256 128 [epochs] [dataDir]`
-/

open Proofs.StableHLO in
/-- Render the faithful train-step + forward MLIR for `mlp_{d₁}x{d₂}` (B=128, lr baked
    to 0.1/128 = the mean-loss equiv of the book's 0.1). Values are erased by `pretty`,
    so the zero placeholders print the exact text the `den` theorems certify. -/
def renderGrid (d₁ d₂ : Nat) : IO Unit := do
  -- ⚠ The SAME directory the net reads from (`VerifiedNet.mlirDir`), derived rather than
  -- restated: these are build products and must not land in the pinned `verified_mlir/`.
  let dir := (mlpG d₁ d₂).mlirDir
  IO.FS.createDirAll dir
  let slug := s!"mlp_{d₁}x{d₂}"
  -- The renderers hardcode `@mlp_train_step` / `@mlp_fwd`; the driver invokes
  -- `m.{slug}_train_step`, so rename the exported func symbol to the slug (module
  -- stays `@m`; the rename touches only the `func.func @…` line + any self-reference).
  let ts := (mlpTrainStepFaithfulV 128 784 d₁ d₂ 10 "0.00078125"
      (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
      (fun _ => 0)).replace "@mlp_train_step" s!"@{slug}_train_step"
  let fwd := (mlpFwdModuleV 128 784 d₁ d₂ 10
      (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
      (fun _ => 0)).replace "@mlp_fwd" s!"@{slug}_fwd"
  IO.FS.writeFile s!"{dir}/{slug}_train_step.mlir" ts
  IO.FS.writeFile s!"{dir}/{slug}_fwd.mlir" fwd

def main (argv : List String) : IO Unit := do
  match argv with
  | d1s :: d2s :: rest =>
    let some d₁ := d1s.toNat? | throw (.userError s!"bad d₁: {d1s}")
    let some d₂ := d2s.toNat? | throw (.userError s!"bad d₂: {d2s}")
    let epochs := (rest.head?.bind (·.toNat?)).getD 12
    let dataDir := rest[1]?.getD "data"
    renderGrid d₁ d₂
    (mlpG d₁ d₂).train { epochs := epochs, batchSize := 128 } dataDir
  | _ => throw (.userError "usage: mnist-mlp-grid <d₁> <d₂> [epochs] [dataDir]")
