import LeanMlir.VerifiedNets
import LeanMlir.Proofs.Codegen.CnnRender
import LeanMlir.Proofs.Codegen.StableHLO

/-! # `mnist-cnn-grid` — FC-width-parametric MNIST CNN demo (the 2D peer of `mnist-mlp-grid`)

The Chapter-3 CNN with the **conv feature extractor held fixed** (two 3×3 convs @ 32
channels, maxpool 28→14, flatten 6272) and only the **dense classifier head swept**:
`flatten(6272) → dense 6272→d → relu → dense d→d → relu → dense d→10`. Reads the FC width
`d` from argv, renders `verified_mlir/cnn_{d}_{train_step,fwd}.mlir` from the **faithful**
renderers (`cnnTrainStepFaithfulV` — every line is `pretty` of a den-certified verified AST
node — and `cnnFwdModuleV`), then trains on that render (Lean → IREE FFI → GPU).

The faithful CNN renderer takes a single dense width (both FC hidden layers share `d`), so
this is a 1-D sweep of the classifier head — the honest den-certified path. Architecture is
`cnnG d` (`LeanMlir.VerifiedNets`); the canonical Chapter-3 demo is `mnist-cnn-grid 512`.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-cnn-grid 128 [epochs] [dataDir]`
-/

open Proofs.StableHLO in
/-- Render the faithful CNN train-step + forward MLIR for `cnn_{d}` (conv @ 32, post-pool
    14×14, 3×3 kernels, B=128, lr 0.1/128). Values are erased by `pretty`, so the zero
    placeholders print the exact text the `den` theorems certify. -/
def renderCnnGrid (d : Nat) : IO Unit := do
  IO.FS.createDirAll "verified_mlir"
  let slug := s!"cnn_{d}"
  let ts := (cnnTrainStepFaithfulV 128 1 32 14 14 d 10 3 3 "0.00078125"
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
      (fun _ => 0)).replace "@cnn_train_step" s!"@{slug}_train_step"
  let fwd := (cnnFwdModuleV 128 1 32 14 14 d 10 3 3
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
      (fun _ => 0)).replace "@cnn_fwd" s!"@{slug}_fwd"
  IO.FS.writeFile s!"verified_mlir/{slug}_train_step.mlir" ts
  IO.FS.writeFile s!"verified_mlir/{slug}_fwd.mlir" fwd

def main (argv : List String) : IO Unit := do
  match argv with
  | ds :: rest =>
    let some d := ds.toNat? | throw (.userError s!"bad FC width: {ds}")
    let epochs := (rest.head?.bind (·.toNat?)).getD 10
    let dataDir := rest[1]?.getD "data"
    renderCnnGrid d
    (cnnG d).train { epochs := epochs, batchSize := 128 } dataDir
  | _ => throw (.userError "usage: mnist-cnn-grid <fc-width> [epochs] [dataDir]")
