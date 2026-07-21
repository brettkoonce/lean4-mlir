import LeanMlir.Proofs.Codegen.CnnRender

/-! Temporary: re-render `verified_mlir/cifar8_train_step.mlir` with the no-BN SGD lr set to
    0.02 (baked as 0.02/128 = 0.00015625, since the render multiplies the SUM gradient). Same
    faithful renderer as the committed lr=0.1 artifact — only the lr constant changes. Restore
    the committed lr=0.1 version with `git checkout verified_mlir/cifar8_train_step.mlir`. -/

open Proofs.StableHLO

def main : IO Unit :=
  IO.FS.writeFile "verified_mlir/cifar8_train_step.mlir"
    (cifar8TrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3 "0.00015625"
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
      (fun _ => 0))

#eval main
