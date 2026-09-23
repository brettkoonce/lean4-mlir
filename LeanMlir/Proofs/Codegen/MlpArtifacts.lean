import LeanMlir.Proofs.Codegen.MlpRender

/-! # The chapter 2 artifact writer

The `#eval` below writes the committed `verified_mlir/mlp_train_step.mlir` from `MlpRender.lean`'s
faithful renderer when this module is elaborated. Nothing imports this file: the proofs import
`MlpRender`, so building them never rewrites the artifact. -/

-- Regenerate `verified_mlir/mlp_train_step.mlir` (what MainMnistMlpVerified trains on)
-- from the faithful renderer; the den-certified proofs live in MlpFold.lean.
#eval IO.FS.writeFile "verified_mlir/mlp_train_step.mlir"
  (Proofs.StableHLO.mlpTrainStepFaithfulV 128 784 512 512 10 "0.00078125"
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ => 0))

