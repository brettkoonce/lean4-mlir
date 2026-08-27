import LeanMlir.Proofs.Codegen.ResNet50RenderB
#eval IO.FS.writeFile "/tmp/claude-1000/-home-skoonce-lean-klawd-max-power-lean4-jax-mlir/483d90d7-5c43-451c-a94f-f3dfff3f2c04/scratchpad/cand_lambaccdp4x128wxclipbcebf16.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 128 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 4) "resnet50in" (bce := true) (q := 7)
    (wdExclude := true) (gradClip := true) (bf16 := true))
#eval IO.FS.writeFile "/tmp/claude-1000/-home-skoonce-lean-klawd-max-power-lean4-jax-mlir/483d90d7-5c43-451c-a94f-f3dfff3f2c04/scratchpad/cand_emalambaccdp4x128wxclipdropbce.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 128 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 4) "resnet50in" (bce := true) (q := 7)
    (wdExclude := true) (gradClip := true) (ema := true) (sd := true))
