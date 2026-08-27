import LeanMlir.Proofs.Codegen.ResNet50RenderB
-- Candidate: RSB-A2's own factorisation of 2048 — k = 4 accumulation x 128 per device x 4 replicas
-- (the reference's GRAD_ACCUM = 4, MICRO_BATCH = 512 over 4 devices). Rendered to scratch first,
-- because whether 128 per device at 224^2 FITS is the whole question.
#eval IO.FS.writeFile "/tmp/claude-1000/-home-skoonce-lean-klawd-max-power-lean4-jax-mlir/483d90d7-5c43-451c-a94f-f3dfff3f2c04/scratchpad/cand_lambaccdp4x128wxclipbce.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 128 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 4) "resnet50in" (bce := true) (q := 7)
    (wdExclude := true) (gradClip := true))
