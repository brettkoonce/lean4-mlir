import LeanMlir.SyncBnCheck
import LeanMlir.Proofs.Codegen.EfficientNetRender

/-! # `efficientnet-syncbn-check` — synchronised BatchNorm on EfficientNet-B0: 2×32 IS 1×64

    lake build efficientnet-syncbn-check
    unset CUDA_VISIBLE_DEVICES
    PJRT_REPLICAS=2 .lake/build/bin/efficientnet-syncbn-check

`resnet34-syncbn-check`'s gate on the 49-BN-layer EfficientNet-B0 render (`LeanMlir.SyncBnCheck`):
the committed `efficientnet_adamdp_train_step` (2×32, sync-BN) against the 1×64 two-pass step and
the one-replica sync graphs, both rendered at run time. ⚠ B0's loss divisor is a string argument,
so the run-time renders pass `"{B}.0"` — the batch each is rendered at. Two GPUs, XLA backend.
-/

def main (_args : List String) : IO Unit :=
  SyncBnCheck.run
    { slug := "efficientnet", net := efficientnetVerified.toNet, bs := 32
      sgPath := "verified_mlir/efficientnet_adam_train_step.mlir"
      dpPath := "verified_mlir/efficientnet_adamdp_train_step.mlir"
      render := fun B fs => Proofs.StableHLO.efficientnetAdamTrainStepFaithful B 10 "1.0e-5"
        "0.100000" "-0.010000" s!"{B}.0" (forceSync := fs)
      entry := fun B r => s!"m.efficientnet_{Proofs.StableHLO.enetAdamVariant B r}_train_step" }
