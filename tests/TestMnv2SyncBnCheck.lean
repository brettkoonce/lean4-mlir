import LeanMlir.SyncBnCheck
import LeanMlir.Proofs.Codegen.MobileNetV2RenderB

/-! # `mobilenetv2-syncbn-check` — synchronised BatchNorm on MobileNetV2: 2×32 IS 1×64

    lake build mobilenetv2-syncbn-check
    unset CUDA_VISIBLE_DEVICES
    PJRT_REPLICAS=2 .lake/build/bin/mobilenetv2-syncbn-check

`resnet34-syncbn-check`'s gate on the 52-BN-layer MobileNetV2 render (`LeanMlir.SyncBnCheck`):
the committed `mobilenetv2_adamdp_train_step` (2×32, sync-BN) against the 1×64 two-pass step and
the one-replica sync graphs, both rendered at run time. Two GPUs, XLA backend.
-/

def main (_args : List String) : IO Unit :=
  SyncBnCheck.run
    { slug := "mobilenetv2", net := mobilenetv2Verified.toNet, bs := 32
      sgPath := "verified_mlir/mobilenetv2_adam_train_step.mlir"
      dpPath := "verified_mlir/mobilenetv2_adamdp_train_step.mlir"
      render := fun B fs => Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB B 10 "1.0e-5"
        (forceSync := fs)
      entry := fun B r => s!"m.mobilenetv2_{Proofs.StableHLO.mnv2AdamVariant B r}_train_step"
      -- ⚠ 3e-3, not R34's 1e-3. Measured 2026-09-21: the whole-net split error on the statistics
      -- is 0.97e-3 / 1.05e-3 on two runs of the same seeds (XLA's GPU reductions are not
      -- bit-deterministic), and `SYNCBN_VERBOSE=1` shows why it is rounding: the first six BN
      -- layers are split-EXACT (0.000000 on mean and var), and the variance error then grows
      -- smoothly with depth to 1.4e-3 at the 52nd layer — against a per-replica CONTROL of 1.7e-2
      -- and a 1e-4-input SENSITIVITY of 1.4e-3. The first-layer criterion stays at 1e-5.
      statsTol := 3e-3 }
