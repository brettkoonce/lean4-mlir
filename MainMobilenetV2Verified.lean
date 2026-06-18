import LeanMlir.VerifiedNets

/-! # `mobilenetv2-verified` — train a real MobileNetV2 on the VERIFIED-rendered codegen

Chapter 7: the FULL paper-spec DOWNSAMPLING MobileNetV2 (`[t,c,n,s]` table, stride-2
depthwise) on IMAGENETTE 3×224×224 (paper-native resolution):

  stem 3×3-s2 conv (3→32) → BN → relu6 → 17 inverted-residual blocks (full `[t,c,n,s]`:
  t=1 b1 32→16 NO-expand, then 16→24→32→64→96→160→320; 4 stride-2 depthwise downsamples
  112→56→28→14→7) → head 1×1 conv (320→1280) → BN → relu6 → GAP → dense 1280→10 + softmax-CE.

The model is `mobilenetv2Verified` (in `LeanMlir.VerifiedNets`); its derived 210-param layout
(canonical torchvision t=1 no-expand b1) is kernel-`#guard`ed against the audited
`MobileNetV2Layout`. Trains on `verified_mlir/mobilenetv2_train_step.mlir` — the PROOF-TIED
`mnv2TrainStepFaithfulVPaper` render (`MobileNetV2Render.lean`): every line is `pretty` of a
verified `SHlo` node, the whole 210-param train step is `render(provenGraph)`, every param
`den = certified`, and the whole net is den-tied through the real forward + loss-driven backward
(`Proofs.Mnv2TiePoC.mnv2_net_tied_certified`, §1a tie) — plus `mobilenetv2_fwd.mlir` for eval,
via the packed-params `VerifiedNet.train` driver (per-channel BN, He-init, mean-loss SGD lr=0.3).
The whole-net VJP witness `mobilenetv2_has_vjp_at` is still a representative stem+2-block net
(the nonzero-Jacobian seal is therefore representative; the den-tie above is at the full net).

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mobilenetv2-verified data`
-/

def mobilenetv2Config : VerifiedConfig where
  epochs    := 20
  batchSize := 32
  lr        := 0.3

def main (argv : List String) : IO Unit :=
  mobilenetv2Verified.train mobilenetv2Config (argv.head?.getD "data")
