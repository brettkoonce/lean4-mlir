import LeanMlir.VerifiedNets

/-! # `mobilenetv2-verified` — train a real MobileNetV2 on the VERIFIED-rendered codegen

Chapter 7: a real DOWNSAMPLING MobileNetV2 (inverted-residual `[t,c,n,s]`, stride-2
depthwise) on IMAGENETTE 3×224×224 (paper-native resolution):

  stem 3×3-s2 conv (3→16) → BN → relu6 → 6 inverted-residual blocks (16→24→24→32→32→64→64,
  4 stride-2 depthwise downsamples 112→56→28→14→7) → head 1×1 conv (64→128) → BN → relu6 →
  GAP → dense 128→10 + softmax-CE.

The model is `mobilenetv2Verified` (in `LeanMlir.VerifiedNets`); its derived 82-param layout
is kernel-`#guard`ed against the audited `MobileNetV2Layout`. Trains on
`verified_mlir/mobilenetv2_{train_step,fwd}.mlir` (rendered by tests/TestMobilenetV2*) through
the packed-params `VerifiedNet.train` driver (`mlpTrainStepV`, per-channel BN, He-init,
mean-loss SGD lr=0.3). Each op fragment is a proven-faithful emitter (depthwise stride-1/2,
relu6, per-channel BN, 1×1 convs); the whole-net VJP witness `mobilenetv2_has_vjp_at` is a
representative stem+2-block net (the full-net B/C tie is therefore representative).

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mobilenetv2-verified data`
-/

def mobilenetv2Config : VerifiedConfig where
  epochs    := 20
  batchSize := 32
  lr        := 0.3

def main (argv : List String) : IO Unit :=
  mobilenetv2Verified.train mobilenetv2Config (argv.head?.getD "data")
