import LeanMlir.VerifiedNets

/-! # `convnext-verified` — train ConvNeXt-T on the VERIFIED-rendered codegen

Chapter 9: faithful ConvNeXt-T (Liu et al. 2022) on IMAGENETTE 3×224×224 (paper-native
resolution, patchify /4 then /2 between stages, 224→56→28→14→7):

  4×4-s4 patchify (3→96) → [3,3,9,3] ConvNeXt blocks @ [96,192,384,768]
  (depthwise 7×7 → scalar-LN → 1×1 expand c→4c → GELU → 1×1 project 4c→c → layerScale)
  with 3 between-stage (LN + 2×2-s2) downsamples → GAP → LN(768) → dense 768→10 + softmax-CE.

The model is `convnextVerified` (in `LeanMlir.VerifiedNets`); its derived 180-param layout is
kernel-`#guard`ed against the audited `ConvNeXtLayout`. Trains on
`verified_mlir/convnext_{train_step,fwd}.mlir` (rendered by tests/TestConvNeXt*) through the
packed-params `VerifiedNet.train` driver (`mlpTrainStepV`, global-scalar LN, He-init). Each op
fragment is a proven-faithful emitter (GELU/LN/layerScale/depthwise-7×7/even-kernel patchify+
downsample); the whole-net VJP `convnext_has_vjp` is a representative witness (full B/C deferred).

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/convnext-verified data`
-/

def convnextConfig : VerifiedConfig where
  epochs    := 20
  batchSize := 32

def main (argv : List String) : IO Unit :=
  convnextVerified.train convnextConfig (argv.head?.getD "data")
