import LeanMlir.VerifiedNets

/-! # `vit-verified` — train ViT-Tiny on the VERIFIED-rendered codegen

Chapter 10: the Vision Transformer (Dosovitskiy et al. 2021) on IMAGENETTE 224×224, patch-16:

  16×16-s16 conv patch embed (3→192) → flatten 196 patches → prepend CLS → +positional embed
  (197 tokens) → 12× pre-norm transformer block (dim 192, 3 heads, MLP 768) → final LayerNorm
  → CLS token (row 0) → dense 192→10 + softmax-CE.

The model is `vitVerified` (in `LeanMlir.VerifiedNets`); its derived 200-param layout is
kernel-`#guard`ed against the audited `ViTLayout`. Trains on `verified_mlir/vit_{train_step,
fwd}.mlir` (rendered by tests/TestViT*) through the packed-params `VerifiedNet.train` driver
(`mlpTrainStepV`, per-channel LayerNorm, He-init). Each op fragment is a proven-faithful
emitter (row-softmax / batched multi-head SDPA / per-channel LN / GELU / patch conv); the
whole-net VJP `vit_full_has_vjp_correct` is the scalar-LN witness (full B/C deferred).

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/vit-verified data`
-/

def vitConfig : VerifiedConfig where
  epochs    := 20
  batchSize := 32

def main (argv : List String) : IO Unit :=
  vitVerified.train vitConfig (argv.head?.getD "data")
