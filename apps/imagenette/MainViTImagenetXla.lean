import apps.imagenette.ViTImagenetCommon

/-! # `vit-imagenet-verified-xla` — ViT-Tiny on full ImageNet-1k, verified renderer → XLA/PJRT

The ViT peer of `resnet34-imagenet-verified-xla` (handoff §2p). Nothing new was needed in the
renderer to get here — `nClasses`, `bs` and `replicas` are all parameters of it, so the three
ImageNet artifacts are three `#eval`s, exactly as §2k found for ResNet-34.

XLA-only by construction: collectives live on the PJRT path, and IREE has no measured
ImageNet-scale number here to compare against.

⚠ **This does not move the verification tier** — the proof-carrying claims stop at Imagenette. See
`ViTImagenetCommon`'s claim-ceiling note, and note in particular that this is not the DeiT recipe.

⚠ **Set `SHIM_WORKERS=2`.** ViT is the first net here whose step rate outruns a single shim
producer (~1,940 img/s wanted against ~1,530 delivered); without it the GPUs wait on data.

```bash
(cd jax && lake exe resnet34-imagenet default --shim)
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build vit-imagenet-verified-xla
PJRT_FFI_RESIDENT=1 CUDA_VISIBLE_DEVICES=0,1,2,3 SHIM_WORKERS=2 \
  LEAN_MLIR_VARIANT=adamdp128x4 LEAN_MLIR_BATCH=128 \
  LEAN_MLIR_REPLICAS=4 PJRT_REPLICAS=4 \
  .lake/build/bin/vit-imagenet-verified-xla data
```
-/

def main (argv : List String) : IO Unit := runViTImagenet argv
