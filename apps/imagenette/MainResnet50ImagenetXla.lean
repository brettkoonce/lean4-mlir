import apps.imagenette.Resnet50ImagenetCommon

/-! # `resnet50-imagenet-verified-xla` — ResNet-50 on full ImageNet-1k, verified renderer → XLA/PJRT

R50 phase 3. Nothing new was needed in the renderer to reach ImageNet scale — `nClasses`, `B`,
`replicas`, `opt` and `slug` are all parameters of `resnet50TrainStepFaithfulB`, so the four
artifacts are four `#eval`s. What was needed was the three bottleneck block VJPs (phase 1) and the
renderer itself (phase 2).

⚠ Read `Resnet50ImagenetCommon`'s two warnings before quoting anything from this: it is NOT
RSB-A3 (no LAMB, no bs2048, no gradient accumulation), and R50 has no incumbent render to tie
against, so the swap license every other net had does not exist here.

```bash
scripts/gen_shims.sh
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build resnet50-imagenet-verified-xla
lake env lean tests/TestR50Contract.lean
CUDA_VISIBLE_DEVICES=0,2,3,4 PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 \
  PJRT_FFI_RESIDENT=1 SHIM_WORKERS=1 LEAN_MLIR_SKIP_EVAL=1 LEAN_MLIR_G2_STEPS=40 \
  .lake/build/bin/resnet50-imagenet-verified-xla data
```
-/

def main (argv : List String) : IO Unit := runResnet50Imagenet argv
