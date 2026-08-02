import apps.imagenette.Resnet34ImagenetCommon

/-! # `resnet34-imagenet-verified-xla` — ResNet-34 on full ImageNet-1k, verified renderer → XLA/PJRT

The scale tier of handoff §2k, and the payoff the whole verified track was pointed at: the certified
renderer at ImageNet shapes, fed by the same augmentation pipeline the Lean→JAX reference trainer
uses, so the two are a **matched pair** and JAX is an external oracle rather than a separate story.

Nothing new was needed in the renderer to get here — `nClasses`, `B`, `opt` and `slug` are all
parameters of it, so the three ImageNet artifacts are three `#eval`s. What was needed was the data
path: `VerifiedData.imagenet` and the tfds shim.

XLA-only by construction: the shim path is where the throughput is, and IREE has no measured
ImageNet-scale number here to compare against.

⚠ **This does not move the verification tier** — the proof-carrying claims stop at Imagenette. See
`Resnet34ImagenetCommon`'s claim-ceiling note before quoting a result from it.

```bash
scripts/gen_shims.sh                       # this net's OWN data shim (⚠ NOT R34's — see VerifiedNet.shimScript)
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build resnet34-imagenet-verified-xla
HIP_VISIBLE_DEVICES=0 LEAN_MLIR_BASE_LR_U=100000 \
  .lake/build/bin/resnet34-imagenet-verified-xla data
```
-/

def main (argv : List String) : IO Unit := runResnet34Imagenet argv
