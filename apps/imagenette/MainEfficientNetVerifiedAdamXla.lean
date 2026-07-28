import apps.imagenette.EfficientNetAdamCommon

/-! # `efficientnet-verified-adam-xla` — the EfficientNet AdamW trainer on XLA/PJRT

The same program as `efficientnet-verified-adam` (shared body in
`apps/imagenette/EfficientNetAdamCommon.lean`), linked against `ffi/libpjrt_ffi.so`.

It exists for **multi-GPU**: collectives live only on the PJRT path, so the `adamdp` variant
(handoff §2e-bis) cannot run under IREE at all — the shim refuses the DP entry point rather than
silently running single-device. Single-device it is also a second trusted lowerer over the same
certified bytes, which is worth having on a net this convolution-heavy.

**This does not move the verification tier** (§9).

```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build efficientnet-verified-adam-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/efficientnet-verified-adam-xla data

# data-parallel over 2 GPUs
unset HIP_VISIBLE_DEVICES
LEAN_MLIR_VARIANT=adamdp LEAN_MLIR_REPLICAS=2 PJRT_REPLICAS=2 \
  .lake/build/bin/efficientnet-verified-adam-xla data
```
-/

def main (argv : List String) : IO Unit := runEfficientNetAdam argv
