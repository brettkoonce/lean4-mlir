import apps.imagenette.Resnet50AdamCommon

/-! # `resnet50-verified-adam-xla` — ResNet-50 on Imagenette, via XLA/PJRT

The same program as `resnet50-verified-adam` (shared body in
`apps/imagenette/Resnet50AdamCommon.lean`), linked against `ffi/libpjrt_ffi.so`.

The bottleneck peer of `resnet34-verified-adam` (on XLA): same harness, same 80-epoch
bs32 AdamW schedule, one net swapped. `resnet50Verified` was a layout skeleton
with no artifacts until `Proofs/Codegen/ResNet50RenderB.lean` grew its
`resnet50_*` renders.

**This does not move the verification tier**, and it moves it less than R34's row
does: R34/Imagenette carries §1a ties, `resnet50Verified` carries none yet. What
this trainer produces is a measurement on the certified renderer's output.

```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build resnet50-verified-adam-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/resnet50-verified-adam-xla data
```

⚠ `IREE_BACKEND` is inert here — PJRT does not read it.
-/

def main (argv : List String) : IO Unit := runResnet50Adam argv
