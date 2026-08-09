import apps.imagenette.MobilenetV4AdamCommon

/-! # `mobilenetv4-verified-adam-xla` — the MobileNetV4-Conv-S AdamW trainer on XLA/PJRT

Shared body in `apps/imagenette/MobilenetV4AdamCommon.lean`, linked against `ffi/libpjrt_ffi.so`.

Phase 4 of `planning/mnv4_verified.md`: 80 epochs, bs32, AdamW, target **84.58%** — the JAX-baseline
path's number for this block table. The forward and the gradient are both tied against that
reference (§3e, §3i), so the two paths are the same net and the number is a reproduction rather than
a fresh measurement.

⚠ **This does not move the verification tier.** Every op the render composes carries a proven `den`,
but MNv4 has no composed-backward theorem yet — see `MobilenetV4AdamCommon`'s header.

```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build mobilenetv4-verified-adam-xla

# ⚠ the data arg is the ROOT, not the dataset dir — `loadData` appends `/imagenette` itself.
#   Passing `data/imagenette` fails only at the loader, AFTER every artifact compiles and the
#   full header prints, which reads like a data problem and is an argv problem (§3h trap 1).
# ⚠ and move any stale checkpoint aside first, or the run resumes the OLD net and exits zero:
mv .lake/build/mnv4_adam_ckpt_xla.bin{,.bak} 2>/dev/null

HIP_VISIBLE_DEVICES=0 .lake/build/bin/mobilenetv4-verified-adam-xla data
```
-/

def main (argv : List String) : IO Unit := runMobilenetV4Adam argv
