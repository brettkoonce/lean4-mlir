import apps.imagenette.Resnet34AdamCommon

/-! # `resnet34-verified-adam-xla` — rung 3 of the XLA ladder

The same program as `resnet34-verified-adam` (shared body in
`apps/imagenette/Resnet34AdamCommon.lean`), linked against `ffi/libpjrt_ffi.so`.

**This is the rung that decides the migration.** Every measurement below it was on
a small net where IREE's convolution weakness barely applies — the speedups came in
at 1.6× / 2.9× / 3.2× / 1.32×, nowhere near the 20–40× that motivated the whole
exercise. R34 at 224² with 64–512 channels is the regime those numbers came from.

It is also the first net with **BN running statistics** (`bnChannels` non-empty):
the train step carries per-layer batch mean/var out in passthrough slots, the
driver EMAs them, and eval runs `@resnet34_fwd_eval` with the running stats rather
than batch stats.

**This does not move the verification tier** (§9).

```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build resnet34-verified-adam-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/resnet34-verified-adam-xla data
```
-/

def main (argv : List String) : IO Unit := runResnet34Adam argv
