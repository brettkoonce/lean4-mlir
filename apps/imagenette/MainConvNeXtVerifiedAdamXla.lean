import apps.imagenette.ConvNeXtAdamCommon

/-! # `convnext-verified-adam-xla` — the ConvNeXt-T AdamW trainer on XLA/PJRT

The same program as `convnext-verified-adam` (shared body in
`apps/imagenette/ConvNeXtAdamCommon.lean`), linked against `ffi/libpjrt_ffi.so`.

Two reasons it exists (handoff §2h):

* **Speed.** XLA measured **4.6×** IREE on EfficientNet (§2e-quinquies), and ConvNeXt was IREE-only,
  so every long run was paying that factor. Re-measure per net rather than assuming 4.6× — that is
  one net's number, on a depthwise-convolution-heavy net.
* **Multi-GPU is only reachable this way.** Collectives live only on the PJRT path; the IREE shim
  refuses a DP entry point outright. ConvNeXt was the last large net with **no `replicas` support in
  its renderer at all**; that closed on 2026-07-29 (§2h-quater) and this binary is what runs it:

  ```
  unset HIP_VISIBLE_DEVICES
  LEAN_MLIR_VARIANT=adamdp LEAN_MLIR_REPLICAS=2 PJRT_REPLICAS=2 \
    .lake/build/bin/convnext-verified-adam-xla data
  ```

  Measured: **1.68×** on 2× 7900 XTX (marginal epoch, train-only, 77.5 s → 46.0 s), which is the
  same figure EfficientNet and mnv2 reach — the shortfall from 2× is the host-resident `[θ|m|v]`
  push, not anything about the net (§2d.3).

The risk that ViT's `miopenStatusUnknownError` blocker generalised was real enough to measure first:
ConvNeXt has a structurally similar op, the 4×4/s4 patchify weight gradient
(`convStride4WeightGrad`, §2f-bis), whose cotangent dilates to a large filter exactly as ViT's
16×16/s16 does. Measured (§2h): `@convnext_train_step` compiles in 6.6 s on XLA/PJRT and **executes**,
A-vs-A gradient bit-exact, 0/180 params disagreeing. **So ConvNeXt is not ViT-shaped**, and that
further narrows the ViT blocker to its one 16×16/s16 patch-embed shape rather than to strided weight
gradients as a class.

**This does not move the verification tier** (§9).

```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build convnext-verified-adam-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/convnext-verified-adam-xla data
```
-/

def main (argv : List String) : IO Unit := runConvNeXtAdam argv
