import apps.imagenette.MobilenetV2AdamCommon

/-! # `mobilenetv2-verified-adam-xla` — the MobileNetV2 AdamW trainer on XLA/PJRT

The same program as `mobilenetv2-verified-adam` (shared body in
`apps/imagenette/MobilenetV2AdamCommon.lean`), linked against `ffi/libpjrt_ffi.so`.

Two reasons it exists (handoff §2h):

* **Speed.** XLA measured **4.6×** IREE on EfficientNet (§2e-quinquies), and mnv2 was IREE-only, so
  every long run was paying that factor. Re-measure per net rather than assuming 4.6× — that is one
  net's number.
* **Multi-GPU is only reachable this way.** Collectives live only on the PJRT path; the IREE shim
  refuses a DP entry point outright. mnv2's renderer already takes `replicas`, but **no `#eval`
  writes a DP artifact yet**, so that is a later step — unreachable without this binary first.

The risk that ViT's `miopenStatusUnknownError` blocker generalised was **measured, not assumed**:
`@mobilenetv2_train_step` compiles in 3.0 s on XLA/PJRT and executes, A-vs-A gradient bit-exact
2253738/2253738 (§2h). mnv2's depthwise weight gradients are EfficientNet's, which §2e-bis already
runs on XLA.

**This does not move the verification tier** (§9).

```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build mobilenetv2-verified-adam-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/mobilenetv2-verified-adam-xla data
```
-/

def main (argv : List String) : IO Unit := runMobilenetV2Adam argv
