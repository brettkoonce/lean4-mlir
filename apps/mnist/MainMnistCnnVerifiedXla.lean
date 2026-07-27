import apps.mnist.CnnVerifiedCommon

/-! # `mnist-cnn-verified-xla` — the first CONVOLUTIONAL rung of the XLA ladder

The same program as `mnist-cnn-verified` (shared body in
`apps/mnist/CnnVerifiedCommon.lean`), linked against `ffi/libpjrt_ffi.so`.

This is the rung that matters for the *reason* the ladder exists. Rungs 0 and 1
are dense-only, so IREE's measured ~1%-of-peak convolution codegen never comes
into play and the speedups were modest (1.6× and 2.9×). `cnn_train_step` is the
first graph with real convolutions — `conv 1→32 → conv 32→32 → maxpool → dense
6272→512 → …` — which is where XLA's dispatch to MIOpen/cuDNN should show up.

It still needs no new shim code: `VerifiedNet.train` routes through
`iree_ffi_invoke_f32` regardless of whether the params are 2-D or 4-D.

**This does not move the verification tier** (§9). IREE and XLA are both
unverified lowerers on the same trusted tier.

```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build mnist-cnn-verified-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/mnist-cnn-verified-xla data
```
-/

def main (argv : List String) : IO Unit := runCnnVerified argv
