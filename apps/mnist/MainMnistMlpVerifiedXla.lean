import apps.mnist.MlpVerifiedCommon

/-! # `mnist-mlp-verified-xla` — rung 1 of the XLA ladder

The same program as `mnist-mlp-verified` (shared body in
`apps/mnist/MlpVerifiedCommon.lean`), linked against `ffi/libpjrt_ffi.so` instead
of `ffi/libiree_ffi.so`, consuming the same `verified_mlir/mlp_train_step.mlir`
and `mlp_fwd.mlir`.

Rung 1 adds what rung 0 (the linear classifier) could not exercise: **depth and
multiple parameter tensors** — 6 of them, 669,706 floats, threaded through the
packed-params `mlpTrainStepV` path rather than the 2-argument linear one. It
needed no new shim code: `lean_iree_mlp_train_step_v` and `lean_iree_forward_f32`
both route through `iree_ffi_invoke_f32`, which `ffi/pjrt_ffi.c` already
implements.

It also exercises **He init**, which rung 0 dodged by being zero-init. That is
handled the way `planning/xla_pjrt_ladder.md` §4 prescribes: Lean owns the init
(`mkParam`, seeded by `LEAN_MLIR_SEED`, default 1), so both backends start from a
byte-identical state with no parameter names to reverse-engineer.

**This does not move the verification tier** (§9). IREE and XLA are both
unverified lowerers on the same trusted tier.

```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build mnist-mlp-verified-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/mnist-mlp-verified-xla data
```
-/

def main (argv : List String) : IO Unit := runMlpVerified argv
