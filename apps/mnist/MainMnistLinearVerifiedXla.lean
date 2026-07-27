import apps.mnist.LinearVerifiedCommon

/-! # `mnist-linear-verified-xla` — rung 0 of the XLA ladder

Bit-for-bit the same program as `mnist-linear-verified` (shared body in
`apps/mnist/LinearVerifiedCommon.lean`), linked against `ffi/libpjrt_ffi.so`
instead of `ffi/libiree_ffi.so`. It consumes the *same*
`verified_mlir/linear_train_step.mlir` and `linear_fwd.mlir`; only the trusted
lowerer changes, IREE → XLA.

**This does not move the verification tier.** The theorem is "emitted graph ≡
spec". IREE and XLA are both unverified lowerers occupying the same trusted tier
(`planning/xla_pjrt_ladder.md` §8). Swapping between them buys nothing and costs
nothing in what is proven.

The point of running both is gate **G2**: from byte-identical initial parameters
(here, zero-init — so identical for free), the two backends must produce the same
parameters step for step. Only the *forward* has been tied so far; "the forward
matches so the step must" is the inference this project has been burned by.

Build the shim, then the exe:
```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build mnist-linear-verified-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/mnist-linear-verified-xla data
```
Override the plugin location with `$PJRT_PLUGIN`.
-/

def main (argv : List String) : IO Unit := runLinearVerified argv
