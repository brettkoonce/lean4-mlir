import apps.cifar.Cifar8BnCommon

/-! # `cifar8-bn-verified-xla` — the XLA/PJRT peer of `cifar8-bn-verified`

The same program as `cifar8-bn-verified` (shared body in
`apps/cifar/Cifar8BnCommon.lean`), linked against `ffi/libpjrt_ffi.so` instead of
`ffi/libiree_ffi.so`, consuming the same `verified_mlir/cifar8_bn_train_step.mlir`
and `cifar8_bn_fwd.mlir`.

**Why this exists: it is the conv anchor for `lake run benchmark-xla`**
(`planning/xla_pjrt_handoff.md` §2j). Every other cifar `-xla` target goes through
`trainAdamSched`; this net's IREE driver goes through `VerifiedNet.train`, whose SGD
update and lr are baked into the artifact. Probing a *different* optimizer on the XLA
side would need its own reference constant and would stop the two benchmark tables
being comparable row by row, so the peer was added instead.

It needed **no new shim code** and adds no new rung: `VerifiedNet.train` drives
`lean_iree_mlp_train_step_v` and `lean_iree_forward_f32`, both of which route through
`iree_ffi_invoke_f32` that `ffi/pjrt_ffi.c` already implements — the same path rung 1
(`mnist-mlp-verified-xla`) established. Per-channel BN is per-example, so there are no
running stats and the still-stubbed `train_step_adam*` family is not reached.

Two hazards that bite the AdamW peers do **not** apply here: `VerifiedNet.train` keeps
no optimizer state and writes no checkpoint (so §4's stale-checkpoint trap is out of
scope), and `mkSession` on the XLA path compiles in-process without ever writing a
`.vmfb` (so the cross-backend `.vmfb`-reuse trap is out of scope too).

**This does not move the verification tier** (§9). IREE and XLA are both unverified
lowerers on the same trusted tier.

```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build cifar8-bn-verified-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/cifar8-bn-verified-xla data
```
-/

def main (argv : List String) : IO Unit := runCifar8Bn argv
