import apps.cifar.Cifar8BnAdamCommon

/-! # `cifar8-bn-verified-adam-xla` — rung 2 of the XLA ladder

The same program as `cifar8-bn-verified-adam` (shared body in
`apps/cifar/Cifar8BnAdamCommon.lean`), linked against `ffi/libpjrt_ffi.so`.

Rung 2 introduces **Adam moment buffers** and **runtime scalars**. Both ride
inside the packed params blob: `adamShapes` is
`paramShapes ++ paramShapes ++ paramShapes ++ #[#[], #[], #[]]`, i.e. `[θ|m|v]`
followed by three **rank-0** entries for `lr`, `bc₁`, `bc₂`. So the graph is driven
through `mlpTrainStepV` → `iree_ffi_invoke_f32` like rungs 0–1, and the genuinely
new thing for the XLA shim is **rank-0 tensor inputs**
(`PJRT_Client_BufferFromHostBuffer` with `num_dims = 0`).

Note this net has **no BN running stats** — per-channel BN is per-example, so
train == eval, `bnChannels` is empty, and the γ/β are Adam-updated like any other
parameter. The `train_step_adam*` FFI family (which does carry running stats out
in passthrough slots) is still unported and still stubbed; R34 at rung 3 is where
that finally has to be written.

**Checkpoints are backend-scoped** (`..._ckpt_xla.bin`). Without that, this
driver's auto-resume would happily continue an XLA run from an IREE checkpoint,
fusing two trajectories into one while looking entirely normal.

**This does not move the verification tier** (§9).

```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build cifar8-bn-verified-adam-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/cifar8-bn-verified-adam-xla data
```
-/

def main (argv : List String) : IO Unit := runCifar8BnAdam argv
