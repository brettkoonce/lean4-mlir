# Gradient Accumulation — the large-batch reproducibility lever

## Why

Paper-faithful reproduction of large-batch recipes (RSB's LAMB @ **bs2048**,
ViT/DeiT, etc.) normally needs an 8×A100-80GB node — the batch simply doesn't fit
in activations on modest GPUs. Gradient accumulation decouples *the batch size the
recipe needs* from *the batch size your GPUs can hold*: run `K` micro-batches of
`batchSize` and sum their grads before one optimizer step, giving an **effective
batch of `batchSize × K`** at the peak-activation cost of **one** micro-batch.

Concretely: RSB bs2048 becomes `batchSize=512 × gradAccumSteps=4` on a 4×16 GB box
(the klawd 4060 Ti's) instead of renting a DGX. It trades wall-clock (K× forward/
backward per optimizer step) for memory — the exact lever a low-end user needs.

This was motivated by the R50 RSB-A3 diagnosis (see `project_r50_a3_lowval_diagnostic`
in memory): LAMB at bs512 landed 40.8% vs the ~78% target because **LAMB is a
large-batch optimizer** and was starved of its batch. AdamW@bs512 fixed it (46.8%
by ep10), but to reproduce the paper's LAMB@bs2048 *faithfully* on small GPUs, you
need this lever.

## Config lever (`LeanMlir/Types.lean`)

```
gradAccumSteps : Nat := 1
```
- `MICRO = batchSize` (per-forward, memory-bound)
- `ACCUM = gradAccumSteps`
- `EFFECTIVE = MICRO × ACCUM` — drives the data batch, `steps_per_epoch`, and the
  LR schedule. **`learningRate` must target the EFFECTIVE batch** (e.g. RSB A3's
  8e-3@2048, not the 512-scaled 2e-3).
- `N ≤ 1` → single-shot update, **byte-identical codegen** (verified). ImageNet-
  streaming main only.

## Codegen design (`jax/Jax/Codegen.lean`)

### train_step (`emitLossAndTraining`, `gradPrelude`)
The effective batch is reshaped into `ACCUM` micro-batches and `lax.scan`-ed for
grad accumulation before the (unchanged) optimizer update runs **once**:

```python
_K = GRAD_ACCUM
_xs = x.reshape(_K, x.shape[0] // _K, *x.shape[1:])
_ys = y.reshape(_K, y.shape[0] // _K, *y.shape[1:])
_xs = jax.lax.with_sharding_constraint(_xs, NamedSharding(mesh, P(None, 'batch')))
_ys = jax.lax.with_sharding_constraint(_ys, NamedSharding(mesh, P(None, 'batch')))
def _accum(_carry, _inp):
    _gacc, _bnc = _carry
    _xi, _yi = _inp
    (_l, _nbn), _g = value_and_grad(loss_fn, has_aux=True)(params, _bnc, _xi, _yi)
    return (jax.tree.map(jnp.add, _gacc, _g), _nbn), _l
_g0 = jax.tree.map(jnp.zeros_like, params)
(_gsum, _new_bn), _ls = jax.lax.scan(_accum, (_g0, bn), (_xs, _ys))
grads = jax.tree.map(lambda _a: _a / _K, _gsum)
loss = jnp.mean(_ls)
# ... existing LAMB / AdamW / SGD block runs ONCE on `grads` ...
```
It produces the same `grads`/`_new_bn`/`loss` names the optimizer blocks consume,
so all five optimizer branches stay untouched. When `dropPath>0` a `drop_key` is
split `K`-ways and threaded through the scan.

### The sharding subtlety (the one non-obvious bit)
`x` arrives sharded on axis-0 (`EFFECTIVE` rows split across GPUs). Reshaping to
`(K, micro, F)` makes GSPMD shard the **K axis** — one whole micro-batch per GPU →
`micro` activations/GPU → OOM, defeating the purpose. The two
`with_sharding_constraint(..., P(None, 'batch'))` calls pin the **micro axis**
sharded instead, so each scan step's forward is `micro/n_devices` per GPU — peak
activation stays at one micro-batch. Costs one reshard (all-to-all) per step.

### accounting (`emitMainImagenet`)
- `MICRO_BATCH = (batchSize // n_devices)*n_devices`; `BATCH_SIZE = MICRO_BATCH * GRAD_ACCUM`
- `steps_per_epoch = train_examples // BATCH_SIZE`, LR schedule in EFFECTIVE steps
- train iter batches at `BATCH_SIZE` (effective); **validation batches at
  `MICRO_BATCH`** (eval_batch does one forward — can't fit the effective batch)
- mixup/cutmix apply to the full effective batch in the Python loop (unchanged),
  then train_step reshapes — fine, the mixed images just distribute across micros

## Caveats (read before trusting a reproduction)

1. **Ghost-BN semantics.** BN computes stats per micro-batch (`micro`), not over the
   full effective batch — grad accumulation can't see across micro-steps. This is
   *Ghost BatchNorm* (Hoffer et al. 2017), a benign/often-beneficial variant at
   `micro ≥ 256`. Not bit-identical to true large-batch BN. (Exact large-batch BN
   would need a 2-pass sync — not worth it.)
2. **No throughput win — memory-for-time.** K forward/backward per optimizer step
   = K× per-step wall-clock. Same total FLOPs as the real large batch, serialized.
   That *is* the point: same result, cheaper hardware, more hours.
3. **LR targets EFFECTIVE batch** — set `learningRate` to the paper's large-batch
   value, not the micro-batch-scaled one.
4. **Compile cost.** `lax.scan` keeps the jaxpr small (one micro traced), so compile
   time ≈ unchanged. The per-step reshard adds comms.

## Usage

```lean
def resnet50ImagenetConfigRSBFaithful : TrainConfig :=
  { resnet50ImagenetConfigShort with
      learningRate   := 0.008      -- RSB A3 lr @ bs2048 (NOT the 512-scaled 0.002)
      optimizer      := .lamb      -- the paper's optimizer, now with its real batch
      gradAccumSteps := 4 }        -- 512 × 4 = effective bs2048 on 4×16GB
```

Pairs with `wdExcludeNormBias` (timm no_weight_decay skip-list) as the
"reproducibility levers" set: batch size you can't afford + the decay skip-list let
a modest box match the paper.

## Status
- Implemented + built. `gradAccumSteps=1` verified **byte-identical** to prior
  codegen (no regression across all nets). `gradAccumSteps=4` generates valid,
  `py_compile`-clean code with the correct scan + sharding-constraint + accounting.
- **GPU-validated** (R50, accum=4, effective bs2048, 4×4060 Ti): no OOM; peak
  memory **~12.25 GB/GPU — identical to bs512 single-micro**, confirming the
  `with_sharding_constraint` kept the micro axis sharded (not the accum axis).
  Config reported effective 2048 / micro 512 (4×128) / steps_per_epoch 625.
  Steady step ≈ 4× the single-micro time → throughput ≈ bs512 (memory-for-time, no
  throughput change, as designed). The Python-loop fallback is **not** needed.
- Remaining open item: a full training run confirming LAMB@effective-2048 (via
  accum) recovers the RSB accuracy target — an accuracy check, not a mechanics one.
