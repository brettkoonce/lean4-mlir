"""Steady-state ms/step of the repo's own Lean-emitted JAX trainers (jax/generated/), on random data —
the idiomatic-JAX reference point for the verified-MLIR numbers on the same device.
  jax_bench.py {r34|vit} {f32|bf16} BATCH [STEPS=60 WARM=10]
  JAX_BENCH_WARM=1 jax_bench.py ...   # lower + compile train_step only, no execution (atlas's fake device)
tensorflow/tfds are stubbed: only the model, loss and train_step are used, not the input pipeline.
State is built on the CPU and moved over, so the only graph compiled for the device is train_step.
r34 = generated_resnet34_imagenet.py (SGD-momentum, 1000 classes, NCHW); vit = generated_vit_tiny_imagenet.py (Adam)."""
import os, sys, time, importlib.util, pathlib


class _Stub:                       # any attribute, any call: enough for the module-level tf/tfds lines
    def __getattr__(self, name): return self
    def __call__(self, *a, **k): return self
for m in ('tensorflow', 'tensorflow_datasets'): sys.modules[m] = _Stub()

net, dt, B = sys.argv[1], sys.argv[2], int(sys.argv[3])
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 60; WARM = int(sys.argv[5]) if len(sys.argv) > 5 else 10
src = {'r34': 'generated_resnet34_imagenet.py', 'vit': 'generated_vit_tiny_imagenet.py'}[net]
path = pathlib.Path(__file__).resolve().parents[2] / 'jax' / 'generated' / src
spec = importlib.util.spec_from_file_location('g', path); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
import jax, jax.numpy as jnp, numpy as np
D = jnp.float32 if dt == 'f32' else jnp.bfloat16
g.DT = D; g.CONV_DT = D          # read at trace time by mm()/convdt()
dev, cpu = jax.devices()[0], jax.devices('cpu')[0]
print(f'{net} {dt} B={B} backend={jax.default_backend()} device={dev} NEURON_CC_FLAGS={os.environ.get("NEURON_CC_FLAGS")}', flush=True)

with jax.default_device(cpu):
    params = g.init_params(jax.random.PRNGKey(0)); g.WD_MASK = g._wd_mask(params)   # main() builds the mask
    x = jnp.asarray(np.random.default_rng(0).standard_normal((B, 3 * 224 * 224), dtype=np.float32))
    y = jnp.asarray(np.random.default_rng(1).integers(0, 10, B, dtype=np.int32))
    lr = jnp.float32(1e-3)
    z = jax.tree.map(jnp.zeros_like, params)
    state = (z, g.init_bn_state()) if net == 'r34' else (z, z, jnp.float32(0))
put = lambda t: jax.device_put(t, dev)
args = (params, *state, x, y, lr) if net == 'r34' else (params, state, x, y, lr)

if os.environ.get('JAX_BENCH_WARM') == '1':
    t0 = time.time()
    abstract = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), args)
    try:
        g.train_step.lower(*abstract).compile()
        print(f'WARM compiled train_step in {time.time() - t0:.0f}s', flush=True)
    except Exception as e:     # the fake device fails when it loads the NEFF; the cache entry is already written
        print(f'WARM compile ended after {time.time() - t0:.0f}s: {str(e)[:200]}', flush=True)
    sys.exit(0)

params, state, x, y, lr = put(params), put(state), put(x), put(y), put(lr)
if net == 'r34':
    def step(p, s):
        p, v, bn, loss = g.train_step(p, s[0], s[1], x, y, lr); return p, (v, bn), loss
else:
    def step(p, s):
        p, s, loss = g.train_step(p, s, x, y, lr); return p, s, loss
t0 = time.time(); params, state, loss = step(params, state); jax.block_until_ready(loss)
print(f'first step (compile+run) {time.time() - t0:.1f}s loss {float(loss):.3f}', flush=True)
for _ in range(WARM): params, state, loss = step(params, state)
jax.block_until_ready(loss); t0 = time.time()
for _ in range(STEPS): params, state, loss = step(params, state)
jax.block_until_ready(loss); ms = (time.time() - t0) / STEPS * 1000
print(f'RESULT {net} {dt} B={B} {ms:.1f} ms/step  {ms / B:.2f} ms/img  loss {float(loss):.3f}', flush=True)
