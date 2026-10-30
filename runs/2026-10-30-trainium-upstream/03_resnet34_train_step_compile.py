"""neuronx-cc fails to compile a textbook ResNet-34 training step (f32, batch 32, 224x224, SGD).
Plain JAX, no frameworks: conv + batch-norm (batch statistics) + ReLU, the standard 3-4-6-3 basic-block
stack, global average pool, a linear head, softmax cross-entropy, one SGD update. The same step compiles
with bf16 activations, or at batch 16. Compile only: lower().compile(), no execution.
  python 03_resnet34_train_step_compile.py [BATCH=32] [f32|bf16]"""
import sys, time, jax, jax.numpy as jnp
from jax import lax
B = int(sys.argv[1]) if len(sys.argv) > 1 else 32
DT = {'f32': jnp.float32, 'bf16': jnp.bfloat16}[sys.argv[2] if len(sys.argv) > 2 else 'f32']

def conv(x, w, s=1):
    k = w.shape[-1]
    return lax.conv_general_dilated(x.astype(DT), w.astype(DT), (s, s), [((k - 1) // 2,) * 2] * 2,
                                    dimension_numbers=('NCHW', 'OIHW', 'NCHW')).astype(jnp.float32)

def bn(x, g, b):
    m = x.mean((0, 2, 3), keepdims=True); v = x.var((0, 2, 3), keepdims=True)
    return (x - m) * lax.rsqrt(v + 1e-5) * g.reshape(1, -1, 1, 1) + b.reshape(1, -1, 1, 1)

def init(key):
    ks = iter(jax.random.split(key, 64)); p = {}
    he = lambda o, i, k: jax.random.normal(next(ks), (o, i, k, k)) * (2 / (i * k * k)) ** 0.5
    p['stem'] = (he(64, 3, 7), jnp.ones(64), jnp.zeros(64)); blocks = []; cin = 64
    for cout, n, s in [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]:
        for j in range(n):
            st = s if j == 0 else 1
            blk = {'c1': (he(cout, cin, 3), jnp.ones(cout), jnp.zeros(cout)),
                   'c2': (he(cout, cout, 3), jnp.ones(cout), jnp.zeros(cout)), 's': st}
            if st != 1 or cin != cout: blk['sc'] = (he(cout, cin, 1), jnp.ones(cout), jnp.zeros(cout))
            blocks.append(blk); cin = cout
    p['fc'] = (jax.random.normal(next(ks), (512, 10)) * 0.01, jnp.zeros(10))
    return p, blocks

def forward(p, blocks, x):
    w, g, b = p['stem']; x = jax.nn.relu(bn(conv(x, w, 2), g, b))
    x = lax.reduce_window(x, -jnp.inf, lax.max, (1, 1, 3, 3), (1, 1, 2, 2), [(0, 0), (0, 0), (1, 1), (1, 1)])
    for blk, bp in zip(blocks, p['blocks']):
        s = blk['s']; h = jax.nn.relu(bn(conv(x, bp['c1'][0], s), *bp['c1'][1:]))
        h = bn(conv(h, bp['c2'][0]), *bp['c2'][1:])
        sc = bn(conv(x, bp['sc'][0], s), *bp['sc'][1:]) if 'sc' in bp else x
        x = jax.nn.relu(h + sc)
    x = x.mean((2, 3)); return x @ p['fc'][0] + p['fc'][1]

with jax.default_device(jax.devices('cpu')[0]):   # shapes only; nothing runs on the device
    p, blocks = init(jax.random.PRNGKey(0))
params = {'stem': p['stem'], 'fc': p['fc'], 'blocks': [{k: v for k, v in b.items() if k != 's'} for b in blocks]}

def loss(params, x, y):
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(forward(params, blocks, x)), y[:, None], 1))

@jax.jit
def train_step(params, x, y):
    l, g = jax.value_and_grad(loss)(params, x, y)
    return jax.tree.map(lambda w, d: w - 0.1 * d, params, g), l

dev = jax.devices()[0]
print(f'jax {jax.__version__}, device {dev}, batch {B}, {DT.__name__}', flush=True)
a = jax.tree.map(lambda t: jax.ShapeDtypeStruct(t.shape, t.dtype), params)
t0 = time.time()
try:
    train_step.lower(a, jax.ShapeDtypeStruct((B, 3, 224, 224), jnp.float32), jax.ShapeDtypeStruct((B,), jnp.int32)).compile()
    print(f'compiled in {time.time() - t0:.0f}s')
except Exception as e:
    import re; m = str(e); codes = sorted(set(re.findall(r'(?:NCC_|IBIR)[A-Z0-9]+', m)))
    print(f'after {time.time() - t0:.0f}s: {type(e).__name__} {codes}\n{m[:300]}\n...\n{m[-900:]}')
