"""PoC: functional running-BN-stats for the phase-2 JAX trainers (gap A).

De-risks the conceptually-hard part of the running-BN fix BEFORE the big codegen
rollout: the JAX functional threading. Proves on the GPU that
  1. running mean/var are updated by an EMA of batch stats during training,
     threaded out of `forward` as has_aux (NOT differentiated);
  2. gradients flow only to `params`, never to the BN buffers;
  3. eval with running stats DIFFERS from eval with batch stats — i.e. it
     actually does the faithful thing the current trainer skips.

Standalone (tiny conv-BN net), so it validates the mechanism without touching
the shared codegen. The full rollout = applying this thread to the ~20 inline-BN
sites in the block helpers + the imagenet loss/train/eval/state plumbing.
"""
import jax, jax.numpy as jnp
from jax import random, value_and_grad, jit

def _conv(x, w, stride=1):
    return jax.lax.conv_general_dilated(x, w, (stride, stride), 'SAME',
                                        dimension_numbers=('NCHW', 'OIHW', 'NCHW'))

def conv_bn(x, w, gamma, beta, rm, rv, training, momentum=0.99, eps=1e-5):
    """The proposed conv_bn: returns (out, new_running_mean, new_running_var).
    Train normalises with batch stats + EMA-updates the running buffers; eval
    normalises with the running buffers and leaves them unchanged."""
    x = _conv(x, w)
    if training:
        bm = jnp.mean(x, axis=(0, 2, 3))
        bv = jnp.var(x, axis=(0, 2, 3))
        xn = (x - bm.reshape(1, -1, 1, 1)) / jnp.sqrt(bv.reshape(1, -1, 1, 1) + eps)
        new_rm = momentum * rm + (1.0 - momentum) * bm
        new_rv = momentum * rv + (1.0 - momentum) * bv
    else:
        xn = (x - rm.reshape(1, -1, 1, 1)) / jnp.sqrt(rv.reshape(1, -1, 1, 1) + eps)
        new_rm, new_rv = rm, rv
    return xn * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1), new_rm, new_rv

def forward(params, x, bn_state, training):
    (w0, g0, b0), (w1, g1, b1), (wd, bd) = params
    (rm0, rv0), (rm1, rv1) = bn_state
    x, n0m, n0v = conv_bn(x, w0, g0, b0, rm0, rv0, training); x = jax.nn.relu(x)
    x, n1m, n1v = conv_bn(x, w1, g1, b1, rm1, rv1, training); x = jax.nn.relu(x)
    x = jnp.mean(x, axis=(2, 3))                       # global avg pool -> [B, C]
    return x @ wd + bd, [(n0m, n0v), (n1m, n1v)]

def loss_fn(params, bn_state, x, y):
    logits, new_bn = forward(params, x, bn_state, training=True)
    lp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(jax.nn.one_hot(y, logits.shape[-1]) * lp, axis=-1)), new_bn

@jit
def train_step(params, bn_state, x, y, lr):
    (loss, new_bn), grads = value_and_grad(loss_fn, has_aux=True)(params, bn_state, x, y)
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return params, new_bn, loss, grads

print("devices:", jax.devices(), "| kind:", jax.devices()[0].device_kind)
assert jax.devices()[0].platform == "gpu", f"not on a GPU: {jax.devices()[0]}"
k = random.PRNGKey(0)
C = 8
params = [
    (random.normal(random.PRNGKey(1), (C, 3, 3, 3)) * 0.1, jnp.ones(C), jnp.zeros(C)),
    (random.normal(random.PRNGKey(2), (C, C, 3, 3)) * 0.1, jnp.ones(C), jnp.zeros(C)),
    (random.normal(random.PRNGKey(3), (C, 10)) * 0.1, jnp.zeros(10)),
]
bn_state = [(jnp.zeros(C), jnp.ones(C)), (jnp.zeros(C), jnp.ones(C))]  # standard BN init
x = random.normal(random.PRNGKey(4), (16, 3, 32, 32))
y = random.randint(random.PRNGKey(5), (16,), 0, 10)

for step in range(20):
    params, bn_state, loss, grads = train_step(params, bn_state, x, y, jnp.float32(0.05))

# 1. running stats actually moved from init (mean 0, var 1)
rm_moved = float(jnp.max(jnp.abs(bn_state[0][0])))
rv_moved = float(jnp.max(jnp.abs(bn_state[0][1] - 1.0)))
assert rm_moved > 1e-4 and rv_moved > 1e-4, "running stats did not update"
print(f"[ok] running stats updated by EMA: max|mean|={rm_moved:.3f}, max|var-1|={rv_moved:.3f}")

# 2. gradients have the params structure but NOT the bn_state (buffers undifferentiated)
assert jax.tree.structure(grads) == jax.tree.structure(params), "grad tree != params tree"
print(f"[ok] grads cover params only ({len(jax.tree.leaves(grads))} leaves); BN buffers carry no grad")

# 3. eval with running stats DIFFERS from eval with batch stats (the faithful fix)
logits_running, _ = forward(params, x, bn_state, training=False)
logits_batch, _ = forward(params, x, bn_state, training=True)
d = float(jnp.max(jnp.abs(logits_running - logits_batch)))
assert d > 1e-3, "running-stat eval == batch-stat eval (fix would be a no-op)"
assert bool(jnp.all(jnp.isfinite(logits_running))), "non-finite eval logits"
print(f"[ok] eval(running) != eval(batch): max|Δlogit|={d:.3f}  (this is the gap A closes)")
print("\nRUNNING-BN MECHANISM PROVEN")
