#!/usr/bin/env python
"""PGD adversarial attack vs Lipschitz certificate on an MNIST MLP (planning/robustness.md).

The two sides of adversarial robustness, bracketing the truth:
    certified robust acc  <=  TRUE robust acc  <=  PGD-empirical robust acc
     (proof, all attacks)                          (one strong attack)

- PGD  (Madry et al. 2017): gradient *ascent on the input*, projected to an eps-ball — the
  empirical UPPER bound on robust accuracy (finds adversarial examples; can't prove safety).
- Lipschitz certificate (Tsuzuku et al. 2018): global Lipschitz L = prod ||W_i||_2 (ReLU is
  1-Lipschitz), per-example margin m(x); any L2 perturbation < m(x)/(sqrt(2)*L) provably cannot
  flip the prediction — the LOWER bound (holds vs every attack).

Net: 784->512->512->10 ReLU MLP (matches the verified `mnistMlp`). Demo-first JAX phase-2 path:
JAX autodiff gives the input gradient `grad_x` for free. UNVERIFIED (a future in-system version
would reuse the proven input-VJP `mlpInputGrad_floatBridges`).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import jax, jax.numpy as jnp
from functools import partial

SEED = 0
EPOCHS = 8
BATCH = 128
N_PGD = 2000          # test-subset size for the (expensive) PGD eval
PGD_STEPS = 40
rng = np.random.default_rng(SEED)

# ---- data -------------------------------------------------------------------
from tensorflow import keras
(xtr, ytr), (xte, yte) = keras.datasets.mnist.load_data()
xtr = (xtr.reshape(-1, 784) / 255.0).astype(np.float32)
xte = (xte.reshape(-1, 784) / 255.0).astype(np.float32)
ytr = ytr.astype(np.int32); yte = yte.astype(np.int32)

# ---- model: 784 -> 512 -> 512 -> 10, ReLU -----------------------------------
def init_params(key):
    ks = jax.random.split(key, 3)
    def he(k, fin, fout): return jax.random.normal(k, (fin, fout)) * np.sqrt(2.0 / fin)
    return {
        "W0": he(ks[0], 784, 512), "b0": jnp.zeros(512),
        "W1": he(ks[1], 512, 512), "b1": jnp.zeros(512),
        "W2": he(ks[2], 512, 10),  "b2": jnp.zeros(10),
    }

def logits_fn(p, x):
    h = jax.nn.relu(x @ p["W0"] + p["b0"])
    h = jax.nn.relu(h @ p["W1"] + p["b1"])
    return h @ p["W2"] + p["b2"]

def ce_loss(p, x, y):
    lg = logits_fn(p, x)
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(lg), y[:, None], axis=1))

# ---- train (manual Adam) ----------------------------------------------------
@partial(jax.jit, static_argnums=())
def adam_step(p, m, v, x, y, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    g = jax.grad(ce_loss)(p, x, y)
    m = jax.tree_util.tree_map(lambda m, g: b1*m + (1-b1)*g, m, g)
    v = jax.tree_util.tree_map(lambda v, g: b2*v + (1-b2)*g*g, v, g)
    mh = jax.tree_util.tree_map(lambda m: m/(1-b1**t), m)
    vh = jax.tree_util.tree_map(lambda v: v/(1-b2**t), v)
    p = jax.tree_util.tree_map(lambda p, mh, vh: p - lr*mh/(jnp.sqrt(vh)+eps), p, mh, vh)
    return p, m, v

@jax.jit
def accuracy(p, x, y):
    return jnp.mean(jnp.argmax(logits_fn(p, x), 1) == y)

def train():
    p = init_params(jax.random.PRNGKey(SEED))
    m = jax.tree_util.tree_map(jnp.zeros_like, p)
    v = jax.tree_util.tree_map(jnp.zeros_like, p)
    t = 0
    for ep in range(EPOCHS):
        perm = rng.permutation(len(xtr))
        for i in range(0, len(xtr) - BATCH, BATCH):
            idx = perm[i:i+BATCH]
            t += 1
            p, m, v = adam_step(p, m, v, jnp.asarray(xtr[idx]), jnp.asarray(ytr[idx]), t)
        acc = float(accuracy(p, jnp.asarray(xte), jnp.asarray(yte)))
        print(f"  epoch {ep+1}/{EPOCHS}: test acc = {acc*100:.2f}%")
    return p

# ---- PGD attack -------------------------------------------------------------
@partial(jax.jit, static_argnums=(4, 5))
def pgd(p, x, y, eps, steps, linf):
    """Projected gradient ascent on the input. linf=True -> L-inf ball, else L2."""
    alpha = (eps / 4.0) if not linf else (2.5 * eps / steps)
    # random start inside the ball
    if linf:
        d = jax.random.uniform(jax.random.PRNGKey(1), x.shape, minval=-eps, maxval=eps)
    else:
        d = jax.random.normal(jax.random.PRNGKey(1), x.shape); d = d / (jnp.linalg.norm(d, axis=1, keepdims=True)+1e-12) * eps
    xa = jnp.clip(x + d, 0.0, 1.0)
    def body(_, xa):
        g = jax.grad(lambda xx: ce_loss(p, xx, y))(xa)
        if linf:
            xa = xa + alpha * jnp.sign(g)
            xa = jnp.clip(xa, x - eps, x + eps)
        else:
            gn = g / (jnp.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
            xa = xa + alpha * gn
            d = xa - x
            n = jnp.linalg.norm(d, axis=1, keepdims=True)
            d = d * jnp.minimum(1.0, eps / (n + 1e-12))
            xa = x + d
        return jnp.clip(xa, 0.0, 1.0)
    xa = jax.lax.fori_loop(0, steps, body, xa)
    return xa

# ---- Lipschitz certificate --------------------------------------------------
def certify(p, x, y):
    # global L2 Lipschitz of the logit map = product of layer spectral norms (ReLU is 1-Lip)
    sv = [float(jnp.linalg.svd(p[w], compute_uv=False)[0]) for w in ("W0", "W1", "W2")]
    L = sv[0] * sv[1] * sv[2]
    lg = np.asarray(logits_fn(p, x))
    pred = lg.argmax(1)
    top2 = np.sort(lg, 1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]                       # top1 - top2
    cert_radius = margin / (np.sqrt(2.0) * L)              # L2 radius provably safe
    cert_radius = np.where(pred == np.asarray(y), cert_radius, 0.0)  # wrong preds: radius 0
    return L, sv, cert_radius

# ---- run --------------------------------------------------------------------
if __name__ == "__main__":
    print(f"jax {jax.__version__} on {jax.devices()}")
    print("Training MNIST MLP 784-512-512-10 ...")
    p = train()
    clean = float(accuracy(p, jnp.asarray(xte), jnp.asarray(yte)))
    print(f"\nClean test accuracy: {clean*100:.2f}%\n")

    xs = jnp.asarray(xte[:N_PGD]); ys = jnp.asarray(yte[:N_PGD])

    print("=== L-inf PGD (the standard MNIST benchmark, Madry et al.) ===")
    print(f"{'eps':>6} | {'PGD adv acc':>12}")
    for eps in (0.0, 0.1, 0.2, 0.3):
        xa = pgd(p, xs, ys, float(eps), PGD_STEPS, True) if eps > 0 else xs
        print(f"{eps:6.2f} | {float(accuracy(p, xa, ys))*100:11.2f}%")

    print("\n=== L2 sandwich: PGD (upper) vs Lipschitz certificate (lower) ===")
    L, sv, cert_r = certify(p, xs, ys)
    print(f"layer spectral norms ||W||_2 = [{sv[0]:.2f}, {sv[1]:.2f}, {sv[2]:.2f}]  =>  global L = {L:.1f}")
    print(f"median certified L2 radius = {np.median(cert_r):.4f}  (correctly-classified)")
    print(f"{'L2 eps':>6} | {'PGD adv acc':>12} | {'certified acc':>13}")
    for eps in (0.5, 1.0, 1.5, 2.0):
        xa = pgd(p, xs, ys, float(eps), PGD_STEPS, False)
        pgd_acc = float(accuracy(p, xa, ys))
        cert_acc = float(np.mean(cert_r >= eps))
        print(f"{eps:6.2f} | {pgd_acc*100:11.2f}% | {cert_acc*100:12.2f}%")

    print("\nTakeaway: PGD crushes the undefended net (upper bound -> ~0); the naive Lipschitz")
    print("certificate proves only a tiny safe radius (lower bound). The gap is the research.")
