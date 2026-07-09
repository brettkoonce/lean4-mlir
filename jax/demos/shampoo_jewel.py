#!/usr/bin/env python
"""Shampoo numerical jewel demo (planning/shampoo.md §4 step 1).

Three parts, all tiny (CPU is fine):

  (a) THE JEWEL: single-step Shampoo IS Muon —
      (GG^T)^{-1/4} G (G^T G)^{-1/4} = UV^T (the polar factor).
      Verified numerically via eigh-based inverse roots vs SVD,
      including the eps-regularized deviation (what init eps does).

  (b) NS INVERSE ROOT: the matmul-only iteration the codegen will
      emit — coupled-Newton inverse-sqrt applied twice for ^{-1/4},
      with trace scaling (the convergence-basin requirement). This
      pins the iteration count + eps the MLIR emit needs.

  (c) TOY A/B: accumulated Shampoo vs Muon vs Adam vs SGD on a
      deterministic quadratic + a tiny MLP regression. Shows the
      memory knob: Shampoo interpolates Adagrad <-> Muon.

Mirrors the Lean-side plan: same scaling, same iteration budget the
emitShampooUpdate codegen will use.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # toy sizes; leave the GPUs alone
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- (a) jewel
def inv_root_eigh(M, power):
    """M^{-1/power} for symmetric PSD M via eigendecomposition (oracle)."""
    lam, Q = np.linalg.eigh(M)
    lam = np.maximum(lam, 0.0)
    with np.errstate(divide="ignore"):
        d = np.where(lam > 0, lam ** (-1.0 / power), 0.0)
    return (Q * d) @ Q.T


def polar_factor(G):
    U, _, Vt = np.linalg.svd(G, full_matrices=False)
    return U @ Vt


def jewel_check(m=48, n=32, eps_list=(0.0, 1e-8, 1e-4, 1e-2)):
    print("== (a) the jewel: single-step Shampoo = Muon = UV^T ==")
    G = rng.standard_normal((m, n))
    muon = polar_factor(G)
    for eps in eps_list:
        L = G @ G.T + eps * np.eye(m)
        R = G.T @ G + eps * np.eye(n)
        shampoo = inv_root_eigh(L, 4) @ G @ inv_root_eigh(R, 4)
        rel = np.linalg.norm(shampoo - muon) / np.linalg.norm(muon)
        tag = "EXACT (the theorem)" if eps == 0.0 else "eps-regularized"
        print(f"   eps={eps:<8g} ||shampoo - UV^T||/||UV^T|| = {rel:.3e}   {tag}")
    print()


# ------------------------------------------------- (b) NS inverse 4th root
def ns_inv_sqrt(M, iters):
    """Coupled-Newton inverse sqrt: X -> M^{-1/2} for SPD M scaled so
    eigenvalues sit in (0, 1]. Matmul-only (what the codegen emits).
       Y_0 = M, Z_0 = I
       T   = (3I - Z Y)/2 ;  Y <- Y T ; Z <- T Z
       Y -> M^{1/2}, Z -> M^{-1/2}
    """
    n = M.shape[0]
    I = np.eye(n)
    Y, Z = M.copy(), I.copy()
    for _ in range(iters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Z


def inv_fourth_root_ns(L, iters=30):
    """L^{-1/4} by two NS-inv-sqrt passes, matmul-only:
    NS-inv-sqrt of (scaled) L gives A = L^{-1/2}; A is SPD with
    eigenvalues lam^{-1/2}, so NS-inv-sqrt of A gives A^{-1/2} = L^{1/4};
    the update wants L^{-1/4} = A @ L^{1/4} = A @ A^{-1/2}. Trace-scaling
    before each pass keeps the iteration in its convergence basin.
    """
    n = L.shape[0]
    s = np.trace(L) / n
    Ls = L / s
    # scale again so max-eig <= 1 for basin safety: divide by n (tr bound)
    c = float(n)
    A = ns_inv_sqrt(Ls / c, iters) / np.sqrt(c)       # (L/s)^{-1/2}
    # A is SPD, eigs lam^{-1/2} in [lam_max^{-1/2}, ...]; scale into basin
    ta = np.trace(A) / n
    ca = float(n)
    Ahalf_inv = ns_inv_sqrt(A / ta / ca, iters) / np.sqrt(ta * ca)  # A^{-1/2} = (L/s)^{1/4}
    inv_quarter_scaled = A @ Ahalf_inv                # (L/s)^{-1/2} (L/s)^{1/4} = (L/s)^{-1/4}
    return inv_quarter_scaled / (s ** 0.25)


def ns_check(m=48, n=32):
    print("== (b) matmul-only NS inverse 4th root (the codegen kernel) ==")
    G = rng.standard_normal((m, n))
    for eps, iters in [(1e-6, 20), (1e-6, 30), (1e-4, 20), (1e-4, 30)]:
        L = G @ G.T + eps * np.eye(m)
        oracle = inv_root_eigh(L, 4)
        ns = inv_fourth_root_ns(L, iters=iters)
        rel = np.linalg.norm(ns - oracle) / np.linalg.norm(oracle)
        # end-to-end: the actual preconditioned update vs oracle
        R = G.T @ G + eps * np.eye(n)
        upd_o = oracle @ G @ inv_root_eigh(R, 4)
        upd_n = ns @ G @ inv_fourth_root_ns(R, iters=iters)
        rel_upd = np.linalg.norm(upd_n - upd_o) / np.linalg.norm(upd_o)
        print(f"   eps={eps:g} iters={iters}: ||NS - eigh||/||eigh|| = {rel:.3e}   update rel-err = {rel_upd:.3e}")
    print()


# ----------------------------------------------------------- (c) toy A/B
def optimizers_ab():
    print("== (c) accumulated Shampoo vs Muon vs Adam vs SGD ==")

    # -- c1: deterministic ill-conditioned quadratic f(W) = 0.5||A W B - C||_F^2
    m, n = 24, 16
    A = inv_root_eigh(np.diag(rng.uniform(0.05, 5.0, m)), 1) @ np.linalg.qr(rng.standard_normal((m, m)))[0]
    B = np.linalg.qr(rng.standard_normal((n, n)))[0] @ np.diag(rng.uniform(0.1, 3.0, n))
    Wstar = rng.standard_normal((m, n))
    C = A @ Wstar @ B

    def loss_grad(W):
        Rr = A @ W @ B - C
        return 0.5 * float(np.sum(Rr * Rr)), A.T @ Rr @ B.T

    def run(update, lr, steps=300):
        W = np.zeros((m, n))
        state = {}
        hist = []
        for t in range(steps):
            l, G = loss_grad(W)
            hist.append(l)
            W = W - lr * update(G, state, t)
        return hist

    def sgd(G, s, t):
        return G

    def adam(G, s, t, b1=0.9, b2=0.999, e=1e-8):
        s.setdefault("m", np.zeros_like(G)); s.setdefault("v", np.zeros_like(G))
        s["m"] = b1 * s["m"] + (1 - b1) * G
        s["v"] = b2 * s["v"] + (1 - b2) * G * G
        mh = s["m"] / (1 - b1 ** (t + 1)); vh = s["v"] / (1 - b2 ** (t + 1))
        return mh / (np.sqrt(vh) + e)

    def muon(G, s, t):
        return polar_factor(G)

    def shampoo(G, s, t, eps=1e-8):
        s.setdefault("L", eps * np.eye(G.shape[0])); s.setdefault("R", eps * np.eye(G.shape[1]))
        s["L"] = s["L"] + G @ G.T
        s["R"] = s["R"] + G.T @ G
        return inv_root_eigh(s["L"], 4) @ G @ inv_root_eigh(s["R"], 4)

    print("   quadratic  (300 steps, per-opt tuned lr):")
    for name, upd, lr in [("sgd", sgd, 0.002), ("adam", adam, 0.05),
                          ("muon", muon, 0.05), ("shampoo", shampoo, 0.35)]:
        h = run(upd, lr)
        print(f"     {name:<8s} lr={lr:<5g} loss: step0={h[0]:9.3f}  step50={h[50]:11.3e}  step299={h[-1]:11.3e}")

    # -- c2: single-step-Shampoo == Muon on the SAME trajectory (the jewel, dynamically)
    print("   jewel dynamically: fresh-L/R Shampoo step == Muon step at every point of a trajectory:")
    W = rng.standard_normal((m, n))
    max_dev = 0.0
    for t in range(50):
        _, G = loss_grad(W)
        fresh = inv_root_eigh(G @ G.T, 4) @ G @ inv_root_eigh(G.T @ G, 4)
        max_dev = max(max_dev, np.linalg.norm(fresh - polar_factor(G)) / np.linalg.norm(polar_factor(G)))
        W = W - 0.05 * fresh
    print(f"     max rel deviation over 50 steps: {max_dev:.3e}")

    # -- c3: tiny MLP regression (jax grad), 2D weights get the matrix opts
    print("   tiny MLP (784->64->10 regression on synthetic data, 200 steps):")
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    X = jax.random.normal(k1, (256, 784)) * 0.1
    Wt1 = jax.random.normal(k2, (784, 64)) * 0.05
    Wt2 = jax.random.normal(k3, (64, 10)) * 0.05
    Y = jnp.tanh(X @ Wt1) @ Wt2

    def mlp_loss(params):
        h = jnp.tanh(X @ params["W1"])
        return jnp.mean((h @ params["W2"] - Y) ** 2)

    grad_fn = jax.jit(jax.value_and_grad(mlp_loss))

    def train(kind, lr, steps=200):
        params = {"W1": jax.random.normal(k4, (784, 64)) * 0.05,
                  "W2": jax.random.normal(k2, (64, 10)) * 0.05}
        st = {p: {} for p in params}
        losses = []
        for t in range(steps):
            l, g = grad_fn(params)
            losses.append(float(l))
            for p in params:
                G = np.asarray(g[p])
                if kind == "adam":
                    d = adam(G, st[p], t)
                elif kind == "muon":
                    d = polar_factor(G)
                elif kind == "shampoo":
                    d = shampoo(G, st[p], t)
                else:
                    d = G
                params[p] = params[p] - lr * d
        return losses

    for name, lr in [("sgd", 0.5), ("adam", 0.01), ("muon", 0.02), ("shampoo", 0.05)]:
        h = train(name, lr)
        print(f"     {name:<8s} lr={lr:<5g} loss: step0={h[0]:.5f}  step50={h[50]:.6f}  step199={h[-1]:.6f}")
    print()


if __name__ == "__main__":
    jewel_check()
    ns_check()
    optimizers_ab()
    print("done (shampoo jewel demo).")
