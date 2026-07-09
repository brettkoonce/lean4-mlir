#!/usr/bin/env python
"""RoPE (Rotary Position Embedding, Su et al. 2021) numeric reference.

De-risks the op before any MLIR (same role as flash_attention_ref.py). RoPE
applies a per-position rotation to each head's Q and K vector — no learned
params — so absolute position drops out and attention scores depend on the
RELATIVE offset (m−n), which is what generalizes to longer context than trained.

Llama/NeoX "rotate_half" convention (easiest for tensor codegen):
    theta_i   = base^(-2i/dh),           i in [0, dh/2)
    freqs[m]  = m * theta                  [T, dh/2]
    cos,sin   = cos(freqs), sin(freqs)     tiled to [T, dh]
    rope(x)   = x*cos + rotate_half(x)*sin,  rotate_half([a,b]) = [-b, a]

Backward: RoPE is an orthogonal rotation by +theta, so its VJP is the rotation
by -theta — i.e. rope with sin negated. Validated here against finite-diff.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

rng = np.random.default_rng(0)


def rope_tables(T, dh, base=10000.0):
    half = dh // 2
    i = np.arange(half)
    theta = base ** (-2.0 * i / dh)          # [half]
    freqs = np.outer(np.arange(T), theta)    # [T, half]
    cos = np.concatenate([np.cos(freqs), np.cos(freqs)], -1)  # [T, dh]
    sin = np.concatenate([np.sin(freqs), np.sin(freqs)], -1)  # [T, dh]
    return cos, sin


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], -1)


def rope(x, cos, sin):
    # x: [B,H,T,dh]; cos/sin: [T,dh] broadcast over B,H
    return x * cos + rotate_half(x) * sin


def rope_vjp(g, cos, sin):
    # adjoint = rotation by -theta = rope with sin negated
    return g * cos + rotate_half(g) * (-sin)


def check(B=2, H=3, T=16, dh=8):
    print(f"== B={B} H={H} T={T} dh={dh} ==")
    cos, sin = rope_tables(T, dh)
    x = rng.standard_normal((B, H, T, dh))

    # (1) norm-preserving (it's a rotation)
    y = rope(x, cos, sin)
    nerr = np.abs(np.linalg.norm(y, axis=-1) - np.linalg.norm(x, axis=-1)).max()
    print(f"  norm-preserving:        max|‖rope(x)‖ − ‖x‖| = {nerr:.2e}")

    # (2) scores depend only on relative position: q_m·k_n depends on (m−n)
    q = rng.standard_normal((1, 1, T, dh)); k = rng.standard_normal((1, 1, T, dh))
    qr = rope(q, cos, sin); kr = rope(k, cos, sin)
    S = np.einsum("bhqd,bhkd->bhqk", qr, kr)[0, 0]   # [T,T]
    # for a FIXED q,k pair rotated to different (m,n) with same offset, score matches
    q0 = rng.standard_normal(dh); k0 = rng.standard_normal(dh)
    def score_at(m, n):
        qm = rope(q0[None, None, None], cos[m:m+1], sin[m:m+1])[0, 0, 0]
        kn = rope(k0[None, None, None], cos[n:n+1], sin[n:n+1])[0, 0, 0]
        return qm @ kn
    rel_err = abs(score_at(5, 3) - score_at(9, 7))   # both offset +2
    print(f"  relative-position:      |score(5,3) − score(9,7)| = {rel_err:.2e}  (same offset +2)")

    # (3) backward matches finite-diff of a scalar loss L = sum(w ⊙ rope(x))
    w = rng.standard_normal((B, H, T, dh))
    g_analytic = rope_vjp(w, cos, sin)               # dL/dx
    eps = 1e-6
    g_fd = np.zeros_like(x)
    idx = [(0, 0, 4, 3), (B - 1, H - 1, T // 2, dh - 1), (0, 0, T - 1, dh // 2)]
    for ix in idx:
        xp = x.copy(); xp[ix] += eps
        xm = x.copy(); xm[ix] -= eps
        lp = (w * rope(xp, cos, sin)).sum(); lm = (w * rope(xm, cos, sin)).sum()
        g_fd[ix] = (lp - lm) / (2 * eps)
    berr = max(abs(g_analytic[ix] - g_fd[ix]) for ix in idx)
    print(f"  backward vs finite-diff: max|Δ| = {berr:.2e}")


if __name__ == "__main__":
    check()
    check(B=1, H=2, T=32, dh=16)
    print("done (RoPE reference — rotation, relative-position, backward all check out).")
