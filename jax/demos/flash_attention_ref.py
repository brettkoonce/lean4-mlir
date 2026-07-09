#!/usr/bin/env python
"""FlashAttention numeric reference (planning/flash_attention.md).

De-risks the algorithm BEFORE any MLIR: the exact block recurrence the
StableHLO `while`-loop codegen will emit, validated against dense
attention (forward) and against dense autodiff (backward). Same role the
shampoo_jewel.py reference played for the NS inverse-root kernel.

Two things must hold to a tight tolerance:
  (a) tiled online-softmax forward  == dense softmax(QKᵀ/√d + mask) @ V
  (b) flash backward (dQ,dK,dV via the recompute + D-trick)
                                      == dense autodiff of the same

If both pass, the codegen has a correct spec to emit; the online-softmax
recurrence and the softmax-Jacobian backward are the only fiddly parts and
they're pinned here.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

rng = np.random.default_rng(0)


# ------------------------------------------------------------ dense (oracle)
def dense_attention(Q, K, V, causal):
    """Q,K,V: [B,H,T,d] -> O: [B,H,T,d]. The thing flash must match."""
    B, H, T, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    S = np.einsum("bhqd,bhkd->bhqk", Q, K) * scale          # [B,H,T,T]
    if causal:
        mask = np.tril(np.ones((T, T), bool))
        S = np.where(mask, S, -np.inf)
    S = S - S.max(-1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", P, V)               # [B,H,T,d]


# ------------------------------------------------ flash forward (tiled online softmax)
def flash_forward(Q, K, V, causal, Bq=32, Bk=32):
    """Online-softmax over key blocks — the exact recurrence the `while`
    loop emits. Never materializes the full [.,.,T,T] scores. Also returns
    the per-query logsumexp L = m + log(l) (needed by the backward)."""
    B, H, T, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    O = np.zeros_like(Q)
    Lse = np.zeros((B, H, T))                               # logsumexp per query row
    for qi in range(0, T, Bq):
        qs = slice(qi, min(qi + Bq, T))
        Qi = Q[:, :, qs, :]                                 # [B,H,bq,d]
        Oi = np.zeros_like(Qi)
        mi = np.full((B, H, Qi.shape[2]), -np.inf)
        li = np.zeros((B, H, Qi.shape[2]))
        kmax = min(qi + Bq, T) if causal else T             # causal: skip blocks fully in the future
        for kj in range(0, kmax, Bk):
            ks = slice(kj, min(kj + Bk, T))
            Kj, Vj = K[:, :, ks, :], V[:, :, ks, :]         # [B,H,bk,d]
            Sij = np.einsum("bhqd,bhkd->bhqk", Qi, Kj) * scale   # [B,H,bq,bk]
            if causal:
                qidx = np.arange(qi, qi + Qi.shape[2])[:, None]
                kidx = np.arange(kj, kj + Kj.shape[2])[None, :]
                Sij = np.where(qidx >= kidx, Sij, -np.inf)
            mij = Sij.max(-1)                               # [B,H,bq]
            m_new = np.maximum(mi, mij)
            # guard fully-masked (-inf) rows so exp(-inf - -inf)=exp(nan)→0
            alpha = np.exp(np.where(np.isinf(m_new), 0.0, mi - m_new))     # rescale prior
            Pij = np.exp(Sij - m_new[..., None])
            Pij = np.where(np.isinf(Sij), 0.0, Pij)
            li = alpha * li + Pij.sum(-1)
            Oi = alpha[..., None] * Oi + np.einsum("bhqk,bhkd->bhqd", Pij, Vj)
            mi = m_new
        li_safe = np.where(li == 0.0, 1.0, li)              # fully-masked rows → 0 output
        O[:, :, qs, :] = Oi / li_safe[..., None]
        Lse[:, :, qs] = mi + np.log(li_safe)
    return O, Lse


# ------------------------------------------------ flash backward (recompute + D-trick)
def flash_backward(Q, K, V, dO, O, Lse, causal, Bq=32, Bk=32):
    """dQ,dK,dV without ever storing [.,.,T,T]. Recompute Pij = exp(Sij-Lse)
    per block; D = rowsum(dO ⊙ O) is the softmax-Jacobian correction term.
    Matches the dense softmax VJP block-by-block."""
    B, H, T, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)
    D = (dO * O).sum(-1)                                    # [B,H,T] rowsum(dO⊙O)
    for qi in range(0, T, Bq):
        qs = slice(qi, min(qi + Bq, T))
        Qi, dOi = Q[:, :, qs, :], dO[:, :, qs, :]
        Lsei, Di = Lse[:, :, qs], D[:, :, qs]
        dQi = np.zeros_like(Qi)
        kmax = min(qi + Bq, T) if causal else T
        for kj in range(0, kmax, Bk):
            ks = slice(kj, min(kj + Bk, T))
            Kj, Vj = K[:, :, ks, :], V[:, :, ks, :]
            Sij = np.einsum("bhqd,bhkd->bhqk", Qi, Kj) * scale
            if causal:
                qidx = np.arange(qi, qi + Qi.shape[2])[:, None]
                kidx = np.arange(kj, kj + Kj.shape[2])[None, :]
                Sij = np.where(qidx >= kidx, Sij, -np.inf)
            Pij = np.exp(Sij - Lsei[..., None])             # [B,H,bq,bk]
            Pij = np.where(np.isinf(Sij), 0.0, Pij)
            dV[:, :, ks, :] += np.einsum("bhqk,bhqd->bhkd", Pij, dOi)
            dPij = np.einsum("bhqd,bhkd->bhqk", dOi, Vj)    # [B,H,bq,bk]
            dSij = Pij * (dPij - Di[..., None]) * scale     # softmax Jacobian
            dQi += np.einsum("bhqk,bhkd->bhqd", dSij, Kj)
            dK[:, :, ks, :] += np.einsum("bhqk,bhqd->bhkd", dSij, Qi)
        dQ[:, :, qs, :] = dQi
    return dQ, dK, dV


def dense_backward_autodiff(Q, K, V, dO, causal):
    """Finite-diff-free oracle: analytic dense softmax-attention VJP."""
    B, H, T, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    S = np.einsum("bhqd,bhkd->bhqk", Q, K) * scale
    if causal:
        mask = np.tril(np.ones((T, T), bool))
        S = np.where(mask, S, -np.inf)
    S = S - S.max(-1, keepdims=True)
    E = np.exp(S)
    P = E / E.sum(-1, keepdims=True)
    dV = np.einsum("bhqk,bhqd->bhkd", P, dO)
    dP = np.einsum("bhqd,bhkd->bhqk", dO, V)
    dS = P * (dP - (dP * P).sum(-1, keepdims=True))
    dQ = np.einsum("bhqk,bhkd->bhqd", dS, K) * scale
    dK = np.einsum("bhqk,bhqd->bhkd", dS, Q) * scale
    return dQ, dK, dV


def check(B=2, H=4, T=96, d=32, Bq=32, Bk=32):
    print(f"== B={B} H={H} T={T} d={d}, blocks Bq={Bq} Bk={Bk} ==")
    for causal in (False, True):
        Q = rng.standard_normal((B, H, T, d))
        K = rng.standard_normal((B, H, T, d))
        V = rng.standard_normal((B, H, T, d))
        dO = rng.standard_normal((B, H, T, d))
        O_dense = dense_attention(Q, K, V, causal)
        O_flash, Lse = flash_forward(Q, K, V, causal, Bq, Bk)
        f_err = np.abs(O_flash - O_dense).max()
        dQd, dKd, dVd = dense_backward_autodiff(Q, K, V, dO, causal)
        dQf, dKf, dVf = flash_backward(Q, K, V, dO, O_flash, Lse, causal, Bq, Bk)
        b_err = max(np.abs(dQf - dQd).max(), np.abs(dKf - dKd).max(), np.abs(dVf - dVd).max())
        tag = "causal" if causal else "full  "
        print(f"  {tag}: forward max|Δ| = {f_err:.2e}   backward max|Δ| = {b_err:.2e}")


if __name__ == "__main__":
    check(T=96, Bq=32, Bk=32)          # T not a multiple pattern of blocks
    check(T=128, Bq=64, Bk=32)         # unequal blocks
    check(T=100, Bq=32, Bk=48)         # ragged tail
    print("done (flash attention reference — fwd matches dense, bwd matches dense VJP).")
