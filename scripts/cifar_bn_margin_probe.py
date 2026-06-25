#!/usr/bin/env python3
"""Measured-vs-proven probe for the CIFAR / BatchNorm float bridge.

The f32/f64-twin analogue of `margin_probe.py`, for the Chapter-5 CIFAR-BN path
(`CifarFloatBridge.lean` + `BnFloatBridge.lean`). It runs the *verified* CIFAR-BN
computation — per-example, per-channel BN (reduce over h·w, the
`bnForward`/`bnPerChannelFlat` the lemmas prove), same-pad 3×3 conv, ReLU,
2×2 maxpool, 3-dense head, softmax-CE — once in float32 (deployed precision) and
once in float64 (the exact-ℝ proxy), and at each stage measures the actual drift
`|f32 − f64|`, alongside the *proven* closed-form budgets from the Lean lemmas
evaluated at the measured magnitude profile.

Two sections:
  (1) FORWARD per-stage drift vs proven budgets — conv / BN mean,var,istd,out.
      Surfaces the BN-istd worst-case (`1/(2ε√ε)`) vs operating-point
      (`1/(2(σ²+ε)√(σ²+ε))`, = `bnIstd_close_at`) vs measured gap.
  (2) BACKWARD / SGD-step drift over a short coupled trajectory — per-conv
      weight-step drift `|fl-step − ℝ-step|` vs the `cifar_stage{1,2}_convW`
      budgets `(a·g)/150` (32×32) and `(a·g)/2000` (16×16).

Proven formulas transcribed from the Lean defs (u = 2⁻²⁴ for binary32).

Usage: scripts/cifar_bn_margin_probe.py             # synthetic normalized input
       scripts/cifar_bn_margin_probe.py <cifar-dir> # real CIFAR: <dir>/{x,y}.npy
"""
import os, sys
import numpy as np

U32 = 2.0 ** -24
EPS = 1e-5
LR  = np.float64(0.1)
rng = np.random.default_rng(0)

CH = [3, 32, 32, 64, 64]
POOL_AFTER = {1, 3}
D_FLAT, D_H, NCLS = 64 * 8 * 8, 512, 10

# ── proven closed-form budgets (from the Lean lemmas) ─────────────────────
def layer_budget(u, m, w, beta, A, E):
    return ((1+u)**(m+2) - 1) * (m*w*(A+E) + beta) + m*w*E
def mul_err(u, A, C, ea, ec):
    return u*((A+ea)*(C+ec)) + (A*ec + ea*C + ea*ec)
def bn_mean_budget(u, n, A):
    return u*((1+u)**(n+1) * A) + ((1+u)**(n+1) - 1)*A
def bn_var_budget(u, D, emean, n):
    g  = (1+u)**(n+1) - 1; es1 = u*(D+emean) + emean; esq = mul_err(u, D, D, es1, es1)
    return u*(D*D + g*(D*D+esq) + esq) + (g*(D*D+esq) + esq)
def bn_istd_budget(ers, evar, floor):              # floor = ε (worst) or σ²+ε (a-post)
    return ers/np.sqrt(floor) + evar/(2*floor*np.sqrt(floor))
def bn_norm_budget(u, D, S, G, Bb, emean, eistd):
    es2 = mul_err(u, D, S, u*(D+emean)+emean, eistd); es3 = mul_err(u, G, D*S, 0.0, es2)
    return u*(G*(D*S) + es3 + Bb) + es3

# ── verified ops, dtype-parametric, capturing backward intermediates ──────
def conv_same3(x, W, b, dt):
    x, W, b = x.astype(dt), W.astype(dt), b.astype(dt)
    N, C, H, Wd = x.shape; O = W.shape[0]
    xp = np.zeros((N, C, H+2, Wd+2), dt); xp[:, :, 1:H+1, 1:Wd+1] = x
    out = np.zeros((N, O, H, Wd), dt)
    for di in range(3):
        for dj in range(3):
            out += np.einsum('nchw,oc->nohw', xp[:, :, di:di+H, dj:dj+Wd],
                             W[:, :, di, dj], optimize=True)
    return out + b[None, :, None, None]

def conv_weight_grad(x, dy, dt):
    x, dy = x.astype(dt), dy.astype(dt)
    N, C, H, Wd = x.shape; O = dy.shape[1]
    xp = np.zeros((N, C, H+2, Wd+2), dt); xp[:, :, 1:H+1, 1:Wd+1] = x
    dW = np.zeros((O, C, 3, 3), dt)
    for di in range(3):
        for dj in range(3):
            dW[:, :, di, dj] = np.einsum('nohw,nchw->oc', dy, xp[:, :, di:di+H, dj:dj+Wd], optimize=True)
    return dW

def conv_input_grad(dy, W, dt):                    # → dx at same spatial (stride-1 same-pad)
    dy, W = dy.astype(dt), W.astype(dt)
    N, O, H, Wd = dy.shape; C = W.shape[1]
    dxp = np.zeros((N, C, H+2, Wd+2), dt)
    for di in range(3):
        for dj in range(3):
            dxp[:, :, di:di+H, dj:dj+Wd] += np.einsum('nohw,oc->nchw', dy, W[:, :, di, dj], optimize=True)
    return dxp[:, :, 1:H+1, 1:Wd+1]

def bn_pc(x, gamma, beta, dt):                     # per (n,c), reduce over h·w
    x, gamma, beta = x.astype(dt), gamma.astype(dt), beta.astype(dt)
    mu  = x.mean(axis=(2, 3), keepdims=True)
    var = ((x - mu)**2).mean(axis=(2, 3), keepdims=True)
    istd = dt(1.0) / np.sqrt(var + dt(EPS))
    xhat = (x - mu) * istd
    y = gamma[None, :, None, None] * xhat + beta[None, :, None, None]
    return y, mu, var, istd, xhat

def bn_pc_back(dy, x, mu, istd, xhat, gamma, dt):  # → (dx, dgamma, dbeta)
    dy = dy.astype(dt); m = x.shape[2] * x.shape[3]
    dgamma = (dy * xhat).sum(axis=(0, 2, 3))
    dbeta  = dy.sum(axis=(0, 2, 3))
    g = gamma[None, :, None, None]
    s1 = dy.sum(axis=(2, 3), keepdims=True)
    s2 = (dy * xhat).sum(axis=(2, 3), keepdims=True)
    dx = (g * istd / dt(m)) * (dt(m) * dy - s1 - xhat * s2)
    return dx, dgamma, dbeta

def maxpool(a, dt):                                # 2×2, return pooled + argmax mask
    N, C, H, Wd = a.shape
    blk = a.reshape(N, C, H//2, 2, Wd//2, 2)
    pooled = blk.max(axis=(3, 5))
    mask = (blk == pooled[:, :, :, None, :, None]).astype(dt)
    # break ties (keep first) so backward routes to one cell
    mask = mask.reshape(N, C, H//2, 2, Wd//2, 2)
    return pooled, mask

def maxpool_back(dp, mask, dt):
    N, C, h, Wd2 = dp.shape
    full = (dp[:, :, :, None, :, None] * mask).reshape(N, C, h*2, Wd2*2)
    return full

def relu(z, dt): return np.maximum(z, dt(0))

# ── weights (He init; magnitudes are the realistic profile the budgets use) ─
def he(fan_in, shape): return (rng.standard_normal(shape) * np.sqrt(2.0/fan_in)).astype(np.float32)
convW = [he(CH[i]*9, (CH[i+1], CH[i], 3, 3)) for i in range(4)]
convB = [np.zeros(CH[i+1], np.float32) for i in range(4)]
gam   = [(1 + 0.1*rng.standard_normal(CH[i+1])).astype(np.float32) for i in range(4)]
bet   = [(0.1*rng.standard_normal(CH[i+1])).astype(np.float32) for i in range(4)]
dW5, db5 = he(D_FLAT, (D_FLAT, D_H)), np.zeros(D_H, np.float32)
dW6, db6 = he(D_H, (D_H, D_H)),       np.zeros(D_H, np.float32)
dW7, db7 = he(D_H, (D_H, NCLS)),      np.zeros(NCLS, np.float32)

# ── input: real CIFAR if given, else synthetic normalized ─────────────────
real = len(sys.argv) > 1 and os.path.exists(f"{sys.argv[1]}/x.npy")
if real:
    X = np.load(f"{sys.argv[1]}/x.npy")[:8].astype(np.float32)
    y = np.load(f"{sys.argv[1]}/y.npy")[:8].astype(np.int64)
else:
    X = rng.standard_normal((8, 3, 32, 32)).astype(np.float32)
    y = rng.integers(0, NCLS, 8).astype(np.int64)

def forward(dt, P):
    cW, cB, gm, bt, W5, B5, W6, B6, W7, B7 = P
    a = X.astype(dt); rec = {'a-1': a}
    for i in range(4):
        c = conv_same3(a, cW[i], cB[i], dt); rec[f'conv{i}'] = c
        bn, mu, var, istd, xhat = bn_pc(c, gm[i], bt[i], dt)
        rec[f'bnmu{i}'], rec[f'bnvar{i}'], rec[f'bnistd{i}'], rec[f'bn{i}'] = mu, var, istd, bn
        rec[f'xhat{i}'] = xhat
        a = relu(bn, dt); rec[f'relu{i}'] = a
        if i in POOL_AFTER:
            a, mask = maxpool(a, dt); rec[f'pool{i}'] = a; rec[f'mask{i}'] = mask
        rec[f'a{i}'] = a
    flat = a.reshape(a.shape[0], -1); rec['flat'] = flat
    h5 = relu(flat @ W5.astype(dt) + B5.astype(dt), dt); rec['h5'] = h5
    h6 = relu(h5 @ W6.astype(dt) + B6.astype(dt), dt); rec['h6'] = h6
    z = h6 @ W7.astype(dt) + B7.astype(dt); rec['z'] = z
    return rec

def backward(dt, P, rec):
    cW, cB, gm, bt, W5, B5, W6, B6, W7, B7 = P
    z = rec['z']; zs = z - z.max(1, keepdims=True); e = np.exp(zs); s = e/e.sum(1, keepdims=True)
    g = s.copy().astype(dt); g[np.arange(len(y)), y] -= dt(1); g /= dt(len(y))
    c6 = (g @ W7.astype(dt).T) * (rec['h6'] > 0)
    c5 = (c6 @ W6.astype(dt).T) * (rec['h5'] > 0)
    cflat = c5 @ W5.astype(dt).T
    a = cflat.reshape(rec['a3'].shape)                       # cot at last conv-stage output
    dWc = [None]*4
    for i in reversed(range(4)):
        if i in POOL_AFTER:
            a = maxpool_back(a, rec[f'mask{i}'], dt)         # → cot at relu output
        a = a * (rec[f'bn{i}'] > 0)                          # relu back → cot at BN output
        dx, _, _ = bn_pc_back(a, rec[f'conv{i}'], rec[f'bnmu{i}'], rec[f'bnistd{i}'],
                              rec[f'xhat{i}'], gm[i], dt)     # → cot at conv output
        prev = rec[f'a{i-1}'] if i > 0 else rec['a-1']
        dWc[i] = (conv_weight_grad(prev, dx, dt), float(np.abs(prev).max()), float(np.abs(dx).max()))
        if i > 0:
            a = conv_input_grad(dx, cW[i], dt)               # → cot at prev activation
    return dWc

P32 = [convW, convB, gam, bet, dW5, db5, dW6, db6, dW7, db7]
r32, r64 = forward(np.float32, P32), forward(np.float64, P32)
def drift(k): return np.abs(r32[k].astype(np.float64) - r64[k]).max()

print(f"input: {'real CIFAR' if real else 'synthetic normalized'}  batch {X.shape[0]}   "
      f"u=2^-24={U32:.2e}  ε={EPS}\n")
print("(1) FORWARD per-stage drift")
print(f"{'stage':<14}{'measured |f32-f64|':>20}{'proven budget':>16}{'ratio':>10}")
for i in range(4):
    H = 32 if i < 2 else 16; n = H*H
    A = float(np.abs(r64[f'conv{i}']).max())
    e_in = drift(f'conv{i-1}') if i > 0 else 0.0
    cb = layer_budget(U32, CH[i]*9, float(np.abs(convW[i]).max()), float(np.abs(convB[i]).max()), A, e_in)
    print(f"conv{i} out{'':<5}{drift(f'conv{i}'):>20.3e}{cb:>16.3e}{drift(f'conv{i}')/cb:>10.1e}")
    emean = bn_mean_budget(U32, n, A); evar = bn_var_budget(U32, 2*A, emean, n)
    print(f"  bn{i} mean{'':<4}{drift(f'bnmu{i}'):>20.3e}{emean:>16.3e}{drift(f'bnmu{i}')/emean:>10.1e}")
    print(f"  bn{i} var{'':<5}{drift(f'bnvar{i}'):>20.3e}{evar:>16.3e}{drift(f'bnvar{i}')/evar:>10.1e}")
    ers = float((np.abs(r32[f'bnistd{i}'].astype(np.float64) - r64[f'bnistd{i}']) / r64[f'bnistd{i}']).max())
    vmin = float(r64[f'bnvar{i}'].min())
    wc = bn_istd_budget(ers, evar, EPS); ap = bn_istd_budget(ers, evar, vmin + EPS)
    print(f"  bn{i} istd{'':<4}{drift(f'bnistd{i}'):>20.3e}{wc:>16.3e}{drift(f'bnistd{i}')/wc:>10.1e}"
          f"   [a-post(σ²+ε) {ap:.3e}  σ²min {vmin:.3e}  ers {ers:.2e}]")
    D, S, G, Bb = 2*A, 1.0/np.sqrt(EPS), float(np.abs(gam[i]).max()), float(np.abs(bet[i]).max())
    nb = bn_norm_budget(U32, D, S, G, Bb, emean, wc)
    print(f"  bn{i} out{'':<5}{drift(f'bn{i}'):>20.3e}{nb:>16.3e}{drift(f'bn{i}')/nb:>10.1e}")
print(f"logit drift max|z32-z64| : {drift('z'):.3e}")

print("\n(2) BACKWARD / conv-weight SGD-step drift over a short coupled trajectory")
print(f"{'step.conv':<12}{'measured |fl-ℝ step|':>22}{'budget (a·g)/rate':>20}{'ratio':>10}")
for stp in range(4):
    rc32 = forward(np.float32, P32)
    g32 = backward(np.float32, P32, rc32)
    g64 = backward(np.float64, P32, forward(np.float64, P32))
    for i in range(4):
        rate = 150 if i < 2 else 2000
        dW32, a, gmag = g32[i]; dW64, _, _ = g64[i]
        step32 = (convW[i] - np.float32(LR) * dW32).astype(np.float64)
        step64 = convW[i].astype(np.float64) - LR * dW64
        meas = float(np.abs(step32 - step64).max())
        bud = (a * gmag) / rate + 1e-7
        print(f"  s{stp}.conv{i}{'':<2}{meas:>22.3e}{bud:>20.3e}{meas/bud:>10.1e}")
    for i in range(4):                                       # advance f32 trajectory
        convW[i] -= np.float32(LR) * g32[i][0]

print("\nread: 'ratio' = measured / proven. bn*.istd worst-case uses 1/(2ε√ε)")
print("(vacuous); a-post(σ²+ε) = bnIstd_close_at, the operating-point budget.")
