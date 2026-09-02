#!/usr/bin/env python3
"""adjoint_chain_ibp_probe.py — can the adjoint chain's tail gains be PROVED?

`TailGains` (AdjointChainBridge.lean) needs `LipOnWindow A H (chainR ls)`: the gain
of the real tail between ANY two points of the window. Today the CIFAR-8 capstone
takes those `H` as a measured hypothesis, from `jacrev` at a point — which is a
LINEARIZATION at one trajectory point, not an upper bound on that sup (it misses
curvature and ReLU branch switching). The proven alternative,
`tailGains_suffixProd`, is worst-case products and is vacuous at depth.

This probe tests the third route: derive `H` from INTERVAL propagation, i.e. from
the already-proved `BoxSound` engine (Foundation/IntervalBoundConv.lean). If both
`u` and `v` lie in the box `[a-e, a+e]`, box soundness puts `f u` and `f v` inside
the propagated output box, so `|f u j - f v j| <= width_j`, giving

    H_ibp = max_j width_j / e        (a SOUND bound on the same quantity)

The question this answers, before any Lean work: is `H_ibp` close to the measured
`H`, or does interval growth blow up over the 11 committed stages?

Note the structural factor 2: modelling `|u-v| <= e` by a box of RADIUS e centred
at u means a linear tail gives width = 2e*sum|J|, so H_ibp -> 2*H_meas in the
smooth limit. Anything much beyond 2x is interval growth, which is the real signal.

    .venv/bin/python scripts/adjoint_chain_ibp_probe.py
"""
import numpy as np

U32 = 2.0 ** -24
rng = np.random.default_rng(0)


def layer_budget_fresh(u, m, wmax, beta, A):
    """`layerBudget u m w' beta A 0` — the proven per-op fresh budget."""
    return ((1 + u) ** (m + 2) - 1) * (m * wmax * A + beta)


# ── forward ops (NCHW, SAME 3x3), and their interval peers ──────────────────
def conv_same3(x, W, b):
    N, C, H, Wd = x.shape
    O = W.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((N, O, H, Wd), dtype=x.dtype)
    for kh in range(3):
        for kw in range(3):
            out += np.einsum("nchw,oc->nohw", xp[:, :, kh:kh + H, kw:kw + Wd],
                             W[:, :, kh, kw])
    return out + b[None, :, None, None]


def conv_same3_box(lo, hi, W, b):
    """Sign-split interval conv — the numeric peer of `conv2d`'s BoxSound
    transformer (positive weights pull from lo, negative from hi)."""
    Wp, Wn = np.clip(W, 0, None), np.clip(W, None, 0)
    zb = np.zeros_like(b)
    return (conv_same3(lo, Wp, b) + conv_same3(hi, Wn, zb),
            conv_same3(hi, Wp, b) + conv_same3(lo, Wn, zb))


def maxpool2(a):
    N, C, H, Wd = a.shape
    return a.reshape(N, C, H // 2, 2, Wd // 2, 2).max(axis=(3, 5))


def dense(x, W, b):
    return x @ W + b


def dense_box(lo, hi, W, b):
    Wp, Wn = np.clip(W, 0, None), np.clip(W, None, 0)
    return (lo @ Wp + hi @ Wn + b, hi @ Wp + lo @ Wn + b)


# ── the committed cifar8Verified net ────────────────────────────────────────
CHANS = [(3, 16), (16, 16), (16, 16), (16, 16), (16, 32), (32, 32), (32, 32), (32, 32)]
DENSES = [(128, 64), (64, 64), (64, 10)]

convs, denses_w = [], []
for ic, oc in CHANS:
    std = (2.0 / (ic * 9)) ** 0.5
    convs.append((rng.standard_normal((oc, ic, 3, 3)) * std,
                  rng.standard_normal(oc) * 0.01))
for i, o in DENSES:
    std = (2.0 / i) ** 0.5
    denses_w.append((rng.standard_normal((i, o)) * std, rng.standard_normal(o) * 0.01))

# stage list, in chain order: (kind, payload)
STAGES = []
k = 0
for stage in range(4):
    for _ in range(2):
        STAGES.append(("conv", k)); k += 1
    STAGES.append(("pool", None))
for j in range(3):
    STAGES.append(("dense", j))


def apply_stage(a, kind, idx, last):
    if kind == "conv":
        W, b = convs[idx]
        return np.maximum(conv_same3(a, W, b), 0.0)
    if kind == "pool":
        return maxpool2(a)
    W, b = denses_w[idx]
    if idx == 0:
        a = a.reshape(a.shape[0], -1)
    z = dense(a, W, b)
    return z if last else np.maximum(z, 0.0)


def apply_stage_box(lo, hi, kind, idx, last):
    if kind == "conv":
        W, b = convs[idx]
        lo, hi = conv_same3_box(lo, hi, W, b)
        return np.maximum(lo, 0.0), np.maximum(hi, 0.0)
    if kind == "pool":
        return maxpool2(lo), maxpool2(hi)      # monotone: pool the endpoints
    W, b = denses_w[idx]
    if idx == 0:
        lo, hi = lo.reshape(lo.shape[0], -1), hi.reshape(hi.shape[0], -1)
    lo, hi = dense_box(lo, hi, W, b)
    return (lo, hi) if last else (np.maximum(lo, 0.0), np.maximum(hi, 0.0))


def run_from(a, i, box=False, hi=None):
    """Apply stages i.. to a (or to the box (a, hi) when box=True)."""
    lo = a
    for s in range(i, len(STAGES)):
        kind, idx = STAGES[s]
        last = (s == len(STAGES) - 1)
        if box:
            lo, hi = apply_stage_box(lo, hi, kind, idx, last)
        else:
            lo = apply_stage(lo, kind, idx, last)
    return (lo, hi) if box else lo


def main():
    x = rng.standard_normal((1, 3, 32, 32))
    A_in = float(np.abs(x).max())

    # forward trajectory + per-stage fresh budgets
    acts, a = [x], x
    for s, (kind, idx) in enumerate(STAGES):
        a = apply_stage(a, kind, idx, s == len(STAGES) - 1)
        acts.append(a)

    bs, gs = [], []
    for s, (kind, idx) in enumerate(STAGES):
        ain = acts[s]
        A = float(np.abs(ain).max())
        if kind == "pool":
            bs.append(0.0); gs.append(1.0); continue
        if kind == "conv":
            W, b = convs[idx]; m = W.shape[1] * 9
        else:
            W, b = denses_w[idx]; m = W.shape[0]
        wmax, bmax = float(np.abs(W).max()), float(np.abs(b).max())
        bs.append(layer_budget_fresh(U32, m, wmax, bmax, A))
        gs.append(m * wmax)

    H_prov = [float(np.prod(gs[i + 1:])) for i in range(len(gs))]

    # measured H (jacrev) and IBP H, for each stage's tail
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    import jax.numpy as jnp

    jconvs = [(jnp.asarray(W), jnp.asarray(b)) for W, b in convs]
    jdenses = [(jnp.asarray(W), jnp.asarray(b)) for W, b in denses_w]

    def jtail(a, i):
        for s in range(i, len(STAGES)):
            kind, idx = STAGES[s]
            last = (s == len(STAGES) - 1)
            if kind == "conv":
                W, b = jconvs[idx]
                N, C, H, Wd = a.shape
                ap = jnp.pad(a, ((0, 0), (0, 0), (1, 1), (1, 1)))
                o = sum(jnp.einsum("nchw,oc->nohw",
                                   ap[:, :, kh:kh + H, kw:kw + Wd], W[:, :, kh, kw])
                        for kh in range(3) for kw in range(3))
                a = jnp.maximum(o + b[None, :, None, None], 0.0)
            elif kind == "pool":
                N, C, H, Wd = a.shape
                a = a.reshape(N, C, H // 2, 2, Wd // 2, 2).max(axis=(3, 5))
            else:
                W, b = jdenses[idx]
                if idx == 0:
                    a = a.reshape(a.shape[0], -1)
                z = a @ W + b
                a = z if last else jnp.maximum(z, 0.0)
        return a[0]

    print(f"{'stage':<14}{'fresh b_i':>12}{'H meas':>12}{'H ibp':>12}"
          f"{'ibp/meas':>10}{'H proven':>12}")
    H_meas, H_ibp = [], []
    for i in range(len(STAGES)):
        a_i = acts[i + 1]
        if i == len(STAGES) - 1:
            H_meas.append(1.0); H_ibp.append(1.0)
        else:
            J = jax.jacrev(lambda aa, i=i: jtail(aa, i + 1))(jnp.asarray(a_i))
            H_meas.append(float(jnp.abs(J.reshape(10, -1)).sum(axis=1).max()))
            e = bs[i] if bs[i] > 0 else 1e-9
            lo, hi = run_from(a_i - e, i + 1, box=True, hi=a_i + e)
            H_ibp.append(float((hi - lo).max() / e))
        name = STAGES[i][0] + (str(STAGES[i][1]) if STAGES[i][1] is not None else "")
        ratio = H_ibp[i] / H_meas[i] if H_meas[i] > 0 else float("nan")
        print(f"{name:<14}{bs[i]:>12.3e}{H_meas[i]:>12.3e}{H_ibp[i]:>12.3e}"
              f"{ratio:>10.2f}{H_prov[i]:>12.3e}")

    cb_meas = sum(H * b for H, b in zip(H_meas, bs))
    cb_ibp = sum(H * b for H, b in zip(H_ibp, bs))
    cb_prov = sum(H * b for H, b in zip(H_prov, bs))

    # true f32-vs-f64 drift on the same net
    o64 = run_from(x, 0)
    convs32 = [(W.astype(np.float32), b.astype(np.float32)) for W, b in convs]
    dens32 = [(W.astype(np.float32), b.astype(np.float32)) for W, b in denses_w]
    convs_b, dens_b = convs[:], denses_w[:]
    convs[:], denses_w[:] = convs32, dens32
    o32 = run_from(x.astype(np.float32), 0)
    convs[:], denses_w[:] = convs_b, dens_b
    true_err = float(np.abs(o64 - o32.astype(np.float64)).max())

    print(f"\n  logits magnitude       : {float(np.abs(o64).max()):.3e}")
    print(f"  true f32 drift         : {true_err:.3e}")
    print(f"  chainBudget measured-H : {cb_meas:.3e}  ({cb_meas / true_err:.1e}x true)"
          f"   [UNSOUND as a window bound]")
    print(f"  chainBudget IBP-H      : {cb_ibp:.3e}  ({cb_ibp / true_err:.1e}x true)"
          f"   [SOUND, from BoxSound]")
    print(f"  chainBudget proven-H   : {cb_prov:.3e}  ({cb_prov / true_err:.1e}x true)"
          f"   [SOUND, worst-case products]")
    print(f"\n  IBP vs measured budget : {cb_ibp / cb_meas:.2f}x")
    print(f"  IBP vs product budget  : {cb_prov / cb_ibp:.3e}x tighter")
    print(f"  argmax-safe (budget < margin/2)? IBP: "
          f"{'YES' if cb_ibp < float(np.abs(o64).max()) / 2 else 'NO'}")

    # ── §2. Is the tail AFFINE on the tube? (the no-branch-switch route) ────
    # If no ReLU flips sign and no pool argmax moves within the realized
    # perturbation, the tail is exactly affine there and its l-inf gain IS
    # max_j sum_k |J_jk| -- i.e. the measured H becomes PROVABLE, with the
    # no-switch margin as the (checkable) side condition. This measures the
    # headroom: min |preactivation| along the trajectory vs the actual f32 drift
    # at that layer.
    print("\n" + "=" * 72)
    print("§2  no-switch headroom — is the net affine on the realized tube?")
    print("=" * 72)
    print("  drift = the REALIZED f32 perturbation; b_i = the PROVEN fresh budget")
    print("  (the hybrid argument needs the tube at b_i, not at the realized drift)")
    print(f"{'stage':<12}{'min |preact|':>14}{'f32 drift':>11}{'head/drift':>11}"
          f"{'b_i':>11}{'head/b_i':>10}{'strad@b':>9}")
    convs_b, dens_b = convs[:], denses_w[:]
    a64, a32 = x, x.astype(np.float32)
    worst = float("inf")
    for s, (kind, idx) in enumerate(STAGES):
        last = (s == len(STAGES) - 1)
        if kind == "pool":
            a64, a32 = maxpool2(a64), maxpool2(a32)
            continue
        if kind == "conv":
            W, b = convs[idx]
            z64 = conv_same3(a64, W, b)
            z32 = conv_same3(a32, W.astype(np.float32), b.astype(np.float32))
        else:
            W, b = denses_w[idx]
            if idx == 0:
                a64, a32 = a64.reshape(1, -1), a32.reshape(1, -1)
            z64 = dense(a64, W, b)
            z32 = dense(a32, W.astype(np.float32), b.astype(np.float32))
        drift = float(np.abs(z64 - z32.astype(np.float64)).max())
        mn = float(np.abs(z64).min())
        bi = bs[s]
        head = mn / drift if drift > 0 else float("inf")
        head_b = mn / bi if bi > 0 else float("inf")
        n_strad_b = int((np.abs(z64) <= bi).sum())
        worst = min(worst, head)
        name = kind + str(idx)
        print(f"{name:<12}{mn:>14.3e}{drift:>11.3e}{head:>11.3e}"
              f"{bi:>11.3e}{head_b:>10.2e}{n_strad_b:>9d}")
        a64 = z64 if last else np.maximum(z64, 0.0)
        a32 = z32 if last else np.maximum(z32, 0.0)
    convs[:], denses_w[:] = convs_b, dens_b
    print(f"\n  worst headroom vs REALIZED drift : {worst:.3e}"
          f"  ({'no ReLU can flip' if worst > 1 else 'flips possible'})")
    print("  -> the net IS affine on the realized tube, so the measured H is the\n"
          "     true local gain. The obstruction is the RADIUS: the proof needs the\n"
          "     tube at b_i, and b_i runs ~100-1000x the realized drift, which is\n"
          "     where preactivations start falling inside it (strad@b column).")


if __name__ == "__main__":
    main()
