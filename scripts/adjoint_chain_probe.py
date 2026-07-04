"""Adjoint-chain probe: measure the tail gains H_i that discharge
`Proofs.chain_adjointClose` (AdjointChainBridge.lean) at the MEASURED tier,
and validate the resulting depth-LINEAR chain budget against real silicon.

The theorem (proven once, by induction on the layer list, 3-axiom clean):

    |chainF x - chainR x|_elementwise  <=  chainBudget = sum_i H_i * b_i

  b_i = layer i's FRESH budget on the window — the proven per-op FloatClose
        modulus at input error 0, here layerBudget(u32, m, w'_i, beta_i, A, 0)
        (FloatBridge.lean:606).
  H_i = a windowed Lipschitz gain of the REAL tail f_n ∘ … ∘ f_{i+1}
        (`LipOnWindow`). The telescoping is exact — nothing is linearized —
        so soundness rests entirely on how the H_i are discharged.

Two discharges compared here, on relu∘dense stacks (width 64) of depth
3/6/12/24, f32 via IREE on gfx1100 (rocm/hip) vs an f64 numpy oracle:

  PROVEN   (tailGains_suffixProd): H_i = prod_{j>i} m·max|W_j| — the
           worst-case per-layer gains (lipOnWindow_dense/relu). This is
           EXACTLY the old FloatClose.comp interval fold; it explodes
           (m·w)^depth and is ~10^6× loose at depth 3, ~10^40× at depth 24.
  MEASURED (this probe): H_i = the L∞→L∞ operator norm (max abs row sum) of
           the tail Jacobian evaluated along the trajectory, maxed over
           samples. Depth-linear budget, non-vacuous at depth 24 (~0.8 vs
           activations O(10)), observed ~10^4–10^5× above true error at all
           depths (the residual slack is the per-layer Higham-vs-FMA
           conservatism, NOT depth compounding).

MEASURED-tier caveat (state it wherever these numbers are used): the
on-trajectory Jacobian norm is a *local* probe of the window-supremum
Lipschitz constant `LipOnWindow` demands; like esig/egelu it is supplied as
a named, provenance-stated hypothesis, not proven. The trajectory-tube
refinement (gain over a tube of radius = accumulated budget) is the v2 that
would close most of that gap formally.

Run under the IREE/JAX venv (see ROCM.md):
  /home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python scripts/adjoint_chain_probe.py
"""
import numpy as np

U32 = 2.0 ** -24
B, N = 32, 64
NJAC = 8            # samples used for the measured tail-gain max
rng = np.random.default_rng(42)


# ── IREE on gfx1100 (rocm backend / hip runtime) ─────────────────────────────
def run_iree(mlir: str, inputs, fn: str = "main", module_name: str = "module"):
    from iree.compiler import compile_str
    from iree import runtime as ireert
    vmfb = compile_str(mlir, target_backends=["rocm"],
                       extra_args=["--iree-rocm-target=gfx1100"],
                       input_type="stablehlo")
    cfg = ireert.Config("hip")
    ctx = ireert.SystemContext(config=cfg)
    ctx.add_vm_module(ireert.VmModule.copy_buffer(ctx.instance, vmfb))
    out = getattr(ctx.modules, module_name)[fn](*inputs)
    if isinstance(out, (list, tuple)):
        return [np.asarray(o.to_host()) for o in out]
    return np.asarray(out.to_host())


def mlp_mlir(depth: int) -> str:
    """relu(dense) × depth as StableHLO — the deployed dense kernel shape."""
    A = f"tensor<{B}x{N}xf32>"
    W = f"tensor<{N}x{N}xf32>"
    V = f"tensor<{N}xf32>"
    args = [f"%x: {A}"] + [f"%w{i}: {W}, %b{i}: {V}" for i in range(depth)]
    lines = [f"module {{",
             f"  func.func @main({', '.join(args)}) -> {A} {{",
             f"    %zero = stablehlo.constant dense<0.0> : {A}"]
    h = "%x"
    for i in range(depth):
        lines += [
            f"    %d{i} = stablehlo.dot_general {h}, %w{i}, "
            f"contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] "
            f": ({A}, {W}) -> {A}",
            f"    %bb{i} = stablehlo.broadcast_in_dim %b{i}, dims = [1] "
            f": ({V}) -> {A}",
            f"    %s{i} = stablehlo.add %d{i}, %bb{i} : {A}",
            f"    %h{i} = stablehlo.maximum %s{i}, %zero : {A}",
        ]
        h = f"%h{i}"
    lines += [f"    return {h} : {A}", f"  }}", f"}}"]
    return "\n".join(lines)


def layer_budget_fresh(u, m, wmax, beta, A):
    """layerBudget u m w beta A 0 — FloatBridge.lean:606 at E=0 (the
    LayerCert.of_floatClose fresh budget)."""
    return ((1 + u) ** (m + 2) - 1) * (m * wmax * A + beta)


# ═══════════════════════════════════════════════════════════════════════════
# (2) COMMITTED MNIST-CNN — the adjoint chain on a real rendered net
#     (heterogeneous per-stage windows — the v2-ladder item 3 shape, evaluated
#     numerically; the induction is identical, the windows are just indexed)
# ═══════════════════════════════════════════════════════════════════════════
def conv_same3(x, W, b, dt):                              # NCHW / OIHW, 3×3 SAME
    x, W, b = x.astype(dt), W.astype(dt), b.astype(dt)
    N, C, H, Wd = x.shape
    Oc = W.shape[0]
    xp = np.zeros((N, C, H + 2, Wd + 2), dt)
    xp[:, :, 1:H + 1, 1:Wd + 1] = x
    out = np.zeros((N, Oc, H, Wd), dt)
    for di in range(3):
        for dj in range(3):
            out += np.einsum('nchw,oc->nohw', xp[:, :, di:di + H, dj:dj + Wd],
                             W[:, :, di, dj], optimize=True)
    return out + b[None, :, None, None]


def maxpool2(a):
    N, C, H, Wd = a.shape
    return a.reshape(N, C, H // 2, 2, Wd // 2, 2).max(axis=(3, 5))


def cnn_probe():
    """chainBudget = Σ_i H_i·b_i on the committed MNIST-CNN render
    (conv1→relu→conv2→relu→maxpool→flatten→dense→relu→dense→relu→dense),
    per-stage fresh budgets from the proven layerBudget, tail gains H_i
    measured as the L∞ norm of the exact tail Jacobian (10 rows — one
    backward pass per logit), vs the proven suffix-product face and the
    true GPU drift from kernel_faithfulness_probe's per-stage render."""
    from kernel_faithfulness_probe import cnn_mlir, layer_budget

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    print("\n" + "═" * 78)
    print("(2) COMMITTED MNIST-CNN render — adjoint chain vs interval fold, gfx1100")
    print("═" * 78)
    # renderer-realistic magnitude profile (same as kernel_faithfulness_probe)
    x = rng.standard_normal((4, 1, 28, 28)).astype(np.float32)
    W0 = (rng.standard_normal((32, 1, 3, 3)) * 0.1).astype(np.float32)
    b0 = (rng.standard_normal(32) * 0.1).astype(np.float32)
    W1 = (rng.standard_normal((32, 32, 3, 3)) * 0.05).astype(np.float32)
    b1 = (rng.standard_normal(32) * 0.1).astype(np.float32)
    W2 = (rng.standard_normal((6272, 512)) * 0.02).astype(np.float32)
    b2 = (rng.standard_normal(512) * 0.05).astype(np.float32)
    W3 = (rng.standard_normal((512, 512)) * 0.05).astype(np.float32)
    b3 = (rng.standard_normal(512) * 0.05).astype(np.float32)
    W4 = (rng.standard_normal((512, 10)) * 0.05).astype(np.float32)
    b4 = (rng.standard_normal(10) * 0.05).astype(np.float32)

    # true GPU drift at the logits (whole committed render, one IREE run)
    outs = run_iree(cnn_mlir(), [x, W0, b0, W1, b1, W2, b2, W3, b3, W4, b4])
    out_g = outs[-1].astype(np.float64)

    # f64 exact-ℝ trajectory
    dt = np.float64
    c0a = conv_same3(x, W0, b0, dt)
    h0 = np.maximum(c0a, 0)
    c1a = conv_same3(h0, W1, b1, dt)
    h1 = np.maximum(c1a, 0)
    pool = maxpool2(h1)
    flat = pool.reshape(4, -1)
    h2 = np.maximum(flat @ W2.astype(dt) + b2.astype(dt), 0)
    h3 = np.maximum(h2 @ W3.astype(dt) + b3.astype(dt), 0)
    out = h3 @ W4.astype(dt) + b4.astype(dt)
    true_err = float(np.abs(out_g - out).max())

    # ── per-stage FRESH budgets b_i (layerBudget at E=0, per-stage window) ──
    mags = lambda a: float(np.abs(a).max())
    stages = [   # (name, fan-in m, W, b, input activation)
        ("conv1+relu", 9,     W0, b0, x),
        ("conv2+relu", 288,   W1, b1, h0),
        ("maxpool",    None,  None, None, h1),   # exact in float: b = 0
        ("dense0+relu", 6272, W2, b2, flat),
        ("dense1+relu", 512,  W3, b3, h2),
        ("logits",      512,  W4, b4, h3),
    ]
    bs = [layer_budget(U32, m, mags(W), mags(b), mags(a), 0.0)
          if m is not None else 0.0
          for (_, m, W, b, a) in stages]

    # ── PROVEN tail gains: suffix products of per-stage row-sum gains ──
    gs = [9 * mags(W0), 288 * mags(W1), 1.0,
          6272 * mags(W2), 512 * mags(W3), 512 * mags(W4)]
    H_prov = [float(np.prod(gs[i + 1:])) for i in range(len(gs))]

    # ── MEASURED tail gains: L∞ norm of the exact tail Jacobian (jacrev:
    #    10 logits ⇒ 10 backward passes per stage), max over the 4 samples ──
    jW0, jb0 = jnp.asarray(W0, dt), jnp.asarray(b0, dt)
    jW1, jb1 = jnp.asarray(W1, dt), jnp.asarray(b1, dt)
    jW2, jb2 = jnp.asarray(W2, dt), jnp.asarray(b2, dt)
    jW3, jb3 = jnp.asarray(W3, dt), jnp.asarray(b3, dt)
    jW4, jb4 = jnp.asarray(W4, dt), jnp.asarray(b4, dt)

    def jconv(a, W, b):
        y = jax.lax.conv_general_dilated(a, W, (1, 1), ((1, 1), (1, 1)),
                                         dimension_numbers=('NCHW', 'OIHW',
                                                            'NCHW'))
        return y + b[None, :, None, None]

    def jpool(a):
        N, C, H, Wd = a.shape
        return a.reshape(N, C, H // 2, 2, Wd // 2, 2).max(axis=(3, 5))

    def tail(from_stage, a):
        """logits from stage `from_stage`'s OUTPUT activation a (batch 1)."""
        if from_stage <= 0:
            a = jnp.maximum(jconv(a, jW0, jb0), 0.0) if from_stage < 0 else a
        if from_stage <= 1:
            a = jnp.maximum(jconv(a, jW1, jb1), 0.0)
        if from_stage <= 2:
            a = jpool(a).reshape(a.shape[0], -1)
        if from_stage <= 3:
            a = jnp.maximum(a @ jW2 + jb2, 0.0)
        if from_stage <= 4:
            a = jnp.maximum(a @ jW3 + jb3, 0.0)
        return a @ jW4 + jb4

    acts = [h0, h1, flat, h2, h3]                          # stage outputs 1..5
    H_meas = []
    for i, act in enumerate(acts, start=1):
        gain = 0.0
        for s in range(4):
            a1 = jnp.asarray(act[s:s + 1], dt)
            J = jax.jacrev(lambda aa: tail(i, aa)[0])(a1)  # (10, *act.shape)
            gain = max(gain, float(jnp.abs(J.reshape(10, -1)).sum(axis=1).max()))
        H_meas.append(gain)
    H_meas.append(1.0)                                     # tail after logits = id

    cb_meas = sum(H * b for H, b in zip(H_meas, bs))
    cb_prov = sum(H * b for H, b in zip(H_prov, bs))
    print(f"\n{'stage':<13}{'fresh b_i':>12}{'H_i meas':>11}{'H_i·b_i':>11}"
          f"{'H_i proven':>12}")
    for (name, *_), b, Hm, Hp in zip(stages, bs, H_meas, H_prov):
        print(f"{name:<13}{b:>12.3e}{Hm:>11.3e}{Hm * b:>11.3e}{Hp:>12.3e}")
    print(f"\n  true GPU logits drift : {true_err:.3e}")
    print(f"  chainBudget measured-H: {cb_meas:.3e}  ({cb_meas / true_err:.1e}× true)")
    print(f"  chainBudget proven-H  : {cb_prov:.3e}  ({cb_prov / true_err:.1e}× true)"
          f"   [= the interval fold]")
    print(f"  logits magnitude      : {mags(out):.3e}")


# ═══════════════════════════════════════════════════════════════════════════
# (3) COMMITTED CIFAR-8 (no BN) — the depth-dominated regime
#     8 convs [16,16,32,32] + 4 pools (32→2) + 128→64→64→10 head = 15 stages,
#     small fan-ins (≤288) — the mirror image of the MNIST-CNN finding: here
#     the local Higham budgets are tame and COMPOSITION is the whole game.
# ═══════════════════════════════════════════════════════════════════════════
def cifar8_probe():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    from kernel_faithfulness_probe import layer_budget

    print("\n" + "═" * 78)
    print("(3) COMMITTED CIFAR-8 (no BN) — adjoint chain vs interval fold, gfx1100")
    print("═" * 78)
    # cifar8Verified (VerifiedNets.lean): conv 3→16,16,16,16,32,32,32,32 (3×3
    # SAME), pool after every 2 convs (32→16→8→4→2), flatten 128, dense
    # 64/64/10.  He-init magnitude profile.
    chans = [(3, 16), (16, 16), (16, 16), (16, 16),
             (16, 32), (32, 32), (32, 32), (32, 32)]
    convs = []
    for (ic, oc) in chans:
        std = (2.0 / (ic * 9)) ** 0.5
        convs.append(((rng.standard_normal((oc, ic, 3, 3)) * std)
                      .astype(np.float32),
                      (rng.standard_normal(oc) * 0.01).astype(np.float32)))
    dense_dims = [(128, 64), (64, 64), (64, 10)]
    denses = []
    for (di, do) in dense_dims:
        std = (2.0 / di) ** 0.5
        denses.append(((rng.standard_normal((di, do)) * std)
                       .astype(np.float32),
                       (rng.standard_normal(do) * 0.01).astype(np.float32)))
    x = rng.standard_normal((4, 3, 32, 32)).astype(np.float32)

    # ── emit the forward as StableHLO (conv/bias/relu ×2 → pool, ×4, head) ──
    T = lambda *s: "tensor<" + "x".join(map(str, s)) + "xf32>"

    def conv_op(o, xn, Wn, src_t, w_t, out_t):
        return (f'    {o} = "stablehlo.convolution"({xn}, {Wn}) {{\n'
                f'        batch_group_count = 1 : i64,\n'
                f'        dimension_numbers = #stablehlo.conv<raw\n'
                f'          input_batch_dimension = 0, input_feature_dimension = 1,\n'
                f'          input_spatial_dimensions = [2, 3],\n'
                f'          kernel_output_feature_dimension = 0,\n'
                f'          kernel_input_feature_dimension = 1,\n'
                f'          kernel_spatial_dimensions = [2, 3],\n'
                f'          output_batch_dimension = 0, output_feature_dimension = 1,\n'
                f'          output_spatial_dimensions = [2, 3]>,\n'
                f'        feature_group_count = 1 : i64,\n'
                f'        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,\n'
                f'        rhs_dilation = array<i64: 1, 1>,\n'
                f'        window_strides = array<i64: 1, 1>\n'
                f'      }} : ({src_t}, {w_t}) -> {out_t}\n')

    args = ["%x: " + T(4, 3, 32, 32)]
    for i, ((ic, oc), _) in enumerate(zip(chans, convs)):
        args += [f"%cw{i}: {T(oc, ic, 3, 3)}", f"%cb{i}: {T(oc)}"]
    for i, (di, do) in enumerate(dense_dims):
        args += [f"%dw{i}: {T(di, do)}", f"%db{i}: {T(do)}"]
    body = []
    cur, sp, ch = "%x", 32, 3
    k = 0
    for stage in range(4):
        for _ in range(2):
            ic, oc = chans[k]
            src_t, out_t = T(4, ic, sp, sp), T(4, oc, sp, sp)
            body.append(conv_op(f"%cv{k}", cur, f"%cw{k}", src_t,
                                T(oc, ic, 3, 3), out_t))
            body.append(f"    %cbb{k} = stablehlo.broadcast_in_dim %cb{k}, "
                        f"dims = [1] : ({T(oc)}) -> {out_t}\n")
            body.append(f"    %ca{k} = stablehlo.add %cv{k}, %cbb{k} : {out_t}\n")
            body.append(f"    %cz{k} = stablehlo.constant dense<0.0> : {out_t}\n")
            body.append(f"    %ch{k} = stablehlo.maximum %ca{k}, %cz{k} : {out_t}\n")
            cur, ch = f"%ch{k}", oc
            k += 1
        in_t, out_t = T(4, ch, sp, sp), T(4, ch, sp // 2, sp // 2)
        body.append(f"    %ninf{stage} = stablehlo.constant "
                    f"dense<0xFF800000> : tensor<f32>\n")
        body.append(f'    %pool{stage} = "stablehlo.reduce_window"({cur}, '
                    f'%ninf{stage}) ({{\n'
                    f"      ^bb0(%a: tensor<f32>, %bb: tensor<f32>):\n"
                    f"        %m = stablehlo.maximum %a, %bb : tensor<f32>\n"
                    f'        "stablehlo.return"(%m) : (tensor<f32>) -> ()\n'
                    f"      }}) {{ window_dimensions = array<i64: 1, 1, 2, 2>,\n"
                    f"           window_strides    = array<i64: 1, 1, 2, 2> }}\n"
                    f"      : ({in_t}, tensor<f32>) -> {out_t}\n")
        cur, sp = f"%pool{stage}", sp // 2
    body.append(f"    %flat = stablehlo.reshape {cur} : "
                f"({T(4, 32, 2, 2)}) -> {T(4, 128)}\n")
    cur, d = "%flat", 128
    for i, (di, do) in enumerate(dense_dims):
        body.append(f"    %dd{i} = stablehlo.dot_general {cur}, %dw{i}, "
                    f"contracting_dims = [1] x [0], precision = "
                    f"[DEFAULT, DEFAULT] : ({T(4, di)}, {T(di, do)}) -> "
                    f"{T(4, do)}\n")
        body.append(f"    %dbb{i} = stablehlo.broadcast_in_dim %db{i}, "
                    f"dims = [1] : ({T(do)}) -> {T(4, do)}\n")
        body.append(f"    %da{i} = stablehlo.add %dd{i}, %dbb{i} : {T(4, do)}\n")
        if i < 2:
            body.append(f"    %dz{i} = stablehlo.constant dense<0.0> : "
                        f"{T(4, do)}\n")
            body.append(f"    %dh{i} = stablehlo.maximum %da{i}, %dz{i} : "
                        f"{T(4, do)}\n")
            cur = f"%dh{i}"
        else:
            cur = f"%da{i}"
    mlir = ("module {\n  func.func @main(\n      "
            + ",\n      ".join(args)
            + f"\n    ) -> {T(4, 10)} {{\n"
            + "".join(body)
            + f"    return {cur} : {T(4, 10)}\n  }}\n}}\n")

    flat_inputs = [x]
    for w, b in convs:
        flat_inputs += [w, b]
    for w, b in denses:
        flat_inputs += [w, b]
    out_g = run_iree(mlir, flat_inputs).astype(np.float64)

    # ── f64 oracle trajectory, stage list for budgets/gains ────────────────
    dt = np.float64
    acts = []          # (name, kind, W, b, input_act, output_act)
    a = x.astype(dt)
    k = 0
    for stage in range(4):
        for _ in range(2):
            w, b = convs[k]
            z = conv_same3(a, w, b, dt)
            h = np.maximum(z, 0)
            acts.append((f"conv{k+1}+relu", "conv", w, b, a, h))
            a = h
            k += 1
        p = maxpool2(a)
        acts.append((f"pool{stage+1}", "pool", None, None, a, p))
        a = p
    a = a.reshape(4, -1)
    for i, (w, b) in enumerate(denses):
        z = a @ w.astype(dt) + b.astype(dt)
        h = np.maximum(z, 0) if i < 2 else z
        acts.append((f"dense{i}" + ("+relu" if i < 2 else " (logits)"),
                     "dense", w, b, a, h))
        a = h
    out = a
    true_err = float(np.abs(out_g - out).max())

    mags = lambda arr: float(np.abs(arr).max())
    bs, gs = [], []
    for (_name, kind, w, b, ain, _aout) in acts:
        if kind == "pool":
            bs.append(0.0)
            gs.append(1.0)
        else:
            m = w.shape[1] * 9 if kind == "conv" else w.shape[0]
            bs.append(layer_budget(U32, m, mags(w), mags(b),
                                   mags(ain.reshape(4, -1)), 0.0))
            gs.append(m * mags(w))
    H_prov = [float(np.prod(gs[i + 1:])) for i in range(len(gs))]

    # ── measured tail gains: jacrev per stage (10 logits ⇒ 10 VJPs) ────────
    jconvs = [(jnp.asarray(w, dt), jnp.asarray(b, dt)) for w, b in convs]
    jdenses = [(jnp.asarray(w, dt), jnp.asarray(b, dt)) for w, b in denses]

    def stage_fns():
        fns = []
        k = 0
        for stage in range(4):
            for _ in range(2):
                w, b = jconvs[k]

                def f(aa, w=w, b=b):
                    y = jax.lax.conv_general_dilated(
                        aa, w, (1, 1), ((1, 1), (1, 1)),
                        dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
                    return jnp.maximum(y + b[None, :, None, None], 0.0)
                fns.append(f)
                k += 1

            def p(aa):
                N, C, H, Wd = aa.shape
                return aa.reshape(N, C, H // 2, 2, Wd // 2, 2).max(axis=(3, 5))
            fns.append(p)
        for i, (w, b) in enumerate(jdenses):
            def d(aa, w=w, b=b, act=(i < 2)):
                if aa.ndim > 2:
                    aa = aa.reshape(aa.shape[0], -1)
                z = aa @ w + b
                return jnp.maximum(z, 0.0) if act else z
            fns.append(d)
        return fns

    fns = stage_fns()

    H_meas = []
    for i in range(len(acts) - 1):
        aout = acts[i][5]     # 4D conv/pool acts are fine: the first dense
        gain = 0.0            # stage fn flattens its input itself
        for s in range(4):
            a1 = jnp.asarray(aout[s:s + 1], dt)

            def tail(aa, i=i):
                for f in fns[i + 1:]:
                    aa = f(aa)
                return aa[0]
            J = jax.jacrev(tail)(a1)
            gain = max(gain, float(jnp.abs(J.reshape(10, -1)).sum(axis=1).max()))
        H_meas.append(gain)
    H_meas.append(1.0)

    cb_meas = sum(H * b for H, b in zip(H_meas, bs))
    cb_prov = sum(H * b for H, b in zip(H_prov, bs))
    print(f"\n{'stage':<18}{'fresh b_i':>12}{'H_i meas':>11}{'H_i·b_i':>11}"
          f"{'H_i proven':>13}")
    for (name, *_), b, Hm, Hp in zip(acts, bs, H_meas, H_prov):
        print(f"{name:<18}{b:>12.3e}{Hm:>11.3e}{Hm * b:>11.3e}{Hp:>13.3e}")
    print(f"\n  true GPU logits drift : {true_err:.3e}")
    print(f"  chainBudget measured-H: {cb_meas:.3e}  "
          f"({cb_meas / true_err:.1e}× true)")
    print(f"  chainBudget proven-H  : {cb_prov:.3e}  "
          f"({cb_prov / true_err:.1e}× true)   [= the interval fold]")
    print(f"  logits magnitude      : {mags(out):.3e}")


# ═══════════════════════════════════════════════════════════════════════════
# (4) COMMITTED CIFAR-8-BN — the normalization case
#     cifar8Verified + per-EXAMPLE per-channel BN (stats over h·w, γ=1/β=0 —
#     "train=eval", no batch coupling, VerifiedNets.lean cifar8BnVerified).
#     BN fresh budgets from the BnFloatBridge parts (cifar_bn_margin_probe's
#     transcriptions), istd at the A-POSTERIORI floor σ²min+ε; interval-face
#     BN gain likewise a-posteriori — the strict ε-floor face is far worse.
# ═══════════════════════════════════════════════════════════════════════════
EPS_BN = 1e-5
ERS = 2 * U32          # rsqrt accuracy, pinned on gfx1100 (cifar_bn_margin_probe)


def bn_mean_budget(u, n, A):
    return u * ((1 + u) ** (n + 1) * A) + ((1 + u) ** (n + 1) - 1) * A


def mul_err(u, A, C, ea, ec):
    return u * ((A + ea) * (C + ec)) + (A * ec + ea * C + ea * ec)


def bn_var_budget(u, D, emean, n):
    g = (1 + u) ** (n + 1) - 1
    es1 = u * (D + emean) + emean
    esq = mul_err(u, D, D, es1, es1)
    return u * (D * D + g * (D * D + esq) + esq) + (g * (D * D + esq) + esq)


def bn_istd_budget(ers, evar, floor):
    return ers / np.sqrt(floor) + evar / (2 * floor * np.sqrt(floor))


def bn_norm_budget(u, D, S, G, Bb, emean, eistd):
    es2 = mul_err(u, D, S, u * (D + emean) + emean, eistd)
    es3 = mul_err(u, G, D * S, 0.0, es2)
    return u * (G * (D * S) + es3 + Bb) + es3


def bn_per_example(z):
    """per-(sample,channel) BN over h·w, γ=1/β=0 (population var, EPS_BN)."""
    mu = z.mean(axis=(2, 3), keepdims=True)
    var = ((z - mu) ** 2).mean(axis=(2, 3), keepdims=True)
    return (z - mu) / np.sqrt(var + EPS_BN)


def cifar8bn_probe():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    from kernel_faithfulness_probe import layer_budget

    print("\n" + "═" * 78)
    print("(4) COMMITTED CIFAR-8-BN — adjoint chain vs interval fold, gfx1100")
    print("═" * 78)
    chans = [(3, 16), (16, 16), (16, 16), (16, 16),
             (16, 32), (32, 32), (32, 32), (32, 32)]
    convs = []
    for (ic, oc) in chans:
        std = (2.0 / (ic * 9)) ** 0.5
        convs.append(((rng.standard_normal((oc, ic, 3, 3)) * std)
                      .astype(np.float32),
                      (rng.standard_normal(oc) * 0.01).astype(np.float32)))
    dense_dims = [(128, 64), (64, 64), (64, 10)]
    denses = []
    for (di, do) in dense_dims:
        std = (2.0 / di) ** 0.5
        denses.append(((rng.standard_normal((di, do)) * std)
                       .astype(np.float32),
                       (rng.standard_normal(do) * 0.01).astype(np.float32)))
    x = rng.standard_normal((4, 3, 32, 32)).astype(np.float32)

    # ── StableHLO emission: conv→BN(per-example)→relu ×8 + pools + head ────
    T = lambda *s: "tensor<" + "x".join(map(str, s)) + "xf32>"

    def conv_op(o, xn, Wn, src_t, w_t, out_t):
        return (f'    {o} = "stablehlo.convolution"({xn}, {Wn}) {{\n'
                f'        batch_group_count = 1 : i64,\n'
                f'        dimension_numbers = #stablehlo.conv<raw\n'
                f'          input_batch_dimension = 0, input_feature_dimension = 1,\n'
                f'          input_spatial_dimensions = [2, 3],\n'
                f'          kernel_output_feature_dimension = 0,\n'
                f'          kernel_input_feature_dimension = 1,\n'
                f'          kernel_spatial_dimensions = [2, 3],\n'
                f'          output_batch_dimension = 0, output_feature_dimension = 1,\n'
                f'          output_spatial_dimensions = [2, 3]>,\n'
                f'        feature_group_count = 1 : i64,\n'
                f'        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,\n'
                f'        rhs_dilation = array<i64: 1, 1>,\n'
                f'        window_strides = array<i64: 1, 1>\n'
                f'      }} : ({src_t}, {w_t}) -> {out_t}\n')

    def spatial_sum(o, xn, c, sp, tag):
        """reduce_window(add) over the full h·w extent → (4,c,1,1)."""
        in_t, out_t = T(4, c, sp, sp), T(4, c, 1, 1)
        return (f"    %z{tag} = stablehlo.constant dense<0.0> : tensor<f32>\n"
                f'    {o} = "stablehlo.reduce_window"({xn}, %z{tag}) ({{\n'
                f"      ^bb0(%a: tensor<f32>, %bb: tensor<f32>):\n"
                f"        %m = stablehlo.add %a, %bb : tensor<f32>\n"
                f'        "stablehlo.return"(%m) : (tensor<f32>) -> ()\n'
                f"      }}) {{ window_dimensions = array<i64: 1, 1, {sp}, {sp}>,\n"
                f"           window_strides    = array<i64: 1, 1, {sp}, {sp}> }}\n"
                f"      : ({in_t}, tensor<f32>) -> {out_t}\n")

    def bn_ops(k, xn, c, sp):
        """per-example per-channel BN of %xn: (4,c,sp,sp), γ=1/β=0."""
        t4, t1 = T(4, c, sp, sp), T(4, c, 1, 1)
        hw = float(sp * sp)
        s = spatial_sum(f"%bs{k}", xn, c, sp, f"bn{k}a")
        s += (
            f"    %bhw{k} = stablehlo.constant dense<{hw:.1f}> : {t1}\n"
            f"    %bmu{k} = stablehlo.divide %bs{k}, %bhw{k} : {t1}\n"
            f"    %bmub{k} = stablehlo.broadcast_in_dim %bmu{k}, "
            f"dims = [0, 1, 2, 3] : ({t1}) -> {t4}\n"
            f"    %bd{k} = stablehlo.subtract {xn}, %bmub{k} : {t4}\n"
            f"    %bd2{k} = stablehlo.multiply %bd{k}, %bd{k} : {t4}\n")
        s += spatial_sum(f"%bs2{k}", f"%bd2{k}", c, sp, f"bn{k}b")
        s += (
            f"    %bvar{k} = stablehlo.divide %bs2{k}, %bhw{k} : {t1}\n"
            f"    %beps{k} = stablehlo.constant dense<{EPS_BN:.1e}> : {t1}\n"
            f"    %bve{k} = stablehlo.add %bvar{k}, %beps{k} : {t1}\n"
            f"    %bistd{k} = stablehlo.rsqrt %bve{k} : {t1}\n"
            f"    %bistdb{k} = stablehlo.broadcast_in_dim %bistd{k}, "
            f"dims = [0, 1, 2, 3] : ({t1}) -> {t4}\n"
            f"    %bn{k} = stablehlo.multiply %bd{k}, %bistdb{k} : {t4}\n")
        return s, f"%bn{k}"

    args = ["%x: " + T(4, 3, 32, 32)]
    for i, ((ic, oc), _) in enumerate(zip(chans, convs)):
        args += [f"%cw{i}: {T(oc, ic, 3, 3)}", f"%cb{i}: {T(oc)}"]
    for i, (di, do) in enumerate(dense_dims):
        args += [f"%dw{i}: {T(di, do)}", f"%db{i}: {T(do)}"]
    body = []
    cur, sp, ch = "%x", 32, 3
    k = 0
    for stage in range(4):
        for _ in range(2):
            ic, oc = chans[k]
            src_t, out_t = T(4, ic, sp, sp), T(4, oc, sp, sp)
            body.append(conv_op(f"%cv{k}", cur, f"%cw{k}", src_t,
                                T(oc, ic, 3, 3), out_t))
            body.append(f"    %cbb{k} = stablehlo.broadcast_in_dim %cb{k}, "
                        f"dims = [1] : ({T(oc)}) -> {out_t}\n")
            body.append(f"    %ca{k} = stablehlo.add %cv{k}, %cbb{k} : {out_t}\n")
            bs, bn_out = bn_ops(k, f"%ca{k}", oc, sp)
            body.append(bs)
            body.append(f"    %cz{k} = stablehlo.constant dense<0.0> : {out_t}\n")
            body.append(f"    %ch{k} = stablehlo.maximum {bn_out}, %cz{k} : "
                        f"{out_t}\n")
            cur, ch = f"%ch{k}", oc
            k += 1
        in_t, out_t = T(4, ch, sp, sp), T(4, ch, sp // 2, sp // 2)
        body.append(f"    %ninf{stage} = stablehlo.constant "
                    f"dense<0xFF800000> : tensor<f32>\n")
        body.append(f'    %pool{stage} = "stablehlo.reduce_window"({cur}, '
                    f'%ninf{stage}) ({{\n'
                    f"      ^bb0(%a: tensor<f32>, %bb: tensor<f32>):\n"
                    f"        %m = stablehlo.maximum %a, %bb : tensor<f32>\n"
                    f'        "stablehlo.return"(%m) : (tensor<f32>) -> ()\n'
                    f"      }}) {{ window_dimensions = array<i64: 1, 1, 2, 2>,\n"
                    f"           window_strides    = array<i64: 1, 1, 2, 2> }}\n"
                    f"      : ({in_t}, tensor<f32>) -> {out_t}\n")
        cur, sp = f"%pool{stage}", sp // 2
    body.append(f"    %flat = stablehlo.reshape {cur} : "
                f"({T(4, 32, 2, 2)}) -> {T(4, 128)}\n")
    cur = "%flat"
    for i, (di, do) in enumerate(dense_dims):
        body.append(f"    %dd{i} = stablehlo.dot_general {cur}, %dw{i}, "
                    f"contracting_dims = [1] x [0], precision = "
                    f"[DEFAULT, DEFAULT] : ({T(4, di)}, {T(di, do)}) -> "
                    f"{T(4, do)}\n")
        body.append(f"    %dbb{i} = stablehlo.broadcast_in_dim %db{i}, "
                    f"dims = [1] : ({T(do)}) -> {T(4, do)}\n")
        body.append(f"    %da{i} = stablehlo.add %dd{i}, %dbb{i} : {T(4, do)}\n")
        if i < 2:
            body.append(f"    %dz{i} = stablehlo.constant dense<0.0> : "
                        f"{T(4, do)}\n")
            body.append(f"    %dh{i} = stablehlo.maximum %da{i}, %dz{i} : "
                        f"{T(4, do)}\n")
            cur = f"%dh{i}"
        else:
            cur = f"%da{i}"
    mlir = ("module {\n  func.func @main(\n      "
            + ",\n      ".join(args)
            + f"\n    ) -> {T(4, 10)} {{\n"
            + "".join(body)
            + f"    return {cur} : {T(4, 10)}\n  }}\n}}\n")

    flat_inputs = [x]
    for w, b in convs:
        flat_inputs += [w, b]
    for w, b in denses:
        flat_inputs += [w, b]
    out_g = run_iree(mlir, flat_inputs).astype(np.float64)

    # ── f64 oracle + per-stage budgets/gains ───────────────────────────────
    dt = np.float64
    mags = lambda arr: float(np.abs(arr).max())
    acts, bs, gs = [], [], []
    a = x.astype(dt)
    k = 0
    for stage in range(4):
        for _ in range(2):
            w, b = convs[k]
            z = conv_same3(a, w, b, dt)
            m = w.shape[1] * 9
            acts.append((f"conv{k+1}", "conv", a, z))
            bs.append(layer_budget(U32, m, mags(w), mags(b),
                                   mags(a.reshape(4, -1)), 0.0))
            gs.append(m * mags(w))
            # BN(+relu) stage: budgets from the BnFloatBridge parts at the
            # a-posteriori operating profile (floor = σ²min + ε)
            n_bn = z.shape[2] * z.shape[3]
            mu = z.mean(axis=(2, 3), keepdims=True)
            var = ((z - mu) ** 2).mean(axis=(2, 3), keepdims=True)
            D = mags(z - mu)
            S = float((1.0 / np.sqrt(var + EPS_BN)).max())
            floor = float(var.min()) + EPS_BN
            emean = bn_mean_budget(U32, n_bn, mags(z))
            evar = bn_var_budget(U32, D, emean, n_bn)
            eistd = bn_istd_budget(ERS, evar, floor)
            h = np.maximum(bn_per_example(z), 0)
            acts.append((f"bn{k+1}+relu", "bn", z, h))
            bs.append(bn_norm_budget(U32, D, S, 1.0, 0.0, emean, eistd))
            gs.append(1.0 * (2 * S + 2 * D * D / floor ** 1.5))
            a = h
            k += 1
        p = maxpool2(a)
        acts.append((f"pool{stage+1}", "pool", a, p))
        bs.append(0.0)
        gs.append(1.0)
        a = p
    a = a.reshape(4, -1)
    for i, (w, b) in enumerate(denses):
        z = a @ w.astype(dt) + b.astype(dt)
        h = np.maximum(z, 0) if i < 2 else z
        acts.append((f"dense{i}" + ("+relu" if i < 2 else " (logits)"),
                     "dense", a, h))
        bs.append(layer_budget(U32, w.shape[0], mags(w), mags(b),
                               mags(a), 0.0))
        gs.append(w.shape[0] * mags(w))
        a = h
    out = a
    true_err = float(np.abs(out_g - out).max())
    H_prov = [float(np.prod(gs[i + 1:])) for i in range(len(gs))]

    # ── measured tail gains (per-sample: per-example BN ⇒ no batch coupling)
    jconvs = [(jnp.asarray(w, dt), jnp.asarray(b, dt)) for w, b in convs]
    jdenses = [(jnp.asarray(w, dt), jnp.asarray(b, dt)) for w, b in denses]

    def stage_fns():
        fns = []
        k = 0
        for stage in range(4):
            for _ in range(2):
                w, b = jconvs[k]

                def cf(aa, w=w, b=b):
                    y = jax.lax.conv_general_dilated(
                        aa, w, (1, 1), ((1, 1), (1, 1)),
                        dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
                    return y + b[None, :, None, None]
                fns.append(cf)

                def bf(zz):
                    mu = zz.mean(axis=(2, 3), keepdims=True)
                    var = ((zz - mu) ** 2).mean(axis=(2, 3), keepdims=True)
                    return jnp.maximum((zz - mu) / jnp.sqrt(var + EPS_BN), 0.0)
                fns.append(bf)
                k += 1

            def pf(aa):
                N, C, H, Wd = aa.shape
                return aa.reshape(N, C, H // 2, 2, Wd // 2, 2).max(axis=(3, 5))
            fns.append(pf)
        for i, (w, b) in enumerate(jdenses):
            def df(aa, w=w, b=b, act=(i < 2)):
                if aa.ndim > 2:
                    aa = aa.reshape(aa.shape[0], -1)
                z = aa @ w + b
                return jnp.maximum(z, 0.0) if act else z
            fns.append(df)
        return fns

    fns = stage_fns()
    H_meas = []
    for i in range(len(acts) - 1):
        aout = acts[i][3]
        gain = 0.0
        for s in range(4):
            a1 = jnp.asarray(aout[s:s + 1], dt)

            def tail(aa, i=i):
                for f in fns[i + 1:]:
                    aa = f(aa)
                return aa[0]
            J = jax.jacrev(tail)(a1)
            gain = max(gain, float(jnp.abs(J.reshape(10, -1)).sum(axis=1).max()))
        H_meas.append(gain)
    H_meas.append(1.0)

    cb_meas = sum(H * b for H, b in zip(H_meas, bs))
    cb_prov = sum(H * b for H, b in zip(H_prov, bs))
    print(f"\n{'stage':<18}{'fresh b_i':>12}{'H_i meas':>11}{'H_i·b_i':>11}"
          f"{'H_i proven':>13}")
    for (name, *_), b, Hm, Hp in zip(acts, bs, H_meas, H_prov):
        print(f"{name:<18}{b:>12.3e}{Hm:>11.3e}{Hm * b:>11.3e}{Hp:>13.3e}")
    print(f"\n  true GPU logits drift : {true_err:.3e}")
    print(f"  chainBudget measured-H: {cb_meas:.3e}  "
          f"({cb_meas / true_err:.1e}× true)")
    print(f"  chainBudget proven-H  : {cb_prov:.3e}  "
          f"({cb_prov / true_err:.1e}× true)   [= interval fold, a-post floor]")
    print(f"  logits magnitude      : {mags(out):.3e}")
    print("\n  BN budgets/gains at the A-POSTERIORI floor (σ²min+ε); the strict")
    print("  ε-floor Lean face multiplies each BN gain by ~(σ²+ε)^1.5/ε^1.5.")


# ═══════════════════════════════════════════════════════════════════════════
# (5) ResNet-34 @224² (resnet34Verified twin) — residual blocks as chain layers
#     Eval-mode BN (running-stats per-channel affine, calibrated from the
#     probe batch — the DEPLOYED forward, BnEvalFloatBridge's case; γ=1/β=0).
#     Chain granularity = the r34_floatBridges granularity: stem / maxpool /
#     16 residual blocks / GAP / dense head = 20 stages. Block fresh budgets
#     are the mini interval fold INSIDE the block (5-9 ops — harmless); the
#     depth-wise composition across the 20 stages is the adjoint chain.
#     Tail gains measured in f32 on the GPU (jax rocm) — gains are O(1..1e3)
#     numbers, f32 precision is ample; oracle/budgets stay f64 CPU.
# ═══════════════════════════════════════════════════════════════════════════
def conv_np(x, W, stride, pad, dt):
    """generic NCHW/OIHW conv, any kernel/stride/pad (f64 oracle)."""
    x, W = x.astype(dt), W.astype(dt)
    N, C, H, Wd = x.shape
    Oc, _, kh, kw = W.shape
    Ho = (H + 2 * pad - kh) // stride + 1
    Wo = (Wd + 2 * pad - kw) // stride + 1
    xp = np.zeros((N, C, H + 2 * pad, Wd + 2 * pad), dt)
    xp[:, :, pad:pad + H, pad:pad + Wd] = x
    out = np.zeros((N, Oc, Ho, Wo), dt)
    for di in range(kh):
        for dj in range(kw):
            out += np.einsum('nchw,oc->nohw',
                             xp[:, :, di:di + stride * Ho:stride,
                                dj:dj + stride * Wo:stride],
                             W[:, :, di, dj], optimize=True)
    return out


def affine_fresh(u, amax, bmax, A, E):
    """fresh+inherited budget of the per-channel eval-BN affine fl(fl(a·x)⊕b)
    (bnEvalErr shape: fan-in 1, no Higham γ, no rsqrt)."""
    return (2 * u + u * u) * (amax * (A + E) + bmax) + amax * E


def r34_probe():
    import jax
    import jax.numpy as jnp
    from kernel_faithfulness_probe import layer_budget

    print("\n" + "═" * 78)
    print("(5) ResNet-34 @224² twin (resnet34Verified) — adjoint chain, gfx1100")
    print("═" * 78)
    B = 2
    EPS = 1e-5
    dt = np.float64
    mags = lambda arr: float(np.abs(arr).max())

    # ── build weights (He) and the block plan from the committed spec ──────
    stage_cfg = [(64, 64, 3, 1), (64, 128, 4, 2),
                 (128, 256, 6, 2), (256, 512, 3, 2)]

    def he_conv(oc, ic, k):
        std = (2.0 / (ic * k * k)) ** 0.5
        return (rng.standard_normal((oc, ic, k, k)) * std).astype(np.float32)

    Wstem = he_conv(64, 3, 7)
    blocks = []          # dicts: W1, W2, [Wd], ic, oc, stride
    for (ic, oc, n, s) in stage_cfg:
        for j in range(n):
            bic, bs = (ic, s) if j == 0 else (oc, 1)
            blk = {"W1": he_conv(oc, bic, 3), "W2": he_conv(oc, oc, 3),
                   "ic": bic, "oc": oc, "stride": bs,
                   "down": (bs != 1 or bic != oc)}
            if blk["down"]:
                blk["Wd"] = he_conv(oc, bic, 1)
            blocks.append(blk)
    Whead = ((rng.standard_normal((512, 10)) * (2.0 / 512) ** 0.5)
             .astype(np.float32))
    bhead = (rng.standard_normal(10) * 0.01).astype(np.float32)
    x = rng.standard_normal((B, 3, 224, 224)).astype(np.float32)

    # ── pass 1: f64 forward on f32-cast weights, CALIBRATE eval-BN affines ─
    def calib(z):
        mu = z.mean(axis=(0, 2, 3))
        var = z.var(axis=(0, 2, 3))
        a = (1.0 / np.sqrt(var + EPS)).astype(np.float32)
        b = (-mu * a).astype(np.float32)
        return a, b

    def bn_apply(z, ab):
        a, b = ab
        return z * a.astype(z.dtype)[None, :, None, None] \
            + b.astype(z.dtype)[None, :, None, None]

    xa = x.astype(dt)
    z = conv_np(xa, Wstem, 2, 3, dt)
    ab_stem = calib(z)
    cal = {"stem": ab_stem}
    h = np.maximum(bn_apply(z, ab_stem), 0)
    h = maxpool2(h)
    for i, blk in enumerate(blocks):
        z1 = conv_np(h, blk["W1"], blk["stride"], 1, dt)
        blk["ab1"] = calib(z1)
        y1 = np.maximum(bn_apply(z1, blk["ab1"]), 0)
        z2 = conv_np(y1, blk["W2"], 1, 1, dt)
        blk["ab2"] = calib(z2)
        y2 = bn_apply(z2, blk["ab2"])
        if blk["down"]:
            zd = conv_np(h, blk["Wd"], blk["stride"], 0, dt)
            blk["abd"] = calib(zd)
            sk = bn_apply(zd, blk["abd"])
        else:
            sk = h
        h = np.maximum(y2 + sk, 0)

    # ── pass 2: the reference trajectory with the f32-cast affines,
    #    collecting per-stage activations + budgets + proven gains ──────────
    # Budgets use the EXACT data-dependent dot_close/denseErr form
    # (((1+u)^(m+2)−1)·(Σᵢ|Wᵢⱼ||xᵢ| + rowsum·E) + rowsum·E — the Lean lemma's
    # actual coefficients, evaluated at the measured operating profile) rather
    # than the uniform m·w·A layerBudget upper bound; likewise gains use the
    # exact row sum Σ|W| (what m·w' upper-bounds in lipOnWindow_dense).
    acts, bs_, gs = [], [], []

    def conv_bn_budget(x_act, W, stride, pad, ab, z, E):
        """conv (exact denseErr coefficients) then eval-BN affine, E threaded."""
        m = W.shape[1] * W.shape[2] * W.shape[3]
        sumabs = float(conv_np(np.abs(x_act), np.abs(W.astype(dt)),
                               stride, pad, dt).max())
        rowsum = float(np.abs(W.astype(dt)).sum(axis=(1, 2, 3)).max())
        Ec = ((1 + U32) ** (m + 2) - 1) * (sumabs + rowsum * E) + rowsum * E
        amax, bmax = mags(ab[0]), mags(ab[1])
        return affine_fresh(U32, amax, bmax, mags(z), Ec), amax * rowsum

    a = x.astype(dt)
    z = conv_np(a, Wstem, 2, 3, dt)
    zb = bn_apply(z, ab_stem)
    b_stem, g_stem = conv_bn_budget(a, Wstem, 2, 3, ab_stem, z, 0.0)
    h = np.maximum(zb, 0)
    acts.append(("stem", h))
    bs_.append(b_stem)
    gs.append(g_stem)
    h = maxpool2(h)
    acts.append(("maxpool", h))
    bs_.append(0.0)
    gs.append(1.0)
    for i, blk in enumerate(blocks):
        z1 = conv_np(h, blk["W1"], blk["stride"], 1, dt)
        y1 = np.maximum(bn_apply(z1, blk["ab1"]), 0)
        z2 = conv_np(y1, blk["W2"], 1, 1, dt)
        y2 = bn_apply(z2, blk["ab2"])
        # branch budget: conv1→bn1 (E=0), relu, conv2→bn2 (E threaded)
        E1, g1 = conv_bn_budget(h, blk["W1"], blk["stride"], 1,
                                blk["ab1"], z1, 0.0)
        E2, g2 = conv_bn_budget(y1, blk["W2"], 1, 1, blk["ab2"], z2, E1)
        if blk["down"]:
            zd = conv_np(h, blk["Wd"], blk["stride"], 0, dt)
            sk = bn_apply(zd, blk["abd"])
            Ed, gd = conv_bn_budget(h, blk["Wd"], blk["stride"], 0,
                                    blk["abd"], zd, 0.0)
        else:
            sk, Ed, gd = h, 0.0, 1.0
        out = y2 + sk
        h = np.maximum(out, 0)
        # skip-add rounding + relu exact
        bs_.append(E2 + Ed + U32 * (mags(out) + E2 + Ed))
        gs.append(g2 * g1 + gd)
        acts.append((f"block{i+1}" + ("↓" if blk["down"] else ""), h))
    hw = h.shape[2] * h.shape[3]
    gap = h.mean(axis=(2, 3))
    gb = U32 * (1 + U32) ** (hw + 1) * mags(h) \
        + ((1 + U32) ** (hw + 1) - 1) * mags(h)
    acts.append(("gap", gap))
    bs_.append(gb)
    gs.append(1.0)
    out = gap @ Whead.astype(dt) + bhead.astype(dt)
    acts.append(("dense (logits)", out))
    sumabs_h = float((np.abs(gap) @ np.abs(Whead.astype(dt))).max())
    bs_.append(((1 + U32) ** (512 + 2) - 1) * (sumabs_h + mags(bhead)))
    gs.append(float(np.abs(Whead.astype(dt)).sum(axis=0).max()))
    H_prov = [float(np.prod(gs[i + 1:])) for i in range(len(gs))]

    # ── the real deployed kernel: emit StableHLO, run on gfx1100 ───────────
    T = lambda *s: "tensor<" + "x".join(map(str, s)) + "xf32>"

    def econv(o, xn, Wn, ic, oc, k, s, p, spi, spo):
        src_t, w_t, out_t = T(B, ic, spi, spi), T(oc, ic, k, k), T(B, oc, spo, spo)
        return (f'    {o} = "stablehlo.convolution"({xn}, {Wn}) {{\n'
                f'        batch_group_count = 1 : i64,\n'
                f'        dimension_numbers = #stablehlo.conv<raw\n'
                f'          input_batch_dimension = 0, input_feature_dimension = 1,\n'
                f'          input_spatial_dimensions = [2, 3],\n'
                f'          kernel_output_feature_dimension = 0,\n'
                f'          kernel_input_feature_dimension = 1,\n'
                f'          kernel_spatial_dimensions = [2, 3],\n'
                f'          output_batch_dimension = 0, output_feature_dimension = 1,\n'
                f'          output_spatial_dimensions = [2, 3]>,\n'
                f'        feature_group_count = 1 : i64,\n'
                f'        padding = dense<[[{p}, {p}], [{p}, {p}]]> : tensor<2x2xi64>,\n'
                f'        rhs_dilation = array<i64: 1, 1>,\n'
                f'        window_strides = array<i64: {s}, {s}>\n'
                f'      }} : ({src_t}, {w_t}) -> {out_t}\n')

    def eaffine(tag, xn, an, bn_, c, sp):
        t4 = T(B, c, sp, sp)
        return (f"    %ab{tag} = stablehlo.broadcast_in_dim {an}, dims = [1] "
                f": ({T(c)}) -> {t4}\n"
                f"    %am{tag} = stablehlo.multiply {xn}, %ab{tag} : {t4}\n"
                f"    %bb{tag} = stablehlo.broadcast_in_dim {bn_}, dims = [1] "
                f": ({T(c)}) -> {t4}\n"
                f"    %aa{tag} = stablehlo.add %am{tag}, %bb{tag} : {t4}\n")

    def erelu(tag, xn, c, sp):
        t4 = T(B, c, sp, sp)
        return (f"    %rz{tag} = stablehlo.constant dense<0.0> : {t4}\n"
                f"    %rl{tag} = stablehlo.maximum {xn}, %rz{tag} : {t4}\n")

    def ewindow(o, xn, c, spi, wk, ws, red, init, tag):
        spo = (spi - wk) // ws + 1
        in_t, out_t = T(B, c, spi, spi), T(B, c, spo, spo)
        return (f"    %wi{tag} = stablehlo.constant dense<{init}> : tensor<f32>\n"
                f'    {o} = "stablehlo.reduce_window"({xn}, %wi{tag}) ({{\n'
                f"      ^bb0(%a: tensor<f32>, %bb: tensor<f32>):\n"
                f"        %m = stablehlo.{red} %a, %bb : tensor<f32>\n"
                f'        "stablehlo.return"(%m) : (tensor<f32>) -> ()\n'
                f"      }}) {{ window_dimensions = array<i64: 1, 1, {wk}, {wk}>,\n"
                f"           window_strides    = array<i64: 1, 1, {ws}, {ws}> }}\n"
                f"      : ({in_t}, tensor<f32>) -> {out_t}\n")

    args = [f"%x: {T(B, 3, 224, 224)}",
            f"%ws: {T(64, 3, 7, 7)}", f"%as: {T(64)}", f"%bs: {T(64)}"]
    params = [x, Wstem, ab_stem[0], ab_stem[1]]
    for i, blk in enumerate(blocks):
        ic, oc = blk["ic"], blk["oc"]
        args += [f"%w1_{i}: {T(oc, ic, 3, 3)}", f"%a1_{i}: {T(oc)}",
                 f"%b1_{i}: {T(oc)}",
                 f"%w2_{i}: {T(oc, oc, 3, 3)}", f"%a2_{i}: {T(oc)}",
                 f"%b2_{i}: {T(oc)}"]
        params += [blk["W1"], blk["ab1"][0], blk["ab1"][1],
                   blk["W2"], blk["ab2"][0], blk["ab2"][1]]
        if blk["down"]:
            args += [f"%wd_{i}: {T(oc, ic, 1, 1)}", f"%ad_{i}: {T(oc)}",
                     f"%bd_{i}: {T(oc)}"]
            params += [blk["Wd"], blk["abd"][0], blk["abd"][1]]
    args += [f"%wh: {T(512, 10)}", f"%bh: {T(10)}"]
    params += [Whead, bhead]

    body = [econv("%cs", "%x", "%ws", 3, 64, 7, 2, 3, 224, 112),
            eaffine("s", "%cs", "%as", "%bs", 64, 112),
            erelu("s", "%aas", 64, 112),
            ewindow("%mp", "%rls", 64, 112, 2, 2, "maximum",
                    "0xFF800000", "mp")]
    cur, sp = "%mp", 56
    for i, blk in enumerate(blocks):
        ic, oc, s = blk["ic"], blk["oc"], blk["stride"]
        spo = sp // s
        body.append(econv(f"%c1_{i}", cur, f"%w1_{i}", ic, oc, 3, s, 1, sp, spo))
        body.append(eaffine(f"x1_{i}", f"%c1_{i}", f"%a1_{i}", f"%b1_{i}",
                            oc, spo))
        body.append(erelu(f"x1_{i}", f"%aax1_{i}", oc, spo))
        body.append(econv(f"%c2_{i}", f"%rlx1_{i}", f"%w2_{i}", oc, oc, 3, 1,
                          1, spo, spo))
        body.append(eaffine(f"x2_{i}", f"%c2_{i}", f"%a2_{i}", f"%b2_{i}",
                            oc, spo))
        if blk["down"]:
            body.append(econv(f"%cd_{i}", cur, f"%wd_{i}", ic, oc, 1, s, 0,
                              sp, spo))
            body.append(eaffine(f"xd_{i}", f"%cd_{i}", f"%ad_{i}", f"%bd_{i}",
                                oc, spo))
            skip = f"%aaxd_{i}"
        else:
            skip = cur
        t4 = T(B, oc, spo, spo)
        body.append(f"    %sum{i} = stablehlo.add %aax2_{i}, {skip} : {t4}\n")
        body.append(erelu(f"o{i}", f"%sum{i}", oc, spo))
        cur, sp = f"%rlo{i}", spo
    body.append(ewindow("%gs", cur, 512, 7, 7, 7, "add", "0.0", "gap"))
    body.append(f"    %gc = stablehlo.constant dense<49.0> : {T(B, 512, 1, 1)}\n")
    body.append(f"    %gd = stablehlo.divide %gs, %gc : {T(B, 512, 1, 1)}\n")
    body.append(f"    %gf = stablehlo.reshape %gd : ({T(B, 512, 1, 1)}) -> "
                f"{T(B, 512)}\n")
    body.append(f"    %hd = stablehlo.dot_general %gf, %wh, "
                f"contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]"
                f" : ({T(B, 512)}, {T(512, 10)}) -> {T(B, 10)}\n")
    body.append(f"    %hbb = stablehlo.broadcast_in_dim %bh, dims = [1] : "
                f"({T(10)}) -> {T(B, 10)}\n")
    body.append(f"    %out = stablehlo.add %hd, %hbb : {T(B, 10)}\n")
    mlir = ("module {\n  func.func @main(\n      "
            + ",\n      ".join(args)
            + f"\n    ) -> {T(B, 10)} {{\n"
            + "".join(body)
            + f"    return %out : {T(B, 10)}\n  }}\n}}\n")

    out_g = run_iree(mlir, params).astype(np.float64)
    true_err = float(np.abs(out_g - out).max())

    # ── measured tail gains, f32 on the GPU (jax rocm) ─────────────────────
    def jx(a):
        return jnp.asarray(np.asarray(a, np.float32))

    def stage_fns():
        fns = []

        def jconv(a, W, s, p):
            return jax.lax.conv_general_dilated(
                a, W, (s, s), ((p, p), (p, p)),
                dimension_numbers=('NCHW', 'OIHW', 'NCHW'))

        def jaff(a, ab):
            return a * jx(ab[0])[None, :, None, None] \
                + jx(ab[1])[None, :, None, None]

        # stage 0 = stem (relu output), stage 1 = maxpool
        fns.append(lambda a: jnp.maximum(
            jaff(jconv(a, jx(Wstem), 2, 3), ab_stem), 0.0))
        fns.append(lambda a: a.reshape(a.shape[0], a.shape[1],
                                       a.shape[2] // 2, 2,
                                       a.shape[3] // 2, 2).max(axis=(3, 5)))
        for blk in blocks:
            def bf(a, blk=blk):
                y = jnp.maximum(jaff(jconv(a, jx(blk["W1"]), blk["stride"], 1),
                                     blk["ab1"]), 0.0)
                y = jaff(jconv(y, jx(blk["W2"]), 1, 1), blk["ab2"])
                sk = jaff(jconv(a, jx(blk["Wd"]), blk["stride"], 0),
                          blk["abd"]) if blk["down"] else a
                return jnp.maximum(y + sk, 0.0)
            fns.append(bf)
        fns.append(lambda a: a.mean(axis=(2, 3)))
        fns.append(lambda a: a @ jx(Whead) + jx(bhead))
        return fns

    gpu = jax.devices()[0]
    with jax.default_device(gpu):
        fns = stage_fns()
        H_meas = []
        for i in range(len(acts) - 1):
            aout = acts[i][1]
            gain = 0.0
            for smp in range(B):
                a1 = jx(aout[smp:smp + 1])

                def tail(aa, i=i):
                    for f in fns[i + 1:]:
                        aa = f(aa)
                    return aa[0]
                J = jax.jacrev(tail)(a1)
                gain = max(gain, float(jnp.abs(J.reshape(10, -1))
                                       .sum(axis=1).max()))
            H_meas.append(gain)
        H_meas.append(1.0)

    cb_meas = sum(H * b for H, b in zip(H_meas, bs_))
    cb_prov = sum(H * b for H, b in zip(H_prov, bs_))
    print(f"\n{'stage':<16}{'fresh b_i':>12}{'H_i meas':>11}{'H_i·b_i':>11}"
          f"{'H_i proven':>13}")
    for (name, _), b, Hm, Hp in zip(acts, bs_, H_meas, H_prov):
        print(f"{name:<16}{b:>12.3e}{Hm:>11.3e}{Hm * b:>11.3e}{Hp:>13.3e}")
    print(f"\n  true GPU logits drift : {true_err:.3e}")
    print(f"  chainBudget measured-H: {cb_meas:.3e}  "
          f"({cb_meas / true_err:.1e}× true)")
    print(f"  chainBudget proven-H  : {cb_prov:.3e}  "
          f"({cb_prov / true_err:.1e}× true)   [= the interval fold]")
    print(f"  logits magnitude      : {mags(out):.3e}")
    print("\n  eval-BN (running-stats affine, batch-calibrated) — the deployed")
    print("  forward; block = one chain layer, exactly the r34_floatBridges")
    print("  granularity. Gains measured in f32 on GPU (ample for O(1e3) gains).")


# ═══════════════════════════════════════════════════════════════════════════
# (6) EfficientNet-B0 — the COMMITTED AUDITED RENDER, not a twin
#     Runs verified_mlir/efficientnet_fwd_eval.mlir itself on gfx1100 (batch
#     32 @224²), arg values generated from its own parsed signature (He convs,
#     γ=1/β=0, running stats batch-calibrated). Chain = stem / 16 MBConv-SE
#     blocks / head conv / GAP / dense = 20 stages; per-block fresh budget =
#     the mini fold over expand→BN→swish→depthwise→BN→swish→SE→project→BN
#     (+skip), exact-coefficient conv budgets, esig = 2u32 for logistic.
# ═══════════════════════════════════════════════════════════════════════════
ESIG = 2 * U32          # logistic accuracy on gfx1100 (transcendental_probe)


def dwconv_np(x, W, stride, pad, dt):
    """depthwise NCHW conv, W: (C,1,kh,kw)."""
    x, W = x.astype(dt), W.astype(dt)
    N, C, H, Wd = x.shape
    kh, kw = W.shape[2], W.shape[3]
    Ho = (H + 2 * pad - kh) // stride + 1
    Wo = (Wd + 2 * pad - kw) // stride + 1
    xp = np.zeros((N, C, H + 2 * pad, Wd + 2 * pad), dt)
    xp[:, :, pad:pad + H, pad:pad + Wd] = x
    out = np.zeros((N, C, Ho, Wo), dt)
    for di in range(kh):
        for dj in range(kw):
            out += xp[:, :, di:di + stride * Ho:stride,
                      dj:dj + stride * Wo:stride] * W[None, :, 0, di, dj,
                                                      None, None]
    return out


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def enet_probe():
    import jax
    import jax.numpy as jnp
    import re
    import os

    print("\n" + "═" * 78)
    print("(6) EfficientNet-B0 — the COMMITTED render (fwd_eval), gfx1100")
    print("═" * 78)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlir = open(os.path.join(root,
                             "verified_mlir/efficientnet_fwd_eval.mlir")).read()
    header = mlir[:mlir.index("{", mlir.index("func.func"))]
    sig = [(m.group(1), tuple(int(d) for d in m.group(2).split("x")[:-1]))
           for m in re.finditer(r"%(\w+): tensor<([^>]+)>", header)]

    dt = np.float64
    mags = lambda a: float(np.abs(a).max())
    # blocks: (ic, mid, oc, r, k, stride) — strides at blocks 2/4/6/12
    # (verified against the render's stride-2 convs: b2dc/b4dc/b6dc/b12dc)
    cfg = [(32, 32, 16, 8, 3, 1), (16, 96, 24, 4, 3, 2), (24, 144, 24, 6, 3, 1),
           (24, 144, 40, 6, 5, 2), (40, 240, 40, 10, 5, 1),
           (40, 240, 80, 10, 3, 2), (80, 480, 80, 20, 3, 1),
           (80, 480, 80, 20, 3, 1), (80, 480, 112, 20, 5, 1),
           (112, 672, 112, 28, 5, 1), (112, 672, 112, 28, 5, 1),
           (112, 672, 192, 28, 5, 2), (192, 1152, 192, 48, 5, 1),
           (192, 1152, 192, 48, 5, 1), (192, 1152, 192, 48, 5, 1),
           (192, 1152, 320, 48, 3, 1)]

    # ── generate arg values by naming convention (stats calibrated below) ──
    vals = {}
    for name, shape in sig:
        if name == "x":
            vals[name] = rng.standard_normal((32, 150528)).astype(np.float32)
        elif name.endswith("nmu") or name.endswith("nvar"):
            vals[name] = None
        elif len(shape) == 4:
            fan = shape[1] * shape[2] * shape[3]
            vals[name] = (rng.standard_normal(shape)
                          * (2.0 / fan) ** 0.5).astype(np.float32)
        elif len(shape) == 2:
            vals[name] = (rng.standard_normal(shape)
                          * (2.0 / shape[0]) ** 0.5).astype(np.float32)
        elif name.endswith("g"):
            vals[name] = np.ones(shape, np.float32)
        elif name.endswith("bt"):
            vals[name] = np.zeros(shape, np.float32)
        else:
            vals[name] = (rng.standard_normal(shape) * 0.01).astype(np.float32)

    # BN tag → (γ, β) arg names; stats live at tag+"nmu"/"nvar"
    GAMMA = lambda t: {"st": "sg", "h": "hg"}.get(t, t + "g")
    BETA = lambda t: {"st": "sbt", "h": "hbt"}.get(t, t + "bt")

    def bn_eval(z, tag):
        mu = vals[tag + "nmu"].astype(z.dtype)
        var = vals[tag + "nvar"].astype(z.dtype)
        g = vals[GAMMA(tag)].astype(z.dtype)
        bt = vals[BETA(tag)].astype(z.dtype)
        istd = 1.0 / np.sqrt(var + 1e-5)
        return (z - mu[None, :, None, None]) * (istd * g)[None, :, None, None] \
            + bt[None, :, None, None]

    def calib(tag, z):
        if vals[tag + "nmu"] is None:
            vals[tag + "nmu"] = z.mean(axis=(0, 2, 3)).astype(np.float32)
            vals[tag + "nvar"] = z.var(axis=(0, 2, 3)).astype(np.float32)
        return bn_eval(z, tag)

    def swish(x):
        return x * sigmoid_np(x)

    def bn_gain(tag):
        g = vals[GAMMA(tag)].astype(dt)
        istd = 1.0 / np.sqrt(vals[tag + "nvar"].astype(dt) + 1e-5)
        return float(np.abs(g * istd).max())

    def bn_fresh(a_eff, A, E):
        """eval-BN (sub, mul istd, mul γ, add β = rounded elementwise chain)"""
        return 5 * U32 * (a_eff * (A + E) + A) + a_eff * E

    def swish_fold(A, E):
        return 1.1 * E + (ESIG + 2 * U32) * A

    def conv_budget(x_act, Wname, bname, stride, pad, E, depthwise=False):
        W = vals[Wname]
        Wd_ = np.abs(W.astype(dt))
        if depthwise:
            sumabs = float(dwconv_np(np.abs(x_act), Wd_, stride, pad,
                                     dt).max())
            m = W.shape[2] * W.shape[3]
        else:
            sumabs = float(conv_np(np.abs(x_act), Wd_, stride, pad, dt).max())
            m = W.shape[1] * W.shape[2] * W.shape[3]
        rowsum = float(Wd_.sum(axis=(1, 2, 3)).max())
        b = ((1 + U32) ** (m + 2) - 1) * (sumabs + mags(vals[bname])
                                          + rowsum * E) + rowsum * E
        return b, rowsum

    # ── the f64 oracle, run twice (pass 1 calibrates; pass 2 = reference +
    #    inline per-stage budgets/gains at the frozen f32 stats) ────────────
    def oracle(with_budgets):
        acts, names, bs_, gs = [], [], [], []
        x4 = vals["x"].astype(dt).reshape(32, 3, 224, 224)
        z = conv_np(x4, vals["sW"], 2, 1, dt) \
            + vals["sb"].astype(dt)[None, :, None, None]
        zb = calib("st", z)
        h = swish(zb)
        acts.append(h)
        names.append("stem")
        if with_budgets:
            Ec, rs = conv_budget(x4, "sW", "sb", 2, 1, 0.0)
            a_eff = bn_gain("st")
            bs_.append(swish_fold(mags(zb), bn_fresh(a_eff, mags(zb), Ec)))
            gs.append(1.1 * a_eff * rs)
        for i, (ic, mid, oc, r, k, s) in enumerate(cfg, start=1):
            p = f"b{i}"
            inp = h
            E, g_blk = 0.0, 1.0
            if mid != ic:
                ze = conv_np(h, vals[p + "eW"], 1, 0, dt) \
                    + vals[p + "eb"].astype(dt)[None, :, None, None]
                zeb = calib(p + "e", ze)
                if with_budgets:
                    Ec, rs = conv_budget(h, p + "eW", p + "eb", 1, 0, E)
                    a_eff = bn_gain(p + "e")
                    E = swish_fold(mags(zeb), bn_fresh(a_eff, mags(zeb), Ec))
                    g_blk *= 1.1 * a_eff * rs
                h = swish(zeb)
                del ze, zeb
            zd = dwconv_np(h, vals[p + "dW"], s, (k - 1) // 2, dt) \
                + vals[p + "db"].astype(dt)[None, :, None, None]
            zdb = calib(p + "d", zd)
            if with_budgets:
                Ec, rs = conv_budget(h, p + "dW", p + "db", s, (k - 1) // 2,
                                     E, depthwise=True)
                a_eff = bn_gain(p + "d")
                E = swish_fold(mags(zdb), bn_fresh(a_eff, mags(zdb), Ec))
                g_blk *= 1.1 * a_eff * rs
            ds = swish(zdb)
            del zd, zdb, h
            sq = ds.mean(axis=(2, 3))
            d1 = sq @ vals[p + "zW1"].astype(dt) + vals[p + "zb1"].astype(dt)
            a1 = swish(d1)
            d2 = a1 @ vals[p + "zW2"].astype(dt) + vals[p + "zb2"].astype(dt)
            gate = sigmoid_np(d2)
            se = ds * gate[:, :, None, None]
            if with_budgets:
                A_ds = mags(ds)
                rs1 = float(np.abs(vals[p + "zW1"].astype(dt))
                            .sum(axis=0).max())
                rs2 = float(np.abs(vals[p + "zW2"].astype(dt))
                            .sum(axis=0).max())
                hw = ds.shape[2] * ds.shape[3]
                E_pool = E + ((1 + U32) ** (hw + 1) - 1) * A_ds
                E_d1 = ((1 + U32) ** (mid + 2) - 1) * (
                    float(np.abs(sq).max()) * rs1
                    + mags(vals[p + "zb1"])) + rs1 * E_pool
                E_sw = swish_fold(mags(d1), E_d1)
                E_d2 = ((1 + U32) ** (r + 2) - 1) * (
                    float(np.abs(a1).max()) * rs2
                    + mags(vals[p + "zb2"])) + rs2 * E_sw
                E_gate = 0.25 * E_d2 + ESIG
                E = float(gate.max()) * E + A_ds * E_gate + 2 * U32 * A_ds
                g_blk *= float(gate.max()) + A_ds * 0.25 * rs2 * 1.1 * rs1
            zp = conv_np(se, vals[p + "pW"], 1, 0, dt) \
                + vals[p + "pb"].astype(dt)[None, :, None, None]
            h2 = calib(p + "p", zp)
            if with_budgets:
                Ec, rs = conv_budget(se, p + "pW", p + "pb", 1, 0, E)
                a_eff = bn_gain(p + "p")
                E = bn_fresh(a_eff, mags(h2), Ec)
                g_blk *= a_eff * rs
            del ds, se, zp
            if ic == oc and s == 1:
                h2 = h2 + inp
                if with_budgets:
                    E = E + U32 * mags(h2)
                    g_blk += 1.0
            h = h2
            del inp
            acts.append(h)
            names.append(f"block{i}" + ("↓" if s == 2 else ""))
            if with_budgets:
                bs_.append(E)
                gs.append(g_blk)
        zh = conv_np(h, vals["hW"], 1, 0, dt) \
            + vals["hb"].astype(dt)[None, :, None, None]
        zhb = calib("h", zh)
        if with_budgets:
            Ec, rs = conv_budget(h, "hW", "hb", 1, 0, 0.0)
            a_eff = bn_gain("h")
            bs_.append(swish_fold(mags(zhb), bn_fresh(a_eff, mags(zhb), Ec)))
            gs.append(1.1 * a_eff * rs)
        h = swish(zhb)
        acts.append(h)
        names.append("head conv")
        g = h.mean(axis=(2, 3))
        acts.append(g)
        names.append("gap")
        if with_budgets:
            A_h = mags(h)
            bs_.append(U32 * (1 + U32) ** 50 * A_h
                       + ((1 + U32) ** 50 - 1) * A_h)
            gs.append(1.0)
        out = g @ vals["Wd"].astype(dt) + vals["bd"].astype(dt)
        acts.append(out)
        names.append("dense (logits)")
        if with_budgets:
            sumabs_h = float((np.abs(g) @ np.abs(vals["Wd"].astype(dt))).max())
            bs_.append(((1 + U32) ** (1280 + 2) - 1)
                       * (sumabs_h + mags(vals["bd"])))
            gs.append(float(np.abs(vals["Wd"].astype(dt)).sum(axis=0).max()))
        return acts, names, bs_, gs

    oracle(False)                               # pass 1: calibrate stats
    acts, names, bs_, gs = oracle(True)         # pass 2: reference + budgets
    out = acts[-1]
    H_prov = [float(np.prod(gs[i + 1:])) for i in range(len(gs))]

    # ── the committed render, as-is, on gfx1100 ────────────────────────────
    out_g = run_iree(mlir, [vals[n] for n, _ in sig],
                     fn="efficientnet_fwd_eval",
                     module_name="m").astype(np.float64)
    true_err = float(np.abs(out_g - out).max())

    # ── measured tail gains (f32, per-sample — eval BN ⇒ no batch coupling)
    def jx(a):
        return jnp.asarray(np.asarray(a, np.float32))

    def jbn(z, tag):
        istd = 1.0 / jnp.sqrt(jx(vals[tag + "nvar"]) + 1e-5)
        return (z - jx(vals[tag + "nmu"])[None, :, None, None]) \
            * (istd * jx(vals[GAMMA(tag)]))[None, :, None, None] \
            + jx(vals[BETA(tag)])[None, :, None, None]

    def jswish(x):
        return x * jax.nn.sigmoid(x)

    def jconv(a, W, s, p, groups=1):
        return jax.lax.conv_general_dilated(
            a, W, (s, s), ((p, p), (p, p)),
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
            feature_group_count=groups)

    def stage_fns():
        fns = []
        for i, (ic, mid, oc, r, k, s) in enumerate(cfg, start=1):
            p = f"b{i}"

            def bf(a, p=p, ic=ic, mid=mid, oc=oc, k=k, s=s):
                inp = a
                if mid != ic:
                    a = jswish(jbn(jconv(a, jx(vals[p + "eW"]), 1, 0)
                                   + jx(vals[p + "eb"])[None, :, None, None],
                                   p + "e"))
                a = jswish(jbn(jconv(a, jx(vals[p + "dW"]), s, (k - 1) // 2,
                                     groups=mid)
                               + jx(vals[p + "db"])[None, :, None, None],
                               p + "d"))
                sq = a.mean(axis=(2, 3))
                a1 = jswish(sq @ jx(vals[p + "zW1"]) + jx(vals[p + "zb1"]))
                gate = jax.nn.sigmoid(a1 @ jx(vals[p + "zW2"])
                                      + jx(vals[p + "zb2"]))
                a = a * gate[:, :, None, None]
                a = jbn(jconv(a, jx(vals[p + "pW"]), 1, 0)
                        + jx(vals[p + "pb"])[None, :, None, None], p + "p")
                return a + inp if (ic == oc and s == 1) else a
            fns.append(bf)
        fns.append(lambda a: jswish(jbn(
            jconv(a, jx(vals["hW"]), 1, 0)
            + jx(vals["hb"])[None, :, None, None], "h")))
        fns.append(lambda a: a.mean(axis=(2, 3)))
        fns.append(lambda a: a @ jx(vals["Wd"]) + jx(vals["bd"]))
        return fns

    fns = stage_fns()       # fns[i] is the stage AFTER acts[i]'s output
    H_meas = []
    for i in range(len(acts) - 1):
        aout = acts[i]
        gain = 0.0
        for smp in range(2):
            a1 = jx(aout[smp:smp + 1])

            def tail(aa, i=i):
                for f in fns[i:]:
                    aa = f(aa)
                return aa[0]
            J = jax.jacrev(tail)(a1)
            gain = max(gain, float(jnp.abs(J.reshape(10, -1))
                                   .sum(axis=1).max()))
        H_meas.append(gain)
    H_meas.append(1.0)

    cb_meas = sum(H * b for H, b in zip(H_meas, bs_))
    cb_prov = sum(H * b for H, b in zip(H_prov, bs_))
    print(f"\n{'stage':<16}{'fresh b_i':>12}{'H_i meas':>11}{'H_i·b_i':>11}"
          f"{'H_i proven':>13}")
    for name, b, Hm, Hp in zip(names, bs_, H_meas, H_prov):
        print(f"{name:<16}{b:>12.3e}{Hm:>11.3e}{Hm * b:>11.3e}{Hp:>13.3e}")
    print(f"\n  true GPU logits drift : {true_err:.3e}")
    print(f"  chainBudget measured-H: {cb_meas:.3e}  "
          f"({cb_meas / true_err:.1e}× true)")
    print(f"  chainBudget proven-H  : {cb_prov:.3e}  "
          f"({cb_prov / true_err:.1e}× true)   [= the interval fold]")
    print(f"  logits magnitude      : {mags(out):.3e}")
    print("\n  THE COMMITTED RENDER (verified_mlir/efficientnet_fwd_eval.mlir)")
    print("  run as-is on gfx1100; oracle mirrors it op-for-op in f64.")


# ═══════════════════════════════════════════════════════════════════════════
# (7) ConvNeXt-T — the COMMITTED RENDER (verified_mlir/convnext_fwd.mlir)
#     run as-is on gfx1100 (batch 32 @224²; args from its parsed signature at
#     the repo's own init: He convs, γ/layerScale = ones, β/biases = zeros).
#     LayerNorm here is the committed SCALAR-LN: per-sample over the WHOLE
#     C·H·W extent (n up to 301056!) with scalar γ/β, ε = 1e-6 — LN budgets
#     from the same BnFloatBridge parts as §4 (mean/var/istd/norm) at the
#     a-posteriori floor. GELU is the render's tanh form; fresh = egelu=8u32,
#     gain = the floatClose_gelu magnitude-poly modulus (transcribed as-is).
#     Chain = patchify / 18 blocks / 3 downsamples / GAP / head-LN / dense.
# ═══════════════════════════════════════════════════════════════════════════
EGELU = 8 * U32         # gelu accuracy on gfx1100 (transcendental_probe)


def gelu_tanh_np(x):
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_gain(A):
    """PROVEN gelu input-shift gain: min of floatClose_gelu's magnitude-poly
    modulus (ViTFloatBridge) and the flat saturation-aware 3/2 bound
    (Proofs.geluScalar_lipschitz, GeluLipschitz.lean) — both proven faces."""
    poly = 1.0 + np.sqrt(2.0 / np.pi) / 2.0 * A * (1.0 + 3 * 0.044715 * A * A)
    return min(poly, 1.5)


def convnext_probe():
    import jax
    import jax.numpy as jnp
    import re
    import os

    print("\n" + "═" * 78)
    print("(7) ConvNeXt-T — the COMMITTED render (convnext_fwd), gfx1100")
    print("═" * 78)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlir = open(os.path.join(root,
                             "verified_mlir/convnext_fwd.mlir")).read()
    header = mlir[:mlir.index("{", mlir.index("func.func"))]
    sig = [(m.group(1), tuple(int(d) for d in m.group(2).split("x")[:-1]))
           for m in re.finditer(r"%(\w+): tensor<([^>]+)>", header)]

    dt = np.float64
    mags = lambda a: float(np.abs(a).max())
    EPS_LN = 1e-6
    stages_cfg = [(96, 3), (192, 3), (384, 9), (768, 3)]   # (c, nblocks)

    # ── args at the repo's own init (kind 0/1/2 = He / ones / zeros) ───────
    vals = {}
    for name, shape in sig:
        if name == "x":
            vals[name] = rng.standard_normal((32, 150528)).astype(np.float32)
        elif len(shape) == 4:
            fan = shape[1] * shape[2] * shape[3]
            vals[name] = (rng.standard_normal(shape)
                          * (2.0 / fan) ** 0.5).astype(np.float32)
        elif len(shape) == 2:
            vals[name] = (rng.standard_normal(shape)
                          * (2.0 / shape[0]) ** 0.5).astype(np.float32)
        elif name.endswith("g"):                 # LN γ (scalar) / layerScale
            vals[name] = np.ones(shape if shape else (), np.float32)
        else:                                    # LN β / biases
            vals[name] = np.zeros(shape if shape else (), np.float32)

    def ln_scalar(z, gname, btname):
        """per-sample LN over the WHOLE flattened extent, scalar γ/β."""
        N = z.shape[0]
        flat = z.reshape(N, -1)
        mu = flat.mean(axis=1, keepdims=True)
        var = ((flat - mu) ** 2).mean(axis=1, keepdims=True)
        y = (flat - mu) / np.sqrt(var + EPS_LN) \
            * float(vals[gname]) + float(vals[btname])
        return y.reshape(z.shape), flat - mu, var

    def ln_budget_gain(z, E):
        """scalar-LN fresh budget + input-shift gain at the operating
        profile (BnFloatBridge parts; G=|γ|=1, B=|β|=0 at init)."""
        N = z.shape[0]
        n = z.reshape(N, -1).shape[1]
        flat = z.reshape(N, -1)
        mu = flat.mean(axis=1, keepdims=True)
        var = ((flat - mu) ** 2).mean(axis=1)
        D = float(np.abs(flat - mu).max())
        floor = float(var.min()) + EPS_LN
        S = 1.0 / np.sqrt(floor)
        A = mags(flat)
        emean = bn_mean_budget(U32, n, A)
        evar = bn_var_budget(U32, D, emean, n)
        eistd = bn_istd_budget(ERS, evar, floor)
        fresh = bn_norm_budget(U32, D, S, 1.0, 0.0, emean, eistd)
        gain = 1.0 * (2 * S + 2 * D * D / floor ** 1.5)
        return gain * E + fresh, gain

    def conv_budget(x_act, Wname, bname, stride, pad, E, depthwise=False):
        W = vals[Wname]
        Wd_ = np.abs(W.astype(dt))
        if depthwise:
            sumabs = float(dwconv_np(np.abs(x_act), Wd_, stride, pad,
                                     dt).max())
            m = W.shape[2] * W.shape[3]
        else:
            sumabs = float(conv_np(np.abs(x_act), Wd_, stride, pad, dt).max())
            m = W.shape[1] * W.shape[2] * W.shape[3]
        rowsum = float(Wd_.sum(axis=(1, 2, 3)).max())
        b = ((1 + U32) ** (m + 2) - 1) * (sumabs + mags(vals[bname])
                                          + rowsum * E) + rowsum * E
        return b, rowsum

    # ── the f64 oracle with inline budgets (mirrors the render op-for-op) ──
    def oracle():
        acts, names, bs_, gs = [], [], [], []
        x4 = vals["x"].astype(dt).reshape(32, 3, 224, 224)
        h = conv_np(x4, vals["psW"], 4, 0, dt) \
            + vals["psb"].astype(dt)[None, :, None, None]
        Ec, rs = conv_budget(x4, "psW", "psb", 4, 0, 0.0)
        acts.append(h)
        names.append("patchify")
        bs_.append(Ec)
        gs.append(rs)
        for si, (c, nb) in enumerate(stages_cfg):
            if si > 0:
                d = f"d{si - 1}"
                zn, _, _ = ln_scalar(h, d + "ng", d + "nbt")
                E, g_ln = ln_budget_gain(h, 0.0)
                h2 = conv_np(zn, vals[d + "W"], 2, 0, dt) \
                    + vals[d + "b"].astype(dt)[None, :, None, None]
                Ec, rs = conv_budget(zn, d + "W", d + "b", 2, 0, E)
                h = h2
                acts.append(h)
                names.append(f"downsample{si - 1}")
                bs_.append(Ec)
                gs.append(rs * g_ln)
            for bi in range(nb):
                p = f"s{si}b{bi}"
                inp = h
                zd = dwconv_np(h, vals[p + "dW"], 1, 3, dt) \
                    + vals[p + "db"].astype(dt)[None, :, None, None]
                E, rs_d = conv_budget(h, p + "dW", p + "db", 1, 3, 0.0,
                                      depthwise=True)
                g_blk = rs_d
                zn, _, _ = ln_scalar(zd, p + "ng", p + "nbt")
                E, g_ln = ln_budget_gain(zd, E)
                g_blk *= g_ln
                del zd
                ze = conv_np(zn, vals[p + "eW"], 1, 0, dt) \
                    + vals[p + "eb"].astype(dt)[None, :, None, None]
                Ec, rs = conv_budget(zn, p + "eW", p + "eb", 1, 0, E)
                E = Ec
                g_blk *= rs
                del zn
                A_e = mags(ze)
                zg = gelu_tanh_np(ze)
                E = gelu_gain(A_e) * E + EGELU
                g_blk *= gelu_gain(A_e)
                del ze
                zp = conv_np(zg, vals[p + "pW"], 1, 0, dt) \
                    + vals[p + "pb"].astype(dt)[None, :, None, None]
                Ec, rs = conv_budget(zg, p + "pW", p + "pb", 1, 0, E)
                E = Ec
                g_blk *= rs
                del zg
                ls_max = mags(vals[p + "lg"])
                out_b = zp * vals[p + "lg"].astype(dt)[None, :, None, None]
                E = ls_max * E + U32 * mags(out_b)
                g_blk *= ls_max
                h = out_b + inp
                E = E + U32 * mags(h)
                g_blk += 1.0
                del zp, out_b, inp
                acts.append(h)
                names.append(p)
                bs_.append(E)
                gs.append(g_blk)
        g = h.mean(axis=(2, 3))
        A_h = mags(h)
        acts.append(g)
        names.append("gap")
        bs_.append(U32 * (1 + U32) ** 50 * A_h + ((1 + U32) ** 50 - 1) * A_h)
        gs.append(1.0)
        zn, _, _ = ln_scalar(g[:, :, None, None], "hng", "hnbt")
        E, g_ln = ln_budget_gain(g[:, :, None, None], 0.0)
        g2 = zn.reshape(g.shape)
        acts.append(g2)
        names.append("head LN")
        bs_.append(E)
        gs.append(g_ln)
        out = g2 @ vals["Wd"].astype(dt) + vals["bd"].astype(dt)
        acts.append(out)
        names.append("dense (logits)")
        sumabs_h = float((np.abs(g2) @ np.abs(vals["Wd"].astype(dt))).max())
        bs_.append(((1 + U32) ** (768 + 2) - 1)
                   * (sumabs_h + mags(vals["bd"])))
        gs.append(float(np.abs(vals["Wd"].astype(dt)).sum(axis=0).max()))
        return acts, names, bs_, gs

    acts, names, bs_, gs = oracle()
    out = acts[-1]
    H_prov = [float(np.prod(gs[i + 1:])) for i in range(len(gs))]

    out_g = run_iree(mlir, [vals[n] for n, _ in sig],
                     fn="convnext_fwd", module_name="m").astype(np.float64)
    true_err = float(np.abs(out_g - out).max())

    # ── measured tail gains (f32; per-sample LN ⇒ no batch coupling) ───────
    def jx(a):
        return jnp.asarray(np.asarray(a, np.float32))

    def jln(z, gname, btname):
        N = z.shape[0]
        flat = z.reshape(N, -1)
        mu = flat.mean(axis=1, keepdims=True)
        var = ((flat - mu) ** 2).mean(axis=1, keepdims=True)
        y = (flat - mu) / jnp.sqrt(var + EPS_LN) \
            * float(vals[gname]) + float(vals[btname])
        return y.reshape(z.shape)

    def jgelu(x):
        inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + jnp.tanh(inner))

    def jconv(a, W, s, p, groups=1):
        return jax.lax.conv_general_dilated(
            a, W, (s, s), ((p, p), (p, p)),
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
            feature_group_count=groups)

    def stage_fns():
        fns = []
        for si, (c, nb) in enumerate(stages_cfg):
            if si > 0:
                d = f"d{si - 1}"

                def df(a, d=d):
                    return jconv(jln(a, d + "ng", d + "nbt"),
                                 jx(vals[d + "W"]), 2, 0) \
                        + jx(vals[d + "b"])[None, :, None, None]
                fns.append(df)
            for bi in range(nb):
                p = f"s{si}b{bi}"

                def bf(a, p=p, c=c):
                    inp = a
                    a = jconv(a, jx(vals[p + "dW"]), 1, 3, groups=c) \
                        + jx(vals[p + "db"])[None, :, None, None]
                    a = jln(a, p + "ng", p + "nbt")
                    a = jconv(a, jx(vals[p + "eW"]), 1, 0) \
                        + jx(vals[p + "eb"])[None, :, None, None]
                    a = jgelu(a)
                    a = jconv(a, jx(vals[p + "pW"]), 1, 0) \
                        + jx(vals[p + "pb"])[None, :, None, None]
                    a = a * jx(vals[p + "lg"])[None, :, None, None]
                    return a + inp
                fns.append(bf)
        fns.append(lambda a: a.mean(axis=(2, 3)))
        fns.append(lambda a: jln(a[:, :, None, None], "hng",
                                 "hnbt").reshape(a.shape[0], -1))
        fns.append(lambda a: a @ jx(vals["Wd"]) + jx(vals["bd"]))
        return fns

    fns = stage_fns()
    H_meas = []
    for i in range(len(acts) - 1):
        aout = acts[i]
        gain = 0.0
        for smp in range(2):
            a1 = jx(aout[smp:smp + 1])

            def tail(aa, i=i):
                for f in fns[i:]:
                    aa = f(aa)
                return aa[0]
            J = jax.jacrev(tail)(a1)
            gain = max(gain, float(jnp.abs(J.reshape(10, -1))
                                   .sum(axis=1).max()))
        H_meas.append(gain)
    H_meas.append(1.0)

    cb_meas = sum(H * b for H, b in zip(H_meas, bs_))
    cb_prov = sum(H * b for H, b in zip(H_prov, bs_))
    print(f"\n{'stage':<16}{'fresh b_i':>12}{'H_i meas':>11}{'H_i·b_i':>11}"
          f"{'H_i proven':>13}")
    for name, b, Hm, Hp in zip(names, bs_, H_meas, H_prov):
        print(f"{name:<16}{b:>12.3e}{Hm:>11.3e}{Hm * b:>11.3e}{Hp:>13.3e}")
    print(f"\n  true GPU logits drift : {true_err:.3e}")
    print(f"  chainBudget measured-H: {cb_meas:.3e}  "
          f"({cb_meas / true_err:.1e}× true)")
    print(f"  chainBudget proven-H  : {cb_prov:.3e}  "
          f"({cb_prov / true_err:.1e}× true)   [= the interval fold]")
    print(f"  logits magnitude      : {mags(out):.3e}")
    print("\n  THE COMMITTED RENDER (verified_mlir/convnext_fwd.mlir) run")
    print("  as-is on gfx1100; scalar-LN over the whole C·H·W extent per")
    print("  sample (n up to 301056) — its Higham mean/var budgets at that n")
    print("  are the LN-analogue of the SE finding to watch in the table.")


# ═══════════════════════════════════════════════════════════════════════════
# (8) ViT-Tiny — the COMMITTED RENDER (verified_mlir/vit_fwd.mlir) as-is on
#     gfx1100 (batch 32 @224², args from its parsed signature at the repo
#     init: He denses, LN γ=ones/β=zeros, biases/CLS/pos = zeros). The
#     architecturally FRIENDLY case for the adjoint chain: per-token LN
#     (n=192 — no big-n Higham face), no BN/SE, gelu at the proven 3/2 gain,
#     softmax fresh budgets via the smRho/smKappa pieces and input-shift via
#     the e^{2δ}−1 modulus evaluated AT the small inherited δ (expm1 — the
#     modulus needs no window inside the fold); the PROVEN suffix face uses
#     lipOnWindow_softmax's window gain (e^{4A_s}−1)/(2A_s) at the measured
#     score magnitude. Chain = patch / 12 blocks / final LN / head = 15.
# ═══════════════════════════════════════════════════════════════════════════
def vit_probe():
    import jax
    import jax.numpy as jnp
    import re
    import os

    print("\n" + "═" * 78)
    print("(8) ViT-Tiny — the COMMITTED render (vit_fwd), gfx1100")
    print("═" * 78)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlir = open(os.path.join(root, "verified_mlir/vit_fwd.mlir")).read()
    header = mlir[:mlir.index("{", mlir.index("func.func"))]
    sig = [(m.group(1), tuple(int(d) for d in m.group(2).split("x")[:-1]))
           for m in re.finditer(r"%(\w+): tensor<([^>]+)>", header)]

    dt = np.float64
    mags = lambda a: float(np.abs(a).max())
    D, NH, DH, NT = 192, 3, 64, 197
    EPS_LN = 1e-5
    EEXP = 2 * U32

    # ── args at the repo init (He / ones / zeros per kind conventions) ─────
    vals = {}
    for name, shape in sig:
        if name == "x":
            vals[name] = rng.standard_normal((32, 150528)).astype(np.float32)
        elif len(shape) == 4:                       # patch conv: He fan-in 768
            fan = shape[1] * shape[2] * shape[3]
            vals[name] = (rng.standard_normal(shape)
                          * (2.0 / fan) ** 0.5).astype(np.float32)
        elif len(shape) == 2 and name != "pos":     # dense W: He fan-in
            vals[name] = (rng.standard_normal(shape)
                          * (2.0 / shape[0]) ** 0.5).astype(np.float32)
        elif name.endswith("g1") or name.endswith("g2") or name == "gF":
            vals[name] = np.ones(shape, np.float32)
        else:                                       # biases / β / cls / pos
            vals[name] = np.zeros(shape, np.float32)

    def ln_tok(z, gname, btname):
        """per-token LN over the 192 features + per-channel affine."""
        mu = z.mean(axis=-1, keepdims=True)
        var = ((z - mu) ** 2).mean(axis=-1, keepdims=True)
        xh = (z - mu) / np.sqrt(var + EPS_LN)
        return xh * vals[gname].astype(z.dtype) + vals[btname].astype(z.dtype)

    def ln_budget_gain(z, gname, E):
        """per-token-LN budgets (BnFloatBridge parts, n=192): PER-TOKEN
        bookkeeping — each token's deviation D_t paired with ITS OWN variance
        floor (the perRowFlat/bnPerChannel granularity the Lean LN lemmas
        actually have). Max-over-tokens of the per-token budget/gain; a
        zero-variance token (the zero-init CLS at block 0) then contributes
        a large GAIN but a ~zero FRESH budget instead of poisoning both."""
        mu = z.mean(axis=-1, keepdims=True)
        dev = z - mu
        var = (dev ** 2).mean(axis=-1)                    # (B, T)
        D_t = np.abs(dev).max(axis=-1)                    # (B, T)
        A_t = np.abs(z).max(axis=-1)
        floor_t = var + EPS_LN
        S_t = 1.0 / np.sqrt(floor_t)
        G = mags(vals[gname])
        emean_t = bn_mean_budget(U32, D, A_t)
        evar_t = bn_var_budget(U32, D_t, emean_t, D)
        eistd_t = bn_istd_budget(ERS, evar_t, floor_t)
        fresh = float(bn_norm_budget(U32, D_t, S_t, G, 0.0,
                                     emean_t, eistd_t).max())
        gain = float((G * (2 * S_t + 2 * D_t * D_t / floor_t ** 1.5)).max())
        return gain * E + fresh, gain

    def dense_budget(h2d, Wname, bname, E):
        """per-token dense: exact denseErr coefficients at the profile."""
        W = np.abs(vals[Wname].astype(dt))
        m = W.shape[0]
        sumabs = float((np.abs(h2d) @ W).max())
        rowsum = float(W.sum(axis=0).max())
        return ((1 + U32) ** (m + 2) - 1) * (sumabs + mags(vals[bname])
                                             + rowsum * E) + rowsum * E, rowsum

    # softmax float rounding (smRho/smKappa/smErr pieces, n = 197 tokens)
    rho = ((1 + U32) ** (NT + 1) - 1) * (1 + EEXP) + EEXP
    kappa = (EEXP + rho) / (1 - rho)
    SM_FRESH = U32 * (1 + kappa) + kappa

    def softmax_np(s):
        e = np.exp(s)
        return e / e.sum(axis=-1, keepdims=True)

    # ── oracle with inline budgets (mirrors the render op-for-op) ──────────
    def oracle():
        acts, names, bs_, gs = [], [], [], []
        x4 = vals["x"].astype(dt).reshape(32, 3, 224, 224)
        p = conv_np(x4, vals["wConv"], 16, 0, dt) \
            + vals["bConv"].astype(dt)[None, :, None, None]
        tok = p.transpose(0, 2, 3, 1).reshape(32, 196, D)
        h = np.concatenate(
            [np.broadcast_to(vals["cls"].astype(dt), (32, 1, D)), tok], axis=1)
        h = h + vals["pos"].astype(dt)[None]
        Wp = np.abs(vals["wConv"].astype(dt))
        sumabs_p = float(conv_np(np.abs(x4), Wp, 16, 0, dt).max())
        b_patch = ((1 + U32) ** (768 + 2) - 1) * (sumabs_p
                                                  + mags(vals["bConv"])) \
            + U32 * mags(h)                                # + pos-add rounding
        acts.append(h)
        names.append("patch+cls+pos")
        bs_.append(b_patch)
        gs.append(float(Wp.sum(axis=(1, 2, 3)).max()))
        for bi in range(12):
            p_ = f"b{bi}_"
            inp = h
            ln1 = ln_tok(h, p_ + "g1", p_ + "bt1")
            E_ln1, g_ln1 = ln_budget_gain(h, p_ + "g1", 0.0)
            ln1f = ln1.reshape(-1, D)
            q = (ln1f @ vals[p_ + "Wq"].astype(dt)
                 + vals[p_ + "bq"].astype(dt)).reshape(32, NT, D)
            k = (ln1f @ vals[p_ + "Wk"].astype(dt)
                 + vals[p_ + "bk"].astype(dt)).reshape(32, NT, D)
            v = (ln1f @ vals[p_ + "Wv"].astype(dt)
                 + vals[p_ + "bv"].astype(dt)).reshape(32, NT, D)
            E_q, rs_q = dense_budget(ln1f, p_ + "Wq", p_ + "bq", E_ln1)
            E_k, rs_k = dense_budget(ln1f, p_ + "Wk", p_ + "bk", E_ln1)
            E_v, rs_v = dense_budget(ln1f, p_ + "Wv", p_ + "bv", E_ln1)
            att = np.zeros_like(v)
            E_cat, A_s_max, g_score_path = 0.0, 0.0, 0.0
            for hd in range(NH):
                sl = slice(hd * DH, (hd + 1) * DH)
                qh, kh, vh = q[:, :, sl], k[:, :, sl], v[:, :, sl]
                s = np.einsum('btd,bsd->bts', qh, kh) * 0.125
                al = softmax_np(s)
                att[:, :, sl] = np.einsum('bts,bsd->btd', al, vh)
                A_q, A_k, A_v = mags(qh), mags(kh), mags(vh)
                A_s = mags(s)
                A_s_max = max(A_s_max, A_s)
                Eqk = max(E_q, E_k)
                sumabs_s = float(np.einsum('btd,bsd->bts', np.abs(qh),
                                           np.abs(kh)).max()) * 0.125
                E_s = ((1 + U32) ** (DH + 2) - 1) * sumabs_s \
                    + 0.125 * DH * ((A_q + Eqk) * Eqk + A_k * Eqk) + U32 * A_s
                E_al = float(np.expm1(2 * E_s)) + SM_FRESH
                sumabs_av = float(np.einsum('bts,bsd->btd', al,
                                            np.abs(vh)).max())
                E_h = ((1 + U32) ** (NT + 2) - 1) * sumabs_av \
                    + NT * E_al * (A_v + E_v) + E_v
                E_cat = max(E_cat, E_h)
                # proven linear face of the score path (per head, take max)
                W_sm = float(np.expm1(4 * A_s)) / (2 * A_s)
                g_s = 0.125 * DH * (A_q * rs_k + A_k * rs_q)
                g_score_path = max(g_score_path,
                                   NT * A_v * W_sm * g_s + rs_v)
            attf = att.reshape(-1, D)
            wo = (attf @ vals[p_ + "Wo"].astype(dt)
                  + vals[p_ + "bo"].astype(dt)).reshape(32, NT, D)
            E_wo, rs_o = dense_budget(attf, p_ + "Wo", p_ + "bo", E_cat)
            h1 = inp + wo
            E_attn = E_wo + U32 * mags(h1)
            ln2 = ln_tok(h1, p_ + "g2", p_ + "bt2")
            E_ln2, g_ln2 = ln_budget_gain(h1, p_ + "g2", E_attn)
            ln2f = ln2.reshape(-1, D)
            f1 = (ln2f @ vals[p_ + "Wfc1"].astype(dt)
                  + vals[p_ + "bfc1"].astype(dt))
            E_f1, rs_f1 = dense_budget(ln2f, p_ + "Wfc1", p_ + "bfc1", E_ln2)
            g1v = gelu_tanh_np(f1)
            E_g = 1.5 * E_f1 + EGELU
            f2 = (g1v @ vals[p_ + "Wfc2"].astype(dt)
                  + vals[p_ + "bfc2"].astype(dt)).reshape(32, NT, D)
            E_f2, rs_f2 = dense_budget(g1v, p_ + "Wfc2", p_ + "bfc2", E_g)
            h = h1 + f2
            E_out = E_f2 + E_attn + U32 * mags(h)
            # proven per-block gain: (1 + attn path)·(1 + mlp path); the LN
            # gains multiply their own sublayer path
            g_mlp_sub = 1.0 + g_ln2 * rs_f1 * 1.5 * rs_f2
            g_blk = (1.0 + g_ln1 * g_score_path * rs_o) * g_mlp_sub
            acts.append(h)
            names.append(f"block{bi}  (A_s={A_s_max:.1f})")
            bs_.append(E_out)
            gs.append(g_blk)
        lnF = ln_tok(h, "gF", "btF")
        E_lnF, g_lnF = ln_budget_gain(h, "gF", 0.0)
        acts.append(lnF)
        names.append("final LN")
        bs_.append(E_lnF)
        gs.append(g_lnF)
        clsv = lnF[:, 0, :]
        out = clsv @ vals["Wc"].astype(dt) + vals["bc"].astype(dt)
        E_hd, rs_hd = dense_budget(clsv, "Wc", "bc", 0.0)
        acts.append(out)
        names.append("cls+dense")
        bs_.append(E_hd)
        gs.append(rs_hd)
        return acts, names, bs_, gs

    acts, names, bs_, gs = oracle()
    out = acts[-1]
    # suffix products in log10 (the softmax window face overflows float64)
    lg = [np.log10(max(g, 1e-300)) for g in gs]
    H_prov_log = [float(np.sum(lg[i + 1:])) for i in range(len(lg))]
    H_prov = [10.0 ** min(h, 300.0) for h in H_prov_log]

    out_g = run_iree(mlir, [vals[n] for n, _ in sig],
                     fn="vit_fwd", module_name="m").astype(np.float64)
    true_err = float(np.abs(out_g - out).max())

    # ── measured tail gains (f32; per-token LN + per-sample attn) ──────────
    def jx(a):
        return jnp.asarray(np.asarray(a, np.float32))

    def jln(z, gname, btname):
        mu = z.mean(axis=-1, keepdims=True)
        var = ((z - mu) ** 2).mean(axis=-1, keepdims=True)
        return (z - mu) / jnp.sqrt(var + EPS_LN) * jx(vals[gname]) \
            + jx(vals[btname])

    def jgelu2(x):
        inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + jnp.tanh(inner))

    def stage_fns():
        fns = []
        for bi in range(12):
            p_ = f"b{bi}_"

            def bf(a, p_=p_):
                inp = a
                ln1 = jln(a, p_ + "g1", p_ + "bt1")
                q = ln1 @ jx(vals[p_ + "Wq"]) + jx(vals[p_ + "bq"])
                k = ln1 @ jx(vals[p_ + "Wk"]) + jx(vals[p_ + "bk"])
                v = ln1 @ jx(vals[p_ + "Wv"]) + jx(vals[p_ + "bv"])
                outs = []
                for hd in range(NH):
                    sl = slice(hd * DH, (hd + 1) * DH)
                    s = jnp.einsum('btd,bsd->bts', q[:, :, sl],
                                   k[:, :, sl]) * 0.125
                    e = jnp.exp(s)
                    al = e / e.sum(axis=-1, keepdims=True)
                    outs.append(jnp.einsum('bts,bsd->btd', al, v[:, :, sl]))
                att = jnp.concatenate(outs, axis=-1)
                wo = att @ jx(vals[p_ + "Wo"]) + jx(vals[p_ + "bo"])
                h1 = inp + wo
                ln2 = jln(h1, p_ + "g2", p_ + "bt2")
                f1 = jgelu2(ln2 @ jx(vals[p_ + "Wfc1"])
                            + jx(vals[p_ + "bfc1"]))
                f2 = f1 @ jx(vals[p_ + "Wfc2"]) + jx(vals[p_ + "bfc2"])
                return h1 + f2
            fns.append(bf)
        fns.append(lambda a: jln(a, "gF", "btF"))
        fns.append(lambda a: a[:, 0, :] @ jx(vals["Wc"]) + jx(vals["bc"]))
        return fns

    fns = stage_fns()
    H_meas = []
    for i in range(len(acts) - 1):
        aout = acts[i]
        gain = 0.0
        for smp in range(2):
            a1 = jx(aout[smp:smp + 1])

            def tail(aa, i=i):
                for f in fns[i:]:
                    aa = f(aa)
                return aa[0]
            J = jax.jacrev(tail)(a1)
            gain = max(gain, float(jnp.abs(J.reshape(10, -1))
                                   .sum(axis=1).max()))
        H_meas.append(gain)
    H_meas.append(1.0)

    cb_meas = sum(H * b for H, b in zip(H_meas, bs_))
    lt = [hl + np.log10(max(b, 1e-300)) for hl, b in zip(H_prov_log, bs_)]
    cb_prov_log = max(lt) + np.log10(sum(10.0 ** (t - max(lt)) for t in lt))
    print(f"\n{'stage':<22}{'fresh b_i':>12}{'H_i meas':>11}{'H_i·b_i':>11}"
          f"{'H_i proven':>13}")
    for name, b, Hm, Hpl in zip(names, bs_, H_meas, H_prov_log):
        print(f"{name:<22}{b:>12.3e}{Hm:>11.3e}{Hm * b:>11.3e}"
              f"{'1e%+d' % round(Hpl):>13}")
    print(f"\n  true GPU logits drift : {true_err:.3e}")
    print(f"  chainBudget measured-H: {cb_meas:.3e}  "
          f"({cb_meas / true_err:.1e}× true)")
    print(f"  chainBudget proven-H  : ~1e{cb_prov_log:+.0f}  "
          f"(~1e{cb_prov_log - np.log10(true_err):+.0f}× true)"
          f"   [= the interval fold]")
    print(f"  logits magnitude      : {mags(out):.3e}")
    print("\n  THE COMMITTED RENDER (verified_mlir/vit_fwd.mlir) run as-is on")
    print("  gfx1100; per-token LN (n=192), 3 heads of 64, exact per-head")
    print("  slice/pad structure and no-max-shift softmax mirrored in f64.")


def main():
    print("═" * 78)
    print("adjoint-chain probe — chain_adjointClose budget vs real gfx1100 drift")
    print("  chainBudget = Σ_i H_i · layerBudget(u32, m, max|W_i|, max|b_i|, A, 0)")
    print("═" * 78)
    print(f"{'depth':>6} {'true GPU err':>13} {'measured-H':>13} "
          f"{'/true':>10} {'proven-H':>13} {'/true':>10}")
    for depth in [3, 6, 12, 24]:
        params = []
        for _ in range(depth):
            w = rng.standard_normal((N, N)) * (2.0 / N) ** 0.5
            b = rng.standard_normal(N) * 0.01
            params.append((w, b))
        x64 = rng.standard_normal((B, N))
        x32 = x64.astype(np.float32)
        p32 = [(w.astype(np.float32), b.astype(np.float32)) for w, b in params]
        pr64 = [(w.astype(np.float64), b.astype(np.float64)) for w, b in p32]
        xr64 = x32.astype(np.float64)

        # f64 trajectory + relu masks (the oracle side of the bridge)
        hs, masks = [xr64], []
        for w, b in pr64:
            z = hs[-1] @ w + b
            masks.append(z > 0)
            hs.append(np.maximum(z, 0.0))
        y64 = hs[-1]

        # true deployed-kernel drift: IREE f32 on gfx1100 vs the f64 oracle
        flat = [x32]
        for w, b in p32:
            flat += [w, b]
        gpu = run_iree(mlp_mlir(depth), flat).astype(np.float64)
        true_err = float(np.abs(gpu - y64).max())

        # the uniform window A (max activation magnitude, real and float)
        Awin = max(float(np.abs(h).max()) for h in hs)
        Awin = max(Awin, float(np.abs(gpu).max()))

        bs = [layer_budget_fresh(U32, N, float(np.abs(w).max()),
                                 float(np.abs(b).max()), Awin)
              for w, b in pr64]

        # PROVEN discharge: suffix products of the per-layer row-sum gains
        gs = [N * float(np.abs(w).max()) for w, _ in pr64]
        H_proven = [float(np.prod(gs[i + 1:])) for i in range(depth)]

        # MEASURED discharge: L∞→L∞ norm of the tail Jacobian on-trajectory.
        # Row-vector convention h' = relu(hW + b):  J_j[k,i] = mask_k · W[i,k];
        # tail after layer i = J_n ⋯ J_{i+1} (recorded before folding J_i in).
        H_meas = [0.0] * depth
        for s in range(NJAC):
            Jt = np.eye(N)
            for i in range(depth - 1, -1, -1):
                H_meas[i] = max(H_meas[i],
                                float(np.abs(Jt).sum(axis=1).max()))
                w, _ = pr64[i]
                Jt = Jt @ (w.T * masks[i][s][:, None])

        cb_meas = sum(H * b for H, b in zip(H_meas, bs))
        cb_prov = sum(H * b for H, b in zip(H_proven, bs))
        print(f"{depth:>6} {true_err:>13.3e} {cb_meas:>13.3e} "
              f"{cb_meas / true_err:>9.1e}x {cb_prov:>13.3e} "
              f"{cb_prov / true_err:>9.1e}x")

    print("\n  read: proven-H = the old interval fold (exponentially vacuous with")
    print("        depth); measured-H = the SAME theorem with on-trajectory tail")
    print("        gains — depth-linear, non-vacuous at every depth. The gap that")
    print("        remains vs true is per-layer Higham conservatism, not depth.")


if __name__ == "__main__":
    main()
    cnn_probe()
    cifar8_probe()
    cifar8bn_probe()
    # note: run these last — earlier sections pin jax to CPU; when run
    # standalone they measure their gains on the GPU
    r34_probe()
    enet_probe()
    convnext_probe()
    vit_probe()
