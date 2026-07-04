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
def run_iree(mlir: str, inputs, fn: str = "main"):
    from iree.compiler import compile_str
    from iree import runtime as ireert
    vmfb = compile_str(mlir, target_backends=["rocm"],
                       extra_args=["--iree-rocm-target=gfx1100"],
                       input_type="stablehlo")
    cfg = ireert.Config("hip")
    ctx = ireert.SystemContext(config=cfg)
    ctx.add_vm_module(ireert.VmModule.copy_buffer(ctx.instance, vmfb))
    out = ctx.modules.module[fn](*inputs)
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
