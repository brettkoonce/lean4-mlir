#!/usr/bin/env python3
"""Kernel-faithfulness probe — does the *kernel IREE actually runs* stay inside
the *proven `FloatModel` budget*?  (planning/floatbridge_certificate_gaps.md §1.)

The float bridge proves `|float_op − real_op| ≤ budget` over the abstract
`FloatModel` (`|rnd x − x| ≤ u·|x|`, `u = u32 = 2⁻²⁴`).  That bounds the *model*.
The deployed path is  den(ℝ) → Float32 → IREE-emitted GPU kernel  (FMA, tree-
reduction order, accumulation precision) — today the biggest TRUSTED item.  This
script does **differential validation**: it runs the emitted StableHLO on the
*real* gfx1100 GPU in f32, diffs against an f64 exact-ℝ proxy, and checks the
measured drift stays inside the proven budget at the real fan-in.  Same shape as
`scripts/transcendental_probe.py` (transcendentals) and `cifar_bn_margin_probe.py`
(CPU f32/f64 twin) — but here the f32 side is the **real GPU kernel via IREE**,
not a numpy twin.  That distinction *is* the §1 gap.

Two parts:

  (1) CORE dot_general — the FMA + reduction-order question.  Emits a tiny
      `stablehlo.dot_general` at controlled fan-in `n`, runs it on gfx1100, and
      checks per-output-element
          |gpu_f32 − f64| ≤ ((1+u32)^(n+1) − 1)·Σ_k|x_ik·y_kj|
      the **`dot_close` Higham budget** (FloatBridge.lean:224, exponent n+1).
      Also reports the CPU-f32 twin (the same dot in f32 on the host) for
      contrast: the GPU should be *within* the model AND typically *closer* than
      the naive twin (FMA + higher-precision accumulate).  "The model is
      conservative w.r.t. the real kernel" — that contrast is the headline.

  (2) WHOLE rendered-net forward — the committed MNIST-CNN render
      (mlir_poc/mnist_cnn.mlir, MainCnn.lean architecture), emitted with every
      stage as a result, run once on gfx1100.  Each conv/dense pre-activation
      stage's measured GPU drift is tabled against the proven per-stage
      `layerBudget` (FloatBridge `layerBudget`, same formula as
      cifar_bn_margin_probe.py) evaluated at the measured magnitude profile, with
      the input-error `E` threaded from the previous stage's measured drift.

Run under the IREE/JAX venv:
  /home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python scripts/kernel_faithfulness_probe.py

What this does NOT do: formally prove kernel = model (that needs a verified
compiler).  It *validates* the trusted boundary on real silicon and documents
precisely what remains trusted (see the closing statement).

Reference run (2026-06-25, gfx1100 / Radeon RX 7900 XTX, IREE rocm, seed 0):
  (1) dot_general — GPU drift / dot_close budget, by fan-in:
        n=64    5.1e-02     n=512   6.9e-03
        n=6272  4.6e-04     n=25088 1.0e-04
      i.e. the proven budget is ~20× (small n) to ~10⁴× (large n) conservative
      w.r.t. the real GPU kernel — and the kernel is solidly INSIDE the model at
      every fan-in.  (GPU drift runs a few× the host-BLAS f32 twin; both ≪ budget.)
  (2) MNIST-CNN whole-net forward — every conv/dense stage's GPU drift sits at
      ratio 4e-2 … 2e-6 of its proven layerBudget; maxpool passes error through
      without amplification.  The real composed kernel stays inside the bridge's
      bound end-to-end.
"""
import sys
import numpy as np

U32 = 2.0 ** -24                 # binary32 unit roundoff (the FloatModel u)
rng = np.random.default_rng(0)


# ── IREE on gfx1100 (rocm backend / hip runtime), multi-result ───────────────
def run_iree(mlir: str, inputs, fn: str = "main"):
    """Compile `mlir` for gfx1100 and run `fn` on `inputs`; return host arrays
    (a single np.ndarray, or a list if the function returns multiple results)."""
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


# ═══════════════════════════════════════════════════════════════════════════
# (1) CORE dot_general — FMA + reduction order vs the Higham `dot_close` budget
# ═══════════════════════════════════════════════════════════════════════════
def dot_mlir(B: int, n: int, m: int) -> str:
    """`[B,n] · [n,m] → [B,m]`, contracting the fan-in axis — the deployed dense
    kernel shape (verbatim the renderer's dense `dot_general`, IRPrint.lean)."""
    A, W, O = f"tensor<{B}x{n}xf32>", f"tensor<{n}x{m}xf32>", f"tensor<{B}x{m}xf32>"
    return (f"module {{\n  func.func @main(%a: {A}, %w: {W}) -> {O} {{\n"
            f"    %o = stablehlo.dot_general %a, %w, contracting_dims = [1] x [0],\n"
            f"           precision = [DEFAULT, DEFAULT] : ({A}, {W}) -> {O}\n"
            f"    return %o : {O}\n  }}\n}}\n")


def dot_probe():
    print("═" * 78)
    print("(1) CORE dot_general — |gpu_f32 − f64| vs Higham dot_close budget")
    print(f"    budget_ij = ((1+u32)^(n+1) − 1)·Σ_k|x_ik·y_kj|   (FloatBridge dot_close)")
    print("═" * 78)
    B, M = 64, 64
    # fan-ins spanning the committed nets: dense head 512, MNIST flatten 6272,
    # r34/convnext-scale 25088, plus a small 64 control.
    print(f"\n{'fan-in n':>9}{'GPU max|f32-f64|':>18}{'GPU/budget':>12}"
          f"{'CPU-twin/bud':>14}{'GPU vs twin':>12}")
    for n in (64, 512, 6272, 25088):
        x = rng.standard_normal((B, n)).astype(np.float32)
        w = rng.standard_normal((n, M)).astype(np.float32)
        x64, w64 = x.astype(np.float64), w.astype(np.float64)

        gpu = run_iree(dot_mlir(B, n, M), [x, w]).astype(np.float64)
        ref = x64 @ w64                                   # f64 exact-ℝ proxy
        cpu = (x @ w).astype(np.float64)                  # CPU f32 twin (formula in f32)

        # per-output-element Higham budget Σ_k|x_ik·y_kj|
        sum_abs = np.abs(x64) @ np.abs(w64)
        budget = ((1 + U32) ** (n + 1) - 1) * sum_abs
        gpu_err, cpu_err = np.abs(gpu - ref), np.abs(cpu - ref)
        gpu_r = float((gpu_err / budget).max())
        cpu_r = float((cpu_err / budget).max())
        contrast = gpu_err.max() / max(cpu_err.max(), 1e-300)
        print(f"{n:>9}{gpu_err.max():>18.3e}{gpu_r:>12.2e}{cpu_r:>14.2e}"
              f"{contrast:>11.2f}×")
    print("\n  read: GPU/budget < 1 ⇒ real kernel inside the proven FloatModel (the")
    print("        budget is conservative w.r.t. the deployed kernel — by ~20×+).")
    print("        GPU vs twin = GPU drift / host-BLAS f32 twin; both ≪ budget. >1 just")
    print("        means the host BLAS reduction is tighter than the GPU's parallel one.")


# ═══════════════════════════════════════════════════════════════════════════
# (2) WHOLE rendered-net forward — committed MNIST-CNN, per-stage measured-vs-proven
# ═══════════════════════════════════════════════════════════════════════════
# proven per-stage budget (FloatBridge `layerBudget`; same as cifar_bn_margin_probe)
def layer_budget(u, m, w, beta, A, E):
    """`((1+u)^(m+2) − 1)·(m·w·(A+E) + β) + m·w·E` — conv/dense fan-in budget,
    m = fan-in, w = max|weight|, β = max|bias|, A = max|input|, E = input error."""
    return ((1 + u) ** (m + 2) - 1) * (m * w * (A + E) + beta) + m * w * E


def conv_same3(x, W, b, dt):                              # NCHW / OIHW, 3×3 SAME
    x, W, b = x.astype(dt), W.astype(dt), b.astype(dt)
    N, C, H, Wd = x.shape; Oc = W.shape[0]
    xp = np.zeros((N, C, H + 2, Wd + 2), dt); xp[:, :, 1:H + 1, 1:Wd + 1] = x
    out = np.zeros((N, Oc, H, Wd), dt)
    for di in range(3):
        for dj in range(3):
            out += np.einsum('nchw,oc->nohw', xp[:, :, di:di + H, dj:dj + Wd],
                             W[:, :, di, dj], optimize=True)
    return out + b[None, :, None, None]


def maxpool2(a, dt):                                      # 2×2 stride 2
    N, C, H, Wd = a.shape
    return a.reshape(N, C, H // 2, 2, Wd // 2, 2).max(axis=(3, 5))


def cnn_mlir() -> str:
    """The committed mnist_cnn.mlir forward, emitted with EVERY stage as a result
    so one GPU run yields the full drift profile.  Op sequence verbatim from
    mlir_poc/mnist_cnn.mlir (MainCnn.lean)."""
    def conv(o, x, W, src_t, w_t, out_t):
        return (f'    {o} = "stablehlo.convolution"({x}, {W}) {{\n'
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
    T = lambda *s: "tensor<" + "x".join(map(str, s)) + "xf32>"
    t28 = T(4, 32, 28, 28)
    return (
        "module {\n"
        "  func.func @main(\n"
        "      %x:  tensor<4x1x28x28xf32>,\n"
        "      %W0: tensor<32x1x3x3xf32>,  %b0: tensor<32xf32>,\n"
        "      %W1: tensor<32x32x3x3xf32>, %b1: tensor<32xf32>,\n"
        "      %W2: tensor<6272x512xf32>,  %b2: tensor<512xf32>,\n"
        "      %W3: tensor<512x512xf32>,   %b3: tensor<512xf32>,\n"
        "      %W4: tensor<512x10xf32>,    %b4: tensor<10xf32>\n"
        f"    ) -> ({t28}, {t28}, {T(4,32,14,14)}, {T(4,512)}, {T(4,512)}, {T(4,10)}) {{\n"
        # conv1 + bias + relu
        + conv("%cv0", "%x", "%W0", T(4, 1, 28, 28), T(32, 1, 3, 3), t28) +
        f"    %b0b = stablehlo.broadcast_in_dim %b0, dims = [1] : (tensor<32xf32>) -> {t28}\n"
        f"    %c0a = stablehlo.add %cv0, %b0b : {t28}\n"
        f"    %z0 = stablehlo.constant dense<0.0> : {t28}\n"
        f"    %h0 = stablehlo.maximum %c0a, %z0 : {t28}\n"
        # conv2 + bias + relu
        + conv("%cv1", "%h0", "%W1", t28, T(32, 32, 3, 3), t28) +
        f"    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> {t28}\n"
        f"    %c1a = stablehlo.add %cv1, %b1b : {t28}\n"
        f"    %h1 = stablehlo.maximum %c1a, %z0 : {t28}\n"
        # maxpool 2×2 → 14×14
        "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
        '    %pool = "stablehlo.reduce_window"(%h1, %ninf) ({\n'
        "      ^bb0(%a: tensor<f32>, %bb: tensor<f32>):\n"
        "        %m = stablehlo.maximum %a, %bb : tensor<f32>\n"
        '        "stablehlo.return"(%m) : (tensor<f32>) -> ()\n'
        "      }) { window_dimensions = array<i64: 1, 1, 2, 2>,\n"
        "           window_strides    = array<i64: 1, 1, 2, 2> }\n"
        f"      : ({t28}, tensor<f32>) -> {T(4,32,14,14)}\n"
        f"    %flat = stablehlo.reshape %pool : ({T(4,32,14,14)}) -> {T(4,6272)}\n"
        # dense0 + relu
        f"    %d0 = stablehlo.dot_general %flat, %W2, contracting_dims = [1] x [0],\n"
        f"            precision = [DEFAULT, DEFAULT] : ({T(4,6272)}, tensor<6272x512xf32>) -> {T(4,512)}\n"
        f"    %b2b = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<512xf32>) -> {T(4,512)}\n"
        f"    %d0a = stablehlo.add %d0, %b2b : {T(4,512)}\n"
        f"    %z2 = stablehlo.constant dense<0.0> : {T(4,512)}\n"
        f"    %h2 = stablehlo.maximum %d0a, %z2 : {T(4,512)}\n"
        # dense1 + relu
        f"    %d1 = stablehlo.dot_general %h2, %W3, contracting_dims = [1] x [0],\n"
        f"            precision = [DEFAULT, DEFAULT] : ({T(4,512)}, tensor<512x512xf32>) -> {T(4,512)}\n"
        f"    %b3b = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<512xf32>) -> {T(4,512)}\n"
        f"    %d1a = stablehlo.add %d1, %b3b : {T(4,512)}\n"
        f"    %h3 = stablehlo.maximum %d1a, %z2 : {T(4,512)}\n"
        # dense2 (logits)
        f"    %d2 = stablehlo.dot_general %h3, %W4, contracting_dims = [1] x [0],\n"
        f"            precision = [DEFAULT, DEFAULT] : ({T(4,512)}, tensor<512x10xf32>) -> {T(4,10)}\n"
        f"    %b4b = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<10xf32>) -> {T(4,10)}\n"
        f"    %out = stablehlo.add %d2, %b4b : {T(4,10)}\n"
        f"    return %c0a, %c1a, %pool, %d0a, %d1a, %out : {t28}, {t28}, {T(4,32,14,14)}, {T(4,512)}, {T(4,512)}, {T(4,10)}\n"
        "  }\n}\n")


def cnn_probe():
    print("\n" + "═" * 78)
    print("(2) WHOLE-NET forward — committed MNIST-CNN render on gfx1100")
    print("    measured GPU drift vs proven per-stage layerBudget (E threaded)")
    print("═" * 78)
    # weights at the renderer's realistic magnitude profile (validate_cnn.py init)
    x  = rng.standard_normal((4, 1, 28, 28)).astype(np.float32)
    W0 = (rng.standard_normal((32, 1, 3, 3)) * 0.1).astype(np.float32);  b0 = (rng.standard_normal(32) * 0.1).astype(np.float32)
    W1 = (rng.standard_normal((32, 32, 3, 3)) * 0.05).astype(np.float32); b1 = (rng.standard_normal(32) * 0.1).astype(np.float32)
    W2 = (rng.standard_normal((6272, 512)) * 0.02).astype(np.float32);    b2 = (rng.standard_normal(512) * 0.05).astype(np.float32)
    W3 = (rng.standard_normal((512, 512)) * 0.05).astype(np.float32);     b3 = (rng.standard_normal(512) * 0.05).astype(np.float32)
    W4 = (rng.standard_normal((512, 10)) * 0.05).astype(np.float32);      b4 = (rng.standard_normal(10) * 0.05).astype(np.float32)
    inputs = [x, W0, b0, W1, b1, W2, b2, W3, b3, W4, b4]

    c0a_g, c1a_g, pool_g, d0a_g, d1a_g, out_g = (
        a.astype(np.float64) for a in run_iree(cnn_mlir(), inputs))

    # f64 exact-ℝ proxy forward (the accumulated f64 path)
    dt = np.float64
    c0a = conv_same3(x, W0, b0, dt);            h0 = np.maximum(c0a, 0)
    c1a = conv_same3(h0, W1, b1, dt);           h1 = np.maximum(c1a, 0)
    pool = maxpool2(h1, dt);                    flat = pool.reshape(4, -1)
    d0a = flat @ W2.astype(dt) + b2.astype(dt); h2 = np.maximum(d0a, 0)
    d1a = h2 @ W3.astype(dt) + b3.astype(dt);   h3 = np.maximum(d1a, 0)
    out = h3 @ W4.astype(dt) + b4.astype(dt)

    def drift(g, r): return float(np.abs(g - r).max())
    print(f"\n{'stage':<16}{'fan-in':>7}{'measured |gpu-f64|':>20}{'proven budget':>16}{'ratio':>9}")
    # stage tuples: name, fan-in m, max|W|, max|b|, max|input|, input-err E, gpu, ref
    e0 = 0.0
    e_c0 = drift(c0a_g, c0a)
    rows = [
        ("conv1 out", 1 * 9,  W0, b0, x,    e0,  c0a_g, c0a),
        ("conv2 out", 32 * 9, W1, b1, h0,   e_c0, c1a_g, c1a),
    ]
    e_prev = {}
    for name, m, W, b, A, E, g, r in rows:
        d = drift(g, r)
        bud = layer_budget(U32, m, float(np.abs(W).max()), float(np.abs(b).max()),
                           float(np.abs(A.astype(np.float64)).max()), E)
        print(f"{name:<16}{m:>7}{d:>20.3e}{bud:>16.3e}{d / bud:>9.2e}")
        e_prev[name] = d
    # maxpool: error passes through (1-Lipschitz select) — report, no fan-in budget
    d_pool = drift(pool_g, pool)
    print(f"{'maxpool out':<16}{'—':>7}{d_pool:>20.3e}{'(≤ conv2 drift)':>16}"
          f"{d_pool / max(e_prev['conv2 out'],1e-300):>9.2e}")
    # dense stages: fan-in = input dim, E threaded from prev measured drift
    dense_rows = [
        ("dense0 out", 6272, W2, b2, flat, d_pool, d0a_g, d0a),
        ("dense1 out", 512,  W3, b3, h2,   drift(d0a_g, d0a), d1a_g, d1a),
        ("logits",     512,  W4, b4, h3,   drift(d1a_g, d1a), out_g, out),
    ]
    for name, m, W, b, A, E, g, r in dense_rows:
        d = drift(g, r)
        bud = layer_budget(U32, m, float(np.abs(W).max()), float(np.abs(b).max()),
                           float(np.abs(A.astype(np.float64)).max()), E)
        print(f"{name:<16}{m:>7}{d:>20.3e}{bud:>16.3e}{d / bud:>9.2e}")
    print("\n  read: ratio = measured GPU drift / proven layerBudget; < 1 ⇒ the real")
    print("        composed kernel stays inside the bridge's bound end-to-end.")


def main():
    print(f"kernel_faithfulness_probe: gfx1100 / IREE rocm,  u32 = 2^-24 = {U32:.3e}\n"
          f"  f32 side = REAL GPU kernel (StableHLO→IREE→hip), f64 side = exact-ℝ proxy.")
    dot_probe()
    cnn_probe()
    print("\n" + "═" * 78)
    print("WHAT REMAINS TRUSTED (validated, not proven):")
    print("  • IREE lowering preserves StableHLO f32 semantics (no verified compiler).")
    print("  • The FFI / iree-runtime host↔device boundary.")
    print("  • Measured at one magnitude profile / seed; not an ∀-inputs proof.")
    print("  This probe VALIDATES the den→f32→GPU-kernel boundary on real silicon")
    print("  and bounds the residual; it does not formally close it.")
    print("═" * 78)


if __name__ == "__main__":
    main()
