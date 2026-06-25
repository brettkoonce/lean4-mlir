#!/usr/bin/env python3
"""Empirical pin of the float-bridge transcendental constants `esig` / `egelu`
against REAL ROCm silicon (gfx1100), via IREE — the deployed-kernel path.

The float bridge supplies, as hypotheses, the per-coordinate accuracy of the
GPU's transcendental activations:

    sigmoid:  |fsig t  - sigmoidScalar t| ≤ esig           (EnetFloatBridge.sigmoid_close)
    gelu   :  |fgelu t - geluScalar  t| ≤ egelu            (ViTFloatBridge.gelu_close)

where `sigmoidScalar t = 1/(1+e^-t)` and `geluScalar t = ½t(1 + tanh(√(2/π)(t+0.044715 t³)))`.
`eexp` was pinned at ~1–2 ULP by `margin_probe.py` (f32/f64 twin) and `ers` (rsqrt) by
`cifar_bn_margin_probe.py`; this is the same empirical pin for `esig`/`egelu`, but run on
the **actual GPU** rather than a CPU twin, because the bound is on the *silicon* op.

What it does:
  * Compiles the EXACT op sequences the renderer emits (`IRPrint.lean`):
      - sigmoid  → `stablehlo.logistic`
      - gelu     → tanh-form: u = c·(x + a·x³),  t = tanh u,  y = ½x(1+t)
        with the DEPLOYED f32 constants  c = 0.7978845608,  a = 0.044715.
  * Runs them on gfx1100 (IREE rocm backend, hip runtime) over a dense input sweep.
  * Compares the f32 GPU output to an f64 "exact-ℝ proxy" reference:
      esig  = max |logistic_gpu(t)            − 1/(1+e^-t)|
      egelu = max |gelu_gpu(t)                − geluScalar_exact(t)|   (c = √(2/π))
  * Also reports the deployed-vs-truncated-c split for gelu (how much of egelu is the
    f32 constant truncation `0.7978845608` vs √(2/π)) and a CPU f32/f64 twin for contrast.

Usage:  scripts/transcendental_probe.py            # default sweep [-40,40], 4M pts
        scripts/transcendental_probe.py LO HI N    # custom range / point count

Run under the IREE/JAX venv, e.g.:
  /home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python scripts/transcendental_probe.py

Reference run (2026-06-25, gfx1100 / Radeon RX 7900 XTX, IREE rocm 3.12, sweep
[-40,40] × 4,000,001 pts):
  esig  ≤ 9.016e-08  (≈ 1.51·u32)   — stablehlo.logistic; max at t=+2.87, ≈1.5 ULP@out.
                                      CPU f32/f64 twin of the same formula: 9.150e-08.
  egelu ≤ 4.330e-07  (≈ 7.27·u32)   — tanh-form; max at t=+4.71 (the transition region;
                                      ≈0.91 ULP of the |output|≈4.7 there). The const
                                      truncation (c_f32 vs √(2/π), Δc≈2.3e-8) contributes
                                      only 7.2e-9 — egelu is essentially the silicon tanh
                                      + formula rounding. For large |t| gelu→identity
                                      (tanh saturates), so the error does NOT grow with |t|.
Both well within "a few ULP"; use esig ≤ 2·u32, egelu ≤ 8·u32 as safe pinned constants.
"""
import sys
import numpy as np

C_EXACT = np.sqrt(2.0 / np.pi)        # √(2/π) = 0.7978845608028654...   (geluScalar)
C_F32   = np.float64(np.float32(0.7978845608))  # the DEPLOYED kernel constant, as f32
A_GELU  = 0.044715
U32     = 2.0 ** -24                  # binary32 unit roundoff (the FloatModel u)

LO = float(sys.argv[1]) if len(sys.argv) > 1 else -40.0
HI = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
N  = int(sys.argv[3])   if len(sys.argv) > 3 else 4_000_001


# ── deployed op sequences (transcribed verbatim from IRPrint.lean) ──────────
def sigmoid_mlir(n: int) -> str:
    t = f"tensor<{n}xf32>"
    return (f"module {{\n  func.func @main(%x: {t}) -> {t} {{\n"
            f"    %s = stablehlo.logistic %x : {t}\n"
            f"    return %s : {t}\n  }}\n}}\n")

def gelu_mlir(n: int) -> str:
    t = f"tensor<{n}xf32>"
    return (f"module {{\n  func.func @main(%x: {t}) -> {t} {{\n"
            f"    %a = stablehlo.constant dense<0.044715> : {t}\n"
            f"    %c = stablehlo.constant dense<0.7978845608> : {t}\n"
            f"    %half = stablehlo.constant dense<0.5> : {t}\n"
            f"    %one = stablehlo.constant dense<1.0> : {t}\n"
            f"    %x2 = stablehlo.multiply %x, %x : {t}\n"
            f"    %x3 = stablehlo.multiply %x2, %x : {t}\n"
            f"    %ax3 = stablehlo.multiply %a, %x3 : {t}\n"
            f"    %inn = stablehlo.add %x, %ax3 : {t}\n"
            f"    %u = stablehlo.multiply %c, %inn : {t}\n"
            f"    %tnh = stablehlo.tanh %u : {t}\n"
            f"    %ot = stablehlo.add %one, %tnh : {t}\n"
            f"    %hx = stablehlo.multiply %half, %x : {t}\n"
            f"    %y = stablehlo.multiply %hx, %ot : {t}\n"
            f"    return %y : {t}\n  }}\n}}\n")


# ── f64 exact-ℝ proxies (the geluScalar / sigmoidScalar the bridge proves against) ──
def sigmoid_exact(x):  # 1/(1+e^-x)
    return 1.0 / (1.0 + np.exp(-x))

def gelu_exact(x, c):  # ½x(1 + tanh(c(x + a x³)))
    return 0.5 * x * (1.0 + np.tanh(c * (x + A_GELU * x ** 3)))


def run_iree(mlir: str, x: np.ndarray) -> np.ndarray:
    from iree.compiler import compile_str
    from iree import runtime as ireert
    vmfb = compile_str(mlir, target_backends=["rocm"],
                       extra_args=["--iree-rocm-target=gfx1100"],
                       input_type="stablehlo")
    cfg = ireert.Config("hip")
    ctx = ireert.SystemContext(config=cfg)
    ctx.add_vm_module(ireert.VmModule.copy_buffer(ctx.instance, vmfb))
    return np.asarray(ctx.modules.module["main"](x).to_host())


def report(name, err, x, ulp_of):
    """Max-abs error over the whole sweep + a few operating sub-ranges."""
    print(f"\n── {name} : |gpu_f32 − f64_exact| ──")
    for lab, lo, hi in [("operating |t|≤8 ", -8, 8), ("mid  |t|≤15 ", -15, 15),
                        (f"full [{LO:.0f},{HI:.0f}]", LO, HI)]:
        m = (x >= lo) & (x <= hi)
        i = np.argmax(err[m])
        xm = x[m][i]; em = err[m][i]
        print(f"   {lab}: max {em:.3e}  at t={xm:+.4f}   "
              f"(≈ {em/U32:6.2f}·u32, ≈ {em/ulp_of(xm):5.2f} ULP@out)")
    return float(err.max())


def main():
    x32 = np.linspace(LO, HI, N).astype(np.float32)
    x64 = x32.astype(np.float64)
    print(f"transcendental_probe: gfx1100 / IREE rocm, sweep [{LO},{HI}] × {N} pts "
          f"(step {(HI-LO)/(N-1):.2e}),  u32 = 2^-24 = {U32:.3e}")

    # ── sigmoid ──────────────────────────────────────────────────────────
    sig_gpu = run_iree(sigmoid_mlir(N), x32).astype(np.float64)
    sig_ref = sigmoid_exact(x64)
    sig_err = np.abs(sig_gpu - sig_ref)
    # contrast: CPU f32 twin of the same formula
    sig_cpu = (1.0 / (1.0 + np.exp(-x32))).astype(np.float64)
    ulp_sig = lambda t: max(np.spacing(np.float32(sigmoid_exact(np.float64(t)))), 1e-45)
    esig = report("SIGMOID (stablehlo.logistic)", sig_err, x64, ulp_sig)
    print(f"   CPU f32/f64 twin (same formula): max {np.max(np.abs(sig_cpu-sig_ref)):.3e}")

    # ── gelu ─────────────────────────────────────────────────────────────
    gel_gpu = run_iree(gelu_mlir(N), x32).astype(np.float64)
    gel_ref = gelu_exact(x64, C_EXACT)            # vs the EXACT geluScalar (c = √(2/π))
    gel_ref_tc = gelu_exact(x64, C_F32)           # vs the truncated-c formula
    gel_err = np.abs(gel_gpu - gel_ref)
    gel_err_tc = np.abs(gel_gpu - gel_ref_tc)
    ulp_gel = lambda t: max(np.spacing(np.float32(abs(gelu_exact(np.float64(t), C_EXACT)) + 1e-30)), 1e-45)
    egelu = report("GELU (tanh-form, deployed c=0.7978845608)", gel_err, x64, ulp_gel)
    print(f"   vs truncated-c formula (isolates silicon, no const-trunc): "
          f"max {gel_err_tc.max():.3e}")
    print(f"   const-trunc share  max|gelu_exact(c=√2/π) − gelu_exact(c_f32)|: "
          f"{np.max(np.abs(gel_ref - gel_ref_tc)):.3e}   "
          f"(Δc = {abs(C_EXACT-C_F32):.2e})")

    # ── verdict ──────────────────────────────────────────────────────────
    print("\n══ PIN (max over full sweep, real gfx1100 silicon) ══")
    print(f"   esig  ≤ {esig:.3e}   (≈ {esig/U32:.2f}·u32)")
    print(f"   egelu ≤ {egelu:.3e}   (≈ {egelu/U32:.2f}·u32)")
    print("   Both are absolute per-coordinate bounds; feed as the egelu/esig")
    print("   hypotheses of gelu_close / sigmoid_close (validated, not assumed).")


if __name__ == "__main__":
    main()
