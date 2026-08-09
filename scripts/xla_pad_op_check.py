#!/usr/bin/env python3
"""Known-answer guard for `convStridedXla` / `depthwiseStridedXla` (`planning/mnv4_verified.md` §3e).

⭐ WHY AT THE OP LEVEL. These tokens are byte-identical to their symmetric siblings apart from four
numbers in the emitted `pad`. Types, output shapes, op counts and feature-group widths all agree,
so no `#guard`, arity audit or shape check can tell a correct render from one that picked the wrong
token. A whole-net tie can only say "something is off somewhere"; this says WHICH OP, which is what
you want when three nets are about to depend on these two.

Each probe module (from `tests/TestXlaPadOps.lean`) is run through iree-compile + iree-run-module
and compared against `jax.lax.conv_general_dilated(…, padding='SAME')` on the same weights.

Three things are asserted, and the second and third matter as much as the first:
  1. the Xla tokens MATCH XLA `SAME`;
  2. the symmetric control does NOT — otherwise the two conventions coincide at this shape and the
     test is vacuous while looking green;
  3. the Xla tokens do NOT match symmetric padding — the other direction of the same trap.

Usage:  .venv/bin/python3 scripts/xla_pad_op_check.py
        (run `lake env lean tests/TestXlaPadOps.lean` first to emit the modules)
"""
import os, re, subprocess, sys, tempfile
import numpy as np

IREE_C = ".venv/bin/iree-compile"
IREE_R = os.environ.get("IREE_RUN_MODULE",
    "/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/iree-run-module")

# (module, func, kernel-shape, grouped?, kH) — B=2, 16x16 -> 8x8 throughout.
PROBES = [
    (".lake/build/xlapad_conv.mlir",     "conv_xla",  (8, 3, 3, 3), False, 3, "xla"),
    (".lake/build/xlapad_dw_k3.mlir",    "dw_xla_k3", (6, 1, 3, 3), True,  3, "xla"),
    (".lake/build/xlapad_dw_k5.mlir",    "dw_xla_k5", (6, 1, 5, 5), True,  5, "xla"),
    (".lake/build/xlapad_conv_sym.mlir", "conv_sym",  (8, 3, 3, 3), False, 3, "sym"),
]
B, HIN = 2, 16


def run_module(mlir, fn, arrays, work):
    flags = []
    for i, a in enumerate(arrays):
        p = f"{work}/{fn}_i{i}.npy"
        np.save(p, a)
        flags.append(f"--input=@{p}")
    r = subprocess.run([IREE_C, "--iree-hal-target-backends=llvm-cpu", mlir,
                        "-o", f"{work}/{fn}.vmfb"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"iree-compile FAILED on {mlir}:\n{r.stderr[:2000]}")
    r = subprocess.run([IREE_R, "--device=local-task", f"--module={work}/{fn}.vmfb",
                        f"--function={fn}", *flags, f"--output=@{work}/{fn}_o.npy"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"iree-run-module FAILED on {fn}:\n{r.stderr[:2000]}")
    return np.load(f"{work}/{fn}_o.npy").astype(np.float64)


def main():
    import jax, jax.numpy as jnp

    missing = [m for m, *_ in PROBES if not os.path.exists(m)]
    if missing:
        sys.exit(f"missing probe modules: {missing}\n"
                 f"run `lake env lean tests/TestXlaPadOps.lean` first")

    work = tempfile.mkdtemp(prefix="xlapad_")
    rng = np.random.default_rng(7)
    print(f"{'probe':<12} {'vs XLA SAME':>14} {'vs symmetric':>14}   verdict")
    print("-" * 62)
    failures = []

    for mlir, fn, wshape, grouped, k, kind in PROBES:
        c_in = wshape[0] if grouped else wshape[1]
        x = (rng.standard_normal((B, c_in, HIN, HIN)) * 0.5).astype(np.float32)
        W = (rng.standard_normal(wshape) * 0.5).astype(np.float32)
        bias = (rng.standard_normal((wshape[0],)) * 0.5).astype(np.float32)

        got = run_module(mlir, fn, [x.reshape(B, -1), W, bias], work)
        got = got.reshape(B, wshape[0], HIN // 2, HIN // 2)

        def ref(pad):
            y = jax.lax.conv_general_dilated(
                jnp.asarray(x), jnp.asarray(W), (2, 2), pad,
                dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
                feature_group_count=(c_in if grouped else 1))
            return np.asarray(y).astype(np.float64) + bias.astype(np.float64)[None, :, None, None]

        p = (k - 1) // 2
        d_same = np.abs(got - ref('SAME')).max()
        d_sym = np.abs(got - ref(((p, p), (p, p)))).max()

        if kind == "xla":
            ok = d_same < 1e-4 and d_sym > 1e-3
            verdict = "✓ XLA SAME" if ok else "✗ WRONG PADDING"
        else:
            ok = d_sym < 1e-4 and d_same > 1e-3
            verdict = "✓ symmetric (control)" if ok else "✗ CONTROL BROKEN"
        if not ok:
            failures.append(fn)
        print(f"{fn:<12} {d_same:>14.3e} {d_sym:>14.3e}   {verdict}")

    # ── BACKWARD probes, against jax.vjp of the actual Xla-padded forward ──
    # These emits had no reference to copy: their pad was DERIVED (`p-1`, `p+1`). A one-off in
    # either direction type-checks, gets the shape right, and still descends. jax.vjp is the only
    # thing that catches it, so it is not optional.
    print()
    print(f"{'backward probe':<16} {'vs jax.vjp':>12} {'vs SYMMETRIC vjp':>18}   verdict")
    print("-" * 62)
    # `expect` says WHICH convention this op is supposed to implement. The symmetric strided
    # depthwise backward is the one MobileNetV4's UIB blocks and EfficientNet's MBConv blocks use,
    # and until now nothing checked it at all — the probes were added alongside the `…Xla…` ops and
    # stopped there. For those rows the two columns swap roles: `vs SYMMETRIC vjp` is the answer
    # and `vs jax.vjp('SAME')` is the control that must NOT match.
    for fn, mlir, grouped, c_in, oc, k, kind, expect in [
        ("dw_xla_back",     ".lake/build/xlapad_dw_back.mlir",         True,  6, 6, 3, "gx", "same"),
        ("conv_xla_wgrad",  ".lake/build/xlapad_conv_wgrad.mlir",      False, 3, 8, 3, "gw", "same"),
        ("dw_xla_wgrad",    ".lake/build/xlapad_dw_wgrad.mlir",        True,  6, 6, 3, "gw", "same"),
        ("dw_sym_back_k3",  ".lake/build/xlapad_dw_sym_back_k3.mlir",  True,  6, 6, 3, "gx", "sym"),
        ("dw_sym_back_k5",  ".lake/build/xlapad_dw_sym_back_k5.mlir",  True,  6, 6, 5, "gx", "sym"),
        ("dw_sym_wgrad_k5", ".lake/build/xlapad_dw_sym_wgrad_k5.mlir", True,  6, 6, 5, "gw", "sym"),
    ]:
        if not os.path.exists(mlir):
            sys.exit(f"missing {mlir} — re-run `lake env lean tests/TestXlaPadOps.lean`")
        wshape = (oc, 1, k, k) if grouped else (oc, c_in, k, k)
        x = (rng.standard_normal((B, c_in, HIN, HIN)) * 0.5).astype(np.float32)
        W = (rng.standard_normal(wshape) * 0.5).astype(np.float32)
        dy = (rng.standard_normal((B, oc, HIN // 2, HIN // 2)) * 0.5).astype(np.float32)
        pp = (k - 1) // 2

        def fwd(xx, ww, pad):
            return jax.lax.conv_general_dilated(
                xx, ww, (2, 2), pad, dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
                feature_group_count=(c_in if grouped else 1))

        def grads(pad):
            _, vjp = jax.vjp(lambda xx, ww: fwd(xx, ww, pad), jnp.asarray(x), jnp.asarray(W))
            gx, gw = vjp(jnp.asarray(dy))
            return np.asarray(gx).astype(np.float64), np.asarray(gw).astype(np.float64)

        gx_x, gw_x = grads('SAME')                       # the Xla-padded net's true gradients
        gx_s, gw_s = grads(((pp, pp), (pp, pp)))         # the symmetric net's — the wrong answer

        if kind == "gx":
            got = run_module(mlir, fn, [dy.reshape(B, -1), W], work).reshape(B, c_in, HIN, HIN)
            ref_x, ref_s = gx_x, gx_s
        else:
            got = run_module(mlir, fn, [dy.reshape(B, -1), x.reshape(B, -1)], work)
            ref_x, ref_s = gw_x.reshape(got.shape), gw_s.reshape(got.shape)

        d_x, d_s = np.abs(got - ref_x).max(), np.abs(got - ref_s).max()
        hit, miss = (d_x, d_s) if expect == "same" else (d_s, d_x)
        ok = hit < 1e-3 and miss > 1e-3
        label = "XLA-SAME" if expect == "same" else "symmetric"
        verdict = f"✓ matches {label} vjp" if ok else "✗ WRONG GRADIENT"
        if hit < 1e-3 and miss < 1e-3:
            verdict = "✗ VACUOUS (both match)"
        if not ok:
            failures.append(fn)
        print(f"{fn:<16} {d_x:>12.3e} {d_s:>18.3e}   {verdict}")

    print()
    if failures:
        print(f"✗ FAILED: {', '.join(failures)}")
        print("  If BOTH columns are small the two conventions coincide at this shape and the")
        print("  probe is vacuous — change the shape, do not loosen the tolerance.")
        return 1
    print("✓ ALL PROBES PASS — the Xla tokens compute XLA 'SAME', the symmetric control does not,")
    print("  and neither matches the other. The two conventions are genuinely distinguished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
