#!/usr/bin/env python3
"""MNv4 forward tie — the verified render against the JAX reference, on SHARED weights.

⭐ THIS IS THE GATE NOTHING ELSE CAN SUBSTITUTE FOR. `uib-layout-tie` pins the parameter
LAYOUT and `mnv4-fwd-smoke` pins the op counts and the per-block depthwise widths, but a
pre-DW and a post-DW at the same `k` and channel count have identical shapes, identical
counts and identical group widths — so a renderer that SWAPS them passes both, type-checks,
trains, descends, and is not MobileNetV4. Same invisible class as R50's stride-on-the-3x3 and
the 2x2 stem pool. Only running both sides on one set of weights pins the ORDER.

What it does:
  1. parses `@mnv4_fwd`'s input shapes out of the emitted MLIR
  2. draws one fixed-random set of weights
  3. runs the verified render through iree-compile + iree-run-module
  4. runs `forward()` from `jax/.lake/build/generated_mobilenet_v4.py` — the ACTUAL reference,
     not a re-implementation — on the same weights
  5. compares logits

⚠ Two known convention differences, handled explicitly rather than absorbed into a loose
tolerance (a tolerance that hides a convention bug is worse than no gate):

  * **Classifier weight orientation.** The verified layout carries `%Wd : [1280, nClasses]`
    (`VLayer.dense ic oc -> (#[ic,oc], 0)`, shared with R34/R50); the reference computes
    `mm(x, params[52][0].T)`, i.e. it stores `[nClasses, 1280]`. Transposed at the boundary.
  * **The reference module is not importable** — its training loop runs at module level. We
    exec only the prefix up to `def loss_fn`, which is every function `forward` needs and
    none of the loop.

The reference itself is pinned to timm's `mobilenetv4_conv_medium` by `scripts/parity/mnv4_timm_parity.py`;
this tie carries that to the render.

Usage:
    scripts/parity/mnv4_forward_tie.py                 # batch 2, seed 42
    scripts/parity/mnv4_forward_tie.py --batch 4 --tol 2e-4
"""
import argparse, os, re, sys, tempfile
import numpy as np

REF_PY = "jax/.lake/build/generated_mobilenet_v4.py"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))  # the shared helpers
import _iree  # noqa: E402


def parse_input_shapes(mlir_path, fn):
    txt = open(mlir_path).read()
    m = re.search(rf'func\.func @{re.escape(fn)}\((.*?)\)\s*->', txt, re.S)
    if not m:
        sys.exit(f"could not find func @{fn} in {mlir_path}")
    return re.findall(r'tensor<([0-9x]*)f32>', m.group(1))


def load_reference_forward():
    """exec the reference's function prefix — the real generated code, minus its train loop."""
    src = open(REF_PY).read().split("\n")
    cut = next((i for i, l in enumerate(src) if l.startswith("def loss_fn")), None)
    if cut is None:
        sys.exit(f"{REF_PY}: no `def loss_fn` to cut at; the generator's shape changed")
    body = "\n".join(src[:cut])
    mod = {}
    exec(body, mod)
    if "forward" not in mod:
        sys.exit("reference prefix did not define forward()")
    return mod["forward"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir", default=".lake/build/mnv4_fwd.mlir")
    ap.add_argument("--fn", default="mnv4_fwd")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=1e-4, help="max |Δ| over the logits")
    ap.add_argument("--backend", default="llvm-cpu", choices=_iree.BACKENDS)
    args = ap.parse_args()

    if not os.path.exists(args.mlir):
        sys.exit(f"{args.mlir} missing — run `lake build mnv4-fwd-smoke && "
                 f".lake/build/bin/mnv4-fwd-smoke` to emit it")

    work = tempfile.mkdtemp(prefix="mnv4tie_")
    shapes = parse_input_shapes(args.mlir, args.fn)
    print(f"func @{args.fn}: {len(shapes)} inputs  (workdir {work})")

    rng = np.random.default_rng(args.seed)
    arrays = []
    for i, s in enumerate(shapes):
        dims = [int(d) for d in s.split("x") if d]
        a = (np.asarray(rng.standard_normal(dims)).astype(np.float32) * args.scale)
        arrays.append(a)

    # ── the verified render ──
    # local-task first, local-sync on a silent 245 / -11 — `_iree.devices`; the first Conv-M tie
    # run without that fallback looked like a broken artifact (2026-09-07).
    got = _iree.compile_and_run(args.mlir, args.fn, arrays, work, 1,
                                args.backend)[0].astype(np.float64)

    # ── the reference, on the same weights ──
    import jax.numpy as jnp
    x = arrays[0].reshape(args.batch, 3, 224, 224)          # %x is flat [B, 3*224*224]
    ws = arrays[1:]
    params = [tuple(ws[i:i + 3]) for i in range(0, len(ws) - 2, 3)]  # 77 (W, γ, β) triples
    params.append((ws[-2].T, ws[-1]))                        # dense: [1280,K] -> reference's [K,1280]
    print(f"  reference params: {len(params)} entries "
          f"({sum(1 for p in params if len(p) == 3)} triples + dense)")
    want = np.asarray(load_reference_forward()(params, jnp.asarray(x))).astype(np.float64)

    if got.shape != want.shape:
        sys.exit(f"SHAPE MISMATCH: render {got.shape} vs reference {want.shape}")
    d = np.abs(got - want)
    scale = max(np.abs(want).max(), 1e-12)
    print(f"  logits range (reference): [{want.min():.4f}, {want.max():.4f}]")
    print(f"  max |Δ|      : {d.max():.3e}")
    print(f"  mean |Δ|     : {d.mean():.3e}")
    print(f"  max |Δ|/scale: {d.max()/scale:.3e}")
    if d.max() <= args.tol:
        print(f"  ✓ FORWARD TIE PASSES (tol {args.tol:.1e}) — the verified render computes "
              f"the reference's function, block order included")
        return 0
    print(f"  ✗ FORWARD TIE FAILS (tol {args.tol:.1e})")
    print("    A clean tie is the ONLY evidence that the pre/post-DW order is right.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
