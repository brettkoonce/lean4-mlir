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

⚠ KNOWN MISMATCH THIS IS EXPECTED TO SURFACE (as of 2026-08-08): the stem. The reference
calls `conv_bn(..., stride=(2,2), padding='SAME')`, and XLA 'SAME' on a 3x3/s2 at 224 pads
(0,1) — asymmetric. The verified render emits symmetric (1,1). Both yield 112x112, so no
shape check anywhere can see it. If this run reports a mismatch concentrated in the logits
rather than a clean tie, that is the first place to look.

Usage:
    scripts/mnv4_forward_tie.py                 # batch 2, seed 42
    scripts/mnv4_forward_tie.py --batch 4 --tol 2e-4
"""
import argparse, os, re, subprocess, sys, tempfile
import numpy as np

REF_PY = "jax/.lake/build/generated_mobilenet_v4.py"
CHIP = os.environ.get("IREE_CHIP", "gfx1100")
IREE_C = ".venv/bin/iree-compile"
# ⚠ iree-run-module is NOT in this repo's .venv (only iree-compile is). It ships with the
# lean4-jax venv — the same absolute path the PJRT plugin resolves through.
IREE_R = os.environ.get("IREE_RUN_MODULE",
    "/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/iree-run-module")


def parse_input_shapes(mlir_path, fn):
    txt = open(mlir_path).read()
    m = re.search(rf'func\.func @{re.escape(fn)}\((.*?)\)\s*->', txt, re.S)
    if not m:
        sys.exit(f"could not find func @{fn} in {mlir_path}")
    return re.findall(r'tensor<([0-9x]*)f32>', m.group(1))


def load_reference_forward(stem_symmetric=False):
    """exec the reference's function prefix — the real generated code, minus its train loop.

    `stem_symmetric` is a DIAGNOSTIC: it rewrites the stem's `padding='SAME'` to the symmetric
    ((1,1),(1,1)) the verified render emits. If the tie passes only with this on, the stem
    padding is the entire disagreement and nothing else is wrong — which is a different repair
    from "the block order is wrong", so it is worth isolating rather than guessing."""
    src = open(REF_PY).read().split("\n")
    cut = next((i for i, l in enumerate(src) if l.startswith("def loss_fn")), None)
    if cut is None:
        sys.exit(f"{REF_PY}: no `def loss_fn` to cut at; the generator's shape changed")
    body = "\n".join(src[:cut])
    if stem_symmetric:
        old = "params[0][2], stride=(2,2), padding='SAME')"
        if old not in body:
            sys.exit("stem line not found — the reference generator's shape changed")
        body = body.replace(old, "params[0][2], stride=(2,2), padding=((1,1),(1,1)))")
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
    ap.add_argument("--backend", default="llvm-cpu", help="llvm-cpu (portable) or rocm")
    ap.add_argument("--diag", action="store_true",
                    help="also evaluate the reference with a SYMMETRIC stem, to isolate padding")
    args = ap.parse_args()

    if not os.path.exists(args.mlir):
        sys.exit(f"{args.mlir} missing — run `lake build mnv4-fwd-smoke && "
                 f".lake/build/bin/mnv4-fwd-smoke` to emit it")

    work = tempfile.mkdtemp(prefix="mnv4tie_")
    os.makedirs(f"{work}/in", exist_ok=True)
    shapes = parse_input_shapes(args.mlir, args.fn)
    print(f"func @{args.fn}: {len(shapes)} inputs  (workdir {work})")

    rng = np.random.default_rng(args.seed)
    arrays, in_flags = [], []
    for i, s in enumerate(shapes):
        dims = [int(d) for d in s.split("x") if d]
        a = (np.asarray(rng.standard_normal(dims)).astype(np.float32) * args.scale)
        arrays.append(a)
        p = f"{work}/in/i{i}.npy"
        np.save(p, a)
        in_flags.append(f"--input=@{p}")

    # ── the verified render ──
    cflags = ([f"--iree-hal-target-backends=rocm", f"--iree-rocm-target={CHIP}"]
              if args.backend == "rocm" else ["--iree-hal-target-backends=llvm-cpu"])
    r = subprocess.run([IREE_C, *cflags, args.mlir, "-o", f"{work}/m.vmfb"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"iree-compile FAILED:\n{r.stderr[:3000]}")
    dev = "hip" if args.backend == "rocm" else "local-task"
    r = subprocess.run([IREE_R, f"--device={dev}", f"--module={work}/m.vmfb",
                        f"--function={args.fn}", *in_flags,
                        f"--output=@{work}/out.npy"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"iree-run-module FAILED:\n{r.stderr[:3000]}")
    got = np.load(f"{work}/out.npy").astype(np.float64)

    # ── the reference, on the same weights ──
    import jax.numpy as jnp
    x = arrays[0].reshape(args.batch, 3, 224, 224)          # %x is flat [B, 3*224*224]
    ws = arrays[1:]
    params = [tuple(ws[i:i + 3]) for i in range(0, len(ws) - 2, 3)]  # 52 (W, γ, β) triples
    params.append((ws[-2].T, ws[-1]))                        # dense: [1280,K] -> reference's [K,1280]
    print(f"  reference params: {len(params)} entries "
          f"({sum(1 for p in params if len(p) == 3)} triples + dense)")
    def evalref(sym):
        return np.asarray(load_reference_forward(sym)(params, jnp.asarray(x))).astype(np.float64)
    want = evalref(False)
    if args.diag:
        alt = evalref(True)
        for lbl, w in (("reference as-is ('SAME' stem)", want),
                       ("stem patched to SYMMETRIC (1,1)", alt)):
            dd = np.abs(got - w)
            print(f"  DIAG {lbl:34s} max|Δ| {dd.max():.3e}  mean {dd.mean():.3e}")

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
    print("    First suspect: the stem's padding (symmetric (1,1) here vs XLA 'SAME' (0,1)).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
