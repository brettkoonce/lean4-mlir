#!/usr/bin/env python3
"""EfficientNet-B0 forward tie — the verified render against the JAX reference, on SHARED weights.

The third and last net in the padding sweep (`planning/mnv4_verified.md` §3d(d) listed it as
"structurally exposed, NOT measured" — this is the measurement, and it CORRECTS that entry).

⭐ WHAT READING THE GENERATOR ALREADY SETTLED. §3c/§3d inferred EfficientNet was exposed at all
five of its stride-2 convolutions, by analogy with MobileNetV2's `depthwise_conv` default. It is
not: `mbconv_block` computes `pad = ((ksize-1)//2, (ksize-1)//2)` and passes it EXPLICITLY
(generated_efficientnet_b0.py:144), exactly as MobileNetV4's `uib_block` does. So its depthwises
are symmetric and the render's `[[1,1]]` (k=3) / `[[2,2]]` (k=5) are already right. **Only the stem
is exposed** — one site, like MNv4, not five like MNv2.

⚠ The interesting parameter-layout wrinkles, both handled explicitly rather than absorbed into a
loose tolerance:

  * **The SE block is a matmul here and a 1×1 conv there.** The verified layout carries
    `%bNzW1 : [c, r]` and `%bNzW2 : [r, c]` (`BatchableOp.seBlock` holds `Mat`s); the reference runs
    `conv_general_dilated` on a `[B,c,1,1]` tensor, so it wants `[r,c,1,1]` and `[c,r,1,1]`. Same
    function, transposed and reshaped at the boundary.
  * **Entry arity is not uniform.** Blocks are (expand?, dw, se1, se2, project) = triple, triple,
    PAIR, PAIR, triple — so the "group into threes" trick the mnv2/mnv4 ties use does not apply.
    The grouping is built from the block table instead, and the entry count is asserted.

⚠ Batch is FIXED at 32 by the render and the forward uses BATCH BN, so it cannot be shrunk.

Usage:  .venv/bin/python3 scripts/enet_forward_tie.py --diag
"""
import argparse, os, re, subprocess, sys, tempfile
import numpy as np

REF_PY = "jax/.lake/build/generated_efficientnet_b0.py"
CHIP = os.environ.get("IREE_CHIP", "gfx1100")
IREE_C = ".venv/bin/iree-compile"
IREE_R = os.environ.get("IREE_RUN_MODULE",
    "/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/iree-run-module")

STEM_SAME = "params[0][2], stride=(2,2), padding='SAME')"
STEM_SYM  = "params[0][2], stride=(2,2), padding=((1,1),(1,1)))"
BN_BATCH  = "axis=(0, 2, 3)"
BN_PEREX  = "axis=(2, 3)"

# ⚠⚠ THE THIRD AXIS, and the one nobody was looking for. The verified render's stem and head are
# **swish** (`EfficientNetRender.lean:722,729` — "stem … → bn → swish"), which is EfficientNet-B0 as
# PUBLISHED: the paper uses SiLU throughout. The JAX reference uses **relu** at both, not by
# design but because the generic `.convBn` layer emitter appends `jax.nn.relu(x)` unconditionally
# (`jax/Jax/Codegen.lean`, the `.convBn` case) and has no activation parameter. So here the RENDER
# is the paper-faithful side and the REFERENCE is the deviation — the opposite direction from the
# padding story. Patching these two lines is what lets the padding axis be read at all.
ACT_SITES = [
    ("params[0][2], stride=(2,2), padding='SAME')\n    x = jax.nn.relu(x)",
     "params[0][2], stride=(2,2), padding='SAME')\n    x = swish(x)"),
    ("params[80][2], padding='SAME')\n    x = jax.nn.relu(x)",
     "params[80][2], padding='SAME')\n    x = swish(x)"),
]

# The B0 block table, as `forward()` calls it: (expand, …). 16 blocks; only the first has expand=1
# (hence no expand conv, hence 4 entries instead of 5).
EXPANDS = [1] + [6] * 15


def parse_input_shapes(mlir_path, fn):
    txt = open(mlir_path).read()
    m = re.search(rf'func\.func @{re.escape(fn)}\((.*?)\)\s*->', txt, re.S)
    if not m:
        sys.exit(f"could not find func @{fn} in {mlir_path}")
    return re.findall(r'tensor<([0-9x]*)f32>', m.group(1))


def load_reference_forward(stem_sym=False, bn_perex=False, act_swish=False):
    src = open(REF_PY).read().split("\n")
    cut = next((i for i, l in enumerate(src) if l.startswith("def loss_fn")), None)
    if cut is None:
        sys.exit(f"{REF_PY}: no `def loss_fn` to cut at; the generator's shape changed")
    body = "\n".join(src[:cut])
    # ⚠ ORDER MATTERS: the activation patches anchor on the stem line INCLUDING its
    # `padding='SAME')` text, so they must run BEFORE the padding patch rewrites it. Their
    # replacements preserve that text, so the padding patch still finds its site afterwards.
    patches = [(True, o, n) for o, n in ACT_SITES] if act_swish else []
    patches += [(stem_sym, STEM_SAME, STEM_SYM), (bn_perex, BN_BATCH, BN_PEREX)]
    for want, old, new in patches:
        if not want:
            continue
        # ⚠ ACT sites are now ABSENT by design: `NetSpec.convBnAct` fixed the generator
        # (2026-08-08), so the reference already emits the right activation. Skip rather than
        # fail — this axis is kept only so the deviation can be re-created for comparison.
        if old not in body and any(old == o for o, _ in ACT_SITES):
            continue
        if old not in body:
            sys.exit(f"patch site not found — generator changed:\n  {old}")
        body = body.replace(old, new)
    mod = {}
    exec(body, mod)
    if "forward" not in mod:
        sys.exit("reference prefix did not define forward()")
    return mod["forward"]


def group_params(ws):
    """Render tensor order -> the reference's `params` list of tuples.

    Order per block is (expand?) triple, dw triple, se1 PAIR, se2 PAIR, project triple — which is
    exactly the render's argument order, so this walks both in lockstep. The SE pairs are the only
    entries that need a transform (matmul layout -> 1x1-conv layout)."""
    out, i = [], 0

    def take(n):
        nonlocal i
        g = ws[i:i + n]
        i += n
        return g

    out.append(tuple(take(3)))                                   # stem conv-BN
    for e in EXPANDS:
        if e > 1:
            out.append(tuple(take(3)))                           # expand 1x1 + BN
        out.append(tuple(take(3)))                               # depthwise + BN
        w1, b1 = take(2)                                         # SE reduce  [c,r] -> [r,c,1,1]
        out.append((w1.T[:, :, None, None], b1))
        w2, b2 = take(2)                                         # SE expand  [r,c] -> [c,r,1,1]
        out.append((w2.T[:, :, None, None], b2))
        out.append(tuple(take(3)))                               # project 1x1 + BN
    out.append(tuple(take(3)))                                   # head conv-BN
    wd, bd = take(2)                                             # dense [1280,K] -> ref [K,1280]
    out.append((wd.T, bd))
    if i != len(ws):
        sys.exit(f"grouping consumed {i} of {len(ws)} tensors — layout drifted")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir", default="verified_mlir/efficientnet_fwd.mlir")
    ap.add_argument("--fn", default="efficientnet_fwd")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--backend", default="llvm-cpu")
    ap.add_argument("--diag", action="store_true",
                    help="cross the padding and BN axes, so each is attributable")
    args = ap.parse_args()

    for p in (args.mlir, REF_PY):
        if not os.path.exists(p):
            sys.exit(f"{p} missing"
                     + ("" if p != REF_PY else " — run `cd jax && lake exe efficientnet-b0`"))

    work = tempfile.mkdtemp(prefix="enettie_")
    os.makedirs(f"{work}/in", exist_ok=True)
    shapes = parse_input_shapes(args.mlir, args.fn)
    batch = int(shapes[0].split("x")[0])
    print(f"func @{args.fn}: {len(shapes)} inputs, batch {batch}  (workdir {work})")

    rng = np.random.default_rng(args.seed)
    arrays, in_flags = [], []
    for i, s in enumerate(shapes):
        dims = [int(d) for d in s.split("x") if d]
        a = (np.asarray(rng.standard_normal(dims)).astype(np.float32) * args.scale)
        arrays.append(a)
        p = f"{work}/in/i{i}.npy"
        np.save(p, a)
        in_flags.append(f"--input=@{p}")

    cflags = ([f"--iree-hal-target-backends=rocm", f"--iree-rocm-target={CHIP}"]
              if args.backend == "rocm" else ["--iree-hal-target-backends=llvm-cpu"])
    print(f"  iree-compile ({args.backend}) …", flush=True)
    r = subprocess.run([IREE_C, *cflags, args.mlir, "-o", f"{work}/m.vmfb"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"iree-compile FAILED:\n{r.stderr[:3000]}")
    # ⚠ `local-sync`, NOT `local-task` — and this cost half an hour, so it is written down.
    # `local-task` (the multithreaded CPU device the mnv2/mnv4 ties use happily) dies on THIS
    # module with **exit 245, empty stderr and no output file**, after printing `EXEC
    # @efficientnet_fwd`. There is no error message on either stream. `local-sync` runs the same
    # vmfb to completion in ~12 MiB. So a silent 245 here is a device/threading problem, not a
    # problem with the render — do not go looking for a bug in the MLIR.
    dev = "hip" if args.backend == "rocm" else "local-sync"
    print("  iree-run-module …", flush=True)
    r = subprocess.run([IREE_R, f"--device={dev}", f"--module={work}/m.vmfb",
                        f"--function={args.fn}", *in_flags,
                        f"--output=@{work}/out.npy"], capture_output=True, text=True)
    if r.returncode != 0:
        # ⚠ Report the code, not just stderr. A NEGATIVE returncode is a signal (-9 = OOM-killed),
        # and it comes with EMPTY stderr — which reads exactly like "failed silently" and sends you
        # looking for a bug in the module that is not there. This net at batch 32 is the first in
        # the sweep big enough to hit it.
        sys.exit(f"iree-run-module FAILED (returncode {r.returncode}"
                 + (f", signal {-r.returncode}" if r.returncode < 0 else "") + ")\n"
                 + (r.stderr[:3000] or "  <no stderr — killed by signal, most likely OOM>"))
    got = np.load(f"{work}/out.npy").astype(np.float64)

    import jax.numpy as jnp
    x = arrays[0].reshape(batch, 3, 224, 224)
    params = group_params(arrays[1:])
    print(f"  reference params: {len(params)} entries "
          f"({sum(1 for p in params if len(p) == 3)} triples + "
          f"{sum(1 for p in params if len(p) == 2)} pairs)")
    if len(params) != 82:
        sys.exit(f"expected 82 entries (stem + 16 blocks + head + dense), got {len(params)}")

    def evalref(stem_sym=False, bn_perex=False, act_swish=False):
        f = load_reference_forward(stem_sym, bn_perex, act_swish)
        return np.asarray(f(params, jnp.asarray(x))).astype(np.float64)

    if args.diag:
        # Three axes now, all crossed. Sweeping one at a time attributes NOTHING when more than one
        # convention differs — MNv2 taught this and EfficientNet has one more than MNv2 did.
        for act in (False, True):
            for bn_perex in (False, True):
                lbl = (f"act {'swish' if act else 'relu '} (stem/head) · "
                       f"BN {'per-ex' if bn_perex else 'batch '}")
                print(f"  ── {lbl} ──")
                for plbl, s in (("stem pad as-is (SAME)", False), ("stem pad symmetric", True)):
                    dd = np.abs(got - evalref(s, bn_perex, act))
                    print(f"    {plbl:24s} max|Δ| {dd.max():.3e}   mean {dd.mean():.3e}")
        print("  ── deviations, reference vs itself (no render involved) ──")
        base = evalref(False, False, False)
        rng_ = max(base.max() - base.min(), 1e-12)
        for lbl, alt in (("stem padding SAME→symmetric", evalref(True, False, False)),
                         ("stem+head relu→swish      ", evalref(False, False, True))):
            dd = np.abs(base - alt)
            print(f"    {lbl}: max|Δ| {dd.max():.3e}  mean {dd.mean():.3e}"
                  f"  ({100*dd.max()/rng_:.1f}% of logit range)")

    want = evalref(False, False, False)
    if got.shape != want.shape:
        sys.exit(f"SHAPE MISMATCH: render {got.shape} vs reference {want.shape}")
    d = np.abs(got - want)
    print(f"  logits range (reference): [{want.min():.4f}, {want.max():.4f}]")
    print(f"  max |Δ|      : {d.max():.3e}")
    print(f"  mean |Δ|     : {d.mean():.3e}")
    if d.max() <= args.tol:
        print(f"  ✓ FORWARD TIE PASSES (tol {args.tol:.1e}) — the verified render computes the "
              f"published reference's function")
        return 0
    print(f"  ✗ FORWARD TIE FAILS (tol {args.tol:.1e}) against the PUBLISHED reference.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
