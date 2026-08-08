#!/usr/bin/env python3
"""MNv2 forward tie — the verified render against the JAX reference, on SHARED weights.

⭐ WHY THIS EXISTS. `planning/mnv4_verified.md` §3c inferred, by READING `sep_conv` and
`depthwise_conv`, that MobileNetV2's verified render is exposed to the same padding defect the
MNv4 tie measured — and at FIVE sites rather than one. Inference is not measurement. This is the
measurement, and it decides whether the asymmetric-pad descriptor has one consumer or three.

The defect, precisely:
  * `VLayer` has zero occurrences of `Padding`, so EVERY verified render emits symmetric padding.
    `verified_mlir/mobilenetv2_fwd.mlir` has 5 stride-2 convolutions, all `pad = [[1,1],[1,1]]`.
  * The reference's stem is `conv_bn(..., stride=(2,2), padding='SAME')`, and its four strided
    depthwises inherit `depthwise_conv`'s default `padding='SAME'` (`jax/Jax/Codegen.lean:679`).
    XLA 'SAME' on a 3x3/s2 at an EVEN input pads (0,1) — asymmetric.
  * Both conventions yield the same output size, so no shape check, no `#guard`, no op count and
    no arity audit anywhere can see the difference. Only running both sides on one set of weights
    can. Same invisible class as R50's stride-on-the-3x3 and the 2x2 stem pool.

⚠ WHAT A FAILURE HERE DOES AND DOES NOT MEAN. It does NOT invalidate MNv2's §1a tie — that tie is
Lean-level (spec ↔ render) and still holds: the render faithfully implements the
`VerifiedNetSpec`. What it puts in doubt is whether that SPEC is MobileNetV2-as-published at the
strided convs. Both can be true, and only the second is in question. The two claims are easy to
blur and the blur favours us, so state them separately.

`--diag` attributes the disagreement by patching the reference's two padding sites independently:

    as-is        both sites XLA 'SAME'          (the published net)
    stem         stem symmetric, depthwises SAME
    dw           stem SAME, depthwises symmetric
    both         both symmetric                 (what the render computes)

If `both` ties and nothing else does, padding is the ENTIRE disagreement and the block structure
is confirmed — which is the same shape of result the MNv4 tie produced at 1.8e-6.

⚠ Batch is FIXED at 32 by the render (`%x : tensor<32x150528xf32>`) and the forward uses BATCH
BN, so the batch cannot be shrunk to make this cheaper — the statistics are over all 32.

Usage:
    scripts/mnv2_forward_tie.py --diag
    scripts/mnv2_forward_tie.py --diag --backend rocm
"""
import argparse, os, re, subprocess, sys, tempfile
import numpy as np

REF_PY = "jax/.lake/build/generated_mobilenet_v2.py"
CHIP = os.environ.get("IREE_CHIP", "gfx1100")
IREE_C = ".venv/bin/iree-compile"
# ⚠ iree-run-module is NOT in this repo's .venv (only iree-compile is). It ships with the
# lean4-jax venv — the same absolute path the PJRT plugin resolves through.
IREE_R = os.environ.get("IREE_RUN_MODULE",
    "/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/iree-run-module")

# ── the two reference sites that disagree with the render, patched independently ──
STEM_SAME = "params[0][2], stride=(2,2), padding='SAME')"
STEM_SYM  = "params[0][2], stride=(2,2), padding=((1,1),(1,1)))"

DW_SAME = ("def depthwise_conv(x, w, stride=(1,1), padding='SAME'):\n"
           "    return jax.lax.conv_general_dilated(")
DW_SYM  = ("def depthwise_conv(x, w, stride=(1,1), padding=None):\n"
           "    if padding is None:\n"
           "        padding = (((w.shape[2] - 1) // 2,) * 2, ((w.shape[3] - 1) // 2,) * 2)\n"
           "    return jax.lax.conv_general_dilated(")

# ⚠ THE SECOND AXIS, and the bigger one. `verified_mlir/mobilenetv2_fwd.mlir` reduces its BN
# statistics across `dimensions = [2, 3]` and divides by H·W — PER-EXAMPLE BN, instance-norm
# shaped. The reference reduces `axis=(0, 2, 3)` over B·H·W — true BATCH BN. This is the
# documented two-worlds split (memory `r34-bn-two-worlds`: it runs along the WRITER, not the
# net), and it dwarfs the padding difference, so padding cannot be attributed until BN is
# matched. `global_avg_pool` is already `axis=(2, 3)` and is left alone by this substring.
BN_BATCH   = "axis=(0, 2, 3)"
BN_PEREX   = "axis=(2, 3)"
# ⚠⚠ THE THIRD DIFFERENCE, and it is INVISIBLE at small weights. The verified render applies
# **relu6** at the stem and head; the reference applies plain **relu** — because the generic
# `.convBn` layer emitter appends `jax.nn.relu(x)` unconditionally and has no activation parameter
# (`jax/Jax/Codegen.lean`, the `.convBn` case). MobileNetV2 as published is ReLU6 throughout, so
# here the RENDER is the paper-faithful side and the REFERENCE deviates — exactly the same defect
# EfficientNet has with swish (`planning/mnv4_verified.md` §3f), from the same emitter.
#
# ⭐ Why the original --diag tie still passed at 6.08e-06 without this: at `--scale 0.1` the
# activations never reach 6, so relu6 and relu AGREE pointwise. The deviation only becomes visible
# under He init, which is what the --eval probe needs. A convention difference that is inert on
# your test inputs is still a difference in the net.
ACT_SITES = [
    ("params[0][2], stride=(2,2), padding='SAME')\n    x = jax.nn.relu(x)",
     "params[0][2], stride=(2,2), padding='SAME')\n    x = jnp.minimum(jax.nn.relu(x), 6.0)"),
    ("params[51][2], padding='SAME')\n    x = jax.nn.relu(x)",
     "params[51][2], padding='SAME')\n    x = jnp.minimum(jax.nn.relu(x), 6.0)"),
]

# Frozen-stat BN, to match `@mobilenetv2_fwd_eval` fed mu=0 / var=1. Replaces the reduction
# entirely, so no batch statistic of any kind enters and the BN axis stops being a variable.
BN_FROZEN  = [("mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)", "mean = 0.0"),
              ("var = jnp.var(x, axis=(0, 2, 3), keepdims=True)",   "var = 1.0")]

PAD_VARIANTS = [
    ("pad as-is (XLA 'SAME')",  False, False),
    ("pad stem symmetric",      True,  False),
    ("pad dw symmetric",        False, True),
    ("pad both symmetric",      True,  True),
]


def parse_input_shapes(mlir_path, fn):
    txt = open(mlir_path).read()
    m = re.search(rf'func\.func @{re.escape(fn)}\((.*?)\)\s*->', txt, re.S)
    if not m:
        sys.exit(f"could not find func @{fn} in {mlir_path}")
    return re.findall(r'tensor<([0-9x]*)f32>', m.group(1))


def load_reference_forward(stem_sym, dw_sym, bn_perex=False, bn_frozen=False):
    """exec the reference's function prefix — the real generated code, minus its train loop.

    The module is not importable: its training loop runs at module level and wants
    `data/imagenette/train.bin`. We exec only the prefix up to `def loss_fn`, which is every
    function `forward` needs and none of the loop."""
    src = open(REF_PY).read().split("\n")
    cut = next((i for i, l in enumerate(src) if l.startswith("def loss_fn")), None)
    if cut is None:
        sys.exit(f"{REF_PY}: no `def loss_fn` to cut at; the generator's shape changed")
    body = "\n".join(src[:cut])
    patches = [(stem_sym, STEM_SAME, STEM_SYM), (dw_sym, DW_SAME, DW_SYM),
               (bn_perex, BN_BATCH, BN_PEREX)]
    if bn_frozen:
        # ⚠ ACT_SITES must precede the padding patch: they anchor on the stem line INCLUDING its
        # `padding='SAME')`, which the padding patch rewrites. Same ordering trap as the enet tie.
        patches = ([(True, o, n) for o, n in ACT_SITES]
                   + [(True, o, n) for o, n in BN_FROZEN] + patches[:2])
    for want, old, new in patches:
        if not want:
            continue
        if old not in body:
            sys.exit("patch site not found — the reference generator's shape changed:\n"
                     f"  {old.splitlines()[0]}")
        body = body.replace(old, new)
    mod = {}
    exec(body, mod)
    if "forward" not in mod:
        sys.exit("reference prefix did not define forward()")
    return mod["forward"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir", default="verified_mlir/mobilenetv2_fwd.mlir")
    ap.add_argument("--fn", default="mobilenetv2_fwd")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=1e-4, help="max |Δ| over the logits")
    ap.add_argument("--backend", default="llvm-cpu", help="llvm-cpu (portable) or rocm")
    ap.add_argument("--eval", action="store_true",
                    help="tie @mobilenetv2_fwd_eval instead — the artifact the ADAM run scores "
                         "with, and the one that carries the new XLA-SAME padding")
    ap.add_argument("--diag", action="store_true",
                    help="evaluate all four padding variants, to attribute the disagreement")
    args = ap.parse_args()

    # ⭐ EVAL MODE — how the NEW net gets tied without touching stat ordering.
    # After the 2026-08-08 switch, `@mobilenetv2_fwd` (per-example BN, SYMMETRIC pad) and
    # `@mobilenetv2_fwd_eval` (frozen-stat BN, XLA-SAME pad) are deliberately different nets: the
    # first partners the SGD train step, the second the Adam one. Only the second is the net the
    # 89.35% re-run trains and scores, so only the second is worth tying.
    # The trick that makes it cheap: feed mu=0, var=1 for all 104 stat slots and patch the
    # reference's BN to the same constants. Both sides then compute one deterministic function and
    # padding is the ONLY thing left that could differ — and no stat SLOT ORDERING is exercised,
    # which is exactly the silent failure mode this script must not depend on being right.
    if args.eval:
        # ⚠ only DEFAULT the path — an explicit --mlir must win, or the control that runs the
        # pre-switch artifact silently re-runs the new one and "confirms" whatever you hoped.
        if args.mlir == "verified_mlir/mobilenetv2_fwd.mlir":
            args.mlir = "verified_mlir/mobilenetv2_fwd_eval.mlir"
        args.fn = "mobilenetv2_fwd_eval"
    if not os.path.exists(args.mlir):
        sys.exit(f"{args.mlir} missing")
    if not os.path.exists(REF_PY):
        sys.exit(f"{REF_PY} missing — run `cd jax && lake exe mobilenet-v2` (it will fail at "
                 f"the training loop for want of data/imagenette; the .py is written first)")

    work = tempfile.mkdtemp(prefix="mnv2tie_")
    os.makedirs(f"{work}/in", exist_ok=True)
    shapes = parse_input_shapes(args.mlir, args.fn)
    batch = int(shapes[0].split("x")[0])       # the render pins it; batch BN forbids changing it
    print(f"func @{args.fn}: {len(shapes)} inputs, batch {batch}  (workdir {work})")

    rng = np.random.default_rng(args.seed)
    nStat = 104 if args.eval else 0            # 52 BN layers x (mu, var), the LAST inputs
    firstStat = len(shapes) - nStat
    arrays, in_flags = [], []
    for i, s in enumerate(shapes):
        dims = [int(d) for d in s.split("x") if d]
        if args.eval and i >= firstStat:
            # interleaved mu, var per layer -> 0.0 and 1.0, so eval BN is a pure affine
            a = np.full(dims, 0.0 if (i - firstStat) % 2 == 0 else 1.0, dtype=np.float32)
        elif args.eval and i >= 1:
            # ⚠⚠ EVAL MODE NEEDS HE INIT, NOT A FLAT SCALE — and this is the difference between a
            # real probe and a vacuous one. With frozen var=1 the BN does no renormalisation, so a
            # flat 0.1 scale makes every layer shrink the signal; after 52 layers the trunk is
            # ~1e-50 and the logits are pure bias. The FIRST version of this probe did exactly
            # that and reported ~1e-7 for every padding variant — a green tie that measured
            # nothing. Here: gamma=1, beta=0 (so frozen BN is the identity) and conv weights at
            # sqrt(2/fan_in), which is norm-preserving and keeps the trunk alive end to end.
            k = (i - 1) % 3
            if len(dims) == 1 and k == 1:
                a = np.ones(dims, dtype=np.float32)                       # gamma
            elif len(dims) == 1 and k == 2:
                a = np.zeros(dims, dtype=np.float32)                      # beta
            elif len(dims) == 1:
                a = np.zeros(dims, dtype=np.float32)                      # dense bias
            else:
                fan_in = int(np.prod(dims[1:])) if len(dims) > 1 else dims[0]
                a = (np.asarray(rng.standard_normal(dims)).astype(np.float32)
                     * np.float32(np.sqrt(2.0 / max(fan_in, 1))))
        else:
            a = (np.asarray(rng.standard_normal(dims)).astype(np.float32) * args.scale)
        arrays.append(a)
        p = f"{work}/in/i{i}.npy"
        np.save(p, a)
        in_flags.append(f"--input=@{p}")

    # ── the verified render ──
    cflags = ([f"--iree-hal-target-backends=rocm", f"--iree-rocm-target={CHIP}"]
              if args.backend == "rocm" else ["--iree-hal-target-backends=llvm-cpu"])
    print(f"  iree-compile ({args.backend}) …", flush=True)
    r = subprocess.run([IREE_C, *cflags, args.mlir, "-o", f"{work}/m.vmfb"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"iree-compile FAILED:\n{r.stderr[:3000]}")
    # ⚠ `local-sync`, not `local-task`: the multithreaded CPU device dies on the larger modules
    # here with a nonzero code and EMPTY stderr (first seen on `efficientnet_fwd`, exit 245;
    # `mobilenetv2_fwd_eval` does it too). It is a device/threading problem, not a bad render.
    dev = "hip" if args.backend == "rocm" else "local-sync"
    print(f"  iree-run-module …", flush=True)
    r = subprocess.run([IREE_R, f"--device={dev}", f"--module={work}/m.vmfb",
                        f"--function={args.fn}", *in_flags,
                        f"--output=@{work}/out.npy"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"iree-run-module FAILED (returncode {r.returncode})\n"
                 + (r.stderr[:3000] or "  <no stderr — killed by signal or device fault>"))
    got = np.load(f"{work}/out.npy").astype(np.float64)

    # ── the reference, on the same weights ──
    import jax.numpy as jnp
    x = arrays[0].reshape(batch, 3, 224, 224)                # %x is flat [B, 3*224*224]
    ws = arrays[1:firstStat] if args.eval else arrays[1:]
    params = [tuple(ws[i:i + 3]) for i in range(0, len(ws) - 2, 3)]  # 52 (W, γ, β) triples
    params.append((ws[-2].T, ws[-1]))            # dense: [1280,K] -> reference's [K,1280]
    ntri = sum(1 for p in params if len(p) == 3)
    print(f"  reference params: {len(params)} entries ({ntri} triples + dense)")
    if ntri != 52:
        sys.exit(f"expected 52 triples (stem + 17 invres + head), got {ntri} — layout drifted")

    def evalref(stem_sym, dw_sym, bn_perex=False):
        f = load_reference_forward(stem_sym, dw_sym, bn_perex, bn_frozen=args.eval)
        return np.asarray(f(params, jnp.asarray(x))).astype(np.float64)

    if args.diag:
        # Two axes, crossed. Attributing padding requires BN to be matched first — with the
        # wrong BN world every padding row is dominated by the same large residual and the
        # table says nothing. Cross them so which axis carries the disagreement is visible.
        for bn_perex, bnlbl in ((False, "BN batch  (0,2,3) — the reference's own"),
                                (True,  "BN per-ex (2,3)   — what the render computes")):
            print(f"  ── {bnlbl} ──")
            for lbl, s, d in PAD_VARIANTS:
                w = evalref(s, d, bn_perex)
                dd = np.abs(got - w)
                print(f"    {lbl:26s} max|Δ| {dd.max():.3e}   mean {dd.mean():.3e}")

        # ── padding cost, reference against ITSELF ──
        # The render above is per-example BN, but the artifact that TRAINED the quoted Imagenette
        # number (`mobilenetv2_adam_train_step.mlir`) is BATCH BN with the same symmetric padding.
        # So the size of the padding deviation in the batch-BN world cannot be read off the rows
        # above. Measuring it reference-against-reference needs no render and answers exactly
        # "how far is the net that was trained from the net that was published".
        print("  ── padding deviation, reference vs itself (no render involved) ──")
        for bn_perex, bnlbl in ((False, "batch BN "), (True, "per-ex BN")):
            a = evalref(False, False, bn_perex)      # published: XLA 'SAME'
            b = evalref(True,  True,  bn_perex)      # what the render/trainer computes
            dd = np.abs(a - b)
            rng_ = max(a.max() - a.min(), 1e-12)
            print(f"    {bnlbl}  SAME vs symmetric: max|Δ| {dd.max():.3e}   "
                  f"mean {dd.mean():.3e}   ({100*dd.max()/rng_:.1f}% of logit range)")

    want = evalref(False, False)                              # the PUBLISHED net
    if args.diag:
        a, b = evalref(False, False), evalref(True, True)
        span = max(a.max() - a.min(), 1e-30)
        if np.abs(a - b).max() / span < 1e-4:
            sys.exit("✗ VACUOUS PROBE: the two padding conventions are indistinguishable on these\n"
                     "  weights, so a PASS here would mean nothing. Almost always the signal has\n"
                     "  decayed (see the He-init note above) — fix the weights, not the tolerance.")
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
              f"the published reference's function")
        return 0
    print(f"  ✗ FORWARD TIE FAILS (tol {args.tol:.1e}) against the PUBLISHED reference.")
    print("    Run with --diag: if the `both symmetric` row ties, padding is the whole story "
          "and the block structure is confirmed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
