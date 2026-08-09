#!/usr/bin/env python3
"""MNv4 GRADIENT tie — the verified train step's gradient against `jax.grad` of the reference.

⭐ THIS IS PHASE 2'S GATE (`planning/mnv4_verified.md` §4). The forward tie pinned the block
ORDER at 1.423e-06; nothing so far has looked at the backward at all. Op counts, arities and the
forward-prefix check (`mnv4-train-smoke`) are all blind to a backward that differentiates an
ExtraDW block as an FFN, masks the swish site with `selectPos`, or contracts a weight gradient
against the wrong saved activation. Every one of those type-checks, keeps every shape, and
DESCENDS. §3g already caught exactly this class once — a backward pad derived by symmetry with its
siblings, right for two ops and wrong for the third, invisible to everything but a numeric check.

⭐⭐ HOW THE GRADIENT IS RECOVERED, since the train step returns updated parameters, not gradients:
AdamW's first moment is `m' = β₁·m + (1−β₁)·g`. Feed **m = 0** and it collapses to `m' = 0.1·g`
exactly — so `g = 10·m'`, read straight out of the returned m-slots with no optimizer inversion and
no dependence on lr, the bias corrections, or the weight decay (which is decoupled and touches only
θ'). Nothing about this is approximate.

The reference side is `jax.grad` of the LABEL-SMOOTHED cross-entropy:

    L = −(1/B) Σ_b Σ_k [(1−α)·onehot + α/K]_k · log_softmax(logits)_k       (α = 0.1)

⚠ NOT the reference file's own `loss_fn`, which is plain CE (`generated_mobilenet_v4.py:1120`).
The render's cotangent is smoothed — `dy = (softmax − onehot + α·onehot − α/K)/B` — and tying a
smoothed backward against an unsmoothed loss reports a real-looking mismatch on every parameter.
§2b shipped that exact confusion on ResNet-34 in the other direction (plain CE against a smoothed
cotangent) and only a numeric tie caught it, so the two are spelled together here.

Usage:
    lake build mnv4-train-smoke && .lake/build/bin/mnv4-train-smoke   # emits the batch-2 artifact
    scripts/mnv4_grad_tie.py
    scripts/mnv4_grad_tie.py --topk 20        # show the worst 20 parameters
"""
import argparse, os, re, subprocess, sys, tempfile
import numpy as np

REF_PY = "jax/.lake/build/generated_mobilenet_v4.py"
CHIP = os.environ.get("IREE_CHIP", "gfx1100")
IREE_C = ".venv/bin/iree-compile"
IREE_R = os.environ.get("IREE_RUN_MODULE",
    "/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/iree-run-module")
ALPHA = 0.1


def parse_io_shapes(mlir_path, fn):
    """Shapes AND names, both straight out of the emitted signature — so the diagnostic names a
    parameter the way the render does instead of an index nobody can map back to a block."""
    txt = open(mlir_path).read()
    m = re.search(rf'func\.func @{re.escape(fn)}\((.*?)\)\s*->\s*\(', txt, re.S)
    if not m:
        sys.exit(f"could not find func @{fn} in {mlir_path}")
    sig = m.group(1)
    ins = re.findall(r'tensor<([0-9x]*)f32>', sig)
    names = re.findall(r'(%[A-Za-z0-9_]+)\s*:\s*tensor<', sig)
    return ins, names


def block_of(nm):
    """`%u12dW` -> `u12`, `%f0cW` -> `f0`, `%sW` -> `stem`, `%hW`/`%Wd` -> `head`."""
    n = nm.lstrip("%")
    if n.startswith("u"):
        i = 1
        while i < len(n) and n[i].isdigit():
            i += 1
        return "u" + n[1:i]
    if n.startswith("f"):
        return "f0"
    if n in ("Wd", "bd") or n.startswith("h"):
        return "head"
    return "stem"


def load_reference_forward():
    """exec the reference's function prefix — the real generated code, minus its train loop."""
    src = open(REF_PY).read().split("\n")
    cut = next((i for i, l in enumerate(src) if l.startswith("def loss_fn")), None)
    if cut is None:
        sys.exit(f"{REF_PY}: no `def loss_fn` to cut at; the generator's shape changed")
    mod = {}
    exec("\n".join(src[:cut]), mod)
    if "forward" not in mod:
        sys.exit("reference prefix did not define forward()")
    return mod["forward"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir", default=".lake/build/mnv4_adam_train_step_b2.mlir")
    ap.add_argument("--fn", default="mnv4_adam2_train_step")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--nclasses", type=int, default=10)
    ap.add_argument("--nparams", type=int, default=158)
    ap.add_argument("--nstats", type=int, default=104)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="max RELATIVE error per parameter (‖Δ‖∞ / ‖g_ref‖∞)")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--ratio", type=float, default=10.0,
                    help="render's error vs f64 may be at most this x the perturbed reference's. "
                         "10x is not slack: the render is a DIFFERENT fp32 algorithm (different op "
                         "decomposition, reduction order and IREE fusion), so an order of "
                         "magnitude is ordinary. The margin is what makes it a gate — a real "
                         "wiring bug measured 1000-10000x here, three decades clear of this line.")
    ap.add_argument("--backend", default="llvm-cpu")
    ap.add_argument("--from-npz", default=None,
                    help="re-analyse a previous run's gradients instead of recomputing")
    ap.add_argument("--nokink", action="store_true",
                    help="set every BN beta=+5, gamma=1 so no pre-activation sits near 0 and the "
                         "relu masks are unambiguous — isolates WIRING from the relu knife-edge")
    ap.add_argument("--trace", action="store_true",
                    help="print EVERY parameter in forward order — the cotangent trace")
    args = ap.parse_args()

    if not os.path.exists(args.mlir):
        sys.exit(f"{args.mlir} missing — run `lake build mnv4-train-smoke && "
                 f".lake/build/bin/mnv4-train-smoke` to emit it")

    work = tempfile.mkdtemp(prefix="mnv4grad_")
    os.makedirs(f"{work}/in", exist_ok=True)
    shapes, argnames = parse_io_shapes(args.mlir, args.fn)
    nP, nS, B, K = args.nparams, args.nstats, args.batch, args.nclasses
    want_in = 1 + 3 * nP + 3 + nS + 1
    if len(shapes) != want_in:
        sys.exit(f"input arity {len(shapes)} != expected {want_in} "
                 f"(1 + 3x{nP} + 3 + {nS} + 1)")
    print(f"func @{args.fn}: {len(shapes)} inputs  (workdir {work})")

    rng = np.random.default_rng(args.seed)
    arrays = []
    for i, s in enumerate(shapes):
        dims = [int(d) for d in s.split("x") if d]
        if i == 0:                                   # %x
            a = rng.standard_normal(dims).astype(np.float32) * args.scale
        elif i < 1 + nP:                             # theta
            a = rng.standard_normal(dims).astype(np.float32) * args.scale
            # ⭐ --nokink: a ReLU net's gradient is a DISCONTINUOUS function of its forward, so at
            # any position whose pre-activation sits within the render-vs-reference forward gap of
            # zero the two masks disagree and the cotangent there differs by O(dy) — through no
            # fault of the backward. Pushing every BN beta to +5 puts every pre-activation ~5 sigma
            # clear of zero, making relu the identity and the comparison SMOOTH. What survives is a
            # test of the wiring alone: dispatch, saved-activation choice, strided placement, the
            # skip fan-in, the AdamW slot order. (It cannot catch masking with the WRONG tensor —
            # every candidate is all-positive here — so it is a complement to the default run,
            # not a replacement for it.)
            if args.nokink:
                nm = argnames[i].lstrip("%")
                if nm.endswith("bt"):
                    # +5 with gamma=1 puts every pre-activation ~5 sigma clear of zero.
                    # ⚠ A MIXED +-3 pattern was tried, to make the mask channel-specific and so
                    # catch masking with the wrong tensor — it does NOT work: at 3 sigma about 0.1%
                    # of positions still cross zero, which is ~130 kinks per layer, and the control
                    # itself degraded from 2e-5 to 2e-2. Kink-freedom needs the shift to be many
                    # sigma, and that necessarily makes every mask all-ones.
                    # The wrong-mask case is covered instead by the DEFAULT run: masking with the
                    # wrong tensor keeps or zeroes ~half of every layer's positions wrongly, which
                    # is an O(1) error, not the O(1e-2) knife-edge residual measured there.
                    a = np.full(dims, 5.0, np.float32)
                elif nm.endswith("g"):
                    a = np.ones(dims, np.float32)
        elif i < 1 + 3 * nP:                         # ⭐ m and v: ZERO. m=0 is what makes m'=0.1*g
            a = np.zeros(dims, np.float32)
        elif i < 1 + 3 * nP + 3:                     # lr, bc1, bc2 (scalars; irrelevant to m')
            a = np.array(1.0, np.float32) if i > 1 + 3 * nP else np.array(1e-3, np.float32)
        elif i < 1 + 3 * nP + 3 + nS:                # BN stat inputs: batch BN recomputes them
            a = np.zeros(dims, np.float32)
        else:                                        # %onehot
            lab = rng.integers(0, K, size=B)
            a = np.eye(K, dtype=np.float32)[lab]
        arrays.append(a)

    in_flags = []
    for i, a in enumerate(arrays):
        p = f"{work}/in/i{i}.npy"
        np.save(p, a)
        in_flags.append(f"--input=@{p}")

    # ── the verified train step ──
    cflags = ([f"--iree-hal-target-backends=rocm", f"--iree-rocm-target={CHIP}"]
              if args.backend == "rocm" else ["--iree-hal-target-backends=llvm-cpu"])
    print("  compiling...")
    r = subprocess.run([IREE_C, *cflags, args.mlir, "-o", f"{work}/m.vmfb"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"iree-compile FAILED:\n{r.stderr[:3000]}")
    outs = [f"--output=@{work}/o{j}.npy" for j in range(3 * nP + 3 + nS)]
    # ⚠ `--device=local-task` dies on big modules with **exit 245 and EMPTY stderr** — no output,
    # no diagnostic, right after printing the EXEC line. `planning/mnv4_verified.md` §3f hit this
    # on `efficientnet_fwd` and `local-sync` ran the identical vmfb fine. A silent 245 is a
    # device/threading problem, NOT a bad render — do not go hunting in the MLIR. So: try the
    # fast device, fall back, and SAY which one ran.
    devs = ["hip"] if args.backend == "rocm" else ["local-task", "local-sync"]
    r = None
    for dev in devs:
        print(f"  running on --device={dev} ...")
        r = subprocess.run([IREE_R, f"--device={dev}", f"--module={work}/m.vmfb",
                            f"--function={args.fn}", *in_flags, *outs],
                           capture_output=True, text=True)
        if r.returncode == 0:
            break
        print(f"    (rc {r.returncode}{', empty stderr' if not r.stderr.strip() else ''})")
    if r.returncode != 0:
        sys.exit(f"iree-run-module FAILED on every device (rc {r.returncode}):\n{r.stderr[:3000]}")
    got_g = [np.load(f"{work}/o{j}.npy").astype(np.float64) / (1.0 - 0.9)
             for j in range(nP, 2 * nP)]
    got_loss = float(np.load(f"{work}/o{3 * nP}.npy"))

    # ── the reference gradient, same weights, same smoothed loss ──
    import jax, jax.numpy as jnp
    # ⭐⭐ THE f64 CONTROL (§3h's rule: a probe with no control cannot convict anything).
    # Both sides here are fp32, and a 52-layer batch-BN BACKWARD is badly conditioned — two
    # legitimate fp32 evaluations of it need not agree to 1e-3. So the reference is also evaluated
    # in float64, and the render's error is reported ALONGSIDE the fp32 reference's own error
    # against that. If they are the same size, the gap is arithmetic and not wiring, and judging
    # the render against the fp32 reference would be convicting it of the reference's noise.
    jax.config.update("jax_enable_x64", True)
    fwd = load_reference_forward()
    x = arrays[0].reshape(B, 3, 224, 224)
    ws = arrays[1:1 + nP]
    params = [tuple(jnp.asarray(w) for w in ws[i:i + 3]) for i in range(0, nP - 2, 3)]
    params.append((jnp.asarray(ws[-2].T), jnp.asarray(ws[-1])))   # dense: [1280,K] -> ref's [K,1280]
    onehot = jnp.asarray(arrays[-1])
    print(f"  reference params: {len(params)} entries "
          f"({sum(1 for p in params if len(p) == 3)} triples + dense)")

    def smoothed_loss(ps):
        logits = fwd(ps, jnp.asarray(x))
        lp = jax.nn.log_softmax(logits, axis=-1)
        tgt = (1.0 - ALPHA) * onehot + ALPHA / K
        return -jnp.mean(jnp.sum(lp * tgt, axis=-1))

    print("  jax.grad (fp32) ...")
    gref = jax.grad(smoothed_loss)(params)
    print("  jax.grad (fp64 control) ...")
    params64 = [tuple(jnp.asarray(np.asarray(w, np.float64)) for w in t) for t in params]
    x64 = jnp.asarray(x.astype(np.float64))
    onehot64 = jnp.asarray(np.asarray(onehot, np.float64))
    def smoothed_loss64(ps):
        logits = fwd(ps, x64)
        lp = jax.nn.log_softmax(logits, axis=-1)
        tgt = (1.0 - ALPHA) * onehot64 + ALPHA / K
        return -jnp.mean(jnp.sum(lp * tgt, axis=-1))
    gref64 = jax.grad(smoothed_loss64)(params64)
    # ⭐⭐⭐ THE PERTURBATION CONTROL — the one that makes this tie interpretable.
    #
    # The render and the reference are two different fp32 ALGORITHMS whose forwards agree only to
    # ~1e-6 (the forward tie). This net is a ReLU net: at any position whose pre-activation sits
    # within that 1e-6 of zero, the two forwards disagree about the SIGN, so `1[x>0]` flips and the
    # cotangent at that position changes by O(dy). A gradient is a DISCONTINUOUS function of the
    # forward here, and no amount of correct backward code removes that.
    #
    # So the floor for "how close can the render's gradient possibly get" is not fp32 arithmetic —
    # it is the gradient's own sensitivity to a 1e-6 forward perturbation. Measure that directly:
    # nudge the reference's input by a relative 1e-7, re-take the gradient, and report how far THAT
    # legitimate evaluation lands from f64 truth. The render must be no worse.
    pk = np.random.default_rng(args.seed + 1).standard_normal(x.shape).astype(np.float32)
    xp = jnp.asarray((x * (1.0 + 1e-7 * pk)).astype(np.float32))
    def smoothed_loss_pert(ps):
        logits = fwd(ps, xp)
        lp = jax.nn.log_softmax(logits, axis=-1)
        tgt = (1.0 - ALPHA) * onehot + ALPHA / K
        return -jnp.mean(jnp.sum(lp * tgt, axis=-1))
    gpert = jax.grad(smoothed_loss_pert)(params)
    dloss_pert = abs(float(smoothed_loss_pert(params)) - float(smoothed_loss(params)))
    print(f"  perturbation control: input nudged 1e-7 relative -> loss moved {dloss_pert:.3e} "
          f"(render-vs-ref loss gap was {abs(got_loss - float(smoothed_loss(params))):.3e})")
    # flatten the reference gradient back into the render's parameter order
    want_g = []
    for i in range(nP - 2):
        want_g.append(np.asarray(gref[i // 3][i % 3], dtype=np.float64))
    want_g.append(np.asarray(gref[-1][0], dtype=np.float64).T)     # dense W back to [1280,K]
    want_g.append(np.asarray(gref[-1][1], dtype=np.float64))
    ctrl_g = []
    for i in range(nP - 2):
        ctrl_g.append(np.asarray(gref64[i // 3][i % 3], dtype=np.float64))
    ctrl_g.append(np.asarray(gref64[-1][0], dtype=np.float64).T)
    ctrl_g.append(np.asarray(gref64[-1][1], dtype=np.float64))
    pert_g = []
    for i in range(nP - 2):
        pert_g.append(np.asarray(gpert[i // 3][i % 3], dtype=np.float64))
    pert_g.append(np.asarray(gpert[-1][0], dtype=np.float64).T)
    pert_g.append(np.asarray(gpert[-1][1], dtype=np.float64))

    # ── compare ──
    np.savez(f"{work}/grads.npz", got=np.array([g.ravel() for g in got_g], dtype=object),
             want=np.array([np.asarray(w).ravel() for w in want_g], dtype=object),
             ctrl=np.array([np.asarray(c).ravel() for c in ctrl_g], dtype=object),
             names=np.array(argnames[1:1 + nP]), loss=np.array([got_loss]),
             allow_pickle=True)
    print(f"  (gradients cached at {work}/grads.npz — re-analyse with --from-npz)")

    # ── ⭐ FORWARD CROSS-CHECK FIRST. `%loss` is the train step's OWN forward, evaluated
    #    through the very graph that produced these gradients. If it disagrees with the reference
    #    loss, the problem is upstream of the backward and every gradient row below is noise.
    ref_loss = float(smoothed_loss(params))
    lrel = abs(got_loss - ref_loss) / max(abs(ref_loss), 1e-30)
    print(f"\n  %loss (render) {got_loss:.8f}   reference {ref_loss:.8f}   rel {lrel:.3e}")
    if lrel > 1e-4:
        print("  ⚠ THE FORWARD INSIDE THE TRAIN STEP DISAGREES — fix that before reading any"
              " gradient row;\n    the backward is being blamed for an upstream difference.")

    names = argnames[1:1 + nP]
    rows = []
    for i, (g, w) in enumerate(zip(got_g, want_g)):
        g = g.reshape(-1); w = np.asarray(w).reshape(-1)
        if g.shape != w.shape:
            sys.exit(f"param {i} ({names[i]}): shape {g.shape} vs reference {w.shape}")
        c = np.asarray(ctrl_g[i]).reshape(-1)
        pt = np.asarray(pert_g[i]).reshape(-1)
        sc = max(np.abs(c).max(), 1e-30)
        # all three measured against the SAME f64 truth, so they are directly comparable
        rows.append((np.abs(g - c).max() / sc, np.abs(pt - c).max() / sc, sc, i,
                     np.abs(w - c).max() / sc))
    rows.sort(reverse=True)
    print(f"\n  worst {args.topk} parameters, all vs the f64 control:")
    print(f"    {'param':<8s} {'RENDER':>11s} {'PERTURBED-ref':>14s} {'fp32-ref':>11s}   render/pert")
    for rel, pertrel, sc, i, refrel in rows[:args.topk]:
        print(f"    {names[i]:<8s} {rel:11.3e} {pertrel:14.3e} {refrel:11.3e}   "
              f"{rel / max(pertrel, 1e-30):8.2f}x")

    if args.trace:
        # ⭐ THE COTANGENT TRACE. A BN's β gradient is exactly `Σ dy` over that BN's output
        # cotangent — no weight contraction in the way — so reading the `*bt` rows in FORWARD
        # order shows precisely where the cotangent stops matching. Weights inherit whatever the
        # cotangent already is, so they blur the boundary; β does not.
        by_i = {i: (rel, pertrel, sc) for rel, pertrel, sc, i, _ in rows}
        print("\n  cotangent trace (forward order; `bt` rows are the clean readout):")
        for i, nm in enumerate(names):
            rel, refrel, sc = by_i[i]
            dead = " (dead: |g|~0)" if sc < 1e-7 else ""
            mark = "  <<<" if sc >= 1e-7 and rel > max(args.ratio * refrel, args.tol) else ""
            print(f"    {i:3d} {nm:<9s} render {rel:.3e}  ref {refrel:.3e}{dead}{mark}")

    # ── per-block rollup: the localiser. A whole block failing while its neighbours pass names
    #    the family; a smooth gradient of error with depth is accumulation, not wiring.
    agg = {}
    for rel, pertrel, sc, i, refrel in rows:
        b = block_of(names[i])
        cur = agg.get(b, (0.0, 0.0, 0))
        agg[b] = (max(cur[0], rel), max(cur[1], pertrel), cur[2] + 1)
    order = ["stem", "f0"] + [f"u{i}" for i in range(1, 15)] + ["head"]
    print("\n  per-block worst error vs the f64 control:")
    print(f"    {'block':<6s} {'RENDER':>11s} {'PERT-ref':>11s}   ratio")
    for b in order:
        if b in agg:
            w, wr, tot = agg[b]
            print(f"    {b:<6s} {w:11.3e} {wr:11.3e}   {w / max(wr, 1e-30):6.2f}x")
    # ── STRUCTURALLY-DEAD parameters, excluded from the RELATIVE verdict and checked absolutely ──
    # A BN's beta gradient is `Sum_{b,h,w} dy`. The BN backward's own output has zero per-channel
    # sum by construction (its three-term formula subtracts the mean), and a 1x1 conv-back preserves
    # that (`Sum dx[c] = Sum_c' W[c',c] Sum dy[c'] = 0`). So any BN whose cotangent arrives through
    # that path has a beta gradient of EXACTLY zero, and a relative error on it is 0/0 — the two
    # `%u14pbt`-shaped rows that dominate every table above are all of this kind. They still get
    # checked, just absolutely: the render must also produce ~0.
    med = float(np.median([r[2] for r in rows]))
    dead = [r for r in rows if r[2] < 1e-3 * med]
    live = [r for r in rows if r[2] >= 1e-3 * med]
    if dead:
        worst_dead = max(np.abs(np.asarray(got_g[r[3]]).ravel()).max() for r in dead)
        print(f"\n  structurally-dead parameters (reference gradient == 0): {len(dead)}")
        print(f"    {', '.join(names[r[3]] for r in dead)}")
        print(f"    worst |render gradient| there: {worst_dead:.3e}  "
              f"(typical live scale {med:.3e}) -> {'✓ also ~0' if worst_dead < 1e-3 * med else '✗ NOT zero'}")

    worst = rows[0][0]
    worst_ref = max(r[1] for r in rows)
    # ⭐ THE VERDICT IS RELATIVE TO THE CONTROL, not to an absolute tolerance: the render must be
    # no worse than `--ratio`x the reference's OWN fp32 error against f64 truth. An absolute
    # threshold here would either convict the render of the reference's conditioning or, set loose
    # enough to pass, stop being a gate at all.
    nbad = sum(1 for r in live if r[0] > max(args.ratio * r[1], args.tol))
    print(f"\n  worst RENDER error vs f64   : {worst:.3e}")
    print(f"  worst PERTURBED-ref vs f64  : {worst_ref:.3e}   (the relu-discontinuity floor)")
    print(f"  live parameters worse than {args.ratio}x the PERTURBED reference: {nbad}/{len(live)}")

    # ── VACUITY GUARD (§3h's lesson: a green tie on a degenerate probe is worse than none) ──
    # If the reference gradient is ~0 everywhere the comparison says nothing. Require that a
    # healthy majority of parameters carry a gradient well clear of zero.
    print(f"  parameters with a non-trivial reference gradient: {len(live)}/{nP}")
    if len(live) < nP * 0.9:
        print("  ✗ VACUOUS: most reference gradients are ~0, so this tie measures nothing.")
        return 2

    if nbad == 0:
        mode = ("with the relu kinks removed (--nokink)" if args.nokink
                else "on the raw net")
        print(f"\n  ✓ GRADIENT TIE PASSES {mode} — the verified backward computes the "
              f"reference's\n    gradient to within the reference's own fp32 accuracy: family "
              f"dispatch, activation\n    VJPs, strided depthwise placement and the AdamW slot "
              f"order all included")
        if not args.nokink:
            return 0
        print("\n  ⚠ --nokink cannot catch masking with the WRONG tensor (every candidate is "
              "all-positive\n    under it). Run the default mode too and read its residual as the "
              "knife-edge bound.")
        return 0
    print(f"\n  ✗ GRADIENT TIE FAILS")
    if not args.nokink:
        print("    ⚠ BEFORE suspecting the wiring, re-run with --nokink. A ReLU net's gradient is")
        print("    DISCONTINUOUS in its forward: where a pre-activation sits within the forward")
        print("    tie's ~1e-6 of zero, the two masks disagree and that channel's beta/gamma")
        print("    gradient moves by O(dy). One such position per layer reproduces exactly the")
        print("    'one bad channel per BN, growing with depth' pattern.")
    print("    A block whose parameters ALL fail while its neighbours pass localises the family;")
    print("    scattered small failures are more likely fp accumulation than a wiring bug.")
    return 1


def _shape_names():
    return []


if __name__ == "__main__":
    sys.exit(main())
