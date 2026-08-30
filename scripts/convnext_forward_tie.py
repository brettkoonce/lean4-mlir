#!/usr/bin/env python3
"""ConvNeXt-T forward tie — the verified render against the JAX reference, on SHARED weights.

The fourth of these, and the one whose absence is on the record. `enet_forward_tie.py`,
`mnv2_forward_tie.py` and `mnv4_forward_tie.py` existed; ConvNeXt had no peer, so nothing in the
repo compared its two phases numerically. That is a real part of how **B1** survived: the paper's
head LayerNorm (`GAP → LN → Linear`) was deleted from the verified render by §2m to match a JAX
reference that was itself missing it, and both sides then agreed with each other for weeks. A
forward tie does not care whether the two sides agree with the paper — it cares whether they agree
with *each other*, which is exactly the thing a shared defect defeats. **So read what this gate
does and does not buy** (see the ⚠⚠ below) before treating a green as fidelity.

⭐⭐ **IT TIES THE IMAGENET PAIR, AND THAT IS THE WHOLE POINT OF THE FILE.**
`scripts/enet_forward_tie.py` ties `efficientnet_fwd` against `generated_efficientnet_b0.py` —
the **Imagenette** twin — while the defect that mattered (`convBnAct` unset, stem/head trained
ReLU where the net is swish) was in `generated_efficientnet_b0_imagenet.py`, the twin that gate
never looks at. A tie that green-lights the pair that was already correct is worse than no tie: it
reads as coverage. Here both sides are the 1000-class ImageNet artifacts, by default and on
purpose, and `--net` only ever selects another matched pair.

⚠⚠ **WHAT A GREEN HERE MEANS.** It means the two lowerers compute the same function from the same
weights: layer order, padding, the LN axis, the head, the parameter LAYOUT. It does **not** mean
either side matches the paper — B1 was a defect this gate would have stayed green through until
one side was fixed, at which point it goes red and names the disagreement. The external check for
"is the architecture right at all" is the parameter count against timm
(`sum(p.numel() for p in timm.create_model('convnext_tiny', num_classes=1000).parameters())` =
28,589,128), which is what actually caught B1. Run both; they fail in different directions.

**Mechanism.** IREE compiles `verified_mlir/convnextin_fwd.mlir` to `llvm-cpu` and runs it on
random weights; the same arrays are regrouped into the reference's `params` list and pushed through
the generated `forward()`. CPU on both sides — no GPU, so this is safe to run beside a training job,
and there is no device nondeterminism in the comparison.

⚠ **Batch is 32 and cannot be shrunk.** `ConvNeXtRender.cBS` pins it, and the artifact's `%x` is
`tensor<32x150528xf32>`. Unlike the BN nets there is no statistical reason it must be 32 — ConvNeXt
normalises with LayerNorm, which never reduces over the batch — it is simply what the render emits.

⚠ **The parameter REGROUPING is the part that can silently lie.** The render's signature is 182
flat tensors; the reference wants 100 tuples. The map below is built from the same `[3,3,9,3]`
structure both sides walk, and the entry count is asserted against the reference's own last index
rather than hardcoded — a regrouping that is off by one tuple produces a shape error most of the
time and a *wrong but plausible* tie the rest of the time.

⭐ **Measured 2026-08-30, and the control is what makes the tolerance defensible.** The tie passes
at max |Δ| **1.44e-03** (tol 2e-03, logits in [-0.31, 0.40]) — which looks uncomfortably close to
the bar until you run `--break`, which reintroduces the B1 defect on the reference side and lands
at **6.83e-01**. That is **475×** the passing residual, so the 2e-03 bar sits in a very wide gap,
not on a knife edge. The residual itself is two fp32 backends (IREE llvm-cpu vs XLA-CPU) sixty-odd
layers deep; it is accumulation, not disagreement.

    .venv/bin/python3 scripts/convnext_forward_tie.py
    .venv/bin/python3 scripts/convnext_forward_tie.py --break        # expect a FAIL; rc 0 if it fails
    .venv/bin/python3 scripts/convnext_forward_tie.py --net convnextsin --depths 3,3,27,3
"""
import argparse, os, re, subprocess, sys, tempfile
import numpy as np

IREE_C = os.environ.get(
    "IREE_COMPILE", "/home/skoonce/lean4-mlir/.venv/bin/iree-compile")
IREE_R = os.environ.get(
    "IREE_RUN_MODULE", "/home/skoonce/lean/klawd_max_power/iree-build/tools/iree-run-module")

# (mlir slug, reference file). Both members of each pair are the 1000-class ImageNet artifacts.
NETS = {
    "convnextin":  ("verified_mlir/convnextin_fwd.mlir",
                    "jax/generated/generated_convnext_tiny_imagenet.py"),
    "convnextsin": ("verified_mlir/convnextsin_fwd.mlir",
                    "jax/generated/generated_convnext_s_imagenet.py"),
}


def parse_input_shapes(path, fn):
    """The `tensor<AxBx...xf32>` of every argument of `func.func @fn`, in signature order."""
    src = open(path).read()
    m = re.search(r"func\.func @" + re.escape(fn) + r"\((.*?)\)\s*->", src, re.S)
    if not m:
        sys.exit(f"{path}: no `func.func @{fn}(...)` — wrong artifact or renamed entry point")
    return re.findall(r"tensor<([0-9x]*)f32>", m.group(1))


def load_reference_forward(ref_py, drop_head_ln=False):
    """exec the reference's function prefix — the real generated code, minus its train loop.

    ⚠ Cut at `def loss_fn`, as the other three ties do: everything above it is pure definitions,
    everything below reaches for tfds and a GPU. If the generator's shape changes this fails loudly
    rather than exec'ing a training script.

    `drop_head_ln` is `--break`: it comments out the reference's `head_layer_norm` call, i.e. it
    reintroduces exactly the B1 defect on the reference side only. The params list is untouched, so
    the tuple-count assertion still passes and the ONLY difference is the missing layer — which is
    the shape of the thing this gate exists to catch."""
    src = open(ref_py).read().split("\n")
    cut = next((i for i, l in enumerate(src) if l.startswith("def loss_fn")), None)
    if cut is None:
        sys.exit(f"{ref_py}: no `def loss_fn` to cut at; the generator's shape changed")
    body = "\n".join(src[:cut])
    if drop_head_ln:
        hits = [l for l in body.split("\n") if "x = head_layer_norm(" in l]
        if len(hits) != 1:
            sys.exit(f"--break: expected exactly one `head_layer_norm` call, found {len(hits)}")
        body = body.replace(hits[0], "    pass  # --break: head LN removed (the B1 defect)")
    mod = {"__name__": "not_main"}
    exec(body, mod)
    if "forward" not in mod:
        sys.exit("reference prefix did not define forward()")
    return mod["forward"], mod


def regroup(ws, depths):
    """The render's flat tensor list -> the reference's `params` list of tuples.

    Both sides walk the same `[3,3,9,3]` structure; this is that walk, written once.

      stem            psW,psb | psng,psnbt                      -> (W,b), (γ,β)
      block  (x9)     dW,db | ng,nbt | eW,eb | pW,pb | lg       -> 5 tuples
      downsample (x4) ng,nbt | W,b                              -> 2 tuples
      head LN         hng,hnbt                                  -> (γ,β)
      dense           Wd,bd                                     -> (Wᵀ, b)

    ⚠ The dense TRANSPOSE is not cosmetic. The render carries `Wd : [768, 1000]`; the reference
    computes `mm(x, params[-1][0].T)`, so it wants `[1000, 768]`. Feeding it untransposed is a
    shape error at 1000 ≠ 768 — which is the good case. It would be silent on a square head.
    ⚠ The head LN tuple goes BEFORE the dense one, matching `allParams` and the layer list. Get
    this backwards and the tie still runs: γ/β are `[768]` and so is nothing else here, but the
    dense would consume them and the result is nonsense rather than an error."""
    p, i = [], 0
    def take(n):
        nonlocal i
        out = ws[i:i + n]; i += n
        return out
    W, b, g, bt = take(4)
    p += [(W, b), (g, bt)]
    for si, d in enumerate(depths):
        for _ in range(d):
            dW, db, ng, nbt, eW, eb, pW, pb, lg = take(9)
            p += [(dW, db), (ng, nbt), (eW, eb), (pW, pb), (lg,)]
        if si < len(depths) - 1:
            ng, nbt, cW, cb = take(4)
            p += [(ng, nbt), (cW, cb)]
    hg, hb = take(2)
    p.append((hg, hb))                       # head LN — restored 2026-08-30, planning §7.1
    dW, db = take(2)
    p.append((dW.T, db))                     # dense: render [768,K] -> reference's [K,768]
    if i != len(ws):
        sys.exit(f"regroup consumed {i} of {len(ws)} tensors — the render's signature moved")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="convnextin", choices=sorted(NETS))
    ap.add_argument("--depths", default="3,3,9,3", help="stage depths of the selected net")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=0.05,
                    help="σ of the random weights; LN is scale-free but the conv stack is not")
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="max |Δ| over the logits (llvm-cpu vs XLA-CPU fp32, 60+ layers deep)")
    ap.add_argument("--keep", action="store_true", help="keep the work directory")
    ap.add_argument("--break", dest="brk", action="store_true",
                    help="THE CONTROL: delete the head LN from the reference and expect a FAIL. "
                         "This is the B1 defect, reintroduced on one side only — a gate nobody "
                         "has watched go red is an assertion, not a test.")
    args = ap.parse_args()

    mlir, ref_py = NETS[args.net]
    fn = args.net + "_fwd"
    depths = [int(d) for d in args.depths.split(",")]
    for f in (mlir, ref_py):
        if not os.path.exists(f):
            sys.exit(f"{f} missing — `scripts/regen_verified_mlir.sh proofs` / "
                     f"`scripts/regen_jax_generated.sh`")
    for exe in (IREE_C, IREE_R):
        if not os.path.exists(exe):
            sys.exit(f"{exe} missing — set $IREE_COMPILE / $IREE_RUN_MODULE")

    work = tempfile.mkdtemp(prefix="cnxtie_")
    os.makedirs(f"{work}/in", exist_ok=True)
    shapes = parse_input_shapes(mlir, fn)
    print(f"func @{fn}: {len(shapes)} inputs  (workdir {work})")

    rng = np.random.default_rng(args.seed)
    arrays, in_flags = [], []
    for i, s in enumerate(shapes):
        dims = [int(d) for d in s.split("x") if d]
        a = (rng.standard_normal(dims) * args.scale).astype(np.float32)
        arrays.append(a)
        path = f"{work}/in/i{i}.npy"
        np.save(path, a)
        in_flags.append(f"--input=@{path}")

    # ── the verified render, through IREE on the CPU backend ──
    r = subprocess.run([IREE_C, "--iree-hal-target-backends=llvm-cpu", mlir,
                        "-o", f"{work}/m.vmfb"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"iree-compile FAILED:\n{r.stderr[:3000]}")
    # ⚠⚠ `local-sync`, NOT `local-task`. The other three ties use `local-task` (IREE's
    # multithreaded CPU executor) and it **SIGSEGVs** on this module — rc -11 with an empty stderr,
    # after printing `EXEC @convnextin_fwd`, so it looks like a hang-then-die rather than a crash
    # with a message. `local-sync` runs the identical vmfb to completion. Measured 2026-08-30 on
    # IREE 3.12.0rc20260428; not diagnosed further, because the tie does not need the parallel
    # executor and a segfaulting runner is not evidence about the render.
    r = subprocess.run([IREE_R, "--device=local-sync", f"--module={work}/m.vmfb",
                        f"--function={fn}", *in_flags, f"--output=@{work}/out.npy"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"iree-run-module FAILED:\n{r.stderr[:3000]}")
    got = np.load(f"{work}/out.npy").astype(np.float64)

    # ── the reference, on the same weights ──
    import jax.numpy as jnp
    batch = int(shapes[0].split("x")[0])
    x = arrays[0].reshape(batch, -1)
    params = regroup(arrays[1:], depths)
    fwd, mod = load_reference_forward(ref_py, drop_head_ln=args.brk)
    # ⚠ Asserted, not assumed: the reference's own `forward` indexes its last two groups by
    # literal number, so a regrouping with the wrong tuple count can still run and be wrong.
    src = open(ref_py).read()
    last = max(int(n) for n in re.findall(r"params\[(\d+)\]", src))
    if last != len(params) - 1:
        sys.exit(f"regrouped into {len(params)} tuples but the reference's highest index is "
                 f"{last} — the two sides disagree about the parameter layout")
    print(f"  regrouped {len(arrays)-1} tensors -> {len(params)} reference tuples "
          f"(highest params[] index in the generator: {last}) ✓")
    want = np.asarray(fwd(params, jnp.asarray(x))).astype(np.float64)

    if got.shape != want.shape:
        sys.exit(f"SHAPE MISMATCH: render {got.shape} vs reference {want.shape}")
    d = np.abs(got - want)
    scale = max(np.abs(want).max(), 1e-12)
    print(f"  logits range (reference): [{want.min():.4f}, {want.max():.4f}]")
    print(f"  max |Δ|      : {d.max():.3e}")
    print(f"  mean |Δ|     : {d.mean():.3e}")
    print(f"  max |Δ|/scale: {d.max()/scale:.3e}")
    if not args.keep:
        subprocess.run(["rm", "-rf", work])
    if args.brk:
        ok = d.max() > args.tol
        print(f"  {'✓' if ok else '⛔'} CONTROL: with the head LN removed from the reference the tie "
              f"{'FAILS as it must' if ok else 'STILL PASSES — THE GATE IS BLIND'}")
        return 0 if ok else 1
    if d.max() <= args.tol:
        print(f"  ✓ FORWARD TIE PASSES (tol {args.tol:.1e}) — the verified render computes the "
              f"reference's function: layer order, LN axis, head LN and parameter layout")
        return 0
    print(f"  ✗ FORWARD TIE FAILS (tol {args.tol:.1e})")
    print("    Suspects, in the order they have actually bitten: a layer present on one side and")
    print("    not the other (the head LN, §7.1); the LN axis (channel vs feature); the dense")
    print("    transpose; a downsample's `VALID` vs the render's explicit pad.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
