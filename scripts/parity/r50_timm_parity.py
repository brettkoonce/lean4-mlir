#!/usr/bin/env python3
"""ResNet-50 timm parity: the JAX reference AND the verified render against timm's `resnet50`, on
SHARED weights.

timm 1.0.28 is the architecture spec for the RSB runs (`resnet50.a{1,2,3}_in1k` are plain
`resnet50`). A census cannot see which conv of a downsampling bottleneck carries the stride, whether
the stem pool pads symmetrically, or which operand order a block adds; a forward on one set of
weights sees all of them. R50 had no forward tie on either path until this gate.

What it does:
  1. `.venv-timm` builds timm's net (random init, BN affine + running statistics randomised) and
     dumps its parameters in forward order with logits in train mode at 224 and 160, eval mode at
     224 and 288, and its per-block drop-path probabilities (`_r50_timm_dump.py`).
  2. JAX: the `a2-accum` trainer is re-emitted from the CURRENT source into a scratch directory
     (nothing under `jax/.lake/build` is read or written), and its `forward` runs from the function
     prefix on CPU at f32 in all four modes. Its drop-path keeps are read off the emitted
     `forward` and compared with timm's.
  3. Verified: `ResNet50RenderB`'s forwards are rendered at the gate's batch into the scratch
     directory (`resnet50FwdFaithfulV` at 224 and 160, `resnet50FwdEvalFaithfulV` at 224 and 288 —
     the functions behind `resnet50in_fwd`, `resnet50in160_fwd`, `resnet50in_fwd_eval` and
     `resnet50in_fwd_eval_s288`) and run through iree-compile on CPU. The verified spec's drop keeps
     (`resnet50ImagenetVerified.dropKeeps`) are compared with timm's too.

Usage:
    scripts/parity/r50_timm_parity.py              # batch 2, seed 0, tol 1e-3 (relative to max |logit|)
    scripts/parity/r50_timm_parity.py --skip-verified   # the JAX half only (no IREE needed)
    scripts/parity/r50_timm_parity.py --break      # control: one BN's γ and β swapped on our side; must go red
"""
import argparse, os, re, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIMM_PY = os.path.join(ROOT, ".venv-timm", "bin", "python")
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))

EMIT_JAX = """import MainResnet50Imagenet
#eval IO.FS.writeFile "{out}" (JaxCodegen.generate resnet50Imagenet resnet50ImagenetConfigA2Accum .imagenet "data")
"""
EMIT_VERIFIED = """import LeanMlir.Proofs.Codegen.ResNet50RenderB
import LeanMlir.Verified.NetsCore
#eval IO.FS.writeFile "{d}/train224.mlir" (Proofs.StableHLO.resnet50FwdFaithfulV {B} 1000 "1.0e-05" "resnet50in")
#eval IO.FS.writeFile "{d}/train160.mlir" (Proofs.StableHLO.resnet50FwdFaithfulV {B} 1000 "1.0e-05" "resnet50in160" (q := 5))
#eval IO.FS.writeFile "{d}/eval224.mlir" (Proofs.StableHLO.resnet50FwdEvalFaithfulV {B} 1000 "1.0e-05" "resnet50in")
#eval IO.FS.writeFile "{d}/eval288.mlir" (Proofs.StableHLO.resnet50FwdEvalFaithfulV {B} 1000 "1.0e-05" "resnet50in" (q := 9) (vSuffix := "_s288"))
#eval IO.FS.writeFile "{d}/keeps.txt" (String.intercalate " " (resnet50ImagenetVerified.dropKeeps.toList.map toString))
"""
# (mode, resolution, verified artifact in the scratch dir, its entry point)
MODES = [("train", 224, "train224.mlir", "resnet50in_fwd"),
         ("train", 160, "train160.mlir", "resnet50in160_fwd"),
         ("eval", 224, "eval224.mlir", "resnet50in_fwd_eval"),
         ("eval", 288, "eval288.mlir", "resnet50in_fwd_eval_s288")]


def lean(src_text, path, cwd):
    with open(path, "w") as f:
        f.write(src_text)
    r = subprocess.run(["lake", "env", "lean", path], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"⛔ `lake env lean {path}` failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")


def load_forward(path):
    src = open(path).read().split("\n")
    cut = next((i for i, l in enumerate(src) if l.startswith("def loss_fn")), None)
    if cut is None:
        sys.exit(f"{path}: no `def loss_fn` to cut at; the generator's shape changed")
    mod = {}
    exec("\n".join(src[:cut]), mod)
    import jax.numpy as jnp
    # `mm` / `convdt` read these at call time; f32 makes the comparison about the FUNCTION.
    for k in ("DT", "CONV_DT"):
        if k in mod:
            mod[k] = jnp.float32
    keeps = [float(k) for k in re.findall(r"dpkeys\[\d+\], ([0-9.]+)\)", "\n".join(src[:cut]))]
    return mod["forward"], keeps


def rel_err(a, b):
    return float(np.max(np.abs(a - b)) / max(1e-12, np.max(np.abs(b))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--skip-verified", action="store_true")
    ap.add_argument("--break", dest="brk", action="store_true",
                    help="negative control: swap γ/β of the stage-3 entry block's 3x3 BN on both paths")
    a = ap.parse_args()
    if not os.path.exists(TIMM_PY):
        sys.exit(f"⛔ {TIMM_PY} missing — the pinned timm env (requirements-timm-lock.txt)")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp

    fails = 0

    def report(label, e):
        nonlocal fails
        ok = e <= a.tol
        fails += not ok
        print(f"  {label:22s} max|Δ|/max|timm| = {e:.3e}  {'✅' if ok else '⛔'}")

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "timm.npz")
        r = subprocess.run([TIMM_PY, os.path.join(ROOT, "scripts", "parity", "_r50_timm_dump.py"), out,
                            "--batch", str(a.batch), "--seed", str(a.seed)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"⛔ timm dump failed:\n{r.stdout}{r.stderr}")
        print("  " + r.stdout.strip())
        d = np.load(out)
        nP, nS = int(d["n_params"]), int(d["n_stats"])
        groups = [[d[f"p{k}_{j}"] for j in range(3 if f"p{k}_2" in d else 2)] for k in range(nP)]
        stats = [(d[f"s{k}_0"], d[f"s{k}_1"]) for k in range(nS)]
        drop = d["drop_probs"]
        if a.brk:
            # group 25 = layer3.0's conv2/bn2 (stem, then 4 groups per downsample block, 3 per block):
            # same shapes, a different net. Both paths must fail every mode.
            groups[25] = [groups[25][0], groups[25][2], groups[25][1]]
            print("  ⚠ --break: layer3.0.bn2 γ and β swapped — every logit check below must fail")

        # ── the JAX reference ──
        print("[jax] resnet50Imagenet, the a2-accum trainer's forward")
        py = os.path.join(tmp, "r50.py")
        lean(EMIT_JAX.format(out=py), os.path.join(tmp, "EmitR50.lean"), os.path.join(ROOT, "jax"))
        fwd, keeps = load_forward(py)
        params = [tuple(jnp.asarray(t) for t in g) for g in groups]
        bn = [(jnp.asarray(mu), jnp.asarray(v)) for mu, v in stats]
        for mode, res, _, _ in MODES:
            y = fwd(params, jnp.asarray(d[f"x_{mode}{res}"]), bn, mode == "train")[0]
            report(f"{mode} @ {res}", rel_err(np.asarray(y), d[f"y_{mode}{res}"]))
        e = float(np.max(np.abs((1.0 - np.array(keeps)) - drop))) if len(keeps) == len(drop) else 1.0
        ok = e < 1e-6
        fails += not ok
        print(f"  drop-path ramp         {len(keeps)} keeps, max|Δ drop prob| = {e:.1e}  {'✅' if ok else '⛔'}")

        # ── the verified render ──
        if not a.skip_verified:
            import _iree
            print(f"[verified] ResNet50RenderB forwards at batch {a.batch}, iree llvm-cpu")
            lean(EMIT_VERIFIED.format(d=tmp, B=a.batch), os.path.join(tmp, "EmitR50V.lean"), ROOT)
            # the verified dense weight is [in, out]; timm's (and the reference's) is [out, in]
            flat = [t for g in groups[:-1] for t in g] + [groups[-1][0].T.copy(), groups[-1][1]]
            flat_stats = [t for mu_v in stats for t in mu_v]
            for mode, res, mlir, fn in MODES:
                x = d[f"x_{mode}{res}"].reshape(a.batch, -1)
                arrays = [x] + flat + (flat_stats if mode == "eval" else [])
                arrays = [np.ascontiguousarray(t, dtype=np.float32) for t in arrays]
                work = os.path.join(tmp, f"iree_{mode}{res}")
                os.makedirs(work)
                y = _iree.compile_and_run(os.path.join(tmp, mlir), fn, arrays, work, 1, "llvm-cpu")[0]
                report(f"{mode} @ {res}", rel_err(np.asarray(y, dtype=np.float64), d[f"y_{mode}{res}"]))
            vk = [float(t) for t in open(os.path.join(tmp, "keeps.txt")).read().split()]
            e = float(np.max(np.abs((1.0 - np.array(vk)) - drop))) if len(vk) == len(drop) else 1.0
            ok = e < 1e-6
            fails += not ok
            print(f"  drop-path ramp         {len(vk)} keeps (resnet50ImagenetVerified), "
                  f"max|Δ drop prob| = {e:.1e}  {'✅' if ok else '⛔'}")

    if a.brk:
        want = 4 * (1 if a.skip_verified else 2)
        if fails >= want:
            print(f"✅ control: {fails} check(s) went red, as they must")
            return
        sys.exit(f"⛔ control: only {fails} of {want} logit checks went red — the gate cannot see a BN swap")
    if fails:
        sys.exit(f"⛔ {fails} check(s) failed: not timm's resnet50")
    print("✅ the JAX reference and the verified render compute timm's resnet50")


if __name__ == "__main__":
    main()
