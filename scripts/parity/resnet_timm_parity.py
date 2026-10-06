#!/usr/bin/env python3
"""ResNet timm parity: the JAX references AND the verified renders against timm's `resnet50` and
`resnet34`, on SHARED weights.

timm 1.0.28 is the architecture spec (`resnet50.a{1,2,3}_in1k` / `resnet50.tv_in1k` are plain
`resnet50`, `resnet34.tv_in1k` plain `resnet34`). A census cannot see which conv of a downsampling
block carries the stride, whether the stem pool pads symmetrically, or which operand order a block
adds; a forward on one set of weights sees all of them. Neither net had a forward tie on either path
until this gate.

What it does, per net:
  1. `.venv-timm` builds timm's net (random init, BN affine + running statistics randomised) and
     dumps its parameters in forward order with logits in train and eval mode at the net's sizes,
     and its per-block drop-path probabilities (`_resnet_timm_dump.py`).
  2. JAX: the trainer (R50 `a2-accum`, R34 `default`) is re-emitted from the CURRENT source into a
     scratch directory (nothing under `jax/.lake/build` is read or written), and its `forward` runs
     from the function prefix on CPU at f32 in every mode. Its drop-path keeps are read off the
     emitted `forward` and compared with timm's.
  3. Verified: the renderer's forwards are rendered at the gate's batch into the scratch directory
     and run through iree-compile on CPU — R50: `resnet50FwdText` at 224 and 160,
     `resnet50FwdEvalText` at 224 and 288 (`resnet50in_fwd`, `resnet50in160_fwd`,
     `resnet50in_fwd_eval`, `resnet50in_fwd_eval_s288`); R34: `resnet34FwdText` and
     `resnet34FwdEvalText` at 224 (`resnet34in_fwd`, `resnet34in_fwd_eval`). The verified
     spec's drop keeps are compared with timm's too.

Usage:
    scripts/parity/resnet_timm_parity.py                 # both nets, batch 2, seed 0, tol 1e-3 of max |logit|
    scripts/parity/resnet_timm_parity.py --arch resnet34
    scripts/parity/resnet_timm_parity.py --skip-verified # the JAX half only (no IREE needed)
    scripts/parity/resnet_timm_parity.py --break         # control: one BN's γ and β swapped on our side; must go red
"""
import argparse, os, re, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIMM_PY = os.path.join(ROOT, ".venv-timm", "bin", "python")
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))

ARCHS = {
    "resnet50": dict(
        jax="""import MainResnet50Imagenet
#eval IO.FS.writeFile "{out}" (JaxCodegen.generate resnet50Imagenet resnet50ImagenetConfigA2Accum .imagenet "data")
""",
        verified="""import LeanMlir.Proofs.Codegen.ResNet50RenderB
import LeanMlir.Verified.NetsCore
#eval IO.FS.writeFile "{d}/train224.mlir" (Proofs.StableHLO.resnet50FwdText {B} 1000 "1.0e-05" "resnet50in")
#eval IO.FS.writeFile "{d}/train160.mlir" (Proofs.StableHLO.resnet50FwdText {B} 1000 "1.0e-05" "resnet50in160" (q := 5))
#eval IO.FS.writeFile "{d}/eval224.mlir" (Proofs.StableHLO.resnet50FwdEvalText {B} 1000 "1.0e-05" "resnet50in")
#eval IO.FS.writeFile "{d}/eval288.mlir" (Proofs.StableHLO.resnet50FwdEvalText {B} 1000 "1.0e-05" "resnet50in" (q := 9) (vSuffix := "_s288"))
#eval IO.FS.writeFile "{d}/keeps.txt" (String.intercalate " " (resnet50ImagenetVerified.dropKeeps.toList.map toString))
""",
        spec="resnet50ImagenetVerified", drop=0.05,
        # (mode, resolution, verified artifact in the scratch dir, its entry point)
        modes=[("train", 224, "train224.mlir", "resnet50in_fwd"),
               ("train", 160, "train160.mlir", "resnet50in160_fwd"),
               ("eval", 224, "eval224.mlir", "resnet50in_fwd_eval"),
               ("eval", 288, "eval288.mlir", "resnet50in_fwd_eval_s288")],
        # --break target: layer3.0's conv2/bn2 (stem, then 4 groups per downsample block, 3 per block)
        brk=25),
    "resnet34": dict(
        jax="""import MainResnetImagenet
#eval IO.FS.writeFile "{out}" (JaxCodegen.generate resnet34Imagenet resnet34ImagenetConfig .imagenet "data")
""",
        verified="""import LeanMlir.Proofs.Codegen.ResNet34RenderB
import LeanMlir.Verified.NetsCore
#eval IO.FS.writeFile "{d}/train224.mlir" (Proofs.StableHLO.resnet34FwdText {B} 1000 "1.0e-05" "resnet34in")
#eval IO.FS.writeFile "{d}/eval224.mlir" (Proofs.StableHLO.resnet34FwdEvalText {B} 1000 "1.0e-05" "resnet34in")
#eval IO.FS.writeFile "{d}/keeps.txt" (String.intercalate " " (resnet34ImagenetVerified.dropKeeps.toList.map toString))
""",
        spec="resnet34ImagenetVerified", drop=0.0,
        modes=[("train", 224, "train224.mlir", "resnet34in_fwd"),
               ("eval", 224, "eval224.mlir", "resnet34in_fwd_eval")],
        # layer3.0's conv2/bn2 (stem, 3 + 3 basic blocks of 2 in layer1/2 bar layer2.0's 3)
        brk=17),
}


def lean(src_text, path, cwd):
    with open(path, "w") as f:
        f.write(src_text)
    # `lake env lean` loads whatever .olean is on disk and never rebuilds. A module built before a
    # `TrainConfig` field was added still loads, and its config then reads back wrong values, so
    # build what the source imports first; a no-op when it is current.
    for mod in re.findall(r"(?m)^import (\S+)", src_text):
        b = subprocess.run(["lake", "build", mod], cwd=cwd, capture_output=True, text=True)
        if b.returncode != 0:
            sys.exit(f"⛔ `lake build {mod}` (in {cwd}) failed\n{b.stdout[-3000:]}{b.stderr[-3000:]}")
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


def ramp_err(keeps, drop):
    """max |Δ| between our drop probabilities (1 − keep) and timm's. A net with no drop sites
    (R34) matches a timm net whose every probability is 0."""
    if not keeps:
        return float(np.max(np.abs(drop))) if len(drop) else 0.0
    return float(np.max(np.abs((1.0 - np.array(keeps)) - drop))) if len(keeps) == len(drop) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all", choices=["all", *ARCHS])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--skip-verified", action="store_true")
    ap.add_argument("--break", dest="brk", action="store_true",
                    help="negative control: swap γ/β of layer3.0's second BN on both paths")
    a = ap.parse_args()
    if not os.path.exists(TIMM_PY):
        sys.exit(f"⛔ {TIMM_PY} missing — the pinned timm env (requirements-timm-lock.txt)")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp

    fails = 0
    red_wanted = 0

    def report(label, e):
        nonlocal fails
        ok = e <= a.tol
        fails += not ok
        print(f"  {label:22s} max|Δ|/max|timm| = {e:.3e}  {'✅' if ok else '⛔'}")

    def ramp(label, keeps, drop):
        nonlocal fails
        e = ramp_err(keeps, drop)
        ok = e < 1e-6
        fails += not ok
        print(f"  drop-path ramp         {len(keeps)} keeps{label}, max|Δ drop prob| = {e:.1e}  "
              f"{'✅' if ok else '⛔'}")

    archs = list(ARCHS) if a.arch == "all" else [a.arch]
    with tempfile.TemporaryDirectory() as tmp:
        for arch in archs:
            A = ARCHS[arch]
            modes = A["modes"]
            red_wanted += len(modes) * (1 if a.skip_verified else 2)
            wd = os.path.join(tmp, arch)
            os.makedirs(wd)
            out = os.path.join(wd, "timm.npz")
            tr = ",".join(str(r) for m, r, _, _ in modes if m == "train")
            ev = ",".join(str(r) for m, r, _, _ in modes if m == "eval")
            r = subprocess.run([TIMM_PY, os.path.join(ROOT, "scripts", "parity", "_resnet_timm_dump.py"),
                                out, "--model", arch, "--batch", str(a.batch), "--seed", str(a.seed),
                                "--drop-path", str(A["drop"]), "--train-res", tr, "--eval-res", ev],
                               capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit(f"⛔ timm dump failed:\n{r.stdout}{r.stderr}")
            print(f"[{arch}]  " + r.stdout.strip())
            d = np.load(out)
            nP, nS = int(d["n_params"]), int(d["n_stats"])
            groups = [[d[f"p{k}_{j}"] for j in range(3 if f"p{k}_2" in d else 2)] for k in range(nP)]
            stats = [(d[f"s{k}_0"], d[f"s{k}_1"]) for k in range(nS)]
            drop = d["drop_probs"]
            if a.brk:
                # same shapes, a different net: both paths must fail every mode
                k = A["brk"]
                groups[k] = [groups[k][0], groups[k][2], groups[k][1]]
                print(f"  ⚠ --break: param group {k} (layer3.0's second BN) γ and β swapped")

            # ── the JAX reference ──
            print(f"  [jax] the re-emitted trainer's forward")
            py = os.path.join(wd, "ref.py")
            lean(A["jax"].format(out=py), os.path.join(wd, "EmitJax.lean"), os.path.join(ROOT, "jax"))
            fwd, keeps = load_forward(py)
            params = [tuple(jnp.asarray(t) for t in g) for g in groups]
            bn = [(jnp.asarray(mu), jnp.asarray(v)) for mu, v in stats]
            for mode, res, _, _ in modes:
                y = fwd(params, jnp.asarray(d[f"x_{mode}{res}"]), bn, mode == "train")[0]
                report(f"{mode} @ {res}", rel_err(np.asarray(y), d[f"y_{mode}{res}"]))
            ramp("", keeps, drop)

            # ── the verified render ──
            if not a.skip_verified:
                import _iree
                print(f"  [verified] the renderer's forwards at batch {a.batch}, iree llvm-cpu")
                lean(A["verified"].format(d=wd, B=a.batch), os.path.join(wd, "EmitVerified.lean"), ROOT)
                # the verified dense weight is [in, out]; timm's (and the reference's) is [out, in]
                flat = [t for g in groups[:-1] for t in g] + [groups[-1][0].T.copy(), groups[-1][1]]
                flat_stats = [t for mu_v in stats for t in mu_v]
                for mode, res, mlir, fn in modes:
                    x = d[f"x_{mode}{res}"].reshape(a.batch, -1)
                    arrays = [x] + flat + (flat_stats if mode == "eval" else [])
                    arrays = [np.ascontiguousarray(t, dtype=np.float32) for t in arrays]
                    work = os.path.join(wd, f"iree_{mode}{res}")
                    os.makedirs(work)
                    y = _iree.compile_and_run(os.path.join(wd, mlir), fn, arrays, work, 1, "llvm-cpu")[0]
                    report(f"{mode} @ {res}", rel_err(np.asarray(y, dtype=np.float64), d[f"y_{mode}{res}"]))
                vk = [float(t) for t in open(os.path.join(wd, "keeps.txt")).read().split()]
                ramp(f" ({A['spec']})", vk, drop)

    if a.brk:
        if fails >= red_wanted:
            print(f"✅ control: {fails} check(s) went red, as they must")
            return
        sys.exit(f"⛔ control: only {fails} of {red_wanted} logit checks went red — the gate cannot see a BN swap")
    if fails:
        sys.exit(f"⛔ {fails} check(s) failed: not timm's {' / '.join(archs)}")
    print(f"✅ the JAX references and the verified renders compute timm's {' / '.join(archs)}")


if __name__ == "__main__":
    main()
