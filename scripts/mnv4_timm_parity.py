#!/usr/bin/env python3
"""MNv4 timm parity: the JAX reference against timm's `mobilenetv4_conv_medium`, on SHARED weights.

timm 1.0.28 is the architecture spec for MobileNetV4-Conv-M (planning/mnv4_timm_parity.md). The
block table's `#guard`s pin kernel sizes and channel counts; they cannot see which conv carries a
stride, whether a BN is followed by an activation, or whether the head pools before or after
`conv_head`. Five such deviations from timm (a pre-DW stride, a pre-DW relu, a swish stage 0, a
7×7 `conv_head`, XLA-`SAME` at the stem) were invisible to every census, shape and op-count gate,
and to every tie against the JAX reference, because the reference had them too. A forward on one
set of weights is the check that sees all of them.

What it does:
  1. `.venv-timm` builds timm's net (random init, BN affine + running statistics randomised) and
     dumps its parameters in the JAX `params[k][j]` order with train- and eval-mode logits.
  2. The two JAX emitters (`jax/MainMobilenetV4.lean`, 10-class, batch-BN only; and
     `jax/MainMobilenetV4Imagenet.lean`, 1000-class, running BN) are re-emitted from the CURRENT
     source into a scratch directory; nothing under `jax/.lake/build` is read or written.
  3. Each emitted `forward` runs from its function prefix (the module-level train loop is cut off
     at `def loss_fn`, as `mnv4_forward_tie.py` does) with the bf16 matmul/conv dtypes forced to
     f32 and on CPU.
  4. Logits are compared: Imagenette in train mode; ImageNet in train mode AND in eval mode
     against timm's running statistics.

Usage:
    scripts/mnv4_timm_parity.py              # batch 4, seed 0, tol 1e-3 (relative to max |logit|)
    scripts/mnv4_timm_parity.py --tol 1e-4
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMM_PY = os.path.join(ROOT, ".venv-timm", "bin", "python")

EMIT = """import {mod}
#eval IO.FS.writeFile "{out}" (JaxCodegen.generate {spec} {cfg} {ds} "data")
"""
NETS = [
    # (label, module, spec, config, dataset, classes, has running BN)
    ("imagenette", "MainMobilenetV4", "mobilenetV4Medium", "mobilenetV4Config", ".imagenette", 10, False),
    ("imagenet", "MainMobilenetV4Imagenet", "mobilenetV4ConvMImagenet",
     "mobilenetV4ConvMImagenetConfig", ".imagenet", 1000, True),
]


def emit(tmp, mod, spec, cfg, ds):
    out = os.path.join(tmp, f"{mod}.py")
    src = os.path.join(tmp, f"Emit{mod}.lean")
    with open(src, "w") as f:
        f.write(EMIT.format(mod=mod, out=out, spec=spec, cfg=cfg, ds=ds))
    r = subprocess.run(["lake", "env", "lean", src], cwd=os.path.join(ROOT, "jax"),
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        sys.exit(f"⛔ emitting {mod} failed (run `cd jax && lake build {mod}` first?)\n{r.stdout}{r.stderr}")
    return out


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
    return mod["forward"]


def timm_dump(tmp, classes, batch, seed):
    out = os.path.join(tmp, f"timm_{classes}.npz")
    r = subprocess.run([TIMM_PY, os.path.join(ROOT, "scripts", "_mnv4_timm_dump.py"), out,
                        "--classes", str(classes), "--batch", str(batch), "--seed", str(seed)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"⛔ timm dump failed:\n{r.stdout}{r.stderr}")
    print("  " + r.stdout.strip())
    return np.load(out)


def rel_err(a, b):
    return float(np.max(np.abs(a - b)) / max(1e-12, np.max(np.abs(b))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args()
    if not os.path.exists(TIMM_PY):
        sys.exit(f"⛔ {TIMM_PY} missing — the pinned timm env (requirements-timm-lock.txt)")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp

    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        for label, mod, spec, cfg, ds, classes, running in NETS:
            print(f"[{label}] {spec}")
            d = timm_dump(tmp, classes, a.batch, a.seed)
            fwd = load_forward(emit(tmp, mod, spec, cfg, ds))
            params = [tuple(jnp.asarray(d[f"p{k}_{j}"]) for j in range(3 if f"p{k}_2" in d else 2))
                      for k in range(int(d["n_params"]))]
            x = jnp.asarray(d["x"])
            checks = []
            if running:
                bn = [(jnp.asarray(d[f"s{k}_0"]), jnp.asarray(d[f"s{k}_1"]))
                      for k in range(int(d["n_stats"]))]
                checks.append(("train", fwd(params, x, bn, True)[0], d["y_train"]))
                checks.append(("eval", fwd(params, x, bn, False)[0], d["y_eval"]))
            else:
                checks.append(("train", fwd(params, x), d["y_train"]))
            for mode, y, ref in checks:
                e = rel_err(np.asarray(y), ref)
                ok = e <= a.tol
                fails += not ok
                print(f"  {mode:5s}: max|Δ|/max|timm| = {e:.3e}  {'✅' if ok else '⛔'}")
    if fails:
        sys.exit(f"⛔ {fails} check(s) above tolerance {a.tol}: the JAX reference is not timm's net")
    print("✅ the JAX MNv4 references compute timm's mobilenetv4_conv_medium")


if __name__ == "__main__":
    main()
