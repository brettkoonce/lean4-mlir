#!/usr/bin/env python3
"""EfficientNet-B0 timm parity: the JAX reference against timm's EfficientNet-B0, on SHARED
weights (planning/imagenet_parity.md G3).

`enet_forward_tie.py` ties the verified render to the Imagenette JAX file only, which is how the
ImageNet stem/head ran ReLU instead of swish unseen (2026-08-30): nothing compared either reference
with an independent EfficientNet. This does, for both JAX emitters.

The spec is TF's EfficientNet-B0 (SAME padding at every conv, BN ε 1e-3). The JAX and verified
paths pad the stem SAME but the strided depthwise convs symmetrically — a hybrid neither TF nor
timm's `efficientnet_b0` computes. The gate pins that hybrid (`--pad hybrid`, the default: timm's
`efficientnet_b0` with a SAME stem) so any OTHER structural drift is caught, and `--pad tf` reports
the distance to TF's net itself — the one deviation left, and it is disclosed, not hidden.

What it does:
  1. `.venv-timm` builds timm's net (random init, BN affine + running statistics randomised) and
     dumps its parameters in the JAX `params[k][j]` order with train- and eval-mode logits.
  2. The two JAX emitters (`jax/MainEfficientNet.lean`, 10-class, batch-BN only; and
     `jax/MainEfficientNetImagenet.lean`, 1000-class, running BN) are re-emitted from the CURRENT
     source into a scratch directory; nothing under `jax/.lake/build` is read or written.
  3. Each emitted `forward` runs from its function prefix (the module-level train loop is cut off
     at `def loss_fn`, as `mnv4_forward_tie.py` does) with the bf16 matmul/conv dtypes forced to
     f32 and on CPU.
  4. Logits are compared: Imagenette in train mode; ImageNet in train mode AND in eval mode
     against timm's running statistics.

Usage:
    scripts/parity/enet_timm_parity.py              # batch 4, seed 0, tol 1e-3 (relative to max |logit|)
    scripts/parity/enet_timm_parity.py --tol 1e-4
    scripts/parity/enet_timm_parity.py --controls   # also RED on symmetric padding and on ε 1e-5
    scripts/parity/enet_timm_parity.py --pad tf     # the distance to TF's all-SAME net (expected above tol)
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIMM_PY = os.path.join(ROOT, ".venv-timm", "bin", "python")

EMIT = """import {mod}
#eval IO.FS.writeFile "{out}" (JaxCodegen.generate {spec} {cfg} {ds} "data")
"""
NETS = [
    # (label, module, spec, config, dataset, classes, has running BN, BN ε)
    ("imagenette", "MainEfficientNet", "efficientNetB0", "efficientNetConfig", ".imagenette", 10,
     False, 1e-5),
    ("imagenet", "MainEfficientNetImagenet", "efficientNetB0Imagenet",
     "efficientNetB0ImagenetConfigFull", ".imagenet", 1000, True, 1e-3),
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


def timm_dump(tmp, classes, batch, seed, eps, pad="hybrid"):
    out = os.path.join(tmp, f"timm_{classes}_{pad}_{eps}.npz")
    r = subprocess.run([TIMM_PY, os.path.join(ROOT, "scripts", "parity", "_enet_timm_dump.py"), out,
                        "--classes", str(classes), "--batch", str(batch), "--seed", str(seed),
                        "--eps", repr(eps), "--pad", pad],
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
    ap.add_argument("--pad", default="hybrid", choices=["hybrid", "tf", "sym"],
                    help="the timm target: our SAME-stem hybrid (the gate), TF's all-SAME net, or symmetric")
    ap.add_argument("--controls", action="store_true",
                    help="also run the ImageNet net against timm at symmetric padding and at ε 1e-5; "
                         "both must come out ABOVE tolerance, or the gate cannot see those defects")
    a = ap.parse_args()
    if not os.path.exists(TIMM_PY):
        sys.exit(f"⛔ {TIMM_PY} missing — the pinned timm env (requirements-timm-lock.txt)")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp

    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        for label, mod, spec, cfg, ds, classes, running, eps in NETS:
            print(f"[{label}] {spec} / {cfg}")
            d = timm_dump(tmp, classes, a.batch, a.seed, eps, a.pad)
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
        if a.controls:
            label, mod, spec, cfg, ds, classes, running, eps = NETS[1]
            fwd = load_forward(emit(tmp, mod, spec, cfg, ds))
            for why, pad, ceps in (("symmetric padding everywhere (timm's efficientnet_b0)", "sym", eps),
                                   ("BN ε 1e-5 against the recipe's 1e-3", "hybrid", 1e-5)):
                d = timm_dump(tmp, classes, a.batch, a.seed, ceps, pad)
                params = [tuple(jnp.asarray(d[f"p{k}_{j}"]) for j in range(3 if f"p{k}_2" in d else 2))
                          for k in range(int(d["n_params"]))]
                bn = [(jnp.asarray(d[f"s{k}_0"]), jnp.asarray(d[f"s{k}_1"]))
                      for k in range(int(d["n_stats"]))]
                e = rel_err(np.asarray(fwd(params, jnp.asarray(d["x"]), bn, False)[0]), d["y_eval"])
                red = e > a.tol
                fails += not red
                print(f"  CONTROL {why}: eval {e:.3e}  {'✅ red, as it must be' if red else '⛔ GREEN — the gate is blind to this'}")
    if fails:
        sys.exit(f"⛔ {fails} check(s) failed at tolerance {a.tol}: the JAX reference is not that net, "
                 "or a control came out green")
    print(f"✅ the JAX B0 references compute timm's EfficientNet-B0 at pad '{a.pad}'")


if __name__ == "__main__":
    main()
