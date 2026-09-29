#!/usr/bin/env python3
"""ConvNeXt timm parity: the JAX references against timm's `convnext_{tiny,small,base}`, on SHARED
weights.

Nothing compared the ConvNeXt references with an independent ConvNeXt: `convnext_forward_tie.py`
ties the verified render to the JAX reference, and both are built from the same spec. This runs the
three JAX emitters (`jax/MainConvNeXt{,S,B}Imagenet.lean`, re-emitted from the CURRENT source into a
scratch directory, bf16 dtypes forced to f32, CPU) against timm on one set of weights, at 224 and at
timm's 288 test size, and compares each forward's drop-path keeps with timm's per-block rates.

The reference computes tanh GELU where timm's ConvNeXt uses erf (`imagenet_parity.md` §5.4, an
open decision). As in `vit_timm_parity.py`, the gate pins what we compute (timm built with tanh) so
any OTHER drift is caught; `--paper` reports the distance to timm's own erf GELU.

Usage:
    .venv/bin/python scripts/parity/cnx_timm_parity.py            # batch 2, seed 0, tol 1e-4
    .venv/bin/python scripts/parity/cnx_timm_parity.py --controls # also RED on erf GELU and on a LayerScale swap
    .venv/bin/python scripts/parity/cnx_timm_parity.py --paper    # distance to timm's erf GELU
"""
import argparse, os, re, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIMM_PY = os.path.join(ROOT, ".venv-timm", "bin", "python")

EMIT = """import {mod}
#eval IO.FS.writeFile "{out}" (JaxCodegen.generate {spec} {cfg} .imagenet "data")
"""
NETS = [
    # (label, module, spec, config, timm model)
    ("T", "MainConvNeXtImagenet", "convNeXtTinyImagenet", "convNeXtTinyImagenetConfig", "convnext_tiny"),
    ("S", "MainConvNeXtSImagenet", "convNeXtSImagenet", "convNeXtSImagenetConfig", "convnext_small"),
    ("B", "MainConvNeXtBImagenet", "convNeXtBImagenet", "convNeXtBImagenetConfig", "convnext_base"),
]


def emit(tmp, mod, spec, cfg):
    out = os.path.join(tmp, f"{mod}.py")
    src = os.path.join(tmp, f"Emit{mod}.lean")
    with open(src, "w") as f:
        f.write(EMIT.format(mod=mod, out=out, spec=spec, cfg=cfg))
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
    prefix = "\n".join(src[:cut])
    mod = {}
    exec(prefix, mod)
    import jax.numpy as jnp
    for k in ("DT", "CONV_DT"):
        if k in mod:
            mod[k] = jnp.float32
    keeps = [float(k) for k in re.findall(r"dpkeys\[\d+\], ([0-9.]+)\)", prefix)]
    return mod["forward"], keeps


def timm_dump(tmp, model, batch, seed, *flags):
    out = os.path.join(tmp, f"timm_{model}_{'_'.join(flags) or 'ours'}.npz")
    r = subprocess.run([TIMM_PY, os.path.join(ROOT, "scripts", "parity", "_cnx_timm_dump.py"), out,
                        "--model", model, "--batch", str(batch), "--seed", str(seed), *flags],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"⛔ timm dump failed:\n{r.stdout}{r.stderr}")
    print("  " + r.stdout.strip())
    return np.load(out)


def run(fwd, d, res, swap_ls=False):
    import jax.numpy as jnp
    params = [tuple(jnp.asarray(d[f"p{k}_{j}"]) for j in range(int(d[f"n{k}"])))
              for k in range(int(d["n_params"]))]
    if swap_ls:
        # the first two blocks' LayerScale γ exchanged: same shapes, a different net
        ls = [k for k, p in enumerate(params) if len(p) == 1]
        params[ls[0]], params[ls[1]] = params[ls[1]], params[ls[0]]
    y = np.asarray(fwd(params, jnp.asarray(d[f"x{res}"])))
    ref = d[f"y{res}"]
    return float(np.max(np.abs(y - ref)) / max(1e-12, np.max(np.abs(ref))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    # 1e-4: agreement is ~1e-6 at 224; erf against tanh GELU is ~1e-3 over 18–36 blocks
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--paper", action="store_true", help="compare against timm's own erf GELU")
    a = ap.parse_args()
    if not os.path.exists(TIMM_PY):
        sys.exit(f"⛔ {TIMM_PY} missing — the pinned timm env (requirements-timm-lock.txt)")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")

    target = ("--gelu", "erf") if a.paper else ()
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        for label, mod, spec, cfg, model in NETS:
            print(f"[{label}] {spec} / {cfg}")
            fwd, keeps = load_forward(emit(tmp, mod, spec, cfg))
            d = timm_dump(tmp, model, a.batch, a.seed, *target)
            for res in (224, 288):
                e = run(fwd, d, res)
                ok = e <= a.tol
                fails += not ok
                print(f"  @ {res}: max|Δ|/max|timm| = {e:.3e}  {'✅' if ok else '⛔'}")
            drop = d["drop_probs"]
            e = float(np.max(np.abs((1.0 - np.array(keeps)) - drop))) if len(keeps) == len(drop) else 1.0
            ok = e < 1e-6
            fails += not ok
            print(f"  drop-path ramp: {len(keeps)} keeps, max|Δ drop prob| = {e:.1e}  {'✅' if ok else '⛔'}")
            if a.controls and label == "T" and not a.paper:
                for why, dd, swap in (("erf GELU against the reference's tanh",
                                       timm_dump(tmp, model, a.batch, a.seed, "--gelu", "erf"), False),
                                      ("two blocks' LayerScale swapped", d, True)):
                    e = run(fwd, dd, 224, swap)
                    red = e > a.tol
                    fails += not red
                    print(f"  CONTROL {why}: {e:.3e}  "
                          f"{'✅ red, as it must be' if red else '⛔ GREEN — the gate is blind to this'}")
    if fails:
        sys.exit(f"⛔ {fails} check(s) failed at tolerance {a.tol}")
    print("✅ the JAX ConvNeXt-T/S/B references compute timm's convnext_{tiny,small,base}"
          + (" with timm's own erf GELU" if a.paper else " (tanh GELU)"))


if __name__ == "__main__":
    main()
