#!/usr/bin/env python3
"""ViT timm parity: the JAX references against timm's `deit_{tiny,small,base}_patch16_224`, on SHARED weights
(planning/imagenet_parity.md G3).

Nothing compared the ViT reference with an independent ViT: `vit-dp-check` and the forward ties
compare the render against the reference, both built from the same spec. This runs both JAX emitters
(`jax/MainVit.lean`, 10-class; `jax/MainVitImagenet.lean`, 1000-class) — re-emitted from the CURRENT
source into a scratch directory, bf16 dtypes forced to f32, CPU — against timm's DeiT-Ti.

The reference computes tanh GELU and LayerNorm ε 1e-5 where DeiT uses erf and 1e-6 (§2.1, disclosed).
The gate pins what we compute (timm built with tanh / 1e-5) so any OTHER drift is caught, and
`--deit` reports the distance to DeiT's own settings. ViT has no BatchNorm and runs drop-free here,
so one forward per net is the whole check.

**The verified half** (2026-09-29): the ImageNet forwards `ViTRenderB` writes (`vitFwdRenderB` at
`vitTiDims` / `vitSDims` / `vitBDims`, i.e. `vitin_fwd`, `vitsin_fwd`, `vitbin_fwd`) are rendered at
the gate's batch into the scratch directory and run through iree-compile on CPU against the same
timm dump. The render stores every Linear as [in, out], CLS as [D] and pos as [197, D]; those are
the only transforms. `--skip-verified` runs the JAX half alone (no IREE).

Usage:
    .venv/bin/python scripts/parity/vit_timm_parity.py              # batch 4, seed 0, tol 1e-5 of max |logit|
    .venv/bin/python scripts/parity/vit_timm_parity.py --controls   # also RED on k/v swapped and on erf GELU
    .venv/bin/python scripts/parity/vit_timm_parity.py --deit       # distance to DeiT's erf / 1e-6
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIMM_PY = os.path.join(ROOT, ".venv-timm", "bin", "python")
EMIT = """import {mod}
#eval IO.FS.writeFile "{out}" (JaxCodegen.generate {spec} {cfg} {ds} "data")
"""
NETS = [
    # (label, module, spec, config, dataset, classes, timm model, verified (slug, VitDims) or None)
    ("imagenette", "MainVit", "vitTiny", "vitConfig", ".imagenette", 10, "deit_tiny_patch16_224", None),
    ("imagenet", "MainVitImagenet", "vitTinyImagenet", "vitTinyImagenetConfig", ".imagenet", 1000,
     "deit_tiny_patch16_224", ("vitin", "vitTiDims")),
    # the side-quest widths: the same emitter at D = 384 / 768
    ("imagenet-S", "MainVitSImagenet", "vitSImagenet", "vitSImagenetConfig", ".imagenet", 1000,
     "deit_small_patch16_224", ("vitsin", "vitSDims")),
    ("imagenet-B", "MainVitBImagenet", "vitBImagenet", "vitBImagenetConfig", ".imagenet", 1000,
     "deit_base_patch16_224", ("vitbin", "vitBDims")),
]
EMIT_VERIFIED = """import LeanMlir.Proofs.Codegen.ViTRenderB
#eval IO.FS.writeFile "{out}" (Proofs.StableHLO.vitFwdRenderB "{slug}_fwd" 1000 (vbB := {B}) (V := Proofs.StableHLO.{dims}))
"""


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
    for k in ("DT", "CONV_DT"):
        if k in mod:
            mod[k] = jnp.float32
    return mod["forward"]


def timm_dump(tmp, model, classes, batch, seed, *flags):
    out = os.path.join(tmp, f"timm_{model}_{classes}_{'_'.join(flags) or 'ours'}.npz")
    r = subprocess.run([TIMM_PY, os.path.join(ROOT, "scripts", "parity", "_vit_timm_dump.py"), out,
                        "--model", model, "--classes", str(classes), "--batch", str(batch),
                        "--seed", str(seed), *flags], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"⛔ timm dump failed:\n{r.stdout}{r.stderr}")
    print("  " + r.stdout.strip())
    return np.load(out)


def run(fwd, d):
    import jax.numpy as jnp
    params = [tuple(jnp.asarray(d[f"p{k}_{j}"]) for j in range(int(d[f"n{k}"])))
              for k in range(int(d["n_params"]))]
    y = np.asarray(fwd(params, jnp.asarray(d["x"])))
    return float(np.max(np.abs(y - d["y"])) / max(1e-12, np.max(np.abs(d["y"]))))


def run_verified(tmp, slug, dims, d, batch):
    """The render at the dump's batch through IREE on CPU; relative error against timm's logits."""
    sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
    import _iree
    mlir = os.path.join(tmp, f"{slug}_fwd.mlir")
    src = os.path.join(tmp, f"Emit_{slug}.lean")
    with open(src, "w") as f:
        f.write(EMIT_VERIFIED.format(out=mlir, slug=slug, dims=dims, B=batch))
    r = subprocess.run(["lake", "env", "lean", src], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(mlir):
        sys.exit(f"⛔ rendering {slug}_fwd failed\n{r.stdout}{r.stderr}")
    groups = [[d[f"p{k}_{j}"] for j in range(int(d[f"n{k}"]))] for k in range(int(d["n_params"]))]
    flat = []
    for k, g in enumerate(groups):
        for arr in g:
            if k == 1 or k == 2:                  # CLS [1,1,D] -> [D], pos [1,197,D] -> [197,D]
                arr = arr.reshape(arr.shape[-2:] if k == 2 else arr.shape[-1:])
            elif arr.ndim == 2:                   # every Linear: timm's [out, in] -> the render's [in, out]
                arr = arr.T
            flat.append(np.ascontiguousarray(arr, dtype=np.float32))
    x = np.ascontiguousarray(d["x"].reshape(batch, -1), dtype=np.float32)
    work = os.path.join(tmp, f"iree_{slug}")
    os.makedirs(work, exist_ok=True)
    y = np.asarray(_iree.compile_and_run(mlir, f"{slug}_fwd", [x] + flat, work, 1, "llvm-cpu")[0],
                   dtype=np.float64)
    return float(np.max(np.abs(y - d["y"])) / max(1e-12, np.max(np.abs(d["y"]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    # 1e-5, not the CNN gates' 1e-3: agreement is ~7e-7, and tanh vs erf GELU is only ~4e-5 of scale
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--skip-verified", action="store_true", help="the JAX half only (no IREE)")
    ap.add_argument("--deit", action="store_true", help="compare against DeiT's erf GELU / LN ε 1e-6")
    a = ap.parse_args()
    if not os.path.exists(TIMM_PY):
        sys.exit(f"⛔ {TIMM_PY} missing — the pinned timm env (requirements-timm-lock.txt)")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")

    target = ("--gelu", "erf", "--ln-eps", "1e-6") if a.deit else ()
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        for label, mod, spec, cfg, ds, classes, model, verified in NETS:
            print(f"[{label}] {spec} / {cfg}")
            fwd = load_forward(emit(tmp, mod, spec, cfg, ds))
            d = timm_dump(tmp, model, classes, a.batch, a.seed, *target)
            e = run(fwd, d)
            ok = e <= a.tol
            fails += not ok
            print(f"  jax      max|Δ|/max|timm| = {e:.3e}  {'✅' if ok else '⛔'}")
            if verified and not a.skip_verified:
                e = run_verified(tmp, *verified, d, a.batch)
                ok = e <= a.tol
                fails += not ok
                print(f"  verified max|Δ|/max|timm| = {e:.3e}  {'✅' if ok else '⛔'}  ({verified[0]}_fwd, iree llvm-cpu)")
            if a.controls and label == "imagenet":
                for why, flags in (("k and v swapped", ("--swap-kv",)),
                                   ("erf GELU against the reference's tanh", ("--gelu", "erf"))):
                    e = run(fwd, timm_dump(tmp, model, classes, a.batch, a.seed, *flags))
                    red = e > a.tol
                    fails += not red
                    print(f"  CONTROL {why}: {e:.3e}  "
                          f"{'✅ red, as it must be' if red else '⛔ GREEN — the gate is blind to this'}")
    if fails:
        sys.exit(f"⛔ {fails} check(s) failed at tolerance {a.tol}")
    print("✅ the JAX ViT-Ti/S/B references" + ("" if a.skip_verified else " and the verified renders")
          + " compute timm's deit_{tiny,small,base}_patch16_224"
          + (" at DeiT's own settings" if a.deit else " (tanh GELU, LN ε 1e-5)"))


if __name__ == "__main__":
    main()
