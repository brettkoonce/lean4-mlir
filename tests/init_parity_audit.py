#!/usr/bin/env python3
"""init_parity_audit.py — does each verified ImageNet net START where its JAX reference starts?

    python3 tests/init_parity_audit.py <specs.jsonl>          # specs from the Lean dump below
    python3 tests/init_parity_audit.py <specs.jsonl> --all    # every tensor, not just mismatches

WHY (2026-10-01). The MNv2 350-epoch pair trails by a steady −0.5 and the leading suspect is init:
`mkParam`'s rank-4 rule is He fan-OUT `2/(dims[0]·k²)`, which for a depthwise kernel `(C,1,k,k)`
is `2/(C·k²)`, while every JAX depthwise emitter hard-codes fan = k². The `mkParam` docstring says
it mirrors `jax/Jax/Codegen.lean`; nothing checked that, per tensor, per net. This does.

HOW. Both sides are read from the code that actually runs, not from a description of it:
  * JAX: import the generated reference module (its `Main` is `__main__`-guarded) and CALL its own
    `init_params` — so every emitter form (uniform ±√(6/fan), normal·0.02, zero-γ, layer scale,
    ViT's PyTorch-default patch embed) is measured, not parsed. Variance is empirical.
  * verified: `VerifiedNetSpec.toSpecs` dumped from Lean (dims + init kind, func-arg order), with
    `mkParam`'s variance rule applied analytically, including the per-net flags
    (`cnxInit`, `vitInit`, `dwFanK2`) the drivers set.
Distribution is NOT compared: `F32.heInit` is Bates-3 (≈normal) where JAX draws one uniform —
a documented, deliberate gap. Only the first two moments are.

Spec dump (scratch; uses existing oleans, rebuilds nothing):

    import LeanMlir.Verified.NetsCore
    def dumpOne (s : VerifiedNetSpec) : String := … s.toSpecs … (see planning/init_parity.md §A)

Read-only. Runs on CPU in seconds (JAX_PLATFORMS=cpu is forced).
"""
import importlib.util
import json
import os
import re
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(REPO, "jax/.lake/build")

# verified slug -> (JAX reference module, verified config flags). The reference is the one the
# pair's JAX run trained (scripts/jobs/*jax*.conf, or the run's RESULTS.md where no conf exists).
PAIRS = {
    "resnet34in":     ("generated_resnet34_imagenet.py", {}),
    # MainResnet50Imagenet's config sets zeroGammaInit (every recipe, 224 and 160).
    "resnet50in":     ("generated_resnet50_imagenet_2018.py", {"zeroGamma": True}),
    "resnet50in160":  ("generated_resnet50_imagenet_rsbfaithful.py", {"zeroGamma": True}),
    "mobilenetv2in":  ("generated_mobilenet_v2_imagenet_full.py", {}),
    "efficientnetin": ("generated_efficientnet_b0_imagenet_full.py", {}),
    "mnv4in":         ("generated_mobilenet_v4_imagenet.py", {}),
    "convnextin":     ("generated_convnext_tiny_imagenet_full.py", {"cnxInit": True}),
    "vitin":          ("generated_vit_tiny_imagenet.py", {"vitInit": True}),   # `default` carries DeiT init
}


def verified_moments(dims, kind, flags):
    """(mean, variance) of `mkParam seed dims kind …` — LeanMlir/Verified/Train.lean, mkParam."""
    if kind == 1:
        return 1.0, 0.0
    if kind == 3:
        return 1e-6, 0.0
    if kind == 2:
        return 0.0, 0.0
    if kind == 4:   # zero-γ (a bottleneck's residual-closing BN): 0 under zeroGammaInit, else 1
        return (0.0, 0.0) if flags.get("zeroGamma") else (1.0, 0.0)
    if kind == 5:   # embedding (ViT CLS / pos): σ 0.02 under vitInit, zeros without it
        return (0.0, 0.0004) if flags.get("vitInit") else (0.0, 0.0)
    if flags.get("cnxInit"):
        return 0.0, 0.0004
    if flags.get("vitInit"):
        if len(dims) == 4:
            return 0.0, 1.0 / (3.0 * dims[1] * dims[2] * dims[3])
        return 0.0, 0.0004
    if flags.get("dwFanK2") and len(dims) == 4 and dims[1] == 1 and dims[0] > 1:
        return 0.0, 2.0 / (dims[2] * dims[3])
    if len(dims) == 4:
        return 0.0, 2.0 / (dims[0] * dims[2] * dims[3])
    if len(dims) == 2:
        return 0.0, 2.0 / (dims[0] + dims[1])
    return 0.0, 2.0 / dims[0]


def jax_leaves(path):
    import jax
    spec = importlib.util.spec_from_file_location("ref_" + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    params = mod.init_params(jax.random.PRNGKey(0))
    return [np.asarray(l, np.float64) for l in jax.tree.leaves(params)]


def classify(d):
    if len(d) == 4:
        return "dw" if (d[1] == 1 and d[0] > 1) else ("conv1x1" if d[2] == d[3] == 1 else f"conv{d[2]}x{d[3]}")
    return {2: "dense", 1: "vec"}.get(len(d), f"rank{len(d)}")


def audit(slug, specs, show_all):
    ref, flags = PAIRS[slug]
    J = jax_leaves(os.path.join(GEN, ref))
    print(f"\n━━ {slug}  ↔  {ref}  flags={flags or '-'}  ({len(specs)} verified / {len(J)} JAX tensors)")
    if len(J) != len(specs):
        print(f"  ⛔ tensor COUNT differs — cannot align by order; skipping")
        return None
    rows, bad = [], 0
    for i, ((dims, kind), j) in enumerate(zip(specs, J)):
        shp = list(j.shape)
        # Same tensor, three spellings: identical; a dense stored transposed; or EfficientNet's SE
        # FCs, which verified carries as a rank-2 dense `[in, out]` and JAX as a 1×1 conv
        # `[out, in, 1, 1]` (the two are the same linear map).
        se_as_conv = len(dims) == 2 and shp == [dims[1], dims[0], 1, 1]
        if shp != dims and not (len(dims) == 2 and shp == dims[::-1]) and not se_as_conv:
            print(f"  ⛔ #{i}: shape {dims} vs JAX {shp} — order diverges here; stopping")
            return None
        vm, vv = verified_moments(dims, kind, flags)
        jm, jv = float(j.mean()), float(j.var())
        # scale-aware: weights compare std ratio; constants compare value
        if vv == 0.0 and jv < 1e-12:
            ok = abs(vm - jm) <= 1e-6 * max(1.0, abs(jm))
            ratio = None
        else:
            ratio = np.sqrt(vv / jv) if jv > 0 else float("inf")
            ok = abs(ratio - 1.0) < 0.05          # empirical-variance noise is <1% on these sizes
        if not ok:
            bad += 1
        rows.append((i, classify(dims), dims, kind, vm, vv, jm, jv, ratio, ok))
    groups = {}
    for r in rows:
        if not r[9]:
            key = (r[1], r[3], round(r[8], 3) if r[8] is not None else f"const {r[4]:g} vs {r[6]:g}")
            groups.setdefault(key, []).append(r)
    if not bad:
        print("  ✅ every tensor matches (std ratio within 5%, constants exact)")
    for (cls, kind, ratio), rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ex = rs[0]
        what = (f"std ratio verified/JAX {ratio}" if isinstance(ratio, float)
                else ratio)
        print(f"  ✗ {len(rs):3d} × {cls:8s} kind {kind}: {what}   e.g. #{ex[0]} {ex[2]}  "
              f"verified var {ex[5]:.3g}  JAX var {ex[7]:.3g} mean {ex[6]:.3g}")
    if show_all:
        for r in rows:
            print(f"    #{r[0]:3d} {r[1]:8s} {str(r[2]):22s} k{r[3]}  ver ({r[4]:.3g},{r[5]:.3g})  "
                  f"jax ({r[6]:.3g},{r[7]:.3g})  {'ok' if r[9] else 'MISMATCH'}")
    return {"slug": slug, "ref": ref, "n": len(rows), "mismatched": bad,
            "groups": {f"{k[0]}|k{k[1]}|{k[2]}": len(v) for k, v in groups.items()}}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    show_all = "--all" in sys.argv
    out = []
    for line in open(sys.argv[1]):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d["slug"] not in PAIRS:
            print(f"(no JAX pair registered for {d['slug']}; skipped)")
            continue
        res = audit(d["slug"], [(s[0], s[1]) for s in d["specs"]], show_all)
        if res:
            out.append(res)
    print("\nsummary:")
    for r in out:
        print(f"  {r['slug']:15s} {r['mismatched']:3d}/{r['n']} tensors start differently")


if __name__ == "__main__":
    main()
