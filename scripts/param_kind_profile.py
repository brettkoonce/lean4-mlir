#!/usr/bin/env python3
"""Per-parameter-KIND magnitude profile of a checkpoint.

Provenance for the `w'` / `gl` / `bb` bounds the float-budget folds are stated at
(`scripts/float_budget_envelope.py`, `planning/float_budget_numbers.md` §3.3(a), §3.7, §3.9).
A single uniform `|param| ≤ B` is the maximum over every stored f32, and that maximum is
routinely on a kind the conv fan-in does NOT multiply — ConvNeXt-T's layer scale is 14× its
conv kernels, and splitting is the difference between a statable number and none.

⭐ Which kind is the outlier is NOT predictable: on ResNet-34 and EfficientNet-B0 the maximum
is a BatchNorm γ and the kernels are tighter; on MobileNetV2 the maximum IS a kernel. Measure.

The layout comes from the net's own generated `init_params_from_file`, parsed rather than
re-derived, so this cannot drift from the loader that actually reads the checkpoint.

Run: python3 scripts/param_kind_profile.py [<loader.py> <checkpoint.bin>]
"""
import re
import sys

import numpy as np

# The generated loaders name their slots; these are the kinds the budgets bound separately.
KIND = {'W': 'kernel', 'gamma': 'bn_gamma', 'beta': 'bn_beta', 'b': 'bias'}

DEFAULTS = [
    ("ResNet-34", "jax/generated/generated_resnet34_imagenet_short.py",
     "/home/skoonce/resnet/r34_imagenet_bf16_e79.bin"),
    ("MobileNetV2", "jax/generated/generated_mobilenet_v2_imagenet.py",
     "/home/skoonce/mnv2_350ep/mobilenet_v2_imagenet.bin"),
    ("EfficientNet-B0", "jax/generated/generated_efficientnet_b0_imagenet.py",
     "/home/skoonce/enet_b0_350_4gpu/efficientnet_b0_imagenet.bin"),
]


def slots(loader_path):
    """(slot name, element count) in file order, read off `init_params_from_file`'s
    `<name> = jnp.array(buf[idx:idx+N]…)` lines."""
    body = open(loader_path).read().split("def init_params_from_file", 1)[1].split("\ndef ", 1)[0]
    return [(m.group(1), int(m.group(2)))
            for m in re.finditer(r"^\s*(\w+) = jnp\.array\(buf\[idx:idx\+(\d+)\]", body, re.M)]


def profile(loader_path, ckpt_path):
    buf = np.fromfile(ckpt_path, dtype=np.float32)
    i, kinds = 0, {}
    for name, n in slots(loader_path):
        seg = np.abs(buf[i:i + n])
        i += n
        k = KIND.get(name, name)
        cnt, mx = kinds.get(k, (0, 0.0))
        kinds[k] = (cnt + n, max(mx, float(seg.max()) if n else 0.0))
    return i, len(buf), kinds


def report(tag, loader_path, ckpt_path):
    used, total, kinds = profile(loader_path, ckpt_path)
    ok = "OK" if used == total else f"⛔ MISMATCH — the loader spells {used}"
    print(f"\n{tag}: {total:,} f32   ({ok})")
    for k, (c, mx) in sorted(kinds.items(), key=lambda kv: -kv[1][0]):
        print(f"   {k:<10} {c:>12,}   max |·| = {mx:.4f}")
    print(f"   {'UNIFORM':<10} {sum(c for c, _ in kinds.values()):>12,}   max |·| = "
          f"{max(mx for _, mx in kinds.values()):.4f}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        report(sys.argv[2], sys.argv[1], sys.argv[2])
    else:
        for tag, loader, ckpt in DEFAULTS:
            try:
                report(tag, loader, ckpt)
            except FileNotFoundError as ex:
                print(f"\n{tag}: skipped — {ex.filename} not present")
