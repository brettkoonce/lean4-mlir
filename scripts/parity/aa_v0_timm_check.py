#!/usr/bin/env python3
"""aa_v0_timm_check.py — EfficientNet-B0's emitted AutoAugment against timm's policy v0.

  .venv/bin/python scripts/parity/aa_v0_timm_check.py [jax/generated/<an autoAugmentV0 trainer>.py]

EfficientNet-B0 trains on TF TPU EfficientNet's AutoAugment policy v0, which timm 1.0.28 (the pinned
spec) carries as `auto_augment_policy_v0` and builds through `create_transform(auto_augment='v0')`.
The shim's default table (`autoAugmentPy` in jax/Jax/Codegen.lean) is the 2018 paper's sub-policies
with TF-v0 Posterize levels, which is neither; `TrainConfig.autoAugmentV0` emits `aaV0Py` over it.

Checks, every one against timm's own objects (run in .venv-timm, read back as JSON):
  1. the 25 sub-policies: names, probabilities, magnitudes, in order;
  2. level -> arg for every op the table uses, at magnitudes 0..10 (timm's LEVEL_TO_ARG; abs value
     for the randomly negated ones), and the shim's sign flag = whether timm randomly negates;
  3. the fill = the fillcolor timm's create_transform hands every op (img_mean for B0's mean);
  4. Posterize pixels against PIL's ImageOps.posterize (what timm calls) at the table's bit counts,
     0 bits included: timm's "results in black image".
Control: the same checks on the block without the v0 override (the default table) must fail.

Our side is exec'd out of the generated file, never restated, so it cannot agree with a copy of
itself. Pixel parity of the geometric ops (bicubic, fill) is scripts/gates/aug_bicubic_pil_check.py.
"""
import json, os, subprocess, sys
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import tensorflow as tf
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIMM_PY = os.path.join(ROOT, ".venv-timm", "bin", "python")
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "jax", "generated", "generated_efficientnet_b0_imagenet_full.py")
V0_MARK = "# ── AutoAugment policy v0"

TIMM_SIDE = r'''
import json, random, timm
import timm.data.auto_augment as A
from timm.data import create_transform
h = dict(A._HPARAMS_DEFAULT)
pol = A.auto_augment_policy_v0(h)
policy = [[[op.name, op.prob, op.magnitude] for op in sp] for sp in pol]
names = sorted({op.name for sp in pol for op in sp})
levels, negated = {}, {}
for n in names:
    fn = A.LEVEL_TO_ARG[n]
    if fn is None:
        levels[n] = None; negated[n] = False; continue
    random.seed(0)
    signs = set()
    for _ in range(64):
        v = fn(10, h)[0]; signs.add(v > 0)
    negated[n] = len(signs) == 2
    levels[n] = [abs(fn(m, h)[0]) for m in range(11)]
cfg = timm.models.get_pretrained_cfg("tf_efficientnet_b0").to_dict()
t = create_transform(224, is_training=True, auto_augment="v0", interpolation="bicubic",
                     mean=cfg["mean"], std=cfg["std"])
aa = next(x for x in t.transforms if type(x).__name__ == "AutoAugment")
fills = {tuple(op.kwargs["fillcolor"]) for sp in aa.policy for op in sp}
interp = {str(op.kwargs["resample"]) for sp in aa.policy for op in sp}
print(json.dumps(dict(version=timm.__version__, policy=policy, levels=levels, negated=negated,
                      fill=sorted(fills), interp=sorted(interp), mean=cfg["mean"])))
'''


def timm_side():
    if not os.path.exists(TIMM_PY):
        sys.exit(f"⛔ {TIMM_PY} missing — build .venv-timm from requirements-timm-lock.txt")
    r = subprocess.run([TIMM_PY, "-c", TIMM_SIDE], capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    if r.returncode:
        sys.exit("⛔ timm side failed:\n" + r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def load_block(src_lines, with_v0):
    a = next(i for i, l in enumerate(src_lines) if l.startswith("_AA_MAX"))
    b = next(i for i, l in enumerate(src_lines) if l.startswith("def _imagenet_decode_random_crop_flip"))
    c = next((i for i in range(a, b) if src_lines[i].startswith(V0_MARK)), None)
    if with_v0 and c is None:
        sys.exit(f"⛔ {path} carries no v0 block — not an autoAugmentV0 recipe")
    g = {"tf": tf, "os": os, "np": np}
    exec("\n".join(src_lines[a:b] if with_v0 else src_lines[a:c if c is not None else b]), g)
    return g


rng = np.random.default_rng(0)
IMG = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


def check(g, T, label):
    bad = []
    ours = [[list(op) for op in sp] for sp in g["_AA_POLICY"]]
    theirs = T["policy"]
    if len(ours) != len(theirs):
        bad.append(f"table has {len(ours)} sub-policies, timm {len(theirs)}")
    nrow = sum(1 for o, t in zip(ours, theirs)
               if [[n, float(p), int(m)] for n, p, m in o] != [[n, float(p), int(m)] for n, p, m in t])
    if nrow:
        bad.append(f"{nrow}/{len(theirs)} sub-policies differ from timm v0")
    for n, lv in T["levels"].items():
        if n not in g["_AA_OPS"]:
            bad.append(f"{n}: not in the op registry"); continue
        fn, argfn, signed = g["_AA_OPS"][n]
        if lv is None:
            if argfn is not None: bad.append(f"{n}: takes a level, timm's takes none")
            continue
        if argfn is None:
            bad.append(f"{n}: takes no level, timm's does"); continue
        mine = [float(argfn(float(m))) for m in range(11)]
        d = [m for m in range(11) if abs(mine[m] - float(lv[m])) > 1e-6]
        if d:
            bad.append(f"{n}: level->arg differs at m={d} (ours {[mine[m] for m in d]}, timm {[lv[m] for m in d]})")
        if bool(signed) != bool(T["negated"][n]):
            bad.append(f"{n}: sign flag {signed}, timm negates={T['negated'][n]}")
    fill = tuple(int(v) for v in g["_AA_FILL"].numpy()) if "_AA_FILL" in g else (128, 128, 128)
    if [list(fill)] != T["fill"]:
        bad.append(f"fill {fill}, timm create_transform {T['fill']}")
    pfn, pargs, _ = g["_AA_OPS"]["Posterize"]
    for m in sorted({m for sp in theirs for n, _, m in sp if n == "Posterize"}):
        bits = int(pargs(float(m)))
        ref = np.asarray(ImageOps.posterize(Image.fromarray(IMG), bits)) if bits < 8 else IMG
        got = pfn(tf.constant(IMG), bits).numpy()
        if not np.array_equal(got, ref):
            bad.append(f"Posterize m={m} ({bits} bits): {int((got != ref).sum())} values differ from PIL")
    print(f"── {label} ──")
    for b in bad:
        print(f"  ✗ {b}")
    if not bad:
        print("  ✓ table, level->arg, signs, fill and posterize pixels equal timm v0")
    return not bad


def main():
    T = timm_side()
    print(f"timm {T['version']} (pinned spec); B0 mean {T['mean']}, fill {T['fill']}, interp {T['interp']}")
    src = open(path).read().split("\n")
    ok = check(load_block(src, True), T, f"emitted v0  ({os.path.relpath(path, ROOT)})")
    ctl = not check(load_block(src, False), T, "CONTROL: default table (no v0 override)")
    print(f"CONTROL {'✅ red, as it must be' if ctl else '⛔ GREEN — the gate is blind'}")
    if not (ok and ctl):
        sys.exit("⛔ the emitted AutoAugment is not timm's policy v0")
    print("✅ the emitted AutoAugment is timm's policy v0, op for op")


if __name__ == "__main__":
    main()
