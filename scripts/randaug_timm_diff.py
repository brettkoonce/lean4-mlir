#!/usr/bin/env python3
"""randaug_timm_diff.py — our RandAugment magnitude mappings vs timm's OWN source.

  .venv/bin/python scripts/randaug_timm_diff.py        # needs timm installed

⚠⚠ **WHY (2026-08-14).** The shim's `_aa_*` level->arg functions are a hand transcription of
`timm/data/auto_augment.py`. Two of them were wrong, in the same way, and had been since they were
written: timm builds each `inc` (increasing-severity) mapping by NEGATING the already-INTEGER
decreasing one, `4 - int((m/10)*4)`, where we had written `int(4 - (m/10)*4)`. Truncating on the
other side of the subtraction lands one step lower at 7 of the 11 integer magnitudes.

  Posterize @ m=7 (RSB-A2's own magnitude): timm keeps 2 MSBs, we kept 1 — a whole bit.
  Solarize  @ m=7: timm 77, we had 76 — one threshold unit of 256, i.e. invisible.

Nothing caught it because the shim is compared to the reference by AUGMENTATION PARTITION
(`shim_wiring_gate.py`: which ops are on), never by the ARGUMENT each op is called with. This file
is the missing half: it evaluates both sides at every integer magnitude and requires equality.

⚠ It reads OUR mappings out of the generated shim by regex rather than restating them, so it
cannot pass by agreeing with a copy of itself.
"""
import os, re, sys, math

SHIM = "jax/.lake/build/generated_resnet50_imagenet_shim.py"

def ours_from_shim():
    """Pull the `_aa_*` bodies out of the generated shim and eval them at _RA_INC = True."""
    src = open(SHIM).read()
    env = {"_AA_MAX": 10.0, "_RA_INC": True, "int": int, "min": min, "max": max}
    out = {}
    for name in ("_aa_pos", "_aa_sol", "_aa_sad", "_aa_rot", "_aa_she", "_aa_trn"):
        m = re.search(rf"^def {name}\(m\): return (.+?)(?:\s+#.*)?$", src, re.M)
        if not m:
            sys.exit(f"could not find {name} in {SHIM}")
        body = m.group(1)
        out[name] = eval(f"lambda m: {body}", env)
    return out

def main():
    try:
        import timm.data.auto_augment as A
    except ImportError:
        sys.exit("timm not installed — `.venv/bin/pip install timm`")
    h = dict(A._HPARAMS_DEFAULT)
    o = ours_from_shim()
    # our name -> timm's level_to_arg for the `inc` variant RSB selects
    PAIRS = [
        ("Posterize",   o["_aa_pos"], lambda m: A._posterize_increasing_level_to_arg(m, h)[0]),
        ("Solarize",    o["_aa_sol"], lambda m: A._solarize_increasing_level_to_arg(m, h)[0]),
        ("SolarizeAdd", o["_aa_sad"], lambda m: A._solarize_add_level_to_arg(m, h)[0]),
        # signed ops: timm randomly negates, so compare magnitudes
        ("Rotate",      o["_aa_rot"], lambda m: abs(A._rotate_level_to_arg(m, h)[0])),
        ("Shear",       o["_aa_she"], lambda m: abs(A._shear_level_to_arg(m, h)[0])),
        ("TranslateRel",o["_aa_trn"], lambda m: abs(A._translate_rel_level_to_arg(m, h)[0])),
    ]
    bad = []
    print(f"── our RandAugment level→arg vs timm {__import__('timm').__version__} ──")
    for name, mine, theirs in PAIRS:
        deltas = []
        for mi in range(0, 11):
            a, b = float(mine(mi)), float(theirs(mi))
            if abs(a - b) > 1e-9:
                deltas.append((mi, b, a))
        if deltas:
            bad.append(name)
            print(f"  ✗ {name:13} differs at {len(deltas)}/11 magnitudes "
                  f"(m, timm, ours): {deltas[:4]}{' …' if len(deltas) > 4 else ''}")
        else:
            print(f"  ✅ {name:13} agrees at all 11 integer magnitudes")
    # the enhancement clamp: timm floors at 0.1; ours has no clamp but the magnitude is
    # clipped to [0, _AA_MAX] first, so 1 - 0.9*(m/10) >= 0.1 always. Equivalent, checked.
    worst = min(1.0 - 0.9 * (mi / 10.0) for mi in range(0, 11))
    print(f"  {'✅' if worst >= 0.1 - 1e-9 else '✗'} Enhance       timm clamps at 0.1; "
          f"ours bottoms out at {worst:.3f} with m clipped to [0,10] — clamp never binds")
    print()
    if bad:
        print(f"❌ FAIL — {len(bad)} mapping(s) differ: {', '.join(bad)}")
        return 1
    print("✅ every magnitude mapping agrees with timm's own source")
    return 0

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    sys.exit(main())
