"""Does R50's stochastic-depth site sit on the RESIDUAL BRANCH, or on the block OUTPUT?

⚠⚠ **NO STRUCTURAL CHECK IN THIS REPO CAN TELL THEM APART.** The misplaced render has the same SSA
names, the same order, the same types, the same op counts and the same arity — and at an ALL-ONES
mask it computes the same function bit-for-bit, because `1 ⊙ (branch + x) = branch + x`. So every
endpoint gate, every prefix audit and every arity check passes on it (`stochastic_depth.md` §7b,
measured on EfficientNet). Only a numeric run at a NON-ones mask separates them.

    correct     drop(branch) + skip   ->  s*branch + x
    misplaced   drop(branch + skip)   ->  s*(branch + x)

Run at q = 1 (32x32) and B = 2 so it fits on CPU. Same renderer, same 16 sites; the placement is a
property of the emitter, not of the resolution.

    .venv/bin/python runs/2026-08-27-r50-a2-a1-ema-fifth-region/drop_site_placement.py <dir>
"""
import os, sys, zlib
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax
from jax._src import xla_bridge
from jax._src.lib import xla_client as xc
from jax._src.interpreters import mlir as jmlir
import jaxlib.mlir.ir as ir
import re

D = sys.argv[1] if len(sys.argv) > 1 else "."
B, NC, NDROP = 2, 10, 16


def seed_of(name):
    """⚠ `crc32`, NOT `hash()`. Python salts `hash()` per PROCESS, so the two renders got the same
    inputs within one run but different ones between runs — the TEST figure moved four orders
    between two invocations of this file. A gate whose number is not reproducible is a demo."""
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def load(path):
    txt = open(path).read()
    txt = re.sub(r'func\.func @\w+\(', 'func.func @main(', txt, count=1)
    backend = xla_bridge.get_backend()
    devices = xc.DeviceList(tuple(backend.local_devices()[:1]))
    ctx = jmlir.make_ir_context()
    with ctx, ir.Location.unknown(ctx):
        return backend.compile_and_load(ir.Module.parse(txt),
                                        executable_devices=devices,
                                        compile_options=xc.CompileOptions())


def arg_shapes(path):
    m = re.search(r'func\.func @\w+\((.*?)\) -> \(', open(path).read(), re.S)
    out = []
    for a in m.group(1).split(', '):
        name, ty = a.split(': ')
        # ⚠ `re.findall(r'\d+', ...)` picks the 32 out of `f32`. Take the numeric leading
        # components only — `tensor<f32>` is rank 0 and must come back as ().
        body = ty.split('tensor<')[1].rstrip('> ')
        dims = tuple(int(d) for d in body.split('x')[:-1])
        out.append((name.strip(), dims))
    return out


def run(path, masks, rng):
    exe = load(path)
    args = []
    for name, shp in arg_shapes(path):
        if name.startswith('%dp'):
            args.append(np.full(shp, masks[int(name[3:])], np.float32))
        elif name == '%onehot':
            oh = np.zeros(shp, np.float32); oh[:, 0] = 1.0; args.append(oh)
        elif name == '%lr':
            args.append(np.float32(0.01))
        elif name in ('%bc1', '%bc2'):
            args.append(np.float32(0.5))
        elif name.endswith('v') and shp:
            # ⚠⚠ **THE `v` REGION MUST BE NON-NEGATIVE.** AdamW's update takes `sqrt(v/bc2)`, so a
            # random-signed second moment gives NaN for every parameter whose slot went negative —
            # 50 of 161 on the first run of this file, with a perfectly finite loss beside them.
            # ⚠ No parameter name in `r50SigList` ends in `v`, so this cannot catch a weight.
            r = np.random.default_rng(seed_of(name))
            args.append(np.abs(r.standard_normal(shp) * 0.05).astype(np.float32))
        elif re.match(r'^%(s|s\d+b\d+)g(\d|p)?$', name):
            # ⚠ BN GAMMAS AT 1, NOT AT A SMALL RANDOM. At gamma ~ 0 every BN output is ~ 0, the
            # softmax is uniform and the whole graph returns NaN through the log — which is what
            # the first run of this file did, and a NaN is not a small difference, it is no answer.
            args.append(np.ones(shp, np.float32))
        else:
            r = np.random.default_rng(seed_of(name))
            args.append((r.standard_normal(shp) * 0.05).astype(np.float32) if shp
                        else np.float32(0.0))
    return [np.asarray(o) for o in exe.execute([jax.device_put(a) for a in args])]


rng = np.random.default_rng(20260827)
ones = np.ones(NDROP, np.float32)
# a real draw at RSB-A2's ramp: bernoulli(keep_i)/keep_i, i.e. 0 or 1/keep_i
keeps = 1.0 - 0.05 * np.arange(NDROP) / 15.0
# ⚠⚠ **THE TEST MASK IS NOT A2's RAMP DRAW, and that is deliberate.** At keeps 0.95..1.0 a random
# draw over 16 sites usually drops NOTHING, and a mask of all ones is exactly the case in which the
# two placements are IDENTICAL — the run would have printed 0.00e+00 and read as a pass. The
# question here is where the site SITS, which is a property of the emitter and not of the rate, so
# the mask is forced: every third site dropped, the rest scaled by 1/keep as the driver draws them.
draw = np.where(np.arange(NDROP) % 3 == 0, 0.0, 1.0 / keeps).astype(np.float32)

print("── R50 stochastic depth: is the site on the BRANCH or on the block OUTPUT? ──")
print(f"  q=1 (32x32), B={B}, {NDROP} sites, keeps {keeps[0]:.4f}..{keeps[-1]:.4f}")
print(f"  test mask: {int((draw == 0).sum())} of {NDROP} sites dropped (FORCED — see the note)")

correct = os.path.join(D, "r50_tiny_sd.mlir")
misp    = os.path.join(D, "r50_tiny_misplaced.mlir")


def worst(a, b):
    r = 0.0
    for x, y in zip(a, b):
        assert np.isfinite(x).all() and np.isfinite(y).all(), "non-finite output — no answer"
        r = max(r, float(np.max(np.abs(x - y))) / (float(np.max(np.abs(y))) or 1.0))
    return r


c1, m1 = run(correct, ones, rng), run(misp, ones, rng)
c2, m2 = run(correct, draw, rng), run(misp, draw, rng)
cn     = run(os.path.join(D, "r50_tiny_nosd.mlir"), ones, rng)

L = 3 * 161                                    # theta' | m' | v' | loss,bc1,bc2 | 106 BN stats
fwd = lambda o: [o[L]] + o[L + 3:]             # the FORWARD's outputs: the loss and the BN stats
grad = lambda o: o[161:L]                      # m' and v', which are LINEAR in the gradient

print(f"\n  ⭐ TEST     at a REAL draw, correct vs misplaced      : {worst(c2, m2):.2e}")
print( "             the only check that sees the placement at all")
print(f"  CONTROL  at an ALL-ONES mask, correct vs misplaced   : {worst(c1, m1):.2e}")
print( "             must be ZERO — same ops, same fusion, `1 * (b+x) == 1*b + x`")
print(f"  CONTROL  ones mask, sd vs NO-sd render, FORWARD      : {worst(fwd(c1), fwd(cn)):.2e}")
print( "             `dropPath_ones_id`: loss + all 106 BN stats, bit-exact")
print(f"  CONTROL  ones mask, sd vs NO-sd render, m' and v'    : {worst(grad(c1), grad(cn)):.2e}")
print( "             ⚠ NOT bit-exact and correctly so: the extra `multiply` changes XLA's")
print( "             fusion, so the 161-parameter reduction chain reassociates. The gradients")
print( "             are what agree, at ~1e-6 relative; theta' amplifies that through AdamW's")
print( "             m/(sqrt(v)+eps) wherever v is small, exactly as `opt_step_tie.py`'s own")
print( "             rtol note describes. A bit-exactness claim on theta' would be false.")

ok = (worst(c2, m2) > 1e-3 and worst(c1, m1) == 0.0
      and worst(fwd(c1), fwd(cn)) == 0.0 and worst(grad(c1), grad(cn)) < 1e-5)
print(f"\n{'✓' if ok else '✗'} the site is on the RESIDUAL BRANCH, and only a non-ones mask says so")
sys.exit(0 if ok else 1)
