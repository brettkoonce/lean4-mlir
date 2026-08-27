"""CONTROL for the new EMA row: does it go RED on the two defects it claims to catch?

  (a) the shadow reads the INCOMING theta instead of the updated theta'
  (b) the region order is swapped -- E before G
"""
import sys, os, numpy as np
sys.path.insert(0, "scripts")
import opt_step_tie as T

rng = np.random.default_rng(20260814)
f32 = lambda *s: rng.standard_normal(s).astype(np.float32)
LR, TT = 0.005, 3
bc1 = np.float32(1.0 - 0.9 ** TT); bc2 = np.float32(1.0 - 0.999 ** TT)
theta = [f32(*s) for s in T.SHAPES]
m0 = [f32(*s) * 0.1 for s in T.SHAPES]
v0 = [np.abs(f32(*s)) * 0.01 for s in T.SHAPES]
G0 = [f32(*s) * 5.0 for s in T.SHAPES]
dg = [f32(*s) * 5.0 for s in T.SHAPES]
e0 = [f32(*s) * 0.5 for s in T.SHAPES]

ref = "generated_resnet50_imagenet_a2accum.py"
_, d = T.run_ref_ema(ref, e0, theta, TT - 1)
gsum = [a + b for a, b in zip(G0, dg)]
arrays = ([*theta, *m0, *v0, *G0, *e0, np.float32(LR), bc1, bc2,
           np.float32(1.0), np.float32(1.0), np.float32(d), np.float32(1.0 - d), *dg])
outs = T.run_render("emalambacc8wxclip", arrays, 15)
rt, rm, rv = T.run_ref(ref, 8, theta, m0, v0, gsum, LR, TT)

correct, _ = T.run_ref_ema(ref, e0, rt, TT - 1)      # shadow of theta'  -- what ships
lagged,  _ = T.run_ref_ema(ref, e0, theta, TT - 1)   # (a) shadow of theta -- the defect

def rel(a, b):
    return float(np.max(np.abs(a - b))) / (float(np.max(np.abs(b))) or 1.0)

print(f"  rtol gate                                 {T.RTOL:.0e}")
print(f"  TEST     E' vs ema_update(e0, theta')     {max(rel(outs[12+i], correct[i]) for i in range(3)):.2e}")
print(f"  CONTROL a  E' vs ema_update(e0, theta)    {max(rel(outs[12+i], lagged[i]) for i in range(3)):.2e}   <- one-step lag")
print(f"  CONTROL b  E' read at offset 9 (G's slot) {max(rel(outs[9+i], correct[i]) for i in range(3)):.2e}   <- swapped region order")
print(f"  CONTROL c  G' read at offset 12 (E's slot){max(rel(outs[12+i], gsum[i]) for i in range(3)):.2e}   <- swapped region order")
