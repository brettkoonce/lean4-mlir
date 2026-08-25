#!/usr/bin/env python3
"""Does keeping a value bf16 ACROSS an op beat converting to f32 and back?

    .venv/bin/python scripts/bf16_boundary_probe.py

⛔⛔ **THIS SCRIPT CANCELLED A SCOPED PROJECT.** `planning/bf16_dtype_ir.md` proposed threading a
dtype through the emit stack so activations could stay bf16 between ops, and wrote its own
refutation test into its §8 step 3: hand-build the bf16-through block first, and if it does not beat
the bf16-with-boundary block at the nets' real shapes, do not build the project. This is that test.
It does not beat it — see `bf16_renderer.md` §20.3.

⭐⭐ What it DID find is the thing that mattered: a `dot_general` with bf16 operands and an
**f32-typed result** makes the gemm write twice the bytes. `bf16_renderer.md` §9.2 had measured that
result type for CORRECTNESS (bf16 reaches the tensor cores either way) and never for SPEED. Giving
ViT's dots bf16-typed results took the net from 1.23x to 1.46x — one line per emit, no dtype in the
IR at all (§20.1).

Arms, per chain:
  f32       control
  boundary  bf16 operands, f32-typed result — `dotInBf16`'s shape, and what ViT shipped at §19
  bf16out   bf16 operands, bf16-TYPED result, convert back — what ViT ships now, and what the
            convolution ops have always done (forced on them by §9.2's correctness finding)
  elemonly  f32 result, but the elementwise between the two dots runs bf16
  through   bf16 result AND bf16 elementwise/pass-through — the cancelled project's proposal

⚠ Read the WALL CLOCK, not the convert count. `bf16_dtype_ir.md` §5.1 made a falling convert count
the gate; ViT's went 864 -> 1224 and the step got 1.19x faster (§20.2).
"""

import importlib.util, sys, time
import numpy as np

spec = importlib.util.spec_from_file_location("g2", "scripts/bf16_gate2.py")
g2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g2)
ctxs = g2._lazy(); jax, jex = ctxs[0], ctxs[1]

B, TOK, D, M, HD = 32, 197, 192, 768, 64
DEPTH = 12

def t(dims, dt="f32"):
    return f"tensor<{'x'.join(str(d) for d in dims)}x{dt}>"

class Emit:
    def __init__(self): self.n = 0; self.lines = []
    def fresh(self): self.n += 1; return f"%v{self.n}"
    def add(self, s): self.lines.append("    " + s)

def gelu(e, x, dims, dt):
    """The repo's tanh-approximation GELU, verbatim in op sequence, at dtype `dt`."""
    T = t(dims, dt)
    x2, x3, ck, kx3, inn, cs, u, th, one, opt, ch, hx, o = [e.fresh() for _ in range(13)]
    e.add(f"{x2} = stablehlo.multiply {x}, {x} : {T}")
    e.add(f"{x3} = stablehlo.multiply {x2}, {x} : {T}")
    e.add(f"{ck} = stablehlo.constant dense<0.044715> : {T}")
    e.add(f"{kx3} = stablehlo.multiply {ck}, {x3} : {T}")
    e.add(f"{inn} = stablehlo.add {x}, {kx3} : {T}")
    e.add(f"{cs} = stablehlo.constant dense<0.797884583> : {T}")
    e.add(f"{u} = stablehlo.multiply {cs}, {inn} : {T}")
    e.add(f"{th} = stablehlo.tanh {u} : {T}")
    e.add(f"{one} = stablehlo.constant dense<1.0> : {T}")
    e.add(f"{opt} = stablehlo.add {one}, {th} : {T}")
    e.add(f"{ch} = stablehlo.constant dense<0.5> : {T}")
    e.add(f"{hx} = stablehlo.multiply {ch}, {x} : {T}")
    e.add(f"{o} = stablehlo.multiply {hx}, {opt} : {T}")
    return o

def cvt(e, x, dims, frm, to):
    o = e.fresh()
    e.add(f"{o} = stablehlo.convert {x} : ({t(dims, frm)}) -> {t(dims, to)}")
    return o

def dot2(e, x, w, xd, wd, od, dt_in, dt_out):
    """[B,tok,a] x [a,c] -> [B,tok,c], contracting [2]x[0]."""
    o = e.fresh()
    e.add(f"{o} = stablehlo.dot_general {x}, {w}, contracting_dims = [2] x [0], "
          f"precision = [DEFAULT, DEFAULT] : ({t(xd, dt_in)}, {t(wd, dt_in)}) -> {t(od, dt_out)}")
    return o

def bias(e, x, b, dims, dt):
    bb, o = e.fresh(), e.fresh()
    e.add(f"{bb} = stablehlo.broadcast_in_dim {b}, dims = [2] : ({t([dims[2]], dt)}) -> {t(dims, dt)}")
    e.add(f"{o} = stablehlo.add {x}, {bb} : {t(dims, dt)}")
    return o

# ── chain M: fc1 -> bias -> GELU -> fc2, DEPTH deep, [B,TOK,D] -> [B,TOK,D] ────────────
def chain_M(mode):
    e = Emit()
    args = [f"%x: {t([B,TOK,D])}"]
    for i in range(DEPTH):
        args += [f"%W1_{i}: {t([D,M])}", f"%b1_{i}: {t([M])}", f"%W2_{i}: {t([M,D])}"]
    cur = "%x"
    for i in range(DEPTH):
        if mode == "f32":
            h = dot2(e, cur, f"%W1_{i}", [B,TOK,D], [D,M], [B,TOK,M], "f32", "f32")
            h = bias(e, h, f"%b1_{i}", [B,TOK,M], "f32")
            h = gelu(e, h, [B,TOK,M], "f32")
            cur = dot2(e, h, f"%W2_{i}", [B,TOK,M], [M,D], [B,TOK,D], "f32", "f32")
        elif mode == "boundary":
            xb = cvt(e, cur, [B,TOK,D], "f32", "bf16")
            wb = cvt(e, f"%W1_{i}", [D,M], "f32", "bf16")
            h = dot2(e, xb, wb, [B,TOK,D], [D,M], [B,TOK,M], "bf16", "f32")
            h = bias(e, h, f"%b1_{i}", [B,TOK,M], "f32")
            h = gelu(e, h, [B,TOK,M], "f32")
            hb = cvt(e, h, [B,TOK,M], "f32", "bf16")
            w2b = cvt(e, f"%W2_{i}", [M,D], "f32", "bf16")
            cur = dot2(e, hb, w2b, [B,TOK,M], [M,D], [B,TOK,D], "bf16", "f32")
        elif mode == "bf16out":
            # dot writes bf16, then ONE convert up for the f32 elementwise, then bf16 for the next
            # dot. Isolates "the dot's result type" from "the elementwise dtype".
            xb = cvt(e, cur, [B,TOK,D], "f32", "bf16")
            wb = cvt(e, f"%W1_{i}", [D,M], "f32", "bf16")
            h = dot2(e, xb, wb, [B,TOK,D], [D,M], [B,TOK,M], "bf16", "bf16")
            h = cvt(e, h, [B,TOK,M], "bf16", "f32")
            h = bias(e, h, f"%b1_{i}", [B,TOK,M], "f32")
            h = gelu(e, h, [B,TOK,M], "f32")
            hb = cvt(e, h, [B,TOK,M], "f32", "bf16")
            w2b = cvt(e, f"%W2_{i}", [M,D], "f32", "bf16")
            cur = dot2(e, hb, w2b, [B,TOK,M], [M,D], [B,TOK,D], "bf16", "f32")
        elif mode == "elemonly":
            # dot keeps its f32 result (today's shape) but the ELEMENTWISE runs bf16. The mirror
            # image of `bf16out`, so the two together attribute `through`'s win.
            xb = cvt(e, cur, [B,TOK,D], "f32", "bf16")
            wb = cvt(e, f"%W1_{i}", [D,M], "f32", "bf16")
            h = dot2(e, xb, wb, [B,TOK,D], [D,M], [B,TOK,M], "bf16", "f32")
            hb = cvt(e, h, [B,TOK,M], "f32", "bf16")
            bb = cvt(e, f"%b1_{i}", [M], "f32", "bf16")
            hb = bias(e, hb, bb, [B,TOK,M], "bf16")
            hb = gelu(e, hb, [B,TOK,M], "bf16")
            w2b = cvt(e, f"%W2_{i}", [M,D], "f32", "bf16")
            cur = dot2(e, hb, w2b, [B,TOK,M], [M,D], [B,TOK,D], "bf16", "f32")
        else:  # through
            xb = cvt(e, cur, [B,TOK,D], "f32", "bf16")
            wb = cvt(e, f"%W1_{i}", [D,M], "f32", "bf16")
            h = dot2(e, xb, wb, [B,TOK,D], [D,M], [B,TOK,M], "bf16", "bf16")   # bf16 RESULT
            bb = cvt(e, f"%b1_{i}", [M], "f32", "bf16")
            h = bias(e, h, bb, [B,TOK,M], "bf16")
            h = gelu(e, h, [B,TOK,M], "bf16")                                   # GELU in bf16
            w2b = cvt(e, f"%W2_{i}", [M,D], "f32", "bf16")
            cur = dot2(e, h, w2b, [B,TOK,M], [M,D], [B,TOK,D], "bf16", "f32")
    body = "\n".join(e.lines)
    return (f"module @m {{\n  func.func public @main({', '.join(args)}) -> {t([B,TOK,D])} {{\n"
            f"{body}\n    return {cur} : {t([B,TOK,D])}\n  }}\n}}\n")

# ── chain P: dot -> slice -> transpose -> batched dot. Pure PASS-THROUGH in between. ───
def chain_P(mode):
    e = Emit(); NH = 36   # 12 blocks x 3 heads
    args = [f"%x: {t([B,TOK,D])}"] + [f"%W_{i}: {t([D,D])}" for i in range(NH)]
    outs = []
    for i in range(NH):
        dt = "bf16" if mode == "through" else "f32"
        if mode == "f32":
            a = dot2(e, "%x", f"%W_{i}", [B,TOK,D], [D,D], [B,TOK,D], "f32", "f32")
            sl, tr, o = e.fresh(), e.fresh(), e.fresh()
            e.add(f"{sl} = stablehlo.slice {a} [0:{B}, 0:{TOK}, 0:{HD}] : ({t([B,TOK,D])}) -> {t([B,TOK,HD])}")
            e.add(f"{tr} = stablehlo.transpose {sl}, dims = [0, 2, 1] : ({t([B,TOK,HD])}) -> {t([B,HD,TOK])}")
            e.add(f"{o} = stablehlo.dot_general {sl}, {tr}, batching_dims = [0] x [0], contracting_dims = [2] x [1], "
                  f"precision = [DEFAULT, DEFAULT] : ({t([B,TOK,HD])}, {t([B,HD,TOK])}) -> {t([B,TOK,TOK])}")
        elif mode == "boundary":
            xb = cvt(e, "%x", [B,TOK,D], "f32", "bf16"); wb = cvt(e, f"%W_{i}", [D,D], "f32", "bf16")
            a = dot2(e, xb, wb, [B,TOK,D], [D,D], [B,TOK,D], "bf16", "f32")
            sl, tr, sb, tb, o = [e.fresh() for _ in range(5)]
            e.add(f"{sl} = stablehlo.slice {a} [0:{B}, 0:{TOK}, 0:{HD}] : ({t([B,TOK,D])}) -> {t([B,TOK,HD])}")
            e.add(f"{tr} = stablehlo.transpose {sl}, dims = [0, 2, 1] : ({t([B,TOK,HD])}) -> {t([B,HD,TOK])}")
            e.add(f"{sb} = stablehlo.convert {sl} : ({t([B,TOK,HD])}) -> {t([B,TOK,HD],'bf16')}")
            e.add(f"{tb} = stablehlo.convert {tr} : ({t([B,HD,TOK])}) -> {t([B,HD,TOK],'bf16')}")
            e.add(f"{o} = stablehlo.dot_general {sb}, {tb}, batching_dims = [0] x [0], contracting_dims = [2] x [1], "
                  f"precision = [DEFAULT, DEFAULT] : ({t([B,TOK,HD],'bf16')}, {t([B,HD,TOK],'bf16')}) -> {t([B,TOK,TOK])}")
        elif mode == "bf16out":
            # WHAT SHIPS AFTER STAGE 2: the dot writes bf16, then converts back to f32, and the
            # slice/transpose run in f32 and re-cast for the next dot. The gap between THIS and
            # `through` is precisely what a dtype-carrying emit stack would still buy on ViT.
            xb = cvt(e, "%x", [B,TOK,D], "f32", "bf16"); wb = cvt(e, f"%W_{i}", [D,D], "f32", "bf16")
            ab = dot2(e, xb, wb, [B,TOK,D], [D,D], [B,TOK,D], "bf16", "bf16")
            a = cvt(e, ab, [B,TOK,D], "bf16", "f32")
            sl, tr, sb, tb, o = [e.fresh() for _ in range(5)]
            e.add(f"{sl} = stablehlo.slice {a} [0:{B}, 0:{TOK}, 0:{HD}] : ({t([B,TOK,D])}) -> {t([B,TOK,HD])}")
            e.add(f"{tr} = stablehlo.transpose {sl}, dims = [0, 2, 1] : ({t([B,TOK,HD])}) -> {t([B,HD,TOK])}")
            e.add(f"{sb} = stablehlo.convert {sl} : ({t([B,TOK,HD])}) -> {t([B,TOK,HD],'bf16')}")
            e.add(f"{tb} = stablehlo.convert {tr} : ({t([B,HD,TOK])}) -> {t([B,HD,TOK],'bf16')}")
            e.add(f"{o} = stablehlo.dot_general {sb}, {tb}, batching_dims = [0] x [0], contracting_dims = [2] x [1], "
                  f"precision = [DEFAULT, DEFAULT] : ({t([B,TOK,HD],'bf16')}, {t([B,HD,TOK],'bf16')}) -> {t([B,TOK,TOK],'bf16')}")
        else:  # through — the slice and transpose carry bf16, no round trip at all
            xb = cvt(e, "%x", [B,TOK,D], "f32", "bf16"); wb = cvt(e, f"%W_{i}", [D,D], "f32", "bf16")
            a = dot2(e, xb, wb, [B,TOK,D], [D,D], [B,TOK,D], "bf16", "bf16")
            sl, tr, o = e.fresh(), e.fresh(), e.fresh()
            e.add(f"{sl} = stablehlo.slice {a} [0:{B}, 0:{TOK}, 0:{HD}] : ({t([B,TOK,D],'bf16')}) -> {t([B,TOK,HD],'bf16')}")
            e.add(f"{tr} = stablehlo.transpose {sl}, dims = [0, 2, 1] : ({t([B,TOK,HD],'bf16')}) -> {t([B,HD,TOK],'bf16')}")
            e.add(f"{o} = stablehlo.dot_general {sl}, {tr}, batching_dims = [0] x [0], contracting_dims = [2] x [1], "
                  f"precision = [DEFAULT, DEFAULT] : ({t([B,TOK,HD],'bf16')}, {t([B,HD,TOK],'bf16')}) -> {t([B,TOK,TOK],'bf16')}")
        if mode in ("bf16out", "through"):
            o = cvt(e, o, [B,TOK,TOK], "bf16", "f32")
        outs.append(o)
    res = ", ".join(outs); rty = ", ".join([t([B,TOK,TOK])] * NH)
    body = "\n".join(e.lines)
    return (f"module @m {{\n  func.func public @main({', '.join(args)}) -> ({rty}) {{\n"
            f"{body}\n    return {res} : {rty}\n  }}\n}}\n")

def run(name, src, reps=20):
    import re, tempfile, os
    fd, path = tempfile.mkstemp(suffix=".mlir"); os.write(fd, src.encode()); os.close(fd)
    exe = g2.compile_mlir(path, ctxs)
    dev = jex.backend.get_backend().local_devices()[0]
    sig = re.search(r"@main\((.*?)\)\s*->", src, re.S).group(1)
    rng = np.random.default_rng(7)
    args = [jax.device_put((rng.standard_normal(int(np.prod([int(d) for d in m.group(1).split("x") if d])))
                            * 0.05).reshape([int(d) for d in m.group(1).split("x") if d]).astype(np.float32), dev)
            for m in re.finditer(r"tensor<([0-9x]*)f32>", sig)]
    def once():
        exe.execute_sharded(args).disassemble_into_single_device_arrays()[0][0].block_until_ready()
    once()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); once(); ts.append(time.perf_counter() - t0)
    hlo = exe.hlo_modules()[0].to_string()
    ncvt = len(re.findall(r"=\s*\S+\s+convert\(", hlo))
    os.unlink(path)
    return float(np.median(ts)) * 1e3, ncvt

for label, fn, modes in [
        ("M: dot -> bias -> GELU -> dot, x12 (the §5 increment)", chain_M,
         ("f32", "boundary", "bf16out", "elemonly", "through")),
        ("P: dot -> slice -> transpose -> dot, x36 (pure pass-through)", chain_P,
         ("f32", "boundary", "bf16out", "through"))]:
    print(f"\n===== chain {label} =====")
    base = None
    for mode in modes:
        ms, ncvt = run(f"{label}/{mode}", fn(mode))
        if mode == "f32": base = ms
        print(f"  {mode:9s} {ms:8.2f} ms   speedup {base/ms:5.2f}x   surviving converts in optimized HLO: {ncvt}")


# ── ConvNeXt-T block interior — bf16_dtype_ir.md §8 step 3 VERBATIM ──────────────────────
import importlib.util, os, re, tempfile, time
import numpy as np
spec = importlib.util.spec_from_file_location("g2", "scripts/bf16_gate2.py")
g2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g2)
ctxs = g2._lazy(); jax, jex = ctxs[0], ctxs[1]

B, C, E, H = 32, 96, 384, 56
DEPTH = 3   # ConvNeXt-T stage 1 has 3 blocks at these shapes

def t(d, dt="f32"): return f"tensor<{'x'.join(map(str,d))}x{dt}>"

class E_:
    def __init__(self): self.n=0; self.l=[]
    def f(self): self.n+=1; return f"%v{self.n}"
    def a(self,s): self.l.append("    "+s)

def conv1x1(e, x, w, xd, wd, od, dti, dto):
    o = e.f()
    e.a(f"{o} = stablehlo.convolution({x}, {w})")
    e.a("  dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],")
    e.a("  window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}")
    e.a("  {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
        f" : ({t(xd,dti)}, {t(wd,dti)}) -> {t(od,dto)}")
    return o

def cvt(e,x,d,a,b):
    o=e.f(); e.a(f"{o} = stablehlo.convert {x} : ({t(d,a)}) -> {t(d,b)}"); return o

def gelu(e,x,d,dt):
    T=t(d,dt); n=[e.f() for _ in range(13)]
    x2,x3,ck,kx3,inn,cs,u,th,one,opt,ch,hx,o = n
    e.a(f"{x2} = stablehlo.multiply {x}, {x} : {T}");  e.a(f"{x3} = stablehlo.multiply {x2}, {x} : {T}")
    e.a(f"{ck} = stablehlo.constant dense<0.044715> : {T}"); e.a(f"{kx3} = stablehlo.multiply {ck}, {x3} : {T}")
    e.a(f"{inn} = stablehlo.add {x}, {kx3} : {T}"); e.a(f"{cs} = stablehlo.constant dense<0.797884583> : {T}")
    e.a(f"{u} = stablehlo.multiply {cs}, {inn} : {T}"); e.a(f"{th} = stablehlo.tanh {u} : {T}")
    e.a(f"{one} = stablehlo.constant dense<1.0> : {T}"); e.a(f"{opt} = stablehlo.add {one}, {th} : {T}")
    e.a(f"{ch} = stablehlo.constant dense<0.5> : {T}"); e.a(f"{hx} = stablehlo.multiply {ch}, {x} : {T}")
    e.a(f"{o} = stablehlo.multiply {hx}, {opt} : {T}"); return o

def chain(mode):
    e=E_(); args=[f"%x: {t([B,C,H,H])}"]
    for i in range(DEPTH): args += [f"%We_{i}: {t([E,C,1,1])}", f"%Wp_{i}: {t([C,E,1,1])}"]
    cur="%x"
    for i in range(DEPTH):
        if mode=="f32":
            h = conv1x1(e,cur,f"%We_{i}",[B,C,H,H],[E,C,1,1],[B,E,H,H],"f32","f32")
            h = gelu(e,h,[B,E,H,H],"f32")
            cur = conv1x1(e,h,f"%Wp_{i}",[B,E,H,H],[C,E,1,1],[B,C,H,H],"f32","f32")
        elif mode=="bf16out":
            xb=cvt(e,cur,[B,C,H,H],"f32","bf16"); wb=cvt(e,f"%We_{i}",[E,C,1,1],"f32","bf16")
            h = conv1x1(e,xb,wb,[B,C,H,H],[E,C,1,1],[B,E,H,H],"bf16","bf16")
            h = cvt(e,h,[B,E,H,H],"bf16","f32")
            h = gelu(e,h,[B,E,H,H],"f32")
            hb=cvt(e,h,[B,E,H,H],"f32","bf16"); wp=cvt(e,f"%Wp_{i}",[C,E,1,1],"f32","bf16")
            cb = conv1x1(e,hb,wp,[B,E,H,H],[C,E,1,1],[B,C,H,H],"bf16","bf16")
            cur = cvt(e,cb,[B,C,H,H],"bf16","f32")
        else:  # through — NO f32 in the middle, which is what §8 step 3 asks for
            xb=cvt(e,cur,[B,C,H,H],"f32","bf16"); wb=cvt(e,f"%We_{i}",[E,C,1,1],"f32","bf16")
            h = conv1x1(e,xb,wb,[B,C,H,H],[E,C,1,1],[B,E,H,H],"bf16","bf16")
            h = gelu(e,h,[B,E,H,H],"bf16")
            wp=cvt(e,f"%Wp_{i}",[C,E,1,1],"f32","bf16")
            cb = conv1x1(e,h,wp,[B,E,H,H],[C,E,1,1],[B,C,H,H],"bf16","bf16")
            cur = cvt(e,cb,[B,C,H,H],"bf16","f32")
    return (f"module @m {{\n  func.func public @main({', '.join(args)}) -> {t([B,C,H,H])} {{\n"
            + "\n".join(e.l) + f"\n    return {cur} : {t([B,C,H,H])}\n  }}\n}}\n")

def run(src, reps=20):
    fd,path=tempfile.mkstemp(suffix=".mlir"); os.write(fd,src.encode()); os.close(fd)
    exe=g2.compile_mlir(path,ctxs); dev=jex.backend.get_backend().local_devices()[0]
    sig=re.search(r"@main\((.*?)\)\s*->",src,re.S).group(1); rng=np.random.default_rng(7)
    args=[jax.device_put((rng.standard_normal(int(np.prod([int(d) for d in m.group(1).split("x") if d])))*0.05)
          .reshape([int(d) for d in m.group(1).split("x") if d]).astype(np.float32),dev)
          for m in re.finditer(r"tensor<([0-9x]*)f32>",sig)]
    def once(): exe.execute_sharded(args).disassemble_into_single_device_arrays()[0][0].block_until_ready()
    once(); ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); once(); ts.append(time.perf_counter()-t0)
    hlo=exe.hlo_modules()[0].to_string()
    os.unlink(path)
    return float(np.median(ts))*1e3, len(re.findall(r"=\s*\S+\s+convert\(",hlo))


print("\n===== ConvNeXt-T stage-1 block interior: conv1x1 -> GELU -> conv1x1, x3 =====")
print("      (c=96, 4c=384, 56x56, B=32 — bf16_dtype_ir.md §8 step 3 verbatim)")
base = None
for mode in ("f32", "bf16out", "through"):
    ms, nc = run(chain(mode))
    if mode == "f32": base = ms
    tag = ("  <- WHAT ConvNeXt SHIPS TODAY" if mode == "bf16out"
           else "  <- THE CANCELLED PROJECT'S PROPOSAL" if mode == "through" else "")
    print(f"  {mode:9s} {ms:8.2f} ms   speedup {base/ms:5.2f}x   converts {nc}{tag}")
