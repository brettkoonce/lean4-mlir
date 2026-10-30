"""slice.py NAME TARGET [FRONTIER...] — the target's backward slice of the (sas-rewritten) graph, with
the frontier values (and the function's own args) as arguments; run it on CPU and Neuron with seeded
random arguments; write the module to slice_TARGET.mlir."""
import re, sys, numpy as np, jax
sys.path.insert(0, 'scripts/platform'); import tier2
from sas_rewrite import rewrite
from jax._src import xla_bridge
from jax._src.lib import xla_client as xc
from jax._src.interpreters import mlir as jmlir
import jaxlib.mlir.ir as ir
name, target, frontier = sys.argv[1], sys.argv[2], set(sys.argv[3:])
fn, ins, outs = tier2.signature(name)
txt, _ = rewrite((tier2.ROOT / 'verified_mlir' / f'{name}.mlir').read_text())
lines = txt.split('\n')
body0 = next(i for i, l in enumerate(lines) if 'func.func @' in l) + 1
ret = next(i for i, l in enumerate(lines) if l.startswith('    return '))
stm = []; cur = None
for l in lines[body0:ret]:
    m = re.match(r'^    (%[\w.]+) = ', l)
    if m: cur = [m.group(1), [l]]; stm.append(cur)
    elif cur and not l.strip().startswith('//'): cur[1].append(l)
defs = {s[0]: s for s in stm}
sig = re.search(r'func\.func @\w+\((.*?)\)\s*->', txt, re.S).group(1)
argty = dict(re.findall(r'(%\w+)\s*:\s*(tensor<[^>]*>)', sig))
def rty(s): return re.findall(r'tensor<[^>]*>', ' '.join(s[1]))[-1]
need = set(); stack = [target]; args = []
while stack:
    v = stack.pop()
    if v in need: continue
    need.add(v)
    if v in frontier or v not in defs:
        args.append(v); continue
    body = ' '.join(defs[v][1]).split('=', 1)[1]
    body = re.sub(r'\^bb0\(.*?\)', '', body)
    for u in re.findall(r'%[\w.]+', body):
        if u in defs or u in argty: stack.append(u)
args = sorted(set(args), key=lambda a: (a not in argty, a))
aty = {a: argty.get(a) or rty(defs[a]) for a in args}
keep = [s for s in stm if s[0] in need and s[0] not in frontier]
T = rty(defs[target])
mod = ('module @m {\n  func.func @main(' + ', '.join(f'{a}: {aty[a]}' for a in args) + f') -> {T} {{\n'
       + '\n'.join('\n'.join(s[1]) for s in keep) + f'\n    return {target} : {T}\n  }}\n}}\n')
open(f'slice_{target[1:]}.mlir', 'w').write(mod)
shp = lambda t: tuple(int(d) for d in re.match(r'tensor<((?:\d+x)*)', t).group(1).split('x') if d)
real = dict(zip([n for n, _ in ins], tier2.make_inputs(name, ins)))
fr = [a for a in args if a not in argty]
if fr:   # the frontier's real values: the full graph on XLA:CPU, returning them
    L = list(lines)
    L[ret] = '    return ' + ', '.join(fr) + ' : ' + ', '.join(aty[a] for a in fr)
    si = body0 - 1
    L[si] = re.sub(r'\)\s*->\s*\(.*\)\s*\{\s*$', ') -> (' + ', '.join(aty[a] for a in fr) + ') {', L[si])
    t = '\n'.join(L).replace(f'func.func @{fn}(', 'func.func @main(', 1)
    be = xla_bridge.get_backend('cpu'); devs = xc.DeviceList(tuple(be.local_devices()[:1]))
    with jmlir.make_ir_context() as ctx, ir.Location.unknown(ctx):
        exe = be.compile_and_load(ir.Module.parse(t), executable_devices=devs, compile_options=xc.CompileOptions())
    for a, o in zip(fr, exe.execute([jax.device_put(real[n], devs[0]) for n, _ in ins])): real[a[1:]] = np.asarray(o)
vals = [real[a[1:]] for a in args]
np.savez(f'slice_{target[1:]}.npz', **{a[1:]: v for a, v in zip(args, vals)})
r = []
for plat in ('cpu', 'neuron'):
    be = xla_bridge.get_backend(plat); devs = xc.DeviceList(tuple(be.local_devices()[:1]))
    with jmlir.make_ir_context() as ctx, ir.Location.unknown(ctx):
        exe = be.compile_and_load(ir.Module.parse(mod), executable_devices=devs, compile_options=xc.CompileOptions())
    r.append(np.asarray(exe.execute([jax.device_put(a, devs[0]) for a in vals])[0], np.float64))
e = np.linalg.norm(r[1] - r[0]) / np.linalg.norm(r[0])
print(f'{target} frontier={sorted(frontier)}: {len(keep)} ops, {len(args)} args, rel err {e:.1e}', flush=True)
