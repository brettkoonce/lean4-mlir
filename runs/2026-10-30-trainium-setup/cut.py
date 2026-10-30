"""Bisect a verified_mlir graph on Neuron: return one intermediate SSA value, compare with XLA:CPU,
binary-search the first value (program order) that diverges. sas-rewritten on both sides."""
import re, sys, numpy as np, jax
sys.path.insert(0, 'scripts/platform'); import tier2
from sas_rewrite import rewrite
from jax._src import xla_bridge
from jax._src.lib import xla_client as xc
from jax._src.interpreters import mlir as jmlir
import jaxlib.mlir.ir as ir
name = sys.argv[1]
fn, ins, outs = tier2.signature(name)
inputs = tier2.make_inputs(name, ins)
txt, _ = rewrite((tier2.ROOT / 'verified_mlir' / f'{name}.mlir').read_text())
lines = txt.split('\n')
# statements: (name, type, line index of its last line)
stmts = []; cur = None
for i, l in enumerate(lines):
    m = re.match(r'^    (%[\w.]+) = ', l)
    if m or re.match(r'^    return ', l):
        if cur: stmts.append(cur)
        cur = [m.group(1), None, i] if m else None
    if cur:
        ts = re.findall(r'tensor<[^>]*>', l)
        if ts: cur[1] = ts[-1]
        cur[2] = i
stmts = [s for s in stmts if s[1] and 'xi1' not in s[1] and s[1] != 'tensor<f32>' and not s[0].startswith('%nsas')]
ret = next(i for i, l in enumerate(lines) if l.startswith('    return '))
sig = next(i for i, l in enumerate(lines) if 'func.func @' in l)
exes = {}
def run(k):
    nm, ty, _ = stmts[k]
    L = list(lines)
    L[ret] = f'    return {nm} : {ty}'
    L[sig] = re.sub(r'\)\s*->\s*\(.*\)\s*\{\s*$', f') -> {ty} {{', L[sig])
    t = '\n'.join(L).replace(f'func.func @{fn}(', 'func.func @main(', 1)
    r = []
    for plat in ('cpu', 'neuron'):
        be = xla_bridge.get_backend(plat); devs = xc.DeviceList(tuple(be.local_devices()[:1]))
        with jmlir.make_ir_context() as ctx, ir.Location.unknown(ctx):
            exe = be.compile_and_load(ir.Module.parse(t), executable_devices=devs, compile_options=xc.CompileOptions())
        r.append(np.asarray(exe.execute([jax.device_put(a, devs[0]) for a in inputs])[0], np.float64))
    e = float(np.linalg.norm(r[1] - r[0]) / max(np.linalg.norm(r[0]), 1e-30))
    print(f'  [{k:3d}/{len(stmts)}] {nm:8s} {ty:28s} rel err {e:.1e}', flush=True)
    return e
if len(sys.argv) > 2 and sys.argv[2] == 'scan':
    for k in range(int(sys.argv[3]), int(sys.argv[4])):
        nm, ty, end = stmts[k]
        start = max(i for i in range(end + 1) if lines[i].startswith(f'    {nm} = '))
        print('     ' + ' '.join(l.strip() for l in lines[start:end + 1])[:400])
        run(k)
    sys.exit()
if len(sys.argv) > 2 and sys.argv[2] == 'names':
    for nm in sys.argv[3:]: run(next(k for k, st in enumerate(stmts) if st[0] == nm))
    sys.exit()
lo, hi = 0, len(stmts) - 1
if len(sys.argv) > 2: lo, hi = int(sys.argv[2]), int(sys.argv[3])
while lo < hi:
    mid = (lo + hi) // 2
    if run(mid) > 1e-4: hi = mid
    else: lo = mid + 1
print(f'first diverging: {stmts[lo][0]}, line {stmts[lo][2]+1}:')
s = stmts[lo]
start = max(i for i in range(s[2] + 1) if lines[i].startswith(f'    {s[0]} = '))
print('\n'.join(lines[start:s[2] + 1]))
