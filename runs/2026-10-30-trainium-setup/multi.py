"""Return several SSA values from one cut of the graph; compare each with XLA:CPU."""
import re, sys, numpy as np, jax
sys.argv = [sys.argv[0], sys.argv[1], 'none'] + sys.argv[2:]
src = open('cut.py').read().split("if len(sys.argv) > 2 and sys.argv[2] == 'scan':")[0]
exec(src)
want = sys.argv[3:]
ty = {s[0]: s[1] for s in stmts}
L = list(lines)
L[ret] = f'    return {", ".join(want)} : {", ".join(ty[w] for w in want)}'
L[sig] = re.sub(r'\)\s*->\s*\(.*\)\s*\{\s*$', f') -> ({", ".join(ty[w] for w in want)}) {{', L[sig])
t = '\n'.join(L).replace(f'func.func @{fn}(', 'func.func @main(', 1)
r = []
for plat in ('cpu', 'neuron'):
    be = xla_bridge.get_backend(plat); devs = xc.DeviceList(tuple(be.local_devices()[:1]))
    with jmlir.make_ir_context() as ctx, ir.Location.unknown(ctx):
        exe = be.compile_and_load(ir.Module.parse(t), executable_devices=devs, compile_options=xc.CompileOptions())
    r.append([np.asarray(o, np.float64) for o in exe.execute([jax.device_put(a, devs[0]) for a in inputs])])
for w, c, n in zip(want, *r):
    print(f'  {w:8s} {ty[w]:28s} rel err {float(np.linalg.norm(n - c) / max(np.linalg.norm(c), 1e-30)):.1e}', flush=True)
