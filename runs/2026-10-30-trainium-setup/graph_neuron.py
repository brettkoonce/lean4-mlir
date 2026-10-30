"""Whole verified_mlir graph: XLA:CPU on the original text vs Neuron on the sas-rewritten text."""
import sys, numpy as np, jax
sys.path.insert(0, 'scripts/platform'); import tier2
from sas_rewrite import rewrite
from jax._src import xla_bridge
from jax._src.lib import xla_client as xc
from jax._src.interpreters import mlir as jmlir
import jaxlib.mlir.ir as ir
name = sys.argv[1]
fn, ins, outs = tier2.signature(name)
inputs = tier2.make_inputs(name, ins)
orig = (tier2.ROOT / 'verified_mlir' / f'{name}.mlir').read_text()
new, k = rewrite(orig)
print(f'{name}: {k} select_and_scatter rewritten', flush=True)
res = {}
for plat, txt in (('cpu', orig), ('neuron', new)):
    txt = txt.replace(f'func.func @{fn}(', 'func.func @main(', 1)
    be = xla_bridge.get_backend(plat)
    devs = xc.DeviceList(tuple(be.local_devices()[:1]))
    with jmlir.make_ir_context() as ctx, ir.Location.unknown(ctx):
        exe = be.compile_and_load(ir.Module.parse(txt), executable_devices=devs, compile_options=xc.CompileOptions())
    res[plat] = [np.asarray(o) for o in exe.execute([jax.device_put(a, devs[0]) for a in inputs])]
pair = tier2.pairing(name, ins, outs)
errs = []
for i, (c, n, p) in enumerate(zip(res['cpu'], res['neuron'], pair)):
    if p is not None: c, n = c - inputs[p], n - inputs[p]
    errs.append(float(np.linalg.norm(n - c) / max(np.linalg.norm(c), 1e-30)))
print('  rel err per output:', ' '.join(f'{e:.0e}' for e in errs))
print(f'  max {max(errs):.1e}, median {np.median(errs):.1e}')
