"""runmod.py ARGS.npz MOD.mlir... — each module on XLA:CPU and Neuron with the saved arguments (by name)."""
import re, sys, numpy as np, jax
from jax._src import xla_bridge
from jax._src.lib import xla_client as xc
from jax._src.interpreters import mlir as jmlir
import jaxlib.mlir.ir as ir
z = np.load(sys.argv[1])
for f in sys.argv[2:]:
    t = open(f).read()
    names = re.findall(r'%(\w+):\s*tensor', re.search(r'func\.func @main\((.*?)\)\s*->', t, re.S).group(1))
    vals = [z[n] for n in names]
    r = []
    try:
        for plat in ('cpu', 'neuron'):
            be = xla_bridge.get_backend(plat); devs = xc.DeviceList(tuple(be.local_devices()[:1]))
            with jmlir.make_ir_context() as ctx, ir.Location.unknown(ctx):
                exe = be.compile_and_load(ir.Module.parse(t), executable_devices=devs, compile_options=xc.CompileOptions())
            r.append(np.asarray(exe.execute([jax.device_put(a, devs[0]) for a in vals])[0], np.float64))
        e = np.linalg.norm(r[1] - r[0]) / np.linalg.norm(r[0])
        print(f'{"BAD" if e > 1e-4 else "OK "} {f:40s} rel err {e:.1e}', flush=True)
    except Exception as ex:
        print(f'ERR {f}: {str(ex).splitlines()[0][:150]}', flush=True)
