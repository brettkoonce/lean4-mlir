"""Bisect cifar8's tier-2 failure on Neuron: each op family on the neuron device vs JAX's CPU,
then the whole verified_mlir graph through JAX's own neuron client (no shim)."""
import sys, numpy as np, jax, jax.numpy as jnp
from jax import lax

cpu = jax.devices('cpu')[0]
neu = jax.devices('neuron')[0]
rng = np.random.default_rng(0)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def both(name, f, *xs):
    xs = [np.asarray(x, np.float32) for x in xs]
    outs = []
    for d in (cpu, neu):
        r = jax.jit(f, device=d)(*[jax.device_put(x, d) for x in xs])
        outs.append(jax.tree.leaves(r))
    errs = [rel(n, c) for c, n in zip(*outs)]
    print(f'{"OK " if max(errs) < 1e-3 else "BAD"} {name:28s} rel err ' + ' '.join(f'{e:.1e}' for e in errs), flush=True)


B = 128
x = rng.standard_normal((B, 16, 32, 32))
w = rng.standard_normal((16, 16, 3, 3)) * 0.1
dn = ('NCHW', 'OIHW', 'NCHW')
conv = lambda x, w: lax.conv_general_dilated(x, w, (1, 1), [(1, 1), (1, 1)], dimension_numbers=dn)
pool = lambda x: lax.reduce_window(x, -jnp.inf, lax.max, (1, 1, 2, 2), (1, 1, 2, 2), 'VALID')

both('conv fwd', conv, x, w)
both('conv grad (dx, dw)', jax.grad(lambda x, w: (conv(x, w) ** 2).sum(), argnums=(0, 1)), x, w)
both('maxpool fwd', pool, x)
both('maxpool grad (sel&scat)', jax.grad(lambda x: (pool(x) * jnp.arange(16 * 16 * 16.).reshape(16, 16, 16)).sum()), x)
both('relu grad', jax.grad(lambda x: (jnp.maximum(x, 0) * x).sum()), x)
both('conv+pool grad', jax.grad(lambda x, w: (pool(jnp.maximum(conv(x, w), 0)) ** 2).sum(), argnums=(0, 1)), x, w)
both('reduce sum axis (0,2,3)', lambda x: x.sum((0, 2, 3)), x)
both('transpose 0231', lambda x: jnp.transpose(x, (0, 2, 3, 1)), x)
both('reverse 23', lambda w: lax.rev(w, (2, 3)), w)

# The whole graph through JAX's neuron client, as tier2.py's run_cpu does on the CPU.
if len(sys.argv) > 1:
    sys.path.insert(0, 'scripts/platform')
    import tier2
    from pathlib import Path
    from jax._src import xla_bridge
    from jax._src.lib import xla_client as xc
    from jax._src.interpreters import mlir as jmlir
    import jaxlib.mlir.ir as ir
    name = sys.argv[1]
    fn, ins, outs = tier2.signature(name)
    inputs = tier2.make_inputs(name, ins)
    res = {}
    for plat in ('cpu', 'neuron'):
        txt = (tier2.ROOT / 'verified_mlir' / f'{name}.mlir').read_text().replace(f'func.func @{fn}(', 'func.func @main(', 1)
        be = xla_bridge.get_backend(plat)
        devs = xc.DeviceList(tuple(be.local_devices()[:1]))
        with jmlir.make_ir_context() as ctx, ir.Location.unknown(ctx):
            exe = be.compile_and_load(ir.Module.parse(txt), executable_devices=devs, compile_options=xc.CompileOptions())
        res[plat] = [np.asarray(o) for o in exe.execute([jax.device_put(a, devs[0]) for a in inputs])]
    pair = tier2.pairing(name, ins, outs)
    for i, (c, n, p) in enumerate(zip(res['cpu'], res['neuron'], pair)):
        if p is not None: c, n = c - inputs[p], n - inputs[p]
        print(f'  out {i:2d} {str(c.shape):22s} rel err {rel(n, c):.1e}', flush=True)
