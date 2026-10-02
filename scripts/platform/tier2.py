#!/usr/bin/env python3
"""Tier 2 of the platform suite: verified_mlir/ artifacts against XLA:CPU goldens.

One artifact per op family, f32 and bf16 (scripts/platform/tier2_artifacts.tsv). Each runs
once on seeded inputs; the outputs are compared with what XLA:CPU computed from the same
graph and the same inputs, under scripts/platform/tolerances.tsv. A failure here, with
tiers 0-1 green, points at the backend's codegen for that op family — the bf16 conv/dot
result-type traps are the kind of thing it exists to catch.

    tier2.py goldens [--check] [--only NAME]    XLA:CPU, the pinned .venv's JAX
    tier2.py run NAME --runner BIN --work DIR [--backend B]

`run` needs numpy only: the device side is tier2_run.c through the shim, so the box under
test needs no JAX.

WHAT IS STORED. Outputs run to tens of millions of floats, so a golden is a summary per
output: element count, L2 norm, max |x|, the non-finite count, and the values at 128
seeded positions. A layout or indexing fault moves nearly every sampled value; a scale
fault moves the norm; a dropped op usually moves both.

TRAIN STEPS COMPARE THE STEP, NOT THE PARAMETERS. An output that is `p − lr·g` is all `p`,
so comparing it raw would hide any error in `g`. Each output is paired, in order, with the
next input of the same shape (the signature is x, params…, moments…, scalars, labels and
the outputs follow it) and the comparison is on `out − in`. Both sides use the same pairing,
so a mispairing costs sensitivity, never correctness.
"""
import argparse, os, re, subprocess, sys, zlib
from math import prod
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'scripts' / 'platform'
GOLD = HERE / 'goldens'
NSAMPLE = 128
SEED = 20261002


def artifacts():
    rows = []
    for ln in (HERE / 'tier2_artifacts.tsv').read_text().splitlines():
        if ln.strip() and not ln.startswith('#'):
            name, family, dtype = ln.split('\t')[:3]
            rows.append({'name': name, 'family': family, 'dtype': dtype})
    return rows


def signature(name):
    t = (ROOT / 'verified_mlir' / f'{name}.mlir').read_text()
    m = re.search(r'func\.func @(\w+)\((.*?)\)\s*->\s*\(?(.*?)\)?\s*\{', t, re.S)
    if not m: sys.exit(f'{name}: no func signature')
    shp = lambda s: tuple(int(d) for d in s.split('x') if d)
    ins = [(n, shp(s)) for n, s, dt in re.findall(r'%(\w+)\s*:\s*tensor<((?:\d+x)*)([a-z]\w*)>', m.group(2))]
    for n, s, dt in re.findall(r'%(\w+)\s*:\s*tensor<((?:\d+x)*)([a-z]\w*)>', m.group(2)):
        if dt != 'f32': sys.exit(f'{name}: input %{n} is {dt}; the shim invokes f32 only')
    outs = [shp(s) for s, dt in re.findall(r'tensor<((?:\d+x)*)([a-z]\w*)>', m.group(3))]
    return m.group(1), ins, outs


def make_inputs(name, ins):
    """Seeded, and shaped so a train step stays finite and well-conditioned: real one-hot rows,
    keep-scale 1 on drop-path inputs, He-scaled weights, γ near 1.

    ⚠ The optimizer state is m = 0, v = 1, no bias correction — NOT random. With a small random
    v, Adam's update m̂/√v̂ is close to sign(g) and v' − v cancels, so a gradient that is noise in
    its low bits flips signs and the comparison measures that amplified noise (0.5 relative on
    MobileNetV2 f32, both sides correct). With v = 1 the update is lr·(1−β₁)·g, linear in g,
    and lr = 1 puts the parameter deltas on the moments' scale, which the error floor assumes."""
    rng = np.random.default_rng(zlib.crc32(name.encode()) ^ SEED)
    names = {n: s for n, s in ins}
    out = []
    for n, s in ins:
        if n == 'onehot':
            a = np.zeros(s, np.float32); a[np.arange(s[0]), rng.integers(0, s[1], s[0])] = 1
        elif n == 'lr': a = np.float32(1.0)
        elif n in ('bc1', 'bc2'): a = np.float32(1.0)
        elif re.fullmatch(r'dp\d+', n): a = np.ones(s, np.float32)
        elif n[-1] in 'mv' and names.get(n[:-1]) == s:              # an optimizer moment
            a = np.zeros(s, np.float32) if n[-1] == 'm' else np.ones(s, np.float32)
        elif n == 'x': a = rng.standard_normal(s, np.float32)
        elif len(s) == 0: a = np.float32(0.5)
        elif len(s) == 1:
            a = rng.standard_normal(s, np.float32) * 0.1
            if re.search(r'g\d*$', n) or n == 'gF': a += 1
        else:
            fan = max(s[0], prod(s[1:]))
            a = rng.standard_normal(s, np.float32) * np.float32(np.sqrt(2.0 / fan))
        out.append(np.ascontiguousarray(a, np.float32))
    return out


def pairing(name, ins, outs):
    """outs[i] ↔ the next unused input of the same shape, past x (train steps only)."""
    if 'train_step' not in name: return [None] * len(outs)
    pair, j = [], 1
    for s in outs:
        k = j
        while k < len(ins) and ins[k][1] != s: k += 1
        if k < len(ins): pair.append(k); j = k + 1
        else: pair.append(None)
    return pair


def summarize(name, ins, outs, inputs, results):
    pair = pairing(name, ins, outs)
    rows = []
    for i, (y, p) in enumerate(zip(results, pair)):
        y = np.asarray(y, np.float64).reshape(-1)
        if p is not None: y = y - np.asarray(inputs[p], np.float64).reshape(-1)
        r = np.random.default_rng(zlib.crc32(f'{name}:{i}'.encode()))
        idx = np.arange(y.size) if y.size <= NSAMPLE else np.sort(r.choice(y.size, NSAMPLE, replace=False))
        fin = np.isfinite(y)
        rows.append({'n': y.size, 'l2': float(np.sqrt(np.sum(y[fin] ** 2))),
                     'maxabs': float(np.max(np.abs(y[fin]))) if fin.any() else 0.0,
                     'nonfinite': int((~fin).sum()), 'idx': idx, 'vals': y[idx]})
    return rows


def save_golden(name, rows):
    GOLD.mkdir(exist_ok=True)
    np.savez_compressed(GOLD / f'{name}.npz',
                        n=np.array([r['n'] for r in rows]), l2=np.array([r['l2'] for r in rows]),
                        maxabs=np.array([r['maxabs'] for r in rows]),
                        nonfinite=np.array([r['nonfinite'] for r in rows]),
                        vals=np.concatenate([r['vals'] for r in rows]).astype(np.float32),
                        counts=np.array([len(r['vals']) for r in rows]))


def load_golden(name):
    z = np.load(GOLD / f'{name}.npz')
    vals = np.split(z['vals'].astype(np.float64), np.cumsum(z['counts'])[:-1])
    return [{'n': int(n), 'l2': float(l), 'maxabs': float(m), 'nonfinite': int(f), 'vals': v}
            for n, l, m, f, v in zip(z['n'], z['l2'], z['maxabs'], z['nonfinite'], vals)]


FLOOR = 1e-2


def compare(got, ref):
    """Per-output error, then its median and 90th percentile over the artifact's outputs.

    An output's error is the larger of (rms of the sampled differences) and (the L2 norms'
    difference), both over the reference output's rms — floored at FLOOR × the artifact's median
    output rms. Non-finite counts and element counts must match exactly.

    ⚠ WHY A PERCENTILE AND A FLOOR, NOT THE MAX. Two kinds of output are noise on both sides,
    however correct the backend. (1) Analytically-zero gradients: a BN β feeding straight into
    the next batch BN has its shift cancelled, so XLA:CPU says 1e-10 and TF32 says 3e-8 — 300×
    "apart". (2) Early-layer gradients of a deep batch-BN net at random init, which amplify
    rounding through every layer backward (ResNet-34's stem γ/β differ by ~30 % under TF32, both
    sides correct). Either one decides a max. A codegen fault in an op family moves most outputs,
    which is what the median and p90 see; the max is reported, not gated.
    """
    if len(got) != len(ref): return None, f'{len(got)} outputs, golden has {len(ref)}'
    rms = [r['l2'] / np.sqrt(r['n']) for r in ref]
    floor = FLOOR * float(np.median([x for x in rms if x > 0] or [1.0]))
    errs = []
    for i, (g, r) in enumerate(zip(got, ref)):
        if g['n'] != r['n']: return None, f'output {i}: {g["n"]} elements, golden {r["n"]}'
        if g['nonfinite'] != r['nonfinite']:
            return None, f'output {i}: {g["nonfinite"]} non-finite, golden {r["nonfinite"]}'
        scale = max(rms[i], floor)
        d = g['vals'] - r['vals']; d = d[np.isfinite(d)]
        es = float(np.sqrt(np.mean(d ** 2))) / scale if d.size else 0.0
        en = abs(g['l2'] - r['l2']) / (scale * np.sqrt(r['n']))
        errs.append(max(es, en))
    e = np.array(errs)
    return {'median': float(np.median(e)), 'p90': float(np.percentile(e, 90)),
            'max': float(e.max()), 'worst': int(e.argmax())}, ''


def tolerance(backend, dtype, name, family):
    """Last matching row of tolerances.tsv wins; `*` matches anything."""
    tol = None
    for ln in (HERE / 'tolerances.tsv').read_text().splitlines():
        if not ln.strip() or ln.startswith('#'): continue
        b, d, f, s, n = ln.split('\t')[:5]
        if b in ('*', backend) and d in ('*', dtype) and f in ('*', family, name):
            tol = (float(s), float(n))
    if tol is None: sys.exit(f'no tolerance row for {backend}/{dtype}/{family}')
    return tol


def run_cpu(fn, path, inputs):
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import jax
    from jax._src import xla_bridge
    from jax._src.lib import xla_client as xc
    from jax._src.interpreters import mlir as jmlir
    import jaxlib.mlir.ir as ir
    txt = path.read_text().replace(f'func.func @{fn}(', 'func.func @main(', 1)
    backend = xla_bridge.get_backend('cpu')
    devices = xc.DeviceList(tuple(backend.local_devices()[:1]))
    with jmlir.make_ir_context() as ctx, ir.Location.unknown(ctx):
        exe = backend.compile_and_load(ir.Module.parse(txt), executable_devices=devices,
                                       compile_options=xc.CompileOptions())
    outs = exe.execute([jax.device_put(a, devices[0]) for a in inputs])
    return [np.asarray(o) for o in outs]


def cmd_goldens(a):
    bad = []
    for art in artifacts():
        name = art['name']
        if a.only and name != a.only: continue
        fn, ins, outs = signature(name)
        inputs = make_inputs(name, ins)
        rows = summarize(name, ins, outs, inputs, run_cpu(fn, ROOT / 'verified_mlir' / f'{name}.mlir', inputs))
        if a.check:
            st, msg = compare(rows, load_golden(name))
            ok = not msg and st['max'] <= 1e-3      # same graph, same backend: only threading
            print(f'{"✓" if ok else "✗"} {name}: ' + (msg or f'max {st["max"]:.1e}'))
            if not ok: bad.append(name)
        else:
            save_golden(name, rows)
            print(f'wrote {name}: {len(rows)} outputs, {sum(r["nonfinite"] for r in rows)} non-finite')
    if bad: sys.exit(f'stale goldens: {", ".join(bad)}')


def cmd_run(a):
    art = next((r for r in artifacts() if r['name'] == a.name), None)
    if not art: sys.exit(f'{a.name} is not in tier2_artifacts.tsv')
    fn, ins, outs = signature(a.name)
    inputs = make_inputs(a.name, ins)
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    spec, inb, outb = work / f'{a.name}.spec', work / f'{a.name}.in', work / f'{a.name}.out'
    spec.write_text(f'{len(ins)} {len(outs)}\n' + ''.join(f'{len(s)} {" ".join(map(str, s))}\n' for _, s in ins)
                    + ''.join(f'{prod(s)}\n' for s in outs))
    with open(inb, 'wb') as f:
        for x in inputs: f.write(x.tobytes())
    try:
        r = subprocess.run([a.runner, str(ROOT / 'verified_mlir' / f'{a.name}.mlir'), f'm.{fn}',
                            str(spec), str(inb), str(outb)], cwd=ROOT, capture_output=True, text=True)
        sys.stderr.write(r.stderr); sys.stdout.write(r.stdout)
        if r.returncode: sys.exit(f'FAIL runner exit {r.returncode}')
        flat = np.fromfile(outb, np.float32)
    finally:
        for p in (inb, outb): p.unlink(missing_ok=True)
    results = np.split(flat, np.cumsum([prod(s) for s in outs])[:-1])
    st, msg = compare(summarize(a.name, ins, outs, inputs, results), load_golden(a.name))
    tm, tp = tolerance(a.backend, art['dtype'], a.name, art['family'])
    line = msg or (f'median {st["median"]:.1e} (≤{tm:.2g}), p90 {st["p90"]:.1e} (≤{tp:.2g}), '
                   f'max {st["max"]:.1e} at output {st["worst"]}')
    if msg or st['median'] > tm or st['p90'] > tp:
        print(f'FAIL {art["family"]}/{art["dtype"]}: {line}'); sys.exit(1)
    print(f'OK {art["family"]}/{art["dtype"]}: {line}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('goldens'); g.add_argument('--check', action='store_true'); g.add_argument('--only')
    r = sub.add_parser('run'); r.add_argument('name'); r.add_argument('--runner', required=True)
    r.add_argument('--work', required=True); r.add_argument('--backend', default='cuda')
    a = ap.parse_args()
    {'goldens': cmd_goldens, 'run': cmd_run}[a.cmd](a)


if __name__ == '__main__':
    main()
