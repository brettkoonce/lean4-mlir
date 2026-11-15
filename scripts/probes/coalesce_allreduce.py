"""coalesce_allreduce.py — TIMING EXPERIMENT ONLY. Rewrites a rendered data-parallel train step so
its per-tensor gradient all_reduces become NGROUPS concatenated exchanges, optionally exchanging
only part of the gradient. Driven by scripts/probes/allreduce_bytes_probe.sh; its output is never a
committed render.

    coalesce_allreduce.py in.mlir out.mlir [ngroups]
    SKIP_FRAC=f   the first f of the gradient ELEMENTS skip the exchange (`g + g` stands in for
                  the all_reduce). The step is then numerically WRONG; only its time means anything.

With SKIP_FRAC unset the rewrite is exact: 2026-11-15 on 4x RTX 3060, ConvNeXt-S, 1 and 8 groups
gave losses bit-identical to the 344-all_reduce render, at the same step time.
"""
import re, sys
src = open(sys.argv[1]).read().split('\n'); G = int(sys.argv[3]) if len(sys.argv) > 3 else 1
pat = re.compile(r'^(\s*)(%arsum\w+) = "stablehlo.all_reduce"\((%\w+)\) \(\{$')
end = re.compile(r'^\s*\}\) \{ replica_groups = (dense<[^>]*> : tensor<[^>]*>) \} : \(tensor<([0-9x]*)f32>\) -> tensor<[0-9x]*f32>$')
items = []; out = []; i = 0; first = None; rg = None
while i < len(src):
    m = pat.match(src[i])
    if m:
        j = i
        while not end.match(src[j]): j += 1
        e = end.match(src[j]); rg = e.group(1)
        dims = [int(d) for d in e.group(2).split('x') if d]
        items.append((m.group(2), m.group(3), dims))
        if first is None: first = len(out); ind = m.group(1)
        i = j + 1; continue
    out.append(src[i]); i += 1
def ty(d): return 'tensor<' + ''.join(f'{x}x' for x in d) + 'f32>'
def prod(d):
    p = 1
    for x in d: p *= x
    return p
import os
skip = float(os.environ.get("SKIP_FRAC", "0")); pre = []; cum = 0; keep = []
for it in items:
    if cum < skip * sum(prod(d) for _, _, d in items):
        pre.append(f'{ind}{it[0]} = stablehlo.add {it[1]}, {it[1]} : {ty(it[2])}')   # TIMING ONLY: no exchange
    else: keep.append(it)
    cum += prod(it[2])
items = keep
# contiguous groups balanced by element count
tot = sum(prod(d) for _, _, d in items); groups = [[]]; acc = 0
for it in items:
    if acc >= tot / G * len(groups) and len(groups) < G: groups.append([])
    groups[-1].append(it); acc += prod(it[2])
new = []
for gi, grp in enumerate(groups):
    n = sum(prod(d) for _, _, d in grp); flats = []
    for k, (name, v, d) in enumerate(grp):
        f = f'%cf{gi}_{k}'; flats.append((f, prod(d)))
        new.append(f'{ind}{f} = stablehlo.reshape {v} : ({ty(d)}) -> tensor<{prod(d)}xf32>')
    new.append(f'{ind}%ccat{gi} = stablehlo.concatenate ' + ', '.join(f for f, _ in flats) + ', dim = 0 : (' + ', '.join(f'tensor<{p}xf32>' for _, p in flats) + f') -> tensor<{n}xf32>')
    new += [f'{ind}%cred{gi} = "stablehlo.all_reduce"(%ccat{gi}) ' + '({',
            f'{ind}^bb0(%cra{gi}: tensor<f32>, %crb{gi}: tensor<f32>):',
            f'{ind}  %crs{gi} = stablehlo.add %cra{gi}, %crb{gi} : tensor<f32>',
            f'{ind}  stablehlo.return %crs{gi} : tensor<f32>',
            f'{ind}' + '}) { replica_groups = ' + rg + ' } : ' + f'(tensor<{n}xf32>) -> tensor<{n}xf32>']
    off = 0
    for k, (name, v, d) in enumerate(grp):
        p = prod(d)
        new.append(f'{ind}%csl{gi}_{k} = stablehlo.slice %cred{gi} [{off}:{off+p}] : (tensor<{n}xf32>) -> tensor<{p}xf32>')
        new.append(f'{ind}{name} = stablehlo.reshape %csl{gi}_{k} : (tensor<{p}xf32>) -> {ty(d)}')
        off += p
out[first:first] = pre + new
open(sys.argv[2], 'w').write('\n'.join(out))
print(len(items), 'all_reduces ->', len(groups), 'elements', tot, file=sys.stderr)
