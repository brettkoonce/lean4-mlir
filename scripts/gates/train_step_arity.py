#!/usr/bin/env python3
"""Every committed train step satisfies the PJRT shim's output contract: `#out = #in − 2`.

The driver copies the whole output blob back into the next step's parameter blob (`pbuf := out`),
so a train step returns every input except `%x` and `%onehot` — parameters and moments, the report
scalars, the accumulation / EMA passthroughs, the BN stats and the per-example masks
(`%dp*` drop-path, the dropout mask). `ffi/pjrt_ffi.c`'s G4 check refuses a render that returns
fewer, but only at run time on a GPU; a compile probe never reaches it.

⛔ That is how the 12 R50 drop-path renders (RSB-A2/A1) went from 08-27 to 10-05 returning 16 values
short: their arity check encoded "stochastic depth adds INPUTS and nothing else", and nothing ran
them. This gate is text-only, so it runs first in proofs.yml.

    python3 scripts/gates/train_step_arity.py              # every committed train step
    python3 scripts/gates/train_step_arity.py <file.mlir>…  # just these (the job confs' PRECHECK)
"""
import glob
import re
import sys

SIG = re.compile(r'func\.func @(\w*train_step\w*)\((.*?)\)\s*->\s*\((.*?)\)\s*\{', re.S)
RET = re.compile(r'^\s*(?:func\.)?return\s+(.*?)\s*:', re.M)
MASK = re.compile(r'%(?:dp\d+|do\w*)\b')


def check(path):
    src = open(path).read()
    m = SIG.search(src)
    if not m:
        return None  # no packed train-step entry (e.g. a forward-only module)
    ins = [a.split(':')[0].strip() for a in m.group(2).split(', ')]
    n_out = len(m.group(3).split(', '))
    body = src[m.end():]
    nxt = body.find('\n  func.func')
    rets = RET.findall(body if nxt < 0 else body[:nxt])
    returned = set(re.findall(r'%[\w#]+', rets[-1])) if rets else set()
    errs = []
    if n_out != len(ins) - 2:
        errs.append(f"{len(ins)} inputs, {n_out} outputs (want {len(ins) - 2})")
    if rets and len(re.findall(r'%', rets[-1])) != n_out:
        errs.append(f"return lists {len(re.findall(r'%', rets[-1]))} values, signature {n_out}")
    lost = [a for a in ins if MASK.fullmatch(a) and a not in returned]
    if lost:
        errs.append(f"masks not passed through: {', '.join(lost[:4])}{' …' if len(lost) > 4 else ''}")
    return errs


bad, n = [], 0
for path in sys.argv[1:] or sorted(glob.glob('verified_mlir/*train_step*.mlir')):
    errs = check(path)
    if errs is None:
        continue
    n += 1
    if errs:
        bad.append((path, errs))

for path, errs in bad:
    print(f"✗ {path}: {'; '.join(errs)}")
print(f"{'✓' if not bad else '✗'} {n - len(bad)}/{n} train steps return #in − 2 values with every mask "
      f"passed through")
sys.exit(1 if bad else 0)
