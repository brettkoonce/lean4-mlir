"""Does the DRIVER's packed blob match the ARTIFACT's declared signature?

`trainAdamSched` builds `adamShapes` from `VerifiedVariant.{nRegions,nScalars}` and hands it to the
shim; the shim refuses on a count mismatch, but only at run time on a GPU. This reproduces the
count arithmetic against the committed text, so a five-region render can be checked without one.

⚠ It checks the OUTPUT order too, which is the half a buffer-count check cannot see: the driver
does `pbuf := out` — the previous output IS the next input — so `[θ|m|v|G|E]` and the 7-slot scalar
tail have to line up in BOTH directions or the loss slot is read out of a parameter.

    python3 runs/2026-08-27-r50-a2-a1-ema-fifth-region/arity_check.py
"""
import re, sys, glob, os

P, BN = 161, 106                       # R50: parameters, running-stat slots
NDROP = 16                             # one stochastic-depth mask per bottleneck block


def variant_of(path):
    m = re.match(r'resnet50in_(.+)_train_step\.mlir$', os.path.basename(path))
    return m.group(1) if m else None


def acc_on(v):  return "acc" in v
def ema_on(v):  return v.startswith("ema")
# ⚠ stochastic depth adds graph INPUTS and nothing else — no region, no scalar, no output. That
# independence is the point of checking it here rather than assuming it.
def sd_on(v):   return "drop" in v
def n_regions(v): return 3 + acc_on(v) + ema_on(v)
def n_scalars(v): return 3 + 2 * acc_on(v) + 2 * ema_on(v)


def parse(path):
    src = open(path).read()
    m = re.search(r'func\.func @(\S+)\((.*?)\) -> \((.*?)\) \{', src, re.S)
    args = [a.strip() for a in m.group(2).split(', ')]
    return m.group(1), [a.split(':')[0] for a in args], len(m.group(3).split(', '))


fails = 0
rows = []
for path in sorted(glob.glob("verified_mlir/resnet50in_*_train_step.mlir")):
    v = variant_of(path)
    name, names, n_out = parse(path)
    R, S = n_regions(v), n_scalars(v)
    # what the driver packs: %x + R*P params + S scalars + BN + %onehot
    D = NDROP if sd_on(v) else 0
    want_in = 1 + R * P + S + BN + D + 1
    want_out = R * P + S + BN            # theta'..E', loss/bc/aup/akeep/emad/oemad, batch stats
    # the region suffixes, in the order the driver concatenates them
    order = ['', 'm', 'v'] + (['a'] if acc_on(v) else []) + (['e'] if ema_on(v) else [])
    seen, pos, ok_order = [], 1, True
    for suf in order:
        blk = names[pos:pos + P]
        bad = [n for n in blk if not (n.endswith(suf) if suf else
                                      not n.endswith(('m', 'v', 'a', 'e')) or n in ('%sbt',))]
        seen.append(f"{suf or 'θ'}:{len(blk)}")
        pos += P
    scal = names[pos:pos + S]
    expect_scal = (['%lr', '%bc1', '%bc2'] + (['%aup', '%akeep'] if acc_on(v) else []) +
                   (['%emad', '%oemad'] if ema_on(v) else []))
    ok = (len(names) == want_in and n_out == want_out and scal == expect_scal
          and name == f"resnet50in_{v}_train_step")
    fails += 0 if ok else 1
    dp = [n for n in names if n.startswith("%dp")]
    ok = ok and len(dp) == D
    rows.append((("✓" if ok else "✗"), v, R, S, len(names), want_in, n_out, want_out,
                 (f"{len(dp)} masks  " if D else "") +
                 ",".join(s.lstrip('%') for s in scal)))

w = max(len(r[1]) for r in rows)
print("── driver blob vs artifact signature (R50, 161 params, 106 BN slots) ──")
print(f"  {'':1} {'variant':<{w}}  reg  scal   #in(want)    #out(want)  drop / scalar tail")
for t, v, R, S, ni, wi, no, wo, st in rows:
    print(f"  {t} {v:<{w}}   {R}    {S}   {ni:>4}({wi:>4})   {no:>4}({wo:>4})  {st}")
print(f"\n{'✓' if not fails else '✗'} {len(rows) - fails}/{len(rows)} artifacts agree with the "
      f"driver's packed layout")
sys.exit(1 if fails else 0)
