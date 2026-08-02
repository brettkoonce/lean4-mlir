#!/usr/bin/env python3
"""Perturb a `wx` render's weight-decay mask, to prove `wdx-tie` can go red.

The mask is carried entirely by which scalar each param's `broadcast_in_dim` reads — `%wd`
(decayed) or `%wdz` (excluded). Nothing else in the artifact distinguishes them: same arity, same
types, same op count, same line count. That is exactly why a numeric known answer is the only gate
that can see a wrong mask, and it is what these controls demonstrate.

    invert : swap every %wd <-> %wdz          -> decays the 126, excludes the 74
    swap1  : flip ONE decayed param to excluded and ONE excluded to decayed (counts UNCHANGED)

`swap1` is the sharp one: the counts still read 74/126, so a gate that checked only how many params
moved would pass it. Only the per-parameter partition catches it.
"""
import re, sys

src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(src).read().split("\n")
PAT = re.compile(r'^(\s*%\w+ = stablehlo\.broadcast_in_dim )(%wdz|%wd)(, dims = \[\] :.*)$')

# comment lines mention both names; never rewrite them
idx = [i for i, l in enumerate(lines) if PAT.match(l) and not l.lstrip().startswith("//")]
dec = [i for i in idx if PAT.match(lines[i]).group(2) == "%wd"]
exc = [i for i in idx if PAT.match(lines[i]).group(2) == "%wdz"]
print(f"  found {len(dec)} decayed / {len(exc)} excluded sites")

def setop(i, name):
    m = PAT.match(lines[i]); lines[i] = m.group(1) + name + m.group(3)

if mode == "invert":
    for i in dec: setop(i, "%wdz")
    for i in exc: setop(i, "%wd")
    n = len(idx)
elif mode == "swap1":
    setop(dec[0], "%wdz"); setop(exc[0], "%wd")
    n = 2
else:
    raise SystemExit(f"mode must be invert|swap1, got {mode}")

# an inverted render still needs both constants declared; `swap1` too. Both already are.
open(dst, "w").write("\n".join(lines))
print(f"  {mode}: rewrote {n} operand(s) -> {dst}")
