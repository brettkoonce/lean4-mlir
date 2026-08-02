"""Move every drop site from the RESIDUAL BRANCH onto the BLOCK OUTPUT.

    correct     %B = multiply %mask, %branch ; %C = add %B, %skip     ->  s*branch + x
    misplaced   %B = add %branch, %skip      ; %C = multiply %mask, %B ->  s*(branch + x)

Same SSA names, same order, same types — so nothing downstream moves and the arity, the op
counts and the prefix are all unchanged. That is the point: this is exactly the misplacement no
structural check in the repo can see.
"""
import re, sys
src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().split("\n")
out, n = [], 0
i = 0
while i < len(lines):
    m = re.match(r'^(\s*)(%\w+) = stablehlo\.broadcast_in_dim (%dp\d+), dims = \[0\] : \((.*?)\) -> (.*)$', lines[i])
    if m and i + 2 < len(lines):
        ind, bcast, dp, _srcty, vty = m.groups()
        mm = re.match(r'^\s*(%\w+) = stablehlo\.multiply ' + re.escape(bcast) + r', (%\w+) : (.*)$', lines[i+1])
        ma = re.match(r'^\s*(%\w+) = stablehlo\.add (%\w+), (%\w+) : (.*)$', lines[i+2])
        if mm and ma and ma.group(2) == mm.group(1):
            mulOut, branch, mty = mm.groups()
            addOut, _, skip, aty = ma.groups()
            out.append(lines[i])
            out.append(f"{ind}{mulOut} = stablehlo.add {branch}, {skip} : {aty}")
            out.append(f"{ind}{addOut} = stablehlo.multiply {bcast}, {mulOut} : {mty}")
            i += 3; n += 1
            continue
    out.append(lines[i]); i += 1
open(dst, "w").write("\n".join(out))
print(f"misplaced {n} drop sites: {src} -> {dst}")
