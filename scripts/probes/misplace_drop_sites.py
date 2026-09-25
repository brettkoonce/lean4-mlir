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
        # ⚠ THE BRANCH MAY BE EITHER OPERAND OF THE ADD, and this was NOT a generalisation for its
        # own sake — it is a defect this script had. EfficientNet and ConvNeXt both emit
        # `add %branch, %skip` (branch first), so the original matched only that. ViT emits
        # `add %skip, %branch` — `hres = addVB(xin, o)`, the skip FIRST — so the original script
        # silently rewrote ZERO of ViT's 24 sites and printed "misplaced 0 drop sites". A control
        # that quietly does nothing reads exactly like a control that ran: the gate it licenses
        # would have gone green against an unmodified render. Found by running it and reading the
        # count, which is the only reason a `0` is now a hard failure below.
        if mm and ma:
            mulOut, branch, mty = mm.groups()
            addOut, aL, aR, aty = ma.groups()
            skip = aR if aL == mulOut else (aL if aR == mulOut else None)
            if skip is not None:
                # Preserve the operand ORDER of the original add, so the misplaced render differs
                # from the correct one in exactly the two lines this swaps and in nothing else.
                addArgs = f"{branch}, {skip}" if aL == mulOut else f"{skip}, {branch}"
                out.append(lines[i])
                out.append(f"{ind}{mulOut} = stablehlo.add {addArgs} : {aty}")
                out.append(f"{ind}{addOut} = stablehlo.multiply {bcast}, {mulOut} : {mty}")
                i += 3; n += 1
                continue
    out.append(lines[i]); i += 1
open(dst, "w").write("\n".join(out))
print(f"misplaced {n} drop sites: {src} -> {dst}")
# ⚠ A CONTROL THAT REWRITES NOTHING IS NOT A CONTROL. It produces a byte-identical copy, the gate
# it feeds goes green, and the run reads exactly like "the control did not fire". Refuse instead.
if n == 0:
    sys.exit(f"REFUSING: no drop site matched in {src} — the output would be a copy, and the gate "
             f"it feeds would pass against an UNMODIFIED render. Check the broadcast/multiply/add "
             f"shape actually emitted (the branch may be either operand of the add).")
