#!/usr/bin/env python3
"""r50_dp_render_tie.py — the 4-replica R50 render is the 1-replica render plus an all-reduce.

    python3 tests/r50_dp_render_tie.py
    python3 tests/r50_dp_render_tie.py <1-replica.mlir> <4-replica.mlir>   # negative control

WHAT IT GATES, AND WHY IT EXISTS. `lake build r50-gradcheck` certifies the GRADIENT of
`verified_mlir/resnet50in_adam64_train_step.mlir` — the single-device render. The driver defaults
to `LEAN_MLIR_VARIANT=adamdp64`, the 4-replica one, so on its own that gate certifies an artifact
production does not run.

⚠ The obvious fix — point the gradcheck at the DP artifact — DOES NOT WORK, and the reason is worth
recording. The DP render all-reduces the GRADIENT (`ViTRender.emitGradAllReduce`) but NOT the loss:
`%loss` is whatever replica 0 computed on its own shard. So the adjoint identity
`⟨g, δ⟩ = (L₊ − L₋)/2` compares a 256-sample gradient against a 64-sample loss and fails on a
correct render. (Tier 1's homogeneity identities are unaffected — they hold per parameter for any
batch — but tier 2, the part that pins the SCALE, cannot run there.)

▶ So the claim is carried across by TEXT instead, which is stronger than a numeric tie anyway: the
two artifacts are shown to be the SAME PROGRAM except for the gradient handoff. Concretely, each
`%arsum<P> = "stablehlo.all_reduce"(%vN)` in the DP file names the gradient SSA `%vN` it consumes;
substitute `@G<P>` for that SSA in the single-device file and for `%armean<P>` in the DP file, drop
the all-reduce blocks, and **the two must be identical line for line**. Everything that COMPUTES the
gradient — the whole forward and backward — is then literally the same text, and the gradcheck's
verdict transfers.

⚠ It is a text check on purpose: no GPU, no tolerance, no autotuning. Run it in any pre-commit
sweep. Exits non-zero on any residual difference, and refuses a vacuous pass (0 parameters found).
"""
import re
import sys

A = "verified_mlir/resnet50in_adam64_train_step.mlir"
B = "verified_mlir/resnet50in_adamdp64_train_step.mlir"
# Overridable so the NEGATIVE CONTROL can be run: point B at a mutated copy and this must FAIL.
if len(sys.argv) == 3:
    A, B = sys.argv[1], sys.argv[2]

# The all-reduce block emitted per parameter, in `emitGradAllReduce` order. Every one of these
# lines is the TRUSTED CARVE-OUT the DP render documents in its own header; they are removed from
# the DP file rather than matched, because the single-device file has no peer for them.
AR_LINE = re.compile(
    r'^\s*(%arsum\w+ = "stablehlo\.all_reduce"|\^bb0\(%ara|%aradd\w+ = stablehlo\.add'
    r'|stablehlo\.return %aradd|\}\) \{ replica_groups|%arn\w+ = stablehlo\.constant'
    r'|%armean\w+ = stablehlo\.divide)'
)
# Two header-comment lines the DP render adds to describe that carve-out, and one it replaces.
COMMENT = re.compile(r'^\s*//')
# `%vNNNN` as a whole token — MLIR delimits SSA names with , ) : or whitespace.
#
# ⚠ USES ONLY, never the defining LHS. The gradient's own definition is unchanged between the two
# renders (the all-reduce reads it and produces a NEW value), so rewriting the LHS would report 161
# spurious differences — exactly what the first version of this script did.
def sub_ssa(line: str, old: str, new: str) -> str:
    lhs, eq, rhs = line.partition(" = ")
    if not eq:
        lhs, rhs = "", line
    return lhs + eq + re.sub(r'(?<![\w%])' + re.escape(old) + r'(?![\w])', new, rhs)


def main() -> int:
    try:
        a = open(A).read().splitlines()
        b = open(B).read().splitlines()
    except OSError as e:
        print(f"✗ {e}")
        return 2

    # 1. Learn the gradient SSA each all-reduce consumes: %arsum<P> = all_reduce(%vN).
    grads = {}
    for line in b:
        m = re.match(r'\s*%arsum(\w+) = "stablehlo\.all_reduce"\((%\w+)\)', line)
        if m:
            grads[m.group(1)] = m.group(2)
    if not grads:
        print(f"✗ DEGENERATE: no all_reduce blocks found in {B} — this tie would pass vacuously")
        return 2

    # 2. Normalise. In A the gradient SSA becomes @G<P>; in B %armean<P> becomes the same token.
    #    Comments and the function name differ by design and are dropped/normalised.
    fname = re.compile(r'resnet50in_adam(?:dp)?64_train_step')
    an = [fname.sub('resnet50in_OPT', l) for l in a if not COMMENT.match(l)]
    for p, g in grads.items():
        an = [sub_ssa(l, g, f'@G{p}') for l in an]
    bn = [fname.sub('resnet50in_OPT', l) for l in b if not COMMENT.match(l) and not AR_LINE.match(l)]
    for p in grads:
        bn = [sub_ssa(l, f'%armean{p}', f'@G{p}') for l in bn]

    # 3. The verdict.
    print(f"R50 DP render tie — {A}")
    print(f"                 vs {B}")
    print(f"  {len(grads)} parameters all-reduced;  {len(a)} vs {len(b)} lines, "
          f"{len(an)} vs {len(bn)} after removing the carve-out")
    if an == bn:
        print(f"  ✓ IDENTICAL line for line once each gradient is routed through its all_reduce.")
        print(f"    The forward and backward are the same text, so `r50-gradcheck`'s verdict on")
        print(f"    {A.split('/')[-1]} carries to {B.split('/')[-1]}.")
        return 0

    diff = [(i, x, y) for i, (x, y) in enumerate(zip(an, bn)) if x != y]
    print(f"  ✗ TIE FAILED: {len(diff)} differing lines"
          f"{'' if len(an) == len(bn) else f', and the lengths differ ({len(an)} vs {len(bn)})'}")
    for i, x, y in diff[:5]:
        print(f"    line {i}:\n      1-replica: {x.strip()[:140]}\n      4-replica: {y.strip()[:140]}")
    print("  The DP render is NOT the single-device render plus an all-reduce, so the gradcheck's")
    print("  verdict does NOT transfer. Gate the DP artifact directly or fix the divergence.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
