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

# Every (1-replica, 4-replica) pair rendered at the SAME optimizer and the SAME k. The accumulation
# pair is here for the same reason as the plain one: `r50-accum-tie` can only run single-device (the
# DP render's %loss is replica-local, see below), so its verdict reaches `accdp4x64` by this text
# argument or not at all. ⚠ `accdp8x64` — the effective-batch-2048 artifact — has no matched-k
# single-device peer and is therefore NOT covered here; it is the same renderer at a different
# baked constant, which is an inheritance argument rather than a check, and it says so.
PAIRS = [
    ("verified_mlir/resnet50in_adam64_train_step.mlir",
     "verified_mlir/resnet50in_adamdp64_train_step.mlir"),
    ("verified_mlir/resnet50in_acc4x64_train_step.mlir",
     "verified_mlir/resnet50in_accdp4x64_train_step.mlir"),
    # ▶▶ `a3_paper_fidelity.md` §3.2, closed 2026-08-14. These three were the composed RSB-A3
    # renders, and the FIRST of them is the artifact the 77.43% run actually trained on — so until
    # now the gradcheck's verdict reached the graph that produced this repo's headline number by
    # nothing at all. ⚠ They are `resnet50in160`, a different NET SPEC (q = 5) and not a suffix,
    # which is why the name normaliser below had to learn the resolution.
    ("verified_mlir/resnet50in160_lambacc8x64bce_train_step.mlir",
     "verified_mlir/resnet50in160_lambaccdp8x64bce_train_step.mlir"),
    ("verified_mlir/resnet50in160_lambacc8x64wxbce_train_step.mlir",
     "verified_mlir/resnet50in160_lambaccdp8x64wxbce_train_step.mlir"),
    # ⭐⭐ D1's pair, and it is the one that made this check say something NEW rather than more of
    # the same. Every pair above all-reduces inside `optOne`, per parameter, interleaved with that
    # parameter's optimizer ops. The clip render HOISTS all 161 collectives to the top of the
    # optimizer stage — they must precede the norm fold — so the DP and single-device files differ
    # in WHERE the carve-out sits, not merely in whether it is present. That the substitution still
    # lands line for line is a real result: the clip changed the collective's PLACEMENT without
    # changing the program it is a carve-out from.
    ("verified_mlir/resnet50in160_lambacc8x64wxclipbce_train_step.mlir",
     "verified_mlir/resnet50in160_lambaccdp8x64wxclipbce_train_step.mlir"),
]
# ⚠ NOT COVERED, and each for a stated reason rather than by omission:
#   `lambaccdp8x128bce` — B = 128 per replica, so its single-device peer would be a different
#       baked batch; there is no matched-B pair to make.
#   `lambacc4x64bce`    — k = 4, no DP peer rendered.
#   `accdp8x64`         — the pre-existing note above: same renderer, different baked constant,
#       which is an inheritance argument rather than a check.
# Overridable so the NEGATIVE CONTROL can be run: point B at a mutated copy and this must FAIL.
if len(sys.argv) == 3:
    PAIRS = [(sys.argv[1], sys.argv[2])]

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


def check(A: str, B: str) -> int:
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
    # ⚠ The RESOLUTION is captured and KEPT (`\1`), not normalised away with the optimizer. The
    # 224 net (`resnet50in`) and the 160 net (`resnet50in160`) are different NET SPECS — q = 7 vs
    # q = 5 — so a pair that crossed them must fail on this line rather than be smoothed into
    # agreement. It would fail on the tensor shapes too; this makes it fail for the RIGHT reason,
    # on line 1, instead of reporting ten thousand differing lines.
    fname = re.compile(r'resnet50in(\d*)_\w+_train_step')
    an = [fname.sub(r'resnet50in\1_OPT', l) for l in a if not COMMENT.match(l)]
    for p, g in grads.items():
        an = [sub_ssa(l, g, f'@G{p}') for l in an]
    bn = [fname.sub(r'resnet50in\1_OPT', l) for l in b if not COMMENT.match(l) and not AR_LINE.match(l)]
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


def main() -> int:
    rc = 0
    for a, b in PAIRS:
        rc = max(rc, check(a, b))
    return rc


if __name__ == "__main__":
    sys.exit(main())
