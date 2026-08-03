"""Break ONE conv input-VJP kernel flip, inside the transpose/reverse allowance.

    scripts/perturb_conv_vjp.py <src.mlir> <dst.mlir> <new-variant-name>

▶ WHY THIS EXISTS. `tests/TestConvNeXtFwdBTie.lean` allows exactly one difference between the
per-example and batched ConvNeXt chains: the conv input-VJP prepares its kernel with a
`transpose` (dims [1,0,2,3]) and a `reverse` (dims [2,3]), and the two emitters order them
differently. They act on DISJOINT axes, so they commute and both compute the same kernel — an
argument about StableHLO semantics, which §5 says is audited-but-trusted.

A keep = 1 numeric tie between `convnext_adam` (per-example chain) and `convnext_adamdrop`
(batched chain) therefore tests TWO things at once, and its natural control — running the drop at
its real ramp — only proves the harness is wired to the DROP. This script supplies the other half:
it perturbs one line inside the allowance and the tie must go red.

▶ WHICH LINE, AND WHY IT HAS TO BE CHOSEN RATHER THAN PICKED. Most of ConvNeXt's conv-VJP reverses
act on 1x1 kernels, where `reverse dims = [2, 3]` is the IDENTITY — perturbing one of those is
inert and the control would fail to fire for a reason that has nothing to do with the render. The
2x2/s2 downsample kernels are the ones with a real spatial extent, and they are demonstrably inside
the allowance: the two chains reverse them at DIFFERENT operand types (`192x96x2x2` batched against
`96x192x2x2` per-example), because one reverses before the transpose and the other after.

So: drop axis 3 from the first 2x2 reverse. The type is unchanged (reverse is shape-preserving), so
it compiles; the kernel flip is now half-done, so it is a different function.

▶ THE OUTPUT IS NOT COMMITTED. It renders under a variant name no config sets, which is the rule
`planning/grad_clip.md` §11 arrived at the hard way: an artifact baking a value nothing asks for IS
a silent-hyperparameter artifact. Generate it, run it, delete it.
"""
import re, sys

src, dst, variant = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(src).read().split("\n")

# The entry name must track the variant or the shim refuses the call ("entry mismatch") — which is
# the right failure, but it would look like the control firing when it had not run at all.
old_entry = None
for l in lines:
    m = re.search(r"func\.func @(\w+)_train_step", l)
    if m:
        old_entry = m.group(1)
        break
if old_entry is None:
    sys.exit("no `func.func @<slug>_<variant>_train_step` — is this a train-step artifact?")
slug = old_entry.split("_")[0]
new_entry = f"{slug}_{variant}"

out, hit = [], 0
for l in lines:
    l = l.replace(f"@{old_entry}_train_step", f"@{new_entry}_train_step")
    if hit == 0:
        m = re.match(r"^(\s*%\w+ = stablehlo\.reverse %\w+, dims = )\[2, 3\]( : tensor<\d+x\d+x2x2xf32>)$", l)
        if m:
            l = m.group(1) + "[2]" + m.group(2)
            hit += 1
    out.append(l)

if hit == 0:
    sys.exit("no 2x2 conv-VJP reverse found — nothing was perturbed, so this is not a control")
open(dst, "w").write("\n".join(out))
print(f"perturbed {hit} conv-VJP kernel flip (2x2 reverse [2,3] -> [2]); "
      f"entry @{old_entry}_train_step -> @{new_entry}_train_step")
print(f"  {src} -> {dst}")
