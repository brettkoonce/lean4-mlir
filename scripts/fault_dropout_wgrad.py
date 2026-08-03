#!/usr/bin/env python3
"""Build the CONTROL render for classifier dropout's weight-gradient gate.

The defect this exists to make visible, in one line: the classifier weight gradient is

    ∂L/∂W_d = Σ_b (dense input)_b ⊗ dy_b

and with dropout on, the dense's input is the DROPPED activation, not the pooled one. Feeding it
the pooled activation type-checks, trains, descends, and is wrong by the mask on the very parameter
dropout acts through — 1 of 262 parameters, so nothing about the run looks unusual.

⚠⚠ **AND IT IS INVISIBLE TO EVERY ONES-MASK GATE THIS FEATURE HAS.** At `mask ≡ 1` the dropped and
pooled activations are the same buffer, so the keep = 1 tie, the forward prefix audit and the
`dropout_ones_id` endpoint check all pass on the faulted render, bit-exact. That is not a weakness
in those gates; it is what `xla_pjrt_handoff.md` §0.4 finding 1 says about identity gates in
general, and this script is what turns that from an argument into a measurement.

It is the same defect ConvNeXt's LayerScale γ gradient had (handoff §0.10), where it was found by
tracing operands by hand rather than by any gate. This is that finding, mechanised.

    scripts/fault_dropout_wgrad.py verified_mlir/efficientnet_adamdo_train_step.mlir /tmp/fault.mlir

The rewrite: find the dropout site `%D = stablehlo.multiply %do, %G`, then find the weight-gradient
`dot_general` whose FIRST operand is `%D`, and swap that operand to `%G`. Exactly one line changes;
SSA names, ordering, types, op counts and line count are all unchanged, so every structural audit
in the repo still passes on the output. That is deliberate — a control that a cheap check could
reject would not be testing what this one tests.

⚠ **IT REFUSES AT ZERO MATCHES**, and that is not defensive coding. `misplace_drop_sites.py`
matched only one operand order and therefore rewrote 0 of ViT's 24 sites, wrote a byte-identical
copy, and the gate it fed went green against an unmodified render (handoff §0.11). A control that
quietly does nothing reads exactly like a control that ran.
"""

import re
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src).read()

    # ── 1. the dropout site: %D = multiply %do, %G ──────────────────────────────────────────────
    # ⚠ Both operand orders, for `misplace_drop_sites.py`'s reason. The renderer emits `%do` first
    # today; a render that emitted the activation first would be equally correct and this script
    # must not silently stop matching if it ever does.
    sites = re.findall(
        r"^\s*(%\w+) = stablehlo\.multiply (%do), (%\w+) : (tensor<[\dx]+xf32>)\s*$",
        text, re.M) + re.findall(
        r"^\s*(%\w+) = stablehlo\.multiply (%\w+), (%do) : (tensor<[\dx]+xf32>)\s*$",
        text, re.M)
    if not sites:
        print(f"REFUSING: no `stablehlo.multiply` against %do in {src} — this render carries no "
              f"classifier dropout, so there is nothing to fault and a copy would be a false PASS.",
              file=sys.stderr)
        return 1

    # The FORWARD site is the first one in emission order; the second is the backward's cotangent
    # scale. Only the forward's output feeds the dense and its weight gradient.
    dropped, _, other, _ = sites[0]
    pooled = other if other != "%do" else sites[0][2]
    if pooled == "%do":
        print(f"REFUSING: could not identify the pooled operand at the dropout site in {src}.",
              file=sys.stderr)
        return 1

    # ── 2. the weight gradient: dot_general whose FIRST operand is the dropped activation ───────
    # ⚠ Matched by OPERAND, not by line number or by position in the file. The dense forward also
    # reads `%dropped` as its first `dot_general` operand, so the two are told apart by their
    # contracting dims: the forward contracts the FEATURE axis ([1] x [0]), the weight gradient
    # contracts the BATCH axis ([0] x [0]). Getting that backwards would fault the forward instead
    # and produce a render that is wrong in a completely different way.
    wgrad = re.compile(
        r"^(\s*%\w+ = stablehlo\.dot_general )" + re.escape(dropped) +
        r"(, %\w+, contracting_dims = \[0\] x \[0\])", re.M)
    faulted, n = wgrad.subn(r"\1" + pooled + r"\2", text)
    if n == 0:
        print(f"REFUSING: found the dropout site ({dropped} = {pooled} scaled by %do) but no "
              f"weight-gradient dot_general reading it with contracting_dims = [0] x [0]. The "
              f"render's classifier backward is not the shape this control assumes; a copy would "
              f"be a false PASS.", file=sys.stderr)
        return 1
    if n != 1:
        print(f"REFUSING: {n} weight-gradient sites matched, expected exactly 1. EfficientNet-B0 "
              f"has ONE classifier; more than one match means this is not the render it thinks.",
              file=sys.stderr)
        return 1

    # ── 3. the rewrite must be exactly one line, or it is not the control it claims to be ───────
    a, b = text.splitlines(), faulted.splitlines()
    if len(a) != len(b):
        print(f"REFUSING: line count moved ({len(a)} → {len(b)}).", file=sys.stderr)
        return 1
    changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(changed) != 1:
        print(f"REFUSING: {len(changed)} lines differ, expected exactly 1.", file=sys.stderr)
        return 1

    open(dst, "w").write(faulted)
    print(f"  faulted {src} → {dst}")
    print(f"    dropout site      : {dropped} = {pooled} ⊙ %do")
    print(f"    weight gradient   : now reads {pooled} (the POOLED activation) instead of "
          f"{dropped} (the DROPPED one)")
    print(f"    line {changed[0] + 1}, and it is the ONLY line that moved — same SSA names, same "
          f"order, same types, same op count")
    print(f"    ⚠ this render trains and descends; at mask ≡ 1 it is BIT-IDENTICAL to the real one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
