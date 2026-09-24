#!/usr/bin/env python3
"""The batch and its LOSS DIVISOR are two spellings of one fact — assert they agree.

⛔ THE DEFECT THIS EXISTS FOR (found 2026-09-17 during ConvNeXt's batch-64 rescope). A train-step
renderer takes the per-replica batch as a Nat, deciding the SHAPES, and takes the loss-gradient
batch divisor as a separate STRING, deciding what the smoothed-loss cotangent is divided by. Move
one without the other and you get exactly what that rescope produced before it was noticed:

    %v1745 = stablehlo.constant dense<32.0> : tensor<64x1000xf32>
    %v1746 = stablehlo.divide %v1744, %v1745 : tensor<64x1000xf32>

A 64-row tensor divided by 32.0 — **every gradient exactly 2x too large, and NOTHING FAILS.** The
graph is well-typed (the divisor is a scalar broadcast, so there is no shape error), the lowerer is
happy, and the run trains, descends and reports a plausible loss. It reads as a 2x learning rate.
On the ImageNet tier there is no accuracy gate to notice, and the pair it was built for is void.

⭐ THE RULE, and it is checkable without resolving the SSA graph: in a train step over B examples
and K classes, an INTEGER-valued constant typed `tensor<BxKxf32>` may only be

    B      the batch — the `%bsc` divisor and the smoothed-loss `divConst`, the softmax-CE case
    B*K    the BCE normaliser — binary cross-entropy averages over ALL B*K logits, not over B
           (every `*bce*` R50 variant: 64000.0 at batch 64, 128000.0 at batch 128)
    1      ones / identity

Label smoothing contributes only NON-integer constants (alpha, -alpha/K), which are ignored. So an
integer constant on [BxK] outside that set is a batch spelled twice and disagreeing.
⚠ B*K is in the set because it is a DIFFERENT correct normaliser, not to make the gate pass: it
still discriminates the real defect, since a wrong batch gives B'/B (e.g. 32 against 64), never
B*K. The `--control` proves the gate fails on the actual defect.

⚠ Deliberately NOT a source-level check. The first version of this gate read the `#eval` sites and
produced 19 FALSE POSITIVES: it took the first `"<n>.0"` string in each block, which for R34's
`wd00` variants is `(wdStr := "0.0")` — the weight decay — and it assumed a default batch of 32
when it saw no keyword it recognised, which is wrong for every renderer that passes the batch
POSITIONALLY (EfficientNet, R34). The artifact is ground truth and has neither problem.

Usage:  python3 scripts/batch_divisor_gate.py            # assert
        python3 scripts/batch_divisor_gate.py --list     # every artifact
        python3 scripts/batch_divisor_gate.py --control  # prove the gate can FAIL
"""
import re, sys, glob, os, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONST = re.compile(r'stablehlo\.constant dense<(-?[0-9.eE+-]+)> : tensor<(\d+)x(\d+)xf32>')
XARG  = re.compile(r'%x: tensor<(\d+)x')

def is_int(v):
    try: f = float(v)
    except ValueError: return False
    return f == int(f) and abs(f) >= 1

def check(path):
    """-> (batch, nClasses, [offending values]) or None if not applicable."""
    src = open(path).read()
    m = XARG.search(src)
    if not m: return None
    B = int(m.group(1))
    # nClasses: the K of the [BxK] logits/cotangent tensors actually present
    ks = {int(k) for v, b, k in CONST.findall(src) if int(b) == B}
    ks |= {int(k) for k in re.findall(r'tensor<' + str(B) + r'x(10|1000)xf32>', src)}
    ks = {k for k in ks if k in (10, 1000)}
    if not ks: return None
    bad = []
    for v, b, k in CONST.findall(src):
        allowed = {1.0, float(B)} | {float(B * k) for k in ks}
        if int(b) == B and int(k) in ks and is_int(v) and float(v) not in allowed:
            bad.append((v, f"{b}x{k}"))
    return (B, sorted(ks), bad)

def main():
    show = "--list" in sys.argv
    if "--control" in sys.argv:
        # ⚠ A gate nobody has seen FAIL is not a gate. Reproduce the real defect in a temp copy:
        # rewrite the batch divisor of the in-flight artifact to half the batch and require a catch.
        src = f"{ROOT}/verified_mlir/convnextin_adamdpwxclipdropbf16_train_step.mlir"
        s = open(src).read()
        B = int(XARG.search(s).group(1))
        broken = s.replace(f"dense<{B}.0> : tensor<{B}x1000xf32>",
                           f"dense<{B//2}.0> : tensor<{B}x1000xf32>", 1)
        assert broken != s, "control could not inject the defect"
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as t:
            t.write(broken); tmp = t.name
        r = check(tmp); os.unlink(tmp)
        if r and r[2]:
            print(f"✅ CONTROL FIRES — injected dense<{B//2}.0> on a {B}-row tensor, gate caught: {r[2]}")
            return 0
        print("⛔ CONTROL DID NOT FIRE — this gate cannot see the defect it exists for"); return 1

    rows, bad, skipped = [], [], []
    paths = sorted(glob.glob(f"{ROOT}/verified_mlir/*_train_step.mlir"))
    if not paths:
        print(f"⛔ no verified_mlir/*_train_step.mlir under {ROOT} — nothing to check"); return 1
    for path in paths:
        r = check(path)
        if r is None:
            skipped.append(os.path.basename(path)[:-5]); continue
        B, ks, off = r
        rows.append((os.path.basename(path)[:-5], B, ks, off))
        if off: bad.append((os.path.basename(path)[:-5], B, off))
    if show:
        print(f"{'ARTIFACT':<56} {'BATCH':>6} {'K':>10}  VERDICT")
        for n, B, ks, off in rows:
            print(f"{n:<56} {B:>6} {str(ks):>10}  {'ok' if not off else '⛔ ' + str(off)}")
        print()
    print(f"── batch/divisor gate: {len(rows)} train-step artifacts, {len(rows)-len(bad)} ok, {len(bad)} MISMATCH"
          + (f", {len(skipped)} SKIPPED (no %x or no [B x 10|1000] tensor)" if skipped else ""))
    for n in skipped:
        print(f"   skipped: {n}")
    if not rows:
        print("⛔ every artifact was skipped — the gate checked nothing"); return 1
    if bad:
        print("⛔ an integer constant on [batch x nClasses] is not the batch, B*K or 1 — the batch is")
        print("   spelled twice and the spellings disagree. Every gradient is scaled by their ratio:")
        for n, B, off in bad:
            print(f"   {n}: batch {B}, offending {off}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
