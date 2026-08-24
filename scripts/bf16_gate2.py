#!/usr/bin/env python3
"""Gate 2 for the bf16 render path: did the bf16 actually reach the hardware?

    .venv/bin/python scripts/bf16_gate2.py verified_mlir/resnet34in_momdp64bf16_train_step.mlir
    .venv/bin/python scripts/bf16_gate2.py <bf16.mlir> --against <f32.mlir>   # + a speedup

⚠⚠ WHY THIS EXISTS, AND WHY IT DOES NOT GREP. A `stablehlo.convolution` line carries only its
RESULT type. Grepping it for "bf16" reports success for a graph whose operands XLA has quietly
converted back to f32 — which is exactly what happens to a conv with bf16 operands and an
f32-typed result (`xla_allow_excess_precision`, and `=false` does not rescue it). That mistake
was made twice in one session before this script existed. So: compile, then resolve each
convolution's OPERAND SSA names to their dtypes in the OPTIMIZED HLO.

▶ `planning/bf16_renderer.md` §9.2 has the measurement; `BatchableOp.convBf16` has the note.
"""
import argparse, re, sys, time

def _lazy():
    import jax, jax.extend as jex, numpy as np
    from jax._src.lib import xla_client as xc
    from jax._src.interpreters import mlir as jmlir
    from jax._src import compiler as jcomp
    from jaxlib.mlir import ir
    return jax, jex, np, xc, jmlir, jcomp, ir

def compile_mlir(path, ctxs):
    jax, jex, np, xc, jmlir, jcomp, ir = ctxs
    client = jex.backend.get_backend()
    dl = xc.DeviceList(tuple(client.local_devices()[:1]))
    src = open(path).read()
    # XLA insists the entry be `main`; the artifacts name theirs after the variant
    # (`@resnet34in_momdp64bf16_train_step`) because the driver matches on it. Rename for the
    # check only — this rewrites nothing on disk and changes no op in the body.
    src, n = re.subn(r"func\.func\s+(public\s+)?@[\w.]+\(", "func.func public @main(", src, count=1)
    if n == 0:
        raise SystemExit(f"{path}: no `func.func @…(` found — not a rendered artifact?")
    ctx = jmlir.make_ir_context()
    with ctx, ir.Location.unknown(ctx):
        mod = ir.Module.parse(src)
        return jcomp.backend_compile_and_load(client, mod, dl, xc.CompileOptions(), [])

def conv_operand_dtypes(exe):
    """[(op-kind, [operand dtypes])] for every convolution in the optimized HLO."""
    txt = exe.hlo_modules()[0].to_string()
    types = {}
    for line in txt.splitlines():
        m = re.match(r"\s*(%[\w.\-]+) = ([a-z0-9]+)\[", line.strip())
        if m:
            types[m.group(1)] = m.group(2)
    out = []
    for line in txt.splitlines():
        s = line.strip()
        is_conv = ("cudnn" in s and "custom-call" in s) or re.search(r"=\s*\S+\s+convolution\(", s)
        if not is_conv:
            continue
        g = re.search(r"(?:custom-call|convolution)\(([^)]*)\)", s)
        if g:
            names = [o.strip() for o in g.group(1).split(",")][:2]
            out.append((("cudnn" if "cudnn" in s else "hlo"), [types.get(n, "?") for n in names]))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir")
    ap.add_argument("--against", help="the f32 peer, to also report a speedup")
    ap.add_argument("--expect", default="bf16", choices=["bf16", "f32"],
                    help="dtype every convolution operand must have (default bf16)")
    a = ap.parse_args()
    ctxs = _lazy()

    exe = compile_mlir(a.mlir, ctxs)
    convs = conv_operand_dtypes(exe)
    if not convs:
        print(f"⛔ no convolution found in {a.mlir} — is this the right artifact?")
        return 1
    bad = [(k, d) for k, d in convs if not all(x == a.expect for x in d)]
    print(f"  {a.mlir}")
    print(f"  {len(convs)} convolutions; {len(convs) - len(bad)} with all operands {a.expect}")
    if bad:
        print(f"\n  ⛔ GATE 2 FAILED — {len(bad)} convolution(s) not {a.expect}:")
        for k, d in bad[:8]:
            print(f"       {k}: operands={d}")
        print("\n  ▶ If these are f32 in a bf16 artifact, the emit gave the convolution an")
        print("    f32-TYPED result and XLA deleted the casts. See bf16_renderer.md §9.2.")
        return 1
    what = ("bf16 reached the hardware in every convolution" if a.expect == "bf16"
            else "every convolution is f32, as expected of the control arm")
    print(f"  ✅ GATE 2: {what}")

    if a.against:
        jax, jex, np, *_ = ctxs
        ref = compile_mlir(a.against, ctxs)
        print(f"\n  ⚠ speedup not measured: it needs matching input buffers for both modules,")
        print(f"    which this script does not synthesize. Use LEAN_MLIR_MAX_STEPS on the real")
        print(f"    trainer instead — that is the number worth quoting anyway (§10.4 step 6).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
